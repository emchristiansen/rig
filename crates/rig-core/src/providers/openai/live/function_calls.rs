//! Collecting the function calls of Responses-delegated work and continuing
//! the backend once each has an output.
//!
//! The backend's forwarded lifecycle snapshots carry an empty `output`, so the
//! calls are read from nested `response.output_item.done` events and grouped
//! by response. When the response ends with calls pending, the caller submits
//! an output for every call and then sends `response.create`.

use std::collections::BTreeSet;

use super::{BackendEvent, ClientEvent, FunctionCall, ResponseOutcome, ServerEvent};

/// The function calls of one ended backend response, all of which need an
/// output before the backend continues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingFunctionCalls {
    /// The delegation the response belongs to, when the server correlated one.
    pub delegation_id: Option<String>,
    /// The response id.
    pub response_id: String,
    /// The calls, in the order they finished. Never empty.
    pub calls: Vec<FunctionCall>,
}

/// Outputs that do not answer the pending calls exactly once each.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "the outputs do not answer the pending calls once each: missing {missing:?}, unknown {unknown:?}, repeated {repeated:?}"
)]
pub struct MismatchedOutputs {
    /// Pending call ids with no output.
    pub missing: Vec<String>,
    /// Output call ids that name no pending call.
    pub unknown: Vec<String>,
    /// Call ids given more than one output.
    pub repeated: Vec<String>,
}

impl PendingFunctionCalls {
    /// The client events that continue the backend: one
    /// [`ClientEvent::FunctionCallOutput`] per pending call, in call order,
    /// then [`ClientEvent::ResponseCreate`]. Sent in that order, they submit
    /// every result before continuing.
    ///
    /// `outputs` pairs each call id with its output text. Refused unless it
    /// answers every pending call exactly once and names no other call.
    pub fn submit<I, C, O>(&self, outputs: I) -> Result<Vec<ClientEvent>, MismatchedOutputs>
    where
        I: IntoIterator<Item = (C, O)>,
        C: Into<String>,
        O: Into<String>,
    {
        let pending: BTreeSet<&str> = self
            .calls
            .iter()
            .map(|call| call.call_id.as_str())
            .collect();
        let mut answered: Vec<(String, String)> = Vec::new();
        let mut unknown = Vec::new();
        let mut repeated = Vec::new();
        for (call_id, output) in outputs {
            let call_id = call_id.into();
            if !pending.contains(call_id.as_str()) {
                unknown.push(call_id);
            } else if answered.iter().any(|(answered, _)| *answered == call_id) {
                if !repeated.contains(&call_id) {
                    repeated.push(call_id);
                }
            } else {
                answered.push((call_id, output.into()));
            }
        }
        let missing: Vec<String> = self
            .calls
            .iter()
            .filter(|call| !answered.iter().any(|(id, _)| *id == call.call_id))
            .map(|call| call.call_id.clone())
            .collect();
        if !missing.is_empty() || !unknown.is_empty() || !repeated.is_empty() {
            return Err(MismatchedOutputs {
                missing,
                unknown,
                repeated,
            });
        }
        let mut events: Vec<ClientEvent> = Vec::with_capacity(self.calls.len() + 1);
        for call in &self.calls {
            if let Some(index) = answered.iter().position(|(id, _)| *id == call.call_id) {
                let (call_id, output) = answered.swap_remove(index);
                events.push(ClientEvent::FunctionCallOutput {
                    call_id,
                    output,
                    event_id: None,
                });
            }
        }
        events.push(ClientEvent::ResponseCreate { event_id: None });
        Ok(events)
    }
}

/// What an observed event means for the backend's function calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallsUpdate {
    /// The response completed with function calls: submit their outputs,
    /// then continue with `response.create`.
    Submit(PendingFunctionCalls),
    /// The response ended with no function calls to answer.
    Ended {
        /// The delegation the response belongs to, when correlated.
        delegation_id: Option<String>,
        /// The response id.
        response_id: String,
        /// How it ended.
        outcome: ResponseOutcome,
    },
    /// The response failed or was incomplete after requesting function
    /// calls. Their outputs cannot continue it.
    Abandoned {
        /// The calls the response requested.
        pending: PendingFunctionCalls,
        /// How it ended.
        outcome: ResponseOutcome,
    },
}

/// A backend response whose calls are being collected.
#[derive(Clone, Debug)]
struct OpenResponse {
    delegation_id: Option<String>,
    /// `None` until `response.created` or the response's end names it.
    response_id: Option<String>,
    calls: Vec<FunctionCall>,
}

/// Collects the function calls of each backend response from the server
/// events and reports when a response's calls need outputs.
///
/// Calls are attributed to the latest response opened for their event's
/// `delegation_id`. A call that arrives before its `response.created` is
/// held for the next response of that delegation, and a repeated `call_id`
/// within a response is kept once.
#[derive(Clone, Debug, Default)]
pub struct FunctionCallCollector {
    open: Vec<OpenResponse>,
}

impl FunctionCallCollector {
    /// A collector with no open responses.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one server event, in delivery order. Returns an update when the
    /// event ends a backend response; every other event returns `None`.
    pub fn observe(&mut self, event: &ServerEvent) -> Option<CallsUpdate> {
        let ServerEvent::ResponseEvent(event) = event else {
            return None;
        };
        let delegation_id = event.delegation_id.as_ref();
        match &event.backend {
            BackendEvent::ResponseCreated { response_id } => {
                self.created(delegation_id, response_id);
                None
            }
            BackendEvent::FunctionCallDone(call) => {
                self.call_done(delegation_id, call);
                None
            }
            BackendEvent::ResponseEnded {
                response_id,
                outcome,
            } => Some(self.ended(delegation_id, response_id, *outcome)),
            BackendEvent::Other(_) => None,
        }
    }

    /// Whether any response has calls still being collected.
    #[must_use]
    pub fn is_collecting(&self) -> bool {
        self.open.iter().any(|response| !response.calls.is_empty())
    }

    fn created(&mut self, delegation_id: Option<&String>, response_id: &str) {
        if self
            .open
            .iter()
            .any(|open| open.response_id.as_deref() == Some(response_id))
        {
            return;
        }
        // Calls that finished before their response was announced belong to it.
        if let Some(open) = self.unnamed_mut(delegation_id) {
            open.response_id = Some(response_id.to_owned());
            return;
        }
        self.open.push(OpenResponse {
            delegation_id: delegation_id.cloned(),
            response_id: Some(response_id.to_owned()),
            calls: Vec::new(),
        });
    }

    fn call_done(&mut self, delegation_id: Option<&String>, call: &FunctionCall) {
        let index = match self
            .open
            .iter()
            .rposition(|open| open.delegation_id.as_ref() == delegation_id)
        {
            Some(index) => index,
            None => {
                self.open.push(OpenResponse {
                    delegation_id: delegation_id.cloned(),
                    response_id: None,
                    calls: Vec::new(),
                });
                self.open.len() - 1
            }
        };
        if let Some(open) = self.open.get_mut(index)
            && !open.calls.iter().any(|held| held.call_id == call.call_id)
        {
            open.calls.push(call.clone());
        }
    }

    fn ended(
        &mut self,
        delegation_id: Option<&String>,
        response_id: &str,
        outcome: ResponseOutcome,
    ) -> CallsUpdate {
        let index = self
            .open
            .iter()
            .position(|open| open.response_id.as_deref() == Some(response_id))
            .or_else(|| {
                self.open.iter().position(|open| {
                    open.response_id.is_none() && open.delegation_id.as_ref() == delegation_id
                })
            });
        let open = index.map(|index| self.open.remove(index));
        let (delegation_id, calls) = match open {
            Some(open) => (open.delegation_id, open.calls),
            None => (delegation_id.cloned(), Vec::new()),
        };
        if calls.is_empty() {
            return CallsUpdate::Ended {
                delegation_id,
                response_id: response_id.to_owned(),
                outcome,
            };
        }
        let pending = PendingFunctionCalls {
            delegation_id,
            response_id: response_id.to_owned(),
            calls,
        };
        match outcome {
            ResponseOutcome::Completed => CallsUpdate::Submit(pending),
            ResponseOutcome::Failed | ResponseOutcome::Incomplete => {
                CallsUpdate::Abandoned { pending, outcome }
            }
        }
    }

    fn unnamed_mut(&mut self, delegation_id: Option<&String>) -> Option<&mut OpenResponse> {
        self.open
            .iter_mut()
            .find(|open| open.response_id.is_none() && open.delegation_id.as_ref() == delegation_id)
    }
}

#[cfg(test)]
mod tests;
