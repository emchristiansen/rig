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
    /// The response ended while calls whose response is ambiguous were held,
    /// and it may own some of them. The collector neither submits nor
    /// discards them; they stay available through
    /// [`FunctionCallCollector::unresolved_calls`].
    Unresolved {
        /// The delegation the response belongs to, when correlated.
        delegation_id: Option<String>,
        /// The response id.
        response_id: String,
        /// How it ended.
        outcome: ResponseOutcome,
        /// The calls known to belong to this response.
        owned: Vec<FunctionCall>,
        /// The calls that may belong to this response or to another one.
        uncertain: Vec<FunctionCall>,
    },
}

/// A backend response whose calls are being collected.
#[derive(Clone, Debug)]
struct OpenResponse {
    /// The collector's own identity for the response, stable before the
    /// response id is known.
    key: u64,
    /// `None` until an event with a correlation names it.
    delegation_id: Option<String>,
    /// `None` until `response.created` or the response's end names it.
    response_id: Option<String>,
    calls: Vec<FunctionCall>,
}

/// A call that could belong to more than one open response.
#[derive(Clone, Debug)]
struct UnresolvedCall {
    call: FunctionCall,
    /// The keys of the open responses that could own it and have not ended.
    candidates: Vec<u64>,
}

/// Whether events with these delegation correlations can belong to the same
/// response: an absent correlation matches any.
fn compatible(left: Option<&String>, right: Option<&String>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

/// Collects the function calls of each backend response from the server
/// events and reports when a response's calls need outputs.
///
/// The outer `delegation_id` may be absent on any event. A finished call is
/// attributed to an open response only when exactly one open response is
/// compatible with its correlation. With none, the call is held for the next
/// compatible response. With more than one, its ownership is ambiguous: it is
/// kept as unresolved and every candidate response that ends reports it as
/// [`CallsUpdate::Unresolved`]. A repeated `call_id` is kept once.
#[derive(Clone, Debug, Default)]
pub struct FunctionCallCollector {
    open: Vec<OpenResponse>,
    unresolved: Vec<UnresolvedCall>,
    next_key: u64,
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

    /// Whether any calls are held: collected for an open response or
    /// unresolved.
    #[must_use]
    pub fn is_collecting(&self) -> bool {
        !self.unresolved.is_empty() || self.open.iter().any(|open| !open.calls.is_empty())
    }

    /// The calls whose response is ambiguous, in the order they finished.
    pub fn unresolved_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.unresolved.iter().map(|held| &held.call)
    }

    /// Remove and return the calls whose response is ambiguous, for the
    /// caller to settle.
    pub fn take_unresolved(&mut self) -> Vec<FunctionCall> {
        std::mem::take(&mut self.unresolved)
            .into_iter()
            .map(|held| held.call)
            .collect()
    }

    fn open_response(&mut self, delegation_id: Option<&String>, response_id: Option<&str>) -> u64 {
        let key = self.next_key;
        self.next_key += 1;
        self.open.push(OpenResponse {
            key,
            delegation_id: delegation_id.cloned(),
            response_id: response_id.map(ToOwned::to_owned),
            calls: Vec::new(),
        });
        key
    }

    /// The open responses not yet named by `response.created` that are
    /// compatible with `delegation_id`, by index.
    fn unnamed_matching(&self, delegation_id: Option<&String>) -> Vec<usize> {
        self.open
            .iter()
            .enumerate()
            .filter(|(_, open)| {
                open.response_id.is_none() && compatible(open.delegation_id.as_ref(), delegation_id)
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn created(&mut self, delegation_id: Option<&String>, response_id: &str) {
        if let Some(open) = self
            .open
            .iter_mut()
            .find(|open| open.response_id.as_deref() == Some(response_id))
        {
            if open.delegation_id.is_none() {
                open.delegation_id = delegation_id.cloned();
            }
            return;
        }
        let matching = self.unnamed_matching(delegation_id);
        if let [index] = matching.as_slice() {
            // Calls that finished before their response was announced belong to it.
            if let Some(open) = self.open.get_mut(*index) {
                open.response_id = Some(response_id.to_owned());
                if open.delegation_id.is_none() {
                    open.delegation_id = delegation_id.cloned();
                }
            }
            return;
        }
        let key = self.open_response(delegation_id, Some(response_id));
        if matching.is_empty() {
            return;
        }
        // Several held groups could be this response's: their calls become
        // unresolved, with this response as a candidate.
        let mut index = 0;
        while let Some(open) = self.open.get(index) {
            if open.response_id.is_none() && compatible(open.delegation_id.as_ref(), delegation_id)
            {
                let held = self.open.remove(index);
                self.unresolved
                    .extend(held.calls.into_iter().map(|call| UnresolvedCall {
                        call,
                        candidates: vec![key],
                    }));
            } else {
                index += 1;
            }
        }
    }

    fn call_done(&mut self, delegation_id: Option<&String>, call: &FunctionCall) {
        let held = self
            .open
            .iter()
            .flat_map(|open| open.calls.iter())
            .chain(self.unresolved_calls())
            .any(|held| held.call_id == call.call_id);
        if held {
            return;
        }
        let candidates: Vec<u64> = self
            .open
            .iter()
            .filter(|open| compatible(open.delegation_id.as_ref(), delegation_id))
            .map(|open| open.key)
            .collect();
        match candidates.as_slice() {
            [] => {
                let key = self.open_response(delegation_id, None);
                if let Some(open) = self.open.iter_mut().find(|open| open.key == key) {
                    open.calls.push(call.clone());
                }
            }
            [key] => {
                if let Some(open) = self.open.iter_mut().find(|open| open.key == *key) {
                    open.calls.push(call.clone());
                }
            }
            _ => self.unresolved.push(UnresolvedCall {
                call: call.clone(),
                candidates,
            }),
        }
    }

    fn ended(
        &mut self,
        delegation_id: Option<&String>,
        response_id: &str,
        outcome: ResponseOutcome,
    ) -> CallsUpdate {
        let named = self
            .open
            .iter()
            .position(|open| open.response_id.as_deref() == Some(response_id));
        let unnamed = self.unnamed_matching(delegation_id);
        let index = named.or(match unnamed.as_slice() {
            [index] => Some(*index),
            _ => None,
        });
        let mut uncertain = Vec::new();
        let (delegation_id, owned) = match index.map(|index| self.open.remove(index)) {
            Some(open) => {
                for held in &mut self.unresolved {
                    if let Some(position) = held.candidates.iter().position(|key| *key == open.key)
                    {
                        held.candidates.remove(position);
                        uncertain.push(held.call.clone());
                    }
                }
                (
                    open.delegation_id.or_else(|| delegation_id.cloned()),
                    open.calls,
                )
            }
            None => {
                // An unannounced response matching several held groups may
                // own any of their calls; they stay held for their own end.
                if unnamed.len() > 1 {
                    uncertain.extend(
                        unnamed
                            .iter()
                            .filter_map(|index| self.open.get(*index))
                            .flat_map(|open| open.calls.iter().cloned()),
                    );
                }
                (delegation_id.cloned(), Vec::new())
            }
        };
        let response_id = response_id.to_owned();
        if !uncertain.is_empty() {
            return CallsUpdate::Unresolved {
                delegation_id,
                response_id,
                outcome,
                owned,
                uncertain,
            };
        }
        if owned.is_empty() {
            return CallsUpdate::Ended {
                delegation_id,
                response_id,
                outcome,
            };
        }
        let pending = PendingFunctionCalls {
            delegation_id,
            response_id,
            calls: owned,
        };
        match outcome {
            ResponseOutcome::Completed => CallsUpdate::Submit(pending),
            ResponseOutcome::Failed | ResponseOutcome::Incomplete => {
                CallsUpdate::Abandoned { pending, outcome }
            }
        }
    }
}

#[cfg(test)]
mod tests;
