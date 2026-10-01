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
    /// discards them: every `uncertain` call is available through
    /// [`FunctionCallCollector::unresolved_calls`] until
    /// [`FunctionCallCollector::take_unresolved`] drains it, and is never
    /// attributed to a response afterwards.
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

/// A backend response announced by `response.created` or ended while the
/// collector held calls for it.
#[derive(Clone, Debug)]
struct OpenResponse {
    /// The collector's own identity for the response.
    key: u64,
    /// `None` until an event correlated with it names its delegation.
    delegation_id: Option<String>,
    response_id: String,
    calls: Vec<FunctionCall>,
}

/// Calls that finished while no announced response could own them, held for
/// the next response of exactly their correlation. Calls with different
/// correlations, an absent one included, are never in one group.
#[derive(Clone, Debug)]
struct WaitingGroup {
    delegation_id: Option<String>,
    calls: Vec<FunctionCall>,
}

/// A call whose owner is ambiguous.
#[derive(Clone, Debug)]
struct UnresolvedCall {
    call: FunctionCall,
    /// The announced responses that could own it and have not ended.
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
/// The outer `delegation_id` may be absent on any event, and a known
/// correlation is never discarded. A finished call is attributed to an
/// announced response only when it is the call's sole possible owner. With
/// no announced candidate, the call waits for the next response of exactly
/// its correlation; an uncorrelated waiting call is claimed only when no
/// differently correlated call is also waiting. A call with several possible
/// owners becomes unresolved: each candidate response that ends reports it as
/// [`CallsUpdate::Unresolved`], it stays retrievable until
/// [`Self::take_unresolved`] drains it, and it is never submitted under any
/// response. A repeated `call_id` is kept once.
#[derive(Clone, Debug, Default)]
pub struct FunctionCallCollector {
    open: Vec<OpenResponse>,
    waiting: Vec<WaitingGroup>,
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

    /// Whether any calls are held: owned by an open response, waiting for
    /// one, or unresolved.
    #[must_use]
    pub fn is_collecting(&self) -> bool {
        !self.unresolved.is_empty()
            || self.waiting.iter().any(|group| !group.calls.is_empty())
            || self.open.iter().any(|open| !open.calls.is_empty())
    }

    /// The calls whose owner is ambiguous, in the order they became so.
    /// Every call a [`CallsUpdate::Unresolved`] reported is here until
    /// drained.
    pub fn unresolved_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.unresolved.iter().map(|held| &held.call)
    }

    /// Remove and return the calls whose owner is ambiguous, for the caller
    /// to settle. No later update reports them.
    pub fn take_unresolved(&mut self) -> Vec<FunctionCall> {
        std::mem::take(&mut self.unresolved)
            .into_iter()
            .map(|held| held.call)
            .collect()
    }

    fn holds(&self, call_id: &str) -> bool {
        self.open
            .iter()
            .flat_map(|open| open.calls.iter())
            .chain(self.waiting.iter().flat_map(|group| group.calls.iter()))
            .chain(self.unresolved_calls())
            .any(|held| held.call_id == call_id)
    }

    fn call_done(&mut self, delegation_id: Option<&String>, call: &FunctionCall) {
        if self.holds(&call.call_id) {
            return;
        }
        let candidates: Vec<u64> = self
            .open
            .iter()
            .filter(|open| compatible(open.delegation_id.as_ref(), delegation_id))
            .map(|open| open.key)
            .collect();
        // An uncorrelated call may also belong to the not yet announced
        // response a correlated waiting call stands for.
        let waiting_owner = delegation_id.is_none()
            && self
                .waiting
                .iter()
                .any(|group| group.delegation_id.is_some());
        match candidates.as_slice() {
            [] => self.wait(delegation_id, call.clone()),
            [key] if !waiting_owner => {
                if let Some(open) = self.open.iter_mut().find(|open| open.key == *key) {
                    if open.delegation_id.is_none() {
                        open.delegation_id = delegation_id.cloned();
                    }
                    open.calls.push(call.clone());
                }
            }
            _ => self.unresolved.push(UnresolvedCall {
                call: call.clone(),
                candidates,
            }),
        }
    }

    fn wait(&mut self, delegation_id: Option<&String>, call: FunctionCall) {
        match self
            .waiting
            .iter_mut()
            .find(|group| group.delegation_id.as_ref() == delegation_id)
        {
            Some(group) => group.calls.push(call),
            None => self.waiting.push(WaitingGroup {
                delegation_id: delegation_id.cloned(),
                calls: vec![call],
            }),
        }
    }

    /// Take the waiting calls a response correlated as `delegation_id` can
    /// claim. Returns the claimed calls with the correlation they establish,
    /// and the calls it may or may not own, which leave the waiting groups.
    fn claim_waiting(
        &mut self,
        delegation_id: Option<&String>,
    ) -> (Vec<FunctionCall>, Option<String>, Vec<FunctionCall>) {
        let mut claimed = Vec::new();
        let mut uncertain = Vec::new();
        match delegation_id {
            Some(delegation) => {
                if let Some(index) = self
                    .waiting
                    .iter()
                    .position(|group| group.delegation_id.as_ref() == Some(delegation))
                {
                    claimed.extend(self.waiting.remove(index).calls);
                }
                if let Some(index) = self
                    .waiting
                    .iter()
                    .position(|group| group.delegation_id.is_none())
                {
                    let others_wait = self
                        .waiting
                        .iter()
                        .any(|group| group.delegation_id.is_some());
                    let group = self.waiting.remove(index);
                    if others_wait {
                        uncertain.extend(group.calls);
                    } else {
                        claimed.extend(group.calls);
                    }
                }
                (claimed, Some(delegation.clone()), uncertain)
            }
            None => match self.waiting.len() {
                0 => (claimed, None, uncertain),
                1 => {
                    let group = self.waiting.remove(0);
                    (group.calls, group.delegation_id, uncertain)
                }
                _ => {
                    for group in std::mem::take(&mut self.waiting) {
                        uncertain.extend(group.calls);
                    }
                    (claimed, None, uncertain)
                }
            },
        }
    }

    fn created(&mut self, delegation_id: Option<&String>, response_id: &str) {
        if let Some(open) = self
            .open
            .iter_mut()
            .find(|open| open.response_id == response_id)
        {
            if open.delegation_id.is_none() {
                open.delegation_id = delegation_id.cloned();
            }
            return;
        }
        let key = self.next_key;
        self.next_key += 1;
        let (calls, claimed_delegation, uncertain) = self.claim_waiting(delegation_id);
        self.open.push(OpenResponse {
            key,
            delegation_id: delegation_id.cloned().or(claimed_delegation),
            response_id: response_id.to_owned(),
            calls,
        });
        self.unresolved
            .extend(uncertain.into_iter().map(|call| UnresolvedCall {
                call,
                candidates: vec![key],
            }));
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
            .position(|open| open.response_id == response_id);
        let (delegation_id, owned, uncertain) = match named.map(|index| self.open.remove(index)) {
            Some(open) => {
                let mut uncertain = Vec::new();
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
                    uncertain,
                )
            }
            None => {
                // An unannounced response claims the waiting calls it can;
                // those it may own are reported and become unresolved.
                let (owned, claimed_delegation, uncertain) = self.claim_waiting(delegation_id);
                self.unresolved
                    .extend(uncertain.iter().cloned().map(|call| UnresolvedCall {
                        call,
                        candidates: Vec::new(),
                    }));
                (
                    delegation_id.cloned().or(claimed_delegation),
                    owned,
                    uncertain,
                )
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
