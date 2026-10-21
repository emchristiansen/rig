//! Streamed-turn assembly and early invalid-call diagnostics for [`super::AgentRun`].
//!
//! Drivers ingest a completion stream's items, resolve invalid calls before
//! continuing, and finish with the response the stream folded into. The
//! assembler performs no I/O and returns forwarding instructions as
//! [`StreamedTurnEvent`].
//!
//! ```
//! use rig_agent::run::streamed::StreamedTurnAssembler;
//! let assembler = StreamedTurnAssembler::new(Default::default());
//! assert!(assembler.aggregated_text().is_empty());
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use rig_core::completion::{FinishReason, Message};
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolName, ToolResult,
};
use rig_core::streaming::{Item, Part, PartKind, StreamEvent};

use super::policy::{DiagnosticCall, InvalidToolCallReason, TurnPolicy};

/// Detect unknown payloads containing assistant content that assembly would lose:
/// tagged assistant blocks or text with a malformed provider item.
fn unknown_payload_loses_assistant_content(payload: &serde_json::Value) -> bool {
    // Deserialize by reference to avoid cloning large unknown payloads.
    if AssistantContent::deserialize(payload).is_ok() {
        return true;
    }
    // Malformed metadata must not hide the loss of an otherwise valid text field.
    payload
        .get("text")
        .is_some_and(serde_json::Value::is_string)
        && payload.get("native").is_some()
}

/// One invalid tool call surfaced mid-stream, awaiting a resolution from
/// `AgentRun::resolve_streamed_invalid_tool_call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedInvalidToolCall {
    /// The rejected tool call.
    pub tool_call: DiagnosticCall,
    /// Raw argument payload for diagnostics, when available.
    pub args: Option<String>,
    /// The tool policy of the turn the call was judged under.
    pub policy: TurnPolicy,
    /// Why this diagnostic call cannot be dispatched.
    pub reason: InvalidToolCallReason,
}

/// Snapshot of a streamed turn at the moment an invalid tool call appeared.
/// Used by the machine to build diagnostics and rollback messages from
/// exactly what the model has produced so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialStreamedTurn {
    /// The turn the reply began, with no content: its origin, so a rolled
    /// back turn is known to come from its model. A reply cut before its end
    /// keeps no provider item, so the rolled back turn replays canonically.
    #[serde(default)]
    pub head: AssistantMessage,
    /// The parts that arrived so far, in the order they started, text still
    /// streaming included.
    pub content: Vec<AssistantContent>,
    /// Tool calls already validated (or repaired) this turn.
    pub pending_tool_calls: Vec<ToolCall>,
}

impl PartialStreamedTurn {
    /// The assistant message representing this partial turn, in arrival
    /// order: each validated call as validated, `current_tool_call` in its
    /// own place, and any other call left out. A call keeps a provider item
    /// only where the partial reply kept it, which a reply cut before its
    /// end never does. `None` when the turn has produced no representable
    /// content.
    pub fn assistant_message(&self, current_tool_call: Option<ToolCall>) -> Option<Message> {
        let mut calls: Vec<ToolCall> = self
            .pending_tool_calls
            .iter()
            .cloned()
            .chain(current_tool_call)
            .collect();
        let mut content: Vec<AssistantContent> = self
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::ToolCall(call) => {
                    let at = calls.iter().position(|kept| kept.id == call.id)?;
                    let mut kept = calls.remove(at);
                    kept.native.clone_from(&call.native);
                    Some(AssistantContent::ToolCall(kept))
                }
                part => (!part.is_blank()).then(|| part.clone()),
            })
            .collect();
        content.extend(calls.into_iter().map(|mut call| {
            call.native = None;
            AssistantContent::ToolCall(call)
        }));
        if content.is_empty() {
            return None;
        }
        Some(Message::Assistant(self.head.clone().with_content(content)))
    }

    /// Rollback messages for a retried or skipped streamed turn: the partial
    /// assistant turn plus a user message carrying `feedback` for the invalid
    /// call and a synthetic "not executed" result for each validated peer.
    pub fn rollback_messages(
        &self,
        invalid_tool_call: ToolCall,
        feedback: String,
    ) -> Option<(Message, Message)> {
        // Preserve call IDs so synthetic results correlate with their diagnostic calls.
        let assistant_message = self.assistant_message(Some(invalid_tool_call.clone()))?;
        let Message::Assistant(turn) = &assistant_message else {
            return None;
        };
        let user_message = Message::User {
            content: rig_core::transcript::invalid_call_feedback(
                &turn.content,
                &invalid_tool_call.id,
                &feedback,
            ),
        };
        Some((assistant_message, user_message))
    }
}

/// The assembled streamed turn, fed to
/// `AgentRun::streamed_turn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedTurn {
    /// The turn's origin, stop and provider message, without content.
    pub head: AssistantMessage,
    /// The assistant content to record in history, in the order its parts
    /// started, with ignored calls left out and repaired calls renamed.
    pub choice: Vec<AssistantContent>,
    /// The tool policy the turn was prepared with.
    pub policy: TurnPolicy,
    /// Provider-reported terminal reason for this turn, when available.
    pub finish_reason: Option<FinishReason>,
}

/// Resolution a driver must apply to a mid-stream invalid tool call.
#[derive(Debug)]
pub enum StreamedResolution {
    /// The tool name was repaired. Apply it via
    /// [`StreamedTurnAssembler::resolve_pending_invalid`] and keep consuming
    /// the provider stream.
    Repaired {
        /// The validated replacement tool name.
        tool_name: String,
    },
    /// The turn was rolled back (retry) or the call skipped; corrective
    /// messages are already in the history. Finish the provider stream for
    /// usage, record the completion call, then call
    /// `AgentRun::next_step`.
    TurnAbandoned {
        /// For a skipped call, the synthetic tool result to surface to the
        /// consumer stream.
        skipped_tool_result: Option<ToolResult>,
    },
    /// The invalid call is dropped and the turn goes on without it: the
    /// run's `UnhandledInvalidToolCall::Ignore` on the streaming
    /// surface. Apply it via
    /// [`StreamedTurnAssembler::resolve_pending_invalid`] and keep consuming
    /// the provider stream; nothing of the call enters the run.
    Ignored,
}

/// Required driver action after ingesting a stream item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamedTurnEvent {
    /// Forward the ingested item to the consumer as-is.
    EmitIngested,
    /// Forward the ingested argument fragment of a call to an allowed tool,
    /// once the tool-call delta hooks let it through. The call is not
    /// executable until its end.
    EmitToolCallDelta,
    /// Hold the ingested item back: it belongs to a call naming a tool the
    /// turn does not allow, which is resolved at its end.
    HoldToolCall,
    /// Forward the held items of the call that just ended, then its end
    /// carrying `call` (the call as validated, with a repaired name).
    EmitToolCall {
        /// The validated call.
        call: ToolCall,
    },
    /// The model emitted an unknown or disallowed tool call. Resolve it via
    /// `AgentRun::resolve_streamed_invalid_tool_call`, then apply the
    /// outcome with [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(StreamedInvalidToolCall),
    /// A completed namespaced or custom call refused by name-keyed dispatch.
    UndispatchableToolCall(RefusedStreamedCall),
}

/// A completed provider call that name-keyed dispatch cannot execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefusedStreamedCall {
    /// The typed dispatch refusal.
    pub call: rig_core::message::UndispatchableToolCall,
    /// The actual finalized provider content.
    pub content: AssistantContent,
}

/// Engine forwarding instructions, including early delivery of real held items.
#[derive(Debug)]
pub(crate) enum DriverTurnEvent {
    Public(StreamedTurnEvent),
    EmitResolvedItems { items: Vec<Item<StreamEvent>> },
}

impl From<StreamedTurnEvent> for DriverTurnEvent {
    fn from(event: StreamedTurnEvent) -> Self { Self::Public(event) }
}

fn driver_events(
    items: Vec<Item<StreamEvent>>,
    events: Vec<StreamedTurnEvent>,
) -> Vec<DriverTurnEvent> {
    let mut output = Vec::new();
    if !items.is_empty() { output.push(DriverTurnEvent::EmitResolvedItems { items }); }
    output.extend(events.into_iter().map(DriverTurnEvent::Public));
    output
}

/// A tool call still streaming: the tool it named as it started, whether
/// its items stream live, and its argument text so far.
#[derive(Clone, Serialize, Deserialize)]
struct OpenCall {
    name: Option<ToolName>,
    live: bool,
    arguments: String,
    held: Vec<Item<StreamEvent>>,
    ignored: bool,
}

/// A complete tool call with a disallowed name, awaiting resolution.
#[derive(Clone, Serialize, Deserialize)]
enum PendingInvalid {
    FullCall(ToolCall),
    Name { part: Part },
    Malformed { id: CallId },
}

/// Serializable accumulator for one streamed turn. Persisted state requires the
/// same library version; it does not itself resume a provider connection.
/// Drivers must resolve pending invalid calls before ingesting more events.
/// Dropping warns once if assistant content was excluded, otherwise stays silent.
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamedTurnAssembler {
    policy: TurnPolicy,
    text: String,
    /// Reasoning text per part, by the part's position.
    reasoning: BTreeMap<usize, String>,
    /// Tool calls still streaming, by the part's position.
    #[serde(default)]
    open_calls: BTreeMap<usize, OpenCall>,
    pending_tool_calls: Vec<ToolCall>,
    pending_invalid: Option<PendingInvalid>,
    /// Calls an [`StreamedResolution::Ignored`] dropped: the stream's
    /// response still carries them, and [`Self::finish`] leaves them out.
    ignored_calls: Vec<CallId>,
    /// Calls a [`StreamedResolution::Repaired`] renamed, applied to the
    /// stream's response at [`Self::finish`].
    repaired_calls: Vec<(CallId, ToolName)>,
    repaired_parts: BTreeMap<usize, ToolName>,
    /// Replayed assistant blocks excluded from assembly this turn (see
    /// [`unknown_payload_loses_assistant_content`]): counted per item,
    /// surfaced as one warning when the guard drops.
    excluded_assistant_content: ExclusionCount,
}

/// Persisted count of excluded assistant blocks. Each clone warns once on drop
/// when nonzero, including cancellation and error paths. A separate drop guard
/// leaves the assembler's fields movable.
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
struct ExclusionCount(usize);

impl Drop for ExclusionCount {
    fn drop(&mut self) {
        if self.0 > 0 {
            tracing::warn!(
                excluded = self.0,
                "stream items matching rig's tagged assistant-content \
                 serialization were excluded from the assembled assistant \
                 message — replayed assistant blocks are not stream-item \
                 shapes, and their content is lost from assembled history"
            );
        }
    }
}

impl StreamedTurnAssembler {
    /// Create an assembler for one streamed turn with the policy the turn
    /// was prepared with.
    pub fn new(policy: TurnPolicy) -> Self {
        Self {
            policy,
            text: String::new(),
            reasoning: BTreeMap::new(),
            open_calls: BTreeMap::new(),
            pending_tool_calls: Vec::new(),
            pending_invalid: None,
            ignored_calls: Vec::new(),
            repaired_calls: Vec::new(),
            repaired_parts: BTreeMap::new(),
            excluded_assistant_content: ExclusionCount::default(),
        }
    }

    /// Replayed assistant blocks excluded from assembly so far this turn.
    /// Zero on well-formed provider streams; non-zero means transcript
    /// content was lost (one warning summarizes the count).
    pub fn excluded_assistant_content(&self) -> usize {
        self.excluded_assistant_content.0
    }

    /// Aggregated assistant text streamed so far this turn.
    pub fn aggregated_text(&self) -> &str {
        &self.text
    }

    /// The reasoning text accumulated so far for the part at `index`.
    pub fn aggregated_reasoning(&self, index: usize) -> Option<&str> {
        self.reasoning.get(&index).map(String::as_str)
    }

    /// The argument text the streaming call at part `index` sent so far.
    pub fn aggregated_arguments(&self, index: usize) -> Option<&str> {
        self.open_calls
            .get(&index)
            .map(|call| call.arguments.as_str())
    }

    /// The tool the streaming call at part `index` named as it started.
    pub fn streaming_tool_name(&self, index: usize) -> Option<&ToolName> {
        self.open_calls
            .get(&index)
            .and_then(|call| call.name.as_ref())
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when an invalid tool call is still awaiting
    /// resolution. Pending names and early repairs hold genuine arrivals until
    /// an object call ends; the engine's private view receives them earlier.
    pub fn ingest(
        &mut self,
        item: &Item<StreamEvent>,
    ) -> Result<Vec<StreamedTurnEvent>, ProviderError> {
        self.ingest_inner(item, None)
    }

    /// Ingest for the engine, delivering genuine held items at a name binding.
    pub(crate) fn ingest_for_driver(
        &mut self,
        item: &Item<StreamEvent>,
    ) -> Result<Vec<DriverTurnEvent>, ProviderError> {
        let mut items = Vec::new();
        let events = self.ingest_inner(item, Some(&mut items))?;
        Ok(driver_events(items, events))
    }

    fn ingest_inner(
        &mut self,
        item: &Item<StreamEvent>,
        early_items: Option<&mut Vec<Item<StreamEvent>>>,
    ) -> Result<Vec<StreamedTurnEvent>, ProviderError> {
        if self.pending_invalid.is_some() {
            return Err(ProviderError::Response(
                "streamed turn ingested while an invalid tool call awaits resolution".to_string(),
            ));
        }
        let event = match item {
            Item::Event(event) => event,
            Item::Unknown(payload) => {
                // Unknown items have no assistant-content representation. Count lost
                // assistant blocks for one warning without exposing payloads or flooding logs.
                if unknown_payload_loses_assistant_content(payload.value()) {
                    self.excluded_assistant_content.0 += 1;
                    tracing::debug!(
                        excluded = self.excluded_assistant_content.0,
                        "stream item is a replayed assistant block, not a \
                         stream-item shape; excluded from assembly"
                    );
                }
                return Ok(vec![StreamedTurnEvent::EmitIngested]);
            }
        };
        match event {
            StreamEvent::Text { text, .. } => {
                self.text.push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Reasoning { part, text } => {
                self.reasoning
                    .entry(part.index())
                    .or_default()
                    .push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Start { part, kind: PartKind::ToolCall, name, namespace, .. } => {
                let call = self.open_calls.entry(part.index()).or_insert_with(|| OpenCall {
                    name: None, live: false, arguments: String::new(), held: Vec::new(), ignored: false,
                });
                // PendingName and its later Named binding share these argument arrivals.
                call.name.clone_from(name);
                call.live = namespace.is_none() && name.as_ref().is_some_and(|name| self.policy.allows(name.as_str()));
                if call.ignored { return Ok(Vec::new()); }
                if call.live {
                    if call.held.is_empty() { return Ok(vec![StreamedTurnEvent::EmitIngested]); }
                    let Some(items) = early_items else {
                        call.live = false;
                        call.held.push(item.clone());
                        return Ok(vec![StreamedTurnEvent::HoldToolCall]);
                    };
                    items.append(&mut call.held);
                    items.push(item.clone());
                    return Ok(Vec::new());
                }
                call.held.push(item.clone());
                if namespace.is_none() && let Some(name) = name {
                    let diagnostic = DiagnosticCall::early_name(part.diagnostic_call_id(), name.to_string(), call.arguments.clone());
                    let reason = self.name_reason(name.as_str());
                    return Ok(self.surface_invalid_call(diagnostic, PendingInvalid::Name { part: *part }, reason));
                }
                Ok(vec![StreamedTurnEvent::HoldToolCall])
            }
            StreamEvent::Arguments { part, json } => {
                let call = self.open_calls.entry(part.index()).or_insert_with(|| OpenCall {
                    name: None, live: false, arguments: String::new(), held: Vec::new(), ignored: false,
                });
                if call.ignored { return Ok(Vec::new()); }
                call.arguments.push_str(json);
                if call.live { Ok(vec![StreamedTurnEvent::EmitToolCallDelta]) }
                else {
                    call.held.push(item.clone());
                    Ok(vec![StreamedTurnEvent::HoldToolCall])
                }
            }
            StreamEvent::End { part, content: AssistantContent::ToolCall(tool_call), .. } => {
                let open = self.open_calls.remove(&part.index());
                if open.as_ref().is_some_and(|call| call.ignored) {
                    self.ignored_calls.push(tool_call.id.clone());
                    return Ok(Vec::new());
                }
                if let Err(call) = tool_call.name_dispatchable() {
                    return Ok(vec![StreamedTurnEvent::UndispatchableToolCall(RefusedStreamedCall { call, content: AssistantContent::ToolCall(tool_call.clone()) })]);
                }
                let mut tool_call = tool_call.clone();
                if let Some(name) = self.repaired_parts.get(&part.index()) {
                    tool_call.function.name = name.clone();
                    self.repaired_calls.push((tool_call.id.clone(), name.clone()));
                }
                if let Some(raw) = open.as_ref().map(|call| call.arguments.as_str())
                    .filter(|raw| !raw.trim().is_empty())
                    && let Ok(value) = serde_json::from_str::<serde_json::Value>(raw)
                    && !value.is_object()
                {
                    let diagnostic = DiagnosticCall {
                        name: tool_call.function.name.to_string(),
                        id: tool_call.id.clone(), provider: None, raw: Some(raw.to_owned()),
                        arguments: super::policy::DiagnosticArguments::Parsed(value),
                    };
                    let reason = InvalidToolCallReason::malformed_arguments(raw);
                    return Ok(self.surface_invalid_call(diagnostic,
                        PendingInvalid::Malformed { id: tool_call.id.clone() }, reason));
                }
                if let Some(raw) = &tool_call.function.invalid_arguments {
                    let diagnostic = DiagnosticCall::from_call(&tool_call, Some(raw.clone()));
                    let reason = InvalidToolCallReason::malformed_arguments(raw);
                    return Ok(self.surface_invalid_call(diagnostic, PendingInvalid::Malformed { id: tool_call.id.clone() }, reason));
                }
                if !self.policy.allows(tool_call.function.name.as_str()) {
                    let diagnostic = DiagnosticCall::from_call(&tool_call, Some(tool_call.function.arguments_value().to_string()));
                    let reason = self.name_reason(tool_call.function.name.as_str());
                    return Ok(self.surface_invalid_call(diagnostic, PendingInvalid::FullCall(tool_call), reason));
                }
                self.pending_tool_calls.push(tool_call.clone());
                Ok(vec![StreamedTurnEvent::EmitToolCall { call: tool_call }])
            }
            StreamEvent::End { content: AssistantContent::CustomToolCall(call), .. } => {
                Ok(vec![StreamedTurnEvent::UndispatchableToolCall(RefusedStreamedCall {
                    call: rig_core::message::UndispatchableToolCall::Custom { id: call.id.clone(), name: call.name.clone(), namespace: call.namespace.clone() }, content: AssistantContent::CustomToolCall(call.clone()),
                })])
            }
            StreamEvent::Discard { part, .. } => {
                self.open_calls.remove(&part.index());
                self.repaired_parts.remove(&part.index());
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Input { .. } => Ok(vec![StreamedTurnEvent::HoldToolCall]),
            StreamEvent::Resume { part } => {
                if let Some(call) = self.open_calls.get_mut(&part.index()) {
                    if call.ignored { return Ok(Vec::new()); }
                    if !call.live {
                        call.held.push(item.clone());
                        return Ok(vec![StreamedTurnEvent::HoldToolCall]);
                    }
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Start { .. } | StreamEvent::End { .. } => {
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
        }
    }

    /// Apply the machine's resolution for the invalid tool call surfaced by
    /// the last [`StreamedTurnEvent::InvalidToolCall`]. A repaired call is
    /// returned to forward under its new name.
    pub fn resolve_pending_invalid(
        &mut self,
        resolution: &StreamedResolution,
    ) -> Vec<StreamedTurnEvent> {
        self.resolve_inner(resolution, None)
    }

    /// Apply a resolution for the engine's early-delivery view.
    pub(crate) fn resolve_pending_invalid_for_driver(
        &mut self,
        resolution: &StreamedResolution,
    ) -> Vec<DriverTurnEvent> {
        let mut items = Vec::new();
        let events = self.resolve_inner(resolution, Some(&mut items));
        driver_events(items, events)
    }

    fn resolve_inner(
        &mut self,
        resolution: &StreamedResolution,
        early_items: Option<&mut Vec<Item<StreamEvent>>>,
    ) -> Vec<StreamedTurnEvent> {
        let Some(pending) = self.pending_invalid.take() else {
            return Vec::new();
        };

        match pending {
            PendingInvalid::FullCall(mut tool_call) => match resolution {
                StreamedResolution::Repaired { tool_name } => {
                    if let Ok(name) = ToolName::new(tool_name.clone()) {
                        self.repaired_calls.push((tool_call.id.clone(), name.clone()));
                        tool_call.function.name = name;
                    }
                    self.pending_tool_calls.push(tool_call.clone());
                    vec![StreamedTurnEvent::EmitToolCall { call: tool_call }]
                }
                StreamedResolution::TurnAbandoned { .. } => Vec::new(),
                StreamedResolution::Ignored => { self.ignored_calls.push(tool_call.id); Vec::new() }
            },
            PendingInvalid::Name { part } => {
                let Some(call) = self.open_calls.get_mut(&part.index()) else { return Vec::new(); };
                match resolution {
                    StreamedResolution::Repaired { tool_name } => {
                        let Ok(name) = ToolName::new(tool_name.clone()) else { return Vec::new(); };
                        call.name = Some(name.clone());
                        call.live = true;
                        self.repaired_parts.insert(part.index(), name.clone());
                        for item in &mut call.held {
                            if let Item::Event(StreamEvent::Start { name: old, .. }) = item {
                                if old.is_some() { *old = Some(name.clone()); }
                            }
                        }
                        match early_items {
                            None => call.live = false,
                            Some(items) => items.append(&mut call.held),
                        }
                        Vec::new()
                    }
                    StreamedResolution::Ignored => { call.ignored = true; call.held.clear(); Vec::new() }
                    StreamedResolution::TurnAbandoned { .. } => Vec::new(),
                }
            }
            PendingInvalid::Malformed { id } => {
                if matches!(resolution, StreamedResolution::Ignored) { self.ignored_calls.push(id); }
                Vec::new()
            },
        }
    }

    /// Snapshot of the turn so far, for diagnostics and rollback messages:
    /// `partial` is what the stream folded so far
    /// ([`Streamed::partial`](rig_core::streaming::Streamed::partial)).
    pub fn partial_turn(
        &self,
        partial: &rig_core::completion::CompletionResponse,
    ) -> PartialStreamedTurn {
        PartialStreamedTurn {
            head: partial.continued(Vec::new()),
            content: partial.choice.clone(),
            pending_tool_calls: self.pending_tool_calls.clone(),
        }
    }

    /// Assemble the completed turn from the response the stream folded
    /// into: its choice in start order, with the calls this turn ignored
    /// left out and the ones it repaired renamed.
    pub fn finish(self, response: &rig_core::completion::CompletionResponse) -> StreamedTurn {
        let choice = response
            .choice
            .iter()
            .filter(|content| match content {
                AssistantContent::ToolCall(call) => !self.ignored_calls.contains(&call.id),
                _ => true,
            })
            .cloned()
            .map(|content| match content {
                AssistantContent::ToolCall(mut call) => {
                    if let Some((_, name)) =
                        self.repaired_calls.iter().find(|(id, _)| *id == call.id)
                    {
                        call.function.name = name.clone();
                    }
                    AssistantContent::ToolCall(call)
                }
                other => other,
            })
            .collect();

        StreamedTurn {
            head: response.head(),
            choice,
            policy: self.policy,
            finish_reason: response.finish_reason(),
        }
    }

    /// Park resolution on `tool_call` and surface it to the caller as an
    /// [`StreamedTurnEvent::InvalidToolCall`].
    fn name_reason(&self, name: &str) -> InvalidToolCallReason {
        if self.policy.executable().contains(name) { InvalidToolCallReason::DisallowedByToolChoice }
        else { InvalidToolCallReason::UnknownTool }
    }

    fn surface_invalid_call(
        &mut self,
        tool_call: DiagnosticCall,
        pending: PendingInvalid,
        reason: InvalidToolCallReason,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            args: tool_call.raw.clone(), tool_call,
            policy: self.policy.clone(), reason,
        };
        self.pending_invalid = Some(pending);
        vec![StreamedTurnEvent::InvalidToolCall(invalid)]
    }

    /// Surface the exact provider diagnostic without manufacturing a canonical call.
    pub fn surface_malformed_input(&mut self, detail: &rig_core::error::MalformedToolInput) -> Vec<StreamedTurnEvent> {
        self.surface_invalid_call(DiagnosticCall::malformed(detail), PendingInvalid::Malformed { id: detail.id.clone() },
            InvalidToolCallReason::MalformedArguments { error: detail.error.clone() })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pending_delivery_tests;
