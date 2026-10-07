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
use rig_core::streaming::{Item, PartKind, StreamEvent};

use super::policy::TurnPolicy;

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
    pub tool_call: ToolCall,
    /// Raw argument payload for diagnostics, when available.
    pub args: Option<String>,
    /// The tool policy of the turn the call was judged under.
    pub policy: TurnPolicy,
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
    /// The model emitted a tool call that name-keyed dispatch cannot run: a
    /// namespaced call or a custom call. Refused as it completes, before
    /// name authorization or invalid-call recovery; the driver fails the run
    /// through `AgentRun::refuse_streamed_undispatchable`.
    UndispatchableToolCall(RefusedStreamedCall),
    /// The model emitted an unknown or disallowed tool call. Resolve it via
    /// `AgentRun::resolve_streamed_invalid_tool_call`, then apply the
    /// outcome with [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(StreamedInvalidToolCall),
    /// The provider supplied its typed final payload. Record its usage (see
    /// `AgentRun::record_streamed_completion_call`);
    /// this does not establish that the provider stream reached EOF. When
    /// `emit_final` is set, the turn streamed text and the driver should buffer
    /// the final item until EOF finalizes the turn.
    Completed {
        /// Provider-reported usage for this call. Usage whose counters are
        /// all `None` means the provider reported no usage metrics.
        usage: Usage,
        /// Whether the ingested final item should be forwarded to the
        /// consumer (set when the turn streamed text).
        emit_final: bool,
        /// Why the provider stopped generating, when reported.
        finish_reason: Option<FinishReason>,
        /// The provider's own terminal record, as carried on
        /// `StreamFinal::raw`: what the driver records with
        /// [`AgentRun::record_streamed_completion_call`](super::AgentRun::record_streamed_completion_call)
        /// so the call carries this attempt's payload.
        raw: serde_json::Value,
    },
}

/// A completed streamed call refused by
/// [`rig_core::message::name_dispatchable_call`]: the refusal, and the call
/// as the stream delivered it for diagnostic history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefusedStreamedCall {
    /// Why the call cannot be dispatched.
    pub call: rig_core::message::UndispatchableToolCall,
    /// The refused call, verbatim.
    pub content: AssistantContent,
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct ToolCallDeltaState {
    name_validated: bool,
    buffered_arguments: Vec<String>,
    /// The block's call was ignored (`StreamedResolution::Ignored` on its
    /// name delta): its later deltas and its end are swallowed, and the
    /// state is kept so they have something to land on.
    ignored: bool,
}

/// One reasoning part of the turn, in first-arrival order. A part opens as
/// delta text keyed by the stream's block id and is
/// superseded in place when a completed block restating the same part
/// arrives; a completed block matching no open part occupies its own slot.
#[derive(Clone, Serialize, Deserialize)]
struct ReasoningPart {
    correlator: Option<BlockId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    aliases: Vec<BlockId>,
    provider_id: Option<String>,
    state: ReasoningPartState,
}

impl ReasoningPart {
    fn matches_key(&self, key: &BlockId) -> bool {
        self.correlator.as_ref() == Some(key) || self.aliases.contains(key)
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum ReasoningPartState {
    /// Delta text accumulated so far for a part with no completed block.
    Pending(String),
    /// The authoritative completed block (may carry signatures or encrypted
    /// content the deltas lacked).
    Completed(Reasoning),
}

/// Assemble one part's reasoning: a completed block as-is, a non-empty pending
/// delta buffer as its own block carrying only the part's provider-issued id.
fn reasoning_from_part(
    state: ReasoningPartState,
    provider_id: Option<String>,
) -> Option<Reasoning> {
    match state {
        ReasoningPartState::Completed(reasoning) => Some(reasoning),
        ReasoningPartState::Pending(text) if !text.is_empty() => {
            let mut assembled = Reasoning::new(&text);
            if let Some(id) = provider_id {
                assembled = assembled.with_id(id);
            }
            Some(assembled)
        }
        ReasoningPartState::Pending(_) => None,
    }
}

// Keep independently addressable blocks internally so a trailing signature
// replaces only its own content. Group completed provider-item parts only when
// exposing history, preserving the established grouping of distinct keys while
// keeping repeated-key siblings separate.
fn group_reasoning(parts: impl Iterator<Item = ReasoningPart>) -> Vec<Reasoning> {
    let mut groups: Vec<(Reasoning, Vec<BlockId>, bool)> = Vec::new();
    for part in parts {
        let completed = matches!(part.state, ReasoningPartState::Completed(_));
        let keys: Vec<_> = part.correlator.into_iter().chain(part.aliases).collect();
        let Some(reasoning) = reasoning_from_part(part.state, part.provider_id) else {
            continue;
        };
        if completed
            && reasoning.id.is_some()
            && let Some((existing, existing_keys, _)) = groups
                .iter_mut()
                .rev()
                .find(|(existing, _, completed)| *completed && existing.id == reasoning.id)
            && keys.iter().all(|key| !existing_keys.contains(key))
        {
            existing.content.extend(reasoning.content);
            existing_keys.extend(keys);
            continue;
        }
        groups.push((reasoning, keys, completed));
    }
    groups
        .into_iter()
        .map(|(reasoning, _, _)| reasoning)
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
enum PendingInvalid {
    /// A complete tool call with a disallowed name.
    FullCall {
        tool_call: ToolCall,
        block_id: BlockId,
    },
    /// A streamed tool-name delta with a disallowed name.
    NameDelta { block_id: BlockId },
    /// A complete tool call whose arguments were not JSON. The provider's
    /// accumulator already finalized the block; nothing is buffered here
    /// to replay, so the only resolutions are abandon or fail.
    MalformedArgs { tool_call: ToolCall },
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

    /// Return the newest matching reasoning part's provider ID, or `None` if
    /// absent. Reused correlators refer to their latest pending or completed part.
    pub fn reasoning_provider_id(&self, correlator: &BlockId) -> Option<&str> {
        self.reasoning_parts
            .iter()
            .rev()
            .find(|part| part.matches_key(correlator))
            .and_then(|part| part.provider_id.as_deref())
    }

    /// The reasoning text accumulated so far under `correlator`.
    pub fn aggregated_reasoning(&self, correlator: &BlockId) -> Option<&str> {
        self.reasoning_parts
            .iter()
            .find_map(|part| match (&part.state, part.correlator.as_ref()) {
                (ReasoningPartState::Pending(text), Some(id)) if id == correlator => {
                    Some(text.as_str())
                }
                _ => None,
            })
    }

    /// Combine accepted calls and reasoning with provider text, provider items
    /// and images in canonical order. Without calls or reasoning, preserve the
    /// provider choice.
    fn canonical_choice_with(
        pending_tool_calls: Vec<(ToolCall, BlockId)>,
        reasoning: Vec<Reasoning>,
        provider_choice: &[AssistantContent],
    ) -> Vec<AssistantContent> {
        if !pending_tool_calls.is_empty() || !reasoning.is_empty() {
            let parts = reasoning
                .into_iter()
                .map(AssistantContent::Reasoning)
                // Text and provider items, in the order the provider sent them.
                .chain(provider_choice.iter().filter_map(|content| {
                    match content {
                        AssistantContent::ProviderItem(_) => Some(content.clone()),
                        other => assistant_text_items_from_choice(std::slice::from_ref(other))
                            .into_iter()
                            .next(),
                    }
                }))
                .chain(
                    pending_tool_calls
                        .into_iter()
                        .map(|(tool_call, _)| AssistantContent::ToolCall(tool_call)),
                )
                .chain(
                    provider_choice
                        .iter()
                        .filter(|content| matches!(content, AssistantContent::Image(_)))
                        .cloned(),
                )
                .collect();
            canonical_streamed_choice(parts)
        } else {
            provider_choice.to_vec()
        }
    }

    /// A completed block replaces its pending deltas. After a key closes,
    /// an explicit whole block or a second signature opens a sibling; only
    /// trailing metadata for an unsigned close updates that completed part.
    /// The normalized end retains this distinction in its original payload.
    fn ingest_completed_reasoning(
        &mut self,
        reasoning: &Reasoning,
        correlator: &BlockId,
        restatement: bool,
    ) {
        let replace_at = self
            .reasoning_parts
            .iter()
            .rposition(|part| {
                part.matches_key(correlator) && matches!(part.state, ReasoningPartState::Pending(_))
            })
            .or_else(|| {
                // A known block key is more specific than a shared provider
                // item id. Its next whole block must not consume a different
                // key's still-pending content.
                if self
                    .reasoning_parts
                    .iter()
                    .any(|part| part.matches_key(correlator))
                {
                    return None;
                }
                self.reasoning_parts.iter().rposition(|part| {
                    matches!(part.state, ReasoningPartState::Pending(_))
                        && matches!(
                            (&part.provider_id, &reasoning.id),
                            (Some(pending_id), Some(incoming_id)) if pending_id == incoming_id
                        )
                })
            });
        if let Some(part) = replace_at.and_then(|index| self.reasoning_parts.get_mut(index)) {
            if reasoning.id.is_some() {
                part.provider_id.clone_from(&reasoning.id);
            }
            if !part.matches_key(correlator) {
                part.aliases.push(correlator.clone());
            }
            part.state = ReasoningPartState::Completed(reasoning.clone());
            return;
        }

        if let Some(part) = self
            .reasoning_parts
            .iter_mut()
            .rev()
            .find(|part| part.matches_key(correlator))
        {
            let signed = matches!(&part.state, ReasoningPartState::Completed(existing)
                if existing.content.iter().any(|content| matches!(content,
                    rig_core::message::ReasoningContent::Text { signature: Some(_), .. })));
            if !restatement && !signed {
                if reasoning.id.is_some() {
                    part.provider_id.clone_from(&reasoning.id);
                }
                part.state = ReasoningPartState::Completed(reasoning.clone());
                return;
            }
            // Keep this completed sibling separately addressable. History
            // grouping recognizes key reuse and preserves both parts.
            self.reasoning_parts.push(ReasoningPart {
                aliases: Vec::new(),
                correlator: Some(correlator.clone()),
                provider_id: reasoning.id.clone(),
                state: ReasoningPartState::Completed(reasoning.clone()),
            });
            return;
        }

        self.reasoning_parts.push(ReasoningPart {
            aliases: Vec::new(),
            correlator: Some(correlator.clone()),
            provider_id: reasoning.id.clone(),
            state: ReasoningPartState::Completed(reasoning.clone()),
        });
    }

    /// The turn's reasoning in first-arrival order. Completed parts from
    /// distinct keys sharing a provider item are grouped; repeated-key siblings
    /// and pending delta buffers retain their separate slots.
    fn assembled_reasoning(&self) -> Vec<Reasoning> {
        group_reasoning(self.reasoning_parts.iter().cloned())
    }

    /// Consume reasoning parts in canonical grouping, avoiding copies of large
    /// encrypted payloads.
    fn drain_reasoning(&mut self) -> Vec<Reasoning> {
        group_reasoning(std::mem::take(&mut self.reasoning_parts).into_iter())
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when an invalid tool call is still awaiting
    /// resolution.
    pub fn ingest(
        &mut self,
        item: &Item<StreamEvent>,
    ) -> Result<Vec<StreamedTurnEvent>, ProviderError> {
        if self.pending_invalid.is_some() {
            return Err(ProviderError::Response(
                "streamed turn ingested while an invalid tool call awaits resolution".to_string(),
            ));
        }

        match item {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => {
                if !self.saw_text {
                    self.text.clear();
                    self.saw_text = true;
                }
                self.text.push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            // The driver and provider aggregate retain message identity and text metadata.
            // A provider item's blocks too: the aggregate records the item.
            StreamEvent::BlockStart {
                kind:
                    BlockKind::Message
                    | BlockKind::Text { .. }
                    | BlockKind::ToolCall
                    | BlockKind::ProviderItem,
                ..
            }
            | StreamEvent::BlockDelta {
                delta: Delta::TextMeta { .. },
                ..
            }
            | StreamEvent::BlockEnd {
                end: BlockClose::Text | BlockClose::ProviderItem(_),
                ..
            } => Ok(vec![StreamedTurnEvent::EmitIngested]),
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { provider_id },
                ..
            } => {
                let pending = self.reasoning_parts.iter_mut().find(|part| {
                    part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                });
                match pending {
                    Some(part) => {
                        if part.provider_id.is_none() {
                            part.provider_id.clone_from(provider_id);
                        }
                    }
                    None => self.reasoning_parts.push(ReasoningPart {
                        aliases: Vec::new(),
                        correlator: Some(id.clone()),
                        provider_id: provider_id.clone(),
                        state: ReasoningPartState::Pending(String::new()),
                    }),
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockEnd {
                id,
                end:
                    BlockClose::Reasoning {
                        reasoning: restatement,
                        ..
                    },
                block,
            } => {
                if let Some(AssistantContent::Reasoning(reasoning)) = block {
                    self.ingest_completed_reasoning(reasoning, id, restatement.is_some());
                } else if let Some(part) = self.reasoning_parts.iter_mut().rev().find(|part| {
                    part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                }) && let ReasoningPartState::Pending(text) = &part.state
                {
                    // A synthesized silent close publishes no whole block,
                    // but still ends this part. Later deltas reopen the key;
                    // they must not extend or overwrite the completed text.
                    let content = if text.is_empty() {
                        Vec::new()
                    } else {
                        Reasoning::new(text).content
                    };
                    part.state = ReasoningPartState::Completed(Reasoning {
                        provider: None,
                        id: part.provider_id.clone(),
                        content,
                    });
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { text: reasoning },
            } => {
                // Keep unsigned deltas separate until an authoritative completed block
                // supplies metadata. Only provider IDs, not correlators, enter history.
                let index = self
                    .reasoning_parts
                    .iter()
                    .position(|part| {
                        part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                    })
                    .unwrap_or_else(|| {
                        self.reasoning_parts.push(ReasoningPart {
                            aliases: Vec::new(),
                            correlator: Some(id.clone()),
                            provider_id: None,
                            state: ReasoningPartState::Pending(String::new()),
                        });
                        self.reasoning_parts.len() - 1
                    });
                if let Some(part) = self.reasoning_parts.get_mut(index)
                    && let ReasoningPartState::Pending(text) = &mut part.state
                {
                    text.push_str(reasoning);
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            // A custom call is never dispatchable: refused whole, even when
            // its input parses as JSON. A repeated end finalizes nothing.
            StreamEvent::BlockEnd {
                id: block_id,
                end: BlockClose::CustomToolCall(_),
                block,
            } => {
                self.delta_states.remove(block_id);
                Ok(block
                    .as_ref()
                    .and_then(Self::refuse_undispatchable)
                    .into_iter()
                    .collect())
            }
            // A namespaced function call is refused before the name-keyed
            // allowed-set check and before invalid-call recovery could expose
            // it by its bare name.
            StreamEvent::BlockEnd {
                id: block_id,
                end: BlockClose::ToolCall(_),
                block: Some(block),
            } if Self::refuse_undispatchable(block).is_some() => {
                self.delta_states.remove(block_id);
                Ok(Self::refuse_undispatchable(block).into_iter().collect())
            }
            StreamEvent::BlockEnd {
                id: block_id,
                end: BlockClose::ToolCall(_),
                block,
            } => {
                // The driver emits completed calls at turn commit, not at block end.
                if self
                    .delta_states
                    .get(block_id)
                    .is_some_and(|state| state.ignored)
                {
                    // The final durable ID may differ from the name delta's block key;
                    // retain it so provider snapshots cannot restore an ignored call.
                    if let Some(AssistantContent::ToolCall(call)) = block {
                        self.ignored_calls.push(call.id.clone());
                    }
                    self.delta_states.remove(block_id);
                    return Ok(Vec::new());
                }
                let Some(AssistantContent::ToolCall(tool_call)) = block else {
                    self.delta_states.remove(block_id);
                    return Ok(Vec::new());
                };
                if !self.allowed_tool_names.contains(&tool_call.function.name) {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        block_id.clone(),
                        Some(json_utils::serialize_json_value(
                            &tool_call.function.arguments,
                        )),
                        PendingInvalid::FullCall {
                            tool_call: tool_call.clone(),
                            block_id: block_id.clone(),
                        },
                        InvalidToolCallReason::UnknownTool,
                    ));
                }

                self.pending_tool_calls
                    .push((tool_call.clone(), block_id.clone()));
                Ok(Vec::new())
            }
            StreamEvent::BlockDelta {
                id: block_id,
                delta: delta @ (Delta::ToolName { .. } | Delta::ToolArguments { .. }),
            } => {
                let key = block_id.clone();
                if self
                    .delta_states
                    .get(&key)
                    .is_some_and(|state| state.ignored)
                {
                    return Ok(Vec::new());
                }
                match delta {
                    Delta::ToolName { name } => {
                        if !self.allowed_tool_names.contains(name) {
                            let buffered_args = self
                                .delta_states
                                .get(&key)
                                .map(|state| state.buffered_arguments.join(""))
                                .unwrap_or_default();
                            let tool_call =
                                self.name_delta_diagnostic_tool_call(&key, name, &buffered_args);
                            return Ok(self.surface_invalid_call(
                                tool_call,
                                block_id.clone(),
                                Some(buffered_args),
                                PendingInvalid::NameDelta {
                                    block_id: block_id.clone(),
                                },
                                InvalidToolCallReason::UnknownTool,
                            ));
                        }

                        Ok(self.validate_delta_name(key, name.clone()))
                    }
                    Delta::ToolArguments { arguments } => {
                        let state = self.delta_states.entry(key).or_default();
                        if state.name_validated {
                            Ok(vec![StreamedTurnEvent::EmitToolCallDelta {
                                block_id: block_id.clone(),
                                delta: Delta::ToolArguments {
                                    arguments: arguments.clone(),
                                },
                            }])
                        } else {
                            state.buffered_arguments.push(arguments.clone());
                            Ok(Vec::new())
                        }
                    }
                    Delta::Text { .. } | Delta::TextMeta { .. } | Delta::Reasoning { .. } => {
                        Ok(vec![StreamedTurnEvent::EmitIngested])
                    }
                }
            }
            StreamEvent::Final(final_response) => {
                if let Some(err) = self.pending_delta_error() {
                    return Err(err);
                }

                let usage = final_response.usage;
                let emit_final = self.saw_text;
                self.saw_text = false;
                // `StreamingCompletionResponse` has already reconciled this
                // against the tool calls the accumulator actually saw (see
                // `FinishReason::reconcile_with_output`), so it is consumed
                // as-is and never re-reconciled here.
                let finish_reason = final_response.finish_reason.clone();
                self.finish_reason.clone_from(&finish_reason);
                Ok(vec![StreamedTurnEvent::Completed {
                    usage,
                    emit_final,
                    finish_reason,
                    raw: final_response.raw.clone(),
                }])
            }
            StreamEvent::Unknown(payload) => {
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
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name,
            } => {
                let live = name
                    .as_ref()
                    .is_some_and(|name| self.policy.allows(name.as_str()));
                self.open_calls.insert(
                    part.index(),
                    OpenCall {
                        name: name.clone(),
                        live,
                        arguments: String::new(),
                    },
                );
                Ok(vec![if live {
                    StreamedTurnEvent::EmitIngested
                } else {
                    StreamedTurnEvent::HoldToolCall
                }])
            }
            StreamEvent::Arguments { part, json } => {
                let call = self.open_calls.entry(part.index()).or_insert(OpenCall {
                    name: None,
                    live: false,
                    arguments: String::new(),
                });
                call.arguments.push_str(json);
                Ok(vec![if call.live {
                    StreamedTurnEvent::EmitToolCallDelta
                } else {
                    StreamedTurnEvent::HoldToolCall
                }])
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(tool_call),
                reasoning_authority: _,
            } => {
                self.open_calls.remove(&part.index());
                if !self.policy.allows(tool_call.function.name.as_str()) {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        Some(tool_call.function.arguments_value().to_string()),
                    ));
                }
                self.pending_tool_calls.push(tool_call.clone());
                Ok(vec![StreamedTurnEvent::EmitToolCall {
                    call: tool_call.clone(),
                }])
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
        let Some(pending) = self.pending_invalid.take() else {
            return Vec::new();
        };

        let PendingInvalid { mut tool_call } = pending;
        match resolution {
            StreamedResolution::Repaired { tool_name } => {
                if let Ok(tool_name) = ToolName::new(tool_name.clone()) {
                    self.repaired_calls
                        .push((tool_call.id.clone(), tool_name.clone()));
                    tool_call.function.name = tool_name;
                }
                self.pending_tool_calls.push(tool_call.clone());
                vec![StreamedTurnEvent::EmitToolCall { call: tool_call }]
            }
            StreamedResolution::TurnAbandoned { .. } => Vec::new(),
            StreamedResolution::Ignored => {
                self.ignored_calls.push(tool_call.id);
                Vec::new()
            }
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
                AssistantContent::ToolCall(call) => !ignored.contains(&call.id),
                // Refused at its block end; a finished turn never holds one.
                AssistantContent::CustomToolCall(_)
                | AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::ProviderItem(_) => true,
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

    /// The refusal event for a completed call name-keyed dispatch cannot run,
    /// or `None` for a dispatchable call or non-call content.
    fn refuse_undispatchable(block: &AssistantContent) -> Option<StreamedTurnEvent> {
        rig_core::message::name_dispatchable_call(block)
            .err()
            .map(|call| {
                StreamedTurnEvent::UndispatchableToolCall(RefusedStreamedCall {
                    call,
                    content: block.clone(),
                })
            })
    }

    /// Park resolution on `pending` and surface the rejected call to the
    /// caller as an [`StreamedTurnEvent::InvalidToolCall`].
    fn surface_invalid_call(
        &mut self,
        tool_call: ToolCall,
        args: Option<String>,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            tool_call: tool_call.clone(),
            args,
            policy: self.policy.clone(),
        };
        self.pending_invalid = Some(PendingInvalid { tool_call });
        vec![StreamedTurnEvent::InvalidToolCall(invalid)]
    }

    /// Surface malformed completed arguments for resolution. Retains raw argument
    /// text and call identity, uses `Null` parsed arguments, and clears pending
    /// delta bookkeeping because the provider accumulator already closed the block.
    pub fn surface_malformed_input(
        &mut self,
        detail: &rig_core::error::MalformedToolInput,
    ) -> Vec<StreamedTurnEvent> {
        let tool_call = ToolCall {
            id: detail.id.clone(),
            provider: detail.provider.clone(),
            function: rig_core::message::ToolFunction::new(
                detail.name.clone(),
                serde_json::Value::Null,
            ),
            signature: None,
            additional_params: None,
        };
        let block_id = detail
            .provider
            .as_ref()
            .map(|provider| BlockId::wire(provider.call_id.as_str()))
            .or_else(|| detail.id.generated().cloned())
            .unwrap_or_else(|| BlockId::wire(detail.id.wire_hint().as_ref()));
        // The accumulator's block is closed; any delta bookkeeping this
        // assembler kept for it must not trip the pending-delta check.
        self.delta_states.remove(&block_id);
        self.surface_invalid_call(
            tool_call.clone(),
            block_id,
            Some(detail.raw.clone()),
            PendingInvalid::MalformedArgs { tool_call },
            InvalidToolCallReason::MalformedArguments {
                error: detail.error.clone(),
            },
        )
    }

    fn name_delta_diagnostic_tool_call(
        &self,
        key: &BlockId,
        name: &str,
        buffered_args: &str,
    ) -> ToolCall {
        let diagnostic_args = if buffered_args.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(buffered_args).unwrap_or(serde_json::Value::Null)
        };
        // Provider identity is not known yet; use the block's deterministic ID
        // to correlate diagnostics and rollback results without inventing a provider ID.
        ToolCall::new(
            rig_core::message::ToolCallId::from_block(key),
            ToolFunction::new(name.to_string(), diagnostic_args),
        )
    }

    fn validate_delta_name(&mut self, key: BlockId, name: String) -> Vec<StreamedTurnEvent> {
        let state = self.delta_states.entry(key.clone()).or_default();
        state.name_validated = true;
        let buffered_arguments = std::mem::take(&mut state.buffered_arguments);

        let mut events = vec![StreamedTurnEvent::EmitToolCallDelta {
            block_id: key.clone(),
            delta: Delta::ToolName { name },
        }];
        events.extend(buffered_arguments.into_iter().map(|arguments| {
            StreamedTurnEvent::EmitToolCallDelta {
                block_id: key.clone(),
                delta: Delta::ToolArguments { arguments },
            }
        }));
        events
    }
}

#[cfg(test)]
mod tests;
