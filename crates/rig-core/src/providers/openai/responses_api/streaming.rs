//! Responses frame classification, event decoding, and terminal metadata.
//!
//! ```
//! use rig_core::providers::openai::responses_api::streaming::{ResponsesDecoder, ResponsesStreamOptions};
//! let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
//! ```

use crate::error::ProviderError;
use crate::operation::AdapterOutput;
use crate::operation::Completion;
use crate::providers::internal::wire::{self, WireEvent};
use crate::providers::openai::responses_api::{
    ClassifiedArguments, FunctionCallArguments, IncompleteDetailsReason, ReasoningSummary,
    ReasoningTextContent, ResponseStatus, ResponsesUsage, ToolStatus,
};
use crate::streaming::{
    BlockId, CustomToolCallEnd, StreamFinal, ToolCallEnd, UnparseableToolInput,
};
use crate::wire::Decoder;
use crate::wire::WireFrame;
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, Output};

/// Response lifecycle event or output-item event.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    Response(ResponseChunk),
    Delta(ItemChunk),
}

/// Provider terminal metadata serialized into [`StreamFinal::raw`].
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Token usage from the terminal response event; `None` when the event
    /// carried no `usage` object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    /// The complete object-shaped reasoning metadata from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// The effective reasoning context from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_context: Option<String>,
    /// The `status` reported by the terminal `response.completed` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ResponseStatus>,
    /// Why the response stopped short, when the provider said so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetailsReason>,
    /// The assistant message ID (`msg_...`) carried by the terminal response's
    /// output items.
    ///
    /// Distinct from [`Self::response_id`] (`resp_...`), which names the whole
    /// response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The response ID (`resp_...`) reported by the terminal
    /// `response.completed` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The model identifier reported by the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Transport request ID, if supplied by the caller.
    /// The driver stamps connection headers onto the normalized final record instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl StreamingCompletionResponse {
    /// Create a terminal record carrying only usage; the remaining metadata is
    /// filled in from the terminal `response.completed` event as it arrives.
    pub fn new(usage: Option<ResponsesUsage>) -> Self {
        Self {
            usage,
            provider_request_id: None,
            reasoning_metadata: None,
            reasoning_context: None,
            status: None,
            incomplete_details: None,
            message_id: None,
            response_id: None,
            model: None,
        }
    }
}

/// Normalize the Responses API's terminal stream record.
///
/// The provider descriptor name is an input for the same reason it is on the
/// unary conversion: ChatGPT and Copilot stream this exact wire shape, so a
/// baked-in `"openai"` would mislabel them.
///
/// The finish reason is left exactly as the provider reported it;
/// [`crate::streaming::StreamingCompletionResponse`] applies the tool-call
/// reconciliation afterwards, using the calls the stream actually emitted.
///
/// The native record is serialized onto [`StreamFinal::raw`]; a
/// serialization failure is the caller's to surface as an in-band error.
fn terminal_record(
    provider: &str,
    upstream_reasoning_issuer: bool,
    response: StreamingCompletionResponse,
) -> Result<StreamFinal, ProviderError> {
    let raw = serde_json::to_value(&response)?;
    let issuer = upstream_reasoning_issuer
        .then_some(response.model.as_deref())
        .flatten()
        .map(|model| crate::providers::openai::wire::upstream_reasoning_issuer(provider, model));
    let finish_reason = response
        .status
        .as_ref()
        .and_then(|status| super::map_finish_reason(status, response.incomplete_details.as_ref()));

    let terminal = StreamFinal::new(provider, crate::completion::Usage::from(&response), raw)
        .with_optional_finish_reason(finish_reason)
        .with_optional_message_id(response.message_id)
        .with_optional_response_id(response.response_id)
        .with_optional_provider_request_id(response.provider_request_id)
        .with_optional_model(response.model);
    Ok(match issuer {
        Some(issuer) => terminal.with_reasoning_issuer(issuer),
        None => terminal,
    })
}

/// Combine summaries, content, and encrypted data into one reasoning restatement.
/// Preserve `provider_id` and wire field order. Return `None` for empty content;
/// the caller must close the existing block with the returned restatement.
pub(crate) fn reasoning_from_done_item(
    provider_id: Option<&str>,
    summary: Vec<ReasoningSummary>,
    content: Vec<ReasoningTextContent>,
    encrypted_content: Option<String>,
    signature: Option<String>,
) -> Option<crate::message::Reasoning> {
    // Same builder as the unary decode, so the restatement and the
    // non-streaming conversion of one item cannot drift.
    let blocks = super::reasoning_content_blocks(summary, content, encrypted_content, signature);

    if blocks.is_empty() {
        return None;
    }

    Some(crate::message::Reasoning {
        provider: None,
        id: provider_id.map(str::to_owned),
        content: blocks,
    })
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(response: &StreamingCompletionResponse) -> Self {
        response.usage.as_ref().map(Self::from).unwrap_or_default()
    }
}

/// A response chunk from OpenAI's response API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseChunk {
    /// The response chunk type
    #[serde(rename = "type")]
    pub kind: ResponseChunkKind,
    /// The response itself
    pub response: CompletionResponse,
    /// The item sequence
    pub sequence_number: u64,
}

/// Response chunk type.
/// Renames are used to ensure that this type gets (de)serialized properly.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ResponseChunkKind {
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    #[serde(rename = "response.failed")]
    ResponseFailed,
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete,
}

/// Whether `kind` is a Responses SSE event type this client models.
///
/// The union of [`ResponseChunkKind`]'s and [`ItemChunkKind`]'s wire names: a
/// frame carrying one of these that still fails to deserialize is a data-level
/// defect in a known event, not an unknown event type, and must surface as an
/// error rather than be skipped.
fn is_known_responses_event_type(kind: &str) -> bool {
    matches!(
        kind,
        "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.delta"
            | "response.output_text.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done"
    )
}

/// Classify a tagged Responses event with strict decoding for known event types.
/// Callers must handle `error` and WebSocket `response.done` separately.
#[doc(hidden)]
pub fn classify_responses_frame(data: &str) -> WireEvent<StreamingCompletionChunk> {
    wire::classify_tagged_frame(data, "type", is_known_responses_event_type)
}

/// How a decoder takes a terminal `response.incomplete` event (for example a
/// `max_output_tokens` truncation).
///
/// The contract is transport-specific: an HTTP SSE stream refuses it by
/// default and accepts it only when the caller opts in
/// ([`super::wire::Responses::with_streamed_incomplete`]); a unary reply and a
/// websocket session accept it. Either way incomplete is never reported as
/// completed: an accepted one ends the turn with its truthful finish reason
/// (max-output → `Length`, content-filter → `ContentFilter`, else `Other`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteTerminal {
    /// End the reply with an error carrying the provider's terminal event.
    /// The HTTP SSE default.
    #[default]
    Refuse,
    /// Keep the partial output and usage and end the turn with the finish
    /// reason the provider's `incomplete_details` states.
    Accept,
}

/// The decoder's two independent policy axes: when a completed tool call is
/// published, and how a terminal `response.incomplete` is taken.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct ResponsesStreamOptions {
    /// Publish a call the wire delivered whole at its `output_item.done`
    /// rather than buffering it to the terminal.
    immediate_tool_calls: bool,
    /// How a terminal `response.incomplete` is taken.
    incomplete: IncompleteTerminal,
}

impl ResponsesStreamOptions {
    /// Buffered tool calls; a terminal `response.incomplete` is refused.
    #[doc(hidden)]
    pub const fn strict() -> Self {
        Self {
            immediate_tool_calls: false,
            incomplete: IncompleteTerminal::Refuse,
        }
    }

    pub(crate) const fn strict_with_immediate_tool_calls() -> Self {
        Self {
            immediate_tool_calls: true,
            incomplete: IncompleteTerminal::Refuse,
        }
    }

    /// The same options with `incomplete` selecting how a terminal
    /// `response.incomplete` is taken.
    #[doc(hidden)]
    pub const fn with_incomplete(mut self, incomplete: IncompleteTerminal) -> Self {
        self.incomplete = incomplete;
        self
    }

    const fn emits_completed_tool_calls_immediately(self) -> bool {
        self.immediate_tool_calls
    }

    const fn refuses_incomplete(self) -> bool {
        matches!(self.incomplete, IncompleteTerminal::Refuse)
    }
}

/// A tool call the wire delivered whole, buffered to the terminal.
enum BufferedCall {
    /// A function call, finalized by the shared accumulator.
    Function(ToolCallEnd),
    /// A custom call: complete as delivered, with verbatim input.
    Custom(CustomToolCallEnd),
}

#[doc(hidden)]
pub struct RawChoiceAccumulator {
    /// Stable descriptor name stamped on the terminal record: ChatGPT and
    /// Copilot stream this exact wire shape, so it is an input rather than
    /// a baked-in `"openai"`.
    provider: String,
    /// The terminal record under assembly: what the terminal event says
    /// about the turn, filled in as the stream reports it.
    terminal: StreamingCompletionResponse,
    /// Buffered tool-call ends for calls delivered whole by
    /// `output_item.done`, flushed at the terminal (or before a terminal
    /// error) as `BlockEnd`s keyed by the slot's assembly id. Assembly and
    /// internal-id correlation live in the shared accumulator, keyed by the
    /// function-call item id the added/delta/done events share.
    tool_calls: Vec<(BlockId, BufferedCall)>,
    /// Whether a genuine terminal event (`response.completed` or
    /// `response.incomplete`) arrived. Without one the stream was truncated,
    /// and `finish` withholds the terminal record.
    saw_terminal: bool,
    /// Reasoning assembly keys fixed by the first event in each output slot.
    /// Later wire IDs do not change an established key.
    reasoning_slots: std::collections::HashMap<u64, crate::streaming::BlockId>,
    /// Tool-call assembly keys fixed per output slot, sharing a mint counter
    /// with reasoning so id-less blocks cannot collide.
    tool_slots: crate::providers::internal::tool_call_bridge::ToolCallBridge<u64>,
    /// The `call_…` correlator each open slot announced on
    /// `output_item.added`, kept beside the bridge so a slot closed by the
    /// terminal drain (its `output_item.done` frame was lost) still
    /// finalizes with the dual-wire identity Responses replay pairs on.
    pending_call_ids: std::collections::HashMap<u64, String>,
    /// The namespace each open slot announced on `output_item.added`, kept
    /// for the same reason as [`Self::pending_call_ids`]: a slot the terminal
    /// drain closes must not lose its qualifier.
    pending_namespaces: std::collections::HashMap<u64, String>,
    /// The argument fragments each open function-call slot streamed, assembled
    /// exactly as the shared accumulator assembles its buffer for that slot.
    /// The done item's verdict reads it to know whether that buffer would
    /// answer for a restatement that does not parse.
    argument_fragments: std::collections::HashMap<u64, AssembledArguments>,
    message_parts: std::collections::BTreeMap<(u64, u64), MessagePart>,
    active_message_part: Option<BlockId>,
    reasoning_parts: std::collections::BTreeMap<u64, ReasoningParts>,
    finished_reasoning: std::collections::HashSet<u64>,
    part_ids: crate::streaming::SyntheticIds,
    /// Whether reasoning belongs to the upstream model's family rather than
    /// to `provider`, a gateway ([`crate::providers::openai::wire::upstream_reasoning_issuer`]).
    upstream_reasoning_issuer: bool,
    /// The parent item kind each output slot's typed evidence established.
    /// An unknown content part carries no parent kind of its own.
    parent_kinds: std::collections::HashMap<u64, PartParent>,
    /// Unknown content parts seen before their slot's parent kind, kept raw per
    /// `(output_index, content_index)` in arrival order. Identical restatements
    /// collapse; distinct observations are all kept.
    undecided_parts: std::collections::BTreeMap<(u64, u64), Vec<serde_json::Value>>,
    /// A protocol refusal raised mid-event, taken by the decoder, which ends the
    /// reply after the usual pre-error flush.
    refusal: Option<ProviderError>,
    /// Message slots whose text a delta delivered. Their item and terminal
    /// snapshots carry no phase to the stream: that keeps the recorded
    /// replay requests of streamed turns byte-identical. Unary and
    /// terminal-only messages keep their phase.
    delta_message_slots: std::collections::HashSet<u64>,
    /// The output slot each message id owns: the position where the id was
    /// first seen, or, for a completed snapshot of an unseen id, the slot its
    /// id-less deltas filled. A frame whose envelope lost its `output_index`
    /// still reaches the message it belongs to.
    message_slot_by_id: std::collections::HashMap<String, u64>,
    /// Slots whose text arrived by deltas carrying no item id, not yet owned
    /// by any message id.
    unattributed_message_slots: std::collections::BTreeSet<u64>,
}

/// The item kind that owns a slot's content parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartParent {
    Message,
    Reasoning,
}

struct MessagePart {
    key: BlockId,
    text: String,
    metadata: Option<crate::message::AdditionalParams>,
    phase: Option<String>,
    refusal_marked: bool,
}

impl MessagePart {
    fn is_opaque(&self) -> bool {
        super::opaque_message_part(self.metadata.as_ref()).is_some()
    }
}

/// The kind name a content part has for [`conflicting_part_kind`].
fn part_kind_name(opaque: bool) -> &'static str {
    if opaque { "opaque" } else { "text" }
}

/// One content slot stated first as one kind and then as the other. No
/// replacement or merge policy exists between an opaque part and known text.
fn conflicting_part_kind(
    output_index: u64,
    array: &str,
    index: u64,
    stored_opaque: bool,
    incoming_opaque: bool,
) -> ProviderError {
    ProviderError::Response(format!(
        "conflicting Responses content part kind at output {output_index}, {array} \
         {index}: stored {} part restated as {}",
        part_kind_name(stored_opaque),
        part_kind_name(incoming_opaque),
    ))
}

/// A reasoning part stated inside an item already evidenced as a message.
fn conflicting_parent_kind(output_index: u64, content_index: u64) -> ProviderError {
    ProviderError::Response(format!(
        "conflicting Responses parent kind at output {output_index}, content \
         {content_index}: a reasoning_text part in a message"
    ))
}

/// Which reasoning array a part belongs to.
#[derive(Clone, Copy)]
enum ReasoningArray {
    Summary,
    Content,
}

impl ReasoningArray {
    fn name(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Content => "content",
        }
    }
}

/// Full output-text snapshots use citation suffixes when possible. A changed
/// object or a replaced array needs an owned-key reset before the replacement.
fn message_snapshot_metadata(
    previous: Option<&crate::message::AdditionalParams>,
    current: Option<&crate::message::AdditionalParams>,
    phase: Option<&str>,
    out: &mut AdapterOutput,
) {
    use super::OPENAI_RESPONSES_EXTRAS_KEY as KEY;
    use serde_json::{Map, Value};
    let old = previous.and_then(|params| params.wire_extras(KEY));
    let new = current.and_then(|params| params.wire_extras(KEY));
    if old != new {
        let empty = Map::new();
        let old = old.unwrap_or(&empty);
        let new = new.unwrap_or(&empty);
        let mut replacement = old.keys().any(|key| !new.contains_key(key));
        let mut delta = Map::new();
        for (key, value) in new {
            match old.get(key) {
                Some(prior) if prior == value => {}
                Some(Value::Array(prior)) => match value {
                    Value::Array(next) => match next.strip_prefix(prior.as_slice()) {
                        Some(suffix) => {
                            delta.insert(key.clone(), Value::Array(suffix.to_vec()));
                        }
                        None => replacement = true,
                    },
                    _ => replacement = true,
                },
                Some(Value::Object(_)) => replacement = true,
                _ => {
                    delta.insert(key.clone(), value.clone());
                }
            }
        }
        if replacement {
            if let Some(reset) =
                crate::message::AdditionalParams::from_entries(Some((KEY, Value::Null)))
            {
                out.text_meta(reset);
            }
            delta = new.clone();
            if let Some(phase) = phase {
                delta.insert(
                    super::OPENAI_RESPONSES_PHASE_KEY.into(),
                    Value::String(phase.into()),
                );
            }
        }
        if (replacement || !delta.is_empty())
            && let Some(params) =
                crate::message::AdditionalParams::from_entries(Some((KEY, Value::Object(delta))))
        {
            out.text_meta(params);
        }
    }
    // Opaque and refusal markers occupy their own key, outside the extras map.
    let old_marker = previous.and_then(|params| params.get(super::OPENAI_RESPONSES_PART_KEY));
    let marker = current.and_then(|params| params.get(super::OPENAI_RESPONSES_PART_KEY));
    if old_marker != marker
        && let Some(marker) = marker
        && let Some(params) = crate::message::AdditionalParams::from_entries(Some((
            super::OPENAI_RESPONSES_PART_KEY,
            marker.clone(),
        )))
    {
        out.text_meta(params);
    }
}

#[derive(Default)]
struct ReasoningParts {
    id: Option<String>,
    summary: std::collections::BTreeMap<u64, ReasoningSummary>,
    content: std::collections::BTreeMap<u64, ReasoningTextContent>,
    encrypted: Option<String>,
    signature: Option<String>,
    // Item/whole-body completion supplies an authoritative end even when its
    // publication waits for the last snapshot of the turn.
    wire_sent: bool,
    ready: bool,
}

impl ReasoningParts {
    fn has_opaque(&self) -> bool {
        self.summary
            .values()
            .any(|part| matches!(part, ReasoningSummary::Unknown(_)))
            || self
                .content
                .values()
                .any(|part| matches!(part, ReasoningTextContent::Unknown(_)))
    }

    fn absorb(
        &mut self,
        summary: Vec<ReasoningSummary>,
        content: Vec<ReasoningTextContent>,
        encrypted: Option<String>,
        signature: Option<String>,
    ) {
        // An empty string is no ciphertext, so it never claims authority.
        let encrypted = encrypted.filter(|value| !value.is_empty());
        let signature = signature.filter(|value| !value.is_empty());
        // Empty restatements never erase already delivered content.
        self.summary.extend(
            summary
                .into_iter()
                .enumerate()
                .map(|(i, part)| (i as u64, part)),
        );
        self.content.extend(
            content
                .into_iter()
                .enumerate()
                .map(|(i, part)| (i as u64, part)),
        );
        if encrypted.is_some() && (!self.wire_sent || self.encrypted.is_none()) {
            self.encrypted = encrypted;
        }
        if signature.is_some() && (!self.wire_sent || self.signature.is_none()) {
            self.signature = signature;
        }
    }

    fn into_reasoning(self) -> crate::message::Reasoning {
        reasoning_from_done_item(
            self.id.as_deref(),
            self.summary.into_values().collect(),
            self.content.into_values().collect(),
            self.encrypted,
            self.signature,
        )
        .unwrap_or_else(|| crate::message::Reasoning {
            id: self.id,
            provider: None,
            content: Vec::new(),
        })
    }
}

/// The assistant message ID (`msg_...`) a terminal response object carries,
/// which is deliberately not the response's own `resp_...` id.
fn message_id_from_response(response: &CompletionResponse) -> Option<String> {
    response.output.iter().find_map(|item| match item {
        Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
}

impl RawChoiceAccumulator {
    /// `initial_usage` seeds the terminal's usage for replayed bodies whose
    /// SSE frames may not carry one (the unary Responses body's own `usage`).
    #[doc(hidden)]
    pub fn new(provider: impl Into<String>, initial_usage: Option<ResponsesUsage>) -> Self {
        Self {
            provider: provider.into(),
            terminal: StreamingCompletionResponse::new(initial_usage),
            tool_calls: Vec::new(),
            saw_terminal: false,
            reasoning_slots: std::collections::HashMap::new(),
            tool_slots:
                crate::providers::internal::tool_call_bridge::ToolCallBridge::with_minted_namespace(
                    crate::streaming::SyntheticIds::output(),
                ),
            pending_call_ids: std::collections::HashMap::new(),
            pending_namespaces: std::collections::HashMap::new(),
            argument_fragments: std::collections::HashMap::new(),
            message_parts: Default::default(),
            active_message_part: None,
            reasoning_parts: Default::default(),
            finished_reasoning: Default::default(),
            part_ids: crate::streaming::SyntheticIds::new(crate::streaming::MintKind::Refusal),
            upstream_reasoning_issuer: false,
            parent_kinds: Default::default(),
            undecided_parts: Default::default(),
            refusal: None,
            delta_message_slots: Default::default(),
            message_slot_by_id: Default::default(),
            unattributed_message_slots: Default::default(),
        }
    }

    /// Take the unknown parts buffered for `output_index`, in content order.
    fn take_undecided(&mut self, output_index: u64) -> Vec<(u64, serde_json::Value)> {
        let keys: Vec<_> = self
            .undecided_parts
            .range((output_index, 0)..=(output_index, u64::MAX))
            .map(|(key, _)| *key)
            .collect();
        keys.into_iter()
            .filter_map(|key| {
                self.undecided_parts
                    .remove(&key)
                    .map(|values| (key.1, values))
            })
            .flat_map(|(content_index, values)| {
                values.into_iter().map(move |value| (content_index, value))
            })
            .collect()
    }

    /// The slot a message event addresses. An id keeps the slot it first
    /// owned. An unseen id owns its own position, except that a completed
    /// snapshot (`done` or terminal) of an unseen id whose own slot holds no
    /// message content takes over a slot that id-less deltas filled: that is
    /// the same message restated. An `output_item.added` never aliases.
    fn message_slot(&mut self, output_index: u64, item_id: Option<&str>, snapshot: bool) -> u64 {
        let Some(id) = item_id.filter(|id| !id.is_empty()) else {
            return output_index;
        };
        if let Some(slot) = self.message_slot_by_id.get(id) {
            return *slot;
        }
        let own_slot_empty = self
            .message_parts
            .range((output_index, 0)..=(output_index, u64::MAX))
            .next()
            .is_none();
        let slot = if snapshot && own_slot_empty {
            self.unattributed_message_slots
                .first()
                .copied()
                .unwrap_or(output_index)
        } else {
            output_index
        };
        self.unattributed_message_slots.remove(&slot);
        self.message_slot_by_id.insert(id.to_owned(), slot);
        slot
    }

    /// The slot an id-bearing event refers to, without claiming one: an unknown
    /// part may belong to reasoning, so only a known message id redirects it.
    fn known_message_slot(&self, output_index: u64, item_id: Option<&str>) -> u64 {
        item_id
            .and_then(|id| self.message_slot_by_id.get(id))
            .copied()
            .unwrap_or(output_index)
    }

    /// The stored kind a message part would contradict: a published part, or
    /// an unknown part still waiting for its parent (always opaque).
    fn message_kind_conflict(
        &self,
        slot: u64,
        content_index: u64,
        incoming_opaque: bool,
    ) -> Option<ProviderError> {
        let stored_opaque = self
            .message_parts
            .get(&(slot, content_index))
            .map(MessagePart::is_opaque)
            .or_else(|| {
                self.undecided_parts
                    .contains_key(&(slot, content_index))
                    .then_some(true)
            })?;
        (stored_opaque != incoming_opaque).then(|| {
            conflicting_part_kind(
                slot,
                "content",
                content_index,
                stored_opaque,
                incoming_opaque,
            )
        })
    }

    /// The stored kind a reasoning part would contradict, counting unknown
    /// parts still waiting for their parent as stored content.
    fn reasoning_kind_conflict(
        &self,
        output_index: u64,
        array: ReasoningArray,
        index: u64,
        incoming_opaque: bool,
    ) -> Option<ProviderError> {
        let parts = self.reasoning_parts.get(&output_index);
        let stored_opaque = match array {
            ReasoningArray::Summary => parts
                .and_then(|parts| parts.summary.get(&index))
                .map(|part| matches!(part, ReasoningSummary::Unknown(_))),
            ReasoningArray::Content => parts
                .and_then(|parts| parts.content.get(&index))
                .map(|part| matches!(part, ReasoningTextContent::Unknown(_)))
                .or_else(|| {
                    self.undecided_parts
                        .contains_key(&(output_index, index))
                        .then_some(true)
                }),
        }?;
        (stored_opaque != incoming_opaque).then(|| {
            conflicting_part_kind(
                output_index,
                array.name(),
                index,
                stored_opaque,
                incoming_opaque,
            )
        })
    }

    /// Refuse a reasoning snapshot that restates any stored part as the
    /// other kind, before any of it is absorbed.
    fn reasoning_snapshot_conflict(
        &self,
        output_index: u64,
        summary: &[ReasoningSummary],
        content: &[ReasoningTextContent],
    ) -> Option<ProviderError> {
        summary
            .iter()
            .enumerate()
            .find_map(|(index, part)| {
                self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Summary,
                    index as u64,
                    matches!(part, ReasoningSummary::Unknown(_)),
                )
            })
            .or_else(|| {
                content.iter().enumerate().find_map(|(index, part)| {
                    self.reasoning_kind_conflict(
                        output_index,
                        ReasoningArray::Content,
                        index as u64,
                        matches!(part, ReasoningTextContent::Unknown(_)),
                    )
                })
            })
    }

    /// Record message evidence for the slot and publish any unknown parts that
    /// were waiting for it through ordinary message assembly.
    fn bind_message(&mut self, output_index: u64, item_id: Option<&str>, out: &mut AdapterOutput) {
        self.parent_kinds
            .entry(output_index)
            .or_insert(PartParent::Message);
        for (content_index, value) in self.take_undecided(output_index) {
            self.publish_message_part(
                output_index,
                content_index,
                item_id,
                super::AssistantContent::Unknown(value),
                None,
                out,
            );
        }
    }

    /// Unknown parts whose slot never showed its parent kind leave raw on the
    /// passthrough channel: without that evidence they have no assistant
    /// content to join.
    fn flush_undecided_parts(&mut self, out: &mut AdapterOutput) {
        for value in std::mem::take(&mut self.undecided_parts)
            .into_values()
            .flatten()
        {
            out.unknown(value.into());
        }
    }

    fn message_part(
        &mut self,
        output_index: u64,
        content_index: u64,
        item_id: Option<&str>,
        refusal: bool,
    ) -> &mut MessagePart {
        self.message_parts
            .entry((output_index, content_index))
            .or_insert_with(|| MessagePart {
                key: if content_index == 0 && !refusal {
                    item_id
                        .filter(|id| !id.is_empty())
                        .map_or_else(|| self.part_ids.mint(), |id| BlockId::wire(id.to_owned()))
                } else {
                    self.part_ids.mint()
                },
                text: String::new(),
                metadata: None,
                phase: None,
                refusal_marked: false,
            })
    }

    fn message_delta(
        &mut self,
        output_index: u64,
        content_index: u64,
        item_id: Option<&str>,
        delta: String,
        refusal: bool,
        out: &mut AdapterOutput,
    ) {
        if item_id.is_none_or(str::is_empty)
            && !self
                .message_slot_by_id
                .values()
                .any(|slot| *slot == output_index)
        {
            self.unattributed_message_slots.insert(output_index);
        }
        let output_index = self.message_slot(output_index, item_id, false);
        // Refuse before any state changes, as a snapshot would: known text
        // cannot join a published or waiting opaque part.
        if let Some(error) = self.message_kind_conflict(output_index, content_index, false) {
            self.refusal = Some(error);
            return;
        }
        self.bind_message(output_index, item_id, out);
        self.delta_message_slots.insert(output_index);
        let part = self.message_part(output_index, content_index, item_id, refusal);
        let marker = (refusal && !part.refusal_marked)
            .then(super::refusal_marker)
            .flatten();
        part.refusal_marked |= refusal;
        // The marker a delta delivered is the part's metadata, so a later
        // snapshot of the same refusal does not restate it.
        if part.metadata.is_none() {
            part.metadata = marker.clone();
        }
        part.text.push_str(&delta);
        let key = part.key.clone();
        if self.active_message_part.as_ref() != Some(&key) || marker.is_some() {
            out.text_start(key.clone(), marker);
            self.active_message_part = Some(key);
        }
        out.text(delta);
    }

    fn publish_message_part(
        &mut self,
        output_index: u64,
        content_index: u64,
        item_id: Option<&str>,
        content: super::AssistantContent,
        phase: Option<&str>,
        out: &mut AdapterOutput,
    ) {
        let refusal = matches!(content, super::AssistantContent::Refusal { .. });
        let incoming_opaque = matches!(content, super::AssistantContent::Unknown(_));
        // Refuse before any text or marker changes: nothing may replace an
        // opaque part with text, or text with an opaque part.
        if let Some(stored_opaque) = self
            .message_parts
            .get(&(output_index, content_index))
            .map(MessagePart::is_opaque)
            .filter(|stored_opaque| *stored_opaque != incoming_opaque)
        {
            self.refusal = Some(conflicting_part_kind(
                output_index,
                "content",
                content_index,
                stored_opaque,
                incoming_opaque,
            ));
            return;
        }
        let text = super::text_block(content);
        let existed = self
            .message_parts
            .contains_key(&(output_index, content_index));
        let part = self.message_part(output_index, content_index, item_id, refusal);
        let has_new_text = text
            .text
            .strip_prefix(&part.text)
            .is_some_and(|suffix| !suffix.is_empty());
        let has_new_phase = phase.is_some_and(|phase| part.phase.as_deref() != Some(phase));
        if existed && !has_new_text && !has_new_phase && part.metadata == text.additional_params {
            return;
        }
        let key = part.key.clone();
        // A refusal block carries its marker from its start, as its deltas'
        // block would.
        let start_marker = (refusal && !part.refusal_marked)
            .then(super::refusal_marker)
            .flatten();
        if self.active_message_part.as_ref() != Some(&key) {
            out.text_start(key.clone(), start_marker.clone());
            self.active_message_part = Some(key);
        } else if let Some(marker) = start_marker.clone() {
            out.text_meta(marker);
        }
        let part = self.message_part(output_index, content_index, item_id, refusal);
        part.refusal_marked |= refusal;
        if start_marker.is_some() && part.metadata.is_none() {
            part.metadata = start_marker;
        }
        // A restatement may complete a prefix, but never repeats delivered text.
        if let Some(suffix) = text.text.strip_prefix(&part.text) {
            if !suffix.is_empty() {
                out.text(suffix.to_owned());
            }
            part.text = text.text;
        }
        if part.metadata != text.additional_params {
            message_snapshot_metadata(
                part.metadata.as_ref(),
                text.additional_params.as_ref(),
                phase.or(part.phase.as_deref()),
                out,
            );
            part.metadata = text.additional_params;
        }
        if let Some(phase) = phase
            && part.phase.as_deref() != Some(phase)
        {
            let mut text = crate::message::Text::new("");
            super::stamp_phase(&mut text, Some(phase));
            if let Some(params) = text.additional_params {
                out.text_meta(params);
            }
            part.phase = Some(phase.to_owned());
        }
    }

    fn publish_message_text(
        &mut self,
        output_index: u64,
        message: &super::OutputMessage,
        snapshot: bool,
        out: &mut AdapterOutput,
    ) {
        let output_index = self.message_slot(output_index, Some(&message.id), snapshot);
        // A contradicted part refuses the whole snapshot before any of it lands.
        if let Some(error) = message
            .content
            .iter()
            .enumerate()
            .find_map(|(index, content)| {
                self.message_kind_conflict(
                    output_index,
                    index as u64,
                    matches!(content, super::AssistantContent::Unknown(_)),
                )
            })
        {
            self.refusal = Some(error);
            return;
        }
        self.bind_message(output_index, Some(&message.id), out);
        // Only a message no delta delivered takes its snapshot phase.
        let phase = message
            .phase
            .as_deref()
            .filter(|_| !self.delta_message_slots.contains(&output_index));
        for (index, content) in message.content.iter().cloned().enumerate() {
            if self.refusal.is_some() {
                return;
            }
            self.publish_message_part(
                output_index,
                index as u64,
                Some(&message.id),
                content,
                phase,
                out,
            );
        }
    }

    fn merge_terminal_body_text(&mut self, response: &CompletionResponse, out: &mut AdapterOutput) {
        for (output_index, item) in response.output.iter().enumerate() {
            if self.refusal.is_some() {
                return;
            }
            match item {
                Output::Message(message) => {
                    self.publish_message_text(output_index as u64, message, true, out)
                }
                Output::Reasoning { .. } => {
                    self.push_output_item_done(item.clone(), output_index as u64, out, true)
                }
                _ => {}
            }
        }
    }

    fn merge_terminal_reasoning(&mut self, response: &CompletionResponse, out: &mut AdapterOutput) {
        for (index, item) in response.output.iter().enumerate() {
            if matches!(item, Output::Reasoning { .. }) {
                self.push_output_item_done(item.clone(), index as u64, out, true);
            }
        }
    }

    fn reasoning_parts(
        &mut self,
        output_index: u64,
        item_id: Option<&str>,
        _out: &mut AdapterOutput,
    ) -> &mut ReasoningParts {
        // Reasoning evidence binds the slot: unknown parts that waited for it
        // are reasoning content.
        self.parent_kinds
            .entry(output_index)
            .or_insert(PartParent::Reasoning);
        let undecided = self.take_undecided(output_index);
        let parts = self.reasoning_parts.entry(output_index).or_default();
        for (content_index, value) in undecided {
            parts
                .content
                .insert(content_index, ReasoningTextContent::Unknown(value));
        }
        // Like its ciphertext and signature, the provider ID the item's
        // completion stated stands; a later snapshot only fills a missing one.
        if let Some(id) = item_id.filter(|id| !id.is_empty())
            && (!parts.wire_sent || parts.id.is_none())
        {
            parts.id = Some(id.to_owned());
        }
        parts
    }

    fn flush_reasoning(&mut self, out: &mut AdapterOutput) {
        let slots: Vec<_> = self
            .reasoning_parts
            .iter()
            .filter(|(index, parts)| {
                !self.finished_reasoning.contains(index) && (parts.ready || parts.has_opaque())
            })
            .map(|(index, _)| *index)
            .collect();
        for index in slots {
            if let Some(parts) = self.reasoning_parts.remove(&index) {
                let key = self.reasoning_slot_key(index, parts.id.as_deref(), out);
                let wire_sent = parts.wire_sent;
                out.reasoning_end(key, Some(parts.into_reasoning()), None, wire_sent);
                self.finished_reasoning.insert(index);
            }
        }
    }

    /// Return the slot's established reasoning key, creating it from `item_id`
    /// or the shared mint counter only on first use.
    fn reasoning_slot_key(
        &mut self,
        output_index: u64,
        item_id: Option<&str>,
        out: &mut AdapterOutput,
    ) -> crate::streaming::BlockId {
        if let Some(key) = self.reasoning_slots.get(&output_index) {
            out.declare_indexed_reasoning(key);
            return key.clone();
        }
        // A shared counter prevents collisions between reasoning and tool assemblies.
        let key = item_id.map_or_else(
            || self.tool_slots.minted_ids().mint(),
            crate::streaming::BlockId::wire,
        );
        out.declare_indexed_reasoning(&key);
        self.reasoning_slots.insert(output_index, key.clone());
        key
    }

    /// Map one item/delta event onto grammar events, pushed to `out`.
    #[doc(hidden)]
    pub fn decode_item_chunk(
        &mut self,
        chunk: ItemChunk,
        options: ResponsesStreamOptions,
        out: &mut AdapterOutput,
    ) {
        let ItemChunk {
            item_id: outer_item_id,
            output_index,
            data: item,
        } = chunk;

        match item {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item: Output::FunctionCall(func),
                ..
            }) => {
                // A function-call item interleaving a message item closes the
                // open text block; forget it so a later delta for that message
                // re-emits its text `BlockStart` and reactivates its block
                // downstream.
                self.active_message_part = None;
                out.end_active_text();
                // Without call_id, mint an assembly key so the item ID cannot become
                // a fabricated tool-result correlator.
                let wire_id = (!func.call_id.is_empty()).then_some(func.id.as_str());
                let key = self
                    .tool_slots
                    .open(output_index, wire_id, Some(&func.name))
                    .key()
                    .to_owned();
                if !func.call_id.is_empty() {
                    self.pending_call_ids
                        .insert(output_index, func.call_id.clone());
                }
                if let Some(namespace) = func.namespace {
                    self.pending_namespaces.insert(output_index, namespace);
                }
                out.tool_name(&key, func.name);
            }
            ItemChunkKind::OutputItemDone(message) => {
                // Refusals have explicit part boundaries. Ordinary same-key
                // text keeps the existing lifecycle across item snapshots.
                if self.active_message_part.as_ref().is_some_and(|key| {
                    self.message_parts
                        .values()
                        .any(|part| &part.key == key && part.refusal_marked)
                }) {
                    out.end_active_text();
                }
                self.active_message_part = None;
                self.push_output_item_done(
                    message.item,
                    output_index,
                    out,
                    options.emits_completed_tool_calls_immediately(),
                );
            }
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item: Output::Message(message),
                ..
            }) => {
                self.publish_message_text(output_index, &message, false, out);
            }
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item:
                    Output::Reasoning {
                        id,
                        summary,
                        content,
                        encrypted_content,
                        signature,
                        ..
                    },
                ..
            }) => {
                if let Some(error) =
                    self.reasoning_snapshot_conflict(output_index, &summary, &content)
                {
                    self.refusal = Some(error);
                    return;
                }
                let parts = self.reasoning_parts(output_index, Some(&id), out);
                parts.absorb(summary, content, encrypted_content, signature);
            }
            ItemChunkKind::OutputTextDelta(DeltaTextChunk {
                delta,
                content_index,
                ..
            }) => self.message_delta(
                output_index,
                content_index,
                outer_item_id.as_deref(),
                delta,
                false,
                out,
            ),
            ItemChunkKind::RefusalDelta(DeltaTextChunk {
                delta,
                content_index,
                ..
            }) => self.message_delta(
                output_index,
                content_index,
                outer_item_id.as_deref(),
                delta,
                true,
                out,
            ),
            ItemChunkKind::ContentPartAdded(chunk) | ItemChunkKind::ContentPartDone(chunk) => {
                match chunk.part {
                    ContentPartChunkPart::SummaryText { text } => {
                        if let Some(error) = self.reasoning_kind_conflict(
                            output_index,
                            ReasoningArray::Summary,
                            chunk.content_index,
                            false,
                        ) {
                            self.refusal = Some(error);
                            return;
                        }
                        self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                            .summary
                            .insert(chunk.content_index, ReasoningSummary::SummaryText { text });
                    }
                    // Known reasoning content: its deltas append to it and its
                    // done event completes it, as with `reasoning_text.done`.
                    ContentPartChunkPart::ReasoningText { text } => {
                        // A documented reasoning part never joins a message.
                        let message_slot =
                            self.known_message_slot(output_index, outer_item_id.as_deref());
                        if [output_index, message_slot]
                            .iter()
                            .any(|slot| self.parent_kinds.get(slot) == Some(&PartParent::Message))
                        {
                            self.refusal =
                                Some(conflicting_parent_kind(output_index, chunk.content_index));
                            return;
                        }
                        if let Some(error) = self.reasoning_kind_conflict(
                            output_index,
                            ReasoningArray::Content,
                            chunk.content_index,
                            false,
                        ) {
                            self.refusal = Some(error);
                            return;
                        }
                        self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                            .content
                            .insert(
                                chunk.content_index,
                                ReasoningTextContent::ReasoningText { text },
                            );
                    }
                    // An unknown part names no parent kind. It joins the slot's
                    // established item, or waits for evidence of one; the item
                    // ID's spelling is not that evidence.
                    ContentPartChunkPart::Unknown(value) => {
                        let output_index =
                            self.known_message_slot(output_index, outer_item_id.as_deref());
                        match self.parent_kinds.get(&output_index) {
                            Some(PartParent::Reasoning) => {
                                if let Some(error) = self.reasoning_kind_conflict(
                                    output_index,
                                    ReasoningArray::Content,
                                    chunk.content_index,
                                    true,
                                ) {
                                    self.refusal = Some(error);
                                    return;
                                }
                                self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                                    .content
                                    .insert(
                                        chunk.content_index,
                                        ReasoningTextContent::Unknown(value),
                                    );
                            }
                            Some(PartParent::Message) => self.publish_message_part(
                                output_index,
                                chunk.content_index,
                                outer_item_id.as_deref(),
                                super::AssistantContent::Unknown(value),
                                None,
                                out,
                            ),
                            None => {
                                let seen = self
                                    .undecided_parts
                                    .entry((output_index, chunk.content_index))
                                    .or_default();
                                if !seen.contains(&value) {
                                    seen.push(value);
                                }
                            }
                        }
                    }
                    part => {
                        let content = match part {
                            ContentPartChunkPart::OutputText { text } => {
                                super::AssistantContent::OutputText(super::OutputText::new(text))
                            }
                            ContentPartChunkPart::Refusal { refusal } => {
                                super::AssistantContent::Refusal { refusal }
                            }
                            ContentPartChunkPart::Unknown(_)
                            | ContentPartChunkPart::SummaryText { .. }
                            | ContentPartChunkPart::ReasoningText { .. } => return,
                        };
                        let output_index =
                            self.message_slot(output_index, outer_item_id.as_deref(), false);
                        if let Some(error) =
                            self.message_kind_conflict(output_index, chunk.content_index, false)
                        {
                            self.refusal = Some(error);
                            return;
                        }
                        self.bind_message(output_index, outer_item_id.as_deref(), out);
                        if self.refusal.is_some() {
                            return;
                        }
                        self.publish_message_part(
                            output_index,
                            chunk.content_index,
                            outer_item_id.as_deref(),
                            content,
                            None,
                            out,
                        );
                    }
                }
            }
            ItemChunkKind::ReasoningSummaryPartAdded(chunk)
            | ItemChunkKind::ReasoningSummaryPartDone(chunk) => {
                if let Some(error) = self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Summary,
                    chunk.summary_index,
                    matches!(chunk.part, ReasoningSummary::Unknown(_)),
                ) {
                    self.refusal = Some(error);
                    return;
                }
                self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                    .summary
                    .insert(chunk.summary_index, chunk.part);
            }
            ItemChunkKind::ReasoningSummaryTextDelta(SummaryTextDeltaChunk {
                delta,
                summary_index,
                ..
            }) => {
                if let Some(error) = self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Summary,
                    summary_index,
                    false,
                ) {
                    self.refusal = Some(error);
                    return;
                }
                self.active_message_part = None;
                let parts = self.reasoning_parts(output_index, outer_item_id.as_deref(), out);
                if let ReasoningSummary::SummaryText { text } = parts
                    .summary
                    .entry(summary_index)
                    .or_insert_with(|| ReasoningSummary::SummaryText {
                        text: String::new(),
                    })
                {
                    text.push_str(&delta);
                }
                let id = self.reasoning_slot_key(output_index, outer_item_id.as_deref(), out);
                out.reasoning_delta(
                    &id,
                    outer_item_id
                        .clone()
                        .and_then(crate::streaming::non_empty_id),
                    delta,
                );
            }
            ItemChunkKind::ReasoningTextDelta(DeltaTextChunkWithItemId {
                delta,
                content_index,
                ..
            }) => {
                if let Some(error) = self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Content,
                    content_index.unwrap_or(0),
                    false,
                ) {
                    self.refusal = Some(error);
                    return;
                }
                self.active_message_part = None;
                let parts = self.reasoning_parts(output_index, outer_item_id.as_deref(), out);
                if let ReasoningTextContent::ReasoningText { text } = parts
                    .content
                    .entry(content_index.unwrap_or(0))
                    .or_insert_with(|| ReasoningTextContent::ReasoningText {
                        text: String::new(),
                    })
                {
                    text.push_str(&delta);
                }
                let id = self.reasoning_slot_key(output_index, outer_item_id.as_deref(), out);
                out.reasoning_delta(
                    &id,
                    outer_item_id
                        .clone()
                        .and_then(crate::streaming::non_empty_id),
                    delta,
                );
            }
            ItemChunkKind::ReasoningSummaryTextDone(chunk) => {
                if let Some(error) = self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Summary,
                    chunk.summary_index,
                    false,
                ) {
                    self.refusal = Some(error);
                    return;
                }
                self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                    .summary
                    .insert(
                        chunk.summary_index,
                        ReasoningSummary::SummaryText { text: chunk.text },
                    );
            }
            ItemChunkKind::ReasoningTextDone(chunk) => {
                if let Some(error) = self.reasoning_kind_conflict(
                    output_index,
                    ReasoningArray::Content,
                    chunk.content_index,
                    false,
                ) {
                    self.refusal = Some(error);
                    return;
                }
                self.reasoning_parts(output_index, outer_item_id.as_deref(), out)
                    .content
                    .insert(
                        chunk.content_index,
                        ReasoningTextContent::ReasoningText { text: chunk.text },
                    );
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                // Tool output interleaving text is a block boundary too.
                self.active_message_part = None;
                out.end_active_text();
                // Establish identity before done arrives; late IDs must not move buffered fragments.
                let slot = self
                    .tool_slots
                    .open(output_index, outer_item_id.as_deref(), None);
                slot.saw_arguments_delta = true;
                let key = slot.key().clone();
                self.argument_fragments
                    .entry(output_index)
                    .or_default()
                    .push(&delta.delta);
                out.tool_arguments(&key, delta.delta);
            }
            _ => {}
        }
    }

    #[doc(hidden)]
    pub fn record_response_chunk(
        &mut self,
        kind: ResponseChunkKind,
        response: CompletionResponse,
        raw_event_data: &str,
        out: &mut AdapterOutput,
    ) -> Result<(), ProviderError> {
        match kind {
            // `response.incomplete` is a genuine terminal (e.g. hitting
            // `max_output_tokens`): the partial output and usage are kept, and
            // the recorded status/incomplete_details map to the finish reason
            // downstream, matching the unary path's `map_finish_reason`.
            ResponseChunkKind::ResponseCompleted | ResponseChunkKind::ResponseIncomplete => {
                // The terminal restates the whole turn, so the message text
                // no delta delivered is published here: a gateway that
                // states its answer only in the terminal body still lands
                // it in the choice, and one that streamed the text first
                // does not state it twice.
                self.merge_terminal_body_text(&response, out);
                // A terminal that contradicts a streamed part's kind is no
                // genuine end of this turn.
                if let Some(error) = self.refusal.take() {
                    return Err(error);
                }
                self.saw_terminal = true;
                self.flush_undecided_parts(out);
                self.flush_reasoning(out);
                // A terminal can close calls missing their done event; incomplete arguments drop.
                self.argument_fragments.clear();
                for (index, slot) in self.tool_slots.drain_ordered_indexed() {
                    let mut end = slot.end(UnparseableToolInput::Drop);
                    end.call_id = self.pending_call_ids.remove(&index);
                    end.namespace = self.pending_namespaces.remove(&index);
                    self.tool_calls
                        .push((slot.key().clone(), BufferedCall::Function(end)));
                }
                // The terminal event is the only place the stream learns how the
                // turn ended, which model answered, and which assistant message
                // (`msg_...`, not the response's `resp_...`) carried the output.
                if let Some(message_id) = message_id_from_response(&response) {
                    self.terminal.message_id = Some(message_id);
                }
                if !response.id.is_empty() {
                    self.terminal.response_id = Some(response.id.clone());
                }
                if !response.model.is_empty() {
                    self.terminal.model = Some(response.model.clone());
                }
                self.terminal.status = Some(response.status);
                if response.incomplete_details.is_some() {
                    self.terminal.incomplete_details = response.incomplete_details;
                }
                if response.usage.is_some() {
                    self.terminal.usage = response.usage;
                }
                if response.reasoning_metadata.is_some() {
                    self.terminal.reasoning_metadata = response.reasoning_metadata;
                }
                if response.reasoning_context.is_some() {
                    self.terminal.reasoning_context = response.reasoning_context;
                }
                Ok(())
            }
            ResponseChunkKind::ResponseFailed => {
                self.merge_terminal_reasoning(&response, out);
                Err(crate::error::ProviderError::from_provider_body(
                    raw_event_data,
                ))
            }
            _ => Ok(()),
        }
    }

    fn push_output_item_done(
        &mut self,
        item: Output,
        output_index: u64,
        out: &mut AdapterOutput,
        emit_completed_tool_calls_immediately: bool,
    ) {
        match item {
            Output::FunctionCall(func) => {
                // Authoritative done fields replace fragments under the slot's established key.
                let slot = self.tool_slots.remove(output_index);
                // The done item restates its own call_id and namespace; the
                // announce-time copies are only for slots the terminal drain
                // must close.
                self.pending_call_ids.remove(&output_index);
                self.pending_namespaces.remove(&output_index);
                let item_id = match &slot {
                    // Keep the key that owns any accumulated fragments.
                    Some(slot) => slot.key().clone(),
                    // Use the shared mint counter to avoid claiming missing provider identity.
                    None if func.id.is_empty() || func.call_id.is_empty() => {
                        self.tool_slots.minted_ids().mint()
                    }
                    None => BlockId::wire(func.id.clone()),
                };
                let fragments = self.argument_fragments.remove(&output_index);
                // A parsed restatement is authoritative, so the accumulator
                // never consults this policy for it; it keeps the spelling
                // effect logs already recorded for such an end.
                let mut end = ToolCallEnd::new(UnparseableToolInput::Drop);
                end.name = Some(func.name);
                // The done item is authoritative for the qualifier too: without
                // it the streamed call would lose a namespace the unary decode
                // of the same item keeps.
                end.namespace = func.namespace;
                // An empty call_id means no provider identity. Never substitute
                // the fc_* item id: replay requires a real call_id.
                end.call_id = crate::streaming::non_empty_id(func.call_id.clone());
                // The finalized call reports the authoritative wire id even
                // when assembly keyed on a minted slot identity (the
                // accumulator honors the override), but only as the item half
                // of a correlated pair.
                end.tool_id = end
                    .call_id
                    .as_ref()
                    .and_then(|_| crate::streaming::non_empty_id(func.id.clone()));
                // The restatement is authoritative in both of its states. A
                // parsed one supersedes the fragments. One that does not parse
                // is judged under the policy its item status selects, and the
                // buffer is made to hold unparseable bytes so it never answers
                // for the restatement with arguments the provider did not
                // assert.
                match func.arguments.classify() {
                    ClassifiedArguments::Parsed(arguments) => end.arguments = Some(arguments),
                    ClassifiedArguments::Unparseable(restated) => {
                        // Item status supplies protocol confidence, never a
                        // cause: it says how far the provider stands behind
                        // this call, and nothing about why its arguments fail
                        // to parse. A call it does not assert is dropped; one
                        // it asserts (`completed`, or `in_progress` at this
                        // terminal normalization point) must not be silently
                        // lost, so it is refused by name.
                        end.on_unparseable = match func.status {
                            ToolStatus::Incomplete => UnparseableToolInput::Drop,
                            ToolStatus::Completed | ToolStatus::InProgress => {
                                UnparseableToolInput::Error
                            }
                        };
                        match fragments {
                            // No fragment preceded it: the restatement is the
                            // whole input, and reaches the buffer once.
                            None => out.tool_arguments(&item_id, restated),
                            // The fragments are already unparseable, as when they
                            // stream the same bytes the restatement repeats:
                            // re-emitting would put those bytes in the buffer
                            // twice, and the verdict is already the policy's.
                            Some(fragments) if !fragments.parses() => {}
                            // The fragments parse but the provider restated
                            // something that does not (a whitespace or `{}` delta
                            // followed by malformed text). Appending the
                            // restatement leaves the buffer unparseable: a
                            // complete JSON value, or whitespace, followed by
                            // non-whitespace that does not parse, never parses.
                            Some(_) => out.tool_arguments(&item_id, restated),
                        }
                    }
                }

                if emit_completed_tool_calls_immediately {
                    out.tool_end(item_id, end);
                } else {
                    self.tool_calls.push((item_id, BufferedCall::Function(end)));
                }
            }
            // A custom call never fragments here: this client models no
            // `custom_tool_call_input` delta, so no slot is open and the done
            // item is the whole call. Its input travels verbatim, never
            // parsed, because it is not JSON arguments.
            Output::CustomToolCall(call) => {
                self.active_message_part = None;
                out.end_active_text();
                // Mirror the function-call identity rule: an item id keys the
                // block only as half of a correlated pair; otherwise mint from
                // the bridge's one counter.
                let key = if call.id.is_empty() || call.call_id.is_empty() {
                    self.tool_slots.minted_ids().mint()
                } else {
                    BlockId::wire(call.id.clone())
                };
                let mut end = CustomToolCallEnd::new(call.name, call.input)
                    .with_call_id(call.call_id)
                    .with_namespace(call.namespace);
                if end.call_id.is_some() {
                    end = end.with_tool_id(call.id);
                }
                if emit_completed_tool_calls_immediately {
                    out.custom_tool_call(key, end);
                } else {
                    self.tool_calls.push((key, BufferedCall::Custom(end)));
                }
            }
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                ..
            } => {
                if self.finished_reasoning.contains(&output_index) {
                    return;
                }
                if let Some(error) =
                    self.reasoning_snapshot_conflict(output_index, &summary, &content)
                {
                    self.refusal = Some(error);
                    return;
                }
                let had_slot = self.reasoning_slots.contains_key(&output_index);
                let nonempty = !summary.is_empty()
                    || !content.is_empty()
                    || encrypted_content.as_ref().is_some_and(|s| !s.is_empty())
                    || signature.is_some();
                let parts = self.reasoning_parts(output_index, Some(&id), out);
                parts.absorb(summary, content, encrypted_content, signature);
                parts.wire_sent = true;
                if nonempty || parts.has_opaque() || (!had_slot && parts.id.is_some()) {
                    parts.ready = true;
                    let id = parts.id.clone();
                    let key = self.reasoning_slot_key(output_index, id.as_deref(), out);
                    // Keep the slot's original position while delaying its one
                    // authoritative close until terminal or partial/error flush.
                    out.reasoning_start(&key, id);
                }
            }
            Output::Message(message) => {
                self.publish_message_text(output_index, &message, true, out);
                // A message item with no id starts no block: there is
                // nothing to key it on.
                if let Some(id) = crate::streaming::non_empty_id(message.id) {
                    out.message_id(id);
                }
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`) arriving on `response.output_item.done`. Surface
            // the raw item to stream consumers, mirroring how the non-streaming
            // decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                out.unknown(value.into());
            }
            // A compaction item mid-stream: surfaced raw like an unmodeled
            // item so a stateless consumer can capture it from the stream.
            Output::Compaction(fields) => {
                let mut map = fields;
                map.insert(
                    "type".to_string(),
                    serde_json::Value::String("compaction".to_string()),
                );
                out.unknown(serde_json::Value::Object(map).into());
            }
        }
    }

    /// Emit whole-body output items in order, then record terminal metadata.
    /// Structured reasoning suppresses the top-level reasoning display string.
    pub(crate) fn replay_whole_response(
        &mut self,
        response: CompletionResponse,
        out: &mut AdapterOutput,
    ) {
        // A compatible backend that reports its reasoning as one top-level
        // string has no stream event for it, so the body is the only place it
        // is stated. Structured `reasoning` items supersede it: publishing
        // both would carry one chain of thought twice.
        let structured_reasoning = response
            .output
            .iter()
            .any(|item| matches!(item, Output::Reasoning { .. }));
        if !structured_reasoning
            && let Some(reasoning) = response
                .provider_reasoning
                .as_deref()
                .filter(|reasoning| !reasoning.is_empty())
        {
            // Minted from the bridge's ONE counter, as a delta would be:
            // this block has no wire id to key it on.
            let key = self.tool_slots.minted_ids().mint();
            out.reasoning_block(
                key,
                None,
                crate::message::ReasoningContent::Text {
                    text: reasoning.to_owned(),
                    signature: None,
                },
            );
        }

        for (output_index, item) in response.output.iter().cloned().enumerate() {
            let output_index = output_index as u64;
            // Immediate publication preserves output order when the fold registers parts.
            self.push_output_item_done(item, output_index, out, true);
        }

        // `response.completed` and `response.incomplete` are the wire's two
        // genuine terminals; any other status (`failed`, `cancelled`) rides
        // through `map_finish_reason` verbatim on the completed path, exactly
        // as the unary conversion always did.
        let kind = if matches!(response.status, ResponseStatus::Incomplete) {
            ResponseChunkKind::ResponseIncomplete
        } else {
            ResponseChunkKind::ResponseCompleted
        };
        // The raw body is read only for a `response.failed` error payload,
        // which neither of those kinds is.
        if let Err(error) = self.record_response_chunk(kind, response, "", out) {
            out.error(error);
        }
    }

    /// Flush retained reasoning and fully-delivered tool calls without finishing the
    /// stream. The errored-terminal path flushes these before the error and
    /// must not produce a terminal record.
    #[doc(hidden)]
    pub fn flush_tool_calls(&mut self, out: &mut AdapterOutput) {
        self.flush_undecided_parts(out);
        self.flush_reasoning(out);
        for (id, call) in std::mem::take(&mut self.tool_calls) {
            match call {
                BufferedCall::Function(end) => out.tool_end(id, end),
                BufferedCall::Custom(end) => out.custom_tool_call(id, end),
            }
        }
    }

    /// Flush the buffered tool calls, then the terminal record when a
    /// genuine terminal event arrived.
    #[doc(hidden)]
    pub fn finish(mut self, out: &mut AdapterOutput) {
        self.flush_tool_calls(out);
        // Only a genuine terminal event (`response.completed` or
        // `response.incomplete`) counts as the provider ending the turn; a
        // stream that ended without one was truncated,
        // and a synthesized terminal record would present the partial turn as
        // a successful, default-usage completion.
        if !self.saw_terminal {
            return;
        }
        match terminal_record(
            &self.provider,
            self.upstream_reasoning_issuer,
            self.terminal,
        ) {
            Ok(record) => out.final_record(record),
            Err(error) => out.error(error),
        }
    }
}

/// One function-call slot's streamed argument fragments, assembled by the
/// accumulator's own rule
/// ([`append_tool_input_fragment`](crate::streaming::append_tool_input_fragment)),
/// so it holds what the accumulator's buffer for that slot holds.
#[derive(Debug, Default)]
struct AssembledArguments {
    bytes: String,
    /// The accumulation bound refused a fragment. The accumulator then
    /// finalizes the call as unparseable whatever the retained bytes say.
    overflowed: bool,
}

impl AssembledArguments {
    fn push(&mut self, fragment: &str) {
        if !crate::streaming::append_tool_input_fragment(&mut self.bytes, fragment) {
            self.overflowed = true;
        }
    }

    /// Whether the accumulator would finalize these fragments as arguments,
    /// under the one classification every reader of a Responses argument
    /// string shares.
    fn parses(&self) -> bool {
        !self.overflowed
            && matches!(
                FunctionCallArguments(self.bytes.clone()).classify(),
                ClassifiedArguments::Parsed(_)
            )
    }
}

/// Fill absent sequence, output, content, and summary indices with zero.
/// Preserve existing fields and content. Return `None` for invalid or non-object
/// JSON or serialization failure. Missing output indices can merge distinct items
/// into slot zero; callers must enable repair only for compatible dialects.
fn repair_envelope_less_frame(data: &str) -> Option<String> {
    let mut value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let object = value.as_object_mut()?;
    for field in [
        "sequence_number",
        "output_index",
        "content_index",
        "summary_index",
    ] {
        object
            .entry(field)
            .or_insert_with(|| serde_json::Value::from(0));
    }
    serde_json::to_string(&value).ok()
}

/// Classified Responses stream frame, whole reply, error, or sentinel.
pub enum ResponsesEvent {
    /// Decoded stream frame with raw bytes retained for provider errors.
    Frame {
        /// The frame's payload, verbatim.
        raw: String,
        /// The frame, decoded.
        chunk: StreamingCompletionChunk,
    },
    /// The unary reply: the response object itself, which carries no `type`
    /// discriminator because it is not an event.
    Whole(Box<CompletionResponse>),
    /// A success whose body is the provider's error envelope instead of a
    /// response, with the raw body the error preserves. Both the stream's
    /// own `error` event and a 200 whose whole body is an envelope reach
    /// this.
    Failure(String),
    /// `[DONE]` sentinel. Does not independently establish successful completion.
    Sentinel,
}

/// The `response` object of a raw response-lifecycle frame, exactly as the
/// provider sent it.
fn native_response_object(raw: &str) -> Option<serde_json::Value> {
    let mut frame = serde_json::from_str::<serde_json::Value>(raw).ok()?;
    frame
        .as_object_mut()?
        .remove("response")
        .filter(serde_json::Value::is_object)
}

/// Top-level presence markers for response bodies, including error envelopes.
const WHOLE_BODY_MARKERS: &[&str] = &["object", "output", "status", "error"];

/// Whether valid JSON carries the `error` event tag.
fn is_error_event(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data)
        .is_ok_and(|value| value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
}

/// The error envelope a success body can carry instead of a response. The
/// error itself is built from the raw body; this only proves the shape.
#[derive(Deserialize)]
struct ErrorEnvelope {
    // Validate envelope presence while retaining raw bytes for the reported error.
    #[allow(dead_code)]
    error: serde_json::Value,
}

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
///
/// Holds the per-reply assembly state ([`RawChoiceAccumulator`]); frame
/// triage policy lives in the driver, not here.
pub struct ResponsesDecoder {
    /// The reply's own envelope, captured from the terminal event.
    ///
    /// A unary call on a dialect that always streams answers with an event
    /// stream, so there is no reply document for the driver to parse; the
    /// terminal `response.completed` carries it, and this is what makes
    /// `CompletionResponse::raw` the reply rather than a summary of it.
    document: Option<serde_json::Value>,
    accumulator: RawChoiceAccumulator,
    options: ResponsesStreamOptions,
    /// Whether to repair absent envelope indices before retrying classification.
    /// Selected by dialect, independently of unary or streaming mode.
    repair_envelopes: bool,
    /// A `response.failed` event (or a success-status error envelope) ended
    /// the turn: the flush-then-`Err` sequence has been pushed and the
    /// driver stops consuming.
    finished: bool,
}

impl ResponsesDecoder {
    /// A decoder for one reply of `provider`'s Responses endpoint.
    pub fn new(provider: &str, options: ResponsesStreamOptions) -> Self {
        Self {
            document: None,
            accumulator: RawChoiceAccumulator::new(provider, None),
            options,
            repair_envelopes: false,
            finished: false,
        }
    }

    /// Salvage replayed frames that omit their envelope bookkeeping.
    pub fn with_envelope_repair(mut self) -> Self {
        self.repair_envelopes = true;
        self
    }

    /// Record reasoning as issued by the upstream model's family, for a
    /// gateway that relays each upstream's own reasoning state.
    pub fn with_upstream_reasoning_issuer(mut self) -> Self {
        self.accumulator.upstream_reasoning_issuer = true;
        self
    }

    /// Seed the terminal's usage for a replayed body whose frames may not
    /// carry one (the unary Responses body's own `usage`).
    pub fn with_initial_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        self.accumulator.terminal.usage = usage;
        self
    }

    /// Recognize sentinels and tagged events before considering an untagged
    /// whole response or error envelope.
    fn classify_payload(&self, data: &str) -> WireEvent<ResponsesEvent> {
        if data.trim() == "[DONE]" {
            return WireEvent::Known(ResponsesEvent::Sentinel);
        }
        if is_error_event(data) {
            return WireEvent::Known(ResponsesEvent::Failure(data.to_owned()));
        }
        // A discriminator selects the event contract. Its decode failure must
        // not be rescued by unrelated whole-response fields on the same object.
        if serde_json::from_str::<serde_json::Value>(data)
            .is_ok_and(|value| value.get("type").is_some())
        {
            return classify_responses_frame(data).map(|chunk| ResponsesEvent::Frame {
                raw: data.to_owned(),
                chunk,
            });
        }
        let body = |data: &str| {
            wire::classify_marker_keyed_frame::<CompletionResponse>(data, WHOLE_BODY_MARKERS)
                .map(|response| ResponsesEvent::Whole(Box::new(response)))
        };
        let envelope = |data: &str| {
            // Error envelopes must retain the provider's diagnostic on every dialect.
            wire::classify_marker_keyed_frame::<ErrorEnvelope>(data, &["error"])
                .map(|_| ResponsesEvent::Failure(data.to_owned()))
        };
        wire::classify_or(
            data,
            |data| {
                classify_responses_frame(data).map(|chunk| ResponsesEvent::Frame {
                    raw: data.to_owned(),
                    chunk,
                })
            },
            |data| wire::classify_or(data, body, envelope),
        )
    }

    fn interpret_frame(
        &mut self,
        raw: String,
        chunk: StreamingCompletionChunk,
        out: &mut AdapterOutput,
    ) {
        match chunk {
            StreamingCompletionChunk::Delta(chunk) => {
                self.accumulator.decode_item_chunk(chunk, self.options, out);
                // A contradictory part ends the reply like any terminal
                // error: delivered content flushes first.
                if let Some(error) = self.accumulator.refusal.take() {
                    self.accumulator.flush_tool_calls(out);
                    out.error(error);
                    self.finished = true;
                }
            }
            StreamingCompletionChunk::Response(chunk) => {
                let ResponseChunk { kind, response, .. } = chunk;
                // Keep the latest snapshot so raw output includes final status
                // and usage. The provider's own `response` object is kept
                // rather than the typed re-serialization: the typed decode
                // drops an echoed metadata key whose value does not fit its
                // field, and the capture is where that native evidence stays.
                self.document =
                    native_response_object(&raw).or_else(|| serde_json::to_value(&response).ok());
                if matches!(kind, ResponseChunkKind::ResponseIncomplete)
                    && self.options.refuses_incomplete()
                {
                    // A streamed turn the provider cut short is not a
                    // completed one: unless the caller opted into partial
                    // success, it ends in an error carrying the provider's
                    // terminal event. Fully-delivered tool calls flush first,
                    // as before any terminal error.
                    self.accumulator.merge_terminal_reasoning(&response, out);
                    self.accumulator.flush_tool_calls(out);
                    out.error(crate::error::ProviderError::from_provider_body(&raw));
                    self.finished = true;
                    return;
                }
                if matches!(kind, ResponseChunkKind::ResponseCompleted) {
                    // Inert under the driver, which records the same fields
                    // off the terminal record; the client layer's stream
                    // loop has no other recording site.
                    let span = tracing::Span::current();
                    span.record("gen_ai.response.id", response.id.as_str());
                    span.record("gen_ai.response.model", response.model.as_str());
                }
                if let Err(error) = self
                    .accumulator
                    .record_response_chunk(kind, response, &raw, out)
                {
                    // `response.failed`: fully-delivered tool calls flush
                    // before the terminal error, which ends the reply with
                    // no terminal record, preserving the failure signal.
                    self.accumulator.flush_tool_calls(out);
                    out.error(error);
                    self.finished = true;
                }
            }
        }
    }

    /// Flush what the accumulator still holds: the buffered tool calls, then
    /// the terminal record when a genuine terminal arrived.
    fn flush(&mut self, out: &mut AdapterOutput) {
        let provider = self.accumulator.provider.clone();
        let mut fresh = RawChoiceAccumulator::new(provider, None);
        fresh.upstream_reasoning_issuer = self.accumulator.upstream_reasoning_issuer;
        let accumulator = std::mem::replace(&mut self.accumulator, fresh);
        accumulator.finish(out);
    }
}

impl Decoder<Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesEvent> {
        let data = frame.as_str().into_owned();
        if !self.repair_envelopes {
            return self.classify_payload(&data);
        }
        // Replayed bodies omit envelope bookkeeping fields; salvage through
        // the SAME interpreter, with the operation-error wording the
        // buffered driver surfaces verbatim.
        wire::classify_with_repair(
            &data,
            |data| self.classify_payload(data),
            repair_envelope_less_frame,
            |corrupt| {
                <serde_json::Error as serde::de::Error>::custom(format!(
                    "invalid JSON frame in buffered Responses SSE body: {corrupt}"
                ))
            },
            || {
                let kind = serde_json::from_str::<serde_json::Value>(&data)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .unwrap_or_default();
                <serde_json::Error as serde::de::Error>::custom(format!(
                    "malformed `{kind}` event in buffered Responses SSE body"
                ))
            },
        )
    }

    fn interpret(&mut self, event: ResponsesEvent, out: &mut AdapterOutput) {
        if self.finished {
            return;
        }

        match event {
            ResponsesEvent::Frame { raw, chunk } => self.interpret_frame(raw, chunk, out),
            // The unary reply is the same turn stated at once: replay it as
            // the events the stream sends, then close it with the terminal
            // the body itself is.
            ResponsesEvent::Whole(response) => {
                self.document = serde_json::to_value(&*response).ok();
                self.accumulator.replay_whole_response(*response, out);
                self.flush(out);
            }
            ResponsesEvent::Failure(raw) => {
                self.accumulator.flush_tool_calls(out);
                out.error(crate::error::ProviderError::from_provider_body(&raw));
                self.finished = true;
            }
            // Nothing to interpret: the terminal record comes from
            // `response.completed`, or from the driver's EOF flush.
            ResponsesEvent::Sentinel => {}
        }
    }

    fn finish(&mut self, out: &mut AdapterOutput) {
        self.flush(out);
    }

    fn flush_before_terminal_error(&mut self, out: &mut AdapterOutput) {
        // Tool calls the provider fully delivered are content: they flush
        // before the terminal error reaches the consumer.
        self.accumulator.flush_tool_calls(out);
    }

    fn document(&self) -> Option<serde_json::Value> {
        self.document.clone()
    }

    fn project(&self, payload: &[u8], sink: &mut dyn crate::wire::ObservationSink) {
        super::wire::project_payload(payload, sink);
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Output-item event with its slot index and optional provider item ID.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ItemChunk {
    /// Item ID. Optional.
    pub item_id: Option<String>,
    /// The output index of the item from a given streamed response.
    pub output_index: u64,
    /// The item type chunk, as well as the inner data.
    #[serde(flatten)]
    pub data: ItemChunkKind,
}

/// The item chunk type from OpenAI's Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ItemChunkKind {
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(StreamingItemDoneOutput),
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(StreamingItemDoneOutput),
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartChunk),
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartChunk),
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(DeltaTextChunk),
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(OutputTextChunk),
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta(DeltaTextChunk),
    #[serde(rename = "response.refusal.done")]
    RefusalDone(RefusalTextChunk),
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta(DeltaTextChunkWithItemId),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextDeltaChunk),
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextDoneChunk),
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta(DeltaTextChunkWithItemId),
    /// Raw-reasoning text restatement. Decoded but not emitted to avoid
    /// duplicating accumulated reasoning deltas.
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone(OutputTextChunk),
    // No `#[serde(other)]` catch-all: unknown event types are triaged by the
    // classify layer (`classify_responses_frame` checks the `type` tag against
    // `is_known_responses_event_type` BEFORE decoding), so a frame that
    // reaches this decoder with an unmodeled tag is a known-set/enum drift
    // and must fail loudly (`Corrupt`) rather than be silently absorbed.
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingItemDoneOutput {
    pub sequence_number: u64,
    pub item: Output,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPartChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub part: ContentPartChunkPart,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartChunkPart {
    Refusal {
        refusal: String,
    },
    OutputText {
        text: String,
    },
    SummaryText {
        text: String,
    },
    /// Reasoning content, announced by `response.content_part.*` on some
    /// gateways before its `response.reasoning_text.*` deltas.
    ReasoningText {
        text: String,
    },
    /// Unmodeled content part retained verbatim in neutral opaque metadata.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Decode known tags only with a string text field and preserve unknown tags.
/// Absent and nonstring tags return an error. Duplicate keys use the last value retained by
/// `serde_json::Value`.
impl<'de> Deserialize<'de> for ContentPartChunkPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let text_field = |part: &str| -> Result<String, D::Error> {
            value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "`{part}` content part is missing a string `text` field"
                    ))
                })
        };
        match value.get("type").cloned() {
            Some(serde_json::Value::String(tag)) => match tag.as_str() {
                "refusal" => Ok(Self::Refusal {
                    refusal: value
                        .get("refusal")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            serde::de::Error::custom("refusal part requires a string `refusal`")
                        })?
                        .to_owned(),
                }),
                "output_text" => Ok(Self::OutputText {
                    text: text_field("output_text")?,
                }),
                "summary_text" => Ok(Self::SummaryText {
                    text: text_field("summary_text")?,
                }),
                "reasoning_text" => Ok(Self::ReasoningText {
                    text: text_field("reasoning_text")?,
                }),
                _ => Ok(Self::Unknown(value)),
            },
            Some(_) => Err(serde::de::Error::custom(
                "content part `type` must be a string",
            )),
            None => Err(serde::de::Error::custom(
                "a Responses content part needs a string `type` tag",
            )),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunkWithItemId {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefusalTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub refusal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArgsTextChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    pub sequence_number: u64,
    /// The call's arguments, in the same type (and so under the same
    /// classification) as the `output_item.done` item's
    /// [`OutputFunctionCall::arguments`](super::OutputFunctionCall::arguments);
    /// [`FunctionCallArguments::reconcile`] settles whether the two agree.
    pub arguments: FunctionCallArguments,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryPartChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub part: SummaryPartChunkPart,
}

/// A reasoning-summary text fragment (`response.reasoning_summary_text.delta`).
///
/// The delta and done events are distinct shapes with distinct members: a
/// `delta` is required here and a `text` member is not accepted in its place,
/// so a done-shaped payload can never parse as a fragment (nor the reverse).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextDeltaChunk {
    /// The reasoning summary this fragment belongs to.
    pub summary_index: u64,
    pub sequence_number: u64,
    /// The incremental text.
    pub delta: String,
}

/// A reasoning-summary text restatement (`response.reasoning_summary_text.done`).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextDoneChunk {
    /// The reasoning summary this restatement closes.
    pub summary_index: u64,
    pub sequence_number: u64,
    /// The summary's full text.
    pub text: String,
}

pub type SummaryPartChunkPart = ReasoningSummary;

#[cfg(test)]
mod tests;
