//! Strict Responses decoding on the branded completion writer.
//!
//! Known frames must satisfy their wire schema. Unknown events retain their
//! original JSON. Output and content positions remain separate coordinates,
//! and only a genuine provider terminal establishes an end of response.

use crate::completion::{Cost, FinishReason, ProviderMessageId, Usage};
use crate::error::{ProviderCallEvidence, ProviderError};
use crate::json_utils::Lenient;
use crate::message::{CallId, CarriedPosition, Source, SourceLocation, ToolName};
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::streaming::{DiscardReason, ReasoningAuthority, SourceOrder};
use crate::wire::{Decoder, Flow, IncompleteEof, Out, WireCitation, WireEvent, WireFrame};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

mod reader;
#[cfg(feature = "websocket")]
pub(crate) use reader::websocket_done;

/// The protected accumulator's per-call fragment bound.
const MAX_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

const ITEM_EVENTS: &str = "output_item.added output_item.done content_part.added content_part.done \
    output_text.delta output_text.done refusal.delta refusal.done function_call_arguments.delta \
    function_call_arguments.done reasoning_summary_part.added reasoning_summary_part.done \
    reasoning_summary_text.delta reasoning_summary_text.done reasoning_text.delta reasoning_text.done";

fn is_known_responses_event_type(kind: &str) -> bool {
    is_lifecycle_event(kind)
        || kind
            .strip_prefix("response.")
            .is_some_and(|event| ITEM_EVENTS.split_whitespace().any(|known| known == event))
}

pub(crate) fn is_lifecycle_event(kind: &str) -> bool {
    matches!(
        kind,
        "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete"
    )
}

fn is_document_event(kind: &str) -> bool {
    is_known_responses_event_type(kind) || matches!(kind, "error" | "response.queued")
}

/// Known lifecycle discriminants shared with the explicit observation reader.
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

/// Whether a streamed incomplete terminal is refused or retained truthfully.
/// Unary bodies accept incomplete replies independently of this policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteTerminal {
    #[default]
    Refuse,
    Accept,
}

/// Independent completed-call publication and streamed-incomplete policies.
#[derive(Clone, Copy, Debug)]
pub struct ResponsesStreamOptions {
    immediate_tool_calls: bool,
    incomplete: IncompleteTerminal,
}

impl ResponsesStreamOptions {
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
    pub const fn with_incomplete(mut self, incomplete: IncompleteTerminal) -> Self {
        self.incomplete = incomplete;
        self
    }
}

/// Strict discriminator-first classification, without envelope repair.
pub fn classify_responses_payload(data: &str) -> WireEvent<ResponsesEvent> {
    reader::classify(data)
}

#[cfg(any(feature = "websocket", test))]
pub(crate) fn classify_websocket_payload(data: &str) -> WireEvent<ResponsesEvent> {
    ReaderPolicy::WebSocket.classify(data)
}

#[derive(Clone, Copy, Debug)]
enum ReaderPolicy {
    HttpSse {
        repair_envelopes: bool,
    },
    #[cfg(any(feature = "websocket", test))]
    WebSocket,
}

impl ReaderPolicy {
    fn classify(self, data: &str) -> WireEvent<ResponsesEvent> {
        match self {
            Self::HttpSse { repair_envelopes } => classify_reader_payload(data, repair_envelopes),
            #[cfg(any(feature = "websocket", test))]
            Self::WebSocket => reader::classify_websocket(data),
        }
    }
}

/// Strict tagged classification. Whole bodies use `classify_responses_payload`.
#[doc(hidden)]
pub fn classify_responses_frame(data: &str) -> WireEvent<ResponsesEvent> {
    reader::tagged(data)
}

/// A classified Responses payload.
#[derive(Debug)]
pub enum ResponsesEvent {
    /// A stream event of a known `type`, as the provider sent it.
    Frame {
        /// The event's `type`.
        kind: String,
        /// The event.
        frame: Value,
        /// The frame's payload, verbatim, for the error a failed reply
        /// reports.
        raw: String,
    },
    /// The unary reply: the response object itself, which carries no `type`
    /// because it is not an event.
    Whole(Value),
    /// The provider's error, as the stream's `error` event or a success
    /// body holding an error envelope instead of a response.
    Failure(String),
    /// The `[DONE]` sentinel. The provider's end is `response.completed`.
    Sentinel,
}

/// The keys a response object, or the error envelope a success body can
/// hold instead, always has one of.
const WHOLE_BODY_MARKERS: &[&str] = &["object", "output", "status", "id", "error"];

/// A payload that names its `type`, read as the JSON it is.
struct Tagged(Value);

impl<'de> serde::Deserialize<'de> for Tagged {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.str("type").is_some() {
            Ok(Self(value))
        } else {
            Err(serde::de::Error::custom("the payload names no `type`"))
        }
    }
}

/// Classify one payload by its `type`; a payload without one is the unary
/// response object or an error envelope.
fn classify_document_payload(data: &str) -> WireEvent<ResponsesEvent> {
    if data.trim() == "[DONE]" {
        return WireEvent::Known(ResponsesEvent::Sentinel);
    }
    wire::classify_or_untagged(
        data,
        "type",
        |data| {
            wire::classify_tagged_frame::<Tagged>(data, "type", is_document_event).map(
                |Tagged(frame)| match frame.str("type") {
                    Some("error") => ResponsesEvent::Failure(data.to_owned()),
                    kind => ResponsesEvent::Frame {
                        kind: kind.unwrap_or_default().to_owned(),
                        raw: data.to_owned(),
                        frame,
                    },
                },
            )
        },
        |data| {
            wire::classify_marker_keyed_frame::<Value>(data, WHOLE_BODY_MARKERS).map(|body| {
                let error = body.get("error").is_some_and(|error| !error.is_null());
                if error && body.get("output").is_none() && body.get("status").is_none() {
                    ResponsesEvent::Failure(data.to_owned())
                } else {
                    ResponsesEvent::Whole(body)
                }
            })
        },
    )
}

/// The usage a Responses `usage` value reports; a counter that is absent or
/// not a count is unreported. Its cost is the one the provider reports.
pub(crate) fn usage_of(usage: &Value) -> Usage {
    let count = |pointer: &str| usage.at(pointer).and_then(Lenient::as_u64_lenient);
    Usage {
        input_tokens: count("/input_tokens"),
        output_tokens: count("/output_tokens"),
        total_tokens: count("/total_tokens"),
        cached_input_tokens: count("/input_tokens_details/cached_tokens"),
        cache_creation_input_tokens: count("/input_tokens_details/cache_write_tokens"),
        reasoning_tokens: count("/output_tokens_details/reasoning_tokens"),
        ..Usage::default()
    }
    .cost(reported_cost(usage))
}

/// xAI cost ticks per USD.
const TICKS_PER_USD: f64 = 1e10;

/// The total a `usage` value reports in USD: xAI's integer
/// `cost_in_usd_ticks`, or OpenRouter's `cost` in credits of one USD.
fn reported_cost(usage: &Value) -> Option<Cost> {
    let total = match usage.u64("cost_in_usd_ticks") {
        Some(ticks) => ticks as f64 / TICKS_PER_USD,
        None => usage.f64("cost").filter(|cost| cost.is_finite())?,
    };
    Some(Cost::from_total(total))
}

/// The citations a message item's `output_text` parts state, in part order.
/// The API does not document the unit of `start_index` and `end_index` and
/// quotes no text, so each cites the whole block and its offsets stay in the
/// item.
fn citations_of(item: &Value) -> Vec<WireCitation> {
    item.arr("content")
        .iter()
        .flat_map(|part| part.arr("annotations"))
        .filter_map(source_of)
        .map(|source| WireCitation::new(None, vec![source]))
        .collect()
}

/// The source an annotation names; `None` for an annotation that names
/// none or is of an unknown type.
fn source_of(annotation: &Value) -> Option<Source> {
    let text = |key: &str| {
        annotation
            .str(key)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let location = match annotation.str("type")? {
        "url_citation" => SourceLocation::Url { url: text("url")? },
        "file_citation" | "file_path" | "container_file_citation" => SourceLocation::File {
            file_id: text("file_id")?,
            filename: text("filename"),
            container_id: text("container_id"),
        },
        _ => return None,
    };
    let source = Source::new(location);
    Some(match text("title") {
        Some(title) => source.title(title),
        None => source,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parent {
    Message,
    Reasoning,
}

#[derive(Default)]
struct MessagePart {
    at: Option<usize>,
    text: String,
    raw: Option<Value>,
    opaque: bool,
    refusal: bool,
    /// Last non-delta snapshot phase observed for activation, independent
    /// of the containing message's canonical phase policy.
    observed_phase: Option<String>,
}

#[derive(Default)]
struct MessageState {
    id: Option<String>,
    phase: Option<String>,
    parts: BTreeMap<u64, MessagePart>,
    delta: bool,
    complete: bool,
    item: Option<Value>,
}

#[derive(Default)]
struct ReasoningState {
    at: Option<usize>,
    id: Option<String>,
    summary: BTreeMap<u64, Value>,
    content: BTreeMap<u64, Value>,
    item: Option<Value>,
    complete: bool,
    encrypted: Option<String>,
    signature: Option<String>,
}

impl ReasoningState {
    fn raw(&self) -> Value {
        let mut item = self
            .item
            .clone()
            .unwrap_or_else(|| json!({"type":"reasoning"}));
        let fields = item.as_object_mut().expect("reasoning item is an object");
        if let Some(id) = &self.id {
            fields.insert("id".into(), json!(id));
        }
        fields.insert(
            "summary".into(),
            json!(self.summary.values().collect::<Vec<_>>()),
        );
        if !self.content.is_empty() {
            fields.insert(
                "content".into(),
                json!(self.content.values().collect::<Vec<_>>()),
            );
        } else {
            fields.remove("content");
        }
        if let Some(encrypted) = &self.encrypted {
            fields.insert("encrypted_content".into(), json!(encrypted));
        }
        if let Some(signature) = &self.signature {
            fields.insert("signature".into(), json!(signature));
        }
        item
    }

    fn carried(&self) -> bool {
        self.encrypted.is_some()
            || self.signature.is_some()
            || (self.complete && self.id.is_some())
            || !self.content.is_empty()
            || !self.summary.is_empty()
            || self.summary.values().any(|part| {
                part.str("type") != Some("summary_text")
                    || part.as_object().is_some_and(|fields| {
                        fields.keys().any(|key| key != "type" && key != "text")
                    })
            })
    }

    fn has_opaque(&self) -> bool {
        self.summary
            .values()
            .any(|part| reasoning_opaque(part, true))
            || self
                .content
                .values()
                .any(|part| reasoning_opaque(part, false))
    }
}

/// Completion confidence comes only from the authoritative function item.
/// It is independent of the parser's error and never inferred from fragments.
enum CallConfidence {
    Incomplete,
    Asserted,
}

enum FunctionInput {
    Parsed,
    Malformed {
        error: serde_json::Error,
        confidence: CallConfidence,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FlushBoundary {
    Partial,
    GenuineTerminal,
}

enum CallEnding {
    Pending,
    Custom,
    Function(FunctionInput),
}

impl CallEnding {
    fn is_complete(&self) -> bool {
        !matches!(self, Self::Pending)
    }
}

struct CallState {
    at: usize,
    id: CallId,
    item_id: Option<String>,
    name: String,
    namespace: Option<String>,
    arguments: String,
    sent: String,
    overflowed: bool,
    ending: CallEnding,
}

#[derive(Clone, PartialEq)]
struct UndecidedPart {
    id: Option<String>,
    raw: Value,
}

/// One reply's strict decoder. All emitted parts belong to `Out`'s branded
/// completion fold; this state holds wire correlation and provider restatements.
pub struct ResponsesDecoder {
    provider: String,
    options: ResponsesStreamOptions,
    reader_policy: ReaderPolicy,
    upstream_reasoning_issuer: bool,
    messages: BTreeMap<u64, MessageState>,
    message_ids: HashMap<String, u64>,
    reasoning: BTreeMap<u64, ReasoningState>,
    calls: BTreeMap<u64, CallState>,
    parents: HashMap<u64, Parent>,
    undecided: BTreeMap<(u64, u64), Vec<UndecidedPart>>,
    provider_items: HashSet<u64>,
    finished_calls: HashSet<u64>,
    call_items: HashMap<String, u64>,
    last_message_id: Option<String>,
    /// The real text Draft last activated by a provider event. Interleaving
    /// resets this without closing that Part or duplicating canonical content.
    active_message_part: Option<usize>,
    pending_flush_error: Option<ProviderError>,
    terminal_issuer: bool,
    finished: bool,
}

impl ResponsesDecoder {
    /// A decoder for the named dialect and explicitly selected policies.
    pub fn new(provider: &str, options: ResponsesStreamOptions) -> Self {
        Self {
            provider: provider.to_owned(),
            options,
            reader_policy: ReaderPolicy::HttpSse {
                repair_envelopes: false,
            },
            upstream_reasoning_issuer: false,
            messages: BTreeMap::new(),
            message_ids: HashMap::new(),
            reasoning: BTreeMap::new(),
            calls: BTreeMap::new(),
            parents: HashMap::new(),
            undecided: BTreeMap::new(),
            provider_items: HashSet::new(),
            finished_calls: HashSet::new(),
            call_items: HashMap::new(),
            last_message_id: None,
            active_message_part: None,
            pending_flush_error: None,
            terminal_issuer: false,
            finished: false,
        }
    }

    /// Repair only absent envelope coordinates, preserving invalid supplied
    /// coordinates as defects and retaining the original corruption evidence.
    pub fn with_envelope_repair(mut self) -> Self {
        match &mut self.reader_policy {
            ReaderPolicy::HttpSse { repair_envelopes } => *repair_envelopes = true,
            #[cfg(any(feature = "websocket", test))]
            ReaderPolicy::WebSocket => {}
        }
        self
    }

    #[cfg(any(feature = "websocket", test))]
    pub(crate) fn with_websocket_reader(mut self) -> Self {
        self.reader_policy = ReaderPolicy::WebSocket;
        self
    }

    /// Resolve carried reasoning and items to the gateway's upstream issuer
    /// once the terminal supplies the actual model.
    pub fn with_upstream_reasoning_issuer(mut self) -> Self {
        self.upstream_reasoning_issuer = true;
        self
    }

    fn issuer(&self, response: &Value) -> String {
        match response.str("model").filter(|model| !model.is_empty()) {
            Some(model) if self.upstream_reasoning_issuer => {
                crate::providers::openai::wire::upstream_reasoning_issuer(&self.provider, model)
            }
            _ => self.provider.clone(),
        }
    }

    fn message_slot(&mut self, index: u64, id: Option<&str>, snapshot: Option<&[Value]>) -> u64 {
        let Some(id) = id.filter(|id| !id.is_empty()) else {
            return index;
        };
        if let Some(existing) = self.message_ids.get(id) {
            return *existing;
        }
        // Only a completed exact restatement can claim anonymous deltas. An
        // added item never aliases a distinct anonymous message by coincidence.
        let own_empty = self
            .messages
            .get(&index)
            .is_none_or(|message| message.parts.is_empty())
            && !self.undecided.iter().any(|((slot, _), parts)| {
                *slot == index
                    && parts
                        .iter()
                        .any(|part| part.id.as_deref().is_none_or(|own| own == id))
            });
        let anonymous = own_empty.then_some(snapshot).flatten().and_then(|content| {
            self.messages.iter().find_map(|(slot, message)| {
                (message.id.is_none()
                    && message.delta
                    && !message.parts.is_empty()
                    && content.len() == message.parts.len()
                    && content.iter().enumerate().all(|(part_index, raw)| {
                        message
                            .parts
                            .get(&(part_index as u64))
                            .is_some_and(|part| part_matches(part, raw))
                    }))
                .then_some(*slot)
            })
        });
        let slot = anonymous.unwrap_or(index);
        self.message_ids.insert(id.to_owned(), slot);
        slot
    }

    fn bind(
        &mut self,
        index: u64,
        parent: Parent,
        id: Option<&str>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if parent == Parent::Reasoning && self.parents.get(&index) == Some(&Parent::Message) {
            return Err(ProviderError::Response(format!(
                "conflicting Responses parent kind at output {index}: reasoning in a message"
            )));
        }
        self.parents.entry(index).or_insert(parent);
        // Parts with an explicit item id follow that item to its established
        // position. Id-less parts bind only by their supplied output position.
        let keys: Vec<_> = self.undecided.keys().copied().collect();
        let mut ready = Vec::new();
        for key in keys {
            let Some(parts) = self.undecided.get_mut(&key) else {
                continue;
            };
            let mut retained = Vec::new();
            for part in std::mem::take(parts) {
                let binds = match part.id.as_deref() {
                    Some(own) => id == Some(own),
                    None => key.0 == index,
                };
                if binds {
                    ready.push((key.1, part.raw));
                } else {
                    retained.push(part);
                }
            }
            *parts = retained;
        }
        self.undecided.retain(|_, parts| !parts.is_empty());
        for (part, raw) in ready {
            match parent {
                Parent::Message => self.message_part(index, part, id, raw, out)?,
                Parent::Reasoning => {
                    self.reasoning_part(index, part, false, id, raw, false, out)?
                }
            }
        }
        Ok(())
    }

    fn message_delta(
        &mut self,
        index: u64,
        part_index: u64,
        id: Option<&str>,
        delta: &str,
        refusal: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = self.message_slot(index, id, None);
        self.check_message_kind(index, part_index, false, id)?;
        self.bind(index, Parent::Message, id, out)?;
        let message = self.messages.entry(index).or_default();
        if message.id.is_none() {
            message.id = nonempty(id);
        }
        message.delta = true;
        // deltas do not revoke the phase an actual message snapshot supplied.
        let part = message.parts.entry(part_index).or_default();
        part.refusal |= refusal;
        let at = text_at(part, index, part_index, out)?;
        activate_text(
            &mut self.active_message_part,
            at,
            !part.text.is_empty(),
            out,
        )?;
        part.text.push_str(delta);
        out.push(at, delta)
    }

    fn check_message_kind(
        &self,
        index: u64,
        part: u64,
        opaque: bool,
        id: Option<&str>,
    ) -> Result<(), ProviderError> {
        if let Some(stored) = self
            .messages
            .get(&index)
            .and_then(|message| message.parts.get(&part))
            && stored.opaque != opaque
        {
            return Err(conflicting_kind(
                index,
                "content",
                part,
                stored.opaque,
                opaque,
            ));
        }
        if !opaque
            && self.undecided.get(&(index, part)).is_some_and(|parts| {
                parts
                    .iter()
                    .any(|part| part.id.as_deref().is_none_or(|own| Some(own) == id))
            })
        {
            return Err(conflicting_kind(index, "content", part, true, false));
        }
        Ok(())
    }

    fn message_part(
        &mut self,
        index: u64,
        part_index: u64,
        id: Option<&str>,
        raw: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.message_part_with_phase(index, part_index, id, raw, None, out)
    }

    fn message_part_with_phase(
        &mut self,
        index: u64,
        part_index: u64,
        id: Option<&str>,
        raw: Value,
        phase: Option<&str>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = raw.str("type");
        let opaque = !matches!(kind, Some("output_text" | "refusal"));
        self.check_message_kind(index, part_index, opaque, id)?;
        let message = self.messages.entry(index).or_default();
        if message.id.is_none() {
            message.id = nonempty(id);
        }
        let part = message.parts.entry(part_index).or_default();
        part.opaque = opaque;
        part.refusal |= kind == Some("refusal");
        if !opaque {
            let stated = raw
                .str(if part.refusal { "refusal" } else { "text" })
                .unwrap_or_default();
            let at = text_at(part, index, part_index, out)?;
            let metadata_changed = message_activation_metadata(part.raw.as_ref())
                != message_activation_metadata(Some(&raw));
            let phase_changed =
                phase.is_some_and(|phase| part.observed_phase.as_deref() != Some(phase));
            let has_new_text = stated
                .strip_prefix(&part.text)
                .is_some_and(|suffix| !suffix.is_empty());
            if has_new_text || metadata_changed || phase_changed {
                activate_text(
                    &mut self.active_message_part,
                    at,
                    !part.text.is_empty(),
                    out,
                )?;
            }
            // Snapshots extend deltas; a shorter or contradictory snapshot
            // cannot erase text the consumer already received.
            if let Some(suffix) = stated.strip_prefix(&part.text) {
                out.push(at, suffix)?;
                part.text = stated.to_owned();
            }
        }
        if let Some(phase) = phase {
            part.observed_phase = Some(phase.to_owned());
        }
        part.raw = Some(raw);
        Ok(())
    }

    fn message_item(
        &mut self,
        index: u64,
        item: &Value,
        completed: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let content = item.arr("content");
        let index = self.message_slot(index, item.str("id"), completed.then_some(content));
        // Check the whole snapshot before publishing any part of it.
        for (part, raw) in content.iter().enumerate() {
            let opaque = !matches!(raw.str("type"), Some("output_text" | "refusal"));
            self.check_message_kind(index, part as u64, opaque, item.str("id"))?;
        }
        self.bind(index, Parent::Message, item.str("id"), out)?;
        // the actual snapshot retains phase custody after streamed deltas.
        let phase = item.str("phase");
        for (part, raw) in content.iter().enumerate() {
            self.message_part_with_phase(
                index,
                part as u64,
                item.str("id"),
                raw.clone(),
                phase,
                out,
            )?;
        }
        let message = self.messages.entry(index).or_default();
        if message.id.is_none() {
            message.id = nonempty(item.str("id"));
        }
        message.phase = item.str("phase").map(str::to_owned);
        if completed {
            message.complete = true;
            message.item = Some(item.clone());
            if let Some(id) = item.str("id").filter(|id| !id.is_empty()) {
                self.last_message_id = Some(id.to_owned());
            }
        }
        Ok(())
    }

    fn reasoning_part(
        &mut self,
        index: u64,
        part: u64,
        summary: bool,
        id: Option<&str>,
        raw: Value,
        append: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if self.parents.get(&index) == Some(&Parent::Message) {
            return Err(ProviderError::Response(format!(
                "conflicting Responses parent kind at output {index}, content {part}: a reasoning_text part in a message"
            )));
        }
        self.parents.entry(index).or_insert(Parent::Reasoning);
        let reasoning = self.reasoning.entry(index).or_default();
        if reasoning.id.is_none() {
            reasoning.id = nonempty(id);
        }
        let parts = if summary {
            &mut reasoning.summary
        } else {
            &mut reasoning.content
        };
        let opaque = reasoning_opaque(&raw, summary);
        if let Some(old) = parts.get(&part) {
            let stored = reasoning_opaque(old, summary);
            if stored != opaque {
                return Err(conflicting_kind(
                    index,
                    if summary { "summary" } else { "content" },
                    part,
                    stored,
                    opaque,
                ));
            }
        }
        let delta = append.then(|| raw.str("text").unwrap_or_default().to_owned());
        if append && let Some(previous) = parts.get_mut(&part) {
            let mut text = previous
                .as_str()
                .or_else(|| previous.str("text"))
                .unwrap_or_default()
                .to_owned();
            text.push_str(raw.str("text").unwrap_or_default());
            if previous.is_string() {
                *previous = json!(text);
            } else {
                previous["text"] = json!(text);
            }
        } else {
            parts.insert(part, raw);
        }
        if let Some(delta) = delta {
            self.reasoning_delta(index, &delta, out)?;
        }
        Ok(())
    }

    fn reasoning_delta(
        &mut self,
        index: u64,
        delta: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.active_message_part = None;
        let reasoning = self.reasoning.get_mut(&index).expect("reasoning state");
        if reasoning.at.is_none() {
            let at = out.fresh_index();
            out.in_source_order(SourceOrder::new(index, 0), |out| {
                out.open(at, Block::Reasoning { redacted: false }, Value::Null)
            })?;
            reasoning.at = Some(at);
        }
        if let Some(at) = reasoning.at {
            out.push(at, delta)?;
        }
        Ok(())
    }

    fn reasoning_item(
        &mut self,
        index: u64,
        item: &Value,
        completed: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.bind(index, Parent::Reasoning, item.str("id"), out)?;
        // All kind contradictions are checked before changing either array.
        if let Some(state) = self.reasoning.get(&index) {
            for (array, old, summary) in [
                ("summary", &state.summary, true),
                ("content", &state.content, false),
            ] {
                for (part, raw) in reasoning_array(item, array).iter().enumerate() {
                    if let Some(stored) = old.get(&(part as u64)) {
                        let was_opaque = reasoning_opaque(stored, summary);
                        let opaque = reasoning_opaque(raw, summary);
                        if was_opaque != opaque {
                            return Err(conflicting_kind(
                                index,
                                array,
                                part as u64,
                                was_opaque,
                                opaque,
                            ));
                        }
                    }
                }
            }
        }
        let state = self.reasoning.entry(index).or_default();
        if !state.complete || state.id.is_none() {
            state.id = nonempty(item.str("id")).or_else(|| state.id.clone());
        }
        for (part, raw) in reasoning_array(item, "summary").into_iter().enumerate() {
            state.summary.insert(part as u64, raw);
        }
        for (part, raw) in reasoning_array(item, "content").into_iter().enumerate() {
            state.content.insert(part as u64, raw);
        }
        // A completed item owns its cipher/signature; terminal restatements
        // fill missing fields instead of replacing that first authority.
        if !state.complete || state.encrypted.is_none() {
            state.encrypted =
                nonempty(item.str("encrypted_content")).or_else(|| state.encrypted.clone());
        }
        if !state.complete || state.signature.is_none() {
            state.signature = nonempty(item.str("signature")).or_else(|| state.signature.clone());
        }
        if state.item.is_none() || !state.complete {
            state.item = Some(item.clone());
        }
        state.complete |= completed;
        Ok(())
    }

    fn call_added(
        &mut self,
        index: u64,
        item: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        // Custom input is meaningful only in its whole done item. The original
        // custom-input delta/done tags remain on the unknown channel.
        if item.str("type") == Some("custom_tool_call") {
            return Ok(());
        }
        self.active_message_part = None;
        if self.finished_calls.contains(&index) {
            return Ok(());
        }
        if let Some(id) = item.str("id").filter(|id| !id.is_empty()) {
            self.call_items.entry(id.to_owned()).or_insert(index);
        }
        if let Some(call) = self.calls.get_mut(&index) {
            if call.name.is_empty()
                && let Some(name) = item.str("name").filter(|name| !name.is_empty())
            {
                name.clone_into(&mut call.name);
                if let Some(id) = item.str("call_id").filter(|id| !id.is_empty()) {
                    call.id = CallId::from_wire(id);
                }
                if call.item_id.is_none() {
                    call.item_id = nonempty(item.str("id"));
                }
                call.namespace = item.str("namespace").map(str::to_owned);
                out.qualify(call.at, call.namespace.clone())?;
                out.fragment(
                    Some(call.at),
                    CallFragment {
                        id: item.str("call_id").filter(|id| !id.is_empty()),
                        name: Some(&call.name),
                        arguments: None,
                    },
                )?;
            }
            return Ok(());
        }
        let at = out.fresh_index();
        let id = CallId::from_wire(item.str("call_id").unwrap_or_default());
        let name = item.str("name").unwrap_or_default().to_owned();
        let namespace = item.str("namespace").map(str::to_owned);
        // Fragmented calls allow the first delta to precede an announced name.
        out.in_source_order(SourceOrder::new(index, 0), |out| {
            let fragment = CallFragment {
                id: item.str("call_id").filter(|id| !id.is_empty()),
                name: None,
                arguments: None,
            };
            if name.is_empty() {
                out.pending_call(at, fragment)
            } else {
                out.fragment(Some(at), fragment)
            }
        })?;
        out.qualify(at, namespace.clone())?;
        out.fragment(
            Some(at),
            CallFragment {
                name: Some(&name),
                ..CallFragment::default()
            },
        )?;
        self.calls.insert(
            index,
            CallState {
                at,
                id,
                item_id: nonempty(item.str("id")),
                name,
                namespace,
                arguments: String::new(),
                sent: String::new(),
                overflowed: false,
                ending: CallEnding::Pending,
            },
        );
        Ok(())
    }

    fn call_delta(
        &mut self,
        index: u64,
        id: Option<&str>,
        delta: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.active_message_part = None;
        if self.finished_calls.contains(&index)
            || self
                .calls
                .get(&index)
                .is_some_and(|call| call.ending.is_complete())
        {
            return Ok(());
        }
        if !self.calls.contains_key(&index) {
            self.call_added(
                index,
                &json!({"type":"function_call", "id":id.unwrap_or_default()}),
                out,
            )?;
        }
        let call = self.calls.get_mut(&index).expect("call state");
        // Mirror the accumulator's bounded fragment rule, including the
        // gateway's replaceable standalone null placeholder.
        if call.arguments.trim() == "null" && !delta.trim().is_empty() {
            call.arguments.clear();
            call.sent.clear();
        }
        if call.arguments.len().saturating_add(delta.len()) > MAX_TOOL_INPUT_BYTES {
            call.overflowed = true;
        } else {
            call.arguments.push_str(delta);
            call.sent.push_str(delta);
        }
        out.fragment(
            Some(call.at),
            CallFragment {
                arguments: Some(delta),
                ..CallFragment::default()
            },
        )
    }

    fn call_done(
        &mut self,
        index: u64,
        item: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = item
            .str("id")
            .filter(|id| !id.is_empty())
            .and_then(|id| self.call_items.get(id).copied())
            .unwrap_or(index);
        if self.finished_calls.contains(&index)
            || self
                .calls
                .get(&index)
                .is_some_and(|call| call.ending.is_complete())
        {
            return Ok(());
        }
        if let Some(id) = item.str("id").filter(|id| !id.is_empty()) {
            self.call_items.entry(id.to_owned()).or_insert(index);
        }
        if item.str("type") == Some("custom_tool_call") {
            self.active_message_part = None;
            let at = out.fresh_index();
            let id = CallId::from_wire(item.str("call_id").unwrap_or_default());
            let name = item.str("name").unwrap_or_default();
            let name =
                ToolName::new(name).map_err(|error| ProviderError::Response(error.to_string()))?;
            let namespace = item.str("namespace").map(str::to_owned);
            let input = item.str("input").expect("strict custom input");
            out.in_source_order(SourceOrder::new(index, 0), |out| {
                out.open(
                    at,
                    Block::CustomCall {
                        id: id.clone(),
                        name,
                        namespace: namespace.clone(),
                    },
                    item.clone(),
                )
            })?;
            out.push(at, input)?;
            self.calls.insert(
                index,
                CallState {
                    at,
                    id,
                    item_id: nonempty(item.str("id")),
                    name: item.str("name").unwrap_or_default().to_owned(),
                    namespace,
                    arguments: input.to_owned(),
                    sent: input.to_owned(),
                    overflowed: false,
                    ending: CallEnding::Custom,
                },
            );
        } else {
            if !self.calls.contains_key(&index) {
                self.call_added(index, item, out)?;
            }
            let call = self.calls.get_mut(&index).expect("call state");
            if call.name.is_empty() {
                call.name = item.str("name").unwrap_or_default().to_owned();
            }
            if let Some(id) = item.str("call_id").filter(|id| !id.is_empty()) {
                call.id = CallId::from_wire(id);
            }
            if call.item_id.is_none() {
                call.item_id = nonempty(item.str("id"));
            }
            call.namespace = item.str("namespace").map(str::to_owned);
            out.qualify(call.at, call.namespace.clone())?;
            out.fragment(
                Some(call.at),
                CallFragment {
                    id: item.str("call_id").filter(|id| !id.is_empty()),
                    name: Some(&call.name),
                    arguments: None,
                },
            )?;
            let stated = item.str("arguments").expect("strict function arguments");
            // Empty and whitespace arguments mean the parameterless object.
            let arguments = if stated.trim().is_empty() {
                "{}"
            } else {
                stated
            };
            call.ending = CallEnding::Function(match serde_json::from_str::<Value>(arguments) {
                Ok(_) => FunctionInput::Parsed,
                Err(error) => FunctionInput::Malformed {
                    error,
                    confidence: match item.str("status") {
                        Some("incomplete") => CallConfidence::Incomplete,
                        Some("completed" | "in_progress") => CallConfidence::Asserted,
                        _ => unreachable!("strict function-item status"),
                    },
                },
            });
            call.arguments = arguments.to_owned();
            // A done item states the complete input; it is not an arguments
            // delta. Retain earlier real fragments without inventing another.
            out.restate(call.at, arguments)?;
            call.sent = arguments.to_owned();
            // The authoritative item wins over any overflowed fragments.
            call.overflowed = false;
            out.edit(call.at, |native| *native = item.clone())?;
        }
        if self.options.immediate_tool_calls {
            self.finish_call(index, FlushBoundary::Partial, out)?;
        }
        Ok(())
    }

    fn finish_call(
        &mut self,
        index: u64,
        boundary: FlushBoundary,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        // A terminal cannot invent a tool name or close an unnamed slot.
        // Its real Part and already delivered arguments remain a prefix.
        if self
            .calls
            .get(&index)
            .is_some_and(|call| call.name.is_empty())
        {
            return Ok(());
        }
        if boundary == FlushBoundary::Partial
            && self
                .calls
                .get(&index)
                .is_some_and(|call| !call.ending.is_complete())
        {
            return Ok(());
        }
        let Some(call) = self.calls.remove(&index) else {
            return Ok(());
        };
        self.finished_calls.insert(index);
        let complete = call.ending.is_complete();
        match call.ending {
            CallEnding::Custom => return out.finish(call.at),
            CallEnding::Function(FunctionInput::Malformed {
                confidence: CallConfidence::Incomplete,
                ..
            }) => return out.discard_call(call.at, DiscardReason::Incomplete),
            CallEnding::Function(FunctionInput::Malformed {
                error,
                confidence: CallConfidence::Asserted,
            }) => {
                let provider = call.id.provider().map(|call_id| ProviderCallEvidence {
                    call_id: call_id.clone(),
                    item_id: call.item_id,
                });
                return Err(out.malformed_call_error(
                    call.at,
                    call.arguments,
                    error.to_string(),
                    provider,
                ));
            }
            CallEnding::Pending => {
                let arguments = if call.arguments.trim().is_empty() {
                    "{}"
                } else {
                    &call.arguments
                };
                if call.overflowed || serde_json::from_str::<Value>(arguments).is_err() {
                    // Genuine terminal evidence closes this never-done assembly;
                    // neither item confidence nor a truncation cause is inferred.
                    return out.discard_call(call.at, DiscardReason::UnfinishedAtTerminal);
                }
            }
            CallEnding::Function(FunctionInput::Parsed) => {}
        }
        if call.arguments.trim().is_empty() {
            out.restate(call.at, "{}")?;
        } else if serde_json::from_str::<Value>(&call.arguments)
            .is_ok_and(|arguments| !arguments.is_object())
        {
            // This carried boundary preserves F's valid JSON values without
            // changing the default object-oriented ToolFunction parser.
            out.preserve_non_object_call_arguments(call.at)?;
        }
        if complete {
            out.finish(call.at)
        } else {
            out.close(call.at)
        }
    }

    fn provider_item(
        &mut self,
        index: u64,
        item: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if !self.provider_items.insert(index) {
            return Ok(());
        }
        let at = out.fresh_index();
        out.in_source_order(SourceOrder::new(index, 0), |out| {
            out.whole(at, Block::ProviderItem { provider: None }, item.clone(), "")
        })
    }

    fn output(
        &mut self,
        index: u64,
        item: &Value,
        completed: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match item.str("type") {
            Some("message") => self.message_item(index, item, completed, out),
            Some("reasoning") => self.reasoning_item(index, item, completed, out),
            Some("function_call" | "custom_tool_call") if completed => {
                self.call_done(index, item, out)
            }
            Some("function_call" | "custom_tool_call") => self.call_added(index, item, out),
            _ if completed => self.provider_item(index, item, out),
            _ => Ok(()),
        }
    }

    fn item_event(
        &mut self,
        kind: &str,
        frame: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = frame.u64("output_index").expect("strict output index");
        let id = frame.str("item_id");
        let index = id
            .and_then(|id| self.message_ids.get(id))
            .copied()
            .unwrap_or(index);
        let part = frame.u64("content_index").unwrap_or(0);
        match kind {
            "response.output_item.added" => self.output(index, &frame["item"], false, out),
            "response.output_item.done" => {
                self.active_message_part = None;
                self.output(index, &frame["item"], true, out)
            }
            "response.output_text.delta" | "response.refusal.delta" => self.message_delta(
                index,
                part,
                id,
                frame.str("delta").expect("strict delta"),
                kind == "response.refusal.delta",
                out,
            ),
            "response.content_part.added" | "response.content_part.done" => {
                let raw = frame["part"].clone();
                match raw.str("type") {
                    Some("output_text" | "refusal") => {
                        let index = self.message_slot(index, id, None);
                        self.check_message_kind(index, part, false, id)?;
                        self.bind(index, Parent::Message, id, out)?;
                        self.message_part(index, part, id, raw, out)
                    }
                    Some("summary_text") => {
                        self.bind(index, Parent::Reasoning, id, out)?;
                        self.reasoning_part(index, part, true, id, raw, false, out)
                    }
                    Some("reasoning_text") => {
                        self.bind(index, Parent::Reasoning, id, out)?;
                        self.reasoning_part(index, part, false, id, raw, false, out)
                    }
                    _ => match self.parents.get(&index) {
                        Some(Parent::Message) => self.message_part(index, part, id, raw, out),
                        Some(Parent::Reasoning) => {
                            self.reasoning_part(index, part, false, id, raw, false, out)
                        }
                        None => {
                            let waiting = UndecidedPart {
                                id: id.map(str::to_owned),
                                raw,
                            };
                            let parts = self.undecided.entry((index, part)).or_default();
                            if !parts.contains(&waiting) {
                                parts.push(waiting);
                            }
                            Ok(())
                        }
                    },
                }
            }
            "response.reasoning_summary_part.added" | "response.reasoning_summary_part.done" => {
                self.bind(index, Parent::Reasoning, id, out)?;
                self.reasoning_part(
                    index,
                    frame.u64("summary_index").expect("strict summary index"),
                    true,
                    id,
                    frame["part"].clone(),
                    false,
                    out,
                )
            }
            "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done" => {
                let summary = kind.starts_with("response.reasoning_summary");
                let append = kind.ends_with(".delta");
                let part = if summary {
                    frame.u64("summary_index").expect("strict summary index")
                } else {
                    part
                };
                let text = frame
                    .str(if append { "delta" } else { "text" })
                    .expect("strict reasoning text");
                self.bind(index, Parent::Reasoning, id, out)?;
                self.reasoning_part(index, part, summary, id, json!({"type":if summary { "summary_text" } else { "reasoning_text" }, "text":text}), append, out)
            }
            "response.function_call_arguments.delta" => self.call_delta(
                index,
                id,
                frame.str("delta").expect("strict arguments delta"),
                out,
            ),
            // These done text/argument fields are typed observations but their
            // output-item restatement is the owning completion authority.
            "response.output_text.done"
            | "response.refusal.done"
            | "response.function_call_arguments.done" => Ok(()),
            _ => Ok(()),
        }
    }

    fn flush_messages(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        for (index, message) in std::mem::take(&mut self.messages) {
            let unknown_fields = message
                .item
                .as_ref()
                .map(super::unknown_message_fields)
                .unwrap_or_default();
            let carried = !unknown_fields.is_empty()
                || message.parts.len() != 1
                || message.phase.is_some()
                || message
                    .item
                    .as_ref()
                    .is_some_and(|item| item.str("phase").is_some())
                || message.parts.values().any(|part| {
                    part.opaque
                        || part.refusal
                        || part
                            .raw
                            .as_ref()
                            .and_then(Value::as_object)
                            .is_some_and(|fields| {
                                fields.keys().any(|key| key != "type" && key != "text")
                            })
                });
            for (part_index, mut part) in message.parts {
                if carried {
                    let mut raw = part.raw.take().unwrap_or_else(
                        || json!({"type":if part.refusal {"refusal"} else {"output_text"}}),
                    );
                    if !part.opaque {
                        raw[if part.refusal { "refusal" } else { "text" }] = json!(part.text);
                    }
                    let item = json!({"content":[raw], "id":message.id, "phase":message.phase});
                    let mut text = super::carried_message_parts(&item, index, None)?.remove(0);
                    text = text.with_unknown_message_fields(unknown_fields.clone());
                    text.payload.position = CarriedPosition::MessageContent {
                        item_index: index,
                        part_index,
                    };
                    let at = text_at(&mut part, index, part_index, out)?;
                    if !part.opaque {
                        out.set_citations(
                            at,
                            text.payload
                                .item
                                .arr("annotations")
                                .iter()
                                .filter_map(source_of)
                                .map(|source| WireCitation::new(None, vec![source]))
                                .collect(),
                        );
                    }
                    out.finish_carried_text(at, text)?;
                } else if let Some(at) = part.at {
                    if message.complete {
                        if let Some(item) = &message.item {
                            let mut item = item.clone();
                            if let Some(raw) = item
                                .get_mut("content")
                                .and_then(Value::as_array_mut)
                                .and_then(|parts| parts.first_mut())
                            {
                                raw["text"] = json!(part.text);
                            }
                            out.edit(at, |native| *native = item)?;
                            out.set_citations(
                                at,
                                message.item.as_ref().map(citations_of).unwrap_or_default(),
                            );
                        }
                        out.finish(at)?;
                    } else {
                        out.close(at)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn flush_reasoning(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        for (index, mut state) in std::mem::take(&mut self.reasoning) {
            // Typed part/done restatements alone are not live deltas or
            // completed items. Unknown parts retain their partial evidence.
            if !state.complete && state.at.is_none() && !state.has_opaque() {
                continue;
            }
            let item = state.raw();
            if state.carried() {
                // blank signature metadata has no canonical text contribution.
                // Retain the untouched item as evidence, including its original signature.
                let mut projection = item.clone();
                if projection.str("signature").is_some_and(|value| value.trim().is_empty())
                    && let Some(fields) = projection.as_object_mut()
                {
                    fields.remove("signature");
                }
                let mut reasoning = super::carried_reasoning(&projection, None)?;
                reasoning.payload.item = item.clone();
                let known_root = item.as_object().is_some_and(|fields| {
                    fields.keys().all(|key| matches!(key.as_str(),
                        "type" | "id" | "summary" | "content" | "encrypted_content"
                        | "signature" | "status"))
                });
                let empty_known = reasoning.id.as_deref().is_some_and(|id| !id.is_empty())
                    && known_root && reasoning.content.iter().all(|part| match part {
                        crate::message::CarriedReasoningPart::Summary(text) => text.is_empty(),
                        crate::message::CarriedReasoningPart::Text { text, signature } => {
                            text.is_empty() && signature.as_deref()
                                .is_none_or(|signature| signature.trim().is_empty())
                        }
                        _ => false,
                    });
                if empty_known {
                    if let Some(at) = state.at {
                        out.close(at)?;
                    }
                    continue;
                }
                let at = match state.at {
                    Some(at) => at,
                    None => {
                        let at = out.fresh_index();
                        out.in_source_order(SourceOrder::new(index, 0), |out| {
                            out.open(at, Block::Reasoning { redacted: false }, Value::Null)
                        })?;
                        at
                    }
                };
                out.restate(at, &reasoning.display_text())?;
                let authority = if state.complete {
                    ReasoningAuthority::ProviderRestated
                } else {
                    ReasoningAuthority::Partial
                };
                out.finish_carried_reasoning_with_authority(at, reasoning, authority)?;
            } else if let Some(at) = state.at.take() {
                if state.complete {
                    out.edit(at, |native| *native = item)?;
                    out.finish(at)?;
                } else {
                    out.close(at)?;
                }
            }
        }
        Ok(())
    }

    fn flush(
        &mut self,
        boundary: FlushBoundary,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        for (_, parts) in std::mem::take(&mut self.undecided) {
            for part in parts {
                out.unknown(part.raw.into());
            }
        }
        self.flush_messages(out)?;
        self.flush_reasoning(out)?;
        let mut first_error = None;
        for index in self.calls.keys().copied().collect::<Vec<_>>() {
            if let Err(error) = self.finish_call(index, boundary, out) {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn merge_reasoning(
        &mut self,
        response: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        for (index, item) in response.arr("output").iter().enumerate() {
            if item.str("type") == Some("reasoning") {
                self.reasoning_item(index as u64, item, true, out)?;
            }
        }
        Ok(())
    }

    /// retain observed buffered custody before the original local refusal.
    /// Flush errors never replace the first error or prove a provider terminal.
    fn refuse_buffered(
        &mut self,
        error: ProviderError,
        out: &mut Out<'_, Completion>,
    ) -> ProviderError {
        if !self.terminal_issuer {
            out.stamp_carried_issuer(&self.provider);
        }
        if let Err(flush_error) = self.flush(FlushBoundary::Partial, out) {
            self.pending_flush_error.get_or_insert(flush_error);
        }
        self.finished = true;
        error
    }

    fn terminal(
        &mut self,
        response: Value,
        whole: bool,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let terminal_message_id = response
            .arr("output")
            .iter()
            .find(|item| item.str("type") == Some("message"))
            .and_then(|item| nonempty(item.str("id")));
        if whole
            && self.reasoning.is_empty()
            && !response
                .arr("output")
                .iter()
                .any(|item| item.str("type") == Some("reasoning"))
            && let Some(text) = response.str("reasoning").filter(|text| !text.is_empty())
        {
            let at = out.fresh_index();
            out.whole_reasoning_with_authority(
                at,
                Value::Null,
                text,
                ReasoningAuthority::ProviderRestated,
            )?;
        }
        // Whole-response indices are source positions, not writer Part indices.
        let mut first_error = None;
        for (index, item) in response.arr("output").iter().enumerate() {
            // after a conflict, only independently validated later reasoning
            // may join the buffered prefix. No successful terminal is sealed.
            if first_error.is_some() && item.str("type") != Some("reasoning") {
                continue;
            }
            // A streamed terminal drains never-done calls from their actual
            // fragments. Whole-body decoding alone supplies whole calls.
            if whole || !matches!(item.str("type"), Some("function_call" | "custom_tool_call")) {
                if let Err(error) = self.output(index as u64, item, true, &mut out) {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(self.refuse_buffered(error, &mut out));
        }
        // Some gateways report only a top-level reasoning string.
        if !whole
            && self.reasoning.is_empty()
            && let Some(text) = response.str("reasoning").filter(|text| !text.is_empty())
        {
            let at = out.fresh_index();
            out.whole(at, Block::Reasoning { redacted: false }, Value::Null, text)?;
        }
        self.flush(FlushBoundary::GenuineTerminal, &mut out)?;
        let issuer = self.issuer(&response);
        out.stamp_carried_issuer(&issuer);
        self.terminal_issuer = true;
        let message_id = terminal_message_id.or_else(|| self.last_message_id.clone());
        let (reason, error) = finish_reason_of(&response);
        // A valid whole reply follows the fork's completed decoding path even
        // for failed/cancelled status. Its raw error stays in response evidence;
        // only an actual failed event supplies an in-band provider error.
        let error = if whole { None } else { error };
        self.finished = true;
        Ok(out.end(Finish {
            usage: response.get("usage").map(usage_of).unwrap_or_default(),
            reason,
            message_id: message_id.and_then(ProviderMessageId::new),
            response_id: nonempty(response.str("id")),
            model: nonempty(response.str("model")),
            error,
        }))
    }
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value.filter(|value| !value.is_empty()).map(str::to_owned)
}

fn conflicting_kind(
    output: u64,
    array: &str,
    index: u64,
    stored: bool,
    incoming: bool,
) -> ProviderError {
    ProviderError::Response(format!(
        "conflicting Responses content part kind at output {output}, {array} {index}: stored {} part restated as {}",
        if stored { "opaque" } else { "text" },
        if incoming { "opaque" } else { "text" }
    ))
}

fn part_matches(part: &MessagePart, raw: &Value) -> bool {
    match raw.str("type") {
        Some("output_text") => {
            !part.opaque && !part.refusal && raw.str("text") == Some(part.text.as_str())
        }
        Some("refusal") => {
            !part.opaque && part.refusal && raw.str("refusal") == Some(part.text.as_str())
        }
        _ => part.opaque && part.raw.as_ref() == Some(raw),
    }
}

fn text_at(
    part: &mut MessagePart,
    index: u64,
    part_index: u64,
    out: &mut Out<'_, Completion>,
) -> Result<usize, ProviderError> {
    if let Some(at) = part.at {
        return Ok(at);
    }
    let at = out.fresh_index();
    out.in_source_order(SourceOrder::new(index, part_index), |out| {
        out.open(at, Block::Text, Value::Null)
    })?;
    part.at = Some(at);
    Ok(at)
}

/// A provider return reactivates the existing live Part; a first fragment
/// continues to let the default writer allocate its one Start.
fn activate_text(
    active: &mut Option<usize>,
    at: usize,
    started: bool,
    out: &mut Out<'_, Completion>,
) -> Result<(), ProviderError> {
    if *active != Some(at) && started {
        out.resume_text(at)?;
    }
    *active = Some(at);
    Ok(())
}

/// F's text_block metadata comparison, used only to decide activation.
/// Raw sibling bytes stay intact in MessagePart and carried canonical text.
fn message_activation_metadata(raw: Option<&Value>) -> (bool, serde_json::Map<String, Value>) {
    let refusal = raw.and_then(|raw| raw.str("type")) == Some("refusal");
    let extras = raw
        .filter(|raw| raw.str("type") == Some("output_text"))
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|fields| fields.iter())
        .filter(|(key, value)| {
            key.as_str() != "type"
                && key.as_str() != "text"
                && !value.is_null()
                && !value.as_array().is_some_and(Vec::is_empty)
                && !value.as_object().is_some_and(serde_json::Map::is_empty)
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    (refusal, extras)
}

fn reasoning_array(item: &Value, field: &str) -> Vec<Value> {
    match item.get(field) {
        Some(Value::String(text)) => vec![json!({"type":"reasoning_text", "text":text})],
        Some(Value::Array(parts)) => parts.clone(),
        _ => Vec::new(),
    }
}

fn reasoning_opaque(part: &Value, summary: bool) -> bool {
    if !summary && part.is_string() {
        return false;
    }
    part.str("type")
        != Some(if summary {
            "summary_text"
        } else {
            "reasoning_text"
        })
}

/// Preserve the provider's exact incomplete reason without synthesizing a
/// failure for an explicitly accepted partial terminal.
pub(crate) fn finish_reason_of(response: &Value) -> (Option<FinishReason>, Option<String>) {
    let reason = match response.str("status") {
        Some("completed") => Some(FinishReason::Stop),
        Some("incomplete") => Some(
            match response
                .at("/incomplete_details/reason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty())
            {
                Some("max_output_tokens") => FinishReason::Length,
                Some("content_filter") => FinishReason::ContentFilter,
                Some(reason) => FinishReason::Other(reason.to_owned()),
                None => FinishReason::Other("incomplete".to_owned()),
            },
        ),
        Some("queued" | "in_progress") | None | Some("") => None,
        Some(status) => Some(FinishReason::Other(status.to_owned())),
    };
    let error = match response.str("status") {
        Some(status @ ("failed" | "cancelled")) => {
            let parts: Vec<_> = ["/error/code", "/error/message"]
                .iter()
                .filter_map(|key| response.at(key).and_then(Value::as_str))
                .collect();
            Some(if parts.is_empty() {
                format!("Response {status}")
            } else {
                parts.join(": ")
            })
        }
        _ => None,
    };
    (reason, error)
}

impl<'id> Decoder<'id, Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesEvent> {
        self.reader_policy.classify(&frame.as_str())
    }

    fn decode(
        &mut self,
        event: ResponsesEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        if self.finished {
            return Ok(Flow::More);
        }
        match event {
            ResponsesEvent::Frame { kind, frame, raw } => match kind.as_str() {
                "response.completed" => self.terminal(frame["response"].clone(), false, out),
                "response.incomplete" if self.options.incomplete == IncompleteTerminal::Accept => {
                    let mut response = frame["response"].clone();
                    response["status"] = json!("incomplete");
                    self.terminal(response, false, out)
                }
                "response.incomplete" | "response.failed" => {
                    // Refused terminals retain only already observed content,
                    // complete calls and their final reasoning authority.
                    if let Err(error) = self.merge_reasoning(&frame["response"], &mut out) {
                        self.pending_flush_error = Some(error);
                    }
                    if let Err(error) = self.flush(FlushBoundary::Partial, &mut out) {
                        self.pending_flush_error.get_or_insert(error);
                    }
                    self.finished = true;
                    Err(ProviderError::from_provider_body(raw))
                }
                kind if is_lifecycle_event(kind) => Ok(Flow::More),
                kind => {
                    match self.item_event(kind, frame, &mut out) {
                        Ok(()) => Ok(Flow::More),
                        Err(error) => Err(self.refuse_buffered(error, &mut out)),
                    }
                }
            },
            ResponsesEvent::Whole(body) => self.terminal(body, true, out),
            ResponsesEvent::Failure(raw) => {
                if let Err(error) = self.flush(FlushBoundary::Partial, &mut out) {
                    self.pending_flush_error = Some(error);
                }
                self.finished = true;
                Err(ProviderError::from_provider_body(raw))
            }
            ResponsesEvent::Sentinel => Ok(Flow::More),
        }
    }

    fn flush_before_terminal_error(
        &mut self,
        mut out: Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        if !self.terminal_issuer {
            out.stamp_carried_issuer(&self.provider);
        }
        let flushed = self.flush(FlushBoundary::Partial, &mut out);
        match self.pending_flush_error.take() {
            Some(error) => Err(error),
            None => flushed,
        }
    }

    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        if !self.terminal_issuer {
            out.stamp_carried_issuer(&self.provider);
        }
        self.flush(FlushBoundary::Partial, &mut out)?;
        Ok(Flow::More)
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        IncompleteEof::KeepPartial
    }
}

/// The decoder and its strict reassembler share one acceptance rule.
fn classify_reader_payload(data: &str, repair_envelopes: bool) -> WireEvent<ResponsesEvent> {
    if !repair_envelopes {
        return classify_responses_payload(data);
    }
    let mut classified = wire::classify_with_repair(
        data,
        classify_responses_payload,
        reader::repair,
        |error| {
            <serde_json::Error as serde::de::Error>::custom(format!(
                "invalid JSON frame in buffered Responses SSE body: {error}"
            ))
        },
        || {
            let kind = serde_json::from_str::<Value>(data)
                .ok()
                .and_then(|value| nonempty(value.str("type")))
                .unwrap_or_else(|| "unknown".to_owned());
            <serde_json::Error as serde::de::Error>::custom(format!(
                "malformed `{kind}` event in buffered Responses SSE body"
            ))
        },
    );
    // Typed repair never replaces the provider's received refusal evidence.
    if let WireEvent::Known(ResponsesEvent::Frame { raw, .. }) = &mut classified {
        *raw = data.to_owned();
    }
    classified
}

#[cfg(test)]
mod authority_tests;
#[cfg(test)]
mod carried_reader_tests;
#[cfg(test)]
mod citation_tests;
pub(crate) mod document;
#[cfg(feature = "completion-observations")]
pub mod observed;
#[cfg(test)]
mod pending_tests;
#[cfg(test)]
mod resume_tests;
#[cfg(test)]
mod tests;
