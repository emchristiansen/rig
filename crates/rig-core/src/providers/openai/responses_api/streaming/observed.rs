use super::super::ToolStatus;
use super::super::{observed_types as ot, observed_wire as ow};
use crate::providers::internal::wire::{self, WireEvent};
use crate::error::{ErrorReport, ProviderError};
use ow::FunctionCallArguments;
use ow::{Output, ReasoningSummary};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A native error, separate from observation custody and capture state.
#[derive(Debug, thiserror::Error)]
pub enum ObservedInterpretationError {
    /// The provider decoder or finalization failed.
    #[error("{0}")]
    NativeProvider(#[from] ProviderError),
    /// The normalized fold refused completed content.
    #[error("{0:?}")]
    NativeReport(ErrorReport),
}

/// Incremental caller-visible information; original payloads live on the handle.
#[derive(Clone, Debug)]
pub enum ObservedEvent {
    /// Newly delivered text, including a terminal restatement's new suffix.
    TextDelta { text: String },
    /// A forward-compatible frame, not normalized assistant output.
    Unknown { event_type: String, value: Value },
}

/// Normalized response and its separate provider-native typed view.
/// The normalized response contains stream terminal metadata in `raw`.
/// The observation handle separately retains the original provider strings.
#[derive(Clone, Debug)]
pub struct ObservedResponsesResultBody {
    pub response: ot::CompletionResponse,
    pub native: ow::CompletionResponse,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum Chunk {
    Response(ResponseChunk),
    Delta(ItemChunk),
}
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ResponseChunk {
    #[serde(rename = "type")]
    kind: super::ResponseChunkKind,
    response: ow::CompletionResponse,
    sequence_number: u64,
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

/// Decode a known tag only when its text field (`refusal` for a refusal part,
/// `text` otherwise) is a string, and preserve unknown tags.
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
    /// `OutputFunctionCall::arguments`;
    /// `FunctionCallArguments::reconcile` settles whether the two agree.
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

enum Payload {
    Frame { raw: String, chunk: Chunk },
    Whole(Box<ow::CompletionResponse>),
    Failure(String),
    Sentinel,
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    #[allow(dead_code)]
    error: Value,
}

fn classify_or<T>(
    data: &str,
    first: impl Fn(&str) -> WireEvent<T>,
    then: impl Fn(&str) -> WireEvent<T>,
) -> WireEvent<T> {
    match first(data) {
        WireEvent::Corrupt(first_error) => match then(data) {
            WireEvent::Known(event) => WireEvent::Known(event),
            WireEvent::Corrupt(error) => WireEvent::Corrupt(error),
            WireEvent::Unknown { .. } => WireEvent::Corrupt(first_error),
        },
        event => event,
    }
}

fn classify_payload(data: &str) -> WireEvent<Payload> {
    if data.trim() == "[DONE]" {
        return WireEvent::Known(Payload::Sentinel);
    }
    let parsed = serde_json::from_str::<Value>(data);
    if parsed
        .as_ref()
        .is_ok_and(|v| v.get("type").and_then(Value::as_str) == Some("error"))
    {
        return WireEvent::Known(Payload::Failure(data.to_owned()));
    }
    let tagged = |data: &str| {
        wire::classify_tagged_frame::<Chunk>(data, "type", super::is_known_responses_event_type)
            .map(|chunk| Payload::Frame {
                raw: data.to_owned(),
                chunk,
            })
    };
    if parsed.as_ref().is_ok_and(|v| v.get("type").is_some()) {
        return tagged(data);
    }
    classify_or(data, tagged, |data| {
        classify_or(
            data,
            |data| {
                wire::classify_marker_keyed_frame::<ow::CompletionResponse>(
                    data,
                    &["object", "output", "status", "error"],
                )
                .map(|response| Payload::Whole(Box::new(response)))
            },
            |data| {
                wire::classify_marker_keyed_frame::<ErrorEnvelope>(data, &["error"])
                    .map(|_| Payload::Failure(data.to_owned()))
            },
        )
    })
}

fn classify(data: &str) -> WireEvent<Payload> {
    wire::classify_with_repair(
        data,
        classify_payload,
        |data| {
            let mut value = serde_json::from_str::<Value>(data).ok()?;
            let object = value.as_object_mut()?;
            for field in [
                "sequence_number",
                "output_index",
                "content_index",
                "summary_index",
            ] {
                object.entry(field).or_insert_with(|| Value::from(0));
            }
            serde_json::to_string(&value).ok()
        },
        |error| {
            <serde_json::Error as serde::de::Error>::custom(format!(
                "invalid JSON frame in buffered Responses SSE body: {error}"
            ))
        },
        || {
            let kind = serde_json::from_str::<Value>(data)
                .ok()
                .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_default();
            <serde_json::Error as serde::de::Error>::custom(format!(
                "malformed `{kind}` event in buffered Responses SSE body"
            ))
        },
    )
}

fn json_subsumes(outer: &Value, inner: &Value) -> bool {
    match (outer, inner) {
        (Value::Object(outer), Value::Object(inner)) => inner.iter().all(|(key, value)| {
            outer
                .get(key)
                .is_some_and(|outer| json_subsumes(outer, value))
        }),
        (outer, inner) => outer == inner,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Parent {
    Message,
    Reasoning,
}

#[derive(Clone)]
struct MessagePart {
    order: u64,
    text: ot::Text,
    opaque: bool,
    refusal: bool,
    phase: Option<String>,
}

#[derive(Clone, PartialEq)]
struct UndecidedPart {
    item_id: Option<String>,
    value: Value,
}

#[derive(Default)]
struct ReasoningParts {
    order: u64,
    id: Option<String>,
    summary: BTreeMap<u64, ow::ReasoningSummary>,
    content: BTreeMap<u64, ow::ReasoningTextContent>,
    encrypted: Option<String>,
    signature: Option<String>,
    wire_sent: bool,
    ready: bool,
    opened: bool,
}

struct ToolParts {
    order: u64,
    minted: u64,
    item_id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    namespace: Option<String>,
    arguments: Option<String>,
    overflowed: bool,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ItemKey {
    Wire(String),
    Minted(u64),
}

enum PendingCall {
    Function(ow::OutputFunctionCall, Option<ToolParts>),
    Custom(ow::OutputCustomToolCall),
}

/// One explicitly observed ChatGPT Responses operation. The source passes each
/// original supplied string here only after committing it to its observation ledger.
pub struct ObservedAssembler {
    provider: String,
    messages: BTreeMap<(u64, u64), MessagePart>,
    message_slots: HashMap<String, u64>,
    unattributed: BTreeSet<u64>,
    delta_messages: BTreeSet<u64>,
    parents: HashMap<u64, Parent>,
    undecided: BTreeMap<(u64, u64), Vec<UndecidedPart>>,
    reasoning: BTreeMap<u64, ReasoningParts>,
    finished_reasoning: BTreeSet<u64>,
    tools: BTreeMap<u64, ToolParts>,
    unclosed_tools: BTreeMap<u64, ToolParts>,
    completed: BTreeMap<(u64, u64, u64), ot::AssistantContent>,
    next_content: u64,
    provider_slots: BTreeSet<u64>,
    provider_keys: BTreeSet<ItemKey>,
    finished_tools: BTreeSet<ItemKey>,
    next_tool: u64,
    pending_calls: Vec<(u64, PendingCall)>,
    prefix: Vec<ot::AssistantContent>,
    terminal: Option<ow::CompletionResponse>,
    terminal_usage: Option<ow::ResponsesUsage>,
    terminal_response_id: Option<String>,
    terminal_model: Option<String>,
    terminal_message_id: Option<String>,
    // Selected terminal metadata projection; original frames belong to the handle.
    document: serde_json::Map<String, Value>,
    message_id: Option<String>,
    provider_request_id: Option<String>,
    response_headers: BTreeMap<String, String>,
    stopped: bool,
    whole_finished: bool,
}

impl ObservedAssembler {
    /// Create the strict-streaming interpretation, with Codex envelope repair.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            messages: BTreeMap::new(),
            message_slots: HashMap::new(),
            unattributed: BTreeSet::new(),
            delta_messages: BTreeSet::new(),
            parents: HashMap::new(),
            undecided: BTreeMap::new(),
            reasoning: BTreeMap::new(),
            finished_reasoning: BTreeSet::new(),
            tools: BTreeMap::new(),
            unclosed_tools: BTreeMap::new(),
            completed: BTreeMap::new(),
            next_content: 0,
            provider_slots: BTreeSet::new(),
            provider_keys: BTreeSet::new(),
            finished_tools: BTreeSet::new(),
            next_tool: 0,
            pending_calls: Vec::new(),
            prefix: Vec::new(),
            terminal: None,
            terminal_usage: None,
            terminal_response_id: None,
            terminal_model: None,
            terminal_message_id: None,
            document: serde_json::Map::new(),
            message_id: None,
            provider_request_id: None,
            response_headers: BTreeMap::new(),
            stopped: false,
            whole_finished: false,
        }
    }

    /// Attach metadata observed on this operation's successful connection.
    pub fn set_response_metadata(
        &mut self,
        request_id: Option<String>,
        headers: BTreeMap<String, String>,
    ) {
        self.provider_request_id = request_id.filter(|id| !id.is_empty());
        self.response_headers = headers;
    }

    /// Interpret an already captured supplied SSE data string, without I/O.
    /// Events emitted before an error remain in `out`.
    pub fn push(
        &mut self,
        data: &str,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        if self.stopped || self.whole_finished || data.trim().is_empty() {
            return Ok(());
        }
        let result = self.push_inner(data, out);
        if result.is_err() {
            self.stopped = true;
        }
        result
    }

    fn push_inner(
        &mut self,
        data: &str,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        match classify(data) {
            WireEvent::Unknown { event_type, value } => out.push(ObservedEvent::Unknown {
                event_type,
                value: value.value().clone(),
            }),
            WireEvent::Corrupt(error) => {
                #[derive(Deserialize)]
                struct Discriminator {
                    #[serde(rename = "type")]
                    kind: Option<String>,
                }
                // Failed discriminator scans establish no event type.
                let event_type = serde_json::from_str::<Discriminator>(data)
                    .ok()
                    .and_then(|tag| tag.kind);
                return Err(ProviderError::CorruptFrame(
                    crate::error::CorruptFrame::text(
                        event_type, data, error,
                    ),
                )
                .into());
            }
            WireEvent::Known(Payload::Sentinel) => {}
            WireEvent::Known(Payload::Failure(raw)) => {
                self.flush_pending_calls()?;
                return Err(ProviderError::from_provider_body(raw).into());
            }
            WireEvent::Known(Payload::Frame {
                raw,
                chunk: Chunk::Response(chunk),
            }) => match chunk.kind {
                super::ResponseChunkKind::ResponseFailed
                | super::ResponseChunkKind::ResponseIncomplete => {
                    self.merge_failed_reasoning(&chunk.response);
                    self.flush_pending_calls()?;
                    return Err(ProviderError::from_provider_body(raw).into());
                }
                super::ResponseChunkKind::ResponseCompleted => {
                    self.complete(chunk.response, out)?
                }
                _ => {}
            },
            WireEvent::Known(Payload::Frame {
                chunk: Chunk::Delta(chunk),
                ..
            }) => self.item(chunk, out)?,
            WireEvent::Known(Payload::Whole(response)) => {
                if !response
                    .output
                    .iter()
                    .any(|item| matches!(item, Output::Reasoning { .. }))
                {
                    if let Some(text) = response
                        .provider_reasoning
                        .as_ref()
                        .filter(|text| !text.is_empty())
                    {
                        // Whole-body top-level reasoning precedes output items.
                        self.prefix
                            .push(ot::AssistantContent::Reasoning(ot::Reasoning {
                                id: None,
                                content: vec![ot::ReasoningContent::Text {
                                    text: text.clone(),
                                    signature: None,
                                }],
                                provider: Some(self.provider.clone()),
                            }));
                    }
                }
                for (index, item) in response.output.iter().cloned().enumerate() {
                    self.done_immediately(index as u64, item, out)?;
                }
                self.complete(*response, out)?;
                self.flush_pending_calls()?;
                self.flush_unclosed_tools();
                self.whole_finished = true;
            }
        }
        Ok(())
    }

    fn failure(message: String) -> ObservedInterpretationError {
        ProviderError::Response(message).into()
    }
    fn conflict(
        output: u64,
        array: &str,
        index: u64,
        stored: bool,
        incoming: bool,
    ) -> ObservedInterpretationError {
        Self::failure(format!(
            "conflicting Responses content part kind at output {output}, {array} {index}: stored {} part restated as {}",
            if stored { "opaque" } else { "text" },
            if incoming { "opaque" } else { "text" }
        ))
    }
    fn waiting_binds(&self, key: (u64, u64), id: Option<&str>) -> bool {
        self.undecided.get(&key).is_some_and(|parts| {
            parts
                .iter()
                .any(|p| p.item_id.as_deref().is_none_or(|own| id == Some(own)))
        })
    }
    fn message_conflict(
        &self,
        slot: u64,
        index: u64,
        opaque: bool,
        id: Option<&str>,
    ) -> Result<(), ObservedInterpretationError> {
        let prior = self
            .messages
            .get(&(slot, index))
            .map(|p| p.opaque)
            .or_else(|| self.waiting_binds((slot, index), id).then_some(true));
        if let Some(prior) = prior.filter(|prior| *prior != opaque) {
            return Err(Self::conflict(slot, "content", index, prior, opaque));
        }
        Ok(())
    }
    fn take_undecided(&mut self, slot: u64, id: Option<&str>) -> Vec<(u64, Value)> {
        let keys: Vec<_> = self
            .undecided
            .range((slot, 0)..=(slot, u64::MAX))
            .map(|(k, _)| *k)
            .collect();
        let mut taken = Vec::new();
        for key in keys {
            if let Some(parts) = self.undecided.remove(&key) {
                let (binding, waiting): (Vec<_>, Vec<_>) = parts
                    .into_iter()
                    .partition(|p| p.item_id.as_deref().is_none_or(|own| id == Some(own)));
                if !waiting.is_empty() {
                    self.undecided.insert(key, waiting);
                }
                taken.extend(binding.into_iter().map(|p| (key.1, p.value)));
            }
        }
        taken
    }
    fn rekey_undecided(&mut self, id: &str, slot: u64) {
        let keys: Vec<_> = self
            .undecided
            .iter()
            .filter(|((at, _), parts)| {
                *at != slot && parts.iter().any(|p| p.item_id.as_deref() == Some(id))
            })
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(parts) = self.undecided.remove(&key) {
                let (moving, waiting): (Vec<_>, Vec<_>) = parts
                    .into_iter()
                    .partition(|p| p.item_id.as_deref() == Some(id));
                if !waiting.is_empty() {
                    self.undecided.insert(key, waiting);
                }
                let target = self.undecided.entry((slot, key.1)).or_default();
                for p in moving {
                    if !target.contains(&p) {
                        target.push(p);
                    }
                }
            }
        }
    }
    fn restated(part: &MessagePart, content: &ow::AssistantContent) -> bool {
        match content {
            ow::AssistantContent::Unknown(value) => {
                ow::opaque_message_part(part.text.additional_params.as_ref()) == Some(value)
            }
            ow::AssistantContent::Refusal { refusal } => {
                part.refusal && part.text.text == *refusal
            }
            ow::AssistantContent::OutputText(text) => {
                !part.refusal && !part.opaque && part.text.text == text.text
            }
        }
    }
    fn slot(
        &mut self,
        index: u64,
        id: Option<&str>,
        restatement: Option<(Option<u64>, &[ow::AssistantContent])>,
    ) -> u64 {
        let Some(id) = id.filter(|id| !id.is_empty()) else {
            return index;
        };
        if let Some(slot) = self.message_slots.get(id) {
            return *slot;
        }
        let empty = self
            .messages
            .range((index, 0)..=(index, u64::MAX))
            .next()
            .is_none()
            && !self
                .undecided
                .range((index, 0)..=(index, u64::MAX))
                .any(|(_, parts)| {
                    parts
                        .iter()
                        .any(|p| p.item_id.as_deref().is_none_or(|own| own == id))
                });
        let found = if empty {
            restatement.and_then(|(part_index, content)| {
                self.unattributed.iter().copied().find(|slot| {
                    let mut stored = self
                        .messages
                        .range((*slot, 0)..=(*slot, u64::MAX))
                        .peekable();
                    if stored.peek().is_none() {
                        return false;
                    }
                    if let Some(at) = part_index {
                        return self
                            .messages
                            .get(&(*slot, at))
                            .zip(content.first())
                            .is_some_and(|(p, c)| Self::restated(p, c));
                    }
                    stored.all(|((_, at), part)| {
                        usize::try_from(*at)
                            .ok()
                            .and_then(|at| content.get(at))
                            .is_some_and(|c| Self::restated(part, c))
                    })
                })
            })
        } else {
            None
        };
        let slot = found.unwrap_or(index);
        self.unattributed.remove(&slot);
        self.message_slots.insert(id.to_owned(), slot);
        self.rekey_undecided(id, slot);
        slot
    }
    fn bind_message(
        &mut self,
        slot: u64,
        id: Option<&str>,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        self.parents.entry(slot).or_insert(Parent::Message);
        for (at, value) in self.take_undecided(slot, id) {
            self.snapshot_part(slot, at, ow::AssistantContent::Unknown(value), None, out)?;
        }
        Ok(())
    }
    fn snapshot_part(
        &mut self,
        slot: u64,
        index: u64,
        content: ow::AssistantContent,
        phase: Option<&str>,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        let opaque = matches!(content, ow::AssistantContent::Unknown(_));
        let refusal = matches!(content, ow::AssistantContent::Refusal { .. });
        self.message_conflict(slot, index, opaque, None)?;
        let text = ow::text_block(content);
        let part = self
            .messages
            .entry((slot, index))
            .or_insert_with(|| MessagePart {
                order: {
                    let order = self.next_content;
                    self.next_content += 1;
                    order
                },
                text: ot::Text::new(""),
                opaque,
                refusal,
                phase: None,
            });
        if let Some(suffix) = text.text.strip_prefix(&part.text.text) {
            if !suffix.is_empty() {
                out.push(ObservedEvent::TextDelta {
                    text: suffix.to_owned(),
                });
            }
            part.text.text = text.text;
        }
        part.text.additional_params = text.additional_params;
        part.refusal |= refusal;
        if let Some(phase) = phase {
            ow::stamp_phase(&mut part.text, Some(phase));
            part.phase = Some(phase.to_owned());
        } else if let Some(phase) = part.phase.as_deref() {
            ow::stamp_phase(&mut part.text, Some(phase));
        }
        Ok(())
    }
    fn message(
        &mut self,
        index: u64,
        message: ow::OutputMessage,
        snapshot: bool,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        let slot = self.slot(
            index,
            Some(&message.id),
            snapshot.then_some((None, message.content.as_slice())),
        );
        for (at, content) in message.content.iter().enumerate() {
            self.message_conflict(
                slot,
                at as u64,
                matches!(content, ow::AssistantContent::Unknown(_)),
                Some(&message.id),
            )?;
        }
        self.bind_message(slot, Some(&message.id), out)?;
        let phase = message
            .phase
            .as_deref()
            .filter(|_| !self.delta_messages.contains(&slot));
        for (at, content) in message.content.into_iter().enumerate() {
            self.snapshot_part(slot, at as u64, content, phase, out)?;
        }
        Ok(())
    }
    fn delta(
        &mut self,
        index: u64,
        at: u64,
        id: Option<&str>,
        text: String,
        refusal: bool,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        if id.is_none_or(str::is_empty)
            && !self.message_slots.values().any(|slot| *slot == index)
        {
            self.unattributed.insert(index);
        }
        let slot = self.slot(index, id, None);
        self.message_conflict(slot, at, false, id)?;
        self.bind_message(slot, id, out)?;
        self.delta_messages.insert(slot);
        let part = self
            .messages
            .entry((slot, at))
            .or_insert_with(|| MessagePart {
                order: {
                    let order = self.next_content;
                    self.next_content += 1;
                    order
                },
                text: ot::Text::new(""),
                opaque: false,
                refusal: false,
                phase: None,
            });
        if refusal && !part.refusal && part.text.additional_params.is_none() {
            part.text.additional_params = ow::refusal_marker();
        }
        part.refusal |= refusal;
        part.text.text.push_str(&text);
        out.push(ObservedEvent::TextDelta { text });
        Ok(())
    }

    fn reasoning_conflict(
        &self,
        slot: u64,
        id: Option<&str>,
        summary: bool,
        index: u64,
        opaque: bool,
    ) -> Result<(), ObservedInterpretationError> {
        let parts = self.reasoning.get(&slot);
        let prior = if summary {
            parts
                .and_then(|p| p.summary.get(&index))
                .map(|p| matches!(p, ow::ReasoningSummary::Unknown(_)))
        } else {
            parts
                .and_then(|p| p.content.get(&index))
                .map(|p| matches!(p, ow::ReasoningTextContent::Unknown(_)))
                .or_else(|| {
                    (self.waiting_binds((slot, index), id)
                        || id.filter(|id| !id.is_empty()).is_some_and(|id| {
                            !self.reasoning.contains_key(&slot)
                                && self.undecided.iter().any(|((at, ci), ps)| {
                                    *at != slot
                                        && *ci == index
                                        && ps.iter().any(|p| p.item_id.as_deref() == Some(id))
                                })
                        }))
                    .then_some(true)
                })
        };
        if let Some(prior) = prior.filter(|prior| *prior != opaque) {
            return Err(Self::conflict(
                slot,
                if summary { "summary" } else { "content" },
                index,
                prior,
                opaque,
            ));
        }
        Ok(())
    }
    fn reasoning_parts(&mut self, slot: u64, id: Option<&str>) -> &mut ReasoningParts {
        self.parents.entry(slot).or_insert(Parent::Reasoning);
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            if !self.reasoning.contains_key(&slot) {
                self.rekey_undecided(id, slot);
            }
        }
        let pending = self.take_undecided(slot, id);
        let parts = self.reasoning.entry(slot).or_insert_with(|| {
            let order = self.next_content;
            self.next_content += 1;
            ReasoningParts {
                order,
                ..Default::default()
            }
        });
        for (at, value) in pending {
            parts
                .content
                .insert(at, ow::ReasoningTextContent::Unknown(value));
        }
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            if !parts.wire_sent || parts.id.is_none() {
                parts.id = Some(id.to_owned());
            }
        }
        parts
    }
    fn reasoning_snapshot(
        &mut self,
        slot: u64,
        id: String,
        summary: Vec<ow::ReasoningSummary>,
        content: Vec<ow::ReasoningTextContent>,
        encrypted: Option<String>,
        signature: Option<String>,
        done: bool,
    ) -> Result<(), ObservedInterpretationError> {
        if self.finished_reasoning.contains(&slot) {
            return Ok(());
        }
        for (at, p) in summary.iter().enumerate() {
            self.reasoning_conflict(
                slot,
                Some(&id),
                true,
                at as u64,
                matches!(p, ow::ReasoningSummary::Unknown(_)),
            )?;
        }
        for (at, p) in content.iter().enumerate() {
            self.reasoning_conflict(
                slot,
                Some(&id),
                false,
                at as u64,
                matches!(p, ow::ReasoningTextContent::Unknown(_)),
            )?;
        }
        let nonempty = !summary.is_empty()
            || !content.is_empty()
            || encrypted.as_ref().is_some_and(|s| !s.is_empty())
            || signature.as_ref().is_some_and(|s| !s.is_empty());
        let parts = self.reasoning_parts(slot, Some(&id));
        parts
            .summary
            .extend(summary.into_iter().enumerate().map(|(i, p)| (i as u64, p)));
        parts
            .content
            .extend(content.into_iter().enumerate().map(|(i, p)| (i as u64, p)));
        if let Some(v) = encrypted.filter(|s| !s.is_empty()) {
            if !parts.wire_sent || parts.encrypted.is_none() {
                parts.encrypted = Some(v);
            }
        }
        if let Some(v) = signature.filter(|s| !s.is_empty()) {
            if !parts.wire_sent || parts.signature.is_none() {
                parts.signature = Some(v);
            }
        }
        if done {
            parts.wire_sent = true;
            parts.ready |= nonempty || parts.id.is_some();
        }
        Ok(())
    }
    fn merge_failed_reasoning(&mut self, response: &ow::CompletionResponse) {
        for (slot, item) in response.output.iter().cloned().enumerate() {
            if let Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                ..
            } = item
            {
                // Selected failed/incomplete events keep their provider cause
                // primary even when an attempted reasoning restatement conflicts.
                let _ = self.reasoning_snapshot(
                    slot as u64,
                    id,
                    summary,
                    content,
                    encrypted_content,
                    signature,
                    true,
                );
            }
        }
    }
    fn tool_slot(&mut self, slot: u64, id: Option<&str>) -> &mut ToolParts {
        if !self.tools.contains_key(&slot) {
            if let Some(id) = id.filter(|id| !id.is_empty()) {
                self.finished_tools.remove(&ItemKey::Wire(id.to_owned()));
            }
        }
        self.tools.entry(slot).or_insert_with(|| {
            let minted = self.next_tool;
            if id.is_none_or(str::is_empty) {
                self.next_tool += 1;
            }
            ToolParts {
                order: {
                    let order = self.next_content;
                    self.next_content += 1;
                    order
                },
                minted,
                item_id: id.filter(|id| !id.is_empty()).map(str::to_owned),
                call_id: None,
                name: None,
                namespace: None,
                arguments: None,
                overflowed: false,
            }
        })
    }
    fn append_arguments(part: &mut ToolParts, fragment: &str) {
        let buffer = part.arguments.get_or_insert_with(String::new);
        if buffer.trim() == "null" && !fragment.trim().is_empty() {
            buffer.clear();
        }
        if buffer.len().saturating_add(fragment.len()) > 32 * 1024 * 1024 {
            part.overflowed = true;
        } else {
            buffer.push_str(fragment);
        }
    }
    fn function_done(
        &mut self,
        slot: u64,
        call: ow::OutputFunctionCall,
    ) -> Result<(), ObservedInterpretationError> {
        let mut slot = slot;
        let mut part = self.tools.remove(&slot);
        if part.is_none() && !call.id.is_empty() && !call.call_id.is_empty() {
            if let Ok(arguments) = call.arguments.parse() {
                let candidates: Vec<_> = self
                    .tools
                    .iter()
                    .map(|(slot, part)| (false, *slot, part))
                    .chain(
                        self.unclosed_tools
                            .iter()
                            .map(|(slot, part)| (true, *slot, part)),
                    )
                    .filter(|(_, _, part)| part.item_id.is_none())
                    .collect();
                if let [(drained, candidate, parts)] = candidates.as_slice() {
                    let covers = parts.arguments.as_deref().is_none_or(|buffer| {
                        buffer.trim().is_empty()
                            || crate::json_utils::parse_tool_arguments(buffer).is_ok_and(
                                |partial| {
                                    partial.is_null() || json_subsumes(&arguments, &partial)
                                },
                            )
                    });
                    if parts
                        .name
                        .as_deref()
                        .is_none_or(|name| name.is_empty() || name == call.name)
                        && covers
                    {
                        let (drained, candidate) = (*drained, *candidate);
                        part = if drained {
                            self.unclosed_tools.remove(&candidate)
                        } else {
                            self.tools.remove(&candidate)
                        };
                        slot = candidate;
                    }
                }
            }
        }
        let minted = part.as_ref().map(|p| p.minted).unwrap_or_else(|| {
            let n = self.next_tool;
            if call.id.is_empty() || call.call_id.is_empty() {
                self.next_tool += 1;
            }
            n
        });
        let key = part
            .as_ref()
            .and_then(|part| part.item_id.as_ref())
            .map(|id| ItemKey::Wire(id.clone()))
            .unwrap_or_else(|| {
                if !call.id.is_empty() && !call.call_id.is_empty() {
                    ItemKey::Wire(call.id.clone())
                } else {
                    ItemKey::Minted(minted)
                }
            });
        if !self.finished_tools.insert(key) {
            return Ok(());
        }
        let order = part.as_ref().map(|part| part.order).unwrap_or_else(|| {
            let order = self.next_content;
            self.next_content += 1;
            order
        });
        let name = if call.name.is_empty() {
            part.as_ref()
                .and_then(|part| part.name.clone())
                .unwrap_or_default()
        } else {
            call.name
        };
        if name.is_empty() {
            return Ok(());
        }
        let provider = ot::ProviderCallId::new(call.call_id)
            .map(|p| p.with_item_id(call.id))
            .or_else(|| {
                part.as_ref()
                    .and_then(|part| part.item_id.clone())
                    .and_then(ot::ProviderCallId::new)
            });
        let id =
            ot::ToolCallId::for_provider_or(provider.as_ref(), ot::ToolCallId::minted(minted));
        let parsed = call.arguments.parse();
        let arguments = match parsed {
            Ok(value) => value,
            Err(_) => {
                if matches!(call.status, ToolStatus::Incomplete) {
                    return Ok(());
                }
                let restated = call.arguments.as_str().to_owned();
                let mut buffer = part
                    .as_ref()
                    .and_then(|p| p.arguments.clone())
                    .unwrap_or_default();
                let mut overflowed = part.as_ref().is_some_and(|p| p.overflowed);
                if part.as_ref().is_none_or(|p| p.arguments.is_none())
                    || (!overflowed && crate::json_utils::parse_tool_arguments(&buffer).is_ok())
                {
                    if buffer.trim() == "null" && !restated.trim().is_empty() {
                        buffer.clear();
                    }
                    if buffer.len().saturating_add(restated.len()) > 32 * 1024 * 1024 {
                        overflowed = true;
                    } else {
                        buffer.push_str(&restated);
                    }
                }
                let error = if overflowed {
                    <serde_json::Error as serde::de::Error>::custom(
                        "tool-call input exceeded the accumulation bound",
                    )
                } else {
                    match crate::json_utils::parse_tool_arguments(&buffer) {
                        Err(e) => e,
                        Ok(_) => {
                            return Err(Self::failure(
                                "malformed completed tool input unexpectedly parsed".to_owned(),
                            ));
                        }
                    }
                };
                use crate::error::{
                    ErrorDetail, ErrorKind, MalformedToolInput,
                };
                return Err(ObservedInterpretationError::NativeReport(
                    ErrorReport::new(
                        ErrorKind::Response,
                        format!(
                            "tool call `{}` arrived with malformed JSON input: {error}",
                            name
                        ),
                    )
                    .with_detail(ErrorDetail::MalformedToolInput(
                        MalformedToolInput {
                            name,
                            id,
                            provider,
                            raw: buffer,
                            error: error.to_string(),
                        },
                    )),
                ));
            }
        };
        self.completed.insert(
            (slot, 0, order),
            ot::AssistantContent::ToolCall(ot::ToolCall {
                id,
                provider,
                function: ot::ToolFunction::new(name, arguments).with_namespace(call.namespace),
                signature: None,
                additional_params: None,
            }),
        );
        Ok(())
    }
    fn done(
        &mut self,
        slot: u64,
        item: Output,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        match item {
            Output::FunctionCall(call) => {
                let parts = self.tools.remove(&slot);
                self.pending_calls
                    .push((slot, PendingCall::Function(call, parts)));
                Ok(())
            }
            Output::CustomToolCall(call) => {
                self.pending_calls.push((slot, PendingCall::Custom(call)));
                Ok(())
            }
            item => self.done_immediately(slot, item, out),
        }
    }

    fn flush_pending_calls(&mut self) -> Result<(), ObservedInterpretationError> {
        for (slot, call) in std::mem::take(&mut self.pending_calls) {
            match call {
                PendingCall::Function(call, parts) => {
                    if let Some(parts) = parts {
                        self.tools.insert(slot, parts);
                    }
                    self.function_done(slot, call)?;
                }
                PendingCall::Custom(call) => {
                    self.done_immediately(slot, Output::CustomToolCall(call), &mut Vec::new())?;
                }
            }
        }
        Ok(())
    }

    /// Flush completed calls before the owning stream reports a transport error.
    /// This does not finalize the stream or close incomplete tool input.
    pub(in super::super) fn flush_before_terminal_error(
        &mut self,
    ) -> Result<(), ObservedInterpretationError> {
        self.flush_pending_calls()
    }

    /// Whether a whole-response payload has already emitted its terminal result.
    /// The owning stream drains ready events, then stops polling its source.
    pub(in super::super) fn whole_response_finished(&self) -> bool {
        self.whole_finished
    }

    fn flush_unclosed_tools(&mut self) {
        for (slot, part) in std::mem::take(&mut self.unclosed_tools) {
            let key = part
                .item_id
                .as_ref()
                .map(|id| ItemKey::Wire(id.clone()))
                .unwrap_or(ItemKey::Minted(part.minted));
            if !self.finished_tools.insert(key) {
                continue;
            }
            let Some(name) = part.name.filter(|name| !name.is_empty()) else {
                continue;
            };
            if part.overflowed {
                continue;
            }
            let arguments = match part.arguments {
                Some(raw) => match crate::json_utils::parse_tool_arguments(&raw) {
                    Ok(v) => v,
                    Err(_) => continue,
                },
                None => Value::Object(Default::default()),
            };
            let provider =
                part.call_id
                    .and_then(ot::ProviderCallId::new)
                    .map(|p| match part.item_id {
                        Some(id) => p.with_item_id(id),
                        None => p,
                    });
            let id = ot::ToolCallId::for_provider_or(
                provider.as_ref(),
                ot::ToolCallId::minted(part.minted),
            );
            self.completed.insert(
                (slot, 0, part.order),
                ot::AssistantContent::ToolCall(ot::ToolCall {
                    id,
                    provider,
                    function: ot::ToolFunction::new(name, arguments)
                        .with_namespace(part.namespace),
                    signature: None,
                    additional_params: None,
                }),
            );
        }
    }

    /// Flush fully delivered content at transport EOF, before finalization.
    /// A refused completed tool input is an item error, not a finish error.
    pub fn eof(&mut self) -> Result<Vec<ObservedEvent>, ObservedInterpretationError> {
        let result = self.flush_pending_calls();
        if result.is_err() {
            self.stopped = true;
        }
        result?;
        self.flush_unclosed_tools();
        Ok(Vec::new())
    }

    fn done_immediately(
        &mut self,
        slot: u64,
        item: Output,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        match item {
            Output::Message(message) => {
                let id = message.id.clone();
                self.message(slot, message, true, out)?;
                if !id.is_empty() {
                    self.message_id = Some(id);
                }
            }
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                ..
            } => self.reasoning_snapshot(
                slot,
                id,
                summary,
                content,
                encrypted_content,
                signature,
                true,
            )?,
            Output::FunctionCall(call) => self.function_done(slot, call)?,
            Output::CustomToolCall(call) => {
                let minted = self.next_tool;
                if call.id.is_empty() || call.call_id.is_empty() {
                    self.next_tool += 1;
                }
                let key = if !call.id.is_empty() && !call.call_id.is_empty() {
                    ItemKey::Wire(call.id.clone())
                } else {
                    ItemKey::Minted(minted)
                };
                if !self.finished_tools.insert(key) {
                    return Ok(());
                }
                let order = self.next_content;
                self.next_content += 1;
                let provider =
                    ot::ProviderCallId::new(call.call_id).map(|p| p.with_item_id(call.id));
                let id = ot::ToolCallId::for_provider_or(
                    provider.as_ref(),
                    ot::ToolCallId::minted(minted),
                );
                self.completed.insert(
                    (slot, 0, order),
                    ot::AssistantContent::CustomToolCall(ot::CustomToolCall {
                        id,
                        provider,
                        name: call.name,
                        namespace: call.namespace,
                        input: call.input,
                    }),
                );
            }
            Output::Unknown(value) => self.provider_item(slot, value),
            Output::Compaction(mut fields) => {
                fields.insert("type".to_owned(), Value::String("compaction".to_owned()));
                self.provider_item(slot, Value::Object(fields));
            }
        }
        Ok(())
    }
    fn provider_item(&mut self, slot: u64, value: Value) {
        self.provider_slots.insert(slot);
        let key = value
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(|id| ItemKey::Wire(id.to_owned()))
            .unwrap_or(ItemKey::Minted(slot));
        if !self.provider_keys.insert(key) {
            return;
        }
        let order = self.next_content;
        self.next_content += 1;
        self.completed.insert(
            (slot, 0, order),
            ot::AssistantContent::ProviderItem(ot::ProviderItem::new(
                value,
                self.provider.clone(),
            )),
        );
    }
    fn item(
        &mut self,
        chunk: ItemChunk,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        let ItemChunk {
            item_id,
            output_index: slot,
            data,
        } = chunk;
        let part_done = matches!(data, ItemChunkKind::ContentPartDone(_));
        match data {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput { item, .. }) => {
                match item {
                    Output::Message(message) => self.message(slot, message, false, out)?,
                    Output::Reasoning {
                        id,
                        summary,
                        content,
                        encrypted_content,
                        signature,
                        ..
                    } => self.reasoning_snapshot(
                        slot,
                        id,
                        summary,
                        content,
                        encrypted_content,
                        signature,
                        false,
                    )?,
                    Output::FunctionCall(call) => {
                        let part = self.tool_slot(
                            slot,
                            (!call.call_id.is_empty()).then_some(call.id.as_str()),
                        );
                        part.name = Some(call.name);
                        part.namespace = call.namespace;
                        part.call_id = (!call.call_id.is_empty()).then_some(call.call_id);
                    }
                    _ => {}
                }
            }
            ItemChunkKind::OutputItemDone(chunk) => self.done(slot, chunk.item, out)?,
            ItemChunkKind::OutputTextDelta(delta) => self.delta(
                slot,
                delta.content_index,
                item_id.as_deref(),
                delta.delta,
                false,
                out,
            )?,
            ItemChunkKind::RefusalDelta(delta) => self.delta(
                slot,
                delta.content_index,
                item_id.as_deref(),
                delta.delta,
                true,
                out,
            )?,
            ItemChunkKind::ContentPartAdded(part) | ItemChunkKind::ContentPartDone(part) => {
                let is_refusal = matches!(&part.part, ContentPartChunkPart::Refusal { .. });
                match part.part {
                    ContentPartChunkPart::SummaryText { text } => {
                        self.reasoning_conflict(
                            slot,
                            item_id.as_deref(),
                            true,
                            part.content_index,
                            false,
                        )?;
                        self.reasoning_parts(slot, item_id.as_deref())
                            .summary
                            .insert(
                                part.content_index,
                                ow::ReasoningSummary::SummaryText { text },
                            );
                    }
                    ContentPartChunkPart::ReasoningText { text } => {
                        let known = item_id
                            .as_ref()
                            .and_then(|id| self.message_slots.get(id))
                            .copied()
                            .unwrap_or(slot);
                        if [slot, known]
                            .iter()
                            .any(|at| self.parents.get(at) == Some(&Parent::Message))
                        {
                            return Err(Self::failure(format!(
                                "conflicting Responses parent kind at output {slot}, content {}: a reasoning_text part in a message",
                                part.content_index
                            )));
                        }
                        self.reasoning_conflict(
                            slot,
                            item_id.as_deref(),
                            false,
                            part.content_index,
                            false,
                        )?;
                        self.reasoning_parts(slot, item_id.as_deref())
                            .content
                            .insert(
                                part.content_index,
                                ow::ReasoningTextContent::ReasoningText { text },
                            );
                    }
                    ContentPartChunkPart::Unknown(value) => {
                        let slot = item_id
                            .as_ref()
                            .and_then(|id| self.message_slots.get(id))
                            .copied()
                            .unwrap_or(slot);
                        match self.parents.get(&slot) {
                            Some(Parent::Message) => self.snapshot_part(
                                slot,
                                part.content_index,
                                ow::AssistantContent::Unknown(value),
                                None,
                                out,
                            )?,
                            Some(Parent::Reasoning) => {
                                self.reasoning_conflict(
                                    slot,
                                    item_id.as_deref(),
                                    false,
                                    part.content_index,
                                    true,
                                )?;
                                self.reasoning_parts(slot, item_id.as_deref())
                                    .content
                                    .insert(
                                        part.content_index,
                                        ow::ReasoningTextContent::Unknown(value),
                                    );
                            }
                            None => {
                                let at = part.content_index;
                                let pending = UndecidedPart {
                                    item_id: item_id.filter(|id| !id.is_empty()),
                                    value,
                                };
                                let parts = self.undecided.entry((slot, at)).or_default();
                                if !parts.contains(&pending) {
                                    parts.push(pending);
                                }
                            }
                        }
                    }
                    ContentPartChunkPart::OutputText { text }
                    | ContentPartChunkPart::Refusal { refusal: text } => {
                        let content = if is_refusal {
                            ow::AssistantContent::Refusal { refusal: text }
                        } else {
                            ow::AssistantContent::OutputText(ow::OutputText {
                                text,
                                extras: Default::default(),
                            })
                        };
                        let slot = self.slot(
                            slot,
                            item_id.as_deref(),
                            part_done.then_some((
                                Some(part.content_index),
                                std::slice::from_ref(&content),
                            )),
                        );
                        self.message_conflict(
                            slot,
                            part.content_index,
                            false,
                            item_id.as_deref(),
                        )?;
                        self.bind_message(slot, item_id.as_deref(), out)?;
                        self.snapshot_part(slot, part.content_index, content, None, out)?;
                    }
                }
            }
            ItemChunkKind::ReasoningSummaryPartAdded(part)
            | ItemChunkKind::ReasoningSummaryPartDone(part) => {
                self.reasoning_conflict(
                    slot,
                    item_id.as_deref(),
                    true,
                    part.summary_index,
                    matches!(part.part, ow::ReasoningSummary::Unknown(_)),
                )?;
                self.reasoning_parts(slot, item_id.as_deref())
                    .summary
                    .insert(part.summary_index, part.part);
            }
            ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                self.reasoning_conflict(
                    slot,
                    item_id.as_deref(),
                    true,
                    delta.summary_index,
                    false,
                )?;
                let parts = self.reasoning_parts(slot, item_id.as_deref());
                parts.opened = true;
                if let ow::ReasoningSummary::SummaryText { text } = parts
                    .summary
                    .entry(delta.summary_index)
                    .or_insert_with(|| ow::ReasoningSummary::SummaryText {
                        text: String::new(),
                    })
                {
                    text.push_str(&delta.delta);
                }
            }
            ItemChunkKind::ReasoningTextDelta(delta) => {
                let at = delta.content_index.unwrap_or(0);
                self.reasoning_conflict(slot, item_id.as_deref(), false, at, false)?;
                let parts = self.reasoning_parts(slot, item_id.as_deref());
                parts.opened = true;
                if let ow::ReasoningTextContent::ReasoningText { text } = parts
                    .content
                    .entry(at)
                    .or_insert_with(|| ow::ReasoningTextContent::ReasoningText {
                        text: String::new(),
                    })
                {
                    text.push_str(&delta.delta);
                }
            }
            ItemChunkKind::ReasoningSummaryTextDone(part) => {
                self.reasoning_conflict(
                    slot,
                    item_id.as_deref(),
                    true,
                    part.summary_index,
                    false,
                )?;
                self.reasoning_parts(slot, item_id.as_deref())
                    .summary
                    .insert(
                        part.summary_index,
                        ow::ReasoningSummary::SummaryText { text: part.text },
                    );
            }
            ItemChunkKind::ReasoningTextDone(part) => {
                self.reasoning_conflict(
                    slot,
                    item_id.as_deref(),
                    false,
                    part.content_index,
                    false,
                )?;
                self.reasoning_parts(slot, item_id.as_deref())
                    .content
                    .insert(
                        part.content_index,
                        ow::ReasoningTextContent::ReasoningText { text: part.text },
                    );
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                Self::append_arguments(self.tool_slot(slot, item_id.as_deref()), &delta.delta)
            }
            ItemChunkKind::FunctionCallArgsDone(_)
            | ItemChunkKind::OutputTextDone(_)
            | ItemChunkKind::RefusalDone(_) => {}
        }
        Ok(())
    }
    fn complete(
        &mut self,
        response: ow::CompletionResponse,
        out: &mut Vec<ObservedEvent>,
    ) -> Result<(), ObservedInterpretationError> {
        for (slot, item) in response.output.iter().cloned().enumerate() {
            match item {
                Output::Message(message) => self.message(slot as u64, message, true, out)?,
                Output::Reasoning {
                    id,
                    summary,
                    content,
                    encrypted_content,
                    signature,
                    ..
                } => self.reasoning_snapshot(
                    slot as u64,
                    id,
                    summary,
                    content,
                    encrypted_content,
                    signature,
                    true,
                )?,
                item @ (Output::Unknown(_) | Output::Compaction(_)) => {
                    if !self.provider_slots.contains(&(slot as u64)) {
                        self.done(slot as u64, item, out)?;
                    }
                }
                // Selected terminal-only known tools remain raw-only.
                _ => {}
            }
        }
        for part in std::mem::take(&mut self.undecided).into_values().flatten() {
            out.push(ObservedEvent::Unknown {
                event_type: String::new(),
                value: part.value,
            });
        }
        for (slot, parts) in std::mem::take(&mut self.reasoning) {
            let opaque = parts
                .summary
                .values()
                .any(|p| matches!(p, ow::ReasoningSummary::Unknown(_)))
                || parts
                    .content
                    .values()
                    .any(|p| matches!(p, ow::ReasoningTextContent::Unknown(_)));
            if !self.finished_reasoning.contains(&slot)
                && (parts.ready || parts.opened || opaque)
            {
                let content = ow::reasoning_content_blocks(
                    parts.summary.into_values().collect(),
                    parts.content.into_values().collect(),
                    parts.encrypted,
                    parts.signature,
                );
                self.completed.insert(
                    (slot, 0, parts.order),
                    ot::AssistantContent::Reasoning(ot::Reasoning {
                        id: parts.id,
                        content,
                        provider: Some(self.provider.clone()),
                    }),
                );
                self.finished_reasoning.insert(slot);
            }
        }
        self.unclosed_tools.extend(std::mem::take(&mut self.tools));
        if response.usage.is_some() {
            self.terminal_usage = response.usage.clone();
        }
        if !response.id.is_empty() {
            self.terminal_response_id = Some(response.id.clone());
        }
        if !response.model.is_empty() {
            self.terminal_model = Some(response.model.clone());
        }
        if let Some(id) = response.output.iter().find_map(|item| match item {
            Output::Message(message) => Some(message.id.clone()),
            _ => None,
        }) {
            self.terminal_message_id = Some(id);
        }
        // Match selected StreamingCompletionResponse serialization: absent
        // optional fields are omitted, and later absence retains prior metadata.
        // Connection request IDs are stamped on the normalized response only.
        for (key, value) in [
            ("usage", serde_json::to_value(&response.usage)),
            (
                "reasoning_metadata",
                serde_json::to_value(&response.reasoning_metadata),
            ),
            (
                "reasoning_context",
                serde_json::to_value(&response.reasoning_context),
            ),
            ("status", serde_json::to_value(&response.status)),
            (
                "incomplete_details",
                serde_json::to_value(&response.incomplete_details),
            ),
            (
                "message_id",
                serde_json::to_value(&self.terminal_message_id),
            ),
            (
                "response_id",
                serde_json::to_value(&self.terminal_response_id),
            ),
            ("model", serde_json::to_value(&self.terminal_model)),
        ] {
            let value = value.map_err(ProviderError::Json)?;
            if !value.is_null() {
                self.document.insert(key.to_owned(), value);
            }
        }
        self.terminal = Some(response);
        Ok(())
    }
    /// Finalize after transport EOF or a whole response. A genuine terminal is required.
    pub fn finish(
        mut self,
    ) -> Result<ObservedResponsesResultBody, ObservedInterpretationError> {
        let native=self.terminal.take().ok_or_else(||Self::failure("provider stream ended without a terminal record; treating the turn as truncated".to_owned()))?;
        if self.stopped {
            return Err(Self::failure(
                "provider stream ended after an interpretation failure".to_owned(),
            ));
        }
        for (key, part) in self.messages {
            self.completed.insert(
                (key.0, key.1, part.order),
                ot::AssistantContent::Text(part.text),
            );
        }
        let usage = self
            .terminal_usage
            .as_ref()
            .map(ot::Usage::from)
            .unwrap_or_default();
        let message_id = self
            .message_id
            .or(self.terminal_message_id)
            .filter(|id| !id.is_empty());
        let reason = match &native.status {
            super::super::ResponseStatus::Completed => Some(ot::FinishReason::Stop),
            super::super::ResponseStatus::Incomplete => Some(
                match self
                    .document
                    .get("incomplete_details")
                    .and_then(|details| details.get("reason"))
                    .and_then(Value::as_str)
                    .filter(|reason| !reason.is_empty())
                {
                    Some("max_output_tokens") => ot::FinishReason::Length,
                    Some("content_filter") => ot::FinishReason::ContentFilter,
                    Some(other) => ot::FinishReason::Other(other.to_owned()),
                    None => ot::FinishReason::Other("incomplete".to_owned()),
                },
            ),
            super::super::ResponseStatus::Failed => {
                Some(ot::FinishReason::Other("failed".to_owned()))
            }
            super::super::ResponseStatus::Cancelled => {
                Some(ot::FinishReason::Other("cancelled".to_owned()))
            }
            super::super::ResponseStatus::Other(status) if status.is_empty() => None,
            super::super::ResponseStatus::Other(status) => {
                Some(ot::FinishReason::Other(status.clone()))
            }
            super::super::ResponseStatus::InProgress | super::super::ResponseStatus::Queued => {
                None
            }
        };
        self.prefix.extend(self.completed.into_values());
        let mut response = ot::CompletionResponse {
            choice: self.prefix,
            usage,
            message_id,
            response_id: self.terminal_response_id,
            provider_request_id: self.provider_request_id,
            provider_response_headers: self.response_headers,
            finish_reason: None,
            provider: self.provider,
            model: self.terminal_model,
            raw: Value::Object(self.document),
        };
        response.set_finish_reason(reason);
        Ok(ObservedResponsesResultBody { response, native })
    }
}
