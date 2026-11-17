//! Typed Responses stream events.
//!
//! Rig's Responses wire hands a consumer JSON frames. [`decode_frame`] reads
//! one such frame into the typed chunk it is: a response lifecycle event
//! ([`ResponseChunk`]), an output-item event ([`ItemChunk`]), or, for an event
//! type this crate does not model, the whole payload with its tag. A frame of
//! a modelled type that fails its typed decoding is a
//! [`CorruptFrame`] carrying the frame as
//! received, never an unknown event.
//!
//! ```
//! use rig_codex::responses_api::streaming::{DecodedFrame, ItemChunkKind, decode_frame};
//!
//! let frame = r#"{"type":"response.output_text.delta","item_id":"msg_1",
//!     "output_index":0,"content_index":0,"sequence_number":3,"delta":"hi"}"#;
//! let DecodedFrame::Chunk(chunk) = decode_frame(frame)? else { panic!("modelled") };
//! # let rig_codex::responses_api::streaming::StreamingCompletionChunk::Delta(item) = chunk else { panic!() };
//! assert!(matches!(item.data, ItemChunkKind::OutputTextDelta(_)));
//! # Ok::<(), rig_codex::error::CorruptFrame>(())
//! ```

use super::{
    CompletionResponse, FunctionCallArguments, IncompleteDetailsReason, Output, ReasoningSummary,
    ResponseStatus, ResponsesUsage,
};
use crate::error::CorruptFrame;
use rig_core::providers::internal::wire::classify_tagged_frame;
use rig_core::streaming::UnknownPayload;
use rig_core::wire::WireEvent;
use serde::{Deserialize, Serialize};

/// Response lifecycle event or output-item event.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    /// A response lifecycle event, such as `response.created` or `response.completed`.
    Response(ResponseChunk),
    /// An output-item event, such as `response.output_text.delta`.
    Delta(ItemChunk),
}

/// The terminal metadata of a streamed response.
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

impl From<&StreamingCompletionResponse> for rig_core::completion::Usage {
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
    /// `response.created`: the response was created.
    #[serde(rename = "response.created")]
    ResponseCreated,
    /// `response.in_progress`: the response is being generated.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    /// `response.completed`: the response finished.
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    /// `response.failed`: the response failed.
    #[serde(rename = "response.failed")]
    ResponseFailed,
    /// `response.incomplete`: the response stopped before it finished.
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
    /// `response.output_item.added`: an output item began.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(StreamingItemDoneOutput),
    /// `response.output_item.done`: an output item is complete.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(StreamingItemDoneOutput),
    /// `response.content_part.added`: a content part began.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartChunk),
    /// `response.content_part.done`: a content part is complete.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartChunk),
    /// `response.output_text.delta`: more output text.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(DeltaTextChunk),
    /// `response.output_text.done`: a text part's full text.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(OutputTextChunk),
    /// `response.refusal.delta`: more refusal text.
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta(DeltaTextChunk),
    /// `response.refusal.done`: a refusal's full text.
    #[serde(rename = "response.refusal.done")]
    RefusalDone(RefusalTextChunk),
    /// `response.function_call_arguments.delta`: more of a function call's arguments.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta(DeltaTextChunkWithItemId),
    /// `response.function_call_arguments.done`: a function call's full arguments.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    /// `response.reasoning_summary_part.added`: a reasoning summary part began.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    /// `response.reasoning_summary_part.done`: a reasoning summary part is complete.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    /// `response.reasoning_summary_text.delta`: more reasoning summary text.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextDeltaChunk),
    /// `response.reasoning_summary_text.done`: a reasoning summary's full text.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextDoneChunk),
    /// `response.reasoning_text.delta`: more reasoning text.
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

/// An output item, as an `output_item` event carries it.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingItemDoneOutput {
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The output item.
    pub item: Output,
}

/// A content part, as a `content_part` event carries it.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPartChunk {
    /// The part's index in its item's content.
    pub content_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The content part.
    pub part: ContentPartChunkPart,
}

/// The part a `content_part` event carries.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartChunkPart {
    /// A refusal.
    Refusal {
        /// The refusal's text so far.
        refusal: String,
    },
    /// Output text.
    OutputText {
        /// The text so far.
        text: String,
    },
    /// Reasoning summary text.
    SummaryText {
        /// The summary text so far.
        text: String,
    },
    /// Reasoning content, announced by `response.content_part.*` on some
    /// gateways before its `response.reasoning_text.*` deltas.
    ReasoningText {
        /// The reasoning text so far.
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

/// A text delta.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunk {
    /// The index of the content part the delta extends.
    pub content_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The text to append.
    pub delta: String,
}

/// A text delta of an event that may name no content part.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunkWithItemId {
    /// The index of the content part the delta extends, when the event names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The text to append.
    pub delta: String,
}

/// A part's full text.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputTextChunk {
    /// The index of the content part.
    pub content_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The full text.
    pub text: String,
}

/// A refusal's full text.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefusalTextChunk {
    /// The index of the content part.
    pub content_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The full refusal text.
    pub refusal: String,
}

/// A function call's full arguments.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArgsTextChunk {
    /// The index of the content part, when the event names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The call's arguments, in the same type (and so under the same
    /// classification) as the `output_item.done` item's
    /// [`OutputFunctionCall::arguments`](super::OutputFunctionCall::arguments);
    /// [`FunctionCallArguments::reconcile`] settles whether the two agree.
    pub arguments: FunctionCallArguments,
}

/// A reasoning summary part, as a `reasoning_summary_part` event carries it.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryPartChunk {
    /// The part's index in its item's summary.
    pub summary_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The summary part.
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
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The incremental text.
    pub delta: String,
}

/// A reasoning-summary text restatement (`response.reasoning_summary_text.done`).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextDoneChunk {
    /// The reasoning summary this restatement closes.
    pub summary_index: u64,
    /// The event's position in the stream.
    pub sequence_number: u64,
    /// The summary's full text.
    pub text: String,
}

/// The part a `reasoning_summary_part` event carries.
pub type SummaryPartChunkPart = ReasoningSummary;

/// One Responses frame, read into the typed value it is.
#[derive(Debug, Clone)]
pub enum DecodedFrame {
    /// An event this crate models, fully decoded.
    Chunk(StreamingCompletionChunk),
    /// An event of a type this crate does not model, kept whole.
    Unknown {
        /// The event's `type`; empty for JSON that is not an object.
        event_type: String,
        /// The complete parsed payload.
        payload: UnknownPayload,
    },
}

/// Read one Responses frame's text into its typed chunk.
///
/// The frame is triaged by its top-level `type`. A type this crate models is
/// decoded strictly, and so is an object with no string `type`. Any other
/// type, and JSON that is not an object, is returned whole as
/// [`DecodedFrame::Unknown`]. The websocket `error` and `response.done`
/// events are not Responses stream events: a caller reads those first.
///
/// # Errors
///
/// A [`CorruptFrame`] with `frame` exactly as given, for invalid JSON, a
/// repeated `type`, or a modelled event that fails its typed decoding.
pub fn decode_frame(frame: &str) -> Result<DecodedFrame, CorruptFrame> {
    match classify_tagged_frame::<StreamingCompletionChunk>(
        frame,
        "type",
        is_known_responses_event_type,
    ) {
        WireEvent::Known(chunk) => Ok(DecodedFrame::Chunk(chunk)),
        WireEvent::Unknown { event_type, value } => Ok(DecodedFrame::Unknown {
            event_type,
            payload: value,
        }),
        WireEvent::Corrupt(error) => Err(CorruptFrame::text(event_type_of(frame), frame, error)),
    }
}

/// The frame's top-level string `type`, when it has one.
fn event_type_of(frame: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Tagged {
        #[serde(rename = "type")]
        kind: String,
    }
    serde_json::from_str::<Tagged>(frame)
        .ok()
        .map(|tagged| tagged.kind)
}

/// Read one Responses frame already parsed as JSON into its typed chunk.
///
/// The frame as received is not in hand, so a failure carries
/// [`FrameEvidence::Absent`](crate::error::FrameEvidence::Absent): the caller
/// still holds the value it passed.
impl TryFrom<&serde_json::Value> for DecodedFrame {
    type Error = CorruptFrame;

    fn try_from(frame: &serde_json::Value) -> Result<Self, Self::Error> {
        decode_frame(&frame.to_string()).map_err(CorruptFrame::without_frame)
    }
}

/// The terminal record of the response a terminal event carries.
impl From<&CompletionResponse> for StreamingCompletionResponse {
    fn from(response: &CompletionResponse) -> Self {
        Self {
            usage: response.usage,
            reasoning_metadata: response.reasoning_metadata.clone(),
            reasoning_context: response.reasoning_context.clone(),
            status: Some(response.status.clone()),
            incomplete_details: response.incomplete_details.clone(),
            message_id: response.output.iter().find_map(|item| match item {
                Output::Message(message) => Some(message.id.clone()),
                _ => None,
            }),
            response_id: (!response.id.is_empty()).then(|| response.id.clone()),
            model: (!response.model.is_empty()).then(|| response.model.clone()),
            provider_request_id: response.provider_request_id.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
