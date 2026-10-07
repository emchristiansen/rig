//! Wire envelopes and classification for observed Responses payloads.

use super::super::observed_wire as ow;
use crate::providers::internal::wire::{self, WireEvent};
use ow::{FunctionCallArguments, Output, ReasoningSummary};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)] // Response and output-item envelopes.
pub(in super::super) enum Chunk {
    Response(ResponseChunk),
    Delta(ItemChunk),
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(in super::super) struct ResponseChunk {
    #[serde(rename = "type")]
    pub(in super::super) kind: super::super::streaming::ResponseChunkKind,
    pub(in super::super) response: ow::CompletionResponse,
    pub(in super::super) sequence_number: u64,
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
    #[serde(untagged)] // Serialization of classified opaque content.
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

pub(in super::super) enum Payload {
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

fn classify_payload(data: &str, is_known_event_type: fn(&str) -> bool) -> WireEvent<Payload> {
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
        wire::classify_tagged_frame::<Chunk>(data, "type", is_known_event_type).map(|chunk| {
            Payload::Frame {
                raw: data.to_owned(),
                chunk,
            }
        })
    };
    if parsed.as_ref().is_ok_and(|v| v.get("type").is_some()) {
        return tagged(data);
    }
    wire::classify_or(data, tagged, |data| {
        wire::classify_or(
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

pub(in super::super) fn classify(
    data: &str,
    is_known_event_type: fn(&str) -> bool,
) -> (WireEvent<Payload>, String) {
    let event = wire::classify_with_repair(
        data,
        |data| classify_payload(data, is_known_event_type),
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
    );
    let event_type = match &event {
        WireEvent::Unknown { event_type, .. } => event_type.clone(),
        _ => String::new(),
    };
    (event, event_type)
}
