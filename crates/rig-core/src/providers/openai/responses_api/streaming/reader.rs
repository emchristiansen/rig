//! Strict schemas for known Responses payloads. Unknown items and parts keep
//! their original JSON; these private schemas prove the fields we interpret.

use serde::Deserialize;
use serde_json::Value;

use super::{ResponsesEvent, is_known_responses_event_type};
use crate::providers::internal::wire;
use crate::wire::WireEvent;

type Result<T> = std::result::Result<T, serde_json::Error>;

fn fields<T: serde::de::DeserializeOwned>(value: &Value) -> Result<T> {
    serde_json::from_value(value.clone())
}

fn invalid(message: impl std::fmt::Display) -> serde_json::Error {
    <serde_json::Error as serde::de::Error>::custom(message)
}

// Values validated here remain in the original Value, including unmodeled
// siblings and explicit nulls. Proof structs never replace that input.
#[allow(dead_code)]
#[derive(Deserialize)]
struct WholeFields {
    id: String,
    object: ResponseObject,
    created_at: u64,
    status: String,
    model: String,
    #[serde(default)]
    output: Vec<Value>,
    error: Option<ResponseError>,
    incomplete_details: Option<IncompleteDetails>,
    instructions: Option<String>,
    max_output_tokens: Option<u64>,
    usage: Option<UsageFields>,
    #[serde(default)]
    tools: Vec<super::super::ResponsesToolDefinition>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct UsageFields {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    input_tokens_details: Option<InputDetails>,
    output_tokens_details: Option<OutputDetails>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct InputDetails {
    cached_tokens: Option<u64>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct OutputDetails {
    reasoning_tokens: u64,
}

#[derive(Deserialize)]
enum ResponseObject {
    #[serde(rename = "response")]
    Response,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ResponseError {
    code: String,
    message: String,
}

#[cfg(any(feature = "websocket", test))]
#[derive(Deserialize, serde::Serialize)]
struct WebSocketErrorFields {
    #[serde(rename = "code", skip_serializing_if = "Option::is_none")]
    _code: Option<String>,
    #[serde(rename = "message")]
    _message: String,
}

#[cfg(any(feature = "websocket", test))]
#[derive(Deserialize)]
struct WebSocketResponseRoot {
    #[serde(rename = "id")]
    _id: String,
    #[serde(rename = "object")]
    _object: ResponseObject,
    #[serde(rename = "created_at")]
    _created_at: u64,
    #[serde(rename = "status")]
    _status: String,
    #[serde(rename = "model")]
    _model: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct IncompleteDetails {
    reason: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct MessageFields {
    id: String,
    role: Assistant,
    status: String,
    content: Vec<Value>,
    phase: Option<String>,
}

#[derive(Deserialize)]
enum Assistant {
    #[serde(rename = "assistant")]
    Assistant,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FunctionFields {
    #[serde(default)]
    id: String,
    arguments: String,
    call_id: String,
    name: String,
    namespace: Option<String>,
    status: ToolStatus,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct CustomFields {
    #[serde(default)]
    id: String,
    call_id: String,
    name: String,
    namespace: Option<String>,
    input: String,
    status: Option<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ReasoningFields {
    id: String,
    #[serde(default)]
    summary: Vec<Value>,
    #[serde(default)]
    content: Value,
    encrypted_content: Option<String>,
    signature: Option<String>,
    status: Option<ToolStatus>,
}

pub(super) fn whole(value: &Value) -> Result<()> {
    let whole: WholeFields = fields(value)?;
    for item in &whole.output {
        output(item)?;
    }
    Ok(())
}

/// Validate a sparse WebSocket terminal response without changing its evidence.
/// Its id and status are strings; future status spellings remain valid.
#[cfg(any(feature = "websocket", test))]
pub(crate) fn websocket_done(value: &Value) -> Result<()> {
    let response = value
        .as_object()
        .ok_or_else(|| invalid("response.done.response must be an object"))?;
    for key in ["id", "status"] {
        let value = response
            .get(key)
            .ok_or_else(|| invalid(format!("missing field `{key}`")))?;
        let _: String = fields(value)?;
    }
    done_field::<ResponseObject>(value, "object")?;
    done_field::<u64>(value, "created_at")?;
    done_field::<String>(value, "model")?;
    done_field::<Option<WebSocketErrorFields>>(value, "error")?;
    done_field::<Option<IncompleteDetails>>(value, "incomplete_details")?;
    done_field::<Option<String>>(value, "instructions")?;
    done_field::<Option<u64>>(value, "max_output_tokens")?;
    done_field::<Option<UsageFields>>(value, "usage")?;
    if let Some(items) = response.get("output") {
        for item in fields::<Vec<Value>>(items)? {
            output(&item)?;
        }
    }
    if let Some(tools) = response.get("tools") {
        let _: Vec<super::super::ResponsesToolDefinition> = fields(tools)?;
        for tool in fields::<Vec<Value>>(tools)? {
            // WebSocket responses retain the original description
            // contract, which accepts absence but requires a present string.
            done_field::<String>(&tool, "description")?;
        }
    }
    Ok(())
}

/// Validate a full WebSocket lifecycle response without changing its evidence.
/// Sparse terminal objects require the separate [`websocket_done`] entry.
#[cfg(any(feature = "websocket", test))]
pub(crate) fn websocket_response(value: &Value) -> Result<()> {
    let _: WebSocketResponseRoot = fields(value)?;
    websocket_done(value)
}

#[cfg(any(feature = "websocket", test))]
fn done_field<T: serde::de::DeserializeOwned>(value: &Value, key: &str) -> Result<()> {
    if let Some(field) = value.get(key) {
        let _: T =
            fields(field).map_err(|error| invalid(format!("WebSocket response.{key}: {error}")))?;
    }
    Ok(())
}

pub(super) fn output(value: &Value) -> Result<()> {
    match value.get("type").and_then(Value::as_str) {
        Some("message") => {
            let message: MessageFields = fields(value)?;
            for content in &message.content {
                part(content, &["output_text", "refusal"])?;
            }
        }
        Some("function_call") => {
            let _: FunctionFields = fields(value)?;
        }
        Some("custom_tool_call") => {
            let _: CustomFields = fields(value)?;
        }
        Some("reasoning") => {
            let reasoning: ReasoningFields = fields(value)?;
            for summary in &reasoning.summary {
                part(summary, &["summary_text"])?;
            }
            match &reasoning.content {
                Value::Null if value.get("content").is_none() => {}
                Value::String(_) => {}
                Value::Array(content) => {
                    for content in content {
                        if content.is_string() {
                            continue;
                        }
                        part(content, &["reasoning_text"])?;
                    }
                }
                _ => return Err(invalid("reasoning content must be a string or an array")),
            }
        }
        // Missing/non-string types and unknown types are faithful raw items.
        _ => {}
    }
    Ok(())
}

fn part(value: &Value, known: &[&str]) -> Result<()> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("a Responses content part needs a string `type` tag"))?;
    if known.contains(&kind) {
        let key = if kind == "refusal" { "refusal" } else { "text" };
        if value.get(key).and_then(Value::as_str).is_none() {
            return Err(invalid(format!(
                "`{kind}` content part requires a string `{key}`"
            )));
        }
    }
    Ok(())
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    kind: String,
    sequence_number: u64,
    item_id: Option<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ItemEnvelope {
    output_index: u64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ContentEnvelope {
    content_index: u64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct OptionalContentEnvelope {
    content_index: Option<u64>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct SummaryEnvelope {
    summary_index: u64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Delta {
    delta: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Text {
    text: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Refusal {
    refusal: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Arguments {
    arguments: String,
}

struct Known(Value);

impl<'de> Deserialize<'de> for Known {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        event(&value).map_err(serde::de::Error::custom)?;
        Ok(Self(value))
    }
}

#[cfg(any(feature = "websocket", test))]
struct WebSocketKnown(Value);

#[cfg(any(feature = "websocket", test))]
#[derive(Deserialize)]
struct WebSocketLifecycleEnvelope {
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "sequence_number")]
    _sequence_number: u64,
}

#[cfg(any(feature = "websocket", test))]
impl<'de> Deserialize<'de> for WebSocketKnown {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let envelope: WebSocketLifecycleEnvelope =
            fields(&value).map_err(serde::de::Error::custom)?;
        if super::is_lifecycle_event(&envelope.kind) {
            let response = value
                .get("response")
                .ok_or_else(|| <D::Error as serde::de::Error>::missing_field("response"))?;
            websocket_response(response).map_err(serde::de::Error::custom)?;
        } else {
            event(&value).map_err(serde::de::Error::custom)?;
        }
        Ok(Self(value))
    }
}

fn event(value: &Value) -> Result<()> {
    let envelope: Envelope = fields(value)?;
    let required = |key: &str| {
        value
            .get(key)
            .ok_or_else(|| invalid(format!("missing field `{key}`")))
    };
    let kind = envelope.kind.as_str();
    if super::is_lifecycle_event(kind) {
        return whole(required("response")?);
    }
    let _: ItemEnvelope = fields(value)?;
    match kind {
        "response.output_item.added" | "response.output_item.done" => output(required("item")?),
        "response.content_part.added" | "response.content_part.done" => {
            let _: ContentEnvelope = fields(value)?;
            part(
                required("part")?,
                &["refusal", "output_text", "summary_text", "reasoning_text"],
            )
        }
        "response.reasoning_summary_part.added" | "response.reasoning_summary_part.done" => {
            let _: SummaryEnvelope = fields(value)?;
            part(required("part")?, &["summary_text"])
        }
        "response.output_text.delta" | "response.refusal.delta" => {
            let _: ContentEnvelope = fields(value)?;
            let _: Delta = fields(value)?;
            Ok(())
        }
        "response.output_text.done" | "response.reasoning_text.done" => {
            let _: ContentEnvelope = fields(value)?;
            let _: Text = fields(value)?;
            Ok(())
        }
        "response.refusal.done" => {
            let _: ContentEnvelope = fields(value)?;
            let _: Refusal = fields(value)?;
            Ok(())
        }
        "response.function_call_arguments.delta" | "response.reasoning_text.delta" => {
            let _: OptionalContentEnvelope = fields(value)?;
            let _: Delta = fields(value)?;
            Ok(())
        }
        "response.function_call_arguments.done" => {
            let _: OptionalContentEnvelope = fields(value)?;
            let _: Arguments = fields(value)?;
            Ok(())
        }
        "response.reasoning_summary_text.delta" => {
            let _: SummaryEnvelope = fields(value)?;
            let _: Delta = fields(value)?;
            Ok(())
        }
        "response.reasoning_summary_text.done" => {
            let _: SummaryEnvelope = fields(value)?;
            let _: Text = fields(value)?;
            Ok(())
        }
        _ => Err(invalid(format!("unmodeled known Responses event `{kind}`"))),
    }
}

struct Whole(Value);
impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        whole(&value).map_err(serde::de::Error::custom)?;
        Ok(Self(value))
    }
}

#[cfg(any(feature = "websocket", test))]
struct WebSocketWhole(Value);

#[cfg(any(feature = "websocket", test))]
impl<'de> Deserialize<'de> for WebSocketWhole {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        websocket_response(&value).map_err(serde::de::Error::custom)?;
        Ok(Self(value))
    }
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct ErrorEnvelope {
    error: Value,
}

pub(super) fn tagged(data: &str) -> WireEvent<ResponsesEvent> {
    wire::classify_tagged_frame::<Known>(data, "type", is_known_responses_event_type).map(
        |Known(frame)| ResponsesEvent::Frame {
            kind: frame["type"].as_str().expect("strict known tag").to_owned(),
            frame,
            raw: data.to_owned(),
        },
    )
}

pub(super) fn classify(data: &str) -> WireEvent<ResponsesEvent> {
    if data.trim() == "[DONE]" {
        return WireEvent::Known(ResponsesEvent::Sentinel);
    }
    // Error events have priority, and their exact provider payload decides
    // the refusal. They are not ordinary typed response lifecycle events.
    if serde_json::from_str::<Value>(data)
        .is_ok_and(|v| v.get("type").and_then(Value::as_str) == Some("error"))
    {
        return WireEvent::Known(ResponsesEvent::Failure(data.to_owned()));
    }
    wire::classify_or_untagged(data, "type", tagged, |data| {
        wire::classify_or(
            data,
            |data| {
                wire::classify_marker_keyed_frame::<Whole>(
                    data,
                    &["object", "output", "status", "error"],
                )
                .map(|Whole(body)| ResponsesEvent::Whole(body))
            },
            |data| {
                wire::classify_marker_keyed_frame::<ErrorEnvelope>(data, &["error"])
                    .map(|_| ResponsesEvent::Failure(data.to_owned()))
            },
        )
    })
}

#[cfg(any(feature = "websocket", test))]
pub(super) fn classify_websocket(data: &str) -> WireEvent<ResponsesEvent> {
    if data.trim() == "[DONE]" {
        return WireEvent::Known(ResponsesEvent::Sentinel);
    }
    if serde_json::from_str::<Value>(data)
        .is_ok_and(|v| v.get("type").and_then(Value::as_str) == Some("error"))
    {
        return WireEvent::Known(ResponsesEvent::Failure(data.to_owned()));
    }
    wire::classify_or_untagged(
        data,
        "type",
        |data| {
            wire::classify_tagged_frame::<WebSocketKnown>(
                data,
                "type",
                is_known_responses_event_type,
            )
            .map(|WebSocketKnown(frame)| ResponsesEvent::Frame {
                kind: frame["type"].as_str().expect("strict known tag").to_owned(),
                frame,
                raw: data.to_owned(),
            })
        },
        |data| {
            wire::classify_marker_keyed_frame::<WebSocketWhole>(data, super::WHOLE_BODY_MARKERS)
                .map(|WebSocketWhole(body)| ResponsesEvent::Whole(body))
        },
    )
}

pub(super) fn repair(data: &str) -> Option<String> {
    let mut value: Value = serde_json::from_str(data).ok()?;
    let object = value.as_object_mut()?;
    let kind = object.get("type")?.as_str()?;
    let required: &[&str] = match kind {
        kind if super::is_lifecycle_event(kind) => &["sequence_number"],
        "response.output_item.added"
        | "response.output_item.done"
        | "response.function_call_arguments.delta"
        | "response.function_call_arguments.done"
        | "response.reasoning_text.delta" => &["sequence_number", "output_index"],
        "response.content_part.added"
        | "response.content_part.done"
        | "response.output_text.delta"
        | "response.output_text.done"
        | "response.refusal.delta"
        | "response.refusal.done"
        | "response.reasoning_text.done" => &["sequence_number", "output_index", "content_index"],
        "response.reasoning_summary_part.added"
        | "response.reasoning_summary_part.done"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_summary_text.done" => {
            &["sequence_number", "output_index", "summary_index"]
        }
        _ => return None,
    };
    for field in required {
        object.entry(*field).or_insert_with(|| Value::from(0));
    }
    serde_json::to_string(&value).ok()
}

#[cfg(test)]
mod tests;
