//! Request and response types for the OpenAI Responses API.
//! Endpoint and dialect configuration lives in [`wire`].
//!
//! ```no_run
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.responses(openai::GPT_5_2);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use crate::completion::history::Replay;
use crate::completion::options::{BaseInput, FinalBody, RawAt, Rewrite, request_params};
use crate::error::EncodeError;
use crate::json_utils;
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, Document, DocumentData, DocumentMediaType, DocumentSourceKind, Message,
    MimeType, ToolResultContent, UserContent,
};
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::Mode;
use crate::{completion, message};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

pub mod codex_identity;
/// Caller-held observation custody for one Responses operation.
#[cfg(feature = "completion-observations")]
pub mod observation;
pub mod request_compression;
pub mod request_headers;
pub mod responses_lite;
pub mod streaming;
#[cfg(feature = "completion-observations")]
pub use observation::{
    CaptureCause, CaptureFault, ExistingDiagnosticBasis, Observation, ObservationDispatch,
    ObservationHandle, ObservationProvenance, ObservationStage, ObservationView, ObservationWriter,
    ObservedFailure, ObservedFailureKind, ObservedReplyHead, ObservedResponsesResult,
    ObservedResponsesStream,
};
/// Completion types used only by the observed Responses operation.
///
/// Structural models preserve the diagnostic format independently of
/// the default completion and durable message types.
#[cfg(feature = "completion-observations")]
pub mod observed_types;
/// Private wire types for observed Responses decoding and assembly.
#[cfg(feature = "completion-observations")]
pub(crate) mod observed_wire;
/// Provider-native response retained by an observed Responses operation.
#[cfg(feature = "completion-observations")]
pub use observed_wire::CompletionResponse as ObservedNativeResponse;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod websocket;
pub mod wire;

/// The `input_image` part `image` goes as: a data URL for typed base64
/// data, its URL, or its file id. `None` for any other source, which
/// [`wire::Responses`] tells the adapter it does not carry.
fn image_part(image: &message::Image, omit_detail: bool) -> Option<Value> {
    let (key, source) = match &image.data {
        DocumentSourceKind::Base64(data) => (
            "image_url",
            format!(
                "data:{};base64,{data}",
                image.media_type.as_ref()?.to_mime_type()
            ),
        ),
        DocumentSourceKind::Url(url) => ("image_url", url.clone()),
        DocumentSourceKind::FileId(file_id) => ("file_id", file_id.clone()),
        _ => return None,
    };
    let mut part = json!({"type": "input_image"});
    if !omit_detail {
        set(
            &mut part,
            "detail",
            json!(image.detail.clone().unwrap_or_default()),
        );
    }
    set(&mut part, key, Value::String(source));
    Some(part)
}

/// The content part `document` goes as: an `input_file` for a file id, a URL
/// or base64 PDF data, and its text for a text document. `None` for any other
/// form; the adapter sends a text document's text instead.
fn document_part(document: &Document) -> Option<Value> {
    let source = match &document.data {
        DocumentData::Text(text) => return Some(json!({"type": "input_text", "text": text})),
        DocumentData::File(source) => source,
    };
    Some(match source {
        DocumentSourceKind::FileId(file_id) => json!({"type": "input_file", "file_id": file_id}),
        // `input_file` reads the type of a file it fetches itself.
        DocumentSourceKind::Url(url) => json!({"type": "input_file", "file_url": url}),
        DocumentSourceKind::Base64(data) if document.media_type == Some(DocumentMediaType::PDF) => {
            json!({
                "type": "input_file",
                "file_data": format!("data:application/pdf;base64,{data}"),
                "filename": "document.pdf",
            })
        }
        _ => return None,
    })
}

/// `value` with `key` set to `field`, when `value` is an object.
fn set(value: &mut Value, key: &str, field: Value) {
    if let Some(fields) = value.as_object_mut() {
        fields.insert(key.to_owned(), field);
    }
}

/// The refusal for a part a request was not prepared to carry.
fn unsendable(part: &str) -> EncodeError {
    EncodeError::request(format!(
        "the Responses API cannot carry this {part}; prepare the request first"
    ))
}

/// A tool result's `output`: one text as a string, anything else as its
/// ordered parts.
fn result_output(
    content: &[ToolResultContent],
    omit_image_detail: bool,
) -> Result<Value, EncodeError> {
    let mut parts = content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(json!({"type": "input_text", "text": text.text})),
            ToolResultContent::Json { value } => {
                Ok(json!({"type": "input_text", "text": value.to_string()}))
            }
            ToolResultContent::Image(image) => {
                image_part(image, omit_image_detail).ok_or_else(|| unsendable("tool-result image"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match parts.as_mut_slice() {
        [part] if part.str("type") == Some("input_text") => {
            part.get_mut("text").map(Value::take).unwrap_or_default()
        }
        _ => Value::Array(parts),
    })
}

/// The custom tools of a request: a call to one is a `custom_tool_call`,
/// answered by a `custom_tool_call_output`, as pi decides by name. A result
/// answering a call the request sends takes that call's kind.
struct Custom {
    /// The custom tools the request declares.
    tools: std::collections::HashSet<String>,
    /// Whether each call id the request sends is a custom call.
    calls: std::collections::HashMap<String, bool>,
}

/// The full request's resolved context for lowering later canonical deltas.
#[cfg(feature = "websocket")]
#[derive(Clone, Debug)]
pub(crate) struct DeltaContext {
    pub(crate) model: String,
    pub(crate) custom: std::collections::HashSet<String>,
    pub(crate) stateless: bool,
}

/// A custom result has one verbatim string and cannot carry an image.
fn custom_result_output(content: &[ToolResultContent]) -> Result<String, EncodeError> {
    let mut output = String::new();
    for part in content {
        match part {
            ToolResultContent::Text(text) => output.push_str(&text.text),
            ToolResultContent::Json { value } => output.push_str(&value.to_string()),
            ToolResultContent::Image(_) => {
                return Err(EncodeError::request(
                    "a custom tool result carries text, not an image: the custom_tool_call_output wire item has no image member",
                ));
            }
        }
    }
    Ok(output)
}

struct CarriedMessageGroup<'a> {
    item_index: u64,
    first: &'a message::CarriedText,
    input_index: usize,
}

impl CarriedMessageGroup<'_> {
    fn contains(&self, item_index: u64, part: &message::CarriedText) -> bool {
        self.item_index == item_index
            && self.first.payload.provider == part.payload.provider
            && self.first.item_id == part.item_id
            && self.first.phase == part.phase
    }
}

/// The input items of `history` for `target`, which addresses `model`. A
/// turn the target model produced keeps its provider items (the adapter
/// cleared the rest), sent as pi sends them: a current item verbatim with
/// its call id spelled, an edited block rebuilt under its item's identity,
/// and any other block rebuilt as pi rebuilds another model's turn: text as
/// a completed output message under a synthetic id, and a call with no
/// item id. Reasoning only its item can carry goes only with its identity.
/// A `stateless` request resolves no stored item, so reasoning without its
/// ciphertext is left out and the block after it is rebuilt with no item
/// id, as another model's turn is.
fn input(
    history: &[Message],
    target: &wire::Responses,
    model: &str,
    custom: &mut Custom,
    stateless: bool,
    system_items: &mut Vec<(usize, String)>,
    mut boundaries: Option<&mut Vec<std::ops::Range<usize>>>,
) -> Result<Vec<Value>, EncodeError> {
    let ids = WireIds::for_target(history, target, model).map_err(EncodeError::request)?;
    let omit_image_detail = target.codex_request_shape.is_lite();
    let mut items = Vec::new();
    for (position, message) in history.iter().enumerate() {
        let start = items.len();
        match message {
            Message::System { content } => {
                system_items.push((items.len(), content.clone()));
                items.push(json!({
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": content}],
                }));
            }
            Message::User { content } => {
                for (content_index, part) in content.iter().enumerate() {
                    let part = match part {
                        // Blank text says nothing, and some backends reject it.
                        UserContent::Text(text) if text.text.trim().is_empty() => continue,
                        UserContent::Text(text) => json!({"type": "input_text", "text": text.text}),
                        // A function output has no error field: a failed
                        // result says so in its text.
                        UserContent::ToolResult(result) => {
                            let call_id = ids
                                .spell(position, content_index)
                                .map_err(EncodeError::request)?;
                            items.push(match result.answers {
                                message::AnsweredToolCall::Function => json!({"type": "function_call_output", "call_id": call_id, "output": result_output(&result.content, omit_image_detail)?, "status": "completed"}),
                                message::AnsweredToolCall::Custom => json!({"type": "custom_tool_call_output", "call_id": call_id, "name": result.name, "output": custom_result_output(&result.content)?}),
                            });
                            continue;
                        }
                        UserContent::Image(image) => image_part(image, omit_image_detail)
                            .ok_or_else(|| unsendable("image"))?,
                        UserContent::Document(document) => {
                            document_part(document).ok_or_else(|| unsendable("document"))?
                        }
                        UserContent::Audio(_) => return Err(unsendable("audio")),
                        UserContent::Video(_) => return Err(unsendable("video")),
                    };
                    items.push(json!({"type": "message", "role": "user", "content": [part]}));
                }
            }
            Message::Assistant(turn) => {
                let mut texts = 0usize;
                let mut unpaired = false;
                let mut carried_messages: Vec<CarriedMessageGroup<'_>> = Vec::new();
                for (content_index, block) in turn.content.iter().enumerate() {
                    match block {
                        AssistantContent::ProviderItem(item) => {
                            use crate::completion::ReplayTarget;
                            target.check_provider_item(item)?;
                            items.push(item.payload.item.clone());
                            continue;
                        }
                        AssistantContent::CarriedText(text) => {
                            let part = match text.replay_item().map_err(EncodeError::request)? {
                                message::CarriedTextReplay::ProviderPart(part) => part,
                                message::CarriedTextReplay::KnownText(known) => {
                                    if known.is_empty() {
                                        continue;
                                    }
                                    // Gemini signatures and inspection-only raw fields never
                                    // become Responses request authority.
                                    let item = if let Some(id) =
                                        text.item_id.as_ref().filter(|id| !id.is_empty())
                                    {
                                        json!({"type":"message", "role":"assistant", "status":"completed", "id":id,
                                            "content":[{"type":"output_text", "text":known, "annotations":[]}]})
                                    } else {
                                        json!({"type":"message", "role":"assistant", "content":known})
                                    };
                                    items.push(item);
                                    continue;
                                }
                            };
                            let message::CarriedPosition::MessageContent { item_index, .. } =
                                text.payload.position
                            else {
                                return Err(EncodeError::request(
                                    "a carried Responses message part requires its containing message position",
                                ));
                            };
                            let group = carried_messages
                                .iter()
                                .find(|group| group.contains(item_index, text));
                            if let Some(group) = group {
                                let content = items[group.input_index]
                                    .get_mut("content")
                                    .and_then(Value::as_array_mut)
                                    .ok_or_else(|| {
                                        EncodeError::request(
                                            "a carried Responses message requires a content array",
                                        )
                                    })?;
                                content.push(part);
                            } else {
                                let mut item = json!({"type": "message", "role": "assistant", "status": "completed", "content": [part]});
                                if let Some(id) = text.item_id.as_ref().filter(|id| !id.is_empty())
                                {
                                    set(&mut item, "id", json!(id));
                                }
                                if let Some(phase) = &text.phase {
                                    set(&mut item, "phase", json!(phase));
                                }
                                carried_messages.push(CarriedMessageGroup {
                                    item_index,
                                    first: text,
                                    input_index: items.len(),
                                });
                                items.push(item);
                            }
                            continue;
                        }
                        AssistantContent::CarriedReasoning(reasoning) => {
                            let issuers = crate::providers::openai::wire::replay_issuers(
                                &target.provider.dialect,
                                model,
                            );
                            if incompatible_reasoning(reasoning, &issuers) {
                                return Err(EncodeError::request(
                                    message::UnrepresentableOpaqueContent::new(
                                        "OpenAI Responses with incompatible reasoning provenance",
                                    ),
                                ));
                            }
                            if let Some(item) = carried_reasoning_item(reasoning)? {
                                items.push(item);
                            }
                            continue;
                        }
                        _ => {}
                    }
                    let mut replay = block.replay(target, ids.of(position, content_index));
                    if unpaired && !matches!(block, AssistantContent::Reasoning(_)) {
                        unpaired = false;
                        replay = Replay::Rebuild;
                    }
                    let ciphertext = |item: &Map<String, Value>| {
                        item.get("encrypted_content")
                            .and_then(Value::as_str)
                            .is_some_and(|cipher| !cipher.is_empty())
                    };
                    let unresolvable = stateless
                        && matches!(block, AssistantContent::Reasoning(_))
                        && match &replay {
                            Replay::Item(item) => !item.as_object().is_some_and(ciphertext),
                            Replay::Identity(identity) => !ciphertext(identity),
                            Replay::Rebuild => false,
                        };
                    if unresolvable {
                        unpaired = true;
                        continue;
                    }
                    let identity = match replay {
                        Replay::Item(item) => {
                            if let Some(call_id) = item.str("call_id") {
                                let kind = item.str("type");
                                if matches!(kind, Some("custom_tool_call" | "function_call")) {
                                    custom.calls.insert(
                                        call_id.to_owned(),
                                        kind == Some("custom_tool_call"),
                                    );
                                }
                            }
                            items.push(item.into_owned());
                            continue;
                        }
                        Replay::Identity(identity) => identity,
                        Replay::Rebuild => Map::new(),
                    };
                    let id = |prefix: &str| {
                        identity
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty() && id.starts_with(prefix) && id.len() <= 64)
                            .map(str::to_owned)
                    };
                    let item = match block {
                        AssistantContent::Text(text) => {
                            if text.text.is_empty() {
                                continue;
                            }
                            let synthetic = match texts {
                                0 => format!("msg_rig_{position}"),
                                n => format!("msg_rig_{position}_{n}"),
                            };
                            texts += 1;
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text.text, "annotations": []}],
                                "status": "completed",
                                "id": id("").unwrap_or(synthetic),
                            });
                            if let Some(phase) =
                                identity.get("phase").filter(|phase| phase.is_string())
                            {
                                set(&mut item, "phase", phase.clone());
                            }
                            items.push(item);
                            continue;
                        }
                        AssistantContent::ToolCall(call) => {
                            let call_id = ids
                                .spell(position, content_index)
                                .map_err(EncodeError::request)?;
                            let name = call.function.name.as_str();
                            let kind = identity.get("type").and_then(Value::as_str);
                            let arguments = call.function.arguments_value().to_string();
                            let (kind, key, payload, prefix) = if kind == Some("custom_tool_call")
                                || (kind.is_none() && custom.tools.contains(name))
                            {
                                custom.calls.insert(call_id.clone(), true);
                                let input =
                                    call.function.arguments.get("input").and_then(Value::as_str);
                                let input = input.map_or(arguments, str::to_owned);
                                ("custom_tool_call", "input", input, "ctc_")
                            } else {
                                custom.calls.insert(call_id.clone(), false);
                                let arguments = call
                                    .function
                                    .invalid_arguments
                                    .as_ref()
                                    .map(|raw| match serde_json::from_str::<Value>(raw) {
                                        Ok(value) if !value.is_object() => value.to_string(),
                                        Ok(_) => arguments.clone(),
                                        Err(_) => raw.clone(),
                                    })
                                    .unwrap_or(arguments);
                                ("function_call", "arguments", arguments, "fc_")
                            };
                            let mut item = json!({"type": kind, "call_id": call_id, "name": name});
                            set(&mut item, key, Value::String(payload));
                            if let Some(namespace) = &call.function.namespace {
                                set(&mut item, "namespace", json!(namespace));
                            }
                            if let Some(id) = id(prefix) {
                                set(&mut item, "id", json!(id));
                            }
                            item
                        }
                        AssistantContent::CustomToolCall(call) => {
                            let call_id = ids
                                .spell(position, content_index)
                                .map_err(EncodeError::request)?;
                            custom.calls.insert(call_id.clone(), true);
                            let mut item = json!({"type": "custom_tool_call", "call_id": call_id, "name": call.name, "input": call.input});
                            if let Some(namespace) = &call.namespace {
                                set(&mut item, "namespace", json!(namespace));
                            }
                            if let Some(item_id) = id("") {
                                set(&mut item, "id", json!(item_id));
                            }
                            item
                        }
                        // An edited reasoning block keeps its item's id and
                        // ciphertext, so the items after it stay paired.
                        AssistantContent::Reasoning(reasoning) => {
                            let Some(rs) = id("") else {
                                continue;
                            };
                            let summary: Vec<Value> = (!reasoning.text.is_empty())
                                .then(|| json!({"type": "summary_text", "text": reasoning.text}))
                                .into_iter()
                                .collect();
                            let mut item =
                                json!({"type": "reasoning", "id": rs, "summary": summary});
                            if let Some(ciphertext) = identity.get("encrypted_content") {
                                set(&mut item, "encrypted_content", ciphertext.clone());
                            }
                            item
                        }
                        AssistantContent::Opaque(opaque) if opaque.replay => opaque.item.clone(),
                        AssistantContent::Opaque(_) => continue,
                        AssistantContent::Image(_) => return Err(unsendable("assistant image")),
                        AssistantContent::ProviderItem(_)
                        | AssistantContent::CarriedText(_)
                        | AssistantContent::CarriedReasoning(_) => continue,
                    };
                    items.push(item);
                }
            }
        }
        if let Some(boundaries) = &mut boundaries {
            boundaries.push(start..items.len());
        }
    }
    Ok(items)
}

/// The JSON of a tool choice.
fn tool_choice(choice: message::ToolChoice) -> Result<Value, EncodeError> {
    Ok(match choice {
        message::ToolChoice::Auto => json!("auto"),
        message::ToolChoice::None => json!("none"),
        message::ToolChoice::Required => json!("required"),
        message::ToolChoice::Specific { function_names } => match function_names.as_slice() {
            [] => {
                return Err(EncodeError::request(
                    "ToolChoice::Specific requires at least one function name",
                ));
            }
            [name] => json!({"type": "function", "name": name}),
            names => json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": names.iter().map(|name| json!({"type": "function", "name": name})).collect::<Vec<_>>(),
            }),
        },
    })
}

/// Ask for the reasoning ciphertext, without which reasoning replays only
/// from stored state.
pub(crate) fn include_ciphertext(body: &mut Map<String, Value>) {
    const CIPHERTEXT: &str = "reasoning.encrypted_content";
    let mut include = body
        .get("include")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !include.iter().any(|item| item == CIPHERTEXT) {
        include.push(json!(CIPHERTEXT));
    }
    body.insert("include".to_owned(), Value::Array(include));
}

/// The fork's recognized raw fields, independently of newer typed options.
/// This projection is private; the U canonical request and FinalBody are the
/// only request representations used by the operation and transports.
#[derive(Deserialize, Serialize, Default)]
#[serde(rename = "AdditionalParameters")]
struct RawParameterProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<RawTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<RawInclude>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<RawTruncation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Map::is_empty",
        deserialize_with = "raw_metadata"
    )]
    metadata: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<RawReasoning>,
    // The fork's service-tier reader retained arbitrary provider strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    client_metadata: std::collections::BTreeMap<String, String>,
}

fn raw_metadata<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Map<String, Value>, D::Error> {
    Ok(Option::<Map<String, Value>>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Deserialize, Serialize)]
#[serde(rename = "TextConfig")]
struct RawTextConfig {
    format: RawTextFormat,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawTextFormat {
    JsonSchema(RawStructuredOutput),
    Text,
}

#[derive(Deserialize, Serialize)]
#[serde(rename = "StructuredOutputsInput")]
struct RawStructuredOutput {
    #[serde(default)]
    name: String,
    schema: Value,
    #[serde(default)]
    strict: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(rename = "Reasoning")]
struct RawReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<RawReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<RawReasoningSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<RawReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<RawReasoningContext>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RawReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RawReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RawReasoningMode {
    Pro,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RawReasoningContext {
    Auto,
    AllTurns,
    CurrentTurn,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RawTruncation {
    Auto,
    Disabled,
}

#[derive(Deserialize, Serialize)]
enum RawInclude {
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    #[serde(rename = "computer_call.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
}

/// Apply the fork's raw-field reader before the U option layers merge.
pub(crate) fn project_raw_parameters(raw: &Value) -> Result<Map<String, Value>, EncodeError> {
    if raw.is_null() || raw.is_boolean() {
        return Ok(Map::new());
    }
    let mut raw = raw.clone();
    let tools = if let Some(fields) = raw.as_object_mut() {
        if fields.contains_key("__rig_tolerate_incomplete") {
            return Err(EncodeError::request(
                "`additional_params[\"__rig_tolerate_incomplete\"]` is no longer read: select the incomplete-terminal policy on the Responses wire with `Responses::with_streamed_incomplete`",
            ));
        }
        let tools = fields.remove("tools");
        if let Some(tools) = &tools {
            declared_tools(tools.clone()).map_err(|error| {
                EncodeError::request(format!(
                    "Invalid OpenAI Responses tools payload in additional_params: {error}"
                ))
            })?;
        }
        fields.remove("stream");
        tools
    } else {
        None
    };
    let parameters = serde_json::from_value::<RawParameterProjection>(raw).map_err(|error| {
        EncodeError::request(format!(
            "Invalid OpenAI Responses additional_params payload: {error}"
        ))
    })?;
    let mut fields = match serde_json::to_value(parameters)? {
        Value::Object(fields) => fields,
        _ => unreachable!("the raw parameter projection serializes a struct"),
    };
    if let Some(tools) = tools {
        fields.insert("tools".to_owned(), tools);
    }
    Ok(fields)
}

/// Where a Responses body goes: the HTTP endpoint in a mode, or a WebSocket
/// session's `response.create` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Delivery {
    /// `POST /responses`, streamed or not.
    Http(Mode),
    /// A WebSocket session, which ignores `stream` and `background`.
    WebSocket,
}

/// Whether this request generates output or only prepares stored state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputRequirement {
    NonEmpty,
    MayBeEmpty,
}

#[cfg(feature = "websocket")]
impl InputRequirement {
    pub(crate) fn for_create(options: &websocket::ResponsesWebSocketCreateOptions) -> Self {
        if options.generate == Some(false) {
            Self::MayBeEmpty
        } else {
            Self::NonEmpty
        }
    }
}

impl wire::Responses {
    /// The Responses request body this wire sends: its encoding of the
    /// request, then the mapped options, then the provider options, then
    /// `additional_params`, merged by [`request_params`], then the post-merge
    /// rewrites. The WebSocket session calls it too, so both transports send
    /// one body.
    pub(crate) fn responses_request(
        &self,
        request: &completion::CompletionRequest,
        delivery: Delivery,
    ) -> Result<FinalBody, EncodeError> {
        self.responses_request_with_identity(
            request,
            delivery,
            self.codex_identity.as_ref(),
            InputRequirement::NonEmpty,
        )
    }

    pub(crate) fn responses_request_with_identity(
        &self,
        request: &completion::CompletionRequest,
        delivery: Delivery,
        identity: Option<&codex_identity::CodexIdentity>,
        input_requirement: InputRequirement,
    ) -> Result<FinalBody, EncodeError> {
        self.responses_request_with_rewrites(
            request,
            delivery,
            identity,
            input_requirement,
            &[],
            #[cfg(feature = "websocket")]
            None,
        )
    }

    #[cfg(feature = "websocket")]
    pub(crate) fn websocket_request_with_identity(
        &self,
        request: &completion::CompletionRequest,
        identity: Option<&codex_identity::CodexIdentity>,
        input_requirement: InputRequirement,
        create: websocket::WebSocketCreate,
    ) -> Result<FinalBody, EncodeError> {
        let delta = matches!(&create.input, websocket::WebSocketCreateInput::Delta { .. });
        let rewrites = [Rewrite::ResponsesWebSocketCreate(Box::new(create))];
        if delta {
            return request_params(
                self,
                request,
                |_| {
                    Err(EncodeError::request(
                        "a captured WebSocket delta cannot rebuild its request body",
                    ))
                },
                RawAt::Top,
                &rewrites,
            );
        }
        self.responses_request_with_rewrites(
            request,
            Delivery::WebSocket,
            identity,
            input_requirement,
            &rewrites,
            None,
        )
    }

    /// Build a full frame and retain the context its input was lowered with.
    /// The context is returned only after the merged body passes catalog checks.
    #[cfg(feature = "websocket")]
    pub(crate) fn prepare_websocket_request_with_identity(
        &self,
        request: &completion::CompletionRequest,
        identity: Option<&codex_identity::CodexIdentity>,
        input_requirement: InputRequirement,
        create: websocket::WebSocketCreate,
    ) -> Result<(FinalBody, DeltaContext), EncodeError> {
        if !matches!(&create.input, websocket::WebSocketCreateInput::Full) {
            return Err(EncodeError::request(
                "a websocket delta context must be captured from a full request",
            ));
        }
        let mut context = None;
        let body = self.responses_request_with_rewrites(
            request,
            Delivery::WebSocket,
            identity,
            input_requirement,
            &[Rewrite::ResponsesWebSocketCreate(Box::new(create))],
            Some(&mut context),
        )?;
        let context = context.ok_or_else(|| {
            EncodeError::request("the full websocket request did not capture its delta context")
        })?;
        Ok((body, context))
    }

    /// Lower canonical messages as one history, preserving literal raw items
    /// in their original positions. Delta systems remain in input.
    #[cfg(feature = "websocket")]
    pub(crate) fn lower_websocket_delta(
        &self,
        delta: websocket::InputDelta,
        context: &DeltaContext,
    ) -> Result<websocket::WireDelta, EncodeError> {
        let entries = delta.into_vec();
        let history: Vec<Message> = entries
            .iter()
            .filter_map(|entry| match entry {
                websocket::InputDeltaItem::Canonical(message) => Some(message.clone()),
                websocket::InputDeltaItem::Raw(_) => None,
            })
            .collect();
        let mut custom = Custom {
            tools: context.custom.clone(),
            calls: Default::default(),
        };
        let mut boundaries = Vec::with_capacity(history.len());
        let canonical = input(
            &history,
            self,
            &context.model,
            &mut custom,
            context.stateless,
            &mut Vec::new(),
            Some(&mut boundaries),
        )?;
        let mut boundaries = boundaries.into_iter();
        let mut canonical = canonical.into_iter();
        let mut items = Vec::new();
        for entry in entries {
            match entry {
                websocket::InputDeltaItem::Canonical(_) => {
                    let boundary = boundaries.next().ok_or_else(|| {
                        EncodeError::request("a canonical websocket delta has no output boundary")
                    })?;
                    items.extend(canonical.by_ref().take(boundary.len()));
                }
                websocket::InputDeltaItem::Raw(value) => items.push(value),
            }
        }
        websocket::WireDelta::new(items).map_err(EncodeError::request)
    }

    fn responses_request_with_rewrites(
        &self,
        request: &completion::CompletionRequest,
        delivery: Delivery,
        identity: Option<&codex_identity::CodexIdentity>,
        input_requirement: InputRequirement,
        ending_rewrites: &[Rewrite],
        #[cfg(feature = "websocket")] delta_context: Option<&mut Option<DeltaContext>>,
    ) -> Result<FinalBody, EncodeError> {
        use crate::completion::ReplayTarget;
        self.check_provider_items(&request.chat_history)?;
        let model = request.model.as_deref().unwrap_or(&self.model);
        let issuers = crate::providers::openai::wire::replay_issuers(&self.provider.dialect, model);
        for message in &request.chat_history {
            if let Message::Assistant(turn) = message {
                for block in &turn.content {
                    match block {
                        AssistantContent::CarriedText(text) => {
                            text.replay_item().map_err(EncodeError::request)?;
                        }
                        AssistantContent::CarriedReasoning(reasoning)
                            if incompatible_reasoning(reasoning, &issuers) =>
                        {
                            return Err(EncodeError::request(
                                message::UnrepresentableOpaqueContent::new(
                                    "OpenAI Responses with incompatible reasoning provenance",
                                ),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        let codex =
            self.provider.dialect.quirks.responses.contract == wire::ResponsesContract::Codex;
        responses_lite::validate_activation(
            self.codex_request_shape,
            self.provider.dialect.quirks.responses.contract,
            self.provider.dialect.name,
            self.system_instructions,
            identity,
        )
        .map_err(EncodeError::request)?;
        let mut rewrites = vec![Rewrite::ResponsesTypedControls];
        match delivery {
            Delivery::Http(Mode::Streaming) => rewrites.push(Rewrite::Stream(true)),
            Delivery::Http(Mode::Unary) | Delivery::WebSocket => rewrites.push(Rewrite::NoStream),
        }
        if delivery == Delivery::WebSocket {
            rewrites.push(Rewrite::NoBackground);
        }
        if codex {
            rewrites.push(Rewrite::ResponsesCodex);
        }
        rewrites.push(Rewrite::ResponsesLite {
            shape: self.codex_request_shape,
            contract: self.provider.dialect.quirks.responses.contract,
            dialect: self.provider.dialect.name,
            placement: self.system_instructions,
            identity: identity.cloned(),
        });
        rewrites.push(Rewrite::ResponsesReasoningCiphertext);
        rewrites.push(Rewrite::ResponsesIdentity(identity.cloned()));
        rewrites.extend_from_slice(ending_rewrites);
        let body = request_params(
            self,
            request,
            |layers| {
                self.base(
                    request,
                    codex,
                    input_requirement,
                    layers,
                    #[cfg(feature = "websocket")]
                    delta_context,
                )
            },
            RawAt::Top,
            &rewrites,
        )?;
        crate::providers::openai::options::check_body(
            self,
            request,
            body,
            crate::providers::openai::options::Endpoint::Responses,
        )
    }

    /// This wire's own encoding of `request`: model, input, instructions,
    /// tools and the caller's typed fields. Codex restrictions are checked
    /// against the merged body so an explicit choice cannot disappear.
    fn base(
        &self,
        request: &completion::CompletionRequest,
        codex: bool,
        input_requirement: InputRequirement,
        layers: &mut BaseInput<'_>,
        #[cfg(feature = "websocket")] delta_context: Option<&mut Option<DeltaContext>>,
    ) -> Result<Map<String, Value>, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        if layers.param("__rig_tolerate_incomplete").is_some() {
            return Err(EncodeError::request(
                "`additional_params[\"__rig_tolerate_incomplete\"]` is no longer read: select the incomplete-terminal policy on the Responses wire with `Responses::with_streamed_incomplete`",
            ));
        }
        let mut tools: Vec<ResponsesRequestTool> = request
            .tools
            .iter()
            .cloned()
            .map(ResponsesToolDefinition::from)
            .map(ResponsesRequestTool::Defined)
            .collect();
        let extra = layers.raw_tools()?;
        if !extra.is_empty() {
            tools.extend(declared_tools(Value::Array(extra)).map_err(|err| {
                EncodeError::request(format!(
                    "Invalid OpenAI Responses tools payload in additional_params: {err}"
                ))
            })?);
        }
        tools.extend(
            self.tools
                .iter()
                .cloned()
                .map(ResponsesRequestTool::Defined),
        );
        if self.strict_tools {
            tools = tools
                .into_iter()
                .map(ResponsesRequestTool::with_strict)
                .collect();
        }
        let mut custom = Custom {
            tools: tools
                .iter()
                .filter(|tool| tool.definition().kind == "custom")
                .map(|tool| tool.definition().name.clone())
                .collect(),
            calls: Default::default(),
        };
        let stateless = codex || layers.param("store") == Some(&json!(false));
        #[cfg(feature = "websocket")]
        if let Some(context) = delta_context {
            let model = match layers.param("model") {
                Some(Value::String(model)) => model.clone(),
                None => model.clone(),
                Some(_) => {
                    return Err(EncodeError::request(
                        "the full websocket request's model must be a string",
                    ));
                }
            };
            let custom = match layers.param("tools") {
                Some(tools) => declared_tools(tools.clone())
                    .map_err(EncodeError::request)?
                    .iter()
                    .filter(|tool| tool.definition().kind == "custom")
                    .map(|tool| tool.definition().name.clone())
                    .collect(),
                None => custom.tools.clone(),
            };
            *context = Some(DeltaContext {
                model,
                custom,
                stateless,
            });
        }
        let mut system_items = Vec::new();
        let mut items = input(
            &request.chat_history,
            self,
            &model,
            &mut custom,
            stateless,
            &mut system_items,
            None,
        )?;
        let before = items.len();
        let mut lifted = Vec::new();
        match self.system_instructions {
            // Only canonical system messages are lifted. Carried raw items
            // keep their bytes even when their fields resemble a system item.
            SystemInstructionsPlacement::Instructions => {
                let leading = system_items
                    .iter()
                    .enumerate()
                    .take_while(|(expected, (index, _))| expected == index)
                    .count();
                if leading < items.len() || input_requirement == InputRequirement::MayBeEmpty {
                    lifted.extend(
                        system_items
                            .iter()
                            .take(leading)
                            .map(|(_, text)| text.clone()),
                    );
                    items.drain(..leading);
                }
            }
            SystemInstructionsPlacement::AllInstructions => {
                let mut index = 0;
                items.retain(|_| {
                    let system = system_items.iter().find(|(position, _)| *position == index);
                    index += 1;
                    if let Some((_, text)) = system {
                        lifted.push(text.clone());
                    }
                    system.is_none()
                });
            }
            SystemInstructionsPlacement::InputSystemMessages => {}
        }
        if items.is_empty() && input_requirement == InputRequirement::NonEmpty {
            return Err(EncodeError::request(if items.len() < before {
                "OpenAI Responses request input must contain at least one non-system item \
             (system messages were lifted into the top-level `instructions` field)"
            } else {
                "OpenAI Responses request input must contain at least one item"
            }));
        }
        let lifted: Vec<&str> = lifted
            .iter()
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .collect();
        let lifted = lifted.join("\n\n");
        // A gateway's own instructions go ahead of the caller's, once.
        let instructions = match &self.provider.instructions {
            Some(gateway) if lifted.is_empty() => Some(gateway.clone()),
            Some(gateway) if !lifted.contains(gateway.as_str()) => {
                Some(format!("{gateway}\n\n{lifted}"))
            }
            _ => (!lifted.is_empty()).then_some(lifted),
        };
        let text = request
            .output_schema
            .clone()
            .map(|schema| {
                let (name, schema) = super::structured_output_schema(schema);
                json!({"format": {"type": "json_schema", "name": name, "schema": schema, "strict": true}})
            });

        let fields = [
            ("model", Some(Value::String(model))),
            ("input", Some(Value::Array(items))),
            ("instructions", instructions.map(Value::from)),
            ("max_output_tokens", request.max_tokens.map(Value::from)),
            ("temperature", request.temperature.map(Value::from)),
            (
                "tool_choice",
                request.tool_choice.clone().map(tool_choice).transpose()?,
            ),
            ("tools", (!tools.is_empty()).then(|| json!(tools))),
            ("text", text),
        ];
        Ok(fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect())
    }
}

/// A function or hosted tool available to a Responses request.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct ResponsesToolDefinition {
    /// The type of tool.
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name
    #[serde(default)]
    pub name: String,
    /// Parameters - this should be a JSON schema. Strict function tools must use OpenAI's supported strict schema subset.
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Whether to use strict mode. Disabled by default; opt in with [`Self::with_strict`]
    /// or [`wire::Responses::with_strict_tools`].
    ///
    /// Always serialized on a function tool: the Responses API treats an omitted `strict`
    /// as "attempt strict mode", so `false` must reach the wire for non-strict tools to
    /// actually be non-strict. Never serialized on a hosted tool, which answers the field
    /// with a 400 (`Unknown parameter: 'tools[0].strict'`).
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub strict: bool,
    /// Tool description.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub description: String,
    /// Additional provider-specific configuration for hosted tools.
    #[serde(flatten, default)]
    pub config: Map<String, Value>,
}

impl Serialize for ResponsesToolDefinition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", &self.kind)?;
        if !self.name.is_empty() {
            map.serialize_entry("name", &self.name)?;
        }
        if !self.parameters.is_null() {
            map.serialize_entry("parameters", &self.parameters)?;
        }
        if self.kind == "function" {
            map.serialize_entry("strict", &self.strict)?;
        }
        if !self.description.is_empty() {
            map.serialize_entry("description", &self.description)?;
        }
        for (key, value) in &self.config {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl ResponsesToolDefinition {
    /// Creates a function tool definition with strict mode disabled.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            name: name.into(),
            parameters,
            strict: false,
            description: description.into(),
            config: Map::new(),
        }
    }

    /// Creates a strict function tool definition.
    ///
    /// The schema is sanitized to OpenAI's strict subset (`additionalProperties: false`
    /// added and every property forced into `required`).
    pub fn strict_function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self::function(name, description, parameters).with_strict()
    }

    /// Enables strict mode for this function tool.
    ///
    /// Function schemas are sanitized to OpenAI's strict subset. Hosted tools are
    /// returned unchanged because strict mode only applies to function tools.
    pub fn with_strict(mut self) -> Self {
        if self.kind == "function" {
            super::sanitize_schema(&mut self.parameters);
            self.strict = true;
        }
        self
    }

    /// Creates a hosted tool definition for an arbitrary hosted tool type.
    pub fn hosted(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            name: String::new(),
            parameters: Value::Null,
            strict: false,
            description: String::new(),
            config: Map::new(),
        }
    }

    /// Creates a hosted `web_search` tool definition.
    pub fn web_search() -> Self {
        Self::hosted("web_search")
    }

    /// Creates a hosted `file_search` tool definition.
    pub fn file_search() -> Self {
        Self::hosted("file_search")
    }

    /// Creates a hosted `computer_use` tool definition.
    pub fn computer_use() -> Self {
        Self::hosted("computer_use")
    }

    /// Adds hosted-tool configuration fields.
    pub fn with_config(mut self, key: impl Into<String>, value: Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

impl From<completion::ToolDefinition> for ResponsesToolDefinition {
    fn from(value: completion::ToolDefinition) -> Self {
        let completion::ToolDefinition {
            name,
            parameters,
            description,
        } = value;

        Self::function(name, description, parameters)
    }
}

/// One entry of a Responses request's `tools` array.
///
/// Rig builds most tools from typed [`ResponsesToolDefinition`]s, and those
/// serialize exactly as the typed shape says, `strict` included. A tool supplied
/// through `additional_params["tools"]` is different: it is the caller's own
/// declaration, and the provider must receive the JSON values the caller wrote.
/// Round-tripping it through [`ResponsesToolDefinition`] changes it: the typed
/// shape omits an empty `name` or `description` and a null `parameters`, and it
/// always writes `strict`, so an explicit `"description": ""` on a namespace
/// disappears and a namespace or hosted tool written without `strict` gains one.
/// A provider that requires the member, or rejects the added one, then refuses
/// the request. So a declared tool keeps its original JSON object and serializes
/// that object, while its typed view stays available to code that inspects or
/// normalizes the request.
///
/// A declared tool is preserved except for a transformation the caller asked
/// for: strict mode sets `strict` and sanitizes `parameters` on a declared
/// function, and leaves every other member as written. A declared namespace or
/// hosted tool is never changed. The fidelity is of JSON values, not of bytes:
/// key order and whitespace may change when the object is serialized again.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponsesRequestTool {
    /// A tool rig built from a typed definition.
    Defined(ResponsesToolDefinition),
    /// A tool declared verbatim through `additional_params["tools"]`.
    Declared(DeclaredResponsesTool),
}

/// A caller-declared tool: the JSON object as written, and its typed view.
///
/// Both are fixed at construction and cannot be changed independently, so the
/// object that is serialized is always the one the typed view was parsed from.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredResponsesTool {
    value: Map<String, Value>,
    definition: ResponsesToolDefinition,
}

impl DeclaredResponsesTool {
    /// Parse one declared tool, keeping the object it came from.
    ///
    /// The typed parse still runs, so a declaration that is not a tool object is
    /// refused as it was before: a non-object, a missing `type`, or a member of
    /// the wrong JSON type.
    pub fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        let Value::Object(object) = value else {
            return Err(<serde_json::Error as serde::de::Error>::custom(
                "a Responses tool declaration must be a JSON object",
            ));
        };
        let definition =
            serde_json::from_value::<ResponsesToolDefinition>(Value::Object(object.clone()))?;
        Ok(Self {
            value: object,
            definition,
        })
    }

    /// The JSON object the caller declared, which is what gets serialized.
    pub fn value(&self) -> &Map<String, Value> {
        &self.value
    }

    /// The typed view of the declaration.
    pub fn definition(&self) -> &ResponsesToolDefinition {
        &self.definition
    }
}

/// The tools declared in `additional_params["tools"]`, each kept as written.
fn declared_tools(raw: Value) -> Result<Vec<ResponsesRequestTool>, serde_json::Error> {
    let Value::Array(items) = raw else {
        return Err(<serde_json::Error as serde::de::Error>::custom(
            "`tools` must be a JSON array of tool declarations",
        ));
    };
    items
        .into_iter()
        .map(|item| DeclaredResponsesTool::from_value(item).map(ResponsesRequestTool::Declared))
        .collect()
}

impl ResponsesRequestTool {
    /// The typed view of this tool, whichever way it was supplied.
    pub fn definition(&self) -> &ResponsesToolDefinition {
        match self {
            Self::Defined(definition) => definition,
            Self::Declared(declared) => &declared.definition,
        }
    }

    /// Enables strict mode, as [`ResponsesToolDefinition::with_strict`] does.
    ///
    /// A declared tool stays a declaration. Strict mode's own transformation,
    /// `strict: true` and the sanitized `parameters` schema, is written into the
    /// declared object, and every other member the caller wrote is kept as it was.
    /// That includes an empty `description` and members the typed view does not
    /// model. Only a function tool changes; a hosted tool or a namespace comes back
    /// untouched, with no `strict` added. Applying it twice changes nothing further.
    pub fn with_strict(self) -> Self {
        match self {
            Self::Defined(definition) => Self::Defined(definition.with_strict()),
            Self::Declared(mut declared) => {
                let rewritten = declared.definition.clone().with_strict();
                if rewritten.strict != declared.definition.strict {
                    declared
                        .value
                        .insert("strict".to_string(), Value::Bool(rewritten.strict));
                }
                if rewritten.parameters != declared.definition.parameters {
                    declared
                        .value
                        .insert("parameters".to_string(), rewritten.parameters.clone());
                }
                declared.definition = rewritten;
                Self::Declared(declared)
            }
        }
    }

    pub(crate) fn normalize(self) -> Self {
        self.with_strict()
    }
}

impl From<ResponsesToolDefinition> for ResponsesRequestTool {
    fn from(definition: ResponsesToolDefinition) -> Self {
        Self::Defined(definition)
    }
}

impl Serialize for ResponsesRequestTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Defined(definition) => definition.serialize(serializer),
            Self::Declared(declared) => declared.value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ResponsesRequestTool {
    /// A deserialized request's tools are the JSON a caller wrote, so each one is
    /// kept as a declaration rather than re-derived from its typed view.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        DeclaredResponsesTool::from_value(value)
            .map(Self::Declared)
            .map_err(serde::de::Error::custom)
    }
}

/// Controls where Rig system instructions are placed in an OpenAI Responses request.
///
/// Serialized because it is a field of the [`wire::Responses`] wire, which is
/// data a host may store.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemInstructionsPlacement {
    /// Send the leading run of system instructions (the preamble and any system
    /// messages that open the conversation) through the official top-level
    /// `instructions` field. Mid-conversation system messages keep their
    /// position in `input`.
    #[default]
    Instructions,
    /// Send every system message through the top-level `instructions` field,
    /// including mid-conversation ones.
    ///
    /// Use this for backends that reject the `system` role in `input` entirely.
    AllInstructions,
    /// Send system instructions as `system` messages in `input`.
    ///
    /// Use this only for OpenAI-compatible providers that do not support top-level
    /// `instructions`.
    InputSystemMessages,
}

#[cfg(test)]
mod corrupt_frame_fixtures;
#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod opaque_content_tests;
#[cfg(test)]
mod openapi_schema;
#[cfg(test)]
mod stateless_replay_tests;
#[cfg(test)]
mod streaming_conformance;
#[cfg(test)]
mod tests;
#[cfg(all(test, feature = "websocket"))]
mod websocket_delta_tests;

/// Unknown parent-message fields have inspection custody, never request authority.
pub(crate) fn unknown_message_fields(item: &Value) -> serde_json::Map<String, Value> {
    item.as_object()
        .into_iter()
        .flat_map(|fields| fields.iter())
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "type" | "id" | "role" | "status" | "content" | "phase"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Carry every part of one message so edits cannot discard unknown siblings.
pub(crate) fn carried_message_parts(
    item: &Value,
    item_index: u64,
    issuer: Option<String>,
) -> Result<Vec<message::CarriedText>, serde_json::Error> {
    use message::{CarriedPayload, CarriedPosition, CarriedText, CarriedTextContent};
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            <serde_json::Error as serde::de::Error>::custom("message content must be an array")
        })?;
    parts
        .iter()
        .enumerate()
        .map(|(part_index, raw)| {
            let kind = raw.get("type").and_then(Value::as_str).ok_or_else(|| {
                <serde_json::Error as serde::de::Error>::custom(
                    "content part requires an object with a string `type`",
                )
            })?;
            let content = match kind {
                "output_text" => CarriedTextContent::Text(required_part_text(raw, "text")?),
                "refusal" => CarriedTextContent::Refusal(required_part_text(raw, "refusal")?),
                _ => CarriedTextContent::Opaque(String::new()),
            };
            let mut part = CarriedText::new(
                content,
                CarriedPayload::new(raw.clone(), issuer.clone()).with_position(
                    CarriedPosition::MessageContent {
                        item_index,
                        part_index: part_index as u64,
                    },
                ),
            );
            part = part.with_unknown_message_fields(unknown_message_fields(item));
            part.item_id = item.str("id").map(str::to_owned);
            part.phase = item.str("phase").map(str::to_owned);
            Ok(part)
        })
        .collect()
}

fn required_part_text(part: &Value, key: &str) -> Result<String, serde_json::Error> {
    part.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            <serde_json::Error as serde::de::Error>::custom(format!(
                "content part requires a string `{key}`"
            ))
        })
}

/// Decode ordered reasoning without interpreting or replacing its issuer.
pub(crate) fn carried_reasoning(
    item: &Value,
    issuer: Option<String>,
) -> Result<message::CarriedReasoning, serde_json::Error> {
    use message::{
        CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart as Part,
    };
    let mut blocks = Vec::new();
    if let Some(summary) = item.get("summary") {
        let summary = summary.as_array().ok_or_else(|| {
            <serde_json::Error as serde::de::Error>::custom("reasoning summary must be an array")
        })?;
        for (index, raw) in summary.iter().enumerate() {
            let kind = required_part_text(raw, "type")?;
            blocks.push(if kind == "summary_text" {
                Part::Summary(required_part_text(raw, "text")?)
            } else {
                Part::OpaqueSummary(
                    CarriedPayload::new(raw.clone(), issuer.clone()).with_position(
                        CarriedPosition::ReasoningSummary {
                            part_index: index as u64,
                        },
                    ),
                )
            });
        }
    }
    if let Some(content) = item.get("content") {
        let parts = match content {
            Value::String(_) => std::slice::from_ref(content),
            Value::Array(parts) => parts.as_slice(),
            _ => {
                return Err(<serde_json::Error as serde::de::Error>::custom(
                    "reasoning content must be an array or string",
                ));
            }
        };
        for (index, raw) in parts.iter().enumerate() {
            let text = match raw {
                Value::String(text) => Some(text.clone()),
                _ if required_part_text(raw, "type")? == "reasoning_text" => {
                    Some(required_part_text(raw, "text")?)
                }
                _ => None,
            };
            blocks.push(match text {
                Some(text) => Part::Text {
                    text,
                    signature: None,
                },
                None => Part::OpaqueContent(
                    CarriedPayload::new(raw.clone(), issuer.clone()).with_position(
                        CarriedPosition::ReasoningContent {
                            part_index: index as u64,
                        },
                    ),
                ),
            });
        }
    }
    let optional_text = |key: &str| -> Result<Option<String>, serde_json::Error> {
        item.get(key)
            .cloned()
            .map(serde_json::from_value::<Option<String>>)
            .transpose()
            .map(Option::flatten)
    };
    if let Some(signature) = optional_text("signature")? {
        match blocks.iter_mut().rev().find_map(|part| match part {
            Part::Text { signature, .. } => Some(signature),
            _ => None,
        }) {
            Some(slot) => *slot = Some(signature),
            None => blocks.push(Part::Text {
                text: String::new(),
                signature: Some(signature),
            }),
        }
    }
    if let Some(data) = optional_text("encrypted_content")?.filter(|data| !data.is_empty()) {
        blocks.push(Part::Encrypted(data));
    }
    Ok(
        CarriedReasoning::new(blocks, CarriedPayload::new(item.clone(), issuer))
            .with_id(optional_text("id")?),
    )
}

fn incompatible_reasoning(reasoning: &message::CarriedReasoning, issuers: &[String]) -> bool {
    use message::CarriedReasoningPart as Part;
    reasoning.has_opaque_parts()
        && (!issuers.iter().any(|issuer| reasoning.replayable_to(issuer))
            || reasoning.content.iter().any(|part| match part {
                Part::OpaqueSummary(payload) | Part::OpaqueContent(payload) => {
                    !issuers.iter().any(|issuer| payload.replayable_to(issuer))
                }
                _ => false,
            }))
}

/// Rebuild the fork's reasoning categories while retaining raw sibling fields.
fn carried_reasoning_item(
    reasoning: &message::CarriedReasoning,
) -> Result<Option<Value>, EncodeError> {
    use message::CarriedReasoningPart as Part;
    let Some(id) = &reasoning.id else {
        if reasoning.has_opaque_parts() {
            return Err(EncodeError::request(
                message::UnrepresentableOpaqueContent::new(
                    "OpenAI Responses without a reasoning item ID",
                ),
            ));
        }
        return Ok(None);
    };
    if id.is_empty() && reasoning.has_opaque_parts() {
        return Err(EncodeError::request(
            message::UnrepresentableOpaqueContent::new(
                "OpenAI Responses without a reasoning item ID",
            ),
        ));
    }
    let mut item = match &reasoning.payload.item {
        Value::Object(item) => item.clone(),
        Value::Null
            if reasoning.payload.provider.is_none()
                && !id.is_empty()
                && !reasoning.has_opaque_parts() =>
        {
            Map::new()
        }
        _ => {
            return Err(EncodeError::request(
                "a carried Responses reasoning item must be an object",
            ));
        }
    };
    let mut summary = Vec::new();
    let mut content = Vec::new();
    let mut encrypted = None;
    let mut signature = None;
    let raw_summary = reasoning
        .payload
        .item
        .get("summary")
        .and_then(Value::as_array);
    let raw_content = reasoning
        .payload
        .item
        .get("content")
        .and_then(Value::as_array);
    let known = |raw: Option<&Value>, kind: &str, text: &str| {
        let mut value = raw
            .filter(|raw| raw.str("type") == Some(kind))
            .cloned()
            .unwrap_or_else(|| json!({"type": kind}));
        set(&mut value, "text", json!(text));
        value
    };
    for part in &reasoning.content {
        match part {
            Part::Summary(text) => summary.push(known(
                raw_summary.and_then(|parts| parts.get(summary.len())),
                "summary_text",
                text,
            )),
            Part::OpaqueSummary(raw) => summary.push(raw.item.clone()),
            Part::Text {
                text,
                signature: signed,
            } => {
                if !(text.is_empty() && signed.is_some()) {
                    content.push(known(
                        raw_content.and_then(|parts| parts.get(content.len())),
                        "reasoning_text",
                        text,
                    ));
                }
                if let Some(signed) = signed {
                    signature = Some(signed.clone());
                }
            }
            Part::OpaqueContent(raw) => content.push(raw.item.clone()),
            Part::Encrypted(data) | Part::Redacted { data } => {
                encrypted.get_or_insert_with(|| data.clone());
            }
        }
    }
    item.insert("type".to_owned(), json!("reasoning"));
    item.insert("id".to_owned(), json!(id));
    item.insert("summary".to_owned(), Value::Array(summary));
    item.remove("status");
    item.remove("content");
    item.remove("encrypted_content");
    item.remove("signature");
    if !content.is_empty() {
        item.insert("content".to_owned(), Value::Array(content));
    }
    if let Some(data) = encrypted {
        item.insert("encrypted_content".to_owned(), json!(data));
    }
    if let Some(data) = signature {
        item.insert("signature".to_owned(), json!(data));
    }
    Ok(Some(Value::Object(item)))
}

#[cfg(test)]
mod carried_tests;

#[cfg(test)]
mod parent_fields_tests;
