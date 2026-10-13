//! The Gemini `GenerateContent` completion wire over gRPC.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_gemini_grpc::GeminiGrpcError> {
//! let model = GeminiGrpc::new("API_KEY").await?.completion(GEMINI_2_5_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use futures::StreamExt;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::EncodeError;
use rig_core::error::ProviderError;
use rig_core::message;
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion as rest;
use rig_core::wire::{Descriptor, Mode, Wire};

use super::GeminiGrpc;
use super::proto::{GenerateContentRequest, GenerateContentResponse};

/// The `GenerateContent` endpoint for one model: `GenerateContent` for a
/// unary call, `StreamGenerateContent` for a streamed one.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateContent {
    pub model: String,
}

impl GenerateContent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = GenerateContentRequest;
    type Frame = GenerateContentResponse;
    type Decoder<'id> = crate::streaming::GrpcAdapter;
    type Reassembler = rig_core::providers::gemini::streaming::document::GenerateContentResponse;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    /// The REST wire's request, transcoded: every field the shared encoder
    /// builds reaches the protobuf request, and one the proto does not
    /// declare is an error rather than dropped.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<GenerateContentRequest, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let body = rest::request_body(&request, self, &model, None, |body| {
            strip_source_user_image_data_uris(&request, body);
            body.insert("model".to_owned(), format!("models/{model}").into());
        })?;
        Ok(crate::rest::from_rest(serde_json::to_value(&body)?)?)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        crate::streaming::GrpcAdapter::default()
    }
}

// The shared base emits one part per user content, in order, even when it
// separates function results into their own contents. Follow that source order
// before the raw merge so only canonical user images receive this legacy strip.
fn strip_source_user_image_data_uris(
    request: &CompletionRequest,
    body: &mut serde_json::Map<String, serde_json::Value>,
) {
    use message::{DocumentSourceKind, Message, UserContent};
    let mut source = request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            Message::System { .. } | Message::Assistant(_) => None,
        })
        .flatten();
    let Some(contents) = body
        .get_mut("contents")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for content in contents {
        if content.get("role").and_then(serde_json::Value::as_str) != Some("user") {
            continue;
        }
        let Some(parts) = content
            .get_mut("parts")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for part in parts {
            let Some(UserContent::Image(image)) = source.next() else {
                continue;
            };
            let DocumentSourceKind::Base64(data) = &image.data else {
                continue;
            };
            let Some((_, data)) = data
                .trim()
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(','))
            else {
                continue;
            };
            if let Some(encoded) = part.pointer_mut("/inlineData/data") {
                *encoded = serde_json::Value::String(data.to_owned());
            }
        }
    }
}
impl rig_core::completion::ReplayTarget for GenerateContent {
    /// The GenerateContent mapping, for what the gRPC request declares.
    fn map_options(
        &self,
        request: &rig_core::completion::CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        rest::generate_content_options(model, rest::Route::Grpc, fields)
    }

    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("gemini.generate_content")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// What the model reads, as on every GenerateContent wire.
    fn accepts(&self, model: &str) -> rig_core::completion::Accepts {
        rest::accepts(model)
    }

    /// The media the Gemini API takes, as on the REST wire.
    fn encodes(&self, _model: &str, media: rig_core::completion::Media<'_>) -> bool {
        rest::encodes(media, false)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&message::Origin>,
    ) -> String {
        rest::normalize_tool_call_id(model, id)
    }

    /// Gemini takes system text only in `systemInstruction`: later system
    /// messages fold into the leading one, as pi's `collapseSystemMessages`.
    fn later_system(&self, _model: &str) -> rig_core::completion::LaterSystem {
        rig_core::completion::LaterSystem::Leading
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        rest::CALL_ID_SLOT
    }

    /// Tools in `additional_params` or a cached content count, as on the
    /// REST wire.
    fn declares_tools(&self, request: &rig_core::completion::CompletionRequest) -> bool {
        rest::declares_tools(self, request)
    }
}

impl Transport<GenerateContent> for GeminiGrpc {
    fn send(
        &self,
        request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let mode = exchange.mode;
        let mut client = self.grpc_client();
        Opening::new(async move {
            Ok(match mode {
                Mode::Unary => match client.generate_content(request).await {
                    Ok(response) => unary(response.into_inner())?,
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
                Mode::Streaming => match client.stream_generate_content(request).await {
                    Ok(response) => {
                        let mut chunks = response.into_inner();
                        // Stop receiving after a tonic failure.
                        Opened::new(async_stream::stream! {
                            while let Some(item) = chunks.next().await {
                                match item {
                                    Ok(chunk) => yield Ok(chunk),
                                    Err(status) => {
                                        yield Err(rpc_error(&status));
                                        break;
                                    }
                                }
                            }
                        })
                    }
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
            })
        })
    }
}

/// A unary reply: its one message, whose REST JSON is the response's `raw`.
/// A message that does not transcode fails as its decoding would.
pub(crate) fn unary(
    response: GenerateContentResponse,
) -> Result<Opened<GenerateContentResponse>, ProviderError> {
    let document = crate::streaming::document::rest_document(&response)?;
    Ok(Opened::new(futures::stream::iter([Ok(response)])).with_document(document))
}

/// Stable descriptor name reported on normalized responses from this provider.
pub const PROVIDER_NAME: &str = "gemini-grpc";

/// Preserves tonic status display text with RPC code and retry classification.
/// Transport failures use the same provider-body representation.
pub(crate) fn rpc_error(status: &tonic::Status) -> ProviderError {
    ProviderError::from_provider_body(status.to_string())
        .with_provider_code(Some(grpc_code_name(status.code())))
}

/// The gRPC status code's canonical name (`UNAVAILABLE`): the code the
/// provider answered with, kept apart from the message so a report can
/// key on it.
pub(crate) fn grpc_code_name(code: tonic::Code) -> String {
    format!("{code:?}")
        .chars()
        .fold(String::new(), |mut name, c| {
            if c.is_ascii_uppercase() && !name.is_empty() {
                name.push('_');
            }
            name.push(c.to_ascii_uppercase());
            name
        })
}

/// Recognizes transient gRPC codes; other codes are non-transient.
pub(crate) fn transient_grpc_code(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unavailable
            | tonic::Code::ResourceExhausted
            | tonic::Code::DeadlineExceeded
            | tonic::Code::Aborted
    )
}

pub(crate) fn create_grpc_request(
    model: &str,
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, ProviderError> {
    let CompletionRequest {
        model: _,
        chat_history,
        documents: _,
        tools,
        temperature,
        max_tokens,
        tool_choice: _,
        additional_params: _,
        output_schema: _,
        record_telemetry_content: _,
    } = completion_request;

    let mut chat_history = chat_history;
    rig_core::message::retain_replayable_reasoning(&mut chat_history, &[REASONING_ISSUER]);
    let (history_system, mut chat_history) = split_system_messages_from_history(chat_history);
    // functionResponse.name keys the replay: cross-provider ingested
    // results arrive with an empty name and their call carries it.
    rig_core::providers::internal::resolve_empty_tool_result_names(&mut chat_history);
    let mut contents = Vec::new();

    for msg in chat_history {
        contents.push(rig_message_to_grpc_content(msg)?);
    }

    let mut system_parts = Vec::new();
    for content in history_system {
        if !content.is_empty() {
            system_parts.push(text_part(content));
        }
    }
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(proto::Content {
            parts: system_parts,
            role: "model".to_string(),
        })
    };

    let generation_config = if temperature.is_some() || max_tokens.is_some() {
        Some(proto::GenerationConfig {
            temperature: temperature.map(|t| t as f32),
            max_output_tokens: max_tokens.map(|t| t as i32),
            ..Default::default()
        })
    } else {
        None
    };

    let tools = if !tools.is_empty() {
        let function_declarations = tools
            .into_iter()
            .map(|tool| {
                Ok(proto::FunctionDeclaration {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool_parameters_to_proto_schema(&tool.parameters)?,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, ProviderError>>()?;

        vec![proto::Tool {
            function_declarations,
            code_execution: None,
        }]
    } else {
        vec![]
    };

    Ok(GenerateContentRequest {
        model: format!("models/{model}"),
        contents,
        tools,
        safety_settings: vec![],
        generation_config,
        tool_config: None,
        system_instruction,
        cached_content: String::new(),
    })
}

fn rig_message_to_grpc_content(msg: message::Message) -> Result<proto::Content, ProviderError> {
    match msg {
        message::Message::System { .. } => Err(ProviderError::Request(
            "System messages must be sent via Gemini gRPC system_instruction".into(),
        )),
        message::Message::User { content } => {
            let parts = content
                .into_iter()
                .map(rig_user_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "user".to_string(),
            })
        }
        message::Message::Assistant { content, .. } => {
            let parts = content
                .into_iter()
                .map(rig_assistant_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "model".to_string(),
            })
        }
    }
}

use rig_core::providers::gemini::completion::split_system_messages_from_history;

fn rig_user_content_to_grpc_part(
    content: message::UserContent,
) -> Result<proto::Part, ProviderError> {
    match content {
        message::UserContent::Text(message::Text { text, .. }) => Ok(text_part(text)),
        message::UserContent::ToolResult(result) => {
            let mut values = result
                .content
                .into_iter()
                .map(|content| match content {
                    message::ToolResultContent::Text(t) => Ok(serde_json::Value::String(t.text)),
                    message::ToolResultContent::Json { value } => Ok(value),
                    message::ToolResultContent::Image(_) => Err(ProviderError::Request(
                        "Gemini gRPC does not support images in tool results".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let result_value = if values.len() == 1 {
                values.remove(0)
            } else {
                serde_json::Value::Array(values)
            };

            let response_struct =
                json_to_prost_struct(serde_json::json!({ "result": result_value }))?;

            // Replay the function name and only provider-issued IDs; local
            // correlation handles must not reach the wire.
            Ok(data_part(proto::part::Data::FunctionResponse(
                proto::FunctionResponse {
                    name: result.name,
                    response: Some(response_struct),
                    id: result
                        .provider
                        .map(|provider| provider.call_id)
                        .unwrap_or_default(),
                },
            )))
        }
        message::UserContent::Image(img) => {
            let Some(media_type) = img.media_type else {
                return Err(ProviderError::Request(
                    "Media type for image is required for Gemini".into(),
                ));
            };

            match media_type {
                message::ImageMediaType::JPEG
                | message::ImageMediaType::PNG
                | message::ImageMediaType::WEBP
                | message::ImageMediaType::HEIC
                | message::ImageMediaType::HEIF => {}
                _ => {
                    return Err(ProviderError::Request(
                        format!("Unsupported image media type {media_type:?}").into(),
                    ));
                }
            }

            let mime_type = media_type.to_mime_type().to_string();

            let data = match img.data {
                message::DocumentSourceKind::Url(file_uri) => {
                    return Ok(data_part(proto::part::Data::FileData(proto::FileData {
                        mime_type,
                        file_uri,
                    })));
                }
                message::DocumentSourceKind::Raw(bytes) => bytes,
                message::DocumentSourceKind::Base64(data)
                | message::DocumentSourceKind::String(data) => decode_base64_bytes(&data)?,
                message::DocumentSourceKind::Unknown => {
                    return Err(ProviderError::Request("Image content has no body".into()));
                }
                _ => {
                    return Err(ProviderError::Request(
                        "Unsupported document source kind".into(),
                    ));
                }
            };

            Ok(data_part(proto::part::Data::InlineData(proto::Blob {
                mime_type,
                data,
            })))
        }
        _ => Err(ProviderError::Request(
            "Unsupported user content type".into(),
        )),
    }
}

fn rig_assistant_content_to_grpc_part(
    content: message::AssistantContent,
) -> Result<proto::Part, ProviderError> {
    match content {
        message::AssistantContent::Text(text) => Ok(proto::Part {
            thought_signature: decode_optional_base64(
                text_thought_signature(&text).map(str::to_owned),
            )?,
            ..text_part(text.text)
        }),
        message::AssistantContent::ToolCall(tool_call) => {
            let args = json_to_prost_struct(tool_call.function.arguments)?;

            Ok(proto::Part {
                thought_signature: decode_optional_base64(tool_call.signature)?,
                ..data_part(proto::part::Data::FunctionCall(proto::FunctionCall {
                    name: tool_call.function.name,
                    args: Some(args),
                    // Only a provider-issued id may travel back on the
                    // wire; minted correlation handles stay internal.
                    id: tool_call
                        .provider
                        .map(|provider| provider.call_id)
                        .unwrap_or_default(),
                }))
            })
        }
        message::AssistantContent::Reasoning(reasoning) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(reasoning.display_text())),
            thought: true,
            thought_signature: decode_optional_base64(
                reasoning
                    .first_signature()
                    .map(std::string::ToString::to_string),
            )?,
            part_metadata: None,
        }),
        _ => Err(ProviderError::Request(
            "Unsupported assistant content type".into(),
        )),
    }
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse {
    type Error = ProviderError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        // The provider's own document, captured before the response is
        // consumed into normalized content.
        let raw = serde_json::to_value(&response)?;
        let candidate = response
            .candidates
            .first()
            .ok_or_else(|| ProviderError::Response("No response candidates in response".into()))?;

        // Same helper (and therefore the same message) as the streaming path,
        // so a tool-protocol abort reads identically on both surfaces.
        if let Some(err) = tool_protocol_finish_reason_error(
            candidate.finish_reason,
            candidate.finish_message.as_deref(),
        ) {
            return Err(err);
        }

        let content_ref = candidate.content.as_ref().ok_or_else(|| {
            ProviderError::Response(format!(
                "Gemini candidate missing content (finish_reason={})",
                candidate.finish_reason
            ))
        })?;

        let mut assistant_contents = Vec::new();

        let mut tool_index = 0u64;
        for part in &content_ref.parts {
            let assistant_content = match &part.data {
                Some(proto::part::Data::Text(text)) => {
                    if part.thought {
                        completion::AssistantContent::Reasoning(
                            Reasoning::new_with_signature(
                                text,
                                encode_optional_base64(&part.thought_signature),
                            )
                            .with_provider(REASONING_ISSUER),
                        )
                    } else {
                        // A signature on answer text returns on that text part.
                        completion::AssistantContent::Text(message::Text {
                            text: text.clone(),
                            additional_params: encode_optional_base64(&part.thought_signature)
                                .and_then(|signature| {
                                    text_signature_extras(GEMINI_TEXT_EXTRAS_KEY, signature)
                                }),
                        })
                    }
                }
                Some(proto::part::Data::InlineData(inline_data)) => {
                    let mime_type = message::MediaType::from_mime_type(&inline_data.mime_type);
                    match mime_type {
                        Some(message::MediaType::Image(media_type)) => {
                            let b64 =
                                base64::engine::general_purpose::STANDARD.encode(&inline_data.data);
                            completion::AssistantContent::image_base64(
                                b64,
                                Some(media_type),
                                Some(message::ImageDetail::default()),
                            )
                        }
                        _ => {
                            return Err(ProviderError::Response(format!(
                                "Unsupported media type {mime_type:?}"
                            )));
                        }
                    }
                }
                Some(proto::part::Data::FunctionCall(function_call)) => {
                    let args = function_call.args.as_ref().map_or(
                        serde_json::Value::Object(serde_json::Map::new()),
                        prost_struct_to_json,
                    );

                    // Index-based handles distinguish repeated calls to one tool.
                    let index = tool_index;
                    tool_index += 1;
                    let tool_call = message::ToolCall::from_wire_indexed(
                        function_call.id.clone(),
                        index,
                        message::ToolFunction::new(function_call.name.clone(), args),
                    )
                    .with_signature(encode_optional_base64(&part.thought_signature));

                    completion::AssistantContent::ToolCall(tool_call)
                }
                _ => {
                    return Err(ProviderError::Response(
                        "Response did not contain a message or tool call".into(),
                    ));
                }
            };

            assistant_contents.push(assistant_content);
        }

        rig_core::message::normalize_missing_tool_call_ids(&mut assistant_contents);
        let choice = rig_core::message::require_non_empty_response(assistant_contents)?;

        let usage = map_usage(response.usage_metadata.as_ref());

        let finish_reason = response
            .candidates
            .first()
            .and_then(|candidate| map_finish_reason(candidate.finish_reason));
        let model = Some(response.model_version.clone()).filter(|model| !model.is_empty());
        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME, raw)
                .with_optional_finish_reason(finish_reason)
                .with_optional_response_id(
                    Some(response.response_id.clone()).filter(|id| !id.is_empty()),
                )
                .with_optional_model(model),
        )
    }
}

impl ProviderResponseExt for GenerateContentResponse {
    type Usage = proto::UsageMetadata;

    fn response_id(&self) -> Option<&str> {
        if self.response_id.is_empty() {
            None
        } else {
            Some(self.response_id.as_str())
        }
    }

    fn response_model_name(&self) -> Option<&str> {
        if self.model_version.is_empty() {
            None
        } else {
            Some(self.model_version.as_str())
        }
    }

    fn text_response(&self) -> Option<String> {
        self.candidates.first().and_then(|c| {
            c.content.as_ref().and_then(|content| {
                let text: Vec<String> = content
                    .parts
                    .iter()
                    // Reasoning text is not the user-facing answer.
                    .filter(|part| !part.thought)
                    .filter_map(|part| {
                        if let Some(proto::part::Data::Text(text)) = &part.data {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if text.is_empty() {
                    None
                } else {
                    Some(text.join("\n"))
                }
            })
        })
    }

    fn usage(&self) -> Option<Self::Usage> {
        self.usage_metadata
    }
}

fn decode_base64_bytes(input: &str) -> Result<Vec<u8>, ProviderError> {
    let data = input.trim();

    // Allow `data:<mime>;base64,<data>` inputs.
    let data = if let Some(rest) = data.strip_prefix("data:") {
        rest.split_once(',').map_or(data, |(_, b64)| b64)
    } else {
        data
    };

    let mut last_err: Option<String> = None;

    for engine in [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ] {
        match engine.decode(data) {
            Ok(bytes) => return Ok(bytes),
            Err(err) => last_err = Some(err.to_string()),
        }
    }

    let err = last_err.unwrap_or_else(|| "unknown base64 decode error".to_string());
    Err(ProviderError::Request(
        format!("Invalid base64 data: {err}").into(),
    ))
}

fn decode_optional_base64(sig: Option<String>) -> Result<Vec<u8>, ProviderError> {
    let Some(sig) = sig else {
        return Ok(Vec::new());
    };
    decode_base64_bytes(&sig)
}

/// Map Gemini's `UsageMetadata` onto rig's normalized `Usage`.
///
/// Tool-use, reasoning, and cache-write token counts remain `None`.
pub(crate) fn map_usage(usage: Option<&proto::UsageMetadata>) -> completion::Usage {
    usage
        .map(|usage| completion::Usage {
            input_tokens: Some(usage.prompt_token_count as u64),
            output_tokens: Some(usage.candidates_token_count as u64),
            total_tokens: Some(usage.total_token_count as u64),
            cached_input_tokens: Some(usage.cached_content_token_count as u64),
            cache_creation_input_tokens: None,
            tool_use_prompt_tokens: None,
            reasoning_tokens: None,
        })
        .unwrap_or_default()
}

pub(crate) fn encode_optional_base64(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn json_to_prost_struct(value: serde_json::Value) -> Result<proto::Struct, ProviderError> {
    match value {
        serde_json::Value::Object(map) => Ok(proto::Struct {
            fields: map
                .into_iter()
                .map(|(k, v)| (k, json_to_prost_value(v)))
                .collect(),
        }),
        _ => Err(ProviderError::Request(
            "Expected a JSON object for google.protobuf.Struct".into(),
        )),
    }
}

fn json_to_prost_value(value: serde_json::Value) -> proto::Value {
    match value {
        serde_json::Value::Null => proto::Value {
            kind: Some(proto::value::Kind::NullValue(
                proto::NullValue::NullValue as i32,
            )),
        },
        serde_json::Value::Bool(b) => proto::Value {
            kind: Some(proto::value::Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => proto::Value {
            kind: Some(proto::value::Kind::NumberValue(
                n.as_f64().unwrap_or_default(),
            )),
        },
        serde_json::Value::String(s) => proto::Value {
            kind: Some(proto::value::Kind::StringValue(s)),
        },
        serde_json::Value::Array(items) => proto::Value {
            kind: Some(proto::value::Kind::ListValue(proto::ListValue {
                values: items.into_iter().map(json_to_prost_value).collect(),
            })),
        },
        serde_json::Value::Object(map) => proto::Value {
            kind: Some(proto::value::Kind::StructValue(proto::Struct {
                fields: map
                    .into_iter()
                    .map(|(k, v)| (k, json_to_prost_value(v)))
                    .collect(),
            })),
        },
    }
}

pub(crate) fn prost_struct_to_json(st: &proto::Struct) -> serde_json::Value {
    let mut out = serde_json::Map::with_capacity(st.fields.len());
    for (k, v) in &st.fields {
        out.insert(k.clone(), prost_value_to_json(v));
    }
    serde_json::Value::Object(out)
}

fn prost_value_to_json(v: &proto::Value) -> serde_json::Value {
    match &v.kind {
        None | Some(proto::value::Kind::NullValue(_)) => serde_json::Value::Null,
        Some(proto::value::Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(proto::value::Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        Some(proto::value::Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}

/// Converts tool parameters to protobuf schema through the shared Gemini conversion.
/// Empty object schemas map to `None`.
fn tool_parameters_to_proto_schema(
    value: &serde_json::Value,
) -> Result<Option<proto::Schema>, ProviderError> {
    tool_parameters_to_schema(value.clone())
        .map(|schema| schema.map(gemini_schema_to_proto_schema))
        .map_err(ProviderError::from)
}

fn gemini_schema_to_proto_schema(schema: GeminiSchema) -> proto::Schema {
    proto::Schema {
        r#type: json_type_to_proto_type(&schema.r#type) as i32,
        format: schema.format.unwrap_or_default(),
        description: schema.description.unwrap_or_default(),
        nullable: schema.nullable.unwrap_or(false),
        r#enum: schema.r#enum.unwrap_or_default(),
        items: schema
            .items
            .map(|items| Box::new(gemini_schema_to_proto_schema(*items))),
        properties: schema
            .properties
            .unwrap_or_default()
            .into_iter()
            .map(|(name, schema)| (name, gemini_schema_to_proto_schema(schema)))
            .collect(),
        required: schema.required.unwrap_or_default(),
    }
}

fn json_type_to_proto_type(t: &str) -> proto::Type {
    match t {
        "string" => proto::Type::String,
        "number" => proto::Type::Number,
        "integer" => proto::Type::Integer,
        "boolean" => proto::Type::Boolean,
        "array" => proto::Type::Array,
        "object" => proto::Type::Object,
        "null" => proto::Type::Null,
        _ => proto::Type::Unspecified,
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]
pub(crate) mod tests;

#[cfg(test)]
mod schema_tests;

#[cfg(test)]
mod transport_tests;

#[cfg(test)]
mod request_tests;

#[cfg(test)]
mod reasoning_scope_tests;

#[cfg(test)]
mod reply_tests;
