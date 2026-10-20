//! Chat Completions: the request built as JSON, and one lenient decoder for
//! a whole reply and a stream of chunks.
//!
//! ```
//! use rig_core::providers::openai::wire::OpenAIConfig;
//! let wire = OpenAIConfig::new("key").chat("gpt-5.2");
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::completion::options::{BaseInput, RawAt, Rewrite, request_params};
use crate::completion::{CompletionRequest, FinishReason, ProviderCapabilities, Replay};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, AssistantMessage, DocumentData, DocumentMediaType,
    DocumentSourceKind as Source, Message, MimeType, ToolCall, ToolResult, ToolResultContent,
    UserContent,
};
use crate::observe::ObservedError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::openai_chat_completions_compatible::{
    native_finish_reason, provider_error_envelope,
};
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::{
    AdapterUsage, AdapterVerdict, Capabilities, Decoder, Descriptor, Encoded, Flow, Framing, Mode,
    ObservationSink, Out, UsageObservation, Wire, WireEvent, WireFrame,
};

use super::dto::merge_fields;
use super::{BodyRewrite, OpenAIConfig, OutputCap, Quirks};

/// The chat-completions wire: a provider configuration, a model, and the
/// per-turn options the endpoint takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    /// Which provider, and how to reach it.
    pub provider: OpenAIConfig,
    /// The model this wire addresses.
    pub model: String,
    /// Whether tool schemas are sanitized for OpenAI's strict mode:
    /// `additionalProperties: false` on every object, every property
    /// required, and `strict: true` on each function definition.
    pub strict_tools: bool,
    /// Whether tool-result messages serialize their content as arrays.
    pub tool_result_array_content: bool,
    /// Ask OpenRouter to cache the system prompt with an ephemeral marker.
    #[serde(default)]
    pub prompt_caching: bool,
}

/// The error for media in a form Chat Completions cannot carry, which the
/// adapter replaces before a request is encoded
/// ([`ReplayTarget::encodes`](crate::completion::ReplayTarget::encodes)).
fn unsendable(what: &str) -> EncodeError {
    EncodeError::request(format!("Chat Completions cannot carry {what} in this form"))
}

/// `data` as a URL a content part names: a URL, or typed base64 data as a
/// data URI.
fn media_url(data: &Source, mime: Option<&str>) -> Option<String> {
    match (data, mime) {
        (Source::Url(url), _) => Some(url.clone()),
        (Source::Base64(data), Some(mime)) => Some(format!("data:{mime};base64,{data}")),
        _ => None,
    }
}

/// The `image_url` part an image is sent as; a user image always names a
/// detail level, `auto` by default.
fn image_url(image: &crate::message::Image, user: bool) -> Result<Value, EncodeError> {
    let mime = image.media_type.as_ref().map(MimeType::to_mime_type);
    let url = media_url(&image.data, mime).ok_or_else(|| unsendable("an image"))?;
    let mut part = Map::from_iter([("url".to_owned(), Value::String(url))]);
    let detail = match &image.detail {
        Some(detail) => Some(detail.clone()),
        None => user.then(Default::default),
    };
    if let Some(detail) = detail {
        part.insert("detail".to_owned(), serde_json::to_value(detail)?);
    }
    Ok(json!({"type": "image_url", "image_url": part}))
}

/// Check the original PDF source before adaptation or direct encoding.
/// No current dialect quirk declares URL-document support.
fn check_pdf_document(document: &crate::message::Document) -> Result<(), EncodeError> {
    if document.media_type != Some(DocumentMediaType::PDF) {
        return Ok(());
    }
    match &document.data {
        DocumentData::File(Source::Url(_)) => Err(EncodeError::request(
            crate::message::MessageError::ConversionError(
                "OpenAI chat completions does not accept URL files; use the Responses API or pass base64-encoded bytes".to_owned(),
            ),
        )),
        DocumentData::File(Source::Raw(_)) => Err(EncodeError::request(
            crate::message::MessageError::ConversionError(
                "Raw files not supported, encode as base64 first".to_owned(),
            ),
        )),
        DocumentData::Text(_)
        | DocumentData::File(Source::Base64(_) | Source::FileId(_) | Source::Unknown) => Ok(()),
    }
}

/// One user content part as Chat Completions carries it.
fn user_part(part: &UserContent) -> Result<Value, EncodeError> {
    let file = |file: Value| json!({"type": "file", "file": file});
    Ok(match part {
        UserContent::Text(text) => json!({"type": "text", "text": text.text}),
        UserContent::Image(image) => image_url(image, true)?,
        UserContent::Document(document) => {
            check_pdf_document(document)?;
            let pdf = document.media_type == Some(DocumentMediaType::PDF);
            match &document.data {
                DocumentData::Text(text) => json!({"type": "text", "text": text}),
                DocumentData::File(Source::FileId(id)) => file(json!({ "file_id": id })),
                DocumentData::File(Source::Base64(data)) if pdf => file(json!({
                    "file_data": format!("data:application/pdf;base64,{data}"),
                    "filename": "document.pdf",
                })),
                // The original PDF check above refuses the legacy URL projection.
                DocumentData::File(Source::Url(url)) if pdf => {
                    file(json!({"file_data": url, "filename": "document.pdf"}))
                }
                DocumentData::File(_) => return Err(unsendable("a document")),
            }
        }
        UserContent::Audio(audio) => match &audio.data {
            Source::Base64(data) => json!({"type": "input_audio", "input_audio": {
                "data": data,
                "format": audio.media_type.clone().unwrap_or(crate::message::AudioMediaType::MP3),
            }}),
            _ => return Err(unsendable("audio")),
        },
        UserContent::Video(video) => {
            let mime = video.media_type.as_ref().map(MimeType::to_mime_type);
            let url = media_url(&video.data, mime).ok_or_else(|| unsendable("a video"))?;
            json!({"type": "video_url", "video_url": {"url": url}})
        }
        UserContent::ToolResult(_) => return Err(unsendable("a tool result as a content part")),
    })
}

/// A result as the `tool` message that answers its call: its text joined
/// into one string, or its parts as an array when the wire asks for one or
/// it holds an image, which has no string form.
fn tool_message(
    result: &ToolResult,
    wire_id: &str,
    array: bool,
    supports_image_tool_results: bool,
) -> Result<Value, EncodeError> {
    if !supports_image_tool_results
        && result
            .content
            .iter()
            .any(|part| matches!(part, ToolResultContent::Image(_)))
    {
        return Err(EncodeError::request(concat!(
            "this provider does not accept an image in a tool result. ",
            "Official OpenAI refuses it on Chat Completions (and the GPT-5 ",
            "family accepts the request while ignoring the image); use the ",
            "Responses API, which carries images in `function_call_output`, ",
            "or a server that sets `SUPPORTS_IMAGE_TOOL_RESULTS` ",
            "(llama.cpp does)",
        )));
    }
    let parts = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(json!({"type": "text", "text": text.text})),
            ToolResultContent::Json { value } => {
                Ok(json!({"type": "text", "text": value.to_string()}))
            }
            ToolResultContent::Image(image) => image_url(image, false),
        })
        .collect::<Result<Vec<_>, EncodeError>>()?;
    let content = if array
        || parts
            .iter()
            .any(|part| part.str("type") == Some("image_url"))
    {
        Value::Array(parts)
    } else {
        let texts: Vec<&str> = parts.iter().filter_map(|part| part.str("text")).collect();
        Value::String(texts.join("\n"))
    };
    Ok(json!({"role": "tool", "tool_call_id": wire_id, "content": content}))
}

/// Push the user parts gathered so far as one message: a lone text part as
/// a plain string.
fn user_message(messages: &mut Vec<Value>, parts: &mut Vec<Value>) {
    let content = match parts.as_slice() {
        [] => return,
        [part] if part.str("type") == Some("text") => part.get("text").cloned().unwrap_or_default(),
        _ => Value::Array(std::mem::take(parts)),
    };
    parts.clear();
    messages.push(json!({"role": "user", "content": content}));
}

impl Chat {
    pub(crate) fn encode_with_headers(
        &self,
        request: CompletionRequest,
        mode: Mode,
        headers: impl FnOnce(
            &OpenAIConfig,
            &CompletionRequest,
            http::request::Builder,
        ) -> Result<http::request::Builder, EncodeError>,
    ) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks;
        // Azure's deployment URL remains pinned to the handle, not a request override.
        let uri = self.provider.uri(
            quirks.completion_path,
            self.provider.deployment(&self.model),
        );
        let builder = headers(
            &self.provider,
            &request,
            http::Request::post(uri).header("Content-Type", "application/json"),
        )?;
        crate::completion::ReplayTarget::check_provider_items(self, &request.chat_history)
            .map_err(EncodeError::request)?;
        crate::completion::ReplayTarget::check_nested_content(self, &request.chat_history)
            .map_err(EncodeError::request)?;
        if !quirks.accepts_file_ids {
            refuse_file_ids(&request)?;
        }
        let mut rewrites = Vec::new();
        if quirks.output_cap == OutputCap::OpenAiReasoningFamilies {
            rewrites.push(Rewrite::OutputCapRename);
        }
        if mode == Mode::Streaming {
            // A caller's own stream options, `include_usage` among them, stand.
            if quirks.stream_include_usage {
                rewrites.push(Rewrite::StreamUsage);
            }
            rewrites.push(Rewrite::Stream(true));
        }
        rewrites.push(Rewrite::ChatDialect {
            kind: quirks.rewrite,
            reasoning_field: self.listed_reasoning_field(&self.request_model(&request)),
        });
        if quirks.rewrite == BodyRewrite::OpenRouter && self.prompt_caching {
            rewrites.push(Rewrite::OpenRouterPromptCaching);
        }
        let raw_at = match quirks.rewrite {
            BodyRewrite::Mira => RawAt::Ignored(
                "Additional parameters are not supported by Mira and will be ignored",
            ),
            _ => RawAt::Top,
        };
        let body = request_params(
            self,
            &request,
            |input| self.base(&request, input),
            raw_at,
            &rewrites,
        )?;
        let body = crate::providers::openai::options::check_body(
            self,
            &request,
            body,
            crate::providers::openai::options::Endpoint::ChatCompletions,
        )?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI Chat Completions request",
            &body,
        );
        let request = builder.body(body.into_body())?;
        let framing = match mode {
            Mode::Streaming => Framing::Sse,
            Mode::Unary => Framing::Whole,
        };
        Ok(Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header)
            .with_captured_response_headers(self.provider.dialect.response_header_prefix)
            .with_projection(ChatDecoder::project)
            .with_route(Some(quirks.completion_path)))
    }

    /// The field every assistant message carries its reasoning under,
    /// empty when the turn has none (pi's
    /// `requiresReasoningContentOnAssistantMessages`): the dialect's own,
    /// `reasoning_content` for DeepSeek at any base URL, and the field a
    /// catalog entry names for the model on this dialect, on Moonshot's own
    /// API under its last path segment (Kimi K3 behind any gateway), or on
    /// OpenRouter under its full id (`moonshotai/kimi-k2.6` behind any
    /// gateway). Each lookup is asked for the field, not just the entry, so a
    /// gateway's own row without one does not hide the vendor's.
    fn reasoning_field(&self, model: &str) -> Option<&'static str> {
        let deepseek = self
            .provider
            .base_url
            .to_ascii_lowercase()
            .contains("deepseek.com");
        self.provider
            .dialect
            .quirks
            .reasoning_field
            .or(deepseek.then_some("reasoning_content"))
            .or_else(|| self.listed_reasoning_field(model))
    }

    /// The reasoning field a catalog entry names for `model`: the entry on
    /// this dialect, Moonshot's under the id's last path segment, or
    /// OpenRouter's under the full id. A dialect's history rewrite keeps
    /// this field on assistant messages, as upstream sends it
    /// (mec-rig-bodies-answers-1 Q3).
    fn listed_reasoning_field(&self, model: &str) -> Option<&'static str> {
        use crate::providers::openai::wire::{MOONSHOT, OPENROUTER};
        let name = model.rsplit('/').next().unwrap_or(model);
        let field = |vendor: &str, id: &str| {
            crate::catalog::lookup(vendor, id)
                .and_then(|spec| spec.compat.reasoning_field.as_deref())
        };
        field(self.provider.dialect.name, model)
            .or_else(|| field(MOONSHOT.name, name))
            .or_else(|| field(OPENROUTER.name, model))
    }

    /// The model id the body names: the request's model or the wire's,
    /// qualified for a Hugging Face router sub-provider that demands it.
    fn request_model(&self, request: &CompletionRequest) -> String {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        if self.provider.dialect.quirks.rewrite == BodyRewrite::HuggingFaceRouter {
            // Some sub-providers (Fireworks) address models by a qualified id.
            return self.provider.route().model_identifier(&model);
        }
        model
    }

    /// The wire for `model` on `provider`, with every option off.
    pub fn new(provider: OpenAIConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
            prompt_caching: false,
        }
    }

    /// Sanitize tool schemas for OpenAI's strict mode, so the provider can
    /// guarantee a tool call matches its schema exactly.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Serialize tool-result content as arrays.
    pub fn with_tool_result_array_content(mut self) -> Self {
        self.tool_result_array_content = true;
        self
    }

    /// Ask OpenRouter to cache the system prompt.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// The wire's own encoding of `request`: messages, tools and the typed
    /// fields. The mapped options and `additional_params` merge over it.
    fn base(
        &self,
        request: &CompletionRequest,
        input: &mut BaseInput<'_>,
    ) -> Result<Map<String, Value>, EncodeError> {
        let quirks = &self.provider.dialect.quirks;
        let model = self.request_model(request);
        let passthrough = input.raw_tools()?;
        let groq_native = quirks.rewrite == BodyRewrite::GroqCompoundTools;
        // A call to a custom tool the request declares is a custom call (pi).
        let custom: Vec<String> = passthrough
            .iter()
            .filter(|tool| tool.str("type") == Some("custom"))
            .filter_map(|tool| tool.at("/custom/name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        let messages = self.messages(&request.chat_history, &model, &custom)?;

        let mut tools: Vec<Value> = Vec::new();
        let mut tool_choice = None;
        if quirks.supports_tools {
            for tool in &request.tools {
                let mut parameters = tool.parameters.clone();
                let mut function = json!({"name": tool.name, "description": tool.description});
                if self.strict_tools {
                    crate::providers::openai::sanitize_schema(&mut parameters);
                }
                if let Some(function) = function.as_object_mut() {
                    function.insert("parameters".to_owned(), parameters);
                    if self.strict_tools {
                        function.insert("strict".to_owned(), Value::Bool(true));
                    }
                }
                tools.push(json!({"type": "function", "function": function}));
            }
            // Passthrough tools, function or native (Groq's `browser_search`),
            // join the typed ones in one array.
            if !groq_native {
                tools.extend(passthrough);
            }
            tool_choice = request
                .tool_choice
                .clone()
                .map(tool_choice_value)
                .transpose()?
                .filter(|_| !tools.is_empty());
        } else {
            if !request.tools.is_empty() {
                tracing::warn!("Tool use is not supported by this provider; tools will be ignored");
            }
            if request.tool_choice.is_some() {
                tracing::warn!("Tool choice is not supported by this provider and will be ignored");
            }
            // The caller's own tools still reach a dialect that takes none,
            // as `additional_params` always did.
            if !groq_native {
                tools.extend(passthrough);
            }
        }

        if request.output_schema.is_some() && !quirks.supports_response_format {
            tracing::warn!(
                "Structured outputs are not supported by this provider; ignoring output_schema"
            );
        }
        // Defer schemas until a tool result exists unless the dialect supports both at once.
        let answered = messages
            .iter()
            .any(|message| message.str("role") == Some("tool"));
        let response_format = match request.output_schema.clone() {
            Some(schema)
                if quirks.supports_response_format
                    && (quirks.response_format_with_tools || tools.is_empty() || answered) =>
            {
                let (name, schema) = crate::providers::openai::structured_output_schema(schema);
                Some(json!({"type": "json_schema",
                    "json_schema": {"name": name, "strict": true, "schema": schema}}))
            }
            _ => None,
        };

        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("tool_choice", tool_choice),
            ("temperature", request.temperature.map(Value::from)),
            ("max_tokens", request.max_tokens.map(Value::from)),
            ("response_format", response_format),
        ];
        Ok(fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect())
    }

    /// The history as Chat messages, each call and result spelled by one
    /// [`WireIds`].
    fn messages(
        &self,
        history: &[Message],
        model: &str,
        custom: &[String],
    ) -> Result<Vec<Value>, EncodeError> {
        for message in history {
            if let Message::Assistant(turn) = message {
                for block in &turn.content {
                    if let AssistantContent::ToolCall(call) = block {
                        call.for_json_only_wire("OpenAI-compatible Chat Completions")?;
                    }
                }
            }
        }
        let ids = WireIds::for_target(history, self, model).map_err(EncodeError::request)?;
        let mut messages = Vec::new();
        for (message_index, message) in history.iter().enumerate() {
            match message {
                Message::System { content } => messages.push(json!({"role": "system",
                    "content": [{"type": "text", "text": content}]})),
                Message::User { content } => {
                    let mut parts = Vec::new();
                    for (content_index, part) in content.iter().enumerate() {
                        if let UserContent::ToolResult(result) = part {
                            user_message(&mut messages, &mut parts);
                            let array = self.tool_result_array_content;
                            let wire_id = ids
                                .spell(message_index, content_index)
                                .map_err(EncodeError::request)?;
                            messages.push(tool_message(
                                result,
                                &wire_id,
                                array,
                                self.provider.dialect.quirks.supports_image_tool_results,
                            )?);
                        } else {
                            parts.push(user_part(part)?);
                        }
                    }
                    user_message(&mut messages, &mut parts);
                }
                Message::Assistant(turn) => {
                    messages.extend(self.assistant(turn, message_index, &ids, custom, model)?);
                }
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "OpenAI Chat Completions request has no provider-compatible messages after \
                 conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn as pi's `openai-completions` rebuilds it: text
    /// joined into a `content` string, or a part array when a block is a
    /// content part (Mistral's thinking); reasoning under the field it
    /// arrived in, or the dialect's; an item's other fields beside it; and
    /// each call with its canonical name and arguments. A message with
    /// neither content nor calls is `None`, as pi skips it.
    fn assistant(
        &self,
        turn: &AssistantMessage,
        message_index: usize,
        ids: &WireIds,
        custom: &[String],
        model: &str,
    ) -> Result<Option<Value>, EncodeError> {
        let reasoning_field = self.reasoning_field(model);
        let (mut text, mut parts, mut has_parts) = (String::new(), Vec::new(), false);
        let mut reasoning: Vec<(String, String)> = Vec::new();
        let (mut fields, mut calls) = (Map::new(), Vec::new());
        for (content_index, block) in turn.content.iter().enumerate() {
            let replay = block.replay(self, ids.of(message_index, content_index));
            match block {
                AssistantContent::Text(block) => {
                    text.push_str(&block.text);
                    parts.push(json!({"type": "text", "text": block.text}));
                    // A text item is message fields: an answer's audio.
                    if let Replay::Item(item) = replay
                        && let Value::Object(item) = item.into_owned()
                    {
                        fields.extend(item);
                    }
                }
                AssistantContent::Reasoning(block) => {
                    let field = match replay {
                        Replay::Item(item) if item.get("type").is_some() => {
                            has_parts = true;
                            parts.push(item.into_owned());
                            None
                        }
                        Replay::Item(item) => {
                            let mut field = None;
                            for (key, value) in item.as_object().into_iter().flatten() {
                                if value.is_string() {
                                    field.get_or_insert_with(|| key.clone());
                                } else {
                                    let detail = Map::from_iter([(key.clone(), value.clone())]);
                                    merge_fields(&mut fields, &detail);
                                }
                            }
                            field
                        }
                        Replay::Identity(identity)
                            if identity.get("type").and_then(Value::as_str) == Some("thinking") =>
                        {
                            has_parts = true;
                            parts.push(json!({"type": "thinking",
                                "thinking": [{"type": "text", "text": block.text}]}));
                            None
                        }
                        Replay::Identity(identity) => identity.keys().next().cloned(),
                        Replay::Rebuild => reasoning_field.map(str::to_owned),
                    };
                    if let Some(field) = field {
                        match reasoning.iter_mut().find(|(name, _)| *name == field) {
                            Some((_, joined)) => {
                                joined.push('\n');
                                joined.push_str(&block.text);
                            }
                            None => reasoning.push((field, block.text.clone())),
                        }
                    }
                }
                AssistantContent::ToolCall(call) => {
                    let wire_id = ids
                        .spell(message_index, content_index)
                        .map_err(EncodeError::request)?;
                    calls.push(call_item(call, replay, &wire_id, custom));
                }
                AssistantContent::CustomToolCall(call) => {
                    return Err(call
                        .refused_by_json_only_wire("OpenAI-compatible Chat Completions")
                        .into());
                }
                AssistantContent::ProviderItem(item) => {
                    return Err(crate::message::UnreplayableProviderItem::new(
                        "OpenAI Chat Completions",
                        item,
                    )
                    .into());
                }
                AssistantContent::CarriedText(block) => {
                    let replay = match block.foreign_text() {
                        Some(known) => crate::message::CarriedTextReplay::KnownText(known),
                        None => block.replay_item().map_err(EncodeError::request)?,
                    };
                    match replay {
                        crate::message::CarriedTextReplay::KnownText(known) => {
                            text.push_str(known);
                            parts.push(json!({"type":"text", "text":known}));
                            // The fork's official Chat wire keeps known text parts,
                            // including an empty inspection marker, as an array.
                            has_parts |= self.provider.dialect.name == super::dialects::OPENAI.name;
                        }
                        crate::message::CarriedTextReplay::ProviderPart(_) => {
                            return Err(crate::message::UnrepresentableOpaqueContent::new(
                                "OpenAI Chat Completions",
                            )
                            .into());
                        }
                    }
                }
                AssistantContent::CarriedReasoning(block) => {
                    if block.has_opaque_parts() {
                        return Err(crate::message::UnrepresentableOpaqueContent::new(
                            "OpenAI Chat Completions",
                        )
                        .into());
                    }
                    let issuers = super::replay_issuers(&self.provider.dialect, model);
                    if !issuers.iter().any(|issuer| block.replayable_to(issuer)) {
                        continue;
                    }
                    if self.provider.dialect.quirks.reasoning_details {
                        let start = fields
                            .get(REASONING_DETAILS)
                            .and_then(Value::as_array)
                            .map_or(0, Vec::len);
                        let details = carried_details(block, start)?;
                        merge_details(&mut fields, details);
                    } else {
                        let display = block.display_text();
                        if !display.is_empty() {
                            reasoning.push(("reasoning_content".to_owned(), display));
                        }
                    }
                }
                // An opaque is a content part; one with no `type` is not
                // sent, so its keys never land on the message.
                AssistantContent::Opaque(opaque) if opaque.replay => {
                    if opaque.item.get("type").is_some() {
                        has_parts = true;
                        parts.push(opaque.item.clone());
                    }
                }
                AssistantContent::Opaque(_) | AssistantContent::Image(_) => {}
            }
        }
        // Groq rejects `reasoning_content` on an assistant message (HTTP 400,
        // "property 'reasoning_content' is unsupported") but takes reasoning
        // back under `reasoning`, the field it delivers it in, as upstream's
        // live Groq recordings replay it.
        if self.provider.dialect.quirks.rewrite == BodyRewrite::GroqCompoundTools {
            reasoning.retain(|(field, _)| field.as_str() != "reasoning_content");
        }
        let mut message = Map::from_iter([("role".to_owned(), Value::from("assistant"))]);
        if has_parts {
            message.insert("content".to_owned(), Value::Array(parts));
        } else if !text.is_empty() {
            let content = if self.provider.dialect.quirks.rewrite == BodyRewrite::GroqCompoundTools
            {
                json!([{"type": "text", "text": text}])
            } else {
                Value::String(text)
            };
            message.insert("content".to_owned(), content);
        }
        for (field, text) in reasoning {
            message.insert(field, Value::String(text));
        }
        message.extend(fields);
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(calls));
        }
        let has_content = message
            .get(REASONING_DETAILS)
            .and_then(Value::as_array)
            .is_some_and(|details| !details.is_empty())
            || message.contains_key("audio")
            || message.contains_key("tool_calls")
            || match message.get("content") {
                Some(Value::String(text)) => !text.is_empty(),
                Some(Value::Array(parts)) => !parts.is_empty(),
                _ => false,
            };
        if let Some(field) = reasoning_field
            && self.provider.dialect.quirks.rewrite != BodyRewrite::GroqCompoundTools
        {
            message
                .entry(field)
                .or_insert_with(|| Value::String(String::new()));
        }
        Ok(has_content.then_some(Value::Object(message)))
    }
}

/// The dialect's rewrite of the finished body ([`BodyRewrite`]): a refusal
/// where the provider would answer with an error or silently do something
/// else, Moonshot's steering for `required`, and each dialect's spellings and
/// content shapes. It runs after the merge and reads the merged body.
///
/// `reasoning_field` is the assistant reasoning field the catalog names for
/// the request's model (`Chat::listed_reasoning_field`). The plain-text
/// history rewrites (Perplexity, Mira, Hyperbolic) and Mistral keep it on
/// assistant messages, as upstream sends it for such a model on any gateway;
/// for every other model they remove `reasoning_content` as before. Groq's
/// removal does not depend on the model: Groq rejects the field.
pub(crate) fn rewrite_body(
    kind: BodyRewrite,
    reasoning_field: Option<&str>,
    map: &mut Map<String, Value>,
) -> Result<(), EncodeError> {
    let forced = map
        .get("tool_choice")
        .and_then(|choice| choice.at("/function/name"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    match kind {
        BodyRewrite::LlamaCpp => {
            if let Some(name) = forced {
                return Err(EncodeError::request(format!(
                    "llama.cpp cannot force a specific tool: `llama-server` accepts only \
                     `auto`, `none` or `required` for tool_choice and silently treats \
                     anything else as `auto`, so requesting `{name}` would return whichever \
                     tool the model picked. Use `ToolChoice::Required` to force a call, or \
                     advertise only `{name}` in `tools`."
                )));
            }
        }
        BodyRewrite::Moonshot => {
            if forced.is_some() {
                return Err(EncodeError::request(
                    "Moonshot does not support forcing a specific tool",
                ));
            }
            if map.get("tool_choice").and_then(Value::as_str) == Some("required") {
                tracing::warn!(
                    "Moonshot does not support tool_choice=required; coercing to auto with an \
                     additional steering message"
                );
                map.insert("tool_choice".to_owned(), Value::from("auto"));
                if let Some(Value::Array(messages)) = map.get_mut("messages") {
                    messages.push(json!({"role": "user",
                        "content": "Please select a tool to handle the current issue."}));
                }
            }
        }
        // The gateway takes every message's content as one string.
        // Perplexity spends output budget on a text-part array (a short
        // `max_tokens` comes back empty at `length`, checked live), so
        // text-only arrays go as one string; the gateway takes every
        // content as one string.
        BodyRewrite::Perplexity | BodyRewrite::Mira | BodyRewrite::Hyperbolic => {
            if let Some(Value::Array(messages)) = map.get_mut("messages") {
                match kind {
                    BodyRewrite::Perplexity => sanitize_plain_text_history(
                        messages,
                        Some(("\n", true)),
                        false,
                        true,
                        reasoning_field,
                    ),
                    BodyRewrite::Mira => sanitize_plain_text_history(
                        messages,
                        Some(("\n", false)),
                        true,
                        false,
                        reasoning_field,
                    ),
                    BodyRewrite::Hyperbolic => {
                        sanitize_plain_text_history(messages, None, false, false, reasoning_field)
                    }
                    _ => {}
                }
            }
        }
        BodyRewrite::GroqCompoundTools => {
            for message in messages_mut(map) {
                if message.get("role").and_then(Value::as_str) == Some("assistant") {
                    message.shift_remove("reasoning_content");
                }
            }
        }
        BodyRewrite::DeepSeek => finalize_deepseek(map),
        BodyRewrite::Mistral => finalize_mistral(map, reasoning_field)?,
        BodyRewrite::Ollama => finalize_ollama(map)?,
        // OpenRouter takes assistant `reasoning_content` and an image part's
        // `detail` as the shared encoder writes them, as upstream sends and
        // re-recorded them (mec-rig-bodies-answers-1 Q4). Its routing
        // preferences reach the body through `additional_params` as
        // `{"provider": …}`.
        BodyRewrite::OpenRouter | BodyRewrite::None | BodyRewrite::HuggingFaceRouter => {}
    }
    Ok(())
}

/// `call` as the item the wire sends, from what replay hands the encoder:
/// the current item, an edited one's kind, or nothing. A call is `custom`
/// when its item or kept kind says so, or when it is rebuilt and the request
/// declares a custom tool by its name (pi); its arguments' `input` is then
/// its input. Any other is a function call with its arguments as JSON text.
fn call_item(call: &ToolCall, replay: Replay<'_>, wire_id: &str, custom: &[String]) -> Value {
    let mut item = match replay {
        Replay::Item(item) => match item.into_owned() {
            Value::Object(item) => item,
            _ => Map::new(),
        },
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    let name = call.function.name.as_str();
    let custom = match item.get("type").and_then(Value::as_str) {
        Some(kind) => kind == "custom",
        None => {
            let declared = custom.iter().any(|tool| tool == name);
            let kind = if declared { "custom" } else { "function" };
            item.insert("type".to_owned(), Value::from(kind));
            declared
        }
    };
    item.insert("id".to_owned(), Value::String(wire_id.to_owned()));
    let (slot, key, value) = if custom {
        let input = match call.function.arguments.get("input") {
            Some(Value::String(input)) => input.clone(),
            Some(input) => input.to_string(),
            None => String::new(),
        };
        ("custom", "input", input)
    } else {
        let arguments = call
            .function
            .invalid_arguments
            .clone()
            .unwrap_or_else(|| Value::Object(call.function.arguments.clone()).to_string());
        ("function", "arguments", arguments)
    };
    let slot = item
        .entry(slot)
        .or_insert_with(|| Value::Object(Map::new()));
    if !slot.is_object() {
        *slot = Value::Object(Map::new());
    }
    if let Value::Object(fields) = slot {
        fields.insert("name".to_owned(), Value::from(name));
        if let Some(namespace) = &call.function.namespace {
            fields.insert("namespace".to_owned(), Value::String(namespace.clone()));
        }
        fields.insert(key.to_owned(), Value::String(value));
    }
    Value::Object(item)
}

/// The Chat `tool_choice` for a canonical one.
fn tool_choice_value(choice: crate::message::ToolChoice) -> Result<Value, EncodeError> {
    use crate::message::ToolChoice;
    Ok(match choice {
        ToolChoice::Auto => Value::from("auto"),
        ToolChoice::None => Value::from("none"),
        ToolChoice::Required => Value::from("required"),
        ToolChoice::Specific { function_names } => {
            let [name] = function_names.as_slice() else {
                return Err(EncodeError::request(
                    "Provider only supports forcing exactly one specific tool",
                ));
            };
            json!({"type": "function", "function": {"name": name}})
        }
    })
}

/// Ollama's OpenAI-compatible body: `num_ctx` and `options` are refused,
/// because this API ignores them and only the native route sends them.
fn finalize_ollama(map: &mut Map<String, Value>) -> Result<(), EncodeError> {
    if let Some(key) = ["num_ctx", "options"]
        .into_iter()
        .find(|key| map.contains_key(*key))
    {
        return Err(EncodeError::request(format!(
            "Ollama's OpenAI-compatible API ignores `{key}`; send it through the native \
             route (`Ollama::native_completion`)"
        )));
    }
    Ok(())
}

/// The body's messages, each as its object.
fn messages_mut(map: &mut Map<String, Value>) -> impl Iterator<Item = &mut Map<String, Value>> {
    let messages = map.get_mut("messages").and_then(Value::as_array_mut);
    messages
        .into_iter()
        .flatten()
        .filter_map(Value::as_object_mut)
}

/// DeepSeek retains system content parts and flattens other text content,
/// echoes tool calls with an `index`, and supplies empty content for a
/// tool-call-only assistant turn.
fn finalize_deepseek(map: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(message) = message.as_object_mut() else {
                continue;
            };
            let role = message.get("role").and_then(serde_json::Value::as_str);
            let is_assistant = role == Some("assistant");
            let is_system = role == Some("system");

            if !is_system && let Some(content) = message.get_mut("content") {
                let separator = if is_assistant { "" } else { "\n" };
                // Preserve nontext parts so unsupported attachments are rejected
                // rather than silently omitted from the prompt.
                flatten_text_content_parts(content, separator, true);
            } else if is_assistant {
                message.insert(
                    "content".to_owned(),
                    serde_json::Value::String(String::new()),
                );
            }

            if is_assistant
                && let Some(tool_calls) =
                    message.get_mut("tool_calls").and_then(Value::as_array_mut)
            {
                for tool_call in tool_calls {
                    if let Some(tool_call) = tool_call.as_object_mut() {
                        tool_call
                            .entry("index")
                            .or_insert_with(|| serde_json::json!(0));
                    }
                }
            }
        }
    }

    // DeepSeek rejects forced tool choices unless thinking is explicitly
    // disabled; suppress them to an explicit `null` otherwise.
    let thinking_disabled = map
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|mode| mode.eq_ignore_ascii_case("disabled"));
    if !thinking_disabled
        && let Some(tool_choice) = map.get_mut("tool_choice")
        && (tool_choice.is_object() || tool_choice.as_str() == Some("required"))
    {
        *tool_choice = serde_json::Value::Null;
    }
}

/// Mistral's wire-level differences: the relaxation that lets a structured
/// format ride beside tools, its content chunks, and the reasoning field its
/// assistant messages refuse.
///
/// A forced tool choice goes as the shared encoder spells it (`required`),
/// and assistant messages carry no `prefix` and no inserted empty
/// `content`, as upstream's live re-recordings send them. Assistant
/// `reasoning_content` is removed unless it is the field the catalog names
/// for the model (`reasoning_field`), which upstream sends.
///
/// Its multimodal content mapping is [`mistral_content`].
fn finalize_mistral(
    map: &mut serde_json::Map<String, serde_json::Value>,
    reasoning_field: Option<&str>,
) -> Result<(), EncodeError> {
    // Mistral rejects forced tool calls beside JSON response formats.
    // Relax the choice rather than discard the requested schema; text formats are exempt.
    let forces_a_tool_call = map
        .get("tool_choice")
        .is_some_and(|choice| !matches!(choice.as_str(), Some("auto" | "none")));
    let has_tools = map
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let has_structured_format = map
        .get("response_format")
        .and_then(|format| format.get("type"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|kind| matches!(kind, "json_schema" | "json_object"));
    if forces_a_tool_call && has_tools && has_structured_format {
        tracing::debug!(
            "relaxing tool_choice to `auto`: Mistral rejects a forced tool choice \
             alongside a response format"
        );
        map.insert(
            "tool_choice".to_owned(),
            serde_json::Value::String("auto".to_owned()),
        );
    }

    let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        let is_assistant =
            message.get("role").and_then(serde_json::Value::as_str) == Some("assistant");

        // Mistral carries text, images, audio and documents as its own
        // chunk array; a part of a type it has no chunk for is forwarded as
        // it is, never removed.
        if let Some(content) = message.get_mut("content") {
            mistral_content(content)?;
        }

        if is_assistant && reasoning_field != Some("reasoning_content") {
            // Mistral rejects unknown assistant fields; hidden reasoning
            // cannot be echoed back, except under the field the catalog
            // names for the model.
            message.shift_remove("reasoning_content");
        }
    }
    Ok(())
}

/// Mistral's text chunk tag.
const MISTRAL_TEXT: &str = "text";
/// Mistral's image chunk tag.
const MISTRAL_IMAGE: &str = "image_url";
/// Mistral's audio chunk tag.
const MISTRAL_AUDIO: &str = "input_audio";
/// Mistral's document chunk tag.
const MISTRAL_DOCUMENT: &str = "document_url";
/// Mistral's uploaded-file chunk tag.
const MISTRAL_FILE: &str = "file";
/// OpenAI's refusal part: textual, but under a key Mistral's chunk schema
/// has no field for, so it is re-tagged rather than forwarded.
const MISTRAL_REFUSAL: &str = "refusal";

/// The text a part carries, under either key the shared conversion uses.
fn mistral_part_text(part: &serde_json::Value) -> Option<&str> {
    part.get(MISTRAL_TEXT)
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            part.get(MISTRAL_REFUSAL)
                .and_then(serde_json::Value::as_str)
        })
}

fn mistral_unsupported(what: &str) -> EncodeError {
    crate::message::MessageError::ConversionError(format!(
        "Mistral cannot carry {what}. Mistral messages accept text, `{MISTRAL_IMAGE}`, \
         `{MISTRAL_AUDIO}`, `{MISTRAL_DOCUMENT}` and `{MISTRAL_FILE}` content; convert the \
         content to one of those before sending it."
    ))
    .into()
}

/// Convert file data to a document URL or a file reference to a top-level file ID.
/// Preserve optional filenames for inline documents. Return a conversion error
/// when neither file data nor a file ID is present.
fn mistral_file_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    let file = part.get(MISTRAL_FILE);
    let field = |name: &str| {
        file.and_then(|file| file.get(name))
            .and_then(serde_json::Value::as_str)
    };

    // Accept already-converted file references to keep finalization idempotent.
    if let Some(file_id) = part.get("file_id").and_then(serde_json::Value::as_str) {
        return Ok(serde_json::json!({"type": MISTRAL_FILE, "file_id": file_id}));
    }

    if let Some(data) = field("file_data") {
        // `document_name` is left out entirely rather than sent as null when
        // the part has no filename.
        Ok(match field("filename") {
            Some(filename) => serde_json::json!({
                "type": MISTRAL_DOCUMENT,
                MISTRAL_DOCUMENT: data,
                "document_name": filename,
            }),
            None => serde_json::json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: data}),
        })
    } else if let Some(file_id) = field("file_id") {
        Ok(serde_json::json!({"type": MISTRAL_FILE, "file_id": file_id}))
    } else {
        Err(mistral_unsupported(
            "a file content part carrying neither `file_data` nor `file_id`",
        ))
    }
}

/// Convert string or object audio payloads to Mistral's base64-string form.
/// Return a conversion error for missing or nonstring audio data.
fn mistral_audio_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    let payload = part.get(MISTRAL_AUDIO).ok_or_else(|| {
        mistral_unsupported("an audio content part carrying no `input_audio` payload")
    })?;

    let data = match payload {
        serde_json::Value::String(data) => data.as_str(),
        payload => payload
            .get("data")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                mistral_unsupported(
                    "an audio content part whose `input_audio` payload is not base64 data",
                )
            })?,
    };

    Ok(serde_json::json!({"type": MISTRAL_AUDIO, MISTRAL_AUDIO: data}))
}

/// One content part as the Mistral chunk that carries it.
///
/// Dispatched on the `type` tag, which the shared conversion always emits, so
/// a part naming a chunk kind is converted as that kind regardless of what
/// other keys it carries. A part of any other type (Magistral `thinking`, a
/// decoded `reference`, or content Mistral has no chunk for), and an
/// untagged part carrying no text, is forwarded verbatim, as upstream
/// forwards it.
fn mistral_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    /// Text and refusal parts are both re-tagged `text`: Mistral's schema has
    /// no `refusal` field and every chunk forbids unknown keys.
    fn text_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
        let text = mistral_part_text(part)
            .ok_or_else(|| mistral_unsupported("a text content part carrying no text"))?;
        Ok(serde_json::json!({"type": MISTRAL_TEXT, MISTRAL_TEXT: text}))
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) => text_chunk(part),
        // Rebuild the envelope because Mistral rejects unknown sibling fields.
        Some(MISTRAL_IMAGE) => {
            let image = part.get(MISTRAL_IMAGE).ok_or_else(|| {
                mistral_unsupported("an image content part carrying no `image_url` payload")
            })?;
            Ok(serde_json::json!({"type": MISTRAL_IMAGE, MISTRAL_IMAGE: image}))
        }
        Some(MISTRAL_AUDIO) => mistral_audio_chunk(part),
        Some(MISTRAL_FILE) => mistral_file_chunk(part),
        // Accept already-converted document parts to keep finalization idempotent.
        Some(MISTRAL_DOCUMENT) => {
            let url = part.get(MISTRAL_DOCUMENT).ok_or_else(|| {
                mistral_unsupported("a document content part carrying no `document_url`")
            })?;
            Ok(match part.get("document_name") {
                Some(name) => serde_json::json!({
                    "type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: url, "document_name": name,
                }),
                None => serde_json::json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: url}),
            })
        }
        // Any other type, whether a Mistral chunk already (Magistral
        // `thinking`, a decoded `reference`) or one Mistral has no chunk
        // for, is forwarded as it is rather than refused or removed.
        Some(_) => Ok(part.clone()),
        // Untagged, but textual: it converts as text.
        None if mistral_part_text(part).is_some() => text_chunk(part),
        // Untagged and not textual: forwarded as it is, as upstream does.
        None => Ok(part.clone()),
    }
}

/// Convert every part of an array `content` to its Mistral content chunk
/// ([`mistral_chunk`]); a text-only array stays an array, as upstream sends
/// it. Nonarrays remain unchanged. A part of a known chunk type missing its
/// payload returns a conversion error.
fn mistral_content(content: &mut serde_json::Value) -> Result<(), EncodeError> {
    for part in content.as_array_mut().into_iter().flatten() {
        *part = mistral_chunk(part)?;
    }
    Ok(())
}

pub(crate) fn apply_openrouter_prompt_caching(
    map: &mut serde_json::Map<String, serde_json::Value>,
) {
    let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let Some(system) = messages
        .iter_mut()
        .find(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("system"))
    else {
        return;
    };
    match system.get("content").cloned() {
        Some(serde_json::Value::String(text)) => {
            if let Some(object) = system.as_object_mut() {
                object.insert(
                    "content".to_owned(),
                    serde_json::json!([{
                        "type": "text",
                        "text": text,
                        "cache_control": { "type": "ephemeral" }
                    }]),
                );
            }
        }
        Some(serde_json::Value::Array(mut parts)) => {
            // The last block marks the cache boundary without altering earlier content.
            if let Some(last) = parts.last_mut()
                && let Some(object) = last.as_object_mut()
            {
                object.insert(
                    "cache_control".to_owned(),
                    serde_json::json!({ "type": "ephemeral" }),
                );
            }
            if let Some(object) = system.as_object_mut() {
                object.insert("content".to_owned(), serde_json::Value::Array(parts));
            }
        }
        _ => {}
    }
}

/// Return a request error for document or image inputs using provider file IDs.
fn refuse_file_ids(request: &CompletionRequest) -> Result<(), EncodeError> {
    use crate::message::{DocumentSourceKind, Message, UserContent};

    let refusal = || {
        EncodeError::request("Provider file IDs are not supported for OpenRouter document inputs")
    };
    for message in &request.chat_history {
        let Message::User { content, .. } = message else {
            continue;
        };
        for part in content {
            match part {
                UserContent::Document(document) => {
                    if matches!(
                        document.data,
                        crate::message::DocumentData::File(DocumentSourceKind::FileId(_))
                    ) {
                        return Err(refusal());
                    }
                }
                UserContent::Image(image) => {
                    if matches!(image.data, DocumentSourceKind::FileId(_)) {
                        return Err(refusal());
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

impl Wire for Chat {
    type Op = crate::operation::Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder;
    type Reassembler = document::ChatCompletion;

    /// Format deferral permits tool composition; dialects without schema
    /// support require the agent's tool-mode enforcement instead.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(
                    self.provider.dialect.quirks.supports_response_format,
                ),
            ))
            .replay(self)
            .usage_observation(UsageObservation::PayloadSnapshots)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAIConfig::completion_headers)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::new(self.provider.dialect.quirks).with_provider(self.provider.dialect.name)
    }

    fn decoder_in<'id>(&self, mode: Mode) -> Self::Decoder<'id> {
        let mut decoder = self.decoder();
        decoder.unary = mode == Mode::Unary;
        decoder
    }

    fn credential_stamp(&self) -> Option<crate::wire::CredentialStamp> {
        self.provider.credential_stamp()
    }

    fn reassembler(&self) -> Self::Reassembler {
        document::ChatCompletion::new(self.provider.dialect.quirks)
    }
}

impl crate::completion::ReplayTarget for Chat {
    fn check_document(&self, document: &crate::message::Document) -> Result<(), EncodeError> {
        check_pdf_document(document)
    }

    /// Section 6.2 of the typed-options design, by dialect.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::providers::openai::options::chat_options(self, request, fields)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("openai.chat")
    }

    fn provider(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Chat reads no images in assistant messages, images in tool results
    /// only on a dialect that says so, and tools where the dialect takes
    /// them. Which models read user images follows each provider's
    /// documented model rules (`reads_images`).
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        let quirks = &self.provider.dialect.quirks;
        let user_images = reads_images(&self.provider.dialect, model);
        crate::completion::Accepts {
            user_images,
            assistant_images: false,
            tool_result_images: user_images && quirks.supports_image_tool_results,
            tools: quirks.supports_tools,
        }
    }

    /// What the Chat encoder carries, by dialect: an image as a URL or
    /// typed data; audio as data; video as a URL or typed data, except to
    /// OpenAI, Azure and Mistral; a PDF as data, with original URL and raw
    /// sources refused before adaptation; a file id where the dialect takes
    /// one; and a text document as text. DeepSeek and Mira take text only;
    /// Perplexity and
    /// Cohere take no media but images, and Ollama only images as data.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::Media;
        let dialect = &self.provider.dialect;
        let rewrite = dialect.quirks.rewrite;
        let parts = !matches!(rewrite, BodyRewrite::DeepSeek | BodyRewrite::Mira);
        // Cohere and Ollama read images and text, and no other media.
        let files = parts
            && rewrite != BodyRewrite::Perplexity
            && ![super::dialects::COHERE.name, super::dialects::OLLAMA.name]
                .contains(&dialect.name);
        let linked = |source: &Source, typed: bool| match source {
            Source::Url(_) => true,
            Source::Base64(_) => typed,
            Source::Raw(_) | Source::FileId(_) | Source::Unknown => false,
        };
        match media {
            Media::Image(image, place) => {
                parts
                    && place != crate::completion::Place::Assistant
                    && linked(&image.data, image.media_type.is_some())
                    // Ollama reads an image as data and never fetches a URL.
                    && !(dialect.name == super::dialects::OLLAMA.name
                        && matches!(image.data, Source::Url(_)))
            }
            Media::Audio(audio) => files && matches!(audio.data, Source::Base64(_)),
            Media::Video(video) => {
                files
                    && rewrite != BodyRewrite::Mistral
                    && ![super::dialects::OPENAI.name, super::dialects::AZURE.name]
                        .contains(&dialect.name)
                    && linked(&video.data, video.media_type.is_some())
            }
            Media::Document(document) => {
                let pdf = document.media_type == Some(DocumentMediaType::PDF);
                match &document.data {
                    DocumentData::Text(_) => true,
                    DocumentData::File(Source::FileId(_)) => {
                        files && dialect.quirks.accepts_file_ids
                    }
                    DocumentData::File(Source::Base64(_)) => files && pdf,
                    DocumentData::File(Source::Url(_)) => {
                        // Original/direct PDF checks refuse this legacy URL predicate.
                        pdf && matches!(rewrite, BodyRewrite::OpenRouter | BodyRewrite::Mistral)
                    }
                    DocumentData::File(Source::Raw(_) | Source::Unknown) => false,
                }
            }
        }
    }

    /// pi's rule for this wire. A `call|item` id joins its sanitized halves
    /// with `_`, ending a result over 40 characters in a hash of the whole
    /// id; OpenAI's own ids are cut to 40; any other id is kept. Mistral
    /// takes any such id (checked live on eight models), where pi derives
    /// nine alphanumerics.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        let sanitized =
            |part: &str| crate::providers::internal::wire_ids::legal_call_id(part, usize::MAX);
        if let Some((call, item)) = id.split_once('|') {
            let call = sanitized(call);
            let item = sanitized(item);
            let combined = if item.is_empty() {
                call.clone()
            } else {
                format!("{call}_{item}")
            };
            if combined.len() <= 40 {
                return combined;
            }
            let hash: String = crate::providers::internal::wire_ids::short_hash(id)
                .chars()
                .take(8)
                .collect();
            let prefix: String = call.chars().take((40 - hash.len() - 1).max(1)).collect();
            return format!("{prefix}_{hash}");
        }
        if self.provider.dialect.name == super::dialects::OPENAI.name {
            return id.chars().take(40).collect();
        }
        id.to_owned()
    }

    /// What an edited block keeps of its item: a custom call's kind and
    /// Mistral's thinking part (`type`), or the field reasoning arrived in,
    /// as pi keeps `thinkingSignature`. Signed data such as
    /// `reasoning_details` never survives an edit.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        if let Some(kind @ ("custom" | "thinking")) = item.str("type") {
            return Map::from_iter([("type".to_owned(), Value::from(kind))]);
        }
        REASONING_TEXT_KEYS
            .iter()
            .find(|key| item.get(**key).is_some_and(Value::is_string))
            .map(|key| Map::from_iter([((*key).to_owned(), Value::String(String::new()))]))
            .unwrap_or_default()
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn later_system(&self, _model: &str) -> crate::completion::LaterSystem {
        self.provider.dialect.quirks.later_system
    }

    /// In array mode a result's parts go as an array.
    fn result_parts(&self, _model: &str) -> bool {
        self.tool_result_array_content
    }

    /// A Chat message needs content or calls, as pi's rebuild does: text
    /// only when it is not empty or its item carries audio, reasoning only
    /// as a content part (Mistral's thinking), an opaque item only as a
    /// typed part, and never an image.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        match (block, block.replay(self, None)) {
            (AssistantContent::ToolCall(_) | AssistantContent::CustomToolCall(_), _) => true,
            (AssistantContent::CarriedReasoning(reasoning), _) => {
                self.provider.dialect.quirks.reasoning_details && !reasoning.content.is_empty()
            }
            (AssistantContent::CarriedText(text), _) => text.foreign_text().is_some_and(|known| {
                !known.is_empty() || self.provider.dialect.name == super::dialects::OPENAI.name
            }),
            (AssistantContent::ProviderItem(_), _) => false,
            (AssistantContent::Text(text), Replay::Item(item)) => {
                !text.text.is_empty() || item.get("audio").is_some()
            }
            (AssistantContent::Text(text), _) => !text.text.is_empty(),
            (AssistantContent::Reasoning(_), Replay::Item(item)) => item.get("type").is_some(),
            (AssistantContent::Reasoning(_), Replay::Identity(identity)) => {
                identity.get("type").and_then(Value::as_str) == Some("thinking")
            }
            (AssistantContent::Opaque(opaque), _) => opaque.item.get("type").is_some(),
            (AssistantContent::Reasoning(_), Replay::Rebuild) | (AssistantContent::Image(_), _) => {
                false
            }
        }
    }
}

/// The catalog vendor whose entries say which models `vendor` serves read
/// images, and the naming rule for an id the catalog does not list:
/// `vendor` is a dialect's name or the vendor an OpenRouter model id starts
/// with. Groq reads images on its Llama 4 models; Mistral's Codestral and
/// Devstral are text models; OpenAI (on Azure too), xAI, Cohere, Z.AI,
/// Moonshot, MiniMax and MiMo apply the rules their other wires share; any
/// other vendor's unlisted model reads them. `None` for DeepSeek's API,
/// which takes text content only (it answers an image part with a 400), and
/// Mira's gateway, which takes text, whatever the model.
fn image_rule(vendor: &str) -> Option<(&str, fn(&str) -> bool)> {
    use crate::providers::{minimax, moonshot, xiaomimimo, zai};
    Some(match vendor {
        "deepseek" | "mira" => return None,
        "groq" => ("groq", |model| model.contains("llama-4")),
        "mistral" | "mistralai" => ("mistral", |model| {
            !(model.starts_with("codestral") || model.starts_with("devstral"))
        }),
        "openai" | "azure.openai" => ("openai", crate::providers::openai::reads_images),
        "xai" | "x-ai" => ("xai", crate::providers::xai::reads_images),
        "cohere" => ("cohere", crate::providers::cohere::reads_images),
        "zai" | "z-ai" => ("zai", zai::reads_images),
        "moonshot" | "moonshotai" => ("moonshot", moonshot::reads_images),
        "minimax" => ("minimax", minimax::reads_images),
        "xiaomimimo" | "xiaomi" => ("xiaomimimo", xiaomimimo::reads_images),
        vendor => (vendor, |_| true),
    })
}

/// Whether `model` reads user images on `dialect`, as the model catalog
/// lists it, or by the naming rule of [`image_rule`] for a model it does not
/// list. OpenRouter names a model `vendor/model`: its own catalog entry
/// decides, or failing that the vendor's naming rule.
fn reads_images(dialect: &super::Dialect, model: &str) -> bool {
    match model.split_once('/') {
        Some((vendor, upstream)) if dialect.quirks.rewrite == BodyRewrite::OpenRouter => {
            crate::catalog::reads_images_or(dialect.name, model, |_| {
                image_rule(vendor).is_some_and(|(_, rule)| rule(upstream))
            })
        }
        _ => image_rule(dialect.name)
            .is_some_and(|(vendor, rule)| crate::catalog::reads_images_or(vendor, model, rule)),
    }
}

/// Classified Chat Completions frame, including whole replies and terminal signals.
pub enum ChatEvent {
    /// A `chat.completion.chunk`: one step of a streamed turn.
    Chunk(Value),
    /// A `chat.completion`: the whole turn in one frame.
    Whole(Value),
    /// The `[DONE]` sentinel: the provider ended the stream.
    Done,
    /// The wire's in-band error envelope, delivered with a 200 status.
    Failure(ProviderError),
    /// A bare JSON string where an envelope belongs: the whole answer, with
    /// no metadata and no terminal reason. Mira's gateway sends this.
    BareText(String),
}

/// The keys an assistant message carries reasoning under, in the order
/// their text is read: compatible servers that send several send the same
/// text under each, so the first that carries text is the one replayed.
const REASONING_TEXT_KEYS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// The structured reasoning key, kept verbatim with the reasoning text.
const REASONING_DETAILS: &str = "reasoning_details";

/// What one content part of a Chat message is. Every part becomes a block.
#[derive(Debug, PartialEq)]
pub(crate) enum Part {
    /// Answer text: a `text` part, or a `refusal` part's text.
    Text(String),
    /// Mistral and Magistral reasoning: a `thinking` part's text.
    Thinking(String),
    /// An image the model produced.
    Image,
    /// A part rig has no canonical meaning for, kept as it came.
    Unknown,
}

impl Part {
    /// The part `part` is, by its `type`. A part with no `type` and a
    /// string `text` is text.
    pub(crate) fn of(part: &Value) -> Self {
        let text = |key: &str| part.str(key).unwrap_or_default().to_owned();
        match part.str("type") {
            Some("text") => Self::Text(text("text")),
            None if part.str("text").is_some() => Self::Text(text("text")),
            Some("refusal") => Self::Text(text("refusal")),
            Some("thinking") => Self::Thinking(match part.get("thinking") {
                Some(Value::Array(chunks)) => chunks
                    .iter()
                    .filter_map(|chunk| chunk.str("text"))
                    .collect(),
                _ => text("thinking"),
            }),
            Some("image_url") => Self::Image,
            _ => Self::Unknown,
        }
    }
}

/// What one tool call of a Chat message is.
#[derive(Debug, PartialEq)]
pub(crate) enum CallKind {
    /// A function call: its arguments are JSON text.
    Function,
    /// A custom tool call: its input is free text, the call's `{"input"}`.
    Custom,
    /// A call kind rig cannot answer, kept but never sent back.
    Unknown,
}

impl CallKind {
    /// The kind of `call`, by its `type`; a call without one is a function
    /// call.
    pub(crate) fn of(call: &Value) -> Self {
        match call.str("type") {
            None | Some("function") => Self::Function,
            Some("custom") => Self::Custom,
            Some(_) => Self::Unknown,
        }
    }
}

/// The kind of run a decoder is writing: answer text, or Mistral's
/// thinking parts, which pi's `mistral-conversations` keeps apart.
#[derive(Clone, Copy, PartialEq)]
enum Writing {
    Text,
    Thinking,
}

/// A tool call the reply has open: its writer index, the id it states,
/// whether its kind is one rig cannot answer (an opaque item), and the
/// function arguments streamed so far.
struct OpenCall {
    at: usize,
    id: Option<String>,
    opaque: bool,
    arguments: String,
}

impl OpenCall {
    /// Whether its function arguments are already a whole object.
    fn complete(&self) -> bool {
        matches!(
            crate::json_utils::parse_tool_arguments(&self.arguments),
            Ok(Value::Object(_))
        )
    }
}

/// The chat-completions decoder: one state machine for a whole reply and a
/// stream of chunks, reading every frame leniently as JSON. A whole reply
/// is the one chunk carrying its message. All of a message's reasoning is
/// one block, as pi's `openai-completions` keeps it, open until the message
/// ends; the turn holds it first however late it arrived, while its events
/// keep their arrival order. A run of text is a block, as is a run of
/// Mistral thinking parts, and each tool call, image and unknown content
/// part is a block of its own; a run closes when another block starts. An
/// answer's audio transcript is its text when it has no other, and its
/// first text block holds the audio's id.
#[derive(Default)]
pub struct ChatDecoder {
    quirks: Quirks,
    provider: Option<&'static str>,
    saw_content: bool,
    unary: bool,
    /// The run being written, its writer index, and a thinking run's text.
    writing: Option<(Writing, usize)>,
    thinking_text: String,
    /// The writer index of the message's reasoning block while it is open,
    /// and its item as it will hold it: the field its text arrived in, its
    /// text, and every `reasoning_details` entry in order.
    reasoning: Option<usize>,
    reasoning_field: Option<&'static str>,
    reasoning_text: String,
    reasoning_details: Map<String, Value>,
    /// The id of the answer's audio.
    audio_id: Option<Value>,
    /// The calls still open, in the order they opened.
    calls: Vec<OpenCall>,
    usage: Option<Value>,
    /// The top-level [`reported::REPLY_FIELDS`] as last stated.
    reply_fields: Map<String, Value>,
    /// The writer index of the message's first text block, which the
    /// message-level citations cite.
    first_text: Option<usize>,
    /// The writer index of the latest text block, which a Mistral
    /// `reference` chunk after it cites.
    last_text: Option<usize>,
    /// Citations of the turn's first text block held until it opens:
    /// message annotations, and references that came before any text.
    turn_citations: Vec<crate::wire::WireCitation>,
    finish: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Whether a finish reason or a whole reply ended the turn.
    ended: bool,
    /// Whether a stream chunk arrived.
    chunked: bool,
}

impl ChatDecoder {
    fn new(quirks: Quirks) -> Self {
        Self {
            quirks,
            ..Self::default()
        }
    }

    fn with_provider(mut self, provider: &'static str) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Absorb the metadata every frame carries, and take its primary choice.
    /// Usage is the frame's, or its choice's where Moonshot reports it (pi).
    fn absorb(&mut self, frame: &Value) -> Option<Value> {
        if let Some(id) = frame.str("id") {
            self.response_id = Some(id.to_owned());
        }
        if let Some(model) = frame.str("model") {
            self.response_model = Some(model.to_owned());
        }
        // `n > 1` streams interleave candidates told apart only by
        // `choices[].index`; candidate 0 is the turn, as in a whole reply.
        let choice = frame
            .arr("choices")
            .iter()
            .find(|choice| {
                let index = choice.get("index").and_then(Value::as_u64);
                index.is_none_or(|index| index == 0)
            })
            .cloned();
        if let Some(usage) = frame
            .at("/usage")
            .filter(|usage| !usage.is_null())
            .or_else(|| {
                choice
                    .as_ref()
                    .and_then(|choice| choice.at("/usage").filter(|usage| !usage.is_null()))
            })
        {
            self.usage = Some(usage.clone());
        }
        for key in reported::REPLY_FIELDS {
            if let Some(value) = frame.get(key).filter(|value| !value.is_null()) {
                self.reply_fields.insert(key.to_owned(), value.clone());
            }
        }
        let choice = choice?;
        // A gateway's upstream-native reason is consulted only when the
        // normalized field is absent or empty (OpenRouter's precedence).
        let reason = match choice
            .str("finish_reason")
            .filter(|reason| !reason.is_empty())
        {
            Some(reason) => Some(carried_finish_reason(reason, &self.quirks)),
            None => choice
                .str("native_finish_reason")
                .filter(|reason| self.quirks.native_finish_reason && !reason.is_empty())
                .map(native_finish_reason),
        };
        if let Some(reason) = reason {
            self.finish = Some(reason);
            self.ended = true;
        }
        Some(choice)
    }

    /// One delta of the assistant message: its reasoning text and
    /// `reasoning_details` into the reasoning block, then the content,
    /// images and calls.
    fn delta(
        &mut self,
        delta: &Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let reasoning = REASONING_TEXT_KEYS.iter().find_map(|key| {
            delta
                .get(*key)
                .and_then(Value::as_str)
                .map(|text| (*key, text))
        });
        // F's reasoning_content.or(reasoning) gives even an explicitly
        // empty first field precedence over the alternate spelling.
        if let Some((key, text)) = reasoning.filter(|(_, text)| !text.is_empty()) {
            self.saw_content |= !text.is_empty();
            self.reason(text, out)?;
            self.reasoning_field.get_or_insert(key);
        }
        if let Some(Value::Array(details)) = delta.get(REASONING_DETAILS)
            && !details.is_empty()
        {
            if self.quirks.reasoning_details {
                self.emit_carried_details(details, false, out)?;
                // Signatures and text fragments still belong to the canonical
                // plaintext block when encrypted entries share their frame.
                let plain: Vec<_> = details
                    .iter()
                    .filter(|detail| detail.str("type") != Some("reasoning.encrypted"))
                    .cloned()
                    .collect();
                if !plain.is_empty() {
                    self.reason("", out)?;
                    merge_details(&mut self.reasoning_details, plain);
                }
            } else {
                self.reason("", out)?;
                merge_details(&mut self.reasoning_details, details.clone());
            }
        }
        let audio = delta.get("audio");
        if let Some(id) = audio.and_then(|audio| audio.at("/id")) {
            self.audio_id = Some(id.clone());
        }
        match delta.get("content") {
            Some(Value::Array(parts)) => {
                for part in parts {
                    self.part(part, out)?;
                }
            }
            Some(part @ Value::Object(_)) => self.part(part, out)?,
            _ => {
                let text = ["content", "refusal"]
                    .iter()
                    .find_map(|key| {
                        delta
                            .get(*key)
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                    })
                    .or_else(|| audio.and_then(|audio| audio.str("transcript")))
                    .filter(|text| !text.is_empty());
                if let Some(text) = text {
                    self.write(Writing::Text, text, out)?;
                }
            }
        }
        for image in delta
            .get("images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.image(image, out)?;
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.call(call, out)?;
        }
        self.turn_citations.extend(
            delta
                .get("annotations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(reported::annotation),
        );
        self.cite_turn(out);
        Ok(())
    }

    /// Cite the turn's first text block with the citations held for it, as
    /// soon as it exists, so a reply cut off before its end keeps them.
    fn cite_turn(&mut self, out: &mut Out<'_, Completion>) {
        if let Some(text) = self.first_text {
            for citation in self.turn_citations.drain(..) {
                out.cite(text, citation);
            }
        }
    }

    /// Append `text` to the message's reasoning block, opening it first.
    fn reason(&mut self, text: &str, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let at = match self.reasoning {
            Some(at) => at,
            None => {
                let at = out.fresh_index();
                out.open(at, Block::Reasoning { redacted: false }, Value::Null)?;
                out.lead(at)?;
                self.reasoning = Some(at);
                at
            }
        };
        self.reasoning_text.push_str(text);
        out.push(at, text)
    }

    /// Close the message's reasoning block, holding its item: the field its
    /// text arrived in and its details.
    fn close_reasoning(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some(at) = self.reasoning.take() else {
            return Ok(());
        };
        let text = std::mem::take(&mut self.reasoning_text);
        let mut item = Map::new();
        if let Some(field) = self.reasoning_field.take() {
            item.insert(field.to_owned(), text.clone().into());
        }
        item.append(&mut self.reasoning_details);
        if self.quirks.reasoning_details {
            let details = item
                .get(REASONING_DETAILS)
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default();
            if let Some(reasoning) = self.signed_reasoning(&text, &item, details) {
                return out.finish_carried_reasoning(at, reasoning);
            }
        }
        out.edit(at, |slot| *slot = Value::Object(item))?;
        out.finish(at)
    }

    /// Append `text` to a block of kind `writing`, first closing the block
    /// being written and opening a new one when that is of another kind.
    fn write(
        &mut self,
        writing: Writing,
        text: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.saw_content |= !text.is_empty();
        let index = match self.writing {
            Some((current, index)) if current == writing => index,
            _ => {
                self.close_writing(out)?;
                let index = out.fresh_index();
                let block = match writing {
                    Writing::Thinking => Block::Reasoning { redacted: false },
                    Writing::Text => {
                        self.first_text.get_or_insert(index);
                        self.last_text = Some(index);
                        Block::Text
                    }
                };
                out.open(index, block, Value::Null)?;
                self.writing = Some((writing, index));
                index
            }
        };
        if writing == Writing::Thinking {
            self.thinking_text.push_str(text);
        }
        out.push(index, text)
    }

    /// Close the run being written, holding its item: the thinking part,
    /// or the audio's id.
    fn close_writing(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let (index, item) = match self.writing.take() {
            None => return Ok(()),
            Some((Writing::Text, index)) => (
                index,
                self.audio_id
                    .take()
                    .map(|id| json!({ "audio": { "id": id } })),
            ),
            Some((Writing::Thinking, index)) => {
                let text = std::mem::take(&mut self.thinking_text);
                (
                    index,
                    Some(json!({"type": "thinking", "thinking": [{"type": "text", "text": text}]})),
                )
            }
        };
        if let Some(item) = item {
            out.edit(index, |slot| *slot = item)?;
        }
        out.finish(index)
    }

    /// One content part, through [`Part::of`].
    #[deny(clippy::wildcard_enum_match_arm)]
    fn part(&mut self, part: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match Part::of(part) {
            Part::Text(text) => self.write(Writing::Text, &text, out),
            Part::Thinking(text) => self.write(Writing::Thinking, &text, out),
            Part::Image => self.image(part, out),
            Part::Unknown => {
                self.saw_content = true;
                self.close_writing(out)?;
                if let Some(citation) = reported::reference(part) {
                    match self.last_text {
                        Some(text) => out.cite(text, citation),
                        None => self.turn_citations.push(citation),
                    }
                }
                let index = out.fresh_index();
                out.whole(index, Block::Opaque { replay: true }, part.clone(), "")
            }
        }
    }

    /// One image the model produced, an `image_url` part, whole: a data URL
    /// becomes base64 data, any other URL stays a URL.
    fn image(&mut self, part: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        use crate::message::{Image, ImageMediaType};
        self.saw_content = true;
        let url = part
            .at("/image_url/url")
            .or_else(|| part.get("image_url"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let inline = url
            .strip_prefix("data:")
            .and_then(|rest| rest.split_once(";base64,"));
        let mut image = Image::default();
        let data = match inline {
            Some((mime, data)) => {
                image.data = Source::Base64(String::new());
                image.media_type = ImageMediaType::from_mime_type(mime);
                data
            }
            None => {
                image.data = Source::Url(url.to_owned());
                ""
            }
        };
        self.close_writing(out)?;
        let index = out.fresh_index();
        out.whole(index, Block::Image(image), part.clone(), data)
    }

    /// One tool-call fragment. It goes to the call at its wire index, where
    /// the writer tells a new id under it from the call it held. Without an
    /// index (or with `null`) it goes to the open call that states its id;
    /// one that states no id continues the latest call while that call's
    /// arguments are not yet a whole object, and otherwise opens a new call
    /// when it brings a name or arguments. Its fields merge into the call's
    /// item.
    fn call(&mut self, call: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        self.close_writing(out)?;
        self.saw_content = true;
        let id = match call.get("id") {
            Some(Value::String(id)) if !id.is_empty() && id != "null" => Some(id.clone()),
            Some(id @ Value::Number(_)) => Some(id.to_string()),
            _ => None,
        };
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        let arguments = call
            .at("/function/arguments")
            .map(crate::json_utils::value_to_json_string);
        let name = call
            .at("/function/name")
            .or_else(|| call.at("/custom/name"))
            .and_then(Value::as_str);
        let starts = name.is_some_and(|name| !name.is_empty())
            || arguments.as_deref().is_some_and(|text| !text.is_empty());
        let at = match index {
            Some(index) => index,
            None => match id.as_deref() {
                Some(id) => self
                    .calls
                    .iter()
                    .find(|open| open.id.as_deref() == Some(id)),
                None => self
                    .calls
                    .last()
                    .filter(|open| open.opaque || !open.complete() || !starts),
            }
            .map_or_else(|| out.fresh_index(), |open| open.at),
        };
        let opaque = match self.calls.iter_mut().find(|open| open.at == at) {
            Some(open) => {
                if id.is_some() {
                    open.id.clone_from(&id);
                }
                open.arguments
                    .push_str(arguments.as_deref().unwrap_or_default());
                open.opaque
            }
            None => {
                let opaque = CallKind::of(call) == CallKind::Unknown;
                if opaque {
                    out.open(at, Block::Opaque { replay: false }, Value::Null)?;
                }
                self.calls.push(OpenCall {
                    at,
                    id: id.clone(),
                    opaque,
                    arguments: arguments.clone().unwrap_or_default(),
                });
                opaque
            }
        };
        if !opaque {
            out.fragment(
                Some(at),
                CallFragment {
                    id: id.as_deref(),
                    name,
                    arguments: arguments.as_deref(),
                },
            )?;
        }
        let mut fields = call.as_object().cloned().unwrap_or_default();
        // The index orders the stream; the call it assembles has none.
        fields.shift_remove("index");
        out.edit(at, |item| {
            if !item.is_object() {
                *item = Value::Object(Map::new());
            }
            if let Value::Object(item) = item {
                merge_fields(item, &fields);
            }
        })?;
        if self.quirks.emits_complete_single_chunk_tool_calls
            && name.is_some_and(|name| !name.is_empty())
            && arguments
                .as_deref()
                .is_some_and(|arguments| !arguments.is_empty())
            && self.calls.iter().any(|open| {
                open.at == at
                    && !open.opaque
                    && crate::json_utils::parse_tool_arguments(&open.arguments).is_ok()
            })
        {
            // F probes the slot this complete fragment names; another
            // slot's parsed arguments do not make it complete here.
            self.calls.retain(|open| open.at != at);
            out.finish(at)?;
        }
        Ok(())
    }

    /// Close the open calls, with their items as natives: every one at the
    /// end (`all`), only those whose arguments are whole at a finish chunk,
    /// since a provider may still send fragments after it. A custom call's
    /// arguments are its `{"input"}`. An explicit output-length finish
    /// discards an open function call whose input is empty or malformed,
    /// preserving its streamed fragments without declaring a complete call.
    fn close_calls(
        &mut self,
        out: &mut Out<'_, Completion>,
        all: bool,
    ) -> Result<(), ProviderError> {
        let cut = self.finish == Some(FinishReason::Length);
        let mut waiting = Vec::new();
        for call in std::mem::take(&mut self.calls) {
            if call.opaque {
                if all && self.ended {
                    out.finish(call.at)?;
                } else {
                    waiting.push(call);
                }
                continue;
            }
            let (mut input, mut complete, mut substantive, mut named, mut malformed) =
                (None, false, false, false, None);
            out.edit(call.at, |item| {
                if CallKind::of(item) == CallKind::Custom {
                    let custom = item.at("/custom/input").cloned();
                    input = Some(custom.unwrap_or_else(|| Value::from("")));
                }
                named = item
                    .at("/function/name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| !name.is_empty());
                complete = match item.at("/function/arguments") {
                    Some(Value::String(text)) => {
                        substantive = !text.trim().is_empty();
                        match crate::json_utils::parse_tool_arguments(text) {
                            Ok(_) => true,
                            Err(error) => {
                                malformed = Some((text.clone(), error.to_string()));
                                false
                            }
                        }
                    }
                    Some(_) => {
                        substantive = true;
                        true
                    }
                    // F permits a parameterless call at an ordinary finish.
                    None => true,
                };
            })?;
            if cut && input.is_none() && (!substantive || !complete) {
                out.discard_call(call.at, crate::streaming::DiscardReason::Length)?;
                continue;
            }
            // F flushes only fully delivered function calls before a
            // transport/provider failure. U-only custom items keep their
            // original unfinished path, and malformed function prefixes
            // remain visible without a fabricated end.
            if (!all && named && !complete) || (input.is_some() && (!all || !self.ended)) {
                waiting.push(call);
                continue;
            }
            if named
                && input.is_none()
                && let Some((raw, error)) = malformed
            {
                let provider =
                    call.id
                        .and_then(crate::message::ProviderCallId::new)
                        .map(|call_id| crate::error::ProviderCallEvidence {
                            call_id,
                            item_id: None,
                        });
                return Err(out.malformed_call_error(call.at, raw, error, provider));
            }
            if let Some(input) = input {
                out.announce(call.at, json!({ "input": input }))?;
            }
            if self.ended {
                out.finish(call.at)?;
            } else {
                // Delivered JSON does not establish an upstream native
                // item without the provider's terminal assertion.
                out.close(call.at)?;
            }
        }
        self.calls = waiting;
        Ok(())
    }

    /// One `chat.completion.chunk`.
    fn chunk(&mut self, frame: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Ok(());
        };
        if let Some(delta) = choice.obj("delta") {
            self.delta(delta, out)?;
        }
        // Text and reasoning stay open until the stream ends, as pi closes
        // its blocks (`openai-completions.ts` around 680): a provider may
        // still send either after the finish chunk.
        if self.finish == Some(FinishReason::ToolCalls) {
            self.close_calls(out, true)?;
        }
        Ok(())
    }

    /// The `chat.completion` body: the one chunk whose delta is its whole
    /// message, each call at its position in the list, then the end.
    fn whole(
        &mut self,
        frame: &Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Err(ProviderError::Response(
                "Response contained no choices".to_owned(),
            ));
        };
        let Some(mut message) = choice.obj("message").cloned() else {
            return Err(ProviderError::Response(
                "Response did not contain a valid message or tool call".to_owned(),
            ));
        };
        self.ended = true;
        if self.quirks.reasoning_details {
            let details = message
                .get(REASONING_DETAILS)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let structured = self.emit_carried_details(&details, true, &mut out)?;
            let signed = if structured {
                false
            } else {
                let plaintext = REASONING_TEXT_KEYS
                    .iter()
                    .find_map(|field| message.get(*field).and_then(Value::as_str))
                    .filter(|text| !text.is_empty());
                if let Some(text) = plaintext {
                    if let Some(reasoning) = self.signed_reasoning(text, &message, &details) {
                        out.carried_reasoning(reasoning)?;
                        self.saw_content = true;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if structured || signed {
                message.shift_remove(REASONING_DETAILS);
                for field in REASONING_TEXT_KEYS {
                    message.shift_remove(field);
                }
            }
        }
        if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
            for (index, call) in calls.iter_mut().enumerate() {
                if let Some(call) = call.as_object_mut() {
                    call.insert("index".to_owned(), index.into());
                }
            }
        }
        self.delta(&message, &mut out)?;
        if !self.saw_content
            && !self
                .finish
                .as_ref()
                .is_some_and(FinishReason::truncated_output)
        {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        self.close_calls(&mut out, true)?;
        self.end(out)
    }

    /// Write the provider's end of the reply. Its `raw` is the wire's
    /// reassembler's.
    fn end(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        self.close_writing(&mut out)?;
        self.close_reasoning(&mut out)?;
        self.cite_turn(&mut out);
        if let Some(text) = self.first_text {
            for citation in reported::listed(&self.reply_fields) {
                out.cite(text, citation);
            }
        }
        let cost = reported::cost(self.usage.as_ref(), &self.reply_fields);
        let usage = self
            .usage
            .as_ref()
            .map(|usage| normalized_usage(usage, &self.quirks))
            .unwrap_or_default()
            .cost(cost);
        Ok(out.end(Finish {
            usage,
            reason: self.finish.take(),
            response_id: self.response_id.take(),
            model: self.response_model.take(),
            ..Finish::default()
        }))
    }

    /// The stream ended: flush the calls the provider delivered, then end
    /// the reply. A stream no finish reason ended was cut short.
    fn finish(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        self.close_calls(&mut out, true)?;
        if !self.ended {
            if self.unary && !self.chunked {
                return Err(ProviderError::Response(
                    crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
                ));
            }
            return Err(ProviderError::Truncated);
        }
        self.end(out)
    }

    /// The `[DONE]` sentinel. A dialect whose streams omit the finish reason
    /// ends a turn that streamed a chunk as pi does without
    /// `supportsFinishReason`: a tool call when it holds one, else a stop.
    /// Any other `[DONE]` before a finish reason was cut short.
    fn done(&mut self, out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        if !self.ended && self.chunked && self.quirks.done_without_finish_reason {
            self.finish = Some(if self.calls.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            });
            self.ended = true;
        }
        self.finish(out)
    }

    fn reasoning_payload(&self, item: Value) -> crate::message::CarriedPayload {
        let issuer = self.provider.map(|provider| {
            if self.quirks.upstream_reasoning_issuer {
                self.response_model
                    .as_deref()
                    .map(|model| super::upstream_reasoning_issuer(provider, model))
                    .unwrap_or_else(|| provider.to_owned())
            } else {
                provider.to_owned()
            }
        });
        crate::message::CarriedPayload::new(item, issuer)
    }

    fn signed_reasoning(
        &self,
        text: &str,
        item: &Map<String, Value>,
        details: &[Value],
    ) -> Option<crate::message::CarriedReasoning> {
        if text.is_empty() {
            return None;
        }
        let signature = details
            .iter()
            .filter(|detail| detail.str("type") == Some("reasoning.text"))
            .find_map(|detail| {
                detail
                    .str("signature")
                    .filter(|signature| !signature.is_empty())
            })?;
        Some(crate::message::CarriedReasoning::new(
            vec![crate::message::CarriedReasoningPart::Text {
                text: text.to_owned(),
                signature: Some(signature.to_owned()),
            }],
            self.reasoning_payload(Value::Object(item.clone())),
        ))
    }

    /// Gateway details retain their independent issuer and block boundaries.
    fn emit_carried_details(
        &mut self,
        details: &[Value],
        whole: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<bool, ProviderError> {
        use crate::message::{CarriedReasoning, CarriedReasoningPart};
        let mut emitted = false;
        for detail in details {
            let part = match detail.str("type") {
                Some("reasoning.encrypted") => detail
                    .str("data")
                    .filter(|data| !whole || !data.is_empty())
                    .map(|data| CarriedReasoningPart::Encrypted(data.to_owned())),
                Some("reasoning.summary") if whole => detail
                    .str("summary")
                    .filter(|text| !text.is_empty())
                    .map(|text| CarriedReasoningPart::Summary(text.to_owned())),
                Some("reasoning.text") if whole => detail
                    .str("text")
                    .filter(|text| !text.is_empty())
                    .map(|text| CarriedReasoningPart::Text {
                        text: text.to_owned(),
                        signature: detail
                            .str("signature")
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned),
                    }),
                _ => None,
            };
            if let Some(part) = part {
                let reasoning =
                    CarriedReasoning::new(vec![part], self.reasoning_payload(detail.clone()))
                        .with_id(
                            detail
                                .str("id")
                                .filter(|id| !id.is_empty())
                                .map(str::to_owned),
                        );
                out.carried_reasoning(reasoning)?;
                self.saw_content = true;
                emitted = true;
            }
        }
        Ok(emitted)
    }
}

/// Append streamed reasoning details to the block's, by pi's merge: a
/// text or summary fragment continues the last entry of its type and
/// index, appending its text and filling the fields that entry lacks, and an
/// encrypted entry stays whole.
fn merge_details(block: &mut Map<String, Value>, details: Vec<Value>) {
    let Value::Array(merged) = block
        .entry(REASONING_DETAILS)
        .or_insert_with(|| Value::Array(Vec::new()))
    else {
        return;
    };
    for detail in details {
        let kind = detail.str("type");
        let continues = matches!(kind, Some("reasoning.text" | "reasoning.summary"))
            && merged.last().is_some_and(|last| {
                last.str("type") == kind && last.get("index") == detail.get("index")
            });
        match (continues, merged.last_mut(), detail) {
            (true, Some(Value::Object(last)), Value::Object(fields)) => {
                for (key, value) in fields {
                    let missing = last
                        .get(&key)
                        .is_none_or(|existing| existing.is_null() || existing.as_str() == Some(""));
                    match (last.get_mut(&key), value) {
                        (Some(Value::String(text)), Value::String(more))
                            if matches!(key.as_str(), "text" | "summary") =>
                        {
                            text.push_str(&more);
                        }
                        (_, value) if missing => {
                            last.insert(key, value);
                        }
                        _ => {}
                    }
                }
            }
            (_, _, detail) => merged.push(detail),
        }
    }
}

#[deny(clippy::wildcard_enum_match_arm)]
impl<'id> Decoder<'id, Completion> for ChatDecoder {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ChatEvent> {
        let data = frame.as_str();
        // `[DONE]` is the wire's terminal sentinel, not JSON.
        if data == "[DONE]" {
            return WireEvent::Known(ChatEvent::Done);
        }
        // The in-band error envelope arrives with a 200 status and is this
        // wire's own terminal failure.
        if let Some(error) = provider_error_envelope(&data) {
            return WireEvent::Known(ChatEvent::Failure(error));
        }
        if self.quirks.accepts_bare_string_reply
            && let Ok(Value::String(text)) = serde_json::from_str::<serde_json::Value>(&data)
        {
            return WireEvent::Known(ChatEvent::BareText(text));
        }
        let classified = classify_chat_completions_frame::<Value>(&data).map(|mut frame| {
            carried::check_choice_list(&frame, &data)?;
            // Reassembled unary documents retain the stream's object tag, so
            // in unary mode a first frame carrying a message is the whole
            // reply unless it is a chunk: one with a delta and no finish yet
            // (Perplexity streams the message so far beside each delta).
            // Once a chunk has been read, and on a stream, the tag still
            // distinguishes cumulative messages from deltas so their text is
            // not repeated.
            let choices = frame.arr("choices");
            let has_message = choices.iter().any(|choice| choice.obj("message").is_some());
            let chunk_shaped = choices.iter().any(|choice| choice.obj("delta").is_some())
                && !choices.iter().any(|choice| {
                    choice
                        .str("finish_reason")
                        .is_some_and(|reason| !reason.is_empty())
                });
            let whole = (self.unary && !self.chunked && has_message && !chunk_shaped)
                || match frame.str("object") {
                    Some(object) => object == "chat.completion",
                    None => has_message,
                };
            if whole {
                carried::prepare_whole_calls(&mut frame, &data, &self.quirks)?;
                Ok(ChatEvent::Whole(frame))
            } else {
                Ok(ChatEvent::Chunk(frame))
            }
        });
        match classified {
            WireEvent::Known(Ok(event)) => WireEvent::Known(event),
            WireEvent::Known(Err(error)) => WireEvent::Corrupt(error),
            WireEvent::Unknown { event_type, value } => WireEvent::Unknown { event_type, value },
            WireEvent::Corrupt(error) => WireEvent::Corrupt(error),
        }
    }

    fn decode(
        &mut self,
        event: ChatEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ChatEvent::Chunk(frame) => {
                self.chunked = true;
                self.chunk(&frame, &mut out)?;
                Ok(Flow::More)
            }
            ChatEvent::Whole(frame) => self.whole(&frame, out),
            ChatEvent::Done => self.done(out),
            // A bare string is the whole answer, ended: pi's stop for a
            // reply that names no reason.
            ChatEvent::BareText(text) => {
                self.ended = true;
                self.finish = Some(FinishReason::Stop);
                if !text.is_empty() {
                    self.write(Writing::Text, &text, &mut out)?;
                }
                self.end(out)
            }
            ChatEvent::Failure(error) => {
                self.close_calls(&mut out, false)?;
                Err(error)
            }
        }
    }

    fn flush_before_terminal_error(
        &mut self,
        mut out: Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        self.close_calls(&mut out, false)
    }

    /// A stream that stops after a finish reason without `[DONE]` still
    /// ended: some dialects (Perplexity) never send the sentinel.
    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.finish(out)
    }
}

/// A Chat usage object, normalized for a dialect with `quirks` and read
/// leniently, so no usage shape fails a reply: a counter that is absent,
/// `null` or not a non-negative integer is unreported, and one inside a
/// details object the usage has is zero. Cached input falls back to
/// Mistral's `num_cached_tokens` and then DeepSeek's
/// `prompt_cache_hit_tokens`; audio input counts as input when the total
/// says it sits beside the prompt; output is the remainder of the total when
/// unreported; and the reasoning count is left out where the dialect's
/// cannot be trusted ([`Quirks::reliable_reasoning_count`]).
fn normalized_usage(usage: &Value, quirks: &Quirks) -> crate::completion::Usage {
    let count = |pointer: &str| usage.at(pointer).and_then(Value::as_u64);
    let detail = |object: &str, key: &str| {
        count(&format!("/{object}/{key}")).or(usage.obj(object).map(|_| 0))
    };
    let (prompt, completion, total) = (
        count("/prompt_tokens"),
        count("/completion_tokens"),
        count("/total_tokens"),
    );
    let audio = count("/prompt_tokens_details/audio_tokens").unwrap_or(0);
    let input = prompt.map(|prompt| {
        let beside = prompt.saturating_add(audio);
        let accounted = beside.saturating_add(completion.unwrap_or(0));
        if audio != 0 && Some(accounted) == total {
            beside
        } else {
            prompt
        }
    });
    crate::completion::Usage {
        input_tokens: input,
        output_tokens: completion
            .or_else(|| total.map(|total| total.saturating_sub(input.unwrap_or(0)))),
        total_tokens: total,
        cached_input_tokens: detail("prompt_tokens_details", "cached_tokens")
            .or(count("/num_cached_tokens"))
            .or(count("/prompt_cache_hit_tokens")),
        cache_creation_input_tokens: count("/prompt_tokens_details/cache_write_tokens"),
        reasoning_tokens: detail("completion_tokens_details", "reasoning_tokens")
            .filter(|_| quirks.reliable_reasoning_count),
        ..Default::default()
    }
}

impl ChatDecoder {
    /// Verdict, model, response id and error envelope, read off a raw
    /// payload before normalization discards them. The driver calls it for
    /// the unary reply and for every stream frame.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<ObservedPayload>(payload) else {
            return;
        };
        let usage = payload
            .usage
            .map(|usage| AdapterUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .and_then(|details| details.cached_tokens),
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|details| details.reasoning_tokens),
                tool_input_tokens: None,
            })
            .or_else(|| {
                // Preserve U's actual choice-level usage source as a raw
                // snapshot after F's whole-payload shape gate succeeds.
                // A reported top-level usage object always wins.
                payload
                    .choices
                    .iter()
                    .find(|choice| {
                        choice
                            .index
                            .as_ref()
                            .and_then(Value::as_u64)
                            .is_none_or(|index| index == 0)
                    })
                    .and_then(|choice| choice.usage.as_ref())
                    .filter(|usage| usage.is_object())
                    .map(|usage| {
                        let count = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64);
                        AdapterUsage {
                            input_tokens: count("/prompt_tokens"),
                            output_tokens: count("/completion_tokens"),
                            total_tokens: count("/total_tokens"),
                            cached_input_tokens: count("/prompt_tokens_details/cached_tokens"),
                            reasoning_tokens: count("/completion_tokens_details/reasoning_tokens"),
                            tool_input_tokens: None,
                        }
                    })
            });
        if let Some(usage) = usage {
            sink.usage_snapshot(usage);
        }
        // Only the chunk that carries the finish reason is a verdict, so the
        // model rides with it rather than on each delta.
        let choice = payload.choices.into_iter().next().unwrap_or_default();
        let verdict = match choice.finish_reason {
            Some(reason) => AdapterVerdict {
                finish_reason: Some(sink.scrub(&reason)),
                block_reason: None,
                detail: None,
                model: payload.model.map(|value| sink.scrub(&value)),
            },
            None => AdapterVerdict::default(),
        };
        let response_id = payload.id.map(|value| sink.scrub(&value));
        sink.provider(verdict, response_id);
        if let Some(error) = payload.error {
            ObservedError {
                code: error.code,
                kind: error.kind,
                message: error.message,
            }
            .emit(sink);
        }
    }
}

/// F's optional observation fields still gate the entire projection: a
/// malformed field suppresses usage, verdict and error together. Unknown
/// fields remain ignored, including the U choice-level fallback fields.
#[derive(Deserialize)]
struct ObservedPayload {
    id: Option<String>,
    model: Option<String>,
    usage: Option<ObservedUsage>,
    #[serde(default)]
    choices: Vec<ObservedChoice>,
    error: Option<ObservedChatError>,
}

#[derive(Deserialize)]
struct ObservedUsage {
    #[serde(default, deserialize_with = "observed_count")]
    prompt_tokens: Option<u64>,
    #[serde(default, deserialize_with = "observed_count")]
    completion_tokens: Option<u64>,
    #[serde(default, deserialize_with = "observed_count")]
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<ObservedTokenDetails>,
    completion_tokens_details: Option<ObservedTokenDetails>,
}

#[derive(Deserialize)]
struct ObservedTokenDetails {
    #[serde(default, deserialize_with = "observed_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "observed_count")]
    reasoning_tokens: Option<u64>,
}

fn observed_count<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u64>, D::Error> {
    Ok(Value::deserialize(deserializer)?.as_u64())
}

#[derive(Default, Deserialize)]
struct ObservedChoice {
    finish_reason: Option<String>,
    index: Option<Value>,
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct ObservedChatError {
    code: Option<Value>,
    #[serde(rename = "type", alias = "status")]
    kind: Option<String>,
    message: Option<String>,
}

pub(crate) use carried::fold_groq_native_tools;
use carried::{
    carried_details, carried_finish_reason, flatten_text_content_parts, sanitize_plain_text_history,
};
mod carried;
mod document;
mod reported;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod hard_case_tests;

#[cfg(test)]
mod history_tests;

#[cfg(test)]
mod request_tests;

#[cfg(test)]
mod carried_tests;

#[cfg(test)]
mod gemini_tests;

#[cfg(test)]
mod document_diagnostics_tests;
