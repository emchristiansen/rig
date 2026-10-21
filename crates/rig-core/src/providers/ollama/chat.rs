//! The native Ollama chat wire: `POST /api/chat`, a whole reply or a stream
//! of NDJSON records, both read by [`ChatDecoder`].
//!
//! The request carries the daemon's own fields: `think` and `keep_alive` at
//! the top level, model parameters such as `num_ctx` in `options`, and the
//! output schema as `format`.
//!
//! ```
//! use rig_core::providers::ollama::OllamaConfig;
//!
//! let wire = OllamaConfig::new().client().native_completion("qwen3:4b").wire;
//! assert_eq!(wire.model, "qwen3:4b");
//! ```

use serde_json::{Map, Value, json};

use crate::completion::options::{BaseInput, FinalBody, RawAt, request_params};
use crate::completion::{
    Accepts, CompletionRequest, Media, Place, ProviderCapabilities, Replay, ReplayTarget,
};
use crate::error::EncodeError;
use crate::message::{
    AssistantContent, AssistantMessage, DocumentData, DocumentSourceKind as Source, Message,
    ToolCall, ToolResult, ToolResultContent, UserContent,
};
use crate::operation::Completion;
use crate::completion::history::OptionalCallId;

const OLLAMA_CHAT_WIRE: &str = "Ollama Chat";
use crate::wire::{Capabilities, Descriptor, Encoded, Framing, Mode, Wire};

use super::streaming::ChatDecoder;
use super::{OllamaConfig, PROVIDER_NAME};

/// The chat endpoint, relative to the daemon's address.
const CHAT_PATH: &str = "/api/chat";

/// The `additional_params` keys `/api/chat` reads at the top level of its
/// request. Every other key is a model parameter and goes in `options`.
const TOP_LEVEL: &[&str] = &[
    "think",
    "format",
    "keep_alive",
    "logprobs",
    "top_logprobs",
    "truncate",
    "shift",
];

/// The native chat wire of an Ollama daemon.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Chat {
    /// The daemon's address and credential.
    pub provider: OllamaConfig,
    /// The model to address, e.g. [`QWEN3`](super::QWEN3).
    pub model: String,
}

impl Chat {
    /// The wire for `model` on `provider`.
    pub fn new(provider: OllamaConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// The `/api/chat` body `request` sends in `mode`: the wire's encoding,
    /// then the mapped options, then `additional_params`, split by where the
    /// daemon reads each key. `think` and the keys in [`TOP_LEVEL`] go at the
    /// top level, `tools` join the request's tools, an `options` object
    /// merges into `options`, and every other key is an `options` entry.
    /// `temperature` and `max_tokens` (as `num_predict`) go in `options`,
    /// where a caller's own entries win.
    fn body(&self, request: &CompletionRequest, mode: Mode) -> Result<FinalBody, EncodeError> {
        self.check_provider_items(request)?;
        self.check_nested_opaque(request)?;
        self.check_documents(request)?;
        let mut request = request.clone();
        if let Some(params) = request
            .additional_params
            .as_mut()
            .and_then(Value::as_object_mut)
        {
            if let Some(think) = params.get_mut("think") {
                match think {
                    Value::Bool(_) => {}
                    Value::String(level)
                        if matches!(
                            level.to_lowercase().as_str(),
                            "low" | "medium" | "high" | "max"
                        ) =>
                    {
                        *level = level.to_lowercase()
                    }
                    _ => {
                        return Err(EncodeError::request(
                            "`think` must be a 'low', 'medium', 'high', 'max' or bool",
                        ));
                    }
                }
            }
            if params
                .get("keep_alive")
                .is_some_and(|value| !value.is_string())
            {
                return Err(EncodeError::request("`keep_alive` must be a string"));
            }
        }
        request_params(
            self,
            &request,
            |input| self.base(&request, mode, input),
            RawAt::Split {
                top: TOP_LEVEL,
                rest: "options",
            },
            &[],
        )
    }

    /// The wire's own encoding of `request`.
    fn base(
        &self,
        request: &CompletionRequest,
        mode: Mode,
        input: &mut BaseInput<'_>,
    ) -> Result<Map<String, Value>, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        if input
            .param("keep_alive")
            .is_some_and(|value| !(value.is_string() || value.is_number()))
        {
            return Err(EncodeError::request(
                "Ollama `keep_alive` must be a duration string or a number of seconds",
            ));
        }
        if input
            .param("options")
            .is_some_and(|value| !value.is_object())
        {
            return Err(EncodeError::request(
                "Ollama `additional_params.options` must be an object",
            ));
        }
        let messages = self.messages(&request.chat_history)?;

        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }})
            })
            .collect();
        tools.extend(input.raw_tools()?);
        if request.tool_choice.is_some() {
            tracing::warn!("Ollama has no tool choice; `tool_choice` is ignored");
        }

        let mut options = Map::new();
        if let Some(temperature) = request.temperature {
            options.insert("temperature".to_owned(), Value::from(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".to_owned(), Value::from(max_tokens));
        }

        // Defer the schema until a tool result exists: a constrained reply
        // cannot call a tool.
        let answered = messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("tool"));
        let format = request
            .output_schema
            .clone()
            .filter(|_| tools.is_empty() || answered)
            .map(|schema| schema.to_value());

        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("format", format),
            ("options", Some(Value::Object(options))),
            ("stream", Some(Value::Bool(mode == Mode::Streaming))),
        ];
        Ok(fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect())
    }

    /// The history as name-keyed `/api/chat` messages. Provider-issued
    /// handles remain exact; local identities have no optional wire field.
    fn messages(&self, history: &[Message]) -> Result<Vec<Value>, EncodeError> {
        let mut messages = Vec::new();
        for message in history {
            match message {
                Message::System { content } => {
                    messages.push(json!({"role": "system", "content": content}));
                }
                Message::User { content } => {
                    let mut user = UserParts::default();
                    for part in content {
                        match part {
                            UserContent::ToolResult(result) => {
                                user.push_to(&mut messages);
                                messages.push(tool_message(result)?);
                            }
                            part => user.add(part)?,
                        }
                    }
                    user.push_to(&mut messages);
                }
                Message::Assistant(turn) => {
                    messages.extend(self.assistant(turn)?);
                }
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "Ollama chat request has no messages after conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn: its text joined as `content`, its reasoning as
    /// `thinking`, and its calls. A turn with none of them is `None`.
    fn assistant(
        &self,
        turn: &AssistantMessage,
    ) -> Result<Option<Value>, EncodeError> {
        let (mut text, mut thinking, mut calls) = (Vec::new(), None, Vec::new());
        for block in &turn.content {
            match block {
                AssistantContent::Text(block) => text.push(block.text.clone()),
                AssistantContent::Reasoning(block) if !block.text.is_empty() => {
                    thinking = Some(block.text.clone());
                }
                AssistantContent::ToolCall(call) => {
                    call.for_json_only_wire(OLLAMA_CHAT_WIRE)?;
                    let id = match &call.id {
                        crate::message::CallId::Provider(id) => OptionalCallId::ProviderHandle(id),
                        crate::message::CallId::Local(_) => OptionalCallId::OmitLocal,
                    };
                    calls.push(call_item(call, block.replay_optional_call_id(self, id)));
                }
                AssistantContent::CarriedReasoning(reasoning) => {
                    if reasoning.has_opaque_parts() {
                        return Err(crate::message::UnrepresentableOpaqueContent::new(
                            OLLAMA_CHAT_WIRE,
                        )
                        .into());
                    }
                    let display = reasoning.display_text();
                    if !display.is_empty() {
                        thinking = Some(display);
                    }
                }
                AssistantContent::CarriedText(part) => {
                    if let Some(known) = part.known_text_projection() {
                        text.push(known.to_owned());
                    } else {
                        return Err(crate::message::UnrepresentableOpaqueContent::new(
                            OLLAMA_CHAT_WIRE,
                        ).into());
                    }
                }
                AssistantContent::CustomToolCall(call) => {
                    return Err(call.refused_by_json_only_wire(OLLAMA_CHAT_WIRE).into());
                }
                AssistantContent::ProviderItem(item) => {
                    return Err(crate::message::UnreplayableProviderItem::new(
                        OLLAMA_CHAT_WIRE,
                        item,
                    )
                    .into());
                }
                AssistantContent::Image(_) => {
                    return Err(EncodeError::request(
                        "Ollama currently doesn't support images.",
                    ));
                }
                AssistantContent::Reasoning(_) | AssistantContent::Opaque(_) => {}
            }
        }
        if text.is_empty() && thinking.is_none() && calls.is_empty() {
            return Ok(None);
        }
        let mut message = Map::from_iter([
            ("role".to_owned(), Value::from("assistant")),
            ("content".to_owned(), Value::String(text.join(" "))),
        ]);
        if let Some(thinking) = thinking {
            message.insert("thinking".to_owned(), Value::String(thinking));
        }
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(calls));
        }
        Ok(Some(Value::Object(message)))
    }
}

/// The text and images of one user message as they gather.
#[derive(Default)]
struct UserParts {
    texts: Vec<String>,
    images: Vec<String>,
}

impl UserParts {
    /// Add a part: text, or an image as base64 data. The adapter has
    /// replaced every other part ([`ReplayTarget::encodes`]).
    fn add(&mut self, part: &UserContent) -> Result<(), EncodeError> {
        match part {
            UserContent::Text(text) => self.texts.push(text.text.clone()),
            UserContent::Image(image) => match &image.data {
                Source::Base64(data) => self.images.push(data.clone()),
                _ => {
                    return Err(EncodeError::request(
                        "Ollama images must be base64 encoded data",
                    ));
                }
            },
            UserContent::Document(document) => match &document.data {
                DocumentData::Text(text) => self.texts.push(text.clone()),
                DocumentData::File(Source::Base64(data)) => self.texts.push(data.clone()),
                DocumentData::File(_) => {
                    return Err(EncodeError::request(
                        "Ollama documents must be string or base64 encoded data",
                    ));
                }
            },
            UserContent::Audio(_) => {
                return Err(EncodeError::request(
                    "Ollama does not support audio user content",
                ));
            }
            UserContent::Video(_) => {
                return Err(EncodeError::request(
                    "Ollama does not support video user content",
                ));
            }
            UserContent::ToolResult(_) => return Err(unsendable("a tool result as user content")),
        }
        Ok(())
    }

    /// Push the parts gathered so far as one user message, if there are any.
    fn push_to(&mut self, messages: &mut Vec<Value>) {
        if self.texts.is_empty() && self.images.is_empty() {
            return;
        }
        let mut message = Map::from_iter([
            ("role".to_owned(), Value::from("user")),
            ("content".to_owned(), Value::String(self.texts.join(" "))),
        ]);
        if !self.images.is_empty() {
            message.insert(
                "images".to_owned(),
                Value::Array(self.images.drain(..).map(Value::String).collect()),
            );
        }
        self.texts.clear();
        messages.push(Value::Object(message));
    }
}

/// The error for content `/api/chat` cannot carry, which the adapter
/// replaces before a request is encoded.
fn unsendable(what: &str) -> EncodeError {
    EncodeError::request(format!("Ollama chat cannot carry {what}"))
}

/// A result as the `tool` message answering its call: its text joined into
/// one string, with the tool's name and the call's id.
fn tool_message(result: &ToolResult) -> Result<Value, EncodeError> {
    if result.answers == crate::message::AnsweredToolCall::Custom {
        return Err(crate::message::UnrepresentableToolCall::CustomCall {
            wire: OLLAMA_CHAT_WIRE,
            name: result.name.clone(),
        }
        .into());
    }
    let texts = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(text.text.clone()),
            ToolResultContent::Json { value } => Ok(value.to_string()),
            ToolResultContent::Image(_) => Err(EncodeError::request(
                "Ollama does not support images in tool results",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut message = json!({
        "role": "tool",
        "content": texts.join("\n"),
        "tool_name": result.name,
    });
    match &result.call {
        crate::message::CallId::Provider(id) => message["tool_call_id"] = json!(id.as_str()),
        crate::message::CallId::Local(_) => {}
    }
    Ok(message)
}

/// `call` as a `tool_calls` entry: its item as the daemon sent it when it is
/// current, with the canonical name and arguments and the call's spelled id.
fn call_item(call: &ToolCall, replay: Replay<'_>) -> Value {
    let mut item = match replay {
        Replay::Item(item) => match item.into_owned() {
            Value::Object(item) => item,
            _ => Map::new(),
        },
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    match &call.id {
        crate::message::CallId::Provider(id) => {
            item.insert("id".to_owned(), Value::String(id.as_str().to_owned()));
        }
        crate::message::CallId::Local(_) => {}
    }
    let function = item
        .entry("function")
        .or_insert_with(|| Value::Object(Map::new()));
    if !function.is_object() {
        *function = Value::Object(Map::new());
    }
    if let Value::Object(function) = function {
        function.insert("name".to_owned(), Value::from(call.function.name.as_str()));
        function.insert(
            "arguments".to_owned(),
            Value::Object(call.function.arguments.clone()),
        );
    }
    Value::Object(item)
}

impl Wire for Chat {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder;
    type Reassembler = super::streaming::document::ChatResponse;

    /// The output schema is deferred while tools are unanswered, so it
    /// composes with tools.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(true),
            ))
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = self.body(&request, mode)?;
        let target = match mode {
            Mode::Unary => crate::providers::internal::LogTarget::Completions,
            Mode::Streaming => crate::providers::internal::LogTarget::Streaming,
        };
        crate::providers::internal::trace_json(target, "Ollama chat request", &body);
        let request = self
            .provider
            .request(http::Method::POST, CHAT_PATH)
            .body(body.into_body())?;
        let framing = match mode {
            Mode::Unary => Framing::Whole,
            Mode::Streaming => Framing::Ndjson,
        };
        Ok(Encoded::new(request, framing)
            .with_route(Some(CHAT_PATH))
            .with_projection(ChatDecoder::project))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::default()
    }

    fn decoder_in<'id>(&self, mode: Mode) -> Self::Decoder<'id> {
        ChatDecoder::new(mode)
    }
}

impl ReplayTarget for Chat {
    /// Section 6.5 of the typed-options design, for `/api/chat`.
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        use crate::completion::options::{Mapping, OptionFields, OptionMap};
        use crate::completion::{CacheRetention, Effort, Reasoning};
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = fields;
        const NO_FIELD: &str = "Ollama's `/api/chat` has no such field";
        OptionMap {
            reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
                Reasoning::Off => Mapping::Send(json!({"think": false})),
                Reasoning::Effort(
                    effort @ (Effort::Low | Effort::Medium | Effort::High | Effort::Max),
                ) => Mapping::Send(json!({"think": effort.as_str()})),
                Reasoning::Effort(effort) => Mapping::unsupported(format!(
                    "Ollama has no `{}` thinking level",
                    effort.as_str()
                )),
                Reasoning::Budget { .. } => {
                    Mapping::unsupported("Ollama takes a thinking level, not a budget")
                }
            }),
            cache: Mapping::of(cache, |cache| match cache {
                CacheRetention::None => Mapping::Omit("Ollama keeps no prompt cache to stop"),
                CacheRetention::Short | CacheRetention::Long => Mapping::unsupported(
                    "Ollama has no prompt cache retention; `keep_alive` keeps the model loaded",
                ),
            }),
            service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(NO_FIELD)),
            verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(NO_FIELD)),
            parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
                Mapping::unsupported(NO_FIELD)
            }),
            top_p: Mapping::of(top_p, |top_p| {
                Mapping::Send(json!({"options": {"top_p": top_p}}))
            }),
            seed: Mapping::of(seed, |seed| {
                Mapping::Send(json!({"options": {"seed": seed}}))
            }),
            stop: Mapping::of_stop(stop, |stop| {
                Mapping::Send(json!({"options": {"stop": stop}}))
            }),
        }
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("ollama.chat")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Images in user messages, and none in assistant turns or tool results,
    /// which the adapter moves to a user message.
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts {
            assistant_images: false,
            tool_result_images: false,
            ..Accepts::ALL
        }
    }

    /// An image as base64 data in a user message. The daemon never fetches a
    /// URL, and reads no audio, video or document; a text document's text
    /// is sent by the adapter.
    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        match media {
            Media::Image(image, Place::User) => matches!(image.data, Source::Base64(_)),
            Media::Document(document) => self.check_document(document).is_ok(),
            Media::Image(_, _) | Media::Audio(_) | Media::Video(_) => false,
        }
    }

    fn check_document(&self, document: &crate::message::Document) -> Result<(), EncodeError> {
        match &document.data {
            DocumentData::Text(_) | DocumentData::File(Source::Base64(_)) => Ok(()),
            DocumentData::File(_) => Err(EncodeError::request(
                "Ollama documents must be string or base64 encoded data",
            )),
        }
    }

    fn states_finish_reason(&self) -> bool {
        false
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    /// A message needs content, thinking or calls.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        match block {
            AssistantContent::ToolCall(_) => true,
            AssistantContent::Text(text) => !text.text.is_empty(),
            AssistantContent::Reasoning(reasoning) => !reasoning.text.is_empty(),
            AssistantContent::CarriedReasoning(reasoning) => {
                reasoning.has_opaque_parts() || !reasoning.display_text().is_empty()
            }
            AssistantContent::CustomToolCall(_)
            | AssistantContent::ProviderItem(_)
            | AssistantContent::CarriedText(_) => true,
            AssistantContent::Image(_) | AssistantContent::Opaque(_) => false,
        }
    }
}

#[cfg(test)]
mod tests;
