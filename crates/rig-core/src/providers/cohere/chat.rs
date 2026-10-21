//! Cohere's native chat API, `POST /v2/chat`: the request built as JSON,
//! with the request's documents sent as Cohere `documents`. Its replies
//! are read by [`ChatDecoder`].
//!
//! ```
//! use rig_core::providers::cohere::{COMMAND_A_03_2025, CohereConfig, NativeChat};
//!
//! let wire = NativeChat::new(CohereConfig::new("key"), COMMAND_A_03_2025);
//! assert_eq!(wire.model, COMMAND_A_03_2025);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::CohereConfig;
use super::streaming::ChatDecoder;
use crate::completion::options::{BaseInput, FinalBody, RawAt, request_params};
use crate::completion::{CompletionRequest, Document, ProviderCapabilities, Replay, ReplayTarget};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, AssistantMessage, Message, ToolChoice, ToolResult, ToolResultContent,
    UserContent,
};
use crate::operation::Completion;
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::{Capabilities, Descriptor, Encoded, Framing, Mode, Wire};

/// Where the native chat endpoint sits under the API root.
const CHAT_PATH: &str = "/v2/chat";

/// The replay API of turns made on the native chat endpoint.
pub(crate) const API: &str = "cohere.chat";
const COHERE_CHAT_WIRE: &str = "Cohere Chat";

/// The native chat wire: a provider configuration, a model, and the
/// per-turn options the endpoint takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NativeChat {
    /// Which provider, and how to reach it.
    pub provider: CohereConfig,
    /// The model this wire addresses.
    pub model: String,
    /// Whether requests ask Cohere to hold tool calls to their schemas
    /// (`strict_tools`).
    pub strict_tools: bool,
}

impl NativeChat {
    /// The wire for `model` on `provider`, with every option off.
    pub fn new(provider: CohereConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            strict_tools: false,
        }
    }

    /// Ask Cohere to hold every tool call to its tool's schema.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// The request body: the wire's encoding of `request`, then the mapped
    /// options, then `additional_params`, merged key by key.
    fn body(&self, request: &CompletionRequest, mode: Mode) -> Result<FinalBody, EncodeError> {
        self.check_provider_items(&request.chat_history)?;
        self.check_nested_content(&request.chat_history)?;
        self.check_documents(request)?;
        request_params(
            self,
            request,
            |input| self.base(request, mode, input),
            RawAt::Top,
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
        let messages = self.messages(&request.chat_history, &model)?;
        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({"type": "function", "function": {
                    "name": tool.name, "description": tool.description,
                    "parameters": tool.parameters,
                }})
            })
            .collect();
        tools.extend(input.raw_tools()?);
        let tool_choice = match &request.tool_choice {
            None => None,
            Some(ToolChoice::None) => Some("NONE"),
            Some(ToolChoice::Required) if !tools.is_empty() => Some("REQUIRED"),
            Some(ToolChoice::Required) => {
                return Err(EncodeError::request(
                    "Cohere requires at least one tool when tool_choice is REQUIRED",
                ));
            }
            Some(ToolChoice::Auto) => {
                return Err(EncodeError::request(
                    "\"auto\" is not an allowed tool_choice value in the Cohere API; omit tool_choice to let the model decide",
                ));
            }
            Some(ToolChoice::Specific { .. }) => {
                return Err(EncodeError::request(
                    "the Cohere API cannot be forced to call specific tools by name; use ToolChoice::Required and restrict the tools you pass instead",
                ));
            }
        };
        let documents: Vec<Value> = request.documents.iter().map(document_value).collect();
        let response_format = request
            .output_schema
            .clone()
            .map(|schema| json!({"type": "json_object", "schema": schema.to_value()}));
        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            ("documents", Some(Value::Array(documents))),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("tool_choice", tool_choice.map(Value::from)),
            (
                "strict_tools",
                self.strict_tools.then_some(Value::Bool(true)),
            ),
            ("response_format", response_format),
            ("temperature", request.temperature.map(Value::from)),
            ("max_tokens", request.max_tokens.map(Value::from)),
            (
                "stream",
                (mode == Mode::Streaming).then_some(Value::Bool(true)),
            ),
        ];
        Ok(fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect())
    }

    /// The history as Cohere messages, each call and result spelled by one
    /// [`WireIds`].
    fn messages(&self, history: &[Message], model: &str) -> Result<Vec<Value>, EncodeError> {
        let ids = WireIds::for_target(history, self, model).map_err(EncodeError::request)?;
        let mut messages = Vec::new();
        for (message_index, message) in history.iter().enumerate() {
            match message {
                Message::System { content } => {
                    messages.push(json!({"role": "system", "content": content}));
                }
                Message::User { content } => {
                    let mut parts = Vec::new();
                    for (content_index, part) in content.iter().enumerate() {
                        if let UserContent::ToolResult(result) = part {
                            user_message(&mut messages, &mut parts);
                            messages.push(tool_message(
                                result,
                                &ids,
                                message_index,
                                content_index,
                            )?);
                        } else {
                            parts.push(user_part(part)?);
                        }
                    }
                    user_message(&mut messages, &mut parts);
                }
                Message::Assistant(turn) => {
                    messages.extend(self.assistant(turn, &ids, message_index)?)
                }
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "Cohere chat request has no messages after conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn: text, thinking and unknown parts as content
    /// parts, a tool plan as `tool_plan`, each call under `tool_calls`, and
    /// the citations each block's item holds, pointed at the part they
    /// cite. A current item goes back as it came, less its citations. A
    /// turn with nothing to send is `None`.
    fn assistant(
        &self,
        turn: &AssistantMessage,
        ids: &WireIds,
        message_index: usize,
    ) -> Result<Option<Value>, EncodeError> {
        let (mut content, mut plan, mut calls, mut citations) =
            (Vec::new(), String::new(), Vec::new(), Vec::new());
        for (content_index, block) in turn.content.iter().enumerate() {
            let replay = block.replay(self, ids.of(message_index, content_index));
            let kind = replay_kind(&replay);
            let mut item = match replay {
                Replay::Item(item) => match item.into_owned() {
                    Value::Object(item) => item,
                    _ => Map::new(),
                },
                Replay::Identity(_) | Replay::Rebuild => Map::new(),
            };
            let cited = match item.shift_remove("citations") {
                Some(Value::Array(cited)) => cited,
                _ => Vec::new(),
            };
            let part = match block {
                AssistantContent::Text(text) if !text.text.is_empty() => {
                    json!({"type": "text", "text": text.text})
                }
                AssistantContent::Reasoning(reasoning) if kind.as_deref() == Some(PLAN) => {
                    citations.extend(pointed(cited, None));
                    plan.push_str(&reasoning.text);
                    continue;
                }
                AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                    json!({"type": "thinking", "thinking": reasoning.text})
                }
                AssistantContent::Opaque(opaque)
                    if opaque.replay && opaque.item.get("type").is_some() =>
                {
                    content.push(opaque.item.clone());
                    continue;
                }
                AssistantContent::ToolCall(call) => {
                    call.for_json_only_wire(COHERE_CHAT_WIRE)?;
                    item.entry("type")
                        .or_insert_with(|| Value::from("function"));
                    item.insert(
                        "id".to_owned(),
                        Value::String(
                            ids.spell(message_index, content_index)
                                .map_err(EncodeError::request)?,
                        ),
                    );
                    let function = item
                        .entry("function")
                        .or_insert_with(|| Value::Object(Map::new()));
                    if !function.is_object() {
                        *function = Value::Object(Map::new());
                    }
                    if let Value::Object(function) = function {
                        function
                            .insert("name".to_owned(), Value::from(call.function.name.as_str()));
                        function.insert(
                            "arguments".to_owned(),
                            Value::String(call.function.arguments_value().to_string()),
                        );
                    }
                    calls.push(Value::Object(item));
                    continue;
                }
                AssistantContent::CarriedReasoning(reasoning) => {
                    if reasoning.has_opaque_parts() {
                        return Err(crate::message::UnrepresentableOpaqueContent::new(
                            COHERE_CHAT_WIRE,
                        )
                        .into());
                    }
                    json!({"type": "thinking", "thinking": reasoning.display_text()})
                }
                AssistantContent::CarriedText(part) => {
                    match part.known_text_projection() {
                        Some(text) => json!({"type": "text", "text": text}),
                        None => return Err(crate::message::UnrepresentableOpaqueContent::new(
                            COHERE_CHAT_WIRE,
                        ).into()),
                    }
                }
                AssistantContent::ProviderItem(item) => {
                    return Err(crate::message::UnreplayableProviderItem::new(
                        COHERE_CHAT_WIRE,
                        item,
                    )
                    .into());
                }
                AssistantContent::CustomToolCall(call) => {
                    return Err(call.refused_by_json_only_wire(COHERE_CHAT_WIRE).into());
                }
                AssistantContent::Image(_) => {
                    return Err(EncodeError::request(
                        "Cohere currently doesn't support images.",
                    ));
                }
                AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Opaque(_) => continue,
            };
            citations.extend(pointed(cited, Some(content.len())));
            content.push(if item.is_empty() {
                part
            } else {
                Value::Object(item)
            });
        }
        if content.is_empty() && calls.is_empty() {
            return Ok(None);
        }
        let fields = [
            ("role", Some(Value::from("assistant"))),
            (
                "content",
                (!content.is_empty()).then_some(Value::Array(content)),
            ),
            (
                "tool_plan",
                (!plan.is_empty()).then_some(Value::String(plan)),
            ),
            (
                "tool_calls",
                (!calls.is_empty()).then_some(Value::Array(calls)),
            ),
            (
                "citations",
                (!citations.is_empty()).then_some(Value::Array(citations)),
            ),
        ];
        Ok(Some(Value::Object(
            fields
                .into_iter()
                .filter_map(|(key, value)| Some((key.to_owned(), value?)))
                .collect(),
        )))
    }
}

/// The item kind a tool plan's reasoning block holds.
pub(crate) const PLAN: &str = "tool_plan";

/// The kind of item replay hands the encoder for a block: the current
/// item's `type`, or the one an edited block kept.
fn replay_kind(replay: &Replay<'_>) -> Option<String> {
    match replay {
        Replay::Item(item) => item.str("type").map(str::to_owned),
        Replay::Identity(identity) => identity
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_owned),
        Replay::Rebuild => None,
    }
}

/// The kind of item `block` replays as on `target`.
fn kind(block: &AssistantContent, target: &NativeChat) -> Option<String> {
    replay_kind(&block.replay(target, None))
}

/// `citations` as a request sends them back: each at `content_index`, the
/// position of the part it cites, or with none for a tool plan's.
fn pointed(citations: Vec<Value>, content_index: Option<usize>) -> Vec<Value> {
    citations
        .into_iter()
        .map(|mut citation| {
            if let (Some(citation), Some(index)) = (citation.as_object_mut(), content_index) {
                citation.insert("content_index".to_owned(), Value::from(index));
            }
            citation
        })
        .collect()
}

/// `document` as a Cohere document: its metadata and `text` under `data`,
/// keys sorted so the same document always sends the same bytes, under its
/// id, including an empty id.
fn document_value(document: &Document) -> Value {
    let mut data: BTreeMap<&str, &str> = document
        .additional_props
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    data.insert("text", &document.text);
    json!({"id": document.id, "data": data})
}

/// One text part on Cohere's native wire. Other content returns a request error.
fn user_part(part: &UserContent) -> Result<Value, EncodeError> {
    Ok(match part {
        UserContent::Text(text) => json!({"type": "text", "text": text.text}),
        _ => {
            return Err(EncodeError::request(
                "Only text content is supported by Cohere",
            ));
        }
    })
}

/// Push the user parts gathered so far as one message.
fn user_message(messages: &mut Vec<Value>, parts: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    messages.push(json!({"role": "user", "content": std::mem::take(parts)}));
}

/// A result as the `tool` message that answers its call, its parts as text.
fn tool_message(
    result: &ToolResult,
    ids: &WireIds,
    message_index: usize,
    content_index: usize,
) -> Result<Value, EncodeError> {
    if result.answers == crate::message::AnsweredToolCall::Custom {
        return Err(crate::message::UnrepresentableToolCall::CustomCall {
            wire: COHERE_CHAT_WIRE,
            name: result.name.clone(),
        }
        .into());
    }
    let content: Vec<Value> = result
        .content
        .iter()
        .map(|part| {
            let text = match part {
                ToolResultContent::Text(text) => text.text.clone(),
                ToolResultContent::Json { value } => value.to_string(),
                ToolResultContent::Image(_) => {
                    return Err(EncodeError::request(
                        "Only text tool result content is supported by Cohere",
                    ));
                }
            };
            Ok(json!({"type": "text", "text": text}))
        })
        .collect::<Result<Vec<_>, EncodeError>>()?;
    Ok(
        json!({"role": "tool", "tool_call_id": ids.spell(message_index, content_index).map_err(EncodeError::request)?, "content": content}),
    )
}

impl Wire for NativeChat {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder;
    type Reassembler = super::streaming::document::ChatResponse;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(true),
            ))
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let body = self.body(&request, mode)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Cohere chat request",
            &body,
        );
        let request = self.provider.post(CHAT_PATH).body(body.into_body())?;
        let framing = match mode {
            Mode::Streaming => Framing::Sse,
            Mode::Unary => Framing::Whole,
        };
        Ok(Encoded::new(request, framing)
            .with_projection(ChatDecoder::project)
            .with_route(Some(CHAT_PATH)))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::default()
    }

    fn decoder_in<'id>(&self, mode: Mode) -> Self::Decoder<'id> {
        ChatDecoder::new(mode)
    }
}

impl crate::completion::ReplayTarget for NativeChat {
    /// Section 6.5 of the typed-options design, for the native API.
    fn map_options(
        &self,
        request: &CompletionRequest,
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
        let model = request.model.as_deref().unwrap_or(&self.model);
        // A model that thinks does so by default. An id the catalog does
        // not list thinks when its name says `reasoning`.
        let thinks = super::thinks(model);
        let reasons = thinks.unwrap_or_else(|| model.contains("reasoning"));
        const NO_FIELD: &str = "Cohere's chat API has no such field";
        OptionMap {
            reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
                Reasoning::Off if reasons => {
                    Mapping::Send(json!({"thinking": {"type": "disabled"}}))
                }
                Reasoning::Off => Mapping::Omit("the model does not think"),
                Reasoning::Effort(_) | Reasoning::Budget { .. } if thinks == Some(false) => {
                    Mapping::unsupported("the model does not think")
                }
                Reasoning::Effort(Effort::High) => {
                    Mapping::Send(json!({"thinking": {"type": "enabled"}}))
                }
                Reasoning::Effort(effort) => Mapping::unsupported(format!(
                    "Cohere takes thinking on or a token budget, not `{}`",
                    effort.as_str()
                )),
                Reasoning::Budget { tokens } => Mapping::Send(json!({
                    "thinking": {"type": "enabled", "token_budget": tokens},
                })),
            }),
            cache: Mapping::of(cache, |cache| match cache {
                CacheRetention::None => Mapping::Omit("Cohere does not cache prompts"),
                CacheRetention::Short | CacheRetention::Long => {
                    Mapping::unsupported("Cohere has no prompt cache")
                }
            }),
            service_tier: Mapping::of(service_tier, |_| {
                Mapping::unsupported("Cohere's `priority` is a queue position, not a tier")
            }),
            verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(NO_FIELD)),
            parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| {
                Mapping::unsupported(NO_FIELD)
            }),
            top_p: Mapping::of(top_p, |top_p| {
                if (0.01..=0.99).contains(&top_p) {
                    Mapping::Send(json!({ "p": top_p }))
                } else {
                    Mapping::unsupported("Cohere takes `p` from 0.01 to 0.99")
                }
            }),
            seed: Mapping::of(seed, |seed| Mapping::Send(json!({ "seed": seed }))),
            stop: Mapping::of_stop(stop, |stop| match stop.len() {
                0..=5 => Mapping::Send(json!({ "stop_sequences": stop })),
                _ => Mapping::unsupported("Cohere takes at most 5 stop sequences"),
            }),
        }
    }

    fn check_document(&self, _document: &crate::message::Document) -> Result<(), EncodeError> {
        Err(EncodeError::request(
            "Only text content is supported by Cohere",
        ))
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static(API)
    }

    fn provider(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Native chat carries text and tools in history.
    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts {
            user_images: false,
            assistant_images: false,
            tool_result_images: false,
            tools: true,
        }
    }

    /// Native history carries no media; retrieval documents use `documents`.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        let _ = media;
        false
    }

    /// An edited block keeps its kind: a tool plan stays a plan.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        item.str("type")
            .filter(|kind| *kind == PLAN)
            .map(|kind| Map::from_iter([("type".to_owned(), Value::from(kind))]))
            .unwrap_or_default()
    }

    /// A bare `message-end` is a genuine terminal even without a reason.
    fn states_finish_reason(&self) -> bool {
        false
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn takes_documents(&self) -> bool {
        true
    }

    /// Text and thinking with text are content parts, an opaque item with
    /// a `type` is a part, and a call is a call. A tool plan rides with its
    /// calls, and images are never sent.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        match block {
            AssistantContent::Text(text) => !text.text.is_empty(),
            AssistantContent::Reasoning(reasoning) => {
                !reasoning.text.is_empty() && kind(block, self).as_deref() != Some(PLAN)
            }
            AssistantContent::Opaque(opaque) => opaque.item.get("type").is_some(),
            AssistantContent::ToolCall(_) => true,
            AssistantContent::CarriedReasoning(reasoning) => !reasoning.display_text().is_empty(),
            AssistantContent::CarriedText(part) => part.known_text_projection().is_some(),
            AssistantContent::CustomToolCall(_) => true,
            AssistantContent::ProviderItem(_)
            | AssistantContent::Image(_) => false,
        }
    }
}

#[cfg(test)]
mod tests;
