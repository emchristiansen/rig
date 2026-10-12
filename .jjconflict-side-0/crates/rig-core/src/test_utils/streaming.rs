//! Streaming helpers for [`MockCompletionModel`](super::MockCompletionModel):
//! the script a mock reply is written from, and the decoder that writes it
//! through the completion writer, as a wire's decoder does.

use std::collections::HashMap;

use crate::completion::{CompletionResponse, Usage};
use crate::message::{CarriedPayload, CarriedReasoning, CarriedReasoningPart};
use crate::error::ProviderError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::wire::{Decoder, Flow, Out, WireEvent};

/// Provider descriptor name reported by the test doubles.
pub const MOCK_PROVIDER: &str = "mock";

/// The end the mock model's reply finishes with, carrying `usage`.
pub fn mock_final(usage: Usage) -> Finish {
    Finish {
        usage,
        ..Finish::default()
    }
}

/// A fixture's provider item: a non-empty JSON object, or `None` for
/// `null` and `{}`. Any other value is a scripting mistake surfaced as a
/// stream error.
fn fixture_item(value: serde_json::Value) -> Result<Option<serde_json::Value>, ProviderError> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(map) if map.is_empty() => Ok(None),
        serde_json::Value::Object(map) => Ok(Some(serde_json::Value::Object(map))),
        other => Err(ProviderError::Provider(format!(
            "mock stream fixture `additional_params` must be a JSON object, got: {other}"
        ))),
    }
}

// MOCK-R1-006: private composition at the two scripted text metadata sites.
fn merge_fixture_metadata(
    item: &mut serde_json::Value,
    fields: serde_json::Map<String, serde_json::Value>,
) {
    fn merge_value(existing: &mut serde_json::Value, incoming: serde_json::Value) {
        match (existing, incoming) {
            (serde_json::Value::Object(existing), serde_json::Value::Object(incoming)) => {
                for (key, value) in incoming {
                    match existing.get_mut(&key) {
                        Some(existing) => merge_value(existing, value),
                        None => { existing.insert(key, value); }
                    }
                }
            }
            (serde_json::Value::Array(existing), serde_json::Value::Array(mut incoming)) => {
                existing.append(&mut incoming);
            }
            (existing, incoming) => *existing = incoming,
        }
    }
    if !item.is_object() {
        *item = serde_json::Value::Object(serde_json::Map::new());
    }
    let Some(item) = item.as_object_mut() else { return; };
    for (key, value) in fields {
        let atomic = key == "openai_responses_part"
            && value.get("kind").and_then(serde_json::Value::as_str) == Some("opaque")
            && value.get("value").is_some();
        if atomic {
            item.insert(key, value);
        } else {
            match item.get_mut(&key) {
                Some(existing) => merge_value(existing, value),
                None => { item.insert(key, value); }
            }
        }
    }
}

/// The end of a reply whose usage has only `total_tokens` set.
pub fn mock_final_with_total_tokens(total_tokens: u64) -> Finish {
    mock_final(Usage {
        total_tokens: Some(total_tokens),
        ..Default::default()
    })
}

/// Scripted streaming event yielded by [`MockCompletionModel`](super::MockCompletionModel).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MockStreamEvent {
    /// Text chunk.
    Text(String),
    /// Start a new text part, with the provider item it is decoded from.
    TextStart {
        id: String,
        additional_params: Option<serde_json::Value>,
    },
    /// Fields merged into the current text part's provider item.
    TextAdditionalParams(serde_json::Value),
    /// Complete tool call event.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
        call_id: Option<String>,
    },
    /// A tool call's name, as a wire streams it.
    ToolCallNameDelta { id: String, name: String },
    /// A fragment of a tool call's arguments.
    ToolCallArgumentsDelta { id: String, arguments: String },
    /// The end of a tool call streamed as fragments: the call closes with
    /// what they carried, as a wire's step-stop does.
    ToolCallEnd { id: String },
    /// Complete reasoning event.
    Reasoning { id: String, content: CarriedReasoningPart },
    /// Reasoning delta event.
    ReasoningDelta { id: String, reasoning: String },
    /// The genuine provider-issued assistant message ID, independent of the terminal.
    MessageId(String),
    /// Provider-native output item that Rig does not model.
    Unknown(serde_json::Value),
    /// The transport request id of this turn. The mock transport reports
    /// it, as a real transport reports a response header; the decoder never
    /// sees it. Its position in the turn does not matter; when a turn
    /// scripts more than one, the first wins.
    RequestId(String),
    /// The provider's end of the reply.
    FinalResponse(Finish),
    /// A failure, which ends the reply.
    Error(MockError),
}

use super::completion::MockError;

/// The provider id a fixture spells, when it spells one. Corpus fixtures
/// are plain data: the renderings `reasoning-0`, `block-3`, `output-1`,
/// `tool-2` and `text-0` name a part the wire gave no id, and anything else
/// is the wire's own id.
fn fixture_provider_id(id: &str) -> Option<&str> {
    let unnamed = ["reasoning-", "block-", "output-", "tool-", "text-"]
        .iter()
        .any(|namespace| {
            id.strip_prefix(namespace)
                .is_some_and(|rest| rest.parse::<u64>().is_ok())
        });
    (!id.is_empty() && !unnamed).then_some(id)
}

impl MockStreamEvent {
    /// Create a text chunk.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Start a new text content block identified by `id`.
    pub fn text_start(id: impl Into<String>, additional_params: Option<serde_json::Value>) -> Self {
        Self::TextStart {
            id: id.into(),
            additional_params,
        }
    }

    /// Add provider-specific metadata to the current text content block.
    pub fn text_additional_params(additional_params: serde_json::Value) -> Self {
        Self::TextAdditionalParams(additional_params)
    }

    /// Create a complete tool call event.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
            call_id: None,
        }
    }

    /// Attach a provider-specific call ID to a complete tool call event.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        if let Self::ToolCall { call_id: id, .. } = &mut self {
            *id = Some(call_id.into());
        }
        self
    }

    /// Create a tool call name delta.
    pub fn tool_call_name_delta(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolCallNameDelta {
            id: id.into(),
            name: name.into(),
        }
    }

    /// Create a tool call arguments delta.
    pub fn tool_call_arguments_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolCallArgumentsDelta {
            id: id.into(),
            arguments: arguments.into(),
        }
    }

    /// Create the end of a tool call streamed as deltas.
    pub fn tool_call_end(id: impl Into<String>) -> Self {
        Self::ToolCallEnd { id: id.into() }
    }

    /// Create a complete reasoning event with the default mock id
    /// (`"reasoning-0"`). Use [`Self::with_reasoning_id`] for tests that
    /// need distinct reasoning items.
    pub fn reasoning(reasoning: impl Into<String>) -> Self {
        Self::Reasoning {
            id: "reasoning-0".to_string(),
            content: CarriedReasoningPart::Text {
                text: reasoning.into(),
                signature: None,
            },
        }
    }

    /// Attach a provider-specific reasoning ID to a complete reasoning event.
    pub fn with_reasoning_id(mut self, reasoning_id: impl Into<String>) -> Self {
        if let Self::Reasoning { id, .. } = &mut self {
            *id = reasoning_id.into();
        }
        self
    }

    /// Create a reasoning delta event with the default mock id
    /// (`"reasoning-0"`). Use [`Self::reasoning_delta_with_id`] for tests
    /// that need distinct reasoning items.
    pub fn reasoning_delta(reasoning: impl Into<String>) -> Self {
        Self::reasoning_delta_with_id("reasoning-0", reasoning)
    }

    /// Create a reasoning delta event with an explicit reasoning item id.
    pub fn reasoning_delta_with_id(id: impl Into<String>, reasoning: impl Into<String>) -> Self {
        Self::ReasoningDelta {
            id: id.into(),
            reasoning: reasoning.into(),
        }
    }

    /// Create a provider-issued assistant message-ID event.
    pub fn message_id(id: impl Into<String>) -> Self {
        Self::MessageId(id.into())
    }

    /// Create an unmodeled provider output item.
    pub fn unknown(value: serde_json::Value) -> Self {
        Self::Unknown(value)
    }

    /// Create the provider's end of the reply, with usage.
    pub fn final_response(usage: Usage) -> Self {
        Self::FinalResponse(mock_final(usage))
    }

    /// Create a final response event whose usage reports no counter.
    pub fn final_response_with_default_usage() -> Self {
        Self::FinalResponse(mock_final(Usage::default()))
    }

    /// Create a final response event whose usage has only `total_tokens` set.
    pub fn final_response_with_total_tokens(total_tokens: u64) -> Self {
        Self::FinalResponse(mock_final_with_total_tokens(total_tokens))
    }

    /// Create a stream error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(MockError::provider(message))
    }
}

/// One step of a mock reply: a scripted event, or a whole response.
#[derive(Clone, Debug)]
pub enum MockFrame {
    /// A scripted event.
    Event(MockStreamEvent),
    /// A whole response, as a unary turn answers.
    Response(Box<CompletionResponse>),
}

/// The reply document of [`MockScript`](super::MockScript): a whole
/// response's `raw`, or a stream's scripted end, serialized. A stream that
/// failed before its end has none.
#[derive(Debug, Default)]
pub struct MockDocument {
    document: Option<serde_json::Value>,
    failed: bool,
}

impl crate::wire::document::Serves<crate::operation::Completion> for MockDocument {}

impl crate::wire::document::Reassemble<MockFrame> for MockDocument {
    fn absorb(&mut self, frame: &MockFrame) {
        if self.document.is_some() || self.failed {
            return;
        }
        match frame {
            MockFrame::Response(response) => self.document = Some(response.raw.clone()),
            MockFrame::Event(MockStreamEvent::FinalResponse(finish)) => {
                self.document = serde_json::to_value(finish).ok();
            }
            MockFrame::Event(MockStreamEvent::Error(_)) => self.failed = true,
            MockFrame::Event(_) => {}
        }
    }

    fn finish(self) -> serde_json::Value {
        self.document.unwrap_or(serde_json::Value::Null)
    }
}

// MOCK-R1-005: the exact F assembly bound remains private to this mock.
const MOCK_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct MockCallInput {
    raw: String,
    overflowed: bool,
    named: bool,
}

impl MockCallInput {
    fn append(&mut self, fragment: &str) {
        if self.raw.trim() == "null" && !fragment.trim().is_empty() {
            self.raw.clear();
        }
        if self.raw.len().saturating_add(fragment.len()) > MOCK_TOOL_INPUT_BYTES {
            self.overflowed = true;
        } else {
            self.raw.push_str(fragment);
        }
    }
}

/// The decoder of [`MockScript`](super::MockScript): each scripted step is
/// written through the completion writer.
#[derive(Default)]
pub struct MockDecoder<'id> {
    /// The reasoning item each scripted id streams into, in start order.
    reasoning: Vec<(String, usize)>,
    /// The item index each scripted call id streams under.
    calls: HashMap<String, usize>,
    /// One empty-script-key assembly; the keyed map never stores an empty key.
    empty_call: Option<usize>,
    /// F's bounded argument assembly, scoped to each actual open call index.
    call_inputs: HashMap<usize, MockCallInput>,
    /// An explicit scripted message-ID event outranks the terminal field, as in F.
    message_id: Option<crate::completion::ProviderMessageId>,
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

/// The provider item a scripted id names, when it names one.
fn id_item(id: &str) -> serde_json::Value {
    fixture_provider_id(id).map_or(
        serde_json::Value::Null,
        |id| serde_json::json!({ "id": id }),
    )
}

impl<'id> MockDecoder<'id> {
    fn call_index(&mut self, out: &mut Out<'id, Completion>, id: &str) -> usize {
        if id.is_empty() {
            if let Some(index) = self.empty_call {
                return index;
            }
            let index = out.fresh_index();
            self.empty_call = Some(index);
            return index;
        }
        if let Some(index) = self.calls.get(id) {
            return *index;
        }
        let index = out.fresh_index();
        self.calls.insert(id.to_owned(), index);
        index
    }

    fn event(
        &mut self,
        event: MockStreamEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            MockStreamEvent::Text(text) => {
                out.run(Block::Text, &text)?;
            }
            MockStreamEvent::TextStart {
                id: _,
                additional_params,
            } => {
                out.end_run()?;
                let index = out.run(Block::Text, "")?;
                if let Some(serde_json::Value::Object(fields)) =
                    additional_params.map(fixture_item).transpose()?.flatten()
                {
                    out.edit(index, |slot| merge_fixture_metadata(slot, fields))?;
                }
            }
            MockStreamEvent::TextAdditionalParams(additional_params) => {
                // An empty fixture object is a scripting mistake, not a no-op.
                let Some(serde_json::Value::Object(fields)) = fixture_item(additional_params)?
                else {
                    return Err(ProviderError::Provider(
                        "mock stream fixture `TextAdditionalParams` carries no data — \
                         drop the event instead"
                            .to_string(),
                    ));
                };
                let index = out.run(Block::Text, "")?;
                out.edit(index, |item| merge_fixture_metadata(item, fields))?;
            }
            MockStreamEvent::ToolCall {
                id,
                name,
                arguments,
                call_id,
            } => {
                out.end_run()?;
                // An id-less call is a wire that sends none: rig issues the
                // id. A call scripted as fragments under this id is the one
                // this restatement closes.
                let held = if id.is_empty() {
                    self.empty_call.take()
                } else {
                    self.calls.remove(&id)
                };
                if let Some(index) = held {
                    drop(self.call_inputs.remove(&index));
                }
                let index = held.unwrap_or_else(|| out.fresh_index());
                let restated_object = arguments.is_object();
                if restated_object && held.is_none() {
                    // Keep the whole-only arguments available before its first Start.
                    out.fragment(
                        Some(index),
                        CallFragment {
                            id: call_id.as_deref().or(fixture_provider_id(&id)),
                            ..CallFragment::default()
                        },
                    )?;
                    out.restate(index, &arguments.to_string())?;
                }
                out.fragment(
                    Some(index),
                    CallFragment {
                        id: call_id.as_deref().or(fixture_provider_id(&id)),
                        name: Some(name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
                if restated_object {
                    if held.is_some() {
                        out.restate(index, &arguments.to_string())?;
                    }
                } else {
                    out.announce(index, arguments)?;
                }
                out.finish(index)?;
            }
            MockStreamEvent::ToolCallNameDelta { id, name } => {
                out.end_run()?;
                let index = self.call_index(&mut out, &id);
                if !name.is_empty() {
                    self.call_inputs.entry(index).or_default().named = true;
                }
                out.fragment(
                    Some(index),
                    CallFragment {
                        id: fixture_provider_id(&id),
                        name: Some(name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            MockStreamEvent::ToolCallArgumentsDelta { id, arguments } => {
                out.end_run()?;
                let index = self.call_index(&mut out, &id);
                self.call_inputs.entry(index).or_default().append(&arguments);
                out.fragment(
                    Some(index),
                    CallFragment {
                        id: fixture_provider_id(&id),
                        arguments: Some(arguments.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            MockStreamEvent::ToolCallEnd { id } => {
                let index = if id.is_empty() {
                    self.empty_call.take()
                } else {
                    self.calls.remove(&id)
                };
                let Some(index) = index else { return Ok(Flow::More); };
                let input = self.call_inputs.remove(&index).unwrap_or_default();
                if input.named {
                    let parsed = crate::json_utils::parse_tool_arguments(&input.raw)
                        .and_then(|arguments| {
                            if input.overflowed {
                                Err(<serde_json::Error as serde::de::Error>::custom(
                                    "tool-call input exceeded the accumulation bound",
                                ))
                            } else {
                                Ok(arguments)
                            }
                        });
                    if let Err(error) = parsed {
                        let provider = fixture_provider_id(&id)
                            .and_then(crate::message::ProviderCallId::new)
                            .map(|call_id| crate::error::ProviderCallEvidence {
                                call_id,
                                item_id: None,
                            });
                        return Err(out.malformed_call_error(
                            index, input.raw, error.to_string(), provider,
                        ));
                    }
                }
                out.finish(index)?;
            }
            ref received @ MockStreamEvent::Reasoning { ref id, ref content } => {
                // Capture the received record before moving or converting its content.
                let item = serde_json::to_value(received)?;
                out.end_run()?;
                match content {
                    CarriedReasoningPart::Text { text, signature: None } => {
                        // The unsigned helper retains U's delta-close behavior.
                        match self.reasoning.iter().position(|(open, _)| open == id) {
                            Some(at) => {
                                let (_, index) = self.reasoning.remove(at);
                                out.finish(index)?;
                            }
                            None => {
                                let index = out.fresh_index();
                                out.whole(
                                    index,
                                    Block::Reasoning { redacted: false },
                                    id_item(id),
                                    text,
                                )?;
                            }
                        }
                    }
                    CarriedReasoningPart::Text { signature: Some(_), .. }
                    | CarriedReasoningPart::Summary(_)
                    | CarriedReasoningPart::Encrypted(_)
                    | CarriedReasoningPart::Redacted { .. }
                    | CarriedReasoningPart::OpaqueSummary(_)
                    | CarriedReasoningPart::OpaqueContent(_) => {
                        let reasoning = CarriedReasoning::new(
                            vec![content.clone()],
                            CarriedPayload::new(item, Some(MOCK_PROVIDER.to_owned())),
                        ).with_id(fixture_provider_id(id).map(str::to_owned));
                        match self.reasoning.iter().position(|(open, _)| open == id) {
                            Some(at) => {
                                let (_, index) = self.reasoning.remove(at);
                                out.restate(index, &reasoning.display_text())?;
                                out.finish_carried_reasoning(index, reasoning)?;
                            }
                            None => out.carried_reasoning(reasoning)?,
                        }
                    }
                }
            }
            MockStreamEvent::ReasoningDelta { id, reasoning } => {
                out.end_run()?;
                let index = match self.reasoning.iter().find(|(open, _)| *open == id) {
                    Some((_, index)) => *index,
                    None => {
                        let index = out.fresh_index();
                        out.open(index, Block::Reasoning { redacted: false }, id_item(&id))?;
                        self.reasoning.push((id, index));
                        index
                    }
                };
                out.push(index, &reasoning)?;
            }
            MockStreamEvent::MessageId(id) => {
                if let Some(id) = crate::completion::ProviderMessageId::new(id) {
                    self.message_id = Some(id);
                }
            }
            MockStreamEvent::Unknown(value) => out.unknown(value.into()),
            MockStreamEvent::RequestId(_) => {}
            MockStreamEvent::FinalResponse(mut finish) => {
                finish.message_id = self.message_id.clone().or(finish.message_id);
                out.end_run()?;
                for (_, index) in std::mem::take(&mut self.reasoning) {
                    out.finish(index)?;
                }
                return Ok(out.end(finish));
            }
            MockStreamEvent::Error(error) => return Err(error.into_completion_error()),
        }
        Ok(Flow::More)
    }

    fn response(
        &mut self,
        response: CompletionResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        for content in response.choice.iter().cloned() {
            out.content(content)?;
        }
        Ok(out.end(Finish {
            usage: response.usage,
            reason: response.finish_reason(),
            message_id: response.message_id.clone(),
            response_id: response.response_id().map(str::to_owned),
            model: response.model().map(str::to_owned),
            error: response.error.clone(),
        }))
    }
}

impl<'id> Decoder<'id, Completion, MockFrame> for MockDecoder<'id> {
    type Event = MockFrame;

    fn classify(&self, frame: MockFrame) -> WireEvent<MockFrame> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: MockFrame,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match frame {
            MockFrame::Event(event) => self.event(event, out),
            MockFrame::Response(response) => self.response(*response, out),
        }
    }
}
