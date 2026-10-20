//! Fixture driving at the public stream and canonical history boundaries.

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::driver::Model;
use crate::error::ProviderError;
use crate::message::{AssistantContent, CarriedReasoning, CarriedTextContent, Message};
use crate::operation::Completion;
use crate::providers::openai::wire::OpenAIConfig;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::{Mode, Operation, Wire};
use futures::StreamExt;
use serde_json::{Value, json};

pub(super) struct Driven {
    pub items: Vec<Item<StreamEvent>>,
    pub errors: Vec<ProviderError>,
    pub response: CompletionResponse,
}

/// Poll the actual driver to EOF; `partial` retains delivered output without
/// pretending that a clean EOF is a provider terminal.
pub(super) fn drive(frames: &[Value]) -> Driven {
    futures::executor::block_on(async {
        let sse: String = frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect();
        let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
        let mut stream = Model::new(
            wire,
            MockStreamingClient {
                sse_bytes: sse.into(),
            },
        )
        .stream(CompletionRequest::new("fixture"))
        .expect("actual mock transport stream");
        let mut items = Vec::new();
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => items.push(item),
                Err(error) => errors.push(error),
            }
        }
        Driven {
            items,
            errors,
            response: stream.partial(),
        }
    })
}

pub(super) fn sample_response(status: &str) -> Value {
    json!({"id":"resp_123", "object":"response", "created_at":0,
        "status":status, "model":"gpt-5.4", "output":[], "tools":[]})
}

pub(super) fn completed_with_output(output: Value) -> Value {
    let mut response = sample_response("completed");
    response["output"] = output;
    json!({"type":"response.completed","sequence_number":9,"response":response})
}

pub(super) fn message_item(id: &str, content: Value) -> Value {
    json!({"type":"message","id":id,"role":"assistant","status":"completed","content":content})
}

pub(super) fn fold_driven(frames: &[Value]) -> (Vec<AssistantContent>, Vec<Item<StreamEvent>>) {
    let driven = drive(frames);
    assert!(driven.errors.is_empty(), "{:?}", driven.errors);
    (driven.response.choice, driven.items)
}

pub(super) fn driven_frames(frames: &[Value]) -> (Vec<Item<StreamEvent>>, Vec<ProviderError>) {
    let driven = drive(frames);
    (driven.items, driven.errors)
}

pub(super) fn unknown_payloads(items: &[Item<StreamEvent>]) -> Vec<Value> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Unknown(payload) => Some(payload.value().clone()),
            Item::Event(_) => None,
        })
        .collect()
}

pub(super) fn folded_texts(choice: &[AssistantContent]) -> Vec<&str> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::CarriedText(text) => Some(text.text()),
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

pub(super) fn reasoning_items(choice: &[AssistantContent]) -> Vec<&CarriedReasoning> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::CarriedReasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect()
}

pub(super) fn opaque_texts(choice: &[AssistantContent]) -> Vec<&Value> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::CarriedText(text)
                if matches!(&text.content, CarriedTextContent::Opaque(_)) =>
            {
                Some(&text.payload.item)
            }
            _ => None,
        })
        .collect()
}

pub(super) fn replay_opaque_choice(choice: Vec<AssistantContent>) -> Value {
    let wire = OpenAIConfig::new("test-key").responses("different-model");
    let request = CompletionRequest::new("continue").message(Message::from(choice));
    let request =
        Completion::prepare(request, &wire.describe()).expect("actual history adaptation");
    let payload = wire
        .encode(request, Mode::Unary)
        .expect("actual same-issuer Responses encode");
    crate::test_utils::json_body(&payload.request)["input"].clone()
}

pub(super) fn opaque_message_fixture(parts: Value) -> Value {
    message_item("msg_opaque", parts)
}

/// Drive the selected real decoder through the branded writer. Genuine
/// terminal fixtures retain its result; clean EOF uses the public stream's
/// actual partial snapshot rather than fabricating an `Out::end`.
pub(super) fn folded_body_with(
    options: super::super::ResponsesStreamOptions,
    frames: &[Value],
) -> Result<CompletionResponse, ProviderError> {
    if frames.iter().any(|frame| {
        matches!(
            frame["type"].as_str(),
            Some("response.completed" | "response.incomplete" | "response.failed")
        )
    }) {
        crate::driver::feed_frames!(
            super::super::ResponsesDecoder::new("openai", options),
            super::super::document::Response::with_strict_reader(false),
            "openai",
            frames
                .iter()
                .map(|frame| crate::wire::WireFrame::Text(frame.to_string()))
        )
        .outcome
    } else {
        let mut driven = drive_eof(options, frames);
        if driven.errors.is_empty() {
            Ok(driven.response)
        } else {
            Err(driven.errors.remove(0))
        }
    }
}

fn drive_eof(options: super::super::ResponsesStreamOptions, frames: &[Value]) -> Driven {
    assert!(
        !options.immediate_tool_calls,
        "EOF fixture uses the real default OpenAI wire"
    );
    drive(frames)
}

pub(super) fn response_choice_labels(choice: &[AssistantContent]) -> Vec<&str> {
    choice
        .iter()
        .map(|part| match part {
            AssistantContent::ProviderItem(_) => "provider",
            AssistantContent::ToolCall(_) => "function",
            AssistantContent::CustomToolCall(_) => "custom",
            AssistantContent::CarriedText(text)
                if matches!(&text.content, CarriedTextContent::Refusal(_)) =>
            {
                "refusal"
            }
            AssistantContent::CarriedText(_) | AssistantContent::Text(_) => "text",
            AssistantContent::CarriedReasoning(_) | AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Image(_) => "image",
            AssistantContent::Opaque(_) => "opaque",
        })
        .collect()
}

pub(super) fn unary_choice(output: Value) -> Vec<AssistantContent> {
    let mut whole = sample_response("completed");
    whole["output"] = output;
    crate::driver::feed_frames!(
        super::super::ResponsesDecoder::new(
            "openai",
            super::super::ResponsesStreamOptions::strict()
        ),
        super::super::document::Response::with_strict_reader(false),
        "openai",
        [crate::wire::WireFrame::Text(whole.to_string())]
    )
    .outcome
    .expect("actual whole-response decoder")
    .choice
}

pub(super) async fn responses_stream<H>(http: H) -> crate::streaming::CompletionStream
where
    H: crate::driver::Transport<crate::providers::openai::responses_api::wire::Responses>,
{
    Model::new(OpenAIConfig::new("test-key").responses("gpt-5.4"), http)
        .stream(CompletionRequest::new("hello"))
        .expect("actual scripted transport stream")
}

pub(super) async fn responses_stream_of(events: &[Value]) -> crate::streaming::CompletionStream {
    let sse: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    responses_stream(MockStreamingClient {
        sse_bytes: sse.into(),
    })
    .await
}

pub(super) async fn delivered_calls_and_errors(
    events: &[Value],
) -> (
    Vec<crate::message::ToolCall>,
    Vec<crate::error::ErrorReport>,
) {
    let mut stream = responses_stream_of(events).await;
    let mut calls = Vec::new();
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            })) => calls.push(call),
            Ok(_) => {}
            Err(error) => errors.push(error.report()),
        }
    }
    (calls, errors)
}

pub(super) fn malformed_function_call_done(status: &str, sequence_number: u64) -> Value {
    json!({"type":"response.output_item.done","output_index":0,"sequence_number":sequence_number,
        "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"add",
            "arguments":"{\"x\":481","status":status}})
}

pub(super) fn argument_delta(delta: &str) -> Value {
    json!({"type":"response.function_call_arguments.delta","item_id":"fc_1",
        "output_index":0,"sequence_number":1,"delta":delta})
}

#[derive(Clone)]
pub(super) struct ScriptedFrames(pub Vec<crate::wire::WireFrame>);

impl crate::driver::Transport<crate::providers::openai::responses_api::wire::Responses>
    for ScriptedFrames
{
    fn send(
        &self,
        _payload: <crate::providers::openai::responses_api::wire::Responses as Wire>::Payload,
        _exchange: crate::driver::Exchange,
    ) -> crate::driver::Opening<crate::wire::WireFrame> {
        let frames = self.0.clone();
        crate::driver::Opening::new(async move {
            Ok(crate::driver::Opened::new(futures::stream::iter(
                frames.into_iter().map(Ok),
            )))
        })
    }
}

pub(super) async fn finish_events(
    frames: &[Value],
    incomplete: super::super::IncompleteTerminal,
) -> CompletionResponse {
    let sse: String = frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect();
    let wire = OpenAIConfig::new("test-key")
        .responses("gpt-5.4")
        .with_streamed_incomplete(incomplete);
    let mut stream = Model::new(
        wire,
        MockStreamingClient {
            sse_bytes: sse.into(),
        },
    )
    .stream(CompletionRequest::new("hello"))
    .expect("actual streaming request");
    while let Some(item) = stream.next().await {
        item.expect("the fixture reaches its genuine terminal without an error");
    }
    stream
        .finish()
        .await
        .expect("actual genuine terminal response")
}

pub(super) fn incomplete_turn_events() -> [Value; 2] {
    let mut response = sample_response("incomplete");
    response["incomplete_details"] = json!({"reason":"max_output_tokens"});
    [
        json!({"type":"response.output_text.delta","content_index":0,"delta":"partial",
            "item_id":"msg_incomplete_1","output_index":0,"sequence_number":1}),
        json!({"type":"response.incomplete","sequence_number":2,"response":response}),
    ]
}

pub(super) fn decode_repaired(frames: &[Value]) -> crate::driver::Decoded<Completion> {
    crate::driver::feed_frames!(
        super::super::ResponsesDecoder::new(
            "openai",
            super::super::ResponsesStreamOptions::strict()
        )
        .with_envelope_repair(),
        super::super::document::Response::with_strict_reader(true),
        "openai",
        frames
            .iter()
            .map(|frame| crate::wire::WireFrame::Text(frame.to_string()))
    )
}
