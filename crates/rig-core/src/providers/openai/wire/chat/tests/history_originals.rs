//! Original recorded history subjects, separated at the G240 preparation boundary.
//!
//! Direct Chat encoding keeps the original no-tools serialization assertions.
//! The real Model separately reads the unchanged recorded response to the same
//! unchanged caller request. Its prepared outbound history still relabels calls
//! and results as text; these are not end-to-end cassette matching controls.
//! G241 names the old Final/optional-terminal serde stage replaced by native
//! whole-document validation of actual stream.finish().raw.

use anyhow::Result;
use bytes::Bytes;
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::message::{AssistantContent, AssistantMessage, CallId, Message, ToolName,
    ToolResultContent, UserContent};
use crate::providers::openai;
use crate::providers::openai::wire::OpenAIConfig;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::{MockStreamingClient, RecordingHttpClient, json_body};
use crate::wire::{Framing, Mode, Wire};

use super::{recorded, recorded_json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelVariant {
    Gpt4oMini,
    Gpt41Mini,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Raw,
    Normalized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Text,
    SingleTool,
    ParallelTool,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: ModelVariant,
    surface: Surface,
    shape: Shape,
}

#[derive(Debug)]
struct Observation {
    text: String,
    saw_terminal: bool,
}
fn prompt(shape: Shape) -> &'static str {
    match shape {
        Shape::Text => "Reply with exactly the marker from the prior conversation.",
        Shape::SingleTool => "Reply with exactly the value returned by lookup_marker.",
        Shape::ParallelTool => {
            "Reply with exactly the alpha and beta tool results joined by one hyphen, in call order."
        }
    }
}

fn expected_text(shape: Shape) -> &'static str {
    match shape {
        Shape::Text => "lantern-42",
        Shape::SingleTool => "azimuth-47",
        Shape::ParallelTool => "red-blue",
    }
}
fn history(shape: Shape) -> Vec<Message> {
    match shape {
        Shape::Text => vec![
            Message::User { content: vec![UserContent::text(
                "Unicode context: café 東京. The marker is exactly: lantern-42.",
            )] },
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::text("lantern-42")])),
        ],
        Shape::SingleTool => vec![
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
                "call_history_single", ToolName::new("lookup_marker").expect("tool name"),
                json!({ "key": "harbor" }),
            )])),
            Message::User { content: vec![UserContent::tool_result(
                CallId::from_wire("call_history_single"),
                ToolName::new("lookup_marker").expect("tool name"),
                vec![ToolResultContent::text("azimuth-47")],
            )] },
        ],
        Shape::ParallelTool => vec![
            Message::Assistant(AssistantMessage::new(vec![
                AssistantContent::tool_call("call_history_alpha", ToolName::new("alpha").expect("tool name"), json!({ "slot": 1 })),
                AssistantContent::tool_call("call_history_beta", ToolName::new("beta").expect("tool name"), json!({ "slot": 2 })),
            ])),
            Message::User { content: vec![
                UserContent::tool_result(CallId::from_wire("call_history_alpha"), ToolName::new("alpha").expect("tool name"), vec![ToolResultContent::text("red")]),
                UserContent::tool_result(CallId::from_wire("call_history_beta"), ToolName::new("beta").expect("tool name"), vec![ToolResultContent::text("blue")]),
            ] },
        ],
    }
}

fn request(cell: Cell) -> CompletionRequest {
    let mut request = CompletionRequest::new(prompt(cell.shape)).max_tokens(24);
    for message in history(cell.shape) {
        request = request.message(message);
    }
    request
}

fn normalized_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The text the provider's own reply carried, read out of
/// [`crate::completion::CompletionResponse::raw`] — the value the deleted raw
/// surface returned, serialized.
fn provider_text(raw: &Value) -> Result<String> {
    let reply = serde_json::from_value::<openai::completion::CompletionResponse>(raw.clone())?;
    Ok(reply
        .choices
        .iter()
        .filter_map(|choice| match &choice.message {
            openai::completion::Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            openai::completion::AssistantContent::Text { text } => Some(text.as_str()),
            openai::completion::AssistantContent::Refusal { .. } => None,
        })
        .collect())
}
fn model_name(model: ModelVariant) -> &'static str {
    match model {
        ModelVariant::Gpt4oMini => "gpt-4o-mini",
        ModelVariant::Gpt41Mini => "gpt-4.1-mini",
    }
}
async fn run_cell(scenario: &str, cell: Cell) -> Result<()> {
    let wire = OpenAIConfig::new("sk-test").chat(model_name(cell.model));
    let request = request(cell);
    let mode = match cell.transport {
        Transport::Blocking => Mode::Unary,
        Transport::Streaming => Mode::Streaming,
    };
    // G240: this is the original request, before Model's upstream preparation.
    let encoded = wire.encode(request.clone(), mode)?;
    let actual_request = json_body(&encoded.request);
    let original_request = recorded_request(scenario);
    let response_body = recorded("then", &fixture(scenario));
    let observation = match (cell.transport, cell.surface) {
        (Transport::Blocking, Surface::Raw) => {
            let response = Model::new(wire, RecordingHttpClient::new(response_body))
                .call(request).await?;
            Observation { text: provider_text(&response.raw)?, saw_terminal: true }
        }
        (Transport::Blocking, Surface::Normalized) => {
            let response = Model::new(wire, RecordingHttpClient::new(response_body))
                .call(request).await?;
            Observation { text: normalized_text(&response.choice), saw_terminal: true }
        }
        (Transport::Streaming, surface) => {
            let model = Model::new(wire, MockStreamingClient { sse_bytes: Bytes::from(response_body) });
            let mut stream = model.stream(request)?;
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: fragment, .. }) = item? {
                    text.push_str(&fragment);
                }
            }
            let response = stream.finish().await?;
            if surface == Surface::Raw {
                // G241: existing native whole-document serde on the ACTUAL raw,
                // not the retired optional terminal shape or a Value round trip.
                let native_text = provider_text(&response.raw)?;
                assert_eq!(native_text, text, "{scenario}: native whole-document text");
            }
            Observation { text, saw_terminal: true }
        }
    };
    // Keep the original recording premises and independently test real encoding.
    // Both share the unchanged received-response and text/terminal predicates.
    assert_cell(scenario, cell, &original_request, &observation);
    assert_cell(scenario, cell, &actual_request, &observation);
    Ok(())
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

fn recorded_text(scenario: &str, transport: Transport) -> String {
    match transport {
        Transport::Blocking => {
            content_text(&recorded_response(scenario)["choices"][0]["message"]["content"])
        }
        Transport::Streaming => recorded_stream_chunks(scenario)
            .iter()
            .flat_map(|chunk| chunk["choices"].as_array().into_iter().flatten())
            .filter(|choice| choice["index"].as_u64() == Some(0))
            .map(|choice| content_text(&choice["delta"]["content"]))
            .collect(),
    }
}
fn fixture(scenario: &str) -> String {
    format!("chat_history_roundtrip_originals/{scenario}.yaml")
}

fn recorded_request(scenario: &str) -> Value {
    recorded_json("when", &fixture(scenario))
}

fn recorded_response(scenario: &str) -> Value {
    recorded_json("then", &fixture(scenario))
}

fn recorded_stream_chunks(scenario: &str) -> Vec<Value> {
    let body = recorded("then", &fixture(scenario));
    // Existing transport framing and existing JSON parser; no fixture parser.
    Framing::Sse.split(body.as_bytes()).into_iter()
        .filter(|frame| frame.as_str().trim() != "[DONE]")
        .map(|frame| serde_json::from_str(frame.as_str())
            .expect("recorded SSE frame should be JSON"))
        .collect()
}

fn assert_cell(scenario: &str, cell: Cell, request: &Value, observation: &Observation) {
    assert_eq!(
        request["model"],
        model_name(cell.model),
        "{scenario}: model"
    );
    assert_eq!(request["max_tokens"], 24, "{scenario}: output cap");
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );
    assert!(request.get("tools").is_none(), "{scenario}: history only");

    let messages = request["messages"]
        .as_array()
        .expect("recorded request messages");
    match cell.shape {
        Shape::Text => {
            assert_eq!(messages.len(), 3, "{scenario}: text history length");
            assert_eq!(messages[0]["role"], "user", "{scenario}");
            assert!(
                content_text(&messages[0]["content"]).contains("café 東京"),
                "{scenario}: Unicode history survives"
            );
            assert_eq!(messages[1]["role"], "assistant", "{scenario}");
            assert_eq!(content_text(&messages[1]["content"]), "lantern-42");
        }
        Shape::SingleTool => {
            assert_eq!(messages.len(), 3, "{scenario}: single-tool history length");
            assert_eq!(messages[0]["role"], "assistant", "{scenario}");
            let calls = messages[0]["tool_calls"]
                .as_array()
                .expect("assistant tool calls");
            assert_eq!(calls.len(), 1, "{scenario}");
            assert_eq!(calls[0]["function"]["name"], "lookup_marker", "{scenario}");
            assert_eq!(
                serde_json::from_str::<Value>(
                    calls[0]["function"]["arguments"]
                        .as_str()
                        .expect("tool arguments string")
                )
                .expect("tool arguments JSON"),
                json!({ "key": "harbor" }),
                "{scenario}"
            );
            assert_eq!(messages[1]["role"], "tool", "{scenario}");
            assert_eq!(messages[1]["tool_call_id"], calls[0]["id"], "{scenario}");
            assert_eq!(content_text(&messages[1]["content"]), "azimuth-47");
        }
        Shape::ParallelTool => {
            assert_eq!(messages.len(), 4, "{scenario}: parallel history length");
            assert_eq!(messages[0]["role"], "assistant", "{scenario}");
            let calls = messages[0]["tool_calls"]
                .as_array()
                .expect("assistant tool calls");
            assert_eq!(calls.len(), 2, "{scenario}");
            assert_eq!(calls[0]["function"]["name"], "alpha", "{scenario}");
            assert_eq!(calls[1]["function"]["name"], "beta", "{scenario}");
            for (index, (name, slot, result)) in [("alpha", 1, "red"), ("beta", 2, "blue")]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    serde_json::from_str::<Value>(
                        calls[index]["function"]["arguments"]
                            .as_str()
                            .expect("tool arguments string")
                    )
                    .expect("tool arguments JSON"),
                    json!({ "slot": slot }),
                    "{scenario}: {name} arguments"
                );
                assert_eq!(messages[index + 1]["role"], "tool", "{scenario}");
                assert_eq!(
                    messages[index + 1]["tool_call_id"],
                    calls[index]["id"],
                    "{scenario}: {name} correlation id"
                );
                assert_eq!(
                    content_text(&messages[index + 1]["content"]),
                    result,
                    "{scenario}: {name} result"
                );
            }
        }
    }
    assert_eq!(
        messages.last().expect("current prompt")["role"],
        "user",
        "{scenario}: current prompt role"
    );
    assert_eq!(
        content_text(&messages.last().expect("current prompt")["content"]),
        prompt(cell.shape),
        "{scenario}: current prompt text"
    );

    let wire_text = recorded_text(scenario, cell.transport);
    assert_eq!(
        wire_text.trim(),
        expected_text(cell.shape),
        "{scenario}: recorded provider followed the history premise"
    );

    assert!(observation.saw_terminal, "{scenario}: terminal observed");
    assert_eq!(observation.text, wire_text, "{scenario}: surface text");
}
fn cell(transport: Transport, model: ModelVariant, surface: Surface, shape: Shape) -> Cell {
    Cell {
        transport,
        model,
        surface,
        shape,
    }
}

#[tokio::test]
async fn blocking_gpt_4o_mini_raw_text() -> Result<()> {
    run_cell("blocking_gpt_4o_mini_raw_text", cell(Transport::Blocking, ModelVariant::Gpt4oMini, Surface::Raw, Shape::Text)).await
}

#[tokio::test]
async fn blocking_gpt_4o_mini_raw_single_tool() -> Result<()> {
    run_cell("blocking_gpt_4o_mini_raw_single_tool", cell(Transport::Blocking, ModelVariant::Gpt4oMini, Surface::Raw, Shape::SingleTool)).await
}

#[tokio::test]
async fn blocking_gpt_4o_mini_raw_parallel_tool() -> Result<()> {
    run_cell("blocking_gpt_4o_mini_raw_parallel_tool", cell(Transport::Blocking, ModelVariant::Gpt4oMini, Surface::Raw, Shape::ParallelTool)).await
}

#[tokio::test]
async fn blocking_gpt_4o_mini_normalized_single_tool() -> Result<()> {
    run_cell("blocking_gpt_4o_mini_normalized_single_tool", cell(Transport::Blocking, ModelVariant::Gpt4oMini, Surface::Normalized, Shape::SingleTool)).await
}

#[tokio::test]
async fn blocking_gpt_4o_mini_normalized_parallel_tool() -> Result<()> {
    run_cell("blocking_gpt_4o_mini_normalized_parallel_tool", cell(Transport::Blocking, ModelVariant::Gpt4oMini, Surface::Normalized, Shape::ParallelTool)).await
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_raw_text() -> Result<()> {
    run_cell("blocking_gpt_4_1_mini_raw_text", cell(Transport::Blocking, ModelVariant::Gpt41Mini, Surface::Raw, Shape::Text)).await
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_raw_single_tool() -> Result<()> {
    run_cell("blocking_gpt_4_1_mini_raw_single_tool", cell(Transport::Blocking, ModelVariant::Gpt41Mini, Surface::Raw, Shape::SingleTool)).await
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_raw_parallel_tool() -> Result<()> {
    run_cell("blocking_gpt_4_1_mini_raw_parallel_tool", cell(Transport::Blocking, ModelVariant::Gpt41Mini, Surface::Raw, Shape::ParallelTool)).await
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_normalized_single_tool() -> Result<()> {
    run_cell("blocking_gpt_4_1_mini_normalized_single_tool", cell(Transport::Blocking, ModelVariant::Gpt41Mini, Surface::Normalized, Shape::SingleTool)).await
}

#[tokio::test]
async fn blocking_gpt_4_1_mini_normalized_parallel_tool() -> Result<()> {
    run_cell("blocking_gpt_4_1_mini_normalized_parallel_tool", cell(Transport::Blocking, ModelVariant::Gpt41Mini, Surface::Normalized, Shape::ParallelTool)).await
}

#[tokio::test]
async fn streaming_gpt_4o_mini_raw_text() -> Result<()> {
    run_cell("streaming_gpt_4o_mini_raw_text", cell(Transport::Streaming, ModelVariant::Gpt4oMini, Surface::Raw, Shape::Text)).await
}

#[tokio::test]
async fn streaming_gpt_4o_mini_raw_single_tool() -> Result<()> {
    run_cell("streaming_gpt_4o_mini_raw_single_tool", cell(Transport::Streaming, ModelVariant::Gpt4oMini, Surface::Raw, Shape::SingleTool)).await
}

#[tokio::test]
async fn streaming_gpt_4o_mini_raw_parallel_tool() -> Result<()> {
    run_cell("streaming_gpt_4o_mini_raw_parallel_tool", cell(Transport::Streaming, ModelVariant::Gpt4oMini, Surface::Raw, Shape::ParallelTool)).await
}

#[tokio::test]
async fn streaming_gpt_4o_mini_normalized_single_tool() -> Result<()> {
    run_cell("streaming_gpt_4o_mini_normalized_single_tool", cell(Transport::Streaming, ModelVariant::Gpt4oMini, Surface::Normalized, Shape::SingleTool)).await
}

#[tokio::test]
async fn streaming_gpt_4o_mini_normalized_parallel_tool() -> Result<()> {
    run_cell("streaming_gpt_4o_mini_normalized_parallel_tool", cell(Transport::Streaming, ModelVariant::Gpt4oMini, Surface::Normalized, Shape::ParallelTool)).await
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_raw_text() -> Result<()> {
    run_cell("streaming_gpt_4_1_mini_raw_text", cell(Transport::Streaming, ModelVariant::Gpt41Mini, Surface::Raw, Shape::Text)).await
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_raw_single_tool() -> Result<()> {
    run_cell("streaming_gpt_4_1_mini_raw_single_tool", cell(Transport::Streaming, ModelVariant::Gpt41Mini, Surface::Raw, Shape::SingleTool)).await
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_raw_parallel_tool() -> Result<()> {
    run_cell("streaming_gpt_4_1_mini_raw_parallel_tool", cell(Transport::Streaming, ModelVariant::Gpt41Mini, Surface::Raw, Shape::ParallelTool)).await
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_normalized_single_tool() -> Result<()> {
    run_cell("streaming_gpt_4_1_mini_normalized_single_tool", cell(Transport::Streaming, ModelVariant::Gpt41Mini, Surface::Normalized, Shape::SingleTool)).await
}

#[tokio::test]
async fn streaming_gpt_4_1_mini_normalized_parallel_tool() -> Result<()> {
    run_cell("streaming_gpt_4_1_mini_normalized_parallel_tool", cell(Transport::Streaming, ModelVariant::Gpt41Mini, Surface::Normalized, Shape::ParallelTool)).await
}
