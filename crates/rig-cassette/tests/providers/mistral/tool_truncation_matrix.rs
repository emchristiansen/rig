//! Live Chat Completions matrix for the outer-reason truncation contract.
//!
//! A live budget sweep on 2026-08-16 found the same deterministic boundary on
//! `mistral-small-latest` and `ministral-3b-latest`: cap 4 opens a call with an
//! empty argument string and ends `length`, cap 16 ends `length` mid-string,
//! and cap 48 carries the complete object and `tool_calls`. Caps 1 and 2 end
//! before any call exists; those equivalent no-call states were pruned because
//! they do not enter the changed deserializer or streaming tool assembler.
//!
//! The space is 2 transports × 2 models × 3 budgets × 2 public surfaces = 24
//! cells. The one test is the streaming model cell at cap 4 on
//! `ministral-3b-latest`: it drives the normalized stream and asserts its
//! request, finish reason, and accumulated argument bytes from the fixture.
//!
//! Upstream kept a cut call with what its arguments state. The port instead
//! discards a Chat call cut at an explicit `length` finish
//! (coord-a-rig-port-discard-15): the stream yields no canonical call for it,
//! closes its Part with the typed `Discard` reason `Length`, and leaves the
//! argument fragments it delivered visible as `Arguments` events.
//!
//! | dimension | values |
//! |---|---|
//! | transport | blocking, streaming |
//! | model | `mistral-small-latest`, `ministral-3b-latest` |
//! | output cap | 4 empty/incomplete, 16 partial, 48 complete |
//! | surface | model, one-turn agent |
//!
//! Each test maps to
//! `crates/rig-cassette/fixtures/cassettes/mistral/tool_truncation_matrix/<test-name>.yaml`.
//! Compound impossible shapes remain unit tests.
//!
//! | recorded cells | exact fixture set |
//! |---|---|
//! | 1 of 24 | `crates/rig-cassette/fixtures/cassettes/mistral/tool_truncation_matrix/streaming_ministral_3b_low_model.yaml` |

use rig::streaming::{DiscardReason, Item, PartKind};
use rig_test_support::cassette_models::OpenAiModels;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{AssistantContent, FinishReason};
use rig::streaming::StreamEvent;
use serde_json::{Value, json};

use super::support::with_mistral_tool_truncation_cassette_result;
use rig::completion::CompletionRequest;

const PREAMBLE: &str = "Call file_report exactly once. Copy the entire user incident verbatim into the required summary argument. Do not answer in prose.";
const PROMPT: &str = "The cache warmer raced the artifact uploader, the retry storm saturated the queue, three regions were drained by hand, dashboards lagged nine minutes, and rollback took forty minutes.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Ministral3b,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Budget {
    Low,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Model,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    transport: Transport,
    model: Model,
    budget: Budget,
    surface: Surface,
}

#[derive(Debug, Default)]
struct Observation {
    finish_reason: Option<FinishReason>,
    arguments: Vec<Value>,
    /// Each call the stream closed with a typed `Discard`: the tool its Part
    /// started under, the reason, and the argument fragments it delivered.
    discarded: Vec<(Option<String>, DiscardReason, String)>,
    errors: Vec<String>,
}

type SharedObservation = Arc<Mutex<Option<Observation>>>;

fn cell(transport: Transport, model: Model, budget: Budget, surface: Surface) -> Cell {
    Cell {
        transport,
        model,
        budget,
        surface,
    }
}

fn model_name(model: Model) -> &'static str {
    match model {
        Model::Ministral3b => "ministral-3b-latest",
    }
}

fn max_tokens(budget: Budget) -> u64 {
    match budget {
        Budget::Low => 4,
        Budget::Complete => 48,
    }
}

fn tool_definition() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: rig_core::message::ToolName::new("file_report").expect("tool name"),
        description: "File an incident report".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "summary": { "type": "string" } },
            "required": ["summary"]
        }),
    }
}

fn request(cell: Cell) -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT)
        .preamble(PREAMBLE.to_owned())
        .tool(tool_definition())
        .additional_params(json!({ "tool_choice": "any" }))
        .max_tokens(max_tokens(cell.budget))
}

async fn run_model(client: OpenAiModels, cell: Cell) -> Observation {
    let model = client.completion(model_name(cell.model));
    match cell.transport {
        Transport::Streaming => {
            let raw = match model.stream(request(cell)) {
                Ok(raw) => raw,
                Err(error) => {
                    return Observation {
                        errors: vec![error.to_string()],
                        ..Default::default()
                    };
                }
            };
            let mut stream = raw;
            let mut observation = Observation::default();
            let mut names = std::collections::HashMap::new();
            let mut fragments: std::collections::HashMap<_, String> =
                std::collections::HashMap::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(tool_call),
                        ..
                    })) => {
                        observation
                            .arguments
                            .push(tool_call.function.arguments_value());
                    }
                    Ok(Item::Event(StreamEvent::Start {
                        part,
                        kind: PartKind::ToolCall,
                        name,
                        ..
                    })) => {
                        names.insert(part, name.map(|name| name.as_str().to_owned()));
                    }
                    Ok(Item::Event(StreamEvent::Arguments { part, json })) => {
                        fragments.entry(part).or_default().push_str(&json);
                    }
                    Ok(Item::Event(StreamEvent::Discard { part, reason })) => {
                        observation.discarded.push((
                            names.get(&part).cloned().flatten(),
                            reason,
                            fragments.remove(&part).unwrap_or_default(),
                        ));
                    }
                    Ok(_) => {}
                    Err(error) => observation.errors.push(error.to_string()),
                }
            }
            if let Ok(terminal) = stream.finish().await {
                observation.finish_reason = terminal.finish_reason();
            }
            observation
        }
    }
}

async fn run_cell(client: OpenAiModels, cell: Cell, observed: SharedObservation) -> Result<()> {
    let observation = match cell.surface {
        Surface::Model => run_model(client, cell).await,
    };
    *observed.lock().expect("observation mutex poisoned") = Some(observation);
    Ok(())
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("mistral", scenario)
}

fn recorded_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("mistral", scenario)
}

fn recorded_finish_and_arguments(scenario: &str, transport: Transport) -> (String, Vec<String>) {
    match transport {
        Transport::Streaming => {
            let mut finish = String::new();
            let mut arguments = Vec::<String>::new();
            for chunk in recorded_chunks(scenario) {
                for choice in chunk["choices"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|choice| choice["index"].as_u64() == Some(0))
                {
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        finish = reason.to_owned();
                    }
                    for call in choice["delta"]["tool_calls"]
                        .as_array()
                        .into_iter()
                        .flatten()
                    {
                        let index = call["index"].as_u64().unwrap_or(0) as usize;
                        if arguments.len() <= index {
                            arguments.resize(index + 1, String::new());
                        }
                        if let Some(fragment) = call["function"]["arguments"].as_str() {
                            arguments[index].push_str(fragment);
                        }
                    }
                }
            }
            (finish, arguments)
        }
    }
}

fn assert_cell(scenario: &str, cell: Cell, observed: SharedObservation) {
    let request = recorded_request(scenario);
    assert_eq!(
        request["model"],
        model_name(cell.model),
        "{scenario}: model"
    );
    assert_eq!(
        request["max_tokens"],
        max_tokens(cell.budget),
        "{scenario}: cap"
    );
    assert!(
        request.get("max_completion_tokens").is_none(),
        "{scenario}: these legacy Chat Completions models use max_tokens"
    );
    assert_eq!(request["tool_choice"], "any", "{scenario}: forced tool");
    assert_eq!(
        request["tools"].as_array().map(Vec::len),
        Some(1),
        "{scenario}: tool schema"
    );
    assert_eq!(
        request["stream"].as_bool().unwrap_or(false),
        cell.transport == Transport::Streaming,
        "{scenario}: transport"
    );

    let (finish, wire_arguments) = recorded_finish_and_arguments(scenario, cell.transport);
    assert_eq!(wire_arguments.len(), 1, "{scenario}: exactly one wire call");
    let complete = cell.budget == Budget::Complete;
    if complete {
        assert_eq!(finish, "tool_calls", "{scenario}: completion control");
        let value: Value =
            serde_json::from_str(&wire_arguments[0]).expect("complete wire arguments");
        assert_eq!(
            value["summary"], PROMPT,
            "{scenario}: exact complete arguments"
        );
    } else {
        assert_eq!(finish, "length", "{scenario}: truncation premise");
        assert_eq!(
            wire_arguments[0].is_empty(),
            cell.budget == Budget::Low,
            "{scenario}: empty boundary premise"
        );
        assert!(
            serde_json::from_str::<Value>(&wire_arguments[0]).is_err(),
            "{scenario}: partial JSON premise"
        );
    }

    let observation = observed
        .lock()
        .expect("observation mutex poisoned")
        .take()
        .expect("cell observation");
    match cell.surface {
        Surface::Model if complete => {
            assert!(
                observation.errors.is_empty(),
                "{scenario}: {:?}",
                observation.errors
            );
            assert_eq!(
                observation.finish_reason,
                Some(FinishReason::ToolCalls),
                "{scenario}"
            );
            assert_eq!(observation.arguments.len(), 1, "{scenario}");
            assert_eq!(observation.arguments[0]["summary"], PROMPT, "{scenario}");
            assert!(
                observation.discarded.is_empty(),
                "{scenario}: a complete call is never discarded: {:?}",
                observation.discarded
            );
        }
        Surface::Model => {
            assert!(
                observation.errors.is_empty(),
                "{scenario}: truncated turn survives: {:?}",
                observation.errors
            );
            assert_eq!(
                observation.finish_reason,
                Some(FinishReason::Length),
                "{scenario}"
            );
            // A call cut at Length is discarded: no canonical call, the
            // typed reason Length, and its delivered fragments visible.
            assert!(
                observation.arguments.is_empty(),
                "{scenario}: the cut call yields no canonical call: {:?}",
                observation.arguments
            );
            assert_eq!(
                observation.discarded,
                vec![(
                    Some("file_report".to_owned()),
                    DiscardReason::Length,
                    wire_arguments[0].clone(),
                )],
                "{scenario}: the cut call is discarded with reason Length, its fragments visible"
            );
        }
    }
}

async fn execute(scenario: &'static str, cell: Cell, observed: SharedObservation) {
    assert_cell(scenario, cell, observed);
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

#[tokio::test]
async fn streaming_ministral_3b_low_model() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_ministral_3b_low_model";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_ministral_3b_low_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
