//! OpenAI Chat Completions tool truncation across 24 retained fixtures.
//!
//! Original 16- and 32-token partial-call predicates remain distinct from the
//! 48-token complete-call controls, across both models, transports and
//! model/agent surfaces. Partial calls are dropped and never invoked; complete
//! calls retain the exact incident summary and are invoked once.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use futures::StreamExt as _;
use rig::completion::{AssistantContent, CompletionRequest, FinishReason};
use rig::streaming::{Item, StreamEvent};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::{OpenAiCassette, with_openai_tool_truncation_cassette_result};

const PREAMBLE: &str = "Call file_report exactly once. Copy the entire user incident verbatim into the required summary argument. Do not answer in prose.";
const PROMPT: &str = "The cache warmer raced the artifact uploader, the retry storm saturated the queue, three regions were drained by hand, dashboards lagged nine minutes, and rollback took forty minutes.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Blocking,
    Streaming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Gpt4oMini,
    Gpt41Mini,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Budget {
    Low,
    Mid,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Model,
    Agent,
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
    errors: Vec<String>,
    invocations: usize,
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
        Model::Gpt4oMini => "gpt-4o-mini",
        Model::Gpt41Mini => "gpt-4.1-mini",
    }
}

fn max_tokens(budget: Budget) -> u64 {
    match budget {
        Budget::Low => 16,
        Budget::Mid => 32,
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
        .additional_params(json!({ "tool_choice": "required" }))
        .max_tokens(max_tokens(cell.budget))
}

fn calls(choice: &[AssistantContent]) -> Vec<Value> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.function.arguments_value()),
            _ => None,
        })
        .collect()
}

#[derive(Clone)]
struct FileReport {
    invocations: Arc<AtomicUsize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FileReportArgs {
    summary: String,
}

#[derive(Debug, thiserror::Error)]
#[error("file report failed")]
struct FileReportError;

impl Tool for FileReport {
    const NAME: &'static str = "file_report";
    type Error = FileReportError;
    type Args = FileReportArgs;
    type Output = String;

    fn description(&self) -> String {
        "File an incident report".to_owned()
    }
    fn parameters(&self) -> Value {
        tool_definition().parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> std::result::Result<Self::Output, Self::Error> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok(args.summary)
    }
}

async fn run_model(client: OpenAiCassette, cell: Cell) -> Observation {
    let model = client.openai.chat(model_name(cell.model));
    match cell.transport {
        // The provider-native reply and the normalized view are one call now:
        // the driver decodes the native response and hands back the
        // normalization, keeping the native value on `CompletionResponse::raw`.
        Transport::Blocking => match model.call(request(cell)).await {
            Ok(response) => Observation {
                finish_reason: response.finish_reason(),
                arguments: calls(&response.choice),
                ..Default::default()
            },
            Err(error) => Observation {
                errors: vec![error.to_string()],
                ..Default::default()
            },
        },
        Transport::Streaming => {
            let mut stream = match model.stream(request(cell)) {
                Ok(stream) => stream,
                Err(error) => {
                    return Observation {
                        errors: vec![error.to_string()],
                        ..Default::default()
                    };
                }
            };
            let mut observation = Observation::default();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(tool_call),
                        ..
                    })) => {
                        observation.arguments.push(tool_call.function.arguments_value());
                    }
                    Ok(_) => {}
                    Err(error) => observation.errors.push(error.to_string()),
                }
            }
            match stream.finish().await {
                Ok(terminal) => observation.finish_reason = terminal.finish_reason(),
                Err(error) => observation.errors.push(error.to_string()),
            }
            observation
        }
    }
}

async fn run_agent(client: OpenAiCassette, cell: Cell) -> Observation {
    let invocations = Arc::new(AtomicUsize::new(0));
    let agent = rig::AgentBuilder::new(client.chat.completion(model_name(cell.model)))
        .preamble(PREAMBLE)
        .tool(FileReport {
            invocations: Arc::clone(&invocations),
        })
        .additional_params(json!({ "tool_choice": "required" }))
        .max_tokens(max_tokens(cell.budget))
        .default_max_turns(1)
        .build();
    let mut errors = Vec::new();
    match cell.transport {
        Transport::Blocking => {
            if let Err(error) = agent.prompt(PROMPT).await {
                errors.push(error.to_string());
            }
        }
        Transport::Streaming => {
            let mut stream = agent
                .prompt(PROMPT)
                .history(Vec::<rig::completion::Message>::new())
                .max_turns(1)
                .stream();
            errors = crate::support::collect_stream_observation(&mut stream)
                .await
                .errors;
        }
    }
    Observation {
        errors,
        invocations: invocations.load(Ordering::SeqCst),
        ..Default::default()
    }
}

async fn run_cell(client: OpenAiCassette, cell: Cell, observed: SharedObservation) -> Result<()> {
    let observation = match cell.surface {
        Surface::Model => run_model(client, cell).await,
        Surface::Agent => run_agent(client, cell).await,
    };
    *observed.lock().expect("observation mutex poisoned") = Some(observation);
    Ok(())
}

fn recorded_request(scenario: &str) -> Value {
    crate::cassettes::recorded_json_request("openai", scenario)
}

fn recorded_response(scenario: &str) -> Value {
    crate::cassettes::recorded_json_response("openai", scenario)
}

fn recorded_chunks(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_sse_json_frames("openai", scenario)
}

fn recorded_finish_and_arguments(scenario: &str, transport: Transport) -> (String, Vec<String>) {
    match transport {
        Transport::Blocking => {
            let response = recorded_response(scenario);
            let finish = response["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let arguments = response["choices"][0]["message"]["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .filter_map(|call| {
                            call["function"]["arguments"].as_str().map(str::to_owned)
                        })
                        .collect()
                })
                .unwrap_or_default();
            (finish, arguments)
        }
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
    assert_eq!(
        request["tool_choice"], "required",
        "{scenario}: forced tool"
    );
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
        assert!(
            !wire_arguments[0].is_empty(),
            "{scenario}: OpenAI boundary is non-empty"
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
            assert!(
                observation.arguments.is_empty(),
                "{scenario}: partial call is dropped"
            );
        }
        Surface::Agent if complete => assert_eq!(
            observation.invocations, 1,
            "{scenario}: complete call invoked once"
        ),
        Surface::Agent => {
            assert_eq!(
                observation.invocations, 0,
                "{scenario}: partial call is never invoked"
            );
            assert!(
                observation
                    .errors
                    .iter()
                    .all(|error| !error.contains("ProviderResponseError")),
                "{scenario}: truncation reached agent loop: {:?}",
                observation.errors
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
async fn blocking_gpt4o_low_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_low_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_low_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_mid_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_mid_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_mid_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_complete_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_complete_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_complete_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_low_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_low_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_low_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_mid_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_mid_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_mid_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_complete_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_complete_model";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_complete_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_low_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_low_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_low_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_mid_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_mid_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_mid_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_complete_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_complete_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_complete_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_low_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_low_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_low_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_mid_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_mid_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_mid_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_complete_model() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_complete_model";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Model,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_complete_model",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
