//! Exact request-contract refusals for seven DeepSeek recordings.
//! These witnesses preserve the original request inputs and strict replay
//! failures; their downstream successful effect traces are unavailable.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use futures::StreamExt;
use rig::http_client::{self, HeaderMap, HttpMiddleware, Method, Uri};
use rig::providers::openai::wire::{DEEPSEEK, OpenAIConfig, OpenAiWire};
use rig_agent::agent::{Agent, AgentBuilder, MultiTurnStreamItem, PromptResponse};
use rig_agent::completion::PromptError;
use rig_core::error::{ErrorKind, ProviderError};
use rig_core::provider_response::ProviderResponseError;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::cassettes::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette};
use crate::corpus_matrix::{
    Wire, agent,
    cells::{Bus, Cell},
    corpus::{Program, Unhandled},
};

type DeepSeekWire = Wire<rig::Model<OpenAiWire>>;

/// The seven recorded DeepSeek requests with exact negative expectations.
#[derive(Clone, Copy, Debug)]
pub(super) enum NamedCase {
    ReasoningToolStreamed,
    ShapingPreambleSecondTurn,
    ServingConcurrentConcurrencyTwo,
    OutputToolChoiceSpecificOutput,
    ShapingToolChoiceNoneOnCommittedOutput,
    ParallelBatch,
    ToolErrorStreamed,
}

impl NamedCase {
    fn scenario(self) -> &'static str {
        match self {
            Self::ReasoningToolStreamed => "reasoning_matrix/tool_streamed",
            Self::ShapingPreambleSecondTurn => "corpus_matrix/shaping_preamble_second_turn",
            Self::ServingConcurrentConcurrencyTwo => {
                "corpus_matrix/serving_concurrent_concurrency_two"
            }
            Self::OutputToolChoiceSpecificOutput => {
                "corpus_matrix/output_tool_choice_specific_output"
            }
            Self::ShapingToolChoiceNoneOnCommittedOutput => {
                "corpus_matrix/shaping_tool_choice_none_on_committed_output"
            }
            Self::ParallelBatch => "checkpoint_matrix/parallel_batch",
            Self::ToolErrorStreamed => "corpus_faults/tool_error_streamed",
        }
    }

    fn cell_name(self) -> &'static str {
        match self {
            Self::ReasoningToolStreamed => "reasoning_tool_streamed",
            Self::ShapingPreambleSecondTurn => "shaping_preamble_second_turn",
            Self::ServingConcurrentConcurrencyTwo => "serving_concurrent_concurrency_two",
            Self::OutputToolChoiceSpecificOutput => "output_tool_choice_specific_output",
            Self::ShapingToolChoiceNoneOnCommittedOutput => {
                "shaping_tool_choice_none_on_committed_output"
            }
            Self::ParallelBatch => "checkpoint_parallel_batch",
            Self::ToolErrorStreamed => "fault_tool_error_streamed",
        }
    }

    fn refused_interaction(self) -> usize {
        match self {
            Self::OutputToolChoiceSpecificOutput => 0,
            Self::ReasoningToolStreamed
            | Self::ShapingPreambleSecondTurn
            | Self::ServingConcurrentConcurrencyTwo
            | Self::ShapingToolChoiceNoneOnCommittedOutput
            | Self::ParallelBatch
            | Self::ToolErrorStreamed => 1,
        }
    }

    fn recorded_interactions(self) -> usize {
        match self {
            Self::OutputToolChoiceSpecificOutput => 1,
            Self::ShapingToolChoiceNoneOnCommittedOutput => 3,
            Self::ReasoningToolStreamed
            | Self::ShapingPreambleSecondTurn
            | Self::ServingConcurrentConcurrencyTwo
            | Self::ParallelBatch
            | Self::ToolErrorStreamed => 2,
        }
    }

    fn expected_refused_body(self, mut recorded: Value) -> Result<Value> {
        match self {
            Self::OutputToolChoiceSpecificOutput => {
                let choice = recorded
                    .pointer_mut("/tool_choice")
                    .context("the recorded forced tool choice exists")?;
                ensure!(
                    *choice == json!({"function":{"name":"final_result"},"type":"function"}),
                    "the recorded forced choice changed: {choice:?}"
                );
                *choice = Value::Null;
            }
            Self::ReasoningToolStreamed
            | Self::ShapingPreambleSecondTurn
            | Self::ServingConcurrentConcurrencyTwo
            | Self::ShapingToolChoiceNoneOnCommittedOutput
            | Self::ParallelBatch
            | Self::ToolErrorStreamed => {
                if matches!(
                    self,
                    Self::ReasoningToolStreamed
                        | Self::ShapingPreambleSecondTurn
                        | Self::ServingConcurrentConcurrencyTwo
                        | Self::ParallelBatch
                ) {
                    let message = recorded
                        .pointer_mut("/messages/2")
                        .and_then(Value::as_object_mut)
                        .context("the recorded assistant message exists")?;
                    ensure!(
                        !message.contains_key("content"),
                        "recorded assistant content was absent: {message:?}"
                    );
                    message.insert("content".to_owned(), json!(""));
                }
                let call_count = match self {
                    Self::ServingConcurrentConcurrencyTwo => 2,
                    Self::ParallelBatch => 3,
                    Self::ReasoningToolStreamed
                    | Self::ShapingPreambleSecondTurn
                    | Self::ShapingToolChoiceNoneOnCommittedOutput
                    | Self::ToolErrorStreamed => 1,
                    Self::OutputToolChoiceSpecificOutput => {
                        bail!("forced-choice witness has no index additions")
                    }
                };
                let calls = recorded
                    .pointer_mut("/messages/2/tool_calls")
                    .and_then(Value::as_array_mut)
                    .context("the recorded assistant tool calls exist")?;
                ensure!(
                    calls.len() == call_count,
                    "the recorded call count changed: {calls:?}"
                );
                for call in calls {
                    let call = call
                        .as_object_mut()
                        .context("the recorded tool call is an object")?;
                    ensure!(
                        !call.contains_key("index"),
                        "the recorded call index was absent: {call:?}"
                    );
                    call.insert("index".to_owned(), json!(0));
                }
            }
        }
        Ok(recorded)
    }
}

#[derive(Clone, Debug)]
struct SentRequest {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
}

#[derive(Clone, Default)]
struct RequestObserver(Arc<Mutex<Vec<SentRequest>>>);

impl HttpMiddleware for RequestObserver {
    fn before_request_body<'a>(
        &'a self,
        method: &'a Method,
        uri: &'a Uri,
        headers: &'a HeaderMap,
        body: Bytes,
    ) -> WasmBoxedFuture<'a, http_client::Result<Bytes>> {
        Box::pin(async move {
            self.0
                .lock()
                .map_err(|_| {
                    http_client::Error::instance(std::io::Error::other(
                        "DeepSeek request observer mutex poisoned",
                    ))
                })?
                .push(SentRequest {
                    method: method.clone(),
                    uri: uri.clone(),
                    headers: headers.clone(),
                    body: body.clone(),
                });
            Ok(body)
        })
    }
}

async fn operation_result(
    agent: &Agent,
    program: &Program,
) -> Result<Result<PromptResponse, PromptError>> {
    let mut runner = agent.prompt(rig_core::message::Message::user(program.prompt));
    if let Some(history) = program.history {
        runner = runner.history(history());
    }
    if let Some(max_turns) = program.max_turns {
        runner = runner.max_turns(max_turns);
    }
    if let Some(concurrency) = program.tool_concurrency {
        runner = runner.tool_concurrency(concurrency);
    }
    runner = runner.unhandled_invalid_tool_call(match program.unhandled {
        Unhandled::Fail => rig_agent::run::UnhandledInvalidToolCall::Fail,
        Unhandled::Ignore => rig_agent::run::UnhandledInvalidToolCall::Ignore,
    });
    if !program.streamed {
        return Ok(runner.await);
    }
    let mut stream = runner.stream();
    let outcome = tokio::time::timeout(Duration::from_secs(180), async {
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            match item {
                Err(error) => return Ok(Err(error)),
                Ok(MultiTurnStreamItem::FinalResponse(response)) => final_response = Some(response),
                Ok(_) => {}
            }
        }
        let response =
            final_response.context("the DeepSeek stream ended without an operation outcome")?;
        Ok::<_, anyhow::Error>(Ok(response))
    })
    .await;
    drop(stream);
    outcome.context("the DeepSeek stream did not end within 180 seconds")?
}

fn provider_response(error: &PromptError) -> Result<&ProviderResponseError> {
    match error {
        PromptError::Provider(ProviderError::ProviderResponse(response)) => Ok(response),
        PromptError::Report(report) if report.kind == ErrorKind::ProviderResponse => {
            ensure!(
                report.http_status == Some(404),
                "the actual operation report is not HTTP 404: {report:?}"
            );
            report
                .provider_response
                .as_ref()
                .context("the operation report preserves the real HTTP response")
        }
        other => bail!("the operation did not fail at the strict replay matcher: {other:?}"),
    }
}

/// Replay one named refusal using its original wire and cell.
/// Any changed request, operation error or replay consumption fails the witness.
pub(super) async fn with_deepseek_request_mismatch(
    scenario: &'static str,
    case: NamedCase,
    make_wire: fn(&OpenAiModels) -> DeepSeekWire,
    cell: &Cell,
) -> Result<()> {
    ensure!(
        CassetteMode::current() == CassetteMode::Replay,
        "request-mismatch witnesses only replay"
    );
    ensure!(
        !std::env::var("RIG_CASSETTE_SNAPSHOTS")
            .is_ok_and(|value| value.eq_ignore_ascii_case("write")),
        "request-mismatch witnesses never write request snapshots"
    );
    ensure!(
        scenario == case.scenario(),
        "the named scenario changed: {scenario}, {case:?}"
    );
    ensure!(
        cell.name == case.cell_name(),
        "the named cell changed: {} / {case:?}",
        cell.name
    );
    ensure!(
        cell.bus == Bus::Own && cell.image.is_none() && cell.program.second_prompt.is_none(),
        "the seven named programs use their own bus and one text prompt"
    );
    let recorded = crate::cassettes::recorded_interaction_bodies("deepseek", scenario);
    ensure!(
        recorded.len() == case.recorded_interactions(),
        "the recorded exchange count changed: {}",
        recorded.len()
    );
    let observer = RequestObserver::default();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "deepseek",
        CassetteSpec::new(scenario).exact_matched(),
        "https://api.deepseek.com",
    )
    .await;
    let client = OpenAiModels::new(
        OpenAIConfig::with_key(&DEEPSEEK, cassette.api_key("DEEPSEEK_API_KEY"))
            .with_base_url(cassette.base_url()),
        rig_test_support::cassettes::local_http().with_middleware(observer.clone()),
    );
    let wire = make_wire(&client);
    let program = wire.program(cell);
    let recorder = if cell.events {
        rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig_cassette::effect_log::EffectLogRecorder::new()
    };
    let builder =
        agent::configure(AgentBuilder::new(wire.model.clone()), &program, None).record_to(recorder);
    let agent = agent::grant(builder, &wire, cell, &program);
    let outcome = operation_result(&agent, &program).await;
    drop(agent);
    let finished = cassette.try_finish().await;

    // Finalization precedes every outcome assertion, including unexpected success.
    let requests = observer
        .0
        .lock()
        .map_err(|_| anyhow::anyhow!("DeepSeek request observer mutex poisoned"))?
        .clone();
    let error = match outcome? {
        Err(error) => error,
        Ok(response) => {
            bail!("the named incompatible request unexpectedly succeeded: {response:?}")
        }
    };
    let response = provider_response(&error)?;
    ensure!(
        response.status == Some(http::StatusCode::NOT_FOUND),
        "the actual response is not HTTP 404: {response:?}"
    );
    let diagnostic: Value =
        serde_json::from_str(&response.body).context("the real replay response is JSON")?;
    ensure!(
        diagnostic.get("message") == Some(&json!("Request did not match any route or mock")),
        "unexpected matcher response: {diagnostic:?}"
    );
    ensure!(
        diagnostic.get("actual_method") == Some(&json!("POST"))
            && diagnostic.get("actual_path") == Some(&json!("/chat/completions"))
            && diagnostic.get("actual_query") == Some(&Value::Null)
            && diagnostic.get("actual_query_params") == Some(&json!([]))
            && diagnostic.get("missing_required_headers") == Some(&json!([])),
        "the matcher refused different request attributes: {diagnostic:?}"
    );
    let refused = case.refused_interaction();
    ensure!(
        requests.len() == refused + 1,
        "unexpected actual request count: {}",
        requests.len()
    );
    for (index, sent) in requests.iter().enumerate() {
        ensure!(
            sent.method == Method::POST
                && sent.uri.path() == "/chat/completions"
                && sent.uri.query().is_none(),
            "unexpected actual request attributes at {index}: {sent:?}"
        );
        let body: Value =
            serde_json::from_slice(&sent.body).context("the complete actual request is JSON")?;
        let recorded_body = &recorded
            .get(index)
            .context("the original exchange exists")?
            .0;
        let original: Value =
            serde_json::from_str(recorded_body).context("the complete recorded request is JSON")?;
        let expected = if index == refused {
            case.expected_refused_body(original.clone())?
        } else {
            original.clone()
        };
        ensure!(
            body == expected,
            "request {index} has differences beyond its named contract: actual={body:?}, expected={expected:?}"
        );
        if index == refused {
            ensure!(
                body != original,
                "the refused request must have its named difference"
            );
        }
    }
    let candidates = diagnostic
        .get("candidates")
        .and_then(Value::as_array)
        .context("the real matcher lists its candidates")?;
    ensure!(
        candidates.len() == case.recorded_interactions(),
        "the matcher candidate count changed: {candidates:?}"
    );
    for (index, candidate) in candidates.iter().enumerate() {
        ensure!(
            candidate.get("index") == Some(&json!(index))
                && candidate.get("consumed") == Some(&json!(index < refused))
                && candidate.get("expected_method") == Some(&json!("POST"))
                && candidate.get("expected_path") == Some(&json!("/chat/completions")),
            "the original matcher interaction state changed: {candidate:?}"
        );
    }
    let candidate = candidates
        .get(refused)
        .context("the original refused interaction exists")?;
    for field in [
        "method_matches",
        "path_matches",
        "query_matches",
        "headers_match",
        "required_headers_match",
    ] {
        ensure!(
            candidate.get(field) == Some(&json!(true)),
            "the refusal also changed {field}: {candidate:?}; actual headers={:?}",
            requests
                .get(refused)
                .context("the actual refused request exists")?
                .headers
        );
    }
    ensure!(
        candidate.get("body_matches") == Some(&json!(false))
            && candidate.get("shape_matches") == Some(&Value::Null),
        "the real matcher did not refuse only the exact body: {candidate:?}"
    );
    let finish_error = finished
        .err()
        .context("strict replay finalization must retain the refused request")?;
    let CassetteError::ReplayMismatch {
        path,
        unused_interactions,
        unexpected_requests,
    } = &finish_error
    else {
        bail!("unexpected finalizer error: {finish_error:?}");
    };
    ensure!(
        *path == crate::cassettes::cassette_path("deepseek", scenario),
        "the finalizer reports a different fixture: {path:?}"
    );
    let unused: Vec<String> = (refused..case.recorded_interactions())
        .map(|index| format!("[{index}] POST /chat/completions"))
        .collect();
    ensure!(
        *unused_interactions == unused,
        "unexpected unconsumed interactions: {unused_interactions:?}, expected={unused:?}"
    );
    let expected_diagnostic = match diagnostic.get("snapshot_diff") {
        None => response.body.clone(),
        Some(Value::String(difference)) => {
            format!("{}\nsnapshot difference at {difference}", response.body)
        }
        Some(other) => bail!("the matcher snapshot difference is not text: {other:?}"),
    };
    ensure!(
        unexpected_requests.as_slice() == [expected_diagnostic],
        "the finalizer did not retain exactly the real refused request: {unexpected_requests:?}"
    );
    Ok(())
}
