//! Exact request refusals for fifteen named Gemini recordings.
//! The real replay transport retains the refusal; these witnesses do not
//! establish the unavailable downstream golden traces or clock exhaustion.

use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use rig::http_client::{self, HeaderMap, HttpMiddleware, Method, Uri};
use rig::providers::gemini::{GeminiConfig, completion::GenerateContent};
use rig_agent::agent::{Agent, AgentBuilder, MultiTurnStreamItem};
use rig_agent::completion::PromptError;
use rig_core::error::{ErrorKind, ProviderError};
use rig_core::provider_response::ProviderResponseError;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_test_support::cassette_models::GeminiModels;
use serde_json::{Value, json};

use crate::cassettes::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette};
use crate::corpus_matrix::{
    OWNER, Wire, agent,
    cells::{self, Bus, Cell},
    checkpoint,
    corpus::{Program, Unhandled},
    faults, long_loop,
};

type GeminiWire = Wire<rig::Model<GenerateContent>>;
pub(super) type Attempt = Result<Result<(), PromptError>>;

/// The complete set of admitted recorded Gemini request differences.
#[derive(Clone, Copy, Debug)]
pub(super) enum NamedCase {
    CustomAtOutcome,
    CausalCompletionStreamed,
    ShapingPreambleSecondTurn,
    RetrievedToolsOne,
    OutputToolUnary,
    TwoTurnsSerial,
    ContextAndTools,
    RetrievedToolsWithStatic,
    LongStreamed,
    OutputToolStreamed,
    ToolDispatchCancelled,
    LargeResult,
    ToolErrorStreamed,
    ShapingActiveToolsNoneSecondTurn,
    CausalCompletionConcurrent,
}

impl NamedCase {
    fn scenario(self) -> &'static str {
        match self {
            Self::CustomAtOutcome => "corpus_breadth/custom_at_outcome",
            Self::CausalCompletionStreamed => "corpus_matrix/causal_completion_streamed",
            Self::ShapingPreambleSecondTurn => "corpus_matrix/shaping_preamble_second_turn",
            Self::RetrievedToolsOne => "corpus_retrieval/retrieved_tools_one",
            Self::OutputToolUnary => "corpus_breadth/output_tool_unary",
            Self::TwoTurnsSerial => {
                "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag"
            }
            Self::ContextAndTools => "corpus_retrieval/context_and_tools",
            Self::RetrievedToolsWithStatic => "corpus_retrieval/retrieved_tools_with_static",
            Self::LongStreamed => "long_loop_matrix/long_streamed",
            Self::OutputToolStreamed => "corpus_breadth/output_tool_streamed",
            Self::ToolDispatchCancelled => "corpus_breadth/tool_dispatch_cancelled",
            Self::LargeResult => "checkpoint_matrix/large_result",
            Self::ToolErrorStreamed => "corpus_faults/tool_error_streamed",
            Self::ShapingActiveToolsNoneSecondTurn => {
                "corpus_matrix/shaping_active_tools_none_second_turn"
            }
            Self::CausalCompletionConcurrent => "corpus_matrix/causal_completion_concurrent",
        }
    }

    fn refused_interaction(self) -> usize {
        match self {
            Self::RetrievedToolsOne | Self::RetrievedToolsWithStatic => 2,
            Self::ContextAndTools => 4,
            _ => 0,
        }
    }

    fn recorded_interactions(self) -> usize {
        match self {
            Self::OutputToolUnary | Self::OutputToolStreamed | Self::ToolDispatchCancelled => 1,
            Self::CustomAtOutcome
            | Self::ShapingPreambleSecondTurn
            | Self::LargeResult
            | Self::ToolErrorStreamed
            | Self::ShapingActiveToolsNoneSecondTurn => 2,
            Self::CausalCompletionStreamed
            | Self::TwoTurnsSerial
            | Self::CausalCompletionConcurrent => 3,
            Self::RetrievedToolsOne => 5,
            Self::RetrievedToolsWithStatic => 7,
            Self::ContextAndTools => 8,
            Self::LongStreamed => 11,
        }
    }

    fn declarations(self) -> &'static [usize] {
        match self {
            Self::TwoTurnsSerial | Self::RetrievedToolsWithStatic => &[0, 1],
            Self::LongStreamed => &[0, 1, 2, 3],
            _ => &[0],
        }
    }

    fn cell(self) -> Result<&'static Cell> {
        match self {
            Self::CausalCompletionStreamed => Ok(&cells::CAUSAL_COMPLETION_STREAMED),
            Self::ShapingPreambleSecondTurn => Ok(&cells::SHAPING_PREAMBLE_SECOND_TURN),
            Self::LongStreamed => Ok(&long_loop::LONG_STREAMED),
            Self::LargeResult => Ok(&checkpoint::LARGE_RESULT),
            Self::ToolErrorStreamed => Ok(&faults::TOOL_ERROR_STREAMED),
            Self::ShapingActiveToolsNoneSecondTurn => {
                Ok(&cells::SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN)
            }
            Self::CausalCompletionConcurrent => Ok(&cells::CAUSAL_COMPLETION_CONCURRENT),
            other => bail!("{other:?} has its own original builder, not a matrix cell"),
        }
    }

    fn expected_refused_body(self, mut recorded: Value) -> Result<Value> {
        let declarations = recorded
            .pointer_mut("/tools/0/functionDeclarations")
            .and_then(Value::as_array_mut)
            .context("the recorded declarations exist")?;
        ensure!(
            declarations.len() == self.declarations().len(),
            "the named declaration count changed: {declarations:?}"
        );
        for &index in self.declarations() {
            let declaration = declarations
                .get_mut(index)
                .and_then(Value::as_object_mut)
                .context("the named declaration is an object")?;
            ensure!(
                !declaration.contains_key("parameters"),
                "the recording already has parameters: {declaration:?}"
            );
            let mut schema = declaration
                .shift_remove("parametersJsonSchema")
                .context("the original JSON Schema container exists")?;
            if matches!(self, Self::LongStreamed | Self::LargeResult) {
                let schema = schema
                    .as_object_mut()
                    .context("the recorded schema is an object")?;
                ensure!(
                    schema.get("additionalProperties") == Some(&Value::Bool(false)),
                    "the named recorded root constraint is exactly false: {schema:?}"
                );
                schema.shift_remove("additionalProperties");
            }
            declaration.insert("parameters".to_owned(), schema);
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
                        "Gemini request observer mutex poisoned",
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

/// Keep the actual first stream error; an absent outcome is a failure.
pub(super) async fn stream_result(stream: &mut rig::agent::StreamingResult) -> Attempt {
    tokio::time::timeout(Duration::from_secs(180), async {
        let mut final_response = false;
        while let Some(item) = stream.next().await {
            match item {
                Err(error) => return Ok(Err(error)),
                Ok(MultiTurnStreamItem::FinalResponse(_)) => final_response = true,
                Ok(_) => {}
            }
        }
        ensure!(
            final_response,
            "the Gemini stream ended without an operation outcome"
        );
        Ok(Ok(()))
    })
    .await
    .context("the Gemini stream did not end within 180 seconds")?
}

async fn operation_result(agent: &Agent, program: &Program) -> Attempt {
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
    if program.streamed {
        let mut stream = runner.stream();
        let result = stream_result(&mut stream).await;
        drop(stream);
        result
    } else {
        Ok(runner.await.map(|_| ()))
    }
}

/// Attempt only one of the seven original matrix programs, retaining cleanup.
pub(super) async fn attempt_cell(wire: &GeminiWire, case: NamedCase) -> Attempt {
    let cell = case.cell()?;
    let program = wire.program(cell);
    ensure!(
        cell.image.is_none() && program.second_prompt.is_none() && cell.reasoning.is_none(),
        "the named refusal uses one text prompt without a reasoning settlement"
    );
    let _lease = if matches!(case, NamedCase::LongStreamed) {
        Some(long_loop::lease(cell).await)
    } else {
        None
    };
    let recorder = if cell.events {
        rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig_cassette::effect_log::EffectLogRecorder::new()
    };
    if cell.bus.declared() {
        let mut builder = AgentBuilder::new(wire.model.clone());
        if cell.bus != Bus::Own {
            builder = builder.configure_bus(cell.bus.policy());
        }
        let builder = agent::configure(builder, &program, None).record_to(recorder);
        let agent = agent::grant(builder, wire, cell, &program);
        if let Some(label) = program.late_route {
            agent.register_model(label, wire.route());
        }
        let result = operation_result(&agent, &program).await;
        drop(agent);
        result
    } else {
        let (dispatcher, registrar, mut driver) =
            rig_agent::bus::Bus::channel_with(cell.bus.policy());
        let model_key = rig_core::effect::HandlerKey::from(format!("{OWNER}/model:default"));
        driver
            .register_erased(
                model_key.clone(),
                rig_core::serve::ErasedHandler::new(rig_core::serve::adapters::ModelAdapter::new(
                    "default",
                    wire.model.clone(),
                )),
            )
            .context("the original host model key is fresh")?;
        ensure!(
            !cell.notes,
            "the named causal cells have no host note handler"
        );
        driver.record_to(recorder);
        let driver = tokio::spawn(driver);
        let result = AssertUnwindSafe(async {
            let builder =
                AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), OWNER, model_key);
            let builder = agent::configure(builder, &program, None);
            let agent = agent::grant(builder, wire, cell, &program);
            let result = operation_result(&agent, &program).await;
            drop(agent);
            result
        })
        .catch_unwind()
        .await;
        drop((dispatcher, registrar));
        let joined = driver.await;
        match result {
            Err(payload) => resume_unwind(payload),
            Ok(result) => {
                joined.context("the original causal host driver joins")?;
                result
            }
        }
    }
}

fn provider_response(error: &PromptError) -> Result<&ProviderResponseError> {
    ensure!(
        error.provider_response_status() == Some(http::StatusCode::NOT_FOUND),
        "the actual operation error is not HTTP404: {error:?}"
    );
    match error {
        PromptError::Provider(ProviderError::ProviderResponse(response)) => Ok(response),
        PromptError::Report(report) if report.kind == ErrorKind::ProviderResponse => report
            .provider_response
            .as_ref()
            .context("the report retains the actual HTTP response"),
        other => bail!("the operation did not report the real replay refusal: {other:?}"),
    }
}

/// Bind one literal scenario to its real strict request refusal.
/// Setup failures and panics remain failures after the session is finalized.
pub(super) async fn with_gemini_request_witness_cassette<F, Fut>(
    scenario: &'static str,
    case: NamedCase,
    body: F,
) -> Result<()>
where
    F: FnOnce(GeminiModels) -> Fut,
    Fut: Future<Output = Attempt>,
{
    ensure!(
        CassetteMode::current() == CassetteMode::Replay,
        "Gemini request witnesses only replay"
    );
    ensure!(
        !std::env::var("RIG_CASSETTE_SNAPSHOTS")
            .is_ok_and(|value| value.eq_ignore_ascii_case("write")),
        "Gemini request witnesses never write request snapshots"
    );
    ensure!(
        scenario == case.scenario(),
        "the named scenario changed: {scenario}, {case:?}"
    );
    let recorded = crate::cassettes::recorded_interaction_bodies("gemini", scenario);
    let paths = crate::cassettes::recorded_request_paths("gemini", scenario);
    let queries = crate::cassettes::recorded_request_query_pairs("gemini", scenario);
    ensure!(
        recorded.len() == case.recorded_interactions()
            && paths.len() == recorded.len()
            && queries.len() == recorded.len(),
        "the original recorded interaction count changed"
    );
    let observer = RequestObserver::default();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "gemini",
        CassetteSpec::new(scenario).exact_matched(),
        "https://generativelanguage.googleapis.com",
    )
    .await;
    let result = AssertUnwindSafe(async {
        let config = GeminiConfig::new(cassette.api_key("GEMINI_API_KEY"))
            .with_base_url(cassette.base_url());
        let client = GeminiModels::new(
            config,
            crate::cassettes::local_http().with_middleware(observer.clone()),
        );
        body(client).await
    })
    .catch_unwind()
    .await;
    let finished = cassette.try_finish().await;
    let outcome = match result {
        Err(payload) => resume_unwind(payload),
        Ok(result) => result?,
    };
    let error = match outcome {
        Err(error) => error,
        Ok(()) => bail!("the named incompatible Gemini request unexpectedly succeeded"),
    };
    let response = provider_response(&error)?;
    ensure!(
        response.status == Some(http::StatusCode::NOT_FOUND),
        "the actual response is not HTTP404: {response:?}"
    );
    let diagnostic: Value =
        serde_json::from_str(&response.body).context("the real matcher response is JSON")?;
    ensure!(
        diagnostic.get("message") == Some(&json!("Request did not match any route or mock"))
            && diagnostic.get("actual_method") == Some(&json!("POST"))
            && diagnostic.get("missing_required_headers") == Some(&json!([])),
        "the actual response is not the named strict matcher refusal: {diagnostic:?}"
    );
    let requests = observer
        .0
        .lock()
        .map_err(|_| anyhow::anyhow!("Gemini request observer mutex poisoned"))?
        .clone();
    let refused = case.refused_interaction();
    ensure!(
        requests.len() == refused + 1,
        "unexpected actual send count: {}",
        requests.len()
    );
    for (index, sent) in requests.iter().enumerate() {
        ensure!(
            sent.method == Method::POST && sent.uri.path() == paths[index],
            "different actual method/path at{index}: {sent:?}"
        );
        let sent_query: Vec<(String, String)> =
            url::form_urlencoded::parse(sent.uri.query().unwrap_or_default().as_bytes())
                .into_owned()
                .collect();
        ensure!(
            sent_query == queries[index],
            "different complete ordered actual query at{index}: {sent:?}"
        );
        let actual: Value =
            serde_json::from_slice(&sent.body).context("the complete actual request is JSON")?;
        let original: Value = serde_json::from_str(&recorded[index].0)
            .context("the complete recorded request is JSON")?;
        let expected = if index == refused {
            case.expected_refused_body(original.clone())?
        } else {
            original.clone()
        };
        ensure!(
            actual == expected,
            "request{index} differs beyond its exact named schema coordinates: actual={actual:?}, expected={expected:?}"
        );
        if index == refused {
            ensure!(
                actual != original,
                "the actual request must have a nonempty admitted difference"
            );
        }
    }
    let sent = requests
        .get(refused)
        .context("the actual refused request exists")?;
    ensure!(
        diagnostic.get("actual_path") == Some(&json!(paths[refused])),
        "the matcher diagnostic names a different actual path: {diagnostic:?}"
    );
    let diagnostic_query = diagnostic
        .get("actual_query")
        .and_then(Value::as_str)
        .context("the matcher retains the actual Gemini query")?;
    let diagnostic_query: Vec<(String, String)> =
        url::form_urlencoded::parse(diagnostic_query.as_bytes())
            .into_owned()
            .collect();
    ensure!(
        diagnostic_query == queries[refused],
        "the matcher diagnostic names a different complete query: {diagnostic:?}; actual URI={:?}",
        sent.uri
    );
    let candidates = diagnostic
        .get("candidates")
        .and_then(Value::as_array)
        .context("the matcher lists actual candidates")?;
    ensure!(
        candidates.len() == recorded.len(),
        "the matcher candidate count changed: {candidates:?}"
    );
    for (index, candidate) in candidates.iter().enumerate() {
        ensure!(
            candidate.get("index") == Some(&json!(index))
                && candidate.get("consumed") == Some(&json!(index < refused))
                && candidate.get("expected_method") == Some(&json!("POST"))
                && candidate.get("expected_path") == Some(&json!(paths[index])),
            "the indexed original matcher state changed: {candidate:?}"
        );
    }
    let candidate = candidates
        .get(refused)
        .context("the intended refused interaction exists")?;
    for field in [
        "method_matches",
        "path_matches",
        "query_matches",
        "headers_match",
        "required_headers_match",
    ] {
        ensure!(
            candidate.get(field) == Some(&json!(true)),
            "the refusal also changes{field}: {candidate:?}; actualheaders={:?}",
            sent.headers
        );
    }
    ensure!(
        candidate.get("body_matches") == Some(&json!(false))
            && candidate.get("shape_matches") == Some(&Value::Null),
        "the real matcher did not refuse only the exact body: {candidate:?}"
    );
    let finish_error = finished
        .err()
        .context("finalization must retain the strict refusal")?;
    let CassetteError::ReplayMismatch {
        path,
        unused_interactions,
        unexpected_requests,
    } = &finish_error
    else {
        bail!("unexpected finalizer error: {finish_error:?}");
    };
    ensure!(
        *path == crate::cassettes::cassette_path("gemini", scenario),
        "the finalizer names another fixture: {path:?}"
    );
    let unused: Vec<String> = (refused..recorded.len())
        .map(|index| format!("[{index}] POST {}", paths[index]))
        .collect();
    ensure!(
        *unused_interactions == unused,
        "different unused exchanges: {unused_interactions:?}, expected={unused:?}"
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
        "finalization did not retain exactly the actual refused request: {unexpected_requests:?}"
    );
    // ReplayMismatch precedes the engine's clock check; this is no clock proof.
    Ok(())
}
