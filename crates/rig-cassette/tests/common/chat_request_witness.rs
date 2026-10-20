//! Exact named request-mismatch witnesses for OpenAI Chat recordings.
//!
//! A provider's closed `NamedCase` enum states every [`Change`] the port
//! makes to a recorded request: the JSON Pointer and the exact values. Each
//! witness replays a named program against the real cassette engine and
//! requires every sent body to equal its recording with exactly those
//! changes applied, so any other difference fails.
//!
//! - [`with_refusal_witness`]: strict exact replay serves each request equal
//!   to its recording and refuses each changed one with its real HTTP 404;
//!   finalization retains exactly those refusals
//!   ([`CassetteError::ReplayMismatch`]). What the program does after a
//!   refusal is unavailable to the witness.
//! - [`with_concurrent_refusal_witness`]: as [`with_refusal_witness`] for a
//!   concurrent program whose requests are all refused, in whatever order
//!   and number it sends them; finalization keeps exactly the refusals the
//!   engine received.
//! - [`with_snapshot_witness`]: for changes that keep a request's coarse
//!   shape, shape-matched replay serves every request, the program runs to
//!   completion, and finalization refuses exactly the named changes against
//!   the absent request snapshot ([`CassetteError::SnapshotMismatch`]).
//!
//! Both errors precede the engine's clock check; neither witness is a clock
//! proof.

use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use rig::http_client::{self, HeaderMap, HttpMiddleware, Method, Request, StatusCode, Uri};
use rig::providers::openai::wire::{Dialect, OpenAIConfig};
use rig_agent::agent::MultiTurnStreamItem;
use rig_agent::completion::PromptError;
use rig_core::ProviderResponseError;
use rig_core::error::{ErrorKind, ProviderError};
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_test_support::cassette_models::OpenAiModels;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::cassettes::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette};

/// The engine's preview of a value in a snapshot difference: its canonical
/// JSON, cut after this many characters.
const PREVIEW_LIMIT: usize = 160;

/// One operation's outcome.
pub(crate) type Outcome = Result<(), PromptError>;

/// One member the port sends differently from its recording: a JSON Pointer
/// into the request body and the exact values.
#[derive(Clone, Debug)]
pub(crate) enum Change {
    /// The recording lacks the member; the port sends it as `sent`.
    Added { pointer: String, sent: Value },
    /// The recording holds the member as `recorded`; the port sends `sent`.
    Replaced {
        pointer: String,
        recorded: Value,
        sent: Value,
    },
}

impl Change {
    fn pointer(&self) -> &str {
        match self {
            Self::Added { pointer, .. } | Self::Replaced { pointer, .. } => pointer,
        }
    }

    /// Apply exactly this change to a recorded body.
    fn apply(&self, body: &mut Value) -> Result<()> {
        let (parent, member) = self
            .pointer()
            .rsplit_once('/')
            .context("a change pointer names a member")?;
        let member = member.replace("~1", "/").replace("~0", "~");
        let parent = body
            .pointer_mut(parent)
            .and_then(Value::as_object_mut)
            .with_context(|| format!("the parent of {} is an object", self.pointer()))?;
        match self {
            Self::Added { pointer, sent } => {
                ensure!(
                    !parent.contains_key(&member),
                    "the recording already has {pointer}: {parent:?}"
                );
                parent.insert(member, sent.clone());
            }
            Self::Replaced {
                pointer,
                recorded,
                sent,
            } => {
                let slot = parent
                    .get_mut(&member)
                    .with_context(|| format!("the recording has {pointer}"))?;
                ensure!(
                    slot == recorded,
                    "the recorded {pointer} is exactly {recorded}: {slot}"
                );
                *slot = sent.clone();
            }
        }
        Ok(())
    }

    /// The line the engine's request snapshot check reports for this change.
    fn snapshot_line(&self) -> String {
        match self {
            Self::Added { pointer, sent } => {
                format!("  {pointer}: not in the snapshot, sent {}", preview(sent))
            }
            Self::Replaced {
                pointer,
                recorded,
                sent,
            } => format!(
                "  {pointer}: snapshot {}, sent {}",
                preview(recorded),
                preview(sent)
            ),
        }
    }
}

/// `value` with every object's keys sorted, as the engine compares bodies.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                if let Some(member) = map.get(key) {
                    sorted.insert(key.clone(), canonical(member));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn preview(value: &Value) -> String {
    let text = canonical(value).to_string();
    if text.chars().count() <= PREVIEW_LIMIT {
        return text;
    }
    let mut cut: String = text.chars().take(PREVIEW_LIMIT).collect();
    cut.push_str("...");
    cut
}

/// One recorded request: its line and its complete body.
#[derive(Clone, Debug)]
struct RecordedRequest {
    method: String,
    path: String,
    body: Value,
}

/// A named case's recording: its cassette coordinates, its wire and every
/// recorded request in wire order.
pub(crate) struct Recording {
    provider: &'static str,
    scenario: &'static str,
    dialect: &'static Dialect,
    requests: Vec<RecordedRequest>,
}

impl Recording {
    /// `interactions` are the scenario's recorded `(request, response)`
    /// bodies. Every recorded request is a Chat request without a query.
    pub(crate) fn new(
        provider: &'static str,
        scenario: &'static str,
        dialect: &'static Dialect,
        interactions: Vec<(String, String)>,
    ) -> Result<Self> {
        let paths = crate::cassettes::recorded_request_paths(provider, scenario);
        let queries = crate::cassettes::recorded_request_query_pairs(provider, scenario);
        let methods = recorded_request_methods(provider, scenario)?;
        ensure!(
            paths.len() == interactions.len()
                && queries.len() == interactions.len()
                && methods.len() == interactions.len(),
            "{provider} {scenario}: the recorded request lines and bodies disagree in count"
        );
        ensure!(
            queries.iter().all(Vec::is_empty),
            "{provider} {scenario}: a recorded Chat request carries a query: {queries:?}"
        );
        let requests = interactions
            .iter()
            .zip(paths)
            .zip(methods)
            .map(|(((request, _), path), method)| {
                Ok(RecordedRequest {
                    method,
                    path,
                    body: serde_json::from_str(request)
                        .context("the complete recorded request is JSON")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            provider,
            scenario,
            dialect,
            requests,
        })
    }

    pub(crate) fn scenario(&self) -> &'static str {
        self.scenario
    }

    /// How many requests the recording holds.
    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }

    /// Recorded request `index`'s complete body.
    pub(crate) fn body(&self, index: usize) -> Result<&Value> {
        self.requests
            .get(index)
            .map(|request| &request.body)
            .with_context(|| format!("{} records request {index}", self.scenario))
    }

    fn request(&self, index: usize) -> Result<&RecordedRequest> {
        self.requests
            .get(index)
            .with_context(|| format!("{} records request {index}", self.scenario))
    }

    /// Recorded request `index` with exactly `changes` applied.
    fn changed(&self, index: usize, changes: &[Change]) -> Result<Value> {
        let mut body = self.body(index)?.clone();
        for change in changes {
            change.apply(&mut body)?;
        }
        Ok(body)
    }

    fn cassette_path(&self) -> std::path::PathBuf {
        crate::cassettes::cassette_path(self.provider, self.scenario)
    }

    fn unused(&self, index: usize) -> Result<String> {
        let request = self.request(index)?;
        Ok(format!("[{index}] {} {}", request.method, request.path))
    }
}

/// Each recorded request's method, in wire order.
fn recorded_request_methods(provider: &str, scenario: &str) -> Result<Vec<String>> {
    let path = crate::cassettes::cassette_path(provider, scenario);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("{} is readable", path.display()))?;
    serde_yaml::Deserializer::from_str(&text)
        .map(|document| {
            let document =
                serde_yaml::Value::deserialize(document).context("a cassette document")?;
            document
                .get("when")
                .and_then(|when| when.get("method"))
                .and_then(serde_yaml::Value::as_str)
                .map(str::to_owned)
                .context("every recorded request has a method")
        })
        .collect()
}

/// What strict replay does with one sent request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Replay {
    /// It equals its recording, and replay serves it.
    Served,
    /// Replay refuses it, and the operation that sent it fails with that
    /// refusal.
    Refused,
}

/// One request a named program sends, in send order: the recorded
/// interaction it stands for, what replay does with it and how it differs
/// from that recording.
#[derive(Clone, Debug)]
pub(crate) struct Expected {
    pub(crate) interaction: usize,
    pub(crate) replay: Replay,
    pub(crate) changes: Vec<Change>,
}

impl Expected {
    pub(crate) fn served(interaction: usize) -> Self {
        Self {
            interaction,
            replay: Replay::Served,
            changes: Vec::new(),
        }
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

impl RequestObserver {
    fn requests(&self) -> Result<Vec<SentRequest>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("Chat request observer mutex poisoned"))?
            .clone())
    }
}

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
                        "Chat request observer mutex poisoned",
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
pub(crate) async fn stream_result(stream: &mut rig::agent::StreamingResult) -> Result<Outcome> {
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
            "the stream ended without an operation outcome"
        );
        Ok(Ok(()))
    })
    .await
    .context("the stream did not end within 180 seconds")?
}

fn provider_response(error: &PromptError) -> Result<&ProviderResponseError> {
    ensure!(
        error.provider_response_status() == Some(http::StatusCode::NOT_FOUND),
        "the actual operation error is not HTTP 404: {error:?}"
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

fn ensure_replay_only() -> Result<()> {
    ensure!(
        CassetteMode::current() == CassetteMode::Replay,
        "request witnesses only replay"
    );
    ensure!(
        !std::env::var("RIG_CASSETTE_SNAPSHOTS")
            .is_ok_and(|value| value.eq_ignore_ascii_case("write")),
        "request witnesses never write request snapshots"
    );
    Ok(())
}

/// Start `spec`'s session, run `body` on models whose requests are
/// observed, and finalize the session before any outcome is judged. A panic
/// in `body` resumes after finalization.
async fn observed_session<T, F, Fut>(
    recording: &Recording,
    spec: CassetteSpec,
    body: F,
) -> Result<(Result<T>, Result<(), CassetteError>, Vec<SentRequest>)>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let observer = RequestObserver::default();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        recording.provider,
        spec,
        recording.dialect.base_url,
    )
    .await;
    let result = AssertUnwindSafe(async {
        let config = OpenAIConfig::with_key(
            recording.dialect,
            cassette.api_key(recording.dialect.api_key_env),
        )
        .with_base_url(cassette.base_url());
        let client = OpenAiModels::new(
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
        Ok(outcome) => outcome,
    };
    Ok((outcome, finished, observer.requests()?))
}

/// Every sent request's line and complete body against its statement.
fn ensure_sent(
    recording: &Recording,
    sent: &[SentRequest],
    expected: &[(usize, Value)],
) -> Result<()> {
    ensure!(
        sent.len() == expected.len(),
        "unexpected actual send count: {}, expected {}",
        sent.len(),
        expected.len()
    );
    for (index, (request, (interaction, body))) in sent.iter().zip(expected).enumerate() {
        let recorded = recording.request(*interaction)?;
        ensure!(
            request.method.as_str() == recorded.method
                && request.uri.path() == recorded.path
                && request.uri.query().is_none(),
            "different actual request line at {index}: {request:?}"
        );
        let actual: Value =
            serde_json::from_slice(&request.body).context("the complete actual request is JSON")?;
        ensure!(
            actual == *body,
            "request {index} differs beyond its exact named changes: actual={actual}, expected={body}"
        );
    }
    Ok(())
}

/// The real matcher's refusal JSON for a request standing for
/// `interaction`, with `consumed` the interactions served before it.
fn ensure_matcher_refusal(
    recording: &Recording,
    diagnostic: &Value,
    headers: &HeaderMap,
    interaction: usize,
    consumed: &[bool],
) -> Result<()> {
    let recorded = recording.request(interaction)?;
    ensure!(
        diagnostic.get("message") == Some(&json!("Request did not match any route or mock"))
            && diagnostic.get("actual_method") == Some(&json!(recorded.method))
            && diagnostic.get("actual_path") == Some(&json!(recorded.path))
            && diagnostic.get("actual_query") == Some(&Value::Null)
            && diagnostic.get("actual_query_params") == Some(&json!([]))
            && diagnostic.get("missing_required_headers") == Some(&json!([])),
        "the actual response is not the named strict matcher refusal: {diagnostic:?}"
    );
    let candidates = diagnostic
        .get("candidates")
        .and_then(Value::as_array)
        .context("the matcher lists actual candidates")?;
    ensure!(
        candidates.len() == recording.len(),
        "the matcher candidate count changed: {candidates:?}"
    );
    for (index, candidate) in candidates.iter().enumerate() {
        let original = recording.request(index)?;
        ensure!(
            candidate.get("index") == Some(&json!(index))
                && candidate.get("consumed") == Some(&json!(consumed.get(index) == Some(&true)))
                && candidate.get("expected_method") == Some(&json!(original.method))
                && candidate.get("expected_path") == Some(&json!(original.path)),
            "the indexed original matcher state changed: {candidate:?}"
        );
    }
    let candidate = candidates
        .get(interaction)
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
            "the refusal also changes {field}: {candidate:?}; actual headers={headers:?}"
        );
    }
    ensure!(
        candidate.get("body_matches") == Some(&json!(false))
            && candidate.get("shape_matches") == Some(&Value::Null),
        "the real matcher did not refuse only the exact body: {candidate:?}"
    );
    Ok(())
}

/// The real matcher's refusal of the request standing for `interaction`,
/// the next unplayed one with `consumed` served before it, and the engine's
/// snapshot difference it carries: exactly the named changes. Returns the
/// diagnostic finalization keeps.
fn ensure_refusal(
    recording: &Recording,
    response: &ProviderResponseError,
    request: &SentRequest,
    interaction: usize,
    consumed: &[bool],
    changes: &[Change],
) -> Result<String> {
    ensure!(
        response.status == Some(http::StatusCode::NOT_FOUND),
        "the actual response is not HTTP 404: {response:?}"
    );
    let diagnostic: Value =
        serde_json::from_str(&response.body).context("the real matcher response is JSON")?;
    ensure_matcher_refusal(
        recording,
        &diagnostic,
        &request.headers,
        interaction,
        consumed,
    )?;
    let offered = consumed
        .iter()
        .position(|played| !played)
        .context("a refused request has an unplayed interaction")?;
    ensure!(
        offered == interaction,
        "a recorded-order refusal stands for the next unplayed interaction {offered}, not {interaction}"
    );
    let lines: Vec<String> = changes.iter().map(Change::snapshot_line).collect();
    let named = format!("interaction {interaction}:\n{}", lines.join("\n"));
    ensure!(
        diagnostic.get("snapshot_diff") == Some(&json!(named)),
        "the engine's snapshot difference is not exactly the named changes: {diagnostic:?}, expected {named:?}"
    );
    Ok(format!("{}\nsnapshot difference at {named}", response.body))
}

/// Bind one named program to its real strict request refusals.
///
/// `expected` states every request the program sends, in send order. `body`
/// gets the session's models and returns each operation's outcome, in
/// order; each must be the refusal of the next [`Replay::Refused`] request.
/// Setup failures and panics remain failures after the session is
/// finalized.
pub(crate) async fn with_refusal_witness<F, Fut>(
    recording: Recording,
    expected: Vec<Expected>,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<Vec<Outcome>>>,
{
    ensure_replay_only()?;
    let mut bodies = Vec::with_capacity(expected.len());
    for request in &expected {
        ensure!(
            (request.replay == Replay::Served) == request.changes.is_empty(),
            "a served request has no change and a refused one has some: {request:?}"
        );
        bodies.push((
            request.interaction,
            recording.changed(request.interaction, &request.changes)?,
        ));
    }
    ensure!(
        expected.last().map(|request| request.replay) == Some(Replay::Refused),
        "the named program ends with a reported refusal"
    );
    let spec = CassetteSpec::new(recording.scenario()).exact_matched();
    let (outcomes, finished, sent) = observed_session(&recording, spec, body).await?;
    let outcomes = outcomes?;
    ensure_sent(&recording, &sent, &bodies)?;

    let reported = expected
        .iter()
        .filter(|request| request.replay == Replay::Refused)
        .count();
    ensure!(
        outcomes.len() == reported,
        "{} operation outcomes for {reported} reported refusals",
        outcomes.len()
    );
    let mut outcomes = outcomes.iter();
    let mut consumed = vec![false; recording.len()];
    let mut diagnostics = Vec::new();
    for (index, request) in expected.iter().enumerate() {
        match request.replay {
            Replay::Served => {
                let slot = consumed
                    .get_mut(request.interaction)
                    .context("the served interaction exists")?;
                ensure!(
                    !*slot,
                    "interaction {} is served twice",
                    request.interaction
                );
                *slot = true;
            }
            Replay::Refused => {
                let error = match outcomes.next() {
                    Some(Err(error)) => error,
                    Some(Ok(())) => {
                        bail!("the named incompatible request {index} unexpectedly succeeded")
                    }
                    None => bail!("no operation reports refused request {index}"),
                };
                let diagnostic = ensure_refusal(
                    &recording,
                    provider_response(error)?,
                    sent.get(index)
                        .context("the actual refused request exists")?,
                    request.interaction,
                    &consumed,
                    &request.changes,
                )?;
                diagnostics.push(diagnostic);
            }
        }
    }

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
        *path == recording.cassette_path(),
        "the finalizer names another fixture: {path:?}"
    );
    let unused = consumed
        .iter()
        .enumerate()
        .filter(|(_, played)| !**played)
        .map(|(index, _)| recording.unused(index))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        *unused_interactions == unused,
        "different unused exchanges: {unused_interactions:?}, expected={unused:?}"
    );
    ensure!(
        *unexpected_requests == diagnostics,
        "finalization did not retain exactly the actual refused requests: {unexpected_requests:?}"
    );
    Ok(())
}

/// Bind one named program to the request snapshot's exact refusal of its
/// shape-preserving changes.
///
/// `changes` lists every changed member as `(interaction, change)`; every
/// other recorded request is sent unchanged. Shape-matched replay serves
/// every request in recorded order, `body` runs the program's own
/// predicates, and finalization must refuse exactly those changes. Setup
/// failures and panics remain failures after the session is finalized.
pub(crate) async fn with_snapshot_witness<F, Fut>(
    recording: Recording,
    changes: Vec<(usize, Change)>,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    ensure_replay_only()?;
    ensure!(
        std::env::var("RIG_CASSETTE_SNAPSHOTS")
            .is_ok_and(|value| value.eq_ignore_ascii_case("check")),
        "a snapshot witness needs RIG_CASSETTE_SNAPSHOTS=check"
    );
    let snapshot = crate::cassettes::request_snapshot(&recording.cassette_path());
    ensure!(
        !snapshot.exists(),
        "the witness states changes from the recording, so {} must not exist",
        snapshot.display()
    );
    let mut bodies = Vec::with_capacity(recording.len());
    let mut differences = Vec::new();
    for interaction in 0..recording.len() {
        let named: Vec<Change> = changes
            .iter()
            .filter(|(index, _)| *index == interaction)
            .map(|(_, change)| change.clone())
            .collect();
        if !named.is_empty() {
            let lines: Vec<String> = named.iter().map(Change::snapshot_line).collect();
            differences.push(format!("interaction {interaction}:\n{}", lines.join("\n")));
        }
        bodies.push((interaction, recording.changed(interaction, &named)?));
    }
    ensure!(
        changes
            .iter()
            .all(|(interaction, _)| *interaction < recording.len()),
        "a named change is outside the recording: {changes:?}"
    );
    ensure!(
        !differences.is_empty(),
        "a snapshot witness names at least one change"
    );
    let spec = CassetteSpec::new(recording.scenario()).shape_matched();
    let (outcome, finished, sent) = observed_session(&recording, spec, body).await?;
    outcome?;
    ensure_sent(&recording, &sent, &bodies)?;

    let finish_error = finished
        .err()
        .context("finalization must refuse the named changes")?;
    let CassetteError::SnapshotMismatch {
        path,
        snapshot: refused_snapshot,
        differences: refused,
    } = &finish_error
    else {
        bail!("unexpected finalizer error: {finish_error:?}");
    };
    ensure!(
        *path == recording.cassette_path() && *refused_snapshot == snapshot,
        "the finalizer names another fixture: {path:?}, {refused_snapshot:?}"
    );
    ensure!(
        *refused == differences,
        "the snapshot check did not refuse exactly the named changes: {refused:?}, expected={differences:?}"
    );
    Ok(())
}

/// The engine's readable lines for how `sent` differs from `expected`, as a
/// refused request's `snapshot_diff` states them: one line per change,
/// object members in key order, an array's changed middle element by
/// element when its length is unchanged and as one splice otherwise, the
/// list cut after twenty lines.
fn engine_difference(index: usize, expected: &Value, sent: &Value) -> Option<String> {
    const LISTED: usize = 20;
    let mut lines = Vec::new();
    difference_into(String::new(), expected, sent, &mut lines);
    let total = lines.len();
    lines.truncate(LISTED);
    if total > LISTED {
        lines.push(format!("  ... and {} more difference(s)", total - LISTED));
    }
    (!lines.is_empty()).then(|| format!("interaction {index}:\n{}", lines.join("\n")))
}

fn difference_into(path: String, expected: &Value, sent: &Value, lines: &mut Vec<String>) {
    if expected == sent {
        return;
    }
    let shown = |path: &str| {
        if path.is_empty() {
            "body".to_owned()
        } else {
            path.to_owned()
        }
    };
    match (expected, sent) {
        (Value::Object(expected), Value::Object(sent)) => {
            let keys: std::collections::BTreeSet<&String> =
                expected.keys().chain(sent.keys()).collect();
            for key in keys {
                let at = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                match (expected.get(key), sent.get(key)) {
                    (Some(expected), Some(sent)) => difference_into(at, expected, sent, lines),
                    (Some(expected), None) => {
                        lines.push(format!("  {at}: snapshot {}, not sent", preview(expected)));
                    }
                    (None, Some(sent)) => {
                        lines.push(format!(
                            "  {at}: not in the snapshot, sent {}",
                            preview(sent)
                        ));
                    }
                    (None, None) => {}
                }
            }
        }
        (Value::Array(expected), Value::Array(sent)) => {
            let prefix = expected
                .iter()
                .zip(sent)
                .take_while(|(expected, sent)| expected == sent)
                .count();
            let expected_rest = expected.get(prefix..).unwrap_or_default();
            let sent_rest = sent.get(prefix..).unwrap_or_default();
            let suffix = expected_rest
                .iter()
                .rev()
                .zip(sent_rest.iter().rev())
                .take_while(|(expected, sent)| expected == sent)
                .count();
            let expected_middle = expected_rest
                .get(..expected_rest.len() - suffix)
                .unwrap_or_default();
            let sent_middle = sent_rest
                .get(..sent_rest.len() - suffix)
                .unwrap_or_default();
            if expected_middle.len() == sent_middle.len() {
                for (offset, (expected, sent)) in
                    expected_middle.iter().zip(sent_middle).enumerate()
                {
                    difference_into(format!("{path}/{}", prefix + offset), expected, sent, lines);
                }
            } else {
                lines.push(format!(
                    "  {} from item {prefix}: snapshot {}, sent {}",
                    shown(&path),
                    preview(&Value::Array(expected_middle.to_vec())),
                    preview(&Value::Array(sent_middle.to_vec()))
                ));
            }
        }
        _ => lines.push(format!(
            "  {}: snapshot {}, sent {}",
            shown(&path),
            preview(expected),
            preview(sent)
        )),
    }
}

/// How one request sent through [`Receipts`] ended.
#[derive(Clone, Debug)]
enum Returned {
    /// The transport returned a non-success response with its body.
    Refused { status: StatusCode, body: String },
    /// Anything else: a response, or another transport failure.
    Other(String),
}

/// One request the transport sent, and how it ended if its sender awaited
/// it: a sender dropped mid-flight leaves `returned` empty.
#[derive(Clone, Debug)]
struct Receipt {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
    returned: Option<Returned>,
}

/// A transport that keeps a [`Receipt`] for every request it sends.
#[derive(Clone)]
struct Receipts {
    inner: http_client::DynHttpClient,
    log: Arc<Mutex<Vec<Receipt>>>,
}

impl Receipts {
    fn poisoned() -> http_client::Error {
        http_client::Error::instance(std::io::Error::other("request receipt mutex poisoned"))
    }

    fn open(log: &Mutex<Vec<Receipt>>, request: &Request<Bytes>) -> http_client::Result<usize> {
        let mut log = log.lock().map_err(|_| Self::poisoned())?;
        log.push(Receipt {
            method: request.method().clone(),
            uri: request.uri().clone(),
            headers: request.headers().clone(),
            body: request.body().clone(),
            returned: None,
        });
        Ok(log.len() - 1)
    }

    fn close(
        log: &Mutex<Vec<Receipt>>,
        index: usize,
        returned: Returned,
    ) -> http_client::Result<()> {
        let mut log = log.lock().map_err(|_| Self::poisoned())?;
        if let Some(receipt) = log.get_mut(index) {
            receipt.returned = Some(returned);
        }
        Ok(())
    }

    fn returned<T>(result: &http_client::Result<http_client::Response<T>>) -> Returned {
        match result {
            Err(http_client::Error::InvalidStatusCodeWithDetails { status, body, .. }) => {
                Returned::Refused {
                    status: *status,
                    body: body.clone(),
                }
            }
            Err(other) => Returned::Other(other.to_string()),
            Ok(response) => Returned::Other(format!("status {}", response.status())),
        }
    }

    fn receipts(&self) -> Result<Vec<Receipt>> {
        Ok(self
            .log
            .lock()
            .map_err(|_| anyhow::anyhow!("request receipt mutex poisoned"))?
            .clone())
    }
}

impl http_client::HttpClientExt for Receipts {
    fn send<T, U>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::Response<http_client::LazyBody<U>>>>
    + rig_core::wasm_compat::WasmCompatSend
    + 'static
    where
        T: Into<Bytes>,
        T: rig_core::wasm_compat::WasmCompatSend,
        U: From<Bytes>,
        U: rig_core::wasm_compat::WasmCompatSend + 'static,
    {
        let request = request.map(Into::into);
        let this = self.clone();
        async move {
            let index = Self::open(&this.log, &request)?;
            let result = this.inner.send::<Bytes, U>(request).await;
            Self::close(&this.log, index, Self::returned(&result))?;
            result
        }
    }

    fn send_multipart<U>(
        &self,
        request: Request<http_client::MultipartForm>,
    ) -> impl Future<Output = http_client::Result<http_client::Response<http_client::LazyBody<U>>>>
    + rig_core::wasm_compat::WasmCompatSend
    + 'static
    where
        U: From<Bytes>,
        U: rig_core::wasm_compat::WasmCompatSend + 'static,
    {
        self.inner.send_multipart(request)
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>>
    + rig_core::wasm_compat::WasmCompatSend
    where
        T: Into<Bytes> + rig_core::wasm_compat::WasmCompatSend,
    {
        let request = request.map(Into::into);
        let this = self.clone();
        async move {
            let index = Self::open(&this.log, &request)?;
            let result = this.inner.send_streaming::<Bytes>(request).await;
            Self::close(&this.log, index, Self::returned(&result))?;
            result
        }
    }
}

/// Split one kept refusal into its matcher JSON and its snapshot
/// difference.
fn kept_refusal(diagnostic: &str) -> Result<(Value, &str)> {
    let (message, difference) = diagnostic
        .split_once("\nsnapshot difference at ")
        .with_context(|| format!("a kept refusal names its snapshot difference: {diagnostic}"))?;
    let message: Value =
        serde_json::from_str(message).context("a kept refusal's matcher response is JSON")?;
    ensure!(
        message.get("snapshot_diff") == Some(&json!(difference)),
        "a kept refusal's difference is not its response's: {diagnostic}"
    );
    Ok((message, difference))
}

/// Bind one named concurrent program to its real strict refusals, in any
/// order and number.
///
/// `named` lists each recorded interaction the program may send and its
/// exact changes. Strict unordered replay refuses every one, so no
/// interaction is ever served and the engine compares each refused request
/// with interaction 0. `body` runs the program and returns its outcome,
/// which must be one of those refusals.
///
/// The program drops its in-flight requests at its first failure, so which
/// of them reach the engine is not fixed. The witness requires every sent
/// body to equal exactly one named changed body; every refusal returned to
/// the program to be the real HTTP 404 of the named matcher state, kept by
/// finalization word for word; and every further kept refusal to be a real
/// refusal of a sent request whose sender was dropped, matched one to one.
/// Setup failures and panics remain failures after the session is
/// finalized.
pub(crate) async fn with_concurrent_refusal_witness<F, Fut>(
    recording: Recording,
    named: Vec<(usize, Vec<Change>)>,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<Outcome>>,
{
    ensure_replay_only()?;
    let snapshot = crate::cassettes::request_snapshot(&recording.cassette_path());
    ensure!(
        !snapshot.exists(),
        "the witness states changes from the recording, so {} must not exist",
        snapshot.display()
    );
    let mut bodies = Vec::with_capacity(named.len());
    for (interaction, changes) in &named {
        ensure!(
            !changes.is_empty(),
            "every named concurrent request has a change: {interaction}"
        );
        bodies.push((*interaction, recording.changed(*interaction, changes)?));
    }
    let offered = recording.body(0)?;
    let consumed = vec![false; recording.len()];

    let receipts = Receipts {
        inner: crate::cassettes::local_http(),
        log: Arc::default(),
    };
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        recording.provider,
        CassetteSpec::new(recording.scenario())
            .exact_matched()
            .unordered(),
        recording.dialect.base_url,
    )
    .await;
    let result = AssertUnwindSafe(async {
        let config = OpenAIConfig::with_key(
            recording.dialect,
            cassette.api_key(recording.dialect.api_key_env),
        )
        .with_base_url(cassette.base_url());
        let client = OpenAiModels::new(config, http_client::DynHttpClient::new(receipts.clone()));
        body(client).await
    })
    .catch_unwind()
    .await;
    let finished = cassette.try_finish().await;
    let outcome = match result {
        Err(payload) => resume_unwind(payload),
        Ok(outcome) => outcome?,
    };
    let sent = receipts.receipts()?;

    // Every sent request is exactly one named changed body; the engine's
    // refusal of it is a function of its body alone, since no interaction
    // is ever served.
    let mut returned = Vec::new();
    let mut dropped = Vec::new();
    for (index, request) in sent.iter().enumerate() {
        let actual: Value =
            serde_json::from_slice(&request.body).context("the complete actual request is JSON")?;
        let matches: Vec<usize> = bodies
            .iter()
            .filter(|(_, body)| *body == actual)
            .map(|(interaction, _)| *interaction)
            .collect();
        let [interaction] = matches.as_slice() else {
            bail!(
                "request {index} is not exactly one named changed body: {actual}, matching {matches:?}"
            );
        };
        let recorded = recording.request(*interaction)?;
        ensure!(
            request.method.as_str() == recorded.method
                && request.uri.path() == recorded.path
                && request.uri.query().is_none(),
            "different actual request line at {index}: {request:?}"
        );
        let difference = engine_difference(0, offered, &actual)
            .context("a named changed body differs from interaction 0")?;
        match &request.returned {
            Some(Returned::Refused { status, body }) => {
                ensure!(
                    *status == StatusCode::NOT_FOUND,
                    "request {index} was answered {status}, not refused: {body}"
                );
                let message: Value =
                    serde_json::from_str(body).context("the real matcher response is JSON")?;
                ensure_matcher_refusal(
                    &recording,
                    &message,
                    &request.headers,
                    *interaction,
                    &consumed,
                )?;
                ensure!(
                    message.get("snapshot_diff") == Some(&json!(difference)),
                    "request {index}'s snapshot difference is not exactly its named body's: {message:?}, expected {difference:?}"
                );
                returned.push((
                    body.clone(),
                    format!("{body}\nsnapshot difference at {difference}"),
                ));
            }
            Some(Returned::Other(other)) => {
                bail!("request {index} was not refused: {other}")
            }
            None => dropped.push((*interaction, &request.headers, difference)),
        }
    }
    ensure!(!returned.is_empty(), "no refusal returned to the program");
    let error = match outcome {
        Err(error) => error,
        Ok(()) => bail!("the named incompatible program unexpectedly succeeded"),
    };
    let response = provider_response(&error)?;
    ensure!(
        returned.iter().any(|(body, _)| *body == response.body),
        "the program's error is not a refusal it received: {response:?}"
    );

    let finish_error = finished
        .err()
        .context("finalization must retain the strict refusals")?;
    let CassetteError::ReplayMismatch {
        path,
        unused_interactions,
        unexpected_requests,
    } = &finish_error
    else {
        bail!("unexpected finalizer error: {finish_error:?}");
    };
    ensure!(
        *path == recording.cassette_path(),
        "the finalizer names another fixture: {path:?}"
    );
    let unused = (0..recording.len())
        .map(|index| recording.unused(index))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        *unused_interactions == unused,
        "different unused exchanges: {unused_interactions:?}, expected={unused:?}"
    );
    let mut kept: Vec<&String> = unexpected_requests.iter().collect();
    for (_, diagnostic) in &returned {
        let position = kept
            .iter()
            .position(|kept| *kept == diagnostic)
            .with_context(|| {
                format!("finalization lost a refusal the program received: {diagnostic}")
            })?;
        kept.remove(position);
    }
    for diagnostic in kept {
        let (message, difference) = kept_refusal(diagnostic)?;
        let position = dropped
            .iter()
            .position(|(_, _, expected)| expected == difference)
            .with_context(|| {
                format!("finalization kept a refusal of no dropped sent request: {diagnostic}")
            })?;
        let (interaction, headers, _) = dropped.remove(position);
        ensure_matcher_refusal(&recording, &message, headers, interaction, &consumed)?;
    }
    Ok(())
}
