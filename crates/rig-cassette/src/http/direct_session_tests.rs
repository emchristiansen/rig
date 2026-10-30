//! A direct-recording session end to end: the endpoint it hands out, the
//! partial snapshots it writes, the keys it reads and the ledger lines its
//! client's multipart uploads write; and what a replay session reports about
//! itself, how it pauses and what it rethrows.

use super::*;
use futures::{FutureExt, StreamExt};
use serde_json::json;
use std::panic::AssertUnwindSafe;

const UPSTREAM: &str = "https://example.invalid/v1";
const FILE_REPLY: &str = r#"{"id":"file-1","object":"file"}"#;

struct Scratch {
    dir: assert_fs::TempDir,
}

impl Scratch {
    fn new() -> Self {
        Self {
            dir: assert_fs::TempDir::new().expect("scratch directory"),
        }
    }

    fn fixture(&self) -> PathBuf {
        self.dir.path().join("fixtures").join("example.yaml")
    }

    fn attempts(&self) -> PathBuf {
        self.dir.path().join("attempts")
    }

    async fn start(
        &self,
        provider: &'static str,
        upstream: &str,
        mode: CassetteMode,
    ) -> ProviderCassette {
        ProviderCassette::try_start_with_attempts(
            RecordVia::Direct,
            provider,
            CassetteSpec::new("direct-session"),
            upstream,
            mode,
            self.fixture(),
            self.attempts(),
        )
        .await
        .expect("the session starts")
    }
}

/// Hand a direct-recording session one exchange answered with `body`.
async fn record(recorder: &DirectRecorder, body: &str) {
    recorder
        .record_http_interaction(
            DirectHttpRequest {
                method: "POST",
                uri: "https://example.invalid/v1/answer",
                headers: [("content-type", "application/json")],
                body: br#"{"input":0}"#,
            },
            DirectHttpResponse {
                status: 200,
                headers: [("content-type", "application/json")],
                body: body.as_bytes(),
            },
        )
        .await;
}

#[tokio::test]
async fn a_direct_session_serves_the_upstream_and_snapshots_only_safe_exchanges() {
    let scratch = Scratch::new();
    let cassette = scratch
        .start("example", UPSTREAM, CassetteMode::Record)
        .await;
    assert_eq!(cassette.base_url(), UPSTREAM);
    let partial = scratch.dir.path().join("partial.yaml");

    assert!(
        !cassette.checkpoint_recording(&partial).await,
        "nothing has been exchanged"
    );
    assert!(!partial.exists());

    let recorder = cassette
        .direct_recorder()
        .expect("a direct-recording session");
    record(&recorder, r#"{"answer":0}"#).await;
    assert!(cassette.checkpoint_recording(&partial).await);
    let snapshot = fs::read_to_string(&partial).expect("the snapshot is readable");
    let exchanges: Vec<(String, String, u16)> = parse_cassette_interactions(&partial, &snapshot)
        .into_iter()
        .map(|interaction| {
            (
                interaction.when.method,
                interaction.when.path,
                interaction.then.status,
            )
        })
        .collect();
    assert_eq!(
        exchanges,
        vec![("POST".to_string(), "/v1/answer".to_string(), 200)]
    );

    // Scrubbing keeps response text, so a credential echoed in a body
    // stays unsafe and the snapshot is not replaced.
    record(&recorder, r#"{"note":"Bearer leaked"}"#).await;
    assert!(!cassette.checkpoint_recording(&partial).await);
    assert_eq!(
        fs::read_to_string(&partial).expect("the snapshot is readable"),
        snapshot
    );
}

#[tokio::test]
async fn a_recording_session_reads_the_named_environment_key() {
    let scratch = Scratch::new();
    let cassette = scratch
        .start("example", UPSTREAM, CassetteMode::Record)
        .await;
    // A variable the test runner sets, so no credential is ever read.
    let (name, value) = ["CARGO_PKG_NAME", "CARGO_MANIFEST_DIR", "PATH"]
        .into_iter()
        .find_map(|name| Some((name, std::env::var(name).ok()?)))
        .expect("the test runner sets a package variable or PATH");
    assert_eq!(cassette.api_key(name), value);

    let unset = "RIG_CASSETTE_DIRECT_SESSION_TEST_KEY_NEVER_SET";
    assert_eq!(std::env::var_os(unset), None, "{unset} must stay unset");
    let payload = std::panic::catch_unwind(AssertUnwindSafe(|| cassette.api_key(unset)))
        .expect_err("an unset key panics while recording");
    assert_eq!(
        payload.downcast_ref::<String>(),
        Some(&format!(
            "{unset} must be set when RIG_PROVIDER_TEST_MODE=Record"
        ))
    );
}

#[tokio::test]
async fn the_direct_client_logs_multipart_uploads_and_passes_streams_through() {
    let app = Router::new().fallback(|body: Bytes| async move {
        drop(body);
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            FILE_REPLY,
        )
    });
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a local port");
    let address = listener.local_addr().expect("its address");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let scratch = Scratch::new();
    let cassette = scratch
        .start(
            "openai",
            &format!("http://{address}/v1"),
            CassetteMode::Record,
        )
        .await;
    let client = DirectRecordingHttpClient::new(cassette.direct_recorder());

    let upload = HttpRequest::post(format!("http://{address}/v1/files"))
        .body(MultipartForm::new().text("purpose", "assistants"))
        .expect("a multipart request");
    let response = client
        .send_multipart::<Bytes>(upload)
        .await
        .expect("the local upstream answers the upload");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .await
        .expect("the upload reply is buffered");
    assert_eq!(body, Bytes::from_static(FILE_REPLY.as_bytes()));

    let streamed = HttpRequest::get(format!("http://{address}/v1/stream"))
        .body(Bytes::new())
        .expect("a streaming request");
    let response = client
        .send_streaming(streamed)
        .await
        .expect("the local upstream answers the stream");
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body();
    let mut streamed_body = Vec::new();
    while let Some(chunk) = stream.next().await {
        streamed_body.extend_from_slice(&chunk.expect("a body chunk"));
    }
    assert_eq!(streamed_body, FILE_REPLY.as_bytes());

    assert_eq!(
        ledger::outstanding(&scratch.attempts().join(ledger::LEDGER_FILE)),
        vec![ledger::CreatedResource {
            provider: "openai".to_string(),
            scenario: "direct-session".to_string(),
            kind: ledger::ResourceKind::File,
            id: "file-1".to_string(),
            delete_url: format!("http://{address}/v1/files/file-1"),
        }],
        "the upload is in the ledger; the streamed GET created nothing"
    );
}

#[tokio::test]
async fn a_replay_session_describes_itself_pauses_at_once_and_rethrows_the_tests_panic() {
    let scratch = Scratch::new();
    let fixture = scratch.fixture();
    fs::create_dir_all(fixture.parent().expect("fixture parent")).expect("fixture directory");
    let interaction = CassetteInteraction {
        when: CassetteRequest {
            path: "/v1/answer".into(),
            method: "POST".into(),
            query_param: Vec::new(),
            header: vec![NameValue {
                name: "content-type".into(),
                value: "application/json".into(),
            }],
            body: Some(json!({ "input": 0 }).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
        then: CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: Some(json!({ "answer": 0 }).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
    };
    fs::write(&fixture, serialize_cassette_interactions(&[interaction])).expect("write fixture");
    let cassette = scratch
        .start("example", UPSTREAM, CassetteMode::Replay)
        .await;

    assert_eq!(
        format!("{cassette:?}"),
        format!("ProviderCassette {{ cassette_path: {fixture:?}, mode: Replay, .. }}")
    );
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), cassette.pause(MAX_PAUSE)).await,
        Ok(()),
        "replay never waits"
    );

    let status = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .post(format!("{}/answer", cassette.base_url()))
        .json(&json!({ "input": 0 }))
        .send()
        .await
        .expect("local replay response")
        .status();
    assert_eq!(status, StatusCode::OK);

    let panic: PanicPayload = Box::new("the test body panicked");
    let outcome = AssertUnwindSafe(cassette.finish_after_test_result::<&str>(Err(panic)))
        .catch_unwind()
        .await;
    let payload = outcome.expect_err("the test's panic resumes");
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"the test body panicked")
    );
}

#[tokio::test]
async fn a_failed_test_keeps_no_attempt_when_nothing_was_recorded_or_it_cannot_be_written() {
    let empty = Scratch::new();
    let cassette = empty.start("example", UPSTREAM, CassetteMode::Record).await;
    assert_eq!(
        cassette
            .finish_after_test_result(Ok(Err("the test's error")))
            .await,
        Err("the test's error")
    );
    assert!(
        !empty.attempts().exists(),
        "an empty recording keeps nothing"
    );

    let blocked = Scratch::new();
    fs::write(blocked.attempts(), "not a directory").expect("a file where attempts go");
    let cassette = blocked
        .start("example", UPSTREAM, CassetteMode::Record)
        .await;
    let recorder = cassette
        .direct_recorder()
        .expect("a direct-recording session");
    record(&recorder, r#"{"answer":0}"#).await;
    assert_eq!(
        cassette
            .finish_after_test_result(Ok(Err("the test's error")))
            .await,
        Err("the test's error"),
        "an attempt that cannot be written does not replace the test's error"
    );
    assert_eq!(
        fs::read_to_string(blocked.attempts()).expect("the blocking file stays"),
        "not a directory"
    );
    assert!(!blocked.fixture().exists());
}
