//! What a *refused* upgrade reaches the caller as, end to end.
//!
//! A provider that rejects the handshake never opens a websocket: it answers
//! with an ordinary HTTP response carrying a status, its own error envelope, an
//! `x-request-id`, and : on a `429` : the rate-limit headers a caller needs to
//! back off. That information crosses two boundaries on its way out: the
//! backend turns `tungstenite::Error::Http` into
//! `http_client::Error::non_success_with_details`, and the Codex session
//! turns that into a `ProviderError` with its own request-id header read back
//! off it.
//!
//! Both halves have unit tests. This asserts the seam between them, against a
//! real socket, because the regression it guards (rig#2314, rig#2315, rig#2210)
//! is exactly the kind that survives two passing unit tests.

#![cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_codex::responses_api::codex_identity::CodexIdentity;
use rig_codex::responses_api::websocket::codex::{
    CodexWebSocketSession, CodexWebSocketSessionBuilder,
};
use rig_codex::responses_api::wire::CodexResponses;
use rig_codex::tungstenite::TungsteniteClient;
use rig_codex::wire::CallerIdentity;
use rig_core::error::ProviderError;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The live shape of an invalid-key refusal.
const REJECTION_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***-key.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#;

/// Refuse one upgrade with `status` and the given headers and body.
async fn serve_one_rejection(
    status: &'static str,
    headers: &'static [(&str, &str)],
    body: &str,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let body = body.to_string();
    tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let mut buffer = [0u8; 4096];
        let _ = stream.read(&mut buffer).await;

        let mut response = format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\n", body.len());
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str("connection: close\r\n\r\n");
        response.push_str(&body);
        let _ = stream.write_all(response.as_bytes()).await;
        let _ = stream.flush().await;
    });
    format!("http://{address}/v1")
}

/// A session is not `Debug` (it owns a live connection), so unwrap the
/// refusal by hand.
fn expect_refusal<T>(result: Result<T, ProviderError>, context: &str) -> ProviderError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

/// Open a Codex session against the gateway at `base_url`.
async fn connect(base_url: &str) -> Result<CodexWebSocketSession, ProviderError> {
    let wire = CodexResponses::new(Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "sk-invalid-key").with_base_url(base_url),
        chatgpt::GPT_5_3_CODEX,
    ))
    .expect("the ChatGPT dialect speaks the Codex contract")
    .with_caller_identity(
        CallerIdentity::new("rig-codex-tests", "rig-codex-tests/0.0 (test)", None)
            .expect("a valid caller identity"),
    )
    .with_codex_identity(CodexIdentity::from_ids("session-1", "thread-1").expect("valid ids"))
    .expect("the ChatGPT dialect speaks the Codex contract");
    CodexWebSocketSessionBuilder::new(wire)
        .expect("a Codex wire")
        .connect_with(&TungsteniteClient::new())
        .await
}

#[tokio::test]
async fn a_refused_upgrade_keeps_the_status_body_and_request_id() {
    let base_url = serve_one_rejection(
        "401 Unauthorized",
        &[("x-request-id", "req_websocket_live_1")],
        REJECTION_BODY,
    )
    .await;

    let error = expect_refusal(connect(&base_url).await, "an invalid key should be refused");

    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED),
        "the refusal's status must survive, got {error}"
    );
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
    assert_eq!(error.provider_request_id(), Some("req_websocket_live_1"));
    assert_eq!(
        error
            .provider_response_json()
            .expect("the body should be valid JSON")
            .expect("the body should be present")["error"]["code"],
        "invalid_api_key",
        "the provider's own diagnosis must reach the caller"
    );
}

/// rig#2210: a rate-limited upgrade carries the backoff metadata its HTTP twin
/// does.
#[tokio::test]
async fn a_rate_limited_upgrade_keeps_its_backoff_headers() {
    let base_url = serve_one_rejection(
        "429 Too Many Requests",
        &[
            ("x-request-id", "req_websocket_live_2"),
            ("retry-after", "20"),
            ("x-ratelimit-remaining", "0"),
        ],
        r#"{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}"#,
    )
    .await;

    let error = expect_refusal(
        connect(&base_url).await,
        "a rate-limited upgrade should be refused",
    );

    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    let headers = error
        .provider_response_headers()
        .expect("the refusal's headers should be preserved");
    assert_eq!(
        headers.get("retry-after").and_then(|v| v.to_str().ok()),
        Some("20")
    );
    assert_eq!(
        headers
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0")
    );
    assert_eq!(error.provider_request_id(), Some("req_websocket_live_2"));
}

/// A failure that never reached a provider : nothing listening : has no
/// response to preserve and must not pretend otherwise.
#[tokio::test]
async fn a_connection_failure_reports_no_provider_response() {
    // Bind and drop, so the port is closed and the connect is refused.
    let address = {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        listener.local_addr().expect("address")
    };

    let error = expect_refusal(
        connect(&format!("http://{address}/v1")).await,
        "a closed port should fail to connect",
    );

    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_request_id(), None);
}
