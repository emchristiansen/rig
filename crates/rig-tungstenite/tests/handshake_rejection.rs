//! A refused raw websocket upgrade retains its HTTP status, headers and body.
//! Local sockets exercise the transport boundary without a provider session.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_core::http_client::{Error, NoBody, Request};
use rig_core::ws_client::{ConnectOptions, WebSocketClientExt as _};
use rig_tungstenite::TungsteniteClient;
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
    format!("ws://{address}/v1")
}

async fn refusal(url: &str) -> Error {
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    match TungsteniteClient::new().connect(request, ConnectOptions::new()).await {
        Ok(_) => panic!("the upgrade should be refused"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn a_refused_upgrade_keeps_the_status_body_and_request_id() {
    let base_url = serve_one_rejection(
        "401 Unauthorized",
        &[("x-request-id", "req_websocket_live_1")],
        REJECTION_BODY,
    )
    .await;

    let error = refusal(&base_url).await;
    let Error::InvalidStatusCodeWithDetails { status, body, headers } = error else {
        panic!("the refusal should retain its response: {error:?}");
    };
    assert_eq!(status, http::StatusCode::UNAUTHORIZED);
    assert_eq!(body, REJECTION_BODY);
    assert_eq!(headers["x-request-id"], "req_websocket_live_1");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&body).expect("JSON")["error"]["code"], "invalid_api_key");
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

    let error = refusal(&base_url).await;
    let Error::InvalidStatusCodeWithDetails { status, body, headers } = error else {
        panic!("the refusal should retain its response: {error:?}");
    };
    assert_eq!(status, http::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body, r#"{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}"#);
    assert_eq!(headers["retry-after"], "20");
    assert_eq!(headers["x-ratelimit-remaining"], "0");
    assert_eq!(headers["x-request-id"], "req_websocket_live_2");
}

/// A failure that never reached a provider — nothing listening — has no
/// response to preserve and must not pretend otherwise.
#[tokio::test]
async fn a_connection_failure_reports_no_provider_response() {
    // Bind and drop, so the port is closed and the connect is refused.
    let address = {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        listener.local_addr().expect("address")
    };

    let error = refusal(&format!("ws://{address}/v1")).await;
    assert!(matches!(error, Error::Instance(_)), "a transport failure has no provider response: {error:?}");
}
