//! Idle keepalive and the Codex handshake, against a real local socket, over
//! both of the bundled backend's connections.
//!
//! An idle session is serviced through the transport's ready-read capability.
//! The two connections implement it very differently — the direct one polls
//! the caller's socket once, the forwarded one asks its actor on the fallback
//! runtime — and a drain that polled `recv()` once and gave up would see
//! nothing at all on the forwarded one, because its answer always needs a
//! round trip. So the same drain is driven over both here: inside a tokio
//! runtime (direct) and from `futures::executor` with no runtime (forwarded).
//!
//! The loopback server is in-process; nothing here reaches a network service.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use futures::{SinkExt, StreamExt};
use rig_core::completion::CompletionModel as _;
use rig_core::driver::Bound;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAI;
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketSession;
use rig_core::providers::openai::responses_api::websocket::codex::{
    CodexWebSocketSessionBuilder, UnrecognizedEvent,
};
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::test_utils::RecordingHttpClient;
use rig_tungstenite::{DefaultWebSocketBuilder as _, DefaultWebSocketClient as _};
use std::sync::mpsc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

fn completed(response_id: &str) -> String {
    serde_json::json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "usage": null,
            "output": [],
            "tools": []
        }
    })
    .to_string()
}

fn done(response_id: &str) -> String {
    serde_json::json!({
        "type": "response.done",
        "response": { "id": response_id, "status": "completed" },
    })
    .to_string()
}

/// Serve two turns on a thread of its own. After the first turn it sends, in
/// order, an unmodelled event, the trailing `done` and a ping, then waits for
/// the pong and reports it. The second turn proves the session is still
/// healthy and still chains `resp_1`.
///
/// On the direct connection nothing reads the socket but the session, so only
/// an idle drain can answer that ping. The forwarded connection's actor reads
/// eagerly and answers pings on its own; what the drain proves there is that
/// it reaches the frames the actor is holding.
fn serve_idle_window() -> (String, mpsc::Receiver<()>) {
    let (address_tx, address_rx) = mpsc::channel();
    let (pong_tx, pong_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime should build");
        runtime.block_on(async move {
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address should send");

                let (stream, _) = listener.accept().await.expect("accept");
                let mut socket = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");

                let _first = socket.next().await.expect("first request").expect("valid");
                socket
                    .send(Message::text(completed("resp_1")))
                    .await
                    .expect("completed should send");
                socket
                    .send(Message::text(
                        serde_json::json!({ "type": "codex.rate_limits", "seen": "idle" })
                            .to_string(),
                    ))
                    .await
                    .expect("unknown event should send");
                socket
                    .send(Message::text(done("resp_1")))
                    .await
                    .expect("done should send");
                socket
                    .send(Message::Ping(b"idle".to_vec().into()))
                    .await
                    .expect("ping should send");

                match socket.next().await.expect("the client should answer") {
                    Ok(Message::Pong(payload)) => {
                        assert_eq!(&payload[..], b"idle", "the pong echoes the ping");
                    }
                    Ok(other) => panic!("expected the idle pong first, got {other:?}"),
                    Err(error) => panic!("socket error while idle: {error}"),
                }
                pong_tx.send(()).expect("pong report should send");

                let second = socket
                    .next()
                    .await
                    .expect("second request")
                    .expect("valid")
                    .into_text()
                    .expect("text");
                assert!(
                    second.contains("\"previous_response_id\":\"resp_1\""),
                    "the idle drain must not disturb the tip, got {second}"
                );
                socket
                    .send(Message::text(completed("resp_2")))
                    .await
                    .expect("second completed should send");

                while let Some(Ok(message)) = socket.next().await {
                    if message.is_close() {
                        break;
                    }
                }
            })
            .await;
        });
    });

    let address = address_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("server should report its address");
    (format!("http://{address}/v1"), pong_rx)
}

fn bound(base_url: &str) -> Bound<Responses, RecordingHttpClient> {
    let wire = OpenAI::new("test-key")
        .with_base_url(base_url)
        .responses("gpt-5.4");
    Bound::new(wire, RecordingHttpClient::new("{}"))
}

/// Drain the idle session until the server has seen its pong, then once more,
/// collecting what every drain recovered.
///
/// Frames arrive asynchronously, so a drain may run before they do. The
/// server sends the ping last, so once its pong exists every earlier idle
/// frame has reached the client — consumed by a drain already, or held by the
/// transport for the final one. Bounded by the caller's deadline.
async fn drain_until_ponged(
    session: &mut ResponsesWebSocketSession,
    pong_rx: &mpsc::Receiver<()>,
) -> Vec<UnrecognizedEvent> {
    let mut recovered = Vec::new();
    let mut ponged = false;
    loop {
        let (frames, ending) = session.keepalive().await.into_parts();
        recovered.extend(frames);
        if let Some(error) = ending {
            panic!("the idle drain should service the socket, got {error}");
        }
        if ponged {
            return recovered;
        }
        ponged = pong_rx.try_recv().is_ok();
        if !ponged {
            rig_core::wasm_compat::sleep(Duration::from_millis(10)).await;
        }
    }
}

async fn exercise_idle_window(
    bound: Bound<Responses, RecordingHttpClient>,
    pong_rx: mpsc::Receiver<()>,
) {
    let mut session = match bound.responses_websocket().await {
        Ok(session) => session,
        Err(error) => panic!("session should connect: {error}"),
    };
    session
        .completion(bound.completion_request("first").build())
        .await
        .expect("first turn should complete");

    let recovered = drain_until_ponged(&mut session, &pong_rx).await;
    let kinds: Vec<&str> = recovered.iter().map(|event| event.kind.as_str()).collect();
    assert_eq!(kinds, vec!["codex.rate_limits"]);
    assert_eq!(recovered[0].payload.value()["seen"], "idle");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    session
        .completion(bound.completion_request("second").build())
        .await
        .expect("second turn should complete after the idle window");
    session.close().await.expect("close should succeed");
}

/// The direct connection: the caller's own tokio runtime polls the socket.
#[tokio::test]
async fn an_idle_drain_services_a_direct_connection() {
    let (base_url, pong_rx) = serve_idle_window();
    tokio::time::timeout(
        Duration::from_secs(20),
        exercise_idle_window(bound(&base_url), pong_rx),
    )
    .await
    .expect("the idle window should finish in time");
}

/// The forwarded connection: no tokio runtime on the calling thread, so the
/// socket lives on the fallback runtime and every read is a round trip.
#[test]
fn an_idle_drain_services_a_forwarded_connection() {
    let (base_url, pong_rx) = serve_idle_window();
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime"
    );
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        Duration::from_secs(20),
        exercise_idle_window(bound(&base_url), pong_rx),
    ))
    .expect("the idle window should finish in time");
}

/// The Codex handshake reaches a real server with exactly the identity the
/// session then stamps on its frames.
#[tokio::test]
async fn the_codex_handshake_carries_the_session_identity() {
    use std::sync::{Arc, Mutex};
    use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let captured: Arc<Mutex<Option<http::HeaderMap>>> = Arc::new(Mutex::new(None));
    let captured_server = captured.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let callback =
            move |request: &Request, response: Response| -> Result<Response, ErrorResponse> {
                *captured_server.lock().expect("capture lock") = Some(request.headers().clone());
                Ok(response)
            };
        let mut socket = tokio_tungstenite::accept_hdr_async(stream, callback)
            .await
            .expect("upgrade");
        let request = socket
            .next()
            .await
            .expect("request")
            .expect("valid")
            .into_text()
            .expect("text")
            .to_string();
        socket
            .send(Message::text(completed("resp_1")))
            .await
            .expect("completed should send");
        while let Some(Ok(message)) = socket.next().await {
            if message.is_close() {
                break;
            }
        }
        request
    });

    let wire = OpenAI::with_key(&chatgpt::DIALECT, "test-token")
        .with_base_url(format!("http://{address}/backend-api/codex"))
        .with_account_id("acct-123")
        .responses(chatgpt::GPT_5_3_CODEX);
    let builder = CodexWebSocketSessionBuilder::new(wire).expect("a Codex wire");
    let identity = builder.identity().clone();
    let mut session = builder.connect().await.expect("session should connect");
    assert_eq!(session.identity(), &identity);

    session
        .send(rig_core::completion::CompletionRequest {
            model: None,
            chat_history: vec![rig_core::completion::Message::user("hello")],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .await
        .expect("root send should succeed");
    let terminal = session.next_event().await.expect("terminal event");
    assert!(terminal.is_terminal());
    session.close().await.expect("close should succeed");

    let frame = server.await.expect("server should finish");
    let headers = captured
        .lock()
        .expect("capture lock")
        .take()
        .expect("handshake headers should be captured");
    let header = |name: &str| {
        headers
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .to_str()
            .expect("ascii header")
            .to_owned()
    };
    assert_eq!(header("authorization"), "Bearer test-token");
    assert_eq!(header("chatgpt-account-id"), "acct-123");
    assert_eq!(header("session-id"), identity.session_id());
    assert_eq!(header("thread-id"), identity.thread_id());
    assert_eq!(header("x-client-request-id"), identity.thread_id());
    assert_eq!(header("openai-beta"), "responses_websockets=2026-02-06");

    let frame: serde_json::Value = serde_json::from_str(&frame).expect("frame is JSON");
    assert_eq!(frame["prompt_cache_key"], identity.thread_id());
    assert_eq!(
        frame["client_metadata"]["session_id"],
        identity.session_id()
    );
    assert!(frame.get("previous_response_id").is_none());
}
