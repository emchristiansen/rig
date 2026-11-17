use super::*;
use crate::responses_api::codex_identity::CodexIdentity;
use crate::responses_api::request_headers::RequestHeaders;
use crate::wire::{CallerIdentity, MissingCallerIdentity};
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_http::http_client::Error;
use rig_http::wasm_compat::WasmBoxedFuture;
use rig_http::ws_client::{CloseFrame, Frame, WebSocketConnection};
use std::sync::{Arc, Mutex};

/// A backend that keeps each handshake it is asked to open.
#[derive(Clone, Default)]
struct Recorder {
    handshakes: Arc<Mutex<Vec<Request<NoBody>>>>,
}

struct Silent;

impl WebSocketConnection for Silent {
    fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(std::future::ready(Ok(None)))
    }

    fn close(&mut self, _frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

impl WebSocketClientExt for Recorder {
    fn connect(
        &self,
        request: Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = Result<BoxedWebSocketConnection>> + WasmCompatSend {
        self.handshakes.lock().expect("unpoisoned").push(request);
        async move { Ok(Box::new(Silent) as BoxedWebSocketConnection) }
    }
}

fn anonymous_wire() -> CodexResponses {
    CodexResponses::new(Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        chatgpt::GPT_5_3_CODEX,
    ))
    .expect("the ChatGPT dialect speaks the Codex contract")
}

fn wire() -> CodexResponses {
    anonymous_wire()
        .with_caller_identity(
            CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
                .expect("a valid caller identity"),
        )
        .with_codex_identity(
            CodexIdentity::from_ids("session-derived", "thread-derived").expect("header-safe ids"),
        )
        .expect("the ChatGPT dialect speaks the Codex contract")
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-window-id", "thread-1:0")
                .expect("a valid header"),
        )
}

/// The handshake Rig's own session builds on the ChatGPT dialect: its
/// credential, an identity Rig derived, and a session id of Rig's own.
fn rigs_handshake(session_id: &str) -> Request<NoBody> {
    Request::builder()
        .method(http::Method::GET)
        .uri("wss://chatgpt.com/backend-api/codex/responses")
        .header("authorization", "Bearer test-token")
        .header("originator", "rig")
        .header("user-agent", "rig/0 (linux x86_64; rig)")
        .header("version", "0.46.0")
        .header("session-id", session_id)
        .header("session_id", session_id)
        .body(NoBody)
        .expect("builds")
}

fn sorted(request: &Request<NoBody>) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = request
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                value.to_str().expect("ascii header").to_owned(),
            )
        })
        .collect();
    headers.sort();
    headers
}

/// The handshake that leaves carries exactly the Codex set: the request's own
/// credential, the wire's caller identity in place of any other, the
/// conversation's identity in place of any other session id, the caller's
/// request headers and the websocket beta opt-in. A `version` the caller
/// identity does not name is not sent.
#[tokio::test]
async fn the_handshake_leaves_with_exactly_the_codex_header_set() {
    let recorder = Recorder::default();
    let client = CodexWebSocketClient::new(recorder.clone(), wire());
    client
        .connect(rigs_handshake("rig-made-1"), ConnectOptions::new())
        .await
        .expect("connects");

    let handshakes = recorder.handshakes.lock().expect("unpoisoned");
    assert_eq!(handshakes.len(), 1);
    assert_eq!(
        handshakes[0].uri(),
        "wss://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        sorted(&handshakes[0]),
        [
            ("authorization", "Bearer test-token"),
            ("openai-beta", "responses_websockets=2026-02-06"),
            ("originator", "client_exec"),
            ("session-id", "session-derived"),
            ("thread-id", "thread-derived"),
            ("user-agent", "client_exec/1.2.3 (test)"),
            ("x-client-request-id", "thread-derived"),
            ("x-codex-window-id", "thread-1:0"),
        ]
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
    );
}

/// What leaves is a function of the wire and the request's credential: two
/// handshakes that differ only in the session id Rig made leave identical.
#[tokio::test]
async fn the_handshake_is_a_function_of_the_wire_not_of_rigs_own_session_id() {
    let recorder = Recorder::default();
    let client = CodexWebSocketClient::new(recorder.clone(), wire());
    for made in ["rig-made-1", "rig-made-2"] {
        client
            .connect(rigs_handshake(made), ConnectOptions::new())
            .await
            .expect("connects");
    }
    let handshakes = recorder.handshakes.lock().expect("unpoisoned");
    assert_eq!(sorted(&handshakes[0]), sorted(&handshakes[1]));
}

/// Without a caller identity nothing is opened, and the refusal keeps its
/// type.
#[tokio::test]
async fn a_wire_without_a_caller_identity_opens_nothing() {
    let recorder = Recorder::default();
    let client = CodexWebSocketClient::new(recorder.clone(), anonymous_wire());
    let Err(error) = client
        .connect(rigs_handshake("rig-made-1"), ConnectOptions::new())
        .await
    else {
        panic!("no caller identity, no handshake");
    };
    match error {
        Error::Instance(inner) => assert_eq!(
            inner.downcast_ref::<MissingCallerIdentity>(),
            Some(&MissingCallerIdentity { dialect: "chatgpt" })
        ),
        other => panic!("expected the typed refusal, got {other:?}"),
    }
    assert!(recorder.handshakes.lock().expect("unpoisoned").is_empty());
    assert_eq!(client.wire(), &anonymous_wire());
    assert!(
        client
            .inner()
            .handshakes
            .lock()
            .expect("unpoisoned")
            .is_empty()
    );
}
