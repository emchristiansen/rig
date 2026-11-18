//! The control socket over an in-memory connection. The protocol has no
//! request/response pairing for the HTTP cassette engine to record, so the
//! frames come from the probe's recording or are written here.

use super::*;
use rig_core::http_client::{self, NoBody, Request};
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use rig_core::ws_client::CloseFrame;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    inbound: VecDeque<Frame>,
    sent: Vec<String>,
    /// Transport closes that were started.
    close_calls: usize,
    /// Transport closes that completed.
    closed: usize,
    handshakes: Vec<Request<NoBody>>,
    /// Scripted outcomes of the next sends, then of the next closes;
    /// `Succeed` once each list is empty.
    send_outcomes: VecDeque<Outcome>,
    close_outcomes: VecDeque<Outcome>,
}

/// How one scripted send or close ends.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    Succeed,
    Fail,
    /// Never completes, so a caller can only drop it.
    Hang,
}

fn scripted_error() -> http_client::Error {
    http_client::Error::StreamEnded
}

/// A backend whose connections read queued frames, then end, and record
/// what is written.
#[derive(Clone, Default)]
struct Scripted(Arc<Mutex<State>>);

impl Scripted {
    fn with(frames: impl IntoIterator<Item = Frame>) -> Self {
        let scripted = Self::default();
        scripted.state().inbound.extend(frames);
        scripted
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().expect("unpoisoned")
    }

    fn sent_json(&self) -> Vec<serde_json::Value> {
        self.state()
            .sent
            .iter()
            .map(|text| serde_json::from_str(text).expect("JSON"))
            .collect()
    }
}

impl WebSocketConnection for Scripted {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let Frame::Text(text) = frame else {
            panic!("the control socket writes text frames only, got {frame:?}");
        };
        let mut state = self.state();
        match state.send_outcomes.pop_front().unwrap_or(Outcome::Succeed) {
            Outcome::Succeed => {
                state.sent.push(text);
                Box::pin(std::future::ready(Ok(())))
            }
            Outcome::Fail => Box::pin(std::future::ready(Err(scripted_error()))),
            Outcome::Hang => Box::pin(std::future::pending()),
        }
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let frame = self.state().inbound.pop_front();
        Box::pin(std::future::ready(Ok(frame)))
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = self.state();
        state.close_calls += 1;
        match state.close_outcomes.pop_front().unwrap_or(Outcome::Succeed) {
            Outcome::Succeed => {
                state.closed += 1;
                Box::pin(std::future::ready(Ok(())))
            }
            Outcome::Fail => Box::pin(std::future::ready(Err(scripted_error()))),
            Outcome::Hang => Box::pin(std::future::pending()),
        }
    }
}

impl WebSocketClientExt for Scripted {
    fn connect(
        &self,
        request: Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        self.state().handshakes.push(request);
        let connection: BoxedWebSocketConnection = Box::new(self.clone());
        std::future::ready(Ok(connection))
    }
}

/// The exact caller identity the tests send.
fn test_caller_identity() -> crate::support::CallerIdentity {
    crate::support::CallerIdentity::new("rig_test", "rig-test/0 (test identity)", None)
        .expect("a valid test identity")
}

fn conversation() -> crate::support::CodexIdentity {
    crate::support::CodexIdentity::from_ids("session-1", "thread-1").expect("valid ids")
}

fn calls() -> LiveCalls {
    LiveCalls::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token")
            .with_account_id("acct-123")
            .client(),
        conversation(),
    )
    .expect("the Codex backend")
    .with_caller_identity(test_caller_identity())
}

fn call_id() -> CallId {
    CallId::new("rtc_u2_test").expect("a call id")
}

/// Connecting sends the handshake `control_request` builds, then events
/// arrive typed in order, pings are skipped, and the socket's end is `None`.
#[tokio::test]
async fn events_arrive_typed_until_the_server_ends_the_socket() {
    let backend = Scripted::with([
        Frame::Text(r#"{"type":"session.started","session":{"id":"rtc_u2_test"}}"#.to_owned()),
        Frame::Ping(bytes::Bytes::new()),
        Frame::Binary(bytes::Bytes::from_static(
            br#"{"type":"turn.delta","turn_id":"t1","delta":" hi"}"#,
        )),
        Frame::Text(r#"{"type":"session.closed"}"#.to_owned()),
    ]);
    let calls = calls();
    let mut socket = calls
        .connect_control(&backend, call_id())
        .await
        .expect("connects");
    assert_eq!(socket.call_id(), &call_id());

    let handshake = backend.state().handshakes.pop().expect("one handshake");
    let expected = calls.control_request(&call_id()).expect("builds");
    assert_eq!(handshake.uri(), expected.uri());
    assert_eq!(handshake.headers(), expected.headers());

    assert!(matches!(
        socket.next_event().await,
        Ok(Some(ServerEvent::SessionStarted(_)))
    ));
    match socket.next_event().await {
        Ok(Some(ServerEvent::TurnDelta(delta))) => assert_eq!(delta.delta, " hi"),
        other => panic!("expected a turn delta, got {other:?}"),
    }
    assert!(matches!(
        socket.next_event().await,
        Ok(Some(ServerEvent::Unknown(_)))
    ));
    assert!(matches!(socket.next_event().await, Ok(None)));
}

/// A close frame ends the stream like a bare end.
#[tokio::test]
async fn a_close_frame_ends_the_stream() {
    let backend = Scripted::with([Frame::Close(None)]);
    let mut socket = ControlSocket::from_connection(Box::new(backend), call_id());
    assert!(matches!(socket.next_event().await, Ok(None)));
}

/// Appends go out chunked, one event per chunk; close sends `session.close`
/// once and completes the handshake once.
#[tokio::test]
async fn appends_are_chunked_and_close_is_sent_once() {
    let backend = Scripted::default();
    let mut socket = ControlSocket::from_connection(Box::new(backend.clone()), call_id());
    socket
        .append_delegation_context("item_1", Some(ContextChannel::Speakable), &"a".repeat(501))
        .await
        .expect("sent");
    socket
        .append_session_context(None, "note")
        .await
        .expect("sent");
    socket.close().await.expect("closed");
    socket.close().await.expect("a second close does nothing");

    let sent = backend.sent_json();
    let kinds: Vec<&str> = sent
        .iter()
        .map(|event| event["type"].as_str().expect("a type"))
        .collect();
    assert_eq!(
        kinds,
        [
            "delegation.context.append",
            "delegation.context.append",
            "session.context.append",
            "session.close"
        ]
    );
    assert_eq!(sent[1]["content"][0]["text"], "a");
    assert_eq!(sent[1]["delegation_item_id"], "item_1");
    assert_eq!(sent[1]["channel"], "speakable");
    assert_eq!(backend.state().closed, 1);
}

fn sent_kinds(backend: &Scripted) -> Vec<String> {
    backend
        .sent_json()
        .iter()
        .map(|event| event["type"].as_str().expect("a type").to_owned())
        .collect()
}

/// A failed `session.close` send still closes the transport and reports the
/// send's error; the transport is not closed twice.
#[tokio::test]
async fn a_failed_session_close_send_still_closes_the_transport() {
    let backend = Scripted::default();
    backend.state().send_outcomes.push_back(Outcome::Fail);
    let mut socket = ControlSocket::from_connection(Box::new(backend.clone()), call_id());

    assert!(socket.close().await.is_err(), "the send error is reported");
    assert_eq!(backend.state().closed, 1, "the transport closed anyway");
    assert!(sent_kinds(&backend).is_empty());

    socket.close().await.expect("nothing is left to do");
    assert_eq!(backend.state().close_calls, 1);
}

/// A failed transport close is retried by the next call, which does not
/// send `session.close` again.
#[tokio::test]
async fn a_failed_transport_close_is_retried() {
    let backend = Scripted::default();
    backend.state().close_outcomes.push_back(Outcome::Fail);
    let mut socket = ControlSocket::from_connection(Box::new(backend.clone()), call_id());

    assert!(socket.close().await.is_err());
    assert_eq!(backend.state().closed, 0);
    socket.close().await.expect("the retry completes");
    assert_eq!(backend.state().close_calls, 2);
    assert_eq!(backend.state().closed, 1);
    assert_eq!(sent_kinds(&backend), ["session.close"]);
    socket
        .close()
        .await
        .expect("a completed close does nothing");
    assert_eq!(backend.state().close_calls, 2);
}

/// A close dropped while sending `session.close` leaves both steps to the
/// next call.
#[tokio::test]
async fn a_close_cancelled_during_the_send_is_completed_by_a_retry() {
    let backend = Scripted::default();
    backend.state().send_outcomes.push_back(Outcome::Hang);
    let mut socket = ControlSocket::from_connection(Box::new(backend.clone()), call_id());

    assert!(futures::FutureExt::now_or_never(socket.close()).is_none());
    assert_eq!(backend.state().close_calls, 0);
    socket.close().await.expect("the retry completes");
    assert_eq!(sent_kinds(&backend), ["session.close"]);
    assert_eq!(backend.state().closed, 1);
}

/// A close dropped during the transport close retries only that step.
#[tokio::test]
async fn a_close_cancelled_during_the_transport_close_is_completed_by_a_retry() {
    let backend = Scripted::default();
    backend.state().close_outcomes.push_back(Outcome::Hang);
    let mut socket = ControlSocket::from_connection(Box::new(backend.clone()), call_id());

    assert!(futures::FutureExt::now_or_never(socket.close()).is_none());
    assert_eq!(backend.state().close_calls, 1);
    assert_eq!(backend.state().closed, 0);
    socket.close().await.expect("the retry completes");
    assert_eq!(sent_kinds(&backend), ["session.close"]);
    assert_eq!(backend.state().close_calls, 2);
    assert_eq!(backend.state().closed, 1);
}

/// A binary frame that is not UTF-8 is a corrupt frame carrying its exact
/// bytes.
#[tokio::test]
async fn a_binary_frame_that_is_not_utf8_keeps_its_bytes() {
    let received = vec![b'{', 0xff, 0xfe, b'}'];
    let backend = Scripted::with([Frame::Binary(bytes::Bytes::from(received.clone()))]);
    let mut socket = ControlSocket::from_connection(Box::new(backend), call_id());

    let error = socket.next_event().await.expect_err("the frame is refused");
    let corrupt = error.corrupt_frame().expect("a corrupt frame");
    assert_eq!(
        corrupt.evidence(),
        &crate::support::error::FrameEvidence::Bytes(received)
    );
    assert_eq!(corrupt.event_type(), None);
}

/// Under a credential source the handshake carries the caller's headers
/// once, beside the re-stamped credential and account.
#[tokio::test]
async fn a_credential_source_keeps_the_callers_headers_on_the_handshake() {
    struct Source;
    impl crate::support::CredentialSource for Source {
        fn current(
            &self,
        ) -> WasmBoxedFuture<
            '_,
            Result<crate::support::Credential, crate::support::CredentialSourceError>,
        > {
            Box::pin(async {
                Ok(crate::support::Credential::new("rotated-token")
                    .with_account_id("rotated-account"))
            })
        }
    }
    let calls = LiveCalls::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token")
            .with_account_id("acct-123")
            .client(),
        conversation(),
    )
    .expect("the Codex backend")
    .with_caller_identity(test_caller_identity())
    .with_credential_source(Source)
    .with_request_headers(
        crate::support::RequestHeaders::new()
            .with("x-codex-turn-metadata", r#"{"thread_source":"user"}"#)
            .expect("a valid header"),
    )
    .expect("not a header this protocol owns");
    let backend = Scripted::default();
    calls
        .connect_control(&backend, call_id())
        .await
        .expect("connects");
    let state = backend.state();
    let [handshake] = state.handshakes.as_slice() else {
        panic!("one handshake, got {}", state.handshakes.len());
    };
    let all = |name: &str| -> Vec<&str> {
        handshake
            .headers()
            .get_all(name)
            .iter()
            .map(|value| value.to_str().expect("ascii"))
            .collect()
    };
    assert_eq!(
        all("x-codex-turn-metadata"),
        [r#"{"thread_source":"user"}"#]
    );
    assert_eq!(all("authorization"), ["Bearer rotated-token"]);
    assert_eq!(all("chatgpt-account-id"), ["rotated-account"]);
}

/// The corrupt frame a control read carries, found by walking the source
/// chain of a boxed error, as a caller wrapping [`ControlError`] would.
fn corrupt_in_chain(error: ControlError) -> Option<crate::support::error::FrameEvidence> {
    let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(error);
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(boxed.as_ref());
    while let Some(error) = current {
        if let Some(corrupt) = error.downcast_ref::<CorruptFrame>() {
            return Some(corrupt.evidence().clone());
        }
        current = error.source();
    }
    None
}

/// A text frame that does not decode reaches the caller as
/// [`ControlError::CorruptFrame`] with the exact frame, its `type` and the
/// decode error, and the corrupt frame is in the error's source chain. The
/// socket goes on reading after it.
#[tokio::test]
async fn a_text_frame_that_does_not_decode_keeps_its_text_and_decode_error() {
    use crate::support::error::FrameEvidence;

    let malformed = r#"{"type":"turn.done","turn":{"id":1}}"#;
    let backend = Scripted::with([
        Frame::Text(malformed.into()),
        Frame::Text("not json".into()),
    ]);
    let mut socket = ControlSocket::from_connection(Box::new(backend), call_id());

    let error = socket.next_event().await.expect_err("the frame is refused");
    let corrupt = error.corrupt_frame().expect("a corrupt frame");
    assert_eq!(
        corrupt.evidence(),
        &FrameEvidence::Text(malformed.to_owned())
    );
    assert_eq!(corrupt.frame_bytes(), Some(malformed.as_bytes()));
    assert_eq!(corrupt.event_type(), Some("turn.done"));
    assert_eq!(
        corrupt.error().classify(),
        serde_json::error::Category::Data
    );
    assert_eq!(
        corrupt_in_chain(error),
        Some(FrameEvidence::Text(malformed.to_owned()))
    );

    let ControlError::CorruptFrame(not_json) =
        socket.next_event().await.expect_err("the frame is refused")
    else {
        panic!("a corrupt frame");
    };
    assert_eq!(not_json.frame(), Some("not json"));
    assert_eq!(not_json.event_type(), None);
    assert_eq!(
        not_json.error().classify(),
        serde_json::error::Category::Syntax
    );
}

/// A binary frame that is not UTF-8 keeps its exact bytes in the source
/// chain too, with the UTF-8 failure as its decode error.
#[tokio::test]
async fn a_binary_frame_that_is_not_utf8_is_in_the_source_chain() {
    let received = vec![b'{', 0xff, 0xfe, b'}'];
    let backend = Scripted::with([Frame::Binary(bytes::Bytes::from(received.clone()))]);
    let mut socket = ControlSocket::from_connection(Box::new(backend), call_id());

    let error = socket.next_event().await.expect_err("the frame is refused");
    let corrupt = error.corrupt_frame().expect("a corrupt frame");
    assert_eq!(corrupt.frame_bytes(), Some(received.as_slice()));
    assert_eq!(corrupt.frame(), None);
    assert!(
        corrupt.error().to_string().contains("invalid utf-8"),
        "{corrupt}"
    );
    assert_eq!(
        corrupt_in_chain(error),
        Some(crate::support::error::FrameEvidence::Bytes(received))
    );
}
