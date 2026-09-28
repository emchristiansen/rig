//! The control socket over an in-memory connection. The protocol has no
//! request/response pairing for the HTTP cassette engine to record, so the
//! frames come from the probe's recording or are written here.

use super::*;
use crate::http_client::{self, NoBody, Request};
use crate::providers::chatgpt;
use crate::providers::openai::OpenAI;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use crate::ws_client::CloseFrame;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    inbound: VecDeque<Frame>,
    sent: Vec<String>,
    closed: usize,
    handshakes: Vec<Request<NoBody>>,
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
        match frame {
            Frame::Text(text) => self.state().sent.push(text),
            other => panic!("the control socket writes text frames only, got {other:?}"),
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let frame = self.state().inbound.pop_front();
        Box::pin(std::future::ready(Ok(frame)))
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.state().closed += 1;
        Box::pin(std::future::ready(Ok(())))
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

fn calls() -> LiveCalls {
    LiveCalls::new(
        OpenAI::with_key(&chatgpt::DIALECT, "test-token")
            .with_account_id("acct-123")
            .with_caller_identity(crate::test_utils::test_caller_identity()),
    )
    .expect("the Codex backend")
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
