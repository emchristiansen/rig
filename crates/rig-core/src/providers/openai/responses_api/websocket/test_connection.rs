//! A scripted in-memory [`WebSocketConnection`] for the session unit tests.
//!
//! The "server" is a queue of frames. Scripted turns are released one per
//! text frame the session writes, the way a real endpoint answers each
//! `response.create`; idle frames can be queued at any point. Reads report
//! pings they consume as owed pongs, and a flush is what writes them, so a
//! test can assert that an idle drain actually answered the server.

use crate::http_client::{self, Error};
use crate::wasm_compat::WasmBoxedFuture;
use crate::ws_client::{
    BoxedWebSocketConnection, CloseFrame, Frame, ReadyFrame, UnsupportedCapability,
    WebSocketConnection,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ScriptState {
    turns: VecDeque<Vec<Frame>>,
    inbound: VecDeque<Frame>,
    /// Once `inbound` is empty: end the stream (`true`) or wait forever.
    ended: bool,
    sent: Vec<String>,
    pongs_owed: usize,
    pongs_flushed: usize,
    flushes: usize,
    closed: bool,
    without_recv_ready: bool,
    without_flush: bool,
    flush_stalls: bool,
    flush_fails: bool,
    /// The upgrade response's headers a new connection reports; `None`
    /// refuses the capability, as a backend that does not keep them does.
    handshake_response_headers: Option<http::HeaderMap>,
}

/// A scripted connection, cloneable so a test can inspect what the session
/// wrote after handing the connection to it.
#[derive(Clone, Default)]
pub(super) struct Script(Arc<Mutex<ScriptState>>);

impl Script {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, ScriptState> {
        self.0.lock().expect("script lock")
    }

    /// Queue a turn: JSON payloads released by the session's next write.
    #[must_use]
    pub(super) fn turn<I: IntoIterator<Item = String>>(self, frames: I) -> Self {
        self.state()
            .turns
            .push_back(frames.into_iter().map(Frame::Text).collect());
        self
    }

    /// Report `headers` as the upgrade response's headers of the connections
    /// this script hands out from now on.
    #[must_use]
    pub(super) fn with_handshake_response_headers(self, headers: http::HeaderMap) -> Self {
        self.state().handshake_response_headers = Some(headers);
        self
    }

    /// Make frames available now, as if they arrived while the session idled.
    pub(super) fn arrive<I: IntoIterator<Item = Frame>>(&self, frames: I) {
        self.state().inbound.extend(frames);
    }

    /// Make JSON payloads available now.
    pub(super) fn arrive_text<I: IntoIterator<Item = String>>(&self, payloads: I) {
        self.arrive(payloads.into_iter().map(Frame::Text));
    }

    /// End the stream once the queued frames are read.
    pub(super) fn end_stream(&self) {
        self.state().ended = true;
    }

    /// A backend that keeps the trait's default `recv_ready`.
    #[must_use]
    pub(super) fn without_recv_ready(self) -> Self {
        self.state().without_recv_ready = true;
        self
    }

    /// A backend that keeps the trait's default `flush`.
    #[must_use]
    pub(super) fn without_flush(self) -> Self {
        self.state().without_flush = true;
        self
    }

    /// Make every flush wait forever (or stop doing so).
    pub(super) fn set_flush_stalls(&self, stalls: bool) {
        self.state().flush_stalls = stalls;
    }

    /// Make every flush fail.
    pub(super) fn set_flush_fails(&self, fails: bool) {
        self.state().flush_fails = fails;
    }

    /// Every text payload the session has written, in order.
    pub(super) fn sent(&self) -> Vec<String> {
        self.state().sent.clone()
    }

    /// Every text payload the session has written, parsed.
    pub(super) fn sent_json(&self) -> Vec<serde_json::Value> {
        self.sent()
            .iter()
            .map(|frame| serde_json::from_str(frame).expect("sent frames are JSON"))
            .collect()
    }

    /// Pongs a flush has written for pings a read consumed.
    pub(super) fn pongs_flushed(&self) -> usize {
        self.state().pongs_flushed
    }

    /// How many flushes the session asked for.
    pub(super) fn flushes(&self) -> usize {
        self.state().flushes
    }

    /// Frames still queued for the session to read.
    pub(super) fn unread(&self) -> usize {
        self.state().inbound.len()
    }

    /// Whether the session completed a close handshake.
    pub(super) fn closed(&self) -> bool {
        self.state().closed
    }

    /// The scripted connection handle to hand to a session.
    pub(super) fn connection(&self) -> BoxedWebSocketConnection {
        let headers = self.state().handshake_response_headers.clone();
        Box::new(ScriptedConnection(self.clone(), headers))
    }
}

struct ScriptedConnection(Script, Option<http::HeaderMap>);

impl ScriptedConnection {
    /// Take the next queued frame, recording a consumed ping as an owed pong.
    fn take(&self) -> Option<Frame> {
        let mut state = self.0.state();
        let frame = state.inbound.pop_front();
        if matches!(frame, Some(Frame::Ping(_))) {
            state.pongs_owed += 1;
        }
        frame
    }
}

impl WebSocketConnection for ScriptedConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = self.0.state();
        match frame {
            Frame::Text(text) => state.sent.push(text),
            other => panic!("the session only writes text frames, got {other:?}"),
        }
        if let Some(turn) = state.turns.pop_front() {
            state.inbound.extend(turn);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        match self.take() {
            Some(frame) => Box::pin(std::future::ready(Ok(Some(frame)))),
            None if self.0.state().ended => Box::pin(std::future::ready(Ok(None))),
            None => Box::pin(std::future::pending()),
        }
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.0.state().closed = true;
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, http_client::Result<ReadyFrame>> {
        if self.0.state().without_recv_ready {
            return Box::pin(std::future::ready(Err(Error::instance(
                UnsupportedCapability {
                    capability: "recv_ready",
                },
            ))));
        }
        let answer = match self.take() {
            Some(frame) => ReadyFrame::Frame(frame),
            None if self.0.state().ended => ReadyFrame::Ended,
            None => ReadyFrame::Empty,
        };
        Box::pin(std::future::ready(Ok(answer)))
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = self.0.state();
        if state.without_flush {
            return Box::pin(std::future::ready(Err(Error::instance(
                UnsupportedCapability {
                    capability: "flush",
                },
            ))));
        }
        state.flushes += 1;
        if state.flush_stalls {
            return Box::pin(std::future::pending());
        }
        if state.flush_fails {
            return Box::pin(std::future::ready(Err(Error::StreamEnded)));
        }
        state.pongs_flushed += std::mem::take(&mut state.pongs_owed);
        Box::pin(std::future::ready(Ok(())))
    }

    fn handshake_response_headers(&self) -> http_client::Result<&http::HeaderMap> {
        self.1.as_ref().ok_or_else(|| {
            Error::instance(UnsupportedCapability {
                capability: "handshake_response_headers",
            })
        })
    }
}
