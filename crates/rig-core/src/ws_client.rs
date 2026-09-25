//! Transport-independent websocket handshakes, connections, and frames.
//! Backends preserve rejected upgrades as HTTP errors with status, headers, and body.
//!
//! ```
//! use rig_core::ws_client::websocket_url;
//!
//! assert_eq!(websocket_url("https://example.com/v1", "responses")?,
//!            "wss://example.com/v1/responses");
//! # Ok::<(), rig_core::http_client::Error>(())
//! ```

use crate::http_client::{Error, NoBody, Request, Result};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use bytes::Bytes;
use std::time::Duration;

/// One websocket data or control frame, in either direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    /// A UTF-8 text frame.
    Text(String),
    /// A binary frame.
    Binary(Bytes),
    /// A ping, with its application payload.
    Ping(Bytes),
    /// A pong, with its application payload.
    Pong(Bytes),
    /// A close frame, with the peer's status and reason when it sent one.
    Close(Option<CloseFrame>),
}

/// The status code and reason carried by a websocket close frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CloseFrame {
    /// The RFC 6455 close code.
    pub code: u16,
    /// The peer's reason, empty when it sent none.
    pub reason: String,
}

/// Options enforced by the backend during the handshake.
#[derive(Clone, Debug, Default)]
pub struct ConnectOptions {
    /// Backend-enforced handshake timeout, separate from session event timeouts.
    /// `None` imposes no handshake deadline.
    pub timeout: Option<Duration>,
}

impl ConnectOptions {
    /// Options with no connect timeout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the handshake timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Opens websocket connections from requests with WS(S) URIs and authentication
/// headers. Backends supply websocket handshake headers such as
/// `Sec-WebSocket-Key`; callers must not supply those headers.
pub trait WebSocketClientExt: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Opens a connection, preserving rejected upgrades with their HTTP status,
    /// headers, and response body.
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = Result<BoxedWebSocketConnection>> + WasmCompatSend;
}

/// What a [`WebSocketConnection::recv_ready`] read found without waiting on the
/// peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadyFrame {
    /// A frame that had already arrived.
    Frame(Frame),
    /// The peer had already ended the stream.
    Ended,
    /// Nothing had arrived yet.
    Empty,
}

/// A backend that does not implement an optional [`WebSocketConnection`]
/// capability.
///
/// The capability's default implementation returns this rather than a guess,
/// so a caller that needs it is told by name instead of being handed an answer
/// the backend never gave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("this websocket backend does not support `{capability}`")]
pub struct UnsupportedCapability {
    /// The trait method the backend does not implement.
    pub capability: &'static str,
}

impl UnsupportedCapability {
    /// Whether `error` is this refusal, for the named capability.
    #[must_use]
    pub fn is(error: &Error, capability: &'static str) -> bool {
        match error {
            Error::Instance(inner) => inner
                .downcast_ref::<Self>()
                .is_some_and(|unsupported| unsupported.capability == capability),
            _ => false,
        }
    }
}

/// One open websocket connection, usable as a trait object.
/// Calls are sequential: sessions must not poll send and receive concurrently.
/// WASM-compatible bounds preserve the containing session's thread-safety contract.
pub trait WebSocketConnection: WasmCompatSend + WasmCompatSync {
    /// Write one frame.
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>>;

    /// Read the next frame; `Ok(None)` means the peer ended the stream.
    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>>;

    /// Completes a close handshake. Callers must avoid repeated closes;
    /// backends may return an error for an already closed socket.
    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>>;

    /// Take the next frame that has already arrived, without waiting for the
    /// peer to send one: [`ReadyFrame::Empty`] when nothing is buffered.
    ///
    /// This is what an idle session uses to service a connection between
    /// turns. It exists because [`Self::recv`] cannot be used for that: a
    /// transport-neutral caller cannot tell "nothing has arrived" from "the
    /// answer is still on its way", and polling `recv` once and dropping it is
    /// lossless only for backends whose receive happens to be cancel-safe.
    ///
    /// Implementations must not lose a frame when the returned future is
    /// dropped before it resolves: a frame not handed back stays readable.
    ///
    /// The default refuses with [`UnsupportedCapability`] rather than
    /// reporting an empty buffer it never inspected.
    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        Box::pin(std::future::ready(Err(Error::instance(
            UnsupportedCapability {
                capability: "recv_ready",
            },
        ))))
    }

    /// Write out anything the backend queued on its own, such as the automatic
    /// pong it owes for a ping a read consumed.
    ///
    /// The default refuses with [`UnsupportedCapability`] rather than claiming
    /// a queue it never inspected is empty.
    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Err(Error::instance(
            UnsupportedCapability {
                capability: "flush",
            },
        ))))
    }
}

/// A type-erased [`WebSocketConnection`].
pub type BoxedWebSocketConnection = Box<dyn WebSocketConnection>;

impl WebSocketConnection for BoxedWebSocketConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).send(frame)
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        (**self).recv()
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).close(frame)
    }

    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        (**self).recv_ready()
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).flush()
    }
}

/// A base URL that cannot be turned into a websocket URL.
#[derive(Debug, thiserror::Error)]
#[error("invalid websocket base URL: {0}")]
pub struct InvalidWebSocketUrl(String);

/// Appends `path` to a base URL, converting HTTP(S) to WS(S) and retaining
/// existing WS(S) schemes, query, and fragment. Trims boundary slashes from the
/// appended path. Returns an error for invalid URLs or unsupported schemes.
pub fn websocket_url(base_url: &str, path: &str) -> Result<String> {
    fn invalid(message: impl Into<String>) -> Error {
        Error::instance(InvalidWebSocketUrl(message.into()))
    }

    let mut url =
        url::Url::parse(base_url).map_err(|error| invalid(format!("{base_url}: {error}")))?;

    let scheme = match url.scheme() {
        "https" | "wss" => "wss",
        "http" | "ws" => "ws",
        other => {
            return Err(invalid(format!(
                "unsupported base URL scheme for websocket mode: {other}"
            )));
        }
    };
    url.set_scheme(scheme)
        .map_err(|()| invalid(format!("failed to convert {base_url} to a websocket URL")))?;

    let path = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        path.trim_matches('/')
    );
    url.set_path(&path);
    Ok(url.to_string())
}

#[cfg(test)]
mod tests;
