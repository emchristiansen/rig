//! Websocket connections that can be serviced while idle.
//!
//! Rig's [`rig_http::ws_client`] connection reads by waiting on the peer. An
//! idle Codex session also needs to take what has already arrived without
//! waiting, and to write out what the backend queued on its own.
//! [`WebSocketConnection`] adds those two operations to Rig's connection, and
//! [`WebSocketClientExt`] opens such connections. Rig's frame and option types
//! are re-exported, so this module names everything a backend needs.
//!
//! Two wrappers work on Rig's own connections too: [`CodexWebSocketClient`]
//! gives a handshake the Codex header set, and [`FrameTee`] keeps the frames
//! a connection read.
//!
//! ```
//! use rig_codex::ws_client::ReadyFrame;
//!
//! assert_ne!(ReadyFrame::Empty, ReadyFrame::Ended);
//! ```

use rig_http::http_client::{Error, NoBody, Request, Result};
use rig_http::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

pub use rig_http::ws_client::{CloseFrame, ConnectOptions, Frame, websocket_url};

mod codex_client;
mod frame_tee;

pub use codex_client::CodexWebSocketClient;
pub use frame_tee::FrameTee;

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

/// One open websocket connection that can also be serviced while idle.
///
/// Calls are sequential: sessions must not poll send and receive concurrently.
pub trait WebSocketConnection: rig_http::ws_client::WebSocketConnection {
    /// Take the next frame that has already arrived, without waiting for the
    /// peer to send one: [`ReadyFrame::Empty`] when nothing is buffered.
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

impl rig_http::ws_client::WebSocketConnection for BoxedWebSocketConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).send(frame)
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        (**self).recv()
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).close(frame)
    }
}

impl WebSocketConnection for BoxedWebSocketConnection {
    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        (**self).recv_ready()
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).flush()
    }
}

/// Opens [`WebSocketConnection`]s from requests with WS(S) URIs and
/// authentication headers. Backends supply websocket handshake headers such as
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

#[cfg(test)]
mod tests;
