//! The long-lived control socket of a GPT-Live call. One connection carries
//! many session events and commands, so this is a session rather than a
//! single request/response wire.

use std::time::Duration;

use super::call::{CallId, LiveCalls};
use super::{ClientEvent, ContextChannel, ServerEvent};
use crate::providers::live_support::exchange::reply_error;
use crate::providers::live_support::error::{CorruptFrame, ProviderError};
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt, WebSocketConnection,
};

/// How long the handshake may take unless the caller says otherwise.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The control socket of one call: typed server events in, typed client
/// events out.
///
/// Reads and writes are sequential, as [`WebSocketConnection`] requires. A
/// read future dropped before it resolves loses no frame only when the
/// backend's receive is cancel-safe. Forwarded native receive may lose a frame
/// when its queued result is dropped before the caller observes it.
pub struct ControlSocket {
    connection: BoxedWebSocketConnection,
    call_id: CallId,
    close: CloseProgress,
}

/// Which steps of [`ControlSocket::close`] have completed. Each flag is set
/// only after its step returned `Ok`, so an error or a dropped future leaves
/// the step to be retried.
#[derive(Clone, Copy, Debug, Default)]
struct CloseProgress {
    session_close_sent: bool,
    transport_closed: bool,
}

impl LiveCalls {
    /// Join the control socket of `call_id` over `backend`, with a 30 second
    /// handshake timeout. The already-resolved credential is sent unchanged.
    /// A rejected upgrade keeps its status, supplied headers, body
    /// and request id.
    pub async fn connect_control<W: WebSocketClientExt>(
        &self,
        backend: &W,
        call_id: CallId,
    ) -> Result<ControlSocket, ProviderError> {
        self.connect_control_with_timeout(backend, call_id, Some(DEFAULT_CONNECT_TIMEOUT))
            .await
    }

    /// [`Self::connect_control`] with `timeout` for the handshake; `None`
    /// sets no deadline.
    pub async fn connect_control_with_timeout<W: WebSocketClientExt>(
        &self,
        backend: &W,
        call_id: CallId,
        timeout: Option<Duration>,
    ) -> Result<ControlSocket, ProviderError> {
        let request = self.control_request(&call_id)?;
        let connection = backend
            .connect(request, ConnectOptions::new().with_timeout(timeout))
            .await
            .map_err(|error| reply_error(error, self.request_id_header()))?;
        Ok(ControlSocket::from_connection(connection, call_id))
    }
}

impl ControlSocket {
    /// A control socket over a connection the caller opened for `call_id`,
    /// for example with [`LiveCalls::control_request`].
    pub fn from_connection(connection: BoxedWebSocketConnection, call_id: CallId) -> Self {
        Self {
            connection,
            call_id,
            close: CloseProgress::default(),
        }
    }

    /// The call this socket controls.
    #[must_use]
    pub fn call_id(&self) -> &CallId {
        &self.call_id
    }

    /// The next server event. `Ok(None)` means the server ended the socket,
    /// with or without a close frame. Pings and pongs are skipped. A frame
    /// that does not decode, including a binary frame that is not UTF-8, is
    /// [`ProviderError::CorruptFrame`] carrying the frame as received.
    pub async fn next_event(&mut self) -> Result<Option<ServerEvent>, ProviderError> {
        loop {
            let frame = self
                .connection
                .recv()
                .await
                .map_err(ProviderError::from_transport_error)?;
            let text = match frame {
                None | Some(Frame::Close(_)) => return Ok(None),
                Some(Frame::Ping(_) | Frame::Pong(_)) => continue,
                Some(Frame::Text(text)) => text,
                Some(Frame::Binary(bytes)) => {
                    String::from_utf8(bytes.to_vec()).map_err(|error| {
                        let reason = serde::de::Error::custom(error.utf8_error());
                        ProviderError::CorruptFrame(CorruptFrame::bytes(error.into_bytes(), reason))
                    })?
                }
            };
            return ServerEvent::parse(&text).map(Some);
        }
    }

    /// Send one client event as a text frame.
    pub async fn send(&mut self, event: &ClientEvent) -> Result<(), ProviderError> {
        let text = serde_json::to_string(event)?;
        self.connection
            .send(Frame::Text(text))
            .await
            .map_err(ProviderError::from_transport_error)
    }

    /// Answer the delegation `delegation_item_id` with `text`, sent as
    /// consecutive `delegation.context.append` events of at most
    /// [`CONTEXT_APPEND_MAX_BYTES`](super::CONTEXT_APPEND_MAX_BYTES) bytes.
    pub async fn append_delegation_context(
        &mut self,
        delegation_item_id: &str,
        channel: Option<ContextChannel>,
        text: &str,
    ) -> Result<(), ProviderError> {
        for event in ClientEvent::delegation_context_append(delegation_item_id, channel, text) {
            self.send(&event).await?;
        }
        Ok(())
    }

    /// Add `text` to the session's context, sent as consecutive
    /// `session.context.append` events of at most
    /// [`CONTEXT_APPEND_MAX_BYTES`](super::CONTEXT_APPEND_MAX_BYTES) bytes.
    pub async fn append_session_context(
        &mut self,
        channel: Option<ContextChannel>,
        text: &str,
    ) -> Result<(), ProviderError> {
        for event in ClientEvent::session_context_append(channel, text) {
            self.send(&event).await?;
        }
        Ok(())
    }

    /// End the session: send `session.close`, then complete the websocket
    /// close handshake.
    ///
    /// The transport close is attempted even when sending `session.close`
    /// fails, and the first error is returned. Each step counts as done only
    /// once it returns `Ok`, so a call that failed or was dropped part way
    /// can be retried and repeats only the unfinished steps. Once the
    /// transport close has completed, further calls return `Ok(())` without
    /// sending anything, even if `session.close` was never delivered.
    pub async fn close(&mut self) -> Result<(), ProviderError> {
        if self.close.transport_closed {
            return Ok(());
        }
        let sent = if self.close.session_close_sent {
            Ok(())
        } else {
            self.send(&ClientEvent::SessionClose).await
        };
        if sent.is_ok() {
            self.close.session_close_sent = true;
        }
        let closed = self
            .connection
            .close(None)
            .await
            .map_err(ProviderError::from_transport_error);
        if closed.is_ok() {
            self.close.transport_closed = true;
        }
        sent.and(closed)
    }
}

impl std::fmt::Debug for ControlSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlSocket")
            .field("call_id", &self.call_id)
            .field("session_close_sent", &self.close.session_close_sent)
            .field("transport_closed", &self.close.transport_closed)
            .finish_non_exhaustive()
    }
}

/// Native control sockets satisfy Send and Sync.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ControlSocket>();
    assert_send_sync::<LiveCalls>();
};

#[cfg(test)]
mod tests;
