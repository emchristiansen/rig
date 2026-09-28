//! The control socket of a GPT-Live call.

use std::time::Duration;

use super::call::{CallId, LiveCalls, reply_error};
use super::{ClientEvent, ContextChannel, ServerEvent};
use crate::error::ProviderError;
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
/// backend's receive is cancel-safe, as `rig-tungstenite`'s is.
pub struct ControlSocket {
    connection: BoxedWebSocketConnection,
    call_id: CallId,
    closed: bool,
}

impl LiveCalls {
    /// Join the control socket of `call_id` over `backend`, with a 30 second
    /// handshake timeout. The credential source, when the provider has one,
    /// is read once here. A rejected upgrade keeps its status, headers, body
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
        let mut request = self.control_request(&call_id)?;
        self.authorize(request.headers_mut()).await?;
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
            closed: false,
        }
    }

    /// The call this socket controls.
    #[must_use]
    pub fn call_id(&self) -> &CallId {
        &self.call_id
    }

    /// The next server event. `Ok(None)` means the server ended the socket,
    /// with or without a close frame. Pings and pongs are skipped. A frame
    /// that does not decode is [`ProviderError::CorruptFrame`].
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
                Some(Frame::Binary(bytes)) => String::from_utf8(bytes.to_vec())
                    .map_err(|error| ProviderError::Response(error.to_string()))?,
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
    /// close handshake. A second call does nothing.
    pub async fn close(&mut self) -> Result<(), ProviderError> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.send(&ClientEvent::SessionClose).await?;
        self.connection
            .close(None)
            .await
            .map_err(ProviderError::from_transport_error)
    }
}

impl std::fmt::Debug for ControlSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlSocket")
            .field("call_id", &self.call_id)
            .field("closed", &self.closed)
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
