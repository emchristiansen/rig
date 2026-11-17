//! A websocket backend that opens Rig's own sessions as the Codex backend
//! expects them.
//!
//! [`CodexWebSocketSession`](crate::responses_api::websocket::codex::CodexWebSocketSession)
//! builds its whole handshake itself. Rig's own Responses websocket session
//! builds one with the dialect's headers; [`CodexWebSocketClient`] wraps the
//! backend that session connects through and makes that handshake carry
//! exactly the wire's caller identity, its conversation identity, its request
//! headers and the websocket beta opt-in.
//!
//! ```
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_codex::ws_client::CodexWebSocketClient;
//!
//! fn wrap<W>(backend: W, wire: CodexResponses) -> CodexWebSocketClient<W> {
//!     CodexWebSocketClient::new(backend, wire)
//! }
//! # let _ = wrap::<()>;
//! ```

use crate::responses_api::websocket::codex::{OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE};
use crate::responses_api::wire::{CodexRequestError, CodexResponses, set};
use rig_http::http_client::{NoBody, Request, Result};
use rig_http::wasm_compat::WasmCompatSend;
use rig_http::ws_client::{BoxedWebSocketConnection, ConnectOptions, WebSocketClientExt};

/// A [`WebSocketClientExt`] that applies a Codex wire's settings to every
/// handshake before `inner` opens it.
#[derive(Clone, Debug)]
pub struct CodexWebSocketClient<W> {
    inner: W,
    wire: CodexResponses,
}

impl<W> CodexWebSocketClient<W> {
    /// Connect through `inner` with `wire`'s caller identity, conversation
    /// identity and request headers.
    pub fn new(inner: W, wire: CodexResponses) -> Self {
        Self { inner, wire }
    }

    /// The Codex wire whose settings every handshake carries.
    pub fn wire(&self) -> &CodexResponses {
        &self.wire
    }

    /// The backend connections are opened through.
    pub fn inner(&self) -> &W {
        &self.inner
    }

    /// The handshake as it leaves: `request` with the wire's exact headers
    /// and the websocket beta opt-in. The credential is the one `request`
    /// already carries.
    fn prepare(
        &self,
        mut request: Request<NoBody>,
    ) -> std::result::Result<Request<NoBody>, CodexRequestError> {
        let headers = request.headers_mut();
        self.wire
            .stamp_headers(self.wire.codex_identity.as_ref(), headers)?;
        set(headers, OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE)?;
        Ok(request)
    }
}

impl<W: WebSocketClientExt> WebSocketClientExt for CodexWebSocketClient<W> {
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let prepared = self.prepare(request);
        let inner = self.inner.clone();
        async move { inner.connect(prepared?, options).await }
    }
}

#[cfg(test)]
mod tests;
