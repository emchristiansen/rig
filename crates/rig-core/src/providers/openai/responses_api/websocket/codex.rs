//! Codex (ChatGPT subscription) Responses websocket sessions.
//!
//! [`CodexWebSocketSession`] composes the Responses websocket session over the
//! wire's own request shaping and a caller-chosen
//! [`WebSocketClientExt`](crate::ws_client::WebSocketClientExt) backend. It adds
//! what the Codex gateway needs on top of the shared session:
//!
//! - **Root full sends never chain.** Codex runs full replay: [`send`] roots a
//!   fresh chain and adds no `previous_response_id`.
//! - **Explicit delta sends do chain.** [`send_incremental`] sends only an
//!   [`InputDelta`], reusing the `response.create` envelope captured from the
//!   last completed turn and chaining its response ID. It is refused by name
//!   ([`IncrementalSendRefused`]) when there is no eligible tip, including
//!   after a turn that ended `incomplete`.
//! - **One stable identity per session.** One [`CodexIdentity`] value supplies
//!   the dashed `session-id` / `thread-id` handshake headers,
//!   `x-client-request-id`, and the `prompt_cache_key` / `client_metadata`
//!   body fields of every frame, so cache routing stays sticky across turns.
//! - **Responses Lite is marked per frame.** An opted-in wire uses the shared
//!   deterministic developer prefix on full sends and adds the Lite marker to
//!   every frame's `client_metadata`; the upgrade handshake stays unchanged.
//! - **Single in-flight custody**, inherited from the shared session.
//!
//! The handshake carries the wire's credential as it stands when the session
//! connects: [`OpenAI::api_key`](crate::providers::openai::OpenAI::api_key),
//! or, when the wire has a
//! [credential source](crate::providers::openai::OpenAI::with_credential_source),
//! what that source supplies at connect. An open connection keeps the
//! credential it was opened with; a rotated credential takes effect at the
//! next session's connect.
//!
//! ```no_run
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAI;
//! use rig_core::providers::openai::responses_api::websocket::codex::CodexWebSocketSessionBuilder;
//!
//! # async fn example(backend: &impl rig_core::ws_client::WebSocketClientExt)
//! #     -> Result<(), rig_core::error::ProviderError> {
//! let wire = OpenAI::with_key(&chatgpt::DIALECT, "access-token")
//!     .with_account_id("account-id")
//!     .responses(chatgpt::GPT_5_3_CODEX);
//! let mut session = CodexWebSocketSessionBuilder::new(wire)?
//!     .connect_with(backend)
//!     .await?;
//! # session.close().await
//! # }
//! ```
//!
//! [`send`]: CodexWebSocketSession::send
//! [`send_incremental`]: CodexWebSocketSession::send_incremental

use super::{
    Chaining, DEFAULT_CONNECT_TIMEOUT, ResponsesWebSocketSession, WEBSOCKET_PATH, connect,
    encode_frame,
};

// The shared session types this session's API speaks, re-exported beside it.
pub use super::{
    EmptyInputDeltaError, IncrementalSendRefused, InputDelta, KeepaliveDrain,
    ResponsesWebSocketCreateOptions, ResponsesWebSocketEvent, UncollectedRecoveredFrames,
    UnrecognizedEvent,
};
use crate::completion;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::providers::openai::responses_api::wire::Responses;
use crate::ws_client::{BoxedWebSocketConnection, WebSocketClientExt};
use std::time::Duration;

/// The handshake header carrying the ChatGPT subscription account id.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The beta opt-in header enabling Codex's Responses websocket protocol.
pub const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";

/// The Codex Responses websocket beta value.
pub const RESPONSES_WEBSOCKETS_BETA_VALUE: &str = "responses_websockets=2026-02-06";

// The identity itself lives beside the HTTP wire that also stamps it; its
// public path here is kept.
pub(crate) use super::super::codex_identity::require_codex;
pub use super::super::codex_identity::{
    CodexIdentity, InvalidCodexIdentity, NotACodexWire, SESSION_ID_HEADER, SESSION_ID_METADATA_KEY,
    THREAD_ID_HEADER, THREAD_ID_METADATA_KEY, X_CLIENT_REQUEST_ID_HEADER,
};

impl CodexIdentity {
    /// The handshake request that opens a Codex websocket carrying this
    /// identity: `GET {base_url}/responses` on the websocket scheme, with the
    /// wire's credential, its caller identity (`originator` and `user-agent`,
    /// exactly as its HTTP requests carry them) when it has one, its account
    /// id when set, the dashed identity headers, `x-client-request-id`, and
    /// the `OpenAI-Beta` opt-in. Nothing else: the per-request `session_id`
    /// HTTP header is not sent, because the dashed pair is the session
    /// identity here.
    ///
    /// Public so a caller that opens its own connection can open it with the
    /// same identity the session will stamp on every frame. It stamps the
    /// wire's static credential; a caller with a credential source reads it
    /// itself, as [`CodexWebSocketSessionBuilder::connect_with`] does.
    pub fn handshake_request(
        &self,
        wire: &Responses,
    ) -> Result<http_client::Request<NoBody>, EncodeError> {
        let url = crate::ws_client::websocket_url(&wire.provider.base_url, WEBSOCKET_PATH)
            .map_err(EncodeError::request)?;

        let mut builder = wire.provider.identify(
            wire.provider.authenticate(
                http_client::Request::builder()
                    .method(http::Method::GET)
                    .uri(url),
            ),
        );
        if let Some(account_id) = &wire.provider.account_id {
            builder = builder.header(CHATGPT_ACCOUNT_ID_HEADER, account_id);
        }
        self.stamp_headers(builder)
            .header(OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE)
            .body(NoBody)
            .map_err(|error| {
                EncodeError::request(format!("Failed to build Codex websocket request: {error}"))
            })
    }
}

/// A builder for a [`CodexWebSocketSession`].
///
/// The default builder applies a 30 second connection timeout and leaves the
/// per-event timeout disabled.
pub struct CodexWebSocketSessionBuilder {
    wire: Responses,
    identity: CodexIdentity,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
}

impl CodexWebSocketSessionBuilder {
    /// Configure a session for `wire`, carrying the wire's own Codex identity
    /// when it has one ([`Responses::with_codex_identity`]), so its HTTP
    /// requests and this session name one conversation, and a freshly
    /// generated one otherwise. Refuses a wire whose dialect does not speak
    /// the Codex contract.
    pub fn new(wire: Responses) -> Result<Self, NotACodexWire> {
        require_codex(&wire)?;
        let identity = wire
            .codex_identity
            .clone()
            .unwrap_or_else(CodexIdentity::generate);
        Ok(Self {
            wire,
            identity,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
        })
    }

    /// The identity the session will carry.
    #[must_use]
    pub fn identity(&self) -> &CodexIdentity {
        &self.identity
    }

    /// Carry `identity` instead: for example one built with
    /// [`CodexIdentity::from_ids`] from ids the caller derives.
    #[must_use]
    pub fn with_identity(mut self, identity: CodexIdentity) -> Self {
        self.identity = identity;
        self
    }

    /// Sets the timeout for establishing the websocket connection.
    #[must_use]
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Disables the websocket connection timeout.
    #[must_use]
    pub fn without_connect_timeout(mut self) -> Self {
        self.connect_timeout = None;
        self
    }

    /// Sets the timeout for waiting on the next websocket event.
    #[must_use]
    pub fn event_timeout(mut self, timeout: Duration) -> Self {
        self.event_timeout = Some(timeout);
        self
    }

    /// Disables the websocket event timeout.
    #[must_use]
    pub fn without_event_timeout(mut self) -> Self {
        self.event_timeout = None;
        self
    }

    /// Open the session over `backend`.
    /// Return handshake construction, transport, or provider errors; a
    /// rejected upgrade keeps its status, headers, body and request ID.
    pub async fn connect_with<W>(self, backend: &W) -> Result<CodexWebSocketSession, ProviderError>
    where
        W: WebSocketClientExt,
    {
        let mut request = self.identity.handshake_request(&self.wire)?;
        super::authorize_handshake(&self.wire, &mut request).await?;
        let connection = connect(backend, request, self.connect_timeout).await?;
        Ok(CodexWebSocketSession {
            session: ResponsesWebSocketSession::from_connection(
                self.wire,
                connection,
                self.event_timeout,
            ),
            identity: self.identity,
        })
    }
}

/// A sequential Codex Responses websocket session. See the
/// [module documentation](self) for its chaining and identity contract.
///
/// Call [`Self::close`] to perform a close handshake.
pub struct CodexWebSocketSession {
    session: ResponsesWebSocketSession,
    identity: CodexIdentity,
}

impl CodexWebSocketSession {
    /// Build a session over an already-open connection, which the caller
    /// opened with [`CodexIdentity::handshake_request`] for `identity`.
    /// `event_timeout: None` waits indefinitely for each event.
    pub fn from_connection(
        wire: Responses,
        identity: CodexIdentity,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Result<Self, NotACodexWire> {
        require_codex(&wire)?;
        Ok(Self {
            session: ResponsesWebSocketSession::from_connection(wire, connection, event_timeout),
            identity,
        })
    }

    /// The identity every header and frame of this session carries.
    #[must_use]
    pub fn identity(&self) -> &CodexIdentity {
        &self.identity
    }

    /// The ID of the last response a turn on this session produced, if any.
    ///
    /// A turn that ended `incomplete` keeps its ID here as terminal evidence,
    /// but [`Self::send_incremental`] still refuses to continue it.
    #[must_use]
    pub fn previous_response_id(&self) -> Option<&str> {
        self.session.previous_response_id()
    }

    /// Drop the live tip. [`Self::send_incremental`] is refused until a full
    /// send establishes a new one.
    pub fn clear_previous_response_id(&mut self) {
        self.session.clear_previous_response_id();
    }

    /// Send a root full-replay turn. It never chains `previous_response_id`
    /// (unless the request itself names one), and its envelope becomes the one
    /// later deltas reuse if the turn completes.
    pub async fn send(
        &mut self,
        completion_request: completion::CompletionRequest,
    ) -> Result<(), ProviderError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::default(),
        )
        .await
    }

    /// [`Self::send`] with explicit websocket-mode options.
    pub async fn send_with_options(
        &mut self,
        completion_request: completion::CompletionRequest,
        options: ResponsesWebSocketCreateOptions,
    ) -> Result<(), ProviderError> {
        self.session.ensure_can_send()?;
        let envelope = self.session.prepare_request(
            completion_request,
            Chaining::Root,
            Some(&self.identity),
        )?;
        let responses_lite = self.session.wire.codex_request_shape.is_lite();
        let frame = encode_frame(&envelope, options.generate, |body| {
            stamp_frame(&self.identity, responses_lite, body)
        })?;
        self.session.send_encoded(envelope, frame).await
    }

    /// Continue the live tip with a forward-only incremental turn.
    ///
    /// Sends exactly `delta` as the new input, chaining the tip's response ID
    /// and reusing the non-input configuration (model, instructions, tools,
    /// reasoning, include, cache identity) of the envelope captured from the
    /// last completed turn. Changing any of that requires a full send.
    ///
    /// Refused by name before anything is written when the session has no
    /// eligible tip: no full send has completed, the last turn failed or ended
    /// `incomplete`, or the tip was cleared. Never falls back to a full
    /// replay.
    pub async fn send_incremental(&mut self, delta: InputDelta) -> Result<(), ProviderError> {
        self.session.ensure_can_send()?;
        let (envelope, tip) = self
            .session
            .continuation()
            .map_err(|refused| ProviderError::Request(Box::new(refused)))?;
        let envelope = envelope.clone();
        let mut request = envelope.clone();
        request.input = delta.into_vec();
        request.additional_parameters.previous_response_id = Some(tip.to_owned());

        let responses_lite = self.session.wire.codex_request_shape.is_lite();
        if responses_lite {
            super::super::responses_lite::shape_delta(&mut request.input);
        }
        let frame = encode_frame(&request, None, |body| {
            stamp_frame(&self.identity, responses_lite, body)
        })?;
        // The captured envelope, not the delta request, is what a later
        // delta continues from.
        self.session.send_encoded(envelope, frame).await
    }

    /// Reads the next server event for the current in-flight turn.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, ProviderError> {
        self.session.next_event().await
    }

    /// Service the idle connection between turns. See
    /// [`ResponsesWebSocketSession::keepalive`].
    pub async fn keepalive(&mut self) -> KeepaliveDrain {
        self.session.keepalive().await
    }

    /// Send a root warmup turn (`generate: false`) and return its response ID.
    pub async fn warmup(
        &mut self,
        completion_request: completion::CompletionRequest,
    ) -> Result<String, ProviderError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        Ok(self.session.wait_for_completed_response().await?.id)
    }

    /// Send a root turn and collect its normalized response. An incomplete
    /// turn is a terminal whose finish reason says why it stopped.
    pub async fn completion(
        &mut self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        self.send(completion_request).await?;
        self.session.collect_completion().await
    }

    /// Closes the websocket connection with a close handshake.
    pub async fn close(&mut self) -> Result<(), ProviderError> {
        self.session.close().await
    }
}

fn stamp_frame(
    identity: &CodexIdentity,
    responses_lite: bool,
    body: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), EncodeError> {
    identity.stamp(body)?;
    if responses_lite {
        super::super::responses_lite::stamp_websocket_marker(body)?;
    }
    Ok(())
}

impl std::fmt::Debug for CodexWebSocketSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodexWebSocketSession")
            .field("identity", &self.identity)
            .field("previous_response_id", &self.previous_response_id())
            .finish_non_exhaustive()
    }
}

/// Native sessions and builders satisfy Send and Sync.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CodexWebSocketSession>();
    assert_send_sync::<CodexWebSocketSessionBuilder>();
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
