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
//! - **Single in-flight custody**, inherited from the shared session.
//!
//! Credentials are the wire's static credential: the handshake carries
//! whatever [`OpenAI::api_key`](crate::providers::openai::OpenAI::api_key)
//! holds when the session connects. A caller that refreshes tokens builds the
//! wire from the fresh token before connecting.
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
use crate::providers::openai::wire::ResponsesContract;
use crate::ws_client::{BoxedWebSocketConnection, WebSocketClientExt};
use serde_json::{Map, Value};
use std::time::Duration;

/// The handshake header carrying the ChatGPT subscription account id.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The dashed session identity header of the Codex websocket transport.
pub const SESSION_ID_HEADER: &str = "session-id";

/// The dashed thread identity header of the Codex websocket transport.
pub const THREAD_ID_HEADER: &str = "thread-id";

/// The correlation header; Codex sets it to the thread id.
pub const X_CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// The beta opt-in header enabling Codex's Responses websocket protocol.
pub const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";

/// The Codex Responses websocket beta value.
pub const RESPONSES_WEBSOCKETS_BETA_VALUE: &str = "responses_websockets=2026-02-06";

/// The top-level body field carrying the cache-routing key.
const PROMPT_CACHE_KEY_FIELD: &str = "prompt_cache_key";

/// The top-level body field carrying correlation metadata.
const CLIENT_METADATA_FIELD: &str = "client_metadata";

/// The `client_metadata` key carrying the session id (Codex spelling).
pub const SESSION_ID_METADATA_KEY: &str = "session_id";

/// The `client_metadata` key carrying the thread id (Codex spelling).
pub const THREAD_ID_METADATA_KEY: &str = "thread_id";

/// The stable Codex cache and correlation identity of one websocket session.
///
/// Generated once per session and used for every header and body field that
/// names the session, so the handshake and every frame agree by construction.
/// The ids are opaque correlation strings from [`crate::id::generate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodexIdentity {
    session_id: String,
    thread_id: String,
}

impl CodexIdentity {
    /// A fresh identity.
    #[must_use]
    pub fn generate() -> Self {
        Self {
            session_id: crate::id::generate(),
            thread_id: crate::id::generate(),
        }
    }

    /// The session id: the `session-id` header and the `session_id` metadata key.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The thread id: the `thread-id` and `x-client-request-id` headers, the
    /// `thread_id` metadata key, and the default `prompt_cache_key`.
    #[must_use]
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// The cache-routing key. Codex derives it from the thread id.
    #[must_use]
    pub fn prompt_cache_key(&self) -> &str {
        &self.thread_id
    }

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
    /// same identity the session will stamp on every frame.
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
        builder
            .header(SESSION_ID_HEADER, &self.session_id)
            .header(THREAD_ID_HEADER, &self.thread_id)
            .header(X_CLIENT_REQUEST_ID_HEADER, &self.thread_id)
            .header(OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE)
            .body(NoBody)
            .map_err(|error| {
                EncodeError::request(format!("Failed to build Codex websocket request: {error}"))
            })
    }

    /// Stamp this identity onto a `response.create` body: `prompt_cache_key`
    /// and the `client_metadata` session and thread keys, each only where the
    /// request does not already carry its own value.
    fn stamp(&self, body: &mut Map<String, Value>) -> Result<(), EncodeError> {
        body.entry(PROMPT_CACHE_KEY_FIELD)
            .or_insert_with(|| Value::String(self.prompt_cache_key().to_owned()));

        let metadata = body
            .entry(CLIENT_METADATA_FIELD)
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(metadata) = metadata else {
            return Err(EncodeError::request(format!(
                "the Codex request's `{CLIENT_METADATA_FIELD}` must be a JSON object to carry the session identity, got {metadata}"
            )));
        };
        metadata
            .entry(SESSION_ID_METADATA_KEY)
            .or_insert_with(|| Value::String(self.session_id.clone()));
        metadata
            .entry(THREAD_ID_METADATA_KEY)
            .or_insert_with(|| Value::String(self.thread_id.clone()));
        Ok(())
    }
}

/// A wire that does not speak the Codex Responses contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "a Codex websocket session needs a Responses wire speaking the Codex contract; dialect `{dialect}` does not"
)]
pub struct NotACodexWire {
    /// The wire's dialect.
    pub dialect: &'static str,
}

impl From<NotACodexWire> for ProviderError {
    fn from(error: NotACodexWire) -> Self {
        ProviderError::Request(Box::new(error))
    }
}

fn require_codex(wire: &Responses) -> Result<(), NotACodexWire> {
    if wire.provider.dialect.quirks.responses.contract == ResponsesContract::Codex {
        Ok(())
    } else {
        Err(NotACodexWire {
            dialect: wire.provider.dialect.name,
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
    /// Configure a session for `wire`, with a freshly generated identity.
    /// Refuses a wire whose dialect does not speak the Codex contract.
    pub fn new(wire: Responses) -> Result<Self, NotACodexWire> {
        require_codex(&wire)?;
        Ok(Self {
            wire,
            identity: CodexIdentity::generate(),
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
        })
    }

    /// The identity the session will carry.
    #[must_use]
    pub fn identity(&self) -> &CodexIdentity {
        &self.identity
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
        let request = self.identity.handshake_request(&self.wire)?;
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
        let envelope = self
            .session
            .prepare_request(completion_request, Chaining::Root)?;
        let frame = encode_frame(&envelope, options.generate, |body| {
            self.identity.stamp(body)
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

        let frame = encode_frame(&request, None, |body| self.identity.stamp(body))?;
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
