//! Codex (ChatGPT subscription) Responses websocket sessions.
//!
//! [`CodexWebSocketSession`] composes the Responses websocket session over the
//! wire's own request shaping and a caller-chosen
//! [`WebSocketClientExt`] backend. It adds
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
//! - **Caller frame metadata is stamped on every frame.** Keys the caller
//!   sets ([`CodexWebSocketSession::set_frame_metadata`]), such as a turn's
//!   `x-codex-turn-state`, go into the `client_metadata` of every frame, full
//!   and incremental alike; an incremental send would otherwise repeat the
//!   captured envelope's stale values. A session given a [`FrameClock`] also
//!   stamps each frame's send instant as
//!   [`WS_STREAM_REQUEST_START_MS_METADATA_KEY`]. Rig reads no clock itself.
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
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::ws_client::{BoxedWebSocketConnection, WebSocketClientExt};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// The handshake header carrying the ChatGPT subscription account id.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The beta opt-in header enabling Codex's Responses websocket protocol.
pub const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";

/// The Codex Responses websocket beta value.
pub const RESPONSES_WEBSOCKETS_BETA_VALUE: &str = "responses_websockets=2026-02-06";

/// The Codex sticky-routing token: the header name inside a
/// `response.metadata` event, and the `client_metadata` key the official
/// client sends it back under.
pub const TURN_STATE_METADATA_KEY: &str = "x-codex-turn-state";

/// The `client_metadata` key carrying a frame's send instant, in milliseconds
/// since the Unix epoch, stamped when the session has a [`FrameClock`].
pub const WS_STREAM_REQUEST_START_MS_METADATA_KEY: &str = "x-codex-ws-stream-request-start-ms";

/// The instant a Codex websocket frame is sent, as the caller keeps time.
///
/// Rig never reads a clock itself: a session stamps
/// [`WS_STREAM_REQUEST_START_MS_METADATA_KEY`] only when given one of these
/// ([`CodexWebSocketSessionBuilder::with_request_start_ms_stamp`]), and asks
/// it once per frame, while encoding that frame.
pub trait FrameClock: WasmCompatSend + WasmCompatSync {
    /// Milliseconds since the Unix epoch, now.
    fn unix_millis(&self) -> u64;
}

/// Frame metadata keys a caller may not set, because the session stamps each
/// from the source that owns it.
const RESERVED_FRAME_METADATA_KEYS: &[&str] = &[
    SESSION_ID_METADATA_KEY,
    THREAD_ID_METADATA_KEY,
    super::super::responses_lite::WS_METADATA_KEY,
    WS_STREAM_REQUEST_START_MS_METADATA_KEY,
];

/// A frame metadata key the session stamps itself: the identity's session
/// and thread ids, the Responses Lite marker, or the send instant.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{key}` is a frame metadata key the Codex session stamps itself")]
pub struct ReservedFrameMetadataKey {
    /// The key as supplied.
    pub key: String,
}

impl From<ReservedFrameMetadataKey> for ProviderError {
    fn from(error: ReservedFrameMetadataKey) -> Self {
        ProviderError::Request(Box::new(error))
    }
}

/// What a session stamps on every frame besides its identity: the caller's
/// frame metadata and, when it has a clock, the send instant.
#[derive(Clone, Default)]
struct FrameStamps {
    metadata: BTreeMap<String, String>,
    clock: Option<Arc<dyn FrameClock>>,
}

impl FrameStamps {
    fn set(&mut self, key: String, value: String) -> Result<(), ReservedFrameMetadataKey> {
        if RESERVED_FRAME_METADATA_KEYS.contains(&key.as_str()) {
            return Err(ReservedFrameMetadataKey { key });
        }
        self.metadata.insert(key, value);
        Ok(())
    }

    /// Stamp the frame metadata over the frame's own `client_metadata`
    /// values, then the send instant. The identity has already made
    /// `client_metadata` an object.
    fn stamp(
        &self,
        body: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), EncodeError> {
        if self.metadata.is_empty() && self.clock.is_none() {
            return Ok(());
        }
        let Some(serde_json::Value::Object(metadata)) =
            body.get_mut(super::super::codex_identity::CLIENT_METADATA_FIELD)
        else {
            return Err(EncodeError::request(
                "a Codex frame's `client_metadata` must be a JSON object to carry frame metadata",
            ));
        };
        for (key, value) in &self.metadata {
            metadata.insert(key.clone(), serde_json::Value::String(value.clone()));
        }
        if let Some(clock) = &self.clock {
            metadata.insert(
                WS_STREAM_REQUEST_START_MS_METADATA_KEY.to_owned(),
                serde_json::Value::String(clock.unix_millis().to_string()),
            );
        }
        Ok(())
    }
}

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
    /// id when set, the dashed identity headers, `x-client-request-id`, the
    /// `OpenAI-Beta` opt-in, and last the wire's
    /// [request headers](Responses::with_request_headers). Nothing else: the
    /// per-request `session_id` HTTP header is not sent, because the dashed
    /// pair is the session identity here.
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
        wire.request_headers
            .stamp(
                self.stamp_headers(builder)
                    .header(OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE),
            )
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
    frame_stamps: FrameStamps,
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
            frame_stamps: FrameStamps::default(),
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

    /// Stamp `key: value` into the `client_metadata` of every frame the
    /// session sends, over the frame's own value; see
    /// [`CodexWebSocketSession::set_frame_metadata`]. Refuses a key the
    /// session stamps itself.
    pub fn with_frame_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, ReservedFrameMetadataKey> {
        self.frame_stamps.set(key.into(), value.into())?;
        Ok(self)
    }

    /// Stamp every frame's send instant, read from `clock` while the frame is
    /// encoded, as [`WS_STREAM_REQUEST_START_MS_METADATA_KEY`] in its
    /// `client_metadata`. Without a clock no instant is stamped.
    #[must_use]
    pub fn with_request_start_ms_stamp(mut self, clock: Arc<dyn FrameClock>) -> Self {
        self.frame_stamps.clock = Some(clock);
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
            frame_stamps: self.frame_stamps,
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
    frame_stamps: FrameStamps,
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
            frame_stamps: FrameStamps::default(),
        })
    }

    /// The identity every header and frame of this session carries.
    #[must_use]
    pub fn identity(&self) -> &CodexIdentity {
        &self.identity
    }

    /// Stamp `key: value` into the `client_metadata` of every frame this
    /// session sends from now on, full and incremental alike, over any value
    /// the frame itself carries for `key`: for a value that belongs to the
    /// current turn rather than to one request. Replaces the key's earlier
    /// value. Refuses a key the session stamps itself.
    ///
    /// The stamp is applied as each frame is encoded and is never stored in
    /// the envelope an incremental send reuses, so a turn-scoped key belongs
    /// here and never in a request's own `client_metadata`: once removed, a
    /// later incremental frame would carry the request's value again.
    ///
    /// The Codex backend's `x-codex-turn-state` sticky-routing token is such
    /// a value. The backend hands it over in the `headers` of a
    /// `response.metadata` event, which the session records whichever call
    /// reads it ([`Self::received_turn_state`]), and the official client sends
    /// it back on every later frame of that turn, never across turns.
    pub fn set_frame_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), ReservedFrameMetadataKey> {
        self.frame_stamps.set(key.into(), value.into())
    }

    /// The first `x-codex-turn-state` token a `response.metadata` event of
    /// this session's turns handed over since the token was last taken,
    /// whichever call read the event: [`Self::next_event`], [`Self::warmup`],
    /// [`Self::completion`] or a streamed turn. Later tokens are ignored
    /// until it is taken, as the official client keeps the first token of a
    /// turn. Rig never sends it back; a caller that does puts it in the
    /// [frame metadata](Self::set_frame_metadata) under
    /// [`TURN_STATE_METADATA_KEY`].
    #[must_use]
    pub fn received_turn_state(&self) -> Option<&str> {
        self.session.turn_state()
    }

    /// Take the recorded turn-state token, so the next one a
    /// `response.metadata` event carries is recorded: at a turn boundary,
    /// where the official client starts with no token.
    pub fn take_received_turn_state(&mut self) -> Option<String> {
        self.session.take_turn_state()
    }

    /// Stop stamping `key`, returning the value it had.
    pub fn remove_frame_metadata(&mut self, key: &str) -> Option<String> {
        self.frame_stamps.metadata.remove(key)
    }

    /// The frame metadata this session stamps on every frame.
    #[must_use]
    pub fn frame_metadata(&self) -> &BTreeMap<String, String> {
        &self.frame_stamps.metadata
    }

    /// Stamp each frame's send instant from `clock`, or stop stamping it with
    /// `None`; see [`CodexWebSocketSessionBuilder::with_request_start_ms_stamp`].
    pub fn set_request_start_ms_stamp(&mut self, clock: Option<Arc<dyn FrameClock>>) {
        self.frame_stamps.clock = clock;
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
            super::super::InputRequirement::for_create(&options),
        )?;
        let responses_lite = self.session.wire.codex_request_shape.is_lite();
        let frame = encode_frame(&envelope, options.generate, |body| {
            stamp_frame(&self.identity, responses_lite, &self.frame_stamps, body)
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
            stamp_frame(&self.identity, responses_lite, &self.frame_stamps, body)
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

    /// Send a root warmup turn (`generate: false`), wait for it to complete,
    /// and return its response ID. The completed warmup is the live tip, so
    /// [`Self::send_incremental`] continues it.
    ///
    /// A warmup generates nothing, so its request may carry no conversation
    /// at all: only its instructions (system messages) and tools, sent with
    /// an empty `input`. That is the Codex client's session-start prewarm:
    /// the warmup of the request the session's first turn will make, before
    /// that turn's items exist. The first turn then goes out as a
    /// [`send_incremental`](Self::send_incremental) of its items, chaining
    /// the warmup's response ID and reusing its model, instructions, tools
    /// and other properties. When the first turn needs different properties,
    /// it is a full [`send`](Self::send) instead, as the official client
    /// falls back to a full request when they do not match. Set the frame
    /// metadata and the wire's request headers for the warmup before calling
    /// it, and the turn's own before the first turn. A turn-state token the
    /// warmup's `response.metadata` handed over belongs to the first turn
    /// too, as it does in the official client: read it with
    /// [`Self::received_turn_state`] and stamp it before that turn.
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
    frame_stamps: &FrameStamps,
    body: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), EncodeError> {
    identity.stamp(body)?;
    if responses_lite {
        super::super::responses_lite::stamp_websocket_marker(body)?;
    }
    frame_stamps.stamp(body)
}

impl std::fmt::Debug for CodexWebSocketSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodexWebSocketSession")
            .field("identity", &self.identity)
            .field("frame_metadata", &self.frame_stamps.metadata)
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
