//! Codex (ChatGPT subscription) Responses websocket sessions.
//!
//! [`CodexWebSocketSession`] owns one websocket to the Codex gateway, opened
//! over a caller-chosen [`WebSocketClientExt`] backend. Rig's Responses wire
//! encodes each request body; the session shapes it into the frames the Codex
//! gateway takes.
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
//! - **Caller frame metadata is stamped on every frame.** Keys the caller
//!   sets ([`CodexWebSocketSession::set_frame_metadata`]), such as a turn's
//!   `x-codex-turn-state`, go into the `client_metadata` of every frame, full
//!   and incremental alike. A session given a [`FrameClock`] also stamps each
//!   frame's send instant as [`WS_STREAM_REQUEST_START_MS_METADATA_KEY`]. The
//!   session reads no clock itself.
//! - **Single in-flight custody.** One turn is in flight at a time.
//!
//! The handshake carries the wire's credential as it stands when the session
//! connects. An open connection keeps the credential it was opened with.
//!
//! ```no_run
//! use rig_codex::responses_api::websocket::codex::CodexWebSocketSessionBuilder;
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::providers::openai::responses_api::wire::Responses;
//!
//! # async fn example(
//! #     backend: &impl rig_codex::ws_client::WebSocketClientExt,
//! #     identity: rig_codex::wire::CallerIdentity,
//! #     conversation: rig_codex::responses_api::codex_identity::CodexIdentity,
//! # ) -> Result<(), rig_core::error::ProviderError> {
//! let provider =
//!     OpenAIConfig::with_key(&chatgpt::DIALECT, "access-token").with_account_id("account-id");
//! let wire = CodexResponses::new(Responses::new(provider, chatgpt::GPT_5_3_CODEX))?
//!     .with_caller_identity(identity)
//!     .with_codex_identity(conversation)?;
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
    Continuation, DEFAULT_CONNECT_TIMEOUT, KEEPALIVE_FLUSH_TIMEOUT, MAX_KEEPALIVE_DRAIN_FRAMES,
    RESPONSE_METADATA_EVENT, TurnEnding, WEBSOCKET_PATH, parse_server_event,
    provider_error_from_event, terminal_response_result, websocket_frame_to_text,
    websocket_provider_error,
};

// The shared session types this session's API speaks, re-exported beside it.
pub use super::{
    EmptyInputDeltaError, IncrementalSendRefused, InputDelta, KeepaliveDrain,
    ResponsesWebSocketCreateOptions, ResponsesWebSocketEvent, UncollectedRecoveredFrames,
    UnrecognizedEvent,
};
use crate::error::CodexSessionError;
use crate::responses_api::codex_identity::{CLIENT_METADATA_FIELD, require_codex};
use crate::responses_api::streaming::ResponseChunkKind;
use crate::responses_api::wire::{CodexResponses, UnsupportedCodexControl, set};
use crate::responses_api::ResponseStatus;
use crate::responses_api::body::RequestBody;
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, ReadyFrame, UnsupportedCapability,
    WebSocketClientExt,
};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::http_client::{self, NoBody};
use rig_core::operation::Completion;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use rig_http::ws_client::WebSocketConnection as _;
use serde::Serialize;
use serde::ser::SerializeMap;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

pub use crate::responses_api::wire::CHATGPT_ACCOUNT_ID_HEADER;

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

/// The Responses Lite marker key. This crate shapes no Lite frame, so a
/// caller may not stamp the marker either.
const RESPONSES_LITE_METADATA_KEY: &str =
    "ws_request_header_x_openai_internal_codex_responses_lite";

/// The text of the user message that stands in for a conversation while Rig's
/// encoder shapes an input-free warmup. It never reaches the wire.
const WARMUP_PLACEHOLDER: &str = "warmup";

/// The instant a Codex websocket frame is sent, as the caller keeps time.
///
/// The session never reads a clock itself: it stamps
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
    RESPONSES_LITE_METADATA_KEY,
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
        ProviderError::request(error)
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
    fn stamp(&self, body: &mut RequestBody) -> Result<(), EncodeError> {
        if self.metadata.is_empty() && self.clock.is_none() {
            return Ok(());
        }
        let Some(Ok(mut metadata)) = body.get::<Map<String, Value>>(CLIENT_METADATA_FIELD) else {
            return Err(EncodeError::request(
                "a Codex frame's `client_metadata` must be a JSON object to carry frame metadata",
            ));
        };
        for (key, value) in &self.metadata {
            metadata.insert(key.clone(), Value::String(value.clone()));
        }
        if let Some(clock) = &self.clock {
            metadata.insert(
                WS_STREAM_REQUEST_START_MS_METADATA_KEY.to_owned(),
                Value::String(clock.unix_millis().to_string()),
            );
        }
        body.insert(CLIENT_METADATA_FIELD, &metadata)
            .map_err(EncodeError::request)
    }
}

pub use crate::responses_api::codex_identity::{
    CodexIdentity, InvalidCodexIdentity, MissingCodexIdentity, NotACodexWire, SESSION_ID_HEADER,
    SESSION_ID_METADATA_KEY, THREAD_ID_HEADER, THREAD_ID_METADATA_KEY, X_CLIENT_REQUEST_ID_HEADER,
};

impl CodexIdentity {
    /// The handshake request that opens a Codex websocket carrying this
    /// identity: `GET {base_url}/responses` on the websocket scheme, with the
    /// wire's credential, its account id when set, its caller identity
    /// (`originator`, `user-agent` and any `version`; refused with
    /// [`MissingCallerIdentity`](crate::wire::MissingCallerIdentity) when the
    /// wire has none), the dashed identity headers, `x-client-request-id`,
    /// the wire's [request headers](CodexResponses::with_request_headers) and
    /// the `OpenAI-Beta` opt-in. Nothing else.
    ///
    /// Public so a caller that opens its own connection can open it with the
    /// same identity the session will stamp on every frame.
    pub fn handshake_request(
        &self,
        wire: &CodexResponses,
    ) -> Result<http_client::Request<NoBody>, EncodeError> {
        let url = crate::ws_client::websocket_url(&wire.wire.provider.base_url, WEBSOCKET_PATH)
            .map_err(EncodeError::request)?;
        let mut request = http_client::Request::builder()
            .method(http::Method::GET)
            .uri(url)
            .body(NoBody)
            .map_err(|error| {
                EncodeError::request(format!("Failed to build Codex websocket request: {error}"))
            })?;
        let headers = request.headers_mut();
        wire.stamp_credential(headers)?;
        wire.stamp_headers(Some(self), headers)?;
        set(headers, OPENAI_BETA_HEADER, RESPONSES_WEBSOCKETS_BETA_VALUE)?;
        Ok(request)
    }
}

/// A builder for a [`CodexWebSocketSession`].
///
/// The default builder applies a 30 second connection timeout and leaves the
/// per-event timeout disabled.
pub struct CodexWebSocketSessionBuilder {
    wire: CodexResponses,
    identity: Option<CodexIdentity>,
    frame_stamps: FrameStamps,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
}

impl CodexWebSocketSessionBuilder {
    /// Configure a session for `wire`, carrying the wire's own Codex identity
    /// when it has one ([`CodexResponses::with_codex_identity`]), so its HTTP
    /// requests and this session name one conversation. Refuses a wire whose
    /// dialect does not speak the Codex contract.
    ///
    /// The session draws no identity of its own: without one on the wire or
    /// from [`Self::with_identity`], [`Self::connect_with`] refuses with
    /// [`MissingCodexIdentity`].
    pub fn new(wire: CodexResponses) -> Result<Self, NotACodexWire> {
        require_codex(&wire.wire)?;
        let identity = wire.codex_identity.clone();
        Ok(Self {
            wire,
            identity,
            frame_stamps: FrameStamps::default(),
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
        })
    }

    /// The identity the session will carry, when one is named.
    #[must_use]
    pub fn identity(&self) -> Option<&CodexIdentity> {
        self.identity.as_ref()
    }

    /// Carry `identity` instead of the wire's: one built with
    /// [`CodexIdentity::from_ids`] from ids the caller derives.
    #[must_use]
    pub fn with_identity(mut self, identity: CodexIdentity) -> Self {
        self.identity = Some(identity);
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

    /// Open the session over the bundled
    /// [`TungsteniteClient`](crate::tungstenite::TungsteniteClient) backend;
    /// see [`Self::connect_with`].
    #[cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "tungstenite")))]
    pub async fn connect(self) -> Result<CodexWebSocketSession, ProviderError> {
        self.connect_with(&crate::tungstenite::TungsteniteClient::new())
            .await
    }

    /// Open the session over `backend`.
    /// Return handshake construction, transport, or provider errors; a
    /// rejected upgrade keeps its status, headers, body and request ID.
    pub async fn connect_with<W>(self, backend: &W) -> Result<CodexWebSocketSession, ProviderError>
    where
        W: WebSocketClientExt,
    {
        let identity = self.identity.ok_or(MissingCodexIdentity)?;
        let request = identity.handshake_request(&self.wire)?;
        let connection = backend
            .connect(
                request,
                ConnectOptions::new().with_timeout(self.connect_timeout),
            )
            .await
            .map_err(websocket_provider_error)?;
        let mut session =
            CodexWebSocketSession::over(self.wire, identity, connection, self.event_timeout);
        session.frame_stamps = self.frame_stamps;
        Ok(session)
    }
}

/// A sequential Codex Responses websocket session. See the
/// [module documentation](self) for its chaining and identity contract.
///
/// Call [`Self::close`] to perform a close handshake.
pub struct CodexWebSocketSession {
    wire: CodexResponses,
    identity: CodexIdentity,
    frame_stamps: FrameStamps,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    /// What an incremental continuation may chain onto.
    continuation: Continuation,
    /// Unmodelled frames an idle drain took off the socket and has not yet
    /// handed back. Held here rather than in the drain's own future so that a
    /// drain dropped at any suspension point cannot destroy them.
    recovered: Vec<UnrecognizedEvent>,
    socket: BoxedWebSocketConnection,
    /// The first Codex `x-codex-turn-state` token a `response.metadata` event
    /// of an in-flight turn handed the session, until the caller takes it.
    turn_state: Option<String>,
    in_flight: bool,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl CodexWebSocketSession {
    fn over(
        wire: CodexResponses,
        identity: CodexIdentity,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Self {
        Self {
            wire,
            identity,
            frame_stamps: FrameStamps::default(),
            previous_response_id: None,
            pending_done_response_id: None,
            continuation: Continuation::Blocked(IncrementalSendRefused::NoCompletedTurn),
            recovered: Vec::new(),
            socket: connection,
            turn_state: None,
            in_flight: false,
            event_timeout,
            closed: false,
            failed: false,
        }
    }

    /// Build a session over an already-open connection, which the caller
    /// opened with [`CodexIdentity::handshake_request`] for `identity`.
    /// `event_timeout: None` waits indefinitely for each event.
    ///
    /// Every send refuses with
    /// [`MissingCallerIdentity`](crate::wire::MissingCallerIdentity) until
    /// `wire` carries a caller identity, whoever opened the connection.
    pub fn from_connection(
        wire: CodexResponses,
        identity: CodexIdentity,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Result<Self, NotACodexWire> {
        require_codex(&wire.wire)?;
        Ok(Self::over(wire, identity, connection, event_timeout))
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
    /// `response.metadata` event, which the session records
    /// ([`Self::received_turn_state`]), and the official client sends it back
    /// on every later frame of that turn, never across turns.
    pub fn set_frame_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), ReservedFrameMetadataKey> {
        self.frame_stamps.set(key.into(), value.into())
    }

    /// The first `x-codex-turn-state` token a `response.metadata` event of
    /// this session's turns handed over since the token was last taken.
    /// Later tokens are ignored until it is taken, as the official client
    /// keeps the first token of a turn. The session never sends it back; a
    /// caller that does puts it in the
    /// [frame metadata](Self::set_frame_metadata) under
    /// [`TURN_STATE_METADATA_KEY`].
    #[must_use]
    pub fn received_turn_state(&self) -> Option<&str> {
        self.turn_state.as_deref()
    }

    /// Take the recorded turn-state token, so the next one a
    /// `response.metadata` event carries is recorded: at a turn boundary,
    /// where the official client starts with no token.
    ///
    /// The session spans turns and marks no boundary itself, so nothing
    /// clears the token but this call.
    pub fn take_received_turn_state(&mut self) -> Option<String> {
        self.turn_state.take()
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
        self.previous_response_id.as_deref()
    }

    /// Drop the live tip. [`Self::send_incremental`] is refused until a full
    /// send establishes a new one.
    pub fn clear_previous_response_id(&mut self) {
        self.previous_response_id = None;
        self.continuation = Continuation::Blocked(IncrementalSendRefused::TipCleared);
    }

    /// Send a root full-replay turn. It never chains `previous_response_id`
    /// (unless the request itself names one), and its envelope becomes the one
    /// later deltas reuse if the turn completes.
    pub async fn send(
        &mut self,
        completion_request: CompletionRequest,
    ) -> Result<(), ProviderError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::default(),
        )
        .await
    }

    /// [`Self::send`] with explicit websocket-mode options.
    ///
    /// A warmup (`generate: false`) generates nothing, so its request may
    /// carry no conversation at all: only its instructions (system messages)
    /// and tools, sent with an empty `input`. It still needs at least one
    /// message, so a warmup without instructions is refused before anything
    /// is sent. That is the Codex client's session-start prewarm: the warmup
    /// of the request the session's first turn will make, before that turn's
    /// items exist. The first turn then goes out as a
    /// [`send_incremental`](Self::send_incremental) of its items, chaining
    /// the warmup's response ID and reusing its model, instructions, tools
    /// and other properties.
    pub async fn send_with_options(
        &mut self,
        completion_request: CompletionRequest,
        options: ResponsesWebSocketCreateOptions,
    ) -> Result<(), ProviderError> {
        self.ensure_can_send()?;
        let envelope = self.prepare_request(completion_request, &options)?;
        let frame = self.encode_frame(&envelope, options.generate)?;
        self.send_encoded(envelope, frame).await
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
        self.ensure_can_send()?;
        let (envelope, tip) = match &self.continuation {
            Continuation::Eligible { envelope, tip } => (envelope.clone(), tip.clone()),
            Continuation::Blocked(reason) => return Err(ProviderError::request(reason.clone())),
            // `ensure_can_send` refuses an in-flight session first.
            Continuation::InFlight(_) => {
                return Err(ProviderError::request(
                    IncrementalSendRefused::LastTurnNotCompleted,
                ));
            }
        };
        let mut request = (*envelope).clone();
        request.insert("input", &delta.into_vec())?;
        request.insert("previous_response_id", &tip)?;
        let frame = self.encode_frame(&request, None)?;
        // The captured envelope, not the delta request, is what a later
        // delta continues from.
        self.send_encoded(*envelope, frame).await
    }

    /// Send a warmup turn (`generate: false`), read it to its end and return
    /// the resulting response ID.
    ///
    /// This is [`Self::send_with_options`] with
    /// [`ResponsesWebSocketCreateOptions::warmup`], so its request may carry
    /// instructions and tools alone. A failed turn, an error event and a
    /// `response.done` without a response body are errors; an incomplete turn
    /// returns its ID. Set the frame metadata for the warmup before calling
    /// it, and the turn's own before the first turn. A turn-state token the
    /// warmup's `response.metadata` handed over belongs to the first turn:
    /// read it with [`Self::received_turn_state`].
    pub async fn warmup(
        &mut self,
        completion_request: CompletionRequest,
    ) -> Result<String, CodexSessionError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        loop {
            match self.next_event().await? {
                ResponsesWebSocketEvent::Response(chunk) => {
                    if matches!(
                        chunk.kind,
                        ResponseChunkKind::ResponseCompleted
                            | ResponseChunkKind::ResponseFailed
                            | ResponseChunkKind::ResponseIncomplete
                    ) {
                        return Ok(terminal_response_result(chunk.response)?.id);
                    }
                }
                ResponsesWebSocketEvent::Done(done) => {
                    return match done.response.completion_response() {
                        Some(response) => Ok(terminal_response_result(response.clone())?.id),
                        None => Err(ProviderError::Provider(format!(
                            "OpenAI websocket turn ended with response.done before a terminal response body was available (response_id={})",
                            done.response.id()
                        ))
                        .into()),
                    };
                }
                ResponsesWebSocketEvent::Error(error) => {
                    return Err(provider_error_from_event(&error).into());
                }
                ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {}
            }
        }
    }

    /// Refuse a send the session cannot take: closed, failed, a turn already
    /// in flight, or recovered frames the caller has not collected.
    fn ensure_can_send(&self) -> Result<(), ProviderError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(ProviderError::Provider(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        if !self.recovered.is_empty() {
            return Err(ProviderError::request(UncollectedRecoveredFrames {
                count: self.recovered.len(),
            }));
        }

        Ok(())
    }

    /// Shape a Rig request into this session's `response.create` body through
    /// Rig's own Responses encoder.
    ///
    /// The body is the HTTP body of the same request, which on the Codex
    /// contract states `stream: true` as the official client does on every
    /// frame, without the `background` flag an event-driven session ignores.
    /// A full send roots a fresh chain: no `previous_response_id` is added.
    fn prepare_request(
        &self,
        mut completion_request: CompletionRequest,
        options: &ResponsesWebSocketCreateOptions,
    ) -> Result<RequestBody, ProviderError> {
        // Direct session requests bypass builder validation.
        completion_request.validate_message_content()?;
        if let Some(control) = UnsupportedCodexControl::in_request(&completion_request) {
            return Err(control.into());
        }

        // Rig's encoder refuses a request with no input item. A warmup may
        // have none, so a stand-in message is encoded and then taken out.
        let input_free = options.generate == Some(false)
            && completion_request.documents.is_empty()
            && completion_request
                .chat_history
                .iter()
                .all(|message| matches!(**message, Message::System { .. }));
        if input_free {
            completion_request
                .chat_history
                .push(Arc::new(Message::user(WARMUP_PLACEHOLDER)));
        }

        let request = Completion::prepare(completion_request, &self.wire.wire.describe())?;
        let (_, mut body) = self.wire.encode_body(request, Mode::Streaming)?;
        if input_free {
            remove_warmup_placeholder(&mut body)?;
        }
        body.insert("stream", &true)?;
        body.remove("background");
        Ok(body)
    }

    /// Serialize one `response.create` frame for `request`, stamping the
    /// session's identity and frame metadata into it.
    fn encode_frame(
        &self,
        request: &RequestBody,
        generate: Option<bool>,
    ) -> Result<String, ProviderError> {
        let mut body = request.clone();
        if let Some(generate) = generate {
            body.insert("generate", &generate)?;
        }
        self.identity.stamp(&mut body)?;
        self.frame_stamps.stamp(&mut body)?;
        Ok(serde_json::to_string(&CreateFrame(&body))?)
    }

    /// Write one encoded `response.create` frame, taking custody of the turn.
    ///
    /// `envelope` is the full request the turn continues from: it becomes the
    /// eligible envelope for a later incremental continuation only if this
    /// turn completes.
    async fn send_encoded(
        &mut self,
        envelope: RequestBody,
        frame: String,
    ) -> Result<(), ProviderError> {
        self.ensure_can_send()?;
        // A session built over a connection someone else opened has had no
        // handshake of its own to refuse, so the identity rule is checked
        // again here, before any frame is written.
        self.wire.sendable_identity()?;

        if let Err(error) = self.socket.send(Frame::Text(frame)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        self.continuation = Continuation::InFlight(Box::new(envelope));
        self.in_flight = true;

        Ok(())
    }

    /// Reads the next server event for the current in-flight turn.
    ///
    /// A frame that fails to decode fails the session and is returned as
    /// [`CodexSessionError::CorruptFrame`], carrying the frame as received.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, CodexSessionError> {
        self.ensure_open()?;

        if !self.in_flight {
            return Err(ProviderError::Provider(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            )
            .into());
        }

        loop {
            let message = match self.read_next_frame().await? {
                Ok(message) => message,
                Err(error) => {
                    return Err(self.fail_session(websocket_provider_error(error)).into());
                }
            };

            let Some(message) = message else {
                self.mark_closed();
                return Err(ProviderError::Provider(
                    "The OpenAI websocket connection closed before the turn finished".to_string(),
                )
                .into());
            };

            let payload = match websocket_frame_to_text(message) {
                Ok(Some(payload)) => payload,
                Ok(None) => continue,
                Err(error) => {
                    self.mark_failed();
                    return Err(error);
                }
            };
            let event = match parse_server_event(&payload) {
                Ok(event) => event,
                Err(corrupt) => {
                    self.mark_failed();
                    return Err(corrupt.into());
                }
            };
            self.observe_turn_state(&event);
            if let ResponsesWebSocketEvent::Done(done) = &event {
                // OpenAI may emit `response.done` after the turn has already ended at
                // `response.completed`. Ignore that trailing event on the next turn.
                if self.pending_done_response_id.as_deref() == done.response_id() {
                    self.pending_done_response_id = None;
                    continue;
                }
            }
            self.update_state_for_event(&event);
            return Ok(event);
        }
    }

    /// Services the live connection between turns without consuming a response.
    ///
    /// The provider sends keepalive pings while a session is idle, and between
    /// turns nothing reads the socket. This takes every frame that has already
    /// arrived, letting the backend queue its automatic pong for each ping,
    /// then makes one bounded flush so those pongs reach the wire.
    ///
    /// It never reads a turn's response: it is a no-op while a turn is in
    /// flight (and after the session has closed or failed), it never sends
    /// `response.create`, and it never advances `previous_response_id`. The
    /// only modelled event it consumes is the trailing `response.done` for the
    /// turn that just ended, which [`Self::next_event`] would filter anyway;
    /// any other modelled event while idle is a protocol violation and fails
    /// the session rather than being discarded.
    ///
    /// Unmodelled events are returned in socket arrival order rather than
    /// dropped, and **a failure never costs the caller what was already
    /// consumed**: every exit (close, read error, parse error, unexpected
    /// modelled event, flood, flush error, flush timeout) returns the complete
    /// ordered prefix beside the failure.
    ///
    /// The whole call is bounded without an outer timeout: reads never wait on
    /// the peer, a frame budget bounds the read loop, and the one flush is
    /// bounded internally. Each recovered frame moves into the session the
    /// moment it is consumed, so even a caller that drops this future midway
    /// loses nothing: the next call returns those frames first, and a send is
    /// refused ([`UncollectedRecoveredFrames`]) until they are collected.
    ///
    /// A backend without
    /// [`recv_ready`](crate::ws_client::WebSocketConnection::recv_ready) or
    /// [`flush`](crate::ws_client::WebSocketConnection::flush) ends the drain
    /// with a [`ProviderError::Request`] naming the missing capability; the
    /// session is left open, since nothing was lost.
    pub async fn keepalive(&mut self) -> KeepaliveDrain {
        use crate::ws_client::WebSocketConnection as _;

        if self.closed || self.failed || self.in_flight {
            return KeepaliveDrain::serviced(std::mem::take(&mut self.recovered));
        }

        for _ in 0..MAX_KEEPALIVE_DRAIN_FRAMES {
            let frame = match self.socket.recv_ready().await {
                Ok(ReadyFrame::Frame(frame)) => frame,
                Ok(ReadyFrame::Empty) => return self.flush_after_drain().await,
                Ok(ReadyFrame::Ended) => {
                    self.mark_closed();
                    return self.drain_failed(
                        ProviderError::Provider(
                            "The OpenAI websocket connection closed during idle keepalive"
                                .to_string(),
                        )
                        .into(),
                    );
                }
                Err(error) if UnsupportedCapability::is(&error, "recv_ready") => {
                    return self.drain_failed(unsupported_capability_error("recv_ready").into());
                }
                Err(error) => {
                    let error = self.fail_session(websocket_provider_error(error));
                    return self.drain_failed(error.into());
                }
            };

            let payload = match websocket_frame_to_text(frame) {
                // A ping or pong carries no turn data; the backend has queued
                // the pong a ping is owed, and the flush below writes it.
                Ok(None) => continue,
                Ok(Some(payload)) => payload,
                Err(error) => {
                    self.mark_failed();
                    return self.drain_failed(error);
                }
            };

            let event = match parse_server_event(&payload) {
                Ok(event) => event,
                Err(corrupt) => {
                    self.mark_failed();
                    return self.drain_failed(corrupt.into());
                }
            };

            match event {
                // Unmodelled: collected into the session's custody, never
                // acted on. The caller decides what it means.
                ResponsesWebSocketEvent::Unknown(event) => self.recovered.push(event),
                // The trailing `response.done` for the turn that just ended is
                // the one modelled event expected between turns.
                ResponsesWebSocketEvent::Done(done)
                    if self.pending_done_response_id.as_deref() == done.response_id() =>
                {
                    self.pending_done_response_id = None;
                }
                // Real turn data with no turn in flight: fail loudly, and hand
                // back the unmodelled frames that preceded it.
                ResponsesWebSocketEvent::Response(_)
                | ResponsesWebSocketEvent::Item(_)
                | ResponsesWebSocketEvent::Error(_)
                | ResponsesWebSocketEvent::Done(_) => {
                    let error = self.fail_session(ProviderError::Provider(
                        "The OpenAI websocket delivered an unexpected server event during idle keepalive"
                            .to_string(),
                    ));
                    return self.drain_failed(error.into());
                }
            }
        }

        // The peer delivers faster than the drain consumes, so the socket
        // cannot be read to a known state. What was consumed is still the
        // caller's.
        let error = self.fail_session(keepalive_flood_error(MAX_KEEPALIVE_DRAIN_FRAMES));
        self.drain_failed(error.into())
    }

    /// The one bounded suspension after the reads: write out the pongs the
    /// drain's reads queued.
    async fn flush_after_drain(&mut self) -> KeepaliveDrain {
        use crate::ws_client::WebSocketConnection as _;

        match rig_core::wasm_compat::timeout(KEEPALIVE_FLUSH_TIMEOUT, self.socket.flush()).await {
            Ok(Ok(())) => KeepaliveDrain::serviced(std::mem::take(&mut self.recovered)),
            Ok(Err(error)) if UnsupportedCapability::is(&error, "flush") => {
                self.drain_failed(unsupported_capability_error("flush").into())
            }
            Ok(Err(error)) => {
                let error = self.fail_session(websocket_provider_error(error));
                self.drain_failed(error.into())
            }
            Err(_elapsed) => {
                let error =
                    self.fail_session(keepalive_flush_timeout_error(KEEPALIVE_FLUSH_TIMEOUT));
                self.drain_failed(error.into())
            }
        }
    }

    /// End a drain with `error`, surrendering every frame it recovered.
    fn drain_failed(&mut self, error: CodexSessionError) -> KeepaliveDrain {
        KeepaliveDrain::failed(std::mem::take(&mut self.recovered), error)
    }

    /// Closes the websocket connection with a close handshake.
    pub async fn close(&mut self) -> Result<(), ProviderError> {
        if self.closed {
            return Ok(());
        }

        let result = self
            .socket
            .close(None)
            .await
            .map_err(websocket_provider_error);
        self.mark_closed();
        result
    }

    /// Record the Codex sticky-routing token a `response.metadata` event
    /// carries in its `headers`, when the session holds none: the first
    /// token wins until it is taken, as the official client keeps the first
    /// one of a turn. A name is matched case-insensitively and a value is a
    /// string, or the first string of an array, as the official client
    /// reads them.
    fn observe_turn_state(&mut self, event: &ResponsesWebSocketEvent) {
        fn first_string(value: &Value) -> Option<&str> {
            match value {
                Value::String(value) => Some(value),
                Value::Array(items) => items.first().and_then(first_string),
                _ => None,
            }
        }
        if self.turn_state.is_some() {
            return;
        }
        let ResponsesWebSocketEvent::Unknown(event) = event else {
            return;
        };
        if event.kind != RESPONSE_METADATA_EVENT {
            return;
        }
        let Some(headers) = event
            .payload
            .value()
            .get("headers")
            .and_then(Value::as_object)
        else {
            return;
        };
        // Like the official client, a name whose value is not a string does
        // not end the search: another spelling of the name may carry one.
        self.turn_state = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case(TURN_STATE_METADATA_KEY))
            .find_map(|(_, value)| first_string(value))
            .map(str::to_owned);
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        let ending = match event {
            ResponsesWebSocketEvent::Response(chunk) => match chunk.kind {
                ResponseChunkKind::ResponseCompleted => {
                    let response_id = chunk.response.id.clone();
                    self.previous_response_id = Some(response_id.clone());
                    self.pending_done_response_id = Some(response_id.clone());
                    TurnEnding::Completed {
                        response_id: Some(response_id),
                    }
                }
                // An incomplete turn still produced a response, so it keeps
                // `previous_response_id` like a completed one. It is not an
                // eligible tip for an incremental continuation.
                ResponseChunkKind::ResponseIncomplete => {
                    let response_id = chunk.response.id.clone();
                    self.previous_response_id = Some(response_id.clone());
                    self.pending_done_response_id = Some(response_id.clone());
                    TurnEnding::Incomplete {
                        response_id: Some(response_id),
                    }
                }
                ResponseChunkKind::ResponseFailed => {
                    self.pending_done_response_id = Some(chunk.response.id.clone());
                    self.previous_response_id = None;
                    TurnEnding::NotCompleted
                }
                ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {
                    return;
                }
            },
            ResponsesWebSocketEvent::Done(done) => {
                let response_id = done.response_id().map(str::to_owned);
                let ending = match done.status() {
                    ResponseStatus::Completed => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Completed { response_id }
                    }
                    ResponseStatus::Incomplete => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Incomplete { response_id }
                    }
                    ResponseStatus::Failed
                    | ResponseStatus::Cancelled
                    | ResponseStatus::Other(_) => {
                        self.previous_response_id = None;
                        TurnEnding::NotCompleted
                    }
                    ResponseStatus::InProgress | ResponseStatus::Queued => TurnEnding::NotCompleted,
                };
                self.pending_done_response_id = None;
                ending
            }
            ResponsesWebSocketEvent::Error(_) => {
                self.previous_response_id = None;
                self.pending_done_response_id = None;
                TurnEnding::NotCompleted
            }
            // An unmodelled event carries no turn-lifecycle signal.
            ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => return,
        };
        self.in_flight = false;
        let current = std::mem::replace(
            &mut self.continuation,
            Continuation::Blocked(IncrementalSendRefused::LastTurnNotCompleted),
        );
        self.continuation = current.settle(ending);
    }

    fn abort_turn(&mut self) {
        self.previous_response_id = None;
        self.pending_done_response_id = None;
        self.continuation = Continuation::Blocked(IncrementalSendRefused::LastTurnNotCompleted);
        self.in_flight = false;
    }

    fn mark_closed(&mut self) {
        self.abort_turn();
        self.closed = true;
        self.failed = false;
    }

    fn mark_failed(&mut self) {
        self.abort_turn();
        self.failed = true;
    }

    fn ensure_open(&self) -> Result<(), ProviderError> {
        if self.closed || self.failed {
            return Err(ProviderError::Provider(
                "The OpenAI websocket session is closed".to_string(),
            ));
        }

        Ok(())
    }

    fn fail_session(&mut self, error: ProviderError) -> ProviderError {
        self.mark_failed();
        error
    }

    /// Read a frame with a WASM-compatible timeout.
    /// Timeout failure marks the session failed; transport results remain nested.
    async fn read_next_frame(
        &mut self,
    ) -> Result<http_client::Result<Option<Frame>>, ProviderError> {
        let Some(timeout_duration) = self.event_timeout else {
            return Ok(self.socket.recv().await);
        };

        match rig_core::wasm_compat::timeout(timeout_duration, self.socket.recv()).await {
            Ok(message) => Ok(message),
            Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
        }
    }
}

impl Drop for CodexWebSocketSession {
    fn drop(&mut self) {
        if !self.closed {
            tracing::warn!(
                target: "rig::completions",
                in_flight = self.in_flight,
                "Dropping an OpenAI websocket session without calling close(); the connection will end without a close handshake"
            );
        }
    }
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

/// One `response.create` frame: the event tag, then the body's fields.
struct CreateFrame<'a>(&'a RequestBody);

impl Serialize for CreateFrame<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len().saturating_add(1)))?;
        map.serialize_entry("type", "response.create")?;
        self.0.serialize_into(&mut map)?;
        map.end()
    }
}

/// Take the warmup stand-in back out of an encoded body's `input`, leaving
/// the input the request really has.
fn remove_warmup_placeholder(body: &mut RequestBody) -> Result<(), EncodeError> {
    let misplaced = || {
        EncodeError::request(
            "the Responses encoder did not place the warmup stand-in last in `input`",
        )
    };
    let mut input: Vec<Box<RawValue>> = body
        .get("input")
        .ok_or_else(misplaced)?
        .map_err(EncodeError::request)?;
    let stand_in: Value = input
        .pop()
        .ok_or_else(misplaced)
        .and_then(|item| serde_json::from_str(item.get()).map_err(EncodeError::request))?;
    let is_stand_in = stand_in.get("role").and_then(Value::as_str) == Some("user")
        && stand_in.pointer("/content/0/text").and_then(Value::as_str) == Some(WARMUP_PLACEHOLDER);
    if !is_stand_in {
        return Err(misplaced());
    }
    body.insert("input", &input).map_err(EncodeError::request)
}

fn event_timeout_error(timeout: Duration) -> ProviderError {
    ProviderError::Provider(format!(
        "Timed out waiting for the next OpenAI websocket event after {timeout:?}"
    ))
}

fn keepalive_flush_timeout_error(timeout: Duration) -> ProviderError {
    ProviderError::Provider(format!(
        "Timed out flushing the OpenAI websocket idle keepalive after {timeout:?}"
    ))
}

fn keepalive_flood_error(budget: usize) -> ProviderError {
    ProviderError::Provider(format!(
        "The OpenAI websocket delivered at least {budget} buffered frames during idle keepalive"
    ))
}

/// A backend capability the keepalive needs and the backend lacks.
fn unsupported_capability_error(capability: &'static str) -> ProviderError {
    ProviderError::request(UnsupportedCapability { capability })
}

/// Native sessions and builders satisfy Send and Sync.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CodexWebSocketSession>();
    assert_send_sync::<CodexWebSocketSessionBuilder>();
    assert_send_sync::<KeepaliveDrain>();
};

#[cfg(test)]
mod tests;
