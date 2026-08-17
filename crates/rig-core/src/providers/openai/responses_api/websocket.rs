//! WebSocket session support for the OpenAI Responses API.
//!
//! This module implements OpenAI's `/v1/responses` WebSocket mode as a stateful,
//! sequential session. Each connection supports a single in-flight response at a
//! time, which matches OpenAI's current protocol constraints.

use crate::completion::NormalizeCompletionResponse;
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::providers::internal::adapter::{TriagedFrame, triage_frame};
use crate::providers::openai::responses_api::streaming::{
    ItemChunk, RawChoiceAccumulator, ResponseChunk, ResponseChunkKind, ResponsesStreamOptions,
    StreamingCompletionChunk, classify_responses_frame, completion_response_from_raw_choices,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::{FutureExt, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async, connect_async_tls_with_config,
    tungstenite::{self, Message, client::IntoClientRequest},
};
use url::Url;

use super::{CompletionResponse, ResponseStatus, ResponsesCompletionModel, ResponsesUsage};

type OpenAIWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WebSocketRawChoice = crate::streaming::RawStreamingChoice<
    crate::providers::openai::responses_api::streaming::StreamingCompletionResponse,
>;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The most frames one idle keepalive drain consumes before it stops and hands
/// back what it has taken.
///
/// The drain's reads never suspend — see
/// [`keepalive`](ResponsesWebSocketSession::keepalive) — so a peer delivering
/// faster than the loop consumes is the only way the loop fails to terminate
/// promptly. A frame budget bounds it without introducing an await, which a
/// timer would require; an await inside the read loop is exactly what would put
/// the already-consumed frames at risk of being dropped by a cancellation.
const MAX_KEEPALIVE_DRAIN_FRAMES: usize = 4_096;

const _: () = assert!(
    MAX_KEEPALIVE_DRAIN_FRAMES > 0,
    "a zero budget would consume nothing and report a flood on every idle drain"
);

/// How long the post-drain pong flush may take before the socket is treated as
/// unserviceable.
///
/// This bound is internal on purpose. It is the one suspension point in the
/// whole drain, and it holds the recovered frames, so bounding it from the
/// outside with a `timeout`/`select!` would cancel the future and destroy them.
/// Bounded here, expiry *returns* them alongside the failure instead.
const KEEPALIVE_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// An explicit TLS connector for the Responses WebSocket connection.
///
/// Wraps a [`tokio_tungstenite::Connector`] so callers can inject a custom TLS
/// configuration (for example a pre-built [`rustls::ClientConfig`]) instead of
/// relying on the default connector that `tokio-tungstenite` builds internally.
/// This is the seam hosts use to supply their own crypto provider and trust
/// roots without Rig having to construct a default `ClientConfig` on the connect
/// path.
#[derive(Clone)]
pub struct WebSocketTlsConnector(Connector);

impl WebSocketTlsConnector {
    /// Builds a connector from an explicit rustls client configuration.
    pub fn rustls(config: std::sync::Arc<rustls::ClientConfig>) -> Self {
        Self(Connector::Rustls(config))
    }

    fn into_connector(self) -> Connector {
        self.0
    }
}

impl std::fmt::Debug for WebSocketTlsConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("WebSocketTlsConnector").finish()
    }
}

/// Provider seam for a Responses WebSocket session.
///
/// A backend describes how to reach a Responses-compatible provider and how to
/// build its `response.create` payload: the connection target, the async
/// handshake headers, and provider-specific request shaping. The
/// [`ResponsesWebSocketSession`] itself owns the transport, event parsing, and
/// turn state, so a backend stays small.
///
/// The trait is intentionally not object-safe — the session is generic over the
/// backend (`ResponsesWebSocketSession<B>`) and dispatches statically — so the
/// async method uses return-position `impl Future` rather than `async_trait`.
pub trait ResponsesWebSocketBackend: WasmCompatSend + WasmCompatSync {
    /// The HTTP(S) base URL the session converts into a `ws://`/`wss://` endpoint.
    fn base_url(&self) -> &str;

    /// Shapes a Rig completion request into the provider's Responses request.
    fn shape_request(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<super::CompletionRequest, CompletionError>;

    /// Builds the WebSocket handshake headers, awaiting any async auth work.
    fn handshake_headers(
        &self,
    ) -> impl std::future::Future<Output = Result<http::HeaderMap, CompletionError>> + WasmCompatSend;

    /// The provider identity stamped onto normalized responses.
    ///
    /// `U`'s `completion` read this off a concrete `ResponsesCompletionModel`
    /// field, which `F`'s backend-parametric session does not have. Identity is
    /// a property of the backend, so it is asked for here — that keeps the
    /// session generic and stops `openai` being hard-coded on a path ChatGPT
    /// also travels. Deliberately without a default: a default would let a new
    /// backend silently inherit another provider's identity.
    fn provider_name(&self) -> &'static str;

    /// Whether the session auto-chains turns via `previous_response_id`.
    ///
    /// OpenAI's Responses WebSocket mode chains by default; replay-style backends
    /// override this to keep each turn independent.
    fn chains_previous_response_id(&self) -> bool {
        true
    }
}

/// The default [`ResponsesWebSocketBackend`] backing OpenAI's Responses WebSocket mode.
///
/// Wraps a [`ResponsesCompletionModel`] so the session reaches OpenAI through the
/// model's configured client (base URL and static auth headers) and shapes
/// requests with the model's Responses request conversion.
pub struct OpenAIResponsesWebSocketBackend<H = reqwest::Client> {
    model: ResponsesCompletionModel<H>,
}

impl<H> OpenAIResponsesWebSocketBackend<H> {
    pub(crate) fn new(model: ResponsesCompletionModel<H>) -> Self {
        Self { model }
    }
}

impl<H> ResponsesWebSocketBackend for OpenAIResponsesWebSocketBackend<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    fn base_url(&self) -> &str {
        self.model.client.base_url()
    }

    fn shape_request(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<super::CompletionRequest, CompletionError> {
        self.model.create_completion_request(request)
    }

    async fn handshake_headers(&self) -> Result<http::HeaderMap, CompletionError> {
        // OpenAI's auth headers are static, so no async work is needed here; the
        // async signature exists for backends (such as ChatGPT/Codex) that refresh
        // credentials before each connect.
        Ok(self.model.client.headers().clone())
    }

    fn provider_name(&self) -> &'static str {
        self.model.provider_name()
    }
}

/// Options for a `response.create` message sent over OpenAI WebSocket mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesWebSocketCreateOptions {
    /// When set to `false`, OpenAI prepares request state without generating a model output.
    ///
    /// This is the "warmup" mode described in the OpenAI WebSocket mode guide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<bool>,
}

impl ResponsesWebSocketCreateOptions {
    /// Creates warmup options equivalent to `generate: false`.
    #[must_use]
    pub fn warmup() -> Self {
        Self {
            generate: Some(false),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ResponsesWebSocketClientEvent {
    #[serde(rename = "type")]
    kind: ResponsesWebSocketClientEventKind,
    #[serde(flatten)]
    request: super::CompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
enum ResponsesWebSocketClientEventKind {
    #[serde(rename = "response.create")]
    ResponseCreate,
}

/// A protocol error event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSocketErrorEvent {
    /// The event type.
    #[serde(rename = "type")]
    pub kind: ResponsesWebSocketErrorEventKind,
    /// The provider error payload.
    pub error: ResponsesWebSocketErrorPayload,
}

impl std::fmt::Display for ResponsesWebSocketErrorEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

/// The event kind for an OpenAI WebSocket protocol error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsesWebSocketErrorEventKind {
    #[serde(rename = "error")]
    Error,
}

/// The payload carried by an OpenAI WebSocket protocol error event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesWebSocketErrorPayload {
    /// Provider-specific error code when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Any extra fields supplied by the provider.
    #[serde(flatten, default)]
    pub extra: Map<String, Value>,
}

impl std::fmt::Display for ResponsesWebSocketErrorPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.code, &self.message) {
            (Some(code), Some(message)) => write!(f, "{code}: {message}"),
            (None, Some(message)) => f.write_str(message),
            (Some(code), None) => f.write_str(code),
            (None, None) => f.write_str("OpenAI websocket error"),
        }
    }
}

/// The optional `response.done` event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSocketDoneEvent {
    /// The event type.
    #[serde(rename = "type")]
    pub kind: ResponsesWebSocketDoneEventKind,
    /// The provider payload for the finished response.
    pub response: Value,
}

impl ResponsesWebSocketDoneEvent {
    /// Returns the response ID if the payload includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        self.response.get("id").and_then(Value::as_str)
    }

    fn status(&self) -> Option<ResponseStatus> {
        self.response
            .get("status")
            .cloned()
            .and_then(|status| serde_json::from_value(status).ok())
    }

    fn as_completion_response(&self) -> Option<CompletionResponse> {
        serde_json::from_value(self.response.clone()).ok()
    }
}

/// The event kind for the terminal websocket event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsesWebSocketDoneEventKind {
    #[serde(rename = "response.done")]
    ResponseDone,
}

/// A server event emitted by OpenAI WebSocket mode.
#[derive(Debug, Clone)]
pub enum ResponsesWebSocketEvent {
    /// A response lifecycle event such as `response.created` or `response.completed`.
    Response(Box<ResponseChunk>),
    /// A streaming item/delta event such as `response.output_text.delta`.
    Item(ItemChunk),
    /// A protocol-level websocket error event.
    Error(ResponsesWebSocketErrorEvent),
    /// An optional `response.done` event emitted by OpenAI over WebSockets.
    Done(ResponsesWebSocketDoneEvent),
    /// A server event whose `type` this client does not model, surfaced with its
    /// parsed JSON payload rather than discarded.
    ///
    /// Consumers that reconstruct the turn's canonical text cannot treat an
    /// unmodelled event as absent: it may carry output this client would
    /// otherwise silently drop. Reporting it lets such a consumer fail closed on
    /// something it cannot place. `payload` is the complete parsed JSON value of
    /// the server event — every field, not a modelled subset. It is not the raw
    /// bytes: parsing discards lexical formatting and may normalize representation,
    /// so this preserves the event's JSON *value*, not its wire spelling.
    ///
    /// D2: the name is `U`'s, the retained `kind` tag and its `KeepaliveDrain`
    /// contract are `F`'s, and the payload is `U`'s non-printing newtype —
    /// which is why this composes rather than choosing a side. `U` also
    /// forwards these onto `RawStreamingChoice::Unknown`; `F` declined to,
    /// because at `M` no higher-level path consumed them. That is no longer
    /// true, so the forwarding is adopted.
    Unknown(UnrecognizedEvent),
}

/// An outer server event this client does not model.
///
/// A named value rather than inline variant fields because it is returned on its
/// own by [`ResponsesWebSocketSession::keepalive`]. That signature is the reason
/// it exists: a `Vec<UnrecognizedEvent>` says in the type that every element is
/// an unmodelled event, where a `Vec<ResponsesWebSocketEvent>` would oblige the
/// caller to re-match variants the drain has already decided.
#[derive(Debug, Clone, PartialEq)]
pub struct UnrecognizedEvent {
    /// The event's `type` field.
    ///
    /// `F`'s addition, and the reason this is a named struct: a
    /// `KeepaliveDrain` consumer failing closed on an unplaceable frame must
    /// be able to say what it was.
    pub kind: String,
    /// The complete parsed JSON value of the event.
    ///
    /// `U`'s newtype, whose `Debug` prints structural metadata only — an
    /// unmodelled frame can carry model output, and a `?payload` capture in a
    /// log was a recurring leak class. Content is opt-in via
    /// [`UnknownPayload::value`](crate::streaming::UnknownPayload::value).
    pub payload: crate::streaming::UnknownPayload,
}

/// The new input for a forward-only incremental turn — never empty.
///
/// `send_incremental` chains onto a live tip and replaces the cached envelope's
/// `input` wholesale, so an empty delta would send a chained `response.create`
/// carrying nothing new. An incremental turn must carry new input: that is what
/// makes it a continuation rather than a re-send of the tip.
///
/// `U` removed `OneOrMany` and moved the analogous request-level guarantee to a
/// *rule* checked by
/// [`CompletionRequest::validate_message_content`](crate::completion::CompletionRequest::validate_message_content).
/// That rule cannot cover this path: an incremental turn never builds a
/// [`crate::completion::CompletionRequest`], it mutates a cached provider
/// envelope. So the invariant lives in this type, which asserts nothing else.
///
/// The field is private, so the constructors below are the only way in. There
/// is deliberately no `Deserialize`: a derived one would bypass them, and the
/// type is never serialized anyway — only the `Vec` it surrenders to
/// [`Self::into_vec`] rides on the wire, inside the request envelope.
#[derive(Debug, Clone)]
pub struct InputDelta(Vec<super::InputItem>);

/// Returned when an incremental delta would carry no input items.
#[derive(Debug, thiserror::Error)]
#[error("an incremental delta must carry at least one input item")]
pub struct EmptyInputDeltaError;

impl InputDelta {
    /// A delta of exactly one item.
    ///
    /// Infallible by construction, so the common single-item case needs no
    /// error handling at all.
    #[must_use]
    pub fn one(item: super::InputItem) -> Self {
        Self(vec![item])
    }

    /// A delta from many items, rejecting the empty case.
    pub fn new(items: Vec<super::InputItem>) -> Result<Self, EmptyInputDeltaError> {
        if items.is_empty() {
            return Err(EmptyInputDeltaError);
        }
        Ok(Self(items))
    }

    /// The items, surrendered to the caller.
    ///
    /// Consuming rather than borrowing: the only consumer moves these straight
    /// into the request envelope, and handing out a slice would invite callers
    /// to rebuild a `Vec` that could then be emptied.
    #[must_use]
    pub fn into_vec(self) -> Vec<super::InputItem> {
        self.0
    }
}

impl TryFrom<Vec<super::InputItem>> for InputDelta {
    type Error = EmptyInputDeltaError;

    fn try_from(items: Vec<super::InputItem>) -> Result<Self, Self::Error> {
        Self::new(items)
    }
}

/// How far an idle [`keepalive`](ResponsesWebSocketSession::keepalive) drain
/// got, and every unmodelled frame it consumed getting there.
///
/// **Deliberately not a `Result`.** The recovered frames have already been taken
/// off the socket and can never be re-read, so a shape with an error position
/// would let `?` — or any `From` conversion a later caller adds — return the
/// failure while silently discarding them. A caller that lost them this way
/// could not tell a socket that closed after delivering an unaccountable frame
/// from one that merely closed, which is the difference between a proven and an
/// unproven session. There is no error position here, and the ending is
/// reachable only through [`Self::into_parts`], which surrenders the frames in
/// the same expression.
#[derive(Debug)]
pub struct KeepaliveDrain {
    recovered: Vec<UnrecognizedEvent>,
    ending: KeepaliveEnding,
}

/// How a [`KeepaliveDrain`] ended.
///
/// Private: the two states are total, but the public surface is
/// [`KeepaliveDrain::into_parts`], which hands the ending back as an `Option`
/// beside the frames. Keeping the enum internal is what stops a caller from
/// destructuring an ending on its own and leaving the frames behind.
#[derive(Debug)]
enum KeepaliveEnding {
    /// Read to the end of what was buffered, and the queued pongs reached the
    /// wire.
    Serviced,
    /// The drain stopped here, and the session has been marked failed or closed.
    Failed(CompletionError),
}

impl KeepaliveDrain {
    fn serviced(recovered: Vec<UnrecognizedEvent>) -> Self {
        Self {
            recovered,
            ending: KeepaliveEnding::Serviced,
        }
    }

    fn failed(recovered: Vec<UnrecognizedEvent>, error: CompletionError) -> Self {
        Self {
            recovered,
            ending: KeepaliveEnding::Failed(error),
        }
    }

    /// The unmodelled frames the drain consumed, in socket arrival order.
    #[must_use]
    pub fn recovered(&self) -> &[UnrecognizedEvent] {
        &self.recovered
    }

    /// Split into the recovered frames and the failure that ended the drain, if
    /// one did.
    ///
    /// The only way to read the ending, so a caller that wants the failure
    /// necessarily receives everything consumed before it. Frames recovered
    /// ahead of a failure are ordinary recovered frames: the failure says the
    /// socket stopped being readable, never that they did not arrive.
    #[must_use]
    pub fn into_parts(self) -> (Vec<UnrecognizedEvent>, Option<CompletionError>) {
        let ending = match self.ending {
            KeepaliveEnding::Serviced => None,
            KeepaliveEnding::Failed(error) => Some(error),
        };
        (self.recovered, ending)
    }
}

impl ResponsesWebSocketEvent {
    /// Returns the response ID when the event includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        match self {
            Self::Response(chunk) => Some(&chunk.response.id),
            Self::Done(done) => done.response_id(),
            Self::Item(_) | Self::Error(_) | Self::Unknown(_) => None,
        }
    }

    /// Returns `true` when this event ends the current in-flight websocket turn.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Response(chunk) => matches!(
                chunk.kind,
                ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseFailed
                    | ResponseChunkKind::ResponseIncomplete
            ),
            Self::Error(_) | Self::Done(_) => true,
            // An unmodelled event ends nothing at the protocol level; whether it
            // is tolerable is the consumer's decision, not this predicate's.
            Self::Item(_) | Self::Unknown(_) => false,
        }
    }
}

/// A builder for an OpenAI Responses WebSocket session.
///
/// The default builder applies a 30 second connection timeout and leaves the
/// per-event timeout disabled.
pub struct ResponsesWebSocketSessionBuilder<B = OpenAIResponsesWebSocketBackend> {
    backend: B,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
    tls_connector: Option<WebSocketTlsConnector>,
}

impl<B> ResponsesWebSocketSessionBuilder<B> {
    pub(crate) fn new(backend: B) -> Self {
        Self {
            backend,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
            tls_connector: None,
        }
    }

    /// Sets an explicit TLS connector for establishing the websocket connection.
    ///
    /// When unset, the default `tokio-tungstenite` connector is used and Rig makes
    /// a best-effort installation of a default rustls crypto provider before
    /// connecting.
    #[must_use]
    pub fn tls_connector(mut self, connector: WebSocketTlsConnector) -> Self {
        self.tls_connector = Some(connector);
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
}

impl<B> ResponsesWebSocketSessionBuilder<B>
where
    B: ResponsesWebSocketBackend,
{
    /// Opens the websocket session using the configured builder options.
    pub async fn connect(self) -> Result<ResponsesWebSocketSession<B>, CompletionError> {
        ResponsesWebSocketSession::connect_with_timeouts(
            self.backend,
            self.connect_timeout,
            self.event_timeout,
            self.tls_connector,
        )
        .await
    }
}

/// A stateful OpenAI Responses WebSocket session.
///
/// This session keeps track of the most recent successful `response.id` so later
/// turns can automatically chain via `previous_response_id` unless the request
/// explicitly sets a different one.
///
/// Call [`ResponsesWebSocketSession::close`] when you are finished with the
/// session so the websocket can complete a close handshake cleanly.
pub struct ResponsesWebSocketSession<B = OpenAIResponsesWebSocketBackend> {
    backend: B,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    /// The non-input request envelope/config captured from the last successful
    /// full-replay turn, reused by forward-only incremental continuations.
    last_envelope: Option<super::CompletionRequest>,
    /// The envelope of the turn currently in flight, promoted to `last_envelope`
    /// when that turn completes successfully and dropped if it does not.
    pending_envelope: Option<super::CompletionRequest>,
    socket: OpenAIWebSocket,
    in_flight: bool,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl<B> ResponsesWebSocketSession<B>
where
    B: ResponsesWebSocketBackend,
{
    async fn connect_with_timeouts(
        backend: B,
        connect_timeout: Option<Duration>,
        event_timeout: Option<Duration>,
        tls_connector: Option<WebSocketTlsConnector>,
    ) -> Result<Self, CompletionError> {
        let url = websocket_url(backend.base_url())?;
        let headers = backend.handshake_headers().await?;
        let request = websocket_request(&url, &headers)?;
        let socket = connect_websocket(request, connect_timeout, tls_connector).await?;

        Ok(Self {
            backend,
            previous_response_id: None,
            pending_done_response_id: None,
            last_envelope: None,
            pending_envelope: None,
            socket,
            in_flight: false,
            event_timeout,
            closed: false,
            failed: false,
        })
    }

    /// Returns the most recent successful `response.id` tracked by this session.
    #[must_use]
    pub fn previous_response_id(&self) -> Option<&str> {
        self.previous_response_id.as_deref()
    }

    /// Clears the cached `previous_response_id` so the next turn starts a fresh chain.
    pub fn clear_previous_response_id(&mut self) {
        self.previous_response_id = None;
    }

    /// Sends a `response.create` event for a Rig completion request.
    pub async fn send(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::default(),
        )
        .await
    }

    /// Sends a `response.create` event with explicit websocket-mode options.
    pub async fn send_with_options(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
        options: ResponsesWebSocketCreateOptions,
    ) -> Result<(), CompletionError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(CompletionError::ProviderError(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        // The session takes a raw `CompletionRequest`, bypassing the builder's
        // `send`/`stream` — so this is a direct-to-model surface and validates
        // here, per `validate_message_content`'s own contract. Every session
        // entry point (`send`, `warmup`, `completion`, `raw_completion`)
        // funnels through this method.
        //
        // Validation precedes `F`'s backend shaping deliberately: the spec
        // requires an invalid message/history/tool-result payload to be
        // rejected before either backend shaping or socket I/O.
        completion_request.validate_message_content()?;

        let request = self.prepare_request(completion_request)?;

        let payload = ResponsesWebSocketClientEvent {
            kind: ResponsesWebSocketClientEventKind::ResponseCreate,
            request: request.clone(),
            generate: options.generate,
        };

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI websocket request",
            &payload,
        );

        let payload = serde_json::to_string(&payload)?;

        if let Err(error) = self.socket.send(Message::text(payload)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        // Capture this turn's non-input envelope/config so a later incremental
        // continuation can reuse it; it is promoted to `last_envelope` only once
        // the turn completes successfully.
        self.pending_envelope = Some(request);
        self.in_flight = true;

        Ok(())
    }

    /// Sends a forward-only incremental continuation of the current live tip.
    ///
    /// Rebuilds the `response.create` frame from the envelope captured by the last
    /// successful full-replay turn, replacing `input` with exactly `delta` and
    /// injecting the current `previous_response_id` regardless of whether the
    /// backend auto-chains. It establishes nothing on its own: it requires both a
    /// captured envelope and a live tip, and never falls back to full replay.
    pub(crate) async fn send_incremental_frame(
        &mut self,
        delta: InputDelta,
    ) -> Result<(), CompletionError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(CompletionError::ProviderError(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        let previous_response_id = self.previous_response_id.clone().ok_or_else(|| {
            CompletionError::ProviderError(
                "Cannot send an incremental turn before a completed send() established a live tip"
                    .to_string(),
            )
        })?;
        let mut request = self.last_envelope.clone().ok_or_else(|| {
            CompletionError::ProviderError(
                "Cannot send an incremental turn before a completed send() captured a request envelope"
                    .to_string(),
            )
        })?;

        request.input = delta.into_vec();
        request.additional_parameters.previous_response_id = Some(previous_response_id);

        let payload = ResponsesWebSocketClientEvent {
            kind: ResponsesWebSocketClientEventKind::ResponseCreate,
            request,
            generate: None,
        };

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI websocket incremental request: {}",
                serde_json::to_string_pretty(&payload)?
            );
        }

        let payload = serde_json::to_string(&payload)?;

        if let Err(error) = self.socket.send(Message::text(payload)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        // The incremental turn reuses the captured envelope, so `last_envelope`
        // must persist across the continuation; leave `pending_envelope` unset so
        // completion does not overwrite it.
        self.in_flight = true;

        Ok(())
    }

    /// Reads the next server event for the current in-flight turn.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, CompletionError> {
        self.ensure_open()?;

        if !self.in_flight {
            return Err(CompletionError::ProviderError(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            ));
        }

        loop {
            let message = match self.read_next_message().await {
                Ok(message) => message,
                Err(error) => return Err(error),
            };

            let Some(message) = message else {
                self.mark_closed();
                return Err(CompletionError::ProviderError(
                    "The OpenAI websocket connection closed before the turn finished".to_string(),
                ));
            };

            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.fail_session(websocket_provider_error(error))),
            };
            let payload = match websocket_message_to_text(message) {
                Ok(Some(payload)) => payload,
                Ok(None) => continue,
                Err(error) => return Err(self.fail_session(error)),
            };
            let event = match parse_server_event(&payload) {
                Ok(Some(event)) => event,
                Ok(None) => continue,
                Err(error) => return Err(self.fail_session(error)),
            };
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

    /// Services the live websocket between turns without consuming a response.
    ///
    /// OpenAI's Responses websocket sends server keepalive pings while a session
    /// is idle. Between turns nothing polls the socket, so those pings go
    /// unanswered and the provider eventually closes the connection with a
    /// keepalive ping timeout. This drains every frame that is already buffered
    /// — letting `tokio-tungstenite` enqueue its automatic pong replies — and
    /// then flushes the sink so those pongs reach the wire.
    ///
    /// It never reads a turn's response: it is a no-op while a turn is in flight
    /// (and after the session has closed or failed), it never sends
    /// `response.create`, and it never advances `previous_response_id`, so it can
    /// neither root nor advance the live tip. The only server event it consumes
    /// is the trailing `response.done` for the turn that just completed (the same
    /// event [`next_event`](Self::next_event) filters); any other *modelled*
    /// server event arriving while idle is a protocol violation, so it fails the
    /// session loudly rather than silently discarding a semantic frame.
    ///
    /// Unmodelled events are returned rather than dropped, in socket arrival
    /// order. They were previously skipped here, which made this a reader that
    /// could discard a frame a consumer needed to place — an unmodelled
    /// `response.*` arriving after the terminal chunk carries turn data no caller
    /// could then account for. Returning them keeps the drain's tolerance (an
    /// unknown event still never fails an idle socket) while leaving the decision
    /// with the caller, which is the same split [`next_event`](Self::next_event)
    /// already makes. An empty [`KeepaliveDrain::recovered`] means nothing
    /// unmodelled was buffered.
    ///
    /// **A failure never costs the caller what was already consumed.** Every
    /// exit — close, read error, parse error, unexpected modelled event, flood,
    /// flush error, flush timeout — returns a [`KeepaliveDrain`] carrying the
    /// complete ordered prefix taken before it. Returning a bare error instead
    /// is what made a consumed unaccountable frame followed by a socket close
    /// indistinguishable from an ordinary transport fault.
    ///
    /// **Callers must not add an outer bound to protect that prefix, because
    /// doing so is what would lose it.** The reads use `now_or_never` and
    /// therefore never suspend, so the read loop has no cancellation point
    /// between consuming a frame and returning it; it terminates on
    /// [`MAX_KEEPALIVE_DRAIN_FRAMES`] rather than a timer for that exact reason.
    /// The single suspension is the pong flush, bounded internally by
    /// [`KEEPALIVE_FLUSH_TIMEOUT`]. The whole operation is therefore already
    /// bounded, and wrapping it in a caller-side `timeout` or racing it in a
    /// `select!` would only reintroduce the drop this shape exists to prevent:
    /// `flush_pongs` owns the recovered frames by value across that await, so a
    /// caller that drops the future there loses them with no [`KeepaliveDrain`]
    /// to return them. The guarantee is unconditional for callers that observe
    /// this contract, not for cancellation at an arbitrary point.
    pub async fn keepalive(&mut self) -> KeepaliveDrain {
        if self.closed || self.failed || self.in_flight {
            return KeepaliveDrain::serviced(Vec::new());
        }

        let mut unrecognized = Vec::new();

        // Drain only frames that are already buffered so an idle socket with
        // nothing to read returns immediately instead of blocking; reading a
        // server ping lets `tokio-tungstenite` enqueue its automatic pong.
        for _ in 0..MAX_KEEPALIVE_DRAIN_FRAMES {
            let Some(message) = self.socket.next().now_or_never() else {
                return self.flush_pongs(unrecognized).await;
            };

            let Some(message) = message else {
                self.mark_closed();
                return KeepaliveDrain::failed(
                    unrecognized,
                    CompletionError::ProviderError(
                        "The OpenAI websocket connection closed during idle keepalive".to_string(),
                    ),
                );
            };

            let message = match message {
                Ok(message) => message,
                Err(error) => {
                    let error = self.fail_session(websocket_provider_error(error));
                    return KeepaliveDrain::failed(unrecognized, error);
                }
            };

            match websocket_message_to_text(message) {
                // Ping/pong/control frames carry no turn data; keep draining.
                Ok(None) => continue,
                Ok(Some(payload)) => {
                    let event = match parse_server_event(&payload) {
                        Ok(Some(event)) => event,
                        Ok(None) => continue,
                        Err(error) => {
                            let error = self.fail_session(error);
                            return KeepaliveDrain::failed(unrecognized, error);
                        }
                    };
                    // An unmodelled event is collected rather than acted on: this
                    // session still cannot place it, but discarding it here is
                    // what let a trailing frame vanish between turns. Idle
                    // tolerance is unchanged — it does not fail the drain — and
                    // the caller decides what the frame means.
                    if let ResponsesWebSocketEvent::Unknown(event) = event {
                        unrecognized.push(event);
                        continue;
                    }
                    // The trailing `response.done` for the just-finished turn is
                    // the only server event expected between turns; consume it
                    // exactly as `next_event` does and keep the tip untouched.
                    if let ResponsesWebSocketEvent::Done(done) = &event {
                        if self.pending_done_response_id.as_deref() == done.response_id() {
                            self.pending_done_response_id = None;
                            continue;
                        }
                    }
                    // Any other server event is real turn data with no turn in
                    // flight. Fail loudly rather than discard a semantic frame —
                    // and hand back the unmodelled frames that preceded it, which
                    // is the case where losing them mattered most: an unexpected
                    // modelled event is itself evidence the projection is broken.
                    let error = self.fail_session(CompletionError::ProviderError(
                        "The OpenAI websocket delivered an unexpected server event during idle keepalive"
                            .to_string(),
                    ));
                    return KeepaliveDrain::failed(unrecognized, error);
                }
                Err(error) => {
                    let error = self.fail_session(error);
                    return KeepaliveDrain::failed(unrecognized, error);
                }
            }
        }

        // The peer is delivering faster than this loop consumes. The socket
        // cannot be read to a known state, so it is not serviceable — but what
        // was consumed still belongs to the caller.
        let error = self.fail_session(keepalive_flood_error(MAX_KEEPALIVE_DRAIN_FRAMES));
        KeepaliveDrain::failed(unrecognized, error)
    }

    /// Flush so any pong enqueued by the drain reaches the server.
    ///
    /// Takes the recovered frames by value because this is the one place the
    /// drain suspends, and the bound is inside: expiry returns them with a typed
    /// failure rather than abandoning them. A stalled peer can leave a write
    /// buffer unflushable indefinitely, and this session is serviced from a
    /// single caller-side task, so an unbounded wait here is not a slow
    /// keepalive — it is a socket that never answers again.
    async fn flush_pongs(&mut self, recovered: Vec<UnrecognizedEvent>) -> KeepaliveDrain {
        match tokio::time::timeout(KEEPALIVE_FLUSH_TIMEOUT, self.socket.flush()).await {
            Ok(Ok(())) => KeepaliveDrain::serviced(recovered),
            Ok(Err(error)) => {
                let error = self.fail_session(websocket_provider_error(error));
                KeepaliveDrain::failed(recovered, error)
            }
            Err(_elapsed) => {
                let error =
                    self.fail_session(keepalive_flush_timeout_error(KEEPALIVE_FLUSH_TIMEOUT));
                KeepaliveDrain::failed(recovered, error)
            }
        }
    }

    /// Sends a warmup turn (`generate: false`) and returns the resulting response ID.
    pub async fn warmup(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<String, CompletionError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        let response = self.wait_for_completed_response().await?;
        Ok(response.id)
    }

    /// Sends a completion turn and collects the final OpenAI response,
    /// normalized.
    ///
    /// Use [`ResponsesWebSocketSession::raw_completion`] when the provider's own
    /// wire response is needed.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let provider = self.backend.provider_name();
        self.send(completion_request).await?;
        let (response, raw_choices) = self.wait_for_terminal_response().await?;
        // Replay the accumulated deltas through the shared normalization
        // pipeline so streamed partial output survives even when the terminal
        // body's `output` is empty (e.g. an incomplete turn). A turn that
        // carried no deltas (e.g. a `response.done`-only turn) falls back to
        // normalizing the terminal body itself.
        match completion_response_from_raw_choices(provider, raw_choices, &response).await? {
            Some(normalized) => Ok(normalized),
            None => response.normalize(provider),
        }
    }

    /// Sends a completion turn and returns the provider's own wire response.
    ///
    /// Shares the send/receive path with
    /// [`ResponsesWebSocketSession::completion`], which calls it and then
    /// applies the provider-local mapping — one websocket turn either way.
    pub async fn raw_completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.send(completion_request).await?;
        self.wait_for_completed_response().await
    }

    /// Closes the websocket connection.
    ///
    /// Call this when you are finished with the session so the websocket can
    /// terminate with a clean close handshake.
    pub async fn close(&mut self) -> Result<(), CompletionError> {
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

    fn prepare_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<super::CompletionRequest, CompletionError> {
        let mut request = self.backend.shape_request(completion_request)?;

        // WebSocket mode is always event-driven, so these HTTP/SSE-specific flags
        // are ignored by the provider and only add noise to the payload.
        request.stream = None;
        request.additional_parameters.background = None;

        if self.backend.chains_previous_response_id()
            && request.additional_parameters.previous_response_id.is_none()
        {
            request.additional_parameters.previous_response_id = self.previous_response_id.clone();
        }

        Ok(request)
    }

    async fn wait_for_completed_response(&mut self) -> Result<CompletionResponse, CompletionError> {
        Ok(self.wait_for_terminal_response().await?.0)
    }

    /// Drives the shared [`RawChoiceAccumulator`] over the websocket events —
    /// the same decode state machine the SSE path uses, fed by a different
    /// transport — so streamed deltas survive alongside the terminal body.
    ///
    /// **A failed turn discards the choices collected so far, deliberately
    /// (#2258 G3).** Every error exit below — the `?` on `next_event()`, the
    /// `response.done`-without-a-body branch, and the provider `error` event —
    /// returns `Err` and drops `accumulator`/`raw_choices` with whatever text,
    /// reasoning and tool calls had already arrived.
    ///
    /// That is not a divergence from the SSE side: the right comparison is the
    /// *buffered* SSE path, `run_wire_buffered`, which likewise fails the whole
    /// operation on the first `Err` rather than returning partial content plus
    /// an error. Only the *live* SSE surface can do better, and only because it
    /// is a `Stream`: it yields the partial items first and the `Err` as a
    /// later element. This session exposes a unary surface —
    /// [`completion()`](Self::wait_for_completed_response) /
    /// `raw_completion()` return one `Result<CompletionResponse, _>` — and a
    /// unary return type cannot express partial-content-plus-error without
    /// inventing a second channel. Keeping the failed turn's fragments would
    /// mean returning a `CompletionResponse` that never completed, which is the
    /// exact fabrication the terminal-record rules exist to prevent.
    ///
    /// If a caller needs the partial content of a failed websocket turn, the
    /// fix is a streaming websocket surface, not a partial unary response.
    async fn wait_for_terminal_response(
        &mut self,
    ) -> Result<(CompletionResponse, Vec<WebSocketRawChoice>), CompletionError> {
        let mut accumulator = RawChoiceAccumulator::new(ResponsesUsage::new());
        let mut raw_choices = Vec::new();
        loop {
            match self.next_event().await? {
                ResponsesWebSocketEvent::Response(chunk) => {
                    if matches!(
                        chunk.kind,
                        ResponseChunkKind::ResponseCompleted
                            | ResponseChunkKind::ResponseFailed
                            | ResponseChunkKind::ResponseIncomplete
                    ) {
                        return finish_terminal_response(accumulator, chunk.response, raw_choices);
                    }
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if let Some(response) = done.as_completion_response() {
                        return finish_terminal_response(accumulator, response, raw_choices);
                    }

                    let message = if let Some(response_id) = done.response_id() {
                        format!(
                            "OpenAI websocket turn ended with response.done before a terminal response body was available (response_id={response_id})"
                        )
                    } else {
                        "OpenAI websocket turn ended with response.done before a terminal response body was available"
                            .to_string()
                    };

                    return Err(CompletionError::ProviderError(message));
                }
                ResponsesWebSocketEvent::Error(error) => {
                    // Genuine provider error event: preserve the serialized payload
                    // (code + message + any extra fields) so provider_response_json()
                    // parses it, matching the response.failed path. No HTTP status on
                    // the websocket stream, so status: None.
                    return Err(provider_error_from_event(error));
                }
                // `F` ignored item chunks here; `U` accumulates them into the
                // raw-choice channel, which is new behavior rather than a
                // rename, so the fork's ignore is superseded. Only the
                // tool-call-timing axis is read by `decode_item_chunk`, which
                // `strict()` and the terminal `tolerate_incomplete()` below set
                // identically — the two option values do not disagree.
                ResponsesWebSocketEvent::Item(chunk) => {
                    raw_choices.extend(
                        accumulator.decode_item_chunk(chunk, ResponsesStreamOptions::strict()),
                    );
                }
                ResponsesWebSocketEvent::Unknown(event) => {
                    // Semantic skip, raw passthrough: the accumulator never
                    // sees the frame, but the streaming surface still yields
                    // it verbatim. D2's event also carries `F`'s `kind` tag;
                    // only the payload belongs on this channel.
                    raw_choices.push(crate::streaming::RawStreamingChoice::Unknown(event.payload));
                }
            }
        }
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        match event {
            ResponsesWebSocketEvent::Response(chunk) => match chunk.kind {
                // An incomplete turn still produced a response the next turn
                // can chain from, so it keeps `previous_response_id` like a
                // completed one.
                ResponseChunkKind::ResponseCompleted | ResponseChunkKind::ResponseIncomplete => {
                    let response_id = chunk.response.id.clone();
                    self.previous_response_id = Some(response_id.clone());
                    self.pending_done_response_id = Some(response_id);
                    if let Some(envelope) = self.pending_envelope.take() {
                        self.last_envelope = Some(envelope);
                    }
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseFailed => {
                    self.pending_done_response_id = Some(chunk.response.id.clone());
                    self.previous_response_id = None;
                    self.pending_envelope = None;
                    self.in_flight = false;
                }
                ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {}
            },
            ResponsesWebSocketEvent::Done(done) => {
                match done.status() {
                    Some(ResponseStatus::Completed) | Some(ResponseStatus::Incomplete) => {
                        if let Some(response_id) = done.response_id() {
                            self.previous_response_id = Some(response_id.to_string());
                        }
                        if let Some(envelope) = self.pending_envelope.take() {
                            self.last_envelope = Some(envelope);
                        }
                    }
                    Some(ResponseStatus::Failed)
                    | Some(ResponseStatus::Cancelled)
                    | Some(ResponseStatus::Other(_)) => {
                        self.previous_response_id = None;
                        self.pending_envelope = None;
                    }
                    Some(ResponseStatus::InProgress | ResponseStatus::Queued) | None => {}
                }
                self.pending_done_response_id = None;
                self.in_flight = false;
            }
            ResponsesWebSocketEvent::Error(_) => {
                self.previous_response_id = None;
                self.pending_done_response_id = None;
                self.pending_envelope = None;
                self.in_flight = false;
            }
            // Neither advances the turn's lifecycle state. An unmodelled event is
            // not evidence about `in_flight`, the response id, or the envelope.
            ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {}
        }
    }

    fn abort_turn(&mut self) {
        self.previous_response_id = None;
        self.pending_done_response_id = None;
        self.pending_envelope = None;
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

    fn ensure_open(&self) -> Result<(), CompletionError> {
        if self.closed || self.failed {
            return Err(CompletionError::ProviderError(
                "The OpenAI websocket session is closed".to_string(),
            ));
        }

        Ok(())
    }

    fn fail_session(&mut self, error: CompletionError) -> CompletionError {
        self.mark_failed();
        error
    }

    async fn read_next_message(
        &mut self,
    ) -> Result<Option<Result<Message, tungstenite::Error>>, CompletionError> {
        if let Some(timeout_duration) = self.event_timeout {
            match tokio::time::timeout(timeout_duration, self.socket.next()).await {
                Ok(message) => Ok(message),
                Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
            }
        } else {
            Ok(self.socket.next().await)
        }
    }
}

impl<B> Drop for ResponsesWebSocketSession<B> {
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

/// Records the terminal event into the accumulator and drains it, so the raw
/// choices end with the terminal record exactly as the SSE path produces them.
fn finish_terminal_response(
    mut accumulator: RawChoiceAccumulator,
    response: CompletionResponse,
    mut raw_choices: Vec<WebSocketRawChoice>,
) -> Result<(CompletionResponse, Vec<WebSocketRawChoice>), CompletionError> {
    let response = terminal_response_result(response)?;
    // Only completed/incomplete get through `terminal_response_result`, so the
    // accumulator's failed-event error mapping (which needs the raw event
    // bytes this path no longer has) is unreachable here.
    let kind = if matches!(response.status, ResponseStatus::Incomplete) {
        ResponseChunkKind::ResponseIncomplete
    } else {
        ResponseChunkKind::ResponseCompleted
    };
    // `terminal_response_result` above admits only `Completed` and
    // `Incomplete`, so this path has already ruled an incomplete turn a
    // successful terminal. Strict options would contradict that precondition
    // and error here — with an empty body, since this path retains no raw
    // event bytes to report.
    accumulator.record_response_chunk(
        kind,
        response.clone(),
        "",
        ResponsesStreamOptions::tolerate_incomplete(),
    )?;
    raw_choices.extend(accumulator.finish());
    Ok((response, raw_choices))
}

fn terminal_response_result(
    response: CompletionResponse,
) -> Result<CompletionResponse, CompletionError> {
    match response.status {
        ResponseStatus::Completed => Ok(response),
        // Deliberate two-tier behaviour: when the provider supplies its own error
        // object we preserve the full failed-response envelope through
        // `from_provider_body` (status: None, no HTTP status on the websocket
        // stream) so `provider_response_json()` parses it — consistent with the
        // `error` event and the streaming paths. The body is re-serialized from
        // the parsed response (not byte-identical to the wire bytes, which aren't
        // retained past parsing) — semantically the provider's payload. When the
        // object is absent we have nothing provider-authored to surface, so we
        // emit a Rig-authored `ProviderError` diagnostic (provider_response_body()
        // is None).
        ResponseStatus::Failed => match response.error.as_ref() {
            Some(error) => Err(CompletionError::from_provider_body(
                serde_json::to_string(&response).unwrap_or_else(|_| error.message.clone()),
            )),
            None => Err(CompletionError::ProviderError(response_error_message(
                "failed response",
            ))),
        },
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // normalization path maps the status/incomplete_details to a finish
        // reason via `map_finish_reason`, matching the unary and SSE paths.
        ResponseStatus::Incomplete => Ok(response),
        other => Err(CompletionError::ProviderError(format!(
            "OpenAI websocket response ended in state {other:?}"
        ))),
    }
}

fn response_error_message(fallback: &str) -> String {
    format!("OpenAI websocket returned a {fallback}")
}

/// Maps a provider `error` event into a [`CompletionError`] that preserves the
/// raw error payload as JSON (code + message + any extra provider fields) so the
/// `provider_response_*` helpers can inspect it. The websocket stream carries no
/// HTTP status, so `status` is `None`. The body is the event re-serialized from
/// the parsed representation (not byte-identical to the original wire bytes,
/// which are not retained past parsing) — semantically the provider's payload.
fn provider_error_from_event(error: ResponsesWebSocketErrorEvent) -> CompletionError {
    CompletionError::from_provider_body(
        serde_json::to_string(&error).unwrap_or_else(|_| error.to_string()),
    )
}

/// Parses one websocket JSON payload into a server event.
///
/// Only the websocket-only envelope types (`error`, `response.done`) are
/// dispatched here; every other frame classifies through the same
/// [`classify_responses_frame`] interpreter the SSE paths use, so the modeled
/// Responses event set — and its strict decode policy — is stated once for the
/// wire family rather than duplicated per transport.
///
/// `F`'s local `is_known_streaming_event` is dropped rather than merged: its 20
/// event types are a strict subset of the classifier's 21 (which adds
/// `response.reasoning_text.done`), including `F`'s own
/// `response.reasoning_text.delta` addition. Keeping a second copy of the set
/// is what let the two drift apart in the first place.
fn parse_server_event(payload: &str) -> Result<Option<ResponsesWebSocketEvent>, CompletionError> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let event_type = serde_json::from_str::<EventType>(payload)?;
    match event_type.kind.as_str() {
        "error" => serde_json::from_str(payload)
            .map(|e| Some(ResponsesWebSocketEvent::Error(e)))
            .map_err(CompletionError::from),
        "response.done" => serde_json::from_str(payload)
            .map(|d| Some(ResponsesWebSocketEvent::Done(d)))
            .map_err(CompletionError::from),
        // Shared per-frame triage (`Unknown` is warned and forwarded raw for
        // the passthrough channel, `Corrupt` fails the turn — this surface
        // has no stream to carry `Err` items). `F`'s separate known-event arm
        // and its `json_utils::from_str` routing are both subsumed: the
        // classifier owns the known/unknown split, and the fork's
        // arbitrary-precision decode rides inside it as `ApRoutedChunk`.
        // `F`'s `debug!` here is dropped as a duplicate — `triage_frame` warns
        // on the same frame through `warn_unmodeled`, at a higher level and
        // with the payload redacted.
        _ => Ok(Some(
            match triage_frame(classify_responses_frame(payload))? {
                TriagedFrame::Event(StreamingCompletionChunk::Response(response)) => {
                    ResponsesWebSocketEvent::Response(response)
                }
                TriagedFrame::Event(StreamingCompletionChunk::Delta(item)) => {
                    ResponsesWebSocketEvent::Item(item)
                }
                // `triage_frame` spends the classifier's own `event_type` on
                // that warning and returns only the payload, so `F`'s retained
                // `kind` tag comes from the `EventType` probe above. The two
                // cannot disagree: reaching this arm proves the probe already
                // decoded exactly one top-level string `type`, and the
                // classifier re-scans the same bytes for the same key.
                TriagedFrame::Unknown(payload) => {
                    ResponsesWebSocketEvent::Unknown(UnrecognizedEvent {
                        kind: event_type.kind,
                        payload,
                    })
                }
            },
        )),
    }
}

fn websocket_message_to_text(message: Message) -> Result<Option<String>, CompletionError> {
    match message {
        Message::Text(text) => Ok(Some(text.to_string())),
        Message::Binary(bytes) => String::from_utf8(bytes.to_vec())
            .map(Some)
            .map_err(|error| CompletionError::ResponseError(error.to_string())),
        Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => Ok(None),
        Message::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason.to_string())
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(CompletionError::ProviderError(format!(
                "The OpenAI websocket connection closed {reason}"
            )))
        }
    }
}

fn websocket_url(base_url: &str) -> Result<String, CompletionError> {
    let mut url = Url::parse(base_url)?;
    match url.scheme() {
        "https" => {
            url.set_scheme("wss").map_err(|_| {
                CompletionError::ProviderError("Failed to convert https URL to wss".to_string())
            })?;
        }
        "http" => {
            url.set_scheme("ws").map_err(|_| {
                CompletionError::ProviderError("Failed to convert http URL to ws".to_string())
            })?;
        }
        scheme => {
            return Err(CompletionError::ProviderError(format!(
                "Unsupported base URL scheme for OpenAI websocket mode: {scheme}"
            )));
        }
    }

    let path = format!("{}/responses", url.path().trim_end_matches('/'));
    url.set_path(&path);
    Ok(url.to_string())
}

fn websocket_request(
    url: &str,
    headers: &http::HeaderMap,
) -> Result<http::Request<()>, CompletionError> {
    let mut request = url.into_client_request().map_err(|error| {
        CompletionError::ProviderError(format!("Failed to build OpenAI websocket request: {error}"))
    })?;

    for (name, value) in headers {
        request.headers_mut().insert(name, value.clone());
    }

    Ok(request)
}

async fn connect_websocket(
    request: http::Request<()>,
    connect_timeout: Option<Duration>,
    tls_connector: Option<WebSocketTlsConnector>,
) -> Result<OpenAIWebSocket, CompletionError> {
    let connect = async move {
        match tls_connector {
            Some(connector) => {
                connect_async_tls_with_config(
                    request,
                    None,
                    false,
                    Some(connector.into_connector()),
                )
                .await
            }
            None => {
                ensure_default_crypto_provider();
                connect_async(request).await
            }
        }
    };

    if let Some(timeout_duration) = connect_timeout {
        match tokio::time::timeout(timeout_duration, connect).await {
            Ok(result) => result
                .map(|(socket, _)| socket)
                .map_err(websocket_provider_error),
            Err(_) => Err(connect_timeout_error(timeout_duration)),
        }
    } else {
        connect
            .await
            .map(|(socket, _)| socket)
            .map_err(websocket_provider_error)
    }
}

/// Best-effort installation of a process-wide default rustls crypto provider.
///
/// On the no-connector path, `tokio-tungstenite` builds an implicit default
/// `rustls::ClientConfig`. When more than one crypto provider is linked into the
/// final binary, constructing that config panics unless a process-wide default
/// provider has been installed. This installs one idempotently and treats an
/// already-installed provider as success — the common case for hosts (such as
/// Muninn) that install their own provider at startup before Rig runs.
fn ensure_default_crypto_provider() {
    use rustls::crypto::{CryptoProvider, aws_lc_rs};

    if CryptoProvider::get_default().is_none() {
        // `install_default` returns `Err` only if another thread installed a
        // provider between the check above and this call. That still satisfies
        // our requirement (a default is now present), so both outcomes are success.
        let _ = aws_lc_rs::default_provider().install_default();
    }
}

fn connect_timeout_error(timeout: Duration) -> CompletionError {
    CompletionError::ProviderError(format!(
        "Timed out connecting to the OpenAI websocket after {timeout:?}"
    ))
}

fn event_timeout_error(timeout: Duration) -> CompletionError {
    CompletionError::ProviderError(format!(
        "Timed out waiting for the next OpenAI websocket event after {timeout:?}"
    ))
}

fn websocket_provider_error(error: tungstenite::Error) -> CompletionError {
    CompletionError::ProviderError(error.to_string())
}

fn keepalive_flush_timeout_error(timeout: Duration) -> CompletionError {
    CompletionError::ProviderError(format!(
        "Timed out flushing the OpenAI websocket idle keepalive pong after {timeout:?}"
    ))
}

fn keepalive_flood_error(budget: usize) -> CompletionError {
    CompletionError::ProviderError(format!(
        "The OpenAI websocket delivered at least {budget} buffered frames during idle keepalive"
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        InputDelta, KeepaliveDrain, MAX_KEEPALIVE_DRAIN_FRAMES, ResponsesWebSocketCreateOptions,
        ResponsesWebSocketDoneEvent, ResponsesWebSocketEvent, UnrecognizedEvent,
        parse_server_event, terminal_response_result, websocket_url,
    };
    use crate::client::CompletionClient;
    use crate::completion::CompletionError;
    use crate::completion::CompletionModel;
    use crate::providers::openai::responses_api::streaming::ItemChunkKind;
    use crate::providers::openai::responses_api::{
        CompletionResponse, IncompleteDetailsReason, Output, ResponseError, ResponseObject,
        ResponseStatus, ResponsesUsage,
    };
    use futures::{SinkExt, StreamExt};
    use serde_json::json;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::time::sleep;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    /// The one invariant `InputDelta` exists to carry, asserted directly on the
    /// type rather than only through a caller that happens to pass a non-empty
    /// `Vec`.
    ///
    /// `U` removed `OneOrMany` and moved the analogous request-level guarantee
    /// to `validate_message_content`, which never runs on this path — an
    /// incremental turn mutates a cached provider envelope instead of building
    /// a `CompletionRequest`. So this type is the only thing standing between a
    /// caller and a chained `response.create` carrying no new input, and every
    /// way in is checked here.
    #[test]
    fn an_input_delta_admits_items_and_rejects_emptiness_by_every_route() {
        let item = Vec::<crate::providers::openai::responses_api::InputItem>::try_from(
            crate::completion::Message::user("hello"),
        )
        .expect("a user message converts into input items")
        .pop()
        .expect("that conversion yields at least one item");

        // `one` is infallible by construction — the type has no way to spell an
        // empty single-item delta.
        assert_eq!(InputDelta::one(item.clone()).into_vec().len(), 1);

        // The fallible routes agree with each other in both directions.
        assert_eq!(
            InputDelta::new(vec![item.clone(), item.clone()])
                .expect("a two-item delta is non-empty")
                .into_vec()
                .len(),
            2
        );
        assert!(
            InputDelta::try_from(vec![item]).is_ok(),
            "`TryFrom` must accept what `new` accepts"
        );

        InputDelta::new(Vec::new()).expect_err("`new` must reject an empty delta");
        InputDelta::try_from(Vec::new()).expect_err("`TryFrom` must reject an empty delta");
    }

    /// The frames of a drain that must have serviced the socket cleanly.
    fn serviced(drain: KeepaliveDrain, context: &str) -> Vec<UnrecognizedEvent> {
        let (recovered, ending) = drain.into_parts();
        if let Some(error) = ending {
            panic!("{context}, but the drain failed: {error}");
        }
        recovered
    }

    /// The frames a failing drain still recovered, and the failure itself.
    ///
    /// Both halves, always: the whole point of the drain's shape is that a
    /// caller cannot take the failure without also taking what preceded it, so
    /// a test helper that returned only the error would hide the property under
    /// test.
    fn failed(drain: KeepaliveDrain, context: &str) -> (Vec<UnrecognizedEvent>, CompletionError) {
        let (recovered, ending) = drain.into_parts();
        let Some(error) = ending else {
            panic!("{context}, but the drain reported success");
        };
        (recovered, error)
    }

    #[test]
    fn websocket_error_event_preserves_provider_payload_as_json() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "type".to_string(),
            serde_json::Value::String("invalid_request_error".to_string()),
        );
        let event = super::ResponsesWebSocketErrorEvent {
            kind: super::ResponsesWebSocketErrorEventKind::Error,
            error: super::ResponsesWebSocketErrorPayload {
                code: Some("rate_limit_exceeded".to_string()),
                message: Some("slow down".to_string()),
                extra,
            },
        };

        let err = super::provider_error_from_event(event);

        // No HTTP status on the websocket stream, and the raw payload round-trips
        // through provider_response_json() (code + message + extra all preserved).
        assert_eq!(err.provider_response_status(), None);
        let json = err
            .provider_response_json()
            .expect("preserved body should be valid JSON")
            .expect("provider response body should be present");
        assert_eq!(json["error"]["code"], "rate_limit_exceeded");
        assert_eq!(json["error"]["message"], "slow down");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            provider_request_id: None,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5.4".to_string(),
            usage: Some(ResponsesUsage {
                input_tokens: 1,
                input_tokens_details: None,
                output_tokens: 2,
                output_tokens_details: Some(
                    crate::providers::openai::responses_api::OutputTokensDetails {
                        reasoning_tokens: 0,
                    },
                ),
                total_tokens: 3,
            }),
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: Default::default(),
            provider_reasoning: None,
            reasoning_metadata: None,
            reasoning_context: None,
        }
    }

    #[test]
    fn warmup_options_serialize_generate_false() {
        let options = ResponsesWebSocketCreateOptions::warmup();
        let json = serde_json::to_value(options).expect("options should serialize");

        assert_eq!(json, json!({ "generate": false }));
    }

    #[test]
    fn websocket_url_converts_https_to_wss() {
        let url = websocket_url("https://api.openai.com/v1").expect("url should convert");
        assert_eq!(url, "wss://api.openai.com/v1/responses");
    }

    #[test]
    fn parse_done_event_exposes_response_id() {
        let payload = json!({
            "type": "response.done",
            "response": {
                "id": "resp_done_1",
                "status": "completed"
            }
        });

        let event = parse_server_event(&payload.to_string())
            .expect("done event should deserialize")
            .expect("done event should not be skipped");

        assert!(matches!(
            event,
            ResponsesWebSocketEvent::Done(ResponsesWebSocketDoneEvent { .. })
        ));
        assert_eq!(event.response_id(), Some("resp_done_1"));
        assert!(event.is_terminal());
    }

    #[test]
    fn parse_response_completed_event_is_terminal() {
        let payload = json!({
            "type": "response.completed",
            "sequence_number": 12,
            "response": {
                "id": "resp_completed_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.4",
                "usage": null,
                "output": [],
                "tools": []
            }
        });

        let event = parse_server_event(&payload.to_string())
            .expect("response event should deserialize")
            .expect("response event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Response(_)));
        assert!(event.is_terminal());
        assert_eq!(event.response_id(), Some("resp_completed_1"));
    }

    #[test]
    fn parse_live_output_item_added_event() {
        let payload = json!({
            "type": "response.output_item.added",
            "item": {
                "id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant"
            },
            "output_index": 0,
            "sequence_number": 2
        });

        let event = parse_server_event(&payload.to_string())
            .expect("output item event should parse")
            .expect("output item event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn parse_live_content_part_added_event() {
        let payload = json!({
            "type": "response.content_part.added",
            "content_index": 0,
            "item_id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
            "output_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": ""
            },
            "sequence_number": 3
        });

        let event = parse_server_event(&payload.to_string())
            .expect("content part event should parse")
            .expect("content part event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn parse_live_output_text_delta_event() {
        let payload = json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "Web",
            "item_id": "msg_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
            "logprobs": [],
            "obfuscation": "2YYErYq7jkqqM",
            "output_index": 0,
            "sequence_number": 4
        });

        let event = parse_server_event(&payload.to_string())
            .expect("output text delta event should parse")
            .expect("output text delta event should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    }

    #[test]
    fn parse_reasoning_text_delta_event() {
        let payload = json!({
            "type": "response.reasoning_text.delta",
            "content_index": 0,
            "delta": "thinking",
            "item_id": "rs_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
            "output_index": 0,
            "sequence_number": 4
        });

        let event = parse_server_event(&payload.to_string())
            .expect("reasoning text delta event should parse")
            .expect("reasoning text delta event should not be skipped");

        let ResponsesWebSocketEvent::Item(chunk) = event else {
            panic!("expected an item event");
        };
        assert!(matches!(
            chunk.data,
            ItemChunkKind::ReasoningTextDelta(ref delta) if delta.delta == "thinking"
        ));
    }

    #[test]
    fn parse_reasoning_summary_delta_event_into_delta_chunk() {
        let payload = json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_REDACTED_1",
            "output_index": 0,
            "sequence_number": 51,
            "summary_index": 0,
            "delta": " far"
        });

        let event = parse_server_event(&payload.to_string())
            .expect("reasoning summary delta event should parse")
            .expect("reasoning summary delta event should not be skipped");

        let ResponsesWebSocketEvent::Item(chunk) = event else {
            panic!("expected an item event");
        };
        assert!(matches!(
            chunk.data,
            ItemChunkKind::ReasoningSummaryTextDelta(ref delta)
                if delta.summary_index == 0 && delta.sequence_number == 51 && delta.delta == " far"
        ));
    }

    #[test]
    fn parse_reasoning_summary_done_event_into_text_chunk() {
        let payload = json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_REDACTED_1",
            "output_index": 0,
            "sequence_number": 54,
            "summary_index": 0,
            "text": "The problem: A train leaves Station A at 60 km/h."
        });

        let event = parse_server_event(&payload.to_string())
            .expect("reasoning summary done event should parse")
            .expect("reasoning summary done event should not be skipped");

        let ResponsesWebSocketEvent::Item(chunk) = event else {
            panic!("expected an item event");
        };
        assert!(matches!(
            chunk.data,
            ItemChunkKind::ReasoningSummaryTextDone(ref done)
                if done.summary_index == 0
                    && done.sequence_number == 54
                    && done.text == "The problem: A train leaves Station A at 60 km/h."
        ));
    }

    #[test]
    fn reasoning_summary_done_rejects_delta_shaped_payload() {
        // A `response.reasoning_summary_text.done` payload carrying `delta`
        // instead of the real `text` field must fail to deserialize rather than
        // silently defaulting `text` to empty — neither new struct declares a
        // serde default, so a missing required field is a hard parse error.
        let payload = json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_REDACTED_1",
            "output_index": 0,
            "sequence_number": 54,
            "summary_index": 0,
            "delta": "wrong field shape"
        });

        let error = parse_server_event(&payload.to_string())
            .expect_err("delta-shaped payload should not parse as a done chunk");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );
    }

    #[test]
    fn reasoning_summary_delta_rejects_done_shaped_payload() {
        // The inverse of `reasoning_summary_done_rejects_delta_shaped_payload`:
        // a `response.reasoning_summary_text.delta` payload carrying `text`
        // instead of the real `delta` field must also fail to deserialize.
        // `SummaryTextDeltaChunk::delta` has no serde default either, so both
        // halves of the public wire-shape split are strict in the same way.
        let payload = json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_REDACTED_1",
            "output_index": 0,
            "sequence_number": 51,
            "summary_index": 0,
            "text": "wrong field shape"
        });

        let error = parse_server_event(&payload.to_string())
            .expect_err("done-shaped payload should not parse as a delta chunk");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );
    }

    #[test]
    fn terminal_response_requires_completed_status() {
        let completed = terminal_response_result(sample_response(ResponseStatus::Completed))
            .expect("completed response should succeed");
        assert_eq!(completed.id, "resp_123");

        let failed = terminal_response_result(sample_response(ResponseStatus::Failed))
            .expect_err("failed response should error");
        assert!(failed.to_string().contains("failed response"));
    }

    #[tokio::test]
    async fn incomplete_turn_keeps_streamed_partial_output() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            // The content exists ONLY in the delta events; the terminal
            // `response.incomplete` body has an empty `output`, which is a
            // sequence the wire protocol permits.
            socket
                .send(Message::text(
                    json!({
                        "type": "response.output_text.delta",
                        "content_index": 0,
                        "delta": "partial",
                        "item_id": "msg_incomplete_1",
                        "logprobs": [],
                        "output_index": 0,
                        "sequence_number": 1
                    })
                    .to_string(),
                ))
                .await
                .expect("delta event should send");

            let mut response = sample_response(ResponseStatus::Incomplete);
            response.incomplete_details = Some(IncompleteDetailsReason {
                reason: "max_output_tokens".to_string(),
            });
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.incomplete",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("incomplete event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("incomplete turn should be a successful terminal");

        // The streamed partial text survives, and normalization maps the
        // incomplete status to the same finish reason as the unary path.
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(normalized.usage.input_tokens, 1);
        assert_eq!(normalized.usage.output_tokens, 2);
        assert_eq!(normalized.usage.total_tokens, 3);
        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "partial"
        ));

        server.await.expect("server task should finish");
    }

    /// #2258 P2: the websocket session shares `decode_item_chunk`, so text for
    /// one message item interleaved with reasoning must aggregate as one text
    /// part here too.
    #[tokio::test]
    async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            let events = [
                json!({
                    "type": "response.output_text.delta",
                    "content_index": 0,
                    "delta": "hello ",
                    "item_id": "msg_1",
                    "logprobs": [],
                    "output_index": 0,
                    "sequence_number": 1
                }),
                json!({
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "because",
                    "item_id": "rs_2",
                    "output_index": 1,
                    "summary_index": 0,
                    "sequence_number": 2
                }),
                json!({
                    "type": "response.output_text.delta",
                    "content_index": 0,
                    "delta": "world",
                    "item_id": "msg_1",
                    "logprobs": [],
                    "output_index": 0,
                    "sequence_number": 3
                }),
                json!({
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
                        .expect("response should serialize"),
                }),
            ];
            for event in events {
                socket
                    .send(Message::text(event.to_string()))
                    .await
                    .expect("event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("interleaved turn should normalize");

        let texts: Vec<_> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            ["hello world"],
            "same-item text must aggregate as one part around the reasoning"
        );
        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(_)
            )),
            "the interleaved reasoning must survive"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn completed_turn_without_deltas_falls_back_to_terminal_body() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            // No delta events at all: the terminal body carries the full
            // output, so normalization must fall back to it.
            let mut response = sample_response(ResponseStatus::Completed);
            response.output = vec![
                serde_json::from_value::<Output>(json!({
                    "type": "message",
                    "id": "msg_terminal_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "annotations": [], "text": "hello there" }]
                }))
                .expect("output message should deserialize"),
            ];
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("completed turn should normalize");

        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "hello there"
        ));
        assert_eq!(normalized.message_id.as_deref(), Some("msg_terminal_1"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn incomplete_turn_without_deltas_normalizes_terminal_body_output() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            // No delta events at all AND an incomplete terminal whose body
            // carries the partial output: the body must be normalized rather
            // than the turn reading as empty.
            let mut response = sample_response(ResponseStatus::Incomplete);
            response.incomplete_details = Some(IncompleteDetailsReason {
                reason: "max_output_tokens".to_string(),
            });
            response.output = vec![
                serde_json::from_value::<Output>(json!({
                    "type": "message",
                    "id": "msg_body_only_1",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "annotations": [], "text": "partial from body" }]
                }))
                .expect("output message should deserialize"),
            ];
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.incomplete",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("incomplete event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("incomplete turn with body output should normalize");

        assert!(matches!(
            normalized.choice.first(),
            Some(crate::completion::AssistantContent::Text(text)) if text.text == "partial from body"
        ));
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(normalized.message_id.as_deref(), Some("msg_body_only_1"));

        server.await.expect("server task should finish");
    }

    #[test]
    fn terminal_failed_response_with_error_preserves_raw_payload() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "the model failed to generate a response".to_string(),
        });

        let err = match terminal_response_result(response) {
            Ok(_) => panic!("failed response with an error object should fail"),
            Err(e) => e,
        };

        // The full failed-response envelope is preserved as a ProviderResponse with
        // no HTTP status (the websocket stream carries none), so the raw JSON parses
        // back with the provider error nested under `error` — proving the whole
        // envelope is kept, not just the error object.
        assert_eq!(err.provider_response_status(), None);

        let json = err
            .provider_response_json()
            .expect("preserved body should parse as JSON")
            .expect("preserved body should not be empty");
        assert_eq!(
            json["error"]["message"],
            "the model failed to generate a response"
        );
        assert_eq!(json["error"]["code"], "server_error");
    }

    #[test]
    fn terminal_failed_response_without_error_is_rig_diagnostic() {
        let err = match terminal_response_result(sample_response(ResponseStatus::Failed)) {
            Ok(_) => panic!("failed response should fail"),
            Err(e) => e,
        };

        // No provider error object, so this is a Rig-authored diagnostic and exposes
        // no preserved provider response body.
        assert_eq!(err.provider_response_body(), None);
        assert!(err.to_string().contains("failed response"));
    }

    #[tokio::test]
    async fn malformed_known_event_rejects_reuse_and_allows_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed"
                    })
                    .to_string(),
                ))
                .await
                .expect("malformed known event should send");

            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        let error = session
            .next_event()
            .await
            .expect_err("malformed known event should fail");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );

        let closed = session
            .send(model.completion_request("retry").build())
            .await
            .expect_err("session should close after fatal parse error");
        assert!(
            closed.to_string().contains("session is closed"),
            "expected closed-session error, got {closed}"
        );

        session
            .close()
            .await
            .expect("explicit close after fatal parse error should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn event_timeout_rejects_reuse_and_allows_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            sleep(Duration::from_millis(60)).await;
            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket_builder("gpt-4o")
            .event_timeout(Duration::from_millis(20))
            .connect()
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        let error = session
            .next_event()
            .await
            .expect_err("next_event should time out");
        assert!(
            error
                .to_string()
                .contains("Timed out waiting for the next OpenAI websocket event"),
            "expected timeout error, got {error}"
        );

        let closed = session
            .send(model.completion_request("retry").build())
            .await
            .expect_err("timed-out session should close");
        assert!(
            closed.to_string().contains("session is closed"),
            "expected closed-session error, got {closed}"
        );

        session
            .close()
            .await
            .expect("explicit close after timeout should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn late_response_done_is_ignored_on_next_turn() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for (index, response_id) in ["resp_1", "resp_2"].iter().enumerate() {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                let response = sample_response(ResponseStatus::Completed);
                let response = serde_json::to_value(CompletionResponse {
                    id: (*response_id).to_string(),
                    ..response
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.completed",
                            "sequence_number": (index * 2) + 1,
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("completed event should send");
                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": {
                                "id": response_id,
                                "status": "completed",
                            },
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn clearing_previous_response_id_does_not_disable_late_done_filter() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                let response = sample_response(ResponseStatus::Completed);
                let response = serde_json::to_value(CompletionResponse {
                    id: response_id.to_string(),
                    ..response
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.completed",
                            "sequence_number": 1,
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("completed event should send");
                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": {
                                "id": response_id,
                                "status": "completed",
                            },
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");

        session.clear_previous_response_id();
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn failed_turn_keeps_late_done_out_of_next_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let first_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = first_request
                .into_text()
                .expect("failed request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            let failed_response = serde_json::to_value(CompletionResponse {
                id: "resp_failed".to_string(),
                status: ResponseStatus::Failed,
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("failed response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.failed",
                        "sequence_number": 1,
                        "response": failed_response,
                    })
                    .to_string(),
                ))
                .await
                .expect("failed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": {
                            "id": "resp_failed",
                            "status": "failed",
                        },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            let second_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            let response = sample_response(ResponseStatus::Completed);
            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..response
            })
            .expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": {
                            "id": "resp_2",
                            "status": "completed",
                        },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let error = session
            .wait_for_completed_response()
            .await
            .expect_err("failed response should error");
        assert!(error.to_string().contains("failed response"));
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn done_first_completed_turn_updates_previous_response_id() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create payload, got {payload}"
                );

                if response_id == "resp_2" {
                    assert!(
                        payload.contains("\"previous_response_id\":\"resp_1\""),
                        "expected chained previous_response_id in payload, got {payload}"
                    );
                }

                let response = serde_json::to_value(CompletionResponse {
                    id: response_id.to_string(),
                    ..sample_response(ResponseStatus::Completed)
                })
                .expect("response should serialize");

                socket
                    .send(Message::text(
                        json!({
                            "type": "response.done",
                            "response": response,
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("done event should send");
            }
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn done_first_failed_turn_does_not_chain_next_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let first_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = first_request
                .into_text()
                .expect("first request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );
            assert!(
                !payload.contains("\"previous_response_id\""),
                "did not expect previous_response_id in first payload, got {payload}"
            );

            let failed_response = serde_json::to_value(CompletionResponse {
                id: "resp_failed".to_string(),
                status: ResponseStatus::Failed,
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("failed response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": failed_response,
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            let second_request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );
            assert!(
                !payload.contains("\"previous_response_id\""),
                "did not expect chained previous_response_id in payload, got {payload}"
            );

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let error = session
            .wait_for_completed_response()
            .await
            .expect_err("failed response should error");
        assert!(error.to_string().contains("failed response"));
        assert_eq!(session.previous_response_id(), None);

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[test]
    fn websocket_url_converts_http_to_ws() {
        let url = websocket_url("http://localhost:8080/v1").expect("url should convert");
        assert_eq!(url, "ws://localhost:8080/v1/responses");
    }

    #[test]
    fn websocket_url_rejects_unsupported_scheme() {
        let result = websocket_url("ftp://example.com/v1");
        assert!(result.is_err());
    }

    #[test]
    fn websocket_url_trims_trailing_slash() {
        let url = websocket_url("https://api.openai.com/v1/").expect("url should convert");
        assert_eq!(url, "wss://api.openai.com/v1/responses");
    }

    #[test]
    fn unknown_event_type_is_forwarded_raw_with_its_complete_parsed_payload() {
        let payload = json!({
            "type": "response.some_future_event",
            "data": "hello",
            "nested": { "sequence_number": 7 }
        });

        let event = parse_server_event(&payload.to_string())
            .expect("unknown event should not error")
            .expect("unknown event should be surfaced rather than skipped");

        // D2 composes both parents' claims on one event: `F`'s retained `kind`
        // tag and complete-payload preservation, and `U`'s raw passthrough
        // variant. They were never exclusive — only spelled on different
        // shapes.
        let ResponsesWebSocketEvent::Unknown(UnrecognizedEvent {
            kind,
            payload: parsed,
        }) = event
        else {
            panic!("an unmodelled event type must parse as the Unknown passthrough");
        };
        assert_eq!(kind, "response.some_future_event");
        // Compared as a JSON value, not as text: parsing preserves the event's
        // value, and its wire spelling is deliberately not a claim this makes.
        // Content is reached through `value()` because the payload newtype
        // redacts its `Debug`; that opt-in is the point of the newtype.
        assert_eq!(
            parsed.value(),
            &payload,
            "every field must survive, including ones no modelled variant keeps"
        );
        // The same payload is what the streaming surface yields on the
        // `RawStreamingChoice::Unknown` passthrough.
        assert_eq!(parsed, payload.clone().into());
    }

    #[test]
    fn unknown_event_is_nonterminal_and_carries_no_response_id() {
        let event =
            parse_server_event(&json!({ "type": "response.some_future_event" }).to_string())
                .expect("unknown event should not error")
                .expect("unknown event should be surfaced");

        assert!(
            !event.is_terminal(),
            "an unmodelled event ends no turn at the protocol level"
        );
        assert_eq!(event.response_id(), None);
    }

    #[test]
    fn malformed_known_event_returns_error() {
        let payload = json!({
            "type": "response.completed"
        });

        let error = parse_server_event(&payload.to_string())
            .expect_err("malformed known event should error");
        assert!(
            error.to_string().contains("StreamingCompletionChunk"),
            "expected strict decode failure, got {error}"
        );
    }

    #[tokio::test]
    async fn close_is_idempotent() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session.close().await.expect("first close should succeed");
        session.close().await.expect("second close should succeed");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn send_while_in_flight_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // Read the first request but don't respond — keep it in-flight
            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            // Wait for client to finish its test
            sleep(Duration::from_millis(100)).await;
            let _ = socket.close(None).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");

        let error = session
            .send(model.completion_request("second").build())
            .await
            .expect_err("second send while in-flight should error");
        assert!(
            error.to_string().contains("already in flight"),
            "expected in-flight error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn send_after_close_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let _socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");
            sleep(Duration::from_millis(100)).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session.close().await.expect("close should succeed");

        let error = session
            .send(model.completion_request("after close").build())
            .await
            .expect_err("send after close should error");
        assert!(
            error.to_string().contains("session is closed"),
            "expected closed-session error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn next_event_without_send_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let _socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");
            sleep(Duration::from_millis(100)).await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let error = session
            .next_event()
            .await
            .expect_err("next_event without send should error");
        assert!(
            error
                .to_string()
                .contains("No OpenAI websocket response is currently in flight"),
            "expected not-in-flight error, got {error}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn unknown_event_reaches_next_event_before_the_terminal_event() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.some_future_event",
                        "sequence_number": 1,
                        "output_index": 0,
                        "text": "content a modelled variant would drop",
                    })
                    .to_string(),
                ))
                .await
                .expect("unknown event should send");

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_after_unknown".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("send should succeed");

        let event = session.next_event().await.expect("event should arrive");
        let ResponsesWebSocketEvent::Unknown(UnrecognizedEvent { kind, payload }) = event else {
            panic!("the unmodelled event must reach next_event rather than be dropped");
        };
        assert_eq!(kind, "response.some_future_event");
        assert_eq!(
            payload
                .value()
                .get("text")
                .and_then(serde_json::Value::as_str),
            Some("content a modelled variant would drop"),
            "the payload must carry fields no modelled variant would have kept"
        );

        let terminal = session.next_event().await.expect("terminal should arrive");
        assert!(
            terminal.is_terminal(),
            "surfacing the unmodelled event must not disturb the turn's terminal event"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_returns_an_unknown_event_between_turns() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_1".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
            // Ordered before the trailing `done` so the two are indistinguishable
            // in arrival: whenever `done` has reached the buffer, this has too.
            socket
                .send(Message::text(
                    json!({
                        "type": "response.some_future_event",
                        "data": "arriving while the session is idle",
                    })
                    .to_string(),
                ))
                .await
                .expect("unknown event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": { "id": "resp_1", "status": "completed" },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            // Hold the socket open and serve a whole second turn, so the proof is
            // the session's continued health rather than any close/timing order.
            let second_request = socket
                .next()
                .await
                .expect("second request should exist")
                .expect("second request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"previous_response_id\":\"resp_1\""),
                "expected the tip to survive the returned idle event, got {payload}"
            );

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("second completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("send should succeed");
        let _response = session
            .wait_for_completed_response()
            .await
            .expect("response should complete");

        // Let the idle frames reach the buffer before servicing them, exactly as
        // `keepalive_consumes_trailing_done_and_preserves_tip` does.
        sleep(Duration::from_millis(100)).await;

        // Idle tolerance is unchanged: an unmodelled event does not fail the
        // session, which is what `keepalive_fails_loud_on_unexpected_data_frame`
        // proves happens to a genuine unexpected data frame. What changed is that
        // the frame is now handed back instead of dropped — a trailing unmodelled
        // `response.*` is exactly the case a caller must be able to account for.
        let drained = serviced(
            session.keepalive().await,
            "an unknown idle event must not fail the session",
        );
        assert_eq!(
            drained.len(),
            1,
            "the unmodelled idle event must be returned, not discarded"
        );
        assert_eq!(drained[0].kind, "response.some_future_event");
        assert_eq!(
            drained[0].payload.value()["type"],
            "response.some_future_event",
            "the complete parsed value is returned, not just the kind"
        );
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        // The session is still usable, so the skip neither failed it nor left the
        // stream misaligned.
        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete after the skipped idle event");
        assert_eq!(second.id, "resp_2");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_returns_every_unknown_event_in_arrival_order() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_1".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");

            // Three unmodelled events from three different namespaces, with the
            // trailing `done` in the middle: the drain must neither reorder them
            // nor let its own `done` handling drop the one that follows it.
            for kind in ["codex.rate_limits", "responsesapi.websocket_timing"] {
                socket
                    .send(Message::text(
                        json!({ "type": kind, "seen": kind }).to_string(),
                    ))
                    .await
                    .expect("unknown event should send");
            }
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": { "id": "resp_1", "status": "completed" },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
            socket
                .send(Message::text(
                    json!({ "type": "response.some_future_event", "seen": "last" }).to_string(),
                ))
                .await
                .expect("trailing unknown event should send");

            // Hold the socket open so the assertions are about the drain rather
            // than about a close race.
            let _ = socket.next().await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("send should succeed");
        let _response = session
            .wait_for_completed_response()
            .await
            .expect("response should complete");

        sleep(Duration::from_millis(100)).await;

        let drained = serviced(
            session.keepalive().await,
            "unknown idle events must not fail the session",
        );

        let kinds: Vec<&str> = drained.iter().map(|event| event.kind.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "codex.rate_limits",
                "responsesapi.websocket_timing",
                "response.some_future_event",
            ],
            "every unmodelled frame must be returned exactly once, in arrival \
             order, including the one that arrived after the trailing `done`"
        );
        assert_eq!(
            session.previous_response_id(),
            Some("resp_1"),
            "draining unknowns must not disturb the tip"
        );

        drop(session);
        let _ = server.await;
    }

    /// A drain that ends in failure still hands back everything it consumed.
    ///
    /// This is the case the previous `Result` shape could not express. The
    /// unmodelled frame is taken off the socket and can never be re-read, so
    /// returning only the close error made a session that had delivered an
    /// unaccountable frame indistinguishable from one that had merely closed —
    /// and a caller deciding whether its turn is provable needs exactly that
    /// distinction.
    #[tokio::test]
    async fn keepalive_recovers_unknown_frames_consumed_before_a_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.some_future_event",
                        "sequence_number": 1,
                    })
                    .to_string(),
                ))
                .await
                .expect("unknown event should send");

            // Drop without a close handshake, so the client sees the frame and
            // then the end of the stream in the same drain.
            drop(socket);
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        // Let both the frame and the end of the stream reach the buffer.
        sleep(Duration::from_millis(100)).await;

        let (recovered, error) = failed(
            session.keepalive().await,
            "a socket that ends mid-drain must report the failure",
        );
        assert_eq!(
            recovered.len(),
            1,
            "the frame consumed before the close is still the caller's: {error}"
        );
        assert_eq!(recovered[0].kind, "response.some_future_event");

        server.await.expect("server task should finish");
    }

    /// The drain consumes at most [`MAX_KEEPALIVE_DRAIN_FRAMES`], and stopping
    /// there costs the caller nothing it already took.
    ///
    /// The budget is what makes the read loop terminate without an await. A
    /// timer would bound it too, but only by introducing the one thing this
    /// design excludes: a suspension point between consuming a frame and
    /// returning it, where a cancellation could drop the whole prefix.
    #[tokio::test]
    async fn keepalive_stops_at_the_frame_budget_and_keeps_what_it_consumed() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // One more than the budget, so the cap is what stops the drain
            // rather than the supply running out. The frames are as small as an
            // unmodelled event can be for the same reason: the client drains
            // without ever awaiting, so it outruns anything it has to wait for,
            // and only a batch that fits in the receive buffer whole can prove
            // the budget rather than the network stopped it.
            let frame = json!({ "type": "z" }).to_string();
            for _ in 0..=MAX_KEEPALIVE_DRAIN_FRAMES {
                socket
                    .send(Message::text(frame.clone()))
                    .await
                    .expect("unknown event should send");
            }

            // Hold the socket open so the drain stops on the budget rather than
            // on an end of stream.
            futures::future::pending::<()>().await;
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        sleep(Duration::from_millis(500)).await;

        let (recovered, error) = failed(
            session.keepalive().await,
            "exceeding the frame budget must be reported",
        );
        assert_eq!(
            recovered.len(),
            MAX_KEEPALIVE_DRAIN_FRAMES,
            "the drain stops at the budget and keeps every frame up to it"
        );
        assert!(
            error.to_string().contains("buffered frames"),
            "expected a flood error, got {error}"
        );

        // A socket that cannot be read to a known state is not serviceable, so
        // the session is failed and a later drain is the ordinary no-op.
        let after = serviced(
            session.keepalive().await,
            "a failed session drains as a no-op",
        );
        assert!(
            after.is_empty(),
            "a failed session must not keep consuming frames"
        );

        server.abort();
    }

    #[tokio::test]
    async fn unknown_event_is_ignored_by_response_assembly_and_reasoning_metadata_is_preserved() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");

            // Send an unknown event type first
            socket
                .send(Message::text(
                    json!({
                        "type": "response.some_future_event",
                        "data": "should be skipped"
                    })
                    .to_string(),
                ))
                .await
                .expect("unknown event should send");

            // Then send the real completed response, including reasoning
            // metadata to verify that the WebSocket path preserves it.
            let mut response = sample_response(ResponseStatus::Completed);
            response.id = "resp_after_unknown".to_string();
            response.reasoning_metadata = Some(
                json!({
                    "context": "all_turns",
                    "effort": "ultra",
                    "summary": null,
                    "future_control": true
                })
                .as_object()
                .expect("reasoning metadata should be an object")
                .clone(),
            );
            response.reasoning_context = Some("all_turns".to_string());
            let response = serde_json::to_value(response).expect("response should serialize");

            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("send should succeed");
        let response = session
            .wait_for_completed_response()
            .await
            .expect("response should complete despite unknown event");
        assert_eq!(response.id, "resp_after_unknown");
        assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
        assert_eq!(
            response.reasoning_metadata.as_ref(),
            json!({
                "context": "all_turns",
                "effort": "ultra",
                "summary": null,
                "future_control": true
            })
            .as_object()
        );

        server.await.expect("server task should finish");
    }

    fn empty_client_config() -> rustls::ClientConfig {
        let provider = rustls::crypto::aws_lc_rs::default_provider();
        rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(provider))
            .with_safe_default_protocol_versions()
            .expect("default protocol versions should build")
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth()
    }

    #[test]
    fn ensure_default_crypto_provider_is_idempotent() {
        // The no-connector connect path calls this before building tokio-tungstenite's
        // implicit default config; calling it repeatedly must remain panic-free and
        // always leave a process-wide default provider installed.
        super::ensure_default_crypto_provider();
        super::ensure_default_crypto_provider();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }

    #[test]
    fn rustls_connector_maps_to_rustls_variant() {
        let connector =
            super::WebSocketTlsConnector::rustls(std::sync::Arc::new(empty_client_config()));

        // `Clone` and `Debug` are part of the connector's public contract.
        let cloned = connector.clone();
        assert_eq!(format!("{cloned:?}"), "WebSocketTlsConnector");

        // The injected config must surface as the rustls connector variant, not the
        // default connector that tokio-tungstenite would otherwise build.
        assert!(matches!(
            connector.into_connector(),
            super::Connector::Rustls(_)
        ));
    }

    #[test]
    fn openai_backend_defaults_to_chaining() {
        use super::{OpenAIResponsesWebSocketBackend, ResponsesWebSocketBackend};

        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url("https://api.openai.com/v1")
            .build()
            .expect("client should build");

        let backend = OpenAIResponsesWebSocketBackend::new(client.completion_model("gpt-4o"));

        // OpenAI must keep auto-chaining turns via `previous_response_id`.
        assert!(backend.chains_previous_response_id());
        assert!(backend.base_url().contains("api.openai.com"));
    }

    #[test]
    fn builder_records_injected_tls_connector() {
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .build()
            .expect("client should build");

        let default_builder = client.responses_websocket_builder("gpt-4o");
        assert!(default_builder.tls_connector.is_none());

        let connector =
            super::WebSocketTlsConnector::rustls(std::sync::Arc::new(empty_client_config()));
        let configured_builder = client
            .responses_websocket_builder("gpt-4o")
            .tls_connector(connector);
        assert!(configured_builder.tls_connector.is_some());
    }

    #[tokio::test]
    async fn keepalive_flushes_pong_for_server_ping() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            socket
                .send(Message::Ping(Vec::new().into()))
                .await
                .expect("server ping should send");

            let message = socket
                .next()
                .await
                .expect("pong should arrive")
                .expect("pong should be valid");
            assert!(
                matches!(message, Message::Pong(_)),
                "expected pong, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        // Let the server's ping reach the socket buffer before servicing it.
        sleep(Duration::from_millis(100)).await;
        serviced(
            session.keepalive().await,
            "keepalive should service the server ping",
        );
        // Servicing a ping must not invent or advance a live tip.
        assert_eq!(session.previous_response_id(), None);

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_consumes_trailing_done_and_preserves_tip() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let first_request = socket
                .next()
                .await
                .expect("first request should exist")
                .expect("first request should be valid");
            let payload = first_request
                .into_text()
                .expect("first request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_1".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": { "id": "resp_1", "status": "completed" },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");

            let second_request = socket
                .next()
                .await
                .expect("second request should exist")
                .expect("second request should be valid");
            let payload = second_request
                .into_text()
                .expect("second request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );
            assert!(
                payload.contains("\"previous_response_id\":\"resp_1\""),
                "expected chained previous_response_id after keepalive, got {payload}"
            );

            let response = serde_json::to_value(CompletionResponse {
                id: "resp_2".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 2,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.done",
                        "response": { "id": "resp_2", "status": "completed" },
                    })
                    .to_string(),
                ))
                .await
                .expect("done event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("first").build())
            .await
            .expect("first request should send");
        let first = session
            .wait_for_completed_response()
            .await
            .expect("first response should complete");
        assert_eq!(first.id, "resp_1");
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        // Let the trailing response.done reach the buffer, then service it.
        sleep(Duration::from_millis(100)).await;
        serviced(
            session.keepalive().await,
            "keepalive should consume the trailing done",
        );
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send(model.completion_request("second").build())
            .await
            .expect("second request should send");
        let second = session
            .wait_for_completed_response()
            .await
            .expect("second response should complete");
        assert_eq!(second.id, "resp_2");
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_fails_loud_on_unexpected_data_frame() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // A streaming delta with no turn in flight is a protocol violation.
            socket
                .send(Message::text(
                    json!({
                        "type": "response.output_text.delta",
                        "content_index": 0,
                        "delta": "stray",
                        "item_id": "msg_stray",
                        "logprobs": [],
                        "output_index": 0,
                        "sequence_number": 1
                    })
                    .to_string(),
                ))
                .await
                .expect("stray delta should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        // Let the stray frame reach the buffer before servicing the idle socket.
        sleep(Duration::from_millis(100)).await;
        let (recovered, error) = failed(
            session.keepalive().await,
            "an idle data frame should fail keepalive",
        );
        assert!(
            error.to_string().contains("unexpected server event"),
            "expected unexpected-server-event error, got {error}"
        );
        assert!(
            recovered.is_empty(),
            "nothing unmodelled preceded the stray frame, so nothing is recovered"
        );

        let closed = session
            .send(model.completion_request("after failure").build())
            .await
            .expect_err("session should be unusable after a failed keepalive");
        assert!(
            closed.to_string().contains("session is closed"),
            "expected closed-session error, got {closed}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_is_noop_while_in_flight() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            // Reply only after the client has had a chance to call keepalive,
            // proving keepalive did not consume the in-flight response.
            sleep(Duration::from_millis(50)).await;
            let response = serde_json::to_value(CompletionResponse {
                id: "resp_in_flight".to_string(),
                ..sample_response(ResponseStatus::Completed)
            })
            .expect("response should serialize");
            socket
                .send(Message::text(
                    json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    })
                    .to_string(),
                ))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");
        let drained = serviced(
            session.keepalive().await,
            "keepalive should be a no-op while in flight",
        );
        assert!(
            drained.is_empty(),
            "a no-op drain must return nothing: reading here would steal the \
             in-flight turn's own events"
        );

        let response = session
            .wait_for_completed_response()
            .await
            .expect("in-flight response should still be readable after keepalive");
        assert_eq!(response.id, "resp_in_flight");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn keepalive_is_noop_after_close() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let message = socket
                .next()
                .await
                .expect("close frame should arrive")
                .expect("close frame should be valid");
            assert!(
                matches!(message, Message::Close(_)),
                "expected close frame, got {message:?}"
            );
        });

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        session.close().await.expect("close should succeed");
        serviced(
            session.keepalive().await,
            "keepalive should be a no-op after close",
        );

        server.await.expect("server task should finish");
    }

    /// Re-wraps SSE conformance fixture frames as websocket text payloads: the
    /// wire events are identical across the two transports, only the framing
    /// (`data:` lines vs. one JSON message per ws frame) differs.
    fn ws_messages_from_sse_frames<'a>(
        frames: impl IntoIterator<Item = &'a bytes::Bytes>,
    ) -> Vec<String> {
        frames
            .into_iter()
            .flat_map(|frame| {
                std::str::from_utf8(frame)
                    .expect("SSE fixture frames should be UTF-8")
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                    .filter(|data| !data.is_empty() && *data != "[DONE]")
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn spawn_ws_server_with_messages(
        listener: TcpListener,
        messages: Vec<String>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create payload, got {payload}"
            );

            for message in messages {
                socket
                    .send(Message::text(message))
                    .await
                    .expect("event should send");
            }
        })
    }

    /// Websocket conformance invocation over the shared Responses fixture:
    /// the SAME frames the SSE conformance suite streams, re-wrapped as ws
    /// messages, must yield the same content through the shared
    /// `classify_responses_frame` + accumulator interpretation — text and
    /// tool-call deltas delivered, the unknown event skipped, usage and finish
    /// reason taken from the terminal.
    #[tokio::test]
    async fn websocket_conformance_replays_sse_fixture_frames() {
        let fixture =
            crate::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
        // The shared fixture scripts byte frames; re-wrap them as ws messages.
        let byte_frame = |frame: &crate::test_utils::streaming_conformance::WireInput| {
            frame
                .as_bytes()
                .cloned()
                .expect("the Responses fixture scripts byte frames")
        };
        let mut frames: Vec<bytes::Bytes> = Vec::new();
        frames.extend(fixture.text_frames.iter().map(byte_frame));
        frames.extend(fixture.tool_call_frames.iter().map(byte_frame));
        frames.extend(fixture.unknown_event_frame.iter().map(byte_frame));
        frames.extend(fixture.terminal_frames.iter().map(byte_frame));
        let messages = ws_messages_from_sse_frames(frames.iter());

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let server = spawn_ws_server_with_messages(listener, messages);

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("fixture turn should normalize");

        let texts: Vec<&str> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, fixture.expected_texts);
        let tool_names: Vec<&str> = normalized
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::ToolCall(call) => {
                    Some(call.function.name.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(tool_names, vec![fixture.expected_tool_name]);
        assert_eq!(normalized.usage.total_tokens, fixture.expected_usage_total);
        // The fixture's expected finish reason applies to its text-only
        // sequences; this combined replay carries a tool call, which the
        // shared normalization maps to `ToolCalls` on every transport.
        assert_eq!(
            normalized.finish_reason(),
            Some(crate::completion::FinishReason::ToolCalls)
        );

        server.await.expect("server task should finish");
    }

    /// Regression for the diverged websocket dispatch: `response.reasoning_text.delta`
    /// was absent from the ws-private known-event list and silently dropped,
    /// while the SSE path delivered it. Routed through the shared classifier,
    /// the reasoning delta must survive to the normalized response.
    #[tokio::test]
    async fn reasoning_text_delta_arrives_over_websocket() {
        let messages = vec![
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "thinking hard",
            })
            .to_string(),
            json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "answer",
                "item_id": "msg_1",
                "output_index": 0,
                "sequence_number": 2,
            })
            .to_string(),
            json!({
                "type": "response.completed",
                "sequence_number": 3,
                "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
                    .expect("response should serialize"),
            })
            .to_string(),
        ];

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let server = spawn_ws_server_with_messages(listener, messages);

        let base_url = format!("http://{address}/v1");
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-4o");
        let mut session = client
            .responses_websocket("gpt-4o")
            .await
            .expect("session should connect");

        let normalized = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("turn with reasoning deltas should normalize");

        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(reasoning)
                    if reasoning.content.iter().any(|block| matches!(
                        block,
                        crate::message::ReasoningContent::Text { text, .. }
                            if text.contains("thinking hard")
                    ))
            )),
            "reasoning delta should survive over websocket, got {:?}",
            normalized.choice
        );
        assert!(
            normalized.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Text(text) if text.text == "answer"
            )),
            "text delta should survive alongside reasoning, got {:?}",
            normalized.choice
        );

        server.await.expect("server task should finish");
    }

    #[test]
    fn parse_reasoning_text_delta_event_is_item() {
        let payload = json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        });

        let event = parse_server_event(&payload.to_string())
            .expect("reasoning delta should parse")
            .expect("reasoning delta should not be skipped");

        assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
        assert!(!event.is_terminal());
    }
}
