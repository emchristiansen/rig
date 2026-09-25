//! Stateful Responses WebSocket sessions over caller-supplied connections.
//! Sessions permit one in-flight turn and chain completed or incomplete response IDs.
//! Between turns, [`ResponsesWebSocketSession::keepalive`] services the idle
//! connection and hands back every unmodelled frame it had to consume.
//! The Codex (ChatGPT subscription) session is [`codex::CodexWebSocketSession`].
//!
//! ```
//! use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketCreateOptions;
//! let options = ResponsesWebSocketCreateOptions::warmup();
//! assert_eq!(options.generate, Some(false));
//! ```

use crate::completion;
use crate::driver::{Bound, WireDriver};
use crate::driver::{TriagedFrame, triage_frame};
use crate::error::{EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::operation::Completion;
use crate::providers::openai::responses_api::streaming::{
    IncompleteTerminal, ItemChunk, ResponseChunk, ResponseChunkKind, ResponsesDecoder,
    StreamingCompletionChunk, classify_responses_frame,
};
use crate::providers::openai::responses_api::wire::Responses;
use crate::streaming::StreamEvent;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::WireFrame;
use crate::wire::{Fold, Operation, Reply, Wire};
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, ReadyFrame, UnsupportedCapability,
    WebSocketClientExt, WebSocketConnection,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;

use crate::providers::openai::responses_api::{CompletionResponse, ResponseStatus};

pub mod codex;

/// The websocket endpoint's path, appended to the client's configured base URL.
const WEBSOCKET_PATH: &str = "responses";

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Request-ID header read from rejected WebSocket upgrades.
const REQUEST_ID_HEADER: Option<&'static str> =
    crate::providers::openai::wire::OPENAI.request_id_header;

/// The most frames one idle [`keepalive`](ResponsesWebSocketSession::keepalive)
/// consumes before it stops and hands back what it has taken.
///
/// A peer that delivers faster than the drain consumes is the only way the
/// drain could fail to terminate: [`WebSocketConnection::recv_ready`] never
/// waits on the peer. A frame budget bounds it without a timer.
const MAX_KEEPALIVE_DRAIN_FRAMES: usize = 4_096;

const _: () = assert!(
    MAX_KEEPALIVE_DRAIN_FRAMES > 0,
    "a zero budget would consume nothing and report a flood on every idle drain"
);

/// How long the one post-drain flush may take before the connection is treated
/// as unserviceable.
///
/// Internal on purpose: expiry returns the recovered frames beside a typed
/// failure, so a caller never needs an outer timeout of its own to bound
/// [`keepalive`](ResponsesWebSocketSession::keepalive).
const KEEPALIVE_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

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
struct ResponsesWebSocketClientEvent<'a> {
    #[serde(rename = "type")]
    kind: ResponsesWebSocketClientEventKind,
    #[serde(flatten)]
    request: &'a crate::providers::openai::responses_api::CompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate: Option<bool>,
}

impl<'a> ResponsesWebSocketClientEvent<'a> {
    fn create(
        request: &'a crate::providers::openai::responses_api::CompletionRequest,
        generate: Option<bool>,
    ) -> Self {
        Self {
            kind: ResponsesWebSocketClientEventKind::ResponseCreate,
            request,
            generate,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
enum ResponsesWebSocketClientEventKind {
    #[serde(rename = "response.create")]
    ResponseCreate,
}

/// A protocol error event emitted by OpenAI WebSocket mode.
///
/// The Codex backend wraps HTTP-shaped failures in this event: a top-level
/// `status` (also spelled `status_code`), the `error` object, and a
/// `headers` map that carries its `x-codex-*` rate-limit values. Every field
/// but the tag is optional, and unmodelled top-level fields are kept in
/// `extra`, so the event is carried whole rather than trimmed to a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesWebSocketErrorEvent {
    /// The event type.
    #[serde(rename = "type")]
    pub kind: ResponsesWebSocketErrorEventKind,
    /// The HTTP status the provider reports for this failure, when it does.
    #[serde(
        default,
        alias = "status_code",
        skip_serializing_if = "Option::is_none"
    )]
    pub status: Option<u16>,
    /// The provider error payload; empty when the event carries none,
    /// whether the field is missing or explicitly `null`.
    #[serde(
        default,
        deserialize_with = "crate::json_utils::null_or_default",
        skip_serializing_if = "ResponsesWebSocketErrorPayload::is_empty"
    )]
    pub error: ResponsesWebSocketErrorPayload,
    /// Response headers the provider reports with this failure, when it does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Map<String, Value>>,
    /// Any other top-level fields supplied by the provider.
    #[serde(flatten, default)]
    pub extra: Map<String, Value>,
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

impl ResponsesWebSocketErrorPayload {
    /// Whether the payload carries nothing: no code, no message, no extras.
    pub fn is_empty(&self) -> bool {
        self.code.is_none() && self.message.is_none() && self.extra.is_empty()
    }
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
    Response(ResponseChunk),
    /// A streaming item/delta event such as `response.output_text.delta`.
    Item(ItemChunk),
    /// A protocol-level websocket error event.
    Error(ResponsesWebSocketErrorEvent),
    /// An optional `response.done` event emitted by OpenAI over WebSockets.
    Done(ResponsesWebSocketDoneEvent),
    /// A server event whose `type` this client does not model, surfaced with
    /// its tag and complete parsed payload rather than discarded, and retained
    /// for [`StreamEvent::Unknown`] passthrough.
    Unknown(UnrecognizedEvent),
}

/// A server event this client does not model.
///
/// A named value rather than inline variant fields because
/// [`ResponsesWebSocketSession::keepalive`] returns it on its own: a
/// `Vec<UnrecognizedEvent>` states in the type that every element is
/// unmodelled.
#[derive(Debug, Clone, PartialEq)]
pub struct UnrecognizedEvent {
    /// The event's `type` field, so a consumer failing closed on a frame it
    /// cannot place can say what the frame was.
    pub kind: String,
    /// The complete parsed JSON value of the event: every field, not a
    /// modelled subset. Parsing discards lexical formatting, so this is the
    /// event's JSON value rather than its wire spelling.
    pub payload: crate::streaming::UnknownPayload,
}

/// The new input for a forward-only incremental turn — never empty.
///
/// An incremental turn chains onto a live tip and replaces the captured
/// envelope's `input` wholesale, so an empty delta would send a chained
/// `response.create` carrying nothing new. The field is private, so the
/// constructors below are the only way in, and there is deliberately no
/// `Deserialize` that could bypass them.
#[derive(Debug, Clone)]
pub struct InputDelta(Vec<super::InputItem>);

/// Returned when an incremental delta would carry no input items.
#[derive(Debug, thiserror::Error)]
#[error("an incremental delta must carry at least one input item")]
pub struct EmptyInputDeltaError;

impl InputDelta {
    /// A delta of exactly one item. Infallible by construction.
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

/// Why an incremental continuation may not follow the session's last turn.
///
/// Returned (inside [`ProviderError::Request`]) by an incremental send before
/// anything is written, so a refused continuation never reaches the wire.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IncrementalSendRefused {
    /// No full send on this session has completed yet, so there is no tip and
    /// no captured envelope to continue.
    #[error(
        "cannot send an incremental turn before a completed full send established a live tip and captured its envelope"
    )]
    NoCompletedTurn,
    /// The last turn ended `incomplete`. Its response ID is kept as terminal
    /// evidence, but a truncated turn is not an eligible tip: a full send must
    /// establish one.
    #[error(
        "cannot send an incremental turn after the last turn ended incomplete (response_id={response_id:?}); a full send must establish an eligible tip"
    )]
    LastTurnIncomplete {
        /// The incomplete response's ID, when the provider reported one.
        response_id: Option<String>,
    },
    /// The last turn completed without a response ID to chain onto.
    #[error(
        "cannot send an incremental turn: the last turn completed without a response ID to chain onto"
    )]
    CompletedWithoutResponseId,
    /// The last turn failed, was reported as an error, or ended without a
    /// completed verdict.
    #[error(
        "cannot send an incremental turn after the last turn failed or ended without completing; a full send must establish a new tip"
    )]
    LastTurnNotCompleted,
    /// The caller cleared the tip with
    /// [`clear_previous_response_id`](ResponsesWebSocketSession::clear_previous_response_id).
    #[error("cannot send an incremental turn after the live tip was cleared")]
    TipCleared,
}

/// A send refused because an interrupted keepalive left recovered frames the
/// caller has not collected yet.
///
/// Those frames were taken off the socket before the new turn; sending first
/// would let the next turn's reads run ahead of frames nobody has accounted
/// for. Call [`keepalive`](ResponsesWebSocketSession::keepalive) to collect
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "{count} unmodelled frame(s) recovered by an interrupted keepalive have not been collected; call keepalive() before sending"
)]
pub struct UncollectedRecoveredFrames {
    /// How many recovered frames the session is holding.
    pub count: usize,
}

/// What the next incremental continuation may chain onto.
#[derive(Debug, Clone)]
enum Continuation {
    /// Nothing may be continued, for the stated reason.
    Blocked(IncrementalSendRefused),
    /// A turn built from this envelope is in flight. It becomes the eligible
    /// envelope only if that turn completes.
    InFlight(Box<super::CompletionRequest>),
    /// The last turn completed as `tip`; a delta may continue it by reusing
    /// `envelope`'s non-input configuration.
    Eligible {
        envelope: Box<super::CompletionRequest>,
        tip: String,
    },
}

impl Continuation {
    /// The next state once the in-flight turn ends in `ending`.
    fn settle(self, ending: TurnEnding) -> Self {
        match ending {
            TurnEnding::Completed {
                response_id: Some(tip),
            } => match self {
                Self::InFlight(envelope) => Self::Eligible { envelope, tip },
                // Blocked while the turn was in flight (the caller cleared
                // the tip): the envelope is gone, and so is the reason.
                Self::Blocked(reason) => Self::Blocked(reason),
                // Every send captures its envelope as in flight, so a
                // completion never finds an eligible one; if it did, nothing
                // this session started would be what completed.
                Self::Eligible { .. } => {
                    Self::Blocked(IncrementalSendRefused::LastTurnNotCompleted)
                }
            },
            TurnEnding::Completed { response_id: None } => {
                Self::Blocked(IncrementalSendRefused::CompletedWithoutResponseId)
            }
            TurnEnding::Incomplete { response_id } => {
                Self::Blocked(IncrementalSendRefused::LastTurnIncomplete { response_id })
            }
            TurnEnding::NotCompleted => Self::Blocked(IncrementalSendRefused::LastTurnNotCompleted),
        }
    }
}

/// How an in-flight turn ended, as far as continuation eligibility cares.
enum TurnEnding {
    Completed { response_id: Option<String> },
    Incomplete { response_id: Option<String> },
    NotCompleted,
}

/// How far an idle [`keepalive`](ResponsesWebSocketSession::keepalive) drain
/// got, and every unmodelled frame it consumed getting there.
///
/// **Deliberately not a `Result`.** The recovered frames have already been
/// taken off the socket and can never be re-read, so a shape with an error
/// position would let `?` return the failure while silently discarding them.
/// The ending is reachable only through [`Self::into_parts`], which surrenders
/// the frames in the same expression.
#[derive(Debug)]
pub struct KeepaliveDrain {
    recovered: Vec<UnrecognizedEvent>,
    ending: KeepaliveEnding,
}

/// How a [`KeepaliveDrain`] ended. Private, so an ending cannot be taken
/// without the frames beside it.
#[derive(Debug)]
enum KeepaliveEnding {
    /// Read to the end of what had arrived, and the flush succeeded.
    Serviced,
    /// The drain stopped here. Unless the error names an
    /// [`UnsupportedCapability`] of the backend, the session has been marked
    /// failed or closed.
    Failed(ProviderError),
}

impl KeepaliveDrain {
    fn serviced(recovered: Vec<UnrecognizedEvent>) -> Self {
        Self {
            recovered,
            ending: KeepaliveEnding::Serviced,
        }
    }

    fn failed(recovered: Vec<UnrecognizedEvent>, error: ProviderError) -> Self {
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

    /// Split into the recovered frames and the failure that ended the drain,
    /// if one did. Frames recovered ahead of a failure are ordinary recovered
    /// frames: the failure says the socket stopped being serviceable, never
    /// that they did not arrive.
    #[must_use]
    pub fn into_parts(self) -> (Vec<UnrecognizedEvent>, Option<ProviderError>) {
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
pub struct ResponsesWebSocketSessionBuilder {
    wire: Responses,
    connect_timeout: Option<Duration>,
    event_timeout: Option<Duration>,
}

impl ResponsesWebSocketSessionBuilder {
    pub(crate) fn new(wire: Responses) -> Self {
        Self {
            wire,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            event_timeout: None,
        }
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

impl ResponsesWebSocketSessionBuilder {
    /// Open a session over `backend` with the configured timeouts.
    /// Return handshake construction, transport, or provider errors.
    pub async fn connect_with<W>(
        self,
        backend: &W,
    ) -> Result<ResponsesWebSocketSession, ProviderError>
    where
        W: WebSocketClientExt,
    {
        ResponsesWebSocketSession::connect_with_timeouts(
            backend,
            self.wire,
            self.connect_timeout,
            self.event_timeout,
        )
        .await
    }
}

/// Sequential Responses session with automatic response-ID chaining.
/// Completed and incomplete responses update the chain unless a request supplies
/// its own `previous_response_id`. Call [`Self::close`] to perform a close handshake.
pub struct ResponsesWebSocketSession {
    wire: Responses,
    previous_response_id: Option<String>,
    pending_done_response_id: Option<String>,
    /// What an incremental continuation may chain onto.
    continuation: Continuation,
    /// Unmodelled frames an idle drain took off the socket and has not yet
    /// handed back. Held here rather than in the drain's own future so that a
    /// drain dropped at any suspension point cannot destroy them.
    recovered: Vec<UnrecognizedEvent>,
    socket: BoxedWebSocketConnection,
    in_flight: bool,
    event_timeout: Option<Duration>,
    closed: bool,
    failed: bool,
}

impl ResponsesWebSocketSession {
    async fn connect_with_timeouts<W>(
        backend: &W,
        wire: Responses,
        connect_timeout: Option<Duration>,
        event_timeout: Option<Duration>,
    ) -> Result<Self, ProviderError>
    where
        W: WebSocketClientExt,
    {
        let request = websocket_request(&wire)?;
        let socket = connect(backend, request, connect_timeout).await?;
        Ok(Self::from_connection(wire, socket, event_timeout))
    }

    /// Build a session over an already-open, authenticated connection.
    /// `event_timeout: None` waits indefinitely for each event.
    pub fn from_connection(
        wire: Responses,
        connection: BoxedWebSocketConnection,
        event_timeout: Option<Duration>,
    ) -> Self {
        Self {
            wire,
            previous_response_id: None,
            pending_done_response_id: None,
            continuation: Continuation::Blocked(IncrementalSendRefused::NoCompletedTurn),
            recovered: Vec::new(),
            socket: connection,
            in_flight: false,
            event_timeout,
            closed: false,
            failed: false,
        }
    }

    /// Return the response ID retained for automatic chaining, if any.
    #[must_use]
    pub fn previous_response_id(&self) -> Option<&str> {
        self.previous_response_id.as_deref()
    }

    /// Clears the cached `previous_response_id` so the next turn starts a fresh
    /// chain. An incremental continuation is refused until a full send
    /// establishes a new tip.
    pub fn clear_previous_response_id(&mut self) {
        self.previous_response_id = None;
        self.continuation = Continuation::Blocked(IncrementalSendRefused::TipCleared);
    }

    /// Sends a `response.create` event for a Rig completion request.
    pub async fn send(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<(), ProviderError> {
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
    ) -> Result<(), ProviderError> {
        self.ensure_can_send()?;
        let request = self.prepare_request(completion_request, Chaining::Automatic)?;
        let frame = encode_frame(&request, options.generate, |_| Ok(()))?;
        self.send_encoded(request, frame).await
    }

    /// Refuse a send the session cannot take: closed, failed, a turn already
    /// in flight, or recovered frames the caller has not collected.
    pub(crate) fn ensure_can_send(&self) -> Result<(), ProviderError> {
        self.ensure_open()?;

        if self.in_flight {
            return Err(ProviderError::Provider(
                "An OpenAI websocket response is already in flight on this session".to_string(),
            ));
        }

        if !self.recovered.is_empty() {
            return Err(ProviderError::Request(Box::new(
                UncollectedRecoveredFrames {
                    count: self.recovered.len(),
                },
            )));
        }

        Ok(())
    }

    /// The envelope and tip an incremental continuation would chain onto, or
    /// the reason there is none.
    pub(crate) fn continuation(
        &self,
    ) -> Result<(&super::CompletionRequest, &str), IncrementalSendRefused> {
        match &self.continuation {
            Continuation::Eligible { envelope, tip } => Ok((envelope, tip)),
            Continuation::Blocked(reason) => Err(reason.clone()),
            // `ensure_can_send` refuses an in-flight session first; a caller
            // that skipped it is told the turn has not completed.
            Continuation::InFlight(_) => Err(IncrementalSendRefused::LastTurnNotCompleted),
        }
    }

    /// Write one encoded `response.create` frame, taking custody of the turn.
    ///
    /// `envelope` is the full request the turn continues from: it becomes the
    /// eligible envelope for a later incremental continuation only if this
    /// turn completes.
    pub(crate) async fn send_encoded(
        &mut self,
        envelope: super::CompletionRequest,
        frame: String,
    ) -> Result<(), ProviderError> {
        self.ensure_can_send()?;

        if let Err(error) = self.socket.send(Frame::Text(frame)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        self.continuation = Continuation::InFlight(Box::new(envelope));
        self.in_flight = true;

        Ok(())
    }

    /// Reads the next server event for the current in-flight turn.
    pub async fn next_event(&mut self) -> Result<ResponsesWebSocketEvent, ProviderError> {
        self.next_event_with_payload().await.map(|(event, _)| event)
    }

    /// Reads the next lifecycle event and retains its payload for content decoding
    /// by [`ResponsesDecoder`]. Returns the same session errors as [`Self::next_event`].
    async fn next_event_with_payload(
        &mut self,
    ) -> Result<(ResponsesWebSocketEvent, String), ProviderError> {
        self.ensure_open()?;

        if !self.in_flight {
            return Err(ProviderError::Provider(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            ));
        }

        loop {
            let message = match self.read_next_frame().await? {
                Ok(message) => message,
                Err(error) => return Err(self.fail_session(websocket_provider_error(error))),
            };

            let Some(message) = message else {
                self.mark_closed();
                return Err(ProviderError::Provider(
                    "The OpenAI websocket connection closed before the turn finished".to_string(),
                ));
            };

            let payload = match websocket_frame_to_text(message) {
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
            return Ok((event, payload));
        }
    }

    /// Services the live connection between turns without consuming a response.
    ///
    /// The provider sends keepalive pings while a session is idle, and between
    /// turns nothing reads the socket. This takes every frame that has already
    /// arrived — letting the backend queue its automatic pong for each ping —
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
    /// consumed**: every exit — close, read error, parse error, unexpected
    /// modelled event, flood, flush error, flush timeout — returns the complete
    /// ordered prefix beside the failure.
    ///
    /// The whole call is bounded without an outer timeout: reads never wait on
    /// the peer, a frame budget bounds the read loop, and the one flush is
    /// bounded internally. Each recovered frame moves into the session the
    /// moment it is consumed, so even a caller that drops this future midway
    /// loses nothing: the next call returns those frames first, and a send is
    /// refused ([`UncollectedRecoveredFrames`]) until they are collected.
    ///
    /// A backend without [`WebSocketConnection::recv_ready`] or
    /// [`WebSocketConnection::flush`] ends the drain with a
    /// [`ProviderError::Request`] naming the missing capability; the session
    /// is left open, since nothing was lost.
    pub async fn keepalive(&mut self) -> KeepaliveDrain {
        if self.closed || self.failed || self.in_flight {
            return KeepaliveDrain::serviced(std::mem::take(&mut self.recovered));
        }

        for _ in 0..MAX_KEEPALIVE_DRAIN_FRAMES {
            let frame = match self.socket.recv_ready().await {
                Ok(ReadyFrame::Frame(frame)) => frame,
                Ok(ReadyFrame::Empty) => return self.flush_after_drain().await,
                Ok(ReadyFrame::Ended) => {
                    self.mark_closed();
                    return self.drain_failed(ProviderError::Provider(
                        "The OpenAI websocket connection closed during idle keepalive".to_string(),
                    ));
                }
                Err(error) if UnsupportedCapability::is(&error, "recv_ready") => {
                    return self.drain_failed(unsupported_capability_error("recv_ready"));
                }
                Err(error) => {
                    let error = self.fail_session(websocket_provider_error(error));
                    return self.drain_failed(error);
                }
            };

            let payload = match websocket_frame_to_text(frame) {
                // A ping or pong carries no turn data; the backend has queued
                // the pong a ping is owed, and the flush below writes it.
                Ok(None) => continue,
                Ok(Some(payload)) => payload,
                Err(error) => {
                    let error = self.fail_session(error);
                    return self.drain_failed(error);
                }
            };

            let event = match parse_server_event(&payload) {
                Ok(Some(event)) => event,
                Ok(None) => continue,
                Err(error) => {
                    let error = self.fail_session(error);
                    return self.drain_failed(error);
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
                    return self.drain_failed(error);
                }
            }
        }

        // The peer delivers faster than the drain consumes, so the socket
        // cannot be read to a known state. What was consumed is still the
        // caller's.
        let error = self.fail_session(keepalive_flood_error(MAX_KEEPALIVE_DRAIN_FRAMES));
        self.drain_failed(error)
    }

    /// The one bounded suspension after the reads: write out the pongs the
    /// drain's reads queued.
    async fn flush_after_drain(&mut self) -> KeepaliveDrain {
        match crate::wasm_compat::timeout(KEEPALIVE_FLUSH_TIMEOUT, self.socket.flush()).await {
            Ok(Ok(())) => KeepaliveDrain::serviced(std::mem::take(&mut self.recovered)),
            Ok(Err(error)) if UnsupportedCapability::is(&error, "flush") => {
                self.drain_failed(unsupported_capability_error("flush"))
            }
            Ok(Err(error)) => {
                let error = self.fail_session(websocket_provider_error(error));
                self.drain_failed(error)
            }
            Err(_elapsed) => {
                let error =
                    self.fail_session(keepalive_flush_timeout_error(KEEPALIVE_FLUSH_TIMEOUT));
                self.drain_failed(error)
            }
        }
    }

    /// End a drain with `error`, surrendering every frame it recovered.
    fn drain_failed(&mut self, error: ProviderError) -> KeepaliveDrain {
        KeepaliveDrain::failed(std::mem::take(&mut self.recovered), error)
    }

    /// Sends a warmup turn (`generate: false`) and returns the resulting response ID.
    pub async fn warmup(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<String, ProviderError> {
        self.send_with_options(
            completion_request,
            ResponsesWebSocketCreateOptions::warmup(),
        )
        .await?;
        let response = self.wait_for_completed_response().await?;
        Ok(response.id)
    }

    /// Sends a completion turn and collects the final OpenAI response,
    /// normalized; its `raw` is the provider's own terminal response object.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        self.send(completion_request).await?;
        self.collect_completion().await
    }

    /// Collect the in-flight turn and normalize it.
    ///
    /// An incomplete turn is a terminal, not a failure: its partial output is
    /// kept and its finish reason says why it stopped.
    pub(crate) async fn collect_completion(
        &mut self,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        let provider = self.wire.name().to_owned();
        let (response, events) = self.wait_for_terminal_response().await?;
        let folded = fold_events(&provider, events, &response)?;
        if folded.choice.is_empty() {
            // The turn carried no content events but its terminal body
            // restates `output[]` (the shape a warmed-up or replayed session
            // answers with): fold that body, through the same decoder's
            // unary variant.
            return super::wire::fold_body(&provider, response);
        }
        Ok(folded)
    }

    /// Closes the websocket connection.
    ///
    /// Call this when you are finished with the session so the websocket can
    /// terminate with a clean close handshake.
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

    /// Validate and shape a Rig request into this session's Responses request.
    ///
    /// [`Chaining::Automatic`] continues the retained response ID unless the
    /// request names its own; [`Chaining::Root`] never adds one.
    pub(crate) fn prepare_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
        chaining: Chaining,
    ) -> Result<crate::providers::openai::responses_api::CompletionRequest, ProviderError> {
        // Direct session requests bypass builder validation.
        completion_request.validate_message_content()?;

        let mut request = self.wire.responses_request(completion_request, false)?;

        // WebSocket mode is always event-driven, so these HTTP/SSE-specific flags
        // are ignored by the provider and only add noise to the payload.
        request.stream = None;
        request.additional_parameters.background = None;

        match chaining {
            Chaining::Automatic => {
                if request.additional_parameters.previous_response_id.is_none() {
                    request
                        .additional_parameters
                        .previous_response_id
                        .clone_from(&self.previous_response_id);
                }
            }
            Chaining::Root => {}
        }

        Ok(request)
    }

    pub(crate) async fn wait_for_completed_response(
        &mut self,
    ) -> Result<CompletionResponse, ProviderError> {
        Ok(self.wait_for_terminal_response().await?.0)
    }

    /// The decoder one websocket turn's events are folded through.
    ///
    /// The websocket transport accepts `response.incomplete` as a terminal
    /// and reports it with its truthful finish reason, whatever the wire's
    /// streamed-SSE policy says: that policy is the HTTP stream's contract.
    fn turn_decoder(&self) -> ResponsesDecoder {
        self.wire
            .decoder_with_incomplete(IncompleteTerminal::Accept)
    }

    /// Collect decoded events and the provider's completed or incomplete response.
    /// Transport, protocol, and decoder failures return an error and discard
    /// collected events. A terminal event without a response body is an error.
    async fn wait_for_terminal_response(
        &mut self,
    ) -> Result<(CompletionResponse, Vec<StreamEvent>), ProviderError> {
        // Frames arrive incrementally; finish only after a provider terminal event.
        let mut driver = WireDriver::<Completion, _>::new(self.turn_decoder());
        let mut events = Vec::new();
        loop {
            let (event, payload) = self.next_event_with_payload().await?;
            match event {
                ResponsesWebSocketEvent::Response(chunk) => {
                    let terminal = matches!(
                        chunk.kind,
                        ResponseChunkKind::ResponseCompleted
                            | ResponseChunkKind::ResponseFailed
                            | ResponseChunkKind::ResponseIncomplete
                    );
                    if !terminal {
                        drain(&mut driver, &mut events, payload)?;
                        continue;
                    }
                    // A failed turn is reported from its own envelope; only a
                    // completed or incomplete one reaches the decoder, whose
                    // terminal record closes the turn.
                    let response = terminal_response_result(chunk.response)?;
                    drain(&mut driver, &mut events, payload)?;
                    driver.finish();
                    for item in driver.drain() {
                        events.push(item?);
                    }
                    return Ok((response, events));
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if let Some(response) = done.as_completion_response() {
                        // A failed turn is reported from its own envelope, as
                        // on the `response.failed` path.
                        let response = terminal_response_result(response)?;
                        // `response.done` carries the response object itself,
                        // which is the decoder's unary shape: hand it over as
                        // the frame it is.
                        let body = serde_json::to_string(&done.response)?;
                        drain(&mut driver, &mut events, body)?;
                        driver.finish();
                        for item in driver.drain() {
                            events.push(item?);
                        }
                        return Ok((response, events));
                    }

                    let message = if let Some(response_id) = done.response_id() {
                        format!(
                            "OpenAI websocket turn ended with response.done before a terminal response body was available (response_id={response_id})"
                        )
                    } else {
                        "OpenAI websocket turn ended with response.done before a terminal response body was available"
                            .to_string()
                    };

                    return Err(ProviderError::Provider(message));
                }
                ResponsesWebSocketEvent::Error(error) => {
                    // Genuine provider error event: preserve the serialized payload
                    // (status, code, message, headers and any extra fields) so
                    // provider_response_json() parses it, matching the
                    // response.failed path. A status the event reports is kept.
                    return Err(provider_error_from_event(&error));
                }
                // Unknown frames retain their raw payload through decoder passthrough.
                ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {
                    drain(&mut driver, &mut events, payload)?;
                }
            }
        }
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
                // An incomplete turn still produced a response the next
                // automatically chained turn can continue, so it keeps
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
                    Some(ResponseStatus::Completed) => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Completed { response_id }
                    }
                    Some(ResponseStatus::Incomplete) => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Incomplete { response_id }
                    }
                    Some(
                        ResponseStatus::Failed
                        | ResponseStatus::Cancelled
                        | ResponseStatus::Other(_),
                    ) => {
                        self.previous_response_id = None;
                        TurnEnding::NotCompleted
                    }
                    Some(ResponseStatus::InProgress | ResponseStatus::Queued) | None => {
                        TurnEnding::NotCompleted
                    }
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
        self.settle_continuation(ending);
    }

    fn settle_continuation(&mut self, ending: TurnEnding) {
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

        match crate::wasm_compat::timeout(timeout_duration, self.socket.recv()).await {
            Ok(message) => Ok(message),
            Err(_) => Err(self.fail_session(event_timeout_error(timeout_duration))),
        }
    }
}

impl Drop for ResponsesWebSocketSession {
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

/// Whether a full send continues the session's retained response ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Chaining {
    /// Continue the retained response ID unless the request names its own
    /// (OpenAI websocket mode).
    Automatic,
    /// Root a fresh chain: add no `previous_response_id` (Codex full sends).
    Root,
}

/// Serialize one `response.create` frame for `request`, letting `stamp` add
/// provider fields at the top level of the body before it is written.
fn encode_frame(
    request: &super::CompletionRequest,
    generate: Option<bool>,
    stamp: impl FnOnce(&mut Map<String, Value>) -> Result<(), EncodeError>,
) -> Result<String, ProviderError> {
    let event = ResponsesWebSocketClientEvent::create(request, generate);
    let mut body = match serde_json::to_value(&event)? {
        Value::Object(body) => body,
        other => {
            return Err(EncodeError::request(format!(
                "an OpenAI websocket response.create event must serialize to a JSON object, got {other}"
            ))
            .into());
        }
    };
    stamp(&mut body)?;

    crate::providers::internal::trace_json(
        crate::providers::internal::LogTarget::Completions,
        "OpenAI websocket request",
        &body,
    );

    Ok(serde_json::to_string(&body)?)
}

/// Feed one message to the wire's decoder and take what it produced.
///
/// This surface is unary, so an `Err` item the decoder pushed (a corrupt
/// frame, a terminal record that failed to serialize) fails the turn, as it
/// would on a buffered HTTP reply.
fn drain(
    driver: &mut WireDriver<Completion, ResponsesDecoder>,
    events: &mut Vec<StreamEvent>,
    payload: String,
) -> Result<(), ProviderError> {
    driver.push(WireFrame::Text(payload));
    for item in driver.drain() {
        events.push(item?);
    }
    Ok(())
}

/// Fold events into a normalized response, retaining the terminal body as raw JSON.
/// Return fold or serialization errors.
fn fold_events(
    provider: &str,
    events: Vec<StreamEvent>,
    response: &CompletionResponse,
) -> Result<completion::CompletionResponse, ProviderError> {
    let mut fold = <Completion as Operation>::Fold::default();
    for event in events {
        fold.absorb(event)?;
    }
    fold.finish(Reply {
        provider: provider.to_owned(),
        raw: serde_json::to_value(response)?,
        // The websocket carries no reply headers past the handshake.
        provider_request_id: None,
        response_headers: crate::completion::ProviderResponseHeaders::new(),
    })
}

fn terminal_response_result(
    response: CompletionResponse,
) -> Result<CompletionResponse, ProviderError> {
    match response.status {
        ResponseStatus::Completed => Ok(response),
        // Preserve provider error envelopes as reserialized JSON without an HTTP status.
        // Without an error object, return a local diagnostic instead.
        ResponseStatus::Failed => match response.error.as_ref() {
            Some(error) => Err(ProviderError::from_provider_body(
                serde_json::to_string(&response).unwrap_or_else(|_| error.message.clone()),
            )),
            None => Err(ProviderError::Provider(response_error_message(
                "failed response",
            ))),
        },
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // normalization path maps the status/incomplete_details to a finish
        // reason via `map_finish_reason`, matching the unary and SSE paths.
        ResponseStatus::Incomplete => Ok(response),
        other => Err(ProviderError::Provider(format!(
            "OpenAI websocket response ended in state {other:?}"
        ))),
    }
}

fn response_error_message(fallback: &str) -> String {
    format!("OpenAI websocket returned a {fallback}")
}

/// Preserve an error event as reserialized provider JSON. When the event
/// reports an HTTP status, the error carries it, with the event's headers, so
/// it classifies by status as the same failure over HTTP would; otherwise it
/// has none. Fall back to its display text if serialization fails.
fn provider_error_from_event(error: &ResponsesWebSocketErrorEvent) -> ProviderError {
    let body = serde_json::to_string(&error).unwrap_or_else(|_| error.to_string());
    match error
        .status
        .and_then(|status| http::StatusCode::from_u16(status).ok())
    {
        Some(status) => ProviderError::from_http_response(status, body)
            .with_response_headers(error.headers.as_ref().map(header_map_from_json)),
        None => ProviderError::from_provider_body(body),
    }
}

/// Converts an error event's JSON `headers` map to an HTTP header map. String
/// values are taken as-is and scalars by their JSON text. An entry that is not
/// a valid header name or value is left out of the map; it is still present
/// in the preserved body, which keeps the whole event.
fn header_map_from_json(headers: &Map<String, Value>) -> http::HeaderMap {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let text = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            Some((
                http::HeaderName::from_bytes(name.as_bytes()).ok()?,
                http::HeaderValue::from_str(&text).ok()?,
            ))
        })
        .collect()
}

/// Decode WebSocket error and done events or delegate to Responses classification.
/// Return parsing and triage errors; preserve unknown payloads with their tag.
fn parse_server_event(payload: &str) -> Result<Option<ResponsesWebSocketEvent>, ProviderError> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let event_type = serde_json::from_str::<EventType>(payload)?;
    match event_type.kind.as_str() {
        "error" => serde_json::from_str(payload)
            .map(|e| Some(ResponsesWebSocketEvent::Error(e)))
            .map_err(ProviderError::from),
        "response.done" => serde_json::from_str(payload)
            .map(|d| Some(ResponsesWebSocketEvent::Done(d)))
            .map_err(ProviderError::from),
        _ => Ok(Some(
            match triage_frame(classify_responses_frame(payload))? {
                TriagedFrame::Event(StreamingCompletionChunk::Response(response)) => {
                    ResponsesWebSocketEvent::Response(response)
                }
                TriagedFrame::Event(StreamingCompletionChunk::Delta(item)) => {
                    ResponsesWebSocketEvent::Item(item)
                }
                // The tag comes from the probe above: reaching this arm proves
                // it decoded exactly one top-level string `type`, the same key
                // the classifier read.
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

/// Lower one websocket frame onto the JSON payload the protocol carries.
///
/// `Ok(None)` is a frame with no protocol payload (a keepalive), which the
/// session skips; a close frame mid-turn is an error naming the peer's reason.
fn websocket_frame_to_text(frame: Frame) -> Result<Option<String>, ProviderError> {
    match frame {
        Frame::Text(text) => Ok(Some(text)),
        Frame::Binary(bytes) => String::from_utf8(bytes.to_vec())
            .map(Some)
            .map_err(|error| ProviderError::Response(error.to_string())),
        Frame::Ping(_) | Frame::Pong(_) => Ok(None),
        Frame::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason)
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(ProviderError::Provider(format!(
                "The OpenAI websocket connection closed {reason}"
            )))
        }
    }
}

/// Build the handshake request: the websocket URL derived from the client's
/// base URL, carrying the client's own auth headers.
///
/// The backend supplies the websocket-specific handshake headers; this only
/// states where to connect and who is connecting.
fn websocket_request(wire: &Responses) -> Result<http_client::Request<NoBody>, EncodeError> {
    let url = crate::ws_client::websocket_url(&wire.provider.base_url, WEBSOCKET_PATH)
        .map_err(EncodeError::request)?;

    let request = wire.provider.headers(
        http_client::Request::builder()
            .method(http::Method::GET)
            .uri(url),
    );

    request.body(NoBody).map_err(|error| {
        EncodeError::request(format!("Failed to build OpenAI websocket request: {error}"))
    })
}

/// Open a connection for `request` over `backend`, mapping a rejected upgrade
/// to a provider error that keeps its status, body and request ID.
async fn connect<W>(
    backend: &W,
    request: http_client::Request<NoBody>,
    connect_timeout: Option<Duration>,
) -> Result<BoxedWebSocketConnection, ProviderError>
where
    W: WebSocketClientExt,
{
    backend
        .connect(request, ConnectOptions::new().with_timeout(connect_timeout))
        .await
        .map_err(websocket_provider_error)
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
    ProviderError::Request(Box::new(UnsupportedCapability { capability }))
}

/// Convert transport errors, retaining rejected-upgrade status, body, and request ID.
/// Failures without a provider response retain transport error classification.
fn websocket_provider_error(error: http_client::Error) -> ProviderError {
    let provider_request_id = error.non_success_headers().and_then(|headers| {
        crate::providers::internal::request_id_from_headers(headers, REQUEST_ID_HEADER)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(provider_request_id)
}

/// Construct Responses WebSocket sessions from a bound wire and a supplied backend.
pub trait ResponsesWebSocketExt {
    /// Start configuring a websocket session for this wire's model.
    fn responses_websocket_builder(&self) -> ResponsesWebSocketSessionBuilder;

    /// Open a websocket session over `backend`, with default options.
    fn responses_websocket_with<W>(
        &self,
        backend: &W,
    ) -> impl std::future::Future<Output = Result<ResponsesWebSocketSession, ProviderError>>
    + WasmCompatSend
    where
        W: WebSocketClientExt + WasmCompatSync,
        Self: WasmCompatSync;
}

impl<H> ResponsesWebSocketExt for Bound<Responses, H> {
    fn responses_websocket_builder(&self) -> ResponsesWebSocketSessionBuilder {
        ResponsesWebSocketSessionBuilder::new(self.wire.clone())
    }

    fn responses_websocket_with<W>(
        &self,
        backend: &W,
    ) -> impl std::future::Future<Output = Result<ResponsesWebSocketSession, ProviderError>>
    + WasmCompatSend
    where
        W: WebSocketClientExt + WasmCompatSync,
        Self: WasmCompatSync,
    {
        let builder = self.responses_websocket_builder();
        async move { builder.connect_with(backend).await }
    }
}

/// Native sessions and builders satisfy Send and Sync.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResponsesWebSocketSession>();
    assert_send_sync::<ResponsesWebSocketSessionBuilder>();
    assert_send_sync::<KeepaliveDrain>();
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod test_connection;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
