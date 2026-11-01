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
use crate::completion::options::FinalBody;
use crate::driver::Model;
use crate::error::{CorruptFrame, EncodeError, ProviderError};
use crate::http_client::{self, NoBody};
use crate::operation::{Completion, Turn};
use crate::providers::openai::responses_api::streaming::{
    IncompleteTerminal, ResponsesDecoder, ResponsesEvent, classify_websocket_payload, document,
    is_lifecycle_event,
};
use crate::providers::openai::responses_api::wire::Responses;
use crate::streaming::Item;
use crate::wire::{Call, Flow, Mode, Operation, Reply, Shared, Wire, WireFrame};
use crate::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, ReadyFrame, UnsupportedCapability,
    WebSocketClientExt, WebSocketConnection,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;

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

/// The canonical prepared request and its sealed websocket frame body.
#[derive(Clone, Debug)]
pub(crate) struct PreparedTurn {
    pub(crate) request: completion::CompletionRequest,
    pub(crate) body: FinalBody,
    pub(crate) context: super::DeltaContext,
}

/// The closed frame plan applied by the Responses request merge.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct WebSocketCreate {
    pub(crate) input: WebSocketCreateInput,
    pub(crate) stamps: Map<String, Value>,
    pub(crate) identity: Option<super::codex_identity::CodexIdentity>,
    pub(crate) lite: bool,
    pub(crate) automatic_previous: Option<String>,
    pub(crate) generate: Option<bool>,
}

/// The input a websocket frame sends before its body is sealed.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum WebSocketCreateInput {
    Full,
    Delta {
        envelope: FinalBody,
        input: WireDelta,
        previous_response_id: String,
    },
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
    /// A number no HTTP status can be refuses the event as malformed.
    #[serde(
        default,
        alias = "status_code",
        deserialize_with = "http_status",
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

/// An error event's reported status: absent or `null` is none, and a number
/// is kept only when it is a valid HTTP status. Anything else is a malformed
/// known field, refused with a typed decode error rather than read as none.
fn http_status<'de, D>(deserializer: D) -> Result<Option<u16>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<u16>::deserialize(deserializer)? {
        Some(status) if http::StatusCode::from_u16(status).is_err() => Err(
            serde::de::Error::custom(format!("`{status}` is not an HTTP status")),
        ),
        status => Ok(status),
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
    /// The complete terminal object. Sparse objects retain every field;
    /// modeled fields are checked whenever present.
    #[serde(deserialize_with = "websocket_done_response")]
    pub response: Value,
}

fn websocket_done_response<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let response = Value::deserialize(deserializer)?;
    super::streaming::websocket_done(&response).map_err(serde::de::Error::custom)?;
    Ok(response)
}

impl ResponsesWebSocketDoneEvent {
    /// Returns the response ID if the payload includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        self.response.get("id").and_then(Value::as_str)
    }

    fn status(&self) -> Option<&str> {
        self.response.get("status").and_then(Value::as_str)
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
    /// A response lifecycle event such as `response.created` or
    /// `response.completed`, with the response object it carries.
    Response {
        /// The event's `type`.
        kind: String,
        /// The response object, as the provider sent it.
        response: Value,
    },
    /// A streaming item or delta event such as `response.output_text.delta`,
    /// as the provider sent it.
    Item(Value),
    /// A protocol-level websocket error event.
    Error(ResponsesWebSocketErrorEvent),
    /// An optional `response.done` event emitted by OpenAI over WebSockets.
    Done(ResponsesWebSocketDoneEvent),
    /// A server event whose `type` this client does not model, surfaced with
    /// its tag and complete parsed payload rather than discarded, and retained
    /// for [`Item::Unknown`] passthrough.
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

/// One incremental input with explicit canonical or literal wire provenance.
#[derive(Debug, Clone, PartialEq)]
pub enum InputDeltaItem {
    /// A Rig message lowered through the Responses input encoder.
    Canonical(crate::message::Message),
    /// A literal provider input item sent with its value unchanged.
    Raw(Value),
}

/// The new source input for a forward-only incremental turn.
/// Lowering must also produce at least one wire item before the frame is sent.
#[derive(Debug, Clone, PartialEq)]
pub struct InputDelta(Vec<InputDeltaItem>);

/// Returned when an incremental delta carries no input items.
#[derive(Debug, thiserror::Error)]
#[error("an incremental delta must carry at least one input item")]
pub struct EmptyInputDeltaError;

impl InputDelta {
    /// A delta containing one canonical or literal input item.
    #[must_use]
    pub fn one(item: InputDeltaItem) -> Self {
        Self(vec![item])
    }

    /// A delta from source items, refusing the empty case.
    pub fn new(items: Vec<InputDeltaItem>) -> Result<Self, EmptyInputDeltaError> {
        if items.is_empty() {
            return Err(EmptyInputDeltaError);
        }
        Ok(Self(items))
    }

    /// Return the source items with their provenance unchanged.
    #[must_use]
    pub fn into_vec(self) -> Vec<InputDeltaItem> {
        self.0
    }
}

impl TryFrom<Vec<InputDeltaItem>> for InputDelta {
    type Error = EmptyInputDeltaError;
    fn try_from(items: Vec<InputDeltaItem>) -> Result<Self, Self::Error> {
        Self::new(items)
    }
}

/// Nonempty wire input admitted by the internal WebSocket frame merge.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct WireDelta(Vec<Value>);

impl WireDelta {
    pub(crate) fn new(items: Vec<Value>) -> Result<Self, EmptyInputDeltaError> {
        if items.is_empty() {
            return Err(EmptyInputDeltaError);
        }
        Ok(Self(items))
    }
    pub(crate) fn into_vec(self) -> Vec<Value> {
        self.0
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
    InFlight(Box<PreparedTurn>),
    /// The last turn completed as `tip`; a delta may continue it by reusing
    /// `envelope`'s non-input configuration.
    Eligible {
        envelope: Box<PreparedTurn>,
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
            Self::Response { response, .. } => response.get("id").and_then(Value::as_str),
            Self::Done(done) => done.response_id(),
            Self::Item(_) | Self::Error(_) | Self::Unknown(_) => None,
        }
    }

    /// Returns `true` when this event ends the current in-flight websocket turn.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Response { kind, .. } => is_terminal_kind(kind),
            Self::Error(_) | Self::Done(_) => true,
            // An unmodelled event ends nothing at the protocol level; whether it
            // is tolerable is the consumer's decision, not this predicate's.
            Self::Item(_) | Self::Unknown(_) => false,
        }
    }
}

/// Whether a lifecycle event of `kind` ends the turn.
fn is_terminal_kind(kind: &str) -> bool {
    matches!(
        kind,
        "response.completed" | "response.failed" | "response.incomplete"
    )
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
    /// Open a session over the bundled tungstenite backend with the configured
    /// timeouts. Return handshake construction, transport, or provider errors.
    #[cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "tungstenite")))]
    pub async fn connect(self) -> Result<ResponsesWebSocketSession, ProviderError> {
        self.connect_with(&rig_tungstenite::TungsteniteClient::new())
            .await
    }

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
///
/// Over a Codex-contract wire this session is not an emulation of the official
/// client: its frames carry the wire's Codex shaping, such as `store: false`,
/// but not the members only a [`codex::CodexWebSocketSession`] states, such as
/// `stream: true`. Use that session for the official client's frames.
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
    /// The first Codex `x-codex-turn-state` token a `response.metadata` event
    /// of an in-flight turn handed the session, until the caller takes it.
    turn_state: Option<String>,
    in_flight: bool,
    /// The fold of the turn in flight, opened when it was sent.
    turn: Option<Turn>,
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
        let mut request = websocket_request(&wire)?;
        authorize_handshake(&wire, &mut request).await?;
        let socket = connect(backend, request, connect_timeout).await?;
        Ok(Self::from_connection(wire, socket, event_timeout))
    }

    /// Build a session over an already-open, authenticated connection.
    /// `event_timeout: None` waits indefinitely for each event.
    ///
    /// On a dialect that requires the caller's exact identity, every send
    /// refuses with [`MissingCallerIdentity`](crate::providers::openai::wire::MissingCallerIdentity)
    /// until `wire` carries one, whoever opened the connection.
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
            turn: None,
            event_timeout,
            closed: false,
            failed: false,
            turn_state: None,
        }
    }

    /// Record the Codex sticky-routing token a `response.metadata` event
    /// carries in its `headers`, when the session holds none: the first
    /// token wins until it is taken, as the official client keeps the first
    /// one of a turn. A name is matched case-insensitively and a value is a
    /// string, or the first string of an array, as the official client
    /// reads them.
    fn observe_turn_state(&mut self, event: &ResponsesWebSocketEvent) {
        fn first_string(value: &serde_json::Value) -> Option<&str> {
            match value {
                serde_json::Value::String(value) => Some(value),
                serde_json::Value::Array(items) => items.first().and_then(first_string),
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
            .and_then(serde_json::Value::as_object)
        else {
            return;
        };
        // Like the official client, a name whose value is not a string does
        // not end the search: another spelling of the name may carry one.
        self.turn_state = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case(codex::TURN_STATE_METADATA_KEY))
            .find_map(|(_, value)| first_string(value))
            .map(str::to_owned);
    }

    /// The Codex turn-state token recorded since it was last taken.
    pub(crate) fn turn_state(&self) -> Option<&str> {
        self.turn_state.as_deref()
    }

    /// Take the recorded Codex turn-state token, so the next one is recorded.
    pub(crate) fn take_turn_state(&mut self) -> Option<String> {
        self.turn_state.take()
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
        let input_requirement = super::InputRequirement::for_create(&options);
        let envelope = self.prepare_request(
            completion_request,
            Chaining::Automatic,
            None,
            input_requirement,
        )?;
        let body = self.wire.websocket_request_with_identity(
            &envelope.request,
            None,
            input_requirement,
            WebSocketCreate {
                input: WebSocketCreateInput::Full,
                stamps: Map::new(),
                identity: None,
                lite: false,
                automatic_previous: self.previous_response_id.clone(),
                generate: options.generate,
            },
        )?;
        let frame = encode_frame(
            &body,
            self.wire.provider.dialect.quirks.responses.contract
                == super::wire::ResponsesContract::Codex
                && self.wire.codex_request_shape.is_standard(),
        )?;
        self.send_encoded(envelope, frame).await
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
            return Err(ProviderError::request(UncollectedRecoveredFrames {
                count: self.recovered.len(),
            }));
        }

        Ok(())
    }

    /// The envelope and tip an incremental continuation would chain onto, or
    /// the reason there is none.
    pub(crate) fn continuation(&self) -> Result<(&PreparedTurn, &str), IncrementalSendRefused> {
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
        envelope: PreparedTurn,
        frame: String,
    ) -> Result<(), ProviderError> {
        self.ensure_can_send()?;
        // A session built over a connection someone else opened has had no
        // handshake of Rig's to refuse, so the identity rule is checked again
        // here, before any frame is written.
        self.wire.provider.sendable_identity()?;
        let describe = self.wire.describe();
        let turn = Completion::fold(
            &envelope.request,
            &mut Call::new(&describe, Mode::Streaming),
        );

        if let Err(error) = self.socket.send(Frame::Text(frame)).await {
            return Err(self.fail_session(websocket_provider_error(error)));
        }
        self.continuation = Continuation::InFlight(Box::new(envelope));
        self.in_flight = true;
        self.turn = Some(turn);

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
                ResponsesWebSocketEvent::Response { .. }
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
        let response = self.wait_for_terminal_response().await?.0;
        response
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                ProviderError::Response("the websocket warmup has no response id".into())
            })
    }

    /// Sends a completion turn and collects the final OpenAI response,
    /// normalized; its `raw` is the `Response` the turn's events rebuild.
    pub async fn completion(
        &mut self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        self.send(completion_request).await?;
        self.collect_completion().await
    }

    /// Collect the in-flight turn through its decoder and reassembler.
    /// Incomplete turns retain their partial output and truthful finish reason.
    pub(crate) async fn collect_completion(
        &mut self,
    ) -> Result<completion::CompletionResponse, ProviderError> {
        Ok(self.wait_for_terminal_response().await?.1)
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

    /// Prepare the canonical request and seal its full websocket frame.
    /// Automatic chaining fills an absent effective response ID;
    /// a root send keeps only the response ID the request names.
    pub(crate) fn prepare_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
        chaining: Chaining,
        identity: Option<&super::codex_identity::CodexIdentity>,
        input_requirement: super::InputRequirement,
    ) -> Result<PreparedTurn, ProviderError> {
        let request = Completion::prepare(completion_request, &self.wire.describe())?;
        let (body, context) = self.wire.prepare_websocket_request_with_identity(
            &request,
            identity,
            input_requirement,
            WebSocketCreate {
                input: WebSocketCreateInput::Full,
                stamps: Map::new(),
                identity: identity.cloned(),
                lite: identity.is_some() && self.wire.codex_request_shape.is_lite(),
                automatic_previous: (chaining == Chaining::Automatic)
                    .then(|| self.previous_response_id.clone())
                    .flatten(),
                generate: None,
            },
        )?;
        Ok(PreparedTurn {
            request,
            body,
            context,
        })
    }

    pub(crate) async fn wait_for_completed_response(&mut self) -> Result<Value, ProviderError> {
        Ok(self.wait_for_terminal_response().await?.0)
    }

    /// Collect the completed or incomplete turn through the current decoder
    /// and reassembler. Transport, protocol and decoder failures are errors.
    async fn wait_for_terminal_response(
        &mut self,
    ) -> Result<(Value, completion::CompletionResponse), ProviderError> {
        let provider = self.wire.describe().name.to_owned();
        let Some(turn) = self.turn.take() else {
            return Err(ProviderError::Provider(
                "No OpenAI websocket response is currently in flight on this session".to_string(),
            ));
        };
        let reply = std::sync::Mutex::new(Shared::new(turn));
        let mut decoder = self
            .wire
            .decoder_with_incomplete(IncompleteTerminal::Accept)
            .with_websocket_reader();
        let mut document = document::Response::with_websocket_reader();
        loop {
            let (event, payload) = self.next_event_with_payload().await?;
            match event {
                ResponsesWebSocketEvent::Response { kind, response } => {
                    if !is_terminal_kind(&kind) {
                        feed(&mut decoder, &mut document, &reply, payload)?;
                        continue;
                    }
                    // A failed turn is reported from its own envelope; only a
                    // completed or incomplete one reaches the decoder, whose
                    // end closes the turn.
                    let response = terminal_response_result(response)?;
                    let ended = feed(&mut decoder, &mut document, &reply, payload)?;
                    let folded = fold_reply(
                        reply,
                        ended,
                        &provider,
                        crate::wire::document::Reassemble::finish(document),
                    )?;
                    return Ok((response, folded));
                }
                ResponsesWebSocketEvent::Done(done) => {
                    if ["id", "object", "created_at", "status", "model"]
                        .iter()
                        .all(|key| done.response.get(*key).is_some())
                    {
                        // A failed turn is reported from its own envelope, as
                        // on the `response.failed` path.
                        let response = terminal_response_result(done.response)?;
                        let ended =
                            feed(&mut decoder, &mut document, &reply, response.to_string())?;
                        let folded = fold_reply(
                            reply,
                            ended,
                            &provider,
                            crate::wire::document::Reassemble::finish(document),
                        )?;
                        return Ok((response, folded));
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
                // Unknown frames keep their raw payload.
                ResponsesWebSocketEvent::Item(_) | ResponsesWebSocketEvent::Unknown(_) => {
                    feed(&mut decoder, &mut document, &reply, payload)?;
                }
            }
        }
    }

    fn update_state_for_event(&mut self, event: &ResponsesWebSocketEvent) {
        let ending = match event {
            ResponsesWebSocketEvent::Response { kind, response } => {
                let response_id = response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                match kind.as_str() {
                    "response.completed" => {
                        self.previous_response_id.clone_from(&response_id);
                        self.pending_done_response_id.clone_from(&response_id);
                        TurnEnding::Completed { response_id }
                    }
                    // Automatic sends may continue an incomplete response,
                    // while incremental sends require a completed tip.
                    "response.incomplete" => {
                        self.previous_response_id.clone_from(&response_id);
                        self.pending_done_response_id.clone_from(&response_id);
                        TurnEnding::Incomplete { response_id }
                    }
                    "response.failed" => {
                        self.pending_done_response_id = response_id;
                        self.previous_response_id = None;
                        TurnEnding::NotCompleted
                    }
                    _ => return,
                }
            }
            ResponsesWebSocketEvent::Done(done) => {
                let response_id = done.response_id().map(str::to_owned);
                let ending = match done.status() {
                    Some("completed") => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Completed { response_id }
                    }
                    Some("incomplete") => {
                        if let Some(response_id) = &response_id {
                            self.previous_response_id = Some(response_id.clone());
                        }
                        TurnEnding::Incomplete { response_id }
                    }
                    Some("in_progress" | "queued") => TurnEnding::NotCompleted,
                    _ => {
                        self.previous_response_id = None;
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
        self.turn = None;
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
    Automatic,
    Root,
}

/// Serialize the frame sealed by the Responses request merge.
fn encode_frame(request: &FinalBody, routing_first: bool) -> Result<String, ProviderError> {
    crate::providers::internal::trace_json(
        crate::providers::internal::LogTarget::Completions,
        "OpenAI websocket request",
        request,
    );
    Ok(if routing_first {
        serde_json::to_string(&super::codex_request::RoutingFirst(request))?
    } else {
        serde_json::to_string(request)?
    })
}

/// Feed one message to the turn's reassembler and decoder.
fn feed(
    decoder: &mut ResponsesDecoder,
    document: &mut document::Response,
    reply: &std::sync::Mutex<Shared<Completion>>,
    payload: String,
) -> Result<bool, ProviderError> {
    let flow = crate::driver::step(
        decoder,
        Some(document),
        reply,
        WireFrame::Text(payload),
        None,
    )?;
    let mut shared = reply
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while let Some(item) = shared.take() {
        let _ = item?;
    }
    Ok(matches!(flow, Flow::Ended(_)))
}

/// Fold the turn's events and end into the normalized response, whose raw
/// is the `Response` the turn's events rebuild, as a stream's is.
fn fold_reply(
    reply: std::sync::Mutex<Shared<Completion>>,
    ended: bool,
    provider: &str,
    raw: Value,
) -> Result<completion::CompletionResponse, ProviderError> {
    let fed = if ended {
        Ok(())
    } else {
        Err(ProviderError::Truncated)
    };
    let reply_of = Reply {
        provider: provider.to_owned(),
        raw,
        // The websocket carries no reply headers past the handshake.
        provider_request_id: None,
        response_headers: crate::completion::ProviderResponseHeaders::new(),
    };
    crate::driver::settle(reply, fed, reply_of).outcome
}

fn terminal_response_result(response: Value) -> Result<Value, ProviderError> {
    match response.get("status").and_then(Value::as_str) {
        // An incomplete response (e.g. hitting `max_output_tokens`) is a
        // genuine terminal: the partial output and usage are kept, and the
        // decoder maps its status and reason to the finish.
        Some("completed" | "incomplete") | None => Ok(response),
        // Preserve provider error envelopes as reserialized JSON without an HTTP status.
        // Without an error object, return a local diagnostic instead.
        Some("failed") => match response.get("error").filter(|error| !error.is_null()) {
            Some(_) => Err(ProviderError::from_provider_body(response.to_string())),
            None => Err(ProviderError::Provider(response_error_message(
                "failed response",
            ))),
        },
        Some(other) => Err(ProviderError::Provider(format!(
            "OpenAI websocket response ended in state `{other}`"
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

/// The unmodelled Responses event carrying server-supplied headers, among
/// them the Codex turn-state token.
const RESPONSE_METADATA_EVENT: &str = "response.metadata";

/// Decode WebSocket error and done events or delegate to Responses classification.
/// Return parsing and triage errors; preserve unknown payloads with their tag.
/// A frame that fails to decode is [`ProviderError::CorruptFrame`] with the
/// payload as received, the same error a Responses SSE stream returns.
fn parse_server_event(payload: &str) -> Result<Option<ResponsesWebSocketEvent>, ProviderError> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let corrupt = |event_type: Option<&str>, error| {
        ProviderError::CorruptFrame(
            CorruptFrame::text(event_type.map(ToOwned::to_owned), payload, error).into(),
        )
    };
    let event_type =
        serde_json::from_str::<EventType>(payload).map_err(|error| corrupt(None, error))?;
    match event_type.kind.as_str() {
        kind @ "error" => serde_json::from_str(payload)
            .map(|e| Some(ResponsesWebSocketEvent::Error(e)))
            .map_err(|error| corrupt(Some(kind), error)),
        kind @ "response.done" => serde_json::from_str(payload)
            .map(|d| Some(ResponsesWebSocketEvent::Done(d)))
            .map_err(|error| corrupt(Some(kind), error)),
        _ => Ok(
            match crate::driver::triage(classify_websocket_payload(payload))? {
                Item::Event(ResponsesEvent::Frame {
                    kind, mut frame, ..
                }) => {
                    if is_lifecycle_event(&kind) {
                        let response = frame
                            .as_object_mut()
                            .and_then(|frame| frame.shift_remove("response"))
                            .unwrap_or_else(|| Value::Object(Map::new()));
                        Some(ResponsesWebSocketEvent::Response { kind, response })
                    } else {
                        Some(ResponsesWebSocketEvent::Item(frame))
                    }
                }
                Item::Event(ResponsesEvent::Whole(value)) => {
                    Some(ResponsesWebSocketEvent::Unknown(UnrecognizedEvent {
                        kind: event_type.kind,
                        payload: value.into(),
                    }))
                }
                Item::Event(ResponsesEvent::Failure(body)) => {
                    return Err(ProviderError::from_provider_body(body));
                }
                Item::Event(ResponsesEvent::Sentinel) => None,
                Item::Unknown(value) => Some(ResponsesWebSocketEvent::Unknown(UnrecognizedEvent {
                    kind: event_type.kind,
                    payload: value,
                })),
            },
        ),
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

    let request = wire.request_headers.stamp(
        wire.provider.headers(
            http_client::Request::builder()
                .method(http::Method::GET)
                .uri(url),
        )?,
    );

    request.body(NoBody).map_err(|error| {
        EncodeError::request(format!("Failed to build OpenAI websocket request: {error}"))
    })
}

/// Stamp the wire's send-time credential onto a handshake about to be sent,
/// when the wire reads one from a credential source. The one authorization
/// step of every managed websocket connect, plain and Codex alike: the source
/// is read once, before anything reaches the backend, a failing source
/// refuses the connect, and the open connection keeps the credential it was
/// opened with, so a rotation takes effect at the next connect.
pub(super) async fn authorize_handshake(
    wire: &Responses,
    request: &mut http_client::Request<NoBody>,
) -> Result<(), ProviderError> {
    if let Some(stamp) = wire.provider.credential_stamp() {
        stamp.authorize(request.headers_mut()).await?;
    }
    Ok(())
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
    ProviderError::request(UnsupportedCapability { capability })
}

/// Convert transport errors, retaining rejected-upgrade status, body, and request ID.
/// Failures without a provider response retain transport error classification.
fn websocket_provider_error(error: http_client::Error) -> ProviderError {
    let provider_request_id = error.non_success_headers().and_then(|headers| {
        crate::driver::http_transport::request_id_from(headers, REQUEST_ID_HEADER)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(provider_request_id)
}

impl<T> Model<Responses, T> {
    /// Start configuring a Responses WebSocket session for this model. Open
    /// it with [`connect_with`](ResponsesWebSocketSessionBuilder::connect_with)
    /// and a backend, or, under the `tungstenite` feature, with
    /// [`connect`](ResponsesWebSocketSessionBuilder::connect).
    pub fn responses_websocket(&self) -> ResponsesWebSocketSessionBuilder {
        ResponsesWebSocketSessionBuilder::new(self.wire.clone())
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
