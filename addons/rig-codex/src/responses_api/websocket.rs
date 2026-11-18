//! The events and values a Codex Responses websocket session speaks.
//!
//! The session itself is [`codex::CodexWebSocketSession`]. It permits one
//! in-flight turn, and between turns
//! [`keepalive`](codex::CodexWebSocketSession::keepalive) services the idle
//! connection and hands back every unmodelled frame it had to consume.
//!
//! ```
//! use rig_codex::responses_api::websocket::ResponsesWebSocketCreateOptions;
//! let options = ResponsesWebSocketCreateOptions::warmup();
//! assert_eq!(options.generate, Some(false));
//! ```

use crate::error::{CodexSessionError, CorruptFrame};
use crate::responses_api::body::RequestBody;
use crate::responses_api::streaming::{
    DecodedFrame, ItemChunk, ResponseChunk, ResponseChunkKind, StreamingCompletionChunk,
    decode_frame,
};
use crate::responses_api::{
    CompletionResponse, IncompleteDetailsReason, InputItem, Output, ResponseError, ResponseObject,
    ResponseStatus, ResponsesToolDefinition, ResponsesUsage,
};
use crate::ws_client::Frame;
use rig_core::error::ProviderError;
use rig_core::http_client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::time::Duration;

pub use rig_core::providers::openai::responses_api::websocket::{
    ResponsesWebSocketCreateOptions, ResponsesWebSocketDoneEventKind,
    ResponsesWebSocketErrorEventKind,
};

pub mod codex;

/// The websocket endpoint's path, appended to the client's configured base URL.
pub(crate) const WEBSOCKET_PATH: &str = "responses";

pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The most frames one idle [`keepalive`](codex::CodexWebSocketSession::keepalive)
/// consumes before it stops and hands back what it has taken.
///
/// A peer that delivers faster than the drain consumes is the only way the
/// drain could fail to terminate:
/// [`recv_ready`](crate::ws_client::WebSocketConnection::recv_ready) never
/// waits on the peer. A frame budget bounds it without a timer.
pub(crate) const MAX_KEEPALIVE_DRAIN_FRAMES: usize = 4_096;

const _: () = assert!(
    MAX_KEEPALIVE_DRAIN_FRAMES > 0,
    "a zero budget would consume nothing and report a flood on every idle drain"
);

/// How long the one post-drain flush may take before the connection is treated
/// as unserviceable.
///
/// Internal on purpose: expiry returns the recovered frames beside a typed
/// failure, so a caller never needs an outer timeout of its own to bound
/// [`keepalive`](codex::CodexWebSocketSession::keepalive).
pub(crate) const KEEPALIVE_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);
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
        deserialize_with = "rig_core::json_utils::null_or_default",
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
    pub response: ResponsesWebSocketDoneResponse,
}

/// A validated WebSocket terminal object, including supported sparse replies.
/// Every modeled structural field is checked when present. Unknown fields and
/// optional echoed metadata retain the full response decoder's leniency.
#[derive(Debug, Clone)]
pub struct ResponsesWebSocketDoneResponse {
    raw: Map<String, Value>,
    id: String,
    status: ResponseStatus,
    full: Option<Box<CompletionResponse>>,
}

impl ResponsesWebSocketDoneResponse {
    /// The complete provider object, including fields the client does not model.
    #[must_use]
    pub fn as_object(&self) -> &Map<String, Value> {
        &self.raw
    }

    /// The provider's response identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The provider's status, including future string values.
    #[must_use]
    pub fn status(&self) -> &ResponseStatus {
        &self.status
    }

    /// The typed response, when the object carries every field a response
    /// requires. `None` for a sparse terminal object.
    #[must_use]
    pub fn completion_response(&self) -> Option<&CompletionResponse> {
        self.full.as_deref()
    }
}

impl Serialize for ResponsesWebSocketDoneResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ResponsesWebSocketDoneResponse {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Map::<String, Value>::deserialize(deserializer)?;
        let id = serde_json::from_value(
            raw.get("id")
                .cloned()
                .ok_or_else(|| <D::Error as serde::de::Error>::missing_field("id"))?,
        )
        .map_err(serde::de::Error::custom)?;
        let status = serde_json::from_value(
            raw.get("status")
                .cloned()
                .ok_or_else(|| <D::Error as serde::de::Error>::missing_field("status"))?,
        )
        .map_err(serde::de::Error::custom)?;

        // These are the response object's structural fields and exact types;
        // metadata and its deliberately permissive reasoning echo stay raw.
        validate_done_field::<ResponseObject>(&raw, "object")
            .and_then(|()| validate_done_field::<u64>(&raw, "created_at"))
            .and_then(|()| validate_done_field::<String>(&raw, "model"))
            .and_then(|()| validate_done_field::<Option<ResponseError>>(&raw, "error"))
            .and_then(|()| {
                validate_done_field::<Option<IncompleteDetailsReason>>(&raw, "incomplete_details")
            })
            .and_then(|()| validate_done_field::<Option<String>>(&raw, "instructions"))
            .and_then(|()| validate_done_field::<Option<u64>>(&raw, "max_output_tokens"))
            .and_then(|()| validate_done_field::<Option<ResponsesUsage>>(&raw, "usage"))
            .and_then(|()| validate_done_field::<Vec<Output>>(&raw, "output"))
            .and_then(|()| validate_done_field::<Vec<ResponsesToolDefinition>>(&raw, "tools"))
            .map_err(serde::de::Error::custom)?;

        let full = if ["id", "object", "created_at", "status", "model"]
            .iter()
            .all(|key| raw.contains_key(*key))
        {
            Some(Box::new(
                serde_json::from_value(Value::Object(raw.clone()))
                    .map_err(serde::de::Error::custom)?,
            ))
        } else {
            None
        };
        Ok(Self {
            raw,
            id,
            status,
            full,
        })
    }
}

fn validate_done_field<T: serde::de::DeserializeOwned>(
    raw: &Map<String, Value>,
    field: &str,
) -> Result<(), serde_json::Error> {
    if let Some(value) = raw.get(field) {
        serde_json::from_value::<T>(value.clone()).map_err(|error| {
            <serde_json::Error as serde::de::Error>::custom(format!(
                "response.done.response.{field}: {error}"
            ))
        })?;
    }
    Ok(())
}

impl ResponsesWebSocketDoneEvent {
    /// Returns the response ID if the payload includes one.
    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        Some(self.response.id())
    }

    pub(crate) fn status(&self) -> &ResponseStatus {
        self.response.status()
    }
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
    /// its tag and complete parsed payload rather than discarded.
    Unknown(UnrecognizedEvent),
}

/// A server event this client does not model.
///
/// A named value rather than inline variant fields because
/// [`codex::CodexWebSocketSession::keepalive`] returns it on its own: a
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
    pub payload: rig_core::streaming::UnknownPayload,
}

/// The new input for a forward-only incremental turn : never empty.
///
/// An incremental turn chains onto a live tip and replaces the captured
/// envelope's `input` wholesale, so an empty delta would send a chained
/// `response.create` carrying nothing new. The field is private, so the
/// constructors below are the only way in, and there is deliberately no
/// `Deserialize` that could bypass them.
#[derive(Debug, Clone)]
pub struct InputDelta(Vec<InputItem>);

/// Returned when an incremental delta would carry no input items.
#[derive(Debug, thiserror::Error)]
#[error("an incremental delta must carry at least one input item")]
pub struct EmptyInputDeltaError;

impl InputDelta {
    /// A delta of exactly one item. Infallible by construction.
    #[must_use]
    pub fn one(item: InputItem) -> Self {
        Self(vec![item])
    }

    /// A delta from many items, rejecting the empty case.
    pub fn new(items: Vec<InputItem>) -> Result<Self, EmptyInputDeltaError> {
        if items.is_empty() {
            return Err(EmptyInputDeltaError);
        }
        Ok(Self(items))
    }

    /// The items, surrendered to the caller.
    #[must_use]
    pub fn into_vec(self) -> Vec<InputItem> {
        self.0
    }
}

impl TryFrom<Vec<InputItem>> for InputDelta {
    type Error = EmptyInputDeltaError;

    fn try_from(items: Vec<InputItem>) -> Result<Self, Self::Error> {
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
    /// [`clear_previous_response_id`](codex::CodexWebSocketSession::clear_previous_response_id).
    #[error("cannot send an incremental turn after the live tip was cleared")]
    TipCleared,
}

/// A send refused because an interrupted keepalive left recovered frames the
/// caller has not collected yet.
///
/// Those frames were taken off the socket before the new turn; sending first
/// would let the next turn's reads run ahead of frames nobody has accounted
/// for. Call [`keepalive`](codex::CodexWebSocketSession::keepalive) to collect
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "{count} unmodelled frame(s) recovered by an interrupted keepalive have not been collected; call keepalive() before sending"
)]
pub struct UncollectedRecoveredFrames {
    /// How many recovered frames the session is holding.
    pub count: usize,
}

/// A send was refused because an earlier send on the session did not finish.
///
/// The earlier call's future was dropped while the connection was still
/// writing its frame, so the backend may have received the frame and may
/// answer it. The session keeps that turn in flight rather than present an
/// idle socket: a second request written behind it would have its events
/// read as the first's. Read the turn with
/// [`next_event`](codex::CodexWebSocketSession::next_event), or close the
/// session and resend on a new one, which may deliver the request twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "an earlier send on this session was dropped before the connection confirmed it; the \
     backend may have received it. Read the turn with next_event(), or close this session \
     and resend on a new one"
)]
pub struct UnconfirmedSend;

/// What the next incremental continuation may chain onto.
#[derive(Debug, Clone)]
pub(crate) enum Continuation {
    /// Nothing may be continued, for the stated reason.
    Blocked(IncrementalSendRefused),
    /// A turn built from this envelope is in flight. It becomes the eligible
    /// envelope only if that turn completes.
    InFlight(Box<RequestBody>),
    /// The last turn completed as `tip`; a delta may continue it by reusing
    /// `envelope`'s non-input configuration.
    Eligible {
        envelope: Box<RequestBody>,
        tip: String,
    },
}

impl Continuation {
    /// The next state once the in-flight turn ends in `ending`.
    pub(crate) fn settle(self, ending: TurnEnding) -> Self {
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
pub(crate) enum TurnEnding {
    /// The turn completed.
    Completed { response_id: Option<String> },
    /// The turn ended incomplete.
    Incomplete { response_id: Option<String> },
    /// The turn failed, errored or never reached a completed verdict.
    NotCompleted,
}

/// How far an idle [`keepalive`](codex::CodexWebSocketSession::keepalive) drain
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
    /// [`UnsupportedCapability`](crate::ws_client::UnsupportedCapability) of
    /// the backend, the session has been marked
    /// failed or closed.
    Failed(CodexSessionError),
}

impl KeepaliveDrain {
    pub(crate) fn serviced(recovered: Vec<UnrecognizedEvent>) -> Self {
        Self {
            recovered,
            ending: KeepaliveEnding::Serviced,
        }
    }

    pub(crate) fn failed(recovered: Vec<UnrecognizedEvent>, error: CodexSessionError) -> Self {
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
    pub fn into_parts(self) -> (Vec<UnrecognizedEvent>, Option<CodexSessionError>) {
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

/// The unmodelled Responses event carrying server-supplied headers, among
/// them the Codex turn-state token.
pub(crate) const RESPONSE_METADATA_EVENT: &str = "response.metadata";

/// Decode one server event. Websocket `error` and `response.done` events are
/// read here; every other payload is read by
/// [`decode_frame`](crate::responses_api::streaming::decode_frame). An event
/// of a type this crate does not model is surfaced whole with its tag. A
/// frame that fails to decode is a [`CorruptFrame`] with the payload as
/// received.
pub(crate) fn parse_server_event(payload: &str) -> Result<ResponsesWebSocketEvent, CorruptFrame> {
    #[derive(Deserialize)]
    struct EventType {
        #[serde(rename = "type")]
        kind: String,
    }

    let corrupt = |event_type: Option<&str>, error: serde_json::Error| {
        CorruptFrame::text(event_type.map(ToOwned::to_owned), payload, error)
    };
    let EventType { kind } =
        serde_json::from_str::<EventType>(payload).map_err(|error| corrupt(None, error))?;
    match kind.as_str() {
        "error" => serde_json::from_str(payload)
            .map(ResponsesWebSocketEvent::Error)
            .map_err(|error| corrupt(Some(&kind), error)),
        "response.done" => serde_json::from_str(payload)
            .map(ResponsesWebSocketEvent::Done)
            .map_err(|error| corrupt(Some(&kind), error)),
        _ => Ok(match decode_frame(payload)? {
            DecodedFrame::Chunk(StreamingCompletionChunk::Response(response)) => {
                ResponsesWebSocketEvent::Response(response)
            }
            DecodedFrame::Chunk(StreamingCompletionChunk::Delta(item)) => {
                ResponsesWebSocketEvent::Item(item)
            }
            // The tag comes from the probe above: reaching this arm proves
            // it decoded exactly one top-level string `type`, the same key
            // the classifier read.
            DecodedFrame::Unknown { payload, .. } => {
                ResponsesWebSocketEvent::Unknown(UnrecognizedEvent { kind, payload })
            }
        }),
    }
}

/// The request-ID header read from a rejected websocket upgrade.
const REQUEST_ID_HEADER: &str = "x-request-id";

/// Lower one websocket frame onto the JSON payload the protocol carries.
///
/// `Ok(None)` is a frame with no protocol payload (a keepalive), which the
/// session skips; a close frame mid-turn is an error naming the peer's reason.
/// A binary frame that is not UTF-8 cannot be JSON: it is a [`CorruptFrame`]
/// carrying its bytes exactly as received.
pub(crate) fn websocket_frame_to_text(frame: Frame) -> Result<Option<String>, CodexSessionError> {
    match frame {
        Frame::Text(text) => Ok(Some(text)),
        Frame::Binary(bytes) => match String::from_utf8(bytes.to_vec()) {
            Ok(text) => Ok(Some(text)),
            Err(not_utf8) => {
                let message = not_utf8.to_string();
                let bytes = not_utf8.into_bytes();
                match serde_json::from_slice::<Value>(&bytes) {
                    Err(error) => Err(CorruptFrame::bytes(bytes, error).into()),
                    // JSON is UTF-8, so bytes that are not never parse; the
                    // failure is still reported if a parser ever disagrees.
                    Ok(_) => Err(ProviderError::Response(message).into()),
                }
            }
        },
        Frame::Ping(_) | Frame::Pong(_) => Ok(None),
        Frame::Close(frame) => {
            let reason = frame
                .map(|frame| frame.reason)
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "without a close reason".to_string());
            Err(
                ProviderError::Provider(format!("The OpenAI websocket connection closed {reason}"))
                    .into(),
            )
        }
    }
}

/// The response a terminal event carries, or the error its status states.
pub(crate) fn terminal_response_result(
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
            None => Err(ProviderError::Provider(
                "OpenAI websocket returned a failed response".to_owned(),
            )),
        },
        // An incomplete response (such as one that hit `max_output_tokens`)
        // is a genuine terminal: its partial output and usage are kept.
        ResponseStatus::Incomplete => Ok(response),
        other => Err(ProviderError::Provider(format!(
            "OpenAI websocket response ended in state {other:?}"
        ))),
    }
}

/// Preserve an error event as reserialized provider JSON. When the event
/// reports an HTTP status, the error carries it, with the event's headers, so
/// it classifies by status as the same failure over HTTP would; otherwise it
/// has none. Fall back to its display text if serialization fails.
pub(crate) fn provider_error_from_event(error: &ResponsesWebSocketErrorEvent) -> ProviderError {
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

/// Convert transport errors, retaining rejected-upgrade status, body, and request ID.
/// Failures without a provider response retain transport error classification.
pub(crate) fn websocket_provider_error(error: http_client::Error) -> ProviderError {
    let provider_request_id = error
        .non_success_headers()
        .and_then(|headers| headers.get(REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    ProviderError::from_transport_error(error).with_provider_request_id(provider_request_id)
}

#[cfg(test)]
mod test_connection;

#[cfg(test)]
mod tests;
