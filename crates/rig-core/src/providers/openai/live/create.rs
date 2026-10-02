//! Session creation: the WebRTC offer and session configuration go to
//! `POST /live/sessions`, and the session id and answer come back.

use serde::Deserialize;
use serde_json::{Map, Value};

use super::{LIVE_SESSIONS_PATH, SessionConfig};
use crate::ProviderResponseError;
use crate::error::{EncodeError, ProviderError};
use crate::providers::openai::OpenAI;
use crate::providers::openai::wire::OPENAI;
use crate::wire::CredentialStamp;

/// The provider's dialect is not official OpenAI, the only dialect that
/// speaks the public Live API. This restricts the dialect, not the origin: an
/// [`OPENAI`] provider with a custom base URL is accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the public Live API needs the official OpenAI dialect; got dialect `{dialect}`")]
pub struct NotOfficialOpenAi {
    /// The provider's dialect.
    pub dialect: &'static str,
}

impl From<NotOfficialOpenAi> for ProviderError {
    fn from(error: NotOfficialOpenAi) -> Self {
        ProviderError::Request(Box::new(error))
    }
}

/// An empty SDP offer, which the provider refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the SDP offer is empty")]
pub struct EmptyOffer;

/// A created WebRTC session: its id and the SDP answer for the caller's peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CreatedSession {
    /// The opaque session id, exactly as sent, prefix included.
    pub session_id: String,
    /// The SDP answer, exactly as sent.
    pub answer_sdp: String,
}

/// The `error` object of a provider error body, and of an
/// [`ErrorEvent`](super::ErrorEvent).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ApiErrorDetail {
    /// A machine-readable code, such as `project_spend_limit_exceeded`.
    #[serde(default)]
    pub code: Option<String>,
    /// A human-readable message.
    #[serde(default)]
    pub message: Option<String>,
    /// The error category, such as `invalid_request_error`.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// The parameter at fault.
    #[serde(default)]
    pub param: Option<String>,
    /// The `event_id` of the client event that caused the error, when the
    /// server names one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The status, headers and request id of a reply the Live API sent.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplyHead {
    /// The HTTP status.
    pub status: http::StatusCode,
    /// The response headers, verbatim.
    pub headers: http::HeaderMap,
    /// The provider's request id, from the dialect's request id header.
    pub request_id: Option<String>,
}

/// The body of a reply the Live API sent, as far as it is known. Only
/// [`Self::Received`] holds the exact bytes; the other states hold the
/// transport's text or no body at all.
#[derive(Debug)]
pub enum ReplyBody {
    /// The body bytes exactly as received, when the transport delivered the
    /// body as bytes.
    Received(Vec<u8>),
    /// The body as text from a transport that reported the reply as an
    /// error. The transport made the text, so it may not be byte-exact.
    TransportText(String),
    /// The body could not be read after the status and headers arrived.
    Unreadable(ProviderError),
}

impl ReplyBody {
    /// The body's bytes as they are known: exact for [`Self::Received`], the
    /// transport's text for [`Self::TransportText`], none when unreadable.
    #[must_use]
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Received(bytes) => Some(bytes),
            Self::TransportText(text) => Some(text.as_bytes()),
            Self::Unreadable(_) => None,
        }
    }

    /// A text presentation of the body for display. Invalid UTF-8 in
    /// [`Self::Received`] is replaced with U+FFFD here only; the bytes are
    /// kept unchanged.
    #[must_use]
    pub fn text(&self) -> Option<std::borrow::Cow<'_, str>> {
        match self {
            Self::Received(bytes) => Some(String::from_utf8_lossy(bytes)),
            Self::TransportText(text) => Some(std::borrow::Cow::Borrowed(text)),
            Self::Unreadable(_) => None,
        }
    }
}

/// A reply the Live API sent: its head and its body.
#[derive(Debug)]
pub struct LiveReply {
    /// Status, headers and request id.
    pub head: ReplyHead,
    /// The body, as far as it was received.
    pub body: ReplyBody,
}

impl LiveReply {
    /// The reply as a shared [`ProviderResponseError`]. Its body is text, so
    /// invalid UTF-8 is replaced with U+FFFD and an unreadable body is empty.
    #[must_use]
    pub fn to_provider_response(&self) -> ProviderResponseError {
        ProviderResponseError::new(
            self.head.status,
            self.body
                .text()
                .map(|text| text.into_owned())
                .unwrap_or_default(),
        )
        .with_headers(Some(self.head.headers.clone()))
        .with_provider_request_id(self.head.request_id.clone())
    }
}

impl std::fmt::Display for LiveReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "status {}", self.head.status)?;
        if let Some(request_id) = &self.head.request_id {
            write!(f, ", request id {request_id}")?;
        }
        match &self.body {
            ReplyBody::Unreadable(error) => write!(f, "; body unreadable: {error}"),
            body => write!(f, ": {}", body.text().unwrap_or_default()),
        }
    }
}

/// A non-success reply: the decoded `error` object, when the body was read
/// and has one, and the whole reply.
#[derive(Debug)]
pub struct LiveErrorReply {
    /// The body's `error` object. `None` when the body is unreadable or is
    /// not `{"error": {...}}`.
    pub error: Option<ApiErrorDetail>,
    /// The reply.
    pub reply: LiveReply,
}

impl std::fmt::Display for LiveErrorReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.reply.fmt(f)
    }
}

/// Why a success reply is not a created WebRTC session.
#[derive(Debug, thiserror::Error)]
pub enum MalformedReason {
    /// The body does not decode as a created session.
    #[error("the body does not decode as a created session: {0}")]
    Undecodable(#[source] serde_json::Error),
    /// The body decodes but carries no WebRTC SDP answer.
    #[error("the created session has no WebRTC SDP answer")]
    NoWebRtcAnswer,
}

/// A success reply that is not a created WebRTC session.
#[derive(Debug)]
pub struct MalformedSession {
    /// Why it is not one.
    pub reason: MalformedReason,
    /// The reply.
    pub reply: LiveReply,
}

impl std::fmt::Display for MalformedSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}; {}", self.reason, self.reply)
    }
}

/// The `error.code` values of a reached hard spend limit.
const SPEND_LIMIT_CODES: [&str; 2] = [
    "project_spend_limit_exceeded",
    "organization_spend_limit_exceeded",
];

/// A failed session creation, classified so that a caller can stop rather
/// than retry when retrying cannot succeed. Callers that need the
/// spend-limit distinction or the reply evidence must branch on this type,
/// not on the [`ProviderError`] it converts into.
///
/// A failure after a reply arrived keeps the reply's status, headers and
/// request id. Its body is kept as exact bytes only when the body was read
/// as bytes; otherwise it is the transport's text, not guaranteed
/// byte-exact, or an explicit unreadable state (see [`ReplyBody`]).
#[derive(Debug, thiserror::Error)]
pub enum LiveApiError {
    /// 401 or 403: the credential was rejected, whether or not the body was
    /// read. Terminal.
    #[error("the Live API rejected the credential: {0}")]
    Authentication(LiveErrorReply),
    /// 429 whose read body has `error.code` `project_spend_limit_exceeded`
    /// or `organization_spend_limit_exceeded`: a hard spend limit is
    /// reached. Terminal until the limit is raised or resets. A 429 whose
    /// body was not read is [`Self::Rejected`].
    #[error("a Live API hard spend limit is reached: {0}")]
    SpendLimit(LiveErrorReply),
    /// Any other non-success reply.
    #[error("the Live API refused the session: {0}")]
    Rejected(LiveErrorReply),
    /// A success reply that is not a created WebRTC session.
    #[error("the Live API sent no usable session: {0}")]
    Malformed(MalformedSession),
    /// A success status arrived but its body could not be read. The session
    /// may exist, and its creation may be billed, but its id and SDP answer
    /// are unknown.
    #[error("the Live session outcome is unknown: status {}, body unreadable: {read_error}", head.status)]
    OutcomeUnknown {
        /// The received status, headers and request id.
        head: ReplyHead,
        /// Why the body could not be read.
        read_error: ProviderError,
    },
    /// The request could not be built or its credential could not be read.
    /// Nothing was sent.
    #[error("the Live session request was not sent: {0}")]
    Request(ProviderError),
    /// The send failed before any reply arrived.
    #[error("the Live session request got no reply: {0}")]
    Transport(ProviderError),
}

impl LiveApiError {
    /// Whether retrying the same request cannot succeed: rejected
    /// credentials or a reached spend limit.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Authentication(_) | Self::SpendLimit(_))
    }

    /// Whether retrying may succeed: a [`Self::Rejected`] reply whose status
    /// [`crate::error::retryable_status`] accepts, or a [`Self::Request`] or
    /// [`Self::Transport`] failure that [`ProviderError::is_retryable`]
    /// accepts. Never true for a terminal error or an unknown outcome.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Rejected(reply) => {
                crate::error::retryable_status(Some(reply.reply.head.status.as_u16()))
            }
            Self::Request(error) | Self::Transport(error) => error.is_retryable(),
            Self::Authentication(_)
            | Self::SpendLimit(_)
            | Self::Malformed(_)
            | Self::OutcomeUnknown { .. } => false,
        }
    }

    /// The received status, headers and request id, when a reply arrived.
    #[must_use]
    pub fn head(&self) -> Option<&ReplyHead> {
        match self {
            Self::Authentication(reply) | Self::SpendLimit(reply) | Self::Rejected(reply) => {
                Some(&reply.reply.head)
            }
            Self::Malformed(malformed) => Some(&malformed.reply.head),
            Self::OutcomeUnknown { head, .. } => Some(head),
            Self::Request(_) | Self::Transport(_) => None,
        }
    }

    /// The reply, when the error is a non-success reply.
    #[must_use]
    pub fn reply(&self) -> Option<&LiveErrorReply> {
        match self {
            Self::Authentication(reply) | Self::SpendLimit(reply) | Self::Rejected(reply) => {
                Some(reply)
            }
            Self::Malformed(_)
            | Self::OutcomeUnknown { .. }
            | Self::Request(_)
            | Self::Transport(_) => None,
        }
    }
}

/// [`LiveApiError::Authentication`] becomes
/// [`ProviderError::InvalidAuthentication`], the other non-success replies
/// [`ProviderError::ProviderResponse`], and a malformed success or an unknown
/// outcome [`ProviderError::Response`].
///
/// The conversion weakens the evidence. Body bytes become text with invalid
/// UTF-8 replaced, an unreadable body becomes an empty one, and a malformed
/// success or unknown outcome keeps only a message, without its headers or
/// structured reply. The decoded `error` object, [`MalformedReason`] and
/// read error are no longer typed, and the spend-limit classification does
/// not survive: [`ProviderError::is_retryable`] judges a 429 by its status.
impl From<LiveApiError> for ProviderError {
    fn from(error: LiveApiError) -> Self {
        match error {
            LiveApiError::Authentication(reply) => {
                ProviderError::InvalidAuthentication(reply.reply.to_provider_response())
            }
            LiveApiError::SpendLimit(reply) | LiveApiError::Rejected(reply) => {
                ProviderError::ProviderResponse(reply.reply.to_provider_response())
            }
            error @ (LiveApiError::Malformed(_) | LiveApiError::OutcomeUnknown { .. }) => {
                ProviderError::Response(error.to_string())
            }
            LiveApiError::Request(error) | LiveApiError::Transport(error) => error,
        }
    }
}

/// Creates public Live API sessions over WebRTC with an OpenAI API-key
/// provider.
///
/// [`Self::create_session`] sends one creation over an HTTP transport. A
/// caller with its own driver uses [`Self::create_request`],
/// [`Self::credential_stamp`] and [`Self::decode_reply`] instead. Every
/// request carries the provider's credential and its configured caller
/// identity, if any.
#[derive(Clone, Debug)]
pub struct PublicLiveSessions {
    provider: OpenAI,
}

impl PublicLiveSessions {
    /// Sessions over `provider`. Refuses a provider whose dialect is not
    /// [`OPENAI`], including the Codex subscription backend and every other
    /// OpenAI-shaped gateway. The base URL is the provider's own, so a custom
    /// one is kept.
    pub fn new(provider: OpenAI) -> Result<Self, NotOfficialOpenAi> {
        if provider.dialect.name != OPENAI.name {
            return Err(NotOfficialOpenAi {
                dialect: provider.dialect.name,
            });
        }
        Ok(Self { provider })
    }

    /// The creation request for `offer_sdp` and `session`, with the
    /// provider's static credential: `POST {base_url}/live/sessions` with
    /// the JSON body `{"session", "transport": {"type": "webrtc", "sdp"}}`.
    ///
    /// When [`Self::credential_stamp`] is `Some`, the driver authorizes the
    /// request with it before sending. Refused for an empty offer, or when
    /// the provider's dialect requires a caller identity it lacks.
    pub fn create_request(
        &self,
        offer_sdp: &str,
        session: &SessionConfig,
    ) -> Result<http::Request<Vec<u8>>, EncodeError> {
        #[derive(serde::Serialize)]
        struct Transport<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            sdp: &'a str,
        }
        #[derive(serde::Serialize)]
        struct Body<'a> {
            session: &'a SessionConfig,
            transport: Transport<'a>,
        }
        if offer_sdp.is_empty() {
            return Err(EncodeError::request(EmptyOffer));
        }
        let url = format!(
            "{}/{LIVE_SESSIONS_PATH}",
            self.provider.base_url.trim_end_matches('/')
        );
        let body = serde_json::to_vec(&Body {
            session,
            transport: Transport {
                kind: "webrtc",
                sdp: offer_sdp,
            },
        })?;
        Ok(self
            .provider
            .identify(self.provider.authenticate(http::Request::post(url)))?
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(body)?)
    }

    /// The credential source the driver reads once for each request, when
    /// the provider has one.
    #[must_use]
    pub fn credential_stamp(&self) -> Option<CredentialStamp> {
        self.provider.credential_stamp()
    }

    /// The reply header naming the provider's request id.
    #[must_use]
    pub fn request_id_header(&self) -> Option<&'static str> {
        self.provider.dialect.request_id_header
    }

    /// Decode the reply to [`Self::create_request`] after the driver has read
    /// its whole body.
    ///
    /// A success status with `{"session": {"id"}, "transport": {"type":
    /// "webrtc", "sdp"}}` is the [`CreatedSession`], which keeps only the
    /// session id and the SDP answer. A non-success status is classified
    /// into [`LiveApiError`], and a success body of any other shape is
    /// [`LiveApiError::Malformed`]; both keep the status, headers, request id
    /// and `body` as [`ReplyBody::Received`].
    pub fn decode_reply(
        &self,
        status: http::StatusCode,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<CreatedSession, LiveApiError> {
        self.decode(
            self.reply_head(status, headers.clone()),
            ReplyBody::Received(body.to_vec()),
        )
    }

    /// The head of a received reply, with the request id read from the
    /// dialect's header.
    pub(crate) fn reply_head(
        &self,
        status: http::StatusCode,
        headers: http::HeaderMap,
    ) -> ReplyHead {
        let request_id =
            crate::providers::internal::request_id_from_headers(&headers, self.request_id_header());
        ReplyHead {
            status,
            headers,
            request_id,
        }
    }

    /// Decode a received reply. A success whose body is unreadable is
    /// [`LiveApiError::OutcomeUnknown`].
    pub(crate) fn decode(
        &self,
        head: ReplyHead,
        body: ReplyBody,
    ) -> Result<CreatedSession, LiveApiError> {
        if !head.status.is_success() {
            return Err(classify_failure(LiveReply { head, body }));
        }
        let created = match body {
            ReplyBody::Unreadable(read_error) => {
                return Err(LiveApiError::OutcomeUnknown { head, read_error });
            }
            ReplyBody::Received(bytes) => {
                decode_created(&bytes).map_err(|reason| (reason, ReplyBody::Received(bytes)))
            }
            ReplyBody::TransportText(text) => decode_created(text.as_bytes())
                .map_err(|reason| (reason, ReplyBody::TransportText(text))),
        };
        created.map_err(|(reason, body)| {
            LiveApiError::Malformed(MalformedSession {
                reason,
                reply: LiveReply { head, body },
            })
        })
    }
}

/// Classify a non-success reply. The spend-limit code is read only from a
/// body that arrived.
fn classify_failure(reply: LiveReply) -> LiveApiError {
    #[derive(Deserialize)]
    struct Envelope {
        error: ApiErrorDetail,
    }
    let error = reply
        .body
        .bytes()
        .and_then(|bytes| serde_json::from_slice::<Envelope>(bytes).ok())
        .map(|envelope| envelope.error);
    let code = error.as_ref().and_then(|error| error.code.as_deref());
    let spend_limit = code.is_some_and(|code| SPEND_LIMIT_CODES.contains(&code));
    let status = reply.head.status;
    let reply = LiveErrorReply { error, reply };
    match status {
        http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN => {
            LiveApiError::Authentication(reply)
        }
        http::StatusCode::TOO_MANY_REQUESTS if spend_limit => LiveApiError::SpendLimit(reply),
        _ => LiveApiError::Rejected(reply),
    }
}

fn decode_created(body: &[u8]) -> Result<CreatedSession, MalformedReason> {
    #[derive(Deserialize)]
    struct Created {
        session: Session,
        transport: Transport,
    }
    #[derive(Deserialize)]
    struct Session {
        id: String,
    }
    #[derive(Deserialize)]
    struct Transport {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        sdp: Option<String>,
    }
    let created: Created = serde_json::from_slice(body).map_err(MalformedReason::Undecodable)?;
    match (created.transport.kind.as_str(), created.transport.sdp) {
        ("webrtc", Some(answer_sdp)) if !answer_sdp.is_empty() => Ok(CreatedSession {
            session_id: created.session.id,
            answer_sdp,
        }),
        _ => Err(MalformedReason::NoWebRtcAnswer),
    }
}

#[cfg(test)]
mod tests;
