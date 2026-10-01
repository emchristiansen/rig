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

/// A non-success reply: the decoded `error` object, when the body has one,
/// and the whole reply.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveErrorReply {
    /// The body's `error` object. `None` when the body is not
    /// `{"error": {...}}`.
    pub error: Option<ApiErrorDetail>,
    /// The reply: status, verbatim body, headers and request id.
    pub response: ProviderResponseError,
}

impl std::fmt::Display for LiveErrorReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.response.fmt(f)
    }
}

/// The `error.code` values of a reached hard spend limit.
const SPEND_LIMIT_CODES: [&str; 2] = [
    "project_spend_limit_exceeded",
    "organization_spend_limit_exceeded",
];

/// A failed session creation, classified so that a caller can stop rather
/// than retry when retrying cannot succeed. Callers that need the
/// spend-limit distinction must branch on this type, not on the
/// [`ProviderError`] it converts into.
#[derive(Debug, thiserror::Error)]
pub enum LiveApiError {
    /// 401 or 403: the credential was rejected. Terminal.
    #[error("the Live API rejected the credential: {0}")]
    Authentication(LiveErrorReply),
    /// 429 with `error.code` `project_spend_limit_exceeded` or
    /// `organization_spend_limit_exceeded`: a hard spend limit is reached.
    /// Terminal until the limit is raised or resets.
    #[error("a Live API hard spend limit is reached: {0}")]
    SpendLimit(LiveErrorReply),
    /// Any other non-success reply.
    #[error("the Live API refused the session: {0}")]
    Rejected(LiveErrorReply),
    /// A success reply that is not a created WebRTC session, as
    /// [`ProviderError::Response`] carrying the body.
    #[error(transparent)]
    Malformed(ProviderError),
    /// The request could not be built or its credential could not be read.
    /// Nothing was sent.
    #[error("the Live session request was not sent: {0}")]
    Request(ProviderError),
    /// The send or the body read failed without a provider reply.
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
    /// [`ProviderResponseError::is_retryable`] accepts, or a
    /// [`Self::Request`] or [`Self::Transport`] failure that
    /// [`ProviderError::is_retryable`] accepts. Never true for a terminal
    /// error.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Rejected(reply) => reply.response.is_retryable(),
            Self::Request(error) | Self::Transport(error) => error.is_retryable(),
            Self::Authentication(_) | Self::SpendLimit(_) | Self::Malformed(_) => false,
        }
    }

    /// The reply, when the error is a non-success reply.
    #[must_use]
    pub fn reply(&self) -> Option<&LiveErrorReply> {
        match self {
            Self::Authentication(reply) | Self::SpendLimit(reply) | Self::Rejected(reply) => {
                Some(reply)
            }
            Self::Malformed(_) | Self::Request(_) | Self::Transport(_) => None,
        }
    }
}

/// [`LiveApiError::Authentication`] becomes
/// [`ProviderError::InvalidAuthentication`] and the other replies
/// [`ProviderError::ProviderResponse`]. The spend-limit classification does
/// not survive: [`ProviderError::is_retryable`] judges a 429 by its status.
impl From<LiveApiError> for ProviderError {
    fn from(error: LiveApiError) -> Self {
        match error {
            LiveApiError::Authentication(reply) => {
                ProviderError::InvalidAuthentication(reply.response)
            }
            LiveApiError::SpendLimit(reply) | LiveApiError::Rejected(reply) => {
                ProviderError::ProviderResponse(reply.response)
            }
            LiveApiError::Malformed(error)
            | LiveApiError::Request(error)
            | LiveApiError::Transport(error) => error,
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
    /// it.
    ///
    /// A success status with `{"session": {"id"}, "transport": {"type":
    /// "webrtc", "sdp"}}` is the [`CreatedSession`]. A non-success status is
    /// classified into [`LiveApiError`], keeping its status, verbatim body,
    /// headers and request id. A success body of any other shape is
    /// [`LiveApiError::Malformed`]. Never [`LiveApiError::Request`] or
    /// [`LiveApiError::Transport`], which only a send produces.
    pub fn decode_reply(
        &self,
        status: http::StatusCode,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<CreatedSession, LiveApiError> {
        if !status.is_success() {
            return Err(self.classify_failure(status, headers, body));
        }
        decode_created(body).map_err(LiveApiError::Malformed)
    }

    /// Classify a non-success reply.
    pub(crate) fn classify_failure(
        &self,
        status: http::StatusCode,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> LiveApiError {
        #[derive(Deserialize)]
        struct Envelope {
            error: ApiErrorDetail,
        }
        let error = serde_json::from_slice::<Envelope>(body)
            .ok()
            .map(|envelope| envelope.error);
        let request_id =
            crate::providers::internal::request_id_from_headers(headers, self.request_id_header());
        let response = ProviderResponseError::new(status, String::from_utf8_lossy(body))
            .with_headers(Some(headers.clone()))
            .with_provider_request_id(request_id);
        let code = error.as_ref().and_then(|error| error.code.as_deref());
        let spend_limit = code.is_some_and(|code| SPEND_LIMIT_CODES.contains(&code));
        let reply = LiveErrorReply { error, response };
        match status {
            http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN => {
                LiveApiError::Authentication(reply)
            }
            http::StatusCode::TOO_MANY_REQUESTS if spend_limit => LiveApiError::SpendLimit(reply),
            _ => LiveApiError::Rejected(reply),
        }
    }
}

fn decode_created(body: &[u8]) -> Result<CreatedSession, ProviderError> {
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
    let body_text = || String::from_utf8_lossy(body).into_owned();
    let created: Created = serde_json::from_slice(body).map_err(|error| {
        ProviderError::Response(format!(
            "the created session does not decode: {error}; body: {}",
            body_text()
        ))
    })?;
    match (created.transport.kind.as_str(), created.transport.sdp) {
        ("webrtc", Some(answer_sdp)) if !answer_sdp.is_empty() => Ok(CreatedSession {
            session_id: created.session.id,
            answer_sdp,
        }),
        _ => Err(ProviderError::Response(format!(
            "the created session has no WebRTC SDP answer; body: {}",
            body_text()
        ))),
    }
}

#[cfg(test)]
mod tests;
