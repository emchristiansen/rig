//! Call creation: the offer and session go to the Codex backend, and the
//! answer and call id come back.

use super::{
    CONTROL_SOCKET_BASE_URL, OPENAI_ALPHA_HEADER, QUICKSILVER_V2, REALTIME_CALLS_PATH,
    REALTIME_CALLS_QUERY, SessionConfig, X_SESSION_ID_HEADER,
};
use crate::error::{EncodeError, ProviderError};
use crate::http_client;
use crate::providers::openai::OpenAI;
use crate::providers::openai::responses_api::codex_identity::{
    CodexIdentity, InvalidCodexIdentity, NotACodexWire, SESSION_ID_HEADER, THREAD_ID_HEADER,
};
use crate::providers::openai::wire::ResponsesContract;
use crate::wire::CredentialStamp;

/// A GPT-Live call id: the last `Location` segment of a created call, either
/// `rtc_` followed by at least one character or a dashed 36-character UUID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallId(String);

/// A string that is not a GPT-Live call id.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{0}` is not a GPT-Live call id")]
pub struct InvalidCallId(pub String);

impl From<InvalidCallId> for ProviderError {
    fn from(error: InvalidCallId) -> Self {
        ProviderError::Request(Box::new(error))
    }
}

impl CallId {
    /// Accept `id` when it has the call id shape.
    pub fn new(id: impl Into<String>) -> Result<Self, InvalidCallId> {
        let id = id.into();
        if is_call_id(&id) {
            Ok(Self(id))
        } else {
            Err(InvalidCallId(id))
        }
    }

    /// The id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The call id in a `Location` value: its last path segment, ignoring
    /// any query, that has the call id shape.
    pub fn from_location(location: &str) -> Option<Self> {
        location
            .split('?')
            .next()
            .unwrap_or(location)
            .rsplit('/')
            .find(|segment| is_call_id(segment))
            .map(|segment| Self(segment.to_owned()))
    }
}

/// Decode the successful call-creation response after the driver has read it.
pub(crate) fn decode_created_call(
    headers: &http::HeaderMap,
    body: &[u8],
) -> Result<RealtimeCall, ProviderError> {
    let location = headers
        .get(http::header::LOCATION)
        .ok_or_else(|| ProviderError::Response("the created call has no `Location`".into()))?
        .to_str()
        .map_err(|error| ProviderError::Response(format!("unreadable `Location`: {error}")))?;
    let call_id = CallId::from_location(location).ok_or_else(|| {
        ProviderError::Response(format!("`Location` names no call id: {location}"))
    })?;
    let answer_sdp = String::from_utf8(body.to_vec()).map_err(|error| {
        ProviderError::Response(format!("the SDP answer is not UTF-8: {error}"))
    })?;
    Ok(RealtimeCall {
        answer_sdp,
        call_id,
    })
}

impl std::fmt::Display for CallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn is_call_id(segment: &str) -> bool {
    if let Some(rest) = segment.strip_prefix("rtc_") {
        return !rest.is_empty();
    }
    segment.len() == 36
        && segment.char_indices().all(|(index, ch)| match index {
            8 | 13 | 18 | 23 => ch == '-',
            _ => ch.is_ascii_hexdigit(),
        })
}

/// A created call: the SDP answer for the caller's peer and the call id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RealtimeCall {
    /// The SDP answer, exactly as the backend sent it.
    pub answer_sdp: String,
    /// The call id the control socket joins.
    pub call_id: CallId,
}

/// Creates GPT-Live calls for one conversation identity and joins their
/// control sockets.
///
/// Every request carries the provider's credential (read from its
/// credential source when it has one), `ChatGPT-Account-Id` when an account
/// is set, the exact caller identity, `openai-alpha: quicksilver=v2`, the
/// dashed `session-id` and `thread-id`, and `x-session-id`.
#[derive(Clone, Debug)]
pub struct LiveCalls {
    provider: OpenAI,
    identity: CodexIdentity,
    realtime_session_id: String,
    control_base_url: String,
}

/// The dialect is not the Codex subscription backend.
pub type NotTheCodexBackend = NotACodexWire;

impl LiveCalls {
    /// Calls over `provider`, with a fresh identity whose session id is also
    /// the realtime session id. Refuses a provider whose dialect does not
    /// speak the Codex contract.
    pub fn new(provider: OpenAI) -> Result<Self, NotTheCodexBackend> {
        if provider.dialect.quirks.responses.contract != ResponsesContract::Codex {
            return Err(NotACodexWire {
                dialect: provider.dialect.name,
            });
        }
        let identity = CodexIdentity::generate();
        Ok(Self {
            realtime_session_id: identity.session_id().to_owned(),
            provider,
            identity,
            control_base_url: CONTROL_SOCKET_BASE_URL.to_owned(),
        })
    }

    /// Carry `identity` as `session-id` and `thread-id`, and its session id
    /// as `x-session-id`.
    #[must_use]
    pub fn with_identity(mut self, identity: CodexIdentity) -> Self {
        self.realtime_session_id = identity.session_id().to_owned();
        self.identity = identity;
        self
    }

    /// Send `id` as `x-session-id` instead of the identity's session id.
    /// Refuses an empty or header-unsafe id.
    pub fn with_realtime_session_id(
        mut self,
        id: impl Into<String>,
    ) -> Result<Self, InvalidCodexIdentity> {
        let id = id.into();
        if id.is_empty() || http::HeaderValue::from_str(&id).is_err() {
            return Err(InvalidCodexIdentity {
                field: "realtime_session_id",
                value: id,
            });
        }
        self.realtime_session_id = id;
        Ok(self)
    }

    /// Join control sockets under `base_url` instead of
    /// [`CONTROL_SOCKET_BASE_URL`]; the call id is appended as a path segment.
    #[must_use]
    pub fn with_control_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.control_base_url = base_url.into();
        self
    }

    /// The identity every request carries.
    #[must_use]
    pub fn identity(&self) -> &CodexIdentity {
        &self.identity
    }

    /// The `x-session-id` every request carries.
    #[must_use]
    pub fn realtime_session_id(&self) -> &str {
        &self.realtime_session_id
    }

    /// The call-creation request for `offer_sdp` and `session`, with the
    /// provider's static credential: `POST
    /// {base_url}/realtime/calls?intent=quicksilver&architecture=avas` with
    /// the JSON body `{"sdp", "session"}`. Refused without the caller's exact
    /// identity.
    pub fn call_request(
        &self,
        offer_sdp: &str,
        session: &SessionConfig,
    ) -> Result<http::Request<Vec<u8>>, EncodeError> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            sdp: &'a str,
            session: &'a SessionConfig,
        }
        let url = format!(
            "{}/{REALTIME_CALLS_PATH}?{REALTIME_CALLS_QUERY}",
            self.provider.base_url.trim_end_matches('/')
        );
        let body = serde_json::to_vec(&Body {
            sdp: offer_sdp,
            session,
        })?;
        Ok(self
            .headers(http::Request::post(url))?
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(body)?)
    }

    /// The control socket handshake for `call_id`, with the provider's static
    /// credential: `GET {control base}/{call_id}` carrying the same headers
    /// as call creation, without a body or content type. Refused without the
    /// caller's exact identity.
    pub fn control_request(
        &self,
        call_id: &CallId,
    ) -> Result<http_client::Request<http_client::NoBody>, EncodeError> {
        let url = format!(
            "{}/{}",
            self.control_base_url.trim_end_matches('/'),
            call_id.as_str()
        );
        Ok(self
            .headers(http::Request::get(url))?
            .body(http_client::NoBody)?)
    }

    fn headers(
        &self,
        builder: http::request::Builder,
    ) -> Result<http::request::Builder, EncodeError> {
        let mut builder = self
            .provider
            .identify(self.provider.authenticate(builder))?;
        if let Some(account_id) = &self.provider.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }
        Ok(builder
            .header(OPENAI_ALPHA_HEADER, QUICKSILVER_V2)
            .header(SESSION_ID_HEADER, self.identity.session_id())
            .header(THREAD_ID_HEADER, self.identity.thread_id())
            .header(X_SESSION_ID_HEADER, &self.realtime_session_id))
    }

    /// The reply header naming the provider's request id.
    pub(crate) fn request_id_header(&self) -> Option<&'static str> {
        self.provider.dialect.request_id_header
    }

    /// The credential source the driver reads once for each request, when
    /// the provider has one.
    pub(crate) fn credential_stamp(&self) -> Option<CredentialStamp> {
        self.provider.credential_stamp()
    }
}

#[cfg(test)]
mod tests;
