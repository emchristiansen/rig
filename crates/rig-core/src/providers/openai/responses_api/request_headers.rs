//! Caller-supplied request headers a Responses wire adds to everything it
//! sends.
//!
//! A gateway can ask for headers Rig does not model, such as the Codex
//! backend's `x-codex-*` request headers. [`RequestHeaders`] carries them
//! exactly as the caller states them: a [`Responses`] wire given a set
//! ([`Responses::with_request_headers`]) adds it to every HTTP request and to
//! every websocket handshake it opens. Rig builds none of the values.
//!
//! Headers Rig already sends from another source are refused, so one value
//! never reaches the wire twice or contradicts the source that owns it: the
//! credential, the caller identity, the Codex conversation identity, the
//! account, the websocket beta opt-in, the Responses Lite marker, content
//! framing, and the headers a websocket backend generates for the upgrade.
//!
//! [`Responses`]: super::wire::Responses
//! [`Responses::with_request_headers`]: super::wire::Responses::with_request_headers

use serde::{Deserialize, Serialize};

use crate::error::ProviderError;

/// Header names a caller may not supply, lowercase: Rig sends each from the
/// one source that owns it, or the transport generates it.
const RESERVED: &[&str] = &[
    // The credential.
    "authorization",
    "api-key",
    // The caller identity (`OpenAI::with_caller_identity`).
    "originator",
    "user-agent",
    super::super::wire::VERSION_HEADER,
    // The Codex conversation identity (`CodexIdentity`), and the per-request
    // session id the ChatGPT dialect sends without one.
    super::codex_identity::SESSION_ID_HEADER,
    super::codex_identity::THREAD_ID_HEADER,
    super::codex_identity::X_CLIENT_REQUEST_ID_HEADER,
    "session_id",
    // The account and the websocket beta opt-in.
    "chatgpt-account-id",
    "openai-beta",
    // The Responses Lite marker (`Responses::with_responses_lite`).
    super::responses_lite::HTTP_HEADER,
    // Content framing.
    "accept",
    "content-type",
    "content-encoding",
    "content-length",
    "transfer-encoding",
    // Connection and websocket upgrade headers.
    "host",
    "connection",
    "upgrade",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
];

/// A header a caller cannot add to a Responses wire's requests.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InvalidRequestHeader {
    /// The name is not a valid HTTP header name.
    #[error("`{name}` is not a valid HTTP header name")]
    Name {
        /// The name as supplied.
        name: String,
    },
    /// The value is empty or not a valid HTTP header value.
    #[error(
        "the `{name}` header's value must be a non-empty, header-safe string; `{value}` is not"
    )]
    Value {
        /// The header's name, lowercase.
        name: String,
        /// The value as supplied.
        value: String,
    },
    /// Rig already sends this header from the source that owns it.
    #[error("`{name}` is a header Rig sends itself; it cannot be supplied as a request header")]
    Reserved {
        /// The header's name, lowercase.
        name: String,
    },
    /// The set already names this header.
    #[error("the `{name}` header is already in this set")]
    Duplicate {
        /// The header's name, lowercase.
        name: String,
    },
}

impl From<InvalidRequestHeader> for ProviderError {
    fn from(error: InvalidRequestHeader) -> Self {
        ProviderError::Request(Box::new(error))
    }
}

/// An ordered set of caller-supplied request headers, each valid, none
/// reserved and none repeated. Names are kept lowercase; values exactly as
/// given.
///
/// Built with [`Self::with`]; its serialized form is validated the same way
/// on the way back in.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<(String, String)>", into = "Vec<(String, String)>")]
pub struct RequestHeaders {
    headers: Vec<(String, String)>,
}

impl RequestHeaders {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// This set with `name: value` added after the headers already in it.
    /// Refuses an invalid name or value, a name Rig sends itself, and a name
    /// already in the set.
    pub fn with(
        mut self,
        name: impl AsRef<str>,
        value: impl Into<String>,
    ) -> Result<Self, InvalidRequestHeader> {
        let supplied = name.as_ref();
        let name = http::HeaderName::from_bytes(supplied.as_bytes())
            .map_err(|_| InvalidRequestHeader::Name {
                name: supplied.to_owned(),
            })?
            .as_str()
            .to_owned();
        if RESERVED.contains(&name.as_str()) {
            return Err(InvalidRequestHeader::Reserved { name });
        }
        if self.headers.iter().any(|(existing, _)| *existing == name) {
            return Err(InvalidRequestHeader::Duplicate { name });
        }
        let value = value.into();
        if value.is_empty() || http::HeaderValue::from_str(&value).is_err() {
            return Err(InvalidRequestHeader::Value { name, value });
        }
        self.headers.push((name, value));
        Ok(self)
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// The headers, lowercase name and value, in the order they were added.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
    }

    /// Add every header in the set to `builder`, in order.
    pub(crate) fn stamp(&self, builder: http::request::Builder) -> http::request::Builder {
        self.iter().fold(builder, |builder, (name, value)| {
            builder.header(name, value)
        })
    }
}

impl TryFrom<Vec<(String, String)>> for RequestHeaders {
    type Error = InvalidRequestHeader;

    fn try_from(headers: Vec<(String, String)>) -> Result<Self, Self::Error> {
        headers
            .into_iter()
            .try_fold(Self::new(), |set, (name, value)| set.with(name, value))
    }
}

impl From<RequestHeaders> for Vec<(String, String)> {
    fn from(set: RequestHeaders) -> Self {
        set.headers
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
