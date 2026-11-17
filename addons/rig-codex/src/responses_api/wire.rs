//! The Codex settings a Responses wire is sent with.
//!
//! Rig's [`Responses`] wire encodes the request. [`CodexResponses`] pairs it
//! with what the Codex backend needs beside it: the caller's exact identity,
//! the conversation's [`CodexIdentity`], caller-supplied
//! [`RequestHeaders`] and [`RequestCompression`]. A
//! [`CodexHttpClient`](crate::http_client::CodexHttpClient) applies them to
//! every HTTP request, and a
//! [`CodexWebSocketSession`](super::websocket::codex::CodexWebSocketSession)
//! to its handshake and frames.
//!
//! ```
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::providers::openai::responses_api::wire::Responses;
//!
//! let wire = OpenAIConfig::with_key(&chatgpt::DIALECT, "access-token");
//! let wire = CodexResponses::new(Responses::new(wire, "gpt-5.4"))?;
//! assert!(wire.codex_identity.is_none());
//! # Ok::<(), rig_codex::responses_api::codex_identity::NotACodexWire>(())
//! ```

use super::codex_identity::{CodexIdentity, NotACodexWire, require_codex};
use super::request_compression::{RequestCompression, RequestCompressionError};
use super::request_headers::RequestHeaders;
use crate::wire::{CallerIdentity, MissingCallerIdentity, ORIGINATOR_HEADER, VERSION_HEADER};
use http::{HeaderMap, HeaderName, HeaderValue};
use rig_core::completion::CompletionRequest;
use rig_core::error::{EncodeError, ProviderError};
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::providers::openai::wire::Auth;
use serde::{Deserialize, Serialize};

/// The header naming the account a credential belongs to.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The underscore session header some dialects send per request. A Codex
/// identity replaces it with the dashed pair.
const SESSION_ID_PER_REQUEST_HEADER: &str = "session_id";

/// The header [`Auth::ApiKeyHeader`] sends the credential in.
const API_KEY_HEADER: &str = "api-key";

/// A Responses wire with the Codex settings its requests are sent with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodexResponses {
    /// The Responses wire that encodes each request.
    pub wire: Responses,
    /// The caller identity every request carries. `None` refuses every send
    /// with [`MissingCallerIdentity`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_identity: Option<CallerIdentity>,
    /// The conversation identity stamped on every HTTP request, and the one a
    /// websocket session built from this wire carries. `None` leaves HTTP
    /// requests as Rig encodes them, and a websocket session then takes its
    /// identity from
    /// [`with_identity`](super::websocket::codex::CodexWebSocketSessionBuilder::with_identity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_identity: Option<CodexIdentity>,
    /// Caller-supplied headers added to every HTTP request and every
    /// websocket handshake.
    #[serde(default, skip_serializing_if = "RequestHeaders::is_empty")]
    pub request_headers: RequestHeaders,
    /// How HTTP request bodies are encoded. Sent as they are by default.
    #[serde(default, skip_serializing_if = "RequestCompression::is_none")]
    pub request_compression: RequestCompression,
}

/// A Codex request that could not be given its headers or body fields.
#[derive(Debug, thiserror::Error)]
pub enum CodexRequestError {
    /// The wire's dialect does not speak the Codex contract.
    #[error(transparent)]
    NotACodexWire(#[from] NotACodexWire),
    /// No caller identity is configured.
    #[error(transparent)]
    MissingCallerIdentity(#[from] MissingCallerIdentity),
    /// A header value the request cannot carry.
    #[error("the `{name}` header cannot carry its value: {source}")]
    Header {
        /// The header's name.
        name: String,
        /// Why the value was refused.
        #[source]
        source: http::header::InvalidHeaderValue,
    },
    /// A caller-supplied header whose name is not a header name.
    #[error("`{name}` is not a valid HTTP header name")]
    HeaderName {
        /// The name as supplied.
        name: String,
    },
    /// The request body is not a JSON object.
    #[error("a Codex request body must be a JSON object: {0}")]
    Body(#[source] serde_json::Error),
    /// The body's `client_metadata` is not a JSON object.
    #[error(
        "the Codex request's `client_metadata` must be a JSON object to carry the session identity, got {found}"
    )]
    ClientMetadata {
        /// The value found in its place.
        found: serde_json::Value,
    },
    /// The body could not be compressed.
    #[error(transparent)]
    Compression(#[from] RequestCompressionError),
}

// Each conversion hands on a refusal that has its own type as that type, so a
// caller can downcast to it: `MissingCallerIdentity` stays one however it
// reached the caller.
impl From<CodexRequestError> for ProviderError {
    fn from(error: CodexRequestError) -> Self {
        match error {
            CodexRequestError::NotACodexWire(error) => ProviderError::request(error),
            CodexRequestError::MissingCallerIdentity(error) => ProviderError::request(error),
            CodexRequestError::Compression(error) => ProviderError::request(error),
            other => ProviderError::request(other),
        }
    }
}

impl From<CodexRequestError> for EncodeError {
    fn from(error: CodexRequestError) -> Self {
        match error {
            CodexRequestError::NotACodexWire(error) => EncodeError::request(error),
            CodexRequestError::MissingCallerIdentity(error) => EncodeError::request(error),
            CodexRequestError::Compression(error) => EncodeError::request(error),
            other => EncodeError::request(other),
        }
    }
}

impl From<CodexRequestError> for rig_http::http_client::Error {
    fn from(error: CodexRequestError) -> Self {
        use rig_http::http_client::Error;
        match error {
            CodexRequestError::NotACodexWire(error) => Error::instance(error),
            CodexRequestError::MissingCallerIdentity(error) => Error::instance(error),
            CodexRequestError::Compression(error) => Error::instance(error),
            other => Error::instance(other),
        }
    }
}

/// A request control this crate refuses to send on the Codex contract.
///
/// Rig's encoder leaves a caller-set output-token cap and temperature off a
/// Codex request without saying so. A
/// [`CodexWebSocketSession`](super::websocket::codex::CodexWebSocketSession)
/// refuses each by name instead, before anything is sent, so a caller learns
/// that the control it set would not have reached the backend. This states
/// adapter policy, not a rule the backend is known to enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnsupportedCodexControl {
    /// A caller-set output-token cap (`max_output_tokens`).
    #[error(
        "this adapter refuses a caller-set `max_output_tokens` on the Codex Responses \
         contract; leave the output-token cap unset for this dialect"
    )]
    MaxOutputTokens,
    /// A caller-set `store: true`; `store: false` is what this adapter states
    /// for the Codex contract.
    #[error(
        "this adapter refuses a caller-set `store: true` on the Codex Responses contract; \
         leave `store` unset or send `false`"
    )]
    Store,
    /// A caller-set nucleus-sampling value (`top_p`).
    #[error(
        "this adapter refuses a caller-set `top_p` on the Codex Responses contract; \
         leave `top_p` unset for this dialect"
    )]
    TopP,
    /// A caller-set sampling temperature (`temperature`).
    #[error(
        "this adapter refuses a caller-set `temperature` on the Codex Responses contract; \
         leave `temperature` unset for this dialect"
    )]
    Temperature,
}

impl UnsupportedCodexControl {
    /// The first control `request` sets that the Codex contract does not
    /// take, in the order output-token cap, temperature, `top_p`, `store`.
    #[must_use]
    pub fn in_request(request: &CompletionRequest) -> Option<Self> {
        let param = |name: &str| {
            request
                .additional_params
                .as_ref()
                .and_then(|params| params.get(name))
                .filter(|value| !value.is_null())
        };
        if request.max_tokens.is_some() {
            Some(Self::MaxOutputTokens)
        } else if request.temperature.is_some() {
            Some(Self::Temperature)
        } else if param("top_p").is_some() {
            Some(Self::TopP)
        } else if param("store") == Some(&serde_json::Value::Bool(true)) {
            Some(Self::Store)
        } else {
            None
        }
    }
}

impl From<UnsupportedCodexControl> for ProviderError {
    fn from(error: UnsupportedCodexControl) -> Self {
        ProviderError::request(error)
    }
}

impl CodexResponses {
    /// Codex settings for `wire`, with no caller identity, no conversation
    /// identity, no request headers and no compression. Refuses a wire whose
    /// dialect does not speak the Codex contract.
    pub fn new(wire: Responses) -> Result<Self, NotACodexWire> {
        require_codex(&wire)?;
        Ok(Self {
            wire,
            caller_identity: None,
            codex_identity: None,
            request_headers: RequestHeaders::new(),
            request_compression: RequestCompression::default(),
        })
    }

    /// Send every request with exactly `identity`'s `originator`,
    /// `user-agent` and optional `version` headers.
    #[must_use]
    pub fn with_caller_identity(mut self, identity: CallerIdentity) -> Self {
        self.caller_identity = Some(identity);
        self
    }

    /// Name one conversation on every HTTP request and on the websocket
    /// session built from this wire. Refuses a wire whose dialect does not
    /// speak the Codex contract.
    pub fn with_codex_identity(mut self, identity: CodexIdentity) -> Result<Self, NotACodexWire> {
        require_codex(&self.wire)?;
        self.codex_identity = Some(identity);
        Ok(self)
    }

    /// Add `headers` to every HTTP request and every websocket handshake.
    #[must_use]
    pub fn with_request_headers(mut self, headers: RequestHeaders) -> Self {
        self.request_headers = headers;
        self
    }

    /// Encode every HTTP request body with `compression`.
    #[must_use]
    pub fn with_request_compression(mut self, compression: RequestCompression) -> Self {
        self.request_compression = compression;
        self
    }

    /// The caller identity a request is sent with, or the refusal when the
    /// wire has none.
    pub fn sendable_identity(&self) -> Result<&CallerIdentity, MissingCallerIdentity> {
        self.caller_identity.as_ref().ok_or(MissingCallerIdentity {
            dialect: self.wire.provider.dialect.name,
        })
    }

    /// Set the credential and account headers Rig's own encoder sends, for a
    /// request this crate builds itself.
    pub(crate) fn stamp_credential(
        &self,
        headers: &mut HeaderMap,
    ) -> Result<(), CodexRequestError> {
        let provider = &self.wire.provider;
        let bearer = || format!("Bearer {}", provider.api_key.expose());
        match provider.auth {
            Auth::Bearer => set(headers, http::header::AUTHORIZATION.as_str(), &bearer())?,
            Auth::OptionalBearer if provider.api_key.is_empty() => {}
            Auth::OptionalBearer => set(headers, http::header::AUTHORIZATION.as_str(), &bearer())?,
            Auth::ApiKeyHeader => set(headers, API_KEY_HEADER, provider.api_key.expose())?,
        }
        if let Some(account_id) = &provider.account_id {
            set(headers, CHATGPT_ACCOUNT_ID_HEADER, account_id)?;
        }
        Ok(())
    }

    /// Make `headers` carry exactly this wire's caller identity, the Codex
    /// identity `identity` when there is one, and the caller's request
    /// headers, replacing any value another source put under those names.
    pub(crate) fn stamp_headers(
        &self,
        identity: Option<&CodexIdentity>,
        headers: &mut HeaderMap,
    ) -> Result<(), CodexRequestError> {
        require_codex(&self.wire)?;
        let caller = self.sendable_identity()?;
        set(headers, ORIGINATOR_HEADER, caller.originator())?;
        set(
            headers,
            http::header::USER_AGENT.as_str(),
            caller.user_agent(),
        )?;
        match caller.version() {
            Some(version) => set(headers, VERSION_HEADER, version)?,
            None => {
                headers.remove(VERSION_HEADER);
            }
        }
        if let Some(identity) = identity {
            // The conversation's identity replaces a per-request session id.
            headers.remove(SESSION_ID_PER_REQUEST_HEADER);
            for (name, value) in identity.headers() {
                set(headers, name, value)?;
            }
        }
        for (name, value) in self.request_headers.iter() {
            set(headers, name, value)?;
        }
        Ok(())
    }
}

/// Set `name` to `value` in `headers`, replacing any value it had.
pub(crate) fn set(
    headers: &mut HeaderMap,
    name: &str,
    value: &str,
) -> Result<(), CodexRequestError> {
    let header =
        HeaderName::from_bytes(name.as_bytes()).map_err(|_| CodexRequestError::HeaderName {
            name: name.to_owned(),
        })?;
    let value = HeaderValue::from_str(value).map_err(|source| CodexRequestError::Header {
        name: name.to_owned(),
        source,
    })?;
    headers.insert(header, value);
    Ok(())
}

#[cfg(test)]
mod tests;
