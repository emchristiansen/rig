//! The Codex cache and correlation identity, shared by both transports.
//!
//! One [`CodexIdentity`] names a conversation to the Codex backend: the
//! dashed `session-id` and `thread-id` headers, the `x-client-request-id`
//! correlation header, and the `prompt_cache_key` and `client_metadata` body
//! fields. A websocket session stamps it on its handshake and every frame
//! ([`super::websocket::codex`]); a Codex wire given one
//! ([`super::wire::CodexResponses::with_codex_identity`]) has it stamped on
//! every HTTP request its [`CodexHttpClient`](crate::http_client::CodexHttpClient)
//! sends. The same identity on both keeps a conversation's cache affinity
//! whichever transport carries a turn.
//!
//! ```
//! use rig_codex::responses_api::codex_identity::CodexIdentity;
//!
//! let identity = CodexIdentity::from_ids("session-1", "thread-1")?;
//! assert_eq!(identity.prompt_cache_key(), "thread-1");
//! # Ok::<(), rig_codex::responses_api::codex_identity::InvalidCodexIdentity>(())
//! ```

use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::providers::openai::wire::ResponsesContract;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::wire::CodexRequestError;
use rig_core::error::ProviderError;

/// The dashed session identity header of the Codex transports.
pub const SESSION_ID_HEADER: &str = "session-id";

/// The dashed thread identity header of the Codex transports.
pub const THREAD_ID_HEADER: &str = "thread-id";

/// The correlation header; Codex sets it to the thread id.
pub const X_CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// The top-level body field carrying the cache-routing key.
pub(crate) const PROMPT_CACHE_KEY_FIELD: &str = "prompt_cache_key";

/// The top-level body field carrying correlation metadata.
pub(crate) const CLIENT_METADATA_FIELD: &str = "client_metadata";

/// The `client_metadata` key carrying the session id (Codex spelling).
pub const SESSION_ID_METADATA_KEY: &str = "session_id";

/// The `client_metadata` key carrying the thread id (Codex spelling).
pub const THREAD_ID_METADATA_KEY: &str = "thread_id";

/// The stable Codex cache and correlation identity of one conversation.
///
/// Used for every header and body field that names the conversation, so all
/// of them agree by construction. The ids are always the caller's
/// ([`Self::from_ids`]): this crate draws none itself, so everything a
/// request or frame carries is a function of what the caller supplied.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CodexIdentityIds", into = "CodexIdentityIds")]
pub struct CodexIdentity {
    session_id: String,
    thread_id: String,
}

/// The serialized form of a [`CodexIdentity`], validated on the way back in.
#[derive(Serialize, Deserialize)]
struct CodexIdentityIds {
    session_id: String,
    thread_id: String,
}

impl TryFrom<CodexIdentityIds> for CodexIdentity {
    type Error = InvalidCodexIdentity;

    fn try_from(ids: CodexIdentityIds) -> Result<Self, Self::Error> {
        Self::from_ids(ids.session_id, ids.thread_id)
    }
}

impl From<CodexIdentity> for CodexIdentityIds {
    fn from(identity: CodexIdentity) -> Self {
        Self {
            session_id: identity.session_id,
            thread_id: identity.thread_id,
        }
    }
}

/// A caller-supplied Codex id that cannot name a conversation: empty, or not
/// a valid HTTP header value.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a Codex {field} must be a non-empty, header-safe string; `{value}` is not")]
pub struct InvalidCodexIdentity {
    /// Which id: `session_id` or `thread_id`.
    pub field: &'static str,
    /// The id as supplied.
    pub value: String,
}

impl From<InvalidCodexIdentity> for ProviderError {
    fn from(error: InvalidCodexIdentity) -> Self {
        ProviderError::request(error)
    }
}

impl CodexIdentity {
    /// An identity from ids the caller derives (for example from a
    /// prefix-cache seed), sent exactly as given. Refuses an empty id, or one
    /// no HTTP header can carry.
    pub fn from_ids(
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
    ) -> Result<Self, InvalidCodexIdentity> {
        fn checked(field: &'static str, value: String) -> Result<String, InvalidCodexIdentity> {
            if value.is_empty() || http::HeaderValue::from_str(&value).is_err() {
                Err(InvalidCodexIdentity { field, value })
            } else {
                Ok(value)
            }
        }
        Ok(Self {
            session_id: checked("session_id", session_id.into())?,
            thread_id: checked("thread_id", thread_id.into())?,
        })
    }

    /// The session id: the `session-id` header and the `session_id` metadata key.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The thread id: the `thread-id` and `x-client-request-id` headers, the
    /// `thread_id` metadata key, and the default `prompt_cache_key`.
    #[must_use]
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// The cache-routing key, taken from the thread id. (Codex's own client
    /// keys its cache by its session id instead.)
    #[must_use]
    pub fn prompt_cache_key(&self) -> &str {
        &self.thread_id
    }

    /// The identity headers: dashed `session-id` and `thread-id`, and
    /// `x-client-request-id` set to the thread id.
    pub(crate) fn headers(&self) -> [(&'static str, &str); 3] {
        [
            (SESSION_ID_HEADER, &self.session_id),
            (THREAD_ID_HEADER, &self.thread_id),
            (X_CLIENT_REQUEST_ID_HEADER, &self.thread_id),
        ]
    }

    /// Stamp this identity onto a `response.create` or `/responses` body:
    /// `prompt_cache_key` and the `client_metadata` session and thread keys,
    /// each only where the request does not already carry its own value.
    pub(crate) fn stamp(&self, body: &mut Map<String, Value>) -> Result<(), CodexRequestError> {
        body.entry(PROMPT_CACHE_KEY_FIELD)
            .or_insert_with(|| Value::String(self.prompt_cache_key().to_owned()));

        let metadata = body
            .entry(CLIENT_METADATA_FIELD)
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(metadata) = metadata else {
            return Err(CodexRequestError::ClientMetadata {
                found: metadata.clone(),
            });
        };
        metadata
            .entry(SESSION_ID_METADATA_KEY)
            .or_insert_with(|| Value::String(self.session_id.clone()));
        metadata
            .entry(THREAD_ID_METADATA_KEY)
            .or_insert_with(|| Value::String(self.thread_id.clone()));
        Ok(())
    }
}

/// A Codex websocket session with no conversation identity to carry.
///
/// The session draws no ids of its own: the wire names the conversation
/// ([`CodexResponses::with_codex_identity`](super::wire::CodexResponses::with_codex_identity))
/// or the builder is given one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, thiserror::Error)]
#[error(
    "a Codex websocket session needs a conversation identity; neither the wire nor the builder names one, so nothing is sent"
)]
pub struct MissingCodexIdentity;

impl From<MissingCodexIdentity> for ProviderError {
    fn from(error: MissingCodexIdentity) -> Self {
        ProviderError::request(error)
    }
}

/// A wire that does not speak the Codex Responses contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "a Codex session identity needs a Responses wire speaking the Codex contract; dialect `{dialect}` does not"
)]
pub struct NotACodexWire {
    /// The wire's dialect.
    pub dialect: &'static str,
}

impl From<NotACodexWire> for ProviderError {
    fn from(error: NotACodexWire) -> Self {
        ProviderError::request(error)
    }
}

pub(crate) fn require_codex(wire: &Responses) -> Result<(), NotACodexWire> {
    if wire.provider.dialect.quirks.responses.contract == ResponsesContract::Codex {
        Ok(())
    } else {
        Err(NotACodexWire {
            dialect: wire.provider.dialect.name,
        })
    }
}

#[cfg(test)]
mod tests;
