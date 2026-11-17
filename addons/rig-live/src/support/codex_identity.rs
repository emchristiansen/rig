//! The Codex conversation identity a GPT-Live call carries.
//!
//! One [`CodexIdentity`] names a conversation to the Codex backend: the
//! dashed `session-id` and `thread-id` headers every request of a call
//! carries. The same identity on a call and on the conversation's Responses
//! turns keeps them in one conversation.

use rig_core::error::ProviderError;
use serde::{Deserialize, Serialize};

/// The dashed session identity header of the Codex transports.
pub const SESSION_ID_HEADER: &str = "session-id";

/// The dashed thread identity header of the Codex transports.
pub const THREAD_ID_HEADER: &str = "thread-id";

/// The correlation header; Codex sets it to the thread id.
pub const X_CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// The stable Codex cache and correlation identity of one conversation.
///
/// Used for every header that names the conversation, so all of them agree
/// by construction. [`Self::generate`] draws opaque ids from
/// [`rig_core::id::generate`]; [`Self::from_ids`] takes ids the caller
/// derives.
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
    /// Which id: `session_id`, `thread_id` or `realtime_session_id`.
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
    /// A fresh identity.
    #[must_use]
    pub fn generate() -> Self {
        Self {
            session_id: rig_core::id::generate(),
            thread_id: rig_core::id::generate(),
        }
    }

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

    /// The session id: the `session-id` header.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The thread id: the `thread-id` header and the default cache-routing
    /// key.
    #[must_use]
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// The cache-routing key, taken from the thread id.
    #[must_use]
    pub fn prompt_cache_key(&self) -> &str {
        &self.thread_id
    }
}

/// A provider whose dialect does not speak the Codex Responses contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "a Codex session identity needs a Responses wire speaking the Codex contract; dialect `{dialect}` does not"
)]
pub struct NotACodexWire {
    /// The provider's dialect.
    pub dialect: &'static str,
}

impl From<NotACodexWire> for ProviderError {
    fn from(error: NotACodexWire) -> Self {
        ProviderError::request(error)
    }
}
