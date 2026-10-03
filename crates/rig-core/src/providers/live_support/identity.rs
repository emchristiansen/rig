//! Validated caller and conversation identities shared by observed requests and Live.
use serde::{Deserialize, Serialize};
/// Caller originator header.
pub const ORIGINATOR_HEADER: &str = "originator";
/// Conversation session header.
pub const SESSION_ID_HEADER: &str = "session-id";
/// Conversation thread header.
pub const THREAD_ID_HEADER: &str = "thread-id";
/// The identity a gateway requires on every request, resolved: the
/// `originator` and `user-agent` headers, and optionally a `version` header.
///
/// Every value is a non-empty, header-safe string, sent exactly as given.
/// [`Self::new`], deserialization and the dialect's environment variables
/// refuse anything else.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CallerIdentityValues", into = "CallerIdentityValues")]
pub struct CallerIdentity {
    originator: String,
    user_agent: String,
    version: Option<String>,
}

/// The serialized form of a [`CallerIdentity`], validated on the way back in.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CallerIdentityValues {
    originator: String,
    user_agent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

impl TryFrom<CallerIdentityValues> for CallerIdentity {
    type Error = InvalidCallerIdentity;

    fn try_from(values: CallerIdentityValues) -> Result<Self, Self::Error> {
        Self::new(values.originator, values.user_agent, values.version)
    }
}

impl From<CallerIdentity> for CallerIdentityValues {
    fn from(identity: CallerIdentity) -> Self {
        Self {
            originator: identity.originator,
            user_agent: identity.user_agent,
            version: identity.version,
        }
    }
}

/// The `version` header a [`CallerIdentity`] carries when it names one.
pub const VERSION_HEADER: &str = "version";

/// A caller identity value no request could carry: empty, or not a valid
/// HTTP header value.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a caller identity's {field} must be a non-empty, header-safe string; `{value}` is not")]
pub struct InvalidCallerIdentity {
    /// Which value: `originator`, `user_agent` or `version`.
    pub field: &'static str,
    /// The value as supplied.
    pub value: String,
}

/// Refuse a caller identity value no request could carry.
fn checked_identity_value(
    field: &'static str,
    value: String,
) -> Result<String, InvalidCallerIdentity> {
    if value.is_empty() || http::HeaderValue::from_str(&value).is_err() {
        Err(InvalidCallerIdentity { field, value })
    } else {
        Ok(value)
    }
}

impl CallerIdentity {
    /// An identity sending `originator`, `user_agent` and, when given,
    /// `version`, each exactly as given. Refuses an empty or header-unsafe
    /// value, naming which.
    pub fn new(
        originator: impl Into<String>,
        user_agent: impl Into<String>,
        version: Option<String>,
    ) -> Result<Self, InvalidCallerIdentity> {
        Ok(Self {
            originator: checked_identity_value("originator", originator.into())?,
            user_agent: checked_identity_value("user_agent", user_agent.into())?,
            version: version
                .map(|version| checked_identity_value("version", version))
                .transpose()?,
        })
    }

    /// Add the `originator`, `user-agent` and, when this identity names one,
    /// `version` headers, each exactly as given: what a model request
    /// carries.
    pub(crate) fn stamp(&self, builder: http::request::Builder) -> http::request::Builder {
        let builder = self.stamp_client(builder);
        match &self.version {
            Some(version) => builder.header(VERSION_HEADER, version),
            None => builder,
        }
    }

    /// Add only the `originator` and `user-agent` headers, exactly as given:
    /// what the official client's own HTTP client carries on every request,
    /// its sign-in service's token refresh included. `version` is a model
    /// request header, so it is not added here.
    pub(crate) fn stamp_client(&self, builder: http::request::Builder) -> http::request::Builder {
        builder
            .header(ORIGINATOR_HEADER, &self.originator)
            .header(http::header::USER_AGENT, &self.user_agent)
    }

    /// The `originator` header.
    #[must_use]
    pub fn originator(&self) -> &str {
        &self.originator
    }

    /// The `user-agent` header.
    #[must_use]
    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }

    /// The `version` header, when this identity names one.
    #[must_use]
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }
}

/// The stable Codex cache and correlation identity of one conversation.
///
/// Used for every header and body field that names the conversation, so all
/// of them agree by construction. [`Self::generate`] draws opaque ids from
/// [`crate::id::generate`]; [`Self::from_ids`] takes ids the caller derives.
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

impl CodexIdentity {
    /// A fresh identity.
    #[must_use]
    pub fn generate() -> Self {
        Self {
            session_id: crate::id::generate(),
            thread_id: crate::id::generate(),
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
}

#[cfg(test)]
mod tests;
