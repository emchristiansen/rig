//! The caller identity the Codex backend requires on every request.
//!
//! The backend has no default identity. A [`CallerIdentity`] states the
//! `originator`, `user-agent` and optional `version` headers exactly, and a
//! Codex request without one is refused with [`MissingCallerIdentity`] before
//! anything is sent.
//!
//! ```
//! use rig_codex::wire::CallerIdentity;
//!
//! let identity = CallerIdentity::new("codex_exec", "codex_exec/1.0", None)?;
//! assert_eq!(identity.originator(), "codex_exec");
//! # Ok::<(), rig_codex::wire::InvalidCallerIdentity>(())
//! ```

use rig_core::error::{EncodeError, ProviderError};
use serde::{Deserialize, Serialize};

/// The header carrying a caller identity's originator.
pub(crate) const ORIGINATOR_HEADER: &str = "originator";

/// The `version` header a [`CallerIdentity`] carries when it names one.
pub const VERSION_HEADER: &str = "version";

/// The identity the Codex backend requires on every request: the `originator`
/// and `user-agent` headers, and optionally a `version` header.
///
/// Every value is a non-empty, header-safe string, sent exactly as given.
/// [`Self::new`] and deserialization refuse anything else.
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

/// A Codex request refused before it was sent, because no caller identity is
/// configured.
///
/// The backend has no default identity, and this crate never derives a user
/// agent for it: the request carries exactly the caller's values or is not
/// sent.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "`{dialect}` requires the caller's exact identity (`CodexResponses::with_caller_identity`); none is configured, so the request is not sent"
)]
pub struct MissingCallerIdentity {
    /// The dialect's name.
    pub dialect: &'static str,
}

impl From<MissingCallerIdentity> for EncodeError {
    fn from(error: MissingCallerIdentity) -> Self {
        EncodeError::request(error)
    }
}

impl From<MissingCallerIdentity> for ProviderError {
    fn from(error: MissingCallerIdentity) -> Self {
        ProviderError::request(error)
    }
}

#[cfg(test)]
mod tests;
