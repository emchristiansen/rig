//! The caller identity the Codex backend requires on every request of a
//! call: `originator`, `user-agent` and, when the caller names one,
//! `version`, each exactly as the caller states it.

use rig_core::error::{EncodeError, ProviderError};
use serde::{Deserialize, Serialize};

/// The header carrying a caller identity's originator.
const ORIGINATOR_HEADER: &str = "originator";

/// The `version` header a [`CallerIdentity`] carries when it names one.
pub const VERSION_HEADER: &str = "version";

/// The identity a gateway requires on every request, resolved: the
/// `originator` and `user-agent` headers, and optionally a `version` header.
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

    /// Add the `originator`, `user-agent` and, when this identity names one,
    /// `version` headers, each exactly as given.
    pub(crate) fn stamp(&self, builder: http::request::Builder) -> http::request::Builder {
        let builder = builder
            .header(ORIGINATOR_HEADER, &self.originator)
            .header(http::header::USER_AGENT, &self.user_agent);
        match &self.version {
            Some(version) => builder.header(VERSION_HEADER, version),
            None => builder,
        }
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

/// A request refused before it was sent, because its dialect requires the
/// caller's exact identity and none was given
/// ([`LiveCalls::with_caller_identity`](crate::chatgpt::LiveCalls::with_caller_identity)).
///
/// Rig never derives a user agent for such a dialect: the request carries
/// exactly the caller's values or is not sent.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MissingCallerIdentity {
    /// No caller identity is configured.
    #[error(
        "`{dialect}` requires the caller's exact identity (`with_caller_identity`); none is configured, so the request is not sent"
    )]
    Unset {
        /// The dialect's name.
        dialect: &'static str,
    },
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
