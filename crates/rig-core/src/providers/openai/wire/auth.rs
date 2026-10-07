//! How an OpenAI-shaped request carries its credential and the caller
//! identity a gateway requires.
//!
//! ```
//! use rig_core::providers::openai::{OpenAIConfig, wire::Auth};
//! let config = OpenAIConfig::new("key").with_auth(Auth::OptionalBearer);
//! assert_eq!(config.auth, Auth::OptionalBearer);
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::error::{EncodeError, ProviderError};
use crate::wire::{
    CredentialPlacement, CredentialSource, CredentialSourceHandle, CredentialStamp, Secret,
    TokenPlacement,
};

use super::{Dialect, OpenAIConfig, Quirks};

/// Credential header policy, including omission of empty optional tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Auth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `Authorization: Bearer <key>`, omitted entirely when the key is empty.
    OptionalBearer,
    /// Azure's `api-key: <key>`.
    ApiKeyHeader,
}

/// Alternative credential environment variable and its authentication policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthAlternative {
    /// The variable holding this credential.
    pub api_key_env: &'static str,
    /// How it is sent.
    pub auth: Auth,
}

/// The caller identity a gateway requires on every request.
///
/// The gateway has no default identity: the caller supplies one exactly
/// ([`OpenAIConfig::with_caller_identity`], or both of the environment variables
/// below), and every request refuses with [`MissingCallerIdentity`] until
/// it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// The environment variable carrying `originator`. It is read only
    /// together with [`Self::user_agent_env`].
    pub originator_env: &'static str,
    /// The environment variable carrying `user-agent`. It is read only
    /// together with [`Self::originator_env`].
    pub user_agent_env: &'static str,
    /// Whether every request carries a fresh `session_id` header.
    pub session_ids: bool,
}

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

/// The caller identity a configuration sends, and whether its user agent is
/// the caller's own.
///
/// Serialized with a `kind` tag, so a configuration serialized before the
/// distinction existed does not deserialize at all rather than turning a
/// derived user agent into an exact one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfiguredIdentity {
    /// Exactly the caller's values, from [`OpenAIConfig::with_caller_identity`] or
    /// from both of a dialect's identity environment variables.
    Exact(CallerIdentity),
    /// Only an originator, from [`OpenAIConfig::with_originator`]. Rig sends a
    /// `rig/…` user agent naming it, so a dialect that requires a caller
    /// identity refuses to send this one ([`MissingCallerIdentity`]).
    Originator {
        /// The `originator` header, taken as given: one a header cannot
        /// carry fails when the request is built.
        originator: String,
    },
}

/// A request refused before it was sent, because its dialect requires the
/// caller's exact identity ([`Identity`]) and the configuration has none.
///
/// Such a dialect has no default identity, and Rig never derives a user
/// agent for it: the request carries exactly the caller's values or is not
/// sent.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MissingCallerIdentity {
    /// No caller identity is configured.
    #[error(
        "`{dialect}` requires the caller's exact identity (`OpenAIConfig::with_caller_identity`); none is configured, so the request is not sent"
    )]
    Unset {
        /// The dialect's name.
        dialect: &'static str,
    },
    /// Only an originator is configured ([`OpenAIConfig::with_originator`]), whose
    /// user agent Rig would derive.
    #[error(
        "`{dialect}` requires the caller's exact identity (`OpenAIConfig::with_caller_identity`); only the originator `{originator}` is configured, and Rig derives no user agent for it, so the request is not sent"
    )]
    OriginatorOnly {
        /// The dialect's name.
        dialect: &'static str,
        /// The configured originator.
        originator: String,
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

/// The header carrying a caller identity's originator.
const ORIGINATOR_HEADER: &str = "originator";

/// The user agent sent with an originator-only identity on a dialect that
/// requires no caller identity: the crate, the host, and who is calling.
fn default_user_agent(originator: &str) -> String {
    format!(
        "rig/{} ({} {}; {originator})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// The exact caller identity a dialect's environment variables name, read
/// through `lookup`: both variables set (non-empty) give one, neither gives
/// none, and one without the other is refused, since Rig derives no half of
/// an identity.
pub(super) fn identity_from_env(
    identity: Identity,
    lookup: impl Fn(&'static str) -> Result<Option<String>, EnvError>,
) -> Result<Option<CallerIdentity>, EnvError> {
    let read = |name: &'static str| -> Result<Option<String>, EnvError> {
        Ok(lookup(name)?.filter(|value| !value.is_empty()))
    };
    let originator = read(identity.originator_env)?;
    let user_agent = read(identity.user_agent_env)?;
    let (originator, user_agent) = match (originator, user_agent) {
        (None, None) => return Ok(None),
        (Some(originator), Some(user_agent)) => (originator, user_agent),
        (set, _) => {
            let (present, missing) = if set.is_some() {
                (identity.originator_env, identity.user_agent_env)
            } else {
                (identity.user_agent_env, identity.originator_env)
            };
            return Err(EnvError::Invalid {
                name: missing,
                detail: format!(
                    "`{present}` is set, so `{missing}` must be set too: this gateway takes the caller's exact identity, never half of one"
                ),
            });
        }
    };
    CallerIdentity::new(originator, user_agent, None)
        .map(Some)
        .map_err(|error| EnvError::Invalid {
            name: if error.field == "originator" {
                identity.originator_env
            } else {
                identity.user_agent_env
            },
            detail: error.to_string(),
        })
}

impl OpenAIConfig {
    /// `dialect` with the credential it accepts through its
    /// [`alternate_auth`](Dialect::alternate_auth) variable, sent with that
    /// alternative's header.
    ///
    /// Azure's account key and its Entra bearer token are both credentials
    /// for the same account but go out under different headers, so which one
    /// is held has to be recorded rather than guessed from the value.
    pub fn with_alternate_key(dialect: &Dialect, api_key: impl Into<Secret>) -> Self {
        let auth = dialect
            .alternate_auth
            .map_or(dialect.quirks.auth, |alternative| alternative.auth);
        Self {
            auth,
            ..Self::with_key(dialect, api_key)
        }
    }

    /// The credential `dialect` reads from the environment, and how it is
    /// sent. Alternative credentials require their own header policy; the
    /// primary is preferred.
    pub(crate) fn credential_from_env(dialect: &Dialect) -> Result<(String, Auth), EnvError> {
        let quirks = &dialect.quirks;
        Ok(match dialect.alternate_auth {
            Some(alternative) => match env::optional(dialect.api_key_env)? {
                Some(api_key) => (api_key, quirks.auth),
                None => match env::optional(alternative.api_key_env)? {
                    Some(api_key) => (api_key, alternative.auth),
                    None => {
                        return Err(EnvError::Invalid {
                            name: dialect.api_key_env,
                            detail: format!(
                                "either `{}` or `{}` must be set",
                                dialect.api_key_env, alternative.api_key_env
                            ),
                        });
                    }
                },
            },
            None => (env::required(dialect.api_key_env)?, quirks.auth),
        })
    }

    /// Send the credential with `auth`'s header.
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Name the account the credential belongs to (`ChatGPT-Account-Id`).
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Apply the dialect's authentication to a request builder.
    pub(crate) fn authenticate(&self, builder: http::request::Builder) -> http::request::Builder {
        match self.auth {
            Auth::Bearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::OptionalBearer if self.api_key.is_empty() => builder,
            Auth::OptionalBearer => {
                builder.header("Authorization", format!("Bearer {}", self.api_key.expose()))
            }
            Auth::ApiKeyHeader => builder.header("api-key", self.api_key.expose()),
        }
    }

    /// Stamp the configured caller identity, `originator`, `user-agent` and,
    /// when it names one, `version`; with none configured, leave the request
    /// as is.
    ///
    /// The one source of caller identity for every transport: the HTTP
    /// requests and the websocket handshakes all stamp it here. A dialect
    /// that requires the caller's exact identity ([`Quirks::identity`])
    /// refuses here, before anything is sent, when there is none or only an
    /// originator.
    pub(crate) fn identify(
        &self,
        builder: http::request::Builder,
    ) -> Result<http::request::Builder, MissingCallerIdentity> {
        Ok(match self.sendable_identity()? {
            Some(ConfiguredIdentity::Exact(identity)) => identity.stamp(builder),
            Some(ConfiguredIdentity::Originator { originator }) => builder
                .header(ORIGINATOR_HEADER, originator)
                .header(http::header::USER_AGENT, default_user_agent(originator)),
            None => builder,
        })
    }

    /// The identity this configuration may send with, refusing a dialect
    /// that requires the caller's exact identity when there is none or only
    /// an originator. The one rule every send checks: [`Self::identify`]
    /// before stamping a request, and a websocket session before writing a
    /// frame, whoever opened its connection.
    pub(crate) fn sendable_identity(
        &self,
    ) -> Result<Option<&ConfiguredIdentity>, MissingCallerIdentity> {
        let required = self.dialect.quirks.identity.is_some();
        match &self.identity {
            Some(ConfiguredIdentity::Originator { originator }) if required => {
                Err(MissingCallerIdentity::OriginatorOnly {
                    dialect: self.dialect.name,
                    originator: originator.clone(),
                })
            }
            None if required => Err(MissingCallerIdentity::Unset {
                dialect: self.dialect.name,
            }),
            identity => Ok(identity.as_ref()),
        }
    }

    /// Apply authentication, configured identity, account, and per-request
    /// session headers. Refuses as [`Self::identify`] does.
    pub(crate) fn headers(
        &self,
        builder: http::request::Builder,
    ) -> Result<http::request::Builder, MissingCallerIdentity> {
        let mut builder = self.identify(self.authenticate(builder))?;
        if self
            .dialect
            .quirks
            .identity
            .is_some_and(|identity| identity.session_ids)
        {
            // Session identity must be fresh for each request.
            builder = builder.header("session_id", crate::providers::chatgpt::session_id());
        }
        if let Some(account_id) = &self.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }
        Ok(builder)
    }

    /// Read the credential from `source` each time a request is sent,
    /// instead of sending [`Self::api_key`].
    ///
    /// The source is read once per send attempt (every request, every page
    /// of a paged reply, every managed websocket connect), so a credential
    /// rotated by its owner reaches the next request without rebuilding
    /// anything. Its credential replaces the header [`Self::auth`] names
    /// (`Authorization: Bearer …` or `api-key`), and it is authoritative for
    /// the account too: a credential naming an account sends
    /// `ChatGPT-Account-Id`, and one naming none sends no account header,
    /// even when [`Self::account_id`] is set. A source that fails refuses the
    /// request with [`crate::wire::CredentialUnavailable`] before anything is
    /// sent.
    pub fn with_credential_source(mut self, source: impl CredentialSource + 'static) -> Self {
        self.credential_source = Some(CredentialSourceHandle::new(source));
        self
    }

    /// This configuration's send-time credential: `None` unless a
    /// [credential source](Self::with_credential_source) is set. Its token
    /// goes in the header [`Self::auth`] names, the one
    /// [`Self::authenticate`] writes, and it is authoritative for
    /// `ChatGPT-Account-Id`.
    pub(crate) fn credential_stamp(&self) -> Option<CredentialStamp> {
        self.credential_source
            .as_ref()
            .map(|source| CredentialStamp {
                source: source.clone(),
                placement: CredentialPlacement {
                    token: match self.auth {
                        Auth::Bearer => TokenPlacement::Bearer,
                        Auth::OptionalBearer => TokenPlacement::OptionalBearer,
                        Auth::ApiKeyHeader => {
                            TokenPlacement::Header(http::HeaderName::from_static(API_KEY_HEADER))
                        }
                    },
                    account: Some(http::HeaderName::from_static(ACCOUNT_ID_HEADER)),
                },
            })
    }

    /// Send `originator` as the `originator` header, with a `rig/…`
    /// `user-agent` naming it.
    ///
    /// A dialect whose gateway asks for no caller identity gains one, so
    /// both headers are sent. A dialect that requires the caller's exact
    /// identity ([`Quirks::identity`]) refuses every request with
    /// [`MissingCallerIdentity::OriginatorOnly`] instead: use
    /// [`Self::with_caller_identity`] there.
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.identity = Some(ConfiguredIdentity::Originator {
            originator: originator.into(),
        });
        self
    }

    /// Send exactly `identity` as the caller identity: its `originator`,
    /// its `user-agent` as given (no default is derived), and its `version`
    /// header when it names one. Every transport that stamps the caller
    /// identity carries it: each HTTP request and the Codex websocket
    /// handshake.
    ///
    /// Replaces any identity the dialect's environment variables or
    /// [`Self::with_originator`] configured. A dialect whose gateway asks for
    /// no caller identity gains this one; a dialect that requires one
    /// ([`Quirks::identity`]) sends nothing without it.
    /// [`CallerIdentity::new`] has already refused any value a request could
    /// not carry.
    #[must_use]
    pub fn with_caller_identity(mut self, identity: CallerIdentity) -> Self {
        self.identity = Some(ConfiguredIdentity::Exact(identity));
        self
    }
}

/// The header `Auth::ApiKeyHeader` sends the credential in.
const API_KEY_HEADER: &str = "api-key";
/// The header naming the account a credential belongs to.
const ACCOUNT_ID_HEADER: &str = "chatgpt-account-id";

#[cfg(test)]
mod carried_tests;
