//! Already-resolved access and explicit protocol selection; no acquisition or refresh.
use super::error::EncodeError;
use super::identity::CallerIdentity;

/// The protocol this resolved configuration is allowed to address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveBackend {
    /// The public OpenAI Live API.
    Public,
    /// The ChatGPT subscription Live API.
    Subscription,
}
impl LiveBackend {
    /// A stable protocol name, independent of the chosen URL.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Public => "openai",
            Self::Subscription => "chatgpt",
        }
    }
}

/// Static access resolved by the caller at the point of use.
///
/// This value never discovers, refreshes or reads credentials. Subscription
/// requests require an exact validated caller identity before they can be built.
#[derive(Clone, Debug)]
pub struct LiveConfiguration {
    pub(crate) backend: LiveBackend,
    pub(crate) base_url: String,
    access_token: String,
    pub(crate) account_id: Option<String>,
    caller_identity: Option<CallerIdentity>,
}
impl LiveConfiguration {
    /// Public API access with the standard OpenAI base URL.
    pub fn public(access_token: impl Into<String>) -> Self {
        Self {
            backend: LiveBackend::Public,
            base_url: "https://api.openai.com/v1".into(),
            access_token: access_token.into(),
            account_id: None,
            caller_identity: None,
        }
    }
    /// Subscription access with the standard ChatGPT Codex base URL.
    pub fn subscription(access_token: impl Into<String>) -> Self {
        Self {
            backend: LiveBackend::Subscription,
            base_url: "https://chatgpt.com/backend-api/codex".into(),
            access_token: access_token.into(),
            account_id: None,
            caller_identity: None,
        }
    }
    /// Select a custom base URL without changing the protocol discriminator.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
    /// Supply the account to send with subscription requests.
    #[must_use]
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }
    /// Supply the caller's exact validated originator, user agent and optional version.
    #[must_use]
    pub fn with_caller_identity(mut self, identity: CallerIdentity) -> Self {
        self.caller_identity = Some(identity);
        self
    }
    /// The explicitly selected protocol.
    pub fn backend(&self) -> LiveBackend {
        self.backend
    }
    /// The configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    /// The supplied caller identity, if any.
    pub fn caller_identity(&self) -> Option<&CallerIdentity> {
        self.caller_identity.as_ref()
    }
    pub(crate) fn authenticate(&self, builder: http::request::Builder) -> http::request::Builder {
        builder.header(
            http::header::AUTHORIZATION,
            format!("Bearer {}", self.access_token),
        )
    }
    pub(crate) fn identify(
        &self,
        builder: http::request::Builder,
    ) -> Result<http::request::Builder, EncodeError> {
        match &self.caller_identity {
            Some(identity) => Ok(identity.stamp(builder)),
            None if self.backend == LiveBackend::Subscription => {
                Err(EncodeError::request(MissingCallerIdentity))
            }
            None => Ok(builder),
        }
    }
}
/// Subscription access cannot build a request without the caller's exact identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "the subscription Live request requires the caller's exact originator and user agent; nothing was sent"
)]
pub struct MissingCallerIdentity;

#[cfg(test)]
mod tests;
