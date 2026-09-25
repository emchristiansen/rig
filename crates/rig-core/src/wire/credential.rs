//! Credentials read at send time, from a source the caller owns.
//!
//! A [`CredentialSource`] hands the wire the current credential each time it
//! is about to send, so a credential rotated elsewhere reaches the next
//! request without rebuilding the client. The source is read-only by
//! construction: nothing here can ask it to refresh or rotate.
//!
//! ```
//! use rig_core::wasm_compat::WasmBoxedFuture;
//! use rig_core::wire::{Credential, CredentialSource, CredentialSourceError};
//!
//! struct Fixed;
//!
//! impl CredentialSource for Fixed {
//!     fn current(&self) -> WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>> {
//!         Box::pin(async { Ok(Credential::new("access-token")) })
//!     }
//! }
//! ```

use std::sync::Arc;

use crate::error::ProviderError;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

use super::Secret;

/// What a [`CredentialSource`] reports when it cannot supply a credential:
/// the crate's boxed error, which is `Send + Sync` except on WASM, so a
/// browser-local source can return a browser-local error whole.
pub type CredentialSourceError = crate::error::BoxError;

/// The credential a request is sent with: the token, and the account it
/// belongs to when the gateway asks which (`ChatGPT-Account-Id`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Credential {
    /// The token itself. Its `Debug` form is redacted (see [`Secret`]).
    pub token: Secret,
    /// The account the token belongs to. With a source set, this is
    /// authoritative: `None` sends no account header, even when the
    /// configuration names one.
    pub account_id: Option<String>,
}

impl Credential {
    /// A credential for `token`, naming no account.
    pub fn new(token: impl Into<Secret>) -> Self {
        Self {
            token: token.into(),
            account_id: None,
        }
    }

    /// Name the account `token` belongs to.
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }
}

/// A read-only supplier of the current credential, owned by someone else.
///
/// Read once per send attempt: every HTTP request, every page of a paged
/// reply, and every websocket connect. An implementor must therefore be
/// cheap, must not itself make a provider round trip, and must never rotate
/// the credential: the owner keeps refresh authority, and the request path is
/// never the rotator.
pub trait CredentialSource: WasmCompatSend + WasmCompatSync {
    /// The current credential.
    fn current(&self) -> WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>>;
}

/// A shared [`CredentialSource`] held by a provider configuration.
///
/// Clones share the source. Two handles are equal only when they share it.
/// `Debug` names the handle without reading the source.
#[derive(Clone)]
pub struct CredentialSourceHandle(pub Arc<dyn CredentialSource>);

impl CredentialSourceHandle {
    /// Share `source`.
    pub fn new(source: impl CredentialSource + 'static) -> Self {
        Self(Arc::new(source))
    }
}

impl std::fmt::Debug for CredentialSourceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CredentialSourceHandle")
    }
}

impl PartialEq for CredentialSourceHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A request refused because its [`CredentialSource`] could not supply a
/// credential. Nothing was sent. The source's own error is kept whole, as
/// this error's [`source`](std::error::Error::source).
#[derive(Debug)]
pub struct CredentialUnavailable {
    source: CredentialSourceError,
}

impl CredentialUnavailable {
    /// Refuse because `source` failed.
    pub fn new(source: CredentialSourceError) -> Self {
        Self { source }
    }
}

impl std::fmt::Display for CredentialUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the credential source could not supply a credential, so nothing was sent: {}",
            self.source
        )
    }
}

impl std::error::Error for CredentialUnavailable {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

impl From<CredentialUnavailable> for ProviderError {
    fn from(refusal: CredentialUnavailable) -> Self {
        ProviderError::Request(Box::new(refusal))
    }
}

/// Where a credential's token goes in a request's headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenPlacement {
    /// `Authorization: Bearer <token>`.
    Bearer,
    /// `Authorization: Bearer <token>`, or no credential header at all when
    /// the token is empty.
    OptionalBearer,
    /// `<name>: <token>`, the token alone (Azure's `api-key`, for example).
    Header(http::HeaderName),
}

/// How a send-time credential is written into a request's headers: where the
/// token goes, and which header, if any, names its account.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialPlacement {
    /// Where the token goes.
    pub token: TokenPlacement,
    /// The header naming the credential's account. With one set, the
    /// credential is authoritative for it: an account is sent when the
    /// credential names one, and the header is removed when it names none.
    pub account: Option<http::HeaderName>,
}

impl CredentialPlacement {
    /// Write `credential` into `headers`, replacing the credential and account
    /// headers encoding wrote. A pure function of its inputs, shared by every
    /// transport that authorizes at send time, so they cannot drift.
    ///
    /// A token or account no header can carry refuses the request; the
    /// refusal names the problem, never the value.
    pub fn apply(
        &self,
        headers: &mut http::HeaderMap,
        credential: &Credential,
    ) -> Result<(), ProviderError> {
        let token = credential.token.expose();
        headers.remove(http::header::AUTHORIZATION);
        match &self.token {
            TokenPlacement::OptionalBearer if token.is_empty() => {}
            TokenPlacement::Bearer | TokenPlacement::OptionalBearer => {
                headers.insert(
                    http::header::AUTHORIZATION,
                    header_value(&format!("Bearer {token}"))?,
                );
            }
            TokenPlacement::Header(name) => {
                headers.remove(name);
                headers.insert(name.clone(), header_value(token)?);
            }
        }
        if let Some(account) = &self.account {
            headers.remove(account);
            if let Some(account_id) = &credential.account_id {
                headers.insert(account.clone(), header_value(account_id)?);
            }
        }
        Ok(())
    }
}

/// A wire's send-time credential: the source it is read from and where it
/// goes. Data only; the driver and the websocket handshake read and apply it
/// through [`Self::authorize`].
#[derive(Clone, Debug, PartialEq)]
pub struct CredentialStamp {
    /// The source read at send time.
    pub source: CredentialSourceHandle,
    /// Where its credential is written.
    pub placement: CredentialPlacement,
}

impl CredentialStamp {
    /// Read the source once and write its credential into `headers`. A
    /// failing source refuses with [`CredentialUnavailable`] before anything
    /// is written, so nothing is sent.
    ///
    /// Called once per send attempt, after encoding and before the request
    /// reaches the transport, so whatever it writes is seen by the transport,
    /// and by any recording or scrubbing there, exactly as a credential
    /// written while encoding would be.
    pub async fn authorize(&self, headers: &mut http::HeaderMap) -> Result<(), ProviderError> {
        let credential = self
            .source
            .0
            .current()
            .await
            .map_err(CredentialUnavailable::new)?;
        self.placement.apply(headers, &credential)
    }
}

/// A credential header value, refusing one no header can carry. The refusal
/// names the problem, never the value.
fn header_value(value: &str) -> Result<http::HeaderValue, ProviderError> {
    http::HeaderValue::from_str(value).map_err(|_| {
        ProviderError::Request(
            "the credential source supplied a value that cannot be sent in a header".into(),
        )
    })
}

#[cfg(test)]
mod tests;
