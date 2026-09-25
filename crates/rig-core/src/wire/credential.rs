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

/// Stamps a request's credential headers at send time, after encoding.
///
/// The driver awaits it once per send attempt, before the request reaches
/// the transport, so whatever it writes is seen by the transport, and by
/// any recording or scrubbing there, exactly as a credential written while
/// encoding would be.
pub trait Authorizer: WasmCompatSend + WasmCompatSync {
    /// Write the current credential headers into `headers`. An error refuses
    /// the request before anything is sent.
    fn authorize<'a>(
        &'a self,
        headers: &'a mut http::HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<(), ProviderError>>;
}
