//! ChatGPT access-token configuration and native OAuth authentication.
//!
//! A token refresh is a request on the ChatGPT subscription channel too, so
//! it carries the caller's exact identity, as the official client's does:
//! an [`AuthSource::OAuth`] cannot be built without one. The device sign-in
//! carries none, as the official client's does not.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler};
//! use rig_core::providers::openai::wire::CallerIdentity;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let identity = CallerIdentity::new("my-originator", "my-user-agent/1.0", None)?;
//! let auth = Authenticator::new(
//!     AuthSource::OAuth { identity },
//!     None,
//!     DeviceCodeHandler::default(),
//!     true,
//! );
//! # Ok(())
//! # }
//! ```

use crate::http_client::HttpClientExt;
use crate::providers::openai::wire::CallerIdentity;
use crate::wire::Secret;
use futures::lock::Mutex;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::providers::internal::auth::{DeviceCodeHandler, DeviceCodePrompt};

#[cfg(not(target_family = "wasm"))]
mod native;

#[derive(Clone)]
pub enum AuthSource {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    /// Sign in and refresh through OAuth. A token refresh carries
    /// `identity`'s `originator` and `user-agent` headers exactly, as the
    /// official client's does; a `version` it names is a model request
    /// header and is not sent there.
    OAuth { identity: CallerIdentity },
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
            Self::OAuth { identity } => {
                f.debug_struct("OAuth").field("identity", identity).finish()
            }
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub struct Authenticator {
    source: AuthSource,
    auth_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
    allow_device_flow: bool,
    /// Held across a cache refresh to prevent concurrent updates.
    refresh_lock: Arc<Mutex<()>>,
}

pub use crate::providers::internal::auth::AuthError;

#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Resolved credential. Use [`Secret::expose`] only when raw bytes are required.
    pub access_token: Secret,
    pub account_id: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            auth_file,
            device_code_handler,
            allow_device_flow,
            refresh_lock: Arc::default(),
        }
    }

    /// Resolve the access token and account id, refreshing through `http` as needed.
    /// Return cache, transport, or authorization errors. OAuth is unsupported
    /// on WASM; explicit access tokens remain available.
    #[cfg_attr(target_family = "wasm", allow(unused_variables))]
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone().into(),
                account_id: account_id.clone(),
            }),
            #[cfg(not(target_family = "wasm"))]
            AuthSource::OAuth { identity } => self.auth_context_oauth(http, identity).await,
            #[cfg(target_family = "wasm")]
            AuthSource::OAuth { .. } => Err(AuthError::Message(
                "ChatGPT OAuth is not supported on wasm targets".into(),
            )),
        }
    }
}
