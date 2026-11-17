//! What Live calls and sessions share beyond their protocols: the caller and
//! conversation identities, the caller's extra headers, the send-time
//! credential source and the error types.
//!
//! [`error`] names `rig-core`'s provider error types next to this crate's
//! [`CorruptFrame`]. The [`OpenAI`](rig_core::providers::openai::OpenAI)
//! client is what
//! [`LiveCalls::new`](crate::chatgpt::LiveCalls::new) and
//! [`PublicLiveSessions::new`](crate::openai::PublicLiveSessions::new) take,
//! and their `create_call` and `create_session` send the creation exchange.

mod codex_identity;
mod credential;
pub mod error;
mod identity;
mod request_headers;

pub use codex_identity::{
    CodexIdentity, InvalidCodexIdentity, NotACodexWire, SESSION_ID_HEADER, THREAD_ID_HEADER,
    X_CLIENT_REQUEST_ID_HEADER,
};
pub use credential::{
    Credential, CredentialPlacement, CredentialSource, CredentialSourceError,
    CredentialSourceHandle, CredentialStamp, CredentialUnavailable, TokenPlacement,
};
pub use error::{CorruptFrame, EncodeError, FrameEvidence};
pub use identity::{CallerIdentity, InvalidCallerIdentity, MissingCallerIdentity, VERSION_HEADER};
pub use request_headers::{InvalidRequestHeader, RequestHeaders};
