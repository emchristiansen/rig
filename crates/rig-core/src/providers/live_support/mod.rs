//! Paths of the 0.41 Live support module, kept for consumers written against it.
//!
//! Every item here is the 0.42 type itself, re-exported: [`error`] names
//! [`crate::error`]'s types and [`CallerIdentity`] is
//! [`openai::wire::CallerIdentity`](crate::providers::openai::wire::CallerIdentity).
//! There is no second error or identity hierarchy, so a value obtained through
//! either path is the same type. Configuration is the
//! [`OpenAI`](crate::providers::openai::wire::OpenAI) provider value, which
//! [`LiveCalls::new`](crate::providers::chatgpt::realtime::LiveCalls::new) and
//! [`PublicLiveSessions::new`](crate::providers::openai::live::PublicLiveSessions::new)
//! take, and their `create_call` and `create_session` send the creation
//! exchange.

/// The 0.42 provider error types, under their 0.41 Live support path.
pub mod error {
    pub use crate::ProviderResponseError;
    pub use crate::error::{
        BoxError, CorruptFrame, CorruptFrameDetail, EncodeError, ErrorDetail, ErrorKind,
        ErrorReport, FrameEvidence, MalformedToolInput, ProviderError, retryable_status,
        transient_transport,
    };
}

pub use crate::error::{CorruptFrame, EncodeError, FrameEvidence};
pub use crate::providers::openai::responses_api::codex_identity::{
    CodexIdentity, InvalidCodexIdentity,
};
pub use crate::providers::openai::wire::{CallerIdentity, InvalidCallerIdentity};

