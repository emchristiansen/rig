//! Static configuration and evidence-preserving support for Live protocols.
pub mod identity;
pub use identity::{CallerIdentity, CodexIdentity, InvalidCallerIdentity, InvalidCodexIdentity};
#[cfg(feature = "live")]
pub mod configuration;
pub mod error;
#[cfg(feature = "live")]
pub(crate) mod exchange;
#[cfg(feature = "live")]
pub use configuration::{LiveBackend, LiveConfiguration, MissingCallerIdentity};
pub use error::{CorruptFrame, EncodeError, FrameEvidence, LiveProviderError, LiveResponseError};
