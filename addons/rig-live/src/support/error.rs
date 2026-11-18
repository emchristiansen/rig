//! The provider error types Live calls and sessions report, and the corrupt
//! frame a server event that fails to decode becomes.
//!
//! ```
//! use rig_live::support::error::{CorruptFrame, FrameEvidence};
//!
//! fn record(corrupt: &CorruptFrame) {
//!     match corrupt.evidence() {
//!         FrameEvidence::Text(frame) => eprintln!("exact frame: {frame}"),
//!         FrameEvidence::Bytes(bytes) => eprintln!("exact bytes: {bytes:?}"),
//!     }
//! }
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

pub use rig_core::ProviderResponseError;
pub use rig_core::error::{
    BoxError, EncodeError, ErrorKind, ErrorReport, ProviderError, retryable_status,
    transient_transport,
};

/// A server event that failed to decode, with what was kept of it.
///
/// It is invalid JSON, or an event whose `type` this crate models but whose
/// fields failed typed decoding. A malformed modelled event is never demoted
/// to an unmodelled one; it surfaces as this error, carrying the frame so a
/// caller can keep it.
///
/// There is deliberately no conversion into [`ProviderError`]: it has no
/// variant that could carry the frame, so a caller keeps this type, or
/// [`ControlError`](crate::chatgpt::ControlError) from a control socket.
#[derive(Debug)]
pub struct CorruptFrame {
    event_type: Option<String>,
    evidence: FrameEvidence,
    error: serde_json::Error,
}

/// What a [`CorruptFrame`] kept of the frame. Both forms are the frame
/// exactly as received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "frame", rename_all = "snake_case")]
pub enum FrameEvidence {
    /// The frame's text, exactly as received.
    Text(String),
    /// The frame's bytes, exactly as received, whether or not they are UTF-8.
    Bytes(Vec<u8>),
}

impl CorruptFrame {
    /// A text frame that failed to decode, with the `type` the classifier
    /// read from it, if any.
    pub(crate) fn text(
        event_type: Option<String>,
        frame: impl Into<String>,
        error: serde_json::Error,
    ) -> Self {
        Self {
            event_type,
            evidence: FrameEvidence::Text(frame.into()),
            error,
        }
    }

    /// A byte frame that failed to decode.
    #[cfg(feature = "websocket")]
    pub(crate) fn bytes(bytes: Vec<u8>, error: serde_json::Error) -> Self {
        Self {
            event_type: None,
            evidence: FrameEvidence::Bytes(bytes),
            error,
        }
    }

    /// The frame's `type`, when the classifier read a string one.
    pub fn event_type(&self) -> Option<&str> {
        self.event_type.as_deref()
    }

    /// What was kept of the frame.
    pub fn evidence(&self) -> &FrameEvidence {
        &self.evidence
    }

    /// The frame's text exactly as received. `None` for a byte frame.
    pub fn frame(&self) -> Option<&str> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text),
            FrameEvidence::Bytes(_) => None,
        }
    }

    /// The frame's bytes exactly as received, text or not. Always `Some`:
    /// both kinds of evidence are exact.
    pub fn frame_bytes(&self) -> Option<&[u8]> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text.as_bytes()),
            FrameEvidence::Bytes(bytes) => Some(bytes),
        }
    }

    /// Why the frame failed to decode.
    pub fn error(&self) -> &serde_json::Error {
        &self.error
    }
}

impl fmt::Display for CorruptFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.event_type {
            Some(event_type) => write!(f, "`{event_type}` frame failed to decode: {}", self.error)?,
            None => write!(f, "frame failed to decode: {}", self.error)?,
        }
        match &self.evidence {
            FrameEvidence::Text(text) => write!(f, "; frame: {text}"),
            FrameEvidence::Bytes(bytes) => write!(f, "; frame bytes: {}", bytes.escape_ascii()),
        }
    }
}

impl std::error::Error for CorruptFrame {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
