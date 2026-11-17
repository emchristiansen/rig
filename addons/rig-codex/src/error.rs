//! Evidence kept of a frame the Codex session could not decode.
//!
//! A frame that is invalid JSON, or that names an event this crate models but
//! fails its typed decoding, is a [`CorruptFrame`]: the frame exactly as
//! received, beside the decode error. A session read returns it in
//! [`CodexSessionError::CorruptFrame`], and [`CorruptFrameDetail`] is its
//! serializable form. Converting to a [`ProviderError`] keeps the frame:
//! [`ProviderErrorExt::corrupt_frame`] reads it back.
//!
//! ```
//! use rig_codex::error::{CodexSessionError, FrameEvidence};
//!
//! fn record(error: &CodexSessionError) {
//!     if let Some(corrupt) = error.corrupt_frame() {
//!         match corrupt.evidence() {
//!             FrameEvidence::Text(frame) => eprintln!("exact frame: {frame}"),
//!             FrameEvidence::Bytes(bytes) => eprintln!("exact bytes: {bytes:?}"),
//!             FrameEvidence::Decoded(text) => eprintln!("replacement-decoded: {text}"),
//!             FrameEvidence::Absent => eprintln!("no frame: {}", corrupt.error()),
//!         }
//!     }
//! }
//! ```

use rig_core::error::ProviderError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A frame the provider sent that this crate could not decode, with what was
/// kept of it, labelled by how faithful it is.
///
/// It is invalid JSON, or a frame whose discriminator this crate models but
/// whose payload failed typed decoding. A malformed modeled frame is never
/// demoted to an unmodeled one.
#[derive(Debug)]
pub struct CorruptFrame {
    event_type: Option<String>,
    evidence: FrameEvidence,
    error: serde_json::Error,
}

/// What a [`CorruptFrame`] kept of the frame, and how faithful it is.
///
/// Only [`Self::Text`] and [`Self::Bytes`] are the frame exactly as received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "frame", rename_all = "snake_case")]
pub enum FrameEvidence {
    /// The frame's text, exactly as received.
    Text(String),
    /// The frame's bytes, exactly as received. They are not valid UTF-8, so
    /// there is no exact text.
    Bytes(Vec<u8>),
    /// Text decoded from bytes that were not all valid UTF-8, each invalid
    /// sequence replaced with U+FFFD. Not the frame as received; the original
    /// bytes were not kept.
    Decoded(String),
    /// No frame: the transport delivered an already-decoded event.
    Absent,
}

impl CorruptFrame {
    /// A text frame that failed to decode, with the discriminator value read
    /// from it, if any.
    pub fn text(
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
    pub fn bytes(bytes: Vec<u8>, error: serde_json::Error) -> Self {
        Self {
            event_type: None,
            evidence: FrameEvidence::Bytes(bytes),
            error,
        }
    }

    /// A decode failure with no frame: an event that was handed over already
    /// parsed, so there is no text as received to keep.
    pub fn absent(event_type: Option<String>, error: serde_json::Error) -> Self {
        Self {
            event_type,
            evidence: FrameEvidence::Absent,
            error,
        }
    }

    /// This failure without its frame text, for a frame that was read from an
    /// already-parsed event rather than from the text as received.
    pub(crate) fn without_frame(self) -> Self {
        Self {
            evidence: FrameEvidence::Absent,
            ..self
        }
    }

    /// The frame's discriminator value (for a Responses frame, its `type`),
    /// when a string one was read.
    pub fn event_type(&self) -> Option<&str> {
        self.event_type.as_deref()
    }

    /// What was kept of the frame, and how faithful it is.
    pub fn evidence(&self) -> &FrameEvidence {
        &self.evidence
    }

    /// The frame's text exactly as received. `None` when there is no exact
    /// text: see [`Self::evidence`].
    pub fn frame(&self) -> Option<&str> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The frame's bytes exactly as received, text or not. `None` when they
    /// were not kept: see [`Self::evidence`].
    pub fn frame_bytes(&self) -> Option<&[u8]> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text.as_bytes()),
            FrameEvidence::Bytes(bytes) => Some(bytes),
            FrameEvidence::Decoded(_) | FrameEvidence::Absent => None,
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
            FrameEvidence::Bytes(bytes) => {
                write!(f, "; frame bytes (not UTF-8): {}", bytes.escape_ascii())
            }
            FrameEvidence::Decoded(text) => {
                write!(f, "; frame decoded with U+FFFD replacement: {text}")
            }
            FrameEvidence::Absent => Ok(()),
        }
    }
}

impl std::error::Error for CorruptFrame {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<CorruptFrame> for ProviderError {
    fn from(corrupt: CorruptFrame) -> Self {
        ProviderError::request(corrupt)
    }
}

/// Why a Codex websocket session read failed.
///
/// A frame that failed to decode is kept apart from every other failure, so
/// the frame as received reaches the caller in the type rather than through a
/// conversion.
#[derive(Debug, thiserror::Error)]
pub enum CodexSessionError {
    /// A frame that could not be decoded, with what was kept of it.
    #[error("CorruptFrameError: {0}")]
    CorruptFrame(#[source] CorruptFrame),
    /// Any other failure: transport, protocol, timeout, or a refused call.
    #[error(transparent)]
    Provider(ProviderError),
}

impl CodexSessionError {
    /// The undecodable frame and what was kept of it, when this error is one.
    #[must_use]
    pub fn corrupt_frame(&self) -> Option<&CorruptFrame> {
        match self {
            Self::CorruptFrame(corrupt) => Some(corrupt),
            Self::Provider(error) => error.corrupt_frame(),
        }
    }
}

impl From<CorruptFrame> for CodexSessionError {
    fn from(corrupt: CorruptFrame) -> Self {
        Self::CorruptFrame(corrupt)
    }
}

impl From<ProviderError> for CodexSessionError {
    fn from(error: ProviderError) -> Self {
        Self::Provider(error)
    }
}

/// The frame stays readable through [`ProviderErrorExt::corrupt_frame`].
impl From<CodexSessionError> for ProviderError {
    fn from(error: CodexSessionError) -> Self {
        match error {
            CodexSessionError::CorruptFrame(corrupt) => corrupt.into(),
            CodexSessionError::Provider(error) => error,
        }
    }
}

/// A [`CorruptFrame`] in wire form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorruptFrameDetail {
    /// The frame's discriminator value (for a Responses frame, its `type`),
    /// when a string one was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_type: Option<String>,
    /// What was kept of the frame, and how faithful it is.
    pub evidence: FrameEvidence,
    /// Why the frame failed to decode.
    pub error: String,
}

impl From<&CorruptFrame> for CorruptFrameDetail {
    fn from(corrupt: &CorruptFrame) -> Self {
        Self {
            event_type: corrupt.event_type.clone(),
            evidence: corrupt.evidence.clone(),
            error: corrupt.error.to_string(),
        }
    }
}

/// Reads this crate's typed failures back out of a [`ProviderError`].
pub trait ProviderErrorExt {
    /// The undecodable frame and what was kept of it, when this error is one.
    fn corrupt_frame(&self) -> Option<&CorruptFrame>;
}

impl ProviderErrorExt for ProviderError {
    fn corrupt_frame(&self) -> Option<&CorruptFrame> {
        match self {
            Self::Request(reason) => reason.downcast_ref::<CorruptFrame>(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
