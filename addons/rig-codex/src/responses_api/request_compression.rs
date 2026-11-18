//! Compression of a Codex wire's HTTP request bodies.
//!
//! The Codex backend's own client sends its `/responses` bodies
//! zstd-compressed. A [`CodexResponses`] wire given
//! [`RequestCompression::Zstd`]
//! ([`CodexResponses::with_request_compression`]) has every HTTP request body
//! compressed and its encoding named in `Content-Encoding`. Websocket frames
//! are never compressed this way.
//!
//! [`RequestCompression::Zstd`] exists only in builds that can compress: the
//! `request-compression` feature on a native target.
//!
//! ```
//! use rig_codex::responses_api::request_compression::RequestCompression;
//!
//! assert!(RequestCompression::default().is_none());
//! ```
//!
//! [`CodexResponses`]: super::wire::CodexResponses
//! [`CodexResponses::with_request_compression`]: super::wire::CodexResponses::with_request_compression

use serde::{Deserialize, Serialize};

/// How a Codex wire's HTTP request bodies are encoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCompression {
    /// Send the body as it is (the default).
    #[default]
    None,
    /// Compress the body with zstd at level 3 and send
    /// `Content-Encoding: zstd`, as the Codex backend's own client does.
    #[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "request-compression")))]
    Zstd,
}

/// A request body that could not be compressed.
#[derive(Debug, thiserror::Error)]
#[error("failed to {encoding}-compress the request body: {source}")]
pub struct RequestCompressionError {
    /// The encoding that failed.
    pub encoding: &'static str,
    /// Why it failed.
    #[source]
    pub source: std::io::Error,
}

impl RequestCompression {
    /// Whether bodies are sent as they are.
    #[must_use]
    pub fn is_none(&self) -> bool {
        *self == Self::None
    }

    /// Encode `body` for the wire. Returns the `Content-Encoding` value that
    /// names the encoding, when there is one, and the bytes to send.
    pub(crate) fn apply(
        self,
        body: Vec<u8>,
    ) -> Result<(Option<&'static str>, Vec<u8>), RequestCompressionError> {
        match self {
            Self::None => Ok((None, body)),
            #[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
            Self::Zstd => zstd::stream::encode_all(body.as_slice(), ZSTD_LEVEL)
                .map(|compressed| (Some(ZSTD_ENCODING), compressed))
                .map_err(|source| RequestCompressionError {
                    encoding: ZSTD_ENCODING,
                    source,
                }),
        }
    }
}

/// The `Content-Encoding` value of a zstd-compressed body.
#[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
const ZSTD_ENCODING: &str = "zstd";

/// The zstd level the Codex backend's own client compresses at.
#[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
const ZSTD_LEVEL: i32 = 3;

#[cfg(all(test, feature = "request-compression", not(target_family = "wasm")))]
mod tests;
