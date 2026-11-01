//! Compression of a Responses wire's HTTP request bodies.
//!
//! Some gateways expect a compressed request body: the Codex backend's own
//! client sends its `/responses` bodies zstd-compressed. A [`Responses`] wire
//! given [`RequestCompression::Zstd`]
//! ([`Responses::with_request_compression`]) compresses every HTTP request
//! body it sends and names the encoding in `Content-Encoding`. Websocket
//! frames are never compressed this way.
//!
//! [`RequestCompression::Zstd`] exists only in builds that can compress: the
//! `request-compression` feature on a native target.
//!
//! [`Responses`]: super::wire::Responses
//! [`Responses::with_request_compression`]: super::wire::Responses::with_request_compression

use serde::{Deserialize, Serialize};

use crate::completion::options::FinalBody;
use crate::error::EncodeError;
use crate::wire::Body;

/// How a Responses wire encodes its HTTP request bodies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCompression {
    /// Send the body as it is (the default).
    #[default]
    None,
    /// Compress the body with zstd at level 3 and send
    /// `Content-Encoding: zstd`, as the Codex backend's own client does.
    #[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
    Zstd,
}

pub(super) enum BodyOrder {
    Original,
    RoutingFirst,
}

impl RequestCompression {
    /// Whether bodies are sent as they are.
    #[must_use]
    pub fn is_none(&self) -> bool {
        *self == Self::None
    }

    /// Encode the sealed `body`, naming the encoding on `builder`.
    pub(super) fn apply(
        self,
        builder: http::request::Builder,
        body: FinalBody,
        order: BodyOrder,
    ) -> Result<(http::request::Builder, EncodedBody), EncodeError> {
        let body = match order {
            BodyOrder::Original => serde_json::to_vec(&body)?,
            BodyOrder::RoutingFirst => {
                serde_json::to_vec(&super::codex_request::RoutingFirst(&body))?
            }
        };
        match self {
            Self::None => Ok((builder, EncodedBody(body))),
            #[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
            Self::Zstd => {
                let compressed =
                    zstd::stream::encode_all(body.as_slice(), ZSTD_LEVEL).map_err(|error| {
                        EncodeError::request(format!(
                            "Failed to zstd-compress the request body: {error}"
                        ))
                    })?;
                Ok((
                    builder.header(http::header::CONTENT_ENCODING, "zstd"),
                    EncodedBody(compressed),
                ))
            }
        }
    }
}

/// Transport bytes encoded from a sealed Responses request body.
pub(super) struct EncodedBody(Vec<u8>);

impl EncodedBody {
    /// Consume the encoded body as HTTP request bytes.
    pub(super) fn into_body(self) -> Body {
        Body::Bytes(self.0)
    }
}

/// The zstd level the Codex backend's own client compresses at.
#[cfg(all(feature = "request-compression", not(target_family = "wasm")))]
const ZSTD_LEVEL: i32 = 3;

#[cfg(all(test, feature = "request-compression", not(target_family = "wasm")))]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
