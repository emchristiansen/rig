//! Transport and credential I/O for a GPT-Live call. The provider builds
//! requests and parses identifiers; this driver sends one call-creation
//! exchange and keeps its transport errors intact.

use bytes::Bytes;

use crate::error::ProviderError;
use crate::http_client::{self, HttpClientExt, LazyBody};
use crate::providers::chatgpt::realtime::{
    LiveCalls, RealtimeCall, SessionConfig, decode_created_call,
};

impl LiveCalls {
    /// Create a call for `offer_sdp` configured by `session`, over `http`.
    ///
    /// Returns the answer SDP and the call id read from `Location`. A
    /// non-success reply is [`ProviderError::ProviderResponse`] keeping its
    /// status, body, headers and request id. A success without a `Location`
    /// call id, or with a body that is not UTF-8, is
    /// [`ProviderError::Response`].
    pub async fn create_call<H: HttpClientExt>(
        &self,
        http: &H,
        offer_sdp: &str,
        session: &SessionConfig,
    ) -> Result<RealtimeCall, ProviderError> {
        let mut request = self.call_request(offer_sdp, session)?;
        if let Some(stamp) = self.credential_stamp() {
            stamp.authorize(request.headers_mut()).await?;
        }
        let request_id_header = self.request_id_header();
        let response = http
            .send::<_, Bytes>(request)
            .await
            .map_err(|error| reply_error(error, request_id_header))?;
        let (parts, body) = response.into_parts();
        let body: LazyBody<Bytes> = body;
        let body = body
            .await
            .map_err(|error| reply_error(error, request_id_header))?;
        if !parts.status.is_success() {
            return Err(reply_error(
                http_client::Error::non_success_with_details(
                    parts.status,
                    parts.headers,
                    String::from_utf8_lossy(&body).into_owned(),
                ),
                request_id_header,
            ));
        }
        decode_created_call(&parts.headers, &body)
    }
}

/// A transport failure, keeping a rejected reply's status, body, headers and
/// request id.
pub(crate) fn reply_error(
    error: http_client::Error,
    request_id_header: Option<&'static str>,
) -> ProviderError {
    let request_id = error.non_success_headers().and_then(|headers| {
        crate::providers::internal::request_id_from_headers(headers, request_id_header)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(request_id)
}
