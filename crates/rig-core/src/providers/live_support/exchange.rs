//! One Live creation exchange over the existing HTTP trait on a driven runtime.
use super::error::{ProviderError, request_id_from_headers};
use crate::http_client::{self, HttpClientExt, LazyBody};
use crate::providers::chatgpt::realtime::{
    LiveCalls, RealtimeCall, SessionConfig, decode_created_call,
};
use crate::providers::openai::live::{
    CreatedSession, LiveApiError, PublicLiveSessions, ReplyBody,
    SessionConfig as PublicSessionConfig,
};
use bytes::Bytes;

impl LiveCalls {
    /// Send one creation request with the caller-resolved static access.
    /// The HTTP implementation must be polled on its required driven runtime.
    pub async fn create_call<H: HttpClientExt>(
        &self,
        http: &H,
        offer_sdp: &str,
        session: &SessionConfig,
    ) -> Result<RealtimeCall, ProviderError> {
        let request = self.call_request(offer_sdp, session)?;
        let response = http
            .send::<_, Bytes>(request)
            .await
            .map_err(|e| reply_error(e, self.request_id_header()))?;
        let (parts, body) = response.into_parts();
        let body: LazyBody<Bytes> = body;
        let body = body
            .await
            .map_err(|e| reply_error(e, self.request_id_header()))?;
        if !parts.status.is_success() {
            return Err(reply_error(
                http_client::Error::InvalidStatusCodeWithDetails {
                    status: parts.status,
                    headers: Box::new(parts.headers),
                    body: String::from_utf8_lossy(&body).into_owned(),
                },
                self.request_id_header(),
            ));
        }
        decode_created_call(&parts.headers, &body)
    }
}
impl PublicLiveSessions {
    /// Send one creation request. A received reply preserves its status,
    /// supplied headers, and exact bytes or transport-text provenance.
    /// A successful head followed by an unreadable body is an unknown outcome.
    /// The HTTP implementation must be polled on its required driven runtime.
    pub async fn create_session<H: HttpClientExt>(
        &self,
        http: &H,
        offer_sdp: &str,
        session: &PublicSessionConfig,
    ) -> Result<CreatedSession, LiveApiError> {
        let request = self
            .create_request(offer_sdp, session)
            .map_err(|e| LiveApiError::Request(e.into()))?;
        let response = match http.send::<_, Bytes>(request).await {
            Ok(response) => response,
            Err(http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            }) => {
                return self.decode(
                    self.reply_head(status, *headers),
                    ReplyBody::TransportText(body),
                );
            }
            Err(http_client::Error::InvalidStatusCodeWithMessage(status, body)) => {
                let mut head = self.reply_head(status, http::HeaderMap::new());
                head.headers_available = false;
                return self.decode(head, ReplyBody::TransportText(body));
            }
            Err(http_client::Error::InvalidStatusCode(status)) => {
                let mut head = self.reply_head(status, http::HeaderMap::new());
                head.headers_available = false;
                return self.decode(head, ReplyBody::Unavailable);
            }
            Err(other) => {
                return Err(LiveApiError::Transport(
                    ProviderError::from_transport_error(other),
                ));
            }
        };
        let (parts, body) = response.into_parts();
        let head = self.reply_head(parts.status, parts.headers);
        let body: LazyBody<Bytes> = body;
        let body = match body.await {
            Ok(bytes) => ReplyBody::Received(bytes.to_vec()),
            Err(
                http_client::Error::InvalidStatusCodeWithDetails { body, .. }
                | http_client::Error::InvalidStatusCodeWithMessage(_, body),
            ) => ReplyBody::TransportText(body),
            Err(other) => ReplyBody::Unreadable(ProviderError::from_transport_error(other)),
        };
        self.decode(head, body)
    }
}
pub(crate) fn reply_error(
    error: http_client::Error,
    request_id_header: Option<&'static str>,
) -> ProviderError {
    let request_id = match &error {
        http_client::Error::InvalidStatusCodeWithDetails { headers, .. } => {
            request_id_from_headers(headers, request_id_header)
        }
        _ => None,
    };
    ProviderError::from_transport_error(error).with_provider_request_id(request_id)
}
#[cfg(test)]
mod tests;
