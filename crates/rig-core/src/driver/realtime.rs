//! Transport and credential I/O for a GPT-Live call and a public Live
//! session. The provider builds requests and parses identifiers; this driver
//! sends one creation exchange and keeps its transport errors intact.

use bytes::Bytes;

use crate::error::ProviderError;
use crate::http_client::{self, HttpClientExt, LazyBody};
use crate::providers::chatgpt::realtime::{
    LiveCalls, RealtimeCall, SessionConfig, decode_created_call,
};
use crate::providers::openai::live::{
    CreatedSession, LiveApiError, PublicLiveSessions, ReplyBody,
    SessionConfig as PublicSessionConfig,
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

impl PublicLiveSessions {
    /// Create a public Live session for `offer_sdp` configured by `session`,
    /// over `http`.
    ///
    /// Reads the provider's credential source, when it has one, before
    /// sending; a failure there is [`LiveApiError::Request`] and nothing is
    /// sent. A send that fails before any reply is [`LiveApiError::Transport`].
    /// Every received reply keeps its status, headers and request id: a
    /// rejected credential, a reached spend limit and any other refusal stay
    /// distinct whether the transport returned the reply or reported it as an
    /// error, and a success whose body cannot be read is
    /// [`LiveApiError::OutcomeUnknown`].
    pub async fn create_session<H: HttpClientExt>(
        &self,
        http: &H,
        offer_sdp: &str,
        session: &PublicSessionConfig,
    ) -> Result<CreatedSession, LiveApiError> {
        let mut request = self
            .create_request(offer_sdp, session)
            .map_err(|error| LiveApiError::Request(error.into()))?;
        if let Some(stamp) = self.credential_stamp() {
            stamp
                .authorize(request.headers_mut())
                .await
                .map_err(LiveApiError::Request)?;
        }
        let response = match http.send::<_, Bytes>(request).await {
            Ok(response) => response,
            Err(http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            }) => {
                return self.decode(
                    self.reply_head(status, headers),
                    ReplyBody::TransportText(body),
                );
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
            Err(http_client::Error::InvalidStatusCodeWithDetails { body, .. }) => {
                ReplyBody::TransportText(body)
            }
            Err(other) => ReplyBody::Unreadable(ProviderError::from_transport_error(other)),
        };
        self.decode(head, body)
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

#[cfg(test)]
mod tests;
