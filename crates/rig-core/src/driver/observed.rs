//! Transport for an observed streamed reply. The request is encoded, its
//! credential read, sent once, refused and content-type checked exactly as
//! [`stream`](super::stream) does; the reply's SSE events are then yielded
//! with their data unchanged, for the caller's ledger, rather than decoded.

use futures::{Stream, StreamExt};

use super::{accept_header, byte_request, reject_response, scope_to_wire, wrong_content_type};
use crate::error::ProviderError;
use crate::http_client::HttpClientExt;
use crate::http_client::framing::{Framing, SseFramer};
use crate::providers::openai::responses_api::observation::ObservationHandle;
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Encoded, Mode, Request, Wire};

/// One item of an observed reply.
pub(crate) enum ObservedSourceEvent {
    /// The reply was accepted as an event stream; its head is recorded.
    Open,
    /// One dispatched SSE event's data, exactly as the framer joined it.
    Message(String),
}

/// Opens one observed streamed reply from exactly one byte-body request.
/// Encoding errors return immediately; the credential read, the send and the
/// reply arrive as stream items. Transport work and the send-attempt mark
/// begin on the first poll.
pub(crate) fn observed_sse<W, H>(
    wire: &W,
    http: &H,
    request: Request<W>,
    observation: ObservationHandle,
) -> Result<
    impl Stream<Item = Result<ObservedSourceEvent, ProviderError>> + WasmCompatSend + 'static,
    ProviderError,
>
where
    W: Wire,
    H: HttpClientExt + Clone + 'static,
{
    let mut request = request;
    scope_to_wire(wire, &mut request);
    let Encoded {
        requests,
        framing,
        request_id_header,
        relaxed_content_type,
        ..
    } = wire.encode(request, Mode::Streaming)?;
    if framing != Framing::Sse {
        return Err(ProviderError::Request(
            "an observed reply must be an event stream".into(),
        ));
    }
    let [http_request] = <[_; 1]>::try_from(requests).map_err(|requests| {
        ProviderError::Request(
            format!(
                "a streamed reply takes exactly one request, not {}",
                requests.len()
            )
            .into(),
        )
    })?;
    let mut http_request = http_request;
    accept_header(&mut http_request, framing);
    let http_request = byte_request(http_request)?;
    let http = http.clone();
    let credential_stamp = wire.credential_stamp();

    Ok(async_stream::stream! {
        let mut http_request = http_request;
        if let Some(stamp) = &credential_stamp
            && let Err(error) = stamp.authorize(http_request.headers_mut()).await
        {
            yield Err(error);
            return;
        }
        observation.mark_send_attempt_started();
        let response = match http.send_streaming(http_request).await {
            Ok(response) => response,
            Err(error) => {
                // A transport that returns a refused reply as an error still
                // delivered its head.
                if let Some(status) = error.non_success_status() {
                    let headers = error.non_success_headers().cloned().unwrap_or_default();
                    observation.record_reply_head(status, &headers, false);
                }
                yield Err(rejected(error, request_id_header));
                return;
            }
        };
        observation.record_reply_head(response.status(), response.headers(), false);
        if response.status() != http::StatusCode::OK {
            yield Err(rejected(reject_response(response).await, request_id_header));
            return;
        }
        if let Some(error) = wrong_content_type(response.headers(), framing, relaxed_content_type) {
            yield Err(rejected(error, request_id_header));
            return;
        }
        observation.record_reply_head(response.status(), response.headers(), true);
        yield Ok(ObservedSourceEvent::Open);
        let mut body = response.into_body();
        let mut framer = SseFramer::new();
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(ProviderError::from_transport_error(error));
                    return;
                }
            };
            for event in framer.push(&chunk).collect::<Vec<_>>() {
                if event.replaced_invalid_utf8 {
                    // The ledger keeps only original strings; a replaced
                    // one is not what the provider sent.
                    yield Err(ProviderError::Response(
                        "an event of the observed reply is not valid UTF-8".into(),
                    ));
                    return;
                }
                yield Ok(ObservedSourceEvent::Message(event.data));
            }
        }
    })
}

/// A failed send or refused reply, keeping its status, body, headers and
/// request id, as [`stream`](super::stream) keeps them.
fn rejected(
    error: crate::http_client::Error,
    request_id_header: Option<&'static str>,
) -> ProviderError {
    let request_id = error.non_success_headers().and_then(|headers| {
        crate::providers::internal::request_id_from_headers(headers, request_id_header)
    });
    ProviderError::from_transport_error(error).with_provider_request_id(request_id)
}
