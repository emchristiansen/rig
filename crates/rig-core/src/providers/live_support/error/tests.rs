//! Evidence retention at the baseline transport bridge; no provider needed.
use super::*;
#[test]
fn legacy_status_does_not_invent_body_or_headers() {
    let error=ProviderError::from_transport_error(http_client::Error::InvalidStatusCode(http::StatusCode::UNAUTHORIZED));
    assert_eq!(error.provider_response_status(),Some(http::StatusCode::UNAUTHORIZED));
    assert!(error.provider_response_body().is_none());
    assert!(error.provider_response_headers().is_none());
    let error=ProviderError::from_transport_error(http_client::Error::InvalidStatusCodeWithMessage(http::StatusCode::FORBIDDEN,"exact body".into()));
    assert_eq!(error.provider_response_body(),Some("exact body"));
    assert!(error.provider_response_headers().is_none());
}
#[test]
fn detailed_reply_and_corrupt_frame_keep_original_evidence() {
    let mut headers=http::HeaderMap::new();
    headers.insert("x-request-id",http::HeaderValue::from_static("req-1"));
    let error=ProviderError::from_transport_error(http_client::Error::InvalidStatusCodeWithDetails { status:http::StatusCode::BAD_GATEWAY,body:"upstream exact".into(),headers:Box::new(headers.clone()) });
    assert_eq!(error.provider_response_headers(),Some(&headers));
    let source=serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
    let corrupt=CorruptFrame::text(Some("event".into()),"not json",source);
    assert_eq!(corrupt.frame_bytes(),Some(&b"not json"[..]));
    let report=ProviderError::CorruptFrame(corrupt).report();
    let Some(ErrorDetail::CorruptFrame(detail))=report.detail else { panic!("missing detail") };
    assert_eq!(detail.evidence,FrameEvidence::Text("not json".into()));
}
