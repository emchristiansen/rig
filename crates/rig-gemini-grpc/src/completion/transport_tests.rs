use super::*;
use crate::proto;
use anyhow::{Context, bail, ensure};
use message::{DocumentSourceKind, Image, ImageMediaType, UserContent};

fn image_request(data: &str) -> CompletionRequest {
    let mut request = CompletionRequest::new("hello");
    request.chat_history = vec![message::Message::User {
        content: vec![UserContent::Image(Image {
            data: DocumentSourceKind::base64(data),
            media_type: Some(ImageMediaType::PNG),
            ..Default::default()
        })],
    }];
    request
}

fn encoded_image(data: &str) -> anyhow::Result<Vec<u8>> {
    let request = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(image_request(data), Mode::Unary)
        .context("the source image encodes")?;
    let content = request
        .contents
        .first()
        .with_context(|| format!("encoded image has no content: {request:?}"))?;
    let part = content
        .parts
        .first()
        .with_context(|| format!("encoded image content has no part: {content:?}"))?;
    match &part.data {
        Some(proto::part::Data::InlineData(blob)) => Ok(blob.data.clone()),
        other => bail!("expected inline image bytes, got {other:?}"),
    }
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_with_padding() -> anyhow::Result<()> {
    let actual = encoded_image("_-wgVQA=")?;
    let expected = vec![0xFF, 0xEC, 0x20, 0x55, 0x00];
    ensure!(
        actual == expected,
        "image bytes differ: actual {actual:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_no_pad() -> anyhow::Result<()> {
    let actual = encoded_image("_-wgVQA")?;
    let expected = vec![0xFF, 0xEC, 0x20, 0x55, 0x00];
    ensure!(
        actual == expected,
        "image bytes differ: actual {actual:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn test_decode_base64_bytes_accepts_standard_no_pad() -> anyhow::Result<()> {
    let actual = encoded_image("Zg")?;
    let expected = b"f".to_vec();
    ensure!(
        actual == expected,
        "image bytes differ: actual {actual:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn test_decode_base64_bytes_accepts_data_uri_prefix() -> anyhow::Result<()> {
    let actual = encoded_image("data:text/plain;base64,Zm9v")?;
    let expected = b"foo".to_vec();
    ensure!(
        actual == expected,
        "image bytes differ: actual {actual:?}, expected {expected:?}"
    );
    Ok(())
}

// Raw parameters merge after the source-derived strip and keep their value.
#[test]
fn raw_data_uri_image_is_not_normalized() -> anyhow::Result<()> {
    let mut request = image_request("data:text/plain;base64,Zm9v");
    request.additional_params = Some(serde_json::json!({
        "contents": [{"role":"user","parts":[{"inlineData":{
            "mimeType":"image/png","data":"data:text/plain;base64,Zm9v"
        }}]}]
    }));
    let error = match GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary) {
        Err(error) => error,
        Ok(request) => bail!(
            "the raw data URI remains invalid protobuf bytes, got encoded request {request:?}"
        ),
    };
    let error = ProviderError::from(error);
    ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected Request error, got {error:?}"
    );
    Ok(())
}

#[test]
fn rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = rpc_error(&status);

    // The raw provider error text is preserved verbatim, and there is no
    // HTTP status because gRPC is a non-HTTP transport.
    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
}

#[test]
fn rpc_codes_classify_retryability_and_keep_the_code() -> anyhow::Result<()> {
    use rig_core::error::ErrorKind;
    let cells = [
        (tonic::Code::Unavailable, true, "UNAVAILABLE"),
        (tonic::Code::ResourceExhausted, true, "RESOURCE_EXHAUSTED"),
        (tonic::Code::DeadlineExceeded, true, "DEADLINE_EXCEEDED"),
        (tonic::Code::Aborted, true, "ABORTED"),
        (tonic::Code::InvalidArgument, false, "INVALID_ARGUMENT"),
        (tonic::Code::NotFound, false, "NOT_FOUND"),
        (tonic::Code::PermissionDenied, false, "PERMISSION_DENIED"),
        (tonic::Code::Unauthenticated, false, "UNAUTHENTICATED"),
        (
            tonic::Code::FailedPrecondition,
            false,
            "FAILED_PRECONDITION",
        ),
        (tonic::Code::Unimplemented, false, "UNIMPLEMENTED"),
        (tonic::Code::Internal, false, "INTERNAL"),
        (tonic::Code::Unknown, false, "UNKNOWN"),
        (tonic::Code::Cancelled, false, "CANCELLED"),
    ];
    for (code, retryable, name) in cells {
        let err = rpc_error(&tonic::Status::new(code, "boom"));
        ensure!(
            err.is_retryable() == retryable,
            "{name}: retryability differs: actual {:?}, expected {retryable:?}",
            err.is_retryable()
        );
        ensure!(
            err.provider_response_status().is_none(),
            "{name}: expected no HTTP status, got {:?}",
            err.provider_response_status()
        );
        let report = err.report();
        ensure!(
            report.kind == ErrorKind::ProviderResponse,
            "{name}: report kind differs: actual {:?}, expected {:?}",
            report.kind,
            ErrorKind::ProviderResponse
        );
        ensure!(
            report.retryable == retryable,
            "{name}: report retryability differs: actual {:?}, expected {retryable:?}",
            report.retryable
        );
        ensure!(
            report.http_status.is_none(),
            "{name}: expected no report HTTP status, got {:?}",
            report.http_status
        );
        ensure!(
            report.code.as_deref() == Some(name),
            "{name}: report code differs: actual {:?}, expected {:?}",
            report.code.as_deref(),
            Some(name)
        );
        let response = report
            .provider_response
            .as_ref()
            .with_context(|| format!("the reply rides on the report: {report:?}"))?;
        ensure!(
            response.code.as_deref() == Some(name),
            "response code differs: actual {:?}, expected {:?}",
            response.code.as_deref(),
            Some(name)
        );
        ensure!(
            response.transient.is_none(),
            "expected no transient flag, got {:?}",
            response.transient
        );
    }
    Ok(())
}

#[test]
fn source_order_keeps_a_video_override_out_of_the_image_strip() -> anyhow::Result<()> {
    let data = "data:text/plain;base64,Zm9v";
    let mut request = image_request(data);
    let message = request
        .chat_history
        .first_mut()
        .context("image fixture has no first message")?;
    let message::Message::User { content } = message else {
        bail!("image fixture has a user message, got {message:?}");
    };
    content.insert(
        0,
        UserContent::Video(message::Video {
            data: DocumentSourceKind::url("https://example.com/clip.mp4"),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: Some(serde_json::json!({"inlineData":{
                "mimeType":"image/png", "data":data
            }})),
        }),
    );
    let error = match GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary) {
        Err(error) => error,
        Ok(request) => bail!(
            "the video override retains its invalid protobuf data URI, got encoded request {request:?}"
        ),
    };
    let error = ProviderError::from(error);
    ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected Request error, got {error:?}"
    );
    Ok(())
}
