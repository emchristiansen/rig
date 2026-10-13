use super::*;
use crate::proto;
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

fn encoded_image(data: &str) -> Vec<u8> {
    let request = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(image_request(data), Mode::Unary)
        .expect("the source image encodes");
    let part = &request.contents[0].parts[0];
    match &part.data {
        Some(proto::part::Data::InlineData(blob)) => blob.data.clone(),
        other => panic!("expected inline image bytes, got {other:?}"),
    }
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_with_padding() {
    assert_eq!(
        encoded_image("_-wgVQA="),
        vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
    );
}

#[test]
fn test_decode_base64_bytes_accepts_url_safe_no_pad() {
    assert_eq!(encoded_image("_-wgVQA"), vec![0xFF, 0xEC, 0x20, 0x55, 0x00]);
}

#[test]
fn test_decode_base64_bytes_accepts_standard_no_pad() {
    assert_eq!(encoded_image("Zg"), b"f".to_vec());
}

#[test]
fn test_decode_base64_bytes_accepts_data_uri_prefix() {
    assert_eq!(
        encoded_image("data:text/plain;base64,Zm9v"),
        b"foo".to_vec()
    );
}

// Raw parameters merge after the source-derived strip and keep their value.
#[test]
fn raw_data_uri_image_is_not_normalized() {
    let mut request = image_request("data:text/plain;base64,Zm9v");
    request.additional_params = Some(serde_json::json!({
        "contents": [{"role":"user","parts":[{"inlineData":{
            "mimeType":"image/png","data":"data:text/plain;base64,Zm9v"
        }}]}]
    }));
    let error = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)
        .expect_err("the raw data URI remains invalid protobuf bytes");
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::Request(_)
    ));
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
fn rpc_codes_classify_retryability_and_keep_the_code() {
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
        assert_eq!(err.is_retryable(), retryable, "{name}");
        assert_eq!(err.provider_response_status(), None, "{name}");
        let report = err.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{name}");
        assert_eq!(report.retryable, retryable, "{name}");
        assert_eq!(report.http_status, None, "{name}");
        assert_eq!(report.code.as_deref(), Some(name), "{name}");
        let response = report
            .provider_response
            .expect("the reply rides on the report");
        assert_eq!(response.code.as_deref(), Some(name));
        assert_eq!(response.transient, None);
    }
}

#[test]
fn source_order_keeps_a_video_override_out_of_the_image_strip() {
    let data = "data:text/plain;base64,Zm9v";
    let mut request = image_request(data);
    let message::Message::User { content } = &mut request.chat_history[0] else {
        panic!("image fixture has a user message");
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
    let error = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)
        .expect_err("the video override retains its invalid protobuf data URI");
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::Request(_)
    ));
}
