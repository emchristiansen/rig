//! Exact malformed-frame custody through real framed and byte transports.

use super::super::{ResponsesDecoder, ResponsesStreamOptions};
use super::helpers::*;
use crate::error::{ErrorKind, ProviderError};
use crate::providers::openai::wire::OpenAIConfig;
use crate::test_utils::MockStreamingClient;
use crate::wire::WireFrame;
use futures::StreamExt;

// F streaming/tests.rs:5857
#[test]
fn a_malformed_known_frame_surfaces_on_the_sse_stream_carrying_the_frame_as_sent() {
    use super::super::super::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES;

    for malformed in MALFORMED_KNOWN_FRAMES {
        let body = format!(
            "event: {}\ndata: {}\n\n",
            malformed.event_type, malformed.frame
        );
        let mut framer = rig_http::http_client::framing::SseFramer::new();
        let frames = framer
            .push(body.as_bytes())
            .map(|event| WireFrame::Text(event.data));
        let error = crate::driver::feed_frames!(
            ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
            "openai",
            frames
        )
        .outcome
        .unwrap_err();
        let ProviderError::CorruptFrame(corrupt) = &error else {
            panic!(
                "{}: expected a corrupt frame, got {error:?}",
                malformed.case
            );
        };
        assert_eq!(
            corrupt.event_type(),
            Some(malformed.event_type),
            "{}",
            malformed.case
        );
        assert_eq!(corrupt.frame(), Some(malformed.frame), "{}", malformed.case);
        assert_eq!(error.kind(), ErrorKind::Json, "{}", malformed.case);
        assert!(!error.is_retryable(), "{}", malformed.case);
    }
}

// F streaming/tests.rs:5898
#[tokio::test]
async fn a_malformed_known_frame_reaches_the_stream_report_as_structured_detail() {
    use super::super::super::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES;
    use crate::error::{CorruptFrameDetail, ErrorDetail, FrameEvidence};

    for malformed in MALFORMED_KNOWN_FRAMES {
        let mut stream = responses_stream(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(format!("data: {}\n\n", malformed.frame)),
        })
        .await;
        let mut report = None;
        while let Some(item) = stream.next().await {
            if let Err(error) = item {
                report = Some(error.report());
                break;
            }
        }
        let report = report.unwrap_or_else(|| panic!("{}: the stream must fail", malformed.case));
        assert_eq!(report.kind, ErrorKind::Json, "{}", malformed.case);
        let Some(ErrorDetail::CorruptFrame(CorruptFrameDetail {
            event_type,
            evidence,
            ..
        })) = &report.detail
        else {
            panic!(
                "{}: expected corrupt-frame detail, got {report:?}",
                malformed.case
            );
        };
        assert_eq!(
            event_type.as_deref(),
            Some(malformed.event_type),
            "{}",
            malformed.case
        );
        assert_eq!(
            evidence,
            &FrameEvidence::Text(malformed.frame.to_owned()),
            "{}",
            malformed.case
        );
    }
}

// F streaming/tests.rs:5947
#[tokio::test]
async fn a_corrupt_frame_from_replacement_decoded_sse_is_labelled_decoded() {
    use crate::error::{CorruptFrameDetail, ErrorDetail, FrameEvidence};

    let mut body = br#"data: {"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"refusal","note":""#.to_vec();
    body.push(0xff);
    body.extend_from_slice(b"\"}}\n\n");
    let decoded = String::from_utf8_lossy(&body)
        .strip_prefix("data: ")
        .and_then(|data| data.strip_suffix("\n\n"))
        .expect("one data line")
        .to_owned();

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(body),
    })
    .await;
    let mut report = None;
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            report = Some(error.report());
            break;
        }
    }
    let report = report.expect("the malformed refusal part must surface an error");
    let Some(ErrorDetail::CorruptFrame(CorruptFrameDetail {
        event_type,
        evidence,
        ..
    })) = &report.detail
    else {
        panic!("expected corrupt-frame detail, got {report:?}");
    };
    assert_eq!(event_type.as_deref(), Some("response.content_part.added"));
    assert_eq!(evidence, &FrameEvidence::Decoded(decoded));
}

// F streaming/tests.rs:5987
#[tokio::test]
async fn a_corrupt_byte_frame_keeps_its_exact_bytes_on_every_driver_entrypoint() {
    use crate::error::FrameEvidence;

    let received = vec![0x7b, 0xff];

    let error = crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        "openai",
        [WireFrame::Bytes(received.clone())]
    )
    .outcome
    .expect_err("the frame cannot decode");
    let ProviderError::CorruptFrame(corrupt) = &error else {
        panic!("expected a corrupt frame, got {error:?}");
    };
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(received.clone()));

    let model = crate::driver::Model::new(
        OpenAIConfig::new("test-key").responses("gpt-5.4"),
        ScriptedFrames(vec![WireFrame::Bytes(received.clone())]),
    );
    let mut stream = model
        .stream(crate::completion::CompletionRequest::new("fixture"))
        .expect("actual Responses wire over an actual byte-frame Transport");
    let mut error = None;
    while let Some(item) = stream.next().await {
        if let Err(found) = item {
            error = Some(found);
            break;
        }
    }
    let error = error.expect("the frame cannot decode");
    let ProviderError::CorruptFrame(corrupt) = &error else {
        panic!("expected a corrupt frame, got {error:?}");
    };
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(received));
}
