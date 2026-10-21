use super::*;
use aws_smithy_types::event_stream::{Header, HeaderValue};
use serde_json::json;

fn message(kind: &str, event: &str, payload: &[u8]) -> Vec<u8> {
    let name = if kind == "exception" {
        ":exception-type"
    } else {
        ":event-type"
    };
    let message = Message::new(payload.to_vec())
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String(kind.to_owned().into()),
        ))
        .add_header(Header::new(
            name,
            HeaderValue::String(event.to_owned().into()),
        ));
    let mut bytes = Vec::new();
    aws_smithy_eventstream::frame::write_message_to(&message, &mut bytes).expect("writes");
    bytes
}

/// Events and exceptions come out whole however the body is chunked.
#[test]
fn events_read_whole_messages_across_chunks() {
    let mut body = message("event", "messageStop", br#"{"stopReason":"end_turn"}"#);
    body.extend(message(
        "exception",
        "throttlingException",
        br#"{"message":"slow"}"#,
    ));
    body.extend(message(
        "event",
        "metadata",
        br#"{"metrics":{"latencyMs":5}}"#,
    ));
    let mut events = Events::default();
    let mut read = Vec::new();
    for chunk in body.chunks(7) {
        read.extend(events.read(chunk).into_iter().map(|event| event.expect("valid event")));
    }
    assert_eq!(
        read,
        [
            json!({ "messageStop": { "stopReason": "end_turn" } }),
            json!({ "throttlingException": { "message": "slow" } }),
            json!({ "metadata": { "metrics": { "latencyMs": 5 } } }),
        ]
    );
}

#[test]
fn malformed_payload_keeps_exact_frame_and_stops_after_prior_events() {
    let prior = message("event", "messageStart", br#"{"role":"assistant"}"#);
    let malformed = message("event", "contentBlockDelta", b"{not json\xff");
    let later = message("event", "metadata", br#"{"usage":{"inputTokens":1}}"#);
    let mut bytes = prior.clone();
    bytes.extend(&malformed);
    bytes.extend(&later);
    let mut events = Events::default();
    let mut read = Vec::new();
    for chunk in bytes.chunks(7) { read.extend(events.read(chunk)); }
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].as_ref().expect("prior event"), &json!({"messageStart":{"role":"assistant"}}));
    let corrupt = read[1].as_ref().expect_err("malformed payload");
    assert_eq!(corrupt.evidence(), &rig_core::error::FrameEvidence::Bytes(malformed));
    assert!(corrupt.frame().is_none());
    assert!(events.read(&later).is_empty());
}

#[test]
fn checksum_error_keeps_frame_before_sdk_consumes_it() {
    let mut malformed = message("event", "messageStop", br#"{"stopReason":"end_turn"}"#);
    let last = malformed.len() - 1;
    malformed[last] ^= 1;
    let later = message("event", "metadata", b"{}");
    let mut bytes = malformed.clone();
    bytes.extend(&later);
    let mut sdk = MessageFrameDecoder::default();
    let reason = sdk.decode_frame(&mut BytesMut::from(malformed.as_slice())).expect_err("checksum failure").to_string();
    let mut events = Events::default();
    let read = events.read(&bytes);
    assert_eq!(read.len(), 1);
    let corrupt = read[0].as_ref().expect_err("checksum failure");
    assert_eq!(corrupt.evidence(), &rig_core::error::FrameEvidence::Bytes(malformed));
    assert!(corrupt.to_string().contains(&reason));
    assert!(events.read(&later).is_empty());
}
