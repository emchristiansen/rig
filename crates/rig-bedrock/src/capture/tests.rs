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
        read.extend(events.read(chunk));
    }
    let read: Vec<Value> = read
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("valid messages");
    assert_eq!(
        read,
        [
            json!({ "messageStop": { "stopReason": "end_turn" } }),
            json!({ "throttlingException": { "message": "slow" } }),
            json!({ "metadata": { "metrics": { "latencyMs": 5 } } }),
        ]
    );
}

/// A malformed payload shares a chunk with the valid prefix and later metadata.
#[test]
fn malformed_json_keeps_the_same_chunk_prefix_and_stops_once() {
    let payload = br#"{"contentBlockIndex":0,"delta":"#;
    let bad = message("event", "contentBlockDelta", payload);
    let later = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    let mut suffix = bad;
    suffix.extend(&later);
    let mut body = message(
        "event",
        "contentBlockDelta",
        br#"{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}"#,
    );
    body.extend(message(
        "event",
        "contentBlockStop",
        br#"{"contentBlockIndex":0}"#,
    ));
    body.extend(&suffix);
    let mut events = Events::default();
    let read = events.read(&body);
    assert_eq!(read.len(), 3);
    assert_eq!(
        read[0].as_ref().expect("first event"),
        &json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
    );
    assert_eq!(
        read[1].as_ref().expect("second event"),
        &json!({"contentBlockStop":{"contentBlockIndex":0}})
    );
    let error = read[2].as_ref().expect_err("malformed JSON is fatal");
    let cause =
        serde_json::from_slice::<Value>(payload).expect_err("the exact payload is malformed");
    let expected = format!(
        "Bedrock Converse event-stream JSON payload failed: {cause}; payload_bytes={payload:?}; unvalidated_wire_bytes={suffix:?}"
    );
    assert!(matches!(error, ProviderError::Response(detail) if detail == &expected));
    assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
    assert!(!error.is_retryable());
    assert!(error.corrupt_frame().is_none());
    assert!(std::error::Error::source(error).is_none());
    assert!(events.failed);
    assert_eq!(events.buffer.as_ref(), suffix.as_slice());
    assert!(events.read(&later).is_empty());
    assert!(events.read(&[]).is_empty());
}

#[test]
fn invalid_utf8_payload_keeps_exact_bytes_in_the_fatal_response() {
    let payload = [255, 0, 123];
    let body = message("event", "contentBlockDelta", &payload);
    let mut events = Events::default();
    let read = events.read(&body);
    assert_eq!(read.len(), 1);
    let error = read[0].as_ref().expect_err("invalid UTF-8 is fatal");
    let cause = serde_json::from_slice::<Value>(&payload).expect_err("the payload is not JSON");
    let expected = format!(
        "Bedrock Converse event-stream JSON payload failed: {cause}; payload_bytes={payload:?}; unvalidated_wire_bytes={body:?}"
    );
    assert!(matches!(error, ProviderError::Response(detail) if detail == &expected));
    assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
    assert!(!error.is_retryable());
    assert!(std::error::Error::source(error).is_none());
    assert!(events.read(&body).is_empty());
}

#[test]
fn split_prelude_keeps_the_original_malformed_frame_evidence() {
    let payload = br#"{"contentBlockIndex":0,"delta":"#;
    let bad = message("event", "contentBlockDelta", payload);
    let later = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    let mut events = Events::default();
    let mut prefix = message(
        "event",
        "contentBlockDelta",
        br#"{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}"#,
    );
    prefix.extend(message(
        "event",
        "contentBlockStop",
        br#"{"contentBlockIndex":0}"#,
    ));
    let read = events.read(&prefix);
    assert_eq!(read.len(), 2);
    assert_eq!(
        read[0].as_ref().expect("valid delta"),
        &json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
    );
    assert_eq!(
        read[1].as_ref().expect("valid stop"),
        &json!({"contentBlockStop":{"contentBlockIndex":0}})
    );
    assert!(events.read(&bad[..12]).is_empty());
    assert_eq!(events.buffer.as_ref(), &bad[..12]);
    assert!(!events.failed);
    let mut rest = bad[12..].to_vec();
    rest.extend(&later);
    let read = events.read(&rest);
    let mut suffix = bad;
    suffix.extend(&later);
    assert_eq!(read.len(), 1);
    let cause = serde_json::from_slice::<Value>(payload).expect_err("malformed JSON");
    let expected = format!(
        "Bedrock Converse event-stream JSON payload failed: {cause}; payload_bytes={payload:?}; unvalidated_wire_bytes={suffix:?}"
    );
    assert!(matches!(&read[0], Err(ProviderError::Response(detail)) if detail == &expected));
    assert_eq!(events.buffer.as_ref(), suffix.as_slice());
    assert!(events.read(&later).is_empty());
}

#[test]
fn message_crc_failure_keeps_prefix_and_full_suffix_across_chunks() {
    let mut bad = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    let last = bad.len() - 1;
    bad[last] ^= 1;
    let later = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    let mut suffix = bad.clone();
    suffix.extend(&later);
    let mut probe = bad.as_slice();
    let cause = match MessageFrameDecoder::default().decode_frame(&mut probe) {
        Err(error) => error,
        Ok(_) => panic!("the real SDK must refuse the changed message checksum"),
    };
    let expected = format!(
        "Bedrock Converse event-stream framing failed: {cause}; unvalidated_wire_bytes={suffix:?}"
    );
    for split in [false, true] {
        let mut events = Events::default();
        let mut first = message(
            "event",
            "contentBlockDelta",
            br#"{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}"#,
        );
        first.extend(message(
            "event",
            "contentBlockStop",
            br#"{"contentBlockIndex":0}"#,
        ));
        let read = if split {
            first.extend(&bad[..12]);
            let prefix = events.read(&first);
            assert_eq!(prefix.len(), 2);
            assert_eq!(
                prefix[0].as_ref().expect("valid delta"),
                &json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
            );
            assert_eq!(
                prefix[1].as_ref().expect("valid stop"),
                &json!({"contentBlockStop":{"contentBlockIndex":0}})
            );
            assert_eq!(events.buffer.as_ref(), &bad[..12]);
            let mut rest = bad[12..].to_vec();
            rest.extend(&later);
            events.read(&rest)
        } else {
            first.extend(&suffix);
            let mut read = events.read(&first);
            assert_eq!(read.len(), 3);
            assert_eq!(
                read.remove(0).expect("valid delta"),
                json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
            );
            assert_eq!(
                read.remove(0).expect("valid stop"),
                json!({"contentBlockStop":{"contentBlockIndex":0}})
            );
            read
        };
        assert_eq!(read.len(), 1);
        let error = read[0].as_ref().expect_err("message checksum failure");
        assert!(matches!(error, ProviderError::Response(detail) if detail == &expected));
        assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
        assert!(!error.is_retryable());
        assert!(error.corrupt_frame().is_none());
        assert!(std::error::Error::source(error).is_none());
        assert_eq!(events.buffer.as_ref(), suffix.as_slice());
        assert!(events.read(&later).is_empty());
    }
}

#[test]
fn prelude_crc_failure_keeps_the_full_unvalidated_wire_suffix() {
    let mut bad = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    bad[8] ^= 1;
    let later = message(
        "event",
        "metadata",
        br#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
    );
    let mut suffix = bad.clone();
    suffix.extend(&later);
    let mut probe = bad.as_slice();
    let cause = match MessageFrameDecoder::default().decode_frame(&mut probe) {
        Err(error) => error,
        Ok(_) => panic!("the real SDK must refuse the changed prelude checksum"),
    };
    let mut body = message(
        "event",
        "contentBlockDelta",
        br#"{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}"#,
    );
    body.extend(message(
        "event",
        "contentBlockStop",
        br#"{"contentBlockIndex":0}"#,
    ));
    body.extend(&suffix);
    let mut events = Events::default();
    let read = events.read(&body);
    assert_eq!(read.len(), 3);
    assert_eq!(
        read[0].as_ref().expect("valid delta"),
        &json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
    );
    assert_eq!(
        read[1].as_ref().expect("valid stop"),
        &json!({"contentBlockStop":{"contentBlockIndex":0}})
    );
    let error = read[2].as_ref().expect_err("prelude checksum failure");
    let expected = format!(
        "Bedrock Converse event-stream framing failed: {cause}; unvalidated_wire_bytes={suffix:?}"
    );
    assert!(matches!(error, ProviderError::Response(detail) if detail == &expected));
    assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
    assert!(!error.is_retryable());
    assert!(std::error::Error::source(error).is_none());
    assert_eq!(events.buffer.as_ref(), suffix.as_slice());
    assert!(events.read(&later).is_empty());
}

#[test]
fn valid_json_null_and_future_tags_remain_values() {
    let mut body = message("event", "futureEvent", b"null");
    body.extend(message(
        "event",
        "contentBlockDelta",
        br#"{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}"#,
    ));
    let mut events = Events::default();
    let read = events
        .read(&body)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("valid JSON");
    assert_eq!(
        read,
        [
            json!({"futureEvent":null}),
            json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"valid prefix"}}})
        ]
    );
    assert!(!events.failed);
    assert!(events.buffer.is_empty());
}

/// Valid binary messages that have no routable semantic headers still skip JSON.
#[test]
fn unroutable_headers_keep_the_existing_skip_before_json_policy() {
    for payload in [b"{".as_slice(), b"".as_slice()] {
        let messages = [
            Message::new(payload.to_vec()),
            Message::new(payload.to_vec())
                .add_header(Header::new(":message-type", HeaderValue::Bool(true))),
            Message::new(payload.to_vec()).add_header(Header::new(
                ":message-type",
                HeaderValue::String("unknown".into()),
            )),
            Message::new(payload.to_vec()).add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            )),
            Message::new(payload.to_vec())
                .add_header(Header::new(
                    ":message-type",
                    HeaderValue::String("event".into()),
                ))
                .add_header(Header::new(":event-type", HeaderValue::Bool(true))),
            Message::new(payload.to_vec()).add_header(Header::new(
                ":message-type",
                HeaderValue::String("exception".into()),
            )),
            Message::new(payload.to_vec())
                .add_header(Header::new(
                    ":message-type",
                    HeaderValue::String("exception".into()),
                ))
                .add_header(Header::new(":exception-type", HeaderValue::Bool(true))),
        ];
        for message in messages {
            let mut body = Vec::new();
            aws_smithy_eventstream::frame::write_message_to(&message, &mut body)
                .expect("real SDK frame");
            body.extend(self::message(
                "event",
                "messageStop",
                br#"{"stopReason":"end_turn"}"#,
            ));
            let mut events = Events::default();
            let read = events
                .read(&body)
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .expect("unroutable JSON is skipped");
            assert_eq!(read, [json!({"messageStop":{"stopReason":"end_turn"}})]);
            assert!(!events.failed);
            assert!(events.buffer.is_empty());
        }
    }
}
