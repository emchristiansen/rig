use super::*;
use rig_core::error::ErrorKind;
use std::error::Error as _;

fn json_error() -> serde_json::Error {
    serde_json::from_str::<serde_json::Value>("{").expect_err("an unclosed object")
}

fn corrupt() -> CorruptFrame {
    CorruptFrame::text(Some("text.delta".to_owned()), "{", json_error())
}

/// The message and the wire form the pin's error report gave a corrupt frame.
#[test]
fn a_corrupt_frame_reads_and_serializes_as_it_did_at_the_pin() {
    let error = CodexSessionError::from(corrupt());
    assert_eq!(
        error.to_string(),
        "CorruptFrameError: `text.delta` frame failed to decode: EOF while parsing an object at line 1 column 1; frame: {"
    );
    let mut chain = Vec::new();
    let mut source = error.source();
    while let Some(next) = source {
        chain.push(next.to_string());
        source = next.source();
    }
    assert_eq!(
        chain,
        [
            "`text.delta` frame failed to decode: EOF while parsing an object at line 1 column 1; frame: {",
            "EOF while parsing an object at line 1 column 1",
        ]
    );

    let detail = CorruptFrameDetail::from(error.corrupt_frame().expect("a corrupt frame"));
    assert_eq!(
        serde_json::to_string(&detail).expect("the detail serializes"),
        r#"{"event_type":"text.delta","evidence":{"kind":"text","frame":"{"},"error":"EOF while parsing an object at line 1 column 1"}"#
    );
}

/// Each kind of evidence says how faithful it is, and round-trips.
#[test]
fn every_kind_of_evidence_round_trips_in_wire_form() {
    for (evidence, wire) in [
        (
            FrameEvidence::Text("{\"a\" : 1}".to_owned()),
            r#"{"kind":"text","frame":"{\"a\" : 1}"}"#,
        ),
        (
            FrameEvidence::Bytes(vec![0xff, 0x7b]),
            r#"{"kind":"bytes","frame":[255,123]}"#,
        ),
        (
            FrameEvidence::Decoded("\u{fffd}{".to_owned()),
            "{\"kind\":\"decoded\",\"frame\":\"\u{fffd}{\"}",
        ),
        (FrameEvidence::Absent, r#"{"kind":"absent"}"#),
    ] {
        assert_eq!(
            serde_json::to_string(&evidence).expect("evidence serializes"),
            wire
        );
        assert_eq!(
            serde_json::from_str::<FrameEvidence>(wire).expect("evidence deserializes"),
            evidence
        );
        let detail = CorruptFrameDetail {
            event_type: None,
            evidence: evidence.clone(),
            error: "expected value".to_owned(),
        };
        let round_trip: CorruptFrameDetail =
            serde_json::from_str(&serde_json::to_string(&detail).expect("the detail serializes"))
                .expect("the detail deserializes");
        assert_eq!(round_trip, detail);
    }
}

/// Only text and bytes are the frame exactly as received.
#[test]
fn the_exact_frame_is_offered_only_when_it_was_kept() {
    let text = corrupt();
    assert_eq!(text.frame(), Some("{"));
    assert_eq!(text.frame_bytes(), Some(&b"{"[..]));
    assert_eq!(text.event_type(), Some("text.delta"));

    let bytes = CorruptFrame::bytes(vec![0xff, 0x7b], json_error());
    assert_eq!(bytes.frame(), None, "the bytes are not text");
    assert_eq!(bytes.frame_bytes(), Some(&[0xff, 0x7b][..]));
    assert_eq!(bytes.event_type(), None);
    assert!(
        bytes
            .to_string()
            .contains(r"frame bytes (not UTF-8): \xff{"),
        "{bytes}"
    );

    let absent = CorruptFrame::absent(Some("text.delta".to_owned()), json_error());
    assert_eq!(absent.evidence(), &FrameEvidence::Absent);
    assert_eq!(absent.frame(), None);
    assert_eq!(absent.frame_bytes(), None);
    assert_eq!(
        absent.to_string(),
        "`text.delta` frame failed to decode: EOF while parsing an object at line 1 column 1"
    );
    assert_eq!(text.without_frame().evidence(), &FrameEvidence::Absent);
}

/// A corrupt frame that leaves as a `ProviderError` still carries the frame
/// as received: the conversion is never the end of the evidence.
#[test]
fn the_frame_survives_every_conversion_to_a_provider_error() {
    let direct = ProviderError::from(corrupt());
    assert_eq!(
        direct.corrupt_frame().and_then(CorruptFrame::frame),
        Some("{")
    );
    assert_eq!(direct.kind(), ErrorKind::Request);
    assert!(!direct.is_retryable());

    let through_session = ProviderError::from(CodexSessionError::from(corrupt()));
    let kept = through_session
        .corrupt_frame()
        .expect("the frame is still there");
    assert_eq!(kept.frame(), Some("{"));
    assert_eq!(kept.event_type(), Some("text.delta"));
    assert_eq!(
        CorruptFrameDetail::from(kept),
        CorruptFrameDetail::from(&corrupt())
    );

    // And back: a session error holding that provider error reads it too.
    let wrapped = CodexSessionError::from(through_session);
    assert!(matches!(wrapped, CodexSessionError::Provider(_)));
    assert_eq!(
        wrapped.corrupt_frame().and_then(CorruptFrame::frame),
        Some("{")
    );
}

/// Another failure is not a corrupt frame, whichever way it is asked.
#[test]
fn another_failure_is_not_a_corrupt_frame() {
    let provider = ProviderError::Provider("the connection closed".to_owned());
    assert!(provider.corrupt_frame().is_none());
    let request = ProviderError::request("not a frame");
    assert!(request.corrupt_frame().is_none());
    let session = CodexSessionError::from(provider);
    assert!(session.corrupt_frame().is_none());
    assert_eq!(session.to_string(), "ProviderError: the connection closed");
}
