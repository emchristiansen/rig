use super::{Decoder, Json, WireEvent, WireFrame};
use crate::error::FrameEvidence;
use crate::operation::Verify;
use crate::streaming::UnknownPayload;
use serde_json::Value;

fn classify(frame: &str) -> WireEvent<((), Value)> {
    <Json as Decoder<'static, Verify>>::classify(&Json, WireFrame::Text(frame.to_owned()))
}

#[test]
fn a_typed_transport_names_an_unmodeled_event_with_its_detail() {
    let event: WireEvent<()> = WireEvent::unrecognized("future.event", "the payload");
    let WireEvent::Unknown { event_type, value } = event else {
        panic!("an unrecognized event is unknown, not known or corrupt");
    };
    assert_eq!(event_type, "future.event");
    assert_eq!(
        value,
        UnknownPayload::new(Value::String("the payload".to_owned()))
    );
}

#[test]
fn a_typed_transport_decode_failure_is_corrupt_with_no_frame() {
    let event: WireEvent<()> = WireEvent::malformed("the event lost its payload");
    let WireEvent::Corrupt(corrupt) = event else {
        panic!("a malformed event is corrupt");
    };
    assert_eq!(corrupt.event_type(), None);
    assert_eq!(corrupt.evidence(), &FrameEvidence::Absent);
    assert_eq!(corrupt.error().to_string(), "the event lost its payload");
}

#[test]
fn the_json_decoder_keeps_a_document_that_is_not_json_as_corrupt_text() {
    let expected = serde_json::from_str::<Value>("{\"open\":").expect_err("invalid JSON");
    let WireEvent::Corrupt(corrupt) = classify("{\"open\":") else {
        panic!("invalid JSON is corrupt");
    };
    assert_eq!(corrupt.event_type(), None);
    assert_eq!(
        corrupt.evidence(),
        &FrameEvidence::Text("{\"open\":".to_owned())
    );
    assert_eq!(corrupt.error().to_string(), expected.to_string());
}

#[test]
fn the_json_decoder_keeps_a_document_that_is_not_the_end_as_corrupt_text() {
    let expected = serde_json::from_str::<()>("{\"a\":1}").expect_err("an object is not unit");
    let WireEvent::Corrupt(corrupt) = classify("{\"a\":1}") else {
        panic!("a document that is not the end is corrupt");
    };
    assert_eq!(corrupt.event_type(), None);
    assert_eq!(
        corrupt.evidence(),
        &FrameEvidence::Text("{\"a\":1}".to_owned())
    );
    assert_eq!(corrupt.error().to_string(), expected.to_string());

    let WireEvent::Known(((), document)) = classify("null") else {
        panic!("null is the unit end");
    };
    assert_eq!(document, Value::Null);
}
