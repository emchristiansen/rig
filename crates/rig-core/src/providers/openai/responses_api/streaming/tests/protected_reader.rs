//! Strict known/unknown classification at the actual Responses reader.

use super::helpers::*;
use super::super::{ResponsesEvent, ResponsesStreamOptions, classify_responses_frame, classify_responses_payload};
use crate::message::AssistantContent;
use crate::wire::WireEvent;
use serde_json::{Value, json};

fn part_frame(part: Value) -> Value {
    json!({"type":"response.content_part.added","item_id":"msg_1","output_index":0,
        "content_index":0,"sequence_number":1,"part":part})
}

// F streaming/tests.rs:30
#[test]
fn known_event_cannot_fall_back_to_a_whole_response() {
    let mut body = json!({
        "id":"resp_1", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-test"
    });
    assert!(matches!(
        classify_responses_payload(&body.to_string()),
        WireEvent::Known(ResponsesEvent::Whole(_))
    ));
    body["type"] = json!("response.completed");
    assert!(matches!(
        classify_responses_payload(&body.to_string()),
        WireEvent::Corrupt(_)
    ));
    // Even a duplicate discriminator must not make body decoding rescue the
    // event's invalid envelope.
    let duplicate = format!("{{\"type\":\"future.event\",{}", &body.to_string()[1..]);
    assert!(matches!(
        classify_responses_payload(&duplicate),
        WireEvent::Corrupt(_)
    ));
}

// F streaming/tests.rs:55
#[test]
fn classify_known_event_decodes() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "hi",
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. } if kind == "response.output_text.delta" && frame["delta"] == "hi")
    ));
}

// F streaming/tests.rs:72
#[test]
fn classify_unknown_event_type_is_unknown() {
    let frame = json!({
        "type": "response.web_search_call.searching",
        "output_index": 0,
        "sequence_number": 1,
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Unknown { event_type, .. } if event_type == "response.web_search_call.searching"
    ));
}

// F streaming/tests.rs:98
#[test]
fn classify_reasoning_text_done_is_known_and_decodes() {
    let frame = json!({
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 7,
        "text": "the model's raw chain of thought",
    })
    .to_string();

    let event = classify_responses_frame(&frame);
    assert!(
        !matches!(event, WireEvent::Unknown { .. }),
        "the tag must be in the known-event set: {event:?}"
    );
    assert!(
        !matches!(event, WireEvent::Corrupt(_)),
        "a known tag with no matching ItemChunkKind variant decodes to Corrupt, which the \
             driver surfaces as an in-band Err — worse than the warn it replaced: {event:?}"
    );
    assert!(matches!(event,
        WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. })
            if kind == "response.reasoning_text.done" && frame["text"] == "the model's raw chain of thought"
    ));
}

// F streaming/tests.rs:129
#[test]
fn reasoning_text_done_emits_nothing() {
    let driven = drive(&[json!({
        "type":"response.reasoning_text.done","item_id":"rs_1","output_index":0,
        "content_index":0,"sequence_number":7,"text":"the model's raw chain of thought"
    })]);
    assert!(driven.errors.is_empty(), "{:?}", driven.errors);
    assert!(driven.items.is_empty(), "the bookkeeping done emits no reasoning fragments or fabricated part: {:?}", driven.items);
    assert!(driven.response.choice.is_empty());
}

// F streaming/tests.rs:150
#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(
        classify_responses_frame("{not json"),
        WireEvent::Corrupt(_)
    ));
}

// F streaming/tests.rs:158
#[test]
fn classify_known_event_with_defective_payload_is_corrupt() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": 42,
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

// F streaming/tests.rs:178
#[test]
fn classify_defective_known_content_part_is_corrupt() {
    let frame = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": {"type": "output_text", "text": 42},
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

// F streaming/tests.rs:195
#[test]
fn content_part_known_tag_decodes() {
    let part = json!({"type":"output_text","text":"hi"});
    let WireEvent::Known(ResponsesEvent::Frame { frame, .. }) =
        classify_responses_frame(&part_frame(part.clone()).to_string()) else { panic!("known output_text part"); };
    assert_eq!(frame["part"], part);
    assert_eq!(frame["part"]["text"], "hi");
}

// F streaming/tests.rs:202
#[test]
fn content_part_known_tag_with_defective_payload_errors() {
    for part in [json!({"type":"output_text","text":42}), json!({"type":"summary_text","text":42})] {
        assert!(matches!(classify_responses_frame(&part_frame(part).to_string()), WireEvent::Corrupt(_)));
    }
}

// F streaming/tests.rs:214
#[test]
fn content_part_non_string_type_errors() {
    for part in [json!({"type":42,"text":"hi"}), json!({"type":null,"text":"hi"})] {
        assert!(matches!(classify_responses_frame(&part_frame(part).to_string()), WireEvent::Corrupt(_)));
    }
}

// F streaming/tests.rs:226
#[test]
fn content_part_duplicate_type_key_dispatches_on_the_last_occurrence() {
    let raw = r#"{"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"bogus","type":"output_text","text":"hi"}}"#;
    let WireEvent::Known(ResponsesEvent::Frame { frame, raw: received, .. }) = classify_responses_frame(raw)
        else { panic!("nested part discriminator resolves its last occurrence"); };
    assert_eq!(frame["part"], json!({"type":"output_text","text":"hi"}));
    assert_eq!(received, raw);
}

// F streaming/tests.rs:236
#[test]
fn content_part_unknown_tag_is_preserved_verbatim() {
    let part = json!({"type":"future_refusal","refusal":"no"});
    let event = part_frame(part.clone());
    let WireEvent::Known(ResponsesEvent::Frame { frame, raw, .. }) =
        classify_responses_frame(&event.to_string()) else { panic!("unknown nested part stays a known valid envelope"); };
    assert_eq!(frame["part"], part);
    assert_eq!(serde_json::from_str::<Value>(&raw).unwrap()["part"], part);
}

// F streaming/tests.rs:737
#[test]
fn content_part_added_deserializes_snake_case_part_type() {
    let value = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 3,
        "part": {
            "type": "output_text",
            "text": "hello"
        }
    });
    let WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) =
        classify_responses_frame(&value.to_string()) else { panic!("known snake-case part event"); };
    assert_eq!(kind, "response.content_part.added");
    assert_eq!(frame, value);
}

// F streaming/tests.rs:762
#[test]
fn content_part_done_deserializes_snake_case_part_type() {
    let value = json!({
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 4,
        "part": {
            "type": "summary_text",
            "text": "done"
        }
    });
    let WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) =
        classify_responses_frame(&value.to_string()) else { panic!("known snake-case part event"); };
    assert_eq!(kind, "response.content_part.done");
    assert_eq!(frame, value);
}

// F streaming/tests.rs:787
#[test]
fn reasoning_summary_part_added_deserializes_snake_case_part_type() {
    let value = json!({
        "type": "response.reasoning_summary_part.added",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 5,
        "part": {
            "type": "summary_text",
            "text": "step 1"
        }
    });
    let WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) =
        classify_responses_frame(&value.to_string()) else { panic!("known snake-case part event"); };
    assert_eq!(kind, "response.reasoning_summary_part.added");
    assert_eq!(frame, value);
}

// F streaming/tests.rs:812
#[test]
fn reasoning_summary_part_done_deserializes_snake_case_part_type() {
    let value = json!({
        "type": "response.reasoning_summary_part.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 6,
        "part": {
            "type": "summary_text",
            "text": "step 2"
        }
    });
    let WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) =
        classify_responses_frame(&value.to_string()) else { panic!("known snake-case part event"); };
    assert_eq!(kind, "response.reasoning_summary_part.done");
    assert_eq!(frame, value);
}

// F streaming/tests.rs:3011
#[test]
fn summary_delta_and_done_keep_their_own_members() {
    let delta_with_text = json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 1,
        "text": "wrong member",
    });
    assert!(
        matches!(
            classify_responses_frame(&delta_with_text.to_string()),
            WireEvent::Corrupt(_)
        ),
        "a delta without `delta` must not decode"
    );

    let done_with_delta = json!({
        "type": "response.reasoning_summary_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 2,
        "delta": "wrong member",
    });
    assert!(
        matches!(
            classify_responses_frame(&done_with_delta.to_string()),
            WireEvent::Corrupt(_)
        ),
        "a done without `text` must not decode"
    );

    let done = json!({
        "type": "response.reasoning_summary_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 3,
        "text": "the whole summary",
    });
    let WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) = classify_responses_frame(&done.to_string())
        else { panic!("a well-formed done event decodes"); };
    assert_eq!(kind, "response.reasoning_summary_text.done");
    assert_eq!(frame["text"], "the whole summary");
}

// F streaming/tests.rs:3066
#[test]
fn arguments_done_and_the_done_item_reconcile_through_one_type() {
    let args_done = json!({"type":"response.function_call_arguments.done","item_id":"fc_1",
        "output_index":0,"sequence_number":1,"arguments":"{\"a\": 1}"});
    let WireEvent::Known(ResponsesEvent::Frame { frame, raw, .. }) =
        classify_responses_frame(&args_done.to_string()) else { panic!("args done decodes"); };
    assert_eq!(frame["arguments"], "{\"a\": 1}");
    assert_eq!(serde_json::from_str::<Value>(&raw).unwrap()["arguments"], "{\"a\": 1}");
    let item = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"f",
        "arguments":"{\"a\":1}","status":"completed"});
    let response = folded_body_with(ResponsesStreamOptions::strict(), &[
        args_done,
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}),
        completed_with_output(json!([item])),
    ]).expect("same actual call assembly");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else { panic!("one reconciled call"); };
    assert_eq!(call.function.arguments_value(), json!({"a":1}));
}

