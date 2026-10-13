//! Protected snapshot custody and refusal before mutation.

use super::helpers::*;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::streaming::{Item, PartKind, StreamEvent};
use serde_json::json;

// F streaming/tests.rs:4885
#[test]
fn repeated_reasoning_snapshots_publish_exactly_one_close() {
    let unknown = json!({"type":"future_content","list":[1]});
    let item = json!({"type":"reasoning","id":"rs_once","summary":[{"type":"summary_text","text":"known"}],"content":[unknown]});
    let driven = drive(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}),
        completed_with_output(json!([item])),
    ]);
    assert!(driven.errors.is_empty()); assert_eq!(driven.items.iter().filter(|item|matches!(item,Item::Event(StreamEvent::End {content:AssistantContent::CarriedReasoning(_),..}))).count(),1);
}

// F streaming/tests.rs:4911
#[test]
fn a_terminal_restating_known_text_as_opaque_is_refused() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let driven = drive(&[
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        completed_with_output(json!([opaque_message_fixture(json!([unknown]))])),
    ]);
    assert_eq!(driven.errors.len(),1); assert!(driven.errors[0].to_string().contains("conflicting Responses content part kind")); assert!(driven.response.response_id().is_none()); assert!(opaque_texts(&driven.response.choice).is_empty(),"known text gains no opaque marker");
}

// F streaming/tests.rs:4936
#[test]
fn an_empty_done_after_reasoning_deltas_keeps_the_streamed_reasoning() {
    // The empty restatement neither erases the deltas nor cancels their one
    // close; the streamed reasoning still reaches the folded choice.
    let empty = json!({"type":"reasoning","id":"rs_deltas","summary":[],"content":[]});
    let driven = drive(&[
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_deltas","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":empty}),
        completed_with_output(json!([empty])),
    ]);
    assert!(driven.errors.is_empty()); let [AssistantContent::CarriedReasoning(reasoning)]=driven.response.choice.as_slice() else {panic!("one reasoning item")}; assert_eq!(reasoning.display_text(),"known");
}

// F streaming/tests.rs:5069
#[test]
fn an_added_position_owns_its_id_against_later_repaired_deltas() {
    let message = message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]));
    let driven = drive(&[
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":0,"item":message_item("msg_1", json!([]))}),
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1,"delta":"Hello"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":2,"item":message}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message
        ])),
    ]);
    assert!(driven.errors.is_empty()); assert_eq!(folded_texts(&driven.response.choice),["Hello"]);
}

// F streaming/tests.rs:5091
#[test]
fn an_added_message_never_takes_over_an_id_less_slot() {
    // msg_b is announced at its own slot; its annotations must not land on
    // the text an id-less delta streamed into slot 0.
    let annotated = json!([{"type":"output_text","text":"B","annotations":[{"url":"b"}]}]);
    let driven = drive(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([]))}),
        json!({"type":"response.output_text.delta","item_id":"msg_b","output_index":1,"content_index":0,"sequence_number":2,"delta":"B"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":3,"item":message_item("msg_b", annotated.clone())}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_b", annotated)
        ])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(folded_texts(&driven.response.choice), ["A", "B"]);
    let AssistantContent::Text(_) = &driven.response.choice[0] else {
        panic!("text first")
    };
    let a = driven.response.choice[0]
        .native_item()
        .expect("A retains the actual received message");
    assert!(
        a["content"][0].get("annotations").is_none(),
        "B's annotations never land on A"
    );
}

// F streaming/tests.rs:5126
#[test]
fn an_opaque_terminal_part_elsewhere_never_marks_repaired_streamed_text() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let driven = drive(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_2", json!([unknown]))
        ])),
    ]);
    assert!(matches!(driven.errors.as_slice(), [ProviderError::Response(message)]
        if message == "conflicting Responses parent kind at output 0: reasoning in a message"));
    assert!(!driven.errors[0].is_retryable());
    assert!(driven.response.response_id().is_none());
    assert_eq!(folded_texts(&driven.response.choice), ["Hello"]);
    assert!(matches!(driven.response.choice.as_slice(),
        [AssistantContent::Text(text)] if text.text == "Hello"));
    assert!(opaque_texts(&driven.response.choice).is_empty());
}

// F streaming/tests.rs:5203
#[test]
fn known_text_over_a_waiting_opaque_part_is_refused_before_binding() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let driven = drive(&[
        json!({"type":"response.content_part.added","item_id":"msg_w","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_w","output_index":0,"content_index":0,"sequence_number":1,"delta":"late"}),
    ]);
    assert_eq!(driven.errors.len(),1); assert!(driven.errors[0].to_string().contains("conflicting Responses content part kind")); assert!(!driven.items.iter().any(|item|matches!(item,Item::Event(StreamEvent::Text {..}|StreamEvent::Start {..})))); assert_eq!(unknown_payloads(&driven.items),[unknown]);
}

// F streaming/tests.rs:5232
#[test]
fn a_content_part_without_a_type_tag_is_a_malformed_frame() {
    let driven = drive(&[
        json!({"type":"response.content_part.added","item_id":"msg_t","output_index":0,"content_index":0,"sequence_number":0,"part":{"text":"x"}}),
    ]);
    assert!(matches!(driven.errors.as_slice(),[ProviderError::CorruptFrame(corrupt)] if corrupt.event_type()==Some("response.content_part.added"))); assert!(unknown_payloads(&driven.items).is_empty());
}

// F streaming/tests.rs:5306
#[test]
fn a_reasoning_text_part_inside_a_message_is_refused() {
    let driven = drive(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":message_item("msg_r", json!([]))}),
        json!({"type":"response.content_part.added","item_id":"msg_r","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":"think"}}),
    ]);
    assert_eq!(driven.errors.len(),1); assert!(driven.errors[0].to_string().contains("conflicting Responses parent kind")); assert!(!driven.items.iter().any(|item|matches!(item,Item::Event(StreamEvent::Start {kind:PartKind::Reasoning,..}))));
}

