//! Protected waiting-part, item ownership and source-coordinate regressions.
//! Original fixtures: F6871 streaming/tests.rs5351–5850.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedReasoningPart, CarriedTextContent};
use crate::streaming::{Item, PartKind, StreamEvent};
use serde_json::json;

// F streaming/tests.rs:5351
#[test]
fn a_done_only_second_message_keeps_its_own_text() {
    // Id-less deltas fill slot 0; a different message arrives only as its
    // done at slot 1. It does not restate slot 0, so it keeps its position.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"output_text","text":"B"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_b", json!([{"type":"output_text","text":"B"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", "B"]);
}

// F streaming/tests.rs:5366
#[test]
fn a_done_only_message_extending_streamed_text_is_a_different_message() {
    // Only an exact restatement takes over an id-less slot: "Yes, and more"
    // is not "Yes" restated.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Yes"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"output_text","text":"Yes, and more"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"Yes"}])),
            message_item(
                "msg_b",
                json!([{"type":"output_text","text":"Yes, and more"}])
            )
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["Yes", "Yes, and more"]);
}

// F streaming/tests.rs:5384
#[test]
fn a_done_only_refusal_never_reopens_an_id_less_refusal() {
    let (choice, items) = fold_driven(&[
        json!({"type":"response.refusal.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"no"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"refusal","refusal":"other"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"refusal","refusal":"no"}])),
            message_item("msg_b", json!([{"type":"refusal","refusal":"other"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["no", "other"]);
    let starts: Vec<_> = items.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::Start { part, kind: PartKind::Text, .. }) => Some(*part),
        _ => None,
    }).collect();
    let ends: Vec<_> = items.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::End { part, content: AssistantContent::CarriedText(text), .. })
            if matches!(&text.content, CarriedTextContent::Refusal(_)) => Some(*part),
        _ => None,
    }).collect();
    assert_eq!(starts, ends, "each distinct refusal starts and ends as its own actual part");
}

// F streaming/tests.rs:5407
#[test]
fn a_done_part_cannot_move_an_anonymous_message_away_from_a_terminal_conflict() {
    let driven = drive(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Hello","annotations":[]}}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]))
        ])),
    ]);
    assert!(matches!(driven.errors.as_slice(), [crate::error::ProviderError::Response(message)]
        if message == "conflicting Responses parent kind at output 0: reasoning in a message"));
    assert!(!driven.errors[0].is_retryable());
    assert!(driven.response.response_id().is_none());
    assert_eq!(folded_texts(&driven.response.choice), ["Hello", "Hello"]);
}

// F streaming/tests.rs:5420
#[test]
fn a_content_part_added_never_takes_over_an_id_less_slot() {
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.added","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Hello","annotations":[]}}),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", "Hello"]);
}

// F streaming/tests.rs:5429
#[test]
fn a_terminal_refusal_still_closes_later_reasoning() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"summary_index":0,"sequence_number":1,"delta":"think"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think"}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind at output 0"),
        "{errors:?}"
    );
    assert!(events.iter().any(|item| matches!(item,
        Item::Event(StreamEvent::End { content: AssistantContent::CarriedReasoning(reasoning), .. })
            if reasoning.id.as_deref() == Some("rs_1") && reasoning.display_text() == "think"
    )), "{events:?}");
}

// F streaming/tests.rs:5459
#[test]
fn an_empty_signature_is_no_more_content_than_empty_ciphertext() {
    let done_with = |field: &str| {
        let mut item = json!({"type":"reasoning","id":"rs_1","summary":[]});
        item[field] = json!("");
        driven_frames(&[
            json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":0,"delta":"t"}),
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        ])
    };
    let signature = done_with("signature");
    let ciphertext = done_with("encrypted_content");
    assert!(signature.1.is_empty() && ciphertext.1.is_empty());
    let projection = |items: &[Item<StreamEvent>]| items.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::Reasoning { text, .. }) => Some(text.clone()),
        Item::Event(StreamEvent::End { content: AssistantContent::CarriedReasoning(reasoning), .. })
            => Some(reasoning.display_text()),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(projection(&signature.0), projection(&ciphertext.0));
    for items in [&signature.0, &ciphertext.0] {
        assert_eq!(items.iter().filter(|item| matches!(item,
            Item::Event(StreamEvent::End { content: AssistantContent::CarriedReasoning(_), .. })
        )).count(), 1, "empty metadata adds no second reasoning part");
    }
}

// F streaming/tests.rs:5475
#[test]
fn a_waiting_part_binds_only_to_the_item_its_event_named() {
    // Repair puts msg_1's unknown part at slot 0, where the terminal's
    // reasoning sits; the part belongs to msg_1 alone.
    let unknown = json!({"type":"future_part","payload":[1]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([unknown]))
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    // G188: the empty known rs_1 contributes no canonical reasoning item.
    assert!(reasoning.is_empty(), "{choice:?}");
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5503
#[test]
fn a_waiting_part_whose_item_never_arrives_leaves_raw() {
    let unknown = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]})
        ])),
    ]);
    assert!(
        reasoning_items(&choice)
            .iter()
            .all(|reasoning| reasoning.content.is_empty()),
        "{choice:?}"
    );
    assert_eq!(unknown_payloads(&events), vec![unknown]);
}

// F streaming/tests.rs:5521
#[test]
fn distinct_waiting_observations_bind_to_reasoning_last_writer_wins() {
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.done","output_index":0,"content_index":0,"sequence_number":1,"part":second}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":2,"delta":"t"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"t"}]})
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    let opaque: Vec<_> = reasoning[0]
        .content
        .iter()
        .filter_map(|content| match content {
            CarriedReasoningPart::OpaqueContent(value) => Some(&value.item),
            _ => None,
        })
        .collect();
    assert_eq!(opaque, [&second], "{choice:?}");
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5547
#[test]
fn reasoning_text_added_text_stays_off_the_live_deltas() {
    // As with summary parts, the text a content_part.added carries is not a
    // live reasoning delta.
    let (_, events) = fold_driven(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":{"type":"reasoning","id":"rs_x","summary":[]}}),
        json!({"type":"response.content_part.added","item_id":"rs_x","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":"x"}}),
    ]);
    assert!(
        !events.iter().any(|event| matches!(
            event,
            Item::Event(StreamEvent::Reasoning { .. })
        )),
        "{events:?}"
    );
}

// F streaming/tests.rs:5579
#[test]
fn another_items_waiting_part_never_blocks_a_known_delta() {
    // Repair puts msg_z's unknown part at msg_a's position; it can never
    // bind to msg_a, so msg_a's text is no kind contradiction.
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":1,"delta":"A"}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", ""]);
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5597
#[test]
fn another_items_waiting_part_never_blocks_reasoning_text() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":1,"delta":"r"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"r"}]}),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    assert_eq!(reasoning[0].display_text(), "r");
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5615
#[test]
fn a_terminal_binds_another_items_waiting_part_once_to_its_item() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", ""]);
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5630
#[test]
fn a_waiting_part_of_the_same_item_still_refuses_a_contradicting_kind() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (_, errors) = driven_frames(&[
        json!({"type":"response.content_part.added","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":1,"delta":"A"}),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind"),
        "{errors:?}"
    );
}

// F streaming/tests.rs:5646
#[test]
fn a_reasoning_items_waiting_part_moves_to_its_own_slot() {
    // Repair puts rs_1's unknown part at rs_0's position; it binds once to
    // rs_1 when rs_1 establishes its slot; empty known rs_0 is not canonical.
    let unknown = json!({"type":"future_part","payload":[4]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_0","summary":[]}),
            json!({"type":"reasoning","id":"rs_1","summary":[],"content":[]})
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    // G188: empty known rs_0 is absent; opaque rs_1 remains at its own coordinate.
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning[0].content,
        vec![CarriedReasoningPart::OpaqueContent(crate::message::CarriedPayload::new(unknown.clone(), Some("openai".into())).with_position(crate::message::CarriedPosition::ReasoningContent { part_index: 0 }))]
    );
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5670
#[test]
fn another_items_waiting_part_leaves_a_snapshots_own_position_free() {
    // Only msg_z's unknown part waits at msg_1's position 1; it can never
    // bind to msg_1, so msg_1's position counts as empty and its done takes
    // over the id-less slot it restates exactly, instead of duplicating it.
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":1,"content_index":0,"sequence_number":1,"part":unknown}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":2,"item":message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]))}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"output_text","text":"Hello"}])),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", ""]);
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

// F streaming/tests.rs:5690
#[test]
fn a_reasoning_snapshot_never_overwrites_its_own_waiting_part() {
    // rs_1's unknown part waits at repaired (0, 0); rs_1's snapshot at
    // position 1 then states text at content 0. The part would move there
    // when rs_1 establishes its slot, so the snapshot contradicts it and
    // refuses rather than silently replacing it.
    let unknown = json!({"type":"future_part","payload":[4]});
    let snapshot = json!({"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"r"}]});
    let waiting = json!({"type":"response.content_part.added","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    let terminal = completed_with_output(json!([
        json!({"type":"reasoning","id":"rs_0","summary":[]}),
        snapshot.clone()
    ]));
    for (label, frames) in [
        (
            "output_item.done",
            vec![
                waiting.clone(),
                json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":snapshot}),
                terminal.clone(),
            ],
        ),
        ("terminal", vec![waiting.clone(), terminal.clone()]),
    ] {
        let (_, errors) = driven_frames(&frames);
        assert_eq!(errors.len(), 1, "{label}: {errors:?}");
        assert!(
            errors[0]
                .to_string()
                .contains("conflicting Responses content part kind"),
            "{label}: {errors:?}"
        );
    }
}

// F streaming/tests.rs:5725
#[test]
fn an_added_position_with_anonymous_text_refuses_a_conflicting_terminal() {
    // An anonymous delta does not prove ownership of the separately announced ID.
    let driven = drive(&[
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":0,"item":message_item("msg_1", json!([]))}),
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":1,"delta":"Hello"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]))
        ])),
    ]);
    assert!(matches!(driven.errors.as_slice(), [crate::error::ProviderError::Response(message)]
        if message == "conflicting Responses parent kind at output 0: reasoning in a message"));
    assert!(!driven.errors[0].is_retryable());
    assert!(driven.response.response_id().is_none());
    assert_eq!(folded_texts(&driven.response.choice), ["Hello"]);
}

// F streaming/tests.rs:5742
#[test]
fn a_terminal_refusal_still_starts_and_closes_terminal_only_reasoning() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_t","summary":[{"type":"summary_text","text":"t"}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    let closed = events.iter().find_map(|item| match item {
        Item::Event(StreamEvent::End { part, content: AssistantContent::CarriedReasoning(reasoning), .. })
            if reasoning.id.as_deref() == Some("rs_t") && reasoning.display_text() == "t" => Some(*part),
        _ => None,
    }).expect("terminal-only reasoning closes after the earlier text refusal");
    assert!(events.iter().any(|item| matches!(item,
        Item::Event(StreamEvent::Start { part, kind: PartKind::Reasoning, .. }) if *part == closed
    )), "{events:?}");
}

// F streaming/tests.rs:5773
#[test]
fn the_first_terminal_refusal_stands_over_a_later_reasoning_conflict() {
    let (_, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"summary_index":0,"sequence_number":1,"delta":"think"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"future_summary","payload":[2]}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind at output 0, content 0"),
        "{errors:?}"
    );
}

// F streaming/tests.rs:5792
#[test]
fn an_empty_signature_on_an_id_less_first_done_publishes_nothing() {
    // No prior slot and no provider id: an empty signature, like empty
    // ciphertext, is no content, so the done opens no reasoning block.
    for field in ["signature", "encrypted_content"] {
        let mut item = json!({"type":"reasoning","id":"","summary":[]});
        item[field] = json!("");
        let (events, errors) = driven_frames(&[
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":0,"item":item}),
        ]);
        assert!(errors.is_empty(), "{field}: {errors:?}");
        assert!(
            !events.iter().any(|event| matches!(
                event,
                Item::Event(StreamEvent::Start { .. } | StreamEvent::End { .. })
            )),
            "{field}: {events:?}"
        );
    }
}

// F streaming/tests.rs:5813
#[test]
fn equal_text_of_another_kind_is_not_a_restatement() {
    // "no" streamed id-less as text, then a done-only refusal "no": not the
    // same part, so it keeps its own position.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"no"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"refusal","refusal":"no"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"no"}])),
            message_item("msg_b", json!([{"type":"refusal","refusal":"no"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["no", "no"]);
}

// F streaming/tests.rs:5828
#[test]
fn a_non_matching_content_part_done_keeps_its_own_position() {
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Other","annotations":[]}}),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", "Other"]);
}

// F streaming/tests.rs:5837
#[test]
fn two_items_waiting_parts_at_one_position_each_bind_to_their_own_item() {
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.added","item_id":"msg_b","output_index":0,"content_index":0,"sequence_number":1,"part":second}),
        completed_with_output(json!([
            message_item("msg_a", json!([first])),
            message_item("msg_b", json!([second]))
        ])),
    ]);
    assert_eq!(opaque_texts(&choice), [&first, &second]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}
