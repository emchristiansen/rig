//! Protected assembly identity and delivered fragments.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::streaming::{Item, PartKind, StreamEvent};
use serde_json::json;

// F streaming/tests.rs:1607
#[test]
fn mixed_id_and_id_less_reasoning_frames_share_one_slot_key() {
    let with_id = json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 1,
        "delta": "s1 ",
    });
    let id_less = json!({
        "type": "response.reasoning_summary_text.delta",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 2,
        "delta": "s2",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 3,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "s1 s2"}],
            "content": [],
            "status": "completed",
        },
    });
    let completed = json!({
        "type": "response.completed",
        "response": sample_response("completed"),
    });

    let decoded = decode_repaired(&[with_id, id_less, done, completed]);
    let keys = decoded
        .events()
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: PartKind::Reasoning,
                ..
            }
            | StreamEvent::Reasoning { part, .. }
            | StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(_),
                ..
            } => Some(*part),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(keys.len(), 1, "one slot has one actual branded part");
    let response = decoded.outcome.expect("the mixed slot normalizes");
    assert_eq!(
        reasoning_items(&response.choice).len(),
        1,
        "no orphaned delta-built part"
    );
}

// F streaming/tests.rs:1693
#[test]
fn envelope_less_reasoning_deltas_are_superseded_by_their_done_item() {
    let delta = json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 2,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "think"}],
            "content": [],
            "status": "completed",
        },
    });
    let completed = json!({
        "type": "response.completed",
        "response": sample_response("completed"),
    });

    let decoded = decode_repaired(&[delta, done, completed]);
    let start = decoded
        .events()
        .into_iter()
        .find_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: PartKind::Reasoning,
                ..
            } => Some(*part),
            _ => None,
        })
        .expect("repaired delta opens a real reasoning part");
    assert!(decoded.events().iter().any(|event| matches!(event, StreamEvent::End {part,content:AssistantContent::CarriedReasoning(_),..} if *part == start)));
    let response = decoded.outcome.expect("replay normalizes");
    let reasoning = reasoning_items(&response.choice);
    assert_eq!(reasoning.len(), 1);
    let occurrences = reasoning
        .iter()
        .flat_map(|item| &item.content)
        .filter(|part| match part {
            CarriedReasoningPart::Summary(text) | CarriedReasoningPart::Text { text, .. } => {
                text.contains("think")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        occurrences, 1,
        "done supersedes deltas instead of duplicating them"
    );
}

// F streaming/tests.rs:1764
#[test]
fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
    let events = [
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hello "
        }),
        json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_2",
            "output_index": 1,
            "summary_index": 0,
            "sequence_number": 2,
            "delta": "because"
        }),
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "world"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response("completed"),
        }),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    let text_parts = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start {
                part,
                kind: PartKind::Text,
                ..
            })
            | Item::Event(StreamEvent::Text { part, .. }) => Some(*part),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(
        text_parts.len(),
        1,
        "the same actual part resumes across reasoning"
    );
    // A29 preserves F's two public activations without opening a second part.
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(
                    StreamEvent::Start {
                        kind: PartKind::Text,
                        ..
                    } | StreamEvent::Resume { .. }
                )
            ))
            .count(),
        2,
        "one start plus one real same-part resume"
    );
    let fragments = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Text { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(fragments, ["hello ", "world"]);
    assert_eq!(folded_texts(&driven.response.choice), ["hello world"]);
    assert_eq!(reasoning_items(&driven.response.choice).len(), 1);
}

// F streaming/tests.rs:1853
#[test]
fn mixed_id_and_id_less_events_share_one_slot_key() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_real",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        // Id-less delta for the same slot: must resolve to the slot's
        // established key, not mint a second identity.
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "function_call",
                "id": "fc_real",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "{\"x\":1}",
                "status": "completed",
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response("completed"),
        }),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    let keys = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                ..
            })
            | Item::Event(StreamEvent::Arguments { part, .. })
            | Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(_),
                ..
            }) => Some(*part),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(keys.len(), 1, "one function slot has one part");
    let [AssistantContent::ToolCall(call)] = driven.response.choice.as_slice() else {
        panic!("one call")
    };
    assert_eq!(call.function.name.as_str(), "tool_a");
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_a"));
    assert_eq!(
        call.native.as_ref().map(|native| &native.item["id"]),
        Some(&json!("fc_real"))
    );
}

// F streaming/tests.rs:1948
#[test]
fn parallel_id_less_function_calls_assemble_distinctly() {
    let call_item = |name: &str, call_id: &str, arguments: &str| {
        json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        })
    };
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": call_item("tool_a", "call_a", ""),
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "sequence_number": 2,
            "item": call_item("tool_b", "call_b", ""),
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 3,
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "sequence_number": 4,
            "delta": "{\"y\":2}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 5,
            "item": call_item("tool_a", "call_a", "{\"x\":1}"),
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "sequence_number": 6,
            "item": call_item("tool_b", "call_b", "{\"y\":2}"),
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 7,
            "response": sample_response("completed"),
        }),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    let mut calls = driven
        .response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some((
                call.function.name.as_str().to_owned(),
                call.function.arguments_value().to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    calls.sort();
    assert_eq!(
        calls,
        [
            ("tool_a".to_owned(), json!({"x":1}).to_string()),
            ("tool_b".to_owned(), json!({"y":2}).to_string())
        ]
    );
}

// F streaming/tests.rs:2042
#[test]
fn a_lost_done_frame_does_not_discard_a_provider_completed_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"city\":\"Paris\"}"
        }),
        // The output_item.done frame is lost; the terminal still arrives.
        json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": sample_response("completed"),
        }),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    let calls = driven
        .response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1, "genuine completed terminal drains the call");
    assert_eq!(calls[0].function.name.as_str(), "get_weather");
    assert_eq!(calls[0].function.arguments_value(), json!({"city":"Paris"}));
    assert_eq!(
        calls[0].id.provider().map(|id| id.as_str()),
        Some("call_abc")
    );
    // G171: the never-done call has no completed native item; fc_1 stays in the fixture.
    assert!(calls[0].native.is_none());
}

// F streaming/tests.rs:2104
#[test]
fn id_less_args_deltas_surface_and_truncation_fabricates_no_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"loc\":"
        }),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert!(driven.items.iter().any(
        |item| matches!(item,Item::Event(StreamEvent::Arguments {json,..}) if json == "{\"loc\":")
    ));
    assert!(
        driven.response.choice.is_empty(),
        "partial arguments without done never fabricate a call"
    );
    assert!(driven.response.response_id().is_none());
    assert!(!driven.items.iter().any(|item| matches!(
        item,
        Item::Event(
            StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            } | StreamEvent::Discard { .. }
        )
    )));
}

// F streaming/tests.rs:2157
#[test]
fn refusal_content_part_frames_do_not_fail_the_buffered_body() {
    // The ChatGPT buffered route replays recorded SSE bodies; a refusal
    // turn's `content_part` frames (an unmodeled `refusal` part) must not
    // fail the whole completion — the refusal text arrives via the
    // modeled `response.refusal.delta`.
    let part_added = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": { "type": "refusal", "refusal": "" }
    });
    let refusal_delta = json!({
        "type": "response.refusal.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 2,
        "delta": "no"
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": sample_response("completed"),
    });

    let decoded = decode_repaired(&[part_added, refusal_delta, completed]);
    assert!(
        decoded
            .events()
            .into_iter()
            .any(|event| matches!(event,StreamEvent::Text {text,..} if text == "no"))
    );
    decoded
        .outcome
        .expect("refusal parts do not fail buffered decode");
}
