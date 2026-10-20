//! Protected output/content coordinates and whole-item custody.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedTextContent};
use crate::streaming::{Item, PartKind, SourceOrder, StreamEvent};
use serde_json::json;

// F streaming/tests.rs:3672
#[test]
fn terminal_reconciles_only_unseen_provider_item_slots() {
    let done = json!({
        "type": "web_search_call",
        "status": "completed",
        "action": {"query": "done"}
    });
    let added_only = json!({
        "type": "web_search_call",
        "status": "completed",
        "action": {"query": "added"}
    });
    let terminal_only = json!({"type": "compaction", "id": "cmp_2", "encrypted_content": "opaque"});
    let output = json!([done, added_only, terminal_only]);
    let events = [
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": output[0]
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "sequence_number": 2,
            "item": output[1]
        }),
        completed_with_output(output.clone()),
    ];
    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ProviderItem(_),
                    ..
                })
            ))
            .count(),
        3,
        "no duplicate done, both unseen slots filled"
    );
    for output_index in 0..3 {
        assert!(driven.items.iter().any(|item|matches!(item,Item::Event(StreamEvent::Start {kind:PartKind::ProviderItem,source_order:Some(order),..}) if *order==SourceOrder::new(output_index,0))));
    }
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["provider", "provider", "provider"]
    );
    assert_eq!(driven.response.choice, unary_choice(output));
}

// F streaming/tests.rs:3801
#[test]
fn reversed_first_observations_keep_provider_output_order_and_unary_parity() {
    let provider = json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"query": "rig"}
    });
    let function = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
        "status": "completed"
    });
    let custom = json!({
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_2",
        "name": "shell",
        "input": "pwd",
        "status": "completed"
    });
    let message = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}]
    });
    let output = json!([provider, function, custom, message]);
    let events = [
        json!({
            "type": "response.output_text.delta", "item_id": "msg_1",
            "output_index": 3, "content_index": 0, "sequence_number": 1,
            "delta": "done"
        }),
        json!({
            "type": "response.output_item.done", "output_index": 1,
            "sequence_number": 2, "item": output[1]
        }),
        json!({
            "type": "response.output_item.done", "output_index": 0,
            "sequence_number": 3, "item": output[0]
        }),
        json!({
            "type": "response.output_item.done", "output_index": 2,
            "sequence_number": 4, "item": output[2]
        }),
        // A repeated late end for the same identity and source slot is inert.
        json!({
            "type": "response.output_item.done", "output_index": 1,
            "sequence_number": 5, "item": output[1]
        }),
        json!({
            "type": "response.output_item.done", "output_index": 3,
            "sequence_number": 6, "item": output[3]
        }),
        completed_with_output(output.clone()),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    ..
                })
            ))
            .count(),
        1,
        "repeated done adds no second positioned start"
    );
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["provider", "function", "custom", "text"]
    );
    assert_eq!(driven.response.choice, unary_choice(output));
}

// F streaming/tests.rs:3895
#[test]
fn separated_message_text_parts_stay_separate_parts_with_unary_parity() {
    let message = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "yes", "annotations": []},
            {"type": "refusal", "refusal": "no"},
            {"type": "output_text", "text": " later", "annotations": []}
        ]
    });
    let events = [
        json!({
            "type": "response.output_text.delta", "item_id": "msg_1",
            "output_index": 0, "content_index": 0, "sequence_number": 1,
            "delta": "yes"
        }),
        json!({
            "type": "response.refusal.delta", "item_id": "msg_1",
            "output_index": 0, "content_index": 1, "sequence_number": 2,
            "delta": "no"
        }),
        json!({
            "type": "response.output_text.delta", "item_id": "msg_1",
            "output_index": 0, "content_index": 2, "sequence_number": 3,
            "delta": " later"
        }),
        json!({
            "type": "response.output_item.done", "output_index": 0,
            "sequence_number": 4, "item": message
        }),
        completed_with_output(json!([message])),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["text", "refusal", "text"]
    );
    assert_eq!(
        folded_texts(&driven.response.choice),
        ["yes", "no", " later"]
    );
    assert_eq!(driven.response.choice, unary_choice(json!([message])));
}

// F streaming/tests.rs:3955
#[test]
fn reversed_message_content_observations_use_the_content_coordinate() {
    let message = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "yes", "annotations": []},
            {"type": "refusal", "refusal": "no"}
        ]
    });
    let events = [
        json!({
            "type": "response.refusal.delta", "item_id": "msg_1",
            "output_index": 0, "content_index": 1, "sequence_number": 1,
            "delta": "no"
        }),
        json!({
            "type": "response.output_text.delta", "item_id": "msg_1",
            "output_index": 0, "content_index": 0, "sequence_number": 2,
            "delta": "yes"
        }),
        json!({
            "type": "response.output_item.done", "output_index": 0,
            "sequence_number": 3, "item": message
        }),
        completed_with_output(json!([message])),
    ];

    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["text", "refusal"]
    );
    assert_eq!(driven.response.choice, unary_choice(json!([message])));
}

// F streaming/tests.rs:3287
#[test]
fn refusal_parts_stream_into_their_own_marked_blocks() {
    let delta = |kind: &str, content_index: u64, sequence_number: u64, text: &str| {
        json!({
            "type": kind,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": content_index,
            "sequence_number": sequence_number,
            "delta": text
        })
    };
    let events = [
        delta("response.output_text.delta", 0, 1, "Hello"),
        delta("response.refusal.delta", 1, 2, "I can't"),
        delta("response.refusal.delta", 1, 3, " do that"),
        delta("response.output_text.delta", 0, 4, " there"),
        delta("response.refusal.delta", 2, 5, "Nor that"),
        json!({
            "type": "response.completed",
            "sequence_number": 6,
            "response": sample_response("completed"),
        }),
    ];
    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    let refusals = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedText(text),
                ..
            }) if matches!(&text.content, CarriedTextContent::Refusal(_)) => Some(*part),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(refusals.len(), 2, "one marked actual part per refusal");
    assert_ne!(refusals[0], refusals[1]);
    assert_eq!(
        folded_texts(&driven.response.choice),
        ["Hello there", "I can't do that", "Nor that"]
    );
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["text", "refusal", "refusal"]
    );
}

// F streaming/tests.rs:3367
#[test]
fn an_id_less_text_delta_after_a_message_done_never_extends_a_refusal() {
    let events = [
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "No"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "refusal", "refusal": "No" }]
            }
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "Hello"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response("completed"),
        }),
    ];
    let driven = drive(&events);
    assert!(driven.errors.is_empty());
    assert_eq!(folded_texts(&driven.response.choice), ["No", "Hello"]);
    assert_eq!(
        response_choice_labels(&driven.response.choice),
        ["refusal", "text"]
    );
}

// F streaming/tests.rs:3436
#[test]
fn opaque_output_items_join_the_answer_alike_unary_and_streamed() {
    let compaction = json!({"type":"compaction","id":"cmp_1","encrypted_content":"opaque"});
    let hosted = json!({"type":"web_search_call","status":"completed","action":{"query":"rig"}});
    let message = message_item(
        "msg_1",
        json!([{"type":"output_text","text":"hi","annotations":[]}]),
    );
    let items = [compaction.clone(), hosted.clone(), message];
    let unary = unary_choice(json!(items));
    let mut frames = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if index == 2 {
            frames.push(json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":2,"content_index":0,"sequence_number":3,"delta":"hi"}));
        }
        frames.push(json!({"type":"response.output_item.done","output_index":index,"sequence_number":index+1,"item":item}));
    }
    let mut completed = completed_with_output(json!([]));
    completed["sequence_number"] = json!(9);
    frames.push(completed);
    let driven = drive(&frames);
    assert!(driven.errors.is_empty());
    assert!(unknown_payloads(&driven.items).is_empty());
    assert!(driven.items.iter().any(|item|matches!(item,Item::Event(StreamEvent::Start {kind:PartKind::ProviderItem,source_order:Some(order),..}) if *order==SourceOrder::new(1,0))));
    assert_eq!(unary, driven.response.choice);
    let providers = driven
        .response
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ProviderItem(item) => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(providers.len(), 2);
    assert_eq!(providers[0].payload.item, compaction);
    assert_eq!(providers[1].payload.item, hosted);
    for item in providers {
        assert_eq!(item.payload.provider.as_deref(), Some("openai"));
    }
}

// F streaming/tests.rs:3652
#[test]
fn top_level_reasoning_fallback_uses_the_reasoning_namespace() {
    let mut whole = sample_response("completed");
    whole["reasoning"] = json!("think");
    let driven = drive(&[whole]);
    assert!(driven.errors.is_empty());
    assert!(driven.items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::Start {
            kind: PartKind::Reasoning,
            ..
        })
    )));
    assert_eq!(
        driven.response.choice,
        vec![AssistantContent::reasoning("think")]
    );
}

// F streaming/tests.rs:3745
#[test]
fn whole_response_replay_publishes_each_provider_item_once() {
    let output = json!([{"type":"web_search_call","status":"completed","action":{"query":"rig"}},{"type":"compaction","encrypted_content":"opaque"}]);
    let choice = unary_choice(output.clone());
    assert_eq!(response_choice_labels(&choice), ["provider", "provider"]);
    let mut whole = sample_response("completed");
    whole["output"] = output;
    let driven = drive(&[whole]);
    assert!(driven.errors.is_empty());
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ProviderItem(_),
                    ..
                })
            ))
            .count(),
        2
    );
    assert_eq!(driven.response.choice, choice);
}

// F streaming/tests.rs:3521
#[test]
fn anonymous_tool_and_reasoning_keys_do_not_collide_with_indexed_provider_items() {
    let provider = json!({"type":"web_search_call","status":"completed","action":{"query":"rig"}});
    let function = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"});
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think"}],"content":[],"status":"completed"});
    for (kind, second, delta) in [
        (
            PartKind::ToolCall,
            function,
            json!({"type":"response.function_call_arguments.delta","output_index":1,"sequence_number":1,"delta":"{}"}),
        ),
        (
            PartKind::Reasoning,
            reasoning,
            json!({"type":"response.reasoning_summary_text.delta","output_index":1,"summary_index":0,"sequence_number":1,"delta":"think"}),
        ),
    ] {
        let output = json!([provider, second]);
        let driven = drive(&[
            delta,
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":output[0]}),
            json!({"type":"response.output_item.done","output_index":1,"sequence_number":3,"item":output[1]}),
            completed_with_output(output.clone()),
        ]);
        assert!(driven.errors.is_empty());
        let part = driven
            .items
            .iter()
            .find_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part, kind: actual, ..
                }) if *actual == kind => Some(*part),
                _ => None,
            })
            .expect("anonymous slot opens its kind");
        let opaque = driven
            .items
            .iter()
            .find_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: PartKind::ProviderItem,
                    source_order: Some(order),
                    ..
                }) if *order == SourceOrder::new(0, 0) => Some(*part),
                _ => None,
            })
            .expect("provider item has output0 coordinate");
        assert_ne!(part, opaque, "distinct kinds never share an actual part");
        assert_eq!(
            response_choice_labels(&driven.response.choice),
            if kind == PartKind::ToolCall {
                ["provider", "function"]
            } else {
                ["provider", "reasoning"]
            }
        );
        assert_eq!(driven.response.choice, unary_choice(output));
    }
}
