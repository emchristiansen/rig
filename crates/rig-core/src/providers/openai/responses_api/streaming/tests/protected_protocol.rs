//! Protected repaired envelopes, raw call fragments and atomic kind refusal.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedTextContent};
use crate::streaming::{Item, PartKind, StreamEvent};
use serde_json::json;

// F streaming/tests.rs:1439
#[test]
fn envelope_less_frames_repair_onto_the_shared_interpreter() {
    let completed = json!({"type":"response.completed","response":sample_response("completed")});
    let text = decode_repaired(&[
        json!({"type":"response.output_text.delta","delta":"hi"}),
        completed.clone(),
    ]);
    assert!(
        text.events()
            .iter()
            .any(|event| matches!(event,StreamEvent::Text {text,..} if text=="hi"))
    );
    text.outcome
        .expect("repaired text reaches genuine terminal");
    let args = decode_repaired(&[
        json!({"type":"response.function_call_arguments.delta","delta":"{}"}),
        completed.clone(),
    ]);
    assert!(
        args.events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Arguments { .. })),
        "A28 preserves the unnamed repaired argument observation"
    );
    args.outcome
        .expect("repaired arguments are not a decode failure");
    let bookkeeping = decode_repaired(&[
        json!({"type":"response.output_text.done","text":"hi"}),
        completed.clone(),
    ]);
    assert_eq!(
        bookkeeping
            .outcome
            .expect("done bookkeeping remains a no-op")
            .response_id(),
        Some("resp_123")
    );
    let reasoning = decode_repaired(&[
        json!({"type":"response.reasoning_summary_text.delta","delta":"think"}),
        completed,
    ]);
    assert!(
        reasoning
            .events()
            .iter()
            .any(|event| matches!(event,StreamEvent::Reasoning {text,..} if text=="think"))
    );
    reasoning
        .outcome
        .expect("repaired reasoning reaches genuine terminal");
}

// F streaming/tests.rs:1513
#[test]
fn an_unparseable_restatement_is_not_reemitted_over_streamed_fragments() {
    let driven = drive(&[
        json!({"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"sequence_number":1,"delta":"{\"x\":481"}),
        malformed_function_call_done("incomplete", 2),
    ]);
    assert!(driven.errors.is_empty());
    let fragments = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Arguments { json, .. }) => Some(json.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fragments,
        ["{\"x\":481"],
        "the authoritative malformed restatement does not reemit prior bytes"
    );
}

// F streaming/tests.rs:1563
#[test]
fn a_fragmentless_unparseable_restatement_keeps_raw_and_is_discarded_as_incomplete() {
    // G186: this exact incomplete fixture is discarded, not a live Arguments event.
    let frame = malformed_function_call_done("incomplete", 1);
    let driven = drive(std::slice::from_ref(&frame));
    assert!(driven.errors.is_empty());
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(item, Item::Event(StreamEvent::Arguments { .. })))
            .count(),
        0,
        "restatement does not fabricate a streamed fragment"
    );
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::Discard {
                    reason: crate::streaming::DiscardReason::Incomplete,
                    ..
                })
            ))
            .count(),
        1
    );
    assert!(!driven.items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(_),
            ..
        })
    )));
    assert_eq!(driven.response.raw["output"][0], frame["item"]);
}

// F streaming/tests.rs:4456; A29 replaces implicit closure with one real close.
#[test]
fn unchanged_message_snapshots_do_not_reopen_text_blocks() {
    let item = opaque_message_fixture(json!([{"type":"output_text","text":"A"}]));
    let driven = drive(&[
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        completed_with_output(json!([item])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::Start {
                    kind: PartKind::Text,
                    ..
                })
            ))
            .count(),
        1
    );
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(item, Item::Event(StreamEvent::Resume { .. })))
            .count(),
        0,
        "unchanged snapshots never reactivate the part"
    );
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(item,Item::Event(StreamEvent::Text {text,..}) if text=="A"))
            .count(),
        1,
        "no duplicate fragment"
    );
    assert_eq!(
        driven
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::Text(_) | AssistantContent::CarriedText(_),
                    ..
                })
            ))
            .count(),
        1,
        "one canonical text close replaces implicit old accumulator closure"
    );
    assert_eq!(folded_texts(&driven.response.choice), ["A"]);
    // G166: canonical reconstruction adds exactly this empty annotations member.
    let mut expected = item["content"].clone();
    expected[0]["annotations"] = json!([]);
    assert_eq!(
        replay_opaque_choice(driven.response.choice)[0]["content"],
        expected
    );
}

// F streaming/tests.rs:4687
#[test]
fn a_part_restated_as_the_other_kind_is_refused_before_any_mutation() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let opaque_first = json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    let text_first = json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"});
    let added = json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":opaque_message_fixture(json!([]))});
    for (index,(frames,stored_opaque)) in [
        (vec![added.clone(),opaque_first.clone(),json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":2,"delta":"late"})],true),
        (vec![added.clone(),opaque_first.clone(),json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":2,"part":{"type":"output_text","text":"late"}})],true),
        (vec![added,opaque_first,completed_with_output(json!([opaque_message_fixture(json!([{"type":"output_text","text":"late"}]))]))],true),
        (vec![text_first,json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":1,"part":unknown})],false),
    ].into_iter().enumerate(){
        let driven=drive(&frames);
        assert_eq!(driven.errors.len(),1,"case {index}");
        assert!(driven.errors[0].to_string().contains("conflicting Responses content part kind"));
        assert!(driven.response.response_id().is_none(),"kind conflict proves no terminal");
        let texts=driven.items.iter().filter_map(|item|match item{Item::Event(StreamEvent::Text {text,..})=>Some(text.as_str()),_=>None}).collect::<Vec<_>>();
        if stored_opaque{
            assert!(texts.is_empty(),"no known text appended beside opaque");
            assert_eq!(opaque_texts(&driven.response.choice),[&unknown],"exact opaque marker retained");
        }else{
            assert_eq!(texts,["A"]);
            assert!(opaque_texts(&driven.response.choice).is_empty(),"known text gains no opaque marker");
            assert!(driven.response.choice.iter().all(|part| !matches!(part,AssistantContent::CarriedText(text) if matches!(text.content,CarriedTextContent::Opaque(_)))));
        }
    }
}
