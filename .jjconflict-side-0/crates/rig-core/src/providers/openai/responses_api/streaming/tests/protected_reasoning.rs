//! Protected known reasoning, provider item and incomplete stream boundaries.

use super::helpers::*;
use super::super::IncompleteTerminal;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::streaming::{Item, PartKind, ReasoningAuthority, SourceOrder, StreamEvent};
use serde_json::json;

fn done(item: serde_json::Value) -> serde_json::Value {
    json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item})
}

fn reasoning_ends(items: &[Item<StreamEvent>]) -> Vec<(&crate::message::CarriedReasoning, Option<ReasoningAuthority>)> {
    items.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::End { content: AssistantContent::CarriedReasoning(reasoning), reasoning_authority, .. }) => Some((reasoning, *reasoning_authority)),
        _ => None,
    }).collect()
}

// F streaming/tests.rs:455
#[test]
fn reasoning_done_item_fuses_summary_content_and_encrypted_into_one_end() {
    let driven = drive(&[done(json!({"type":"reasoning","id":"rs_1",
        "summary":[{"type":"summary_text","text":"step 1"},{"type":"summary_text","text":"step 2"}],
        "content":[{"type":"reasoning_text","text":"private reasoning"}],"encrypted_content":"enc_blob"}))]);
    assert!(driven.errors.is_empty(), "{:?}", driven.errors);
    let ends = reasoning_ends(&driven.items);
    assert_eq!(ends.len(), 1, "one restatement contains every block");
    let (reasoning, authority) = ends[0];
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(reasoning.content, vec![
        CarriedReasoningPart::Summary("step 1".into()), CarriedReasoningPart::Summary("step 2".into()),
        CarriedReasoningPart::Text { text: "private reasoning".into(), signature: None },
        CarriedReasoningPart::Encrypted("enc_blob".into()),
    ]);
    assert_eq!(authority, Some(ReasoningAuthority::ProviderRestated));
}

// F streaming/tests.rs:494
#[test]
fn reasoning_output_item_done_emits_reasoning_text_content() {
    let driven = drive(&[done(json!({"type":"reasoning","id":"rs_text_1","summary":[],
        "content":[{"type":"reasoning_text","text":"visible reasoning"}],"status":"completed"}))]);
    assert!(driven.errors.is_empty());
    let start = driven.items.iter().find_map(|item| match item {
        Item::Event(StreamEvent::Start { part, kind: PartKind::Reasoning, .. }) => Some(*part),
        _ => None,
    }).expect("the done item opens its actual reasoning part");
    let ends: Vec<_> = driven.items.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::End { part, content: AssistantContent::CarriedReasoning(reasoning), reasoning_authority, .. }) => Some((*part, reasoning, *reasoning_authority)),
        _ => None,
    }).collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0, start);
    assert_eq!(ends[0].1.id.as_deref(), Some("rs_text_1"));
    assert_eq!(ends[0].1.content, [CarriedReasoningPart::Text { text: "visible reasoning".into(), signature: None }]);
    assert_eq!(ends[0].2, Some(ReasoningAuthority::ProviderRestated));
}

// F streaming/tests.rs:553
#[test]
fn envelope_less_reasoning_then_text_decodes_without_violation() {
    let driven = drive(&[
        json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"sequence_number":1,"delta":"thinking"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":{
            "type":"reasoning","id":"","summary":[{"type":"summary_text","text":"thinking, complete"}],"status":"completed"}}),
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":3,"delta":"the answer"}),
    ]);
    assert!(driven.errors.is_empty(), "{:?}", driven.errors);
    assert!(driven.items.iter().any(|item| matches!(item, Item::Event(StreamEvent::Text { text, .. }) if text == "the answer")));
}

// F streaming/tests.rs:593
#[test]
fn reasoning_text_delta_emits_reasoning_delta() {
    let driven = drive(&[json!({"type":"response.reasoning_text.delta","item_id":"rs_delta_1",
        "output_index":0,"content_index":0,"sequence_number":1,"delta":"thinking"})]);
    assert!(driven.errors.is_empty());
    let Some(Item::Event(StreamEvent::Start { part, kind: PartKind::Reasoning, .. })) = driven.items.first() else {
        panic!("the first delta opens a reasoning part: {:?}", driven.items);
    };
    assert!(matches!(driven.items.get(1), Some(Item::Event(StreamEvent::Reasoning { part: actual, text })) if actual == part && text == "thinking"));
    assert_eq!(reasoning_items(&driven.response.choice)[0].id.as_deref(), Some("rs_delta_1"));
}

// F streaming/tests.rs:629
#[test]
fn unknown_output_item_surfaces_as_a_verbatim_provider_item_block() {
    let item = json!({"type":"web_search_call","id":"ws_001","status":"completed",
        "action":{"type":"search","queries":["rig framework"]}});
    let driven = drive(&[done(item.clone())]);
    assert!(driven.errors.is_empty());
    assert!(unknown_payloads(&driven.items).is_empty(), "a carried provider item is not an unknown event");
    let part = driven.items.iter().find_map(|event| match event {
        Item::Event(StreamEvent::Start { part, kind: PartKind::ProviderItem, source_order: Some(order), .. })
            if *order == SourceOrder::new(0, 0) => Some(*part),
        _ => None,
    }).expect("provider item opens at its real source coordinate");
    let closed = driven.items.iter().find_map(|event| match event {
        Item::Event(StreamEvent::End { part: actual, content: AssistantContent::ProviderItem(value), .. }) if *actual == part => Some(value),
        _ => None,
    }).expect("same actual part closes");
    assert_eq!(closed.item, item, "verbatim whole item reaches the consumer");
}

// F streaming/tests.rs:683
#[test]
fn reasoning_done_item_without_encrypted_emits_summary_only() {
    let driven = drive(&[done(json!({"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"only summary"}]}))]);
    assert!(driven.errors.is_empty());
    let ends = reasoning_ends(&driven.items);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0.id.as_deref(), Some("rs_2"));
    assert_eq!(ends[0].0.content, [CarriedReasoningPart::Summary("only summary".into())]);
}

// F streaming/tests.rs:700
#[test]
fn empty_encrypted_reasoning_is_not_emitted() {
    let driven = drive(&[done(json!({"type":"reasoning","id":"rs_1","summary":[],
        "content":[{"type":"reasoning_text","text":"visible reasoning"}],"encrypted_content":""}))]);
    assert!(driven.errors.is_empty());
    let ends = reasoning_ends(&driven.items);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0.content, [CarriedReasoningPart::Text { text: "visible reasoning".into(), signature: None }]);
    let empty = drive(&[done(json!({"type":"reasoning","id":"rs_1","summary":[],"content":[],"encrypted_content":""}))]);
    assert!(empty.errors.is_empty());
    assert!(reasoning_ends(&empty.items).is_empty(), "an entirely empty done item says nothing at the boundary");
}

// F streaming/tests.rs:885
#[tokio::test]
async fn an_opted_in_response_incomplete_chunk_is_a_terminal_with_mapped_finish_reason() {
    let mut events = incomplete_turn_events();
    events[1]["response"]["usage"] = json!({"input_tokens":10,"input_tokens_details":null,
        "output_tokens":5,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":15});
    let response = finish_events(&events, IncompleteTerminal::Accept).await;
    assert_eq!(folded_texts(&response.choice), ["partial"]);
    assert_eq!(response.finish_reason(), Some(crate::completion::FinishReason::Length));
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

// F streaming/tests.rs:963
#[test]
fn multi_block_reasoning_done_item_yields_one_part() {
    let item = json!({"type":"reasoning","id":"rs_1","summary":[
        {"type":"summary_text","text":"step 1"},{"type":"summary_text","text":"step 2"}],
        "content":[],"encrypted_content":"enc_blob"});
    let driven = drive(&[done(item), completed_with_output(json!([]))]);
    assert!(driven.errors.is_empty());
    let ends = reasoning_ends(&driven.items);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0.id.as_deref(), Some("rs_1"));
    assert_eq!(ends[0].0.content, vec![
        CarriedReasoningPart::Summary("step 1".into()), CarriedReasoningPart::Summary("step 2".into()),
        CarriedReasoningPart::Encrypted("enc_blob".into()),
    ]);
    assert_eq!(reasoning_items(&driven.response.choice).len(), 1, "canonical history carries one reasoning item for rs_1");
}
