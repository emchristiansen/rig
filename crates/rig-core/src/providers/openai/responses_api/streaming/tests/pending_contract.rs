//! Pending function names at actual stream/fold boundaries (A28).

use super::helpers::*;
use crate::message::AssistantContent;
use crate::streaming::{Item, StreamEvent, ToolSlotState};
use serde_json::json;

#[test]
fn an_unnamed_argument_prefix_binds_a_real_later_name_on_the_same_part_once() {
    let driven = drive(&[
        json!({"type":"response.function_call_arguments.delta","output_index":0,"sequence_number":0,"delta":"{\"x\":1}"}),
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":{
            "type":"function_call","id":"fc_1","call_id":"call_1","name":"add","arguments":"","status":"in_progress"}}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":{
            "type":"function_call","id":"fc_1","call_id":"call_1","name":"add","arguments":"{\"x\":1}","status":"completed"}}),
        completed_with_output(json!([])),
    ]);
    assert!(driven.errors.is_empty(), "{:?}", driven.errors);
    let starts = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start {
                part,
                name,
                tool_slot_state,
                ..
            }) => Some((
                *part,
                name.as_ref().map(|name| name.as_str()),
                *tool_slot_state,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].1, None);
    assert_eq!(starts[0].2, Some(ToolSlotState::PendingName));
    assert_eq!(starts[1].1, Some("add"));
    assert_eq!(starts[1].2, Some(ToolSlotState::Named));
    assert_eq!(
        starts[0].0, starts[1].0,
        "a real name binds the pending part"
    );
    let fragments = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Arguments { part, json }) => Some((*part, json.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fragments,
        [(starts[0].0, "{\"x\":1}")],
        "late name does not replay prior argument bytes"
    );
    let [AssistantContent::ToolCall(call)] = driven.response.choice.as_slice() else {
        panic!("one genuinely named call")
    };
    assert_eq!(call.function.name.as_str(), "add");
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
    assert!(driven.items.iter().any(|item|matches!(item,Item::Event(StreamEvent::End {part,content:AssistantContent::ToolCall(_),..}) if *part==starts[0].0)));
}

#[test]
fn unnamed_arguments_at_clean_eof_remain_partial_without_a_name_call_or_close() {
    let driven = drive(&[
        json!({"type":"response.function_call_arguments.delta","output_index":0,"sequence_number":0,"delta":"{}"}),
    ]);
    assert!(driven.errors.is_empty());
    let [
        Item::Event(StreamEvent::Start {
            part,
            name: None,
            tool_slot_state: Some(ToolSlotState::PendingName),
            ..
        }),
        Item::Event(StreamEvent::Arguments {
            part: argument_part,
            json,
        }),
    ] = driven.items.as_slice()
    else {
        panic!("pending prefix: {:?}", driven.items)
    };
    assert_eq!(part, argument_part);
    assert_eq!(json, "{}");
    assert!(driven.response.choice.is_empty());
    assert!(driven.response.response_id().is_none());
    assert!(!driven.items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::End { .. } | StreamEvent::Discard { .. })
    )));
    crate::streaming::Transcript::parse_prefix(serde_json::to_value(&driven.items).unwrap())
        .expect("the actual pending prefix stays valid");
}
