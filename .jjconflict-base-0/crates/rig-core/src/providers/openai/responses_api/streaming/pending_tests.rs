use super::{ResponsesDecoder, ResponsesStreamOptions};
use crate::driver::Decoded;
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::streaming::{Item, PartKind, SequenceError, StreamEvent, ToolSlotState, Transcript};
use crate::wire::WireFrame;
use serde_json::{Value, json};

fn delta(sequence: u64, text: &str) -> Value {
    json!({"type":"response.function_call_arguments.delta","sequence_number":sequence,
        "output_index":0,"item_id":"fc_1","delta":text})
}

fn call(kind: &str, sequence: u64) -> Value {
    json!({"type":kind,"sequence_number":sequence,"output_index":0,
        "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup",
            "status":"completed","arguments":"{\"x\":1}"}})
}

fn terminal(output: Value) -> Value {
    json!({"type":"response.completed","sequence_number":10,
        "response":{"id":"resp_1","object":"response","created_at":1,"model":"gpt-5.4",
            "status":"completed","output":output}})
}

fn decode(frames: &[Value]) -> Decoded<Completion> {
    crate::driver::feed_frames!(ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()), "openai",
        frames.iter().map(|frame| WireFrame::Text(frame.to_string())))
}

fn prefix(result: &Decoded<Completion>) -> Transcript {
    let items = result.items.iter().filter_map(|item| item.as_ref().ok().cloned()).collect::<Vec<_>>();
    assert!(items.iter().all(|item| !matches!(item, Item::Unknown(_))), "recognized argument deltas stay on the event channel");
    Transcript::from_items(items).expect("actual decoder grammar")
}

#[test]
fn nameless_delta_is_a_real_pending_part_before_any_provider_name_arrives() {
    let result = decode(&[delta(0, "{\"x\":")]);
    assert!(result.outcome.is_err(), "EOF is not a provider terminal");
    let transcript = prefix(&result);
    let events = transcript.events().collect::<Vec<_>>();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], StreamEvent::Start { kind: PartKind::ToolCall, name: None, tool_slot_state: Some(ToolSlotState::PendingName), .. }));
    assert!(matches!(events[1], StreamEvent::Arguments { json, .. } if json == "{\"x\":"));
    assert_eq!(events[0].part(), events[1].part());
    assert!(matches!(Transcript::parse(serde_json::to_value(transcript).expect("actual prefix")), Err(SequenceError::Unclosed(_))));
}

#[test]
fn genuine_added_or_done_name_binds_the_same_part_and_keeps_original_argument_fragments() {
    for added_first in [true, false] {
        let mut frames = vec![delta(0, "{\"x\":")];
        if added_first { frames.push(call("response.output_item.added", 1)); }
        frames.push(delta(2, "1}"));
        frames.push(call("response.output_item.done", 3));
        frames.push(terminal(json!([])));
        let result = decode(&frames);
        let transcript = prefix(&result);
        let events = transcript.events().collect::<Vec<_>>();
        let starts = events.iter().filter(|event| matches!(event, StreamEvent::Start { .. })).collect::<Vec<_>>();
        assert_eq!(starts.len(), 2);
        assert!(matches!(starts[0], StreamEvent::Start { name: None, tool_slot_state: Some(ToolSlotState::PendingName), .. }));
        assert!(matches!(starts[1], StreamEvent::Start { name: Some(name), tool_slot_state: Some(ToolSlotState::Named), .. } if name.as_str() == "lookup"));
        assert!(events.iter().all(|event| event.part() == starts[0].part()));
        let fragments = events.iter().filter_map(|event| match event {
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(fragments, vec!["{\"x\":", "1}"], "binding introduces no synthetic argument delta");
        assert_eq!(events.iter().filter(|event| matches!(event, StreamEvent::End { .. })).count(), 1);
        Transcript::parse(serde_json::to_value(&transcript).expect("actual complete stream")).expect("one real Part closes");
        let response = result.outcome.expect("genuine named terminal");
        let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else { panic!("one canonical named call"); };
        assert_eq!(call.function.name.as_str(), "lookup");
        assert_eq!(call.id.wire(), "call_1");
        assert_eq!(call.function.arguments_value(), json!({"x":1}));
    }
}

#[test]
fn pending_terminal_keeps_arguments_partial_without_an_end_discard_or_canonical_call() {
    let answer = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed",
        "content":[{"type":"output_text","text":"answer","annotations":[]}]});
    for arguments in ["{\"x\":1}", "{", "null"] {
        let result = decode(&[delta(0, arguments), terminal(json!([answer.clone()]))]);
        let transcript = prefix(&result);
        let part = transcript.events().find(|event| matches!(event, StreamEvent::Start { tool_slot_state: Some(ToolSlotState::PendingName), .. })).expect("pending actual Part").part();
        assert!(transcript.events().filter(|event| event.part() == part).all(|event| !matches!(event, StreamEvent::End { .. } | StreamEvent::Discard { .. })));
        assert!(matches!(Transcript::parse(serde_json::to_value(&transcript).expect("provider terminal with partial slot")), Err(SequenceError::Unclosed(_))));
        let response = result.outcome.expect("terminal still carries its actual answer");
        assert!(response.choice.iter().all(|content| !matches!(content, AssistantContent::ToolCall(_))));
    }
}

#[test]
fn preserved_pending_null_prefixes_are_real_deltas_and_gateway_replacement_keeps_the_named_value() {
    let result = decode(&[delta(0, "n"), delta(1, "ull"), delta(2, "{\"x\":1}"),
        call("response.output_item.done", 3), terminal(json!([]))]);
    let transcript = prefix(&result);
    let fragments = transcript.events().filter_map(|event| match event {
        StreamEvent::Arguments { json, .. } => Some(json.as_str()),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(fragments, vec!["n", "ull", "{\"x\":1}"]);
    let response = result.outcome.expect("genuine named done and terminal");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else { panic!("actual named call"); };
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
}
