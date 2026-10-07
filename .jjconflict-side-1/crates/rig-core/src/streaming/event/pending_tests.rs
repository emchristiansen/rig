use super::{Item, SequenceError, StreamEvent, ToolSlotState, Transcript};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::operation::{Block, CallFragment, Completion, Turn};
use crate::wire::Out;
use serde_json::{Value, json};

fn write(action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>) -> Decoded<Completion> {
    decode_with(Turn::relayed("pending"), "pending", |reply| {
        action(&mut reply.out())?;
        Err(ProviderError::Truncated)
    })
}

fn items(decoded: &Decoded<Completion>) -> Vec<Item<StreamEvent>> {
    decoded.items.iter().filter_map(|item| item.as_ref().ok().cloned()).collect()
}

fn bound() -> Decoded<Completion> {
    write(|out| {
        out.pending_call(7, CallFragment { id: Some("early-id"), arguments: Some("{"), ..CallFragment::default() })?;
        out.fragment(Some(7), CallFragment { arguments: Some("\"x\":1}"), ..CallFragment::default() })?;
        out.fragment(Some(7), CallFragment { id: Some("call_1"), name: Some("lookup"), arguments: None })?;
        out.finish(7)
    })
}

#[test]
fn real_pending_arguments_bind_once_on_the_original_part_without_replaying_a_delta() {
    let result = bound();
    let transcript = Transcript::from_items(items(&result)).expect("real writer grammar");
    let events = transcript.events().collect::<Vec<_>>();
    assert_eq!(events.len(), 5);
    assert!(matches!(events[0], StreamEvent::Start { name: None, tool_slot_state: Some(ToolSlotState::PendingName), .. }));
    assert!(matches!(events[1], StreamEvent::Arguments { json, .. } if json == "{"));
    assert!(matches!(events[2], StreamEvent::Arguments { json, .. } if json == "\"x\":1}"));
    assert!(matches!(events[3], StreamEvent::Start { name: Some(name), tool_slot_state: Some(ToolSlotState::Named), .. } if name.as_str() == "lookup"));
    assert!(events.iter().all(|event| event.part() == events[0].part()));
    let StreamEvent::End { content: AssistantContent::ToolCall(call), .. } = events[4] else { panic!("genuine named call"); };
    assert_eq!(call.id.wire(), "call_1");
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
    Transcript::parse(serde_json::to_value(transcript).expect("actual events")).expect("bound Part closes once");
}

#[test]
fn pending_prefix_carries_null_token_fragments_and_never_fabricates_an_end() {
    let result = write(|out| {
        out.pending_call(7, CallFragment { arguments: Some("n"), ..CallFragment::default() })?;
        out.fragment(Some(7), CallFragment { arguments: Some("ull"), ..CallFragment::default() })?;
        out.close(7)
    });
    let encoded = serde_json::to_value(items(&result)).expect("actual writer prefix");
    let transcript = Transcript::parse_prefix(encoded.clone()).expect("pending is a valid partial stream");
    let fragments = transcript.events().filter_map(|event| match event {
        StreamEvent::Arguments { json, .. } => Some(json.as_str()),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(fragments, vec!["n", "ull"]);
    assert!(transcript.events().all(|event| !matches!(event, StreamEvent::End { .. } | StreamEvent::Discard { .. })));
    assert!(matches!(Transcript::parse(encoded), Err(SequenceError::Unclosed(_))));
}

#[test]
fn transcript_refuses_second_binding_wrong_part_and_pending_closure() {
    let encoded = serde_json::to_value(items(&bound())).expect("actual bound writer stream");
    let original = encoded.as_array().expect("items");
    let named = original[3].clone();
    let mut repeated = original.clone();
    repeated.insert(4, named.clone());
    assert!(matches!(Transcript::parse_prefix(Value::Array(repeated)), Err(SequenceError::ToolSlotTransition(_))));
    let mut wrong_part = original.clone();
    wrong_part[3]["value"]["part"] = json!(99);
    assert!(matches!(Transcript::parse_prefix(Value::Array(wrong_part)), Err(SequenceError::ToolSlotTransition(_))));
    let mut premature = original[..3].to_vec();
    premature.push(original[4].clone());
    assert!(matches!(Transcript::parse_prefix(Value::Array(premature)), Err(SequenceError::ToolSlotTransition(_))));
    let mut pending_input = original[..3].to_vec();
    pending_input[1]["value"]["event"] = json!("input");
    pending_input[1]["value"]["input"] = json!("{");
    assert!(matches!(Transcript::parse_prefix(Value::Array(pending_input)), Err(SequenceError::WrongKind(_))));
}

#[test]
fn default_call_starts_still_wait_for_a_name_and_serialize_identically() {
    let result = write(|out| {
        out.fragment(Some(7), CallFragment { arguments: Some("{\"x\":1}"), ..CallFragment::default() })?;
        out.fragment(Some(7), CallFragment { id: Some("call_1"), name: Some("lookup"), arguments: None })?;
        out.finish(7)
    });
    let transcript = Transcript::from_items(items(&result)).expect("default grammar");
    let events = transcript.events().collect::<Vec<_>>();
    assert_eq!(events.len(), 3);
    assert_eq!(serde_json::to_string(events[0]).expect("old Start bytes"), r#"{"event":"start","part":0,"kind":"tool_call","name":"lookup"}"#);
    assert!(matches!(events[1], StreamEvent::Arguments { json, .. } if json == "{\"x\":1}"));
    let mut encoded = serde_json::to_value(&transcript).expect("actual default stream");
    encoded[0]["value"].as_object_mut().expect("Start").remove("name");
    assert!(matches!(Transcript::parse_prefix(encoded), Err(SequenceError::WrongKind(_))));
    let text = write(|out| out.whole(9, Block::Text, Value::Null, "plain"));
    let mut encoded = serde_json::to_value(items(&text)).expect("actual text stream");
    encoded[0]["value"]["tool_slot_state"] = json!("pending_name");
    assert!(matches!(Transcript::parse_prefix(encoded), Err(SequenceError::WrongKind(_))));
}
