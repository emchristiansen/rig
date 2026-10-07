use serde_json::{Value, json};

use super::streaming::{ResponsesDecoder, ResponsesStreamOptions, document};
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::{AssistantContent, CarriedPayload, CarriedText, CarriedTextContent, Message};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::streaming::{Item, StreamEvent};
use crate::wire::{Mode, Operation, Wire, WireFrame};

fn response(item: Value) -> Value {
    json!({"id":"resp_parent_fields", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-5.4", "output":[item], "tools":[]})
}

fn message() -> Value {
    json!({"type":"message", "id":"msg_2", "role":"assistant", "status":"completed",
        "content":[{"type":"output_text", "text":"answer"}],
        "future_field":{"nested":[1, null, {"literal":"kept"}]}})
}

fn decode(frames: &[Value]) -> (CompletionResponse, Vec<StreamEvent>) {
    let decoded = crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        document::Response::with_strict_reader(false),
        "openai",
        frames.iter().map(|frame| WireFrame::Text(frame.to_string()))
    );
    let events = decoded.items.iter().filter_map(|item| {
        if let Item::Event(event) = item.as_ref().expect("the actual decoder emits no error") {
            Some(event.clone())
        } else {
            None
        }
    }).collect();
    (decoded.outcome.expect("the actual response decoder finishes"), events)
}

fn only_carried(response: &CompletionResponse) -> &CarriedText {
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("one carried message part: {:?}", response.choice);
    };
    text
}

fn encoded_messages(turn: Message) -> Vec<Value> {
    let wire = OpenAIConfig::new("test-key").responses("different-model");
    let request = CompletionRequest::from(vec![Message::user("question"), turn, Message::user("next")]);
    let request = Completion::prepare(request, &wire.describe()).expect("prepare actual history");
    let encoded = wire.encode(request, Mode::Unary).expect("actual Responses encoder");
    crate::test_utils::json_body(&encoded.request)["input"].as_array()
        .expect("input array").iter()
        .filter(|item| item["role"] == "assistant")
        .cloned().collect()
}

#[test]
fn unknown_parent_field_forces_carried_history_but_never_reaches_the_request() {
    let raw = message();
    let (decoded, _) = decode(&[response(raw.clone())]);
    let text = only_carried(&decoded);
    let expected = json!({"future_field": raw["future_field"]});
    assert_eq!(text.unknown_message_fields(), expected.as_object());
    assert_eq!(text.item_id.as_deref(), Some("msg_2"));
    assert_eq!(text.phase, None);
    assert_eq!(text.text(), "answer");
    assert!(decoded.choice[0].native_item().is_none());
    assert_eq!(decoded.choice[0].canonical(), decoded.choice[0]);
    let stored_part = serde_json::to_value(text).expect("carried history part");
    assert!(stored_part.get("native").is_none());
    assert!(stored_part.get("origin").is_none());

    let history = decoded.message().expect("decoded turn");
    let stored = serde_json::to_value(&history).expect("raw history");
    assert_eq!(stored["content"][0]["unknown_message_fields"], expected);
    let restored: Message = serde_json::from_value(stored).expect("restore raw history");
    assert_eq!(restored, history);
    assert_eq!(encoded_messages(restored), [json!({
        "type":"message", "role":"assistant", "status":"completed", "id":"msg_2",
        "content":[{"type":"output_text", "text":"answer"}]
    })]);
}

#[test]
fn streamed_parent_metadata_reaches_the_actual_end_and_folded_history() {
    let raw = message();
    let completed = json!({"type":"response.completed", "sequence_number":2,
        "response":response(raw.clone())});
    let (decoded, events) = decode(&[
        json!({"type":"response.output_text.delta", "item_id":"msg_2", "output_index":0,
            "content_index":0, "sequence_number":0, "delta":"answer"}),
        json!({"type":"response.output_item.done", "output_index":0, "sequence_number":1, "item":raw}),
        completed,
    ]);
    let text = only_carried(&decoded);
    let ends: Vec<_> = events.iter().filter_map(|event| {
        if let StreamEvent::End { part, content: AssistantContent::CarriedText(ended), .. } = event {
            Some((part, ended))
        } else {
            None
        }
    }).collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].1.unknown_message_fields(), text.unknown_message_fields());
    assert_eq!(ends[0].1.item_id, text.item_id);
    assert_eq!(ends[0].1.content, text.content);
    assert_eq!(ends[0].1.payload.item, text.payload.item);
    assert_eq!(ends[0].1.payload.position, text.payload.position);
    assert!(events.iter().all(|event| event.part() == *ends[0].0));
    let fragments: String = events.iter().filter_map(|event| {
        if let StreamEvent::Text { text, .. } = event { Some(text.as_str()) } else { None }
    }).collect();
    assert_eq!(fragments, "answer");
    let sent = encoded_messages(decoded.message().expect("decoded turn"));
    assert_eq!(sent[0]["id"], "msg_2");
    assert!(sent[0].get("future_field").is_none());
}

#[test]
fn recognized_parent_fields_and_text_edits_keep_their_own_authority() {
    let mut raw = message();
    raw["phase"] = json!("final_answer");
    raw["future_null"] = Value::Null;
    raw["future_empty"] = json!([]);
    let (mut decoded, _) = decode(&[response(raw.clone())]);
    let AssistantContent::CarriedText(text) = &mut decoded.choice[0] else {
        panic!("carried text");
    };
    let metadata = text.unknown_message_fields().expect("retained unknown parent fields").clone();
    assert_eq!(text.item_id.as_deref(), Some("msg_2"));
    assert_eq!(text.phase.as_deref(), Some("final_answer"));
    for recognized in ["type", "id", "role", "status", "content", "phase"] {
        assert!(!metadata.contains_key(recognized));
    }
    assert_eq!(metadata.get("future_null"), Some(&Value::Null));
    assert_eq!(metadata.get("future_empty"), Some(&json!([])));
    *text.text_mut() = "edited answer".to_owned();
    text.item_id = Some("msg_edited".to_owned());
    text.phase = Some("analysis".to_owned());
    assert_eq!(text.unknown_message_fields(), Some(&metadata));
    assert_eq!(text.payload.item["text"], "answer");
    assert!(matches!(&text.content, CarriedTextContent::Text(value) if value == "edited answer"));

    let stored = serde_json::to_value(decoded.message().expect("edited decoded turn")).expect("history");
    let restored: Message = serde_json::from_value(stored.clone()).expect("restore edited history");
    assert_eq!(stored["content"][0]["unknown_message_fields"], Value::Object(metadata));
    assert_eq!(encoded_messages(restored), [json!({
        "type":"message", "role":"assistant", "status":"completed", "id":"msg_edited",
        "phase":"analysis", "content":[{"type":"output_text", "text":"edited answer"}]
    })]);
}

#[test]
fn absent_parent_metadata_preserves_the_existing_history_bytes() {
    let legacy = br#"{"content":{"type":"text","text":"answer"},"payload":{"item":{"type":"output_text","text":"answer"}}}"#;
    let text: CarriedText = serde_json::from_slice(legacy).expect("existing carried text history");
    assert_eq!(text.unknown_message_fields(), None);
    assert_eq!(serde_json::to_vec(&text).expect("unchanged history bytes"), legacy.to_vec());
    let constructed = CarriedText::new(
        CarriedTextContent::Text("answer".to_owned()),
        CarriedPayload::new(json!({"type":"output_text", "text":"answer"}), None),
    );
    assert_eq!(serde_json::to_vec(&constructed).expect("default constructor bytes"), legacy.to_vec());
    let cleared = text.with_unknown_message_fields(serde_json::Map::new());
    assert_eq!(cleared.unknown_message_fields(), None);
    assert_eq!(serde_json::to_vec(&cleared).expect("empty metadata omitted"), legacy.to_vec());
}
