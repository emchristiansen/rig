//! Approved Gemini carried known-text projection at the actual foreign wires.
use super::*;
use crate::message::{CarriedPayload, CarriedPosition, CarriedText, CarriedTextContent,
    GeminiSignedAnswer, GeminiThoughtSignature};
use crate::wire::Operation;

fn carried(content: CarriedTextContent) -> CarriedText {
    CarriedText::new(content, CarriedPayload::new(
        json!({"text":"original", "thoughtSignature":"raw-secret-signature",
            "inlineData":{"mimeType":"image/png","data":"raw-media"},
            "future":{"nested":[null,7]}}), Some("gemini".to_owned()))
        .with_position(CarriedPosition::MessageContent { item_index:0, part_index:1 }))
}
fn history(text: CarriedText) -> CompletionRequest {
    let message = Message::Assistant(AssistantMessage {
        content:vec![AssistantContent::CarriedText(text)], ..Default::default()
    });
    let stored = serde_json::to_value(message).expect("history serialization");
    let restored: Message = serde_json::from_value(stored).expect("history restoration");
    CompletionRequest::from(vec![Message::user("first"),restored,Message::user("next")])
}
fn chat(text: CarriedText) -> Value {
    let wire = OpenAIConfig::new("test-key").chat("different-model");
    let request = Completion::prepare(history(text),&wire.describe()).expect("Chat prepare");
    let encoded = wire.encode(request,Mode::Unary).expect("Chat encode");
    crate::test_utils::json_body(&encoded.request)
}
fn responses(text: CarriedText) -> Value {
    let wire = OpenAIConfig::new("test-key").responses("different-model");
    let request = Completion::prepare(history(text),&wire.describe()).expect("Responses prepare");
    let encoded = wire.encode(request,Mode::Unary).expect("Responses encode");
    crate::test_utils::json_body(&encoded.request)
}
fn assistant_items<'a>(body:&'a Value,key:&str)->Vec<&'a Value> {
    body[key].as_array().expect("actual request history").iter()
        .filter(|item| item["role"]=="assistant").collect()
}
#[test]
fn edited_signed_answer_projects_known_text_without_signature_or_inspection_fields() {
    let mut text=carried(CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
        "original",GeminiThoughtSignature::new("typed-signature"))));
    text.text_mut().clear(); text.text_mut().push_str("edited answer");
    let chat=chat(text.clone());
    assert_eq!(assistant_items(&chat,"messages"),[&json!({"role":"assistant",
        "content":[{"type":"text","text":"edited answer"}]})]);
    let responses=responses(text);
    assert_eq!(assistant_items(&responses,"input"),[&json!({"type":"message","role":"assistant",
        "content":"edited answer"})]);
}
#[test]
fn empty_inspection_marker_keeps_chat_slot_and_is_omitted_by_responses() {
    let text=carried(CarriedTextContent::InspectionOnlyMedia(String::new()));
    assert_eq!(assistant_items(&chat(text.clone()),"messages"),[&json!({"role":"assistant",
        "content":[{"type":"text","text":""}]})]);
    assert!(assistant_items(&responses(text),"input").is_empty());
}
#[test]
fn edited_inspection_marker_uses_current_text_and_governed_containing_id() {
    let mut text=carried(CarriedTextContent::InspectionOnlyMedia("edited marker".to_owned()));
    text.item_id=Some("msg_existing".to_owned());
    text=text.with_unknown_message_fields(json!({"future_field":{"raw":true}})
        .as_object().expect("inspection object").clone());
    assert_eq!(assistant_items(&chat(text.clone()),"messages"),[&json!({"role":"assistant",
        "content":[{"type":"text","text":"edited marker"}]})]);
    assert_eq!(assistant_items(&responses(text),"input"),[&json!({"type":"message",
        "role":"assistant","status":"completed","id":"msg_existing",
        "content":[{"type":"output_text","text":"edited marker","annotations":[]}]})]);
}
#[test]
fn empty_known_marker_with_containing_id_still_has_no_responses_outbound_item() {
    let mut text=carried(CarriedTextContent::InspectionOnlyMedia(String::new()));
    text.item_id=Some("msg_existing".to_owned());
    assert!(assistant_items(&responses(text),"input").is_empty());
}
