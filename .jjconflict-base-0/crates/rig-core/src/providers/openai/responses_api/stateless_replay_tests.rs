//! Stateless Responses items through actual decoding, history and wire replay.

use crate::completion::{self, CompletionRequest};
use crate::message::{self, AssistantContent, AssistantMessage, CarriedTextContent, Message};
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Mode, Wire, WireEvent, WireFrame};
use serde_json::{Value, json};

use super::streaming::{ResponsesEvent, classify_responses_payload};
use super::wire::Responses;

fn wire() -> Responses {
    OpenAIConfig::new("fixture-key").responses("gpt-5.4")
}

fn response_body(output: Vec<Value>) -> Value {
    json!({"id":"resp_1","object":"response","created_at":1,"status":"completed",
        "model":"gpt-5.4","output":output})
}

fn decoded_output(output: Vec<Value>) -> completion::CompletionResponse {
    let body = response_body(output);
    crate::test_utils::decode_reply(
        &wire(),
        &CompletionRequest::new("fixture"),
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
        Value::Null,
    )
    .expect("the actual Responses decoder and fold accept the fixture")
}

fn classified_output(item: Value) -> Value {
    let body = response_body(vec![item]);
    let WireEvent::Known(ResponsesEvent::Whole(body)) =
        classify_responses_payload(&body.to_string())
    else {
        panic!("the actual strict Responses reader must accept the fixture");
    };
    body["output"][0].clone()
}

fn replayed_input(history: Message) -> Vec<Value> {
    let encoded = wire()
        .encode(CompletionRequest::new(history), Mode::Unary)
        .expect("the actual Responses input encoder accepts the history");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .expect("the encoded input is an array")
        .clone()
}

fn decoded_history(output: Vec<Value>) -> Message {
    decoded_output(output)
        .message()
        .expect("actual decoding supplies the assistant history")
}

#[test]
fn compaction_output_item_round_trips_verbatim() {
    let original = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes",
        "status": "completed",
        "future_field": {"nested": [1, 2, 3]}
    });
    let response = decoded_output(vec![original.clone()]);
    let [AssistantContent::ProviderItem(item)] = response.choice.as_slice() else {
        panic!("expected the actual decoder's compaction carrier");
    };
    let fields = item
        .payload
        .item
        .as_object()
        .expect("whole compaction object");
    assert_eq!(fields.get("id"), Some(&json!("cmp_123")));
    assert_eq!(fields.get("type"), Some(&json!("compaction")));
    assert_eq!(item.payload.item, original);
    let back = replayed_input(response.message().expect("decoded assistant history"));
    assert_eq!(back, [original]);
}

#[test]
fn compaction_input_item_round_trips_verbatim() {
    let original = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes"
    });
    let response = decoded_output(vec![original.clone()]);
    assert!(matches!(
        response.choice.as_slice(),
        [AssistantContent::ProviderItem(_)]
    ));
    let back = replayed_input(response.message().expect("decoded assistant history"));
    assert_eq!(back, [original]);
}

#[test]
fn compacted_window_decodes_with_every_item_typed() {
    let window = vec![
        json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "..."}),
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
            "arguments": "{}", "status": "completed"}),
    ];
    let response = decoded_output(window.clone());
    assert!(matches!(
        response.choice[0],
        AssistantContent::ProviderItem(_)
    ));
    assert!(matches!(
        response.choice[1],
        AssistantContent::Text(_) | AssistantContent::CarriedText(_)
    ));
    assert!(matches!(response.choice[2], AssistantContent::ToolCall(_)));
    assert_eq!(response.raw["output"], json!(window));
}

#[test]
fn output_message_phase_decodes_and_is_absent_by_default() {
    let with = classified_output(json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [], "phase": "final_answer"
    }));
    assert_eq!(with["phase"].as_str(), Some("final_answer"));
    let without = classified_output(json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": []
    }));
    assert_eq!(without.get("phase"), None);
    assert!(
        without.get("phase").is_none(),
        "absent phase must not become null"
    );
}

#[test]
fn phase_survives_history_and_is_resent_on_the_assistant_item() {
    let history = decoded_history(vec![json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type":"output_text","text":"the answer"}], "phase": "final_answer"
    })]);
    let items = replayed_input(history);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["id"], "msg_1");
    assert_eq!(items[0]["phase"].as_str(), Some("final_answer"));
    let block = &items[0]["content"][0];
    assert!(
        block.get("phase").is_none(),
        "phase leaked onto the text block: {block}"
    );
    assert_eq!(items[0]["phase"], "final_answer");
    assert_eq!(items[0]["id"], "msg_1");
}

#[test]
fn history_without_phase_replays_without_the_key() {
    let history = Message::Assistant(AssistantMessage::new(vec![AssistantContent::text("plain")]));
    let items = replayed_input(history);
    assert!(items[0].get("phase").is_none(), "{}", items[0]);
}

#[test]
fn a_refusal_part_stays_distinct_in_history_and_replays_as_a_refusal() {
    let response = decoded_output(vec![json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [
            {"type":"output_text","text":"Here is part of it."},
            {"type":"refusal","refusal":"I can't help with the rest."}
        ]
    })]);
    let texts: Vec<_> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::CarriedText(text) => text,
            other => panic!("expected carried text blocks, got {other:?}"),
        })
        .collect();
    assert_eq!(texts.len(), 2, "the refusal is not merged into the text");
    assert_eq!(texts[0].text(), "Here is part of it.");
    assert!(!matches!(&texts[0].content, CarriedTextContent::Refusal(_)));
    assert_eq!(texts[1].text(), "I can't help with the rest.");
    assert!(matches!(&texts[1].content, CarriedTextContent::Refusal(_)));
    let items = replayed_input(response.message().expect("decoded assistant history"));
    assert_eq!(items.len(), 1, "one assistant item");
    assert_eq!(
        items[0]["content"],
        json!([
            {"type":"output_text","text":"Here is part of it."},
            {"type":"refusal","refusal":"I can't help with the rest."}
        ])
    );
}

#[test]
fn a_provider_refusal_field_on_output_text_is_not_the_refusal_marker() {
    for value in [json!(true), json!("partial")] {
        let part = json!({"type":"output_text","text":"ok","refusal":value});
        let response = decoded_output(vec![json!({
            "type":"message","id":"msg_1","role":"assistant","status":"completed",
            "content":[part.clone()]
        })]);
        let [content] = response.choice.as_slice() else {
            panic!("expected one text block, got {:?}", response.choice);
        };
        let output_text = match content {
            AssistantContent::Text(_) => true,
            AssistantContent::CarriedText(text) => {
                matches!(&text.content, CarriedTextContent::Text(_))
            }
            _ => false,
        };
        assert!(output_text, "a `refusal: {value}` sibling marked the block");
        let items = replayed_input(response.message().expect("decoded assistant history"));
        assert_eq!(items[0]["content"], json!([part]), "for `refusal: {value}`");
    }
}

#[test]
fn a_refusal_in_a_phased_message_replays_as_a_refusal() {
    let response = decoded_output(vec![json!({
        "type":"message","id":"msg_1","role":"assistant","status":"completed",
        "content":[{"type":"refusal","refusal":"I can't help with that."}],
        "phase":"final_answer"
    })]);
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("expected one carried refusal, got {:?}", response.choice);
    };
    assert!(matches!(&text.content, CarriedTextContent::Refusal(_)));
    let items = replayed_input(response.message().expect("decoded assistant history"));
    assert_eq!(
        items[0]["content"],
        json!([{"type":"refusal","refusal":"I can't help with that."}])
    );
    assert_eq!(items[0]["phase"], "final_answer");
}

fn assistant_with(items: Vec<message::ProviderItem>) -> Message {
    Message::Assistant(AssistantMessage::new(
        items
            .into_iter()
            .map(AssistantContent::ProviderItem)
            .collect(),
    ))
}

#[test]
fn provider_items_replay_verbatim_as_their_own_input_items() {
    let hosted = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
        "action": {"query": "rig", "sources": [{"url": "https://example.invalid"}]}});
    let compaction = json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"});
    let history = assistant_with(vec![
        message::ProviderItem::new(compaction.clone(), "openai"),
        message::ProviderItem::new(hosted.clone(), "openai"),
    ]);
    let items = replayed_input(history);
    assert_eq!(items[0]["type"], "compaction");
    assert_eq!(items, [compaction, hosted]);
}

#[test]
fn a_history_with_a_provider_item_has_a_stable_serialized_form() {
    let history = assistant_with(vec![message::ProviderItem::new(
        json!({"id":"ws_1","type":"web_search_call"}),
        "openai",
    )]);
    assert_eq!(
        serde_json::to_string(&history).expect("serializes"),
        r#"{"role":"assistant","content":[{"type":"provider_item","item":{"id":"ws_1","type":"web_search_call"},"provider":"openai"}]}"#
    );
    let old: Message = serde_json::from_value(json!({
        "role":"assistant","id":"msg_1","content":[{"type":"text","text":"hi"}]
    }))
    .expect("a history from before provider items decodes");
    assert!(matches!(old, Message::Assistant(_)));
}

#[test]
fn retain_replayable_provider_items_drops_only_what_the_destination_cannot_replay() {
    let item = |provider: Option<&str>| message::ProviderItem {
        payload: message::CarriedPayload::new(
            json!({"type":"web_search_call"}),
            provider.map(str::to_owned),
        ),
    };
    let mut history = vec![
        assistant_with(vec![
            item(Some("openai")),
            item(Some("anthropic")),
            item(None),
        ]),
        assistant_with(vec![item(Some("anthropic"))]),
    ];
    message::retain_replayable_provider_items(&mut history, &["openai"]);
    assert_eq!(
        history,
        vec![assistant_with(vec![item(Some("openai")), item(None)])]
    );
}
