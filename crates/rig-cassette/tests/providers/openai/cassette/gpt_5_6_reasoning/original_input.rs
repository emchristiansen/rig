//! Sparse caller-authored history for the original five-turn request literals.

use super::StoredStreamingTurn;
use rig::message::{AssistantContent, CarriedPayload, CarriedPosition, CarriedText, CarriedTextContent, Message};
use serde_json::json;

pub(super) fn assistant(turn: &StoredStreamingTurn) -> Message {
    let Message::Assistant(head) = &turn.assistant else {
        panic!("the stored turn must hold its genuine assistant head");
    };
    assert_eq!(head.content, turn.final_response.choice);
    assert!(!head.content.is_empty(), "the received answer must contain known text");
    let content = head.content.iter().map(|block| {
        let AssistantContent::CarriedText(source) = block else {
            panic!("the received answer must be a carried known-text part");
        };
        let CarriedTextContent::Text(answer) = &source.content else {
            panic!("the received answer must be known Text, without an opaque fallback");
        };
        let item_id = source.item_id.as_ref().filter(|id| !id.is_empty())
            .expect("the received message must supply a genuine nonempty item ID");
        let CarriedPosition::MessageContent { item_index, part_index } = source.payload.position else {
            panic!("the received known text must retain its message-content position");
        };
        let received = turn.final_response.raw.get("output").and_then(serde_json::Value::as_array)
            .and_then(|items| items.get(usize::try_from(item_index).expect("item position fits usize")))
            .expect("the genuine response must contain the received message at its original position");
        assert_eq!(received.get("type").and_then(serde_json::Value::as_str), Some("message"));
        assert_eq!(received.get("id").and_then(serde_json::Value::as_str), Some(item_id.as_str()));
        let received_part = received.get("content").and_then(serde_json::Value::as_array)
            .and_then(|parts| parts.get(usize::try_from(part_index).expect("part position fits usize")))
            .expect("the genuine response must contain the received known-text part");
        assert_eq!(received_part.get("type").and_then(serde_json::Value::as_str), Some("output_text"));
        assert_eq!(received_part.get("text").and_then(serde_json::Value::as_str), Some(answer.as_str()));
        let mut fresh = CarriedText::new(
            CarriedTextContent::Text(answer.clone()),
            CarriedPayload::new(json!({"type": "output_text", "text": answer}), None)
                .with_position(source.payload.position),
        );
        fresh.item_id = Some(item_id.clone());
        AssistantContent::CarriedText(fresh)
    }).collect();
    Message::Assistant(head.clone().with_content(content))
}

#[cfg(test)]
mod tests;
