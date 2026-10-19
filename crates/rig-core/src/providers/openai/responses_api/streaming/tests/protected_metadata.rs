//! Deleted TextMeta ownership cases at the actual carried snapshot boundary.

use super::helpers::*;
use crate::message::AssistantContent;
use serde_json::json;

fn text_payload(choice: &[AssistantContent]) -> &serde_json::Value {
    let [AssistantContent::CarriedText(text)] = choice else {
        panic!("one carried text part")
    };
    &text.payload.item
}

fn snapshot(part: serde_json::Value, sequence_number: u64) -> serde_json::Value {
    json!({"type":"response.output_item.done","output_index":0,"sequence_number":sequence_number,
        "item":message_item("msg_opaque",json!([part]))})
}

// F streaming/tests.rs:4182. Tombstone/replacement and independent values
// remain literal assertions; the deleted TextMeta method is not restored.
#[test]
fn complete_snapshot_replacement_resets_only_owned_extras_contiguously() {
    let before = json!({"annotations":["a","b"],"old":true});
    let after = json!({"annotations":["replacement"],"phase":"final_answer"});
    let part = |owned| {
        json!({"type":"output_text","text":"",
        "unrelated":{"list":[1]},"openai_responses_part":{"kind":"refusal"},"openai_responses":owned})
    };
    let initial = snapshot(part(before.clone()), 0);
    let tombstone = snapshot(part(json!(null)), 1);
    let replacement = snapshot(part(after.clone()), 2);
    let after_tombstone = drive(&[initial.clone(), tombstone.clone()]);
    assert!(after_tombstone.errors.is_empty());
    assert_eq!(
        text_payload(&after_tombstone.response.choice)["openai_responses"],
        json!(null),
        "namespace tombstone is exact"
    );
    let final_state = drive(&[initial, tombstone, replacement]);
    assert!(final_state.errors.is_empty());
    let params = text_payload(&final_state.response.choice);
    assert_eq!(params["unrelated"], json!({"list":[1]}));
    assert_eq!(params["openai_responses_part"], json!({"kind":"refusal"}));
    assert_eq!(params["openai_responses"], after);
    assert!(
        params["openai_responses"].get("old").is_none(),
        "replacement does not merge stale keys"
    );
    assert_eq!(
        params["openai_responses"]["annotations"],
        json!(["replacement"]),
        "annotation arrays replace contiguously"
    );
}

// F streaming/tests.rs:4239
#[test]
fn complete_snapshot_removing_all_extras_emits_empty_replacement() {
    let before = json!({"annotations":["a"]});
    let part = |owned| json!({"type":"output_text","text":"","openai_responses":owned});
    let initial = snapshot(part(before), 0);
    let tombstone = snapshot(part(json!(null)), 1);
    let replacement = snapshot(part(json!({})), 2);
    let after_tombstone = drive(&[initial.clone(), tombstone.clone()]);
    assert!(after_tombstone.errors.is_empty());
    assert_eq!(
        text_payload(&after_tombstone.response.choice)["openai_responses"],
        json!(null)
    );
    let final_state = drive(&[initial, tombstone, replacement]);
    assert!(final_state.errors.is_empty());
    assert_eq!(
        text_payload(&final_state.response.choice)["openai_responses"],
        json!({}),
        "empty replacement clears the owned namespace"
    );
}
