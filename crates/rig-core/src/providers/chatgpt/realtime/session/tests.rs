use super::*;
use serde_json::json;

/// The session the live probe created its calls with serializes to exactly
/// the object it sent.
#[test]
fn the_probe_session_serializes_exactly() {
    let session = SessionConfig::new("Answer briefly.")
        .with_initial_item(InitialItem::developer("The report text."));
    assert_eq!(
        serde_json::to_value(&session).expect("serializes"),
        json!({
            "model": "gpt-live-1-codex",
            "instructions": "Answer briefly.",
            "audio": {"output": {"voice": "cove"}},
            "delegation": {"type": "client"},
            "initial_items": [{
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "The report text."}]
            }]
        })
    );
}

/// Without initial items none are sent; the ack filler, model and voice go
/// where the Codex client puts them; assistant items carry `output_text`.
#[test]
fn optional_fields_and_roles_serialize_as_codex_sends_them() {
    let bare = serde_json::to_value(SessionConfig::new("x")).expect("serializes");
    assert!(bare.get("initial_items").is_none());
    assert!(bare["delegation"].get("ack_filler").is_none());

    let session = SessionConfig::new("x")
        .with_model("gpt-live-next")
        .with_voice(Voice::Juniper)
        .with_delegation_ack_filler(false)
        .with_initial_item(InitialItem::user("u"))
        .with_initial_item(InitialItem::assistant("a"));
    let value = serde_json::to_value(&session).expect("serializes");
    assert_eq!(value["model"], "gpt-live-next");
    assert_eq!(value["audio"]["output"]["voice"], "juniper");
    assert_eq!(
        value["delegation"],
        json!({"type": "client", "ack_filler": false})
    );
    assert_eq!(
        value["initial_items"],
        json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "u"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]}
        ])
    );
}

/// The V3 voices, spelled as the Codex client spells them.
#[test]
fn voices_use_the_v3_names() {
    let voices = [
        Voice::Juniper,
        Voice::Maple,
        Voice::Spruce,
        Voice::Ember,
        Voice::Vale,
        Voice::Breeze,
        Voice::Arbor,
        Voice::Sol,
        Voice::Cove,
    ];
    let names: Vec<_> = voices
        .iter()
        .map(|voice| serde_json::to_value(voice).expect("serializes"))
        .collect();
    assert_eq!(
        names,
        [
            "juniper", "maple", "spruce", "ember", "vale", "breeze", "arbor", "sol", "cove"
        ]
        .map(|name| json!(name))
    );
    assert_eq!(Voice::default(), Voice::Cove);
}
