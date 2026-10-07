use super::{Citation, Source, SourceLocation, Text, attach, attach_carried};
use crate::message::{CarriedPayload, CarriedPosition, CarriedText, CarriedTextContent};
use crate::wire::{SpanUnit, WireCitation, WireSpan};
use serde_json::json;

fn carried(text: &str) -> CarriedText {
    let mut part = CarriedText::new(
        CarriedTextContent::Text(text.to_owned()),
        CarriedPayload::new(json!({"type":"output_text","text":text,"future":[null,7]}), Some("openai".to_owned()))
            .with_position(CarriedPosition::MessageContent { item_index: 2, part_index: 3 }),
    );
    part.item_id = Some("msg_1".to_owned());
    part.phase = Some("final_answer".to_owned());
    part
}

fn source() -> Source {
    Source::new(SourceLocation::Url { url: "https://example.com/source".to_owned() })
}

#[test]
fn carried_text_accessor_binds_edits_restores_and_clears_like_text_without_changing_payload() {
    let original = carried("Dock Seven");
    let citation = Citation::new([source()]);
    let mut part = original.clone().with_citations([citation.clone()]);
    let mut text = Text::new("Dock Seven").with_citations([citation]);
    assert_eq!(part.citations(), text.citations());
    assert_eq!(part.payload, original.payload);
    assert_eq!(part.item_id, original.item_id);
    assert_eq!(part.phase, original.phase);
    assert_eq!(part.replay_item().expect("known replay"), crate::message::CarriedTextReplay::ProviderPart(original.payload.item.clone()));
    part.text_mut().push('!');
    text.text.push('!');
    assert_eq!(part.citations(), text.citations());
    assert!(part.citations().is_empty());
    part.text_mut().pop();
    text.text.pop();
    assert_eq!(part.citations(), text.citations());
    assert_eq!(part.citations().len(), 1);
    part.clear_citations();
    text.clear_citations();
    assert_eq!(part.citations(), text.citations());
    assert!(serde_json::to_value(&part).expect("stored part").get("citations").is_none());
    assert_eq!(part, original);
}

#[test]
fn carried_text_uses_existing_span_resolution_and_stored_span_checks() {
    let value = "é 𝄞 Dock Seven";
    let wire = vec![
        WireCitation::new(None, vec![source()]),
        WireCitation::new(Some(WireSpan::new(4, 14, SpanUnit::Chars).quoted("Dock Seven")), vec![source()]),
        WireCitation::new(Some(WireSpan::new(3, 4, SpanUnit::Utf16)), vec![source()]),
    ];
    let mut text = Text::new(value);
    let mut part = carried(value);
    attach(&mut text, Vec::new(), wire.clone(), "test", 0);
    attach_carried(&mut part, Vec::new(), wire, "test", 0);
    assert_eq!(part.citations(), text.citations());
    assert_eq!(part.citations().len(), 2);
    assert_eq!(part.citations()[1].span.expect("resolved span").range(), 8..18);
    let mut invalid = Citation::new([source()]);
    invalid.span = Some(super::Span { start: 0, end: 1 });
    assert!(part.clone().with_citations([invalid.clone()]).citations().is_empty());
    let mut stored = serde_json::to_value(&part).expect("stored part");
    stored["citations"]["list"] = json!([invalid]);
    let loaded: CarriedText = serde_json::from_value(stored).expect("lenient stored citations");
    assert!(loaded.citations().is_empty());
    assert!(serde_json::to_value(loaded).expect("checked part").get("citations").is_none());
}

#[test]
fn absent_citations_keep_existing_bytes_and_unreadable_citations_load_as_none() {
    let old = r#"{"content":{"type":"text","text":"Dock"},"payload":{"item":null}}"#;
    let part: CarriedText = serde_json::from_str(old).expect("existing stored part");
    assert_eq!(serde_json::to_string(&part).expect("unchanged bytes"), old);
    for invalid in [json!(null), json!(7), json!({"fingerprint":"bad","list":[]})] {
        let mut stored: serde_json::Value = serde_json::from_str(old).expect("old JSON");
        stored["citations"] = invalid;
        let loaded: CarriedText = serde_json::from_value(stored).expect("lenient citation field");
        assert!(loaded.citations().is_empty());
        assert_eq!(serde_json::to_string(&loaded).expect("old shape restored"), old);
    }
    let part = carried("Dock Seven").with_citations([Citation::new([source()])]);
    let loaded: CarriedText = serde_json::from_value(serde_json::to_value(&part).expect("store citations")).expect("load citations");
    assert_eq!(loaded, part);
    assert_eq!(loaded.citations(), part.citations());
    let mut stale = serde_json::to_value(&part).expect("stored citation fingerprint");
    stale["content"]["text"] = json!("Dock Eight");
    let loaded: CarriedText = serde_json::from_value(stale).expect("stored edited text");
    assert!(loaded.citations().is_empty(), "loading does not rebind an old fingerprint");
}
