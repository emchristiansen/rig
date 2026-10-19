use super::*;
use serde_json::json;

fn payload(item: serde_json::Value, position: CarriedPosition) -> CarriedPayload {
    CarriedPayload::new(item, Some("foreign/issuer".to_owned())).with_position(position)
}

#[test]
fn whole_provider_items_keep_the_exact_existing_wire_bytes() {
    let known = ProviderItem::new(json!({"type": "future"}), "openai");
    let encoded = serde_json::to_string(&known).expect("serialize whole item");
    assert_eq!(encoded, r#"{"item":{"type":"future"},"provider":"openai"}"#);
    let decoded = serde_json::from_str::<ProviderItem>(&encoded).expect("read old shape");
    assert_eq!(decoded, known);
    assert_eq!(decoded.payload.position, CarriedPosition::Whole);
    assert_eq!(decoded.item_type(), Some("future"));
    assert!(decoded.replayable_to("openai"));
    assert!(!decoded.replayable_to("another"));

    let unknown = serde_json::from_str::<ProviderItem>(r#"{"item":[1,null,{"x":true}]}"#)
        .expect("read old unknown issuer");
    assert_eq!(unknown.payload.provider, None);
    assert_eq!(unknown.payload.position, CarriedPosition::Whole);
    assert_eq!(
        serde_json::to_string(&unknown).expect("write old shape"),
        r#"{"item":[1,null,{"x":true}]}"#
    );
    assert!(unknown.replayable_to("any"));
    assert_eq!(unknown.item_type(), None);
}

#[test]
fn payload_positions_and_issuer_spelling_round_trip_exactly() {
    for (position, encoded_position) in [
        (CarriedPosition::Whole, None),
        (
            CarriedPosition::MessageContent {
                item_index: u64::MAX,
                part_index: 9,
            },
            Some(json!({"kind": "message_content", "item_index": u64::MAX, "part_index": 9})),
        ),
        (
            CarriedPosition::ReasoningSummary {
                part_index: u64::MAX,
            },
            Some(json!({"kind": "reasoning_summary", "part_index": u64::MAX})),
        ),
        (
            CarriedPosition::ReasoningContent { part_index: 4 },
            Some(json!({"kind": "reasoning_content", "part_index": 4})),
        ),
    ] {
        let value = CarriedPayload::new(
            json!({"type": "future", "unknown": [null, {"text": "raw"}]}),
            Some(" \tissuer/ ".to_owned()),
        )
        .with_position(position);
        let encoded = serde_json::to_value(&value).expect("serialize payload");
        assert_eq!(encoded["item"], value.item);
        assert_eq!(encoded["provider"], " \tissuer/ ");
        assert_eq!(encoded.get("position"), encoded_position.as_ref());
        assert!(encoded.get("origin").is_none());
        assert!(encoded.get("native").is_none());
        assert_eq!(
            serde_json::from_value::<CarriedPayload>(encoded).expect("read payload"),
            value
        );
    }
    for invalid in [
        json!({"kind": "reasoning_content", "part_index": -1}),
        json!({"kind": "message_content", "item_index": "0", "part_index": 1}),
        json!({"kind": "message_content", "item_index": 0}),
    ] {
        assert!(serde_json::from_value::<CarriedPosition>(invalid).is_err());
    }
    let family = CarriedPayload::new(json!(null), Some("openrouter/openai".to_owned()));
    assert!(family.replayable_to("openrouter/"));
    assert!(!family.replayable_to("openrouter"));
    assert!(!family.replayable_to("openai"));
}

#[test]
fn known_text_and_refusal_edits_update_only_their_actual_wire_field() {
    for (content, field, raw) in [
        (
            CarriedTextContent::Text("before".to_owned()),
            "text",
            json!({"type": "output_text", "text": "before", "refusal": "sibling", "future": {"list": [1, null]}, "annotations": []}),
        ),
        (
            CarriedTextContent::Refusal("before".to_owned()),
            "refusal",
            json!({"type": "refusal", "refusal": "before", "text": "sibling", "future": {"list": [1, null]}}),
        ),
    ] {
        let mut part = CarriedText::new(
            content,
            payload(
                raw.clone(),
                CarriedPosition::MessageContent {
                    item_index: 7,
                    part_index: 2,
                },
            ),
        );
        part.item_id = Some(String::new());
        part.phase = Some(" \tfinal_answer ".to_owned());
        *part.text_mut() = " \nchanged\t ".to_owned();
        assert_eq!(part.text(), " \nchanged\t ");
        let mut expected = raw.clone();
        expected[field] = json!(" \nchanged\t ");
        assert_eq!(
            part.replay_item().expect("replay known edit"),
            CarriedTextReplay::ProviderPart(expected.clone())
        );
        assert_eq!(
            part.payload.item, raw,
            "the original payload remains evidence"
        );
        assert_eq!(
            part.payload.position,
            CarriedPosition::MessageContent {
                item_index: 7,
                part_index: 2
            }
        );
        let encoded = serde_json::to_value(&part).expect("serialize part");
        assert_eq!(encoded["item_id"], "");
        assert_eq!(encoded["phase"], " \tfinal_answer ");
        let decoded = serde_json::from_value::<CarriedText>(encoded).expect("read edited part");
        assert_eq!(decoded, part);
        assert_eq!(
            decoded.replay_item().expect("replay after serde"),
            CarriedTextReplay::ProviderPart(expected.clone())
        );
    }
}

#[test]
fn opaque_raw_only_edits_replay_exactly_and_canonical_text_edits_refuse() {
    let raw = json!({"type": "future_part", "text": "raw text", "payload": [1, null, {"x": true}]});
    let mut part = CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        payload(
            raw,
            CarriedPosition::MessageContent {
                item_index: 2,
                part_index: 1,
            },
        ),
    );
    assert_eq!(part.text(), "");
    part.payload.item["payload"] = json!([7, {"text": "raw-only edit"}]);
    let expected = part.payload.item.clone();
    assert_eq!(
        part.replay_item().expect("raw-only edit replays"),
        CarriedTextReplay::ProviderPart(expected.clone())
    );
    let decoded = serde_json::from_value::<CarriedText>(
        serde_json::to_value(&part).expect("serialize opaque part"),
    )
    .expect("read opaque part");
    assert_eq!(
        decoded.replay_item().expect("opaque serde replay"),
        CarriedTextReplay::ProviderPart(expected.clone())
    );
    for edit in ["changed", " ", "\t"] {
        *part.text_mut() = edit.to_owned();
        let error = part
            .replay_item()
            .expect_err("any nonempty canonical opaque text is refused");
        assert_eq!(error, CarriedTextReplayError::EditedOpaqueText);
        assert_eq!(
            error.to_string(),
            "an opaque Responses message part carries non-empty text: the raw part replays verbatim, so the text cannot be sent"
        );
        assert_eq!(part.payload.item, expected);
    }
    part.text_mut().clear();
    assert_eq!(
        part.replay_item().expect("cleared canonical edit"),
        CarriedTextReplay::ProviderPart(expected.clone())
    );
    for raw in [json!(null), json!(["unknown", 2]), json!("scalar")] {
        part.payload.item = raw.clone();
        assert_eq!(
            part.replay_item().expect("unknown raw shape retained"),
            CarriedTextReplay::ProviderPart(raw.clone())
        );
    }
}

#[test]
fn malformed_known_parts_fail_without_replacing_their_preserved_json() {
    for raw in [json!(null), json!([]), json!("raw")] {
        for (content, field) in [
            (CarriedTextContent::Text("text".to_owned()), "text"),
            (CarriedTextContent::Refusal("refusal".to_owned()), "refusal"),
        ] {
            let part = CarriedText::new(content, CarriedPayload::new(raw.clone(), None));
            assert_eq!(
                part.replay_item(),
                Err(CarriedTextReplayError::InvalidKnownPart { field })
            );
            assert_eq!(part.payload.item, raw);
        }
    }
}

#[test]
fn ordered_message_parts_keep_known_edits_opaque_siblings_and_item_metadata() {
    let raw = [
        json!({"type": "output_text", "text": "A", "future": [1]}),
        json!({"type": "future", "payload": {"nested": [null, 2]}}),
        json!({"type": "refusal", "refusal": "No", "future": [3]}),
        json!({"type": "output_text", "text": "B"}),
    ];
    let contents = [
        CarriedTextContent::Text("A".to_owned()),
        CarriedTextContent::Opaque(String::new()),
        CarriedTextContent::Refusal("No".to_owned()),
        CarriedTextContent::Text("B".to_owned()),
    ];
    let mut parts: Vec<_> = contents
        .into_iter()
        .zip(raw.clone())
        .enumerate()
        .map(|(at, (content, raw))| {
            let mut part = CarriedText::new(
                content,
                payload(
                    raw,
                    CarriedPosition::MessageContent {
                        item_index: 8,
                        part_index: at as u64,
                    },
                ),
            );
            part.item_id = Some("msg_part_only".to_owned());
            part.phase = Some("final_answer".to_owned());
            part
        })
        .collect();
    *parts[0].text_mut() = "edited A".to_owned();
    *parts[2].text_mut() = "edited No".to_owned();
    let decoded = serde_json::from_value::<Vec<CarriedText>>(
        serde_json::to_value(&parts).expect("serialize all message parts"),
    )
    .expect("read ordered message parts");
    assert_eq!(decoded, parts);
    let replay: Vec<_> = decoded
        .iter()
        .map(|part| part.replay_item().expect("replay individual part"))
        .collect();
    assert_eq!(
        replay,
        [
            json!({"type": "output_text", "text": "edited A", "future": [1]}),
            raw[1].clone(),
            json!({"type": "refusal", "refusal": "edited No", "future": [3]}),
            raw[3].clone(),
        ]
        .map(CarriedTextReplay::ProviderPart)
    );
    for (at, part) in decoded.iter().enumerate() {
        assert_eq!(part.payload.item, raw[at]);
        assert_eq!(
            part.payload.position,
            CarriedPosition::MessageContent {
                item_index: 8,
                part_index: at as u64
            }
        );
        assert_eq!(part.item_id.as_deref(), Some("msg_part_only"));
        assert_eq!(part.phase.as_deref(), Some("final_answer"));
    }
}

#[test]
fn structured_reasoning_keeps_opaque_order_issuers_and_known_edits_independent() {
    let summary = payload(
        json!({"type": "future_summary", "text": "not visible", "array": [1, null]}),
        CarriedPosition::ReasoningSummary { part_index: 4 },
    );
    let content = CarriedPayload::new(
        json!({"type": "future_content", "text": "also not visible"}),
        None,
    )
    .with_position(CarriedPosition::ReasoningContent { part_index: 9 });
    let raw_item = json!({"type": "reasoning", "id": "rs_1", "future": {"array": [true, null]}});
    let mut reasoning = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Text {
                text: "first".to_owned(),
                signature: Some(" sig ".to_owned()),
            },
            CarriedReasoningPart::Text {
                text: String::new(),
                signature: Some(String::new()),
            },
            CarriedReasoningPart::Summary("sum".to_owned()),
            CarriedReasoningPart::Encrypted("encrypted bytes".to_owned()),
            CarriedReasoningPart::Redacted {
                data: "withheld bytes".to_owned(),
            },
            CarriedReasoningPart::OpaqueSummary(summary.clone()),
            CarriedReasoningPart::OpaqueContent(content.clone()),
            CarriedReasoningPart::Summary("last".to_owned()),
        ],
        payload(raw_item.clone(), CarriedPosition::Whole),
    )
    .with_id(Some("rs_1".to_owned()));
    assert_eq!(
        reasoning.display_text(),
        "first\n\nsum\nwithheld bytes\nlast"
    );
    assert!(reasoning.has_opaque_parts());
    assert_eq!(reasoning.provider(), Some("foreign/issuer"));
    assert!(
        !reasoning.replayable_to("openai"),
        "provenance comparison does not itself filter nested reasoning"
    );
    let CarriedReasoningPart::Text { text, signature } = &mut reasoning.content[0] else {
        panic!("text part");
    };
    *text = "edited first".to_owned();
    assert_eq!(signature.as_deref(), Some(" sig "));
    assert_eq!(
        reasoning.content[5],
        CarriedReasoningPart::OpaqueSummary(summary)
    );
    assert_eq!(
        reasoning.content[6],
        CarriedReasoningPart::OpaqueContent(content)
    );
    assert_eq!(reasoning.payload.item, raw_item);
    assert_eq!(
        reasoning.display_text(),
        "edited first\n\nsum\nwithheld bytes\nlast"
    );
    let decoded = serde_json::from_value::<CarriedReasoning>(
        serde_json::to_value(&reasoning).expect("serialize structured reasoning"),
    )
    .expect("read structured reasoning");
    assert_eq!(decoded, reasoning);
    assert_eq!(decoded.provider(), Some("foreign/issuer"));
    assert_eq!(decoded.content.len(), 8);
}

#[test]
fn known_reasoning_does_not_claim_opacity_or_normalize_optional_evidence() {
    let reasoning = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Text {
                text: " \ttext ".to_owned(),
                signature: None,
            },
            CarriedReasoningPart::Encrypted(" encrypted ".to_owned()),
            CarriedReasoningPart::Redacted {
                data: " redacted ".to_owned(),
            },
        ],
        CarriedPayload::new(json!(["original"]), Some(String::new())),
    )
    .with_id(Some(String::new()));
    assert!(!reasoning.has_opaque_parts());
    assert_eq!(reasoning.display_text(), " \ttext \n redacted ");
    assert_eq!(reasoning.provider(), Some(""));
    assert_eq!(reasoning.id.as_deref(), Some(""));
    let encoded = serde_json::to_value(&reasoning).expect("serialize exact optional evidence");
    assert_eq!(encoded["id"], "");
    assert_eq!(encoded["payload"]["provider"], "");
    assert!(encoded["content"][0]["content"].get("signature").is_none());
    assert_eq!(
        serde_json::from_value::<CarriedReasoning>(encoded).expect("read exact evidence"),
        reasoning
    );
    let absent = CarriedReasoning::new(Vec::new(), CarriedPayload::new(json!(null), None));
    assert_eq!(absent.display_text(), "");
    assert!(!absent.has_opaque_parts());
    let encoded = serde_json::to_value(absent).expect("serialize absent evidence");
    assert!(encoded.get("id").is_none());
    assert!(encoded["payload"].get("provider").is_none());
}

#[test]
fn nested_refusal_owns_the_exact_destination_and_fork_display() {
    let error = UnrepresentableOpaqueContent::new(" Dynamic destination ".to_owned());
    assert_eq!(error.wire, " Dynamic destination ");
    assert_eq!(
        error.to_string(),
        " Dynamic destination  cannot replay opaque OpenAI Responses content"
    );
    assert_eq!(error.clone(), error);
    let item = ProviderItem::new(json!({"type": "future"}), "foreign");
    let whole_error = UnreplayableProviderItem::new("destination", &item);
    assert_eq!(whole_error.issuer.as_deref(), Some("foreign"));
    assert_eq!(whole_error.item_type.as_deref(), Some("future"));
    assert_eq!(
        whole_error.to_string(),
        "destination cannot replay the `future` item issued by foreign; drop it with retain_replayable_provider_items to switch providers deliberately"
    );
}
