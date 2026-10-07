use super::*;
use serde_json::json;

fn signed() -> CarriedText {
    CarriedText::new(
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
            " original answer ",
            GeminiThoughtSignature::new(" \tsignature\n "),
        )),
        CarriedPayload::new(
            json!({"text":" original answer ","thoughtSignature":" \tsignature\n ","future":{"bytes":[null,7]}}),
            Some("gemini".to_owned()),
        ).with_position(CarriedPosition::MessageContent { item_index: 3, part_index: 2 }),
    )
}

#[test]
fn signed_answer_edits_project_current_text_with_independent_exact_signature() {
    let mut part = signed();
    let payload = part.payload.clone();
    *part.text_mut() = " \nedited answer\t ".to_owned();
    assert_eq!(part.known_text_projection(), Some(" \nedited answer\t "));
    assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText(" \nedited answer\t ")));
    assert_eq!(part.signed_answer().expect("signed category").signature.as_str(), " \tsignature\n ");
    assert_eq!(part.payload, payload);
    part.payload.item["thoughtSignature"] = json!("inspection edit");
    assert_eq!(part.signed_answer().expect("typed signature").signature.as_str(), " \tsignature\n ");
    assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText(" \nedited answer\t ")));
    let decoded: CarriedText = serde_json::from_value(serde_json::to_value(&part).expect("write edited signed answer"))
        .expect("read edited signed answer");
    assert_eq!(decoded, part);
    assert_eq!(decoded.payload.position, CarriedPosition::MessageContent { item_index: 3, part_index: 2 });
}

#[test]
fn signature_presence_is_typed_and_exact_without_empty_normalization() {
    for signature in ["", " ", "\t\n", "sig"] {
        let answer = GeminiSignedAnswer::new("", GeminiThoughtSignature::new(signature));
        assert_eq!(serde_json::to_value(&answer).expect("write signature"), json!({"text":"","signature":signature}));
        let decoded: GeminiSignedAnswer = serde_json::from_value(json!({"text":"","signature":signature}))
            .expect("read exact signature");
        assert_eq!(decoded, answer);
        let part = CarriedText::new(CarriedTextContent::GeminiSignedAnswer(answer), CarriedPayload::new(json!(null), None));
        assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText("")));
    }
    for malformed in [json!({"text":"answer"}), json!({"text":"answer","signature":null}), json!({"text":"answer","signature":17})] {
        assert!(serde_json::from_value::<GeminiSignedAnswer>(malformed).is_err());
    }
}

#[test]
fn inspection_media_survives_exactly_but_never_returns_an_outbound_raw_part() {
    let raw = json!({"inlineData":{"mimeType":"image/png","data":" exact bytes "},"thoughtSignature":"raw signature","thought":true,"future":[null,{"nested":" \n "}]});
    let payload = CarriedPayload::new(raw.clone(), Some(" foreign/issuer ".to_owned()))
        .with_position(CarriedPosition::MessageContent { item_index: 9, part_index: 4 });
    let mut part = CarriedText::new(CarriedTextContent::InspectionOnlyMedia(String::new()), payload.clone());
    assert_eq!(part.known_text_projection(), Some(""));
    assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText("")));
    assert!(part.signed_answer().is_none());
    let encoded = serde_json::to_value(&part).expect("write raw inspector");
    assert_eq!(encoded["payload"]["item"], raw);
    assert_eq!(encoded["payload"]["provider"], " foreign/issuer ");
    for absent in ["item_id", "phase", "unknown_message_fields", "citations", "native", "origin"] {
        assert!(encoded.get(absent).is_none());
    }
    let decoded: CarriedText = serde_json::from_value(encoded).expect("read raw inspector");
    assert_eq!(decoded, part);
    *part.text_mut() = " \nknown edit\t ".to_owned();
    assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText(" \nknown edit\t ")));
    assert_eq!(part.payload, payload);
    part.payload.item["future"] = json!({"raw-only":"edit"});
    assert_eq!(part.replay_item(), Ok(CarriedTextReplay::KnownText(" \nknown edit\t ")));
    assert_eq!(part.text(), " \nknown edit\t ");
}

#[test]
fn responses_categories_keep_their_exact_stored_shapes_and_replay_contract() {
    for (content, encoded, raw, expected) in [
        (CarriedTextContent::Text("x".to_owned()), json!({"type":"text","text":"x"}), json!({"type":"output_text","text":"before","future":true}), json!({"type":"output_text","text":"x","future":true})),
        (CarriedTextContent::Refusal("x".to_owned()), json!({"type":"refusal","text":"x"}), json!({"type":"refusal","refusal":"before","future":true}), json!({"type":"refusal","refusal":"x","future":true})),
        (CarriedTextContent::Opaque(String::new()), json!({"type":"opaque","text":""}), json!({"type":"future","raw":[null,4]}), json!({"type":"future","raw":[null,4]})),
    ] {
        assert_eq!(serde_json::to_value(&content).expect("existing category bytes"), encoded);
        let part = CarriedText::new(content, CarriedPayload::new(raw.clone(), None));
        assert!(part.known_text_projection().is_none());
        assert!(part.signed_answer().is_none());
        assert_eq!(part.replay_item(), Ok(CarriedTextReplay::ProviderPart(expected)));
        assert_eq!(part.payload.item, raw);
        let stored = serde_json::to_value(&part).expect("existing absent fields");
        assert!(stored.get("item_id").is_none());
        assert!(stored.get("phase").is_none());
        assert!(stored["payload"].get("provider").is_none());
        assert!(stored["payload"].get("position").is_none());
    }
    let mut opaque = CarriedText::new(CarriedTextContent::Opaque(String::new()), CarriedPayload::new(json!({"type":"future"}), None));
    *opaque.text_mut() = " ".to_owned();
    assert_eq!(opaque.replay_item(), Err(CarriedTextReplayError::EditedOpaqueText));
    assert_eq!(opaque.replay_item().expect_err("existing opaque edit refusal").to_string(),
        "an opaque Responses message part carries non-empty text: the raw part replays verbatim, so the text cannot be sent");
}

#[test]
fn signed_answer_and_inspection_blocks_never_acquire_native_authority() {
    for part in [signed(), CarriedText::new(CarriedTextContent::InspectionOnlyMedia(String::new()), CarriedPayload::new(json!({"inlineData":{"data":"bytes"}}), None))] {
        let block = AssistantContent::CarriedText(part);
        assert!(block.is_carried());
        assert!(!block.is_blank());
        assert_eq!(block.canonical(), block);
        assert_eq!(block.clone().with_native(json!({"native":"must not attach"})), block);
        assert!(block.native_item().is_none());
        assert!(block.stale_item().is_none());
    }
}
