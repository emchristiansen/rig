//! Pending signatures belong to the received media slot, not the next answer.
//! Late signatures after an already typed image remain outside these controls.

use crate::completion::{CompletionRequest, CompletionResponse, FinishReason};
use crate::driver::Model;
use crate::error::{FrameEvidence, ProviderError};
use crate::message::{AssistantContent, CarriedPosition, CarriedTextContent};
use crate::providers::gemini::{GeminiConfig, completion::PROVIDER_NAME};
use crate::wire::{Decoder, Mode, Wire, WireEvent, WireFrame};
use serde_json::{Value, json};

fn media() -> Value {
    json!({"inlineData":{"mimeType":"image/png","data":"AA=="}})
}
fn answer() -> Value {
    json!({"text":"answer"})
}
fn body(parts: Vec<Value>) -> Value {
    json!({"candidates":[{"content":{"role":"model","parts":parts},"finishReason":"STOP"}]})
}
fn complete(received: &Value) -> Result<CompletionResponse, ProviderError> {
    let wire = GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    futures::executor::block_on(
        Model::new(
            wire,
            crate::test_utils::RecordingHttpClient::new(
                serde_json::to_vec(received).expect("fixture JSON"),
            ),
        )
        .call(CompletionRequest::new("inspect")),
    )
}
fn assert_media_answer(
    response: &CompletionResponse,
    received: &Value,
    signature: Option<&str>,
    index: u64,
) {
    let [
        AssistantContent::CarriedText(image),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!(
            "one actual inspection-only media followed by unsigned answer: {:?}",
            response.choice
        );
    };
    assert_eq!(
        image.content,
        CarriedTextContent::InspectionOnlyMedia(String::new())
    );
    let mut expected = media();
    if let Some(signature) = signature {
        expected["thoughtSignature"] = json!(signature);
    }
    assert_eq!(image.payload.item, expected);
    assert_eq!(image.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(
        image.payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: index
        }
    );
    assert!(
        image.signed_answer().is_none(),
        "inspection media is not signed answer text"
    );
    assert_eq!(text.text, "answer");
    assert_eq!(
        response.choice[1].native_item(),
        Some(&answer()),
        "the real answer receives no borrowed signature"
    );
    assert_eq!(response.text(), "answer");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        response.raw, *received,
        "actual unary capture retains every original signature part"
    );
    assert!(
        !response
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::Image(_)))
    );
}

#[test]
fn g218_rest_leading_signature_belongs_to_inline_media() {
    let received = body(vec![json!({"thoughtSignature":"c2ln"}), media(), answer()]);
    let response = complete(&received).expect("actual REST decoder/fold");
    assert_media_answer(&response, &received, Some("c2ln"), 1);
}

#[test]
fn g218_rest_bare_signature_after_signed_inspection_targets_that_media_slot() {
    let received = body(vec![
        json!({"thoughtSignature":"c2ln"}),
        media(),
        json!({"thoughtSignature":"dGFpbA=="}),
        answer(),
    ]);
    let response = complete(&received).expect("actual REST decoder/fold");
    assert_media_answer(&response, &received, Some("dGFpbA=="), 1);
}

#[test]
fn g218_rest_unsigned_inline_media_keeps_its_existing_inspection_category() {
    let received = body(vec![media(), answer()]);
    let response = complete(&received).expect("actual unsigned REST reply");
    assert_media_answer(&response, &received, None, 0);
}

#[test]
fn g219_future_and_bare_objects_refuse_wrong_thought_and_signature_types_as_corrupt_frames() {
    for future in [false, true] {
        for (field, value, expected) in [
            ("thought", json!("yes"), "expected a boolean"),
            ("thoughtSignature", json!(481), "expected a string"),
            ("thoughtSignature", json!(false), "expected a string"),
            ("thoughtSignature", json!({"x":1}), "expected a string"),
            ("thoughtSignature", json!([1]), "expected a string"),
        ] {
            let mut part = if future {
                json!({"futureField":{"n":1}})
            } else {
                json!({})
            };
            part[field] = value;
            let received = body(vec![part, answer()]);
            let payload = received.to_string();
            let wire = GeminiConfig::new("test-key").completion("gemini-2.5-flash");
            for mode in [Mode::Unary, Mode::Streaming] {
                let WireEvent::Corrupt(refusal) = wire
                    .decoder_in(mode)
                    .classify(WireFrame::Text(payload.clone()))
                else {
                    panic!("every object validates Part fields before recognized-key selection");
                };
                assert_eq!(refusal.evidence(), &FrameEvidence::Text(payload.clone()));
                assert!(refusal.error().to_string().contains(expected));
            }
            let error = complete(&received)
                .expect_err("the actual REST Model refuses the malformed object");
            let ProviderError::CorruptFrame(refusal) = error else {
                panic!("actual typed CorruptFrame required");
            };
            assert_eq!(refusal.evidence(), &FrameEvidence::Text(payload));
            assert!(refusal.error().to_string().contains(expected));
        }
    }
}

#[test]
fn g219_correct_future_and_signature_only_objects_keep_exact_received_raw() {
    let future = json!({"futureField":{"n":1},"thought":false,"thoughtSignature":"c2ln"});
    let received = body(vec![future.clone(), answer()]);
    let response = complete(&received).expect("valid future metadata is accepted");
    let [
        AssistantContent::Opaque(opaque),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!("future part keeps its actual opaque category");
    };
    assert_eq!(opaque.item, future);
    assert_eq!(text.text, "answer");
    assert_eq!(response.choice[1].native_item(), Some(&answer()));
    assert_eq!(response.raw, received);
    let received = body(vec![json!({"thoughtSignature":"c2ln"}), answer()]);
    let response = complete(&received).expect("valid leading signature remains accepted");
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("the real following answer owns the signature when no media intervenes");
    };
    let signed = text.signed_answer().expect("actual signed selected text");
    assert_eq!(signed.text, "answer");
    assert_eq!(signed.signature.as_str(), "c2ln");
    assert_eq!(
        text.payload.item,
        json!({"text":"answer","thoughtSignature":"c2ln"})
    );
    assert_eq!(response.raw, received);
}

#[test]
fn g219_nonobject_parts_keep_their_exact_existing_raw_and_opaque_categories() {
    for raw in [
        Value::Null,
        json!(481),
        json!("raw"),
        json!(true),
        json!([1, null]),
    ] {
        let received = body(vec![raw.clone(), answer()]);
        let response = complete(&received).expect("nonobject parts retain the existing path");
        let [
            AssistantContent::Opaque(opaque),
            AssistantContent::Text(text),
        ] = response.choice.as_slice()
        else {
            panic!("one nonreplaying opaque part followed by the answer");
        };
        assert_eq!(opaque.item, raw);
        assert!(!opaque.replay);
        assert_eq!(text.text, "answer");
        assert_eq!(response.raw, received);
    }
}
