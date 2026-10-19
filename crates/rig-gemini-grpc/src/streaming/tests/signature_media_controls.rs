//! The real protobuf adapter assigns pending signatures to media custody.
//! These complete replies never place a late signature after a typed image.

use crate::proto;
use rig_core::completion::{CompletionResponse, FinishReason};
use rig_core::message::{AssistantContent, CarriedPosition, CarriedTextContent};
use serde_json::{Value, json};

fn signature(bytes: &[u8]) -> proto::Part {
    proto::Part {
        thought_signature: bytes.to_vec(),
        ..Default::default()
    }
}
fn media() -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::InlineData(proto::Blob {
            mime_type: "image/png".to_owned(),
            data: vec![0],
        })),
        ..Default::default()
    }
}
fn answer() -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text("answer".to_owned())),
        ..Default::default()
    }
}
fn reply(parts: Vec<proto::Part>) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                role: "model".to_owned(),
                parts,
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        ..Default::default()
    }
}
fn assert_media_answer(
    response: &CompletionResponse,
    raw: &Value,
    signature: Option<&str>,
    index: u64,
) {
    let [
        AssistantContent::CarriedText(image),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!("real GrpcAdapter must retain inspection media and unsigned answer");
    };
    let mut expected = json!({"inlineData":{"mimeType":"image/png","data":"AA=="}});
    if let Some(signature) = signature {
        expected["thoughtSignature"] = json!(signature);
    }
    assert_eq!(
        image.content,
        CarriedTextContent::InspectionOnlyMedia(String::new())
    );
    assert_eq!(image.payload.item, expected);
    assert_eq!(
        image.payload.provider.as_deref(),
        Some(rig_core::providers::gemini::completion::PROVIDER_NAME)
    );
    assert_eq!(
        image.payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: index
        }
    );
    assert!(image.signed_answer().is_none());
    assert_eq!(text.text, "answer");
    assert_eq!(
        response.choice[1].native_item(),
        Some(&json!({"text":"answer"}))
    );
    assert_eq!(response.text(), "answer");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        &response.raw, raw,
        "the actual unary protobuf document remains exact"
    );
    assert!(
        !response
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::Image(_)))
    );
}

#[test]
fn g218_grpc_leading_signature_belongs_to_inline_media() {
    let received = reply(vec![signature(b"sig"), media(), answer()]);
    let raw = crate::rest::to_rest(&received).expect("actual proto3 JSON");
    assert_eq!(
        raw["candidates"][0]["content"]["parts"],
        json!([
            {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}}, {"text":"answer"}
        ])
    );
    let response = crate::completion::tests::complete(received)
        .expect("real unary protobuf Model/GrpcAdapter/fold");
    assert_media_answer(&response, &raw, Some("c2ln"), 1);
}

#[test]
fn g218_grpc_bare_signature_after_signed_inspection_targets_that_media_slot() {
    let received = reply(vec![
        signature(b"sig"),
        media(),
        signature(b"tail"),
        answer(),
    ]);
    let raw = crate::rest::to_rest(&received).expect("actual proto3 JSON");
    assert_eq!(
        raw["candidates"][0]["content"]["parts"],
        json!([
            {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}},
            {"thoughtSignature":"dGFpbA=="}, {"text":"answer"}
        ])
    );
    let response = crate::completion::tests::complete(received).expect("real protobuf adapter");
    assert_media_answer(&response, &raw, Some("dGFpbA=="), 1);
}

#[test]
fn g218_grpc_unsigned_media_keeps_its_existing_inspection_category() {
    let received = reply(vec![media(), answer()]);
    let raw = crate::rest::to_rest(&received).expect("actual proto3 JSON");
    let response =
        crate::completion::tests::complete(received).expect("real unsigned protobuf reply");
    assert_media_answer(&response, &raw, None, 0);
}
