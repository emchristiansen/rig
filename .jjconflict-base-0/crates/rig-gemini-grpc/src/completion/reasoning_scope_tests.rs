use super::*;
use base64::Engine as _;
use message::{
    AssistantContent, AssistantMessage, CarriedPayload, CarriedReasoning, CarriedReasoningPart,
    Message,
};
use serde_json::{Value, json};

fn reasoning(text: &str, bytes: &[u8], issuer: Option<&str>) -> AssistantContent {
    AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: text.to_owned(),
            signature: Some(base64::prelude::BASE64_STANDARD.encode(bytes)),
        }],
        CarriedPayload::new(Value::Null, issuer.map(str::to_owned)),
    ))
}

fn request(parts: Vec<AssistantContent>) -> CompletionRequest {
    CompletionRequest::from(vec![
        Message::user("What is 2 + 2?"),
        Message::Assistant(AssistantMessage::new(parts)),
        Message::user("And 3 + 3?"),
    ])
}

use crate::proto;
use rig_core::providers::gemini::completion::PROVIDER_NAME as REASONING_ISSUER;

fn encoded(request: CompletionRequest) -> Result<GenerateContentRequest, EncodeError> {
    GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary)
}

fn thoughts(request: &GenerateContentRequest) -> Vec<(&str, Vec<u8>)> {
    request
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .filter_map(|part| match (&part.data, part.thought) {
            (Some(proto::part::Data::Text(text)), true) => {
                Some((text.as_str(), part.thought_signature.clone()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn only_gemini_reasoning_is_replayed() {
    let req = encoded(request(vec![
        reasoning("grpc thought", b"grpc", Some(REASONING_ISSUER)),
        reasoning("rest thought", b"rest", Some(REASONING_ISSUER)),
        reasoning("anthropic thought", b"anthropic", Some("anthropic")),
        AssistantContent::text("4"),
    ]))
    .expect("request build");
    assert_eq!(
        thoughts(&req),
        vec![
            ("grpc thought", b"grpc".to_vec()),
            ("rest thought", b"rest".to_vec())
        ]
    );
    assert_eq!(req.contents[1].parts.len(), 3);
    assert!(
        matches!(&req.contents[1].parts[2].data, Some(proto::part::Data::Text(text)) if text == "4")
    );
}

#[test]
fn genuine_gemini_reasoning_keeps_its_signature() {
    let req = encoded(request(vec![reasoning(
        "grpc thought",
        b"grpc",
        Some(REASONING_ISSUER),
    )]))
    .expect("request build");
    assert_eq!(thoughts(&req), vec![("grpc thought", b"grpc".to_vec())]);
}

#[test]
fn unknown_provenance_reasoning_keeps_its_signature() {
    let req =
        encoded(request(vec![reasoning("grpc thought", b"grpc", None)])).expect("request build");
    assert_eq!(thoughts(&req), vec![("grpc thought", b"grpc".to_vec())]);
}

#[test]
fn a_turn_emptied_only_by_foreign_reasoning_is_omitted() {
    let req = encoded(request(vec![reasoning(
        "anthropic thought",
        b"anthropic",
        Some("anthropic"),
    )]))
    .expect("request build");
    assert_eq!(req.contents.len(), 2);
    assert!(req.contents.iter().all(|content| content.role == "user"));
    assert!(
        matches!(&req.contents[0].parts[0].data, Some(proto::part::Data::Text(text)) if text == "What is 2 + 2?")
    );
    assert!(
        matches!(&req.contents[1].parts[0].data, Some(proto::part::Data::Text(text)) if text == "And 3 + 3?")
    );
}

#[test]
fn ordinary_canonical_reasoning_remains_unscoped() {
    let req =
        encoded(request(vec![AssistantContent::reasoning("ordinary")])).expect("request build");
    assert_eq!(thoughts(&req), vec![("ordinary", Vec::new())]);
}

#[test]
fn foreign_opaque_reasoning_is_refused_and_not_filtered_out() {
    let part = AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
            json!({"type":"future_part","payload":{"list":[1,2]}}),
            Some("anthropic".to_owned()),
        ))],
        CarriedPayload::new(Value::Null, Some("anthropic".to_owned())),
    ));
    let error = ProviderError::from(
        encoded(request(vec![AssistantContent::text("known"), part]))
            .expect_err("opaque is refused"),
    );
    assert!(matches!(error, ProviderError::Request(_)));
    assert!(error.to_string().contains("opaque"), "{error}");
}
