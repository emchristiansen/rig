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

fn encoded(request: CompletionRequest) -> Result<Value, EncodeError> {
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    Ok(crate::test_utils::json_body(
        &wire.encode(request, Mode::Unary)?.request,
    ))
}

#[test]
fn only_gemini_reasoning_is_replayed() {
    let body = encoded(request(vec![
        reasoning("grpc thought", b"grpc", Some(PROVIDER_NAME)),
        reasoning("rest thought", b"rest", Some(PROVIDER_NAME)),
        reasoning("anthropic thought", b"anthropic", Some("anthropic")),
        AssistantContent::text("4"),
    ]))
    .expect("request build");
    let signatures: Vec<_> = body["contents"]
        .as_array()
        .expect("contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("parts"))
        .filter(|part| part["thought"] == true)
        .map(|part| {
            base64::prelude::BASE64_STANDARD
                .decode(part["thoughtSignature"].as_str().expect("signature"))
                .expect("signature bytes")
        })
        .collect();
    assert_eq!(signatures, vec![b"grpc".to_vec(), b"rest".to_vec()]);
    assert_eq!(body["contents"][1]["parts"][0]["text"], "grpc thought");
    assert_eq!(body["contents"][1]["parts"][1]["text"], "rest thought");
    assert_eq!(body["contents"][1]["parts"][2]["text"], "4");
    assert_eq!(
        body["contents"][1]["parts"]
            .as_array()
            .expect("parts")
            .len(),
        3
    );
}

#[test]
fn genuine_gemini_reasoning_keeps_its_signature() {
    let body = encoded(request(vec![reasoning(
        "grpc thought",
        b"grpc",
        Some(PROVIDER_NAME),
    )]))
    .expect("request build");
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"thought":true,"text":"grpc thought","thoughtSignature":"Z3JwYw=="}])
    );
}

#[test]
fn unknown_provenance_reasoning_keeps_its_signature() {
    let body =
        encoded(request(vec![reasoning("grpc thought", b"grpc", None)])).expect("request build");
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"thought":true,"text":"grpc thought","thoughtSignature":"Z3JwYw=="}])
    );
}

#[test]
fn a_turn_emptied_only_by_foreign_reasoning_is_omitted() {
    let body = encoded(request(vec![reasoning(
        "anthropic thought",
        b"anthropic",
        Some("anthropic"),
    )]))
    .expect("request build");
    assert_eq!(
        body["contents"],
        json!([
            {"role":"user","parts":[{"text":"What is 2 + 2?","thought":false}]},
            {"role":"user","parts":[{"text":"And 3 + 3?","thought":false}]}
        ])
    );
}

#[test]
fn ordinary_canonical_reasoning_remains_unscoped() {
    let body =
        encoded(request(vec![AssistantContent::reasoning("ordinary")])).expect("request build");
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"thought":true,"text":"ordinary"}])
    );
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
