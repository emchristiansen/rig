use super::*;
use anyhow::{Context, ensure};
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
fn only_gemini_reasoning_is_replayed() -> anyhow::Result<()> {
    let req = encoded(request(vec![
        reasoning("grpc thought", b"grpc", Some(REASONING_ISSUER)),
        reasoning("rest thought", b"rest", Some(REASONING_ISSUER)),
        reasoning("anthropic thought", b"anthropic", Some("anthropic")),
        AssistantContent::text("4"),
    ]))
    .context("request build")?;
    ensure!(
        thoughts(&req)
            == vec![
                ("grpc thought", b"grpc".to_vec()),
                ("rest thought", b"rest".to_vec())
            ],
        "replayed reasoning differs: {req:?}"
    );
    let content = req
        .contents
        .get(1)
        .with_context(|| format!("second content is missing: {req:?}"))?;
    ensure!(
        content.parts.len() == 3,
        "expected three parts: {content:?}"
    );
    let part = content
        .parts
        .get(2)
        .with_context(|| format!("third part is missing: {req:?}"))?;
    ensure!(
        matches!(&part.data, Some(proto::part::Data::Text(text)) if text == "4"),
        "the answer part differs: {part:?}"
    );
    Ok(())
}

#[test]
fn genuine_gemini_reasoning_keeps_its_signature() -> anyhow::Result<()> {
    let req = encoded(request(vec![reasoning(
        "grpc thought",
        b"grpc",
        Some(REASONING_ISSUER),
    )]))
    .context("request build")?;
    ensure!(
        thoughts(&req) == vec![("grpc thought", b"grpc".to_vec())],
        "genuine reasoning signature differs: {req:?}"
    );
    Ok(())
}

#[test]
fn unknown_provenance_reasoning_keeps_its_signature() -> anyhow::Result<()> {
    let req = encoded(request(vec![reasoning("grpc thought", b"grpc", None)]))
        .context("request build")?;
    ensure!(
        thoughts(&req) == vec![("grpc thought", b"grpc".to_vec())],
        "unscoped reasoning signature differs: {req:?}"
    );
    Ok(())
}

#[test]
fn a_turn_emptied_only_by_foreign_reasoning_is_omitted() -> anyhow::Result<()> {
    let req = encoded(request(vec![reasoning(
        "anthropic thought",
        b"anthropic",
        Some("anthropic"),
    )]))
    .context("request build")?;
    ensure!(req.contents.len() == 2, "expected two contents: {req:?}");
    ensure!(
        req.contents.iter().all(|content| content.role == "user"),
        "only user contents should remain: {req:?}"
    );
    let first = req
        .contents
        .first()
        .and_then(|content| content.parts.first())
        .with_context(|| format!("first user part is missing: {req:?}"))?;
    ensure!(
        matches!(&first.data, Some(proto::part::Data::Text(text)) if text == "What is 2 + 2?"),
        "first user part differs: {first:?}"
    );
    let second = req
        .contents
        .get(1)
        .and_then(|content| content.parts.first())
        .with_context(|| format!("second user part is missing: {req:?}"))?;
    ensure!(
        matches!(&second.data, Some(proto::part::Data::Text(text)) if text == "And 3 + 3?"),
        "second user part differs: {second:?}"
    );
    Ok(())
}

#[test]
fn ordinary_canonical_reasoning_remains_unscoped() -> anyhow::Result<()> {
    let req =
        encoded(request(vec![AssistantContent::reasoning("ordinary")])).context("request build")?;
    ensure!(
        thoughts(&req) == vec![("ordinary", Vec::new())],
        "ordinary canonical reasoning differs: {req:?}"
    );
    Ok(())
}

#[test]
fn foreign_opaque_reasoning_is_refused_and_not_filtered_out() -> anyhow::Result<()> {
    let part = AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
            json!({"type":"future_part","payload":{"list":[1,2]}}),
            Some("anthropic".to_owned()),
        ))],
        CarriedPayload::new(Value::Null, Some("anthropic".to_owned())),
    ));
    let error = match encoded(request(vec![AssistantContent::text("known"), part])) {
        Ok(request) => anyhow::bail!("opaque is refused, got successful request: {request:?}"),
        Err(error) => ProviderError::from(error),
    };
    ensure!(matches!(error, ProviderError::Request(_)), "{error:?}");
    ensure!(error.to_string().contains("opaque"), "{error}");
    Ok(())
}
