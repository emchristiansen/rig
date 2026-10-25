use super::*;

type TestResult = anyhow::Result<()>;

fn encoded(
    request: CompletionRequest,
) -> Result<vertexai::model::GenerateContentRequest, EncodeError> {
    GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary)
}

#[test]
fn generation_config_rejects_non_finite_typed_values() -> TestResult {
    let mut request = CompletionRequest::new("test");
    request.temperature = Some(f64::NAN);
    let error = match encoded(request) {
        Ok(_) => return Err(anyhow::anyhow!("non-finite typed temperature must fail")),
        Err(error) => ProviderError::from(error),
    };
    anyhow::ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected request error"
    );
    anyhow::ensure!(
        error.to_string().contains("temperature"),
        "temperature context missing"
    );
    Ok(())
}

#[test]
fn positive_and_negative_infinite_legacy_temperatures_are_refused() -> TestResult {
    for temperature in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut request = CompletionRequest::new("test");
        request.temperature = Some(temperature);
        let error = match encoded(request) {
            Ok(_) => return Err(anyhow::anyhow!("infinite typed temperature must fail")),
            Err(error) => ProviderError::from(error),
        };
        anyhow::ensure!(
            matches!(&error, ProviderError::Request(_)),
            "expected request error"
        );
        anyhow::ensure!(
            error.to_string().contains("temperature"),
            "temperature context missing"
        );
        anyhow::ensure!(
            error.report().kind == rig_core::error::ErrorKind::Request,
            "request error kind changed"
        );
    }
    Ok(())
}

#[test]
fn finite_vertex_temperatures_keep_the_original_generation_controls() -> TestResult {
    for (temperature, expected) in [(0.7_f64, 0.7_f32), (0.0, 0.0), (1.0, 1.0)] {
        let mut request = CompletionRequest::new("test");
        request.temperature = Some(temperature);
        request.max_tokens = Some(100);
        let config = encoded(request)?
            .generation_config
            .ok_or_else(|| anyhow::anyhow!("generation config missing"))?;
        anyhow::ensure!(config.temperature == Some(expected), "temperature changed");
        anyhow::ensure!(
            config.max_output_tokens == Some(100),
            "max output tokens changed"
        );
        anyhow::ensure!(config.candidate_count == Some(1), "candidate count changed");
    }
    Ok(())
}

#[test]
fn finite_typed_controls_override_raw_values_without_changing_candidate_or_schema() -> TestResult {
    let mut request = CompletionRequest::new("test");
    request.temperature = Some(0.7);
    request.max_tokens = Some(100);
    request.additional_params = Some(serde_json::json!({"generationConfig": {
        "temperature": f64::MAX,
        "maxOutputTokens": 2_147_483_648_u64,
        "candidateCount": 1,
        "responseJsonSchema": {"type": "object"},
    }}));
    let config = encoded(request)?
        .generation_config
        .ok_or_else(|| anyhow::anyhow!("generation config missing"))?;
    anyhow::ensure!(
        config.temperature == Some(0.7),
        "typed temperature override changed"
    );
    anyhow::ensure!(
        config.max_output_tokens == Some(100),
        "typed max output tokens override changed"
    );
    anyhow::ensure!(config.candidate_count == Some(1), "candidate count changed");
    anyhow::ensure!(
        serde_json::to_value(
            config
                .response_json_schema
                .ok_or_else(|| anyhow::anyhow!("JSON schema missing"))?
        )? == serde_json::json!({"type": "object"}),
        "response JSON schema changed"
    );
    Ok(())
}

use rig_core::message::{
    AssistantContent, AssistantMessage, CarriedPayload, CarriedReasoning, CarriedReasoningPart,
    Message,
};
use serde_json::{Value, json};

fn signed(text: &str, issuer: Option<&str>) -> AssistantContent {
    AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: text.to_owned(),
            signature: Some("c2lnbmF0dXJl".to_owned()),
        }],
        CarriedPayload::new(Value::Null, issuer.map(str::to_owned)),
    ))
}

fn reasoning_request(parts: Vec<AssistantContent>) -> CompletionRequest {
    let mut request = CompletionRequest::new("test");
    request.chat_history = vec![
        Message::user("What is 2 + 2?"),
        Message::Assistant(AssistantMessage::new(parts)),
        Message::user("And 3 + 3?"),
    ];
    request
}

fn thoughts(request: &vertexai::model::GenerateContentRequest) -> Vec<(String, Vec<u8>)> {
    request
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .filter(|part| part.thought)
        .filter_map(|part| {
            part.text()
                .map(|text| (text.clone(), part.thought_signature.to_vec()))
        })
        .collect()
}

#[test]
fn only_vertex_reasoning_is_replayed() -> TestResult {
    let request = encoded(reasoning_request(vec![
        signed(
            "vertex thought",
            Some(crate::types::completion_response::PROVIDER_NAME),
        ),
        signed("anthropic thought", Some("anthropic")),
        AssistantContent::text("4"),
    ]))?;
    let texts: Vec<_> = thoughts(&request)
        .into_iter()
        .map(|(text, _)| text)
        .collect();
    anyhow::ensure!(
        texts == ["vertex thought"],
        "replayed thought texts changed"
    );
    anyhow::ensure!(
        thoughts(&request) == [("vertex thought".to_owned(), b"signature".to_vec())],
        "replayed thoughts or signatures changed"
    );
    let parts = &request
        .contents
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("assistant content missing"))?
        .parts;
    anyhow::ensure!(parts.len() == 2, "assistant part count changed");
    anyhow::ensure!(
        parts
            .get(1)
            .and_then(|part| part.text())
            .map(String::as_str)
            == Some("4"),
        "assistant answer text changed"
    );
    Ok(())
}

#[test]
fn genuine_vertex_reasoning_keeps_its_signature() -> TestResult {
    let request = encoded(reasoning_request(vec![signed(
        "vertex thought",
        Some(PROVIDER_NAME),
    )]))?;
    anyhow::ensure!(
        thoughts(&request) == [("vertex thought".to_owned(), b"signature".to_vec())],
        "Vertex thought or signature changed"
    );
    Ok(())
}

#[test]
fn unknown_provenance_reasoning_keeps_its_signature() -> TestResult {
    let request = encoded(reasoning_request(vec![signed("vertex thought", None)]))?;
    anyhow::ensure!(
        thoughts(&request) == [("vertex thought".to_owned(), b"signature".to_vec())],
        "unknown-provenance thought or signature changed"
    );
    Ok(())
}

#[test]
fn a_turn_emptied_only_by_foreign_reasoning_is_omitted() -> TestResult {
    let request = encoded(reasoning_request(vec![signed(
        "anthropic thought",
        Some("anthropic"),
    )]))?;
    let contents: Vec<_> = request
        .contents
        .iter()
        .map(|content| {
            let texts: Vec<_> = content
                .parts
                .iter()
                .filter_map(|part| part.text().cloned())
                .collect();
            (content.role.clone(), texts)
        })
        .collect();
    anyhow::ensure!(
        contents
            == [
                ("user".to_owned(), vec!["What is 2 + 2?".to_owned()]),
                ("user".to_owned(), vec!["And 3 + 3?".to_owned()]),
            ],
        "remaining conversation contents changed"
    );
    Ok(())
}

#[test]
fn ordinary_canonical_reasoning_remains_unscoped() -> TestResult {
    let request = encoded(reasoning_request(vec![AssistantContent::reasoning(
        "ordinary",
    )]))?;
    anyhow::ensure!(
        thoughts(&request) == [("ordinary".to_owned(), Vec::new())],
        "canonical reasoning changed"
    );
    Ok(())
}

#[test]
fn foreign_opaque_reasoning_is_refused_and_not_filtered_out() -> TestResult {
    let raw = json!({"type":"future_part","payload":{"list":[1,2]}});
    for summary in [true, false] {
        let payload = CarriedPayload::new(raw.clone(), Some("openai".to_owned()));
        let part = if summary {
            CarriedReasoningPart::OpaqueSummary(payload)
        } else {
            CarriedReasoningPart::OpaqueContent(payload)
        };
        let reasoning = CarriedReasoning::new(
            vec![part],
            CarriedPayload::new(Value::Null, Some("openai".to_owned())),
        )
        .with_id((!summary).then(|| "rs_1".to_owned()));
        let mut request = CompletionRequest::new("test");
        request.chat_history = vec![Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("known"),
            AssistantContent::CarriedReasoning(reasoning),
        ]))];
        let error = match encoded(request) {
            Ok(_) => return Err(anyhow::anyhow!("foreign opaque reasoning must be refused")),
            Err(error) => ProviderError::from(error),
        };
        anyhow::ensure!(
            matches!(&error, ProviderError::Request(_)),
            "expected request error"
        );
        anyhow::ensure!(
            error.to_string().contains("opaque"),
            "opaque refusal context missing"
        );
    }
    Ok(())
}
