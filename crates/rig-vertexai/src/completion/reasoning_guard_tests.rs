use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

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
        Ok(_) => return Err("non-finite typed temperature must fail".into()),
        Err(error) => ProviderError::from(error),
    };
    assert!(matches!(&error, ProviderError::Request(_)));
    assert!(error.to_string().contains("temperature"));
    Ok(())
}

#[test]
fn positive_and_negative_infinite_legacy_temperatures_are_refused() -> TestResult {
    for temperature in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut request = CompletionRequest::new("test");
        request.temperature = Some(temperature);
        let error = match encoded(request) {
            Ok(_) => return Err("infinite typed temperature must fail".into()),
            Err(error) => ProviderError::from(error),
        };
        assert!(matches!(&error, ProviderError::Request(_)));
        assert!(error.to_string().contains("temperature"));
        assert_eq!(error.report().kind, rig_core::error::ErrorKind::Request);
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
            .ok_or("generation config missing")?;
        assert_eq!(config.temperature, Some(expected));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.candidate_count, Some(1));
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
        "candidateCount": 2,
        "responseJsonSchema": {"type": "object"},
    }}));
    let config = encoded(request)?
        .generation_config
        .ok_or("generation config missing")?;
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(100));
    assert_eq!(config.candidate_count, Some(1));
    assert_eq!(
        serde_json::to_value(config.response_json_schema.ok_or("JSON schema missing")?)?,
        serde_json::json!({"type": "object"}),
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
    assert_eq!(texts, ["vertex thought"]);
    assert_eq!(
        thoughts(&request),
        [("vertex thought".to_owned(), b"signature".to_vec())]
    );
    let parts = &request
        .contents
        .get(1)
        .ok_or("assistant content missing")?
        .parts;
    assert_eq!(parts.len(), 2);
    assert_eq!(
        parts
            .get(1)
            .and_then(|part| part.text())
            .map(String::as_str),
        Some("4")
    );
    Ok(())
}

#[test]
fn genuine_vertex_reasoning_keeps_its_signature() -> TestResult {
    let request = encoded(reasoning_request(vec![signed(
        "vertex thought",
        Some(PROVIDER_NAME),
    )]))?;
    assert_eq!(
        thoughts(&request),
        [("vertex thought".to_owned(), b"signature".to_vec())]
    );
    Ok(())
}

#[test]
fn unknown_provenance_reasoning_keeps_its_signature() -> TestResult {
    let request = encoded(reasoning_request(vec![signed("vertex thought", None)]))?;
    assert_eq!(
        thoughts(&request),
        [("vertex thought".to_owned(), b"signature".to_vec())]
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
    assert_eq!(
        contents,
        [
            ("user".to_owned(), vec!["What is 2 + 2?".to_owned()]),
            ("user".to_owned(), vec!["And 3 + 3?".to_owned()]),
        ]
    );
    Ok(())
}

#[test]
fn ordinary_canonical_reasoning_remains_unscoped() -> TestResult {
    let request = encoded(reasoning_request(vec![AssistantContent::reasoning(
        "ordinary",
    )]))?;
    assert_eq!(thoughts(&request), [("ordinary".to_owned(), Vec::new())]);
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
            Ok(_) => return Err("foreign opaque reasoning must be refused".into()),
            Err(error) => ProviderError::from(error),
        };
        assert!(matches!(&error, ProviderError::Request(_)));
        assert!(error.to_string().contains("opaque"));
    }
    Ok(())
}
