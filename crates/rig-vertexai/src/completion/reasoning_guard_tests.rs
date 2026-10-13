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
