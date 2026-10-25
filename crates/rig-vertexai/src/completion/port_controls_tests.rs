use super::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::{
    AnsweredToolCall, AssistantContent, AssistantMessage, CallId, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, CustomToolCall, Message, ToolCall,
    ToolFunction, ToolName, ToolResult, ToolResultContent, UnrepresentableToolCall, UserContent,
};
use serde_json::{Value, json};
use std::error::Error;

type TestResult = anyhow::Result<()>;

fn minimal_request() -> CompletionRequest {
    CompletionRequest::new("test")
}

fn sdk_request(
    request: CompletionRequest,
) -> Result<vertexai::model::GenerateContentRequest, ProviderError> {
    GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)
        .map_err(ProviderError::from)
}

fn tool_name(value: &str) -> Result<ToolName, rig_core::message::EmptyToolName> {
    ToolName::new(value)
}

#[test]
fn vertex_refuses_namespaced_and_custom_calls() -> TestResult {
    let namespaced = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(tool_name("add")?, json!({"x": 1})),
    )
    .with_namespace(Some("math".to_owned()));
    let custom = CustomToolCall::new(
        CallId::from_wire("call_2"),
        tool_name("apply_patch")?,
        r#"{"x": 1}"#,
    );
    for (content, refusal, kind) in [
        (
            AssistantContent::ToolCall(namespaced),
            "cannot represent tool namespaces, so call `add` qualified by namespace `math`",
            "namespace",
        ),
        (
            AssistantContent::CustomToolCall(custom),
            "carries JSON tool arguments only, so custom tool call `apply_patch`",
            "custom",
        ),
    ] {
        let mut request = minimal_request();
        request.chat_history = vec![Message::Assistant(AssistantMessage::new(vec![content]))];
        let error = sdk_request(request)
            .err()
            .ok_or_else(|| anyhow::anyhow!("refused, not dropped"))?;
        let ProviderError::Request(reason) = &error else {
            return Err(anyhow::anyhow!("a request refusal, got {error:?}"));
        };
        let reason = reason.to_string();
        anyhow::ensure!(reason.starts_with("Gemini generateContent "), "{reason}");
        anyhow::ensure!(reason.contains(refusal), "{reason}");
        let typed = error
            .source()
            .and_then(|source| source.downcast_ref::<UnrepresentableToolCall>())
            .ok_or_else(|| anyhow::anyhow!("the request failure retains the typed refusal"))?;
        match (kind, typed) {
            (
                "namespace",
                UnrepresentableToolCall::Namespace {
                    wire,
                    name,
                    namespace,
                },
            ) => {
                anyhow::ensure!(
                    (*wire, name.as_str(), namespace.as_str())
                        == ("Gemini generateContent", "add", "math"),
                    "typed namespace refusal changed"
                );
            }
            ("custom", UnrepresentableToolCall::CustomCall { wire, name }) => {
                anyhow::ensure!(
                    (*wire, name.as_str()) == ("Gemini generateContent", "apply_patch"),
                    "typed custom-call refusal changed"
                );
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "expected the {kind} refusal, got {typed:?}"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn tool_result_serializes_the_executed_name_not_an_identifier() -> TestResult {
    let call = |wire_id: &str, name: &str| -> Result<Message, rig_core::message::EmptyToolName> {
        Ok(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire(wire_id),
                ToolFunction::new(tool_name(name)?, json!({})),
            )),
        ])))
    };
    let result = |wire_id: &str, name: &str| -> Result<Message, rig_core::message::EmptyToolName> {
        Ok(Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire(wire_id),
                name: tool_name(name)?,
                answers: AnsweredToolCall::Function,
                content: vec![ToolResultContent::text("out")],
                is_error: false,
            })],
        })
    };
    let mut request = minimal_request();
    request.chat_history = vec![
        call("call_1", "add")?,
        result("call_1", "sum")?,
        call("call_abc", "get_weather")?,
        result("call_abc", "get_weather")?,
        call("call_9", "get_time")?,
        result("call_9", "get_time")?,
    ];
    let contents = sdk_request(request)?.contents;
    let response_names: Vec<String> = contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| part.function_response().map(|fr| fr.name.clone()))
        .collect();
    anyhow::ensure!(
        response_names
            == vec![
                "sum".to_owned(),
                "get_weather".to_owned(),
                "get_time".to_owned(),
            ],
        "executed tool response names changed"
    );
    Ok(())
}

#[test]
fn test_system_instruction_from_preamble() -> TestResult {
    let mut request = minimal_request();
    request
        .chat_history
        .insert(0, Message::system("You are a helpful assistant."));
    let system_instruction = sdk_request(request)?.system_instruction;
    anyhow::ensure!(system_instruction.is_some(), "system instruction missing");
    let content = system_instruction.ok_or_else(|| anyhow::anyhow!("system instruction exists"))?;
    anyhow::ensure!(
        content.role.as_str() == "user",
        "system instruction role changed"
    );
    anyhow::ensure!(
        content.parts.len() == 1,
        "system instruction part count changed"
    );
    anyhow::ensure!(
        content.parts[0].text() == Some(&"You are a helpful assistant.".to_string()),
        "system instruction text changed"
    );
    Ok(())
}

#[test]
fn test_no_tool_choice_when_not_specified() -> TestResult {
    let mut request = minimal_request();
    request.tool_choice = None;
    request.tools = vec![ToolDefinition {
        name: tool_name("test_tool")?,
        description: "A test tool".to_string(),
        parameters: json!({"type": "object", "properties": {}}),
    }];
    let tool_config = sdk_request(request)?.tool_config;
    anyhow::ensure!(tool_config.is_some(), "tool config missing");
    let config = tool_config.ok_or_else(|| anyhow::anyhow!("tool config exists"))?;
    let function_calling_config = config.function_calling_config.as_ref();
    anyhow::ensure!(
        function_calling_config.is_some(),
        "function calling config missing"
    );
    anyhow::ensure!(
        function_calling_config
            .ok_or_else(|| anyhow::anyhow!("function calling config exists"))?
            .mode
            == vertexai::model::function_calling_config::Mode::Auto,
        "default function calling mode changed"
    );
    Ok(())
}

#[test]
fn generation_config_maps_thinking_levels() -> TestResult {
    use vertexai::model::generation_config::thinking_config::ThinkingLevel as VertexThinkingLevel;
    for (level, expected) in [
        ("minimal", VertexThinkingLevel::Minimal),
        ("low", VertexThinkingLevel::Low),
        ("medium", VertexThinkingLevel::Medium),
        ("high", VertexThinkingLevel::High),
    ] {
        let mut request = minimal_request();
        request.additional_params = Some(json!({
            "generationConfig": {"thinkingConfig": {"thinkingLevel": level}},
        }));
        let config = sdk_request(request)?
            .generation_config
            .ok_or_else(|| anyhow::anyhow!("generation config exists"))?;
        let thinking = config
            .thinking_config
            .ok_or_else(|| anyhow::anyhow!("thinking config should be mapped"))?;
        anyhow::ensure!(
            thinking.thinking_level == Some(expected),
            "thinking level changed"
        );
    }
    Ok(())
}

#[test]
fn generation_config_rejects_audio_response_modality() -> TestResult {
    let mut request = minimal_request();
    request.additional_params =
        Some(json!({"generationConfig": {"responseModalities": ["AUDIO"]}}));
    let error = sdk_request(request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("audio output must fail before the API call"))?;
    anyhow::ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected request error"
    );
    anyhow::ensure!(
        error.to_string().contains("responseModalities AUDIO"),
        "audio modality refusal context missing"
    );
    anyhow::ensure!(
        error
            .source()
            .ok_or_else(|| anyhow::anyhow!("the original leaf is retained"))?
            .to_string()
            == "responseModalities AUDIO is unsupported because Rig cannot represent assistant audio responses",
        "audio modality refusal source changed"
    );
    Ok(())
}

fn assert_sdk_data_error(error: &ProviderError, expected: &str) -> TestResult {
    anyhow::ensure!(
        matches!(error, ProviderError::Request(_)),
        "expected request error"
    );
    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<serde_json::Error>())
        .ok_or_else(|| anyhow::anyhow!("actual SDK serde error retained"))?;
    anyhow::ensure!(source.is_data(), "expected SDK data error");
    anyhow::ensure!(source.to_string().contains(expected), "{source}");
    Ok(())
}

#[test]
fn generation_config_rejects_out_of_f32_range_typed_and_provider_values() -> TestResult {
    let mut typed_request = minimal_request();
    typed_request.temperature = Some(f64::MAX);
    let typed_error = sdk_request(typed_request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("typed temperature beyond f32 must fail"))?;
    assert_sdk_data_error(&typed_error, "a 32-bit float in ProtoJSON format")?;
    let mut provider_request = minimal_request();
    provider_request.additional_params = Some(json!({"generationConfig": {"topP": f64::MAX}}));
    let provider_error = sdk_request(provider_request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("provider top_p beyond f32 must fail"))?;
    assert_sdk_data_error(&provider_error, "a 32-bit float in ProtoJSON format")?;
    Ok(())
}

#[test]
fn generation_config_rejects_out_of_range_max_output_tokens_without_typed_override() -> TestResult {
    let mut provider_request = minimal_request();
    provider_request.additional_params =
        Some(json!({"generationConfig": {"maxOutputTokens": 2_147_483_648_u64}}));
    let provider_error = sdk_request(provider_request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("provider max output tokens beyond i32 must fail"))?;
    assert_sdk_data_error(&provider_error, "a 32-bit signed integer")?;
    let mut typed_request = minimal_request();
    typed_request.max_tokens = Some(2_147_483_648);
    let typed_error = sdk_request(typed_request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("typed max output tokens beyond i32 must fail"))?;
    assert_sdk_data_error(&typed_error, "a 32-bit signed integer")?;
    Ok(())
}

#[test]
fn generation_config_rejects_conflicting_response_schema_forms() -> TestResult {
    for json_schema_key in ["responseJsonSchema", "_responseJsonSchema"] {
        let mut request = minimal_request();
        request.additional_params = Some(json!({"generationConfig": {
            "responseSchema": {"type": "OBJECT"}, (json_schema_key): {"type": "object"},
        }}));
        let error = sdk_request(request)
            .err()
            .ok_or_else(|| anyhow::anyhow!("response schema forms cannot be combined"))?;
        anyhow::ensure!(
            matches!(&error, ProviderError::Request(_)),
            "expected request error"
        );
        anyhow::ensure!(
            error
                .to_string()
                .contains("responseSchema cannot be combined"),
            "schema conflict refusal context missing"
        );
        anyhow::ensure!(
            error
                .source()
                .ok_or_else(|| anyhow::anyhow!("the original leaf is retained"))?
                .to_string()
                == "responseSchema cannot be combined with responseJsonSchema or _responseJsonSchema",
            "schema conflict refusal source changed"
        );
    }
    Ok(())
}

#[test]
fn generation_config_rejects_both_response_json_schema_aliases() -> TestResult {
    let mut request = minimal_request();
    request.additional_params = Some(json!({"generationConfig": {
        "responseJsonSchema": {"type": "object"}, "_responseJsonSchema": {"type": "object"},
    }}));
    let error = sdk_request(request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("JSON schema aliases cannot be combined"))?;
    anyhow::ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected request error"
    );
    anyhow::ensure!(
        error
            .to_string()
            .contains("responseJsonSchema cannot be combined with _responseJsonSchema"),
        "schema alias conflict refusal context missing"
    );
    anyhow::ensure!(
        error
            .source()
            .ok_or_else(|| anyhow::anyhow!("the original leaf is retained"))?
            .to_string()
            == "responseJsonSchema cannot be combined with _responseJsonSchema",
        "schema alias conflict refusal source changed"
    );
    Ok(())
}

#[test]
fn generation_config_rejects_invalid_thinking_config() -> TestResult {
    let mut malformed_request = minimal_request();
    malformed_request.additional_params = Some(json!({"generationConfig": {
        "thinkingConfig": {"thinkingBudget": "invalid"},
    }}));
    anyhow::ensure!(
        sdk_request(malformed_request).is_err(),
        "malformed thinking config must fail"
    );
    let mut conflicting_request = minimal_request();
    conflicting_request.additional_params = Some(json!({"generationConfig": {
        "thinkingConfig": {"thinkingBudget": 1024, "thinkingLevel": "high"},
    }}));
    let error = sdk_request(conflicting_request)
        .err()
        .ok_or_else(|| anyhow::anyhow!("mutually exclusive thinking values must fail"))?;
    anyhow::ensure!(
        error
            .to_string()
            .contains("thinking_budget and thinking_level cannot both be set"),
        "thinking conflict refusal context missing"
    );
    anyhow::ensure!(
        matches!(&error, ProviderError::Request(_)),
        "expected request error"
    );
    anyhow::ensure!(
        error
            .source()
            .ok_or_else(|| anyhow::anyhow!("the original leaf is retained"))?
            .to_string()
            == "thinking_budget and thinking_level cannot both be set",
        "thinking conflict refusal source changed"
    );
    Ok(())
}

#[test]
fn opaque_responses_parts_are_refused_before_request_content_is_discarded() -> TestResult {
    let raw = json!({"type":"future_part","payload":{"list":[1,2]}});
    let opaque = AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(raw.clone(), None),
    ));
    let summary = AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
            raw.clone(),
            Some("openai".into()),
        ))],
        CarriedPayload::new(Value::Null, Some("openai".into())),
    ));
    let content = AssistantContent::CarriedReasoning(
        CarriedReasoning::new(
            vec![CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
                raw,
                Some("openai".into()),
            ))],
            CarriedPayload::new(Value::Null, Some("openai".into())),
        )
        .with_id(Some("rs_1".into())),
    );
    for part in [opaque, summary, content] {
        let mut request = minimal_request();
        request.chat_history = vec![Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("known"),
            part,
        ]))];
        let error = sdk_request(request)
            .err()
            .ok_or_else(|| anyhow::anyhow!("opaque content refused"))?;
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

#[test]
fn vertex_base_defaults_preserve_raw_only_and_raw_override_controls() -> TestResult {
    let encoded = sdk_request(minimal_request())?;
    anyhow::ensure!(
        encoded.system_instruction.is_none(),
        "unexpected default system instruction"
    );
    anyhow::ensure!(
        encoded.tool_config.is_none(),
        "unexpected default tool config"
    );
    let mut raw_system = minimal_request();
    raw_system.additional_params = Some(json!({"systemInstruction": {
        "role": "model", "parts": [{"text": "You are a helpful assistant."}],
    }}));
    anyhow::ensure!(
        sdk_request(raw_system)?
            .system_instruction
            .ok_or_else(|| anyhow::anyhow!("raw system exists"))?
            .role
            == "model",
        "raw system instruction role changed"
    );
    let mut raw_choice = minimal_request();
    raw_choice.tools = vec![ToolDefinition {
        name: tool_name("test_tool")?,
        description: "A test tool".into(),
        parameters: json!({"type": "object", "properties": {}}),
    }];
    raw_choice.additional_params =
        Some(json!({"toolConfig": {"functionCallingConfig": {"mode": "NONE"}}}));
    anyhow::ensure!(
        sdk_request(raw_choice)?
            .tool_config
            .ok_or_else(|| anyhow::anyhow!("raw choice exists"))?
            .function_calling_config
            .ok_or_else(|| anyhow::anyhow!("function calling config exists"))?
            .mode
            == vertexai::model::function_calling_config::Mode::None,
        "raw function calling mode override changed"
    );
    Ok(())
}

#[test]
fn portable_effort_reaches_known_sdk_thinking_level() -> TestResult {
    let encoded = GenerateContent::new("gemini-3-flash-preview").encode(
        minimal_request().reasoning(rig_core::completion::Effort::High),
        Mode::Unary,
    )?;
    anyhow::ensure!(
        encoded
            .generation_config
            .ok_or_else(|| anyhow::anyhow!("config exists"))?
            .thinking_config
            .ok_or_else(|| anyhow::anyhow!("thinking exists"))?
            .thinking_level
            == Some(vertexai::model::generation_config::thinking_config::ThinkingLevel::High),
        "portable reasoning effort mapping changed"
    );
    Ok(())
}

#[test]
fn future_thinking_level_remains_sdk_owned() -> TestResult {
    let mut request = minimal_request();
    request.additional_params =
        Some(json!({"generationConfig": {"thinkingConfig": {"thinkingLevel": "future"}}}));
    let level = sdk_request(request)?
        .generation_config
        .ok_or_else(|| anyhow::anyhow!("config exists"))?
        .thinking_config
        .ok_or_else(|| anyhow::anyhow!("thinking exists"))?
        .thinking_level
        .ok_or_else(|| anyhow::anyhow!("level exists"))?;
    anyhow::ensure!(
        serde_json::to_value(level)? == json!("future"),
        "SDK future thinking level changed"
    );
    Ok(())
}

#[test]
fn standalone_schema_alias_preserves_the_exact_schema() -> TestResult {
    let mut request = minimal_request();
    request.additional_params =
        Some(json!({"generationConfig": {"_responseJsonSchema": {"type": "object"}}}));
    let config = sdk_request(request)?
        .generation_config
        .ok_or_else(|| anyhow::anyhow!("config exists"))?;
    anyhow::ensure!(
        serde_json::to_value(
            config
                .response_json_schema
                .ok_or_else(|| anyhow::anyhow!("JSON schema exists"))?
        )? == json!({"type": "object"}),
        "standalone response schema alias changed"
    );
    Ok(())
}

#[test]
fn portable_reasoning_and_opposite_raw_field_meet_vertex_exclusion() -> TestResult {
    for (model, reasoning, opposite) in [
        (
            "gemini-3-flash-preview",
            rig_core::completion::Reasoning::Effort(rig_core::completion::Effort::High),
            json!({"thinkingBudget": 1024}),
        ),
        (
            GEMINI_2_5_FLASH,
            rig_core::completion::Reasoning::Budget { tokens: 1024 },
            json!({"thinkingLevel": "high"}),
        ),
    ] {
        let request = minimal_request()
            .reasoning(reasoning)
            .additional_params(json!({"generationConfig": {"thinkingConfig": opposite}}));
        let error = GenerateContent::new(model)
            .encode(request, Mode::Unary)
            .err()
            .map(ProviderError::from)
            .ok_or_else(|| anyhow::anyhow!("both thinking values refused"))?;
        anyhow::ensure!(
            matches!(&error, ProviderError::Request(_)),
            "expected request error"
        );
        anyhow::ensure!(
            error
                .source()
                .ok_or_else(|| anyhow::anyhow!("exact leaf"))?
                .to_string()
                == "thinking_budget and thinking_level cannot both be set",
            "portable and raw thinking conflict refusal source changed"
        );
    }
    Ok(())
}
