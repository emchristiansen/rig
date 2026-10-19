use super::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Message, Text, ToolCall, ToolChoice, ToolFunction,
    ToolName, ToolResult, ToolResultContent, UserContent,
};
use rig_core::wire::Wire as _;

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the fixture tool name is nonempty")
}

fn minimal_request() -> CompletionRequest {
    CompletionRequest::new("test")
}

fn sdk_request(
    request: CompletionRequest,
) -> Result<vertexai::model::GenerateContentRequest, EncodeError> {
    GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary)
}

#[test]
fn test_user_text_message_conversion() {
    let message = Message::User {
        content: vec![rig_core::message::UserContent::Text(Text::new(
            "Hello".to_string(),
        ))],
    };

    let mut request = minimal_request();
    request.chat_history = vec![message];
    let vertex_content = sdk_request(request).map(|request| request.contents[0].clone());

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "user");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].text(), Some(&"Hello".to_string()));
}

#[test]
fn test_assistant_text_message_conversion() {
    let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::Text(
        Text::new("Hi there".to_string()),
    )]));

    let mut request = minimal_request();
    request.chat_history = vec![message];
    let vertex_content = sdk_request(request).map(|request| request.contents[0].clone());

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "model");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].text(), Some(&"Hi there".to_string()));
}

#[test]
fn test_assistant_tool_call_message_conversion() {
    let tool_call = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(name("add"), serde_json::json!({"x": 5, "y": 3})),
    );
    let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        tool_call,
    )]));
    let mut request = minimal_request();
    request.chat_history = vec![message];
    let vertex_content = sdk_request(request).map(|request| request.contents[0].clone());
    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "model");
    assert_eq!(content.parts.len(), 1);
    let function_call = content.parts[0].function_call();
    assert!(function_call.is_some());
    let function_call = function_call.unwrap();
    assert_eq!(function_call.name.as_str(), "add");
}

#[test]
fn image_tool_result_maps_to_native_function_response_part() {
    let raw = vec![0, 1, 2, 255];
    let message = Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("inspect"),
            answers: rig_core::message::AnsweredToolCall::Function,
            content: vec![ToolResultContent::image_base64(
                BASE64.encode(&raw),
                Some(rig_core::message::ImageMediaType::PNG),
                None,
            )],
            is_error: false,
        })],
    };
    let mut request = minimal_request();
    request.chat_history = vec![message];
    let request = sdk_request(request).expect("image tool result should convert");
    let content = &request.contents[0];
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(response.parts.len(), 1);
    let blob = response.parts[0].inline_data().expect("inline image");
    assert_eq!(blob.mime_type, "image/png");
    assert_eq!(blob.data.as_ref(), raw.as_slice());
    assert_eq!(blob.display_name, "rig_tool_result_image_0");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!({"$ref": "rig_tool_result_image_0"}))
    );
}

#[test]
fn test_tool_choice_auto_conversion() {
    // Test that rig's ToolChoice::Auto converts to Vertex AI Auto mode
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = Some(ToolChoice::Auto);
    request.tools = vec![ToolDefinition {
        name: name("test_tool"),
        description: "A test tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tool_config = encoded.tool_config;

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::Auto
    );
}

#[test]
fn test_tool_choice_required_conversion() {
    // Test that rig's ToolChoice::Required converts to Vertex AI Any mode
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = Some(ToolChoice::Required);
    request.tools = vec![ToolDefinition {
        name: name("test_tool"),
        description: "A test tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tool_config = encoded.tool_config;

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::Any
    );
}

#[test]
fn test_tool_choice_none_conversion() {
    // Test that rig's ToolChoice::None converts to Vertex AI None mode
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = Some(ToolChoice::None);
    request.tools = vec![ToolDefinition {
        name: name("test_tool"),
        description: "A test tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tool_config = encoded.tool_config;

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::None
    );
}

#[test]
fn test_tool_choice_specific_conversion() {
    // Test that rig's ToolChoice::Specific converts to Vertex AI Any mode with allowed function names
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = Some(ToolChoice::Specific {
        function_names: vec![name("test_tool")],
    });
    request.tools = vec![ToolDefinition {
        name: name("test_tool"),
        description: "A test tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tool_config = encoded.tool_config;

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    let fcc = function_calling_config.unwrap();
    assert_eq!(
        fcc.mode,
        vertexai::model::function_calling_config::Mode::Any
    );
    // Verify allowed function names are set
    assert!(!fcc.allowed_function_names.is_empty());
    assert_eq!(fcc.allowed_function_names.len(), 1);
    assert_eq!(fcc.allowed_function_names[0], "test_tool");
}

#[test]
fn test_system_instruction_from_system_history_and_contents_skip_system() {
    let mut request = minimal_request();
    request.model = None;
    request.chat_history = vec![
        Message::system("System from history"),
        Message::User {
            content: vec![UserContent::Text(Text::new("hello".to_string()))],
        },
    ];

    let vertex_request = sdk_request(request).expect("the fixture encodes");
    let system_instruction = vertex_request.system_instruction.clone();
    assert!(system_instruction.is_some());
    let system_instruction = system_instruction.unwrap();
    assert_eq!(system_instruction.parts.len(), 1);
    assert_eq!(
        system_instruction.parts[0].text(),
        Some(&"System from history".to_string())
    );

    let contents = vertex_request.contents;
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0].role.as_str(), "user");
}

#[test]
fn test_tools_conversion() {
    // Test that ToolDefinition converts to FunctionDeclaration
    let mut request = minimal_request();
    request.model = None;
    request.tools = vec![ToolDefinition {
        name: name("add"),
        description: "Add two numbers".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"]
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tools = encoded.tools.into_iter().next();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    // Verify function declarations exist
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "add");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Add two numbers"
    );
}

#[test]
fn test_tool_with_empty_parameters() {
    // Test that tools with empty parameters work correctly
    let mut request = minimal_request();
    request.model = None;
    request.tools = vec![ToolDefinition {
        name: name("document_list"),
        description: "Lists all documents".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tools = encoded.tools.into_iter().next();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "document_list");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Lists all documents"
    );
}

#[test]
fn test_tool_with_parameters() {
    // Test that tools with complex parameters work correctly
    let mut request = minimal_request();
    request.model = None;
    request.tools = vec![ToolDefinition {
        name: name("get_weather"),
        description: "Get weather for a location".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }),
    }];

    let encoded = sdk_request(request).expect("the fixture encodes");
    let tools = encoded.tools.into_iter().next();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "get_weather");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Get weather for a location"
    );
}

#[test]
fn test_generation_config_with_temperature_and_max_tokens() {
    // Test that temperature and max_tokens convert to GenerationConfig
    let mut request = minimal_request();
    request.model = None;
    request.temperature = Some(0.7);
    request.max_tokens = Some(100);

    let generation_config = sdk_request(request)
        .expect("generation config should parse")
        .generation_config;

    assert!(generation_config.is_some());
    let config = generation_config.unwrap();
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(100));
    assert_eq!(config.candidate_count, Some(1));
}

#[test]
fn generation_config_maps_thinking_budget_and_include_thoughts() {
    let mut request = minimal_request();
    request.additional_params = Some(serde_json::json!({
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 1024,
                "includeThoughts": true
            }
        }
    }));

    let config = sdk_request(request)
        .expect("generation config should parse")
        .generation_config
        .expect("generation config should exist");
    let thinking = config
        .thinking_config
        .expect("thinking config should be mapped");

    assert_eq!(thinking.thinking_budget, Some(1024));
    assert_eq!(thinking.include_thoughts, Some(true));
    assert_eq!(thinking.thinking_level, None);
}

#[test]
fn generation_config_maps_supported_fields_and_typed_fields_take_precedence() {
    let mut request = minimal_request();
    request.temperature = Some(0.7);
    request.max_tokens = Some(100);
    request.additional_params = Some(serde_json::json!({
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 20,
            "candidateCount": 2,
            "stopSequences": ["END"],
            "responseMimeType": "application/json",
            "responseJsonSchema": { "type": "object" },
            "topP": 0.8,
            "topK": 40,
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "responseLogprobs": true,
            "logprobs": 3,
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": "16:9",
                "imageSize": "1K"
            }
        }
    }));

    let config = sdk_request(request)
        .expect("generation config should parse")
        .generation_config
        .expect("generation config should exist");

    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(100));
    assert_eq!(config.candidate_count, Some(1));
    assert_eq!(config.stop_sequences, ["END"]);
    assert_eq!(config.response_mime_type, "application/json");
    assert_eq!(
        serde_json::to_value(
            config
                .response_json_schema
                .expect("JSON schema should be mapped")
        )
        .expect("Vertex JSON schema should serialize"),
        serde_json::json!({ "type": "object" })
    );
    assert_eq!(config.top_p, Some(0.8));
    assert_eq!(config.top_k, Some(40.0));
    assert_eq!(config.presence_penalty, Some(0.1));
    assert_eq!(config.frequency_penalty, Some(0.2));
    assert_eq!(config.response_logprobs, Some(true));
    assert_eq!(config.logprobs, Some(3));
    assert_eq!(
        config.response_modalities,
        [
            vertexai::model::generation_config::Modality::Text,
            vertexai::model::generation_config::Modality::Image,
        ]
    );
    let image = config.image_config.expect("image config should be mapped");
    assert_eq!(image.aspect_ratio.as_deref(), Some("16:9"));
    assert_eq!(image.image_size.as_deref(), Some("1K"));
}

#[test]
fn generation_config_maps_response_schema() {
    let mut request = minimal_request();
    request.additional_params = Some(serde_json::json!({
        "generationConfig": {
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": { "type": "STRING" }
                }
            }
        }
    }));

    let config = sdk_request(request)
        .expect("generation config should parse")
        .generation_config
        .expect("generation config should exist");

    let schema = config
        .response_schema
        .expect("response schema should be mapped");
    assert_eq!(schema.r#type, vertexai::model::Type::Object);
    assert_eq!(
        schema
            .properties
            .get("name")
            .map(|property| &property.r#type),
        Some(&vertexai::model::Type::String)
    );
}

#[test]
fn generation_config_typed_max_overrides_out_of_range_provider_max() {
    let mut request = minimal_request();
    request.max_tokens = Some(100);
    request.additional_params = Some(serde_json::json!({
        "generationConfig": { "maxOutputTokens": 2_147_483_648_u64 }
    }));

    let config = sdk_request(request)
        .expect("typed max should override an unrepresentable provider max")
        .generation_config
        .expect("generation config should exist");
    assert_eq!(config.max_output_tokens, Some(100));
}

#[test]
fn generation_config_typed_temperature_overrides_out_of_range_provider_temperature() {
    let mut request = minimal_request();
    request.temperature = Some(0.7);
    request.additional_params = Some(serde_json::json!({
        "generationConfig": { "temperature": f64::MAX }
    }));

    let config = sdk_request(request)
        .expect("typed temperature should override an unrepresentable provider temperature")
        .generation_config
        .expect("generation config should exist");
    assert_eq!(config.temperature, Some(0.7));
}

/// This exact original reply has no finish reason and cannot finish a unary turn.
#[test]
fn test_assistant_image_response_round_trips_through_history_in_order()
-> Result<(), Box<dyn std::error::Error>> {
    let raw_image = vec![0, 1, 2, 255];
    let response = vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new().set_content(
            vertexai::model::Content::new()
                .set_role("model")
                .set_parts([
                    vertexai::model::Part::new().set_text("before"),
                    vertexai::model::Part::new().set_inline_data(
                        vertexai::model::Blob::new()
                            .set_mime_type("image/png")
                            .set_data(raw_image.clone()),
                    ),
                    vertexai::model::Part::new().set_text("after"),
                ]),
        ),
    ]);
    let received =
        serde_json::Value::Object(crate::types::completion_response::rest_chunk(&response)?);
    assert!(received.pointer("/candidates/0/finishReason").is_none());
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let (partial, ended) =
        rig_core::test_utils::history_conformance::partial(&wire, Mode::Unary, [response.clone()]);
    assert!(!ended);
    assert!(partial.stop().is_failure());
    assert_eq!(partial.choice.len(), 3);
    let [
        AssistantContent::Text(before),
        AssistantContent::Image(image),
        AssistantContent::Text(after),
    ] = partial.choice.as_slice()
    else {
        return Err("the original partial must retain before/image/after in order".into());
    };
    assert_eq!(before.text.as_str(), "before");
    assert_eq!(after.text.as_str(), "after");
    assert_eq!(
        image.media_type,
        Some(rig_core::message::ImageMediaType::PNG)
    );
    assert!(
        image.native.is_none(),
        "a failed turn exposes canonical partial content"
    );
    let rig_core::message::DocumentSourceKind::Base64(encoded) = &image.data else {
        return Err("the received partial image must retain base64 bytes".into());
    };
    assert_eq!(base64::Engine::decode(&BASE64, encoded)?, raw_image);
    assert_eq!(
        received
            .pointer("/candidates/0/content/parts/1/inlineData/mimeType")
            .and_then(serde_json::Value::as_str),
        Some("image/png")
    );
    let received_data = received
        .pointer("/candidates/0/content/parts/1/inlineData/data")
        .and_then(serde_json::Value::as_str)
        .ok_or("received image bytes missing")?;
    assert_eq!(encoded, received_data);

    let unary = rig_core::test_utils::history::decode(&wire, Mode::Unary, [response]);
    assert!(matches!(unary, Err(ProviderError::Truncated)));
    Ok(())
}

#[test]
fn completed_assistant_image_replays_all_three_received_parts_through_prepared_history()
-> Result<(), Box<dyn std::error::Error>> {
    use rig_core::wire::Operation as _;
    let raw_image = vec![0, 1, 2, 255];
    let parts = vec![
        vertexai::model::Part::new().set_text("before"),
        vertexai::model::Part::new().set_inline_data(
            vertexai::model::Blob::new()
                .set_mime_type("image/png")
                .set_data(raw_image.clone()),
        ),
        vertexai::model::Part::new().set_text("after"),
    ];
    // This separate completed control adds STOP; the original incomplete fixture above does not.
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new()
            .set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts(parts.clone()),
            )
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop),
    ]);
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let response = rig_core::test_utils::history::decode(&wire, Mode::Unary, [received])?;
    let [
        AssistantContent::Text(before),
        content @ AssistantContent::Image(image),
        AssistantContent::Text(after),
    ] = response.choice.as_slice()
    else {
        return Err("the completed turn must contain before/image/after".into());
    };
    assert_eq!(before.text, "before");
    assert_eq!(after.text, "after");
    assert_eq!(
        image.media_type,
        Some(rig_core::message::ImageMediaType::PNG)
    );
    assert_eq!(
        content.native_item(),
        Some(&serde_json::to_value(&parts[1])?)
    );
    let rig_core::message::DocumentSourceKind::Base64(data) = &image.data else {
        return Err("the completed image must retain base64 source".into());
    };
    assert_eq!(base64::Engine::decode(&BASE64, data)?, raw_image);
    let request = rig_core::operation::Completion::prepare(
        CompletionRequest::from(vec![
            Message::user("inspect"),
            response.message().ok_or("completed message missing")?,
            Message::user("continue"),
        ]),
        &wire.describe(),
    )?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let replayed = encoded
        .contents
        .get(1)
        .ok_or("replayed model turn missing")?;
    assert_eq!(replayed.role, "model");
    assert_eq!(replayed.parts.len(), 3);
    assert_eq!(replayed.parts, parts);
    assert_eq!(replayed.parts[0].text(), Some(&"before".to_owned()));
    assert_eq!(replayed.parts[2].text(), Some(&"after".to_owned()));
    let blob = replayed.parts[1]
        .inline_data()
        .ok_or("replayed inline image missing")?;
    assert_eq!(blob.mime_type, "image/png");
    assert_eq!(blob.data.as_ref(), raw_image.as_slice());
    Ok(())
}

#[test]
fn real_vertex_decoder_selects_all_five_closed_image_mimes()
-> Result<(), Box<dyn std::error::Error>> {
    use rig_core::message::{DocumentSourceKind, ImageMediaType};
    for (mime, expected) in [
        ("image/jpeg", ImageMediaType::JPEG),
        ("image/png", ImageMediaType::PNG),
        ("image/webp", ImageMediaType::WEBP),
        ("image/heic", ImageMediaType::HEIC),
        ("image/heif", ImageMediaType::HEIF),
    ] {
        let part = vertexai::model::Part::new().set_inline_data(
            vertexai::model::Blob::new()
                .set_mime_type(mime)
                .set_data(vec![0, 1, 2, 255]),
        );
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([
            vertexai::model::Candidate::new()
                .set_content(
                    vertexai::model::Content::new()
                        .set_role("model")
                        .set_parts([part.clone()]),
                )
                .set_finish_reason(vertexai::model::candidate::FinishReason::Stop),
        ]);
        let decoded = rig_core::test_utils::history::decode(
            &GenerateContent::new(GEMINI_2_5_FLASH),
            Mode::Unary,
            [response],
        )?;
        let [content @ AssistantContent::Image(image)] = decoded.choice.as_slice() else {
            return Err("real Vertex decoder must select each admitted image mime".into());
        };
        assert_eq!(image.media_type, Some(expected));
        let DocumentSourceKind::Base64(data) = &image.data else {
            return Err("real Vertex image must retain base64 source".into());
        };
        assert_eq!(base64::Engine::decode(&BASE64, data)?, [0, 1, 2, 255]);
        let received_part = serde_json::to_value(&part)?;
        assert_eq!(content.native_item(), Some(&received_part));
        assert_eq!(
            decoded.raw.pointer("/candidates/0/content/parts/0"),
            Some(&received_part)
        );
        assert_eq!(
            decoded.origin.provider,
            crate::types::completion_response::PROVIDER_NAME
        );
    }
    Ok(())
}

#[test]
fn real_vertex_decoder_keeps_unselected_media_and_sdk_unknown_fields_inspection()
-> Result<(), Box<dyn std::error::Error>> {
    let ordinary = || {
        vertexai::model::Part::new().set_inline_data(
            vertexai::model::Blob::new()
                .set_mime_type("image/png")
                .set_data(vec![0, 1, 2, 255]),
        )
    };
    let mut parts = vec![
        ordinary().set_thought(true),
        ordinary().set_thought_signature(b"sig".to_vec()),
    ];
    for mime in [
        "audio/wav",
        "application/x-private",
        "image/gif",
        "image/svg+xml",
    ] {
        parts.push(
            vertexai::model::Part::new().set_inline_data(
                vertexai::model::Blob::new()
                    .set_mime_type(mime)
                    .set_data(vec![0, 1, 2, 255]),
            ),
        );
    }
    for raw in [
        serde_json::json!({"inlineData":{"mimeType":"image/png","data":"AAEC/w=="},"futurePart":{"exact":[0,1,2,255]}}),
        serde_json::json!({"inlineData":{"mimeType":"image/png","data":"AAEC/w==","displayName":"original.png"}}),
    ] {
        let part: vertexai::model::Part = serde_json::from_value(raw.clone())?;
        assert_eq!(
            serde_json::to_value(&part)?,
            raw,
            "the actual SDK retains unknown fields"
        );
        parts.push(part);
    }
    for part in parts {
        let received_part = serde_json::to_value(&part)?;
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([
            vertexai::model::Candidate::new()
                .set_content(
                    vertexai::model::Content::new()
                        .set_role("model")
                        .set_parts([part]),
                )
                .set_finish_reason(vertexai::model::candidate::FinishReason::Stop),
        ]);
        let decoded = rig_core::test_utils::history::decode(
            &GenerateContent::new(GEMINI_2_5_FLASH),
            Mode::Unary,
            [response],
        )?;
        let [AssistantContent::CarriedText(media)] = decoded.choice.as_slice() else {
            return Err("unselected SDK media must retain its inspection carrier".into());
        };
        assert_eq!(
            media.content,
            rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
        );
        assert_eq!(media.payload.item, received_part);
        assert_eq!(
            decoded.raw.pointer("/candidates/0/content/parts/0"),
            Some(&received_part)
        );
        assert_eq!(
            media.payload.provider.as_deref(),
            Some(crate::types::completion_response::PROVIDER_NAME)
        );
    }
    Ok(())
}
