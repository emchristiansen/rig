use super::*;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::message;
use serde_json::json;
use std::collections::HashMap;

/// A whole Responses body decoded as the OpenAI wire's unary reply.
fn fold_body(response: Value) -> Result<completion::CompletionResponse, ProviderError> {
    let body = serde_json::to_string(&response)?;
    crate::test_utils::decode_reply(
        &openai_wire("gpt-5-mini"),
        &crate::completion::CompletionRequest::new("hello"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(body)],
        serde_json::Value::Null,
    )
}

/// The OpenAI Responses wire, for the request-shaping assertions.
fn openai_wire(model: &str) -> wire::Responses {
    crate::providers::openai::OpenAIConfig::new("dummy-key").responses(model)
}

/// The Responses request a wire builds for a Rig request — the one
/// conversion every caller reaches, whatever opened the socket.
fn wire_request(wire: &wire::Responses, request: completion::CompletionRequest) -> Value {
    json_of(wire, request).expect("request should convert")
}

/// The unary Responses body `wire` builds for `request`, as JSON.
fn json_of(
    wire: &wire::Responses,
    request: completion::CompletionRequest,
) -> Result<Value, EncodeError> {
    let body = wire.responses_request(&request, Delivery::Http(crate::wire::Mode::Unary))?;
    Ok(serde_json::to_value(body)?)
}

/// The request the OpenAI wire for `model` builds for `request`.
fn convert(model: &str, request: completion::CompletionRequest) -> Result<Value, EncodeError> {
    json_of(&openai_wire(model), request)
}

/// The request the OpenAI wire for `model` sends for `request`, prepared as
/// the driver prepares it.
fn prepared(model: &str, request: completion::CompletionRequest) -> Value {
    use crate::wire::{Operation, Wire};
    let wire = openai_wire(model);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_of(&wire, request).expect("request should convert")
}

/// The input items `message` alone becomes.
fn input_items(message: completion::Message) -> Result<Vec<Value>, EncodeError> {
    let wire = openai_wire("gpt-5");
    let mut custom = Custom {
        tools: Default::default(),
        calls: Default::default(),
    };
    let mut system_items = Vec::new();
    super::input(
        &[message],
        &wire,
        "gpt-5",
        &mut custom,
        false,
        &mut system_items,
        None,
    )
}

fn test_document(id: &str, text: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

fn weather_tool_definition() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: crate::message::ToolName::new("get_weather").expect("tool name"),
        description: "Get the weather".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }),
    }
}

fn weather_tool_request() -> completion::CompletionRequest {
    completion::CompletionRequest::new("what's the weather?").tools(vec![weather_tool_definition()])
}

fn request_with_preamble(preamble: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::from(vec![
        message::Message::system(preamble),
        message::Message::user("Hello"),
    ])
}

fn system_only_request(system_text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::new(completion::Message::system(system_text))
}

fn rig_tool_result(content: message::ToolResultContent) -> message::Message {
    rig_tool_result_of(vec![content])
}

fn rig_tool_result_of(content: Vec<message::ToolResultContent>) -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            is_error: false,
            call: crate::message::CallId::from_wire("call-id"),
            name: crate::message::ToolName::new("tool".to_string()).expect("tool name"),
            answers: message::AnsweredToolCall::Function,
            content,
        })],
    }
}

const PDF_URL: &str = "https://example.com/resume.pdf";

fn url_pdf_message() -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::document_url(
            PDF_URL,
            Some(message::DocumentMediaType::PDF),
        )],
    }
}

/// Recursively collect every JSON object with `"type": "input_file"`.
fn find_input_files(value: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("input_file") {
                out.push(value.clone());
            }
            map.values().for_each(|v| find_input_files(v, out));
        }
        serde_json::Value::Array(items) => {
            items.iter().for_each(|v| find_input_files(v, out));
        }
        _ => {}
    }
}

fn sole_input_file(value: &serde_json::Value) -> serde_json::Value {
    let mut found = Vec::new();
    find_input_files(value, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one input_file item in {value:#}"
    );
    found.pop().unwrap()
}

fn assert_url_only_input_file(input_file: &serde_json::Value) {
    assert_eq!(
        input_file.get("file_url").and_then(|v| v.as_str()),
        Some(PDF_URL),
        "URL PDF should carry file_url: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename"),
        None,
        "filename must be absent for URL PDFs (issue #1429): {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_data"),
        None,
        "file_data must be absent for URL PDFs: {input_file:#}"
    );
}

/// The `lookup` tool the histories below call.
fn lookup_tool() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: message::ToolName::new("lookup").expect("tool name"),
        description: "Look something up".to_owned(),
        parameters: json!({"type": "object"}),
    }
}

/// Whether `value` holds a part whose `text` is exactly `text`.
fn carries_text(value: &Value, text: &str) -> bool {
    match value {
        Value::Object(fields) => {
            fields.get("text").and_then(Value::as_str) == Some(text)
                || fields.values().any(|field| carries_text(field, text))
        }
        Value::Array(items) => items.iter().any(|item| carries_text(item, text)),
        _ => false,
    }
}

fn declared_tools_payload() -> Value {
    json!([
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "description": "",
                "strict": false,
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
            }]
        },
        {
            "type": "function",
            "name": "post_to_bus",
            "description": "",
            "strict": false,
            "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
        }
    ])
}

fn response_with_service_tier(service_tier: &str) -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [],
        "service_tier": service_tier,
    })
}

#[test]
fn a_stored_continuation_keeps_the_results_of_the_stored_calls() {
    use crate::wire::{Operation, Wire};
    let wire = openai_wire("gpt-5.4");
    let result = completion::Message::tool_result(
        message::CallId::from_wire("call_1"),
        message::ToolName::new("lookup").expect("tool name"),
        "sunny",
    );
    let mut request =
        crate::completion::CompletionRequest::from(vec![result]).tools(vec![lookup_tool()]);
    request.additional_params = Some(json!({"previous_response_id": "resp_1", "store": true}));
    let prepared = crate::operation::Completion::prepare(request.clone(), &wire.describe())
        .expect("the continuation prepares");
    let body = json_of(&wire, prepared).expect("encodes");
    assert_eq!(body["input"][0]["type"], "function_call_output");
    assert_eq!(body["input"][0]["call_id"], "call_1");
    assert_eq!(body["previous_response_id"], "resp_1");

    // Without a stored response, the same result answers nothing.
    request.additional_params = None;
    let error = crate::operation::Completion::prepare(request, &wire.describe())
        .and_then(|prepared| json_of(&wire, prepared).map_err(Into::into));
    assert!(error.is_err(), "{error:?}");
}

#[test]
fn all_instructions_system_only_input_reports_non_system_requirement() {
    for (case, system) in [
        ("system-only input", "System only"),
        (
            "whitespace-only system input, which produces no `instructions` field",
            "   ",
        ),
    ] {
        let Err(err) = json_of(
            &openai_wire("gpt-4o-mini")
                .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions),
            system_only_request(system),
        ) else {
            panic!("{case}: should fail once every item is lifted");
        };

        assert!(
            err.to_string().contains("non-system item"),
            "{case}: the error should explain that lifted system messages left input empty: {err}"
        );
    }
}

#[test]
fn an_unprepared_part_with_no_responses_form_is_refused() {
    let refused = |content: message::UserContent| {
        json_of(
            &openai_wire("gpt-5"),
            completion::CompletionRequest::new(message::Message::User {
                content: vec![content],
            }),
        )
        .map(|body| body.to_string())
        .map_err(|error| error.to_string())
    };
    let document = refused(message::UserContent::Document(message::Document {
        data: DocumentSourceKind::base64("aGVsbG8=").into(),
        media_type: Some(message::DocumentMediaType::TXT),
        additional_params: None,
    }));
    assert!(
        document
            .as_ref()
            .is_err_and(|error| error.contains("cannot carry this document")),
        "{document:?}"
    );
    let image = refused(message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Unknown,
        ..Default::default()
    }));
    assert!(
        image
            .as_ref()
            .is_err_and(|error| error.contains("cannot carry this image")),
        "{image:?}"
    );
}

#[test]
fn base64_pdf_is_data_and_document_text_is_text() -> anyhow::Result<()> {
    use crate::wire::{Operation, Wire};
    let pdf = "JVBERi0xLjQKJcfsj6IKMSAwIG9iago8PD4+CmVuZG9iagp0cmFpbGVyCjw8Pj4KJSVFT0YK";
    let wire = openai_wire("gpt-5");
    let sent = |data: message::DocumentData| {
        let request = completion::CompletionRequest::new(message::Message::User {
            content: vec![
                message::UserContent::text("Summarize the file."),
                message::UserContent::Document(message::Document {
                    data,
                    media_type: Some(message::DocumentMediaType::PDF),
                    additional_params: None,
                }),
            ],
        });
        crate::operation::Completion::prepare(request, &wire.describe())
            .map_err(anyhow::Error::from)
            .and_then(|request| Ok(json_of(&wire, request)?))
    };
    let body = sent(DocumentSourceKind::base64(pdf).into())?;
    anyhow::ensure!(
        !carries_text(&body, pdf),
        "the PDF's base64 goes to the model as text: {body:#}"
    );
    let legacy = serde_json::from_value(json!({"type": "string", "value": pdf}))?;
    let body = sent(legacy)?;
    anyhow::ensure!(
        carries_text(&body, pdf),
        "a text document goes as its text: {body:#}"
    );
    Ok(())
}

#[test]
fn base64_pdf_via_input_item_path_keeps_filename() {
    let input = message::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::base64("dGVzdA==").into(),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        })],
    };

    let items = input_items(input).expect("base64 PDF should convert to input items");
    let json = Value::Array(items);
    let input_file = sole_input_file(&json);

    assert_eq!(
        input_file.get("file_data").and_then(|v| v.as_str()),
        Some("data:application/pdf;base64,dGVzdA=="),
        "base64 PDF should carry file_data: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename").and_then(|v| v.as_str()),
        Some("document.pdf"),
        "base64 PDF should keep the default filename: {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_url"),
        None,
        "base64 PDF should not carry file_url: {input_file:#}"
    );
}

#[test]
fn completion_response_incomplete_reports_the_truncation_reason() {
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "incomplete",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "half an ans" }]
        }],
        "tools": []
    });

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");

    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::Length)
    );
}

#[test]
fn file_id_document_serializes_as_input_item_content() {
    let message = completion::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId("file_abc".to_string()).into(),
            media_type: None,
            additional_params: None,
        })],
    };

    let converted = input_items(message).expect("conversion should succeed");
    let json = converted[0].clone();

    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"][0]["type"], "input_file");
    assert_eq!(json["content"][0]["file_id"], "file_abc");
    assert!(json["content"][0].get("file_data").is_none());
    assert!(json["content"][0].get("file_url").is_none());
}

#[test]
fn foreign_call_ids_are_normalized_as_pi_normalizes_them() {
    use crate::completion::ReplayTarget;
    let wire = openai_wire("gpt-5.4");
    let long = format!("call_{}", "a".repeat(80));
    for (id, expected) in [
        ("call_1", "call_1".to_owned()),
        ("toolu_01|fc_abc", "toolu_01_fc_abc".to_owned()),
        ("call.with spaces..", "call_with_spaces".to_owned()),
        (long.as_str(), long[..64].to_owned()),
    ] {
        assert_eq!(
            wire.normalize_tool_call_id(id, wire.model(), None),
            expected,
            "{id}"
        );
    }
}

#[test]
fn responses_direct_request_keeps_mid_conversation_system_messages_in_input() {
    let request = crate::completion::CompletionRequest::from(vec![
        completion::Message::system("System prompt"),
        completion::Message::assistant("Earlier assistant turn"),
        completion::Message::system("Mid-conversation instruction"),
        completion::Message::user("Prompt"),
    ])
    .documents(vec![test_document("doc1", "Document text.")]);

    let responses_request = prepared("gpt-4o-mini", request);

    let serialized = responses_request;
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(
        serialized["instructions"],
        json!("System prompt"),
        "only the leading run of system messages should be lifted"
    );
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should follow lifted system instructions: {input:?}"
    );
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[2]["role"], "system");
    assert!(
        input[2]
            .to_string()
            .contains("Mid-conversation instruction"),
        "mid-conversation system messages should keep their position: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert_eq!(
        input
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document input should appear exactly once: {input:?}"
    );
}

#[test]
fn responses_request_conversion_keeps_tools_non_strict_by_default() {
    let req = convert("gpt-4o-mini", weather_tool_request()).expect("request should convert");

    let tool = &req["tools"][0];
    assert_eq!(tool["strict"], json!(false));
    assert_eq!(tool["parameters"]["required"], json!(["location"]));
    assert!(tool["parameters"].get("additionalProperties").is_none());
}

#[test]
fn responses_request_keeps_documents_after_lifted_system_messages() {
    let request = crate::completion::CompletionRequest::new("Prompt")
        .message(completion::Message::system("System prompt"))
        .message(completion::Message::user("Earlier user turn"))
        .message(completion::Message::assistant("Earlier assistant turn"))
        .document(test_document("doc1", "Document text."));

    let responses_request = prepared("gpt-4o-mini", request);

    let serialized = responses_request;
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(serialized["instructions"], json!("System prompt"));
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should be first after system instructions are lifted: {input:?}"
    );
    assert_eq!(input[1]["role"], "user");
    assert!(
        input[1].to_string().contains("Earlier user turn"),
        "prior user history should follow document input: {input:?}"
    );
    assert_eq!(input[2]["role"], "assistant");
    assert!(
        input[2].to_string().contains("Earlier assistant turn"),
        "prior assistant history should follow prior user history: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert!(
        input[3].to_string().contains("Prompt"),
        "prompt should remain last: {input:?}"
    );
}

#[test]
fn responses_request_with_only_system_messages_keeps_them_in_input() {
    let req = convert("gpt-4o-mini", system_only_request("System only"))
        .expect("request conversion should succeed");
    let serialized = req;
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(
        serialized.get("instructions").is_none(),
        "lifting a system-only history would leave input empty, so it stays in input"
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("System only"));
}

#[test]
fn responses_request_with_specific_tool_choice_serializes_named_function() {
    let mut request = weather_tool_request();
    request.tool_choice = Some(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    });

    let request_json = convert("gpt-test", request).expect("convert");

    assert_eq!(
        request_json.get("tool_choice"),
        Some(&json!({"type": "function", "name": "get_weather"}))
    );
}

#[test]
fn responses_strict_function_tools_sanitize_schema() {
    let tool = ResponsesToolDefinition::strict_function(
        "get_weather",
        "Get the weather",
        weather_tool_definition().parameters,
    );

    assert!(tool.strict);
    assert_eq!(tool.parameters["additionalProperties"], json!(false));
    assert_eq!(tool.parameters["required"], json!(["location", "unit"]));
}

#[test]
fn responses_tool_choice_modes_serialize_as_plain_strings() {
    for (choice, expected) in [
        (message::ToolChoice::Auto, json!("auto")),
        (message::ToolChoice::None, json!("none")),
        (message::ToolChoice::Required, json!("required")),
    ] {
        assert_eq!(
            super::tool_choice(choice).expect("mode should convert"),
            expected
        );
    }
}

#[test]
fn responses_tool_choice_specific_empty_names_error() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![],
    });

    assert!(matches!(
        converted.map_err(ProviderError::from),
        Err(ProviderError::Request(error))
            if error.to_string().contains("at least one function name")
    ));
}

#[test]
fn responses_tool_choice_specific_multiple_names_serialize_as_allowed_tools() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![
            crate::message::ToolName::new("add").expect("tool name"),
            crate::message::ToolName::new("subtract").expect("tool name"),
        ],
    })
    .expect("multiple specific tools should convert");

    assert_eq!(
        converted,
        json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": "add"},
                {"type": "function", "name": "subtract"}
            ]
        })
    );
}

#[test]
fn responses_tool_choice_specific_single_name_serializes_as_named_function() {
    let converted = super::tool_choice(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    })
    .expect("single specific tool should convert");

    assert_eq!(
        converted,
        json!({"type": "function", "name": "get_weather"})
    );
}

#[test]
fn responses_wire_can_fallback_to_system_messages_in_input() {
    let wire = openai_wire("gpt-4o-mini").with_system_instructions_as_messages();

    let req = wire_request(&wire, request_with_preamble("You are concise."));
    let serialized = req;
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(serialized.get("instructions").is_none());
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("You are concise."));
    assert_eq!(input[1]["role"], "user");
}

#[test]
fn responses_wire_strict_tools_opt_in_sanitizes_all_function_tools() {
    let wire =
        openai_wire("gpt-4o-mini")
            .with_strict_tools()
            .with_tool(completion::ToolDefinition {
                name: crate::message::ToolName::new("lookup").expect("tool name"),
                description: "Look something up".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }),
            });

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({
        "tools": [{
            "type": "function",
            "name": "extra",
            "description": "An additional_params tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
        }]
    }));

    let req = wire_request(&wire, request);

    let tools = req["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 3);
    for tool in tools {
        assert_eq!(
            tool["strict"],
            json!(true),
            "{} should be strict",
            tool["name"]
        );
        assert_eq!(tool["parameters"]["additionalProperties"], json!(false));
    }
}

#[test]
fn tool_result_file_id_image_uses_the_native_wire_field() {
    let input = rig_tool_result(message::ToolResultContent::Image(message::Image {
        data: message::DocumentSourceKind::FileId("file-image-123".to_string()),
        media_type: None,
        detail: None,
        native: None,
    }));

    let items = input_items(input).expect("input item conversion");
    assert_eq!(
        items[0],
        json!({
            "type": "function_call_output",
            "call_id": "call-id",
            "output": [{
                "type": "input_image",
                "file_id": "file-image-123",
                "detail": "auto"
            }],
            "status": "completed"
        })
    );
}

#[test]
fn url_pdf_via_input_item_path_omits_filename() {
    let items = input_items(url_pdf_message()).expect("URL PDF should convert to input items");
    let json = Value::Array(items);
    assert_url_only_input_file(&sole_input_file(&json));
}

#[test]
fn completion_response_deserializes_standard_service_tier() {
    let response =
        fold_body(response_with_service_tier("standard")).expect("response should deserialize");
    let extras = response
        .extras::<crate::providers::openai::extension::OpenAiExt>()
        .expect("the response carries OpenAI extras")
        .expect("service tier should decode");
    assert_eq!(extras.service_tier.as_deref(), Some("standard"));
}

#[test]
fn completion_response_deserializes_priority_service_tier() {
    let response =
        fold_body(response_with_service_tier("priority")).expect("response should deserialize");
    let extras = response
        .extras::<crate::providers::openai::extension::OpenAiExt>()
        .expect("the response carries OpenAI extras")
        .expect("service tier should decode");
    assert_eq!(extras.service_tier.as_deref(), Some("priority"));
}

#[test]
fn completion_response_preserves_unknown_service_tier() {
    let response = fold_body(response_with_service_tier("provider_experimental"))
        .expect("response should deserialize");
    let extras = response
        .extras::<crate::providers::openai::extension::OpenAiExt>()
        .expect("the response carries OpenAI extras")
        .expect("service tier should decode");
    assert_eq!(
        extras.service_tier.as_deref(),
        Some("provider_experimental")
    );
}

#[test]
fn service_tier_serializes_expected_strings() {
    use crate::completion::ServiceTier;
    for (service_tier, expected) in [
        (ServiceTier::Auto, "auto"),
        (ServiceTier::Default, "default"),
        (ServiceTier::Flex, "flex"),
        (ServiceTier::Priority, "priority"),
    ] {
        assert_eq!(
            serde_json::to_value(service_tier).expect("service tier should serialize"),
            json!(expected)
        );
        let request = completion::CompletionRequest::new("hello").service_tier(service_tier);
        let body = wire_request(&openai_wire("gpt-5-mini"), request);
        assert_eq!(body.get("service_tier"), Some(&json!(expected)));
    }
    for service_tier in ["standard", "provider_experimental"] {
        let request = completion::CompletionRequest::new("hello")
            .additional_params(json!({"service_tier":service_tier}));
        let body = wire_request(&openai_wire("gpt-5-mini"), request);
        assert_eq!(body.get("service_tier"), Some(&json!(service_tier)));
    }
}

#[test]
fn a_usage_sum_retains_a_reported_cached_count() {
    let usage = |cached: Option<u64>, reasoning: Option<u64>| {
        completion::Usage::new()
            .input_tokens(10)
            .output_tokens(20)
            .total_tokens(30)
            .cached_input_tokens(cached)
            .reasoning_tokens(reasoning)
    };
    let reported = || usage(Some(2), Some(4));
    let missing = || usage(None, None);
    let empty = || usage(None, None);
    for (label, sum) in [
        ("missing + reported", missing() + reported()),
        ("reported + missing", reported() + missing()),
        ("empty + reported", empty() + reported()),
        ("reported + empty", reported() + empty()),
    ] {
        assert_eq!(sum.cached_input_tokens, Some(2), "{label}");
        assert_eq!(sum.input_tokens, Some(20), "{label}");
        assert_eq!(sum.output_tokens, Some(40), "{label}");
        assert_eq!(sum.reasoning_tokens, Some(4), "{label}");
        assert_eq!(sum.total_tokens, Some(60), "{label}");
    }
    let both = reported() + reported();
    assert_eq!(both.cached_input_tokens, Some(4));
    assert_eq!((missing() + missing()).cached_input_tokens, None);
}

#[test]
fn summed_cached_token_counts_retain_the_reported_parts() {
    let known = completion::Usage::new().cached_input_tokens(4);
    let absent = completion::Usage::new();
    assert_eq!((known + known).cached_input_tokens, Some(8));
    assert_eq!((known + absent).cached_input_tokens, Some(4));
    assert_eq!((absent + known).cached_input_tokens, Some(4));
}

#[test]
fn responses_function_tools_are_non_strict_by_default() {
    let tool = ResponsesToolDefinition::function(
        "get_weather",
        "Get the weather",
        weather_tool_definition().parameters,
    );

    assert!(!tool.strict);
    assert_eq!(tool.parameters["required"], json!(["location"]));
    assert!(tool.parameters.get("additionalProperties").is_none());

    // Omitted `strict` means "try strict" on the Responses API; `false` must be explicit.
    let serialized = serde_json::to_value(tool).expect("tool should serialize");
    assert_eq!(serialized.get("strict"), Some(&json!(false)));
}

#[test]
fn responses_tool_definitions_accept_nullable_strict() {
    let cases = [
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {}
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": null
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": false
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": true
            }),
            true,
        ),
    ];

    for (value, expected) in cases {
        let tool: ResponsesToolDefinition =
            serde_json::from_value(value).expect("tool definition should deserialize");
        assert_eq!(tool.strict, expected);
    }
}

#[test]
fn responses_request_uses_top_level_instructions_for_preamble_by_default() {
    let req = convert("gpt-4o-mini", request_with_preamble("You are concise."))
        .expect("request should convert");
    let serialized = req;
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert_eq!(serialized["instructions"], json!("You are concise."));
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_request_drops_whitespace_only_preamble() {
    let req =
        convert("gpt-4o-mini", request_with_preamble("  \n ")).expect("request should convert");
    let serialized = req;
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(
        serialized.get("instructions").is_none(),
        "a whitespace-only preamble carries no content and is dropped"
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_request_lifts_system_messages_to_top_level_instructions_by_default() {
    let request = completion::CompletionRequest::new("Hello")
        .preamble("System one".to_string())
        .message(completion::Message::system("System two"));
    let serialized = convert("gpt-4o-mini", request).expect("request should convert");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");
    assert_eq!(
        serialized["instructions"],
        json!("System one\n\nSystem two")
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_wire_can_lift_all_system_messages_via_placement() {
    let wire = openai_wire("gpt-4o-mini")
        .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions);
    let request = completion::CompletionRequest::new("again")
        .preamble("System one".to_string())
        .message(completion::Message::user("hi"))
        .message(completion::Message::system("Mid-conversation instruction"));
    let serialized = wire_request(&wire, request);
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");
    assert_eq!(
        serialized["instructions"],
        json!("System one\n\nMid-conversation instruction")
    );
    assert!(
        input.iter().all(|item| item["role"] != "system"),
        "AllInstructions should leave no system items in input: {input:?}"
    );
}

#[test]
fn all_instructions_whitespace_only_system_input_reports_non_system_requirement() {
    let err = json_of(
        &openai_wire("gpt-4o-mini")
            .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions),
        system_only_request("   "),
    )
    .expect_err("whitespace-only system input should fail once every item is lifted");
    assert!(
        err.to_string().contains("non-system item"),
        "even when lifted system text is whitespace-only (so no instructions field is produced), the error should explain that system messages were lifted: {err}"
    );
}

#[test]
fn responses_wire_default_preserves_all_function_tools_as_constructed() {
    let wire = openai_wire("gpt-4o-mini").with_tool(weather_tool_definition());
    let mut request = weather_tool_request();
    request.additional_params = Some(json!({"tools":[{
        "type":"function", "name":"extra", "description":"An additional_params tool",
        "parameters":{"type":"object","properties":{"x":{"type":"string"}}}
    }]}));
    let req = wire_request(&wire, request);
    let tools = req["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 3);
    for tool in tools {
        let tool: ResponsesToolDefinition =
            serde_json::from_value(tool.clone()).expect("tool definition");
        assert!(!tool.strict, "{} should not be strict", tool.name);
        assert!(tool.parameters.get("additionalProperties").is_none());
    }
}

#[test]
fn responses_explicit_strict_tool_stays_strict_on_a_default_wire() {
    let wire = openai_wire("gpt-4o-mini").with_tool(ResponsesToolDefinition::strict_function(
        "lookup",
        "Look something up",
        json!({"type":"object","properties":{"q":{"type":"string"}}}),
    ));
    let req = wire_request(&wire, weather_tool_request());
    let first: ResponsesToolDefinition =
        serde_json::from_value(req["tools"][0].clone()).expect("tool definition");
    let second: ResponsesToolDefinition =
        serde_json::from_value(req["tools"][1].clone()).expect("tool definition");
    assert!(!first.strict);
    assert!(second.strict);
    assert_eq!(second.parameters["additionalProperties"], json!(false));
}

#[test]
fn a_tools_array_survives_a_serialize_deserialize_round_trip() {
    let mut request = weather_tool_request();
    request.additional_params = Some(json!({ "tools": declared_tools_payload() }));
    let req = convert("gpt-4o-mini", request).expect("request should convert");

    // The whole request does not round-trip (its `input` items are
    // serialize-only), so the property is asserted on the `tools` array: a
    // deserialized tool is kept as the JSON it was, and writes back unchanged.
    let tools = req["tools"].clone();
    let reparsed: Vec<ResponsesRequestTool> =
        serde_json::from_value(tools.clone()).expect("tools should deserialize");
    assert!(
        reparsed
            .iter()
            .all(|tool| matches!(tool, ResponsesRequestTool::Declared(_)))
    );
    assert_eq!(
        serde_json::to_value(&reparsed).expect("tools should serialize"),
        tools
    );
}

#[test]
fn a_malformed_tools_payload_is_still_refused() {
    for payload in [
        json!({"type": "function"}),
        json!(["not a tool object"]),
        json!([{"name": "missing_type"}]),
    ] {
        let mut request = weather_tool_request();
        request.additional_params = Some(json!({ "tools": payload.clone() }));
        let err =
            convert("gpt-4o-mini", request).expect_err("a malformed tools payload must be refused");
        assert!(
            err.to_string()
                .contains("Invalid OpenAI Responses tools payload in additional_params"),
            "{payload}: {err}"
        );
    }
}

#[test]
fn client_metadata_serializes_at_the_top_level_and_is_omitted_when_empty() {
    let params: RawParameterProjection = serde_json::from_value(json!({
        "prompt_cache_key": "thread-1",
        "client_metadata": { "session_id": "session-1", "thread_id": "thread-1" },
    }))
    .expect("additional parameters decode");
    assert_eq!(
        params.client_metadata.get("session_id").map(String::as_str),
        Some("session-1")
    );
    let request = completion::CompletionRequest {
        additional_params: Some(serde_json::to_value(&params).expect("serializes")),
        ..weather_tool_request()
    };
    let body = wire_request(&openai_wire("gpt-5.4"), request);
    assert_eq!(
        body["client_metadata"],
        json!({ "session_id": "session-1", "thread_id": "thread-1" })
    );
    assert_eq!(body["prompt_cache_key"], json!("thread-1"));

    let empty = serde_json::to_value(RawParameterProjection::default()).expect("serializes");
    assert_eq!(empty.get("client_metadata"), None);
}

#[test]
fn the_retired_incomplete_tolerance_key_is_refused_by_name() {
    let request = completion::CompletionRequest {
        additional_params: Some(json!({ "__rig_tolerate_incomplete": true })),
        ..weather_tool_request()
    };
    let error = openai_wire("gpt-5.4")
        .responses_request(&request, Delivery::Http(crate::wire::Mode::Unary))
        .expect_err("the retired key must be refused");
    assert!(
        error.to_string().contains("with_streamed_incomplete"),
        "the refusal names the replacement: {error}"
    );
}

#[test]
fn reasoning_effort_max_survives_request_conversion() {
    let request = request_with_preamble("You are concise.")
        .additional_params(json!({"reasoning":{"effort":"max"}}));
    let serialized = convert("gpt-5.6", request).expect("request should convert");
    assert_eq!(serialized["reasoning"], json!({"effort":"max"}));
}

#[test]
fn reasoning_mode_pro_composes_with_independent_effort() {
    let request = request_with_preamble("You are concise.")
        .additional_params(json!({"reasoning":{"effort":"high","mode":"pro"}}));
    let serialized = convert("gpt-5.6", request).expect("request should convert");
    assert_eq!(
        serialized["reasoning"],
        json!({"effort":"high","mode":"pro"})
    );
}

#[test]
fn url_pdf_in_full_completion_request_omits_filename() {
    let core_request = completion::CompletionRequest::new(url_pdf_message());
    let body = convert("gpt-4o", core_request).expect("request should convert");
    assert_url_only_input_file(&sole_input_file(&body));
}

#[tokio::test]
async fn responses_completion_http_non_success_preserves_status_and_body() {
    use crate::driver::Model;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"bad image","type":"invalid_request_error","code":"invalid_value"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let model = Model::new(openai_wire("gpt-4o-mini"), http_client);
    let request = completion::CompletionRequest::new("hello");

    let error = model
        .call(request)
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
    let json = error
        .provider_response_json()
        .expect("raw body should be valid JSON")
        .expect("parsed JSON should be present");
    assert_eq!(json["error"]["code"], "invalid_value");
}

#[test]
fn mixed_user_content_preserves_order_around_tool_results() {
    let input = message::Message::User {
        content: vec![
            message::UserContent::text("before"),
            message::UserContent::tool_result(
                message::CallId::from_wire("call-id"),
                message::ToolName::new("tool").expect("tool name"),
                vec![message::ToolResultContent::text("tool output")],
            ),
            message::UserContent::text("after"),
        ],
    };
    let items = input_items(input).expect("input item conversion");
    assert!(matches!(items.as_slice(), [before,result,after]
        if before["type"] == "message" && before["role"] == "user"
            && before["content"][0]["type"] == "input_text"
            && before["content"][0]["text"] == "before"
            && result["type"] == "function_call_output" && result["call_id"] == "call-id"
            && after["type"] == "message" && after["role"] == "user"
            && after["content"][0]["type"] == "input_text"
            && after["content"][0]["text"] == "after"));
}

#[test]
fn tool_result_literal_text_and_structured_json_render_without_reparsing() {
    let cases = [
        (
            message::ToolResultContent::text(r#"{"status":"ok"}"#),
            r#"{"status":"ok"}"#.to_string(),
        ),
        (
            message::ToolResultContent::json(json!({"status":"ok"})),
            r#"{"status":"ok"}"#.to_string(),
        ),
    ];
    for (content, expected) in cases {
        let items = input_items(rig_tool_result(content)).expect("input item conversion");
        assert!(matches!(items.as_slice(), [item]
            if item["type"] == "function_call_output" && item["output"] == expected));
    }
}

#[test]
fn multiple_text_tool_result_blocks_preserve_order_as_rich_function_output() {
    let content = vec![
        message::ToolResultContent::text("first"),
        message::ToolResultContent::text("second"),
    ];
    let expected =
        json!([{"type":"input_text","text":"first"},{"type":"input_text","text":"second"}]);
    let items = input_items(rig_tool_result_of(content)).expect("input item conversion");
    match items.as_slice() {
        [item] if item["type"] == "function_call_output" => assert_eq!(item["output"], expected),
        other => panic!("expected one function-call output, got {other:?}"),
    }
    assert_eq!(
        items[0],
        json!({"type":"function_call_output","call_id":"call-id","output":[
        {"type":"input_text","text":"first"},{"type":"input_text","text":"second"}
    ],"status":"completed"})
    );
}

#[test]
fn multiple_text_and_json_tool_result_blocks_preserve_boundaries() {
    let content = vec![
        message::ToolResultContent::text("before"),
        message::ToolResultContent::json(json!({"status":"ok"})),
        message::ToolResultContent::text("after"),
    ];
    let output = result_output(&content, false).expect("tool-result conversion should succeed");
    assert_eq!(
        output,
        json!([
            {"type":"input_text","text":"before"},
            {"type":"input_text","text":r#"{"status":"ok"}"#},
            {"type":"input_text","text":"after"}
        ])
    );
}

#[test]
fn tool_result_images_and_text_preserve_order_as_rich_function_output() {
    let content = vec![
        message::ToolResultContent::text("before"),
        message::ToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(message::ImageMediaType::PNG),
            None,
        ),
        message::ToolResultContent::json(json!({"after":true})),
    ];
    let items = input_items(rig_tool_result_of(content)).expect("input item conversion");
    match items.as_slice() {
        [item] if item["type"] == "function_call_output" => {
            let content = item["output"].as_array().expect("rich function output");
            assert!(matches!(content.as_slice(), [before,image,after]
                if before["type"] == "input_text" && before["text"] == "before"
                    && image["type"] == "input_image"
                    && image["image_url"] == "data:image/png;base64,aW1hZ2U="
                    && after["type"] == "input_text" && after["text"] == r#"{"after":true}"#));
        }
        other => panic!("expected one rich function output, got {other:?}"),
    }
}

#[test]
fn completion_response_with_unknown_output_keeps_usage() {
    // Guards the original reason the catch-all exists: an unknown item must
    // not break decoding of the whole response or drop token usage.
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_001",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 25 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 15 },
            "total_tokens": 150,
        },
    });

    let expected_item = response["output"][0].clone();
    let completion = fold_body(response).expect("an unknown item decodes");
    let Some(completion::AssistantContent::ProviderItem(item)) = completion.choice.first() else {
        panic!("the unknown output keeps its actual ProviderItem representation");
    };
    assert_eq!(item.payload.item, expected_item);
    assert_eq!(item.payload.provider.as_deref(), Some("openai"));
    assert_eq!(completion.usage.input_tokens, Some(100));
    assert_eq!(completion.usage.output_tokens, Some(50));
    assert_eq!(completion.usage.total_tokens, Some(150));
    assert_eq!(completion.usage.cached_input_tokens, Some(25));
    assert_eq!(completion.usage.reasoning_tokens, Some(15));
    assert_eq!(completion.raw["output"][1]["id"], "msg_1");
    assert_eq!(completion.text(), "hi");
}

mod raw_capture {
    use super::*;

    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_responses_0001";

    /// A Responses body carrying `service_tier`, which the normalized
    /// response provably lacks.
    const BODY: &str = r#"{
            "id": "resp_raw_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-mini-2024-07-18",
            "service_tier": "default",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_raw_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }],
            "tools": []
        }"#;

    /// The load-bearing capture property: `raw` is the provider's body
    /// exactly as it arrived, and folding that capture again (with the
    /// header id reattached, since the capture is body only) reproduces
    /// every normalized field. The transport id is not part of the capture,
    /// while the normalized response beside it carries the header.
    #[tokio::test]
    async fn completion_captures_the_provider_body_as_raw() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
        let model = crate::driver::Model::new(
            openai_wire("gpt-4o-mini"),
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers),
        );

        let response = model
            .call(crate::completion::CompletionRequest::new("hello"))
            .await
            .expect("completion");

        let raw = &response.raw;
        let body: Value = serde_json::from_str(BODY).expect("the body is JSON");
        assert_eq!(*raw, body, "the capture is the provider's body");
        assert_eq!(raw["service_tier"], "default");
        assert!(raw.get("provider_request_id").is_none());

        let mut refolded = fold_body(raw.clone()).expect("re-fold the capture");
        refolded.provider_request_id = Some(REQUEST_ID.to_string());
        assert_eq!(response.identity(), refolded.identity());
        assert_eq!(response.finish_reason(), refolded.finish_reason());
        assert_eq!(response.model(), refolded.model());
        // The cost prices the counters for each fold's request model, and
        // the re-fold requested another one.
        assert_eq!(response.usage.cost(None), refolded.usage.cost(None));
        assert!(
            response.usage.cost.is_some(),
            "the catalog prices gpt-4o-mini"
        );
        assert_eq!(response.choice, refolded.choice);
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    }
}

use crate::message::{
    AssistantContent as CoreContent, AssistantMessage, CallId, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart as Part, CustomToolCall, ToolCall, ToolFunction, ToolName,
};
use crate::providers::openai::extension::OpenAiExt;

fn named(text: &str) -> ToolName {
    ToolName::new(text).expect("fixture name")
}
fn assistant(parts: Vec<CoreContent>) -> message::Message {
    message::Message::Assistant(AssistantMessage::new(parts))
}
fn item_body(output: Vec<Value>) -> Value {
    json!({"id":"resp_1","object":"response","created_at":0,"status":"completed",
        "error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,
        "model":"gpt-5-mini","usage":null,"output":output,"tools":[]})
}
fn fold_items(output: Vec<Value>) -> completion::CompletionResponse {
    fold_body(item_body(output)).expect("valid received items decode")
}
fn persist(response: &completion::CompletionResponse) {
    let encoded = serde_json::to_value(response).expect("actual response serializes");
    let loaded: completion::CompletionResponse =
        serde_json::from_value(encoded).expect("actual response loads");
    assert_eq!(&loaded, response);
}
fn reasoning_of(response: &completion::CompletionResponse) -> &CarriedReasoning {
    let [CoreContent::CarriedReasoning(reasoning)] = response.choice.as_slice() else {
        panic!("one carried reasoning: {:?}", response.choice)
    };
    reasoning
}
fn received_unknown(item: Value) {
    let response = fold_items(vec![item.clone()]);
    let [CoreContent::ProviderItem(raw)] = response.choice.as_slice() else {
        panic!("one faithful provider item: {:?}", response.choice)
    };
    assert_eq!(raw.payload.item, item);
    assert_eq!(raw.payload.provider.as_deref(), Some("openai"));
    assert_eq!(response.raw["output"][0], item);
    persist(&response);
    let replayed = input_items(response.message().expect("actual received turn"))
        .expect("actual received replay");
    assert_eq!(replayed, vec![item]);
}
fn unissued(id: &str, parts: Vec<Part>) -> CoreContent {
    CoreContent::CarriedReasoning(
        CarriedReasoning::new(parts, CarriedPayload::new(Value::Null, None))
            .with_id(Some(id.to_owned())),
    )
}
fn reasoning_inputs(items: &[Value]) -> Vec<&Value> {
    items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .collect()
}
fn malformed_body(status: &str) -> Value {
    item_body(vec![
        json!({"type":"function_call","id":"fc_1","arguments":"{\"x\":481","call_id":"call_1","name":"add","namespace":null,"status":status}),
    ])
}
fn response_with_object_top_p() -> Value {
    json!({"id":"resp_123","object":"response","created_at":0,"status":"completed","model":"MiniMax-M2",
    "top_p":{"value":0.95},"service_tier":"default","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"location\":\"Paris\"}","status":"completed"}],
    "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}})
}
fn custom_and_namespaced_history() -> Vec<message::Message> {
    let function = ToolCall::new(
        CallId::from_wire("call_fn"),
        ToolFunction::new(named("search"), json!({"q":"rig"}))
            .with_namespace(Some("web".to_owned())),
    );
    let custom = CustomToolCall::new(
        CallId::from_wire("call_custom"),
        named("apply_patch"),
        "*** Begin Patch\n{\"not\": \"arguments\"}",
    )
    .with_namespace(Some("editor".to_owned()));
    let function_result = message::UserContent::ToolResult(
        function.result(vec![message::ToolResultContent::text("found")]),
    );
    let custom_result = message::UserContent::ToolResult(custom.result(vec![
        message::ToolResultContent::text("applied "),
        message::ToolResultContent::json(json!({"ok":true})),
    ]));
    vec![
        message::Message::user("go"),
        assistant(vec![
            CoreContent::ToolCall(function),
            CoreContent::CustomToolCall(custom),
        ]),
        message::Message::User {
            content: vec![function_result, custom_result],
        },
    ]
}

fn current_pair_request() -> completion::CompletionRequest {
    let local = CallId::from_wire("");
    let provider = CallId::from_wire("tool-0");
    let call = |id: CallId| {
        assistant(vec![CoreContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(named("test"), json!({})),
        ))])
    };
    let result = |id: CallId| message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            call: id,
            name: named("possibly_repaired"),
            answers: message::AnsweredToolCall::Function,
            content: vec![],
            is_error: false,
        })],
    };
    completion::CompletionRequest {
        chat_history: vec![
            message::Message::system("system"),
            call(local.clone()),
            call(provider.clone()),
            result(provider),
            message::Message::assistant("intervening text"),
            result(local.clone()),
            call(local.clone()),
            result(local),
        ],
        max_tokens: Some(128),
        ..completion::CompletionRequest::new("unused")
    }
}
fn assert_current_pairs(wire: Value) {
    fn visit(value: &serde_json::Value, calls: &mut Vec<String>, results: &mut Vec<String>) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    visit(value, calls, results);
                }
            }
            serde_json::Value::Object(object) => {
                let kind = object.get("type").and_then(serde_json::Value::as_str);
                let reference = match kind {
                    Some("function_call") => object
                        .get("call_id")
                        .or_else(|| object.get("id"))
                        .map(|id| (true, id)),
                    Some("tool_use") => object.get("id").map(|id| (true, id)),
                    Some("function_call_output" | "function_result") => {
                        object.get("call_id").map(|id| (false, id))
                    }
                    Some("tool_result") => object.get("tool_use_id").map(|id| (false, id)),
                    _ => None,
                };
                if let Some((is_call, id)) = reference {
                    let id = id.as_str().expect("wire identity is a string").to_owned();
                    if is_call {
                        calls.push(id);
                    } else {
                        results.push(id);
                    }
                }
                if object.get("role").and_then(serde_json::Value::as_str) == Some("tool") {
                    results.push(object["tool_call_id"].as_str().unwrap().to_owned());
                }
                if let Some(array) = object
                    .get("tool_calls")
                    .and_then(serde_json::Value::as_array)
                {
                    calls.extend(
                        array
                            .iter()
                            .map(|call| call["id"].as_str().unwrap().to_owned()),
                    );
                }
                for value in object.values() {
                    visit(value, calls, results);
                }
            }
            _ => {}
        }
    }
    let mut calls = Vec::new();
    let mut results = Vec::new();
    visit(&wire, &mut calls, &mut results);
    assert_eq!(calls.len(), 3, "{wire}");
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()],
        "{wire}"
    );
    assert_eq!(calls[1], "tool-0");
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "{wire}"
    );
}

// D01; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_custom_result_carrying_an_image_is_refused() {
    let custom = CustomToolCall::new(CallId::from_wire("call_custom"), named("draw"), "circle");
    let result = message::UserContent::ToolResult(custom.result(vec![
        message::ToolResultContent::image_base64(
            "aGVsbG8=",
            Some(message::ImageMediaType::PNG),
            None,
        ),
    ]));
    let request = completion::CompletionRequest {
        chat_history: vec![
            message::Message::user("go"),
            assistant(vec![CoreContent::CustomToolCall(custom)]),
            message::Message::User {
                content: vec![result],
            },
        ],
        ..weather_tool_request()
    };
    let error = json_of(&openai_wire("gpt-5.4"), request)
        .expect_err("an image in a custom result must be refused");
    assert!(
        error.to_string().contains("custom_tool_call_output"),
        "the refusal names the item that cannot carry it: {error}"
    );
}

// D02; approved G82-G89 current boundary and named exceptions.
#[test]
fn an_unqualified_function_call_carries_no_namespace_member() {
    let item = json!({"type":"function_call","id":"fc_1","arguments":"{}","call_id":"call_1","name":"f","status":"completed"});
    let response = fold_items(vec![item.clone()]);
    let [CoreContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one function call")
    };
    assert_eq!(call.function.namespace, None);
    assert_eq!(response.choice[0].native_item(), Some(&item));
    let replay =
        input_items(response.message().expect("actual received turn")).expect("actual replay");
    assert_eq!(replay, vec![item]);
    assert!(replay[0].get("namespace").is_none());
}

// D03; approved G82-G89 current boundary and named exceptions.
#[test]
fn custom_calls_namespaces_and_answers_encode_to_pinned_bytes() {
    let request = completion::CompletionRequest {
        chat_history: custom_and_namespaced_history(),
        ..weather_tool_request()
    };
    let request = wire_request(&openai_wire("gpt-5.4"), request);
    let input = request["input"].clone();
    let items = input.as_array().expect("input is an array");

    assert_eq!(
        items[1],
        json!({
            "type": "function_call",
            "call_id": "call_fn",
            "name": "search",
            "namespace": "web",
            "arguments": "{\"q\":\"rig\"}",
        })
    );
    assert_eq!(
        items[2],
        json!({
            "type": "custom_tool_call",
            "call_id": "call_custom",
            "name": "apply_patch",
            "namespace": "editor",
            "input": "*** Begin Patch\n{\"not\": \"arguments\"}",
        })
    );
    assert_eq!(
        items[3],
        json!({
            "type": "function_call_output",
            "call_id": "call_fn",
            "output": "found",
            "status": "completed",
        })
    );
    assert_eq!(
        items[4],
        json!({
            "type": "custom_tool_call_output",
            "call_id": "call_custom",
            "name": "apply_patch",
            "output": "applied {\"ok\":true}",
        })
    );
    assert_eq!(items.len(), 5, "nothing else is emitted: {input}");
}

// D04; approved G82-G89 current boundary and named exceptions.
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let request = current_pair_request();
    let wire = convert("test", request).expect("full request preserves original pair matrix");
    assert_current_pairs(wire);
}

// D05; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_custom_tool_call_output_item_decodes_and_round_trips() {
    let wire_item = json!({"type":"custom_tool_call","id":"ctc_9","call_id":"call_9","name":"grammar_tool","input":"SELECT 1","status":"completed"});
    let response = fold_items(vec![wire_item.clone()]);
    let [CoreContent::CustomToolCall(call)] = response.choice.as_slice() else {
        panic!("expected one custom call")
    };
    assert_eq!(call.input, "SELECT 1");
    assert_eq!(call.namespace, None);
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_9"));
    assert_eq!(call.name.as_str(), "grammar_tool");
    assert_eq!(response.choice[0].native_item(), Some(&wire_item));
    persist(&response);
    assert_eq!(
        input_items(response.message().expect("received turn")).expect("replay"),
        vec![wire_item]
    );
}

// D06; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_namespaced_function_call_output_item_keeps_its_namespace_unary() {
    let item = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"search","namespace":"web","arguments":"{\"q\":\"rig\"}","status":"completed"});
    let response = fold_items(vec![item.clone()]);
    let [CoreContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("expected one function call")
    };
    assert_eq!(call.function.namespace.as_deref(), Some("web"));
    assert_eq!(response.choice[0].native_item(), Some(&item));
}

// D07; approved G82-G89 current boundary and named exceptions.
#[test]
fn an_unknown_reasoning_summary_supplies_no_text() {
    let known = json!({"type":"summary_text","text":"known"});
    let unknown = json!({"type":"future_summary","list":[1]});
    let read = |part| {
        carried_reasoning(
            &json!({"type":"reasoning","id":"r1","summary":[part]}),
            Some("openai".to_owned()),
        )
        .expect("summary reader")
    };
    let known = read(known);
    let unknown = read(unknown.clone());
    assert_eq!(known.content[0].visible_text(), Some("known"));
    assert_eq!(unknown.content[0].visible_text(), None);
    let Part::OpaqueSummary(raw) = &unknown.content[0] else {
        panic!("opaque summary")
    };
    assert_eq!(raw.item, json!({"type":"future_summary","list":[1]}));
}

// D08; approved G82-G89 current boundary and named exceptions.
#[test]
fn each_known_tag_decodes_to_its_modeled_variant() {
    for (item, kind) in [
        (
            json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]}),
            0,
        ),
        (
            json!({"type":"function_call","id":"call_1","arguments":"{}","call_id":"c1","name":"f","status":"completed"}),
            1,
        ),
        (json!({"type":"reasoning","id":"r1","summary":[]}), 2),
    ] {
        let response = fold_items(vec![item.clone()]);
        assert_eq!(response.choice.len(), 1);
        assert!(matches!(
            (&response.choice[0], kind),
            (CoreContent::CarriedText(_), 0)
                | (CoreContent::ToolCall(_), 1)
                | (CoreContent::CarriedReasoning(_), 2)
        ));
        assert_eq!(response.raw["output"][0], item);
    }
}

// D09; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_known_variant_round_trips_value_equal() {
    let item = json!({"type":"function_call","id":"fc_1","arguments":"{}","call_id":"c1","name":"search","status":"completed"});
    let response = fold_items(vec![item.clone()]);
    assert!(matches!(
        response.choice.as_slice(),
        [CoreContent::ToolCall(_)]
    ));
    assert_eq!(response.choice[0].native_item(), Some(&item));
    persist(&response);
    assert_eq!(
        input_items(response.message().expect("received turn")).expect("replay"),
        vec![item]
    );
}

// D10; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_known_variant_with_bad_body_errors() {
    let malformed = json!({"type":"function_call","id":"call_1"});
    // The actual public classifier invokes the existing strict item reader in this received-root context.
    let result = fold_body(item_body(vec![malformed]));
    assert!(result.is_err());
}

// D11; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_message_round_trips_value_equal() {
    let item = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]});
    let response = fold_items(vec![item.clone()]);
    assert!(matches!(
        response.choice.as_slice(),
        [CoreContent::CarriedText(_)]
    ));
    assert_eq!(response.text(), "hello");
    assert_eq!(
        response.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_1")
    );
    assert_eq!(response.raw["output"][0], item);
    persist(&response);
    let replay = input_items(response.message().expect("received turn")).expect("replay");
    assert_eq!(replay, vec![item]);
}

// D12; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_reasoning_conversion_omits_empty_encrypted_content() {
    let item = json!({"type":"reasoning","id":"reasoning_1","summary":[],"content":["visible reasoning"],"encrypted_content":"","signature":null,"status":"completed"});
    let response = fold_items(vec![item.clone()]);
    let reasoning = reasoning_of(&response);
    assert_eq!(reasoning.id.as_deref(), Some("reasoning_1"));
    assert_eq!(reasoning.content.len(), 1);
    assert!(
        matches!(reasoning.content.first(),Some(Part::Text{text,..}) if text=="visible reasoning")
    );
    assert_eq!(response.raw["output"][0], item);
}

// D13; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_reasoning_conversion_preserves_non_empty_encrypted_content() {
    let item = json!({"type":"reasoning","id":"reasoning_1","summary":[],"content":[],"encrypted_content":"ciphertext","signature":null,"status":"completed"});
    let response = fold_items(vec![item.clone()]);
    assert_eq!(
        reasoning_of(&response).content,
        vec![Part::Encrypted("ciphertext".to_owned())]
    );
    assert_eq!(response.raw["output"][0], item);
}

// D14; approved G82-G89 current boundary and named exceptions.
#[test]
fn received_reasoning_null_optionals_are_accepted_and_persisted() {
    let item = json!({"type":"reasoning","id":"reasoning_1","summary":[],"encrypted_content":null,"status":null});
    let response = fold_items(vec![item]);
    let value = &response.raw["output"][0];
    assert_eq!(value["type"], "reasoning");
    assert_eq!(value["encrypted_content"], Value::Null);
    assert_eq!(value["status"], Value::Null);
    assert!(value.get("encrypted_content").is_some());
    assert!(value.get("status").is_some());
    persist(&response);
}

// D15; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_reasoning_round_trips_value_equal() {
    let item = json!({"type":"reasoning","id":"reasoning_1","summary":[{"type":"summary_text","text":"weighing options"}],"content":["private reasoning"],"encrypted_content":"ENCRYPTED","signature":null,"status":"completed"});
    let response = fold_items(vec![item.clone()]);
    let reasoning = reasoning_of(&response);
    assert_eq!(reasoning.id.as_deref(), Some("reasoning_1"));
    assert_eq!(
        reasoning.content,
        vec![
            Part::Summary("weighing options".to_owned()),
            Part::Text {
                text: "private reasoning".to_owned(),
                signature: None
            },
            Part::Encrypted("ENCRYPTED".to_owned())
        ]
    );
    assert_eq!(response.raw["output"][0], item);
    persist(&response);
}

// D16; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_unknown_preserves_hosted_tool_payload() {
    received_unknown(
        json!({"type":"web_search_call","id":"ws_001","status":"completed","action":{"type":"search","queries":["rig framework"]}}),
    );
}

// D17; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_unknown_round_trips_value_equal() {
    received_unknown(
        json!({"type":"file_search_call","id":"fs_007","status":"in_progress","queries":["lifecycle"]}),
    );
}

// D18; approved G82-G89 current boundary and named exceptions.
#[test]
fn output_without_usable_type_tag_decodes_to_unknown() {
    for item in [
        json!({"id":"x","note":"no type field"}),
        json!({"type":7,"id":"x"}),
    ] {
        received_unknown(item);
    }
}

// D19; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_signature_only_reasoning_item_round_trips_without_text() {
    for summary in [
        vec![],
        vec![json!({"type":"summary_text","text":"a summary"})],
    ] {
        let raw = json!({"type":"reasoning","id":"rs_1","summary":summary,"content":[],"encrypted_content":null,"signature":"signed"});
        let reasoning =
            carried_reasoning(&raw, None).expect("existing carried reader keeps signature");
        assert!(reasoning.content.iter().any(|part|matches!(part,Part::Text{text,signature:Some(signature)} if text.is_empty()&&signature=="signed")));
        let item = carried_reasoning_item(&reasoning)
            .expect("identified reasoning encodes")
            .expect("reasoning item");
        assert_eq!(item["signature"], "signed");
        assert_eq!(item["summary"], Value::Array(summary));
        assert!(item.get("content").is_none(), "{item}");
    }
}

// D20; approved G82-G89 current boundary and named exceptions.
#[test]
fn assistant_reasoning_text_tool_call_convert_in_responses_replay_order() {
    let items = input_items(assistant(vec![
        unissued(
            "rs_123",
            vec![Part::Summary("structured summary".to_owned())],
        ),
        CoreContent::text("final answer"),
        CoreContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_123"),
            ToolFunction::new(named("lookup"), json!({"query":"rig"})),
        )),
    ]))
    .expect("assistant history converts");
    assert_eq!(items.len(), 3);
    assert!(items[0].get("role").is_none());
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(items[0]["id"], "rs_123");
    assert_eq!(items[1]["role"], "assistant");
    assert_eq!(items[1]["content"][0]["type"], "output_text");
    assert_eq!(items[1]["content"][0]["text"], "final answer");
    assert!(items[2].get("role").is_none());
    assert_eq!(items[2]["type"], "function_call");
    assert_eq!(items[2]["call_id"], "call_123");
    assert_eq!(items[2]["name"], "lookup");
    assert_eq!(items[2]["arguments"], "{\"query\":\"rig\"}");
}

// D21; approved G82-G89 current boundary and named exceptions.
#[test]
fn assistant_text_without_idless_reasoning_replays_as_output_text() {
    let items = input_items(assistant(vec![CoreContent::text("final answer")]))
        .expect("assistant history converts");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["type"], "message");
    assert_eq!(items[0]["content"][0]["type"], "output_text");
    assert_eq!(items[0]["content"][0]["text"], "final answer");
}

// D22; approved G82-G89 current boundary and named exceptions.
#[test]
fn canonical_assistant_text_parts_replay_as_separate_current_messages() {
    let items = input_items(assistant(vec![
        CoreContent::text("first part"),
        CoreContent::text("second part"),
    ]))
    .expect("assistant history converts");
    // G83: canonical parts are separate current rich messages.
    assert_eq!(items.len(), 2);
    let texts: Vec<&str> = items
        .iter()
        .flat_map(|item| item["content"].as_array().expect("rich content"))
        .map(|part| {
            assert_eq!(part["type"], "output_text");
            part["text"].as_str().expect("text")
        })
        .collect();
    assert_eq!(texts, ["first part", "second part"]);
}

// D23; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_history_idless_reasoning_plus_text_preserves_text_input_item() {
    let items = input_items(assistant(vec![
        CoreContent::reasoning("provider reasoning"),
        CoreContent::text("final answer"),
    ]))
    .expect("assistant history converts");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["role"], "assistant");
    assert_eq!(items[0]["content"][0]["type"], "output_text");
    assert_eq!(items[0]["content"][0]["text"], "final answer");
}

// D24; approved G82-G89 current boundary and named exceptions.
#[tokio::test]
async fn cross_provider_minted_reasoning_ids_are_not_serialized_upstream() {
    use futures::StreamExt;
    let model = crate::test_utils::MockCompletionModel::from_stream_turns([vec![
        crate::test_utils::MockStreamEvent::reasoning_delta("thinking hard"),
        crate::test_utils::MockStreamEvent::text("answer"),
        crate::test_utils::MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut stream = model
        .stream(completion::CompletionRequest::new("hi"))
        .expect("mock stream");
    while stream.next().await.is_some() {}
    let choice = stream.partial().choice;
    assert!(choice.iter().any(|content|matches!(content,CoreContent::Reasoning(reasoning) if reasoning.text=="thinking hard")));
    assert!(
        choice
            .iter()
            .any(|content| matches!(content,CoreContent::Text(text) if text.text=="answer"))
    );
    let items = input_items(assistant(choice)).expect("history converts");
    assert!(reasoning_inputs(&items).is_empty());
    // G88: this positive fixture is explicitly unissued carried construction, not a fabricated received item.
    let items = input_items(assistant(vec![unissued(
        "rs_0123",
        vec![Part::Text {
            text: "real item".to_owned(),
            signature: None,
        }],
    )]))
    .expect("identified unissued history converts");
    let reasoning = reasoning_inputs(&items);
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["id"], "rs_0123");
}

// D25; approved G82-G89 current boundary and named exceptions.
#[tokio::test]
async fn delta_only_stream_minted_output_ids_are_not_serialized_upstream() {
    use bytes::Bytes;
    use futures::StreamExt;
    let sse = |frame: &serde_json::Value| Bytes::from(format!("data: {frame}\n\n"));
    let frames = vec![
        // G141: no received `item_id`; do not give an internal slot replay authority.
        sse(&json!({
            "type": "response.reasoning_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "unattributed thought",
        })),
        sse(&json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.4",
                "output": [],
                "tools": [],
                "usage": null,
            },
        })),
    ];
    let body = Bytes::from(
        frames
            .into_iter()
            .flat_map(|frame| frame.to_vec())
            .collect::<Vec<_>>(),
    );
    let model = crate::driver::Model::new(
        openai_wire("gpt-5.4"),
        crate::test_utils::MockStreamingClient { sse_bytes: body },
    );
    let mut stream = model
        .stream(completion::CompletionRequest::new("hello"))
        .expect("stream opens");
    while let Some(item) = stream.next().await {
        item.expect("valid received event");
    }
    let drained = stream.finish().await.expect("stream should complete");
    // G141: only the old plain-Reasoning/minted-id view is lost. The actual
    // carried text and absent provider item ID still cannot go upstream.
    let thought = drained.choice.iter().find_map(|content| match content {
        CoreContent::CarriedReasoning(reasoning) => Some(reasoning),
        _ => None,
    }).expect("the actual carried thought");
    assert_eq!(thought.display_text(), "unattributed thought");
    assert_eq!(thought.content, vec![Part::Text {
        text: "unattributed thought".to_owned(), signature: None,
    }]);
    assert!(thought.id.is_none(), "no provider reasoning item ID was received");
    let items = input_items(assistant(drained.choice)).expect("history converts");
    assert!(reasoning_inputs(&items).is_empty());
}

// D26; approved G82-G89 current boundary and named exceptions.
#[test]
fn idless_assistant_text_replays_as_a_current_rich_message() {
    let items = input_items(assistant(vec![CoreContent::text("final answer")]))
        .expect("assistant history converts");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["type"], "message");
    assert_eq!(items[0]["role"], "assistant");
    assert_eq!(items[0]["content"][0]["type"], "output_text");
    assert_eq!(items[0]["content"][0]["text"], "final answer");
    assert_eq!(items[0]["status"], "completed");
    assert!(
        items[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_rig_"))
    );
}

// D27; approved G82-G89 current boundary and named exceptions.
#[test]
fn idless_reasoning_only_is_skipped_without_empty_input_item() {
    let items = input_items(assistant(vec![CoreContent::reasoning(
        "provider reasoning",
    )]))
    .expect("idless reasoning degrades gracefully at private input boundary");
    assert!(items.is_empty());
}

// D28; approved G82-G89 current boundary and named exceptions.
#[test]
fn mocked_second_turn_request_omits_unreplayable_reasoning() {
    let request = completion::CompletionRequest {
        model: None,
        chat_history: vec![
            message::Message::system("You are concise."),
            message::Message::user("Think briefly, then answer."),
            assistant(vec![
                CoreContent::reasoning("provider reasoning"),
                CoreContent::text("final answer"),
            ]),
            assistant(vec![
                CoreContent::reasoning("provider reasoning only"),
                CoreContent::text(""),
            ]),
            message::Message::user("/no_think Reply with exactly: OK"),
        ],
        max_tokens: Some(64),
        ..completion::CompletionRequest::new("unused")
    };
    let value = convert("Qwen/Qwen3-4B", request).expect("request converts");
    let input = value["input"].as_array().expect("input array");
    assert!(
        !input
            .iter()
            .any(|item| item.get("type") == Some(&json!("reasoning")) && item.get("id").is_none())
    );
    assert!(!input.iter().any(|item| {
        item.get("role") == Some(&json!("assistant"))
            && item
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
    }));
    let assistant_items: Vec<_> = input
        .iter()
        .filter(|item| item.get("role") == Some(&json!("assistant")))
        .collect();
    assert_eq!(assistant_items.len(), 1);
    assert_eq!(assistant_items[0]["content"][0]["type"], "output_text");
    assert_eq!(assistant_items[0]["content"][0]["text"], "final answer");
}

// D29; approved G82-G89 current boundary and named exceptions.
#[test]
fn carried_responses_parts_keep_annotations_and_refuse_opaque_text_edits() {
    // G90.1: actual private carried-part input, issuer None; exact URL-only annotation.
    let annotations = json!([{"type":"url_citation","url":"https://example.com"}]);
    let cited = json!({"type":"output_text","text":"cited","annotations":annotations});
    let message = json!({"type":"message","content":[cited.clone()]});
    let parts = carried_message_parts(&message, 0, None).expect("current private factory");
    assert_eq!(parts[0].text(), "cited");
    assert_eq!(parts[0].payload.provider, None);
    let message::CarriedTextReplay::ProviderPart(replayed) =
        parts[0].replay_item().expect("known raw part replay")
    else {
        panic!("raw part category")
    };
    assert_eq!(replayed, cited);
    assert_eq!(replayed["annotations"], annotations);
    let items = input_items(assistant(
        parts.into_iter().map(CoreContent::CarriedText).collect(),
    ))
    .expect("cited input converts");
    assert_eq!(items[0]["content"][0], cited);

    // G90.2: current raw authority is exact, not the retired reserved-key setter/filter.
    let hostile = json!({"text":"evil","type":"evil_type","annotations":["kept"]});
    let raw_message = json!({"type":"message","content":[hostile.clone()]});
    let mut parts = carried_message_parts(&raw_message, 0, None).expect("opaque raw category");
    let untouched = input_items(assistant(
        parts
            .iter()
            .cloned()
            .map(CoreContent::CarriedText)
            .collect(),
    ))
    .expect("untouched raw input");
    assert_eq!(
        untouched[0]["content"][0], hostile,
        "whole raw part is preserved without filtering"
    );
    *parts[0].text_mut() = "real".to_owned();
    let error = ProviderError::from(
        input_items(assistant(
            parts.into_iter().map(CoreContent::CarriedText).collect(),
        ))
        .expect_err("nonempty opaque edit is refused"),
    );
    let ProviderError::Request(reason) = &error else {
        panic!("typed Request refusal: {error:?}")
    };
    assert_eq!(
        reason.downcast_ref::<message::CarriedTextReplayError>(),
        Some(&message::CarriedTextReplayError::EditedOpaqueText)
    );

    // G90.3: original anthropic_content/server_tool_use/srv_1 setter input and
    // its empty-vector success are retired in the correspondence. This separate
    // G84 canonical empty control carries no foreign bag or raw authority.
    assert!(
        input_items(assistant(vec![CoreContent::text("")]))
            .expect("canonical empty input succeeds")
            .is_empty()
    );

    // G90.4/5: raw positioned empty part emits one, with or without msg_1.
    let empty = json!({"type":"output_text","text":"","annotations":["kept"]});
    for id in [Some("msg_1"), None] {
        let mut message = json!({"type":"message","content":[empty.clone()]});
        if let Some(id) = id {
            message["id"] = json!(id);
        }
        let parts =
            carried_message_parts(&message, 0, None).expect("current empty raw part factory");
        let items = input_items(assistant(
            parts.into_iter().map(CoreContent::CarriedText).collect(),
        ))
        .expect("owned empty raw input converts");
        assert_eq!(
            items.len(),
            1,
            "current raw category remains deliverable with or without containing ID"
        );
        assert_eq!(items[0]["content"][0], empty);
        assert_eq!(items[0]["content"][0]["annotations"], json!(["kept"]));
        if let Some(id) = id {
            assert_eq!(items[0]["id"], id);
        } else {
            assert!(items[0].get("id").is_none());
        }
    }

    // G90.6: keep the direct bare canonical constructor; only the two
    // unavailable additional_params/extras views retire, with no invented body pin.
    let _bare = CoreContent::Text(message::Text::new("plain"));
}

// D30; approved G82-G89 current boundary and named exceptions.
#[test]
fn structured_reasoning_with_id_still_converts_to_input_item() {
    let items = input_items(assistant(vec![unissued(
        "rs_123",
        vec![Part::Summary("structured summary".to_owned())],
    )]))
    .expect("structured reasoning converts");
    assert_eq!(items.len(), 1);
    assert!(items[0].get("role").is_none());
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(items[0]["id"], "rs_123");
    assert_eq!(
        items[0]["summary"],
        json!([{"type":"summary_text","text":"structured summary"}])
    );
}

// D31; approved G82-G89 current boundary and named exceptions.
#[test]
fn the_last_text_signature_is_the_items() {
    let CoreContent::CarriedReasoning(reasoning) = unissued(
        "rs_1",
        vec![
            Part::Text {
                text: "first".to_owned(),
                signature: Some("early".to_owned()),
            },
            Part::Text {
                text: "second".to_owned(),
                signature: Some("final".to_owned()),
            },
        ],
    ) else {
        unreachable!()
    };
    let item = carried_reasoning_item(&reasoning)
        .expect("identified reasoning replays")
        .expect("reasoning item");
    assert_eq!(item["signature"], "final");
    assert_eq!(
        item["content"],
        json!([{"type":"reasoning_text","text":"first"},{"type":"reasoning_text","text":"second"}])
    );
}

// D32; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_unary_asserted_call_with_unparseable_arguments_is_refused_by_name() {
    for status in ["completed", "in_progress"] {
        let error = fold_body(malformed_body(status))
            .expect_err("asserted malformed call is a response defect");
        assert!(
            matches!(&error, ProviderError::Relayed(_)),
            "{status}: {error:?}"
        );
        let report = error.report();
        assert_eq!(report.kind, crate::error::ErrorKind::Response);
        assert!(!report.retryable);
        assert!(
            report
                .message
                .starts_with("tool call `add` arrived with malformed JSON input"),
            "{status}: {error:?}"
        );
        let Some(crate::error::ErrorDetail::MalformedToolInput(input)) = &report.detail else {
            panic!("typed malformed input")
        };
        assert_eq!(input.raw, "{\"x\":481");
        assert_eq!(input.name.as_str(), "add");
        assert_eq!(input.id.provider().map(|id| id.as_str()), Some("call_1"));
    }
}

// D33; approved G82-G89 current boundary and named exceptions.
#[test]
fn a_unary_unasserted_call_with_unparseable_arguments_is_dropped() {
    let response = fold_body(malformed_body("incomplete"))
        .expect("unasserted malformed call is dropped, not an error");
    assert!(response.choice.is_empty(), "{:?}", response.choice);
}

// D34; approved G82-G89 current boundary and named exceptions.
#[test]
fn argument_classification_keeps_empty_and_malformed_meaning() {
    for (arguments, expected) in [
        ("", json!({})),
        ("  ", json!({})),
        ("{\"x\":1}", json!({"x":1})),
    ] {
        let item = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":arguments,"status":"completed"});
        let response = fold_items(vec![item]);
        let [CoreContent::ToolCall(call)] = response.choice.as_slice() else {
            panic!("function call")
        };
        assert_eq!(call.function.arguments_value(), expected);
    }
    let malformed = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":"{\"x\":","status":"completed"});
    let error = fold_body(item_body(vec![malformed])).expect_err("malformed input stays refused");
    let report = error.report();
    let Some(crate::error::ErrorDetail::MalformedToolInput(input)) = &report.detail else {
        panic!("typed raw")
    };
    assert_eq!(input.raw, "{\"x\":");
    let nonstring = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":{"x":1},"status":"completed"});
    assert!(fold_body(item_body(vec![nonstring])).is_err());
}

// D36; approved G82-G89 current boundary and named exceptions.
#[test]
fn an_omitted_cached_token_count_is_absent_not_zero() {
    for (details, expected) in [(json!({}), None), (json!({"cached_tokens":0}), Some(0))] {
        let usage = json!({"input_tokens":3,"input_tokens_details":details,"output_tokens":2,"total_tokens":5});
        let mut body = item_body(vec![]);
        body["usage"] = usage.clone();
        let response = fold_body(body).expect("usage decodes");
        assert_eq!(response.usage.cached_input_tokens, expected);
        assert_eq!(response.raw["usage"], usage);
        assert_eq!(response.raw["usage"]["input_tokens_details"], details);
    }
}

// D37; approved G82-G89 current boundary and named exceptions.
#[test]
fn responses_usage_deserializes_without_output_token_details() {
    let usage = json!({"input_tokens":100,"input_tokens_details":{"cached_tokens":25},"output_tokens":50,"total_tokens":150});
    let mut body = item_body(vec![]);
    body["usage"] = usage.clone();
    let response = fold_body(body).expect("usage decodes without output details");
    assert!(response.raw["usage"].get("output_tokens_details").is_none());
    let token_usage = response.usage;
    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, None);
    assert_eq!(token_usage.total_tokens, Some(150));
}

// D38; approved G82-G89 current boundary and named exceptions.
#[test]
fn responses_usage_token_usage_preserves_reasoning_tokens() {
    let usage = json!({"input_tokens":100,"input_tokens_details":{"cached_tokens":25},"output_tokens":50,"output_tokens_details":{"reasoning_tokens":15},"total_tokens":150});
    let mut body = item_body(vec![]);
    body["usage"] = usage.clone();
    let response = fold_body(body).expect("usage decodes");
    let token_usage = response.usage;
    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, Some(15));
    assert_eq!(token_usage.total_tokens, Some(150));
}

// D39; approved G82-G89 current boundary and named exceptions.
#[tokio::test]
async fn captured_raw_refolds_with_the_same_request_context_and_full_usage() {
    const REQUEST_ID: &str = "req_unit_responses_0001";
    const BODY: &str = r#"{
            "id": "resp_raw_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-mini-2024-07-18",
            "service_tier": "default",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_raw_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }],
            "tools": []
        }"#;
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
    let model = crate::driver::Model::new(
        openai_wire("gpt-4o-mini"),
        crate::test_utils::RecordingHttpClient::with_error_response_headers(
            http::StatusCode::OK,
            BODY,
            headers.clone(),
        ),
    );
    let response = model
        .call(completion::CompletionRequest::new("hello"))
        .await
        .expect("completion");
    let raw = &response.raw;
    assert_eq!(
        *raw,
        serde_json::from_str::<Value>(BODY).expect("original received body")
    );
    assert_eq!(
        response
            .extras::<OpenAiExt>()
            .expect("the actual response is from OpenAI")
            .expect("received OpenAI extras deserialize")
            .service_tier
            .as_deref(),
        Some("default")
    );
    assert_eq!(raw["service_tier"], "default");
    assert!(raw.get("provider_request_id").is_none());
    persist(&response);
    let captured_body = raw.to_string();
    let model = crate::driver::Model::new(
        openai_wire("gpt-4o-mini"),
        crate::test_utils::RecordingHttpClient::with_error_response_headers(
            http::StatusCode::OK,
            captured_body,
            headers,
        ),
    );
    let refolded = model
        .call(completion::CompletionRequest::new("hello"))
        .await
        .expect("re-fold the capture");
    assert_eq!(response.identity(), refolded.identity());
    assert_eq!(response.finish_reason(), refolded.finish_reason());
    assert_eq!(response.model(), refolded.model());
    assert_eq!(response.usage, refolded.usage);
    assert_eq!(response.choice, refolded.choice);
    assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    assert_eq!(
        response
            .identity()
            .message_id
            .as_ref()
            .map(|id| id.as_str()),
        Some("msg_raw_1")
    );
}

// D40; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_accepts_null_metadata() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "openai-compatible-model",
        "metadata": null,
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("response with null metadata decodes");
    assert_eq!(response.raw, received);
    assert_eq!(response.raw["metadata"], Value::Null);
    assert_eq!(response.text(), "done");
}

// D41; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_accepts_reasoning_only_response() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking only",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("reasoning-only response converts");
    assert_eq!(response.raw, received);
    assert_eq!(response.choice.len(), 1);
    assert!(
        matches!(&response.choice[0],CoreContent::Reasoning(reasoning) if reasoning.text=="thinking only")
    );
}

// D42; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_accepts_top_level_reasoning_string() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking through the answer",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("top-level reasoning response converts");
    assert_eq!(response.raw, received);
    assert_eq!(response.raw["reasoning"], "thinking through the answer");
    assert!(
        matches!(response.choice.as_slice(),[CoreContent::Reasoning(reasoning),CoreContent::CarriedText(text)] if reasoning.text=="thinking through the answer"&&text.text()=="done")
    );
    persist(&response);
    let (_, ended, events) = crate::test_utils::history_conformance::cut(
        &openai_wire("gpt-5-mini"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(received.to_string())],
    );
    assert!(ended);
    assert!(events.iter().any(|event|matches!(event,crate::streaming::Item::Event(crate::streaming::StreamEvent::End{content:CoreContent::Reasoning(reasoning),reasoning_authority:Some(crate::streaming::ReasoningAuthority::ProviderRestated),..}) if reasoning.text=="thinking through the answer")));
}

// D43; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_carries_the_message_id_not_the_response_id() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("response converts");
    assert_eq!(response.raw, received);
    assert_eq!(
        response.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_456")
    );
    assert_eq!(response.response_id(), Some("resp_123"));
    assert_eq!(response.provider(), "openai");
    assert_eq!(response.model(), Some("gpt-5.4"));
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
}

// D44; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_completed_with_tool_call_reports_tool_calls() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"London\"}",
            "status": "completed"
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("response converts");
    assert_eq!(response.raw, received);
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

// D45; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_decodes_numeric_top_p() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(0.95);
    let response = fold_body(body.clone()).expect("numeric top_p decodes");
    assert_eq!(response.raw, body);
    assert_eq!(response.raw["top_p"], json!(0.95));
    assert!(matches!(
        response.choice.first(),
        Some(CoreContent::ToolCall(_))
    ));
}

// D46; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_does_not_duplicate_structured_reasoning() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "reasoning": "provider top-level text",
        "output": [{
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{
                "type": "summary_text",
                "text": "structured summary"
            }]
        }, {
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("response converts");
    assert_eq!(response.raw, received);
    let reasoning_count = response
        .choice
        .iter()
        .filter(|item| {
            matches!(
                item,
                CoreContent::Reasoning(_) | CoreContent::CarriedReasoning(_)
            )
        })
        .count();
    assert_eq!(reasoning_count, 1);
    let [
        CoreContent::CarriedReasoning(reasoning),
        CoreContent::CarriedText(text),
    ] = response.choice.as_slice()
    else {
        panic!("structured reasoning then text")
    };
    assert_eq!(
        reasoning.content,
        vec![Part::Summary("structured summary".to_owned())]
    );
    assert_eq!(text.text(), "done");
    assert!(!response.choice.iter().any(|item|matches!(item,CoreContent::Reasoning(reasoning) if reasoning.text=="provider top-level text")));
}

// D47; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_accepts_mistyped_store_and_keeps_the_received_value() {
    let mut body = response_with_object_top_p();
    body["store"] = json!("not-a-bool");
    body["top_p"] = json!(0.5);
    body["prompt_cache_key"] = json!("cache-1");
    let response = fold_body(body.clone()).expect("mistyped store does not fail response");
    assert_eq!(response.raw, body);
    assert_eq!(response.raw["store"], "not-a-bool");
    assert_eq!(response.raw["top_p"], json!(0.5));
    assert_eq!(response.raw["prompt_cache_key"], "cache-1");
}

// D48; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_ignores_unsupported_reasoning_shapes() {
    for reasoning in [Value::Null, json!(["unexpected"]), json!(42), json!(true)] {
        let received = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "openai-compatible-model",
            "reasoning": reasoning,
            "output": [],
            "tools": []
        });
        let response =
            fold_body(received.clone()).expect("unsupported reasoning shapes are nonfatal");
        assert_eq!(response.raw, received);
        assert!(response.choice.is_empty());
    }
}

// D49; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_preserves_context_without_treating_config_as_text() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": {
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("object reasoning is tolerated");
    assert_eq!(response.raw, received);
    let extras = response
        .extras::<OpenAiExt>()
        .expect("the actual response is from OpenAI")
        .expect("received OpenAI extras deserialize");
    assert_eq!(extras.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(extras.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(extras.reasoning_mode.as_deref(), Some("standard"));
    assert_eq!(extras.reasoning_summary, None);
    assert!(
        matches!(response.choice.as_slice(),[CoreContent::CarriedText(text)] if text.text()=="done")
    );
}

// D50; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_preserves_unknown_reasoning_metadata_and_nulls() {
    let metadata = json!({"context":"future_context","effort":"ultra","summary":null,"future_control":{"depth":3}});
    let received = json!({"id":"resp_123","object":"response","created_at":0,"status":"completed","model":"gpt-future","reasoning":metadata,"output":[],"tools":[]});
    let response = fold_body(received.clone()).expect("unknown reasoning metadata decodes");
    assert_eq!(response.raw, received);
    assert_eq!(response.raw["reasoning"], metadata);
    let extras = response
        .extras::<OpenAiExt>()
        .expect("the actual response is from OpenAI")
        .expect("received OpenAI extras deserialize");
    assert_eq!(extras.reasoning_context.as_deref(), Some("future_context"));
    assert_eq!(extras.reasoning_effort.as_deref(), Some("ultra"));
    assert_eq!(extras.reasoning_summary, None);
    persist(&response);
}

// D51; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_provider_comes_from_the_configured_chatgpt_wire() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.3-codex",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "done" }]
        }],
        "tools": []
    });
    let identity = crate::providers::openai::wire::CallerIdentity::new(
        "fixture-originator",
        "fixture-user-agent",
        None,
    )
    .expect("fixture caller");
    let wire = crate::providers::openai::OpenAIConfig::with_key(
        &crate::providers::chatgpt::DIALECT,
        "dummy-key",
    )
    .with_caller_identity(identity)
    .with_account_id("fixture-account")
    .responses("gpt-5.3-codex");
    let response = crate::test_utils::decode_reply(
        &wire,
        &completion::CompletionRequest::new("hello"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(received.to_string())],
        Value::Null,
    )
    .expect("actual ChatGPT wire decode");
    assert_eq!(response.provider(), "chatgpt");
    assert_eq!(response.raw, received);
}

// D53; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_round_trips_echoed_metadata() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(1.0);
    body["store"] = json!(true);
    body["metadata"] = json!({"k":"v"});
    let response = fold_body(body.clone()).expect("response decodes");
    for key in [
        "top_p",
        "store",
        "metadata",
        "service_tier",
        "usage",
        "output",
        "model",
    ] {
        assert_eq!(response.raw[key], body[key], "field {key} retained exactly");
    }
    assert_eq!(response.raw, body);
    persist(&response);
}

// D54; approved G82-G89 current boundary and named exceptions.
#[test]
fn completion_response_tolerates_object_shaped_top_p() {
    let body = response_with_object_top_p();
    let response = fold_body(body.clone()).expect("object top_p does not fail response");
    assert_eq!(response.raw, body);
    assert_eq!(response.raw["top_p"], json!({"value":0.95}));
    assert_eq!(
        response
            .extras::<OpenAiExt>()
            .expect("the actual response is from OpenAI")
            .expect("received OpenAI extras deserialize")
            .service_tier
            .as_deref(),
        Some("default")
    );
    assert!(
        matches!(response.choice.first(),Some(CoreContent::ToolCall(call)) if call.function.name.as_str()=="get_weather")
    );
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

// D55; approved G82-G89 current boundary and named exceptions.
#[test]
fn finish_reason_maps_every_documented_terminal_state() {
    for (status, reason, expected) in [
        ("completed", None, Some(completion::FinishReason::Stop)),
        (
            "incomplete",
            Some("max_output_tokens"),
            Some(completion::FinishReason::Length),
        ),
        (
            "incomplete",
            Some("content_filter"),
            Some(completion::FinishReason::ContentFilter),
        ),
        (
            "completed",
            Some("noise"),
            Some(completion::FinishReason::Stop),
        ),
        ("in_progress", None, None),
        ("queued", None, None),
    ] {
        let mut value = json!({"status":status});
        if let Some(reason) = reason {
            value["incomplete_details"] = json!({"reason":reason});
        }
        assert_eq!(streaming::finish_reason_of(&value).0, expected);
    }
}

// D56; approved G82-G89 current boundary and named exceptions.
#[test]
fn finish_reason_preserves_unknown_values_verbatim() {
    for (status, reason, expected) in [
        ("incomplete", Some("MAX_TOOL_CALLS"), "MAX_TOOL_CALLS"),
        ("failed", None, "failed"),
        ("cancelled", None, "cancelled"),
        ("throttled", None, "throttled"),
        ("incomplete", None, "incomplete"),
        ("incomplete", Some(""), "incomplete"),
    ] {
        let mut value = json!({"status":status});
        if let Some(reason) = reason {
            value["incomplete_details"] = json!({"reason":reason});
        }
        assert_eq!(
            streaming::finish_reason_of(&value).0,
            Some(completion::FinishReason::Other(expected.to_owned()))
        );
        assert_eq!(value["status"], status);
    }
    let mut received = item_body(vec![]);
    received["status"] = json!("throttled");
    let response = fold_body(received.clone()).expect("future Whole status decodes");
    assert_eq!(response.raw, received);
    assert_eq!(response.raw["status"], "throttled");
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::Other("throttled".to_owned()))
    );
}

// D57; approved G82-G89 current boundary and named exceptions.
#[test]
fn truncated_incomplete_response_surfaces_length_not_an_error() {
    let received = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "model": "gpt-test",
        "output": [],
        "tools": []
    });
    let response = fold_body(received.clone()).expect("truncated incomplete Whole is not an error");
    assert_eq!(response.raw, received);
    assert!(response.choice.is_empty());
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::Length)
    );
}

// D58; approved G82-G89 current boundary and named exceptions.
#[test]
fn declared_tools_convert_to_declarations_after_the_typed_tools() {
    let mut request = weather_tool_request();
    request.additional_params = Some(json!({"tools":declared_tools_payload()}));
    let req = convert("gpt-4o-mini", request).expect("request converts");
    let tools = req["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 3);
    assert_eq!(Value::Array(tools[1..].to_vec()), declared_tools_payload());
    let namespace = ResponsesRequestTool::Declared(
        DeclaredResponsesTool::from_value(tools[1].clone()).expect("real declared namespace"),
    );
    let function = ResponsesRequestTool::Declared(
        DeclaredResponsesTool::from_value(tools[2].clone()).expect("real declared function"),
    );
    assert_eq!(namespace.definition().kind, "namespace");
    assert_eq!(function.definition().name, "post_to_bus");
}

// D59; approved G82-G89 current boundary and named exceptions.
#[test]
fn strict_tools_apply_only_their_own_transformation_to_declared_tools() {
    let wire = openai_wire("gpt-4o-mini").with_strict_tools();

    let function = json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "defer_loading": false,
        "x_vendor_hint": null,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    });
    let hosted = json!({"type": "web_search", "search_context_size": "low"});
    let mut declared = declared_tools_payload();
    declared[1] = function.clone();
    declared
        .as_array_mut()
        .expect("the fixture is an array")
        .push(hosted.clone());

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({ "tools": declared.clone() }));
    let req = wire_request(&wire, request);
    let tools = req["tools"].clone();

    // Neither the namespace nor the hosted tool is a function, so strict mode
    // has nothing to change; in particular it adds no `strict`.
    assert_eq!(tools[1], declared[0]);
    assert_eq!(tools[3], hosted);

    // The declared function stays a declaration, gains exactly strict mode's
    // transformation, and keeps every other member it was written with.
    let declaration: ResponsesRequestTool =
        serde_json::from_value(tools[2].clone()).expect("real encoded declaration");
    assert!(matches!(declaration, ResponsesRequestTool::Declared(_)));
    let rewritten = tools[2].as_object().expect("a declared tool is an object");
    assert_eq!(rewritten["strict"], json!(true));
    assert_eq!(
        rewritten["parameters"]["additionalProperties"],
        json!(false)
    );
    let mut untouched = rewritten.clone();
    untouched.remove("strict");
    untouched.remove("parameters");
    let mut expected = function.as_object().expect("fixture is an object").clone();
    expected.remove("strict");
    expected.remove("parameters");
    assert_eq!(
        untouched, expected,
        "strict mode must leave the empty description, the explicit null and \
         false, and the unmodelled member exactly as declared"
    );

    // The typed view agrees with what is serialized.
    let definition = declaration.definition();
    assert!(definition.strict);
    assert_eq!(definition.parameters, rewritten["parameters"]);

    // Strict mode is idempotent on a declared tool.
    let again = declaration.clone().with_strict();
    assert_eq!(again, declaration);
}

// D60; approved G82-G89 current boundary and named exceptions.
#[test]
fn reasoning_context_values_survive_request_conversion() {
    for (context, wire_value) in [
        (RawReasoningContext::Auto, "auto"),
        (RawReasoningContext::AllTurns, "all_turns"),
        (RawReasoningContext::CurrentTurn, "current_turn"),
    ] {
        let reasoning = RawReasoning {
            effort: None,
            summary: None,
            mode: None,
            context: Some(context),
        };
        assert_eq!(
            serde_json::to_value(reasoning).expect("private production projection serializes"),
            json!({"context":wire_value})
        );
        let mut request = request_with_preamble("You are concise.");
        request.additional_params = Some(json!({"reasoning":{"context":wire_value}}));
        let serialized = convert("gpt-5.6", request).expect("request converts");
        assert_eq!(serialized["reasoning"], json!({"context":wire_value}));
    }
}

// D61; approved G82-G89 current boundary and named exceptions.
#[test]
fn reasoning_omits_unset_optional_fields() {
    let reasoning = RawReasoning {
        effort: None,
        summary: None,
        mode: Some(RawReasoningMode::Pro),
        context: None,
    };
    assert_eq!(
        serde_json::to_value(reasoning).expect("production projection serializes"),
        json!({"mode":"pro"})
    );
    let reasoning = RawReasoning {
        effort: Some(RawReasoningEffort::Max),
        summary: Some(RawReasoningSummary::Detailed),
        mode: Some(RawReasoningMode::Pro),
        context: Some(RawReasoningContext::CurrentTurn),
    };
    assert_eq!(
        serde_json::to_value(reasoning).expect("production projection serializes"),
        json!({"effort":"max","mode":"pro","context":"current_turn","summary":"detailed"})
    );
}

// Focused controls authorized with coord-a-rig-g91-1; private current encoder boundary.
#[test]
fn carried_reasoning_object_preserves_root_and_known_part_siblings() {
    let raw = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"summary","future_summary":{"list":[1,2]}}],"content":[{"type":"reasoning_text","text":"visible","future_content":false}],"encrypted_content":"enc","signature":"sig","status":"completed","future_root":{"leaf":null}});
    let reasoning = carried_reasoning(&raw, None).expect("existing private carried reader");
    let encoded = carried_reasoning_item(&reasoning)
        .expect("object reasoning encodes")
        .expect("identified reasoning");
    let mut expected = raw.clone();
    expected.as_object_mut().expect("object").remove("status");
    assert_eq!(
        encoded, expected,
        "only the existing status omission applies; siblings stay exact"
    );
}

#[test]
fn explicitly_unissued_null_reasoning_projects_known_parts_without_raw_evidence() {
    let content = vec![
        Part::Summary("summary".to_owned()),
        Part::Text {
            text: "visible".to_owned(),
            signature: Some("sig".to_owned()),
        },
        Part::Encrypted("enc".to_owned()),
    ];
    let CoreContent::CarriedReasoning(reasoning) = unissued("rs_1", content.clone()) else {
        unreachable!()
    };
    assert_eq!(reasoning.payload.item, Value::Null);
    assert_eq!(reasoning.payload.provider, None);
    assert_eq!(reasoning.content, content);
    let encoded = carried_reasoning_item(&reasoning)
        .expect("explicitly unissued known reasoning encodes")
        .expect("identified reasoning");
    assert_eq!(
        encoded,
        json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"summary"}],"content":[{"type":"reasoning_text","text":"visible"}],"encrypted_content":"enc","signature":"sig"})
    );
}

#[test]
fn null_reasoning_with_any_issuer_is_not_treated_as_unissued() {
    for issuer in ["openai", "anthropic"] {
        let reasoning = CarriedReasoning::new(
            vec![Part::Summary("summary".to_owned())],
            CarriedPayload::new(Value::Null, Some(issuer.to_owned())),
        )
        .with_id(Some("rs_1".to_owned()));
        assert!(
            carried_reasoning_item(&reasoning).is_err(),
            "a Null payload with issuer {issuer} is not unissued"
        );
    }
}

#[test]
fn unissued_null_reasoning_cannot_project_opaque_parts() {
    for summary in [true, false] {
        let raw = CarriedPayload::new(json!({"type":"future","list":[1,2]}), None);
        let part = if summary {
            Part::OpaqueSummary(raw)
        } else {
            Part::OpaqueContent(raw)
        };
        let reasoning = CarriedReasoning::new(vec![part], CarriedPayload::new(Value::Null, None))
            .with_id(Some("rs_1".to_owned()));
        assert!(
            carried_reasoning_item(&reasoning).is_err(),
            "opaque Null payload is not projected"
        );
    }
}

#[test]
fn other_nonobject_payloads_and_empty_identifiers_keep_the_existing_refusal() {
    for raw in [json!("raw"), json!(7), json!(true), json!([])] {
        let reasoning = CarriedReasoning::new(
            vec![Part::Summary("summary".to_owned())],
            CarriedPayload::new(raw, None),
        )
        .with_id(Some("rs_1".to_owned()));
        assert!(carried_reasoning_item(&reasoning).is_err());
    }
    let CoreContent::CarriedReasoning(reasoning) =
        unissued("", vec![Part::Summary("summary".to_owned())])
    else {
        unreachable!()
    };
    assert!(
        carried_reasoning_item(&reasoning).is_err(),
        "a Null projection requires an existing nonempty ID"
    );
}
