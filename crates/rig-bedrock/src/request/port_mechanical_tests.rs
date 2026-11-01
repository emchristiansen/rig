use super::*;
use rig_core::completion::ToolDefinition;
use rig_core::message::{Text, ToolName};
use rig_core::wire::{Mode, Wire as _};

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the fixture tool name is nonempty")
}

fn minimal_request() -> CompletionRequest {
    CompletionRequest::new("test")
}

fn encoded(request: CompletionRequest) -> Value {
    let request = Converse::new("")
        .encode(request, Mode::Unary)
        .expect("the fixture encodes");
    serde_json::to_value(request.body).expect("the body serializes")
}

#[test]
fn test_tool_choice_auto_conversion() {
    // Test that rig's ToolChoice::Auto converts to AWS Auto
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

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert!(config.get("toolChoice").is_some());
    assert!(config["toolChoice"].get("auto").is_some());
}

#[test]
fn test_tool_choice_required_conversion() {
    // Test that rig's ToolChoice::Required converts to AWS Any
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

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert!(config.get("toolChoice").is_some());
    assert!(config["toolChoice"].get("any").is_some());
}

#[test]
fn test_tool_choice_none_conversion() {
    // Test that rig's ToolChoice::None disables Bedrock tool configuration entirely.
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

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_none());
}

#[test]
fn test_tool_choice_specific_conversion() {
    // Test that rig's ToolChoice::Specific converts to AWS Tool
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = Some(ToolChoice::Specific {
        function_names: vec![name("specific_tool")],
    });
    request.tools = vec![ToolDefinition {
        name: name("specific_tool"),
        description: "A specific tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert!(config.get("toolChoice").is_some());
    assert!(config["toolChoice"].get("tool").is_some());
    assert_eq!(config["toolChoice"]["tool"]["name"], "specific_tool");
}

#[test]
fn test_no_tool_choice_when_not_specified() {
    // Test that when tool_choice is None (not set), it defaults to None in AWS
    let mut request = minimal_request();
    request.model = None;
    request.tool_choice = None; // Not set
    request.tools = vec![ToolDefinition {
        name: name("test_tool"),
        description: "A test tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }];

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert!(config.get("toolChoice").is_none());
}

#[test]
fn test_tool_with_empty_parameters() {
    // Test that tools with empty parameters (like document_list) work correctly
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

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let tools = config["tools"].as_array().expect("tools are an array");
    assert_eq!(tools.len(), 1);
    assert!(tools[0].get("toolSpec").is_some());
    let spec = &tools[0]["toolSpec"];
    assert_eq!(spec["name"], "document_list");
    assert_eq!(spec["description"], "Lists all documents");
    assert!(spec.get("inputSchema").is_some());
}

#[test]
fn test_tool_with_parameters() {
    // Test that tools with parameters work correctly
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

    let body = encoded(request);
    let tool_config = body.get("toolConfig");
    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let tools = config["tools"].as_array().expect("tools are an array");
    assert_eq!(tools.len(), 1);
    assert!(tools[0].get("toolSpec").is_some());
    let spec = &tools[0]["toolSpec"];
    assert_eq!(spec["name"], "get_weather");
    assert_eq!(spec["description"], "Get weather for a location");
}

#[test]
fn test_system_prompt_includes_system_history() {
    let mut request = minimal_request();
    request.model = None;
    request.chat_history = vec![
        Message::system("History system instruction"),
        Message::User {
            content: vec![UserContent::Text(Text::new("test".to_string()))],
        },
    ];

    let body = encoded(request);
    let system = body.get("system").expect("system prompt should exist");
    let system = system.as_array().expect("system is an array");
    assert_eq!(system.len(), 1);
    assert_eq!(
        system.first(),
        Some(&json!({"text": "History system instruction"}))
    );
}

#[test]
fn test_messages_exclude_system_history() {
    let mut request = minimal_request();
    request.model = None;
    request.chat_history = vec![
        Message::system("History system instruction"),
        Message::User {
            content: vec![UserContent::Text(Text::new("test".to_string()))],
        },
    ];

    let body = encoded(request);
    let messages = body["messages"].as_array().expect("messages are an array");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_output_config_none_when_no_schema() {
    let request = minimal_request();
    let body = encoded(request);
    assert!(body.get("outputConfig").is_none());
}

#[test]
fn test_output_config_with_schema() {
    let schema: schemars::Schema = serde_json::from_value(serde_json::json!({
        "type": "object",
        "title": "WeatherResponse",
        "properties": {
            "temperature": { "type": "number" },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["temperature", "unit"]
    }))
    .expect("valid schema");

    let mut request = minimal_request();
    request.output_schema = Some(schema);

    let body = encoded(request);
    let config = body.get("outputConfig");
    assert!(config.is_some());
    let config = config.unwrap();
    let text_format = config.get("textFormat").expect("text_format should be set");
    assert_eq!(text_format["type"], "json_schema");
    let structure = text_format
        .get("structure")
        .expect("structure should be set");
    let json_schema = structure
        .get("jsonSchema")
        .expect("should be JsonSchema variant");
    assert_eq!(
        json_schema.get("name").and_then(Value::as_str),
        Some("WeatherResponse")
    );
    let parsed: Value =
        serde_json::from_str(json_schema["schema"].as_str().expect("schema is a string"))
            .expect("schema should be valid JSON");
    assert_eq!(parsed["type"], "object");
    assert!(parsed["properties"]["temperature"].is_object());
}

#[test]
fn test_output_config_uses_default_name() {
    let schema: schemars::Schema = serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "result": { "type": "string" }
        }
    }))
    .expect("valid schema");

    let mut request = minimal_request();
    request.output_schema = Some(schema);

    let body = encoded(request);
    let config = body.get("outputConfig").expect("should have config");
    let text_format = config.get("textFormat").expect("text_format should be set");
    let structure = text_format
        .get("structure")
        .expect("structure should be set");
    let json_schema = structure
        .get("jsonSchema")
        .expect("should be JsonSchema variant");
    assert_eq!(
        json_schema.get("name").and_then(Value::as_str),
        Some("response_schema")
    );
}
