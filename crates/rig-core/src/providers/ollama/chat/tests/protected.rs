use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::json_body;
use crate::wire::{Mode, Wire};
use serde_json::{Value, json};

fn encode(model: &str, request: CompletionRequest) -> Result<Value, EncodeError> {
    Ok(json_body(
        &OllamaConfig::new()
            .completion(model)
            .encode(request, Mode::Unary)?
            .request,
    ))
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": true,
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request =
        encode("qwen3:8b", completion_request).expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": true,
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_low_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "low",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request =
        encode("qwen3:8b", completion_request).expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "low",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_medium_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "medium",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request =
        encode("qwen3:8b", completion_request).expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "medium",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_high_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "high",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request =
        encode("qwen3:8b", completion_request).expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "high",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_invalid_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "invalid",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request = encode("qwen3:8b", completion_request);

    assert!(ollama_request.is_err());
}

// Test that `think` is omitted when not specified, so Ollama applies the
// model's default thinking behavior (issue #1970)
#[test]
fn test_completion_request_with_think_omitted_by_default() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest WITHOUT "think" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.5),
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        ..CompletionRequest::new("")
    };

    // Encode through the native wire.
    let ollama_request =
        encode("llama3.2", completion_request).expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert that "think" is absent (so Ollama uses the model default) and
    // "keep_alive" is not present
    let expected = json!({
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "stream": false,
        "options": {
            "temperature": 0.5
        }
    });

    assert_eq!(serialized, expected);
}

// The native API takes the token limit as `options.num_predict`; an
// explicit `num_predict` in `additional_params` wins over
// `CompletionRequest::max_tokens`.
#[test]
fn test_completion_request_num_predict_from_additional_params_wins() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({ "num_predict": 42 })),
        output_schema: None,
        ..CompletionRequest::new("")
    };

    let ollama_request =
        encode("llama3.2", completion_request).expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(serialized["options"], json!({ "num_predict": 42 }));
    assert_eq!(serialized.get("max_tokens"), None);
}

// The plain path: `max_tokens` with no `additional_params` at all, which
// skips the merge and serializes `base_options` directly. Every other
// `max_tokens` test also sets `additional_params`, so without this one the
// branch the fix exists for is never exercised.
#[test]
fn test_completion_request_num_predict_without_additional_params() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        ..CompletionRequest::new("")
    };

    let ollama_request =
        encode("llama3.2", completion_request).expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(
        serialized["options"],
        json!({ "temperature": 0.7, "num_predict": 1024 })
    );
    // Neither belongs at the top level of a native `/api/chat` payload.
    assert_eq!(serialized.get("max_tokens"), None);
    assert_eq!(serialized.get("temperature"), None);
}

// With nothing to put in it, `options` is an empty object rather than
// carrying `"temperature": null` as it did when temperature was seeded
// unconditionally.
#[test]
fn test_completion_request_options_omit_unset_parameters() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        ..CompletionRequest::new("")
    };

    let ollama_request =
        encode("llama3.2", completion_request).expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(serialized["options"], json!({}));
}

#[test]
fn test_completion_request_with_output_schema() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let schema: schemars::Schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "age": { "type": "integer" },
            "available": { "type": "boolean" }
        },
        "required": ["age", "available"]
    }))
    .expect("Failed to parse schema");

    let completion_request = CompletionRequest {
        model: Some("llama3.1".to_string()),
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new(
                "How old is Ollama?".to_string(),
            ))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: Some(schema),
        ..CompletionRequest::new("")
    };

    let ollama_request =
        encode("llama3.1", completion_request).expect("Failed to create Ollama request");

    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    let format = serialized
        .get("format")
        .expect("format field should be present");
    assert_eq!(
        *format,
        json!({
            "type": "object",
            "properties": {
                "age": { "type": "integer" },
                "available": { "type": "boolean" }
            },
            "required": ["age", "available"]
        })
    );
}

#[test]
fn test_completion_request_without_output_schema() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: Some("llama3.1".to_string()),
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        ..CompletionRequest::new("")
    };

    let ollama_request =
        encode("llama3.1", completion_request).expect("Failed to create Ollama request");

    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert!(
        serialized.get("format").is_none(),
        "format field should be absent when output_schema is None"
    );
}

#[test]
fn ollama_chat_refuses_namespaced_and_custom_calls() {
    use crate::error::ProviderError;
    use crate::message;
    let name = message::ToolName::new("tool").expect("tool name");
    let mut function = message::ToolFunction::new(name.clone(), json!({}));
    function.namespace = Some("".to_owned());
    let namespaced =
        message::AssistantContent::ToolCall(message::ToolCall::from_wire("call_1", function));
    let custom = message::AssistantContent::custom_tool_call(
        "call_2",
        name,
        Some("ns".to_owned()),
        "{raw input}",
    );
    for (content, custom) in [(namespaced, false), (custom, true)] {
        let error: ProviderError = OllamaConfig::new()
            .completion("qwen3:8b")
            .body(
                &CompletionRequest::from(vec![message::Message::Assistant(
                    message::AssistantMessage::new(vec![content]),
                )]),
                Mode::Unary,
            )
            .expect_err("refused, not dropped")
            .into();
        let ProviderError::Request(cause) = error else {
            panic!("the refusal is a typed request failure");
        };
        let cause = cause
            .downcast_ref::<message::UnrepresentableToolCall>()
            .expect("the typed cause survives");
        if custom {
            assert!(
                matches!(cause, message::UnrepresentableToolCall::CustomCall { wire, name }
                if *wire == super::super::OLLAMA_CHAT_WIRE && name.as_str() == "tool")
            );
        } else {
            assert!(
                matches!(cause, message::UnrepresentableToolCall::Namespace { wire, name, namespace }
                if *wire == super::super::OLLAMA_CHAT_WIRE && name.as_str() == "tool" && namespace.is_empty())
            );
        }
    }
}

#[test]
fn mixed_user_content_preserves_message_order() {
    use crate::message::{
        AnsweredToolCall, CallId, LocalCallId, Message, ToolName, ToolResult, ToolResultContent,
        UserContent,
    };
    let message = Message::User {
        content: vec![
            UserContent::text("before"),
            UserContent::ToolResult(ToolResult {
                call: CallId::Local(LocalCallId::new()),
                name: ToolName::new("lookup").expect("name"),
                answers: AnsweredToolCall::Function,
                content: vec![ToolResultContent::json(json!({"ok":true}))],
                is_error: false,
            }),
            UserContent::text("after"),
        ],
    };
    let converted = encode("qwen3:8b", CompletionRequest::from(vec![message]))
        .expect("mixed content should convert");
    let messages = converted["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "before");
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["tool_name"], "lookup");
    assert_eq!(messages[1]["content"], r#"{"ok":true}"#);
    assert!(messages[1].get("tool_call_id").is_none());
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"], "after");
}

#[test]
fn daemon_issued_call_ids_replay_and_minted_handles_do_not() {
    use crate::message::{AssistantContent, AssistantMessage, CallId, LocalCallId, Message,
        ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent};
    for id in [CallId::from_wire("call_daemon_1"), CallId::Local(LocalCallId::new())] {
        let call = ToolCall::new(id.clone(), ToolFunction::new(ToolName::new("add").expect("name"), json!({"x": 1})));
        let result = call.result(vec![ToolResultContent::text("2")]);
        let body = encode("qwen3:8b", CompletionRequest::from(vec![
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call)])),
            Message::User { content: vec![UserContent::ToolResult(result)] },
        ])).expect("optional IDs encode");
        let assistant = &body["messages"][0];
        let tool = &body["messages"][1];
        assert_eq!(tool["tool_name"], "add");
        match id {
            CallId::Provider(_) => {
                assert_eq!(assistant["tool_calls"][0]["id"], "call_daemon_1");
                assert_eq!(tool["tool_call_id"], "call_daemon_1");
            }
            CallId::Local(_) => {
                assert!(assistant["tool_calls"][0].get("id").is_none());
                assert!(tool.get("tool_call_id").is_none());
            }
        }
    }
}

#[test]
fn local_native_call_replays_without_an_id_slot() {
    use crate::message::{AssistantContent, AssistantMessage, Message, ToolResultContent, UserContent};
    let raw = json!({"model":"qwen3:8b", "created_at":"2024-01-01T00:00:00Z", "done":true,
        "message":{"role":"assistant", "content":"", "tool_calls":[
            {"function":{"name":"add", "arguments":{"x":1}}, "wire_extra":"keep"}
        ]}});
    let wire = OllamaConfig::new().completion("qwen3:8b");
    let response = crate::test_utils::history::decode(&wire, Mode::Unary, [crate::wire::WireFrame::Text(raw.to_string())])
        .expect("native response decodes");
    let call = response.tool_calls().next().expect("call").clone();
    assert!(call.id.is_local());
    assert!(call.native.is_some());
    let result = call.result(vec![ToolResultContent::text("2")]);
    let body = encode("qwen3:8b", CompletionRequest::from(vec![
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call)])),
        Message::User { content: vec![UserContent::ToolResult(result)] },
    ])).expect("native optional slot replays");
    assert!(body["messages"][0]["tool_calls"][0].get("id").is_none());
    assert_eq!(body["messages"][0]["tool_calls"][0]["wire_extra"], "keep");
    assert!(body["messages"][1].get("tool_call_id").is_none());
}

#[test]
fn unsupported_user_content_returns_a_conversion_error() {
    use crate::message::{ImageMediaType, Message, UserContent};
    let message = Message::User {
        content: vec![UserContent::image_url(
            "https://example.com/image.png",
            Some(ImageMediaType::PNG),
            None,
        )],
    };
    let error = encode("qwen3:8b", CompletionRequest::from(vec![message]))
        .expect_err("URL image should be rejected");
    assert!(error.to_string().contains("base64"));
}

#[test]
fn test_message_conversion_with_thinking() {
    use crate::message::{AssistantContent, AssistantMessage, Message, Reasoning};
    let message = Message::Assistant(AssistantMessage::new(vec![
        AssistantContent::Reasoning(Reasoning::new("Step 1: Consider the problem")),
        AssistantContent::text("The answer is X"),
    ]));
    let body =
        encode("qwen3:8b", CompletionRequest::from(vec![message])).expect("assistant conversion");
    let messages = body["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["thinking"], "Step 1: Consider the problem");
    assert_eq!(messages[0]["content"], "The answer is X");
}

#[test]
fn test_tool_definition_conversion() {
    use crate::completion::ToolDefinition;
    use crate::message::ToolName;
    let parameters = json!({"type":"object","properties":{"location":{"type":"string","description":"The location to get the weather for, e.g. San Francisco, CA"},"format":{"type":"string","description":"The format to return the weather in, e.g. 'celsius' or 'fahrenheit'","enum":["celsius","fahrenheit"]}},"required":["location","format"]});
    let mut request = CompletionRequest::new("hello");
    request.tools = vec![ToolDefinition::new(
        ToolName::new("get_current_weather").expect("name"),
        "Get the current weather for a location",
        parameters,
    )];
    let body = encode("qwen3:8b", request).expect("tool definition encodes");
    let tool = &body["tools"][0];
    assert_eq!(tool["type"], "function");
    assert_eq!(tool["function"]["name"], "get_current_weather");
    assert_eq!(
        tool["function"]["description"],
        "Get the current weather for a location"
    );
    assert_eq!(
        tool["function"]["parameters"]["properties"]["location"]["type"],
        "string"
    );
}
