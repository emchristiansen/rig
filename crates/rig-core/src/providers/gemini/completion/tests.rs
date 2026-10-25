use crate::message;

use super::*;
use serde_json::json;

#[tokio::test]
async fn test_generate_content_response_deserializes_without_candidates_or_response_id() {
    // Blocked prompt responses can omit default-valued proto fields, including
    // empty repeated `candidates` and empty string `responseId`.
    let body = json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    });
    // A set `blockReason` is the provider's verdict on the prompt: the
    // error names it instead of reporting a generic missing-candidate parse
    // failure.
    let error = fold_unary("gemini-2.5-flash", body.to_string())
        .await
        .expect_err("a blocked prompt is an error");
    assert!(
        matches!(&error, ProviderError::ProviderResponse(response) if response.body.contains("blocked the prompt") && response.body.contains("SAFETY") && response.refusal && response.code.as_deref() == Some("SAFETY")),
        "{error}"
    );
    let report = crate::error::ErrorReport::from(&error);
    assert!(
        report.refusal,
        "the verdict is a refusal on the report: {report:?}"
    );
    assert!(!report.retryable, "{report:?}");
    assert_eq!(report.code.as_deref(), Some("SAFETY"));
}

#[tokio::test]
async fn test_blocked_prompt_error_carries_the_safety_ratings() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({
            "promptFeedback": {
                "blockReason": "PROHIBITED_CONTENT",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                ]
            },
            "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12}
        })
        .to_string(),
    )
    .await
    .expect_err("blocked");
    let message = error.to_string();
    assert!(message.contains("PROHIBITED_CONTENT"), "{message}");
    assert!(
        message.contains("HARM_CATEGORY_DANGEROUS_CONTENT"),
        "{message}"
    );
    assert!(message.contains("HIGH"), "{message}");
}

#[tokio::test]
async fn test_unknown_block_reason_reaches_the_error_verbatim() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "SOMETHING_NEW"}}).to_string(),
    )
    .await
    .expect_err("blocked");
    assert!(
        error.to_string().contains("block_reason=SOMETHING_NEW"),
        "{error}"
    );
    // Unknown means unknown: not a verdict on the content, so retryable.
    assert!(error.is_retryable(), "{error:?}");
}

// A whole reply that names no block and carries no candidate is the
// provider answering with nothing: the shared empty-response rejection,
// not a block.
#[tokio::test]
async fn test_unspecified_block_reason_is_not_a_block() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}}).to_string(),
    )
    .await
    .expect_err("no candidates");
    assert!(
        matches!(&error, ProviderError::Response(message) if message == crate::message::EMPTY_RESPONSE_ERROR),
        "{error}"
    );
}

#[tokio::test]
async fn test_no_candidates_without_prompt_feedback_is_still_a_response_error() {
    let error = fold_unary("gemini-2.5-flash", "{}")
        .await
        .expect_err("empty candidates should become a response error");
    assert!(
        matches!(&error, ProviderError::Response(message) if message == crate::message::EMPTY_RESPONSE_ERROR),
        "{error}"
    );
}

#[test]
fn test_resolve_request_model_uses_override() {
    let mut request = CompletionRequest::new("Hello");
    request.model = Some("gemini-2.5-flash".to_owned());
    let wire = wire("gemini-2.0-flash");
    let unary = wire
        .encode(request.clone(), Mode::Unary)
        .expect("unary request");
    assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(
        unary.request.uri().path(),
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    let streaming = wire
        .encode(request, Mode::Streaming)
        .expect("streaming request");
    assert_eq!(
        streaming.request.uri().path(),
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    );
}

#[test]
fn test_resolve_request_model_uses_default_when_unset() {
    let request = CompletionRequest::new("Hello");
    assert_eq!(request.model, None);
    let encoded = wire("gemini-2.0-flash")
        .encode(request, Mode::Unary)
        .expect("default request");
    assert_eq!(
        encoded.request.uri().path(),
        "/v1beta/models/gemini-2.0-flash:generateContent"
    );
}

#[test]
fn test_message_conversion_user() {
    let msg = message::Message::user("Hello, world!");
    let wire = wire("gemini-2.5-flash");
    let converted = contents(vec![msg], &wire, "gemini-2.5-flash").expect("convert message");
    let content = converted.first().expect("content");
    assert_eq!(content["role"].as_str(), Some("user"));
    let parts = content["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 1);
    if let Some(text) = parts.first().and_then(|part| part["text"].as_str()) {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text part");
    }
}

#[test]
fn test_message_conversion_model() {
    let msg = message::Message::assistant("Hello, user!");
    let wire = wire("gemini-2.5-flash");
    let converted = contents(vec![msg], &wire, "gemini-2.5-flash").expect("convert message");
    let content = converted.first().expect("content");
    assert_eq!(content["role"].as_str(), Some("model"));
    let parts = content["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 1);
    if let Some(text) = parts.first().and_then(|part| part["text"].as_str()) {
        assert_eq!(text, "Hello, user!");
    } else {
        panic!("Expected text part");
    }
}

#[tokio::test]
async fn test_thought_signature_is_preserved_from_response_reasoning_part() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"thinking text","thought":true,"thoughtSignature":"thought_sig_123"}],"role":"model"},"finishReason":"STOP","index":0}]}"#,
    )
    .await;
    let first = converted.choice.first();
    assert!(
        matches!(
            first,
            Some(message::AssistantContent::CarriedReasoning(message::CarriedReasoning { content, .. }))
                if matches!(
                    content.first(),
                    Some(message::CarriedReasoningPart::Text {
                        text,
                        signature: Some(signature)
                    }) if text == "thinking text" && signature == "thought_sig_123"
                )
        ),
        "{first:?}"
    );
}

#[tokio::test]
async fn test_tool_protocol_finish_reason_returns_response_error() {
    for (reason, reason_name, finish_message) in [
        (
            "MALFORMED_FUNCTION_CALL",
            "MalformedFunctionCall",
            "malformed function call: default_api",
        ),
        (
            "UNEXPECTED_TOOL_CALL",
            "UnexpectedToolCall",
            "unexpected tool call: default_api",
        ),
        (
            "MISSING_THOUGHT_SIGNATURE",
            "MissingThoughtSignature",
            "missing thought signature for tool call",
        ),
        (
            "TOO_MANY_TOOL_CALLS",
            "TooManyToolCalls",
            "too many tool calls in response",
        ),
        (
            "MALFORMED_RESPONSE",
            "MalformedResponse",
            "malformed response from provider",
        ),
    ] {
        let body = json!({
            "responseId": "resp_tool_protocol_error",
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "default_api", "args": {"x": 1}}}],
                    "role": "model"
                },
                "finishReason": reason,
                "finishMessage": finish_message,
                "index": 0
            }]
        });

        let err = fold_unary("gemini-2.5-flash", body.to_string())
            .await
            .expect_err("tool protocol finish reason should fail");

        assert!(
            matches!(
                &err,
                ProviderError::Response(message)
                    if message.contains(reason_name)
                        && message.contains(finish_message)
            ),
            "{reason_name}: {err}"
        );
    }
}

#[tokio::test]
async fn test_completion_usage_includes_tool_input_and_reasoning_output() {
    // G199: G190 canonical totals include tool input and thought output.
    let fixture = r#"{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"answer"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":40,"cachedContentTokenCount":20,"candidatesTokenCount":30,"totalTokenCount":100,"thoughtsTokenCount":10,"toolUsePromptTokenCount":12},"modelVersion":"gemini-2.0-flash-001"}"#;
    let converted = unary("gemini-2.5-flash", fixture).await;

    assert_eq!(converted.usage.input_tokens, Some(52));
    assert_eq!(converted.usage.cached_input_tokens, Some(20));
    assert_eq!(converted.usage.output_tokens, Some(40));
    assert_eq!(converted.usage.reasoning_tokens, Some(10));
    assert_eq!(converted.usage.tool_use_prompt_tokens, Some(12));
    assert_eq!(converted.usage.total_tokens, Some(100));
    assert_eq!(
        converted.raw,
        serde_json::from_str::<serde_json::Value>(fixture)
            .expect("the original received body is JSON")
    );
}

#[tokio::test]
async fn test_finish_reason_maps_every_wire_variant() {
    use crate::completion::FinishReason as Normalized;

    for (wire, expected) in [
        ("STOP", Normalized::Stop),
        ("MAX_TOKENS", Normalized::Length),
        ("SAFETY", Normalized::ContentFilter),
        ("BLOCKLIST", Normalized::ContentFilter),
        ("PROHIBITED_CONTENT", Normalized::ContentFilter),
        ("SPII", Normalized::ContentFilter),
        // Everything Gemini reports that rig does not model survives in the
        // provider's own SCREAMING_SNAKE_CASE spelling.
        ("RECITATION", Normalized::Other("RECITATION".to_string())),
        ("LANGUAGE", Normalized::Other("LANGUAGE".to_string())),
        ("OTHER", Normalized::Other("OTHER".to_string())),
        (
            "MALFORMED_FUNCTION_CALL",
            Normalized::Other("MALFORMED_FUNCTION_CALL".to_string()),
        ),
        (
            "UNEXPECTED_TOOL_CALL",
            Normalized::Other("UNEXPECTED_TOOL_CALL".to_string()),
        ),
        (
            "MISSING_THOUGHT_SIGNATURE",
            Normalized::Other("MISSING_THOUGHT_SIGNATURE".to_string()),
        ),
        (
            "TOO_MANY_TOOL_CALLS",
            Normalized::Other("TOO_MANY_TOOL_CALLS".to_string()),
        ),
        (
            "MALFORMED_RESPONSE",
            Normalized::Other("MALFORMED_RESPONSE".to_string()),
        ),
    ] {
        assert_eq!(
            Some(map_google_finish_reason(wire)),
            Some(expected),
            "wire reason {wire:?}"
        );
    }

    // The proto default reports no normalized reason at the decoder boundary.
    let response = unary(
        "gemini-2.5-flash",
        r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"FINISH_REASON_UNSPECIFIED"}]}"#,
    )
    .await;
    assert_eq!(response.finish_reason(), None);
}

#[tokio::test]
async fn test_unary_response_with_unknown_finish_reason_stays_parseable() {
    // A finish reason Google ships tomorrow must not fail the whole
    // payload: content and usage stay intact, and the reason maps to
    // `Other` verbatim — matching the gRPC crate's handling of unknowns.
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-future","candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"FINISH_REASON_FUTURE"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}"#,
    )
    .await;

    assert!(matches!(
        converted.choice.first(),
        Some(message::AssistantContent::Text(text)) if text.text == "hi"
    ));
    assert_eq!(converted.usage.total_tokens, Some(5));
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[tokio::test]
async fn test_completion_response_carries_normalized_metadata() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-meta","modelVersion":"gemini-2.0-flash-001","candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"MAX_TOKENS"}]}"#,
    )
    .await;

    assert_eq!(converted.provider(), PROVIDER_NAME);
    assert_eq!(converted.model(), Some("gemini-2.0-flash-001"));
    assert_eq!(converted.response_id(), Some("resp-meta"));
    assert_eq!(converted.message_id, None);
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn test_completion_response_upgrades_stop_to_tool_calls() {
    // Gemini reports STOP on turns that only emitted a function call; the
    // normalized response must still say `ToolCalls`.
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-tool","candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}],"role":"model"},"finishReason":"STOP"}]}"#,
    )
    .await;

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
    assert_eq!(converted.model(), None);
}

#[test]
fn test_reasoning_signature_is_emitted_in_gemini_part() {
    let reasoning = message::CarriedReasoning::new(
        vec![message::CarriedReasoningPart::Text {
            text: "structured thought".to_owned(),
            signature: Some("reuse_sig_456".to_owned()),
        }],
        message::CarriedPayload::new(Value::Null, None),
    );
    let msg = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::CarriedReasoning(reasoning),
    ]));
    let wire = wire("gemini-2.5-flash");
    let converted = contents(vec![msg], &wire, "gemini-2.5-flash").expect("convert message");
    let first = converted
        .first()
        .and_then(|content| content.pointer("/parts/0"))
        .expect("reasoning part");
    assert_eq!(first["thought"].as_bool(), Some(true));
    assert_eq!(first["thoughtSignature"].as_str(), Some("reuse_sig_456"));
    assert!(matches!(first.get("text"), Some(Value::String(text)) if text == "structured thought"));
}

#[test]
fn test_message_conversion_tool_call() {
    let tool_call = message::ToolCall::from_wire(
        "call-123",
        message::ToolFunction::new(
            message::ToolName::new("test_function").expect("tool name"),
            json!({"arg1":"value1"}),
        ),
    );
    let msg = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::ToolCall(tool_call),
    ]));
    let wire = wire("gemini-2.5-flash");
    let converted = contents(vec![msg], &wire, "gemini-2.5-flash").expect("convert message");
    let content = converted.first().expect("content");
    assert_eq!(content["role"].as_str(), Some("model"));
    let parts = content["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 1);
    if let Some(function_call) = parts.first().and_then(|part| part.get("functionCall")) {
        assert_eq!(function_call["name"].as_str(), Some("test_function"));
        assert_eq!(
            function_call["args"]
                .as_object()
                .expect("arguments")
                .get("arg1"),
            Some(&json!("value1"))
        );
        assert_eq!(function_call["id"].as_str(), Some("call-123"));
    } else {
        panic!("Expected function call part");
    }
}

#[tokio::test]
async fn test_response_function_call_preserves_correlation_id() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"response-123","candidates":[{"content":{"parts":[{"functionCall":{"name":"test_function","args":{"arg1":"value1"},"id":"call-123"}}],"role":"model"},"finishReason":"STOP"}]}"#,
    )
    .await;
    let Some(message::AssistantContent::ToolCall(tool_call)) = converted.choice.first() else {
        panic!("expected a tool call");
    };
    assert_eq!(
        tool_call.id.provider().map(message::ProviderCallId::as_str),
        Some("call-123")
    );
    assert_eq!(
        tool_call.id.provider().expect("wire id").as_str(),
        "call-123"
    );
    assert_eq!(
        tool_call
            .native
            .as_ref()
            .expect("actual decoder retains the received part")
            .item,
        json!({"functionCall":{"name":"test_function","args":{"arg1":"value1"},"id":"call-123"}}),
        "G4: exact received Native item, independent of the retired preparation-only item_id observation"
    );
}

fn encoded_schema_request(parameters: Value) -> Result<Value, ProviderError> {
    use crate::wire::Operation;
    let wire = wire("gemini-2.5-flash");
    let mut request = CompletionRequest::new("hello");
    request.tools = vec![crate::completion::ToolDefinition::new(
        message::ToolName::new("schema_tool").expect("tool name"),
        "schema fixture",
        parameters,
    )];
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire
        .encode(request, Mode::Unary)
        .map_err(ProviderError::from)?;
    Ok(crate::test_utils::json_body(&encoded.request))
}

#[test]
fn test_vec_schema_conversion() {
    let schema_with_ref = json!({
        "type": "array",
        "items": {
            "$ref": "#/$defs/Person"
        },
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": ["string", "null"],
                        "description": "The person's first name, if provided (null otherwise)"
                    },
                    "last_name": {
                        "type": ["string", "null"],
                        "description": "The person's last name, if provided (null otherwise)"
                    },
                    "job": {
                        "type": ["string", "null"],
                        "description": "The person's job, if provided (null otherwise)"
                    }
                },
                "required": []
            }
        }
    });

    let result = schema::parameters(schema_with_ref).map(|schema| schema.expect("schema"));

    match result {
        Ok(schema) => {
            assert_eq!(schema["type"].as_str(), Some("array"));

            if let Some(items) = schema.get("items") {
                println!("item types: {}", items["type"]);

                assert_ne!(
                    items["type"].as_str(),
                    Some(""),
                    "Items type should not be empty string!"
                );
                assert_eq!(
                    items["type"].as_str(),
                    Some("object"),
                    "Items should be object type"
                );
            } else {
                panic!("Schema should have items field for array type");
            }
        }
        Err(e) => println!("Schema conversion failed: {e:?}"),
    }
}

#[test]
fn test_object_schema() {
    let simple_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            }
        }
    });

    let schema = schema::parameters(simple_schema)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema["type"].as_str(), Some("object"));
    assert!(schema.get("properties").is_some());
}

#[test]
fn test_array_with_inline_items() {
    let inline_schema = json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            }
        }
    });

    let schema = schema::parameters(inline_schema)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema["type"].as_str(), Some("array"));

    if let Some(items) = schema.get("items") {
        assert_eq!(items["type"].as_str(), Some("object"));
        assert!(items.get("properties").is_some());
    } else {
        panic!("Schema should have items field");
    }
}
#[test]
fn test_flattened_schema() {
    let ref_schema = json!({
        "type": "array",
        "items": {
            "$ref": "#/$defs/Person"
        },
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                }
            }
        }
    });

    let schema = schema::parameters(ref_schema)
        .expect("schema conversion")
        .expect("schema");

    assert_eq!(schema["type"].as_str(), Some("array"));

    if let Some(items) = schema.get("items") {
        println!("Flattened items type: '{}'", items["type"]);

        assert_eq!(items["type"].as_str(), Some("object"));
        assert!(items.get("properties").is_some());
    }
}

#[test]
fn test_array_without_items_gets_default() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "service_ids": {
                "type": "array",
                "description": "A list of service IDs"
            }
        }
    });

    let schema = schema::parameters(schema_json)
        .expect("schema conversion")
        .expect("schema");
    let props = schema.get("properties").expect("properties");
    let service_ids = props.get("service_ids").expect("service_ids");
    assert_eq!(service_ids["type"].as_str(), Some("array"));
    let items = service_ids
        .get("items")
        .expect("array schema missing items should get a default");
    assert_eq!(items["type"].as_str(), Some("string"));
}

#[test]
fn test_tool_parameters_to_schema_maps_no_arg_tool_to_none() {
    let schema =
        schema::parameters(json!({"type": "object", "properties": {}})).expect("schema conversion");

    assert!(schema.is_none());
}

#[test]
fn test_tool_parameters_to_schema_resolves_defs_ref() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "destination": { "$ref": "#/$defs/Destination" }
        },
        "required": ["destination"],
        "$defs": {
            "Destination": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }
        }
    });

    let schema = schema::parameters(schema_json)
        .expect("schema conversion")
        .expect("schema");
    let props = schema.get("properties").expect("properties");
    let destination = props.get("destination").expect("destination prop");

    assert_eq!(destination["type"].as_str(), Some("object"));
    assert_eq!(destination.get("required"), Some(&json!(["city"])));
}

#[test]
fn test_tool_parameters_to_schema_handles_nullable_type_arrays() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "nickname": { "type": ["null", "string"] }
        }
    });

    let schema = schema::parameters(schema_json)
        .expect("schema conversion")
        .expect("schema");
    let props = schema.get("properties").expect("properties");
    let nickname = props.get("nickname").expect("nickname prop");

    assert_eq!(nickname["type"].as_str(), Some("string"));
    assert_eq!(nickname["nullable"].as_bool(), Some(true));
}

#[test]
fn schema_request_omits_only_null_and_the_exact_empty_object() {
    for parameters in [Value::Null, json!({"type":"object","properties":{}})] {
        let body = encoded_schema_request(parameters).expect("no-argument request");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture"
            }))
        );
    }
    let body = encoded_schema_request(json!({
        "type":"object", "properties":{}, "required":[]
    }))
    .expect("nonempty schema metadata");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parameters":{"type":"object","properties":{},"required":[]}
        }))
    );
}

#[test]
fn schema_request_expands_definitions_and_removes_reference_sections() {
    for definitions in ["$defs", "definitions"] {
        let body = encoded_schema_request(json!({
            "type":"object",
            "properties":{"destination":{"$ref":format!("#/{definitions}/Destination")}},
            "required":["destination"],
            (definitions):{
                "Destination":{
                    "type":"object", "properties":{"city":{"type":["string","null"]}},
                    "required":["city"]
                }
            }
        }))
        .expect("acyclic references");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parameters":{
                    "type":"object",
                    "properties":{"destination":{
                        "type":"object", "properties":{"city":{"type":"string","nullable":true}},
                        "required":["city"]
                    }},
                    "required":["destination"]
                }
            }))
        );
    }
}

#[test]
fn schema_request_sorts_properties_and_carries_every_documented_keyword() {
    let parameters: Value = serde_json::from_str(
        r#"{
        "type":"object", "title":"Booking", "description":"A booking",
        "default":{}, "example":{"zeta":["a"]},
        "minProperties":1, "maxProperties":4, "propertyOrdering":["zeta","alpha","count"],
        "properties":{
            "zeta":{
                "type":"array","minItems":1,"maxItems":3,"default":["a"],
                "items":{"type":"string","format":"date","minLength":2,"maxLength":8,"pattern":"^[a-z]+$"}
            },
            "alpha":{"type":["string","null"],"enum":["a","b"],"example":"a"},
            "count":{"type":"integer","format":"int32","minimum":0,"maximum":10.5,"nullable":true}
        },
        "required":["zeta","alpha"]
    }"#,
    )
    .expect("literal schema field order");
    let body = encoded_schema_request(parameters).expect("projected schema request");
    let schema = body
        .pointer("/tools/0/functionDeclarations/0/parameters")
        .expect("parameters");
    assert_eq!(
        schema,
        &json!({
            "type":"object", "title":"Booking", "description":"A booking",
            "properties":{
                "alpha":{"type":"string","nullable":true,"enum":["a","b"],"example":"a"},
                "count":{
                    "type":"integer","format":"int32","nullable":true,"minimum":0,"maximum":10.5
                },
                "zeta":{
                    "type":"array","maxItems":3,"minItems":1,
                    "items":{
                        "type":"string","format":"date","minLength":2,"maxLength":8,
                        "pattern":"^[a-z]+$"
                    },
                    "default":["a"]
                }
            },
            "required":["zeta","alpha"],
            "propertyOrdering":["zeta","alpha","count"],
            "minProperties":1, "maxProperties":4,
            "example":{"zeta":["a"]}, "default":{}
        })
    );
    assert_eq!(
        schema["properties"]
            .as_object()
            .expect("properties")
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["alpha", "count", "zeta"]
    );
    assert!(
        body.pointer("/tools/0/functionDeclarations/0/parametersJsonSchema")
            .is_none()
    );
}

#[test]
fn schema_request_keeps_every_any_of_branch() {
    let body = encoded_schema_request(json!({
        "type":"object",
        "properties":{"choice":{"description":"A choice","anyOf":[
            {"type":"null"},
            {"type":"string","enum":["a","b"]},
            {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
            {"anyOf":[{"type":"integer","minimum":1},{"type":"boolean"}]}
        ]}}
    }))
    .expect("union request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0/parameters/properties/choice"),
        Some(&json!({
            "description":"A choice",
            "anyOf":[
                {"type":"null"},
                {"type":"string","enum":["a","b"]},
                {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
                {"anyOf":[{"type":"integer","minimum":1},{"type":"boolean"}]}
            ]
        }))
    );
}

#[test]
fn schema_request_drops_exactly_the_named_losses() {
    let body = encoded_schema_request(json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://example.com/args",
        "$comment":"root comment",
        "title":"Args",
        "type":"object",
        "additionalProperties":false,
        "properties":{"inner":{
            "$comment":"inner comment", "$id":"#inner",
            "type":"object", "additionalProperties":false,
            "properties":{"q":{"type":"string"}}
        }}
    }))
    .expect("named losses are accepted");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0/parameters"),
        Some(&json!({
            "type":"object", "title":"Args",
            "properties":{"inner":{"type":"object","properties":{"q":{"type":"string"}}}}
        }))
    );
}

#[test]
fn schema_request_carries_reference_siblings_and_null_enum_values_faithfully() {
    let body = encoded_schema_request(json!({
        "type":"object",
        "properties":{
            "destination":{"$ref":"#/$defs/Place","description":"Where to go"},
            "unit":{"type":["string","null"],"enum":["c",null,"f"]}
        },
        "$defs":{"Place":{
            "type":"object","description":"A place",
            "properties":{"city":{"type":"string"}}
        }}
    }))
    .expect("faithful reference siblings and null enum value");
    // The reference site's own description documents this use of the
    // definition, so it takes the place of the definition's.
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0/parameters/properties"),
        Some(&json!({
            "destination":{
                "type":"object","description":"Where to go",
                "properties":{"city":{"type":"string"}}
            },
            "unit":{"type":"string","nullable":true,"enum":["c","f"]}
        }))
    );
}

/// These controls inspect caller-supplied instance data at the encoded
/// request boundary; provider replies cannot establish its preservation.
#[test]
fn schema_request_preserves_literal_references_in_annotations() {
    for reference in ["#/$defs/T", "instance.json", "#/instance/value"] {
        let literal = json!({
            "$ref":reference,
            "const":{"$ref":"nested.json"},
            "oneOf":[{"$ref":"another.json"}],
            "enum":[0,1]
        });
        let body = encoded_schema_request(json!({
            "$defs":{"T":{"type":"string"}},
            "type":"object",
            "default":literal,
            "example":[literal]
        }))
        .expect("literal annotation references are carried");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parameters":{
                    "type":"object", "default":literal, "example":[literal]
                }
            }))
        );
    }
}

/// Reference-site annotations replace the definition's annotations in the
/// emitted request, independently of any recorded provider response.
#[test]
fn schema_request_replaces_reference_annotations_without_resolving_their_data() {
    let literal = json!({"$ref":"instance.json","nested":[{"$ref":"#/$defs/T"}]});
    let body = encoded_schema_request(json!({
        "$defs":{"T":{
            "type":"object", "title":"Definition", "description":"Definition description",
            "default":{"$ref":"definition-default.json"},
            "example":{"$ref":"definition-example.json"}
        }},
        "type":"object",
        "properties":{"value":{
            "$ref":"#/$defs/T", "title":"Site", "description":"Site description",
            "default":literal, "example":[literal]
        }}
    }))
    .expect("site annotations replace definition annotations");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0/parameters"),
        Some(&json!({
            "type":"object",
            "properties":{"value":{
                "type":"object", "title":"Site", "description":"Site description",
                "default":literal, "example":[literal]
            }}
        }))
    );
}

/// Property names are caller data; request encoding must still resolve
/// their schemas when the names also happen to be schema keywords.
#[test]
fn schema_request_resolves_properties_named_after_schema_keywords() {
    let names = [
        "$ref",
        "default",
        "example",
        "const",
        "enum",
        "oneOf",
        "properties",
    ];
    let properties: serde_json::Map<String, Value> = names
        .iter()
        .map(|name| ((*name).to_owned(), json!({"$ref":"#/$defs/T"})))
        .collect();
    let projected: serde_json::Map<String, Value> = names
        .iter()
        .map(|name| ((*name).to_owned(), json!({"type":"string"})))
        .collect();
    let body = encoded_schema_request(json!({
        "$defs":{"T":{"type":"string"}},
        "type":"object", "properties":properties
    }))
    .expect("property schemas are resolved regardless of their names");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parameters":{"type":"object", "properties":projected}
        }))
    );
}

#[test]
fn schema_constructs_outside_the_subset_are_refused_with_their_path() {
    let place = json!({"Place":{"type":"object","description":"A place"}});
    for (parameters, path, reason) in [
        (
            json!({"type":"object","properties":{"invalid":17}}),
            "/properties/invalid",
            "must be a schema object",
        ),
        (
            json!({"type":"object","properties":{"alpha":{"type":"string","enum":["a",1,"b"]}}}),
            "/properties/alpha/enum/1",
            "Gemini enum values are strings",
        ),
        (
            json!({"type":"string","enum":["a",true]}),
            "/enum/1",
            "Gemini enum values are strings",
        ),
        (
            json!({"type":"string","enum":["a",null]}),
            "/enum/1",
            "a null enum value needs a type that admits null",
        ),
        (
            json!({"type":["string","null"],"enum":[null]}),
            "/enum",
            "Gemini's Schema cannot express an enum of only null",
        ),
        (
            json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q",false]}),
            "/required/1",
            "must be a string",
        ),
        (
            json!({"allOf":[{"type":"string"}]}),
            "/allOf",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
        (
            json!({"type":"integer","exclusiveMinimum":0}),
            "/exclusiveMinimum",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
        (
            json!({"type":"array","items":{"type":"string"},"uniqueItems":true}),
            "/uniqueItems",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
        (
            json!({"type":"string","examples":["a"]}),
            "/examples",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
        (
            json!({"type":"array","max_items":3}),
            "/max_items",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
        (
            json!({"type":"object","additionalProperties":true}),
            "/additionalProperties",
            "only `false` is accepted, as a named loss; Gemini's Schema has no field for it",
        ),
        (
            json!({"type":"object","additionalProperties":{"type":"string"}}),
            "/additionalProperties",
            "only `false` is accepted, as a named loss; Gemini's Schema has no field for it",
        ),
        (
            json!({"type":"object","properties":{"x":{"type":["string","integer"]}}}),
            "/properties/x/type",
            "names more than one type besides null, and a Gemini schema has one type",
        ),
        (
            json!({"type":"date"}),
            "/type",
            "\"date\" is not a Gemini schema type",
        ),
        (
            json!({"type":"string","title":7}),
            "/title",
            "must be a string",
        ),
        (
            json!({"type":"string","minLength":1.5}),
            "/minLength",
            "must be a non-negative integer",
        ),
        (
            json!({"type":"array","items":{"type":"string"},"maxItems":0}),
            "/maxItems",
            "0 is indistinguishable from an unset bound in Gemini's Schema",
        ),
        (
            json!({"type":"number","maximum":"10"}),
            "/maximum",
            "must be a number",
        ),
        (
            json!({"type":"array","items":[{"type":"string"}]}),
            "/items",
            "must be a schema object",
        ),
        (
            json!({"anyOf":[{"type":"string"},true]}),
            "/anyOf/1",
            "must be a schema object",
        ),
        (
            json!({"type":"object","properties":{"x":{"$ref":"#/$defs/Missing"}}}),
            "/properties/x/$ref",
            "the reference cannot be resolved",
        ),
        (
            json!({
                "type":"object",
                "properties":{"x":{"$ref":"#/$defs/Place","type":"string"}},
                "$defs":place
            }),
            "/properties/x/type",
            "differs from the same keyword of #/$defs/Place",
        ),
    ] {
        let error = encoded_schema_request(parameters.clone())
            .expect_err("a construct outside the subset is refused");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        let diagnostic = format!(
            "Tool 'schema_tool' could not be converted to a schema: schema path {path}: {reason}"
        );
        assert_eq!(
            std::error::Error::source(&error)
                .expect("request reason")
                .to_string(),
            diagnostic,
            "{parameters}"
        );
    }
}

/// `oneOf` and `const`, the shapes schemars gives data enums, travel whole as
/// raw JSON Schema for their own declaration only, every variant and value
/// intact; a plain sibling declaration stays on `parameters`
/// (mec-rig-bodies-applied-ack-1).
#[test]
fn one_of_and_const_declarations_go_raw_and_plain_siblings_stay_projected() {
    use crate::wire::Operation;
    let one_of = json!({
        "type":"object",
        "properties":{"shape":{"oneOf":[
            {"type":"object","properties":{"kind":{"const":"circle"},"radius":{"type":"number"}},
             "required":["kind","radius"]},
            {"type":"object","properties":{"kind":{"const":"square"},"side":{"type":"number"}},
             "required":["kind","side"]}
        ]}},
        "required":["shape"]
    });
    let constant = json!({"type":"object","properties":{"unit":{"const":"celsius"}}});
    let plain = json!({"type":"object","properties":{"q":{"type":"string","title":"Query"}}});
    let wire = wire("gemini-2.5-flash");
    let mut request = CompletionRequest::new("hello");
    request.tools = [
        ("shape_tool", &one_of),
        ("unit_tool", &constant),
        ("plain_tool", &plain),
    ]
    .into_iter()
    .map(|(name, parameters)| {
        crate::completion::ToolDefinition::new(
            message::ToolName::new(name).expect("tool name"),
            "schema fixture",
            parameters.clone(),
        )
    })
    .collect();
    let request = Completion::prepare(request, &wire.describe()).expect("prepared");
    let encoded = wire.encode(request, Mode::Unary).expect("encoded");
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations"),
        Some(&json!([
            {"name":"shape_tool","description":"schema fixture","parametersJsonSchema":one_of},
            {"name":"unit_tool","description":"schema fixture","parametersJsonSchema":constant},
            {"name":"plain_tool","description":"schema fixture","parameters":{
                "type":"object","properties":{"q":{"type":"string","title":"Query"}}
            }}
        ]))
    );
}

/// Route selection and exact schema preservation are request properties,
/// so these controls encode through the public wire without provider calls.
#[test]
fn reference_sibling_one_of_and_const_keep_the_exact_raw_declaration() {
    let literal = json!({"$ref":"literal.json","nested":[{"$ref":"#/$defs/T"}]});
    for (definition, sibling) in [
        (json!({"type":"string"}), json!({"const":"a"})),
        (
            json!({"type":"string"}),
            json!({"oneOf":[{"type":"string","enum":["a"]},{"type":"string","enum":["b"]}]}),
        ),
        (
            json!({"type":"object"}),
            json!({"const":literal,"enum":[literal]}),
        ),
    ] {
        let mut choice = sibling.as_object().expect("schema sibling").clone();
        choice.insert("$ref".to_owned(), json!("#/$defs/T"));
        let parameters = json!({
            "$defs":{"T":definition},
            "type":"object", "properties":{"choice":choice}
        });
        let body = encoded_schema_request(parameters.clone()).expect("raw reference sibling");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parametersJsonSchema":parameters
            }))
        );
    }
}

/// Caller schemas under `additionalProperties` need the same exact raw
/// carrier as property schemas; this is an outbound encoding control.
#[test]
fn additional_properties_one_of_and_const_keep_the_exact_raw_declaration() {
    for schema in [
        json!({"const":"a"}),
        json!({"oneOf":[{"type":"string"},{"type":"integer"}]}),
    ] {
        let parameters = json!({"type":"object", "additionalProperties":schema});
        let body = encoded_schema_request(parameters.clone()).expect("raw additional properties");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parametersJsonSchema":parameters
            }))
        );
    }
}

/// Schema positions already visited for references must also select the
/// raw carrier. Exact outbound bodies establish this without provider calls.
#[test]
fn raw_schema_route_follows_the_existing_schema_positions() {
    let literal = json!({"$ref":"instance.json","oneOf":[{"const":"literal"}]});
    let definition = json!({
        "type":"object", "const":literal, "enum":[literal],
        "default":literal, "example":[literal]
    });
    for (keyword, value) in [
        ("properties", json!({"default":{"$ref":"#/$defs/T"}})),
        ("patternProperties", json!({"const":{"$ref":"#/$defs/T"}})),
        ("dependentSchemas", json!({"example":{"$ref":"#/$defs/T"}})),
        ("dependencies", json!({"enum":{"$ref":"#/$defs/T"}})),
        ("items", json!({"$ref":"#/$defs/T"})),
        ("items", json!([{"$ref":"#/$defs/T"}])),
        ("additionalItems", json!({"$ref":"#/$defs/T"})),
        ("additionalProperties", json!({"$ref":"#/$defs/T"})),
        ("contains", json!({"$ref":"#/$defs/T"})),
        ("propertyNames", json!({"$ref":"#/$defs/T"})),
        ("unevaluatedItems", json!({"$ref":"#/$defs/T"})),
        ("unevaluatedProperties", json!({"$ref":"#/$defs/T"})),
        ("not", json!({"$ref":"#/$defs/T"})),
        ("if", json!({"$ref":"#/$defs/T"})),
        ("then", json!({"$ref":"#/$defs/T"})),
        ("else", json!({"$ref":"#/$defs/T"})),
        ("contentSchema", json!({"$ref":"#/$defs/T"})),
        ("anyOf", json!([{"$ref":"#/$defs/T"}])),
        ("oneOf", json!([{"$ref":"#/$defs/T"}])),
        ("allOf", json!([{"$ref":"#/$defs/T"}])),
        ("prefixItems", json!([{"$ref":"#/$defs/T"}])),
    ] {
        let parameters = json!({"$defs":{"T":definition}, (keyword):value});
        let body = encoded_schema_request(parameters.clone()).expect("raw schema position");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parametersJsonSchema":parameters
            })),
            "{keyword}"
        );
    }
}

/// Raw detection still checks every visited schema. These request failures
/// pin reference validation and location reporting without provider calls.
#[test]
fn raw_schema_positions_keep_reference_and_schema_error_paths() {
    for (parameters, path, reason) in [
        (
            json!({"const":{}, "additionalProperties":{"$defs":{},"$ref":"#/$defs/Missing"}}),
            "/additionalProperties/$ref",
            "Reference not found: #/$defs/Missing",
        ),
        (
            json!({"const":{}, "patternProperties":{"a/b~c":{"$defs":[]}}}),
            "/patternProperties/a~1b~0c/$defs",
            "$defs must be an object",
        ),
        (
            json!({
                "$defs":{"T":{"type":"string"}},
                "additionalProperties":{"$ref":"#/$defs/T","const":"a","type":"integer"}
            }),
            "/additionalProperties/type",
            "differs from the same keyword of #/$defs/T",
        ),
        (
            json!({"oneOf":[{"type":"object"}], "unevaluatedProperties":{"$defs":[]}}),
            "/unevaluatedProperties/$defs",
            "$defs must be an object",
        ),
    ] {
        let error = encoded_schema_request(parameters).expect_err("visited schema is checked");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            format!(
                "RequestError: Tool 'schema_tool' could not be converted to a schema: schema path {path}: {reason}"
            )
        );
    }
}

/// A raw carrier cannot make conflicting reference validation keywords
/// faithful. Their refusal and path are observable before any provider call.
#[test]
fn raw_schema_route_still_refuses_conflicting_reference_siblings() {
    for (definition, sibling, keyword) in [
        (
            json!({"type":"string"}),
            json!({"const":"a","type":"integer"}),
            "type",
        ),
        (json!({"const":"a"}), json!({"const":"b"}), "const"),
        (
            json!({"oneOf":[{"type":"string"}]}),
            json!({"oneOf":[{"type":"integer"}]}),
            "oneOf",
        ),
    ] {
        let mut choice = sibling.as_object().expect("schema sibling").clone();
        choice.insert("$ref".to_owned(), json!("#/$defs/T"));
        let error = encoded_schema_request(json!({
            "$defs":{"T":definition},
            "type":"object", "properties":{"choice":choice}
        }))
        .expect_err("conflicting validation siblings are refused");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            format!(
                "RequestError: Tool 'schema_tool' could not be converted to a schema: schema path /properties/choice/{keyword}: differs from the same keyword of #/$defs/T"
            )
        );
    }
}

#[test]
fn numeric_enum_declaration_goes_raw_even_with_constructs_outside_the_subset() {
    let parameters = json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "type":"object",
        "properties":{
            "slot":{"type":"integer","enum":[0,1,2],"exclusiveMinimum":-1},
            "mode":{"oneOf":[{"const":"a"},{"const":"b"}]},
            "unit":{"type":"string","enum":["c","f"],"examples":["c"]}
        },
        "required":["slot"],
        "additionalProperties":true
    });
    let body = encoded_schema_request(parameters.clone()).expect("numeric enum request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parametersJsonSchema":parameters
        }))
    );
}

#[test]
fn schema_request_errors_keep_the_tool_name_and_reference_diagnostic() {
    for (parameters, reason) in [
        (json!([]), "Expected a JSON object for Schema"),
        (json!({"$defs":[]}), "$defs must be an object"),
        (
            json!({"$defs":{},"$ref":"#/$defs/Missing"}),
            "Reference not found: #/$defs/Missing",
        ),
        (
            json!({"$defs":{},"$ref":"#/other/Missing"}),
            "Unsupported reference format: #/other/Missing",
        ),
        (
            json!({"$defs":{},"$ref":"schema.json"}),
            "Only fragment references (#/...) are supported: schema.json",
        ),
    ] {
        let error =
            encoded_schema_request(parameters).expect_err("typed declaration refuses the request");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        let diagnostic = format!("Tool 'schema_tool' could not be converted to a schema: {reason}");
        assert_eq!(
            std::error::Error::source(&error)
                .expect("request reason")
                .to_string(),
            diagnostic
        );
        assert_eq!(error.to_string(), format!("RequestError: {diagnostic}"));
    }
}

#[test]
fn test_txt_document_conversion_to_text_part() {
    use crate::message::{DocumentMediaType, UserContent};
    let doc = UserContent::document_text(
        "Note: test.md\nPath: /test.md\nContent: Hello World!",
        Some(DocumentMediaType::TXT),
    );
    let wire = wire("gemini-2.5-flash");
    let converted = contents(
        vec![message::Message::User { content: vec![doc] }],
        &wire,
        "gemini-2.5-flash",
    )
    .expect("convert document");
    if let Some(text) = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/text"))
        .and_then(Value::as_str)
    {
        assert!(text.contains("Note: test.md"));
        assert!(text.contains("Hello World!"));
    } else {
        panic!("Expected text part for TXT document, got: {converted:?}");
    }
}

#[test]
fn test_tool_result_with_image_content() {
    use crate::message::{CallId, ImageMediaType, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::from_wire("call-123"),
            message::ToolName::new("test_tool").expect("tool name"),
            vec![
                ToolResultContent::text(r#"{"status": "success"}"#),
                ToolResultContent::image_base64(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    Some(ImageMediaType::PNG),
                    None,
                ),
            ],
        )],
    };
    let wire = wire("gemini-3-flash-preview");
    let converted = contents(vec![msg], &wire, "gemini-3-flash-preview")
        .expect("Should convert to Gemini Content");
    let content = converted.first().expect("content");
    assert_eq!(content["role"].as_str(), Some("user"));
    let parts = content["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 1);
    if let Some(function_response) = parts.first().and_then(|part| part.get("functionResponse")) {
        assert_eq!(function_response["name"].as_str(), Some("test_tool"));
        assert_eq!(function_response["id"].as_str(), Some("call-123"));
        assert!(function_response.get("response").is_some());
        assert_eq!(
            function_response.get("response"),
            Some(&json!({"result":r#"{"status": "success"}"#}))
        );
        assert!(function_response.get("parts").is_some());
        let parts = function_response["parts"].as_array().expect("image parts");
        assert_eq!(parts.len(), 1);
        assert!(parts[0].get("inlineData").is_some());
        let inline_data = parts[0].get("inlineData").expect("inline data");
        assert_eq!(inline_data["mimeType"].as_str(), Some("image/png"));
        assert!(
            !inline_data["data"]
                .as_str()
                .expect("base64 image")
                .is_empty()
        );
        assert_eq!(inline_data.get("displayName"), None);
    } else {
        panic!("Expected FunctionResponse part");
    }
}

#[test]
fn mixed_inline_images_and_text_keep_text_response_and_ordered_parts() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::Local(LocalCallId::new()),
            message::ToolName::new("ordered_tool").expect("tool name"),
            vec![
                ToolResultContent::image_base64("first-image", Some(ImageMediaType::PNG), None),
                ToolResultContent::text("between-images"),
                ToolResultContent::image_base64("second-image", Some(ImageMediaType::JPEG), None),
            ],
        )],
    };
    let wire = wire("gemini-2.5-flash");
    let converted =
        contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
    let response = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/functionResponse"))
        .expect("expected a function response");
    assert_eq!(
        response.get("response"),
        Some(&json!({"result":"between-images"}))
    );
    let parts = response["parts"]
        .as_array()
        .expect("images should be inline parts");
    assert_eq!(parts.len(), 2);
    let first = parts[0].get("inlineData").expect("first inline image");
    assert_eq!(first["mimeType"].as_str(), Some("image/png"));
    assert_eq!(first["data"].as_str(), Some("first-image"));
    assert_eq!(first.get("displayName"), None);
    let second = parts[1].get("inlineData").expect("second inline image");
    assert_eq!(second["mimeType"].as_str(), Some("image/jpeg"));
    assert_eq!(second["data"].as_str(), Some("second-image"));
    assert_eq!(second.get("displayName"), None);
}

#[test]
fn mixed_inline_image_and_json_keep_structured_value_and_media_part() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::Local(LocalCallId::new()),
            message::ToolName::new("ordered_tool").expect("tool name"),
            vec![
                ToolResultContent::json(json!({"status":"ok"})),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        )],
    };
    let wire = wire("gemini-2.5-flash");
    let converted =
        contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
    let response = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/functionResponse"))
        .expect("expected a function response");
    assert_eq!(
        response.get("response"),
        Some(&json!({"result":{"status":"ok"}}))
    );
    let parts = response["parts"]
        .as_array()
        .expect("image should be an inline part");
    assert_eq!(parts.len(), 1);
    let inline_data = parts[0].get("inlineData").expect("inline image data");
    assert_eq!(inline_data["data"].as_str(), Some("image-data"));
    assert_eq!(inline_data.get("displayName"), None);
}

#[test]
fn mixed_url_image_and_response_value_is_rejected() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::Local(LocalCallId::new()),
            message::ToolName::new("url_tool").expect("tool name"),
            vec![
                ToolResultContent::image_url(
                    "https://example.com/image.png",
                    Some(ImageMediaType::PNG),
                    None,
                ),
                ToolResultContent::text("after-image"),
            ],
        )],
    };
    let wire = wire("gemini-3-flash-preview");
    let error = contents(vec![msg], &wire, "gemini-3-flash-preview")
        .expect_err("URL-backed tool result images should be rejected");
    assert!(
        error
            .to_string()
            .contains("URL-backed images are not supported"),
        "unexpected error: {error}"
    );
}

#[test]
fn tool_result_rejects_unsupported_image_media_types() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    for media_type in [
        ImageMediaType::GIF,
        ImageMediaType::HEIC,
        ImageMediaType::HEIF,
        ImageMediaType::SVG,
    ] {
        let msg = message::Message::User {
            content: vec![message::UserContent::tool_result(
                CallId::Local(LocalCallId::new()),
                message::ToolName::new("image_tool").expect("tool name"),
                vec![ToolResultContent::image_base64(
                    "image-data",
                    Some(media_type),
                    None,
                )],
            )],
        };
        let wire = wire("gemini-3-flash-preview");
        let error = contents(vec![msg], &wire, "gemini-3-flash-preview")
            .expect_err("unsupported tool result image type should be rejected");
        assert!(
            error
                .to_string()
                .contains("supported types are JPEG, PNG, and WEBP"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn structured_json_refs_remain_literal_with_unreferenced_image_parts() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::Local(LocalCallId::new()),
            message::ToolName::new("collision_tool").expect("tool name"),
            vec![
                ToolResultContent::json(json!({"literal":{"$ref":"tool_result_image_0"}})),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        )],
    };
    let wire = wire("gemini-2.5-flash");
    let converted =
        contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
    let response = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/functionResponse"))
        .expect("expected a function response");
    assert_eq!(
        response.get("response"),
        Some(&json!({"result":{"literal":{"$ref":"tool_result_image_0"}}}))
    );
    assert_eq!(
        response
            .get("parts")
            .and_then(Value::as_array)
            .and_then(|parts| {
                parts
                    .first()
                    .and_then(|part| part.get("inlineData"))
                    .and_then(|data| data.get("displayName"))
            }),
        None
    );
}

#[test]
fn tool_result_literal_text_and_structured_json_remain_distinct() {
    use crate::message::{CallId, LocalCallId, ToolResultContent};
    let cases = [
        (
            ToolResultContent::text(r#"{"status":"ok"}"#),
            json!({"result":"{\"status\":\"ok\"}"}),
        ),
        (
            ToolResultContent::json(json!({"status":"ok"})),
            json!({"result":{"status":"ok"}}),
        ),
    ];
    for (tool_content, expected) in cases {
        let msg = message::Message::User {
            content: vec![message::UserContent::tool_result(
                CallId::Local(LocalCallId::new()),
                message::ToolName::new("test_tool").expect("tool name"),
                vec![tool_content],
            )],
        };
        let wire = wire("gemini-2.5-flash");
        let converted =
            contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
        let response = converted
            .first()
            .and_then(|content| content.pointer("/parts/0/functionResponse"))
            .expect("expected a function response");
        assert_eq!(response.get("response"), Some(&expected));
    }
}

/// A consumer echoing a minted `ToolCall::id` through
/// `tool_result()` must not put that handle on Gemini's wire: the
/// paired functionCall omitted its id (the provider issued none), and
/// an asymmetric functionCall/functionResponse id pair is rejected.
#[test]
fn echoed_minted_handle_never_reaches_the_function_response_id() {
    use crate::message::{CallId, LocalCallId, ToolCall, ToolFunction, ToolResultContent};
    let call = ToolCall::new(
        CallId::Local(LocalCallId::new()),
        ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({}),
        ),
    );
    let msg =
        message::Message::tool_results(vec![call.result(vec![ToolResultContent::text("out")])]);
    let wire = wire("gemini-2.5-flash");
    let converted =
        contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
    let response = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/functionResponse"))
        .expect("expected a function response");
    assert_eq!(response.get("id"), None);
}

/// A cross-provider ingested transcript (rig's inbound converters
/// stamp `name: ""` — Anthropic/OpenAI-chat/Cohere/Bedrock wires carry
/// no name) must reach Gemini with the name resolved from the paired
/// call: `functionResponse.name: ""` is INVALID_ARGUMENT.
#[test]
fn ingested_nameless_results_resolve_their_name_at_request_assembly() {
    use crate::message::{
        AnsweredToolCall, AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction,
        ToolName, ToolResult, ToolResultContent, UserContent,
    };
    use crate::wire::{Mode, Wire as _};
    let call = ToolCall::from_wire(
        "toolu_abc",
        ToolFunction::new(
            ToolName::new("get_weather").expect("original tool name"),
            json!({"city":"Paris"}),
        ),
    );
    let mut prefix = vec![
        Message::user("weather?"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
    ];
    let result = ToolResult::from_history(
        &prefix,
        call.id.clone(),
        AnsweredToolCall::Function,
        vec![ToolResultContent::text("sunny")],
        false,
    )
    .expect("unique preceding original call");
    prefix.push(Message::User {
        content: vec![UserContent::ToolResult(result)],
    });
    let mut request = CompletionRequest::new("weather?");
    request.chat_history = prefix;
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("original direct request should build");
    let body = crate::test_utils::json_body(&encoded.request);
    let response_names: Vec<_> = body["contents"]
        .as_array()
        .expect("contents")
        .iter()
        .flat_map(|content| content["parts"].as_array().expect("parts"))
        .filter_map(|part| {
            part.get("functionResponse")
                .map(|response| response["name"].as_str().expect("function-response name"))
        })
        .collect();
    assert_eq!(response_names, ["get_weather"]);
}

/// A wire-derived result keeps its provider-issued id on replay.
#[test]
fn wire_derived_tool_result_keeps_the_provider_id_on_the_wire() {
    use crate::message::{CallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::from_wire("gemini-issued-id"),
            message::ToolName::new("lookup").expect("tool name"),
            vec![ToolResultContent::text("out")],
        )],
    };
    let wire = wire("gemini-2.5-flash");
    let converted =
        contents(vec![msg], &wire, "gemini-2.5-flash").expect("tool result should convert");
    let response = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/functionResponse"))
        .expect("expected a function response");
    assert_eq!(response["id"].as_str(), Some("gemini-issued-id"));
}

#[test]
fn test_markdown_document_conversion_to_text_part() {
    use crate::message::{DocumentMediaType, UserContent};
    let doc = UserContent::document_text(
        "# Heading\n\n* List item",
        Some(DocumentMediaType::MARKDOWN),
    );
    let wire = wire("gemini-2.5-flash");
    let converted = contents(
        vec![message::Message::User { content: vec![doc] }],
        &wire,
        "gemini-2.5-flash",
    )
    .expect("convert document");
    if let Some(text) = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/text"))
        .and_then(Value::as_str)
    {
        assert_eq!(text, "# Heading\n\n* List item");
    } else {
        panic!("Expected text part for MARKDOWN document, got: {converted:?}");
    }
}

#[test]
fn test_markdown_url_document_conversion_to_file_data_part() {
    use crate::message::{DocumentMediaType, UserContent};
    let doc = UserContent::document_url(
        "https://generativelanguage.googleapis.com/v1beta/files/test-markdown",
        Some(DocumentMediaType::MARKDOWN),
    );
    let wire = wire("gemini-2.5-flash");
    let converted = contents(
        vec![message::Message::User { content: vec![doc] }],
        &wire,
        "gemini-2.5-flash",
    )
    .expect("convert document");
    if let Some(file_data) = converted
        .first()
        .and_then(|content| content.pointer("/parts/0/fileData"))
    {
        assert_eq!(
            file_data["fileUri"].as_str(),
            Some("https://generativelanguage.googleapis.com/v1beta/files/test-markdown")
        );
        assert_eq!(file_data["mimeType"].as_str(), Some("text/markdown"));
    } else {
        panic!("Expected file_data part for URL MARKDOWN document, got: {converted:?}");
    }
}

#[test]
fn test_user_image_url_renders_as_file_data() {
    // A URL-sourced user image is a Files API / Cloud Storage reference on
    // this wire (`fileData`), never fetched inline: an arbitrary HTTPS image
    // is not a Gemini capability, and the adapter does not convert it.
    use crate::message::{ImageMediaType, UserContent};
    let image = UserContent::image_url(
        "https://example.com/red_square.png",
        Some(ImageMediaType::PNG),
        None,
    );
    let wire = wire("gemini-2.5-flash");
    let converted = contents(
        vec![message::Message::User {
            content: vec![image],
        }],
        &wire,
        "gemini-2.5-flash",
    )
    .expect("convert image");
    match converted
        .first()
        .and_then(|content| content.pointer("/parts/0/fileData"))
    {
        Some(file_data) => {
            assert_eq!(
                file_data["fileUri"].as_str(),
                Some("https://example.com/red_square.png")
            );
            assert_eq!(file_data["mimeType"].as_str(), Some("image/png"));
        }
        other => panic!("Expected file_data part for a URL image, got: {other:?}"),
    }
}

#[test]
fn test_tool_result_with_url_image_is_rejected() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    let msg = message::Message::User {
        content: vec![message::UserContent::tool_result(
            CallId::Local(LocalCallId::new()),
            message::ToolName::new("screenshot_tool").expect("tool name"),
            vec![ToolResultContent::image_url(
                "https://example.com/image.png",
                Some(ImageMediaType::PNG),
                None,
            )],
        )],
    };
    let wire = wire("gemini-3-flash-preview");
    let error = contents(vec![msg], &wire, "gemini-3-flash-preview")
        .expect_err("URL-backed tool result images should be rejected");
    assert!(
        error
            .to_string()
            .contains("URL-backed images are not supported"),
        "unexpected error: {error}"
    );
}

#[test]
fn original_tool_result_images_refuse_before_model_media_adaptation() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    use crate::wire::Operation;
    for model in ["gemini-2.5-flash", "gemini-3-flash-preview"] {
        for (image, expected) in [
            (
                ToolResultContent::image_url(
                    "https://example.com/image.png",
                    Some(ImageMediaType::PNG),
                    None,
                ),
                "Gemini tool result images must use base64 inline data; URL-backed images are not supported",
            ),
            (
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::GIF), None),
                "Unsupported image media type GIF for Gemini tool results; supported types are JPEG, PNG, and WEBP",
            ),
            (
                ToolResultContent::image_base64("image-data", None, None),
                "Image media type is required for Gemini tool results",
            ),
            (
                ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::PNG), None),
                "Unsupported image source kind for tool results",
            ),
        ] {
            let mut request = CompletionRequest::new("question");
            request.chat_history.push(message::Message::User {
                content: vec![message::UserContent::tool_result(
                    CallId::Local(LocalCallId::new()),
                    message::ToolName::new("image_tool").expect("tool name"),
                    vec![ToolResultContent::text("before-image"), image],
                )],
            });
            let target = wire(model);
            let error = Completion::prepare(request, &target.describe())
                .expect_err("original result image refuses before movement or omission");
            assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
            let source = std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<message::MessageError>())
                .expect("typed message conversion source");
            assert!(
                matches!(source, message::MessageError::ConversionError(reason) if reason == expected)
            );
            assert_eq!(
                error.to_string(),
                format!("RequestError: Message conversion error: {expected}")
            );
        }
    }
}

#[test]
fn original_result_image_order_and_source_errors_survive_mixed_request_preflight() {
    use crate::message::{CallId, ImageMediaType, LocalCallId, ToolResultContent};
    use crate::wire::Operation;
    for model in ["gemini-2.5-flash", "gemini-3-flash-preview"] {
        for (images, expected) in [
            (
                vec![
                    ToolResultContent::image_url("https://example.com/image.png", None, None),
                    ToolResultContent::image_base64("later", Some(ImageMediaType::GIF), None),
                ],
                "Gemini tool result images must use base64 inline data; URL-backed images are not supported",
            ),
            (
                vec![ToolResultContent::image_raw(vec![1, 2, 3], None, None)],
                "Unsupported image source kind for tool results",
            ),
            (
                vec![
                    ToolResultContent::image_base64("first", Some(ImageMediaType::GIF), None),
                    ToolResultContent::image_url("https://example.com/image.png", None, None),
                ],
                "Unsupported image media type GIF for Gemini tool results; supported types are JPEG, PNG, and WEBP",
            ),
            (
                vec![
                    ToolResultContent::image_base64("first", Some(ImageMediaType::PNG), None),
                    ToolResultContent::image_url("https://example.com/image.png", None, None),
                ],
                "Gemini tool result images must use base64 inline data; URL-backed images are not supported",
            ),
        ] {
            let mut result = vec![ToolResultContent::text("before-image")];
            result.extend(images);
            result.push(ToolResultContent::json(json!({"after":"image"})));
            let mut request = CompletionRequest::new("question");
            request.chat_history.push(message::Message::User {
                content: vec![
                    message::UserContent::text("before-result"),
                    message::UserContent::tool_result(
                        CallId::Local(LocalCallId::new()),
                        message::ToolName::new("image_tool").expect("tool name"),
                        result,
                    ),
                    message::UserContent::text("after-result"),
                ],
            });
            let target = wire(model);
            let error = Completion::prepare(request, &target.describe())
                .expect_err("whole mixed request refuses");
            assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
            let source = std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<message::MessageError>())
                .expect("typed conversion source");
            assert!(
                matches!(source, message::MessageError::ConversionError(reason) if reason == expected)
            );
            assert_eq!(
                error.to_string(),
                format!("RequestError: Message conversion error: {expected}")
            );
        }
    }
}

#[test]
fn original_result_text_and_json_preflight_adds_no_function_or_custom_policy() {
    use crate::message::{AnsweredToolCall, CallId, LocalCallId, ToolResult, ToolResultContent};
    use crate::wire::Operation as _;
    for model in ["gemini-2.5-flash", "gemini-3-flash-preview"] {
        let target = wire(model);
        for answers in [AnsweredToolCall::Function, AnsweredToolCall::Custom] {
            let result = ToolResult {
                call: CallId::Local(LocalCallId::new()),
                name: message::ToolName::new("result_tool").expect("tool name"),
                answers,
                content: vec![
                    ToolResultContent::text(r#"{"literal":true}"#),
                    ToolResultContent::json(json!([1, 2])),
                ],
                is_error: true,
            };
            let request = CompletionRequest::from(vec![message::Message::User {
                content: vec![message::UserContent::ToolResult(result)],
            }]);
            target
                .check_documents(&request)
                .expect("nonimage content is outside this image-only preflight");
        }
        let request = CompletionRequest::new("ordinary text");
        let request =
            Completion::prepare(request, &target.describe()).expect("ordinary request unchanged");
        let encoded = target
            .encode(request, Mode::Unary)
            .expect("ordinary request encodes");
        let body = crate::test_utils::json_body(&encoded.request);
        assert_eq!(
            body["contents"],
            json!([{"role":"user","parts":[{"text":"ordinary text","thought":false}]}])
        );
    }
}

#[test]
fn test_create_request_body_with_documents() {
    use crate::completion::Document;
    use crate::message::Message;
    let mut documents_request = CompletionRequest::new("placeholder");
    documents_request.documents = vec![
        Document {
            id: "doc1".to_owned(),
            text: "Note: first.md\nContent: First note".to_owned(),
            additional_props: std::collections::HashMap::new(),
        },
        Document {
            id: "doc2".to_owned(),
            text: "Note: second.md\nContent: Second note".to_owned(),
            additional_props: std::collections::HashMap::new(),
        },
    ];
    let documents_message = documents_request
        .normalized_documents()
        .expect("documents message");
    let mut completion_request = CompletionRequest::new("What are my notes about?");
    completion_request.chat_history = vec![
        Message::system("You are a helpful assistant"),
        documents_message,
        Message::user("What are my notes about?"),
    ];
    let encoded = wire("gemini-2.5-flash")
        .encode(completion_request, Mode::Unary)
        .expect("request body");
    let request = crate::test_utils::json_body(&encoded.request);
    let contents = request["contents"].as_array().expect("request contents");
    assert_eq!(
        contents.len(),
        2,
        "Expected 2 contents (documents + user message)"
    );
    assert_eq!(contents[0]["role"].as_str(), Some("user"));
    let parts = contents[0]["parts"].as_array().expect("document parts");
    assert_eq!(parts.len(), 2, "Expected 2 document parts");
    for part in parts {
        if let Some(Value::String(text)) = part.get("text") {
            assert!(
                text.contains("Note:") && text.contains("Content:"),
                "Document should contain note metadata"
            );
        } else {
            panic!("Document parts should be text, not {part:?}");
        }
    }
    assert_eq!(contents[1]["role"].as_str(), Some("user"));
    if let Some(Value::String(text)) = contents[1].pointer("/parts/0/text") {
        assert_eq!(text, "What are my notes about?");
    } else {
        panic!("Expected user message to be text");
    }
}

#[test]
fn test_create_request_body_without_documents() {
    use crate::message::Message;
    let mut completion_request = CompletionRequest::new("Hello");
    completion_request.chat_history = vec![
        Message::system("You are a helpful assistant"),
        Message::user("Hello"),
    ];
    let encoded = wire("gemini-2.5-flash")
        .encode(completion_request, Mode::Unary)
        .expect("request body");
    let request = crate::test_utils::json_body(&encoded.request);
    let contents = request["contents"].as_array().expect("request contents");
    assert_eq!(contents.len(), 1, "Expected only user message");
    assert_eq!(contents[0]["role"].as_str(), Some("user"));
    if let Some(Value::String(text)) = contents[0].pointer("/parts/0/text") {
        assert_eq!(text, "Hello");
    } else {
        panic!("Expected user message to be text");
    }
}

/// A non-success reply is reported with the provider's own status and body
/// preserved: the envelope shape is Gemini's business, so nothing on the
/// path may narrow it by parsing before the caller sees it.
#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let error = crate::Model::new(
        wire(super::GEMINI_3_FLASH_PREVIEW),
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
    )
    .call(wire_request("hello"))
    .await
    .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn block_reasons_split_into_final_refusals_and_transient_blocks() {
    // `SAFETY`, `BLOCKLIST` and `PROHIBITED_CONTENT` judge the content and
    // are final. `OTHER` is Google's "blocked due to unknown reasons"; the
    // same prompt is answered on the next call, so it is retryable and it
    // is not a refusal. The classification is typed: the report's
    // `retryable` and `kind` say it, no caller reads the message. The block
    // arrives under a 200, which the driver keeps on the error for the
    // caller to see; a success status is no retry verdict, so the
    // decoder's own stands.
    for (reason, retryable) in [
        ("SAFETY", false),
        ("BLOCKLIST", false),
        ("PROHIBITED_CONTENT", false),
        ("OTHER", true),
        ("SOMETHING_NEW", true),
    ] {
        let error = fold_unary(
            "gemini-2.5-flash",
            json!({"promptFeedback": {"blockReason": reason}}).to_string(),
        )
        .await
        .expect_err(reason);
        assert_eq!(error.is_retryable(), retryable, "{reason}: {error:?}");
        let report = crate::error::ErrorReport::from(&error);
        assert_eq!(report.retryable, retryable, "{reason}: {report:?}");
        assert!(
            report.message.contains(&format!("block_reason={reason}")),
            "{reason}: {}",
            report.message
        );
        assert!(
            matches!(&error, ProviderError::ProviderResponse(response) if response.status == Some(http::StatusCode::OK) && response.refusal != retryable && response.code.as_deref() == Some(reason)),
            "{reason}: {error:?}"
        );
        assert_eq!(report.kind, crate::error::ErrorKind::ProviderResponse);
        assert_eq!(report.refusal, !retryable, "{reason}: {report:?}");
    }
}

// ── the GenerateContent wire ────────────────────────────────────────────
//
// Bodies below are pasted verbatim from committed cassettes, named at each
// constant. The point of the pairs is the property the wire model exists
// for: the unary reply and the streamed reply of the SAME turn, decoded by
// the SAME decoder, fold to the same answer.

use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Mode, Wire};
use futures::StreamExt;

/// `crates/rig-cassette/fixtures/cassettes/gemini/turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap.yaml`
const CEDAR_UNARY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"cedar"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":22,"promptTokensDetails":[{"modality":"TEXT","tokenCount":22}],"serviceTier":"standard","totalTokenCount":24}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap.yaml`
/// — the same turn, streamed. Gemini delivered it as one event.
const CEDAR_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"cedar"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":22,"promptTokensDetails":[{"modality":"TEXT","tokenCount":22}],"serviceTier":"standard","totalTokenCount":24}}"#,
    "\r\n\r\n",
);

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/blocking_keeps_a_trailing_thought_signature.yaml`
const SIGNED_UNARY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"289","thoughtSignature":"signature_REDACTED_1"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":59}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/streaming_twin_agrees_on_a_trailing_thought_signature.yaml`
/// — the same turn, streamed across two events, the signature riding a
/// trailing part that carries no `thought` flag.
const SIGNED_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"289"}],"role":"model"},"index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
    r#"data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"signature_REDACTED_1"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
);

fn wire_request(prompt: &str) -> CompletionRequest {
    CompletionRequest::new(prompt)
}

fn wire(model: &str) -> GenerateContent {
    crate::providers::gemini::GeminiConfig::new("test-key").completion(model)
}

/// What a folded response says, for comparing two transports.
fn folded(
    response: &crate::completion::CompletionResponse,
) -> (
    Vec<message::AssistantContent>,
    crate::completion::Usage,
    Option<crate::completion::FinishReason>,
    Option<String>,
) {
    (
        response.choice.to_vec(),
        response.usage,
        response.finish_reason(),
        response.model().map(str::to_owned),
    )
}

/// Fold one `generateContent` reply body through the bound wire, the way a
/// caller's `completion()` does — errors included.
async fn fold_unary(
    model: &str,
    body: impl Into<bytes::Bytes>,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    crate::Model::new(wire(model), RecordingHttpClient::new(body))
        .call(wire_request("probe"))
        .await
}

async fn unary(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    fold_unary(model, body)
        .await
        .expect("the recorded unary reply decodes")
}

async fn streamed(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    let mut stream = crate::Model::new(
        wire(model),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from_static(body.as_bytes()),
        },
    )
    .stream(wire_request("probe"))
    .expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("the recorded stream carries no in-band error");
    }
    stream
        .finish()
        .await
        .expect("the stream produced a terminal record")
}

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_answer() {
    let buffered = unary("gemini-2.5-flash", CEDAR_UNARY).await;
    let streamed = streamed("gemini-2.5-flash", CEDAR_STREAM).await;
    assert_eq!(folded(&buffered), folded(&streamed));
    // G141: the decoded Text retains its genuine part; the constructor has none.
    assert_eq!(
        buffered
            .choice
            .first()
            .map(message::AssistantContent::canonical),
        Some(message::AssistantContent::text("cedar"))
    );
    assert_eq!(
        buffered
            .choice
            .first()
            .and_then(message::AssistantContent::native_item),
        Some(&json!({"text": "cedar"}))
    );
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(buffered.usage.output_tokens, Some(2));
    assert_eq!(buffered.usage.total_tokens, Some(24));
}

/// Gemini hangs `thoughtSignature` on an answer part, and it must return
/// inside the part that carried it (Gemini's thought-signature rules: never
/// merge a signed part with an unsigned one). The unary reply signs the one
/// answer part; the streamed twin sends the text, then an empty signed part.
/// Each keeps its signature on its own text, and neither moves it onto
/// reasoning.
#[tokio::test]
async fn a_trailing_thought_signature_stays_on_the_part_that_carried_it() {
    use crate::message::{
        AssistantContent, AssistantMessage, CarriedPayload, CarriedPosition, CarriedText,
        CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, Message,
    };
    let buffered = unary("gemini-3-flash-preview", SIGNED_UNARY).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_STREAM).await;
    let signed = |text: &str| {
        AssistantContent::CarriedText(CarriedText::new(
            CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
                text,
                GeminiThoughtSignature::new("signature_REDACTED_1"),
            )),
            CarriedPayload::new(
                json!({"text": text, "thoughtSignature": "signature_REDACTED_1"}),
                Some("gcp.gemini".to_owned()),
            )
            .with_position(CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0,
            }),
        ))
    };
    assert_eq!(buffered.choice.to_vec(), vec![signed("289")]);
    assert_eq!(
        streamed.choice.to_vec(),
        vec![
            AssistantContent::text("289").with_native(json!({"text": "289"})),
            signed("")
        ]
    );
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());

    // Both replay the signature on the text part that carried it.
    for (response, parts) in [
        (&buffered, vec![("289", true)]),
        (&streamed, vec![("289", false), ("", true)]),
    ] {
        let wire = wire("gemini-3-flash-preview");
        let replayed_contents = contents(
            vec![Message::Assistant(AssistantMessage::new(
                response.choice.clone(),
            ))],
            &wire,
            "gemini-3-flash-preview",
        )
        .expect("the turn replays");
        let model_parts = replayed_contents[0]["parts"]
            .as_array()
            .expect("model parts");
        let replayed: Vec<(&str, bool)> = model_parts
            .iter()
            .map(|part| {
                (
                    part["text"].as_str().expect("an answer part"),
                    part["thoughtSignature"].as_str() == Some("signature_REDACTED_1"),
                )
            })
            .collect();
        assert_eq!(replayed, parts);
        assert!(
            model_parts
                .iter()
                .all(|part| part["thought"].as_bool() != Some(true))
        );
    }
}

/// One part carrying both non-empty text and its `thoughtSignature`, in a
/// single frame: both transports keep the signature on that text.
/// Recorded in the effect corpus
/// (`crates/rig-cassette/fixtures/effects/gemini_tool_call_turns.effects.json`).
const SIGNED_ONE_PART: &str = r#"{"candidates":[{"content":{"parts":[{"text":"done","thoughtSignature":"signature_REDACTED_1"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"thoughtsTokenCount":12,"totalTokenCount":27}}"#;

/// The same document as one SSE event: the streamed twin of [`SIGNED_ONE_PART`].
const SIGNED_ONE_PART_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"done","thoughtSignature":"signature_REDACTED_1"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"thoughtsTokenCount":12,"totalTokenCount":27}}"#,
    "\r\n\r\n",
);

#[tokio::test]
async fn a_signature_on_its_own_text_part_stays_on_that_text_on_both_transports() {
    use crate::message::{
        AssistantContent, CarriedPayload, CarriedPosition, CarriedText, CarriedTextContent,
        GeminiSignedAnswer, GeminiThoughtSignature,
    };

    let buffered = unary("gemini-3-flash-preview", SIGNED_ONE_PART).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_ONE_PART_STREAM).await;

    let expected = vec![AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
            "done",
            GeminiThoughtSignature::new("signature_REDACTED_1"),
        )),
        CarriedPayload::new(
            json!({"text": "done", "thoughtSignature": "signature_REDACTED_1"}),
            Some("gcp.gemini".to_owned()),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 0,
        }),
    ))];
    assert_eq!(buffered.choice.to_vec(), expected);
    assert_eq!(streamed.choice.to_vec(), expected);
}

/// The singular HTTP request the wire encodes.
fn sole(encoded: &crate::wire::Encoded) -> &http::Request<crate::wire::Body> {
    &encoded.request
}

#[test]
fn the_mode_chooses_the_endpoint_and_the_framing() {
    let wire = wire("gemini-2.5-flash");

    let unary = wire
        .encode(wire_request("probe"), Mode::Unary)
        .expect("the unary request encodes");
    assert_eq!(
        sole(&unary).uri().path(),
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    // The key is a query parameter on this family, appended last.
    assert_eq!(sole(&unary).uri().query(), Some("key=test-key"));
    assert_eq!(unary.framing, crate::http_client::framing::Framing::Whole);

    let streaming = wire
        .encode(wire_request("probe"), Mode::Streaming)
        .expect("the streaming request encodes");
    assert_eq!(
        sole(&streaming).uri().path(),
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    );
    assert_eq!(sole(&streaming).uri().query(), Some("alt=sse&key=test-key"));
    assert_eq!(streaming.framing, crate::http_client::framing::Framing::Sse);
    // Gemini reports no transport request-id header.
    assert_eq!(streaming.request_id_header, None);
}

/// The span names this wire has always recorded, per mode. Telemetry
/// equivalence is part of the port's contract.
#[test]
fn the_wire_keeps_its_span_names() {
    let wire = wire("gemini-2.5-flash");
    let descriptor = wire.describe();
    let telemetry = descriptor
        .telemetry
        .expect("GenerateContent declares telemetry");
    assert_eq!(telemetry(Mode::Unary), GenAiOperation::GenerateContent);
    // coord-a-rig-gemini-g5-1 adopts GenerateContent for streaming.
    assert_eq!(telemetry(Mode::Streaming), GenAiOperation::GenerateContent);
}

/// Completion inline media retains its exact payload as inspection data.
#[tokio::test]
async fn an_inline_data_part_survives_as_a_raw_content_block() {
    const IMAGE_REPLY: &str = r#"{"candidates":[{"content":{"parts":[{"inlineData":{"data":"iVBORw0KGgoAAAANSUhEUg==","mimeType":"image/png"}}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash-image","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1290,"promptTokenCount":15,"totalTokenCount":1305}}"#;

    let response = unary("gemini-2.5-flash-image", IMAGE_REPLY).await;
    let raw = response
        .choice
        .iter()
        .find_map(|item| match item {
            crate::message::AssistantContent::CarriedText(text) => {
                assert!(matches!(
                    &text.content,
                    crate::message::CarriedTextContent::InspectionOnlyMedia(value)
                        if value.is_empty()
                ));
                Some(text.payload.item.clone())
            }
            _ => None,
        })
        .expect("the inline image part survived as raw content");
    assert_eq!(
        raw.pointer("/inlineData/mimeType")
            .and_then(serde_json::Value::as_str),
        Some("image/png")
    );
    assert_eq!(
        raw.pointer("/inlineData/data")
            .and_then(serde_json::Value::as_str),
        Some("iVBORw0KGgoAAAANSUhEUg==")
    );
}

/// Parts within one document are distinct: two answer parts are two texts,
/// and a signed empty part keeps its own text rather than merging.
#[tokio::test]
async fn answer_parts_in_one_document_stay_distinct() {
    let body = r#"{"candidates":[{"content":{"parts":[{"text":"first"},{"text":"second"},{"text":"","thoughtSignature":"sig-9"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"r","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":1,"totalTokenCount":2}}"#;
    let response = unary("gemini-3-flash-preview", body).await;
    let texts: Vec<(String, Option<String>)> = response
        .choice
        .iter()
        .map(|part| match part {
            crate::message::AssistantContent::Text(text) => (text.text.clone(), None),
            crate::message::AssistantContent::CarriedText(text) => (
                text.text().to_owned(),
                text.signed_answer()
                    .map(|answer| answer.signature.as_str().to_owned()),
            ),
            other => panic!("answer text only: {other:?}"),
        })
        .collect();
    assert_eq!(
        texts,
        [
            ("first".to_owned(), None),
            ("second".to_owned(), None),
            (String::new(), Some("sig-9".to_owned())),
        ]
    );
}

/// A Gemini answer-text signature is Gemini API state: another provider's
/// wire encodes the text and never the signature.
#[test]
fn a_text_signature_reaches_no_other_wire() {
    use crate::message::{AssistantMessage, Message};
    use crate::operation::Completion;
    use crate::wire::{Mode, Wire};

    fn body<W>(wire: &W, request: CompletionRequest) -> String
    where
        W: Wire<Op = Completion, Payload = crate::wire::Encoded>,
    {
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the request encodes");
        crate::test_utils::json_body(&encoded.request).to_string()
    }

    let gemini = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let response = crate::test_utils::decode_reply(&gemini, &CompletionRequest::new("restate"), Mode::Unary, [
        crate::wire::WireFrame::Text(serde_json::json!({"candidates":[{
            "content":{"parts":[{"text":"the answer", "thoughtSignature":"c2lnbmVkLWFuc3dlcg=="}],"role":"model"},
            "finishReason":"STOP"
        }]}).to_string()),
    ], serde_json::Value::Null).expect("signed answer decodes");
    let mut request = CompletionRequest::from(vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(response.choice)),
        Message::user("again"),
    ]);
    request.max_tokens = Some(16);

    let openai = crate::providers::openai::wire::OpenAIConfig::new("sk-test");
    for (wire, encoded) in [
        (
            "anthropic",
            body(
                &crate::providers::anthropic::AnthropicConfig::new("sk-test")
                    .completion("claude-haiku-4-5"),
                request.clone(),
            ),
        ),
        (
            "openai chat",
            body(&openai.chat("gpt-4.1-nano"), request.clone()),
        ),
        (
            "openai responses",
            body(&openai.responses("gpt-5-mini"), request.clone()),
        ),
    ] {
        assert!(encoded.contains("the answer"), "{wire}: {encoded}");
        assert!(
            !encoded.contains("c2lnbmVkLWFuc3dlcg") && !encoded.contains("thoughtSignature"),
            "{wire} must not carry Gemini's signature: {encoded}"
        );
    }
}

/// A signed answer text survives serde, which is how a history persists, and
/// still replays its signature on its own part.
#[test]
fn a_signed_answer_text_round_trips_through_serde() {
    use crate::message::{AssistantMessage, Message};
    let wire = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let response = crate::test_utils::decode_reply(&wire, &CompletionRequest::new("restate"), Mode::Unary, [
        crate::wire::WireFrame::Text(serde_json::json!({"candidates":[{
            "content":{"parts":[{"text":"the answer", "thoughtSignature":"c2lnbmVk"}],"role":"model"},
            "finishReason":"STOP"
        }]}).to_string()),
    ], serde_json::Value::Null).expect("signed answer decodes");
    let message = Message::Assistant(AssistantMessage::new(response.choice));
    let json = serde_json::to_string(&message).expect("the message serializes");
    let loaded: Message = serde_json::from_str(&json).expect("the message loads");
    assert_eq!(loaded, message);
    let replayed =
        contents(vec![loaded], &wire, "gemini-3-flash-preview").expect("the turn replays");
    let parts = replayed[0]["parts"].as_array().expect("model parts");
    let [part] = parts.as_slice() else {
        panic!("one answer part: {parts:?}");
    };
    assert_eq!(part["thoughtSignature"].as_str(), Some("c2lnbmVk"));
    assert_ne!(part["thought"].as_bool(), Some(true));
}

/// A canonical signature-only reasoning part survives serde and replays as a thought.
#[test]
fn a_history_with_a_signature_only_reasoning_block_still_replays() {
    use crate::message::{AssistantMessage, Message};
    let wire = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let response = crate::test_utils::decode_reply(&wire, &CompletionRequest::new("restate"), Mode::Unary, [
        crate::wire::WireFrame::Text(serde_json::json!({"candidates":[{
            "content":{"parts":[{"text":"289"},{"thought":true,"text":"","thoughtSignature":"c2lnbmVk"}],"role":"model"},
            "finishReason":"STOP"
        }]}).to_string()),
    ], serde_json::Value::Null).expect("signature-only reasoning decodes");
    let message = Message::Assistant(AssistantMessage::new(response.choice));
    let json = serde_json::to_string(&message).expect("the message serializes");
    let loaded: Message = serde_json::from_str(&json).expect("the message loads");
    let replayed =
        contents(vec![loaded], &wire, "gemini-3-flash-preview").expect("the turn replays");
    let signed: Vec<(bool, &str)> = replayed[0]["parts"]
        .as_array()
        .expect("model parts")
        .iter()
        .filter_map(|part| {
            part["thoughtSignature"]
                .as_str()
                .map(|signature| (part["thought"].as_bool() == Some(true), signature))
        })
        .collect();
    assert_eq!(signed, [(true, "c2lnbmVk")]);
}

#[test]
fn generate_content_refuses_namespaced_and_custom_calls() {
    use crate::message::{
        AssistantContent, CallId, CustomToolCall, ToolCall, ToolFunction, ToolName,
        UnrepresentableToolCall,
    };
    use std::error::Error;
    let namespaced = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("add").expect("nonempty name"),
            serde_json::json!({"x":1}),
        ),
    )
    .with_namespace(Some("math".to_owned()));
    let custom = CustomToolCall::new(
        CallId::from_wire("call_2"),
        ToolName::new("apply_patch").expect("nonempty name"),
        r#"{"x": 1}"#,
    );
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    for (call, kind) in [
        (AssistantContent::ToolCall(namespaced), "namespace"),
        (AssistantContent::CustomToolCall(custom), "custom"),
    ] {
        let error: crate::error::ProviderError = super::assistant_part(&call, &wire, &wire.model)
            .expect_err("refused, not dropped")
            .into();
        assert!(matches!(&error, crate::error::ProviderError::Request(_)));
        let refusal = error
            .source()
            .and_then(|source| source.downcast_ref::<UnrepresentableToolCall>())
            .expect("the request failure retains the typed refusal");
        match (kind, refusal) {
            (
                "namespace",
                UnrepresentableToolCall::Namespace {
                    wire,
                    name,
                    namespace,
                },
            ) => {
                assert_eq!(
                    (*wire, name.as_str(), namespace.as_str()),
                    ("Gemini generateContent", "add", "math")
                );
            }
            ("custom", UnrepresentableToolCall::CustomCall { wire, name }) => {
                assert_eq!(
                    (*wire, name.as_str()),
                    ("Gemini generateContent", "apply_patch")
                );
            }
            _ => panic!("expected the {kind} refusal, got {refusal:?}"),
        }
    }
}

// CF007: these two selected modeled kinds are typed refusals in both modes.
#[test]
fn selected_function_response_and_file_data_refuse_as_typed_response() {
    use crate::error::{ErrorKind, ProviderError};
    use crate::test_utils::decode_reply;
    use crate::wire::{Mode, WireFrame};
    let function = json!({"functionResponse": {"name": "lookup", "response": {"ok": true}}});
    let file = json!({"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf"}});
    for mode in [Mode::Unary, Mode::Streaming] {
        for (parts, kind) in [
            (vec![function.clone()], "functionResponse"),
            (vec![file.clone()], "fileData"),
            (vec![function.clone(), file.clone()], "functionResponse"),
            (vec![file.clone(), function.clone()], "fileData"),
        ] {
            let document = json!({"candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": "STOP", "index": 0
            }], "responseId": "r"});
            let wire = wire("gemini-3-flash-preview");
            let error = decode_reply(
                &wire,
                &CompletionRequest::new("probe"),
                mode,
                [WireFrame::Text(document.to_string())],
                document,
            )
            .expect_err("selected modeled non-assistant parts must refuse");
            let ProviderError::Response(reason) = &error else {
                panic!("typed Response: {error:?}");
            };
            assert_eq!(
                reason,
                &format!(
                    "Gemini response part kind {kind} carries no assistant content rig can account for"
                )
            );
            assert_eq!(error.kind(), ErrorKind::Response);
            assert!(!error.is_retryable());
        }
    }
}

// Named U exceptions: A25/A48 known projections, A36/A38 hosted code, CF007
// selected refusals, and CF-RESTATED-POSITION-001 per-frame carried provenance.
#[test]
fn every_part_kind_survives_decode_and_replay() {
    use crate::message::{AssistantContent, CarriedPosition, Message};
    use crate::test_utils::history::assert_every_variant;
    use crate::test_utils::{decode_reply, json_body};
    use crate::wire::{Mode, Wire as _, WireFrame};
    let parts = vec![
        json!({"thought": true, "text": "why", "thoughtSignature": "c2ln"}),
        json!({"text": "plain", "futureField": 1}),
        json!({"functionCall": {"name": "lookup", "args": {"q": 1}, "id": "c1", "futureField": true}, "thoughtSignature": "c2ln"}),
        json!({"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}),
        json!({"functionResponse": {"name": "lookup", "response": {"ok": true}}}),
        json!({"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf"}}),
        json!({"executableCode": {"language": "PYTHON", "code": "print(1)"}}),
        json!({"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}}),
        json!({"futureKind": {"x": 1}}),
    ];
    const KINDS: [&str; 7] = [
        "text",
        "inlineData",
        "functionCall",
        "functionResponse",
        "fileData",
        "executableCode",
        "codeExecutionResult",
    ];
    let known: Vec<usize> = parts
        .iter()
        .take(8)
        .map(|part| {
            KINDS
                .iter()
                .position(|kind| part.get(*kind).is_some())
                .expect("a known part")
        })
        .collect();
    assert_every_variant(&known, |kind| *kind, KINDS.len());
    // The two CF007 kinds retain their typed refusal fixtures beside this one.
    let successful: Vec<Value> = [0, 1, 2, 3, 6, 7, 8]
        .into_iter()
        .map(|index| parts[index].clone())
        .collect();
    let end = json!({"finishReason": "STOP", "index": 0});
    let document = |parts: &[Value], end: Option<&Value>| {
        let mut candidate = json!({"content": {"parts": parts, "role": "model"}});
        if let (Some(candidate), Some(Value::Object(end))) = (candidate.as_object_mut(), end) {
            candidate.extend(end.clone());
        }
        WireFrame::Text(json!({"candidates": [candidate], "responseId": "r"}).to_string())
    };
    let whole = vec![document(&successful, Some(&end))];
    let mut streamed: Vec<WireFrame> = successful
        .iter()
        .map(|part| document(std::slice::from_ref(part), None))
        .collect();
    streamed.push(document(&[], Some(&end)));
    let wire = wire("gemini-3-flash-preview");
    let mut messages = Vec::new();
    for (mode, frames) in [(Mode::Unary, whole), (Mode::Streaming, streamed)] {
        let response = decode_reply(
            &wire,
            &CompletionRequest::new("probe"),
            mode,
            frames,
            Value::Null,
        )
        .expect("the narrowed reply decodes");
        assert_eq!(response.choice.len(), 5, "{mode:?}");
        assert_eq!(
            response.raw["candidates"][0]["content"]["parts"],
            json!(successful),
            "all raw kinds remain: {mode:?}"
        );
        let mut carried_count = 0;
        for (index, part) in response.choice.iter().enumerate() {
            let payload = match part {
                AssistantContent::CarriedReasoning(reasoning) => {
                    assert_eq!(index, 0);
                    assert_eq!(
                        reasoning.content,
                        vec![crate::message::CarriedReasoningPart::Text {
                            text: "why".to_owned(),
                            signature: Some("c2ln".to_owned())
                        }]
                    );
                    &reasoning.payload
                }
                AssistantContent::CarriedText(text) => {
                    assert!([1, 3].contains(&index), "the exact two carried text kinds");
                    if index == 1 {
                        assert!(
                            matches!(&text.content, crate::message::CarriedTextContent::InspectionOnlyText(value) if value == "plain")
                        );
                    } else {
                        assert!(
                            matches!(&text.content, crate::message::CarriedTextContent::InspectionOnlyMedia(value) if value.is_empty())
                        );
                    }
                    &text.payload
                }
                AssistantContent::ToolCall(call) => {
                    assert_eq!(index, 2);
                    assert_eq!(call.function.name.as_str(), "lookup");
                    assert_eq!(call.function.arguments_value(), json!({"q": 1}));
                    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("c1"));
                    assert_eq!(
                        call.native
                            .as_ref()
                            .expect("actual decoder retains the received part")
                            .item,
                        parts[2],
                        "G4: exact received Native item, independent of the retired preparation-only item_id observation"
                    );
                    continue;
                }
                AssistantContent::Opaque(_) => {
                    assert_eq!(index, 4);
                    continue;
                }
                AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::CustomToolCall(_)
                | AssistantContent::ProviderItem(_) => {
                    panic!("unexpected canonical content: {part:?}");
                }
            };
            carried_count += 1;
            assert_eq!(payload.provider.as_deref(), Some("gcp.gemini"));
            let original_index = match index {
                0 => 0,
                1 => 1,
                3 => 3,
                _ => panic!("invented carrier"),
            };
            assert_eq!(payload.item, parts[original_index]);
            let part_index = match mode {
                Mode::Unary => original_index as u64,
                Mode::Streaming => 0,
            };
            assert_eq!(
                payload.position,
                CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index
                }
            );
        }
        assert_eq!(carried_count, 3, "exhaustive carrier inventory: {mode:?}");
        let message = response.message().expect("a turn");
        // Original direct subject: encode the canonical turn without an added
        // request-adaptation or declared-tools change.
        let encoded = wire
            .encode(CompletionRequest::from(vec![message.clone()]), Mode::Unary)
            .expect("the turn replays");
        let replayed = json_body(sole(&encoded));
        assert_eq!(
            replayed["contents"][0]["parts"],
            json!([
                parts[0].clone(), {"text": "plain", "thought": false}, parts[2].clone(),
                {"text": "", "thought": false}, parts[8].clone()
            ]),
            "all unruled replay assertions remain exact: {mode:?}"
        );
        messages.push(message);
    }
    let [whole, streamed] = messages.as_slice() else {
        panic!("both modes retained");
    };
    let mut comparison = whole.clone();
    let Message::Assistant(comparison_turn) = &mut comparison else {
        panic!("assistant turn");
    };
    let Message::Assistant(streamed_turn) = streamed else {
        panic!("assistant turn");
    };
    // Only these two already-pinned per-frame positions receive the named
    // comparison exception; actual response/history values remain untouched.
    for index in [1, 3] {
        let AssistantContent::CarriedText(lhs) = &mut comparison_turn.content[index] else {
            panic!("carried text");
        };
        let AssistantContent::CarriedText(rhs) = &streamed_turn.content[index] else {
            panic!("carried text");
        };
        lhs.payload.position = rhs.payload.position;
    }
    assert_eq!(
        &comparison, streamed,
        "any mismatch beyond the two positions is an unresolved source/runtime defect"
    );
}

// Independent upstream wire, replay and prepare subjects retain their own fixtures.

const SIGNED_UNARY_UPSTREAM: &str = r#"{"candidates":[{"content":{"parts":[{"text":"289","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":59}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/streaming_twin_agrees_on_a_trailing_thought_signature.yaml`
/// — the same turn, streamed across two events, the signature riding a
/// trailing part that carries no `thought` flag.
const SIGNED_STREAM_UPSTREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"289"}],"role":"model"},"index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
    r#"data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
);

fn to_content_upstream(message: impl Into<message::Message>) -> Result<Value, EncodeError> {
    to_content_for_upstream("gemini-2.5-flash", message)
}

fn to_content_for_upstream(
    model: &str,
    message: impl Into<message::Message>,
) -> Result<Value, EncodeError> {
    contents_for_upstream(vec![message.into()], model)?
        .into_iter()
        .next()
        .ok_or_else(|| EncodeError::request("no content"))
}

fn contents_for_upstream(
    history: Vec<message::Message>,
    model: &str,
) -> Result<Vec<Value>, EncodeError> {
    contents(history, &wire(model), model)
}

fn body_for_upstream(request: CompletionRequest, model: &str) -> Map<String, Value> {
    request_body(&request, &wire(model), model, None, |_| {})
        .and_then(|body| Ok(body.deserialize()?))
        .expect("the request encodes")
}

fn parts_upstream(content: &Value) -> &[Value] {
    content["parts"].as_array().map_or(&[], Vec::as_slice)
}

fn prepared_body_upstream(model: &str, request: CompletionRequest) -> Value {
    use crate::wire::Operation;
    let wire = wire(model);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let crate::wire::Body::Bytes(bytes) = sole(&encoded).body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

#[test]
fn test_unknown_finish_reason_round_trips_verbatim() {
    // A wire value this crate does not know keeps the provider's spelling.
    assert_eq!(
        map_google_finish_reason("FINISH_REASON_FUTURE"),
        crate::completion::FinishReason::Other("FINISH_REASON_FUTURE".to_string())
    );
}

#[test]
fn test_unknown_block_reason_deserializes_verbatim() {
    // Same contract for prompt feedback: a new block reason must not fail
    // the payload, and the spelling is preserved.
    let error = blocked_prompt_error(&json!({ "blockReason": "BLOCK_REASON_FUTURE" }))
        .expect("an unknown block reason is a block");
    assert!(
        matches!(&error, ProviderError::ProviderResponse(response) if response.code.as_deref() == Some("BLOCK_REASON_FUTURE")),
        "{error:?}"
    );
}

#[tokio::test]
async fn test_streaming_candidate_with_unknown_finish_reason_stays_parseable() {
    // A streamed terminal chunk with an unknown reason still produces the
    // terminal record, with the reason verbatim.
    let converted = streamed(
        "gemini-2.5-flash",
        concat!(
            r#"data: {"candidates":[{"content":{"parts":[{"text":"done"}],"role":"model"},"finishReason":"FINISH_REASON_FUTURE"}]}"#,
            "\r\n\r\n",
        ),
    )
    .await;

    assert_eq!(converted.text(), "done");
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn tool_results_carry_only_jpeg_png_and_webp() {
    use crate::completion::{Media, Place};
    use crate::message::{DocumentSourceKind, Image, ImageMediaType};

    let image = |media_type| Image {
        data: DocumentSourceKind::base64("aW1hZ2U="),
        media_type: Some(media_type),
        ..Image::default()
    };
    for media_type in [
        ImageMediaType::JPEG,
        ImageMediaType::PNG,
        ImageMediaType::WEBP,
    ] {
        assert!(super::encodes(
            Media::Image(&image(media_type), Place::ToolResult),
            false
        ));
    }
    for media_type in [
        ImageMediaType::GIF,
        ImageMediaType::HEIC,
        ImageMediaType::HEIF,
        ImageMediaType::SVG,
    ] {
        assert!(
            !super::encodes(
                Media::Image(&image(media_type.clone()), Place::ToolResult),
                false
            ),
            "{media_type:?}"
        );
    }
    assert!(super::encodes(
        Media::Image(&image(ImageMediaType::HEIC), Place::User),
        false
    ));
}

#[test]
fn a_url_tool_result_image_is_sent_as_file_data() {
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    let tool_result = ToolResult {
        answers: message::AnsweredToolCall::Function,
        is_error: false,
        call: crate::message::CallId::from_wire(""),
        name: crate::message::ToolName::new("screenshot_tool".to_string()).expect("tool name"),
        content: vec![
            ToolResultContent::Image(Image {
                data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                native: None,
            }),
            ToolResultContent::text("after-image"),
        ],
    };

    let error = to_content_for_upstream(
        "gemini-3-flash-preview",
        message::Message::User {
            content: vec![message::UserContent::ToolResult(tool_result)],
        },
    )
    .expect_err("A66: the original URL-backed result refuses before encoding");
    let error = ProviderError::from(error);
    assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
    let source = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<message::MessageError>())
        .expect("the URL refusal retains its typed conversion source");
    assert!(
        matches!(source, message::MessageError::ConversionError(reason)
        if reason == "Gemini tool result images must use base64 inline data; URL-backed images are not supported")
    );
}

#[tokio::test]
async fn a_trailing_thought_signature_joins_the_text_it_follows() {
    let buffered = unary("gemini-3-flash-preview", SIGNED_UNARY_UPSTREAM).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_STREAM_UPSTREAM).await;
    // Gemini25 preserves the signed part's boundary instead of joining it.
    let signed = json!({ "text": "289", "thoughtSignature": "c2lnbmF0dXJl" });
    let [message::AssistantContent::CarriedText(text)] = buffered.choice.as_slice() else {
        panic!("one answer text: {:?}", buffered.choice);
    };
    assert_eq!(text.text(), "289");
    assert_eq!(text.payload.item, signed);
    assert_eq!(
        text.signed_answer().map(|answer| answer.signature.as_str()),
        Some("c2lnbmF0dXJl")
    );
    let [
        message::AssistantContent::Text(first),
        message::AssistantContent::CarriedText(last),
    ] = streamed.choice.as_slice()
    else {
        panic!(
            "an unsigned part followed by its own signed empty part: {:?}",
            streamed.choice
        );
    };
    assert_eq!(first.text, "289");
    assert_eq!(last.text(), "");
    assert_eq!(
        last.payload.item,
        json!({"text":"", "thoughtSignature":"c2lnbmF0dXJl"})
    );
    assert_eq!(
        last.signed_answer().map(|answer| answer.signature.as_str()),
        Some("c2lnbmF0dXJl")
    );

    let replayed = to_content_upstream(buffered.choice.clone()).expect("the turn replays");
    assert_eq!(
        replayed["parts"],
        json!([
            {"text":"289", "thought":false, "thoughtSignature":"c2lnbmF0dXJl"}
        ])
    );
}

#[tokio::test]
async fn unary_and_streamed_twins_report_one_genai_operation() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = crate::test_utils::TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());
    unary("gemini-3-flash-preview", SIGNED_UNARY_UPSTREAM).await;
    streamed("gemini-3-flash-preview", SIGNED_STREAM_UPSTREAM).await;
    tracing::callsite::rebuild_interest_cache();
    capture.clear();

    unary("gemini-3-flash-preview", SIGNED_UNARY_UPSTREAM).await;
    streamed("gemini-3-flash-preview", SIGNED_STREAM_UPSTREAM).await;

    let spans = capture
        .spans()
        .into_iter()
        .filter(|span| span.target == "rig::completions")
        .map(|span| {
            (
                span.name,
                span.value("gen_ai.operation.name").cloned(),
                span.value("gen_ai.request.stream").cloned(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        spans,
        [
            (
                "generate_content",
                Some(json!("generate_content")),
                Some(json!(false))
            ),
            (
                "generate_content",
                Some(json!("generate_content")),
                Some(json!(true))
            ),
        ]
    );
}

#[tokio::test]
async fn a_long_stream_records_its_response_metadata_once() {
    const CHUNKS: usize = 64;
    let chunk = r#"data: {"candidates":[{"content":{"parts":[{"text":"word "}],"role":"model"},"index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"totalTokenCount":17}}"#;
    let last = r#"data: {"candidates":[{"content":{"parts":[{"text":"end"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":65,"promptTokenCount":14,"totalTokenCount":79}}"#;
    let body: &'static str = format!(
        "{}{last}\r\n\r\n",
        format!("{chunk}\r\n\r\n").repeat(CHUNKS)
    )
    .leak();

    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = crate::test_utils::TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());
    streamed("gemini-3-flash-preview", SIGNED_STREAM_UPSTREAM).await;
    tracing::callsite::rebuild_interest_cache();
    capture.clear();

    let response = streamed("gemini-3-flash-preview", body).await;
    assert_eq!(response.usage.output_tokens, Some(65));

    let spans = capture.spans();
    let [span] = spans
        .iter()
        .filter(|span| span.target == "rig::completions")
        .collect::<Vec<_>>()[..]
    else {
        panic!("one completion span: {spans:?}");
    };
    assert_eq!(span.record_count("gen_ai.response.id"), 2);
    assert_eq!(span.record_count("gen_ai.response.model"), 2);
    assert_eq!(
        span.text("gen_ai.response.id").as_deref(),
        Some("id_REDACTED_1")
    );
    assert_eq!(span.u64("gen_ai.usage.output_tokens"), Some(65));
}

#[test]
fn a_text_signature_reaches_no_other_model() {
    let signed = message::AssistantMessage {
        origin: Some(message::Origin::new(
            "gemini.generate_content",
            PROVIDER_NAME,
            "gemini-3-flash-preview",
        )),
        ..message::AssistantMessage::new(vec![
            message::AssistantContent::text("the answer")
                .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
        ])
    };
    let history = vec![message::Message::user("q"), signed.into()];
    for (model, part) in [
        (
            "gemini-3-flash-preview",
            json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" }),
        ),
        ("gemini-2.5-flash", json!({ "text": "the answer" })),
    ] {
        let adapted = crate::completion::adapt(&history, &wire(model));
        let contents = contents_for_upstream(adapted, model).expect("the history encodes");
        assert_eq!(contents[1]["parts"], json!([part]), "{model}");
    }
}

#[test]
fn canonical_blocks_rebuild_as_pi_rebuilds_them() {
    let call = message::AssistantContent::tool_call(
        "call-1",
        message::ToolName::new("lookup").expect("tool name"),
        json!({ "q": 1 }),
    );
    let message = message::Message::from(vec![
        message::AssistantContent::reasoning("why"),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        }),
        message::AssistantContent::text("answer"),
        call,
    ]);
    // A rebuilt canonical functionCall part carries the recorded
    // `"thought": false` marker on the REST JSON wire.
    for (model, call) in [
        (
            "gemini-2.5-flash",
            // Gemini25 preserves genuine provider handles on every model.
            json!({
                "functionCall": { "name": "lookup", "args": { "q": 1 }, "id":"call-1" },
                "thought": false,
            }),
        ),
        (
            "gemini-3-flash-preview",
            json!({
                "functionCall": { "name": "lookup", "args": { "q": 1 }, "id": "call-1" },
                "thought": false,
                "thoughtSignature": "skip_thought_signature_validator",
            }),
        ),
    ] {
        let contents =
            contents_for_upstream(vec![message.clone()], model).expect("the turn encodes");
        assert_eq!(
            contents[0]["parts"],
            json!([
                { "thought": true, "text": "why" },
                { "text": "answer" },
                call,
            ]),
            "{model}"
        );
    }
}

#[test]
fn a_signature_that_is_not_base64_is_left_out() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("a")
            .with_native(json!({ "text": "a", "thoughtSignature": "not base64!", "extra": 1 })),
        message::AssistantContent::text("b")
            .with_native(json!({ "text": "b", "thoughtSignature": "c2ln" })),
    ]);
    let contents =
        contents_for_upstream(vec![message], "gemini-2.5-flash").expect("the turn encodes");
    assert_eq!(
        contents[0]["parts"],
        json!([
            { "text": "a", "extra": 1 },
            { "text": "b", "thoughtSignature": "c2ln" },
        ])
    );
}

#[test]
fn call_ids_follow_the_models_that_take_them() {
    for (model, takes) in [
        ("gemini-2.5-flash", false),
        ("gemini-2.0-flash", false),
        ("gemini-3-flash-preview", true),
        ("gemini-3.8-flash", true),
        ("gemini-live-3-flash", true),
        ("claude-sonnet-4-5", true),
        ("gpt-oss-120b", true),
        ("gemma-3-27b-it", false),
    ] {
        assert_eq!(requires_tool_call_id(model), takes, "{model}");
    }
    let foreign = format!("call.{}|x", "y".repeat(80));
    let normalized = normalize_tool_call_id("gemini-3-flash-preview", &foreign);
    assert_eq!(normalized.len(), 64);
    assert!(normalized.starts_with("call_yyy"));
    assert_eq!(
        normalize_tool_call_id("gemini-2.5-flash", &foreign),
        foreign
    );
}

#[test]
fn current_turn_replay_drops_only_finished_signatures() {
    let signed = |text: &str| {
        message::AssistantContent::text(text)
            .with_native(json!({ "text": text, "thoughtSignature": "c2ln" }))
    };
    let mut body = body_for_upstream(
        CompletionRequest::from(vec![
            message::Message::user("one"),
            message::Message::from(vec![signed("first")]),
            message::Message::user("two"),
            message::Message::from(vec![signed("second")]),
        ]),
        "gemini-2.5-flash",
    );
    let Some(Value::Array(contents)) = body.get_mut("contents") else {
        panic!("contents");
    };
    drop_finished_signatures(contents);
    assert_eq!(contents[1]["parts"], json!([{ "text": "first" }]));
    assert_eq!(
        contents[3]["parts"],
        json!([{ "text": "second", "thoughtSignature": "c2ln" }])
    );
}

#[test]
fn candidate_metadata_reaches_streamed_raw() {
    use crate::wire::WireFrame;

    let metadata = json!({
        "finishReason": "STOP",
        "index": 0,
        "safetyRatings": [{ "category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE" }],
        "citationMetadata": { "citationSources": [{ "uri": "https://example.com", "startIndex": 0, "endIndex": 4 }] },
        "groundingMetadata": { "webSearchQueries": ["rig"], "groundingChunks": [{ "web": { "uri": "https://example.com" } }] },
        "urlContextMetadata": { "urlMetadata": [{ "retrievedUrl": "https://example.com" }] },
    });
    let first = json!({ "candidates": [{ "content": { "parts": [{ "text": "r" }], "role": "model" }, "index": 0 }] });
    let mut last = metadata.clone();
    last["content"] = json!({ "parts": [{ "text": "ig" }], "role": "model" });
    let frames =
        [first, json!({ "candidates": [last] })].map(|chunk| WireFrame::Text(chunk.to_string()));
    let response =
        crate::test_utils::history::decode(&wire("gemini-2.5-flash"), Mode::Streaming, frames)
            .expect("the reply decodes");
    let mut expected = metadata;
    expected["content"] = json!({ "parts": [{ "text": "rig" }], "role": "model" });
    assert_eq!(response.raw, json!({ "candidates": [expected] }));
}

#[test]
fn a_rig_issued_call_id_is_spelled_as_a_request_local_alias() {
    let call = message::ToolCall::from_wire(
        "",
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({}),
        ),
    );
    let history = vec![
        message::Message::from(call.clone()),
        message::Message::tool_results(vec![
            call.result(vec![message::ToolResultContent::text("out")]),
        ]),
    ];
    let contents =
        contents_for_upstream(history, "gemini-3-flash-preview").expect("the history encodes");
    // Gemini25 leaves these optional local identity slots absent.
    assert_eq!(contents[0]["parts"][0]["functionCall"].get("id"), None);
    assert_eq!(contents[1]["parts"][0]["functionResponse"].get("id"), None);
}

#[test]
fn a_request_model_override_decides_call_id_normalization() {
    use crate::completion::ReplayTarget;

    let wire = GenerateContent::new(
        crate::providers::gemini::GeminiConfig::new("k"),
        "gemini-2.5-flash",
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-2.5-flash", None),
        "a.b"
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-3-pro-preview", None),
        "a_b"
    );
}

#[tokio::test]
async fn an_unknown_traffic_type_or_modality_never_fails_the_reply() {
    let response = fold_unary(
        "gemini-2.5-flash",
        r#"{"candidates":[{"content":{"parts":[{"text":"kept"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5,"trafficType":"ON_DEMAND_PRIORITY","promptTokensDetails":[{"modality":"HOLOGRAM","tokenCount":3}],"serviceTier":"flex"}}"#,
    )
    .await
    .expect("a usage label never fails the reply");
    assert_eq!(response.text(), "kept");
    assert_eq!(response.usage.input_tokens, Some(3));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(5));
    assert_eq!(
        response.raw["usageMetadata"]["trafficType"],
        "ON_DEMAND_PRIORITY"
    );
    assert_eq!(
        response.raw["usageMetadata"]["promptTokensDetails"][0]["modality"],
        "HOLOGRAM"
    );
    assert_eq!(response.raw["usageMetadata"]["serviceTier"], "flex");
}

#[test]
fn tool_result_images_reach_gemini_2_in_a_following_user_message() {
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_shot"),
        message::ToolFunction::new(
            message::ToolName::new("shot").expect("a tool name"),
            json!({}),
        ),
    );
    let image = message::Image {
        data: message::DocumentSourceKind::base64("aW1hZ2U="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    };
    let history = vec![
        message::Message::user("look"),
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ])),
        message::Message::User {
            content: vec![message::UserContent::ToolResult(call.result(vec![
                message::ToolResultContent::text("see"),
                message::ToolResultContent::Image(image),
            ]))],
        },
    ];
    // A request that declares no tool gets its history's calls and results
    // as text, so this one declares the tool it replays.
    let request = |history: Vec<message::Message>| {
        let mut request =
            CompletionRequest::new("next").tools(vec![crate::completion::ToolDefinition {
                name: message::ToolName::new("shot").expect("a tool name"),
                description: "Take a screenshot".to_owned(),
                parameters: json!({ "type": "object", "properties": {} }),
            }]);
        request.chat_history = history;
        request
    };

    let body = prepared_body_upstream("gemini-2.5-flash", request(history.clone()));
    let contents = body["contents"].as_array().expect("contents");
    let response = contents
        .iter()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_none(), "{response}");
    assert!(
        contents.iter().any(|content| content["role"] == "user"
            && content["parts"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part.get("inlineData").is_some()))),
        "the image follows in a user message: {body}"
    );

    let body = prepared_body_upstream("gemini-3-flash-preview", request(history));
    let response = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_some(), "{response}");
}

#[test]
fn a_turn_with_nothing_to_send_is_no_content() {
    let turn = message::AssistantMessage::new(vec![message::AssistantContent::reasoning("")]);
    let contents = contents_for_upstream(
        vec![
            message::Message::user("q"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        "gemini-3-flash-preview",
    )
    .unwrap();
    assert!(
        contents
            .iter()
            .all(|content| !parts_upstream(content).is_empty()),
        "{contents:?}"
    );
}

#[test]
fn a_later_system_message_folds_into_the_instruction() {
    let history = vec![
        Message::system("s0"),
        Message::user("a"),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            crate::message::AssistantContent::text("ok"),
        ])),
        Message::user("b"),
        Message::system("later"),
        Message::user("c"),
    ];
    let body = prepared_body_upstream("gemini-2.5-flash", CompletionRequest::from(history));
    let roles: Vec<&str> = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|content| content["role"].as_str())
        .collect();
    assert_eq!(roles, ["user", "model", "user"], "{}", body["contents"]);
    let instruction = body["systemInstruction"].to_string();
    assert!(
        instruction.contains("s0") && instruction.contains("later"),
        "{instruction}"
    );
}

#[test]
fn a_signed_answer_text_round_trips_through_serde_upstream() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("the answer")
            .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
    ]);
    let json = serde_json::to_string(&message).expect("the message serializes");
    let loaded: message::Message = serde_json::from_str(&json).expect("the message loads");
    assert_eq!(loaded, message);
    let content = to_content_upstream(loaded).expect("the turn replays");
    let [part] = parts_upstream(&content) else {
        panic!("one answer part: {content}");
    };
    assert_eq!(part["thoughtSignature"], "c2lnbmVk");
    assert_ne!(part.get("thought"), Some(&json!(true)));
}

mod function_thought_controls;

#[test]
fn numeric_enum_declaration_keeps_the_exact_original_json_schema() {
    for numeric in [
        json!({"type":"integer","enum":[0,1,2]}),
        json!({"type":"number","enum":[0,1.25,-2]}),
    ] {
        let parameters = json!({
            "type":"object",
            "properties":{"slot":numeric},
            "required":["slot"],
            "additionalProperties":false
        });
        let body = encoded_schema_request(parameters.clone()).expect("numeric enum request");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parametersJsonSchema":parameters
            }))
        );
    }
}

#[test]
fn string_enum_declaration_keeps_projected_parameters() {
    let parameters = json!({
        "type":"object",
        "properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},
        "required":["unit"]
    });
    let body = encoded_schema_request(parameters.clone()).expect("string enum request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parameters":parameters
        }))
    );
}

#[test]
fn numeric_and_string_tools_select_their_own_schema_members() {
    use crate::wire::Operation;
    let numeric = json!({
        "type":"object",
        "properties":{"slot":{"type":"integer","enum":[0,1,2]}},
        "required":["slot"],
        "additionalProperties":false
    });
    let strings = json!({
        "type":"object",
        "properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},
        "required":["unit"]
    });
    let wire = wire("gemini-2.5-flash");
    let mut request = CompletionRequest::new("hello");
    request.tools = vec![
        crate::completion::ToolDefinition::new(
            message::ToolName::new("numeric_tool").expect("tool name"),
            "numeric fixture",
            numeric.clone(),
        ),
        crate::completion::ToolDefinition::new(
            message::ToolName::new("string_tool").expect("tool name"),
            "string fixture",
            strings.clone(),
        ),
    ];
    let request = Completion::prepare(request, &wire.describe()).expect("prepared tools");
    let encoded = wire.encode(request, Mode::Unary).expect("encoded tools");
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations"),
        Some(&json!([
            {
                "name":"numeric_tool", "description":"numeric fixture",
                "parametersJsonSchema":numeric
            },
            {
                "name":"string_tool", "description":"string fixture",
                "parameters":strings
            }
        ]))
    );
}

#[test]
fn schema_object_child_errors_keep_their_paths_and_request_error_type() {
    for (parameters, path, reason) in [
        (
            json!({"type":"object","properties":{"slot":{"type":"integer","$defs":[]}}}),
            "/properties/slot/$defs",
            "$defs must be an object",
        ),
        (
            json!({"type":"array","items":{"type":"integer","$defs":[]}}),
            "/items/$defs",
            "$defs must be an object",
        ),
        (
            json!({"type":"object","properties":{"slot":{"$ref":"#/other/Missing"}},"$defs":{}}),
            "/properties/slot/$ref",
            "Unsupported reference format: #/other/Missing",
        ),
        (
            json!({"type":"array","items":{"$defs":{},"$ref":"#/$defs/Missing"}}),
            "/items/$ref",
            "Reference not found: #/$defs/Missing",
        ),
        (
            json!({"type":"object","properties":{"a/b~c":{"$defs":[]}}}),
            "/properties/a~1b~0c/$defs",
            "$defs must be an object",
        ),
        (
            json!({
                "type":"object",
                "properties":{"slot":{"anyOf":[
                    {"type":"null"},
                    {"type":"object","properties":{"inner":{"$defs":[]}}}
                ]}}
            }),
            "/properties/slot/anyOf/1/properties/inner/$defs",
            "$defs must be an object",
        ),
    ] {
        let error = encoded_schema_request(parameters).expect_err("schema child is refused");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        let diagnostic = format!(
            "Tool 'schema_tool' could not be converted to a schema: schema path {path}: {reason}"
        );
        assert_eq!(
            std::error::Error::source(&error)
                .expect("request reason")
                .to_string(),
            diagnostic
        );
        assert_eq!(error.to_string(), format!("RequestError: {diagnostic}"));
    }
}

#[test]
fn numeric_enum_routing_does_not_bypass_schema_object_validation() {
    let parameters = json!({
        "type":"object",
        "properties":{
            "slot":{"type":"integer","enum":[0,1,2]},
            "broken":{"type":"array","items":{"$defs":{},"$ref":"#/$defs/Missing"}}
        },
        "required":["slot"],
        "additionalProperties":false
    });
    let error = encoded_schema_request(parameters).expect_err("numeric request still validates");
    assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
    assert_eq!(
        error.to_string(),
        "RequestError: Tool 'schema_tool' could not be converted to a schema: schema path /properties/broken/items/$ref: Reference not found: #/$defs/Missing"
    );
}

#[test]
fn referenced_numeric_enum_keeps_original_definitions_and_references() {
    let parameters = json!({
        "type":"object",
        "properties":{"slot":{"$ref":"#/$defs/Choice"}},
        "required":["slot"],
        "additionalProperties":false,
        "$defs":{"Choice":{"type":"integer","enum":[0,1,2]}}
    });
    let body = encoded_schema_request(parameters.clone()).expect("referenced numeric request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parametersJsonSchema":parameters
        }))
    );
}

#[test]
fn numeric_enum_in_annotation_does_not_change_the_schema_route() {
    let body = encoded_schema_request(json!({
        "type":"string",
        "enum":["a","b"],
        "default":{"type":"integer","enum":[0,1,2]}
    }))
    .expect("annotation retains the ordinary route");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parameters":{
                "type":"string","enum":["a","b"],
                "default":{"type":"integer","enum":[0,1,2]}
            }
        }))
    );
}

#[test]
fn untyped_numeric_enum_keeps_the_exact_original_json_schema() {
    let parameters = json!({
        "type":"object",
        "properties":{"slot":{"enum":[0,1,2]}},
        "required":["slot"],
        "additionalProperties":false
    });
    let body = encoded_schema_request(parameters.clone()).expect("untyped numeric enum request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parametersJsonSchema":parameters
        }))
    );
}

#[test]
fn numeric_enum_in_unselected_composition_branch_keeps_the_original_schema() {
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let parameters = json!({
            "type":"object",
            "properties":{"choice":{(keyword):[
                {"type":"string","enum":["a","b"]},
                {"enum":[0,1,2]}
            ]}},
            "required":["choice"]
        });
        let body = encoded_schema_request(parameters.clone()).expect("numeric composition request");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parametersJsonSchema":parameters
            }))
        );
    }
}

#[test]
fn nullable_string_enum_keeps_parameters_while_numeric_union_keeps_raw_schema() {
    for string_types in [json!(["string", "null"]), json!(["null", "string"])] {
        let body = encoded_schema_request(json!({
            "type":"object",
            "properties":{"choice":{"type":string_types,"enum":["a","b"]}}
        }))
        .expect("nullable string enum request");
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0"),
            Some(&json!({
                "name":"schema_tool", "description":"schema fixture",
                "parameters":{
                    "type":"object",
                    "properties":{"choice":{"type":"string","nullable":true,"enum":["a","b"]}}
                }
            }))
        );
        // A number in an explicitly string-typed enum is refused, not omitted.
        let error = encoded_schema_request(json!({
            "type":"object",
            "properties":{"choice":{"type":string_types,"enum":["a",1,"b"]}}
        }))
        .expect_err("a numeric value of a string enum is refused");
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            "RequestError: Tool 'schema_tool' could not be converted to a schema: schema path /properties/choice/enum/1: Gemini enum values are strings"
        );
    }
    let parameters = json!({
        "type":"object",
        "properties":{"choice":{"type":["string","number","null"],"enum":["a",0,1.25]}}
    });
    let body = encoded_schema_request(parameters.clone()).expect("numeric union request");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parametersJsonSchema":parameters
        }))
    );
}

#[test]
fn unused_numeric_definition_does_not_select_the_raw_schema_member() {
    let body = encoded_schema_request(json!({
        "type":"object",
        "properties":{"unit":{"type":"string","enum":["a","b"]}},
        "$defs":{"Unused":{"enum":[0,1,2]}}
    }))
    .expect("unused definition does not change the route");
    assert_eq!(
        body.pointer("/tools/0/functionDeclarations/0"),
        Some(&json!({
            "name":"schema_tool", "description":"schema fixture",
            "parameters":{
                "type":"object",
                "properties":{"unit":{"type":"string","enum":["a","b"]}}
            }
        }))
    );
}

#[test]
fn numeric_composition_does_not_hide_a_later_schema_object_error() {
    let error = encoded_schema_request(json!({
        "type":"object",
        "properties":{"choice":{"anyOf":[
            {"type":"string","enum":["a","b"]},
            {"enum":[0,1,2]},
            {"$defs":[]}
        ]}}
    }))
    .expect_err("every schema-object alternative is checked");
    assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
    assert_eq!(
        error.to_string(),
        "RequestError: Tool 'schema_tool' could not be converted to a schema: schema path /properties/choice/anyOf/2/$defs: $defs must be an object"
    );
}
