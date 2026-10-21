use super::*;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::message;
use crate::providers::cohere::streaming::{ChatEvent, finish_of, usage_of};
use crate::test_utils::RecordingHttpClient;
use crate::wire::WireFrame;

fn body(request: CompletionRequest) -> Value {
    serde_json::to_value(
        native()
            .body(&request, Mode::Unary)
            .expect("native request"),
    )
    .expect("JSON body")
}

async fn unary(body: &'static str) -> crate::completion::CompletionResponse {
    Model::new(
        CohereConfig::new("test-key").completion(MODEL),
        RecordingHttpClient::new(body),
    )
    .call(CompletionRequest::new("hello"))
    .await
    .expect("reply decodes")
}

#[test]
fn test_deserialize_completion_response() {
    let json_data = r#"
        {
            "id": "abc123",
            "message": {
                "role": "assistant",
                "tool_plan": "I will use the subtract tool to find the difference between 2 and 5.",
                "tool_calls": [
                        {
                            "id": "subtract_sm6ps6fb6y9f",
                            "type": "function",
                            "function": {
                                "name": "subtract",
                                "arguments": "{\"x\":5,\"y\":2}"
                            }
                        }
                    ]
                },
                "finish_reason": "TOOL_CALL",
                "usage": {
                "billed_units": {
                    "input_tokens": 78,
                    "output_tokens": 27
                },
                "tokens": {
                    "input_tokens": 1028,
                    "output_tokens": 63
                }
            }
        }
        "#;

    let event: ChatEvent = serde_json::from_str(json_data).expect("wire reply");
    let response = event.fields;
    assert_eq!(response["id"], "abc123");
    assert_eq!(response["finish_reason"], "TOOL_CALL");
    assert_eq!(
        response["usage"]["billed_units"]["input_tokens"].as_f64(),
        Some(78.0)
    );
    assert_eq!(
        response["usage"]["billed_units"]["output_tokens"].as_f64(),
        Some(27.0)
    );
    assert_eq!(
        response["usage"]["tokens"]["input_tokens"].as_f64(),
        Some(1028.0)
    );
    assert_eq!(
        response["usage"]["tokens"]["output_tokens"].as_f64(),
        Some(63.0)
    );
    assert!(response["message"].arr("citations").is_empty());
    assert_eq!(response["message"].arr("tool_calls").len(), 1);
    let function = &response["message"]["tool_calls"][0]["function"];
    assert_eq!(function["name"], "subtract");
    assert_eq!(
        serde_json::from_str::<Value>(function["arguments"].as_str().expect("JSON arguments"))
            .expect("arguments"),
        json!({"x":5,"y":2})
    );
}

#[test]
fn finish_reason_maps_every_documented_wire_value() {
    use crate::completion::FinishReason as F;
    for (wire, reason) in [
        ("COMPLETE", F::Stop),
        ("STOP_SEQUENCE", F::Stop),
        ("MAX_TOKENS", F::Length),
        ("TOOL_CALL", F::ToolCalls),
        ("ERROR", F::Other("ERROR".to_owned())),
    ] {
        assert_eq!(finish_of(wire), reason);
    }
}

#[test]
fn unknown_finish_reason_survives_verbatim() {
    assert_eq!(
        finish_of("ERROR_TOXIC"),
        crate::completion::FinishReason::Other("ERROR_TOXIC".to_owned())
    );
}

#[tokio::test]
async fn tool_call_response_normalizes_to_tool_calls_finish_reason() {
    let normalized = unary(r#"{"id":"abc123","message":{"role":"assistant","tool_calls":[{"id":"subtract_1","type":"function","function":{"name":"subtract","arguments":"{\"x\":5,\"y\":2}"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}"#).await;
    assert_eq!(normalized.provider(), "cohere");
    assert_eq!(normalized.response_id(), Some("abc123"));
    assert_eq!(normalized.message_id, None);
    assert_eq!(normalized.model(), None);
    assert_eq!(
        normalized.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
    assert_eq!(normalized.usage.input_tokens, Some(10));
    assert_eq!(normalized.usage.output_tokens, Some(4));
    assert_eq!(normalized.usage.total_tokens, Some(14));
    let Some(AssistantContent::ToolCall(call)) = normalized.choice.first() else {
        panic!("tool call")
    };
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("subtract_1"));
    assert_eq!(call.function.name.as_str(), "subtract");
    assert_eq!(call.function.arguments_value(), json!({"x":5,"y":2}));
}

#[test]
fn usage_is_mapped_from_tokens_and_carries_cached_input() {
    let usage = usage_of(Some(
        &json!({"billed_units":{"input_tokens":135,"output_tokens":24},"cached_tokens":112,"tokens":{"input_tokens":1610,"output_tokens":56}}),
    ));
    assert_eq!(usage.input_tokens, Some(1610));
    assert_eq!(usage.output_tokens, Some(56));
    assert_eq!(usage.total_tokens, Some(1666));
    assert_eq!(usage.cached_input_tokens, Some(112));
}

#[tokio::test]
async fn response_usage_matches_the_canonical_mapping() {
    const BODY: &str = r#"{"id":"abc123","finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"usage":{"billed_units":{"input_tokens":135,"output_tokens":24},"cached_tokens":112,"tokens":{"input_tokens":1610,"output_tokens":56}}}"#;
    let event: ChatEvent = serde_json::from_str(BODY).expect("wire response");
    let mut expected = usage_of(event.fields.get("usage"));
    let converted = unary(BODY).await;
    assert!(
        converted.usage.cost.is_some(),
        "catalog prices the billed counters"
    );
    expected.cost = Some(crate::completion::Cost::from_parts(
        135.0 * 2.5 / 1e6,
        24.0 * 10.0 / 1e6,
        0.0,
        0.0,
    ));
    assert_eq!(converted.usage, expected);
    assert_eq!(converted.usage.input_tokens, Some(1610));
    assert_eq!(converted.usage.cached_input_tokens, Some(112));
    assert_eq!(
        converted.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("hi"))
    );
}

#[test]
fn usage_without_token_counts_is_unreported() {
    for value in [json!({}), json!({"cached_tokens":512})] {
        assert_eq!(usage_of(Some(&value)), crate::completion::Usage::default());
    }
}

#[test]
fn tool_result_content_is_type_tagged() {
    let call = ToolCall::new(
        CallId::from_wire("subtract_1"),
        ToolFunction::new(ToolName::new("subtract").expect("name"), json!({})),
    );
    let history = CompletionRequest::from(vec![Message::User {
        content: vec![UserContent::ToolResult(
            call.result(vec![ToolResultContent::text("-3")]),
        )],
    }]);
    assert_eq!(
        body(history)["messages"][0]["content"],
        json!([{"type":"text","text":"-3"}])
    );
}

#[test]
fn cohere_builder_request_serializes_documents_in_cohere_shape() {
    let request = CompletionRequest::new("What is glarb-glarb?").document(Document {
        id: "doc_1".to_owned(),
        text: "Definition of glarb-glarb: an ancient tool.".to_owned(),
        additional_props: HashMap::from([("source".to_owned(), "field-notes".to_owned())]),
    });
    let documents = &body(request)["documents"];
    assert_eq!(documents.as_array().map(Vec::len), Some(1));
    assert_eq!(documents[0]["id"], "doc_1");
    assert_eq!(
        documents[0],
        json!({"id":"doc_1","data":{"text":"Definition of glarb-glarb: an ancient tool.","source":"field-notes"}})
    );
}

#[test]
fn tool_choice_serializes_as_a_bare_cohere_string() {
    for (choice, expected) in [
        (ToolChoice::Required, "REQUIRED"),
        (ToolChoice::None, "NONE"),
    ] {
        assert_eq!(
            body(
                CompletionRequest::new("hello")
                    .tool(tool("subtract"))
                    .tool_choice(choice)
            )["tool_choice"],
            expected
        );
    }
}

#[test]
fn unsupported_tool_choices_are_rejected_before_the_request_is_sent() {
    for choice in [
        ToolChoice::Auto,
        ToolChoice::Specific {
            function_names: vec![ToolName::new("subtract").expect("name")],
        },
    ] {
        let error = ProviderError::from(
            native()
                .body(
                    &CompletionRequest::new("hello").tool_choice(choice),
                    Mode::Unary,
                )
                .expect_err("Cohere has no encoding"),
        );
        assert!(matches!(error, ProviderError::Request(_)));
    }
}

#[tokio::test]
async fn required_tool_choice_without_tools_is_rejected_before_the_request_is_sent() {
    let http = RecordingHttpClient::new("{}");
    let model = Model::new(
        CohereConfig::new("test-key").completion(MODEL),
        http.clone(),
    );
    let error = model
        .call(CompletionRequest::new("hello").tool_choice(ToolChoice::Required))
        .await
        .expect_err("required needs tools");
    assert!(matches!(error, ProviderError::Request(_)));
    let message = error.to_string();
    assert!(
        message.contains("at least one tool") && message.contains("REQUIRED"),
        "{error:?}"
    );
    assert!(http.requests().is_empty());
}

#[test]
fn required_tool_choice_accepts_raw_tools_from_additional_params() {
    let request = CompletionRequest::new("hello").tool_choice(ToolChoice::Required).additional_params(json!({"tools":[{"type":"function","function":{"name":"ping","description":"Return pong","parameters":{"type":"object","properties":{}}}}]}));
    let body = body(request);
    assert_eq!(body["tool_choice"], json!("REQUIRED"));
    assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
}

#[test]
fn max_tokens_is_forwarded_and_omitted_when_unset() {
    assert_eq!(
        body(CompletionRequest::new("hello").max_tokens(64))["max_tokens"],
        json!(64)
    );
    assert!(
        body(CompletionRequest::new("hello"))
            .get("max_tokens")
            .is_none()
    );
}

#[test]
fn tool_choice_is_omitted_when_unset() {
    assert!(
        body(CompletionRequest::new("hello"))
            .get("tool_choice")
            .is_none()
    );
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let http =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let error = Model::new(CohereConfig::new("test-key").completion(MODEL), http)
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("non-success");
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let name = message::ToolName::new("test").expect("tool name");
    let id = message::CallId::Local(message::LocalCallId::new());
    let local = message::ToolCall::new(id, message::ToolFunction::new(name.clone(), json!({})));
    let genuine =
        message::ToolCall::from_wire("tool-0", message::ToolFunction::new(name, json!({})));
    let result = |call: &message::ToolCall| {
        let mut result = call.result(vec![]);
        result.name = message::ToolName::new("possibly_repaired").expect("tool name");
        message::Message::User {
            content: vec![message::UserContent::ToolResult(result)],
        }
    };
    let assistant = |call: &message::ToolCall| {
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ]))
    };
    let request = CompletionRequest::from(vec![
        message::Message::system("system"),
        assistant(&local),
        assistant(&genuine),
        result(&genuine),
        message::Message::assistant("intervening text"),
        result(&local),
        assistant(&local),
        result(&local),
    ])
    .max_tokens(128);
    let value = body(request);
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for message in value["messages"].as_array().expect("messages") {
        if let Some(items) = message["tool_calls"].as_array() {
            calls.extend(
                items
                    .iter()
                    .map(|call| call["id"].as_str().expect("call ID").to_owned()),
            );
        }
        if message["role"] == "tool" {
            results.push(
                message["tool_call_id"]
                    .as_str()
                    .expect("answered ID")
                    .to_owned(),
            );
        }
    }
    assert_eq!(calls.len(), 3, "{value}");
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()],
        "{value}"
    );
    assert_eq!(calls[1], "tool-0");
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "{value}"
    );
}

#[test]
fn cohere_chat_refuses_namespaced_and_custom_calls() {
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
        let error: ProviderError = native()
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
                if *wire == COHERE_CHAT_WIRE && name.as_str() == "tool")
            );
        } else {
            assert!(
                matches!(cause, message::UnrepresentableToolCall::Namespace { wire, name, namespace }
                if *wire == COHERE_CHAT_WIRE && name.as_str() == "tool" && namespace.is_empty())
            );
        }
    }
}

#[test]
fn native_refuses_carried_provider_items_before_adapting_failed_history() {
    use crate::wire::{Operation, Wire};
    let wire = CohereConfig::new("test-key").completion(MODEL);
    for issuer in [None, Some("cohere"), Some("foreign/Responses")] {
        let item = message::ProviderItem {
            payload: message::CarriedPayload::new(
                json!({"type":"future_item", "payload": [1, 2]}),
                issuer.map(str::to_owned),
            ),
        };
        let turn =
            message::AssistantMessage::new(vec![message::AssistantContent::ProviderItem(item)])
                .with_stop(message::StopReason::Error("failed turn".to_owned()));
        let request = CompletionRequest::from(vec![
            message::Message::Assistant(turn),
            message::Message::user("continue"),
        ]);
        let error = crate::operation::Completion::prepare(request, &wire.describe())
            .expect_err("the original failed turn's item must be refused before adaptation");
        let ProviderError::Request(cause) = error else {
            panic!("the refusal is a typed request failure");
        };
        let cause = cause
            .downcast_ref::<message::UnreplayableProviderItem>()
            .expect("the typed cause survives");
        assert_eq!(cause.issuer.as_deref(), issuer);
        assert_eq!(cause.item_type.as_deref(), Some("future_item"));
    }
}

#[test]
fn native_refuses_carried_opaque_text_before_adapting_failed_history() {
    for issuer in [None, Some("cohere"), Some("foreign/Responses")] {
        let part = message::CarriedText::new(
            message::CarriedTextContent::Opaque(String::new()),
            message::CarriedPayload::new(
                json!({"type":"server_tool_use", "id":"hosted"}),
                issuer.map(str::to_owned),
            ),
        );
        let turn =
            message::AssistantMessage::new(vec![message::AssistantContent::CarriedText(part)])
                .with_stop(message::StopReason::Error("failed turn".to_owned()));
        let request =
            CompletionRequest::from(vec![Message::Assistant(turn), Message::user("continue")]);
        let wire = CohereConfig::new("test-key").completion(MODEL);
        let error = crate::operation::Completion::prepare(request, &wire.describe())
            .expect_err("original carried content is refused");
        let ProviderError::Request(cause) = error else {
            panic!("typed request refusal")
        };
        assert!(
            cause
                .downcast_ref::<message::UnrepresentableOpaqueContent>()
                .is_some()
        );
    }
}
