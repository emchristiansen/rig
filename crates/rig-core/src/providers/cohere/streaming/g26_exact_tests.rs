use super::*;
use crate::completion::{AssistantContent, CompletionRequest, CompletionResponse, FinishReason};
use crate::driver::{Model, feed_frames};
use crate::providers::cohere::extension::CohereExt;
use crate::providers::cohere::{COMMAND_A_03_2025, CohereConfig, NativeChat};
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::{Mode, Wire, WireFrame};
use serde_json::{Value, json};

fn decoded(body: &str) -> CompletionResponse {
    feed_frames!(
        ChatDecoder::default(),
        document::ChatResponse::default(),
        "cohere",
        [WireFrame::Text(body.to_owned())]
    )
    .outcome
    .expect("the actual native reply folds")
}

fn native() -> NativeChat {
    NativeChat::new(CohereConfig::new("test-key"), COMMAND_A_03_2025)
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
    let response = decoded(json_data);
    assert_eq!(
        response.raw,
        serde_json::from_str::<Value>(json_data).expect("the original body is JSON")
    );
    assert_eq!(response.response_id(), Some("abc123"));
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    let extras = response
        .extras::<CohereExt>()
        .expect("Cohere produced the reply")
        .expect("actual current extras decode");
    let billed = extras.billed_units.expect("original billed units present");
    let tokens = extras.tokens.expect("original model tokens present");
    assert_eq!(billed.input_tokens.expect("billed input present"), 78.0);
    assert_eq!(billed.output_tokens.expect("billed output present"), 27.0);
    assert_eq!(tokens.input_tokens.expect("input present"), 1028.0);
    assert_eq!(tokens.output_tokens.expect("output present"), 63.0);
    let citations = response
        .raw
        .pointer("/message/citations")
        .and_then(Value::as_array);
    assert!(citations.is_none_or(Vec::is_empty));
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name.as_str(), "subtract");
    assert_eq!(calls[0].function.arguments_value(), json!({"x":5,"y":2}));
}

#[test]
fn usage_is_mapped_from_tokens_and_carries_cached_input() {
    let original_usage = r#"{
                "billed_units": {"input_tokens": 135, "output_tokens": 24},
                "cached_tokens": 112,
                "tokens": {"input_tokens": 1610, "output_tokens": 56}
            }"#;
    let usage: Value = serde_json::from_str(original_usage).expect("the original usage is JSON");
    let body = json!({"id":"fixture", "finish_reason":"COMPLETE",
        "message":{"role":"assistant","content":[{"type":"text","text":"fixture"}]},
        "usage": usage});
    let mapped = decoded(&body.to_string()).usage;
    assert_eq!(mapped.input_tokens, Some(1610));
    assert_eq!(mapped.output_tokens, Some(56));
    assert_eq!(mapped.total_tokens, Some(1666));
    assert_eq!(mapped.cached_input_tokens, Some(112));
}

#[tokio::test]
async fn response_usage_matches_the_canonical_mapping() {
    const BODY: &str = r#"{
                "id": "abc123",
                "finish_reason": "COMPLETE",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                "usage": {
                    "billed_units": {"input_tokens": 135, "output_tokens": 24},
                    "cached_tokens": 112,
                    "tokens": {"input_tokens": 1610, "output_tokens": 56}
                }
            }"#;
    let original: Value = serde_json::from_str(BODY).expect("the original body is JSON");
    let expected = crate::completion::Usage {
        input_tokens: Some(1610),
        output_tokens: Some(56),
        total_tokens: Some(1666),
        cached_input_tokens: Some(112),
        cost: Some(crate::completion::Cost::from_parts(
            135.0 * 2.5 / 1e6,
            24.0 * 10.0 / 1e6,
            0.0,
            0.0,
        )),
        ..Default::default()
    };
    let converted = Model::new(native(), RecordingHttpClient::new(BODY))
        .call(CompletionRequest::new("hello"))
        .await
        .expect("the exact original known-model reply folds");
    assert_eq!(converted.usage, expected);
    assert_eq!(converted.usage.input_tokens, Some(1610));
    assert_eq!(converted.usage.cached_input_tokens, Some(112));
    assert_eq!(
        converted.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("hi"))
    );
    assert_eq!(converted.raw, original);
}

#[test]
fn required_tool_choice_accepts_raw_tools_from_additional_params() {
    let mut request = CompletionRequest::new("hello");
    request.tool_choice = Some(crate::message::ToolChoice::Required);
    request.additional_params = Some(json!({
        "tools": [{"type":"function","function":{
            "name":"ping", "description":"Return pong",
            "parameters":{"type":"object","properties":{}}
        }}]
    }));
    let encoded = native()
        .encode(request, Mode::Unary)
        .expect("raw Cohere tools satisfy REQUIRED");
    let body = json_body(&encoded.request);
    assert_eq!(body["tool_choice"], json!("REQUIRED"));
    assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = Model::new(native(), http_client);
    let error = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("non-success status must fail");
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn required_tool_choice_without_tools_is_rejected_before_the_request_is_sent() {
    let http_client = RecordingHttpClient::new("{}");
    let model = Model::new(native(), http_client.clone());
    let mut request = CompletionRequest::new("hello");
    request.tool_choice = Some(crate::message::ToolChoice::Required);
    let error = model
        .call(request)
        .await
        .expect_err("REQUIRED without tools fails locally");
    let ProviderError::Request(reason) = &error else {
        panic!("a request refusal, got {error:?}");
    };
    assert_eq!(
        reason.to_string(),
        "Cohere mandatory tool choice has no effective tools"
    );
    assert!(
        http_client.requests().is_empty(),
        "invalid requests fail before the HTTP client"
    );
}

fn original_unrepresentable_turns() -> [(crate::message::Message, &'static str); 2] {
    use crate::message::{
        AssistantMessage, CallId, CustomToolCall, Message, ToolCall, ToolFunction, ToolName,
    };
    let name = |value: &str| ToolName::new(value).expect("the original nonempty name");
    [
        (
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                ToolCall::new(
                    CallId::from_wire("call_1"),
                    ToolFunction::new(name("add"), json!({"x":1}))
                        .with_namespace(Some("math".to_owned())),
                ),
            )])),
            "namespace",
        ),
        (
            Message::Assistant(AssistantMessage::new(vec![
                AssistantContent::CustomToolCall(CustomToolCall::new(
                    CallId::from_wire("call_2"),
                    name("apply_patch"),
                    r#"{"x": 1}"#,
                )),
            ])),
            "custom",
        ),
    ]
}

fn assert_original_typed_refusal(error: &ProviderError, kind: &str, wire: &str) {
    use crate::message::UnrepresentableToolCall;
    let ProviderError::Request(reason) = error else {
        panic!("a request refusal, got {error:?}");
    };
    let refusal = reason
        .downcast_ref::<UnrepresentableToolCall>()
        .expect("the complete typed refusal survives");
    match (kind, refusal) {
        (
            "namespace",
            UnrepresentableToolCall::Namespace {
                wire: got,
                name,
                namespace,
            },
        ) => {
            assert_eq!(
                (*got, name.as_str(), namespace.as_str()),
                (wire, "add", "math")
            );
        }
        ("custom", UnrepresentableToolCall::CustomCall { wire: got, name }) => {
            assert_eq!((*got, name.as_str()), (wire, "apply_patch"));
        }
        _ => panic!("expected the {kind} refusal on {wire}, got {refusal:?}"),
    }
}

#[test]
fn cohere_chat_refuses_namespaced_and_custom_calls() {
    for (turn, kind) in original_unrepresentable_turns() {
        let request = CompletionRequest::new(turn);
        let error = ProviderError::from(
            native()
                .encode(request, Mode::Unary)
                .expect_err("refused, not dropped"),
        );
        assert_original_typed_refusal(&error, kind, "Cohere native chat");
    }
}
