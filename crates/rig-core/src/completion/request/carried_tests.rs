use serde_json::json;

use super::{
    CompletionRequest, CompletionResponse, FinishReason, ProviderMessageId, ResponseIdentity, Usage,
};
use crate::error::{ErrorKind, ProviderError};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, CustomToolCall, Message, Origin, StopReason,
    ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("tool name")
}

fn response(choice: Vec<AssistantContent>) -> CompletionResponse {
    let mut origin = Origin::new("test.api", "provider", "requested-model");
    origin.response_id = Some("response_1".to_owned());
    origin.response_model = Some("reported-model".to_owned());
    let mut response = CompletionResponse::new(
        choice,
        Usage::new()
            .input_tokens(3)
            .output_tokens(2)
            .total_tokens(5),
        origin,
        json!({"id": "raw_item_id", "future": [true, null, {"kept": "exact"}]}),
    );
    response.provider_request_id = Some("request_1".to_owned());
    response
}

fn custom(id: &str, raw: &str) -> CustomToolCall {
    CustomToolCall::new(CallId::from_wire(id), name("patch"), raw)
        .with_namespace(Some(String::new()))
}

#[test]
fn captured_response_headers_round_trip_without_changing_origin_or_identity() {
    let response = response(vec![AssistantContent::text("answer")])
        .with_message_id(ProviderMessageId::new("message_1").expect("message id"))
        .with_provider_response_headers(
            [
                ("x-codex-limit".to_owned(), " 12, 3\t".to_owned()),
                ("x-future".to_owned(), "\u{fffd}, exact".to_owned()),
                ("x-present-empty".to_owned(), String::new()),
            ]
            .into(),
        );
    let expected_identity = ResponseIdentity {
        message_id: ProviderMessageId::new("message_1"),
        response_id: Some("response_1".to_owned()),
        provider_request_id: Some("request_1".to_owned()),
    };
    let encoded = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(
        encoded["provider_response_headers"],
        json!({"x-codex-limit": " 12, 3\t", "x-future": "\u{fffd}, exact", "x-present-empty": ""}),
    );
    let decoded = serde_json::from_value::<CompletionResponse>(encoded.clone())
        .expect("deserialize captured response");
    assert_eq!(decoded, response);
    assert_eq!(decoded.origin, response.origin);
    assert_eq!(decoded.identity(), expected_identity);
    assert_eq!(decoded.provider(), "provider");
    assert_eq!(decoded.model(), Some("reported-model"));
    assert_eq!(decoded.origin.model, "requested-model");
    assert_eq!(decoded.raw, response.raw);
    assert_eq!(
        serde_json::to_value(&decoded).expect("serialize again"),
        encoded
    );
    let Message::Assistant(turn) = decoded.message().expect("a nonempty turn") else {
        panic!("assistant turn");
    };
    assert_eq!(turn.origin, Some(response.origin));
}

#[test]
fn an_empty_header_capture_is_omitted_and_missing_or_empty_capture_loads_empty() {
    let response = response(vec![AssistantContent::text("answer")]);
    assert!(response.provider_response_headers.is_empty());
    let encoded = serde_json::to_value(&response).expect("serialize response");
    assert!(encoded.get("provider_response_headers").is_none());
    let absent = serde_json::from_value::<CompletionResponse>(encoded.clone())
        .expect("missing capture defaults to empty");
    assert_eq!(absent, response);
    let mut encoded = encoded;
    encoded["provider_response_headers"] = json!({});
    let explicit = serde_json::from_value::<CompletionResponse>(encoded)
        .expect("an explicitly empty capture loads");
    assert!(explicit.provider_response_headers.is_empty());
    assert_eq!(explicit.origin, response.origin);
    assert_eq!(explicit.identity(), response.identity());
    assert_eq!(explicit, response);
    assert!(
        serde_json::to_value(explicit)
            .expect("serialize empty capture")
            .get("provider_response_headers")
            .is_none()
    );
}

#[test]
fn custom_only_output_reconciles_stop_through_setters_deserialization_and_history() {
    let choice = vec![AssistantContent::CustomToolCall(custom(
        "custom_1",
        " \n{raw}\t ",
    ))];
    let plain = response(choice.clone()).with_finish_reason(FinishReason::Stop);
    let optional = response(choice.clone()).with_optional_finish_reason(Some(FinishReason::Stop));
    for response in [&plain, &optional] {
        assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
        assert_eq!(response.stop(), StopReason::ToolUse);
        let Message::Assistant(turn) = response.message().expect("custom turn") else {
            panic!("assistant turn");
        };
        assert_eq!(turn.stop, Some(StopReason::ToolUse));
        assert_eq!(turn.content, choice);
    }
    let mut encoded = serde_json::to_value(&plain).expect("serialize custom response");
    encoded["finish_reason"] = json!("stop");
    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize a custom stop");
    assert_eq!(decoded.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(decoded.stop(), StopReason::ToolUse);
    assert_eq!(decoded.choice, choice);
    for response in [
        response(choice.clone()),
        plain.with_optional_finish_reason(None),
    ] {
        assert_eq!(response.finish_reason(), None);
        assert_eq!(response.stop(), StopReason::ToolUse);
    }
    for reason in [
        FinishReason::Length,
        FinishReason::ContentFilter,
        FinishReason::Other("future_stop".to_owned()),
    ] {
        let response = response(choice.clone()).with_finish_reason(reason.clone());
        assert_eq!(response.finish_reason(), Some(reason));
    }
}

#[test]
fn custom_call_iteration_retains_order_raw_whitespace_and_exact_namespaces() {
    let first = custom("custom_1", " \n{\"x\":1}\t ");
    let second = CustomToolCall::new(CallId::from_wire("custom_2"), name("patch"), "plain\0raw")
        .with_namespace(Some("functions".to_owned()));
    let function = ToolCall::from_wire("function_1", ToolFunction::new(name("lookup"), json!({})));
    let response = response(vec![
        AssistantContent::text("before"),
        AssistantContent::CustomToolCall(first.clone()),
        AssistantContent::ToolCall(function.clone()),
        AssistantContent::CustomToolCall(second.clone()),
        AssistantContent::text("after"),
    ]);
    assert_eq!(
        response.custom_tool_calls().collect::<Vec<_>>(),
        [&first, &second]
    );
    assert_eq!(response.tool_calls().collect::<Vec<_>>(), [&function]);
    assert_eq!(response.text(), "beforeafter");
    assert_eq!(
        response.custom_tool_calls().next().expect("first").input,
        " \n{\"x\":1}\t "
    );
    let encoded = serde_json::to_value(&response).expect("serialize calls");
    let decoded = serde_json::from_value::<CompletionResponse>(encoded).expect("read calls");
    assert_eq!(
        decoded.custom_tool_calls().collect::<Vec<_>>(),
        [&first, &second]
    );
    assert_eq!(decoded.choice, response.choice);
}

#[test]
fn empty_function_and_custom_tool_results_report_exact_role_indices_and_name() {
    let function = ToolCall::from_wire("function_1", ToolFunction::new(name("lookup"), json!({})));
    let custom = custom("custom_1", "raw");
    for (call, empty, populated, expected_name) in [
        (
            AssistantContent::ToolCall(function.clone()),
            function.result(Vec::new()),
            function.result(vec![ToolResultContent::text("")]),
            "lookup",
        ),
        (
            AssistantContent::CustomToolCall(custom.clone()),
            custom.result(Vec::new()),
            custom.result(vec![ToolResultContent::text("")]),
            "patch",
        ),
    ] {
        let earlier = populated.clone();
        let mut request = CompletionRequest::from(vec![
            Message::user("hello"),
            Message::Assistant(AssistantMessage::new(vec![call])),
            Message::User {
                content: vec![
                    UserContent::text("context"),
                    UserContent::ToolResult(earlier),
                    UserContent::ToolResult(empty),
                ],
            },
        ]);
        let error = request
            .validate_message_content()
            .expect_err("block-less result is rejected");
        assert!(matches!(&error, ProviderError::Request(_)));
        assert_eq!(error.kind(), ErrorKind::Request);
        assert_eq!(
            error.to_string(),
            format!(
                "RequestError: tool result for `{expected_name}` at index 2 of the user message at index 2 has no content; providers reject empty content blocks"
            )
        );
        let Message::User { content } = &mut request.chat_history[2] else {
            panic!("result message");
        };
        content[2] = UserContent::ToolResult(populated);
        assert!(
            request.validate_message_content().is_ok(),
            "one empty text block is content"
        );
        let encoded = serde_json::to_value(&request).expect("serialize the result kind");
        let decoded =
            serde_json::from_value::<CompletionRequest>(encoded).expect("read the result kind");
        assert!(decoded.validate_message_content().is_ok());
    }
}
