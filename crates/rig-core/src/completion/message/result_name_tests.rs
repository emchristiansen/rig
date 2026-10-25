use super::*;
use serde_json::json;

fn function(id: CallId, name: &str) -> ToolCall {
    ToolCall::new(
        id,
        ToolFunction::new(ToolName::new(name).unwrap(), json!({})),
    )
}

fn assistant(calls: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage::new(calls))
}

fn infer(
    prefix: &[Message],
    id: CallId,
    kind: AnsweredToolCall,
) -> Result<ToolResult, ToolResultNameError> {
    ToolResult::from_history(
        prefix,
        id,
        kind,
        vec![ToolResultContent::text("out")],
        false,
    )
}

// F internal/tests.rs:46 paired-name contract, moved to typed ingestion.
#[test]
fn empty_names_resolve_from_the_paired_call() {
    let first = function(CallId::from_wire("toolu_1"), "get_weather");
    let second = function(CallId::from_wire("toolu_2"), "get_time");
    let mut prefix = vec![Message::from(first.clone())];
    let first_result = infer(&prefix, first.id.clone(), AnsweredToolCall::Function).unwrap();
    prefix.push(Message::from(first_result.clone()));
    prefix.push(Message::from(second.clone()));
    let second_result = infer(&prefix, second.id.clone(), AnsweredToolCall::Function).unwrap();
    assert_eq!(
        [first_result.name.as_str(), second_result.name.as_str()],
        ["get_weather", "get_time"]
    );
}

// F's unmatched empty name is the ruled A79 typed refusal.
#[test]
fn an_unmatched_result_returns_the_typed_missing_identity() {
    let prefix = vec![Message::from(function(
        CallId::from_wire("toolu_1"),
        "get_weather",
    ))];
    let missing = CallId::from_wire("toolu_9");
    assert_eq!(
        infer(&prefix, missing.clone(), AnsweredToolCall::Function),
        Err(ToolResultNameError::Missing(missing))
    );
}

#[test]
fn an_established_name_is_never_overwritten() {
    let call = function(CallId::from_wire("toolu_1"), "add");
    let mut result = call.result(vec![ToolResultContent::text("3")]);
    result.name = ToolName::new("sum").unwrap();
    let prefix = vec![Message::from(call.clone()), Message::from(result.clone())];
    assert_eq!(result.name.as_str(), "sum");
    assert_eq!(
        infer(&prefix, call.id.clone(), AnsweredToolCall::Function),
        Err(ToolResultNameError::Missing(call.id))
    );
    let Message::User { content } = &prefix[1] else {
        panic!("result message")
    };
    let UserContent::ToolResult(kept) = &content[0] else {
        panic!("result")
    };
    assert_eq!(kept.name.as_str(), "sum");
}

#[test]
fn a_handle_only_result_resolves_from_an_id_less_call() {
    let call = function(CallId::Local(LocalCallId::new()), "lookup");
    let result = infer(
        &[Message::from(call.clone())],
        call.id.clone(),
        AnsweredToolCall::Function,
    )
    .unwrap();
    assert_eq!(result.name.as_str(), "lookup");
    assert_eq!(result.call, call.id);
    assert!(result.call.provider().is_none());
}

#[test]
fn a_named_duplicate_result_consumes_only_its_matching_call() {
    let id = CallId::from_wire("same");
    let alpha = function(id.clone(), "alpha");
    let beta = function(id.clone(), "beta");
    let beta_result = beta.result(vec![ToolResultContent::text("out")]);
    let prefix = vec![
        assistant(vec![
            AssistantContent::ToolCall(alpha),
            AssistantContent::ToolCall(beta),
        ]),
        Message::from(beta_result.clone()),
    ];
    let result = infer(&prefix, id, AnsweredToolCall::Function).unwrap();
    assert_eq!(
        [beta_result.name.as_str(), result.name.as_str()],
        ["beta", "alpha"]
    );
}

#[test]
fn name_resolution_preserves_identity_namespaces_and_turn_scope() {
    let generated = CallId::Local(LocalCallId::new());
    let explicit = CallId::from_wire(generated.wire().into_owned());
    assert_ne!(generated, explicit);
    let mut prefix = vec![assistant(vec![
        AssistantContent::ToolCall(function(generated.clone(), "alpha")),
        AssistantContent::ToolCall(function(explicit.clone(), "beta")),
    ])];
    let beta = infer(&prefix, explicit.clone(), AnsweredToolCall::Function).unwrap();
    prefix.push(Message::from(beta.clone()));
    let alpha = infer(&prefix, generated.clone(), AnsweredToolCall::Function).unwrap();
    prefix.push(Message::from(alpha.clone()));
    prefix.push(Message::from(function(generated.clone(), "gamma")));
    let gamma = infer(&prefix, generated.clone(), AnsweredToolCall::Function).unwrap();
    assert_eq!(
        [beta.name.as_str(), alpha.name.as_str(), gamma.name.as_str()],
        ["beta", "alpha", "gamma"]
    );
    assert_eq!(beta.call, explicit);
    assert_eq!(alpha.call, generated);
    assert_eq!(gamma.call, generated);
}

#[test]
fn name_resolution_does_not_consult_future_calls() {
    let id = CallId::Local(LocalCallId::new());
    let mut history = vec![Message::user("before any call")];
    let result_position = history.len();
    history.push(Message::from(function(id.clone(), "future")));
    assert_eq!(
        infer(
            &history[..result_position],
            id.clone(),
            AnsweredToolCall::Function
        ),
        Err(ToolResultNameError::Missing(id))
    );
}

#[test]
fn name_resolution_retains_outstanding_calls_across_assistant_text() {
    let id = CallId::Local(LocalCallId::new());
    let prefix = vec![
        Message::from(function(id.clone(), "lookup")),
        Message::assistant("waiting"),
    ];
    assert_eq!(
        infer(&prefix, id, AnsweredToolCall::Function)
            .unwrap()
            .name
            .as_str(),
        "lookup"
    );
}

#[test]
fn empty_result_names_resolve_from_custom_calls() {
    let call = CustomToolCall::new(
        CallId::from_wire("call_1"),
        ToolName::new("apply_patch").unwrap(),
        "raw",
    );
    let prefix = vec![Message::from(call.clone())];
    let result = infer(&prefix, call.id.clone(), AnsweredToolCall::Custom).unwrap();
    assert_eq!(result.name.as_str(), "apply_patch");
    assert_eq!(result.call, call.id);
    assert_eq!(result.answers, AnsweredToolCall::Custom);
}

#[test]
fn ambiguous_outstanding_calls_return_the_typed_identity() {
    let id = CallId::from_wire("same");
    let prefix = vec![assistant(vec![
        AssistantContent::ToolCall(function(id.clone(), "alpha")),
        AssistantContent::ToolCall(function(id.clone(), "beta")),
    ])];
    assert_eq!(
        infer(&prefix, id.clone(), AnsweredToolCall::Function),
        Err(ToolResultNameError::Ambiguous(id))
    );
}

#[test]
fn result_kind_matches_and_consumes_only_the_same_call_kind() {
    let id = CallId::from_wire("shared");
    let function = function(id.clone(), "function");
    let custom = CustomToolCall::new(id.clone(), ToolName::new("custom").unwrap(), "raw");
    let mut prefix = vec![assistant(vec![
        AssistantContent::ToolCall(function),
        AssistantContent::CustomToolCall(custom),
    ])];
    let result = infer(&prefix, id.clone(), AnsweredToolCall::Custom).unwrap();
    assert_eq!(result.name.as_str(), "custom");
    prefix.push(Message::from(result));
    assert_eq!(
        infer(&prefix, id.clone(), AnsweredToolCall::Function)
            .unwrap()
            .name
            .as_str(),
        "function"
    );
    assert_eq!(
        infer(&prefix, id.clone(), AnsweredToolCall::Custom),
        Err(ToolResultNameError::Missing(id))
    );
}

#[test]
fn a_prior_result_that_answers_no_outstanding_call_consumes_nothing() {
    let call = function(CallId::from_wire("one"), "lookup");
    let orphan = function(CallId::from_wire("never_called"), "lookup")
        .result(vec![ToolResultContent::text("stray")]);
    let prefix = vec![Message::from(call.clone()), Message::from(orphan)];
    assert_eq!(
        infer(&prefix, call.id.clone(), AnsweredToolCall::Function),
        Ok(call.result(vec![ToolResultContent::text("out")]))
    );
}

#[test]
fn a_prior_result_matching_several_same_named_calls_consumes_none_of_them() {
    let id = CallId::from_wire("same");
    let first = function(id.clone(), "lookup");
    let second = function(id.clone(), "lookup");
    let prefix = vec![
        assistant(vec![
            AssistantContent::ToolCall(first.clone()),
            AssistantContent::ToolCall(second),
        ]),
        Message::from(first.result(vec![ToolResultContent::text("out")])),
    ];
    assert_eq!(
        infer(&prefix, id.clone(), AnsweredToolCall::Function),
        Err(ToolResultNameError::Ambiguous(id))
    );
}

#[test]
fn consumed_identity_cannot_supply_a_second_result_name() {
    let call = function(CallId::from_wire("one"), "lookup");
    let prefix = vec![
        Message::from(call.clone()),
        Message::from(call.result(vec![ToolResultContent::text("out")])),
    ];
    assert_eq!(
        infer(&prefix, call.id.clone(), AnsweredToolCall::Function),
        Err(ToolResultNameError::Missing(call.id))
    );
}

#[test]
fn known_name_bytes_and_serde_stay_unchanged() {
    let call = function(CallId::from_wire("toolu_1"), "add");
    let mut named = call.result(vec![ToolResultContent::text("out")]);
    named.name = ToolName::new("sum").unwrap();
    let bytes = r#"{"call":{"provider":"toolu_1"},"name":"sum","answers":"function","content":[{"type":"text","text":"out"}]}"#;
    assert_eq!(serde_json::to_string(&named).unwrap(), bytes);
    assert_eq!(serde_json::from_str::<ToolResult>(bytes).unwrap(), named);
    let inferred = infer(
        &[Message::from(call.clone())],
        call.id.clone(),
        AnsweredToolCall::Function,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string(&inferred).unwrap(),
        serde_json::to_string(&call.result(vec![ToolResultContent::text("out")])).unwrap()
    );
}

#[test]
fn ingested_nameless_results_resolve_their_name_before_request_assembly() {
    use crate::wire::{Mode, Wire};
    let call = ToolCall::from_wire(
        "toolu_abc",
        ToolFunction::new(
            ToolName::new("get_weather").unwrap(),
            json!({"city":"Paris"}),
        ),
    );
    let mut history = vec![Message::user("weather?"), Message::from(call.clone())];
    let result = ToolResult::from_history(
        &history,
        call.id.clone(),
        AnsweredToolCall::Function,
        vec![ToolResultContent::text("sunny")],
        false,
    )
    .unwrap();
    assert_eq!(result.call, call.id);
    history.push(Message::from(result));
    let mut request = crate::completion::CompletionRequest::new("weather?");
    request.chat_history = history;
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    let encoded = wire.encode(request, Mode::Unary).unwrap();
    let body = crate::test_utils::json_body(&encoded.request);
    let response_names: Vec<_> = body["contents"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|content| content["parts"].as_array().unwrap())
        .filter_map(|part| {
            part.get("functionResponse")
                .map(|response| response["name"].as_str().unwrap())
        })
        .collect();
    assert_eq!(response_names, ["get_weather"]);
}

#[test]
fn known_repaired_name_encoding_stays_unchanged() {
    use crate::wire::{Mode, Wire};
    let call = function(CallId::from_wire("toolu_1"), "add");
    let mut result = call.result(vec![ToolResultContent::text("3")]);
    result.name = ToolName::new("sum").unwrap();
    let mut request = crate::completion::CompletionRequest::new("question");
    request.chat_history.push(Message::from(call));
    request.chat_history.push(Message::from(result));
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    let encoded = wire.encode(request, Mode::Unary).unwrap();
    let body = crate::test_utils::json_body(&encoded.request);
    let response = body["contents"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|content| content["parts"].as_array().unwrap())
        .find_map(|part| part.get("functionResponse"))
        .unwrap();
    assert_eq!(response["name"], "sum");
    assert_eq!(response["id"], "toolu_1");
    assert_eq!(response["response"], json!({"result":"3"}));
}

#[test]
fn a_nameless_error_result_keeps_its_incoming_error_flag() {
    let call = function(CallId::from_wire("failed_call"), "lookup");
    let content = vec![ToolResultContent::text("provider failure")];
    let result = ToolResult::from_history(
        &[Message::from(call.clone())],
        call.id.clone(),
        AnsweredToolCall::Function,
        content.clone(),
        true,
    )
    .unwrap();
    assert!(result.is_error);
    assert_eq!(result.call, call.id);
    assert_eq!(result.name.as_str(), "lookup");
    assert_eq!(result.content, content);
    assert_eq!(serde_json::to_value(&result).unwrap()["is_error"], true);
}
