use super::*;
use crate::completion::{Converse, ConverseFrame};
use rig_core::completion::{CompletionResponse, Usage};
use rig_core::message::AssistantContent;
use rig_core::test_utils::history::decode;
use rig_core::wire::Mode;

fn decoded(content: Vec<Value>, usage: Option<Value>) -> Result<CompletionResponse, ProviderError> {
    let mut document = json!({
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
    });
    if let Some(usage) = usage {
        document["usage"] = usage;
    }
    decode(
        &Converse::new(""),
        Mode::Unary,
        [ConverseFrame::Whole(document)],
    )
}

fn text_output(text: &str, usage: Option<Value>) -> CompletionResponse {
    decoded(vec![json!({"text": text})], usage).expect("the original fixture decodes")
}

#[test]
fn provider_response_ext_text_response() {
    let out = text_output("hello world", None);
    assert_eq!(
        out.choice
            .first()
            .and_then(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .map(str::to_owned),
        Some("hello world".to_string())
    );
}

#[test]
fn provider_response_ext_usage_with_tokens() {
    let out = text_output(
        "x",
        Some(json!({"inputTokens": 100, "outputTokens": 50, "totalTokens": 150})),
    );
    let usage = out.usage;
    assert_eq!(usage.input_tokens, Some(100));
    assert_eq!(usage.output_tokens, Some(50));
    assert_eq!(usage.total_tokens, Some(150));
}

#[test]
fn provider_response_ext_usage_none_when_missing() {
    let out = text_output("x", None);
    assert!(!out.usage.is_reported());
}

#[test]
fn token_usage_delegates_to_provider_response_ext() {
    let out = text_output(
        "x",
        Some(json!({"inputTokens": 10, "outputTokens": 20, "totalTokens": 30})),
    );
    let mut expected = Usage::default();
    expected.input_tokens = Some(10);
    expected.output_tokens = Some(20);
    expected.total_tokens = Some(30);
    assert_eq!(out.usage, expected);
}

#[test]
fn aws_converse_output_preserves_parallel_tool_calls_in_completion_response() {
    let completion = decoded(vec![
        json!({"text": "preface"}),
        json!({"toolUse": {"toolUseId": "call_1", "name": "add", "input": {"x": 1, "y": 2}}}),
        json!({"toolUse": {"toolUseId": "call_2", "name": "subtract", "input": {"x": 4, "y": 3}}}),
    ], None).expect("conversion should succeed");
    let choice: Vec<_> = completion.choice.into_iter().collect();
    assert_eq!(choice.len(), 3);
    assert_eq!(choice[0], AssistantContent::Text("preface".into()));
    let AssistantContent::ToolCall(first_tool) = &choice[1] else {
        panic!("expected first tool call");
    };
    assert_eq!(
        first_tool.id.provider().map(|id| id.as_str()),
        Some("call_1")
    );
    assert_eq!(first_tool.function.name.as_str(), "add");
    assert_eq!(
        first_tool.function.arguments_value(),
        json!({"x": 1, "y": 2})
    );
    let AssistantContent::ToolCall(second_tool) = &choice[2] else {
        panic!("expected second tool call");
    };
    assert_eq!(
        second_tool.id.provider().map(|id| id.as_str()),
        Some("call_2")
    );
    assert_eq!(second_tool.function.name.as_str(), "subtract");
    assert_eq!(
        second_tool.function.arguments_value(),
        json!({"x": 4, "y": 3})
    );
}

#[test]
fn aws_content_block_to_assistant_content() {
    let content = decoded(vec![json!({"text": "text"})], None);
    assert!(content.is_ok());
    assert_eq!(
        content.unwrap().choice[0],
        AssistantContent::Text("text".into())
    );
}

/// The original assistant/text/EndTurn fixture through the current whole-frame fold.
#[test]
fn aws_converse_output_to_completion_response() {
    let completion = decoded(vec![json!({"text": "txt"})], None);
    assert!(completion.is_ok());
    let completion = completion.expect("the original whole-frame fixture decodes");
    assert_eq!(
        completion.choice,
        vec![AssistantContent::Text("txt".into())]
    );
}
