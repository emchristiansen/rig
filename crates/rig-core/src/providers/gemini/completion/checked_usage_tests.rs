use super::*;
use crate::completion::{CompletionResponse, FinishReason, Usage};
use crate::wire::{Mode, WireFrame};

fn reply(usage: Option<Value>) -> Value {
    let mut reply = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP", "index": 0}],
        "modelVersion": "gemini-2.5-flash", "responseId": "usage-190"
    });
    if let Some(usage) = usage {
        reply["usageMetadata"] = usage;
    }
    reply
}

fn decoded(reply: &Value, mode: Mode) -> Result<CompletionResponse, ProviderError> {
    let wire = super::super::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    crate::test_utils::history::decode(&wire, mode, [WireFrame::Text(reply.to_string())])
}

#[test]
fn rest_unary_and_streamed_finish_include_tool_input_and_reasoning_output() {
    let raw = reply(Some(
        json!({"promptTokenCount": 100, "toolUsePromptTokenCount": 75,
        "cachedContentTokenCount": 40, "candidatesTokenCount": 30,
        "thoughtsTokenCount": 34, "totalTokenCount": 239}),
    ));
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = decoded(&raw, mode).expect("the real REST decoder completes");
        assert_eq!(response.usage.input_tokens, Some(175));
        assert_eq!(response.usage.tool_use_prompt_tokens, Some(75));
        assert_eq!(response.usage.cached_input_tokens, Some(40));
        assert_eq!(response.usage.output_tokens, Some(64));
        assert_eq!(response.usage.reasoning_tokens, Some(34));
        assert_eq!(response.usage.total_tokens, Some(239));
        assert_eq!(response.usage.cache_creation_input_tokens, None);
        assert_eq!(response.choice.len(), 1);
        assert_eq!(response.text(), "hi");
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.response_id(), Some("usage-190"));
        assert_eq!(response.model(), Some("gemini-2.5-flash"));
        assert_eq!(response.raw, raw);
    }
}

#[test]
fn rest_missing_candidate_and_base_counts_preserve_their_existing_limits() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let raw = reply(Some(json!({"toolUsePromptTokenCount": 75,
            "cachedContentTokenCount": 40, "thoughtsTokenCount": 34})));
        let response = decoded(&raw, mode).expect("missing counts stay optional");
        assert_eq!(response.usage.input_tokens, Some(75));
        assert_eq!(response.usage.output_tokens, None);
        assert_eq!(response.usage.total_tokens, Some(0));
        assert_eq!(response.usage.tool_use_prompt_tokens, Some(75));
        assert_eq!(response.usage.cached_input_tokens, Some(40));
        assert_eq!(response.usage.reasoning_tokens, Some(34));
        assert_eq!(response.raw, raw);

        let raw = reply(Some(
            json!({"candidatesTokenCount": 30, "thoughtsTokenCount": 34,
            "totalTokenCount": 7}),
        ));
        let response = decoded(&raw, mode).expect("reported total is retained");
        assert_eq!(response.usage.input_tokens, Some(0));
        assert_eq!(response.usage.output_tokens, Some(64));
        assert_eq!(response.usage.total_tokens, Some(7));
        assert_eq!(response.usage.reasoning_tokens, Some(34));
        assert_eq!(response.usage.tool_use_prompt_tokens, None);
        assert_eq!(response.raw, raw);

        let raw = reply(Some(json!({})));
        let response = decoded(&raw, mode).expect("empty usage keeps base zero defaults");
        assert_eq!(response.usage.input_tokens, Some(0));
        assert_eq!(response.usage.output_tokens, None);
        assert_eq!(response.usage.total_tokens, Some(0));
        assert_eq!(response.usage.cached_input_tokens, None);
        assert_eq!(response.usage.tool_use_prompt_tokens, None);
        assert_eq!(response.usage.reasoning_tokens, None);
        assert_eq!(response.raw, raw);

        let raw = reply(None);
        let response = decoded(&raw, mode).expect("absent usage stays wholly absent");
        assert_eq!(response.usage, Usage::default());
        assert_eq!(response.raw, raw);
    }
}

#[test]
fn rest_negative_cast_addition_overflows_are_typed_response_errors() {
    for mode in [Mode::Unary, Mode::Streaming] {
        for (usage, message) in [
            (
                json!({"promptTokenCount": -1, "toolUsePromptTokenCount": 1}),
                "Gemini input token count overflow",
            ),
            (
                json!({"candidatesTokenCount": -1, "thoughtsTokenCount": 1}),
                "Gemini output token count overflow",
            ),
        ] {
            let raw = reply(Some(usage));
            let error = decoded(&raw, mode).expect_err("the existing signed cast reaches overflow");
            assert!(matches!(error, ProviderError::Response(ref actual) if actual == message));
        }
        let raw = reply(Some(json!({"thoughtsTokenCount": -1})));
        let response = decoded(&raw, mode).expect("absent candidate counts never add reasoning");
        assert_eq!(response.usage.output_tokens, None);
        assert_eq!(response.usage.reasoning_tokens, Some(u64::MAX));
        assert_eq!(response.raw, raw);
    }
}

#[test]
fn public_usage_helper_retains_its_raw_counter_and_signed_cast_behavior() {
    let raw = json!({"promptTokenCount": 100, "toolUsePromptTokenCount": 75,
        "cachedContentTokenCount": 40, "candidatesTokenCount": 30,
        "thoughtsTokenCount": 34, "totalTokenCount": 239});
    assert_eq!(
        checked_usage_of(&raw).expect("canonical counts fit").cost,
        None
    );
    let usage = usage_of(&raw);
    assert_eq!(usage.input_tokens, Some(100));
    assert_eq!(usage.output_tokens, Some(30));
    assert_eq!(usage.cached_input_tokens, Some(40));
    assert_eq!(usage.tool_use_prompt_tokens, Some(75));
    assert_eq!(usage.reasoning_tokens, Some(34));
    assert_eq!(usage.total_tokens, Some(239));
    let cast = usage_of(&json!({"promptTokenCount": -1, "candidatesTokenCount": -1}));
    assert_eq!(cast.input_tokens, Some(u64::MAX));
    assert_eq!(cast.output_tokens, Some(u64::MAX));
}
