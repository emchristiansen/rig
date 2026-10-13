//! Exact received missing-ID fixtures over the current provider decoders.

use crate::completion::CompletionResponse;
use crate::operation::Completion;
use crate::wire::{Mode, Wire, WireFrame};
use serde_json::{Value, json};

fn fold_document<W: Wire<Op = Completion, Frame = WireFrame>>(
    wire: &W,
    body: &Value,
) -> CompletionResponse {
    crate::test_utils::history::decode(wire, Mode::Unary, [WireFrame::Text(body.to_string())])
        .expect("the original received fixture decodes")
}

fn assert_normalization(convert: impl Fn() -> CompletionResponse) {
    // G44: fresh local custody is checked on both decodes, without claiming
    // that independent decodes issue the same local UUIDs.
    for response in [convert(), convert()] {
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls
                .iter()
                .map(|call| &call.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        assert_eq!(calls[1].id.wire(), "tool-0");
        assert_eq!(calls[1].id.provider().map(|id| id.as_str()), Some("tool-0"));
        for index in [0, 2] {
            assert!(calls[index].id.is_local());
            assert!(calls[index].id.provider().is_none());
        }
        for (index, call) in calls.iter().enumerate() {
            assert_eq!(call.function.arguments_value(), json!({"n": index}));
        }
    }
}

fn chat_wire() -> Value {
    json!({"id":"response", "object":"chat.completion", "created":0, "model":"test",
    "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"prefix", "tool_calls":
        (0..3).map(|i|json!({"id":if i==1 {"tool-0"} else {""},"type":"function","function":{"name":"same","arguments":json!({"n":i}).to_string()}})).collect::<Vec<_>>()
    }}]})
}

#[test]
fn openai_chat_missing_ids_and_later_explicit_collision() {
    let wire = crate::providers::openai::OpenAIConfig::with_key(
        &crate::providers::openai::wire::OPENAI,
        "test-key",
    )
    .chat("test");
    assert_normalization(|| fold_document(&wire, &chat_wire()));
}

#[test]
fn openrouter_missing_ids_and_later_explicit_collision() {
    let wire = crate::providers::openai::OpenAIConfig::with_key(
        &crate::providers::openai::wire::OPENROUTER,
        "test-key",
    )
    .chat("test");
    assert_normalization(|| fold_document(&wire, &chat_wire()));
}

#[test]
fn anthropic_missing_ids_and_later_explicit_collision() {
    let wire = json!({"type":"message","id":"response","model":"test","role":"assistant","stop_reason":"tool_use",
        "usage":{"input_tokens":1,"output_tokens":1},
        "content":(0..3).map(|i|json!({"type":"tool_use","id":if i==1 {"tool-0"} else {""},"name":"same","input":{"n":i}})).collect::<Vec<_>>()});
    let messages = crate::providers::anthropic::AnthropicConfig::new("test-key").completion("test");
    assert_normalization(|| fold_document(&messages, &wire));
}

#[test]
fn cohere_missing_ids_and_later_explicit_collision() {
    let mut message = chat_wire()["choices"][0]["message"].clone();
    message["content"] = json!([{"type":"text","text":"prefix"}]);
    let wire = json!({"id":"response","finish_reason":"TOOL_CALL","message":message});
    let chat = crate::providers::cohere::NativeChat::new(
        crate::providers::cohere::CohereConfig::new("test-key"),
        "test",
    );
    assert_normalization(|| fold_document(&chat, &wire));
}

#[test]
fn gemini_rest_missing_ids_and_later_explicit_collision() {
    let wire = json!({"candidates":[{"content":{"role":"model","parts":
        (0..3).map(|i|json!({"functionCall":{"id":if i==1 {"tool-0"} else {""},"name":"same","args":{"n":i}}})).collect::<Vec<_>>()
    },"finishReason":"STOP"}]});
    let generate = crate::providers::gemini::GeminiConfig::new("test-key").completion("test");
    assert_normalization(|| fold_document(&generate, &wire));
}

#[test]
fn gemini_interactions_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","status":"completed","steps":
        (0..3).map(|i|json!({"type":"function_call","id":if i==1 {"tool-0"} else {""},"name":"same","arguments":{"n":i}})).collect::<Vec<_>>()
    });
    let interactions = crate::providers::gemini::GeminiConfig::new("test-key").interactions("test");
    assert_normalization(|| fold_document(&interactions, &wire));
}

#[test]
fn openai_responses_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","object":"response","created_at":0,"status":"completed","model":"test","tools":[],"output":
        (0..3).map(|i|json!({"type":"function_call","id":format!("item-{i}"),"call_id":if i==1 {"tool-0"} else {""},"name":"same","arguments":json!({"n":i}).to_string(),"status":"completed"})).collect::<Vec<_>>()
    });
    let responses = crate::providers::openai::OpenAIConfig::new("test-key").responses("test");
    assert_normalization(|| fold_document(&responses, &wire));
}
