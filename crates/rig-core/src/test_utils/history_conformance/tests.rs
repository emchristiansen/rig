//! UT020 private controls. Every field below is newly authored synthetic
//! fixture data; no original provider fixture or assertion is modified.

use crate::completion::{CompletionRequest, FinishReason};
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::providers::openai::OpenAIConfig;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::decode_reply;
use crate::wire::{Mode, WireFrame};
use serde_json::{Value, json};

fn frame(value: &Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

fn responses_terminal(kind: &str, status: &str) -> Value {
    let mut response = json!({
        "id": "resp_mode_control",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_mode_control",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]
        }],
        "tools": []
    });
    if status == "incomplete" {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    json!({"type": kind, "sequence_number": 0, "response": response})
}

#[test]
fn completed_responses_use_the_runtime_decoder_in_both_modes() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let request = CompletionRequest::new("fixture");
    let received = responses_terminal("response.completed", "completed");
    for mode in [Mode::Unary, Mode::Streaming] {
        let actual = super::decode(&wire, &request, mode, [frame(&received)])
            .expect("the synthetic completed Responses frame decodes");
        let runtime = decode_reply(&wire, &request, mode, [frame(&received)], Value::Null)
            .expect("the runtime comparison decodes the same frame");
        assert_eq!(actual, runtime);
        assert_eq!(actual.finish_reason(), Some(FinishReason::Stop));
        assert!(matches!(actual.choice.as_slice(),
            [AssistantContent::CarriedText(text)] if text.text() == "done"));
        assert_eq!(actual.response_id(), Some("resp_mode_control"));
        assert_eq!(actual.raw, received["response"]);
    }
}

#[test]
fn tagged_incomplete_responses_accept_unary_and_refuse_streaming() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let request = CompletionRequest::new("fixture");
    let received = responses_terminal("response.incomplete", "incomplete");
    let actual = super::decode(&wire, &request, Mode::Unary, [frame(&received)])
        .expect("Unary accepts the reader-valid synthetic incomplete terminal");
    let runtime = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [frame(&received)],
        Value::Null,
    )
    .expect("the runtime comparison accepts that same Unary terminal");
    assert_eq!(actual, runtime);
    assert_eq!(actual.finish_reason(), Some(FinishReason::Length));
    assert!(!actual.stop().is_failure());
    assert!(matches!(actual.choice.as_slice(),
        [AssistantContent::CarriedText(text)] if text.text() == "done"));
    assert_eq!(actual.raw, received["response"]);

    let actual = super::decode(&wire, &request, Mode::Streaming, [frame(&received)])
        .expect_err("the default Streaming policy refuses the same incomplete frame");
    let runtime = decode_reply(
        &wire,
        &request,
        Mode::Streaming,
        [frame(&received)],
        Value::Null,
    )
    .expect_err("the runtime comparison preserves the same Streaming refusal");
    let (ProviderError::ProviderResponse(actual), ProviderError::ProviderResponse(runtime)) =
        (actual, runtime)
    else {
        panic!("the reader-valid incomplete terminal must reach its typed provider refusal");
    };
    assert_eq!(actual, runtime);
    assert_eq!(actual.body, received.to_string());
}

#[test]
fn completed_chat_uses_the_runtime_decoder_in_both_modes() {
    let wire = OpenAIConfig::new("test-key").chat("gpt-4o");
    let request = CompletionRequest::new("fixture");
    let received = json!({
        "id": "chat_mode_control", "object": "chat.completion", "created": 0,
        "model": "gpt-4o", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop"}]
    });
    for mode in [Mode::Unary, Mode::Streaming] {
        let actual = super::decode(&wire, &request, mode, [frame(&received)])
            .expect("the synthetic completed Chat reply decodes");
        let runtime = decode_reply(&wire, &request, mode, [frame(&received)], Value::Null)
            .expect("the runtime comparison decodes that same Chat reply");
        assert_eq!(actual, runtime);
        assert_eq!(actual.finish_reason(), Some(FinishReason::Stop));
        assert!(matches!(actual.choice.as_slice(),
            [AssistantContent::Text(text)] if text.text.as_str() == "done"));
        assert_eq!(actual.response_id(), Some("chat_mode_control"));
        assert_eq!(actual.raw, received);
    }
}

#[test]
fn empty_unended_chat_eof_uses_the_requested_mode() {
    let wire = OpenAIConfig::new("test-key").chat("gpt-4o");
    let request = CompletionRequest::new("fixture");
    // Synthetic empty frame sequences: neither a chunk nor a terminal arrived.
    for mode in [Mode::Unary, Mode::Streaming] {
        let actual = super::decode(&wire, &request, mode, [])
            .expect_err("the empty sequence cannot finish a Chat reply");
        let runtime = decode_reply(&wire, &request, mode, [], Value::Null)
            .expect_err("the runtime comparison refuses the same empty sequence");
        match (mode, actual, runtime) {
            (Mode::Unary, ProviderError::Response(actual), ProviderError::Response(runtime)) => {
                assert_eq!(actual, crate::message::EMPTY_RESPONSE_ERROR);
                assert_eq!(actual, runtime);
            }
            (Mode::Streaming, ProviderError::Truncated, ProviderError::Truncated) => {}
            _ => panic!("empty Chat EOF must use the mode-specific runtime diagnostic"),
        }
    }
}

#[test]
fn cut_and_partial_preserve_the_received_prefix_without_running_eof() {
    let wire = OpenAIConfig::new("test-key").chat("gpt-4o");
    let request = CompletionRequest::new("restate");
    let received = json!({
        "id": "chat_prefix_control", "object": "chat.completion.chunk", "created": 0,
        "model": "gpt-4o", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "prefix"}, "finish_reason": null}]
    });
    for mode in [Mode::Unary, Mode::Streaming] {
        let (cut, ended, items) = super::cut(&wire, mode, [frame(&received)]);
        let (partial, partial_ended) = super::partial(&wire, mode, [frame(&received)]);
        assert_eq!(partial, cut);
        assert!(!ended);
        assert!(!partial_ended);
        assert_eq!(cut.finish_reason(), None);
        assert_eq!(cut.error, None);
        assert_eq!(
            cut.aborted.as_deref(),
            Some("the caller stopped reading before the provider ended the reply")
        );
        assert!(cut.choice.iter().all(|part| part.native_item().is_none()));
        assert!(items.iter().any(|item| matches!(item,
            Item::Event(StreamEvent::Text { text, .. }) if text == "prefix")));
        assert!(
            items
                .iter()
                .all(|item| !matches!(item, Item::Event(StreamEvent::End { .. })))
        );
        // A real received chunk makes both runtime EOF paths Truncated;
        // cut/partial must not run that EOF path or add its diagnostic.
        let eof = super::decode(&wire, &request, mode, [frame(&received)])
            .expect_err("the received prefix has no provider terminal");
        let runtime = decode_reply(&wire, &request, mode, [frame(&received)], Value::Null)
            .expect_err("the runtime comparison reaches EOF on the same prefix");
        assert!(matches!(eof, ProviderError::Truncated));
        assert!(matches!(runtime, ProviderError::Truncated));
    }
}
