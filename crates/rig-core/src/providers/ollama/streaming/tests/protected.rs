use super::*;
use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use futures::StreamExt;

async fn collect(
    ndjson: &str,
) -> (
    Vec<String>,
    Vec<ProviderError>,
    Result<CompletionResponse, ProviderError>,
) {
    let model = Model::new(
        OllamaConfig::new().completion(MODEL),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::copy_from_slice(ndjson.as_bytes()),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream should open");
    let (mut texts, mut errors) = (Vec::new(), Vec::new());
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(_) => {}
            Err(error) => errors.push(error),
        }
    }
    (texts, errors, stream.finish().await)
}

#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
    );
    let (texts, errors, terminal) = collect(ndjson).await;
    assert_eq!(texts, ["hi"]);
    assert!(errors.is_empty());
    assert!(
        matches!(terminal, Err(ProviderError::Truncated)),
        "EOF must not synthesize a terminal record"
    );
}

#[tokio::test]
async fn malformed_line_is_surfaced_and_the_terminal_still_arrives() {
    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        "{not json\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":" there"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
    );
    let (texts, errors, terminal) = collect(ndjson).await;
    assert_eq!(texts, ["hi", " there"]);
    assert!(
        !errors.is_empty(),
        "the malformed line must reach the consumer"
    );
    let terminal = terminal.expect("the genuine done record must still arrive");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[tokio::test]
async fn content_after_the_done_record_is_not_yielded() {
    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":"stray"},"done":false}"#,
        "\n",
    );
    let (texts, errors, terminal) = collect(ndjson).await;
    assert_eq!(
        texts,
        ["hi"],
        "content after the done record must not be yielded"
    );
    assert!(errors.is_empty());
    let terminal = terminal.expect("the done record must yield the terminal record");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[test]
fn classify_ndjson_line_is_known_or_corrupt() {
    let line = json!({"model":"llama3.2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"hi"},"done":false});
    let decoder = ChatDecoder::default();
    assert!(matches!(
        decoder.classify(WireFrame::Text(line.to_string())),
        WireEvent::Known(_)
    ));
    assert!(matches!(
        decoder.classify(WireFrame::Text("{not json".to_owned())),
        WireEvent::Corrupt(_)
    ));
    assert!(matches!(
        decoder.classify(WireFrame::Text(r#"{"done":42}"#.to_owned())),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn splits_legacy_reasoning_with_or_without_opening_marker() {
    assert_eq!(
        split_legacy_thinking("<think>private reasoning</think>\n\nvisible answer", false),
        (Some("private reasoning"), "visible answer")
    );
    assert_eq!(
        split_legacy_thinking("private reasoning\n</think>\n\nvisible answer", true),
        (Some("private reasoning"), "visible answer")
    );
}

#[test]
fn leaves_unterminated_or_inline_reasoning_markers_visible() {
    for content in [
        "<think>unterminated",
        "The literal marker is <think>.",
        "  visible indentation",
        "The closing token </think> is XML-like.",
        "Example:\n</think>\nis a closing tag.",
    ] {
        assert_eq!(split_legacy_thinking(content, true), (None, content));
    }
}

#[tokio::test]
async fn missing_tool_ids_are_distinct_stable_and_collision_free_in_responses() {
    let wire = json!({"model":"test","created_at":"2024-01-01T00:00:00Z","done":true,"message":{"role":"assistant","content":"","tool_calls":[
        {"function":{"name":"same","arguments":{"value":1}}},
        {"id":"tool-0","function":{"name":"same","arguments":{"value":2}}},
        {"function":{"name":"same","arguments":{"value":3}}}
    ]}});
    let normalize = || async {
        Model::new(
            OllamaConfig::new().completion(MODEL),
            RecordingHttpClient::new(wire.to_string()),
        )
        .call(CompletionRequest::new("hello"))
        .await
        .expect("reply")
    };
    let first = normalize().await;
    assert_eq!(
        serde_json::to_value(&first.choice).expect("serialized"),
        serde_json::to_value(normalize().await.choice).expect("serialized")
    );
    let calls: Vec<_> = first.tool_calls().collect();
    assert_eq!(
        calls
            .iter()
            .map(|call| &call.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(calls[0].id.provider().is_none());
    assert_eq!(
        calls[1]
            .id
            .provider()
            .map(crate::message::ProviderCallId::as_str),
        Some("tool-0")
    );
    assert!(calls[2].id.provider().is_none());
}

async fn unary(body: Value) -> Result<CompletionResponse, ProviderError> {
    Model::new(
        OllamaConfig::new().completion(MODEL),
        RecordingHttpClient::new(body.to_string()),
    )
    .call(CompletionRequest::new("hello"))
    .await
}

#[tokio::test]
async fn test_chat_completion() {
    let sample_chat_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "The sky is blue because of Rayleigh scattering.",
            "images": null,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": {
                            "location": "San Francisco, CA",
                            "format": "celsius"
                        }
                    }
                }
            ]
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });

    let conv = unary(sample_chat_response)
        .await
        .expect("the reply decodes");
    assert!(
        conv.choice
            .iter()
            .any(|c| matches!(c, crate::completion::AssistantContent::Text(t) if t.text == "The sky is blue because of Rayleigh scattering.")),
        "the text survives: {:?}",
        conv.choice
    );
    assert!(
        conv.choice.iter().any(|c| matches!(
            c,
            crate::completion::AssistantContent::ToolCall(call) if call.function.name == "get_current_weather"
        )),
        "the tool call survives: {:?}",
        conv.choice
    );
}

#[tokio::test]
async fn response_metadata_is_normalized() {
    let normalized = unary(json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {"role": "assistant", "content": "Hi!", "tool_calls": []},
        "done": true,
        "done_reason": "length",
        "prompt_eval_count": 12u64,
        "eval_count": 3u64
    }))
    .await
    .expect("normalization should succeed");

    assert_eq!(normalized.provider(), PROVIDER_NAME);
    assert_eq!(normalized.model(), Some("llama3.2"));
    assert_eq!(
        normalized.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    // Ollama assigns no message identifier.
    assert_eq!(normalized.message_id, None);
    assert_eq!(normalized.usage.input_tokens, Some(12));
    assert_eq!(normalized.usage.output_tokens, Some(3));
    assert_eq!(normalized.usage.total_tokens, Some(15));
}

#[tokio::test]
async fn tool_call_turn_upgrades_a_plain_stop_to_tool_calls() {
    let normalized = unary(json!({
        "model": "qwen3:4b",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Berlin"}}}
            ]
        },
        "done": true,
        "done_reason": "stop"
    }))
    .await
    .expect("normalization should succeed");

    assert_eq!(
        normalized.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn nonstreaming_response_preserves_thinking_as_reasoning() {
    let sample_response = json!({
        "model": "qwen3:4b",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
            "images": null,
            "tool_calls": [
                { "type": "function", "function": { "name": "get_weather", "arguments": { "location": "Berlin" } } }
            ]
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });

    let completed = unary(sample_response)
        .await
        .expect("convert to completion response");

    let reasoning = completed.choice.iter().find_map(|c| match c {
        crate::completion::AssistantContent::Reasoning(r) => Some(r.clone()),
        _ => None,
    });
    let has_tool_call = completed
        .choice
        .iter()
        .any(|c| matches!(c, crate::completion::AssistantContent::ToolCall(_)));

    assert!(has_tool_call, "tool call should survive the conversion");
    let reasoning = reasoning.expect(
        "non-streaming response must surface `thinking` as AssistantContent::Reasoning (issue #1926)",
    );
    assert_eq!(
        reasoning.text,
        "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
    );
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":"model not found"}"#;
    let client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let error = Model::new(OllamaConfig::new().completion(MODEL), client)
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("non-success status");
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[test]
fn test_chat_completion_with_thinking() {
    let sample_response = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "The answer is 42.",
            "thinking": "Let me think about this carefully. The question asks for the meaning of life...",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });
    let record: ChatRecord = serde_json::from_value(sample_response).expect("native record parses");
    let message = &record.0["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(
        message["thinking"],
        "Let me think about this carefully. The question asks for the meaning of life..."
    );
    assert_eq!(message["content"], "The answer is 42.");
}

#[test]
fn test_chat_completion_without_thinking() {
    let sample_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Hello!",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 10u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 5u64,
        "eval_duration": 7700000000u64
    });
    let record: ChatRecord = serde_json::from_value(sample_response).expect("native record parses");
    let message = &record.0["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"], "Hello!");
    assert!(message.get("thinking").is_none());
}

#[test]
fn test_streaming_response_with_thinking() {
    let sample_chunk = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Analyzing the problem...",
            "images": null,
            "tool_calls": []
        },
        "done": false
    });
    let record: ChatRecord = serde_json::from_value(sample_chunk).expect("native record parses");
    let message = &record.0["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["thinking"], "Analyzing the problem...");
    assert_eq!(message["content"], "");
}

#[test]
fn test_empty_thinking_content() {
    let sample_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Response",
            "thinking": "",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 10u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 5u64,
        "eval_duration": 7700000000u64
    });
    let record: ChatRecord = serde_json::from_value(sample_response).expect("native record parses");
    let message = &record.0["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["thinking"], "");
    assert_eq!(message["content"], "Response");
}

#[test]
fn test_thinking_with_tool_calls() {
    let sample_response = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Let me check the weather.",
            "thinking": "User wants weather info, I should use the weather tool",
            "images": null,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {
                            "location": "San Francisco"
                        }
                    }
                }
            ]
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 30u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 50u64,
        "eval_duration": 7700000000u64
    });
    let record: ChatRecord = serde_json::from_value(sample_response).expect("native record parses");
    let message = &record.0["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(
        message["thinking"],
        "User wants weather info, I should use the weather tool"
    );
    assert_eq!(message["content"], "Let me check the weather.");
    assert_eq!(message["tool_calls"].as_array().expect("calls").len(), 1);
    assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
}

#[test]
fn done_reason_maps_documented_values_and_preserves_the_rest() {
    for (reason, expected) in [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("load", FinishReason::Other("load".to_owned())),
        ("unload", FinishReason::Other("unload".to_owned())),
    ] {
        let reply = whole(
            MODEL,
            json!({"role":"assistant","content":"answer"}),
            reason,
        );
        let response = decode(MODEL, Mode::Unary, &[reply]).expect("the actual terminal folds");
        assert_eq!(response.finish_reason(), Some(expected));
    }
}

#[tokio::test]
async fn streaming_terminal_record_is_normalized() {
    let terminal = json!({"model":"llama3.2","created_at":"2026-10-05T00:00:00Z","message":{"role":"assistant","content":"answer"},"done":true,"done_reason":"dragons","prompt_eval_count":7,"eval_count":5});
    let (_, errors, result) = collect(&(terminal.to_string() + "\n")).await;
    assert!(errors.is_empty());
    let final_record = result.expect("genuine terminal");
    assert_eq!(final_record.raw, terminal);
    assert_eq!(final_record.provider(), "ollama");
    assert_eq!(final_record.model(), Some("llama3.2"));
    assert_eq!(final_record.message_id, None);
    assert_eq!(
        final_record.finish_reason(),
        Some(FinishReason::Other("dragons".to_owned()))
    );
    assert_eq!(final_record.usage.total_tokens, Some(12));
}
