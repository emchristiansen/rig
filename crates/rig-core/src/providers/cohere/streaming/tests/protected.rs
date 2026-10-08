use super::*;
use crate::providers::cohere::{COMMAND_R_08_2024, CohereConfig};
use crate::test_utils::MockStreamingClient;
use futures::StreamExt;

fn classify(data: &str) -> WireEvent<ChatEvent> {
    ChatDecoder::new(Mode::Streaming).classify(WireFrame::Text(data.to_owned()))
}

async fn collect(
    sse_bytes: bytes::Bytes,
) -> (
    Vec<StreamEvent>,
    Vec<ProviderError>,
    Result<CompletionResponse, ProviderError>,
) {
    let model = crate::driver::Model::new(
        CohereConfig::new("test-key").completion(COMMAND_R_08_2024),
        MockStreamingClient { sse_bytes },
    );
    let mut stream = model
        .stream(crate::completion::CompletionRequest::new("hello"))
        .expect("stream opens");
    let (mut events, mut errors) = (Vec::new(), Vec::new());
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(event)) => events.push(event),
            Ok(_) => {}
            Err(error) => errors.push(error),
        }
    }
    (events, errors, stream.finish().await)
}

fn texts(events: &[StreamEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn classify_known_event_decodes() {
    let frame =
        json!({"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}).to_string();
    assert!(matches!(classify(&frame),WireEvent::Known(event) if event.kind()=="content-delta"));
}
#[test]
fn classify_unknown_event_type_is_unknown() {
    assert!(
        matches!(classify(&json!({"type":"citation-start"}).to_string()),WireEvent::Unknown{event_type,..} if event_type=="citation-start")
    );
}
#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(classify("{not json"), WireEvent::Corrupt(_)));
}
#[test]
fn classify_known_event_with_defective_payload_is_corrupt() {
    assert!(matches!(
        classify(&json!({"type":"content-delta","delta":42}).to_string()),
        WireEvent::Corrupt(_)
    ));
}

#[tokio::test]
async fn stream_terminal_record_is_normalized() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    let terminal = result.expect("terminal record");
    assert_eq!(terminal.provider(), "cohere");
    assert_eq!(terminal.response_id(), Some("msg_1"));
    assert_eq!(terminal.message_id, None);
    assert_eq!(
        terminal.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
    assert_eq!(terminal.usage.total_tokens, Some(14));
    assert_eq!(terminal.model(), None);
    assert!(errors.is_empty());
}

#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    assert_eq!(texts(&events), ["hi"]);
    assert!(errors.is_empty());
    assert!(matches!(result, Err(ProviderError::Truncated)));
}

#[tokio::test]
async fn malformed_frame_is_surfaced_and_the_terminal_still_arrives() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
            "{not json",
            r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    assert_eq!(texts(&events), ["hi"]);
    assert!(!errors.is_empty(), "malformed frame reaches consumer");
    let terminal = result.expect("genuine terminal arrives");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[tokio::test]
async fn known_event_with_malformed_field_is_surfaced_as_an_error() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":42}}}}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    assert!(!errors.is_empty(), "malformed known field reaches consumer");
    for error in errors {
        assert_eq!(error.kind(), crate::error::ErrorKind::Json);
    }
    let terminal = result.expect("genuine terminal arrives");
    assert_eq!(terminal.usage.input_tokens, Some(10));
}

#[tokio::test]
async fn unknown_event_type_is_skipped_and_the_terminal_still_arrives() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"citation-start","delta":{"whatever":true}}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    assert_eq!(texts(&events), ["hi"]);
    assert!(errors.is_empty());
    let terminal = result.expect("genuine terminal arrives");
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[tokio::test]
async fn message_end_without_delta_still_emits_the_terminal_record() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
            r#"{"type":"message-end"}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    assert_eq!(texts(&events), ["hi"]);
    assert!(errors.is_empty());
    let terminal = result.expect("bare message-end is terminal");
    assert_eq!(terminal.usage, crate::completion::Usage::default());
    assert_eq!(terminal.finish_reason(), None);
    assert_eq!(terminal.response_id(), Some("msg_1"));
}

#[tokio::test]
async fn thinking_deltas_aggregate_into_one_reasoning_part_before_the_text() {
    let sse_bytes = bytes::Bytes::from(
        [
            r#"{"type":"message-start","id":"msg_1"}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"thinking":"step one, "}}}}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"thinking":"step two"}}}}"#,
            r#"{"type":"content-delta","delta":{"message":{"content":{"text":"answer"}}}}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>(),
    );

    let (events, errors, result) = collect(sse_bytes).await;
    let reasoning: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, ["step one, ", "step two"]);
    assert!(errors.is_empty());
    let parts = result.expect("terminal").choice;
    assert_eq!(parts.len(), 2, "one reasoning part, one text part");
    assert!(
        matches!(parts.first(),Some(AssistantContent::Reasoning(reasoning)) if reasoning.text=="step one, step two")
    );
    assert!(matches!(parts.get(1),Some(AssistantContent::Text(text)) if text.text=="answer"));
}

#[tokio::test]
async fn empty_tool_call_ids_are_minted_not_keyed_on_the_empty_string() {
    let call = |n: u32| {
        [
            r#"{"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"","function":{"name":"add","arguments":""}}}}}"#.to_owned(),
            format!(
                r#"{{"type":"tool-call-delta","delta":{{"message":{{"tool_calls":{{"function":{{"arguments":"{{\"n\":{n}}}"}}}}}}}}}}"#
            ),
            r#"{"type":"tool-call-end"}"#.to_owned(),
        ]
    };
    let sse_bytes = bytes::Bytes::from(
        std::iter::once(r#"{"type":"message-start","id":"msg_1"}"#.to_owned())
            .chain(call(1))
            .chain(call(2))
            .chain(std::iter::once(
                r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}}"#.to_owned(),
            ))
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    let (events, errors, result) = collect(sse_bytes).await;
    assert!(errors.is_empty());
    let starts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::ToolCall,
                ..
            } => Some(*part),
            _ => None,
        })
        .collect();
    let ends: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::End { part, .. } => Some(*part),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2, "two calls");
    assert_ne!(starts[0], starts[1], "idless calls have distinct blocks");
    let terminal = result.expect("terminal");
    let calls: Vec<_> = terminal
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.id.is_local()));
    assert_ne!(calls[0].id, calls[1].id);
    for start in starts {
        assert!(ends.contains(&start), "each call ends under its key");
    }
}

#[tokio::test]
async fn errored_stream_does_not_synthesize_a_terminal_record() {
    let model = crate::driver::Model::new(
        CohereConfig::new("test-key").completion(COMMAND_R_08_2024),
        crate::test_utils::HttpErrorStreamingClient::new(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"message":"slow down"}"#,
        ),
    );
    let mut stream = model
        .stream(crate::completion::CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if item.is_err() {
            saw_error = true;
        }
    }
    assert!(saw_error, "transport failure reaches consumer");
    assert!(
        stream.finish().await.is_err(),
        "failed stream has no terminal response"
    );
}

#[test]
fn test_message_content_delta_deserialization() {
    let json = json!({
        "type": "content-delta",
        "delta": {
            "message": {
                "content": {
                    "text": "Hello world"
                }
            }
        }
    });

    let event: ChatEvent = serde_json::from_value(json).expect("actual wire event");
    assert!(event.fields.pointer("/delta").is_some_and(Value::is_object));
    assert_eq!(
        event
            .fields
            .pointer("/delta/message/content/text")
            .and_then(Value::as_str),
        Some("Hello world")
    );
}

#[test]
fn test_tool_call_start_deserialization() {
    let json = json!({
        "type": "tool-call-start",
        "delta": {
            "message": {
                "tool_calls": {
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{"
                    }
                }
            }
        }
    });

    let event: ChatEvent = serde_json::from_value(json).expect("actual wire event");
    assert!(event.fields.pointer("/delta").is_some_and(Value::is_object));
    assert_eq!(
        event
            .fields
            .pointer("/delta/message/tool_calls/id")
            .and_then(Value::as_str),
        Some("call_123")
    );
    assert_eq!(
        event
            .fields
            .pointer("/delta/message/tool_calls/function/name")
            .and_then(Value::as_str),
        Some("get_weather")
    );
}

#[test]
fn test_tool_call_delta_deserialization() {
    let json = json!({
        "type": "tool-call-delta",
        "delta": {
            "message": {
                "tool_calls": {
                    "function": {
                        "arguments": "\"location\""
                    }
                }
            }
        }
    });

    let event: ChatEvent = serde_json::from_value(json).expect("actual wire event");
    assert!(event.fields.pointer("/delta").is_some_and(Value::is_object));
    assert_eq!(
        event
            .fields
            .pointer("/delta/message/tool_calls/function/arguments")
            .and_then(Value::as_str),
        Some("\"location\"")
    );
}

#[test]
fn test_tool_call_end_deserialization() {
    let json = json!({
        "type": "tool-call-end"
    });

    let event: ChatEvent = serde_json::from_value(json).expect("actual wire event");
}

#[test]
fn test_message_end_with_usage_deserialization() {
    let json = json!({
        "type": "message-end",
        "delta": {
            "usage": {
                "tokens": {
                    "input_tokens": 100,
                    "output_tokens": 50
                }
            }
        }
    });

    let event: ChatEvent = serde_json::from_value(json).expect("actual wire event");
    assert!(event.fields.pointer("/delta").is_some_and(Value::is_object));
    assert_eq!(
        event
            .fields
            .pointer("/delta/usage/tokens/input_tokens")
            .and_then(Value::as_f64),
        Some(100.0)
    );
    assert_eq!(
        event
            .fields
            .pointer("/delta/usage/tokens/output_tokens")
            .and_then(Value::as_f64),
        Some(50.0)
    );
}

#[test]
fn test_streaming_event_order() {
    // Test that a typical sequence of events deserializes correctly
    let events = vec![
        json!({"type": "message-start"}),
        json!({"type": "content-start"}),
        json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "Sure, "
                    }
                }
            }
        }),
        json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "I can help with that."
                    }
                }
            }
        }),
        json!({"type": "content-end"}),
        json!({"type": "tool-plan"}),
        json!({
            "type": "tool-call-start",
            "delta": {
                "message": {
                    "tool_calls": {
                        "id": "call_abc",
                        "function": {
                            "name": "search",
                            "arguments": ""
                        }
                    }
                }
            }
        }),
        json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "{\"query\":"
                        }
                    }
                }
            }
        }),
        json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "\"Rust\"}"
                        }
                    }
                }
            }
        }),
        json!({"type": "tool-call-end"}),
        json!({
            "type": "message-end",
            "delta": {
                "usage": {
                    "tokens": {
                        "input_tokens": 50,
                        "output_tokens": 25
                    }
                }
            }
        }),
    ];

    for (i, event_json) in events.iter().enumerate() {
        let result = serde_json::from_value::<ChatEvent>(event_json.clone());
        assert!(
            result.is_ok(),
            "Failed to deserialize event at index {}: {:?}",
            i,
            result.err()
        );
    }
}
