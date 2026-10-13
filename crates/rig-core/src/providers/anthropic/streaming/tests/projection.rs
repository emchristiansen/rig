use std::sync::Arc;

use crate::completion::CompletionRequest;
use crate::observe::{
    Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
};
use crate::providers::anthropic::wire::{AnthropicConfig, Messages};
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use futures::StreamExt;

fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
    log.trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect()
}

fn context(log: &Arc<ObservationLog>) -> AdapterContext {
    AdapterContext::new(log.clone(), Subject::default(), "call")
}

fn wire() -> Messages {
    AnthropicConfig::new("test-key").completion("claude-test")
}

fn request() -> CompletionRequest {
    CompletionRequest::new("hello").max_tokens(64)
}

/// A rejected Messages call: the envelope's type and message, and the
/// closure with the one funnel's classification.
#[tokio::test]
async fn messages_rejection_projects_the_envelope() {
    let http = RecordingHttpClient::with_error(
        http::StatusCode::SERVICE_UNAVAILABLE,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    );
    let log = Arc::new(ObservationLog::default());
    let error = crate::driver::Model::new(wire(), http.clone())
        .call_observed(request(), context(&log))
        .await
        .expect_err("the transport rejects the call");
    assert!(error.is_retryable());
    let events = adapter_events(&log);
    assert!(
        matches!(&events[0], AdapterEvent::Started { method, route } if method == "POST" && route == "/v1/messages")
    );
    assert!(events.contains(&AdapterEvent::Response { status: 503 }));
    assert!(events.contains(&AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code: None,
            status: Some("overloaded_error".into()),
            message: Some("Overloaded".into()),
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(503),
                retryable: true,
            }
        })
    );
}

/// A Messages stream: `message_start` carries the id, the model and the
/// prompt usage; `message_delta` carries the stop reason and the answer's
/// usage; the terminal closes the attempt.
#[tokio::test]
async fn messages_stream_projects_usage_stop_reason_and_model() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":9,\"output_tokens\":1,\"cache_read_input_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3,\"output_tokens_details\":{\"thinking_tokens\":2}}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let http = MockStreamingClient {
        sse_bytes: bytes::Bytes::from(sse),
    };
    let log = Arc::new(ObservationLog::default());
    let stream = crate::driver::Model::new(wire(), http.clone())
        .stream_observed(request(), context(&log))
        .expect("the streamed request encodes");
    let mut stream = Box::pin(stream);
    while let Some(item) = stream.next().await {
        item.expect("the recorded stream decodes without an in-band error");
    }
    drop(stream);

    let events = adapter_events(&log);
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cached_input_tokens: Some(0),
            ..AdapterUsage::default()
        }
    }));
    let usages: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AdapterEvent::Usage { usage } => Some(usage),
            _ => None,
        })
        .collect();
    assert_eq!(
        usages.len(),
        2,
        "payload snapshots exclude a final normalized usage event"
    );
    assert!(usages.iter().all(|usage| usage.total_tokens.is_none()));
    // The witnessed reasoning count is the one the response reports.
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            output_tokens: Some(3),
            reasoning_tokens: Some(2),
            ..AdapterUsage::default()
        }
    }));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { model: Some(model), finish_reason: None, .. } }
            if model == "claude-sonnet-4-6"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { finish_reason: Some(reason), .. } }
            if reason == "end_turn"
    )));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    );
}

#[tokio::test]
async fn payload_usage_prefers_top_level_and_precedes_provider_error() {
    let start = serde_json::json!({"type":"message_start", "message":{"id":"msg_1", "model":"claude-test",
        "role":"assistant", "content":[], "stop_reason":null, "usage":{"input_tokens":99,"output_tokens":98}},
        "usage":{"input_tokens":"9", "output_tokens":1, "cache_read_input_tokens":0}});
    let error = serde_json::json!({"type":"error", "usage":{"input_tokens":0,"output_tokens":3,
        "output_tokens_details":{"thinking_tokens":2}}, "error":{"type":"overloaded_error","message":"Overloaded"}});
    let frames = format!("data: {start}\n\ndata: {error}\n\n");
    let log = Arc::new(ObservationLog::default());
    let model = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(frames),
        },
    );
    let mut stream = model
        .stream_observed(request(), context(&log))
        .expect("encodes");
    let mut error_seen = false;
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            assert!(error.is_retryable());
            error_seen = true;
        }
    }
    assert!(error_seen);
    drop(stream);
    let events = adapter_events(&log);
    let usages: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AdapterEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        usages,
        vec![
            AdapterUsage {
                input_tokens: Some(9),
                output_tokens: Some(1),
                cached_input_tokens: Some(0),
                ..Default::default()
            },
            AdapterUsage {
                input_tokens: Some(0),
                output_tokens: Some(3),
                reasoning_tokens: Some(2),
                ..Default::default()
            }
        ]
    );
    let usage_at = events
        .iter()
        .position(|event| {
            matches!(event, AdapterEvent::Usage { usage }
        if usage.output_tokens == Some(3))
        })
        .expect("error payload snapshot");
    let error_at = events
        .iter()
        .position(|event| matches!(event, AdapterEvent::ErrorEnvelope { .. }))
        .expect("error envelope");
    assert!(usage_at < error_at);
    // G156: the default empty Provider verdict is suppressed.
    assert_eq!(error_at, usage_at + 1);
    assert!(matches!(events[usage_at + 1], AdapterEvent::ErrorEnvelope { .. }));
}

#[tokio::test]
async fn zeroed_terminal_input_is_observed_before_gateway_normalization() {
    let start = serde_json::json!({"type":"message_start", "message":{"id":"msg_1", "model":"claude-test",
        "role":"assistant", "content":[], "stop_reason":null, "usage":{"input_tokens":5,"output_tokens":0}}});
    let text = serde_json::json!({"type":"content_block_start", "index":0, "content_block":{"type":"text","text":"hi"}});
    let end = serde_json::json!({"type":"message_delta", "delta":{"stop_reason":"end_turn","stop_sequence":null},
        "usage":{"input_tokens":0,"output_tokens":3}});
    let frames = format!("data: {start}\n\ndata: {text}\n\ndata: {end}\n\n");
    let log = Arc::new(ObservationLog::default());
    let model = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(frames),
        },
    );
    let mut stream = model
        .stream_observed(request(), context(&log))
        .expect("encodes");
    while let Some(item) = stream.next().await {
        item.expect("decodes");
    }
    let response = stream.finish().await.expect("provider terminal");
    assert_eq!(response.usage.input_tokens, Some(5));
    let usages: Vec<_> = adapter_events(&log)
        .into_iter()
        .filter_map(|event| match event {
            AdapterEvent::Usage { usage } => Some(usage),
            _ => None,
        })
        .collect();
    assert_eq!(
        usages,
        vec![
            AdapterUsage {
                input_tokens: Some(5),
                output_tokens: Some(0),
                ..Default::default()
            },
            AdapterUsage {
                input_tokens: Some(0),
                output_tokens: Some(3),
                ..Default::default()
            }
        ]
    );
}
