use super::super::ResponsesDecoder;
use crate::completion::CompletionRequest;
use crate::driver::{Decoded, feed_frames};
use crate::error::{ErrorKind, ErrorReport};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::providers::internal::openai_chat_completions_compatible::tests::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::openai::OpenAIConfig;
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::{Mode, WireFrame};
use futures::StreamExt;
use serde_json::{self, json};

fn sample_response(status: &str) -> serde_json::Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.4",
        "output": [],
        "tools": [],
    })
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
    let mut response = sample_response("failed");
    response["error"] = json!({ "code": "", "message": "maximum context length exceeded" });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("maximum context length exceeded")
    }));
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
    let mut response = sample_response("failed");
    response["error"] =
        json!({ "code": "context_length_exceeded", "message": "maximum context length exceeded" });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed")
            && body.contains("context_length_exceeded")
            && body.contains("maximum context length exceeded")
    }));
}

/// `response.failed` with a transient code (`server_error`,
/// `rate_limit_exceeded`) is retryable, as the same condition is on an HTTP
/// 500 or 429.
///
/// Not a cassette test: a server failure or rate limit is a transient
/// condition that cannot be recorded on demand, and the defect is the
/// verdict the envelope gets, which a hand-written frame pins exactly.
#[tokio::test]
async fn response_failed_with_a_transient_code_is_retryable() -> anyhow::Result<()> {
    for code in ["server_error", "rate_limit_exceeded"] {
        let mut response = sample_response("failed");
        response["error"] = json!({ "code": code, "message": "try again" });
        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });
        let err = first_error_from_event(event).await;
        anyhow::ensure!(err.retryable, "{code} must be retryable: {err:?}");
    }
    Ok(())
}

/// The stream's `error` event with a transient code is retryable.
///
/// Not a cassette test, for the reason above.
#[tokio::test]
async fn an_error_event_with_a_transient_code_is_retryable() -> anyhow::Result<()> {
    let event = json!({
        "type": "error",
        "error": { "message": "boom", "code": "server_error", "type": "server_error" }
    });
    let err = first_error_from_event(event).await;
    anyhow::ensure!(err.retryable, "server_error must be retryable: {err:?}");
    Ok(())
}

#[tokio::test]
async fn streaming_error_event_preserves_full_payload_in_live_loop() {
    use crate::providers::internal::openai_chat_completions_compatible::tests::sse_bytes_from_json_events;
    use crate::test_utils::MockStreamingClient;

    let payload = json!({
        "type": "error",
        "error": {
            "message": "boom",
            "code": "server_error",
            "type": "server_error"
        }
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[payload]),
    })
    .await;

    let err = stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider response error");
    let err = ErrorReport::from(&err);
    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(
        err.provider_response_body()
            .is_some_and(|body| { body.contains("\"type\":\"error\"") && body.contains("boom") })
    );
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after error event"
    );
}

#[tokio::test]
async fn streaming_non_http_transport_error_stays_a_transport_error() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let chunks = vec![Err(crate::http_client::Error::InvalidContentType(
        http::HeaderValue::from_static("application/json"),
    ))];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("non-HTTP transport failure should surface as a transport error");
    let err = ErrorReport::from(&err);
    assert_eq!(
        err.to_string(),
        "HttpError: Invalid content type was returned: \"application/json\""
    );
    assert_eq!(err.kind, ErrorKind::Http);
    // A response-less transport failure has no provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert_eq!(err.http_status, None);
}

/// The OpenAI Responses stream one scripted transport yields: the live
/// loop, over a socket that answers from a script.
async fn responses_stream<H>(http: H) -> crate::streaming::CompletionStream
where
    H: crate::driver::Transport<crate::providers::openai::responses_api::wire::Responses>,
{
    let model = crate::driver::Model::new(OpenAIConfig::new("test-key").responses("gpt-5.4"), http);
    let request = CompletionRequest::new("hello");
    model.stream(request).expect("stream should start")
}

/// The same, for a body scripted as JSON events.
async fn responses_stream_of(events: &[serde_json::Value]) -> crate::streaming::CompletionStream {
    responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(events),
    })
    .await
}

/// Decode a Responses SSE body, its `data:` lines, through the decoder a
/// buffered replay runs.
fn decoded_body(body: &str) -> Decoded<Completion> {
    let frames: Vec<WireFrame> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .map(|data| WireFrame::Text(data.to_owned()))
        .collect();
    // G168: the real document receives each frame before decoding and owns raw.
    feed_frames!(
        ResponsesDecoder::new("openai", super::super::ResponsesStreamOptions::strict()),
        super::super::document::Response::with_strict_reader(false),
        "openai",
        frames
    )
}

/// An SSE body of `events`, one `data:` line each.
fn body_of(events: &[serde_json::Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect()
}

/// The text fragments among `events`.
fn texts_of(events: &[&StreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

async fn first_error_from_event(event: serde_json::Value) -> ErrorReport {
    let mut stream = responses_stream_of(&[event]).await;
    let error = stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider error");
    ErrorReport::from(&error)
}

/// Drain a stream whose provider fully delivered one tool call before a
/// terminal error: the call's part comes first, then the error, then
/// nothing.
async fn flushed_tool_call_then_error(
    stream: &mut crate::streaming::CompletionStream,
) -> (crate::message::ToolCall, ErrorReport) {
    let mut tool_call = None;
    let err = loop {
        match stream
            .next()
            .await
            .expect("stream should yield the flushed tool call, then the error")
        {
            Ok(Item::Event(
                StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    ..
                }
                | StreamEvent::Arguments { .. },
            )) => {}
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            })) => tool_call = Some(call),
            Ok(other) => panic!("expected the flushed tool call first, got {other:?}"),
            Err(err) => break ErrorReport::from(&err),
        }
    };
    let tool_call = tool_call.expect("the flushed tool call must precede the terminal error");
    assert!(
        stream.next().await.is_none(),
        "nothing may follow the terminal error"
    );
    (tool_call, err)
}

/// Raw `reasoning_text` deltas whose done item states no text keep the
/// block's text, and the item stays as the provider stated it: written in
/// as a summary, it would replay one the provider never produced.
#[test]
fn raw_reasoning_text_is_not_written_into_the_item_as_a_summary() {
    let done = json!({"type": "reasoning", "id": "rs_1", "summary": [],
        "encrypted_content": "ENC", "status": "completed"});
    let decoded = decoded_body(&body_of(&[
        json!({"type": "response.output_item.added", "output_index": 0, "sequence_number": 1,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1",
            "content_index": 0, "sequence_number": 2, "delta": "raw chain"}),
        json!({"type": "response.output_item.done", "output_index": 0, "sequence_number": 3,
            "item": done.clone()}),
        completed_with(4, json!([done.clone()])),
    ]));
    let ended = decoded.ended();
    let [AssistantContent::CarriedReasoning(reasoning)] = ended.as_slice() else {
        panic!("one reasoning part: {:?}", decoded.events());
    };
    assert_eq!(reasoning.display_text(), "raw chain");
    assert_eq!(reasoning.payload.item["summary"], json!([]));
    let response = decoded.outcome.expect("genuine completed reply");
    assert_eq!(response.raw["output"][0], done, "the provider's done item remains exact raw evidence");
    // G204: replayed_input is the complete q/assistant/next request.
    let input = replayed_input(&response);
    assert_eq!(input.len(), 3);
    assert_eq!(input[0], json!({"type":"message","role":"user",
        "content":[{"type":"input_text","text":"q"}]}));
    assert_eq!(input[2], json!({"type":"message","role":"user",
        "content":[{"type":"input_text","text":"next"}]}));
    assert_eq!(input.iter().map(|item| (item["type"].clone(), item["id"].clone()))
        .collect::<Vec<_>>(), [
            (json!("message"), serde_json::Value::Null),
            (json!("reasoning"), json!("rs_1")),
            (json!("message"), serde_json::Value::Null),
        ]);
    let replayed = input.iter().find(|item| item["id"] == "rs_1" && item["type"] == "reasoning")
        .expect("the actual received reasoning item replays inside the full ordered input");
    assert_eq!(replayed["summary"], json!([]), "raw reasoning never becomes a fabricated summary");
}

/// A `response.failed` after a fully-delivered tool call: the call ended
/// when its done item arrived, the terminal error follows, and nothing
/// comes after it.
#[tokio::test]
async fn response_failed_follows_the_delivered_tool_call() {
    let tool_call_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });

    let mut response = sample_response("failed");
    response["error"] = json!({ "code": "server_error", "message": "response stream failed" });

    let failed = json!({
        "type": "response.failed",
        "sequence_number": 2,
        "response": response,
    });

    let mut stream = responses_stream_of(&[tool_call_done, failed]).await;
    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    assert_eq!(tool_call.id, crate::message::CallId::from_wire("call_123"));
    assert_eq!(
        tool_call.native.as_ref().map(|native| &native.item["id"]),
        Some(&json!("fc_123"))
    );
    assert_eq!(tool_call.function.name, "example_tool");

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("response stream failed")
    }));
    assert!(stream.finish().await.is_err());
}

/// Same ordering for a transport failure: the fully-delivered tool call,
/// then the error, then the end — with no response.
#[tokio::test]
async fn a_transport_error_follows_the_delivered_tool_call() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let tool_call_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([tool_call_done.to_string()])),
        Err(crate::http_client::Error::non_success_with_details(
            http::StatusCode::BAD_GATEWAY,
            http::HeaderMap::new(),
            r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
        )),
    ];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    assert_eq!(tool_call.id, crate::message::CallId::from_wire("call_123"));
    assert_eq!(
        tool_call.native.as_ref().map(|native| &native.item["id"]),
        Some(&json!("fc_123"))
    );
    assert_eq!(
        err.http_status,
        Some(http::StatusCode::BAD_GATEWAY.as_u16())
    );
    assert!(stream.finish().await.is_err());
}

#[tokio::test]
async fn refusal_content_part_frames_are_no_ops_and_refusal_text_streams() {
    // A refusal turn emits `response.content_part.added/.done` with a
    // `refusal` part — a shape outside the modeled text parts — followed
    // by the refusal text via `response.refusal.delta`. The part frames
    // must parse as no-ops (never error items); the deltas carry the
    // content.
    let events = [
        json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": { "type": "refusal", "refusal": "" }
        }),
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "I can't help with that."
        }),
        json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": { "type": "refusal", "refusal": "I can't help with that." }
        }),
        json!({
            "type": "response.content_part.added",
            "item_id": "rs_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 4,
            "part": { "type": "reasoning_text", "text": "" }
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 5,
            "response": sample_response("completed"),
        }),
    ];
    let decoded = decoded_body(&body_of(&events));
    assert_eq!(texts_of(&decoded.events()), ["I can't help with that."]);
    assert!(decoded.outcome.is_ok(), "the terminal must still arrive");
    let mut stream = responses_stream_of(&events).await;
    let mut delivered = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("refusal part frames never yield an error") {
            delivered.push_str(&text);
        }
    }
    assert_eq!(delivered, "I can't help with that.");
    assert!(stream.finish().await.is_ok(), "the real public stream retains its genuine terminal");
}

/// The `max_output_tokens`-mid-tool-call shape: `arguments_delta` frames
/// stream partial JSON and the done item restates the same truncated bytes.
/// Whether or not fragments preceded the restatement, the call keeps what
/// its arguments state, with the text they arrived as.
#[test]
fn a_cut_off_call_keeps_what_its_arguments_state() {
    let delta = json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "delta": "{\"x\":481",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 2,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":481",
            "status": "incomplete"
        },
    });
    for events in [vec![delta, done.clone()], vec![done]] {
        let decoded = decoded_body(&body_of(&events));
        let calls: Vec<_> = decoded
            .ended()
            .into_iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some((
                    call.function.arguments_value(),
                    call.function.invalid_arguments,
                )),
                _ => None,
            })
            .collect();
        assert!(calls.is_empty(), "A17 preserves the protected incomplete verdict: {:?}", decoded.events());
        assert!(decoded.items.iter().any(|item| matches!(item,
            Ok(Item::Event(StreamEvent::Discard { reason: crate::streaming::DiscardReason::Incomplete, .. }))
        )), "the actual part closes explicitly as incomplete: {:?}", decoded.events());
        assert!(decoded.items.iter().all(|item| !matches!(item,
            Err(error) if !matches!(error, crate::error::ProviderError::Truncated)
        )), "dropping an incomplete call is not an error");
    }
}

#[test]
fn streaming_error_event_preserves_full_payload() {
    let payload = r#"{"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
    let err = decoded_body(&format!("data: {payload}\n"))
        .outcome
        .expect_err("error event should surface as a provider response error");

    assert_eq!(err.provider_response_status(), None);
    assert_eq!(err.provider_response_body(), Some(payload));
    let json = err
        .provider_response_json()
        .expect("raw body should be valid JSON")
        .expect("parsed JSON should be present");
    assert_eq!(json["error"]["code"], "server_error");
}

#[tokio::test]
async fn done_sentinel_is_ignored_without_debug_parse_noise() {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("log buffer mutex should not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut response = sample_response("completed");
    response["usage"] = json!({ "input_tokens": 4, "output_tokens": 2, "total_tokens": 6, "output_tokens_details": { "reasoning_tokens": 0 } });

    // Scoped-subscriber tests must not run concurrently; see
    // `test_utils::scoped_tracing_subscriber_guard`.
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .without_time()
        .with_writer({
            let captured = captured.clone();
            move || SharedWriter(captured.clone())
        })
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&json!({
                "type": "response.completed",
                "sequence_number": 1,
                "response": response,
            }))
            .expect("response event should serialize")
        )),
    })
    .await;

    while let Some(item) = stream.next().await {
        item.expect("stream should complete successfully");
    }
    let usage = stream
        .finish()
        .await
        .expect("expected final response")
        .usage;
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(6));

    let logs = String::from_utf8(
        captured
            .lock()
            .expect("log buffer mutex should not be poisoned")
            .clone(),
    )
    .expect("captured logs should be valid UTF-8");
    assert!(
        !logs.contains("Couldn't deserialize SSE data as StreamingCompletionChunk"),
        "expected [DONE] to bypass the parse-failure debug path, logs were: {logs}"
    );
}

/// A typed corrupt frame is delivered and the genuine terminal remains readable.
#[tokio::test]
async fn a_malformed_frame_ends_the_reply() {
    let delta = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "hello",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response("completed"),
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            delta.to_string(),
            "{not valid json".to_string(),
            completed.to_string(),
        ]),
    };
    let mut stream = responses_stream(http_client).await;

    let mut text = String::new();
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text: chunk, .. })) => text.push_str(&chunk),
            Ok(_) => {}
            Err(err) => error = Some(err),
        }
    }
    assert_eq!(text, "hello");
    let error = error.expect("malformed frame should surface an error item");
    assert_eq!(error.kind(), ErrorKind::Json, "{error:?}");
    assert!(matches!(error.report().detail, Some(crate::error::ErrorDetail::CorruptFrame(_))));
    let response = stream.finish().await.expect("the drained typed corruption does not suppress the genuine terminal");
    assert_eq!(response.response_id(), Some("resp_123"));
}

/// An item id the wire left empty identifies nothing: a text delta under
/// `"item_id": ""` still streams, and a message item with `"id": ""` names
/// no message; neither panics.
#[test]
fn empty_item_ids_identify_nothing_and_do_not_panic() {
    let events = [
        json!({
            "type": "response.output_text.delta",
            "item_id": "",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "still text",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "message",
                "id": "",
                "role": "assistant",
                "status": "completed",
                "content": []
            },
        }),
    ];
    let decoded = decoded_body(&body_of(&events));
    assert_eq!(texts_of(&decoded.events()), ["still text"]);
    let driven = super::helpers::drive(&events);
    assert!(driven.errors.is_empty(), "empty IDs remain harmless through the public stream");
    assert_eq!(super::helpers::folded_texts(&driven.response.choice), ["still text"]);
    assert!(driven.response.message_id.is_none(), "an empty provider message ID names nothing");
    assert!(driven.response.choice.iter().all(|content| match content {
        AssistantContent::CarriedText(text) => text.item_id.as_deref().is_none_or(|id| !id.is_empty()),
        _ => true,
    }), "the delivered partial history never contains an empty message identity");
    assert!(decoded.ended().iter().all(|content| match content {
        AssistantContent::CarriedText(text) => text.item_id.as_deref().is_none_or(|id| !id.is_empty()),
        AssistantContent::ToolCall(call) => call.id.provider().is_none_or(|id| !id.as_str().is_empty()),
        AssistantContent::CustomToolCall(call) => call.id.provider().is_none_or(|id| !id.as_str().is_empty()),
        _ => true,
    }), "no canonical content is keyed by an empty provider identity");
}

/// The `output_item.done` restating `item` at `output_index`.
fn item_done(output_index: u64, sequence: u64, item: serde_json::Value) -> serde_json::Value {
    json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "sequence_number": sequence,
        "item": item,
    })
}

/// The terminal `response.completed` whose `output` is `output`.
fn completed_with(sequence: u64, output: serde_json::Value) -> serde_json::Value {
    let mut response = sample_response("completed");
    response["output"] = output;
    json!({
        "type": "response.completed",
        "sequence_number": sequence,
        "response": response,
    })
}

/// The OpenAI Responses wire every decode below runs.
fn wire() -> crate::providers::openai::responses_api::wire::Responses {
    OpenAIConfig::new("test-key").responses("gpt-5.4")
}

/// One frame per event.
fn frames(events: &[serde_json::Value]) -> Vec<WireFrame> {
    events
        .iter()
        .map(|event| WireFrame::Text(event.to_string()))
        .collect()
}

/// The unary body whose `output` is `output`, as its one frame.
fn whole(output: &[serde_json::Value]) -> Vec<WireFrame> {
    let mut body = sample_response("completed");
    body["output"] = json!(output);
    vec![WireFrame::Text(body.to_string())]
}

/// The response `frames` fold into in `mode`, from a request the wire's own
/// model sent.
fn decode(mode: Mode, frames: Vec<WireFrame>) -> crate::completion::CompletionResponse {
    crate::test_utils::decode_reply(
        &wire(),
        &CompletionRequest::new("hello"),
        mode,
        frames,
        serde_json::Value::Null,
    )
    .expect("the reply decodes")
}

/// `text` in two deltas, so a restatement is never one fragment.
fn halves(text: &str) -> [String; 2] {
    let cut = text
        .char_indices()
        .map(|(at, _)| at)
        .nth(text.chars().count() / 2)
        .unwrap_or(text.len());
    [text[..cut].to_owned(), text[cut..].to_owned()]
}

/// The stream a Responses endpoint sends for `output`: each item added with
/// no content, its text in deltas, then done; then `response.completed`
/// restating the output.
fn restated(output: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut body = sample_response("completed");
    body["output"] = json!(output);
    restated_body(&body)
}

/// [`restated`] for a whole response `body`, which the stream's terminal
/// restates.
fn restated_body(body: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut created = body.clone();
    created["status"] = json!("in_progress");
    created["output"] = json!([]);
    let mut events = vec![json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": created,
    })];
    let output = body["output"].as_array().cloned().unwrap_or_default();
    for (index, item) in output.iter().enumerate() {
        let mut added = item.clone();
        match item["type"].as_str() {
            Some("message") => {
                added["content"] = json!([]);
                added["status"] = json!("in_progress");
            }
            Some("reasoning") => {
                added["summary"] = json!([]);
                if let Some(added) = added.as_object_mut() {
                    added.shift_remove("content");
                    added.shift_remove("encrypted_content");
                }
            }
            Some("function_call") => added["arguments"] = json!(""),
            _ => {}
        }
        events.push(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "sequence_number": events.len(),
            "item": added,
        }));
        let mut delta = |kind: &str, part: (&str, usize), text: &str| {
            for half in halves(text) {
                events.push(json!({
                    "type": kind,
                    "item_id": item["id"],
                    "output_index": index,
                    part.0: part.1,
                    "sequence_number": events.len(),
                    "delta": half,
                }));
            }
        };
        match item["type"].as_str() {
            Some("message") => {
                for (at, part) in item["content"].as_array().into_iter().flatten().enumerate() {
                    match part["type"].as_str() {
                        Some("refusal") => delta(
                            "response.refusal.delta",
                            ("content_index", at),
                            part["refusal"].as_str().unwrap_or_default(),
                        ),
                        _ => delta(
                            "response.output_text.delta",
                            ("content_index", at),
                            part["text"].as_str().unwrap_or_default(),
                        ),
                    }
                }
            }
            Some("reasoning") => {
                for (at, part) in item["summary"].as_array().into_iter().flatten().enumerate() {
                    delta(
                        "response.reasoning_summary_text.delta",
                        ("summary_index", at),
                        part["text"].as_str().unwrap_or_default(),
                    );
                }
            }
            Some("function_call") => delta(
                "response.function_call_arguments.delta",
                ("content_index", 0),
                item["arguments"].as_str().unwrap_or_default(),
            ),
            _ => {}
        }
        events.push(item_done(index as u64, events.len() as u64, item.clone()));
    }
    events.push(json!({
        "type": "response.completed",
        "sequence_number": events.len(),
        "response": body,
    }));
    events
}

fn message(id: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "annotations": [], "logprobs": [], "text": text }],
    })
}

fn reasoning(id: &str, summaries: &[&str]) -> serde_json::Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": summaries
            .iter()
            .map(|text| json!({ "type": "summary_text", "text": text }))
            .collect::<Vec<_>>(),
        "encrypted_content": format!("ciphertext-of-{id}"),
    })
}

fn function_call(id: &str, call_id: &str, arguments: &str) -> serde_json::Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": "completed",
    })
}

/// A reply carrying every kind of output item the wire distinguishes, and
/// a kind and a field no version of rig has seen.
fn every_kind() -> Vec<serde_json::Value> {
    vec![
        reasoning("rs_1", &["Planning.", "Checking twice."]),
        message("msg_1", "Looking it up."),
        function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
        json!({
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_2",
            "name": "apply_patch",
            "input": "*** Begin Patch",
        }),
        json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": { "type": "search", "query": "rig" },
        }),
        json!({ "type": "compaction", "id": "cmp_1", "encrypted_content": "compacted" }),
        json!({
            "type": "computer_call",
            "id": "cu_1",
            "call_id": "call_3",
            "action": { "type": "click", "x": 1, "y": 2 },
            "status": "completed",
        }),
        json!({
            "type": "message",
            "id": "msg_2",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "future_field": { "nested": [1, 2] },
            "content": [
                { "type": "output_text", "annotations": [{ "type": "url_citation", "url": "https://rig.rs" }], "text": "Rig is " },
                { "type": "refusal", "refusal": "all I can say." },
            ],
        }),
        json!({ "type": "future_item", "id": "fut_1", "payload": { "kept": true } }),
    ]
}

/// Inspect the canonical carried payloads. Containing-message fields are
/// history evidence; this projection grants none of them outbound authority.
fn natives(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    let mut items: Vec<serde_json::Value> = Vec::new();
    let mut previous_message: Option<Option<String>> = None;
    for block in &response.choice {
        match block {
            AssistantContent::CarriedText(text) => {
                if previous_message.as_ref() == Some(&text.item_id) {
                    items.last_mut().expect("preceding containing message")["content"]
                        .as_array_mut().expect("message content array")
                        .push(text.payload.item.clone());
                } else {
                    let mut message = json!({"type":"message","role":"assistant",
                        "status":"completed","content":[text.payload.item]});
                    if let Some(id) = &text.item_id { message["id"] = json!(id); }
                    if let Some(phase) = &text.phase { message["phase"] = json!(phase); }
                    if let Some(fields) = text.unknown_message_fields() {
                        message.as_object_mut().expect("history inspection object")
                            .extend(fields.clone());
                    }
                    previous_message = Some(text.item_id.clone());
                    items.push(message);
                }
            }
            other => {
                previous_message = None;
                items.push(match other {
                    AssistantContent::CarriedReasoning(reasoning) => reasoning.payload.item.clone(),
                    AssistantContent::ProviderItem(item) => item.payload.item.clone(),
                    AssistantContent::CustomToolCall(call) => call.native.as_ref().expect("exact custom item").item.clone(),
                    AssistantContent::Opaque(opaque) => opaque.item.clone(),
                    block => block.native_item().cloned().unwrap_or_default(),
                });
            }
        }
    }
    items
}

#[test]
fn a_whole_reply_and_its_restatement_as_a_stream_agree() {
    let replies = [
        every_kind(),
        vec![message("msg_1", "Just text.")],
        vec![
            reasoning("rs_1", &[]),
            function_call("fc_1", "call_1", ""),
            function_call("fc_2", "call_2", r#"{"nested":{"deep":[1,2]}}"#),
        ],
    ];
    for output in replies {
        for response in [decode(Mode::Unary, whole(&output)), decode(Mode::Streaming, frames(&restated(&output)))] {
            let carrying = response.choice.iter().filter_map(|part| match part {
                AssistantContent::CarriedText(text) if text.item_id.as_deref() == Some("msg_2") => Some(text),
                _ => None,
            }).collect::<Vec<_>>();
            if output.iter().any(|item| item.get("future_field").is_some()) {
                assert_eq!(carrying.len(), 2, "both ordered parts retain containing-message evidence");
                for text in carrying {
                    let fields = text.unknown_message_fields().expect("A27 exact history evidence");
                    assert_eq!(fields.len(), 1, "known container fields are separate owners");
                    assert_eq!(fields["future_field"], json!({"nested":[1,2]}));
                    let roundtrip: crate::message::CarriedText = serde_json::from_value(serde_json::to_value(text).unwrap()).unwrap();
                    assert_eq!(roundtrip.unknown_message_fields(), Some(fields));
                }
                assert!(replayed_input(&response).iter().all(|item| item.get("future_field").is_none()), "A27 inspection fields never become request fields");
            }
        }
        crate::test_utils::history::assert_restated_agrees(
            &wire(),
            whole(&output),
            frames(&restated(&output)),
        );
    }
}

/// The `input` a request sends through `wire`, history shaped by the
/// driver's own `prepare`.
fn encoded_input(
    wire: &crate::providers::openai::responses_api::wire::Responses,
    request: CompletionRequest,
) -> Vec<serde_json::Value> {
    use crate::wire::{Operation, Wire};
    // The request declares the tools its history calls, so the calls stay
    // calls.
    let mut request = request;
    let calls = request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            crate::message::Message::Assistant(turn) => Some(turn.tool_calls()),
            _ => None,
        });
    let mut tools: Vec<crate::completion::ToolDefinition> = Vec::new();
    for call in calls.flatten() {
        if !tools.iter().any(|tool| tool.name == call.function.name) {
            tools.push(crate::completion::ToolDefinition {
                name: call.function.name.clone(),
                description: "A tool".to_owned(),
                parameters: json!({"type": "object"}),
            });
        }
    }
    request.tools.extend(tools);
    let request = Completion::prepare(request, &wire.describe()).expect("the history is valid");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let body = crate::test_utils::json_body(&encoded.request);
    body["input"].as_array().cloned().unwrap_or_default()
}

#[test]
fn an_incomplete_turn_reports_why_unless_the_token_limit_cut_it() {
    for (reason, stop) in [
        ("max_output_tokens", crate::message::StopReason::Length),
        (
            "content_filter",
            crate::message::StopReason::Error("Provider finish_reason: content_filter".into()),
        ),
        (
            "max_tool_calls",
            crate::message::StopReason::Error("Provider finish_reason: max_tool_calls".into()),
        ),
    ] {
        let mut body = sample_response("incomplete");
        body["incomplete_details"] = json!({ "reason": reason });
        body["output"] = json!([message("msg_1", "partial")]);
        let response = decode(Mode::Unary, vec![WireFrame::Text(body.to_string())]);
        assert_eq!(response.stop(), stop, "{reason}");
        assert_eq!(response.text(), "partial");
    }
}

#[test]
fn a_reply_names_the_model_and_response_it_came_from() {
    let response = decode(Mode::Unary, whole(&[message("msg_1", "hi")]));
    let Some(crate::message::Message::Assistant(turn)) = response.message() else {
        panic!("one assistant turn");
    };
    let origin = turn.origin.expect("a decoded turn has an origin");
    assert_eq!(origin.api.as_str(), "openai.responses");
    assert_eq!(origin.provider, "openai");
    assert_eq!(origin.model, "gpt-5.4");
    assert_eq!(origin.response_id.as_deref(), Some("resp_123"));
}

/// The whole Responses replies a provider's cassettes record: each unary
/// body, and the response object each recorded stream's terminal restates.
fn recorded_whole_replies(provider: &str) -> Vec<serde_json::Value> {
    let mut replies = Vec::new();
    let mut stack = vec![
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rig-cassette/fixtures/cassettes")
            .join(provider),
    ];
    while let Some(dir) = stack.pop() {
        for path in std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
        {
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for document in serde_yaml::Deserializer::from_str(&text) {
                let Ok(cassette) = <serde_yaml::Value as serde::Deserialize>::deserialize(document) else { continue; };
                let Some(body) = cassette.get("then").and_then(|reply| reply.get("body")).and_then(serde_yaml::Value::as_str) else { continue; };
                let documents: Vec<serde_json::Value> = if let Ok(body) = serde_json::from_str(body) {
                    vec![body]
                } else {
                    let mut framer = rig_http::http_client::framing::SseFramer::new();
                    let documents = framer.push(body.as_bytes())
                        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(&frame.data).ok())
                        .filter(|event| event["type"] == "response.completed")
                        .map(|event| event["response"].clone())
                        .collect();
                    documents
                };
                replies.extend(documents.into_iter().filter(|document| {
                    document["object"] == "response"
                        && document["status"] == "completed"
                        && document["output"].is_array()
                }));
            }
        }
    }
    replies
}

/// Every whole reply the family's cassettes record folds into the same turn
/// restated as a stream, through the shared restate-as-stream harness.
#[test]
fn every_recorded_whole_reply_agrees_with_its_restatement_as_a_stream() {
    let mut checked = 0;
    for (provider, wire) in [
        ("openai", wire()),
        ("xai", wire()),
        ("copilot", wire()),
        (
            "chatgpt",
            OpenAIConfig::with_key(&crate::providers::chatgpt::DIALECT, "token")
                .responses("gpt-5.4"),
        ),
    ] {
        for body in recorded_whole_replies(provider) {
            crate::test_utils::history::assert_restated_agrees(
                &wire,
                [WireFrame::Text(body.to_string())],
                frames(&restated_body(&body)),
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "the family records whole replies: {checked}");
}

/// A failed or cancelled reply ends the turn in an error, so it never
/// replays.
#[test]
fn a_failed_or_cancelled_reply_ends_in_an_error() {
    for status in ["failed", "cancelled"] {
        let mut body = sample_response(status);
        body["output"] = json!([message("msg_1", "partial")]);
        let decoded = crate::test_utils::decode_reply(
            &wire(),
            &CompletionRequest::new("hello"),
            Mode::Unary,
            vec![WireFrame::Text(body.to_string())],
            serde_json::Value::Null,
        );
        let response = decoded.expect("the reply decodes");
        assert!(
            matches!(response.stop(), crate::message::StopReason::Error(_)),
            "{:?}",
            response.stop()
        );
    }
}

/// Unknown nested parts replay exactly; an invalid status on a known
/// function item is typed corruption rather than an unknown output kind.
#[test]
fn unknown_nested_parts_survive_and_invalid_known_function_status_is_corrupt() {
    let output = vec![
        json!({"type": "reasoning", "id": "rs_1", "summary": [
            {"type": "summary_text", "text": "Think."},
            {"type": "summary_image", "url": "u"},
        ]}),
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [
            {"type": "output_text", "text": "Done.", "annotations": []},
            {"type": "output_audio", "transcript": "Done."},
        ]}),
    ];
    // The inherited helper emits known-text deltas even for unknown raw parts.
    // A35 uses genuine snapshots here, keeping kind-change refusals separate.
    let snapshots = [
        item_done(0,1,output[0].clone()),
        item_done(1,2,output[1].clone()),
        completed_with(3,json!(output)),
    ];
    for response in [
        decode(Mode::Streaming, frames(&snapshots)),
        decode(Mode::Unary, whole(&output)),
    ] {
        assert_eq!(natives(&response), output);
        let input = replayed_input(&response);
        assert_eq!(input.iter().find(|item| item["id"] == "rs_1").expect("reasoning replays")["summary"], output[0]["summary"]);
        assert_eq!(input.iter().find(|item| item["id"] == "msg_1").expect("message replays")["content"], output[1]["content"]);
    }
    let function = json!({"type":"function_call","id":"fc_1","call_id":"call_1",
        "name":"lookup","arguments":"{}","status":"searching"});
    let mut body = sample_response("completed");
    body["output"] = json!([function]);
    for (mode, frame, event_type) in [
        (Mode::Unary, body, None),
        (Mode::Streaming, json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":function}), Some("response.output_item.added")),
        (Mode::Streaming, json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":function}), Some("response.output_item.done")),
    ] {
        let raw = frame.to_string();
        let error = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), mode,
            [WireFrame::Text(raw.clone())], serde_json::Value::Null).expect_err("searching is invalid on a known function item");
        assert_eq!(error.kind(), ErrorKind::Json);
        let report = error.report();
        let Some(crate::error::ErrorDetail::CorruptFrame(detail)) = report.detail else {
            panic!("known function corruption must stay typed: {error:?}");
        };
        assert_eq!(detail.event_type.as_deref(), event_type);
        assert_eq!(detail.evidence, crate::error::FrameEvidence::Text(raw));
        assert!(detail.error.contains("searching"));
        assert!(!error.is_retryable());
    }
}

/// The history the next request sends for `response`, on the same model.
fn replayed_input(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    use crate::message::Message;
    let Some(Message::Assistant(turn)) = response.message() else {
        panic!("the reply is an assistant turn");
    };
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::user("next"),
    ];
    encoded_input(&wire(), request)
}

/// Sparse bodies refuse the Whole contract before status mapping. Well-shaped
/// bodies retain exact finish reasons and raw errors without inventing failures.
#[test]
fn sparse_whole_bodies_refuse_and_valid_statuses_keep_exact_finish_reasons() {
    use crate::completion::FinishReason;
    use crate::message::StopReason;
    let incomplete =
        |reason: &str| json!({"status": "incomplete", "incomplete_details": {"reason": reason}});
    let failed = |status: &str| json!({"status": status, "error": {"code": "server_error", "message": "boom"}});
    let cases: [(serde_json::Value, Option<FinishReason>, bool); 11] = [
        (
            json!({"status": "completed"}),
            Some(FinishReason::Stop),
            false,
        ),
        (
            incomplete("max_output_tokens"),
            Some(FinishReason::Length),
            false,
        ),
        (
            incomplete("content_filter"),
            Some(FinishReason::ContentFilter),
            true,
        ),
        (
            incomplete("max_tool_calls"),
            Some(FinishReason::Other("max_tool_calls".into())),
            true,
        ),
        (
            json!({"status": "incomplete"}),
            Some(FinishReason::Other("incomplete".into())),
            true,
        ),
        (
            failed("failed"),
            Some(FinishReason::Other("failed".into())),
            true,
        ),
        (
            failed("cancelled"),
            Some(FinishReason::Other("cancelled".into())),
            true,
        ),
        (
            json!({"status": "queued"}),
            None,
            true, // G170: shape acceptance is not terminal finish proof.
        ),
        (
            json!({"status": "in_progress"}),
            None,
            true, // G170: shape acceptance is not terminal finish proof.
        ),
        (
            json!({"status": "paused"}),
            Some(FinishReason::Other("paused".into())),
            true,
        ),
        (json!({}), None, false),
    ];
    for (status, reason, fails) in cases {
        let mut body = status.clone();
        body["output"] = json!([message("msg_1", "Done.")]);
        let sparse = body.to_string();
        let error = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), Mode::Unary,
            [WireFrame::Text(sparse.clone())], serde_json::Value::Null).expect_err("a sparse original body is not a valid Whole");
        if status.get("error").is_some() {
            assert_eq!(error.kind(), ErrorKind::ProviderResponse, "the fork's ErrorEnvelope fallback retains priority");
            assert_eq!(error.report().provider_response_body(), Some(sparse.as_str()));
        } else {
            assert_eq!(error.kind(), ErrorKind::Json);
            let report = error.report();
            let Some(crate::error::ErrorDetail::CorruptFrame(detail)) = report.detail else {
                panic!("sparse Whole shape refusal must stay typed: {error:?}");
            };
            assert_eq!(detail.event_type, None);
            assert_eq!(detail.evidence, crate::error::FrameEvidence::Text(sparse));
            assert!(detail.error.contains("missing field"));
            assert!(!error.is_retryable(), "typed shape corruption cannot become a retry policy");
        }
        if status.get("status").is_none() {
            let mut missing_status = sample_response("completed");
            missing_status.as_object_mut().unwrap().remove("status");
            missing_status["output"] = body["output"].clone();
            let error = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), Mode::Unary,
                [WireFrame::Text(missing_status.to_string())], serde_json::Value::Null).expect_err("status remains required even when every other Whole field exists");
            assert!(matches!(error, crate::error::ProviderError::CorruptFrame(_)));
            continue;
        }
        let mut valid = sample_response(status["status"].as_str().unwrap());
        valid.as_object_mut().unwrap().extend(status.as_object().unwrap().clone());
        valid["output"] = body["output"].clone();
        let response = decode(Mode::Unary, vec![WireFrame::Text(valid.to_string())]);
        assert_eq!(response.finish_reason(), reason, "{status}");
        assert_eq!(
            response.stop().is_failure(),
            fails,
            "{status}: {:?}",
            response.stop()
        );
        if status.get("error").is_some() {
            assert_eq!(
                response.stop(),
                StopReason::Error(format!("Provider finish_reason: {}", status["status"].as_str().unwrap())),
                "Whole status is a finish reason; its raw error is not an invented canonical failure"
            );
            assert_eq!(response.raw["error"], status["error"]);
        }
        // G187: queued/in_progress lack genuine finish proof; their old
        // error-none observations are replaced by the exact terminal guard.
        if matches!(status["status"].as_str(), Some("queued" | "in_progress")) {
            assert_eq!(response.error.as_deref(),
                Some("the provider ended the reply without a finish reason"), "{status}");
        } else {
            assert!(response.error.is_none(), "a well-shaped Whole does not synthesize response.error: {status}");
        }
        assert_eq!(response.text(), "Done.");
    }
    for (status, reason) in [
        (json!({"status":"throttled"}), Some(FinishReason::Other("throttled".into()))),
        (json!({"status":""}), None),
        (incomplete("MAX_TOOL_CALLS"), Some(FinishReason::Other("MAX_TOOL_CALLS".into()))),
        (incomplete(""), Some(FinishReason::Other("incomplete".into()))),
    ] {
        let mut body = sample_response(status["status"].as_str().unwrap());
        body.as_object_mut().unwrap().extend(status.as_object().unwrap().clone());
        let response = decode(Mode::Unary, vec![WireFrame::Text(body.to_string())]);
        assert_eq!(response.finish_reason(), reason);
        // G187: the empty status also has no finish reason. Only that row's
        // old error-none observation is replaced; every other row stays exact.
        if status["status"].as_str() == Some("") {
            assert_eq!(response.error.as_deref(),
                Some("the provider ended the reply without a finish reason"));
        } else {
            assert!(response.error.is_none());
        }
        assert_eq!(response.raw["status"], status["status"]);
    }
    for required in ["id","object","created_at","status","model"] {
        let mut body = sample_response("completed");
        body.as_object_mut().unwrap().remove(required);
        let raw = body.to_string();
        let error = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), Mode::Unary,
            [WireFrame::Text(raw.clone())], serde_json::Value::Null).expect_err("every required Whole field is enforced by the real reader");
        let report = error.report();
        let Some(crate::error::ErrorDetail::CorruptFrame(detail)) = report.detail else {panic!("{required}: typed Whole shape refusal")};
        assert_eq!(detail.evidence,crate::error::FrameEvidence::Text(raw));
        assert!(detail.error.contains(&format!("missing field `{required}`")),"{required}: {}",detail.error);
    }
    let mut failed = sample_response("failed");
    failed["error"] = json!({"code":"server_error","message":"boom"});
    let event = json!({"type":"response.failed","sequence_number":1,"response":failed});
    let raw = event.to_string();
    let error = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), Mode::Streaming,
        [WireFrame::Text(raw.clone())], serde_json::Value::Null).expect_err("response.failed EVENT retains the provider's actual failure");
    assert_eq!(error.kind(), ErrorKind::ProviderResponse);
    assert_eq!(error.report().provider_response_body(), Some(raw.as_str()));
}

/// Accepted incomplete preserves observed partial reasoning and text;
/// announcing a hosted item supplies no completed replay authority.
#[test]
fn accepted_incomplete_keeps_partial_reasoning_and_text_without_added_hosted_items() {
    let mut incomplete = sample_response("incomplete");
    incomplete["incomplete_details"] = json!({"reason": "max_output_tokens"});
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,"sequence_number":0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0,"sequence_number":1,"summary_index":0,"item_id":"rs_1", "delta": "Thinking"}),
        json!({"type": "response.output_item.added", "output_index": 1,"sequence_number":2,
               "item": {"type": "message", "id": "msg_1", "status": "in_progress", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1,"sequence_number":3,"content_index":0,"item_id":"msg_1", "delta": "Hello there"}),
        json!({"type": "response.output_item.added", "output_index": 2,"sequence_number":4,
               "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}}),
        json!({"type": "response.incomplete","sequence_number":5, "response": incomplete}),
    ];
    let refused = crate::test_utils::decode_reply(&wire(), &CompletionRequest::new("hello"), Mode::Streaming,
        frames(&events), serde_json::Value::Null).expect_err("streamed incomplete is refused by default");
    assert_eq!(refused.kind(), ErrorKind::ProviderResponse);
    assert_eq!(refused.report().provider_response_body(), Some(events[5].to_string().as_str()));
    let accepted = crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", super::super::ResponsesStreamOptions::strict()
            .with_incomplete(super::super::IncompleteTerminal::Accept)),
        super::super::document::Response::with_strict_reader(false), "openai", frames(&events));
    let reasoning_ends = accepted.events().into_iter().filter_map(|event| match event {
        StreamEvent::End {content:AssistantContent::CarriedReasoning(reasoning),reasoning_authority,..} => Some((reasoning,reasoning_authority)),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(reasoning_ends.len(),1);
    assert_eq!(reasoning_ends[0].0.display_text(),"Thinking");
    assert_eq!(*reasoning_ends[0].1,Some(crate::streaming::ReasoningAuthority::Partial));
    let accepted_wire = wire().with_streamed_incomplete(super::super::IncompleteTerminal::Accept);
    let response = crate::test_utils::decode_reply(&accepted_wire, &CompletionRequest::new("hello"), Mode::Streaming,
        frames(&events), serde_json::Value::Null).expect("the real Wire selects explicit Accept for the genuine incomplete terminal");
    assert_eq!(accepted.outcome.as_ref().expect("branded writer reaches its genuine terminal").choice,response.choice);
    assert_eq!(response.stop(), crate::message::StopReason::Length);
    assert_eq!(response.finish_reason(), Some(crate::completion::FinishReason::Length));
    assert_eq!(response.reasoning(),"Thinking");
    assert_eq!(response.text(),"Hello there");
    assert!(response.error.is_none());
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "no block holds a snapshot: {:?}",
        response.choice
    );
    let input = replayed_input(&response);
    let assistant: Vec<&serde_json::Value> = input
        .iter()
        .filter(|item| item.get("role").and_then(serde_json::Value::as_str) != Some("user"))
        .collect();
    assert_eq!(assistant.len(), 2, "observed reasoning and text both go back: {input:?}");
    let reasoning = assistant.iter().find(|item| item["type"] == "reasoning").expect("partial reasoning replays");
    assert_eq!(reasoning["id"],"rs_1");
    assert_eq!(reasoning["summary"],json!([{"type":"summary_text","text":"Thinking"}]));
    let text = assistant.iter().find(|item| item["type"] == "message").expect("partial text replays");
    assert_eq!(text["content"][0]["text"], "Hello there");
    assert!(!response.choice.iter().any(|part| matches!(part,AssistantContent::ProviderItem(_))), "added-only hosted data is not a completed item");
    assert!(
        !json!(input).to_string().contains("ws_1"),
        "an unfinished hosted item stays home"
    );
}

/// G191 original announcement frames strictly refuse. The admitted stage fixture
/// retains both original behavior pins under the separately owned G194/G195 repairs.
#[test]
fn a_call_added_but_never_done_fails_the_turn() {
    use crate::wire::{Decoder, Wire};
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"q\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "\"rig\"}"}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"q\":\"rig\"}"}),
        completed_with(4, json!([])),
    ];
    let restated = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        completed_with(
            1,
            json!([function_call("fc_1", "call_1", r#"{"q":"rig"}"#)]),
        ),
    ];
    for subject in [&events[..], &restated[..]] {
        let raw = subject[0].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
             panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `sequence_number`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
        let mut scaffolded = subject[0].clone();
        scaffolded["sequence_number"] = json!(0);
        let raw = scaffolded.to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
             panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `status`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// pi's pairing rule: an edit of any one of `rs_1, msg_1, fc_1` keeps every
/// item id in the next request. An edited block is rebuilt under its
/// item's identity: reasoning keeps its id and ciphertext, text its id and
/// phase, a call its id.
#[test]
fn reasoning_message_and_call_ids_survive_an_edit_of_each_one() {
    let output = [
        reasoning("rs_1", &["Plan."]),
        json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Looking.", "annotations": []}],
        }),
        function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
    ];
    for edited in 0..3 {
        let mut response = decode(Mode::Unary, whole(&output));
        match response.choice.get_mut(edited) {
            Some(AssistantContent::CarriedReasoning(reasoning)) => {
                let crate::message::CarriedReasoningPart::Summary(text) = &mut reasoning.content[0] else { panic!("known summary"); };
                text.push_str(" (edited)");
            },
            Some(AssistantContent::CarriedText(text)) => text.text_mut().push_str(" (edited)"),
            Some(AssistantContent::ToolCall(call)) => {
                call.function
                    .arguments
                    .insert("q".to_owned(), json!("edited"));
            }
            other => panic!("block {edited}: {other:?}"),
        }
        assert!(
            response.choice[edited].native_item().is_none(),
            "the edit is stale"
        );
        let input = replayed_input(&response);
        let ids: Vec<&str> = input
            .iter()
            .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
            .collect();
        assert_eq!(
            ids,
            ["rs_1", "msg_1", "fc_1"],
            "edit of block {edited}: {input:?}"
        );
        let items: Vec<&serde_json::Value> = input
            .iter()
            .filter(|item| item.get("id").is_some())
            .collect();
        assert_eq!(items[1]["phase"], "commentary", "edit of block {edited}");
        if edited == 0 {
            assert_eq!(
                items[0]["encrypted_content"],
                output[0]["encrypted_content"]
            );
            assert_eq!(items[0]["summary"][0]["text"], "Plan. (edited)");
        }
    }
}

/// `events` as a gateway that names no `output_index` streams them.
fn without_indices(mut events: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    for event in &mut events {
        if let Some(event) = event.as_object_mut() {
            event.shift_remove("output_index");
        }
    }
    events
}

/// The ids of the items the next request sends back.
fn replayed_ids(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    replayed_input(response)
        .iter()
        .filter_map(|item| item.get("id").cloned())
        .collect()
}

/// G191 original missing-output-index frames strictly refuse. The retained-index
/// fixture separately preserves whole/stream equality without acceptance parity.
#[test]
fn a_stream_without_output_indices_agrees_with_its_whole_reply() {
    use crate::wire::{Decoder, Wire};
    let replies = [
        every_kind(),
        vec![
            reasoning("rs_1", &["First."]),
            message("msg_1", "Between."),
            reasoning("rs_2", &["Second."]),
            function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
        ],
    ];
    for output in replies {
        let original = without_indices(restated(&output));
        let raw = original[1].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
             panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `output_index`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// G191 original missing-index frames strictly refuse. The retained-index fixture
/// separately keeps the original no-ciphertext identity and replay predicates.
#[test]
fn reasoning_done_without_ciphertext_keeps_its_item_in_a_stream_without_indices() {
    use crate::wire::{Decoder, Wire};
    let mut thought = reasoning("rs_1", &["Plan."]);
    if let Some(fields) = thought.as_object_mut() {
        fields.shift_remove("encrypted_content");
    }
    let output = [thought.clone(), message("msg_1", "Hello")];
    let original = without_indices(restated(&output));
    let raw = original[1].to_string();
    let configured = wire();
    let decoder = configured.decoder_in(Mode::Streaming);
    let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
        panic!("the exact held fixture must remain a typed strict refusal");
    };
    assert_eq!(refused.error().to_string(), "missing field `output_index`");
    assert_eq!(refused.event_type(), Some("response.output_item.added"));
    assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
}

/// G191 original indexless Hel/lo frames strictly refuse. The explicitly required
/// coordinate fixture separately keeps one Hello and the exact terminal item order.
#[test]
fn a_terminal_only_item_does_not_repeat_text_streamed_without_indices() {
    use crate::wire::{Decoder, Wire};
    let events = [
        json!({"type": "response.output_text.delta", "delta": "Hel"}),
        json!({"type": "response.output_text.delta", "delta": "lo"}),
        completed_with(
            2,
            json!([reasoning("rs_1", &["Why."]), message("msg_1", "Hello")]),
        ),
    ];
    for at in [0, 1] {
        let raw = events[at].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
             panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `sequence_number`");
        assert_eq!(refused.event_type(), Some("response.output_text.delta"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
        let mut scaffolded = events[at].clone();
        scaffolded["sequence_number"] = json!(at);
        let raw = scaffolded.to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
             panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `output_index`");
        assert_eq!(refused.event_type(), Some("response.output_text.delta"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// Contradictory done text keeps delivered text; structured reasoning
/// adopts the provider's restatement without rewriting delivered deltas.
#[test]
fn contradictory_done_keeps_delivered_text_and_restates_structured_reasoning() {
    let output = [reasoning("rs_1", &["Plan B."]), message("msg_1", "Hello")];
    let mut events = restated(&output);
    for event in &mut events {
        match event["type"].as_str() {
            Some("response.output_text.delta") => event["delta"] = json!("Bye"),
            Some("response.reasoning_summary_text.delta") => event["delta"] = json!("Plan A"),
            _ => {}
        }
    }
    let decoded = crate::driver::feed_frames!(
        ResponsesDecoder::new("openai",super::super::ResponsesStreamOptions::strict()),
        super::super::document::Response::with_strict_reader(false),"openai",frames(&events));
    assert_eq!(texts_of(&decoded.events()),["Bye","Bye"],"both original text deltas remain observable");
    let reasoning_deltas = decoded.events().into_iter().filter_map(|event| match event {
        StreamEvent::Reasoning {text,..} => Some(text.as_str()), _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(reasoning_deltas,["Plan A","Plan A"],"both original reasoning deltas remain observable");
    let closes = decoded.events().into_iter().filter_map(|event| match event {
        StreamEvent::End {content:AssistantContent::CarriedReasoning(reasoning),reasoning_authority,..} => Some((reasoning,reasoning_authority)),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(closes.len(),1);
    assert_eq!(closes[0].0.display_text(),"Plan B.");
    assert_eq!(*closes[0].1,Some(crate::streaming::ReasoningAuthority::ProviderRestated));
    let response = decoded.outcome.expect("well-shaped known-text contradictions retain the delivered answer");
    assert_eq!(response.text(), "ByeBye");
    assert_eq!(response.reasoning(), "Plan B.");
    let mut retained = output.clone();
    retained[1]["content"][0]["text"] = json!("ByeBye");
    assert_eq!(natives(&response), retained);
    assert_eq!(response.raw["output"],json!(output),"the contradictory provider restatement remains exact diagnostic raw");
    let replay = replayed_input(&response);
    assert_eq!(replay.iter().find(|item| item["id"]=="rs_1").expect("reasoning replay")["summary"],output[0]["summary"]);
    assert_eq!(replay.iter().find(|item| item["id"]=="msg_1").expect("text replay")["content"][0]["text"],"ByeBye");
}

/// G172 keeps the untyped ProviderItem and its genuine issuer for replay.
/// G203 observes it inside the complete q/assistant/next request.
#[test]
fn an_item_without_a_type_replays_its_received_provider_payload() {
    let output = [
        json!({"id": "x_1", "payload": {"n": 1}}),
        message("msg_1", "Noted."),
    ];
    for response in [
        decode(Mode::Unary, whole(&output)),
        decode(Mode::Streaming, frames(&restated(&output))),
    ] {
        // G172: the genuine untyped item is a ProviderItem with its real issuer.
        let Some(AssistantContent::ProviderItem(item)) = response.choice.first() else {
            panic!("the untyped received item is carried whole: {:?}", response.choice);
        };
        assert_eq!(item.payload.item, output[0]);
        assert_eq!(item.payload.provider.as_deref(), Some("openai"));
        assert_eq!(response.raw["output"][0], output[0]);
        let input = replayed_input(&response);
        assert_eq!(input.len(), 4);
        assert_eq!(input[0], json!({"type":"message","role":"user",
            "content":[{"type":"input_text","text":"q"}]}));
        assert_eq!(input[3], json!({"type":"message","role":"user",
            "content":[{"type":"input_text","text":"next"}]}));
        assert_eq!(input.iter().map(|item| (item["type"].clone(), item["id"].clone()))
            .collect::<Vec<_>>(), [
                (json!("message"), serde_json::Value::Null),
                (serde_json::Value::Null, json!("x_1")),
                (json!("message"), json!("msg_1")),
                (json!("message"), serde_json::Value::Null),
            ]);
        let replayed = input.iter().find(|item| item["id"] == "x_1" && item.get("type").is_none())
            .expect("the actual untyped item replays inside the full ordered input");
        assert_eq!(replayed, &output[0]);
        assert_eq!(replayed_ids(&response), [json!("x_1"), json!("msg_1")]);
    }
}

/// G193 absent statuses strictly refuse and present completed statuses keep default
/// Refuse. A separate explicit Accept fixture preserves the original stop predicates.
#[test]
fn response_incomplete_is_incomplete_whatever_its_status_says() {
    use crate::message::StopReason;
    for (status, reason, _stop) in [
        (None, Some("max_output_tokens"), Some(StopReason::Length)),
        (
            Some("completed"),
            Some("max_output_tokens"),
            Some(StopReason::Length),
        ),
        (None, None, None),
        (Some("completed"), None, None),
        (None, Some("content_filter"), None),
    ] {
        let mut response = sample_response("incomplete");
        match status {
            Some(status) => response["status"] = json!(status),
            None => {
                if let Some(fields) = response.as_object_mut() {
                    fields.shift_remove("status");
                }
            }
        }
        response["incomplete_details"] = json!(reason.map(|reason| json!({"reason": reason})));
        response["output"] = json!([message("msg_1", "partial")]);
        let events = [
            item_done(0, 1, message("msg_1", "partial")),
            json!({"type": "response.incomplete", "sequence_number": 2, "response": response}),
        ];
        use crate::wire::{Decoder, Wire};
        let received = events[1].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let classified = decoder.classify(WireFrame::Text(received.clone()));
        if status.is_none() {
            let crate::wire::WireEvent::Corrupt(refused) = classified else { panic!("original absent status strictly refuses"); };
            assert_eq!(refused.error().to_string(), "missing field `status`");
            assert_eq!(refused.event_type(), Some("response.incomplete"));
            assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(received));
        } else {
            let crate::wire::WireEvent::Known(super::super::ResponsesEvent::Frame { frame, raw, .. }) = classified else { panic!("present completed status is reader-valid"); };
            assert_eq!(frame, events[1]);
            assert_eq!(raw, received);
            let error = crate::test_utils::decode_reply(&configured, &CompletionRequest::new("hello"),
                Mode::Streaming, frames(&events), serde_json::Value::Null)
                .expect_err("G193 retains the default incomplete Refuse option");
            assert_eq!(error.kind(), ErrorKind::ProviderResponse);
            assert_eq!(error.report().provider_response_body(), Some(received.as_str()));
        }
    }
}

/// The compact shape of a decoded stream's events: each event's part and
/// payload.
fn call_events(decoded: &Decoded<Completion>) -> Vec<String> {
    decoded
        .events()
        .into_iter()
        .map(|event| match event {
            StreamEvent::Start { part, name, .. } => format!(
                "{}:start:{}",
                part.index(),
                name.as_ref().map_or("", |name| name.as_str())
            ),
            StreamEvent::Arguments { part, json } => format!("{}:args:{json}", part.index()),
            StreamEvent::Text { part, text } => format!("{}:text:{text}", part.index()),
            StreamEvent::Reasoning { part, .. } => format!("{}:reasoning", part.index()),
            StreamEvent::Resume { part } => format!("{}:resume", part.index()),
            StreamEvent::Input { part, input } => format!("{}:input:{input}", part.index()),
            StreamEvent::Discard { part, reason } => format!("{}:discard:{reason:?}", part.index()),
            StreamEvent::End { part, content, .. } => match content {
                AssistantContent::ToolCall(call) => {
                    format!("{}:end:{}", part.index(), call.function.arguments_value())
                }
                _ => format!("{}:end", part.index()),
            },
        })
        .collect()
}

fn added_call(index: usize, id: &str, call_id: &str, name: &str) -> serde_json::Value {
    json!({"type": "response.output_item.added", "output_index": index,
           "item": {"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": ""}})
}

fn argument_delta(index: usize, delta: &str) -> serde_json::Value {
    json!({"type": "response.function_call_arguments.delta", "output_index": index, "delta": delta})
}

fn done_call(
    index: usize,
    id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> serde_json::Value {
    json!({"type": "response.output_item.done", "output_index": index,
           "item": {"type": "function_call", "id": id, "call_id": call_id, "name": name,
                    "arguments": arguments, "status": "completed"}})
}

/// G191 original absent sequence/status frames strictly refuse. The separately
/// admitted stage fixture preserves every original parallel-call event pin.
#[test]
fn argument_deltas_stream_before_the_done_item() {
    use crate::wire::{Decoder, Wire};
    let events = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        added_call(1, "fc_2", "call_2", "lookup"),
        argument_delta(1, "{\"q\":\"ri"),
        argument_delta(0, "1}"),
        argument_delta(1, "g\"}"),
        done_call(0, "fc_1", "call_1", "add", "{\"x\":1}"),
        done_call(1, "fc_2", "call_2", "lookup", "{\"q\":\"rig\"}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    for at in [0, 2] {
        let raw = events[at].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
            panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `sequence_number`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
        let mut scaffolded = events[at].clone();
        scaffolded["sequence_number"] = json!(at);
        let raw = scaffolded.to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
            panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `status`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// G191 original frames strictly refuse. G192's admitted fixture retains real
/// fragments and complete Ends, naming the old extra suffix-delta loss.
#[test]
fn a_done_item_sends_the_rest_or_restates_the_arguments() {
    use crate::wire::{Decoder, Wire};
    let extended = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        done_call(0, "fc_1", "call_1", "add", "{\"x\":1}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    let restated = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        done_call(0, "fc_1", "call_1", "add", "{\"y\":2}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    for subject in [&extended[..], &restated[..]] {
        let raw = subject[0].to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
            panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `sequence_number`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
        let mut scaffolded = subject[0].clone();
        scaffolded["sequence_number"] = json!(0);
        let raw = scaffolded.to_string();
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
            panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), "missing field `status`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// Original custom delta/done tags stay exact Unknown values. A completed
/// custom item emits raw Input and a distinct call with namespace and dual IDs.
#[test]
fn custom_tags_stay_unknown_and_completed_items_deliver_raw_input() {
    let input = "echo \"hi\"\nls";
    for namespace in [None,Some(""),Some("functions")] {
        let mut added = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"shell","input":""});
        let mut done = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"shell","input":input,"status":"completed"});
        if let Some(namespace) = namespace {
            added["namespace"] = json!(namespace);
            done["namespace"] = json!(namespace);
        }
        let events = [
        json!({"type": "response.output_item.added", "output_index": 0,"sequence_number":0,
               "item": added}),
        json!({"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "echo \"hi"}),
        json!({"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "\"\nls"}),
        json!({"type":"response.custom_tool_call_input.done","output_index":0,"input":input}),
        json!({"type": "response.output_item.done", "output_index": 0,"sequence_number":4,"item":done}),
        json!({"type": "response.completed","sequence_number":5, "response": sample_response("completed")}),
    ];
    let decoded = decoded_body(&body_of(&events));
    let unknown = decoded.items.iter().filter_map(|item| match item {
        Ok(Item::Unknown(payload)) => Some(payload.value().clone()), _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(unknown,events[1..4],"both original deltas and the custom done tag pass through without wrapping");
    assert!(!decoded.events().iter().any(|event|matches!(event,StreamEvent::Arguments {..})),"custom input is never function argument JSON");
    let starts = decoded.events().into_iter().filter_map(|event|match event {
        StreamEvent::Start {part,kind:PartKind::CustomToolCall,name,namespace,..} => Some((part,name,namespace)),_=>None,
    }).collect::<Vec<_>>();
    assert_eq!(starts.len(),1);
    assert_eq!(starts[0].1.as_ref().map(|name|name.as_str()),Some("shell"));
    assert_eq!(starts[0].2.as_deref(),namespace);
    let fragments: Vec<_> = decoded
        .events()
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::Input { part,input } => Some((*part,input.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(fragments,[(*starts[0].0,input)],"the completed whole custom input reaches its real Part once");
    let closes = decoded.events().into_iter().filter_map(|event|match event {
        StreamEvent::End {part,content:AssistantContent::CustomToolCall(call),..} => Some((part,call)),_=>None,
    }).collect::<Vec<_>>();
    assert_eq!(closes.len(),1);
    assert_eq!(closes[0].0,starts[0].0);
    assert_eq!(closes[0].1.input,input);
    assert_eq!(closes[0].1.namespace.as_deref(),namespace);
    assert_eq!(closes[0].1.id.provider().expect("call correlator retained").as_str(),"call_1");
    assert_eq!(closes[0].1.native.as_ref().expect("item identity retained").item,done);
    assert_eq!(closes[0].1.native.as_ref().unwrap().item["id"],"ctc_1");
    let response = decoded.outcome.expect("the custom item reaches its genuine terminal");
    let [AssistantContent::CustomToolCall(call)] = response.choice.as_slice() else {panic!("one distinct custom call")};
    assert_eq!(call.input,input);
    assert_eq!(call.namespace.as_deref(),namespace);
    assert_eq!(call.id.provider().unwrap().as_str(),"call_1");
    assert_eq!(call.native.as_ref().unwrap().item,done);
    }
}

/// Added arguments are announcement fields. Only actual deltas stream;
/// authoritative done sets final arguments without a synthesized suffix.
#[test]
fn added_arguments_do_not_stream_and_done_restates_without_extra_fragments() {
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,"sequence_number":0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": "{\"x\":","status":"in_progress"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"sequence_number":1,"delta":""}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"sequence_number":2,"delta":"1}"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":3,
            "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"add","arguments":"{\"x\":1}","status":"completed"}}),
        json!({"type": "response.completed","sequence_number":4, "response": sample_response("completed")}),
    ];
    assert_eq!(
        call_events(&decoded_body(&body_of(&events))),
        [
            "0:start:add",
            "0:args:1}",
            "0:end:{\"x\":1}"
        ]
    );
    let decoded = decoded_body(&body_of(&events));
    let response = decoded.outcome.expect("well-shaped done supplies authoritative valid arguments");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {panic!("one completed function call")};
    assert_eq!(call.function.arguments_value(),json!({"x":1}));
    assert_eq!(call.id.provider().unwrap().as_str(),"call_1");
    assert_eq!(call.native.as_ref().unwrap().item["id"],"fc_1");
}

/// A message whose two `output_text` parts state every annotation type the
/// API documents, and one of a type it does not.
fn cited_message() -> serde_json::Value {
    json!({
        "type": "message",
        "id": "msg_cited",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "Rig is a Rust library. ",
                "annotations": [
                    { "type": "url_citation", "start_index": 0, "end_index": 22, "url": "https://rig.rs", "title": "Rig" },
                    { "type": "file_citation", "index": 22, "file_id": "file-1", "filename": "notes.md" },
                ],
            },
            {
                "type": "output_text",
                "text": "See the chart.",
                "annotations": [
                    { "type": "container_file_citation", "start_index": 0, "end_index": 14, "container_id": "cntr_1", "file_id": "cfile_2", "filename": "chart.png" },
                    { "type": "file_path", "index": 14, "file_id": "file-3" },
                    { "type": "future_citation", "index": 0 },
                    { "type": "url_citation", "start_index": 0, "end_index": 3 },
                ],
            },
        ],
    })
}

/// The sources of `response`'s text blocks' citations, one list per
/// citation, after checking that each cites its whole block.
fn cited_sources(
    response: &crate::completion::CompletionResponse,
) -> Vec<Vec<crate::message::Source>> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.citations().to_vec()),
            AssistantContent::CarriedText(text) => Some(text.citations().to_vec()),
            _ => None,
        })
        .flatten()
        .map(|citation| {
            assert_eq!(citation.span, None, "the offsets' unit is undocumented");
            citation.sources
        })
        .collect()
}

/// Every documented annotation type becomes one citation of the whole text
/// block, in part order, on the unary and the streamed route alike; an
/// unknown type or one that names no source is left in the item. Hand-built:
/// the corpus records only `url_citation`, which the web search cassettes
/// assert.
#[test]
fn every_annotation_type_cites_the_whole_block_on_both_routes() {
    use crate::message::{Source, SourceLocation};

    let file = |file_id: &str, filename: Option<&str>, container_id: Option<&str>| {
        vec![Source::new(SourceLocation::File {
            file_id: file_id.to_owned(),
            filename: filename.map(str::to_owned),
            container_id: container_id.map(str::to_owned),
        })]
    };
    let expected = vec![
        vec![
            Source::new(SourceLocation::Url {
                url: "https://rig.rs".to_owned(),
            })
            .title("Rig"),
        ],
        file("file-1", Some("notes.md"), None),
        file("cfile_2", Some("chart.png"), Some("cntr_1")),
        file("file-3", None, None),
    ];
    let output = [cited_message()];
    let unary = decode(Mode::Unary, whole(&output));
    let streamed = decode(Mode::Streaming, frames(&restated(&output)));
    assert_eq!(cited_sources(&unary), expected);
    assert_eq!(cited_sources(&streamed), expected);
    assert_eq!(unary.choice, streamed.choice);
    assert_eq!(
        natives(&unary),
        output,
        "the item replays as stated, annotations and offsets included"
    );
    for response in [&unary, &streamed] {
        let input = replayed_input(response);
        let replayed = input.iter().find(|item| item["type"] == "message" && item["role"] == "assistant")
            .expect("the real next request contains the cited assistant message");
        assert_eq!(replayed["content"], output[0]["content"], "real request replay retains every annotation and its original offsets");
    }
}

/// A stream whose terminal restates a message its `output_item.done`
/// already cited keeps one copy of each citation. Hand-built: the recorded
/// citation streams pin the same through their cassette tests.
#[test]
fn a_restated_message_is_cited_once() {
    let output = [cited_message()];
    let streamed = decode(Mode::Streaming, frames(&restated(&output)));
    assert_eq!(cited_sources(&streamed).len(), 4);
}

/// The cost a Responses reply reports is its total in USD: xAI counts
/// 10^10 ticks per USD and OpenRouter states USD credits. A reply that
/// reports none leaves the cost to the catalog. Hand-built for the units;
/// the xAI and OpenRouter cassettes assert recorded figures.
#[test]
fn a_reported_cost_is_its_total_in_usd() {
    use crate::completion::Cost;

    let usage = |extra: serde_json::Value| {
        let mut usage = json!({ "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 });
        if let (Some(usage), Some(extra)) = (usage.as_object_mut(), extra.as_object()) {
            usage.extend(extra.clone());
        }
        super::super::usage_of(&usage).cost
    };
    assert_eq!(
        usage(json!({ "cost_in_usd_ticks": 184_183_500 })),
        Some(Cost::from_total(0.018_418_35))
    );
    assert_eq!(
        usage(
            json!({ "cost": 0.000_029_5, "cost_details": { "upstream_inference_cost": 0.000_029_5 } })
        ),
        Some(Cost::from_total(0.000_029_5))
    );
    assert_eq!(
        usage(json!({ "cost": "0.25" })),
        Some(Cost::from_total(0.25))
    );
    assert_eq!(usage(json!({ "cost": null })), None);
    assert_eq!(usage(json!({})), None);
}


// G191 NEW fixtures supply required sequence and absent call-stage status only.
// Existing values, names, IDs and argument literals are never overwritten.
fn admitted_call_stage_events(events: &[serde_json::Value]) -> Vec<serde_json::Value> {
    events.iter().enumerate().map(|(sequence, original)| {
        let mut event = original.clone();
        event.as_object_mut().expect("fixture envelope")
            .entry("sequence_number").or_insert_with(|| json!(sequence));
        let stage = match event["type"].as_str() {
            Some("response.output_item.added") => Some("in_progress"),
            Some("response.output_item.done") => Some("completed"),
            _ => None,
        };
        if event["item"]["type"] == "function_call" {
            if let Some(stage) = stage {
                event["item"].as_object_mut().expect("fixture function item")
                    .entry("status").or_insert_with(|| json!(stage));
            }
        }
        event
    }).collect()
}


#[test]
fn admitted_argument_deltas_stream_before_the_done_item() {
    let events = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        added_call(1, "fc_2", "call_2", "lookup"),
        argument_delta(1, "{\"q\":\"ri"),
        argument_delta(0, "1}"),
        argument_delta(1, "g\"}"),
        done_call(0, "fc_1", "call_1", "add", "{\"x\":1}"),
        done_call(1, "fc_2", "call_2", "lookup", "{\"q\":\"rig\"}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    let events = admitted_call_stage_events(&events);
    let decoded = decoded_body(&body_of(&events));
    assert_eq!(
        call_events(&decoded),
        [
            "0:start:add",
            "0:args:{\"x\":",
            "1:start:lookup",
            "1:args:{\"q\":\"ri",
            "0:args:1}",
            "1:args:g\"}",
            "0:end:{\"x\":1}",
            "1:end:{\"q\":\"rig\"}",
        ]
    );
}


#[test]
fn admitted_done_item_restates_arguments_without_fabricating_a_suffix_delta() {
    // G192 names only the old extra Arguments "1}" loss; prior fragment and full Ends stay exact.
    let extended = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        done_call(0, "fc_1", "call_1", "add", "{\"x\":1}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    let extended = admitted_call_stage_events(&extended);
    assert_eq!(
        call_events(&decoded_body(&body_of(&extended))),
        [
            "0:start:add",
            "0:args:{\"x\":",
            "0:end:{\"x\":1}"
        ]
    );
    let restated = [
        added_call(0, "fc_1", "call_1", "add"),
        argument_delta(0, "{\"x\":"),
        done_call(0, "fc_1", "call_1", "add", "{\"y\":2}"),
        json!({"type": "response.completed", "response": sample_response("completed")}),
    ];
    let restated = admitted_call_stage_events(&restated);
    assert_eq!(
        call_events(&decoded_body(&body_of(&restated))),
        ["0:start:add", "0:args:{\"x\":", "0:end:{\"y\":2}"]
    );
}


#[test]
fn admitted_never_done_call_fails_and_real_terminal_completed_call_keeps_its_item() {
    // Original behavior pins depend on the separately owned G194/G195 repairs.
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"q\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "\"rig\"}"}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"q\":\"rig\"}"}),
        completed_with(4, json!([])),
    ];
    let events = admitted_call_stage_events(&events);
    let response = decode(Mode::Streaming, frames(&events));
    assert!(
        response.stop().is_failure(),
        "the call was never finished: {:?}",
        response.stop()
    );
    let restated = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        completed_with(
            1,
            json!([function_call("fc_1", "call_1", r#"{"q":"rig"}"#)]),
        ),
    ];
    let restated = admitted_call_stage_events(&restated);
    let response = decode(Mode::Streaming, frames(&restated));
    assert_eq!(response.stop(), crate::message::StopReason::ToolUse);
    assert_eq!(
        natives(&response),
        [function_call("fc_1", "call_1", r#"{"q":"rig"}"#)],
        "a call the terminal restates is finished by it"
    );
}


#[test]
fn retained_index_stream_agrees_with_its_whole_reply() {
    let replies = [
        every_kind(),
        vec![
            reasoning("rs_1", &["First."]),
            message("msg_1", "Between."),
            reasoning("rs_2", &["Second."]),
            function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
        ],
    ];
    for output in replies {
        crate::test_utils::history::assert_restated_agrees(
            &wire(),
            whole(&output),
            frames(&restated(&output)),
        );
    }
}


#[test]
fn retained_index_reasoning_without_ciphertext_keeps_item_and_replayed_ids() {
    let mut thought = reasoning("rs_1", &["Plan."]);
    if let Some(fields) = thought.as_object_mut() {
        fields.shift_remove("encrypted_content");
    }
    let output = [thought.clone(), message("msg_1", "Hello")];
    let response = decode(Mode::Streaming, frames(&restated(&output)));
    assert_eq!(natives(&response), output);
    assert_eq!(replayed_ids(&response), [json!("rs_1"), json!("msg_1")]);
}


#[test]
fn required_coordinate_text_and_terminal_reasoning_keep_one_hello_and_exact_items() {
    let mut events = [
        json!({"type": "response.output_text.delta", "delta": "Hel"}),
        json!({"type": "response.output_text.delta", "delta": "lo"}),
        completed_with(
            2,
            json!([reasoning("rs_1", &["Why."]), message("msg_1", "Hello")]),
        ),
    ];
    // G191 NEW coordinates: terminal reasoning is at0 and message at1.
    // Exact Hel/lo and item fields stay unchanged; this is not missing-index parity.
    for at in [0, 1] {
        events[at]["sequence_number"] = json!(at);
        events[at]["output_index"] = json!(1);
        events[at]["content_index"] = json!(0);
    }
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(response.text(), "Hello");
    assert_eq!(
        natives(&response),
        [reasoning("rs_1", &["Why."]), message("msg_1", "Hello")]
    );
}


#[test]
fn explicit_accept_keeps_original_completed_status_inputs_and_exact_incomplete_outcomes() {
    use crate::message::StopReason;
    for (status, reason, stop) in [
        (None, Some("max_output_tokens"), Some(StopReason::Length)),
        (
            Some("completed"),
            Some("max_output_tokens"),
            Some(StopReason::Length),
        ),
        (None, None, None),
        (Some("completed"), None, None),
        (None, Some("content_filter"), None),
    ] {
        if status.is_none() { continue; } // All three absent-status cases stay in original strict control.
        let mut response = sample_response("incomplete");
        match status {
            Some(status) => response["status"] = json!(status),
            None => {
                if let Some(fields) = response.as_object_mut() {
                    fields.shift_remove("status");
                }
            }
        }
        response["incomplete_details"] = json!(reason.map(|reason| json!({"reason": reason})));
        response["output"] = json!([message("msg_1", "partial")]);
        let events = [
            item_done(0, 1, message("msg_1", "partial")),
            json!({"type": "response.incomplete", "sequence_number": 2, "response": response}),
        ];
        use crate::wire::{Decoder, Wire};
        let configured = wire().with_streamed_incomplete(super::super::IncompleteTerminal::Accept);
        let received = events[1].to_string();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Known(super::super::ResponsesEvent::Frame { frame, raw, .. }) =
            decoder.classify(WireFrame::Text(received.clone())) else { panic!("received full root strictly classifies"); };
        assert_eq!(frame, events[1]);
        assert_eq!(raw, received); // Full received completed-status evidence is unedited.
        let response = crate::test_utils::decode_reply(&configured, &CompletionRequest::new("hello"),
            Mode::Streaming, frames(&events), serde_json::Value::Null).expect("explicit Accept reaches its genuine terminal");
        match stop {
            Some(stop) => assert_eq!(response.stop(), stop, "{status:?} {reason:?}"),
            None => assert!(
                response.stop().is_failure(),
                "{status:?} {reason:?}: {:?}",
                response.stop()
            ),
        }

        assert_eq!(response.text(), "partial");
        // Existing reassembly is event-authoritative; tests never edit received status.
        assert_eq!(response.raw["status"], json!("incomplete"));
        for key in ["id", "object", "created_at", "model", "output", "tools", "incomplete_details"] {
            assert_eq!(response.raw[key], events[1]["response"][key], "unchanged raw root member {key}");
        }
        assert_eq!(events[1]["response"]["status"], json!("completed"));
    }
}
