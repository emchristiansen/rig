//! Protected error custody, recovery, and absence of terminal proof.

use super::helpers::*;
use super::super::{document, ResponsesDecoder, ResponsesStreamOptions};
use crate::driver::feed_frames;
use crate::error::{ErrorKind, ErrorDetail};
use crate::message::AssistantContent;
use crate::streaming::{Item, StreamEvent};
use crate::wire::WireFrame;
use futures::StreamExt;
use serde_json::json;

// F streaming/tests.rs:417
#[test]
fn a_buffered_body_preserves_its_error_payloads() {
    let mut response = sample_response("failed");
    response["error"] = json!({"code":"server_error","message":"response failed"});
    for event in [
        json!({"type":"response.failed","sequence_number":1,"response":response}),
        json!({"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}),
    ] {
        let payload = event.to_string();
        let error = feed_frames!(ResponsesDecoder::new("ChatGPT", ResponsesStreamOptions::strict()).with_envelope_repair(),
            document::Response::with_strict_reader(true), "ChatGPT", [WireFrame::Text(payload.clone())])
            .outcome.expect_err("buffered errors retain their full response body");
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse);
        assert_eq!(report.http_status, None);
        assert_eq!(report.provider_response_body(), Some(payload.as_str()));
    }
}

fn call_done() -> serde_json::Value {
    json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,
        "item":{"type":"function_call","id":"fc_123","arguments":"{}","call_id":"call_123",
            "name":"example_tool","status":"completed"}})
}

async fn flushed_call_then_error(stream: &mut crate::streaming::CompletionStream)
    -> (crate::message::ToolCall, crate::error::ErrorReport)
{
    let mut call = None;
    let error = loop {
        match stream.next().await.expect("call and terminal error must arrive") {
            Ok(Item::Event(StreamEvent::End { content: AssistantContent::ToolCall(value), .. })) => {
                assert!(call.replace(value).is_none(), "the call closes once");
            }
            Ok(Item::Event(StreamEvent::Start { kind: crate::streaming::PartKind::ToolCall, .. })) => {},
            Ok(other) => panic!("unexpected item before error: {other:?}"),
            Err(error) => break error.report(),
        }
    };
    let call = call.expect("the completed call precedes the error");
    assert!(stream.next().await.is_none(), "nothing follows the terminal error");
    assert!(stream.partial().response_id().is_none(), "error does not prove a terminal response");
    (call, error)
}

// F streaming/tests.rs:1034
#[tokio::test]
async fn response_failed_flushes_delivered_tool_calls_before_the_error() {
    let mut response = sample_response("failed");
    response["error"] = json!({"code":"server_error","message":"response stream failed"});
    let mut stream = responses_stream_of(&[
        call_done(), json!({"type":"response.failed","sequence_number":2,"response":response})
    ]).await;
    let (call, error) = flushed_call_then_error(&mut stream).await;
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_123"));
    assert_eq!(call.native.as_ref().map(|native| &native.item["call_id"]), Some(&json!("call_123")));
    assert_eq!(call.native.as_ref().map(|native| &native.item["id"]), Some(&json!("fc_123")));
    assert_eq!(call.function.name.as_str(), "example_tool");
    assert_eq!(error.kind, ErrorKind::ProviderResponse);
    assert_eq!(error.http_status, None);
    assert!(error.provider_response_body().is_some_and(|body| body.contains("response.failed") && body.contains("response stream failed")));
    assert!(stream.next().await.is_none());
}

// F streaming/tests.rs:1091
#[tokio::test]
async fn transport_error_flushes_delivered_tool_calls_before_the_error() {
    let chunks = vec![
        Ok(bytes::Bytes::from(format!("data: {}\n\n", call_done()))),
        Err(crate::http_client::Error::non_success_with_details(
            http::StatusCode::BAD_GATEWAY, http::HeaderMap::new(),
            r#"{"error":{"message":"upstream unavailable"}}"#.to_owned())),
    ];
    let mut stream = responses_stream(crate::test_utils::SequencedStreamingHttpClient::new(chunks)).await;
    let (call, error) = flushed_call_then_error(&mut stream).await;
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_123"));
    assert_eq!(call.native.as_ref().map(|native| &native.item["id"]), Some(&json!("fc_123")));
    assert_eq!(error.http_status, Some(http::StatusCode::BAD_GATEWAY.as_u16()));
    assert!(stream.next().await.is_none());
}

// F streaming/tests.rs:1137
#[tokio::test]
async fn known_terminal_with_malformed_usage_surfaces_error_without_terminal() {
    let mut event = completed_with_output(json!([]));
    event["sequence_number"] = json!(1);
    event["response"]["usage"] = json!("banana");
    let mut stream = responses_stream_of(&[event]).await;
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        let error = item.expect_err("a corrupt terminal emits no content or terminal proof");
        assert_eq!(error.report().kind, ErrorKind::Json);
        assert!(matches!(error.report().detail, Some(ErrorDetail::CorruptFrame(_))));
        saw_error = true;
    }
    assert!(saw_error);
    assert!(stream.partial().response_id().is_none());
    assert!(stream.partial().choice.is_empty());
    assert!(stream.finish().await.is_err());
}

// F streaming/tests.rs:1174
#[tokio::test]
async fn unknown_event_type_is_skipped_and_stream_completes() {
    let unknown = json!({"type":"response.rocket_launch","payload":{"count":3}});
    let mut completed = completed_with_output(json!([]));
    completed["sequence_number"] = json!(2);
    let mut stream = responses_stream_of(&[unknown.clone(), completed]).await;
    let mut unknowns = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Unknown(raw) = item.expect("unknown types are forward compatible") {
            unknowns.push(raw.value().clone());
        }
    }
    assert_eq!(unknowns, [unknown]);
    let response = stream.finish().await.expect("the genuine terminal still completes");
    assert_eq!(response.response_id(), Some("resp_123"));
}

// F streaming/tests.rs:1277
#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    let mut stream = responses_stream_of(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":1,"delta":"hel"}),
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":2,"delta":"lo"}),
    ]).await;
    let mut texts = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("clean EOF retains partial content without an error item") {
            texts.push(text);
        }
    }
    assert_eq!(texts, ["hel", "lo"]);
    assert_eq!(folded_texts(&stream.partial().choice), ["hello"]);
    assert!(stream.partial().response_id().is_none());
    assert!(stream.finish().await.is_err(), "unary finish still requires genuine terminal proof");
}

// F streaming/tests.rs:1364
#[tokio::test]
async fn streaming_http_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"quota exceeded"}}"#;
    let mut stream = responses_stream(crate::test_utils::HttpErrorStreamingClient::new(
        http::StatusCode::TOO_MANY_REQUESTS, body)).await;
    let error = stream.next().await.expect("HTTP error arrives").expect_err("non-success refuses").report();
    assert_eq!(error.http_status, Some(http::StatusCode::TOO_MANY_REQUESTS.as_u16()));
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_json().expect("valid JSON"), Some(json!({"error":{"message":"quota exceeded"}})));
    assert!(stream.next().await.is_none());
}

// F streaming/tests.rs:1398
#[test]
fn corrupt_known_frame_fails_the_buffered_body() {
    let corrupt = json!({"type":"response.output_text.delta","delta":42});
    let mut completed = completed_with_output(json!([]));
    completed["sequence_number"] = json!(2);
    let decode = |first: String| feed_frames!(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        document::Response::with_strict_reader(false), "openai",
        [WireFrame::Text(first), WireFrame::Text(completed.to_string())]);
    let error = decode(corrupt.to_string()).outcome.expect_err("buffered decode fails its first corruption");
    assert!(error.to_string().contains("response.output_text.delta"));
    decode("{not json".into()).outcome.expect_err("invalid JSON also refuses buffered decode");
    let unknown = json!({"type":"response.rocket_launch","count":3});
    let response = decode(unknown.to_string()).outcome.expect("unknown types remain skippable");
    assert_eq!(response.response_id(), Some("resp_123"));
}

// F streaming/tests.rs:2615
#[tokio::test]
async fn malformed_frame_surfaces_error_and_stream_still_completes() {
    let delta = json!({"type":"response.output_text.delta","content_index":0,"delta":"hello",
        "item_id":"msg_1","logprobs":[],"output_index":0,"sequence_number":1});
    let mut completed = completed_with_output(json!([]));
    completed["sequence_number"] = json!(2);
    let sse = format!("data: {delta}\n\ndata: {{not valid json\n\ndata: {completed}\n\n");
    let mut stream = responses_stream(crate::test_utils::MockStreamingClient { sse_bytes: sse.into() }).await;
    let mut text = String::new();
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text: fragment, .. })) => text.push_str(&fragment),
            Ok(Item::Event(StreamEvent::Start { .. } | StreamEvent::End { .. })) => {},
            Ok(other) => panic!("unexpected item: {other:?}"),
            Err(error) => {
                assert_eq!(error.report().kind, ErrorKind::Json);
                assert!(matches!(error.report().detail, Some(ErrorDetail::CorruptFrame(_))));
                saw_error = true;
            }
        }
    }
    assert_eq!(text, "hello");
    assert!(saw_error);
    let response = stream.finish().await.expect("draining recoverable errors leaves a genuine terminal");
    assert_eq!(response.response_id(), Some("resp_123"));
}
