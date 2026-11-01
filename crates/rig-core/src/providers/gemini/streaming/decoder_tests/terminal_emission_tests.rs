use super::*;
use crate::driver::Model;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::{MockStreamingClient, SequencedStreamingHttpClient};
use futures::StreamExt;

const CONTENT_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}],"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;
const TERMINAL_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;

fn sse(frames: &[&str]) -> bytes::Bytes {
    bytes::Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    )
}

async fn collect(
    frames: &[&str],
) -> (Vec<String>, bool, Result<CompletionResponse, ProviderError>) {
    let model = Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: sse(frames),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut texts = Vec::new();
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(Item::Event(
                StreamEvent::Start { .. }
                | StreamEvent::Resume { .. }
                | StreamEvent::Reasoning { .. }
                | StreamEvent::Arguments { .. }
                | StreamEvent::Input { .. }
                | StreamEvent::End { .. }
                | StreamEvent::Discard { .. },
            ))
            | Ok(Item::Unknown(_)) => {}
            Err(_) => saw_error = true,
        }
    }
    (texts, saw_error, stream.finish().await)
}

#[tokio::test]
async fn truncated_stream_yields_content_but_no_terminal_record() {
    let (texts, saw_error, response) = collect(&[CONTENT_CHUNK]).await;
    assert_eq!(texts, ["hi"]);
    assert!(!saw_error);
    assert!(
        matches!(response, Err(ProviderError::Truncated)),
        "EOF without a finishReason chunk must not synthesize a terminal record"
    );
}

#[tokio::test]
async fn silent_stream_has_no_error_or_terminal_record() {
    let (texts, saw_error, response) =
        collect(&[r#"{"candidates":[],"usageMetadata":{"promptTokenCount":9}}"#]).await;
    assert!(texts.is_empty());
    assert!(!saw_error, "a stream that delivered nothing is truncation");
    assert!(matches!(response, Err(ProviderError::Truncated)));
}

#[tokio::test]
async fn malformed_frame_then_eof_yields_error_and_no_terminal_record() {
    let (texts, saw_error, response) = collect(&[CONTENT_CHUNK, "{not json"]).await;
    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the malformed frame must reach the consumer");
    assert!(
        response.is_err(),
        "a parse error followed by EOF must not read as a completed turn"
    );
}

#[tokio::test]
async fn malformed_frame_then_real_terminal_still_completes_the_stream() {
    let (texts, saw_error, response) = collect(&[CONTENT_CHUNK, "{not json", TERMINAL_CHUNK]).await;
    assert_eq!(texts, ["hi", "!"]);
    assert!(saw_error, "the malformed frame must reach the consumer");
    let terminal =
        response.expect("a genuine finishReason after a parse error completes the stream");
    assert_eq!(terminal.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(terminal.response_id(), Some("resp-1"));
}

#[tokio::test]
async fn id_only_frame_updates_terminal_metadata_without_content() {
    let (texts, saw_error, response) = collect(&[
        CONTENT_CHUNK,
        TERMINAL_CHUNK,
        r#"{"responseId":"last-response"}"#,
    ])
    .await;
    assert_eq!(texts, ["hi", "!"]);
    assert!(!saw_error);
    assert_eq!(
        response.expect("terminal record").response_id(),
        Some("last-response")
    );
}

#[tokio::test]
async fn tool_protocol_failure_ends_the_stream_without_draining_later_frames() {
    let frames = [
        CONTENT_CHUNK,
        r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"malformed function call","index":0}]}"#,
        r#"{"candidates":[{"content":{"parts":[{"text":"dead"}],"role":"model"},"index":0}]}"#,
        r#"{"someFutureField":{"x":1}}"#,
        TERMINAL_CHUNK,
    ];
    let model = Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: sse(&frames),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut texts = Vec::new();
    let mut saw_error = false;
    let mut items_after_error = 0usize;
    while let Some(item) = stream.next().await {
        if saw_error {
            items_after_error += 1;
        }
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(Item::Event(
                StreamEvent::Start { .. }
                | StreamEvent::Resume { .. }
                | StreamEvent::Reasoning { .. }
                | StreamEvent::Arguments { .. }
                | StreamEvent::Input { .. }
                | StreamEvent::End { .. }
                | StreamEvent::Discard { .. },
            ))
            | Ok(Item::Unknown(_)) => {}
            Err(error) => {
                assert!(error.to_string().contains("MalformedFunctionCall"));
                saw_error = true;
            }
        }
    }
    assert_eq!(texts, ["hi"]);
    assert!(
        saw_error,
        "the tool-protocol failure must reach the consumer"
    );
    assert_eq!(
        items_after_error, 0,
        "the in-band failure must end the stream: no later text, Unknown passthrough, or terminal"
    );
    assert!(stream.finish().await.is_err());
}

#[tokio::test]
async fn errored_stream_forwards_the_error_and_no_terminal_record() {
    let model = Model::new(
        wire(),
        SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[CONTENT_CHUNK])),
            Err(crate::http_client::Error::non_success_with_details(
                http::StatusCode::BAD_GATEWAY,
                http::HeaderMap::new(),
                "connection reset".to_owned(),
            )),
        ]),
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut texts = Vec::new();
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(Item::Event(
                StreamEvent::Start { .. }
                | StreamEvent::Resume { .. }
                | StreamEvent::Reasoning { .. }
                | StreamEvent::Arguments { .. }
                | StreamEvent::Input { .. }
                | StreamEvent::End { .. }
                | StreamEvent::Discard { .. },
            ))
            | Ok(Item::Unknown(_)) => {}
            Err(_) => saw_error = true,
        }
    }
    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the transport failure must reach the consumer");
    assert!(
        stream.finish().await.is_err(),
        "a failed stream must not synthesize a terminal record"
    );
}
