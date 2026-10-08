use super::*;
use crate::driver::Model;
use crate::message::AssistantContent;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use futures::StreamExt;

/// The recorded request of `crates/rig-cassette/fixtures/cassettes/cohere/agent/
/// max_tokens_sets_max_tokens_finish_reason.yaml` (`POST /v2/chat`).
const RECORDED_REQUEST: &str = r#"{"documents":[],"max_tokens":4,"messages":[{"content":[{"text":"Write a detailed fifty-word description of the ocean.","type":"text"}],"role":"user"}],"model":"command-a-03-2025"}"#;

/// That cassette's recorded reply, byte for byte.
const UNARY_BODY: &str = r#"{"finish_reason":"MAX_TOKENS","id":"20ae3cc4-46d2-4e78-8566-83649fbfc218","message":{"content":[{"text":"The ocean,","type":"text"}],"role":"assistant"},"usage":{"billed_units":{"input_tokens":11,"output_tokens":3},"cached_tokens":448,"tokens":{"input_tokens":506,"output_tokens":4}}}"#;

/// The same turn as a stream: the frame shapes are verbatim from
/// `crates/rig-cassette/fixtures/cassettes/cohere/streaming/streaming_smoke.yaml` (`message-start`
/// carrying the id, an empty `content-start`, `content-delta` text
/// fragments, `content-end`, then `message-end` with usage and finish
/// reason), carrying the unary reply's id, text, usage and finish reason.
/// Cohere records the two modes from separate calls, so a byte-identical
/// turn only exists when it is built this way — and the property under test
/// is exactly that the two shapes fold alike.
const STREAM_BODY: &str = concat!(
    "event: message-start\n",
    r#"data: {"delta":{"message":{"citations":[],"content":[],"role":"assistant","tool_calls":[],"tool_plan":""}},"id":"20ae3cc4-46d2-4e78-8566-83649fbfc218","type":"message-start"}"#,
    "\n\n",
    "event: content-start\n",
    r#"data: {"delta":{"message":{"content":{"text":"","type":"text"}}},"index":0,"type":"content-start"}"#,
    "\n\n",
    "event: content-delta\n",
    r#"data: {"delta":{"message":{"content":{"text":"The ocean"}}},"index":0,"type":"content-delta"}"#,
    "\n\n",
    "event: content-delta\n",
    r#"data: {"delta":{"message":{"content":{"text":","}}},"index":0,"type":"content-delta"}"#,
    "\n\n",
    "event: content-end\n",
    r#"data: {"index":0,"type":"content-end"}"#,
    "\n\n",
    "event: message-end\n",
    r#"data: {"delta":{"finish_reason":"MAX_TOKENS","usage":{"billed_units":{"input_tokens":11,"output_tokens":3},"cached_tokens":448,"tokens":{"input_tokens":506,"output_tokens":4}}},"type":"message-end"}"#,
    "\n\n",
    "data: [DONE]\n\n",
);

fn cohere() -> CohereConfig {
    CohereConfig::new("cohere-test-key")
}

/// The request the recorded cell sent.
fn recorded_request() -> CompletionRequest {
    CompletionRequest::new("Write a detailed fifty-word description of the ocean.").max_tokens(4)
}

fn body_of(encoded: &Encoded) -> serde_json::Value {
    crate::test_utils::json_body(&encoded.request)
}

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_turn() {
    let buffered = Model::new(
        cohere().completion("command-a-03-2025"),
        RecordingHttpClient::new(UNARY_BODY),
    )
    .call(recorded_request())
    .await
    .expect("the recorded reply decodes");

    let streaming = Model::new(
        cohere().completion("command-a-03-2025"),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from_static(STREAM_BODY.as_bytes()),
        },
    );
    let mut response = streaming
        .stream(recorded_request())
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    assert_eq!(text_of(&buffered.choice), "The ocean,");
    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    // Cohere's `/v2/chat` names no model in either mode.
    assert_eq!(buffered.model(), streamed.model());
    assert_eq!(buffered.response_id(), streamed.response_id());
    assert_eq!(
        buffered.usage.output_tokens,
        Some(4),
        "the total-usage counter is the one that is read, not `billed_units`"
    );
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

/// The two modes differ by `stream` and by nothing else — both spellings are
/// recorded (`agent/max_tokens_sets_max_tokens_finish_reason.yaml` without,
/// `streaming/streaming_smoke.yaml` with).
#[test]
fn the_mode_is_the_only_difference_between_the_two_requests() {
    let wire = cohere().completion("command-a-03-2025");
    let unary = wire
        .encode(recorded_request(), Mode::Unary)
        .expect("the request encodes");
    let streamed = wire
        .encode(recorded_request(), Mode::Streaming)
        .expect("the request encodes");

    assert_eq!(
        body_of(&unary),
        serde_json::from_str::<serde_json::Value>(RECORDED_REQUEST).expect("the fixture is JSON")
    );
    assert_eq!(unary.framing, Framing::Whole);

    let mut expected =
        serde_json::from_str::<serde_json::Value>(RECORDED_REQUEST).expect("the fixture is JSON");
    expected["stream"] = serde_json::Value::Bool(true);
    assert_eq!(body_of(&streamed), expected);
    assert_eq!(streamed.framing, Framing::Sse);
}

#[tokio::test]
async fn debug_trace_header_is_not_a_provider_request_id() {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "x-debug-trace-id",
        http::HeaderValue::from_static("trace-only"),
    );
    let http =
        RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, UNARY_BODY, headers);
    let response = Model::new(cohere().completion("command-a-03-2025"), http)
        .call(recorded_request())
        .await
        .expect("native reply");
    assert_eq!(response.provider_request_id, None);
    assert_eq!(
        response.response_id(),
        Some("20ae3cc4-46d2-4e78-8566-83649fbfc218")
    );
    assert_eq!(response.message_id, None);
}
