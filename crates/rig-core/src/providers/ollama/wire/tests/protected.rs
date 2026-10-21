use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::message::AssistantContent;
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Encoded, Framing, Mode, Wire};
use futures::StreamExt;

/// The recorded request of `crates/rig-cassette/fixtures/cassettes/ollama/agent/max_tokens.yaml`
/// (`POST /api/chat`): `max_tokens` rides in `options.num_predict`, and
/// `stream` is spelled out.
const RECORDED_REQUEST: &str = r#"{"messages":[{"content":"You are a concise assistant. Answer directly.","role":"system"},{"content":"In one or two sentences, explain what Rust programming language is and why memory safety matters.","role":"user"}],"model":"qwen3:4b","options":{"num_predict":24},"stream":false,"think":false}"#;

/// That cassette's recorded reply, byte for byte.
const UNARY_BODY: &str = r#"{"created_at":"1970-01-01T00:00:00Z","done":true,"done_reason":"length","eval_count":24,"eval_duration":318697665,"load_duration":4220408500,"message":{"content":"Hmm, the user wants a concise explanation of Rust and why memory safety matters. They specifically asked for one or two sentences","role":"assistant"},"model":"qwen3:4b","prompt_eval_count":42,"prompt_eval_duration":2035633750,"total_duration":6668117083}"#;

/// The same turn as a stream. The record shapes are verbatim from
/// `crates/rig-cassette/fixtures/cassettes/ollama/streaming/streaming_smoke.yaml` — content
/// records with `"done":false`, then an empty-content `"done":true` record
/// carrying the counters — and they carry the unary reply's text, model,
/// counters and `done_reason`. Ollama records the two modes from separate
/// calls, so a byte-identical turn only exists when it is built this way,
/// and the property under test is that the two fold alike.
const STREAM_BODY: &str = concat!(
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"Hmm, the user wants a concise explanation of Rust and why memory safety matters."},"done":false}"#,
    "\n",
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":" They specifically asked for one or two sentences"},"done":false}"#,
    "\n",
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","total_duration":6668117083,"load_duration":4220408500,"prompt_eval_count":42,"prompt_eval_duration":2035633750,"eval_count":24,"eval_duration":318697665}"#,
    "\n",
);

/// The request the recorded cell sent: a preamble, a prompt, 24 tokens, and
/// `think: false` through the provider escape hatch.
fn recorded_request() -> CompletionRequest {
    let mut request = CompletionRequest::new(
        "In one or two sentences, explain what Rust programming language is and why memory safety matters.",
    );
    request.chat_history.insert(
        0,
        crate::message::Message::system("You are a concise assistant. Answer directly."),
    );
    request.max_tokens = Some(24);
    request.additional_params = Some(serde_json::json!({"think": false}));
    request
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
        OllamaConfig::new().completion("qwen3:4b"),
        RecordingHttpClient::new(UNARY_BODY),
    )
    .call(recorded_request())
    .await
    .expect("the recorded reply decodes");

    let streaming = Model::new(
        OllamaConfig::new().completion("qwen3:4b"),
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

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model(), streamed.model());
    assert_eq!(
        text_of(&buffered.choice),
        "Hmm, the user wants a concise explanation of Rust and why memory safety matters. They \
         specifically asked for one or two sentences"
    );
    assert_eq!(buffered.usage.input_tokens, Some(42));
    assert_eq!(buffered.usage.output_tokens, Some(24));
    assert_eq!(buffered.usage.total_tokens, Some(66));
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(buffered.model(), Some("qwen3:4b"));
}

/// The mode picks `stream` and the framer, and changes nothing else. Both
/// spellings are recorded (`agent/max_tokens.yaml` false,
/// `streaming/streaming_smoke.yaml` true).
#[test]
fn the_mode_is_the_only_difference_between_the_two_requests() {
    let wire = OllamaConfig::new().completion("qwen3:4b");
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
    // A streamed reply is newline-delimited JSON, never SSE.
    assert_eq!(streamed.framing, Framing::Ndjson);
}

/// A reasoning model that puts its reasoning in `content` is split on the
/// whole reply only: the shape is the one recorded in
/// `crates/rig-cassette/fixtures/cassettes/ollama/structured_output/raw_with_thinking.yaml`.
#[tokio::test]
async fn a_buffered_reply_splits_legacy_reasoning_out_of_its_content() {
    let body = r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"<think>weighing it up</think>the answer"},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}"#;
    let response = Model::new(
        OllamaConfig::new().completion("deepseek-r1"),
        RecordingHttpClient::new(body),
    )
    .call(recorded_request())
    .await
    .expect("the reply decodes");

    assert_eq!(text_of(&response.choice), "the answer");
    assert!(
        response.choice.iter().any(|block| matches!(
            block,
            AssistantContent::Reasoning(reasoning)
                if reasoning.text == "weighing it up"
        )),
        "the reasoning must survive into history: {:?}",
        response.choice
    );
}

/// A streamed fragment that merely opens a `<think>` marker is not a whole
/// content: splitting it would strip text the turn never finished writing.
#[tokio::test]
async fn a_streamed_fragment_is_never_split_as_legacy_reasoning() {
    let stream = concat!(
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"<think>weighing"},"done":false}"#,
        "\n",
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":" it up</think>the answer"},"done":false}"#,
        "\n",
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}"#,
        "\n",
    );
    let bound = Model::new(
        OllamaConfig::new().completion("deepseek-r1"),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(stream),
        },
    );
    let mut response = bound.stream(recorded_request()).expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    assert_eq!(
        text_of(&streamed.choice),
        "<think>weighing it up</think>the answer"
    );
}
