use super::super::super::completion::CLAUDE_SONNET_4_6;
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::MockStreamingClient;
use futures::StreamExt;

const MESSAGE_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}"#;
const TEXT_START: &str =
    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
const TEXT_DELTA: &str =
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
const MESSAGE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#;

fn sse(frames: &[&str]) -> bytes::Bytes {
    bytes::Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    )
}

/// The texts the stream yielded, whether an error item came, and what
/// it finished with.
async fn collect(
    sse_bytes: bytes::Bytes,
) -> (
    Vec<String>,
    bool,
    Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
) {
    let bound = crate::driver::Model::new(
        AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
        MockStreamingClient { sse_bytes },
    );
    let request = crate::completion::CompletionRequest::new("hello");
    let stream = bound.stream(request).expect("stream should open");
    drain(stream).await
}

async fn drain(
    mut stream: crate::streaming::CompletionStream,
) -> (
    Vec<String>,
    bool,
    Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
) {
    let mut texts = Vec::new();
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(_) => {}
            Err(_) => saw_error = true,
        }
    }
    (texts, saw_error, stream.finish().await)
}

#[tokio::test]
async fn errored_stream_forwards_the_error_and_no_terminal_record() {
    use crate::test_utils::SequencedStreamingHttpClient;

    // A transport failure injected into the byte stream after some
    // content must be forwarded and must not be papered over with a
    // synthesized end.
    let bound = crate::driver::Model::new(
        AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
        SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])),
            Err(crate::http_client::Error::non_success_with_details(
                http::StatusCode::BAD_GATEWAY,
                http::HeaderMap::new(),
                "connection reset".to_string(),
            )),
        ]),
    );
    let request = crate::completion::CompletionRequest::new("hello");
    let stream = bound.stream(request).expect("stream should open");
    let (texts, saw_error, finished) = drain(stream).await;

    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the transport failure must reach the consumer");
    assert!(finished.is_err(), "a failed stream has no response");
}

/// `input_tokens` precedence between `message_start` and the terminal
/// `message_delta`, across all three wire splits at once.
///
/// Not a cassette test: one recording can only witness whichever split
/// the endpoint it was recorded against happens to use, and the defect
/// here is the *precedence rule* relating three of them — the gateway
/// split, Anthropic proper, and the inverse. The gateway split is also
/// covered end-to-end by the recorded
/// `anthropic::cassette::streaming::gateway_reports_input_tokens_on_message_delta`;
/// this pins the two cases a single recording structurally cannot show
/// beside it.
#[tokio::test]
async fn input_tokens_prefer_the_terminal_delta_and_fall_back_to_message_start() {
    fn message_start(input_tokens: usize) -> String {
        format!(
            r#"{{"type":"message_start","message":{{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":0}}}}}}"#
        )
    }
    fn message_delta(input_tokens: usize) -> String {
        format!(
            r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"input_tokens":{input_tokens},"output_tokens":3}}}}"#
        )
    }

    for (start, delta, expected, case) in [
        // OpenRouter's Anthropic Messages shape: `message_start`
        // reports a placeholder zero and the real prompt size lands on
        // the terminal `message_delta`.
        (
            message_start(0),
            message_delta(9),
            9,
            "a gateway reporting the prompt size on message_delta must reach the consumer",
        ),
        // A delta that omits `input_tokens` entirely — the Bedrock-compat
        // and older/leaner shapes. (Not current Anthropic, which sends
        // the count on both frames; that case is the one below, since
        // the two always agree.)
        (
            message_start(5),
            MESSAGE_DELTA.to_owned(),
            5,
            "a delta without input_tokens falls back to message_start",
        ),
        // Anthropic proper: both frames carry the same count.
        (
            message_start(5),
            message_delta(5),
            5,
            "agreeing frames report that count",
        ),
        // The inverse split: a zero on the delta must not erase the
        // real count `message_start` already gave us.
        (
            message_start(5),
            message_delta(0),
            5,
            "a zero on the delta must not erase the message_start count",
        ),
    ] {
        let (_texts, _saw_error, finished) =
            collect(sse(&[&start, TEXT_START, TEXT_DELTA, &delta])).await;

        let response = finished.expect("the turn must complete");
        assert_eq!(response.usage.input_tokens, Some(expected), "{case}");
    }
}

/// The first error a 200 stream carrying `frames` yields, with
/// `request-id: req_stream` on the reply.
async fn in_band_error(frames: &[&str]) -> anyhow::Result<crate::error::ProviderError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("request-id", http::HeaderValue::from_static("req_stream"));
    let bound = crate::driver::Model::new(
        AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
        crate::test_utils::NonSuccessStreamingClient {
            status: http::StatusCode::OK,
            headers,
            body: sse(frames),
        },
    );
    let mut stream = bound.stream(crate::completion::CompletionRequest::new("hello"))?;
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            return Ok(error);
        }
    }
    anyhow::bail!("the stream yielded no error")
}

const OVERLOADED: &str =
    r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;

/// An `overloaded_error` event mid-stream is the same transient
/// condition as the same envelope on an HTTP 529 or 503, so it gets the
/// same retry verdict.
///
/// Not a cassette test: an overload is a transient server condition that
/// cannot be recorded on demand, and the defect is the verdict the
/// envelope gets, which a hand-written frame pins exactly.
#[tokio::test]
async fn an_in_band_overloaded_event_is_retryable_like_its_http_rejection() -> anyhow::Result<()> {
    let error = in_band_error(&[MESSAGE_START, TEXT_START, TEXT_DELTA, OVERLOADED]).await?;
    let rejected = crate::error::ProviderError::from_http_response(
        http::StatusCode::SERVICE_UNAVAILABLE,
        OVERLOADED,
    );
    anyhow::ensure!(rejected.is_retryable(), "the HTTP rejection is retryable");
    anyhow::ensure!(
        error.is_retryable(),
        "the same envelope in-band must be retryable too: {error:?}"
    );
    let report = error.report();
    anyhow::ensure!(report.retryable, "the report must agree: {report:?}");
    anyhow::ensure!(
        report.code.as_deref() == Some("overloaded_error"),
        "the envelope's code is reported: {report:?}"
    );
    Ok(())
}

#[tokio::test]
async fn truncated_stream_yields_content_but_no_terminal_record() {
    let (texts, saw_error, finished) = collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])).await;
    assert_eq!(texts, ["hi"]);
    assert!(
        !saw_error,
        "streaming EOF permits already delivered partial content"
    );
    assert!(
        matches!(finished, Err(crate::error::ProviderError::Truncated)),
        "no genuine provider end"
    );
}

#[tokio::test]
async fn provider_error_event_stops_the_stream_before_a_later_terminal() {
    const ERROR_EVENT: &str =
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let (texts, saw_error, finished) = collect(sse(&[
        MESSAGE_START,
        TEXT_START,
        TEXT_DELTA,
        ERROR_EVENT,
        MESSAGE_DELTA,
    ]))
    .await;
    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the provider error must reach the consumer");
    assert!(
        finished.is_err(),
        "a later message_delta cannot complete the failed turn"
    );
}

#[tokio::test]
async fn streamed_error_envelope_preserves_the_verbatim_body() {
    const ENVELOPE: &str = r#"{"error":{"message":"Overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let error = in_band_error(&[MESSAGE_START, ENVELOPE])
        .await
        .expect("an in-band error");
    assert_eq!(error.provider_response_body(), Some(ENVELOPE));
}

#[tokio::test]
async fn malformed_frame_then_eof_yields_error_and_no_terminal_record() {
    let (texts, saw_error, finished) =
        collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, "{not json"])).await;
    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the malformed frame must reach the consumer");
    assert!(
        finished.is_err(),
        "a malformed frame followed by EOF has no provider end"
    );
}

#[tokio::test]
async fn malformed_frame_then_real_terminal_still_completes_the_stream() {
    let (texts, saw_error, finished) = collect(sse(&[
        MESSAGE_START,
        TEXT_START,
        TEXT_DELTA,
        "{not json",
        MESSAGE_DELTA,
    ]))
    .await;
    assert_eq!(texts, ["hi"]);
    assert!(saw_error, "the malformed frame must reach the consumer");
    let response = finished.expect("the genuine terminal completes the turn");
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(response.usage.output_tokens, Some(3));
}

#[tokio::test]
async fn terminal_raw_round_trips_into_the_terminal_type() {
    const STOP_SEQUENCE: &str = r#"{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"alpha"},"usage":{"output_tokens":3}}"#;
    let (_, _, finished) =
        collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, STOP_SEQUENCE])).await;
    let response = finished.expect("the turn completes");
    let raw = &response.raw;
    assert_eq!(raw["stop_reason"], "stop_sequence");
    assert_eq!(raw["stop_sequence"], "alpha");
    assert_eq!(raw["id"], "msg_1");
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let renormalized = crate::test_utils::decode_reply(
        &wire,
        &crate::completion::CompletionRequest::new("hello"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(raw.to_string())],
        raw.clone(),
    )
    .expect("the capture decodes through the real unary parser");
    assert_eq!(response.identity(), renormalized.identity());
    assert_eq!(response.finish_reason(), renormalized.finish_reason());
    assert_eq!(response.model(), renormalized.model());
    assert_eq!(response.usage, renormalized.usage);
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(response.usage.output_tokens, Some(3));
}
