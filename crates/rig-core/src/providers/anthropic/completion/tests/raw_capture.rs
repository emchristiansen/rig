use super::*;
use crate::test_utils::RecordingHttpClient;

const REQUEST_ID: &str = "req_unit_anthropic_0001";

/// A Messages body whose `stop_sequence` is set: the normalized
/// response maps it to `FinishReason::Stop` and drops which sequence
/// fired, so the capture provably answers more than the fold does.
const BODY: &str = r#"{
        "id": "msg_raw_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "stop_sequence",
        "stop_sequence": "alpha",
        "usage": {"input_tokens": 7, "output_tokens": 2}
    }"#;

fn http() -> RecordingHttpClient {
    let mut headers = http::HeaderMap::new();
    headers.insert("request-id", http::HeaderValue::from_static(REQUEST_ID));
    RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers)
}

/// The load-bearing capture property: `raw` is the reply Anthropic
/// sent, verbatim — it still carries the `type` tag the wire type does
/// not model, which is how you can tell it is the document and not a
/// projection of it — it deserializes into Anthropic's own
/// `CompletionResponse`, and it answers `stop_sequence`, which the
/// normalized response drops.
#[tokio::test]
async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let response = crate::driver::Model::new(wire, http())
        .call(hello_request())
        .await
        .expect("completion");

    let raw = &response.raw;
    assert_eq!(
        raw["type"], "message",
        "raw must be the verbatim reply, tag included"
    );
    let typed: super::super::super::streaming::MessagesEvent =
        serde_json::from_str(&raw.to_string()).expect("raw must pass the actual wire parser");
    assert_eq!(typed.fields["id"], "msg_raw_1");
    assert_eq!(typed.fields["model"], "claude-sonnet-4-6");
    assert_eq!(typed.fields["stop_reason"], "stop_sequence");
    assert_eq!(typed.fields["stop_sequence"], "alpha");
    assert_eq!(typed.fields["usage"]["input_tokens"], 7);
    assert_eq!(typed.fields["usage"]["output_tokens"], 2);
    assert_eq!(raw["stop_sequence"], "alpha");

    // The transport id is not part of any reply document; the driver
    // read it off the `request-id` header and stamped the normalized
    // response with it.
    assert!(
        raw.get("provider_request_id").is_none(),
        "the document carries no transport id"
    );
    assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));

    // The normalized response reports the reason and drops which
    // sequence fired — the whole reason `raw` is worth capturing.
    assert_eq!(
        response.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
    assert_eq!(response.model(), Some("claude-sonnet-4-6"));
}
