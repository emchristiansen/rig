//! Anthropic raw-body seam regression tests.
//!
//! Locks down `completion_with_raw_body`: the exact success-response body
//! bytes are returned verbatim alongside the typed response (including fields
//! the typed representation does not model), while provider error envelopes,
//! non-success HTTP statuses, and malformed bodies return the same errors as
//! `completion` with no raw bytes escaping. Also locks the trait `completion`
//! delegation to identical observable success behavior.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
//! These fixtures are hand-authored (deterministic probe bodies), so record
//! mode is not applicable to them.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;

/// Must match the `then.body` of `raw_body/success_returns_exact_bytes.yaml`
/// and `raw_body/completion_delegates_identically_on_success.yaml` exactly,
/// byte for byte: the replay server serves the fixture body verbatim, so this
/// constant is the ground truth the seam must reproduce.
const EXPECTED_SUCCESS_BODY: &str = r#"{"content":[{"text":"seam","type":"text"}],"id":"msg_REDACTED_1","model":"claude-sonnet-4-6","role":"assistant","stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":21,"output_tokens":5},"x_rig_raw_body_probe":"exact-bytes-fidelity"}"#;

fn text_of(
    response: &rig::completion::CompletionResponse<anthropic::completion::CompletionResponse>,
) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn success_returns_exact_bytes() {
    with_anthropic_cassette(
        "raw_body/success_returns_exact_bytes",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Reply with the single word: seam.")
                .preamble("You are a terse test responder.".to_string())
                .max_tokens(64)
                .build();

            let (response, raw_body) = model
                .completion_with_raw_body(request)
                .await
                .expect("success fixture should convert");

            assert_eq!(
                raw_body.as_ref(),
                EXPECTED_SUCCESS_BODY.as_bytes(),
                "raw body must be the exact success-response bytes, verbatim"
            );

            let raw_json: serde_json::Value =
                serde_json::from_slice(&raw_body).expect("raw body should be the fixture's JSON");
            assert_eq!(
                raw_json["x_rig_raw_body_probe"],
                serde_json::Value::from("exact-bytes-fidelity"),
                "raw bytes must preserve response fields the typed representation does not model"
            );

            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("end_turn")
            );
            assert_eq!(response.raw_response.id, "msg_REDACTED_1");
            assert_eq!(text_of(&response), "seam");
        },
    )
    .await;
}

#[tokio::test]
async fn completion_delegates_identically_on_success() {
    with_anthropic_cassette(
        "raw_body/completion_delegates_identically_on_success",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Reply with the single word: seam.")
                .preamble("You are a terse test responder.".to_string())
                .max_tokens(64)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("delegating completion must succeed on the same success fixture");

            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("end_turn")
            );
            assert_eq!(response.raw_response.id, "msg_REDACTED_1");
            assert_eq!(response.raw_response.usage.input_tokens, 21);
            assert_eq!(response.raw_response.usage.output_tokens, 5);
            assert_eq!(text_of(&response), "seam");
        },
    )
    .await;
}

#[tokio::test]
async fn provider_error_envelope_is_err_without_raw_bytes() {
    with_anthropic_cassette(
        "raw_body/provider_error_envelope_is_err_without_raw_bytes",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Trigger a provider error envelope.")
                .preamble("You are a terse test responder.".to_string())
                .max_tokens(64)
                .build();

            let error = model
                .completion_with_raw_body(request)
                .await
                .expect_err("a 200 error envelope must not yield raw bytes");

            assert!(
                error
                    .to_string()
                    .contains("raw-body seam probe: provider error envelope"),
                "provider error body should be preserved in the error, got: {error}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn http_error_status_is_err_without_raw_bytes() {
    with_anthropic_cassette(
        "raw_body/http_error_status_is_err_without_raw_bytes",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Trigger a rate-limit status.")
                .preamble("You are a terse test responder.".to_string())
                .max_tokens(64)
                .build();

            let error = model
                .completion_with_raw_body(request)
                .await
                .expect_err("a non-success status must not yield raw bytes");

            let rendered = error.to_string();
            assert!(
                rendered.contains("429") || rendered.contains("rate limited"),
                "status-derived error should surface the failing status/body, got: {rendered}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn malformed_success_body_is_err_without_raw_bytes() {
    with_anthropic_cassette(
        "raw_body/malformed_success_body_is_err_without_raw_bytes",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Trigger a malformed success body.")
                .preamble("You are a terse test responder.".to_string())
                .max_tokens(64)
                .build();

            let error = model
                .completion_with_raw_body(request)
                .await
                .expect_err("an unparseable 200 body must error, not yield raw bytes");

            // The exact serde rendering is not part of the contract; erroring
            // (rather than panicking or returning bytes) is.
            let _rendered = error.to_string();
        },
    )
    .await;
}
