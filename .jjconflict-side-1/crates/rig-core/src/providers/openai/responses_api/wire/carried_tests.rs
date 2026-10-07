use serde_json::{Value, json};
use std::error::Error;

use super::{Responses, project_payload};
use crate::observe::{Action, AdapterContext, AdapterEvent, AdapterUsage, ObservationLog, Subject};
use crate::providers::openai::OpenAIConfig;
use crate::wire::{UsageObservation, Wire};

fn project(payloads: &[Value]) -> Vec<AdapterEvent> {
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "responses-usage");
    let mut attempt = context
        .attempt_for(&http::Request::new(()), "/responses")
        .expect("an attempt starts");
    for payload in payloads {
        attempt.project(|sink| project_payload(payload.to_string().as_bytes(), sink));
    }
    drop(attempt);
    log.trace()
        .observations
        .into_iter()
        .filter_map(|observation| match observation.action {
            Action::Adapter { observation } => match observation.event {
                event @ (AdapterEvent::Usage { .. }
                | AdapterEvent::Provider { .. }
                | AdapterEvent::ErrorEnvelope { .. }) => Some(event),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn responses_selects_payload_usage_snapshots() {
    let wire: Responses = OpenAIConfig::new("test-key").responses("gpt-5.4");
    assert_eq!(
        wire.describe().usage_observation,
        UsageObservation::PayloadSnapshots
    );
}

#[test]
fn each_payload_reports_wrapped_usage_and_reported_total_before_its_verdict_and_error() {
    let events = project(&[
        json!({"type":"response.created","response":{"id":"resp_1","status":"in_progress","usage":{"input_tokens":7}}}),
        json!({"type":"response.failed","usage":{"input_tokens":11,"total_tokens":99},"response":{"id":"resp_1","status":"failed","usage":{"input_tokens":100,"output_tokens":3,"total_tokens":103,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":2}},"error":{"code":"server_error","message":"boom"}}}),
    ]);
    assert_eq!(events.len(), 5);
    assert_eq!(
        events[0],
        AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: Some(7),
                output_tokens: None,
                total_tokens: None,
                cached_input_tokens: None,
                reasoning_tokens: None,
                tool_input_tokens: None,
            }
        }
    );
    assert!(matches!(events[1], AdapterEvent::Provider { .. }));
    assert_eq!(
        events[2],
        AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: Some(100),
                output_tokens: Some(3),
                total_tokens: Some(103),
                cached_input_tokens: Some(0),
                reasoning_tokens: Some(2),
                tool_input_tokens: None,
            }
        }
    );
    assert!(matches!(events[3], AdapterEvent::Provider { .. }));
    assert!(matches!(events[4], AdapterEvent::ErrorEnvelope { .. }));
}

#[test]
fn an_error_payload_emits_only_its_exact_envelope_despite_usage_and_wrapped_metadata() {
    use crate::observe::AdapterErrorEnvelope;
    for (payload, expected) in [
        (
            json!({"type":"error","usage":{"output_tokens":2},"code":"server_error","message":"boom","response":{"status":"failed","usage":{"total_tokens":99}}}),
            AdapterErrorEnvelope { code: Some("server_error".to_owned()), status: None, message: Some("boom".to_owned()) },
        ),
        (
            json!({"type":"error","usage":{"output_tokens":2},"code":"ignored","message":"ignored","error":{"code":503,"status":"unavailable","message":"nested"}}),
            AdapterErrorEnvelope { code: Some("503".to_owned()), status: Some("unavailable".to_owned()), message: Some("nested".to_owned()) },
        ),
    ] {
        assert_eq!(project(&[payload]), vec![AdapterEvent::ErrorEnvelope { error: expected }]);
    }
}

#[test]
fn absent_or_unusable_counts_remain_absent_and_zero_is_a_real_snapshot() {
    let events = project(&[
        json!({"usage":{"input_tokens":0,"output_tokens":"unknown","total_tokens":0,"input_tokens_details":{"cached_tokens":null},"output_tokens_details":{"reasoning_tokens":-1}}}),
    ]);
    assert_eq!(
        events[0],
        AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: Some(0),
                output_tokens: None,
                total_tokens: Some(0),
                cached_input_tokens: None,
                reasoning_tokens: None,
                tool_input_tokens: None,
            }
        }
    );
}

#[test]
fn unary_usage_retains_reported_total_without_deriving_an_absent_total() {
    for (usage, expected_total) in [
        (json!({"input_tokens":7,"output_tokens":3,"total_tokens":13}), Some(13)),
        (json!({"input_tokens":7,"output_tokens":3}), None),
        (json!({"input_tokens":7,"output_tokens":3,"total_tokens":"unknown"}), None),
    ] {
        let events = project(&[json!({"status":"completed","usage":usage})]);
        assert_eq!(events[0], AdapterEvent::Usage { usage: AdapterUsage {
            input_tokens: Some(7), output_tokens: Some(3), total_tokens: expected_total,
            cached_input_tokens: None, reasoning_tokens: None, tool_input_tokens: None,
        }});
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1], AdapterEvent::Provider { .. }));
    }
}

#[test]
fn a_wrapped_object_without_usage_does_not_fall_back_to_top_level_usage() {
    let events = project(&[json!({"status":"failed","usage":{"input_tokens":11,"total_tokens":99},"error":{"code":"ignored"},"response":{"status":"completed","model":"wrapped"}})]);
    assert_eq!(events, vec![AdapterEvent::Provider { verdict: crate::wire::AdapterVerdict {
        finish_reason: Some("completed".to_owned()), block_reason: None,
        detail: None, model: Some("wrapped".to_owned()),
    }}]);
}

#[test]
fn malformed_typed_payload_fields_abort_all_projection_including_usable_usage() {
    for corruption in [
        json!({"type":7}), json!({"id":7}), json!({"model":[]}),
        json!({"status":false}), json!({"incomplete_details":false}),
        json!({"incomplete_details":{"reason":7}}), json!({"usage":false}),
        json!({"usage":{"input_tokens_details":false}}),
        json!({"error":"bad"}), json!({"error":{"type":7}}),
        json!({"error":{"message":7}}), json!({"message":7}),
        json!({"response":false}), json!({"response":{"id":7}}),
    ] {
        let mut payload = json!({"status":"failed","usage":{"input_tokens":7,"total_tokens":9},"error":{"code":"capacity","message":"boom"}});
        payload.as_object_mut().expect("payload").extend(corruption.as_object().expect("corruption").clone());
        assert!(project(&[payload.clone()]).is_empty(), "{payload}");
    }
    let malformed_unwrapped = json!({"id":7,"response":{"status":"completed","usage":{"input_tokens":7}}});
    assert!(project(&[malformed_unwrapped]).is_empty(), "the full typed payload is read before choosing its response");
    let malformed_error_event = json!({"type":"error","message":7,"usage":{"input_tokens":7}});
    assert!(project(&[malformed_error_event]).is_empty());
}

fn codex_request(raw: Value) -> crate::completion::CompletionRequest {
    let mut request =
        crate::completion::CompletionRequest::from(vec![crate::message::Message::user("hello")]);
    request.additional_params = Some(raw);
    request
}

fn codex() -> Responses {
    OpenAIConfig::with_key(&crate::providers::chatgpt::DIALECT, "test-key").responses("gpt-5.4")
}

#[test]
fn responses_raw_outer_controls_are_ignored_but_typed_codex_controls_are_refused() {
    use super::super::Delivery;
    use crate::error::ProviderError;
    use crate::wire::Mode;

    let request =
        codex_request(json!({"max_output_tokens":"ignored","temperature":{"ignored":true}}));
    let body = codex()
        .responses_request(&request, Delivery::Http(Mode::Unary))
        .expect("raw outer controls are not fork additional parameters");
    assert!(body.get("max_output_tokens").is_none());
    assert!(body.get("temperature").is_none());
    assert_eq!(body.get("store"), Some(&json!(false)));
    assert!(
        body.get("include").is_none(),
        "store:false alone does not ask for ciphertext"
    );
    for (max_tokens, temperature, expected) in [
        (
            Some(1),
            None,
            super::UnsupportedCodexControl::MaxOutputTokens,
        ),
        (None, Some(0.5), super::UnsupportedCodexControl::Temperature),
    ] {
        let mut request = request.clone();
        request.max_tokens = max_tokens;
        request.temperature = temperature;
        let error = ProviderError::from(
            codex()
                .responses_request(&request, Delivery::Http(Mode::Unary))
                .expect_err("typed outer choice refused"),
        );
        assert_eq!(
            error
                .source()
                .expect("typed source")
                .downcast_ref::<super::UnsupportedCodexControl>(),
            Some(&expected)
        );
    }
}

#[test]
fn recognized_raw_codex_controls_keep_decode_errors_and_valid_typed_refusals() {
    use super::super::Delivery;
    use crate::error::ProviderError;
    use crate::wire::Mode;

    for (field, value, expected) in [
        (
            "store",
            json!("bad"),
            "Invalid OpenAI Responses additional_params payload: invalid type: string \"bad\", expected a boolean",
        ),
        (
            "top_p",
            json!("bad"),
            "Invalid OpenAI Responses additional_params payload: invalid type: string \"bad\", expected f64",
        ),
    ] {
        let request = codex_request(json!({(field):value}));
        let error = codex()
            .responses_request(&request, Delivery::Http(Mode::Unary))
            .expect_err("recognized malformed field refused before shaping");
        assert!(error.to_string().contains(expected), "{error}");
    }
    for (raw, expected) in [
        (json!({"store":true}), super::UnsupportedCodexControl::Store),
        (json!({"top_p":0.5}), super::UnsupportedCodexControl::TopP),
    ] {
        let error = ProviderError::from(
            codex()
                .responses_request(&codex_request(raw), Delivery::Http(Mode::Unary))
                .expect_err("valid unsupported choice refused"),
        );
        assert_eq!(
            error
                .source()
                .expect("typed source")
                .downcast_ref::<super::UnsupportedCodexControl>(),
            Some(&expected)
        );
    }
    let body = codex().responses_request(&codex_request(json!({"store":false,"reasoning":{"effort":"low"},"include":["reasoning.encrypted_content"]})), Delivery::Http(Mode::Unary)).expect("supported controls encode");
    assert_eq!(
        body.get("include"),
        Some(&json!(["reasoning.encrypted_content"]))
    );
}

fn standard_body(raw: Value) -> crate::completion::options::FinalBody {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    wire.responses_request(
        &codex_request(raw),
        super::super::Delivery::Http(crate::wire::Mode::Unary),
    )
    .expect("recognized raw fields encode")
}

#[test]
fn raw_projection_ignores_outer_and_unknown_keys_without_changing_typed_options() {
    use crate::providers::openai::extension::OpenAiOptions;
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let mut request = codex_request(json!({
        "input":[{"role":"developer","content":"ignored"}],
        "model":"ignored-model", "instructions":"ignored", "tool_choice":"required",
        "max_output_tokens":999, "temperature":0.99,
        "conversation":"ignored-raw-conversation", "future_parameter":{"keep":false}
    }))
    .provider_option(OpenAiOptions::default().conversation("typed-conversation"));
    request.max_tokens = Some(7);
    request.temperature = Some(0.25);
    let body = wire
        .responses_request(
            &request,
            super::super::Delivery::Http(crate::wire::Mode::Unary),
        )
        .expect("typed options survive raw projection");
    assert_eq!(body.get("model"), Some(&json!("gpt-5.4")));
    assert_eq!(
        body.get("input"),
        Some(
            &json!([{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}])
        )
    );
    assert_eq!(body.get("max_output_tokens"), Some(&json!(7)));
    assert_eq!(body.get("temperature"), Some(&json!(0.25)));
    assert_eq!(body.get("conversation"), Some(&json!("typed-conversation")));
    for key in ["instructions", "tool_choice", "future_parameter"] {
        assert!(body.get(key).is_none(), "unrecognized raw {key} is absent");
    }
}

#[test]
fn recognized_raw_projection_keeps_fork_nested_defaults_and_string_service_tiers() {
    let body = standard_body(json!({
        "background":false,
        "text":{"format":{"type":"json_schema","schema":{"type":"object"},"future":"ignored"},"future":"ignored"},
        "include":["file_search_call.results","computer_call.output.image_url"],
        "top_p":null, "truncation":"disabled", "user":"",
        "prompt_cache_key":"cache", "prompt_cache_retention":"future-retention",
        "metadata":{"arbitrary":{"nested":[null,1]}}, "parallel_tool_calls":false,
        "previous_response_id":"previous",
        "reasoning":{"effort":"max","summary":"detailed","mode":"pro","context":"all_turns","future":"ignored"},
        "service_tier":"future-tier", "store":false,
        "client_metadata":{"session_id":"caller","future":"value"},
        "tools":[{"type":"web_search","future":{"nested":[null,1]}}]
    }));
    assert_eq!(body.get("background"), Some(&json!(false)));
    assert_eq!(
        body.get("text"),
        Some(
            &json!({"format":{"type":"json_schema","name":"","schema":{"type":"object"},"strict":false}})
        )
    );
    assert_eq!(
        body.get("reasoning"),
        Some(&json!({"effort":"max","summary":"detailed","mode":"pro","context":"all_turns"}))
    );
    assert_eq!(
        body.get("include"),
        Some(&json!([
            "file_search_call.results",
            "computer_call.output.image_url",
            "reasoning.encrypted_content"
        ]))
    );
    assert_eq!(
        body.get("metadata"),
        Some(&json!({"arbitrary":{"nested":[null,1]}}))
    );
    assert_eq!(body.get("service_tier"), Some(&json!("future-tier")));
    assert_eq!(
        body.get("client_metadata"),
        Some(&json!({"session_id":"caller","future":"value"}))
    );
    assert_eq!(
        body.get("tools"),
        Some(&json!([{"type":"web_search","future":{"nested":[null,1]}}]))
    );
    assert!(body.get("top_p").is_none());
}

#[test]
fn raw_null_and_boolean_are_empty_and_nullable_fields_are_omitted() {
    for raw in [
        Value::Null,
        json!(false),
        json!(true),
        json!({"stream":"ignored"}),
        json!({"metadata":null,"text":null,"reasoning":null,"include":null,"store":null}),
    ] {
        let body = standard_body(raw);
        for key in [
            "stream",
            "metadata",
            "text",
            "reasoning",
            "include",
            "store",
            "client_metadata",
        ] {
            assert!(
                body.get(key).is_none(),
                "empty/null raw does not emit {key}"
            );
        }
    }
}

#[test]
fn raw_projection_refuses_malformed_roots_nested_known_fields_and_tools() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    for (raw, expected) in [
        (
            json!([]),
            "Invalid OpenAI Responses additional_params payload: invalid type: sequence, expected struct AdditionalParameters",
        ),
        (
            json!("bad"),
            "Invalid OpenAI Responses additional_params payload: invalid type: string \"bad\", expected struct AdditionalParameters",
        ),
        (
            json!({"text":{}}),
            "Invalid OpenAI Responses additional_params payload: missing field `format`",
        ),
        (
            json!({"text":{"format":{"type":"json_object"}}}),
            "Invalid OpenAI Responses additional_params payload: unknown variant `json_object`, expected `json_schema` or `text`",
        ),
        (
            json!({"reasoning":{"effort":"future"}}),
            "Invalid OpenAI Responses additional_params payload: unknown variant `future`, expected one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`",
        ),
        (
            json!({"reasoning":{"mode":"standard"}}),
            "Invalid OpenAI Responses additional_params payload: unknown variant `standard`, expected `pro`",
        ),
        (
            json!({"include":["message.output_text.logprobs"]}),
            "Invalid OpenAI Responses additional_params payload: unknown variant `message.output_text.logprobs`",
        ),
        (
            json!({"client_metadata":{"session_id":7}}),
            "Invalid OpenAI Responses additional_params payload: invalid type: integer `7`, expected a string",
        ),
        (
            json!({"metadata":[]}),
            "Invalid OpenAI Responses additional_params payload: invalid type: sequence, expected a map",
        ),
        (
            json!({"tools":null}),
            "Invalid OpenAI Responses tools payload in additional_params: `tools` must be a JSON array of tool declarations",
        ),
    ] {
        let error = wire
            .responses_request(
                &codex_request(raw),
                super::super::Delivery::Http(crate::wire::Mode::Unary),
            )
            .expect_err("malformed recognized raw input refused");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
