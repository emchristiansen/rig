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
    // The created payload has no semantic verdict; its usage is still observed.
    assert_eq!(events.len(), 4);
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
    assert_eq!(
        events[1],
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
    assert!(matches!(events[2], AdapterEvent::Provider { .. }));
    assert!(matches!(events[3], AdapterEvent::ErrorEnvelope { .. }));
}

#[test]
fn an_error_payload_emits_only_its_exact_envelope_despite_usage_and_wrapped_metadata() {
    use crate::observe::AdapterErrorEnvelope;
    for (payload, expected) in [
        (
            json!({"type":"error","usage":{"output_tokens":2},"code":"server_error","message":"boom","response":{"status":"failed","usage":{"total_tokens":99}}}),
            AdapterErrorEnvelope {
                code: Some("server_error".to_owned()),
                status: None,
                message: Some("boom".to_owned()),
            },
        ),
        (
            json!({"type":"error","usage":{"output_tokens":2},"code":"ignored","message":"ignored","error":{"code":503,"status":"unavailable","message":"nested"}}),
            AdapterErrorEnvelope {
                code: Some("503".to_owned()),
                status: Some("unavailable".to_owned()),
                message: Some("nested".to_owned()),
            },
        ),
    ] {
        assert_eq!(
            project(&[payload]),
            vec![AdapterEvent::ErrorEnvelope { error: expected }]
        );
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
        (
            json!({"input_tokens":7,"output_tokens":3,"total_tokens":13}),
            Some(13),
        ),
        (json!({"input_tokens":7,"output_tokens":3}), None),
        (
            json!({"input_tokens":7,"output_tokens":3,"total_tokens":"unknown"}),
            None,
        ),
    ] {
        let events = project(&[json!({"status":"completed","usage":usage})]);
        assert_eq!(
            events[0],
            AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: Some(7),
                    output_tokens: Some(3),
                    total_tokens: expected_total,
                    cached_input_tokens: None,
                    reasoning_tokens: None,
                    tool_input_tokens: None,
                }
            }
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1], AdapterEvent::Provider { .. }));
    }
}

#[test]
fn a_wrapped_object_without_usage_does_not_fall_back_to_top_level_usage() {
    let events = project(&[
        json!({"status":"failed","usage":{"input_tokens":11,"total_tokens":99},"error":{"code":"ignored"},"response":{"status":"completed","model":"wrapped"}}),
    ]);
    assert_eq!(
        events,
        vec![AdapterEvent::Provider {
            verdict: crate::wire::AdapterVerdict {
                finish_reason: Some("completed".to_owned()),
                block_reason: None,
                detail: None,
                model: Some("wrapped".to_owned()),
            }
        }]
    );
}

#[test]
fn malformed_typed_payload_fields_abort_all_projection_including_usable_usage() {
    for corruption in [
        json!({"type":7}),
        json!({"id":7}),
        json!({"model":[]}),
        json!({"status":false}),
        json!({"incomplete_details":false}),
        json!({"incomplete_details":{"reason":7}}),
        json!({"usage":false}),
        json!({"usage":{"input_tokens_details":false}}),
        json!({"error":"bad"}),
        json!({"error":{"type":7}}),
        json!({"error":{"message":7}}),
        json!({"message":7}),
        json!({"response":false}),
        json!({"response":{"id":7}}),
    ] {
        let mut payload = json!({"status":"failed","usage":{"input_tokens":7,"total_tokens":9},"error":{"code":"capacity","message":"boom"}});
        payload
            .as_object_mut()
            .expect("payload")
            .extend(corruption.as_object().expect("corruption").clone());
        assert!(project(&[payload.clone()]).is_empty(), "{payload}");
    }
    let malformed_unwrapped =
        json!({"id":7,"response":{"status":"completed","usage":{"input_tokens":7}}});
    assert!(
        project(&[malformed_unwrapped]).is_empty(),
        "the full typed payload is read before choosing its response"
    );
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

/// A raw `max_output_tokens` or `temperature` in `additional_params` is
/// refused on Codex with exactly the error the typed field gets, whatever its
/// value, rather than dropped unseen (mec-rig-bodies-answers-1 Q6). A raw
/// `null` sets nothing: it is neither sent nor refused.
#[test]
fn raw_and_typed_codex_output_caps_and_temperatures_are_refused_alike() {
    use super::super::Delivery;
    use crate::error::ProviderError;
    use crate::wire::Mode;

    let refusal = |request: &crate::completion::CompletionRequest| {
        let error = ProviderError::from(
            codex()
                .responses_request(request, Delivery::Http(Mode::Unary))
                .expect_err("a caller-set control is refused"),
        );
        let typed = error
            .source()
            .expect("typed source")
            .downcast_ref::<super::UnsupportedCodexControl>()
            .copied();
        (typed, error.to_string())
    };
    let mut typed_cap = codex_request(json!({}));
    typed_cap.max_tokens = Some(64);
    let mut typed_temperature = codex_request(json!({}));
    typed_temperature.temperature = Some(0.5);
    for (typed, expected, raws) in [
        (
            typed_cap,
            super::UnsupportedCodexControl::MaxOutputTokens,
            [
                json!({"max_output_tokens":64}),
                json!({"max_output_tokens":"ignored"}),
            ],
        ),
        (
            typed_temperature,
            super::UnsupportedCodexControl::Temperature,
            [
                json!({"temperature":0.5}),
                json!({"temperature":{"ignored":true}}),
            ],
        ),
    ] {
        let typed = refusal(&typed);
        assert_eq!(typed.0, Some(expected));
        for raw in raws {
            assert_eq!(refusal(&codex_request(raw.clone())), typed, "{raw}");
        }
    }

    let request = codex_request(json!({"max_output_tokens":null,"temperature":null,
        "unrecognized":"dropped"}));
    let body = codex()
        .responses_request(&request, Delivery::Http(Mode::Unary))
        .expect("a raw null sets nothing");
    assert!(body.get("max_output_tokens").is_none());
    assert!(body.get("temperature").is_none());
    assert!(body.get("unrecognized").is_none());
    assert_eq!(body.get("store"), Some(&json!(false)));
    assert_eq!(
        body.get("include"),
        Some(&json!(["reasoning.encrypted_content"])),
        "store:false keeps reasoning available for stateless replay"
    );
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

/// mec-rig-port-answers-1 Q7: off Codex, raw keys the fork's reader does not
/// recognize, outer request keys included, are sent as written and outrank
/// the typed provider options, as raw params do on every wire. A raw `model`
/// is never sent (G132), and the typed output cap and temperature win over
/// raw ones, which are sent only where the typed field is unset. Codex keeps
/// the fork's reader and sends none of them.
#[test]
fn raw_projection_sends_unknown_keys_but_not_model_or_raw_cap_and_temperature_over_typed() {
    use crate::providers::openai::extension::OpenAiOptions;
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let raw = json!({
        "input":[{"role":"developer","content":"raw"}],
        "model":"ignored-model", "instructions":"raw", "tool_choice":"required",
        "max_output_tokens":999, "temperature":0.99,
        "conversation":"raw-conversation", "future_parameter":{"keep":false},
        "top_logprobs":2, "user":null
    });
    let mut request = codex_request(raw.clone())
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
    assert_eq!(body.get("max_output_tokens"), Some(&json!(7)));
    assert_eq!(body.get("temperature"), Some(&json!(0.25)));
    for key in [
        "input",
        "instructions",
        "tool_choice",
        "conversation",
        "future_parameter",
        "top_logprobs",
        "user",
    ] {
        assert_eq!(body.get(key), raw.get(key), "raw {key} is sent as written");
    }

    let untyped = wire
        .responses_request(
            &codex_request(raw.clone()),
            super::super::Delivery::Http(crate::wire::Mode::Unary),
        )
        .expect("raw cap and temperature encode where the typed fields are unset");
    assert_eq!(untyped.get("model"), Some(&json!("gpt-5.4")));
    assert_eq!(untyped.get("max_output_tokens"), Some(&json!(999)));
    assert_eq!(untyped.get("temperature"), Some(&json!(0.99)));

    let codex_body = codex()
        .responses_request(
            &codex_request(json!({"future_parameter":{"keep":false},"top_logprobs":2,"user":null})),
            super::super::Delivery::Http(crate::wire::Mode::Unary),
        )
        .expect("Codex keeps the fork's reader");
    for key in ["future_parameter", "top_logprobs", "user"] {
        assert!(codex_body.get(key).is_none(), "Codex sends no raw {key}");
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
    // mec-rig-port-answers-1 Q7: an explicit raw null is sent as null.
    assert_eq!(body.get("top_p"), Some(&Value::Null));
}

/// A null or boolean `additional_params` is empty. Off Codex, an explicit raw
/// null for a key is sent as null (mec-rig-port-answers-1 Q7); nothing else
/// emits these keys.
#[test]
fn raw_null_and_boolean_are_empty_and_explicit_null_fields_are_sent_as_null() {
    for raw in [
        Value::Null,
        json!(false),
        json!(true),
        json!({"stream":"ignored"}),
        json!({"metadata":null,"text":null,"reasoning":null,"include":null,"store":null}),
    ] {
        let body = standard_body(raw.clone());
        for key in [
            "stream",
            "metadata",
            "text",
            "reasoning",
            "include",
            "store",
            "client_metadata",
        ] {
            assert_eq!(
                body.get(key),
                raw.get(key).filter(|value| value.is_null()),
                "raw {raw} emits {key} only as its explicit null"
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

/// Off Codex an unrecognised raw key is sent as written beside the exact
/// recognised value (mec-rig-port-answers-1 Q7).
#[test]
fn recognized_raw_prompt_cache_options_keep_exact_values_and_unknown_keys_are_sent() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    for value in [
        json!({"ttl":"30m","comparison_response_id":"resp_0","future":[null,7]}),
        Value::Null,
    ] {
        let request = codex_request(json!({"prompt_cache_options":value,"unknown_outer":[null,9]}));
        let body = wire
            .responses_request(
                &request,
                super::super::Delivery::Http(crate::wire::Mode::Unary),
            )
            .expect("existing request merge");
        assert_eq!(body.get("prompt_cache_options"), Some(&value));
        assert_eq!(body.get("unknown_outer"), Some(&json!([null, 9])));
    }
}

/// A provider-options entry read back from JSON carries no typed options and
/// is sent as written, so the raw-field reader never decodes the `top_p` or
/// `store` it sets. Final Codex shaping decodes them itself: a malformed one
/// is refused there with the error a malformed raw one gets, and the request
/// is never built.
#[test]
fn malformed_codex_controls_from_untyped_provider_options_are_refused_at_shaping() {
    use super::super::Delivery;
    use crate::completion::ProviderOptions;
    use crate::wire::Mode;

    for (field, value, expected) in [
        (
            "store",
            json!("bad"),
            "RequestError: Invalid OpenAI Responses additional_params payload: invalid type: string \"bad\", expected a boolean",
        ),
        (
            "top_p",
            json!("bad"),
            "RequestError: Invalid OpenAI Responses additional_params payload: invalid type: string \"bad\", expected f64",
        ),
    ] {
        let mut request = codex_request(json!({}));
        request.provider_options = serde_json::from_value::<ProviderOptions>(
            json!({ (crate::providers::chatgpt::PROVIDER_NAME): { "*": { (field): value } } }),
        )
        .expect("an object of object sections");
        let error = codex()
            .responses_request(&request, Delivery::Http(Mode::Unary))
            .expect_err("a malformed Codex control is refused at shaping");
        assert_eq!(error.to_string(), expected, "{field}");
    }
}
