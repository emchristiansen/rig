use super::*;
use serde_json::json;

fn response_with_service_tier(service_tier: &str) -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [],
        "service_tier": service_tier,
    })
}

#[test]
fn completion_response_deserializes_standard_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("standard"))
            .expect("response should deserialize");

    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Standard)
    ));
}

#[test]
fn completion_response_deserializes_priority_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("priority"))
            .expect("response should deserialize");

    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Priority)
    ));
}

#[test]
fn completion_response_preserves_unknown_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("provider_experimental"))
            .expect("response should deserialize");

    let Some(OpenAIServiceTier::Other(service_tier)) = response.additional_parameters.service_tier
    else {
        panic!("expected provider-specific service tier");
    };

    assert_eq!(service_tier, "provider_experimental");
}

/// A response whose echoed `top_p` is object-shaped, as MiniMax-style
/// Responses endpoints emit it, carrying a valid tool call (rig#2483).
fn response_with_object_top_p() -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "MiniMax-M2",
        "top_p": { "value": 0.95 },
        "service_tier": "default",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"location\":\"Paris\"}",
            "status": "completed"
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15
        }
    })
}

/// One optional metadata field disagreeing with its Rust type must never
/// discard the response: the tool call and usage survive, `top_p` reads
/// back as `None`, and sibling metadata still decodes (rig#2483).
#[test]
fn completion_response_tolerates_object_shaped_top_p() {
    let response: CompletionResponse = serde_json::from_value(response_with_object_top_p())
        .expect("an object-shaped top_p must not fail the response");

    assert_eq!(response.additional_parameters.top_p, None);
    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Default)
    ));
    assert!(
        matches!(response.output.first(), Some(Output::FunctionCall(call)) if call.name == "get_weather"),
        "the tool call must survive metadata decode failures: {:?}",
        response.output
    );
    assert_eq!(response.usage.map(|usage| usage.total_tokens), Some(15));
}

/// Numeric `top_p` still decodes : including under
/// `serde_json/arbitrary_precision`, where a `#[serde(flatten)]` into `f64`
/// used to reject it because buffered numbers arrive as an internal map
/// (rig#2493). Run this module with and without that feature.
#[test]
fn completion_response_decodes_numeric_top_p() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(0.95);
    let response: CompletionResponse =
        serde_json::from_str(&body.to_string()).expect("numeric top_p must decode");

    assert_eq!(response.additional_parameters.top_p, Some(0.95));
    assert!(matches!(
        response.output.first(),
        Some(Output::FunctionCall(_))
    ));
}

/// The metadata projection is per key: a bad `top_p` does not take
/// `service_tier` or `store` down with it, and vice versa.
#[test]
fn completion_response_drops_only_the_mistyped_metadata_key() {
    let mut body = response_with_object_top_p();
    body["store"] = json!("not-a-bool");
    body["top_p"] = json!(0.5);
    body["prompt_cache_key"] = json!("cache-1");
    let response: CompletionResponse =
        serde_json::from_value(body).expect("a mistyped store must not fail the response");

    assert_eq!(response.additional_parameters.store, None);
    assert_eq!(response.additional_parameters.top_p, Some(0.5));
    assert_eq!(
        response.additional_parameters.prompt_cache_key.as_deref(),
        Some("cache-1")
    );
}

/// The serializer still emits the echoed metadata it decoded, so a well-formed
/// OpenAI body round-trips on every field the wire serializer writes.
#[test]
fn completion_response_round_trips_echoed_metadata() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(1.0);
    body["store"] = json!(true);
    body["metadata"] = json!({ "k": "v" });
    let response: CompletionResponse =
        serde_json::from_value(body.clone()).expect("response should deserialize");
    let serialized = serde_json::to_value(&response).expect("response should serialize");

    for key in [
        "top_p",
        "store",
        "metadata",
        "service_tier",
        "usage",
        "output",
        "model",
    ] {
        assert_eq!(serialized[key], body[key], "field {key} must round-trip");
    }
}

#[test]
fn service_tier_serializes_expected_strings() {
    let cases = [
        (OpenAIServiceTier::Auto, "auto"),
        (OpenAIServiceTier::Default, "default"),
        (OpenAIServiceTier::Flex, "flex"),
        (OpenAIServiceTier::Priority, "priority"),
        (OpenAIServiceTier::Standard, "standard"),
    ];

    for (service_tier, expected) in cases {
        assert_eq!(
            serde_json::to_value(service_tier).expect("service tier should serialize"),
            json!(expected)
        );
    }

    assert_eq!(
        serde_json::to_value(OpenAIServiceTier::Other(
            "provider_experimental".to_string()
        ))
        .expect("provider-specific service tier should serialize"),
        json!("provider_experimental")
    );
}

#[test]
fn responses_usage_token_usage_preserves_reasoning_tokens() {
    let usage = ResponsesUsage {
        input_tokens: 100,
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: Some(25),
        }),
        output_tokens: 50,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 15,
        }),
        total_tokens: 150,
    };

    let token_usage = rig_core::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, Some(15));
    assert_eq!(token_usage.total_tokens, Some(150));
}

#[test]
fn responses_usage_deserializes_without_output_token_details() {
    let usage: ResponsesUsage = serde_json::from_value(json!({
        "input_tokens": 100,
        "input_tokens_details": {
            "cached_tokens": 25
        },
        "output_tokens": 50,
        "total_tokens": 150
    }))
    .expect("usage should deserialize when output token details are omitted");

    assert!(usage.output_tokens_details.is_none());

    let token_usage = rig_core::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, None);
    assert_eq!(token_usage.total_tokens, Some(150));
}

#[test]
fn completion_response_accepts_top_level_reasoning_string() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking through the answer",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("mistral.rs-style reasoning string should deserialize");

    assert_eq!(
        response.provider_reasoning.as_deref(),
        Some("thinking through the answer")
    );
    assert_eq!(response.reasoning_metadata, None);
    assert_eq!(response.reasoning_context, None);
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!("thinking through the answer")
    );
}

#[test]
fn completion_response_accepts_null_metadata() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "openai-compatible-model",
        "metadata": null,
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("response with null metadata should deserialize");

    assert!(response.additional_parameters.metadata.is_empty());
}

#[test]
fn completion_response_accepts_reasoning_only_response() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking only",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [],
        "tools": []
    }))
    .expect("reasoning-only response should deserialize");

    assert_eq!(
        response.provider_reasoning.as_deref(),
        Some("thinking only")
    );
    assert!(response.output.is_empty());
}

#[test]
fn completion_response_preserves_context_without_treating_config_as_text() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": {
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("object-shaped reasoning should be tolerated");

    assert!(response.provider_reasoning.is_none());
    assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(
        response.reasoning_metadata.as_ref(),
        json!({
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        })
        .as_object()
    );
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        })
    );
}

#[test]
fn completion_response_preserves_unknown_reasoning_metadata_and_nulls() {
    let metadata = json!({
        "context": "future_context",
        "effort": "ultra",
        "summary": null,
        "future_control": { "depth": 3 }
    });
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-future",
        "reasoning": metadata,
        "output": [],
        "tools": []
    }))
    .expect("unknown reasoning metadata should deserialize");

    assert_eq!(
        response.reasoning_context.as_deref(),
        Some("future_context")
    );
    assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        metadata
    );
}

#[test]
fn completion_response_ignores_unsupported_reasoning_shapes() {
    for reasoning in [Value::Null, json!(["unexpected"]), json!(42), json!(true)] {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "openai-compatible-model",
            "reasoning": reasoning,
            "output": [],
            "tools": []
        }))
        .expect("unsupported reasoning shapes should remain non-fatal");

        assert_eq!(response.provider_reasoning, None);
        assert_eq!(response.reasoning_metadata, None);
        assert_eq!(response.reasoning_context, None);
        let serialized = serde_json::to_value(&response).expect("response should serialize");
        assert!(
            !serialized
                .as_object()
                .expect("response should serialize as an object")
                .contains_key("reasoning")
        );
    }
}

#[test]
fn completion_response_reasoning_serialization_precedence_is_stable() {
    let mut response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.6",
        "reasoning": { "context": "all_turns", "effort": "max" },
        "output": [],
        "tools": []
    }))
    .expect("reasoning metadata should deserialize");

    response.reasoning_context = Some("current_turn".to_owned());
    response.additional_parameters.reasoning =
        Some(Reasoning::new().with_effort(ReasoningEffort::Low));
    let serialized = serde_json::to_string(&response).expect("response should serialize");
    assert_eq!(serialized.matches("\"reasoning\":").count(), 1);
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({ "context": "all_turns", "effort": "max" })
    );

    let metadata = response.reasoning_metadata.take();
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({ "context": "current_turn" })
    );

    response.reasoning_metadata = metadata;
    response.provider_reasoning = Some("compatible-provider text".to_owned());
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!("compatible-provider text")
    );
}

#[test]
fn reasoning_omits_unset_optional_fields() {
    let reasoning = serde_json::to_value(Reasoning::new().with_mode(ReasoningMode::Pro))
        .expect("reasoning should serialize");

    assert_eq!(reasoning, json!({ "mode": "pro" }));

    let reasoning = serde_json::to_value(
        Reasoning::new()
            .with_effort(ReasoningEffort::Max)
            .with_mode(ReasoningMode::Pro)
            .with_context(ReasoningContext::CurrentTurn)
            .with_summary_level(ReasoningSummaryLevel::Detailed),
    )
    .expect("reasoning should serialize");

    assert_eq!(
        reasoning,
        json!({
            "effort": "max",
            "mode": "pro",
            "context": "current_turn",
            "summary": "detailed"
        })
    );
}

/// Summing usage where one side reports no input-token breakdown leaves the
/// cached count unknown, whichever side it is and whether the breakdown is
/// missing or empty: absence is not zero, so the other side's count alone is
/// never reported as the total. Output breakdowns, whose counters are
/// required, keep the side that has one.
#[test]
fn a_usage_sum_with_an_unreported_cached_count_leaves_it_unknown() {
    let usage =
        |input_tokens_details: Option<InputTokensDetails>,
         output_tokens_details: Option<OutputTokensDetails>| ResponsesUsage {
            input_tokens: 10,
            input_tokens_details,
            output_tokens: 20,
            output_tokens_details,
            total_tokens: 30,
        };
    let reported = || {
        usage(
            Some(InputTokensDetails {
                cached_tokens: Some(2),
            }),
            Some(OutputTokensDetails {
                reasoning_tokens: 4,
            }),
        )
    };
    let missing = || usage(None, None);
    let empty = || {
        usage(
            Some(InputTokensDetails {
                cached_tokens: None,
            }),
            None,
        )
    };

    for (label, sum) in [
        ("missing + reported", missing() + reported()),
        ("reported + missing", reported() + missing()),
        ("empty + reported", empty() + reported()),
        ("reported + empty", reported() + empty()),
    ] {
        let token_usage = rig_core::completion::Usage::from(&sum);
        assert_eq!(token_usage.cached_input_tokens, None, "{label}");
        assert_eq!(token_usage.input_tokens, Some(20), "{label}");
        assert_eq!(token_usage.output_tokens, Some(40), "{label}");
        assert_eq!(token_usage.reasoning_tokens, Some(4), "{label}");
        assert_eq!(token_usage.total_tokens, Some(60), "{label}");
    }

    // Both sides reported: the counts add.
    let both = rig_core::completion::Usage::from(&(reported() + reported()));
    assert_eq!(both.cached_input_tokens, Some(4));
    // Neither side reported: still no breakdown at all.
    assert!((missing() + missing()).input_tokens_details.is_none());
}

#[test]
fn output_unknown_preserves_hosted_tool_payload() {
    let item = json!({
        "type": "web_search_call",
        "id": "ws_001",
        "status": "completed",
        "action": { "type": "search", "queries": ["rig framework"] },
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("unknown output should deserialize");

    let Output::Unknown(value) = output else {
        panic!("expected Output::Unknown for an unmodeled item type");
    };
    assert_eq!(value, item);
}

#[test]
fn output_unknown_round_trips_value_equal() {
    let item = json!({
        "type": "file_search_call",
        "id": "fs_007",
        "status": "in_progress",
        "queries": ["lifecycle"],
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("unknown output should deserialize");
    let serialized = serde_json::to_value(&output).expect("unknown output should serialize");

    assert_eq!(serialized, item);
}

#[test]
fn output_known_variant_with_bad_body_errors() {
    // A recognized `type` tag with a malformed body must still error rather
    // than silently degrading to `Output::Unknown`.
    let malformed = json!({
        "type": "function_call",
        "id": "call_1",
        // missing `arguments`, `call_id`, `name`
    });

    let result: Result<Output, _> = serde_json::from_value(malformed);
    assert!(result.is_err());
}

#[test]
fn completion_response_with_unknown_output_keeps_usage() {
    // Guards the original reason the catch-all exists: an unknown item must
    // not break decoding of the whole response or drop token usage.
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_001",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 25 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 15 },
            "total_tokens": 150,
        },
    });

    let response: CompletionResponse =
        serde_json::from_value(response).expect("response should deserialize");

    assert!(matches!(response.output.first(), Some(Output::Unknown(_))));
    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn output_known_variant_round_trips_value_equal() {
    // The hand-written Serialize must reproduce the modeled wire shape, so a
    // decoded known item re-serializes value-equal to what it came from
    // (guards the `function_call` arm, including its stringified `arguments`).
    // The item ID uses the provider-native `fc_` prefix; other IDs are
    // intentionally dropped on serialization (see `OutputFunctionCall::id`).
    let item = json!({
        "type": "function_call",
        "id": "fc_1",
        "arguments": "{}",
        "call_id": "c1",
        "name": "search",
        "status": "completed",
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("known output should deserialize");
    assert!(matches!(output, Output::FunctionCall(_)));

    let serialized = serde_json::to_value(&output).expect("known output should serialize");
    assert_eq!(serialized, item);
}

#[test]
fn output_reasoning_round_trips_value_equal() {
    // Highest-value parity guard: the `Reasoning` struct variant threads its
    // fields by hand in *both* directions. Populated `encrypted_content` /
    // `status` (the `#[serde(default)]` optionals) must survive
    // serialize -> deserialize unchanged : catching a dropped field or a
    // forgotten `reasoning` dispatch arm (which would degrade to `Unknown`).
    let original = Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![ReasoningSummary::SummaryText {
            text: "weighing options".to_string(),
        }],
        content: vec!["private reasoning".into()],
        encrypted_content: Some("ENCRYPTED".to_string()),
        signature: None,
        status: Some(ToolStatus::Completed),
    };

    let value = serde_json::to_value(&original).expect("reasoning should serialize");
    let round_tripped: Output =
        serde_json::from_value(value).expect("reasoning should deserialize");

    assert_eq!(round_tripped, original);
}

#[test]
fn output_reasoning_none_optionals_serialize_as_explicit_null() {
    // Wire-anchored complement to the round-trip test: with `None`
    // optionals, the keys must still be emitted as explicit `null` (the
    // derived behavior this hand-written serde replaced has no
    // `skip_serializing_if`). Guards against a future refactor silently
    // dropping the keys and changing the wire shape.
    let value = serde_json::to_value(Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![],
        content: vec![],
        encrypted_content: None,
        signature: None,
        status: None,
    })
    .expect("reasoning should serialize");

    assert_eq!(value["type"], "reasoning");
    assert_eq!(value["encrypted_content"], Value::Null);
    assert_eq!(value["status"], Value::Null);
    assert!(value.get("encrypted_content").is_some());
    assert!(value.get("status").is_some());
}

#[test]
fn output_message_round_trips_value_equal() {
    // Wire-anchored serialize check for the `message` arm (only
    // `function_call` was anchored): a decoded message item re-serializes
    // value-equal to the input, tag included.
    let item = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [ { "type": "output_text", "text": "hello", "annotations": [] } ],
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("message item should deserialize");
    assert!(matches!(output, Output::Message(_)));

    let serialized = serde_json::to_value(&output).expect("message should serialize");
    assert_eq!(serialized, item);
}

#[test]
fn each_known_tag_decodes_to_its_modeled_variant() {
    // Guards every modeled dispatch arm: a well-formed item for each known
    // `type` must decode to its specific variant, never to `Unknown`. Adding
    // an `Output` variant without a matching deserialize arm fails here
    // instead of silently routing real items to `Unknown`.
    let message: Output = serde_json::from_value(json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
    }))
    .expect("message item should decode");
    assert!(matches!(message, Output::Message(_)));

    let function_call: Output = serde_json::from_value(json!({
        "type": "function_call", "id": "call_1", "arguments": "{}",
        "call_id": "c1", "name": "f", "status": "completed",
    }))
    .expect("function_call item should decode");
    assert!(matches!(function_call, Output::FunctionCall(_)));

    let reasoning: Output =
        serde_json::from_value(json!({ "type": "reasoning", "id": "r1", "summary": [] }))
            .expect("reasoning item should decode");
    assert!(matches!(reasoning, Output::Reasoning { .. }));
}

#[test]
fn output_without_usable_type_tag_decodes_to_unknown() {
    // An absent or non-string `type` is itself unmodeled, so it is captured
    // verbatim as `Unknown` rather than erroring.
    for item in [
        json!({ "id": "x", "note": "no type field" }),
        json!({ "type": 7, "id": "x" }),
    ] {
        let output: Output =
            serde_json::from_value(item.clone()).expect("should decode to Unknown");
        assert_eq!(output, Output::Unknown(item));
    }
}

/// A `custom_tool_call` output item decodes into its typed variant, never
/// `Unknown` and never a function call, and round-trips its wire value.
#[test]
fn a_custom_tool_call_output_item_decodes_and_round_trips() {
    let wire_item = json!({
        "type": "custom_tool_call",
        "id": "ctc_9",
        "call_id": "call_9",
        "name": "grammar_tool",
        "input": "SELECT 1",
        "status": "completed",
    });
    let output: Output = serde_json::from_value(wire_item.clone()).expect("decodes");
    assert!(matches!(output, Output::CustomToolCall(_)), "{output:?}");
    assert_eq!(
        serde_json::to_value(&output).expect("serializes"),
        wire_item
    );
}

/// A function call output item keeps its namespace.
#[test]
fn a_namespaced_function_call_output_item_keeps_its_namespace() {
    let output: Output = serde_json::from_value(json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "search",
        "namespace": "web",
        "arguments": "{\"q\":\"rig\"}",
        "status": "completed",
    }))
    .expect("decodes");
    let Output::FunctionCall(call) = &output else {
        panic!("expected one function call, got {output:?}");
    };
    assert_eq!(call.namespace.as_deref(), Some("web"));
}

fn arguments(raw: &str) -> FunctionCallArguments {
    serde_json::from_value(json!(raw)).expect("a string always decodes")
}

/// The one classification: empty and whitespace strings are a
/// parameterless invocation, parseable strings carry their value, and a
/// malformed string keeps its bytes as unparseable.
#[test]
fn argument_classification_keeps_empty_and_malformed_meaning() {
    assert_eq!(
        arguments("").classify(),
        ClassifiedArguments::Parsed(json!({}))
    );
    assert_eq!(
        arguments("  ").classify(),
        ClassifiedArguments::Parsed(json!({}))
    );
    assert_eq!(
        arguments("{\"x\":1}").classify(),
        ClassifiedArguments::Parsed(json!({"x": 1}))
    );
    assert_eq!(
        arguments("{\"x\":").classify(),
        ClassifiedArguments::Unparseable("{\"x\":".to_string())
    );
    // A non-string payload is a schema defect of the known shape.
    assert!(serde_json::from_value::<FunctionCallArguments>(json!({"x": 1})).is_err());
}

/// Two observations of one call agree only when both parse to the same
/// value; an unparseable observation agrees with nothing and names its side.
#[test]
fn reconcile_settles_two_observations_of_one_call() {
    let left = arguments("{\"x\": 1}");
    let agreeing = arguments("{\"x\":1}");
    let disagreeing = arguments("{\"x\":2}");
    let malformed = arguments("{\"x\":");

    assert_eq!(left.reconcile(&agreeing), Ok(json!({"x": 1})));
    assert_eq!(arguments("").reconcile(&arguments("{}")), Ok(json!({})));
    assert_eq!(
        left.reconcile(&disagreeing),
        Err(ReconcileArgumentsError::Mismatch)
    );
    assert_eq!(
        malformed.reconcile(&agreeing),
        Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Left))
    );
    assert_eq!(
        left.reconcile(&malformed),
        Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Right))
    );
    assert_eq!(
        malformed.reconcile(&arguments("{\"x\":")),
        Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Both))
    );
}

/// A failed response whose error omits `code` still decodes: the code is
/// absent rather than the whole body being rejected, and the message
/// survives. Re-serializing keeps it absent instead of inventing an empty one.
#[test]
fn a_response_error_without_a_code_decodes_and_stays_absent() {
    let error: ResponseError = serde_json::from_value(json!({"message": "overloaded"}))
        .expect("an error without a code decodes");
    assert_eq!(error.code, None);
    assert_eq!(error.message, "overloaded");
    assert_eq!(
        serde_json::to_value(&error).expect("the error serializes"),
        json!({"message": "overloaded"}),
    );
}

/// An omitted cached-token count stays absent through decoding and the usage
/// projection; it is never reported as zero, while a reported zero stays zero.
#[test]
fn an_omitted_cached_token_count_is_absent_not_zero() {
    let omitted: ResponsesUsage = serde_json::from_value(json!({
        "input_tokens": 3,
        "input_tokens_details": {},
        "output_tokens": 2,
        "total_tokens": 5,
    }))
    .expect("usage without cached_tokens decodes");
    assert_eq!(
        omitted
            .input_tokens_details
            .map(|details| details.cached_tokens),
        Some(None)
    );
    assert_eq!(completion::Usage::from(&omitted).cached_input_tokens, None);
    assert_eq!(
        serde_json::to_value(omitted.input_tokens_details).expect("details serialize"),
        json!({}),
        "an absent count is omitted, not written as null"
    );

    let zero: ResponsesUsage = serde_json::from_value(json!({
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "total_tokens": 5,
    }))
    .expect("usage with a zero cached_tokens decodes");
    assert_eq!(completion::Usage::from(&zero).cached_input_tokens, Some(0));
}

/// Summing usage details keeps absence: a total is known only when every part
/// is, so one omitted count makes the sum absent rather than undercounting.
#[test]
fn summed_cached_token_counts_are_known_only_when_every_part_is() {
    let known = InputTokensDetails {
        cached_tokens: Some(4),
    };
    let absent = InputTokensDetails {
        cached_tokens: None,
    };
    assert_eq!((known + known).cached_tokens, Some(8));
    assert_eq!((known + absent).cached_tokens, None);
    assert_eq!((absent + known).cached_tokens, None);
}

#[test]
fn an_unknown_reasoning_summary_supplies_no_text() {
    let known: ReasoningSummary =
        serde_json::from_value(serde_json::json!({"type":"summary_text","text":"known"})).unwrap();
    let unknown: ReasoningSummary =
        serde_json::from_value(serde_json::json!({"type":"future_summary","list":[1]})).unwrap();
    assert_eq!(known.text(), Some("known"));
    assert_eq!(unknown.text(), None);
}
