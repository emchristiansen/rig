use super::{Choice, CompletionResponse, Message, Usage};
use serde_json::{Value, json};

fn recorded(finish: &str, calls: Value) -> Value {
    json!({
        "id": "chat_recorded", "model": "recorded-model",
        "choices": [{"message": {"role": "assistant", "content": "answer",
            "tool_calls": calls}, "finish_reason": finish}]
    })
}

fn function(id: &str, arguments: Value) -> Value {
    json!({"id": id, "function": {"name": "lookup", "arguments": arguments}})
}

#[test]
fn recorded_null_envelope_fields_keep_exact_defaults_and_omissions() {
    let mut raw = recorded("", json!(null));
    raw["object"] = Value::Null;
    raw["created"] = Value::Null;
    raw["choices"][0]["index"] = Value::Null;
    raw["choices"][0]["finish_reason"] = Value::Null;
    let response: CompletionResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(response.id, "chat_recorded");
    assert_eq!(response.model, "recorded-model");
    assert_eq!(response.object, "");
    assert_eq!(response.created, 0);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].finish_reason, "");
    let serialized = serde_json::to_value(response).unwrap();
    assert_eq!(serialized["system_fingerprint"], Value::Null);
    assert_eq!(serialized["usage"], Value::Null);
    assert!(serialized.get("service_tier").is_none());
    assert!(
        serialized["choices"][0]["message"]
            .get("tool_calls")
            .is_none()
    );
}

#[test]
fn recorded_gateway_role_reasoning_alias_and_signature_remain_typed() {
    let mut raw = recorded("future_finish", json!([]));
    raw["choices"][0]["message"] = json!({
        "role": "model", "content": "answer", "reasoning": "hidden",
        "reasoning_details": [{"type": "reasoning.text", "text": "thought",
            "signature": "literal-provider-signature"}]
    });
    let response: CompletionResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(response.choices[0].finish_reason, "future_finish");
    let serialized = serde_json::to_value(response).unwrap();
    let message = &serialized["choices"][0]["message"];
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["reasoning_content"], "hidden");
    assert!(message.get("reasoning").is_none());
    let detail = &message["reasoning_details"][0];
    assert_eq!(detail["signature"], "literal-provider-signature");
    for key in ["id", "format", "index"] {
        assert_eq!(detail.get(key), Some(&Value::Null));
    }
}

#[test]
fn recorded_length_drops_only_cut_calls_and_retains_valid_sibling() {
    for finish in ["length", "max_tokens", "model_length"] {
        let raw = recorded(
            finish,
            json!([
                function("before", json!("{\"x\":1}")),
                function("empty", json!("")),
                function("cut", json!("{\"x\":")),
                function("after", json!({"x": 2}))
            ]),
        );
        let response: CompletionResponse = serde_json::from_value(raw).unwrap();
        let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
            panic!("recorded assistant changed role");
        };
        assert_eq!(
            tool_calls
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>(),
            ["before", "after"]
        );
        assert_eq!(tool_calls[0].function.arguments, json!({"x": 1}));
        assert_eq!(tool_calls[1].function.arguments, json!({"x": 2}));
        assert_eq!(response.choices[0].finish_reason, finish);
    }
}

#[test]
fn recorded_nonlength_malformed_arguments_remain_errors() {
    for finish in ["stop", "tool_calls", "future_finish", "LENGTH"] {
        let raw = recorded(finish, json!([function("cut", json!("{\"x\":"))]));
        assert!(
            serde_json::from_value::<CompletionResponse>(raw).is_err(),
            "{finish}"
        );
    }
}

#[test]
fn recorded_length_repair_does_not_hide_missing_function_name() {
    let raw = recorded(
        "length",
        json!([{
            "id": "cut", "function": {"arguments": ""}
        }]),
    );
    let strict_error = serde_json::from_value::<Choice>(raw["choices"][0].clone()).unwrap_err();
    let error = serde_json::from_value::<CompletionResponse>(raw).unwrap_err();
    assert_eq!(error.to_string(), strict_error.to_string());
    let valid_arguments = recorded(
        "length",
        json!([{
            "id": "cut", "function": {"arguments": "{}"}
        }]),
    );
    let error = serde_json::from_value::<CompletionResponse>(valid_arguments).unwrap_err();
    assert_eq!(error.to_string(), "missing field `name`");
}

#[test]
fn recorded_length_repair_does_not_hide_wrong_call_identity_type() {
    let raw = recorded(
        "length",
        json!([{
            "id": 7, "function": {"name": "lookup", "arguments": ""}
        }]),
    );
    let strict_error = serde_json::from_value::<Choice>(raw["choices"][0].clone()).unwrap_err();
    let error = serde_json::from_value::<CompletionResponse>(raw).unwrap_err();
    assert_eq!(error.to_string(), strict_error.to_string());
    let valid_arguments = recorded(
        "length",
        json!([{
            "id": 7, "function": {"name": "lookup", "arguments": "{}"}
        }]),
    );
    let error = serde_json::from_value::<CompletionResponse>(valid_arguments).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid type: integer `7`, expected a string"
    );
}

#[test]
fn recorded_required_fields_choices_shape_and_usage_are_not_relaxed() {
    for key in ["id", "model", "choices"] {
        let mut raw = recorded("stop", json!([]));
        raw.as_object_mut().unwrap().shift_remove(key);
        let error = serde_json::from_value::<CompletionResponse>(raw).unwrap_err();
        assert_eq!(error.to_string(), format!("missing field `{key}`"));
    }
    let mut raw = recorded("stop", json!([]));
    raw["choices"] = Value::Null;
    assert!(serde_json::from_value::<CompletionResponse>(raw).is_err());
    let mut raw = recorded("stop", json!([]));
    raw["usage"] = json!({"prompt_tokens": "7", "total_tokens": 7});
    assert!(serde_json::from_value::<CompletionResponse>(raw).is_err());
}

#[test]
fn recorded_usage_keeps_groq_timings_audio_cache_and_root_type_identity() {
    let mut raw = recorded("stop", json!([]));
    raw["usage"] = json!({
        "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 17,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 4,
            "cache_write_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
        "num_cached_tokens": 9, "queue_time": 0.1, "prompt_time": 0.2,
        "completion_time": 0.3, "total_time": 0.6
    });
    let response: CompletionResponse = serde_json::from_value(raw).unwrap();
    let usage: Usage = response.usage.unwrap();
    let root_usage: crate::providers::openai::Usage = usage;
    assert_eq!(root_usage.queue_time, Some(0.1));
    assert_eq!(root_usage.prompt_time, Some(0.2));
    assert_eq!(root_usage.completion_time, Some(0.3));
    assert_eq!(root_usage.total_time, Some(0.6));
    let normalized = root_usage.to_normalized();
    assert_eq!(normalized.input_tokens, Some(14));
    assert_eq!(normalized.output_tokens, Some(3));
    assert_eq!(normalized.cached_input_tokens, Some(0));
    assert_eq!(normalized.cache_creation_input_tokens, Some(2));
    assert_eq!(normalized.reasoning_tokens, Some(1));
}

#[test]
fn recorded_message_dependencies_keep_user_and_tool_result_serde() {
    let user: Message = serde_json::from_value(json!({
        "role": "user", "content": "question"
    }))
    .unwrap();
    assert_eq!(
        serde_json::to_value(user).unwrap(),
        json!({
            "role": "user", "content": "question"
        })
    );
    let tool: Message = serde_json::from_value(json!({
        "role": "tool", "tool_call_id": "provider-call",
        "content": [{"type": "text", "text": "result"},
            {"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}]
    }))
    .unwrap();
    let serialized = serde_json::to_value(tool).unwrap();
    assert_eq!(serialized["tool_call_id"], "provider-call");
    assert_eq!(
        serialized["content"][1]["image_url"],
        json!({"url": "https://example.invalid/image"})
    );
}
