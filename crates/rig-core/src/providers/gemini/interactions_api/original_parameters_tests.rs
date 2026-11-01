//! How the original `additional_params` project before they are merged:
//! absent fields leave, the original fields keep their types, and
//! `generation_config` keeps only what its nine declared members (and its
//! unknown members) state, in every shape the original codec accepts.

use super::*;

/// The parameters `original` projects to, or the refusal's text.
fn projected(original: Value) -> Result<Option<Value>, String> {
    project_original_parameters(Some(&original)).map_err(|error| error.to_string())
}

/// The `generation_config` that `config` projects to; `None` when the
/// projection omits it.
fn generation_config(config: Value) -> Result<Option<Value>, String> {
    Ok(projected(json!({ "generation_config": config }))?
        .expect("an object projects to an object")
        .get("generation_config")
        .cloned())
}

/// The `tool_choice` that `choice` projects to inside `generation_config`.
fn tool_choice(choice: Value) -> Result<Option<Value>, String> {
    Ok(generation_config(json!({ "tool_choice": choice }))?
        .map(|config| config["tool_choice"].clone()))
}

/// The text of a refusal whose JSON reason reads `message`, wrapped as
/// the projection wraps it.
fn json_refusal(message: &str) -> String {
    use serde::de::Error as _;
    EncodeError::request(ProviderError::from(serde_json::Error::custom(message))).to_string()
}

/// The body `additional_params` builds on the wire for a plain request.
fn body_with(additional_params: Value) -> Result<Value, String> {
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-flash");
    let mut request = CompletionRequest::new("hello");
    request.additional_params = Some(additional_params);
    let body = create_request_body(&wire, &request, None).map_err(|error| error.to_string())?;
    Ok(serde_json::to_value(body).expect("the body is JSON"))
}

#[test]
fn a_json_refusal_reads_as_a_request_error_holding_the_json_reason() {
    assert_eq!(json_refusal("reason"), "RequestError: JsonError: reason");
}

/// No parameters project to none; null `store`, `background` and
/// `generation_config` leave, as does an `agent_config` without an agent.
/// A value that is not an object passes through for the raw layer to
/// refuse.
#[test]
fn absent_original_parameters_are_omitted() {
    assert_eq!(
        project_original_parameters(None).map_err(|error| error.to_string()),
        Ok(None)
    );
    assert_eq!(
        projected(json!({
            "store": null,
            "background": null,
            "generation_config": null,
            "agent_config": {"type": "dynamic"},
            "previous_interaction_id": "int_0",
        })),
        Ok(Some(json!({"previous_interaction_id": "int_0"})))
    );
    assert_eq!(
        projected(json!({"agent": null, "agent_config": {"type": "dynamic"}})),
        Ok(Some(json!({"agent": null})))
    );
    assert_eq!(projected(json!("raw")), Ok(Some(json!("raw"))));
}

/// Set original fields of their declared types pass unchanged, an agent
/// keeps its configuration, and a generation config whose members are
/// all unset is omitted.
#[test]
fn set_original_fields_of_their_types_pass_unchanged() {
    let original = json!({
        "store": true,
        "background": false,
        "stream": true,
        "agent": "deep-research-pro-preview-12-2025",
        "agent_config": {"type": "deep-research"},
        "tools": [],
    });
    assert_eq!(projected(original.clone()), Ok(Some(original)));
    assert_eq!(
        projected(json!({"generation_config": {"seed": null}, "store": false})),
        Ok(Some(json!({"store": false})))
    );
    assert_eq!(generation_config(json!({})), Ok(None));
}

/// An original field of the wrong type is refused with the original
/// codec's reason before anything merges.
#[test]
fn original_fields_of_the_wrong_type_are_refused() {
    for (original, reason) in [
        (
            json!({"store": "yes"}),
            "invalid type: string \"yes\", expected a boolean",
        ),
        (
            json!({"background": 1}),
            "invalid type: integer `1`, expected a boolean",
        ),
        (
            json!({"stream": "no"}),
            "invalid type: string \"no\", expected a boolean",
        ),
        (
            json!({"agent": 7}),
            "invalid type: integer `7`, expected a string",
        ),
        (
            json!({"tools": {}}),
            "invalid type: map, expected a sequence",
        ),
    ] {
        assert_eq!(
            projected(original.clone()),
            Err(json_refusal(reason)),
            "{original}"
        );
    }
}

/// An object keeps every set declared member in its projected form and
/// every unknown member verbatim, in the order received; unset declared
/// members leave.
#[test]
fn a_generation_config_object_projects_its_declared_members() {
    assert_eq!(
        generation_config(json!({
            "top_k": 40,
            "temperature": 1,
            "top_p": 0.5,
            "seed": 7,
            "stop_sequences": ["END"],
            "tool_choice": "auto",
            "thinking_level": {"high": null},
            "thinking_summaries": "none",
            "max_output_tokens": 64,
            "speech_config": [
                {"voice": "Kore", "language": "en-US", "speaker": null, "style": {"pace": "slow"}},
            ],
            "response_logprobs": null,
        })),
        Ok(Some(json!({
            "top_k": 40,
            "temperature": 1.0,
            "top_p": 0.5,
            "seed": 7,
            "stop_sequences": ["END"],
            "tool_choice": "auto",
            "thinking_level": "high",
            "thinking_summaries": "none",
            "max_output_tokens": 64,
            "speech_config": [{"voice": "Kore", "language": "en-US", "style": {"pace": "slow"}}],
            "response_logprobs": null,
        })))
    );
}

/// With no declared member set, unknown members keep the whole object
/// verbatim, unset declared members included; with neither, it leaves.
#[test]
fn a_generation_config_with_only_unknown_members_is_kept_whole() {
    let config = json!({"top_k": 40, "temperature": null});
    assert_eq!(generation_config(config.clone()), Ok(Some(config)));
    assert_eq!(
        generation_config(json!({"temperature": null, "seed": null})),
        Ok(None)
    );
}

/// An array states the nine members by position: exactly nine, each read
/// by its own codec before the length is checked.
#[test]
fn a_generation_config_array_is_positional() {
    assert_eq!(
        generation_config(json!([
            0.5,
            null,
            3,
            null,
            "any",
            "low",
            "auto",
            10,
            [["Puck"]]
        ])),
        Err(json_refusal(
            "invalid length 1, expected struct SpeechConfig with 3 elements"
        ))
    );
    assert_eq!(
        generation_config(json!([0.5, null, 3, null, "any", "low", "auto", 10, null])),
        Ok(Some(json!({
            "temperature": 0.5,
            "seed": 3,
            "tool_choice": "any",
            "thinking_level": "low",
            "thinking_summaries": "auto",
            "max_output_tokens": 10,
        })))
    );
    assert_eq!(
        generation_config(Value::Array(vec![Value::Null; 9])),
        Ok(None)
    );
    for (config, reason) in [
        (
            json!([]),
            "invalid length 0, expected struct GenerationConfig with 9 elements",
        ),
        (
            json!([0.5, 1.0]),
            "invalid length 2, expected struct GenerationConfig with 9 elements",
        ),
        (
            json!([null, null, null, null, null, null, null, null, null, null]),
            "invalid length 10, expected fewer elements in array",
        ),
        (json!(["hot"]), "invalid type: string \"hot\", expected f64"),
    ] {
        assert_eq!(
            generation_config(config.clone()),
            Err(json_refusal(reason)),
            "{config}"
        );
    }
}

/// A generation config that is neither an object nor an array is refused
/// as the original struct refuses it.
#[test]
fn a_generation_config_of_another_shape_is_refused() {
    for (config, reason) in [
        (
            json!("x"),
            "invalid type: string \"x\", expected struct GenerationConfig",
        ),
        (
            json!(true),
            "invalid type: boolean `true`, expected struct GenerationConfig",
        ),
        (
            json!(3),
            "invalid type: integer `3`, expected struct GenerationConfig",
        ),
    ] {
        assert_eq!(
            generation_config(config.clone()),
            Err(json_refusal(reason)),
            "{config}"
        );
    }
}

/// Each declared member refuses a value its original codec refuses.
#[test]
fn generation_members_of_the_wrong_type_are_refused() {
    for (config, reason) in [
        (
            json!({"temperature": "hot"}),
            "invalid type: string \"hot\", expected f64",
        ),
        (
            json!({"top_p": true}),
            "invalid type: boolean `true`, expected f64",
        ),
        (
            json!({"seed": "7"}),
            "invalid type: string \"7\", expected u64",
        ),
        (
            json!({"max_output_tokens": -1}),
            "invalid value: integer `-1`, expected u64",
        ),
        (
            json!({"stop_sequences": "END"}),
            "invalid type: string \"END\", expected a sequence",
        ),
        (
            json!({"stop_sequences": [1]}),
            "invalid type: integer `1`, expected a string",
        ),
        (
            json!({"speech_config": {"voice": "Kore"}}),
            "invalid type: map, expected a sequence",
        ),
    ] {
        assert_eq!(
            generation_config(config.clone()),
            Err(json_refusal(reason)),
            "{config}"
        );
    }
}

/// The thinking level is a unit enum: a variant name, or a single-key map
/// with a null payload, of the four declared levels.
#[test]
fn a_thinking_level_reads_as_its_unit_enum() {
    for level in ["minimal", "low", "medium", "high"] {
        assert_eq!(
            generation_config(json!({ "thinking_level": level })),
            Ok(Some(json!({ "thinking_level": level })))
        );
        assert_eq!(
            generation_config(json!({ "thinking_level": { level: null } })),
            Ok(Some(json!({ "thinking_level": level })))
        );
    }
    for (level, reason) in [
        (
            json!("max"),
            "unknown variant `max`, expected one of `minimal`, `low`, `medium`, `high`",
        ),
        (
            json!({"low": 1}),
            "invalid type: integer `1`, expected unit",
        ),
        (
            json!(3),
            "invalid type: integer `3`, expected string or map",
        ),
        (
            json!({"low": null, "high": null}),
            "invalid value: map, expected map with a single key",
        ),
    ] {
        assert_eq!(
            generation_config(json!({ "thinking_level": level })),
            Err(json_refusal(reason)),
            "{level}"
        );
    }
}

/// The thinking summaries are a unit enum of `auto` and a set `none`.
#[test]
fn thinking_summaries_read_as_their_unit_enum() {
    for summaries in ["auto", "none"] {
        assert_eq!(
            generation_config(json!({ "thinking_summaries": summaries })),
            Ok(Some(json!({ "thinking_summaries": summaries })))
        );
        assert_eq!(
            generation_config(json!({ "thinking_summaries": { summaries: null } })),
            Ok(Some(json!({ "thinking_summaries": summaries })))
        );
    }
    for (summaries, reason) in [
        (
            json!("brief"),
            "unknown variant `brief`, expected `auto` or `none`",
        ),
        (
            json!({"auto": "x"}),
            "invalid type: string \"x\", expected unit",
        ),
    ] {
        assert_eq!(
            generation_config(json!({ "thinking_summaries": summaries })),
            Err(json_refusal(reason)),
            "{summaries}"
        );
    }
}

/// Each speech config is an object of string members (unknown members
/// verbatim) or exactly three positional strings; any other shape, or a
/// member that is not a string, is refused.
#[test]
fn speech_configs_project_their_three_string_members() {
    assert_eq!(
        generation_config(json!({"speech_config": [["Puck", null, "Host"], {"voice": null}]})),
        Ok(Some(
            json!({"speech_config": [{"voice": "Puck", "speaker": "Host"}, {}]})
        ))
    );
    assert_eq!(
        generation_config(json!({"speech_config": []})),
        Ok(Some(json!({"speech_config": []})))
    );
    for (speech, reason) in [
        (
            json!(["Puck"]),
            "invalid length 1, expected struct SpeechConfig with 3 elements",
        ),
        (
            json!(["a", "b", "c", "d"]),
            "invalid length 4, expected fewer elements in array",
        ),
        (
            json!("Puck"),
            "invalid type: string \"Puck\", expected struct SpeechConfig",
        ),
        (
            json!(null),
            "invalid type: null, expected struct SpeechConfig",
        ),
        (
            json!({"voice": 1}),
            "invalid type: integer `1`, expected a string",
        ),
        (
            json!([null, 1, null]),
            "invalid type: integer `1`, expected a string",
        ),
    ] {
        assert_eq!(
            generation_config(json!({ "speech_config": [speech] })),
            Err(json_refusal(reason)),
            "{speech}"
        );
    }
}

/// A tool choice is a mode (a name, or a single-key map with an empty
/// payload) or an allowed-tools configuration (under `allowed_tools`, or
/// as a one-element array), as a map or a `[mode, tools]` pair. Null
/// members are unset; other members are not read.
#[test]
fn a_tool_choice_projects_its_mode_or_allowed_tools() {
    for (choice, expected) in [
        (json!("auto"), json!("auto")),
        (json!("any"), json!("any")),
        (json!("none"), json!("none")),
        (json!("validated"), json!("validated")),
        (json!({"auto": null}), json!("auto")),
        (json!({"validated": {}}), json!("validated")),
        (
            json!({"allowed_tools": {"tools": ["a", "b"], "mode": "validated"}}),
            json!({"allowed_tools": {"mode": "validated", "tools": ["a", "b"]}}),
        ),
        (
            json!({"allowed_tools": {"mode": {"any": null}, "tools": null, "note": 5}, "other": true}),
            json!({"allowed_tools": {"mode": "any"}}),
        ),
        (json!({"allowed_tools": {}}), json!({"allowed_tools": {}})),
        (
            json!([{"tools": ["f"]}]),
            json!({"allowed_tools": {"tools": ["f"]}}),
        ),
        (
            json!([["none", ["f"]]]),
            json!({"allowed_tools": {"mode": "none", "tools": ["f"]}}),
        ),
        (json!([[null, null]]), json!({"allowed_tools": {}})),
    ] {
        assert_eq!(tool_choice(choice.clone()), Ok(Some(expected)), "{choice}");
    }
    assert_eq!(tool_choice(Value::Null), Ok(None));
}

/// Anything else matches no variant of the original untagged enum.
#[test]
fn a_tool_choice_of_no_variant_is_refused() {
    for choice in [
        json!("required"),
        json!({"any": {"x": 1}}),
        json!({"any": 1}),
        json!({"allowed_tools": null}),
        json!({"allowed_tools": ["auto"]}),
        json!({"allowed_tools": {"mode": "required"}}),
        json!({"allowed_tools": {"tools": [1]}}),
        json!({"allowed_tools": {"tools": "f"}}),
        json!([["sometimes", ["f"]]]),
        json!(["auto", ["f"]]),
        json!({"mode": "auto", "tools": []}),
        json!(3),
    ] {
        assert_eq!(
            tool_choice(choice.clone()),
            Err(json_refusal(
                "data did not match any variant of untagged enum ToolChoice"
            )),
            "{choice}"
        );
    }
}

/// On the wire the projection replaces the raw parameters before they
/// merge, and its refusals refuse the request.
#[test]
fn the_wire_merges_the_projected_parameters() {
    let body = body_with(json!({
        "store": null,
        "agent_config": {"type": "dynamic"},
        "generation_config": {"temperature": 1, "top_k": 40, "seed": null},
    }))
    .expect("the request builds");
    assert_eq!(
        body.get("generation_config"),
        Some(&json!({"temperature": 1.0, "top_k": 40}))
    );
    assert_eq!(body.get("store"), None);
    assert_eq!(body.get("agent_config"), None);
    assert_eq!(
        body_with(json!({"generation_config": {"seed": "7"}})),
        Err(json_refusal("invalid type: string \"7\", expected u64"))
    );
}

/// A raw `response_format` needs its `response_mime_type`.
#[test]
fn a_response_format_requires_its_mime_type() {
    assert_eq!(
        body_with(json!({"response_format": {"type": "object"}})),
        Err("RequestError: response_mime_type is required when response_format is set".to_owned())
    );
    let body = body_with(json!({
        "response_format": {"type": "object"},
        "response_mime_type": "application/json",
    }))
    .expect("the request builds");
    assert_eq!(
        body.get("response_format"),
        Some(&json!({"type": "object"}))
    );
    assert_eq!(
        body.get("response_mime_type"),
        Some(&json!("application/json"))
    );
}
