//! The fork's raw-parameter projection: every recognized value round-trips
//! as it decodes, a non-object root is refused as the projection decodes it,
//! and the base refuses the retired incomplete-terminal key whichever upper
//! layer states it.

use serde_json::{Map, Value, json};

use super::{Delivery, UnrecognizedRaw, project_raw_parameters};
use crate::completion::{CompletionRequest, ProviderOptions};
use crate::providers::openai::OpenAIConfig;
use crate::wire::Mode;

const SENT: UnrecognizedRaw = UnrecognizedRaw::Sent {
    typed_output_cap: false,
    typed_temperature: false,
};

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(fields) => fields,
        other => panic!("the fixture is an object, got {other}"),
    }
}

#[test]
fn every_recognized_raw_value_is_sent_as_the_projection_decodes_it() {
    let efforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
    let summaries = ["auto", "concise", "detailed"];
    let contexts = ["auto", "all_turns", "current_turn"];
    let truncations = ["auto", "disabled"];
    for (case, effort) in efforts.into_iter().enumerate() {
        let raw = json!({
            "background": true,
            "text": {"format": {
                "type": "json_schema", "name": "answer",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                "strict": true
            }},
            "include": [
                "file_search_call.results",
                "message.input_image.image_url",
                "computer_call.output.image_url",
                "reasoning.encrypted_content",
                "code_interpreter_call.outputs"
            ],
            "top_p": 0.5,
            "truncation": truncations[case % truncations.len()],
            "user": "user-1",
            "prompt_cache_key": "cache-key",
            "prompt_cache_retention": "24h",
            "metadata": {"trace": "t-1"},
            "parallel_tool_calls": false,
            "previous_response_id": "resp_0",
            "reasoning": {
                "effort": effort,
                "summary": summaries[case % summaries.len()],
                "mode": "pro",
                "context": contexts[case % contexts.len()]
            },
            "service_tier": "flex",
            "store": true,
            "client_metadata": {"session": "s-1"},
            "future_key": [null, 7]
        });
        assert_eq!(
            project_raw_parameters(&raw, SENT).expect("every recognized value decodes"),
            object(raw.clone()),
            "{effort}"
        );
    }
    let plain = json!({"text": {"format": {"type": "text"}}});
    assert_eq!(
        project_raw_parameters(&plain, SENT).expect("a plain text format decodes"),
        object(plain.clone())
    );
}

#[test]
fn a_numeric_raw_root_is_refused_as_the_projection_decodes_it() {
    for unrecognized in [UnrecognizedRaw::Codex, SENT] {
        let error = project_raw_parameters(&json!(5), unrecognized)
            .expect_err("a number is not the additional parameters object");
        assert_eq!(
            error.to_string(),
            "RequestError: Invalid OpenAI Responses additional_params payload: invalid type: integer `5`, expected struct RawParameterProjection",
            "{unrecognized:?}"
        );
    }
}

/// The projection refuses the retired key in `additional_params`; the base
/// refuses it from the provider options, the other layer it reads.
#[test]
fn the_retired_incomplete_key_is_refused_from_the_provider_options() {
    let mut request = CompletionRequest::new("hello");
    request.provider_options = serde_json::from_value::<ProviderOptions>(json!({
        (crate::providers::openai::PROVIDER_NAME): {"*": {"__rig_tolerate_incomplete": true}}
    }))
    .expect("an object of object sections");
    let error = OpenAIConfig::new("test-key")
        .responses("gpt-5.4")
        .responses_request(&request, Delivery::Http(Mode::Unary))
        .expect_err("the retired key is refused wherever it is stated");
    assert_eq!(
        error.to_string(),
        "RequestError: `additional_params[\"__rig_tolerate_incomplete\"]` is no longer read: select the incomplete-terminal policy on the Responses wire with `Responses::with_streamed_incomplete`"
    );
}
