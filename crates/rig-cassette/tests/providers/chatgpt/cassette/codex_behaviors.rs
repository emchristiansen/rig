//! ChatGPT/Codex Responses backend behavior regression tests.
//!
//! Locks down strict tool schemas, ChatGPT-specific request shaping, SSE
//! reconstruction, and system/default instruction behavior.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::message::{AssistantContent, Message};
const CHATGPT_PROVIDER: &str = "chatgpt";
use rig::providers::chatgpt;
use rig::tool::Tool;
use rig_test_support::cassette_models::MapWire;

use super::super::support::{
    recorded_include, with_chatgpt_cassette, with_chatgpt_cassette_default_instructions,
};
use crate::cassettes::recorded_interaction_bodies;
use crate::support::{Adder, TOOLS_PREAMBLE};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_chatgpt_cassette(
        "codex_behaviors/strict_tools_opt_in_roundtrip",
        |client| async move {
            // The recorded request body locks the strict-tools contract:
            // `strict: true` plus the sanitized schema (additionalProperties
            // false, all properties required) must be accepted by the backend.
            let model = client
                .completion(chatgpt::GPT_5_4)
                .map_wire(|wire| wire.with_strict_tools());
            let request = CompletionRequest::new("Use the add tool to add 7 and 5.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&Adder))
                .additional_params(recorded_include());

            let response = model
                .call(request)
                .await
                .expect("strict-tools completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("strict tool call should be produced");
            assert_eq!(tool_call.function.name, Adder::NAME);
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("x")
                    .and_then(serde_json::Value::as_f64),
                Some(7.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("y")
                    .and_then(serde_json::Value::as_f64),
                Some(5.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
        },
    )
    .await;
}

#[tokio::test]
async fn store_false_and_prompt_cache_fields_roundtrip() {
    let scenario = "codex_behaviors/store_false_and_prompt_cache_fields_roundtrip";
    with_chatgpt_cassette(
        "codex_behaviors/store_false_and_prompt_cache_fields_roundtrip",
        |client| async move {
            let model = client.completion(chatgpt::GPT_5_4);
            // The actual terminal raw keeps provider-only fields; usage is
            // checked on the response normalized from that terminal.
            let response = model
                .call(
                    CompletionRequest::new("Reply with exactly this marker: CODEX-STORE-FALSE")
                        .preamble("Return only the requested marker.".to_string())
                        .additional_params(recorded_include()),
                )
                .await
                .expect("basic ChatGPT/Codex completion should succeed");
            assert_eq!(
                response.raw["store"], false,
                "ChatGPT provider must force store=false"
            );
            assert!(
                response.raw["prompt_cache_key"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "ChatGPT backend should return a prompt cache key that cassettes scrub"
            );
            assert!(response.usage.input_tokens.is_some_and(|n| n > 0));
            assert!(response.usage.output_tokens.is_some_and(|n| n > 0));
        },
    )
    .await;

    // `store` is rig's own request field, so the recorded request bytes are
    // where that contract actually lives; the read off the reply above only
    // proves the backend echoed it back.
    let interactions = recorded_interaction_bodies(CHATGPT_PROVIDER, scenario);
    assert_eq!(
        interactions.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let request: serde_json::Value =
        serde_json::from_str(&interactions[0].0).expect("the recorded request body should be JSON");
    assert_eq!(
        request.get("store"),
        Some(&serde_json::Value::Bool(false)),
        "the request ChatGPT receives must carry store=false"
    );
}

#[tokio::test]
async fn default_instructions_merge_with_explicit_preamble() {
    with_chatgpt_cassette_default_instructions(
        "codex_behaviors/default_instructions_merge_with_explicit_preamble",
        "Default instruction marker: always include DEFAULT-CODEX-MARKER when asked for the default marker.",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
                .preamble("Explicit instruction marker: also include EXPLICIT-CODEX-MARKER.")
                .additional_params(recorded_include())
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(
                    "List the default marker and the explicit marker, and nothing else.",
                    &mut history,
                )
                .await
                .expect("default and explicit instructions should both reach the backend");

            assert!(
                result.output().contains("DEFAULT-CODEX-MARKER")
                    && result.output().contains("EXPLICIT-CODEX-MARKER"),
                "merged instructions should influence the answer, got {result:?}"
            );
        },
    )
    .await;
}

#[cfg(test)]
mod explicit_history_tests;
