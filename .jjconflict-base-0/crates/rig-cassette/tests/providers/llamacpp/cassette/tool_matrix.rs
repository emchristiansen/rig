//! The tool-calling matrix.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10964-b29c606e2 —
//! unless a cell says otherwise. The escalation is deliberate and is the rule
//! stated in `cassette_support`: a 1.7B model declines tool calls often enough
//! that a red cell would be a coin flip on the model rather than a finding
//! about rig, and "the model refused" and "rig dropped the tools" look
//! identical from the outside. Arity, `tool_choice` and result-shape cells all
//! need the model to actually make the call they are about, so they run here.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`tool_choice_specific_is_refused_on_the_streaming_path_too`] | the same, streaming | a guard covering one surface would let the old behaviour back |
//! | tool result carrying an image | result: image | `image_tool_result.rs`, on the vision server |
//!
//! # Two interesting rows
//!
//! `tool_choice` naming a specific function is **not representable** on this
//! wire. `llama-server` reads the field with
//! `json_value(body, "tool_choice", std::string("auto"))` — as a string — and
//! its vocabulary is exactly `auto | none | required`
//! (`common_chat_tool_choice_parse_oaicompat`). An OpenAI-shaped object
//! type-mismatches that read and falls through to the default, so the request
//! is served as `auto` and the caller silently gets whichever tool the model
//! felt like. The `BodyRewrite::LlamaCpp` arm of `Chat::encode` refuses it
//! locally instead.
//!
//! And `tool_choice: "none"`:
//!
//! OpenAI's contract for `none` is "the tools are still described to the model,
//! but do not call one". llama.cpp implements that by leaving the tool
//! definitions in the rendered prompt while switching off the tool-call
//! *parser*, so a model that decides to call one anyway emits its template's
//! raw syntax — `<tool_call>{"name": …}</tool_call>` for Qwen — as ordinary
//! assistant text. Rig surfaces exactly what the server sent, which is right;
//! the cell records the shape so nobody mistakes it for a rig defect later.

use rig::message::ToolChoice;
use rig::providers::openai::wire::{LLAMACPP, OpenAIConfig};

use crate::support::{Adder, OperationArgs, Subtract};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;
use rig::message::{Message, AssistantMessage, AssistantContent, UserContent, ToolResult, ToolResultContent, CallId, ToolName};
use serde_json::json;
use crate::cassettes::recorded_json_request;

const NO_THINK: &str = "/no_think ";

// ---------------------------------------------------------------------------
// Arity
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// tool_choice
// ---------------------------------------------------------------------------

/// The refusal applies on the **streaming** transport too.
///
/// A guard that only covered one surface would be worse than none: a caller
/// who moved from `completion` to `stream` would silently get the old
/// behaviour back. `Chat::prepare` runs from `Chat::encode`, which both
/// `Mode::Unary` and `Mode::Stream` go through, and this cell is what makes
/// that a checked fact rather than a reading.
#[tokio::test]
async fn tool_choice_specific_is_refused_on_the_streaming_path_too() {
    let model = OpenAIConfig::with_key(&LLAMACPP, "")
        .with_base_url("http://127.0.0.1:1/v1")
        .connect(rig_test_support::cassettes::local_reqwest())
        .completion(CASSETTE_MODEL);

    let error = model
        .stream(
            CompletionRequest::new(format!("{NO_THINK}Compute 2 + 3."))
                .tool(rig::tool::tool_definition(&Adder))
                .tool(rig::tool::tool_definition(&Subtract))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::message::ToolName::new("subtract").expect("tool name"),
                    ],
                })
                .max_tokens(256),
        )
        .err()
        .expect("opening the stream must fail before anything is sent");

    assert!(
        matches!(error, rig::error::ProviderError::Request(_)),
        "the refusal must be rig's own, not the connection error the dead \
         address would produce: {error:?}"
    );
    assert!(error.to_string().contains("subtract"), "{error}");
}

// ---------------------------------------------------------------------------
// Tool result payloads
// ---------------------------------------------------------------------------

fn lookup_call_turn(id: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
        id,
        ToolName::new("lookup").expect("tool name"),
        json!({}),
    )]))
}

/// A tool result carrying plain text.
#[tokio::test]
async fn a_tool_result_carrying_text_reaches_the_model() {
    with_llamacpp_competent_cassette("tool_matrix/tool_result_text", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let response = model
            .call(CompletionRequest::new(Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            call: CallId::from_wire("call_text"),
                            name: ToolName::new("lookup").expect("tool name"),
                            answers: rig_core::message::AnsweredToolCall::Function,
                                is_error: false,
                            content: vec![ToolResultContent::text("the codeword is heliotrope")],
                        })],
                    })
                    .preamble(
                        "Answer using only the tool result you were given. \
                         Repeat the codeword verbatim."
                            .to_string(),
                    )
                    // A tool result answers a tool *call*: the history has to
                    // carry the assistant turn that asked for it, or the chat
                    // template renders an orphan tool message and the model
                    // has nothing to answer.
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("What is the codeword?")],
                        },
                        lookup_call_turn("call_text"),
                    ])
                    .max_tokens(512)
                    ,
            )
            .await
            .expect("a text tool result should be accepted");

        let text = crate::support::assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.to_ascii_lowercase().contains("heliotrope"),
            "the tool result's text must reach the model: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "tool_matrix/tool_result_text");
    let roles = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert!(
        roles.contains(&"tool".to_string()),
        "the result must ride a `role: \"tool\"` message: {roles:?}"
    );
}

/// A tool result carrying a JSON document.
///
/// The Chat Completions wire has no JSON content part, so a structured result
/// is serialized into the tool message's text. The cell pins that the
/// serialization is the document — not a `Debug` rendering, and not dropped.
#[tokio::test]
async fn a_tool_result_carrying_json_reaches_the_model() {
    with_llamacpp_competent_cassette("tool_matrix/tool_result_json", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let response = model
            .call(CompletionRequest::new(Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            call: CallId::from_wire("call_json"),
                            name: ToolName::new("lookup").expect("tool name"),
                            answers: rig_core::message::AnsweredToolCall::Function,
                                is_error: false,
                            content: vec![ToolResultContent::text(
                                json!({ "codeword": "heliotrope", "confidence": 0.99 }).to_string(),
                            )],
                        })],
                    })
                    .preamble(
                        "Answer using only the JSON tool result you were given. \
                         Report the codeword verbatim."
                            .to_string(),
                    )
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("What is the codeword?")],
                        },
                        lookup_call_turn("call_json"),
                    ])
                    .max_tokens(512)
                    ,
            )
            .await
            .expect("a JSON tool result should be accepted");

        let text = crate::support::assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.to_ascii_lowercase().contains("heliotrope"),
            "the JSON result's contents must reach the model: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "tool_matrix/tool_result_json");
    let tool_message = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("a tool message");
    let serialized = tool_message["content"].to_string();
    assert!(
        serialized.contains("codeword") && serialized.contains("heliotrope"),
        "the document must survive serialization into the tool message: {serialized}"
    );
}

/// The one dimension this file does not own: an *arithmetic* tool round trip
/// through the agent loop, which the pre-merge `tools.rs` already covers on
/// the smoke tier.
///
/// Stated rather than duplicated. `tools::tools_roundtrip` records the same
/// shape against `unsloth/Qwen3-1.7B-GGUF`, and re-recording it here against
/// the competent tier would add a second fixture that tests the model, not
/// rig.
#[test]
fn the_smoke_tier_round_trip_is_covered_elsewhere() {
    let calls = recorded_statuses_and_bodies("llamacpp", "tools/tools_roundtrip");
    assert!(
        calls.len() >= 2,
        "tools/tools_roundtrip must still be a multi-turn round trip, saw {}",
        calls.len()
    );
}

/// Keeps `OperationArgs` referenced from this file so the shared support type
/// cannot be removed without this matrix noticing.
#[allow(dead_code)]
fn _operation_args_is_the_shared_shape(args: OperationArgs) -> (f64, f64) {
    (args.x as f64, args.y as f64)
}
