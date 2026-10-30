//! What an output-budget cut does to a tool call, on both transports.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10964-b29c606e2. A
//! model that reliably makes the call is the precondition for cutting one in
//! half.
//!
//! # Why this dimension exists
//!
//! [#2359](https://github.com/0xPlaygrounds/rig/pull/2359) established a
//! shared policy for the state llama.cpp produces here: a turn that ran out of
//! output budget mid-arguments comes back with `finish_reason: "length"` and
//! `tool_calls[].function.arguments` cut partway through the JSON object. The
//! decoder now keeps the call, as pi does: its arguments are what pi's
//! tolerant parse reads, and its text is kept as `invalid_arguments`, so the
//! agent answers it with an error result and never runs it.
//!
//! That policy was derived from DeepSeek and Mistral. llama.cpp is a third
//! wire that produces the same state, from a completely different server, and
//! nothing had checked the policy against it. Measured on b10499-6d05498 with
//! `tool_choice: required`, the cut lands exactly where the budget runs out:
//! 12 tokens gives `{`, 20 gives `{"note": "The`, 28 gives
//! `{"note": "The quarterly incident review found three unrelate`.
//!
//! The port replaces upstream's retention on Chat
//! (coord-a-rig-port-discard-15): a function call cut at an explicit
//! `length` finish is discarded and yields no canonical call. On a stream its
//! Part closes with the typed `Discard` reason `Length`, and the argument
//! fragments it delivered stay visible as `Arguments` events; a unary reply
//! drops the call before decoding, its arguments visible in the raw
//! document. The turn itself, its usage and finish reason included, still
//! decodes.
//!
//! | Cell | Transport | Pinned |
//! | --- | --- | --- |
//! | [`a_tool_call_cut_mid_arguments_does_not_destroy_the_turn`] | blocking | the response decodes with no call; the cut arguments stay visible in its raw document (a unary reply has no event surface, so no typed `Discard`) |
//! | [`the_streaming_path_discards_the_same_cut_call`] | streaming | the same discard on the stream, which still terminates cleanly |

use rig::completion::FinishReason;
use rig::message::AssistantContent;
use rig::streaming::{DiscardReason, Item, PartKind, StreamEvent};
use serde_json::Value;

use crate::cassettes::{recorded_sse_json_frames, recorded_statuses_and_bodies};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const NOTE_PROMPT: &str = "/no_think Record this note: The quarterly incident review found \
     three unrelated regressions in the billing pipeline.";

/// A budget measured to cut inside the argument object rather than before it.
const CUTTING_CAP: u64 = 20;

fn record_tool() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: rig_core::message::ToolName::new("record").expect("tool name"),
        description: "Record a long note.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "note": { "type": "string" } },
            "required": ["note"],
        }),
    }
}

/// Each function call `items` close with a typed `Discard`, in stream order:
/// the tool its Part started under, the reason, and the argument fragments
/// the Part delivered, joined.
fn discarded_calls(items: &[Item<StreamEvent>]) -> Vec<(String, DiscardReason, String)> {
    let mut discarded = Vec::new();
    for item in items {
        let Item::Event(StreamEvent::Discard { part, reason }) = item else {
            continue;
        };
        let name = items
            .iter()
            .find_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part: started,
                    kind: PartKind::ToolCall,
                    name,
                    ..
                }) if started == part => name.as_ref().map(|name| name.as_str().to_owned()),
                _ => None,
            })
            .unwrap_or_default();
        let fragments: String = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Arguments { part: grown, json }) if grown == part => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect();
        discarded.push((name, *reason, fragments));
    }
    discarded
}

/// How many function calls `items` end with a canonical call.
fn ended_calls(items: &[Item<StreamEvent>]) -> usize {
    items
        .iter()
        .filter(|item| {
            matches!(
                item,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                })
            )
        })
        .count()
}

/// The recorded `(finish_reason, arguments)` of a blocking turn.
fn recorded_call(scenario: &str) -> (String, Option<String>) {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    let choice = &response["choices"][0];
    (
        choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        choice["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .map(str::to_string),
    )
}

/// A tool call cut mid-arguments must not take the turn down with it.
#[tokio::test]
async fn a_tool_call_cut_mid_arguments_does_not_destroy_the_turn() {
    with_llamacpp_competent_cassette(
        "truncation_matrix/tool_call_cut_mid_arguments",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(NOTE_PROMPT)
                        .tool(record_tool())
                        .tool_choice(rig::message::ToolChoice::Required)
                        .max_tokens(CUTTING_CAP),
                )
                .await
                .expect(
                    "a budget cut mid-arguments must still decode — losing the turn's \
                     usage, id and finish reason with the unusable call is the defect \
                     #2359 fixed",
                );

            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Length),
                "the turn ran out of budget, and that is what authorizes the tolerance"
            );
            assert!(
                response
                    .choice
                    .iter()
                    .all(|item| !matches!(item, AssistantContent::ToolCall(_))),
                "a call cut at Length is discarded, never a canonical call: {:?}",
                response.choice
            );
            assert!(
                response.usage.output_tokens.is_some_and(|n| n > 0),
                "the rest of the turn survives, usage included: {:?}",
                response.usage
            );

            // A unary reply has no event surface: the port drops the cut
            // call from the whole document before decoding it, so no typed
            // `Discard` exists for it. Its delivered arguments stay visible
            // in the response's raw document.
            let (_, arguments) = recorded_call("truncation_matrix/tool_call_cut_mid_arguments");
            assert_eq!(
                response.raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                    .as_str(),
                arguments.as_deref(),
                "the cut call's delivered arguments stay visible in the raw document"
            );
        },
    )
    .await;

    // The premise, from the bytes: the wire really did carry a truncated
    // argument string under `length`.
    let (finish_reason, arguments) = recorded_call("truncation_matrix/tool_call_cut_mid_arguments");
    assert_eq!(finish_reason, "length");
    let arguments = arguments.expect("the wire carried a tool call");
    assert!(
        serde_json::from_str::<Value>(&arguments).is_err(),
        "the recorded arguments must be *unparseable* for this cell to be about \
         truncation at all: {arguments:?}"
    );
    assert!(
        arguments.starts_with('{'),
        "and they must be a cut-off object rather than an empty string: {arguments:?}"
    );
}

/// The streaming path applies the same boundary.
#[tokio::test]
async fn the_streaming_path_discards_the_same_cut_call() {
    use futures::StreamExt as _;

    with_llamacpp_competent_cassette(
        "truncation_matrix/streaming_tool_call_cut_mid_arguments",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let mut stream = model
                .stream(
                    CompletionRequest::new(NOTE_PROMPT)
                        .tool(record_tool())
                        .tool_choice(rig::message::ToolChoice::Required)
                        .max_tokens(CUTTING_CAP),
                )
                .expect("stream should start");

            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item.expect("no stream item may be an error"));
            }
            let response = stream.finish().await.expect(
                "the stream must terminate cleanly even though its only tool call was unusable",
            );
            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert_eq!(
                response.tool_calls().count(),
                0,
                "a call cut at Length is discarded, never a canonical call: {:?}",
                response.choice
            );
            assert_eq!(ended_calls(&items), 0, "no completed call is fabricated");
            let discarded = discarded_calls(&items);
            let [(name, reason, fragments)] = discarded.as_slice() else {
                panic!("exactly the cut call is discarded: {discarded:?}");
            };
            assert_eq!(name, "record");
            assert_eq!(
                *reason,
                DiscardReason::Length,
                "the stream closes the cut call with the typed reason Length"
            );
            assert_eq!(
                *fragments,
                recorded_stream_arguments(),
                "the fragments the stream delivered stay visible"
            );
        },
    )
    .await;

    let frames = recorded_sse_json_frames(
        "llamacpp",
        "truncation_matrix/streaming_tool_call_cut_mid_arguments",
    );
    let accumulated = recorded_stream_arguments();
    assert!(
        !accumulated.is_empty() && serde_json::from_str::<Value>(&accumulated).is_err(),
        "the recorded stream must accumulate to an unparseable fragment: {accumulated:?}"
    );
    assert!(
        frames
            .iter()
            .any(|frame| frame["choices"][0]["finish_reason"] == serde_json::json!("length")),
        "and it must end under `length`, which is what authorizes dropping the call"
    );
}

/// The argument text the recorded stream delivered for its one call.
fn recorded_stream_arguments() -> String {
    recorded_sse_json_frames(
        "llamacpp",
        "truncation_matrix/streaming_tool_call_cut_mid_arguments",
    )
    .iter()
    .flat_map(|frame| {
        frame["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    })
    .filter_map(|call| call["function"]["arguments"].as_str().map(str::to_string))
    .collect()
}
