//! Edge matrix for a streamed tool call whose arguments are not JSON
//! (rig#2447).
//!
//! **Bug.** Anthropic streams tool input as `input_json_delta` fragments and
//! closes the block with `content_block_stop`. When the assembled fragments
//! were not JSON, the reply failed and the run ended: one bad byte from the
//! model ended the run, and the model never learned why.
//!
//! **Contract.** A malformed call never fails the reply. It is kept with
//! the arguments its text still states and the text itself in
//! `invalid_arguments`, and with no provider item, so it replays from its
//! canonical fields. The agent never runs the tool. It offers the call to the
//! invalid-call hook with a `MalformedArguments` reason and the raw text,
//! and, when the hook leaves the decision, answers the call with an
//! `is_error` tool result saying "The arguments for tool `X` are not a JSON
//! object: ...", and the run continues, as pi does.
//!
//! **Fixtures.** Every cell is recorded live. Cell 3 caps its first request
//! at [`CUT_AT`] output tokens, which cuts Sonnet 4.6's `subtract` call
//! inside its input, so the assembled input is not JSON and the turn stops
//! on `max_tokens`. Its recorded follow-up carries the call rebuilt from the
//! arguments the cut text states, with no `caller` (which lived only in the
//! provider item), and the `is_error` result naming the raw text. The cell
//! asserts its fixture really holds a cut input, so a re-record that heals
//! it fails.
//!
//! **Cell 3 under the port.** The cut leaves the call's block open: the
//! stream reaches `message_delta` with no `content_block_stop` for it. A
//! call that never finished fails its turn before the finish reason is
//! weighed (G48b, coord-a-rig-g48b-1; G157, coord-a-rig-g138-g184-1), so the
//! run ends with "the turn failed, so none of its tool calls ran: the
//! provider ended the reply with a tool call it never finished", no tool
//! runs and the recorded follow-up is never sent. The cell therefore serves
//! the recorded first reply itself rather than replaying the whole
//! recording. The malformed-arguments path above still covers a call whose
//! block closed on unparseable input.
//!
//! Anthropic is the provider for this matrix because its wire states the
//! close of every tool-input block. Bedrock (`contentBlockStop`) needs AWS
//! credentials to record and is left as follow-up.
//!
//! | # | cell | transport | input on the wire | fixture |
//! |---|------|-----------|-------------------|---------|
//! | 1 | `blocking_healthy_control` | blocking | valid | recorded |
//! | 2 | `streaming_healthy_control` | streaming | valid | recorded |
//! | 3 | `streaming_malformed_call_is_answered_with_an_error` | streaming | cut by `max_tokens`, block never closed: unfinished-call refusal | recorded |
//!
//! Unit cells for the seam itself live beside the code:
//! `crates/rig-agent/src/agent/streaming/malformed_tool_args_tests.rs` (what
//! the model sees on the next request) and the history conformance row
//! `h06_malformed_arguments`.

use futures::StreamExt;
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, MultiTurnStreamItem,
    RequestPatch,
};
use rig::completion::PromptError;
use rig::error::ProviderError;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;
use crate::cassettes::recorded_interaction_bodies;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_matches_recorded_document, assert_mentions_expected_number,
    collect_stream_final_response,
};

/// The tool input each recorded response stream in `scenario` assembles
/// from its `partial_json` fragments, for each response that streams any.
fn recorded_inputs(scenario: &str) -> Vec<String> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .map(|(_, response)| {
            response
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|json| serde_json::from_str::<Value>(json).ok())
                .filter_map(|event| {
                    event
                        .get("delta")
                        .and_then(|delta| delta.get("partial_json"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<String>()
        })
        .filter(|input| !input.is_empty())
        .collect()
}

fn assert_fixture_is_cut(scenario: &str) {
    let inputs = recorded_inputs(scenario);
    let first = inputs
        .first()
        .unwrap_or_else(|| panic!("{scenario}: fixture should stream tool input fragments"));
    assert!(
        serde_json::from_str::<Value>(first).is_err(),
        "{scenario}: the first call's input must be cut short of JSON; a re-record healed it: {first:?}"
    );
    let interactions = crate::cassettes::recorded_interaction_bodies("anthropic", scenario);
    assert!(
        interactions
            .first()
            .is_some_and(|(_, response)| response.contains(r#""stop_reason":"max_tokens""#)),
        "{scenario}: the first reply must stop on `max_tokens`"
    );
}

fn assert_fixture_is_healthy(scenario: &str) {
    let inputs = recorded_inputs(scenario);
    assert!(
        !inputs.is_empty(),
        "{scenario}: fixture should stream tool input fragments"
    );
    for input in inputs {
        assert!(
            serde_json::from_str::<Value>(&input).is_ok(),
            "{scenario}: the recorded control must assemble to valid JSON, got {input:?}"
        );
    }
}

/// The output cap that cuts Sonnet 4.6's first `subtract` call inside its
/// input.
const CUT_AT: u64 = 60;

/// Caps the first request at [`CUT_AT`] output tokens.
#[derive(Clone)]
struct CutFirstCall;

impl AgentHook for CutFirstCall {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if event.turn == 1 {
            CompletionCallAction::patch(RequestPatch::new().max_tokens(CUT_AT))
        } else {
            CompletionCallAction::Continue
        }
    }
}

/// The calculator agent, allowed `calls` model calls.
fn agent(client: AnthropicModels, calls: usize) -> rig::agent::Agent {
    rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .default_max_turns(calls)
        .build()
}

#[tokio::test]
async fn blocking_healthy_control() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/blocking_healthy_control",
        |client| async move {
            let response = agent(client, 2)
                .prompt(STREAMING_TOOLS_PROMPT)
                .await
                .expect("blocking tool prompt should succeed");
            assert_mentions_expected_number(&response.output(), -3);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_healthy_control() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_healthy_control",
        |client| async move {
            let mut stream = agent(client, 2).prompt(STREAMING_TOOLS_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
    assert_fixture_is_healthy("malformed_tool_args_matrix/streaming_healthy_control");
}

/// The exact refusal the cut turn ends with: its call never closed, so the
/// turn fails and none of its calls run (G48b, coord-a-rig-g48b-1; G157,
/// coord-a-rig-g138-g184-1).
const UNFINISHED_CALL_REFUSAL: &str = "the turn failed, so none of its tool calls ran: \
     the provider ended the reply with a tool call it never finished";

/// Cell 3. The name is the recorded scenario's; the contract it now pins is
/// the port's: the `max_tokens` cut leaves the `subtract` call's block open
/// (no `content_block_stop`), so the turn fails with the unfinished-call
/// refusal before any tool runs, and the run gives no final response.
///
/// The error-result follow-up the fixture records as interaction 1 is never
/// sent, and a replay session fails on an interaction left unplayed, so the
/// cell serves the recorded first reply itself and checks the request it
/// answers against the recorded first request. The loss: the recording is
/// no longer replayed end to end, and nothing pins its follow-up.
#[tokio::test]
async fn streaming_malformed_call_is_answered_with_an_error() {
    let scenario = "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error";
    assert_fixture_is_cut(scenario);
    let recorded = recorded_interaction_bodies(
        "anthropic",
        "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error",
    );
    let (first_request, first_reply) = recorded
        .first()
        .cloned()
        .expect("the cut reply is recorded");
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        first_reply,
        "text/event-stream",
    )]);
    let client = AnthropicModels::new(AnthropicConfig::new("cassette"), http.clone());
    let mut stream = agent(client, 4)
        .prompt(STREAMING_TOOLS_PROMPT)
        .add_hook(CutFirstCall)
        .stream();
    let mut executed = Vec::new();
    let mut finals = Vec::new();
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolExecutionCommitted { tool_call }) => {
                executed.push(format!("{:?}", tool_call.function.name));
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                finals.push(response.output().to_owned());
            }
            Ok(_) => {}
            Err(error) => errors.push(error),
        }
    }
    drop(stream);
    let [error] = errors.as_slice() else {
        panic!("the stream fails exactly once: {errors:?}");
    };
    assert!(
        matches!(
            error,
            PromptError::Provider(ProviderError::Response(message))
                if message == UNFINISHED_CALL_REFUSAL
        ),
        "the cut turn fails with the unfinished-call refusal: {error:?}"
    );
    assert!(executed.is_empty(), "no tool is executed: {executed:?}");
    assert!(
        finals.is_empty(),
        "the run gives no final response: {finals:?}"
    );

    // The one request sent is the recorded first request.
    let sent = http.requests();
    let [sent] = sent.as_slice() else {
        panic!("only the cut turn is requested: {sent:?}");
    };
    let sent: Value = serde_json::from_slice(&sent.body).expect("the sent body is JSON");
    let first_request: Value =
        serde_json::from_str(&first_request).expect("the recorded request is JSON");
    assert_matches_recorded_document(&sent, &first_request, &[], "the cut turn's request");
}
