//! Invalid tool-call recovery on [`AgentRun`] exercised against real Gemini
//! turns: the model's `add` call is recorded normally on the wire, while the
//! machine is fed restricted allowed-tool sets to trigger each recovery path
//! (fail, repair, skip, retry-budget exhaustion, bad repair).
//!
//! The recordings send each declaration as `parametersJsonSchema`, which the
//! port's projection sends as `parameters`, so in replay each cell is an
//! exact named request-mismatch witness for the run's first model call, and
//! the recovery it pinned is a named loss.

use rig::agent::run::{AgentRun, AgentRunStep};
use rig::message::ToolChoice;
use rig::providers::gemini;

use super::super::agent_run_support::{FORCE_TOOLS_PREAMBLE, GeminiAgent};
use super::super::request_witness_support::{
    Attempt, NamedCase, model_result, with_gemini_request_witness_cassette,
};

/// Send `run`'s first model call, which a fresh run starts with.
async fn first_model_call(agent: &GeminiAgent, run: &mut AgentRun) -> Attempt {
    let AgentRunStep::CallModel {
        prompt, history, ..
    } = run.next_step()?
    else {
        anyhow::bail!("a fresh run starts with a model call");
    };
    Ok(model_result(
        agent.model.call(agent.request(prompt, history)).await,
    ))
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0 (`add`) and 1 (`sum`) send their
/// recorded `parametersJsonSchema`, value unchanged, as `parameters`, and
/// replay refuses the first model call. Named losses: the `add` call being
/// repaired to `sum` (with exactly the repaired turns suppressing the
/// response hook), `sum` reaching the tool driver, the answer mentioning 5,
/// and history recording `sum` but never `add`.
#[tokio::test]
async fn repair_renames_tool_call_and_executes_it() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "agent_run_recovery/repair_renames_tool_call_and_executes_it",
        NamedCase::RepairRenamesToolCall,
        |client, _clock| async move {
            // `sum` is registered alongside `add` so the post-repair wire
            // history references a tool Gemini saw advertised.
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add", "sum"],
                None,
            );
            let mut run =
                AgentRun::new("Use the add tool to compute 2 + 3, then state the result.")
                    .max_turns(3);
            first_model_call(&agent, &mut run).await
        },
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0 (`add`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses the first model call. Named losses: the `add` call needing
/// resolution under the `subtract`-only policy, and repairing it to the
/// disallowed `multiply` failing with `UnknownToolCall` naming `multiply`
/// and allowing only `subtract`, with portable diagnostics.
#[tokio::test]
async fn repair_to_disallowed_name_fails_with_unknown_tool_call() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "agent_run_recovery/repair_to_disallowed_name_fails_with_unknown_tool_call",
        NamedCase::RepairToDisallowedName,
        |client, _clock| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add"],
                Some(ToolChoice::Required),
            );
            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .max_invalid_tool_call_retries(0);
            first_model_call(&agent, &mut run).await
        },
    )
    .await
}
