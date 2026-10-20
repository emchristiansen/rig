//! Streamed-turn coverage for [`AgentRun`]: hand-driving
//! [`StreamedTurnAssembler`] over real Gemini SSE streams, mid-stream invalid
//! tool-call recovery, per-call usage recording, and the built-in streaming
//! driver divergences pinned by #1899.
//!
//! The recordings send each declaration as `parametersJsonSchema`, which the
//! port's projection sends as `parameters`, so in replay each cell is an
//! exact named request-mismatch witness for its first model call, and the
//! streamed behaviour it pinned is a named loss.
//!
//! [`StreamedTurnAssembler`]: rig::agent::run::StreamedTurnAssembler

use rig::agent::run::{AgentRun, AgentRunStep};
use rig::agent::{AgentHook, DispatchAction, DispatchEvent};
use rig::message::ToolChoice;
use rig::providers::gemini;

use super::super::agent_run_support::{Add, FORCE_TOOLS_PREAMBLE, GeminiAgent};
use super::super::request_witness_support::{
    Attempt, NamedCase, model_stream_result, stream_result, with_gemini_request_witness_cassette,
};

/// Stream `run`'s first model call, which a fresh run starts with.
async fn first_streamed_model_call(agent: &GeminiAgent, run: &mut AgentRun) -> Attempt {
    let AgentRunStep::CallModel {
        prompt, history, ..
    } = run.next_step()?
    else {
        anyhow::bail!("a fresh run starts with a model call");
    };
    model_stream_result(agent.model.stream(agent.request(prompt, history))).await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0 (`add`) and 1 (`sum`) send their
/// recorded `parametersJsonSchema`, value unchanged, as `parameters`, and
/// replay refuses the first streamed model call. Named losses: the streamed
/// `add` call being repaired to `sum` inside the same stream, `sum` reaching
/// the tool driver, the answer mentioning 5, and history never recording
/// `add`.
#[tokio::test]
async fn streamed_repair_continues_the_same_stream() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "agent_run_streamed/streamed_repair_continues_the_same_stream",
        NamedCase::StreamedRepair,
        |client, _clock| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add", "sum"],
                None,
            );
            let mut run =
                AgentRun::new("Use the add tool to compute 2 + 3, then state the result.")
                    .max_turns(3);
            first_streamed_model_call(&agent, &mut run).await
        },
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0 (`add`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses the first streamed model call. Named losses: the restricted first
/// turn being abandoned by the skip with a synthetic tool result that
/// answers a local call handle with no provider id, the retry carrying the
/// abandoned assistant turn and the synthetic tool results, a non-empty
/// answer, and at least two recorded completion calls.
#[tokio::test]
async fn streamed_skip_abandons_the_turn_and_recovers() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "agent_run_streamed/streamed_skip_abandons_the_turn_and_recovers",
        NamedCase::StreamedSkip,
        |client, _clock| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add"],
                None,
            );
            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.").max_turns(3);
            first_streamed_model_call(&agent, &mut run).await
        },
    )
    .await
}

#[derive(Clone)]
struct CancelOnToolCall;

impl AgentHook for CancelOnToolCall {
    async fn on_dispatch(
        &self,
        _ctx: &rig::agent::HookContext,
        event: DispatchEvent<'_>,
    ) -> DispatchAction {
        if event.tool_name().is_none() {
            return DispatchAction::proceed();
        }
        DispatchAction::stop("cancelled by test hook")
    }
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0 (`add`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0 before any tool call reaches the hook. Named
/// losses: the hook cancelling the run without a final response, the
/// `Cancelled` error carrying the hook's reason with portable diagnostics,
/// and its history including the recorded assistant `add` turn (#1899).
#[tokio::test]
async fn builtin_streaming_cancellation_history_includes_assistant_turn() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn",
        NamedCase::StreamingCancellationHistory,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(FORCE_TOOLS_PREAMBLE)
                    .tool(Add)
                    .tool_choice(ToolChoice::Required)
                    .build();

            let mut stream = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .add_hook(CancelOnToolCall)
                .max_turns(2)
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            drop(agent);
            result
        },
    )
    .await
}
