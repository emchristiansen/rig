//! Gemini tool-choice cassette coverage.

use rig::message::ToolChoice;
use rig::providers::gemini;

use super::super::request_witness_support::{
    NamedCase, stream_result, with_gemini_request_witness_cassette,
};
use crate::support::{Adder, Subtract};

/// `ToolChoice::None` while tools are registered.
///
/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0 and 1 send their recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0. Named losses: a stream without errors ending in a
/// final response, no tool calls or tool results, and the answer mentioning
/// 42.
#[tokio::test]
async fn none_streaming_does_not_emit_tool_calls() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "tool_choice/none_streaming_no_tools",
        NamedCase::ToolChoiceNoneStreamed,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble("You are a deterministic calculator test. Answer directly in text.")
                    .temperature(0.0)
                    .tool(Adder)
                    .tool(Subtract)
                    .tool_choice(ToolChoice::None)
                    .build();

            let mut stream = agent
                .prompt("Calculate 20 + 22 directly in text. Do not call tools.")
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            drop(agent);
            result
        },
    )
    .await
}
