//! Matrix C of the effect corpus, the Gemini cell: two tool-call turns on
//! an id-less wire under serial serving. The `hook_stress` cassette the
//! `gemini_tool_call_turns` golden records from serves this program too:
//! the policy changes how the bus serves, not what the program asks.

use rig::providers::gemini;

use super::super::hook_stress_support::CHAIN_PREAMBLE;
use super::super::request_witness_support::{
    NamedCase, stream_result, with_gemini_request_witness_cassette,
};
use super::super::tools_support::{CountingAdd, CountingSubtract};

/// Requires the named request refusal; the former downstream golden is unverified.
#[tokio::test(flavor = "current_thread")]
async fn two_turns_serial_effect_log_is_the_golden_fixture() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        NamedCase::TwoTurnsSerial,
        |client| async move {
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .name("stress-agent")
                    .configure_bus(rig::serve::ServingPolicy {
                        serial_per_handler: true,
                        ..rig::serve::ServingPolicy::default()
                    })
                    .preamble(CHAIN_PREAMBLE)
                    .temperature(0.0)
                    .tool(CountingAdd::default())
                    .tool(CountingSubtract::default())
                    .record_to(recorder.clone())
                    .build();
            let mut stream = agent
                .prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            drop(agent);
            result
        },
    )
    .await
}
