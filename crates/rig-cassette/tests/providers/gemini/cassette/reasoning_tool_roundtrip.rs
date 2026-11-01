//! Gemini reasoning tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::{Message, Reasoning};
use rig::providers::gemini::extension::GeminiOptions;

use super::super::request_witness_support::{NamedCase, with_gemini_request_witness_cassette};
use crate::reasoning::{self, WeatherTool};

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. Named loss: the reasoning tool round trip itself, that is the weather
/// tool being called and the answer (`assert_nonstreaming_universal`).
#[tokio::test]
async fn nonstreaming() -> anyhow::Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_gemini_request_witness_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        NamedCase::ReasoningToolRoundtrip,
        |client, _clock| async move {
            let agent = rig::AgentBuilder::new(client.completion("gemini-2.5-flash"))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .reasoning(Reasoning::Budget { tokens: 4096 })
                .provider_option(include_thoughts())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .map(|_| ());
            drop(agent);
            Ok(result)
        },
    )
    .await
}

/// `includeThoughts`, beside the typed thinking budget.
fn include_thoughts() -> GeminiOptions {
    GeminiOptions::new().include_thoughts(true)
}
