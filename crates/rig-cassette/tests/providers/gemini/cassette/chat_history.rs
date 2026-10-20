//! Gemini high-level Chat history regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::{Message, Reasoning};
use rig::providers::gemini;
use rig::providers::gemini::extension::GeminiOptions;

use super::super::request_witness_support::{NamedCase, with_gemini_request_witness_cassette};
use crate::reasoning::{self, WeatherTool};

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. Named losses: the weather tool being called and the answer
/// (`assert_nonstreaming_universal`), and caller-owned history gaining the
/// reasoning tool turns (`assert_chat_history_preserves_reasoning_tool_roundtrip`).
#[tokio::test]
async fn chat_appends_reasoning_tool_turns_to_caller_history() -> anyhow::Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_gemini_request_witness_cassette(
        "chat_history/chat_appends_reasoning_tool_turns_to_caller_history",
        NamedCase::ChatHistoryReasoningTools,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                    .max_tokens(4096)
                    .tool(WeatherTool::new(call_count.clone()))
                    .reasoning(Reasoning::Budget { tokens: 4096 })
                    .provider_option(include_thoughts())
                    .default_max_turns(2)
                    .build();
            let mut chat_history = Vec::<Message>::new();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
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
