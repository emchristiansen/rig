//! DeepSeek reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::completion::Message;
use rig::providers::deepseek;

use super::request_witness::{NamedCase, with_deepseek_request_witness_cassette};
use crate::reasoning::{self, WeatherTool};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 6): the
/// recording is unchanged; the only request difference is that the replayed
/// `get_weather` call of assistant message 2 carries `"index": 0`, and
/// strict replay refuses interaction 1, the continuation. Before it, the
/// weather tool runs once. Named loss: the answer drawing on the tool's
/// report (the rest of `assert_nonstreaming_universal`).
#[tokio::test]
async fn nonstreaming() -> anyhow::Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let calls = call_count.clone();
    with_deepseek_request_witness_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        NamedCase::ReasoningToolRoundtrip,
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(deepseek::DEEPSEEK_V4_FLASH))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count))
                .additional_params(thinking_params())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .map(|_| ());
            drop(agent);
            Ok(vec![result])
        },
    )
    .await?;
    let calls = calls.load(Ordering::SeqCst);
    anyhow::ensure!(
        calls == 1,
        "the weather tool runs once before the refused continuation: {calls}"
    );
    Ok(())
}
