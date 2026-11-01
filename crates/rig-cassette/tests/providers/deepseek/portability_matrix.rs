//! DeepSeek continues histories other wires produced: Anthropic-signed,
//! OpenAI-encrypted and Gemini-signed reasoning beside a tool exchange.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::providers::deepseek;
use rig_test_support::cassette_models::OpenAiModels;

use super::request_witness::{NamedCase, with_deepseek_request_witness_cassette};
use crate::history_survival::portability::{self, Cell, Source};
use crate::reasoning::{TOOL_SYSTEM_PROMPT, WeatherTool};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAiModels, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "deepseek",
        model: deepseek::DEEPSEEK_V4_FLASH,
        params,
        max_tokens: 512,
        source,
    }
}

/// The ported continuation `portability::run` sends, returning its outcome
/// instead of panicking on a refusal.
async fn continue_ported(
    client: OpenAiModels,
    cell: Cell,
) -> anyhow::Result<Vec<crate::chat_request_witness::Outcome>> {
    let mut builder = rig::AgentBuilder::new(model(client, cell))
        .preamble(TOOL_SYSTEM_PROMPT)
        .max_tokens(cell.max_tokens)
        .default_max_turns(4);
    if let Some(params) = (cell.params)() {
        builder = builder.additional_params(params);
    }
    let agent = builder
        .tool(WeatherTool::new(Arc::new(AtomicUsize::new(0))))
        .build();
    let mut history = cell.source.history();
    let result = agent
        .chat(portability::FOLLOW_UP, &mut history)
        .await
        .map(|_| ());
    drop(agent);
    Ok(vec![result])
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 6): the
/// recording is unchanged; the only request difference is that the ported
/// tool-call-only assistant message 2 carries `"content": ""` and its call
/// `"index": 0`, and strict replay refuses interaction 0, the
/// continuation. The recorded request's survival rule (`assert_recorded`)
/// still holds. Named loss: the target's answer drawing on the ported
/// weather (`assert_run`).
#[tokio::test]
async fn from_openai_responses() -> anyhow::Result<()> {
    let cell = cell(Source::OpenAiResponses);
    with_deepseek_request_witness_cassette(
        "portability_matrix/from_openai_responses",
        NamedCase::PortabilityFromOpenAiResponses,
        |client| continue_ported(client, cell),
    )
    .await?;
    portability::assert_recorded(cell, "portability_matrix/from_openai_responses");
    Ok(())
}
