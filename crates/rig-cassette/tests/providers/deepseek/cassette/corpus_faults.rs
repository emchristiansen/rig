//! The original DeepSeek failed-tool continuation is refused for its single
//! zero call index. Its completed golden trace is unavailable.

use crate::corpus_matrix::{Wire, faults};
use crate::deepseek::request_mismatch::{NamedCase, with_deepseek_request_mismatch};
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn tool_error_streamed() -> anyhow::Result<()> {
    with_deepseek_request_mismatch(
        "corpus_faults/tool_error_streamed",
        NamedCase::ToolErrorStreamed,
        wire,
        &faults::TOOL_ERROR_STREAMED,
    )
    .await
}
