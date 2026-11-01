//! The original DeepSeek checkpoint batch is refused for empty assistant
//! content and three zero call indexes. Its completed golden trace is unavailable.

use crate::corpus_matrix::{Wire, cells, checkpoint};
use crate::deepseek::request_mismatch::{NamedCase, with_deepseek_request_mismatch};
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: Some(|| {
            crate::corpus_matrix::corpus::TypedOptions::generation(
                rig::completion::GenerationOptions::default()
                    .reasoning(rig::completion::Reasoning::Off),
            )
        }),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn parallel_batch() -> anyhow::Result<()> {
    with_deepseek_request_mismatch(
        "checkpoint_matrix/parallel_batch",
        NamedCase::ParallelBatch,
        wire,
        &checkpoint::PARALLEL_BATCH,
    )
    .await
}
