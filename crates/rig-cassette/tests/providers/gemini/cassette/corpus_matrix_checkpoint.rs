//! Focused tool-turn checkpoint matrix on Gemini: gemini-2.5-flash-lite.
//! The large-result recording also backs a negative matcher probe.

use super::super::request_witness_support::{
    NamedCase, attempt_cell, with_gemini_request_witness_cassette,
};
use super::super::support::with_gemini_cassette;
use crate::corpus_matrix::{Wire, cells, checkpoint};
use rig_test_support::cassette_models::GeminiModels;

fn wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion("gemini-2.5-flash-lite"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, "gemini_checkpoint_parallel_batch");
}

/// The strict matcher refuses the large result with its last byte changed.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("gemini", "checkpoint_matrix/large_result").await;
}

/// Requires the named request refusal; the former downstream golden is unverified.
#[tokio::test(flavor = "current_thread")]
async fn large_result() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "checkpoint_matrix/large_result",
        NamedCase::LargeResult,
        |client| async move {
            let wire = wire(&client);
            attempt_cell(&wire, NamedCase::LargeResult).await
        },
    )
    .await
}
