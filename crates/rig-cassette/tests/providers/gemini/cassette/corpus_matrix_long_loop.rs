//! The long tool loop's producer column on Gemini: gemini-2.5-flash.
//! One recording per live cell. Programs, toolset and assertions are
//! `tests/common/corpus_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::request_witness_support::{
    NamedCase, attempt_cell, with_gemini_request_witness_cassette,
};
use crate::corpus_matrix::{Wire, cells};
use rig_test_support::cassette_models::GeminiModels;

// gemini-2.5-flash, not flash-lite: at temperature 0 flash-lite answered the
// `list_files` functionResponse with an empty candidate (no parts,
// finishReason STOP) on both endpoints, 3 attempts, first recording round.
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
        model: client.completion("gemini-2.5-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

/// Requires the named request refusal; the former downstream golden is unverified.
#[tokio::test(flavor = "current_thread")]
async fn long_streamed() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "long_loop_matrix/long_streamed",
        NamedCase::LongStreamed,
        |client| async move {
            let wire = wire(&client);
            attempt_cell(&wire, NamedCase::LongStreamed).await
        },
    )
    .await
}
