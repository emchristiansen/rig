//! Groq history survival: tool-call ids across three prompts on an
//! OpenAI-compatible wire, with parsed reasoning delivered and replayed.
//!
//! Both cells are the live regression for the `reasoning_content` replay
//! that Groq rejected with HTTP 400 before the fix. The parsed reasoning
//! reaches the normalized history and returns on the next request under
//! `reasoning` (coord-a-rig-port-chat-document-12 (2)); `reasoning_content`
//! is not sent.

use super::request_witness::{NamedCase, with_groq_request_witness_cassette};
use crate::history_survival::driver::{self, Cell, Expect, Transport};
use rig_test_support::cassette_models::OpenAiModels;

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "reasoning_format": "parsed" }))
}

fn model(client: OpenAiModels, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "groq",
        model: "openai/gpt-oss-20b",
        params,
        options: crate::history_survival::Options::none,
        max_tokens: 4096,
        transport,
        expect,
    }
}

/// The three-prompt task, its delivered content and the recorded survival
/// rule, all on the recorded replies.
async fn run_cell(client: OpenAiModels, cell: Cell) -> anyhow::Result<()> {
    let observation = driver::run(model(client, cell), cell).await;
    driver::assert_run(cell, &observation);
    Ok(())
}

/// Exact named snapshot witness (coord-a-rig-port-chat-document-12 (2)): the
/// recording is unchanged; the only request difference is that the replayed
/// assistant answers, message 12 of interactions 6 to 8 and message 16 of
/// interaction 8, are each sent as one text part where the recording holds a
/// string. Every replayed `reasoning` is sent as recorded. Replay serves the
/// recorded replies, the task, its delivered content (`assert_run`) and the
/// recorded survival rule (`assert_recorded`) hold, and the request snapshot
/// check refuses exactly those four changes. The kept predicates run on the
/// replies recorded for the string form, which shape-matched replay serves to
/// the text-part request.
#[tokio::test]
async fn unary() -> anyhow::Result<()> {
    let cell = cell(Transport::Unary, Expect::REASONING);
    with_groq_request_witness_cassette(
        "history_survival_matrix/unary",
        NamedCase::HistorySurvivalUnary,
        |client| run_cell(client, cell),
    )
    .await?;
    driver::assert_recorded(cell, "history_survival_matrix/unary");
    Ok(())
}

/// Exact named snapshot witness (coord-a-rig-port-chat-document-12 (2)), as
/// [`unary`] on the streamed transport: message 12 of interactions 6 to 8 and
/// message 16 of interaction 8 are each sent as one text part, every replayed
/// `reasoning` as recorded, and the request snapshot check refuses exactly
/// those four changes. The kept predicates run on the replies recorded for the
/// string form, which shape-matched replay serves to the text-part request.
#[tokio::test]
async fn streaming() -> anyhow::Result<()> {
    let cell = cell(Transport::Streaming, Expect::REASONING);
    with_groq_request_witness_cassette(
        "history_survival_matrix/streaming",
        NamedCase::HistorySurvivalStreaming,
        |client| run_cell(client, cell),
    )
    .await?;
    driver::assert_recorded(cell, "history_survival_matrix/streaming");
    Ok(())
}
