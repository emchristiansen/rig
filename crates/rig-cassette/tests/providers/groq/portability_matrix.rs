//! Groq continues histories other wires produced: Anthropic-signed and
//! Gemini-signed reasoning beside a tool exchange.

use super::request_witness::{NamedCase, with_groq_request_witness_cassette};
use crate::history_survival::portability::{self, Cell, Source};
use rig_test_support::cassette_models::OpenAiModels;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAiModels, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "groq",
        model: "openai/gpt-oss-20b",
        params,
        max_tokens: 1024,
        source,
    }
}

/// Exact named snapshot witness (coord-a-rig-port-chat-document-12 (2)): the
/// recording is unchanged; the only request difference is that the ported
/// assistant message 2 sends its text as one text part where the recording
/// holds a string. Replay serves the recorded reply, the answer draws on the
/// ported weather (`assert_run`), the recorded request keeps the survival rule
/// (`assert_recorded`), and the request snapshot check refuses exactly that
/// change. The kept predicates run on the replies recorded for the string form,
/// which shape-matched replay serves to the text-part request.
#[tokio::test]
async fn from_anthropic() -> anyhow::Result<()> {
    let cell = cell(Source::Anthropic);
    with_groq_request_witness_cassette(
        "portability_matrix/from_anthropic",
        NamedCase::PortabilityFromAnthropic,
        |client| async move {
            let observation = portability::run(model(client, cell), cell).await;
            portability::assert_run(cell, &observation);
            Ok(())
        },
    )
    .await?;
    portability::assert_recorded(cell, "portability_matrix/from_anthropic");
    Ok(())
}
