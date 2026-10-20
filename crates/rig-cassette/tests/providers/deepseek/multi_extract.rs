//! DeepSeek live coverage for batch multi-extract pipelines.

use anyhow::Result;
use rig::providers::deepseek;
use rig_agent::completion::StructuredOutputError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::request_witness::{NamedCase, with_deepseek_request_witness_cassette};
use crate::chat_request_witness::Outcome;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Names {
    names: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Topics {
    topics: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Sentiment {
    sentiment: f64,
    confidence: f64,
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 6): the
/// recording is unchanged; the only request difference is that every
/// extraction sends its forced `submit` choice as `"tool_choice": null`,
/// which the recording lacks, so strict replay refuses all nine recorded
/// extractions, each on its three attempts (`retries(2)`). The witness runs
/// the nine extractions one at a time rather than in the original
/// concurrent batch, whose first refusal drops its siblings mid-flight.
/// Named losses: concurrent batch execution itself, and the three extracted
/// names, topics and sentiment summaries and their non-empty text.
#[tokio::test]
async fn batch_multi_extract_chain() -> Result<()> {
    with_deepseek_request_witness_cassette(
        "multi_extract/batch_multi_extract_chain",
        NamedCase::BatchMultiExtract,
        |client| async move {
            let names_extractor = rig::extractor::ExtractorBuilder::<Names>::new(
                client.completion(deepseek::DEEPSEEK_V4_FLASH),
            )
            .append_preamble("Extract names from the given text.")
            .retries(2)
            .build();
            let topics_extractor = rig::extractor::ExtractorBuilder::<Topics>::new(
                client.completion(deepseek::DEEPSEEK_V4_FLASH),
            )
            .append_preamble("Extract topics from the given text.")
            .retries(2)
            .build();
            let sentiment_extractor = rig::extractor::ExtractorBuilder::<Sentiment>::new(
                client.completion(deepseek::DEEPSEEK_V4_FLASH),
            )
            .append_preamble("Extract sentiment and confidence from the given text.")
            .retries(2)
            .build();

            let inputs = [
                "Ada Lovelace discussed analytical engines and early programming.",
                "I love my dog, but I hate rainy weather.",
                "I'm going to the store to buy milk and bread.",
            ];
            let mut outcomes = Vec::new();
            for text in inputs {
                outcomes.push(refusal(names_extractor.extract(text).await.map(|_| ()))?);
                outcomes.push(refusal(topics_extractor.extract(text).await.map(|_| ()))?);
                outcomes.push(refusal(
                    sentiment_extractor.extract(text).await.map(|_| ()),
                )?);
            }
            Ok(outcomes)
        },
    )
    .await
}

/// An extraction's outcome as its operation's: a failed attempt is the
/// prompt error it carries.
fn refusal(result: Result<(), StructuredOutputError>) -> Result<Outcome> {
    match result {
        Ok(()) => Ok(Ok(())),
        Err(StructuredOutputError::Prompt(error)) => Ok(Err(error)),
        Err(other) => anyhow::bail!("the extraction failed before its request: {other:?}"),
    }
}
