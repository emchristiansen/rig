//! DeepSeek live coverage for batch multi-extract pipelines.

use std::future::IntoFuture;

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::providers::deepseek;
use rig_agent::completion::StructuredOutputError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::request_witness::{NamedCase, with_deepseek_concurrent_witness_cassette};

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
/// which the recording lacks, so strict unordered replay refuses every
/// request of the original concurrent batch. The batch stops at its first
/// failed extraction and drops the requests still in flight, so the
/// witness admits the refusals in any order and number: every sent request
/// is exactly one of the nine named changed bodies, and finalization keeps
/// exactly the refusals the engine received. Named losses: the three
/// extracted names, topics and sentiment summaries and their non-empty
/// text.
#[tokio::test]
async fn batch_multi_extract_chain() -> Result<()> {
    with_deepseek_concurrent_witness_cassette(
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

            let inputs = vec![
                "Ada Lovelace discussed analytical engines and early programming.",
                "I love my dog, but I hate rainy weather.",
                "I'm going to the store to buy milk and bread.",
            ];
            let responses: Result<Vec<String>, StructuredOutputError> = futures::stream::iter(
                inputs,
            )
            .map(|text| {
                let names_extractor = &names_extractor;
                let topics_extractor = &topics_extractor;
                let sentiment_extractor = &sentiment_extractor;
                async move {
                    let (names, topics, sentiment) = futures::try_join!(
                        names_extractor.extract(text).into_future(),
                        topics_extractor.extract(text).into_future(),
                        sentiment_extractor.extract(text).into_future(),
                    )?;
                    Ok(format!(
                        "Extracted names: {}\nExtracted topics: {}\nExtracted sentiment: {} ({})",
                        names.output.names.join(", "),
                        topics.output.topics.join(", "),
                        sentiment.output.sentiment,
                        sentiment.output.confidence,
                    ))
                }
            })
            .buffered(4)
            .try_collect()
            .await;

            match responses {
                Ok(_) => Ok(Ok(())),
                Err(StructuredOutputError::Prompt(error)) => Ok(Err(error)),
                Err(other) => anyhow::bail!("the batch failed before its requests: {other:?}"),
            }
        },
    )
    .await
}
