//! Exact request refusals for the named DeepSeek recordings in [`NamedCase`]
//! outside the corpus matrix (whose seven are in `request_mismatch.rs`).
//!
//! mec-rig-132-dispositions-1 item 6 keeps the established
//! `finalize_deepseek` branches: an assistant turn with no text sends
//! `"content": ""`, every replayed call, parallel calls included, carries
//! `"index": 0`, and a forced `tool_choice` is sent as `null`. Each case
//! names exactly the ones its refused requests show; any other difference
//! fails. The real replay transport retains each refusal; these witnesses do
//! not establish what the programs would have done after it.

use std::future::Future;

use anyhow::Result;
use rig::providers::openai::wire::DEEPSEEK;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::chat_request_witness::{
    Change, Expected, Order, Outcome, Recording, Replay, with_refusal_witness,
};

const PARALLEL_TOOL_CALLS: &str =
    "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming";
const SEQUENTIAL_COMPLEX_TOOL_CALLS: &str =
    "agent_tool_sessions/sequential_complex_tool_calls_streaming";
const BATCH_MULTI_EXTRACT: &str = "multi_extract/batch_multi_extract_chain";
const PORTABILITY_FROM_OPENAI_RESPONSES: &str = "portability_matrix/from_openai_responses";
const REASONING_TOOL_ROUNDTRIP: &str = "reasoning_tool_roundtrip/nonstreaming";

/// The complete set of admitted recorded DeepSeek request differences
/// outside the corpus matrix.
#[derive(Clone, Copy, Debug)]
pub(super) enum NamedCase {
    ParallelToolCallsUnary,
    SequentialComplexToolCallsStreamed,
    BatchMultiExtract,
    PortabilityFromOpenAiResponses,
    ReasoningToolRoundtrip,
}

/// The `finalize_deepseek` branch one refused request shows.
#[derive(Clone, Copy, Debug)]
enum Difference {
    /// Every call of assistant message `message`, `calls` of them recorded
    /// without an `index`, is sent with `"index": 0`.
    ZeroCallIndexes { message: usize, calls: usize },
    /// Assistant message `message`, recorded with calls but no `content`, is
    /// sent with `"content": ""`, and its `calls` calls with `"index": 0`.
    EmptyContentZeroCallIndexes { message: usize, calls: usize },
    /// The request, recorded without a `tool_choice`, sends the forced
    /// choice as `"tool_choice": null`.
    NullToolChoice,
}

impl Difference {
    fn changes(self) -> Vec<Change> {
        let indexes = |message: usize, calls: usize| {
            (0..calls).map(move |call| Change::Added {
                pointer: format!("/messages/{message}/tool_calls/{call}/index"),
                sent: json!(0),
            })
        };
        match self {
            Self::ZeroCallIndexes { message, calls } => indexes(message, calls).collect(),
            Self::EmptyContentZeroCallIndexes { message, calls } => {
                std::iter::once(Change::Added {
                    pointer: format!("/messages/{message}/content"),
                    sent: json!(""),
                })
                .chain(indexes(message, calls))
                .collect()
            }
            Self::NullToolChoice => vec![Change::Added {
                pointer: "/tool_choice".to_owned(),
                sent: Value::Null,
            }],
        }
    }
}

/// The batch's nine recorded interactions in the witness's extraction
/// order: each input in turn through the names, topics and sentiment
/// extractors.
const BATCH_EXTRACTION_ORDER: [usize; 9] = [0, 7, 5, 4, 1, 8, 6, 3, 2];

/// Each extraction's retry budget (`ExtractorBuilder::retries(2)`): every
/// refused extraction is sent three times.
const BATCH_EXTRACTION_RETRIES: usize = 2;

impl NamedCase {
    fn scenario(self) -> &'static str {
        match self {
            Self::ParallelToolCallsUnary => PARALLEL_TOOL_CALLS,
            Self::SequentialComplexToolCallsStreamed => SEQUENTIAL_COMPLEX_TOOL_CALLS,
            Self::BatchMultiExtract => BATCH_MULTI_EXTRACT,
            Self::PortabilityFromOpenAiResponses => PORTABILITY_FROM_OPENAI_RESPONSES,
            Self::ReasoningToolRoundtrip => REASONING_TOOL_ROUNDTRIP,
        }
    }

    /// The recorded bodies. Each is read through its literal coordinate,
    /// which is also how the cassette registry counts the fixture.
    fn recorded(self) -> Vec<(String, String)> {
        match self {
            Self::ParallelToolCallsUnary => {
                crate::cassettes::recorded_interaction_bodies("deepseek", PARALLEL_TOOL_CALLS)
            }
            Self::SequentialComplexToolCallsStreamed => {
                crate::cassettes::recorded_interaction_bodies(
                    "deepseek",
                    SEQUENTIAL_COMPLEX_TOOL_CALLS,
                )
            }
            Self::BatchMultiExtract => {
                crate::cassettes::recorded_interaction_bodies("deepseek", BATCH_MULTI_EXTRACT)
            }
            Self::PortabilityFromOpenAiResponses => crate::cassettes::recorded_interaction_bodies(
                "deepseek",
                PORTABILITY_FROM_OPENAI_RESPONSES,
            ),
            Self::ReasoningToolRoundtrip => {
                crate::cassettes::recorded_interaction_bodies("deepseek", REASONING_TOOL_ROUNDTRIP)
            }
        }
    }

    fn recorded_interactions(self) -> usize {
        match self {
            Self::PortabilityFromOpenAiResponses => 1,
            Self::ParallelToolCallsUnary | Self::ReasoningToolRoundtrip => 2,
            Self::SequentialComplexToolCallsStreamed => 5,
            Self::BatchMultiExtract => 9,
        }
    }

    fn order(self) -> Order {
        match self {
            Self::BatchMultiExtract => Order::Any,
            _ => Order::Recorded,
        }
    }

    /// Every request the named program sends, in send order.
    fn expected(self) -> Vec<Expected> {
        let refused = |interaction: usize, difference: Difference| {
            (0..interaction).map(Expected::served).chain([Expected {
                interaction,
                replay: Replay::Refused,
                changes: difference.changes(),
            }])
        };
        match self {
            Self::ParallelToolCallsUnary => refused(
                1,
                Difference::ZeroCallIndexes {
                    message: 2,
                    calls: 2,
                },
            )
            .collect(),
            Self::SequentialComplexToolCallsStreamed | Self::ReasoningToolRoundtrip => refused(
                1,
                Difference::ZeroCallIndexes {
                    message: 2,
                    calls: 1,
                },
            )
            .collect(),
            Self::PortabilityFromOpenAiResponses => refused(
                0,
                Difference::EmptyContentZeroCallIndexes {
                    message: 2,
                    calls: 1,
                },
            )
            .collect(),
            Self::BatchMultiExtract => BATCH_EXTRACTION_ORDER
                .iter()
                .flat_map(|&interaction| {
                    let attempt = move |replay| Expected {
                        interaction,
                        replay,
                        changes: Difference::NullToolChoice.changes(),
                    };
                    std::iter::repeat_n(
                        attempt(Replay::RefusedThenRetried),
                        BATCH_EXTRACTION_RETRIES,
                    )
                    .chain([attempt(Replay::Refused)])
                })
                .collect(),
        }
    }
}

/// Bind one literal scenario to its real strict request refusals. `body`
/// gets the session's models and returns each operation's outcome in order.
/// Setup failures and panics remain failures after the session is finalized.
pub(super) async fn with_deepseek_request_witness_cassette<F, Fut>(
    scenario: &'static str,
    case: NamedCase,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<Vec<Outcome>>>,
{
    anyhow::ensure!(
        scenario == case.scenario(),
        "the named scenario changed: {scenario}, {case:?}"
    );
    let recording = Recording::new("deepseek", scenario, &DEEPSEEK, case.recorded())?;
    anyhow::ensure!(
        recording.len() == case.recorded_interactions(),
        "the original recorded interaction count changed: {}",
        recording.len()
    );
    with_refusal_witness(recording, case.order(), case.expected(), body).await
}
