//! Exact request refusals for the named DeepSeek recordings in [`NamedCase`]
//! outside the corpus matrix (whose seven are in `request_mismatch.rs`).
//!
//! mec-rig-132-dispositions-1 item 6 keeps the established
//! `finalize_deepseek` branches: an assistant turn with no text sends
//! `"content": ""`, every replayed call, parallel calls included, carries
//! `"index": 0`, a forced `tool_choice` is sent as `null`, and the text parts
//! of a message other than the system message are sent joined into one
//! string. Each case names exactly the ones its requests show; any other
//! difference fails. A refused request's witness keeps the real replay
//! refusal and does not establish what the program would have done after
//! it; the flattened text keeps the request's coarse shape, so its witness
//! replays the whole program and the request snapshot check refuses exactly
//! that change.

use std::future::Future;

use anyhow::{Context, Result, bail, ensure};
use rig::providers::openai::wire::DEEPSEEK;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::chat_request_witness::{
    Change, Expected, Outcome, Recording, Replay, with_concurrent_refusal_witness,
    with_refusal_witness, with_snapshot_witness,
};

const PARALLEL_TOOL_CALLS: &str =
    "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming";
const SEQUENTIAL_COMPLEX_TOOL_CALLS: &str =
    "agent_tool_sessions/sequential_complex_tool_calls_streaming";
const BATCH_MULTI_EXTRACT: &str = "multi_extract/batch_multi_extract_chain";
const PORTABILITY_FROM_OPENAI_RESPONSES: &str = "portability_matrix/from_openai_responses";
const REASONING_TOOL_ROUNDTRIP: &str = "reasoning_tool_roundtrip/nonstreaming";
const RAW_FOLLOWUP: &str = "streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls";

/// The complete set of admitted recorded DeepSeek request differences
/// outside the corpus matrix.
#[derive(Clone, Copy, Debug)]
pub(super) enum NamedCase {
    ParallelToolCallsUnary,
    SequentialComplexToolCallsStreamed,
    BatchMultiExtract,
    PortabilityFromOpenAiResponses,
    ReasoningToolRoundtrip,
    RawFollowup,
}

/// How a case's recording meets the port's requests.
enum Witness {
    /// Strict replay refuses the stated requests, sent in this order.
    Refusal(Vec<Expected>),
    /// Strict unordered replay refuses every one of these named changed
    /// requests a concurrent program sends.
    Concurrent(Vec<(usize, Vec<Change>)>),
    /// Shape-matched replay serves every request, and the request snapshot
    /// check refuses exactly these changes.
    Snapshot(Vec<(usize, Change)>),
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

impl NamedCase {
    fn scenario(self) -> &'static str {
        match self {
            Self::ParallelToolCallsUnary => PARALLEL_TOOL_CALLS,
            Self::SequentialComplexToolCallsStreamed => SEQUENTIAL_COMPLEX_TOOL_CALLS,
            Self::BatchMultiExtract => BATCH_MULTI_EXTRACT,
            Self::PortabilityFromOpenAiResponses => PORTABILITY_FROM_OPENAI_RESPONSES,
            Self::ReasoningToolRoundtrip => REASONING_TOOL_ROUNDTRIP,
            Self::RawFollowup => RAW_FOLLOWUP,
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
            Self::RawFollowup => {
                crate::cassettes::recorded_interaction_bodies("deepseek", RAW_FOLLOWUP)
            }
        }
    }

    fn recorded_interactions(self) -> usize {
        match self {
            Self::PortabilityFromOpenAiResponses => 1,
            Self::ParallelToolCallsUnary | Self::ReasoningToolRoundtrip | Self::RawFollowup => 2,
            Self::SequentialComplexToolCallsStreamed => 5,
            Self::BatchMultiExtract => 9,
        }
    }

    /// How the case's recording meets the port's requests.
    fn witness(self, recording: &Recording) -> Result<Witness> {
        let refused = |interaction: usize, difference: Difference| {
            Witness::Refusal(
                (0..interaction)
                    .map(Expected::served)
                    .chain([Expected {
                        interaction,
                        replay: Replay::Refused,
                        changes: difference.changes(),
                    }])
                    .collect(),
            )
        };
        Ok(match self {
            Self::ParallelToolCallsUnary => refused(
                1,
                Difference::ZeroCallIndexes {
                    message: 2,
                    calls: 2,
                },
            ),
            Self::SequentialComplexToolCallsStreamed | Self::ReasoningToolRoundtrip => refused(
                1,
                Difference::ZeroCallIndexes {
                    message: 2,
                    calls: 1,
                },
            ),
            Self::PortabilityFromOpenAiResponses => refused(
                0,
                Difference::EmptyContentZeroCallIndexes {
                    message: 2,
                    calls: 1,
                },
            ),
            Self::BatchMultiExtract => Witness::Concurrent(
                (0..recording.len())
                    .map(|interaction| (interaction, Difference::NullToolChoice.changes()))
                    .collect(),
            ),
            Self::RawFollowup => {
                Witness::Snapshot(vec![(1, flattened_user_text(recording, 1, 2)?)])
            }
        })
    }
}

/// Message `message` of request `interaction`, a user message recorded as
/// text parts, sent as their texts joined with `\n`.
fn flattened_user_text(
    recording: &Recording,
    interaction: usize,
    message: usize,
) -> Result<Change> {
    let pointer = format!("/messages/{message}/content");
    let recorded = recording.body(interaction)?;
    ensure!(
        recorded.pointer(&format!("/messages/{message}/role")) == Some(&json!("user")),
        "the named message is not a user turn: {recorded}"
    );
    let parts = recorded
        .pointer(&pointer)
        .context("the named recorded content exists")?;
    let texts = parts
        .as_array()
        .filter(|parts| parts.len() > 1)
        .context("the named recorded content is several parts")?
        .iter()
        .map(|part| {
            let text = part.get("text").and_then(Value::as_str);
            match (
                part.as_object().map(serde_json::Map::len),
                part.get("type"),
                text,
            ) {
                (Some(2), Some(kind), Some(text)) if *kind == json!("text") => Ok(text),
                _ => bail!("a named recorded part is not a text part: {part}"),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Change::Replaced {
        pointer,
        recorded: parts.clone(),
        sent: json!(texts.join("\n")),
    })
}

/// Start one named case's recording.
fn recording(scenario: &'static str, case: NamedCase) -> Result<Recording> {
    ensure!(
        scenario == case.scenario(),
        "the named scenario changed: {scenario}, {case:?}"
    );
    let recording = Recording::new("deepseek", scenario, &DEEPSEEK, case.recorded())?;
    ensure!(
        recording.len() == case.recorded_interactions(),
        "the original recorded interaction count changed: {}",
        recording.len()
    );
    Ok(recording)
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
    let recording = recording(scenario, case)?;
    let Witness::Refusal(expected) = case.witness(&recording)? else {
        bail!("{case:?} is not a sequential refusal");
    };
    with_refusal_witness(recording, expected, body).await
}

/// Bind one literal scenario to the real strict refusals of a concurrent
/// program. `body` gets the session's models and returns the program's
/// outcome. Setup failures and panics remain failures after the session is
/// finalized.
pub(super) async fn with_deepseek_concurrent_witness_cassette<F, Fut>(
    scenario: &'static str,
    case: NamedCase,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<Outcome>>,
{
    let recording = recording(scenario, case)?;
    let Witness::Concurrent(named) = case.witness(&recording)? else {
        bail!("{case:?} is not a concurrent refusal");
    };
    with_concurrent_refusal_witness(recording, named, body).await
}

/// Bind one literal scenario to the request snapshot's exact refusal of its
/// named change. `body` gets the session's models and runs the test's own
/// predicates. Setup failures and panics remain failures after the session
/// is finalized.
pub(super) async fn with_deepseek_snapshot_witness_cassette<F, Fut>(
    scenario: &'static str,
    case: NamedCase,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let recording = recording(scenario, case)?;
    let Witness::Snapshot(changes) = case.witness(&recording)? else {
        bail!("{case:?} is not a snapshot refusal");
    };
    with_snapshot_witness(recording, changes, body).await
}
