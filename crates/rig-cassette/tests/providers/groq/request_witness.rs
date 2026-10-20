//! Exact request-snapshot refusals for the named Groq recordings in
//! [`NamedCase`].
//!
//! coord-a-rig-port-chat-document-12 (2): the Groq dialect sends an
//! assistant turn's content as an array of text parts, `[{"type": "text",
//! "text": ...}]`, where these recordings hold a string, and `reasoning`
//! replays. Each case names every assistant message whose recorded string
//! content is sent as one text part; every other member, `reasoning` and
//! `tool_calls` included, and every other request is sent as recorded, so
//! any other difference fails. A string and a one-part text array share a
//! coarse shape, so replay serves the recorded replies, the test's own
//! predicates still run, and the request snapshot check refuses exactly the
//! named changes at finalization.

use std::future::Future;

use anyhow::{Context, Result, ensure};
use rig::providers::openai::wire::GROQ;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use crate::chat_request_witness::{Change, Recording, with_snapshot_witness};

const LONG_HISTORY_REPLAY: &str =
    "agent_tool_sessions/long_history_replay_with_tool_result_continuation";
const HISTORY_SURVIVAL_UNARY: &str = "history_survival_matrix/unary";
const HISTORY_SURVIVAL_STREAMING: &str = "history_survival_matrix/streaming";
const PORTABILITY_FROM_ANTHROPIC: &str = "portability_matrix/from_anthropic";
const PROMPT_CACHING_BLOCKING_PROBE: &str = "prompt_caching/blocking_probe";

/// The complete set of admitted recorded Groq request differences.
#[derive(Clone, Copy, Debug)]
pub(super) enum NamedCase {
    LongHistoryReplay,
    HistorySurvivalUnary,
    HistorySurvivalStreaming,
    PortabilityFromAnthropic,
    PromptCachingBlockingProbe,
}

impl NamedCase {
    const ALL: [Self; 5] = [
        Self::LongHistoryReplay,
        Self::HistorySurvivalUnary,
        Self::HistorySurvivalStreaming,
        Self::PortabilityFromAnthropic,
        Self::PromptCachingBlockingProbe,
    ];

    /// The case whose recording `scenario` names.
    pub(super) fn of(scenario: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|case| case.scenario() == scenario)
            .with_context(|| format!("{scenario} names no Groq request witness"))
    }

    fn scenario(self) -> &'static str {
        match self {
            Self::LongHistoryReplay => LONG_HISTORY_REPLAY,
            Self::HistorySurvivalUnary => HISTORY_SURVIVAL_UNARY,
            Self::HistorySurvivalStreaming => HISTORY_SURVIVAL_STREAMING,
            Self::PortabilityFromAnthropic => PORTABILITY_FROM_ANTHROPIC,
            Self::PromptCachingBlockingProbe => PROMPT_CACHING_BLOCKING_PROBE,
        }
    }

    /// The recorded bodies. Each is read through its literal coordinate,
    /// which is also how the cassette registry counts the fixture.
    fn recorded(self) -> Vec<(String, String)> {
        match self {
            Self::LongHistoryReplay => {
                crate::cassettes::recorded_interaction_bodies("groq", LONG_HISTORY_REPLAY)
            }
            Self::HistorySurvivalUnary => {
                crate::cassettes::recorded_interaction_bodies("groq", HISTORY_SURVIVAL_UNARY)
            }
            Self::HistorySurvivalStreaming => {
                crate::cassettes::recorded_interaction_bodies("groq", HISTORY_SURVIVAL_STREAMING)
            }
            Self::PortabilityFromAnthropic => {
                crate::cassettes::recorded_interaction_bodies("groq", PORTABILITY_FROM_ANTHROPIC)
            }
            Self::PromptCachingBlockingProbe => {
                crate::cassettes::recorded_interaction_bodies("groq", PROMPT_CACHING_BLOCKING_PROBE)
            }
        }
    }

    fn recorded_interactions(self) -> usize {
        match self {
            Self::LongHistoryReplay | Self::PortabilityFromAnthropic => 1,
            Self::PromptCachingBlockingProbe => 3,
            Self::HistorySurvivalUnary | Self::HistorySurvivalStreaming => 9,
        }
    }

    /// Every `(interaction, message)` whose recorded assistant string
    /// content is sent as one text part.
    fn text_part_contents(self) -> &'static [(usize, usize)] {
        match self {
            Self::LongHistoryReplay => &[(0, 2), (0, 4), (0, 8)],
            Self::HistorySurvivalUnary | Self::HistorySurvivalStreaming => {
                &[(6, 12), (7, 12), (8, 12), (8, 16)]
            }
            Self::PortabilityFromAnthropic => &[(0, 2)],
            Self::PromptCachingBlockingProbe => &[(2, 2)],
        }
    }

    /// Whether the recorded requests carry assistant `reasoning`, which then
    /// replays unchanged.
    fn replays_reasoning(self) -> bool {
        match self {
            Self::HistorySurvivalUnary
            | Self::HistorySurvivalStreaming
            | Self::PromptCachingBlockingProbe => true,
            Self::LongHistoryReplay | Self::PortabilityFromAnthropic => false,
        }
    }

    fn changes(self, recording: &Recording) -> Result<Vec<(usize, Change)>> {
        self.text_part_contents()
            .iter()
            .map(|&(interaction, message)| {
                let recorded = recording
                    .body(interaction)?
                    .pointer(&format!("/messages/{message}"))
                    .context("the named recorded message exists")?;
                ensure!(
                    recorded.get("role") == Some(&json!("assistant")),
                    "the named message is not an assistant turn: {recorded}"
                );
                let text = recorded
                    .get("content")
                    .and_then(Value::as_str)
                    .with_context(|| format!("the named recorded content is text: {recorded}"))?;
                Ok((
                    interaction,
                    Change::Replaced {
                        pointer: format!("/messages/{message}/content"),
                        recorded: json!(text),
                        sent: json!([{ "text": text, "type": "text" }]),
                    },
                ))
            })
            .collect()
    }
}

/// Whether any recorded request carries an assistant message's `reasoning`.
fn recording_has_reasoning(recording: &Recording) -> Result<bool> {
    for interaction in 0..recording.len() {
        let messages = recording
            .body(interaction)?
            .get("messages")
            .and_then(Value::as_array)
            .context("a recorded Chat request has messages")?;
        if messages.iter().any(|message| {
            message.get("role") == Some(&json!("assistant"))
                && message.get("reasoning").is_some_and(Value::is_string)
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Bind one literal scenario to the request snapshot's exact refusal of its
/// named changes. `body` gets the session's models and runs the test's own
/// predicates. Setup failures and panics remain failures after the session
/// is finalized.
pub(super) async fn with_groq_request_witness_cassette<F, Fut>(
    scenario: &'static str,
    case: NamedCase,
    body: F,
) -> Result<()>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    ensure!(
        scenario == case.scenario(),
        "the named scenario changed: {scenario}, {case:?}"
    );
    let recording = Recording::new("groq", scenario, &GROQ, case.recorded())?;
    ensure!(
        recording.len() == case.recorded_interactions(),
        "the original recorded interaction count changed: {}",
        recording.len()
    );
    ensure!(
        recording_has_reasoning(&recording)? == case.replays_reasoning(),
        "the recording's assistant reasoning changed: {case:?}"
    );
    let changes = case.changes(&recording)?;
    with_snapshot_witness(recording, changes, body).await
}
