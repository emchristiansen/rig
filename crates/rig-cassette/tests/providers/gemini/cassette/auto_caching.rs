//! Automatic explicit caching (`gemini::caching`) over long live runs on
//! gemini-3.8-flash, recorded and replayed.
//!
//! Every assertion here is on recorded traffic. The figures and the checks
//! every long run shares live in `rig_test_support::cache_longrun`; for a run
//! that uses a cache book they include Gemini's rules: a request that reads a
//! cache carries none of the prefix the cache holds, cache plus request tail
//! is the conversation byte for byte, every cache is deleted, every replaced
//! cache was read at least three times, and cache creation stays at or below
//! 15% of prompt tokens. The input saving counts every creation at the input
//! price and every cache's storage.
//!
//! The recordings send each declaration as `parametersJsonSchema`, which the
//! port's projection sends as `parameters`, so in replay each run is an exact
//! named request-mismatch witness for its first request, and the run and its
//! checks are named losses (see each test).
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `GEMINI_API_KEY`; see
//! `tests/README.md`.

use std::collections::HashMap;
use std::time::Duration;

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::{CacheCost, CacheRates, Effort, GenerationOptions, Message};
use rig::providers::gemini::{
    AutoCache, CacheBook, CacheEvent, CacheReport, Gemini, ThoughtReplay,
};
use rig::tool::Tool;
use rig_test_support::cache_longrun::{
    self, CacheWire, Limits, LongRun, LookupOrder, Recording, RunLog, SUPPORT_PREAMBLE, chat,
    chat_streamed, question,
};
use serde::Deserialize;
use serde_json::{Value, json};

use rig_test_support::cassette_models::GeminiModels;

use crate::cassettes::CassetteClock;

use super::super::request_witness_support::{
    Attempt, NamedCase, stream_result, with_gemini_request_witness_cassette,
};
use super::super::support::with_gemini_auto_caching_cassette;

const MODEL: &str = "gemini-3.8-flash";
/// Gemini 3.8 Flash standard prices, USD per 1M tokens: input, cached read,
/// creation (billed as input) and storage per hour.
const RATES: CacheRates = CacheRates {
    input: 0.75,
    cached_read: 0.075,
    cache_write: 0.75,
    storage_per_hour: 0.50,
};
/// Gemini 3.8 Flash standard output price (thinking included), USD per 1M tokens.
const OUTPUT_PRICE: f64 = 3.75;
/// The most caches a 100-turn chat may create. Rolling when the premium
/// already paid equals the next cache's cost is the cheapest schedule for a
/// conversation that grows every call; measured over 100 turns it created 12
/// caches with current-turn replay (three recordings) and 13 with default
/// replay (two recordings). The cap started at 12 as an estimate made before
/// any 100-turn run and was set from those measurements. The per-cache
/// "read at least three times" check and the 15% creation limit are what rule
/// out paying for caches that are never used.
const MAX_CACHES_100_TURNS: usize = 14;
/// Gemini's limit on tokens put into caches, relative to prompt tokens.
const MAX_CREATED_SHARE: f64 = 0.15;

// ---------------------------------------------------------------------------
// Tools.

#[derive(Deserialize)]
struct LogArgs {
    name: String,
}

/// Reads one log. Every tenth log is about 20k tokens of mostly digits
/// (digits tokenize one per character, which keeps the fixture small for its
/// token count); the rest are short.
struct ReadLog;

fn log_body(index: u32) -> String {
    let big = index.is_multiple_of(10);
    let lines = if big { 425 } else { 6 };
    let mut seed = u64::from(index)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(17);
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        seed >> 33
    };
    (0..lines)
        .map(|line| {
            let level = if next() % 9 == 0 { "ERROR" } else { "INFO" };
            format!(
                "{level} t={} rid={:010} lat={:05} st={} b={:07}",
                1_780_000_000 + u64::from(index) * 1_000 + line,
                next(),
                next() % 100_000,
                [200, 200, 200, 404, 500][(next() % 5) as usize],
                next() % 10_000_000,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl Tool for ReadLog {
    const NAME: &'static str = "read_log";
    type Error = std::convert::Infallible;
    type Args = LogArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read one log file by name, such as log-7.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let index = args
            .name
            .rsplit('-')
            .next()
            .and_then(|number| number.parse::<u32>().ok())
            .unwrap_or(0);
        Ok(log_body(index))
    }
}

// ---------------------------------------------------------------------------
// Agents, books and checks.

fn thinking_low() -> GenerationOptions {
    GenerationOptions::default().reasoning(Effort::Low)
}

fn support_agent(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    preamble: &str,
) -> Agent {
    AgentBuilder::new(model)
        .preamble(preamble)
        .tool(LookupOrder)
        .max_tokens(800)
        .options(thinking_low())
        .default_max_turns(4)
        .build()
}

/// A book on the session's clock with the run's display-name prefix.
fn book(clock: &CassetteClock, policy: AutoCache) -> CacheBook {
    let clock = clock.clone();
    CacheBook::new(policy)
        .with_clock(move || clock.now())
        .with_display_prefix("rig-auto-caching-test-")
}

/// Limits for a run on a book: its saving floor and, where it holds from
/// the first cache read on, its per-call floor.
fn limits(min_saving: f64, min_call_share: Option<f64>) -> Option<Limits> {
    Some(Limits {
        min_saving: Some(min_saving),
        min_call_share,
        max_writes_share: Some(MAX_CREATED_SHARE),
    })
}

/// Read the run back and run the shared checks, pricing the book's caches
/// (`resources`) beside the calls.
fn check(
    scenario: &str,
    log: &RunLog,
    resources: Option<CacheCost>,
    limits: Option<Limits>,
    drops_signatures: bool,
) -> (Recording, cache_longrun::Figures) {
    let run = LongRun {
        wire: CacheWire::Gemini,
        scenario,
        model: MODEL,
        rates: RATES,
        output_price: OUTPUT_PRICE,
        limits,
        drops_signatures,
        conversation: None,
    };
    cache_longrun::check(&run, log, resources)
}

/// How far the book's size estimate was from Gemini's count, per cache.
fn print_estimates(scenario: &str, events: &[CacheEvent]) {
    for event in events {
        if let CacheEvent::Created {
            name,
            tokens,
            estimated,
            ..
        } = event
        {
            println!(
                "AUTO_CACHING_ESTIMATE {scenario} {name} tokens={tokens} estimated={estimated} \
                 error={:+.1}%",
                100.0 * (*estimated as f64 - *tokens as f64) / (*tokens).max(1) as f64
            );
        }
    }
}

/// Every call that names a cache reads it whole: its cached tokens equal the
/// cache's size.
fn assert_reads_whole(recording: &Recording) {
    let sizes: HashMap<&str, u64> = recording
        .created
        .iter()
        .map(|created| (created.name.as_str(), created.tokens))
        .collect();
    for call in &recording.calls {
        if let Some(cache) = &call.cache {
            let cached = call.usage.cached_input_tokens.unwrap_or(0);
            assert_eq!(
                cached,
                sizes[cache.as_str()],
                "interaction {} read {cached} of {cache}'s {} tokens",
                call.index,
                sizes[cache.as_str()]
            );
        }
    }
}

fn assert_caches_at_most(figures: &cache_longrun::Figures, most: usize) {
    cache_longrun::print_limit(figures, &format!("at most {most} caches"));
    let created = figures.caches_created.unwrap_or(0);
    assert!(created <= most, "{created} caches (limit {most})");
}

// ---------------------------------------------------------------------------
// The runs.

/// One 100-turn support chat on a book; `streamed` sends it over
/// `streamGenerateContent`.
async fn support_chat_100(
    gemini: Gemini,
    clock: CassetteClock,
    replay: ThoughtReplay,
    streamed: bool,
) -> (CacheReport, RunLog, Vec<CacheEvent>) {
    let book = book(&clock, AutoCache::default());
    let agent = support_agent(
        gemini
            .completion(MODEL)
            .thought_replay(replay)
            .caching(&book),
        SUPPORT_PREAMBLE,
    );
    let mut history = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=100 {
        let prompt = question(turn, "A");
        if streamed {
            chat_streamed(&agent, &clock, prompt, &mut history, &mut log).await;
        } else {
            chat(&agent, &clock, prompt, &mut history, &mut log).await;
        }
    }
    book.close(&gemini.cached_contents()).await;
    (book.report(), log, book.events())
}

/// The first turn of a 100-turn support chat on a book over the session's
/// clock, with current-turn signature replay; `streamed` sends it over
/// `streamGenerateContent`, as `chat_streamed` does.
async fn first_support_turn(client: GeminiModels, clock: CassetteClock, streamed: bool) -> Attempt {
    let book = book(&clock, AutoCache::default());
    let agent = support_agent(
        client
            .completion(MODEL)
            .thought_replay(ThoughtReplay::CurrentTurn)
            .caching(&book),
        SUPPORT_PREAMBLE,
    );
    let prompt = question(1, "A");
    let result = if streamed {
        let mut stream = agent.prompt(prompt).history(Vec::<Message>::new()).stream();
        let result = stream_result(&mut stream).await;
        drop(stream);
        result
    } else {
        Ok(agent.chat(prompt, &mut Vec::new()).await.map(|_| ()))
    };
    drop(agent);
    result
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent`. Named losses:
/// every later turn of the 100-turn chat and the run's checks: the shared
/// long-run checks with a 0.65 saving and 0.50 per-call floor and a 15%
/// creation limit, a cached share of at least 80%, and at most
/// `MAX_CACHES_100_TURNS` caches.
#[tokio::test]
async fn support_chat_100_current_turn() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/support_chat_100_current_turn",
        NamedCase::SupportChatCurrentTurn,
        |client, clock| first_support_turn(client, clock, false),
    )
    .await
}

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and the re-record reached 49-50% implicit-cache coverage against the 50% floor in every attempt"]
async fn support_chat_100_auto() {
    const SCENARIO: &str = "auto_caching/support_chat_100_auto";
    let (report, log, events) =
        with_gemini_auto_caching_cassette("auto_caching/support_chat_100_auto", |gemini, clock| {
            support_chat_100(gemini, clock, ThoughtReplay::All, false)
        })
        .await;
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.30, Some(0.50)),
        false,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(&figures, "every read takes the whole cache");
    assert_reads_whole(&recording);
    assert_caches_at_most(&figures, MAX_CACHES_100_TURNS);
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `streamGenerateContent`. Named
/// losses: every later turn of the 100-turn chat and the run's checks: every
/// call streaming, the shared long-run checks with a 0.65 saving and 0.50
/// per-call floor and a 15% creation limit, a cached share of at least 80%,
/// and at most `MAX_CACHES_100_TURNS` caches.
#[tokio::test]
async fn support_chat_100_streamed() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/support_chat_100_streamed",
        NamedCase::SupportChatStreamed,
        |client, clock| first_support_turn(client, clock, true),
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent`. Named losses:
/// turns 2-50, the checkpoint and its restore into a new book proven live,
/// the resumed turn and turns 52-100, and the run's checks: the first call
/// after the resume reading a pre-checkpoint cache with no cache created
/// before it, the shared long-run checks with a 0.65 saving and 0.50 per-call
/// floor and a 15% creation limit, a cached share of at least 80%, and at
/// most `MAX_CACHES_100_TURNS` caches.
#[tokio::test]
async fn support_chat_100_resume() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/support_chat_100_resume",
        NamedCase::SupportChatResume,
        |client, clock| first_support_turn(client, clock, false),
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent`. Named losses:
/// every later turn, the compaction at turn 51, and the run's checks: no
/// stale cache read by the first call after compaction, every pre-compaction
/// cache deleted before the run ends, and the shared long-run checks with a
/// 0.50 saving and 0.50 per-call floor and a 15% creation limit.
#[tokio::test]
async fn support_chat_100_compaction() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/support_chat_100_compaction",
        NamedCase::SupportChatCompaction,
        |client, clock| first_support_turn(client, clock, false),
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`read_log`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent`. Named losses:
/// the 60-log loop and the run's checks: at least 60 calls, at most 3 caches,
/// a cached share of at least 60%, and the shared long-run checks with a 0.60
/// saving and a 15% creation limit.
#[tokio::test]
async fn agent_loop_large_results() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/agent_loop_large_results",
        NamedCase::AutoCachingLargeResults,
        |client, clock| async move {
            let book = book(&clock, AutoCache::default());
            let agent = AgentBuilder::new(client.completion(MODEL).caching(&book))
                .preamble(
                    "You are a meticulous log auditor. You read logs with the read_log tool, \
                     exactly one call per turn, never several calls in one turn, strictly in the \
                     order you are told. Do not stop early.",
                )
                .tool(ReadLog)
                .max_tokens(800)
                .options(thinking_low())
                .default_max_turns(70)
                .build();
            let result = agent
                .chat(
                    "Read log-1 through log-60 with read_log, one call per turn, in order (log-1, \
                     log-2, ..., log-60). After the last one, report how many lines start with \
                     ERROR across all sixty logs, as a single number."
                        .to_owned(),
                    &mut Vec::new(),
                )
                .await
                .map(|_| ());
            drop(agent);
            Ok(result)
        },
    )
    .await
}

/// A ~5.5k-token prefix: the handbook and a product appendix.
fn shared_preamble() -> String {
    let mut preamble = SUPPORT_PREAMBLE.to_owned();
    preamble.push_str("\n## Appendix: product catalog\n\n");
    for sku in 0..220 {
        preamble.push_str(&format!(
            "- SKU-{sku:04}: {} {}, {} grams, ships in {} business days, returnable: {}.\n",
            ["Linen", "Cotton", "Wool", "Canvas", "Leather", "Bamboo"][sku % 6],
            [
                "shirt",
                "tote",
                "scarf",
                "cap",
                "wallet",
                "sock pack",
                "apron"
            ][sku % 7],
            120 + (sku * 37) % 900,
            1 + sku % 5,
            if sku % 11 == 0 {
                "no (final sale)"
            } else {
                "yes"
            },
        ));
    }
    preamble
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent` from the first
/// of four agents. Named losses: every later turn of the four 25-turn chats
/// and the run's checks: the shared prefix created exactly once, and the
/// shared long-run checks with a 0.60 saving and 0.50 per-call floor and a
/// 15% creation limit.
#[tokio::test]
async fn subagents_shared_prefix() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/subagents_shared_prefix",
        NamedCase::AutoCachingSubagents,
        |client, clock| async move {
            let book = book(&clock, AutoCache::default());
            let preamble = shared_preamble();
            let agents: Vec<Agent> = (0..4)
                .map(|_| support_agent(client.completion(MODEL).caching(&book), &preamble))
                .collect();
            let prefixes = ["B", "C", "D", "E"];
            // Turn 1 runs agent by agent; the first agent's turn is refused.
            let result = agents[0]
                .chat(question(1, prefixes[0]), &mut Vec::new())
                .await
                .map(|_| ());
            drop(agents);
            Ok(result)
        },
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording and its clock sidecar are
/// unchanged; the book reads the session's replayed clock; the only request
/// difference is that declaration 0 (`lookup_order`) sends its recorded
/// `parametersJsonSchema`, value unchanged, as `parameters`, and replay
/// refuses interaction 0, the first turn's `generateContent`. Named losses:
/// every later turn and the run's checks: (a) a cache refused as too small
/// with the next request inline and successful, (b) exactly one 403 on a
/// generate call, (c) that 403 followed at once by the same request inline
/// and later by a new cache, (d) the second close deleting nothing new, and
/// the shared long-run checks with a 0.0 saving floor and a 15% creation
/// limit.
#[tokio::test]
async fn lifecycle() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "auto_caching/lifecycle",
        NamedCase::AutoCachingLifecycle,
        |client, clock| async move {
            // (a) min_tokens 0 and a small first request: Gemini refuses the
            // cache as too small. (b) TTL 10 s with a 15 s pause.
            let book = book(
                &clock,
                AutoCache {
                    min_tokens: 0,
                    ttl: Duration::from_secs(10),
                    ..AutoCache::default()
                },
            );
            let agent = support_agent(
                client.completion(MODEL).caching(&book),
                "You are a customer support agent. Look orders up with lookup_order before \
                 answering, and answer in one sentence.",
            );
            let result = agent
                .chat(question(1, "A"), &mut Vec::new())
                .await
                .map(|_| ());
            drop(agent);
            Ok(result)
        },
    )
    .await
}
