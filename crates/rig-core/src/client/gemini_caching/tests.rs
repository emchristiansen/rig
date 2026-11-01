//! [`Caching`] over a scripted transport and a clock the test sets: each
//! lifecycle step the recorded `auto_caching` sessions exercised (a roll
//! that pays for itself, reads, extension, expiry across a pause, a refused
//! cache, a cache deleted behind the book's back, retirement, a shared
//! prefix, resume and close), asserted on the exact requests sent and the
//! book's events.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{Value, json};

use super::Caching;
use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::ProviderError;
use crate::http_client::MultipartForm;
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::cached_content::CachedContents;
use crate::providers::gemini::caching::{
    AutoCache, CacheBook, CacheEvent, CacheReport, CreatedCache, Lease, digests, parse,
    short_digest,
};
use crate::providers::gemini::completion::{GenerateContent, ThoughtReplay};
use crate::wire::{Body, Encoded, Framing, Mode, Wire, WireFrame};

const MODEL: &str = "gemini-test";
const KEY: &str = "test-key";
const API: &str = "https://generativelanguage.googleapis.com";
const GENERATE: &str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent?key=test-key";
const STREAM: &str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent?alt=sse&key=test-key";
const CACHES: &str = "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=test-key";
const C1: &str = "cachedContents/c1";
const C2: &str = "cachedContents/c2";

// ---------------------------------------------------------------------------
// The scripted transport.

/// A request body as the transport received it.
#[derive(Clone, Debug, PartialEq)]
enum SentBody {
    Empty,
    Json(Value),
    Text(String),
    Multipart,
}

/// One request the transport received.
#[derive(Clone, Debug, PartialEq)]
struct Sent {
    method: http::Method,
    uri: String,
    body: SentBody,
}

/// How the transport answers one request.
enum Reply {
    /// The reply opens with these frames. A non-success HTTP reply is the
    /// one error frame the HTTP transport turns it into.
    Frames(Vec<Result<WireFrame, ProviderError>>),
    /// The reply fails to open.
    Refused(ProviderError),
}

/// Records every request and answers each with the next scripted reply.
/// Every opening yields once before it resolves, so two sends driven
/// together interleave at each request.
#[derive(Clone, Default)]
struct Script {
    sent: Arc<Mutex<Vec<Sent>>>,
    replies: Arc<Mutex<VecDeque<Reply>>>,
}

impl Script {
    fn push(&self, reply: Reply) {
        self.replies.lock().expect("replies").push_back(reply);
    }

    /// The requests received since the last call.
    fn take(&self) -> Vec<Sent> {
        std::mem::take(&mut *self.sent.lock().expect("sent"))
    }

    fn remaining(&self) -> usize {
        self.replies.lock().expect("replies").len()
    }
}

async fn yield_once() {
    let mut yielded = false;
    futures::future::poll_fn(move |cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
}

impl<W> Transport<W> for Script
where
    W: Wire<Payload = Encoded, Frame = WireFrame>,
{
    fn send(&self, payload: Encoded, _exchange: Exchange) -> Opening<WireFrame> {
        let body = match payload.request.body() {
            Body::Bytes(bytes) if bytes.is_empty() => SentBody::Empty,
            Body::Bytes(bytes) => match serde_json::from_slice(bytes) {
                Ok(value) => SentBody::Json(value),
                Err(_) => SentBody::Text(String::from_utf8_lossy(bytes).into_owned()),
            },
            Body::Multipart(_) => SentBody::Multipart,
        };
        self.sent.lock().expect("sent").push(Sent {
            method: payload.request.method().clone(),
            uri: payload.request.uri().to_string(),
            body,
        });
        let reply = self.replies.lock().expect("replies").pop_front();
        Opening::new(async move {
            yield_once().await;
            match reply {
                Some(Reply::Frames(frames)) => Ok(Opened::new(futures::stream::iter(frames))),
                Some(Reply::Refused(error)) => Err(error),
                None => Err(ProviderError::Provider("an unscripted request".to_owned())),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Replies.

fn answer_text(cached: u64) -> String {
    json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 2000, "cachedContentTokenCount": cached}
    })
    .to_string()
}

/// A finished `generateContent` reply that read `cached` tokens from a cache.
fn answer(cached: u64) -> Reply {
    Reply::Frames(vec![Ok(WireFrame::Text(answer_text(cached)))])
}

fn resource(body: Value) -> Reply {
    Reply::Frames(vec![Ok(WireFrame::Text(body.to_string()))])
}

/// The `cachedContents` resource Gemini answers a create with.
fn created(name: &str, tokens: u64) -> Reply {
    resource(json!({
        "name": name,
        "model": format!("models/{MODEL}"),
        "usageMetadata": {"totalTokenCount": tokens}
    }))
}

/// A non-success reply, as the HTTP transport delivers it.
fn status(code: u16, body: Value) -> Reply {
    let status = http::StatusCode::from_u16(code).expect("a status code");
    Reply::Frames(vec![Err(ProviderError::from_http_response(
        status,
        body.to_string(),
    ))])
}

fn refused(message: &str) -> Reply {
    Reply::Refused(ProviderError::Provider(message.to_owned()))
}

// ---------------------------------------------------------------------------
// Requests.

fn system() -> Value {
    json!({"parts": [{"text": "Answer as the store's support agent. ".repeat(12)}]})
}

fn generation_config() -> Value {
    json!({"maxOutputTokens": 64})
}

/// The `index`th content of the support chat: user questions and model
/// answers, alternating.
fn turn(index: usize) -> Value {
    let number = index / 2 + 1;
    match index % 2 {
        0 => json!({"role": "user", "parts": [{"text": format!("question {number}")}]}),
        _ => json!({"role": "model", "parts": [{"text": format!("answer {number}")}]}),
    }
}

fn body_of(contents: Vec<Value>) -> Value {
    json!({
        "systemInstruction": system(),
        "contents": contents,
        "generationConfig": generation_config()
    })
}

/// The chat's request carrying its first `len` contents.
fn request(len: usize) -> Value {
    body_of((0..len).map(turn).collect())
}

/// The request that reads `name`, which holds the first `from` contents,
/// for the chat's first `to`.
fn reading(name: &str, from: usize, to: usize) -> Value {
    let contents: Vec<Value> = (from..to).map(turn).collect();
    json!({
        "cachedContent": name,
        "contents": contents,
        "generationConfig": generation_config()
    })
}

/// The book's size estimate of a fragment: its compact bytes over four.
fn tokens(fragment: &Value) -> u64 {
    (fragment.to_string().len() / 4) as u64
}

/// The book's estimate of the prefix and the chat's first `covers` contents.
fn estimate(covers: usize) -> u64 {
    tokens(&system()) + (0..covers).map(|index| tokens(&turn(index))).sum::<u64>()
}

/// The digest of the prefix and the first `covers` contents of `body`.
fn digest_of(body: &Value, covers: usize) -> String {
    let parsed = parse(body.to_string().as_bytes()).expect("a generateContent body");
    digests(MODEL, &parsed)[covers].clone()
}

fn digest(covers: usize) -> String {
    digest_of(&request(covers), covers)
}

fn post(uri: &str, body: &Value) -> Encoded {
    let request = http::Request::post(uri)
        .body(Body::Bytes(body.to_string().into_bytes()))
        .expect("a request");
    Encoded::new(request, Framing::Whole)
}

fn generate_sent(body: Value) -> Sent {
    Sent {
        method: http::Method::POST,
        uri: GENERATE.to_owned(),
        body: SentBody::Json(body),
    }
}

/// The create request for a cache of the chat's first `covers` contents.
fn create_sent(covers: usize, ttl_secs: u64) -> Sent {
    let mut body = json!({
        "model": format!("models/{MODEL}"),
        "displayName": format!("rig-test-{}", short_digest(&digest(covers))),
        "ttl": format!("{ttl_secs}s"),
        "systemInstruction": system()
    });
    if covers > 0 {
        let contents: Vec<Value> = (0..covers).map(turn).collect();
        body["contents"] = Value::Array(contents);
    }
    Sent {
        method: http::Method::POST,
        uri: CACHES.to_owned(),
        body: SentBody::Json(body),
    }
}

fn delete_sent(name: &str) -> Sent {
    Sent {
        method: http::Method::DELETE,
        uri: format!("{API}/v1beta/{name}?key={KEY}"),
        body: SentBody::Empty,
    }
}

fn texts(frames: Vec<Result<WireFrame, ProviderError>>) -> Vec<String> {
    frames
        .into_iter()
        .map(|frame| frame.expect("a reply frame").as_str().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// A run: a book on a set clock, and the caching transport over the script.

const START: u64 = 1_000;

/// No price discount, no storage price, no minimum size or gain, implicit
/// caching never assumed: a cache pays once the premium paid on its line
/// reaches the tokens it would hold.
fn eager() -> AutoCache {
    AutoCache {
        ttl: Duration::from_secs(10),
        implicit_floor: u64::MAX,
        implicit_ceiling: u64::MAX,
        min_tokens: 0,
        min_gain: 0,
        cached_ratio: 0.0,
        storage_ratio_per_hour: 0.0,
    }
}

struct Run {
    script: Script,
    book: CacheBook,
    clock: Arc<AtomicU64>,
    caching: Caching<Script>,
}

impl Run {
    fn new(policy: AutoCache, replay: ThoughtReplay) -> Self {
        let script = Script::default();
        let clock = Arc::new(AtomicU64::new(START));
        let reading = Arc::clone(&clock);
        let book = CacheBook::new(policy)
            .with_clock(move || reading.load(Ordering::SeqCst))
            .with_display_prefix("rig-test-");
        let caching = Model::new(
            GenerateContent::new(GeminiConfig::new(KEY), MODEL),
            script.clone(),
        )
        .thought_replay(replay)
        .caching(&book)
        .transport;
        Self {
            script,
            book,
            clock,
            caching,
        }
    }

    fn at(&self, now: u64) {
        self.clock.store(now, Ordering::SeqCst);
    }

    fn reply(&self, reply: Reply) {
        self.script.push(reply);
    }

    fn take(&self) -> Vec<Sent> {
        self.script.take()
    }

    async fn try_send(
        &self,
        payload: Encoded,
        mode: Mode,
    ) -> Result<Vec<Result<WireFrame, ProviderError>>, ProviderError> {
        let exchange = Exchange {
            mode,
            observation: None,
            credential_stamp: None,
        };
        let opened = self.caching.send(payload, exchange).await?;
        Ok(opened.frames.collect().await)
    }

    async fn send(&self, payload: Encoded) -> Vec<String> {
        texts(
            self.try_send(payload, Mode::Unary)
                .await
                .expect("the reply opens"),
        )
    }

    /// Send the chat's first `len` contents.
    async fn generate(&self, len: usize) -> Vec<String> {
        self.send(post(GENERATE, &request(len))).await
    }
}

/// Two inline calls at 1,000 and 1,010: the premium paid is the prefix,
/// then the prefix and two contents. Neither pays for a cache.
async fn warmed(policy: AutoCache) -> Run {
    let run = Run::new(policy, ThoughtReplay::All);
    run.reply(answer(0));
    run.generate(1).await;
    run.reply(answer(0));
    run.at(1_010);
    run.generate(3).await;
    run
}

/// A third call at 1,020, whose premium paid so far covers a cache of its
/// first four contents: the book creates `c1` (Gemini counts twice the
/// estimate, so calibration becomes 2) with a TTL of twice the 10 s gap.
async fn rolled() -> Run {
    let run = warmed(eager()).await;
    run.reply(created(C1, 2 * estimate(4)));
    run.reply(answer(2 * estimate(4)));
    run.at(1_020);
    run.generate(5).await;
    run
}

fn c1() -> Lease {
    Lease {
        name: C1.to_owned(),
        digest: digest(4),
        covers: 4,
        tokens: 2 * estimate(4),
        expires_at: 1_040,
        model: MODEL.to_owned(),
    }
}

fn c1_created() -> CacheEvent {
    CacheEvent::Created {
        name: C1.to_owned(),
        tokens: 2 * estimate(4),
        estimated: estimate(4),
        covers: 4,
        ttl_secs: 20,
        at: 1_020,
    }
}

// ---------------------------------------------------------------------------
// Create, read and extend.

/// The roll happens on the first call whose premium already paid covers
/// the cache, the call itself then reads it, and a later read close to its
/// expiry extends it first.
#[tokio::test]
async fn a_cache_is_rolled_once_its_premium_pays_then_read_and_extended() {
    let run = rolled().await;
    assert_eq!(
        run.take(),
        vec![
            generate_sent(request(1)),
            generate_sent(request(3)),
            create_sent(4, 20),
            generate_sent(reading(C1, 4, 5)),
        ]
    );
    assert_eq!(run.book.leases(), vec![c1()]);

    // At 1,026 the cache has 14 s left, under 1.5 times the line's 10 s gap.
    run.reply(resource(json!({"name": C1})));
    run.reply(answer(2 * estimate(4)));
    run.at(1_026);
    run.generate(7).await;
    assert_eq!(
        run.take(),
        vec![
            Sent {
                method: http::Method::PATCH,
                uri: format!("{API}/v1beta/{C1}?updateMask=ttl&key={KEY}"),
                body: SentBody::Json(json!({"ttl": "20s"})),
            },
            generate_sent(reading(C1, 4, 7)),
        ]
    );
    let extended = Lease {
        expires_at: 1_046,
        ..c1()
    };
    assert_eq!(
        run.book.events(),
        vec![
            c1_created(),
            CacheEvent::Extended {
                name: C1.to_owned(),
                expires_at: 1_046,
                at: 1_026,
            },
        ]
    );
    assert_eq!(
        run.book.report(),
        CacheReport {
            created: vec![CreatedCache {
                name: C1.to_owned(),
                tokens: 2 * estimate(4),
                at: 1_020,
            }],
            reads: BTreeMap::from([(C1.to_owned(), 2)]),
            retired: Vec::new(),
            lost: Vec::new(),
            token_hours: (2 * estimate(4)) as f64 * 6.0 / 3600.0,
            live: vec![extended],
        }
    );
    assert_eq!(run.script.remaining(), 0);
}

/// A roll on a line that already reads a cache replaces it: the new cache
/// is created, the one it replaces deleted, and the call reads the new one.
/// Here Gemini reports no cached tokens on two reads, so their whole size
/// is premium.
#[tokio::test]
async fn a_second_roll_deletes_the_cache_it_replaces() {
    let run = rolled().await;
    run.take();
    run.reply(answer(0));
    run.at(1_022);
    run.generate(7).await;
    run.reply(answer(0));
    run.at(1_024);
    run.generate(9).await;
    run.reply(created(C2, 2 * estimate(10)));
    run.reply(resource(json!({})));
    run.reply(answer(2 * estimate(10)));
    run.at(1_026);
    run.generate(11).await;

    assert_eq!(
        run.take(),
        vec![
            generate_sent(reading(C1, 4, 7)),
            generate_sent(reading(C1, 4, 9)),
            create_sent(10, 20),
            delete_sent(C1),
            generate_sent(reading(C2, 10, 11)),
        ]
    );
    // Calibrated at 2, the roll expected c1's tokens plus twice the six
    // contents after it.
    let tail: u64 = (4..10).map(|index| tokens(&turn(index))).sum();
    assert_eq!(
        run.book.events(),
        vec![
            c1_created(),
            CacheEvent::Created {
                name: C2.to_owned(),
                tokens: 2 * estimate(10),
                estimated: 2 * estimate(4) + 2 * tail,
                covers: 10,
                ttl_secs: 20,
                at: 1_026,
            },
            CacheEvent::Retired {
                name: C1.to_owned(),
                at: 1_026,
            },
        ]
    );
    assert_eq!(
        run.book.leases(),
        vec![Lease {
            name: C2.to_owned(),
            digest: digest(10),
            covers: 10,
            tokens: 2 * estimate(10),
            expires_at: 1_046,
            model: MODEL.to_owned(),
        }]
    );
    assert_eq!(run.script.remaining(), 0);
}

/// With current-turn replay a cache rolls only on a call that starts a new
/// user turn: a call ending in a tool result does not roll even when its
/// premium pays, and the next user turn does.
#[tokio::test]
async fn current_turn_replay_rolls_only_on_a_new_user_turn() {
    let call = json!({"role": "model", "parts": [{"functionCall": {"name": "lookup_order", "args": {"id": "A-1"}}}]});
    let result = json!({"role": "user", "parts": [{"functionResponse": {"name": "lookup_order", "response": {"status": "shipped"}}}]});
    let chat = [turn(0), turn(1), turn(2), call, result, turn(3), turn(4)];
    let body = |len: usize| body_of(chat[..len].to_vec());

    let run = Run::new(eager(), ThoughtReplay::CurrentTurn);
    for (len, at) in [(1, 1_000), (3, 1_010), (5, 1_020)] {
        run.reply(answer(0));
        run.at(at);
        run.send(post(GENERATE, &body(len))).await;
    }
    run.reply(created(C1, 500));
    run.reply(answer(500));
    run.at(1_030);
    run.send(post(GENERATE, &body(7))).await;

    let mut cache = json!({
        "model": format!("models/{MODEL}"),
        "displayName": format!("rig-test-{}", short_digest(&digest_of(&body(6), 6))),
        "ttl": "20s",
        "systemInstruction": system()
    });
    cache["contents"] = Value::Array(chat[..6].to_vec());
    assert_eq!(
        run.take(),
        vec![
            generate_sent(body(1)),
            generate_sent(body(3)),
            generate_sent(body(5)),
            Sent {
                method: http::Method::POST,
                uri: CACHES.to_owned(),
                body: SentBody::Json(cache),
            },
            generate_sent(json!({
                "cachedContent": C1,
                "contents": [turn(4)],
                "generationConfig": generation_config()
            })),
        ]
    );
}

// ---------------------------------------------------------------------------
// Expiry, retirement and loss.

/// After a pause past the cache's expiry (upstream's turn-15 pause), the
/// book never names the expired cache: the request goes out inline, and
/// the cache stays leased until it is retired.
#[tokio::test]
async fn an_expired_cache_is_never_named_after_a_pause() {
    let run = rolled().await;
    run.take();
    run.reply(answer(0));
    run.at(1_045);
    run.generate(7).await;
    assert_eq!(run.take(), vec![generate_sent(request(7))]);
    assert_eq!(run.book.events(), vec![c1_created()]);
    assert_eq!(run.book.leases(), vec![c1()]);
}

/// A conversation cache unread for longer than the idle limit is deleted
/// before the request goes out inline, whatever the delete answers: done,
/// already gone (403), failed (500), or never sent.
#[tokio::test]
async fn an_idle_conversation_cache_is_retired_whatever_the_delete_answers() {
    for outcome in [
        resource(json!({})),
        status(
            403,
            json!({"error": {"code": 403, "message": "CachedContent not found (or permission denied)", "status": "PERMISSION_DENIED"}}),
        ),
        status(
            500,
            json!({"error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}}),
        ),
        refused("connection reset"),
    ] {
        let run = rolled().await;
        run.take();
        run.reply(outcome);
        run.reply(answer(0));
        run.at(1_120);
        run.generate(7).await;
        assert_eq!(run.take(), vec![delete_sent(C1), generate_sent(request(7))]);
        assert_eq!(
            run.book.events(),
            vec![
                c1_created(),
                CacheEvent::Retired {
                    name: C1.to_owned(),
                    at: 1_120,
                },
            ]
        );
        assert_eq!(run.book.leases(), Vec::new());
        assert_eq!(run.book.report().retired, vec![C1.to_owned()]);
    }
}

fn forbidden() -> Reply {
    status(
        403,
        json!({"error": {"code": 403, "message": "CachedContent not found (or permission denied)", "status": "PERMISSION_DENIED"}}),
    )
}

/// A cache deleted behind the book's back answers 403: the book forgets it
/// and sends the same request inline, once, and its reply is the caller's.
#[tokio::test]
async fn a_cache_that_answers_403_is_forgotten_and_the_request_resent_inline() {
    let run = rolled().await;
    run.take();
    run.reply(forbidden());
    run.reply(answer(0));
    run.at(1_022);
    assert_eq!(run.generate(7).await, vec![answer_text(0)]);
    assert_eq!(
        run.take(),
        vec![generate_sent(reading(C1, 4, 7)), generate_sent(request(7))]
    );
    assert_eq!(
        run.book.events(),
        vec![
            c1_created(),
            CacheEvent::Lost {
                name: C1.to_owned(),
                at: 1_022,
            },
        ]
    );
    assert_eq!(
        run.book.report(),
        CacheReport {
            created: vec![CreatedCache {
                name: C1.to_owned(),
                tokens: 2 * estimate(4),
                at: 1_020,
            }],
            reads: BTreeMap::from([(C1.to_owned(), 1)]),
            retired: Vec::new(),
            lost: vec![C1.to_owned()],
            token_hours: (2 * estimate(4)) as f64 * 2.0 / 3600.0,
            live: Vec::new(),
        }
    );
}

/// The inline resend after a 403 is sent once: its failure is the caller's.
#[tokio::test]
async fn a_resend_after_a_403_that_fails_is_the_caller_s_error() {
    let run = rolled().await;
    run.take();
    run.reply(forbidden());
    run.reply(refused("connection reset"));
    run.at(1_022);
    let Err(error) = run.try_send(post(GENERATE, &request(7)), Mode::Unary).await else {
        panic!("the resend failed to open");
    };
    assert_eq!(error.to_string(), "ProviderError: connection reset");
    assert_eq!(
        run.take(),
        vec![generate_sent(reading(C1, 4, 7)), generate_sent(request(7))]
    );
    assert_eq!(
        run.book.events(),
        vec![
            c1_created(),
            CacheEvent::Lost {
                name: C1.to_owned(),
                at: 1_022,
            },
        ]
    );
}

/// A request whose reply fails to open fails the call, cached or not.
#[tokio::test]
async fn a_request_that_fails_to_open_is_the_caller_s_error() {
    let run = Run::new(eager(), ThoughtReplay::All);
    run.reply(refused("connection reset"));
    let Err(error) = run.try_send(post(GENERATE, &request(1)), Mode::Unary).await else {
        panic!("the request failed to open");
    };
    assert_eq!(error.to_string(), "ProviderError: connection reset");
    assert_eq!(run.take(), vec![generate_sent(request(1))]);
}

// ---------------------------------------------------------------------------
// Refused and failed creates.

/// Gemini refuses a cache as too small: the refusal is recorded with its
/// status and message, the request goes out inline, and the line does not
/// try again until it has grown by `min_gain` past the refused size.
#[tokio::test]
async fn a_refused_cache_is_recorded_and_not_retried_until_the_line_grows() {
    let min_gain = tokens(&turn(4)) + tokens(&turn(5)) + 1;
    let run = warmed(AutoCache {
        min_gain,
        ..eager()
    })
    .await;
    run.take();
    let message = format!(
        "Cached content is too small. total_token_count={}, min_total_token_count=1024",
        estimate(4)
    );
    run.reply(status(
        400,
        json!({"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}),
    ));
    run.reply(answer(0));
    run.at(1_020);
    run.generate(5).await;
    assert_eq!(
        run.take(),
        vec![create_sent(4, 20), generate_sent(request(5))]
    );

    // Two contents more is less than `min_gain` past the refusal.
    run.reply(answer(0));
    run.at(1_030);
    run.generate(7).await;
    assert_eq!(run.take(), vec![generate_sent(request(7))]);

    // Four more is enough.
    run.reply(created(C1, 2 * estimate(8)));
    run.reply(answer(2 * estimate(8)));
    run.at(1_040);
    run.generate(9).await;
    assert_eq!(
        run.take(),
        vec![create_sent(8, 20), generate_sent(reading(C1, 8, 9))]
    );
    assert_eq!(
        run.book.events(),
        vec![
            CacheEvent::CreateFailed {
                status: Some(400),
                message,
                at: 1_020,
            },
            CacheEvent::Created {
                name: C1.to_owned(),
                tokens: 2 * estimate(8),
                estimated: estimate(8),
                covers: 8,
                ttl_secs: 20,
                at: 1_040,
            },
        ]
    );
}

/// A create whose request fails to open is recorded without a status, with
/// the transport's error as its message, and the request goes out inline.
#[tokio::test]
async fn a_create_that_fails_to_open_is_recorded_without_a_status() {
    let run = warmed(eager()).await;
    run.take();
    run.reply(refused("quota exhausted"));
    run.reply(answer(0));
    run.at(1_020);
    run.generate(5).await;
    assert_eq!(
        run.take(),
        vec![create_sent(4, 20), generate_sent(request(5))]
    );
    assert_eq!(
        run.book.events(),
        vec![CacheEvent::CreateFailed {
            status: None,
            message: "ProviderError: quota exhausted".to_owned(),
            at: 1_020,
        }]
    );
}

/// A create answered with something that is not a cache resource leaves no
/// lease and no event, and the request goes out inline.
#[tokio::test]
async fn a_create_answered_without_a_resource_leaves_no_lease() {
    let run = warmed(eager()).await;
    run.take();
    run.reply(resource(json!({"unexpected": true})));
    run.reply(answer(0));
    run.at(1_020);
    run.generate(5).await;
    assert_eq!(
        run.take(),
        vec![create_sent(4, 20), generate_sent(request(5))]
    );
    assert_eq!(run.book.events(), Vec::new());
    assert_eq!(run.book.leases(), Vec::new());
}

// ---------------------------------------------------------------------------
// A shared prefix.

fn opening(who: &str) -> Value {
    json!({"role": "user", "parts": [{"text": format!("question from {who}")}]})
}

fn on_prefix(name: &str, contents: Vec<Value>) -> Value {
    json!({
        "cachedContent": name,
        "contents": contents,
        "generationConfig": generation_config()
    })
}

/// A second conversation on one prefix caches the prefix. Two that reach it
/// together create it once: the second waits for the first's create and
/// reads its cache. Each conversation then reads the prefix cache with its
/// whole history after it.
#[tokio::test]
async fn a_shared_prefix_is_created_once_and_read_by_every_conversation() {
    const PREFIX: &str = "cachedContents/prefix";
    let run = Run::new(eager(), ThoughtReplay::All);
    let prefix_tokens = 2 * estimate(0);
    run.reply(answer(0));
    run.send(post(GENERATE, &body_of(vec![opening("A")]))).await;

    run.reply(created(PREFIX, prefix_tokens));
    run.reply(answer(prefix_tokens));
    run.reply(answer(prefix_tokens));
    run.at(1_001);
    let (b, c) = futures::join!(
        run.send(post(GENERATE, &body_of(vec![opening("B")]))),
        run.send(post(GENERATE, &body_of(vec![opening("C")]))),
    );
    assert_eq!(
        (b, c),
        (
            vec![answer_text(prefix_tokens)],
            vec![answer_text(prefix_tokens)]
        )
    );

    run.reply(answer(prefix_tokens));
    run.at(1_002);
    let a = vec![opening("A"), turn(1), turn(2)];
    run.send(post(GENERATE, &body_of(a.clone()))).await;

    assert_eq!(
        run.take(),
        vec![
            generate_sent(body_of(vec![opening("A")])),
            create_sent(0, 10),
            generate_sent(on_prefix(PREFIX, vec![opening("B")])),
            generate_sent(on_prefix(PREFIX, vec![opening("C")])),
            generate_sent(on_prefix(PREFIX, a)),
        ]
    );
    assert_eq!(
        run.book.events(),
        vec![CacheEvent::Created {
            name: PREFIX.to_owned(),
            tokens: prefix_tokens,
            estimated: estimate(0),
            covers: 0,
            ttl_secs: 10,
            at: 1_001,
        }]
    );
    assert_eq!(
        run.book.report().reads,
        BTreeMap::from([(PREFIX.to_owned(), 3)])
    );
}

// ---------------------------------------------------------------------------
// Replies read through a cache.

/// Only a reply frame that ends the reply reports the cache's usage: a
/// partial frame, a frame that is not JSON, a finished frame without usage
/// and a failure are passed through without counting a read.
#[tokio::test]
async fn only_the_finishing_frame_with_usage_counts_a_read() {
    let run = rolled().await;
    run.take();
    let partial = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "o"}]}}],
        "usageMetadata": {"cachedContentTokenCount": 2 * estimate(4)}
    })
    .to_string();
    let unfinished_usage = json!({"candidates": [{"finishReason": "STOP"}]}).to_string();
    let frames = vec![
        Ok(WireFrame::Text(partial.clone())),
        Ok(WireFrame::Text("not json".to_owned())),
        Ok(WireFrame::Text(unfinished_usage.clone())),
        Ok(WireFrame::Text(answer_text(2 * estimate(4)))),
        Err(ProviderError::Truncated),
    ];
    run.reply(Reply::Frames(frames));
    run.at(1_022);
    let mut payload = post(STREAM, &request(7));
    payload.framing = Framing::Sse;
    let mut received = run
        .try_send(payload, Mode::Streaming)
        .await
        .expect("the reply opens");
    let last = received.pop().expect("five frames");
    assert_eq!(
        last.err().map(|error| error.to_string()),
        Some("ResponseError: the reply ended before the provider ended it".to_owned())
    );
    assert_eq!(
        texts(received),
        vec![
            partial,
            "not json".to_owned(),
            unfinished_usage,
            answer_text(2 * estimate(4)),
        ]
    );
    assert_eq!(
        run.take(),
        vec![Sent {
            method: http::Method::POST,
            uri: STREAM.to_owned(),
            body: SentBody::Json(reading(C1, 4, 7)),
        }]
    );
    assert_eq!(
        run.book.report().reads,
        BTreeMap::from([(C1.to_owned(), 2)])
    );
}

// ---------------------------------------------------------------------------
// Requests the book leaves alone.

/// Requests that are not a cacheable `generateContent` body pass through
/// unchanged: another verb, another path, a body that is not a JSON object
/// with a contents array, a multipart body, and a body that already names a
/// cache.
#[tokio::test]
async fn requests_the_book_cannot_cache_pass_through_unchanged() {
    let json_body = |uri: &str, body: Value| {
        (
            post(uri, &body),
            Sent {
                method: http::Method::POST,
                uri: uri.to_owned(),
                body: SentBody::Json(body),
            },
        )
    };
    let bytes_body = |bytes: &[u8], sent: SentBody| {
        let request = http::Request::post(GENERATE)
            .body(Body::Bytes(bytes.to_vec()))
            .expect("a request");
        (
            Encoded::new(request, Framing::Whole),
            Sent {
                method: http::Method::POST,
                uri: GENERATE.to_owned(),
                body: sent,
            },
        )
    };
    let multipart = {
        let request = http::Request::post(GENERATE)
            .body(Body::Multipart(MultipartForm::new().text("file", "x")))
            .expect("a request");
        (
            Encoded::new(request, Framing::Whole),
            Sent {
                method: http::Method::POST,
                uri: GENERATE.to_owned(),
                body: SentBody::Multipart,
            },
        )
    };
    let cases = vec![
        json_body(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:countTokens?key=test-key",
            request(3),
        ),
        json_body(CACHES, request(3)),
        json_body(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test?key=test-key",
            request(3),
        ),
        bytes_body(&b"not json"[..], SentBody::Text("not json".to_owned())),
        json_body(GENERATE, json!([request(3)])),
        json_body(GENERATE, json!({"contents": "question 1"})),
        multipart,
        json_body(
            GENERATE,
            json!({"cachedContent": "cachedContents/mine", "contents": [turn(0)]}),
        ),
    ];
    for (payload, expected) in cases {
        let run = Run::new(eager(), ThoughtReplay::All);
        run.reply(answer(0));
        assert_eq!(run.send(payload).await, vec![answer_text(0)]);
        assert_eq!(run.take(), vec![expected]);
        assert_eq!(run.book.events(), Vec::new());
    }
}

// ---------------------------------------------------------------------------
// Restored leases.

/// A restored lease that claims more contents than the request holds
/// cannot be read: the request goes out inline and counts no read.
#[tokio::test]
async fn a_restored_lease_covering_more_than_the_request_is_not_read() {
    let run = Run::new(eager(), ThoughtReplay::All);
    run.book.restore(vec![Lease {
        name: "cachedContents/restored".to_owned(),
        digest: digest(1),
        covers: 7,
        tokens: 50,
        expires_at: START + 3_600,
        model: MODEL.to_owned(),
    }]);
    run.reply(answer(0));
    run.generate(3).await;
    assert_eq!(run.take(), vec![generate_sent(request(3))]);
    assert_eq!(run.book.report().reads, BTreeMap::new());
}

/// A lease whose name cannot form a request URI is read (the name travels
/// in the body) but never extended: the extension is never sent and
/// nothing is recorded.
#[tokio::test]
async fn a_lease_whose_name_cannot_form_a_uri_is_never_extended() {
    const BAD: &str = "cachedContents/bad name";
    let lease = Lease {
        name: BAD.to_owned(),
        digest: digest(1),
        covers: 1,
        tokens: 50,
        expires_at: START + 20,
        model: MODEL.to_owned(),
    };
    let run = Run::new(eager(), ThoughtReplay::All);
    run.book.restore(vec![lease.clone()]);
    run.reply(answer(50));
    run.generate(3).await;
    // At 1,010 the lease has 10 s left, under 1.5 times the 10 s gap.
    run.reply(answer(50));
    run.at(1_010);
    run.generate(5).await;
    assert_eq!(
        run.take(),
        vec![
            generate_sent(reading(BAD, 1, 3)),
            generate_sent(reading(BAD, 1, 5)),
        ]
    );
    assert_eq!(run.book.events(), Vec::new());
    assert_eq!(run.book.leases(), vec![lease]);
}

// ---------------------------------------------------------------------------
// Resume and close.

fn alive(name: &str, display_name: Option<&str>) -> Reply {
    let mut body = json!({"name": name, "model": format!("models/{MODEL}")});
    if let Some(display_name) = display_name {
        body["displayName"] = json!(display_name);
    }
    resource(body)
}

fn restored(name: &str, digest: &str) -> Lease {
    Lease {
        name: name.to_owned(),
        digest: digest.to_owned(),
        covers: 4,
        tokens: 500,
        expires_at: START + 600,
        model: MODEL.to_owned(),
    }
}

/// Proving keeps a lease only while Google still has a cache under its
/// name whose display name ends with the lease's digest; every other lease
/// is lost.
#[tokio::test]
async fn prove_keeps_only_leases_google_still_holds_under_their_digest() {
    let digest = "ab".repeat(32);
    let leases = vec![
        restored("cachedContents/a-alive", &digest),
        restored("cachedContents/b-renamed", "b-digest"),
        restored("cachedContents/c-unnamed", "c-digest"),
        restored("cachedContents/d-gone", "d-digest"),
    ];
    let run = Run::new(eager(), ThoughtReplay::All);
    run.book.restore(leases.clone());
    run.reply(alive(
        "cachedContents/a-alive",
        Some(&format!("rig-test-{}", &digest[..40])),
    ));
    run.reply(alive("cachedContents/b-renamed", Some("someone-else")));
    run.reply(alive("cachedContents/c-unnamed", None));
    run.reply(forbidden());
    let caches = Model::new(
        CachedContents::new(GeminiConfig::new(KEY)),
        run.script.clone(),
    );

    assert_eq!(run.book.prove(&caches).await, 1);
    assert_eq!(
        run.take(),
        leases
            .iter()
            .map(|lease| Sent {
                method: http::Method::GET,
                uri: format!("{API}/v1beta/{}?key={KEY}", lease.name),
                body: SentBody::Empty,
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(run.book.leases(), vec![leases[0].clone()]);
    assert_eq!(
        run.book.events(),
        leases[1..]
            .iter()
            .map(|lease| CacheEvent::Lost {
                name: lease.name.clone(),
                at: START,
            })
            .collect::<Vec<_>>()
    );
}

/// Closing deletes every lease the book holds and retires each, whether the
/// delete succeeded, found the cache gone (403) or failed (500). A second
/// close has nothing left to delete.
#[tokio::test]
async fn close_retires_every_lease_and_a_second_close_sends_nothing() {
    let leases = vec![
        restored("cachedContents/a", "a-digest"),
        restored("cachedContents/b", "b-digest"),
        restored("cachedContents/c", "c-digest"),
    ];
    let run = Run::new(eager(), ThoughtReplay::All);
    run.book.restore(leases.clone());
    run.reply(resource(json!({})));
    run.reply(forbidden());
    run.reply(status(
        500,
        json!({"error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}}),
    ));
    let caches = Model::new(
        CachedContents::new(GeminiConfig::new(KEY)),
        run.script.clone(),
    );

    run.at(1_300);
    run.book.close(&caches).await;
    assert_eq!(
        run.take(),
        leases
            .iter()
            .map(|lease| delete_sent(&lease.name))
            .collect::<Vec<_>>()
    );
    let retired: Vec<String> = leases.iter().map(|lease| lease.name.clone()).collect();
    assert_eq!(run.book.leases(), Vec::new());
    assert_eq!(run.book.report().retired, retired);
    let events: Vec<CacheEvent> = retired
        .iter()
        .map(|name| CacheEvent::Retired {
            name: name.clone(),
            at: 1_300,
        })
        .collect();
    assert_eq!(run.book.events(), events);

    run.book.close(&caches).await;
    assert_eq!(run.take(), Vec::new());
    assert_eq!(run.book.events(), events);
}

// ---------------------------------------------------------------------------
// Debug.

#[test]
fn a_caching_transport_shows_its_book_and_replay() {
    let run = Run::new(eager(), ThoughtReplay::CurrentTurn);
    assert_eq!(
        format!("{:?}", run.caching),
        format!(
            "Caching {{ book: {:?}, thought_replay: CurrentTurn, .. }}",
            run.book
        )
    );
}
