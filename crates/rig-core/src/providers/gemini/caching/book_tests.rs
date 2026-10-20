//! The cache book's accounting, plan arms and request-body surgery, on a
//! clock the test sets.

use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

const MODEL: &str = "gemini-test";

/// A policy whose arithmetic a test can follow: no price discount, no
/// storage price, no minimum size or gain, implicit caching never assumed.
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

fn clocked(policy: AutoCache, clock: &Arc<AtomicU64>) -> CacheBook {
    let clock = Arc::clone(clock);
    CacheBook::new(policy).with_clock(move || clock.load(Ordering::SeqCst))
}

const SYSTEM: &str =
    r#"{"parts":[{"text":"You are the store's support agent. Answer in one sentence."}]}"#;

/// The `index`th content of a support chat: user questions and model
/// answers, alternating.
fn turn(index: usize) -> String {
    let number = index / 2 + 1;
    match index % 2 {
        0 => format!(r#"{{"role":"user","parts":[{{"text":"question {number}"}}]}}"#),
        _ => format!(r#"{{"role":"model","parts":[{{"text":"answer {number}"}}]}}"#),
    }
}

/// A request on [`SYSTEM`] carrying the chat's first `len` contents.
fn chat(len: usize, first: &str) -> Parsed {
    let mut contents: Vec<String> = (0..len).map(turn).collect();
    if let Some(opening) = contents.first_mut() {
        *opening = format!(r#"{{"role":"user","parts":[{{"text":"{first}"}}]}}"#);
    }
    let body = format!(
        r#"{{"systemInstruction":{SYSTEM},"contents":[{}]}}"#,
        contents.join(",")
    );
    parse(body.as_bytes()).expect("a generateContent body")
}

/// The book's estimate of the prefix and the first `covers` contents of
/// `parsed`.
fn estimated(parsed: &Parsed, covers: usize) -> u64 {
    estimate(SYSTEM)
        + parsed.contents[..covers]
            .iter()
            .map(|content| estimate(content.get()))
            .sum::<u64>()
}

fn line(book: &CacheBook, key: &str) -> Line {
    book.book()
        .lines
        .get(key)
        .cloned()
        .expect("the plan opened the line")
}

/// Storage runs from a cache's (first) restore until it was retired, lost
/// or expired, or until now. A cache restored again under a second digest
/// keeps its first life, and retiring it again keeps its first end.
#[test]
fn a_report_bills_storage_until_each_cache_ended_expired_or_now() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(AutoCache::default(), &clock);
    let lease = |name: &str, digest: &str, tokens, expires_at| Lease {
        name: name.to_owned(),
        digest: digest.to_owned(),
        covers: 2,
        tokens,
        expires_at,
        model: MODEL.to_owned(),
    };
    let one = lease("cachedContents/one", "d-one", 3_600, 8_200);
    let two = lease("cachedContents/two", "d-two", 7_200, 2_800);
    book.restore(vec![one.clone(), two.clone()]);
    clock.store(2_000, Ordering::SeqCst);
    let again = lease("cachedContents/one", "d-one-again", 999, 8_200);
    book.restore(vec![again.clone()]);
    clock.store(4_600, Ordering::SeqCst);
    book.retired(&one);
    clock.store(6_000, Ordering::SeqCst);
    book.retired(&again);
    clock.store(9_000, Ordering::SeqCst);

    // `one`: 3,600 tokens from 1,000 to 4,600 is 3,600 token-hours; `two`:
    // 7,200 tokens from 1,000 until it expired at 2,800 is 3,600 more.
    let report = book.report();
    assert_eq!(
        report,
        CacheReport {
            created: Vec::new(),
            reads: BTreeMap::new(),
            retired: vec!["cachedContents/one".to_owned(); 2],
            lost: Vec::new(),
            token_hours: 7_200.0,
            live: vec![two.clone()],
        }
    );
    assert_eq!(
        CacheCost::from(&report),
        CacheCost {
            storage_token_hours: 7_200.0,
            ..CacheCost::default()
        }
    );

    // Losing `two` after it expired bills nothing more.
    book.lost(&two);
    assert_eq!(
        book.report(),
        CacheReport {
            lost: vec!["cachedContents/two".to_owned()],
            live: Vec::new(),
            ..report
        }
    );
    assert_eq!(
        book.events(),
        vec![
            CacheEvent::Retired {
                name: "cachedContents/one".to_owned(),
                at: 4_600,
            },
            CacheEvent::Retired {
                name: "cachedContents/one".to_owned(),
                at: 6_000,
            },
            CacheEvent::Lost {
                name: "cachedContents/two".to_owned(),
                at: 9_000,
            },
        ]
    );
}

/// Without a clock of its own, a book reads the system clock in Unix
/// seconds.
#[test]
fn a_book_without_a_clock_reads_unix_seconds() {
    let unix = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the system clock is past the epoch")
            .as_secs()
    };
    let before = unix();
    let read = CacheBook::new(AutoCache::default()).now();
    let after = unix();
    assert!(
        (before..=after).contains(&read),
        "{read} outside {before}..={after}"
    );
}

#[test]
fn a_book_shows_its_policy_and_display_prefix() {
    let policy = AutoCache {
        min_gain: 7,
        ..AutoCache::default()
    };
    let book = CacheBook::new(policy);
    assert_eq!(book.policy(), policy);
    assert_eq!(
        format!("{book:?}"),
        format!("CacheBook {{ policy: {policy:?}, display_prefix: \"rig-cache-\", .. }}")
    );
    let book = book.with_display_prefix("team-a-");
    assert_eq!(
        format!("{book:?}"),
        format!("CacheBook {{ policy: {policy:?}, display_prefix: \"team-a-\", .. }}")
    );
}

/// A display name carries the digest's first 40 characters; a shorter
/// digest is carried whole.
#[test]
fn a_short_digest_is_the_first_forty_characters() {
    let long = "0123456789".repeat(7);
    assert_eq!(short_digest(&long), &long[..40]);
    assert_eq!(short_digest("abc"), "abc");
}

/// Every part keeps its exact bytes and the body's key order; a null prefix
/// field is absent; only a non-null `cachedContent` marks the request as
/// already reading a cache. A body that is not an object, or whose
/// `contents` is not an array, is not one the book can cache.
#[test]
fn parse_keeps_each_part_s_bytes_and_flags_a_set_cached_content() {
    let parsed = parse(
        br#"{"generationConfig":{"maxOutputTokens":8},"systemInstruction":null,"tools":[{"functionDeclarations":[]}],"contents":[{"role":"user","parts":[{"text":"q"}]}],"cachedContent":null}"#,
    )
    .expect("a generateContent body");
    assert_eq!(
        parsed
            .rest
            .iter()
            .map(|(key, value)| (key.as_str(), value.get()))
            .collect::<Vec<_>>(),
        vec![
            ("generationConfig", r#"{"maxOutputTokens":8}"#),
            ("cachedContent", "null"),
        ]
    );
    assert_eq!(
        parsed
            .prefix
            .iter()
            .map(|field| field.as_ref().map(|raw| raw.get()))
            .collect::<Vec<_>>(),
        vec![None, Some(r#"[{"functionDeclarations":[]}]"#), None]
    );
    assert_eq!(
        parsed
            .contents
            .iter()
            .map(|content| content.get())
            .collect::<Vec<_>>(),
        vec![r#"{"role":"user","parts":[{"text":"q"}]}"#]
    );
    assert!(!parsed.has_cached_content);

    let reading = parse(br#"{"cachedContent":"cachedContents/x","contents":[]}"#)
        .expect("a generateContent body");
    assert!(reading.has_cached_content);

    for refused in [&b"not json"[..], &b"[1,2]"[..], &br#"{"contents":"q"}"#[..]] {
        assert!(
            parse(refused).is_none(),
            "{}",
            String::from_utf8_lossy(refused)
        );
    }
}

/// A body that reads a cache names it first, sends only the contents after
/// what it holds, drops the prefix and any earlier `cachedContent`, and
/// keeps every other field's bytes. A lease that claims more contents than
/// the request has gives no body. A cache body carries the model, name,
/// TTL, prefix and the contents it covers, none for a prefix cache.
#[test]
fn a_read_body_and_a_cache_body_are_spliced_from_the_request_s_bytes() {
    let parsed = parse(
        br#"{"systemInstruction":{"parts":[{"text":"s"}]},"contents":[{"role":"user","parts":[{"text":"a"}]},{"role":"model","parts":[{"text":"b"}]},{"role":"user","parts":[{"text":"c"}]}],"cachedContent":null,"generationConfig":{"temperature":0}}"#,
    )
    .expect("a generateContent body");
    let lease = Lease {
        name: "cachedContents/x".to_owned(),
        digest: "d".to_owned(),
        covers: 2,
        tokens: 10,
        expires_at: 100,
        model: MODEL.to_owned(),
    };
    let read = stripped(&parsed, &lease).expect("the lease covers part of the request");
    assert_eq!(
        String::from_utf8(read).expect("UTF-8"),
        r#"{"cachedContent":"cachedContents/x","contents":[{"role":"user","parts":[{"text":"c"}]}],"generationConfig":{"temperature":0}}"#
    );
    assert!(
        stripped(
            &parsed,
            &Lease {
                covers: 4,
                ..lease.clone()
            }
        )
        .is_none()
    );

    let prefix = cache_body("gemini-test", &parsed, 0, "rig-x", 30).expect("a cache body");
    assert_eq!(
        String::from_utf8(prefix).expect("UTF-8"),
        r#"{"model":"models/gemini-test","displayName":"rig-x","ttl":"30s","systemInstruction":{"parts":[{"text":"s"}]}}"#
    );
    let conversation = cache_body("gemini-test", &parsed, 2, "rig-x", 30).expect("a cache body");
    assert_eq!(
        String::from_utf8(conversation).expect("UTF-8"),
        r#"{"model":"models/gemini-test","displayName":"rig-x","ttl":"30s","systemInstruction":{"parts":[{"text":"s"}]},"contents":[{"role":"user","parts":[{"text":"a"}]},{"role":"model","parts":[{"text":"b"}]}]}"#
    );
}

/// A request with no contents is its prefix alone: its line is the prefix
/// digest, and it reads a prefix cache when one is live.
#[test]
fn a_request_without_contents_plans_on_its_prefix() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(eager(), &clock);
    let parsed = chat(0, "unused");
    let d = digests(MODEL, &parsed);
    assert_eq!(d.len(), 1);

    let plan = book.plan(MODEL, &d, &parsed, true);
    assert_eq!(plan.read, None);
    assert!(plan.create.is_none());
    assert!(plan.retire.is_empty());
    assert!(plan.extend.is_none());
    assert_eq!(plan.line, d[0]);
    assert_eq!(plan.coverable, estimate(SYSTEM));

    let prefix = Lease {
        name: "cachedContents/prefix".to_owned(),
        digest: d[0].clone(),
        covers: 0,
        tokens: 40,
        expires_at: 2_000,
        model: MODEL.to_owned(),
    };
    book.restore(vec![prefix.clone()]);
    let plan = book.plan(MODEL, &d, &parsed, true);
    assert_eq!(plan.read, Some(prefix));
    assert!(plan.create.is_none());
    assert!(plan.extend.is_none());
    assert_eq!(plan.line, d[0]);
    assert_eq!(plan.coverable, 40);
}

/// A lease that belongs to another model is neither read nor retired, even
/// under this request's digest and long unread.
#[test]
fn another_model_s_lease_is_neither_read_nor_retired() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(eager(), &clock);
    let parsed = chat(3, "question 1");
    let d = digests(MODEL, &parsed);
    book.restore(vec![Lease {
        name: "cachedContents/foreign".to_owned(),
        digest: d[1].clone(),
        covers: 1,
        tokens: 500,
        expires_at: 10_000,
        model: "other-model".to_owned(),
    }]);
    clock.store(1_100, Ordering::SeqCst);
    let plan = book.plan(MODEL, &d, &parsed, true);
    assert_eq!(plan.read, None);
    assert!(plan.retire.is_empty());
}

/// A conversation cache unread for more than 60 seconds (its line had no
/// gap) is retired and not read, though it has not expired; at exactly 60
/// seconds it is still read.
#[test]
fn a_conversation_cache_unread_past_the_idle_limit_is_retired_not_read() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(eager(), &clock);
    let parsed = chat(3, "question 1");
    let d = digests(MODEL, &parsed);
    let lease = Lease {
        name: "cachedContents/conversation".to_owned(),
        digest: d[1].clone(),
        covers: 1,
        tokens: 500,
        expires_at: 10_000,
        model: MODEL.to_owned(),
    };
    book.restore(vec![lease.clone()]);

    clock.store(1_060, Ordering::SeqCst);
    let plan = book.plan(MODEL, &d, &parsed, true);
    assert_eq!(plan.read, Some(lease.clone()));
    assert!(plan.retire.is_empty());

    clock.store(1_061, Ordering::SeqCst);
    let plan = book.plan(MODEL, &d, &parsed, true);
    assert_eq!(plan.read, None);
    assert_eq!(plan.retire, vec![lease]);
}

/// Above the implicit ceiling the book sends inline while implicit caching
/// serves the line (no premium accrues); once its average coverage falls
/// below half, the line pays premium again and rolls as usual.
#[test]
fn past_the_implicit_ceiling_the_book_stays_inline_until_implicit_caching_fails() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let policy = AutoCache {
        implicit_floor: 10,
        implicit_ceiling: estimate(SYSTEM),
        ..eager()
    };
    let book = clocked(policy, &clock);

    let short = chat(3, "question 1");
    let d = digests(MODEL, &short);
    let plan = book.plan(MODEL, &d, &short, true);
    let coverable = estimated(&short, 2);
    assert_eq!(plan.coverable, coverable);
    assert_eq!(plan.read, None);
    assert!(plan.create.is_none());
    assert!(plan.extend.is_none());

    // Whole coverage, then none: the average is 0.5, still serving.
    book.observe(&plan.line, None, coverable, coverable);
    book.observe(&plan.line, None, coverable, 0);
    let served = line(&book, &plan.line);
    assert_eq!((served.implicit, served.premium), (Some(0.5), 0.0));
    // 0.25, then 0.125: failing, so each miss is premium.
    book.observe(&plan.line, None, coverable, 0);
    book.observe(&plan.line, None, coverable, 0);
    let failing = line(&book, &plan.line);
    assert_eq!(
        (failing.implicit, failing.premium),
        (Some(0.125), 2.0 * coverable as f64)
    );

    let long = chat(5, "question 1");
    let d = digests(MODEL, &long);
    let plan = book.plan(MODEL, &d, &long, true);
    assert_eq!(plan.read, None);
    let create = plan.create.expect("the premium pays for a cache");
    assert_eq!(
        (
            create.covers,
            create.digest,
            create.ttl_secs,
            create.estimate,
            create.expected,
            create.replaces,
        ),
        (
            4,
            d[4].clone(),
            10,
            estimated(&long, 4),
            estimated(&long, 4),
            None
        )
    );
}

/// A second conversation on a prefix caches the prefix only when it is big
/// enough and not already leased, even if that lease has expired.
#[test]
fn a_shared_prefix_is_cached_only_when_big_enough_and_not_leased() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let first = chat(1, "from A");
    let second = chat(1, "from B");
    let plans = |book: &CacheBook| {
        let a = book.plan(MODEL, &digests(MODEL, &first), &first, true);
        let b = book.plan(MODEL, &digests(MODEL, &second), &second, true);
        (a.create.is_some(), b.create.is_some())
    };

    let small = clocked(
        AutoCache {
            min_tokens: estimate(SYSTEM) + 1,
            ..eager()
        },
        &clock,
    );
    assert_eq!(plans(&small), (false, false));

    let leased = clocked(eager(), &clock);
    leased.restore(vec![Lease {
        name: "cachedContents/prefix".to_owned(),
        digest: digests(MODEL, &first)[0].clone(),
        covers: 0,
        tokens: 40,
        expires_at: 1_000,
        model: MODEL.to_owned(),
    }]);
    assert_eq!(plans(&leased), (false, false));

    let fresh = clocked(eager(), &clock);
    assert_eq!(plans(&fresh), (false, true));
}

/// An extension the book records for a cache it no longer holds changes no
/// lease, but the event stands.
#[test]
fn extending_a_cache_the_book_no_longer_holds_only_records_the_event() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(eager(), &clock);
    let gone = Lease {
        name: "cachedContents/gone".to_owned(),
        digest: "d".to_owned(),
        covers: 1,
        tokens: 10,
        expires_at: 1_005,
        model: MODEL.to_owned(),
    };
    book.extended(&gone, 30);
    assert_eq!(book.leases(), Vec::new());
    assert_eq!(
        book.events(),
        vec![CacheEvent::Extended {
            name: "cachedContents/gone".to_owned(),
            expires_at: 1_030,
            at: 1_000,
        }]
    );
}

/// Calibration learns Gemini's count over the book's estimate from each
/// created cache, as a running half-and-half average; a cache with no
/// count, or no estimate, teaches nothing.
#[test]
fn creating_a_cache_calibrates_the_estimate_only_from_counted_caches() {
    let clock = Arc::new(AtomicU64::new(1_000));
    let book = clocked(eager(), &clock);
    let create = |digest: &str, estimate| Create {
        covers: 2,
        digest: digest.to_owned(),
        ttl_secs: 10,
        estimate,
        expected: estimate,
        replaces: None,
    };

    let uncounted = book.created(
        "line",
        &create("d-a", 100),
        "cachedContents/a".to_owned(),
        0,
        MODEL,
    );
    assert_eq!(
        uncounted,
        Lease {
            name: "cachedContents/a".to_owned(),
            digest: "d-a".to_owned(),
            covers: 2,
            tokens: 0,
            expires_at: 1_010,
            model: MODEL.to_owned(),
        }
    );
    assert_eq!(book.book().calibration, None);

    book.created(
        "line",
        &create("d-b", 100),
        "cachedContents/b".to_owned(),
        200,
        MODEL,
    );
    assert_eq!(book.book().calibration, Some(2.0));
    book.created(
        "line",
        &create("d-c", 50),
        "cachedContents/c".to_owned(),
        200,
        MODEL,
    );
    assert_eq!(book.book().calibration, Some(3.0));
    book.created(
        "line",
        &create("d-d", 0),
        "cachedContents/d".to_owned(),
        100,
        MODEL,
    );
    assert_eq!(book.book().calibration, Some(3.0));
}
