//! Nothing the add-on crates mint is random.
//!
//! Upstream's `tests/core/no_random_ids.rs` holds `crates/` to the
//! record-replay rule: every value the program produces is a function of what
//! it received, and randomness is drawn only at listed sites that cannot reach
//! a request, a message or an effect record. That guard does not scan
//! `addons/`, so this one applies the same rule there, with its two patterns
//! and the other sources of nondeterminism a Rust crate can reach for. Each
//! allowed site is a file and a pattern, with the reason it is harmless.

#![allow(clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

/// What counts as a source of nondeterminism: upstream's two patterns, then
/// other random generators, clocks, and hash containers whose iteration order
/// is random per process.
const SOURCES: &[&str] = &[
    "fastrand::",
    "id::generate",
    "rand::",
    "getrandom",
    "uuid::",
    "Uuid::",
    "SystemTime::now",
    "Instant::now",
    "RandomState",
    "HashMap",
    "HashSet",
];

/// Where a source may appear (a file relative to `addons/`, and the source),
/// and why it cannot make a recorded value anything but a function of the
/// inputs.
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "rig-webrtc/src/peer.rs",
        "fastrand::",
        "the default RTP SSRC of a peer built without `LivePeerBuilder::with_ssrc`. It reaches \
         only the SDP offer and the RTP headers of sent audio, which webrtc-rs already fills \
         with ICE credentials, a DTLS fingerprint and an SDP session id it draws for every \
         connection, so no record or replay can compare them; a caller that records offers \
         supplies the SSRC",
    ),
    (
        "rig-webrtc/src/peer.rs",
        "HashSet",
        "deduplicates the answer's ICE candidates by membership only; the candidates keep the \
         answer's order, never the set's iteration order",
    ),
];

fn addons_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("rig-live lives in addons/")
        .to_path_buf()
}

fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("{} is readable: {error}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|error| panic!("{} lists: {error}", dir.display()))
                .path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Whether `line` names `source` as a path of its own: `fastrand::` is not
/// also `rand::`, and `HashSet` is not `MyHashSet`.
fn names(line: &str, source: &str) -> bool {
    line.match_indices(source).any(|(at, _)| {
        !line
            .get(..at)
            .and_then(|before| before.chars().next_back())
            .is_some_and(|before| before.is_alphanumeric() || before == '_')
    })
}

/// Code lines of `text`: comment lines are skipped, as upstream's guard does.
fn code_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

/// Every use of a source outside its allowed sites. Test code is skipped:
/// it never runs in a caller's program.
fn offenders(root: &Path) -> Vec<String> {
    let mut offenders = Vec::new();
    for path in source_files(root) {
        let relative = relative(root, &path);
        if relative.contains("/tests/") || relative.ends_with("tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{relative} is readable: {error}"));
        for (number, line) in code_lines(&text) {
            for source in SOURCES {
                let allowed = ALLOWED
                    .iter()
                    .any(|(site, allowed, _)| *site == relative && allowed == source);
                if !allowed && names(line, source) {
                    offenders.push(format!(
                        "{relative}:{}: `{source}` in {}",
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    offenders
}

#[test]
fn randomness_is_drawn_only_where_it_cannot_reach_a_record() {
    let found = offenders(&addons_root());
    assert!(
        found.is_empty(),
        "nondeterminism outside its allowed sites in addons/:\n{}",
        found.join("\n")
    );
}

#[test]
fn the_allowed_sites_are_live() {
    let root = addons_root();
    for (site, source, reason) in ALLOWED {
        let text = std::fs::read_to_string(root.join(site))
            .unwrap_or_else(|error| panic!("allowed site `{site}` ({reason}) is gone: {error}"));
        assert!(
            code_lines(&text).any(|(_, line)| names(line, source)),
            "allowed site `{site}` no longer uses `{source}` ({reason}); drop it"
        );
    }
}

#[test]
fn the_guard_detects_every_source_and_only_where_it_is_named() {
    let root = std::env::temp_dir().join(format!("rig-addons-random-guard-{}", std::process::id()));
    let crate_dir = root.join("probe/src");
    std::fs::create_dir_all(&crate_dir).expect("a probe directory");
    let probe = [
        "let a = fastrand::u32(..);",
        "use rig_core::id::generate;",
        "let b = rand::random::<u64>();",
        "getrandom::fill(&mut buf);",
        "let c = uuid::Uuid::new_v4();",
        "let d = Uuid::now_v7();",
        "let e = std::time::SystemTime::now();",
        "let f = Instant::now();",
        "let g = RandomState::new();",
        "let h: HashMap<u8, u8> = HashMap::new();",
        "let i = HashSet::<u8>::new();",
        "// fastrand::u32(..) in a comment",
        "let j = MyHashSet::new();",
    ]
    .join("\n");
    std::fs::write(crate_dir.join("lib.rs"), probe).expect("the probe is written");
    std::fs::write(
        crate_dir.join("tests.rs"),
        "let skipped = fastrand::u32(..);",
    )
    .expect("the test probe is written");
    let found = offenders(&root);
    std::fs::remove_dir_all(&root).expect("the probe is removed");

    let named = |source: &str| {
        found
            .iter()
            .filter(|offence| offence.contains(&format!("`{source}`")))
            .count()
    };
    for source in SOURCES {
        assert!(named(source) >= 1, "`{source}` is not detected: {found:?}");
    }
    assert_eq!(named("fastrand::"), 1, "{found:?}");
    assert_eq!(named("rand::"), 1, "{found:?}");
    assert_eq!(named("HashSet"), 1, "{found:?}");
    assert_eq!(found.len(), 12, "{found:?}");
}
