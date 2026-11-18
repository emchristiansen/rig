# Fork add-ons

This directory holds everything Eric's fork (`emchristiansen/rig`) adds to
upstream Rig (`0xPlaygrounds/rig`). The fork is an upstream release tag plus
the crates here and the changes to upstream files listed below. The crates
build on the public API of upstream's crates and edit none of their source.
After merging the next upstream release, only these crates and the listed
changes need adjusting to upstream's changes.

Current upstream base: `v0.46.0`.

## Crates

| Crate | What it provides |
|---|---|
| `rig-live` | The typed adapter for the public Live API (`rig_live::openai`), GPT-Live realtime calls for the ChatGPT subscription (`rig_live::chatgpt`), and their shared support code (`rig_live::support`). |
| `rig-webrtc` | The WebRTC transport those realtime calls use. |
| `rig-codex` | The ChatGPT subscription's Codex contract over Rig's Responses wire: `CodexResponses`, the wire a Codex model encodes with (declared tools and `include` as the caller wrote them, the conversation identity, refused controls), `CodexHttpClient` (exact headers, zstd request compression), the Codex Responses WebSocket session, the typed Responses data, the evidence kept of a frame that failed to decode, and, behind the `completion-observations` feature, a caller-owned ledger of the original server-sent events of one streamed call (`rig_codex::responses_api::observation`). |

The crates are not published; consumers depend on them by Git revision.

They live here rather than under `crates/` because upstream's guard tests
scan `crates/` (for example `tests/core/no_random_ids.rs`) and would need
their allowlists edited for fork code. Being outside that scan exempts
nothing: `rig-live/tests/no_random_ids.rs` holds every crate here to the same
record-replay rule, with its own list of allowed sites and their reasons.

## Every upstream file the fork changes

Keep this list exact. When merging an upstream release, these are the only
places where the fork's side must be re-applied.

| File | Change | Why |
|---|---|---|
| `Cargo.toml` | One entry, `"addons/*"`, with its comment, in `[workspace] members` after `"crates/*"`. | Makes the add-on crates workspace members. |
| `Cargo.lock` | Upstream's lockfile with what Cargo adds for the add-on crates: their packages, the edges those packages' features switch on in packages upstream already locks, and a newer version of a package upstream locks only where an add-on's requirement excludes upstream's. | Follows from the line above. Every other package stays at the version upstream locked. Take upstream's file and let Cargo complete it; never merge it by hand, and never delete and regenerate it, which resolves upstream's own packages again. |
| `flake.nix` | `rustToolchain` reads `./addons/rust-toolchain.toml`; a `shellHook` links `target/` to a per-workspace cache under `~/.cache/cargo-target`. | The workstation's development shell. |
| `flake.lock` | The `rust-overlay` input is newer. | Provides Rust 1.100.0. |
| `.gitignore` | First line `target/` becomes `target`; the `.envrc` line is removed. | The development shell makes `target` a symlink, which `target/` does not match; `.envrc` is tracked. |
| `.cargo/config.toml` | Appended `[registry] global-min-publish-age = "14 days"` and `[resolver] incompatible-publish-age = "deny"`. | The workstation's fourteen-day rule for registry crates. |
| `.github/scripts/check-frozen-release-docs.sh` | The release documents' history is named apart from the PR's base (`UPSTREAM_REF`; in a fork's CI, upstream's `main`, fetched), and a release document that differs from the base passes when it is byte for byte what the newest upstream commit contained in the PR has. | Upstream's guard fails any PR that touches a release document, and a PR that merges an upstream release touches all of them. With the change such a PR passes and every other edit, addition or deletion of a release document still fails. The changed files are listed with rename detection off and their names read NUL-separated and matched literally: with upstream's listing, a document deleted beside an identical one the release added was reported only as the added one, and a document moved out of the set, or one whose name Git quotes, was not seen at all. In upstream's own repository the rule is unchanged and that listing is the one difference. Tested by `addons/test_release_docs_guard.py`, which the fork's workflow runs. |
| `.github/workflows/slow.yaml` | In the `stable / test all features` job's condition, `github.repository == '0xPlaygrounds/rig'` becomes `(github.repository == '0xPlaygrounds/rig' \|\| github.event_name == 'workflow_dispatch')`, with one sentence added to the comment above the job. | Upstream runs its whole-workspace, all-features test run only in its own repository, so that a fork's nightly schedule does not start it. That left the fork no way to run it at all. A manual dispatch, which needs write access to the fork, now runs it on the dispatched ref. There are two: of this workflow (`gh workflow run slow.yaml --ref <branch>`), and of `cd.yaml`, which calls this workflow and passes its own event on, so that dispatching `cd.yaml` by hand runs the lane as well. Pull requests, the schedule, the merge queue and `cd.yaml` on a push to `main` behave as before on the fork: skipped. The job reads no secret. The dependency-floors job is unchanged. |
| `crates/rig-core/src/completion/history.rs` | `ReplayTarget` gains `replays_when_last`, default `false`; `paired` reads it for the last block of a turn as written, so an item that needs the item after it is kept there when the target says so. | A divergence from upstream Rig's replay rule, which never sends a Responses reasoning item without the item it preceded, so a turn that holds only reasoning, or ends in it, loses that reasoning. Every target but the one below keeps upstream's rule. A reasoning item whose following block exists but is not replayed as its item is still not sent. |
| `crates/rig-core/src/completion/history.rs`, `crates/rig-core/src/operation/completion.rs`, `crates/rig-core/src/completion/request.rs` | `Completion::prepare_with_sent_ids` returns the prepared request with a `history::SentIds`: for each tool call of the history as the caller gave it, by (message, block), the call id the encoded request spells for it (`Sent::Under`) or that it is not sent as a call (`Sent::Not`); results are found in the body under their call's stated id. `adapt_stated` records the id `Renamed::claim` gave each call as `assistant` shapes it (`Landed`), and a call is stated as sent when the final adapted history holds a call under that id, spelled by `WireIds`; nothing is written into any block, and `reshaped` runs as upstream runs it. `Memo` keeps the ids of its prefix; `insert_after_leading_system` returns where the documents went (`Inserted`). `Operation::prepare` is the same path without the statement. Nothing sent changes. Tested by `crates/rig-core/src/completion/history/tests/sent_ids.rs`, a file the fork adds, declared in `history/tests.rs`. | A consumer that checks its own requests needs the ids stated, not inferred: the renamer is private, and reconstructing its result from the body admitted wrong bodies. |
| `crates/rig-core/src/providers/openai/responses_api/wire.rs` | `Responses` answers `replays_when_last` with `true` for a reasoning item on the Codex contract (`ResponsesContract::Codex`, the ChatGPT backend) only. | The fork's consumer, Muninn's Conta, serves turns that hold only reasoning and must replay them (its spec: "Reasoning continuity is preserved"); the fork's earlier line sent such items. That the ChatGPT backend accepts them, and that its own client keeps them in the history it sends, is not tested here. Tested by `addons/rig-codex/src/responses_api/trailing_reasoning_tests.rs`. |

Files the fork adds outside this directory: `.envrc` (`use flake`), `crates/rig-core/src/completion/history/tests/sent_ids.rs` (the tests of the stated call ids above), and `.github/workflows/addons.yaml`, which runs the add-on crates' format check, lints and tests, and `addons/test_release_docs_guard.py`, on pull requests. A directory directly under `addons/` must be a crate: the workspace takes every one as a member. Upstream's own workflows compile the add-ons as workspace members but do not run their tests: its test jobs name `crates/` packages, and its all-features lane runs outside upstream's repository only when dispatched by hand.

Upstream's `rust-toolchain.toml` is deliberately unchanged. Upstream's CI
reads it, so upstream's own format and lint checks run on the toolchain
upstream wrote them for. The development shell takes its toolchain from
`addons/rust-toolchain.toml` instead.

## Merging an upstream release

1. Merge the release tag into the fork's line. Expect conflicts only in the
   files listed above; take upstream's side and re-apply the listed change.
2. Take upstream's `Cargo.lock` as the release has it and let Cargo complete
   it for the add-on crates (`cargo metadata`); a `--locked` build then
   passes. Do not delete the file and regenerate it: that resolves upstream's
   own packages again, to versions upstream did not lock. Compare the result
   with upstream's by package and version, not by reading the text: the
   difference must be only what the add-ons need.
3. Build and test the add-on crates and fix what upstream's API changes broke.
4. Update "Current upstream base" above.
