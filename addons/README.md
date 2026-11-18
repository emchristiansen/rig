# Fork add-ons

This directory holds everything Eric's fork (`emchristiansen/rig`) adds to
upstream Rig (`0xPlaygrounds/rig`). The fork is an upstream release tag plus
the crates here. The crates build on the public API of upstream's crates and
edit none of their source, so merging the next upstream release cannot
conflict with fork code. After such a merge, only these crates need adjusting
to upstream's API changes.

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
| `Cargo.lock` | Entries for the add-on crates and their dependencies. | Follows from the line above. Regenerate rather than merge by hand. |
| `flake.nix` | `rustToolchain` reads `./addons/rust-toolchain.toml`; a `shellHook` links `target/` to a per-workspace cache under `~/.cache/cargo-target`. | The workstation's development shell. |
| `flake.lock` | The `rust-overlay` input is newer. | Provides Rust 1.100.0. |
| `.gitignore` | First line `target/` becomes `target`; the `.envrc` line is removed. | The development shell makes `target` a symlink, which `target/` does not match; `.envrc` is tracked. |
| `.cargo/config.toml` | Appended `[registry] global-min-publish-age = "14 days"` and `[resolver] incompatible-publish-age = "deny"`. | The workstation's fourteen-day rule for registry crates. |
| `.github/scripts/check-frozen-release-docs.sh` | The release documents' history is named apart from the PR's base (`UPSTREAM_REF`; in a fork's CI, upstream's `main`, fetched), and a release document that differs from the base passes when it is byte for byte what the newest upstream commit contained in the PR has. | Upstream's guard fails any PR that touches a release document, and a PR that merges an upstream release touches all of them. With the change such a PR passes and every other edit, addition or deletion of a release document still fails. The changed files are listed with rename detection off and their names read NUL-separated and matched literally: with upstream's listing, a document deleted beside an identical one the release added was reported only as the added one, and a document moved out of the set, or one whose name Git quotes, was not seen at all. In upstream's own repository the rule is unchanged and that listing is the one difference. Tested by `addons/test_release_docs_guard.py`, which the fork's workflow runs. |

Files the fork adds outside this directory: `.envrc` (`use flake`), and `.github/workflows/addons.yaml`, which runs the add-on crates' format check, lints and tests, and `addons/test_release_docs_guard.py`, on pull requests. A directory directly under `addons/` must be a crate: the workspace takes every one as a member. Upstream's own workflows compile the add-ons as workspace members but do not run their tests: its test jobs name `crates/` packages, and its all-features lane is skipped outside upstream's repository.

Upstream's `rust-toolchain.toml` is deliberately unchanged. Upstream's CI
reads it, so upstream's own format and lint checks run on the toolchain
upstream wrote them for. The development shell takes its toolchain from
`addons/rust-toolchain.toml` instead.

## Merging an upstream release

1. Merge the release tag into the fork's line. Expect conflicts only in the
   files listed above; take upstream's side and re-apply the listed change.
2. Regenerate `Cargo.lock`.
3. Build and test the add-on crates and fix what upstream's API changes broke.
4. Update "Current upstream base" above.
