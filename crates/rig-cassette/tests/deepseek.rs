#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_test_support::cache_conformance;
#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[allow(
    dead_code,
    reason = "each provider's witnesses use their own subset of the shared engine"
)]
#[path = "common/chat_request_witness.rs"]
mod chat_request_witness;
use rig_test_support::cassettes;
use rig_test_support::raw_capture;
use rig_test_support::reasoning;
use rig_test_support::support;

#[path = "providers/deepseek/mod.rs"]
mod deepseek;

use rig_test_support::goldens;

#[allow(
    dead_code,
    reason = "each provider exercises its own subset of matrix cells"
)]
#[path = "common/corpus_matrix.rs"]
mod corpus_matrix;

use rig_test_support::history_survival;
use rig_test_support::matrix;
