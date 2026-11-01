#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

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
use rig_test_support::support;

#[path = "providers/groq/mod.rs"]
mod groq;

use rig_test_support::history_survival;
use rig_test_support::matrix;
