//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
mod memory;
pub mod observations;
mod streaming;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod streaming_conformance;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod streaming_conformance_suite;
mod tracing_isolation;

pub use completion::{MockCompletionModel, MockError, MockTurn};
pub use embeddings::{MockEmbeddingModel, MockMultiTextDocument, MockTextDocument};
pub use http::{
    CapturedHttpRequest, CapturingStreamingClient, HttpErrorStreamingClient, MockHttpResponse,
    MockStreamingClient, NonSuccessStreamingClient, RecordingHttpClient, SequencedHttpClient,
    SequencedStreamingHttpClient,
};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};

/// A caller identity for tests of a dialect that requires one (the ChatGPT
/// dialect): plainly a test value, never an official client's.
#[allow(clippy::expect_used)]
pub fn test_caller_identity() -> crate::providers::openai::wire::CallerIdentity {
    crate::providers::openai::wire::CallerIdentity::new(
        "rig_test",
        "rig-test/0 (test identity)",
        None,
    )
    .expect("a valid test identity")
}
pub use streaming::{MOCK_PROVIDER, MockStreamEvent, mock_final, mock_final_with_total_tokens};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};
