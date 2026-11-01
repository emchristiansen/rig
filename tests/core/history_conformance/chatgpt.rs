//! The ChatGPT (Codex) dialect of the Responses wire: every system message
//! goes to `instructions`, and the gateway always answers with a stream.
//! The dialect sends nothing without the caller's exact identity, so the
//! wire carries the test identity.

use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::streaming::IncompleteTerminal;
use rig_core::providers::openai::responses_api::wire::Responses;

use super::openai_responses::ResponsesHistory;

fn wire(model: &str) -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token")
            .with_caller_identity(rig_core::test_utils::test_caller_identity()),
        model,
    )
}

rig_history_conformance::history_conformance_suite! {
    wire: "chatgpt",
    fixture: ResponsesHistory {
        wire,
        model: chatgpt::GPT_5_4,
        other_model: chatgpt::GPT_5_3_CODEX,
        text_only_model: Some(chatgpt::GPT_5_3_CODEX_SPARK),
        reads_indexless: true,
        buffered: true,
        calls_at_terminal: true,
        accepting: |wire| wire.with_streamed_incomplete(IncompleteTerminal::Accept),
    },
}
