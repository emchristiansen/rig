//! OpenAI's own Chat Completions: an answer's audio, whose transcript is
//! the text, and a tool call.

use rig_core::completion::CompletionRequest;
use rig_core::error::ProviderError;
use rig_core::message::EMPTY_RESPONSE_ERROR;
use rig_core::providers::openai::wire::OPENAI;
use rig_core::wire::Mode;
use rig_history_conformance::{HistoryFixture, Shape, decode};
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "audio": {"id": "audio_1", "transcript": "looking it up", "data": "UklG", "expires_at": 1},
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "content": null,
            "audio": {"id": "audio_1", "transcript": "looking "}}),
        json!({"audio": {"transcript": "it up", "data": "UklG", "expires_at": 1}}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &OPENAI,
    model: "gpt-4o-audio-preview",
    other_model: "gpt-4.1-nano",
    text_only_model: Some("o3-mini"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

/// Rich replies keep whole/stream parity; an empty unary reply is refused.
fn h03_openai_chat_stream_equals_whole(fixture: &ChatHistory) {
    let wire = fixture.wire(fixture.model());
    for shape in [Shape::Rich, Shape::Interleaved, Shape::Unknown] {
        let (Some(whole), Some(streamed)) = (
            fixture.reply(shape, Mode::Unary),
            fixture.reply(shape, Mode::Streaming),
        ) else {
            assert!(shape != Shape::Rich, "every wire supplies the rich reply");
            continue;
        };
        rig_core::test_utils::history::assert_restated_agrees(&wire, whole, streamed);
    }
    if let (Some(whole), Some(streamed)) = (
        fixture.empty_reply(Mode::Unary),
        fixture.empty_reply(Mode::Streaming),
    ) {
        let unary = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            whole,
        );
        assert!(
            matches!(&unary, Err(ProviderError::Response(message)) if message == EMPTY_RESPONSE_ERROR),
            "Unary: an empty reply has the exact shared refusal: {unary:?}"
        );
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Streaming,
            streamed,
        );
        assert!(
            response.is_ok(),
            "Streaming: an empty reply decodes: {response:?}"
        );
        if let Ok(response) = response {
            assert!(
                !response.stop().is_failure(),
                "Streaming: an empty reply that ended cleanly is a success: {:?}",
                response.stop()
            );
        }
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "openai_chat",
    fixture: FIXTURE,
    h03_stream_equals_whole: h03_openai_chat_stream_equals_whole,
}
