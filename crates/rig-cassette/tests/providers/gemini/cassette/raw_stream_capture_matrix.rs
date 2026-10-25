//! Feature matrix for raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: the GenerateContent reassembler rebuilds the
//! `generateContent` body the same turn has unary from the stream's chunks
//! (parts joined, the last finish, usage and ids) and the driver puts it
//! onto the terminal [`rig::completion::CompletionResponse::raw`]. There is
//! no opt-in and nothing about it reaches the wire.
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the terminal record. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded stream must end with a frame carrying
//! `usageMetadata` and a natural `finishReason`, or the terminal it asserts on
//! is not the one this matrix is about.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 3 | `raw_terminal_keeps_stop_on_forced_function_call` | forced tool call (`ToolChoice::Specific`), streamed | terminal `raw` is the unary document; raw `finishReason` spelled `"STOP"` and `finishMessage` == wire while the normalized terminal reports `ToolCalls`; raw carries the streamed `functionCall`; the typed extras equal the unary recording's | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `streamGenerateContent` route.
//!
//! Usage is cumulative per chunk and `finishReason` arrives on the last
//! content frame, so the "wire" side of each premise is the last frame that
//! carries `usageMetadata`, read with the same `data:` framing the streaming
//! tests use.
//!
//! Cell 3 is the tool-turn twin: Gemini spells a call-only turn's finish
//! `"STOP"` on the wire and the decoder reconciles the terminal to
//! `ToolCalls` from the tool call it saw, so this is the one place the
//! terminal `raw` and the normalized terminal legitimately disagree — `raw`
//! must keep the wire spelling and the terminal must report the upgrade.

use rig::message::ToolChoice;
use rig::tool::Tool;
use serde_json::Value;

use super::super::request_witness_support::{
    NamedCase, model_stream_result, with_gemini_request_witness_cassette,
};
use crate::support::Adder;
use rig::completion::CompletionRequest;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded stream stays short.
const MODEL: &str = "gemini-2.5-flash-lite";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded stream carries a `functionCall`.
fn forced_tool_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new(Adder::NAME).expect("tool name")],
        })
}

/// The premise of the forced-tool cell: the recorded stream carries a
/// `functionCall` part naming `add`, and its finish is still spelled `"STOP"`
/// — the wire shape the adapter's terminal mapping reconciles to `ToolCalls`. Returns the
/// last frame carrying `usageMetadata`, as [`last_usage_frame`] does.
fn last_usage_frame_of_function_call_stream(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let called_add = frames.iter().any(|frame| {
        frame
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.pointer("/functionCall/name") == Some(&Value::String("add".into()))
                })
            })
    });
    assert!(
        called_add,
        "{scenario}: no recorded frame carries a functionCall part naming `add`, so this cell \
         does not exercise the tool-turn shape it claims to cover"
    );
    let stopped = frames.iter().any(|frame| {
        frame.pointer("/candidates/0/finishReason") == Some(&Value::String("STOP".to_string()))
    });
    assert!(
        stopped,
        "{scenario}: Gemini spells a call-only turn's finishReason STOP; this cell exists to \
         show the terminal raw keeps that spelling while the normalized terminal reports \
         ToolCalls"
    );
    frames
        .iter()
        .rev()
        .find(|frame| frame.get("usageMetadata").is_some())
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: no recorded frame carries usageMetadata"))
}

// ---------------------------------------------------------------------------
// 3: a forced tool call keeps the wire's STOP while the terminal says ToolCalls
// ---------------------------------------------------------------------------

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. The recorded premise (a frame with a `functionCall` naming `add`, a
/// `"STOP"` finish, a last usage frame with a non-empty `finishMessage`) is
/// still read from the fixture. Named losses, all read from the refused
/// stream: the typed `add(2, 3)` call; the terminal `raw` carrying the
/// streamed call part, the response id, `"STOP"`, the last frame's usage and
/// `finishMessage`; the normalized `ToolCalls` terminal; and the typed
/// `GeminiExt` extras equalling the unary recording's.
#[tokio::test]
async fn raw_terminal_keeps_stop_on_forced_function_call() -> anyhow::Result<()> {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call";
    with_gemini_request_witness_cassette(
        "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call",
        NamedCase::RawStreamForcedFunctionCall,
        |client, _clock| async move {
            model_stream_result(client.completion(MODEL).stream(forced_tool_request())).await
        },
    )
    .await?;

    let last = last_usage_frame_of_function_call_stream(SCENARIO);
    // Gemini annotates a call-only STOP with a `finishMessage`.
    anyhow::ensure!(
        last.pointer("/candidates/0/finishMessage")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty()),
        "{SCENARIO}: the recorded call-only turn should carry Gemini's finishMessage"
    );
    Ok(())
}
