//! Feature matrix for raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: the driver puts the provider's reply body,
//! parsed as JSON, onto [`rig::completion::CompletionResponse::raw`] — here
//! Gemini's own `generateContent` document, verbatim, read back as JSON.
//! There is no opt-in and nothing
//! about it reaches the wire; `raw` is required at construction, so there is
//! no response without the document that produced it and no way for
//! capture to be "not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the normalized response. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns (record mode writes the fixture on the way out): a cell
//! whose recording lost the wire shape it claims to cover must fail loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 3 | `raw_exposes_forced_function_call` | forced tool call (`ToolChoice::Specific`) | `raw` reads back; `candidates[0].content.parts[].functionCall` and `finishMessage` == fixture; raw `finishReason` spelled `"STOP"` while `finish_reason() == ToolCalls` | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cell 3 covers a wire shape a text turn never produces. A forced
//! `functionCall` turn is where Gemini's own `finishReason` (`"STOP"`, even
//! on a call-only turn) and rig's normalized `ToolCalls` visibly disagree,
//! so `raw` must keep the wire spelling while the normalized response reports
//! the upgraded reason.

use rig::completion::CompletionResponse as RigCompletionResponse;
use rig::message::ToolChoice;
use rig::tool::Tool;
use serde_json::Value;

use super::super::request_witness_support::{
    NamedCase, model_result, with_gemini_request_witness_cassette,
};
use crate::raw_capture::capture_completion;
use crate::support::{Adder, Observed};
use rig::completion::CompletionRequest;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded turn is a `functionCall` part.
fn forced_tool_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new(Adder::NAME).expect("tool name")],
        })
}

/// The premise of the forced-tool cell: the recorded body's first candidate
/// carries a `functionCall` part naming `add`, and Gemini still spelled the
/// finish reason `"STOP"` — the wire shape rig upgrades to `ToolCalls`.
fn assert_recorded_function_call_body(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "{scenario}: Gemini spells a call-only turn's finishReason STOP; this cell exists to \
         show raw keeps that spelling while the normalized reason is ToolCalls"
    );
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{scenario}: the recorded candidate should carry parts"));
    assert!(
        parts
            .iter()
            .any(|part| part.pointer("/functionCall/name") == Some(&Value::String("add".into()))),
        "{scenario}: the recorded turn should carry a functionCall part naming `add`, the wire \
         shape this cell reads through `raw`; got {parts:?}"
    );
    body
}

// ---------------------------------------------------------------------------
// 1: the document is recoverable, and tells the same story
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 2: an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 3: a forced tool call keeps the wire's functionCall and finishReason
// ---------------------------------------------------------------------------

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. The recorded premise (a `functionCall` naming `add`, `finishReason`
/// spelled `"STOP"`, a non-empty `finishMessage`) is still read from the
/// fixture. Named losses, all read from the refused reply: `raw` carrying the
/// wire's `functionCall` parts, `finishReason` and `finishMessage` exactly;
/// the normalized `ToolCalls` finish and typed `add(2, 3)` call; the
/// canonical choice without `functionCall`; `totalTokenCount` agreeing with
/// usage; and the typed `GeminiExt` extras (model version, response id
/// `Ag7BauSXOKisz7IPlZS6iAg`, service tier, finish message, 67 prompt text
/// tokens, no Interactions id).
#[tokio::test]
async fn raw_exposes_forced_function_call() -> anyhow::Result<()> {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_forced_function_call";
    with_gemini_request_witness_cassette(
        "raw_capture_matrix/raw_exposes_forced_function_call",
        NamedCase::RawForcedFunctionCall,
        |client, _clock| async move {
            let observed: Observed<RigCompletionResponse> = Observed::default();
            Ok(model_result(
                capture_completion(client.completion(MODEL), forced_tool_request(), observed).await,
            ))
        },
    )
    .await?;

    let body = assert_recorded_function_call_body(SCENARIO);
    // Gemini annotates a call-only STOP with a `finishMessage`.
    anyhow::ensure!(
        body.pointer("/candidates/0/finishMessage")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty()),
        "{SCENARIO}: the recorded call-only turn should carry Gemini's finishMessage"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 4: a structured-output turn keeps the provider-only fields
// ---------------------------------------------------------------------------
