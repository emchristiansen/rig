//! Gemini history survival: thought signatures and function-call ids across
//! three prompts, plus an image tool result.
//!
//! The recordings send each declaration as `parametersJsonSchema`, which the
//! port's projection sends as `parameters`, so in replay each cell is an
//! exact named request-mismatch witness for the shared driver's first
//! request. `history_survival::driver::run` panics on that refusal, so the
//! witness rebuilds its first request from the driver's inputs: its preamble,
//! first prompt and tool declarations are private there and restated below,
//! and the witness proves them equal to the recording. The driver's task and
//! wire assertions (`assert_run`, `assert_recorded`) are named losses.

use rig::completion::{GenerationOptions, ProviderOptions, Reasoning};
use rig::providers::gemini::extension::GeminiOptions;
use rig::tool::Tool;
use serde_json::{Value, json};

use super::super::request_witness_support::{
    Attempt, NamedCase, stream_result, with_gemini_request_witness_cassette,
};
use crate::history_survival::Options;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig_test_support::cassette_models::GeminiModels;

/// Raw parameters for the image cell: its model takes a thinking level, so
/// the typed budget is refused there.
fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({
        "generationConfig": {
            "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
        }
    }))
}

fn no_params() -> Option<serde_json::Value> {
    None
}

fn options() -> Options {
    Options::new(
        GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 1024 }),
        ProviderOptions::new().set(GeminiOptions::new().include_thoughts(true)),
    )
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "gemini",
        model: "gemini-2.5-flash",
        params: no_params,
        options,
        max_tokens: 4096,
        transport,
        expect,
    }
}

/// `gemini-2.5-flash` answers an image inside a function response with
/// "Multimodal function responses are not supported for this model"; the
/// image cell uses the model the multimodal tool-result regression records.
const fn image_cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        model: rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW,
        params,
        options: Options::none,
        ..cell(transport, expect)
    }
}

/// The driver's preamble.
const PREAMBLE: &str = "You are a records assistant. Use the tools for every lookup and \
    verification and never guess a code. When a tool reports a transient failure, call it \
    again once. Answer briefly.";

/// The driver's first prompt for a cell with and without the image tool.
fn first_prompt(with_image: bool) -> &'static str {
    if with_image {
        "Fetch the reference snapshot, then look up the alpha record and the beta record, \
         verify each returned code, and report both codes with their verification status."
    } else {
        "Look up the alpha record and the beta record, verify each returned code, and report \
         both codes with their verification status."
    }
}

/// A tool never run: the refused first request is sent before any call.
#[derive(Debug, thiserror::Error)]
#[error("the driver's tool is declared only; the refused request runs no tool")]
struct DeclaredOnly;

/// The driver's `lookup_record` declaration.
struct Lookup;

impl Tool for Lookup {
    const NAME: &'static str = "lookup_record";
    type Error = DeclaredOnly;
    type Args = Value;
    type Output = String;

    fn description(&self) -> String {
        "Look up a record by name and return its code.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "record": { "type": "string", "description": "alpha or beta" } },
            "required": ["record"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(DeclaredOnly)
    }
}

/// The driver's `verify_code` declaration.
struct Verify;

impl Tool for Verify {
    const NAME: &'static str = "verify_code";
    type Error = DeclaredOnly;
    type Args = Value;
    type Output = String;

    fn description(&self) -> String {
        "Verify a code returned by lookup_record. May fail transiently; retry once.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "code": { "type": "string", "description": "the code to verify" } },
            "required": ["code"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(DeclaredOnly)
    }
}

/// The driver's `reference_snapshot` declaration.
struct Snapshot;

impl Tool for Snapshot {
    const NAME: &'static str = "reference_snapshot";
    type Error = DeclaredOnly;
    type Args = Value;
    type Output = String;

    fn description(&self) -> String {
        "Return the reference snapshot image for the records.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(DeclaredOnly)
    }
}

/// The driver's first prompt for `cell`, on the driver's agent.
async fn first_prompt_of(client: GeminiModels, cell: Cell) -> Attempt {
    let mut builder = rig::AgentBuilder::new(client.completion(cell.model))
        .preamble(PREAMBLE)
        .max_tokens(cell.max_tokens)
        .default_max_turns(8);
    if let Some(params) = (cell.params)() {
        builder = builder.additional_params(params);
    }
    let builder = (cell.options)()
        .configure(builder)
        .tool(Lookup)
        .tool(Verify);
    let agent = if cell.expect.image_tool_result {
        builder.tool(Snapshot).build()
    } else {
        builder.build()
    };
    let prompt = first_prompt(cell.expect.image_tool_result);
    let result = match cell.transport {
        Transport::Unary => Ok(agent.chat(prompt, &mut Vec::new()).await.map(|_| ())),
        Transport::Streaming => {
            let mut stream = agent
                .prompt(prompt)
                .history(Vec::<rig::completion::Message>::new())
                .max_turns(8)
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            result
        }
    };
    drop(agent);
    result
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0 (`lookup_record`) and 1
/// (`verify_code`) send their recorded `parametersJsonSchema`, value
/// unchanged, as `parameters`, and replay refuses the first streamed
/// request. Named losses: the three-prompt task (both codes reported, the
/// transient verifier failure recovered, every call answered), the
/// normalized history's signed shape (`Expect::SIGNED`), and the recorded
/// survival rule `assert_recorded`.
#[tokio::test]
async fn streaming() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "history_survival_matrix/streaming",
        NamedCase::HistorySurvivalStreaming,
        |client, _clock| first_prompt_of(client, cell(Transport::Streaming, Expect::SIGNED)),
    )
    .await
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0 (`lookup_record`), 1
/// (`verify_code`) and 2 (`reference_snapshot`) send their recorded
/// `parametersJsonSchema` as `parameters`, as named in
/// `NamedCase::HistorySurvivalImageToolResult`, and replay refuses the first
/// unary request. Named losses: the three-prompt task (both codes reported,
/// the transient verifier failure recovered, every call answered, the
/// snapshot served), the history's signatures and image tool result
/// (`Expect::SIGNATURES.with_image()`), and the recorded survival rule
/// `assert_recorded`.
#[tokio::test]
async fn unary_image_tool_result() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "history_survival_matrix/unary_image_tool_result",
        NamedCase::HistorySurvivalImageToolResult,
        |client, _clock| {
            first_prompt_of(
                client,
                image_cell(Transport::Unary, Expect::SIGNATURES.with_image()),
            )
        },
    )
    .await
}
