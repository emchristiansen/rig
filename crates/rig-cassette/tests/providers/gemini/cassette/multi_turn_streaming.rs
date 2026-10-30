//! Migrated from `examples/multi_turn_streaming_gemini.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::providers::gemini;
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use super::super::request_witness_support::{
    NamedCase, stream_result, with_gemini_request_witness_cassette,
};
use crate::support::MULTI_TURN_STREAMING_PROMPT;

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declarations 0-3 send their recorded schemars
/// `parametersJsonSchema` as `parameters`, less exactly the members the
/// projected subset does not carry (named in
/// `NamedCase::RunnerMultiTurnStreaming`), and replay refuses interaction 0.
/// Named losses: a substantial final response mentioning
/// `MULTI_TURN_STREAMING_EXPECTED_RESULT`, and add, subtract, multiply and
/// divide each being called at least once.
#[tokio::test]
async fn runner_driven_multi_turn_streaming_loop() -> anyhow::Result<()> {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let divide_calls = Arc::new(AtomicUsize::new(0));

    with_gemini_request_witness_cassette(
        "multi_turn_streaming/manual_multi_turn_streaming_loop",
        NamedCase::RunnerMultiTurnStreaming,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble("You must use tools to answer arithmetic prompts.")
                    .tool(Add::new(add_calls.clone()))
                    .tool(Subtract::new(subtract_calls.clone()))
                    .tool(Multiply::new(multiply_calls.clone()))
                    .tool(Divide::new(divide_calls.clone()))
                    .build();

            let mut stream = agent
                .prompt(MULTI_TURN_STREAMING_PROMPT)
                .max_turns(10)
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            drop(agent);
            result
        },
    )
    .await
}

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

struct Add {
    call_count: Arc<AtomicUsize>,
}

impl Add {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

struct Subtract {
    call_count: Arc<AtomicUsize>,
}

impl Subtract {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}

struct Multiply {
    call_count: Arc<AtomicUsize>,
}

impl Multiply {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Multiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Compute the product of x and y (i.e.: x * y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x * args.y)
    }
}

struct Divide {
    call_count: Arc<AtomicUsize>,
}

impl Divide {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Compute the quotient of x and y.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x / args.y)
    }
}
