//! Request-side pinning of tool definitions: the cassette replay matcher
//! compares request bodies canonically, so these tests freeze the exact
//! `tools` JSON Gemini receives — rich schema conversion, `#[rig_tool]` macro
//! output, and duplicate-name registration semantics. If schema generation
//! changes (e.g. swapping the handrolled definitions for rmcp-derived ones),
//! replay fails with a body mismatch.

use rig::completion::Message;
use rig::providers::gemini;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::super::request_witness_support::{NamedCase, with_gemini_request_witness_cassette};
use super::super::tools_support::MathError;

#[derive(Deserialize, Serialize)]
struct TripArgs {
    destination: String,
    travellers: u64,
    transport: String,
    waypoints: Vec<String>,
    notes: Option<String>,
}

/// Tool with a deliberately rich JSON schema (nested descriptions, enum,
/// array of strings, optional field) to pin Gemini's schema conversion.
#[derive(Clone, Default)]
struct PlanTrip;

impl Tool for PlanTrip {
    const NAME: &'static str = "plan_trip";
    type Error = MathError;
    type Args = TripArgs;
    type Output = String;

    fn description(&self) -> String {
        "Plan a trip itinerary.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination city"
                },
                "travellers": {
                    "type": "integer",
                    "description": "Number of travellers"
                },
                "transport": {
                    "type": "string",
                    "enum": ["train", "car", "plane"],
                    "description": "Mode of transport"
                },
                "waypoints": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Cities to pass through, in order"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-form notes"
                }
            },
            "required": ["destination", "travellers", "transport", "waypoints"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "itinerary booked: {} travellers to {} by {} via {}",
            args.travellers,
            args.destination,
            args.transport,
            args.waypoints.join(" -> ")
        ))
    }
}

/// A schema with nested objects, arrays and enums is one the projected
/// `parameters` subset represents faithfully, so Gemini receives it as
/// `parameters`, not as the recorded `parametersJsonSchema`.
///
/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. Named losses: the plan_trip call deserializing and executing ("itinerary
/// booked: 2 travellers to Lyon by train via Dijon -> Macon") and the
/// non-empty confirmation, both from the refused and later interactions.
#[tokio::test]
async fn rich_json_schema_reaches_gemini_unchanged() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "tool_definitions/rich_json_schema_reaches_gemini_unchanged",
        NamedCase::RichJsonSchema,
        |client, _clock| async move {
            let agent = rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                .preamble("You are a travel planner. You must use the plan_trip tool to plan trips, then confirm the booking to the user.")
                .temperature(0.0)
                .tool(PlanTrip)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let result = agent
                .chat(
                    "Plan a trip for 2 travellers to Lyon by train, passing through Dijon and then Macon, no notes.",
                    &mut history,
                )
                .await
                .map(|_| ());
            drop(agent);
            Ok(result)
        },
    )
    .await
}

mod derive_macro {
    use rig::completion::Message;
    use rig::rig_tool;

    use super::super::super::request_witness_support::{
        NamedCase, with_gemini_request_witness_cassette,
    };

    use rig::providers::gemini;

    #[rig_tool(
        description = "Perform basic arithmetic operations",
        params(
            x = "First number in the calculation",
            y = "Second number in the calculation",
            operation = "The operation to perform (add, subtract, multiply, divide)"
        )
    )]
    async fn macro_calculator(
        x: i64,
        y: i64,
        operation: String,
    ) -> Result<i64, rig::tool::ToolExecutionError> {
        match operation.as_str() {
            "add" => Ok(x + y),
            "subtract" => Ok(x - y),
            "multiply" => Ok(x * y),
            "divide" => Ok(x / y),
            _ => Err(rig::tool::ToolExecutionError::other(format!(
                "Unknown operation: {operation}"
            ))),
        }
    }

    /// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
    /// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the
    /// only request difference is that declaration 0's recorded
    /// `parametersJsonSchema` is sent as `parameters`, less exactly the
    /// schemars members the projected subset does not carry (named in
    /// `NamedCase::MacroToolSchema`), and replay refuses interaction 0. Named
    /// losses: the macro tool executing with the model's arguments (result
    /// "42") and the final answer reporting 42.
    #[tokio::test]
    async fn rig_tool_macro_schema_round_trips() -> anyhow::Result<()> {
        with_gemini_request_witness_cassette(
            "tool_definitions/rig_tool_macro_schema_round_trips",
            NamedCase::MacroToolSchema,
            |client, _clock| async move {
                let agent = rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble("You must use the macro_calculator tool for arithmetic, then report the result.")
                    .temperature(0.0)
                    .tool(MacroCalculator)
                    .default_max_turns(3)
                    .build();

                let mut history = Vec::<Message>::new();
                let result = agent
                    .chat("Multiply 6 by 7.", &mut history)
                    .await
                    .map(|_| ());
                drop(agent);
                Ok(result)
            },
        )
        .await
    }
}
