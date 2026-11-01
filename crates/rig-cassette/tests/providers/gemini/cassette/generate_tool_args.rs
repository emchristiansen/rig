//! Gemini function-call argument shape regression tests.
//!
//! Locks down how functionCall args survive the generateContent wire format:
//! deeply nested objects with arrays, non-ASCII/escaped strings, and optional
//! (nullable) fields, in both streaming and non-streaming form. Zero-argument
//! calls are already covered by `agent_tools` and `streaming_tools`.
//!
//! The recordings send each schema as `parametersJsonSchema`, which the
//! port's projection sends as `parameters`, so in replay every cell is an
//! exact named request-mismatch witness and the argument shapes it pinned
//! are named losses.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::{Message, ToolDefinition};
use rig::providers::gemini;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

use super::super::request_witness_support::{
    NamedCase, model_result, model_stream_result, with_gemini_request_witness_cassette,
};
use rig::completion::CompletionRequest;

const NESTED_ARGS_PREAMBLE: &str = "\
You are a travel booking assistant. Use the plan_trip tool for every booking request \
and copy the requested values into the tool arguments exactly as given.";

const NESTED_ARGS_PROMPT: &str = "\
Book this trip by calling the plan_trip tool exactly once with these exact values: \
itinerary.city = \"Kyoto\", itinerary.days = 3, \
itinerary.activities = [\"temples\", \"tea ceremony\"], \
itinerary.lodging.name = \"Sakura Inn\", itinerary.lodging.rooms = 2. \
After the tool returns, repeat its confirmation code in one short sentence.";

#[derive(Debug, thiserror::Error)]
#[error("Trip planning failed")]
struct PlanTripError;

#[derive(Deserialize)]
struct Lodging {
    name: String,
    rooms: u32,
}

#[derive(Deserialize)]
struct Itinerary {
    city: String,
    days: u32,
    activities: Vec<String>,
    lodging: Lodging,
}

#[derive(Deserialize)]
struct PlanTripArgs {
    itinerary: Itinerary,
}

struct PlanTrip;

impl Tool for PlanTrip {
    const NAME: &'static str = "plan_trip";
    type Error = PlanTripError;
    type Args = PlanTripArgs;
    type Output = String;

    fn description(&self) -> String {
        "Book a trip from a nested itinerary object.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        plan_trip_parameters()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "Booked {} for {} day(s), {} room(s) at {}, with {} planned activities. \
             Confirmation code SAKURA-77.",
            args.itinerary.city,
            args.itinerary.days,
            args.itinerary.lodging.rooms,
            args.itinerary.lodging.name,
            args.itinerary.activities.len(),
        ))
    }
}

fn plan_trip_parameters() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "itinerary": {
                "type": "object",
                "description": "The trip to book.",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                    "activities": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "lodging": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "rooms": { "type": "integer" }
                        },
                        "required": ["name", "rooms"]
                    }
                },
                "required": ["city", "days", "activities", "lodging"]
            }
        },
        "required": ["itinerary"]
    })
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. Named losses: the final answer repeating SAKURA-77 and the history's
/// plan_trip call carrying the nested itinerary (city Kyoto, 3 days,
/// activities `["temples", "tea ceremony"]`, lodging Sakura Inn with 2
/// rooms).
#[tokio::test]
async fn nested_arguments_roundtrip_nonstreaming() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "generate_tool_args/nested_arguments_roundtrip_nonstreaming",
        NamedCase::NestedArgumentsUnary,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(NESTED_ARGS_PREAMBLE)
                    .temperature(0.0)
                    .tool(PlanTrip)
                    .default_max_turns(4)
                    .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(NESTED_ARGS_PROMPT, &mut history)
                .await
                .map(|_| ());
            drop(agent);
            Ok(result)
        },
    )
    .await
}

/// Exact named mismatch witness on the streamed model call: the recording is
/// unchanged; the only request difference is that declaration 0's recorded
/// `parametersJsonSchema` is sent, value unchanged, as `parameters`, and
/// replay refuses interaction 0.
/// Named losses: a stream without errors whose plan_trip call carries the
/// nested itinerary (city Kyoto, 3 days, activities
/// `["temples", "tea ceremony"]`, lodging Sakura Inn with 2 rooms).
#[tokio::test]
async fn nested_arguments_streaming() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "generate_tool_args/nested_arguments_streaming",
        NamedCase::NestedArgumentsStreamed,
        |client, _clock| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request = CompletionRequest::new(NESTED_ARGS_PROMPT)
                .preamble(NESTED_ARGS_PREAMBLE.to_string())
                .temperature(0.0)
                .tool(rig::tool::tool_definition(&PlanTrip));

            model_stream_result(model.stream(request)).await
        },
    )
    .await
}

/// Exact named mismatch witness on the unary model call: the recording is
/// unchanged; the only request difference is that declaration 0's recorded
/// `parametersJsonSchema` (with the `note` member's `nullable: true`) is
/// sent, value unchanged, as `parameters`, and replay refuses interaction 0. Named losses: the reply's log_event call with
/// `name` "deploy" and `note` omitted or null.
#[tokio::test]
async fn optional_nullable_argument_omitted_when_not_requested() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "generate_tool_args/optional_nullable_argument_omitted_when_not_requested",
        NamedCase::OptionalNullableArgument,
        |client, _clock| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request = CompletionRequest::new(
                "Log an event named \"deploy\" using the log_event tool. \
                     Do not attach a note.",
            )
            .preamble(
                "Use the log_event tool for every logging request. Only fill optional \
                     arguments when the user explicitly provides them."
                    .to_string(),
            )
            .temperature(0.0)
            .tool(ToolDefinition {
                name: rig_core::message::ToolName::new("log_event").expect("tool name"),
                description: "Record an event with an optional free-form note.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "note": {
                            "type": "string",
                            "nullable": true,
                            "description": "Optional note; omit when the user gives none."
                        }
                    },
                    "required": ["name"]
                }),
            });

            Ok(model_result(model.call(request).await))
        },
    )
    .await
}
