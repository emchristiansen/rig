//! Golden serialization of the session configuration. These are unit tests,
//! not cassette tests: the configuration is a request body built before any
//! traffic, and its expected JSON is taken from the Live API reference and
//! guides.

use super::*;
use serde_json::json;

fn to_json(value: &impl Serialize) -> Value {
    serde_json::to_value(value).expect("serializable")
}

#[test]
fn a_new_session_sends_only_the_model() {
    assert_eq!(
        to_json(&SessionConfig::new()),
        json!({"model": "gpt-live-1"})
    );
    assert_eq!(SessionConfig::default(), SessionConfig::new());
}

#[test]
fn the_reference_create_example_serializes_exactly() {
    let session =
        SessionConfig::new().with_instructions("Be concise. Ask for clarification when needed.");
    assert_eq!(
        to_json(&session),
        json!({
            "model": "gpt-live-1",
            "instructions": "Be concise. Ask for clarification when needed."
        })
    );
}

#[test]
fn client_delegation_and_voice_serialize_as_the_reference_sip_example() {
    let session = SessionConfig::new()
        .with_instructions("Help the user schedule an appointment.")
        .with_voice(Voice::Marin)
        .with_delegation(Delegation::Client);
    assert_eq!(
        to_json(&session),
        json!({
            "model": "gpt-live-1",
            "instructions": "Help the user schedule an appointment.",
            "audio": {"output": {"voice": "marin"}},
            "delegation": {"type": "client"}
        })
    );
}

#[test]
fn the_webrtc_guide_responses_delegation_serializes_exactly() {
    let backend = ResponsesDelegation::new("gpt-5.6-terra").with_settings(ResponsesSettings {
        instructions: Some(
            "Use web search when current facts are needed. Return concise, grounded results for a spoken conversation."
                .to_owned(),
        ),
        tools: Some(vec![DelegationTool::WebSearch]),
        tool_choice: Some(ToolChoice::Auto),
        ..ResponsesSettings::default()
    });
    let session = SessionConfig::new()
        .with_instructions(
            "Be concise. Delegate requests needing current information to the backend, which can search the web.",
        )
        .with_delegation(Delegation::Responses(backend));
    assert_eq!(
        to_json(&session),
        json!({
            "model": "gpt-live-1",
            "instructions": "Be concise. Delegate requests needing current information to the backend, which can search the web.",
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": "gpt-5.6-terra",
                    "instructions": "Use web search when current facts are needed. Return concise, grounded results for a spoken conversation.",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto"
                }
            }
        })
    );
}

#[test]
fn every_setting_serializes_in_its_documented_place() {
    let settings = ResponsesSettings {
        instructions: Some("Check availability.".to_owned()),
        max_output_tokens: Some(MaxOutputTokens::new(1024).expect("at least 16")),
        parallel_tool_calls: Some(true),
        reasoning: Some(Reasoning {
            effort: Some(ReasoningEffort::Low),
            summary: Some(ReasoningSummary::Concise),
        }),
        service_tier: Some(ServiceTier::Priority),
        text: Some(TextSettings {
            verbosity: Some(Verbosity::Low),
        }),
        tool_choice: Some(ToolChoice::Function {
            name: "book_table".to_owned(),
        }),
        tools: Some(vec![
            DelegationTool::Function(FunctionTool {
                name: "book_table".to_owned(),
                description: Some("Book a table.".to_owned()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"party": {"type": "integer"}},
                    "required": ["party"]
                })),
                strict: Some(true),
            }),
            DelegationTool::WebSearch,
        ]),
    };
    let session = SessionConfig::new()
        .with_model("gpt-live-1")
        .with_instructions("Be brief.")
        .with_voice(Voice::Cedar)
        .with_input_item(InitialItem::developer("Use metric units."))
        .with_input_item(InitialItem::user("Hi."))
        .with_input_item(InitialItem::assistant("Hello."))
        .with_delegation(Delegation::Responses(
            ResponsesDelegation::new("gpt-6-astra").with_settings(settings),
        ))
        .with_store(false);
    assert_eq!(
        to_json(&session),
        json!({
            "model": "gpt-live-1",
            "instructions": "Be brief.",
            "audio": {"output": {"voice": "cedar"}},
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "Use metric units."}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Hi."}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Hello."}]}
            ],
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": "gpt-6-astra",
                    "instructions": "Check availability.",
                    "max_output_tokens": 1024,
                    "parallel_tool_calls": true,
                    "reasoning": {"effort": "low", "summary": "concise"},
                    "service_tier": "priority",
                    "text": {"verbosity": "low"},
                    "tool_choice": {"type": "function", "name": "book_table"},
                    "tools": [
                        {"type": "function", "name": "book_table", "description": "Book a table.",
                         "parameters": {"type": "object",
                                        "properties": {"party": {"type": "integer"}},
                                        "required": ["party"]},
                         "strict": true},
                        {"type": "web_search"}
                    ]
                }
            },
            "store": false
        })
    );
}

#[test]
fn enum_values_match_the_reference_spellings() {
    let voices = [
        (Voice::Alloy, "alloy"),
        (Voice::Ash, "ash"),
        (Voice::Ballad, "ballad"),
        (Voice::Beacon, "beacon"),
        (Voice::Bossa, "bossa"),
        (Voice::Cedar, "cedar"),
        (Voice::Cinder, "cinder"),
        (Voice::Coral, "coral"),
        (Voice::Delta, "delta"),
        (Voice::Echo, "echo"),
        (Voice::Gleam, "gleam"),
        (Voice::Marin, "marin"),
        (Voice::Meridian, "meridian"),
        (Voice::Quartz, "quartz"),
        (Voice::Ripple, "ripple"),
        (Voice::Sage, "sage"),
        (Voice::Shimmer, "shimmer"),
        (Voice::Stone, "stone"),
        (Voice::Tempo, "tempo"),
        (Voice::Verse, "verse"),
        (Voice::Vesper, "vesper"),
        (Voice::Willow, "willow"),
    ];
    for (voice, name) in voices {
        assert_eq!(to_json(&voice), json!(name));
    }
    let efforts = [
        (ReasoningEffort::None, "none"),
        (ReasoningEffort::Minimal, "minimal"),
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::Xhigh, "xhigh"),
    ];
    for (effort, name) in efforts {
        assert_eq!(to_json(&effort), json!(name));
    }
    let summaries = [
        (ReasoningSummary::Concise, "concise"),
        (ReasoningSummary::Detailed, "detailed"),
        (ReasoningSummary::Auto, "auto"),
    ];
    for (summary, name) in summaries {
        assert_eq!(to_json(&summary), json!(name));
    }
    let tiers = [
        (ServiceTier::Auto, "auto"),
        (ServiceTier::Default, "default"),
        (ServiceTier::FastTierTempPilot, "fast_tier_temp_pilot"),
        (ServiceTier::Flex, "flex"),
        (ServiceTier::Priority, "priority"),
        (ServiceTier::Ultrafast, "ultrafast"),
    ];
    for (tier, name) in tiers {
        assert_eq!(to_json(&tier), json!(name));
    }
    let verbosities = [
        (Verbosity::Low, "low"),
        (Verbosity::Medium, "medium"),
        (Verbosity::High, "high"),
    ];
    for (verbosity, name) in verbosities {
        assert_eq!(to_json(&verbosity), json!(name));
    }
    assert_eq!(to_json(&ToolChoice::Auto), json!("auto"));
    assert_eq!(to_json(&ToolChoice::None), json!("none"));
    assert_eq!(to_json(&ToolChoice::Required), json!("required"));
}

#[test]
fn max_output_tokens_refuses_values_below_sixteen() {
    assert_eq!(MaxOutputTokens::new(15), Err(InvalidMaxOutputTokens(15)));
    assert_eq!(MaxOutputTokens::new(0), Err(InvalidMaxOutputTokens(0)));
    assert_eq!(MaxOutputTokens::new(16).map(MaxOutputTokens::get), Ok(16));
}

#[test]
fn a_rig_tool_definition_becomes_a_function_tool() {
    let tool = FunctionTool::from(ToolDefinition {
        name: "lookup".to_owned(),
        description: "Look a thing up.".to_owned(),
        parameters: json!({"type": "object", "properties": {}}),
    });
    assert_eq!(
        to_json(&DelegationTool::from(tool)),
        json!({
            "type": "function",
            "name": "lookup",
            "description": "Look a thing up.",
            "parameters": {"type": "object", "properties": {}}
        })
    );
}

#[test]
fn an_update_sends_only_the_settings_it_changes() {
    let update = ResponsesDelegationUpdate {
        model: None,
        settings: ResponsesSettings {
            instructions: Some(
                "Check restaurant availability. Ask before confirming a booking.".to_owned(),
            ),
            max_output_tokens: Some(MaxOutputTokens::new(1024).expect("at least 16")),
            ..ResponsesSettings::default()
        },
    };
    assert_eq!(
        to_json(&update),
        json!({
            "instructions": "Check restaurant availability. Ask before confirming a booking.",
            "max_output_tokens": 1024
        })
    );
    let clear_tools = ResponsesDelegationUpdate {
        model: Some("gpt-6-luna".to_owned()),
        settings: ResponsesSettings {
            tools: Some(Vec::new()),
            ..ResponsesSettings::default()
        },
    };
    assert_eq!(
        to_json(&clear_tools),
        json!({"model": "gpt-6-luna", "tools": []})
    );
}
