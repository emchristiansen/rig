use super::*;
use crate::completion::{AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_SONNET_4_5, PROVIDER_NAME};
use rig_core::completion::ToolDefinition;
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantMessage, CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart,
    CustomToolCall, ToolCall, ToolFunction, ToolName, UnrepresentableOpaqueContent,
    UnrepresentableToolCall,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};

const REDACTED_BLOB: &[u8] = b"\x00\x01opaque-ciphertext\xff";

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the original fixture has a nonempty name")
}

fn assistant_turn(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage::new(content))
}

fn unrepresentable_calls() -> [(AssistantContent, &'static str, UnrepresentableToolCall); 2] {
    [
        (
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("call_1"),
                ToolFunction::new(name("add"), json!({"x": 1}))
                    .with_namespace(Some("math".to_owned())),
            )),
            "cannot represent tool namespaces, so call `add` qualified by namespace `math`",
            UnrepresentableToolCall::Namespace {
                wire: "Bedrock Converse",
                name: name("add"),
                namespace: "math".to_owned(),
            },
        ),
        (
            AssistantContent::CustomToolCall(CustomToolCall::new(
                CallId::from_wire("call_2"),
                name("apply_patch"),
                r#"{"x": 1}"#,
            )),
            "carries JSON tool arguments only, so custom tool call `apply_patch`",
            UnrepresentableToolCall::CustomCall {
                wire: "Bedrock Converse",
                name: name("apply_patch"),
            },
        ),
    ]
}

fn assert_call_refusal(error: ProviderError, text: &str, expected: &UnrepresentableToolCall) {
    let ProviderError::Request(reason) = &error else {
        panic!("a request refusal, got {error:?}");
    };
    assert_eq!(
        reason.downcast_ref::<UnrepresentableToolCall>(),
        Some(expected)
    );
    let reason = reason.to_string();
    assert!(reason.starts_with("Bedrock Converse "), "{reason}");
    assert!(reason.contains(text), "{reason}");
}

#[test]
fn converse_refuses_namespaced_and_custom_calls() {
    let wire = Converse::new(AMAZON_NOVA_LITE);
    for (content, refusal, expected) in unrepresentable_calls() {
        match &content {
            AssistantContent::ToolCall(call) => {
                assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_1"));
                assert_eq!(call.function.name.as_str(), "add");
                assert_eq!(call.function.namespace.as_deref(), Some("math"));
                assert_eq!(json!(call.function.arguments), json!({"x": 1}));
            }
            AssistantContent::CustomToolCall(call) => {
                assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_2"));
                assert_eq!(call.name.as_str(), "apply_patch");
                assert_eq!(call.namespace, None);
                assert_eq!(call.input, r#"{"x": 1}"#);
            }
            other => panic!("the original call fixture changed: {other:?}"),
        }
        let mut request = CompletionRequest::new("fixture");
        request.chat_history = vec![assistant_turn(vec![content])];
        let error = wire
            .encode(request, Mode::Unary)
            .expect_err("refused, not dropped");
        assert_call_refusal(error.into(), refusal, &expected);
    }
}

#[test]
fn original_calls_are_refused_before_preparation_can_drop_or_reshape_them() {
    let wire = Converse::new(AMAZON_NOVA_LITE);
    for (content, refusal, expected) in unrepresentable_calls() {
        for leading_user in [false, true] {
            for known_neighbor in [false, true] {
                for declared_tools in [false, true] {
                    let mut request = CompletionRequest::new("fixture");
                    request.chat_history = if leading_user {
                        vec![Message::user("hi")]
                    } else {
                        vec![]
                    };
                    let mut blocks = if known_neighbor {
                        vec![AssistantContent::text("known")]
                    } else {
                        vec![]
                    };
                    blocks.push(content.clone());
                    request.chat_history.push(assistant_turn(blocks));
                    if declared_tools {
                        request.tools = ["add", "apply_patch"]
                            .into_iter()
                            .map(|tool| ToolDefinition {
                                name: name(tool),
                                description: tool.to_owned(),
                                parameters: json!({"type":"object"}),
                            })
                            .collect();
                    }
                    let error = Completion::prepare(request, &wire.describe())
                        .expect_err("the original call is refused before adaptation");
                    assert_call_refusal(error, refusal, &expected);
                }
            }
        }
    }
}

fn carried(parts: Vec<CarriedReasoningPart>) -> AssistantContent {
    // The original caller-built reasoning has no native item or issuer.
    AssistantContent::CarriedReasoning(CarriedReasoning::new(
        parts,
        CarriedPayload::new(Value::Null, None),
    ))
}

fn reasoning_body(model: &str, content: AssistantContent, prepared: bool) -> Value {
    let wire = Converse::new(model);
    let mut request = CompletionRequest::new("fixture");
    request.chat_history = vec![Message::user("hi"), assistant_turn(vec![content.clone()])];
    if prepared {
        request = Completion::prepare(request, &wire.describe()).expect("known reasoning prepares");
        let Message::Assistant(turn) = &request.chat_history[1] else {
            panic!("the visible original assistant turn survives");
        };
        assert_eq!(
            turn.content,
            vec![content],
            "preparation retains the full carried input"
        );
    }
    serde_json::to_value(
        wire.encode(request, Mode::Unary)
            .expect("known reasoning encodes")
            .body,
    )
    .expect("the actual final body serializes")
}

fn assert_visible(body: &Value, text: &str, claude: bool) {
    let expected = if claude {
        json!({"text":text})
    } else {
        json!({"reasoningContent":{"reasoningText":{"text":text}}})
    };
    assert_eq!(body["messages"][1]["content"], json!([expected]));
}

#[test]
fn opaque_reasoning_mixed_with_text_drops_the_opaque_part_not_the_request() {
    let encrypted = BASE64_STANDARD.encode(REDACTED_BLOB);
    let original = carried(vec![
        CarriedReasoningPart::Text {
            text: "visible thinking".to_owned(),
            signature: None,
        },
        CarriedReasoningPart::Redacted {
            data: encrypted.clone(),
        },
    ]);
    for (model, claude) in [
        (AMAZON_NOVA_LITE, false),
        (ANTHROPIC_CLAUDE_SONNET_4_5, true),
    ] {
        for prepared in [false, true] {
            let body = reasoning_body(model, original.clone(), prepared);
            assert_visible(&body, "visible thinking", claude);
            assert!(
                !body.to_string().contains(&encrypted),
                "redacted bytes must not enter the request"
            );
        }
    }
}

#[test]
fn responses_summary_plus_encrypted_reasoning_replays_as_text() {
    let original = carried(vec![
        CarriedReasoningPart::Summary("the summary".to_owned()),
        CarriedReasoningPart::Encrypted("enc-opaque-blob".to_owned()),
    ]);
    for (model, claude) in [
        (AMAZON_NOVA_LITE, false),
        (ANTHROPIC_CLAUDE_SONNET_4_5, true),
    ] {
        for prepared in [false, true] {
            let body = reasoning_body(model, original.clone(), prepared);
            assert_visible(&body, "the summary", claude);
            assert!(
                !body.to_string().contains("enc-opaque-blob"),
                "ciphertext must never flatten into the plaintext body"
            );
        }
    }
}

#[test]
fn carried_reasoning_projects_only_text_and_summary_in_original_order() {
    let original = carried(vec![
        CarriedReasoningPart::Text {
            text: "first".to_owned(),
            signature: Some("signature-not-replayed".to_owned()),
        },
        CarriedReasoningPart::Encrypted("encrypted-not-replayed".to_owned()),
        CarriedReasoningPart::Summary("second".to_owned()),
        CarriedReasoningPart::Redacted {
            data: "redacted-not-replayed".to_owned(),
        },
        CarriedReasoningPart::Text {
            text: "third".to_owned(),
            signature: None,
        },
    ]);
    for (model, claude) in [
        (AMAZON_NOVA_LITE, false),
        (ANTHROPIC_CLAUDE_SONNET_4_5, true),
    ] {
        for prepared in [false, true] {
            let body = reasoning_body(model, original.clone(), prepared);
            assert_visible(&body, "first\nsecond\nthird", claude);
            for excluded in [
                "signature-not-replayed",
                "encrypted-not-replayed",
                "redacted-not-replayed",
            ] {
                assert!(!body.to_string().contains(excluded));
            }
        }
    }
}

#[test]
fn nested_opaque_reasoning_is_a_typed_refusal_on_direct_and_prepared_routes() {
    let raw = json!({"type":"future_part","payload":{"list":[1,2]}});
    let opaque = CarriedPayload::new(raw, Some("openai".to_owned()));
    let wire = Converse::new(AMAZON_NOVA_LITE);
    for part in [
        CarriedReasoningPart::OpaqueSummary(opaque.clone()),
        CarriedReasoningPart::OpaqueContent(opaque),
    ] {
        let mut request = CompletionRequest::new("fixture");
        request.chat_history = vec![
            Message::user("hi"),
            assistant_turn(vec![AssistantContent::text("known"), carried(vec![part])]),
        ];
        for error in [
            ProviderError::from(
                wire.encode(request.clone(), Mode::Unary)
                    .expect_err("opaque content refused"),
            ),
            Completion::prepare(request, &wire.describe())
                .expect_err("opaque content refused before shaping"),
        ] {
            let ProviderError::Request(reason) = &error else {
                panic!("a request refusal, got {error:?}");
            };
            assert_eq!(
                reason.downcast_ref::<UnrepresentableOpaqueContent>(),
                Some(&UnrepresentableOpaqueContent::new(PROVIDER_NAME))
            );
            assert!(error.to_string().contains("opaque"));
        }
    }
}

#[test]
fn ordinary_declared_calls_survive_preparation() {
    let wire = Converse::new(AMAZON_NOVA_LITE);
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("add"), json!({"x":1})),
    );
    let mut request = CompletionRequest::new("fixture");
    request.chat_history = vec![
        Message::user("hi"),
        assistant_turn(vec![AssistantContent::ToolCall(call.clone())]),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )],
        },
    ];
    request.tools = vec![ToolDefinition {
        name: name("add"),
        description: "add".to_owned(),
        parameters: json!({"type":"object"}),
    }];
    let prepared = Completion::prepare(request, &wire.describe()).expect("ordinary calls prepare");
    let Message::Assistant(turn) = &prepared.chat_history[1] else {
        panic!("the ordinary call turn survives");
    };
    assert!(
        matches!(turn.content.as_slice(), [AssistantContent::ToolCall(kept)] if kept.id == call.id)
    );
    let body = serde_json::to_value(
        wire.encode(prepared, Mode::Unary)
            .expect("ordinary calls encode")
            .body,
    )
    .expect("the actual body serializes");
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"toolUse":{"toolUseId":"call_1","name":"add","input":{"x":1}}}])
    );
    assert_eq!(
        body["messages"][2]["content"],
        json!([{"toolResult":{"toolUseId":"call_1","content":[{"text":"ok"}]}}])
    );
}

#[test]
fn direct_occurrence_aliases_are_unique_and_pair_each_result() {
    let wire = Converse::new(AMAZON_NOVA_LITE);
    let local = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(name("add"), json!({"x":1})),
    );
    let provider = ToolCall::new(
        CallId::from_wire("tool-0"),
        ToolFunction::new(name("add"), json!({"x":2})),
    );
    let answered =
        |call: &ToolCall| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")]));
    let mut request = CompletionRequest::new("fixture");
    request.chat_history = vec![
        Message::system("system"),
        Message::user("hi"),
        assistant_turn(vec![
            AssistantContent::text("known"),
            AssistantContent::ToolCall(local.clone()),
        ]),
        Message::User {
            content: vec![answered(&local)],
        },
        assistant_turn(vec![
            AssistantContent::ToolCall(local.clone()),
            AssistantContent::ToolCall(provider.clone()),
        ]),
        Message::User {
            content: vec![answered(&provider), answered(&local)],
        },
    ];
    let body = serde_json::to_value(
        wire.encode(request, Mode::Unary)
            .expect("each answered occurrence encodes")
            .body,
    )
    .expect("the actual body serializes");
    assert_eq!(
        body["messages"][1]["content"][1]["toolUse"]["toolUseId"],
        "tool-1"
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        "tool-1"
    );
    assert_eq!(
        body["messages"][3]["content"][0]["toolUse"]["toolUseId"],
        "tool-2"
    );
    assert_eq!(
        body["messages"][3]["content"][1]["toolUse"]["toolUseId"],
        "tool-0"
    );
    assert_eq!(
        body["messages"][4]["content"][0]["toolResult"]["toolUseId"],
        "tool-0"
    );
    assert_eq!(
        body["messages"][4]["content"][1]["toolResult"]["toolUseId"],
        "tool-2"
    );

    let mut ambiguous = CompletionRequest::new("fixture");
    ambiguous.chat_history = vec![
        Message::user("hi"),
        assistant_turn(vec![
            AssistantContent::ToolCall(local.clone()),
            AssistantContent::ToolCall(local),
        ]),
    ];
    let error = ProviderError::from(
        wire.encode(ambiguous, Mode::Unary)
            .expect_err("unanswered reuse is refused"),
    );
    let ProviderError::Request(reason) = &error else {
        panic!("a request refusal, got {error:?}");
    };
    assert_eq!(
        reason.downcast_ref::<rig_core::providers::internal::wire_ids::WireIdError>(),
        Some(
            &rig_core::providers::internal::wire_ids::WireIdError::Ambiguous {
                message: 1,
                content: 1
            }
        )
    );
}
