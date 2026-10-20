use super::*;
use crate::message::{
    CustomToolCall, LocalCallId, ProviderCallId, ToolCall, ToolFunction, ToolName,
};

fn function_call(id: CallId) -> ToolCall {
    ToolCall::new(
        id,
        ToolFunction::new(
            ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    )
}

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(function_call(CallId::from_wire(id)))
}

fn result(id: &str) -> UserContent {
    UserContent::ToolResult(
        function_call(CallId::from_wire(id)).result(vec![ToolResultContent::text("3")]),
    )
}

fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(content))
}

#[test]
fn canonical_transcripts_pass() {
    let history = vec![
        Message::user("hi"),
        assistant(vec![call("c1")]),
        Message::User {
            content: vec![result("c1")],
        },
        assistant(vec![AssistantContent::text("done")]),
        Message::user("thanks"),
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    assert!(validate_canonical(&[]).is_ok());
}

#[test]
fn consecutive_assistant_is_rejected() {
    let history = vec![
        assistant(vec![AssistantContent::text("a")]),
        assistant(vec![AssistantContent::text("b")]),
    ];
    assert_eq!(
        validate_canonical(&history),
        Err(TranscriptError::ConsecutiveAssistant { index: 1 })
    );
}

#[test]
fn unanswered_and_orphan_results_are_rejected() {
    let unanswered = vec![assistant(vec![call("c1")]), Message::user("no result")];
    assert!(matches!(
        validate_canonical(&unanswered),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
    let orphan = vec![
        Message::user("hi"),
        Message::User {
            content: vec![result("ghost")],
        },
    ];
    assert!(matches!(
        validate_canonical(&orphan),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
    let trailing = vec![assistant(vec![call("c1")])];
    assert!(matches!(
        validate_canonical(&trailing),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
}

/// Equal-looking IDs from separate namespaces answer only their own calls.
#[test]
fn typed_identity_transcripts_preserve_namespaces_and_completion_scope() {
    let generated = CallId::Local(LocalCallId::new());
    let explicit =
        CallId::Provider(ProviderCallId::new(generated.wire().into_owned()).expect("explicit ID"));
    assert_eq!(generated.wire(), explicit.wire());
    assert_ne!(generated, explicit);
    let call_for = |id: CallId| AssistantContent::ToolCall(function_call(id));
    let result_for = |id: CallId| {
        UserContent::ToolResult(function_call(id).result(vec![ToolResultContent::text("3")]))
    };
    let history = vec![
        assistant(vec![
            call_for(generated.clone()),
            call_for(explicit.clone()),
        ]),
        Message::User {
            content: vec![result_for(explicit.clone()), result_for(generated.clone())],
        },
        assistant(vec![call_for(generated.clone())]),
        Message::User {
            content: vec![result_for(generated.clone())],
        },
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    let mismatched = vec![
        assistant(vec![call_for(generated)]),
        Message::User {
            content: vec![result_for(explicit)],
        },
    ];
    assert!(matches!(
        validate_canonical(&mismatched),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
}

/// A result must answer the kind of call it pairs with; a custom call is a
/// call like any other for pairing.
#[test]
fn results_pair_with_calls_of_their_own_kind() {
    let custom = CustomToolCall::new(
        CallId::from_wire("c1"),
        ToolName::new("patch").expect("tool name"),
        "raw",
    );
    let answered = |answers| {
        let mut result = custom.result(vec![ToolResultContent::text("ok")]);
        result.answers = answers;
        Message::User {
            content: vec![UserContent::ToolResult(result)],
        }
    };
    let turn = assistant(vec![AssistantContent::CustomToolCall(custom.clone())]);
    validate_canonical(&[
        Message::user("go"),
        turn.clone(),
        answered(custom.answered_by()),
    ])
    .expect("a custom result answers a custom call");
    assert_eq!(
        validate_canonical(&[
            Message::user("go"),
            turn.clone(),
            answered(function_call(custom.id.clone()).answered_by()),
        ]),
        Err(TranscriptError::MispairedToolResult {
            index: 2,
            call_id: custom.id.clone(),
            call: AnsweredToolCall::Custom,
            answers: AnsweredToolCall::Function,
        })
    );
    assert_eq!(
        validate_canonical(&[Message::user("go"), turn]),
        Err(TranscriptError::UnansweredToolCall {
            index: 1,
            call_id: custom.id,
        }),
        "an unanswered custom call is caught like a function call"
    );
}

#[test]
fn stored_results_require_the_kind_of_call_they_answer() {
    let function = function_call(CallId::from_wire("function"));
    let custom = CustomToolCall::new(
        CallId::from_wire("custom"),
        ToolName::new("patch").expect("tool name"),
        "raw",
    );
    for (result, kind) in [
        (function.result(vec![]), "function"),
        (custom.result(vec![]), "custom"),
    ] {
        let mut stored = serde_json::to_value(&result).expect("serialize the result");
        assert_eq!(
            serde_json::from_value::<crate::message::ToolResult>(stored.clone())
                .expect("the complete result should deserialize"),
            result,
        );
        let removed = stored
            .as_object_mut()
            .expect("a result serializes as an object")
            .shift_remove("answers");
        assert_eq!(removed, Some(serde_json::json!(kind)));
        let error = serde_json::from_value::<crate::message::ToolResult>(stored)
            .expect_err("a result with no kind must not be inferred");
        assert!(error.to_string().contains("missing field `answers`"));
    }
}

#[test]
fn canonical_text_includes_every_carried_category_in_choice_order() {
    use crate::message::{
        CarriedPayload, CarriedText, CarriedTextContent, GeminiSignedAnswer,
        GeminiThoughtSignature, Opaque,
    };
    let raw = serde_json::json!({
        "type": "future_part", "text": "RAW_ONLY",
        "sibling": {"text": "RAW_SIBLING_ONLY"},
    });
    let mut content = vec![AssistantContent::text("a|")];
    for text in [
        CarriedTextContent::Text("text|".to_owned()),
        CarriedTextContent::Refusal("refusal|".to_owned()),
        CarriedTextContent::Opaque("opaque-edit|".to_owned()),
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
            "signed|",
            GeminiThoughtSignature::new("sig-exact"),
        )),
        CarriedTextContent::InspectionOnlyText("inspection|".to_owned()),
        CarriedTextContent::InspectionOnlyMedia("media|".to_owned()),
    ] {
        content.push(AssistantContent::CarriedText(CarriedText::new(
            text,
            CarriedPayload::new(raw.clone(), None),
        )));
    }
    content.extend([
        AssistantContent::reasoning("REASONING_ONLY"),
        AssistantContent::provider_item(serde_json::json!({"text":"PROVIDER_RAW_ONLY"}), "test"),
        AssistantContent::Opaque(Opaque {
            item: serde_json::json!({"text":"OPAQUE_RAW_ONLY"}),
            replay: false,
        }),
        call("ignored-call"),
        AssistantContent::text("z"),
    ]);
    assert_eq!(
        assistant_text_from_choice(&content),
        "a|text|refusal|opaque-edit|signed|inspection|media|z",
    );
    for part in &content {
        if let AssistantContent::CarriedText(text) = part {
            assert_eq!(text.payload.item, raw);
            if let Some(answer) = text.signed_answer() {
                assert_eq!(answer.signature.as_str(), "sig-exact");
            }
        }
    }
}

#[test]
fn empty_canonical_carried_text_never_promotes_raw_siblings() {
    use crate::message::{CarriedPayload, CarriedText, CarriedTextContent};
    let raw = serde_json::json!({"text":"RAW_ONLY","sibling":{"text":"RAW_SIBLING_ONLY"}});
    let text = CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(raw.clone(), None),
    );
    let mut content = vec![AssistantContent::CarriedText(text)];
    assert_eq!(assistant_text_from_choice(&content), "");
    let AssistantContent::CarriedText(text) = &mut content[0] else {
        panic!("the synthetic carried part stays carried");
    };
    *text.text_mut() = "edited canonical answer".to_owned();
    assert_eq!(text.payload.item, raw);
    assert_eq!(
        assistant_text_from_choice(&content),
        "edited canonical answer"
    );
}

/// A system message resets the consecutive-assistant check but keeps the
/// pending calls, so the next assistant turn finds the call unanswered.
#[test]
fn pending_calls_survive_a_system_message_into_the_next_assistant_turn() {
    let history = vec![
        assistant(vec![call("c1")]),
        Message::system("note"),
        assistant(vec![AssistantContent::text("moving on")]),
    ];
    assert_eq!(
        validate_canonical(&history),
        Err(TranscriptError::UnansweredToolCall {
            index: 1,
            call_id: CallId::from_wire("c1"),
        })
    );
}

/// Custom calls in a turn with an invalid call are answered like function
/// calls: the invalid one with the feedback, every other one as not
/// executed; non-call parts get no result.
#[test]
fn invalid_call_feedback_answers_custom_calls_by_kind() {
    let invalid = CustomToolCall::new(
        CallId::from_wire("k1"),
        ToolName::new("patch").expect("tool name"),
        "raw one",
    );
    let peer = CustomToolCall::new(
        CallId::from_wire("k2"),
        ToolName::new("apply").expect("tool name"),
        "raw two",
    );
    let function = function_call(CallId::from_wire("f1"));
    let content = vec![
        AssistantContent::text("preamble"),
        AssistantContent::CustomToolCall(invalid.clone()),
        AssistantContent::ToolCall(function.clone()),
        AssistantContent::CustomToolCall(peer.clone()),
    ];
    let custom_error = |call: &CustomToolCall, text: &str| {
        UserContent::ToolResult(crate::message::ToolResult {
            call: call.id.clone(),
            name: call.name.clone(),
            answers: AnsweredToolCall::Custom,
            content: vec![ToolResultContent::text(text)],
            is_error: true,
        })
    };
    assert_eq!(
        invalid_call_feedback(&content, &invalid.id, "bad input"),
        vec![
            custom_error(&invalid, "bad input"),
            UserContent::ToolResult(crate::message::ToolResult {
                call: function.id.clone(),
                name: function.function.name.clone(),
                answers: AnsweredToolCall::Function,
                content: vec![ToolResultContent::text(
                    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                )],
                is_error: true,
            }),
            custom_error(&peer, TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER),
        ]
    );
}
