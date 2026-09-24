use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: ToolCallId::new_or_minted(id, 0),
        provider: None,
        function: ToolFunction {
            name: "add".into(),
            namespace: None,
            arguments: serde_json::json!({}),
        },
        additional_params: None,
        signature: None,
    })
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: ToolCallId::new_or_minted(id, 0),
        provider: None,
        name: "add".into(),
        answers: crate::message::AnsweredToolCall::Function,
        content: vec![ToolResultContent::text("3")],
    })
}
fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant { id: None, content }
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
    let generated = ToolCallId::minted(0);
    let explicit = ToolCallId::new("tool-0").expect("explicit ID");
    let call_for = |id: ToolCallId| {
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new("add".into(), serde_json::json!({})),
        ))
    };
    let result_for = |id: ToolCallId| {
        UserContent::ToolResult(ToolResult {
            call: id,
            provider: None,
            name: "add".into(),
            answers: crate::message::AnsweredToolCall::Function,
            content: vec![ToolResultContent::text("3")],
        })
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
    use crate::message::AnsweredToolCall;
    let custom =
        crate::message::CustomToolCall::from_dual_wire("ctc_1", "c1", "patch", None, "raw");
    let answered = |answers| Message::User {
        content: vec![UserContent::tool_result_for_answering(
            custom.id.clone(),
            custom.provider.clone(),
            "patch",
            answers,
            vec![ToolResultContent::text("ok")],
        )],
    };
    let turn = Message::Assistant {
        id: None,
        content: vec![AssistantContent::CustomToolCall(custom.clone())],
    };
    validate_canonical(&[
        Message::user("go"),
        turn.clone(),
        answered(AnsweredToolCall::Custom),
    ])
    .expect("a custom result answers a custom call");
    assert_eq!(
        validate_canonical(&[
            Message::user("go"),
            turn.clone(),
            answered(AnsweredToolCall::Function),
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
