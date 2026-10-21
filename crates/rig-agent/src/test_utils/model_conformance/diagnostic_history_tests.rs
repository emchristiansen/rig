use super::*;
use crate::run::policy::{DiagnosticCall, DiagnosticHistoryEntry};
use rig_core::message::{CallId, LocalCallId, ToolCall, ToolFunction, ToolName};
use serde_json::json;

#[test]
fn error_history_validators_accept_exact_typed_rejected_values() {
    for raw in [" 7 ", " [1,null] ", " null "] {
        let call = DiagnosticCall::early_name(CallId::Local(LocalCallId::new()), "missing".into(), raw.into());
        let history = vec![DiagnosticHistoryEntry::Message(Message::user("question")), DiagnosticHistoryEntry::RejectedCall(call.clone())];
        let unknown = PromptError::UnknownToolCall {
            tool_name:"missing".into(), available_tools:vec!["add".into()], allowed_tools:vec!["add".into()], chat_history:history.clone(),
        };
        validate_unknown_tool_failure(&unknown, "missing", &["add"]).expect("typed rejected evidence is retained");
        assert!(validate_unknown_tool_failure(&unknown, "missing", &["sum"]).is_err());
        let cancelled = PromptError::Cancelled { chat_history:history, reason:"stop".into() };
        validate_cancelled_failure(&cancelled, "stop", "missing").expect("exact name and reason retained");
        assert!(validate_cancelled_failure(&cancelled, "stop", "other").is_err());
        let PromptError::Cancelled { chat_history, .. } = cancelled else { panic!("cancelled"); };
        assert!(matches!(&chat_history[1], DiagnosticHistoryEntry::RejectedCall(found) if found == &call && found.raw.as_deref() == Some(raw)));
    }
}

#[test]
fn error_history_validators_keep_object_call_meaning() {
    let call = ToolCall::new(CallId::Local(LocalCallId::new()), ToolFunction::new(ToolName::new("add").expect("name"), json!({"x":1,"y":2})));
    let history = vec![DiagnosticHistoryEntry::Message(Message::Assistant(rig_core::message::AssistantMessage::new(vec![AssistantContent::ToolCall(call)])))];
    let unknown = PromptError::UnknownToolCall {
        tool_name:"rejected-repair-target".into(), available_tools:vec!["add".into()], allowed_tools:vec!["add".into()], chat_history:history.clone(),
    };
    validate_unknown_tool_failure(&unknown, "rejected-repair-target", &["add"]).expect("original object call remains even when Repair target is rejected");
    let cancelled = PromptError::Cancelled { chat_history:history, reason:"stop".into() };
    validate_cancelled_failure(&cancelled, "stop", "add").expect("genuine object call retained");
}
