use super::*;
use crate::run::policy::{DiagnosticArguments, DiagnosticCall, DiagnosticHistoryEntry};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{CallId, CarriedPayload, CarriedText, CarriedTextContent, LocalCallId, ToolCall, ToolFunction, ToolName};
use serde_json::json;

#[test]
fn completion_accounting_keeps_message_response_and_request_ids_independent() {
    let base = CompletionCall::new(3, Usage::default(), json!({"raw":1}));
    assert!(serde_json::to_value(&base).expect("serialize").get("message_id").is_none());
    let without_message = base.clone().with_identity(ResponseIdentity {
        message_id: None, response_id: Some("response".into()), provider_request_id: Some("request".into()),
    });
    assert!(without_message.message_id.is_none());
    let identity = ResponseIdentity {
        message_id: ProviderMessageId::new("message"), response_id: Some("response".into()), provider_request_id: Some("request".into()),
    };
    let call = base.with_identity(identity.clone()).with_finish_reason(Some(FinishReason::Stop));
    assert_eq!(call.identity(), identity);
    let wire = serde_json::to_value(&call).expect("serialize");
    assert_eq!(wire["message_id"], "message");
    assert_eq!(wire["response_id"], "response");
    assert_eq!(wire["provider_request_id"], "request");
    let round: CompletionCall = serde_json::from_value(wire).expect("deserialize");
    assert_eq!(round, call);
    let bad = json!({"call_index":0,"usage":{},"message_id":"","raw":null});
    assert!(serde_json::from_value::<CompletionCall>(bad).is_err());
}

#[test]
fn renamed_prompt_errors_keep_protected_display_strings() {
    let provider = ProviderError::Response("bad reply".into());
    let text = format!("CompletionError: {provider}");
    assert_eq!(PromptError::Provider(provider).to_string(), text);
    let memory = MemoryError::Policy("bad history".into());
    assert_eq!(PromptError::Memory(memory).to_string(), "MemoryError: Memory policy error: bad history");
    assert_eq!(PromptError::MaxTurns { max_turns:3, chat_history:vec![], prompt:Message::user("question") }.to_string(),
        "MaxTurnsError: reached max turns limit: 3");
    assert_eq!(PromptError::cancelled([Message::user("question")], "stop").to_string(), "PromptCancelled: stop");
    assert_eq!(PromptError::UnknownToolCall {
        tool_name:"missing".into(), available_tools:vec!["add".into()], allowed_tools:vec!["add".into()], chat_history:vec![],
    }.to_string(), "UnknownToolCall: model attempted to call unknown or disallowed tool `missing`. Available tools: [\"add\"]. Allowed tools for this turn: [\"add\"]");
    let report = ErrorReport::new(ErrorKind::Response, "origin failure");
    assert_eq!(PromptError::Report(report.clone()).to_string(), report.to_string());
    assert!(matches!(PromptError::from(ProviderError::Relayed(Box::new(report.clone()))), PromptError::Report(found) if found == report));
}

#[test]
fn rejected_error_histories_keep_typed_value_and_raw_without_canonical_calls() {
    let rejected = DiagnosticCall::early_name(CallId::Local(LocalCallId::new()), "missing".into(), " [1, null] ".into());
    assert_eq!(rejected.arguments, DiagnosticArguments::Parsed(json!([1,null])));
    let history = vec![DiagnosticHistoryEntry::Message(Message::user("question")), DiagnosticHistoryEntry::RejectedCall(rejected.clone())];
    for error in [
        PromptError::Cancelled { chat_history:history.clone(), reason:"stop".into() },
        PromptError::UnknownToolCall { tool_name:"missing".into(), available_tools:vec![], allowed_tools:vec![], chat_history:history.clone() },
    ] {
        let entries = match error {
            PromptError::Cancelled { chat_history, .. } | PromptError::UnknownToolCall { chat_history, .. } => chat_history,
            _ => panic!("diagnostic error history"),
        };
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[1], DiagnosticHistoryEntry::RejectedCall(call) if call == &rejected));
        let encoded = serde_json::to_value(&entries).expect("diagnostic history serialize");
        assert_eq!(encoded[0], serde_json::to_value(Message::user("question")).expect("message serialize"));
        let decoded: Vec<DiagnosticHistoryEntry> = serde_json::from_value(encoded).expect("diagnostic history deserialize");
        assert!(matches!(&decoded[1], DiagnosticHistoryEntry::RejectedCall(call) if call == &rejected));
    }
}

#[test]
fn output_tool_finalization_removes_protected_carried_prose() {
    let function = ToolFunction::new(ToolName::new("output").expect("nonempty name"), json!({"value":7}));
    let call = ToolCall::new(CallId::Local(LocalCallId::new()), function);
    let carried = CarriedText::new(CarriedTextContent::InspectionOnlyText("prose".into()),
        CarriedPayload::new(json!({"text":"prose","extra":1}), Some("gemini".into())));
    let choice = vec![AssistantContent::Text(rig_core::message::Text::new("plain")),
        AssistantContent::CarriedText(carried), AssistantContent::ToolCall(call)];
    let finalized = finalize_output_tool_choice(&choice, "{\"value\":7}").expect("output tool settled");
    assert_eq!(finalized, vec![AssistantContent::text("{\"value\":7}")]);
    assert!(finalize_output_tool_choice(&[AssistantContent::text("plain")], "answer").is_none());
}
