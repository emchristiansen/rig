use crate::completion::CompletionRequest;
use crate::message::{
    self, AssistantContent, AssistantMessage, CallId, LocalCallId, Message, ToolCall, ToolFunction,
    ToolName,
};
use crate::test_utils::json_body;
use crate::wire::{Mode, Wire as _};
use serde_json::{Value, json};

fn current_pair_request() -> CompletionRequest {
    let name = ToolName::new("test").expect("the original tool name is nonempty");
    let id = CallId::Local(LocalCallId::new());
    let local = ToolCall::new(id, ToolFunction::new(name.clone(), json!({})));
    let genuine = ToolCall::from_wire("tool-0", ToolFunction::new(name, json!({})));
    let result = |call: &ToolCall| {
        let mut result = call.result(vec![]);
        result.name =
            ToolName::new("possibly_repaired").expect("the original result name is nonempty");
        Message::User {
            content: vec![message::UserContent::ToolResult(result)],
        }
    };
    let assistant = |call: &ToolCall| {
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )]))
    };
    CompletionRequest::from(vec![
        Message::system("system"),
        assistant(&local),
        assistant(&genuine),
        result(&genuine),
        Message::assistant("intervening text"),
        result(&local),
        assistant(&local),
        result(&local),
    ])
    .max_tokens(128)
}

fn assert_fixture_limits(request: &CompletionRequest) {
    assert_eq!(request.chat_history.len(), 8);
    assert_eq!(request.max_tokens, Some(128));
    assert!(request.model.is_none());
    assert!(request.tools.is_empty());
    assert!(request.documents.is_empty());
}

fn assert_current_pairs(wire: Value) {
    fn visit(value: &serde_json::Value, calls: &mut Vec<String>, results: &mut Vec<String>) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    visit(value, calls, results);
                }
            }
            serde_json::Value::Object(object) => {
                let kind = object.get("type").and_then(serde_json::Value::as_str);
                let reference = match kind {
                    Some("function_call") => object
                        .get("call_id")
                        .or_else(|| object.get("id"))
                        .map(|id| (true, id)),
                    Some("tool_use") => object.get("id").map(|id| (true, id)),
                    Some("function_call_output" | "function_result") => {
                        object.get("call_id").map(|id| (false, id))
                    }
                    Some("tool_result") => object.get("tool_use_id").map(|id| (false, id)),
                    _ => None,
                };
                if let Some((is_call, id)) = reference {
                    let id = id.as_str().expect("wire identity is a string").to_owned();
                    if is_call {
                        calls.push(id);
                    } else {
                        results.push(id);
                    }
                }
                if object.get("role").and_then(serde_json::Value::as_str) == Some("tool") {
                    results.push(object["tool_call_id"].as_str().unwrap().to_owned());
                }
                if let Some(array) = object
                    .get("tool_calls")
                    .and_then(serde_json::Value::as_array)
                {
                    calls.extend(
                        array
                            .iter()
                            .map(|call| call["id"].as_str().unwrap().to_owned()),
                    );
                }
                for value in object.values() {
                    visit(value, calls, results);
                }
            }
            _ => {}
        }
    }
    let mut calls = Vec::new();
    let mut results = Vec::new();
    visit(&wire, &mut calls, &mut results);
    assert_eq!(calls.len(), 3, "{wire}");
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()],
        "{wire}"
    );
    assert_eq!(calls[1], "tool-0");
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "{wire}"
    );
}

/// G89 retains the one coherent current eight-message fixture. The old
/// call_only/result_only provenance inputs and minted-hint view are retired.
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let request = current_pair_request();
    assert_fixture_limits(&request);
    let wire = crate::providers::cohere::NativeChat::new(
        crate::providers::cohere::CohereConfig::new("test-key"),
        crate::providers::cohere::COMMAND_A_03_2025,
    );
    // Direct native encode: no Completion::prepare repair or pairing workaround.
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the approved complete native Cohere history encodes");
    let value = json_body(&encoded.request);
    assert_eq!(value["model"], crate::providers::cohere::COMMAND_A_03_2025);
    assert_current_pairs(value);
}
