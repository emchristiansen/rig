use crate::completion::{CompletionRequest, ToolDefinition};
use crate::message::{AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction,
    ToolName, ToolResultContent, UserContent};
use crate::operation::Completion;
use crate::wire::{Mode, Operation as _, Wire as _};
use serde_json::{Value, json};

fn request(call: ToolCall, assistant: Message) -> CompletionRequest {
    let mut request = CompletionRequest::new("next");
    request.tools = vec![ToolDefinition {
        name: ToolName::new("lookup").expect("tool name"),
        description: "Lookup a value".to_owned(),
        parameters: json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}),
    }];
    request.chat_history = vec![Message::user("q"), assistant, Message::User {
        content: vec![UserContent::ToolResult(call.result(vec![ToolResultContent::text("done")]))],
    }];
    request
}

fn canonical() -> CompletionRequest {
    let call = ToolCall::from_wire("call_1", ToolFunction::new(
        ToolName::new("lookup").expect("tool name"), json!({"q":"x"}),
    ));
    request(call.clone(), Message::Assistant(AssistantMessage::new(vec![
        AssistantContent::ToolCall(call),
    ])))
}

fn canonical_part() -> Value {
    json!({"functionCall":{"name":"lookup","args":{"q":"x"},"id":"call_1"}})
}

fn native_part() -> Value {
    json!({"functionCall":{"name":"lookup","args":{"q":"x"},"id":"call_1"},
        "thought":true,"thoughtSignature":"c2ln","partMetadata":{"keep":"raw"}})
}

fn received() -> Value {
    json!({"candidates":[{"content":{"role":"model","parts":[native_part()]},
        "finishReason":"STOP","index":0}]})
}

fn encoded(request: CompletionRequest, model: &str, prepared: bool) -> Value {
    let wire = super::wire(model);
    let request = if prepared {
        Completion::prepare(request, &wire.describe()).expect("real preparation")
    } else { request };
    crate::test_utils::json_body(&wire.encode(request, Mode::Unary).expect("real REST encode").request)
}

#[test]
fn canonical_calls_emit_false_in_rest_without_changing_ids_args_or_gemini_3_signature() {
    for model in ["gemini-2.5-flash", "gemini-3-flash-preview"] {
        for prepared in [false, true] {
            for provider_id in [true, false] {
                let mut request = canonical();
                if !provider_id {
                    let Message::Assistant(turn) = &mut request.chat_history[1] else { unreachable!() };
                    let AssistantContent::ToolCall(call) = &mut turn.content[0] else { unreachable!() };
                    call.id = crate::message::CallId::Local(crate::message::LocalCallId::new());
                    let local = call.id.clone();
                    let Message::User { content } = &mut request.chat_history[2] else { unreachable!() };
                    let UserContent::ToolResult(result) = &mut content[0] else { unreachable!() };
                    result.call = local;
                }
                let mut expected = canonical_part();
                expected["thought"] = json!(false);
                if !provider_id {
                    expected["functionCall"].as_object_mut().expect("call object").shift_remove("id");
                }
                if model == "gemini-3-flash-preview" {
                    expected["thoughtSignature"] = json!("skip_thought_signature_validator");
                }
                let body = encoded(request, model, prepared);
                assert_eq!(body["contents"][1]["parts"], json!([expected]), "{model}, prepared={prepared}, provider_id={provider_id}");
            }
        }
    }
}

#[tokio::test]
async fn actual_rest_native_call_replays_true_signature_metadata_and_whole_part_unchanged() {
    let raw = received();
    let response = super::fold_unary("gemini-2.5-flash", serde_json::to_vec(&raw).expect("JSON"))
        .await.expect("actual REST Model receives terminal");
    assert_eq!(response.raw, raw);
    assert_eq!(response.choice.len(), 1);
    let block = &response.choice[0];
    assert_eq!(block.native_item(), Some(&native_part()));
    let AssistantContent::ToolCall(call) = block else { panic!("real decoded function call") };
    assert_eq!(call.id, crate::message::CallId::from_wire("call_1"));
    assert_eq!(call.function.name.as_str(), "lookup");
    assert_eq!(call.function.arguments, json!({"q":"x"}));
    for prepared in [false, true] {
        let body = encoded(request(call.clone(), response.message().expect("genuine received turn")),
            "gemini-2.5-flash", prepared);
        assert_eq!(body["contents"][1]["parts"], json!([native_part()]));
    }
    assert_eq!(response.raw, raw);
    assert_eq!(response.choice[0].native_item(), Some(&native_part()));
}
