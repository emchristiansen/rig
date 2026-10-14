use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction,
    ToolName, ToolResultContent, UserContent};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation as _, Wire as _};
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

use super::super::{GEMINI_2_5_FLASH, GenerateContent};

#[test]
fn shared_json_has_false_but_real_grpc_request_json_omits_the_proto3_default() {
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    for prepared in [false, true] {
        let request = if prepared {
            Completion::prepare(canonical(), &wire.describe()).expect("real preparation")
        } else { canonical() };
        let body = rig_core::providers::gemini::completion::request_body(
            &request, &wire, GEMINI_2_5_FLASH, None, |body| {
                body.insert("model".to_owned(), format!("models/{GEMINI_2_5_FLASH}").into());
            },
        ).expect("shared pre-transcode request");
        let body = serde_json::to_value(&body).expect("shared JSON");
        let mut marked = canonical_part(); marked["thought"] = json!(false);
        assert_eq!(body["contents"][1]["parts"], json!([marked]));
        let encoded = wire.encode(request, Mode::Unary).expect("actual protobuf wire");
        let actual = crate::rest::to_rest(&encoded).expect("actual protobuf request JSON");
        assert_eq!(actual["contents"][1]["parts"], json!([canonical_part()]));
        assert_eq!(actual, json!({
            "model":format!("models/{GEMINI_2_5_FLASH}"),
            "contents":[
                {"role":"user","parts":[{"text":"q"}]},
                {"role":"model","parts":[canonical_part()]},
                {"role":"user","parts":[{"functionResponse":{
                    "name":"lookup","id":"call_1","response":{"result":"done"}
                }}]},
            ],
            "tools":[{"functionDeclarations":[{
                "name":"lookup","description":"Lookup a value",
                "parameters":{"type":"OBJECT","properties":{"q":{"type":"STRING"}},"required":["q"]}
            }]}],
        }));
    }
}

#[test]
fn actual_grpc_native_call_replays_true_signature_metadata_and_whole_part_unchanged() {
    let incoming: crate::proto::GenerateContentResponse = crate::rest::from_rest(received())
        .expect("fixture read by existing protobuf REST transcode");
    let whole = crate::rest::to_rest(&incoming).expect("received proto JSON");
    assert_eq!(whole["candidates"][0]["content"]["parts"], json!([native_part()]));
    let response = super::complete(incoming).expect("actual protobuf Model receives terminal");
    assert_eq!(response.raw, whole);
    assert_eq!(response.choice.len(), 1);
    let block = &response.choice[0];
    assert_eq!(block.native_item(), Some(&native_part()));
    let AssistantContent::ToolCall(call) = block else { panic!("real decoded function call") };
    assert_eq!(call.id, rig_core::message::CallId::from_wire("call_1"));
    assert_eq!(call.function.name.as_str(), "lookup");
    assert_eq!(call.function.arguments_value(), json!({"q":"x"}));
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    for prepared in [false, true] {
        let request = request(call.clone(), response.message().expect("genuine received turn"));
        let request = if prepared { Completion::prepare(request, &wire.describe()).expect("real prepare") }
            else { request };
        let actual = crate::rest::to_rest(&wire.encode(request, Mode::Unary).expect("real grpc encode"))
            .expect("encoded protobuf request JSON");
        assert_eq!(actual["contents"][1]["parts"], json!([native_part()]));
    }
    assert_eq!(response.raw, whole);
    assert_eq!(response.choice[0].native_item(), Some(&native_part()));
}
