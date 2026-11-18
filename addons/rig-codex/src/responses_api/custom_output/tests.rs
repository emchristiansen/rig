//! A custom call's answer is named after the result that produced it, through
//! both writers of a request's `input`: [`input_items`] and
//! [`CodexResponses::encode_body`].

use super::super::fidelity_fixtures::{
    answering, custom_call_item, declared_custom, folded_call, with_provider_options,
};
use super::super::input_items;
use super::super::wire::CodexResponses;
use crate::wire::CallerIdentity;
use rig_core::completion::CompletionRequest;
use rig_core::message::{
    CallId, Message, ProviderCallId, ToolName, ToolResultContent, UserContent,
};
use rig_core::operation::Completion;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Body, Mode, Operation as _, Wire as _};
use serde_json::{Value, json};
use std::sync::Arc;

fn responses() -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        "gpt-5.4",
    )
}

/// The `input` of `request` as each writer writes it: [`input_items`], and
/// the body [`CodexResponses::encode_body`] makes of the request prepared as
/// Rig's driver prepares it.
fn inputs(request: CompletionRequest) -> [Vec<Value>; 2] {
    let projected = input_items(&responses(), request.clone()).expect("the request encodes");
    let wire = CodexResponses::new(responses()).expect("the ChatGPT dialect speaks Codex");
    let prepared =
        Completion::prepare(request, &wire.wire.describe()).expect("Rig prepares the request");
    let (_, body) = wire
        .encode_body(prepared, Mode::Streaming)
        .expect("the request encodes");
    let sent = body
        .get::<Vec<Value>>("input")
        .expect("the body has an `input`")
        .expect("`input` is an array");
    [projected, sent]
}

fn tool(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

/// A user message holding the one result `text` of `name` for `call`.
fn result(call: CallId, name: ToolName, text: &str) -> UserContent {
    UserContent::tool_result(call, name, vec![ToolResultContent::text(text)])
}

fn call_x() -> CallId {
    CallId::from(ProviderCallId::new("call_x").expect("a call id"))
}

/// A custom tool `name` as a caller declares it.
fn declared(name: &str) -> Value {
    json!({"type": "custom", "name": name, "description": "", "format": {"type": "text"}})
}

/// The custom call `call_x` of `first_tool` that the provider holds, as the
/// caller hands it back.
fn held_call() -> Value {
    json!({
        "type": "custom_tool_call", "call_id": "call_x",
        "name": "first_tool", "input": "first input"
    })
}

/// A request that continues what the provider holds: `held_call` as a
/// caller `input` item, its result named `first_tool`, then the turn in
/// which the provider made the call `second` under the same id `call_x`,
/// and its result named after `second`'s tool. It declares `tools`.
fn reusing_call_x(second: &Value, tools: Value) -> CompletionRequest {
    let turn = folded_call(second);
    let [call] = turn.tool_calls().collect::<Vec<_>>()[..] else {
        panic!("the turn makes one call: {:?}", turn.choice);
    };
    let mut request = CompletionRequest::new(Message::User {
        content: vec![result(call_x(), tool("first_tool"), "first output")],
    });
    request.chat_history.extend([
        Arc::new(turn.message().expect("the turn's message")),
        Arc::new(Message::User {
            content: vec![result(
                call.id.clone(),
                call.function.name.clone(),
                "second output",
            )],
        }),
    ]);
    request.additional_params = Some(json!({ "tools": tools }));
    with_provider_options(request, json!({ "input": [held_call()] }))
}

/// Two custom outputs under one call id are each named after their own
/// result: the earlier one is not named after the later result's tool.
#[test]
fn each_custom_output_under_a_reused_call_id_is_named_after_its_own_result() {
    let second = json!({
        "type": "custom_tool_call", "id": "ctc_2", "call_id": "call_x",
        "name": "second_tool", "input": "second input", "status": "completed"
    });
    let request = reusing_call_x(
        &second,
        json!([declared("first_tool"), declared("second_tool")]),
    );
    let expected = vec![
        held_call(),
        json!({
            "type": "custom_tool_call_output", "call_id": "call_x",
            "name": "first_tool", "output": "first output"
        }),
        second.clone(),
        json!({
            "type": "custom_tool_call_output", "call_id": "call_x",
            "name": "second_tool", "output": "second output"
        }),
    ];
    for input in inputs(request) {
        assert_eq!(input, expected);
    }
}

/// A function output under the same id counts among the outputs of that id,
/// and is sent as Rig wrote it, with no name.
#[test]
fn a_function_output_under_the_reused_id_counts_and_is_not_named() {
    let second = json!({
        "type": "function_call", "id": "fc_2", "call_id": "call_x",
        "name": "second_tool", "arguments": "{\"q\":\"rig\"}", "status": "completed"
    });
    let function = json!({
        "type": "function", "name": "second_tool", "description": "",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
    });
    let request = reusing_call_x(&second, json!([declared("first_tool"), function]));
    let expected = vec![
        held_call(),
        json!({
            "type": "custom_tool_call_output", "call_id": "call_x",
            "name": "first_tool", "output": "first output"
        }),
        second.clone(),
        json!({
            "type": "function_call_output", "call_id": "call_x",
            "output": "second output", "status": "completed"
        }),
    ];
    for input in inputs(request) {
        assert_eq!(input, expected);
    }
}

/// One call, one result, one custom output: named after the tool that ran.
#[test]
fn one_result_under_one_id_names_its_output() {
    let request = answering(custom_call_item(), vec![ToolResultContent::text("ok")]);
    let expected = vec![
        json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say hi"}]}),
        custom_call_item(),
        json!({
            "type": "custom_tool_call_output", "call_id": "call_c",
            "name": "apply_patch", "output": "ok"
        }),
    ];
    for input in inputs(request) {
        assert_eq!(input, expected);
    }
}

/// Two results under the id of the one call the request holds. Rig's
/// preparation keeps only the first, so a prepared request sends one output,
/// named after it. A wire given the request unprepared sends both, each
/// named after its own result, in the order the results stand.
#[test]
fn two_results_under_one_call_are_named_in_their_order() {
    let turn = folded_call(&custom_call_item());
    let [call] = turn.tool_calls().collect::<Vec<_>>()[..] else {
        panic!("the turn makes one call: {:?}", turn.choice);
    };
    let mut request = CompletionRequest::new(Message::user("say hi"));
    request.chat_history.extend([
        Arc::new(turn.message().expect("the turn's message")),
        Arc::new(Message::User {
            content: vec![
                result(call.id.clone(), tool("apply_patch"), "first answer"),
                result(call.id.clone(), tool("check_patch"), "second answer"),
            ],
        }),
    ]);
    request.additional_params =
        Some(json!({ "tools": [declared_custom(), declared("check_patch")] }));
    let user = json!({
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "say hi"}]
    });
    let first = json!({
        "type": "custom_tool_call_output", "call_id": "call_c",
        "name": "apply_patch", "output": "first answer"
    });
    let prepared = vec![user.clone(), custom_call_item(), first.clone()];
    for input in inputs(request.clone()) {
        assert_eq!(input, prepared);
    }

    let wire = CodexResponses::new(responses())
        .expect("the ChatGPT dialect speaks Codex")
        .with_caller_identity(
            CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
                .expect("a valid caller identity"),
        );
    let encoded = wire
        .encode(request, Mode::Streaming)
        .expect("the wire encodes the request unprepared");
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a Responses body is bytes");
    };
    let body: Value = serde_json::from_slice(bytes).expect("the body is JSON");
    assert_eq!(
        body.get("input"),
        Some(&json!([
            user,
            custom_call_item(),
            first,
            {
                "type": "custom_tool_call_output", "call_id": "call_c",
                "name": "check_patch", "output": "second answer"
            },
        ]))
    );
}
