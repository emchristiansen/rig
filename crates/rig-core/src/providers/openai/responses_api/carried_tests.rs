use std::error::Error;

use serde_json::{Value, json};

use super::{Delivery, carried_message_parts, carried_reasoning};
use crate::completion::{CompletionRequest, ReplayTarget};
use crate::error::ProviderError;
use crate::message::{
    Api, AssistantContent, AssistantMessage, CarriedPayload, CarriedReasoningPart,
    CarriedTextReplayError, Message, Origin, ProviderItem, UnreplayableProviderItem,
    UnrepresentableOpaqueContent,
};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::wire::Responses;
use crate::wire::{Mode, Operation, Wire};

fn wire() -> Responses {
    OpenAIConfig::new("test-key").responses("gpt-5.4")
}

fn request(parts: Vec<AssistantContent>) -> CompletionRequest {
    CompletionRequest::from(vec![
        Message::user("question"),
        Message::Assistant(AssistantMessage::new(parts)),
        Message::user("next"),
    ])
}

fn encoded_input(target: &Responses, request: CompletionRequest) -> Vec<Value> {
    let request = Completion::prepare(request, &target.describe()).expect("prepare real history");
    let encoded = target
        .encode(request, Mode::Unary)
        .expect("encode real request");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .expect("input array")
        .clone()
}

fn carried(raw: Value, issuer: Option<&str>) -> AssistantContent {
    AssistantContent::ProviderItem(ProviderItem {
        payload: CarriedPayload::new(raw, issuer.map(str::to_owned)),
    })
}

#[test]
fn whole_items_replay_across_models_without_native_or_origin() {
    for issuer in [Some("openai"), None] {
        for raw in [
            json!({"type":"compaction","encrypted_content":"opaque","future":[null,1]}),
            json!([1, null, "untyped"]),
        ] {
            let part = carried(raw.clone(), issuer);
            assert!(part.native_item().is_none());
            let mut req = request(vec![part]);
            req.model = Some("different-model".to_owned());
            let input = encoded_input(&wire(), req);
            assert!(
                input.contains(&raw),
                "whole raw item survives the real request boundary"
            );
        }
    }
}

#[test]
fn whole_items_that_resemble_system_messages_are_never_lifted() {
    let raw = json!({"type":"message","role":"system","content":[{"type":"input_text","text":"raw system lookalike"}],"extra":null});
    let mut target = wire();
    target.system_instructions = super::SystemInstructionsPlacement::AllInstructions;
    let input = encoded_input(&target, request(vec![carried(raw.clone(), Some("openai"))]));
    assert!(input.contains(&raw));
}

#[test]
fn canonical_system_only_input_preserves_the_fork_empty_input_boundary() {
    let request = CompletionRequest::from(vec![Message::System {
        content: "system only".to_owned(),
    }]);
    let retained = wire()
        .responses_request(&request, Delivery::Http(Mode::Unary))
        .expect("default Instructions keeps generating all-system input");
    assert_eq!(retained.get("input"), Some(&json!([{"type":"message","role":"system","content":[{"type":"input_text","text":"system only"}]}])));
    assert!(retained.get("instructions").is_none());
    let mut all = wire();
    all.system_instructions = super::SystemInstructionsPlacement::AllInstructions;
    let error = all
        .responses_request(&request, Delivery::Http(Mode::Unary))
        .expect_err("AllInstructions lifts all canonical system items before NonEmpty check");
    assert!(error.to_string().contains("OpenAI Responses request input must contain at least one non-system item (system messages were lifted into the top-level `instructions` field)"));
    let body = wire()
        .responses_request_with_identity(
            &request,
            Delivery::WebSocket,
            None,
            super::InputRequirement::MayBeEmpty,
        )
        .expect("prewarm allows lifted empty input");
    assert_eq!(body.get("input"), Some(&json!([])));
    assert_eq!(body.get("instructions"), Some(&json!("system only")));
    let all_warmup = all.responses_request_with_identity(
        &request, Delivery::WebSocket, None, super::InputRequirement::MayBeEmpty,
    ).expect("AllInstructions warmup allows the lifted empty input");
    assert_eq!(all_warmup.get("input"), body.get("input"));
    assert_eq!(all_warmup.get("instructions"), body.get("instructions"));
    let as_messages = wire().with_system_instructions_as_messages();
    for requirement in [super::InputRequirement::NonEmpty, super::InputRequirement::MayBeEmpty] {
        let body = as_messages.responses_request_with_identity(&request, Delivery::WebSocket, None, requirement).expect("InputSystemMessages retains canonical systems for both requirements");
        assert_eq!(body.get("input"), retained.get("input"));
        assert!(body.get("instructions").is_none());
    }
}

#[test]
fn canonical_system_placement_keeps_the_fork_leading_and_mid_conversation_boundaries() {
    use super::{InputRequirement, SystemInstructionsPlacement};
    let request = CompletionRequest::from(vec![
        Message::System { content: " first ".to_owned() },
        Message::user("hello"),
        Message::System { content: " mid ".to_owned() },
        Message::user("again"),
    ]);
    for (placement, instructions, roles) in [
        (SystemInstructionsPlacement::Instructions, Some("first"), vec!["user", "system", "user"]),
        (SystemInstructionsPlacement::AllInstructions, Some("first\n\nmid"), vec!["user", "user"]),
        (SystemInstructionsPlacement::InputSystemMessages, None, vec!["system", "user", "system", "user"]),
    ] {
        let target = wire().with_system_instructions_placement(placement);
        for requirement in [InputRequirement::NonEmpty, InputRequirement::MayBeEmpty] {
            let body = target.responses_request_with_identity(
                &request, Delivery::WebSocket, None, requirement,
            ).expect("mixed history has non-system input for each placement");
            let sent_roles: Vec<_> = body.get("input").and_then(Value::as_array).expect("input")
                .iter().map(|item| item["role"].as_str().expect("message role")).collect();
            assert_eq!(sent_roles, roles);
            assert_eq!(body.get("instructions"), instructions.map(|text| json!(text)).as_ref());
        }
    }
}

#[test]
fn foreign_whole_item_is_a_typed_preflight_and_encode_refusal() {
    let raw = json!({"type":"future_item","id":"opaque"});
    let req = request(vec![carried(raw, Some("foreign/issuer"))]);
    for error in [
        Completion::prepare(req.clone(), &wire().describe()).expect_err("preflight refuses"),
        ProviderError::from(
            wire()
                .responses_request(&req, Delivery::Http(Mode::Unary))
                .expect_err("final encoding refuses"),
        ),
    ] {
        let refusal = error
            .source()
            .expect("typed source")
            .downcast_ref::<UnreplayableProviderItem>()
            .expect("whole item refusal");
        assert_eq!(refusal.issuer.as_deref(), Some("foreign/issuer"));
        assert_eq!(refusal.item_type.as_deref(), Some("future_item"));
    }
}

#[test]
fn failed_turn_keeps_carried_items_and_its_exact_origin() {
    let target = wire();
    let origin = Origin::new(
        Api::from_static("openai.responses"),
        "openai",
        "older-model",
    );
    let raw = json!({"type":"compaction","encrypted_content":"failed-turn-data"});
    let turn = AssistantMessage::aborted(
        Some(origin.clone()),
        vec![
            carried(raw.clone(), Some("openai")),
            AssistantContent::text("discard failed text"),
        ],
        "cut",
    );
    let req = CompletionRequest::from(vec![
        Message::user("question"),
        Message::Assistant(turn),
        Message::user("next"),
    ]);
    let prepared = Completion::prepare(req, &target.describe()).expect("prepare failed turn");
    let turn = prepared
        .chat_history
        .iter()
        .find_map(|message| match message {
            Message::Assistant(turn) => Some(turn),
            _ => None,
        })
        .expect("carried failed turn retained");
    assert_eq!(turn.origin.as_ref(), Some(&origin));
    assert_eq!(turn.content, [carried(raw.clone(), Some("openai"))]);
    let encoded = target
        .encode(prepared, Mode::Unary)
        .expect("encode retained item");
    assert!(
        crate::test_utils::json_body(&encoded.request)["input"]
            .as_array()
            .expect("input")
            .contains(&raw)
    );
}

#[test]
fn edited_known_parts_keep_opaque_siblings_in_exact_order_and_message_groups() {
    let raw = json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[
        {"type":"output_text","text":"before","annotations":[],"null":null},
        {"type":"future_part","payload":{"nested":[1,null]},"text":"raw"},
        {"type":"refusal","refusal":"before refusal","future":true},
        {"type":"output_text","text":"last","empty":{}}
    ]});
    let mut parts =
        carried_message_parts(&raw, 3, Some("openai".to_owned())).expect("real conversion");
    *parts[0].text_mut() = "edited".to_owned();
    *parts[2].text_mut() = "edited refusal".to_owned();
    parts[1].payload.item["payload"] = json!({"nested":[2,null,"raw-only edit"]});
    let raw_without_id =
        json!({"type":"message","role":"assistant","content":[{"type":"future_part","x":2}]});
    let mut all: Vec<_> = parts
        .into_iter()
        .map(AssistantContent::CarriedText)
        .collect();
    all.extend(
        carried_message_parts(&raw_without_id, 4, None)
            .expect("id-less conversion")
            .into_iter()
            .map(AssistantContent::CarriedText),
    );
    let input = encoded_input(&wire(), request(all));
    let message = input
        .iter()
        .find(|item| item["id"] == "msg_1")
        .expect("first group");
    let mut expected = raw.clone();
    expected["content"][0]["text"] = json!("edited");
    expected["content"][2]["refusal"] = json!("edited refusal");
    expected["content"][1]["payload"] = json!({"nested":[2,null,"raw-only edit"]});
    assert_eq!(message, &expected);
    let second = input
        .iter()
        .find(|item| item["content"] == raw_without_id["content"])
        .expect("second group");
    assert!(
        second.get("id").is_none(),
        "no response or synthetic ID promotion"
    );
}

#[test]
fn edited_opaque_text_refuses_before_raw_input_override_can_hide_it() {
    let raw = json!({"type":"message","content":[{"type":"future_part","text":"raw"}]});
    let mut parts = carried_message_parts(&raw, 0, None).expect("convert opaque marker");
    *parts[0].text_mut() = " ".to_owned();
    let mut req = request(
        parts
            .into_iter()
            .map(AssistantContent::CarriedText)
            .collect(),
    );
    req.additional_params = Some(json!({"input":[{"role":"user","content":"override"}]}));
    let error = ProviderError::from(
        wire()
            .responses_request(&req, Delivery::Http(Mode::Unary))
            .expect_err("marker edit refused"),
    );
    assert_eq!(
        error
            .source()
            .expect("typed source")
            .downcast_ref::<CarriedTextReplayError>(),
        Some(&CarriedTextReplayError::EditedOpaqueText)
    );
    assert!(error.to_string().contains("an opaque Responses message part carries non-empty text: the raw part replays verbatim, so the text cannot be sent"));
}

#[test]
fn foreign_opaque_reasoning_converts_with_issuer_then_final_request_refuses() {
    let raw = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"future_summary","x":null}],"content":[{"type":"future_reasoning","x":[1]}]});
    let reasoning = carried_reasoning(&raw, Some("foreign/issuer".to_owned()))
        .expect("conversion retains foreign provenance");
    assert_eq!(reasoning.provider(), Some("foreign/issuer"));
    assert_eq!(reasoning.payload.item, raw);
    for part in &reasoning.content {
        let payload = match part {
            CarriedReasoningPart::OpaqueSummary(payload)
            | CarriedReasoningPart::OpaqueContent(payload) => payload,
            _ => panic!("opaque source"),
        };
        assert_eq!(payload.provider.as_deref(), Some("foreign/issuer"));
    }
    let target = wire();
    target
        .check_carried_reasoning(&reasoning)
        .expect("preflight retains provenance for effective-model encoding");
    let req = Completion::prepare(
        request(vec![AssistantContent::CarriedReasoning(reasoning)]),
        &target.describe(),
    )
    .expect("prepare retains opaque reasoning");
    let error = ProviderError::from(
        target
            .responses_request(&req, Delivery::Http(Mode::Unary))
            .expect_err("final request refuses"),
    );
    let refusal = error
        .source()
        .expect("typed source")
        .downcast_ref::<UnrepresentableOpaqueContent>()
        .expect("nested refusal");
    assert_eq!(
        refusal.wire,
        "OpenAI Responses with incompatible reasoning provenance"
    );
}

#[test]
fn own_and_unknown_reasoning_replay_edited_known_parts_and_raw_interleaving() {
    let raw = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"before","future":null},{"type":"opaque_summary","data":[1]},{"type":"summary_text","text":"after"}],"content":[{"type":"reasoning_text","text":"thought","extra":{}},{"type":"opaque_reasoning","data":null},{"type":"reasoning_text","text":"last"}],"signature":"signed","encrypted_content":"cipher","extra":[null]});
    for issuer in [Some("openai".to_owned()), None] {
        let mut reasoning = carried_reasoning(&raw, issuer).expect("convert reasoning");
        let CarriedReasoningPart::Summary(text) = &mut reasoning.content[0] else {
            panic!("first summary");
        };
        *text = "edited".to_owned();
        let input = encoded_input(
            &wire(),
            request(vec![AssistantContent::CarriedReasoning(reasoning)]),
        );
        let item = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("reasoning sent");
        let mut expected = raw.clone();
        expected["summary"][0]["text"] = json!("edited");
        assert_eq!(item, &expected);
    }
}

fn function_call(id: crate::message::CallId) -> crate::message::ToolCall {
    crate::message::ToolCall::new(
        id,
        crate::message::ToolFunction::new(
            crate::message::ToolName::new("function").expect("tool name"),
            json!({}),
        ),
    )
}

fn direct_input(request: CompletionRequest) -> Vec<Value> {
    let encoded = wire()
        .encode(request, Mode::Unary)
        .expect("real Responses encode");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .expect("input")
        .clone()
}

#[test]
fn occurrence_aliases_survive_completed_identity_reuse_and_filtered_user_content() {
    use crate::message::{CallId, ToolResultContent, UserContent};
    let call = function_call(CallId::from_wire(""));
    let result = call.result(vec![ToolResultContent::text("done")]);
    let request = CompletionRequest::from(vec![
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("before"),
            AssistantContent::ToolCall(call.clone()),
        ])),
        Message::User {
            content: vec![
                UserContent::text(" "),
                UserContent::ToolResult(result.clone()),
            ],
        },
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(result)],
        },
    ]);
    let before = serde_json::to_value(&request.chat_history).expect("history serializes");
    let input = direct_input(request.clone());
    let spellings: Vec<_> = input
        .iter()
        .filter_map(|item| item.get("call_id").and_then(Value::as_str))
        .collect();
    assert_eq!(spellings, ["tool-0", "tool-0", "tool-1", "tool-1"]);
    assert_eq!(
        input[1],
        json!({"type":"function_call","call_id":"tool-0","name":"function","arguments":"{}"})
    );
    assert_eq!(
        input[2],
        json!({"type":"function_call_output","call_id":"tool-0","output":"done","status":"completed"})
    );
    assert_eq!(
        before,
        serde_json::to_value(&request.chat_history).expect("history unchanged")
    );
}

#[test]
fn occurrence_coordinates_pair_reversed_function_custom_results_and_keep_raw_order() {
    use crate::message::{CallId, CustomToolCall, ToolName, ToolResultContent, UserContent};
    let custom = CustomToolCall::new(
        CallId::from_wire(""),
        ToolName::new("custom").expect("tool name"),
        "raw custom input",
    );
    let function = function_call(CallId::from_wire(""));
    let raw = json!({"type":"compaction","encrypted_content":"exact","extra":[null]});
    let request = CompletionRequest::from(vec![
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("before"),
            AssistantContent::CustomToolCall(custom.clone()),
            carried(raw.clone(), Some("openai")),
            AssistantContent::ToolCall(function.clone()),
        ])),
        Message::User {
            content: vec![
                UserContent::text(" "),
                UserContent::ToolResult(
                    function.result(vec![ToolResultContent::text("function result")]),
                ),
                UserContent::text(""),
                UserContent::ToolResult(
                    custom.result(vec![ToolResultContent::text("custom result")]),
                ),
            ],
        },
    ]);
    let input = direct_input(request);
    assert_eq!(
        input[1],
        json!({"type":"custom_tool_call","call_id":"tool-0","name":"custom","input":"raw custom input"})
    );
    assert_eq!(input[2], raw);
    assert_eq!(
        input[3],
        json!({"type":"function_call","call_id":"tool-1","name":"function","arguments":"{}"})
    );
    assert_eq!(
        input[4],
        json!({"type":"function_call_output","call_id":"tool-1","output":"function result","status":"completed"})
    );
    assert_eq!(
        input[5],
        json!({"type":"custom_tool_call_output","call_id":"tool-0","name":"custom","output":"custom result"})
    );
}

#[test]
fn occurrence_planning_errors_remain_typed_at_the_responses_encoder() {
    use crate::message::{CallId, ToolResultContent, UserContent};
    use crate::providers::internal::wire_ids::WireIdError;
    let call = function_call(CallId::from_wire(""));
    let result = call.result(vec![ToolResultContent::text("done")]);
    for (history, expected) in [
        (
            vec![Message::Assistant(AssistantMessage::new(vec![
                AssistantContent::ToolCall(call.clone()),
                AssistantContent::ToolCall(call.clone()),
            ]))],
            WireIdError::Ambiguous {
                message: 0,
                content: 1,
            },
        ),
        (
            vec![Message::User {
                content: vec![
                    UserContent::text(" "),
                    UserContent::ToolResult(result.clone()),
                ],
            }],
            WireIdError::OrphanResult {
                message: 0,
                content: 1,
            },
        ),
        (
            vec![
                Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                    call,
                )])),
                Message::User {
                    content: vec![
                        UserContent::ToolResult(result.clone()),
                        UserContent::ToolResult(result),
                    ],
                },
            ],
            WireIdError::DuplicateResult {
                message: 1,
                content: 1,
            },
        ),
    ] {
        let error = ProviderError::from(
            wire()
                .encode(CompletionRequest::from(history), Mode::Unary)
                .expect_err("planning refuses malformed transcript"),
        );
        assert_eq!(
            error
                .source()
                .expect("typed source")
                .downcast_ref::<WireIdError>(),
            Some(&expected)
        );
    }
}

#[test]
fn decoded_native_call_replay_uses_the_planned_occurrence_and_keeps_other_bytes() {
    use crate::message::{ToolResultContent, UserContent};
    use crate::wire::WireFrame;
    for (provider_id, expected_id) in [("call_1", "call_1"), ("", "tool-0")] {
        let raw = json!({"type":"function_call","id":"fc_1","call_id":provider_id,"name":"function","arguments":"{}","status":"completed","future":{"x":null}});
        let response = crate::test_utils::history::decode(&wire(), Mode::Unary, [WireFrame::Text(json!({
            // G139 outer fixture-clock scaffolding; the full call stays exact.
            "id":"resp_1","object":"response","created_at":0,"model":"gpt-5.4","status":"completed","output":[raw.clone()]
        }).to_string())]).expect("real Responses decoder creates native custody");
        let call = response.tool_calls().next().expect("decoded call");
        assert_eq!(response.choice[0].native_item(), Some(&raw));
        let result = call.result(vec![ToolResultContent::text("done")]);
        let request = CompletionRequest::from(vec![
            response.message().expect("decoded turn"),
            Message::User {
                content: vec![UserContent::text(" "), UserContent::ToolResult(result)],
            },
        ]);
        let input = direct_input(request);
        let mut expected = raw;
        expected["call_id"] = json!(expected_id);
        assert_eq!(input[0], expected);
        assert_eq!(input[1]["call_id"], json!(expected_id));
    }
}

fn never_done_function_frames(arguments: &str) -> [Value; 4] {
    let response = |status| json!({
        "id":"resp_arguments", "object":"response", "created_at":1,
        "status":status, "model":"gpt-5.4", "output":[]
    });
    [
        json!({"type":"response.created","sequence_number":0,"response":response("in_progress")}),
        json!({"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{
            "type":"function_call","id":"fc_arguments","call_id":"call_arguments",
            "name":"function","arguments":"","status":"in_progress"
        }}),
        json!({"type":"response.function_call_arguments.delta","sequence_number":2,
            "output_index":0,"item_id":"fc_arguments","delta":arguments}),
        json!({"type":"response.completed","sequence_number":3,"response":response("completed")}),
    ]
}

fn never_done_function_stream(arguments: &str) -> crate::streaming::CompletionStream {
    let sse: String = never_done_function_frames(arguments).iter()
        .map(|frame| format!("data: {frame}\n\n")).collect();
    crate::driver::Model::new(wire(), crate::test_utils::MockStreamingClient {
        sse_bytes: sse.into(),
    }).stream(CompletionRequest::new("restate")).expect("the actual original-mode stream opens")
}

fn completed_function_response(arguments: &str) -> crate::completion::CompletionResponse {
    use crate::wire::WireFrame;
    let mut frames = never_done_function_frames(arguments).to_vec();
    let mut item = frames[1]["item"].clone();
    item["arguments"] = json!(arguments);
    item["status"] = json!("completed");
    frames.insert(3, json!({"type":"response.output_item.done","sequence_number":3,
        "output_index":0,"item":item}));
    crate::test_utils::history::decode(
        &wire(), Mode::Streaming,
        frames.into_iter().map(|frame| WireFrame::Text(frame.to_string())),
    ).expect("the separately completed fixture decodes through the original route")
}

fn assert_failed_unfinished_function(response: &crate::completion::CompletionResponse) {
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one canonical unfinished function call");
    };
    assert_eq!(call.id.provider().expect("original provider ID").as_str(), "call_arguments");
    assert_eq!(call.function.name.as_str(), "function");
    assert_eq!(call.function.namespace, None);
    assert_eq!(response.raw["output"][0]["id"], "fc_arguments");
    assert_eq!(response.raw["output"][0]["call_id"], "call_arguments");
    assert_eq!(response.raw["output"][0]["name"], "function");
    assert!(response.raw["output"][0].get("namespace").is_none());
    assert!(call.native.is_none());
    let error = "the provider ended the reply with a tool call it never finished";
    assert_eq!(response.error.as_deref(), Some(error));
    assert_eq!(response.stop(), crate::message::StopReason::Error(error.to_owned()));
    let Message::Assistant(turn) = response.message().expect("failed assistant history") else {
        panic!("assistant failure history");
    };
    assert_eq!(turn.stop, Some(crate::message::StopReason::Error(error.to_owned())));
    assert_eq!(crate::message::turn_failure(&response.choice, turn.stop.as_ref(),
        response.finish_reason().as_ref()).as_deref(), Some(
            "the turn failed, so none of its tool calls ran: the provider ended the reply with a tool call it never finished"
        ));
}

fn prepared_function_rebuild(
    response: &crate::completion::CompletionResponse,
) -> CompletionRequest {
    use crate::message::{ToolName, ToolResultContent, UserContent};
    let call = response.tool_calls().next().expect("decoded function call");
    let mut request = CompletionRequest::from(vec![
        Message::user("question"),
        response.message().expect("actual decoded assistant message"),
        Message::User {
            content: vec![UserContent::ToolResult(call.result(vec![ToolResultContent::text("done")]))],
        },
    ]).tools(vec![crate::completion::ToolDefinition {
        name: ToolName::new("function").expect("tool name"),
        description: "function".to_owned(),
        parameters: json!({"type":"object"}),
    }]);
    request.model = Some("different-model".to_owned());
    Completion::prepare(request, &wire().describe()).expect("prepare actual different-model history")
}

#[tokio::test]
async fn never_done_non_object_arguments_keep_fork_spelling_in_failed_unfinished_inspection() {
    use futures::StreamExt;
    for (arguments, expected) in [
        ("17", "17"),
        (" [ 1, {\"x\": 2}, null ] ", r#"[1,{"x":2},null]"#),
        (r#""\u0061""#, r#""a""#),
        (r#""{\"x\":1}""#, r#""{\"x\":1}""#),
    ] {
        let mut stream = never_done_function_stream(arguments);
        while let Some(item) = stream.next().await {
            let item = item.expect("valid never-done arguments remain inspection data");
            assert!(!matches!(item, crate::streaming::Item::Event(
                crate::streaming::StreamEvent::Discard { .. }
            )));
        }
        let response = stream.finish().await.expect("the actual failed folded response");
        let call = response.tool_calls().next().expect("decoded scalar/array/string call");
        assert!(call.function.arguments.is_empty());
        assert_eq!(call.function.invalid_arguments.as_deref(), Some(expected));
        assert_failed_unfinished_function(&response);
    }
}

#[test]
fn completed_non_object_function_arguments_use_fork_spelling_after_model_change() {
    for (arguments, expected) in [
        ("17", "17"),
        (" [ 1, {\"x\": 2}, null ] ", r#"[1,{"x":2},null]"#),
        (r#""\u0061""#, r#""a""#),
        (r#""{\"x\":1}""#, r#""{\"x\":1}""#),
    ] {
        let response = completed_function_response(arguments);
        let call = response.tool_calls().next().expect("decoded scalar/array/string call");
        assert!(call.function.arguments.is_empty());
        assert_eq!(call.function.invalid_arguments.as_deref(), Some(expected));
        let prepared = prepared_function_rebuild(&response);
        let rebuilt = prepared.chat_history.iter().find_map(|message| match message {
            Message::Assistant(turn) => turn.content.iter().find_map(|part| match part {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            }),
            _ => None,
        }).expect("prepared canonical call");
        assert!(rebuilt.native.is_none(), "the actual different-model boundary rebuilds");
        assert_eq!(rebuilt.function.invalid_arguments.as_deref(), Some(expected));
        let input = direct_input(prepared);
        let sent = input.iter().find(|item| item["type"] == "function_call").expect("encoded function call");
        assert_eq!(sent["arguments"], json!(expected));
    }
}

#[tokio::test]
async fn never_done_object_arguments_remain_canonical_in_failed_unfinished_inspection() {
    use futures::StreamExt;
    for (edit, stale_raw_object) in [(false, false), (true, false), (true, true)] {
        let mut stream = never_done_function_stream(" { \"old\": 1 } ");
        while let Some(item) = stream.next().await {
            let item = item.expect("valid never-done arguments remain inspection data");
            assert!(!matches!(item, crate::streaming::Item::Event(
                crate::streaming::StreamEvent::Discard { .. }
            )));
        }
        let response = stream.finish().await.expect("the actual failed folded response");
        let call = response.tool_calls().next().expect("decoded object call");
        assert!(call.function.invalid_arguments.is_none());
        assert_eq!(call.function.arguments_value(), json!({"old":1}), "case {edit}/{stale_raw_object}");
        assert_failed_unfinished_function(&response);
    }
}

#[test]
fn completed_object_arguments_use_current_canonical_object_on_rebuild() {
    for (edit, stale_raw_object) in [(false, false), (true, false), (true, true)] {
        let response = completed_function_response(" { \"old\": 1 } ");
        let call = response.tool_calls().next().expect("decoded object call");
        assert!(call.function.invalid_arguments.is_none());
        assert_eq!(call.function.arguments_value(), json!({"old":1}));
        let mut prepared = prepared_function_rebuild(&response);
        if edit {
            for message in &mut prepared.chat_history {
                if let Message::Assistant(turn) = message {
                    for part in &mut turn.content {
                        if let AssistantContent::ToolCall(call) = part {
                            call.function.arguments = serde_json::from_value(json!({"edited":2})).expect("canonical object edit");
                            assert!(call.function.invalid_arguments.is_none());
                            if stale_raw_object {
                                call.function.invalid_arguments = Some(r#"{"old":1}"#.to_owned());
                            }
                        }
                    }
                }
            }
        }
        let input = direct_input(prepared);
        let sent = input.iter().find(|item| item["type"] == "function_call").expect("encoded function call");
        assert_eq!(sent["arguments"], json!(if edit { r#"{"edited":2}"# } else { r#"{"old":1}"# }));
    }
}
