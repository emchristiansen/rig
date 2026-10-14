//! Synthetic received replies use the real Responses decoder and fold.
//! Boundary controls change labelled clones and retain their genuine source.

use crate::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CarriedPosition, CarriedReasoning, CarriedReasoningPart, CarriedText,
    CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, Message, ToolName,
    ToolResultContent, UnrepresentableOpaqueContent, UserContent,
};
use crate::operation::Completion;
use crate::providers::openai::{OpenAIConfig, responses_api::wire::Responses};
use crate::test_utils::{decode_reply, json_body};
use crate::wire::{Encoded, Mode, Operation, Wire, WireFrame};
use serde_json::{Value, json};

struct Received {
    body: Value,
    response: CompletionResponse,
}

fn wire() -> Responses {
    OpenAIConfig::new("synthetic-test-key").responses("gpt-5.4")
}

fn reasoning() -> Value {
    json!({"type":"reasoning", "id":"rs_pair", "summary":[
        {"type":"summary_text", "text":"plan", "future_summary_field":[null,7]}],
        "content":[{"type":"reasoning_text", "text":"thought", "future_content_field":true}],
        "encrypted_content":" exact cipher\n ", "future_reasoning_field":{"kept":true}})
}
fn call() -> Value {
    json!({"type":"function_call", "id":"fc_pair", "call_id":"call_pair",
        "name":"f", "arguments":"{\"a\":1}", "status":"completed"})
}
fn answer_parts() -> Value {
    json!([{ "type":"output_text", "text":"answer", "annotations":[], "future_part":[null,true]},
        {"type":"refusal", "refusal":"declined", "future_refusal":"kept"}])
}
fn answer(parts: Value) -> Value {
    json!({"type":"message", "id":"msg_pair", "role":"assistant", "status":"completed",
        "phase":"final_answer", "content":parts, "future_message":{"exact":[null,true]}})
}
fn received(wire: &Responses, output: Vec<Value>) -> Received {
    let body = json!({"id":"resp_pair", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-5.4", "output":output, "tools":[]});
    let response = decode_reply(
        wire,
        &CompletionRequest::new("synthetic fixture"),
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
        Value::Null,
    )
    .expect("actual decoder and fold");
    assert_eq!(response.raw, body);
    let stored = serde_json::to_vec(&response).expect("whole response serde");
    let restored: CompletionResponse =
        serde_json::from_slice(&stored).expect("response round trip");
    assert_eq!(restored, response);
    assert_eq!(response.head().origin.as_ref(), Some(&response.origin));
    Received { body, response }
}
fn reasoning_mut(history: &mut [Message]) -> &mut CarriedReasoning {
    let Message::Assistant(turn) = &mut history[1] else {
        panic!("decoded assistant");
    };
    let AssistantContent::CarriedReasoning(reason) = &mut turn.content[0] else {
        panic!("carried reasoning");
    };
    reason
}
fn text_mut(history: &mut [Message]) -> &mut CarriedText {
    let Message::Assistant(turn) = &mut history[1] else {
        panic!("decoded assistant");
    };
    let AssistantContent::CarriedText(text) = &mut turn.content[1] else {
        panic!("immediate carried answer");
    };
    text
}
fn history(source: &Received) -> Vec<Message> {
    let results: Vec<_> = source
        .response
        .tool_calls()
        .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")])))
        .collect();
    vec![
        Message::user("q"),
        source.response.message().expect("decoded assistant"),
        if results.is_empty() {
            Message::user("next")
        } else {
            Message::User { content: results }
        },
    ]
}
// Tuple return keeps assertions outside a Result-returning function.
fn send<W: Wire<Op = Completion, Payload = Encoded>>(
    source: &Received,
    target: &W,
    input: &[Message],
    tools: bool,
    stored: bool,
) -> (Result<Value, ProviderError>, ()) {
    let before = input.to_vec();
    let history_serde = serde_json::to_vec(input).expect("whole input history serde");
    let source_before = source.response.clone();
    let source_serde = serde_json::to_vec(&source.response).expect("received response serde");
    let received_before = source.body.clone();
    let mut request = CompletionRequest::from(input.to_vec());
    if tools {
        request.tools = vec![ToolDefinition {
            name: ToolName::new("f").expect("fixture name"),
            description: "A tool".to_owned(),
            parameters: json!({"type":"object"}),
        }];
    }
    request.additional_params = Some(json!({"store":stored}));
    let outcome = Completion::prepare(request, &target.describe()).and_then(|request| {
        target
            .encode(request, Mode::Unary)
            .map(|encoded| json_body(&encoded.request))
            .map_err(ProviderError::from)
    });
    assert_eq!(
        input, before,
        "every subject preserves its complete input history and choice"
    );
    assert_eq!(
        serde_json::to_vec(input).expect("history after"),
        history_serde
    );
    assert_eq!(
        source.response, source_before,
        "every subject preserves its genuine received source"
    );
    assert_eq!(
        serde_json::to_vec(&source.response).expect("response after"),
        source_serde
    );
    assert_eq!(source.body, received_before);
    (outcome, ())
}
fn user(text: &str) -> Value {
    json!({"type":"message", "role":"user", "content":[{"type":"input_text","text":text}]})
}
fn called() -> Value {
    json!({"type":"message", "role":"assistant", "status":"completed", "id":"msg_rig_1",
        "content":[{"type":"output_text", "text":"[called tool f with {\"a\":1}]", "annotations":[]}]})
}
fn carried_answer(parts: Value) -> Value {
    json!({"type":"message", "role":"assistant", "status":"completed", "id":"msg_pair",
        "phase":"final_answer", "content":parts})
}
fn expected(input: Vec<Value>, tools: bool, stored: bool) -> Value {
    let mut body = json!({"model":"gpt-5.4", "input":input, "store":stored});
    if !stored {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    if tools {
        body["tools"] = json!([{"type":"function", "name":"f", "description":"A tool",
        "parameters":{"type":"object"}, "strict":false}]);
    }
    body
}
fn no_tools(source: &Received, target: &Responses, input: &[Message], middle: Vec<Value>) {
    let mut items = vec![user("q")];
    items.extend(middle);
    items.push(user("[tool f result] ok"));
    assert_eq!(
        send(source, target, input, false, true)
            .0
            .expect("old route encodes"),
        expected(items, false, true)
    );
}

#[test]
fn issued_reasoning_and_call_preserve_tools_and_no_tools_omits_only_reasoning() {
    let target = wire();
    let source = received(&target, vec![reasoning(), call()]);
    let [
        AssistantContent::CarriedReasoning(reason),
        AssistantContent::ToolCall(tool),
    ] = source.response.choice.as_slice()
    else {
        panic!("ordered genuine reasoning and call");
    };
    assert_eq!(reason.payload.provider.as_deref(), Some("openai"));
    assert_eq!(reason.id.as_deref(), Some("rs_pair"));
    assert_eq!(source.response.choice[1].native_item(), Some(&call()));
    assert_eq!(tool.id.wire(), "call_pair");
    let input = history(&source);
    for stored in [false, true] {
        assert_eq!(
            send(&source, &target, &input, true, stored)
                .0
                .expect("tools request encodes"),
            expected(
                vec![
                    user("q"),
                    reasoning(),
                    call(),
                    json!({"type":"function_call_output",
                "call_id":"call_pair", "output":"ok", "status":"completed"})
                ],
                true,
                stored
            )
        );
        assert_eq!(
            send(&source, &target, &input, false, stored)
                .0
                .expect("no-tools request encodes"),
            expected(
                vec![user("q"), called(), user("[tool f result] ok")],
                false,
                stored
            )
        );
    }
}

#[test]
fn retained_carried_answer_preserves_cipher_id_phase_parts_and_empty_known_text() {
    let target = wire();
    for parts in [
        answer_parts(),
        json!([{"type":"output_text", "text":"", "annotations":[], "future_part":true}]),
        json!([{"type":"refusal", "refusal":"", "future_refusal":true}]),
    ] {
        let source = received(&target, vec![reasoning(), answer(parts.clone())]);
        assert!(
            matches!(&source.response.choice[1], AssistantContent::CarriedText(text)
            if text.payload.provider.as_deref() == Some("openai") && text.item_id.as_deref() == Some("msg_pair")
                && text.phase.as_deref() == Some("final_answer")
                && text.payload.position == CarriedPosition::MessageContent { item_index:1, part_index:0 })
        );
        let input = history(&source);
        for stored in [false, true] {
            assert_eq!(
                send(&source, &target, &input, false, stored)
                    .0
                    .expect("carried answer encodes"),
                expected(
                    vec![
                        user("q"),
                        reasoning(),
                        carried_answer(parts.clone()),
                        user("next")
                    ],
                    false,
                    stored
                )
            );
        }
    }
}

#[test]
fn unrelated_later_answer_never_substitutes_for_the_immediate_call() {
    let target = wire();
    let source = received(&target, vec![reasoning(), call(), answer(answer_parts())]);
    no_tools(
        &source,
        &target,
        &history(&source),
        vec![called(), carried_answer(answer_parts())],
    );
}

#[test]
fn synthetic_reasoning_provider_and_identity_boundaries_keep_old_no_tools_bodies() {
    let target = wire();
    let source = received(&target, vec![reasoning(), call()]);
    for boundary in [
        "none_provider",
        "foreign_provider",
        "no_id",
        "empty_id",
        "mismatch_id",
        "absent_raw_id",
        "wrong_kind",
        "cipher_mismatch",
        "empty_cipher",
        "missing_cipher",
        "absent_raw_cipher",
    ] {
        let mut input = history(&source);
        let reason = reasoning_mut(&mut input);
        let mut projected = reasoning();
        match boundary {
            "none_provider" => reason.payload.provider = None,
            "foreign_provider" => reason.payload.provider = Some("synthetic-foreign".to_owned()),
            "no_id" => reason.id = None,
            "empty_id" => {
                reason.id = Some(String::new());
                projected["id"] = json!("");
            }
            "mismatch_id" => {
                reason.id = Some("rs_synthetic_mismatch".to_owned());
                projected["id"] = json!("rs_synthetic_mismatch");
            }
            "absent_raw_id" => {
                reason
                    .payload
                    .item
                    .as_object_mut()
                    .expect("raw object")
                    .remove("id");
            }
            "wrong_kind" => reason.payload.item["type"] = json!("synthetic_wrong_kind"),
            "absent_raw_cipher" => {
                reason
                    .payload
                    .item
                    .as_object_mut()
                    .expect("raw object")
                    .remove("encrypted_content");
            }
            "cipher_mismatch" => {
                for part in &mut reason.content {
                    if let CarriedReasoningPart::Encrypted(cipher) = part {
                        *cipher = "synthetic mismatch".to_owned();
                    }
                }
                projected["encrypted_content"] = json!("synthetic mismatch");
            }
            "empty_cipher" => {
                for part in &mut reason.content {
                    if let CarriedReasoningPart::Encrypted(cipher) = part {
                        cipher.clear();
                    }
                }
                reason.payload.item["encrypted_content"] = json!("");
                projected["encrypted_content"] = json!("");
            }
            "missing_cipher" => {
                reason
                    .content
                    .retain(|part| !matches!(part, CarriedReasoningPart::Encrypted(_)));
                reason
                    .payload
                    .item
                    .as_object_mut()
                    .expect("raw object")
                    .remove("encrypted_content");
                projected
                    .as_object_mut()
                    .expect("expected object")
                    .remove("encrypted_content");
            }
            _ => unreachable!("listed synthetic boundary"),
        }
        let mut middle = Vec::new();
        if boundary != "no_id" {
            middle.push(projected);
        }
        middle.push(called());
        no_tools(&source, &target, &input, middle);
    }
}

#[test]
fn synthetic_first_cipher_winners_never_qualify_through_a_later_matching_cipher() {
    let target = wire();
    let source = received(&target, vec![reasoning(), call()]);
    for winner in [
        CarriedReasoningPart::Encrypted("synthetic earlier cipher".to_owned()),
        CarriedReasoningPart::Encrypted(String::new()),
        CarriedReasoningPart::Redacted {
            data: "synthetic earlier withheld data".to_owned(),
        },
        CarriedReasoningPart::Redacted {
            data: " exact cipher\n ".to_owned(),
        },
    ] {
        let mut input = history(&source);
        reasoning_mut(&mut input).content.insert(0, winner.clone());
        let mut projected = reasoning();
        projected["encrypted_content"] = match winner {
            CarriedReasoningPart::Encrypted(data) | CarriedReasoningPart::Redacted { data } => {
                json!(data)
            }
            _ => unreachable!("listed first winner"),
        };
        no_tools(&source, &target, &input, vec![projected, called()]);
    }
}

#[test]
fn opaque_reasoning_keeps_exact_replay_and_its_foreign_typed_refusal() {
    let target = wire();
    for field in ["summary", "content"] {
        let mut raw = reasoning();
        raw[field] =
            json!([{"type":"synthetic_future_reasoning_part", "payload":{"exact":[null,true]}}]);
        let source = received(&target, vec![raw.clone(), call()]);
        no_tools(&source, &target, &history(&source), vec![raw, called()]);
        let mut input = history(&source);
        reasoning_mut(&mut input).payload.provider = Some("synthetic-foreign".to_owned());
        let error = send(&source, &target, &input, false, true)
            .0
            .expect_err("foreign opaque custody refuses");
        let ProviderError::Request(source) = error else {
            panic!("typed Request refusal");
        };
        assert_eq!(
            source.downcast_ref::<UnrepresentableOpaqueContent>(),
            Some(&UnrepresentableOpaqueContent::new(
                "OpenAI Responses with incompatible reasoning provenance"
            ))
        );
    }
}

#[test]
fn malformed_reasoning_still_reaches_the_existing_encoder_error() {
    let target = wire();
    let source = received(&target, vec![reasoning(), call()]);
    for raw in [json!("synthetic malformed object"), json!([]), Value::Null] {
        let mut input = history(&source);
        reasoning_mut(&mut input).payload.item = raw;
        let error = send(&source, &target, &input, false, true)
            .0
            .expect_err("malformed raw must not disappear");
        let ProviderError::Request(source) = error else {
            panic!("existing Request refusal");
        };
        assert_eq!(
            source.to_string(),
            "a carried Responses reasoning item must be an object"
        );
    }
}

#[test]
fn synthetic_carried_follower_boundaries_omit_only_qualified_reasoning() {
    let target = wire();
    let source = received(
        &target,
        vec![
            reasoning(),
            answer(json!([{"type":"output_text", "text":"answer", "annotations":[]}])),
        ],
    );
    for boundary in [
        "none_provider",
        "foreign_provider",
        "no_id",
        "empty_id",
        "wrong_kind",
        "inspection_text",
        "inspection_media",
        "signed",
        "opaque",
        "refusal_as_output_text",
    ] {
        let mut input = history(&source);
        let text = text_mut(&mut input);
        let mut sent =
            carried_answer(json!([{"type":"output_text", "text":"answer", "annotations":[]}]));
        match boundary {
            "none_provider" => text.payload.provider = None,
            "foreign_provider" => text.payload.provider = Some("synthetic-foreign".to_owned()),
            "no_id" => {
                text.item_id = None;
                sent.as_object_mut().expect("message").remove("id");
            }
            "empty_id" => {
                text.item_id = Some(String::new());
                sent.as_object_mut().expect("message").remove("id");
            }
            "wrong_kind" => {
                text.payload.item["type"] = json!("refusal");
                sent["content"][0]["type"] = json!("refusal");
            }
            "refusal_as_output_text" => {
                text.content =
                    CarriedTextContent::Refusal("synthetic canonical refusal".to_owned());
                sent["content"][0]["refusal"] = json!("synthetic canonical refusal");
            }
            "inspection_text" => {
                text.content = CarriedTextContent::InspectionOnlyText("answer".to_owned())
            }
            "inspection_media" => {
                text.content = CarriedTextContent::InspectionOnlyMedia("answer".to_owned())
            }
            "signed" => {
                text.content = CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
                    "answer",
                    GeminiThoughtSignature::new("synthetic signature"),
                ))
            }
            "opaque" => text.content = CarriedTextContent::Opaque(String::new()),
            _ => unreachable!("listed synthetic follower boundary"),
        }
        if matches!(boundary, "inspection_text" | "inspection_media" | "signed") {
            sent.as_object_mut()
                .expect("projected known text")
                .remove("phase");
        }
        assert_eq!(
            send(&source, &target, &input, false, true)
                .0
                .expect("old follower route encodes"),
            expected(vec![user("q"), sent, user("next")], false, true),
            "synthetic boundary {boundary}"
        );
    }
}

#[test]
fn wrong_position_and_edited_opaque_followers_keep_their_original_errors() {
    let target = wire();
    let source = received(
        &target,
        vec![
            reasoning(),
            answer(json!([{"type":"output_text", "text":"answer", "annotations":[]}])),
        ],
    );
    for position in [
        CarriedPosition::Whole,
        CarriedPosition::ReasoningSummary { part_index: 0 },
        CarriedPosition::ReasoningContent { part_index: 0 },
    ] {
        let mut input = history(&source);
        text_mut(&mut input).payload.position = position;
        let error = send(&source, &target, &input, false, true)
            .0
            .expect_err("invalid position still refuses");
        let ProviderError::Request(source) = error else {
            panic!("existing Request refusal");
        };
        assert_eq!(
            source.to_string(),
            "a carried Responses message part requires its containing message position"
        );
    }
    for raw in [json!("synthetic nonobject part"), json!([]), Value::Null] {
        let mut input = history(&source);
        text_mut(&mut input).payload.item = raw;
        let error = send(&source, &target, &input, false, true)
            .0
            .expect_err("invalid known part still refuses");
        let ProviderError::Request(source) = error else {
            panic!("typed Request refusal");
        };
        assert_eq!(
            source.downcast_ref::<crate::message::CarriedTextReplayError>(),
            Some(&crate::message::CarriedTextReplayError::InvalidKnownPart { field: "text" })
        );
    }
    let mut input = history(&source);
    text_mut(&mut input).content = CarriedTextContent::Opaque("synthetic opaque edit".to_owned());
    let error = send(&source, &target, &input, false, true)
        .0
        .expect_err("opaque edit still refuses");
    let ProviderError::Request(source) = error else {
        panic!("typed Request refusal");
    };
    assert_eq!(
        source.downcast_ref::<crate::message::CarriedTextReplayError>(),
        Some(&crate::message::CarriedTextReplayError::EditedOpaqueText)
    );
}

#[test]
fn gateway_reasoning_keeps_the_existing_no_tools_route() {
    let target = OpenAIConfig::with_key(
        &crate::providers::openai::wire::OPENROUTER,
        "synthetic-test-key",
    )
    .responses("gpt-5.4");
    let source = received(&target, vec![reasoning(), call()]);
    let [AssistantContent::CarriedReasoning(reason), _] = source.response.choice.as_slice() else {
        panic!("gateway reasoning and call");
    };
    assert_eq!(source.response.origin.provider, "openrouter");
    assert!(
        reason
            .payload
            .provider
            .as_deref()
            .is_some_and(|issuer| issuer.starts_with("openrouter"))
    );
    no_tools(
        &source,
        &target,
        &history(&source),
        vec![reasoning(), called()],
    );
}

#[test]
fn unknown_origin_history_preserves_the_existing_carried_route() {
    let target = wire();
    let source = received(&target, vec![reasoning(), answer(answer_parts())]);
    let mut input = history(&source);
    let Message::Assistant(turn) = &mut input[1] else {
        panic!("genuine head clone");
    };
    // Synthetic unknown-origin input never acquires fabricated Origin.
    turn.origin = None;
    assert_eq!(
        send(&source, &target, &input, false, true)
            .0
            .expect("cross-origin route"),
        expected(
            vec![
                user("q"),
                reasoning(),
                carried_answer(answer_parts()),
                user("next")
            ],
            false,
            true
        )
    );
}

#[test]
fn actual_chat_api_keeps_the_existing_carried_text_preflight_refusal() {
    let source = received(&wire(), vec![reasoning(), answer(answer_parts())]);
    let target = OpenAIConfig::new("synthetic-test-key").chat("gpt-5.4");
    let error = send(&source, &target, &history(&source), false, true)
        .0
        .expect_err("Chat refuses Responses carried text");
    let ProviderError::Request(source) = error else {
        panic!("typed Chat Request refusal");
    };
    assert_eq!(
        source.downcast_ref::<UnrepresentableOpaqueContent>(),
        Some(&UnrepresentableOpaqueContent::new("openai"))
    );
}

#[test]
fn synthetic_whole_opaque_follower_keeps_its_existing_replay_or_drop_rule() {
    let target = wire();
    let raw = answer(json!([{"type":"output_text", "text":"answer", "annotations":[]}]));
    let source = received(&target, vec![reasoning(), raw.clone()]);
    for replay in [false, true] {
        let mut input = history(&source);
        let Message::Assistant(turn) = &mut input[1] else {
            panic!("genuine head clone");
        };
        // Synthetic category boundary on a received-message clone. No Native,
        // Origin or independent issuer is invented for the opaque value.
        turn.content[1] = AssistantContent::Opaque(crate::message::Opaque {
            item: raw.clone(),
            replay,
        });
        let items = if replay {
            vec![user("q"), reasoning(), raw.clone(), user("next")]
        } else {
            vec![user("q"), user("next")]
        };
        assert_eq!(
            send(&source, &target, &input, false, true)
                .0
                .expect("old opaque follower route"),
            expected(items, false, true)
        );
    }
}
