//! Replay rules the Responses wire holds by construction: each test is an
//! open review item of round 4, in the input shape that reproduced it.

use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use crate::message::{AssistantContent, CallId, Message, ToolCall, ToolName, ToolResultContent};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Mode, Operation, Wire, WireFrame};

use super::ResponsesToolDefinition;
use super::wire::Responses;

fn wire() -> Responses {
    OpenAIConfig::new("key").responses("gpt-5.4")
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn tool(tool: &str) -> ToolDefinition {
    ToolDefinition {
        name: name(tool),
        description: "A tool".to_owned(),
        parameters: json!({"type": "object"}),
    }
}

// G139 outer scaffolding exception: absent envelope metadata is supplied
// on a clone. Original item fields, statuses and output positions stay exact.
fn frames(events: &[Value]) -> Vec<WireFrame> {
    events
        .iter()
        .enumerate()
        .map(|(sequence, event)| {
            let mut event = event.clone();
            if let Some(fields) = event.as_object_mut() {
                fields.entry("sequence_number").or_insert_with(|| json!(sequence));
                if let Some(response) = fields.get_mut("response").and_then(Value::as_object_mut) {
                    response.entry("created_at").or_insert_with(|| json!(0));
                    response.entry("object").or_insert_with(|| json!("response"));
                    response.entry("model").or_insert_with(|| json!("gpt-5.4"));
                }
            }
            WireFrame::Text(event.to_string())
        })
        .collect()
}

fn body(output: Value) -> Value {
    // G139: the outer fixture clock is explicit scaffolding, not received data.
    json!({"id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": "gpt-5.4", "output": output})
}

fn whole(output: Value) -> CompletionResponse {
    crate::test_utils::history::decode(
        &wire(),
        Mode::Unary,
        [WireFrame::Text(body(output).to_string())],
    )
    .expect("the body decodes")
}

fn streamed(events: &[Value]) -> CompletionResponse {
    crate::test_utils::history::decode(&wire(), Mode::Streaming, frames(events))
        .expect("the stream decodes")
}

/// The input `wire` sends for `request`, prepared as the driver prepares it.
fn input_of(wire: &Responses, request: CompletionRequest) -> Vec<Value> {
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// The input the same model gets for `response`'s turn, its calls answered.
fn replayed(response: &CompletionResponse, tools: &[&str]) -> Vec<Value> {
    let Some(turn) = response.message() else {
        panic!("the reply is a turn: {:?}", response.choice);
    };
    let results: Vec<_> = response
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let next = if results.is_empty() {
        Message::user("next")
    } else {
        Message::User { content: results }
    };
    let history = vec![Message::user("q"), turn, next];
    let request =
        CompletionRequest::from(history).tools(tools.iter().map(|name| tool(name)).collect());
    input_of(&wire(), request)
}

/// A reasoning item, then a call streamed as added, deltas and done.
fn streamed_call() -> Vec<Value> {
    vec![
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "plan"}),
        json!({"type": "response.output_item.done", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}],
                        "encrypted_content": "enc"}}),
        json!({"type": "response.output_item.added", "output_index": 1,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"a\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "1}"}),
        json!({"type": "response.output_item.done", "output_index": 1,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
                        "arguments": "{\"a\":1}", "status": "completed"}}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}}),
    ]
}

/// G191: original absent added status and removed done fields remain strict
/// refusals. The separately named intact fixture retains the old reachable pins.
#[test]
fn a_done_call_keeps_what_was_announced_and_streamed() {
    use crate::wire::{Decoder, Wire};
    for missing in ["arguments", "name", "call_id"] {
        let mut events = streamed_call();
        if let Some(item) = events[6]["item"].as_object_mut() {
            item.shift_remove(missing);
        }
        let WireFrame::Text(announced) = frames(&events[3..4]).remove(0) else { panic!("text fixture"); };
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(announced.clone())) else { panic!("the original added status is absent"); };
        assert_eq!(refused.error().to_string(), "missing field `status`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(announced));
        let WireFrame::Text(scaffolded) = frames(&events[6..7]).remove(0) else { panic!("text fixture"); };
        let raw = scaffolded;
        let configured = wire();
        let decoder = configured.decoder_in(Mode::Streaming);
        let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
            panic!("the exact held fixture must remain a typed strict refusal");
        };
        assert_eq!(refused.error().to_string(), format!("missing field `{missing}`"));
        assert_eq!(refused.event_type(), Some("response.output_item.done"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
    }
}

/// G191: original later cuts reach a call without required added status.
/// Cut3 and admitted cutoff/ciphertext behavior have separate named controls.
#[test]
fn a_reply_cut_before_its_terminal_keeps_no_item_and_the_terminal_takes_the_ciphertext() {
    let mut events = streamed_call();
    if let Some(item) = events[2]["item"].as_object_mut() {
        item.shift_remove("encrypted_content");
    }
    // G191 cuts5/7 reach the bad announcement; cut3 is its own behavior.
    for cut in [5, 7] {
        let encoded = frames(&events[..cut]);
        let WireFrame::Text(expected) = &encoded[3] else { panic!("text fixture"); };
        let expected = expected.clone();
        let error = crate::test_utils::history::decode(&wire(), Mode::Streaming, encoded)
            .expect_err("the original added call has no required status");
        let crate::error::ProviderError::CorruptFrame(refused) = error else { panic!("strict shape refusal"); };
        assert_eq!(refused.error().to_string(), "missing field `status`");
        assert_eq!(refused.event_type(), Some("response.output_item.added"));
        assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(expected));
    }
}

/// Round 5 generated-history finding 3: a stream that never sends output 0
/// and is cut before its terminal keeps no item of the call it streamed at
/// output 1, so the turn a runtime rolls back sends the call without an
/// item that would need the reasoning that never arrived.
#[test]
fn a_call_cut_off_after_a_missing_output_index_goes_back_without_its_item() {
    let call = json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
        "arguments": "{}", "status": "completed"});
    let mut added = call.clone();
    added["arguments"] = json!("");
    added["status"] = json!("in_progress");
    let events = [
        json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": added}),
        json!({"type": "response.function_call_arguments.done", "output_index": 1, "item_id": "fc_1", "arguments": "{}"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": call}),
    ];
    let (response, ended) =
        crate::test_utils::history_conformance::partial(&wire(), Mode::Streaming, frames(&events));
    assert!(!ended, "no terminal arrived");
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "{:?}",
        response.choice
    );
    let turn = response.continued(response.choice.clone());
    let results = turn
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let history = vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::User { content: results },
    ];
    let input = input_of(
        &wire(),
        CompletionRequest::from(history).tools(vec![tool("f")]),
    );
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "function_call" && item["call_id"] == "call_1"),
        "{input:?}"
    );
    assert!(
        !input
            .iter()
            .any(|item| item.get("id") == Some(&json!("fc_1"))),
        "the call's item replays without output 0: {input:?}"
    );
}

fn reasoning(id: &str, ciphertext: Option<&str>) -> Value {
    let mut item = json!({"type": "reasoning", "id": id, "summary": [{"type": "summary_text", "text": "plan"}]});
    if let Some(ciphertext) = ciphertext {
        item["encrypted_content"] = json!(ciphertext);
    }
    item
}

fn message(id: &str, text: &str) -> Value {
    json!({"type": "message", "id": id, "role": "assistant", "status": "completed",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn function_call(id: &str, call_id: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call_id, "name": "f",
           "arguments": "{}", "status": "completed"})
}

fn kinds(response: &CompletionResponse) -> Vec<&'static str> {
    response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Text(_) => "text",
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::ToolCall(_) => "call",
            AssistantContent::Image(_) => "image",
            AssistantContent::Opaque(_) => "opaque",
            AssistantContent::CustomToolCall(_) => "custom call",
            AssistantContent::ProviderItem(_) => "provider item",
            AssistantContent::CarriedText(text) => match &text.content {
                crate::message::CarriedTextContent::Text(_)
                | crate::message::CarriedTextContent::Refusal(_)
                | crate::message::CarriedTextContent::GeminiSignedAnswer(_)
                | crate::message::CarriedTextContent::InspectionOnlyText(_)
                | crate::message::CarriedTextContent::InspectionOnlyMedia(_) => "text",
                crate::message::CarriedTextContent::Opaque(_) => "opaque carried text",
            },
            AssistantContent::CarriedReasoning(reasoning) => {
                if reasoning.has_opaque_parts() {
                    "opaque carried reasoning"
                } else {
                    "reasoning"
                }
            }
        })
        .collect()
}

/// The input `wire` sends to continue `response`'s turn with its calls
/// answered, under `params`.
fn continued_on(wire: &Responses, response: &CompletionResponse, params: Value) -> Vec<Value> {
    let Some(turn) = response.message() else {
        panic!("the reply is a turn: {:?}", response.choice);
    };
    let results = response
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let history = vec![Message::user("q"), turn, Message::User { content: results }];
    let mut request = CompletionRequest::from(history).tools(vec![tool("f")]);
    request.additional_params = Some(params);
    input_of(wire, request)
}

/// A `store: false` request resolves no stored item: reasoning without its
/// ciphertext stays out, and the call after it goes without its item id.
#[test]
fn a_stateless_request_sends_no_reasoning_it_cannot_resolve() {
    let response = whole(json!([
        reasoning("rs_1", None),
        function_call("fc_1", "call_1")
    ]));
    let input = continued_on(&wire(), &response, json!({"store": false}));
    assert!(
        !input.iter().any(|item| item["type"] == "reasoning"),
        "{input:?}"
    );
    let call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("the call goes back");
    assert!(call.get("id").is_none(), "{call}");
    assert_eq!(call["call_id"], "call_1");

    let stored = continued_on(&wire(), &response, json!({}));
    assert_eq!(stored[1]["id"], "rs_1", "{stored:?}");
    assert_eq!(stored[2]["id"], "fc_1", "{stored:?}");
}

/// A request in a stored `conversation` continues the provider's state,
/// which holds the calls its first results answer.
#[test]
fn a_conversation_continuation_keeps_the_results_it_answers() {
    let mut request = CompletionRequest::from(vec![Message::tool_result(
        CallId::from_wire("call_1"),
        name("f"),
        "done",
    )])
    .tools(vec![tool("f")]);
    request.additional_params = Some(json!({"conversation": "conv_1"}));
    let input = input_of(&wire(), request);
    assert_eq!(input[0]["type"], "function_call_output", "{input:?}");
    assert_eq!(input[0]["call_id"], "call_1");
}

/// A result takes the kind of the call it answers, even when the request
/// now declares a custom tool of that call's name.
#[test]
fn a_result_takes_the_kind_of_the_call_it_answers() {
    let mut custom = ResponsesToolDefinition::hosted("custom");
    custom.name = "f".to_owned();
    let response = whole(json!([function_call("fc_1", "call_1")]));
    let input = continued_on(&wire().with_tool(custom), &response, json!({}));
    let kinds: Vec<&str> = input
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(kinds, ["message", "function_call", "function_call_output"]);
}

/// A terminal whose output indices are shifted against the stream's
/// restates the streamed message, so its text is said once.
#[test]
fn a_terminal_shifted_against_the_stream_restates_its_items() {
    let events = [
        json!({"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "hi"}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
               "model": "gpt-5.4", "output": [message("msg_1", "hi")]}}),
    ];
    let response = streamed(&events);
    assert_eq!(response.text(), "hi", "{:?}", response.choice);
    assert_eq!(
        response.choice[0].native_item(),
        Some(&message("msg_1", "hi"))
    );
}

/// A hosted item only the terminal states takes its place between the
/// streamed items, as in the whole reply.
#[test]
fn a_terminal_only_hosted_item_keeps_its_place() {
    let search = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                        "action": {"type": "search", "query": "q"}});
    let output = json!([
        reasoning("rs_1", Some("enc")),
        search,
        message("msg_1", "hi")
    ]);
    let events = [
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning("rs_1", Some("enc"))}),
        json!({"type": "response.output_item.done", "output_index": 2, "item": message("msg_1", "hi")}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
               "model": "gpt-5.4", "output": output.clone()}}),
    ];
    let whole = whole(output);
    let streamed = streamed(&events);
    assert_eq!(kinds(&streamed), kinds(&whole));
    assert_eq!(kinds(&whole), ["reasoning", "provider item", "text"]);
    for response in [&whole, &streamed] {
        let Some(AssistantContent::ProviderItem(item)) = response.choice.get(1) else {
            panic!("the hosted item has its distinct provider-item category");
        };
        assert_eq!(item.payload.item, search);
        assert_eq!(item.payload.provider.as_deref(), Some("openai"));
    }
}

/// G191: the original message announcement lacks required status. Its strict
/// refusal and the admitted raw/summary whole-turn parity are distinct subjects.
#[test]
fn raw_reasoning_deltas_under_a_summary_fold_as_the_whole_reply() {
    use crate::wire::{Decoder, Wire};
    let rs = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "sum"}],
                    "content": [{"type": "reasoning_text", "text": "raw"}], "encrypted_content": "enc"});
    let output = json!([rs, message("msg_1", "hi")]);
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1", "content_index": 0, "delta": "raw"}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "item_id": "rs_1", "summary_index": 0, "delta": "sum"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": rs}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_1", "content_index": 0, "delta": "hi"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": message("msg_1", "hi")}),
        json!({"type": "response.completed", "response": body(output.clone())}),
    ];
    let WireFrame::Text(scaffolded) = frames(&events[4..5]).remove(0) else { panic!("text fixture"); };
    let raw = scaffolded;
    let configured = wire();
    let decoder = configured.decoder_in(Mode::Streaming);
    let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
        panic!("the exact held fixture must remain a typed strict refusal");
    };
    assert_eq!(refused.error().to_string(), "missing field `status`");
    assert_eq!(refused.event_type(), Some("response.output_item.added"));
    assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
}

/// G191: the exact nameless call is a strict refusal, replacing the old dropped
/// call/no-replay/later-text success. G47 separately pins genuine unnamed writers.
#[test]
fn reasoning_before_a_dropped_call_does_not_go_back() {
    use crate::wire::{Decoder, Wire};
    let nameless = json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
                          "arguments": "{}", "status": "completed"});
    let raw = body(json!([
        reasoning("rs_1", Some("enc")), nameless, message("msg_1", "hi")
    ])).to_string();
    let configured = wire();
    let decoder = configured.decoder_in(Mode::Unary);
    let crate::wire::WireEvent::Corrupt(refused) = decoder.classify(WireFrame::Text(raw.clone())) else {
        panic!("a deliberately nameless call must remain a typed strict refusal");
    };
    assert_eq!(refused.error().to_string(), "missing field `name`");
    assert_eq!(refused.event_type(), None);
    assert_eq!(refused.evidence(), &crate::error::FrameEvidence::Text(raw));
}


#[test]
fn admitted_raw_reasoning_deltas_under_a_summary_fold_as_the_whole_reply() {
    let rs = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "sum"}],
                    "content": [{"type": "reasoning_text", "text": "raw"}], "encrypted_content": "enc"});
    let output = json!([rs, message("msg_1", "hi")]);
    let mut events = [
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1", "content_index": 0, "delta": "raw"}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "item_id": "rs_1", "summary_index": 0, "delta": "sum"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": rs}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_1", "content_index": 0, "delta": "hi"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": message("msg_1", "hi")}),
        json!({"type": "response.completed", "response": body(output.clone())}),
    ];
    // G191 NEW added-message status; original raw/summary/text parity stays exact.
    events[4]["item"]["status"] = json!("in_progress");
    let streamed = streamed(&events);
    let whole = whole(output.clone());
    assert_eq!(streamed.choice, whole.choice);
    assert_eq!(streamed.raw["output"], whole.raw["output"]);
    assert_eq!(streamed.raw["output"], output);
}


// G191 NEW variant supplies only the absent added-call status. Original helper
// and deliberately missing done fields remain unchanged in negative subjects.
fn admitted_streamed_call() -> Vec<Value> {
    let mut events = streamed_call();
    events[3]["item"]["status"] = json!("in_progress");
    events
}


#[test]
fn the_original_three_frame_reasoning_prefix_keeps_no_native_item() {
    let mut events = streamed_call();
    if let Some(item) = events[2]["item"].as_object_mut() {
        item.shift_remove("encrypted_content");
    }
    let (cut_off, ended) = crate::test_utils::history_conformance::partial(
        &wire(), Mode::Streaming, frames(&events[..3]));
    assert!(!ended, "no original terminal arrived");
    assert!(cut_off.choice.iter().all(|block| block.native_item().is_none()),
        "cut3: {:?}", cut_off.choice);
}


#[test]
fn admitted_reply_cuts_keep_no_native_item_and_terminal_keeps_ciphertext() {
    let mut events = admitted_streamed_call();
    if let Some(item) = events[2]["item"].as_object_mut() {
        item.shift_remove("encrypted_content");
    }
    let partial = |cut: usize| {
        crate::test_utils::history_conformance::partial(
            &wire(),
            Mode::Streaming,
            frames(&events[..cut]),
        )
        .0
    };
    assert!(!partial(7).choice.is_empty(), "both items ended");
    for cut in [3, 5, 7] {
        let cut_off = partial(cut);
        assert!(
            cut_off
                .choice
                .iter()
                .all(|block| block.native_item().is_none()),
            "cut {cut}: {:?}",
            cut_off.choice
        );
    }
    let last = events.len() - 1;
    events[last]["response"]["output"] = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}], "encrypted_content": "enc"},
    ]);
    let response = streamed(&events);
    // G196: terminal reasoning is actual carried data; completed call stays Native.
    let [AssistantContent::CarriedReasoning(reasoning), AssistantContent::ToolCall(call)] =
        response.choice.as_slice() else { panic!("ordered reasoning and call: {:?}", response.choice); };
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(reasoning.display_text(), "plan");
    assert_eq!(reasoning.content, vec![crate::message::CarriedReasoningPart::Summary("plan".to_owned()),
        crate::message::CarriedReasoningPart::Encrypted("enc".to_owned())]);
    assert_eq!(reasoning.payload.item, events[last]["response"]["output"][0]);
    assert_eq!(reasoning.payload.item["encrypted_content"], json!("enc"));
    assert_eq!(response.choice[1].native_item(), Some(&events[6]["item"]));
    assert_eq!(call.function.arguments_value(), json!({"a":1}));
    assert_eq!(call.id.wire(), "call_1");
    assert_eq!(response.raw["output"], events[last]["response"]["output"]);
}


#[test]
fn an_admitted_intact_done_call_preserves_announced_arguments_identity_and_pairing() {
    // G191 intact control supplies no omitted name/call-id/arguments fallback.
    let events = admitted_streamed_call();
    let missing = "intact fixture";
    let response = streamed(&events);
    let calls: Vec<&ToolCall> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1, "without `{missing}`: {:?}", response.choice);
    assert_eq!(calls[0].id.wire(), "call_1", "without `{missing}`");
    assert_eq!(calls[0].function.name.as_str(), "f", "without `{missing}`");
    assert_eq!(
        calls[0].function.arguments_value(),
        json!({"a": 1}),
        "without `{missing}`"
    );
    let input = replayed(&response, &["f"]);
    let call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("the call goes back");
    let output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("its result");
    assert_eq!(
        call["id"], "fc_1",
        "without `{missing}`: the item goes back"
    );
    assert_eq!(call["call_id"], output["call_id"], "without `{missing}`");
}
