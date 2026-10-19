//! Full response.done custody through the real scripted WebSocket collector.
//! Separately named decoder controls observe writer items that the collector
//! consumes internally; they make no collector-transcript access claim.

use super::super::test_connection::Script;
use super::super::{
    ResponsesWebSocketEvent, ResponsesWebSocketSession, parse_server_event,
    terminal_response_result,
};
use crate::completion::{CompletionRequest, CompletionResponse, FinishReason};
use crate::driver::Decoded;
use crate::error::ProviderError;
use crate::message::{AssistantContent, StopReason, turn_failure};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::streaming::{IncompleteTerminal, document};
use crate::streaming::{PartKind, SourceOrder, StreamEvent};
use crate::wire::{Call, Mode, Operation, Wire, WireFrame};
use serde_json::{Value, json};

const PREFIX: &str = "{\"x\":1}";
const RESTATED: &str = "{\"x\":2}";
const UNFINISHED: &str = "the provider ended the reply with a tool call it never finished";
const NO_DISPATCH: &str = "the turn failed, so none of its tool calls ran: the provider ended the reply with a tool call it never finished";

fn item(status: &str, arguments: &str) -> Value {
    json!({"type":"function_call","id":"fc_1","call_id":"call_1",
        "name":"old","arguments":arguments,"status":status})
}

fn added(index: u64) -> Value {
    json!({"type":"response.output_item.added","sequence_number":0,"output_index":index,
        "item":item("in_progress", "")})
}

fn delta(index: u64) -> Value {
    json!({"type":"response.function_call_arguments.delta","sequence_number":1,
        "output_index":index,"delta":PREFIX})
}

fn full_done(output: Value) -> Value {
    // Required full-body fields keep this on the collector's genuine Whole
    // unwrap path. The ROOT is completed; an output call may still be pending.
    json!({"type":"response.done","response":{"id":"resp_g216","object":"response",
        "created_at":0,"status":"completed","model":"gpt-5.4","output":output,"tools":[]}})
}

fn received(index: u64, terminal_item: Value) -> Vec<Value> {
    vec![
        added(index),
        delta(index),
        full_done(json!([terminal_item])),
    ]
}

async fn collect(frames: &[Value]) -> Result<CompletionResponse, ProviderError> {
    let script = Script::new().turn(frames.iter().map(Value::to_string));
    script.end_stream();
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let mut session = ResponsesWebSocketSession::from_connection(wire, script.connection(), None);
    let response = session.completion(CompletionRequest::new("fixture")).await;
    assert_eq!(
        script.sent_json().len(),
        1,
        "one actual response.create send releases this turn"
    );
    assert_eq!(script.sent_json()[0]["type"], "response.create");
    assert_eq!(
        script.unread(),
        0,
        "the collector consumed the exact full Done turn"
    );
    response
}

/// Same frames, parsed by the collector's existing server-event parser. The
/// existing full Done unwrap and terminal response check feed its actual body
/// to the WebSocket reader, exactly as wait_for_terminal_response does.
fn transcript(frames: &[Value]) -> Decoded<Completion> {
    let frames = frames
        .iter()
        .map(|frame| {
            let payload = frame.to_string();
            match parse_server_event(&payload).expect("all synthetic envelopes use the real reader")
            {
                Some(ResponsesWebSocketEvent::Done(done)) => {
                    assert!(
                        ["id", "object", "created_at", "status", "model"]
                            .iter()
                            .all(|key| done.response.get(*key).is_some())
                    );
                    WireFrame::Text(
                        terminal_response_result(done.response)
                            .expect("the received full root is genuinely completed")
                            .to_string(),
                    )
                }
                Some(_) => WireFrame::Text(payload),
                None => panic!("every received fixture frame is modelled"),
            }
        })
        .collect::<Vec<_>>();
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let request = Completion::prepare(CompletionRequest::new("fixture"), &wire.describe())
        .expect("same actual request preparation as the collector");
    let fold = Completion::fold(&request, &mut Call::new(&wire.describe(), Mode::Streaming));
    crate::driver::decode_with(fold, "openai", |reply| {
        let mut decoder = wire
            .decoder_with_incomplete(IncompleteTerminal::Accept)
            .with_websocket_reader();
        reply.feed(
            &mut decoder,
            Some(document::Response::with_websocket_reader()),
            frames,
        )
    })
}

fn assert_failed_prefix(response: &CompletionResponse, terminal: &Value) {
    assert_eq!(response.error.as_deref(), Some(UNFINISHED));
    assert_eq!(response.stop(), StopReason::Error(UNFINISHED.to_owned()));
    assert_eq!(
        response.head().stop,
        Some(StopReason::Error(UNFINISHED.to_owned()))
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(
        response.raw, terminal["response"],
        "the entire received terminal object stays raw evidence"
    );
    let calls = response.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.provider().unwrap().as_str(), "call_1");
    assert_eq!(calls[0].function.name.as_str(), "old");
    assert_eq!(calls[0].function.namespace, None);
    assert_eq!(
        calls[0].function.arguments_value(),
        json!({"x":1}),
        "pending restatement cannot replace the delivered prefix"
    );
    assert_eq!(calls[0].function.invalid_arguments, None);
    assert!(
        calls[0].native.is_none(),
        "no completed provider item was received for this call"
    );
    assert_eq!(
        turn_failure(
            &response.choice,
            response.head().stop.as_ref(),
            response.finish_reason().as_ref()
        )
        .as_deref(),
        Some(NO_DISPATCH)
    );
}

fn assert_completed(response: &CompletionResponse, terminal: &Value) {
    assert_eq!(response.error, None);
    assert_eq!(response.stop(), StopReason::ToolUse);
    assert_eq!(response.raw, terminal["response"]);
    let calls = response.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.provider().unwrap().as_str(), "call_1");
    assert_eq!(calls[0].function.name.as_str(), "old");
    assert_eq!(calls[0].function.arguments_value(), json!({"x":2}));
    assert_eq!(calls[0].function.invalid_arguments, None);
    assert_eq!(
        calls[0]
            .native
            .as_ref()
            .expect("genuine completed terminal item")
            .item,
        terminal["response"]["output"][0]
    );
    assert_eq!(
        turn_failure(
            &response.choice,
            response.head().stop.as_ref(),
            response.finish_reason().as_ref()
        ),
        None
    );
}

#[tokio::test]
async fn full_done_pending_call_keeps_the_delivered_prefix_and_refuses_dispatch() {
    for index in [0, 7] {
        for status in ["in_progress", "incomplete"] {
            let frames = received(index, item(status, RESTATED));
            let response = collect(&frames)
                .await
                .expect("the real collector reaches the full Done terminal");
            assert_failed_prefix(&response, frames.last().unwrap());
        }
    }
}

#[tokio::test]
async fn full_done_completed_call_uses_its_announced_identity_even_at_a_shifted_position() {
    for index in [0, 7] {
        for namespace in [None, Some(""), Some("math")] {
            let mut frames = received(index, item("completed", RESTATED));
            if let Some(namespace) = namespace {
                frames[0]["item"]["namespace"] = json!(namespace);
                frames[2]["response"]["output"][0]["namespace"] = json!(namespace);
            }
            let response = collect(&frames)
                .await
                .expect("the real collector consumes genuine matched completion");
            assert_completed(&response, frames.last().unwrap());
            assert_eq!(
                response
                    .tool_calls()
                    .next()
                    .unwrap()
                    .function
                    .namespace
                    .as_deref(),
                namespace
            );
        }
    }
}

fn contradictions() -> Vec<(&'static str, Value)> {
    let mut name = item("completed", RESTATED);
    name["name"] = json!("new");
    let mut call_id = item("completed", RESTATED);
    call_id["call_id"] = json!("call_new");
    let mut namespace = item("completed", RESTATED);
    namespace["namespace"] = json!("math");
    let mut empty_namespace = item("completed", RESTATED);
    empty_namespace["namespace"] = json!("");
    vec![
        ("name", name),
        ("call ID", call_id),
        ("namespace", namespace),
        ("namespace", empty_namespace),
        (
            "kind",
            json!({"type":"custom_tool_call","id":"fc_1","call_id":"call_1",
            "name":"old","input":"raw","status":"completed"}),
        ),
        (
            "kind",
            json!({"type":"message","id":"fc_1","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":"other","annotations":[]}]}),
        ),
        ("kind", json!({"type":"reasoning","id":"fc_1","summary":[]})),
        (
            "kind",
            json!({"type":"future_provider_item","id":"fc_1","payload":[7]}),
        ),
    ]
}

#[tokio::test]
async fn full_done_known_pending_function_refuses_every_changed_identity_or_kind() {
    for (field, item) in contradictions() {
        for status in ["completed", "in_progress", "incomplete"] {
            let mut item = item.clone();
            if item["type"] == "function_call" {
                item["status"] = json!(status);
            }
            let error = collect(&received(7, item))
                .await
                .expect_err("a known Pending ID cannot change its announcement");
            assert!(matches!(&error, ProviderError::Response(message)
                if message == &format!("conflicting Responses terminal function call fc_1: {field} contradicts its announcement")));
            assert!(!error.is_retryable());
        }
    }
}

fn assert_prefix_transcript(decoded: &Decoded<Completion>, index: u64) -> crate::streaming::Part {
    let events = decoded.events();
    let starts = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                source_order,
                name,
                ..
            } => Some((
                *part,
                *source_order,
                name.as_ref().map(|name| name.as_str()),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!starts.is_empty());
    assert!(starts.iter().all(|(part, _, _)| *part == starts[0].0));
    assert_eq!(starts[0].1, Some(SourceOrder::new(index, 0)));
    assert_eq!(starts.last().unwrap().2, Some("old"));
    assert!(
        starts
            .iter()
            .all(|(_, _, name)| name.is_none() || *name == Some("old"))
    );
    let arguments = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Arguments { part, json } => Some((*part, json.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        arguments,
        [(starts[0].0, PREFIX)],
        "terminal restatement is not an invented Arguments delta"
    );
    starts[0].0
}

#[test]
fn private_full_done_decoder_transcript_keeps_pending_part_and_source_unfinished() {
    for index in [0, 7] {
        for status in ["in_progress", "incomplete"] {
            let frames = received(index, item(status, RESTATED));
            let decoded = transcript(&frames);
            let part = assert_prefix_transcript(&decoded, index);
            let ends = decoded
                .events()
                .into_iter()
                .filter_map(|event| match event {
                    StreamEvent::End {
                        part,
                        content: AssistantContent::ToolCall(call),
                        ..
                    } => Some((*part, call)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(ends.len(), 1);
            assert_eq!(ends[0].0, part);
            assert_eq!(ends[0].1.function.arguments_value(), json!({"x":1}));
            assert!(ends[0].1.native.is_none());
            assert_failed_prefix(
                &decoded
                    .outcome
                    .expect("same full Done guard through the actual fold"),
                frames.last().unwrap(),
            );
        }
    }
}

#[test]
fn private_full_done_decoder_transcript_finishes_the_same_announced_part_after_a_shift() {
    for index in [0, 7] {
        let frames = received(index, item("completed", RESTATED));
        let decoded = transcript(&frames);
        let part = assert_prefix_transcript(&decoded, index);
        let ends = decoded
            .events()
            .into_iter()
            .filter_map(|event| match event {
                StreamEvent::End {
                    part,
                    content: AssistantContent::ToolCall(call),
                    ..
                } => Some((*part, call)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].0, part);
        assert_eq!(ends[0].1.function.arguments_value(), json!({"x":2}));
        assert_eq!(
            ends[0].1.native.as_ref().unwrap().item,
            frames[2]["response"]["output"][0]
        );
        assert_completed(
            &decoded.outcome.expect("same genuine completed terminal"),
            frames.last().unwrap(),
        );
    }
}

#[test]
fn private_full_done_decoder_transcript_refuses_before_replacing_prefix_or_emitting_native() {
    for (field, item) in contradictions() {
        for status in ["completed", "in_progress", "incomplete"] {
            let mut item = item.clone();
            if item["type"] == "function_call" {
                item["status"] = json!(status);
            }
            let decoded = transcript(&received(7, item));
            assert_prefix_transcript(&decoded, 7);
            assert!(
                !decoded.events().iter().any(|event| matches!(
                    event,
                    StreamEvent::End {
                        content: AssistantContent::ToolCall(_),
                        ..
                    }
                )),
                "no completed replacement call or Native is published before the typed refusal"
            );
            assert!(
                matches!(&decoded.outcome, Err(ProviderError::Response(message))
                if message == &format!("conflicting Responses terminal function call fc_1: {field} contradicts its announcement"))
            );
        }
    }
}

#[tokio::test]
async fn full_done_unannounced_function_custom_and_other_slots_keep_their_whole_route() {
    let mut function = item("completed", RESTATED);
    function["id"] = json!("fc_new");
    function["call_id"] = json!("call_new");
    let custom = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_custom",
        "name":"apply_patch","input":"raw","status":"completed"});
    let part = json!({"type":"output_text","text":"other","annotations":[]});
    let message = json!({"type":"message","id":"msg_other","role":"assistant",
        "status":"completed","content":[part]});
    let terminal = full_done(json!([function, custom, message]));
    let response = collect(std::slice::from_ref(&terminal))
        .await
        .expect("unannounced Whole slots retain their current operation");
    assert_eq!(response.error, None);
    assert_eq!(response.raw, terminal["response"]);
    assert_eq!(response.choice.len(), 3);
    let call = response.tool_calls().next().unwrap();
    assert_eq!(call.id.provider().unwrap().as_str(), "call_new");
    assert_eq!(call.function.arguments_value(), json!({"x":2}));
    assert_eq!(call.native.as_ref().unwrap().item, function);
    let AssistantContent::CustomToolCall(call) = &response.choice[1] else {
        panic!("the real custom slot remains custom");
    };
    assert_eq!(call.input, "raw");
    assert_eq!(call.native.as_ref().unwrap().item, custom);
    let AssistantContent::CarriedText(text) = &response.choice[2] else {
        panic!("the annotations-bearing text keeps its real carried representation");
    };
    assert_eq!(text.text(), "other");
    assert_eq!(text.payload.item, part);
    assert_eq!(response.text(), "other");
}

#[tokio::test]
async fn full_done_does_not_reopen_or_rewrite_an_explicitly_done_function() {
    let explicit = item("completed", "{\"x\":3}");
    let mut later = item("completed", RESTATED);
    later["name"] = json!("new");
    let terminal = full_done(json!([later]));
    let frames = vec![
        added(7),
        delta(7),
        json!({"type":"response.output_item.done",
        "sequence_number":2,"output_index":7,"item":explicit}),
        terminal.clone(),
    ];
    let response = collect(&frames)
        .await
        .expect("explicit-done is outside the new pending guard");
    assert_eq!(response.error, None);
    assert_eq!(response.raw, terminal["response"]);
    let calls = response.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name.as_str(), "old");
    assert_eq!(calls[0].function.arguments_value(), json!({"x":3}));
    assert_eq!(calls[0].native.as_ref().unwrap().item, explicit);
}
