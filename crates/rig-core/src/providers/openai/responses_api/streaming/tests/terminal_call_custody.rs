//! G194/G195 synthetic admission controls at the genuine Responses reader/fold.
//! Required root/envelope fields and real in_progress/completed stage statuses
//! are explicit additions; these are not byte-identical historical fixtures.

use super::super::{CallEnding, IncompleteTerminal, ResponsesDecoder, ResponsesStreamOptions};
use super::helpers::*;
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::message::{AssistantContent, StopReason};
use crate::operation::Turn;
use crate::streaming::{Item, PartKind, SourceOrder, StreamEvent};
use crate::wire::{Decoder, Mode, WireEvent, WireFrame};
use serde_json::{Value, json};

const UNFINISHED: &str = "the provider ended the reply with a tool call it never finished";

fn function(status: &str, arguments: &str) -> Value {
    json!({"type":"function_call","id":"fc_1","call_id":"call_1",
        "name":"lookup","arguments":arguments,"status":status})
}

fn added(index: u64) -> Value {
    json!({"type":"response.output_item.added","sequence_number":0,
        "output_index":index,"item":function("in_progress", "")})
}

fn delta(index: u64, raw: &str) -> Value {
    json!({"type":"response.function_call_arguments.delta","sequence_number":1,
        "output_index":index,"delta":raw})
}

fn done(index: u64, item: &Value) -> Value {
    json!({"type":"response.output_item.done","sequence_number":2,
        "output_index":index,"item":item})
}

fn decode(
    frames: Vec<Value>,
    incomplete: IncompleteTerminal,
) -> crate::completion::CompletionResponse {
    let wire = crate::providers::openai::wire::OpenAIConfig::new("test-key")
        .responses("gpt-5.4")
        .with_streamed_incomplete(incomplete);
    crate::test_utils::history::decode(
        &wire,
        Mode::Streaming,
        frames
            .into_iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
    )
    .expect("synthetic admitted fixture reaches its genuine terminal")
}

#[test]
fn a_terminal_completed_call_finishes_only_its_announced_id_at_any_output_position() {
    let item = function("completed", "{\"q\":\"rig\"}");
    for source_index in [0, 7] {
        let frames = [
            added(source_index),
            delta(source_index, "{\"q\":"),
            completed_with_output(json!([item])),
        ];
        let driven = drive(&frames);
        assert!(driven.errors.is_empty(), "{:?}", driven.errors);
        assert_eq!(driven.response.error, None);
        assert_eq!(
            driven.response.finish_reason(),
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(driven.response.head().stop, Some(StopReason::ToolUse));
        let [AssistantContent::ToolCall(call)] = driven.response.choice.as_slice() else {
            panic!("one actual completed call: {:?}", driven.response.choice);
        };
        assert_eq!(call.id.provider().unwrap().as_str(), "call_1");
        assert_eq!(call.function.name.as_str(), "lookup");
        assert_eq!(call.function.namespace, None);
        assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
        assert_eq!(call.function.invalid_arguments, None);
        assert_eq!(
            call.native.as_ref().expect("genuine terminal item").item,
            item
        );
        let starts = driven
            .items
            .iter()
            .filter_map(|event| match event {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: PartKind::ToolCall,
                    source_order,
                    ..
                }) => Some((*part, *source_order)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!starts.is_empty());
        assert!(starts.iter().all(|(part, _)| *part == starts[0].0));
        assert_eq!(starts[0].1, Some(SourceOrder::new(source_index, 0)));
        let arguments = driven
            .items
            .iter()
            .filter_map(|event| match event {
                Item::Event(StreamEvent::Arguments { part, json }) => Some((*part, json.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            arguments,
            [(starts[0].0, "{\"q\":")],
            "terminal restatement is not an invented delta"
        );
        let ends = driven
            .items
            .iter()
            .filter_map(|event| match event {
                Item::Event(StreamEvent::End {
                    part,
                    content: AssistantContent::ToolCall(_),
                    ..
                }) => Some(*part),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ends, [starts[0].0], "same writer Part closes exactly once");
    }
}

fn contradictions() -> Vec<(&'static str, Value)> {
    let mut name = function("completed", "{\"evil\":2}");
    name["name"] = json!("other");
    let mut call_id = function("completed", "{\"evil\":2}");
    call_id["call_id"] = json!("call_other");
    let mut namespace = function("completed", "{\"evil\":2}");
    namespace["namespace"] = json!("math");
    let mut empty_namespace = function("completed", "{\"evil\":2}");
    empty_namespace["namespace"] = json!("");
    vec![
        ("name", name),
        ("call ID", call_id),
        ("namespace", namespace),
        ("namespace", empty_namespace),
        (
            "kind",
            json!({"type":"custom_tool_call","id":"fc_1","call_id":"call_1",
            "name":"lookup","input":"raw","status":"completed"}),
        ),
        (
            "kind",
            message_item("fc_1", json!([{ "type":"output_text","text":"unrelated" }])),
        ),
        ("kind", json!({"type":"reasoning","id":"fc_1","summary":[]})),
        (
            "kind",
            json!({"type":"future_provider_item","id":"fc_1","payload":[7]}),
        ),
    ]
}

#[test]
fn a_known_pending_function_id_refuses_every_terminal_identity_contradiction_without_mutation() {
    for (field, item) in contradictions() {
        let result = crate::driver::decode_with(Turn::relayed("openai"), "openai", |reply| {
            let mut decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
            for frame in [added(7), delta(7, "{\"q\":\"rig\"}")] {
                let WireEvent::Known(event) = decoder.classify(WireFrame::Text(frame.to_string()))
                else {
                    panic!("required-field prefix accepted by the real reader");
                };
                let _ = decoder.decode(event, reply.out())?;
            }
            let terminal = completed_with_output(json!([item]));
            assert!(
                matches!(
                    decoder.classify(WireFrame::Text(terminal.to_string())),
                    WireEvent::Known(_)
                ),
                "the actual strict reader admits this item kind before the custody check"
            );
            let calls = |decoder: &ResponsesDecoder| {
                decoder
                    .calls
                    .iter()
                    .map(|(slot, call)| {
                        (
                            *slot,
                            call.at,
                            call.id.clone(),
                            call.item_id.clone(),
                            call.name.clone(),
                            call.namespace.clone(),
                            call.arguments.clone(),
                            call.sent.clone(),
                            call.overflowed,
                            matches!(call.ending, CallEnding::Pending),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let before_calls = calls(&decoder);
            let before_items = decoder.call_items.clone();
            let before_finished = decoder.finished_calls.clone();
            let before_parents = decoder.parents.clone();
            let before_active = decoder.active_message_part;
            let before_writer = {
                let out = reply.out();
                let shared = out.lock();
                format!("{:?}", shared.items)
            };
            let error = decoder
                .terminal_announced_function_call(&item, &mut reply.out())
                .expect_err("an exact live Pending ID cannot change its identity");
            assert!(matches!(&error, ProviderError::Response(message)
                if message == &format!("conflicting Responses terminal function call fc_1: {field} contradicts its announcement")));
            assert!(!error.is_retryable());
            assert_eq!(calls(&decoder), before_calls);
            assert_eq!(decoder.call_items, before_items);
            assert_eq!(decoder.finished_calls, before_finished);
            assert_eq!(decoder.parents, before_parents);
            assert_eq!(decoder.active_message_part, before_active);
            {
                let out = reply.out();
                let shared = out.lock();
                assert_eq!(format!("{:?}", shared.items), before_writer);
            }
            Err(error)
        });
        assert!(matches!(result.outcome, Err(ProviderError::Response(_))));
        let driven = drive(&[
            added(7),
            delta(7, "{\"q\":\"rig\"}"),
            completed_with_output(json!([item])),
        ]);
        assert!(
            matches!(driven.errors.as_slice(), [ProviderError::Response(message)]
            if message == &format!("conflicting Responses terminal function call fc_1: {field} contradicts its announcement")),
            "full decoder route returns the same typed refusal: {:?}",
            driven.errors
        );
        assert!(!driven.items.iter().any(|event| matches!(
            event,
            Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            })
        )));
        assert!(
            driven.response.response_id().is_none(),
            "no successful provider terminal was sealed"
        );
    }
}

#[test]
fn matching_noncompleted_calls_stay_pending_but_contradictory_identities_are_refused() {
    for status in ["in_progress", "incomplete"] {
        let item = function(status, "{\"replacement\":9}");
        let response = decode(
            vec![
                added(0),
                delta(0, "{\"q\":\"rig\"}"),
                completed_with_output(json!([item])),
            ],
            IncompleteTerminal::Refuse,
        );
        assert_eq!(response.error.as_deref(), Some(UNFINISHED));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
        assert!(
            call.native.is_none(),
            "matching noncompleted terminal input was not adopted"
        );
        for (field, mut item) in contradictions()
            .into_iter()
            .filter(|(field, _)| *field != "kind")
        {
            item["status"] = json!(status);
            let driven = drive(&[
                added(0),
                delta(0, "{\"q\":\"rig\"}"),
                completed_with_output(json!([item])),
            ]);
            assert!(
                matches!(driven.errors.as_slice(), [ProviderError::Response(message)]
                if message == &format!("conflicting Responses terminal function call fc_1: {field} contradicts its announcement"))
            );
        }
    }
}

#[test]
fn a_never_done_valid_json_call_keeps_its_projection_but_fails_the_genuine_terminal() {
    for (raw, value) in [
        ("{\"city\":\"Paris\"}", json!({"city":"Paris"})),
        ("481", json!(481)),
        ("[481,null]", json!([481, null])),
        ("\"quoted\"", json!("quoted")),
        (
            "\"{\\\"city\\\":\\\"Paris\\\"}\"",
            json!("{\"city\":\"Paris\"}"),
        ),
        ("null", Value::Null),
        (" \t null \n", Value::Null),
        ("   ", json!({})),
    ] {
        let response = decode(
            vec![added(0), delta(0, raw), completed_with_output(json!([]))],
            IncompleteTerminal::Refuse,
        );
        assert_eq!(response.error.as_deref(), Some(UNFINISHED), "{raw:?}");
        assert_eq!(
            response.head().stop,
            Some(StopReason::Error(UNFINISHED.to_owned()))
        );
        let call = response
            .tool_calls()
            .next()
            .expect("actual incomplete canonical call");
        assert_eq!(call.id.provider().unwrap().as_str(), "call_1");
        assert_eq!(call.function.name.as_str(), "lookup");
        assert!(call.native.is_none(), "an unfinished call has no Native");
        if value.is_object() {
            assert_eq!(call.function.arguments_value(), value);
            assert_eq!(call.function.invalid_arguments, None);
        } else {
            assert_eq!(call.function.arguments_value(), json!({}));
            assert_eq!(
                call.function.invalid_arguments.as_deref(),
                Some(value.to_string().as_str()),
                "the existing non-object projection is retained before leaving the draft open"
            );
        }
    }
}

#[test]
fn a_never_done_parseable_call_is_refused_even_at_an_accepted_length_terminal() {
    let mut terminal = sample_response("incomplete");
    terminal["incomplete_details"] = json!({"reason":"max_output_tokens"});
    let response = decode(
        vec![
            added(0),
            delta(0, "{\"q\":\"rig\"}"),
            json!({"type":"response.incomplete","sequence_number":2,"response":terminal}),
        ],
        IncompleteTerminal::Accept,
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    assert_eq!(response.error.as_deref(), Some(UNFINISHED));
    assert_eq!(
        response.head().stop,
        Some(StopReason::Error(UNFINISHED.to_owned()))
    );
    let call = response.tool_calls().next().unwrap();
    assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
    assert_eq!(call.function.invalid_arguments, None);
    assert!(call.native.is_none());
}

#[test]
fn explicit_done_and_repeated_finished_items_keep_their_original_native_and_arguments() {
    let original = function("completed", "{\"q\":\"rig\"}");
    let mut contradictory = function("completed", "{\"evil\":2}");
    contradictory["name"] = json!("other");
    for options in [
        ResponsesStreamOptions::strict(),
        ResponsesStreamOptions::strict_with_immediate_tool_calls(),
    ] {
        let response = folded_body_with(
            options,
            &[
                added(7),
                delta(7, "{\"q\":"),
                done(7, &original),
                done(7, &original),
                completed_with_output(json!([contradictory, original])),
            ],
        )
        .expect("completed/finished calls retain the existing route");
        assert_eq!(response.error, None);
        let calls = response.tool_calls().collect::<Vec<_>>();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name.as_str(), "lookup");
        assert_eq!(calls[0].function.arguments_value(), json!({"q":"rig"}));
        assert_eq!(calls[0].native.as_ref().unwrap().item, original);
    }
}

#[test]
fn new_unannounced_terminal_calls_and_custom_items_do_not_acquire_a_pending_function() {
    let mut unrelated = function("completed", "{\"evil\":2}");
    unrelated["id"] = json!("fc_other");
    let custom = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_custom",
        "name":"apply_patch","input":"raw","status":"completed"});
    let response = decode(
        vec![
            added(0),
            delta(0, "{\"q\":\"rig\"}"),
            completed_with_output(json!([unrelated, custom])),
        ],
        IncompleteTerminal::Refuse,
    );
    assert_eq!(response.error.as_deref(), Some(UNFINISHED));
    let calls = response.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name.as_str(), "lookup");
    assert_eq!(calls[0].function.arguments_value(), json!({"q":"rig"}));
    assert!(calls[0].native.is_none());
    let response = decode(
        vec![completed_with_output(json!([function(
            "completed",
            "{\"q\":\"rig\"}"
        )]))],
        IncompleteTerminal::Refuse,
    );
    assert!(
        response.choice.is_empty(),
        "no unannounced streamed call was synthesized"
    );
}

#[tokio::test]
async fn a_pending_function_at_eof_remains_partial_and_finish_is_truncated() {
    use futures::StreamExt;
    let frames = [added(0), delta(0, "{\"q\":\"rig\"}")];
    let driven = drive(&frames);
    assert!(driven.errors.is_empty());
    assert!(driven.response.choice.is_empty());
    assert!(driven.response.response_id().is_none());
    assert!(!driven.items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::End { .. } | StreamEvent::Discard { .. })
    )));
    let mut stream = responses_stream_of(&frames).await;
    while let Some(item) = stream.next().await {
        item.expect("valid prefix");
    }
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[test]
fn terminal_completion_matches_present_empty_and_named_namespaces_exactly() {
    for namespace in ["", "math"] {
        let mut announcement = added(7);
        announcement["item"]["namespace"] = json!(namespace);
        let mut item = function("completed", "{\"q\":\"rig\"}");
        item["namespace"] = json!(namespace);
        let response = decode(
            vec![announcement.clone(), completed_with_output(json!([item]))],
            IncompleteTerminal::Refuse,
        );
        assert_eq!(response.error, None);
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.function.namespace.as_deref(), Some(namespace));
        assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
        assert_eq!(call.native.as_ref().unwrap().item, item);
        let mut omitted = item.clone();
        omitted.as_object_mut().unwrap().shift_remove("namespace");
        let driven = drive(&[announcement, completed_with_output(json!([omitted]))]);
        assert!(
            matches!(driven.errors.as_slice(), [ProviderError::Response(message)]
            if message == "conflicting Responses terminal function call fc_1: namespace contradicts its announcement")
        );
    }
}

#[test]
fn terminal_calls_do_not_supply_missing_announcement_identity() {
    for (field, value) in [("id", ""), ("call_id", "")] {
        let mut announcement = added(0);
        announcement["item"][field] = json!(value);
        let response = decode(
            vec![
                announcement,
                delta(0, "{\"q\":\"rig\"}"),
                completed_with_output(json!([function("completed", "{\"replacement\":9}")])),
            ],
            IncompleteTerminal::Refuse,
        );
        assert_eq!(response.error.as_deref(), Some(UNFINISHED));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
        assert!(call.native.is_none());
        if field == "call_id" {
            assert!(
                call.id.is_local(),
                "no received terminal ID was transplanted"
            );
        }
    }
    let mut announcement = added(0);
    announcement["item"]["name"] = json!("");
    let response = decode(
        vec![
            announcement,
            delta(0, "{\"q\":\"rig\"}"),
            completed_with_output(json!([function("completed", "{\"replacement\":9}")])),
        ],
        IncompleteTerminal::Refuse,
    );
    assert!(
        response.choice.is_empty(),
        "a terminal cannot invent the unannounced name"
    );
}

#[test]
fn completed_custom_calls_keep_the_existing_done_item_route() {
    let item = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_custom",
        "name":"apply_patch","input":"{\"looks\":\"like json\"}","status":"completed"});
    let mut announcement = item.clone();
    announcement["status"] = json!("in_progress");
    let response = decode(
        vec![
            json!({"type":"response.output_item.added","sequence_number":0,
        "output_index":0,"item":announcement}),
            done(0, &item),
            completed_with_output(json!([item])),
        ],
        IncompleteTerminal::Refuse,
    );
    assert_eq!(response.error, None);
    let [AssistantContent::CustomToolCall(call)] = response.choice.as_slice() else {
        panic!("one real done custom call");
    };
    assert_eq!(call.id.provider().unwrap().as_str(), "call_custom");
    assert_eq!(call.name.as_str(), "apply_patch");
    assert_eq!(call.input, "{\"looks\":\"like json\"}");
    assert_eq!(call.native.as_ref().unwrap().item, item);
}

#[test]
fn malformed_pending_input_keeps_its_existing_discard_at_a_genuine_terminal() {
    let driven = drive(&[
        added(0),
        delta(0, "{\"x\":481"),
        completed_with_output(json!([])),
    ]);
    assert!(driven.errors.is_empty());
    assert!(driven.response.choice.is_empty());
    assert!(driven.items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::Discard {
            reason: crate::streaming::DiscardReason::UnfinishedAtTerminal,
            ..
        })
    )));
}

#[test]
fn the_admitted_original_s28_sequences_keep_failure_and_real_terminal_restatement() {
    // The original two fragments and arguments.done are kept. Only required
    // sequence_number and the actual added in_progress status are supplied.
    let mut terminal = completed_with_output(json!([]));
    terminal["sequence_number"] = json!(4);
    let response = decode(
        vec![
            added(0),
            json!({"type":"response.function_call_arguments.delta","sequence_number":1,
            "output_index":0,"delta":"{\"q\":"}),
            json!({"type":"response.function_call_arguments.delta","sequence_number":2,
            "output_index":0,"delta":"\"rig\"}"}),
            json!({"type":"response.function_call_arguments.done","sequence_number":3,
            "output_index":0,"arguments":"{\"q\":\"rig\"}"}),
            terminal,
        ],
        IncompleteTerminal::Refuse,
    );
    assert!(
        response.stop().is_failure(),
        "the call was never finished: {:?}",
        response.stop()
    );
    assert_eq!(response.error.as_deref(), Some(UNFINISHED));
    let call = response.tool_calls().next().unwrap();
    assert_eq!(call.function.arguments_value(), json!({"q":"rig"}));
    assert!(call.native.is_none());
    let item = function("completed", "{\"q\":\"rig\"}");
    let mut terminal = completed_with_output(json!([item]));
    terminal["sequence_number"] = json!(1);
    let response = decode(vec![added(0), terminal], IncompleteTerminal::Refuse);
    assert_eq!(response.stop(), StopReason::ToolUse);
    let natives = response
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => {
                call.native.as_ref().map(|native| native.item.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        natives,
        [item],
        "a call the terminal restates is finished by it"
    );
}
