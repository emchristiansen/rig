//! Observable writer boundaries exercised through the actual decoded fold.

use serde_json::{Value, json};

use super::{Block, CallFragment, Completion, Finish, Turn};
use crate::driver::{Decoded, decode_with};
use crate::error::{ErrorDetail, ErrorKind, ErrorReport, ProviderCallEvidence, ProviderError};
use crate::message::{
    AssistantContent, CallId, CarriedPayload, CarriedPosition, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, Origin, ProviderCallId, ToolName,
};
use crate::streaming::{DiscardReason, PartKind, SourceOrder, StreamEvent, Transcript};
use crate::wire::{Fold, Out, Reply};

fn origin() -> Origin {
    Origin::new("boundary.api", "turn-provider", "requested-model")
}

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("nonempty fixture name")
}

fn reply() -> Reply {
    Reply {
        provider: "transport-provider".to_owned(),
        raw: Value::Null,
        provider_request_id: None,
        response_headers: crate::completion::ProviderResponseHeaders::new(),
    }
}

fn write(
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::new(origin()), "transport-provider", |reply| {
        let mut out = reply.out();
        action(&mut out)?;
        let _ = out.end(Finish::default());
        Ok(())
    })
}

fn transcript(decoded: &Decoded<Completion>) -> Transcript {
    Transcript::from_items(
        decoded
            .items
            .iter()
            .filter_map(|item| item.as_ref().ok().cloned())
            .collect(),
    )
    .expect("actual writer events form a valid prefix")
}

fn closed(decoded: &Decoded<Completion>) -> Transcript {
    let transcript = transcript(decoded);
    let serialized = serde_json::to_value(&transcript).expect("serialize real writer events");
    let parsed = Transcript::parse(serialized).expect("every started part actually closed");
    assert_eq!(parsed, transcript);
    parsed
}

fn relay(events: &Transcript) -> Turn {
    let mut turn = Turn::new(origin());
    for event in events.events() {
        Fold::<Completion>::absorb(&mut turn, event).expect("absorb real emitted event");
    }
    turn
}

fn function(id: &str) -> Block {
    Block::Call {
        id: CallId::from_wire(id),
        name: name("lookup"),
        namespace: None,
    }
}

fn text(value: &str) -> CarriedText {
    let mut text = CarriedText::new(
        CarriedTextContent::Text(value.to_owned()),
        CarriedPayload::new(
            json!({"type":"output_text","text":value,"future":[null,true]}),
            Some("text-issuer".to_owned()),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 8,
            part_index: 2,
        }),
    );
    text.item_id = Some("msg-output".to_owned());
    text.phase = Some(" final_answer ".to_owned());
    text
}

fn reasoning(value: &str) -> CarriedReasoning {
    CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: value.to_owned(),
            signature: Some(" exact signature ".to_owned()),
        }],
        CarriedPayload::new(
            json!({"type":"reasoning","id":"rs-output","future":{"keep":[1,null]}}),
            Some("reasoning-issuer".to_owned()),
        ),
    )
    .with_id(Some("rs-output".to_owned()))
}

#[test]
fn a_started_discard_closes_its_real_part_without_erasing_adjacent_native_siblings() {
    for reason in [
        DiscardReason::Length,
        DiscardReason::Incomplete,
        DiscardReason::UnfinishedAtTerminal,
    ] {
        let before = json!({"type":"text","id":"before","text":"before"});
        let after = json!({"type":"function_call","call_id":"after","arguments":"{}"});
        let decoded = write(|out| {
            out.whole(40, Block::Text, before.clone(), "before")?;
            out.in_source_order(SourceOrder::new(4, 2), |out| {
                out.open(7, function("discarded"), json!({"call_id":"discarded"}))?;
                out.push(7, "{\"cut\":")
            })?;
            out.whole(2, function("after"), after.clone(), "{}")?;
            out.discard_call(7, reason)
        });
        let events = closed(&decoded);
        let discarded: Vec<_> = events
            .events()
            .filter(|event| event.part().index() == 1)
            .collect();
        assert!(matches!(discarded.as_slice(), [StreamEvent::Start {
            kind: PartKind::ToolCall, source_order: Some(order), ..
        }, StreamEvent::Arguments { json, .. }, StreamEvent::Discard { reason: held, .. }]
            if *order == SourceOrder::new(4, 2) && json == "{\"cut\":" && *held == reason));
        assert!(
            discarded
                .iter()
                .all(|event| !matches!(event, StreamEvent::End { .. }))
        );
        assert!(
            discarded
                .iter()
                .all(|event| event.part() == discarded[0].part())
        );
        let complete_partial = relay(&events).partial(Some(&Finish::default()), &reply(), None);
        let response = decoded
            .outcome
            .expect("discard is a successful protocol closure");
        assert_eq!(response.choice.len(), 2);
        assert_eq!(response.choice[0].native_item(), Some(&before));
        assert_eq!(response.choice[1].native_item(), Some(&after));
        assert_eq!(response.tool_calls().count(), 1);
        assert_eq!(
            response
                .tool_calls()
                .next()
                .expect("adjacent completed call")
                .id,
            CallId::from_wire("after")
        );
        assert_eq!(complete_partial.choice, response.choice);
        let complete = Fold::<Completion>::finish(relay(&events), Finish::default(), reply())
            .expect("real events fold identically across a discard hole");
        assert_eq!(complete.choice, response.choice);
    }
}

#[test]
fn a_call_discarded_before_its_name_emits_no_fabricated_start_or_end() {
    let decoded = write(|out| {
        out.fragment(
            Some(3),
            CallFragment {
                id: None,
                name: None,
                arguments: Some("{\"cut\":"),
            },
        )?;
        out.discard_call(3, DiscardReason::Length)?;
        out.whole(9, Block::Text, json!({"id":"answer"}), "answer")
    });
    let events = closed(&decoded);
    assert!(events.events().all(|event| event.part().index() == 1));
    assert_eq!(events.events().count(), 3);
    assert!(
        events
            .events()
            .all(|event| !matches!(event, StreamEvent::Discard { .. }))
    );
    let response = decoded
        .outcome
        .expect("the unseen call does not create a fold hole failure");
    assert_eq!(response.choice.len(), 1);
    assert_eq!(response.text(), "answer");
    assert_eq!(response.tool_calls().count(), 0);
    assert_eq!(
        response.choice[0].native_item(),
        Some(&json!({"id":"answer"}))
    );
}

#[test]
fn discarding_a_missing_or_nonfunction_item_refuses_before_mutating_its_content() {
    let decoded = write(|out| {
        assert!(matches!(
            out.discard_call(99, DiscardReason::Length),
            Err(ProviderError::Response(_))
        ));
        out.open(0, Block::Text, json!({"id":"text"}))?;
        out.push(0, "kept")?;
        assert!(matches!(
            out.discard_call(0, DiscardReason::Length),
            Err(ProviderError::Response(_))
        ));
        out.push(0, " whole")?;
        out.finish(0)?;
        out.open(
            1,
            Block::CustomCall {
                id: CallId::from_wire("custom"),
                name: name("patch"),
                namespace: Some(String::new()),
            },
            json!({"type":"custom_tool_call","input":" raw "}),
        )?;
        out.push(1, " raw ")?;
        assert!(matches!(
            out.discard_call(1, DiscardReason::Incomplete),
            Err(ProviderError::Response(_))
        ));
        out.finish(1)
    });
    let events = closed(&decoded);
    assert!(
        events
            .events()
            .all(|event| !matches!(event, StreamEvent::Discard { .. }))
    );
    let response = decoded
        .outcome
        .expect("rejected disposal left the items open");
    assert_eq!(response.text(), "kept whole");
    let AssistantContent::CustomToolCall(custom) = &response.choice[1] else {
        panic!("raw custom input remains a custom call");
    };
    assert_eq!(custom.input, " raw ");
    assert_eq!(custom.namespace.as_deref(), Some(""));
    assert_eq!(
        response.choice[0].native_item(),
        Some(&json!({"id":"text"}))
    );
    assert!(response.choice[1].native_item().is_some());
}

#[test]
fn terminal_issuer_stamps_only_unknown_folded_carried_provenance_after_ends_were_delivered() {
    let native = json!({"type":"text","id":"native","text":"native answer"});
    let raw = json!({"type":"future_item","exact":{"array":[true,null]}});
    let mut unknown = reasoning("plan");
    unknown.payload.provider = None;
    unknown.content.extend([
        CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
            json!({"type":"future_summary","keep":[3,null]}),
            None,
        )),
        CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
            json!({"type":"foreign_part","keep":true}),
            Some("foreign/part".to_owned()),
        )),
    ]);
    let mut foreign = unknown.clone();
    foreign.payload.provider = Some("foreign/item".to_owned());
    let mut carried_text = text("carried answer");
    carried_text.payload.provider = None;
    let mut delivered = Vec::new();
    let decoded = decode_with(Turn::new(origin()), "transport-provider", |reply| {
        let mut out = reply.out();
        out.whole(0, Block::Text, native.clone(), "native answer")?;
        for (index, provider) in [
            (1, None),
            (2, Some("foreign/item".to_owned())),
            (3, Some(String::new())),
        ] {
            out.open(index, Block::ProviderItem { provider }, raw.clone())?;
            out.finish(index)?;
        }
        out.carried_reasoning(unknown.clone())?;
        out.carried_reasoning(foreign.clone())?;
        out.carried_text(carried_text.clone())?;
        // Take absorbs each event in the live fold before returning the
        // immutable value that the consumer already received.
        {
            let mut shared = out.lock();
            while let Some(item) = shared.take() {
                delivered.push(item.expect("writer item"));
            }
        }
        out.stamp_carried_issuer("terminal/issuer");
        let _ = out.end(Finish {
            model: Some("reported-model".to_owned()),
            response_id: Some("response-id".to_owned()),
            ..Finish::default()
        });
        Ok(())
    });
    let delivered = Transcript::from_items(delivered).expect("actual delivered ends");
    assert_eq!(
        Transcript::parse(serde_json::to_value(&delivered).expect("serialize"))
            .expect("all delivered parts closed"),
        delivered
    );
    let ended: Vec<_> = delivered
        .events()
        .filter_map(|event| {
            if let StreamEvent::End { content, .. } = event {
                Some(content)
            } else {
                None
            }
        })
        .collect();
    let AssistantContent::ProviderItem(delivered_item) = ended[1] else {
        panic!("the delivered whole item remains carried");
    };
    assert_eq!(delivered_item.payload.provider, None);
    assert_eq!(delivered_item.payload.item, raw);
    assert_eq!(
        ended[4],
        &AssistantContent::CarriedReasoning(unknown.clone())
    );
    assert_eq!(
        ended[5],
        &AssistantContent::CarriedReasoning(foreign.clone())
    );
    assert_eq!(
        ended[6],
        &AssistantContent::CarriedText(carried_text.clone())
    );
    assert!(
        decoded.items.is_empty(),
        "the consumer already received every end"
    );
    let response = decoded
        .outcome
        .expect("the already absorbed reply finishes");
    let mut expected_origin = origin();
    expected_origin.response_model = Some("reported-model".to_owned());
    expected_origin.response_id = Some("response-id".to_owned());
    assert_eq!(response.origin, expected_origin);
    assert_eq!(response.choice[0].native_item(), Some(&native));
    let AssistantContent::ProviderItem(stamped) = &response.choice[1] else {
        panic!("the folded item remains carried");
    };
    assert_eq!(stamped.payload.provider.as_deref(), Some("terminal/issuer"));
    assert_eq!(stamped.payload.item, raw);
    for (index, provider) in [(2, "foreign/item"), (3, "")] {
        let AssistantContent::ProviderItem(item) = &response.choice[index] else {
            panic!("explicit issuer item remains carried");
        };
        assert_eq!(item.payload.provider.as_deref(), Some(provider));
        assert_eq!(item.payload.item, raw);
    }
    let mut stamped_reasoning = unknown;
    stamped_reasoning.payload.provider = Some("terminal/issuer".to_owned());
    let CarriedReasoningPart::OpaqueSummary(payload) = &mut stamped_reasoning.content[1] else {
        panic!("fixture's unknown summary");
    };
    payload.provider = Some("terminal/issuer".to_owned());
    assert_eq!(
        response.choice[4],
        AssistantContent::CarriedReasoning(stamped_reasoning)
    );
    assert_eq!(
        response.choice[5],
        AssistantContent::CarriedReasoning(foreign)
    );
    // G179 changes folded text provenance only; the delivered End above stays unknown.
    carried_text.payload.provider = Some("terminal/issuer".to_owned());
    assert_eq!(
        response.choice[6],
        AssistantContent::CarriedText(carried_text)
    );
    assert!(
        response.choice[1..]
            .iter()
            .all(|content| content.native_item().is_none())
    );
}

#[test]
fn carried_text_finalization_keeps_one_part_and_refuses_mismatches_before_mutation() {
    let expected = text(" \nanswer\t ");
    let decoded = write(|out| {
        out.in_source_order(SourceOrder::new(8, 2), |out| {
            out.open(11, Block::Text, json!({"id":"old-native"}))?;
            out.push(11, " \n")?;
            out.push(11, "answer\t ")
        })?;
        assert!(matches!(
            out.finish_carried_text(11, text("different")),
            Err(ProviderError::Response(_))
        ));
        assert!(matches!(
            out.finish_carried_reasoning(11, reasoning(expected.text())),
            Err(ProviderError::Response(_))
        ));
        out.edit(11, |item| item["backfill"] = json!(true))?;
        out.finish_carried_text(11, expected.clone())
    });
    let events = closed(&decoded);
    let events: Vec<_> = events.events().collect();
    assert!(matches!(events.as_slice(), [StreamEvent::Start {
        kind: PartKind::Text, source_order: Some(order), ..
    }, StreamEvent::Text { text: first, .. }, StreamEvent::Text { text: second, .. },
        StreamEvent::End { content: AssistantContent::CarriedText(ended), .. }]
        if *order == SourceOrder::new(8, 2) && first == " \n" && second == "answer\t "
        && ended == &expected));
    assert!(events.iter().all(|event| event.part() == events[0].part()));
    let response = decoded
        .outcome
        .expect("valid replacement still closes the original draft");
    assert_eq!(response.choice, [AssistantContent::CarriedText(expected)]);
    assert_eq!(response.origin, origin());
    assert!(response.choice[0].native_item().is_none());
}

#[test]
fn carried_reasoning_finalization_keeps_one_part_and_exact_ordered_visible_projection() {
    let mut expected = reasoning("think");
    expected.content.extend([
        CarriedReasoningPart::Encrypted(" ciphertext ".to_owned()),
        CarriedReasoningPart::Summary("sum".to_owned()),
        CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(json!({"future":9}), None)),
    ]);
    let decoded = write(|out| {
        out.in_source_order(SourceOrder::new(3, 5), |out| {
            out.open(
                70,
                Block::Reasoning { redacted: false },
                json!({"old":"native"}),
            )?;
            out.push(70, "think")?;
            out.push(70, "\nsum")
        })?;
        assert!(matches!(
            out.finish_carried_reasoning(70, reasoning("think sum")),
            Err(ProviderError::Response(_))
        ));
        assert!(matches!(
            out.finish_carried_text(70, text("think\nsum")),
            Err(ProviderError::Response(_))
        ));
        out.finish_carried_reasoning(70, expected.clone())
    });
    let events = closed(&decoded);
    let events: Vec<_> = events.events().collect();
    assert!(matches!(events.as_slice(), [StreamEvent::Start {
        kind: PartKind::Reasoning, source_order: Some(order), ..
    }, StreamEvent::Reasoning { text: first, .. }, StreamEvent::Reasoning { text: second, .. },
        StreamEvent::End { content: AssistantContent::CarriedReasoning(ended), .. }]
        if *order == SourceOrder::new(3, 5) && first == "think" && second == "\nsum"
        && ended == &expected));
    assert!(events.iter().all(|event| event.part() == events[0].part()));
    let response = decoded
        .outcome
        .expect("metadata closes the unchanged visible text");
    assert_eq!(response.reasoning(), "think\nsum");
    assert_eq!(
        response.choice,
        [AssistantContent::CarriedReasoning(expected)]
    );
    assert_eq!(response.origin, origin());
    assert!(response.choice[0].native_item().is_none());
}

#[test]
fn malformed_input_reports_the_real_draft_id_or_a_distinct_writer_owned_id_and_exact_evidence() {
    let raw = " \n{\"x\":\t ";
    let parser_error = serde_json::from_str::<Value>(raw)
        .expect_err("malformed fixture")
        .to_string();
    for (wire_id, duplicate) in [
        (Some("provider-call"), false),
        (None, false),
        (Some("duplicate"), true),
    ] {
        let provider = wire_id.map(|wire_id| ProviderCallEvidence {
            call_id: ProviderCallId::new(wire_id).expect("fixture provider ID"),
            item_id: Some("output-item-only".to_owned()),
        });
        let decoded = decode_with(Turn::new(origin()), "transport-provider", |reply| {
            let mut out = reply.out();
            if duplicate {
                out.whole(
                    0,
                    function("duplicate"),
                    json!({"call_id":"duplicate","item_id":"first-output"}),
                    "{}",
                )?;
            }
            out.fragment(
                Some(7),
                CallFragment {
                    id: wire_id,
                    name: Some("lookup"),
                    arguments: Some(raw),
                },
            )?;
            out.edit(7, |item| *item = json!({"item_id":"output-item-only"}))?;
            Err(out.malformed_call_error(7, raw.to_owned(), parser_error.clone(), provider.clone()))
        });
        let prefix = transcript(&decoded);
        let failed_part = prefix
            .events()
            .find(|event| matches!(event, StreamEvent::Arguments { json, .. } if json == raw))
            .expect("the exact malformed fragment was emitted")
            .part();
        let failed: Vec<_> = prefix
            .events()
            .filter(|event| event.part() == failed_part)
            .collect();
        assert!(matches!(failed.as_slice(), [StreamEvent::Start {
            kind: PartKind::ToolCall, name: Some(tool), ..
        }, StreamEvent::Arguments { json, .. }] if tool == "lookup" && json == raw));
        assert!(
            prefix
                .events()
                .all(|event| !matches!(event, StreamEvent::Discard { .. }))
        );
        assert!(
            Transcript::parse(serde_json::to_value(&prefix).expect("serialize")).is_err(),
            "a diagnostic never fabricates closure of the failed call"
        );
        let completed: Vec<_> = prefix
            .events()
            .filter_map(|event| {
                if let StreamEvent::End {
                    content: AssistantContent::ToolCall(call),
                    ..
                } = event
                {
                    Some(call)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(completed.len(), usize::from(duplicate));
        let error = decoded
            .outcome
            .expect_err("malformed authoritative call is a typed error");
        assert!(matches!(error, ProviderError::Relayed(_)));
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::Response);
        assert!(!report.retryable);
        let Some(ErrorDetail::MalformedToolInput(detail)) = &report.detail else {
            panic!("the actual draft supplies the structured diagnostic");
        };
        assert_eq!(detail.name, "lookup");
        assert_eq!(detail.raw, raw);
        assert_eq!(detail.error, parser_error);
        assert_eq!(detail.provider, provider);
        if let Some(wire_id) = wire_id.filter(|_| !duplicate) {
            assert_eq!(detail.id, CallId::from_wire(wire_id));
        } else {
            assert!(
                detail.id.is_local(),
                "the writer issues missing/duplicate correlation"
            );
        }
        if let Some(first) = completed.first() {
            assert_ne!(
                detail.id, first.id,
                "duplicate correlation cannot answer the sibling"
            );
            assert_eq!(first.id, CallId::from_wire("duplicate"));
            assert!(
                first.native.is_some(),
                "the completed sibling's delivered End remains exact"
            );
        }
        let roundtrip: ErrorReport = serde_json::from_value(
            serde_json::to_value(&report).expect("serialize typed diagnostic"),
        )
        .expect("diagnostic reads back");
        assert_eq!(roundtrip, report);
        assert_eq!(
            decoded
                .items
                .last()
                .expect("terminal item")
                .as_ref()
                .expect_err("the original typed error is in band")
                .report(),
            report
        );
    }
}

#[test]
fn malformed_call_refusal_cannot_consume_a_missing_or_nonfunction_draft() {
    let decoded = write(|out| {
        let missing = out.malformed_call_error(9, "raw".to_owned(), "parser".to_owned(), None);
        assert!(matches!(missing, ProviderError::Response(_)));
        assert!(missing.report().detail.is_none());
        out.open(0, Block::Text, json!({"id":"text"}))?;
        out.push(0, "before")?;
        let wrong_kind = out.malformed_call_error(0, "raw".to_owned(), "parser".to_owned(), None);
        assert!(matches!(wrong_kind, ProviderError::Response(_)));
        assert!(wrong_kind.report().detail.is_none());
        out.push(0, " after")?;
        out.finish(0)
    });
    let events = closed(&decoded);
    assert_eq!(events.events().count(), 4);
    assert!(
        events
            .events()
            .all(|event| !matches!(event, StreamEvent::Discard { .. }))
    );
    let response = decoded
        .outcome
        .expect("the rejected operation leaves the real draft usable");
    assert_eq!(response.text(), "before after");
    assert_eq!(response.tool_calls().count(), 0);
    assert_eq!(
        response.choice[0].native_item(),
        Some(&json!({"id":"text"}))
    );
}

#[test]
fn default_nonobject_argument_normalization_stays_upstream_behavior() {
    for (raw, expected, invalid) in [
        (" null ", json!({}), None),
        (r#""{\"x\":1}""#, json!({"x":1}), None),
        (" [1,null] ", json!({}), Some(" [1,null] ")),
        (" 7 ", json!({}), Some(" 7 ")),
    ] {
        let decoded = write(|out| out.whole(0, function("call"), Value::Null, raw));
        closed(&decoded);
        let response = decoded
            .outcome
            .expect("default parsing still retains the call");
        let call = response.tool_calls().next().expect("one function");
        assert_eq!(call.function.arguments_value(), expected, "{raw}");
        assert_eq!(call.function.invalid_arguments.as_deref(), invalid, "{raw}");
    }
}

#[test]
fn explicit_nonobject_preservation_keeps_null_and_string_object_as_reported_values() {
    for raw in [" \nnull\t ", r#"  "{\"x\":1}"  "#, " [1,null] "] {
        let expected = serde_json::from_str::<Value>(raw)
            .expect("valid JSON")
            .to_string();
        let native = json!({"type":"function_call","arguments":raw,"call_id":"carried"});
        let decoded = write(|out| {
            out.open(0, function("carried"), native.clone())?;
            out.push(0, raw)?;
            out.preserve_non_object_call_arguments(0)?;
            out.finish(0)
        });
        let events = closed(&decoded);
        let response = decoded
            .outcome
            .expect("valid nonobject value is retained canonically");
        let call = response.tool_calls().next().expect("one function");
        assert_eq!(call.function.arguments_value(), json!({}));
        assert_eq!(
            call.function.invalid_arguments.as_deref(),
            Some(expected.as_str())
        );
        assert_eq!(response.choice[0].native_item(), Some(&native));
        let ended = events
            .events()
            .find_map(|event| {
                if let StreamEvent::End {
                    content: AssistantContent::ToolCall(call),
                    ..
                } = event
                {
                    Some(call)
                } else {
                    None
                }
            })
            .expect("the preserved function's actual End");
        assert_eq!(ended, call);
    }
    let decoded = write(|out| {
        out.open(0, function("object"), Value::Null)?;
        out.push(0, "{\"x\":1}")?;
        assert!(matches!(
            out.preserve_non_object_call_arguments(0),
            Err(ProviderError::Response(_))
        ));
        out.finish(0)
    });
    closed(&decoded);
    let response = decoded
        .outcome
        .expect("rejecting an object as nonobject did not mutate it");
    let call = response.tool_calls().next().expect("object function");
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
    assert!(call.function.invalid_arguments.is_none());
}

#[test]
fn canonical_argument_changes_clear_preservation_while_raw_metadata_backfill_keeps_it() {
    for restate in [false, true] {
        let decoded = write(|out| {
            out.open(0, function("updated"), Value::Null)?;
            out.push(0, "null")?;
            out.preserve_non_object_call_arguments(0)?;
            if restate {
                out.restate(0, "{\"x\":2}")?;
            } else {
                out.push(0, "{\"x\":2}")?;
            }
            out.finish(0)
        });
        closed(&decoded);
        let response = decoded
            .outcome
            .expect("the actual new argument object replaces null");
        let call = response.tool_calls().next().expect("updated function");
        assert_eq!(call.function.arguments_value(), json!({"x":2}));
        assert!(call.function.invalid_arguments.is_none());
    }
    let raw = r#""{\"x\":1}""#;
    let backfilled = json!({"type":"function_call","call_id":"backfilled",
        "arguments":raw,"future":{"keep":[true,null]}});
    let decoded = write(|out| {
        out.open(0, function("backfilled"), json!({"call_id":"backfilled"}))?;
        out.push(0, raw)?;
        out.preserve_non_object_call_arguments(0)?;
        out.edit(0, |item| *item = backfilled.clone())?;
        out.finish(0)
    });
    closed(&decoded);
    let response = decoded
        .outcome
        .expect("metadata backfill keeps parsed argument semantics");
    let call = response.tool_calls().next().expect("backfilled function");
    assert!(call.function.arguments.is_empty());
    assert_eq!(call.function.invalid_arguments.as_deref(), Some(raw));
    assert_eq!(response.choice[0].native_item(), Some(&backfilled));
}
