//! Writer paths and contract refusals no other completion test reaches,
//! exercised through the actual decoded fold.

use serde_json::{Value, json};

use super::{AUTO_INDEX, Block, CallFragment, Completion, Finish, MAX_TOOL_INPUT_BYTES, Turn};
use crate::completion::{CompletionRequest, FinishReason};
use crate::driver::{Decoded, decode_with};
use crate::error::{
    ErrorDetail, ErrorKind, ErrorReport, MalformedToolInput, ProviderCallEvidence, ProviderError,
};
use crate::message::{
    AssistantContent, CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText,
    CarriedTextContent, CustomToolCall, DocumentSourceKind, Image, Opaque, Origin, ProviderCallId,
    ProviderItem, Reasoning, Text, ToolCall, ToolFunction, ToolName,
};
use crate::streaming::{Item, Part, PartKind, StreamEvent, ToolSlotState};
use crate::wire::{Call, Descriptor, Fold, Mode, Operation, Out, Reply};

fn origin() -> Origin {
    Origin::new("coverage.api", "turn-provider", "requested-model")
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

/// Every event the actual writer emitted, in order.
fn events(decoded: &Decoded<Completion>) -> Vec<StreamEvent> {
    decoded
        .items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(event)) => Some(event.clone()),
            _ => None,
        })
        .collect()
}

/// `result` is exactly the writer's response refusal `expected`.
#[track_caller]
fn refused<T: std::fmt::Debug>(result: Result<T, ProviderError>, expected: &str) {
    match result {
        Err(ProviderError::Response(message)) => assert_eq!(message, expected),
        other => panic!("expected the response refusal {expected:?}, got {other:?}"),
    }
}

fn start(index: u32, kind: PartKind) -> StreamEvent {
    StreamEvent::Start {
        part: Part::new(index),
        kind,
        source_order: None,
        name: None,
        namespace: None,
        tool_slot_state: None,
    }
}

fn named_start(
    index: u32,
    kind: PartKind,
    tool: &str,
    tool_slot_state: Option<ToolSlotState>,
) -> StreamEvent {
    StreamEvent::Start {
        part: Part::new(index),
        kind,
        source_order: None,
        name: Some(name(tool)),
        namespace: None,
        tool_slot_state,
    }
}

fn end(index: u32, content: AssistantContent) -> StreamEvent {
    StreamEvent::End {
        part: Part::new(index),
        content,
        reasoning_authority: None,
    }
}

fn custom(id: &str) -> Block {
    Block::CustomCall {
        id: CallId::from_wire(id),
        name: name("patch"),
        namespace: None,
    }
}

fn custom_call(id: &str, input: &str) -> CustomToolCall {
    CustomToolCall::new(CallId::from_wire(id), name("patch"), input)
}

fn function_call(id: &str, tool: &str, arguments: Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire(id),
        ToolFunction::new(name(tool), arguments),
    ))
}

fn carried_text(value: &str) -> CarriedText {
    CarriedText::new(
        CarriedTextContent::Text(value.to_owned()),
        CarriedPayload::new(
            json!({"type":"output_text","text":value}),
            Some("text-issuer".to_owned()),
        ),
    )
}

fn carried_reasoning(value: &str) -> CarriedReasoning {
    CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: value.to_owned(),
            signature: None,
        }],
        CarriedPayload::new(
            json!({"type":"reasoning","id":"rs-run"}),
            Some("reasoning-issuer".to_owned()),
        ),
    )
}

#[test]
fn reasoning_authority_refusals_leave_items_and_events_unchanged() {
    use super::{Body, Closing, Items};
    use crate::streaming::ReasoningAuthority;

    for authority in [
        ReasoningAuthority::Partial,
        ReasoningAuthority::ProviderRestated,
    ] {
        let mut turn = Turn::new(origin());
        let mut items = Items::new();
        let native = json!({"type":"text","text":"kept","extra":{"kept":true}});
        turn.open_item(&mut items, 0, Block::Text, native.clone())
            .expect("the text item opens");
        turn.push_item(&mut items, 0, "kept")
            .expect("the text fragment is accepted");
        let before = [
            start(0, PartKind::Text),
            StreamEvent::Text {
                part: Part::new(0),
                text: "kept".to_owned(),
            },
        ];

        for index in [9, 0] {
            refused(
                turn.close_item_with_reasoning_authority(
                    &mut items,
                    index,
                    Closing::Complete,
                    Some(authority),
                ),
                &format!(
                    "the reply assigned reasoning authority to item {index}, which is not reasoning"
                ),
            );
            assert_eq!(turn.open.len(), 1);
            let draft = turn.open.get(&0).expect("the text item is still open");
            assert_eq!(draft.item, native);
            assert_eq!(draft.part, Part::new(0));
            assert_eq!(draft.source_order, None);
            assert_eq!(draft.tool_slot_state, None);
            assert!(draft.started);
            assert!(draft.citations.is_empty());
            assert!(matches!(&draft.body, Body::Text(text) if text == "kept"));
            assert_eq!(turn.next_part, 1);
            assert!(turn.ended.is_empty());
            assert!(turn.snapshot().is_empty());
            assert_eq!(items.len(), before.len());
            for (actual, expected) in items.iter().zip(&before) {
                assert!(matches!(actual, Ok(Item::Event(event)) if event == expected));
            }
        }

        turn.close_item(&mut items, 0, Closing::Complete)
            .expect("ordinary text still closes without reasoning authority");
        let reasoning = json!({"type":"reasoning","id":"reasoning-1"});
        turn.open_item(
            &mut items,
            1,
            Block::Reasoning { redacted: false },
            reasoning.clone(),
        )
        .expect("the reasoning item opens");
        turn.push_item(&mut items, 1, "thought")
            .expect("the reasoning fragment is accepted");
        turn.close_item_with_reasoning_authority(&mut items, 1, Closing::Complete, Some(authority))
            .expect("reasoning closes with its stated authority");
        let expected = [
            before[0].clone(),
            before[1].clone(),
            end(
                0,
                AssistantContent::Text(Text::new("kept")).with_native(native),
            ),
            start(1, PartKind::Reasoning),
            StreamEvent::Reasoning {
                part: Part::new(1),
                text: "thought".to_owned(),
            },
            StreamEvent::End {
                part: Part::new(1),
                content: AssistantContent::Reasoning(Reasoning {
                    text: "thought".to_owned(),
                    redacted: false,
                    native: None,
                })
                .with_native(reasoning),
                reasoning_authority: Some(authority),
            },
        ];
        assert!(turn.open.is_empty());
        assert_eq!(turn.ended.get(&0), Some(&0));
        assert_eq!(turn.ended.get(&1), Some(&1));
        assert_eq!(items.len(), expected.len());
        for (actual, expected) in items.iter().zip(&expected) {
            assert!(matches!(actual, Ok(Item::Event(event)) if event == expected));
        }
    }
}

#[test]
fn a_wire_without_a_replay_target_is_refused_by_prepare_and_folds_under_its_own_name() {
    match Completion::prepare(
        CompletionRequest::new("question"),
        &Descriptor::new("bare.wire"),
    )
    .map(|_| ())
    {
        Err(ProviderError::Request(source)) => assert_eq!(
            source.to_string(),
            "completion wire `bare.wire` names no replay target"
        ),
        other => panic!("expected the request refusal, got {other:?}"),
    }
    let request = CompletionRequest::new("question");
    let descriptor = Descriptor::new("bare.wire");
    for mode in [Mode::Unary, Mode::Streaming] {
        let turn = Completion::fold(&request, &mut Call::new(&descriptor, mode));
        assert_eq!(turn.origin(), &Origin::new("bare.wire", "bare.wire", ""));
        assert!(!turn.wire);
        assert_eq!(turn.call_id_slot, None);
        assert!(!turn.accept_unknown_finish);
    }
}

#[test]
fn items_that_hold_no_writable_text_refuse_fragments_and_reopening_without_mutation() {
    let image = Image {
        data: DocumentSourceKind::Url("https://example.com/cat.png".to_owned()),
        ..Image::default()
    };
    let opaque = json!({"type":"opaque_item"});
    let carried = json!({"type":"carried_item"});
    let decoded = write(|out| {
        out.open(0, Block::Text, Value::Null)?;
        refused(
            out.open(0, Block::Reasoning { redacted: false }, Value::Null),
            "the reply opened item 0 twice",
        );
        refused(
            out.preserve_non_object_call_arguments(0),
            "the reply preserved non-object arguments for item 0, which is not a function call",
        );
        refused(
            out.fragment(
                Some(0),
                CallFragment {
                    id: None,
                    name: None,
                    arguments: Some("{}"),
                },
            ),
            "the reply sent a call fragment for item 0, which is not a call",
        );
        // Announced arguments belong to calls only; a text ignores them.
        out.announce(0, json!({"ignored":true}))?;
        refused(
            out.announce(9, json!({})),
            "the reply wrote to item 9, which is not open",
        );
        out.push(0, "kept")?;
        out.finish(0)?;
        out.open(1, Block::Image(image.clone()), Value::Null)?;
        refused(
            out.push(1, "aGk="),
            "the reply wrote data to the image item 1, which holds no base64 data",
        );
        out.finish(1)?;
        out.open(2, Block::Opaque { replay: false }, opaque.clone())?;
        refused(
            out.push(2, "text"),
            "the reply wrote text to the opaque item 2",
        );
        out.finish(2)?;
        out.open(3, Block::ProviderItem { provider: None }, carried.clone())?;
        refused(
            out.push(3, "text"),
            "the reply wrote text to the opaque item 3",
        );
        out.finish(3)
    });
    let kept = AssistantContent::Text(Text::new("kept"));
    let opaque = AssistantContent::Opaque(Opaque {
        item: opaque,
        replay: false,
    });
    let carried = AssistantContent::ProviderItem(ProviderItem {
        payload: CarriedPayload::new(carried, None),
    });
    assert_eq!(
        events(&decoded),
        vec![
            start(0, PartKind::Text),
            StreamEvent::Text {
                part: Part::new(0),
                text: "kept".to_owned(),
            },
            end(0, kept.clone()),
            start(1, PartKind::Image),
            end(1, AssistantContent::Image(image.clone())),
            start(2, PartKind::Opaque),
            end(2, opaque.clone()),
            start(3, PartKind::ProviderItem),
            end(3, carried.clone()),
        ]
    );
    let response = decoded.outcome.expect("every refusal left its item usable");
    assert_eq!(
        response.choice,
        vec![kept, AssistantContent::Image(image), opaque, carried]
    );
}

#[test]
fn custom_calls_qualify_restate_and_state_their_empty_input_at_close() {
    let native = json!({"type":"custom_tool_call","call_id":"empty"});
    let decoded = write(|out| {
        refused(
            out.qualify(9, None),
            "the reply wrote to item 9, which is not open",
        );
        out.open(0, Block::Text, Value::Null)?;
        refused(
            out.qualify(0, Some("tools".to_owned())),
            "the reply qualified item 0, which is not a tool call",
        );
        out.push(0, "kept")?;
        out.finish(0)?;
        out.open(1, custom("empty"), native.clone())?;
        out.finish(1)?;
        out.open(2, custom("restated"), Value::Null)?;
        out.qualify(2, Some("tools".to_owned()))?;
        out.push(2, "draft")?;
        out.restate(2, "final")?;
        out.finish(2)?;
        out.open(3, custom("cleared"), Value::Null)?;
        out.push(3, "draft")?;
        out.restate(3, "")?;
        out.finish(3)
    });
    let kept = AssistantContent::Text(Text::new("kept"));
    let empty = AssistantContent::CustomToolCall(custom_call("empty", "")).with_native(native);
    let restated = AssistantContent::CustomToolCall(
        custom_call("restated", "final").with_namespace(Some("tools".to_owned())),
    );
    let cleared = AssistantContent::CustomToolCall(custom_call("cleared", ""));
    let input = |index: u32, input: &str| StreamEvent::Input {
        part: Part::new(index),
        input: input.to_owned(),
    };
    assert_eq!(
        events(&decoded),
        vec![
            start(0, PartKind::Text),
            StreamEvent::Text {
                part: Part::new(0),
                text: "kept".to_owned(),
            },
            end(0, kept.clone()),
            named_start(1, PartKind::CustomToolCall, "patch", None),
            input(1, ""),
            end(1, empty.clone()),
            named_start(2, PartKind::CustomToolCall, "patch", None),
            input(2, "draft"),
            end(2, restated.clone()),
            named_start(3, PartKind::CustomToolCall, "patch", None),
            input(3, "draft"),
            input(3, ""),
            end(3, cleared.clone()),
        ]
    );
    let response = decoded.outcome.expect("every custom call closed");
    assert_eq!(response.choice, vec![kept, empty, restated, cleared]);
}

#[test]
fn pending_slots_refuse_named_or_started_items_and_replace_a_preserved_null_once() {
    const PENDING: &str = "the reply opened a pending tool slot after it was named or started";
    let decoded = write(|out| {
        refused(
            out.pending_call(
                0,
                CallFragment {
                    id: None,
                    name: Some("lookup"),
                    arguments: None,
                },
            ),
            PENDING,
        );
        out.open(1, Block::Text, Value::Null)?;
        refused(out.pending_call(1, CallFragment::default()), PENDING);
        out.pending_call(
            2,
            CallFragment {
                id: Some("call-2"),
                name: None,
                arguments: Some("null"),
            },
        )?;
        refused(out.pending_call(2, CallFragment::default()), PENDING);
        out.push(2, "{\"a\":1}")?;
        out.fragment(
            Some(2),
            CallFragment {
                id: None,
                name: Some("lookup"),
                arguments: None,
            },
        )?;
        out.push(1, "kept")?;
        out.finish(1)?;
        out.finish(2)
    });
    let kept = AssistantContent::Text(Text::new("kept"));
    let call = function_call("call-2", "lookup", json!({"a":1}));
    assert_eq!(
        events(&decoded),
        vec![
            StreamEvent::Start {
                part: Part::new(1),
                kind: PartKind::ToolCall,
                source_order: None,
                name: None,
                namespace: None,
                tool_slot_state: Some(ToolSlotState::PendingName),
            },
            StreamEvent::Arguments {
                part: Part::new(1),
                json: "null".to_owned(),
            },
            StreamEvent::Arguments {
                part: Part::new(1),
                json: "{\"a\":1}".to_owned(),
            },
            named_start(1, PartKind::ToolCall, "lookup", Some(ToolSlotState::Named)),
            start(0, PartKind::Text),
            StreamEvent::Text {
                part: Part::new(0),
                text: "kept".to_owned(),
            },
            end(0, kept.clone()),
            end(1, call.clone()),
        ]
    );
    let response = decoded
        .outcome
        .expect("refused pending slots left the reply usable");
    assert_eq!(response.choice, vec![kept, call]);
}

#[test]
fn an_indexless_fragment_with_a_known_id_continues_the_call_that_holds_it() {
    let decoded = write(|out| {
        out.fragment(
            None,
            CallFragment {
                id: Some("call-a"),
                name: Some("lookup"),
                arguments: Some("{\"q\":"),
            },
        )?;
        out.fragment(
            None,
            CallFragment {
                id: Some("call-b"),
                name: Some("search"),
                arguments: Some("{}"),
            },
        )?;
        out.fragment(
            None,
            CallFragment {
                id: Some("call-a"),
                name: None,
                arguments: Some("1}"),
            },
        )?;
        out.finish_open()
    });
    let arguments = |index: u32, json: &str| StreamEvent::Arguments {
        part: Part::new(index),
        json: json.to_owned(),
    };
    let first = function_call("call-a", "lookup", json!({"q":1}));
    let second = function_call("call-b", "search", json!({}));
    assert_eq!(
        events(&decoded),
        vec![
            named_start(0, PartKind::ToolCall, "lookup", None),
            arguments(0, "{\"q\":"),
            named_start(1, PartKind::ToolCall, "search", None),
            arguments(1, "{}"),
            arguments(0, "1}"),
            end(0, first.clone()),
            end(1, second.clone()),
        ]
    );
    let response = decoded.outcome.expect("both calls finished");
    assert_eq!(response.choice, vec![first, second]);
}

#[test]
fn ended_custom_provider_and_opaque_items_take_their_terminal_backfill() {
    let decoded = write(|out| {
        out.open(
            0,
            custom("custom-1"),
            json!({"type":"custom_tool_call","call_id":"custom-1"}),
        )?;
        out.push(0, "raw")?;
        out.finish(0)?;
        out.open(
            1,
            Block::ProviderItem {
                provider: Some("item-issuer".to_owned()),
            },
            json!({"type":"future_item"}),
        )?;
        out.finish(1)?;
        out.open(
            2,
            Block::Opaque { replay: true },
            json!({"type":"opaque_item"}),
        )?;
        out.finish(2)?;
        for index in [0_usize, 1, 2] {
            out.edit(index, |item| item["backfill"] = json!(index))?;
        }
        Ok(())
    });
    let response = decoded.outcome.expect("the backfilled reply ends");
    assert_eq!(
        response.choice,
        vec![
            AssistantContent::CustomToolCall(custom_call("custom-1", "raw"))
                .with_native(json!({"type":"custom_tool_call","call_id":"custom-1","backfill":0})),
            AssistantContent::ProviderItem(ProviderItem {
                payload: CarriedPayload::new(
                    json!({"type":"future_item","backfill":1}),
                    Some("item-issuer".to_owned()),
                ),
            }),
            AssistantContent::Opaque(Opaque {
                item: json!({"type":"opaque_item","backfill":2}),
                replay: true,
            }),
        ]
    );
}

#[test]
fn index_order_without_source_coordinates_puts_unindexed_blocks_last() {
    let five = json!({"id":"five"});
    let two = json!({"id":"two"});
    let decoded = write(|out| {
        out.order_by_index();
        out.whole(5, Block::Text, five.clone(), "five")?;
        out.content(AssistantContent::Text(Text::new("unindexed")))?;
        out.whole(2, Block::Text, two.clone(), "two")
    });
    let response = decoded.outcome.expect("the ordered reply ends");
    assert_eq!(
        response.choice,
        vec![
            AssistantContent::Text(Text::new("two")).with_native(two),
            AssistantContent::Text(Text::new("five")).with_native(five),
            AssistantContent::Text(Text::new("unindexed")),
        ]
    );
}

#[test]
fn an_end_absorbed_before_its_start_takes_its_own_position() {
    let late = AssistantContent::Text(Text::new("late"));
    let mut turn = Turn::new(origin());
    Fold::<Completion>::absorb(&mut turn, &end(1, late.clone())).expect("the fold absorbs an end");
    assert_eq!(turn.choice, vec![None, Some(late.clone())]);
    assert_eq!(turn.snapshot(), vec![late.clone()]);
    let response = Fold::<Completion>::finish(turn, Finish::default(), reply())
        .expect("the fold finishes with the hole unfilled");
    assert_eq!(response.choice, vec![late]);
}

#[test]
fn carried_finalization_of_a_boundaryless_run_ends_that_run() {
    let text = carried_text("hi");
    let reasoning = carried_reasoning("plan");
    let decoded = write(|out| {
        let first = out.run(Block::Text, "hi")?;
        assert_eq!(first, AUTO_INDEX);
        out.finish_carried_text(first, text.clone())?;
        assert_eq!(out.lock().fold.run, None);
        let second = out.run(Block::Reasoning { redacted: false }, "plan")?;
        assert_eq!(second, AUTO_INDEX + 1);
        out.finish_carried_reasoning(second, reasoning.clone())?;
        assert_eq!(out.lock().fold.run, None);
        // A run left pointing at a finished index would fail to close here.
        let third = out.run(Block::Text, "after")?;
        assert_eq!(third, AUTO_INDEX + 2);
        out.end_run()
    });
    let response = decoded.outcome.expect("every run block closed");
    assert_eq!(
        response.choice,
        vec![
            AssistantContent::CarriedText(text),
            AssistantContent::CarriedReasoning(reasoning),
            AssistantContent::Text(Text::new("after")),
        ]
    );
}

#[test]
fn a_provider_terminal_refuses_arguments_that_overflowed_the_accumulation_bound() {
    let decoded = write(|out| {
        out.open(
            0,
            Block::Call {
                id: CallId::from_wire("call-1"),
                name: name("lookup"),
                namespace: None,
            },
            Value::Null,
        )?;
        out.push(0, "{}")?;
        out.push(0, &"x".repeat(MAX_TOOL_INPUT_BYTES))?;
        out.finish_call_at_bedrock_tool_use_terminal(0)
    });
    assert_eq!(
        events(&decoded),
        vec![
            named_start(0, PartKind::ToolCall, "lookup", None),
            StreamEvent::Arguments {
                part: Part::new(0),
                json: "{}".to_owned(),
            },
        ]
    );
    let error = decoded
        .outcome
        .expect_err("overflowed input is a typed malformed call");
    assert!(matches!(error, ProviderError::Relayed(_)));
    let reason = "tool-call input exceeded the accumulation bound";
    assert_eq!(
        error.report(),
        ErrorReport::new(
            ErrorKind::Response,
            format!("tool call `lookup` arrived with malformed JSON input: {reason}"),
        )
        .with_detail(ErrorDetail::MalformedToolInput(MalformedToolInput {
            name: "lookup".to_owned(),
            id: CallId::from_wire("call-1"),
            provider: Some(ProviderCallEvidence {
                call_id: ProviderCallId::new("call-1").expect("nonempty fixture id"),
                item_id: None,
            }),
            raw: "{}".to_owned(),
            error: reason.to_owned(),
        }))
    );
}

#[test]
fn paused_custody_refuses_open_nontext_and_incompatible_carried_text() {
    let issuer = crate::providers::anthropic::PROVIDER_NAME;
    let pause = FinishReason::Other("pause_turn".to_owned());
    let raw = json!({"type":"text","text":"Searching."});
    let plain = CarriedText::new(
        CarriedTextContent::Text("Searching.".to_owned()),
        CarriedPayload::new(raw.clone(), Some(issuer.to_owned())),
    );
    let foreign = CarriedText::new(
        CarriedTextContent::InspectionOnlyText("Searching.".to_owned()),
        CarriedPayload::new(raw.clone(), Some("other-issuer".to_owned())),
    );
    let turn = Turn::new(Origin::new("anthropic.messages", issuer, "claude-model"));
    let decoded = decode_with(turn, issuer, |reply| {
        let mut out = reply.out();
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(0, "thinking")?;
        refused(
            out.carry_text_for_anthropic_pause(0, &pause),
            "paused item 0 is not native text",
        );
        out.finish(0)?;
        for (index, carried) in [(1, &plain), (2, &foreign)] {
            out.open(index, Block::Text, raw.clone())?;
            out.push(index, "Searching.")?;
            out.finish_carried_text(index, carried.clone())?;
            refused(
                out.carry_text_for_anthropic_pause(index, &pause),
                &format!("paused item {index} has incompatible carried custody"),
            );
        }
        let _ = out.end(Finish {
            reason: Some(pause.clone()),
            ..Finish::default()
        });
        Ok(())
    });
    let response = decoded
        .outcome
        .expect("refused custody changes left the reply intact");
    assert_eq!(
        response.choice,
        vec![
            AssistantContent::Reasoning(Reasoning {
                text: "thinking".to_owned(),
                redacted: false,
                native: None,
            }),
            AssistantContent::CarriedText(plain),
            AssistantContent::CarriedText(foreign),
        ]
    );
}
