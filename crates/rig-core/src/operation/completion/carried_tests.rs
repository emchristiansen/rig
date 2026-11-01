use serde_json::{Value, json};

use super::{Block, CallFragment, Completion, Finish, Turn, events_of};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{AssistantContent, CallId, CustomToolCall, Origin, ProviderItem, ToolName};
use crate::streaming::{Item, PartKind, SourceOrder, StreamEvent, Transcript};
use crate::wire::{Fold, Out, Reply};

const RAW_INPUT: &str = " \n{\"patch\": [1, 2]}\t ";

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("a tool name")
}

fn origin() -> Origin {
    Origin::new("test.api", "turn-provider", "request-model")
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
    write: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::new(origin()), "transport-provider", |reply| {
        let mut out = reply.out();
        write(&mut out)?;
        let _ = out.end(Finish::default());
        Ok(())
    })
}

fn events(decoded: &Decoded<Completion>) -> Vec<StreamEvent> {
    let items = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect();
    Transcript::from_items(items)
        .expect("the writer emits a valid sequence")
        .events()
        .cloned()
        .collect()
}

fn relayed(events: &[StreamEvent]) -> Turn {
    let mut turn = Turn::relayed("relay");
    turn.set_origin(origin());
    for event in events {
        Fold::<Completion>::absorb(&mut turn, event).expect("absorb the emitted event");
    }
    turn
}

#[test]
fn custom_calls_keep_raw_input_and_the_complete_native_item() {
    let native = json!({
        "type": "custom_tool_call", "id": "ctc_output_1", "call_id": "call_custom_1",
        "name": "patch", "namespace": "functions", "input": RAW_INPUT,
        "future": {"nested": [true, null, {"exact": "kept"}]},
    });
    let decoded = write(|out| {
        let mut opening = native.clone();
        opening["input"] = json!("");
        out.open(
            7,
            Block::CustomCall {
                id: CallId::from_wire("call_custom_1"),
                name: name("patch"),
                namespace: Some("functions".to_owned()),
            },
            opening,
        )?;
        for fragment in [" \n", "{\"patch\":", " [1, 2]}", "\t "] {
            out.push(7, fragment)?;
        }
        out.edit(7, |item| item["input"] = json!(RAW_INPUT))?;
        out.finish(7)
    });
    let streamed = events(&decoded);
    let joined: String = streamed
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Input { input, .. } => Some(input.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(joined, RAW_INPUT);
    assert!(
        streamed
            .iter()
            .all(|event| !matches!(event, StreamEvent::Arguments { .. }))
    );
    assert!(matches!(
        streamed.first(),
        Some(StreamEvent::Start {
            kind: PartKind::CustomToolCall,
            name: Some(tool),
            namespace: Some(namespace),
            source_order: None,
            ..
        }) if tool == "patch" && namespace == "functions"
    ));
    let response = decoded.outcome.expect("the custom reply ends");
    assert_eq!(response.origin, origin());
    let [block] = response.choice.as_slice() else {
        panic!("one custom call should survive: {:?}", response.choice);
    };
    let AssistantContent::CustomToolCall(call) = block else {
        panic!("raw input must remain a custom call: {block:?}");
    };
    assert_eq!(call.id, CallId::from_wire("call_custom_1"));
    assert_eq!(call.input, RAW_INPUT);
    assert_eq!(call.namespace.as_deref(), Some("functions"));
    assert_eq!(block.native_item(), Some(&native));
    let canonical = AssistantContent::CustomToolCall(
        CustomToolCall::new(call.id.clone(), name("patch"), RAW_INPUT)
            .with_namespace(Some("functions".to_owned())),
    );
    assert_eq!(block.canonical(), canonical);
    let restated = Transcript::from_items(events_of(&response).expect("restate the custom call"))
        .expect("the restated sequence is valid");
    assert_eq!(
        restated.events().find_map(|event| match event {
            StreamEvent::Input { input, .. } => Some(input.as_str()),
            _ => None,
        }),
        Some(RAW_INPUT),
    );
    assert_eq!(
        restated.events().find_map(|event| match event {
            StreamEvent::End { content, .. } => Some(content),
            _ => None,
        }),
        Some(block),
    );
    let serialized = serde_json::to_value(&restated).expect("serialize the restated sequence");
    assert_eq!(
        Transcript::parse(serialized).expect("parse the closed sequence"),
        restated
    );
}

#[test]
fn named_function_open_preserves_absent_empty_and_named_namespaces() {
    for namespace in [None, Some(String::new()), Some("functions".to_owned())] {
        let decoded = write(|out| {
            out.open(
                0,
                Block::Call {
                    id: CallId::from_wire("named_function"),
                    name: name("add"),
                    namespace: namespace.clone(),
                },
                Value::Null,
            )?;
            out.push(0, "{\"x\":1}")?;
            out.finish(0)
        });
        let streamed = events(&decoded);
        assert!(matches!(
            streamed.first(),
            Some(StreamEvent::Start {
                kind: PartKind::ToolCall,
                name: Some(tool),
                namespace: held,
                source_order: None,
                ..
            }) if tool == "add" && held == &namespace
        ));
        assert!(matches!(
            streamed.get(1),
            Some(StreamEvent::Arguments { json, .. }) if json == "{\"x\":1}"
        ));
        let response = decoded.outcome.expect("the named function reply ends");
        let call = response
            .tool_calls()
            .next()
            .expect("the named call survives");
        assert_eq!(call.id, CallId::from_wire("named_function"));
        assert_eq!(call.function.namespace, namespace);
        assert_eq!(call.function.arguments_value(), json!({"x": 1}));
        assert!(matches!(
            response.choice.first().expect("the call").canonical(),
            AssistantContent::ToolCall(call) if call.function.namespace == namespace
        ));
        let restated = events_of(&response).expect("restate the named function");
        assert!(matches!(
            restated.first(),
            Some(Item::Event(StreamEvent::Start { namespace: held, .. })) if held == &namespace
        ));
    }
}

#[test]
fn function_qualification_preserves_absent_empty_and_named_namespaces() {
    for namespace in [None, Some(String::new()), Some("functions".to_owned())] {
        let decoded = write(|out| {
            out.fragment(
                Some(0),
                CallFragment {
                    id: Some("call_function"),
                    name: None,
                    arguments: None,
                },
            )?;
            out.qualify(0, namespace.clone())?;
            out.fragment(
                Some(0),
                CallFragment {
                    id: None,
                    name: Some("add"),
                    arguments: Some("{\"x\":1}"),
                },
            )?;
            out.finish(0)
        });
        let streamed = events(&decoded);
        assert!(matches!(
            streamed.first(),
            Some(StreamEvent::Start { kind: PartKind::ToolCall, namespace: held, .. })
                if held == &namespace
        ));
        assert!(
            streamed
                .iter()
                .all(|event| !matches!(event, StreamEvent::Input { .. }))
        );
        let response = decoded.outcome.expect("the function reply ends");
        let call = response
            .tool_calls()
            .next()
            .expect("the function call survives");
        assert_eq!(call.function.namespace, namespace);
        assert_eq!(call.function.arguments_value(), json!({"x": 1}));
        let canonical = response.choice.first().expect("the call").canonical();
        assert!(matches!(
            canonical,
            AssistantContent::ToolCall(call) if call.function.namespace == namespace
        ));
        let restated = events_of(&response).expect("restate the namespaced function");
        assert!(matches!(
            restated.first(),
            Some(Item::Event(StreamEvent::Start { namespace: held, .. })) if held == &namespace
        ));
    }
}

#[test]
fn incomplete_custom_calls_keep_raw_input_and_namespace_without_a_native() {
    for namespace in [None, Some(String::new()), Some("functions".to_owned())] {
        let decoded = write(|out| {
            out.open(
                0,
                Block::CustomCall {
                    id: CallId::from_wire("incomplete_custom"),
                    name: name("patch"),
                    namespace: namespace.clone(),
                },
                json!({"id": "ctc_incomplete", "future": [1, null]}),
            )?;
            out.push(0, RAW_INPUT)?;
            out.close(0)
        });
        let streamed = events(&decoded);
        let response = decoded.outcome.expect("the incomplete block is retained");
        let expected = AssistantContent::CustomToolCall(
            CustomToolCall::new(
                CallId::from_wire("incomplete_custom"),
                name("patch"),
                RAW_INPUT,
            )
            .with_namespace(namespace.clone()),
        );
        assert_eq!(response.choice, std::slice::from_ref(&expected));
        assert_eq!(
            response
                .choice
                .first()
                .and_then(AssistantContent::native_item),
            None
        );
        assert!(matches!(
            streamed.first(),
            Some(StreamEvent::Start { namespace: held, .. }) if held == &namespace
        ));
        let partial = relayed(&streamed).partial(None, &reply(), None);
        assert_eq!(partial.choice, [expected]);
        assert_eq!(partial.origin, origin());
        assert!(partial.aborted.is_some());
    }
}

#[test]
fn provider_items_keep_their_exact_issuer_item_and_turn_origin_on_every_path() {
    let item = json!({
        "type": "future_provider_item", "id": "pi_output_1",
        "future": {"array": [null, true, {"spacing": " \nkept\t "}]},
    });
    for provider in [None, Some(String::new()), Some("item-provider".to_owned())] {
        let expected = AssistantContent::ProviderItem(ProviderItem {
            payload: crate::message::CarriedPayload::new(item.clone(), provider.clone()),
        });
        let decoded = write(|out| {
            out.open(
                0,
                Block::ProviderItem {
                    provider: provider.clone(),
                },
                item.clone(),
            )?;
            out.finish(0)
        });
        let streamed = events(&decoded);
        assert!(matches!(
            streamed.as_slice(),
            [StreamEvent::Start {
                kind: PartKind::ProviderItem, name: None, namespace: None, source_order: None, ..
            }, StreamEvent::End { content, .. }] if content == &expected
        ));
        let response = decoded.outcome.expect("the provider-item reply ends");
        assert_eq!(response.choice, std::slice::from_ref(&expected));
        assert_eq!(response.origin, origin());
        assert_eq!(
            response.choice.first().expect("the item").canonical(),
            expected
        );
        let restated = Transcript::from_items(events_of(&response).expect("restate the item"))
            .expect("the item sequence is valid");
        assert_eq!(restated.events().cloned().collect::<Vec<_>>(), streamed);
        let encoded = serde_json::to_value(&restated).expect("serialize the provider item");
        assert_eq!(
            Transcript::parse(encoded).expect("read the item back"),
            restated
        );

        let turn = relayed(&streamed);
        let failure = ProviderError::Response("the reply was interrupted".to_owned());
        for reason in [None, Some(&failure)] {
            let partial = turn.partial(None, &reply(), reason);
            assert_eq!(partial.choice, std::slice::from_ref(&expected));
            assert_eq!(partial.origin, origin());
            assert_eq!(partial.error, reason.map(ToString::to_string));
        }
        let complete_partial = turn.partial(Some(&Finish::default()), &reply(), None);
        assert_eq!(complete_partial.choice, response.choice);
        assert_eq!(complete_partial.origin, response.origin);
    }
}

#[test]
fn an_incomplete_provider_item_remains_exact_instead_of_becoming_opaque() {
    let item = json!({"type": "future", "id": "pi_unfinished", "unknown": [true, null]});
    let expected = AssistantContent::ProviderItem(ProviderItem::new(item.clone(), "item-provider"));
    let decoded = write(|out| {
        out.open(
            0,
            Block::ProviderItem {
                provider: Some("item-provider".to_owned()),
            },
            item,
        )?;
        out.close(0)
    });
    let streamed = events(&decoded);
    let response = decoded
        .outcome
        .expect("the carried item survives an incomplete close");
    assert_eq!(response.choice, std::slice::from_ref(&expected));
    let partial = relayed(&streamed).partial(None, &reply(), None);
    assert_eq!(partial.choice, [expected]);
    assert_eq!(partial.origin, response.origin);
}

#[test]
fn source_coordinates_sort_indexed_slots_and_survive_nested_scopes_and_relay() {
    let expected = vec![
        AssistantContent::text("first-secondary"),
        AssistantContent::text("unindexed"),
        AssistantContent::text("later-secondary"),
        AssistantContent::text("outer-restored"),
        AssistantContent::text("later-primary"),
    ];
    for by_index in [false, true] {
        let decoded = write(|out| {
            if by_index {
                out.order_by_index();
            }
            out.in_source_order(SourceOrder::new(2, 1), |out| {
                out.whole(40, Block::Text, Value::Null, "later-primary")
            })?;
            out.whole(100, Block::Text, Value::Null, "unindexed")?;
            out.in_source_order(SourceOrder::new(1, 9), |out| {
                out.whole(50, Block::Text, Value::Null, "later-secondary")?;
                out.in_source_order(SourceOrder::new(1, 2), |out| {
                    out.whole(10, Block::Text, Value::Null, "first-secondary")
                })?;
                out.whole(30, Block::Text, Value::Null, "outer-restored")
            })
        });
        let streamed = events(&decoded);
        let orders: Vec<_> = streamed
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Start { source_order, .. } => Some(*source_order),
                _ => None,
            })
            .collect();
        assert_eq!(
            orders,
            [
                Some(SourceOrder::new(2, 1)),
                None,
                Some(SourceOrder::new(1, 9)),
                Some(SourceOrder::new(1, 2)),
                Some(SourceOrder::new(1, 9)),
            ]
        );
        let response = decoded.outcome.expect("the source-ordered reply ends");
        assert_eq!(response.choice, expected, "wire-index ordering {by_index}");
        let relay = relayed(&streamed);
        let partial = relay.partial(None, &reply(), None);
        assert_eq!(partial.choice, expected);
        assert_eq!(partial.origin, response.origin);
        let completed = Fold::<Completion>::finish(relay, Finish::default(), reply())
            .expect("the relayed fold completes");
        assert_eq!(completed.choice, response.choice);
        assert_eq!(completed.origin, response.origin);
    }
}
