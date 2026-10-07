use super::*;
use serde_json::json;

fn text_end(part: u32, text: &str) -> serde_json::Value {
    json!({"item": "event", "value": {"event": "end", "part": part, "content": {"type": "text", "text": text}}})
}

#[test]
fn a_writer_sequence_round_trips_through_serde() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": "hi"}},
        {"item": "unknown", "value": {"type": "web_search_call"}},
        text_end(0, "hi"),
    ]);
    let transcript = Transcript::parse(value.clone()).expect("a writer sequence");
    assert_eq!(transcript.len(), 4);
    assert_eq!(
        serde_json::to_value(&transcript).expect("serializes"),
        value
    );
    let back: Transcript = serde_json::from_value(value).expect("deserializes");
    assert_eq!(back, transcript);
}

#[test]
fn an_event_for_a_part_that_never_started_is_refused() {
    let value = json!([{"item": "event", "value": {"event": "text", "part": 0, "text": "hi"}}]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::UnknownPart(0)));
}

#[test]
fn a_start_that_reuses_a_position_is_refused() {
    let start = json!({"item": "event", "value": {"event": "start", "part": 1, "kind": "text"}});
    let value = json!([start.clone(), start]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::UnknownPart(1)));
}

#[test]
fn a_part_that_grows_after_its_end_is_refused() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        text_end(0, ""),
        {"item": "event", "value": {"event": "text", "part": 0, "text": "late"}},
    ]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::EndedTwice(2)));
}

#[test]
fn a_fragment_of_another_kind_is_refused() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "reasoning", "part": 0, "text": "no"}},
    ]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::WrongKind(1)));
}

#[test]
fn an_open_part_is_a_prefix_but_not_a_transcript() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": "cut"}},
    ]);
    assert_eq!(
        Transcript::parse(value.clone()),
        Err(SequenceError::Unclosed(0))
    );
    let prefix = Transcript::parse_prefix(value).expect("a stream cut short");
    assert_eq!(prefix.events().count(), 2);
}

/// A tool call's start names its tool, and no other part's start names one.
#[test]
fn only_a_tool_call_start_names_a_tool() {
    let start = |kind: &str, name: Option<&str>| {
        let mut value = json!({"event": "start", "part": 0, "kind": kind});
        if let Some(name) = name {
            value["name"] = json!(name);
        }
        json!([{"item": "event", "value": value}])
    };
    assert!(Transcript::parse_prefix(start("tool_call", Some("add"))).is_ok());
    assert_eq!(
        Transcript::parse_prefix(start("tool_call", None)).err(),
        Some(SequenceError::WrongKind(0))
    );
    assert_eq!(
        Transcript::parse_prefix(start("text", Some("add"))).err(),
        Some(SequenceError::WrongKind(0))
    );
}

#[test]
fn a_part_reads_back_from_its_serialized_form() {
    let part: Part = serde_json::from_value(json!(7)).expect("a part");
    assert_eq!(part.index(), 7);
    assert_eq!(serde_json::to_value(part).expect("serializes"), json!(7));
}

#[test]
fn a_tool_call_start_names_its_tool_and_no_other_start_does() {
    let call = json!({"event": "start", "part": 0, "kind": "tool_call", "name": "add"});
    let text = json!({"event": "start", "part": 1, "kind": "text"});
    let event: StreamEvent = serde_json::from_value(call.clone()).expect("a call start");
    assert!(matches!(
        &event,
        StreamEvent::Start {
            kind: PartKind::ToolCall,
            name: Some(name),
            namespace: None,
            source_order: None,
            ..
        } if name.as_str() == "add"
    ));
    assert_eq!(serde_json::to_value(&event).expect("serializes"), call);
    let event: StreamEvent = serde_json::from_value(text.clone()).expect("a text start");
    assert!(matches!(
        event,
        StreamEvent::Start {
            name: None,
            namespace: None,
            source_order: None,
            ..
        }
    ));
    assert_eq!(serde_json::to_value(&event).expect("serializes"), text);
}

#[test]
fn an_item_reads_back_without_its_order_being_checked() {
    let late = json!({"item": "event", "value": {"event": "text", "part": 3, "text": "late"}});
    assert_eq!(
        Transcript::parse(json!([late.clone()])),
        Err(SequenceError::UnknownPart(0))
    );
    let item: Item<StreamEvent> = serde_json::from_value(late.clone()).expect("an item");
    assert_eq!(serde_json::to_value(&item).expect("serializes"), late);
}

#[test]
fn a_misshapen_item_is_refused() {
    let value = json!({"item": "event", "value": {"event": "text", "part": -1, "text": "x"}});
    assert!(serde_json::from_value::<Item<StreamEvent>>(value).is_err());
}

/// An image part reads back, checks against its kind, and hands its items
/// back in order; each event names its variant without its payload.
#[test]
fn an_image_part_reads_back_and_names_its_events() {
    use crate::message::{DocumentSourceKind, Image, ImageMediaType};

    let image = AssistantContent::Image(Image {
        data: DocumentSourceKind::base64("aGk="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    });
    let end = json!({"item": "event", "value": {
        "event": "end",
        "part": 1,
        "content": serde_json::to_value(&image).expect("an image serializes"),
    }});
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "reasoning"}},
        {"item": "event", "value": {"event": "reasoning", "part": 0, "text": "hm"}},
        {"item": "event", "value": {"event": "start", "part": 1, "kind": "image"}},
        end,
    ]);
    let transcript = Transcript::parse_prefix(value.clone()).expect("an image part");
    assert_eq!(
        transcript
            .events()
            .map(StreamEvent::name)
            .collect::<Vec<_>>(),
        ["Start", "Reasoning", "Start", "End"]
    );
    assert!(matches!(
        transcript.events().last(),
        Some(StreamEvent::End {
            content: AssistantContent::Image(_),
            ..
        })
    ));
    let items: Vec<Item<StreamEvent>> = transcript.into();
    assert_eq!(serde_json::to_value(&items).expect("serializes"), value);
}

#[test]
fn custom_input_and_provider_items_round_trip_with_an_empty_namespace() {
    use crate::message::{CallId, CustomToolCall, ProviderItem};

    let input = " \n{\"patch\": [1, 2]}\t ";
    let custom = CustomToolCall::new(
        CallId::from_wire("custom_1"),
        ToolName::new("patch").expect("tool name"),
        input,
    )
    .with_namespace(Some(String::new()));
    let carried = ProviderItem::new(
        json!({"type": "future", "id": "item_1", "unknown": [true, null]}),
        "item-provider",
    );
    let items = vec![
        Item::Event(StreamEvent::Start {
            part: Part::new(0),
            kind: PartKind::CustomToolCall,
            name: Some(custom.name.clone()),
            namespace: Some(String::new()),
            source_order: Some(SourceOrder::new(3, 1)),
            tool_slot_state: None,
        }),
        Item::Event(StreamEvent::Input {
            part: Part::new(0),
            input: input.to_owned(),
        }),
        Item::Event(StreamEvent::End {
            part: Part::new(0),
            content: AssistantContent::CustomToolCall(custom),
            reasoning_authority: None,
        }),
        Item::Event(StreamEvent::Start {
            part: Part::new(1),
            kind: PartKind::ProviderItem,
            name: None,
            namespace: None,
            source_order: None,
            tool_slot_state: None,
        }),
        Item::Event(StreamEvent::End {
            part: Part::new(1),
            content: AssistantContent::ProviderItem(carried),
            reasoning_authority: None,
        }),
    ];
    let encoded = serde_json::to_value(&items).expect("serialize the carried sequence");
    let parsed = Transcript::parse(encoded.clone()).expect("parse the carried sequence");
    assert_eq!(parsed.items(), items);
    assert_eq!(
        parsed.events().map(StreamEvent::name).collect::<Vec<_>>(),
        ["Start", "Input", "End", "Start", "End"],
    );
    assert_eq!(
        serde_json::to_value(&parsed).expect("serialize again"),
        encoded
    );
    assert!(matches!(
        parsed.events().next(),
        Some(StreamEvent::Start {
            namespace: Some(namespace),
            source_order: Some(SourceOrder { primary: 3, secondary: 1 }),
            ..
        }) if namespace.is_empty()
    ));
}

#[test]
fn custom_input_and_function_arguments_cannot_cross_part_kinds() {
    for (kind, fragment) in [
        (
            "custom_tool_call",
            json!({"event": "arguments", "part": 0, "json": "{}"}),
        ),
        (
            "tool_call",
            json!({"event": "input", "part": 0, "input": "{}"}),
        ),
        (
            "provider_item",
            json!({"event": "input", "part": 0, "input": "raw"}),
        ),
        (
            "provider_item",
            json!({"event": "arguments", "part": 0, "json": "{}"}),
        ),
    ] {
        let mut start = json!({"event": "start", "part": 0, "kind": kind});
        if kind != "provider_item" {
            start["name"] = json!("patch");
        }
        assert_eq!(
            Transcript::parse(json!([
                {"item": "event", "value": start},
                {"item": "event", "value": fragment},
            ])),
            Err(SequenceError::WrongKind(1)),
            "{kind}",
        );
    }
}

#[test]
fn non_call_parts_cannot_state_a_namespace() {
    for kind in ["text", "reasoning", "image", "opaque", "provider_item"] {
        for namespace in ["", "functions"] {
            assert_eq!(
                Transcript::parse_prefix(json!([{
                    "item": "event",
                    "value": {
                        "event": "start", "part": 0, "kind": kind,
                        "namespace": namespace,
                    },
                }])),
                Err(SequenceError::WrongKind(0)),
                "{kind} namespace {namespace:?}",
            );
        }
    }
}

#[test]
fn a_custom_call_start_requires_its_tool_name() {
    let unnamed = json!([{
        "item": "event",
        "value": {"event": "start", "part": 0, "kind": "custom_tool_call"},
    }]);
    assert_eq!(
        Transcript::parse_prefix(unnamed),
        Err(SequenceError::WrongKind(0))
    );
}

#[test]
fn unknown_items_keep_every_payload_shape_and_require_a_value() {
    for value in [
        json!({"event": "future", "payload": {"nested": [true, null]}}),
        json!([1, "two", null]),
        json!("text"),
        json!(42),
        json!(true),
        serde_json::Value::Null,
    ] {
        let item = Item::<StreamEvent>::Unknown(UnknownPayload::new(value.clone()));
        let encoded = json!({"item": "unknown", "value": value});
        assert_eq!(
            serde_json::to_value(&item).expect("serialize the unknown item"),
            encoded
        );
        assert_eq!(
            serde_json::from_value::<Item<StreamEvent>>(encoded)
                .expect("deserialize the exact payload"),
            item,
        );
        let payload = UnknownPayload::new(value.clone());
        assert_eq!(
            serde_json::to_value(&payload).expect("serialize the payload"),
            value
        );
        assert_eq!(
            serde_json::from_value::<UnknownPayload>(value).expect("read the payload"),
            payload
        );
    }
    assert!(serde_json::from_value::<Item<StreamEvent>>(json!({"item": "unknown"})).is_err());
}
