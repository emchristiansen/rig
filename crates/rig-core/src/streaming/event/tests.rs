use super::*;
use crate::{completion::Usage, message::ReasoningContent, streaming::MintKind};

#[test]
fn every_event_round_trips_through_serde() {
    let events = vec![
        StreamEvent::BlockStart {
            id: BlockId::wire("msg_1"),
            kind: BlockKind::Message,
        },
        StreamEvent::BlockStart {
            id: BlockId::minted(MintKind::Text, 0),
            kind: BlockKind::Text {
                additional_params: AdditionalParams::from_entries([("k", serde_json::json!(1))]),
            },
        },
        StreamEvent::text(BlockId::minted(MintKind::Text, 0), "hi"),
        StreamEvent::BlockDelta {
            id: BlockId::wire("rs_1"),
            delta: Delta::Reasoning {
                text: "think".into(),
            },
        },
        StreamEvent::BlockStart {
            id: BlockId::wire("call_1"),
            kind: BlockKind::ToolCall,
        },
        StreamEvent::BlockDelta {
            id: BlockId::wire("call_1"),
            delta: Delta::ToolName { name: "add".into() },
        },
        StreamEvent::BlockDelta {
            id: BlockId::wire("call_1"),
            delta: Delta::ToolArguments {
                arguments: "{\"x\":1}".into(),
            },
        },
        StreamEvent::BlockEnd {
            id: BlockId::wire("call_1"),
            end: BlockClose::ToolCall(
                ToolCallEnd::new(UnparseableToolInput::Drop).with_call_id("c1"),
            ),
            block: None,
        },
        StreamEvent::BlockEnd {
            id: BlockId::wire("rs_1"),
            end: BlockClose::Reasoning {
                reasoning: Some(Reasoning {
                    provider: None,
                    id: Some("rs_1".into()),
                    content: vec![ReasoningContent::Text {
                        text: "think".into(),
                        signature: Some("sig".into()),
                    }],
                }),
                signature: None,
                wire_sent: true,
            },
            block: None,
        },
        StreamEvent::BlockEnd {
            id: BlockId::minted(MintKind::Text, 0),
            end: BlockClose::Text,
            block: None,
        },
        StreamEvent::Final(StreamFinal::new(
            "mock",
            Usage::default(),
            serde_json::json!({}),
        )),
        StreamEvent::Unknown(UnknownPayload::new(
            serde_json::json!({"type": "web_search_call"}),
        )),
    ];
    for event in events {
        let json = serde_json::to_string(&event).expect("serialize");
        let back: StreamEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event, "{json}");
    }
}

#[test]
fn unknown_event_wraps_every_json_shape_in_a_required_payload() {
    let values = [
        serde_json::json!({
            "event": "provider_future",
            "payload": {
                "event": "nested_future",
                "payload": [true, null],
            },
        }),
        serde_json::json!([1, "two", null]),
        serde_json::json!("text"),
        serde_json::json!(42),
        serde_json::json!(true),
        serde_json::Value::Null,
    ];

    for value in values {
        let event = StreamEvent::Unknown(UnknownPayload::new(value.clone()));
        let encoded = serde_json::to_value(&event).expect("serialize unknown event");
        assert_eq!(
            encoded,
            serde_json::json!({"event": "unknown_v2", "payload": value}),
        );
        assert_eq!(
            serde_json::from_value::<StreamEvent>(encoded).expect("deserialize unknown event"),
            event,
        );
    }
}

#[test]
fn unknown_event_requires_payload_but_accepts_explicit_null() {
    assert!(serde_json::from_str::<StreamEvent>(r#"{"event":"unknown_v2"}"#).is_err());

    let decoded = serde_json::from_str::<StreamEvent>(r#"{"event":"unknown_v2","payload":null}"#)
        .expect("explicit null is an unknown payload");
    assert_eq!(
        decoded,
        StreamEvent::Unknown(UnknownPayload::new(serde_json::Value::Null)),
    );
}

#[test]
fn legacy_unknown_event_encoding_is_rejected() {
    let legacy_encodings = [
        r#"{"event":"unknown","type":"provider_future"}"#,
        r#"{"event":"unknown","payload":null}"#,
        r#"{"event":"unknown","event":"provider_future","payload":{"nested":true}}"#,
    ];

    for encoded in legacy_encodings {
        assert!(
            serde_json::from_str::<StreamEvent>(encoded).is_err(),
            "legacy encoding unexpectedly decoded: {encoded}",
        );
    }
}

#[test]
fn standalone_unknown_payload_encoding_stays_transparent() {
    let values = [
        serde_json::json!({"event": "provider_future", "payload": {"nested": true}}),
        serde_json::json!([1, "two", null]),
        serde_json::json!("text"),
        serde_json::json!(42),
        serde_json::json!(true),
        serde_json::Value::Null,
    ];

    for value in values {
        let payload = UnknownPayload::new(value.clone());
        assert_eq!(
            serde_json::to_value(&payload).expect("serialize standalone payload"),
            value,
        );
        assert_eq!(
            serde_json::from_value::<UnknownPayload>(value)
                .expect("deserialize standalone payload"),
            payload,
        );
    }
}

#[test]
fn unchanged_event_variants_keep_their_wire_bytes() {
    let events = [
        (
            StreamEvent::BlockStart {
                id: BlockId::wire("msg_1"),
                kind: BlockKind::Message,
            },
            r#"{"event":"block_start","id":"wire:msg_1","kind":{"kind":"message"}}"#,
        ),
        (
            StreamEvent::text(BlockId::wire("text_1"), "hello"),
            r#"{"event":"block_delta","id":"wire:text_1","delta":{"delta":"text","text":"hello"}}"#,
        ),
        (
            StreamEvent::BlockEnd {
                id: BlockId::wire("text_1"),
                end: BlockClose::Text,
                block: None,
            },
            r#"{"event":"block_end","id":"wire:text_1","end":{"close":"text"}}"#,
        ),
        (
            StreamEvent::Final(StreamFinal::new(
                "mock",
                Usage::default(),
                serde_json::json!({}),
            )),
            r#"{"event":"final","usage":{},"finish_reason":null,"message_id":null,"response_id":null,"provider":"mock","model":null,"raw":{}}"#,
        ),
    ];

    for (event, expected) in events {
        assert_eq!(
            serde_json::to_string(&event).expect("serialize event"),
            expected
        );
    }
}

#[test]
fn empty_additional_params_are_a_decode_error() {
    let json =
        r#"{"event":"block_start","id":"wire:t","kind":{"kind":"text","additional_params":{}}}"#;
    assert!(serde_json::from_str::<StreamEvent>(json).is_err());
    let json = r#"{"event":"block_start","id":"wire:t","kind":{"kind":"text"}}"#;
    assert!(matches!(
        serde_json::from_str::<StreamEvent>(json).unwrap(),
        StreamEvent::BlockStart {
            kind: BlockKind::Text {
                additional_params: None
            },
            ..
        }
    ));
}
