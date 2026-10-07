use super::{Block, Completion, Finish, Turn, events_of};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart,
    CarriedText, CarriedTextContent, Origin,
};
use crate::streaming::{PartKind, SourceOrder, StreamEvent, Transcript};
use crate::wire::{Fold, Out, Reply};
use serde_json::{Value, json};

fn origin() -> Origin {
    Origin::new("source.api", "turn-provider", "requested-model")
}

fn reply() -> Reply {
    Reply {
        provider: "transport-provider".to_owned(),
        raw: Value::Null,
        provider_request_id: None,
        response_headers: crate::completion::ProviderResponseHeaders::new(),
    }
}

fn parts(issuer: Option<String>) -> Vec<AssistantContent> {
    let mut text = CarriedText::new(
        CarriedTextContent::Text(" \nanswer\t ".to_owned()),
        CarriedPayload::new(json!({"type":"output_text", "text":" \nanswer\t ", "annotations":[], "future":{"keep":[true, null]}}), issuer.clone())
            .with_position(CarriedPosition::MessageContent { item_index: 12, part_index: 0 }),
    );
    text.item_id = Some("msg_output_only".to_owned());
    text.phase = Some(" final_answer ".to_owned());
    let mut opaque = CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(
            json!({"type":"future_part", "text":"raw only", "unknown":[1, null]}),
            issuer.clone(),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 12,
            part_index: 1,
        }),
    );
    opaque.item_id = text.item_id.clone();
    opaque.phase = text.phase.clone();
    let mut refusal = CarriedText::new(
        CarriedTextContent::Refusal(" no ".to_owned()),
        CarriedPayload::new(
            json!({"type":"refusal", "refusal":" no ", "future":true}),
            issuer.clone(),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 12,
            part_index: 2,
        }),
    );
    refusal.item_id = text.item_id.clone();
    refusal.phase = text.phase.clone();
    let reasoning = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Text {
                text: "think".to_owned(),
                signature: Some(" exact signature ".to_owned()),
            },
            CarriedReasoningPart::OpaqueSummary(
                CarriedPayload::new(json!({"type":"future_summary", "unknown":[null, 9]}), None)
                    .with_position(CarriedPosition::ReasoningSummary { part_index: 4 }),
            ),
            CarriedReasoningPart::Encrypted("never visible".to_owned()),
            CarriedReasoningPart::Summary("sum".to_owned()),
            CarriedReasoningPart::OpaqueContent(
                CarriedPayload::new(
                    json!({"type":"future_content", "nested":{"exact":" \t "}}),
                    Some("foreign/part".to_owned()),
                )
                .with_position(CarriedPosition::ReasoningContent { part_index: 8 }),
            ),
            CarriedReasoningPart::Redacted {
                data: "withheld".to_owned(),
            },
        ],
        CarriedPayload::new(
            json!({"type":"reasoning", "id":"rs_output", "unknown":{"exact":[true, null]}}),
            issuer,
        ),
    )
    .with_id(Some("rs_output".to_owned()));
    vec![
        AssistantContent::CarriedText(text),
        AssistantContent::CarriedText(opaque),
        AssistantContent::CarriedText(refusal),
        AssistantContent::CarriedReasoning(reasoning),
    ]
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

fn write_part(out: &mut Out<'_, Completion>, part: &AssistantContent) -> Result<(), ProviderError> {
    match part {
        AssistantContent::CarriedText(text) => out.carried_text(text.clone()),
        AssistantContent::CarriedReasoning(reasoning) => out.carried_reasoning(reasoning.clone()),
        other => panic!("the fixture contains a carried part, got {other:?}"),
    }
}

fn events(decoded: &Decoded<Completion>) -> Vec<StreamEvent> {
    let items = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect();
    Transcript::from_items(items)
        .expect("the actual writer sequence is valid")
        .events()
        .cloned()
        .collect()
}

fn relay(events: &[StreamEvent]) -> Turn {
    let mut turn = Turn::relayed("relay-label");
    turn.set_origin(origin());
    for event in events {
        Fold::<Completion>::absorb(&mut turn, event).expect("absorb actual writer event");
    }
    turn
}

#[test]
fn carried_writer_fold_restatement_and_relay_preserve_every_field_and_public_projection() {
    for issuer in [
        None,
        Some("turn-provider".to_owned()),
        Some("foreign/issuer".to_owned()),
    ] {
        let expected = parts(issuer);
        let decoded = write(|out| {
            for part in &expected {
                write_part(out, part)?;
            }
            Ok(())
        });
        let streamed = events(&decoded);
        let starts: Vec<_> = streamed
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Start {
                    kind,
                    name,
                    namespace,
                    ..
                } => {
                    assert!(name.is_none());
                    assert!(namespace.is_none());
                    Some(*kind)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            [
                PartKind::Text,
                PartKind::Text,
                PartKind::Text,
                PartKind::Reasoning
            ]
        );
        let text: String = streamed
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let reasoning: String = streamed
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Reasoning { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, " \nanswer\t  no ");
        assert_eq!(reasoning, "think\nsum\nwithheld");
        let response = decoded.outcome.expect("the writer reply ends");
        assert_eq!(response.choice, expected);
        assert_eq!(response.origin, origin());
        assert_eq!(response.text(), text);
        assert_eq!(response.reasoning(), reasoning);
        assert!(
            response.message_id.is_none(),
            "containing output item IDs never become response identity"
        );
        for block in &response.choice {
            assert_eq!(block.canonical(), *block);
            assert_eq!(
                block.clone().with_native(json!({"unrelated":"native"})),
                *block
            );
            assert!(block.native_item().is_none());
        }
        let restated =
            Transcript::from_items(events_of(&response).expect("restate carried response"))
                .expect("restatement is a real valid sequence");
        assert_eq!(restated.events().cloned().collect::<Vec<_>>(), streamed);
        let encoded = serde_json::to_value(&restated).expect("serialize carried sequence");
        assert_eq!(
            Transcript::parse(encoded).expect("parse exact carried ends"),
            restated
        );
        let complete_partial = relay(&streamed).partial(Some(&Finish::default()), &reply(), None);
        assert_eq!(complete_partial.choice, response.choice);
        assert_eq!(complete_partial.origin, response.origin);
        let complete = Fold::<Completion>::finish(relay(&streamed), Finish::default(), reply())
            .expect("finish relayed events");
        assert_eq!(complete.choice, response.choice);
        assert_eq!(complete.origin, response.origin);
        assert_eq!(complete.text(), text);
        assert_eq!(complete.reasoning(), reasoning);
    }
}

#[test]
fn interrupted_writer_partial_retains_carried_fields_for_failure_and_cancellation() {
    let expected = parts(Some("foreign/issuer".to_owned()));
    let failure = ProviderError::Response("wire interrupted".to_owned());
    let decoded = decode_with(Turn::new(origin()), "transport-provider", |reply| {
        let mut out = reply.out();
        for part in &expected {
            write_part(&mut out, part)?;
        }
        Err(failure.clone())
    });
    assert!(
        matches!(decoded.outcome, Err(ProviderError::Response(ref message)) if message == "wire interrupted")
    );
    let streamed = events(&decoded);
    let turn = relay(&streamed);
    for reason in [None, Some(&failure)] {
        let partial = turn.partial(None, &reply(), reason);
        assert_eq!(partial.choice, expected);
        assert_eq!(partial.origin, origin());
        assert_eq!(partial.text(), " \nanswer\t  no ");
        assert_eq!(partial.reasoning(), "think\nsum\nwithheld");
        assert_eq!(partial.error, reason.map(ToString::to_string));
        assert_eq!(partial.aborted.is_some(), reason.is_none());
        assert!(
            partial
                .choice
                .iter()
                .all(|part| part.native_item().is_none())
        );
    }
}

#[test]
fn incomplete_upstream_items_and_source_order_do_not_erase_nested_carried_data() {
    let expected = parts(None);
    let decoded = write(|out| {
        out.open(0, Block::Text, json!({"id":"unfinished_native"}))?;
        out.push(0, "unfinished")?;
        out.close(0)?;
        for (at, part) in expected.iter().enumerate().rev() {
            out.in_source_order(SourceOrder::new(12, at as u64), |out| write_part(out, part))?;
        }
        Ok(())
    });
    let streamed = events(&decoded);
    let response = decoded
        .outcome
        .expect("the incomplete ordinary item and carried items remain");
    let expected_choice: Vec<_> = std::iter::once(AssistantContent::text("unfinished"))
        .chain(expected)
        .collect();
    assert_eq!(response.choice, expected_choice);
    assert_eq!(response.origin, origin());
    assert!(
        response
            .choice
            .iter()
            .all(|part| part.native_item().is_none())
    );
    let relayed = relay(&streamed);
    assert_eq!(
        relayed.partial(None, &reply(), None).choice,
        expected_choice
    );
    let complete = Fold::<Completion>::finish(relayed, Finish::default(), reply())
        .expect("relay completes with the same source order");
    assert_eq!(complete.choice, response.choice);
    assert_eq!(complete.origin, response.origin);
}
