//! Same-Part custody for native Anthropic paused text through the real writer.

use super::{Block, Completion, Finish};
use crate::completion::{CompletionRequest, FinishReason};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CarriedPosition, CarriedTextContent, Source, SourceLocation,
};
use crate::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use crate::providers::anthropic::wire::{AnthropicConfig, Messages, ZAI};
use crate::streaming::{Item, SourceOrder, StreamEvent};
use crate::wire::{Mode, Out, Wire, WireCitation};
use serde_json::{Value, json};

fn pause() -> FinishReason {
    FinishReason::Other("pause_turn".to_owned())
}
fn native() -> Messages {
    AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6)
}
fn raw() -> Value {
    json!({"type":"text","text":"Searching.","future":[null,3]})
}

fn write(
    wire: &Messages,
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    let fold =
        crate::test_utils::fold_for(&CompletionRequest::new("search"), wire, Mode::Streaming);
    decode_with(fold, wire.describe().name, |reply| {
        let mut out = reply.out();
        action(&mut out)?;
        let _ = out.end(Finish {
            reason: Some(pause()),
            ..Finish::default()
        });
        Ok(())
    })
}

fn successful_items(out: &Out<'_, Completion>) -> Vec<Item<StreamEvent>> {
    out.lock()
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect()
}

fn received_text(decoded: &Decoded<Completion>) -> &crate::message::CarriedText {
    let response = decoded.outcome.as_ref().expect("actual paused terminal");
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("one genuinely carried paused text");
    };
    assert_eq!(response.finish_reason(), Some(pause()));
    assert!(response.stop().is_failure());
    assert_eq!(
        response.origin.provider,
        crate::providers::anthropic::PROVIDER_NAME
    );
    assert_eq!(text.text(), "Searching.");
    assert!(
        matches!(&text.content, CarriedTextContent::InspectionOnlyText(value) if value == "Searching.")
    );
    assert_eq!(text.payload.item, raw());
    assert_eq!(
        text.payload.provider.as_deref(),
        Some(crate::providers::anthropic::PROVIDER_NAME)
    );
    text
}

#[test]
fn queued_closed_native_text_changes_only_custody_on_the_same_part() {
    let decoded = write(&native(), |out| {
        out.in_source_order(SourceOrder::new(3, 7), |out| {
            out.whole(41, Block::Text, raw(), "Searching.")
        })?;
        let before = successful_items(out);
        let (part_count, orders) = {
            let state = out.lock();
            (state.fold.next_part, state.fold.source_orders.clone())
        };
        assert!(out.carry_text_for_anthropic_pause(41, &pause())?);
        assert!(
            out.carry_text_for_anthropic_pause(41, &pause())?,
            "same custody is idempotent"
        );
        let after = successful_items(out);
        assert_eq!(after.len(), before.len());
        assert_eq!(&after[..2], &before[..2]);
        let (
            Item::Event(StreamEvent::End {
                part: before,
                content: AssistantContent::Text(_),
                ..
            }),
            Item::Event(StreamEvent::End {
                part: after,
                content: AssistantContent::CarriedText(_),
                ..
            }),
        ) = (&before[2], &after[2])
        else {
            panic!("only existing End custody changes");
        };
        assert_eq!(before, after);
        let state = out.lock();
        assert_eq!(state.fold.next_part, part_count);
        assert_eq!(state.fold.source_orders, orders);
        Ok(())
    });
    assert_eq!(
        received_text(&decoded).payload.position,
        CarriedPosition::MessageContent {
            item_index: 3,
            part_index: 7
        }
    );
}

#[test]
fn delivered_text_end_stays_text_while_the_later_fold_becomes_carried() {
    let decoded = write(&native(), |out| {
        out.in_source_order(SourceOrder::new(0, 4), |out| {
            out.whole(4, Block::Text, raw(), "Searching.")
        })?;
        let mut delivered = Vec::new();
        {
            let mut state = out.lock();
            while let Some(item) = state.take() {
                delivered.push(item.expect("actual delivered item"));
            }
        }
        assert!(matches!(
            &delivered[2],
            Item::Event(StreamEvent::End {
                content: AssistantContent::Text(_),
                ..
            })
        ));
        let orders = out.lock().fold.source_orders.clone();
        assert!(out.carry_text_for_anthropic_pause(4, &pause())?);
        // Named G151 observable divergence: old delivered End remains Text;
        // only the current terminal/fold custody is now carried.
        assert!(matches!(
            &delivered[2],
            Item::Event(StreamEvent::End {
                content: AssistantContent::Text(_),
                ..
            })
        ));
        let state = out.lock();
        assert!(state.items.is_empty(), "no second End or new Part");
        assert_eq!(state.fold.source_orders, orders);
        assert!(matches!(
            &state.fold.choice[0],
            Some(AssistantContent::CarriedText(_))
        ));
        Ok(())
    });
    assert_eq!(
        received_text(&decoded).payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 4
        }
    );
}

#[test]
fn an_open_native_text_uses_the_existing_finish_route_on_its_part() {
    let decoded = write(&native(), |out| {
        out.in_source_order(SourceOrder::new(0, 0), |out| {
            out.open(0, Block::Text, raw())
        })?;
        out.push(0, "Searching.")?;
        let before = successful_items(out);
        assert!(out.carry_text_for_anthropic_pause(0, &pause())?);
        let after = successful_items(out);
        assert_eq!(&after[..2], before.as_slice());
        assert_eq!(after.len(), 3);
        let Item::Event(first) = &after[0] else {
            panic!("actual first event");
        };
        assert!(
            after
                .iter()
                .all(|item| matches!(item, Item::Event(event) if event.part() == first.part()))
        );
        Ok(())
    });
    received_text(&decoded);
}

#[test]
fn absent_source_order_keeps_whole_payload_position_absent() {
    for taken in [false, true] {
        let decoded = write(&native(), |out| {
            out.whole(19, Block::Text, raw(), "Searching.")?;
            if taken {
                let mut state = out.lock();
                while let Some(item) = state.take() {
                    item.expect("real item");
                }
            }
            let orders = out.lock().fold.source_orders.clone();
            assert!(orders.is_empty());
            assert!(out.carry_text_for_anthropic_pause(19, &pause())?);
            assert_eq!(out.lock().fold.source_orders, orders);
            Ok(())
        });
        let text = received_text(&decoded);
        assert_eq!(text.payload.position, CarriedPosition::Whole);
        assert!(
            serde_json::to_value(&text.payload)
                .expect("carried payload")
                .get("position")
                .is_none()
        );
    }
}

#[test]
fn real_closed_and_open_citations_survive_paused_custody_exactly() {
    for closed in [false, true] {
        let mut original = None;
        let decoded = write(&native(), |out| {
            out.in_source_order(SourceOrder::new(0, 1), |out| {
                out.open(1, Block::Text, raw())
            })?;
            out.push(1, "Searching.")?;
            out.cite(
                1,
                WireCitation::new(
                    None,
                    vec![Source::new(SourceLocation::Url {
                        url: "https://example.com/search".to_owned(),
                    })],
                ),
            );
            if closed {
                out.finish(1)?;
                original = successful_items(out).iter().find_map(|item| match item {
                    Item::Event(StreamEvent::End {
                        content: AssistantContent::Text(text),
                        ..
                    }) => Some(text.citations().to_vec()),
                    _ => None,
                });
            }
            assert!(out.carry_text_for_anthropic_pause(1, &pause())?);
            Ok(())
        });
        let text = received_text(&decoded);
        assert_eq!(text.citations().len(), 1);
        assert_eq!(
            text.citations()[0].sources,
            vec![Source::new(SourceLocation::Url {
                url: "https://example.com/search".to_owned()
            })]
        );
        if let Some(original) = original {
            assert_eq!(text.citations(), original);
        }
        assert_eq!(
            text.payload.position,
            CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 1
            }
        );
    }
}

#[test]
fn wrong_reason_or_real_compatible_issuer_refuses_without_mutating_items() {
    for wire in [
        native(),
        AnthropicConfig::with_key(&ZAI, "test-key").completion("glm-5"),
    ] {
        let provider = wire.describe().name.to_owned();
        write(&wire, |out| {
            out.whole(0, Block::Text, raw(), "Searching.")?;
            for reason in [
                FinishReason::Stop,
                FinishReason::ToolCalls,
                FinishReason::Length,
                FinishReason::Other("other".to_owned()),
                pause(),
            ] {
                if provider == crate::providers::anthropic::PROVIDER_NAME && reason == pause() {
                    continue;
                }
                let before = successful_items(out);
                let error = out
                    .carry_text_for_anthropic_pause(0, &reason)
                    .expect_err("bounded guard");
                assert!(matches!(&error, ProviderError::Response(_)));
                assert_eq!(successful_items(out), before);
            }
            Ok(())
        });
    }
}

#[test]
fn stale_missing_wrong_shape_and_wrong_kind_native_evidence_refuse_without_mutation() {
    for case in 0..4 {
        write(&native(), |out| {
            match case {
                0 => {
                    out.whole(0, Block::Text, raw(), "Searching.")?;
                    let mut state = out.lock();
                    for item in &mut state.items {
                        if let Ok(Item::Event(StreamEvent::End {
                            content: AssistantContent::Text(text),
                            ..
                        })) = item
                        {
                            text.text.push_str(" edited");
                        }
                    }
                }
                1 => {
                    out.open(0, Block::Text, raw())?;
                    out.push(0, "Searching.")?;
                    out.close(0)?;
                }
                2 => out.whole(
                    0,
                    Block::Text,
                    json!({"type":"future","text":"Searching."}),
                    "Searching.",
                )?,
                _ => out.whole(
                    0,
                    Block::Opaque { replay: true },
                    json!({"type":"future"}),
                    "",
                )?,
            }
            let before = successful_items(out);
            let error = out
                .carry_text_for_anthropic_pause(0, &pause())
                .expect_err("actual evidence guard");
            assert!(matches!(&error, ProviderError::Response(_)));
            assert_eq!(successful_items(out), before);
            assert!(out.carry_text_for_anthropic_pause(999, &pause()).is_err());
            assert_eq!(successful_items(out), before);
            Ok(())
        });
    }
}

#[test]
fn blank_text_preserves_the_existing_open_and_closed_writer_views() {
    for text in ["", " "] {
        for closed in [false, true] {
            write(&native(), |out| {
                out.open(0, Block::Text, json!({"type":"text","text":text}))?;
                out.push(0, text)?;
                if closed {
                    out.finish(0)?;
                }
                let before = successful_items(out);
                let orders = out.lock().fold.source_orders.clone();
                assert!(!out.carry_text_for_anthropic_pause(0, &pause())?);
                assert_eq!(successful_items(out), before);
                assert_eq!(out.lock().fold.source_orders, orders);
                Ok(())
            });
        }
    }
}
