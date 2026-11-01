use super::{PartKind, ReasoningAuthority, SequenceError, StreamEvent, Transcript};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{AssistantContent, CarriedPayload, CarriedReasoning, CarriedReasoningPart};
use crate::operation::{Block, Completion, Turn};
use crate::wire::Out;
use serde_json::{Value, json};

fn write(
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::relayed("authority"), "authority", |reply| {
        action(&mut reply.out())?;
        Err(ProviderError::Truncated)
    })
}

fn transcript(decoded: &Decoded<Completion>) -> Transcript {
    assert!(
        matches!(decoded.outcome, Err(ProviderError::Truncated)),
        "no provider terminal is fabricated"
    );
    let items = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect::<Vec<_>>();
    Transcript::parse(serde_json::to_value(items).expect("actual writer items"))
        .expect("same Part closes exactly once")
}

fn carried() -> CarriedReasoning {
    CarriedReasoning::new(vec![CarriedReasoningPart::Summary("thought".to_owned())],
        CarriedPayload::new(json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thought"}],"future":[null,7]}), Some("exact-issuer".to_owned())))
        .with_id(Some("rs_1".to_owned()))
}

#[test]
fn default_writer_end_bytes_remain_identical_for_plain_text_and_reasoning() {
    for (block, expected) in [
        (
            Block::Text,
            r#"{"event":"end","part":0,"content":{"type":"text","text":"plain"}}"#,
        ),
        (
            Block::Reasoning { redacted: false },
            r#"{"event":"end","part":0,"content":{"type":"reasoning","text":"plain"}}"#,
        ),
    ] {
        let decoded = write(|out| out.whole(7, block, Value::Null, "plain"));
        let events = transcript(&decoded);
        let first = events.events().next().expect("actual start").part();
        assert!(events.events().all(|event| event.part() == first));
        let end = events.events().last().expect("actual end");
        assert_eq!(serde_json::to_string(end).expect("old End bytes"), expected);
        assert!(matches!(
            end,
            StreamEvent::End {
                reasoning_authority: None,
                ..
            }
        ));
        assert_eq!(
            serde_json::from_str::<StreamEvent>(expected).expect("old event still reads"),
            *end
        );
    }
}

#[test]
fn carried_finalizers_keep_default_none_and_explicit_authority_on_the_original_part() {
    for authority in [
        None,
        Some(ReasoningAuthority::Partial),
        Some(ReasoningAuthority::ProviderRestated),
    ] {
        let expected = carried();
        let decoded = write(|out| {
            out.open(7, Block::Reasoning { redacted: false }, Value::Null)?;
            out.push(7, "thought")?;
            match authority {
                None => out.finish_carried_reasoning(7, expected.clone()),
                Some(ReasoningAuthority::Partial) => out.finish_carried_reasoning_with_authority(
                    7,
                    expected.clone(),
                    ReasoningAuthority::Partial,
                ),
                Some(ReasoningAuthority::ProviderRestated) => out
                    .finish_carried_reasoning_with_authority(
                        7,
                        expected.clone(),
                        ReasoningAuthority::ProviderRestated,
                    ),
            }
        });
        let events = transcript(&decoded);
        let start = events.events().next().expect("reasoning start");
        assert!(matches!(
            start,
            StreamEvent::Start {
                kind: PartKind::Reasoning,
                ..
            }
        ));
        assert!(events.events().all(|event| event.part() == start.part()));
        let end = events.events().last().expect("actual carried end");
        assert!(
            matches!(end, StreamEvent::End { content: AssistantContent::CarriedReasoning(content), reasoning_authority, .. } if content == &expected && *reasoning_authority == authority)
        );
        let encoded = serde_json::to_value(end).expect("serialized real end");
        match authority {
            None => assert!(encoded.get("reasoning_authority").is_none()),
            Some(ReasoningAuthority::Partial) => {
                assert_eq!(encoded["reasoning_authority"], "partial")
            }
            Some(ReasoningAuthority::ProviderRestated) => {
                assert_eq!(encoded["reasoning_authority"], "provider_restated")
            }
        }
    }
}

#[test]
fn transcript_rejects_authority_added_to_a_non_reasoning_end() {
    let decoded = write(|out| out.whole(7, Block::Text, Value::Null, "plain"));
    let events = transcript(&decoded);
    for authority in [
        ReasoningAuthority::Partial,
        ReasoningAuthority::ProviderRestated,
    ] {
        let mut encoded = serde_json::to_value(&events).expect("actual writer transcript");
        let end = encoded
            .as_array_mut()
            .expect("items")
            .last_mut()
            .expect("actual end");
        end["value"]["reasoning_authority"] =
            serde_json::to_value(authority).expect("typed authority");
        assert!(matches!(
            Transcript::parse(encoded),
            Err(SequenceError::WrongKind(_))
        ));
    }
}
