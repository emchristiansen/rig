use super::{Item, PartKind, SequenceError, StreamEvent, Transcript};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::operation::{Block, Completion, Turn};
use crate::wire::Out;
use serde_json::{Value, json};

fn write(
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::relayed("resume"), "resume", |reply| {
        action(&mut reply.out())?;
        Err(ProviderError::Truncated)
    })
}

fn items(result: &Decoded<Completion>) -> Vec<Item<StreamEvent>> {
    result
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect()
}

fn resumed(close: bool) -> Decoded<Completion> {
    write(|out| {
        out.open(7, Block::Text, Value::Null)?;
        out.push(7, "hello ")?;
        out.open(8, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(8, "because")?;
        out.resume_text(7)?;
        out.push(7, "world")?;
        if close {
            out.finish(7)?;
            out.finish(8)?;
        }
        Ok(())
    })
}

#[test]
fn resume_reactivates_the_same_text_part_without_a_second_start_or_duplicate_canonical_close() {
    let result = resumed(true);
    let transcript = Transcript::from_items(items(&result)).expect("actual writer grammar");
    let start = transcript
        .events()
        .find(|event| {
            matches!(
                event,
                StreamEvent::Start {
                    kind: PartKind::Text,
                    ..
                }
            )
        })
        .expect("one real text Start");
    let text = transcript
        .events()
        .filter(|event| event.part() == start.part())
        .collect::<Vec<_>>();
    assert_eq!(text.len(), 5);
    assert!(matches!(text[0], StreamEvent::Start { .. }));
    assert!(matches!(text[1], StreamEvent::Text { text, .. } if text == "hello "));
    assert!(matches!(text[2], StreamEvent::Resume { .. }));
    assert!(matches!(text[3], StreamEvent::Text { text, .. } if text == "world"));
    assert!(
        matches!(text[4], StreamEvent::End { content, .. } if *content == AssistantContent::text("hello world"))
    );
    assert_eq!(
        serde_json::to_string(text[2]).expect("real Resume bytes"),
        r#"{"event":"resume","part":0}"#
    );
    Transcript::parse(serde_json::to_value(&transcript).expect("actual writer events"))
        .expect("both real parts close once");
}

#[test]
fn transcript_refuses_resume_for_unstarted_closed_or_non_text_parts() {
    let encoded = serde_json::to_value(items(&resumed(true))).expect("actual writer events");
    let original = encoded.as_array().expect("items");
    let resume = original
        .iter()
        .find(|item| item["value"]["event"] == "resume")
        .expect("actual Resume")
        .clone();
    let mut unstarted = resume.clone();
    unstarted["value"]["part"] = json!(99);
    assert!(matches!(
        Transcript::parse_prefix(json!([unstarted])),
        Err(SequenceError::UnknownPart(_))
    ));
    let mut closed = original.clone();
    closed.push(resume.clone());
    assert!(matches!(
        Transcript::parse_prefix(Value::Array(closed)),
        Err(SequenceError::EndedTwice(_))
    ));
    let mut wrong_kind = original.clone();
    let position = wrong_kind
        .iter()
        .position(|item| item["value"]["event"] == "resume")
        .expect("Resume position");
    wrong_kind[position]["value"]["part"] = json!(1);
    assert!(matches!(
        Transcript::parse_prefix(Value::Array(wrong_kind)),
        Err(SequenceError::WrongKind(_))
    ));
}

#[test]
fn resumed_partial_stream_remains_unclosed_and_default_writer_emits_no_resume() {
    let partial =
        serde_json::to_value(items(&resumed(false))).expect("actual partial writer events");
    Transcript::parse_prefix(partial.clone()).expect("real reactivated prefix");
    assert!(matches!(
        Transcript::parse(partial),
        Err(SequenceError::Unclosed(_))
    ));
    let default = write(|out| {
        out.open(7, Block::Text, Value::Null)?;
        out.push(7, "hello ")?;
        out.whole(
            8,
            Block::Reasoning { redacted: false },
            Value::Null,
            "because",
        )?;
        out.push(7, "world")?;
        out.finish(7)
    });
    let transcript = Transcript::from_items(items(&default)).expect("default writer grammar");
    assert!(
        transcript
            .events()
            .all(|event| !matches!(event, StreamEvent::Resume { .. }))
    );
    assert_eq!(
        serde_json::to_string(transcript.events().next().expect("old Start bytes"))
            .expect("default serde"),
        r#"{"event":"start","part":0,"kind":"text"}"#
    );
}

#[test]
fn resume_hook_refuses_missing_unstarted_wrong_kind_and_closed_real_drafts() {
    let result = write(|out| {
        assert!(out.resume_text(7).is_err());
        out.open(7, Block::Text, Value::Null)?;
        assert!(out.resume_text(7).is_err());
        out.push(7, "text")?;
        out.whole(
            8,
            Block::Reasoning { redacted: false },
            Value::Null,
            "reason",
        )?;
        assert!(out.resume_text(8).is_err());
        out.open(9, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(9, "reason")?;
        assert!(out.resume_text(9).is_err());
        out.finish(7)?;
        assert!(out.resume_text(7).is_err());
        out.finish(9)
    });
    let transcript = Transcript::from_items(items(&result)).expect("refusals did not emit Resume");
    assert!(
        transcript
            .events()
            .all(|event| !matches!(event, StreamEvent::Resume { .. }))
    );
}
