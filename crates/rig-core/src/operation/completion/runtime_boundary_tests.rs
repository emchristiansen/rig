//! G138/G149/G179: source fixtures exercise actual fold/writer boundaries.
use super::*;
use crate::completion::FinishReason;
use crate::driver::decode_with;
use crate::message::{
    CarriedPosition, CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, Source,
    SourceLocation,
};
use crate::wire::{SpanUnit, WireCitation, WireSpan};
use serde_json::{Value, json};

fn reply() -> Reply {
    Reply {
        provider: "transport-provider".to_owned(),
        raw: Value::Null,
        provider_request_id: None,
        response_headers: crate::completion::ProviderResponseHeaders::new(),
    }
}

#[test]
fn missing_end_partial_keeps_absence_aborted_and_actual_error_separate_from_terminal_validation() {
    let turn = Turn {
        wire: true,
        ..Turn::relayed("test")
    };
    let clean = turn.partial(None, &reply(), None);
    assert!(clean.error.is_none());
    assert!(clean.aborted.is_some());
    assert!(clean.finish_reason().is_none());
    assert!(clean.response_id().is_none());
    assert_eq!(clean.usage, Usage::default());
    let error = ProviderError::Provider("ordinary failure mentioning aborted".to_owned());
    let failed = turn.partial(None, &reply(), Some(&error));
    assert_eq!(failed.error.as_deref(), Some(error.to_string().as_str()));
    assert!(failed.aborted.is_none());
    assert!(failed.finish_reason().is_none());
    let genuine = turn.partial(Some(&Finish::default()), &reply(), None);
    assert_eq!(
        genuine.error.as_deref(),
        Some("the provider ended the reply without a finish reason")
    );
    assert!(genuine.aborted.is_none());
}

#[test]
fn genuine_length_terminal_keeps_g48b_unfinished_call_failure_before_length_exemption() {
    let turn = Turn {
        wire: true,
        ..Turn::relayed("test")
    };
    let decoded = decode_with(turn, "transport-provider", |reply| {
        let mut out = reply.out();
        out.open(
            0,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: crate::message::ToolName::new("add").expect("fixture name"),
                namespace: None,
            },
            json!({"type":"function_call","id":"call_1"}),
        )?;
        out.push(0, "{\"x\":1}")?;
        let _ = out.end(Finish {
            reason: Some(FinishReason::Length),
            ..Finish::default()
        });
        Ok(())
    });
    let response = decoded.outcome.expect("genuine terminal response");
    assert_eq!(
        response.error.as_deref(),
        Some("the provider ended the reply with a tool call it never finished")
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    assert!(response.stop().is_failure());
    assert!(
        response
            .choice
            .iter()
            .all(|part| part.native_item().is_none())
    );
}

#[test]
fn only_received_end_stamps_unknown_carried_text_and_never_the_delivered_end() {
    let raw = json!({"type":"output_text","text":"answer","annotations":[],"unknown":[null,7]});
    let mut unknown = CarriedText::new(
        CarriedTextContent::Text("answer".to_owned()),
        CarriedPayload::new(raw.clone(), None).with_position(CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 2,
        }),
    );
    unknown.item_id = Some("msg_citations".to_owned());
    unknown.phase = Some("final_answer".to_owned());
    let mut foreign = unknown.clone();
    foreign.payload.provider = Some("foreign/issuer".to_owned());
    let mut empty = unknown.clone();
    empty.payload.provider = Some(String::new());
    let mut delivered = Vec::new();
    let decoded = decode_with(
        Turn::relayed("turn-provider"),
        "transport-provider",
        |reply_writer| {
            let mut out = reply_writer.out();
            for text in [unknown.clone(), foreign.clone(), empty.clone()] {
                out.carried_text(text)?;
            }
            {
                let mut shared = out.lock();
                while let Some(item) = shared.take() {
                    delivered.push(item.expect("received writer event"));
                }
            }
            out.stamp_carried_issuer("received-terminal/issuer");
            let partial = out.lock().fold.partial(None, &reply(), None);
            assert_eq!(
                partial.choice,
                [
                    AssistantContent::CarriedText(unknown.clone()),
                    AssistantContent::CarriedText(foreign.clone()),
                    AssistantContent::CarriedText(empty.clone())
                ]
            );
            let _ = out.end(Finish {
                reason: Some(FinishReason::Stop),
                ..Finish::default()
            });
            Ok(())
        },
    );
    let ended: Vec<_> = delivered
        .iter()
        .filter_map(|item| match item {
            crate::streaming::Item::Event(crate::streaming::StreamEvent::End {
                content, ..
            }) => Some(content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        ended,
        [
            AssistantContent::CarriedText(unknown.clone()),
            AssistantContent::CarriedText(foreign.clone()),
            AssistantContent::CarriedText(empty.clone())
        ]
    );
    let response = decoded.outcome.expect("genuine received end");
    unknown.payload.provider = Some("received-terminal/issuer".to_owned());
    assert_eq!(
        response.choice,
        [
            AssistantContent::CarriedText(unknown),
            AssistantContent::CarriedText(foreign),
            AssistantContent::CarriedText(empty)
        ]
    );
    assert_eq!(response.origin.provider, "turn-provider");
    assert!(
        response
            .choice
            .iter()
            .all(|part| part.native_item().is_none())
    );
    let mut unissued = Turn::relayed("turn-provider");
    unissued.choice = response.choice.iter().cloned().map(Some).collect();
    if let Some(AssistantContent::CarriedText(text)) = &mut unissued.choice[0] {
        text.payload.provider = None;
    }
    let result = unissued.response(
        Finish {
            reason: Some(FinishReason::Stop),
            ..Finish::default()
        },
        reply(),
    );
    let AssistantContent::CarriedText(text) = &result.choice[0] else {
        panic!("carried text remains")
    };
    assert_eq!(
        text.payload.provider, None,
        "no explicit issuer means no stamp, even at an end"
    );
}

#[test]
fn opaque_closed_carried_text_keeps_raw_and_refuses_canonical_citation_attachment() {
    let raw = json!({"type":"future_part","unknown":[null,7]});
    let carried = CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(raw.clone(), None),
    );
    let decoded = decode_with(Turn::relayed("test"), "test", |reply| {
        let mut out = reply.out();
        out.open(0, Block::Text, Value::Null)?;
        out.finish_carried_text(0, carried.clone())?;
        out.cite(
            0,
            WireCitation::new(
                None,
                vec![Source::new(SourceLocation::Url {
                    url: "https://example.com/final".to_owned(),
                })],
            ),
        );
        let _ = out.end(Finish {
            reason: Some(FinishReason::Stop),
            ..Finish::default()
        });
        Ok(())
    });
    let response = decoded.outcome.expect("writer ended");
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("one opaque carried part")
    };
    assert!(text.citations().is_empty());
    assert_eq!(text.payload.item, raw);
}

const ANSWER: &str = "Spain won \u{2014} 2\u{2013}1. Nico Williams scored first. Oyarzabal won it.";
fn web(url: &str, title: &str, confidence: f32) -> Source {
    Source::new(SourceLocation::Url {
        url: url.to_owned(),
    })
    .title(title)
    .confidence(confidence)
}
fn supports() -> Vec<WireCitation> {
    vec![
        WireCitation::new(
            Some(WireSpan::new(21, 48, SpanUnit::Bytes).quoted("Nico Williams scored first.")),
            vec![web("https://example.com/final", "example.com", 0.75)],
        ),
        WireCitation::new(
            Some(WireSpan::new(49, 66, SpanUnit::Bytes).quoted("Oyarzabal won it.")),
            vec![
                web("https://example.com/final", "example.com", 0.5),
                web("https://example.org/report", "example.org", 0.25),
            ],
        ),
    ]
}
fn assert_supports(text: &CarriedText) {
    assert_eq!(text.citations().len(), 2);
    assert_eq!(
        text.citations()[0].span.as_ref().expect("span").range(),
        21..48
    );
    assert_eq!(
        text.citations()[1].span.as_ref().expect("span").range(),
        49..66
    );
    assert_eq!(
        text.citations()[0].sources,
        [web("https://example.com/final", "example.com", 0.75)]
    );
    assert_eq!(
        text.citations()[1].sources,
        [
            web("https://example.com/final", "example.com", 0.5),
            web("https://example.org/report", "example.org", 0.25)
        ]
    );
}

#[test]
fn late_citations_attach_to_queued_and_consumed_known_carried_text_without_raw_signature_edits() {
    let raw = json!({"text":ANSWER,"thoughtSignature":"c2ln"});
    let carried = CarriedText::new(
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
            ANSWER,
            GeminiThoughtSignature::new("c2ln"),
        )),
        CarriedPayload::new(raw.clone(), Some("gcp.gemini".to_owned())),
    );
    for take_end in [false, true] {
        let decoded = decode_with(Turn::relayed("test"), "test", |reply| {
            let mut out = reply.out();
            out.open(0, Block::Text, Value::Null)?;
            out.push(0, ANSWER)?;
            out.finish_carried_text(0, carried.clone())?;
            if take_end {
                let mut shared = out.lock();
                while let Some(item) = shared.take() {
                    item.expect("received event");
                }
            }
            for citation in supports() {
                out.cite(0, citation);
            }
            out.set_citations(0, supports());
            let _ = out.end(Finish {
                reason: Some(FinishReason::Stop),
                ..Finish::default()
            });
            Ok(())
        });
        let response = decoded.outcome.expect("writer ended");
        let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
            panic!("known signed answer")
        };
        assert_supports(text);
        assert_eq!(text.payload.item, raw);
        assert_eq!(
            text.signed_answer()
                .expect("signed answer")
                .signature
                .as_str(),
            "c2ln"
        );
    }
}
