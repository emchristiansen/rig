//! G245: received bare signatures update known carriers on their existing Part.

use super::{Block, Completion, Turn};
use crate::completion::{CompletionRequest, CompletionResponse, FinishReason};
use crate::driver::{Decoded, Replying, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CarriedPayload, CarriedPosition, CarriedReasoningPart, CarriedTextContent,
    Message, Source, SourceLocation,
};
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::completion::{GenerateContent, PROVIDER_NAME};
use crate::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use crate::streaming::{Item, Part, SourceOrder, StreamEvent};
use crate::wire::{Decoder, Fold, Mode, Operation, Out, Wire, WireCitation, WireFrame};
use serde_json::{Value, json};

fn wire() -> GenerateContent {
    GeminiConfig::new("test-key").completion("gemini-2.5-flash")
}

fn part(thought: bool) -> Value {
    let mut part = json!({"text":"a","thoughtSignature":"sig1","future":{"keep":[3,null]}});
    if thought {
        part["thought"] = json!(true);
    }
    part
}

fn body(parts: Value, terminal: bool) -> Value {
    let mut candidate = json!({"content":{"role":"model","parts":parts}});
    if terminal {
        candidate["finishReason"] = json!("STOP");
    }
    json!({"candidates":[candidate]})
}

fn decode<'id>(
    decoder: &mut GenerateContentDecoder,
    reply: &Replying<'id, Completion>,
    parts: Value,
    terminal: bool,
) -> Result<(), ProviderError> {
    let chunk: GenerateContentChunk = serde_json::from_value(body(parts, terminal))
        .expect("the new fixture uses the real typed reader");
    let _ = decoder.decode(chunk, reply.out())?;
    Ok(())
}

fn write(
    action: impl for<'id> FnOnce(&Replying<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    let wire = wire();
    let fold = crate::test_utils::fold_for(&CompletionRequest::new("q"), &wire, Mode::Streaming);
    decode_with(fold, wire.describe().name, |reply| action(&reply))
}

fn queued(out: &Out<'_, Completion>) -> Vec<Item<StreamEvent>> {
    out.lock()
        .items
        .iter()
        .map(|item| item.as_ref().expect("actual successful item").clone())
        .collect()
}

fn take(out: &Out<'_, Completion>) -> Vec<Item<StreamEvent>> {
    let mut shared = out.lock();
    let mut items = Vec::new();
    while let Some(item) = shared.take() {
        items.push(item.expect("actual delivered item"));
    }
    items
}

fn index(out: &Out<'_, Completion>) -> usize {
    let shared = out.lock();
    assert_eq!(shared.fold.ended.len(), 1);
    *shared
        .fold
        .ended
        .keys()
        .next()
        .expect("one real received slot")
}

fn ended(items: &[Item<StreamEvent>]) -> (Part, &AssistantContent) {
    let ends: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content,
                reasoning_authority,
            }) => {
                assert_eq!(*reasoning_authority, None);
                Some((*part, content))
            }
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 1, "no replacement or second End");
    ends[0]
}

fn assert_signed(content: &AssistantContent, thought: bool, signature: &str) {
    let payload = match content {
        AssistantContent::CarriedText(text) if !thought => {
            let answer = text.signed_answer().expect("real signed answer");
            assert_eq!(answer.text, "a");
            assert_eq!(answer.signature.as_str(), signature);
            assert_eq!(text.item_id, None);
            assert_eq!(text.phase, None);
            &text.payload
        }
        AssistantContent::CarriedReasoning(reasoning) if thought => {
            assert_eq!(
                reasoning.content,
                vec![CarriedReasoningPart::Text {
                    text: "a".to_owned(),
                    signature: Some(signature.to_owned()),
                }]
            );
            assert_eq!(reasoning.id, None);
            &reasoning.payload
        }
        other => panic!("actual known signed carrier: {other:?}"),
    };
    let mut expected = part(thought);
    expected["thoughtSignature"] = json!(signature);
    assert_eq!(payload.item, expected);
    assert_eq!(payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(
        payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 0
        }
    );
    assert!(content.native_item().is_none());
}

fn assert_replay(response: &CompletionResponse, thought: bool, signature: &str) {
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    let wire = wire();
    let request = CompletionRequest::from(vec![
        Message::user("q"),
        response.message().expect("genuine completed turn"),
        Message::user("n"),
    ]);
    let prepared = Completion::prepare(request, &wire.describe()).expect("real history prepare");
    let encoded = wire
        .encode(prepared, Mode::Unary)
        .expect("matching issuer rebuild");
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{
            "text":"a", "thought":thought, "thoughtSignature":signature,
        }])
    );
}

#[test]
fn g245_same_chunk_answer_and_thought_correct_the_original_queued_end() {
    for thought in [false, true] {
        let result = write(|reply| {
            let mut decoder = wire().decoder_in(Mode::Streaming);
            decode(
                &mut decoder,
                reply,
                json!([part(thought), {"thoughtSignature":"sig2"}]),
                true,
            )?;
            let out = reply.out();
            let before = queued(&out);
            let (original, content) = ended(&before);
            assert_signed(content, thought, "sig2");
            let state_before = {
                let state = out.lock();
                (
                    state.fold.next_part,
                    state.fold.source_orders.clone(),
                    state.fold.ended.clone(),
                )
            };
            reply
                .out()
                .edit_gemini_signature(index(&out), "sig2", PROVIDER_NAME)?;
            assert_eq!(queued(&out), before, "idempotent update adds nothing");
            assert_eq!(ended(&queued(&out)).0, original);
            {
                let state = out.lock();
                assert_eq!(
                    (
                        state.fold.next_part,
                        state.fold.source_orders.clone(),
                        state.fold.ended.clone()
                    ),
                    state_before
                );
            }
            let _ = decoder.eof(reply.out())?;
            Ok(())
        });
        let response = result.outcome.expect("actual STOP");
        assert_signed(&response.choice[0], thought, "sig2");
        assert_replay(&response, thought, "sig2");
    }
}

#[test]
fn g245_split_chunks_correct_only_the_queued_end_metadata() {
    for thought in [false, true] {
        let result = write(|reply| {
            let mut decoder = wire().decoder_in(Mode::Streaming);
            decode(&mut decoder, reply, json!([part(thought)]), false)?;
            let out = reply.out();
            let before = queued(&out);
            let (original, content) = ended(&before);
            assert_signed(content, thought, "sig1");
            let state_before = {
                let state = out.lock();
                (
                    state.fold.next_part,
                    state.fold.source_orders.clone(),
                    state.fold.ended.clone(),
                )
            };
            decode(
                &mut decoder,
                reply,
                json!([{"thoughtSignature":"sig2"}]),
                true,
            )?;
            let after = queued(&out);
            assert_eq!(after.len(), before.len());
            assert_eq!(ended(&after).0, original);
            assert_signed(ended(&after).1, thought, "sig2");
            for (old, new) in before.iter().zip(&after) {
                if !matches!(old, Item::Event(StreamEvent::End { .. })) {
                    assert_eq!(old, new);
                }
            }
            {
                let state = out.lock();
                assert_eq!(
                    (
                        state.fold.next_part,
                        state.fold.source_orders.clone(),
                        state.fold.ended.clone()
                    ),
                    state_before
                );
            }
            let _ = decoder.eof(reply.out())?;
            Ok(())
        });
        assert_replay(&result.outcome.expect("actual terminal"), thought, "sig2");
    }
}

#[test]
fn g245_delivered_sig1_end_stays_historical_while_the_same_fold_and_replay_use_sig2() {
    for thought in [false, true] {
        let mut delivered = Vec::new();
        let result = write(|reply| {
            let mut decoder = wire().decoder_in(Mode::Streaming);
            decode(&mut decoder, reply, json!([part(thought)]), false)?;
            let out = reply.out();
            delivered = take(&out);
            let (original, content) = ended(&delivered);
            assert_signed(content, thought, "sig1");
            let state_before = {
                let state = out.lock();
                (
                    state.fold.next_part,
                    state.fold.source_orders.clone(),
                    state.fold.ended.clone(),
                )
            };
            decode(
                &mut decoder,
                reply,
                json!([{"thoughtSignature":"sig2"}]),
                true,
            )?;
            assert!(queued(&out).is_empty(), "no correction event");
            assert_signed(ended(&delivered).1, thought, "sig1");
            let slot = index(&out);
            {
                let state = out.lock();
                assert_eq!(
                    (
                        state.fold.next_part,
                        state.fold.source_orders.clone(),
                        state.fold.ended.clone()
                    ),
                    state_before
                );
                let position = state.fold.ended[&slot];
                assert_eq!(original.index(), position);
                assert_signed(
                    state.fold.choice[position]
                        .as_ref()
                        .expect("same folded slot"),
                    thought,
                    "sig2",
                );
            }
            let _ = decoder.eof(reply.out())?;
            Ok(())
        });
        let response = result.outcome.expect("actual terminal");
        assert_signed(&response.choice[0], thought, "sig2");
        assert_replay(&response, thought, "sig2");
        // Origin is cloned from the genuine terminal, never fabricated.
        let mut refold = Turn::new(response.origin.clone());
        for item in &delivered {
            let Item::Event(event) = item else {
                panic!("only actual events");
            };
            refold.absorb(event).expect("actual delivered event refold");
        }
        assert_signed(&refold.snapshot()[0], thought, "sig1");
    }
}

#[test]
fn g245_open_unsigned_runs_still_capture_bare_metadata_when_they_close() {
    for thought in [false, true] {
        let mut first = part(thought);
        first
            .as_object_mut()
            .expect("part")
            .shift_remove("thoughtSignature");
        if !thought {
            first.as_object_mut().expect("part").shift_remove("future");
        }
        let response = crate::test_utils::history::decode(
            &wire(),
            Mode::Streaming,
            [
                WireFrame::Text(body(json!([first]), false).to_string()),
                WireFrame::Text(body(json!([{"thoughtSignature":"sig2"}]), true).to_string()),
            ],
        )
        .expect("unsigned draft captures received signature");
        match &response.choice[0] {
            AssistantContent::CarriedText(text) if !thought => {
                assert_eq!(
                    text.signed_answer()
                        .expect("signed answer")
                        .signature
                        .as_str(),
                    "sig2"
                );
                assert_eq!(
                    text.payload.item,
                    json!({"text":"a","thoughtSignature":"sig2"})
                );
            }
            AssistantContent::CarriedReasoning(reasoning) if thought => {
                assert_eq!(
                    reasoning.content,
                    vec![CarriedReasoningPart::Text {
                        text: "a".into(),
                        signature: Some("sig2".into())
                    }]
                );
                assert_eq!(reasoning.payload.item["future"], json!({"keep":[3,null]}));
                assert_eq!(reasoning.payload.item["thoughtSignature"], "sig2");
            }
            other => panic!("actual open carrier: {other:?}"),
        }
        assert_replay(&response, thought, "sig2");
    }
}

#[test]
fn g245_empty_and_whitespace_bare_strings_update_both_views_exactly() {
    for thought in [false, true] {
        for signature in ["", " \t "] {
            let result = write(|reply| {
                let mut decoder = wire().decoder_in(Mode::Streaming);
                decode(&mut decoder, reply, json!([part(thought)]), false)?;
                decode(
                    &mut decoder,
                    reply,
                    json!([{"thoughtSignature":signature}]),
                    true,
                )?;
                assert_signed(ended(&queued(&reply.out())).1, thought, signature);
                let _ = decoder.eof(reply.out())?;
                Ok(())
            });
            assert_signed(
                &result.outcome.expect("actual STOP").choice[0],
                thought,
                signature,
            );
        }
    }
}

fn payload(content: &mut AssistantContent) -> &mut CarriedPayload {
    match content {
        AssistantContent::CarriedText(text) => &mut text.payload,
        AssistantContent::CarriedReasoning(reasoning) => &mut reasoning.payload,
        other => panic!("new guard fixture is a carried clone: {other:?}"),
    }
}

fn replace_queued(out: &Out<'_, Completion>, content: AssistantContent) {
    let mut state = out.lock();
    let slot = state
        .items
        .iter_mut()
        .find_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { content, .. })) => Some(content),
            _ => None,
        })
        .expect("one real queued End");
    *slot = content;
}

#[test]
fn g245_incompatible_guard_premises_keep_only_the_old_raw_edit() {
    for thought in [false, true] {
        let received = crate::test_utils::history::decode(
            &wire(),
            Mode::Streaming,
            [WireFrame::Text(
                body(json!([part(thought)]), true).to_string(),
            )],
        )
        .expect("actual received carrier")
        .choice
        .remove(0);
        let mut cases = Vec::new();
        for issuer in [None, Some("foreign.api".to_owned())] {
            let mut clone = received.clone();
            payload(&mut clone).provider = issuer;
            cases.push(clone);
        }
        for raw in [
            Value::Null,
            json!([]),
            json!({"text":"a"}),
            json!({"text":"a","thoughtSignature":null}),
            json!({"text":"a","thoughtSignature":17}),
        ] {
            let mut clone = received.clone();
            payload(&mut clone).item = raw;
            cases.push(clone);
        }
        if thought {
            for tail in [
                CarriedReasoningPart::Encrypted("ciphertext".into()),
                CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
                    json!({"future":"summary"}),
                    None,
                )),
                CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
                    json!({"future":"content"}),
                    None,
                )),
            ] {
                let mut clone = received.clone();
                let AssistantContent::CarriedReasoning(reasoning) = &mut clone else {
                    unreachable!()
                };
                reasoning.content.push(tail);
                cases.push(clone);
            }
            let mut clone = received.clone();
            let AssistantContent::CarriedReasoning(reasoning) = &mut clone else {
                unreachable!()
            };
            let CarriedReasoningPart::Text { signature, .. } = &mut reasoning.content[0] else {
                unreachable!()
            };
            *signature = None;
            cases.push(clone);
        } else {
            for other in [
                CarriedTextContent::Text("a".into()),
                CarriedTextContent::InspectionOnlyText("a".into()),
                CarriedTextContent::InspectionOnlyMedia("a".into()),
                CarriedTextContent::Opaque(String::new()),
            ] {
                let mut clone = received.clone();
                let AssistantContent::CarriedText(text) = &mut clone else {
                    unreachable!()
                };
                text.content = other;
                cases.push(clone);
            }
        }
        for fixture in cases {
            for taken in [false, true] {
                // These altered guard premises are explicitly constructed clones,
                // not claims about what the provider sent or original fixture parity.
                let mut expected = fixture.clone();
                if let Some(raw) = payload(&mut expected).item.as_object_mut() {
                    raw.insert("thoughtSignature".into(), json!("sig2"));
                }
                let result = write(|reply| {
                    let mut decoder = wire().decoder_in(Mode::Streaming);
                    decode(&mut decoder, reply, json!([part(thought)]), true)?;
                    let out = reply.out();
                    replace_queued(&out, fixture.clone());
                    let slot = index(&out);
                    let before = queued(&out);
                    let original = ended(&before).0;
                    if taken {
                        take(&out);
                    }
                    reply
                        .out()
                        .edit_gemini_signature(slot, "sig2", PROVIDER_NAME)?;
                    if taken {
                        assert!(queued(&out).is_empty());
                        let state = out.lock();
                        assert_eq!(
                            state.fold.choice[original.index()].as_ref(),
                            Some(&expected)
                        );
                    } else {
                        let after = queued(&out);
                        assert_eq!(after.len(), before.len());
                        assert_eq!(ended(&after), (original, &expected));
                    }
                    let _ = decoder.eof(reply.out())?;
                    Ok(())
                });
                assert_eq!(
                    result.outcome.expect("actual terminal").choice,
                    vec![expected]
                );
            }
        }
    }
}

#[test]
fn g245_native_call_and_inspection_media_keep_the_existing_raw_only_path() {
    for first in [
        json!({"functionCall":{"id":"call_1","name":"lookup","args":{"q":"x"}}}),
        json!({"inlineData":{"mimeType":"image/png","data":"AA=="}}),
    ] {
        let result = write(|reply| {
            let mut decoder = wire().decoder_in(Mode::Streaming);
            decode(&mut decoder, reply, json!([first.clone()]), false)?;
            let out = reply.out();
            let before = queued(&out);
            let original = ended(&before).0;
            decode(
                &mut decoder,
                reply,
                json!([{"thoughtSignature":"sig2"}]),
                true,
            )?;
            let after = queued(&out);
            assert_eq!(after.len(), before.len());
            assert_eq!(ended(&after).0, original);
            let mut expected = first.clone();
            expected["thoughtSignature"] = json!("sig2");
            match ended(&after).1 {
                AssistantContent::ToolCall(call) => {
                    assert_eq!(call.function.name.as_str(), "lookup");
                    assert_eq!(call.function.arguments_value(), json!({"q":"x"}));
                    assert_eq!(call.id.to_string(), "call_1");
                    assert_eq!(ended(&after).1.native_item(), Some(&expected));
                }
                AssistantContent::CarriedText(text) => {
                    assert!(matches!(
                        &text.content,
                        CarriedTextContent::InspectionOnlyMedia(_)
                    ));
                    assert_eq!(text.signed_answer(), None);
                    assert_eq!(text.payload.item, expected);
                }
                other => panic!("existing native/inspection kind: {other:?}"),
            }
            let _ = decoder.eof(reply.out())?;
            Ok(())
        });
        assert_eq!(result.outcome.expect("actual STOP").choice.len(), 1);
    }
}

#[test]
fn g245_missing_index_and_missing_ended_content_keep_existing_results() {
    let result = write(|reply| {
        let mut decoder = wire().decoder_in(Mode::Streaming);
        decode(&mut decoder, reply, json!([part(false)]), true)?;
        let out = reply.out();
        let before = queued(&out);
        let missing = index(&out) + 1;
        assert!(
            matches!(reply.out().edit_gemini_signature(missing, "sig2", PROVIDER_NAME),
            Err(ProviderError::Response(text)) if text == format!("the reply wrote to item {missing}, which is not open"))
        );
        assert_eq!(queued(&out), before);
        let slot = index(&out);
        let position = ended(&before).0.index();
        take(&out);
        // Explicit defensive guard premise: the mapping survives a missing slot.
        out.lock().fold.choice[position] = None;
        reply
            .out()
            .edit_gemini_signature(slot, "sig2", PROVIDER_NAME)?;
        assert!(queued(&out).is_empty());
        assert!(out.lock().fold.choice[position].is_none());
        let _ = decoder.eof(reply.out())?;
        Ok(())
    });
    assert!(
        result
            .outcome
            .expect("actual received STOP remains")
            .choice
            .is_empty()
    );
}

#[test]
fn g245_open_wrong_kind_and_nonobject_drafts_keep_generic_raw_behavior() {
    for raw in [json!({"future":[true,null]}), Value::Null] {
        let result = write(|reply| {
            let mut decoder = wire().decoder_in(Mode::Streaming);
            decode(&mut decoder, reply, json!([part(false)]), true)?;
            let mut out = reply.out();
            // Explicit open wrong-kind guard fixture; no invented Native/Origin.
            out.open(17, Block::Opaque { replay: false }, raw.clone())?;
            let before = queued(&out);
            out.edit_gemini_signature(17, "sig2", "different-selected-issuer")?;
            assert_eq!(queued(&out), before);
            let mut expected = raw.clone();
            if let Some(fields) = expected.as_object_mut() {
                fields.insert("thoughtSignature".into(), json!("sig2"));
            }
            assert_eq!(out.lock().fold.open[&17].item, expected);
            let _ = decoder.eof(reply.out())?;
            Ok(())
        });
        assert!(result.outcome.is_ok());
    }
}

#[test]
fn g245_signed_answer_keeps_citations_message_metadata_and_source_order_exact() {
    let result = write(|reply| {
        let mut decoder = wire().decoder_in(Mode::Streaming);
        reply.out().in_source_order(SourceOrder::new(4, 8), |_| {
            decode(&mut decoder, reply, json!([part(false)]), true)
        })?;
        let out = reply.out();
        let slot = index(&out);
        reply.out().cite(
            slot,
            WireCitation::new(
                None,
                vec![Source::new(SourceLocation::Url {
                    url: "https://example.com/received-context".to_owned(),
                })],
            ),
        );
        // Explicit metadata premise attached to the genuine received carrier.
        // It neither manufactures Origin/Native nor claims a Gemini item-ID input.
        let mut decorated = ended(&queued(&out)).1.clone();
        let AssistantContent::CarriedText(text) = &mut decorated else {
            panic!("signed answer");
        };
        text.item_id = Some("containing-message".to_owned());
        text.phase = Some("analysis".to_owned());
        assert!(!text.citations().is_empty());
        replace_queued(&out, decorated.clone());
        let before = queued(&out);
        let state_before = {
            let state = out.lock();
            (
                state.fold.next_part,
                state.fold.source_orders.clone(),
                state.fold.ended.clone(),
            )
        };
        assert!(!state_before.1.is_empty());
        reply
            .out()
            .edit_gemini_signature(slot, "sig2", PROVIDER_NAME)?;
        let mut expected = decorated;
        let AssistantContent::CarriedText(text) = &mut expected else {
            unreachable!()
        };
        let CarriedTextContent::GeminiSignedAnswer(answer) = &mut text.content else {
            unreachable!()
        };
        answer.signature = crate::message::GeminiThoughtSignature::new("sig2");
        text.payload.item["thoughtSignature"] = json!("sig2");
        let after = queued(&out);
        assert_eq!(after.len(), before.len());
        assert_eq!(ended(&after), (ended(&before).0, &expected));
        {
            let state = out.lock();
            assert_eq!(
                (
                    state.fold.next_part,
                    state.fold.source_orders.clone(),
                    state.fold.ended.clone()
                ),
                state_before
            );
        }
        let _ = decoder.eof(reply.out())?;
        Ok(())
    });
    let response = result.outcome.expect("actual received STOP");
    let AssistantContent::CarriedText(text) = &response.choice[0] else {
        panic!("same signed carrier");
    };
    assert_eq!(text.item_id.as_deref(), Some("containing-message"));
    assert_eq!(text.phase.as_deref(), Some("analysis"));
    assert_eq!(text.payload.item["future"], json!({"keep":[3,null]}));
    assert_eq!(
        text.signed_answer().expect("signed").signature.as_str(),
        "sig2"
    );
    assert!(!text.citations().is_empty());
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn g245_actual_model_and_reassembler_keep_sig2_terminal_with_explicit_delivery_limit() {
    use crate::driver::Model;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;
    for thought in [false, true] {
        for split in [false, true] {
            let frames = if split {
                vec![
                    body(json!([part(thought)]), false),
                    body(json!([{"thoughtSignature":"sig2"}]), true),
                ]
            } else {
                vec![body(
                    json!([part(thought), {"thoughtSignature":"sig2"}]),
                    true,
                )]
            };
            let sse = frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>();
            let model = Model::new(
                wire(),
                MockStreamingClient {
                    sse_bytes: bytes::Bytes::from(sse),
                },
            );
            let mut stream = model
                .stream(CompletionRequest::new("q"))
                .expect("actual stream opens");
            let mut delivered = Vec::new();
            while let Some(item) = stream.next().await {
                delivered.push(item.expect("actual successful item"));
            }
            assert_signed(
                ended(&delivered).1,
                thought,
                if split { "sig1" } else { "sig2" },
            );
            let response = stream.finish().await.expect("genuine STOP");
            assert_signed(&response.choice[0], thought, "sig2");
            let mut expected_part = part(thought);
            expected_part["thoughtSignature"] = json!("sig2");
            assert_eq!(
                response.raw["candidates"][0]["content"]["parts"],
                json!([expected_part])
            );
            assert_replay(&response, thought, "sig2");
        }
    }
}
