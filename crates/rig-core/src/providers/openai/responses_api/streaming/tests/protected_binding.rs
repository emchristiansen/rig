//! Protected waiting-item binding, source order, and refusal before mutation.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::streaming::{Item, StreamEvent};
use crate::wire::Operation;
use serde_json::json;

// F streaming/tests.rs:4528
#[test]
fn an_unknown_part_before_parent_evidence_joins_the_item_that_evidence_names() {
    let unknown = json!({"type":"future_part","payload":{"list":[1]}});
    let part = json!({"type":"response.content_part.added","item_id":"rs_spelling_only","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    for parent in ["message", "reasoning"] {
        let item = |content| {
            if parent == "message" {
                message_item("rs_spelling_only", content)
            } else {
                json!({"type":"reasoning","id":"rs_spelling_only","summary":[],"content":content})
            }
        };
        for evidence in ["added", "done", "terminal"] {
            let mut frames = vec![part.clone()];
            if evidence != "terminal" {
                frames.push(json!({"type":format!("response.output_item.{evidence}"),"output_index":0,"sequence_number":1,"item":item(json!([]))}));
            }
            frames.push(completed_with_output(if evidence == "terminal" {
                json!([item(json!([unknown]))])
            } else {
                json!([])
            }));
            let driven = drive(&frames);
            assert!(
                driven.errors.is_empty(),
                "{parent}/{evidence}: {:?}",
                driven.errors
            );
            assert!(
                unknown_payloads(&driven.items).is_empty(),
                "{parent}/{evidence}"
            );
            assert_eq!(driven.response.choice.len(), 1, "{parent}/{evidence}");
            if parent == "reasoning" {
                let AssistantContent::CarriedReasoning(reasoning) = &driven.response.choice[0]
                else {
                    panic!("typed reasoning evidence owns this part")
                };
                assert!(
                    matches!(reasoning.content.as_slice(),[CarriedReasoningPart::OpaqueContent(payload)] if payload.item==unknown)
                );
            } else {
                assert!(matches!(
                    &driven.response.choice[0],
                    AssistantContent::CarriedText(_)
                ));
            }
            assert_eq!(
                replay_opaque_choice(driven.response.choice)[0]["content"],
                json!([unknown])
            );
        }
    }
}

// F streaming/tests.rs:4604
#[test]
fn a_known_typed_part_binds_waiting_unknown_siblings_in_content_order() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let driven = drive(&[
        json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":1,"delta":"B"}),
        completed_with_output(json!([])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(
        replay_opaque_choice(driven.response.choice)[0]["content"],
        json!([unknown,{"type":"output_text","text":"B"}])
    );
}

// F streaming/tests.rs:4626
#[tokio::test]
async fn unresolved_unknown_parts_leave_raw_before_the_flush_without_a_fabricated_choice() {
    use futures::StreamExt;
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let parts = [
        json!({"type":"response.content_part.added","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":1,"part":first}),
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":2,"part":second}),
    ];
    for ending in ["eof", "error", "failed", "incomplete", "completed"] {
        let mut frames = parts.to_vec();
        match ending {
            "error" => frames.push(json!({"error":{"message":"failure","type":"server_error"}})),
            "failed" | "incomplete" => {
                let mut terminal = completed_with_output(json!([]));
                terminal["type"] = json!(format!("response.{ending}"));
                terminal["response"]["status"] = json!(ending);
                frames.push(terminal);
            }
            "completed" => frames.push(completed_with_output(json!([]))),
            _ => {}
        }
        let mut stream = responses_stream_of(&frames).await;
        let mut unknowns = Vec::new();
        let mut errors = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(Item::Unknown(raw)) => {
                    // G163: the public partial snapshot reports an arrived genuine end.
                    assert_eq!(
                        stream.partial().response_id(),
                        (ending == "completed").then_some("resp_123")
                    );
                    unknowns.push(raw.value().clone());
                }
                Ok(Item::Event(StreamEvent::Start { .. })) => {
                    panic!("{ending}: no fabricated parent")
                }
                Ok(other) => panic!("unexpected item: {other:?}"),
                Err(_) => errors += 1,
            }
        }
        assert_eq!(
            errors,
            usize::from(!matches!(ending, "eof" | "completed")),
            "{ending}"
        );
        assert_eq!(unknowns, [first.clone(), second.clone()], "{ending}");
        assert!(stream.partial().choice.is_empty());
        if ending == "completed" {
            assert!(stream.finish().await.is_ok());
        }
    }
}

// F streaming/tests.rs:5041
#[test]
fn a_terminal_restating_a_message_at_another_position_does_not_repeat_its_text() {
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[]});
    let message = message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]));
    for id_bearing in [true, false] {
        let mut delta = json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"});
        if id_bearing {
            delta["item_id"] = json!("msg_1");
        }
        let driven = drive(&[delta, completed_with_output(json!([reasoning, message]))]);
        if id_bearing {
            assert!(driven.errors.is_empty());
            assert_eq!(
                folded_texts(&driven.response.choice),
                ["Hello"],
                "id_bearing={id_bearing}"
            );
            let received = driven.response.choice[0]
                .native_item()
                .expect("actual terminal message");
            assert_eq!(received, &message);
        } else {
            assert!(
                matches!(driven.errors.as_slice(), [crate::error::ProviderError::Response(message)]
                if message == "conflicting Responses parent kind at output 0: reasoning in a message")
            );
            assert!(!driven.errors[0].is_retryable());
            assert!(driven.response.response_id().is_none());
            assert_eq!(
                folded_texts(&driven.response.choice),
                ["Hello"],
                "id_bearing={id_bearing}"
            );
        }
    }
}

// F streaming/tests.rs:5169
#[test]
fn a_reasoning_slot_restated_as_the_other_kind_is_refused() {
    let unknown = json!({"type":"future_summary","payload":[1]});
    let known_delta = json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"});
    let unknown_part = |kind: &str| json!({"type":format!("response.reasoning_summary_part.{kind}"),"item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":1,"part":unknown});
    let known_done = json!({"type":"response.reasoning_summary_text.done","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":2,"text":"known"});
    let known_snapshot = json!({"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"type":"reasoning","id":"rs_k","summary":[{"type":"summary_text","text":"known"}]}});
    let undecided = json!({"type":"response.content_part.added","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"future_content","payload":[2]}});
    let reasoning_delta = json!({"type":"response.reasoning_text.delta","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":1,"delta":"text"});
    for (index, frames) in [
        vec![known_delta, unknown_part("done")],
        vec![unknown_part("added"), known_done],
        vec![unknown_part("done"), known_snapshot],
        vec![undecided, reasoning_delta],
    ]
    .into_iter()
    .enumerate()
    {
        let driven = drive(&frames);
        assert_eq!(driven.errors.len(), 1, "case {index}");
        assert!(
            driven.errors[0]
                .to_string()
                .contains("conflicting Responses content part kind")
        );
        assert!(driven.response.response_id().is_none());
    }
}

// F streaming/tests.rs:5248
#[test]
fn responses_replay_refuses_text_beside_an_opaque_marker() {
    let raw = json!({"type":"future_part","payload":[1]});
    for id in [Some("msg_opaque".to_owned()), None] {
        let mut choice = unary_choice(json!([message_item(
            id.as_deref().unwrap_or(""),
            json!([raw])
        )]));
        let [AssistantContent::CarriedText(text)] = choice.as_mut_slice() else {
            panic!("one opaque text part")
        };
        text.item_id = id.clone();
        *text.text_mut() = "replacement".into();
        let wire = crate::providers::openai::wire::OpenAIConfig::new("test-key")
            .responses("different-model");
        let request = crate::completion::CompletionRequest::new("continue")
            .message(crate::message::Message::from(choice));
        let result =
            crate::operation::Completion::prepare(request, &crate::wire::Wire::describe(&wire))
                .and_then(|request| {
                    crate::wire::Wire::encode(&wire, request, crate::wire::Mode::Unary)
                        .map_err(crate::error::ProviderError::from)
                });
        let error = result.expect_err("text beside an opaque marker cannot replay");
        assert!(
            error
                .to_string()
                .contains("an opaque Responses message part carries non-empty text"),
            "id={id:?}: {error}"
        );
    }
}

// F streaming/tests.rs:5270
#[test]
fn reasoning_text_content_parts_are_known_reasoning_content() {
    let driven = drive(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":{"type":"reasoning","id":"rs_or","summary":[]}}),
        json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":""}}),
        json!({"type":"response.reasoning_text.delta","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":2,"delta":"think"}),
        json!({"type":"response.reasoning_text.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":3,"text":"think"}),
        json!({"type":"response.content_part.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":4,"part":{"type":"reasoning_text","text":"think"}}),
        completed_with_output(
            json!([{"type":"reasoning","id":"rs_or","summary":[],"content":[{"type":"reasoning_text","text":"think"}]}]),
        ),
    ]);
    assert!(driven.errors.is_empty());
    let [AssistantContent::CarriedReasoning(reasoning)] = driven.response.choice.as_slice() else {
        panic!("one reasoning item")
    };
    assert_eq!(reasoning.display_text(), "think");
    assert!(!reasoning.has_opaque_parts());
    let malformed = drive(&[
        json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"reasoning_text"}}),
    ]);
    assert!(
        !malformed.errors.is_empty(),
        "known tag without text is malformed"
    );
}

#[test]
fn same_id_terminal_relocation_keeps_the_actual_part_and_its_delivered_source_order() {
    let message = message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]));
    let driven = drive(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,
            "content_index":0,"sequence_number":0,"delta":"Hello"}),
        completed_with_output(json!([
            {"type":"reasoning","id":"rs_1","summary":[]}, message
        ])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(folded_texts(&driven.response.choice), ["Hello"]);
    let starts = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::Text,
                source_order,
                ..
            }) => Some((*part, *source_order)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0].1, Some(crate::streaming::SourceOrder::new(0, 0)));
    let ends = driven
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::Text(text),
                ..
            }) if text.text == "Hello" => Some(*part),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ends, vec![starts[0].0]);
    let received = driven.response.choice[0]
        .native_item()
        .expect("genuine terminal native item");
    assert_eq!(received, &message);
}

#[test]
fn a_same_id_message_carries_its_bound_unknown_part_through_terminal_relocation() {
    let unknown = json!({"type":"future_part","payload":[null,7]});
    let message = message_item(
        "msg_1",
        json!([
            {"type":"output_text","text":"Hello"}, unknown
        ]),
    );
    let driven = drive(&[
        json!({"type":"response.content_part.added","item_id":"msg_1","output_index":2,
            "content_index":1,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,
            "content_index":0,"sequence_number":1,"delta":"Hello"}),
        completed_with_output(json!([
            {"type":"reasoning","id":"rs_1","summary":[]}, message
        ])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(folded_texts(&driven.response.choice), ["Hello", ""]);
    assert_eq!(opaque_texts(&driven.response.choice), [&unknown]);
    assert!(unknown_payloads(&driven.items).is_empty());
}

#[test]
fn every_message_relocation_is_validated_before_any_state_or_id_moves() {
    use super::super::{ResponsesDecoder, ResponsesStreamOptions};
    use crate::wire::{Decoder, WireEvent, WireFrame};
    let delta = |id: &str, index, sequence, text: &str| {
        json!({
            "type":"response.output_text.delta","item_id":id,"output_index":index,
            "content_index":0,"sequence_number":sequence,"delta":text
        })
    };
    for (prefix, output, expected) in [
        (
            vec![
                delta("msg_a", 0, 0, "A"),
                delta("msg_b", 1, 1, "B"),
                json!({"type":"response.content_part.added","item_id":"foreign",
                "output_index":3,"content_index":0,"sequence_number":2,
                "part":{"type":"future_part","payload":[null,7]}}),
            ],
            json!([
                {"type":"reasoning","id":"rs_0","summary":[]},
                {"type":"reasoning","id":"rs_1","summary":[]},
                message_item("msg_a",json!([{"type":"output_text","text":"A"}])),
                message_item("msg_b",json!([{"type":"output_text","text":"B"}]))
            ]),
            "conflicting Responses terminal message pending ownership at output 3, content 0: item msg_b cannot claim an anonymous or foreign part",
        ),
        (
            vec![
                delta("msg_a", 0, 0, "A"),
                delta("msg_b", 1, 1, "B"),
                json!({"type":"response.content_part.added","output_index":3,
                "content_index":0,"sequence_number":2,
                "part":{"type":"future_part","payload":[null,7]}}),
            ],
            json!([
                {"type":"reasoning","id":"rs_0","summary":[]},
                {"type":"reasoning","id":"rs_1","summary":[]},
                message_item("msg_a",json!([{"type":"output_text","text":"A"}])),
                message_item("msg_b",json!([{"type":"output_text","text":"B"}]))
            ]),
            "conflicting Responses terminal message pending ownership at output 3, content 0: item msg_b cannot claim an anonymous or foreign part",
        ),
        (
            vec![delta("msg_a", 0, 0, "A")],
            json!([
                message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
                message_item("msg_a", json!([{"type":"output_text","text":"A"}]))
            ]),
            "conflicting Responses terminal message ID msg_a: claimed outputs 0 and 1",
        ),
        (
            vec![delta("msg_a", 0, 0, "A"), delta("msg_b", 1, 1, "B")],
            json!([
                {"type":"reasoning","id":"rs_0","summary":[]},
                {"type":"reasoning","id":"rs_1","summary":[]},
                message_item("msg_a",json!([{"type":"output_text","text":"A"}])),
                message_item("msg_b",json!([{"type":"future_part","payload":[7]}]))
            ]),
            "conflicting Responses content part kind at output 1, content 0: stored text part restated as opaque",
        ),
        (
            vec![
                delta("msg_a", 0, 0, "A"),
                json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_existing",
                "output_index":1,"summary_index":0,"sequence_number":1,"delta":"thinking"}),
            ],
            json!([
                {"type":"reasoning","id":"rs_existing","summary":[{"type":"summary_text","text":"thinking"}]},
                message_item("msg_a",json!([{"type":"output_text","text":"A"}]))
            ]),
            "conflicting Responses terminal message destination at output 1: occupied while relocating item msg_a from output 0",
        ),
    ] {
        let result = crate::driver::decode_with(
            crate::operation::Turn::relayed("openai"),
            "openai",
            |reply| {
                let mut decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
                for frame in prefix {
                    let WireEvent::Known(event) =
                        decoder.classify(WireFrame::Text(frame.to_string()))
                    else {
                        panic!("actual valid prefix frames");
                    };
                    let _ = decoder.decode(event, reply.out())?;
                }
                let messages = |decoder: &ResponsesDecoder| {
                    decoder
                        .messages
                        .iter()
                        .map(|(slot, state)| {
                            (
                                *slot,
                                state.id.clone(),
                                state.phase.clone(),
                                state.delta,
                                state.complete,
                                state.item.clone(),
                                state
                                    .parts
                                    .iter()
                                    .map(|(part, state)| {
                                        (
                                            *part,
                                            state.at,
                                            state.text.clone(),
                                            state.raw.clone(),
                                            state.opaque,
                                            state.refusal,
                                            state.observed_phase.clone(),
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                let before_messages = messages(&decoder);
                let before_ids = decoder.message_ids.clone();
                let before_parents = decoder.parents.clone();
                let before_pending = decoder.undecided.clone();
                let before_reasoning = decoder
                    .reasoning
                    .iter()
                    .map(|(slot, state)| {
                        (
                            *slot,
                            state.at,
                            state.id.clone(),
                            state.summary.clone(),
                            state.content.clone(),
                            state.item.clone(),
                            state.complete,
                            state.encrypted.clone(),
                            state.signature.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                let before_active = decoder.active_message_part;
                let error = decoder
                    .relocate_terminal_messages(output.as_array().expect("actual terminal output"))
                    .expect_err("all relocation claims must be compatible");
                assert!(
                    matches!(&error, crate::error::ProviderError::Response(message)
                    if message == expected)
                );
                assert!(!error.is_retryable());
                assert_eq!(messages(&decoder), before_messages);
                assert_eq!(decoder.message_ids, before_ids);
                assert_eq!(decoder.parents, before_parents);
                assert!(decoder.undecided == before_pending);
                assert_eq!(
                    decoder
                        .reasoning
                        .iter()
                        .map(|(slot, state)| (
                            *slot,
                            state.at,
                            state.id.clone(),
                            state.summary.clone(),
                            state.content.clone(),
                            state.item.clone(),
                            state.complete,
                            state.encrypted.clone(),
                            state.signature.clone()
                        ))
                        .collect::<Vec<_>>(),
                    before_reasoning
                );
                assert_eq!(decoder.active_message_part, before_active);
                Err(error)
            },
        );
        assert!(result.outcome.is_err());
    }
}
