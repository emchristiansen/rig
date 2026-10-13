use super::{Block, CallFragment, Completion, Finish, Turn};
use crate::completion::CompletionResponse;
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart, ToolName,
};
use crate::streaming::{Item, SourceOrder, StreamEvent};
use crate::wire::{Out, Shared};
use serde_json::{Value, json};
use std::sync::Mutex;

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the original tool name is nonempty")
}

fn write(
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::relayed("test"), "test", |reply| {
        let mut out = reply.out();
        action(&mut out)?;
        let _ = out.end(Finish::default());
        Ok(())
    })
}

fn response(decoded: Decoded<Completion>) -> CompletionResponse {
    decoded.outcome.expect("the current shared fold completes")
}

fn take(shared: &Mutex<Shared<Completion>>) {
    let mut shared = shared.lock().expect("test reply lock");
    while let Some(item) = shared.take() {
        item.expect("the real writer's item is accepted by its fold");
    }
}

fn observed(shared: &Mutex<Shared<Completion>>) -> Vec<AssistantContent> {
    take(shared);
    let shared = shared.lock().expect("test reply lock");
    shared
        .fold
        .partial(shared.end.as_ref(), &shared.reply("test"), None)
        .choice
}

fn finish(shared: Mutex<Shared<Completion>>) -> Vec<AssistantContent> {
    let _ = Out::new(&shared).end(Finish::default());
    take(&shared);
    shared
        .into_inner()
        .expect("test reply lock")
        .conclude("test")
        .expect("the real shared reply ends")
        .choice
}

fn carried(id: &str, content: CarriedReasoningPart) -> CarriedReasoning {
    // The original generic fixture has no provider/raw item. Keep that absence;
    // do not invent received JSON, Native or an issuer for the typed subject.
    CarriedReasoning::new(vec![content], CarriedPayload::new(Value::Null, None))
        .with_id(Some(id.to_owned()))
}

fn reasoning_labels(parts: &[AssistantContent]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(value) => Some(vec![value.text.clone()]),
            AssistantContent::CarriedReasoning(value) => Some(
                value
                    .content
                    .iter()
                    .map(|part| match part {
                        CarriedReasoningPart::Text { text, .. }
                        | CarriedReasoningPart::Summary(text)
                        | CarriedReasoningPart::Encrypted(text) => text.clone(),
                        other => panic!("only the original known reasoning subjects: {other:?}"),
                    })
                    .collect(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

fn labels(parts: &[AssistantContent]) -> Vec<&str> {
    parts
        .iter()
        .map(|part| match part {
            AssistantContent::Text(text) => text.text.as_str(),
            AssistantContent::ToolCall(call) => call.function.name.as_str(),
            _ => "other",
        })
        .collect()
}

#[test]
fn text_metadata_opens_and_merges_into_the_block() {
    let decoded = write(|out| {
        out.open(0, Block::Text, json!({"block":1}))?;
        out.push(0, "cited")?;
        out.edit(0, |item| {
            super::merge(
                item,
                json!({"citations":["a"]}).as_object().expect("object"),
            )
        })?;
        out.finish(0)
    });
    let parts = response(decoded).choice;
    assert_eq!(parts.len(), 1);
    let Some(content @ AssistantContent::Text(text)) = parts.first() else {
        panic!("expected a text part");
    };
    assert_eq!(text.text, "cited");
    // Text.additional_params is retired; the actual writer retains these
    // exact original values in its real native item.
    let metadata = content.native_item().expect("metadata kept by the writer");
    assert_eq!(metadata["block"], 1);
    assert_eq!(metadata["citations"], json!(["a"]));
    assert_eq!(metadata, &json!({"block":1,"citations":["a"]}));
}

#[test]
fn an_end_restatement_replaces_its_delta_accumulation() {
    let decoded = write(|out| {
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(0, "partial ")?;
        out.push(0, "thought")?;
        out.restate(0, "the complete chain")?;
        out.finish_carried_reasoning(
            0,
            carried(
                "rs_1",
                CarriedReasoningPart::Text {
                    text: "the complete chain".to_owned(),
                    signature: None,
                },
            ),
        )
    });
    let parts = response(decoded).choice;
    assert_eq!(reasoning_labels(&parts), vec!["the complete chain"]);
    assert_eq!(parts.len(), 1);
    assert!(
        matches!(&parts[0], AssistantContent::CarriedReasoning(value)
        if value.id.as_deref()==Some("rs_1") && value.payload.provider.is_none())
    );
}

#[test]
fn an_open_part_collapses_across_interleaved_output() {
    let decoded = write(|out| {
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(0, "thinking")?;
        out.whole(
            1,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: name("probe"),
                namespace: None,
            },
            Value::Null,
            "{}",
        )?;
        out.restate(0, "full reasoning")?;
        out.finish_carried_reasoning(
            0,
            carried(
                "rs_1",
                CarriedReasoningPart::Text {
                    text: "full reasoning".to_owned(),
                    signature: None,
                },
            ),
        )
    });
    let parts = response(decoded).choice;
    assert_eq!(reasoning_labels(&parts), vec!["full reasoning"]);
    assert_eq!(
        parts.len(),
        2,
        "reasoning replaced in place beside the call"
    );
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::CarriedReasoning(_))
    ));
    assert!(
        matches!(parts.get(1), Some(AssistantContent::ToolCall(call))
        if call.id==CallId::from_wire("call_1") && call.function.name.as_str()=="probe"
            && call.function.arguments_value()==json!({}))
    );
}

#[test]
fn same_key_whole_blocks_are_siblings_and_all_survive() {
    let decoded = write(|out| {
        for (index, content) in [
            CarriedReasoningPart::Summary("s1".to_owned()),
            CarriedReasoningPart::Summary("s2".to_owned()),
            CarriedReasoningPart::Text {
                text: "visible".to_owned(),
                signature: None,
            },
            CarriedReasoningPart::Encrypted("enc".to_owned()),
        ]
        .into_iter()
        .enumerate()
        {
            let value = carried("rs_1", content);
            out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
            out.push(index, &value.display_text())?;
            out.finish_carried_reasoning(index, value)?;
        }
        Ok(())
    });
    let parts = response(decoded).choice;
    assert_eq!(reasoning_labels(&parts), vec!["s1", "s2", "visible", "enc"]);
    assert_eq!(parts.len(), 4);
}

#[test]
fn deltas_then_sibling_whole_blocks_keep_each_part_once() {
    let decoded = write(|out| {
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(0, "s1")?;
        for (index, content) in [
            CarriedReasoningPart::Summary("s1".to_owned()),
            CarriedReasoningPart::Summary("s2".to_owned()),
            CarriedReasoningPart::Encrypted("enc".to_owned()),
        ]
        .into_iter()
        .enumerate()
        {
            let value = carried("rs_1", content);
            if index == 0 {
                out.restate(0, &value.display_text())?;
            } else {
                out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
                out.push(index, &value.display_text())?;
            }
            out.finish_carried_reasoning(index, value)?;
        }
        Ok(())
    });
    let parts = response(decoded).choice;
    assert_eq!(reasoning_labels(&parts), vec!["s1", "s2", "enc"]);
    assert_eq!(parts.len(), 3);
}

#[test]
fn finish_on_an_empty_stream_yields_no_parts() {
    let parts = response(write(|_| Ok(()))).choice;
    assert!(parts.is_empty());
}

#[test]
fn unordered_tools_keep_current_opened_order_with_the_original_fixture() {
    let decoded = write(|out| {
        out.fragment(
            Some(0),
            CallFragment {
                id: Some("tool-arrival"),
                name: None,
                arguments: None,
            },
        )?;
        out.whole(1, Block::Text, Value::Null, "text")?;
        out.fragment(
            Some(0),
            CallFragment {
                id: None,
                name: Some("tool"),
                arguments: Some("{}"),
            },
        )?;
        out.finish(0)
    });
    // G31 deliberately selects opened order; original completion-arrival
    // ["text", "tool"] is a named behavioral divergence, never parity.
    assert_eq!(labels(&response(decoded).choice), ["tool", "text"]);
}

#[test]
fn ordered_tool_reservation_survives_late_completion() {
    let decoded = write(|out| {
        out.in_source_order(SourceOrder::new(0, 0), |out| {
            out.fragment(
                Some(0),
                CallFragment {
                    id: Some("tool-ordered"),
                    name: None,
                    arguments: None,
                },
            )
        })?;
        out.in_source_order(SourceOrder::new(1, 0), |out| {
            out.whole(1, Block::Text, Value::Null, "text")
        })?;
        out.fragment(
            Some(0),
            CallFragment {
                id: None,
                name: Some("tool"),
                arguments: Some("{}"),
            },
        )?;
        out.finish(0)
    });
    assert_eq!(labels(&response(decoded).choice), ["tool", "text"]);
}

#[test]
fn ordered_slots_sort_around_fixed_unordered_slots() {
    let decoded = write(|out| {
        out.whole(0, Block::Text, Value::Null, "fixed-a")?;
        out.in_source_order(SourceOrder::new(2, 0), |out| {
            out.whole(1, Block::Text, Value::Null, "ordered-high")
        })?;
        out.whole(2, Block::Text, Value::Null, "fixed-b")?;
        out.in_source_order(SourceOrder::new(1, 0), |out| {
            out.whole(3, Block::Text, Value::Null, "ordered-low")
        })
    });
    assert_eq!(
        labels(&response(decoded).choice),
        ["fixed-a", "ordered-low", "fixed-b", "ordered-high"]
    );
}

#[test]
fn equal_source_coordinates_keep_stable_arrival_order() {
    let decoded = write(|out| {
        let coordinate = SourceOrder::new(1, 0);
        out.in_source_order(coordinate, |out| {
            out.whole(0, Block::Text, Value::Null, "first")
        })?;
        out.in_source_order(coordinate, |out| {
            out.whole(1, Block::Text, Value::Null, "second")
        })
    });
    assert_eq!(labels(&response(decoded).choice), ["first", "second"]);
}

#[test]
fn unfinished_ordered_reservations_are_absent() {
    let shared = Mutex::new(Shared::new(Turn::relayed("test")));
    let mut out = Out::new(&shared);
    out.in_source_order(SourceOrder::new(0, 0), |out| {
        out.fragment(
            Some(0),
            CallFragment {
                id: Some("unfinished-tool"),
                name: None,
                arguments: None,
            },
        )
    })
    .expect("unfinished tool reservation");
    out.in_source_order(SourceOrder::new(1, 0), |out| {
        out.whole(1, Block::Text, Value::Null, "survivor")
    })
    .expect("the survivor arrives");
    assert_eq!(labels(&observed(&shared)), ["survivor"]);
    // G48: the genuine end closes incomplete calls instead of old discard.
    // This unnamed reservation contributes no canonical call in either view.
    drop(out);
    assert_eq!(labels(&finish(shared)), ["survivor"]);
}

fn scripted_prefix(out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    out.open(0, Block::Text, Value::Null)?;
    out.push(0, "hello ")?;
    out.push(0, "world")?;
    out.fragment(
        Some(1),
        CallFragment {
            id: Some("call_1"),
            name: Some("lookup"),
            arguments: Some("{\"q\":\"rust\"}"),
        },
    )?;
    out.finish(1)
}

fn scripted_tail(out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    out.open(2, Block::Reasoning { redacted: false }, Value::Null)?;
    out.push(2, "because")?;
    out.finish_carried_reasoning(
        2,
        carried("rs_1", CarriedReasoningPart::Summary("because".to_owned())),
    )?;
    out.finish(0)
}

#[test]
fn snapshot_is_non_destructive_and_does_not_change_finish() {
    let reference = response(write(|out| {
        scripted_prefix(out)?;
        scripted_tail(out)
    }))
    .choice;
    let shared = Mutex::new(Shared::new(Turn::relayed("test")));
    let mut out = Out::new(&shared);
    scripted_prefix(&mut out).expect("the exact original script prefix");
    let first = observed(&shared);
    let second = observed(&shared);
    assert_eq!(first, second, "two snapshots mid-stream are equal");
    assert_eq!(
        first.len(),
        2,
        "text and the completed call so far: {first:?}"
    );
    assert!(matches!(&first[0],AssistantContent::Text(text) if text.text=="hello world"));
    assert!(matches!(&first[1],AssistantContent::ToolCall(call)
        if call.function.name.as_str()=="lookup" && call.function.arguments_value()==json!({"q":"rust"})));
    scripted_tail(&mut out).expect("the exact original script tail");
    let before_finish = observed(&shared);
    drop(out);
    let finished = finish(shared);
    assert_eq!(
        finished, before_finish,
        "finish returns what snapshot would"
    );
    assert_eq!(
        finished, reference,
        "mid-stream snapshots did not change the fold"
    );
    assert_eq!(reasoning_labels(&finished), vec!["because"]);
}

#[test]
fn snapshot_omits_open_calls_and_keeps_open_reasoning() {
    let shared = Mutex::new(Shared::new(Turn::relayed("test")));
    let mut out = Out::new(&shared);
    out.fragment(
        Some(0),
        CallFragment {
            id: Some("call_1"),
            name: Some("ping"),
            arguments: Some("{\"x\":"),
        },
    )
    .expect("the exact original open call");
    out.open(1, Block::Reasoning { redacted: false }, Value::Null)
        .expect("reasoning opens");
    out.push(1, "thinking").expect("original reasoning delta");
    let snapshot = observed(&shared);
    assert_eq!(snapshot.len(), 1, "got {snapshot:?}");
    assert_eq!(reasoning_labels(&snapshot), vec!["thinking"]);
    assert!(
        snapshot
            .iter()
            .all(|part| !matches!(part, AssistantContent::ToolCall(_)))
    );
    // G96 uses the actual partial().choice observation. No provider end is
    // fabricated merely to observe the original open reasoning fixture.
}

#[test]
fn current_snapshot_omits_the_original_open_reasoning_and_call() {
    let shared = Mutex::new(Shared::new(Turn::relayed("test")));
    let mut out = Out::new(&shared);
    out.fragment(
        Some(0),
        CallFragment {
            id: Some("call_1"),
            name: Some("ping"),
            arguments: Some("{\"x\":"),
        },
    )
    .expect("original open call");
    out.open(1, Block::Reasoning { redacted: false }, Value::Null)
        .expect("reasoning opens");
    out.push(1, "thinking").expect("original reasoning delta");
    take(&shared);
    let shared = shared.lock().expect("test reply lock");
    assert!(shared.fold.snapshot().is_empty());
    let partial = shared.fold.partial(None, &shared.reply("test"), None);
    assert_eq!(partial.choice.len(), 1);
    assert_eq!(reasoning_labels(&partial.choice), vec!["thinking"]);
}

#[test]
fn text_aggregate_is_fragmentation_invariant_for_selected_original_cases() {
    // These fixed payloads and character cuts belong to the original
    // .{0,60}/0..1000/0..6 quantified set, without reintroducing its harness.
    for (payload, fragments) in [
        ("", vec!["", ""]),
        ("hello", vec!["he", "", "llo"]),
        ("é🦀z", vec!["é", "🦀", "z"]),
    ] {
        assert_eq!(fragments.concat(), payload);
        let assemble = |fragments: &[&str]| {
            response(write(|out| {
                out.open(0, Block::Text, Value::Null)?;
                for fragment in fragments {
                    out.push(0, fragment)?;
                }
                out.finish(0)
            }))
            .choice
        };
        assert_eq!(assemble(&[payload]), assemble(&fragments));
    }
}

#[test]
fn reasoning_aggregate_is_fragmentation_invariant_for_selected_original_cases() {
    for (payload, fragments) in [
        ("r", vec!["", "r", ""]),
        ("thinking", vec!["think", "", "ing"]),
        ("é🦀z", vec!["é", "🦀", "z"]),
    ] {
        assert_eq!(fragments.concat(), payload);
        let assemble = |fragments: &[&str]| {
            write(|out| {
                out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
                for fragment in fragments.iter().filter(|fragment| !fragment.is_empty()) {
                    out.push(0, fragment)?;
                }
                out.finish(0)
            })
        };
        let whole = assemble(&[payload]);
        let split = assemble(&fragments);
        let completed = |decoded: &Decoded<Completion>| {
            decoded.items.iter().find_map(|item| match item {
                Ok(Item::Event(StreamEvent::End { content, .. })) => Some(content.clone()),
                _ => None,
            })
        };
        assert_eq!(completed(&whole), completed(&split));
        assert_eq!(response(whole).choice, response(split).choice);
    }
}

#[test]
fn tool_arguments_are_fragmentation_invariant_for_selected_original_cases() {
    for (value, fragments) in [
        ("", vec!["", "{\"q\":", "\"\"}"]),
        ("rust", vec!["{\"q\":\"", "ru", "", "st\"}"]),
        (
            "abcdefghijklmnopqrst",
            vec!["{\"q\":\"abcd", "efghijklmnop", "qrst\"}"],
        ),
    ] {
        let payload = format!("{{\"q\":\"{value}\"}}");
        assert_eq!(fragments.concat(), payload);
        let finalize = |fragments: &[&str]| {
            let decoded = write(|out| {
                out.open(
                    0,
                    Block::Call {
                        id: CallId::from_wire("call_1"),
                        name: name("probe"),
                        namespace: None,
                    },
                    Value::Null,
                )?;
                for fragment in fragments {
                    out.push(0, fragment)?;
                }
                out.finish(0)
            });
            response(decoded)
                .tool_calls()
                .next()
                .expect("the valid original call finalizes")
                .function
                .arguments_value()
        };
        let whole = finalize(&[payload.as_str()]);
        let split = finalize(&fragments);
        assert_eq!(whole, split);
        assert_eq!(whole, json!({"q":value}));
    }
}

#[test]
fn snapshots_are_pure_reads_for_selected_original_cases() {
    for payload in ["", "aβ"] {
        for snapshots in [0, 1, 3] {
            let reference = response(write(|out| {
                out.open(0, Block::Text, Value::Null)?;
                out.push(0, payload)?;
                out.push(0, "!")?;
                out.finish(0)
            }))
            .choice;
            let shared = Mutex::new(Shared::new(Turn::relayed("test")));
            let mut out = Out::new(&shared);
            out.open(0, Block::Text, Value::Null).expect("text opens");
            out.push(0, payload).expect("original fixed payload");
            let first = observed(&shared);
            for _ in 0..snapshots {
                assert_eq!(observed(&shared), first);
            }
            out.push(0, "!").expect("original suffix");
            out.finish(0).expect("text completes");
            let last = observed(&shared);
            drop(out);
            assert_eq!(finish(shared), last);
            assert_eq!(reference, {
                let shared = Mutex::new(Shared::new(Turn::relayed("test")));
                let mut out = Out::new(&shared);
                out.open(0, Block::Text, Value::Null).expect("text opens");
                out.push(0, payload).expect("original fixed payload");
                let _ = observed(&shared);
                out.push(0, "!").expect("original suffix");
                out.finish(0).expect("text completes");
                drop(out);
                finish(shared)
            });
        }
    }
}
