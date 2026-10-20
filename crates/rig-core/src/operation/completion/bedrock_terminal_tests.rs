use super::{CallFragment, Completion, Turn};
use crate::driver::{Decoded, decode_with};
use crate::error::{ErrorDetail, ErrorKind, ProviderError};
use crate::message::{AssistantContent, CallId};
use crate::streaming::{PartKind, SequenceError, StreamEvent, Transcript};
use crate::wire::{Fold, Out};
use serde_json::json;

fn write(
    action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::relayed("bedrock"), "bedrock", |reply| {
        action(&mut reply.out())?;
        // The bridge closes parts; it never fabricates MessageStop itself.
        Err(ProviderError::Truncated)
    })
}

fn prefix(result: &Decoded<Completion>) -> Transcript {
    Transcript::from_items(
        result
            .items
            .iter()
            .filter_map(|item| item.as_ref().ok().cloned())
            .collect(),
    )
    .expect("actual Out grammar")
}

#[test]
fn two_real_calls_without_item_done_finalize_once_with_no_native_on_the_bedrock_bridge() {
    let result = write(|out| {
        for (index, id, name, arguments) in [
            (3, "tool_1", "lookup", "{\"x\":1}"),
            (5, "tool_2", "search", "{\"q\":\"query\"}"),
        ] {
            out.fragment(
                Some(index),
                CallFragment {
                    id: Some(id),
                    name: Some(name),
                    arguments: Some(arguments),
                },
            )?;
            out.edit(index, |item| {
                *item = json!({"toolUseId":id,"name":name,"opening_only":true})
            })?;
        }
        out.finish_call_at_bedrock_tool_use_terminal(3)?;
        assert!(
            out.finish_call_at_bedrock_tool_use_terminal(3).is_err(),
            "a finished Part cannot close twice"
        );
        out.finish_call_at_bedrock_tool_use_terminal(5)
    });
    assert!(
        matches!(result.outcome, Err(ProviderError::Truncated)),
        "no response terminal is fabricated"
    );
    let transcript = prefix(&result);
    Transcript::parse(serde_json::to_value(&transcript).expect("actual writer stream"))
        .expect("both real Parts close once");
    let starts = transcript
        .events()
        .filter_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                tool_slot_state: None,
                ..
            } => Some(*part),
            _ => None,
        })
        .collect::<Vec<_>>();
    let ends = transcript
        .events()
        .filter_map(|event| match event {
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(call),
                reasoning_authority: None,
            } => Some((*part, call)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 2);
    assert_eq!(ends.len(), 2);
    assert_ne!(starts[0], starts[1]);
    for ((part, call), expected_part) in ends.iter().zip(starts) {
        assert_eq!(*part, expected_part);
        assert!(
            call.native.is_none(),
            "an opening record is not a restated Native item"
        );
    }
    assert_eq!(ends[0].1.id, CallId::from_wire("tool_1"));
    assert_eq!(ends[0].1.function.name.as_str(), "lookup");
    assert_eq!(ends[0].1.function.arguments_value(), json!({"x":1}));
    assert_eq!(ends[1].1.id, CallId::from_wire("tool_2"));
    assert_eq!(ends[1].1.function.name.as_str(), "search");
    assert_eq!(ends[1].1.function.arguments_value(), json!({"q":"query"}));
    // Actual canonical fold replay of writer-owned events, with no response End.
    let mut fold = Turn::relayed("bedrock");
    for event in transcript.events() {
        fold.absorb(event).expect("canonical fold");
    }
    let canonical = fold.snapshot();
    assert_eq!(canonical.len(), 2);
    assert!(canonical.iter().all(
        |content| matches!(content, AssistantContent::ToolCall(call) if call.native.is_none())
    ));
}

#[test]
fn malformed_bedrock_terminal_keeps_exact_typed_input_in_band_without_a_canonical_call() {
    let raw = " \n{\"x\":\t ";
    let parser_reason = crate::json_utils::parse_tool_arguments(raw)
        .expect_err("malformed fixture")
        .to_string();
    let result = write(|out| {
        out.fragment(
            Some(3),
            CallFragment {
                id: Some("tool_1"),
                name: Some("lookup"),
                arguments: Some(raw),
            },
        )?;
        out.edit(3, |item| {
            *item = json!({"toolUseId":"tool_1","opening_only":true})
        })?;
        out.finish_call_at_bedrock_tool_use_terminal(3)
    });
    let transcript = prefix(&result);
    let events = transcript.events().collect::<Vec<_>>();
    assert!(
        matches!(events.as_slice(), [StreamEvent::Start { name: Some(name), .. }, StreamEvent::Arguments { json, .. }] if name.as_str() == "lookup" && json == raw)
    );
    assert_eq!(events[0].part(), events[1].part());
    assert!(
        transcript
            .events()
            .all(|event| !matches!(event, StreamEvent::End { .. } | StreamEvent::Discard { .. }))
    );
    assert!(matches!(
        Transcript::parse(serde_json::to_value(&transcript).expect("actual partial stream")),
        Err(SequenceError::Unclosed(_))
    ));
    let error = result
        .outcome
        .expect_err("typed malformed terminal refusal");
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::Response);
    let Some(ErrorDetail::MalformedToolInput(detail)) = &report.detail else {
        panic!("retained typed evidence");
    };
    assert_eq!(detail.name, "lookup");
    assert_eq!(detail.id, CallId::from_wire("tool_1"));
    assert_eq!(detail.raw, raw);
    assert_eq!(detail.error, parser_reason);
    assert_eq!(
        detail
            .provider
            .as_ref()
            .expect("actual provider ID")
            .call_id
            .as_str(),
        "tool_1"
    );
    assert!(
        detail
            .provider
            .as_ref()
            .expect("actual provider ID")
            .item_id
            .is_none()
    );
    assert_eq!(
        result
            .items
            .last()
            .expect("in-band failure")
            .as_ref()
            .expect_err("typed error item")
            .report(),
        report
    );
    let mut fold = Turn::relayed("bedrock");
    for event in transcript.events() {
        fold.absorb(event).expect("actual canonical prefix");
    }
    assert!(
        fold.snapshot().is_empty(),
        "malformed input produces no canonical call"
    );
}

#[test]
fn default_nonopted_call_stream_stays_partial_without_a_provider_terminal() {
    for raw in ["{\"x\":1}", "{\"x\":"] {
        let result = write(|out| {
            out.fragment(
                Some(3),
                CallFragment {
                    id: Some("tool_1"),
                    name: Some("lookup"),
                    arguments: Some(raw),
                },
            )
        });
        assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
        let transcript = prefix(&result);
        let events = transcript.events().collect::<Vec<_>>();
        assert!(
            matches!(events.as_slice(), [StreamEvent::Start { tool_slot_state: None, .. }, StreamEvent::Arguments { json, .. }] if json == raw)
        );
        assert!(
            transcript.events().all(|event| !matches!(
                event,
                StreamEvent::End { .. } | StreamEvent::Discard { .. }
            ))
        );
        assert!(matches!(
            Transcript::parse(serde_json::to_value(&transcript).expect("actual default prefix")),
            Err(SequenceError::Unclosed(_))
        ));
    }
}
