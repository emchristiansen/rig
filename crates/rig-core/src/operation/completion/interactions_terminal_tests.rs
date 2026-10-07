use super::{Block, CallFragment, Completion, Turn};
use crate::driver::{Decoded, decode_with};
use crate::error::{ErrorDetail, ErrorKind, ProviderError};
use crate::message::{AssistantContent, CallId, ToolName};
use crate::streaming::{PartKind, SequenceError, StreamEvent, Transcript};
use crate::wire::Out;
use serde_json::{Value, json};

fn write(action: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>) -> Decoded<Completion> {
    decode_with(Turn::relayed("gemini"), "gemini", |reply| {
        action(&mut reply.out())?;
        // A part finalizer alone supplies no interaction.completed event.
        Err(ProviderError::Truncated)
    })
}

fn announced(out: &mut Out<'_, Completion>, item: Value) -> Result<(), ProviderError> {
    out.open(7, Block::Call {
        id: CallId::from_wire("call_1"),
        name: ToolName::new("lookup").expect("genuine announced fixture name"),
        namespace: None,
    }, item)
}

fn prefix(result: &Decoded<Completion>) -> Transcript {
    Transcript::from_items(result.items.iter().filter_map(|item| item.as_ref().ok().cloned()).collect())
        .expect("actual Out grammar")
}

#[test]
fn valid_terminal_hook_finalizes_the_announced_call_without_native_or_response_terminal() {
    let opening = json!({"type":"function_call","id":"call_1","name":"lookup","arguments":{"stale":true}});
    let result = write(|out| {
        announced(out, opening.clone())?;
        out.announce(7, json!({"stale":true}))?;
        out.fragment(Some(7), CallFragment { arguments: Some("{\"x\":"), ..CallFragment::default() })?;
        out.fragment(Some(7), CallFragment { arguments: Some("1}"), ..CallFragment::default() })?;
        out.finish_call_at_terminal_without_native(7)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)), "the finalizer never supplies a response terminal");
    let transcript = prefix(&result);
    let events = transcript.events().collect::<Vec<_>>();
    assert_eq!(events.len(), 4);
    assert!(matches!(events[0], StreamEvent::Start { kind: PartKind::ToolCall, name: Some(name), tool_slot_state: None, .. } if name.as_str() == "lookup"));
    assert!(events.iter().all(|event| event.part() == events[0].part()));
    let fragments = events.iter().filter_map(|event| match event {
        StreamEvent::Arguments { json, .. } => Some(json.as_str()),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(fragments, vec!["{\"x\":", "1}"]);
    let StreamEvent::End { content: AssistantContent::ToolCall(call), reasoning_authority: None, .. } = events[3] else { panic!("one canonical call close"); };
    assert_eq!(call.id, CallId::from_wire("call_1"));
    assert_eq!(call.function.name.as_str(), "lookup");
    assert_eq!(call.function.arguments_value(), json!({"x":1}));
    assert!(call.native.is_none(), "the opening record was not restated complete");
    Transcript::parse(serde_json::to_value(&transcript).expect("actual writer events")).expect("one real Part closed");
}

#[test]
fn only_no_delta_uses_announced_arguments_and_real_empty_or_blank_deltas_supersede_them() {
    for fragment in [None, Some(""), Some(" \t")] {
        let result = write(|out| {
            announced(out, json!({"id":"opening-only"}))?;
            out.announce(7, json!({"announced":true}))?;
            if let Some(arguments) = fragment {
                out.fragment(Some(7), CallFragment { arguments: Some(arguments), ..CallFragment::default() })?;
            }
            out.finish_call_at_terminal_without_native(7)
        });
        let transcript = prefix(&result);
        let ends = transcript.events().filter_map(|event| match event {
            StreamEvent::End { content: AssistantContent::ToolCall(call), .. } => Some(call),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].function.arguments_value(), if fragment.is_none() { json!({"announced":true}) } else { json!({}) });
        assert!(ends[0].native.is_none());
        assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
    }
}

#[test]
fn malformed_terminal_is_typed_in_band_with_exact_retained_input_and_a_partial_part() {
    let raw = " \n{\"x\":\t ";
    let parser_reason = crate::json_utils::parse_tool_arguments(raw).expect_err("malformed fixture").to_string();
    let result = write(|out| {
        announced(out, json!({"id":"opening-only"}))?;
        out.announce(7, json!({"stale":true}))?;
        out.fragment(Some(7), CallFragment { arguments: Some(raw), ..CallFragment::default() })?;
        out.finish_call_at_terminal_without_native(7)
    });
    let transcript = prefix(&result);
    let events = transcript.events().collect::<Vec<_>>();
    assert!(matches!(events.as_slice(), [StreamEvent::Start { .. }, StreamEvent::Arguments { json, .. }] if json == raw));
    assert_eq!(events[0].part(), events[1].part());
    assert!(transcript.events().all(|event| !matches!(event, StreamEvent::End { .. } | StreamEvent::Discard { .. })));
    assert!(matches!(Transcript::parse(serde_json::to_value(&transcript).expect("actual partial writer events")), Err(SequenceError::Unclosed(_))));
    let error = result.outcome.expect_err("genuine malformed terminal input remains refused");
    assert!(matches!(error, ProviderError::Relayed(_)));
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::Response);
    let Some(ErrorDetail::MalformedToolInput(detail)) = &report.detail else { panic!("actual typed diagnostic"); };
    assert_eq!(detail.name, "lookup");
    assert_eq!(detail.id, CallId::from_wire("call_1"));
    assert_eq!(detail.raw, raw);
    assert_eq!(detail.error, parser_reason);
    let provider = detail.provider.as_ref().expect("retained provider correlation");
    assert_eq!(provider.call_id.as_str(), "call_1");
    assert!(provider.item_id.is_none(), "no output-item ID was inferred from the opening JSON");
    assert_eq!(result.items.last().expect("actual in-band failure").as_ref().expect_err("typed error item").report(), report);
}

#[test]
fn default_finish_open_keeps_its_native_and_existing_malformed_argument_policy() {
    let raw = "{\"x\":1,\"unfinished\":";
    let item = json!({"id":"provider-item","unchanged":[null,7]});
    let result = write(|out| {
        announced(out, item.clone())?;
        out.fragment(Some(7), CallFragment { arguments: Some(raw), ..CallFragment::default() })?;
        out.finish_open()
    });
    let transcript = prefix(&result);
    let ends = transcript.events().filter_map(|event| match event {
        StreamEvent::End { content: AssistantContent::ToolCall(call), .. } => Some(call),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].native.as_ref().expect("default Complete still retains Native").item, item);
    assert_eq!(ends[0].function.invalid_arguments.as_deref(), Some(raw));
    assert_eq!(ends[0].function.arguments_value(), json!({"x":1}));
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

#[test]
fn terminal_hook_refuses_missing_wrong_kind_or_unidentified_real_drafts_without_consuming_them() {
    let result = write(|out| {
        assert!(out.finish_call_at_terminal_without_native(7).is_err());
        out.open(7, Block::Text, Value::Null)?;
        out.push(7, "text")?;
        assert!(out.finish_call_at_terminal_without_native(7).is_err());
        out.finish(7)?;
        out.fragment(Some(8), CallFragment { name: Some("lookup"), arguments: Some("{}"), id: None })?;
        assert!(out.finish_call_at_terminal_without_native(8).is_err());
        out.finish_open()
    });
    let transcript = prefix(&result);
    assert_eq!(transcript.events().filter(|event| matches!(event, StreamEvent::End { .. })).count(), 2);
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}
