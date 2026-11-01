//! Terminal message ownership, reasoning parts reached through content-part
//! and text events, call-slot guards, and the flush failure a buffered
//! refusal keeps for the terminal transport error, at the real decoder.

use super::super::{
    MAX_TOOL_INPUT_BYTES, Parent, ResponsesDecoder, ResponsesEvent, ResponsesStreamOptions,
    classify_document_payload,
};
use super::helpers::*;
use crate::driver::decode_with;
use crate::error::{ErrorDetail, ProviderError};
use crate::operation::{Completion, Turn};
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Classify `frame` with the decoder's strict reader and decode it.
fn feed<'id>(
    decoder: &mut ResponsesDecoder,
    frame: &Value,
    out: Out<'id, Completion>,
) -> Result<Flow, ProviderError> {
    let event = known(decoder, frame);
    decoder.decode(event, out)
}

/// `frame` as the decoder's strict reader classifies it.
fn known(decoder: &ResponsesDecoder, frame: &Value) -> ResponsesEvent {
    let WireEvent::Known(event) = decoder.classify(WireFrame::Text(frame.to_string())) else {
        panic!("the real strict reader admits {frame}");
    };
    event
}

fn strict() -> ResponsesDecoder {
    ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
}

fn text_delta(id: &str, index: u64, sequence: u64, text: &str) -> Value {
    json!({"type":"response.output_text.delta","item_id":id,"output_index":index,
        "content_index":0,"sequence_number":sequence,"delta":text})
}

fn summary_delta(id: &str, index: u64, sequence: u64, text: &str) -> Value {
    json!({"type":"response.reasoning_summary_text.delta","item_id":id,"output_index":index,
        "summary_index":0,"sequence_number":sequence,"delta":text})
}

/// The refusal the buffered asserted call `add` with arguments `{"x":481`
/// ends with when it is flushed.
fn assert_malformed_add(error: &ProviderError) {
    let parse = serde_json::from_str::<Value>("{\"x\":481")
        .expect_err("the fixture arguments are malformed JSON");
    let report = error.report();
    assert_eq!(
        report.message,
        format!("tool call `add` arrived with malformed JSON input: {parse}")
    );
    assert!(
        matches!(
            &report.detail,
            Some(ErrorDetail::MalformedToolInput(detail))
                if detail.name.as_str() == "add" && detail.raw == "{\"x\":481"
        ),
        "{report:?}"
    );
}

/// Two message ids join one streamed message slot (Copilot's spelling). A
/// terminal restating either at another output cannot move that shared
/// state: neither id exclusively owns it.
#[test]
fn a_terminal_message_cannot_relocate_state_another_message_id_shares() {
    for relocated in ["msg_b", "msg_a"] {
        let driven = drive(&[
            text_delta("msg_a", 0, 0, "A"),
            text_delta("msg_b", 0, 1, "B"),
            completed_with_output(json!([
                message_item("msg_other", json!([{"type":"output_text","text":"other"}])),
                message_item(relocated, json!([{"type":"output_text","text":"AB"}])),
            ])),
        ]);
        let expected = format!(
            "conflicting Responses terminal message ownership at output 0: item {relocated} does not exclusively own its message state"
        );
        assert!(
            matches!(driven.errors.as_slice(), [ProviderError::Response(message)]
                if *message == expected),
            "{relocated}: {:?}",
            driven.errors
        );
    }
}

#[test]
fn a_summary_text_content_part_opens_the_reasoning_summary_it_names() {
    let part = json!({"type":"summary_text","text":"plan"});
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        let flow = feed(
            &mut decoder,
            &json!({"type":"response.content_part.added","item_id":"rs_1","output_index":0,
                "content_index":0,"sequence_number":0,"part":part}),
            reply.out(),
        )?;
        assert!(matches!(flow, Flow::More));
        assert_eq!(
            decoder.reasoning_ids,
            HashMap::from([("rs_1".to_owned(), 0)])
        );
        assert_eq!(decoder.parents, HashMap::from([(0, Parent::Reasoning)]));
        let state = &decoder.reasoning[&0];
        assert_eq!(state.id.as_deref(), Some("rs_1"));
        assert_eq!(state.summary, BTreeMap::from([(0, part.clone())]));
        assert!(state.content.is_empty());
        assert_eq!(state.at, None, "a restated part is not a live delta");
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

#[test]
fn an_unknown_part_of_a_known_reasoning_item_joins_its_content() {
    let unknown = json!({"type":"future_part","payload":[null,7]});
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        for frame in [
            summary_delta("rs_1", 0, 0, "plan"),
            json!({"type":"response.content_part.added","item_id":"rs_1","output_index":0,
                "content_index":0,"sequence_number":1,"part":unknown}),
        ] {
            assert!(matches!(
                feed(&mut decoder, &frame, reply.out())?,
                Flow::More
            ));
        }
        assert!(
            decoder.undecided.is_empty(),
            "the part's parent is already known"
        );
        let state = &decoder.reasoning[&0];
        assert_eq!(
            state.summary,
            BTreeMap::from([(0, json!({"type":"summary_text","text":"plan"}))])
        );
        assert_eq!(state.content, BTreeMap::from([(0, unknown.clone())]));
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// An added reasoning item may state a content part as a bare string; a
/// later text delta extends that string where it stands.
#[test]
fn a_reasoning_text_delta_extends_a_bare_string_content_part() {
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        for frame in [
            json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,
                "item":{"type":"reasoning","id":"rs_1","summary":[],"content":["abc"]}}),
            json!({"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":0,
                "content_index":0,"sequence_number":1,"delta":"def"}),
        ] {
            assert!(matches!(
                feed(&mut decoder, &frame, reply.out())?,
                Flow::More
            ));
        }
        let state = &decoder.reasoning[&0];
        assert_eq!(state.content, BTreeMap::from([(0, json!("abcdef"))]));
        assert!(
            state.at.is_some(),
            "the delta opened the live reasoning part"
        );
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// An added reasoning item with an empty id claims no identity: it opens the
/// state at its output index and records no reasoning id.
#[test]
fn an_added_reasoning_item_with_an_empty_id_opens_its_output_slot() {
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        assert!(matches!(
            feed(
                &mut decoder,
                &json!({"type":"response.output_item.added","output_index":2,"sequence_number":0,
                    "item":{"type":"reasoning","id":"","summary":[]}}),
                reply.out()
            )?,
            Flow::More
        ));
        assert!(decoder.reasoning_ids.is_empty());
        assert_eq!(decoder.parents, HashMap::from([(2, Parent::Reasoning)]));
        assert_eq!(decoder.reasoning.keys().copied().collect::<Vec<_>>(), [2]);
        assert_eq!(decoder.reasoning[&2].id, None);
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// With immediate publication a finished call's slot is spent: a later
/// announcement of the same item reopens nothing.
#[test]
fn an_announcement_after_immediate_publication_reopens_no_call() {
    let done = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"add",
        "arguments":"{\"q\":1}","status":"completed"});
    let added = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"add",
        "arguments":"","status":"in_progress"});
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = ResponsesDecoder::new(
            "openai",
            ResponsesStreamOptions::strict_with_immediate_tool_calls(),
        );
        for frame in [
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":0,
                "item":done}),
            json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,
                "item":added}),
        ] {
            assert!(matches!(
                feed(&mut decoder, &frame, reply.out())?,
                Flow::More
            ));
        }
        assert!(decoder.calls.is_empty());
        assert_eq!(decoder.finished_calls, HashSet::from([0]));
        assert_eq!(decoder.call_items, HashMap::from([("fc_1".to_owned(), 0)]));
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// A fragment that would take the assembled input past the bound is not
/// retained, and the assembly records that it overflowed; a later fragment
/// within the bound is retained.
#[test]
fn an_argument_fragment_past_the_input_bound_marks_the_call_overflowed() {
    let oversized = "x".repeat(MAX_TOOL_INPUT_BYTES + 1);
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        decoder.call_delta(0, Some("fc_1"), &oversized, &mut reply.out())?;
        let call = &decoder.calls[&0];
        assert!(call.overflowed);
        assert_eq!(call.arguments, "");
        assert_eq!(call.sent, "");
        decoder.call_delta(0, Some("fc_1"), "{}", &mut reply.out())?;
        let call = &decoder.calls[&0];
        assert!(call.overflowed);
        assert_eq!(call.arguments, "{}");
        assert_eq!(call.sent, "{}");
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// A refused item event flushes the buffered asserted call, whose own
/// refusal is kept, not raised in place of the first error. The finished
/// decoder ignores later events and hands the kept refusal to the terminal
/// transport error's flush, once.
#[test]
fn a_refused_event_keeps_its_flush_failure_for_the_transport_error() {
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        for frame in [
            malformed_function_call_done("completed", 0),
            text_delta("msg_1", 1, 1, "hi"),
        ] {
            assert!(matches!(
                feed(&mut decoder, &frame, reply.out())?,
                Flow::More
            ));
        }
        let error = feed(
            &mut decoder,
            &summary_delta("msg_1", 1, 2, "plan"),
            reply.out(),
        )
        .expect_err("reasoning cannot bind into a message");
        assert!(
            matches!(&error, ProviderError::Response(message)
                if message == "conflicting Responses parent kind at output 1: reasoning in a message"),
            "{error:?}"
        );
        assert!(matches!(
            feed(
                &mut decoder,
                &text_delta("msg_1", 1, 3, "late"),
                reply.out()
            ),
            Ok(Flow::More)
        ));
        let kept = decoder
            .flush_before_terminal_error(reply.out())
            .expect_err("the kept flush refusal");
        assert_malformed_add(&kept);
        decoder
            .flush_before_terminal_error(reply.out())
            .expect("the kept refusal is handed over once");
        Err(error)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Response(_))));
}

/// A provider `error` event and a refused `response.failed` terminal both
/// flush the buffered asserted call first; its refusal is kept for the
/// terminal transport error, after the provider's own failure.
#[test]
fn a_provider_failure_keeps_the_buffered_call_refusal() {
    let mut failed = sample_response("failed");
    failed["error"] = json!({"code":"server_error","message":"boom"});
    for terminal in [
        json!({"type":"error","sequence_number":1,"code":"server_error","message":"boom"}),
        json!({"type":"response.failed","sequence_number":1,"response":failed}),
    ] {
        let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
            let mut decoder = strict();
            assert!(matches!(
                feed(
                    &mut decoder,
                    &malformed_function_call_done("completed", 0),
                    reply.out()
                )?,
                Flow::More
            ));
            let error = feed(&mut decoder, &terminal, reply.out())
                .expect_err("the provider failed the reply");
            assert_eq!(
                error.to_string(),
                ProviderError::from_provider_body(terminal.to_string()).to_string()
            );
            let kept = decoder
                .flush_before_terminal_error(reply.out())
                .expect_err("the kept flush refusal");
            assert_malformed_add(&kept);
            Err(error)
        });
        assert!(result.outcome.is_err(), "{terminal}");
    }
}

/// A refused terminal's reasoning restatement that contradicts the streamed
/// part kind is kept as the flush refusal, not raised in place of the
/// provider's failure.
#[test]
fn a_contradictory_failed_reasoning_restatement_is_kept_for_the_transport_error() {
    let mut failed = sample_response("failed");
    failed["output"] = json!([{"type":"reasoning","id":"rs_1",
        "summary":[{"type":"future_summary","raw":[7]}]}]);
    let terminal = json!({"type":"response.failed","sequence_number":1,"response":failed});
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        assert!(matches!(
            feed(
                &mut decoder,
                &summary_delta("rs_1", 0, 0, "plan"),
                reply.out()
            )?,
            Flow::More
        ));
        let error =
            feed(&mut decoder, &terminal, reply.out()).expect_err("the provider failed the reply");
        assert_eq!(
            error.to_string(),
            ProviderError::from_provider_body(terminal.to_string()).to_string()
        );
        let kept = decoder
            .flush_before_terminal_error(reply.out())
            .expect_err("the kept merge refusal");
        assert!(
            matches!(&kept, ProviderError::Response(message)
                if message == "conflicting Responses content part kind at output 0, summary 0: stored text part restated as opaque"),
            "{kept:?}"
        );
        Err(error)
    });
    assert!(result.outcome.is_err());
}

#[test]
fn the_done_sentinel_is_not_the_provider_end() {
    let result = decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = strict();
        let WireEvent::Known(event) = decoder.classify(WireFrame::Text("[DONE]".to_owned())) else {
            panic!("the sentinel is a known payload");
        };
        assert!(matches!(event, ResponsesEvent::Sentinel));
        assert!(matches!(decoder.decode(event, reply.out())?, Flow::More));
        assert!(!decoder.finished);
        Err(ProviderError::Truncated)
    });
    assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
}

/// The lenient document reader reads the sentinel as such, and a success
/// body holding only an error envelope as the provider's failure.
#[test]
fn the_document_reader_classifies_the_sentinel_and_a_bare_error_envelope() {
    assert!(matches!(
        classify_document_payload(" [DONE] "),
        WireEvent::Known(ResponsesEvent::Sentinel)
    ));
    let envelope = json!({"error":{"code":"server_error","message":"boom"}}).to_string();
    assert!(matches!(
        classify_document_payload(&envelope),
        WireEvent::Known(ResponsesEvent::Failure(raw)) if raw == envelope
    ));
}
