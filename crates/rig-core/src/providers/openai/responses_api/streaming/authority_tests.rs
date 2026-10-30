use super::{ResponsesDecoder, ResponsesStreamOptions};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::operation::{Completion, Turn};
use crate::streaming::{PartKind, ReasoningAuthority, StreamEvent, Transcript};
use crate::wire::{Decoder, Flow, WireEvent, WireFrame};
use serde_json::{Value, json};

fn reasoning() -> Value {
    json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thought"}],
        "content":[{"type":"future_reasoning","raw":[null,7]}],"encrypted_content":"cipher","status":"completed"})
}

fn response(output: Value) -> Value {
    json!({"id":"resp_1","object":"response","created_at":1,"model":"gpt-5.4","status":"completed","output":output})
}

fn delta() -> Value {
    json!({"type":"response.reasoning_summary_text.delta","sequence_number":0,"output_index":0,
        "summary_index":0,"item_id":"rs_1","delta":"thought"})
}

fn done() -> Value {
    json!({"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":reasoning()})
}

fn terminal(output: Value) -> Value {
    json!({"type":"response.completed","sequence_number":2,"response":response(output)})
}

fn decoded(frames: &[Value]) -> Decoded<Completion> {
    crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        "openai",
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string()))
    )
}

fn transport_flush(restated: bool) -> Decoded<Completion> {
    decode_with(Turn::relayed("openai"), "openai", |reply| {
        let mut decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
        let mut frames = vec![delta()];
        if restated {
            frames.push(done());
        }
        for frame in frames {
            let WireEvent::Known(event) = decoder.classify(WireFrame::Text(frame.to_string()))
            else {
                panic!("valid provider fixture");
            };
            assert!(matches!(decoder.decode(event, reply.out())?, Flow::More));
        }
        decoder.flush_before_terminal_error(reply.out())?;
        Err(ProviderError::Truncated)
    })
}

fn events(decoded: &Decoded<Completion>) -> Transcript {
    let items = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect::<Vec<_>>();
    Transcript::parse(serde_json::to_value(items).expect("actual decoder events"))
        .expect("all real parts close once")
}

fn authority(
    decoded: &Decoded<Completion>,
    expected: Option<ReasoningAuthority>,
) -> AssistantContent {
    let transcript = events(decoded);
    let start = transcript
        .events()
        .find(|event| {
            matches!(
                event,
                StreamEvent::Start {
                    kind: PartKind::Reasoning,
                    ..
                }
            )
        })
        .expect("actual reasoning start");
    let ends = transcript
        .events()
        .filter(|event| matches!(event, StreamEvent::End { .. }))
        .collect::<Vec<_>>();
    assert_eq!(ends.len(), 1);
    assert!(
        transcript
            .events()
            .all(|event| event.part() == start.part())
    );
    let StreamEvent::End {
        content,
        reasoning_authority,
        ..
    } = ends[0]
    else {
        panic!("actual end");
    };
    assert_eq!(*reasoning_authority, expected);
    content.clone()
}

#[test]
fn fragment_only_eof_generic_failure_and_transport_flush_keep_partial_authority() {
    let failure = json!({"type":"error","code":"failure","message":"exact failure"});
    for result in [
        decoded(&[delta()]),
        decoded(&[delta(), failure]),
        transport_flush(false),
    ] {
        assert!(
            result.outcome.is_err(),
            "a partial close is not a provider terminal"
        );
        let content = authority(&result, Some(ReasoningAuthority::Partial));
        let AssistantContent::CarriedReasoning(reasoning) = content else {
            panic!("preserved fragment reasoning");
        };
        assert_eq!(reasoning.display_text(), "thought");
        assert_eq!(
            reasoning.payload.item["summary"],
            json!([{"type":"summary_text","text":"thought"}])
        );
    }
}

#[test]
fn a_done_item_retains_provider_authority_through_eof_failure_and_transport_flush() {
    let failure = json!({"type":"error","code":"failure","message":"exact failure"});
    for result in [
        decoded(&[delta(), done()]),
        decoded(&[delta(), done(), failure]),
        transport_flush(true),
    ] {
        assert!(result.outcome.is_err());
        let content = authority(&result, Some(ReasoningAuthority::ProviderRestated));
        let AssistantContent::CarriedReasoning(carried) = content else {
            panic!("carried restatement");
        };
        assert_eq!(carried.payload.item, reasoning());
        assert_eq!(carried.id.as_deref(), Some("rs_1"));
    }
}

#[test]
fn whole_and_terminal_reasoning_restatements_keep_provider_authority_without_canonical_changes() {
    for frames in [
        vec![response(json!([reasoning()]))],
        vec![delta(), terminal(json!([reasoning()]))],
    ] {
        let result = decoded(&frames);
        let ended = authority(&result, Some(ReasoningAuthority::ProviderRestated));
        let AssistantContent::CarriedReasoning(ended) = ended else {
            panic!("carried reasoning");
        };
        assert_eq!(ended.payload.item, reasoning());
        let response = result.outcome.expect("genuine provider terminal");
        let [AssistantContent::CarriedReasoning(folded)] = response.choice.as_slice() else {
            panic!("same canonical content");
        };
        assert_eq!(folded.payload.item, ended.payload.item);
        // Delivered parts retain unknown issuer; the fold stamps its actual issuer.
        use crate::message::CarriedReasoningPart as Part;
        let (
            [
                Part::Summary(ended_summary),
                Part::OpaqueContent(ended_opaque),
                Part::Encrypted(ended_cipher),
            ],
            [
                Part::Summary(folded_summary),
                Part::OpaqueContent(folded_opaque),
                Part::Encrypted(folded_cipher),
            ],
        ) = (ended.content.as_slice(), folded.content.as_slice())
        else {
            panic!("exact received summary, opaque content, encrypted ordering");
        };
        assert_eq!(ended_summary, "thought");
        assert_eq!(folded_summary, ended_summary);
        assert_eq!(ended_cipher, "cipher");
        assert_eq!(folded_cipher, ended_cipher);
        assert_eq!(ended_opaque.item, reasoning()["content"][0]);
        assert_eq!(folded_opaque.item, ended_opaque.item);
        assert_eq!(
            ended_opaque.position,
            crate::message::CarriedPosition::ReasoningContent { part_index: 0 }
        );
        assert_eq!(folded_opaque.position, ended_opaque.position);
        assert_eq!(ended_opaque.provider, None);
        assert_eq!(folded_opaque.provider.as_deref(), Some("openai"));
        assert_eq!(ended.payload.provider, None);
        assert_eq!(ended.id.as_deref(), Some("rs_1"));
        assert_eq!(folded.id, ended.id);
        assert_eq!(folded.payload.provider.as_deref(), Some("openai"));
        assert!(
            matches!(response.message(), Some(crate::message::Message::Assistant(turn)) if turn.content == response.choice)
        );
    }
}

#[test]
fn refused_failed_terminal_reasoning_restatement_is_authoritative_without_a_response_end() {
    let mut failed = response(json!([reasoning()]));
    failed["status"] = json!("failed");
    failed["error"] = json!({"code":"failure","message":"exact failure"});
    let frame = json!({"type":"response.failed","sequence_number":2,"response":failed});
    let result = decoded(&[delta(), frame.clone()]);
    let ended = authority(&result, Some(ReasoningAuthority::ProviderRestated));
    let AssistantContent::CarriedReasoning(ended) = ended else {
        panic!("retained reasoning");
    };
    assert_eq!(ended.payload.item, reasoning());
    let error = result
        .outcome
        .expect_err("failed provider terminal remains refused");
    assert_eq!(
        error.provider_response_body(),
        Some(frame.to_string().as_str())
    );
}

#[test]
fn whole_top_level_reasoning_uses_fork_authority_without_carried_or_native_conversion() {
    let mut whole = response(json!([]));
    whole["reasoning"] = json!("top-level thought");
    let result = decoded(&[whole]);
    let ended = authority(&result, Some(ReasoningAuthority::ProviderRestated));
    assert_eq!(ended, AssistantContent::reasoning("top-level thought"));
    let response = result.outcome.expect("genuine whole response");
    assert_eq!(response.choice, vec![ended]);
}

#[test]
fn upstream_streamed_top_level_reasoning_fallback_keeps_default_none() {
    let mut frame = terminal(json!([]));
    frame["response"]["reasoning"] = json!("top-level thought");
    let result = decoded(&[frame]);
    let ended = authority(&result, None);
    assert_eq!(ended, AssistantContent::reasoning("top-level thought"));
    let response = result.outcome.expect("existing upstream terminal");
    assert_eq!(response.choice, vec![ended]);
}
