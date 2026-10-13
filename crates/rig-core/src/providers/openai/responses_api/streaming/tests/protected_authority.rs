//! Protected final-versus-partial reasoning authority and late payload custody.

use super::helpers::*;
use super::super::{ResponsesDecoder, ResponsesStreamOptions};
use crate::driver::decode_with;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::operation::Turn;
use crate::streaming::{Item, PartKind, ReasoningAuthority, StreamEvent};
use crate::wire::{Decoder, Flow, WireEvent, WireFrame};
use serde_json::json;

// F streaming/tests.rs:4277
#[test]
fn reasoning_done_waits_for_late_terminal_parts_without_losing_its_position() {
    let summary=json!({"type":"future_summary","payload":[1]});
    let content=json!({"type":"future_content","payload":{"a":true}});
    let done=json!({"type":"reasoning","id":"rs_late","summary":[{"type":"summary_text","text":"known"}],"content":[],"signature":"signature"});
    let terminal_item=json!({"type":"reasoning","id":"rs_late","summary":[{"type":"summary_text","text":"known"},summary],"content":[content],"signature":"signature"});
    for with_delta in [false,true] {
        let decoded=decode_with(Turn::relayed("openai"),"openai",|reply| {
            let mut decoder=ResponsesDecoder::new("openai",ResponsesStreamOptions::strict());
            let mut frames=Vec::new();
            if with_delta {frames.push(json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}));}
            for sequence_number in [1,2] {frames.push(json!({"type":"response.output_item.done","output_index":0,"sequence_number":sequence_number,"item":done}));}
            for frame in frames {
                let WireEvent::Known(event)=decoder.classify(WireFrame::Text(frame.to_string())) else {panic!("protected valid frame")};
                assert!(matches!(decoder.decode(event,reply.out())?,Flow::More));
            }
            {
                let out=reply.out(); let shared=out.lock();
                // G164: a done-only Start is deferred until the flush.
                assert_eq!(shared.items.iter().any(|item|matches!(item,Ok(Item::Event(StreamEvent::Start {kind:PartKind::Reasoning,..})))), with_delta);
                assert!(!shared.items.iter().any(|item|matches!(item,Ok(Item::Event(StreamEvent::End {content:AssistantContent::CarriedReasoning(_),..})))));
                if with_delta {assert!(shared.items.iter().any(|item|matches!(item,Ok(Item::Event(StreamEvent::Reasoning {text,..})) if text=="known")));}
            }
            for frame in [
                json!({"type":"response.output_text.delta","item_id":"msg_late","output_index":1,"content_index":0,"sequence_number":3,"delta":"answer"}),
                completed_with_output(json!([terminal_item])),
            ] {
                let WireEvent::Known(event)=decoder.classify(WireFrame::Text(frame.to_string())) else {panic!("protected valid frame")};
                if matches!(decoder.decode(event,reply.out())?,Flow::Ended(_)) {return Ok(());}
            }
            panic!("fixture contains a genuine terminal");
        });
        assert_eq!(decoded.events().iter().filter(|event|matches!(event,StreamEvent::End {content:AssistantContent::CarriedReasoning(_),reasoning_authority:Some(ReasoningAuthority::ProviderRestated),..})).count(),1);
        let response=decoded.outcome.expect("genuine terminal");
        assert!(matches!(&response.choice[0],AssistantContent::CarriedReasoning(_)));
        assert_eq!(folded_texts(&response.choice),["answer"]);
        let replay=replay_opaque_choice(response.choice);
        assert_eq!(replay[0]["summary"],terminal_item["summary"]);
        assert_eq!(replay[0]["content"],terminal_item["content"]);
        assert_eq!(replay[0]["signature"],"signature");
    }
    let replay=replay_opaque_choice(unary_choice(json!([terminal_item])));
    assert_eq!(replay[0]["summary"],terminal_item["summary"]);
    assert_eq!(replay[0]["content"],terminal_item["content"]);
}

// F streaming/tests.rs:4355
#[test]
fn opaque_reasoning_partial_flush_keeps_content_and_truthful_wire_end() {
    let raw=json!({"type":"future_summary","payload":[1]});
    let item=json!({"type":"reasoning","id":"rs_partial","summary":[{"type":"summary_text","text":"known"},raw]});
    for with_done in [false,true] {
        for ending in ["eof","error","incomplete","failed"] {
            let mut frames=vec![
                json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_partial","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}),
                json!({"type":"response.reasoning_summary_part.done","item_id":"rs_partial","output_index":0,"summary_index":1,"sequence_number":1,"part":raw}),
            ];
            if with_done {frames.push(json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}));}
            if ending=="error" {frames.push(json!({"error":{"message":"failure","type":"server_error"}}));}
            if matches!(ending,"incomplete"|"failed") {
                let mut terminal=completed_with_output(json!([item]));
                terminal["type"]=json!(format!("response.{ending}"));
                terminal["response"]["status"]=json!(ending);
                frames.push(terminal);
            }
            let driven=drive(&frames);
            assert_eq!(driven.errors.len(),usize::from(ending!="eof"),"{ending}");
            assert!(driven.response.response_id().is_none(),"{ending}: partial flush proves no terminal");
            let ends=driven.items.iter().filter_map(|event|match event {
                Item::Event(StreamEvent::End {content:AssistantContent::CarriedReasoning(reasoning),reasoning_authority,..})=>Some((reasoning,*reasoning_authority)),_=>None,
            }).collect::<Vec<_>>();
            assert_eq!(ends.len(),1,"{ending}");
            assert_eq!(ends[0].1,Some(if with_done||matches!(ending,"incomplete"|"failed"){ReasoningAuthority::ProviderRestated}else{ReasoningAuthority::Partial}),"{ending}");
            assert_eq!(ends[0].0.content.len(),2);
            assert_eq!(ends[0].0.content[0],CarriedReasoningPart::Summary("known".into()));
            assert!(matches!(&ends[0].0.content[1],CarriedReasoningPart::OpaqueSummary(payload) if payload.item==raw));
        }
    }
}

// F streaming/tests.rs:4959
#[test]
fn an_opened_reasoning_block_closes_exactly_once_whatever_its_restatements() {
    let empty=json!({"type":"reasoning","id":"rs_deltas","summary":[],"content":[]});
    let delta=json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_deltas","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"});
    let done=json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":empty});
    let mut failed_response=sample_response("failed");
    failed_response["error"]=json!({"code":"server_error","message":"boom"});
    let failed=json!({"type":"response.failed","sequence_number":2,"response":failed_response});
    for (path,frames,restated) in [
        ("terminal",vec![delta.clone(),done.clone(),completed_with_output(json!([empty]))],true),
        ("partial eof",vec![delta.clone(),done.clone()],true),
        ("deltas only, eof",vec![delta.clone()],false),
        ("error flush",vec![delta.clone(),done.clone(),failed],true),
    ] {
        let driven=drive(&frames);
        let closes=driven.items.iter().filter_map(|event|match event {
            Item::Event(StreamEvent::End {content:AssistantContent::CarriedReasoning(reasoning),reasoning_authority,..})=>{
                assert_eq!(reasoning.id.as_deref(),Some("rs_deltas"));
                Some((reasoning.display_text(),*reasoning_authority))
            },_=>None,
        }).collect::<Vec<_>>();
        assert_eq!(closes,[("known".to_owned(),Some(if restated{ReasoningAuthority::ProviderRestated}else{ReasoningAuthority::Partial}))],"{path}");
    }
}

