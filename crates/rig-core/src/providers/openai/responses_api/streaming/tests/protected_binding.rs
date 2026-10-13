//! Protected waiting-item binding, source order, and refusal before mutation.

use super::helpers::*;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::streaming::{Item, StreamEvent};
use crate::wire::Operation;
use serde_json::json;

// F streaming/tests.rs:4528
#[test]
fn an_unknown_part_before_parent_evidence_joins_the_item_that_evidence_names() {
    let unknown=json!({"type":"future_part","payload":{"list":[1]}});
    let part=json!({"type":"response.content_part.added","item_id":"rs_spelling_only","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    for parent in ["message","reasoning"] {
        let item=|content| if parent=="message" {message_item("rs_spelling_only",content)} else {json!({"type":"reasoning","id":"rs_spelling_only","summary":[],"content":content})};
        for evidence in ["added","done","terminal"] {
            let mut frames=vec![part.clone()];
            if evidence!="terminal" {frames.push(json!({"type":format!("response.output_item.{evidence}"),"output_index":0,"sequence_number":1,"item":item(json!([]))}));}
            frames.push(completed_with_output(if evidence=="terminal"{json!([item(json!([unknown]))])}else{json!([])}));
            let driven=drive(&frames);
            assert!(driven.errors.is_empty(),"{parent}/{evidence}: {:?}",driven.errors);
            assert!(unknown_payloads(&driven.items).is_empty(),"{parent}/{evidence}");
            assert_eq!(driven.response.choice.len(),1,"{parent}/{evidence}");
            if parent=="reasoning" {
                let AssistantContent::CarriedReasoning(reasoning)=&driven.response.choice[0] else {panic!("typed reasoning evidence owns this part")};
                assert!(matches!(reasoning.content.as_slice(),[CarriedReasoningPart::OpaqueContent(payload)] if payload.item==unknown));
            }else{assert!(matches!(&driven.response.choice[0],AssistantContent::CarriedText(_)));}
            assert_eq!(replay_opaque_choice(driven.response.choice)[0]["content"],json!([unknown]));
        }
    }
}

// F streaming/tests.rs:4604
#[test]
fn a_known_typed_part_binds_waiting_unknown_siblings_in_content_order() {
    let unknown=json!({"type":"future_part","payload":[1]});
    let driven=drive(&[
        json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":1,"delta":"B"}),
        completed_with_output(json!([])),
    ]);
    assert!(driven.errors.is_empty());
    assert_eq!(replay_opaque_choice(driven.response.choice)[0]["content"],json!([unknown,{"type":"output_text","text":"B"}]));
}

// F streaming/tests.rs:4626
#[tokio::test]
async fn unresolved_unknown_parts_leave_raw_before_the_flush_without_a_fabricated_choice() {
    use futures::StreamExt;
    let first=json!({"type":"future_part","payload":[1]});
    let second=json!({"type":"future_part","payload":[2]});
    let parts=[
        json!({"type":"response.content_part.added","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":1,"part":first}),
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":2,"part":second}),
    ];
    for ending in ["eof","error","failed","incomplete","completed"] {
        let mut frames=parts.to_vec();
        match ending {
            "error"=>frames.push(json!({"error":{"message":"failure","type":"server_error"}})),
            "failed"|"incomplete"=>{
                let mut terminal=completed_with_output(json!([]));
                terminal["type"]=json!(format!("response.{ending}"));terminal["response"]["status"]=json!(ending);frames.push(terminal);
            },
            "completed"=>frames.push(completed_with_output(json!([]))),_=>{},
        }
        let mut stream=responses_stream_of(&frames).await;
        let mut unknowns=Vec::new();let mut errors=0;
        while let Some(item)=stream.next().await {
            match item {
                Ok(Item::Unknown(raw))=>{
                    assert!(stream.partial().response_id().is_none(),"raw precedes delivered terminal proof");
                    unknowns.push(raw.value().clone());
                },
                Ok(Item::Event(StreamEvent::Start {..}))=>panic!("{ending}: no fabricated parent"),
                Ok(other)=>panic!("unexpected item: {other:?}"),Err(_)=>errors+=1,
            }
        }
        assert_eq!(errors,usize::from(!matches!(ending,"eof"|"completed")),"{ending}");
        assert_eq!(unknowns,[first.clone(),second.clone()],"{ending}");
        assert!(stream.partial().choice.is_empty());
        if ending=="completed" {assert!(stream.finish().await.is_ok());}
    }
}

// F streaming/tests.rs:5041
#[test]
fn a_terminal_restating_a_message_at_another_position_does_not_repeat_its_text() {
    let reasoning=json!({"type":"reasoning","id":"rs_1","summary":[]});
    let message=message_item("msg_1",json!([{"type":"output_text","text":"Hello"}]));
    for id_bearing in [true,false] {
        let mut delta=json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"});
        if id_bearing {delta["item_id"]=json!("msg_1");}
        let driven=drive(&[delta,completed_with_output(json!([reasoning,message]))]);
        assert!(driven.errors.is_empty());
        assert_eq!(folded_texts(&driven.response.choice),["Hello"],"id_bearing={id_bearing}");
    }
}

// F streaming/tests.rs:5169
#[test]
fn a_reasoning_slot_restated_as_the_other_kind_is_refused() {
    let unknown=json!({"type":"future_summary","payload":[1]});
    let known_delta=json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"});
    let unknown_part=|kind:&str|json!({"type":format!("response.reasoning_summary_part.{kind}"),"item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":1,"part":unknown});
    let known_done=json!({"type":"response.reasoning_summary_text.done","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":2,"text":"known"});
    let known_snapshot=json!({"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"type":"reasoning","id":"rs_k","summary":[{"type":"summary_text","text":"known"}]}});
    let undecided=json!({"type":"response.content_part.added","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"future_content","payload":[2]}});
    let reasoning_delta=json!({"type":"response.reasoning_text.delta","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":1,"delta":"text"});
    for (index,frames) in [
        vec![known_delta,unknown_part("done")],vec![unknown_part("added"),known_done],
        vec![unknown_part("done"),known_snapshot],vec![undecided,reasoning_delta],
    ].into_iter().enumerate() {
        let driven=drive(&frames);
        assert_eq!(driven.errors.len(),1,"case {index}");
        assert!(driven.errors[0].to_string().contains("conflicting Responses content part kind"));
        assert!(driven.response.response_id().is_none());
    }
}

// F streaming/tests.rs:5248
#[test]
fn responses_replay_refuses_text_beside_an_opaque_marker() {
    let raw=json!({"type":"future_part","payload":[1]});
    for id in [Some("msg_opaque".to_owned()),None] {
        let mut choice=unary_choice(json!([message_item(id.as_deref().unwrap_or(""),json!([raw]))]));
        let [AssistantContent::CarriedText(text)]=choice.as_mut_slice() else {panic!("one opaque text part")};
        text.item_id=id.clone();*text.text_mut()="replacement".into();
        let wire=crate::providers::openai::wire::OpenAIConfig::new("test-key").responses("different-model");
        let request=crate::completion::CompletionRequest::new("continue").message(crate::message::Message::from(choice));
        let result=crate::operation::Completion::prepare(request,&crate::wire::Wire::describe(&wire))
            .and_then(|request|crate::wire::Wire::encode(&wire,request,crate::wire::Mode::Unary).map_err(crate::error::ProviderError::from));
        let error=result.expect_err("text beside an opaque marker cannot replay");
        assert!(error.to_string().contains("an opaque Responses message part carries non-empty text"),"id={id:?}: {error}");
    }
}

// F streaming/tests.rs:5270
#[test]
fn reasoning_text_content_parts_are_known_reasoning_content() {
    let driven=drive(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":{"type":"reasoning","id":"rs_or","summary":[]}}),
        json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":""}}),
        json!({"type":"response.reasoning_text.delta","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":2,"delta":"think"}),
        json!({"type":"response.reasoning_text.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":3,"text":"think"}),
        json!({"type":"response.content_part.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":4,"part":{"type":"reasoning_text","text":"think"}}),
        completed_with_output(json!([{"type":"reasoning","id":"rs_or","summary":[],"content":[{"type":"reasoning_text","text":"think"}]}])),
    ]);
    assert!(driven.errors.is_empty());
    let [AssistantContent::CarriedReasoning(reasoning)]=driven.response.choice.as_slice() else {panic!("one reasoning item")};
    assert_eq!(reasoning.display_text(),"think");
    assert!(!reasoning.has_opaque_parts());
    let malformed=drive(&[json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"reasoning_text"}})]);
    assert!(!malformed.errors.is_empty(),"known tag without text is malformed");
}
