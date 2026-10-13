//! Direct raw-selection tests keep their preexisting immediate delivery.

use super::*;
use crate::completion::{CompletionRequest, ProviderResponseHeaders};
use crate::message::AssistantContent;
use crate::streaming::{Item, StreamEvent};
use crate::wire::{Call, Operation as _, Reply, Shared, Wire as _};
use serde_json::json;
use std::sync::Mutex;

fn body(part:Value, terminal:bool) -> Value {
    let mut candidate=json!({"content":{"role":"model","parts":[part]}});
    if terminal {candidate["finishReason"]=json!("STOP");}
    json!({"candidates":[candidate]})
}

#[test]
fn g237_direct_raw_vertex_selection_remains_immediate_before_corrupt_or_unknown() -> Result<(), Box<dyn std::error::Error>> {
    for corrupt in [false,true] {
        let wire=super::super::completion::GenerateContent::new(
            super::super::GeminiConfig::new("test-key"),"gemini-2.5-flash");
        let describe=wire.describe();
        let fold=Completion::fold(&CompletionRequest::new("inspect"),
            &mut Call::new(&describe,Mode::Streaming));
        let shared=Mutex::new(Shared::new(fold));
        let mut decoder=GenerateContentDecoder {vertex_inline_images:true,..Default::default()};
        assert!(!decoder.defer_vertex_inline_images);
        let image=json!({"inlineData":{"mimeType":"image/png","data":"AA=="}});
        let middle=if corrupt {body(json!({"futureField":{"n":1},"thought":"yes"}),false)}
            else {json!({"futureFrame":{"n":1}})};
        let fed=crate::driver::feed(&mut decoder,Some(wire.reassembler()),&shared,[
            WireFrame::Text(body(image.clone(),false).to_string()),
            WireFrame::Text(middle.clone().to_string()),
            WireFrame::Text(body(json!({"text":"answer"}),true).to_string()),
        ]);
        let decoded=crate::driver::settle(shared,fed,Reply {
            provider:describe.name.to_owned(),raw:Value::Null,provider_request_id:None,
            response_headers:ProviderResponseHeaders::new(),
        });
        assert!(matches!(&decoded.items[0],Ok(Item::Event(StreamEvent::Start {part,kind:crate::streaming::PartKind::Image,..})) if part.index()==0));
        let Ok(Item::Event(StreamEvent::End {part,content,..}))=&decoded.items[1] else {panic!("immediate real image end");};
        assert_eq!(part.index(),0);
        assert!(matches!(content,AssistantContent::Image(_)));
        assert_eq!(content.native_item(),Some(&image));
        if corrupt {
            let Some(Err(ProviderError::CorruptFrame(error)))=decoded.items.last() else {panic!("exact current classifier refusal");};
            assert_eq!(error.evidence(),&crate::error::FrameEvidence::Text(middle.to_string()));
            assert!(error.error().to_string().contains("expected a boolean"));
            assert!(matches!(decoded.outcome,Err(ProviderError::CorruptFrame(_))));
        } else {
            let Ok(Item::Unknown(value))=&decoded.items[2] else {panic!("unknown remains after emitted image");};
            assert_eq!(value.value(),&middle);
            let response=decoded.outcome?;
            assert!(matches!(&response.choice[0],AssistantContent::Image(_)));
            assert_eq!(response.text(),"answer");
        }
    }
    Ok(())
}
