//! Late bare signatures settle the reserved SDK media slot before later content.
//! Multi-frame transports here are synthetic adapter controls, not network streams.

use futures::{FutureExt as _, StreamExt as _};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, CarriedPosition, CarriedTextContent,
    DocumentSourceKind, ImageMediaType};
use rig_core::streaming::{Item, PartKind, StreamEvent};
use serde_json::{Value, json};
use super::super::{PROVIDER_NAME, rest_chunk};

type Frame = vertexai::model::GenerateContentResponse;
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn media(mime: &str) -> Value { json!({"inlineData":{"mimeType":mime,"data":"AA=="}}) }
fn signature(value: &str) -> Value { json!({"thoughtSignature":value}) }
fn body(parts: Vec<Value>, finish: Option<&str>) -> Value {
    let mut candidate = json!({"content":{"role":"model","parts":parts}});
    if let Some(finish) = finish { candidate["finishReason"] = json!(finish); }
    json!({"candidates":[candidate]})
}
fn frame(parts: Vec<Value>, finish: Option<&str>) -> Result<Frame, serde_json::Error> {
    // Actual SDK serde construction of new synthetic received fixtures only.
    serde_json::from_value(body(parts, finish))
}

#[derive(Clone)]
struct Scripted {
    frames: Vec<Result<Frame, ProviderError>>,
    document: Option<Value>,
}
impl Transport<crate::completion::GenerateContent> for Scripted {
    fn send(&self, _request:vertexai::model::GenerateContentRequest, _exchange:Exchange)
        -> Opening<Frame> {
        let opened = Opened::new(futures::stream::iter(self.frames.clone()));
        Opening::ready(match &self.document {
            Some(raw) => opened.with_document(raw.clone()),
            None => opened,
        })
    }
}
fn model(script:Scripted) -> Model<crate::completion::GenerateContent, Scripted> {
    Model::new(crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH), script)
}
struct Read {
    items: Vec<Result<Item<StreamEvent>, ProviderError>>,
    partial: CompletionResponse,
    outcome: Result<CompletionResponse, ProviderError>,
}
fn read(script: Scripted) -> Result<Read, ProviderError> {
    futures::executor::block_on(async move {
        let mut stream = model(script).stream(CompletionRequest::new("inspect"))?;
        let mut items = Vec::new();
        while let Some(item) = stream.next().await { items.push(item); }
        let partial = stream.partial();
        let outcome = stream.finish().await;
        Ok(Read { items, partial, outcome })
    })
}
fn whole(received: Frame) -> Result<Scripted, ProviderError> {
    let document = Value::Object(rest_chunk(&received)?);
    Ok(Scripted { frames:vec![Ok(received)], document:Some(document) })
}
fn ends(read:&Read) -> Vec<&AssistantContent> {
    read.items.iter().filter_map(|item| match item {
        Ok(Item::Event(StreamEvent::End { content, .. })) => Some(content),
        _ => None,
    }).collect()
}
fn assert_order(read:&Read, kinds:&[PartKind]) {
    let starts:Vec<_> = read.items.iter().filter_map(|item| match item {
        Ok(Item::Event(StreamEvent::Start { part, source_order, kind, .. })) =>
            Some((part.index(), *source_order, *kind)),
        _ => None,
    }).collect();
    assert_eq!(starts, kinds.iter().enumerate().map(|(index, kind)|
        (index, None, *kind)).collect::<Vec<_>>(), "genuine Part positions in SDK arrival order");
    let positions:Vec<_> = read.items.iter().filter_map(|item| match item {
        Ok(Item::Event(StreamEvent::End { part, .. })) => Some(part.index()),
        _ => None,
    }).collect();
    assert_eq!(positions, (0..kinds.len()).collect::<Vec<_>>());
}
fn assert_inspection(content:&AssistantContent, mime:&str, sig:&str, position:u64) {
    let AssistantContent::CarriedText(image) = content else {
        panic!("late signed image must be the real inspection carrier");
    };
    assert_eq!(image.content, CarriedTextContent::InspectionOnlyMedia(String::new()));
    let mut expected = media(mime);
    expected["thoughtSignature"] = json!(sig);
    assert_eq!(image.payload.item, expected);
    assert_eq!(image.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(image.payload.position, CarriedPosition::MessageContent {
        item_index:0, part_index:position,
    });
    assert!(image.signed_answer().is_none());
    assert!(content.native_item().is_none());
}
fn assert_image(content:&AssistantContent, mime:&str, expected:ImageMediaType) {
    let AssistantContent::Image(image) = content else { panic!("unsigned selected image"); };
    assert_eq!(image.media_type, Some(expected));
    assert_eq!(image.data, DocumentSourceKind::Base64("AA==".to_owned()));
    assert_eq!(content.native_item(), Some(&media(mime)));
}
fn assert_unsigned_answer(content:&AssistantContent, text:&str) {
    let AssistantContent::Text(answer) = content else { panic!("unsigned canonical answer"); };
    assert_eq!(answer.text, text);
    assert_eq!(content.native_item(), Some(&json!({"text":text})));
}

#[test]
fn g237_sdk_same_chunk_late_signature_keeps_media_not_the_next_answer() -> TestResult {
    let received = frame(vec![media("image/png"), signature("c2ln"), json!({"text":"answer"})], Some("STOP"))?;
    let script = whole(received)?;
    let raw = script.document.clone().ok_or("captured SDK document")?;
    assert_eq!(raw, body(vec![media("image/png"),signature("c2ln"),json!({"text":"answer"})],Some("STOP")));
    let unary = futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    let response = observed.outcome?;
    assert_eq!(response.choice, unary.choice);
    assert_eq!(response.raw, raw);
    assert_eq!(unary.raw, raw);
    assert_inspection(&response.choice[0], "image/png", "c2ln", 0);
    assert_unsigned_answer(&response.choice[1], "answer");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    Ok(())
}

#[test]
fn g237_sdk_split_chunks_retain_original_media_position_and_reconstructed_raw() -> TestResult {
    let script = Scripted { frames:vec![
        Ok(frame(vec![media("image/png")], None)?),
        Ok(frame(vec![signature("c2ln")], None)?),
        Ok(frame(vec![json!({"text":"answer"})], Some("STOP"))?),
    ], document:None };
    let unary = futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    let response = observed.outcome?;
    assert_eq!(response.choice, unary.choice);
    let mut signed = media("image/png"); signed["thoughtSignature"] = json!("c2ln");
    let expected = body(vec![signed,json!({"text":"answer"})],Some("STOP"));
    assert_eq!(response.raw, expected);
    assert_eq!(unary.raw, expected);
    assert_inspection(&response.choice[0],"image/png","c2ln",0);
    assert_unsigned_answer(&response.choice[1],"answer");
    Ok(())
}

#[test]
fn g237_adjacent_signature_metadata_keeps_the_last_signature_until_content() -> TestResult {
    let received = frame(vec![media("image/png"), signature("c2ln"),
        signature("dGFpbA=="), json!({"text":"answer"})], Some("STOP"))?;
    let script = whole(received)?;
    let raw = script.document.clone().ok_or("captured document")?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text,PartKind::Text]);
    let response = observed.outcome?;
    assert_inspection(&response.choice[0],"image/png","dGFpbA==",0);
    assert_unsigned_answer(&response.choice[1],"answer");
    assert_eq!(response.raw,raw);
    assert_eq!(response.raw["candidates"][0]["content"]["parts"],json!([
        media("image/png"),signature("c2ln"),signature("dGFpbA=="),{"text":"answer"}
    ]));
    Ok(())
}

#[test]
fn g237_all_five_unsigned_sdk_mimes_flush_on_next_content_or_eof() -> TestResult {
    for (mime, kind) in [("image/jpeg",ImageMediaType::JPEG),("image/png",ImageMediaType::PNG),
        ("image/webp",ImageMediaType::WEBP),("image/heic",ImageMediaType::HEIC),("image/heif",ImageMediaType::HEIF)] {
        for with_answer in [false,true] {
            let mut parts=vec![media(mime)];
            if with_answer {parts.push(json!({"text":"answer"}));}
            let script=whole(frame(parts,Some("STOP"))?)?;
            let raw=script.document.clone().ok_or("captured document")?;
            let observed=read(script)?;
            let mut kinds=vec![PartKind::Image];if with_answer {kinds.push(PartKind::Text);}
            assert_order(&observed,&kinds);
            let response=observed.outcome?;
            assert_eq!(response.choice.len(),if with_answer {2}else{1});
            assert_image(&response.choice[0],mime,kind.clone());
            if with_answer {assert_unsigned_answer(&response.choice[1],"answer");}
            assert_eq!(response.raw,raw);
            assert_eq!(response.finish_reason(),Some(FinishReason::Stop));
        }
    }
    Ok(())
}

#[test]
fn g237_all_five_late_signed_sdk_mimes_flush_at_eof_as_exact_inspection() -> TestResult {
    for mime in ["image/jpeg","image/png","image/webp","image/heic","image/heif"] {
        let script=whole(frame(vec![media(mime),signature("c2ln")],Some("STOP"))?)?;
        let raw=script.document.clone().ok_or("captured document")?;
        let observed=read(script)?;
        assert_order(&observed,&[PartKind::Text]);
        let response=observed.outcome?;
        assert_eq!(response.choice.len(),1);
        assert_inspection(&response.choice[0],mime,"c2ln",0);
        assert_eq!(response.raw,raw);
        assert_eq!(response.finish_reason(),Some(FinishReason::Stop));
    }
    Ok(())
}

#[test]
fn g237_pending_media_preserves_real_part_order_between_surrounding_content() -> TestResult {
    let script=whole(frame(vec![json!({"text":"before"}),media("image/png"),signature("c2ln"),
        json!({"text":"answer"}),media("image/webp"),json!({"text":"after"})],Some("STOP"))?)?;
    let raw=script.document.clone().ok_or("captured document")?;
    let observed=read(script)?;
    assert_order(&observed,&[PartKind::Text,PartKind::Text,PartKind::Text,PartKind::Image,PartKind::Text]);
    let response=observed.outcome?;
    assert_eq!(response.choice.len(),5);
    assert_unsigned_answer(&response.choice[0],"before");
    assert_inspection(&response.choice[1],"image/png","c2ln",1);
    assert_unsigned_answer(&response.choice[2],"answer");
    assert_image(&response.choice[3],"image/webp",ImageMediaType::WEBP);
    assert_unsigned_answer(&response.choice[4],"after");
    assert_eq!(response.raw,raw);
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_original_transport_error() -> TestResult {
    for signed in [false,true] {
        let mut frames=vec![Ok(frame(vec![media("image/png")],None)?)];
        if signed {frames.push(Ok(frame(vec![signature("c2ln")],None)?));}
        frames.push(Err(ProviderError::Response("g218-late scripted transport failure".to_owned())));
        let observed=read(Scripted {frames,document:None})?;
        assert_order(&observed,&[if signed {PartKind::Text}else{PartKind::Image}]);
        let parts=ends(&observed);
        assert_eq!(parts.len(),1);
        if signed {assert_inspection(parts[0],"image/png","c2ln",0);}
        else {assert_image(parts[0],"image/png",ImageMediaType::PNG);}
        assert!(matches!(observed.items.last(),Some(Err(ProviderError::Response(message)))
            if message=="g218-late scripted transport failure"));
        let mut expected=media("image/png");if signed {expected["thoughtSignature"]=json!("c2ln");}
        assert_eq!(observed.partial.raw,body(vec![expected],None));
        assert_eq!(observed.partial.choice.len(),1);
        assert!(matches!(&observed.outcome,Err(ProviderError::Response(message))
            if message=="g218-late scripted transport failure"));
    }
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_eof_truncation_without_inventing_stop() -> TestResult {
    for signed in [false,true] {
        let mut frames=vec![Ok(frame(vec![media("image/png")],None)?)];
        if signed {frames.push(Ok(frame(vec![signature("c2ln")],None)?));}
        let observed=read(Scripted {frames,document:None})?;
        assert_order(&observed,&[if signed {PartKind::Text}else{PartKind::Image}]);
        let parts=ends(&observed);assert_eq!(parts.len(),1);
        if signed {assert_inspection(parts[0],"image/png","c2ln",0);}
        else {assert_image(parts[0],"image/png",ImageMediaType::PNG);}
        assert!(matches!(observed.items.last(),Some(Err(ProviderError::Truncated))));
        assert!(matches!(observed.outcome,Err(ProviderError::Truncated)));
        assert_eq!(observed.partial.finish_reason(),None);
        let mut expected=media("image/png");if signed {expected["thoughtSignature"]=json!("c2ln");}
        assert_eq!(observed.partial.raw,body(vec![expected],None));
    }
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_known_sdk_tool_protocol_error() -> TestResult {
    for signed in [false,true] {
        let mut frames=vec![Ok(frame(vec![media("image/png")],None)?)];
        if signed {frames.push(Ok(frame(vec![signature("c2ln")],None)?));}
        let failed:Frame=serde_json::from_value(json!({"candidates":[{
            "finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"
        }]}))?;
        frames.push(Ok(failed));
        let observed=read(Scripted {frames,document:None})?;
        assert_order(&observed,&[if signed {PartKind::Text}else{PartKind::Image}]);
        let parts=ends(&observed);assert_eq!(parts.len(),1);
        if signed {assert_inspection(parts[0],"image/png","c2ln",0);}
        else {assert_image(parts[0],"image/png",ImageMediaType::PNG);}
        let expected="Gemini stopped with finish_reason=MalformedFunctionCall: bad call";
        assert!(matches!(observed.items.last(),Some(Err(ProviderError::Response(message))) if message==expected));
        assert!(matches!(&observed.outcome,Err(ProviderError::Response(message)) if message==expected));
        let mut part=media("image/png");if signed {part["thoughtSignature"]=json!("c2ln");}
        assert_eq!(observed.partial.raw,json!({"candidates":[{
            "content":{"role":"model","parts":[part]},
            "finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"
        }]}));
    }
    Ok(())
}

#[test]
fn g237_next_refused_file_content_cannot_discard_preceding_pending_media() -> TestResult {
    let file=json!({"fileData":{"mimeType":"image/png","fileUri":"gs://image"}});
    let script=whole(frame(vec![media("image/png"),file.clone()],Some("STOP"))?)?;
    let raw=script.document.clone().ok_or("captured document")?;
    let observed=read(script)?;
    assert_order(&observed,&[PartKind::Image]);
    assert_image(ends(&observed)[0],"image/png",ImageMediaType::PNG);
    let expected="Gemini response part kind fileData carries no assistant content rig can account for";
    assert!(matches!(observed.items.last(),Some(Err(ProviderError::Response(message))) if message==expected));
    assert!(matches!(observed.outcome,Err(ProviderError::Response(message)) if message==expected));
    assert_eq!(observed.partial.raw,raw);
    assert_eq!(raw["candidates"][0]["content"]["parts"][1],file);
    Ok(())
}

#[test]
fn g237_unselected_sdk_inline_media_keeps_immediate_inspection_and_exact_fields() -> TestResult {
    let mut thought=media("image/png");thought["thought"]=json!(true);
    let mut signed=media("image/png");signed["thoughtSignature"]=json!("c2ln");
    let mut extra=media("image/png");extra["futurePart"]=json!({"n":1});
    let mut nested=media("image/png");nested["inlineData"]["displayName"]=json!("original.png");
    for part in [thought,signed,extra,nested,media("audio/wav"),media("image/gif"),media("image/svg+xml")] {
        let script=whole(frame(vec![part.clone(),json!({"text":"answer"})],Some("STOP"))?)?;
        let raw=script.document.clone().ok_or("captured document")?;
        assert_eq!(raw["candidates"][0]["content"]["parts"][0],part);
        let observed=read(script)?;
        assert_order(&observed,&[PartKind::Text,PartKind::Text]);
        let response=observed.outcome?;
        let AssistantContent::CarriedText(image)=&response.choice[0] else {panic!("inspection path unchanged");};
        assert_eq!(image.content,CarriedTextContent::InspectionOnlyMedia(String::new()));
        assert_eq!(image.payload.item,part);
        assert_eq!(image.payload.provider.as_deref(),Some(PROVIDER_NAME));
        assert_eq!(image.payload.position,CarriedPosition::MessageContent {item_index:0,part_index:0});
        assert_unsigned_answer(&response.choice[1],"answer");
        assert_eq!(response.raw,raw);
    }
    Ok(())
}

#[derive(Clone)]
struct MetadataOnly(Vec<Frame>);
impl Transport<crate::completion::GenerateContent> for MetadataOnly {
    fn send(&self, _request:vertexai::model::GenerateContentRequest, _exchange:Exchange)
        -> Opening<Frame> {
        Opening::ready(Opened::new(futures::stream::iter(self.0.clone().into_iter().map(Ok))
            .chain(futures::stream::pending::<Result<Frame, ProviderError>>())))
    }
}

#[test]
fn g237_metadata_only_frames_do_not_publish_pending_media_before_content() -> TestResult {
    let received=MetadataOnly(vec![
        frame(vec![media("image/png")],None)?,
        frame(vec![signature("c2ln")],None)?,
        frame(vec![signature("dGFpbA==")],None)?,
    ]);
    let mut stream=Model::new(crate::completion::GenerateContent::new(
        crate::completion::GEMINI_2_5_FLASH),received).stream(CompletionRequest::new("inspect"))?;
    assert!(stream.next().now_or_never().is_none(),
        "actual driver consumed metadata and awaits next content without an emitted item");
    assert!(stream.partial().choice.is_empty());
    // Existing cancellation discards undelivered items. No parity with the
    // old immediate-image delivery is asserted for this deliberate delay.
    stream.cancel();
    assert!(stream.partial().choice.is_empty());
    assert!(matches!(futures::executor::block_on(stream.finish()),Err(ProviderError::Truncated)));
    Ok(())
}
