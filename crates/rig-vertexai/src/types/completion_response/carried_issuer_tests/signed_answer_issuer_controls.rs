//! Actual Vertex/REST answer custody is unchanged; only outbound foreign signatures drop.

use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, AssistantMessage, CarriedPayload, CarriedText,
    CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, Message, SourceLocation,
    UnrepresentableOpaqueContent};
use rig_core::operation::Completion;
use rig_core::providers::gemini::{GeminiConfig, completion::{GEMINI_2_5_FLASH, PROVIDER_NAME as REST_ISSUER}};
use rig_core::wire::{Mode, Operation as _, Wire as _};
use serde_json::{Value, json};
use super::super::{PROVIDER_NAME, rest_chunk};

const SIGNATURE: &str = "c2lnbmF0dXJl";
fn history(message:Message) -> CompletionRequest {
    CompletionRequest::from(vec![Message::user("q"),message,Message::user("next")])
}
#[derive(Clone)]
struct Received(vertexai::model::GenerateContentResponse);
impl Transport<crate::completion::GenerateContent> for Received {
    fn send(&self,_request:vertexai::model::GenerateContentRequest,_exchange:Exchange)
        -> Opening<vertexai::model::GenerateContentResponse> {
        let received=self.0.clone();
        match rest_chunk(&received) {
            Ok(document)=>Opening::ready(Opened::new(futures::stream::iter([Ok(received)]))
                .with_document(Value::Object(document))),
            Err(error)=>Opening::failed(error),
        }
    }
}
fn vertex_response() -> CompletionResponse {
    let part=vertexai::model::Part::new().set_text("4").set_thought_signature(b"signature".to_vec());
    let received=vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new().set_content(vertexai::model::Content::new()
            .set_role("model").set_parts([part]))
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
    ]);
    let raw=Value::Object(rest_chunk(&received).expect("actual SDK REST document"));
    let response=futures::executor::block_on(Model::new(
        crate::completion::GenerateContent::new(GEMINI_2_5_FLASH),Received(received))
        .call(CompletionRequest::new("What is 2 + 2?"))).expect("actual VertexDecoder/fold");
    assert_eq!(response.raw,raw); assert_received(&response,PROVIDER_NAME); response
}
fn rest_response() -> CompletionResponse {
    let raw=json!({"candidates":[{"content":{"role":"model","parts":[
        {"text":"4","thoughtSignature":SIGNATURE}
    ]},"finishReason":"STOP","citationMetadata":{"citationSources":[{
        "startIndex":0,"endIndex":1,"uri":"https://example.com/arith","title":"Arithmetic"
    }]}}]});
    let response=futures::executor::block_on(Model::new(GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH),
        rig_core::test_utils::RecordingHttpClient::new(serde_json::to_vec(&raw).expect("fixture JSON")))
        .call(CompletionRequest::new("What is 2 + 2?"))).expect("actual REST decoder/fold");
    assert_eq!(response.raw,raw); assert_received(&response,REST_ISSUER);
    let AssistantContent::CarriedText(text)=&response.choice[0] else {panic!("signed answer")};
    assert_eq!(text.citations().len(),1);
    assert_eq!(text.citations()[0].span.unwrap().range(),0..1);
    assert_eq!(text.citations()[0].sources[0].location,
        SourceLocation::Url{url:"https://example.com/arith".to_owned()});
    response
}
fn assert_received(response:&CompletionResponse,issuer:&str) {
    let [AssistantContent::CarriedText(text)]=response.choice.as_slice() else {panic!("one actual signed answer")};
    assert_eq!(text.payload.provider.as_deref(),Some(issuer));
    assert_eq!(text.payload.item,json!({"text":"4","thoughtSignature":SIGNATURE}));
    assert_eq!(text.signed_answer().unwrap().text,"4");
    assert_eq!(text.signed_answer().unwrap().signature.as_str(),SIGNATURE);
    assert_eq!(response.text(),"4");
}
fn vertex_encoded(message:Message,expected:Option<&[AssistantContent]>) -> vertexai::model::GenerateContentRequest {
    let wire=crate::completion::GenerateContent::new(GEMINI_2_5_FLASH);
    let prepared=Completion::prepare(history(message),&wire.describe()).expect("actual Vertex preparation");
    if let Some(expected)=expected {
        let Message::Assistant(turn)=&prepared.chat_history[1] else {panic!("foreign known answer remains a turn")};
        assert_eq!(turn.content.as_slice(),expected,"raw payload/content/citations stay intact before lowering");
    }
    wire.encode(prepared,Mode::Unary).expect("actual Vertex SDK request mapping")
}
fn assert_vertex_answer(request:&vertexai::model::GenerateContentRequest,signature:Option<&[u8]>) {
    assert_eq!(request.contents.len(),3);
    assert_eq!(request.contents[1].role,"model");
    assert_eq!(request.contents[1].parts.len(),1);
    let part=&request.contents[1].parts[0];
    assert_eq!(part.text().map(String::as_str),Some("4"));
    match signature {
        Some(signature)=>assert_eq!(part.thought_signature.as_ref(),signature),
        None=>assert!(part.thought_signature.is_empty(),"only outbound foreign signature is omitted"),
    }
}

#[test]
fn g227_actual_vertex_answer_keeps_its_genuine_signature_on_vertex() {
    let response=vertex_response(); let choice=response.choice.clone(); let raw=response.raw.clone();
    assert_vertex_answer(&vertex_encoded(response.message().unwrap(),Some(&choice)),Some(b"signature"));
    assert_eq!(response.raw,raw); assert_eq!(response.choice,choice);
}

#[test]
fn g227_actual_vertex_answer_to_rest_projects_known_text_without_foreign_signature() {
    let response=vertex_response(); let choice=response.choice.clone(); let raw=response.raw.clone();
    let wire=GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH);
    let prepared=Completion::prepare(history(response.message().unwrap()),&wire.describe()).expect("actual REST prepare");
    let Message::Assistant(turn)=&prepared.chat_history[1] else {panic!("known foreign answer is not dropped")};
    assert_eq!(turn.content,choice);
    let body=rig_core::test_utils::json_body(&wire.encode(prepared,Mode::Unary).expect("actual REST encode").request);
    assert_eq!(body["contents"].as_array().unwrap().len(),3);
    assert_eq!(body["contents"][1]["parts"],json!([{"text":"4","thought":false}]));
    assert_eq!(response.raw,raw); assert_eq!(response.choice,choice);
}

#[test]
fn g227_actual_rest_answer_to_vertex_projects_known_text_without_foreign_signature() {
    let response=rest_response(); let choice=response.choice.clone(); let raw=response.raw.clone();
    assert_vertex_answer(&vertex_encoded(response.message().unwrap(),Some(&choice)),None);
    assert_eq!(response.raw,raw); assert_eq!(response.choice,choice);
}

#[test]
fn g227_vertex_unknown_issuer_and_opaque_canonical_controls_keep_existing_routes() {
    // Explicit unissued canonical input, without changing any received issuer.
    let message=Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new("4",GeminiThoughtSignature::new(SIGNATURE))),
        CarriedPayload::new(Value::Null,None),
    ))]));
    assert_vertex_answer(&vertex_encoded(message,None),Some(b"signature"));
    let opaque=Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::Opaque(String::new()),CarriedPayload::new(json!({"futureField":{"n":1}}),None),
    ))]));
    let wire=crate::completion::GenerateContent::new(GEMINI_2_5_FLASH);
    let error=ProviderError::from(Completion::prepare(history(opaque),&wire.describe()).expect_err("opaque stays refused"));
    assert!(matches!(&error,ProviderError::Request(inner)
        if inner.downcast_ref::<UnrepresentableOpaqueContent>().is_some_and(|error|error.wire==wire.describe().name)));
}
