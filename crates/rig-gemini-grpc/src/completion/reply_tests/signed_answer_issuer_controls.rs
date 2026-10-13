//! Real REST/protobuf received answers replay through the selected GenerateContent issuer.

use crate::{completion::{GenerateContent, GEMINI_2_5_FLASH}, proto};
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::Model;
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, AssistantMessage, CarriedPayload, CarriedText,
    CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, Message, SourceLocation, UnrepresentableOpaqueContent};
use rig_core::operation::Completion;
use rig_core::providers::gemini::{GeminiConfig, completion::PROVIDER_NAME as ISSUER};
use rig_core::wire::{Mode, Operation as _, Wire as _};
use serde_json::{Value, json};

const SIGNATURE: &str = "c2lnbmF0dXJl";
fn history(message:Message) -> CompletionRequest {
    CompletionRequest::from(vec![Message::user("q"), message, Message::user("next")])
}
fn grpc_response() -> CompletionResponse {
    let received = proto::GenerateContentResponse { candidates:vec![proto::Candidate {
        content:Some(proto::Content { role:"model".to_owned(), parts:vec![proto::Part {
            data:Some(proto::part::Data::Text("4".to_owned())), thought_signature:b"signature".to_vec(),
            ..Default::default()
        }] }), finish_reason:proto::candidate::FinishReason::Stop as i32,
        citation_metadata:Some(proto::CitationMetadata { citation_sources:vec![proto::CitationSource {
            start_index:Some(0), end_index:Some(1), uri:Some("https://example.com/arith".to_owned()),
            license:Some("test-license".to_owned()),
        }] }), ..Default::default()
    }], ..Default::default() };
    let raw = crate::rest::to_rest(&received).expect("actual protobuf document");
    let response = crate::completion::tests::complete(received).expect("actual GrpcAdapter/Model/fold");
    assert_eq!(response.raw, raw);
    assert_received(&response);
    response
}
fn rest_response() -> CompletionResponse {
    let raw = json!({"candidates":[{"content":{"role":"model","parts":[
        {"text":"4","thoughtSignature":SIGNATURE}
    ]},"finishReason":"STOP","citationMetadata":{"citationSources":[{
        "startIndex":0,"endIndex":1,"uri":"https://example.com/arith"
    }]}}]});
    let response = futures::executor::block_on(Model::new(rig_core::providers::gemini::completion::GenerateContent::new(GeminiConfig::new("test-key"), GEMINI_2_5_FLASH),
        rig_core::test_utils::RecordingHttpClient::new(serde_json::to_vec(&raw).expect("fixture JSON")))
        .call(CompletionRequest::new("q"))).expect("actual REST response");
    assert_eq!(response.raw, raw); assert_received(&response); response
}
fn assert_received(response:&CompletionResponse) {
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else { panic!("one real signed answer"); };
    assert_eq!(text.payload.provider.as_deref(), Some(ISSUER));
    assert_eq!(text.payload.item, json!({"text":"4","thoughtSignature":SIGNATURE}));
    assert_eq!(text.signed_answer().unwrap().text, "4");
    assert_eq!(text.signed_answer().unwrap().signature.as_str(), SIGNATURE);
    assert_eq!(text.citations().len(), 1);
    assert_eq!(text.citations()[0].span.unwrap().range(), 0..1);
    assert_eq!(text.citations()[0].sources[0].location,
        SourceLocation::Url { url:"https://example.com/arith".to_owned() });
}
fn encode_grpc(message:Message, expected:Option<&[AssistantContent]>) -> proto::GenerateContentRequest {
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let prepared = Completion::prepare(history(message), &wire.describe()).expect("actual gRPC prepare");
    if let Some(expected) = expected {
        let Message::Assistant(turn) = &prepared.chat_history[1] else { panic!("answer turn retained"); };
        assert_eq!(turn.content.as_slice(), expected, "received raw/content/citations do not change");
    }
    wire.encode(prepared, Mode::Unary).expect("actual protobuf request lowering")
}
fn assert_grpc_answer(request:&proto::GenerateContentRequest) {
    assert_eq!(request.contents.len(), 3);
    assert_eq!(request.contents[1].role, "model");
    assert_eq!(request.contents[1].parts.len(), 1);
    let part = &request.contents[1].parts[0];
    assert!(matches!(&part.data, Some(proto::part::Data::Text(text)) if text=="4"));
    assert!(!part.thought);
    assert_eq!(part.thought_signature, b"signature".to_vec());
}

#[test]
fn g227_actual_grpc_answer_replays_to_grpc_using_selected_issuer_not_transport_name() {
    let response = grpc_response();
    assert_ne!(GenerateContent::new(GEMINI_2_5_FLASH).describe().name, ISSUER,
        "the protobuf transport name and shared selected issuer differ");
    let before_raw=response.raw.clone(); let before_choice=response.choice.clone();
    let request=encode_grpc(response.message().unwrap(), Some(&before_choice));
    assert_grpc_answer(&request);
    assert_eq!(response.raw,before_raw); assert_eq!(response.choice,before_choice);
}

#[test]
fn g227_actual_grpc_answer_replays_to_rest_with_genuine_shared_issuer() {
    let response=grpc_response(); let before=response.choice.clone(); let raw=response.raw.clone();
    let wire=rig_core::providers::gemini::completion::GenerateContent::new(GeminiConfig::new("test-key"), GEMINI_2_5_FLASH);
    let prepared=Completion::prepare(history(response.message().unwrap()),&wire.describe()).expect("actual REST prepare");
    let Message::Assistant(turn)=&prepared.chat_history[1] else {panic!("answer retained")};
    assert_eq!(turn.content,before);
    let encoded=wire.encode(prepared,Mode::Unary).expect("actual REST encode");
    let rig_core::wire::Body::Bytes(bytes)=encoded.request.body() else {panic!("actual REST byte body")};
    let body:Value=serde_json::from_slice(bytes).expect("actual encoded JSON");
    assert_eq!(body["contents"].as_array().unwrap().len(),3);
    assert_eq!(body["contents"][1]["parts"],json!([{"text":"4","thought":false,"thoughtSignature":SIGNATURE}]));
    assert_eq!(response.raw,raw); assert_eq!(response.choice,before);
}

#[test]
fn g227_actual_rest_answer_replays_to_grpc_with_genuine_shared_issuer() {
    let response=rest_response(); let before=response.choice.clone(); let raw=response.raw.clone();
    assert_grpc_answer(&encode_grpc(response.message().unwrap(),Some(&before)));
    assert_eq!(response.raw,raw); assert_eq!(response.choice,before);
}

#[test]
fn g227_grpc_unknown_issuer_and_opaque_canonical_controls_keep_existing_routes() {
    // Explicit unissued canonical input; no received provenance is changed.
    let message=Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new("4",GeminiThoughtSignature::new(SIGNATURE))),
        CarriedPayload::new(Value::Null,None),
    ))]));
    assert_grpc_answer(&encode_grpc(message,None));
    let opaque=Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::Opaque(String::new()),CarriedPayload::new(json!({"futureField":{"n":1}}),None),
    ))]));
    let wire=GenerateContent::new(GEMINI_2_5_FLASH);
    let error=ProviderError::from(Completion::prepare(history(opaque),&wire.describe()).expect_err("opaque remains refused"));
    assert!(matches!(&error,ProviderError::Request(inner)
        if inner.downcast_ref::<UnrepresentableOpaqueContent>().is_some_and(|error|error.wire==wire.describe().name)));
}
