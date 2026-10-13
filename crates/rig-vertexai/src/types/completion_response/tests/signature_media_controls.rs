//! Vertex's real decoder distinguishes signed inspection media from unsigned images.
//! No late-signature-after-an-already-typed-image behavior is asserted here.

use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{AssistantContent, CarriedPosition, CarriedTextContent,
    DocumentSourceKind, ImageMediaType};
use serde_json::{Value, json};
use super::super::{PROVIDER_NAME, rest_chunk};

fn signature(bytes: &[u8]) -> vertexai::model::Part {
    vertexai::model::Part::new().set_thought_signature(bytes.to_vec())
}
fn media() -> vertexai::model::Part {
    vertexai::model::Part::new().set_inline_data(vertexai::model::Blob::new()
        .set_mime_type("image/png").set_data(vec![0]))
}
fn answer() -> vertexai::model::Part { vertexai::model::Part::new().set_text("answer") }
fn reply(parts: Vec<vertexai::model::Part>) -> vertexai::model::GenerateContentResponse {
    vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new().set_content(vertexai::model::Content::new()
            .set_role("model").set_parts(parts))
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
    ])
}

/// Mirror the production unary backend's existing with_document(rest_chunk)
/// capture, so raw comparisons concern received SDK JSON, not reconstruction.
#[derive(Clone)]
struct Received(vertexai::model::GenerateContentResponse);
impl Transport<crate::completion::GenerateContent> for Received {
    fn send(&self, _request:vertexai::model::GenerateContentRequest, _exchange:Exchange)
        -> Opening<vertexai::model::GenerateContentResponse> {
        let received = self.0.clone();
        match rest_chunk(&received) {
            Ok(document) => Opening::ready(Opened::new(futures::stream::iter([Ok(received)]))
                .with_document(Value::Object(document))),
            Err(error) => Opening::failed(error),
        }
    }
}
fn complete(received:vertexai::model::GenerateContentResponse) -> CompletionResponse {
    futures::executor::block_on(Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Received(received)).call(CompletionRequest::new("inspect")))
        .expect("actual SDK wire/VertexDecoder/fold")
}
fn assert_media_answer(response:&CompletionResponse, raw:&Value, signature:&str) {
    let [AssistantContent::CarriedText(image), AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("pending signed media must be inspection-only, not a typed image or signed answer");
    };
    assert_eq!(image.content, CarriedTextContent::InspectionOnlyMedia(String::new()));
    assert_eq!(image.payload.item, json!({"inlineData":{"mimeType":"image/png","data":"AA=="},
        "thoughtSignature":signature}));
    assert_eq!(image.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(image.payload.position, CarriedPosition::MessageContent { item_index:0, part_index:1 });
    assert!(image.signed_answer().is_none());
    assert_eq!(text.text, "answer");
    assert_eq!(response.choice[1].native_item(), Some(&json!({"text":"answer"})));
    assert_eq!(response.text(), "answer");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(&response.raw, raw, "the full original SDK document stays exact");
    assert!(!response.choice.iter().any(|part| matches!(part, AssistantContent::Image(_))));
}

#[test]
fn g218_vertex_leading_signature_belongs_to_inline_media_before_typed_selection() {
    let received = reply(vec![signature(b"sig"), media(), answer()]);
    let raw = Value::Object(rest_chunk(&received).expect("actual SDK REST spelling"));
    assert_eq!(raw["candidates"][0]["content"]["parts"], json!([
        {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}}, {"text":"answer"}
    ]));
    let response = complete(received);
    assert_media_answer(&response, &raw, "c2ln");
}

#[test]
fn g218_vertex_bare_signature_after_signed_inspection_targets_that_media_slot() {
    let received = reply(vec![signature(b"sig"), media(), signature(b"tail"), answer()]);
    let raw = Value::Object(rest_chunk(&received).expect("actual SDK REST spelling"));
    assert_eq!(raw["candidates"][0]["content"]["parts"], json!([
        {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}},
        {"thoughtSignature":"dGFpbA=="}, {"text":"answer"}
    ]));
    let response = complete(received);
    assert_media_answer(&response, &raw, "dGFpbA==");
}

#[test]
fn g218_vertex_unsigned_selected_inline_png_stays_typed() {
    let received = reply(vec![media(), answer()]);
    let raw = Value::Object(rest_chunk(&received).expect("actual SDK REST spelling"));
    let response = complete(received);
    let [content @ AssistantContent::Image(image), AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("ordinary unsigned Vertex PNG retains the current typed image selection");
    };
    assert_eq!(image.media_type, Some(ImageMediaType::PNG));
    assert_eq!(image.data, DocumentSourceKind::Base64("AA==".to_owned()));
    assert_eq!(content.native_item(), Some(&json!({"inlineData":{"mimeType":"image/png","data":"AA=="}})));
    assert_eq!(text.text, "answer");
    assert_eq!(response.choice[1].native_item(), Some(&json!({"text":"answer"})));
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.raw, raw);
}
