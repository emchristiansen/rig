//! Actual received signed answers retain input custody while replay uses its issuer.

use super::*;
use crate::completion::CompletionResponse;
use crate::driver::Model;
use crate::message::{
    AssistantMessage, CarriedPayload, CarriedText, CarriedTextContent, GeminiSignedAnswer,
    GeminiThoughtSignature, SourceLocation, UnrepresentableOpaqueContent,
};
use crate::wire::Operation as _;

const SIGNATURE: &str = "c2lnbmF0dXJl";

fn received() -> Value {
    json!({"candidates":[{"content":{"role":"model","parts":[
        {"text":"4","thoughtSignature":SIGNATURE}
    ]},"finishReason":"STOP","citationMetadata":{"citationSources":[{
        "startIndex":0,"endIndex":1,"uri":"https://example.com/arith","title":"Arithmetic"
    }]}}]})
}
fn actual_response() -> CompletionResponse {
    let raw = received();
    let response = futures::executor::block_on(
        Model::new(
            super::super::GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH),
            crate::test_utils::RecordingHttpClient::new(
                serde_json::to_vec(&raw).expect("fixture JSON"),
            ),
        )
        .call(CompletionRequest::new("What is 2 + 2?")),
    )
    .expect("the actual REST decoder/fold receives the signed answer");
    assert_eq!(response.raw, raw);
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("one real signed answer");
    };
    assert_eq!(text.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(
        text.payload.item,
        json!({"text":"4","thoughtSignature":SIGNATURE})
    );
    assert_eq!(text.signed_answer().unwrap().text, "4");
    assert_eq!(text.signed_answer().unwrap().signature.as_str(), SIGNATURE);
    assert_eq!(text.citations().len(), 1);
    assert_eq!(text.citations()[0].span.unwrap().range(), 0..1);
    assert_eq!(
        text.citations()[0].sources[0].location,
        SourceLocation::Url {
            url: "https://example.com/arith".to_owned()
        }
    );
    assert_eq!(
        text.citations()[0].sources[0].title.as_deref(),
        Some("Arithmetic")
    );
    response
}
fn history(message: Message) -> CompletionRequest {
    CompletionRequest::from(vec![Message::user("q"), message, Message::user("next")])
}
fn encoded(message: Message) -> Value {
    let wire = super::super::GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH);
    let prepared =
        Completion::prepare(history(message), &wire.describe()).expect("actual preparation");
    crate::test_utils::json_body(
        &wire
            .encode(prepared, Mode::Unary)
            .expect("actual encode")
            .request,
    )
}
fn unknown_answer() -> Message {
    // Explicit unissued canonical input, not a retagged provider reply.
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(
        CarriedText::new(
            CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
                "4",
                GeminiThoughtSignature::new(SIGNATURE),
            )),
            CarriedPayload::new(Value::Null, None),
        ),
    )]))
}

#[test]
fn g227_actual_rest_signed_answer_keeps_genuine_signature_content_raw_and_citations() {
    let response = actual_response();
    let before_raw = response.raw.clone();
    let before_choice = response.choice.clone();
    let message = response.message().expect("actual received turn");
    let wire = super::super::GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH);
    let prepared = Completion::prepare(history(message), &wire.describe()).expect("real prepare");
    let Message::Assistant(turn) = &prepared.chat_history[1] else {
        panic!("known answer turn is retained");
    };
    assert_eq!(turn.content, before_choice);
    let body = crate::test_utils::json_body(
        &wire
            .encode(prepared, Mode::Unary)
            .expect("real encode")
            .request,
    );
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"text":"4","thought":false,"thoughtSignature":SIGNATURE}])
    );
    assert_eq!(response.raw, before_raw);
    assert_eq!(response.choice, before_choice);
}

#[test]
fn g227_unknown_issuer_signed_answer_keeps_the_existing_none_route() {
    let message = unknown_answer();
    let Message::Assistant(turn) = &message else {
        unreachable!()
    };
    let AssistantContent::CarriedText(text) = &turn.content[0] else {
        unreachable!()
    };
    assert_eq!(text.payload.provider, None);
    assert_eq!(text.payload.item, Value::Null);
    let body = encoded(message);
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"text":"4","thought":false,"thoughtSignature":SIGNATURE}])
    );
}

#[test]
fn g227_opaque_carried_text_keeps_the_existing_typed_refusal() {
    // Explicit unissued opaque canonical input claims no provider authority.
    let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(
        CarriedText::new(
            CarriedTextContent::Opaque(String::new()),
            CarriedPayload::new(json!({"futureField":{"n":1}}), None),
        ),
    )]));
    let wire = super::super::GeminiConfig::new("test-key").completion(GEMINI_2_5_FLASH);
    let error = Completion::prepare(history(message), &wire.describe())
        .expect_err("opaque carried content is not a known signed answer");
    assert!(matches!(&error, ProviderError::Request(inner)
        if inner.downcast_ref::<UnrepresentableOpaqueContent>().is_some()));
}
