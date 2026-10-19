//! Original PDF-source diagnostics through actual preflight and direct encoding.

use super::*;
use crate::message::{AssistantMessage, Document, MessageError};
use crate::providers::openai::wire::{Dialect, MISTRAL, OPENAI, OPENROUTER};
use crate::test_utils::json_body;
use crate::wire::Operation;

fn chat(dialect: &Dialect) -> Chat {
    OpenAIConfig::with_key(dialect, "test-key").chat("test-model")
}

fn pdf_sources() -> [(Source, &'static str); 2] {
    [
        (
            Source::Url("https://example.com/x.pdf".to_owned()),
            "OpenAI chat completions does not accept URL files; use the Responses API or pass base64-encoded bytes",
        ),
        (
            Source::Raw(b"%PDF-1.4\n".to_vec()),
            "Raw files not supported, encode as base64 first",
        ),
    ]
}

fn request(source: Source) -> CompletionRequest {
    CompletionRequest::new(Message::User {
        content: vec![UserContent::Document(Document {
            data: DocumentData::File(source),
            media_type: Some(DocumentMediaType::PDF),
            ..Default::default()
        })],
    })
}

fn assert_conversion(error: ProviderError, expected: &str) {
    let ProviderError::Request(source) = error else {
        panic!("expected request refusal: {error:?}");
    };
    match source
        .downcast_ref::<MessageError>()
        .expect("the original concrete conversion error")
    {
        MessageError::ConversionError(diagnostic) => assert_eq!(diagnostic, expected),
        MessageError::UnrepresentableOpaqueContent(_) | MessageError::CarriedTextReplay(_) => {
            panic!("expected PDF conversion diagnostic: {source:?}");
        }
    }
}

#[test]
fn original_pdf_sources_are_refused_before_adaptation() {
    assert_original_pdf_sources_are_refused_before_adaptation(&[&OPENAI, &MISTRAL]);
}

#[test]
fn openrouter_original_pdf_sources_are_refused_before_adaptation() {
    assert_original_pdf_sources_are_refused_before_adaptation(&[&OPENROUTER]);
}

fn assert_original_pdf_sources_are_refused_before_adaptation(dialects: &[&Dialect]) {
    for &dialect in dialects {
        let wire = chat(dialect);
        for (source, expected) in pdf_sources() {
            // URL must not become a placeholder and raw must not become sendable base64.
            let error = Completion::prepare(request(source), &wire.describe())
                .expect_err("the original source must be refused before adaptation");
            assert_conversion(error, expected);
        }
    }
}

#[test]
fn direct_encoding_retains_the_same_typed_pdf_diagnostics() {
    assert_direct_encoding_retains_the_same_typed_pdf_diagnostics(&[&OPENAI, &MISTRAL]);
}

#[test]
fn openrouter_direct_encoding_retains_the_same_typed_pdf_diagnostics() {
    assert_direct_encoding_retains_the_same_typed_pdf_diagnostics(&[&OPENROUTER]);
}

fn assert_direct_encoding_retains_the_same_typed_pdf_diagnostics(dialects: &[&Dialect]) {
    for &dialect in dialects {
        let wire = chat(dialect);
        for (source, expected) in pdf_sources() {
            let error = wire
                .encode(request(source), Mode::Unary)
                .expect_err("the direct user-part path must refuse the same original source");
            assert_conversion(error.into(), expected);
        }
    }
}

#[test]
fn native_custom_replay_keeps_its_original_raw_input() {
    let wire = chat(&OPENAI);
    let raw = "{ exact unparsed input";
    let frame = WireFrame::Text(
        json!({
            "object":"chat.completion.chunk", "id":"reply", "model":"test-model",
            "choices":[{"index":0,"delta":{"tool_calls":[{
                "index":0,"id":"native_call","type":"custom",
                "custom":{"name":"grep","input":raw}
            }]},"finish_reason":"tool_calls"}]
        })
        .to_string(),
    );
    let response = crate::test_utils::history::decode(
        &wire,
        Mode::Streaming,
        [frame, WireFrame::Text("[DONE]".to_owned())],
    )
    .expect("the existing U native custom path decodes");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!(
            "U native custom remains a native ToolCall: {:?}",
            response.choice
        );
    };
    assert_eq!(call.function.namespace, None);
    assert_eq!(
        call.native.as_ref().expect("actual decoder item").item["type"],
        "custom"
    );
    let request =
        CompletionRequest::new(Message::Assistant(AssistantMessage::new(response.choice)));
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("U native custom still replays");
    let body = json_body(&encoded.request);
    let call = &body["messages"][0]["tool_calls"][0];
    assert_eq!(call["type"], "custom");
    assert_eq!(call["id"], "native_call");
    assert_eq!(call["custom"]["name"], "grep");
    assert_eq!(call["custom"]["input"], raw);
}
