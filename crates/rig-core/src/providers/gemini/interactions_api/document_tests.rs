use super::*;
use crate::message::Document;
use crate::wire::{Operation, Wire};
use base64::Engine;

#[test]
fn text_document_inputs_preserve_text_and_utf8_errors() {
    for data in [
        DocumentData::Text("Δ: exact text".to_owned()),
        DocumentData::File(Source::Raw("Δ: exact text".as_bytes().to_vec())),
        DocumentData::File(Source::Base64(base64::engine::general_purpose::STANDARD.encode("Δ: exact text"))),
    ] {
        let document = Document { data, media_type: Some(DocumentMediaType::TXT), additional_params: None };
        assert_eq!(document_content(&document).expect("text document encodes"),
            json!({"type": "text", "text": "Δ: exact text"}));
    }
    for (data, detail) in [
        (DocumentData::File(Source::Raw(vec![255])), "Text document data must be UTF-8:"),
        (DocumentData::File(Source::Base64("/w==".to_owned())), "Text document data must be UTF-8:"),
        (DocumentData::File(Source::Base64("!".to_owned())), "Failed to decode text document base64 data:"),
    ] {
        let document = Document { data, media_type: Some(DocumentMediaType::TXT), additional_params: None };
        assert!(document_content(&document).expect_err("invalid document is refused").to_string().contains(detail));
    }
}

#[test]
fn non_text_documents_keep_declared_media_and_exact_sources() {
    for media_type in [DocumentMediaType::PDF, DocumentMediaType::HTML, DocumentMediaType::MARKDOWN] {
        for (data, key, value) in [
            (DocumentData::Text("raw bytes".to_owned()), "data", "cmF3IGJ5dGVz"),
            (DocumentData::File(Source::Raw(b"raw bytes".to_vec())), "data", "cmF3IGJ5dGVz"),
            (DocumentData::File(Source::Base64("already-encoded".to_owned())), "data", "already-encoded"),
            (DocumentData::File(Source::Url("https://example.org/file".to_owned())), "uri", "https://example.org/file"),
        ] {
            let document = Document { data, media_type: Some(media_type.clone()), additional_params: None };
            let content = document_content(&document).expect("declared document encodes");
            assert_eq!(content["type"], "document");
            assert_eq!(content["mime_type"], media_type.to_mime_type());
            assert_eq!(content[key], value);
        }
    }
}

#[test]
fn invalid_original_documents_are_refused_before_downgrade() {
    let wire = Interactions::new(super::super::GeminiConfig::new("k"), "gemini-2.5-flash");
    for (data, media_type, detail) in [
        (DocumentData::Text("text".to_owned()), None, "Media type for document is required for Gemini"),
        (DocumentData::File(Source::Url("https://example.org/file".to_owned())), Some(DocumentMediaType::TXT), "Text document URLs are not supported for Gemini Interactions inputs"),
        (DocumentData::File(Source::FileId("file-1".to_owned())), Some(DocumentMediaType::PDF), "Provider file IDs are not supported for Gemini Interactions inputs"),
        (DocumentData::File(Source::Unknown), Some(DocumentMediaType::PDF), "Unknown content source"),
    ] {
        let request = CompletionRequest::new("question").message(Message::User {
            content: vec![UserContent::Document(Document { data, media_type, additional_params: None })],
        });
        let error = crate::operation::Completion::prepare(request, &wire.describe())
            .expect_err("invalid declared original document must fail before adaptation");
        assert!(error.to_string().contains(detail), "{error:?}");
    }
}

#[test]
fn signature_only_thought_omits_an_empty_summary_array() {
    use crate::message::{AssistantContent, CarriedPayload, CarriedReasoning, CarriedReasoningPart};
    let target = Interactions::new(super::super::GeminiConfig::new("k"), "gemini-2.5-flash");
    for signature in ["", "exact-signature"] {
        let content = AssistantContent::CarriedReasoning(CarriedReasoning::new(
            vec![CarriedReasoningPart::Text { text: String::new(), signature: Some(signature.to_owned()) }],
            CarriedPayload::new(json!({"type":"thought", "signature":signature}), Some("gcp.gemini".to_owned())),
        ));
        assert_eq!(assistant_step(&content, &target, None).expect("signature-only thought encodes"),
            Some(json!({"type":"thought", "signature":signature})));
    }
}
