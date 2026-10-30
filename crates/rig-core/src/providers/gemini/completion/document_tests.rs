use super::*;
use crate::message::Document;
use crate::wire::Operation;

#[test]
fn test_txt_document_conversion_to_text_part() {
    let doc = UserContent::document_text(
        "Note: test.md\nPath: /test.md\nContent: Hello World!",
        Some(DocumentMediaType::TXT),
    );
    let part = user_part(doc, None).expect("TXT document encodes");
    let text = part["text"].as_str().expect("TXT is a text part");
    assert!(text.contains("Note: test.md"));
    assert!(text.contains("Hello World!"));
}

#[test]
fn test_markdown_document_conversion_to_text_part() {
    let doc = UserContent::document_text(
        "# Heading\n\n* List item",
        Some(DocumentMediaType::MARKDOWN),
    );
    let part = user_part(doc, None).expect("Markdown document encodes");
    assert_eq!(part["text"], "# Heading\n\n* List item");
}

#[test]
fn test_markdown_url_document_conversion_to_file_data_part() {
    let doc = UserContent::document_url(
        "https://generativelanguage.googleapis.com/v1beta/files/test-markdown",
        Some(DocumentMediaType::MARKDOWN),
    );
    let part = user_part(doc, None).expect("Markdown URL encodes");
    assert_eq!(
        part["fileData"]["fileUri"],
        "https://generativelanguage.googleapis.com/v1beta/files/test-markdown"
    );
    assert_eq!(part["fileData"]["mimeType"], "text/markdown");
}

#[test]
fn declared_text_like_base64_documents_decode_into_text() {
    use base64::Engine;
    for media_type in [
        DocumentMediaType::TXT,
        DocumentMediaType::RTF,
        DocumentMediaType::HTML,
        DocumentMediaType::CSS,
        DocumentMediaType::MARKDOWN,
        DocumentMediaType::CSV,
        DocumentMediaType::XML,
        DocumentMediaType::Javascript,
        DocumentMediaType::Python,
    ] {
        let doc = UserContent::document_base64(
            base64::engine::general_purpose::STANDARD.encode("Δ: exact text"),
            Some(media_type),
        );
        let part = user_part(doc, None).expect("text-like inline document encodes");
        assert_eq!(part, json!({"text": "Δ: exact text", "thought": false}));
    }
}

#[test]
fn invalid_original_document_fails_before_media_adaptation() {
    let wire = GenerateContent::new(super::super::GeminiConfig::new("k"), GEMINI_2_5_FLASH);
    for (data, diagnostic) in [
        ("!", "Failed to decode base64:"),
        ("/w==", "Invalid UTF-8 in document:"),
    ] {
        let request = CompletionRequest::new("question").message(Message::User {
            content: vec![UserContent::Document(Document {
                data: DocumentData::File(Source::Base64(data.to_owned())),
                media_type: Some(DocumentMediaType::HTML),
                additional_params: None,
            })],
        });
        let error = Completion::prepare(request, &wire.describe())
            .expect_err("an invalid declared document must not become ordinary text");
        assert!(error.to_string().contains(diagnostic), "{error:?}");
    }
}

#[test]
fn reported_usage_keeps_thought_and_tool_use_counts_separate() {
    let usage = usage_of(&json!({
        "promptTokenCount": 40, "cachedContentTokenCount": 20,
        "candidatesTokenCount": 30, "totalTokenCount": 100,
        "thoughtsTokenCount": 10, "toolUsePromptTokenCount": 12
    }));
    assert_eq!(usage.input_tokens, Some(40));
    assert_eq!(usage.cached_input_tokens, Some(20));
    assert_eq!(usage.output_tokens, Some(30));
    assert_eq!(usage.reasoning_tokens, Some(10));
    assert_eq!(usage.tool_use_prompt_tokens, Some(12));
    assert_eq!(usage.total_tokens, Some(100));
}

#[test]
fn text_documents_without_declared_media_type_keep_the_protected_refusal() {
    let part = UserContent::document_text("text", None);
    let error = user_part(part, None).expect_err("a document must declare its media type");
    assert!(
        error
            .to_string()
            .contains("A mime type is required for document inputs to Gemini")
    );
}
