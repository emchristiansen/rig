use super::*;
use crate::completion::{CompletionRequest, CompletionResponse, ProviderResponseHeaders};
use crate::message::{AssistantContent, CarriedText, CarriedTextContent};
use crate::wire::{Call, Operation as _, Reply, Shared, Wire as _};
use base64::Engine as _;
use std::sync::Mutex;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn body(part: Value) -> Value {
    serde_json::json!({"candidates":[{"content":{"role":"model","parts":[part]},"finishReason":"STOP"}]})
}

fn inline(mime: &str) -> Value {
    serde_json::json!({"inlineData":{"mimeType":mime,"data":"AAEC/w=="}})
}

// Use the existing real classifier, decoder, writer and completion fold.
// The private selection is exercised here without adding another bridge caller.
fn selected(part: Value) -> Result<CompletionResponse, ProviderError> {
    let wire = super::super::completion::GenerateContent::new(
        super::super::GeminiConfig::new("test-key"),
        "gemini-2.5-flash",
    );
    let describe = wire.describe();
    let fold = Completion::fold(
        &CompletionRequest::new("inspect"),
        &mut Call::new(&describe, Mode::Unary),
    );
    let shared = Mutex::new(Shared::new(fold));
    let mut decoder = GenerateContentDecoder {
        vertex_inline_images: true,
        ..Default::default()
    };
    let fed = crate::driver::feed(
        &mut decoder,
        Some(wire.reassembler()),
        &shared,
        [WireFrame::Text(body(part).to_string())],
    );
    crate::driver::settle(
        shared,
        fed,
        Reply {
            provider: describe.name.to_owned(),
            raw: Value::Null,
            provider_request_id: None,
            response_headers: ProviderResponseHeaders::new(),
        },
    )
    .outcome
}

fn assert_inspection(response: &CompletionResponse, part: &Value) -> TestResult {
    let [AssistantContent::CarriedText(media)] = response.choice.as_slice() else {
        return Err("unselected inline media must retain its inspection carrier".into());
    };
    assert_inspection_fields(response, part, media);
    Ok(())
}

fn assert_inspection_fields(response: &CompletionResponse, part: &Value, media: &CarriedText) {
    assert_eq!(
        media.content,
        CarriedTextContent::InspectionOnlyMedia(String::new())
    );
    assert_eq!(&media.payload.item, part);
    assert_eq!(
        response.raw.pointer("/candidates/0/content/parts/0"),
        Some(part)
    );
    assert_eq!(
        media.payload.provider.as_deref(),
        Some(super::super::completion::PROVIDER_NAME)
    );
}

#[test]
fn closed_vertex_selection_accepts_each_of_the_five_image_mimes() {
    for (mime, expected) in [
        ("image/jpeg", ImageMediaType::JPEG),
        ("image/png", ImageMediaType::PNG),
        ("image/webp", ImageMediaType::WEBP),
        ("image/heic", ImageMediaType::HEIC),
        ("image/heif", ImageMediaType::HEIF),
    ] {
        for explicit_false in [false, true] {
            let mut part = inline(mime);
            if explicit_false {
                part["thought"] = Value::Bool(false);
            }
            let response = selected(part.clone()).expect("selected inline image decodes");
            let [content @ AssistantContent::Image(image)] = response.choice.as_slice() else {
                panic!("selected ordinary inline part must produce an image");
            };
            assert_eq!(image.media_type, Some(expected.clone()));
            let DocumentSourceKind::Base64(data) = &image.data else {
                panic!("selected image must retain base64 source");
            };
            assert_eq!(data, "AAEC/w==");
            assert_eq!(
                base64::prelude::BASE64_STANDARD
                    .decode(data)
                    .expect("selected image retains valid base64"),
                [0, 1, 2, 255]
            );
            assert_eq!(content.native_item(), Some(&part));
            assert_eq!(
                response.raw.pointer("/candidates/0/content/parts/0"),
                Some(&part)
            );
        }
    }
}

#[test]
fn default_rest_keeps_all_five_image_mimes_as_exact_inspection() {
    let wire = super::super::completion::GenerateContent::new(
        super::super::GeminiConfig::new("test-key"),
        "gemini-2.5-flash",
    );
    assert!(!GenerateContentDecoder::default().vertex_inline_images);
    for mime in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ] {
        let part = inline(mime);
        let response = crate::test_utils::history::decode(
            &wire,
            Mode::Unary,
            [WireFrame::Text(body(part.clone()).to_string())],
        )
        .expect("ordinary REST inline media decodes");
        assert_inspection(&response, &part).expect("inline media retains exact inspection");
    }
}

#[test]
fn vertex_selection_keeps_signed_and_thought_media_inspection() -> TestResult {
    for (key, value) in [
        ("thought", Value::Bool(true)),
        ("thought", Value::Null),
        ("thoughtSignature", Value::String("c2ln".to_owned())),
        ("thoughtSignature", Value::Null),
    ] {
        let mut part = inline("image/png");
        part[key] = value;
        assert_inspection(&selected(part.clone())?, &part)?;
    }
    Ok(())
}

#[test]
fn vertex_selection_keeps_audio_unknown_gif_and_svg_inspection() -> TestResult {
    for mime in [
        "audio/wav",
        "application/x-private",
        "image/gif",
        "image/svg+xml",
    ] {
        let part = inline(mime);
        assert_inspection(&selected(part.clone())?, &part)?;
    }
    Ok(())
}

#[test]
fn vertex_selection_keeps_root_extra_metadata_inspection() -> TestResult {
    let mut part = inline("image/png");
    part["futurePart"] = serde_json::json!({"exact":[0,1,2,255]});
    assert_inspection(&selected(part.clone())?, &part)
}

#[test]
fn vertex_selection_keeps_nested_display_name_inspection() -> TestResult {
    let mut part = inline("image/png");
    part["inlineData"]["displayName"] = Value::String("original.png".to_owned());
    assert_inspection(&selected(part.clone())?, &part)
}

#[test]
fn vertex_selection_keeps_other_nested_metadata_inspection() -> TestResult {
    let mut part = inline("image/png");
    part["inlineData"]["futureMetadata"] = serde_json::json!({"exact":"retained"});
    assert_inspection(&selected(part.clone())?, &part)
}

#[test]
fn vertex_selection_keeps_an_unselected_modeled_sibling_inspection() {
    let mut part = inline("image/png");
    // preserve_order keeps inlineData selected; the unselected modeled sibling
    // remains exact inspection evidence rather than becoming an image payload.
    part["functionCall"] = serde_json::json!({"unselected":[0,1,2,255]});
    assert_eq!(
        reader::select(part.as_object().expect("part object missing")),
        Some(reader::SelectedKind::InlineData)
    );
    assert_inspection(
        &selected(part.clone()).expect("unselected modeled sibling decodes"),
        &part,
    )
    .expect("unselected modeled sibling retains exact inspection");
}
