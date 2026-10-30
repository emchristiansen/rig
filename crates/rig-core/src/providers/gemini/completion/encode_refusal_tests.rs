//! Encoder refusal arms and the Vertex tool-result image lowering, called on
//! the private encoders directly: no recorded exchange reaches them.

use super::*;
use crate::message::{
    AnsweredToolCall, CallId, Document, Image, ImageMediaType, ToolName, ToolResult,
    UnrepresentableToolCall,
};
use crate::provider_response::ProviderResponseError;
use serde_json::json;

const MODEL: &str = "gemini-2.5-flash";

fn target() -> GenerateContent {
    GenerateContent::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        MODEL,
    )
}

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the fixture name is nonempty")
}

fn result(answers: AnsweredToolCall, content: Vec<ToolResultContent>) -> ToolResult {
    ToolResult {
        call: CallId::from_wire("call-1"),
        name: name("inspect"),
        answers,
        content,
        is_error: false,
    }
}

/// The contents `style` encodes for one user message holding `result`.
fn encoded(result: ToolResult, style: ToolResultStyle) -> Result<Vec<Value>, EncodeError> {
    let history = vec![Message::User {
        content: vec![UserContent::ToolResult(result)],
    }];
    contents_in(history, &target(), MODEL, style, PROVIDER_NAME)
}

fn image(data: Source, media_type: Option<ImageMediaType>) -> ToolResultContent {
    ToolResultContent::Image(Image {
        data,
        media_type,
        ..Image::default()
    })
}

#[test]
fn custom_tool_results_are_refused_on_both_tool_result_styles() {
    // Vertex lowers a successful result itself and hands a failed one to the
    // shared lowering, so each style refuses through both encoders.
    for (style, is_error) in [
        (ToolResultStyle::GenerateContent, false),
        (ToolResultStyle::GenerateContent, true),
        (ToolResultStyle::Vertex, false),
        (ToolResultStyle::Vertex, true),
    ] {
        let mut custom = result(
            AnsweredToolCall::Custom,
            vec![ToolResultContent::text("patched")],
        );
        custom.is_error = is_error;
        let error = ProviderError::from(
            encoded(custom, style).expect_err("a custom result is refused, not lowered"),
        );
        assert!(matches!(&error, ProviderError::Request(_)), "{error:?}");
        let refusal = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<UnrepresentableToolCall>())
            .expect("the request failure keeps the typed refusal");
        assert_eq!(
            refusal,
            &UnrepresentableToolCall::CustomCall {
                wire: "Gemini GenerateContent",
                name: name("inspect"),
            },
            "is_error: {is_error}"
        );
    }
}

#[test]
fn vertex_image_names_skip_refs_reserved_inside_json_arrays() {
    let success = result(
        AnsweredToolCall::Function,
        vec![
            ToolResultContent::json(json!({"items": [{"$ref": "rig_tool_result_image_0"}, 3]})),
            image(Source::Base64("aGk=".to_owned()), Some(ImageMediaType::PNG)),
        ],
    );
    let contents = encoded(success, ToolResultStyle::Vertex).expect("a Vertex success encodes");
    assert_eq!(
        contents,
        vec![json!({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": "inspect",
                    "response": {"output": [
                        {"items": [{"$ref": "rig_tool_result_image_0"}, 3]},
                        {"$ref": "rig_tool_result_image_1"}
                    ]},
                    "parts": [{"inlineData": {
                        "mimeType": "image/png", "data": "aGk=",
                        "displayName": "rig_tool_result_image_1"
                    }}]
                },
                "thought": false
            }]
        })]
    );
}

#[test]
fn vertex_tool_result_images_refuse_missing_types_bad_base64_and_unsendable_sources() {
    use base64::Engine as _;
    let bad = "not base64!";
    let decode = base64::engine::general_purpose::STANDARD
        .decode(bad)
        .expect_err("the fixture is not base64");
    for (content, expected) in [
        (
            image(Source::Base64("aGk=".to_owned()), None),
            "RequestError: Media type for tool-result image is required for Vertex AI".to_owned(),
        ),
        (
            image(Source::Base64(bad.to_owned()), Some(ImageMediaType::PNG)),
            format!("RequestError: Invalid base64 tool-result image data: {decode}"),
        ),
        (
            image(
                Source::FileId("files/image-1".to_owned()),
                Some(ImageMediaType::JPEG),
            ),
            "RequestError: Unsupported Vertex AI tool-result image source: files/image-1"
                .to_owned(),
        ),
        (
            image(Source::Unknown, Some(ImageMediaType::WEBP)),
            "RequestError: Unsupported Vertex AI tool-result image source: <unknown>".to_owned(),
        ),
    ] {
        let success = result(AnsweredToolCall::Function, vec![content]);
        let error = encoded(success, ToolResultStyle::Vertex)
            .expect_err("the image is refused before the request is sent");
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn user_media_refuses_unsendable_sources_and_inline_data_without_a_type() {
    let form = "RequestError: Gemini cannot receive this media in its form";
    let untyped = "RequestError: Gemini cannot receive media without its type";
    for (part, expected) in [
        (
            UserContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::PNG), None),
            form,
        ),
        (
            UserContent::Image(Image {
                data: Source::FileId("files/image-1".to_owned()),
                media_type: Some(ImageMediaType::PNG),
                ..Image::default()
            }),
            form,
        ),
        (UserContent::image_base64("aGk=", None, None), untyped),
        (UserContent::audio_base64("aGk=", None), untyped),
    ] {
        let error = user_part(part, None).expect_err("the media is refused");
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn documents_refuse_raw_file_id_and_bodiless_sources() {
    let document = |data: Source| {
        UserContent::Document(Document {
            data: DocumentData::File(data),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        })
    };
    for (part, expected) in [
        (
            document(Source::Raw(vec![1, 2, 3])),
            "RequestError: Raw files not supported, encode as base64 first",
        ),
        (
            document(Source::FileId("files/doc-1".to_owned())),
            "RequestError: Provider file IDs are not supported for Gemini documents",
        ),
        (
            document(Source::Unknown),
            "RequestError: Document has no body",
        ),
    ] {
        let error = user_part(part, None).expect_err("the document is refused");
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn blocked_prompt_ratings_spell_unread_and_missing_fields() {
    // A category or probability that is no enum spelling keeps its JSON
    // text; an absent one says so.
    let error = blocked_prompt_error(&json!({
        "blockReason": "SAFETY",
        "safetyRatings": [
            {"category": 7, "probability": "HIGH"},
            {"probability": 3},
            {"category": "HARM_CATEGORY_HATE_SPEECH"}
        ]
    }))
    .expect("a set block reason is a block");
    let ProviderError::ProviderResponse(response) = &error else {
        panic!("a provider response error, got {error:?}")
    };
    assert_eq!(
        response,
        &ProviderResponseError::without_status(
            "Gemini blocked the prompt: block_reason=SAFETY, safety_ratings=[7=HIGH, \
             <unset>=3, HARM_CATEGORY_HATE_SPEECH=<unset>]"
        )
        .with_code(Some("SAFETY".to_owned()))
        .with_refusal(true)
    );
}

#[test]
fn is_base64_takes_only_padded_standard_base64() {
    for (text, expected) in [
        ("", false),
        ("==", false),
        ("A===", false),
        ("aGk", false),
        ("c2l*", false),
        ("c2ln", true),
        ("c2k=", true),
    ] {
        assert_eq!(is_base64(text), expected, "{text:?}");
    }
}

#[test]
fn a_current_native_item_loses_only_a_signature_that_is_not_base64() {
    for (signature, expected) in [
        ("", json!({"text": "hi"})),
        ("A===", json!({"text": "hi"})),
        ("c2ln", json!({"text": "hi", "thoughtSignature": "c2ln"})),
    ] {
        let block = AssistantContent::text("hi")
            .with_native(json!({"text": "hi", "thoughtSignature": signature}));
        assert_eq!(
            assistant_part(&block, &target(), MODEL).expect("the native item replays"),
            Some(expected),
            "{signature:?}"
        );
    }
    // A current native item that is no object replays verbatim: there is
    // no signature key to inspect.
    let block = AssistantContent::text("hi").with_native(json!("raw native"));
    assert_eq!(
        assistant_part(&block, &target(), MODEL).expect("the native item replays"),
        Some(json!("raw native"))
    );
}

#[test]
fn audio_needs_a_type_and_a_video_url_off_youtube_needs_one_too() {
    use crate::message::{Audio, AudioMediaType, Video, VideoMediaType};
    let audio = |media_type| Audio {
        data: Source::Url("https://example.com/a.mp3".to_owned()),
        media_type,
    };
    assert!(encodes(
        Media::Audio(&audio(Some(AudioMediaType::MP3))),
        false
    ));
    assert!(!encodes(Media::Audio(&audio(None)), false));
    let video = |url: &str, media_type| Video {
        data: Source::Url(url.to_owned()),
        media_type,
        additional_params: None,
    };
    let elsewhere = "https://example.com/v.mp4";
    assert!(encodes(
        Media::Video(&video(elsewhere, Some(VideoMediaType::MP4))),
        false
    ));
    assert!(!encodes(Media::Video(&video(elsewhere, None)), false));
    assert!(encodes(
        Media::Video(&video("https://www.youtube.com/watch?v=x", None)),
        false
    ));
}

#[test]
fn current_turn_keeps_every_signature_when_no_user_content_holds_text_after_the_first() {
    // The second user content holds only an image, so the newest user text
    // is the first content and no turn before it signed anything.
    let original = vec![
        json!({"role": "user", "parts": [{"text": "one"}]}),
        json!({"role": "model", "parts": [{"text": "a", "thoughtSignature": "c2ln"}]}),
        json!({"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "aGk="}}]}),
        json!({"role": "model", "parts": [{"text": "b", "thoughtSignature": "c2ln"}]}),
    ];
    let mut contents = original.clone();
    drop_finished_signatures(&mut contents);
    assert_eq!(contents, original);
}

#[test]
fn issuer_scoping_keeps_an_assistant_message_that_was_empty_to_begin_with() {
    let original = vec![
        Message::user("hi"),
        Message::Assistant(crate::message::AssistantMessage::new(vec![])),
    ];
    let mut history = original.clone();
    scope_reasoning_to_issuer(&mut history, PROVIDER_NAME);
    assert_eq!(history, original);
}

#[test]
fn an_empty_system_message_sets_no_system_instruction() {
    let request = CompletionRequest::from(vec![
        Message::System {
            content: String::new(),
        },
        Message::user("hi"),
    ]);
    let body = request_body(&request, &target(), MODEL, None, |_| {}).expect("the request encodes");
    assert_eq!(
        serde_json::to_value(&body).expect("the body serializes"),
        json!({
            "contents": [{"parts": [{"text": "hi", "thought": false}], "role": "user"}],
            "generationConfig": null, "safetySettings": null,
            "toolConfig": null, "systemInstruction": null
        })
    );
}
