use super::*;
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::message::{Image, ImageDetail, MimeType, ToolFunction, ToolName};
use crate::operation::Completion;
use crate::wire::{Descriptor, Operation};
use serde_json::json;
use std::sync::Mutex;

#[derive(Debug)]
struct DefaultTarget;

impl ReplayTarget for DefaultTarget {
    fn api(&self) -> Api {
        Api::from_static("image-preflight.test")
    }
    fn provider(&self) -> &str {
        "image-preflight"
    }
    fn model(&self) -> &str {
        "model"
    }
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
    fn declares_tools(&self, _request: &CompletionRequest) -> bool {
        true
    }
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }
}

// The shared-hook fixture pins the protected image conversion's format/source
// refusals. The private Bedrock encoder has separate destination tests.
#[derive(Debug)]
struct OriginalImages {
    seen: Mutex<Vec<Image>>,
    accepts: Accepts,
}

impl OriginalImages {
    fn new(accepts: Accepts) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            accepts,
        }
    }
}

impl ReplayTarget for OriginalImages {
    fn api(&self) -> Api {
        DefaultTarget.api()
    }
    fn provider(&self) -> &str {
        DefaultTarget.provider()
    }
    fn model(&self) -> &str {
        DefaultTarget.model()
    }
    fn accepts(&self, _model: &str) -> Accepts {
        self.accepts
    }
    fn declares_tools(&self, request: &CompletionRequest) -> bool {
        DefaultTarget.declares_tools(request)
    }
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        DefaultTarget.map_options(request, fields)
    }
    fn encodes(&self, _model: &str, _media: Media<'_>) -> bool {
        panic!("image refusal must precede media adaptation");
    }
    fn check_image(&self, image: &Image) -> Result<(), ProviderError> {
        self.seen
            .lock()
            .expect("image observations")
            .push(image.clone());
        if let Some(format) = &image.media_type {
            match format {
                ImageMediaType::JPEG
                | ImageMediaType::PNG
                | ImageMediaType::GIF
                | ImageMediaType::WEBP => {}
                format => {
                    return Err(ProviderError::Provider(format!(
                        "Unsupported format {}",
                        format.to_mime_type()
                    )));
                }
            }
        }
        if !matches!(&image.data, DocumentSourceKind::Base64(_)) {
            return Err(ProviderError::request(
                "Only base64 encoded strings are allowed for image input on AWS Bedrock",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RecordImages(Mutex<Vec<Image>>);

impl ReplayTarget for RecordImages {
    fn api(&self) -> Api {
        DefaultTarget.api()
    }
    fn provider(&self) -> &str {
        DefaultTarget.provider()
    }
    fn model(&self) -> &str {
        DefaultTarget.model()
    }
    fn accepts(&self, model: &str) -> Accepts {
        DefaultTarget.accepts(model)
    }
    fn declares_tools(&self, request: &CompletionRequest) -> bool {
        DefaultTarget.declares_tools(request)
    }
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        DefaultTarget.map_options(request, fields)
    }
    fn check_image(&self, image: &Image) -> Result<(), ProviderError> {
        self.0
            .lock()
            .expect("image observations")
            .push(image.clone());
        Ok(())
    }
}

fn image(data: DocumentSourceKind, media_type: Option<ImageMediaType>) -> Image {
    Image {
        data,
        media_type,
        detail: Some(ImageDetail::High),
        native: None,
    }
}

fn png(data: DocumentSourceKind) -> Image {
    image(data, Some(ImageMediaType::PNG))
}

fn call() -> ToolCall {
    ToolCall::from_wire(
        "call_image",
        ToolFunction::new(ToolName::new("lookup").expect("tool name"), json!({})),
    )
}

fn history(image: Image, place: Place) -> Vec<Message> {
    match place {
        Place::User => vec![Message::User {
            content: vec![UserContent::Image(image)],
        }],
        Place::ToolResult => {
            let call = call();
            let result = call.result(vec![ToolResultContent::Image(image)]);
            vec![
                Message::user("lookup"),
                Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                    call,
                )])),
                Message::User {
                    content: vec![UserContent::ToolResult(result)],
                },
            ]
        }
        Place::Assistant => panic!("fixture covers user and tool-result images"),
    }
}

fn prepare(
    history: Vec<Message>,
    target: &dyn ReplayTarget,
) -> Result<CompletionRequest, ProviderError> {
    Completion::prepare(
        CompletionRequest {
            chat_history: history,
            ..CompletionRequest::new("unused")
        },
        &Descriptor::new("image-preflight").replay(target),
    )
}

#[test]
fn unsupported_declared_format_keeps_provider_error_before_source_conversion() {
    for place in [Place::User, Place::ToolResult] {
        for source in [
            DocumentSourceKind::Base64("aGk=".into()),
            DocumentSourceKind::Raw(b"hi".to_vec()),
        ] {
            let original = image(source, Some(ImageMediaType::HEIC));
            let target = OriginalImages::new(Accepts::ALL);
            let error = prepare(history(original.clone(), place), &target)
                .expect_err("original declared format is refused");
            assert!(matches!(
                &error,
                ProviderError::Provider(message) if message == "Unsupported format image/heic"
            ));
            assert_eq!(
                error.to_string(),
                "ProviderError: Unsupported format image/heic"
            );
            assert_eq!(*target.seen.lock().expect("observed original"), [original]);
        }
    }
}

#[test]
fn raw_user_and_tool_result_images_refuse_before_becoming_base64() {
    for place in [Place::User, Place::ToolResult] {
        let original = png(DocumentSourceKind::Raw(vec![0x89, b'P', b'N', b'G']));
        let target = OriginalImages::new(Accepts::ALL);
        let error = prepare(history(original.clone(), place), &target)
            .expect_err("original raw source is refused");
        assert!(matches!(&error, ProviderError::Request(_)));
        assert_eq!(
            error.to_string(),
            "RequestError: Only base64 encoded strings are allowed for image input on AWS Bedrock"
        );
        assert_eq!(*target.seen.lock().expect("observed original"), [original]);
    }
}

#[test]
fn user_and_tool_result_urls_are_checked_without_s3_widening() {
    for place in [Place::User, Place::ToolResult] {
        for url in [
            "https://example.test/image.png?exact=1",
            "s3://bucket/exact/image.png",
        ] {
            let original = png(DocumentSourceKind::Url(url.into()));
            let target = OriginalImages::new(Accepts::ALL);
            let error = prepare(history(original.clone(), place), &target)
                .expect_err("original URL source is refused");
            assert!(matches!(&error, ProviderError::Request(_)));
            assert_eq!(
                error.to_string(),
                "RequestError: Only base64 encoded strings are allowed for image input on AWS Bedrock"
            );
            assert_eq!(*target.seen.lock().expect("observed original"), [original]);
        }
    }
}

#[test]
fn absent_format_stays_absent_during_original_raw_preflight() {
    for place in [Place::User, Place::ToolResult] {
        let original = image(DocumentSourceKind::Raw(vec![0x89, b'P', b'N', b'G']), None);
        let target = OriginalImages::new(Accepts::ALL);
        let error = prepare(history(original.clone(), place), &target)
            .expect_err("raw source is checked before type sniffing");
        assert!(matches!(&error, ProviderError::Request(_)));
        assert_eq!(
            error.to_string(),
            "RequestError: Only base64 encoded strings are allowed for image input on AWS Bedrock"
        );
        assert_eq!(*target.seen.lock().expect("observed original"), [original]);
    }
}

#[test]
fn original_images_are_visited_in_content_order_and_assistant_images_are_excluded() {
    let first = image(DocumentSourceKind::Raw(vec![0x89, b'P', b'N', b'G']), None);
    let second = png(DocumentSourceKind::Base64("aGk=".into()));
    let third = png(DocumentSourceKind::Url("https://example.test/third".into()));
    let fourth = png(DocumentSourceKind::Raw(b"fourth".to_vec()));
    let fifth = png(DocumentSourceKind::Base64("ZmlmdGg=".into()));
    let assistant = image(
        DocumentSourceKind::Url("https://example.test/assistant".into()),
        Some(ImageMediaType::HEIC),
    );
    let call = call();
    let result = call.result(vec![
        ToolResultContent::text("before"),
        ToolResultContent::Image(second.clone()),
        ToolResultContent::Json {
            value: json!({"between": true}),
        },
        ToolResultContent::Image(third.clone()),
    ]);
    let history = vec![
        Message::system("instructions"),
        Message::user("lookup"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::Image(assistant),
            AssistantContent::ToolCall(call),
        ])),
        Message::User {
            content: vec![
                UserContent::Image(first.clone()),
                UserContent::ToolResult(result),
                UserContent::text("between"),
                UserContent::Image(fourth.clone()),
            ],
        },
        Message::assistant("continue"),
        Message::User {
            content: vec![UserContent::Image(fifth.clone())],
        },
    ];
    let target = RecordImages::default();
    prepare(history, &target).expect("recording hook leaves adaptation unchanged");
    assert_eq!(
        *target.0.lock().expect("ordered observations"),
        [first, second, third, fourth, fifth]
    );
}

#[test]
fn first_original_image_refusal_stops_before_later_images_or_omission() {
    let accepted = png(DocumentSourceKind::Base64("aGk=".into()));
    let refused = png(DocumentSourceKind::Raw(b"raw".to_vec()));
    let later = image(
        DocumentSourceKind::Base64("aGk=".into()),
        Some(ImageMediaType::HEIC),
    );
    let target = OriginalImages::new(Accepts::TEXT);
    let error = prepare(
        vec![Message::User {
            content: vec![
                UserContent::Image(accepted.clone()),
                UserContent::Image(refused.clone()),
                UserContent::Image(later),
            ],
        }],
        &target,
    )
    .expect_err("preflight refuses before a text-only model omits images");
    assert!(matches!(&error, ProviderError::Request(_)));
    assert_eq!(
        error.to_string(),
        "RequestError: Only base64 encoded strings are allowed for image input on AWS Bedrock"
    );
    assert_eq!(*target.seen.lock().expect("first refusal"), [accepted, refused]);
}

#[test]
fn default_image_preflight_keeps_existing_raw_normalization_and_tool_pairing() {
    for place in [Place::User, Place::ToolResult] {
        let original = image(DocumentSourceKind::Raw(vec![0x89, b'P', b'N', b'G']), None);
        DefaultTarget
            .check_image(&original)
            .expect("default hook accepts the original");
        let prepared = prepare(history(original.clone(), place), &DefaultTarget)
            .expect("default target still accepts raw images");
        let expected = image(
            DocumentSourceKind::Base64("iVBORw==".into()),
            Some(ImageMediaType::PNG),
        );
        assert_eq!(prepared.chat_history, history(expected, place));
        assert_eq!(original.media_type, None);
        assert!(matches!(original.data, DocumentSourceKind::Raw(_)));
    }
}

#[derive(Debug)]
struct Routed<'a>(&'a OriginalImages);

impl ReplayTarget for Routed<'_> {
    fn api(&self) -> Api {
        DefaultTarget.api()
    }
    fn provider(&self) -> &str {
        DefaultTarget.provider()
    }
    fn model(&self) -> &str {
        DefaultTarget.model()
    }
    fn accepts(&self, model: &str) -> Accepts {
        DefaultTarget.accepts(model)
    }
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        DefaultTarget.map_options(request, fields)
    }
    fn route(&self, _request: &CompletionRequest) -> Option<&dyn ReplayTarget> {
        Some(self.0)
    }
}

#[test]
fn prepare_checks_the_routed_destinations_original_image_contract() {
    let original = png(DocumentSourceKind::Raw(b"raw".to_vec()));
    let target = OriginalImages::new(Accepts::ALL);
    let error = prepare(history(original.clone(), Place::User), &Routed(&target))
        .expect_err("routed preflight cannot fall back to the outer default");
    assert!(matches!(&error, ProviderError::Request(_)));
    assert_eq!(
        error.to_string(),
        "RequestError: Only base64 encoded strings are allowed for image input on AWS Bedrock"
    );
    assert_eq!(*target.seen.lock().expect("routed original"), [original]);
}
