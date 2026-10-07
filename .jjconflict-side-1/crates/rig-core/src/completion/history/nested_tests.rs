use super::*;
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::message::{
    CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart, CarriedText,
    CarriedTextContent, Document, StopReason,
};
use crate::operation::Completion;
use crate::wire::{Descriptor, Operation};
use serde_json::json;
use std::error::Error as _;

#[derive(Debug)]
struct Target {
    later_system: LaterSystem,
    panic_on_media: bool,
}

impl ReplayTarget for Target {
    fn api(&self) -> Api {
        Api::from_static("nested.test")
    }
    fn provider(&self) -> &str {
        "destination"
    }
    fn model(&self) -> &str {
        "model"
    }
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }
    fn starts_with_user(&self) -> bool {
        true
    }
    fn later_system(&self, _model: &str) -> LaterSystem {
        self.later_system
    }
    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        assert!(
            !self.panic_on_media,
            "adaptation ran before nested preflight"
        );
        !matches!(media, Media::Document(_))
    }
}

const TARGET: Target = Target {
    later_system: LaterSystem::InPlace,
    panic_on_media: false,
};

fn parts(issuer: Option<String>) -> Vec<AssistantContent> {
    let mut text = CarriedText::new(
        CarriedTextContent::Text(" \tknown text ".to_owned()),
        CarriedPayload::new(
            json!({"type":"output_text", "text":" \tknown text ", "future":[null, true]}),
            issuer.clone(),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 8,
            part_index: 0,
        }),
    );
    text.item_id = Some("msg_item".to_owned());
    text.phase = Some(String::new());
    let mut opaque = CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(
            json!({"type":"future_part", "data":{"exact":" \n "}}),
            issuer.clone(),
        )
        .with_position(CarriedPosition::MessageContent {
            item_index: 8,
            part_index: 1,
        }),
    );
    opaque.item_id = text.item_id.clone();
    opaque.phase = text.phase.clone();
    let reasoning = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Text {
                text: "think".to_owned(),
                signature: Some(" signature ".to_owned()),
            },
            CarriedReasoningPart::OpaqueSummary(
                CarriedPayload::new(json!({"type":"future_summary", "raw":[1, null]}), None)
                    .with_position(CarriedPosition::ReasoningSummary { part_index: 3 }),
            ),
            CarriedReasoningPart::OpaqueContent(
                CarriedPayload::new(
                    json!({"type":"future_content", "raw":{"x":true}}),
                    Some("foreign/part".to_owned()),
                )
                .with_position(CarriedPosition::ReasoningContent { part_index: 7 }),
            ),
            CarriedReasoningPart::Encrypted("not visible".to_owned()),
            CarriedReasoningPart::Summary("summary".to_owned()),
        ],
        CarriedPayload::new(
            json!({"type":"reasoning", "id":"rs_item", "future":true}),
            issuer,
        ),
    )
    .with_id(Some("rs_item".to_owned()));
    vec![
        AssistantContent::CarriedText(text),
        AssistantContent::CarriedText(opaque),
        AssistantContent::CarriedReasoning(reasoning),
    ]
}

fn turn(content: Vec<AssistantContent>, origin: Option<Origin>, stop: StopReason) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        origin,
        stop: Some(stop),
    })
}

fn request(history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        chat_history: history,
        ..CompletionRequest::new("unused")
    }
}

fn html() -> Document {
    Document {
        data: DocumentData::Text("<p>exact HTML</p>".to_owned()),
        media_type: Some(DocumentMediaType::HTML),
        additional_params: Some(json!({"title":"original", "future":[null]})),
    }
}

#[test]
fn nested_carried_adaptation_preserves_exact_data_origin_and_canonical_identity() {
    for issuer in [
        None,
        Some("destination".to_owned()),
        Some("foreign/issuer".to_owned()),
    ] {
        for origin in [
            None,
            Some(Origin::new("nested.test", "destination", "model")),
            Some(Origin::new("other.api", "turn-provider", "other-model")),
        ] {
            let content = parts(issuer.clone());
            for block in &content {
                assert!(block.is_carried());
                assert_eq!(block.canonical(), *block);
                assert_eq!(
                    block.clone().with_native(json!({"not":"carried data"})),
                    *block
                );
                assert!(block.native_item().is_none());
                assert!(block.stale_item().is_none());
            }
            let history = vec![
                Message::user("before"),
                turn(content, origin, StopReason::Stop),
                Message::user("after"),
            ];
            assert_eq!(adapt(&history, &TARGET), history);
        }
    }
}

#[test]
fn failed_turns_keep_nested_variants_and_drop_only_upstream_failed_content() {
    for stop in [
        StopReason::Error("failure".into()),
        StopReason::Aborted("cancelled".into()),
    ] {
        let carried = parts(Some("foreign/issuer".to_owned()));
        let origin = Some(Origin::new("source.api", "turn-provider", "source-model"));
        let mut content = vec![
            AssistantContent::text("failed text").with_native(json!({"type":"message"})),
            AssistantContent::reasoning("failed reasoning"),
        ];
        content.extend(carried.clone());
        let history = vec![
            Message::user("before"),
            turn(content, origin.clone(), stop.clone()),
            Message::user("after"),
        ];
        assert_eq!(
            adapt(&history, &TARGET),
            vec![
                Message::user("before"),
                turn(carried, origin, stop),
                Message::user("after"),
            ]
        );
    }
}

#[test]
fn user_first_nested_preflight_refuses_original_parts_before_media_adaptation() {
    let content = parts(None);
    for block in [&content[0], &content[1], &content[2]] {
        let history = vec![
            turn(
                vec![block.clone()],
                Some(Origin::new("source.api", "source", "source-model")),
                StopReason::Stop,
            ),
            Message::User {
                content: vec![UserContent::Document(html())],
            },
        ];
        let adapted = adapt(&history, &TARGET);
        assert_eq!(
            adapted[0], history[0],
            "user-first shaping preserves the leading carried turn"
        );
        assert_eq!(adapted[1], Message::user("<p>exact HTML</p>"));
        let refusal = TARGET
            .check_nested_content(&history)
            .expect_err("default target cannot replay this part");
        assert_eq!(refusal, UnrepresentableOpaqueContent::new("destination"));
        let before_adapt = Target {
            panic_on_media: true,
            ..TARGET
        };
        let error = Completion::prepare(
            request(history),
            &Descriptor::new("destination").replay(&before_adapt),
        )
        .expect_err("preflight precedes the media fallback");
        assert_eq!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<UnrepresentableOpaqueContent>()),
            Some(&refusal)
        );
    }
    let known = AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::Summary("known summary".to_owned())],
        CarriedPayload::new(
            json!({"type":"reasoning", "future":true}),
            Some("foreign/issuer".to_owned()),
        ),
    ));
    let history = vec![
        turn(vec![known], None, StopReason::Stop),
        Message::user("after"),
    ];
    TARGET
        .check_nested_content(&history)
        .expect("known reasoning has no opaque parts");
    let prepared = Completion::prepare(
        request(history.clone()),
        &Descriptor::new("destination").replay(&TARGET),
    )
    .expect("default target accepts known carried reasoning");
    assert_eq!(prepared.chat_history, history);
}

#[test]
fn system_in_place_keeps_positions_and_leading_keeps_joined_behavior() {
    let document = Document {
        data: DocumentData::Text("document text".to_owned()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    };
    let history = vec![
        Message::system(" first block "),
        Message::User {
            content: vec![UserContent::Document(document)],
        },
        Message::user("question"),
        Message::system(" middle block "),
        Message::assistant("answer"),
        Message::user("next"),
    ];
    let rest = vec![
        Message::User {
            content: vec![
                UserContent::text("document text"),
                UserContent::text("question"),
            ],
        },
        Message::assistant("answer"),
        Message::user("next"),
    ];
    for (policy, expected) in [
        (
            LaterSystem::InPlace,
            vec![
                Message::system(" first block "),
                rest[0].clone(),
                Message::system(" middle block "),
                rest[1].clone(),
                rest[2].clone(),
            ],
        ),
        (
            LaterSystem::Leading,
            std::iter::once(Message::system(" first block \n\n middle block "))
                .chain(rest)
                .collect(),
        ),
    ] {
        let target = Target {
            later_system: policy,
            ..TARGET
        };
        assert_eq!(adapt(&history, &target), expected);
        assert_eq!(
            Completion::prepare(
                request(history.clone()),
                &Descriptor::new("destination").replay(&target)
            )
            .expect("system shaping prepares without a provider call")
            .chat_history,
            expected
        );
    }
}

#[derive(Debug)]
struct RefuseHtml;

impl ReplayTarget for RefuseHtml {
    fn api(&self) -> Api {
        TARGET.api()
    }
    fn provider(&self) -> &str {
        TARGET.provider()
    }
    fn model(&self) -> &str {
        TARGET.model()
    }
    fn accepts(&self, model: &str) -> Accepts {
        TARGET.accepts(model)
    }
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        TARGET.map_options(request, fields)
    }
    fn encodes(&self, _model: &str, _media: Media<'_>) -> bool {
        panic!("declared document refusal must precede adaptation");
    }
    fn check_document(&self, document: &Document) -> Result<(), EncodeError> {
        assert_eq!(
            document,
            &html(),
            "preflight sees the original declared media and all metadata"
        );
        Err(EncodeError::request("declared HTML cannot be sent"))
    }
}

#[test]
fn declared_document_refusal_precedes_text_fallback_and_default_preflight_is_unchanged() {
    let history = vec![Message::User {
        content: vec![UserContent::Document(html())],
    }];
    let original = request(history.clone());
    TARGET
        .check_documents(&original)
        .expect("default document policy accepts the original");
    assert!(RefuseHtml.check_documents(&original).is_err());
    let error = Completion::prepare(
        original,
        &Descriptor::new("destination").replay(&RefuseHtml),
    )
    .expect_err("HTML is refused before its own text can be substituted");
    assert!(matches!(error, ProviderError::Request(_)));
    assert_eq!(
        error.to_string(),
        "RequestError: declared HTML cannot be sent"
    );
    let prepared = Completion::prepare(
        request(history.clone()),
        &Descriptor::new("destination").replay(&TARGET),
    )
    .expect("default document preflight still permits text fallback");
    assert_eq!(
        prepared.chat_history,
        vec![Message::user("<p>exact HTML</p>")]
    );
    assert_eq!(
        history,
        vec![Message::User {
            content: vec![UserContent::Document(html())]
        }]
    );
}
