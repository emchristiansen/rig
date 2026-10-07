use super::*;
use crate::completion::CompletionRequest;
use crate::message::{CarriedPayload, CarriedPosition, CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature, StopReason};
use crate::operation::Completion;
use crate::wire::{Descriptor, Operation};
use serde_json::json;
use std::error::Error as _;

#[derive(Debug)]
struct ForeignText;

impl ReplayTarget for ForeignText {
    fn api(&self) -> Api { Api::from_static("foreign.text") }
    fn provider(&self) -> &str { "foreign" }
    fn model(&self) -> &str { "changed-model" }
    fn accepts(&self, _model: &str) -> Accepts { Accepts::ALL }
    fn starts_with_user(&self) -> bool { true }
    fn map_options(&self, _request: &CompletionRequest, fields: crate::completion::options::OptionFields<'_>) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }
    fn encodes(&self, _model: &str, _media: Media<'_>) -> bool {
        panic!("carried preflight must precede media adaptation");
    }
}

fn known_parts(issuer: Option<String>) -> Vec<AssistantContent> {
    vec![
        AssistantContent::CarriedText(CarriedText::new(
            CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(" edited text ", GeminiThoughtSignature::new(" original sig "))),
            CarriedPayload::new(json!({"text":"original","thoughtSignature":" original sig ","future":[null,true]}), issuer.clone())
                .with_position(CarriedPosition::MessageContent { item_index: 2, part_index: 0 }),
        )),
        AssistantContent::CarriedText(CarriedText::new(
            CarriedTextContent::InspectionOnlyMedia(String::new()),
            CarriedPayload::new(json!({"inlineData":{"mimeType":"image/png","data":"raw"},"future":{"a":[null,7]}}), issuer)
                .with_position(CarriedPosition::MessageContent { item_index: 2, part_index: 1 }),
        )),
    ]
}

fn turn(content: Vec<AssistantContent>, stop: StopReason) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        origin: Some(Origin::new("gemini.api", "gemini", "original-model")),
        stop: Some(stop),
    })
}

#[test]
fn foreign_known_text_preflight_and_real_prepare_preserve_original_carried_evidence() {
    for issuer in [None, Some("gemini".to_owned()), Some("another/issuer".to_owned())] {
        let parts = known_parts(issuer);
        let history = vec![turn(parts.clone(), StopReason::Stop), Message::user("next")];
        ForeignText.check_nested_content(&history).expect("known text has no opaque replay requirement");
        assert_eq!(adapt(&history, &ForeignText), history);
        let prepared = Completion::prepare(
            CompletionRequest { chat_history: history.clone(), ..CompletionRequest::new("unused") },
            &Descriptor::new("foreign").replay(&ForeignText),
        ).expect("known-text preflight permits foreign preparation");
        assert_eq!(prepared.chat_history, history);
        let Message::Assistant(adapted) = &prepared.chat_history[0] else { panic!("leading carried turn retained") };
        assert_eq!(adapted.content, parts);
        for block in &adapted.content {
            let AssistantContent::CarriedText(part) = block else { panic!("carried text") };
            assert_eq!(part.known_text_projection(), Some(part.text()));
            assert_eq!(part.replay_item(), Ok(crate::message::CarriedTextReplay::KnownText(part.text())));
            assert!(block.native_item().is_none());
        }
    }
}

#[test]
fn failed_turns_retain_signed_answers_and_raw_inspection_with_origin_and_positions() {
    for stop in [StopReason::Error("failure".into()), StopReason::Aborted("cancelled".into())] {
        let parts = known_parts(Some("gemini".to_owned()));
        let mut content = vec![AssistantContent::text("failed ordinary text")];
        content.extend(parts.clone());
        let history = vec![Message::user("before"), turn(content, stop.clone()), Message::user("after")];
        let original = history.clone();
        let expected = vec![Message::user("before"), turn(parts, stop), Message::user("after")];
        ForeignText.check_nested_content(&history).expect("known categories accepted before failure shaping");
        assert_eq!(adapt(&history, &ForeignText), expected);
        assert_eq!(history, original);
    }
}

#[test]
fn existing_responses_part_and_whole_item_refusals_still_precede_adaptation() {
    for content in [CarriedTextContent::Text("known".to_owned()), CarriedTextContent::Refusal("refusal".to_owned()), CarriedTextContent::Opaque(String::new())] {
        let part = CarriedText::new(content, CarriedPayload::new(json!({"type":"future"}), None));
        let history = vec![turn(vec![AssistantContent::CarriedText(part)], StopReason::Stop), Message::User {
            content: vec![UserContent::Image(crate::message::Image { data: DocumentSourceKind::Url("https://example.invalid/image".to_owned()), media_type: Some(ImageMediaType::PNG), detail: None, native: None })],
        }];
        let refusal = ForeignText.check_nested_content(&history).expect_err("Responses part needs destination policy");
        assert_eq!(refusal, UnrepresentableOpaqueContent::new("foreign"));
        let error = Completion::prepare(
            CompletionRequest { chat_history: history, ..CompletionRequest::new("unused") },
            &Descriptor::new("foreign").replay(&ForeignText),
        ).expect_err("refusal occurs before media adaptation");
        assert_eq!(error.source().and_then(|source| source.downcast_ref::<UnrepresentableOpaqueContent>()), Some(&refusal));
    }
    let item = ProviderItem::new(json!({"type":"future","raw":[null,3]}), "gemini");
    let history = vec![turn(vec![AssistantContent::ProviderItem(item.clone())], StopReason::Stop), Message::user("next")];
    assert_eq!(ForeignText.check_provider_items(&history), Err(UnreplayableProviderItem::new("foreign.text", &item)));
}
