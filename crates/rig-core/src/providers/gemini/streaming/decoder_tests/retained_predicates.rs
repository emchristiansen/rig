use super::*;

#[test]
fn signed_inline_media_marker_survives_both_modes_and_encodes_empty_text_without_signature() {
    use crate::message::{AssistantContent, CarriedTextContent, Message};
    use crate::wire::{Operation, Wire};
    for mode in [Mode::Unary, Mode::Streaming] {
        let part = json!({"inlineData":{"data":"iVBORw0KGgoAAAANSUhEUg==","mimeType":"image/png"},
            "thoughtSignature":"signed-image"});
        let reply = json!({"candidates":[{"content":{"parts":[part.clone()],"role":"model"},"finishReason":"STOP"}]});
        let response = crate::test_utils::decode_reply(
            &wire(),
            &CompletionRequest::new("restate"),
            mode,
            [WireFrame::Text(reply.to_string())],
            Value::Null,
        )
        .expect("protected empty raw marker is delivered");
        let [AssistantContent::CarriedText(marker)] = response.choice.as_slice() else {
            panic!("one empty media marker");
        };
        assert!(
            matches!(&marker.content,CarriedTextContent::InspectionOnlyMedia(text) if text.is_empty())
        );
        assert_eq!(marker.payload.item, part);
        assert_eq!(
            marker
                .payload
                .item
                .pointer("/inlineData/mimeType")
                .and_then(Value::as_str),
            Some("image/png")
        );
        assert_eq!(
            marker
                .payload
                .item
                .pointer("/inlineData/data")
                .and_then(Value::as_str),
            Some("iVBORw0KGgoAAAANSUhEUg==")
        );
        assert_eq!(marker.payload.item["thoughtSignature"], "signed-image");
        assert_eq!(response.text(), "");
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        let target = wire();
        let history = vec![
            Message::user("draw"),
            Message::Assistant(response.continued(response.choice.clone())),
            Message::user("again"),
        ];
        let request = Completion::prepare(CompletionRequest::from(history), &target.describe())
            .expect("empty known projection prepares");
        let encoded = target
            .encode(request, Mode::Unary)
            .expect("marker projects at actual encoder");
        let body = crate::test_utils::json_body(&encoded.request);
        let model = &body["contents"][1]["parts"];
        assert_eq!(model.as_array().expect("one marker").len(), 1);
        assert_eq!(model[0]["text"], "");
        assert!(model.to_string().find("inlineData").is_none());
        assert!(model.to_string().find("signed-image").is_none());
        assert!(model.to_string().find("thoughtSignature").is_none());
    }
}
