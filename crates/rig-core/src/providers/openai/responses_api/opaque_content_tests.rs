use super::*;
use crate::message::{self, AdditionalParams};
use serde_json::json;

fn raw() -> Value {
    json!({"type":"future_part","payload":{"nested":[1,null,{"x":true}]},"text":"opaque"})
}

fn opaque_text() -> message::AssistantContent {
    message::AssistantContent::Text(text_block(AssistantContent::Unknown(raw())))
}

#[test]
fn opaque_leaf_unions_are_lossless_and_known_tags_are_strict() {
    for value in [raw(), json!({"type":"future_part","payload":null})] {
        assert_eq!(
            serde_json::to_value(
                serde_json::from_value::<AssistantContent>(value.clone()).unwrap()
            )
            .unwrap(),
            value
        );
        assert_eq!(
            serde_json::to_value(
                serde_json::from_value::<ReasoningSummary>(value.clone()).unwrap()
            )
            .unwrap(),
            value
        );
        assert_eq!(
            serde_json::to_value(
                serde_json::from_value::<ReasoningTextContent>(value.clone()).unwrap()
            )
            .unwrap(),
            value
        );
    }
    for value in [
        json!({"type":"output_text","text":42}),
        json!({"type":"refusal","refusal":null}),
    ] {
        assert!(serde_json::from_value::<AssistantContent>(value.clone()).is_err());
        assert!(serde_json::from_value::<AssistantContentType>(value).is_err());
    }
    assert!(serde_json::from_value::<ReasoningSummary>(json!({"type":"summary_text"})).is_err());
    assert!(
        serde_json::from_value::<ReasoningTextContent>(json!({"type":"reasoning_text","text":42}))
            .is_err()
    );
    for content in [
        json!(42),
        json!(null),
        json!([{"type":"reasoning_text","text":42}]),
        json!([true]),
    ] {
        assert!(
            serde_json::from_value::<Output>(
                json!({"type":"reasoning","id":"rs_1","content":content})
            )
            .is_err()
        );
    }
    for content in [json!("text"), json!(["text"])] {
        let item: Output =
            serde_json::from_value(json!({"type":"reasoning","id":"rs_1","content":content}))
                .unwrap();
        assert_eq!(
            serde_json::to_value(item).unwrap()["content"],
            json!([{"type":"reasoning_text","text":"text"}])
        );
    }
}

#[test]
fn opaque_message_parts_replay_in_place_with_and_without_ids() {
    let parts = json!([{"type":"output_text","text":"A"},raw(),{"type":"refusal","refusal":"No"},{"type":"output_text","text":"B"}]);
    let output: Output = serde_json::from_value(json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":parts})).unwrap();
    let content = super::tests::folded_choice(vec![output]);
    assert_eq!(content.len(), 4);
    for id in [Some("msg_1".to_owned()), None] {
        let items: Vec<InputItem> = message::Message::Assistant {
            id,
            content: content.clone(),
        }
        .try_into()
        .unwrap();
        let value = serde_json::to_value(items).unwrap();
        if value[0]["content"].is_array() {
            assert_eq!(value[0]["content"], parts);
        } else {
            // Known ID-less text uses the existing string form; the opaque part alone
            // still receives the array form needed to preserve it.
            assert!(
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["content"] == json!([raw()]))
            );
        }
    }
    let items: Vec<InputItem> = message::Message::Assistant {
        id: None,
        content: vec![opaque_text()],
    }
    .try_into()
    .unwrap();
    assert_eq!(
        serde_json::to_value(items).unwrap()[0]["content"],
        json!([raw()])
    );
}

#[test]
fn opaque_reasoning_replays_both_arrays_and_refuses_a_missing_id() {
    let summary = json!([{"type":"summary_text","text":"S"},raw()]);
    let content = json!([raw(),{"type":"reasoning_text","text":"T"}]);
    let output: Output = serde_json::from_value(json!({"type":"reasoning","id":"rs_1","summary":summary,"content":content,"encrypted_content":"encrypted","signature":"signed"})).unwrap();
    let choice = super::tests::folded_choice(vec![output]);
    let items: Vec<InputItem> = message::Message::Assistant {
        id: None,
        content: choice.clone(),
    }
    .try_into()
    .unwrap();
    let replay = serde_json::to_value(items).unwrap();
    assert_eq!(replay[0]["summary"], summary);
    assert_eq!(replay[0]["content"], content);
    assert_eq!(replay[0]["signature"], "signed");
    assert_eq!(replay[0]["encrypted_content"], "encrypted");
    let mut idless = choice;
    if let message::AssistantContent::Reasoning(reasoning) = &mut idless[0] {
        reasoning.id = None;
    }
    let result: Result<Vec<InputItem>, _> = message::Message::Assistant {
        id: None,
        content: idless,
    }
    .try_into();
    assert!(result.unwrap_err().to_string().contains("opaque"));
}

#[test]
fn opaque_replay_union_does_not_capture_known_tool_or_reasoning_forms() {
    for tagged in [true, false] {
        let mut call = json!({"id":"fc_1","call_id":"call_1","name":"tool","arguments":"{}","status":"completed"});
        let mut reasoning = json!({"id":"rs_1","summary":[]});
        if tagged {
            call["type"] = json!("function_call");
            reasoning["type"] = json!("reasoning");
        }
        assert!(matches!(
            serde_json::from_value::<AssistantContentType>(call).unwrap(),
            AssistantContentType::ToolCall(_)
        ));
        assert!(matches!(
            serde_json::from_value::<AssistantContentType>(reasoning).unwrap(),
            AssistantContentType::Reasoning(_)
        ));
    }
    assert!(
        serde_json::from_value::<AssistantContentType>(json!({"type":"function_call","id":"bad"}))
            .is_err()
    );
}

#[test]
fn opaque_marker_replaces_atomically_only_at_the_owned_top_level() {
    let first = json!({"openai_responses_part":{"kind":"opaque","value":raw()},"other":{"array":[1],"object":{"a":1}}});
    let mut params = AdditionalParams::new(first.as_object().unwrap().clone()).unwrap();
    params.merge(params.clone());
    assert_eq!(
        params.get(OPENAI_RESPONSES_PART_KEY).unwrap()["value"],
        raw()
    );
    assert_eq!(params.get("other").unwrap()["array"], json!([1, 1]));
    let replacement = json!({"type":"next","list":[7]});
    let next = text_block(AssistantContent::Unknown(replacement.clone()))
        .additional_params
        .unwrap();
    params.merge(next);
    assert_eq!(
        params.get(OPENAI_RESPONSES_PART_KEY).unwrap()["value"],
        replacement
    );
    let nested =
        json!({"unrelated":{"openai_responses_part":{"kind":"opaque","value":{"list":[1]}}}});
    let mut params = AdditionalParams::new(nested.as_object().unwrap().clone()).unwrap();
    params.merge(params.clone());
    assert_eq!(
        params.get("unrelated").unwrap()["openai_responses_part"]["value"]["list"],
        json!([1, 1])
    );
}

#[test]
fn opaque_parts_are_refused_by_all_core_non_responses_converters() {
    use crate::providers::{anthropic, cohere, gemini, ollama, openai};
    let parts = [
        opaque_text(),
        message::AssistantContent::Reasoning(message::Reasoning {
            id: None,
            provider: Some("openai".into()),
            content: vec![
                message::ReasoningContent::Summary("known".into()),
                message::ReasoningContent::OpaqueSummary(raw()),
                message::ReasoningContent::OpaqueContent(raw()),
            ],
        }),
    ];
    for part in parts {
        let turn = message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::text("known"), part.clone()],
        };
        for details in [true, false] {
            assert!(
                openai::completion::assistant_content_to_messages(
                    vec![message::AssistantContent::text("known"), part.clone()],
                    details
                )
                .unwrap_err()
                .to_string()
                .contains("opaque")
            );
        }
        let anthropic: Result<anthropic::completion::Message, _> = turn.clone().try_into();
        let cohere: Result<Vec<cohere::completion::Message>, _> = turn.clone().try_into();
        let ollama: Result<Vec<ollama::Message>, _> = turn.try_into();
        assert!(anthropic.unwrap_err().to_string().contains("opaque"));
        assert!(cohere.unwrap_err().to_string().contains("opaque"));
        assert!(ollama.unwrap_err().to_string().contains("opaque"));
        let gemini: Result<gemini::completion::gemini_api_types::Part, _> = part.clone().try_into();
        let interactions: Result<gemini::interactions_api::interactions_api_types::Content, _> =
            part.try_into();
        assert!(gemini.unwrap_err().to_string().contains("opaque"));
        assert!(interactions.unwrap_err().to_string().contains("opaque"));
    }
}

#[test]
fn opaque_reasoning_survives_provenance_filter_and_responses_refuses_foreign_issuers() {
    use crate::{completion::CompletionRequestBuilder, test_utils::MockCompletionModel};
    let mut history = vec![message::Message::Assistant {
        id: None,
        content: vec![
            message::AssistantContent::Reasoning(
                message::Reasoning::new("known").with_provider("foreign"),
            ),
            message::AssistantContent::Reasoning(message::Reasoning {
                id: Some("rs_1".into()),
                provider: Some("foreign".into()),
                content: vec![message::ReasoningContent::OpaqueSummary(raw())],
            }),
        ],
    }];
    message::retain_replayable_reasoning(&mut history, &["openai"]);
    let message::Message::Assistant { content, .. } = &history[0] else {
        panic!("assistant")
    };
    assert_eq!(content.len(), 1);
    let mut request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), "prompt").build();
    request.chat_history = history;
    let wire = crate::providers::openai::OpenAI::new("test").responses("gpt-test");
    assert!(
        wire.responses_request(request, false, None)
            .unwrap_err()
            .to_string()
            .contains("opaque")
    );
}

#[test]
fn opaque_neutral_parts_round_trip_without_becoming_display_text() {
    let reasoning = message::Reasoning {
        id: Some("rs_opaque".into()),
        provider: Some("openai".into()),
        content: vec![
            message::ReasoningContent::Summary("known".into()),
            message::ReasoningContent::OpaqueSummary(raw()),
            message::ReasoningContent::OpaqueContent(raw()),
        ],
    };
    assert_eq!(reasoning.display_text(), "known");
    let serialized = serde_json::to_value(&reasoning).unwrap();
    assert_eq!(
        serialized["content"][1],
        json!({"type":"opaque_summary","content":raw()})
    );
    assert_eq!(
        serialized["content"][2],
        json!({"type":"opaque_content","content":raw()})
    );
    assert_eq!(
        serde_json::from_value::<message::Reasoning>(serialized).unwrap(),
        reasoning
    );
}
