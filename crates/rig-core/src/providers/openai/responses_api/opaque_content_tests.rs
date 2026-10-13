use super::*;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::ProviderError;
use crate::message::{
    self, AssistantContent as CoreContent, AssistantMessage, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart, CarriedText, Message,
};
use crate::wire::{Mode, Operation as _, Wire as _};
use serde_json::json;
use std::error::Error as _;

fn raw() -> Value {
    json!({"type":"future_part","payload":{"nested":[1,null,{"x":true}]},"text":"opaque"})
}

fn opaque_text() -> CoreContent {
    let item = json!({"type":"message","content":[raw()]});
    let mut parts = carried_message_parts(&item, 0, Some("openai".to_owned()))
        .expect("the current private factory retains the exact opaque part");
    assert_eq!(parts.len(), 1);
    CoreContent::CarriedText(parts.remove(0))
}

fn assistant(content: Vec<CoreContent>) -> Message {
    Message::Assistant(AssistantMessage::new(content))
}

fn target() -> wire::Responses {
    crate::providers::openai::OpenAIConfig::new("test").responses("gpt-test")
}

/// The existing default wire reader/fold, with only its required terminal root.
fn decode(output: Vec<Value>) -> Result<CompletionResponse, ProviderError> {
    let body = json!({"id":"resp_1","object":"response","created_at":0,
        "status":"completed","model":"gpt-test","output":output});
    crate::test_utils::decode_reply(
        &target(),
        &CompletionRequest::new("prompt"),
        Mode::Unary,
        [crate::wire::WireFrame::Text(serde_json::to_string(&body)?)],
        Value::Null,
    )
}

fn input_items(content: Vec<CoreContent>) -> Result<Vec<Value>, crate::error::EncodeError> {
    let target = target();
    let mut custom = Custom {
        tools: Default::default(),
        calls: Default::default(),
    };
    let mut system_items = Vec::new();
    super::input(
        &[assistant(content)],
        &target,
        "gpt-test",
        &mut custom,
        false,
        &mut system_items,
        None,
    )
}

/// OC01 / G87: current carried serde and real reader preserve the exact leaves;
/// the retired standalone native serde projections are not restored.
#[test]
fn opaque_leaf_unions_are_lossless_and_known_tags_are_strict() {
    for value in [raw(), json!({"type":"future_part","payload":null})] {
        let message = json!({"type":"message","content":[value.clone()]});
        let text = carried_message_parts(&message, 0, None)
            .expect("unknown message leaf retains its value")
            .remove(0);
        let loaded: CarriedText = serde_json::from_value(
            serde_json::to_value(&text).expect("current carried text serializes"),
        )
        .expect("current carried text loads");
        assert_eq!(loaded, text);
        assert_eq!(loaded.payload.item, value);

        let summary = carried_reasoning(
            &json!({"type":"reasoning","id":"rs_1","summary":[value.clone()]}),
            None,
        )
        .expect("unknown summary leaf retains its value");
        let loaded: CarriedReasoning = serde_json::from_value(
            serde_json::to_value(&summary).expect("current summary carrier serializes"),
        )
        .expect("current summary carrier loads");
        assert_eq!(loaded, summary);
        let [CarriedReasoningPart::OpaqueSummary(payload)] = loaded.content.as_slice() else {
            panic!("the unknown summary stays opaque");
        };
        assert_eq!(payload.item, value);

        let reasoning = carried_reasoning(
            &json!({"type":"reasoning","id":"rs_1","content":[value.clone()]}),
            None,
        )
        .expect("unknown reasoning leaf retains its value");
        let loaded: CarriedReasoning = serde_json::from_value(
            serde_json::to_value(&reasoning).expect("current reasoning carrier serializes"),
        )
        .expect("current reasoning carrier loads");
        assert_eq!(loaded, reasoning);
        let [CarriedReasoningPart::OpaqueContent(payload)] = loaded.content.as_slice() else {
            panic!("the unknown reasoning content stays opaque");
        };
        assert_eq!(payload.item, value);
    }
    for value in [
        json!({"type":"output_text","text":42}),
        json!({"type":"refusal","refusal":null}),
    ] {
        let item = json!({"type":"message","id":"msg_1","role":"assistant",
            "status":"completed","content":[value]});
        assert!(carried_message_parts(&item, 0, None).is_err());
        assert!(decode(vec![item]).is_err());
    }
    assert!(
        decode(vec![json!({"type":"reasoning","id":"rs_1",
        "summary":[{"type":"summary_text"}]})])
        .is_err()
    );
    assert!(
        decode(vec![json!({"type":"reasoning","id":"rs_1",
        "content":[{"type":"reasoning_text","text":42}]})])
        .is_err()
    );
    for content in [
        json!(42),
        json!(null),
        json!([{"type":"reasoning_text","text":42}]),
        json!([true]),
    ] {
        assert!(
            decode(vec![json!({"type":"reasoning","id":"rs_1",
            "content":content})])
            .is_err()
        );
    }
    for content in [json!("text"), json!(["text"])] {
        let response = decode(vec![json!({"type":"reasoning","id":"rs_1",
            "content":content})])
        .expect("both original accepted reasoning forms decode");
        let input = input_items(response.choice).expect("the actual current reasoning replays");
        assert_eq!(
            input[0]["content"],
            json!([{"type":"reasoning_text","text":"text"}])
        );
    }
}

/// OC03: actual received carried reasoning retains the original whole arrays.
#[test]
fn opaque_reasoning_replays_both_arrays_and_refuses_a_missing_id() {
    let summary = json!([{"type":"summary_text","text":"S"},raw()]);
    let content = json!([raw(),{"type":"reasoning_text","text":"T"}]);
    let output = json!({"type":"reasoning","id":"rs_1","summary":summary,
        "content":content,"encrypted_content":"encrypted","signature":"signed"});
    let response = decode(vec![output]).expect("the real reader/fold decodes reasoning");
    let replay =
        input_items(response.choice.clone()).expect("the actual carried reasoning replays");
    assert_eq!(replay[0]["summary"], summary);
    assert_eq!(replay[0]["content"], content);
    assert_eq!(replay[0]["signature"], "signed");
    assert_eq!(replay[0]["encrypted_content"], "encrypted");
    let mut idless = response.choice;
    let [CoreContent::CarriedReasoning(reasoning)] = idless.as_mut_slice() else {
        panic!("one actual carried reasoning item");
    };
    reasoning.id = None;
    assert!(
        input_items(idless)
            .unwrap_err()
            .to_string()
            .contains("opaque")
    );
}

/// OC06: the current actual encoding boundaries keep each original refusal.
#[test]
fn opaque_parts_are_refused_by_all_core_non_responses_converters() {
    use crate::providers::{anthropic, cohere, gemini, ollama, openai};
    let reasoning = carried_reasoning(
        &json!({"type":"reasoning","summary":[{"type":"summary_text","text":"known"},raw()],
            "content":[raw()]}),
        Some("openai".to_owned()),
    )
    .expect("the existing private reader preserves all original reasoning parts");
    let parts = [opaque_text(), CoreContent::CarriedReasoning(reasoning)];
    for part in parts {
        let turn = assistant(vec![CoreContent::text("known"), part.clone()]);
        let request = CompletionRequest::from(vec![turn]).max_tokens(128);
        for details in [true, false] {
            let mut dialect = openai::wire::OPENAI;
            dialect.quirks.reasoning_details = details;
            let wire = openai::OpenAIConfig::with_key(&dialect, "test").chat("gpt-test");
            assert!(
                wire.encode(request.clone(), Mode::Unary)
                    .unwrap_err()
                    .to_string()
                    .contains("opaque")
            );
        }
        let anthropic = anthropic::AnthropicConfig::new("test").completion("test");
        let cohere = cohere::NativeChat::new(cohere::CohereConfig::new("test"), "test");
        let ollama = ollama::Chat::new(ollama::OllamaConfig::new(), "test");
        assert!(
            anthropic
                .encode(request.clone(), Mode::Unary)
                .unwrap_err()
                .to_string()
                .contains("opaque")
        );
        assert!(
            cohere
                .encode(request.clone(), Mode::Unary)
                .unwrap_err()
                .to_string()
                .contains("opaque")
        );
        assert!(
            ollama
                .encode(request, Mode::Unary)
                .unwrap_err()
                .to_string()
                .contains("opaque")
        );
        let part_request = CompletionRequest::from(vec![assistant(vec![part])]);
        let gemini = gemini::GeminiConfig::new("test").completion("test");
        let interactions = gemini::GeminiConfig::new("test").interactions("test");
        assert!(
            gemini
                .encode(part_request.clone(), Mode::Unary)
                .unwrap_err()
                .to_string()
                .contains("opaque")
        );
        assert!(
            interactions
                .encode(part_request, Mode::Unary)
                .unwrap_err()
                .to_string()
                .contains("opaque")
        );
    }
}

/// OC02 / G127: these two labelled containing-item inputs replace only the
/// unavailable independent parent-ID assignment; no input parity is claimed.
#[test]
fn opaque_message_parts_replay_in_place_with_and_without_ids() {
    let parts = json!([{"type":"output_text","text":"A"},raw(),{"type":"refusal","refusal":"No"},{"type":"output_text","text":"B"}]);
    for id in [Some("msg_1".to_owned()), None] {
        let mut output = json!({"type":"message","role":"assistant","status":"completed",
            "phase":"final_answer","content":parts});
        if let Some(id) = &id {
            output["id"] = json!(id);
        }
        let content: Vec<_> = carried_message_parts(&output, 0, Some("openai".to_owned()))
            .expect("the private factory supplies containing-item metadata")
            .into_iter()
            .map(CoreContent::CarriedText)
            .collect();
        assert_eq!(content.len(), 4);
        let value = Value::Array(input_items(content).expect("the actual parts replay"));
        if value[0]["content"].is_array() {
            assert_eq!(value[0]["content"], parts);
        } else {
            assert!(
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["content"] == json!([raw()]))
            );
        }
        assert_eq!(value[0]["status"], "completed");
        assert_eq!(value[0]["phase"], "final_answer");
        if let Some(id) = &id {
            assert_eq!(value[0]["id"], *id);
        } else {
            assert!(value[0].get("id").is_none());
        }
    }
    let items = input_items(vec![opaque_text()]).expect("the idless opaque part replays");
    assert_eq!(items[0]["content"], json!([raw()]));
}

/// OC04 / G128: tagged forms retain their current typed categories;
/// untagged original fixtures are genuine unknown whole items, without inference.
#[test]
fn tagged_items_keep_typed_categories_and_untagged_items_keep_provider_payloads() {
    for tagged in [true, false] {
        let mut call = json!({"id":"fc_1","call_id":"call_1","name":"tool",
            "arguments":"{}","status":"completed"});
        let mut reasoning = json!({"id":"rs_1","summary":[]});
        if tagged {
            call["type"] = json!("function_call");
            reasoning["type"] = json!("reasoning");
        }
        let response = decode(vec![call.clone()]).expect("the original call fixture reads");
        assert_eq!(response.choice.len(), 1);
        if tagged {
            assert!(matches!(&response.choice[0], CoreContent::ToolCall(_)));
        } else {
            let CoreContent::ProviderItem(item) = &response.choice[0] else {
                panic!("the untagged original call is a whole unknown item");
            };
            assert_eq!(item.payload.item, call);
        }
        assert_eq!(response.raw["output"][0], call);
        let response =
            decode(vec![reasoning.clone()]).expect("the original reasoning fixture reads");
        assert_eq!(response.choice.len(), 1);
        if tagged {
            assert!(matches!(
                &response.choice[0],
                CoreContent::CarriedReasoning(_)
            ));
        } else {
            let CoreContent::ProviderItem(item) = &response.choice[0] else {
                panic!("the untagged original reasoning is a whole unknown item");
            };
            assert_eq!(item.payload.item, reasoning);
        }
        assert_eq!(response.raw["output"][0], reasoning);
    }
    assert!(decode(vec![json!({"type":"function_call","id":"bad"})]).is_err());
}

/// OC07 / G130: preparation retains both current carried foreign parts.
/// The removed filter/count-one view is lost; final encoding still refuses opaque data.
#[test]
fn prepared_foreign_reasoning_stays_intact_until_the_typed_responses_refusal() {
    let known = CoreContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: "known".to_owned(),
            signature: None,
        }],
        CarriedPayload::new(Value::Null, Some("foreign".to_owned())),
    ));
    let opaque = CoreContent::CarriedReasoning(
        CarriedReasoning::new(
            vec![CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
                raw(),
                Some("foreign".to_owned()),
            ))],
            CarriedPayload::new(Value::Null, Some("foreign".to_owned())),
        )
        .with_id(Some("rs_1".to_owned())),
    );
    let expected = vec![known, opaque];
    let mut request = CompletionRequest::new("prompt");
    request.chat_history = vec![assistant(expected.clone())];
    let wire = target();
    let prepared = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("actual preparation retains both foreign current carriers");
    let [Message::Assistant(turn)] = prepared.chat_history.as_slice() else {
        panic!("the exact original assistant turn stays");
    };
    assert_eq!(turn.content.len(), 2);
    assert_eq!(turn.content, expected);
    let error = ProviderError::from(
        wire.responses_request(&prepared, Delivery::Http(Mode::Unary))
            .expect_err("actual final request refuses the foreign opaque part"),
    );
    assert!(matches!(&error, ProviderError::Request(_)));
    assert!(error.to_string().contains("opaque"));
    let refusal = error
        .source()
        .expect("typed final refusal")
        .downcast_ref::<message::UnrepresentableOpaqueContent>()
        .expect("the existing nested opaque error");
    assert_eq!(
        refusal.wire,
        "OpenAI Responses with incompatible reasoning provenance"
    );
    assert_eq!(
        refusal.to_string(),
        "OpenAI Responses with incompatible reasoning provenance cannot replay opaque OpenAI Responses content"
    );
}

/// OC08 / G131: the current carried wrapper preserves the original raw values;
/// only the two old bare-content serialized views are lost.
#[test]
fn opaque_neutral_parts_round_trip_without_becoming_display_text() {
    let reasoning = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Summary("known".to_owned()),
            CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
                raw(),
                Some("openai".to_owned()),
            )),
            CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
                raw(),
                Some("openai".to_owned()),
            )),
        ],
        CarriedPayload::new(Value::Null, Some("openai".to_owned())),
    )
    .with_id(Some("rs_opaque".to_owned()));
    assert_eq!(reasoning.display_text(), "known");
    assert_eq!(reasoning.id.as_deref(), Some("rs_opaque"));
    assert_eq!(reasoning.provider(), Some("openai"));
    let serialized = serde_json::to_value(&reasoning).unwrap();
    assert_eq!(
        serialized["content"][1],
        json!({"type":"opaque_summary",
        "content":{"item":raw(),"provider":"openai"}})
    );
    assert_eq!(
        serialized["content"][2],
        json!({"type":"opaque_content",
        "content":{"item":raw(),"provider":"openai"}})
    );
    assert_eq!(serialized["content"][1]["content"]["item"], raw());
    assert_eq!(serialized["content"][2]["content"]["item"], raw());
    assert_eq!(
        serde_json::from_value::<CarriedReasoning>(serialized).unwrap(),
        reasoning
    );
}
