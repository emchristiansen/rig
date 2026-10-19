use super::*;
use crate::proto;
use message::{AssistantContent, AssistantMessage, CarriedReasoningPart, Message};
use rig_core::completion::{CompletionResponse, FinishReason};
use serde_json::json;

fn thought_fixture() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![
                    proto::Part {
                        data: Some(proto::part::Data::Text(
                            "Let me work through this...".to_string(),
                        )),
                        thought: true,
                        ..Default::default()
                    },
                    proto::Part {
                        data: Some(proto::part::Data::Text("The answer is 42.".to_string())),
                        thought: false,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn signed_answer_fixture() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![
                    proto::Part {
                        data: Some(proto::part::Data::Text("the chain".to_string())),
                        thought: true,
                        ..Default::default()
                    },
                    proto::Part {
                        data: Some(proto::part::Data::Text("answer".to_string())),
                        thought: false,
                        thought_signature: b"sig-bytes".to_vec(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn missing_id_fixture() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: (0..3)
                    .map(|i| proto::Part {
                        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                            id: if i == 1 {
                                "tool-0".into()
                            } else {
                                String::new()
                            },
                            name: "same".into(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn raw_fixture() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("hello".to_string())),
                    ..Default::default()
                }],
                role: "model".to_string(),
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            index: Some(0),
            finish_message: Some("done".to_string()),
            ..Default::default()
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 10,
            candidates_token_count: 20,
            total_token_count: 30,
            cached_content_token_count: 4,
            ..Default::default()
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-1".to_string(),
        prompt_feedback: None,
    }
}

async fn partial_without_terminal(response: proto::GenerateContentResponse) -> CompletionResponse {
    assert!(
        matches!(
            super::tests::complete(response.clone()),
            Err(ProviderError::Truncated)
        ),
        "the original fixture is also truncated on the actual unary path"
    );
    let mut stream = super::tests::stream_from_events(vec![Ok(response)]);
    let mut truncated = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(_) => {}
            Err(ProviderError::Truncated) => truncated = true,
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
    let partial = stream.partial();
    assert!(truncated, "the original fixture has no genuine terminal");
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
    partial
}

#[tokio::test]
async fn text_response_skips_thought_parts() {
    let partial = partial_without_terminal(thought_fixture()).await;
    assert_eq!(
        partial.text(),
        "The answer is 42.",
        "reasoning must not be reported as the response text"
    );
}

#[test]
fn completed_text_response_skips_thought_parts() {
    let mut response = thought_fixture();
    response.candidates[0].finish_reason = proto::candidate::FinishReason::Stop as i32;
    let response = super::tests::complete(response).expect("the explicit terminal completes");
    assert_eq!(response.text(), "The answer is 42.");
}

fn assert_signed_answer_replay(normalized: &CompletionResponse) {
    assert_eq!(normalized.choice.len(), 2, "{:?}", normalized.choice);
    assert!(
        matches!(normalized.choice.first(), Some(AssistantContent::CarriedReasoning(reasoning))
        if reasoning.content == vec![CarriedReasoningPart::Text { text:"the chain".to_owned(), signature:None }]),
        "unsigned reasoning: {:?}",
        normalized.choice
    );
    let signature = match normalized.choice.get(1) {
        Some(AssistantContent::CarriedText(text)) => {
            text.signed_answer().map(|answer| answer.signature.as_str())
        }
        _ => None,
    }
    .expect("the answer text carries its signature");
    let request = CompletionRequest::from(vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(normalized.choice.clone())),
        Message::user("again"),
    ]);
    let req = GenerateContent::new("gemini-3-flash-preview")
        .encode(request, Mode::Unary)
        .expect("request build");
    let answer = req
        .contents
        .get(1)
        .expect("assistant turn")
        .parts
        .iter()
        .find(|part| matches!(&part.data, Some(proto::part::Data::Text(text)) if text == "answer"))
        .expect("the answer part");
    assert_eq!(answer.thought_signature, b"sig-bytes".to_vec());
    assert!(!signature.is_empty());
}

#[tokio::test]
async fn a_signature_on_answer_text_stays_on_that_text() {
    let partial = partial_without_terminal(signed_answer_fixture()).await;
    assert_signed_answer_replay(&partial);
}

#[test]
fn completed_answer_text_keeps_and_replays_its_signature() {
    let mut response = signed_answer_fixture();
    response.candidates[0].finish_reason = proto::candidate::FinishReason::Stop as i32;
    let normalized = super::tests::complete(response).expect("the explicit terminal completes");
    assert_signed_answer_replay(&normalized);
}

#[tokio::test]
async fn missing_call_ids_remain_distinct_and_do_not_collide_with_explicit_ids() {
    let first = partial_without_terminal(missing_id_fixture()).await;
    let second = partial_without_terminal(missing_id_fixture()).await;
    let calls = |response: &CompletionResponse| {
        response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let a = calls(&first);
    let b = calls(&second);
    for calls in [&a, &b] {
        assert_eq!(calls.len(), 3);
        assert!(
            calls
                .iter()
                .all(|call| call.function.name.as_str() == "same")
        );
        assert!(calls[0].id.is_local() && calls[0].id.provider().is_none());
        assert_eq!(calls[1].id.provider().map(|id| id.as_str()), Some("tool-0"));
        assert!(calls[2].id.is_local() && calls[2].id.provider().is_none());
        assert_ne!(calls[0].id, calls[1].id);
        assert_ne!(calls[0].id, calls[2].id);
        assert_ne!(calls[1].id, calls[2].id);
    }
    assert_ne!(a[0].id, b[0].id);
    assert_ne!(a[2].id, b[2].id);
    assert_eq!(a[1].id, b[1].id);
}

#[test]
fn generate_content_response_round_trips_through_serde_json_value() {
    let raw = raw_fixture();
    let value = crate::rest::to_rest(&raw).expect("transcodes");
    assert_eq!(
        value.pointer("/usageMetadata/cachedContentTokenCount"),
        Some(&json!(4))
    );
    assert_eq!(
        value.pointer("/candidates/0/finishMessage"),
        Some(&json!("done"))
    );
    assert_eq!(
        value.pointer("/modelVersion"),
        Some(&json!("gemini-2.5-flash"))
    );
    let back: proto::GenerateContentResponse =
        crate::rest::from_rest(value.clone()).expect("reads back");
    assert_eq!(crate::rest::to_rest(&back).expect("retranscodes"), value);
    assert_eq!(back, raw);
    let original = super::tests::complete(raw).expect("original converts");
    let restored = super::tests::complete(back).expect("restored converts");
    assert_eq!(original.raw, value);
    assert_eq!(restored.raw, value);
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model(), original.model());
    assert_eq!(restored.model(), Some("gemini-2.5-flash"));
    assert_eq!(restored.text(), "hello");
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-grpc-1")
    );
    assert_eq!(restored.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(restored.usage.input_tokens, Some(10));
    assert_eq!(restored.usage.output_tokens, Some(20));
    assert_eq!(restored.usage.total_tokens, Some(30));
    assert_eq!(restored.usage.cached_input_tokens, Some(4));
}

mod signed_answer_issuer_controls;
