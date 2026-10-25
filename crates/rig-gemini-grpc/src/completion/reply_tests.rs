use super::*;
use crate::proto;
use anyhow::{Context, bail, ensure};
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

async fn partial_without_terminal(
    response: proto::GenerateContentResponse,
) -> anyhow::Result<CompletionResponse> {
    let unary = super::tests::complete(response.clone());
    ensure!(
        matches!(&unary, Err(ProviderError::Truncated)),
        "the original fixture is also truncated on the actual unary path: {unary:?}"
    );
    let mut stream = super::tests::stream_from_events(vec![Ok(response)]);
    let mut truncated = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(_) => {}
            Err(ProviderError::Truncated) => truncated = true,
            Err(error) => bail!("unexpected error: {error:?}"),
        }
    }
    let partial = stream.partial();
    ensure!(truncated, "the original fixture has no genuine terminal");
    let finished = stream.finish().await;
    ensure!(
        matches!(&finished, Err(ProviderError::Truncated)),
        "expected the terminal-less stream to finish as Truncated: {finished:?}"
    );
    Ok(partial)
}

#[tokio::test]
async fn text_response_skips_thought_parts() -> anyhow::Result<()> {
    let partial = partial_without_terminal(thought_fixture()).await?;
    ensure!(
        partial.text() == "The answer is 42.",
        "reasoning must not be reported as the response text: {:?}",
        partial.text()
    );
    Ok(())
}

#[test]
fn completed_text_response_skips_thought_parts() -> anyhow::Result<()> {
    let mut response = thought_fixture();
    response
        .candidates
        .first_mut()
        .context("the thought fixture has a candidate")?
        .finish_reason = proto::candidate::FinishReason::Stop as i32;
    let response = super::tests::complete(response)?;
    ensure!(
        response.text() == "The answer is 42.",
        "expected the completed answer text, got {:?}",
        response.text()
    );
    Ok(())
}

fn assert_signed_answer_replay(normalized: &CompletionResponse) -> anyhow::Result<()> {
    ensure!(normalized.choice.len() == 2, "{:?}", normalized.choice);
    ensure!(
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
    .context("the answer text carries its signature")?;
    let request = CompletionRequest::from(vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(normalized.choice.clone())),
        Message::user("again"),
    ]);
    let req = GenerateContent::new("gemini-3-flash-preview").encode(request, Mode::Unary)?;
    let answer = req
        .contents
        .get(1)
        .context("assistant turn")?
        .parts
        .iter()
        .find(|part| matches!(&part.data, Some(proto::part::Data::Text(text)) if text == "answer"))
        .context("the answer part")?;
    ensure!(
        answer.thought_signature == b"sig-bytes".to_vec(),
        "expected the answer signature {:?}, got {:?}",
        b"sig-bytes".to_vec(),
        answer.thought_signature
    );
    ensure!(!signature.is_empty(), "the answer signature is nonempty");
    Ok(())
}

#[tokio::test]
async fn a_signature_on_answer_text_stays_on_that_text() -> anyhow::Result<()> {
    let partial = partial_without_terminal(signed_answer_fixture()).await?;
    assert_signed_answer_replay(&partial)?;
    Ok(())
}

#[test]
fn completed_answer_text_keeps_and_replays_its_signature() -> anyhow::Result<()> {
    let mut response = signed_answer_fixture();
    response
        .candidates
        .first_mut()
        .context("the signed-answer fixture has a candidate")?
        .finish_reason = proto::candidate::FinishReason::Stop as i32;
    let normalized = super::tests::complete(response)?;
    assert_signed_answer_replay(&normalized)?;
    Ok(())
}

#[tokio::test]
async fn missing_call_ids_remain_distinct_and_do_not_collide_with_explicit_ids()
-> anyhow::Result<()> {
    let first = partial_without_terminal(missing_id_fixture()).await?;
    let second = partial_without_terminal(missing_id_fixture()).await?;
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
        ensure!(calls.len() == 3, "expected three calls, got {calls:?}");
        ensure!(
            calls
                .iter()
                .all(|call| call.function.name.as_str() == "same"),
            "expected every call to be named same, got {calls:?}"
        );
        let [first, explicit, third] = calls.as_slice() else {
            bail!("expected three calls, got {calls:?}");
        };
        ensure!(
            first.id.is_local() && first.id.provider().is_none(),
            "the first call ID must be local without a provider ID: {:?}",
            first.id
        );
        ensure!(
            explicit.id.provider().map(|id| id.as_str()) == Some("tool-0"),
            "expected the explicit tool-0 provider ID, got {:?}",
            explicit.id
        );
        ensure!(
            third.id.is_local() && third.id.provider().is_none(),
            "the third call ID must be local without a provider ID: {:?}",
            third.id
        );
        ensure!(first.id != explicit.id, "call IDs must differ: {calls:?}");
        ensure!(first.id != third.id, "call IDs must differ: {calls:?}");
        ensure!(explicit.id != third.id, "call IDs must differ: {calls:?}");
    }
    ensure!(
        a.first().context("first reply's first call")?.id
            != b.first().context("second reply's first call")?.id,
        "first local IDs must differ between replies: {a:?}, {b:?}"
    );
    ensure!(
        a.get(2).context("first reply's third call")?.id
            != b.get(2).context("second reply's third call")?.id,
        "third local IDs must differ between replies: {a:?}, {b:?}"
    );
    ensure!(
        a.get(1).context("first reply's explicit call")?.id
            == b.get(1).context("second reply's explicit call")?.id,
        "explicit provider IDs must agree between replies: {a:?}, {b:?}"
    );
    Ok(())
}

#[test]
fn generate_content_response_round_trips_through_serde_json_value() -> anyhow::Result<()> {
    let raw = raw_fixture();
    let value = crate::rest::to_rest(&raw)?;
    ensure!(
        value.pointer("/usageMetadata/cachedContentTokenCount") == Some(&json!(4)),
        "expected cachedContentTokenCount 4, got {value:?}"
    );
    ensure!(
        value.pointer("/candidates/0/finishMessage") == Some(&json!("done")),
        "expected candidate finishMessage done, got {value:?}"
    );
    ensure!(
        value.pointer("/modelVersion") == Some(&json!("gemini-2.5-flash")),
        "expected modelVersion gemini-2.5-flash, got {value:?}"
    );
    let back: proto::GenerateContentResponse = crate::rest::from_rest(value.clone())?;
    ensure!(
        crate::rest::to_rest(&back)? == value,
        "expected the restored protobuf's full REST value {value:?}, got protobuf {back:?}"
    );
    ensure!(
        back == raw,
        "protobuf round trip differs: {back:?}, {raw:?}"
    );
    let original = super::tests::complete(raw)?;
    let restored = super::tests::complete(back)?;
    ensure!(
        original.raw == value,
        "original raw differs: {:?}, {value:?}",
        original.raw
    );
    ensure!(
        restored.raw == value,
        "restored raw differs: {:?}, {value:?}",
        restored.raw
    );
    ensure!(
        restored.identity() == original.identity(),
        "identity differs: {:?}, {:?}",
        restored.identity(),
        original.identity()
    );
    ensure!(
        restored.finish_reason() == original.finish_reason(),
        "finish reason differs: {:?}, {:?}",
        restored.finish_reason(),
        original.finish_reason()
    );
    ensure!(
        restored.model() == original.model(),
        "model differs: {:?}, {:?}",
        restored.model(),
        original.model()
    );
    ensure!(
        restored.model() == Some("gemini-2.5-flash"),
        "expected model gemini-2.5-flash, got {:?}",
        restored.model()
    );
    ensure!(
        restored.text() == "hello",
        "expected text hello, got {:?}",
        restored.text()
    );
    ensure!(
        restored.usage == original.usage,
        "usage differs: {:?}, {:?}",
        restored.usage,
        original.usage
    );
    ensure!(
        restored.choice == original.choice,
        "choice differs: {:?}, {:?}",
        restored.choice,
        original.choice
    );
    ensure!(
        restored.identity().response_id.as_deref() == Some("resp-grpc-1"),
        "expected response ID resp-grpc-1, got {:?}",
        restored.identity().response_id
    );
    ensure!(
        restored.finish_reason() == Some(FinishReason::Stop),
        "expected Stop, got {:?}",
        restored.finish_reason()
    );
    ensure!(
        restored.usage.input_tokens == Some(10),
        "expected 10 input tokens, got {:?}",
        restored.usage.input_tokens
    );
    ensure!(
        restored.usage.output_tokens == Some(20),
        "expected 20 output tokens, got {:?}",
        restored.usage.output_tokens
    );
    ensure!(
        restored.usage.total_tokens == Some(30),
        "expected 30 total tokens, got {:?}",
        restored.usage.total_tokens
    );
    ensure!(
        restored.usage.cached_input_tokens == Some(4),
        "expected 4 cached input tokens, got {:?}",
        restored.usage.cached_input_tokens
    );
    Ok(())
}

mod signed_answer_issuer_controls;
