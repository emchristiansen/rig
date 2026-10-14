use super::*;
use crate::driver::Model;
use crate::providers::gemini::{GeminiConfig, completion::GenerateContent};
use crate::providers::openai::{OpenAIConfig, responses_api::wire::Responses};
use crate::test_utils::{
    MockCompletionModel, MockHttpResponse, MockStreamEvent, SequencedHttpClient,
};
use serde_json::json;

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = panic.downcast_ref::<String>() {
        message
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        message
    } else {
        "non-text panic"
    }
}

/// Synthetic fixture isolates canonical conservation through the existing
/// Responses decoder rather than recording a new provider exchange.
#[tokio::test]
async fn rich_carried_text_conserves_deltas_and_missing_choice_fails_law_two()
-> Result<(), ProviderError> {
    // Required envelope and event coordinates are synthetic received data.
    let part = json!({"type":"output_text", "text":"Paris", "annotations":[{
        "type":"url_citation", "url":"https://example.test/paris", "title":"Paris",
        "start_index":0, "end_index":5
    }], "futureField":{"text":"London"}});
    let message = json!({"type":"message", "id":"msg_ut005", "role":"assistant",
        "status":"completed", "phase":"final_answer", "content":[part.clone()]});
    let body = json!({"id":"resp_ut005", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-4o-mini", "tools":[], "output":[message.clone()],
        "usage":{"input_tokens":4, "output_tokens":2, "total_tokens":6}});
    let frames = [
        json!({"type":"response.output_text.delta", "sequence_number":1,
            "output_index":0, "content_index":0, "item_id":"msg_ut005", "delta":"Pa"}),
        json!({"type":"response.output_text.delta", "sequence_number":2,
            "output_index":0, "content_index":0, "item_id":"msg_ut005", "delta":"ris"}),
        json!({"type":"response.output_item.done", "sequence_number":3,
            "output_index":0, "item":message}),
        json!({"type":"response.completed", "sequence_number":4, "response":body}),
    ];
    let sse: String = frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect();
    let http =
        SequencedHttpClient::new([MockHttpResponse::success_typed(sse, "text/event-stream")]);
    let model = Model::new(
        Responses::new(OpenAIConfig::new("synthetic-test-key"), "gpt-4o-mini"),
        http.clone(),
    );
    let drained =
        fixtures::drain(model.stream(CompletionRequest::new("synthetic conservation control"))?)
            .await;
    assert!(drained.completed_cleanly());
    assert_eq!(drained.texts().concat(), "Paris");
    assert_eq!(drained.choice_texts(), vec!["Paris"]);
    assert!(
        matches!(drained.choice.as_slice(), [AssistantContent::CarriedText(text)]
        if text.text() == "Paris" && text.phase.as_deref() == Some("final_answer")
            && text.payload.item == part && text.payload.provider.as_deref() == Some("openai"))
    );
    let response = drained.response.as_ref().ok_or_else(|| {
        ProviderError::Response("synthetic fixture must reach its terminal".to_owned())
    })?;
    assert_eq!(response.raw, body);
    assert_eq!(response.usage.input_tokens, Some(4));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(6));
    assert_eq!(http.requests().len(), 1);
    assert_eq!(http.remaining_responses(), 0);

    // Deliberately corrupt only the synthetic consumer observation, not
    // received provider data: missing canonical text must still fail law 2.
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_valid_event_stream(&drained.items, &[]);
    }));
    let Err(panic) = failure else {
        panic!("missing canonical carried text must violate conservation");
    };
    assert!(panic_message(panic.as_ref()).contains("law 2 (text conservation)"));
    Ok(())
}

/// Synthetic Gemini control covers the real signed/plain decoder route and
/// ordinary text order, without giving payload-only siblings text authority.
#[tokio::test]
async fn ordinary_and_mixed_text_keep_order_without_raw_only_promotion() -> Result<(), ProviderError>
{
    for signed in [false, true] {
        let first = if signed {
            json!({"text":"Pa", "thoughtSignature":"synthetic-signed-pa",
                "futureField":{"text":"London"}})
        } else {
            json!({"text":"Pa"})
        };
        let body = json!({"responseId":"resp_ut005_gemini", "modelVersion":"gemini-2.5-flash",
            "candidates":[{"index":0, "content":{"role":"model",
                "parts":[first.clone(), {"text":"ris"}]}, "finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":4, "candidatesTokenCount":2, "totalTokenCount":6}});
        let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
            format!("data: {body}\n\n"),
            "text/event-stream",
        )]);
        let model = Model::new(
            GenerateContent::new(GeminiConfig::new("synthetic-test-key"), "gemini-2.5-flash"),
            http.clone(),
        );
        let drained =
            fixtures::drain(model.stream(CompletionRequest::new("synthetic text-order control"))?)
                .await;
        assert!(drained.completed_cleanly());
        assert_eq!(drained.texts().concat(), "Paris");
        assert_eq!(drained.choice_texts(), vec!["Pa", "ris"]);
        if signed {
            assert!(matches!(drained.choice.as_slice(),
                [AssistantContent::CarriedText(carried), AssistantContent::Text(plain)]
                if carried.text() == "Pa" && plain.text == "ris"
                    && carried.payload.item == first
                    && carried.payload.provider.as_deref() == Some("gcp.gemini")
                    && carried.signed_answer().map(|answer| answer.signature.as_str()) == Some("synthetic-signed-pa")));
        } else {
            assert!(matches!(drained.choice.as_slice(),
                [AssistantContent::Text(first), AssistantContent::Text(second)]
                if first.text == "Pa" && second.text == "ris"));
        }
        let response = drained.response.as_ref().ok_or_else(|| {
            ProviderError::Response("synthetic fixture must reach its terminal".to_owned())
        })?;
        assert_eq!(response.text(), "Paris");
        assert_eq!(response.usage.total_tokens, Some(6));
        // The streamed document deliberately merges adjacent parts. Pin
        // unchanged fields without claiming unary/raw part-vector equality.
        assert_eq!(response.raw["responseId"], body["responseId"]);
        assert_eq!(response.raw["usageMetadata"], body["usageMetadata"]);
        assert_eq!(http.requests().len(), 1);
        assert_eq!(http.remaining_responses(), 0);
    }
    Ok(())
}

/// Synthetic signed-thought fixture exercises the existing Gemini producer;
/// the negative half alters only an observation by withholding its real End.
#[tokio::test]
async fn carried_reasoning_requires_its_actual_end_for_law_five() -> Result<(), ProviderError> {
    let thought = json!({"text":"think", "thought":true, "thoughtSignature":"synthetic-thought"});
    let body = json!({"responseId":"resp_ut005_thought", "modelVersion":"gemini-2.5-flash",
        "candidates":[{"index":0, "content":{"role":"model",
            "parts":[thought.clone(), {"text":"Paris"}]}, "finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":4, "candidatesTokenCount":2,
            "thoughtsTokenCount":1, "totalTokenCount":7}});
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        format!("data: {body}\n\n"),
        "text/event-stream",
    )]);
    let model = Model::new(
        GenerateContent::new(GeminiConfig::new("synthetic-test-key"), "gemini-2.5-flash"),
        http.clone(),
    );
    let mut drained =
        fixtures::drain(model.stream(CompletionRequest::new("synthetic reasoning control"))?).await;
    assert!(drained.completed_cleanly());
    assert!(matches!(drained.choice.as_slice(),
        [AssistantContent::CarriedReasoning(reasoning), AssistantContent::Text(text)]
        if reasoning.display_text() == "think" && reasoning.payload.item == thought
            && reasoning.payload.provider.as_deref() == Some("gcp.gemini") && text.text == "Paris"));
    assert_eq!(drained.choice_reasoning_texts(), vec!["think".to_owned()]);
    assert!(
        drained.choice_reasoning().is_empty(),
        "the public helper retains its ordinary-only contract"
    );
    assert_eq!(
        drained
            .items
            .iter()
            .filter(|item| matches!(
                item,
                Ok(Item::Event(StreamEvent::End {
                    content: AssistantContent::CarriedReasoning(_),
                    ..
                }))
            ))
            .count(),
        1
    );
    let response = drained.response.as_ref().ok_or_else(|| {
        ProviderError::Response("synthetic fixture must reach its terminal".to_owned())
    })?;
    assert_eq!(response.raw, body);
    assert_eq!(response.usage.reasoning_tokens, Some(1));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(http.requests().len(), 1);
    let before = drained.items.len();
    // Synthetic observation loss: retain all original decoder results and
    // choice, but withhold exactly its one carried-reasoning End from input.
    drained.items.retain(|item| {
        !matches!(
            item,
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::CarriedReasoning(_),
                ..
            }))
        )
    });
    assert_eq!(drained.items.len(), before - 1);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_valid_event_stream(&drained.items, &drained.choice);
    }));
    let Err(panic) = failure else {
        panic!("withheld reasoning End must violate provenance");
    };
    assert!(panic_message(panic.as_ref()).contains("law 5 (reasoning provenance)"));
    Ok(())
}

/// Existing mock operations produce ordinary reasoning through the same
/// writer/fold; this keeps the old public helper and law-5 route covered.
#[tokio::test]
async fn ordinary_reasoning_retains_public_and_private_projection() -> Result<(), ProviderError> {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::reasoning("plain thought"),
        MockStreamEvent::text("Paris"),
        MockStreamEvent::final_response_with_total_tokens(6),
    ]]);
    let drained =
        fixtures::drain(model.stream(CompletionRequest::new("synthetic ordinary control"))?).await;
    assert!(drained.completed_cleanly());
    assert!(matches!(drained.choice.as_slice(),
        [AssistantContent::Reasoning(reasoning), AssistantContent::Text(text)]
        if reasoning.text == "plain thought" && text.text == "Paris"));
    assert_eq!(drained.choice_reasoning().len(), 1);
    assert_eq!(
        drained.choice_reasoning_texts(),
        vec!["plain thought".to_owned()]
    );
    assert_eq!(drained.choice_texts(), vec!["Paris"]);
    assert_eq!(model.request_count(), 1);
    Ok(())
}

/// This uses the unchanged existing synthetic interleaving fixture, rather
/// than minting a signature or substituting Native presence for one.
#[tokio::test]
async fn unchanged_signed_interleaving_fixture_keeps_signature_and_order()
-> Result<(), ConformanceError> {
    let (frames, first, tool_name, second) =
        fixtures::gemini_rest::interleaved_signed_thought_frames();
    let driver = fixtures::gemini_rest::fixture().driver;
    let drained = driver.drive(ok_chunks(frames.clone())).await?;
    assert!(drained.completed_cleanly());
    let [
        AssistantContent::CarriedReasoning(before),
        AssistantContent::ToolCall(call),
        AssistantContent::CarriedReasoning(after),
    ] = drained.choice.as_slice()
    else {
        panic!("the unchanged fixture keeps reasoning, call, reasoning in order");
    };
    assert_eq!(before.display_text(), first);
    assert_eq!(call.function.name.as_str(), tool_name);
    assert_eq!(after.display_text(), second);
    assert_eq!(after.payload.provider.as_deref(), Some("gcp.gemini"));
    assert_eq!(
        after.payload.item,
        json!({"text":"signed conclusion", "thought":true, "thoughtSignature":"sig-1"})
    );
    assert!(
        matches!(after.content.as_slice(), [crate::message::CarriedReasoningPart::Text {text, signature:Some(signature)}]
        if text == second && signature == "sig-1")
    );
    assert!(drained.choice_reasoning().is_empty());
    let report = interleaved_signed_full_reasoning_does_not_erase_prior_thought(
        &driver, frames, first, tool_name, second,
    )
    .await?;
    assert_eq!(
        report.name,
        "interleaved_signed_full_reasoning_does_not_erase_prior_thought"
    );
    Ok(())
}

/// A new unsigned synthetic control removes only thoughtSignature from the
/// existing received fixture and must fail the exact signature requirement.
#[tokio::test]
async fn unsigned_interleaving_fixture_refuses_the_signature_requirement()
-> Result<(), ConformanceError> {
    let (mut frames, first, tool_name, second) =
        fixtures::gemini_rest::interleaved_signed_thought_frames();
    let original = frames[2]
        .as_bytes()
        .map(|bytes| bytes.as_ref())
        .ok_or_else(|| {
            ConformanceError::contract(
                "unsigned_synthetic_fixture",
                "gemini",
                "the existing fixture is SSE bytes",
            )
        })?;
    let text = std::str::from_utf8(original).map_err(|error| {
        ConformanceError::contract("unsigned_synthetic_fixture", "gemini", error.to_string())
    })?;
    let data = text
        .strip_prefix("data: ")
        .and_then(|text| text.strip_suffix("\n\n"))
        .ok_or_else(|| {
            ConformanceError::contract(
                "unsigned_synthetic_fixture",
                "gemini",
                "the existing frame is one SSE data record",
            )
        })?;
    let original_value: serde_json::Value = serde_json::from_str(data).map_err(|error| {
        ConformanceError::contract("unsigned_synthetic_fixture", "gemini", error.to_string())
    })?;
    let mut unsigned = original_value.clone();
    let part = unsigned["candidates"][0]["content"]["parts"][0]
        .as_object_mut()
        .ok_or_else(|| {
            ConformanceError::contract(
                "unsigned_synthetic_fixture",
                "gemini",
                "the thought part is an object",
            )
        })?;
    assert_eq!(part.remove("thoughtSignature"), Some(json!("sig-1")));
    assert_eq!(
        unsigned["candidates"][0]["content"]["parts"][0],
        json!({"text":"signed conclusion", "thought":true})
    );
    frames[2] = WireInput::Bytes(Bytes::from(format!("data: {unsigned}\n\n")));
    let driver = fixtures::gemini_rest::fixture().driver;
    let drained = driver.drive(ok_chunks(frames.clone())).await?;
    assert!(drained.completed_cleanly());
    let [
        AssistantContent::CarriedReasoning(before),
        AssistantContent::ToolCall(call),
        AssistantContent::CarriedReasoning(after),
    ] = drained.choice.as_slice()
    else {
        panic!("withholding the signature must not change canonical interleaving");
    };
    assert_eq!(before.display_text(), first);
    assert_eq!(call.function.name.as_str(), tool_name);
    assert_eq!(after.display_text(), second);
    assert_eq!(
        after.payload.item,
        json!({"text":"signed conclusion", "thought":true})
    );
    assert!(
        matches!(after.content.as_slice(), [crate::message::CarriedReasoningPart::Text {text, signature:None}]
        if text == second)
    );
    let result = interleaved_signed_full_reasoning_does_not_erase_prior_thought(
        &driver, frames, first, tool_name, second,
    )
    .await;
    let Err(ConformanceError::Contract {
        scenario, details, ..
    }) = result
    else {
        panic!("the unsigned fixture must fail the signature requirement");
    };
    assert_eq!(
        scenario,
        "interleaved_signed_full_reasoning_does_not_erase_prior_thought"
    );
    assert_eq!(details, "the post-boundary block must keep its signature");
    Ok(())
}

// The event sequence is produced by the real Mock driver. Only the inserted
// error observations below are synthetic.
async fn law_one_completed_mock_text() -> Result<DrainedStream, ProviderError> {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("Paris"),
        MockStreamEvent::final_response_with_total_tokens(6),
    ]]);
    let drained = fixtures::drain(model.stream(CompletionRequest::new(
        "synthetic error observation control",
    ))?)
    .await;
    assert!(drained.completed_cleanly());
    assert_eq!(drained.texts().concat(), "Paris");
    assert_eq!(drained.choice_texts(), vec!["Paris"]);
    assert_eq!(model.request_count(), 1);
    Ok(drained)
}

fn law_one_synthetic_corrupt_report() -> ErrorReport {
    let Err(error) = serde_json::from_str::<serde_json::Value>("{") else {
        panic!("the synthetic incomplete object must fail JSON parsing");
    };
    let error =
        ProviderError::CorruptFrame(crate::error::CorruptFrame::text(None, "{", error).into());
    let report = ErrorReport::from(&error);
    assert!(matches!(
        report.detail,
        Some(crate::error::ErrorDetail::CorruptFrame(_))
    ));
    report
}

#[tokio::test]
async fn law_one_typed_corrupt_report_allows_genuine_later_events() -> Result<(), ProviderError> {
    let mut drained = law_one_completed_mock_text().await?;
    drained
        .items
        .insert(0, Err(law_one_synthetic_corrupt_report()));
    assert_valid_event_stream(&drained.items, &drained.choice);
    Ok(())
}

#[tokio::test]
async fn law_one_response_error_refuses_genuine_later_events() -> Result<(), ProviderError> {
    let mut drained = law_one_completed_mock_text().await?;
    // The message deliberately mentions corruption; it confers no recovery.
    let report = ErrorReport::from(&ProviderError::Response("corrupt frame".to_owned()));
    assert!(report.detail.is_none());
    drained.items.insert(0, Err(report));
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_valid_event_stream(&drained.items, &drained.choice);
    }));
    let Err(panic) = failure else {
        panic!("an ordinary Response error must be terminal");
    };
    assert!(panic_message(panic.as_ref()).contains("law 1 (terminal error)"));
    Ok(())
}

#[tokio::test]
async fn law_one_missing_corrupt_detail_refuses_genuine_later_events() -> Result<(), ProviderError>
{
    let mut drained = law_one_completed_mock_text().await?;
    let mut report = law_one_synthetic_corrupt_report();
    report.detail = None;
    drained.items.insert(0, Err(report));
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_valid_event_stream(&drained.items, &drained.choice);
    }));
    let Err(panic) = failure else {
        panic!("an error without the typed corrupt detail must be terminal");
    };
    assert!(panic_message(panic.as_ref()).contains("law 1 (terminal error)"));
    Ok(())
}

#[tokio::test]
async fn law_one_fatal_report_last_keeps_genuine_text_events() -> Result<(), ProviderError> {
    let mut drained = law_one_completed_mock_text().await?;
    let report = ErrorReport::from(&ProviderError::Response("fatal control".to_owned()));
    assert!(report.detail.is_none());
    drained.items.push(Err(report));
    assert_valid_event_stream(&drained.items, &drained.choice);
    Ok(())
}

/// The original Responses fixture has an explicit completed item before EOF.
#[tokio::test]
async fn completed_responses_call_survives_eof_exactly_once() -> Result<(), ConformanceError> {
    truncation_preserves_content_without_terminal(&fixtures::openai_responses::fixture()).await?;
    Ok(())
}

/// Terminal transport cleanup flushes the same completed item before its error.
#[tokio::test]
async fn completed_responses_call_precedes_transport_error_exactly_once()
-> Result<(), ConformanceError> {
    transport_error_after_tool_call_yields_err_then_end(&fixtures::openai_responses::fixture())
        .await?;
    Ok(())
}

/// Synthetic consumer observations remove or duplicate a genuine delivered End.
/// Provider input and its actual driver remain unchanged.
fn synthetic_call_observation_fixture(duplicate: bool) -> ProviderWireFixture {
    let mut fixture = fixtures::openai_responses::fixture();
    let driver = std::sync::Arc::new(fixture.driver);
    fixture.driver = WireDriver::new("synthetic call observation", move |chunks| {
        let driver = driver.clone();
        Box::pin(async move {
            let mut drained = driver.drive(chunks).await?;
            let call_end = drained.items.iter().position(|item| {
                matches!(
                    item,
                    Ok(Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(_),
                        ..
                    }))
                )
            });
            if let Some(index) = call_end {
                if duplicate {
                    let repeated = drained.items[index].clone();
                    drained.items.insert(index + 1, repeated);
                } else {
                    drop(drained.items.remove(index));
                }
                // The exact old universal predicate accepts both missing and
                // repeated observations; it cannot establish multiplicity.
                assert!(
                    drained
                        .tool_call_names()
                        .iter()
                        .all(|name| *name == "get_weather")
                );
            }
            Ok(drained)
        })
    });
    fixture
}

#[tokio::test]
async fn synthetic_missing_completed_call_fails_both_scenario_counts() {
    for transport in [false, true] {
        let fixture = synthetic_call_observation_fixture(false);
        let result = if transport {
            transport_error_after_tool_call_yields_err_then_end(&fixture).await
        } else {
            truncation_preserves_content_without_terminal(&fixture).await
        };
        let error = result.expect_err("missing delivered call must fail exact-once");
        let ConformanceError::Contract { details, .. } = error else {
            panic!("the synthetic observation must fail the count predicate");
        };
        assert!(details.contains("exactly once"));
    }
}

#[tokio::test]
async fn synthetic_duplicate_completed_call_fails_both_scenario_counts() {
    for transport in [false, true] {
        let fixture = synthetic_call_observation_fixture(true);
        let result = if transport {
            transport_error_after_tool_call_yields_err_then_end(&fixture).await
        } else {
            truncation_preserves_content_without_terminal(&fixture).await
        };
        let error = result.expect_err("repeated delivered call must fail exact-once");
        let ConformanceError::Contract { details, .. } = error else {
            panic!("the synthetic observation must fail the count predicate");
        };
        assert!(details.contains("exactly once"));
    }
}
