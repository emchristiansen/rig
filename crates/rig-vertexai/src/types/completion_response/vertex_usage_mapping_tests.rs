use super::*;

/// The tool-use prompt is input, as Vertex's `totalTokenCount` counts it:
/// 100 + (60 + 15) + 30 + 34 = 239. Vertex reports it only per modality, so
/// its count is the breakdown's sum. No cassette harness exists for
/// rig-vertexai, so the mapping's arithmetic is pinned here.
#[test]
fn tool_use_prompt_tokens_are_input_in_the_real_conversion() {
    let modality = |modality, tokens| {
        vertexai::model::ModalityTokenCount::new()
            .set_modality(modality)
            .set_token_count(tokens)
    };
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(100)
        .set_tool_use_prompt_tokens_details([
            modality(vertexai::model::Modality::Text, 60),
            modality(vertexai::model::Modality::Image, 15),
        ])
        .set_candidates_token_count(30)
        .set_thoughts_token_count(34)
        .set_total_token_count(239)
        .set_cached_content_token_count(40);

    let response = vertexai::model::GenerateContentResponse::new()
        .set_usage_metadata(usage_metadata)
        .set_candidates(vec![
            vertexai::model::Candidate::new()
                .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
                .set_content(
                    vertexai::model::Content::new()
                        .set_role("model")
                        .set_parts(vec![vertexai::model::Part::new().set_text("hi")]),
                ),
        ]);

    let converted = crate::types::completion_response::tests::complete(response)
        .expect("a response with content should convert");

    let usage = converted.usage;
    assert_eq!(usage.input_tokens, Some(175));
    assert_eq!(usage.tool_use_prompt_tokens, Some(75));
    assert_eq!(usage.cached_input_tokens, Some(40));
    assert_eq!(usage.output_tokens, Some(64));
    assert_eq!(usage.reasoning_tokens, Some(34));
    assert_eq!(usage.total_tokens, Some(239));
}


/// Supply one actual SDK response with the same whole-document capture as the Vertex transport.
#[derive(Clone)]
struct SingleReply(vertexai::model::GenerateContentResponse);

impl rig_core::driver::Transport<crate::completion::GenerateContent> for SingleReply {
    fn send(
        &self,
        _request: vertexai::model::GenerateContentRequest,
        _exchange: rig_core::driver::Exchange,
    ) -> rig_core::driver::Opening<vertexai::model::GenerateContentResponse> {
        let response = self.0.clone();
        rig_core::driver::Opening::new(async move {
            let document = serde_json::Value::Object(rest_chunk(&response)?);
            Ok(rig_core::driver::Opened::new(futures::stream::iter([Ok(response)]))
                .with_document(document))
        })
    }
}

/// The original Vertex thinking-token reply, which states no finish reason.
fn thinking_tokens_without_finish_fixture() -> vertexai::model::GenerateContentResponse {
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(14)
        .set_candidates_token_count(34)
        .set_thoughts_token_count(222)
        .set_total_token_count(270)
        .set_cached_content_token_count(9);

    let response = vertexai::model::GenerateContentResponse::new()
        .set_usage_metadata(usage_metadata)
        .set_candidates(vec![
            vertexai::model::Candidate::new().set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts(vec![vertexai::model::Part::new().set_text("hi")]),
            ),
        ]);

    response
}

/// Without a provider end, usage remains raw evidence and unary completion is truncated.
#[test]
fn thinking_tokens_without_a_finish_remain_raw_and_unary_is_truncated() {
    use futures::StreamExt as _;

    let received = thinking_tokens_without_finish_fixture();
    let expected_raw = serde_json::Value::Object(
        rest_chunk(&received).expect("the exact received SDK response transcodes"),
    );
    let model = rig_core::driver::Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        SingleReply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(rig_core::completion::CompletionRequest::new("hello"))
            .expect("the original response opens");
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            if let Err(error) = item {
                assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                assert!(!saw_truncated, "the original reply fails once");
                saw_truncated = true;
            }
        }
        assert!(saw_truncated, "a reply without a finish reason is truncated");
        let partial = stream.partial();
        assert_eq!(partial.text(), "hi");
        assert_eq!(partial.choice.len(), 1);
        assert_eq!(partial.usage, rig_core::completion::Usage::default());
        assert_eq!(partial.finish_reason(), None);
        assert_eq!(partial.raw, expected_raw);
        for (field, count) in [
            ("promptTokenCount", 14),
            ("candidatesTokenCount", 34),
            ("thoughtsTokenCount", 222),
            ("totalTokenCount", 270),
            ("cachedContentTokenCount", 9),
        ] {
            assert_eq!(partial.raw["usageMetadata"][field], serde_json::json!(count));
        }
        assert_eq!(partial.raw["candidates"][0]["content"]["parts"], serde_json::json!([{ "text": "hi" }]));
        assert!(partial.raw["candidates"][0].get("finishReason").is_none());
        assert!(matches!(stream.finish().await, Err(ProviderError::Truncated)));
        assert!(matches!(
            model.call(rig_core::completion::CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Adding only STOP makes the original candidate and reasoning counts canonical usage.
#[test]
fn completed_thinking_tokens_include_candidate_and_reasoning_output() {
    let mut received = thinking_tokens_without_finish_fixture();
    received.candidates[0].finish_reason = vertexai::model::candidate::FinishReason::Stop;
    let expected_raw = serde_json::Value::Object(
        rest_chunk(&received).expect("the completed SDK response transcodes"),
    );
    let model = rig_core::driver::Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        SingleReply(received),
    );
    let converted = futures::executor::block_on(
        model.call(rig_core::completion::CompletionRequest::new("hello")),
    ).expect("the original response with only STOP added completes");
    assert_eq!(converted.text(), "hi");
    assert_eq!(converted.choice.len(), 1);
    assert_eq!(converted.finish_reason(), Some(rig_core::completion::FinishReason::Stop));
    assert_eq!(converted.raw, expected_raw);
    assert_eq!(converted.raw["candidates"][0]["finishReason"], serde_json::json!("STOP"));
    assert_eq!(converted.usage.reasoning_tokens, Some(222));
    assert_eq!(converted.usage.cached_input_tokens, Some(9));
    assert_eq!(converted.usage.input_tokens, Some(14));
    assert_eq!(converted.usage.output_tokens, Some(256));
    assert_eq!(converted.usage.total_tokens, Some(270));
}
