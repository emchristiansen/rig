use super::*;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{AssistantContent, SourceLocation};
use rig_core::wire::document::Reassemble;
use serde_json::json;

/// Answers every request with one scripted SDK reply.
#[derive(Clone)]
pub(crate) struct Reply(vertexai::model::GenerateContentResponse);

impl Transport<crate::completion::GenerateContent> for Reply {
    fn send(
        &self,
        _payload: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

/// The content a turn holding `choice` replays as, encoded for the model
/// the replies come from.
pub(crate) fn replay(
    choice: Vec<rig_core::message::AssistantContent>,
) -> Result<vertexai::model::Content, rig_core::error::EncodeError> {
    use rig_core::wire::{Mode, Wire};
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![rig_core::message::Message::from(choice)];
    let request = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)?;
    request
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| rig_core::error::EncodeError::request("no content"))
}

/// The reply as the unary endpoint answers it.
pub(crate) trait Complete {
    fn complete(self) -> Result<CompletionResponse, ProviderError>;
}

impl Complete for vertexai::model::GenerateContentResponse {
    fn complete(self) -> Result<CompletionResponse, ProviderError> {
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            Reply(self),
        );
        futures::executor::block_on(model.call(CompletionRequest::new("hello")))
    }
}

pub(crate) fn complete(
    response: vertexai::model::GenerateContentResponse,
) -> Result<CompletionResponse, ProviderError> {
    response.complete()
}

#[test]
fn sdk_grounding_attaches_before_signed_text_close_and_raw_stays_numeric() {
    const ANSWER: &str = "Spain won 2–1. Oyarzabal scored late.";
    let quoted = "Oyarzabal scored late.";
    let at = ANSWER.find(quoted).expect("literal quote");
    let sdk: vertexai::model::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": ANSWER, "thoughtSignature": "AQID"}]},
            "finishReason": 1,
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": "https://example.com/a", "title": "example.com"}}],
                "groundingSupports": [{
                    "segment": {"startIndex": at, "endIndex": at + quoted.len(), "text": quoted},
                    "groundingChunkIndices": [0], "confidenceScores": [0.5]
                }]
            },
            "citationMetadata": {"citations": [
                {"startIndex": 1, "endIndex": 4, "uri": "https://example.com/c", "title": "C"}
            ]}
        }]
    })).expect("SDK literal");
    let expected_raw = serde_json::to_value(&sdk).expect("SDK raw");
    let response = sdk.complete().expect("actual private SDK decoder");
    assert_eq!(response.raw, expected_raw);
    assert_eq!(response.raw["candidates"][0]["finishReason"], 1);
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        panic!("one signed answer: {:?}", response.choice);
    };
    assert_eq!(text.text(), ANSWER);
    assert_eq!(text.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(text.payload.item["thoughtSignature"], "AQID");
    let citations = text.citations();
    assert_eq!(citations.len(), 2);
    assert_eq!(citations[0].span.as_ref().map(|span| (span.start(), span.end())),
        Some((at, at + quoted.len())));
    assert_eq!(citations[0].sources[0].location,
        SourceLocation::Url { url: "https://example.com/a".to_owned() });
    assert_eq!(citations[0].sources[0].title.as_deref(), Some("example.com"));
    assert_eq!(citations[0].sources[0].confidence, Some(0.5));
    assert!(citations[1].span.is_none());
    assert_eq!(citations[1].sources[0].title.as_deref(), Some("C"));
    let replayed = replay(response.choice.clone()).expect("real private request encoder");
    assert_eq!(replayed.parts, [vertexai::model::Part::new()
        .set_text(ANSWER).set_thought_signature(vec![1, 2, 3])]);
}

#[test]
fn sdk_capture_roundtrip_keeps_numeric_enum_and_single_part_boundaries() {
    let sdk: vertexai::model::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "A"}, {"text": "B"}]},
            "finishReason": 1, "safetyRatings": [{"category": 3, "probability": 1}]}],
        "responseId": "response", "modelVersion": "model"
    })).expect("SDK literal");
    let expected = serde_json::to_value(&sdk).expect("SDK raw");
    let mut document = VertexDocument::default();
    document.absorb(&sdk);
    assert_eq!(document.finish(), expected);
    let restored: vertexai::model::GenerateContentResponse = serde_json::from_value(expected.clone())
        .expect("SDK raw reader");
    assert_eq!(serde_json::to_value(&restored).expect("SDK raw writer"), expected);
    let original = sdk.complete().expect("first decoder");
    let restored = restored.complete().expect("restored decoder");
    assert_eq!(original.raw, expected);
    assert_eq!(original.choice, restored.choice);
    assert_eq!(original.choice.len(), 2);
    assert_eq!(original.usage, restored.usage);
    assert_eq!(original.identity(), restored.identity());
}

#[test]
fn sdk_unspecified_finish_stays_absent_and_missing_content_diagnostics_stay_exact() {
    assert_eq!(map_finish_reason(&vertexai::model::candidate::FinishReason::Unspecified), None);
    for (sdk, expected) in [
        (vertexai::model::GenerateContentResponse::new(), "No candidates in response"),
        (vertexai::model::GenerateContentResponse::new().set_candidates([
            vertexai::model::Candidate::new()
        ]), "No content in candidate")
    ] {
        assert!(matches!(sdk.complete(), Err(ProviderError::Provider(message)) if message == expected));
    }
}

fn sdk_parts(parts: Vec<Value>) -> vertexai::model::GenerateContentResponse {
    serde_json::from_value(json!({
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": 1}]
    })).expect("SDK literal")
}

#[test]
fn sdk_thought_images_skip_but_signed_images_refuse_before_the_skip() {
    let image = json!({"inlineData": {"mimeType": "image/png", "data": "AQID"}, "thought": true});
    let response = sdk_parts(vec![json!({"text": "before"}), image.clone(), json!({"text": "after"})])
        .complete().expect("visible text around a thought image");
    assert_eq!(response.choice.len(), 2);
    assert!(matches!(&response.choice[0], AssistantContent::Text(text) if text.text == "before"));
    assert!(matches!(&response.choice[1], AssistantContent::Text(text) if text.text == "after"));
    assert!(matches!(sdk_parts(vec![image.clone()]).complete(),
        Err(ProviderError::Response(message)) if message == rig_core::message::EMPTY_RESPONSE_ERROR));
    let mut signed = image;
    signed["thoughtSignature"] = json!("AQID");
    assert!(matches!(sdk_parts(vec![signed]).complete(),
        Err(ProviderError::Response(message)) if message ==
            "Vertex inline images with thought_signature cannot be replayed through assistant history"));
}

#[test]
fn sdk_unsupported_inline_media_keeps_the_protected_typed_diagnostics() {
    for mime in ["audio/wav", "application/pdf", "application/octet-stream"] {
        let response = sdk_parts(vec![json!({"inlineData": {"mimeType": mime, "data": "AA=="}})]);
        assert!(matches!(response.complete(), Err(ProviderError::Response(message))
            if message == format!("Unsupported Vertex inline media type {mime:?}")));
    }
    for mime in ["image/gif", "image/svg+xml"] {
        let response = sdk_parts(vec![json!({"inlineData": {"mimeType": mime, "data": "AA=="}})]);
        assert!(matches!(response.complete(), Err(ProviderError::Response(message))
            if message.contains("Unsupported Vertex inline image media type")
                && message.contains("cannot be replayed through assistant history")));
    }
}

#[test]
fn sdk_usage_keeps_reported_zero_and_separate_thought_counts() {
    let absent = sdk_parts(vec![json!({"text": "hi"})]).complete().expect("visible text");
    assert_eq!(absent.usage, Usage::default());
    let zero = sdk_parts(vec![json!({"text": "hi"})])
        .set_usage_metadata(vertexai::model::generate_content_response::UsageMetadata::new())
        .complete().expect("reported zero counters");
    assert_eq!(zero.usage.input_tokens, Some(0));
    assert_eq!(zero.usage.output_tokens, Some(0));
    assert_eq!(zero.usage.total_tokens, Some(0));
    assert_eq!(zero.usage.cached_input_tokens, Some(0));
    assert_eq!(zero.usage.reasoning_tokens, Some(0));
    assert_eq!(zero.usage.cache_creation_input_tokens, None);
    assert_eq!(zero.usage.tool_use_prompt_tokens, None);
    let counted = sdk_parts(vec![json!({"text": "hi"})])
        .set_usage_metadata(vertexai::model::generate_content_response::UsageMetadata::new()
            .set_prompt_token_count(14).set_candidates_token_count(34)
            .set_thoughts_token_count(222).set_total_token_count(270)
            .set_cached_content_token_count(9))
        .complete().expect("reported counters");
    assert_eq!(counted.usage.reasoning_tokens, Some(222));
    assert_eq!(counted.usage.cached_input_tokens, Some(9));
    assert_eq!(counted.usage.input_tokens, Some(14));
    assert_eq!(counted.usage.output_tokens, Some(34));
    assert_eq!(counted.usage.total_tokens, Some(270));
    assert_eq!(counted.usage.cache_creation_input_tokens, None);
    assert_eq!(counted.usage.tool_use_prompt_tokens, None);
}

#[test]
fn sdk_missing_call_ids_count_functions_only_and_are_repeatable() {
    let mut parts = Vec::new();
    for n in 0..3 {
        parts.push(json!({"text": format!("before {n}")}));
        parts.push(json!({"functionCall": {"name": "same", "args": {"n": n}}}));
    }
    let sdk = sdk_parts(parts);
    let first = sdk.clone().complete().expect("three calls");
    assert_eq!(first.choice, sdk.complete().expect("identical conversion").choice);
    let calls: Vec<_> = first.choice.iter().filter_map(|item| match item {
        AssistantContent::ToolCall(call) => Some(call),
        _ => None,
    }).collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls.iter().map(|call| &call.id).collect::<std::collections::HashSet<_>>().len(), 3);
    for (ordinal, call) in calls.into_iter().enumerate() {
        assert_eq!(call.id, CallId::Local(LocalCallId::for_vertex_response_position(ordinal as u64)));
        assert_eq!(call.function.arguments, json!({"n": ordinal}));
        assert_eq!(call.item_id, None);
    }
}

#[test]
fn sdk_only_the_first_candidate_is_delivered_but_all_raw_candidates_survive() {
    let sdk: vertexai::model::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "first"}]}, "finishReason": 1},
            {"content": {"role": "model", "parts": [{"text": "second"}]}, "finishReason": 2}
        ]
    })).expect("SDK literal");
    let raw = serde_json::to_value(&sdk).expect("SDK raw");
    let response = sdk.complete().expect("first candidate");
    assert_eq!(response.choice.len(), 1);
    assert!(matches!(&response.choice[0], AssistantContent::Text(text) if text.text == "first"));
    assert_eq!(response.raw, raw);
    assert_eq!(response.raw["candidates"].as_array().map(Vec::len), Some(2));
}
