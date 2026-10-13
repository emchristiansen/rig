use super::*;
use crate::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

fn transcription_request() -> transcription::TranscriptionRequest {
    transcription::TranscriptionRequest {
        data: b"audio bytes".to_vec(),
        filename: "audio.mp3".to_string(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    }
}

/// The bytes one transcription puts on the wire, pinned by
/// `crates/rig-cassette/fixtures/cassettes/gemini/transcription/transcription_smoke.yaml`: the path,
/// the `key` query parameter and the whole JSON document, down to the `null`
/// fields Gemini's request shape keeps. The cassette's `inlineData.data` is a
/// 32 KB mp3; this request's audio is [`transcription_request`]'s, so only
/// that one string differs.
#[test]
fn the_wire_encodes_the_recorded_generate_content_request() {
    let wire = Transcriptions::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_3_FLASH_PREVIEW,
    );

    let encoded = wire
        .encode(transcription_request(), Mode::Unary)
        .expect("the request encodes");

    let request = &encoded.request;
    assert_eq!(request.method().as_str(), "POST");
    assert_eq!(
        request.uri().path(),
        "/v1beta/models/gemini-3-flash-preview:generateContent"
    );
    assert_eq!(request.uri().query(), Some("key=test-key"));
    assert_eq!(
        request
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(encoded.request_id_header, None);

    assert_eq!(
        crate::test_utils::json_body(request),
        serde_json::json!({
            "contents": [{
                "parts": [{
                    "inlineData": {
                        "data": "YXVkaW8gYnl0ZXM=",
                        "mimeType": "audio/mpeg"
                    },
                    "thought": false
                }],
                "role": "user"
            }],
            "generationConfig": {},
            "safetySettings": null,
            "systemInstruction": {
                "parts": [{ "text": TRANSCRIPTION_PREAMBLE, "thought": false }],
                "role": "model"
            },
            "toolConfig": null
        })
    );
}

#[test]
fn actual_decoder_normalizes_usage_and_refuses_checked_overflow() {
    let wire = Transcriptions::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        "gemini-2.5-flash",
    );
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut raw = json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP", "index": 0}],
            "modelVersion": "gemini-2.5-flash", "responseId": "usage-190-transcription",
            "usageMetadata": {"promptTokenCount": 100, "toolUsePromptTokenCount": 75,
                "cachedContentTokenCount": 40, "candidatesTokenCount": 30,
                "thoughtsTokenCount": 34, "totalTokenCount": 239}
        });
        let response = crate::test_utils::decode_reply(
            &wire, &transcription_request(), mode,
            [WireFrame::Text(raw.to_string())], raw.clone(),
        ).expect("the real decoder completes");
        assert_eq!(response.text, "hi");
        assert_eq!(response.usage.input_tokens, Some(175));
        assert_eq!(response.usage.tool_use_prompt_tokens, Some(75));
        assert_eq!(response.usage.cached_input_tokens, Some(40));
        assert_eq!(response.usage.output_tokens, Some(64));
        assert_eq!(response.usage.reasoning_tokens, Some(34));
        assert_eq!(response.usage.total_tokens, Some(239));
        assert_eq!(response.usage.cache_creation_input_tokens, None);
        assert_eq!(response.usage.cost, None);
        assert_eq!(response.model.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(response.response_id.as_deref(), Some("usage-190-transcription"));
        assert_eq!(response.raw, raw);
        raw["usageMetadata"] = json!({"toolUsePromptTokenCount": 75,
            "cachedContentTokenCount": 40, "thoughtsTokenCount": 34});
        let response = crate::test_utils::decode_reply(
            &wire, &transcription_request(), mode,
            [WireFrame::Text(raw.to_string())], raw.clone(),
        ).expect("missing base counts keep their existing defaults");
        assert_eq!(response.usage.input_tokens, Some(75));
        assert_eq!(response.usage.output_tokens, None);
        assert_eq!(response.usage.total_tokens, Some(0));
        assert_eq!(response.usage.cached_input_tokens, Some(40));
        assert_eq!(response.usage.tool_use_prompt_tokens, Some(75));
        assert_eq!(response.usage.reasoning_tokens, Some(34));
        assert_eq!(response.raw, raw);
        for (usage, message) in [
            (json!({"promptTokenCount": -1, "toolUsePromptTokenCount": 1}),
                "Gemini input token count overflow"),
            (json!({"candidatesTokenCount": -1, "thoughtsTokenCount": 1}),
                "Gemini output token count overflow"),
        ] {
            raw["usageMetadata"] = usage;
            let error = crate::test_utils::decode_reply(
                &wire, &transcription_request(), mode,
                [WireFrame::Text(raw.to_string())], raw.clone(),
            ).expect_err("overflow is a typed response refusal");
            assert!(matches!(error, ProviderError::Response(ref actual) if actual == message));
        }
    }
}
