//! A54 raw-document projection through the actual decoder and driver fold.
//! The harness supplies the transport Reply.raw; it does not run HTTP capture.

use super::*;
use crate::test_utils::decode_reply;
use serde_json::json;

#[test]
fn transcription_raw_keeps_unknown_fields_beside_normalized_text() {
    let wire = OpenAIConfig::new("test-key").transcription("whisper-1");
    let request = crate::transcription::TranscriptionRequest {
        data: b"RIFF".to_vec(),
        filename: "clip.wav".to_owned(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    };
    let raw = json!({"text": "heard", "future_field": 7});
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
        raw.clone(),
    )
    .unwrap();
    assert_eq!(response.text, "heard");
    assert_eq!(response.raw, raw);
}

#[test]
fn rerank_raw_keeps_unknown_fields_beside_normalized_scores() {
    let wire = OpenAIConfig::with_key(&super::super::LLAMACPP, "").rerank("requested");
    let request = crate::operation::RerankRequest {
        query: "query".to_owned(),
        documents: vec!["input".to_owned()],
    };
    let raw = json!({"results": [{"index": 0, "score": 0.75}], "future_field": 7});
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
        raw.clone(),
    )
    .unwrap();
    assert_eq!(response.results[0].relevance_score, 0.75);
    assert_eq!(response.raw, raw);
}

#[cfg(feature = "image")]
#[test]
fn image_raw_keeps_unknown_fields_beside_decoded_image() {
    let wire = OpenAIConfig::new("test-key").image_generation("gpt-image-1");
    let request = crate::image_generation::ImageGenerationRequest {
        prompt: "image".to_owned(),
        width: 1024,
        height: 1024,
        additional_params: None,
    };
    let raw = json!({"data": [{"b64_json": "aGk="}], "future_field": 7});
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
        raw.clone(),
    )
    .unwrap();
    assert_eq!(response.image, b"hi");
    assert_eq!(response.raw, raw);
}

#[cfg(feature = "audio")]
#[test]
fn hyperbolic_speech_raw_keeps_full_document_instead_of_f_null() {
    let wire = OpenAIConfig::with_key(&super::super::HYPERBOLIC, "test-key").audio_generation("EN");
    let request = crate::audio_generation::AudioGenerationRequest {
        text: "hello".to_owned(),
        voice: "EN-US".to_owned(),
        speed: 1.0,
        additional_params: None,
    };
    let raw = json!({"audio": "aGk=", "future_field": 7});
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
        raw.clone(),
    )
    .unwrap();
    assert_eq!(response.audio, b"hi");
    assert_eq!(response.raw, json!({"audio": "aGk=", "future_field": 7}));
}
