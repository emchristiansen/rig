//! Frame fidelity and decoded-text regressions through the real modality fold.

use super::*;
use crate::error::{FrameEvidence, ProviderError};
use crate::test_utils::decode_reply;
use serde_json::Value;

#[test]
fn invalid_utf8_embedding_json_keeps_the_received_bytes() {
    // Lossy conversion would make this valid JSON with a replacement model id.
    let body = b"{\"data\":[{\"embedding\":[0.5]}],\"model\":\"\xff\",\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}".to_vec();
    let wire = OpenAIConfig::new("test-key").embedding("requested", None);
    let error = decode_reply(
        &wire,
        &vec!["input".to_owned()],
        Mode::Unary,
        [WireFrame::Bytes(body.clone())],
        Value::Null,
    )
    .expect_err("invalid received UTF-8 must not become a successful replacement model");
    let ProviderError::CorruptFrame(corrupt) = error else {
        panic!("the actual classifier failure must remain a corrupt frame");
    };
    assert_eq!(corrupt.event_type(), None);
    assert_eq!(corrupt.frame(), None);
    assert_eq!(corrupt.frame_bytes(), Some(body.as_slice()));
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(body));
}

#[test]
fn corrupt_decoded_text_keeps_the_driver_provenance() {
    let body = "{\"model\":\"\u{fffd}\",\"data\":";
    let wire = OpenAIConfig::new("test-key").embedding("requested", None);
    let error = decode_reply(
        &wire,
        &vec!["input".to_owned()],
        Mode::Unary,
        [WireFrame::DecodedText(body.to_owned())],
        Value::Null,
    )
    .expect_err("an unfinished decoded document remains corrupt");
    let ProviderError::CorruptFrame(corrupt) = error else {
        panic!("the actual classifier failure must remain a corrupt frame");
    };
    assert_eq!(corrupt.evidence(), &FrameEvidence::Decoded(body.to_owned()));
    assert_eq!(corrupt.frame(), None);
    assert_eq!(corrupt.frame_bytes(), None);
}

#[test]
fn decoded_embedding_json_preserves_the_reported_model_and_vector() {
    let wire = OpenAIConfig::new("test-key").embedding("requested", None);
    let response = decode_reply(
        &wire,
        &vec!["input".to_owned()],
        Mode::Unary,
        [WireFrame::DecodedText(
            "{\"data\":[{\"embedding\":[0.5]}],\"model\":\"m\u{fffd}\",\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}".to_owned(),
        )],
        Value::Null,
    )
    .expect("already-decoded valid JSON still classifies and folds");
    assert_eq!(response.model.as_deref(), Some("m\u{fffd}"));
    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(response.embeddings[0].document, "input");
    assert_eq!(response.embeddings[0].vec, vec![0.5]);
    assert_eq!(response.usage.input_tokens, Some(1));
}

#[test]
fn decoded_transcription_json_preserves_text() {
    let wire = OpenAIConfig::new("test-key").transcription("whisper-1");
    let request = crate::transcription::TranscriptionRequest {
        data: b"RIFF".to_vec(),
        filename: "clip.wav".to_owned(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    };
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::DecodedText(
            "{\"text\":\"heard\u{fffd}\"}".to_owned(),
        )],
        Value::Null,
    )
    .expect("already-decoded transcription JSON still classifies and folds");
    assert_eq!(response.text, "heard\u{fffd}");
    assert_eq!(response.provider, "openai");
}

#[test]
fn decoded_model_catalogue_preserves_the_reported_id_and_limits() {
    let wire = OpenAIConfig::new("test-key").models();
    let response = decode_reply(
        &wire,
        &None,
        Mode::Unary,
        [WireFrame::DecodedText(
            "{\"data\":[{\"id\":\"m\u{fffd}\",\"context_window\":128,\"max_completion_tokens\":32}]}".to_owned(),
        )],
        Value::Null,
    )
    .expect("already-decoded catalogue JSON still classifies and folds");
    let model = response
        .models
        .iter()
        .next()
        .expect("the announced model remains");
    assert_eq!(model.id, "m\u{fffd}");
    assert_eq!(model.context_length, Some(128));
    assert_eq!(model.max_output_tokens, Some(32));
    assert_eq!(response.next, None);
}

#[test]
fn decoded_rerank_json_preserves_score_alias_and_document() {
    let wire = OpenAIConfig::with_key(&super::super::LLAMACPP, "").rerank("requested");
    let request = crate::operation::RerankRequest {
        query: "query".to_owned(),
        documents: vec!["input".to_owned()],
    };
    let response = decode_reply(
        &wire,
        &request,
        Mode::Unary,
        [WireFrame::DecodedText(
            "{\"model\":\"m\u{fffd}\",\"results\":[{\"index\":0,\"score\":0.75,\"text\":\"echo\u{fffd}\"}]}".to_owned(),
        )],
        Value::Null,
    )
    .expect("already-decoded rerank JSON still classifies and folds");
    assert_eq!(response.model.as_deref(), Some("m\u{fffd}"));
    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].index, 0);
    assert_eq!(response.results[0].relevance_score, 0.75);
    assert_eq!(
        response.results[0].document.as_deref(),
        Some("echo\u{fffd}")
    );
}

#[cfg(feature = "image")]
#[test]
fn decoded_image_json_and_huggingface_text_keep_their_distinct_payloads() {
    let request = crate::image_generation::ImageGenerationRequest {
        prompt: "image".to_owned(),
        width: 1024,
        height: 1024,
        additional_params: None,
    };
    let json_wire = OpenAIConfig::new("test-key").image_generation("gpt-image-1");
    let image = decode_reply(
        &json_wire,
        &request,
        Mode::Unary,
        [WireFrame::DecodedText(
            "{\"data\":[{\"b64_json\":\"aGk=\"}]}".to_owned(),
        )],
        Value::Null,
    )
    .expect("already-decoded image JSON still classifies and folds");
    assert_eq!(image.image, b"hi");

    let raw_wire = OpenAIConfig::with_key(&super::super::HUGGINGFACE, "test-key")
        .image_generation("image-model");
    let text = "pixels\u{fffd}";
    for frame in [
        WireFrame::Text(text.to_owned()),
        WireFrame::DecodedText(text.to_owned()),
        WireFrame::Bytes(text.as_bytes().to_vec()),
    ] {
        let image = decode_reply(&raw_wire, &request, Mode::Unary, [frame], Value::Null)
            .expect("Hugging Face keeps the frame payload as the image");
        assert_eq!(image.image, text.as_bytes());
    }
}

#[cfg(feature = "audio")]
#[test]
fn decoded_speech_envelope_and_raw_audio_keep_their_distinct_payloads() {
    let request = crate::audio_generation::AudioGenerationRequest {
        text: "hello".to_owned(),
        voice: "EN-US".to_owned(),
        speed: 1.0,
        additional_params: None,
    };
    let json_wire =
        OpenAIConfig::with_key(&super::super::HYPERBOLIC, "test-key").audio_generation("EN");
    let audio = decode_reply(
        &json_wire,
        &request,
        Mode::Unary,
        [WireFrame::DecodedText("{\"audio\":\"aGk=\"}".to_owned())],
        Value::Null,
    )
    .expect("already-decoded speech JSON still classifies and folds");
    assert_eq!(audio.audio, b"hi");

    let raw_wire = OpenAIConfig::new("test-key").audio_generation("tts-1");
    let text = "audio\u{fffd}";
    for frame in [
        WireFrame::Text(text.to_owned()),
        WireFrame::DecodedText(text.to_owned()),
        WireFrame::Bytes(text.as_bytes().to_vec()),
    ] {
        let audio = decode_reply(&raw_wire, &request, Mode::Unary, [frame], Value::Null)
            .expect("the raw speech branch keeps the frame payload");
        assert_eq!(audio.audio, text.as_bytes());
    }
}
