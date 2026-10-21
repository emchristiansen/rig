//! Decodes unary Vertex SDK replies and retains SDK JSON as diagnostic raw.
//! The raw carrier uses SDK camelCase keys and numeric enums without a format
//! tag. REST conversion is used only by the extras reader.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::Usage;
use rig_core::error::ProviderError;
use rig_core::message::{CallId, CarriedPayload, CarriedPosition, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, DocumentSourceKind,
    GeminiSignedAnswer, GeminiThoughtSignature, Image, ImageDetail, ImageMediaType,
    LocalCallId, MediaType, MimeType, ToolName};
use rig_core::operation::{Block, Completion, Finish};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};
use serde_json::{Map, Value};

mod grounding;

/// Stable descriptor name reported on normalized Vertex AI responses.
pub const PROVIDER_NAME: &str = "vertexai";

/// SDK JSON retained exactly for one reply. Multiple delivered frames use
/// the existing GenerateContent field fold without respelling SDK enum values.
#[derive(Debug, Default)]
pub struct VertexDocument(SdkFrames);

#[derive(Debug, Default)]
enum SdkFrames {
    #[default]
    Empty,
    One(Value),
    Many(rig_core::providers::gemini::streaming::document::GenerateContentResponse),
}

impl rig_core::wire::document::Serves<Completion> for VertexDocument {}
impl rig_core::wire::document::Reassemble<vertexai::model::GenerateContentResponse> for VertexDocument {
    fn absorb(&mut self, frame: &vertexai::model::GenerateContentResponse) {
        // The decoder returns any serialization failure before emitting content.
        let Ok(value) = serde_json::to_value(frame) else { return; };
        match std::mem::take(&mut self.0) {
            SdkFrames::Empty => self.0 = SdkFrames::One(value),
            SdkFrames::One(first) => {
                let mut folded = rig_core::providers::gemini::streaming::document::GenerateContentResponse::default();
                if let Value::Object(first) = first { folded.chunk(first); }
                if let Value::Object(value) = value { folded.chunk(value); }
                self.0 = SdkFrames::Many(folded);
            }
            SdkFrames::Many(mut folded) => {
                if let Value::Object(value) = value { folded.chunk(value); }
                self.0 = SdkFrames::Many(folded);
            }
        }
    }
    fn finish(self) -> Value {
        match self.0 {
            SdkFrames::Empty => Value::Null,
            SdkFrames::One(value) => value,
            SdkFrames::Many(folded) => folded.document(),
        }
    }
}

/// Reads the first candidate using SDK part semantics. Streaming is refused
/// by the wire before preparation and never reaches this unary decoder.
#[derive(Debug, Default)]
pub struct VertexDecoder;

impl<'id> Decoder<'id, Completion, vertexai::model::GenerateContentResponse> for VertexDecoder {
    type Event = vertexai::model::GenerateContentResponse;
    fn classify(&self, frame: Self::Event) -> WireEvent<Self::Event> {
        WireEvent::Known(frame)
    }
    fn decode(&mut self, response: Self::Event, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        serde_json::to_value(&response)?;
        let candidate = response.candidates.first()
            .ok_or_else(|| ProviderError::Provider("No candidates in response".to_owned()))?;
        let content = candidate.content.as_ref()
            .ok_or_else(|| ProviderError::Provider("No content in candidate".to_owned()))?;
        let mut answer = grounding::AnswerText::default();
        let mut placed = Vec::with_capacity(content.parts.len());
        let mut pending_text = Vec::new();
        let mut calls = 0u64;
        let mut delivered = false;
        for (position, part) in content.parts.iter().enumerate() {
            placed.push(None);
            let signature = (!part.thought_signature.is_empty())
                .then(|| BASE64.encode(&part.thought_signature));
            let raw = serde_json::to_value(part)?;
            let payload = CarriedPayload::new(raw.clone(), Some(PROVIDER_NAME.to_owned()))
                .with_position(CarriedPosition::MessageContent {
                    item_index: 0, part_index: position as u64,
                });
            if let Some(call) = part.function_call() {
                let id = CallId::Local(LocalCallId::for_vertex_response_position(calls));
                calls = calls.checked_add(1).ok_or_else(|| ProviderError::Response(
                    "too many Vertex calls in one completion".to_owned()))?;
                let name = ToolName::new(call.name.clone())
                    .map_err(|error| ProviderError::Response(error.to_string()))?;
                let args = call.args.as_ref().map_or_else(|| Value::Object(Map::new()),
                    |args| Value::Object(args.clone()));
                let index = out.fresh_index();
                out.open(index, Block::Call { id, name, namespace: None }, raw)?;
                out.announce(index, &args)?;
                out.finish(index)?;
                delivered = true;
            } else if let Some(text) = part.text() {
                let index = out.fresh_index();
                if part.thought {
                    out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
                    out.push(index, text)?;
                    out.finish_carried_reasoning(index,
                        CarriedReasoning::new(vec![CarriedReasoningPart::Text {
                            text: text.clone(), signature,
                        }], payload))?;
                } else {
                    out.open(index, Block::Text, raw)?;
                    out.push(index, text)?;
                    answer.open(index, text);
                    placed[position] = Some((index, 0));
                    pending_text.push((index, text.clone(), signature, payload));
                }
                delivered = true;
            } else if let Some(blob) = part.inline_data() {
                if signature.is_some() {
                    return Err(ProviderError::Response(
                        "Vertex inline images with thought_signature cannot be replayed through assistant history".to_owned()));
                }
                if part.thought { continue; }
                let media_type = match MediaType::from_mime_type(&blob.mime_type) {
                    Some(MediaType::Image(media @ (ImageMediaType::JPEG | ImageMediaType::PNG
                        | ImageMediaType::WEBP | ImageMediaType::HEIC | ImageMediaType::HEIF))) => media,
                    Some(MediaType::Image(media)) => return Err(ProviderError::Response(format!(
                        "Unsupported Vertex inline image media type {media:?}; it cannot be replayed through assistant history"))),
                    _ => return Err(ProviderError::Response(format!(
                        "Unsupported Vertex inline media type {:?}", blob.mime_type))),
                };
                let index = out.fresh_index();
                out.open(index, Block::Image(Image {
                    data: DocumentSourceKind::Base64(BASE64.encode(&blob.data)),
                    media_type: Some(media_type), detail: Some(ImageDetail::default()),
                    ..Image::default()
                }), Value::Null)?;
                out.finish(index)?;
                delivered = true;
            } else if signature.is_some() {
                tracing::warn!("Vertex response part carries a thought_signature but is neither a function call nor text; signature dropped (no rig-core carrier).");
            }
        }
        if !delivered {
            return Err(ProviderError::Response(rig_core::message::EMPTY_RESPONSE_ERROR.to_owned()));
        }
        // Citations attach while each answer's original Part is still open.
        if let Some(metadata) = &candidate.grounding_metadata {
            for (index, citation) in grounding::grounding(metadata, &placed, &answer) {
                out.cite(index, citation);
            }
        }
        if let Some(metadata) = &candidate.citation_metadata {
            for (index, citation) in grounding::recitations(metadata, &answer) {
                out.cite(index, citation);
            }
        }
        for (index, text, signature, payload) in pending_text {
            match signature {
                Some(signature) => out.finish_carried_text(index, CarriedText::new(
                    CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(text,
                        GeminiThoughtSignature::new(signature))), payload))?,
                None => out.finish(index)?,
            }
        }
        let usage = response.usage_metadata.as_ref().map(|usage| Usage {
            input_tokens: Some(usage.prompt_token_count as u64),
            output_tokens: Some(usage.candidates_token_count as u64),
            total_tokens: Some(usage.total_token_count as u64),
            cached_input_tokens: Some(usage.cached_content_token_count as u64),
            reasoning_tokens: Some(usage.thoughts_token_count as u64),
            cache_creation_input_tokens: None, tool_use_prompt_tokens: None,
            ..Usage::default()
        }).unwrap_or_default();
        let reason = map_finish_reason(&candidate.finish_reason);
        Ok(out.end(Finish {
            usage, reason,
            response_id: (!response.response_id.is_empty()).then_some(response.response_id),
            model: (!response.model_version.is_empty()).then_some(response.model_version),
            ..Finish::default()
        }))
    }
}

/// Normalize the SDK finish spelling; an unspecified value reports no reason.
pub fn map_finish_reason(reason: &vertexai::model::candidate::FinishReason)
    -> Option<rig_core::completion::FinishReason>
{
    use rig_core::completion::FinishReason;
    let wire = reason.name().map_or_else(|| reason.to_string(), str::to_owned);
    Some(match wire.as_str() {
        "FINISH_REASON_UNSPECIFIED" => return None,
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => FinishReason::ContentFilter,
        _ => FinishReason::Other(wire),
    })
}

/// Convert SDK raw to the schema the extras reader inspects. Enums are
/// respelled by name; the tool-use count is the modality breakdown's sum.
/// This conversion does not supply capture, history or outbound data.
pub(crate) fn rest_chunk(
    response: &vertexai::model::GenerateContentResponse,
) -> Result<Map<String, Value>, ProviderError> {
    let Value::Object(mut chunk) = serde_json::to_value(response)? else {
        return Ok(Map::new());
    };
    let candidates = chunk.get_mut("candidates").and_then(Value::as_array_mut);
    for (candidate, json) in response
        .candidates
        .iter()
        .zip(candidates.into_iter().flatten())
    {
        spell(json, "finishReason", candidate.finish_reason.name());
        ratings(json, &candidate.safety_ratings);
        // A part's enums too, so a stored part reads back whatever a store
        // does to its numbers.
        let parts = candidate.content.iter().flat_map(|content| &content.parts);
        let json_parts = json
            .pointer_mut("/content/parts")
            .and_then(Value::as_array_mut);
        for (part, json) in parts.zip(json_parts.into_iter().flatten()) {
            if let (Some(code), Some(json)) =
                (part.executable_code(), json.get_mut("executableCode"))
            {
                spell(json, "language", code.language.name());
            }
            if let (Some(result), Some(json)) = (
                part.code_execution_result(),
                json.get_mut("codeExecutionResult"),
            ) {
                spell(json, "outcome", result.outcome.name());
            }
        }
    }
    if let (Some(feedback), Some(json)) =
        (&response.prompt_feedback, chunk.get_mut("promptFeedback"))
    {
        spell(json, "blockReason", feedback.block_reason.name());
        ratings(json, &feedback.safety_ratings);
    }
    if let (Some(usage), Some(json)) = (&response.usage_metadata, chunk.get_mut("usageMetadata")) {
        spell(json, "trafficType", usage.traffic_type.name());
        for (key, details) in [
            ("promptTokensDetails", &usage.prompt_tokens_details),
            ("cacheTokensDetails", &usage.cache_tokens_details),
            ("candidatesTokensDetails", &usage.candidates_tokens_details),
            (
                "toolUsePromptTokensDetails",
                &usage.tool_use_prompt_tokens_details,
            ),
        ] {
            let json = json.get_mut(key).and_then(Value::as_array_mut);
            for (detail, json) in details.iter().zip(json.into_iter().flatten()) {
                spell(json, "modality", detail.modality.name());
            }
        }
        let tool_use: i64 = usage
            .tool_use_prompt_tokens_details
            .iter()
            .map(|detail| i64::from(detail.token_count))
            .sum();
        if let (Some(json), true) = (json.as_object_mut(), tool_use > 0) {
            json.entry("toolUsePromptTokenCount")
                .or_insert_with(|| tool_use.into());
        }
    }
    Ok(chunk)
}

/// Respell the safety ratings under `json` by name.
fn ratings(json: &mut Value, ratings: &[vertexai::model::SafetyRating]) {
    let json = json.get_mut("safetyRatings").and_then(Value::as_array_mut);
    for (rating, json) in ratings.iter().zip(json.into_iter().flatten()) {
        spell(json, "category", rating.category.name());
        spell(json, "probability", rating.probability.name());
        spell(json, "severity", rating.severity.name());
    }
}

/// Replace the number at `json[key]` with `name`, when both exist.
fn spell(json: &mut Value, key: &str, name: Option<&str>) {
    if let (Some(slot), Some(name)) = (json.get_mut(key), name) {
        *slot = Value::String(name.to_owned());
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod sdk_adapter_tests;

#[cfg(test)]
mod vertex_usage_mapping_tests;
