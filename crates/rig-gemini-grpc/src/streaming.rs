//! Decodes typed Gemini protobuf replies with distinct unary and stream rules.
//! Raw documents use SDK JSON with snake_case fields, numeric enums, and byte
//! arrays. Code language and outcome use the current numeric SDK enums; older
//! known names remain readable. The raw carrier has no format tag. Request
//! transcoding uses proto3 REST JSON independently.

use base64::{Engine as _, prelude::BASE64_STANDARD};
use rig_core::completion::Usage;
use rig_core::error::ProviderError;
use rig_core::message::{CallId, CarriedPayload, CarriedPosition, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, DocumentSourceKind,
    GeminiSignedAnswer, GeminiThoughtSignature, Image, ImageDetail, LocalCallId,
    MediaType, MimeType, ToolName};
use rig_core::operation::{Block, Completion, Finish};
use rig_core::wire::{Decoder, Flow, IncompleteEof, Mode, Out, WireEvent};
use serde_json::{Map, Value};
use crate::proto;

struct Active {
    index: usize,
    thought: bool,
    text: String,
    signature: Option<String>,
    payload: CarriedPayload,
}

/// A protobuf completion. Only a genuine nonzero streamed finish reason ends
/// a stream; EOF flushes delivered partial content without terminal proof.
pub struct GrpcAdapter {
    mode: Mode,
    active: Option<Active>,
    call_ordinal: u64,
    part_ordinal: u64,
}

impl Default for GrpcAdapter {
    fn default() -> Self { Self::in_mode(Mode::Streaming) }
}

impl GrpcAdapter {
    pub(crate) fn in_mode(mode: Mode) -> Self {
        Self { mode, active: None, call_ordinal: 0, part_ordinal: 0 }
    }

    fn flush(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some(active) = self.active.take() else { return Ok(()); };
        if active.thought {
            out.finish_carried_reasoning(active.index,
                CarriedReasoning::new(vec![CarriedReasoningPart::Text {
                    text: active.text, signature: active.signature,
                }], active.payload))
        } else if let Some(signature) = active.signature {
            out.finish_carried_text(active.index, CarriedText::new(
                CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer {
                    text: active.text, signature: GeminiThoughtSignature::new(signature),
                }), active.payload))
        } else { out.finish(active.index) }
    }

    fn part(&mut self, part: &proto::Part, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let position = self.part_ordinal;
        self.part_ordinal = position.checked_add(1).ok_or_else(|| ProviderError::Response(
            "too many Gemini gRPC parts in one completion".into()))?;
        match &part.data {
            Some(proto::part::Data::Text(text)) => {
                if self.active.as_ref().is_some_and(|active| active.thought != part.thought) {
                    self.flush(out)?;
                }
                if self.active.is_none() {
                    let index = out.fresh_index();
                    let block = if part.thought { Block::Reasoning { redacted: false } } else { Block::Text };
                    out.open(index, block, Value::Null)?;
                    let payload = CarriedPayload::new(raw::part(part), Some("gemini".to_owned()))
                        .with_position(CarriedPosition::MessageContent { item_index: 0, part_index: position });
                    self.active = Some(Active { index, thought: part.thought,
                        text: String::new(), signature: None, payload });
                }
                if let Some(active) = &mut self.active {
                    out.push(active.index, text)?;
                    active.text.push_str(text);
                    if !part.thought_signature.is_empty() {
                        active.signature = Some(BASE64_STANDARD.encode(&part.thought_signature));
                    }
                }
                if !part.thought_signature.is_empty() || self.mode == Mode::Unary { self.flush(out)?; }
            }
            Some(proto::part::Data::FunctionCall(call)) => {
                self.flush(out)?;
                let ordinal = self.call_ordinal;
                self.call_ordinal = ordinal.checked_add(1).ok_or_else(|| ProviderError::Response(
                    "too many Gemini gRPC calls in one completion".into()))?;
                let id = if !call.id.is_empty() {
                    CallId::from_wire(call.id.as_str())
                } else {
                    CallId::Local(match self.mode {
                        Mode::Unary => LocalCallId::for_gemini_grpc_response_position(ordinal),
                        Mode::Streaming => LocalCallId::new(),
                    })
                };
                let name = ToolName::new(call.name.clone()).map_err(|error| ProviderError::Response(error.to_string()))?;
                let args = call.args.as_ref().map_or_else(|| Value::Object(Map::new()), struct_json);
                let index = out.fresh_index();
                let item = native_call(part)?;
                out.open(index, Block::Call { id, name, namespace: None }, item)?;
                out.announce(index, &args)?;
                out.finish(index)?;
            }
            Some(proto::part::Data::InlineData(blob)) if self.mode == Mode::Unary => {
                self.flush(out)?;
                let media = MediaType::from_mime_type(&blob.mime_type);
                let Some(MediaType::Image(media_type)) = media else {
                    return Err(ProviderError::Response(format!("Unsupported media type {media:?}")));
                };
                let index = out.fresh_index();
                out.open(index, Block::Image(Image {
                    data: DocumentSourceKind::Base64(BASE64_STANDARD.encode(&blob.data)),
                    media_type: Some(media_type), detail: Some(ImageDetail::default()),
                    ..Image::default()
                }), Value::Null)?;
                out.finish(index)?;
            }
            _ if self.mode == Mode::Unary => return Err(ProviderError::Response(
                "Response did not contain a message or tool call".into())),
            None => tracing::warn!(part = ?part, "skipping Gemini gRPC part without modeled data"),
            Some(_) => {}
        }
        Ok(())
    }
}

impl<'id> Decoder<'id, Completion, proto::GenerateContentResponse> for GrpcAdapter {
    type Event = proto::GenerateContentResponse;
    fn classify(&self, frame: Self::Event) -> WireEvent<Self::Event> { WireEvent::Known(frame) }
    fn decode(&mut self, response: Self::Event, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        let candidate = response.candidates.first();
        if self.mode == Mode::Unary && candidate.is_none() {
            return Err(ProviderError::Response("No response candidates in response".into()));
        }
        if let Some(candidate) = candidate {
            if let Some(error) = protocol_error(candidate.finish_reason, candidate.finish_message.as_deref()) {
                self.flush(&mut out)?;
                return Err(error);
            }
            if self.mode == Mode::Unary && candidate.content.is_none() {
                return Err(ProviderError::Response(format!(
                    "Gemini candidate missing content (finish_reason={})", candidate.finish_reason)));
            }
            if let Some(content) = &candidate.content {
                let mut previous_text = false;
                for part in &content.parts {
                    let text = !part.thought && matches!(&part.data, Some(proto::part::Data::Text(_)));
                    let signed_empty = text && !part.thought_signature.is_empty()
                        && matches!(&part.data, Some(proto::part::Data::Text(text)) if text.is_empty());
                    if text && (previous_text || signed_empty) { self.flush(&mut out)?; }
                    previous_text = text;
                    self.part(part, &mut out)?;
                }
            }
        }
        let reason = candidate.and_then(|candidate| finish_reason(candidate.finish_reason));
        let terminal = self.mode == Mode::Unary || candidate.is_some_and(|candidate| candidate.finish_reason != 0);
        if terminal {
            self.flush(&mut out)?;
            return Ok(out.end(Finish {
                usage: usage(response.usage_metadata.as_ref()), reason,
                response_id: (!response.response_id.is_empty()).then_some(response.response_id),
                model: (!response.model_version.is_empty()).then_some(response.model_version),
                ..Finish::default()
            }));
        }
        Ok(Flow::More)
    }
    fn flush_before_terminal_error(&mut self, mut out: Out<'id, Completion>) -> Result<(), ProviderError> { self.flush(&mut out) }
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.flush(&mut out)?;
        Ok(Flow::More)
    }
    fn incomplete_eof(&self) -> IncompleteEof {
        match self.mode { Mode::Unary => IncompleteEof::Refuse, Mode::Streaming => IncompleteEof::KeepPartial }
    }
}

fn native_call(part: &proto::Part) -> Result<Value, ProviderError> {
    let content = proto::Content { parts: vec![part.clone()], role: "model".to_owned() };
    let document = crate::rest::to_rest(&content)?;
    document.get("parts").and_then(Value::as_array).and_then(|parts| parts.first()).cloned()
        .ok_or_else(|| ProviderError::Response("Gemini gRPC call did not transcode to a part".into()))
}

fn struct_json(value: &prost_types::Struct) -> Value {
    Value::Object(value.fields.iter().map(|(name, value)| (name.clone(), value_json(value))).collect())
}

fn value_json(value: &prost_types::Value) -> Value {
    match &value.kind {
        None | Some(prost_types::value::Kind::NullValue(_)) => Value::Null,
        Some(prost_types::value::Kind::BoolValue(value)) => Value::Bool(*value),
        Some(prost_types::value::Kind::NumberValue(value)) => serde_json::Number::from_f64(*value).map_or(Value::Null, Value::Number),
        Some(prost_types::value::Kind::StringValue(value)) => Value::String(value.clone()),
        Some(prost_types::value::Kind::StructValue(value)) => struct_json(value),
        Some(prost_types::value::Kind::ListValue(value)) => Value::Array(value.values.iter().map(value_json).collect()),
    }
}

fn usage(usage: Option<&proto::UsageMetadata>) -> Usage {
    usage.map(|usage| Usage {
        input_tokens: Some(usage.prompt_token_count as u64),
        output_tokens: Some(usage.candidates_token_count as u64),
        total_tokens: Some(usage.total_token_count as u64),
        cached_input_tokens: Some(usage.cached_content_token_count as u64),
        ..Usage::default()
    }).unwrap_or_default()
}

fn finish_reason(reason: i32) -> Option<rig_core::completion::FinishReason> {
    if reason == 0 { return None; }
    Some(match proto::candidate::FinishReason::try_from(reason) {
        Ok(reason) => rig_core::providers::gemini::completion::map_google_finish_reason(reason.as_str_name()),
        Err(_) => rig_core::completion::FinishReason::Other(format!("FINISH_REASON_{reason}")),
    })
}

fn protocol_error(reason: i32, message: Option<&str>) -> Option<ProviderError> {
    use proto::candidate::FinishReason;
    let reason = FinishReason::try_from(reason).ok()?;
    match reason {
        FinishReason::MalformedFunctionCall | FinishReason::UnexpectedToolCall | FinishReason::TooManyToolCalls => {
            Some(ProviderError::Response(format!("Gemini stopped with finish_reason={}: {}",
                reason.as_str_name(), message.unwrap_or("no finish message provided"))))
        }
        _ => None,
    }
}

pub(crate) mod document;
pub(crate) mod raw;
#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod decoder_tests;
