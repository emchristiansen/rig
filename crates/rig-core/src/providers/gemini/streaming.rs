//! The decoder of GenerateContent replies, a whole `generateContent` body or
//! a stream of `streamGenerateContent` chunks, read as REST JSON. Signed
//! answers and reasoning retain
//! their signatures in carried parts. Inline media remains inspection data
//! by default; the Vertex companion selects closed ordinary inline images.
//! Hosted code stays in the raw diagnostic document.
//!
//! ```
//! use rig_core::providers::gemini::streaming::GenerateContentDecoder;
//!
//! let decoder = GenerateContentDecoder::default();
//! # let _ = decoder;
//! ```

use serde_json::{Map, Value};

use super::completion::blocked_prompt_error;
use super::completion::{checked_usage_of, map_google_finish_reason};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, DocumentSourceKind, Image, ImageMediaType, MimeType, ToolName};
use crate::message::{
    CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart, CarriedText,
    CarriedTextContent, GeminiSignedAnswer, GeminiThoughtSignature,
};
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterVerdict, Decoder, Flow, IncompleteEof, Mode, ObservationSink, Out, WireCitation,
    WireEvent, WireFrame,
};

/// The recognizability markers of a `streamGenerateContent` chunk: every
/// genuine frame carries `candidates`, `usageMetadata` and/or
/// `promptFeedback` (a blocked prompt's only chunk may carry nothing but
/// the feedback), and the service's in-band abort carries only `error`. A
/// frame with any of them is a chunk. A valid ID-only frame is recognized
/// separately as metadata; other JSON is `Unknown`.
const RECOGNIZABLE_CHUNK_KEYS: &[&str] =
    &["candidates", "usageMetadata", "promptFeedback", "error"];

/// A GenerateContent reply document in REST JSON: the whole
/// `generateContent` body or one `streamGenerateContent` chunk.
#[derive(Debug, Default)]
pub struct GenerateContentChunk(pub Map<String, Value>);

pub(super) mod reader;

impl<'de> serde::Deserialize<'de> for GenerateContentChunk {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = <Map<String, Value> as serde::Deserialize>::deserialize(deserializer)?;
        reader::read(&fields).map_err(serde::de::Error::custom)?;
        Ok(Self(fields))
    }
}

/// Decode GenerateContent replies. A text or thought part continues the
/// block of the part before it while the kind stays the same. A thought
/// signature closes its reasoning part. Signed answer parts stay separate. The provider's end is held until EOF because
/// hosted-tool rounds can report intermediate finish reasons. The
/// candidate's grounding and recitation metadata cite the text blocks when
/// the reply ends.
#[derive(Debug)]
pub struct GenerateContentDecoder {
    carried_issuer: &'static str,
    vertex_inline_images: bool,
    defer_vertex_inline_images: bool,
    pending_vertex_image: Option<PendingVertexImage>,
    streaming: bool,
    /// At least one assistant content part was received.
    delivered: bool,
    /// The latest `finishReason`.
    finish: Option<String>,
    /// The latest `usageMetadata`, as Gemini sent it.
    usage: Option<Value>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// The text or thought run later parts continue, and whether it is
    /// thought.
    open: Option<(usize, bool)>,
    /// Whether the open run holds a signature.
    signed: bool,
    /// The index of the block written last.
    last: Option<usize>,
    /// A signature sent alone before any block, which the next block takes.
    signature: Option<String>,
    /// The answer text written so far, which grounding segments cite.
    answer: grounding::AnswerText,
    /// The text block and byte offset in it of each part of the latest
    /// chunk; `None` for a part that is not answer text.
    placed: Vec<Option<(usize, usize)>>,
    /// Where the part being written put its text.
    placement: Option<(usize, usize)>,
    /// The latest `groundingMetadata`'s citations, by block.
    grounding: Vec<(usize, WireCitation)>,
    /// How many chunks carried parts: a reply over more than one counts a
    /// grounding segment's bytes in the whole answer, not in a part.
    chunks: usize,
    /// The `citationMetadata` sources of every chunk, by block.
    recitations: Vec<(usize, WireCitation)>,
    part_index: u64,
    open_position: Option<CarriedPosition>,
}

impl Default for GenerateContentDecoder {
    fn default() -> Self {
        Self {
            carried_issuer: super::completion::PROVIDER_NAME,
            vertex_inline_images: false,
            defer_vertex_inline_images: false,
            pending_vertex_image: None,
            streaming: false,
            delivered: false,
            finish: None,
            usage: None,
            model_version: None,
            response_id: None,
            open: None,
            signed: false,
            last: None,
            signature: None,
            answer: grounding::AnswerText::default(),
            placed: Vec::new(),
            placement: None,
            grounding: Vec::new(),
            chunks: 0,
            recitations: Vec::new(),
            part_index: 0,
            open_position: None,
        }
    }
}

impl<'id> Decoder<'id, Completion> for GenerateContentDecoder {
    type Event = GenerateContentChunk;

    fn classify(&self, frame: WireFrame) -> WireEvent<GenerateContentChunk> {
        // ID-only metadata must not create an unknown-content truncation tail.
        if GenerateContentDecoder::is_analysis_only(&frame) {
            return wire::classify_marker_keyed_frame(&frame.as_str(), &["responseId"]);
        }
        wire::classify_marker_keyed_frame(&frame.as_str(), RECOGNIZABLE_CHUNK_KEYS)
    }

    fn decode(
        &mut self,
        GenerateContentChunk(data): GenerateContentChunk,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let decoded = self.decode_chunk(data, &mut out);
        if decoded.is_err() {
            self.flush_pending_vertex_image(&mut out)?;
        }
        decoded
    }

    /// Gemini ends a reply at EOF, not at its first finish reason: a
    /// hosted-tool round can report one before more content arrives.
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        let cut_short = self.normalized_finish().is_some_and(|reason| {
            matches!(
                reason,
                crate::completion::FinishReason::Length
                    | crate::completion::FinishReason::ContentFilter
            )
        });
        if !self.streaming && !self.delivered && !cut_short {
            self.close(&mut out)?;
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        if self.finish.is_some() {
            return self.end(out);
        }
        self.close(&mut out)?;
        if self.streaming {
            Ok(Flow::More)
        } else {
            Err(ProviderError::Truncated)
        }
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        if self.streaming {
            IncompleteEof::KeepPartial
        } else {
            IncompleteEof::Refuse
        }
    }

    fn flush_before_terminal_error(
        &mut self,
        mut out: Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        self.close(&mut out)
    }
}

#[derive(Debug)]
struct PendingVertexImage {
    index: usize,
    image: Image,
    fields: Map<String, Value>,
    position: CarriedPosition,
}

impl GenerateContentDecoder {
    fn decode_chunk(
        &mut self,
        data: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let data = Value::Object(data);
        // Gemini repeats both on every chunk. Record a value when it changes:
        // every span record counts against the exporter's attribute limit.
        let span = tracing::Span::current();
        if let Some(id) = data.str("responseId").filter(|id| !id.is_empty())
            && self.response_id.as_deref() != Some(id)
        {
            span.record("gen_ai.response.id", id);
            self.response_id = Some(id.to_owned());
        }
        if let Some(model) = data.str("modelVersion").filter(|model| !model.is_empty())
            && self.model_version.as_deref() != Some(model)
        {
            span.record("gen_ai.response.model", model);
            self.model_version = Some(model.to_owned());
        }
        if let Some(usage) = data.get("usageMetadata") {
            self.usage = Some(usage.clone());
        }
        if let Some(error) = data.at("/error") {
            // An in-band failure is a provider error, not a truncation.
            let body = serde_json::json!({ "error": error }).to_string();
            let status = error
                .get("code")
                .and_then(Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .and_then(|code| http::StatusCode::from_u16(code).ok())
                .filter(|status| status.is_client_error() || status.is_server_error());
            return Err(match status {
                Some(status) => ProviderError::from_http_response(status, body),
                None => ProviderError::from_provider_body(body),
            });
        }
        if let Some(blocked) = data.get("promptFeedback").and_then(blocked_prompt_error) {
            return Err(blocked);
        }
        // The candidates, their content and its parts hold every block and
        // the finish, so a wrongly typed one fails the reply.
        let candidate = match data
            .get("candidates")
            .map(|candidates| (candidates, candidates.get(0)))
        {
            None | Some((Value::Null, _) | (Value::Array(_), None)) => return Ok(Flow::More),
            Some((Value::Array(_), Some(candidate @ Value::Object(_)))) => candidate,
            Some((Value::Array(_), Some(_))) => {
                return Err(malformed("a candidate that is not an object"));
            }
            Some(_) => return Err(malformed("candidates that are not a list")),
        };
        // Last one wins: an intermediate `finishReason` is superseded by the
        // reason the turn actually ended on. Proto3 JSON spells an enum
        // value its schema does not know as its number.
        if let Some(reason) = candidate
            .get("finishReason")
            .and_then(reader::finish_reason)
        {
            self.finish = Some(reason);
        }
        if let Some(error) = tool_protocol_finish_reason_error(candidate) {
            return Err(error);
        }
        let parts = match candidate.get("content") {
            None | Some(Value::Null) => None,
            Some(content @ Value::Object(_)) => content.get("parts"),
            Some(_) => return Err(malformed("candidate content that is not an object")),
        };
        match parts {
            None | Some(Value::Null) => {}
            Some(Value::Array(parts)) => {
                self.chunks += 1;
                self.placed.clear();
                let mut previous_text = false;
                for (part_index, part) in parts.iter().enumerate() {
                    // Empty unsigned text cannot settle an image awaiting its signature.
                    if self.pending_vertex_image.is_some()
                        && part.as_object().and_then(reader::select)
                            == Some(reader::SelectedKind::Text)
                        && part.str("text") == Some("")
                        && part.str("thoughtSignature").is_none()
                    {
                        self.part_index = part_index as u64;
                        self.part(part.clone(), out)?;
                        self.placed.push(self.placement.take());
                        continue;
                    }
                    let text = part.as_object().and_then(reader::select)
                        == Some(reader::SelectedKind::Text)
                        && part.bool("thought") != Some(true);
                    if text && previous_text {
                        self.close(out)?;
                    }
                    previous_text = text;
                    self.part_index = part_index as u64;
                    self.part(part.clone(), out)?;
                    self.placed.push(self.placement.take());
                }
            }
            Some(_) => return Err(malformed("candidate parts that are not a list")),
        }
        // Each chunk restates the grounding whole; recitation sources add up.
        if let Some(metadata) = candidate.get("groundingMetadata") {
            self.grounding =
                grounding::grounding(metadata, &self.placed, &self.answer, self.chunks > 1);
        }
        if let Some(metadata) = candidate.get("citationMetadata") {
            let recitations = grounding::recitations(metadata, &self.answer);
            self.recitations.extend(recitations);
        }
        Ok(Flow::More)
    }

    /// Configure independent carried provenance for the Vertex AI companion.
    /// Sole sanctioned caller: rig-vertexai's VertexDecoder. This changes only
    /// the issuer of carried payloads, never content, Origin, Native or endings.
    /// For that companion's internal use; not covered by semver.
    #[doc(hidden)]
    pub fn with_carried_issuer(mut self, issuer: &'static str) -> Self {
        self.carried_issuer = issuer;
        self
    }

    /// Select typed ordinary inline images for the Vertex AI companion.
    /// Sole sanctioned caller: rig-vertexai's VertexDecoder. This changes only
    /// image production for closed, unsigned, non-thought JPEG/PNG/WEBP/HEIC/HEIF
    /// parts. Selection waits until the next content part, EOF or error so a
    /// following bare signature can keep media inspection-only. The writer
    /// retains their original native evidence; provenance,
    /// other media, Origin and endings are unchanged.
    /// For that companion's internal use; not covered by semver.
    #[doc(hidden)]
    pub fn with_vertex_inline_images(mut self) -> Self {
        self.vertex_inline_images = true;
        self.defer_vertex_inline_images = true;
        self
    }

    pub(super) fn in_mode(mut self, mode: Mode) -> Self {
        self.streaming = mode == Mode::Streaming;
        self
    }

    fn normalized_finish(&self) -> Option<crate::completion::FinishReason> {
        self.finish
            .as_deref()
            .filter(|reason| *reason != "FINISH_REASON_UNSPECIFIED")
            .map(map_google_finish_reason)
    }

    /// End the reply on the finish reason the candidate holds. Without one
    /// the reply did not end.
    fn end(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        let reason = self.normalized_finish();
        let Some(_) = self.finish.take() else {
            return Err(ProviderError::Truncated);
        };
        self.close(&mut out)?;
        let citations = std::mem::take(&mut self.grounding)
            .into_iter()
            .chain(std::mem::take(&mut self.recitations));
        for (index, citation) in citations {
            out.cite(index, citation);
        }
        let usage = self
            .usage
            .as_ref()
            .map(checked_usage_of)
            .transpose()?
            .unwrap_or_default();
        let model = self.model_version.take();
        let response_id = self.response_id.take();
        Ok(out.end(Finish {
            usage,
            reason,
            response_id,
            model,
            ..Finish::default()
        }))
    }

    /// Finish the open text or thought run: Gemini states each part whole.
    fn close(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        self.flush_pending_vertex_image(out)?;
        let Some((index, thought)) = self.open.take() else {
            return Ok(());
        };
        let position = self.open_position.take().unwrap_or(CarriedPosition::Whole);
        if thought {
            let mut item = Value::Null;
            out.edit(index, |held| item = held.clone())?;
            let text = item.str("text").unwrap_or_default().to_owned();
            let signature = item.str("thoughtSignature").map(str::to_owned);
            let payload = CarriedPayload::new(item, Some(self.carried_issuer.to_owned()))
                .with_position(position);
            return out.finish_carried_reasoning(
                index,
                CarriedReasoning::new(
                    vec![CarriedReasoningPart::Text { text, signature }],
                    payload,
                ),
            );
        } else {
            let mut item = Value::Null;
            out.edit(index, |held| item = held.clone())?;
            if let Some(signature) = item.str("thoughtSignature") {
                let answer = GeminiSignedAnswer::new(
                    item.str("text").unwrap_or_default(),
                    GeminiThoughtSignature::new(signature),
                );
                let payload = CarriedPayload::new(item, Some(self.carried_issuer.to_owned()))
                    .with_position(position);
                return out.finish_carried_text(
                    index,
                    CarriedText::new(CarriedTextContent::GeminiSignedAnswer(answer), payload),
                );
            }
            if item.as_object().is_some_and(|fields| {
                fields
                    .keys()
                    .any(|key| !matches!(key.as_str(), "text" | "thought" | "thoughtSignature"))
            }) {
                let text = item.str("text").unwrap_or_default().to_owned();
                let payload = CarriedPayload::new(item, Some(self.carried_issuer.to_owned()))
                    .with_position(position);
                return out.finish_carried_text(
                    index,
                    CarriedText::new(CarriedTextContent::InspectionOnlyText(text), payload),
                );
            }
        }
        out.finish(index)
    }

    fn flush_pending_vertex_image(
        &mut self,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let Some(pending) = self.pending_vertex_image.take() else {
            return Ok(());
        };
        // Only this decoder owns the reserved fresh index. Nothing opened it,
        // and opening and finishing it here have no intervening writer call.
        if pending.fields.contains_key("thoughtSignature") {
            let payload = CarriedPayload::new(
                Value::Object(pending.fields),
                Some(self.carried_issuer.to_owned()),
            )
            .with_position(pending.position);
            out.open(pending.index, Block::Text, Value::Null)?;
            out.finish_carried_text(
                pending.index,
                CarriedText::new(
                    CarriedTextContent::InspectionOnlyMedia(String::new()),
                    payload,
                ),
            )
        } else {
            out.open(
                pending.index,
                Block::Image(pending.image),
                Value::Object(pending.fields),
            )?;
            out.finish(pending.index)
        }
    }

    /// Write a block at a fresh index, closing the run before it. A text or
    /// thought block stays open as the run later parts continue; any other
    /// is whole.
    fn open(
        &mut self,
        block: Block,
        item: Value,
        text: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.close(out)?;
        let index = out.fresh_index();
        let mut item = item;
        if let (Some(signature), Some(fields)) = (self.signature.take(), item.as_object_mut()) {
            fields
                .entry("thoughtSignature")
                .or_insert(Value::String(signature));
        }
        self.delivered = true;
        self.last = Some(index);
        self.signed = item
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .is_some();
        if matches!(block, Block::Text) {
            self.answer.open(index, text);
            self.placement = Some((index, 0));
        }
        let run = match &block {
            Block::Text => Some(false),
            Block::Reasoning { .. } => Some(true),
            _ => None,
        };
        out.open(index, block, item)?;
        out.push(index, text)?;
        self.open = run.map(|thought| (index, thought));
        self.open_position = run.map(|_| CarriedPosition::MessageContent {
            item_index: 0,
            part_index: self.part_index,
        });
        self.open.map_or_else(|| out.finish(index), |_| Ok(()))
    }

    /// Write one part of the candidate's content. A part this decoder does
    /// not model is an opaque item that replays; one with no data, or that
    /// is not an object, is kept as one that does not. A signature alone
    /// joins the block before it, as Gemini attaches it to the part before,
    /// or the next block when it comes first.
    fn part(&mut self, part: Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Value::Object(fields) = &part else {
            self.flush_pending_vertex_image(out)?;
            return self.open(Block::Opaque { replay: false }, part, "", out);
        };
        let selected = reader::select(fields);
        let thought = part.bool("thought") == Some(true);
        let signature = part.str("thoughtSignature");
        // An empty unsigned text part carries no content boundary.
        if selected == Some(reader::SelectedKind::Text)
            && part.str("text") == Some("")
            && signature.is_none()
        {
            return Ok(());
        }
        let bare = ["thought", "thoughtSignature", "partMetadata"];
        if !fields.keys().any(|key| !bare.contains(&key.as_str()))
            && let Some(signature) = signature
            && let Some(pending) = self.pending_vertex_image.as_mut()
        {
            pending
                .fields
                .insert("thoughtSignature".to_owned(), Value::from(signature));
            return Ok(());
        }
        self.flush_pending_vertex_image(out)?;
        if let Some(text) = part
            .str("text")
            .filter(|_| selected == Some(reader::SelectedKind::Text))
        {
            // A signed answer part ends its block. A signature on an empty
            // part, or an empty signature, keeps a block of its own and
            // never joins the unsigned text before it. A signed part with
            // text is the last piece of the answer the open unsigned run
            // streams, so it joins that run, as it does whole.
            if !thought
                && signature.is_some_and(|signature| text.is_empty() || signature.is_empty())
            {
                self.open(Block::Text, part.clone(), text, out)?;
                return self.close(out);
            }
            // A second signed part starts a block of its own, so each
            // signature stays with the part that carried it.
            let result = match self.open {
                Some((index, kind)) if kind == thought && !(self.signed && signature.is_some()) => {
                    self.signed |= signature.is_some();
                    if !thought {
                        self.placement = self.answer.push(index, text).map(|at| (index, at));
                    }
                    out.push(index, text)?;
                    out.edit(index, |item| merge_part(item, &part))
                }
                _ if thought => self.open(
                    Block::Reasoning { redacted: false },
                    part.clone(),
                    text,
                    out,
                ),
                _ => self.open(Block::Text, part.clone(), text, out),
            };
            result?;
            if signature.is_some() {
                return self.close(out);
            }
            return Ok(());
        }
        if let Some(call) = part
            .obj("functionCall")
            .filter(|_| selected == Some(reader::SelectedKind::FunctionCall))
        {
            let name = ToolName::new(call.get("name").and_then(Value::as_str).unwrap_or_default())
                .map_err(ProviderError::request)?;
            // Proto3 JSON leaves out an empty `args` Struct.
            let args = call
                .get("args")
                .map_or_else(|| "{}".to_owned(), Value::to_string);
            let id = CallId::from_wire(call.get("id").and_then(Value::as_str).unwrap_or_default());
            return self.open(
                Block::Call {
                    id,
                    name,
                    namespace: None,
                },
                part.clone(),
                &args,
                out,
            );
        }
        if selected == Some(reader::SelectedKind::InlineData) {
            self.close(out)?;
            let mut fields = fields.clone();
            if let Some(signature) = self.signature.take() {
                fields
                    .entry("thoughtSignature".to_owned())
                    .or_insert(Value::String(signature));
            }
            self.delivered = true;
            let index = out.fresh_index();
            self.last = Some(index);
            if self.vertex_inline_images
                && matches!(fields.get("thought"), None | Some(Value::Bool(false)))
                && fields
                    .keys()
                    .all(|key| matches!(key.as_str(), "inlineData" | "thought"))
                && let Some(inline) = fields.get("inlineData").and_then(Value::as_object)
                && inline
                    .keys()
                    .all(|key| matches!(key.as_str(), "mimeType" | "data"))
                && let Some(media_type) = inline
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .and_then(ImageMediaType::from_mime_type)
                && matches!(
                    &media_type,
                    ImageMediaType::JPEG
                        | ImageMediaType::PNG
                        | ImageMediaType::WEBP
                        | ImageMediaType::HEIC
                        | ImageMediaType::HEIF
                )
                && let Some(data) = inline.get("data").and_then(Value::as_str)
            {
                let image = Image {
                    data: DocumentSourceKind::Base64(data.to_owned()),
                    media_type: Some(media_type),
                    ..Image::default()
                };
                if self.defer_vertex_inline_images {
                    self.pending_vertex_image = Some(PendingVertexImage {
                        index,
                        image,
                        fields,
                        position: CarriedPosition::MessageContent {
                            item_index: 0,
                            part_index: self.part_index,
                        },
                    });
                    return Ok(());
                }
                out.open(index, Block::Image(image), Value::Object(fields))?;
                return out.finish(index);
            }
            let payload =
                CarriedPayload::new(Value::Object(fields), Some(self.carried_issuer.to_owned()))
                    .with_position(CarriedPosition::MessageContent {
                        item_index: 0,
                        part_index: self.part_index,
                    });
            out.open(index, Block::Text, Value::Null)?;
            return out.finish_carried_text(
                index,
                CarriedText::new(
                    CarriedTextContent::InspectionOnlyMedia(String::new()),
                    payload,
                ),
            );
        }
        match selected {
            Some(reader::SelectedKind::ExecutableCode) => {
                tracing::warn!(
                    kind = "executableCode",
                    "Gemini delivered an unmodeled hosted-code part"
                );
                return Ok(());
            }
            Some(reader::SelectedKind::CodeExecutionResult) => {
                tracing::warn!(
                    kind = "codeExecutionResult",
                    "Gemini delivered an unmodeled hosted-code part"
                );
                return Ok(());
            }
            Some(reader::SelectedKind::FunctionResponse | reader::SelectedKind::FileData) => {
                let kind = if selected == Some(reader::SelectedKind::FunctionResponse) {
                    "functionResponse"
                } else {
                    "fileData"
                };
                return Err(ProviderError::Response(format!(
                    "Gemini response part kind {kind} carries no assistant content rig can account for"
                )));
            }
            Some(
                reader::SelectedKind::Text
                | reader::SelectedKind::InlineData
                | reader::SelectedKind::FunctionCall,
            )
            | None => {}
        }
        let bare = ["thought", "thoughtSignature", "partMetadata"];
        let data = fields.keys().any(|key| !bare.contains(&key.as_str()));
        if !data && let Some(signature) = signature {
            let Some(index) = self.open.map(|(index, _)| index).or(self.last) else {
                self.signature = Some(signature.to_owned());
                return Ok(());
            };
            return out.edit_gemini_signature(index, signature, self.carried_issuer);
        }
        self.open(Block::Opaque { replay: data }, part.clone(), "", out)
    }

    /// Whether `frame` carries a response id and nothing else, a repeated
    /// key included.
    pub(crate) fn is_analysis_only(frame: &WireFrame) -> bool {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ResponseIdOnly {
            #[serde(rename = "responseId")]
            _id: String,
        }
        matches!(
            wire::classify_marker_keyed_frame::<ResponseIdOnly>(&frame.as_str(), &["responseId"]),
            WireEvent::Known(_)
        )
    }

    /// Project provider verdicts, response identity, and errors
    /// before normalization. A field of another type is left out, and a
    /// payload that is not JSON projects nothing.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(reply) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let candidate = reply.arr("candidates").first().unwrap_or(&Value::Null);
        let scrub = |value: Option<&str>| value.map(|value| sink.scrub(value));
        let block = reply
            .at("/promptFeedback/blockReason")
            .and_then(Value::as_str);
        let verdict = AdapterVerdict {
            finish_reason: scrub(candidate.str("finishReason")),
            block_reason: scrub(block),
            detail: scrub(candidate.str("finishMessage")),
            model: scrub(reply.str("modelVersion")),
        };
        let response_id = scrub(reply.str("responseId"));
        sink.provider(verdict, response_id);
        if let Some(error) = crate::observe::ObservedError::of(&reply) {
            error.emit(sink);
        }
    }
}

fn tool_protocol_finish_reason_error(candidate: &Value) -> Option<ProviderError> {
    let wire_reason = reader::finish_reason(candidate.get("finishReason")?)?;
    let reason = match wire_reason.as_str() {
        "MALFORMED_FUNCTION_CALL" => "MalformedFunctionCall",
        "UNEXPECTED_TOOL_CALL" => "UnexpectedToolCall",
        "MISSING_THOUGHT_SIGNATURE" => "MissingThoughtSignature",
        "TOO_MANY_TOOL_CALLS" => "TooManyToolCalls",
        "MALFORMED_RESPONSE" => "MalformedResponse",
        _ => return None,
    };
    let message = candidate
        .str("finishMessage")
        .unwrap_or("no finish message provided");
    Some(ProviderError::Response(format!(
        "Gemini stopped with finish_reason={reason}: {message}"
    )))
}

fn malformed(what: &str) -> ProviderError {
    ProviderError::Response(format!("Gemini sent {what}"))
}

/// Merge a streamed part into the part its block holds so far: the text
/// appends, a non-empty `thoughtSignature` replaces the one held, and any
/// other field replaces its namesake.
fn merge_part(item: &mut Value, part: &Value) {
    let (Value::Object(held), Value::Object(part)) = (&mut *item, part) else {
        *item = part.clone();
        return;
    };
    for (key, value) in part {
        if key == "text"
            && let (Some(Value::String(text)), Value::String(more)) = (held.get_mut("text"), value)
        {
            text.push_str(more);
        } else if key != "thoughtSignature" || value.as_str() != Some("") {
            held.insert(key.clone(), value.clone());
        }
    }
}

pub mod document;
mod grounding;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod decoder_tests;

#[cfg(test)]
mod upstream_tests;

#[cfg(test)]
mod vertex_image_tests;

#[cfg(test)]
mod vertex_defer_boundary_tests;
