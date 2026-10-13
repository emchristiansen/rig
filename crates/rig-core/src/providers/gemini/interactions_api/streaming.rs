//! The decoder of Interactions replies: a whole interaction resource, or a
//! stream of `event_type`-tagged step events, read as JSON.
//!
//! ```
//! use rig_core::providers::gemini::interactions_api::streaming::InteractionsDecoder;
//!
//! let decoder = InteractionsDecoder::default();
//! # let _ = decoder;
//! ```

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{
    CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart, CarriedText, CarriedTextContent, DocumentRange, DocumentSourceKind, Image, ImageMediaType, Source, SourceLocation,
};
use crate::operation::completion::merge;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use super::PROVIDER_NAME;
use crate::wire::{Decoder, Flow, IncompleteEof, Mode, Out, SpanUnit, WireCitation, WireEvent, WireFrame, WireSpan};

/// Recognized Interactions SSE tags; unlisted tags classify as unknown.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Top-level keys that mark an untagged frame as a whole interaction
/// resource.
const INTERACTION_MARKER_KEYS: &[&str] = &["steps", "status", "usage", "object", "id"];

/// A decoded frame, verbatim: the decoder reads the fields it needs.
pub enum InteractionsEvent {
    /// One `event_type`-tagged streaming event.
    Sse(SseEvent),
    /// The whole interaction resource, as the unary reply delivers it.
    Whole(Map<String, Value>),
}

/// A streaming event: its tag, and its other fields verbatim.
#[derive(Debug, serde::Deserialize)]
pub struct SseEvent {
    /// The event's `event_type`.
    pub event_type: String,
    /// Every other field.
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

/// Decodes Interactions steps and preserves their original content order.
/// Signatures close carried thoughts. Genuine completed events finalize
/// announced calls; EOF retains partial content without a response terminal.
/// Text annotations attach when their model output step closes. Unknown
/// items keep the upstream opaque representation and replay policy.
#[derive(Default)]
pub struct InteractionsDecoder {
    streaming: bool,
    /// The open steps, by wire index.
    steps: BTreeMap<usize, Open>,
    call_order: Vec<usize>,
    call_ids_by_step: BTreeMap<usize, crate::message::ProviderCallId>,
    closed_full_calls: BTreeSet<crate::message::ProviderCallId>,
    full_updates_by_step: BTreeMap<usize, crate::message::ProviderCallId>,
    /// The model `interaction.created` named, for a completion that does
    /// not.
    model: Option<String>,
    /// The last status a stream reported.
    status: Option<String>,
}

/// What an open step decodes to.
enum Open {
    Thought(Option<Thought>),
    /// A model output: the step's other fields, and each content item with
    /// its block's writer index.
    Output(Value, Vec<(usize, Value)>),
    /// A function call, with the argument JSON streamed so far.
    Call { block: usize, arguments: Option<String> },
    Opaque(usize),
}

/// One thought run, retained after its provider step closes the signature.
struct Thought {
    index: usize,
    text: String,
    item: Value,
}

fn thought_text(item: &Value) -> String {
    item.arr("summary").iter()
        .filter(|part| part.str("type") == Some("text"))
        .filter_map(|part| part.str("text"))
        .collect()
}

fn open_thought(index: usize, item: Value, out: &mut Out<'_, Completion>) -> Result<Thought, ProviderError> {
    let text = thought_text(&item);
    out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
    if !text.is_empty() { out.push(index, &text)?; }
    Ok(Thought { index, text, item })
}

fn finish_thought(thought: Thought, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    let signature = thought.item.str("signature").map(str::to_owned);
    if thought.text.is_empty() && signature.is_none() {
        return out.finish(thought.index);
    }
    let reasoning = CarriedReasoning::new(
        vec![CarriedReasoningPart::Text { text: thought.text, signature }],
        CarriedPayload::new(thought.item, Some(PROVIDER_NAME.to_owned())),
    );
    out.finish_carried_reasoning(thought.index, reasoning)
}

/// Rig's usage for an interaction. Input and output stay separate from
/// thought and tool-use counts. A reported total wins; its fallback counts
/// every component only when both base counters exist.
pub fn usage_of(usage: &Value) -> Usage {
    let input_tokens = usage.u64("total_input_tokens");
    let output_tokens = usage.u64("total_output_tokens");
    let tool_use = usage.u64("total_tool_use_tokens");
    let thoughts = usage.u64("total_thought_tokens");
    let derived_total = input_tokens.zip(output_tokens).map(|(input, output)| {
        input + output + thoughts.unwrap_or(0) + tool_use.unwrap_or(0)
    });
    Usage {
        input_tokens,
        output_tokens,
        cached_input_tokens: usage.u64("total_cached_tokens"),
        reasoning_tokens: thoughts,
        tool_use_prompt_tokens: tool_use,
        total_tokens: usage.u64("total_tokens").or(derived_total),
        cache_creation_input_tokens: None,
        cost: None,
    }
}

/// Base64 bytes of a closed unsigned PNG item with no unselected fields.
fn closed_png_data(content: &Value) -> Option<&str> {
    let fields = content.as_object()?;
    if content.str("type") != Some("image")
        || content.str("mime_type") != Some("image/png")
        || !fields.keys().all(|key| matches!(key.as_str(), "type" | "data" | "mime_type"))
    {
        return None;
    }
    content.str("data")
}

/// Open the block for `content`, a model output's newest content item.
fn open_content(
    content: Value,
    step_index: usize,
    content_index: usize,
    out: &mut Out<'_, Completion>,
) -> Result<(usize, Value), ProviderError> {
    let index = out.fresh_index();
    if content.str("type") == Some("function_call") && content.str("name").is_some() {
        call_fields(index, &content, out)?;
        out.finish(index)?;
        return Ok((index, content));
    }
    if content.str("type") == Some("thought") {
        let thought = open_thought(index, content.clone(), out)?;
        finish_thought(thought, out)?;
        return Ok((index, content));
    }
    if content.str("type") == Some("image") {
        if let Some(data) = closed_png_data(&content) {
            let image = Image {
                data: DocumentSourceKind::Base64(data.to_owned()),
                media_type: Some(ImageMediaType::PNG),
                ..Image::default()
            };
            // The completed output step supplies the image's native item.
            out.open(index, Block::Image(image), Value::Null)?;
            return Ok((index, content));
        }
        out.open(index, Block::Text, Value::Null)?;
        let payload = CarriedPayload::new(content.clone(), Some(PROVIDER_NAME.to_owned()))
            .with_position(CarriedPosition::MessageContent { item_index: step_index as u64, part_index: content_index as u64 });
        out.finish_carried_text(index, CarriedText::new(CarriedTextContent::InspectionOnlyMedia(String::new()), payload))?;
        return Ok((index, content));
    }
    let block = match content.str("type") {
        Some("text") => Block::Text,
        _ => Block::Opaque { replay: true },
    };
    let text = matches!(block, Block::Text);
    out.open(index, block, Value::Null)?;
    if let Some(fragment) = content.str("text").filter(|_| text) {
        out.push(index, fragment)?;
    }
    Ok((index, content))
}

impl InteractionsDecoder {
    pub(super) fn in_mode(mut self, mode: Mode) -> Self {
        self.streaming = mode == Mode::Streaming;
        self
    }

    /// Open the step at `index` as it starts, writing the content it
    /// already states.
    fn start(
        &mut self,
        index: usize,
        step: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if step.str("type") != Some("thought") {
            self.flush_thoughts(out)?;
        }
        if step.str("type") == Some("function_call")
            && let Some(Open::Call { block, .. }) = self.steps.get(&index)
        {
            call_fields(*block, &step, out)?;
            if let Some(id) = step.str("id").and_then(crate::message::ProviderCallId::new) {
                self.call_ids_by_step.insert(index, id);
            }
            out.edit(*block, |item| *item = step)?;
            return Ok(());
        }
        let open = match step.str("type") {
            Some("thought") => {
                let signed = step.str("signature").is_some();
                let thought = open_thought(out.fresh_index(), step, out)?;
                if signed {
                    finish_thought(thought, out)?;
                    Open::Thought(None)
                } else {
                    Open::Thought(Some(thought))
                }
            }
            Some("model_output") => {
                let items = step.arr("content").iter();
                let items = items.enumerate().map(|(content_index, content)| open_content(content.clone(), index, content_index, out));
                let items = items.collect::<Result<_, _>>()?;
                let mut step = step;
                if let Some(step) = step.as_object_mut() {
                    step.insert("content".to_owned(), json!([]));
                }
                Open::Output(step, items)
            }
            Some("function_call") => {
                let block = out.fresh_index();
                call_fields(block, &step, out)?;
                if let Some(id) = step.str("id").and_then(crate::message::ProviderCallId::new) {
                    self.call_ids_by_step.insert(index, id);
                }
                out.edit(block, |slot| *slot = step)?;
                self.call_order.push(index);
                Open::Call { block, arguments: None }
            }
            // Input a reply restates is kept and never sent back; a
            // hosted-tool step, or one rig does not know, replays.
            kind => {
                let replay = !matches!(kind, Some("user_input" | "function_result"));
                let block = out.fresh_index();
                out.open(block, Block::Opaque { replay }, step)?;
                Open::Opaque(block)
            }
        };
        self.steps.insert(index, open);
        Ok(())
    }

    /// Apply one delta to the step at `index`. An arguments delta refuses
    /// when both an open call and a genuine retained association are absent.
    /// Provider legitimacy of orphan or resumed fragments remains UNKNOWN.
    fn delta(
        &mut self,
        index: usize,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if kind == "arguments_delta"
            && self.call_ids_by_step.get(&index)
                .is_some_and(|id| self.closed_full_calls.contains(id))
        {
            return Ok(());
        }
        // CLOSED-STEP-REFUSAL-TEXT-TYPED-FORM-LOSS differs from F:
        // Response retains the exact received delta only as compact JSON text.
        if kind == "function_call"
            && let Some(id) = delta.get("id").and_then(Value::as_str)
                .and_then(crate::message::ProviderCallId::new)
            && self.closed_full_calls.contains(&id)
        {
            return Err(ProviderError::Response(format!(
                "interactions: second full function call for closed call {}; received {}",
                id.as_str(), Value::Object(delta),
            )));
        }
        if let Some(id) = self.call_ids_by_step.get(&index)
            .filter(|id| self.closed_full_calls.contains(*id))
        {
            return Err(ProviderError::Response(format!(
                "interactions: {kind} fragment at index {index} belongs to closed call {}; received {}",
                id.as_str(), Value::Object(delta),
            )));
        }
        if kind == "arguments_delta"
            && !matches!(self.steps.get(&index), Some(Open::Call { .. }))
            && !self.call_ids_by_step.contains_key(&index)
        {
            return Err(ProviderError::Response(format!(
                "interactions: arguments fragment at index {index} has no open call; received {}",
                Value::Object(delta),
            )));
        }
        if kind == "function_call"
            && let Some(id) = delta.get("id").and_then(Value::as_str)
                .and_then(crate::message::ProviderCallId::new)
            && !self.closed_full_calls.contains(&id)
            && let Some(arguments) = delta.get("arguments").filter(|value| !value.is_object())
        {
            return Err(ProviderError::Response(format!(
                "interactions full-kind function call arguments are not a JSON object: call_id={} arguments={arguments}",
                id.as_str(),
            )));
        }
        let full_known_call = kind == "function_call"
            && delta.get("name").and_then(Value::as_str)
                .is_some_and(|name| crate::message::ToolName::new(name.to_owned()).is_ok())
            && delta.get("id").and_then(Value::as_str)
                .is_some_and(|id| !id.trim().is_empty())
            && delta.get("arguments").is_some_and(Value::is_object);
        if full_known_call && !self.steps.contains_key(&index) {
            self.flush_thoughts(out)?;
            let block = out.fresh_index();
            let call = Value::Object(delta);
            call_fields(block, &call, out)?;
            out.finish_call_at_terminal_without_native(block)?;
            if let Some(id) = call.str("id").and_then(crate::message::ProviderCallId::new) {
                self.call_ids_by_step.insert(index, id.clone());
                self.closed_full_calls.insert(id);
            }
            return Ok(());
        }
        if !self.steps.contains_key(&index) {
            let step = match kind.as_str() {
                "text" | "text_annotation_delta" | "image" | "audio" | "document" | "video" => {
                    "model_output"
                }
                "thought_summary" | "thought_signature" => "thought",
                "arguments_delta" => "function_call",
                other => other,
            };
            self.start(index, json!({ "type": step }), out)?;
        }
        let Some(open) = self.steps.get_mut(&index) else {
            return Ok(());
        };
        match (open, kind.as_str()) {
            // Text and its annotations extend the last text item and its
            // block; any other delta is a content item of its own.
            (Open::Output(_, items), _) => match items.last_mut() {
                Some((block, item))
                    if matches!(kind.as_str(), "text" | "text_annotation_delta")
                        && item.str("type") == Some("text") =>
                {
                    if let Some(fragment) = delta.get("text").and_then(Value::as_str)
                        && kind == "text"
                    {
                        out.push(*block, fragment)?;
                    }
                    let mut delta = delta;
                    delta.shift_remove("type");
                    merge(item, &delta);
                    Ok(())
                }
                _ => {
                    items.push(open_content(Value::Object(delta), index, items.len(), out)?);
                    Ok(())
                }
            },
            (Open::Thought(thought), "thought_summary" | "thought_signature") => {
                if thought.is_none() {
                    *thought = Some(open_thought(out.fresh_index(), json!({"type":"thought"}), out)?);
                }
                let Some(active) = thought.as_mut() else { return Ok(()); };
                if kind == "thought_summary" {
                    let content = delta.get("content").cloned().unwrap_or_default();
                    if let Some(text) = content.str("text").filter(|_| content.str("type") == Some("text")) {
                        active.text.push_str(text);
                        out.push(active.index, text)?;
                    }
                    merge(&mut active.item, &Map::from_iter([("summary".to_owned(), json!([content]))]));
                } else if let Some(signature) = delta.get("signature").and_then(Value::as_str) {
                    active.item["signature"] = json!(signature);
                    if let Some(active) = thought.take() {
                        finish_thought(active, out)?;
                    }
                }
                Ok(())
            }
            (Open::Call { block, arguments }, "arguments_delta") => {
                let fragment = match delta.get("arguments") {
                    Some(Value::String(fragment)) => fragment.clone(),
                    Some(Value::Null) | None => return Ok(()),
                    Some(other) => other.to_string(),
                };
                arguments.get_or_insert_with(String::new).push_str(&fragment);
                let fragment = CallFragment {
                    arguments: Some(&fragment),
                    ..CallFragment::default()
                };
                out.fragment(Some(*block), fragment)
            }
            (Open::Thought(thought), _) => {
                if let Some(thought) = thought { merge(&mut thought.item, &delta); }
                Ok(())
            }
            (Open::Call { block, .. }, _) => {
                let full_id = full_known_call.then(|| delta.get("id").and_then(Value::as_str)
                    .and_then(crate::message::ProviderCallId::new)).flatten();
                if kind == "function_call" {
                    call_fields(*block, &Value::Object(delta.clone()), out)?;
                }
                out.edit(*block, |item| {
                    if item.get("type") != delta.get("type") {
                        merge(item, &delta);
                    } else if let Some(item) = item.as_object_mut() {
                        item.extend(delta);
                    }
                })?;
                if let Some(id) = full_id {
                    self.call_ids_by_step.insert(index, id.clone());
                    self.full_updates_by_step.insert(index, id);
                }
                Ok(())
            }
            (Open::Opaque(block), _) => out.edit(*block, |item| {
                if item.get("type") != delta.get("type") {
                    merge(item, &delta);
                } else if let Some(item) = item.as_object_mut() {
                    item.extend(delta);
                }
            }),
        }
    }

    /// Finish the step at `index`: the API states it complete.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let full_id = self.full_updates_by_step.remove(&index);
        match self.steps.remove(&index) {
            None => Ok(()),
            Some(Open::Call { block, arguments }) => {
                self.call_order.retain(|open| *open != index);
                // Streamed fragments supersede the arguments the start
                // announced.
                if let Some(arguments) = arguments {
                    let arguments = crate::json_utils::parse_tool_arguments(&arguments).map_err(|error| {
                        out.malformed_call_error(block, arguments.clone(), error.to_string(), None)
                    })?;
                    out.announce(block, arguments.clone())?;
                    out.edit(block, |item| {
                        if let Some(item) = item.as_object_mut() {
                            item.insert("arguments".to_owned(), arguments);
                        }
                    })?;
                }
                out.finish(block)?;
                if let Some(id) = full_id {
                    self.closed_full_calls.insert(id);
                }
                Ok(())
            }
            // An output with no content is one empty text block holding it.
            Some(Open::Output(step, items)) if items.is_empty() => {
                let block = out.fresh_index();
                out.open(block, Block::Text, step)?;
                out.finish(block)
            }
            Some(Open::Output(step, items)) => {
                for (block, item) in items {
                    if (item.str("type") == Some("function_call") && item.str("name").is_some())
                        || item.str("type") == Some("thought")
                        || (item.str("type") == Some("image") && closed_png_data(&item).is_none()) {
                        continue;
                    }
                    for annotation in item.arr("annotations") {
                        if let Some(citation) = annotation_citation(annotation) {
                            out.cite(block, citation);
                        }
                    }
                    let mut step = step.clone();
                    if let Some(step) = step.as_object_mut() {
                        step.insert("content".to_owned(), json!([item]));
                    }
                    out.edit(block, |slot| *slot = step)?;
                    out.finish(block)?;
                }
                Ok(())
            }
            Some(Open::Thought(Some(thought))) => finish_thought(thought, out),
            Some(Open::Thought(None)) => Ok(()),
            Some(Open::Opaque(block)) => out.finish(block),
        }
    }

    fn flush_thoughts(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        for open in self.steps.values_mut() {
            if let Open::Thought(thought) = open && let Some(thought) = thought.take() {
                finish_thought(thought, out)?;
            }
        }
        Ok(())
    }

    /// End the reply with the interaction resource it completed with.
    fn complete(&mut self, interaction: Map<String, Value>, out: Out<'_, Completion>) -> Flow {
        let interaction = Value::Object(interaction);
        let status = interaction.str("status").map(str::to_owned).or_else(|| self.status.take());
        let (reason, error) = ending(status.as_deref(), interaction.arr("errors"));
        self.end_resource(interaction, reason, error, out)
    }

    fn end_resource(
        &mut self,
        interaction: Value,
        reason: Option<FinishReason>,
        error: Option<String>,
        out: Out<'_, Completion>,
    ) -> Flow {
        let field = |key: &str| interaction.str(key).map(str::to_owned);
        // An agent interaction names its agent in place of a model.
        let model = field("model")
            .or_else(|| field("agent"))
            .or_else(|| self.model.take());
        let usage = interaction.get("usage").cloned().unwrap_or(Value::Null);
        let finish = Finish {
            usage: usage_of(&usage),
            reason,
            response_id: field("id"),
            model,
            error,
            ..Finish::default()
        };
        out.end(finish)
    }
}

/// Map the status reported by a completed interaction resource. Other statuses
/// retain their exact spelling without synthesizing a lifecycle error. An SSE
/// completion with no status retains its missing-status error.
fn ending(status: Option<&str>, _errors: &[Value]) -> (Option<FinishReason>, Option<String>) {
    match status {
        Some("completed") => (Some(FinishReason::Stop), None),
        Some("requires_action") => (Some(FinishReason::ToolCalls), None),
        Some("budget_exceeded") => (Some(FinishReason::Length), None),
        Some(status) => (Some(FinishReason::Other(status.to_owned())), None),
        None => (None, Some("The interaction states no status".to_owned())),
    }
}

/// The citation a text item's annotation states: a `url_citation` cites a
/// web page, a `file_citation` a document and its page, a `place_citation`
/// a place by its URL or id. Offsets count bytes of the item's text.
/// Other annotations cite nothing.
fn annotation_citation(annotation: &Value) -> Option<WireCitation> {
    let field = |key: &str| annotation.str(key).map(str::to_owned);
    let (location, title) = match annotation.str("type")? {
        "url_citation" => (SourceLocation::Url { url: field("url")? }, field("title")),
        "file_citation" => {
            let page = annotation
                .u64("page_number")
                .and_then(|page| u32::try_from(page).ok());
            let within = page.map(|page| DocumentRange::Pages(page..page.saturating_add(1)));
            let id = field("document_uri");
            (
                SourceLocation::Document {
                    index: None,
                    id,
                    within,
                },
                field("file_name"),
            )
        }
        "place_citation" => {
            let location = match (field("url"), field("place_id")) {
                (Some(url), _) => SourceLocation::Url { url },
                (None, Some(id)) => SourceLocation::Document {
                    index: None,
                    id: Some(id),
                    within: None,
                },
                (None, None) => return None,
            };
            (location, field("name"))
        }
        _ => return None,
    };
    let mut source = Source::new(location);
    if let Some(title) = title {
        source = source.title(title);
    }
    // Proto3 JSON leaves out a zero start.
    let span = annotation.u64("end_index").map(|end| {
        WireSpan::new(
            annotation.u64("start_index").unwrap_or(0),
            end,
            SpanUnit::Bytes,
        )
    });
    Some(WireCitation::new(span, vec![source]))
}

/// Write the id, name and announced arguments a call step states.
fn call_fields(
    index: usize,
    step: &Value,
    out: &mut Out<'_, Completion>,
) -> Result<(), ProviderError> {
    let (id, name) = (step.str("id"), step.str("name"));
    out.fragment(
        Some(index),
        CallFragment {
            id,
            name,
            arguments: None,
        },
    )?;
    step.get("arguments")
        .map_or(Ok(()), |arguments| out.announce(index, arguments.clone()))
}

/// EOF without `interaction.completed` is truncation, not successful
/// completion, so the decoder has nothing to add at the end of the reply.
impl<'id> Decoder<'id, Completion> for InteractionsDecoder {
    type Event = InteractionsEvent;

    /// Classify a frame by its `event_type` tag, or as a whole resource by
    /// its marker keys when it carries no tag. A tagged frame never passes
    /// for a whole resource.
    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        wire::classify_or_untagged(
            &frame.as_str(),
            "event_type",
            |data| {
                wire::classify_tagged_frame(data, "event_type", |tag| {
                    KNOWN_EVENT_TYPES.contains(&tag)
                })
                .map(InteractionsEvent::Sse)
            },
            |data| {
                wire::classify_marker_keyed_frame(data, INTERACTION_MARKER_KEYS)
                    .map(InteractionsEvent::Whole)
            },
        )
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        if self.streaming { IncompleteEof::KeepPartial } else { IncompleteEof::Refuse }
    }

    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.flush_thoughts(&mut out)?;
        if self.streaming { Ok(Flow::More) } else { Err(ProviderError::Truncated) }
    }

    fn flush_before_terminal_error(&mut self, mut out: Out<'id, Completion>) -> Result<(), ProviderError> {
        self.flush_thoughts(&mut out)
    }

    fn decode(
        &mut self,
        event: InteractionsEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let SseEvent {
            event_type,
            fields: event,
        } = match event {
            InteractionsEvent::Sse(event) => event,
            InteractionsEvent::Whole(interaction) => {
                let steps = Value::Object(interaction.clone());
                for (index, step) in steps.arr("steps").iter().enumerate() {
                    self.start(index, step.clone(), &mut out)?;
                    self.stop(index, &mut out)?;
                }
                if !interaction.contains_key("status") {
                    return Ok(self.end_resource(Value::Object(interaction), None, None, out));
                }
                return Ok(self.complete(interaction, out));
            }
        };
        let event = Value::Object(event);
        let index = || {
            event
                .u64("index")
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    ProviderError::Response(format!(
                        "an Interactions `{event_type}` event names no step index"
                    ))
                })
        };
        let malformed = |field: &str| {
            ProviderError::Response(format!(
                "an Interactions `{event_type}` event carries no `{field}` object"
            ))
        };
        match event_type.as_str() {
            "step.start" => match event.get("step") {
                // A start that states no step leaves its deltas to open it.
                None => {}
                Some(step @ Value::Object(_)) => self.start(index()?, step.clone(), &mut out)?,
                Some(_) => return Err(malformed("step")),
            },
            "step.delta" => match event.get("delta") {
                Some(Value::Object(delta)) => self.delta(index()?, delta.clone(), &mut out)?,
                _ => return Err(malformed("delta")),
            },
            "step.stop" => self.stop(index()?, &mut out)?,
            "interaction.created" => {
                let field = |key: &str| {
                    event
                        .at(&format!("/interaction/{key}"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                };
                self.model = field("model").or_else(|| field("agent"));
                self.status = field("status");
            }
            "interaction.status_update" => {
                if let Some(status) = event.str("status") {
                    self.status = Some(status.to_owned());
                }
            }
            "interaction.completed" => {
                self.flush_thoughts(&mut out)?;
                for index in std::mem::take(&mut self.call_order) {
                    if let Some(Open::Call { block, .. }) = self.steps.remove(&index) {
                        out.finish_call_at_terminal_without_native(block)?;
                    }
                }
                let interaction = event.obj("interaction").cloned().unwrap_or_default();
                return Ok(self.complete(interaction, out));
            }
            "error" => {
                let Value::Object(mut body) = event else {
                    return Err(ProviderError::from_provider_body(event_type));
                };
                body.insert("event_type".to_owned(), Value::from(event_type));
                return Err(ProviderError::from_provider_body(
                    Value::Object(body).to_string(),
                ));
            }
            _ => {}
        }
        Ok(Flow::More)
    }
}

pub mod document;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod status_tests;

#[cfg(test)]
mod terminal_tests;

#[cfg(test)]
mod upstream_tests;
