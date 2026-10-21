//! The Converse reply decoder. It reads the JSON Bedrock sent through
//! [`Lenient`]: a stream's events one by one, and a whole reply as the
//! events that would have streamed it, so both fold into the same turn.
//!
//! Reasoning retains its text, signature, redacted bytes and raw payload in
//! carried parts. Each redacted delta has its own part. Cited text and client
//! calls retain their completed wire items. Hosted tool uses and results stay
//! opaque items and replay together to the same model.

use std::collections::BTreeMap;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::completion::{FinishReason, Usage};
use rig_core::error::ProviderError;
use rig_core::json_utils::Lenient;
use rig_core::message::{
    CallId, CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart,
    DocumentRange, DocumentSourceKind, Image, ImageMediaType, LocalCallId, Source, SourceLocation, ToolName,
};
use rig_core::operation::{Block, CallFragment, Completion, Finish, merge};
use rig_core::streaming::DiscardReason;
use rig_core::wire::{Flow, IncompleteEof, Mode, Out, WireCitation, WireEvent};
use serde_json::{Map, Value, json};

use crate::completion::ConverseFrame;
use crate::types::errors;

/// What an open block is. The writer holds its JSON.
enum Open {
    Text,
    Reasoning,
    Call,
    Hosted,
    Image,
    Opaque,
}

/// One Converse reply's state: a whole reply or a stream of events.
#[derive(Default)]
pub struct StreamState {
    open: BTreeMap<usize, Open>,
    reason: Option<String>,
    mode: Option<Mode>,
    call_ordinal: u64,
}

/// The one key of a Converse union and its value.
fn member(value: &Value) -> Option<(&str, &Value)> {
    let (key, value) = value.as_object()?.iter().next()?;
    Some((key.as_str(), value))
}

fn reasoning_issuer(model: &str) -> &'static str {
    if model.contains("anthropic.claude") { "anthropic" } else { crate::completion::PROVIDER_NAME }
}

fn carried_payload(item: Value, index: usize) -> CarriedPayload {
    CarriedPayload::new(item, None).with_position(CarriedPosition::MessageContent {
        item_index: 0,
        part_index: index as u64,
    })
}

impl StreamState {
    pub(crate) fn in_mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Write one event in delivery order; the reply's end at `metadata`.
    fn event(
        &mut self,
        event: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<Option<Finish>, ProviderError> {
        let Some((kind, payload)) = member(event) else {
            return Ok(None);
        };
        let index = payload
            .u64("contentBlockIndex")
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(0);
        let part = |key| payload.get(key).unwrap_or(&Value::Null);
        match kind {
            "contentBlockStart" => self.start(out, index, part("start"))?,
            "contentBlockDelta" => self.delta(out, index, part("delta"), None)?,
            "contentBlockStop" => self.stop(out, index, None)?,
            "messageStop" => {
                self.reason = payload.str("stopReason").map(str::to_owned);
                let calls: Vec<usize> = self.open.iter()
                    .filter_map(|(index, open)| matches!(open, Open::Call).then_some(*index))
                    .collect();
                if self.reason.as_deref() == Some("tool_use") {
                    for index in calls {
                        self.open.remove(&index);
                        out.finish_call_at_bedrock_tool_use_terminal(index)?;
                    }
                } else {
                    let reason = match self.reason.as_deref() {
                        Some("max_tokens") => DiscardReason::Length,
                        _ => DiscardReason::Incomplete,
                    };
                    for index in calls {
                        self.open.remove(&index);
                        out.discard_call(index, reason)?;
                    }
                }
            }
            // The writer closes every block still open as incomplete.
            "metadata" => {
                let model = out.effective_model();
                out.stamp_carried_issuer(reasoning_issuer(&model));
                return Ok(Some(Finish {
                    usage: payload.at("/usage").map(usage).unwrap_or_default(),
                    reason: self.reason.as_deref().map(finish_reason),
                    ..Finish::default()
                }));
            }
            "messageStart" => {}
            kind if kind.ends_with("Exception") => return Err(errors::exception(kind, payload)),
            kind => skip(kind),
        }
        Ok(None)
    }

    /// Converse starts tool-use, tool-result and image blocks, and any
    /// block it adds later.
    fn start(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        start: &Value,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = member(start) else {
            return Ok(());
        };
        let open = match kind {
            // A call that never names its tool is dropped at its end, since
            // nothing can answer it.
            "toolUse" if body.str("type").is_none() => {
                let ordinal = self.call_ordinal;
                self.call_ordinal = ordinal.checked_add(1).ok_or_else(|| {
                    ProviderError::Response("too many Bedrock tool calls in one completion".into())
                })?;
                let id = body.str("toolUseId").filter(|id| !id.is_empty());
                if id.is_none() && let Some(name) = body.str("name").filter(|name| !name.is_empty()) {
                    let name = ToolName::new(name).map_err(|error| ProviderError::Response(error.to_string()))?;
                    out.open(index, Block::Call {
                        id: CallId::Local(LocalCallId::for_bedrock_response_position(ordinal)),
                        name,
                        namespace: None,
                    }, start.clone())?;
                } else {
                    out.fragment(Some(index), CallFragment {
                        id,
                        name: body.str("name"),
                        arguments: None,
                    })?;
                }
                out.edit(index, |slot| start.clone_into(slot))?;
                Open::Call
            }
            "image" if let Some(media_type) = media_type(body.str("format")) => {
                let image = Image {
                    data: DocumentSourceKind::Base64(String::new()),
                    media_type: Some(media_type),
                    ..Image::default()
                };
                out.open(index, Block::Image(image), Value::Null)?;
                Open::Image
            }
            // An image in a format Rig does not name keeps its JSON and does
            // not replay.
            kind => {
                let replay = kind != "image";
                out.open(index, Block::Opaque { replay }, start.clone())?;
                if kind == "toolUse" {
                    Open::Hosted
                } else {
                    Open::Opaque
                }
            }
        };
        self.open.insert(index, open);
        Ok(())
    }

    /// One delta: its text, reasoning or arguments go out as they arrive,
    /// and it merges into the block's JSON.
    fn delta(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        delta: &Value,
        whole: Option<&Value>,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = member(delta) else {
            return Ok(());
        };
        let redacted = body.get("redactedContent");
        if kind == "reasoningContent" {
            if let Some(data) = redacted.and_then(Value::as_str) {
                if matches!(self.open.get(&index), Some(Open::Reasoning)) {
                    self.stop(out, index, None)?;
                }
                BASE64_STANDARD.decode(data).map_err(|error| {
                    ProviderError::Response(format!("invalid Converse redacted reasoning: {error}"))
                })?;
                let block = out.fresh_index();
                let item = whole.cloned().unwrap_or_else(|| json!({"reasoningContent": body}));
                let reasoning = CarriedReasoning::new(
                    vec![CarriedReasoningPart::Redacted { data: data.to_owned() }],
                    carried_payload(item, index),
                );
                out.open(block, Block::Reasoning { redacted: true }, Value::Null)?;
                out.push(block, data)?;
                return out.finish_carried_reasoning(block, reasoning);
            }
            if !self.open.contains_key(&index)
                && body.str("text").is_none_or(str::is_empty)
                && body.str("signature").is_none()
            {
                return Ok(());
            }
        }
        if !self.open.contains_key(&index) {
            let (block, open) = match kind {
                "text" | "citation" => (Block::Text, Open::Text),
                "reasoningContent" => {
                    let redacted = redacted.is_some();
                    (Block::Reasoning { redacted }, Open::Reasoning)
                }
                "toolUse" => {
                    self.start(out, index, &json!({ "toolUse": {} }))?;
                    return self.delta(out, index, delta, whole);
                }
                kind => {
                    skip(kind);
                    return Ok(());
                }
            };
            out.open(index, block, Value::Null)?;
            self.open.insert(index, open);
        }
        let open = self.open.get(&index);
        if let (Some(text), Some(Open::Text | Open::Reasoning)) =
            (body.as_str().or_else(|| body.str("text")), open)
        {
            out.push(index, text)?;
        }
        let (pointer, merged) = match (kind, open) {
            ("text", Some(Open::Text)) => ("", json!({ "text": body })),
            ("citation", Some(Open::Text)) => {
                if let Some(cited) = citation_of(body) {
                    out.cite(index, cited);
                }
                ("", json!({ "citations": [body] }))
            }
            ("reasoningContent", Some(Open::Reasoning)) => ("", body.clone()),
            ("toolUse", Some(open @ (Open::Call | Open::Hosted))) => {
                if matches!(open, Open::Call) {
                    let arguments = body.str("input");
                    let fragment = CallFragment {
                        id: None,
                        name: None,
                        arguments,
                    };
                    out.fragment(Some(index), fragment)?;
                }
                ("/toolUse", body.clone())
            }
            ("toolResult", Some(Open::Opaque)) => ("/toolResult", json!({ "content": body })),
            ("image", Some(Open::Image)) => {
                if let Some(data) = body.at("/source/bytes").and_then(Value::as_str) {
                    out.push(index, data)?;
                }
                return Ok(());
            }
            (kind, _) => {
                skip(kind);
                return Ok(());
            }
        };
        out.edit(index, |item| {
            if let (Some(at), Value::Object(merged)) = (item.pointer_mut(pointer), &merged) {
                if kind == "reasoningContent" {
                    let mut rest = merged.clone();
                    let signature = rest.remove("signature");
                    merge(at, &rest);
                    if let Some(signature) = signature
                        && let Some(fields) = at.as_object_mut()
                    {
                        fields.insert("signature".to_owned(), signature);
                    }
                } else {
                    merge(at, merged);
                }
            }
        })
    }

    /// The block at `index` stopped: it becomes whole, with `whole` as its
    /// item when it came in a whole reply.
    fn stop(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        whole: Option<&Value>,
    ) -> Result<(), ProviderError> {
        let Some(open) = self.open.remove(&index) else {
            return Ok(());
        };
        if matches!(open, Open::Reasoning) {
            let mut item = Value::Null;
            out.edit(index, |value| item = value.clone())?;
            let text = item.str("text").unwrap_or_default().to_owned();
            let signature = item.str("signature").map(str::to_owned);
            let raw = whole.cloned().unwrap_or_else(|| {
                json!({"reasoningContent": {"reasoningText": item}})
            });
            let reasoning = CarriedReasoning::new(
                vec![CarriedReasoningPart::Text { text, signature }],
                carried_payload(raw, index),
            );
            return out.finish_carried_reasoning(index, reasoning);
        }
        if matches!(open, Open::Call) {
            let mut malformed = None;
            out.edit(index, |item| {
                let input = item.at("/toolUse/input").and_then(Value::as_str)
                    .filter(|input| !input.trim().is_empty()).unwrap_or("{}");
                if let Err(error) = serde_json::from_str::<Value>(input) {
                    let provider = item.at("/toolUse/toolUseId").and_then(Value::as_str)
                        .filter(|id| !id.is_empty()).and_then(|id| {
                            rig_core::message::CallId::from_wire(id).provider().cloned()
                        }).map(|call_id| rig_core::error::ProviderCallEvidence { call_id, item_id: None });
                    malformed = Some((input.to_owned(), error.to_string(), provider));
                }
            })?;
            if let Some((raw, error, provider)) = malformed {
                return Err(out.malformed_call_error(index, raw, error, provider));
            }
        }
        let call = matches!(open, Open::Call | Open::Hosted);
        out.edit(index, |item| {
            let built = match open {
                Open::Text => (!item.arr("citations").is_empty()).then(|| {
                    let content = [json!({ "text": item.get("text") })];
                    json!({ "citationsContent": { "content": content, "citations": item.get("citations") } })
                }),
                Open::Reasoning => None,
                // Converse takes a JSON object as a tool's input; a call's
                // input is the one its arguments were read from.
                Open::Call | Open::Hosted => {
                    let input = item
                        .at("/toolUse/input")
                        .and_then(Value::as_str)
                        .filter(|input| !input.trim().is_empty())
                        .unwrap_or("{}");
                    let input = serde_json::from_str::<Value>(input)
                        .ok()
                        .filter(Value::is_object)
                        .or_else(|| matches!(open, Open::Hosted).then(|| json!({})));
                    input.and_then(|input| {
                        let mut call = item.get("toolUse")?.clone();
                        merge(&mut call, &Map::from_iter([("input".to_owned(), input)]));
                        Some(json!({ "toolUse": call }))
                    })
                }
                Open::Image => None,
                Open::Opaque => Some(item.clone()),
            };
            *item = match (built, whole) {
                (Some(_), Some(whole)) if !call => whole.clone(),
                (built, _) => built.unwrap_or_default(),
            };
        })?;
        out.finish(index)
    }

    /// A whole reply, written as the events that would have streamed it.
    fn whole(
        &mut self,
        document: &Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let content = document
            .at("/output/message")
            .map_or(&[][..], |message| message.arr("content"));
        for (index, block) in content.iter().enumerate() {
            let Some((kind, body)) = member(block) else {
                continue;
            };
            // A started block's streamed field arrives as its delta.
            let (started, deltas) = match kind {
                "text" => (None, vec![json!({ "text": body })]),
                "citationsContent" => {
                    let text = body
                        .arr("content")
                        .iter()
                        .map(|part| json!({ "text": part.get("text") }));
                    let cited = body
                        .arr("citations")
                        .iter()
                        .map(|citation| json!({ "citation": citation }));
                    (None, text.chain(cited).collect())
                }
                "reasoningContent" => {
                    let reasoning = body.get("reasoningText").unwrap_or(body);
                    (None, vec![json!({ "reasoningContent": reasoning })])
                }
                "toolUse" => {
                    let input = body.get("input").map(Value::to_string);
                    (
                        Some("input"),
                        vec![json!({ "toolUse": { "input": input } })],
                    )
                }
                "toolResult" => (
                    Some("content"),
                    vec![json!({ "toolResult": body.get("content") })],
                ),
                "image" => (
                    Some("source"),
                    vec![json!({ "image": { "source": body.get("source") } })],
                ),
                _ => (Some(""), Vec::new()),
            };
            if let Some(field) = started {
                let mut opened = body.clone();
                if let Value::Object(fields) = &mut opened {
                    fields.shift_remove(field);
                }
                self.start(&mut out, index, &json!({ kind: opened }))?;
            }
            for delta in &deltas {
                self.delta(&mut out, index, delta, Some(block))?;
            }
            self.stop(&mut out, index, Some(block))?;
        }
        self.reason = document.str("stopReason").map(str::to_owned);
        let end = json!({ "metadata": { "usage": document.get("usage") } });
        Ok(match self.event(&end, &mut out)? {
            Some(end) => out.end(end),
            None => Flow::More,
        })
    }
}

/// One Converse citation as a whole-block [`WireCitation`]: Converse cites
/// each `citationsContent` block as a whole. A location kind this crate does
/// not know, or a search result without its `source`, is `None` and stays
/// only in the block's item. Ranges are as Anthropic's, which Converse
/// relays: end exclusive, pages from 1, characters and chunks from 0.
fn citation_of(citation: &Value) -> Option<WireCitation> {
    let (kind, at) = member(citation.get("location")?)?;
    let number = |key: &str| at.u64(key).and_then(|n| u32::try_from(n).ok());
    let range = || Some(number("start")?..number("end")?);
    let document = |within: Option<DocumentRange>| SourceLocation::Document {
        index: number("documentIndex"),
        id: None,
        within,
    };
    let location = match kind {
        "documentChar" => document(
            at.u64("start")
                .zip(at.u64("end"))
                .map(|(start, end)| DocumentRange::Chars(start..end)),
        ),
        "documentPage" => document(range().map(DocumentRange::Pages)),
        "documentChunk" => document(range().map(DocumentRange::Blocks)),
        "searchResultLocation" => SourceLocation::SearchResult {
            index: number("searchResultIndex")?,
            source: citation.str("source")?.to_owned(),
            blocks: range(),
        },
        "web" => SourceLocation::Url {
            url: at.str("url")?.to_owned(),
        },
        _ => return None,
    };
    let mut source = Source::new(location);
    if let Some(title) = citation.str("title") {
        source = source.title(title);
    }
    let quoted: Vec<&str> = citation
        .arr("sourceContent")
        .iter()
        .filter_map(|part| part.str("text"))
        .collect();
    if !quoted.is_empty() {
        source = source.cited_text(quoted.concat());
    }
    Some(WireCitation::new(None, vec![source]))
}

/// The image type a Converse image `format` names.
fn media_type(format: Option<&str>) -> Option<ImageMediaType> {
    match format? {
        "gif" => Some(ImageMediaType::GIF),
        "jpeg" => Some(ImageMediaType::JPEG),
        "png" => Some(ImageMediaType::PNG),
        "webp" => Some(ImageMediaType::WEBP),
        _ => None,
    }
}

fn skip(item: &str) {
    tracing::warn!(item, "a Converse item this crate does not model");
}

/// Every stop reason Converse documents. An end of turn or a stop sequence
/// is `Stop`, the token limit and an exceeded context window are `Length`,
/// guardrail and content-filter interventions are `ContentFilter`, and any
/// other reason keeps its wire spelling in `Other`, which fails the turn.
fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "content_filtered" | "guardrail_intervened" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Bedrock's reported input, output and total remain separate from its
/// cache read and write counters.
fn usage(usage: &Value) -> Usage {
    let cache_read = usage.u64("cacheReadInputTokens");
    let cache_write = usage.u64("cacheWriteInputTokens");
    Usage::new()
        .input_tokens(usage.u64("inputTokens"))
        .output_tokens(usage.u64("outputTokens"))
        .total_tokens(usage.u64("totalTokens"))
        .cached_input_tokens(cache_read)
        .cache_creation_input_tokens(cache_write)
}

impl<'id> rig_core::wire::Decoder<'id, Completion, ConverseFrame> for StreamState {
    type Event = ConverseFrame;

    fn classify(&self, frame: ConverseFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(frame)
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        if self.mode == Some(Mode::Streaming) {
            IncompleteEof::KeepPartial
        } else {
            IncompleteEof::Refuse
        }
    }

    /// EOF without Bedrock's `metadata` event is truncation.
    fn decode(
        &mut self,
        frame: ConverseFrame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match frame {
            ConverseFrame::Whole(document) => return self.whole(&document, out),
            ConverseFrame::Event(event) => event,
        };
        let Some(end) = self.event(&event, &mut out)? else {
            return Ok(Flow::More);
        };
        Ok(out.end(end))
    }
}

pub(crate) mod document;

#[cfg(test)]
pub(crate) mod tests;
