//! The Messages reply decoder, for a whole message and for the event stream
//! alike. Each content block becomes one block, in wire order, whose
//! provider item is the block as the provider states it complete, and only
//! when the wire would take that item back.
//!
//! ```
//! use rig_core::providers::anthropic::streaming::MessagesDecoder;
//!
//! let decoder = MessagesDecoder::new(false);
//! # let _ = decoder;
//! ```

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::completion::object;
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, DocumentRange, Source, SourceLocation, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireCitation, WireEvent, WireFrame,
};

/// Recognized Messages event tags; any other tag classifies as unknown.
/// `message` is the whole message a unary reply is.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "message",
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
];

/// One Messages event, or the whole message a unary reply is, as the
/// provider sent it. Known payloads must have their typed fields; unknown
/// sibling fields remain in the original value.
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesEvent {
    /// The event as sent.
    pub fields: Value,
    /// The frame's text when it is an `error` event, whose envelope is
    /// reported verbatim.
    raw: Option<String>,
}

impl<'de> Deserialize<'de> for MessagesEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
        serde_json::from_str::<event::Event>(raw.get()).map_err(serde::de::Error::custom)?;
        let fields = serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        Ok(Self { fields, raw: None })
    }
}

impl MessagesEvent {
    fn kind(&self) -> &str {
        self.fields.str("type").unwrap_or_default()
    }

    /// The content block the event addresses.
    fn index(&self) -> Result<usize, ProviderError> {
        self.fields
            .u64("index")
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                ProviderError::Response(format!("Anthropic `{}` names no block index", self.kind()))
            })
    }

    /// The block or delta under `key`, which must name its `type`.
    fn item(&self, key: &str) -> Result<Map<String, Value>, ProviderError> {
        match self.fields.get(key) {
            Some(Value::Object(item)) if item.get("type").is_some_and(Value::is_string) => {
                Ok(item.clone())
            }
            _ => Err(ProviderError::Response(format!(
                "Anthropic `{}` carries no `{key}` with a string `type`",
                self.kind()
            ))),
        }
    }
}

/// Anthropic's usage counters, read leniently.
#[derive(Debug, Clone, Default, PartialEq)]
struct Counts {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
    /// The per-TTL breakdown of `cache_creation`, as sent.
    cache_creation_split: Option<Value>,
    thinking: Option<u64>,
}

impl Counts {
    fn of(usage: Option<&Value>) -> Self {
        let count = |pointer: &str| usage?.at(pointer)?.as_u64_lenient();
        Self {
            input: count("/input_tokens"),
            output: count("/output_tokens"),
            cache_read: count("/cache_read_input_tokens"),
            cache_creation: count("/cache_creation_input_tokens"),
            cache_creation_split: usage
                .and_then(|usage| usage.get("cache_creation"))
                .filter(|split| split.is_object())
                .cloned(),
            thinking: count("/output_tokens_details/thinking_tokens"),
        }
    }

    /// Rig's input is the uncached count. The total also includes cache reads
    /// and writes, and output already includes thinking.
    fn usage(&self) -> crate::completion::Usage {
        let input = self.input.map(|uncached| {
            uncached + self.cache_read.unwrap_or(0) + self.cache_creation.unwrap_or(0)
        });
        crate::completion::Usage {
            input_tokens: self.input,
            output_tokens: self.output,
            cached_input_tokens: self.cache_read,
            cache_creation_input_tokens: self.cache_creation,
            reasoning_tokens: self.thinking,
            total_tokens: input.zip(self.output).map(|(input, output)| input + output),
            tool_use_prompt_tokens: None,
            cost: None,
        }
    }
}

/// How a Messages `stop_reason` ends the turn, and the error a refusal
/// reports. A reason Anthropic does not document is
/// [`FinishReason::Other`], which fails the turn.
fn finish_of(reason: &str, details: Option<&Value>) -> (Option<FinishReason>, Option<String>) {
    let reason = match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        // A refusal fails the turn with its explanation (pi's rule).
        "refusal" => {
            let explanation = details
                .and_then(|details| details.str("explanation"))
                .filter(|explanation| !explanation.is_empty())
                .unwrap_or("The model refused to complete the request");
            return (
                Some(FinishReason::ContentFilter),
                Some(explanation.to_owned()),
            );
        }
        other => FinishReason::Other(other.to_owned()),
    };
    (Some(reason), None)
}

/// One Messages citation as a whole-block [`WireCitation`]: Anthropic
/// cites each text block as a whole. A location kind rig does not know, or
/// one without the field that names its source, is `None` and stays only in
/// the block's native item, as does `encrypted_index`. Document ranges end
/// exclusive, pages count from 1, characters and blocks from 0.
fn citation_of(citation: &Value) -> Option<WireCitation> {
    let number = |key: &str| citation.u64(key).and_then(|n| u32::try_from(n).ok());
    let range = |start: &str, end: &str| Some(number(start)?..number(end)?);
    let location = match citation.str("type")? {
        "char_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: citation
                .u64("start_char_index")
                .zip(citation.u64("end_char_index"))
                .map(|(start, end)| DocumentRange::Chars(start..end)),
        },
        "page_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: range("start_page_number", "end_page_number").map(DocumentRange::Pages),
        },
        "content_block_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: range("start_block_index", "end_block_index").map(DocumentRange::Blocks),
        },
        "search_result_location" => SourceLocation::SearchResult {
            index: number("search_result_index")?,
            source: citation.str("source")?.to_owned(),
            blocks: range("start_block_index", "end_block_index"),
        },
        "web_search_result_location" => SourceLocation::Url {
            url: citation.str("url")?.to_owned(),
        },
        _ => return None,
    };
    let mut source = Source::new(location);
    if let Some(title) = citation
        .str("document_title")
        .or_else(|| citation.str("title"))
    {
        source = source.title(title);
    }
    if let Some(cited) = citation.str("cited_text") {
        source = source.cited_text(cited);
    }
    Some(WireCitation::new(None, vec![source]))
}

/// What an open content block is, for the checks its end makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Thinking,
    Redacted,
    Call,
    Opaque,
    Carried,
}

/// Decodes Messages replies, a whole message or a stream of events.
/// `content_block_stop` states a block complete; a stop reason states every
/// block still open complete, but a call whose input is not yet JSON.
#[derive(Debug, Default)]
pub struct MessagesDecoder {
    /// Each open block's kind and the input JSON streamed to it.
    open: BTreeMap<usize, (Kind, String)>,
    /// Every index a block opened at, closed or not.
    started: BTreeSet<usize>,
    /// Indices whose signature is assembled from deltas rather than its opening fallback.
    signature_deltas: BTreeSet<usize>,
    /// Whether a block other than a leading `fallback` marker opened.
    opened: bool,
    /// Whether this is Anthropic itself, whose protected hosted blocks are carried.
    carried_hosted: bool,
    /// Exact hosted blocks awaiting their content-block boundary.
    hosted: BTreeMap<usize, Value>,
    /// Whether the dialect takes thinking back without a signature.
    unsigned_thinking: bool,
    /// Whether incomplete EOF may leave partial content available to a stream.
    streaming: bool,
    /// The counters `message_start` reported, for a terminal `message_delta`
    /// that does not repeat them.
    start: Counts,
    message_id: Option<String>,
    response_model: Option<String>,
    container: Option<Value>,
}

impl MessagesDecoder {
    /// A fresh decoder for one reply. `unsigned_thinking` is whether the
    /// dialect takes thinking back without a signature.
    pub fn new(unsigned_thinking: bool) -> Self {
        Self {
            unsigned_thinking,
            carried_hosted: true,
            ..Self::default()
        }
    }

    /// Select the actual issuer before decoding any provider frames.
    pub(super) fn for_provider(mut self, provider: &'static str) -> Self {
        self.carried_hosted = provider == super::PROVIDER_NAME;
        self
    }

    /// Select the reply mode before decoding any frames.
    pub(super) fn in_mode(mut self, mode: crate::wire::Mode) -> Self {
        self.streaming = mode == crate::wire::Mode::Streaming;
        self
    }

    /// Open the content block at `index` as the provider states it.
    fn start(
        &mut self,
        index: usize,
        block: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.started.insert(index);
        self.signature_deltas.remove(&index);
        let block = Value::Object(block);
        let kind = block.str("type").unwrap_or_default();
        // A leading `fallback` names the model that took over; one after
        // output began is a fallback rig cannot represent (pi's rule).
        if kind == "fallback" {
            if self.opened {
                return Err(ProviderError::Response(
                    "Anthropic performed an unsupported mid-output model fallback".to_owned(),
                ));
            }
            self.open.insert(index, (Kind::Opaque, String::new()));
            return out.open(index, Block::Opaque { replay: false }, block);
        }
        self.opened = true;
        if self.carried_hosted
            && matches!(
                kind,
                "server_tool_use" | "web_search_tool_result" | "code_execution_tool_result"
            )
        {
            self.open.insert(index, (Kind::Carried, String::new()));
            self.hosted.insert(index, block);
            return Ok(());
        }
        let (opened, kind, text) = match kind {
            "text" => (Block::Text, Kind::Text, block.str("text")),
            "thinking" => (
                Block::Reasoning { redacted: false },
                Kind::Thinking,
                block.str("thinking"),
            ),
            "redacted_thinking" => (Block::Reasoning { redacted: true }, Kind::Redacted, None),
            "tool_use" => {
                self.open.insert(index, (Kind::Call, String::new()));
                let id = block.str("id").unwrap_or_default().to_owned();
                let input = block.get("input").cloned().unwrap_or_default();
                match ToolName::new(block.str("name").unwrap_or_default()) {
                    Ok(name) => {
                        let id = CallId::from_wire(&id);
                        out.in_source_order(
                            crate::streaming::SourceOrder::new(0, index as u64),
                            |out| {
                                out.open(
                                    index,
                                    Block::Call {
                                        id,
                                        name,
                                        namespace: None,
                                    },
                                    block,
                                )
                            },
                        )?;
                    }
                    // A nameless call: the writer drops it with a warning.
                    Err(_) => out.fragment(
                        Some(index),
                        CallFragment {
                            id: Some(&id),
                            ..CallFragment::default()
                        },
                    )?,
                }
                // A whole reply states the input on the block; a stream
                // streams it, and the fragments win.
                return out.announce(index, input);
            }
            _ => (Block::Opaque { replay: true }, Kind::Opaque, None),
        };
        let text = text.unwrap_or_default().to_owned();
        let citations: Vec<WireCitation> = match kind {
            Kind::Text => block
                .arr("citations")
                .iter()
                .filter_map(citation_of)
                .collect(),
            _ => Vec::new(),
        };
        self.open.insert(index, (kind, String::new()));
        out.in_source_order(crate::streaming::SourceOrder::new(0, index as u64), |out| {
            out.open(index, opened, block)
        })?;
        for citation in citations {
            out.cite(index, citation);
        }
        out.push(index, &text)
    }

    /// Apply a delta to the open block at `index`: text and reasoning grow
    /// both the block and its item, input JSON is assembled, a citation is
    /// appended to the item and cited, and recognized compaction metadata merges into the item.
    fn delta(
        &mut self,
        index: usize,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !matches!(
            kind,
            "text_delta"
                | "thinking_delta"
                | "signature_delta"
                | "input_json_delta"
                | "citations_delta"
                | "compaction_delta"
        ) {
            return Ok(());
        }
        // The fragment a block is built from must be text.
        let fragment = |key: &str| {
            delta.get(key).and_then(Value::as_str).ok_or_else(|| {
                ProviderError::Response(format!("Anthropic `{kind}` carries no string `{key}`"))
            })
        };
        // A gateway that skips `content_block_start` still streams the
        // block's text, so the first delta opens it.
        if !self.started.contains(&index) {
            let opened = match kind {
                "text_delta" | "citations_delta" => Some(json!({"type": "text", "text": ""})),
                "thinking_delta" | "signature_delta" => {
                    Some(json!({"type": "thinking", "thinking": ""}))
                }
                _ => None,
            };
            if let Some(Value::Object(block)) = opened {
                self.start(index, block, out)?;
            }
        }
        match kind {
            "text_delta" => out.push(index, fragment("text")?)?,
            "thinking_delta" => out.push(index, fragment("thinking")?)?,
            "signature_delta" => {
                fragment("signature")?;
                if self.signature_deltas.insert(index) {
                    // The opening signature is a fallback. Deltas replace it.
                    out.edit(index, |item| {
                        if let Some(item) = item.as_object_mut() {
                            item.shift_remove("signature");
                        }
                    })?;
                }
            }
            "input_json_delta" => {
                let fragment = fragment("partial_json")?;
                let Some((kind @ (Kind::Call | Kind::Opaque | Kind::Carried), json)) =
                    self.open.get_mut(&index)
                else {
                    return Err(ProviderError::Response(format!(
                        "Anthropic streamed input to content block {index}, which takes none"
                    )));
                };
                json.push_str(fragment);
                if *kind == Kind::Call {
                    out.push(index, fragment)?;
                }
                return Ok(());
            }
            "citations_delta" => {
                let citation = delta.get("citation").cloned().unwrap_or_default();
                let cited = citation_of(&citation);
                out.edit(index, |item| {
                    if let Some(item) = item.as_object_mut() {
                        match item.get_mut("citations") {
                            Some(Value::Array(citations)) => citations.push(citation),
                            _ => {
                                item.insert("citations".to_owned(), Value::Array(vec![citation]));
                            }
                        }
                    }
                })?;
                if let Some(cited) = cited {
                    out.cite(index, cited);
                }
                return Ok(());
            }
            _ => {}
        }
        // Text, signatures and recognized compaction fragments extend their item.
        out.edit(index, |item| {
            crate::operation::completion::merge(item, &delta)
        })
    }

    /// End the block at `index` as stated complete. Its item becomes the
    /// block's native only when the wire takes it back: text that is not
    /// blank, thinking with its signature (unless the dialect takes it
    /// unsigned), redacted thinking with its data, and a call with an
    /// object `input`, set from what streamed. A hosted item whose input is
    /// not an object never completed.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some((kind, json)) = self.open.remove(&index) else {
            return out.finish(index);
        };
        if kind == Kind::Carried {
            let mut item = self.hosted.remove(&index).ok_or_else(|| {
                ProviderError::Response(format!("Anthropic carried block {index} has no payload"))
            })?;
            if item.str("type") == Some("server_tool_use") {
                let input = if json.is_empty() {
                    item.get("input")
                        .filter(|input| !input.is_null())
                        .cloned()
                        .unwrap_or(json!({}))
                } else {
                    serde_json::from_str(&json)?
                };
                item["input"] = input;
            }
            let payload =
                crate::message::CarriedPayload::new(item, Some(super::PROVIDER_NAME.to_owned()))
                    .with_position(crate::message::CarriedPosition::MessageContent {
                        item_index: 0,
                        part_index: index as u64,
                    });
            let text = crate::message::CarriedText::new(
                crate::message::CarriedTextContent::Opaque(String::new()),
                payload,
            );
            return out
                .in_source_order(crate::streaming::SourceOrder::new(0, index as u64), |out| {
                    out.carried_text(text)
                });
        }
        let streamed = (!json.is_empty()).then(|| crate::json_utils::parse_tool_arguments(&json));
        let unsigned = self.unsigned_thinking;
        let mut complete = true;
        out.edit(index, |item| {
            let input = match streamed {
                Some(Ok(parsed)) => Ok(Some(parsed)),
                Some(Err(_)) => Err(()),
                None => Ok(item.get("input").cloned()),
            };
            let kept = match (kind, input) {
                (Kind::Text, _) => !item.str("text").unwrap_or_default().trim().is_empty(),
                (Kind::Thinking, _) => {
                    unsigned || item.str("signature").is_some_and(|s| !s.is_empty())
                }
                (Kind::Redacted, _) => item.str("data").is_some_and(|data| !data.is_empty()),
                (Kind::Opaque, Ok(None)) => true,
                (Kind::Call | Kind::Opaque, Ok(Some(input @ Value::Object(_)))) => {
                    crate::operation::completion::merge(item, &object([("input", Some(input))]));
                    true
                }
                (Kind::Call, Ok(None | Some(Value::Null))) => {
                    crate::operation::completion::merge(
                        item,
                        &object([("input", Some(json!({})))]),
                    );
                    true
                }
                (Kind::Call, _) => false,
                (Kind::Carried, _) => false,
                (Kind::Opaque, _) => {
                    complete = false;
                    true
                }
            };
            if !kept {
                *item = Value::Null;
            }
        })?;
        if complete {
            out.finish(index)
        } else {
            out.close(index)
        }
    }

    /// Note the metadata a whole message or `message_start` states.
    fn metadata(&mut self, message: &Value) {
        self.start = Counts::of(message.get("usage"));
        self.message_id = message.str("id").map(str::to_owned);
        self.response_model = message.str("model").map(str::to_owned);
        self.note_container(message.get("container"));
    }

    fn note_container(&mut self, container: Option<&Value>) {
        self.container = container
            .filter(|c| !c.is_null())
            .or(self.container.as_ref())
            .cloned();
    }

    /// End the reply with Anthropic's terminal record. Every block still
    /// open is complete, but a call whose input is not JSON yet: the end of
    /// the reply closes it unfinished. The container the reply ran in is a
    /// last opaque block.
    fn end(
        &mut self,
        usage: &Counts,
        stop_reason: Option<&str>,
        details: Option<&Value>,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let open: Vec<usize> = self
            .open
            .iter()
            .filter(|(_, (kind, json))| {
                *kind != Kind::Call
                    || json.is_empty()
                    || crate::json_utils::parse_tool_arguments(json).is_ok_and(|v| v.is_object())
            })
            .map(|(index, _)| *index)
            .collect();
        for index in open {
            self.stop(index, &mut out)?;
        }
        if let Some(container) = &self.container {
            let index = out.fresh_index();
            let item = json!({ "type": "container", "container": container });
            out.whole(index, Block::Opaque { replay: true }, item, "")?;
        }
        let (reason, error) = stop_reason.map_or((None, None), |reason| finish_of(reason, details));
        Ok(out.end(Finish {
            usage: usage.usage(),
            reason,
            message_id: self
                .message_id
                .clone()
                .and_then(crate::completion::ProviderMessageId::new),
            response_id: self.message_id.clone(),
            model: self.response_model.clone(),
            error,
        }))
    }

    /// A whole message, written block by block through the calls a stream
    /// makes, then ended. Empty unary content is valid only for `end_turn`
    /// or `stop_sequence` with a reported sequence.
    fn whole(
        &mut self,
        message: Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.metadata(&message);
        let legal_empty_turn = match message.str("stop_reason") {
            Some("end_turn") => true,
            Some("stop_sequence") => message.str("stop_sequence").is_some(),
            _ => false,
        };
        if message.arr("content").is_empty() && !legal_empty_turn {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        for (index, block) in message.arr("content").iter().enumerate() {
            let Some(block) = block
                .as_object()
                .filter(|block| block.get("type").is_some_and(Value::is_string))
            else {
                return Err(ProviderError::Response(format!(
                    "Anthropic content block {index} has no string `type`"
                )));
            };
            self.start(index, block.clone(), &mut out)?;
            self.stop(index, &mut out)?;
        }
        let usage = self.start.clone();
        self.end(
            &usage,
            message.str("stop_reason"),
            message.get("stop_details"),
            out,
        )
    }

    /// The stream's terminal `message_delta` counters, falling back to
    /// `message_start`'s for those it omits.
    fn terminal(&self, usage: Option<&Value>) -> Counts {
        let (terminal, start) = (Counts::of(usage), self.start.clone());
        Counts {
            // Zero-as-missing is a gateway heuristic for the input count
            // only, not a rule for cache counts.
            input: terminal.input.filter(|tokens| *tokens > 0).or(start.input),
            cache_read: terminal.cache_read.or(start.cache_read),
            cache_creation: terminal.cache_creation.or(start.cache_creation),
            cache_creation_split: terminal.cache_creation_split.or(start.cache_creation_split),
            ..terminal
        }
    }

    fn close_incomplete(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let indexes: Vec<_> = self.open.keys().copied().collect();
        for index in indexes {
            if self
                .open
                .get(&index)
                .is_some_and(|(kind, _)| *kind == Kind::Carried)
            {
                if self.hosted.get(&index).and_then(|item| item.str("type"))
                    == Some("server_tool_use")
                {
                    // The protected server use is published only once its input completes.
                    self.open.remove(&index);
                    self.hosted.remove(&index);
                } else {
                    self.stop(index, out)?;
                }
            } else {
                self.open.remove(&index);
                out.close(index)?;
            }
        }
        Ok(())
    }
}

impl<'id> Decoder<'id, Completion> for MessagesDecoder {
    type Event = MessagesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<MessagesEvent> {
        let data = frame.as_str();
        wire::classify_tagged_frame::<MessagesEvent>(&data, "type", |tag| {
            KNOWN_EVENT_TYPES.contains(&tag)
        })
        .map(|mut event| {
            // The one event whose payload leaves this crate as bytes rather
            // than as decoded fields, so it is captured where the frame is
            // still in hand.
            event.raw = (event.kind() == "error").then(|| data.to_string());
            event
        })
    }

    fn decode(
        &mut self,
        event: MessagesEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event.kind() {
            "message" => return self.whole(event.fields, out),
            // A `message_start` without a message body (a Bedrock-compatible
            // gateway sends one) is a no-op.
            "message_start" => {
                if let Some(message) = event.fields.get("message").filter(|m| m.is_object()) {
                    self.metadata(message);
                }
            }
            "content_block_start" => {
                let block = event.item("content_block")?;
                self.start(event.index()?, block, &mut out)?;
            }
            "content_block_delta" => {
                let delta = event.item("delta")?;
                self.delta(event.index()?, delta, &mut out)?;
            }
            "content_block_stop" => self.stop(event.index()?, &mut out)?,
            "message_delta" => {
                let delta = event.fields.get("delta");
                self.note_container(delta.and_then(|delta| delta.get("container")));
                // Only a `message_delta` carrying a stop reason is the
                // provider's end; without one it is a no-op.
                let Some(reason) = delta.and_then(|delta| delta.str("stop_reason")) else {
                    return Ok(Flow::More);
                };
                let usage = self.terminal(event.fields.get("usage"));
                let details = delta.and_then(|delta| delta.get("stop_details"));
                return self.end(&usage, Some(reason), details, out);
            }
            // Preserve the complete error envelope rather than re-encode
            // modeled fields.
            "error" => {
                self.close_incomplete(&mut out)?;
                return Err(ProviderError::from_provider_body(
                    event.raw.unwrap_or_default(),
                ));
            }
            // `message_stop`, `ping`, and nothing else: the classifier
            // passes only the listed tags.
            _ => {}
        }
        Ok(Flow::More)
    }

    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.close_incomplete(&mut out)?;
        if self.streaming {
            Ok(Flow::More)
        } else {
            Err(ProviderError::Truncated)
        }
    }

    fn incomplete_eof(&self) -> crate::wire::IncompleteEof {
        if self.streaming {
            crate::wire::IncompleteEof::KeepPartial
        } else {
            crate::wire::IncompleteEof::Refuse
        }
    }

    fn flush_before_terminal_error(
        &mut self,
        mut out: Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        self.close_incomplete(&mut out)
    }
}

impl MessagesDecoder {
    /// Messages metadata projected before normalization can discard it: the
    /// stop reason, the model, the message id and any error envelope, on the
    /// unary reply and on the stream's `message_start`, `message_delta` and
    /// `error` events.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<ObservedPayload>(payload) else {
            return;
        };
        let usage = payload.usage;
        let (id, model, stop_reason, nested_usage) = match payload.message {
            Some(message) => (
                message.id,
                message.model,
                message.stop_reason,
                message.usage,
            ),
            None => (payload.id, payload.model, payload.stop_reason, None),
        };
        // Anthropic reports the prompt on `message_start` and the answer's
        // running total on each `message_delta`: each is a snapshot of what it
        // knows, never a sum.
        if let Some(usage) = usage.or(nested_usage) {
            sink.usage_snapshot(crate::observe::AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: None,
                cached_input_tokens: usage.cache_read_input_tokens,
                reasoning_tokens: usage
                    .output_tokens_details
                    .map(|details| details.thinking_tokens),
                tool_input_tokens: None,
            });
        }
        let stop_reason = stop_reason.or(payload.delta.and_then(|delta| delta.stop_reason));
        let verdict = AdapterVerdict {
            finish_reason: stop_reason.map(|v| sink.scrub(&v)),
            block_reason: None,
            detail: None,
            model: model.map(|v| sink.scrub(&v)),
        };
        let response_id = id.map(|v| sink.scrub(&v));
        sink.provider(verdict, response_id);
        if let Some(error) = payload.error
            && let Some(error) = ObservedError::of(&json!({"error":error}))
        {
            error.emit(sink);
        }
    }
}

#[derive(Deserialize)]
struct ObservedPayload {
    id: Option<String>,
    model: Option<String>,
    stop_reason: Option<String>,
    usage: Option<ObservedUsage>,
    message: Option<Box<ObservedPayload>>,
    delta: Option<ObservedDelta>,
    error: Option<Value>,
}

fn observed_count<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u64>, D::Error> {
    Value::deserialize(deserializer).map(|value| value.as_u64_lenient())
}

#[derive(Deserialize)]
struct ObservedUsage {
    #[serde(default, deserialize_with = "observed_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "observed_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "observed_count")]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens_details: Option<ObservedOutputDetails>,
}

#[derive(Deserialize)]
struct ObservedOutputDetails {
    #[serde(default)]
    thinking_tokens: u64,
}

#[derive(Deserialize)]
struct ObservedDelta {
    stop_reason: Option<String>,
}

pub(crate) mod document;
mod event;

/// Validate the private content shape before replaying annotated text.
pub(super) fn validate_content(content: &Value) -> Result<(), serde_json::Error> {
    serde_json::from_value::<event::Content>(content.clone()).map(|_| ())
}

#[cfg(test)]
mod tests;
