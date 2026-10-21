//! The decoder of native Ollama chat replies: a whole `/api/chat` body, or a
//! stream of NDJSON records of the same shape, the last one `done`.
//!
//! A record's `thinking` is reasoning, its `content` text and its
//! `tool_calls` whole calls. A reply with no `thinking` may carry its
//! reasoning as a terminated leading `<think>` block in a terminal record.
//! Nonterminal content remains literal text, including marker fragments.
//!
//! ```
//! use rig_core::providers::ollama::streaming::ChatDecoder;
//!
//! let decoder = ChatDecoder::default();
//! # let _ = decoder;
//! ```

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, LocalCallId, ProviderCallId, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterVerdict, Decoder, Flow, IncompleteEof, Mode, ObservationSink, Out, WireEvent, WireFrame,
};

/// One `/api/chat` record, as the daemon sent it.
#[derive(Debug, Default)]
pub struct ChatRecord(pub Map<String, Value>);

impl<'de> Deserialize<'de> for ChatRecord {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = Map::<String, Value>::deserialize(deserializer)?;
        if !fields.get("error").is_some_and(|error| !error.is_null()) {
            serde_json::from_value::<event::Record>(Value::Object(fields.clone()))
                .map_err(serde::de::Error::custom)?;
        }
        Ok(Self(fields))
    }
}

/// Decode native Ollama chat replies.
///
/// `thinking` and `content` each continue the block of the record before
/// while the kind stays the same, and each call is a whole block. The
/// reply ends at the record that says `done`; its `done_reason` is the
/// finish reason, `tool_calls` when a `stop` reply called a tool, and its
/// `prompt_eval_count`, `prompt_eval_cached_count` (the part of the prompt
/// read from the daemon's cache) and `eval_count` are the usage.
///
/// Terminal content splits a terminated leading `<think>` block when native
/// `thinking` is absent. Qwen3 also permits its prefilled closing boundary.
/// Unterminated blocks and nonterminal content fragments remain text.
#[derive(Debug)]
pub struct ChatDecoder {
    mode: Mode,
    /// Whether the reply called a tool.
    called: bool,
    /// The model the records name.
    model: Option<String>,
    next_call: u64,
}

impl Default for ChatDecoder {
    fn default() -> Self {
        Self::new(Mode::Unary)
    }
}

impl ChatDecoder {
    pub(super) fn new(mode: Mode) -> Self {
        Self {
            mode,
            called: false,
            model: None,
            next_call: 0,
        }
    }
}

/// Split only a terminated leading reasoning block, including Qwen's prefilled
/// blank-line boundary. An unterminated marker remains visible content.
fn split_legacy_thinking(content: &str, permits_omitted_start: bool) -> (Option<&str>, &str) {
    let trimmed = content.trim_start();
    let split = if let Some(reasoning_start) = trimmed.strip_prefix("<think>") {
        reasoning_start.split_once("</think>")
    } else if permits_omitted_start {
        trimmed.split_once("\n</think>\n\n")
    } else {
        None
    };
    let Some((reasoning, visible)) = split else {
        return (None, content);
    };
    let reasoning = reasoning.trim();
    if reasoning.is_empty() {
        return (None, visible.trim_start());
    }
    (Some(reasoning), visible.trim_start())
}

impl<'id> Decoder<'id, Completion> for ChatDecoder {
    type Event = ChatRecord;

    fn classify(&self, frame: WireFrame) -> WireEvent<ChatRecord> {
        wire::classify_untyped_line(frame.as_str().as_bytes())
    }

    /// Modeled fields have already been parsed; only a genuine `done` ends the reply.
    fn decode(
        &mut self,
        ChatRecord(fields): ChatRecord,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let record = Value::Object(fields);
        if let Some(error) = record.get("error").filter(|error| !error.is_null()) {
            let body = serde_json::json!({ "error": error }).to_string();
            return Err(ProviderError::from_provider_body(body));
        }
        if let Some(model) = record.str("model").filter(|model| !model.is_empty()) {
            tracing::Span::current().record("gen_ai.response.model", model);
            self.model = Some(model.to_owned());
        }
        let message = record.get("message").unwrap_or(&Value::Null);
        if message.str("role") == Some("assistant") {
            if let Some(thinking) = message
                .str("thinking")
                .filter(|thinking| !thinking.is_empty())
            {
                // A reply with native thinking splits nothing out of its content.
                reason(thinking, &mut out)?;
            }
            if let Some(content) = message.str("content").filter(|content| !content.is_empty()) {
                if record.bool("done") == Some(true)
                    && message.str("thinking").is_none_or(str::is_empty)
                {
                    let qwen = self
                        .model
                        .as_deref()
                        .is_some_and(|model| model.to_ascii_lowercase().contains("qwen3"));
                    let (legacy, visible) = split_legacy_thinking(content, qwen);
                    if let Some(legacy) = legacy {
                        reason(legacy, &mut out)?;
                    }
                    if !visible.is_empty() {
                        out.run(Block::Text, visible)?;
                    }
                } else {
                    out.run(Block::Text, content)?;
                }
            }
            for call in message.arr("tool_calls") {
                self.call(call, &mut out)?;
            }
        }
        if record.bool("done") != Some(true) {
            return Ok(Flow::More);
        }
        out.end_run()?;
        let reason = record.str("done_reason").map(|reason| match reason {
            "stop" if self.called => FinishReason::ToolCalls,
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            other => FinishReason::Other(other.to_owned()),
        });
        let (input, output) = (record.u64("prompt_eval_count"), record.u64("eval_count"));
        let usage = Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input.zip(output).map(|(input, output)| input + output),
            cached_input_tokens: record.u64("prompt_eval_cached_count"),
            ..Usage::default()
        };
        Ok(out.end(Finish {
            usage,
            reason,
            model: self.model.take(),
            ..Finish::default()
        }))
    }
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        out.end_run()?;
        match self.mode {
            Mode::Streaming => Ok(Flow::More),
            Mode::Unary => Err(ProviderError::Truncated),
        }
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        match self.mode {
            Mode::Streaming => IncompleteEof::KeepPartial,
            Mode::Unary => IncompleteEof::Refuse,
        }
    }

    fn flush_before_terminal_error(
        &mut self,
        mut out: Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        out.end_run()
    }
}

impl ChatDecoder {
    /// Write one whole call, refusing a name canonical tools cannot represent.
    fn call(&mut self, call: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let name = ToolName::new(
            call.at("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )
        .map_err(ProviderError::request)?;
        let arguments = match call.at("/function/arguments") {
            None | Some(Value::Null) => "{}".to_owned(),
            Some(Value::String(arguments)) => arguments.clone(),
            Some(arguments) => arguments.to_string(),
        };
        out.end_run()?;
        self.called = true;
        let position = self.next_call;
        self.next_call = position.checked_add(1)
            .ok_or_else(|| ProviderError::Response("Ollama tool-call position overflowed".to_owned()))?;
        let id = match call.str("id").and_then(ProviderCallId::new) {
            Some(id) => CallId::Provider(id),
            None => CallId::Local(LocalCallId::for_ollama_response_position(position)),
        };
        let index = out.fresh_index();
        out.open(
            index,
            Block::Call {
                id,
                name,
                namespace: None,
            },
            call.clone(),
        )?;
        out.push(index, &arguments)?;
        out.finish(index)
    }

    /// Project the verdict and an error envelope off one raw record
    /// before normalization discards them. A payload that is not JSON
    /// projects nothing.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(record) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let verdict = match record.str("done_reason") {
            Some(reason) => AdapterVerdict {
                finish_reason: Some(sink.scrub(reason)),
                block_reason: None,
                detail: None,
                model: record.str("model").map(|model| sink.scrub(model)),
            },
            None => AdapterVerdict::default(),
        };
        sink.provider(verdict, None);
        if let Some(error) = ObservedError::of(&record) {
            error.emit(sink);
        }
    }
}

/// Write a fragment of reasoning. Its block's item is the `thinking` it
/// joins, so a same-model replay sends it back.
fn reason(text: &str, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    let index = out.run(Block::Reasoning { redacted: false }, text)?;
    out.edit(index, |item| match item {
        Value::Object(fields) => {
            if let Some(Value::String(thinking)) = fields.get_mut("thinking") {
                thinking.push_str(text);
            }
        }
        _ => *item = serde_json::json!({ "thinking": text }),
    })
}

pub(crate) mod document;
mod event;

#[cfg(test)]
mod tests;
