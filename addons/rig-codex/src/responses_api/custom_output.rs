//! The answer to a custom tool call, as the Codex contract sends it.
//!
//! Rig writes a tool result's `output` as a string when the result is one
//! text and as its ordered parts otherwise, for a custom call's answer as for
//! a function's, and states no `name` on a custom call's answer. This crate
//! sends a custom call's answer as the fork always has: its `output` is one
//! string, the result's texts joined in order; a result carrying anything
//! but text is refused; and the executed tool's `name` is stated. The name
//! is the one given by the request's result that produced the answer, else
//! the one the request's call was made with; an answer neither names goes
//! without one, as Rig wrote it.
//!
//! The rule holds for every `custom_tool_call_output` in the `input` Rig
//! encoded, whichever way the request states it: an answer in its history,
//! or an item it wrote itself in its options or `additional_params`. An
//! image has one way into a custom call's answer or none.
//!
//! The official Codex client's type lets a custom call's answer carry the
//! same payload as a function's, parts included, and makes `name` optional,
//! so the refusal and the name are this crate's contract rather than the
//! provider's requirement.
//!
//! The refusal reaches every image Rig would send in the answer. Rig's
//! driver prepares a request before a wire encodes it, and preparation
//! replaces an image the provider or model cannot receive in a tool result
//! with Rig's own words (`TOOL_IMAGE_OMITTED`, or a note that the image
//! follows in a user message). Such an image never reaches this crate, so
//! the answer is sent with those words: a caller that must refuse every
//! image in a custom call's answer refuses it before it calls the model.

use super::wire::CodexRequestError;
use rig_core::completion::CompletionRequest;
use rig_core::message::{CallId, Message, UserContent};
use serde_json::Value;
use serde_json::value::RawValue;
use std::collections::BTreeMap;

const CUSTOM_CALL: &str = "custom_tool_call";
const CUSTOM_OUTPUT: &str = "custom_tool_call_output";
const FUNCTION_OUTPUT: &str = "function_call_output";

/// The names of the tools a request's results say they ran, by the
/// provider's call id.
pub(crate) struct CustomOutputs {
    /// The executed tool of each result under a call id, in the order the
    /// results stand in the history.
    executed: BTreeMap<String, Vec<String>>,
    /// Whether the encoded `input` ends with the history's items. A request
    /// whose `additional_params` state `input` replaces them.
    history_sent: bool,
}

impl CustomOutputs {
    /// The executed tool of each result in `request` that answers a call the
    /// provider identified.
    pub(crate) fn of(request: &CompletionRequest) -> Self {
        let mut executed: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for message in &request.chat_history {
            let Message::User { content } = message.as_ref() else {
                continue;
            };
            for part in content {
                if let UserContent::ToolResult(result) = part
                    && let CallId::Provider(call_id) = &result.call
                {
                    executed
                        .entry(call_id.as_str().to_owned())
                        .or_default()
                        .push(result.name.as_str().to_owned());
                }
            }
        }
        let history_sent = request
            .additional_params
            .as_ref()
            .is_none_or(|params| params.get("input").is_none());
        Self {
            executed,
            history_sent,
        }
    }

    /// The executed tool for each of `items`, the `input` Rig encoded, that
    /// answers a call: `None` for every other item.
    ///
    /// Rig writes one output item per result, in history order, after the
    /// caller's own items. So the last outputs under a call id are its
    /// results' outputs, matched in order; an earlier one is a caller's item
    /// and is named after the last result under its id.
    fn executed_by_item<'a>(
        &self,
        items: impl Iterator<Item = Option<&'a Value>>,
    ) -> Vec<Option<String>> {
        let call_ids: Vec<Option<&str>> = items
            .map(|item| {
                let item = item?;
                matches!(
                    item.get("type").and_then(Value::as_str),
                    Some(CUSTOM_OUTPUT | FUNCTION_OUTPUT)
                )
                .then(|| item.get("call_id").and_then(Value::as_str))
                .flatten()
            })
            .collect();
        let mut outputs: BTreeMap<&str, usize> = BTreeMap::new();
        for call_id in call_ids.iter().flatten() {
            *outputs.entry(call_id).or_default() += 1;
        }
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        call_ids
            .iter()
            .map(|call_id| {
                let call_id = (*call_id)?;
                let names = self.executed.get(call_id)?;
                let ordinal = seen.entry(call_id).or_default();
                let at = *ordinal;
                *ordinal += 1;
                let total = outputs.get(call_id).copied().unwrap_or_default();
                let callers = if self.history_sent {
                    total.saturating_sub(names.len())
                } else {
                    total
                };
                at.checked_sub(callers)
                    .map_or_else(|| names.last(), |at| names.get(at))
                    .cloned()
            })
            .collect()
    }

    /// Rewrite each custom call's answer among `items`, the `input` Rig
    /// encoded. Returns whether any item changed; an item that did not is
    /// left with the bytes it had.
    pub(crate) fn restore_raw(
        &self,
        items: &mut [Box<RawValue>],
    ) -> Result<bool, CodexRequestError> {
        // An answer's place among the answers of its call id decides its name
        // only when that id has more than one result. Only then is a
        // function call's answer read as well; otherwise only a custom call
        // or its answer is, and every other item keeps its bytes unread.
        let placed = self.executed.values().any(|names| names.len() > 1);
        let mut values: Vec<Option<Value>> = Vec::with_capacity(items.len());
        for item in items.iter() {
            let text = item.get();
            values.push(
                if text.contains(CUSTOM_CALL) || (placed && text.contains(FUNCTION_OUTPUT)) {
                    Some(serde_json::from_str(text).map_err(CodexRequestError::Body)?)
                } else {
                    None
                },
            );
        }
        let called = called_names(values.iter().flatten());
        let executed = self.executed_by_item(values.iter().map(Option::as_ref));
        let mut changed = false;
        for ((item, value), executed) in items.iter_mut().zip(values).zip(executed) {
            let Some(mut value) = value else { continue };
            if Self::rewrite(&mut value, executed.as_deref(), &called)? {
                *item = serde_json::value::to_raw_value(&value).map_err(CodexRequestError::Body)?;
                changed = true;
            }
        }
        Ok(changed)
    }

    /// [`Self::restore_raw`] over parsed items.
    pub(crate) fn restore(&self, items: &mut [Value]) -> Result<(), CodexRequestError> {
        let called = called_names(items.iter());
        let executed = self.executed_by_item(items.iter().map(Some));
        for (item, executed) in items.iter_mut().zip(executed) {
            Self::rewrite(item, executed.as_deref(), &called)?;
        }
        Ok(())
    }

    /// Rewrite `item` if it is a custom call's answer, naming it `executed`,
    /// else after its call in `called`, when it states no name. Returns whether
    /// it changed.
    fn rewrite(
        item: &mut Value,
        executed: Option<&str>,
        called: &BTreeMap<String, String>,
    ) -> Result<bool, CodexRequestError> {
        let Value::Object(item) = item else {
            return Ok(false);
        };
        if item.get("type").and_then(Value::as_str) != Some(CUSTOM_OUTPUT) {
            return Ok(false);
        }
        let call_id = item
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let mut changed = false;
        match item.get("output") {
            Some(Value::String(_)) => {}
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    let fragment = (part.get("type").and_then(Value::as_str) == Some("input_text"))
                        .then(|| part.get("text").and_then(Value::as_str))
                        .flatten();
                    match fragment {
                        Some(fragment) => text.push_str(fragment),
                        None => return Err(CodexRequestError::CustomOutput { call_id }),
                    }
                }
                item.insert("output".to_owned(), Value::String(text));
                changed = true;
            }
            _ => return Err(CodexRequestError::CustomOutput { call_id }),
        }
        if !item.contains_key("name")
            && let Some(name) = executed.or_else(|| called.get(&call_id).map(String::as_str))
        {
            item.insert("name".to_owned(), Value::String(name.to_owned()));
            changed = true;
        }
        Ok(changed)
    }
}

/// The name each custom call among `items` was made with, by call id: what
/// its answer is named when the request does not say which tool ran.
fn called_names<'a>(items: impl Iterator<Item = &'a Value>) -> BTreeMap<String, String> {
    items
        .filter(|item| item.get("type").and_then(Value::as_str) == Some(CUSTOM_CALL))
        .filter_map(|item| {
            Some((
                item.get("call_id")?.as_str()?.to_owned(),
                item.get("name")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests;
