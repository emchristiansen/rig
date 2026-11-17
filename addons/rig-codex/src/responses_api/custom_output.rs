//! The answer to a custom tool call, as the Codex contract sends it.
//!
//! Rig writes a tool result's `output` as a string when the result is one
//! text and as its ordered parts otherwise, for a custom call's answer as for
//! a function's, and states no `name` on a custom call's answer. This crate
//! sends a custom call's answer as the fork always has: its `output` is one
//! string, the result's texts joined in order; a result carrying anything
//! but text is refused; and the executed tool's `name` is stated. The name
//! is the one the request's result gives for the call, else the one the
//! request's call was made with; an answer neither names goes without one,
//! as Rig wrote it.
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

/// The names of the tools a request's results say they ran, by the
/// provider's call id.
pub(crate) struct CustomOutputs {
    executed: BTreeMap<String, String>,
}

impl CustomOutputs {
    /// The executed tool of each result in `request` that answers a call the
    /// provider identified.
    pub(crate) fn of(request: &CompletionRequest) -> Self {
        let mut executed = BTreeMap::new();
        for message in &request.chat_history {
            let Message::User { content } = message.as_ref() else {
                continue;
            };
            for part in content {
                if let UserContent::ToolResult(result) = part
                    && let CallId::Provider(call_id) = &result.call
                {
                    executed.insert(call_id.as_str().to_owned(), result.name.as_str().to_owned());
                }
            }
        }
        Self { executed }
    }

    /// Rewrite each custom call's answer among `items`, the `input` Rig
    /// encoded. Returns whether any item changed; an item that did not is
    /// left with the bytes it had.
    pub(crate) fn restore_raw(
        &self,
        items: &mut [Box<RawValue>],
    ) -> Result<bool, CodexRequestError> {
        let mut values: Vec<Option<Value>> = Vec::with_capacity(items.len());
        for item in items.iter() {
            // Only a custom call or its answer is read.
            values.push(if item.get().contains(CUSTOM_CALL) {
                Some(serde_json::from_str(item.get()).map_err(CodexRequestError::Body)?)
            } else {
                None
            });
        }
        let called = called_names(values.iter().flatten());
        let mut changed = false;
        for (item, value) in items.iter_mut().zip(values) {
            let Some(mut value) = value else { continue };
            if self.rewrite(&mut value, &called)? {
                *item = serde_json::value::to_raw_value(&value).map_err(CodexRequestError::Body)?;
                changed = true;
            }
        }
        Ok(changed)
    }

    /// [`Self::restore_raw`] over parsed items.
    pub(crate) fn restore(&self, items: &mut [Value]) -> Result<(), CodexRequestError> {
        let called = called_names(items.iter());
        for item in items {
            self.rewrite(item, &called)?;
        }
        Ok(())
    }

    /// Rewrite `item` if it is a custom call's answer. Returns whether it
    /// changed.
    fn rewrite(
        &self,
        item: &mut Value,
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
            && let Some(name) = self.executed.get(&call_id).or_else(|| called.get(&call_id))
        {
            item.insert("name".to_owned(), Value::String(name.clone()));
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
