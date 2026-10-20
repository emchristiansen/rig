//! Carried Chat request policies and structured reasoning shapes.

use super::Quirks;
use crate::completion::FinishReason;
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::message::{CarriedReasoning, CarriedReasoningPart};
use serde_json::{Map, Value, json};

/// Shared helper for provider `finalize_request_body` hooks whose APIs take
/// message `content` as a plain string: flattens a content-part array to the
/// concatenation of its text parts. When `only_if_all_text` is set, arrays
/// containing non-text parts are left untouched (for APIs with their own
/// multimodal handling); otherwise non-text parts are dropped.
pub(super) fn flatten_text_content_parts(
    content: &mut serde_json::Value,
    separator: &str,
    only_if_all_text: bool,
) {
    // Refusals are textual content too; flatten them alongside text parts.
    // Checked per key so a null-padded `text` next to a string `refusal`
    // still counts as textual.
    fn part_text(part: &serde_json::Value) -> Option<&str> {
        part.get("text")
            .and_then(serde_json::Value::as_str)
            .or_else(|| part.get("refusal").and_then(serde_json::Value::as_str))
    }

    let Some(parts) = content.as_array() else {
        return;
    };
    if only_if_all_text && !parts.iter().all(|part| part_text(part).is_some()) {
        return;
    }
    let mut flattened = String::new();
    for text in parts.iter().filter_map(part_text) {
        if !flattened.is_empty() {
            flattened.push_str(separator);
        }
        flattened.push_str(text);
    }
    *content = serde_json::Value::String(flattened);
}

/// Remove tool messages, assistant tool calls and reasoning, and empty assistant turns.
/// Optionally strip names and flatten content using the supplied separator and
/// text-only guard. With `merge_same_role`, join adjacent user or assistant text
/// messages of the same role with newlines.
///
/// `reasoning_field` is the reasoning field the catalog names for the model.
/// Assistant `reasoning_content` is kept when it is that field (upstream sends
/// it for such a model on any gateway) and removed otherwise. A kept field
/// survives a same-role merge: the non-empty values are joined with newlines.
pub(super) fn sanitize_plain_text_history(
    messages: &mut Vec<serde_json::Value>,
    flatten: Option<(&str, bool)>,
    strip_names: bool,
    merge_same_role: bool,
    reasoning_field: Option<&str>,
) {
    messages
        .retain(|message| message.get("role").and_then(serde_json::Value::as_str) != Some("tool"));

    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(serde_json::Value::as_str) == Some("assistant") {
            object.shift_remove("tool_calls");
            if reasoning_field != Some("reasoning_content") {
                object.shift_remove("reasoning_content");
            }
        }
        if strip_names {
            object.shift_remove("name");
        }
        if let Some((separator, only_if_all_text)) = flatten
            && let Some(content) = object.get_mut("content")
        {
            flatten_text_content_parts(content, separator, only_if_all_text);
        }
    }

    messages.retain(|message| {
        if message.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
            return true;
        }
        match message.get("content") {
            Some(serde_json::Value::String(text)) => !text.is_empty(),
            Some(serde_json::Value::Null) | None => false,
            Some(_) => true,
        }
    });

    if !merge_same_role {
        return;
    }

    let mut merged: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    for message in std::mem::take(messages) {
        let merged_text = if let Some(role) = message
            .get("role")
            .and_then(serde_json::Value::as_str)
            .filter(|role| matches!(*role, "assistant" | "user"))
            && let Some(previous) = merged.last()
            && previous.get("role").and_then(serde_json::Value::as_str) == Some(role)
            && let Some(previous_text) = previous.get("content").and_then(serde_json::Value::as_str)
            && let Some(text) = message.get("content").and_then(serde_json::Value::as_str)
        {
            Some(format!("{previous_text}\n{text}"))
        } else {
            None
        };

        if let Some(text) = merged_text
            && let Some(previous) = merged.last_mut().and_then(serde_json::Value::as_object_mut)
        {
            previous.insert("content".to_string(), serde_json::Value::String(text));
            if let Some(field) = reasoning_field
                && let Some(reasoning) = message.get(field).and_then(serde_json::Value::as_str)
                && !reasoning.is_empty()
            {
                let joined = match previous.get(field).and_then(serde_json::Value::as_str) {
                    Some(earlier) if !earlier.is_empty() => format!("{earlier}\n{reasoning}"),
                    _ => reasoning.to_owned(),
                };
                previous.insert(field.to_owned(), serde_json::Value::String(joined));
            }
            continue;
        }
        merged.push(message);
    }
    *messages = merged;
}

/// Groq's raw built-ins move beside, rather than replacing, function tools.
pub(super) fn merge_groq_tools(compound: &mut Value, native: Vec<Value>) {
    let Some(compound) = compound.as_object_mut() else {
        return;
    };
    let enabled = compound.entry("enabled_tools").or_insert_with(|| json!([]));
    let Some(enabled) = enabled.as_array_mut() else {
        return;
    };
    for tool in native {
        let kind = tool.str("type");
        if !enabled.iter().any(|held| held.str("type") == kind) {
            enabled.push(tool);
        }
    }
}

/// The merged compound options retain their caller-supplied fields.
pub(crate) fn fold_groq_native_tools(
    map: &mut Map<String, Value>,
    raw: Option<&Value>,
) -> Result<(), EncodeError> {
    let Some(raw) = raw else {
        return Ok(());
    };
    let Value::Array(native) = raw else {
        return Err(EncodeError::request(
            "Groq `additional_params.tools` must be an array of native tool objects",
        ));
    };
    let compound = map.entry("compound_custom").or_insert_with(|| json!({}));
    merge_groq_tools(compound, native.clone());
    Ok(())
}

/// The shared Chat finish vocabulary (the fork's direct spellings plus the
/// `end`, `eos`, `end_turn` and `stop_sequence` compatible servers send for a
/// natural stop), then explicit dialect extensions. Anything else is
/// [`FinishReason::Other`].
pub(super) fn carried_finish_reason(reason: &str, quirks: &Quirks) -> FinishReason {
    crate::providers::internal::openai_chat_completions_compatible::CHAT_FINISHES
        .iter()
        .chain(quirks.finishes.iter())
        .find(|(name, _)| *name == reason)
        .map_or_else(
            || FinishReason::Other(reason.to_owned()),
            |(_, finish)| finish.clone(),
        )
}

/// Known carried blocks replay with F's ordered indices and exact signatures.
pub(super) fn carried_details(
    reasoning: &CarriedReasoning,
    start: usize,
) -> Result<Vec<Value>, EncodeError> {
    let id = reasoning.id.as_deref().filter(|id| !id.is_empty());
    reasoning.content.iter().enumerate().map(|(offset, part)| {
        let index = start + offset;
        Ok(match part {
            CarriedReasoningPart::Text { text, signature } => json!({
                "type": "reasoning.text", "id": id, "format": null, "index": index,
                "text": text, "signature": signature,
            }),
            CarriedReasoningPart::Summary(summary) => json!({
                "type": "reasoning.summary", "id": id, "format": null, "index": index, "summary": summary,
            }),
            CarriedReasoningPart::Encrypted(data) | CarriedReasoningPart::Redacted { data } => json!({
                "type": "reasoning.encrypted", "id": id, "format": null, "index": index, "data": data,
            }),
            CarriedReasoningPart::OpaqueSummary(_) | CarriedReasoningPart::OpaqueContent(_) => {
                return Err(crate::message::UnrepresentableOpaqueContent::new("OpenAI Chat Completions").into());
            }
        })
    }).collect()
}

/// F's unary argument contract, before opening any output blocks. The
/// original frame supplies failure evidence; a length budget drops only an
/// incomplete argument string on an otherwise well-formed function call.
pub(super) fn prepare_whole_calls(
    frame: &mut Value,
    source: &str,
    quirks: &Quirks,
) -> Result<(), crate::error::CorruptFrame> {
    let Some(choices) = frame.get_mut("choices").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for choice in choices {
        let cut = choice.str("finish_reason").map(|reason| carried_finish_reason(reason, quirks))
            .or_else(|| quirks.native_finish_reason.then(|| choice.str("native_finish_reason")
                .map(crate::providers::internal::openai_chat_completions_compatible::native_finish_reason)).flatten())
            == Some(FinishReason::Length);
        let Some(calls) = choice
            .pointer_mut("/message/tool_calls")
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        let incomplete = |call: &Value| {
            call.pointer("/function/arguments")
                .and_then(Value::as_str)
                .is_some_and(|raw| {
                    raw.trim().is_empty() || crate::json_utils::parse_tool_arguments(raw).is_err()
                })
        };
        if cut && calls.iter().any(incomplete) {
            // The fork validates the whole choice after repairing only
            // argument strings. A second call's independent defect must
            // keep the malformed reply loud, too.
            let mut probe = calls.clone();
            for call in &mut probe {
                if incomplete(call)
                    && let Some(arguments) = call.pointer_mut("/function/arguments")
                {
                    *arguments = Value::String("{}".to_owned());
                }
            }
            if let Err(error) =
                serde_json::from_value::<Vec<BudgetFunctionCall>>(Value::Array(probe))
            {
                return Err(crate::error::CorruptFrame::text(
                    Some("chat.completion".to_owned()),
                    source,
                    error,
                ));
            }
        }
        let mut kept = Vec::with_capacity(calls.len());
        for call in std::mem::take(calls) {
            let Some(Value::String(raw)) = call.pointer("/function/arguments") else {
                kept.push(call);
                continue;
            };
            let parsed = crate::json_utils::parse_tool_arguments(raw);
            let incomplete = raw.trim().is_empty() || parsed.is_err();
            let function_shape = call.str("type").is_none_or(|kind| kind == "function")
                && call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .is_some();
            if cut && incomplete && function_shape {
                continue;
            }
            if let Err(error) = parsed {
                return Err(crate::error::CorruptFrame::text(
                    Some("chat.completion".to_owned()),
                    source,
                    error,
                ));
            }
            kept.push(call);
        }
        *calls = kept;
    }
    Ok(())
}

// Private validation shapes for F's budget-repair probe; output stays the
// original value, never this parse. No retired provider DTO is reintroduced.
// The call `id` is not required: the decoder accepts an id-less call on
// every other path (streamed, and whole without a length cut), so the cut
// path discards it the same way rather than failing the reply.
#[derive(serde::Deserialize)]
struct BudgetFunctionCall {
    #[serde(default, rename = "type")]
    _kind: BudgetFunctionKind,
    #[serde(rename = "function")]
    _function: BudgetFunction,
}

#[derive(Default, serde::Deserialize)]
enum BudgetFunctionKind {
    #[default]
    #[serde(rename = "function")]
    Function,
}

#[derive(serde::Deserialize)]
struct BudgetFunction {
    #[serde(rename = "name")]
    _name: String,
    #[serde(
        rename = "arguments",
        deserialize_with = "crate::json_utils::stringified_json::deserialize_maybe_stringified"
    )]
    _arguments: Value,
}

/// A present choices field is a list in F; keep malformed-frame evidence
/// even though the U decoder reads valid entries as lenient values.
pub(super) fn check_choice_list(
    frame: &Value,
    source: &str,
) -> Result<(), crate::error::CorruptFrame> {
    #[derive(serde::Deserialize)]
    struct Choices {
        #[serde(rename = "choices")]
        _choices: Vec<Value>,
    }
    if frame
        .get("choices")
        .is_some_and(|choices| !choices.is_null() && !choices.is_array())
        && let Err(error) = serde_json::from_str::<Choices>(source)
    {
        return Err(crate::error::CorruptFrame::text(
            frame.str("object").map(str::to_owned),
            source,
            error,
        ));
    }
    Ok(())
}
