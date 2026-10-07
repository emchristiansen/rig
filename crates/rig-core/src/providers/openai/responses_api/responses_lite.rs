//! Responses Lite request shaping for the Codex Responses contract.
//!
//! Lite moves the effective instructions and tools into a stable developer
//! prefix. The same pure body transform is used by HTTP and WebSocket sends;
//! each transport adds its own opt-in marker afterward.
//!
//! ```no_run
//! use rig_core::providers::{chatgpt, openai::{OpenAIConfig, wire::CallerIdentity, responses_api::wire::Responses}};
//! # fn wire() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::providers::openai::responses_api::codex_identity::CodexIdentity;
//! let caller = CallerIdentity::new("my-originator", "my-user-agent/1.0", None)?;
//! let config = OpenAIConfig::from_env_with(&chatgpt::DIALECT)?.with_caller_identity(caller);
//! let responses = Responses::new(config, chatgpt::GPT_5_3_CODEX)
//!     .with_codex_identity(CodexIdentity::from_ids("session-a", "thread-a")?)?
//!     .with_responses_lite()?;
//! # let _ = responses;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::providers::openai::wire::ResponsesContract;

use super::SystemInstructionsPlacement;
#[cfg(feature = "websocket")]
use super::codex_identity::CLIENT_METADATA_FIELD;
use super::codex_identity::CodexIdentity;

/// The request envelope a [`super::wire::Responses`] wire emits.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CodexRequestShape {
    /// Ordinary Responses fields and input items.
    #[default]
    Standard,
    /// The Codex Responses Lite developer prefix and transport marker.
    ResponsesLite,
}

impl CodexRequestShape {
    pub(crate) fn is_standard(&self) -> bool {
        *self == Self::Standard
    }

    pub(crate) fn is_lite(&self) -> bool {
        *self == Self::ResponsesLite
    }
}

/// A Responses Lite request that cannot be shaped without changing an
/// explicit caller choice or losing the stable identity of its prefix.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ResponsesLiteError {
    /// Lite is confined to the Codex Responses contract.
    #[error(
        "Responses Lite requires a Responses wire speaking the Codex contract; dialect `{dialect}` does not"
    )]
    ResponsesLiteRequiresCodex {
        /// The configured provider dialect.
        dialect: &'static str,
    },
    /// The deterministic prefix needs a stable thread id.
    #[error(
        "Responses Lite requires a stable Codex identity; attach one for HTTP or use the Codex WebSocket wrapper"
    )]
    ResponsesLiteRequiresIdentity,
    /// This placement leaves system messages outside the Lite developer
    /// prefix, contrary to the adapter policy.
    #[error(
        "Responses Lite does not support `InputSystemMessages`; place system instructions in the top-level instructions field"
    )]
    InputSystemMessages,
    /// Lite requires serial tool execution.
    #[error(
        "Responses Lite requires `parallel_tool_calls: false`; an explicit `true` cannot be overwritten"
    )]
    ParallelToolCalls,
    /// Lite requires automatic tool selection.
    #[error(
        "Responses Lite requires `tool_choice: auto`; an explicit conflicting tool choice cannot be overwritten"
    )]
    ToolChoice,
    /// Lite requires reasoning context from all turns.
    #[error(
        "Responses Lite requires `reasoning.context: all_turns`; an explicit conflicting context cannot be overwritten"
    )]
    ReasoningContext,
    /// A declared `functions` namespace selected the Lite fold but did not
    /// carry the typed namespace members that fold requires.
    #[error(
        "Responses Lite cannot fold a declared `functions` namespace whose `{field}` member is missing or has the wrong JSON type"
    )]
    MalformedFunctionsNamespace {
        /// The malformed namespace member.
        field: &'static str,
    },
    /// The effective tools could not be serialized, or a merged body member
    /// has the wrong JSON type for the request transform.
    #[error("Responses Lite could not serialize the effective tools: {0}")]
    ToolSerialization(#[from] serde_json::Error),
}

pub(crate) const HTTP_HEADER: &str = "x-openai-internal-codex-responses-lite";
#[cfg(feature = "websocket")]
pub(crate) const WS_METADATA_KEY: &str = "ws_request_header_x_openai_internal_codex_responses_lite";

pub(crate) fn validate_codex(
    contract: ResponsesContract,
    dialect: &'static str,
) -> Result<(), ResponsesLiteError> {
    if contract == ResponsesContract::Codex {
        Ok(())
    } else {
        Err(ResponsesLiteError::ResponsesLiteRequiresCodex { dialect })
    }
}

pub(crate) fn validate_activation<'identity>(
    shape: CodexRequestShape,
    contract: ResponsesContract,
    dialect: &'static str,
    placement: SystemInstructionsPlacement,
    identity: Option<&'identity CodexIdentity>,
) -> Result<Option<&'identity CodexIdentity>, ResponsesLiteError> {
    if shape.is_standard() {
        return Ok(None);
    }
    validate_codex(contract, dialect)?;
    if placement == SystemInstructionsPlacement::InputSystemMessages {
        return Err(ResponsesLiteError::InputSystemMessages);
    }
    identity
        .map(Some)
        .ok_or(ResponsesLiteError::ResponsesLiteRequiresIdentity)
}

/// Apply Lite controls and the stable developer prefix to the final merged body.
/// Canonical image detail is omitted by the typed input builder; raw items
/// in this body are preserved without traversal.
pub(crate) fn shape_body(
    body: &mut Map<String, Value>,
    identity: &CodexIdentity,
) -> Result<(), ResponsesLiteError> {
    match body.get("parallel_tool_calls") {
        Some(Value::Bool(false)) => {}
        None | Some(Value::Null) => {
            body.insert("parallel_tool_calls".to_owned(), Value::Bool(false));
        }
        Some(_) => return Err(ResponsesLiteError::ParallelToolCalls),
    }
    match body.get("tool_choice") {
        Some(Value::String(choice)) if choice == "auto" => {}
        None | Some(Value::Null) => {
            body.insert("tool_choice".to_owned(), Value::String("auto".to_owned()));
        }
        Some(_) => return Err(ResponsesLiteError::ToolChoice),
    }
    if matches!(body.get("reasoning"), None | Some(Value::Null)) {
        body.insert("reasoning".to_owned(), Value::Object(Map::new()));
    }
    let reasoning = body
        .get_mut("reasoning")
        .and_then(Value::as_object_mut)
        .ok_or(ResponsesLiteError::ReasoningContext)?;
    match reasoning.get("context") {
        Some(Value::String(context)) if context == "all_turns" => {}
        None | Some(Value::Null) => {
            reasoning.insert("context".to_owned(), Value::String("all_turns".to_owned()));
        }
        Some(_) => return Err(ResponsesLiteError::ReasoningContext),
    }

    let tools = match body.get("tools") {
        None => Vec::new(),
        Some(Value::Array(tools)) => tools.clone(),
        Some(_) => {
            return Err(body_type_error(
                "the Responses tools collection did not serialize as an array",
            ));
        }
    };
    let folded_tools = fold_tools(tools)?;
    let instructions = match body.get("instructions") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(body_type_error(
                "the Responses instructions did not serialize as a string",
            ));
        }
    };
    let input = body
        .get_mut("input")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            body_type_error("the Responses input collection did not serialize as an array")
        })?;

    let thread_namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, identity.thread_id().as_bytes());
    let tools_id = Uuid::new_v5(&thread_namespace, &serde_json::to_vec(&folded_tools)?);
    let mut prefix = vec![serde_json::json!({
        "type": "additional_tools",
        "id": format!("at_{tools_id}"),
        "role": "developer",
        "tools": folded_tools,
    })];
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        let message_id = Uuid::new_v5(&thread_namespace, instructions.as_bytes());
        prefix.push(serde_json::json!({
            "type": "message",
            "role": "developer",
            "id": format!("msg_{message_id}"),
            "content": [{"type": "input_text", "text": instructions}],
            "internal_chat_message_metadata_passthrough": {
                "content_item_kinds": ["model.base_instructions"],
            },
        }));
    }
    input.splice(0..0, prefix);
    body.remove("tools");
    body.remove("instructions");
    Ok(())
}

fn body_type_error(message: &str) -> ResponsesLiteError {
    ResponsesLiteError::ToolSerialization(<serde_json::Error as serde::ser::Error>::custom(message))
}

#[cfg(feature = "websocket")]
pub(crate) fn stamp_websocket_marker(
    body: &mut Map<String, Value>,
) -> Result<(), crate::error::EncodeError> {
    let metadata = body
        .entry(CLIENT_METADATA_FIELD)
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(metadata) = metadata else {
        return Err(crate::error::EncodeError::request(format!(
            "the Codex request's `{CLIENT_METADATA_FIELD}` must be a JSON object to carry the Responses Lite marker, got {metadata}"
        )));
    };
    metadata.insert(WS_METADATA_KEY.to_owned(), Value::String("true".to_owned()));
    Ok(())
}

/// Fold every top-level function/custom tool and every `functions` namespace
/// into one `functions` namespace at the position of the first. Nested
/// function/custom values remain exact JSON values, and every other top-level
/// value passes through unchanged.
///
/// The folded envelope is rebuilt from `type`, `name`, `description`, and
/// `tools` alone, so the fold discards caller data in three cases: members
/// other than those four on a `functions` namespace; every description but
/// the last non-blank one; and a namespace whose fold holds no functions,
/// which is removed. An envelope selected by type and name is refused rather
/// than dropped when its `description` is present but not a string, or its
/// `tools` is absent or not an array; an absent `description` counts as
/// blank.
fn fold_tools(tools: Vec<Value>) -> Result<Vec<Value>, ResponsesLiteError> {
    let mut functions = Vec::new();
    let mut description = String::new();
    let mut insertion_index = None;
    let mut passthrough = Vec::new();

    for tool in tools {
        let kind = tool.get("type").and_then(Value::as_str);
        let function_like = matches!(kind, Some("function" | "custom"));
        let functions_namespace = kind == Some("namespace")
            && tool.get("name").and_then(Value::as_str) == Some("functions");

        if function_like {
            insertion_index.get_or_insert(passthrough.len());
            functions.push(tool);
        } else if functions_namespace {
            insertion_index.get_or_insert(passthrough.len());
            let namespace_description = match tool.get("description") {
                Some(Value::String(description)) => description.as_str(),
                Some(_) => {
                    return Err(ResponsesLiteError::MalformedFunctionsNamespace {
                        field: "description",
                    });
                }
                None => "",
            };
            if !namespace_description.trim().is_empty() {
                description = namespace_description.to_owned();
            }
            let namespace_tools = tool
                .get("tools")
                .and_then(Value::as_array)
                .ok_or(ResponsesLiteError::MalformedFunctionsNamespace { field: "tools" })?;
            functions.extend(namespace_tools.iter().cloned());
        } else {
            passthrough.push(tool);
        }
    }

    if let Some(insertion_index) = insertion_index
        && !functions.is_empty()
    {
        let namespace = Value::Object(Map::from_iter([
            ("type".to_owned(), Value::String("namespace".to_owned())),
            ("name".to_owned(), Value::String("functions".to_owned())),
            ("description".to_owned(), Value::String(description)),
            ("tools".to_owned(), Value::Array(functions)),
        ]));
        passthrough.insert(insertion_index, namespace);
    }

    Ok(passthrough)
}

#[cfg(test)]
mod tests;
