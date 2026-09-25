//! Responses Lite request shaping for the Codex Responses contract.
//!
//! Lite moves the effective instructions and tools into a stable developer
//! prefix. The same pure body transform is used by HTTP and WebSocket sends;
//! each transport adds its own opt-in marker afterward.
//!
//! ```no_run
//! use rig_core::providers::{chatgpt, openai::OpenAI};
//! # fn wire() -> Result<(), Box<dyn std::error::Error>> {
//! let responses = OpenAI::with_key(&chatgpt::DIALECT, "access-token")
//!     .responses(chatgpt::GPT_5_3_CODEX)
//!     .with_responses_lite()?;
//! # let _ = responses;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::providers::openai::wire::ResponsesContract;

#[cfg(feature = "websocket")]
use super::codex_identity::CLIENT_METADATA_FIELD;
use super::codex_identity::CodexIdentity;
use super::{
    CompletionRequest, DeveloperRole, InputContent, InputItem,
    InternalChatMessageMetadataPassthrough, Message, Reasoning, ReasoningContext, SystemContent,
    SystemInstructionsPlacement, ToolChoice, ToolResultOutput, ToolResultOutputContent,
    UserContent,
};

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
    /// The effective tool list could not be represented as JSON for hashing
    /// and transmission.
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

/// Apply the transport-independent Lite body transform.
pub(crate) fn shape_request(
    request: &mut CompletionRequest,
    identity: &CodexIdentity,
) -> Result<(), ResponsesLiteError> {
    match request.additional_parameters.parallel_tool_calls {
        Some(true) => return Err(ResponsesLiteError::ParallelToolCalls),
        Some(false) => {}
        None => request.additional_parameters.parallel_tool_calls = Some(false),
    }

    match request.tool_choice.as_ref() {
        Some(ToolChoice::Mode(super::super::completion::ToolChoice::Auto)) => {}
        Some(_) => return Err(ResponsesLiteError::ToolChoice),
        None => {
            request.tool_choice =
                Some(ToolChoice::Mode(super::super::completion::ToolChoice::Auto));
        }
    }

    let reasoning = request
        .additional_parameters
        .reasoning
        .get_or_insert_with(Reasoning::default);
    match reasoning.context.as_ref() {
        Some(ReasoningContext::AllTurns) => {}
        Some(_) => return Err(ResponsesLiteError::ReasoningContext),
        None => reasoning.context = Some(ReasoningContext::AllTurns),
    }

    normalize_image_details(&mut request.input);

    let serialized_tools = serde_json::to_value(&request.tools)?;
    let Value::Array(serialized_tools) = serialized_tools else {
        return Err(ResponsesLiteError::ToolSerialization(
            <serde_json::Error as serde::ser::Error>::custom(
                "the Responses tools collection did not serialize as an array",
            ),
        ));
    };
    let folded_tools = fold_tools(serialized_tools)?;

    let thread_namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, identity.thread_id().as_bytes());
    let tools_id = Uuid::new_v5(&thread_namespace, &serde_json::to_vec(&folded_tools)?);
    let mut prefix = vec![InputItem {
        role: None,
        input: InputContent::AdditionalTools {
            id: format!("at_{tools_id}"),
            role: DeveloperRole::Developer,
            tools: folded_tools,
        },
    }];

    if let Some(instructions) = request.instructions.take()
        && !instructions.is_empty()
    {
        let message_id = Uuid::new_v5(&thread_namespace, instructions.as_bytes());
        prefix.push(InputItem {
            role: None,
            input: InputContent::Message(Message::Developer {
                id: Some(format!("msg_{message_id}")),
                content: vec![SystemContent::InputText { text: instructions }],
                name: None,
                internal_chat_message_metadata_passthrough: Some(
                    InternalChatMessageMetadataPassthrough::base_instructions(),
                ),
            }),
        });
    }

    request.tools.clear();
    request.input.splice(0..0, prefix);
    Ok(())
}

/// Remove image detail from a Lite incremental delta without rebuilding its
/// stable full-request prefix.
#[cfg(feature = "websocket")]
pub(crate) fn shape_delta(input: &mut [InputItem]) {
    normalize_image_details(input);
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
/// which is removed. An envelope selected by type and name whose
/// `description` is not a string, or whose `tools` is not an array, is
/// refused rather than dropped.
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

fn normalize_image_details(items: &mut [InputItem]) {
    for item in items {
        match &mut item.input {
            InputContent::Message(Message::User { content, .. }) => {
                for content in content {
                    if let UserContent::InputImage { detail, .. } = content {
                        *detail = None;
                    }
                }
            }
            InputContent::FunctionCallOutput(output) => {
                if let ToolResultOutput::Content(content) = &mut output.output {
                    for content in content {
                        if let ToolResultOutputContent::InputImage { detail, .. } = content {
                            *detail = None;
                        }
                    }
                }
            }
            InputContent::Message(
                Message::Developer { .. }
                | Message::System { .. }
                | Message::Assistant { .. }
                | Message::AssistantInput { .. },
            )
            | InputContent::AdditionalTools { .. }
            | InputContent::Reasoning(_)
            | InputContent::FunctionCall(_)
            | InputContent::CustomToolCall(_)
            | InputContent::CustomToolCallOutput(_)
            | InputContent::Compaction(_) => {}
        }
    }
}

#[cfg(test)]
mod tests;
