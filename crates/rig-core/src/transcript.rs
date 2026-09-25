//! Conversation validation and constructors for real or synthetic tool results.
//!
//! ```
//! use rig_core::{message::Message, transcript::validate_canonical};
//!
//! validate_canonical(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), rig_core::transcript::TranscriptError>(())
//! ```

use std::collections::BTreeMap;

use crate::message::{
    AnsweredToolCall, AssistantContent, Message, ProviderCallId, ToolCallId, ToolResultContent,
    UserContent,
};
use crate::tool::ToolOutput;

/// Why a history is not a canonical transcript. See [`validate_canonical`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptError {
    /// Two assistant messages in a row (index of the second).
    #[error("consecutive assistant messages at index {index}")]
    ConsecutiveAssistant {
        /// Index of the offending (second) assistant message.
        index: usize,
    },
    /// An assistant tool call whose result is not in the next message.
    #[error("tool call `{call_id}` at index {index} has no result in the following message")]
    UnansweredToolCall {
        /// Index of the assistant message carrying the call.
        index: usize,
        /// The unanswered call id.
        call_id: ToolCallId,
    },
    /// A tool result that answers no call from the immediately preceding
    /// assistant message.
    #[error(
        "tool result `{call_id}` at index {index} answers no call from the preceding assistant message"
    )]
    OrphanToolResult {
        /// Index of the user message carrying the result.
        index: usize,
        /// The orphan result's call id.
        call_id: ToolCallId,
    },
    /// A tool result states that it answers a different kind of call than the
    /// call it pairs with (e.g. a function result for a custom call), so a
    /// wire with more than one result shape would emit the wrong item.
    #[error(
        "tool result `{call_id}` at index {index} answers a {answers:?} call, but the call it pairs with is a {call:?} call"
    )]
    MispairedToolResult {
        /// Index of the user message carrying the result.
        index: usize,
        /// The result's call id.
        call_id: ToolCallId,
        /// The kind of the call the result pairs with.
        call: AnsweredToolCall,
        /// The kind the result states it answers.
        answers: AnsweredToolCall,
    },
}

/// Rejects consecutive assistant messages, unanswered tool-call IDs, results
/// without a pending call, and results whose [`ToolResult::answers`] kind
/// differs from the kind of the call they pair with. Each pending ID must be
/// answered once in the next user message, before another assistant message or
/// the end of history.
///
/// [`ToolResult::answers`]: crate::message::ToolResult::answers
/// System messages reset the consecutive-assistant check but retain pending calls.
/// Duplicate call IDs are treated as one pending ID.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    let mut prev_assistant_calls: Option<BTreeMap<ToolCallId, AnsweredToolCall>> = None;
    let mut prev_was_assistant = false;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant { content, .. } => {
                if prev_was_assistant {
                    return Err(TranscriptError::ConsecutiveAssistant { index });
                }
                if let Some(call_id) = prev_assistant_calls
                    .take()
                    .and_then(|pending| pending.into_keys().next())
                {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index - 1,
                        call_id,
                    });
                }
                let mut calls: BTreeMap<ToolCallId, AnsweredToolCall> = BTreeMap::new();
                for item in content {
                    let (id, kind) = match item {
                        AssistantContent::ToolCall(call) => (&call.id, call.answered_by()),
                        AssistantContent::CustomToolCall(call) => (&call.id, call.answered_by()),
                        AssistantContent::Text(_)
                        | AssistantContent::Reasoning(_)
                        | AssistantContent::Image(_) => continue,
                    };
                    // Duplicate IDs are one pending ID; the first call's kind binds.
                    calls.entry(id.clone()).or_insert(kind);
                }
                prev_assistant_calls = (!calls.is_empty()).then_some(calls);
                prev_was_assistant = true;
            }
            Message::User { content } => {
                let mut pending = prev_assistant_calls.take().unwrap_or_default();
                for item in content.iter() {
                    if let UserContent::ToolResult(result) = item {
                        let id = result.call.clone();
                        match pending.remove(&id) {
                            None => {
                                return Err(TranscriptError::OrphanToolResult {
                                    index,
                                    call_id: id,
                                });
                            }
                            Some(call) if call != result.answers => {
                                return Err(TranscriptError::MispairedToolResult {
                                    index,
                                    call_id: id,
                                    call,
                                    answers: result.answers,
                                });
                            }
                            Some(_) => {}
                        }
                    }
                }
                if let Some(call_id) = pending.into_keys().next() {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index.saturating_sub(1),
                        call_id,
                    });
                }
                prev_was_assistant = false;
            }
            Message::System { .. } => {
                prev_was_assistant = false;
            }
        }
    }
    if let Some(call_id) = prev_assistant_calls.and_then(|pending| pending.into_keys().next()) {
        return Err(TranscriptError::UnansweredToolCall {
            index: messages.len().saturating_sub(1),
            call_id,
        });
    }
    Ok(())
}

fn tool_result_with(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    content: Vec<ToolResultContent>,
) -> UserContent {
    // Replay protocols require the executed tool's name separately from its call ID.
    UserContent::tool_result_for(call, provider, name, content)
}

/// Shape a canonical real tool output as a tool result answering a function
/// call, without reparsing text. A rig tool is a function tool.
pub fn tool_result_output(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    output: ToolOutput,
) -> UserContent {
    tool_result_with(call, provider, name, output.into_content())
}

/// Constructs a synthetic tool result answering a function call, containing
/// verbatim text such as recovery feedback or a skip reason. JSON-shaped text is
/// not reinterpreted as structured or multimodal output.
pub fn tool_result_message(
    call: ToolCallId,
    provider: Option<ProviderCallId>,
    name: String,
    message: String,
) -> UserContent {
    tool_result_with(call, provider, name, vec![ToolResultContent::text(message)])
}

#[cfg(test)]
mod validator_tests;
