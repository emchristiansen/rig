//! The JSON events a Live session exchanges on its `oai-events` data channel.
//!
//! Server events decode by their `type`. A type this module does not model
//! decodes as [`ServerEvent::Unknown`] with its whole JSON object. Every
//! modelled object keeps the fields it does not model in `extra`, and a
//! [`ResponseEvent`] keeps its nested Responses event whole beside the part
//! this module types.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use super::{ApiErrorDetail, ResponsesDelegationUpdate};
use crate::providers::live_support::error::{CorruptFrame, ProviderError};

/// One server event.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerEvent {
    /// `session.started`: the session is running.
    SessionStarted(SessionSnapshot),
    /// `session.updated`: a `session.update` was accepted.
    SessionUpdated(SessionSnapshot),
    /// `session.closed`: the session finished finalizing.
    SessionClosed(SessionClosed),
    /// `session.input_transcript.delta`: a fragment of the user's transcript.
    InputTranscriptDelta(TranscriptDelta),
    /// `session.output_transcript.delta`: a fragment of the Live model's
    /// transcript.
    OutputTranscriptDelta(TranscriptDelta),
    /// `session.delegation.created`: the Live model delegated work.
    DelegationCreated(DelegationCreated),
    /// `response.event`: an event of the Responses backend.
    ResponseEvent(ResponseEvent),
    /// `session.usage.updated`: cumulative audio usage.
    UsageUpdated(UsageUpdated),
    /// `session.instructions.appended`: an instructions append was accepted.
    InstructionsAppended(Appended),
    /// `session.thinking.appended`: a thinking append was accepted.
    ThinkingAppended(Appended),
    /// `session.commentary.appended`: a commentary append was accepted.
    CommentaryAppended(Appended),
    /// `session.input_audio.muted`: input audio no longer reaches the model.
    InputAudioMuted(Acknowledgement),
    /// `session.input_audio.unmuted`: input audio reaches the model again.
    InputAudioUnmuted(Acknowledgement),
    /// `error`: the server reports a failure.
    Error(ErrorEvent),
    /// A `type` this module does not model.
    Unknown(UnknownEvent),
}

impl ServerEvent {
    /// Decode one data channel message.
    ///
    /// A payload that is not a JSON object with a string `type`, or a
    /// modelled type whose fields do not match, is refused as
    /// [`ProviderError::CorruptFrame`] carrying the payload. Any other `type`
    /// decodes as [`ServerEvent::Unknown`].
    pub fn parse(payload: &str) -> Result<Self, ProviderError> {
        let corrupt = |kind: Option<&str>, error| {
            ProviderError::CorruptFrame(CorruptFrame::text(
                kind.map(ToOwned::to_owned),
                payload,
                error,
            ))
        };
        let object: Map<String, Value> =
            serde_json::from_str(payload).map_err(|error| corrupt(None, error))?;
        let Some(Value::String(kind)) = object.get("type") else {
            return Err(corrupt(
                None,
                serde::de::Error::custom("a server event needs a string `type`"),
            ));
        };
        let kind = kind.clone();

        fn body<T: DeserializeOwned>(mut object: Map<String, Value>) -> serde_json::Result<T> {
            object.remove("type");
            serde_json::from_value(Value::Object(object))
        }
        let event = match kind.as_str() {
            "session.started" => body(object).map(Self::SessionStarted),
            "session.updated" => body(object).map(Self::SessionUpdated),
            "session.closed" => body(object).map(Self::SessionClosed),
            "session.input_transcript.delta" => body(object).map(Self::InputTranscriptDelta),
            "session.output_transcript.delta" => body(object).map(Self::OutputTranscriptDelta),
            "session.delegation.created" => body(object).map(Self::DelegationCreated),
            "response.event" => body(object)
                .and_then(ResponseEvent::from_wire)
                .map(Self::ResponseEvent),
            "session.usage.updated" => body(object).map(Self::UsageUpdated),
            "session.instructions.appended" => body(object).map(Self::InstructionsAppended),
            "session.thinking.appended" => body(object).map(Self::ThinkingAppended),
            "session.commentary.appended" => body(object).map(Self::CommentaryAppended),
            "session.input_audio.muted" => body(object).map(Self::InputAudioMuted),
            "session.input_audio.unmuted" => body(object).map(Self::InputAudioUnmuted),
            "error" => body(object).map(Self::Error),
            _ => {
                return Ok(Self::Unknown(UnknownEvent {
                    kind,
                    raw: Value::Object(object),
                }));
            }
        };
        event.map_err(|error| corrupt(Some(&kind), error))
    }
}

/// An event whose `type` is not modelled, kept whole.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownEvent {
    /// The event's `type`.
    pub kind: String,
    /// The whole event object, `type` included.
    pub raw: Value,
}

/// `session.started` and `session.updated`: the resolved session.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SessionSnapshot {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// The resolved session.
    pub session: SessionResource,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The resolved configuration and metadata of a session. The configuration
/// itself (instructions, input, audio, delegation) stays in `extra` as sent.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SessionResource {
    /// The session id.
    pub id: String,
    /// The Live model.
    #[serde(default)]
    pub model: Option<String>,
    /// The session status, such as `active`.
    #[serde(default)]
    pub status: Option<String>,
    /// When the session expires, in seconds since the Unix epoch.
    #[serde(default)]
    pub expires_at: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Why a session ended.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(from = "String")]
pub enum CloseReason {
    /// `close_requested`: the application asked to close or hang up.
    CloseRequested,
    /// `expired`: the session reached its duration limit.
    Expired,
    /// `content`: a safety filter ended it.
    Content,
    /// `remote_hangup`: the remote side disconnected gracefully.
    RemoteHangup,
    /// `connection_lost`: a primary or upstream connection dropped.
    ConnectionLost,
    /// Any other reason, as sent.
    Other(String),
}

impl From<String> for CloseReason {
    fn from(reason: String) -> Self {
        match reason.as_str() {
            "close_requested" => Self::CloseRequested,
            "expired" => Self::Expired,
            "content" => Self::Content,
            "remote_hangup" => Self::RemoteHangup,
            "connection_lost" => Self::ConnectionLost,
            _ => Self::Other(reason),
        }
    }
}

/// `session.closed`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SessionClosed {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Why the session ended.
    pub reason: CloseReason,
    /// The final cumulative usage.
    pub usage: SessionUsage,
    /// The final session snapshot, when sent.
    #[serde(default)]
    pub session: Option<SessionResource>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Cumulative Live audio usage.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SessionUsage {
    /// Billed session seconds so far, silence included.
    pub seconds: f64,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.input_transcript.delta` and `session.output_transcript.delta`.
/// Fragments accumulate in delivery order; no event marks a complete turn.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TranscriptDelta {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// The transcript fragment.
    pub delta: String,
    /// Session-relative start, in milliseconds.
    pub start_ms: u64,
    /// Session-relative end, in milliseconds.
    pub end_ms: u64,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Who a delegation went to.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(from = "String")]
pub enum DelegationTarget {
    /// `client`: the caller's application.
    Client,
    /// `responses`: the session's Responses backend.
    Responses,
    /// Any other target, as sent.
    Other(String),
}

impl From<String> for DelegationTarget {
    fn from(target: String) -> Self {
        match target.as_str() {
            "client" => Self::Client,
            "responses" => Self::Responses,
            _ => Self::Other(target),
        }
    }
}

/// The work a `session.delegation.created` event names. It carries
/// metadata, not the task text.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DelegationInfo {
    /// The delegation id, which `delegation_id` fields name.
    pub id: String,
    /// Where the work went.
    pub target: DelegationTarget,
    /// The Responses response of a Responses delegation.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.delegation.created`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DelegationCreated {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Session-relative position of the delegation, in milliseconds.
    pub offset_ms: u64,
    /// The delegation.
    pub delegation: DelegationInfo,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A function call the Responses backend finished, read from a nested
/// `response.output_item.done` whose item is a `function_call`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct FunctionCall {
    /// The call id its output answers.
    pub call_id: String,
    /// The function name.
    pub name: String,
    /// The arguments, as the JSON text the backend produced.
    pub arguments: String,
    /// The output item id, when sent.
    #[serde(default)]
    pub id: Option<String>,
}

/// How a backend response ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResponseOutcome {
    /// `response.completed`.
    Completed,
    /// `response.failed`.
    Failed,
    /// `response.incomplete`.
    Incomplete,
}

/// The part of a nested Responses event this module types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendEvent {
    /// `response.created`: a backend response began.
    ResponseCreated {
        /// The response id.
        response_id: String,
    },
    /// `response.completed`, `response.failed` or `response.incomplete`.
    /// The forwarded snapshot's `output` is empty even when function calls
    /// need results; the calls come from [`Self::FunctionCallDone`].
    ResponseEnded {
        /// The response id.
        response_id: String,
        /// How it ended.
        outcome: ResponseOutcome,
    },
    /// `response.output_item.done` with a `function_call` item.
    FunctionCallDone(FunctionCall),
    /// Any other nested event, by its `type`.
    Other(String),
}

/// `response.event`: one event of the Responses backend.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseEvent {
    /// The server event id.
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    pub client_event_id: Option<String>,
    /// The delegation the nested event belongs to, when the server could
    /// correlate one.
    pub delegation_id: Option<String>,
    /// The nested event, typed as far as this module models it.
    pub backend: BackendEvent,
    /// The nested event object, exactly as decoded.
    pub event: Value,
    /// Fields not modelled here.
    pub extra: Map<String, Value>,
}

impl ResponseEvent {
    fn from_wire(wire: ResponseEventWire) -> serde_json::Result<Self> {
        let backend = BackendEvent::classify(&wire.event)?;
        Ok(Self {
            event_id: wire.event_id,
            client_event_id: wire.client_event_id,
            delegation_id: wire.delegation_id,
            backend,
            event: wire.event,
            extra: wire.extra,
        })
    }
}

#[derive(Deserialize)]
struct ResponseEventWire {
    #[serde(default)]
    event_id: Option<String>,
    #[serde(default)]
    client_event_id: Option<String>,
    #[serde(default)]
    delegation_id: Option<String>,
    event: Value,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

impl BackendEvent {
    /// Type `event`, refusing a modelled nested event whose fields do not
    /// match.
    fn classify(event: &Value) -> serde_json::Result<Self> {
        #[derive(Deserialize)]
        struct Lifecycle {
            response: ResponseId,
        }
        #[derive(Deserialize)]
        struct ResponseId {
            id: String,
        }
        #[derive(Deserialize)]
        struct ItemDone {
            item: Map<String, Value>,
        }
        let Some(kind) = event.get("type").and_then(Value::as_str) else {
            return Err(serde::de::Error::custom(
                "a nested Responses event needs a string `type`",
            ));
        };
        let response_id = || Lifecycle::deserialize(event).map(|lifecycle| lifecycle.response.id);
        let ended = |outcome| {
            response_id().map(|response_id| Self::ResponseEnded {
                response_id,
                outcome,
            })
        };
        match kind {
            "response.created" => {
                response_id().map(|response_id| Self::ResponseCreated { response_id })
            }
            "response.completed" => ended(ResponseOutcome::Completed),
            "response.failed" => ended(ResponseOutcome::Failed),
            "response.incomplete" => ended(ResponseOutcome::Incomplete),
            "response.output_item.done" => {
                let ItemDone { item } = ItemDone::deserialize(event)?;
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    serde_json::from_value(Value::Object(item)).map(Self::FunctionCallDone)
                } else {
                    Ok(Self::Other(kind.to_owned()))
                }
            }
            _ => Ok(Self::Other(kind.to_owned())),
        }
    }
}

/// `session.usage.updated`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct UsageUpdated {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event this answers, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Cumulative audio usage.
    pub usage: SessionUsage,
    /// The latest context window usage, when the limit is known.
    #[serde(default)]
    pub context_window: Option<ContextWindow>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The latest measured Live context window usage.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ContextWindow {
    /// Active context tokens divided by the model's context limit. It can
    /// fall after compaction.
    pub usage_ratio: f64,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.instructions.appended`, `session.thinking.appended` and
/// `session.commentary.appended`: the append entered the session timeline.
/// It does not mean the model acted on it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Appended {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the append, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Session-relative start, in milliseconds.
    pub start_ms: u64,
    /// Session-relative end, in milliseconds; it can equal `start_ms`.
    pub end_ms: u64,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.input_audio.muted` and `session.input_audio.unmuted`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Acknowledgement {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the command, when it had one.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `error`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ErrorEvent {
    /// The server event id.
    #[serde(default)]
    pub event_id: Option<String>,
    /// The `event_id` of the client event that caused the error, when sent
    /// beside the error object.
    #[serde(default)]
    pub client_event_id: Option<String>,
    /// The error.
    pub error: ApiErrorDetail,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ErrorEvent {
    /// The `event_id` of the client event that caused the error: the outer
    /// field, else the error object's.
    #[must_use]
    pub fn caused_by(&self) -> Option<&str> {
        self.client_event_id
            .as_deref()
            .or(self.error.client_event_id.as_deref())
    }
}

/// The text of an instructions, thinking or commentary append.
///
/// The provider limits `content` to 500 tokens and accepts a non-null
/// `delegation_id` only for a client delegation; this type checks neither.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ContextAppend {
    /// The plain text.
    pub content: String,
    /// The client delegation the text belongs to, or `None` (sent as
    /// `null`) for general session context.
    pub delegation_id: Option<String>,
    /// The client event id, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

impl ContextAppend {
    /// General session context: `delegation_id` is `null`.
    pub fn session(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delegation_id: None,
            event_id: None,
        }
    }

    /// Context for the client delegation `delegation_id`.
    pub fn for_delegation(content: impl Into<String>, delegation_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            delegation_id: Some(delegation_id.into()),
            event_id: None,
        }
    }

    /// Send `event_id` with the append.
    #[must_use]
    pub fn with_event_id(mut self, event_id: impl Into<String>) -> Self {
        self.event_id = Some(event_id.into());
        self
    }
}

/// One client event. Every `event_id` is optional; the server echoes it as
/// `client_event_id`.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    /// `session.update`: change the Responses backend settings.
    SessionUpdate {
        /// The settings to change.
        responses: ResponsesDelegationUpdate,
        /// The client event id.
        event_id: Option<String>,
    },
    /// `session.instructions.append`.
    InstructionsAppend(ContextAppend),
    /// `session.thinking.append`: silent context.
    ThinkingAppend(ContextAppend),
    /// `session.commentary.append`: speakable context.
    CommentaryAppend(ContextAppend),
    /// `response.item.create` with a `function_call_output` item.
    FunctionCallOutput {
        /// The [`FunctionCall::call_id`] answered.
        call_id: String,
        /// The output text.
        output: String,
        /// The client event id.
        event_id: Option<String>,
    },
    /// `response.create`: run or continue the Responses backend.
    ResponseCreate {
        /// The client event id.
        event_id: Option<String>,
    },
    /// `session.input_audio.mute`.
    MuteInputAudio {
        /// The client event id.
        event_id: Option<String>,
    },
    /// `session.input_audio.unmute`.
    UnmuteInputAudio {
        /// The client event id.
        event_id: Option<String>,
    },
    /// `session.close`.
    SessionClose {
        /// The client event id.
        event_id: Option<String>,
    },
}

impl Serialize for ClientEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct SessionUpdate<'a> {
            delegation: DelegationUpdate<'a>,
        }
        #[derive(Serialize)]
        struct DelegationUpdate<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            responses: &'a ResponsesDelegationUpdate,
        }
        #[derive(Serialize)]
        struct OutputItem<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            call_id: &'a str,
            output: &'a str,
        }
        #[derive(Serialize)]
        #[serde(tag = "type")]
        enum Wire<'a> {
            #[serde(rename = "session.update")]
            SessionUpdate {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
                session: SessionUpdate<'a>,
            },
            #[serde(rename = "session.instructions.append")]
            InstructionsAppend(&'a ContextAppend),
            #[serde(rename = "session.thinking.append")]
            ThinkingAppend(&'a ContextAppend),
            #[serde(rename = "session.commentary.append")]
            CommentaryAppend(&'a ContextAppend),
            #[serde(rename = "response.item.create")]
            ItemCreate {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
                item: OutputItem<'a>,
            },
            #[serde(rename = "response.create")]
            ResponseCreate {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
            },
            #[serde(rename = "session.input_audio.mute")]
            Mute {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
            },
            #[serde(rename = "session.input_audio.unmute")]
            Unmute {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
            },
            #[serde(rename = "session.close")]
            Close {
                #[serde(skip_serializing_if = "Option::is_none")]
                event_id: Option<&'a str>,
            },
        }
        match self {
            Self::SessionUpdate {
                responses,
                event_id,
            } => Wire::SessionUpdate {
                event_id: event_id.as_deref(),
                session: SessionUpdate {
                    delegation: DelegationUpdate {
                        kind: "responses",
                        responses,
                    },
                },
            },
            Self::InstructionsAppend(append) => Wire::InstructionsAppend(append),
            Self::ThinkingAppend(append) => Wire::ThinkingAppend(append),
            Self::CommentaryAppend(append) => Wire::CommentaryAppend(append),
            Self::FunctionCallOutput {
                call_id,
                output,
                event_id,
            } => Wire::ItemCreate {
                event_id: event_id.as_deref(),
                item: OutputItem {
                    kind: "function_call_output",
                    call_id,
                    output,
                },
            },
            Self::ResponseCreate { event_id } => Wire::ResponseCreate {
                event_id: event_id.as_deref(),
            },
            Self::MuteInputAudio { event_id } => Wire::Mute {
                event_id: event_id.as_deref(),
            },
            Self::UnmuteInputAudio { event_id } => Wire::Unmute {
                event_id: event_id.as_deref(),
            },
            Self::SessionClose { event_id } => Wire::Close {
                event_id: event_id.as_deref(),
            },
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests;
