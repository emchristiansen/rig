//! The events of a GPT-Live call's control socket and event data channel.
//!
//! Server events decode by their `type`. A type this module does not model
//! decodes as [`ServerEvent::Unknown`] with its whole JSON object, because the
//! protocol is alpha and grows. Every modelled object keeps the fields it does
//! not model in `extra`. Client events are built already split into
//! [`ContextChunk`]s of at most [`CONTEXT_APPEND_MAX_BYTES`] bytes.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use super::CONTEXT_APPEND_MAX_BYTES;
use crate::error::{CorruptFrame, ProviderError};

/// One server event.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerEvent {
    /// `session.started`: the call's session is live.
    SessionStarted(SessionStarted),
    /// `session.input_audio.append`: the caller's audio as the server received it.
    InputAudioAppend(InputAudioAppend),
    /// `session.output_audio.delta`: GPT-Live's audio.
    OutputAudioDelta(OutputAudioDelta),
    /// `input_transcript.added`: a fragment of the caller's transcript.
    InputTranscriptAdded(TranscriptAdded),
    /// `output_transcript.added`: a fragment of GPT-Live's transcript.
    OutputTranscriptAdded(TranscriptAdded),
    /// `turn.created`: a turn began.
    TurnCreated(TurnEvent),
    /// `turn.delta`: more text of an open turn.
    TurnDelta(TurnDelta),
    /// `turn.done`: a turn ended, with its final transcript.
    TurnDone(TurnEvent),
    /// `delegation.created`: GPT-Live hands a question to the caller.
    DelegationCreated(DelegationCreated),
    /// `delegation.context.appended`: GPT-Live took context the caller appended.
    DelegationContextAppended(DelegationContextAppended),
    /// `session.usage.updated`: cumulative usage and the usage limit.
    UsageUpdated(UsageUpdated),
    /// `error`: the server reports a failure.
    Error(ErrorEvent),
    /// A `type` this module does not model.
    Unknown(UnknownEvent),
}

impl ServerEvent {
    /// Decode one event payload.
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
        let mut object: Map<String, Value> =
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
            "session.input_audio.append" => body(object).map(Self::InputAudioAppend),
            "session.output_audio.delta" => body(object).map(Self::OutputAudioDelta),
            "input_transcript.added" => body(object).map(Self::InputTranscriptAdded),
            "output_transcript.added" => body(object).map(Self::OutputTranscriptAdded),
            "turn.created" => body(object).map(Self::TurnCreated),
            "turn.delta" => body(object).map(Self::TurnDelta),
            "turn.done" => body(object).map(Self::TurnDone),
            "delegation.created" => body(object).map(Self::DelegationCreated),
            "delegation.context.appended" => body(object).map(Self::DelegationContextAppended),
            "session.usage.updated" => body(object).map(Self::UsageUpdated),
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

/// `session.started`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SessionStarted {
    /// The started session.
    pub session: StartedSession,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The session a `session.started` event names.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct StartedSession {
    /// The session id, which is the call id.
    pub id: String,
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

/// `session.input_audio.append`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct InputAudioAppend {
    /// Base64 audio.
    pub audio: String,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.output_audio.delta`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct OutputAudioDelta {
    /// Base64 audio.
    pub delta: String,
    /// Session-relative start, in milliseconds.
    #[serde(default)]
    pub start_ms: Option<u64>,
    /// Session-relative end, in milliseconds.
    #[serde(default)]
    pub end_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `input_transcript.added` and `output_transcript.added`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TranscriptAdded {
    /// The transcript fragment.
    pub item: TranscriptItem,
    /// Session-relative start, in milliseconds.
    #[serde(default)]
    pub start_ms: Option<u64>,
    /// Session-relative end, in milliseconds.
    #[serde(default)]
    pub end_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A transcript fragment.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TranscriptItem {
    /// The item id.
    pub id: String,
    /// The item type, such as `input_transcript`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The fragment's text, with its leading space when it has one.
    pub text: String,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Who speaks in a turn.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(from = "String")]
pub enum Role {
    /// `user`, the caller.
    User,
    /// `assistant`, GPT-Live.
    Assistant,
    /// Any other role, as sent.
    Other(String),
}

impl From<String> for Role {
    fn from(role: String) -> Self {
        match role.as_str() {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            _ => Self::Other(role),
        }
    }
}

/// `turn.created` and `turn.done`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TurnEvent {
    /// The turn.
    pub turn: Turn,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One turn of the conversation.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Turn {
    /// The turn id.
    pub id: String,
    /// Who speaks.
    pub role: Role,
    /// The transcript so far; final in `turn.done`.
    pub transcript: String,
    /// Session-relative start, in milliseconds.
    #[serde(default)]
    pub start_ms: Option<u64>,
    /// Session-relative end, in milliseconds.
    #[serde(default)]
    pub end_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `turn.delta`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TurnDelta {
    /// The turn the text belongs to.
    pub turn_id: String,
    /// The new text.
    pub delta: String,
    /// Session-relative start, in milliseconds.
    #[serde(default)]
    pub start_ms: Option<u64>,
    /// Session-relative end, in milliseconds.
    #[serde(default)]
    pub end_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `delegation.created`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DelegationCreated {
    /// The delegation.
    pub item: DelegationItem,
    /// Session-relative offset, in milliseconds.
    #[serde(default)]
    pub offset_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A question GPT-Live hands to a delegation target.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DelegationItem {
    /// The item id, which a [`ClientEvent::DelegationContextAppend`] names.
    pub id: String,
    /// The item type, `delegation`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Who answers it; `client` for a client-delegated session.
    pub target: String,
    /// The question.
    pub content: Vec<ContentPart>,
    /// The handoff id.
    #[serde(default)]
    pub handoff_id: Option<String>,
    /// The caller turn the question came from.
    #[serde(default)]
    pub user_bidi_turn_id: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl DelegationItem {
    /// The question: the text of every `input_text` part, concatenated.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter(|part| part.kind == "input_text")
            .map(|part| part.text.as_str())
            .collect()
    }
}

/// One part of a delegation's content.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ContentPart {
    /// The part type, such as `input_text`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The part's text.
    pub text: String,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `delegation.context.appended`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DelegationContextAppended {
    /// The delegation the context was appended to.
    pub delegation_item_id: String,
    /// Session-relative start, in milliseconds.
    #[serde(default)]
    pub start_ms: Option<u64>,
    /// Session-relative end, in milliseconds.
    #[serde(default)]
    pub end_ms: Option<u64>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `session.usage.updated`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct UsageUpdated {
    /// Cumulative usage.
    pub usage: Usage,
    /// The account's usage limit, when reported.
    #[serde(default)]
    pub usage_limit: Option<UsageLimit>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Cumulative usage of a session.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Usage {
    /// Cumulative billed audio, in milliseconds. It grows with session time,
    /// silence included.
    pub audio_duration_ms: u64,
    /// Backend model usage entries, as sent.
    #[serde(default)]
    pub backend_model_usage: Vec<Value>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The usage limit a `session.usage.updated` event reports. Both fields have
/// only been observed as `null`, so they are kept as sent.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct UsageLimit {
    /// The limit status.
    #[serde(default)]
    pub status: Option<Value>,
    /// Seconds until the limit resets.
    #[serde(default)]
    pub reset_seconds: Option<Value>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `error`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ErrorEvent {
    /// The error object, when sent.
    #[serde(default)]
    pub error: Option<ErrorDetail>,
    /// A top-level message, when sent.
    #[serde(default)]
    pub message: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ErrorEvent {
    /// The top-level message, else the error object's message.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        self.message
            .as_deref()
            .or_else(|| self.error.as_ref()?.message.as_deref())
    }
}

/// The error object of an `error` event.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ErrorDetail {
    /// A machine-readable code.
    #[serde(default)]
    pub code: Option<String>,
    /// A human-readable message.
    #[serde(default)]
    pub message: Option<String>,
    /// The error category.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// The parameter at fault.
    #[serde(default)]
    pub param: Option<String>,
    /// Fields not modelled here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The stream a context append goes to. Without one, the server's default
/// applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextChannel {
    /// `speakable`: GPT-Live may speak the text.
    Speakable,
    /// `commentary`: background context.
    Commentary,
}

/// At most [`CONTEXT_APPEND_MAX_BYTES`] bytes of text, cut on a character
/// boundary; built only by [`context_chunks`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextChunk(String);

impl ContextChunk {
    /// The chunk's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Split `text` into chunks of at most [`CONTEXT_APPEND_MAX_BYTES`] bytes,
/// each ending on a character boundary, in order. Text that fits, the empty
/// text included, is one chunk.
#[must_use]
pub fn context_chunks(text: &str) -> Vec<ContextChunk> {
    let mut chunks = Vec::new();
    let mut rest = text;
    loop {
        let mut end = rest.len().min(CONTEXT_APPEND_MAX_BYTES);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        let (chunk, tail) = rest.split_at(end);
        chunks.push(ContextChunk(chunk.to_owned()));
        if tail.is_empty() {
            return chunks;
        }
        rest = tail;
    }
}

/// One client event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientEvent {
    /// `delegation.context.append`: answer a delegation.
    DelegationContextAppend {
        /// The [`DelegationItem::id`] answered.
        delegation_item_id: String,
        /// Where the text goes.
        channel: Option<ContextChannel>,
        /// The text.
        text: ContextChunk,
    },
    /// `session.context.append`: add context to the session.
    SessionContextAppend {
        /// Where the text goes.
        channel: Option<ContextChannel>,
        /// The text.
        text: ContextChunk,
    },
    /// `session.close`: end the session.
    SessionClose,
}

impl ClientEvent {
    /// The `delegation.context.append` events carrying `text` for the
    /// delegation `delegation_item_id`, one per [`context_chunks`] chunk.
    pub fn delegation_context_append(
        delegation_item_id: &str,
        channel: Option<ContextChannel>,
        text: &str,
    ) -> Vec<Self> {
        context_chunks(text)
            .into_iter()
            .map(|text| Self::DelegationContextAppend {
                delegation_item_id: delegation_item_id.to_owned(),
                channel,
                text,
            })
            .collect()
    }

    /// The `session.context.append` events carrying `text`, one per
    /// [`context_chunks`] chunk.
    pub fn session_context_append(channel: Option<ContextChannel>, text: &str) -> Vec<Self> {
        context_chunks(text)
            .into_iter()
            .map(|text| Self::SessionContextAppend { channel, text })
            .collect()
    }
}

impl Serialize for ClientEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct InputText<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            text: &'a str,
        }
        #[derive(Serialize)]
        #[serde(tag = "type")]
        enum Wire<'a> {
            #[serde(rename = "delegation.context.append")]
            DelegationContextAppend {
                delegation_item_id: &'a str,
                #[serde(skip_serializing_if = "Option::is_none")]
                channel: Option<ContextChannel>,
                content: [InputText<'a>; 1],
            },
            #[serde(rename = "session.context.append")]
            SessionContextAppend {
                #[serde(skip_serializing_if = "Option::is_none")]
                channel: Option<ContextChannel>,
                content: [InputText<'a>; 1],
            },
            #[serde(rename = "session.close")]
            SessionClose,
        }
        let content = |text: &'_ ContextChunk| {
            [InputText {
                kind: "input_text",
                text: text.as_str(),
            }]
        };
        match self {
            Self::DelegationContextAppend {
                delegation_item_id,
                channel,
                text,
            } => Wire::DelegationContextAppend {
                delegation_item_id,
                channel: *channel,
                content: content(text),
            },
            Self::SessionContextAppend { channel, text } => Wire::SessionContextAppend {
                channel: *channel,
                content: content(text),
            },
            Self::SessionClose => Wire::SessionClose,
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests;
