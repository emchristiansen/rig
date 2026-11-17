//! The session object a GPT-Live call is created with.

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use super::GPT_LIVE_1_CODEX;

/// The voice GPT-Live speaks with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Voice {
    /// `juniper`
    Juniper,
    /// `maple`
    Maple,
    /// `spruce`
    Spruce,
    /// `ember`
    Ember,
    /// `vale`
    Vale,
    /// `breeze`
    Breeze,
    /// `arbor`
    Arbor,
    /// `sol`
    Sol,
    /// `cove`, the default.
    #[default]
    Cove,
}

/// Who wrote an [`InitialItem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum InitialItemRole {
    Developer,
    User,
    Assistant,
}

/// One message the session starts with. Its content type follows from its
/// role: `output_text` for an assistant message, `input_text` otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitialItem {
    role: InitialItemRole,
    text: String,
}

impl InitialItem {
    /// A developer message.
    pub fn developer(text: impl Into<String>) -> Self {
        Self {
            role: InitialItemRole::Developer,
            text: text.into(),
        }
    }

    /// A user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: InitialItemRole::User,
            text: text.into(),
        }
    }

    /// An assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: InitialItemRole::Assistant,
            text: text.into(),
        }
    }

    /// The message text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    fn content_type(&self) -> &'static str {
        match self.role {
            InitialItemRole::Developer | InitialItemRole::User => "input_text",
            InitialItemRole::Assistant => "output_text",
        }
    }
}

impl Serialize for InitialItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Content<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            text: &'a str,
        }
        let mut item = serializer.serialize_struct("InitialItem", 3)?;
        item.serialize_field("type", "message")?;
        item.serialize_field("role", &self.role)?;
        item.serialize_field(
            "content",
            &[Content {
                kind: self.content_type(),
                text: &self.text,
            }],
        )?;
        item.end()
    }
}

/// The `session` object sent with call creation: the model, its
/// instructions, its voice, client delegation and the initial items.
///
/// Delegation is always `client`: GPT-Live hands questions it cannot answer
/// to the caller as [`DelegationCreated`](super::DelegationCreated) events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    /// The model id, [`GPT_LIVE_1_CODEX`] unless overridden.
    pub model: String,
    /// The GPT-Live instructions.
    pub instructions: String,
    /// The output voice.
    pub voice: Voice,
    /// Whether GPT-Live speaks a short filler while a delegation is pending.
    /// `None` sends no `ack_filler` field.
    pub delegation_ack_filler: Option<bool>,
    /// Messages the session starts with, in order. None are sent when empty.
    pub initial_items: Vec<InitialItem>,
}

impl SessionConfig {
    /// A [`GPT_LIVE_1_CODEX`] session with `instructions`, the default voice,
    /// no ack filler setting and no initial items.
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            model: GPT_LIVE_1_CODEX.to_owned(),
            instructions: instructions.into(),
            voice: Voice::default(),
            delegation_ack_filler: None,
            initial_items: Vec::new(),
        }
    }

    /// Use `model` instead of [`GPT_LIVE_1_CODEX`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Speak with `voice`.
    #[must_use]
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = voice;
        self
    }

    /// Send `delegation.ack_filler` as `ack_filler`.
    #[must_use]
    pub fn with_delegation_ack_filler(mut self, ack_filler: bool) -> Self {
        self.delegation_ack_filler = Some(ack_filler);
        self
    }

    /// Append `item` to the initial items.
    #[must_use]
    pub fn with_initial_item(mut self, item: InitialItem) -> Self {
        self.initial_items.push(item);
        self
    }
}

impl Serialize for SessionConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Audio {
            output: AudioOutput,
        }
        #[derive(Serialize)]
        struct AudioOutput {
            voice: Voice,
        }
        #[derive(Serialize)]
        struct Delegation {
            #[serde(rename = "type")]
            kind: &'static str,
            #[serde(skip_serializing_if = "Option::is_none")]
            ack_filler: Option<bool>,
        }

        let fields = if self.initial_items.is_empty() { 4 } else { 5 };
        let mut session = serializer.serialize_struct("SessionConfig", fields)?;
        session.serialize_field("model", &self.model)?;
        session.serialize_field("instructions", &self.instructions)?;
        session.serialize_field(
            "audio",
            &Audio {
                output: AudioOutput { voice: self.voice },
            },
        )?;
        session.serialize_field(
            "delegation",
            &Delegation {
                kind: "client",
                ack_filler: self.delegation_ack_filler,
            },
        )?;
        if !self.initial_items.is_empty() {
            session.serialize_field("initial_items", &self.initial_items)?;
        }
        session.end()
    }
}

#[cfg(test)]
mod tests;
