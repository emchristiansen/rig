//! The startup configuration of a Live session and the Responses delegation
//! settings it carries.

use serde::{Serialize, Serializer};
use serde_json::Value;

use super::GPT_LIVE_1;
use crate::completion::ToolDefinition;

/// A built-in voice the Live model speaks with. The server default is
/// [`Voice::Marin`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// `alloy`
    Alloy,
    /// `ash`
    Ash,
    /// `ballad`
    Ballad,
    /// `beacon`
    Beacon,
    /// `bossa`
    Bossa,
    /// `cedar`
    Cedar,
    /// `cinder`
    Cinder,
    /// `coral`
    Coral,
    /// `delta`
    Delta,
    /// `echo`
    Echo,
    /// `gleam`
    Gleam,
    /// `marin`
    Marin,
    /// `meridian`
    Meridian,
    /// `quartz`
    Quartz,
    /// `ripple`
    Ripple,
    /// `sage`
    Sage,
    /// `shimmer`
    Shimmer,
    /// `stone`
    Stone,
    /// `tempo`
    Tempo,
    /// `verse`
    Verse,
    /// `vesper`
    Vesper,
    /// `willow`
    Willow,
}

/// Who wrote an [`InitialItem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum InitialItemRole {
    Developer,
    User,
    Assistant,
}

/// One text message of the history a session starts with. Its content part
/// type follows from its role: `output_text` for an assistant message,
/// `input_text` otherwise.
///
/// The provider accepts at most 128 messages and 8,192 rendered tokens in
/// total; this type does not count them.
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
}

impl Serialize for InitialItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Part<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            text: &'a str,
        }
        #[derive(Serialize)]
        struct Wire<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            role: InitialItemRole,
            content: [Part<'a>; 1],
        }
        let part_kind = match self.role {
            InitialItemRole::Developer | InitialItemRole::User => "input_text",
            InitialItemRole::Assistant => "output_text",
        };
        Wire {
            kind: "message",
            role: self.role,
            content: [Part {
                kind: part_kind,
                text: &self.text,
            }],
        }
        .serialize(serializer)
    }
}

/// How much reasoning effort the delegated Responses model uses. Which
/// values a backend model supports depends on that model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// `none`
    None,
    /// `minimal`
    Minimal,
    /// `low`
    Low,
    /// `medium`
    Medium,
    /// `high`
    High,
    /// `xhigh`
    Xhigh,
}

/// The reasoning summary requested from the delegated Responses model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// `concise`
    Concise,
    /// `detailed`
    Detailed,
    /// `auto`
    Auto,
}

/// Reasoning settings passed to each delegated Responses request. An unset
/// field is not sent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct Reasoning {
    /// The reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// The reasoning summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// The service tier of delegated Responses requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    /// `auto`
    Auto,
    /// `default`
    Default,
    /// `fast_tier_temp_pilot`
    FastTierTempPilot,
    /// `flex`
    Flex,
    /// `priority`, which selects Fast mode where the model and project have it.
    Priority,
    /// `ultrafast`
    Ultrafast,
}

/// The amount of detail in text the Responses backend generates. It does
/// not configure the Live model's spoken delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    /// `low`
    Low,
    /// `medium`
    Medium,
    /// `high`
    High,
}

/// Text settings passed to each delegated Responses request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct TextSettings {
    /// The text verbosity. `None` sends no `verbosity` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
}

/// Which tool the Responses backend uses for a delegated task.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ToolChoice {
    /// `auto`: the backend chooses.
    Auto,
    /// `none`: no tool calls.
    None,
    /// `required`: at least one tool call.
    Required,
    /// `{"type": "function", "name": ...}`: call the named function.
    Function {
        /// The function name.
        name: String,
    },
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Function<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            name: &'a str,
        }
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => Function {
                kind: "function",
                name,
            }
            .serialize(serializer),
        }
    }
}

/// A function the Responses backend may call. The caller executes it and
/// returns its output with
/// [`PendingFunctionCalls::submit`](super::PendingFunctionCalls::submit).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionTool {
    /// The name the backend calls the function by.
    pub name: String,
    /// What the function does and when to call it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema of the function's arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Whether the backend must follow `parameters` exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl From<ToolDefinition> for FunctionTool {
    fn from(tool: ToolDefinition) -> Self {
        Self {
            name: tool.name,
            description: Some(tool.description),
            parameters: Some(tool.parameters),
            strict: None,
        }
    }
}

/// A tool available to the Responses backend.
#[derive(Clone, Debug, PartialEq)]
pub enum DelegationTool {
    /// A function the caller executes.
    Function(FunctionTool),
    /// `{"type": "web_search"}`: the provider's web search.
    WebSearch,
}

impl From<FunctionTool> for DelegationTool {
    fn from(tool: FunctionTool) -> Self {
        Self::Function(tool)
    }
}

impl Serialize for DelegationTool {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Wire<'a> {
            Function(&'a FunctionTool),
            WebSearch,
        }
        match self {
            Self::Function(tool) => Wire::Function(tool),
            Self::WebSearch => Wire::WebSearch,
        }
        .serialize(serializer)
    }
}

/// A per-response output token cap of at least
/// [`MaxOutputTokens::MIN`], the provider's minimum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct MaxOutputTokens(u32);

/// A token cap below [`MaxOutputTokens::MIN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("max_output_tokens must be at least {min}; got {0}", min = MaxOutputTokens::MIN)]
pub struct InvalidMaxOutputTokens(pub u32);

impl MaxOutputTokens {
    /// The smallest cap the provider accepts.
    pub const MIN: u32 = 16;

    /// Accept `tokens` when it is at least [`Self::MIN`].
    pub fn new(tokens: u32) -> Result<Self, InvalidMaxOutputTokens> {
        if tokens >= Self::MIN {
            Ok(Self(tokens))
        } else {
            Err(InvalidMaxOutputTokens(tokens))
        }
    }

    /// The cap.
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The Responses settings other than the backend model. Every unset field
/// is omitted from the wire, so the server default applies at creation and
/// the current value is kept by an update.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResponsesSettings {
    /// Backend instructions, separate from the Live instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The output token cap of each delegated response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<MaxOutputTokens>,
    /// Whether one delegated response may request several tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// The service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Text settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextSettings>,
    /// The tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// The tools, in order. `Some(vec![])` sends an empty list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<DelegationTool>>,
}

/// The backend a Responses-delegated session uses: a model and its settings.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResponsesDelegation {
    /// The backend model, such as `gpt-6-luna`.
    pub model: String,
    /// The other settings.
    #[serde(flatten)]
    pub settings: ResponsesSettings,
}

impl ResponsesDelegation {
    /// Delegate to `model` with the server's default settings.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            settings: ResponsesSettings::default(),
        }
    }

    /// Use `settings`.
    #[must_use]
    pub fn with_settings(mut self, settings: ResponsesSettings) -> Self {
        self.settings = settings;
        self
    }
}

/// A sparse change to a running session's Responses backend, sent with
/// [`ClientEvent::SessionUpdate`](super::ClientEvent::SessionUpdate). Unset
/// fields keep their current values.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResponsesDelegationUpdate {
    /// A new backend model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The other settings to change.
    #[serde(flatten)]
    pub settings: ResponsesSettings,
}

/// Who handles the work the Live model delegates. Fixed at startup.
#[derive(Clone, Debug, PartialEq)]
pub enum Delegation {
    /// `{"type": "client"}`: the caller's application handles it.
    Client,
    /// `{"type": "responses", "responses": ...}`: a Responses backend the
    /// session manages handles it.
    Responses(ResponsesDelegation),
}

impl Serialize for Delegation {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Wire<'a> {
            Client,
            Responses { responses: &'a ResponsesDelegation },
        }
        match self {
            Self::Client => Wire::Client,
            Self::Responses(responses) => Wire::Responses { responses },
        }
        .serialize(serializer)
    }
}

/// The `session` object a Live session is created with. Unset optional
/// fields are omitted, so the server defaults apply.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    /// The Live model, [`GPT_LIVE_1`] unless overridden.
    pub model: String,
    /// The Live model's instructions. The provider limits them to 16,384
    /// tokens and uses its defaults when they are blank.
    pub instructions: Option<String>,
    /// The output voice, sent as `audio.output.voice`.
    pub voice: Option<Voice>,
    /// The history the session starts with, in order, sent as `input`.
    /// None is sent when empty.
    pub input: Vec<InitialItem>,
    /// Who handles delegated work. `None` sends no `delegation`, which the
    /// provider treats as client delegation.
    pub delegation: Option<Delegation>,
    /// Whether the provider stores the session for forking and recording
    /// download.
    pub store: Option<bool>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionConfig {
    /// A [`GPT_LIVE_1`] session with every other setting at its server
    /// default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: GPT_LIVE_1.to_owned(),
            instructions: None,
            voice: None,
            input: Vec::new(),
            delegation: None,
            store: None,
        }
    }

    /// Use `model` instead of [`GPT_LIVE_1`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Send `instructions`.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Speak with `voice`.
    #[must_use]
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Append `item` to the initial history.
    #[must_use]
    pub fn with_input_item(mut self, item: InitialItem) -> Self {
        self.input.push(item);
        self
    }

    /// Delegate with `delegation`.
    #[must_use]
    pub fn with_delegation(mut self, delegation: Delegation) -> Self {
        self.delegation = Some(delegation);
        self
    }

    /// Send `store`.
    #[must_use]
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
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
        struct Wire<'a> {
            model: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            instructions: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            audio: Option<Audio>,
            #[serde(skip_serializing_if = "Option::is_none")]
            input: Option<&'a [InitialItem]>,
            #[serde(skip_serializing_if = "Option::is_none")]
            delegation: Option<&'a Delegation>,
            #[serde(skip_serializing_if = "Option::is_none")]
            store: Option<bool>,
        }
        Wire {
            model: &self.model,
            instructions: self.instructions.as_deref(),
            audio: self.voice.map(|voice| Audio {
                output: AudioOutput { voice },
            }),
            input: (!self.input.is_empty()).then_some(self.input.as_slice()),
            delegation: self.delegation.as_ref(),
            store: self.store,
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests;
