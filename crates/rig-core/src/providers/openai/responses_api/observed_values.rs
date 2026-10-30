//! Nominal metadata values for the observed Responses representations.
//!
//! These values retain their provider wire spellings and optional metadata.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Nonempty JSON object of provider-specific content metadata, serialized as
/// an object under a named `additional_params` field rather than flattened.
/// Constructors return `None` for empty maps. Bare deserialization rejects
/// empty or non-object values; [`optional_additional_params`] maps null and
/// empty objects to absence. Providers must replay only their own metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AdditionalParams(serde_json::Map<String, serde_json::Value>);

impl AdditionalParams {
    /// The canonical constructor: `None` when the map is empty.
    pub fn new(map: serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        if map.is_empty() {
            None
        } else {
            Some(Self(map))
        }
    }

    /// Build from `(key, value)` entries; `None` when the iterator yields
    /// none. `Option<(K, Value)>` is such an iterator, so a conditional
    /// single-key params reads as
    /// `AdditionalParams::from_entries(guard.then(|| (key, value)))`.
    pub fn from_entries<K, I>(entries: I) -> Option<Self>
    where
        K: Into<String>,
        I: IntoIterator<Item = (K, serde_json::Value)>,
    {
        Self::new(
            entries
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// The value stored under `key`, when present.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// The params as a bare JSON object value.
    pub fn into_value(self) -> serde_json::Value {
        serde_json::Value::Object(self.0)
    }

    /// Returns the object under the provider's own key, or `None` for absent
    /// or non-object values. Use [`Self::get`] to diagnose malformed values.
    pub fn wire_extras(
        &self,
        wire_key: &str,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.0.get(wire_key).and_then(serde_json::Value::as_object)
    }

    /// Returns absence for null or empty objects, metadata for nonempty objects,
    /// or the original value as an error for other shapes.
    pub fn try_from_value(value: serde_json::Value) -> Result<Option<Self>, serde_json::Value> {
        match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Object(map) => Ok(Self::new(map)),
            other => Err(other),
        }
    }
}

impl<'de> Deserialize<'de> for AdditionalParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match Self::try_from_value(serde_json::Value::deserialize(deserializer)?) {
            Ok(Some(params)) => Ok(params),
            // `null` and `{}` canonicalize to absence, which a bare
            // (non-`Option`) slot cannot express.
            Ok(None) => Err(serde::de::Error::custom(
                "`additional_params` carries no data — omit the field (an `Option` \
                 field routed through `optional_additional_params` canonicalizes \
                 `{}` and `null` to absent)",
            )),
            Err(_) => Err(serde::de::Error::custom(
                "`additional_params` must be a non-empty JSON object",
            )),
        }
    }
}

/// Deserializes optional metadata, mapping null and empty objects to `None`.
/// Nonempty objects produce metadata; other shapes return a deserialization error.
pub fn optional_additional_params<'de, D>(
    deserializer: D,
) -> Result<Option<AdditionalParams>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(value) => AdditionalParams::try_from_value(value).map_err(|_| {
            serde::de::Error::custom("`additional_params` must be a JSON object (or null)")
        }),
    }
}

/// A response object as an enum (ensures type validation)
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseObject {
    Response,
}

/// The response status as an enum (ensures type validation)
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Incomplete,
    /// A provider-specific status added after this client was released.
    Other(String),
}

/// The wire spelling of a [`ResponseStatus`].
///
/// Statuses outside the normalized finish-reason vocabulary are carried through
/// as [`super::observed_types::FinishReason::Other`], so they must keep OpenAI's own
/// spelling rather than a Rust `Debug` name.
fn response_status_wire_name(status: &ResponseStatus) -> &str {
    match status {
        ResponseStatus::InProgress => "in_progress",
        ResponseStatus::Completed => "completed",
        ResponseStatus::Failed => "failed",
        ResponseStatus::Cancelled => "cancelled",
        ResponseStatus::Queued => "queued",
        ResponseStatus::Incomplete => "incomplete",
        ResponseStatus::Other(status) => status,
    }
}

impl Serialize for ResponseStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(response_status_wire_name(self))
    }
}

impl<'de> Deserialize<'de> for ResponseStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            "queued" => Self::Queued,
            "incomplete" => Self::Incomplete,
            other => Self::Other(other.to_owned()),
        })
    }
}

/// The truncation strategy.
/// When using auto, if the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
/// Otherwise, does nothing (and is disabled by default).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    Auto,
    #[default]
    Disabled,
}

/// Echoed reasoning configuration of an observed native response.
///
/// # Example
/// ```
/// use rig_core::providers::openai::responses_api::observed_values::{
///     Reasoning, ReasoningContext, ReasoningEffort, ReasoningMode,
/// };
///
/// // GPT-5.6 reasoning controls: effort, pro mode, and persisted-reasoning context.
/// let reasoning = Reasoning::new()
///     .with_effort(ReasoningEffort::Max)
///     .with_mode(ReasoningMode::Pro)
///     .with_context(ReasoningContext::AllTurns);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort you want the model to put into thinking/reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// How much effort you want the model to put into writing the reasoning summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryLevel>,
    /// The reasoning mode. Independent from `effort`; the standard mode is
    /// represented by omitting the field. Supported by the GPT-5.6 model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ReasoningMode>,
    /// How persisted reasoning is carried across turns. Supported by the
    /// GPT-5.6 model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
}

impl Reasoning {
    /// Creates a new Reasoning instantiation (with empty values).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds reasoning effort.
    pub fn with_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.effort = Some(reasoning_effort);

        self
    }

    /// Adds summary level (how detailed the reasoning summary will be).
    pub fn with_summary_level(mut self, reasoning_summary_level: ReasoningSummaryLevel) -> Self {
        self.summary = Some(reasoning_summary_level);

        self
    }

    /// Sets the reasoning mode (e.g. pro mode on GPT-5.6 models).
    pub fn with_mode(mut self, reasoning_mode: ReasoningMode) -> Self {
        self.mode = Some(reasoning_mode);

        self
    }

    /// Sets how persisted reasoning is carried across turns (GPT-5.6 models).
    pub fn with_context(mut self, reasoning_context: ReasoningContext) -> Self {
        self.context = Some(reasoning_context);

        self
    }
}

/// The billing service tier that will be used. On auto by default.
#[derive(Clone, Debug, Default)]
pub enum OpenAIServiceTier {
    /// Let OpenAI choose the service tier.
    #[default]
    Auto,
    /// Use the default service tier.
    Default,
    /// Use the flex service tier.
    Flex,
    /// Use the priority service tier.
    Priority,
    /// Use the standard service tier returned by OpenAI-compatible providers.
    Standard,
    /// Preserve an unknown provider-specific service tier.
    Other(String),
}

impl Serialize for OpenAIServiceTier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
            Self::Priority => "priority",
            Self::Standard => "standard",
            Self::Other(value) => value,
        })
    }
}

impl<'de> Deserialize<'de> for OpenAIServiceTier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "auto" => Self::Auto,
            "default" => Self::Default,
            "flex" => Self::Flex,
            "priority" => Self::Priority,
            "standard" => Self::Standard,
            _ => Self::Other(value),
        })
    }
}

/// The amount of reasoning effort that will be used by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
    /// The highest reasoning effort. Supported by the GPT-5.6 model family.
    Max,
}

/// The reasoning mode used by a given model. Independent from
/// [`ReasoningEffort`]; the standard mode is represented by omitting the field
/// (`None` on [`Reasoning::mode`]), so this enum only carries the documented
/// non-default modes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Pro mode. Supported by the GPT-5.6 model family.
    Pro,
}

/// How persisted reasoning is carried across turns. Supported by the GPT-5.6
/// model family.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    /// Let the model decide how much persisted reasoning to reuse.
    #[default]
    Auto,
    /// Reuse persisted reasoning from all previous turns.
    AllTurns,
    /// Only use reasoning from the current turn.
    CurrentTurn,
}

/// The amount of effort that will go into a reasoning summary by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummaryLevel {
    #[default]
    Auto,
    Concise,
    Detailed,
}

/// The status of a given tool.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// The role of an output message.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputRole {
    Assistant,
}
