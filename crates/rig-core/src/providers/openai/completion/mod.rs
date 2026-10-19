//! OpenAI completion model identifiers and retained Chat response types.
//!
//! The response types also read recorded Chat Completions documents.

/// GPT-6 Astra, API ID `gpt-6-astra`: a reasoning model that cannot turn
/// reasoning off. Its catalog entry holds the rules the encoders check,
/// through `on_unsupported`, for a request that sets generation options:
/// no sampling parameters, and tools only through Responses.
pub const GPT_6_ASTRA: &str = "gpt-6-astra";

/// GPT-6.1 Sol, API ID `gpt-6.1-sol`: a reasoning model that cannot turn
/// reasoning off. Its catalog entry holds the rules the encoders check,
/// through `on_unsupported`, for a request that sets generation options:
/// no sampling parameters, and tools only through Responses.
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";

/// GPT-6 Sol, API ID `gpt-6-sol`: a reasoning model. Its catalog entry
/// holds the rules the encoders check, through `on_unsupported`, for a
/// request that sets generation options: sampling parameters only at effort
/// `none`
/// ([`GenerationOptions::reasoning`](crate::completion::GenerationOptions::reasoning)
/// `(Reasoning::Off)`). OpenAI itself refuses Chat Completions tools at
/// any other effort; Responses takes tools at any effort.
pub const GPT_6_SOL: &str = "gpt-6-sol";

/// GPT-6 Luna, API ID `gpt-6-luna`: a reasoning model. Its catalog entry
/// holds the rules the encoders check, through `on_unsupported`, for a
/// request that sets generation options: sampling parameters only at effort
/// `none`
/// ([`GenerationOptions::reasoning`](crate::completion::GenerationOptions::reasoning)
/// `(Reasoning::Off)`). OpenAI itself refuses Chat Completions tools at
/// any other effort; Responses takes tools at any effort.
pub const GPT_6_LUNA: &str = "gpt-6-luna";

/// `gpt-5.6` completion model (alias that routes to GPT-5.6 Sol)
pub const GPT_5_6: &str = "gpt-5.6";

/// `gpt-5.6-sol` completion model
pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";

/// `gpt-5.6-terra` completion model
pub const GPT_5_6_TERRA: &str = "gpt-5.6-terra";

/// `gpt-5.6-luna` completion model
pub const GPT_5_6_LUNA: &str = "gpt-5.6-luna";

/// `gpt-5.5` completion model
pub const GPT_5_5: &str = "gpt-5.5";

/// GPT-5.4, API ID `gpt-5.4`: a reasoning model.
pub const GPT_5_4: &str = "gpt-5.4";

/// GPT-5.4 mini, API ID `gpt-5.4-mini`: a reasoning model.
pub const GPT_5_4_MINI: &str = "gpt-5.4-mini";

/// GPT-5.4 nano, API ID `gpt-5.4-nano`: a reasoning model.
pub const GPT_5_4_NANO: &str = "gpt-5.4-nano";

/// `gpt-5.2` completion model
pub const GPT_5_2: &str = "gpt-5.2";

/// GPT-5.2 Pro, API ID `gpt-5.2-pro`: a reasoning model served on the
/// Responses API only, without structured outputs.
pub const GPT_5_2_PRO: &str = "gpt-5.2-pro";

/// `gpt-5.1` completion model
pub const GPT_5_1: &str = "gpt-5.1";

/// `gpt-5` completion model
pub const GPT_5: &str = "gpt-5";
/// `gpt-5-mini` completion model.
pub const GPT_5_MINI: &str = "gpt-5-mini";
/// `gpt-5-nano` completion model.
pub const GPT_5_NANO: &str = "gpt-5-nano";

/// `gpt-4.5-preview` completion model
pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
/// `gpt-4.5-preview-2025-02-27` completion model
pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
/// `gpt-4o-2024-11-20` completion model.
pub const GPT_4O_2024_11_20: &str = "gpt-4o-2024-11-20";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";

/// `o4-mini-2025-04-16` completion model
pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";
/// `o4-mini` completion model
pub const O4_MINI: &str = "o4-mini";
/// `o3` completion model
pub const O3: &str = "o3";
/// `o3-mini` completion model
pub const O3_MINI: &str = "o3-mini";
/// `o3-mini-2025-01-31` completion model
pub const O3_MINI_2025_01_31: &str = "o3-mini-2025-01-31";
/// `o1-pro` completion model
pub const O1_PRO: &str = "o1-pro";
/// `o1` completion model.
pub const O1: &str = "o1";
/// `o1-2024-12-17` completion model
pub const O1_2024_12_17: &str = "o1-2024-12-17";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini` completion model.
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";

/// `gpt-4.1-mini` completion model
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano` completion model
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-4.1-2025-04-14` completion model
pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
/// `gpt-4.1` completion model
pub const GPT_4_1: &str = "gpt-4.1";

use crate::json_utils::string_or_vec;
use crate::message::{AudioMediaType, ImageDetail};
use crate::{completion, json_utils};
use serde::{Deserialize, Serialize, Serializer};
use std::convert::Infallible;
use std::str::FromStr;

pub use super::embedding::{CompletionTokensDetails, PromptTokensDetails, Usage};

/// Serializes user content as a plain string when there's a single text item,
/// otherwise as an array of content parts.
fn serialize_user_content<S>(content: &[UserContent], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if content.len() == 1
        && let Some(UserContent::Text { text, .. }) = content.first()
    {
        return serializer.serialize_str(text);
    }
    content.serialize(serializer)
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_vec")]
        content: Vec<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(
            deserialize_with = "string_or_vec",
            serialize_with = "serialize_user_content"
        )]
        content: Vec<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    // Gemini-backed OpenAI-compatible gateways (e.g. OpenRouter) can answer
    // with `role: "model"`; accept it on deserialization.
    #[serde(alias = "model")]
    Assistant {
        #[serde(
            default,
            deserialize_with = "json_utils::string_or_vec",
            skip_serializing_if = "Vec::is_empty",
            serialize_with = "serialize_assistant_content_vec"
        )]
        content: Vec<AssistantContent>,
        // OpenAI-compatible providers expose hidden reasoning on this non-standard
        // field, and some require it to be echoed back on assistant tool-call turns.
        // Serialized as `reasoning_content` (llama.cpp/DeepSeek dialect); the
        // `reasoning` alias accepts OpenRouter responses.
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "reasoning_content",
            alias = "reasoning"
        )]
        reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// Structured reasoning blocks used by OpenAI-compatible providers
        /// such as OpenRouter. Empty (and omitted from the wire) for
        /// providers that do not emit or accept them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning_details: Vec<ReasoningDetails>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: ToolResultContentValue,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: vec![content.to_owned().into()],
            name: None,
        }
    }
}

/// Structured reasoning blocks attached to assistant messages by
/// OpenAI-compatible providers such as OpenRouter (`reasoning_details`).
///
/// The `Option` fields are intentionally serialized even when `None`
/// (`"format":null,"id":null`) to match the provider wire format.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningDetails {
    #[serde(rename = "reasoning.summary")]
    Summary {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        summary: String,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        data: String,
    },
    #[serde(rename = "reasoning.text")]
    Text {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        text: Option<String>,
        signature: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    pub r#type: SystemContentType,
    pub text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Refusal { refusal: String },
}

impl From<AssistantContent> for completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        match value {
            AssistantContent::Text { text, .. } => completion::AssistantContent::text(text),
            AssistantContent::Refusal { refusal } => completion::AssistantContent::text(refusal),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
    },
    /// Audio content part, OpenAI's `input_audio` wire tag.
    #[serde(rename = "input_audio")]
    Audio {
        input_audio: InputAudio,
    },
    /// File content part for documents such as PDFs.
    ///
    /// Maps to OpenAI's `{"type":"file","file":{...}}` content type. Either
    /// `file_data` (a base64 data URI like `data:application/pdf;base64,...`)
    /// or `file_id` (a previously uploaded file reference) must be set.
    File {
        file: FileData,
    },
    /// Video content part (URL or base64 data URI), used by OpenAI-compatible
    /// providers such as OpenRouter. Wire tag: `video_url`.
    #[serde(rename = "video_url")]
    Video {
        video_url: VideoUrl,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    /// Image detail level. Optional so that providers whose wire format omits
    /// it (e.g. OpenRouter) can leave the key out entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Video payload for [`UserContent::Video`].
///
/// `url` is either a publicly accessible URL or a base64 data URI
/// (e.g. `data:video/mp4;base64,...`).
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

/// File payload for [`UserContent::File`].
///
/// At least one of `file_data` or `file_id` must be set for the content part
/// to be accepted by OpenAI's chat completions API. `filename` is optional
/// but recommended.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct FileData {
    /// Inline file data as a base64 data URI, e.g.
    /// `data:application/pdf;base64,JVBERi0xLjQK...`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    /// Identifier of a previously uploaded file (OpenAI Files API).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// Display name of the file. Recommended for inline `file_data`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Text or image content in a tool-result message.
/// Image emission requires
/// [`Quirks::supports_image_tool_results`](crate::providers::openai::wire::Quirks::supports_image_tool_results).
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum ToolResultContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
}

impl ToolResultContent {
    /// The text of this part, or `None` for a non-text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            Self::Image { .. } => None,
        }
    }
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent::Text { text: s }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContentValue {
    Array(Vec<ToolResultContent>),
    String(String),
}

impl ToolResultContentValue {
    /// Join text parts with newlines, discarding images. String values are cloned.
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContentValue::Array(arr) => arr
                .iter()
                .filter_map(ToolResultContent::as_text)
                .collect::<Vec<_>>()
                .join("\n"),
            ToolResultContentValue::String(s) => s.clone(),
        }
    }

    /// Whether any part of this result is an image.
    pub fn has_image(&self) -> bool {
        matches!(self, ToolResultContentValue::Array(arr)
            if arr.iter().any(|c| matches!(c, ToolResultContent::Image { .. })))
    }

    pub fn to_array(&self) -> Self {
        match self {
            ToolResultContentValue::Array(_) => self.clone(),
            ToolResultContentValue::String(s) => {
                ToolResultContentValue::Array(vec![ToolResultContent::from(s.clone())])
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(
        serialize_with = "json_utils::stringified_json::serialize",
        deserialize_with = "json_utils::stringified_json::deserialize_maybe_stringified"
    )]
    pub arguments: serde_json::Value,
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text { text: s }
    }
}

impl From<&str> for UserContent {
    fn from(s: &str) -> Self {
        s.to_owned().into()
    }
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for AssistantContent {
    fn from(s: String) -> Self {
        AssistantContent::Text { text: s }
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}
impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    // Null-or-missing tolerated on deserialization: some OpenAI-compatible
    // gateways (HuggingFace router sub-providers, TGI variants, Copilot's
    // multi-vendor chat route) omit them or send explicit `null`.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub object: String,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    /// Service tier that processed the request, when OpenAI reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(
        deserialize_with = "recorded_choices::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// Return a nonempty top-level refusal only when every content part is empty.
/// Applies to whole messages; streaming fallback is evaluated per delta.
#[cfg(test)]
pub(crate) fn assistant_refusal_fallback<'a>(
    content: &[AssistantContent],
    refusal: Option<&'a str>,
) -> Option<&'a str> {
    let has_text = content.iter().any(|part| {
        !match part {
            AssistantContent::Text { text } => text,
            AssistantContent::Refusal { refusal } => refusal,
        }
        .is_empty()
    });

    refusal.filter(|refusal| !has_text && !refusal.is_empty())
}

/// The whole-message text view: every non-empty part in arrival order, with
/// the sibling `refusal` appended only when [`assistant_refusal_fallback`]
/// says it is the turn's text.
///
/// No wire path reads text this way — the driver records off the folded
/// response — so this survives for the OpenAI-compatible providers' unary
/// decode tests, which read a decoded message's text through it.
#[cfg(test)]
pub(crate) fn assistant_message_text_response(message: &Message) -> Option<String> {
    let Message::Assistant {
        content, refusal, ..
    } = message
    else {
        return None;
    };

    let mut segments = content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text { text, .. } => (!text.is_empty()).then(|| text.clone()),
            AssistantContent::Refusal { refusal } => (!refusal.is_empty()).then(|| refusal.clone()),
        })
        .collect::<Vec<_>>();

    if let Some(refusal) = assistant_refusal_fallback(content, refusal.as_deref()) {
        segments.push(refusal.to_owned());
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments.join("\n"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    // Null-or-missing tolerated on deserialization: Copilot's chat route
    // (fronting non-OpenAI vendors) can omit either field or send explicit
    // `null`; normalization treats "" as absent.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub finish_reason: String,
}

fn serialize_assistant_content_vec<S>(
    value: &[AssistantContent],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_empty() {
        serializer.serialize_str("")
    } else {
        value.serialize(serializer)
    }
}

// The retained response reader uses F's exact length-only repair gate.
// This private reader does not change the live Chat decoder's vocabulary.
mod recorded_choices {
    use crate::completion::FinishReason;
    use serde::{Deserialize, Deserializer};

    pub(crate) fn map_openai_finish_reason(reason: &str) -> FinishReason {
        match reason {
            "stop" => FinishReason::Stop,
            // Context-window exhaustion and output-budget exhaustion both truncate.
            "length" | "max_tokens" | "model_length" => FinishReason::Length,
            "tool_calls" | "function_call" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            other => FinishReason::Other(other.to_owned()),
        }
    }

    /// Deserialize choices, dropping incomplete tool calls only for length finishes.
    /// Dropping requires a copy with arguments repaired to `{}` to deserialize as `T`.
    /// Other defects remain deserialization errors.
    pub(crate) fn deserialize_choices_dropping_incomplete_tool_calls<'de, D, T>(
        deserializer: D,
    ) -> Result<Vec<T>, D::Error>
    where
        D: Deserializer<'de>,
        T: serde::de::DeserializeOwned,
    {
        deserialize_choices_dropping_incomplete_tool_calls_when(deserializer, |choice| {
            choice
                .get("finish_reason")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|reason| {
                    matches!(map_openai_finish_reason(reason), FinishReason::Length)
                })
        })
    }

    /// Deserialize choices using `is_output_length` to select truncated turns.
    /// Applies [`deserialize_choices_dropping_incomplete_tool_calls`]'s repair check
    /// before dropping incomplete calls.
    pub(crate) fn deserialize_choices_dropping_incomplete_tool_calls_when<'de, D, T, F>(
        deserializer: D,
        is_output_length: F,
    ) -> Result<Vec<T>, D::Error>
    where
        D: Deserializer<'de>,
        T: serde::de::DeserializeOwned,
        F: Fn(&serde_json::Value) -> bool,
    {
        Vec::<serde_json::Value>::deserialize(deserializer)?
            .into_iter()
            .map(|mut choice| {
                if is_output_length(&choice) {
                    let dropped = drop_tool_calls_cut_by_budget::<T>(&mut choice);
                    if dropped > 0 {
                        tracing::debug!(
                            dropped,
                            "dropping tool calls incomplete under an output-length finish reason"
                        );
                    }
                }

                serde_json::from_value(choice).map_err(serde::de::Error::custom)
            })
            .collect()
    }

    /// Drop incomplete argument strings from a choice and return the number removed.
    /// The caller must establish an output-length finish. Returns zero unchanged if
    /// repairing the arguments to `{}` does not make the choice deserialize as `T`.
    pub(crate) fn drop_tool_calls_cut_by_budget<T>(choice: &mut serde_json::Value) -> usize
    where
        T: serde::de::DeserializeOwned,
    {
        let mut probe = choice.clone();
        if !repair_incomplete_arguments(&mut probe) || serde_json::from_value::<T>(probe).is_err() {
            return 0;
        }
        drop_incomplete_arguments(choice)
    }

    /// Whether arguments are an empty or unparseable string; nonstrings return false.
    /// Empty strings count as incomplete because truncation can precede the first token.
    fn incomplete_arguments(call: &serde_json::Value) -> bool {
        call.get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|raw| {
                raw.trim().is_empty() || crate::json_utils::parse_tool_arguments(raw).is_err()
            })
    }

    fn message_tool_calls_mut(
        choice: &mut serde_json::Value,
    ) -> Option<&mut Vec<serde_json::Value>> {
        choice
            .get_mut("message")
            .and_then(|message| message.get_mut("tool_calls"))
            .and_then(serde_json::Value::as_array_mut)
    }

    fn repair_incomplete_arguments(choice: &mut serde_json::Value) -> bool {
        let Some(tool_calls) = message_tool_calls_mut(choice) else {
            return false;
        };

        let mut repaired = false;
        for call in tool_calls {
            if !incomplete_arguments(call) {
                continue;
            }
            let Some(arguments) = call
                .get_mut("function")
                .and_then(|function| function.get_mut("arguments"))
            else {
                continue;
            };
            *arguments = serde_json::Value::String("{}".to_owned());
            repaired = true;
        }
        repaired
    }

    fn drop_incomplete_arguments(choice: &mut serde_json::Value) -> usize {
        let Some(tool_calls) = message_tool_calls_mut(choice) else {
            return 0;
        };

        let before = tool_calls.len();
        tool_calls.retain(|call| !incomplete_arguments(call));
        before - tool_calls.len()
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod image_tool_result_gate_tests;

#[cfg(test)]
mod retained_response_tests;
