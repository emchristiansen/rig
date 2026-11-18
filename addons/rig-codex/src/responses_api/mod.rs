//! Typed Responses data and the Codex pieces built around Rig's Responses wire.
//!
//! The data types here are plain serde values for the Responses API's
//! response object, its output items and its request parameters. Rig's own
//! Responses wire speaks JSON; these give a caller that reads the provider's
//! frames a typed view of them, obtained with `serde_json::from_value` or
//! [`streaming::decode_frame`]. An output item or content part this crate does
//! not model is kept whole rather than refused.
//!
//! ```
//! use rig_codex::responses_api::{Output, ResponseStatus};
//!
//! let item: Output = serde_json::from_value(serde_json::json!({
//!     "type": "web_search_call", "id": "ws_1", "status": "completed",
//! }))?;
//! assert!(matches!(item, Output::Unknown(_)));
//! let status: ResponseStatus = serde_json::from_value(serde_json::json!("completed"))?;
//! assert_eq!(status, ResponseStatus::Completed);
//! # Ok::<(), serde_json::Error>(())
//! ```

use rig_core::completion::{self, CompletionRequest};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::json_utils;
use rig_core::operation::Completion;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Body, Mode, Operation, Wire};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::ops::Add;

pub(crate) mod body;
pub mod codex_identity;
pub(crate) mod custom_output;
#[cfg(feature = "completion-observations")]
#[cfg_attr(docsrs, doc(cfg(feature = "completion-observations")))]
pub mod observation;
pub mod request_compression;
pub mod request_headers;
pub mod streaming;
pub mod websocket;
pub mod wire;

pub use rig_core::providers::openai::responses_api::{
    ResponsesToolDefinition, SystemInstructionsPlacement,
};

/// One Responses input item, as Rig's encoder writes it.
pub type InputItem = Value;

/// The Responses input items Rig's encoder writes for `request` on `wire`:
/// the request is prepared for the wire as the driver prepares it, encoded
/// as a streamed HTTP request, and the `input` of its body read back.
///
/// This is how a caller projects appended history into the items of an
/// incremental websocket turn
/// ([`InputDelta`](websocket::InputDelta)). A request that continues stored
/// state names it (`previous_response_id` in its `additional_params`), so the
/// encoder accepts results whose calls the request does not carry. A custom
/// call's answer is written as [`CodexResponses`](wire::CodexResponses)
/// writes it: one string, the tool's name stated, an image refused.
pub fn input_items(
    wire: &Responses,
    request: CompletionRequest,
) -> Result<Vec<InputItem>, ProviderError> {
    let request = Completion::prepare(request, &wire.describe())?;
    let custom_outputs = custom_output::CustomOutputs::of(&request);
    let encoded = wire.encode(request, Mode::Streaming)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(EncodeError::request("a Responses request body must be JSON bytes").into());
    };
    let body = body::RequestBody::parse(bytes)?;
    match body.get::<Vec<InputItem>>("input") {
        Some(items) => {
            let mut items = items?;
            custom_outputs
                .restore(&mut items)
                .map_err(EncodeError::request)?;
            Ok(items)
        }
        None => {
            Err(EncodeError::request("the encoded Responses request has no `input` array").into())
        }
    }
}

/// One part of a reasoning item's summary.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    /// A summary as text.
    SummaryText {
        /// The summary text.
        text: String,
    },
    /// An unmodeled summary part, retained verbatim.
    #[serde(untagged)]
    Unknown(Value),
}

impl ReasoningSummary {
    /// The summary's text, when this part carries text.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::SummaryText { text } => Some(text),
            Self::Unknown(_) => None,
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningSummary {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match content_part_tag::<D::Error>(&value)? {
            "summary_text" => {
                #[derive(Deserialize)]
                struct Fields {
                    text: String,
                }
                let fields: Fields =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self::SummaryText { text: fields.text })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

/// A reasoning content part. Known text is decoded strictly; unknown tags retain their JSON.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningTextContent {
    /// Reasoning text.
    ReasoningText {
        /// The reasoning text.
        text: String,
    },
    /// A reasoning content part of a type this crate does not model, kept whole.
    #[serde(untagged)]
    Unknown(Value),
}

impl From<String> for ReasoningTextContent {
    fn from(text: String) -> Self {
        Self::ReasoningText { text }
    }
}

impl From<&str> for ReasoningTextContent {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl<'de> Deserialize<'de> for ReasoningTextContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if let Value::String(text) = value {
            return Ok(Self::ReasoningText { text });
        }
        match content_part_tag::<D::Error>(&value)? {
            "reasoning_text" => {
                #[derive(Deserialize)]
                struct Fields {
                    text: String,
                }
                let fields: Fields =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self::ReasoningText { text: fields.text })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

fn content_part_tag<E: serde::de::Error>(value: &Value) -> Result<&str, E> {
    value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| E::custom("content part requires an object with a string `type`"))
}

fn deserialize_reasoning_text_content<'de, D>(
    deserializer: D,
) -> Result<Vec<ReasoningTextContent>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(text) => Ok(vec![text.into()]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| serde_json::from_value(item).map_err(serde::de::Error::custom))
            .collect(),
        _ => Err(serde::de::Error::custom(
            "reasoning content must be an array or string",
        )),
    }
}

/// Response token counts and optional cached-input and reasoning breakdowns.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Input tokens
    pub input_tokens: u64,
    /// In-depth detail on input tokens (cached tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Output tokens
    pub output_tokens: u64,
    /// In-depth detail on output tokens (reasoning tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Total tokens used (for a given prompt)
    pub total_tokens: u64,
}

impl From<&ResponsesUsage> for completion::Usage {
    fn from(usage: &ResponsesUsage) -> Self {
        let mut normalized = completion::Usage::new()
            .input_tokens(usage.input_tokens)
            .output_tokens(usage.output_tokens);
        normalized.total_tokens = Some(usage.total_tokens);
        normalized.cached_input_tokens = usage
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens);
        normalized.reasoning_tokens = usage
            .output_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens);
        normalized
    }
}

impl From<ResponsesUsage> for completion::Usage {
    fn from(usage: ResponsesUsage) -> Self {
        Self::from(&usage)
    }
}

/// Sums input-token breakdowns. A side without one reports no cached count,
/// so the summed count is unknown, never the other side's value alone:
/// absence is not zero, in aggregation as in decoding.
fn add_input_details(
    lhs: Option<InputTokensDetails>,
    rhs: Option<InputTokensDetails>,
) -> Option<InputTokensDetails> {
    const UNREPORTED: InputTokensDetails = InputTokensDetails {
        cached_tokens: None,
    };
    match (lhs, rhs) {
        (None, None) => None,
        (lhs, rhs) => Some(lhs.unwrap_or(UNREPORTED) + rhs.unwrap_or(UNREPORTED)),
    }
}

/// Adds present breakdowns, preserving a lone value or joint absence.
///
/// Used for output-token breakdowns only, whose counters are required: an
/// absent breakdown there is kept as the other side's (see
/// [`add_input_details`] for the input side, whose count is optional).
fn add_optional_details<T: Add<Output = T>>(lhs: Option<T>, rhs: Option<T>) -> Option<T> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
        (lhs, rhs) => lhs.or(rhs),
    }
}

impl Add for ResponsesUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            input_tokens_details: add_input_details(
                self.input_tokens_details,
                rhs.input_tokens_details,
            ),
            output_tokens: self.output_tokens + rhs.output_tokens,
            output_tokens_details: add_optional_details(
                self.output_tokens_details,
                rhs.output_tokens_details,
            ),
            total_tokens: self.total_tokens + rhs.total_tokens,
        }
    }
}

/// In-depth details on input tokens.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Cached tokens from OpenAI, `None` when the provider omits the count.
    ///
    /// Absence is not zero: an omitted count stays `None` through
    /// deserialization, arithmetic and the usage projection, so a missing
    /// field never fails the whole decode and is never reported as zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

impl Add for InputTokensDetails {
    type Output = Self;
    /// A sum is known only when both parts are: one absent side makes the
    /// total absent rather than silently counting it as zero.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cached_tokens: self
                .cached_tokens
                .zip(rhs.cached_tokens)
                .map(|(left, right)| left + right),
        }
    }
}

/// In-depth details on output tokens.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens
    pub reasoning_tokens: u64,
}

impl Add for OutputTokensDetails {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
        }
    }
}

/// Provider-reported reason for an incomplete response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IncompleteDetailsReason {
    /// The reason for an incomplete [`CompletionResponse`].
    pub reason: String,
}

/// A response error from OpenAI's Response API.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code, `None` when the provider omits it. A failed response
    /// without a code still decodes, so its message and status survive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Error message
    pub message: String,
}

/// A response object as an enum (ensures type validation)
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseObject {
    /// A response (`"response"`).
    Response,
}

/// The response status as an enum (ensures type validation)
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseStatus {
    /// The response is being generated.
    InProgress,
    /// The response finished.
    Completed,
    /// The response failed.
    Failed,
    /// The response was cancelled.
    Cancelled,
    /// The response is waiting to start.
    Queued,
    /// The response stopped before it finished; see `incomplete_details`.
    Incomplete,
    /// A provider-specific status added after this client was released.
    Other(String),
}

/// The wire spelling of a [`ResponseStatus`].
///
/// Statuses outside the normalized finish-reason vocabulary are carried through
/// as [`completion::FinishReason::Other`], so they must keep OpenAI's own
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

/// The standard response format from OpenAI's Responses API.
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    /// The ID of a completion response.
    pub id: String,
    /// The type of the object.
    pub object: ResponseObject,
    /// The time at which a given response has been created, in seconds from the UNIX epoch (01/01/1970 00:00:00).
    pub created_at: u64,
    /// The status of the response.
    pub status: ResponseStatus,
    /// Response error (optional)
    pub error: Option<ResponseError>,
    /// Incomplete response details (optional)
    pub incomplete_details: Option<IncompleteDetailsReason>,
    /// System prompt/preamble
    pub instructions: Option<String>,
    /// The maximum number of tokens the model should output
    pub max_output_tokens: Option<u64>,
    /// The model name
    pub model: String,
    /// Provider-specific top-level reasoning content returned by some
    /// OpenAI-compatible Responses implementations.
    pub provider_reasoning: Option<String>,
    /// Transport request ID from the `x-request-id` header, stamped by the driver.
    /// Body deserialization leaves it `None`; serialization omits it.
    pub provider_request_id: Option<String>,
    /// The complete object-shaped top-level reasoning metadata returned by the provider.
    ///
    /// Unknown fields, unknown values, and null-valued members inside the object
    /// are preserved value-equivalently. A top-level null, missing field, or
    /// unsupported non-object shape is normalized to no reasoning metadata.
    /// When serializing manually constructed responses, [`Self::provider_reasoning`]
    /// takes precedence over this field, and this field takes precedence over
    /// [`Self::reasoning_context`].
    pub reasoning_metadata: Option<Map<String, Value>>,
    /// The effective reasoning context returned by OpenAI.
    ///
    /// This is populated as a convenience projection of
    /// [`Self::reasoning_metadata`]. String-shaped reasoning returned by compatible
    /// providers remains available through [`Self::provider_reasoning`].
    pub reasoning_context: Option<String>,
    /// Token usage
    pub usage: Option<ResponsesUsage>,
    /// The model output (messages, etc will go here)
    pub output: Vec<Output>,
    /// Tools
    pub tools: Vec<ResponsesToolDefinition>,
    /// Additional parameters
    pub additional_parameters: AdditionalParameters,
}

#[derive(Serialize)]
#[serde(untagged)]
enum CompletionResponseReasoningRef<'a> {
    Text(&'a str),
    Metadata(&'a Map<String, Value>),
    Context { context: &'a str },
}

#[derive(Serialize)]
struct CompletionResponseWireRef<'a> {
    id: &'a str,
    object: &'a ResponseObject,
    created_at: u64,
    status: &'a ResponseStatus,
    error: &'a Option<ResponseError>,
    incomplete_details: &'a Option<IncompleteDetailsReason>,
    instructions: &'a Option<String>,
    max_output_tokens: &'a Option<u64>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<CompletionResponseReasoningRef<'a>>,
    usage: &'a Option<ResponsesUsage>,
    output: &'a Vec<Output>,
    tools: &'a Vec<ResponsesToolDefinition>,
    #[serde(flatten)]
    additional_parameters: &'a AdditionalParameters,
}

/// Response body with untyped echoed metadata. Metadata is decoded separately
/// so an incompatible optional field does not reject the response.
#[derive(Deserialize)]
struct CompletionResponseWire {
    id: String,
    object: ResponseObject,
    created_at: u64,
    status: ResponseStatus,
    error: Option<ResponseError>,
    incomplete_details: Option<IncompleteDetailsReason>,
    instructions: Option<String>,
    max_output_tokens: Option<u64>,
    model: String,
    #[serde(default)]
    reasoning: Option<Value>,
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    output: Vec<Output>,
    #[serde(default)]
    tools: Vec<ResponsesToolDefinition>,
    #[serde(flatten)]
    metadata: Map<String, Value>,
}

impl Serialize for CompletionResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Omit request reasoning configuration to avoid duplicate response keys.
        let mut additional_parameters = self.additional_parameters.clone();
        additional_parameters.reasoning = None;

        let reasoning = self
            .provider_reasoning
            .as_deref()
            .map(CompletionResponseReasoningRef::Text)
            .or_else(|| {
                self.reasoning_metadata
                    .as_ref()
                    .map(CompletionResponseReasoningRef::Metadata)
            })
            .or_else(|| {
                self.reasoning_context
                    .as_deref()
                    .map(|context| CompletionResponseReasoningRef::Context { context })
            });

        CompletionResponseWireRef {
            id: &self.id,
            object: &self.object,
            created_at: self.created_at,
            status: &self.status,
            error: &self.error,
            incomplete_details: &self.incomplete_details,
            instructions: &self.instructions,
            max_output_tokens: &self.max_output_tokens,
            model: &self.model,
            reasoning,
            usage: &self.usage,
            output: &self.output,
            tools: &self.tools,
            additional_parameters: &additional_parameters,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CompletionResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let response = CompletionResponseWire::deserialize(deserializer)?;
        let (provider_reasoning, reasoning_metadata) = match response.reasoning {
            Some(Value::String(reasoning)) => (Some(reasoning), None),
            Some(Value::Object(metadata)) => (None, Some(metadata)),
            // Unsupported reasoning shapes must not reject the response.
            _ => (None, None),
        };
        let reasoning_context = reasoning_metadata
            .as_ref()
            .and_then(|reasoning| reasoning.get("context"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        Ok(Self {
            id: response.id,
            object: response.object,
            created_at: response.created_at,
            status: response.status,
            error: response.error,
            incomplete_details: response.incomplete_details,
            instructions: response.instructions,
            max_output_tokens: response.max_output_tokens,
            model: response.model,
            provider_reasoning,
            provider_request_id: None,
            reasoning_metadata,
            reasoning_context,
            usage: response.usage,
            output: response.output,
            tools: response.tools,
            additional_parameters: AdditionalParameters::from_response_metadata(response.metadata),
        })
    }
}

/// Intended to be derived from [`rig_core::completion::CompletionRequest`].
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct AdditionalParameters {
    /// Whether or not a given model task should run in the background (ie a detached process).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// The text response format. This is where you would add structured outputs (if you want them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    /// Additional response fields to request from the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<Include>>,
    /// `top_p`. Mutually exclusive with the `temperature` argument.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Whether or not the response should be truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    /// The username of the user (that you want to use).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// A stable cache routing key for prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    /// Any additional metadata you'd like to add. This will additionally be returned by the response.
    #[serde(
        skip_serializing_if = "Map::is_empty",
        default,
        deserialize_with = "deserialize_metadata"
    )]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Whether or not you want tool calls to run in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Previous response ID. If you are not sending a full conversation, this can help to track the message flow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Add thinking/reasoning to your response. The response will be emitted as a list member of the `output` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// The service tier you're using.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    /// Whether or not to store the response for later retrieval by API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Provider-specific client metadata, serialized at the top level of the
    /// request body (the Codex backend reads its `session_id`/`thread_id`
    /// cache identity here). Empty by default and omitted when empty.
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub client_metadata: BTreeMap<String, String>,
}

fn deserialize_metadata<'de, D>(
    deserializer: D,
) -> Result<serde_json::Map<String, serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(
        Option::<serde_json::Map<String, serde_json::Value>>::deserialize(deserializer)?
            .unwrap_or_default(),
    )
}

impl AdditionalParameters {
    /// Project echoed response metadata into the request-shaped parameters.
    ///
    /// Each key is decoded on its own; a key whose value does not fit its
    /// field is dropped rather than failing the response. A non-numeric
    /// `top_p` from a compatible endpoint therefore reads back as `None`.
    fn from_response_metadata(metadata: Map<String, Value>) -> Self {
        let mut accepted = Map::with_capacity(metadata.len());
        for (key, value) in metadata {
            let probe = Value::Object(Map::from_iter([(key.clone(), value.clone())]));
            if serde_json::from_value::<Self>(probe).is_ok() {
                accepted.insert(key, value);
            } else {
                tracing::debug!(
                    target: "rig::providers::openai",
                    field = %key,
                    "ignoring response metadata field that does not match its expected type"
                );
            }
        }
        // Every remaining key was individually accepted, so this cannot fail;
        // `unwrap_or_default` keeps the projection total without a panic path.
        serde_json::from_value(Value::Object(accepted)).unwrap_or_default()
    }

    /// These parameters as a JSON value, for a request's `additional_params`.
    pub fn to_json(self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|_| serde_json::Value::Object(Map::new()))
    }
}

/// The truncation strategy.
/// When using auto, if the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
/// Otherwise, does nothing (and is disabled by default).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    /// The provider drops older input items to fit the context window.
    Auto,
    /// Nothing is dropped; a request that does not fit fails.
    #[default]
    Disabled,
}

/// The model output format configuration.
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextConfig {
    /// The output format.
    pub format: TextFormat,
}

/// The text format (contained by [`TextConfig`]).
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    /// Output matching a JSON schema.
    JsonSchema(StructuredOutputsInput),
    /// Plain text.
    #[default]
    Text,
}

/// The inputs required for adding structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredOutputsInput {
    /// The name of your schema.
    ///
    /// Compatible providers may omit it when echoing a response configuration.
    #[serde(default)]
    pub name: String,
    /// Your required output schema. It is recommended that you use the JsonSchema macro, which you can check out at <https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html>.
    pub schema: serde_json::Value,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    #[serde(default)]
    pub strict: bool,
}

/// The `reasoning` controls of a Responses request.
///
/// # Example
/// ```
/// use rig_codex::responses_api::{
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
    /// No reasoning.
    None,
    /// The least reasoning.
    Minimal,
    /// Little reasoning.
    Low,
    /// A moderate amount of reasoning.
    #[default]
    Medium,
    /// Much reasoning.
    High,
    /// More reasoning than `high`.
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
    /// The provider chooses the level of detail.
    #[default]
    Auto,
    /// A short summary.
    Concise,
    /// A detailed summary.
    Detailed,
}

/// Additional response fields requested through [`AdditionalParameters::include`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Include {
    /// File search results (`file_search_call.results`).
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Input image URLs (`message.input_image.image_url`).
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    /// Computer call output image URLs (`computer_call_output.output.image_url`).
    #[serde(rename = "computer_call.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    /// Encrypted reasoning content (`reasoning.encrypted_content`), which a request that stores nothing replays reasoning with.
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    /// Code interpreter outputs (`code_interpreter_call.outputs`).
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
}

/// A Responses output item. Unrecognized types, including hosted tools, decode
/// to [`Output::Unknown`] with their JSON value preserved. Malformed known
/// types fail deserialization.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// An assistant message.
    Message(OutputMessage),
    /// A function call.
    FunctionCall(OutputFunctionCall),
    /// A custom (grammar) tool call, whose `input` is verbatim text rather
    /// than JSON arguments.
    CustomToolCall(OutputCustomToolCall),
    /// A reasoning item.
    Reasoning {
        /// The item's ID.
        id: String,
        /// The reasoning summary parts.
        summary: Vec<ReasoningSummary>,
        /// The reasoning text parts, when the provider returns them.
        content: Vec<ReasoningTextContent>,
        /// The reasoning as ciphertext, when the request included it.
        encrypted_content: Option<String>,
        /// The upstream's signature over the reasoning text, when a gateway
        /// relays one (OpenRouter for Claude).
        signature: Option<String>,
        /// The item's status, when the provider reports one.
        status: Option<ToolStatus>,
    },
    /// An opaque compaction item (`"type": "compaction"`), preserved verbatim
    /// so it can be sent back as an input item on the next request. Kept
    /// distinct from [`Output::Unknown`] because OpenAI documents it as a
    /// must-replay item.
    Compaction(Map<String, Value>),
    /// Catch-all for output item types this version does not model. Holds the
    /// raw item object exactly as it appeared in the provider's `output[]`
    /// array, so hosted-tool payloads survive the typed decode.
    Unknown(Value),
}

/// Deserialization fields for [`Output::Reasoning`].
#[derive(Deserialize)]
struct ReasoningFields {
    id: String,
    #[serde(default)]
    summary: Vec<ReasoningSummary>,
    #[serde(default, deserialize_with = "deserialize_reasoning_text_content")]
    content: Vec<ReasoningTextContent>,
    #[serde(default)]
    encrypted_content: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    status: Option<ToolStatus>,
}

impl From<ReasoningFields> for Output {
    fn from(fields: ReasoningFields) -> Self {
        Output::Reasoning {
            id: fields.id,
            summary: fields.summary,
            content: fields.content,
            encrypted_content: fields.encrypted_content,
            signature: fields.signature,
            status: fields.status,
        }
    }
}

/// Serializes an object payload with a `type` tag. Non-object payloads fail.
/// Object key order is not preserved.
fn tagged_output_object<T>(tag: &str, payload: &T) -> Result<Value, serde_json::Error>
where
    T: Serialize,
{
    let mut value = serde_json::to_value(payload)?;
    let map = value.as_object_mut().ok_or_else(|| {
        <serde_json::Error as serde::ser::Error>::custom(
            "output payload must serialize to a JSON object",
        )
    })?;
    map.insert("type".to_string(), Value::String(tag.to_string()));
    Ok(value)
}

impl Serialize for Output {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Output::Message(message) => tagged_output_object("message", message),
            Output::FunctionCall(call) => tagged_output_object("function_call", call),
            Output::CustomToolCall(call) => tagged_output_object("custom_tool_call", call),
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                status,
            } => {
                let mut value = serde_json::json!({
                    "type": "reasoning",
                    "id": id,
                    "summary": summary,
                    "encrypted_content": encrypted_content,
                    "status": status,
                });
                let map = value.as_object_mut().ok_or_else(|| {
                    serde::ser::Error::custom("reasoning output must serialize to an object")
                })?;
                if !content.is_empty() {
                    map.insert(
                        "content".to_string(),
                        serde_json::to_value(content).map_err(serde::ser::Error::custom)?,
                    );
                }
                if let Some(signature) = signature {
                    map.insert("signature".to_string(), Value::String(signature.clone()));
                }
                Ok(value)
            }
            Output::Compaction(fields) => {
                let mut map = fields.clone();
                map.insert("type".to_string(), Value::String("compaction".to_string()));
                return Value::Object(map).serialize(serializer);
            }
            Output::Unknown(value) => return value.serialize(serializer),
        };
        value
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Output {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Preserve unmodeled items, including absent or non-string tags.
        // Malformed bodies with known tags must still fail.
        let value = Value::deserialize(deserializer)?;
        let Some(tag) = value.get("type").and_then(Value::as_str) else {
            return Ok(Output::Unknown(value));
        };
        match tag {
            "message" => serde_json::from_value(value)
                .map(Output::Message)
                .map_err(serde::de::Error::custom),
            "function_call" => serde_json::from_value(value)
                .map(Output::FunctionCall)
                .map_err(serde::de::Error::custom),
            // Must precede the catch-all: without it a custom call decodes as
            // `Unknown` and loses its typed identity silently.
            "custom_tool_call" => serde_json::from_value(value)
                .map(Output::CustomToolCall)
                .map_err(serde::de::Error::custom),
            "reasoning" => serde_json::from_value::<ReasoningFields>(value)
                .map(Output::from)
                .map_err(serde::de::Error::custom),
            "compaction" => {
                let Value::Object(mut map) = value else {
                    return Ok(Output::Unknown(value));
                };
                json_utils::remove_preserving_order(&mut map, "type");
                Ok(Output::Compaction(map))
            }
            _ => Ok(Output::Unknown(value)),
        }
    }
}

/// An OpenAI Responses API tool call. A call ID will be returned that must be used when creating a tool result to send back to OpenAI as a message input, otherwise an error will be received.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputFunctionCall {
    /// Provider-assigned `fc_...` item ID. The Responses API rejects
    /// `function_call` input IDs that are not native `fc` item IDs, so IDs
    /// minted outside the Responses API (by Rig's agent loop or another
    /// provider) are omitted on serialization and the call is paired with its
    /// output by `call_id` alone.
    #[serde(default, skip_serializing_if = "is_not_function_call_item_id")]
    pub id: String,
    /// The call's arguments, as the provider wrote them.
    pub arguments: FunctionCallArguments,
    /// The ID a tool result names this call by.
    pub call_id: String,
    /// The function's name.
    pub name: String,
    /// The namespace qualifying this callable, verbatim; absent for the
    /// default one.
    ///
    /// Present in both directions because this struct is shared by
    /// [`Output::FunctionCall`] and the function-call input item; dropping
    /// it on either side would strip the qualifier before any consumer saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The call's status.
    pub status: ToolStatus,
}

/// An OpenAI Responses custom (grammar) tool call.
///
/// Distinct from [`OutputFunctionCall`] because its payload is `input`:
/// verbatim text that is not JSON arguments, even when it happens to parse.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputCustomToolCall {
    /// Provider-assigned output-item id, omitted when absent.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    /// The correlator its [`CustomToolResult`] echoes back.
    pub call_id: String,
    /// The custom tool's name.
    pub name: String,
    /// The namespace qualifying this callable, verbatim; absent for the
    /// default one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The model's verbatim input, preserved byte for byte.
    pub input: String,
    /// The item status, when the provider stated one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// The output of a custom (grammar) tool call, sent back as input.
///
/// Separate from a function call's result because the custom output item carries a plain
/// string, where a function output carries text or rich blocks.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CustomToolResult {
    /// The output item's own id, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The answered call's correlator.
    pub call_id: String,
    /// The executed tool's name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The tool's output text.
    pub output: String,
}

/// Raw Responses function-call arguments, parsed as JSON at consumption time.
/// Truncated arguments remain valid wire data; parsing them can fail without
/// discarding the enclosing response or its terminal status.
///
/// One type carries the arguments of **both** events that state them (the
/// `response.function_call_arguments.done` event,
/// [`streaming::ArgsTextChunk`], and the `response.output_item.done` item),
/// and [`Self::classify`] is the one parsed/unparseable classification both
/// are read through, so the two observations of one call cannot drift apart
/// by being decoded differently. Deciding whether they agree is
/// [`Self::reconcile`].
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCallArguments(String);

/// What an argument string is, under the one classification every reader of
/// [`FunctionCallArguments`] shares.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassifiedArguments {
    /// The string parsed. An empty or whitespace-only string is a
    /// parameterless invocation and classifies as `{}`.
    Parsed(serde_json::Value),
    /// A non-empty string that does not parse. The bytes are retained as the
    /// provider sent them; nothing is substituted for them.
    Unparseable(String),
}

/// Which side of a [`FunctionCallArguments::reconcile`] did not parse: `Left`
/// is the receiver, `Right` the argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnparseableSide {
    /// Only the left observation.
    Left,
    /// Only the right observation.
    Right,
    /// Both observations.
    Both,
}

/// Why two observations of one call's arguments could not be reconciled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReconcileArgumentsError {
    /// An unparseable observation establishes no arguments, so it agrees
    /// with nothing, including another unparseable one.
    #[error(
        "function-call arguments could not be reconciled: the {0:?} observation does not parse"
    )]
    Unparseable(UnparseableSide),
    /// Both observations parsed, to different values.
    #[error(
        "function-call arguments could not be reconciled: the two observations parse to different values"
    )]
    Mismatch,
}

impl FunctionCallArguments {
    /// Parse the raw wire string into JSON arguments. An empty string is a
    /// parameterless invocation (`{}`); anything else must parse as JSON.
    pub fn parse(&self) -> serde_json::Result<serde_json::Value> {
        json_utils::parse_tool_arguments(&self.0)
    }

    /// The raw wire string, exactly as the provider sent it.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The one parsed/unparseable classification of these arguments, decided
    /// by the same [`json_utils::parse_tool_arguments`] every consumer of the
    /// wire string uses.
    pub fn classify(&self) -> ClassifiedArguments {
        match self.parse() {
            Ok(value) => ClassifiedArguments::Parsed(value),
            Err(_) => ClassifiedArguments::Unparseable(self.0.clone()),
        }
    }

    /// The agreed arguments of two observations of one call (for example the
    /// `function_call_arguments.done` event and the `output_item.done` item).
    ///
    /// An unparseable side refuses, naming which side it was; two parsed
    /// values that differ refuse as a mismatch. Agreement is structural
    /// equality of the parsed values, so spelling differences the parse
    /// erases (whitespace, key order) agree, and an empty string agrees with
    /// `{}` because both are a parameterless invocation.
    pub fn reconcile(&self, other: &Self) -> Result<serde_json::Value, ReconcileArgumentsError> {
        match (self.classify(), other.classify()) {
            (ClassifiedArguments::Unparseable(_), ClassifiedArguments::Unparseable(_)) => {
                Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Both))
            }
            (ClassifiedArguments::Unparseable(_), ClassifiedArguments::Parsed(_)) => {
                Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Left))
            }
            (ClassifiedArguments::Parsed(_), ClassifiedArguments::Unparseable(_)) => {
                Err(ReconcileArgumentsError::Unparseable(UnparseableSide::Right))
            }
            (ClassifiedArguments::Parsed(left), ClassifiedArguments::Parsed(right)) => {
                if left == right {
                    Ok(left)
                } else {
                    Err(ReconcileArgumentsError::Mismatch)
                }
            }
        }
    }
}

impl From<serde_json::Value> for FunctionCallArguments {
    /// Encode already-parsed arguments (Rig's canonical tool-call form) in
    /// the wire's stringified-JSON spelling.
    fn from(value: serde_json::Value) -> Self {
        Self(value.to_string())
    }
}

impl Serialize for FunctionCallArguments {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for FunctionCallArguments {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // The wire spells arguments as a string; a non-string payload is
        // still a schema defect of the known `function_call` shape.
        String::deserialize(deserializer).map(Self)
    }
}

/// See [`OutputFunctionCall::id`]: only provider-native `fc` item IDs may be
/// sent back to the Responses API.
fn is_not_function_call_item_id(id: &str) -> bool {
    !id.starts_with("fc_")
}

/// The status of a given tool.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    /// The item is still being produced.
    InProgress,
    /// The item is complete.
    Completed,
    /// The item stopped before it was complete.
    Incomplete,
}

/// An output message from OpenAI's Responses API.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputMessage {
    /// The message ID. Must be included when sending the message back to OpenAI
    pub id: String,
    /// The role (currently only Assistant is available as this struct is only created when receiving an LLM message as a response)
    pub role: OutputRole,
    /// The status of the response
    pub status: ResponseStatus,
    /// The actual message content
    pub content: Vec<AssistantContent>,
    /// Generation phase, such as `"final_answer"`, preserved for follow-up requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// The role of an output message.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputRole {
    /// The assistant.
    Assistant,
}

/// Text assistant content.
/// Note that the text type in comparison to the Completions API is actually `output_text` rather than `text`.
#[derive(Debug, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    /// Text.
    OutputText(OutputText),
    /// A refusal.
    Refusal {
        /// The refusal's text.
        refusal: String,
    },
    /// A nested message part retained for same-wire replay.
    #[serde(untagged)]
    Unknown(Value),
}

impl<'de> Deserialize<'de> for AssistantContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match content_part_tag::<D::Error>(&value)? {
            "output_text" => {
                // OutputText flattens sibling fields; the tag belongs to the enum.
                let mut value = value;
                if let Some(fields) = value.as_object_mut() {
                    json_utils::remove_preserving_order(fields, "type");
                }
                serde_json::from_value(value)
                    .map(Self::OutputText)
                    .map_err(serde::de::Error::custom)
            }
            "refusal" => {
                #[derive(Deserialize)]
                struct Fields {
                    refusal: String,
                }
                let fields: Fields =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self::Refusal {
                    refusal: fields.refusal,
                })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

/// Responses `output_text` block with unmodeled sibling fields preserved as JSON.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct OutputText {
    /// The text.
    pub text: String,
    /// OpenAI's sibling keys, preserved verbatim for value-equal replay.
    /// The `Map` form (not `Option<Value>`) makes absence and the empty map
    /// one value, so a decoded bare block equals a request-assembled one.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub extras: Map<String, Value>,
}

impl OutputText {
    /// A bare text block, as request assembly emits (no wire extras).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            extras: Map::new(),
        }
    }
}

#[cfg(test)]
pub(crate) mod corrupt_frame_fixtures;
#[cfg(test)]
mod echoed_turn_tests;
#[cfg(test)]
pub(crate) mod fidelity_fixtures;
#[cfg(test)]
mod fold_contract_tests;
#[cfg(test)]
mod fold_edge_tests;
#[cfg(test)]
mod serde_fixed_point_tests;
#[cfg(test)]
mod trailing_reasoning_tests;

#[cfg(test)]
mod tests;
