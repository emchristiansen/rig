use super::observed_types::{self as message, Text};
use super::{
    OpenAIServiceTier, OutputRole, Reasoning, ResponseObject, ResponseStatus, ToolStatus,
    TruncationStrategy,
};
use crate::json_utils;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, ops::Add};

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

impl From<&ResponsesUsage> for message::Usage {
    fn from(usage: &ResponsesUsage) -> Self {
        message::Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.total_tokens),
            cached_input_tokens: usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
            reasoning_tokens: usage
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            ..Default::default()
        }
    }
}

impl From<ResponsesUsage> for message::Usage {
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
    /// Unknown fields, unknown values and null-valued members inside the object
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
            additional_parameters: AdditionalParameters::from_response_metadata(
                response.metadata,
            ),
        })
    }
}
/// Additional parameters for the completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
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

    pub fn to_json(self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|_| serde_json::Value::Object(Map::new()))
    }
}
/// The model output format configuration.
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextConfig {
    pub format: TextFormat,
}
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    JsonSchema(StructuredOutputsInput),
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

/// Additional response fields requested through [`AdditionalParameters::include`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Include {
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    #[serde(rename = "computer_call.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
}
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ResponsesToolDefinition {
    /// The type of tool.
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Parameters - this should be a JSON schema. Strict function tools must use OpenAI's supported strict schema subset.
    #[serde(default, skip_serializing_if = "is_json_null")]
    pub parameters: serde_json::Value,
    /// Whether the provider reported strict mode.
    ///
    /// Always serialized: the Responses API treats an omitted `strict` as "attempt strict
    /// mode", so `false` must reach the wire for non-strict tools to actually be non-strict.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub strict: bool,
    /// Tool description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Additional provider-specific configuration for hosted tools.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub config: Map<String, Value>,
}

fn is_json_null(value: &Value) -> bool {
    value.is_null()
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    SummaryText {
        text: String,
    },
    /// An unmodeled summary part, retained verbatim.
    #[serde(untagged)]
    Unknown(Value),
}

impl ReasoningSummary {
    fn new(input: &str) -> Self {
        Self::SummaryText {
            text: input.to_owned(),
        }
    }

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
    ReasoningText {
        text: String,
    },
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

/// A Responses output item. Unrecognized types, including hosted tools, decode
/// to [`Output::Unknown`] with their JSON value preserved. Malformed known
/// types fail deserialization.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
    /// A custom (grammar) tool call, whose `input` is verbatim text rather
    /// than JSON arguments.
    CustomToolCall(OutputCustomToolCall),
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        content: Vec<ReasoningTextContent>,
        encrypted_content: Option<String>,
        /// The upstream's signature over the reasoning text, when a gateway
        /// relays one (OpenRouter for Claude).
        signature: Option<String>,
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
                map.remove("type");
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
    pub arguments: FunctionCallArguments,
    pub call_id: String,
    pub name: String,
    /// The namespace qualifying this callable, verbatim; absent when the
    /// provider sent none.
    ///
    /// Present in both directions because this struct is shared by
    /// function-call decoding and serialization; dropping
    /// it on either side would strip the qualifier before any consumer saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
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
    /// The correlator the corresponding custom tool result echoes back.
    pub call_id: String,
    /// The custom tool's name.
    pub name: String,
    /// The namespace qualifying this callable, verbatim; absent when the
    /// provider sent none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The model's verbatim input, preserved byte for byte.
    pub input: String,
    /// The item status, when the provider stated one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Raw wire arguments, classified only when consumed.
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
    Left,
    Right,
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
    /// Parse the raw wire string into JSON arguments. An empty or whitespace-only string is a
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
    pub fn reconcile(
        &self,
        other: &Self,
    ) -> Result<serde_json::Value, ReconcileArgumentsError> {
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

/// Text assistant content.
/// Note that the text type in comparison to the Completions API is actually `output_text` rather than `text`.
#[derive(Debug, Serialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    OutputText(OutputText),
    Refusal {
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
                    fields.remove("type");
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
    pub text: String,
    /// OpenAI's sibling keys, preserved verbatim for value-equal replay.
    /// The `Map` form (not `Option<Value>`) makes absence and the empty map
    /// one value, so a decoded bare block equals a request-assembled one.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub extras: Map<String, Value>,
}

/// Builds reasoning blocks in summary, text, encrypted-content order.
/// Empty encrypted content contributes no block. A signature signs the
/// reasoning text, so it rides on the last text block, or on an empty one
/// when the item carried no text.
pub(crate) fn reasoning_content_blocks(
    summary: Vec<ReasoningSummary>,
    content: Vec<ReasoningTextContent>,
    encrypted_content: Option<String>,
    signature: Option<String>,
) -> Vec<message::ReasoningContent> {
    let mut blocks = summary
        .into_iter()
        .map(|summary| match summary {
            ReasoningSummary::SummaryText { text } => message::ReasoningContent::Summary(text),
            ReasoningSummary::Unknown(value) => message::ReasoningContent::OpaqueSummary(value),
        })
        .collect::<Vec<_>>();

    blocks.extend(content.into_iter().map(|part| match part {
        ReasoningTextContent::ReasoningText { text } => message::ReasoningContent::Text {
            text,
            signature: None,
        },
        ReasoningTextContent::Unknown(value) => message::ReasoningContent::OpaqueContent(value),
    }));
    if let Some(signature) = signature {
        match blocks.iter_mut().rev().find_map(|block| match block {
            message::ReasoningContent::Text { signature, .. } => Some(signature),
            _ => None,
        }) {
            Some(slot) => *slot = Some(signature),
            None => blocks.push(message::ReasoningContent::Text {
                text: String::new(),
                signature: Some(signature),
            }),
        }
    }

    if let Some(encrypted_content) = encrypted_content.filter(|content| !content.is_empty()) {
        blocks.push(message::ReasoningContent::Encrypted(encrypted_content));
    }

    blocks
}

const OPENAI_RESPONSES_EXTRAS_KEY: &str = "openai_responses";
const OPENAI_RESPONSES_PHASE_KEY: &str = "phase";
const OPENAI_RESPONSES_PART_KEY: &str = "openai_responses_part";
const REFUSAL_PART_KIND: &str = "refusal";

/// The retained JSON value of an opaque Responses message part, when `params` marks one.
pub(crate) fn opaque_message_part(
    params: Option<&crate::message::AdditionalParams>,
) -> Option<&Value> {
    let marker = params?.wire_extras(OPENAI_RESPONSES_PART_KEY)?;
    (marker.get("kind").and_then(Value::as_str) == Some("opaque"))
        .then(|| marker.get("value"))
        .flatten()
}

/// The metadata that marks a text block as a refusal.
pub(crate) fn refusal_marker() -> Option<crate::message::AdditionalParams> {
    crate::message::AdditionalParams::from_entries(Some((
        OPENAI_RESPONSES_PART_KEY,
        serde_json::json!({ "kind": REFUSAL_PART_KIND }),
    )))
}

/// Whether a text block is marked as a refusal.
pub(crate) fn is_refusal(additional_params: Option<&crate::message::AdditionalParams>) -> bool {
    additional_params
        .and_then(|params| params.wire_extras(OPENAI_RESPONSES_PART_KEY))
        .and_then(|part| part.get("kind"))
        .and_then(Value::as_str)
        == Some(REFUSAL_PART_KIND)
}

/// Record an output message's `phase` on a text block's own-wire extras so
/// the follow-up request can re-send it.
pub(crate) fn stamp_phase(text: &mut Text, phase: Option<&str>) {
    let Some(phase) = phase else {
        return;
    };
    // Keep every other entry — a refusal block's part marker among them.
    let mut entries = match text
        .additional_params
        .take()
        .map(crate::message::AdditionalParams::into_value)
    {
        Some(Value::Object(entries)) => entries,
        _ => Map::new(),
    };
    let mut extras = match entries.remove(OPENAI_RESPONSES_EXTRAS_KEY) {
        Some(Value::Object(extras)) => extras,
        _ => Map::new(),
    };
    extras.insert(
        OPENAI_RESPONSES_PHASE_KEY.to_string(),
        Value::String(phase.to_string()),
    );
    entries.insert(
        OPENAI_RESPONSES_EXTRAS_KEY.to_string(),
        Value::Object(extras),
    );
    text.additional_params = crate::message::AdditionalParams::new(entries);
}

/// Converts output text, a refusal or an opaque part to a Rig text block, retaining nonempty
/// output-text extras under the Responses key. A refusal is marked with
/// [`refusal_marker`], so it stays distinct from output text.
pub(crate) fn text_block(value: AssistantContent) -> Text {
    match value {
        AssistantContent::Unknown(value) => Text {
            text: String::new(),
            additional_params: crate::message::AdditionalParams::from_entries(Some((
                OPENAI_RESPONSES_PART_KEY,
                serde_json::json!({"kind": "opaque", "value": value}),
            ))),
        },
        AssistantContent::Refusal { refusal } => Text {
            text: refusal,
            additional_params: refusal_marker(),
        },
        // Keep this destructuring exhaustive so new wire fields force an
        // explicit capture-or-drop decision.
        AssistantContent::OutputText(OutputText { text, extras }) => {
            // Empty metadata must not change replayed request bytes.
            let extras: Map<String, Value> = extras
                .into_iter()
                .filter(|(_, value)| {
                    !(value.is_null()
                        || value.as_array().is_some_and(Vec::is_empty)
                        || value.as_object().is_some_and(Map::is_empty))
                })
                .collect();
            Text {
                text,
                additional_params: crate::message::AdditionalParams::from_entries(
                    (!extras.is_empty())
                        .then_some((OPENAI_RESPONSES_EXTRAS_KEY, Value::Object(extras))),
                ),
            }
        }
    }
}
