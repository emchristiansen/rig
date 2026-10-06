//! Typed completion values returned by observed Responses calls.

use crate::message::{AdditionalParams, Image, optional_additional_params};
use serde::{Deserialize, Serialize};

/// Normalized generation ending. Unmapped provider values remain in [`Self::Other`].
/// Failure statuses may accompany parseable output; callers must decide whether
/// such output is usable rather than treating every response as successful.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of the response.
    Stop,
    /// The response hit the output-token limit.
    Length,
    /// The model stopped to call one or more tools.
    ToolCalls,
    /// The provider filtered the content.
    ContentFilter,
    /// A provider-specific reason outside the normalized vocabulary, carried
    /// verbatim in the provider's own wire spelling.
    Other(String),
}

impl FinishReason {
    /// Reconcile a natural stop with emitted tool calls.
    pub fn reconcile_with_output(self, has_tool_call: bool) -> Self {
        if has_tool_call && matches!(self, Self::Stop) {
            Self::ToolCalls
        } else {
            self
        }
    }
}
/// Retained response headers, ordered by header name.
pub type ProviderResponseHeaders = std::collections::BTreeMap<String, String>;
/// Assistant content and normalized completion metadata. The choice may be
/// empty, including for truncated or filtered turns. Provider-specific data is
/// available through [`Self::raw`] without retaining a concrete model type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Assistant content returned by the provider, possibly empty.
    pub choice: Vec<AssistantContent>,
    /// Tokens used during prompting and responding
    pub usage: Usage,
    /// Provider-issued assistant message ID. Response-wide IDs belong in
    /// [`Self::response_id`].
    #[serde(default)]
    pub message_id: Option<String>,
    /// Provider-issued response ID for telemetry and diagnostics.
    /// Must not be replayed as an assistant message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Request identifier from HTTP headers or SDK metadata, not the body's
    /// message or response ID. `None` when the provider reports none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Success-reply headers the dialect captures, from the reply that
    /// produced this response. See [`ProviderResponseHeaders`].
    #[serde(default, skip_serializing_if = "ProviderResponseHeaders::is_empty")]
    pub provider_response_headers: ProviderResponseHeaders,
    /// Reported finish reason, reconciled by the setters with tool-call output.
    /// Read through [`Self::finish_reason`].
    #[serde(default)]
    pub(crate) finish_reason: Option<FinishReason>,
    /// Stable descriptor name of the provider that produced this response, for
    /// example `"openai"`. Always populated, including for responses derived
    /// from a stream that ended before its terminal record.
    pub provider: String,
    /// Provider-reported model identifier for the response.
    ///
    /// This is the model named by the wire response, not the model that was
    /// requested; it is `None` when the provider reports no identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Serialized stream terminal usage and metadata for observed responses.
    /// The observation handle retains the original provider strings separately.
    /// Callers constructing a response supply this value; it does not override
    /// the normalized fields.
    pub raw: serde_json::Value,
}
impl CompletionResponse {
    /// Construct the response from the observed provider output.
    pub fn new(
        choice: Vec<AssistantContent>,
        usage: Usage,
        provider: impl Into<String>,
        raw: serde_json::Value,
    ) -> Self {
        Self {
            choice,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            provider_response_headers: Default::default(),
            finish_reason: None,
            provider: provider.into(),
            model: None,
            raw,
        }
    }
    /// The provider's normalized and reconciled finish reason.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }
    /// Record the finish reason, accounting for tool-call output.
    pub fn set_finish_reason(&mut self, reason: Option<FinishReason>) {
        let has_tool_call = self.choice.iter().any(|part| {
            matches!(
                part,
                AssistantContent::ToolCall(_) | AssistantContent::CustomToolCall(_)
            )
        });
        self.finish_reason = reason.map(|reason| reason.reconcile_with_output(has_tool_call));
    }
    /// Attach a finish reason.
    pub fn with_optional_finish_reason(mut self, value: Option<FinishReason>) -> Self {
        self.set_finish_reason(value);
        self
    }
    /// Attach the provider's message identity.
    pub fn with_optional_message_id(mut self, value: Option<String>) -> Self {
        self.message_id = value;
        self
    }
    /// Attach the provider's response identity.
    pub fn with_optional_response_id(mut self, value: Option<String>) -> Self {
        self.response_id = value;
        self
    }
    /// Attach the transport request identity.
    pub fn with_optional_provider_request_id(mut self, value: Option<String>) -> Self {
        self.provider_request_id = value;
        self
    }
    /// Attach the reported model.
    pub fn with_optional_model(mut self, value: Option<String>) -> Self {
        self.model = value;
        self
    }
    /// Attach the retained response headers.
    pub fn with_provider_response_headers(
        mut self,
        value: std::collections::BTreeMap<String, String>,
    ) -> Self {
        self.provider_response_headers = value;
        self
    }
}
///
/// A counter the provider did not send is `None`; a reported zero is
/// `Some(0)`. Serialized as the same keys, absent when `None`.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input ("prompt") tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// The number of output ("completion") tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// We store this separately as some providers may only report one number
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// The number of input tokens read from a provider-managed cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// The number of input tokens written to a provider-managed cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// The number of tool-use prompt tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_prompt_tokens: Option<u64>,
    /// The number of tokens spent on internal reasoning / "thoughts" by reasoning-capable
    /// models (e.g. Gemini thinking, Anthropic extended thinking, OpenAI o-series).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Whether the provider reported any counter at all.
    pub fn is_reported(&self) -> bool {
        *self != Self::default()
    }
}
/// Assistant text, tool calls, reasoning or images.
/// Deserialization requires the lowercase `type` tag.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    /// Plain assistant text.
    Text(Text),
    /// Function tool call requested by the assistant: JSON arguments.
    ToolCall(ToolCall),
    /// Custom tool call requested by the assistant: verbatim, non-JSON input
    /// (the OpenAI Responses `custom_tool_call` item). Serialized with the
    /// `"type": "customtoolcall"` tag.
    ///
    /// A distinct variant rather than a kind of [`ToolFunction::arguments`],
    /// so raw input can never be read as JSON arguments and a JSON value can
    /// never be mistaken for raw input.
    CustomToolCall(CustomToolCall),
    /// Structured reasoning emitted by the assistant.
    Reasoning(Reasoning),
    /// Image content emitted by the assistant.
    Image(Image),
    /// An output item rig does not model, kept whole so it can be replayed
    /// to the wire that issued it or refused by name elsewhere. Serialized
    /// with the `"type": "provider_item"` tag.
    #[serde(rename = "provider_item")]
    ProviderItem(ProviderItem),
}

/// An opaque provider output item with its issuing provider preserved.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProviderItem {
    /// The item as the provider sent it.
    pub item: serde_json::Value,
    /// The service that issued this item, with the same meaning as
    /// [`Reasoning::provider`]. `None` is unknown provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

impl ProviderItem {
    /// An item issued by `provider`.
    pub fn new(item: serde_json::Value, provider: impl Into<String>) -> Self {
        Self {
            item,
            provider: Some(provider.into()),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
/// A typed reasoning block used by providers that emit structured thinking data.
pub enum ReasoningContent {
    /// Plain reasoning text with an optional provider signature.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Provider-encrypted reasoning payload.
    Encrypted(String),
    /// Redacted reasoning payload preserved as opaque data.
    Redacted { data: String },
    /// Provider-generated reasoning summary text.
    Summary(String),
    /// An opaque provider summary part retained for its original wire.
    OpaqueSummary(serde_json::Value),
    /// An opaque provider reasoning-content part retained for its original wire.
    OpaqueContent(serde_json::Value),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
/// Assistant reasoning payload with an optional provider-supplied identifier.
pub struct Reasoning {
    /// Provider reasoning identifier, when supplied by the upstream API.
    pub id: Option<String>,
    /// Ordered reasoning content blocks.
    pub content: Vec<ReasoningContent>,
    /// The service that issued this reasoning; `None` means unknown provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// Text with optional provider metadata under the named `additional_params` key.
/// Unknown sibling fields are ignored on decode, not captured for replay.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    /// Text content.
    pub text: String,
    /// Provider-specific text fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

impl Text {
    /// Construct a new text block with no provider-specific fields.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            additional_params: None,
        }
    }

    /// Returns the inner text string.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { text, .. } = self;
        write!(f, "{text}")
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MintKind {
    /// Anonymous reasoning assemblies, including constant-id wires and
    /// Responses output items without provider IDs.
    Reasoning,
    /// Opaque reasoning payloads without provider IDs. A separate kind prevents
    /// encrypted blocks from replacing accumulated reasoning text.
    EncryptedReasoning,
    /// Content blocks on index-as-id wires (anthropic, bedrock).
    Block,
    /// OpenAI Responses `output_index` fallback for opaque output items
    /// without provider IDs.
    Output,
    /// Tool-call fragments whose wire omits the tool-call id.
    Tool,
    /// Text blocks opened by a bare `Message` on wires that never announce
    /// text-block boundaries.
    Text,
    /// OpenAI Responses message content parts that cannot take the message's
    /// wire id: refusals, parts after the first and parts of an id-less
    /// message. Each is its own text block, minted in stream order, so a
    /// refusal or an opaque part is never merged into the text beside it.
    /// The serialized kind keeps its historical name.
    Refusal,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BlockId {
    /// An identifier the provider put on the wire.
    Wire(String),
    /// A key rig minted at a stream boundary because the wire supplied none.
    ///
    /// Indices restart for each stream. Cross-turn maps must pair this key
    /// with a turn identifier.
    Minted {
        /// The subsystem that minted this key.
        kind: MintKind,
        /// Per-stream counter or unsigned wire index.
        index: u64,
    },
}
impl MintKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::EncryptedReasoning => "encrypted_reasoning",
            Self::Block => "block",
            Self::Output => "output",
            Self::Tool => "tool",
            Self::Text => "text",
            Self::Refusal => "refusal",
        }
    }
    fn parse_name(name: &str) -> Option<Self> {
        [
            Self::Reasoning,
            Self::EncryptedReasoning,
            Self::Block,
            Self::Output,
            Self::Tool,
            Self::Text,
            Self::Refusal,
        ]
        .into_iter()
        .find(|kind| kind.as_str() == name)
    }
}

impl Serialize for BlockId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Wire(wire) => serializer.serialize_str(&format!("wire:{wire}")),
            Self::Minted { kind, index } => {
                serializer.serialize_str(&format!("minted:{}:{index}", kind.as_str()))
            }
        }
    }
}

impl<'de> Deserialize<'de> for BlockId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let text = String::deserialize(deserializer)?;
        if let Some(wire) = text.strip_prefix("wire:") {
            return Ok(Self::Wire(wire.to_owned()));
        }
        if let Some(rest) = text.strip_prefix("minted:")
            && let Some((kind, index)) = rest.rsplit_once(':')
        {
            let kind = MintKind::parse_name(kind)
                .ok_or_else(|| D::Error::custom(format!("unknown mint kind `{kind}`")))?;
            let index = index
                .parse::<u64>()
                .map_err(|_| D::Error::custom(format!("invalid mint index `{index}`")))?;
            return Ok(Self::Minted { kind, index });
        }
        Err(D::Error::custom(format!(
            "a block id is `wire:<id>` or `minted:<kind>:<index>`, got `{text}`"
        )))
    }
}

/// Error when adopting an empty explicit tool-call identifier.
/// Represent absent provider identity with [`ToolCall::provider`] set to `None`,
/// and an absent correlation identity with [`ToolCallId::minted`].
#[derive(Debug, thiserror::Error)]
#[error("a tool-call identifier cannot be the empty string; absence is `None` or a minted id")]
pub struct EmptyToolCallId;

/// A completion-local tool correlation identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "ToolCallIdWire", into = "ToolCallIdWire")]
pub struct ToolCallId(ToolCallIdWire);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "origin", content = "id", rename_all = "snake_case")]
enum ToolCallIdWire {
    Explicit(String),
    Generated(BlockId),
}

impl ToolCallId {
    /// Adopt a nonempty explicit correlation identity.
    pub fn new(id: impl Into<String>) -> Option<Self> {
        let id = id.into();
        (!id.is_empty()).then_some(Self(ToolCallIdWire::Explicit(id)))
    }
    /// Mint a completion-local correlation identity distinct from explicit IDs.
    pub fn minted(index: u64) -> Self {
        Self(ToolCallIdWire::Generated(BlockId::Minted {
            kind: MintKind::Tool,
            index,
        }))
    }
    /// Use an explicit identity when present, otherwise mint one.
    pub fn new_or_minted(id: impl Into<String>, index: u64) -> Self {
        Self::new(id).unwrap_or_else(|| Self::minted(index))
    }
    /// Prefer the provider correlator, preserving the supplied fallback identity.
    pub fn for_provider_or(provider: Option<&ProviderCallId>, minted: Self) -> Self {
        provider
            .and_then(|provider| Self::new(provider.call_id.clone()))
            .unwrap_or(minted)
    }
    /// The explicit identity, when one was supplied.
    pub fn explicit(&self) -> Option<&str> {
        match &self.0 {
            ToolCallIdWire::Explicit(id) => Some(id),
            ToolCallIdWire::Generated(_) => None,
        }
    }
    /// Whether this identity was generated locally.
    pub fn is_generated(&self) -> bool {
        matches!(self.0, ToolCallIdWire::Generated(_))
    }
}
impl TryFrom<ToolCallIdWire> for ToolCallId {
    type Error = EmptyToolCallId;

    fn try_from(id: ToolCallIdWire) -> Result<Self, Self::Error> {
        match id {
            ToolCallIdWire::Explicit(id) => Self::new(id).ok_or(EmptyToolCallId),
            generated @ ToolCallIdWire::Generated(_) => Ok(Self(generated)),
        }
    }
}

impl From<ToolCallId> for ToolCallIdWire {
    fn from(id: ToolCallId) -> Self {
        id.0
    }
}

/// Wire shape for [`ProviderCallId`], so deserialization enforces the
/// non-empty `call_id` invariant.
#[derive(Deserialize)]
struct ProviderCallIdWire {
    call_id: String,
    #[serde(default)]
    item_id: Option<String>,
}

/// Provider-issued identifiers for replay. Single-ID protocols use `call_id`;
/// dual-ID protocols also use `item_id`. Keep each identifier in its protocol slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ProviderCallIdWire")]
pub struct ProviderCallId {
    /// The call-correlation identifier the provider expects echoed back.
    pub call_id: String,
    /// The output-item id issued alongside `call_id` on dual-identifier
    /// wires (OpenAI Responses `fc_…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

impl ProviderCallId {
    /// Adopt a provider-issued call identifier. `None` for the empty
    /// string: absence is not an id.
    pub fn new(call_id: impl Into<String>) -> Option<Self> {
        let call_id = call_id.into();
        if call_id.is_empty() {
            None
        } else {
            Some(Self {
                call_id,
                item_id: None,
            })
        }
    }

    /// Attach the dual-wire output-item id (empty strings are dropped).
    pub fn with_item_id(mut self, item_id: impl Into<String>) -> Self {
        let item_id = item_id.into();
        self.item_id = (!item_id.is_empty()).then_some(item_id);
        self
    }

    /// Uses a nonempty `call_id` as the correlator and `tool_id` as the item ID.
    /// Without a nonempty `call_id`, uses a nonempty `tool_id` as the correlator.
    /// Returns `None` if neither supplies an identifier. For dual-ID protocols
    /// that forbid this fallback, use `new` and attach the item ID separately.
    pub fn from_optional_wire(
        call_id: Option<String>,
        tool_id: Option<String>,
    ) -> Option<Self> {
        let call_id = call_id.filter(|call_id| !call_id.is_empty());
        match (call_id, tool_id) {
            (Some(call_id), tool_id) => Self::new(call_id).map(|provider| match tool_id {
                Some(tool_id) => provider.with_item_id(tool_id),
                None => provider,
            }),
            (None, Some(tool_id)) => Self::new(tool_id),
            (None, None) => None,
        }
    }
}

impl TryFrom<ProviderCallIdWire> for ProviderCallId {
    type Error = EmptyToolCallId;

    fn try_from(wire: ProviderCallIdWire) -> Result<Self, Self::Error> {
        let Some(provider) = Self::new(wire.call_id) else {
            return Err(EmptyToolCallId);
        };
        Ok(match wire.item_id {
            Some(item_id) => provider.with_item_id(item_id),
            None => provider,
        })
    }
}

/// Describes a tool call with an id and function to call, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    /// Rig's correlation handle. Always present; minted when the provider
    /// issued none.
    pub id: ToolCallId,
    /// Provider-issued replay identifiers, or `None` when none were supplied.
    /// Local correlation handles do not establish provider provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderCallId>,
    /// Function name and JSON arguments requested by the model.
    pub function: ToolFunction,
    /// Opaque provider signature preserved for replay. Rig does not verify it.
    #[serde(default)]
    pub signature: Option<String>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    #[serde(default)]
    pub additional_params: Option<serde_json::Value>,
}
/// Describes a tool function to call with a name and arguments, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunction {
    /// Tool/function name to invoke.
    pub name: String,
    /// The namespace the provider qualified this callable with, verbatim;
    /// `None` when the provider sent none. Omitted from serialization when
    /// absent.
    ///
    /// Carried exactly as the provider item spelled it. No spelling (`""`,
    /// `"functions"`) is normalized to absence here: whether a destination
    /// treats one as its default namespace is for that destination to decide,
    /// and a destination must preserve the qualifier or refuse the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// JSON arguments for the tool/function.
    pub arguments: serde_json::Value,
}

impl ToolFunction {
    /// Create a tool function call payload with no namespace.
    pub fn new(name: String, arguments: serde_json::Value) -> Self {
        Self {
            name,
            namespace: None,
            arguments,
        }
    }

    /// Attach (or clear) the namespace the provider qualified this callable with.
    pub fn with_namespace(mut self, namespace: Option<String>) -> Self {
        self.namespace = namespace;
        self
    }
}
/// A custom tool call with verbatim input.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CustomToolCall {
    /// Rig's correlation handle, as on [`ToolCall::id`].
    pub id: ToolCallId,
    /// Provider-issued replay identifiers (`call_id`, and the output-item id
    /// on dual-identifier wires), or `None` when none were supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderCallId>,
    /// Tool name to invoke.
    pub name: String,
    /// The namespace the provider qualified this callable with, verbatim;
    /// omitted from serialization when absent. See [`ToolFunction::namespace`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The model's verbatim input.
    pub input: String,
}
