use crate::error::ProviderError;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};
use thiserror::Error;

/// A provider-agnostic chat message.
///
/// Messages are role-tagged and may contain one or many content items, including
/// text, images, audio, documents, tool calls, and tool results. Provider modules
/// are responsible for translating these generic messages into provider-native
/// request bodies. That conversion may be lossy when a provider does not support
/// a particular content type.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// System message containing instruction text.
    System { content: String },

    /// User message containing one or more content types defined by `UserContent`.
    User { content: Vec<UserContent> },

    /// Assistant message containing one or more content types defined by `AssistantContent`.
    Assistant {
        /// Provider-assigned assistant message ID, when available.
        id: Option<String>,
        content: Vec<AssistantContent>,
    },
}

/// Shared error text for an invalid empty response choice.
/// Provider decoders must exempt legal empty outcomes, including recognized
/// output truncation, before calling [`require_non_empty_response`].
pub const EMPTY_RESPONSE_ERROR: &str = "Response contained no message or tool call (empty)";

/// Returns `items` unchanged unless the list is empty, then calls `error` once.
/// Does not inspect individual items: empty text can carry replay signatures.
/// Request conversions that discard content must validate the converted list
/// when their wire requires at least one block.
pub fn require_non_empty<T, E>(items: Vec<T>, error: impl FnOnce() -> E) -> Result<Vec<T>, E> {
    if items.is_empty() {
        return Err(error());
    }
    Ok(items)
}

/// Returns a response error using [`EMPTY_RESPONSE_ERROR`] for an empty list.
/// Callers must handle provider-legal empty outcomes before invoking this guard.
pub fn require_non_empty_response<T>(items: Vec<T>) -> Result<Vec<T>, ProviderError> {
    require_non_empty(items, || {
        ProviderError::Response(EMPTY_RESPONSE_ERROR.to_owned())
    })
}

/// Returns `None` for an empty list or `Some(items)` otherwise.
/// Individual items are not inspected.
pub fn non_empty<T>(items: Vec<T>) -> Option<Vec<T>> {
    if items.is_empty() { None } else { Some(items) }
}

/// Concatenates reasoning, text, and trailing content in that order without
/// dropping items. Each group's input order is preserved.
pub fn ordered_assistant_content(
    reasoning_items: impl IntoIterator<Item = Reasoning>,
    text_items: impl IntoIterator<Item = AssistantContent>,
    trailing_items: impl IntoIterator<Item = AssistantContent>,
) -> Vec<AssistantContent> {
    let mut content_items = reasoning_items
        .into_iter()
        .map(AssistantContent::Reasoning)
        .collect::<Vec<_>>();
    content_items.extend(text_items);
    content_items.extend(trailing_items);
    content_items
}

/// Returns whether the choice contains no nonempty text, tool call, or image.
/// Reasoning alone is not an answer, even when retained in history.
pub fn turn_delivered_no_answer(choice: &[AssistantContent]) -> bool {
    !choice.iter().any(|content| match content {
        // Real text is an answer; an empty block delivers nothing.
        AssistantContent::Text(text) => !text.text.is_empty(),
        AssistantContent::ToolCall(_) => true,
        AssistantContent::CustomToolCall(_) => true,
        AssistantContent::Image(_) => true,
        // Scratch work and opaque provider items are not an answer.
        AssistantContent::Reasoning(_) | AssistantContent::ProviderItem(_) => false,
    })
}

/// Groups streamed choices as reasoning, text and provider items, tool
/// calls, then images, preserving order within each group. Choices without reasoning or tool calls
/// retain their original order.
pub fn canonical_streamed_choice(choice: Vec<AssistantContent>) -> Vec<AssistantContent> {
    let regroup = choice.iter().any(|part| {
        matches!(
            part,
            AssistantContent::Reasoning(_)
                | AssistantContent::ToolCall(_)
                | AssistantContent::CustomToolCall(_)
        )
    });
    if !regroup {
        return choice;
    }
    let mut reasoning = Vec::new();
    let mut text = Vec::new();
    let mut calls = Vec::new();
    let mut images = Vec::new();
    for part in choice {
        match part {
            AssistantContent::Reasoning(block) => reasoning.push(block),
            // Provider items keep their stream order relative to the text.
            AssistantContent::Text(_) | AssistantContent::ProviderItem(_) => text.push(part),
            AssistantContent::ToolCall(_) | AssistantContent::CustomToolCall(_) => {
                calls.push(part);
            }
            AssistantContent::Image(_) => images.push(part),
        }
    }
    ordered_assistant_content(reasoning, text, calls.into_iter().chain(images))
}

/// User text, tool results, or media. Supported source kinds and media types
/// depend on the target provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    /// Plain text user content.
    Text(Text),
    /// Result of a tool call returned as user-visible context to the model.
    ToolResult(ToolResult),
    /// Image content.
    Image(Image),
    /// Audio content.
    Audio(Audio),
    /// Video content.
    Video(Video),
    /// Document content.
    Document(Document),
}

/// Assistant text, tool calls, reasoning, or images.
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

/// An opaque output item a provider returned that rig does not model (a
/// hosted-tool result, a `compaction` item), kept whole in history.
///
/// Only a wire that can carry an opaque output item back as input replays
/// it, verbatim, and only when [`Self::replayable_to`] that wire: issued
/// there, or of unknown provenance (`provider` is `None`). Every other wire
/// refuses it by name with [`UnreplayableProviderItem`], and so does a
/// replaying wire for an item another issuer produced; none is ever
/// dropped silently. To switch providers deliberately, drop the items the
/// destination cannot replay with [`retain_replayable_provider_items`].
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

    /// The item's `type`, when it names one.
    pub fn item_type(&self) -> Option<&str> {
        self.item.get("type").and_then(serde_json::Value::as_str)
    }

    /// Whether this item may be replayed to `issuer`, exactly as
    /// [`Reasoning::replayable_to`] decides it.
    pub fn replayable_to(&self, issuer: &str) -> bool {
        issued_to(self.provider.as_deref(), issuer)
    }
}

/// Whether content issued by `provider` may be replayed to `issuer`: it was
/// issued there, or its provenance is unknown. An `issuer` ending in `/`
/// names a family and accepts every issuer under it. The one rule for
/// reasoning and provider items alike.
fn issued_to(provider: Option<&str>, issuer: &str) -> bool {
    provider.is_none_or(|provider| {
        provider == issuer || (issuer.ends_with('/') && provider.starts_with(issuer))
    })
}

/// Drop provider items none of `issuers` could replay from `history`.
///
/// Never applied implicitly: an item a destination cannot replay is refused
/// by name when encoded, so calling this is a caller's deliberate choice to
/// drop them, for example before switching providers. Items of unknown
/// provenance are kept, as [`retain_replayable_reasoning`] keeps reasoning.
/// An assistant message left empty by the filter is removed.
pub fn retain_replayable_provider_items(history: &mut Vec<Message>, issuers: &[&str]) {
    history.retain_mut(|message| {
        let Message::Assistant { content, .. } = message else {
            return true;
        };
        let before = content.len();
        content.retain(|part| match part {
            AssistantContent::ProviderItem(item) => {
                issuers.iter().any(|issuer| item.replayable_to(issuer))
            }
            _ => true,
        });
        before == 0 || !content.is_empty()
    });
}

/// A provider item reached a wire that cannot replay it: the wire carries no
/// opaque output item as input, or the item came from another issuer.
///
/// Converts into [`MessageError`], [`crate::error::EncodeError`], and
/// therefore [`ProviderError::Request`]. Drop such items deliberately with
/// [`retain_replayable_provider_items`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error(
    "{wire} cannot replay the `{}` item issued by {}; drop it with retain_replayable_provider_items to switch providers deliberately",
    item_type.as_deref().unwrap_or("untyped"),
    issuer.as_deref().unwrap_or("an unknown provider")
)]
pub struct UnreplayableProviderItem {
    /// The destination wire.
    pub wire: &'static str,
    /// The item's issuer, when known.
    pub issuer: Option<String>,
    /// The item's `type`, when it names one.
    pub item_type: Option<String>,
}

impl UnreplayableProviderItem {
    /// The refusal for replaying `item` to `wire`.
    pub fn new(wire: &'static str, item: &ProviderItem) -> Self {
        Self {
            wire,
            issuer: item.provider.clone(),
            item_type: item.item_type().map(str::to_owned),
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
    /// The service that issued this reasoning, stamped when a completion is
    /// decoded: the wire's [`Wire::name`](crate::wire::Wire::name), or the
    /// model vendor where several transports serve the same models (Claude
    /// on Bedrock records `anthropic`). Signatures, encrypted and redacted
    /// payloads and reasoning ids only mean something to their issuer, so a
    /// request elsewhere omits known reasoning ([`retain_replayable_reasoning`]).
    /// Opaque parts remain until the encoder can reject unsupported replay.
    /// `None` is unknown provenance (reasoning built by hand, or history
    /// serialized before provenance existed) and is replayed as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// Drop reasoning none of `issuers` issued from `history`, before a request
/// that accepts reasoning from `issuers` is encoded. Signatures, encrypted
/// and redacted payloads and reasoning ids only mean something to the
/// service that issued them. Reasoning containing opaque parts is retained so
/// the encoder can refuse unsupported replay instead of silently losing it.
/// Known-only foreign reasoning is dropped; unknown provenance is kept. An
/// assistant turn that held nothing else is dropped with it rather than sent
/// empty, which leaves the user turns around it adjacent.
pub fn retain_replayable_reasoning(history: &mut Vec<Message>, issuers: &[&str]) {
    history.retain_mut(|message| {
        let Message::Assistant { content, .. } = message else {
            return true;
        };
        let before = content.len();
        content.retain(|part| match part {
            AssistantContent::Reasoning(reasoning) => {
                reasoning.has_opaque_parts()
                    || issuers.iter().any(|issuer| reasoning.replayable_to(issuer))
            }
            _ => true,
        });
        before == 0 || !content.is_empty()
    });
}

impl Reasoning {
    /// Whether replay requires preservation of an unmodeled provider part.
    pub fn has_opaque_parts(&self) -> bool {
        self.content.iter().any(|part| {
            matches!(
                part,
                ReasoningContent::OpaqueSummary(_) | ReasoningContent::OpaqueContent(_)
            )
        })
    }

    /// Create a new reasoning item from a single item
    pub fn new(input: &str) -> Self {
        Self::new_with_signature(input, None)
    }

    /// Create a new reasoning item from a single text item and optional signature.
    pub fn new_with_signature(input: &str, signature: Option<String>) -> Self {
        Self {
            provider: None,
            id: None,
            content: vec![ReasoningContent::Text {
                text: input.to_string(),
                signature,
            }],
        }
    }

    /// Record the wire that issued this reasoning.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Whether this reasoning may be replayed to `issuer`: it was issued
    /// there, or its provenance is unknown. An `issuer` ending in `/` names a
    /// family and accepts every issuer under it (`openrouter/` accepts
    /// `openrouter/openai`).
    pub fn replayable_to(&self, issuer: &str) -> bool {
        issued_to(self.provider.as_deref(), issuer)
    }

    /// Set a provider reasoning ID.
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Create reasoning content from multiple text blocks.
    pub fn multi(input: Vec<String>) -> Self {
        Self {
            provider: None,
            id: None,
            content: input
                .into_iter()
                .map(|text| ReasoningContent::Text {
                    text,
                    signature: None,
                })
                .collect(),
        }
    }

    /// Create a redacted reasoning block.
    pub fn redacted(data: impl Into<String>) -> Self {
        Self {
            provider: None,
            id: None,
            content: vec![ReasoningContent::Redacted { data: data.into() }],
        }
    }

    /// Create an encrypted reasoning block.
    pub fn encrypted(data: impl Into<String>) -> Self {
        Self {
            provider: None,
            id: None,
            content: vec![ReasoningContent::Encrypted(data.into())],
        }
    }

    /// Create one reasoning block containing summary items.
    pub fn summaries(input: Vec<String>) -> Self {
        Self {
            provider: None,
            id: None,
            content: input.into_iter().map(ReasoningContent::Summary).collect(),
        }
    }

    /// Render reasoning as displayable text by joining text-like blocks with newlines.
    pub fn display_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|content| match content {
                ReasoningContent::Text { text, .. } => Some(text.as_str()),
                ReasoningContent::Summary(summary) => Some(summary.as_str()),
                ReasoningContent::Redacted { data } => Some(data.as_str()),
                ReasoningContent::Encrypted(_)
                | ReasoningContent::OpaqueSummary(_)
                | ReasoningContent::OpaqueContent(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the first text reasoning block, if present.
    pub fn first_text(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Return the first signature from text reasoning, if present.
    pub fn first_signature(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Text {
                signature: Some(signature),
                ..
            } => Some(signature.as_str()),
            _ => None,
        })
    }

    /// Return the first encrypted reasoning payload, if present.
    pub fn encrypted_content(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Encrypted(data) => Some(data.as_str()),
            _ => None,
        })
    }
}

/// Which kind of tool call a [`ToolResult`] answers.
///
/// A wire offering more than one result item shape chooses between them by
/// this value. The OpenAI Responses wire pairs `custom_tool_call` only with
/// `custom_tool_call_output`, so a result answering a [`CustomToolCall`] sent
/// back as a function output leaves that call unanswered and the turn
/// unpaired. Wires with a single result shape read nothing from it.
///
/// No `Default`, and no serde default on [`ToolResult::answers`], deliberately:
/// defaulting to `Function` would make the mispairing this type prevents the
/// silent outcome of forgetting to state a kind. Where the answered call is in
/// hand, derive the kind from it ([`ToolCall::answered_by`],
/// [`CustomToolCall::answered_by`]); a site that does not hold the call states
/// it through [`UserContent::tool_result_answering`].
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AnsweredToolCall {
    /// Answers a function call ([`AssistantContent::ToolCall`]), whose
    /// arguments were JSON.
    Function,
    /// Answers a custom tool call ([`AssistantContent::CustomToolCall`]),
    /// whose input was verbatim text.
    Custom,
}

/// Tool result content containing information about a tool call and it's resulting content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolResult {
    /// Correlation handle copied from the answered [`ToolCall::id`].
    pub call: ToolCallId,
    /// Provider-issued replay identifiers copied from the answered call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderCallId>,
    /// Executed tool name, which may differ from the model-requested name after
    /// hook repair. Required for provider replay independently of call identity.
    pub name: String,
    /// Which kind of call this answers, and therefore which result item a wire
    /// carrying more than one shape must emit for it. Required on
    /// deserialization: a result that does not say what it answers is
    /// rejected rather than guessed.
    pub answers: AnsweredToolCall,
    /// One or more content items produced by the tool.
    pub content: Vec<ToolResultContent>,
}

impl ToolResult {
    /// A non-empty candidate for a required wire call-ID slot: the exact
    /// provider handle when present, otherwise the local identity's wire hint.
    ///
    /// This single-item helper cannot reserve future provider IDs or pair
    /// repeated turns. Full request adapters must use
    /// [`ToolCallIds`](crate::providers::internal::tool_call_ids::ToolCallIds)
    /// to assign collision-free synthetic references consistently to both legs.
    ///
    /// Wires whose id slot is *optional* (Gemini REST, gRPC) must read
    /// [`ToolResult::provider`] directly instead: minted handles never
    /// travel upstream there.
    pub fn wire_call_id(&self) -> std::borrow::Cow<'_, str> {
        self.provider.as_ref().map_or_else(
            || self.call.wire_hint(),
            |provider| std::borrow::Cow::Borrowed(provider.call_id.as_str()),
        )
    }
}

/// Describes one typed item in a tool result.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolResultContent {
    /// Literal text. Providers must not reinterpret it as structured JSON.
    Text(Text),
    /// An image supplied explicitly by the tool.
    Image(Image),
    /// Structured JSON supplied explicitly by the tool runtime.
    Json {
        /// The structured value.
        value: serde_json::Value,
    },
}

impl ToolResultContent {
    /// Borrow literal text content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            Self::Image(_) | Self::Json { .. } => None,
        }
    }

    /// Borrow structured JSON content.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json { value } => Some(value),
            Self::Text(_) | Self::Image(_) => None,
        }
    }

    /// Deserialize JSON content into a typed value.
    ///
    /// Structured JSON is decoded directly. Literal text is parsed only because
    /// the caller explicitly requested JSON decoding, which supports transcripts
    /// recorded before structured tool output was preserved canonically. This
    /// helper never changes the content sent to a model or provider.
    pub fn deserialize_json<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        match self {
            Self::Json { value } => T::deserialize(value),
            Self::Text(text) => serde_json::from_str(&text.text),
            Self::Image(_) => Err(<serde_json::Error as serde::de::Error>::custom(
                "cannot decode image tool-result content as JSON",
            )),
        }
    }
}

/// Error when adopting an empty explicit tool-call identifier.
/// Represent absent provider identity with [`ToolCall::provider`] set to `None`.
#[derive(Debug, thiserror::Error)]
#[error("a tool-call identifier cannot be the empty string; absence is `None` or a minted id")]
pub struct EmptyToolCallId;

/// Rig's correlation identity for a tool call within one assistant completion.
///
/// Explicit handles and generated assembly keys occupy disjoint namespaces:
/// an explicit `tool-0` never equals a generated tool key at index zero. Provider
/// provenance is separate and lives only on [`ToolCall::provider`]. Applications
/// may also choose explicit handles without claiming provider provenance.
///
/// Generated positions restart for each completion. State spanning completions
/// must pair this identity with the owning turn or effect, or match call/result
/// occurrences chronologically. Results copy their answered call's identity.
///
/// Serialization preserves an explicit origin tag and rejects legacy bare
/// strings. Display is diagnostic text, not a provider handle or lookup key.
///
/// Keep the typed value as a map key and copy it into the corresponding result.
/// Use [`Self::explicit`] or [`Self::generated`] to inspect its origin; outbound
/// adapters separately assign protocol handles for complete call/result histories.
///
/// ```
/// use rig_core::message::ToolCallId;
///
/// let explicit = ToolCallId::new("tool-0").ok_or("empty handle")?;
/// let generated = ToolCallId::minted(0);
/// assert_ne!(explicit, generated);
/// assert_eq!(explicit.explicit(), Some("tool-0"));
/// assert!(generated.is_generated());
/// assert_eq!(
///     serde_json::to_value(&generated)?,
///     serde_json::json!({"origin": "generated", "id": "minted:tool:0"}),
/// );
/// let restored: ToolCallId = serde_json::from_value(serde_json::to_value(&generated)?)?;
/// assert_eq!(restored, generated);
/// assert!(serde_json::from_value::<ToolCallId>(serde_json::json!("tool-0")).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "ToolCallIdWire", into = "ToolCallIdWire")]
pub struct ToolCallId(ToolCallIdWire);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "origin", content = "id", rename_all = "snake_case")]
enum ToolCallIdWire {
    Explicit(String),
    Generated(crate::streaming::BlockId),
}

impl ToolCallId {
    /// Adopt a nonempty explicit correlation handle. This constructor alone
    /// does not claim that any provider issued it; see [`ToolCall::provider`].
    pub fn new(id: impl Into<String>) -> Option<Self> {
        let id = id.into();
        (!id.is_empty()).then_some(Self(ToolCallIdWire::Explicit(id)))
    }

    /// Generates a deterministic completion-local identity. Cross-turn maps
    /// must pair it with a turn identifier because indices restart each turn.
    pub fn minted(index: u64) -> Self {
        Self::from_block(&crate::streaming::BlockId::minted(
            crate::streaming::MintKind::Tool,
            index,
        ))
    }

    /// Derive an identity from the complete typed assembly key. A wire-shaped
    /// assembly key and a minted key with the same display text stay distinct.
    pub fn from_block(block: &crate::streaming::BlockId) -> Self {
        Self(ToolCallIdWire::Generated(block.clone()))
    }

    /// Adopt a nonempty explicit handle, otherwise generate at `index`.
    pub fn new_or_minted(id: impl Into<String>, index: u64) -> Self {
        Self::new(id).unwrap_or_else(|| Self::minted(index))
    }

    /// Derive an explicit handle from provider metadata, or retain the supplied
    /// generated identity when no nonempty provider call identifier exists.
    pub fn for_provider_or(provider: Option<&ProviderCallId>, minted: Self) -> Self {
        provider
            .and_then(|provider| Self::new(provider.call_id.clone()))
            .unwrap_or(minted)
    }

    /// Whether this identity was generated from an assembly key.
    pub fn is_generated(&self) -> bool {
        matches!(self.0, ToolCallIdWire::Generated(_))
    }

    /// The explicitly chosen handle, if any. This is not proof of provider
    /// provenance and must not be used to compare differently typed identities.
    pub fn explicit(&self) -> Option<&str> {
        match &self.0 {
            ToolCallIdWire::Explicit(id) => Some(id),
            ToolCallIdWire::Generated(_) => None,
        }
    }

    /// The typed assembly origin of a generated identity, if any. Explicit
    /// identities have no generated origin, even when their text resembles one.
    pub fn generated(&self) -> Option<&crate::streaming::BlockId> {
        match &self.0 {
            ToolCallIdWire::Generated(block) => Some(block),
            ToolCallIdWire::Explicit(_) => None,
        }
    }

    /// A candidate spelling for protocols requiring string handles. It is not
    /// unique across namespaces: request adapters must reserve actual provider
    /// handles and allocate aliases for colliding call/result occurrences.
    pub fn wire_hint(&self) -> std::borrow::Cow<'_, str> {
        match &self.0 {
            ToolCallIdWire::Explicit(id) => std::borrow::Cow::Borrowed(id),
            ToolCallIdWire::Generated(block) => std::borrow::Cow::Owned(block.to_string()),
        }
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            ToolCallIdWire::Explicit(id) => write!(f, "explicit:{id}"),
            ToolCallIdWire::Generated(crate::streaming::BlockId::Wire(id)) => {
                write!(f, "generated:wire:{id}")
            }
            ToolCallIdWire::Generated(crate::streaming::BlockId::Minted { kind, index }) => {
                write!(f, "generated:minted:{}:{index}", kind.as_str())
            }
        }
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
    /// that forbid this fallback, use [`ToolCall::from_dual_wire`].
    pub fn from_optional_wire(call_id: Option<String>, tool_id: Option<String>) -> Option<Self> {
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

/// Assign deterministic completion-local handles to id-less provider calls.
///
/// A missing handle uses its tool-call position, counting function and custom
/// calls alike so no two id-less calls share one. Generated and explicit handles
/// occupy separate namespaces, so a later explicit provider string never forces
/// renumbering. Provider metadata and the existing explicit-duplicate policy are
/// preserved. Use only at inbound provider boundaries, not on application
/// messages with chosen local IDs or already-published streaming identities.
pub fn normalize_missing_tool_call_ids(content: &mut [AssistantContent]) {
    for (position, (id, provider)) in content
        .iter_mut()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some((&mut call.id, &call.provider)),
            AssistantContent::CustomToolCall(call) => Some((&mut call.id, &call.provider)),
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_)
            | AssistantContent::ProviderItem(_) => None,
        })
        .enumerate()
    {
        if provider.is_none() {
            *id = ToolCallId::minted(position as u64);
        }
    }
}

impl ToolCall {
    fn assemble(provider: Option<ProviderCallId>, index: u64, function: ToolFunction) -> Self {
        Self {
            id: ToolCallId::for_provider_or(provider.as_ref(), ToolCallId::minted(index)),
            provider,
            function,
            signature: None,
            additional_params: None,
        }
    }

    /// A call with an explicit correlation handle and no provider-issued id.
    pub fn new(id: ToolCallId, function: ToolFunction) -> Self {
        Self {
            id,
            ..Self::assemble(None, 0, function)
        }
    }

    /// The single-identifier provider boundary for a response's *only*
    /// call: adopt the wire's id when it issued one, mint at index zero
    /// when it did not (empty or absent ids mint). A converter that walks
    /// a response's parts uses [`ToolCall::from_wire_indexed`] with the
    /// call's position, otherwise two id-less calls in one turn mint the
    /// same handle.
    pub fn from_wire(wire_id: impl Into<String>, function: ToolFunction) -> Self {
        Self::from_wire_indexed(wire_id, 0, function)
    }

    /// Adopts a provider ID or generates [`ToolCallId::minted`] at `index` when
    /// the ID is empty. Supply distinct positions for id-less calls in one turn.
    pub fn from_wire_indexed(
        wire_id: impl Into<String>,
        index: u64,
        function: ToolFunction,
    ) -> Self {
        Self::assemble(ProviderCallId::new(wire_id), index, function)
    }

    /// The dual-identifier provider boundary (OpenAI Responses): `item_id`
    /// is the output-item handle (`fc_…`), `call_id` the correlator
    /// (`call_…`). The correlator drives rig's id; empty ids mint.
    pub fn from_dual_wire(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        function: ToolFunction,
    ) -> Self {
        let provider =
            ProviderCallId::new(call_id).map(|provider| provider.with_item_id(item_id.into()));
        Self::assemble(provider, 0, function)
    }

    /// Attach provider-issued identifiers.
    pub fn with_provider(mut self, provider: ProviderCallId) -> Self {
        self.provider = Some(provider);
        self
    }

    /// A non-empty candidate for a required wire call-ID slot: the exact
    /// provider handle when present, otherwise the local identity's wire hint.
    ///
    /// This single-item helper cannot reserve future provider IDs or pair
    /// repeated turns. Full request adapters must use
    /// [`ToolCallIds`](crate::providers::internal::tool_call_ids::ToolCallIds)
    /// to assign collision-free synthetic references consistently to both legs.
    ///
    /// Wires whose id slot is *optional* (Gemini REST, gRPC) must read
    /// [`ToolCall::provider`] directly instead: minted handles never travel
    /// upstream there.
    pub fn wire_call_id(&self) -> std::borrow::Cow<'_, str> {
        self.provider.as_ref().map_or_else(
            || self.id.wire_hint(),
            |provider| std::borrow::Cow::Borrowed(provider.call_id.as_str()),
        )
    }

    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Attach (or clear) the namespace the provider qualified this call with.
    pub fn with_namespace(mut self, namespace: Option<String>) -> Self {
        self.function.namespace = namespace;
        self
    }

    /// The result kind that answers this call: always [`AnsweredToolCall::Function`].
    pub fn answered_by(&self) -> AnsweredToolCall {
        AnsweredToolCall::Function
    }

    /// This call as name-keyed dispatch may run it: unchanged when it carries
    /// no namespace, otherwise [`UndispatchableToolCall::Namespaced`]. The
    /// function-call half of [`name_dispatchable_call`].
    pub fn name_dispatchable(&self) -> Result<&Self, UndispatchableToolCall> {
        match &self.function.namespace {
            None => Ok(self),
            Some(namespace) => Err(UndispatchableToolCall::Namespaced {
                id: self.id.clone(),
                name: self.function.name.clone(),
                namespace: namespace.clone(),
            }),
        }
    }

    /// This call as a wire that has no namespace concept may encode it: the
    /// call unchanged when it carries no namespace, otherwise
    /// [`UnrepresentableToolCall::Namespace`].
    ///
    /// Every JSON-only provider wire applies this one rule on encode. `Some`
    /// is refused whatever it spells — including `""` and `"functions"` —
    /// because no such wire here has evidence that it treats any spelling as
    /// its default namespace; a wire that gains such evidence documents it
    /// and applies it locally instead of calling this.
    pub fn for_json_only_wire(&self, wire: &'static str) -> Result<&Self, UnrepresentableToolCall> {
        match &self.function.namespace {
            None => Ok(self),
            Some(namespace) => Err(UnrepresentableToolCall::Namespace {
                wire,
                name: self.function.name.clone(),
                namespace: namespace.clone(),
            }),
        }
    }
}

/// The one encode rule for a wire whose tool calls carry JSON arguments and
/// no namespace: pass a function call without a namespace through, refuse a
/// namespaced call and every [`CustomToolCall`], and ignore content that is
/// not a tool call.
///
/// Returns the function call to encode, `None` for non-call content.
pub fn json_only_wire_tool_call<'a>(
    wire: &'static str,
    content: &'a AssistantContent,
) -> Result<Option<&'a ToolCall>, UnrepresentableToolCall> {
    match content {
        AssistantContent::ToolCall(call) => call.for_json_only_wire(wire).map(Some),
        AssistantContent::CustomToolCall(call) => Err(call.refused_by_json_only_wire(wire)),
        // Not a call: each wire refuses a provider item it cannot replay.
        AssistantContent::Text(_)
        | AssistantContent::Reasoning(_)
        | AssistantContent::Image(_)
        | AssistantContent::ProviderItem(_) => Ok(None),
    }
}

/// A model-emitted tool call that name-keyed dispatch cannot run without
/// erasing what it is.
///
/// Rig's agent loop and ECS runtime authorize, dispatch
/// (`EffectKind::ToolCall { name, args }`), and expose calls to hooks by tool
/// name and JSON argument text alone. A namespaced call dispatched by its bare
/// name would run a different tool than the one the model called, and a
/// custom call's raw input is not JSON arguments — even when its text happens
/// to parse as JSON. Both runtimes refuse such a call through
/// [`name_dispatchable_call`] before authorization, effect construction, or
/// any hook sees it.
#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UndispatchableToolCall {
    /// A function call qualified by a namespace.
    #[error(
        "tool call `{name}` is qualified by namespace `{namespace}`, which name-keyed tool dispatch cannot honor"
    )]
    Namespaced {
        /// The call's correlation handle.
        id: ToolCallId,
        /// The callable's name.
        name: String,
        /// The namespace, verbatim.
        namespace: String,
    },
    /// A custom tool call, whose input is raw text rather than JSON arguments.
    #[error(
        "custom tool call `{name}` carries raw input, which JSON-argument tool dispatch cannot run"
    )]
    Custom {
        /// The call's correlation handle.
        id: ToolCallId,
        /// The custom tool's name.
        name: String,
        /// The namespace, verbatim, when the call had one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

/// The one dispatch rule for runtimes that run tools by name with JSON
/// arguments: an unqualified function call is dispatchable; a namespaced call
/// and every custom call are refused ([`UndispatchableToolCall`]); content that
/// is not a tool call yields `None`.
pub fn name_dispatchable_call(
    content: &AssistantContent,
) -> Result<Option<&ToolCall>, UndispatchableToolCall> {
    match content {
        AssistantContent::ToolCall(call) => call.name_dispatchable().map(Some),
        AssistantContent::CustomToolCall(call) => Err(UndispatchableToolCall::Custom {
            id: call.id.clone(),
            name: call.name.clone(),
            namespace: call.namespace.clone(),
        }),
        AssistantContent::Text(_)
        | AssistantContent::Reasoning(_)
        | AssistantContent::Image(_)
        | AssistantContent::ProviderItem(_) => Ok(None),
    }
}

/// The first call in `content` that [`name_dispatchable_call`] refuses, if any.
pub fn first_undispatchable_call(content: &[AssistantContent]) -> Option<UndispatchableToolCall> {
    content
        .iter()
        .find_map(|item| name_dispatchable_call(item).err())
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
    /// treats one as its default namespace is that destination's evidence to
    /// supply, and a wire that cannot represent a namespace refuses the call
    /// ([`ToolCall::for_json_only_wire`]) rather than dropping it.
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

/// A custom tool call: a named callable whose input is verbatim text rather
/// than JSON arguments (the OpenAI Responses `custom_tool_call` item, e.g. a
/// grammar-constrained tool).
///
/// Held in [`AssistantContent::CustomToolCall`]. The input is preserved byte
/// for byte and is never parsed, even when it happens to be valid JSON: text
/// that parses is still not a function call's arguments.
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

impl CustomToolCall {
    /// The dual-identifier provider boundary (OpenAI Responses): `item_id` is
    /// the output-item handle (`ctc_…`), `call_id` the correlator (`call_…`).
    /// The correlator drives rig's id; an empty `call_id` records no provider
    /// identity and mints at index zero.
    pub fn from_dual_wire(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        namespace: Option<String>,
        input: impl Into<String>,
    ) -> Self {
        let provider =
            ProviderCallId::new(call_id).map(|provider| provider.with_item_id(item_id.into()));
        Self {
            id: ToolCallId::for_provider_or(provider.as_ref(), ToolCallId::minted(0)),
            provider,
            name: name.into(),
            namespace,
            input: input.into(),
        }
    }

    /// The result kind that answers this call: always [`AnsweredToolCall::Custom`].
    pub fn answered_by(&self) -> AnsweredToolCall {
        AnsweredToolCall::Custom
    }

    /// A non-empty candidate for a required wire call-ID slot; see
    /// [`ToolCall::wire_call_id`].
    pub fn wire_call_id(&self) -> std::borrow::Cow<'_, str> {
        self.provider.as_ref().map_or_else(
            || self.id.wire_hint(),
            |provider| std::borrow::Cow::Borrowed(provider.call_id.as_str()),
        )
    }

    /// The refusal a wire whose tool calls carry JSON arguments only returns
    /// for this call. See [`UnrepresentableToolCall::CustomCall`].
    pub fn refused_by_json_only_wire(&self, wire: &'static str) -> UnrepresentableToolCall {
        UnrepresentableToolCall::CustomCall {
            wire,
            name: self.name.clone(),
        }
    }
}

/// A tool call reached a destination wire that cannot represent it.
///
/// The shared refusal every JSON-only provider wire applies on encode, so no
/// wire drops a call (which would orphan its result), wraps raw input as a
/// JSON string, or silently strips a namespace. Converts into
/// [`MessageError`], [`crate::error::EncodeError`], and therefore
/// [`ProviderError::Request`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum UnrepresentableToolCall {
    /// A [`CustomToolCall`] was replayed to a wire whose tool calls carry JSON
    /// arguments only. No JSON value stands for the raw input: a JSON string
    /// would present it as a parsed argument, and dropping the call would
    /// leave its result unanswered.
    #[error(
        "{wire} carries JSON tool arguments only, so custom tool call `{name}` and its raw input cannot be sent on it"
    )]
    CustomCall {
        /// The destination wire.
        wire: &'static str,
        /// The custom tool's name.
        name: String,
    },
    /// A namespace-qualified call was replayed to a wire that has no way to
    /// express the qualifier. Sending the bare name would call a different
    /// tool than the one the model called.
    #[error(
        "{wire} cannot represent tool namespaces, so call `{name}` qualified by namespace `{namespace}` cannot be sent on it"
    )]
    Namespace {
        /// The destination wire.
        wire: &'static str,
        /// The callable's name.
        name: String,
        /// The namespace the wire cannot express, verbatim.
        namespace: String,
    },
}

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

    /// The underlying (non-empty) object.
    pub fn as_map(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.0
    }

    /// The params as a bare JSON object value.
    pub fn into_value(self) -> serde_json::Value {
        serde_json::Value::Object(self.0)
    }

    /// Deep-merge `incoming` into `self`: arrays concatenate (streamed
    /// citation deltas), objects merge recursively, scalars take the
    /// incoming value.
    pub fn merge(&mut self, incoming: Self) {
        fn merge_maps(
            existing: &mut serde_json::Map<String, serde_json::Value>,
            incoming: serde_json::Map<String, serde_json::Value>,
        ) {
            for (key, incoming_value) in incoming {
                match existing.get_mut(&key) {
                    Some(existing_value) => merge_value(existing_value, incoming_value),
                    None => {
                        existing.insert(key, incoming_value);
                    }
                }
            }
        }
        fn merge_value(existing: &mut serde_json::Value, incoming: serde_json::Value) {
            match (existing, incoming) {
                (
                    serde_json::Value::Object(existing_map),
                    serde_json::Value::Object(incoming_map),
                ) => merge_maps(existing_map, incoming_map),
                (
                    serde_json::Value::Array(existing_array),
                    serde_json::Value::Array(mut incoming_array),
                ) => existing_array.append(&mut incoming_array),
                (existing, incoming) => *existing = incoming,
            }
        }
        for (key, value) in incoming.0 {
            // Only the top-level owned marker is an atomic provider value.
            if crate::providers::openai::responses_api::is_opaque_part_marker(&key, &value) {
                self.0.insert(key, value);
            } else {
                match self.0.get_mut(&key) {
                    Some(existing) => merge_value(existing, value),
                    None => {
                        self.0.insert(key, value);
                    }
                }
            }
        }
    }

    /// Returns the object under the provider's own key, or `None` for absent
    /// or non-object values. Use [`Self::get`] to diagnose malformed values.
    pub fn wire_extras(
        &self,
        wire_key: &str,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.0.get(wire_key).and_then(serde_json::Value::as_object)
    }

    /// Owned counterpart of [`Self::wire_extras`] for serialization paths
    /// that already own the params (the common replay case): extracts the
    /// wire's object without cloning. Same gate semantics.
    pub fn into_wire_extras(
        mut self,
        wire_key: &str,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        match self.0.remove(wire_key) {
            Some(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        }
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

impl From<AdditionalParams> for serde_json::Value {
    fn from(params: AdditionalParams) -> Self {
        params.into_value()
    }
}

impl std::ops::Index<&str> for AdditionalParams {
    type Output = serde_json::Value;

    /// Returns the value under `key`.
    ///
    /// # Panics
    /// Panics if the key is absent.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, key: &str) -> &serde_json::Value {
        &self.0[key]
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

/// Returns dot-separated paths whose original values are missing or changed
/// after a round trip. Ignores added keys, null object members, and missing
/// object members whose original value was an empty object. Array positions
/// are compared individually.
///
/// ```
/// use rig_core::message;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let original = serde_json::json!({
///     "role": "assistant",
///     "content": [{"type": "text", "text": "cited", "citations": ["not re-nested"]}],
/// });
/// let loaded: message::Message = serde_json::from_value(original.clone())?;
/// let round_tripped = serde_json::to_value(&loaded)?;
/// let lost = message::keys_lost_in_round_trip(&original, &round_tripped);
/// assert_eq!(lost, vec!["content.0.citations".to_string()]);
/// # Ok(())
/// # }
/// ```
pub fn keys_lost_in_round_trip(
    original: &serde_json::Value,
    round_tripped: &serde_json::Value,
) -> Vec<String> {
    fn walk(
        original: &serde_json::Value,
        round_tripped: &serde_json::Value,
        path: &mut String,
        lost: &mut Vec<String>,
    ) {
        match (original, round_tripped) {
            (serde_json::Value::Object(original_map), serde_json::Value::Object(round_map)) => {
                for (key, original_value) in original_map {
                    if original_value.is_null() {
                        continue;
                    }
                    let checkpoint = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(key);
                    match round_map.get(key) {
                        Some(round_value) => walk(original_value, round_value, path, lost),
                        // Empty objects may canonicalize to absent metadata.
                        None => {
                            if !original_value
                                .as_object()
                                .is_some_and(serde_json::Map::is_empty)
                            {
                                lost.push(path.clone());
                            }
                        }
                    }
                    path.truncate(checkpoint);
                }
            }
            (serde_json::Value::Array(original_items), serde_json::Value::Array(round_items)) => {
                for (index, original_value) in original_items.iter().enumerate() {
                    let checkpoint = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(&index.to_string());
                    match round_items.get(index) {
                        Some(round_value) => walk(original_value, round_value, path, lost),
                        None => lost.push(path.clone()),
                    }
                    path.truncate(checkpoint);
                }
            }
            (original, round_tripped) => {
                if original != round_tripped {
                    lost.push(path.clone());
                }
            }
        }
    }

    let mut lost = Vec::new();
    walk(original, round_tripped, &mut String::new(), &mut lost);
    lost
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

/// Image content containing image data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Image {
    /// Image source data.
    pub data: DocumentSourceKind,
    /// Image media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<ImageMediaType>,
    /// Provider-specific image detail preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
    /// Provider-specific image fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// The kind of image source (to be used).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
pub enum DocumentSourceKind {
    /// A file URL/URI.
    Url(String),
    /// A base-64 encoded string.
    Base64(String),
    /// A provider-side uploaded file identifier.
    FileId(String),
    /// Raw bytes
    Raw(Vec<u8>),
    /// A string (or a string literal).
    String(String),
    #[default]
    /// An unknown file source (there's nothing there).
    Unknown,
}

impl DocumentSourceKind {
    /// Create a URL-backed source.
    pub fn url(url: &str) -> Self {
        Self::Url(url.to_string())
    }

    /// Create a base64-backed source.
    pub fn base64(base64_string: &str) -> Self {
        Self::Base64(base64_string.to_string())
    }

    /// Create a provider file ID-backed source.
    pub fn file_id(file_id: &str) -> Self {
        Self::FileId(file_id.to_string())
    }

    /// Create a string-backed source.
    pub fn string(input: &str) -> Self {
        Self::String(input.into())
    }

    /// Return the contained URL, base64 string, or file ID, if this source stores one.
    pub fn try_into_inner(self) -> Option<String> {
        match self {
            Self::Url(s) | Self::Base64(s) | Self::FileId(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for DocumentSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Url(string) => write!(f, "{string}"),
            Self::Base64(string) => write!(f, "{string}"),
            Self::FileId(string) => write!(f, "{string}"),
            Self::String(string) => write!(f, "{string}"),
            Self::Raw(_) => write!(f, "<binary data>"),
            Self::Unknown => write!(f, "<unknown>"),
        }
    }
}

/// Audio content containing audio data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Audio {
    /// Audio source data.
    pub data: DocumentSourceKind,
    /// Audio media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<AudioMediaType>,
    /// Provider-specific audio fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// Video content containing video data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Video {
    /// Video source data.
    pub data: DocumentSourceKind,
    /// Video media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<VideoMediaType>,
    /// Provider-specific video fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// Document content containing document data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Document {
    /// Document source data.
    pub data: DocumentSourceKind,
    /// Document media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
    /// Provider-specific document fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// Content representation as base64, text, or a URL.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    #[default]
    Base64,
    String,
    Url,
}

/// Helper enum that tracks the media type of the content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum MediaType {
    Image(ImageMediaType),
    Audio(AudioMediaType),
    Document(DocumentMediaType),
    Video(VideoMediaType),
}

/// Describes the image media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    PNG,
    GIF,
    WEBP,
    HEIC,
    HEIF,
    SVG,
}

/// Describes the document media type of the content. Not every provider supports every media type.
/// Includes also programming languages as document types for providers who support code running.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
    TXT,
    RTF,
    HTML,
    CSS,
    MARKDOWN,
    CSV,
    XML,
    Javascript,
    Python,
}

impl DocumentMediaType {
    pub fn is_code(&self) -> bool {
        matches!(self, Self::Javascript | Self::Python)
    }
}

/// Describes the audio media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    WAV,
    MP3,
    AIFF,
    AAC,
    OGG,
    FLAC,
    M4A,
    PCM16,
    PCM24,
}

/// Describes the video media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VideoMediaType {
    AVI,
    MP4,
    MPEG,
    MOV,
    WEBM,
}

/// Describes the detail of the image content, which can be low, high, or auto (open-ai specific).
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}

impl Message {
    /// Clones the first text block of a user message, or returns `None`.
    pub fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                for item in content.iter() {
                    if let UserContent::Text(Text { text, .. }) = item {
                        return Some(text.clone());
                    }
                }
                None
            }
            Message::System { .. } => None,
            _ => None,
        }
    }

    /// Creates a system instruction message.
    pub fn system(text: impl Into<String>) -> Self {
        Message::System {
            content: text.into(),
        }
    }

    /// Creates a user message containing one text block.
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::text(text)],
        }
    }

    /// Creates an assistant message containing one text block and no provider ID.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::text(text)],
        }
    }

    /// Creates a user message containing a text tool result.
    /// `call` is an explicit local handle and does not establish provider
    /// provenance. To answer an existing call while preserving its typed identity
    /// and provider metadata, use [`UserContent::tool_result_for`] inside a user
    /// message. `name` is the executed tool's name.
    pub fn tool_result(
        call: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Message::User {
            content: vec![UserContent::tool_result(
                call,
                name,
                vec![ToolResultContent::text(content)],
            )],
        }
    }
}

/// Generates media constructors without fetching or decoding source data.
macro_rules! media_ctors {
    () => {};
    (
        $(#[$meta:meta])* $name:ident => Image($kind:ident: $data:ty);
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name(
            data: impl Into<$data>,
            media_type: Option<ImageMediaType>,
            detail: Option<ImageDetail>,
        ) -> Self {
            Self::Image(Image {
                data: DocumentSourceKind::$kind(data.into()),
                media_type,
                detail,
                additional_params: None,
            })
        }
        media_ctors! { $($rest)* }
    };
    (
        $(#[$meta:meta])* $name:ident => $variant:ident($mt:ty, $kind:ident: $data:ty);
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name(data: impl Into<$data>, media_type: Option<$mt>) -> Self {
            Self::$variant($variant {
                data: DocumentSourceKind::$kind(data.into()),
                media_type,
                additional_params: None,
            })
        }
        media_ctors! { $($rest)* }
    };
}

impl UserContent {
    /// Creates user text content.
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into().into())
    }

    media_ctors! {
        /// Creates user image content from base64-encoded data.
        image_base64 => Image(Base64: String);
        /// Creates user image content from unencoded bytes.
        image_raw => Image(Raw: Vec<u8>);
        /// Creates user image content referencing a URL.
        image_url => Image(Url: String);
        /// Creates user audio content from base64-encoded data.
        audio => Audio(AudioMediaType, Base64: String);
        /// Creates user audio content from unencoded bytes.
        audio_raw => Audio(AudioMediaType, Raw: Vec<u8>);
        /// Creates user audio content referencing a URL.
        audio_url => Audio(AudioMediaType, Url: String);
        /// Creates user video content from base64-encoded data.
        video => Video(VideoMediaType, Base64: String);
        /// Creates user video content from unencoded bytes.
        video_raw => Video(VideoMediaType, Raw: Vec<u8>);
        /// Creates user video content referencing a URL.
        video_url => Video(VideoMediaType, Url: String);
        /// Creates user document content from unencoded bytes.
        document_raw => Document(DocumentMediaType, Raw: Vec<u8>);
        /// Creates user document content referencing a URL.
        document_url => Document(DocumentMediaType, Url: String);
    }

    /// Creates document content from a string without decoding or fetching it.
    pub fn document(data: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
        let data: String = data.into();
        UserContent::Document(Document {
            data: DocumentSourceKind::string(&data),
            media_type,
            additional_params: None,
        })
    }

    /// Creates a result with an explicit local handle and no provider identity.
    /// An empty `call` generates the handle at index zero. To preserve an
    /// existing call's typed identity and provenance, use [`Self::tool_result_for`].
    /// `name` must identify the executed tool.
    pub fn tool_result(
        call: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_minted(call, 0),
            provider: None,
            name: name.into(),
            // A rig `Tool` is a function tool: this constructor answers a
            // function call. A custom call's result is built through
            // `tool_result_answering` or `tool_result_for_custom_call`.
            answers: AnsweredToolCall::Function,
            content,
        })
    }

    /// Creates a result answering a function call from a provider-issued ID.
    /// An empty ID records no provider identity and generates a local handle
    /// at index zero.
    pub fn tool_result_from_wire(
        wire_id: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        let provider = ProviderCallId::new(wire_id);
        let call = ToolCallId::for_provider_or(provider.as_ref(), ToolCallId::minted(0));
        Self::tool_result_for(call, provider, name, content)
    }

    /// Creates a result answering a function call, using the executed call's
    /// identity and provider metadata. `name` identifies the executed tool,
    /// including any hook-repaired name.
    pub fn tool_result_for(
        call: ToolCallId,
        provider: Option<ProviderCallId>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        Self::tool_result_for_answering(call, provider, name, AnsweredToolCall::Function, content)
    }

    /// [`Self::tool_result_for`] stating which kind of call it answers, for a
    /// site that holds the answered call's identity but not the call itself.
    pub fn tool_result_for_answering(
        call: ToolCallId,
        provider: Option<ProviderCallId>,
        name: impl Into<String>,
        answers: AnsweredToolCall,
        content: Vec<ToolResultContent>,
    ) -> Self {
        UserContent::ToolResult(ToolResult {
            call,
            provider,
            name: name.into(),
            answers,
            content,
        })
    }

    /// Creates the result answering `call`, copying its identity and provider
    /// metadata; the answered kind is derived from the call.
    pub fn tool_result_for_call(call: &ToolCall, content: Vec<ToolResultContent>) -> Self {
        Self::tool_result_for_answering(
            call.id.clone(),
            call.provider.clone(),
            call.function.name.clone(),
            call.answered_by(),
            content,
        )
    }

    /// Creates the result answering a custom tool `call`, copying its identity
    /// and provider metadata; the answered kind is derived from the call.
    pub fn tool_result_for_custom_call(
        call: &CustomToolCall,
        content: Vec<ToolResultContent>,
    ) -> Self {
        Self::tool_result_for_answering(
            call.id.clone(),
            call.provider.clone(),
            call.name.clone(),
            call.answered_by(),
            content,
        )
    }

    /// Tool result content answering a function call on a dual-identifier
    /// wire (OpenAI Responses): `item_id` is the output-item handle (`fc_…`),
    /// `call_id` the correlator (`call_…`). Empty ids record no provider id
    /// and mint.
    pub fn tool_result_with_call_id(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        Self::tool_result_answering(item_id, call_id, name, AnsweredToolCall::Function, content)
    }

    /// Dual-identifier result that states which kind of call it answers.
    ///
    /// The form a relay uses when it rebuilds a turn from its own records and
    /// the answered call may have been a custom tool rather than a function:
    /// the provider pairs `custom_tool_call` only with
    /// `custom_tool_call_output`, so the kind must be stated, not guessed.
    pub fn tool_result_answering(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        answers: AnsweredToolCall,
        content: Vec<ToolResultContent>,
    ) -> Self {
        let provider = ProviderCallId::new(call_id).map(|provider| provider.with_item_id(item_id));
        let call = ToolCallId::for_provider_or(provider.as_ref(), ToolCallId::minted(0));
        Self::tool_result_for_answering(call, provider, name, answers, content)
    }
}

impl AssistantContent {
    /// Creates assistant text content.
    pub fn text(text: impl Into<String>) -> Self {
        AssistantContent::Text(text.into().into())
    }

    media_ctors! {
        /// Creates assistant image content from base64-encoded data.
        image_base64 => Image(Base64: String);
    }

    /// Creates a tool call from a provider-issued ID, name, and arguments.
    /// An empty ID records no provider identity and generates a handle at index zero.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::new(name.into(), arguments),
        ))
    }

    /// Dual-identifier variant (OpenAI Responses): `id` is the output-item
    /// handle (`fc_…`), `call_id` the correlator (`call_…`).
    pub fn tool_call_with_call_id(
        id: impl Into<String>,
        call_id: String,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::tool_call_with_namespace(id, call_id, name, None, arguments)
    }

    /// Dual-identifier function call carrying the namespace the provider
    /// qualified the callable with (OpenAI Responses namespaced tools).
    pub fn tool_call_with_namespace(
        id: impl Into<String>,
        call_id: String,
        name: impl Into<String>,
        namespace: Option<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall::from_dual_wire(
            id,
            call_id,
            ToolFunction::new(name.into(), arguments).with_namespace(namespace),
        ))
    }

    /// Dual-identifier custom tool call (OpenAI Responses `custom_tool_call`):
    /// `id` is the output-item handle, `call_id` the correlator, and `input`
    /// the model's verbatim input.
    pub fn custom_tool_call(
        id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        namespace: Option<String>,
        input: impl Into<String>,
    ) -> Self {
        AssistantContent::CustomToolCall(CustomToolCall::from_dual_wire(
            id, call_id, name, namespace, input,
        ))
    }

    pub fn reasoning(reasoning: impl AsRef<str>) -> Self {
        AssistantContent::Reasoning(Reasoning::new(reasoning.as_ref()))
    }
}

impl ToolResultContent {
    /// Creates literal text tool-result content.
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text(text.into().into())
    }

    /// Creates structured JSON tool-result content.
    pub fn json(value: serde_json::Value) -> Self {
        ToolResultContent::Json { value }
    }

    media_ctors! {
        /// Creates tool-result image content from base64-encoded data.
        image_base64 => Image(Base64: String);
        /// Creates tool-result image content from raw, unencoded bytes.
        image_raw => Image(Raw: Vec<u8>);
        /// Creates tool-result image content referencing a URL.
        image_url => Image(Url: String);
    }
}

/// Trait for converting between MIME types and media types.
pub trait MimeType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized;
    fn to_mime_type(&self) -> &'static str;
}

impl MimeType for MediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        ImageMediaType::from_mime_type(mime_type)
            .map(MediaType::Image)
            .or_else(|| DocumentMediaType::from_mime_type(mime_type).map(MediaType::Document))
            .or_else(|| AudioMediaType::from_mime_type(mime_type).map(MediaType::Audio))
            .or_else(|| VideoMediaType::from_mime_type(mime_type).map(MediaType::Video))
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            MediaType::Image(media_type) => media_type.to_mime_type(),
            MediaType::Audio(media_type) => media_type.to_mime_type(),
            MediaType::Document(media_type) => media_type.to_mime_type(),
            MediaType::Video(media_type) => media_type.to_mime_type(),
        }
    }
}

// Emits both directions of a [`MimeType`] impl from a single pair list, so a
// variant's parse and emit spellings cannot drift apart. Extra `| "alias"`
// spellings parse to the same variant; only the first (canonical) string is
// emitted by `to_mime_type`.
macro_rules! impl_mime_type {
    ($ty:ident { $($variant:ident => $canonical:literal $(| $alias:literal)*),+ $(,)? }) => {
        impl MimeType for $ty {
            fn from_mime_type(mime_type: &str) -> Option<Self> {
                match mime_type {
                    $($canonical $(| $alias)* => Some($ty::$variant),)+
                    _ => None,
                }
            }

            fn to_mime_type(&self) -> &'static str {
                match self {
                    $($ty::$variant => $canonical,)+
                }
            }
        }
    };
}

impl_mime_type!(ImageMediaType {
    JPEG => "image/jpeg",
    PNG => "image/png",
    GIF => "image/gif",
    WEBP => "image/webp",
    HEIC => "image/heic",
    HEIF => "image/heif",
    SVG => "image/svg+xml",
});

impl_mime_type!(DocumentMediaType {
    PDF => "application/pdf",
    TXT => "text/plain",
    RTF => "text/rtf",
    HTML => "text/html",
    CSS => "text/css",
    MARKDOWN => "text/markdown" | "text/md",
    CSV => "text/csv",
    XML => "text/xml",
    Javascript => "application/x-javascript" | "text/x-javascript",
    Python => "application/x-python" | "text/x-python",
});

impl_mime_type!(AudioMediaType {
    WAV => "audio/wav",
    MP3 => "audio/mp3",
    AIFF => "audio/aiff",
    AAC => "audio/aac",
    OGG => "audio/ogg",
    FLAC => "audio/flac",
    M4A => "audio/m4a",
    PCM16 => "audio/pcm16",
    PCM24 => "audio/pcm24",
});

impl_mime_type!(VideoMediaType {
    AVI => "video/avi",
    MP4 => "video/mp4",
    MPEG => "video/mpeg",
    MOV => "video/mov",
    WEBM => "video/webm",
});

impl std::str::FromStr for ImageDetail {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(ImageDetail::Low),
            "high" => Ok(ImageDetail::High),
            "auto" => Ok(ImageDetail::Auto),
            _ => Err(()),
        }
    }
}

/// `From` impls for [`Text`] from string-like types.
macro_rules! text_from {
    ($($src:ty),+ $(,)?) => {$(
        impl From<$src> for Text {
            fn from(text: $src) -> Self {
                Text {
                    text: text.into(),
                    additional_params: None,
                }
            }
        }
    )+};
}

text_from!(String, &String, &str);

/// `From<String>` impls that forward into a content type's `text` constructor.
macro_rules! text_content_from_string {
    ($($ty:ident),+ $(,)?) => {$(
        impl From<String> for $ty {
            fn from(text: String) -> Self {
                $ty::text(text)
            }
        }
    )+};
}

text_content_from_string!(ToolResultContent, AssistantContent, UserContent);

/// One-line `From<T> for Message` forwards: convert the value, wrap it in the
/// named content variant, and build a single-content message.
macro_rules! single_content_message_from {
    (User { $($src:ty => $variant:ident),+ $(,)? }) => {$(
        impl From<$src> for Message {
            fn from(value: $src) -> Self {
                Message::User {
                    content: vec![UserContent::$variant(value.into())],
                }
            }
        }
    )+};
    (Assistant { $($src:ty => $variant:ident),+ $(,)? }) => {$(
        impl From<$src> for Message {
            fn from(value: $src) -> Self {
                Message::Assistant {
                    id: None,
                    content: vec![AssistantContent::$variant(value.into())],
                }
            }
        }
    )+};
}

single_content_message_from!(User {
    String => Text,
    &str => Text,
    &String => Text,
    Text => Text,
    Image => Image,
    Audio => Audio,
    Document => Document,
    ToolResult => ToolResult,
});

single_content_message_from!(Assistant {
    ToolCall => ToolCall,
    CustomToolCall => CustomToolCall,
});

impl FromStr for Text {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl From<&Message> for Message {
    fn from(msg: &Message) -> Self {
        msg.clone()
    }
}

impl From<AssistantContent> for Message {
    fn from(content: AssistantContent) -> Self {
        Message::Assistant {
            id: None,
            content: vec![content],
        }
    }
}

impl From<UserContent> for Message {
    fn from(content: UserContent) -> Self {
        Message::User {
            content: vec![content],
        }
    }
}

impl From<Vec<AssistantContent>> for Message {
    fn from(content: Vec<AssistantContent>) -> Self {
        Message::Assistant { id: None, content }
    }
}

impl From<Vec<UserContent>> for Message {
    fn from(content: Vec<UserContent>) -> Self {
        Message::User { content }
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific {
        function_names: Vec<String>,
    },
}

/// A destination cannot faithfully replay an opaque Responses content part.
#[derive(Clone, Debug, Error)]
#[error("{wire} cannot replay opaque OpenAI Responses content")]
pub struct UnrepresentableOpaqueContent {
    /// Destination wire that cannot represent the part.
    pub wire: &'static str,
}

impl UnrepresentableOpaqueContent {
    /// Name the destination whose encoder refused the opaque part.
    pub fn new(wire: &'static str) -> Self {
        Self { wire }
    }
}

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    /// Opaque content the destination cannot replay faithfully.
    #[error(transparent)]
    UnrepresentableOpaqueContent(#[from] UnrepresentableOpaqueContent),
    #[error("Message conversion error: {0}")]
    ConversionError(String),
    /// A tool call the destination wire cannot represent.
    #[error(transparent)]
    UnrepresentableToolCall(#[from] UnrepresentableToolCall),
    /// A provider item the destination wire cannot replay.
    #[error(transparent)]
    UnreplayableProviderItem(#[from] UnreplayableProviderItem),
}

impl From<MessageError> for ProviderError {
    fn from(error: MessageError) -> Self {
        ProviderError::Request(error.into())
    }
}

/// Assistant turns every JSON-only wire must refuse, for the per-wire
/// refusal tests: one namespaced function call, one custom call.
#[cfg(test)]
pub(crate) fn unrepresentable_turns() -> [(Message, &'static str); 2] {
    [
        (
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::tool_call_with_namespace(
                    "fc_1",
                    "call_1".to_owned(),
                    "add",
                    Some("math".to_owned()),
                    serde_json::json!({"x": 1}),
                )],
            },
            "namespace",
        ),
        (
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::custom_tool_call(
                    "ctc_1",
                    "call_2",
                    "apply_patch",
                    None,
                    r#"{"x": 1}"#,
                )],
            },
            "custom",
        ),
    ]
}

/// Assert `refusal` is the shared JSON-only refusal of the `kind` turn from
/// [`unrepresentable_turns`] on `wire`.
#[cfg(test)]
pub(crate) fn assert_refused(refusal: &UnrepresentableToolCall, kind: &str, wire: &str) {
    match (kind, refusal) {
        (
            "namespace",
            UnrepresentableToolCall::Namespace {
                wire: got,
                name,
                namespace,
            },
        ) => {
            assert_eq!(
                (*got, name.as_str(), namespace.as_str()),
                (wire, "add", "math")
            );
        }
        ("custom", UnrepresentableToolCall::CustomCall { wire: got, name }) => {
            assert_eq!((*got, name.as_str()), (wire, "apply_patch"));
        }
        _ => panic!("expected the {kind} refusal on {wire}, got {refusal:?}"),
    }
}

/// [`assert_refused`] through a [`MessageError`].
#[cfg(test)]
pub(crate) fn assert_message_refused(error: &MessageError, kind: &str, wire: &str) {
    match error {
        MessageError::UnrepresentableToolCall(refusal) => assert_refused(refusal, kind, wire),
        other => panic!("expected the {kind} refusal on {wire}, got {other:?}"),
    }
}

#[cfg(test)]
mod tests;
