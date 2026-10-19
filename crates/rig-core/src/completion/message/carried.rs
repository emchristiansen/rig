//! Carried per-item provenance and tool shapes that remain distinct in the
//! canonical model. Their meaning does not depend on a turn's `Origin`.

use super::{
    AssistantContent, CallId, Message, Native, ToolCall, ToolName, ToolResult, ToolResultContent,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The place a carried value occupied in its original provider item.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CarriedPosition {
    /// The whole output item; omitted from its stored representation.
    #[default]
    Whole,
    /// One content part of one output message.
    MessageContent {
        /// The containing output item's original index.
        item_index: u64,
        /// The original content array index.
        part_index: u64,
    },
    /// One part of a reasoning item's summary array.
    ReasoningSummary {
        /// The original summary array index.
        part_index: u64,
    },
    /// One part of a reasoning item's content array.
    ReasoningContent {
        /// The original reasoning-content array index.
        part_index: u64,
    },
}

impl CarriedPosition {
    /// Whether this value is the whole item.
    pub fn is_whole(&self) -> bool {
        matches!(self, Self::Whole)
    }
}

/// Exact provider data with its independent issuer and original position.
/// It carries neither a turn's Origin nor a Native fingerprint.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CarriedPayload {
    /// The original JSON value, including unknown fields.
    pub item: serde_json::Value,
    /// The item's issuer. `None` means unknown provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// The original part position, absent for a whole item.
    #[serde(default, skip_serializing_if = "CarriedPosition::is_whole")]
    pub position: CarriedPosition,
}

impl CarriedPayload {
    /// A whole value with its exact issuer, including unknown provenance.
    pub fn new(item: serde_json::Value, provider: Option<String>) -> Self {
        Self {
            item,
            provider,
            position: CarriedPosition::Whole,
        }
    }

    /// Record the original position without changing the value or issuer.
    pub fn with_position(mut self, position: CarriedPosition) -> Self {
        self.position = position;
        self
    }

    /// The original value's `type`, when it names one.
    pub fn item_type(&self) -> Option<&str> {
        self.item.get("type").and_then(serde_json::Value::as_str)
    }

    /// Whether the issuer is unknown, exact, or in the requested family.
    /// This describes provenance; the destination decides its replay policy.
    pub fn replayable_to(&self, issuer: &str) -> bool {
        self.provider.as_deref().is_none_or(|provider| {
            provider == issuer || (issuer.ends_with('/') && provider.starts_with(issuer))
        })
    }
}

/// An unmodelled provider output item, preserved whole with its own issuer.
/// Unsupported or foreign destinations refuse this whole item explicitly.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProviderItem {
    /// The whole item's exact data and provenance, flattened on the wire.
    #[serde(flatten)]
    pub payload: CarriedPayload,
}

impl ProviderItem {
    /// An item issued by `provider`.
    pub fn new(item: serde_json::Value, provider: impl Into<String>) -> Self {
        Self {
            payload: CarriedPayload::new(item, Some(provider.into())),
        }
    }

    /// The item's `type`, when it names one.
    pub fn item_type(&self) -> Option<&str> {
        self.payload.item_type()
    }

    /// Whether this item came from `issuer`, or its provenance is unknown.
    pub fn replayable_to(&self, issuer: &str) -> bool {
        self.payload.replayable_to(issuer)
    }
}

/// Gemini's exact signature for a known answer text, independent of its text.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct GeminiThoughtSignature(String);

impl GeminiThoughtSignature {
    /// Retain the exact signature, including an empty or whitespace value.
    pub fn new(signature: impl Into<String>) -> Self {
        Self(signature.into())
    }

    /// The exact provider signature.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A signed Gemini answer whose known text can be edited independently.
/// Its signature does not depend on a Native fingerprint or turn Origin.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GeminiSignedAnswer {
    /// The current canonical answer text.
    pub text: String,
    /// The signature Gemini supplied for this answer part.
    pub signature: GeminiThoughtSignature,
}

impl GeminiSignedAnswer {
    /// Known answer text and its independently retained signature.
    pub fn new(text: impl Into<String>, signature: GeminiThoughtSignature) -> Self {
        Self {
            text: text.into(),
            signature,
        }
    }
}

/// The independently editable text of one carried message part.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", content = "text", rename_all = "snake_case")]
pub enum CarriedTextContent {
    /// Known output text.
    Text(String),
    /// Known refusal text.
    Refusal(String),
    /// Editable canonical text beside an opaque raw part; initially empty.
    Opaque(String),
    /// Known Gemini answer text with its independent provider signature.
    GeminiSignedAnswer(GeminiSignedAnswer),
    /// Editable known output text beside unselected inspection-only siblings.
    /// Every encoder projects the current text without replaying raw fields.
    InspectionOnlyText(String),
    /// Editable known text beside media retained only for history and inspection.
    /// The preserved media never has outbound replay authority.
    InspectionOnlyMedia(String),
}

impl CarriedTextContent {
    /// The exact canonical text, including an edit beside an opaque value.
    pub fn text(&self) -> &str {
        match self {
            Self::Text(text)
            | Self::Refusal(text)
            | Self::Opaque(text)
            | Self::InspectionOnlyText(text)
            | Self::InspectionOnlyMedia(text) => text,
            Self::GeminiSignedAnswer(answer) => &answer.text,
        }
    }

    /// Edit canonical text independently of the preserved JSON value.
    pub fn text_mut(&mut self) -> &mut String {
        match self {
            Self::Text(text)
            | Self::Refusal(text)
            | Self::Opaque(text)
            | Self::InspectionOnlyText(text)
            | Self::InspectionOnlyMedia(text) => text,
            Self::GeminiSignedAnswer(answer) => &mut answer.text,
        }
    }
}

/// One ordered carried message part, kept with its original inspection data.
/// Responses parts may replay their provider shape. Gemini signed answers and
/// inspection-only media project known text without replaying their raw payload.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(from = "CarriedTextRepr")]
pub struct CarriedText {
    /// Independently editable text with its explicit carried category.
    pub content: CarriedTextContent,
    /// The exact part, issuer and original message/content position.
    pub payload: CarriedPayload,
    /// The containing output message's ID, independent of response identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    /// The containing output message's exact phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Unknown containing-message fields retained for history and inspection.
    /// These fields are never included in an outbound request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unknown_message_fields: Option<serde_json::Map<String, serde_json::Value>>,
    /// Citations bound to this part's known text; read through `citations()`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) citations: Option<super::citation::Citations>,
}

#[derive(Deserialize)]
struct CarriedTextRepr {
    content: CarriedTextContent,
    payload: CarriedPayload,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    unknown_message_fields: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, deserialize_with = "super::citation::lenient")]
    citations: Option<super::citation::Citations>,
}

impl From<CarriedTextRepr> for CarriedText {
    fn from(repr: CarriedTextRepr) -> Self {
        Self {
            content: repr.content,
            payload: repr.payload,
            item_id: repr.item_id,
            phase: repr.phase,
            unknown_message_fields: repr.unknown_message_fields,
            citations: repr.citations,
        }
        .checked()
    }
}

impl CarriedText {
    /// A part with no containing-message metadata yet attached.
    pub fn new(content: CarriedTextContent, payload: CarriedPayload) -> Self {
        Self {
            content,
            payload,
            item_id: None,
            phase: None,
            unknown_message_fields: None,
            citations: None,
        }
    }

    /// The containing message's uninterpreted fields, for inspection only.
    /// Outbound encoders reconstruct recognized fields without this map.
    pub fn unknown_message_fields(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.unknown_message_fields.as_ref()
    }

    /// Retain unknown containing-message fields without giving them replay authority.
    pub fn with_unknown_message_fields(
        mut self,
        fields: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.unknown_message_fields = (!fields.is_empty()).then_some(fields);
        self
    }

    /// The exact canonical text.
    pub fn text(&self) -> &str {
        self.content.text()
    }

    /// Edit this part without editing its preserved JSON or sibling parts.
    pub fn text_mut(&mut self) -> &mut String {
        self.content.text_mut()
    }

    /// Current known text for signed answers and inspection-only media.
    /// This projection gives the preserved payload no outbound authority.
    pub fn known_text_projection(&self) -> Option<&str> {
        match &self.content {
            CarriedTextContent::GeminiSignedAnswer(answer) => Some(&answer.text),
            CarriedTextContent::InspectionOnlyText(text)
            | CarriedTextContent::InspectionOnlyMedia(text) => Some(text),
            CarriedTextContent::Text(_)
            | CarriedTextContent::Refusal(_)
            | CarriedTextContent::Opaque(_) => None,
        }
    }

    /// The signed answer, whose signature is independent of text edits/model.
    pub fn signed_answer(&self) -> Option<&GeminiSignedAnswer> {
        match &self.content {
            CarriedTextContent::GeminiSignedAnswer(answer) => Some(answer),
            CarriedTextContent::Text(_)
            | CarriedTextContent::Refusal(_)
            | CarriedTextContent::Opaque(_)
            | CarriedTextContent::InspectionOnlyText(_)
            | CarriedTextContent::InspectionOnlyMedia(_) => None,
        }
    }

    /// What an encoder may send for this part, requiring an explicit choice.
    /// Responses parts update only a known text/refusal field or replay opaque
    /// JSON exactly. Signed answers/media return only their current known text.
    pub fn replay_item(&self) -> Result<CarriedTextReplay<'_>, CarriedTextReplayError> {
        let (key, text) = match &self.content {
            CarriedTextContent::Text(text) => ("text", text),
            CarriedTextContent::Refusal(text) => ("refusal", text),
            CarriedTextContent::Opaque(text) if text.is_empty() => {
                return Ok(CarriedTextReplay::ProviderPart(self.payload.item.clone()));
            }
            CarriedTextContent::Opaque(_) => return Err(CarriedTextReplayError::EditedOpaqueText),
            CarriedTextContent::GeminiSignedAnswer(answer) => {
                return Ok(CarriedTextReplay::KnownText(&answer.text));
            }
            CarriedTextContent::InspectionOnlyText(text)
            | CarriedTextContent::InspectionOnlyMedia(text) => {
                return Ok(CarriedTextReplay::KnownText(text));
            }
        };
        let mut item = self.payload.item.clone();
        let serde_json::Value::Object(fields) = &mut item else {
            return Err(CarriedTextReplayError::InvalidKnownPart { field: key });
        };
        fields.insert(key.to_owned(), serde_json::Value::String(text.clone()));
        Ok(CarriedTextReplay::ProviderPart(item))
    }
}

/// The closed outbound projection of one carried text part.
/// Known text never carries the inspection payload's raw JSON.
#[derive(Clone, Debug, PartialEq)]
pub enum CarriedTextReplay<'a> {
    /// A Responses part whose explicit category gives it replay authority.
    ProviderPart(serde_json::Value),
    /// Current canonical text; the destination constructs its own known shape.
    KnownText(&'a str),
}

/// A carried part cannot replay without losing its canonical edit or raw shape.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CarriedTextReplayError {
    /// The raw opaque part has no known text field that can carry this edit.
    #[error(
        "an opaque Responses message part carries non-empty text: the raw part replays verbatim, so the text cannot be sent"
    )]
    EditedOpaqueText,
    /// A known part's text field requires a JSON object.
    #[error("a carried Responses part with a `{field}` field is not a JSON object")]
    InvalidKnownPart { field: &'static str },
}

/// One structured reasoning block, with unknown values kept as carried data.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum CarriedReasoningPart {
    /// Plain reasoning text with its exact optional signature.
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Known summary text.
    Summary(String),
    /// Encrypted data, excluded from the visible text projection.
    Encrypted(String),
    /// Withheld reasoning data, retained in the fork's visible projection.
    Redacted { data: String },
    /// An exact unknown summary part with its position and issuer.
    OpaqueSummary(CarriedPayload),
    /// An exact unknown reasoning-content part with its position and issuer.
    OpaqueContent(CarriedPayload),
}

impl CarriedReasoningPart {
    /// The fork's visible text projection of this part.
    pub fn visible_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } | Self::Summary(text) => Some(text),
            Self::Redacted { data } => Some(data),
            Self::Encrypted(_) | Self::OpaqueSummary(_) | Self::OpaqueContent(_) => None,
        }
    }
}

/// The complete ordered known/unknown sequence of one carried reasoning item.
/// Conversion retains the exact issuer as evidence. The destination encoder
/// applies its provenance policy before replaying the preserved parts.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CarriedReasoning {
    /// The containing reasoning item's exact optional provider ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Its structured blocks, in their original order.
    pub content: Vec<CarriedReasoningPart>,
    /// The original containing reasoning item and its independent issuer.
    pub payload: CarriedPayload,
}

impl CarriedReasoning {
    /// Structured reasoning with no item ID yet attached.
    pub fn new(content: Vec<CarriedReasoningPart>, payload: CarriedPayload) -> Self {
        Self {
            id: None,
            content,
            payload,
        }
    }

    /// Attach or clear the exact item ID, without normalizing an empty value.
    pub fn with_id(mut self, id: Option<String>) -> Self {
        self.id = id;
        self
    }

    /// Visible text-like blocks in order, joined with newlines as in the fork.
    pub fn display_text(&self) -> String {
        self.content
            .iter()
            .filter_map(CarriedReasoningPart::visible_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Whether a destination must preserve an unknown summary/content part.
    pub fn has_opaque_parts(&self) -> bool {
        self.content.iter().any(|part| {
            matches!(
                part,
                CarriedReasoningPart::OpaqueSummary(_) | CarriedReasoningPart::OpaqueContent(_)
            )
        })
    }

    /// The original containing item's issuer, independent of turn Origin.
    pub fn provider(&self) -> Option<&str> {
        self.payload.provider.as_deref()
    }

    /// The shared provenance comparison, not a destination refusal policy.
    pub fn replayable_to(&self, issuer: &str) -> bool {
        self.payload.replayable_to(issuer)
    }
}

/// A destination cannot faithfully replay a nested opaque Responses part.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{wire} cannot replay opaque OpenAI Responses content")]
pub struct UnrepresentableOpaqueContent {
    /// The exact destination that cannot represent the part.
    pub wire: String,
}

impl UnrepresentableOpaqueContent {
    /// Name the destination whose preflight refused the nested part.
    pub fn new(wire: impl Into<String>) -> Self {
        Self { wire: wire.into() }
    }
}

/// Deliberately drop whole items none of `issuers` can replay. Never applied
/// implicitly, and does not filter nested carried Text/Reasoning parts.
pub fn retain_replayable_provider_items(history: &mut Vec<Message>, issuers: &[&str]) {
    history.retain_mut(|message| {
        let Message::Assistant(turn) = message else {
            return true;
        };
        let before = turn.content.len();
        turn.content.retain(|part| match part {
            AssistantContent::ProviderItem(item) => {
                issuers.iter().any(|issuer| item.replayable_to(issuer))
            }
            _ => true,
        });
        before == 0 || !turn.content.is_empty()
    });
}

/// A destination cannot replay this whole item's wire shape or issuer.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{wire} cannot replay the `{}` item issued by {}; drop it with retain_replayable_provider_items to switch providers deliberately", item_type.as_deref().unwrap_or("untyped"), issuer.as_deref().unwrap_or("an unknown provider"))]
pub struct UnreplayableProviderItem {
    /// The destination wire.
    pub wire: String,
    /// The item's issuer, when known.
    pub issuer: Option<String>,
    /// The item's `type`, when known.
    pub item_type: Option<String>,
}

impl UnreplayableProviderItem {
    /// The refusal for sending `item` to `wire`.
    pub fn new(wire: impl Into<String>, item: &ProviderItem) -> Self {
        Self {
            wire: wire.into(),
            issuer: item.payload.provider.clone(),
            item_type: item.item_type().map(str::to_owned),
        }
    }
}

/// Which call shape a result answers. Required on deserialization: a result
/// with no kind is rejected instead of inferred from the current tools.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AnsweredToolCall {
    /// A function call with JSON arguments.
    Function,
    /// A custom call with verbatim text input.
    Custom,
}

/// A custom tool call. Its input is verbatim text even when it parses as JSON.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CustomToolCall {
    /// The call's one identity, copied by its result.
    pub id: CallId,
    /// The custom tool name.
    pub name: ToolName,
    /// The exact namespace, including an explicitly empty spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The model's raw input; never parsed as function arguments.
    pub input: String,
    /// The provider item, with its separate output-item id when present.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "super::native::lenient"
    )]
    pub native: Option<Native>,
}

impl CustomToolCall {
    /// A custom call with no provider item.
    pub fn new(id: CallId, name: ToolName, input: impl Into<String>) -> Self {
        Self {
            id,
            name,
            namespace: None,
            input: input.into(),
            native: None,
        }
    }

    /// Attach or clear the namespace without normalizing its spelling.
    pub fn with_namespace(mut self, namespace: Option<String>) -> Self {
        self.namespace = namespace;
        self
    }

    /// The result answering this exact custom call.
    pub fn result(&self, content: Vec<ToolResultContent>) -> ToolResult {
        ToolResult {
            call: self.id.clone(),
            name: self.name.clone(),
            answers: AnsweredToolCall::Custom,
            content,
            is_error: false,
        }
    }

    /// The result kind that answers this custom call.
    pub fn answered_by(&self) -> AnsweredToolCall {
        AnsweredToolCall::Custom
    }

    /// A failed result answering this exact custom call.
    pub fn error_result(&self, content: Vec<ToolResultContent>) -> ToolResult {
        ToolResult {
            is_error: true,
            ..self.result(content)
        }
    }

    /// A JSON-only wire cannot represent this call's raw input.
    pub fn refused_by_json_only_wire(&self, wire: &'static str) -> UnrepresentableToolCall {
        UnrepresentableToolCall::CustomCall {
            wire,
            name: self.name.clone(),
        }
    }
}

/// A wire cannot represent a call without erasing its kind or namespace.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum UnrepresentableToolCall {
    /// Raw custom input cannot stand for JSON function arguments.
    #[error(
        "{wire} carries JSON tool arguments only, so custom tool call `{name}` and its raw input cannot be sent on it"
    )]
    CustomCall { wire: &'static str, name: ToolName },
    /// Sending only the bare name would call a different tool.
    #[error(
        "{wire} cannot represent tool namespaces, so call `{name}` qualified by namespace `{namespace}` cannot be sent on it"
    )]
    Namespace {
        wire: &'static str,
        name: ToolName,
        namespace: String,
    },
}

/// A name-keyed dispatcher cannot honor this call.
#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UndispatchableToolCall {
    /// The function was qualified by a namespace.
    #[error(
        "tool call `{name}` is qualified by namespace `{namespace}`, which name-keyed tool dispatch cannot honor"
    )]
    Namespaced {
        id: CallId,
        name: ToolName,
        namespace: String,
    },
    /// Raw input cannot be dispatched as JSON arguments.
    #[error(
        "custom tool call `{name}` carries raw input, which JSON-argument tool dispatch cannot run"
    )]
    Custom {
        id: CallId,
        name: ToolName,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

/// An unqualified function call passes; every custom or qualified call is
/// refused before a JSON-only destination can encode it.
pub fn json_only_wire_tool_call<'a>(
    wire: &'static str,
    content: &'a AssistantContent,
) -> Result<Option<&'a ToolCall>, UnrepresentableToolCall> {
    match content {
        AssistantContent::ToolCall(call) => call.for_json_only_wire(wire).map(Some),
        AssistantContent::CustomToolCall(call) => Err(call.refused_by_json_only_wire(wire)),
        AssistantContent::Text(_)
        | AssistantContent::CarriedText(_)
        | AssistantContent::Reasoning(_)
        | AssistantContent::CarriedReasoning(_)
        | AssistantContent::Image(_)
        | AssistantContent::ProviderItem(_)
        | AssistantContent::Opaque(_) => Ok(None),
    }
}

/// The one rule for name-keyed dispatch of JSON arguments.
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
        | AssistantContent::CarriedText(_)
        | AssistantContent::Reasoning(_)
        | AssistantContent::CarriedReasoning(_)
        | AssistantContent::Image(_)
        | AssistantContent::ProviderItem(_)
        | AssistantContent::Opaque(_) => Ok(None),
    }
}

/// The first call name-keyed dispatch cannot run.
pub fn first_undispatchable_call(content: &[AssistantContent]) -> Option<UndispatchableToolCall> {
    content
        .iter()
        .find_map(|part| name_dispatchable_call(part).err())
}

#[cfg(test)]
mod nested_tests;

#[cfg(test)]
mod gemini_tests;
