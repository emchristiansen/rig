//! The events of a completion stream. A completion decoder never builds one:
//! it writes through the reply's part handles
//! ([`Out`](crate::wire::Out)), and the writer emits each part's start, its
//! fragments and its end, in that order. A [`Part`] is a part's position in
//! the response's `choice`; only this crate constructs one in code, so code
//! outside the writer cannot build an event out of order. An item also reads
//! back from its serialized form on its own, unchecked, so a client can
//! decode a relayed stream as it arrives; [`Transcript::parse`] is the one
//! place a sequence is checked.
//!
//! ```
//! use rig_core::streaming::{PartKind, StreamEvent, Transcript};
//!
//! let transcript = Transcript::parse(serde_json::json!([
//!     {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
//!     {"item": "event", "value": {"event": "text", "part": 0, "text": "Hello"}},
//!     {"item": "event", "value": {"event": "end", "part": 0,
//!         "content": {"type": "text", "text": "Hello"}}},
//! ]))?;
//! assert!(matches!(
//!     transcript.events().next(),
//!     Some(StreamEvent::Start { kind: PartKind::Text, .. })
//! ));
//! # Ok::<(), rig_core::streaming::SequenceError>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::message::{AssistantContent, ToolName};

use super::UnknownPayload;

/// A provider's position for a content block. The primary coordinate orders
/// output items; the secondary coordinate orders content within one item.
/// Unindexed blocks retain their arrival slots while indexed blocks sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceOrder {
    /// The output item's position.
    pub primary: u64,
    /// The content position within that item.
    pub secondary: u64,
}

impl SourceOrder {
    /// Construct a provider position without compressing two coordinates.
    pub const fn new(primary: u64, secondary: u64) -> Self {
        Self { primary, secondary }
    }
}

/// A part's position in the response's `choice`. Only this crate constructs
/// one, though it reads back from its serialized form:
///
/// ```compile_fail,E0423
/// let part = rig_core::streaming::Part(0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Part(u32);

impl Part {
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The part's position in the response's `choice`.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What kind of part a [`StreamEvent::Start`] opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    /// Answer text.
    Text,
    /// Reasoning.
    Reasoning,
    /// A function call with JSON arguments.
    ToolCall,
    /// A custom tool call with verbatim text input.
    CustomToolCall,
    /// A carried item with its own per-item issuer.
    ProviderItem,
    /// An image.
    Image,
    /// A provider item with no canonical meaning.
    Opaque,
}

/// One event of a completion stream: a part starts, grows, or ends with the
/// content it finalized. Deserializing one event checks nothing about its
/// order; read a sequence through [`Transcript::parse`] to check it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A part opened. Default tool calls open once named. Preserved Responses
    /// may explicitly open a pending slot and later bind its name on this Part.
    Start {
        /// The part.
        part: Part,
        /// What it holds.
        kind: PartKind,
        /// The provider's position, when the wire states one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_order: Option<SourceOrder>,
        /// The tool a function or custom call names as it opens; `None`
        /// for every other kind. The end states the name it settled on.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<ToolName>,
        /// The tool's exact namespace, when known at its start. The end
        /// retains a namespace the provider states only when it closes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        /// Explicit pending-name assembly, used only by preserved Responses.
        /// A second Start may bind a PendingName slot to Named exactly once.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_slot_state: Option<ToolSlotState>,
    },
    /// Preserved Responses returned to an already started, still open text
    /// Part after interleaving another part. This is no second Start and
    /// creates no additional canonical content.
    Resume {
        /// The same text Part being reactivated.
        part: Part,
    },
    /// A text part grew.
    Text {
        /// The part.
        part: Part,
        /// The fragment.
        text: String,
    },
    /// A reasoning part grew.
    Reasoning {
        /// The part.
        part: Part,
        /// The fragment.
        text: String,
    },
    /// A function call's arguments grew. The fragments of one call, joined,
    /// are the argument JSON the provider sent; read an incomplete prefix
    /// with [`parse_partial_arguments`](super::parse_partial_arguments).
    Arguments {
        /// The part.
        part: Part,
        /// The fragment of raw argument JSON.
        json: String,
    },
    /// A custom call's raw input grew. This is verbatim text, never JSON
    /// function arguments, even when the text happens to parse as JSON.
    Input {
        /// The part.
        part: Part,
        /// The raw input fragment.
        input: String,
    },
    /// A part ended with the content it finalized.
    End {
        /// The part.
        part: Part,
        /// The finalized content.
        content: AssistantContent,
        /// How a preserved reasoning part obtained its final content.
        /// Absent on default stream events and non-reasoning parts.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_authority: Option<ReasoningAuthority>,
    },
    /// A started function call closed without producing a canonical call.
    /// Previously delivered argument fragments remain part of the stream.
    Discard {
        /// The function-call part being closed.
        part: Part,
        /// The reported reason or observed terminal boundary that closed it.
        reason: DiscardReason,
    },
}

/// Whether an explicitly preserved function-call slot has its genuine tool
/// name. Pending arguments belong to a real Part, without a canonical call;
/// only a later named Start on that same Part binds the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSlotState {
    /// The provider has delivered no nonempty tool name for this slot.
    PendingName,
    /// The provider bound the pending slot to its genuine tool name.
    Named,
}

/// Whether preserved reasoning was restated by the provider or closed from
/// partial observations. This describes only the reasoning part's evidence;
/// it does not establish a response terminal or alter canonical history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningAuthority {
    /// No complete provider reasoning item restated this part before its flush.
    Partial,
    /// A done item, whole response or terminal body restated this reasoning.
    /// A later error or EOF does not erase that already observed authority.
    ProviderRestated,
}

/// Why a function-call assembly closed without a completed call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscardReason {
    /// The provider explicitly reported its output-length limit.
    Length,
    /// The provider's authoritative function item states it is incomplete.
    Incomplete,
    /// A genuine response terminal arrived before the function-call item
    /// completed, and its accumulated arguments were malformed.
    UnfinishedAtTerminal,
}

impl StreamEvent {
    /// The part this event is about.
    pub const fn part(&self) -> Part {
        match self {
            Self::Start { part, .. }
            | Self::Resume { part }
            | Self::Text { part, .. }
            | Self::Reasoning { part, .. }
            | Self::Arguments { part, .. }
            | Self::Input { part, .. }
            | Self::End { part, .. }
            | Self::Discard { part, .. } => *part,
        }
    }

    /// A stable variant name that exposes no payload to logs.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Start { .. } => "Start",
            Self::Resume { .. } => "Resume",
            Self::Text { .. } => "Text",
            Self::Reasoning { .. } => "Reasoning",
            Self::Arguments { .. } => "Arguments",
            Self::Input { .. } => "Input",
            Self::End { .. } => "End",
            Self::Discard { .. } => "Discard",
        }
    }
}

/// One item of a stream: an event of the operation, or a payload the
/// provider sent that the decoder does not model. An unmodeled payload
/// always reaches the consumer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "item", content = "value", rename_all = "snake_case")]
pub enum Item<E> {
    /// An event of the operation.
    Event(E),
    /// A payload the decoder does not model.
    Unknown(UnknownPayload),
}

/// A completion stream's items read back from their serialized form, in an
/// order the writer could have produced: each part opens at a position no
/// part took before, grows only while open and with its own kind of fragment,
/// and ends once. An explicit pending tool slot binds its name once on the
/// same position. Positions follow the order provider
/// items opened, so a dropped item leaves a gap. Unmodeled payloads may come
/// anywhere.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    items: Vec<Item<StreamEvent>>,
    /// Each started part's kind, and whether it ended.
    parts: std::collections::BTreeMap<usize, (PartKind, bool, Option<ToolSlotState>)>,
}

impl PartialEq for Transcript {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl Serialize for Transcript {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.items.serialize(serializer)
    }
}

/// Why a serialized event sequence is not one the writer could produce.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SequenceError {
    /// The value is not a list of stream items.
    #[error("not a list of stream items: {0}")]
    Shape(String),
    /// An event names a part that has not started, or a start reuses a
    /// position.
    #[error("item {0} names a part that has not started")]
    UnknownPart(usize),
    /// A part ended twice, or grew after its end.
    #[error("item {0} touches a part that already ended")]
    EndedTwice(usize),
    /// A fragment or end does not match the part's kind, or a start names a
    /// tool for a part that is not a call, or none for one that is.
    #[error("item {0} does not match its part's kind")]
    WrongKind(usize),
    /// A pending slot was bound twice, bound on another Part, or closed
    /// before the provider supplied its name.
    #[error("item {0} has an invalid tool-slot transition")]
    ToolSlotTransition(usize),
    /// A part started and never ended.
    #[error("part {0} never ended")]
    Unclosed(usize),
}

impl Transcript {
    /// Read a serialized item sequence, refusing one the writer could not
    /// have produced. Every part must have ended.
    pub fn parse(value: serde_json::Value) -> Result<Self, SequenceError> {
        let transcript = Self::parse_prefix(value)?;
        transcript.check_closed()?;
        Ok(transcript)
    }

    /// [`Self::parse`] for a stream that stopped early (an error or a
    /// cancellation): parts may still be open at its end.
    pub fn parse_prefix(value: serde_json::Value) -> Result<Self, SequenceError> {
        let items: Vec<Item<StreamEvent>> = serde_json::from_value(value)
            .map_err(|error| SequenceError::Shape(error.to_string()))?;
        Self::from_items(items)
    }

    /// Append the next item a stream yielded, refusing one the writer could
    /// not have produced after the items so far.
    pub fn push(&mut self, item: Item<StreamEvent>) -> Result<(), SequenceError> {
        let position = self.items.len();
        if let Item::Event(event) = &item {
            let index = event.part().index();
            let kind = match event {
                StreamEvent::Start {
                    kind,
                    name,
                    namespace,
                    tool_slot_state,
                    ..
                } => {
                    let call = matches!(kind, PartKind::ToolCall | PartKind::CustomToolCall);
                    match tool_slot_state {
                        None => {
                            if call != name.is_some() || (!call && namespace.is_some()) {
                                return Err(SequenceError::WrongKind(position));
                            }
                            if self.parts.contains_key(&index) {
                                return Err(SequenceError::UnknownPart(position));
                            }
                            self.parts.insert(index, (*kind, false, None));
                        }
                        Some(ToolSlotState::PendingName) => {
                            if *kind != PartKind::ToolCall || name.is_some() {
                                return Err(SequenceError::WrongKind(position));
                            }
                            if self.parts.contains_key(&index) {
                                return Err(SequenceError::ToolSlotTransition(position));
                            }
                            self.parts
                                .insert(index, (*kind, false, Some(ToolSlotState::PendingName)));
                        }
                        Some(ToolSlotState::Named) => {
                            if *kind != PartKind::ToolCall || name.is_none() {
                                return Err(SequenceError::WrongKind(position));
                            }
                            match self.parts.get_mut(&index) {
                                Some((
                                    PartKind::ToolCall,
                                    false,
                                    state @ Some(ToolSlotState::PendingName),
                                )) => {
                                    *state = Some(ToolSlotState::Named);
                                }
                                None
                                | Some((_, _, None))
                                | Some((_, _, Some(ToolSlotState::Named)))
                                | Some((_, _, Some(ToolSlotState::PendingName))) => {
                                    return Err(SequenceError::ToolSlotTransition(position));
                                }
                            }
                        }
                    }
                    None
                }
                StreamEvent::Text { .. } => Some(PartKind::Text),
                StreamEvent::Resume { .. } => Some(PartKind::Text),
                StreamEvent::Reasoning { .. } => Some(PartKind::Reasoning),
                StreamEvent::Arguments { .. } => Some(PartKind::ToolCall),
                StreamEvent::Input { .. } => Some(PartKind::CustomToolCall),
                StreamEvent::Discard { .. } => Some(PartKind::ToolCall),
                StreamEvent::End {
                    content,
                    reasoning_authority,
                    ..
                } => {
                    let kind = match content {
                        AssistantContent::Text(_) | AssistantContent::CarriedText(_) => {
                            PartKind::Text
                        }
                        AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_) => {
                            PartKind::Reasoning
                        }
                        AssistantContent::ToolCall(_) => PartKind::ToolCall,
                        AssistantContent::CustomToolCall(_) => PartKind::CustomToolCall,
                        AssistantContent::ProviderItem(_) => PartKind::ProviderItem,
                        AssistantContent::Image(_) => PartKind::Image,
                        AssistantContent::Opaque(_) => PartKind::Opaque,
                    };
                    match reasoning_authority {
                        None => {}
                        Some(
                            ReasoningAuthority::Partial | ReasoningAuthority::ProviderRestated,
                        ) if kind != PartKind::Reasoning => {
                            return Err(SequenceError::WrongKind(position));
                        }
                        Some(_) => {}
                    }
                    Some(kind)
                }
            };
            if let Some(kind) = kind {
                match self.parts.get_mut(&index) {
                    None => return Err(SequenceError::UnknownPart(position)),
                    Some((_, true, _)) => return Err(SequenceError::EndedTwice(position)),
                    Some((open, false, _)) if *open != kind => {
                        return Err(SequenceError::WrongKind(position));
                    }
                    Some((_, ended, state)) => {
                        match state {
                            Some(ToolSlotState::PendingName) => {
                                if matches!(
                                    event,
                                    StreamEvent::End { .. } | StreamEvent::Discard { .. }
                                ) {
                                    return Err(SequenceError::ToolSlotTransition(position));
                                }
                            }
                            None | Some(ToolSlotState::Named) => {}
                        }
                        *ended =
                            matches!(event, StreamEvent::End { .. } | StreamEvent::Discard { .. })
                    }
                }
            }
        }
        self.items.push(item);
        Ok(())
    }

    fn check_closed(&self) -> Result<(), SequenceError> {
        match self.parts.iter().find(|(_, (_, ended, _))| !ended) {
            Some((part, _)) => Err(SequenceError::Unclosed(*part)),
            None => Ok(()),
        }
    }

    /// Items a stream yielded, checked as [`Self::push`] checks them: a
    /// stream that stopped early may leave parts open.
    pub fn from_items(items: Vec<Item<StreamEvent>>) -> Result<Self, SequenceError> {
        let mut transcript = Self::default();
        for item in items {
            transcript.push(item)?;
        }
        Ok(transcript)
    }

    /// The number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The items, in order.
    pub fn items(&self) -> &[Item<StreamEvent>] {
        &self.items
    }

    /// The items, in order.
    pub fn into_items(self) -> Vec<Item<StreamEvent>> {
        self.items
    }

    /// The events, in order, without the unmodeled payloads.
    pub fn events(&self) -> impl Iterator<Item = &StreamEvent> {
        self.items.iter().filter_map(|item| match item {
            Item::Event(event) => Some(event),
            Item::Unknown(_) => None,
        })
    }
}

impl From<Transcript> for Vec<Item<StreamEvent>> {
    fn from(transcript: Transcript) -> Self {
        transcript.items
    }
}

impl<'de> Deserialize<'de> for Transcript {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::parse_prefix(value).map_err(serde::de::Error::custom)
    }
}

// The stream vocabulary crosses threads on every target: the bus sends it
// over a channel, the effect log records it, and a client reads it back.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<StreamEvent>();
    assert_wire::<Transcript>();
    assert_wire::<Item<StreamEvent>>();
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod authority_tests;

#[cfg(test)]
mod pending_tests;

#[cfg(test)]
mod resume_tests;
