//! Generating an assistant turn. A completion decoder writes its reply one
//! provider item at a time, keyed by the item's wire index: the item opens
//! when its first event arrives and takes the next position in the reply,
//! its fragments grow it, and closing it finalizes its block with the
//! provider's item as the block's native. Nothing regroups: blocks keep the
//! order their items opened in. The provider's end of the reply is a
//! [`Finish`], which the fold needs to produce the response.
//!
//! ```
//! use rig_core::operation::Finish;
//! use rig_core::completion::FinishReason;
//!
//! let finish = Finish {
//!     reason: Some(FinishReason::Stop),
//!     ..Finish::default()
//! };
//! assert_eq!(finish.reason, Some(FinishReason::Stop));
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::completion::{CompletionRequest, CompletionResponse, FinishReason, Usage};
use crate::error::ProviderError;
use crate::message::citation;
use crate::message::{
    Api, AssistantContent, CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText, CustomToolCall,
    Image, LocalCallId, Opaque, Origin, ProviderItem, Reasoning, Text, ToolCall, ToolFunction,
    ToolName,
};
use crate::streaming::{DiscardReason, Item, Part, PartKind, ReasoningAuthority, SourceOrder, StreamEvent, ToolSlotState};
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{
    Assembled, Call, Descriptor, Emit, Fold, Mode, Operation, Out, Reply, Shared, WireCitation,
};

/// Generating an assistant turn, unary or streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion;

impl Operation for Completion {
    type Request = CompletionRequest;
    type Event = StreamEvent;
    type End = Finish;
    type Response = CompletionResponse;
    type Fold = Turn;
    type Emit = Assembled;

    /// The call's span names the model the request overrides to, when it
    /// names one: every wire honours the override on encode.
    fn fold(request: &Self::Request, call: &mut Call<'_>) -> Self::Fold {
        let telemetry = call
            .wire
            .telemetry
            .map_or(GenAiOperation::Chat, |telemetry| telemetry(call.mode));
        debug_assert!(telemetry.is_completion());
        // A wire that picks its API per request records the one it used.
        let replay = call
            .wire
            .replay
            .map(|target| target.route(request).unwrap_or(target));
        let model = request
            .model
            .as_deref()
            .or(call.wire.model)
            .or_else(|| replay.map(|target| target.model()))
            .unwrap_or_default();
        let span = SpanBuilder::new(call.wire.name, model, telemetry)
            .streaming(matches!(call.mode, Mode::Streaming))
            .system_instructions(
                request.system_instructions(),
                request.record_telemetry_content,
            )
            .build();
        call.instrument(span.clone());
        let api = replay.map_or_else(
            || Api::from(call.wire.name.to_owned()),
            |target| target.api(),
        );
        let mut origin = Origin::new(api, call.wire.name, model);
        // Only a wire that binds items to the context needs it recorded.
        if replay.is_some_and(|target| target.binds_context(model)) {
            origin.context =
                replay.map(|target| crate::completion::history::context_of(request, target, model));
        }
        Turn {
            span,
            wire: replay.is_some_and(|target| target.states_finish_reason()),
            call_id_slot: replay.and_then(|target| target.call_id_slot()),
            accept_unknown_finish: request.accept_unknown_finish_reasons,
            ..Turn::new(origin)
        }
    }

    /// The reply's normalized usage.
    fn observed_usage(finish: &Finish) -> Option<&crate::completion::Usage> {
        Some(&finish.usage)
    }

    /// Resolve the model the request addresses once, as the request's
    /// `model`: the one it names, else the wire's. The history is checked
    /// with [`CompletionRequest::validate_message_content`], then
    /// [`adapt`](crate::completion::adapt)ed for that model on the wire's
    /// replay target and checked again. Encoders, the fold and replay all
    /// read the one resolved model. Each option the routed target refuses
    /// is reported first, through
    /// [`options::check`](crate::completion::options::check).
    fn prepare(
        mut request: Self::Request,
        wire: &Descriptor<'_>,
    ) -> Result<Self::Request, ProviderError> {
        let Some(target) = wire.replay else {
            return Err(ProviderError::request(format!(
                "completion wire `{}` names no replay target",
                wire.name
            )));
        };
        // The caller's history is checked as written, so a rejection names
        // its own messages; adapting never empties a message it keeps.
        request.validate_message_content()?;
        // A wire that addresses no model (an interaction read back) leaves
        // it to the reply, which names the model the turn is from.
        request.model = request
            .model
            .take()
            .filter(|model| !model.is_empty())
            .or_else(|| Some(target.model().to_owned()).filter(|model| !model.is_empty()));
        let target = target.route(&request).unwrap_or(target);
        // Check the original carried items before any shared history path
        // can trim or reshape a turn. This refusal retains its concrete type.
        target
            .check_provider_items(&request.chat_history)
            .map_err(ProviderError::request)?;
        target
            .check_nested_content(&request.chat_history)
            .map_err(ProviderError::request)?;
        // Every option the routed wire cannot honour is reported here, before
        // any wire encodes, whatever its encoder does.
        crate::completion::options::check(target, &mut request)?;
        target
            .check_documents(&request)
            .map_err(ProviderError::from)?;
        crate::completion::history::check_images(&request.chat_history, target)?;
        // Documents join the history before it is adapted, so the adapter's
        // rules apply to them, unless the encoder sends them itself.
        if !target.takes_documents() {
            request.chat_history = request.chat_history_with_documents();
            request.documents.clear();
        }
        let stored = target.continues_stored(&request);
        let shape = crate::completion::history::Request {
            model: request.model.as_deref(),
            stored,
            // A conversation the provider stores holds its own tools, so a
            // continuation that declares none still calls them.
            tools: stored || target.declares_tools(&request),
            context: (!target.drops_unbound_items(&request)).then(|| {
                crate::completion::history::context_of(
                    &request,
                    target,
                    request.model.as_deref().unwrap_or(target.model()),
                )
            }),
        };
        request.chat_history =
            crate::completion::history::adapt_for(&request.chat_history, target, &shape);
        request.validate_message_content()?;
        Ok(request)
    }
}

impl crate::wire::reply::Closing<Completion> for Assembled {
    fn close(shared: &mut Shared<Completion>) {
        let Shared { fold, items, .. } = shared;
        fold.close_open(items);
    }
}

impl Emit<Completion> for Assembled {}

/// What the provider sends when it ends a completion reply.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Finish {
    /// Token usage the provider reported. A counter it did not report is
    /// `None`. A cost the provider reported goes in `usage.cost` and is
    /// kept; without one the fold prices the counters at the catalog's
    /// [`Pricing`](crate::catalog::Pricing) for the request's model, and
    /// leaves the cost `None` when the catalog has no price for it.
    pub usage: Usage,
    /// Why the model stopped, when the provider said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<FinishReason>,
    /// The independent assistant message ID, only when the provider supplied it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<crate::completion::ProviderMessageId>,
    /// The response id.
    pub response_id: Option<String>,
    /// The model the provider reports.
    pub model: Option<String>,
    /// The provider's report that the turn failed, such as a refusal's
    /// explanation. Upstream-native content is not replayed from a failed
    /// turn; carried ProviderItems retain their separate replay contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a provider item becomes when it opens.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Answer text.
    Text,
    /// Reasoning, possibly withheld by the provider.
    Reasoning {
        /// Whether the provider withheld the text.
        redacted: bool,
    },
    /// An image, its canonical fields stated at open; fragments append
    /// base64 data.
    Image(Image),
    /// A tool call with its id and name.
    Call {
        /// The call's id.
        id: CallId,
        /// The tool's name.
        name: ToolName,
        /// The exact namespace known at open, including an empty value.
        namespace: Option<String>,
    },
    /// A custom tool call. Its input grows as raw text, never JSON.
    CustomCall {
        /// The call's id.
        id: CallId,
        /// The custom tool name.
        name: ToolName,
        /// The exact namespace, including an explicitly empty value.
        namespace: Option<String>,
    },
    /// A carried item whose issuer remains separate from the turn's Origin.
    ProviderItem {
        /// The item's per-item issuer; None means unknown provenance.
        provider: Option<String>,
    },
    /// An item with no canonical meaning.
    Opaque {
        /// Whether the item goes back to the model that produced it.
        replay: bool,
    },
}

/// The completion fold. The driver and the bus writer build it; no other
/// code can feed one.
///
/// It is both sides of one reply: the writer state a decoder writes through
/// (the open items by wire index) and the blocks the consumer has taken, in
/// their position.
pub struct Turn {
    span: tracing::Span,
    /// Who the reply is from; the end adds the provider's model and id.
    origin: Origin,
    // The writer.
    open: BTreeMap<usize, Draft>,
    /// The position of the block each wire index last closed, whose item
    /// [`Out::edit`] may still amend until the reply ends.
    ended: HashMap<usize, usize>,
    next_part: u32,
    /// The provider coordinate active while a decoder writes one item.
    source_order: Option<SourceOrder>,
    /// Coordinates by branded part position, separate from wire indices.
    source_orders: BTreeMap<usize, SourceOrder>,
    call_ids: HashSet<CallId>,
    /// Whether the reply's end must name a finish reason: a wire's fold
    /// whose target states one (a relayed or written reply states its own).
    wire: bool,
    /// Where the wire's call items hold their id ([`ReplayTarget::call_id_slot`]).
    ///
    /// [`ReplayTarget::call_id_slot`]: crate::completion::ReplayTarget::call_id_slot
    call_id_slot: Option<&'static str>,
    /// Whether the request accepts an unknown finish reason as a normal stop
    /// ([`CompletionRequest::accept_unknown_finish_reasons`]).
    ///
    /// [`CompletionRequest::accept_unknown_finish_reasons`]: field@CompletionRequest::accept_unknown_finish_reasons
    accept_unknown_finish: bool,
    /// The first position whose item closed without the provider stating it
    /// complete. An item there may be the partner a later one needs, so
    /// blocks from it on replay from their canonical fields.
    first_incomplete: Option<usize>,
    /// Whether a call was still open when the provider ended the reply.
    unfinished_call: bool,
    /// The call the last index-less fragment went to.
    last_call: Option<usize>,
    /// The index of the block a boundary-less wire is streaming.
    run: Option<usize>,
    next_auto: usize,
    /// The wire index of each position, for a wire whose choice follows
    /// its indices ([`Out::order_by_index`]).
    by_index: Option<BTreeMap<usize, usize>>,
    /// The position of the block the choice holds first ([`Out::lead`]).
    lead: Option<usize>,
    // The fold.
    choice: Vec<Option<AssistantContent>>,
    /// The text the consumer took of text and reasoning parts still open,
    /// by position.
    open_text: BTreeMap<usize, AssistantContent>,
    /// An explicit terminal issuer for carried response fields only.
    /// Already yielded events retain the provenance they originally stated.
    carried_issuer: Option<String>,
    /// Parts explicitly closed without content, rather than left unfinished.
    discarded: HashSet<usize>,
}

/// One open provider item.
struct Draft {
    part: Part,
    source_order: Option<SourceOrder>,
    /// Whether its start was emitted.
    started: bool,
    /// Explicit Responses-only pending-name assembly; default calls keep None.
    tool_slot_state: Option<ToolSlotState>,
    /// The provider's item as assembled so far; `Null` for none.
    item: serde_json::Value,
    body: Body,
    /// The citations of a text item, resolved when it closes.
    citations: Vec<WireCitation>,
}

/// Whether a closing item keeps its provider item as the block's native.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Closing {
    /// The provider stated the item complete: it becomes the native.
    Complete,
    /// The item never completed: the block has no native, and an opaque
    /// item does not replay.
    Incomplete,
}

enum Body {
    Text(String),
    Image(Image),
    Reasoning {
        text: String,
        redacted: bool,
    },
    Call {
        id: Option<CallId>,
        name: String,
        namespace: Option<String>,
        arguments: Arguments,
    },
    CustomCall {
        id: CallId,
        name: ToolName,
        namespace: Option<String>,
        input: String,
        streamed: usize,
    },
    ProviderItem {
        provider: Option<String>,
    },
    Opaque {
        replay: bool,
    },
}

impl Body {
    fn kind(&self) -> PartKind {
        match self {
            Self::Text(_) => PartKind::Text,
            Self::Reasoning { .. } => PartKind::Reasoning,
            Self::Image(_) => PartKind::Image,
            Self::Call { .. } => PartKind::ToolCall,
            Self::CustomCall { .. } => PartKind::CustomToolCall,
            Self::ProviderItem { .. } => PartKind::ProviderItem,
            Self::Opaque { .. } => PartKind::Opaque,
        }
    }
}

/// A tool call's argument text as it arrives.
#[derive(Default)]
struct Arguments {
    text: String,
    /// Whether any argument fragment arrived, including an empty one.
    /// Only the named Interactions and Bedrock terminal hooks read this distinction.
    saw_fragments: bool,
    /// How many bytes of `text` the stream has carried.
    streamed: usize,
    overflowed: bool,
    /// Whether any fragment carried a non-blank byte.
    substantive: bool,
    /// Arguments the provider announced when the call opened, used only
    /// when no fragment arrives.
    announced: Option<serde_json::Value>,
    /// A carried Responses JSON value that is valid but not an argument object.
    /// Set only at its provider boundary; further fragments or restatements clear it.
    preserved_non_object: Option<String>,
}

/// The most argument bytes one call accumulates.
const MAX_TOOL_INPUT_BYTES: usize = 32 * 1024 * 1024;

impl Arguments {
    fn push(&mut self, fragment: &str, name: &str) {
        self.saw_fragments = true;
        self.preserved_non_object = None;
        self.substantive |= !fragment.trim().is_empty();
        // Some OpenAI-compatible gateways send a literal `null` before the
        // real fragments; a non-blank fragment supersedes it.
        if self.text.trim() == "null" && !fragment.trim().is_empty() {
            self.text.clear();
        }
        if self.text.len().saturating_add(fragment.len()) > MAX_TOOL_INPUT_BYTES {
            if !self.overflowed {
                self.overflowed = true;
                tracing::warn!(
                    tool = name,
                    "streamed tool-call input exceeded the accumulation bound; truncating"
                );
            }
        } else {
            self.text.push_str(fragment);
        }
    }

    /// The call to `name` these arguments make: the announced ones when no
    /// fragment arrived, else the text, read by [`ToolFunction::parse`].
    fn function(&self, name: ToolName) -> ToolFunction {
        if self.text.is_empty()
            && let Some(announced) = &self.announced
        {
            return ToolFunction::new(name, announced.clone());
        }
        let mut function = ToolFunction::parse(name, &self.text);
        if self.overflowed {
            function.invalid_arguments = Some(self.text.clone());
        }
        if let Some(raw) = &self.preserved_non_object {
            function.arguments.clear();
            function.invalid_arguments = Some(raw.clone());
        }
        function
    }

    /// The text the stream has not carried yet, now marked carried. A text
    /// that may still be a gateway's placeholder `null` waits, since the
    /// first real fragment replaces it.
    fn unstreamed(&mut self) -> Option<String> {
        if "null".starts_with(self.text.trim()) {
            return None;
        }
        self.unstreamed_exact()
    }

    /// Preserved Responses carries real fragments even before a name or a
    /// complete JSON token arrives. Default wires still withhold null above.
    fn unstreamed_exact(&mut self) -> Option<String> {
        let fragment = self.text.get(self.streamed..)?;
        if fragment.is_empty() {
            return None;
        }
        let fragment = fragment.to_owned();
        self.streamed = self.text.len();
        Some(fragment)
    }

    /// Whether the text is a complete JSON object.
    fn complete(&self) -> bool {
        !self.overflowed
            && matches!(
                crate::json_utils::parse_tool_arguments(&self.text),
                Ok(serde_json::Value::Object(_))
            )
    }
}

/// Whether `id` names a call: some providers send `""` or `"null"` for a
/// call they gave no id.
fn stated(id: &str) -> bool {
    !id.is_empty() && id != "null"
}

/// What one fragment of a buffered tool call carries.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallFragment<'a> {
    /// The provider's id for the call.
    pub id: Option<&'a str>,
    /// The tool's name.
    pub name: Option<&'a str>,
    /// A fragment of the argument JSON.
    pub arguments: Option<&'a str>,
}

/// The first index the writer hands out itself ([`Out::fresh_index`]); a
/// provider's wire indices stay below it.
pub const AUTO_INDEX: usize = 1 << 30;

fn not_open(index: usize) -> ProviderError {
    ProviderError::Response(format!(
        "the reply wrote to item {index}, which is not open"
    ))
}

/// Carry what the open call `draft` has not streamed yet: its start once it
/// names a tool, then the argument text since the last fragment. A call
/// that never names a tool stays unseen, as its close drops it.
fn stream_call(items: &mut Items, draft: &mut Draft) {
    if let Body::CustomCall {
        name,
        namespace,
        input,
        streamed,
        ..
    } = &mut draft.body
    {
        if !draft.started {
            draft.started = true;
            emit(
                items,
                StreamEvent::Start {
                    part: draft.part,
                    source_order: draft.source_order,
                    kind: PartKind::CustomToolCall,
                    name: Some(name.clone()),
                    namespace: namespace.clone(),
                    tool_slot_state: None,
                },
            );
        }
        if let Some(fragment) = input.get(*streamed..).filter(|text| !text.is_empty()) {
            emit(
                items,
                StreamEvent::Input {
                    part: draft.part,
                    input: fragment.to_owned(),
                },
            );
            *streamed = input.len();
        }
        return;
    }
    let Body::Call {
        name,
        namespace,
        arguments,
        ..
    } = &mut draft.body
    else {
        return;
    };
    let start = match draft.tool_slot_state {
        None => {
            if !draft.started {
                let Ok(name) = ToolName::new(name.clone()) else { return; };
                Some((Some(name), None))
            } else { None }
        }
        Some(ToolSlotState::PendingName) => {
            if let Ok(name) = ToolName::new(name.clone()) {
                draft.tool_slot_state = Some(ToolSlotState::Named);
                Some((Some(name), Some(ToolSlotState::Named)))
            } else if !draft.started {
                Some((None, Some(ToolSlotState::PendingName)))
            } else { None }
        }
        Some(ToolSlotState::Named) => None,
    };
    if let Some((name, tool_slot_state)) = start {
        draft.started = true;
        emit(items, StreamEvent::Start {
            part: draft.part,
            source_order: draft.source_order,
            kind: PartKind::ToolCall,
            name,
            namespace: namespace.clone(),
            tool_slot_state,
        });
    }
    let json = match draft.tool_slot_state {
        None => arguments.unstreamed(),
        Some(ToolSlotState::PendingName | ToolSlotState::Named) => arguments.unstreamed_exact(),
    };
    if let Some(json) = json {
        emit(
            items,
            StreamEvent::Arguments {
                part: draft.part,
                json,
            },
        );
    }
}

impl Turn {
    /// The writer of a reply from `origin`.
    pub(crate) fn new(origin: Origin) -> Self {
        Self {
            span: tracing::Span::none(),
            origin,
            open: BTreeMap::new(),
            ended: HashMap::new(),
            next_part: 0,
            source_order: None,
            source_orders: BTreeMap::new(),
            call_ids: HashSet::new(),
            wire: false,
            call_id_slot: None,
            accept_unknown_finish: false,
            first_incomplete: None,
            unfinished_call: false,
            last_call: None,
            run: None,
            next_auto: AUTO_INDEX,
            by_index: None,
            lead: None,
            choice: Vec::new(),
            open_text: BTreeMap::new(),
            carried_issuer: None,
            discarded: HashSet::new(),
        }
    }

    /// The fold of a stream relayed from another fold, which built its
    /// events: it only collects them, and takes its origin from the relay
    /// ([`Self::set_origin`]). Until then it names the relay `label`.
    pub(crate) fn relayed(label: impl Into<String>) -> Self {
        let label = label.into();
        Self::new(Origin::new(label.clone(), label.clone(), label))
    }

    /// Who the reply is from, as the relay or the writer's owner states it.
    pub(crate) fn set_origin(&mut self, origin: Origin) {
        self.origin = origin;
    }

    /// The next position in the choice.
    fn next(&mut self) -> Part {
        let part = Part::new(self.next_part);
        self.next_part += 1;
        if let Some(order) = self.source_order {
            self.source_orders.insert(part.index(), order);
        }
        part
    }

    fn fresh_index(&mut self) -> usize {
        let index = self.next_auto;
        self.next_auto += 1;
        index
    }

    fn draft(&mut self, index: usize) -> Result<&mut Draft, ProviderError> {
        self.open.get_mut(&index).ok_or_else(|| not_open(index))
    }

    pub(crate) fn open_item(
        &mut self,
        items: &mut Items,
        index: usize,
        block: Block,
        item: serde_json::Value,
    ) -> Result<(), ProviderError> {
        if self.open.contains_key(&index) {
            return Err(ProviderError::Response(format!(
                "the reply opened item {index} twice"
            )));
        }
        let body = match block {
            Block::Text => Body::Text(String::new()),
            Block::Image(image) => Body::Image(image),
            Block::Reasoning { redacted } => Body::Reasoning {
                text: String::new(),
                redacted,
            },
            Block::Call {
                id,
                name,
                namespace,
            } => Body::Call {
                id: Some(id),
                name: name.into(),
                namespace,
                arguments: Arguments::default(),
            },
            Block::Opaque { replay } => Body::Opaque { replay },
            Block::CustomCall {
                id,
                name,
                namespace,
            } => Body::CustomCall {
                id,
                name,
                namespace,
                input: String::new(),
                streamed: 0,
            },
            Block::ProviderItem { provider } => Body::ProviderItem { provider },
        };
        self.insert(index, body, item);
        stream_call(items, self.draft(index)?);
        Ok(())
    }

    fn insert(&mut self, index: usize, body: Body, item: serde_json::Value) {
        self.ended.remove(&index);
        let part = self.next();
        if let Some(by_index) = &mut self.by_index {
            by_index.insert(part.index(), index);
        }
        let started = false;
        self.open.insert(
            index,
            Draft {
                part,
                source_order: self.source_order,
                started,
                tool_slot_state: None,
                item,
                body,
                citations: Vec::new(),
            },
        );
    }

    pub(crate) fn push_item(
        &mut self,
        items: &mut Items,
        index: usize,
        fragment: &str,
    ) -> Result<(), ProviderError> {
        if fragment.is_empty() {
            return Ok(());
        }
        let draft = self.draft(index)?;
        let preserved = draft.tool_slot_state.is_some();
        let event = match &mut draft.body {
            Body::Text(text) => {
                text.push_str(fragment);
                StreamEvent::Text {
                    part: draft.part,
                    text: fragment.to_owned(),
                }
            }
            Body::Reasoning { text, .. } => {
                text.push_str(fragment);
                StreamEvent::Reasoning {
                    part: draft.part,
                    text: fragment.to_owned(),
                }
            }
            Body::Call {
                name, arguments, ..
            } => {
                if preserved && arguments.text.trim() == "null" && !fragment.trim().is_empty() {
                    arguments.streamed = 0;
                }
                arguments.push(fragment, name);
                stream_call(items, draft);
                return Ok(());
            }
            Body::CustomCall { input, .. } => {
                input.push_str(fragment);
                stream_call(items, draft);
                return Ok(());
            }
            Body::Image(image) => {
                if let crate::message::DocumentSourceKind::Base64(data) = &mut image.data {
                    data.push_str(fragment);
                    return Ok(());
                }
                return Err(ProviderError::Response(format!(
                    "the reply wrote data to the image item {index}, which holds no base64 data"
                )));
            }
            Body::Opaque { .. } | Body::ProviderItem { .. } => {
                return Err(ProviderError::Response(format!(
                    "the reply wrote text to the opaque item {index}"
                )));
            }
        };
        if !draft.started {
            draft.started = true;
            emit(
                items,
                StreamEvent::Start {
                    part: draft.part,
                    source_order: draft.source_order,
                    kind: draft.body.kind(),
                    name: None,
                    namespace: None,
                    tool_slot_state: None,
                },
            );
        }
        emit(items, event);
        Ok(())
    }

    /// The id a call closing now takes, and the provider item it keeps:
    /// `id`, or, when the provider sent none or one an earlier call of the
    /// reply holds, a fresh rig-issued id (as pi issues). A renamed call
    /// keeps no item, since the item names the id it lost; a call sent with
    /// no id keeps its item only on a wire whose items have an id slot,
    /// which replay then fills with the id its result gets.
    fn distinct_call_id(
        &mut self,
        id: Option<CallId>,
        item: serde_json::Value,
    ) -> (CallId, serde_json::Value) {
        let (id, item) = match id {
            Some(id) if !self.call_ids.contains(&id) => (id, item),
            Some(id) => {
                tracing::warn!(%id, "the provider named two tool calls with one id; renaming the second");
                (CallId::Local(LocalCallId::new()), serde_json::Value::Null)
            }
            None if self.call_id_slot.is_some() => (CallId::Local(LocalCallId::new()), item),
            None => (CallId::Local(LocalCallId::new()), serde_json::Value::Null),
        };
        self.call_ids.insert(id.clone());
        (id, item)
    }

    /// Close the item at `index`: its block becomes visible, holding the
    /// provider's item as its native when `closing` says it is complete.
    fn close_item(
        &mut self,
        items: &mut Items,
        index: usize,
        closing: Closing,
    ) -> Result<(), ProviderError> {
        self.close_item_with_reasoning_authority(items, index, closing, None)
    }

    fn close_item_with_reasoning_authority(
        &mut self,
        items: &mut Items,
        index: usize,
        closing: Closing,
        reasoning_authority: Option<ReasoningAuthority>,
    ) -> Result<(), ProviderError> {
        if reasoning_authority.is_some()
            && !matches!(self.open.get(&index).map(|draft| &draft.body), Some(Body::Reasoning { .. }))
        {
            return Err(ProviderError::Response(format!(
                "the reply assigned reasoning authority to item {index}, which is not reasoning"
            )));
        }
        let mut draft = self.open.remove(&index).ok_or_else(|| not_open(index))?;
        // A call streams what it has not yet: its start, if it names a tool,
        // and the argument text since its last fragment.
        stream_call(items, &mut draft);
        if self.run == Some(index) {
            self.run = None;
        }
        self.ended.insert(index, draft.part.index());
        let Draft {
            part,
            source_order,
            started,
            item,
            body,
            citations,
            tool_slot_state: _,
        } = draft;
        let item = match (closing, &body) {
            (Closing::Complete, _)
            | (Closing::Incomplete, Body::Opaque { .. } | Body::ProviderItem { .. }) => item,
            (Closing::Incomplete, _) => {
                self.cut_at(part.index());
                serde_json::Value::Null
            }
        };
        let content = match body {
            Body::Text(text) => {
                if text.is_empty() && item.is_null() {
                    return Ok(());
                }
                // Resolved at close: a provider may cite a text before its
                // last fragment arrives.
                let mut text = Text::new(text);
                citation::attach(
                    &mut text,
                    Vec::new(),
                    citations,
                    &self.origin.provider,
                    index,
                );
                with_item(AssistantContent::Text(text), item)
            }
            Body::Reasoning { text, redacted } => {
                if text.is_empty() && !redacted && item.is_null() {
                    return Ok(());
                }
                let reasoning = Reasoning {
                    text,
                    redacted,
                    native: None,
                };
                with_item(AssistantContent::Reasoning(reasoning), item)
            }
            Body::Image(image) => with_item(AssistantContent::Image(image), item),
            Body::Opaque { replay } => AssistantContent::Opaque(Opaque {
                item,
                replay: replay && closing == Closing::Complete,
            }),
            Body::ProviderItem { provider } => AssistantContent::ProviderItem(ProviderItem {
                payload: CarriedPayload::new(item, provider),
            }),
            Body::CustomCall {
                id,
                name,
                namespace,
                input,
                streamed,
            } => {
                let (id, item) = self.distinct_call_id(Some(id), item);
                if streamed == 0 {
                    emit(
                        items,
                        StreamEvent::Input {
                            part,
                            input: input.clone(),
                        },
                    );
                }
                let call = CustomToolCall::new(id, name, input).with_namespace(namespace);
                let content = with_item(AssistantContent::CustomToolCall(call), item);
                emit(items, StreamEvent::End { part, content, reasoning_authority: None });
                return Ok(());
            }
            Body::Call {
                id,
                name,
                namespace,
                arguments,
            } => {
                let Ok(name) = ToolName::new(name) else {
                    tracing::warn!(
                        index,
                        "the provider closed a tool call without a name; nothing can answer it"
                    );
                    return Ok(());
                };
                let function = arguments.function(name).with_namespace(namespace);
                let (id, item) = self.distinct_call_id(id, item);
                // A call that streamed no text (none arrived, or only a
                // placeholder `null`) states the arguments it ends with.
                if arguments.streamed == 0 {
                    let json = serde_json::Value::Object(function.arguments.clone()).to_string();
                    emit(items, StreamEvent::Arguments { part, json });
                }
                let call = ToolCall::new(id, function);
                let content = with_item(AssistantContent::ToolCall(call), item);
                emit(items, StreamEvent::End { part, content, reasoning_authority: None });
                return Ok(());
            }
        };
        if !started {
            emit(
                items,
                StreamEvent::Start {
                    part,
                    source_order,
                    kind: kind_of(&content),
                    name: None,
                    namespace: None,
                    tool_slot_state: None,
                },
            );
        }
        emit(items, StreamEvent::End { part, content, reasoning_authority });
        Ok(())
    }

    /// Edit the item of the block the wire index `index` last closed, while
    /// its end event waits to be taken or once the fold holds it.
    fn edit_ended(
        &mut self,
        items: &mut Items,
        index: usize,
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<(), ProviderError> {
        let position = *self.ended.get(&index).ok_or_else(|| not_open(index))?;
        let queued = items.iter_mut().find_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { part, content, .. })) if part.index() == position => {
                Some(content)
            }
            _ => None,
        });
        let content = match queued {
            Some(content) => Some(content),
            None => self.choice.get_mut(position).and_then(Option::as_mut),
        };
        if let Some(item) = content.and_then(item_of) {
            edit(item);
        }
        Ok(())
    }

    /// Add `citations` to the text item at `index`, or with `replace` put
    /// them in place of its own. An open item resolves them when it closes;
    /// a closed one, against its final text now, while its end event waits
    /// to be taken or once the fold holds it. Citations for any other item
    /// are dropped with a warning.
    fn cite_item(
        &mut self,
        items: &mut Items,
        index: usize,
        citations: Vec<WireCitation>,
        replace: bool,
    ) {
        let provider = self.origin.provider.as_str();
        if let Some(draft) = self.open.get_mut(&index) {
            if !matches!(draft.body, Body::Text(_)) {
                tracing::warn!(
                    provider,
                    index,
                    "dropped citations of an item that is not text"
                );
            } else if replace {
                draft.citations = citations;
            } else {
                draft.citations.extend(citations);
            }
            return;
        }
        let Some(position) = self.ended.get(&index).copied() else {
            tracing::warn!(
                provider,
                index,
                "dropped citations of an item the reply never opened"
            );
            return;
        };
        let queued = items.iter_mut().find_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { part, content, .. })) if part.index() == position => {
                Some(content)
            }
            _ => None,
        });
        let content = match queued {
            Some(content) => Some(content),
            None => self.choice.get_mut(position).and_then(Option::as_mut),
        };
        match content {
            Some(AssistantContent::Text(text)) => {
                let kept = if replace {
                    Vec::new()
                } else {
                    text.citations().to_vec()
                };
                citation::attach(text, kept, citations, provider, index);
            }
            _ => tracing::warn!(
                provider,
                index,
                "dropped citations of an item that is not text"
            ),
        }
    }

    pub(crate) fn run_item(
        &mut self,
        items: &mut Items,
        block: Block,
        fragment: &str,
    ) -> Result<usize, ProviderError> {
        let current = self.run.filter(|index| {
            self.open.get(index).is_some_and(|draft| {
                matches!(
                    (&draft.body, &block),
                    (Body::Text(_), Block::Text)
                        | (
                            Body::Reasoning {
                                redacted: false,
                                ..
                            },
                            Block::Reasoning { redacted: false }
                        )
                )
            })
        });
        let index = match current {
            Some(index) => index,
            None => {
                self.end_run(items)?;
                let index = self.fresh_index();
                self.open_item(items, index, block, serde_json::Value::Null)?;
                self.run = Some(index);
                index
            }
        };
        self.push_item(items, index, fragment)?;
        Ok(index)
    }

    pub(crate) fn end_run(&mut self, items: &mut Items) -> Result<(), ProviderError> {
        match self.run.take() {
            Some(index) => self.close_item(items, index, Closing::Complete),
            None => Ok(()),
        }
    }

    /// One whole block at the next position, its native kept as given.
    pub(crate) fn write_content(
        &mut self,
        items: &mut Items,
        content: AssistantContent,
    ) -> Result<(), ProviderError> {
        let content = match content {
            AssistantContent::ToolCall(mut call) => {
                let item = call
                    .native
                    .take()
                    .map_or(serde_json::Value::Null, |native| native.item);
                let (id, item) = self.distinct_call_id(Some(call.id), item);
                call.id = id;
                with_item(AssistantContent::ToolCall(call), item)
            }
            AssistantContent::CustomToolCall(mut call) => {
                let item = call
                    .native
                    .take()
                    .map_or(serde_json::Value::Null, |native| native.item);
                let (id, item) = self.distinct_call_id(Some(call.id), item);
                call.id = id;
                with_item(AssistantContent::CustomToolCall(call), item)
            }
            content => content,
        };
        let part = self.next();
        let (name, namespace) = match &content {
            AssistantContent::ToolCall(call) => (
                Some(call.function.name.clone()),
                call.function.namespace.clone(),
            ),
            AssistantContent::CustomToolCall(call) => {
                (Some(call.name.clone()), call.namespace.clone())
            }
            _ => (None, None),
        };
        emit(
            items,
            StreamEvent::Start {
                part,
                source_order: self.source_order,
                kind: kind_of(&content),
                name,
                namespace,
                tool_slot_state: None,
            },
        );
        match &content {
            AssistantContent::Text(text) if !text.text.is_empty() => emit(
                items,
                StreamEvent::Text {
                    part,
                    text: text.text.clone(),
                },
            ),
            AssistantContent::CarriedText(text) if !text.text().is_empty() => emit(
                items,
                StreamEvent::Text {
                    part,
                    text: text.text().to_owned(),
                },
            ),
            AssistantContent::CarriedReasoning(reasoning)
                if !reasoning.display_text().is_empty() =>
            {
                emit(
                    items,
                    StreamEvent::Reasoning {
                        part,
                        text: reasoning.display_text(),
                    },
                )
            }
            AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => emit(
                items,
                StreamEvent::Reasoning {
                    part,
                    text: reasoning.text.clone(),
                },
            ),
            AssistantContent::ToolCall(call) => emit(
                items,
                StreamEvent::Arguments {
                    part,
                    json: call.function.arguments_value().to_string(),
                },
            ),
            AssistantContent::CustomToolCall(call) => emit(
                items,
                StreamEvent::Input {
                    part,
                    input: call.input.clone(),
                },
            ),
            _ => {}
        }
        emit(items, StreamEvent::End { part, content, reasoning_authority: None });
        Ok(())
    }

    /// Close every item still open when the provider ends the reply, in the
    /// order they opened. The block a boundary-less wire was streaming is
    /// complete; any other item was never stated complete, so its block has
    /// no native. Calls keep what their arguments state.
    pub(crate) fn close_open(&mut self, items: &mut Items) {
        let mut open: Vec<(Part, usize)> = self
            .open
            .iter()
            .map(|(index, draft)| (draft.part, *index))
            .collect();
        open.sort();
        for (_, index) in open {
            let closing = if self.run == Some(index) {
                Closing::Complete
            } else {
                Closing::Incomplete
            };
            if self.open.get(&index).is_some_and(|draft| {
                matches!(draft.body, Body::Call { .. } | Body::CustomCall { .. })
            }) {
                self.unfinished_call = true;
            }
            if let Err(error) = self.close_item(items, index, closing) {
                items.push_back(Err(error));
            }
        }
    }

    /// Record that the item at `position` never completed.
    fn cut_at(&mut self, position: usize) {
        self.first_incomplete = Some(
            self.first_incomplete
                .map_or(position, |first| first.min(position)),
        );
    }

    /// The parts taken so far, in their position (in wire-index order on a
    /// wire that asked for it, after the lead block); a part that has not
    /// ended is not among them, and parts from the first incomplete one on
    /// keep no provider item.
    pub fn snapshot(&self) -> Vec<AssistantContent> {
        self.ordered(
            self.choice.iter().cloned().enumerate().collect(),
            self.first_incomplete,
        )
    }

    /// Who the reply is from.
    pub fn origin(&self) -> &Origin {
        &self.origin
    }

    /// What arrived so far as a response: every part that ended, the text
    /// the consumer already took of a text or reasoning part still open,
    /// and the provider's end when it arrived. A reply the provider did not
    /// end is a failed turn: `failure` is the error that ended it, and
    /// without one the caller stopped reading. Unless the provider ended the
    /// reply and the consumer took every part, no block keeps its provider
    /// item.
    pub(crate) fn partial(
        &self,
        end: Option<&Finish>,
        reply: &Reply,
        failure: Option<&ProviderError>,
    ) -> CompletionResponse {
        let mut response = self.response(end.cloned().unwrap_or_default(), reply.clone());
        if end.is_none() {
            match failure {
                Some(error) => {
                    response.error.get_or_insert_with(|| error.to_string());
                }
                None => {
                    response.aborted = Some(
                        "the caller stopped reading before the provider ended the reply".to_owned(),
                    );
                }
            }
        }
        // A turn the consumer has not wholly taken replays only its canonical
        // fields, as pi drops an unfinished turn's provider data: no item of
        // it can then need a partner that never arrived.
        let unfinished = end.is_none()
            || !self.open.is_empty()
            || self.choice.iter().enumerate().any(|(position, part)| {
                part.is_none() && !self.discarded.contains(&position)
            });
        let canonical_from = if unfinished {
            Some(0)
        } else {
            self.first_incomplete
        };
        let parts = self
            .choice
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let part = part.clone().or_else(|| self.open_text.get(&index).cloned());
                (index, part)
            })
            .collect();
        response.choice = self.ordered(parts, canonical_from);
        self.stamp_carried_choice(&mut response.choice);
        response
    }

    /// `parts` by position, in wire-index order on a wire that asked for
    /// it and after the lead block, each from position `canonical_from` on
    /// with no provider item.
    fn ordered(
        &self,
        parts: Vec<(usize, Option<AssistantContent>)>,
        canonical_from: Option<usize>,
    ) -> Vec<AssistantContent> {
        let mut parts: Vec<(usize, AssistantContent)> = parts
            .into_iter()
            .filter_map(|(position, part)| {
                let part = part?;
                Some(match canonical_from {
                    Some(first) if position >= first => (position, canonical(part)),
                    _ => (position, part),
                })
            })
            .collect();
        if self.source_orders.is_empty()
            && let Some(by_index) = &self.by_index
        {
            parts.sort_by_key(|(position, _)| {
                (
                    by_index.get(position).copied().unwrap_or(usize::MAX),
                    *position,
                )
            });
        }
        if let Some(lead) = self.lead {
            parts.sort_by_key(|(position, _)| *position != lead);
        }
        // Explicit two-coordinate positions replace the one-index policy.
        // They order only indexed slots; unindexed blocks stay in place.
        let mut indexed: Vec<_> = parts
            .iter()
            .filter_map(|(position, content)| {
                self.source_orders
                    .get(position)
                    .map(|order| (*order, *position, content.clone()))
            })
            .collect();
        indexed.sort_by_key(|(order, position, _)| (*order, *position));
        let mut sorted = indexed.into_iter();
        for (position, content) in &mut parts {
            if self.source_orders.contains_key(position)
                && let Some((_, _, ordered)) = sorted.next()
            {
                *content = ordered;
            }
        }
        parts.into_iter().map(|(_, part)| part).collect()
    }

    fn stamp_carried_choice(&self, choice: &mut [AssistantContent]) {
        let Some(issuer) = &self.carried_issuer else {
            return;
        };
        for content in choice {
            match content {
                AssistantContent::ProviderItem(item) if item.payload.provider.is_none() => {
                    item.payload.provider = Some(issuer.clone());
                }
                AssistantContent::CarriedReasoning(reasoning)
                    if reasoning.payload.provider.is_none() =>
                {
                    reasoning.payload.provider = Some(issuer.clone());
                    for part in &mut reasoning.content {
                        if let CarriedReasoningPart::OpaqueSummary(payload)
                        | CarriedReasoningPart::OpaqueContent(payload) = part
                            && payload.provider.is_none()
                        {
                            payload.provider = Some(issuer.clone());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn response(&self, end: Finish, reply: Reply) -> CompletionResponse {
        let Finish {
            message_id,
            usage,
            reason,
            response_id,
            model,
            error,
        } = end;
        use crate::provider_response::reported;
        let mut origin = self.origin.clone();
        origin.response_model = reported(model);
        // A request that named no model is from the model the provider
        // reports.
        if origin.model.is_empty()
            && let Some(model) = &origin.response_model
        {
            origin.model.clone_from(model);
        }
        origin.response_id = reported(response_id);
        let usage = priced(usage, &origin);
        let error = error.or_else(|| {
            if !self.wire {
                return None;
            }
            match &reason {
                None => Some("the provider ended the reply without a finish reason".to_owned()),
                Some(FinishReason::Length) => None,
                Some(_) if self.unfinished_call => Some(
                    "the provider ended the reply with a tool call it never finished".to_owned(),
                ),
                Some(_) => None,
            }
        });
        let mut choice = self.snapshot();
        self.stamp_carried_choice(&mut choice);
        let mut response = CompletionResponse::new(choice, usage, origin, reply.raw)
            .with_optional_finish_reason(reason)
            .accept_unknown_finish_reasons(self.accept_unknown_finish);
        response.message_id = message_id;
        response.error = error;
        response.provider_request_id = reported(reply.provider_request_id);
        response.provider_response_headers = reply.response_headers;
        response
    }
}

/// `usage` with its cost: the one the provider reported, else the counters
/// priced at the catalog's pricing for the requested model, else none.
fn priced(usage: Usage, origin: &Origin) -> Usage {
    if usage.cost.is_some() {
        return usage;
    }
    let cost = catalog_cost(origin, &usage);
    usage.cost(cost)
}

/// The providers whose replies the catalog does not price: ChatGPT and
/// Copilot bill a subscription, not the per-token API rates the catalog
/// lists, and an Ollama server is usually local, while the catalog's
/// `ollama` prices are Ollama Cloud's and the reply does not say which
/// served it.
const UNPRICED: [&str; 3] = [
    crate::providers::chatgpt::PROVIDER_NAME,
    crate::providers::copilot::PROVIDER_NAME,
    crate::providers::ollama::PROVIDER_NAME,
];

/// What `usage` costs at the built-in catalog's pricing for `origin`'s
/// model, or `None` when the catalog has no price for it or the provider
/// is one it does not price ([`UNPRICED`]).
fn catalog_cost(origin: &Origin, usage: &Usage) -> Option<crate::completion::Cost> {
    if UNPRICED.contains(&origin.provider.as_str()) {
        return None;
    }
    crate::catalog::lookup(&origin.provider, &origin.model)
        .or_else(|| crate::catalog::lookup_snapshot(&origin.provider, &origin.model))
        .and_then(|spec| spec.pricing.as_ref())
        .and_then(|pricing| pricing.cost(usage))
}

/// `block` with no provider item: an opaque item no longer replays.
pub(crate) fn canonical(block: AssistantContent) -> AssistantContent {
    match block {
        AssistantContent::Opaque(opaque) => AssistantContent::Opaque(Opaque {
            replay: false,
            ..opaque
        }),
        block => block.canonical(),
    }
}

/// The provider item `content` holds, if any.
fn item_of(content: &mut AssistantContent) -> Option<&mut serde_json::Value> {
    let native = match content {
        AssistantContent::Text(text) => text.native.as_mut(),
        AssistantContent::ToolCall(call) => call.native.as_mut(),
        AssistantContent::CustomToolCall(call) => call.native.as_mut(),
        AssistantContent::ProviderItem(item) => return Some(&mut item.payload.item),
        AssistantContent::CarriedText(text) => return Some(&mut text.payload.item),
        AssistantContent::CarriedReasoning(reasoning) => return Some(&mut reasoning.payload.item),
        AssistantContent::Reasoning(reasoning) => reasoning.native.as_mut(),
        AssistantContent::Image(image) => image.native.as_mut(),
        AssistantContent::Opaque(opaque) => return Some(&mut opaque.item),
    };
    native.map(|native| &mut native.item)
}

/// `block` holding `item` as its native, unless there is no item.
fn with_item(block: AssistantContent, item: serde_json::Value) -> AssistantContent {
    if item.is_null() {
        block
    } else {
        block.with_native(item)
    }
}

fn kind_of(content: &AssistantContent) -> PartKind {
    match content {
        AssistantContent::Text(_) | AssistantContent::CarriedText(_) => PartKind::Text,
        AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_) => {
            PartKind::Reasoning
        }
        AssistantContent::ToolCall(_) => PartKind::ToolCall,
        AssistantContent::CustomToolCall(_) => PartKind::CustomToolCall,
        AssistantContent::ProviderItem(_) => PartKind::ProviderItem,
        AssistantContent::Image(_) => PartKind::Image,
        AssistantContent::Opaque(_) => PartKind::Opaque,
    }
}

pub(crate) type Items = std::collections::VecDeque<Result<Item<StreamEvent>, ProviderError>>;

fn emit(items: &mut Items, event: StreamEvent) {
    items.push_back(Ok(Item::Event(event)));
}

/// Merge `delta` into `item`: each string field but `type` appends to the
/// item's string of that key, an array extends the item's array, and any
/// other value replaces the key. A delta kind rig has never seen still
/// lands in the item.
pub fn merge(item: &mut serde_json::Value, delta: &serde_json::Map<String, serde_json::Value>) {
    use serde_json::Value;
    if !item.is_object() {
        *item = Value::Object(serde_json::Map::new());
    }
    let Value::Object(item) = item else {
        return;
    };
    for (key, value) in delta {
        if key == "type" {
            continue;
        }
        match (item.get_mut(key), value) {
            (Some(Value::String(existing)), Value::String(fragment)) => {
                existing.push_str(fragment);
            }
            (Some(Value::Array(existing)), Value::Array(more)) => {
                existing.extend(more.iter().cloned());
            }
            _ => {
                item.insert(key.clone(), value.clone());
            }
        }
    }
}

impl Fold<Completion> for Turn {
    fn absorb(&mut self, event: &StreamEvent) -> Result<(), ProviderError> {
        match event {
            StreamEvent::Start {
                part, source_order, ..
            } => {
                if let Some(order) = source_order {
                    self.source_orders.entry(part.index()).or_insert(*order);
                }
                if self.choice.len() <= part.index() {
                    self.choice.resize(part.index() + 1, None);
                }
            }
            StreamEvent::End { part, content, .. } => {
                if self.choice.len() <= part.index() {
                    self.choice.resize(part.index() + 1, None);
                }
                if let Some(slot) = self.choice.get_mut(part.index()) {
                    *slot = Some(content.clone());
                }
                self.open_text.remove(&part.index());
            }
            StreamEvent::Discard { part, .. } => {
                self.discarded.insert(part.index());
                if let Some(slot) = self.choice.get_mut(part.index()) {
                    *slot = None;
                }
                self.open_text.remove(&part.index());
            }
            StreamEvent::Text { part, text } => {
                if let AssistantContent::Text(open) = self
                    .open_text
                    .entry(part.index())
                    .or_insert_with(|| AssistantContent::text(""))
                {
                    open.text.push_str(text);
                }
            }
            StreamEvent::Reasoning { part, text } => {
                if let AssistantContent::Reasoning(open) = self
                    .open_text
                    .entry(part.index())
                    .or_insert_with(|| AssistantContent::reasoning(""))
                {
                    open.text.push_str(text);
                }
            }
            StreamEvent::Arguments { .. } | StreamEvent::Input { .. } => {}
            StreamEvent::Resume { .. } => {}
        }
        Ok(())
    }

    fn finish(self, end: Finish, reply: Reply) -> Result<CompletionResponse, ProviderError> {
        if let Some(FinishReason::Other(reason)) = &end.reason {
            tracing::warn!(
                provider = %self.origin.provider,
                reason = %reason,
                accepted = self.accept_unknown_finish,
                "the provider ended the reply with an unknown finish reason"
            );
        }
        let response = self.response(end, reply);
        self.span
            .record_response(response.response_id(), response.model(), &response.usage);
        Ok(response)
    }
}

impl<'id> Out<'id, Completion> {
    /// Reactivate a real started text Part for preserved Responses, without
    /// changing its Draft, accumulated text or canonical fold position.
    /// Default wires never select this explicit hook.
    pub(crate) fn resume_text(&mut self, index: usize) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let draft = fold.open.get(&index).ok_or_else(|| not_open(index))?;
        if !draft.started || !matches!(&draft.body, Body::Text(_)) {
            return Err(ProviderError::Response(format!(
                "the reply resumed item {index}, which is not started open text"
            )));
        }
        emit(items, StreamEvent::Resume { part: draft.part });
        Ok(())
    }

    /// The effective request model already held by this completion's fold.
    /// Returns an owned clone under the reply mutex; reading it changes no
    /// Origin, provider identity, carried issuer or decoder policy.
    pub fn effective_model(&self) -> String {
        self.lock().fold.origin.model.clone()
    }

    /// Record the actual terminal issuer for the folded response's carried
    /// reasoning and whole provider items whose issuer is still unknown.
    /// This fills only missing provenance when the response is built; it
    /// changes neither already yielded events nor Origin, Native, or text.
    pub fn stamp_carried_issuer(&mut self, issuer: &str) {
        self.lock().fold.carried_issuer = Some(issuer.to_owned());
    }

    /// Write an action under one provider content position. These coordinates
    /// take precedence over `order_by_index`. Starts keep the
    /// coordinate even after this scope ends; nested scopes restore the outer
    /// coordinate. The wire index still identifies the open assembly.
    pub fn in_source_order<R>(
        &mut self,
        order: SourceOrder,
        action: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let previous = self.lock().fold.source_order.replace(order);
        let result = action(self);
        self.lock().fold.source_order = previous;
        result
    }

    /// What `usage` costs at the built-in catalog's pricing for the
    /// requested model, for a decoder whose provider bills other counters
    /// than the ones it reports as usage.
    pub(crate) fn catalog_cost(&self, usage: &Usage) -> Option<crate::completion::Cost> {
        catalog_cost(&self.lock().fold.origin, usage)
    }

    /// Open the provider item at wire `index` as `block`: it takes the next
    /// position in the reply. `item` is the item as its first event states
    /// it, the base its deltas merge into; `Null` when the block keeps no
    /// provider item. An index already open is an error.
    pub fn open(
        &mut self,
        index: usize,
        block: Block,
        item: serde_json::Value,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.open_item(items, index, block, item)
    }

    /// Append a fragment to the open item at `index`: text, reasoning,
    /// function argument JSON, or custom raw input, by the item's block.
    pub fn push(&mut self, index: usize, fragment: &str) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.push_item(items, index, fragment)
    }

    /// Set the exact namespace of the call at `index`. When the wire
    /// announces it at open, use `Block::Call::namespace`; for a fragmented
    /// call, qualify before its first named fragment to put it on the start.
    /// A terminal restatement remains on the end.
    /// An empty namespace is present, not equivalent to absence.
    pub fn qualify(
        &mut self,
        index: usize,
        namespace: Option<String>,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        match &mut shared.fold.draft(index)?.body {
            Body::Call {
                namespace: held, ..
            }
            | Body::CustomCall {
                namespace: held, ..
            } => {
                *held = namespace;
                Ok(())
            }
            _ => Err(ProviderError::Response(format!(
                "the reply qualified item {index}, which is not a tool call"
            ))),
        }
    }

    /// Edit the item at `index` in place: apply a delta with [`merge`], or
    /// replace the whole item a provider restates when it finishes.
    /// Until the reply ends, an index whose block already closed edits that
    /// block's item: the terminal backfill of a field the provider states
    /// only at its end.
    pub fn edit(
        &mut self,
        index: usize,
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        match fold.open.get_mut(&index) {
            Some(draft) => {
                edit(&mut draft.item);
                Ok(())
            }
            None => fold.edit_ended(items, index, edit),
        }
    }

    /// Close the item at `index` the provider never stated complete: its
    /// block becomes visible with no native, and an opaque item is kept but
    /// does not replay. Empty text and reasoning are dropped, and so is a
    /// call with no name. A call's arguments are read by
    /// [`ToolFunction::parse`], so malformed ones never fail the reply; a
    /// call that never got an id, or reuses one an earlier call took, gets
    /// one rig issues.
    pub fn close(&mut self, index: usize) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_item(items, index, Closing::Incomplete)
    }

    /// Close the item at `index` the provider stated complete: [`Self::close`],
    /// with the item as assembled becoming the block's native.
    pub fn finish(&mut self, index: usize) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_item(items, index, Closing::Complete)
    }

    /// Finalize an announced function call at a genuine Interactions terminal,
    /// using its retained name, typed ID and assembled arguments. No provider
    /// item or item-restatement authority is invented. Default finalizers are
    /// unchanged; the preserved Interactions decoder alone selects this hook.
    /// Malformed input returns the existing typed diagnostic, leaving the
    /// stream Part partial without an End, Discard or canonical call.
    pub(crate) fn finish_call_at_terminal_without_native(
        &mut self,
        index: usize,
    ) -> Result<(), ProviderError> {
        self.finish_call_at_provider_terminal_without_native(index)
    }

    /// Internal cross-crate bridge solely for rig-bedrock's
    /// MessageStop(ToolUse) decoder. This is not part of the supported public
    /// API and is semver-exempt. Finalizes the existing function-call Part
    /// without Native; supplies neither an item restatement nor a response
    /// terminal. Other callers must retain their default finalizer policy.
    #[doc(hidden)]
    pub fn finish_call_at_bedrock_tool_use_terminal(
        &mut self,
        index: usize,
    ) -> Result<(), ProviderError> {
        self.finish_call_at_provider_terminal_without_native(index)
    }

    fn finish_call_at_provider_terminal_without_native(
        &mut self,
        index: usize,
    ) -> Result<(), ProviderError> {
        let malformed = {
            let mut shared = self.lock();
            let Shared { fold, items, .. } = &mut *shared;
            let draft = fold.draft(index)?;
            let Body::Call { id: Some(id), name, arguments, .. } = &mut draft.body else {
                return Err(ProviderError::Response(format!(
                    "the reply finalized item {index} at a provider terminal without a named function call and retained ID"
                )));
            };
            ToolName::new(name.clone())
                .map_err(|error| ProviderError::Response(error.to_string()))?;
            let parsed = if !arguments.saw_fragments {
                arguments.announced.clone().map(Ok)
                    .unwrap_or_else(|| crate::json_utils::parse_tool_arguments(&arguments.text))
            } else {
                crate::json_utils::parse_tool_arguments(&arguments.text).and_then(|value| {
                    if arguments.overflowed {
                        Err(<serde_json::Error as serde::de::Error>::custom(
                            "tool-call input exceeded the accumulation bound",
                        ))
                    } else { Ok(value) }
                })
            };
            match parsed {
                Err(error) => {
                    let provider = id.provider().map(|call_id| crate::error::ProviderCallEvidence {
                        call_id: call_id.clone(),
                        item_id: None,
                    });
                    Some((arguments.text.clone(), error.to_string(), provider))
                }
                Ok(_) => {
                    if arguments.saw_fragments {
                        // F announce fallback applies only when no delta arrived.
                        // Empty/blank real deltas still supersede that fallback.
                        arguments.announced = None;
                    }
                    draft.item = serde_json::Value::Null;
                    // The genuine terminal closes the canonical call, but its
                    // absent provider item cannot become Native evidence.
                    fold.close_item(items, index, Closing::Complete)?;
                    None
                }
            }
        };
        match malformed {
            Some((raw, error, provider)) => Err(self.malformed_call_error(index, raw, error, provider)),
            None => Ok(()),
        }
    }

    /// Preserve a carried Responses call's valid non-object JSON arguments.
    /// Default argument parsing stays unchanged. The provider boundary invokes
    /// this only after validating its final assembled value; JSON whitespace
    /// and escapes use the fork's Value serialization rather than wire spelling.
    /// A later fragment or restatement clears this preservation.
    pub(crate) fn preserve_non_object_call_arguments(
        &mut self,
        index: usize,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let draft = shared.fold.draft(index)?;
        let Body::Call { arguments, .. } = &mut draft.body else {
            return Err(ProviderError::Response(format!(
                "the reply preserved non-object arguments for item {index}, which is not a function call"
            )));
        };
        let value: serde_json::Value = serde_json::from_str(&arguments.text)
            .map_err(|error| ProviderError::Response(error.to_string()))?;
        if value.is_object() {
            return Err(ProviderError::Response(format!(
                "the reply preserved item {index} as non-object arguments, but it is an object"
            )));
        }
        arguments.preserved_non_object = Some(value.to_string());
        Ok(())
    }

    /// Refuse a malformed authoritative function call with its actual
    /// writer-owned correlation ID. `raw` and `provider` are the exact
    /// argument text and identifier evidence the decoder received.
    /// No finalized call or fabricated end is emitted on this error path.
    pub fn malformed_call_error(
        &mut self,
        index: usize,
        raw: String,
        error: String,
        provider: Option<crate::error::ProviderCallEvidence>,
    ) -> ProviderError {
        let mut shared = self.lock();
        let fold = &mut shared.fold;
        let Some(draft) = fold.open.get(&index) else {
            return not_open(index);
        };
        let Body::Call { .. } = &draft.body else {
            return ProviderError::Response(format!(
                "the reply refused item {index} as a malformed function call, but it is not a function call"
            ));
        };
        let Some(draft) = fold.open.remove(&index) else {
            return not_open(index);
        };
        let Body::Call { id, name, .. } = draft.body else {
            unreachable!("the open item's function kind was checked under the same lock")
        };
        if fold.run == Some(index) {
            fold.run = None;
        }
        fold.ended.insert(index, draft.part.index());
        let (id, _) = fold.distinct_call_id(id, draft.item);
        ProviderError::Relayed(Box::new(
            crate::error::ErrorReport::new(
                crate::error::ErrorKind::Response,
                format!("tool call `{name}` arrived with malformed JSON input: {error}"),
            )
            .with_detail(crate::error::ErrorDetail::MalformedToolInput(
                crate::error::MalformedToolInput {
                    name,
                    id,
                    provider,
                    raw,
                    error,
                },
            )),
        ))
    }

    /// Discard a function call the protocol explicitly leaves unasserted.
    /// A started call closes with a typed discard, without a canonical call
    /// or invented end. A call never started emits no event.
    pub fn discard_call(
        &mut self,
        index: usize,
        reason: DiscardReason,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let draft = fold.open.get(&index).ok_or_else(|| not_open(index))?;
        if !matches!(&draft.body, Body::Call { .. }) {
            return Err(ProviderError::Response(format!(
                "the reply discarded item {index} as a function call, but it is not a function call"
            )));
        }
        let draft = fold.open.remove(&index).ok_or_else(|| not_open(index))?;
        if fold.run == Some(index) {
            fold.run = None;
        }
        fold.ended.insert(index, draft.part.index());
        fold.discarded.insert(draft.part.index());
        if draft.started {
            emit(
                items,
                StreamEvent::Discard {
                    part: draft.part,
                    reason,
                },
            );
        }
        Ok(())
    }

    /// Finish an open text part with its carried metadata, retaining the
    /// original part and previously emitted text. Refuses any replacement
    /// whose visible text differs from the accumulated text.
    pub fn finish_carried_text(
        &mut self,
        index: usize,
        mut text: CarriedText,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let draft = fold.open.get(&index).ok_or_else(|| not_open(index))?;
        match &draft.body {
            Body::Text(value) if value == text.content.text() => {}
            Body::Text(_) => {
                return Err(ProviderError::Response(format!(
                    "the reply replaced text item {index} with different carried text"
                )));
            }
            _ => {
                return Err(ProviderError::Response(format!(
                    "the reply finished item {index} as carried text, but it is not text"
                )));
            }
        }
        let draft = fold.open.remove(&index).ok_or_else(|| not_open(index))?;
        let kept = text.citations().to_vec();
        citation::attach_carried(
            &mut text,
            kept,
            draft.citations,
            &fold.origin.provider,
            index,
        );
        if fold.run == Some(index) {
            fold.run = None;
        }
        fold.ended.insert(index, draft.part.index());
        if !draft.started {
            emit(
                items,
                StreamEvent::Start {
                    part: draft.part,
                    source_order: draft.source_order,
                    kind: PartKind::Text,
                    name: None,
                    namespace: None,
                    tool_slot_state: None,
                },
            );
        }
        emit(
            items,
            StreamEvent::End {
                part: draft.part,
                content: AssistantContent::CarriedText(text),
                reasoning_authority: None,
            },
        );
        Ok(())
    }

    /// Finish an open reasoning part with its carried metadata, retaining
    /// the original part and previously emitted text. Refuses any replacement
    /// whose visible text differs from the accumulated reasoning.
    pub fn finish_carried_reasoning(
        &mut self,
        index: usize,
        reasoning: CarriedReasoning,
    ) -> Result<(), ProviderError> {
        self.finish_carried_reasoning_inner(index, reasoning, None)
    }

    /// Finish preserved reasoning with the provider boundary's explicit
    /// authority, without changing its canonical content or response end.
    pub fn finish_carried_reasoning_with_authority(
        &mut self,
        index: usize,
        reasoning: CarriedReasoning,
        authority: ReasoningAuthority,
    ) -> Result<(), ProviderError> {
        self.finish_carried_reasoning_inner(index, reasoning, Some(authority))
    }

    fn finish_carried_reasoning_inner(
        &mut self,
        index: usize,
        reasoning: CarriedReasoning,
        reasoning_authority: Option<ReasoningAuthority>,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let draft = fold.open.get(&index).ok_or_else(|| not_open(index))?;
        match &draft.body {
            Body::Reasoning { text, .. } if *text == reasoning.display_text() => {}
            Body::Reasoning { .. } => {
                return Err(ProviderError::Response(format!(
                    "the reply replaced reasoning item {index} with different carried text"
                )));
            }
            _ => {
                return Err(ProviderError::Response(format!(
                    "the reply finished item {index} as carried reasoning, but it is not reasoning"
                )));
            }
        }
        let draft = fold.open.remove(&index).ok_or_else(|| not_open(index))?;
        if fold.run == Some(index) {
            fold.run = None;
        }
        fold.ended.insert(index, draft.part.index());
        if !draft.started {
            emit(
                items,
                StreamEvent::Start {
                    part: draft.part,
                    source_order: draft.source_order,
                    kind: PartKind::Reasoning,
                    name: None,
                    namespace: None,
                    tool_slot_state: None,
                },
            );
        }
        emit(
            items,
            StreamEvent::End {
                part: draft.part,
                content: AssistantContent::CarriedReasoning(reasoning),
                reasoning_authority,
            },
        );
        Ok(())
    }

    /// End every item still open as stated complete, for a wire whose end
    /// of reply is the provider's statement that its items are done.
    pub fn finish_open(&mut self) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        let mut open: Vec<(Part, usize)> = fold
            .open
            .iter()
            .map(|(index, draft)| (draft.part, *index))
            .collect();
        open.sort();
        for (_, index) in open {
            fold.close_item(items, index, Closing::Complete)?;
        }
        Ok(())
    }

    /// Open and finish the item at `index` in one step: a whole block a
    /// provider states in one piece, `item` its provider item and `text`
    /// its text, reasoning, argument JSON or base64 image data.
    pub fn whole(
        &mut self,
        index: usize,
        block: Block,
        item: serde_json::Value,
        text: &str,
    ) -> Result<(), ProviderError> {
        self.open(index, block, item)?;
        self.push(index, text)?;
        self.finish(index)
    }

    /// Write a whole preserved reasoning block with explicit provider
    /// authority through the same open, fragment and finish path.
    pub fn whole_reasoning_with_authority(
        &mut self,
        index: usize,
        item: serde_json::Value,
        text: &str,
        authority: ReasoningAuthority,
    ) -> Result<(), ProviderError> {
        self.open(index, Block::Reasoning { redacted: false }, item)?;
        self.push(index, text)?;
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.close_item_with_reasoning_authority(items, index, Closing::Complete, Some(authority))
    }

    /// Order the response's blocks by wire index rather than by when they
    /// opened, for a wire whose indices are the provider's item order and
    /// whose end may state items it never streamed. Call it before the
    /// first item opens; blocks with no wire index go last.
    pub fn order_by_index(&mut self) {
        let mut shared = self.lock();
        if shared.fold.by_index.is_none() {
            shared.fold.by_index = Some(BTreeMap::new());
        }
    }

    /// Hold the open item at `index` first in the response whenever it
    /// opened, for a wire that states one block of the turn apart from its
    /// order. Its events keep their arrival order. A turn leads with the
    /// first item named.
    pub fn lead(&mut self, index: usize) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let position = shared.fold.draft(index)?.part.index();
        shared.fold.lead.get_or_insert(position);
        Ok(())
    }

    /// Replace the text, reasoning or argument JSON of the open item at
    /// `index` with `text`, the whole of it as the provider restates it at
    /// its end. The fragments already streamed stand and nothing more
    /// streams for them; the block ends holding `text`.
    pub fn restate(&mut self, index: usize, text: &str) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let draft = shared.fold.draft(index)?;
        match &mut draft.body {
            Body::Text(body) | Body::Reasoning { text: body, .. } => text.clone_into(body),
            Body::Call { arguments, .. } => {
                arguments.preserved_non_object = None;
                text.clone_into(&mut arguments.text);
                // A call not yet started streams the restated text whole.
                arguments.streamed = if draft.started { text.len() } else { 0 };
            }
            Body::CustomCall {
                input, streamed, ..
            } => {
                text.clone_into(input);
                *streamed = if draft.started { text.len() } else { 0 };
            }
            Body::Image(_) | Body::Opaque { .. } | Body::ProviderItem { .. } => {}
        }
        Ok(())
    }

    /// Write one fragment of a tool call the provider streams, opening the
    /// call at its first fragment. Its id and name may arrive in any
    /// fragment, and the first name it is given stays: the call starts
    /// streaming once it names a tool, carrying
    /// the argument text held until then, and its end states its id. Calls
    /// are told apart by `index` when the wire gives one. A new id under an
    /// index starts a new call once the held call's arguments are a complete
    /// object or when it names a tool: some providers send a fresh id, and
    /// no name, with every chunk of one call. Without an index (`None`, or
    /// the wire's `null`), a fragment with an unseen id opens a call, and one
    /// with no id continues the latest call while its arguments are
    /// incomplete.
    pub fn fragment(
        &mut self,
        index: Option<usize>,
        fragment: CallFragment<'_>,
    ) -> Result<(), ProviderError> {
        self.fragment_inner(index, fragment, false)
    }

    /// Opt a preserved Responses function-call slot into pending-name
    /// streaming. Allocates the same Draft/Part as normal fragments, emitting
    /// raw arguments immediately. A later genuine name binds that Part once.
    /// A named or already-started slot cannot enter the pending state.
    pub(crate) fn pending_call(
        &mut self,
        index: usize,
        fragment: CallFragment<'_>,
    ) -> Result<(), ProviderError> {
        self.fragment_inner(Some(index), fragment, true)
    }

    fn fragment_inner(
        &mut self,
        index: Option<usize>,
        fragment: CallFragment<'_>,
        pending: bool,
    ) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared {
            fold: turn, items, ..
        } = &mut *shared;
        if pending && (fragment.name.is_some_and(|name| !name.is_empty())
            || index.and_then(|index| turn.open.get(&index)).is_some_and(|draft| {
                draft.started || !matches!(&draft.body, Body::Call { name, .. } if name.is_empty())
            }))
        {
            return Err(ProviderError::Response(
                "the reply opened a pending tool slot after it was named or started".to_owned(),
            ));
        }
        let new_id = fragment.id.filter(|id| stated(id));
        let index = match index {
            Some(index) => {
                let names = fragment.name.is_some_and(|name| !name.is_empty());
                let held = turn.open.get(&index).and_then(|draft| match &draft.body {
                    Body::Call {
                        id: Some(id),
                        arguments,
                        ..
                    } if new_id.is_some_and(|new| id.wire() != new)
                        && draft.tool_slot_state != Some(ToolSlotState::PendingName) => Some(arguments.complete()),
                    _ => None,
                });
                if let Some(complete) = held.filter(|complete| *complete || names) {
                    // Another call took over the index: the one it held ends.
                    let moved = turn.fresh_index();
                    if let Some(draft) = turn.open.remove(&index) {
                        turn.open.insert(moved, draft);
                    }
                    let closing = if complete {
                        Closing::Complete
                    } else {
                        Closing::Incomplete
                    };
                    turn.close_item(items, moved, closing)?;
                }
                index
            }
            None => {
                let owner = new_id.and_then(|new| {
                    turn.open
                        .iter()
                        .find_map(|(index, draft)| match &draft.body {
                            Body::Call { id: Some(id), .. } if id.wire() == new => Some(*index),
                            _ => None,
                        })
                });
                let continues = |last: &usize| {
                    turn.open.get(last).is_some_and(|draft| {
                        matches!(&draft.body, Body::Call { arguments, .. } if !arguments.complete())
                    })
                };
                match (owner, new_id, turn.last_call) {
                    (Some(index), _, _) => index,
                    (None, None, Some(last)) if continues(&last) => last,
                    _ => turn.fresh_index(),
                }
            }
        };
        turn.last_call = Some(index);
        if !turn.open.contains_key(&index) {
            let body = Body::Call {
                id: None,
                name: String::new(),
                namespace: None,
                arguments: Arguments::default(),
            };
            turn.insert(index, body, serde_json::Value::Null);
        }
        if pending {
            turn.draft(index)?.tool_slot_state = Some(ToolSlotState::PendingName);
        }
        let preserved = turn.draft(index)?.tool_slot_state.is_some();
        let Body::Call {
            id,
            name,
            arguments,
            ..
        } = &mut turn.draft(index)?.body
        else {
            return Err(ProviderError::Response(format!(
                "the reply sent a call fragment for item {index}, which is not a call"
            )));
        };
        if let Some(call_id) = fragment.id.filter(|id| stated(id)) {
            *id = Some(CallId::from_wire(call_id));
        }
        // A call keeps the first name it is given, so the tool its start
        // names is the tool its end names.
        if name.is_empty()
            && let Some(fragment) = fragment.name.filter(|name| !name.is_empty())
        {
            fragment.clone_into(name);
        }
        if let Some(fragment) = fragment.arguments {
            if preserved && arguments.text.trim() == "null" && !fragment.trim().is_empty() {
                // Explicit preserved slots already delivered the placeholder;
                // replacement carries only the real new fragment, once.
                arguments.streamed = 0;
            }
            arguments.push(fragment, name);
        }
        stream_call(items, turn.draft(index)?);
        Ok(())
    }

    /// Arguments the provider announced for the call at `index`, used only
    /// if no fragment arrives.
    pub fn announce(
        &mut self,
        index: usize,
        announced: serde_json::Value,
    ) -> Result<(), ProviderError> {
        if let Body::Call { arguments, .. } = &mut self.lock().fold.draft(index)?.body {
            arguments.announced = Some(announced);
        }
        Ok(())
    }

    /// An index no provider item uses, for a wire that indexes nothing.
    pub fn fresh_index(&mut self) -> usize {
        self.lock().fold.fresh_index()
    }

    /// A fragment on a wire that marks no item boundaries: it continues the
    /// block the last fragment went to while `block` is the same kind, and
    /// otherwise closes that block and opens a new one. Returns the block's
    /// index, for merging its provider item.
    pub fn run(&mut self, block: Block, fragment: &str) -> Result<usize, ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.run_item(items, block, fragment)
    }

    /// Close the block [`Self::run`] is extending, if any: output of
    /// another kind interleaved it.
    pub fn end_run(&mut self) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.end_run(items)
    }

    /// Add `citation` to the text item at wire `index`, open or ended. Its
    /// span is resolved to bytes of the text when the item closes, or at
    /// once for an item already closed. A span that does not resolve, or
    /// covers other text than its `quoted`, drops the citation with a
    /// warning, as do citations of an item that is not text: a citation
    /// never fails a reply.
    pub fn cite(&mut self, index: usize, citation: WireCitation) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.cite_item(items, index, vec![citation], false);
    }

    /// Put `citations` in place of the text item's own, for a provider that
    /// restates them; [`Self::cite`] otherwise.
    pub fn set_citations(&mut self, index: usize, citations: Vec<WireCitation>) {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.cite_item(items, index, citations, true);
    }

    /// Emit one complete carried text part at the current source coordinate.
    /// Its raw payload and per-item issuer remain independent of Origin.
    pub fn carried_text(&mut self, text: CarriedText) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.write_content(items, AssistantContent::CarriedText(text))
    }

    /// Emit one complete carried reasoning item at the current source coordinate.
    /// Its ordered parts and per-item issuer remain independent of Origin.
    pub fn carried_reasoning(&mut self, reasoning: CarriedReasoning) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.write_content(items, AssistantContent::CarriedReasoning(reasoning))
    }

    /// A whole block of an already assembled response, at the next
    /// position, its native kept as given.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn content(&mut self, content: AssistantContent) -> Result<(), ProviderError> {
        let mut shared = self.lock();
        let Shared { fold, items, .. } = &mut *shared;
        fold.write_content(items, content)
    }
}

/// The events a stream of `response` would have carried: each part whole,
/// in order.
pub(crate) fn events_of(
    response: &CompletionResponse,
) -> Result<Vec<Item<StreamEvent>>, ProviderError> {
    let mut turn = Turn::new(response.origin.clone());
    let mut items = Items::new();
    for content in &response.choice {
        turn.write_content(&mut items, content.clone())?;
    }
    items.into_iter().collect()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod carried_tests;

#[cfg(test)]
mod nested_tests;

#[cfg(test)]
mod writer_boundary_tests;

#[cfg(test)]
mod interactions_terminal_tests;

#[cfg(test)]
mod bedrock_terminal_tests;

#[cfg(test)]
mod effective_model_tests;
