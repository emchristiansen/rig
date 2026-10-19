//! Effect log headers, checkpoint envelopes, and stable JSON hashing.
//!
//! ```
//! let hash = rig_cassette::effect_log::stable_hash(&"run")?;
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::collections::BTreeMap;

use rig_core::completion::CompletionResponse;
use rig_core::effect::{
    EffectId, EffectKind, EffectRecord, EffectRow, HandlerDescriptor, MemoryOp, MemoryOutcome,
    Outcome,
};
use rig_core::error::{ErrorDetail, ErrorKind, ErrorReport};
use rig_core::json_utils::to_canonical_string;
use rig_core::message::{AssistantContent, CallId, LocalCallId, Message, UserContent};
use rig_core::serve::ServingPolicy;
use rig_core::streaming::{Item, StreamEvent};
use serde::{Deserialize, Serialize};

/// The checkpoint envelope format this crate writes and reads. This versions
/// [`Checkpoint`], not [`LogHeader`]; logs have no global format number.
pub const CHECKPOINT_FORMAT: u32 = 6;

/// Recorded run identity, handler declarations, and optional delivery metadata
/// used to validate replay compatibility. Deserialization rejects unknown fields;
/// headers have no global format number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogHeader {
    /// Consumer-visible deliveries, in observation order, when the runtime
    /// records schedule boundaries. `None` supplies no delivery guarantee.
    /// A batch groups transitions observed in one pass; it is not a clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deliveries: Option<Vec<rig_core::effect::Delivery>>,
    /// Reasons this recording cannot establish policy-visible delivery.
    /// Exchange replay remains possible; exact policy mode must refuse it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delivery_limitations: Vec<String>,
    /// Error items omitted from `EffectRecord::events`, at their original
    /// positions among all stream items. Empty for streams without errors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stream_errors: BTreeMap<EffectId, Vec<RecordedStreamError>>,
    /// Complete response items omitted from `EffectRecord::events`, at their
    /// original positions among events, errors and responses.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stream_done: BTreeMap<EffectId, Vec<RecordedStreamDone>>,
    /// A hash of the run spec the run was recorded under, when an agent
    /// recorded it (`None` for a bare-bus record). An agent that replays
    /// compares it with its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_spec: Option<u64>,
    /// The handlers registered on the bus when recording began, stamped
    /// with their keys.
    pub handlers: Vec<HandlerDescriptor>,
    /// The effect signature: which keys the run performed effects on, and
    /// of which family, derived from the trace.
    pub signature: EffectRow,
    /// Ordered hook type names, with nested stacks flattened.
    /// Replay requires the same stack because hooks execute again.
    pub hooks: Vec<String>,
    /// The program's required effect row at record time: every key it could
    /// dispatch to (its model, its tools, its memory, its retrieval
    /// indexes) with the family it needs. A replay checks this row against
    /// what the log's handlers serve, not only against what happened to be
    /// dispatched.
    pub required: EffectRow,
    /// The recorded serving policy, when known. Per-key order remains dispatch
    /// order under either policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bus: Option<ServingPolicy>,
    /// Required effects and policy hashes per `EffectRecord::scope` for logs
    /// shared by several programs. Single-agent logs use `run_spec` and
    /// `required` instead.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub programs: BTreeMap<String, ProgramIdentity>,
}

/// One program's identity in a shared log: its required effect row and
/// the stable hash of its policy (what `run_spec` is for a single agent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramIdentity {
    /// Every key the program could dispatch to, with the family it needs.
    pub required: EffectRow,
    /// [`stable_hash`] of the program's policy.
    pub policy: u64,
}

/// An error's place in a kept stream, including errors after a terminal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedStreamError {
    /// Zero-based position among events, errors and complete responses.
    pub item: usize,
    /// The original error item; it need not be the stream's folded outcome.
    pub error: ErrorReport,
}

/// A complete response's place in a kept stream, including later responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedStreamDone {
    /// Zero-based position among events, errors and complete responses.
    pub item: usize,
    /// The complete response item, independently of the folded outcome.
    pub response: CompletionResponse,
}

impl Default for LogHeader {
    fn default() -> Self {
        Self {
            deliveries: None,
            delivery_limitations: Vec::new(),
            stream_errors: BTreeMap::new(),
            stream_done: BTreeMap::new(),
            run_spec: None,
            handlers: Vec::new(),
            signature: EffectRow::new(),
            hooks: Vec::new(),
            required: EffectRow::new(),
            bus: None,
            programs: BTreeMap::new(),
        }
    }
}

/// A recorded run with metadata and exchanges in dispatch order.
/// Dereferences to the record slice and supports owned or borrowed iteration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectLog {
    /// What the log says about the run.
    pub header: LogHeader,
    /// The exchanges, in dispatch order.
    pub records: Vec<EffectRecord>,
}

impl EffectLog {
    /// A log over `records` with a default header (no spec, no handlers,
    /// the signature read off the records).
    pub fn from_records(records: Vec<EffectRecord>) -> Self {
        let mut header = LogHeader::default();
        for record in &records {
            header
                .signature
                .insert_if_absent(record.key.clone(), record.kind.family());
        }
        Self { header, records }
    }

    /// Return records from `at` onward with cloned metadata restricted to those
    /// records. An out-of-range position yields an empty tail.
    pub fn tail(&self, at: usize) -> Self {
        let mut tail = Self {
            header: self.header.clone(),
            records: self.records.get(at..).unwrap_or_default().to_vec(),
        };
        tail.retain_recorded_deliveries();
        tail
    }

    /// Drop delivery entries outside this log's records, preserving order and
    /// batch identities. Used for snapshots and tails of a shared recorder.
    pub(crate) fn retain_recorded_deliveries(&mut self) {
        let ids: std::collections::BTreeSet<_> =
            self.records.iter().map(|record| record.id).collect();
        self.header.stream_errors.retain(|id, _| ids.contains(id));
        self.header.stream_done.retain(|id, _| ids.contains(id));
        if let Some(deliveries) = &mut self.header.deliveries {
            deliveries.retain(|delivery| ids.contains(&delivery.id));
        }
    }

    /// Return a checkpoint containing `state`, position `at`, and the next
    /// record ID, together with the remaining log. Positions at or beyond the
    /// end produce an empty tail.
    pub fn checkpoint<S>(&self, at: usize, state: S) -> (Checkpoint<S>, Self) {
        let checkpoint = Checkpoint {
            format: CHECKPOINT_FORMAT,
            at,
            next: self.records.get(at).map(|record| record.id),
            state,
        };
        (checkpoint, self.tail(at))
    }

    /// Validate and return `tail`, or an error if the checkpoint format or next
    /// record ID does not match. An ending checkpoint requires an empty tail.
    pub fn from_checkpoint<S>(checkpoint: &Checkpoint<S>, tail: Self) -> Result<Self, ErrorReport> {
        if checkpoint.format != CHECKPOINT_FORMAT {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                format!(
                    "resume refused: the checkpoint is format {}, this rig reads format {}",
                    checkpoint.format, CHECKPOINT_FORMAT
                ),
            ));
        }
        let first = tail.records.first().map(|record| record.id);
        if first != checkpoint.next {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                match (checkpoint.next, first) {
                    (Some(next), Some(first)) => format!(
                        "resume refused: the checkpoint at {} expects record {next} next, the tail begins at {first}",
                        checkpoint.at
                    ),
                    (Some(next), None) => format!(
                        "resume refused: the checkpoint at {} expects record {next} next, the tail is empty",
                        checkpoint.at
                    ),
                    (None, Some(first)) => format!(
                        "resume refused: the checkpoint at {} ends the log, the tail begins at {first}",
                        checkpoint.at
                    ),
                    (None, None) => unreachable_refusal(),
                },
            ));
        }
        Ok(tail)
    }
}

/// Return the fallback diagnostic for a mismatched checkpoint tail.
fn unreachable_refusal() -> String {
    "resume refused: the tail does not follow the checkpoint".to_owned()
}

/// A suspended log position, expected next record ID, and driver-owned state.
/// Resume validation requires a tail whose first ID matches `next`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint<S> {
    /// The checkpoint envelope format ([`CHECKPOINT_FORMAT`]).
    pub format: u32,
    /// The position in the log: `at` records were performed before it.
    pub at: usize,
    /// The id of the record the tail begins with; `None` when the
    /// checkpoint ends the log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<EffectId>,
    /// What the driver persists, in its own type.
    pub state: S,
}

impl std::ops::Deref for EffectLog {
    type Target = [EffectRecord];

    fn deref(&self) -> &[EffectRecord] {
        &self.records
    }
}

impl From<Vec<EffectRecord>> for EffectLog {
    fn from(records: Vec<EffectRecord>) -> Self {
        Self::from_records(records)
    }
}

impl FromIterator<EffectRecord> for EffectLog {
    fn from_iter<I: IntoIterator<Item = EffectRecord>>(records: I) -> Self {
        Self::from_records(records.into_iter().collect())
    }
}

impl IntoIterator for EffectLog {
    type Item = EffectRecord;
    type IntoIter = std::vec::IntoIter<EffectRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.into_iter()
    }
}

impl<'a> IntoIterator for &'a EffectLog {
    type Item = &'a EffectRecord;
    type IntoIter = std::slice::Iter<'a, EffectRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.iter()
    }
}

/// Return the 64-bit FNV-1a hash of `value` serialized as JSON with recursively
/// sorted object keys, or a serialization error.
/// Sorting makes identity independent of JSON map insertion order.
pub fn stable_hash<T: Serialize>(value: &T) -> Result<u64, serde_json::Error> {
    let json = to_canonical_string(&serde_json::to_value(value)?);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in json.into_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(hash)
}

/// Rename rig-issued call identities in a serialized `EffectKind`, `EffectLog`,
/// or `Vec<EffectRecord>` to fixed UUIDs in order of first typed appearance.
/// Only local call identity fields change; provider identities, arbitrary data
/// and values outside these typed roots remain unchanged.
///
/// ```
/// let data = serde_json::json!({"local": "7ecaf57b-5bbb-4b9e-a440-63738a5939ec"});
/// assert_eq!(
///     rig_cassette::effect_log::canonical_local_ids(data.clone()),
///     data,
/// );
/// ```
pub fn canonical_local_ids(mut log: serde_json::Value) -> serde_json::Value {
    let mut locations = LocalCallIdLocations::default();
    if let Ok(kind) = serde_json::from_value::<EffectKind>(log.clone()) {
        locations.kind(&kind, "");
    } else if let Ok(typed) = serde_json::from_value::<EffectLog>(log.clone()) {
        for (effect, errors) in &typed.header.stream_errors {
            for (index, error) in errors.iter().enumerate() {
                locations.error(
                    &error.error,
                    &format!("/header/stream_errors/{}/{index}/error", effect.as_u64()),
                );
            }
        }
        for (effect, responses) in &typed.header.stream_done {
            for (index, done) in responses.iter().enumerate() {
                locations.choice(
                    &done.response.choice,
                    &format!(
                        "/header/stream_done/{}/{index}/response/choice",
                        effect.as_u64()
                    ),
                );
            }
        }
        locations.records(&typed.records, "/records");
    } else if let Ok(records) = serde_json::from_value::<Vec<EffectRecord>>(log.clone()) {
        locations.records(&records, "");
    }

    let mut ids = Vec::new();
    for (path, id) in locations.0 {
        if let Some(value) = log.pointer_mut(&path) {
            let index = ids.iter().position(|seen| *seen == id).unwrap_or_else(|| {
                ids.push(id);
                ids.len() - 1
            });
            *value = serde_json::Value::String(format!("00000000-0000-4000-8000-{index:012}"));
        }
    }
    log
}

#[derive(Default)]
struct LocalCallIdLocations(Vec<(String, LocalCallId)>);

impl LocalCallIdLocations {
    fn id(&mut self, id: &CallId, path: &str) {
        if let CallId::Local(id) = id {
            self.0.push((format!("{path}/local"), *id));
        }
    }

    fn content(&mut self, content: &AssistantContent, path: &str) {
        match content {
            AssistantContent::ToolCall(call) => self.id(&call.id, &format!("{path}/id")),
            AssistantContent::CustomToolCall(call) => self.id(&call.id, &format!("{path}/id")),
            _ => {}
        }
    }

    fn choice(&mut self, choice: &[AssistantContent], path: &str) {
        for (index, content) in choice.iter().enumerate() {
            self.content(content, &format!("{path}/{index}"));
        }
    }

    fn messages(&mut self, messages: &[Message], path: &str) {
        for (index, message) in messages.iter().enumerate() {
            let path = format!("{path}/{index}/content");
            match message {
                Message::Assistant(turn) => {
                    for (index, content) in turn.content.iter().enumerate() {
                        self.content(content, &format!("{path}/{index}"));
                    }
                }
                Message::User { content } => {
                    for (index, content) in content.iter().enumerate() {
                        if let UserContent::ToolResult(result) = content {
                            self.id(&result.call, &format!("{path}/{index}/call"));
                        }
                    }
                }
                Message::System { .. } => {}
            }
        }
    }

    fn kind(&mut self, kind: &EffectKind, path: &str) {
        match kind {
            EffectKind::Completion { request, .. } => {
                self.messages(
                    &request.chat_history,
                    &format!("{path}/request/chat_history"),
                );
            }
            EffectKind::Memory {
                op: MemoryOp::Append { messages, .. },
            } => self.messages(messages, &format!("{path}/op/messages")),
            _ => {}
        }
    }

    fn error(&mut self, error: &ErrorReport, path: &str) {
        if let Some(ErrorDetail::MalformedToolInput(detail)) = &error.detail {
            self.id(&detail.id, &format!("{path}/detail/id"));
        }
    }

    fn records(&mut self, records: &[EffectRecord], path: &str) {
        for (index, record) in records.iter().enumerate() {
            let path = format!("{path}/{index}");
            self.kind(&record.kind, &format!("{path}/kind"));
            match &record.outcome {
                Ok(Outcome::Completion(response)) => {
                    self.choice(&response.choice, &format!("{path}/outcome/Ok/choice"));
                }
                Ok(Outcome::Memory(MemoryOutcome::Loaded { messages })) => {
                    self.messages(messages, &format!("{path}/outcome/Ok/messages"));
                }
                Err(error) => self.error(error, &format!("{path}/outcome/Err")),
                _ => {}
            }
            if let Some(events) = &record.events {
                for (index, item) in events.items().iter().enumerate() {
                    if let Item::Event(StreamEvent::End { content, .. }) = item {
                        self.content(content, &format!("{path}/events/{index}/value/content"));
                    }
                }
            }
        }
    }
}

/// A tool call's JSON arguments with object keys sorted, or the text as is
/// when it is not JSON. Two argument strings that differ only in key order
/// describe the same call; which order a build emits depends on whether
/// `serde_json`'s `preserve_order` feature is unified into it.
pub(crate) fn canonical_tool_args(args: &str) -> String {
    serde_json::from_str::<serde_json::Value>(args)
        .map(|value| to_canonical_string(&value))
        .unwrap_or_else(|_| args.to_owned())
}

// A log serializes and crosses threads on every target.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<LogHeader>();
    assert_wire::<EffectLog>();
};

#[cfg(test)]
mod stable_hash_tests;

#[cfg(test)]
mod canonical_args_tests;
