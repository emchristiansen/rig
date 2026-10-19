//! Request-local wire IDs paired by chronological call occurrence. Provider IDs
//! remain exact; each local occurrence gets its own deterministic `tool-<n>`
//! alias through the target's normalization rules. Aliases never take genuine
//! or hosted handles, including handles first appearing later in the history.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::message::{AssistantContent, CallId, CarriedReasoningPart, Message, UserContent};

/// A required-ID wire cannot pair a transcript or locate its planned occurrence.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WireIdError {
    /// Another occurrence of this identity is still unanswered.
    #[error("ambiguous tool call identity at message {message}, content {content}")]
    Ambiguous { message: usize, content: usize },
    /// A result repeats an already consumed reference.
    #[error("duplicate tool result at message {message}, content {content}")]
    DuplicateResult { message: usize, content: usize },
    /// A local result has no preceding call or provider handle.
    #[error("local-only orphan tool result at message {message}, content {content}")]
    OrphanResult { message: usize, content: usize },
    /// The coordinate is not a call or result in the planned history.
    #[error("missing planned call at message {message}, content {content}")]
    MissingOccurrence { message: usize, content: usize },
}

/// The wire spelling at each original message/content position of one history.
#[derive(Debug, Default)]
pub struct WireIds {
    by_position: BTreeMap<(usize, usize), String>,
}

struct Occurrence<'a> {
    id: &'a CallId,
    /// A call, or an Interactions result with no call in the supplied history.
    first: (usize, usize),
    result: Option<(usize, usize)>,
}

/// Whether a local result may begin an occurrence without a visible call.
#[derive(Clone, Copy)]
enum ResultPolicy {
    RequireLocalCall,
    InteractionsStandalone,
}

impl WireIds {
    /// Plan the exact supplied history, before encoders filter its content.
    /// Completed identities may recur; unanswered reuse is refused. Provider
    /// orphan results retain their handles for server-held partial history.
    /// Local orphan results cannot be paired and are refused.
    pub fn for_target(
        history: &[Message],
        target: &dyn crate::completion::ReplayTarget,
        model: &str,
    ) -> Result<Self, WireIdError> {
        Self::plan(history, target, model, ResultPolicy::RequireLocalCall)
    }

    /// Plan Interactions history, where a standalone local result is its own
    /// alias-bearing occurrence. It remains local, and a repeated result is
    /// refused unless a new call occurrence has intervened. All genuine/hosted
    /// handles are reserved before the shared chronological alias allocation.
    pub(crate) fn for_interactions(
        history: &[Message],
        target: &dyn crate::completion::ReplayTarget,
        model: &str,
    ) -> Result<Self, WireIdError> {
        Self::plan(history, target, model, ResultPolicy::InteractionsStandalone)
    }

    fn plan(
        history: &[Message],
        target: &dyn crate::completion::ReplayTarget,
        model: &str,
        policy: ResultPolicy,
    ) -> Result<Self, WireIdError> {
        let mut occurrences: Vec<Occurrence<'_>> = Vec::new();
        let mut outstanding: HashMap<&CallId, usize> = HashMap::new();
        let mut answered: HashSet<&CallId> = HashSet::new();
        let mut used: HashSet<String> = HashSet::new();
        let mut by_position = BTreeMap::new();
        for (message, entry) in history.iter().enumerate() {
            match entry {
                Message::Assistant(turn) => {
                    for (content, block) in turn.content.iter().enumerate() {
                        let id = match block {
                            AssistantContent::ToolCall(call) => Some(&call.id),
                            AssistantContent::CustomToolCall(call) => Some(&call.id),
                            AssistantContent::Opaque(opaque) => {
                                used.extend(target.hosted_pair(&opaque.item).map(|(_, id)| id));
                                None
                            }
                            AssistantContent::ProviderItem(item) => {
                                used.extend(
                                    target.hosted_pair(&item.payload.item).map(|(_, id)| id),
                                );
                                None
                            }
                            AssistantContent::CarriedText(text) => {
                                used.extend(
                                    target.hosted_pair(&text.payload.item).map(|(_, id)| id),
                                );
                                None
                            }
                            AssistantContent::CarriedReasoning(reasoning) => {
                                used.extend(
                                    target
                                        .hosted_pair(&reasoning.payload.item)
                                        .map(|(_, id)| id),
                                );
                                for part in &reasoning.content {
                                    if let CarriedReasoningPart::OpaqueSummary(payload)
                                    | CarriedReasoningPart::OpaqueContent(payload) = part
                                    {
                                        used.extend(
                                            target.hosted_pair(&payload.item).map(|(_, id)| id),
                                        );
                                    }
                                }
                                None
                            }
                            _ => None,
                        };
                        if let Some(id) = id {
                            if outstanding.contains_key(id) {
                                return Err(WireIdError::Ambiguous { message, content });
                            }
                            used.extend(id.provider().map(|id| id.as_str().to_owned()));
                            outstanding.insert(id, occurrences.len());
                            occurrences.push(Occurrence {
                                id,
                                first: (message, content),
                                result: None,
                            });
                        }
                    }
                }
                Message::User { content } => {
                    for (content, part) in content.iter().enumerate() {
                        let UserContent::ToolResult(result) = part else {
                            continue;
                        };
                        let id = &result.call;
                        used.extend(id.provider().map(|id| id.as_str().to_owned()));
                        if let Some(index) = outstanding.remove(id) {
                            occurrences
                                .get_mut(index)
                                .ok_or(WireIdError::MissingOccurrence { message, content })?
                                .result = Some((message, content));
                        } else if answered.contains(id) {
                            return Err(WireIdError::DuplicateResult { message, content });
                        } else if let Some(provider) = id.provider() {
                            by_position.insert((message, content), provider.as_str().to_owned());
                        } else {
                            match policy {
                                ResultPolicy::RequireLocalCall => {
                                    return Err(WireIdError::OrphanResult { message, content });
                                }
                                ResultPolicy::InteractionsStandalone => {
                                    occurrences.push(Occurrence {
                                        id,
                                        first: (message, content),
                                        result: None,
                                    });
                                }
                            }
                        }
                        answered.insert(id);
                    }
                }
                Message::System { .. } => {}
            }
        }
        let mut next = 0usize;
        for occurrence in occurrences {
            let spelled = match occurrence.id {
                CallId::Provider(provider) => provider.as_str().to_owned(),
                CallId::Local(_) => loop {
                    let alias = target.normalize_tool_call_id(&format!("tool-{next}"), model, None);
                    next += 1;
                    if used.insert(alias.clone()) {
                        break alias;
                    }
                },
            };
            if let Some(result) = occurrence.result {
                by_position.insert(result, spelled.clone());
            }
            by_position.insert(occurrence.first, spelled);
        }
        Ok(Self { by_position })
    }

    /// The spelling at the exact supplied history/content coordinate.
    pub fn of(&self, message: usize, content: usize) -> Option<&str> {
        self.by_position
            .get(&(message, content))
            .map(String::as_str)
    }

    /// Require a planned occurrence; there is no unplanned identity fallback.
    pub fn spell(&self, message: usize, content: usize) -> Result<String, WireIdError> {
        self.of(message, content)
            .map(str::to_owned)
            .ok_or(WireIdError::MissingOccurrence { message, content })
    }
}

/// `id` in the alphabet most wires take for call ids, `[A-Za-z0-9_-]`,
/// cut to `max` characters: every other character becomes `_`.
pub fn legal_call_id(id: &str, max: usize) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(max)
        .collect()
}

/// pi's `shortHash`: two 32-bit multiplicative hashes of the UTF-16 code
/// units, written in base 36.
pub(crate) fn short_hash(text: &str) -> String {
    fn base36(mut value: u32) -> String {
        let mut digits = Vec::new();
        loop {
            digits.push(char::from_digit(value % 36, 36).unwrap_or('0'));
            value /= 36;
            if value == 0 {
                break;
            }
        }
        digits.iter().rev().collect()
    }
    let (mut h1, mut h2) = (0xdead_beef_u32, 0x41c6_ce57_u32);
    for unit in text.encode_utf16().map(u32::from) {
        h1 = (h1 ^ unit).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ unit).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", base36(h2), base36(h1))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) mod tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod interactions_tests;
