//! A live view of one completion's stream. Each tool call is addressed by
//! its part: the start opens the tool, a late name binds the same pending
//! view, every argument fragment prints with
//! the best-effort object the text so far states, and the end brings the
//! call's id and final arguments. The summary checks that each call's
//! joined fragments are the arguments its end states.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rig::message::AssistantContent;
use rig::streaming::{PartKind, StreamEvent, ToolSlotState, parse_partial_arguments};
use serde_json::Value;

/// One tool call as it streamed.
struct CallView {
    tool: Option<String>,
    fragments: usize,
    arguments: String,
    /// When each fragment arrived.
    arrivals: Vec<Duration>,
    ended: Option<(String, Value)>,
}

impl CallView {
    /// How evenly the fragments arrived: the 50 ms windows they landed in,
    /// and the longest wait between two of them. A provider that holds an
    /// argument back sends it in a few windows after a long wait.
    fn pacing(&self) -> (usize, Duration) {
        let windows: std::collections::BTreeSet<u128> =
            self.arrivals.iter().map(|at| at.as_millis() / 50).collect();
        let longest = self
            .arrivals
            .windows(2)
            .filter_map(|pair| match pair {
                [earlier, later] => Some(later.saturating_sub(*earlier)),
                _ => None,
            })
            .max()
            .unwrap_or_default();
        (windows.len(), longest)
    }
}

/// The tool calls of one completion, by part.
pub struct StreamView {
    started: Instant,
    calls: BTreeMap<usize, CallView>,
}

impl StreamView {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            calls: BTreeMap::new(),
        }
    }

    fn stamp(&self) -> String {
        format!("[{:>6}ms]", self.started.elapsed().as_millis())
    }

    /// Print `event` and fold it into the view.
    pub fn event(&mut self, event: &StreamEvent) {
        let stamp = self.stamp();
        let at = self.started.elapsed();
        match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name,
                tool_slot_state,
                ..
            } => {
                let new_call = |tool| CallView {
                    tool,
                    fragments: 0,
                    arguments: String::new(),
                    arrivals: Vec::new(),
                    ended: None,
                };
                match tool_slot_state {
                    None => {
                        let tool = name.as_ref().map(|name| name.as_str().to_owned());
                        println!(
                            "{stamp} part {} start  tool call `{}`",
                            part.index(),
                            tool.as_deref().unwrap_or("?"),
                        );
                        self.calls.insert(part.index(), new_call(tool));
                    }
                    Some(ToolSlotState::PendingName) => {
                        println!(
                            "{stamp} part {} start  tool call pending name",
                            part.index(),
                        );
                        self.calls.insert(part.index(), new_call(None));
                    }
                    Some(ToolSlotState::Named) => {
                        let (Some(name), Some(call)) =
                            (name.as_ref(), self.calls.get_mut(&part.index()))
                        else {
                            return;
                        };
                        call.tool = Some(name.as_str().to_owned());
                        println!(
                            "{stamp} part {} name   tool call `{name}`",
                            part.index(),
                        );
                    }
                }
            }
            StreamEvent::Start { part, kind, .. } => {
                println!("{stamp} part {} start  {kind:?}", part.index());
            }
            StreamEvent::Resume { part } => {
                println!("{stamp} part {} resume", part.index());
            }
            StreamEvent::Arguments { part, json } => {
                let Some(call) = self.calls.get_mut(&part.index()) else {
                    return;
                };
                call.fragments += 1;
                call.arguments.push_str(json);
                call.arrivals.push(at);
                let partial = Value::Object(parse_partial_arguments(&call.arguments));
                println!(
                    "{stamp} part {} args   +{:>4}B {:?}  now {}",
                    part.index(),
                    json.len(),
                    clip(json, 24),
                    preview(&partial),
                );
            }
            StreamEvent::Input { part, input } => {
                // Custom input is raw text, including JSON-looking fragments.
                println!(
                    "{stamp} part {} input  +{:>4}B {input:?}",
                    part.index(),
                    input.len(),
                );
            }
            StreamEvent::Discard { part, reason } => {
                // Earlier printed fragments remain diagnostic; no call completed.
                self.calls.remove(&part.index());
                println!("{stamp} part {} discard {reason:?}", part.index());
            }
            StreamEvent::Text { part, text } => {
                println!("{stamp} part {} text   {:?}", part.index(), clip(text, 60));
            }
            StreamEvent::Reasoning { part, text } => {
                println!("{stamp} part {} think  {:?}", part.index(), clip(text, 60));
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(tool_call),
                reasoning_authority: _,
            } => {
                let id = tool_call.id.to_string();
                println!(
                    "{stamp} part {} end    `{}` id {id}",
                    part.index(),
                    tool_call.function.name
                );
                if let Some(call) = self.calls.get_mut(&part.index()) {
                    call.ended = Some((id, tool_call.function.arguments_value()));
                }
            }
            StreamEvent::End { part, .. } => {
                println!("{stamp} part {} end", part.index());
            }
        }
    }

    /// Print one row per tool call and whether its joined fragments are
    /// the arguments its end states. Returns whether every ended call's
    /// fragments matched.
    pub fn summary(&self) -> bool {
        if self.calls.is_empty() {
            return true;
        }
        println!(
            "\n{:<5} {:<12} {:>9} {:>14} {:>8} {:>9} {:<32} matches end",
            "part", "tool", "fragments", "first..last", "windows", "max gap", "call id"
        );
        let mut all_match = true;
        for (part, call) in &self.calls {
            let window = match (call.arrivals.first(), call.arrivals.last()) {
                (Some(first), Some(last)) => {
                    format!("{}..{}ms", first.as_millis(), last.as_millis())
                }
                _ => "-".to_owned(),
            };
            let (windows, longest) = call.pacing();
            let (id, matches) = match &call.ended {
                Some((id, arguments)) => {
                    let joined = parse_partial_arguments(&call.arguments);
                    // The end's arguments are the joined text, read whole.
                    let matches = serde_json::from_str::<Value>(&call.arguments).ok().as_ref()
                        == Some(arguments)
                        && Value::Object(joined) == *arguments;
                    all_match &= matches;
                    (id.as_str(), if matches { "yes" } else { "NO" })
                }
                None => ("(never ended)", "-"),
            };
            println!(
                "{part:<5} {:<12} {:>9} {window:>14} {windows:>8} {:>7}ms {:<32} {matches}",
                call.tool.as_deref().unwrap_or("?"),
                call.fragments,
                longest.as_millis(),
                clip(id, 32)
            );
        }
        all_match
    }
}

/// `text` cut to `max` characters.
fn clip(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text.to_owned(),
    }
}

/// `value` with long strings shortened to their head and length.
fn preview(value: &Value) -> String {
    fn shorten(value: &Value) -> Value {
        match value {
            Value::String(text) if text.chars().count() > 24 => Value::String(format!(
                "{} ({} chars)",
                clip(text, 16),
                text.chars().count()
            )),
            Value::Array(items) => Value::Array(items.iter().map(shorten).collect()),
            Value::Object(object) => Value::Object(
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), shorten(value)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    shorten(value).to_string()
}

#[cfg(test)]
mod carried_tests {
    use super::*;
    use rig::message::{CallId, ToolCall, ToolFunction, ToolName};
    use rig::streaming::Transcript;
    use serde_json::json;

    #[test]
    fn late_name_preserves_pending_fragments_and_arrivals_on_the_same_part() {
        let call = ToolCall::new(
            CallId::from_wire("call_1"),
            ToolFunction::new(ToolName::new("lookup").expect("actual name"), json!({"q": 1})),
        );
        let ended = serde_json::to_value(AssistantContent::ToolCall(call))
            .expect("actual canonical tool call");
        let transcript = Transcript::parse(json!([
            {"item":"event", "value":{"event":"start", "part":0,
                "kind":"tool_call", "tool_slot_state":"pending_name"}},
            {"item":"event", "value":{"event":"arguments", "part":0, "json":"{\"q\":"}},
            {"item":"event", "value":{"event":"start", "part":0,
                "kind":"tool_call", "tool_slot_state":"named", "name":"lookup"}},
            {"item":"event", "value":{"event":"arguments", "part":0, "json":"1}"}},
            {"item":"event", "value":{"event":"end", "part":0, "content":ended}}
        ])).expect("checked pending-name transcript closes its one call");
        let events: Vec<_> = transcript.events().collect();
        let mut view = StreamView::new();
        for event in &events[..2] {
            view.event(event);
        }
        let pending = view.calls.get(&0).expect("pending view exists");
        assert_eq!(pending.tool, None);
        assert_eq!(pending.arguments, "{\"q\":");
        assert_eq!(pending.fragments, 1);
        let arrivals = pending.arrivals.clone();

        view.event(events[2]);
        let named = view.calls.get(&0).expect("same view after binding");
        assert_eq!(named.tool.as_deref(), Some("lookup"));
        assert_eq!(named.arguments, "{\"q\":");
        assert_eq!(named.fragments, 1);
        assert_eq!(named.arrivals, arrivals);
        assert!(named.ended.is_none());
        for event in &events[3..] {
            view.event(event);
        }
        assert_eq!(view.calls.len(), 1);
        let completed = view.calls.get(&0).expect("original call closes");
        assert_eq!(completed.arguments, "{\"q\":1}");
        assert_eq!(completed.fragments, 2);
        assert_eq!(completed.arrivals.len(), 2);
        assert_eq!(completed.arrivals[..1], arrivals);
        assert_eq!(completed.ended, Some(("call_1".to_owned(), json!({"q": 1}))));
        assert!(view.summary());
    }

    #[test]
    fn default_named_start_keeps_the_existing_insertion_behavior() {
        let transcript = Transcript::parse_prefix(json!([
            {"item":"event", "value":{"event":"start", "part":0,
                "kind":"tool_call", "name":"lookup"}},
            {"item":"event", "value":{"event":"arguments", "part":0, "json":"{}"}}
        ])).expect("checked default open call");
        let mut view = StreamView::new();
        for event in transcript.events() {
            view.event(event);
        }
        assert_eq!(view.calls.len(), 1);
        let call = view.calls.get(&0).expect("default named view");
        assert_eq!(call.tool.as_deref(), Some("lookup"));
        assert_eq!(call.arguments, "{}");
        assert_eq!(call.fragments, 1);
        assert_eq!(call.arrivals.len(), 1);
        assert!(call.ended.is_none());
    }

    #[test]
    fn discard_removes_only_its_call_and_preserves_other_pending_fragments() {
        let events: Vec<StreamEvent> = serde_json::from_value(json!([
            {"event": "start", "part": 0, "kind": "tool_call", "name": "first"},
            {"event": "arguments", "part": 0, "json": "{\"unfinished\":"},
            {"event": "start", "part": 1, "kind": "tool_call", "name": "second"},
            {"event": "arguments", "part": 1, "json": " { \"kept\": true } "},
            {"event": "discard", "part": 0, "reason": "length"}
        ]))
        .expect("actual stream event shapes");
        let mut view = StreamView::new();
        for event in &events[..4] {
            view.event(event);
        }
        let discarded = view.calls.get(&0).expect("first call started");
        assert_eq!(discarded.arguments, "{\"unfinished\":");
        assert!(discarded.ended.is_none());

        view.event(&events[4]);

        assert!(!view.calls.contains_key(&0));
        assert_eq!(view.calls.len(), 1);
        let kept = view.calls.get(&1).expect("other call remains open");
        assert_eq!(kept.arguments, " { \"kept\": true } ");
        assert_eq!(kept.fragments, 1);
        assert!(kept.ended.is_none(), "discard does not complete another call");
    }
}
