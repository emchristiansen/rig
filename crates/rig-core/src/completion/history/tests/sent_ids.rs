//! The ids a prepared request states it sends for each tool call
//! ([`SentIds`]), against the body the wire encodes for it. Turns are folded by the wire's own
//! decoder, requests prepared by `Completion::prepare_with_sent_ids` and
//! encoded by the wire, as a run sends them.

use std::sync::Arc;

use serde_json::{Value, json};

use super::super::{Sent, SentIds};
use crate::completion::{CompletionRequest, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, LocalCallId, Message, StopReason, ToolCall,
    ToolFunction, ToolName,
};
use crate::operation::Completion;
use crate::providers::anthropic::wire::{AnthropicConfig, Messages};
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::wire::Responses;
use crate::test_utils::json_body;
use crate::wire::{Encoded, Mode, Wire, WireFrame};

const SONNET: &str = "claude-sonnet-4-6";
const HAIKU: &str = "claude-haiku-4-5";
const GPT: &str = "gpt-5.4";
const OTHER_GPT: &str = "gpt-5.2";

/// A wire under test: the model requests go to, another model of the same
/// wire, and how its replies and bodies are written.
trait Case: Wire<Op = Completion, Frame = WireFrame, Payload = Encoded> {
    /// The same wire addressing the other model.
    fn other(&self) -> Self;
    /// A reply of this wire holding one call per `(id, x)`, each to `add`
    /// with `{"x": x}`.
    fn reply(&self, calls: &[(&str, i64)]) -> Value;
    /// The call ids of `body`'s call items and of its result items, in
    /// order.
    fn items(body: &Value) -> (Vec<String>, Vec<String>);
}

impl Case for Messages {
    fn other(&self) -> Self {
        AnthropicConfig::new("test-key").completion(HAIKU)
    }

    fn reply(&self, calls: &[(&str, i64)]) -> Value {
        let content: Vec<Value> = calls
            .iter()
            .map(|(id, x)| json!({"type": "tool_use", "id": id, "name": "add", "input": {"x": x}}))
            .collect();
        json!({
            "type": "message", "id": "msg_1", "model": self.model, "role": "assistant",
            "content": content, "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
    }

    fn items(body: &Value) -> (Vec<String>, Vec<String>) {
        let blocks = body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|message| message["content"].as_array().into_iter().flatten());
        let (mut calls, mut results) = (Vec::new(), Vec::new());
        for block in blocks {
            match block["type"].as_str() {
                Some("tool_use") => calls.push(text(&block["id"])),
                Some("tool_result") => results.push(text(&block["tool_use_id"])),
                _ => {}
            }
        }
        (calls, results)
    }
}

impl Case for Responses {
    fn other(&self) -> Self {
        let mut other = self.clone();
        other.model = OTHER_GPT.to_owned();
        other
    }

    fn reply(&self, calls: &[(&str, i64)]) -> Value {
        let output: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(at, (id, x))| {
                json!({
                    "type": "function_call", "id": format!("fc_{at}"), "call_id": id,
                    "name": "add", "arguments": json!({"x": x}).to_string(),
                    "status": "completed"
                })
            })
            .collect();
        json!({
            "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
            "model": self.model, "output": output, "tools": []
        })
    }

    fn items(body: &Value) -> (Vec<String>, Vec<String>) {
        let (mut calls, mut results) = (Vec::new(), Vec::new());
        for item in body["input"].as_array().into_iter().flatten() {
            match item["type"].as_str() {
                Some("function_call" | "custom_tool_call") => calls.push(text(&item["call_id"])),
                Some("function_call_output" | "custom_tool_call_output") => {
                    results.push(text(&item["call_id"]))
                }
                _ => {}
            }
        }
        (calls, results)
    }
}

fn text(value: &Value) -> String {
    value.as_str().expect("an id is a string").to_owned()
}

fn messages() -> Messages {
    AnthropicConfig::new("test-key").completion(SONNET)
}

fn responses() -> Responses {
    Responses::new(OpenAIConfig::new("sk-test"), GPT)
}

/// The Codex dialect of Responses, the ChatGPT backend.
fn codex() -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&crate::providers::chatgpt::DIALECT, "test-token"),
        GPT,
    )
}

/// The turn `wire`'s decoder folds from its reply holding `calls`.
fn fold<W: Case>(wire: &W, calls: &[(&str, i64)]) -> AssistantMessage {
    let reply = wire.reply(calls);
    let response = crate::wire::decode_reply(
        wire,
        &CompletionRequest::new("hi"),
        Mode::Unary,
        [WireFrame::Text(reply.to_string())],
    )
    .expect("the reply folds");
    let Some(Message::Assistant(turn)) = response.message() else {
        panic!("an assistant turn: {:?}", response.choice);
    };
    turn
}

/// A result answering `call` with `output`.
fn answer(call: &CallId, output: &str) -> Message {
    Message::tool_result(call.clone(), add(), output)
}

/// Results answering each call of `turn` in order.
fn answers(turn: &AssistantMessage) -> Message {
    let content = turn
        .tool_calls()
        .flat_map(|call| match answer(&call.id, "ok") {
            Message::User { content } => content,
            _ => Vec::new(),
        })
        .collect();
    Message::User { content }
}

fn add() -> ToolName {
    ToolName::new("add").expect("a tool name")
}

/// `history` as a request, declaring the tool `add` when `tools`.
fn request(history: Vec<Message>, tools: bool) -> CompletionRequest {
    let mut request = CompletionRequest::new("hi");
    request.chat_history = history.into_iter().map(Arc::new).collect();
    if tools {
        request.tools = vec![Arc::new(ToolDefinition {
            name: add(),
            description: "add".to_owned(),
            parameters: json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
            annotations: Default::default(),
        })];
    }
    request
}

/// The body `wire` encodes `prepared` to.
fn body<W: Case>(wire: &W, prepared: CompletionRequest) -> Value {
    let encoded = wire
        .encode(prepared, Mode::Unary)
        .expect("the request encodes");
    json_body(&encoded.request)
}

/// The statement and the body for `history` on `wire`, declaring the tool
/// `add` when `tools`.
fn sent<W: Case>(wire: &W, history: Vec<Message>, tools: bool) -> (SentIds, Value) {
    let (prepared, sent) =
        Completion::prepare_with_sent_ids(request(history, tools), &wire.describe())
            .expect("the request prepares");
    (sent, body(wire, prepared))
}

/// The ids `sent` states as sent, in history order.
fn stated(sent: &SentIds) -> Vec<String> {
    sent.calls()
        .filter_map(|(_, sent)| match sent {
            Sent::Under(id) => Some(id.clone()),
            Sent::Not => None,
        })
        .collect()
}

fn under(id: &str) -> Sent {
    Sent::Under(id.to_owned())
}

/// `[(position, what is sent)]` of every call.
fn statement(sent: &SentIds) -> Vec<((usize, usize), Sent)> {
    sent.calls().map(|(at, sent)| (at, sent.clone())).collect()
}

fn ids(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|id| (*id).to_owned()).collect()
}

/// One model's turns, calls written `call_0`, `call_0`, `call_1`: the
/// repeated id takes a counter, so the next takes one too.
fn three_turns<W: Case>(wire: &W) -> Vec<Message> {
    let mut history = vec![Message::user("hi")];
    for (id, x) in [("call_0", 1), ("call_0", 2), ("call_1", 3)] {
        let turn = fold(wire, &[(id, x)]);
        let results = answers(&turn);
        history.extend([Message::Assistant(turn), results]);
    }
    history
}

fn assert_three_turns<W: Case>(wire: &W) {
    let (sent, body) = sent(wire, three_turns(wire), true);
    assert_eq!(
        statement(&sent),
        vec![
            ((1, 0), under("call_0")),
            ((3, 0), under("call_1")),
            ((5, 0), under("call_2")),
        ]
    );
    assert_eq!(
        W::items(&body),
        (
            ids(&["call_0", "call_1", "call_2"]),
            ids(&["call_0", "call_1", "call_2"])
        )
    );
}

#[test]
fn one_models_repeated_ids_are_stated_as_renamed_on_messages() {
    assert_three_turns(&messages());
}

#[test]
fn one_models_repeated_ids_are_stated_as_renamed_on_responses() {
    assert_three_turns(&responses());
    assert_three_turns(&codex());
}

/// Another model's `call_0`, the target model's `call_0`, another model's
/// `call_1`, one tool and one payload: each takes the next free id in
/// history order.
fn assert_mixed_origins<W: Case>(wire: &W) {
    let other = wire.other();
    let mut history = vec![Message::user("hi")];
    for turn in [
        fold(&other, &[("call_0", 1)]),
        fold(wire, &[("call_0", 1)]),
        fold(&other, &[("call_1", 1)]),
    ] {
        let results = answers(&turn);
        history.extend([Message::Assistant(turn), results]);
    }
    let (sent, body) = sent(wire, history, true);
    assert_eq!(
        statement(&sent),
        vec![
            ((1, 0), under("call_0")),
            ((3, 0), under("call_1")),
            ((5, 0), under("call_2")),
        ]
    );
    assert_eq!(
        W::items(&body),
        (
            ids(&["call_0", "call_1", "call_2"]),
            ids(&["call_0", "call_1", "call_2"])
        )
    );
}

#[test]
fn mixed_origins_are_stated_in_history_order_on_messages() {
    assert_mixed_origins(&messages());
}

#[test]
fn mixed_origins_are_stated_in_history_order_on_responses() {
    assert_mixed_origins(&responses());
    assert_mixed_origins(&codex());
}

/// A call the provider sent without an id gets one rig issues, spelled
/// `tool-<n>` past every id a provider issued in the request: another
/// model's call already holds `tool-0`.
fn assert_local_ids<W: Case>(wire: &W) {
    let earlier = fold(&wire.other(), &[("tool-0", 1)]);
    let idless = fold(wire, &[("", 2)]);
    let Some(AssistantContent::ToolCall(call)) = idless.content.first() else {
        panic!("one call: {:?}", idless.content);
    };
    assert!(call.id.is_local(), "the call gets an id rig issues");
    let history = vec![
        Message::user("hi"),
        Message::Assistant(earlier.clone()),
        answers(&earlier),
        Message::Assistant(idless.clone()),
        answers(&idless),
    ];
    let (sent, body) = sent(wire, history, true);
    assert_eq!(
        statement(&sent),
        vec![((1, 0), under("tool-0")), ((3, 0), under("tool-1"))]
    );
    assert_eq!(
        W::items(&body),
        (ids(&["tool-0", "tool-1"]), ids(&["tool-0", "tool-1"]))
    );
}

#[test]
fn rig_issued_ids_are_stated_as_their_alias() {
    assert_local_ids(&messages());
    assert_local_ids(&responses());
    assert_local_ids(&codex());
}

/// A request that declares no tools gets its calls and results as text:
/// no call is sent as one.
fn assert_narrated<W: Case>(wire: &W) {
    let (sent, body) = sent(wire, three_turns(wire), false);
    assert_eq!(
        statement(&sent),
        vec![
            ((1, 0), Sent::Not),
            ((3, 0), Sent::Not),
            ((5, 0), Sent::Not)
        ]
    );
    assert_eq!(W::items(&body), (Vec::new(), Vec::new()));
}

#[test]
fn calls_narrated_without_tools_are_stated_as_not_sent() {
    assert_narrated(&messages());
    assert_narrated(&responses());
}

/// A turn that failed or was stopped is dropped with its calls, and the
/// result answering one is not sent either; the turn after it keeps the id
/// the dropped call never claimed.
fn assert_dropped_turn<W: Case>(wire: &W) {
    for stop in [
        StopReason::Error("connection lost".into()),
        StopReason::Aborted("cancelled".into()),
    ] {
        let mut failed = fold(wire, &[("call_0", 1)]);
        failed.stop = Some(stop);
        let retried = fold(wire, &[("call_0", 1)]);
        let history = vec![
            Message::user("hi"),
            Message::Assistant(failed.clone()),
            answers(&failed),
            Message::user("retry"),
            Message::Assistant(retried.clone()),
            answers(&retried),
        ];
        let (sent, body) = sent(wire, history, true);
        assert_eq!(
            statement(&sent),
            vec![((1, 0), Sent::Not), ((4, 0), under("call_0"))]
        );
        assert_eq!(W::items(&body), (ids(&["call_0"]), ids(&["call_0"])));
    }
}

#[test]
fn a_dropped_turns_calls_are_stated_as_not_sent() {
    assert_dropped_turn(&messages());
    assert_dropped_turn(&responses());
}

/// The user message holding one result per `(id, output)`, in order.
fn results(answered: &[(&CallId, &str)]) -> Message {
    let content = answered
        .iter()
        .flat_map(|(id, output)| match answer(id, output) {
            Message::User { content } => content,
            _ => Vec::new(),
        })
        .collect();
    Message::User { content }
}

/// Two calls of one turn written with one provider id, answered by two
/// results under it: the second call takes a counter, and the body's
/// results answer the calls in call order under the stated ids.
fn assert_shared_id_in_one_turn<W: Case>(wire: &W) {
    let call = |x: i64| AssistantContent::tool_call("call_0", add(), json!({ "x": x }));
    let turn = AssistantMessage::new(vec![call(1), call(2)]);
    let written = CallId::from_wire("call_0");
    let history = vec![
        Message::user("hi"),
        Message::Assistant(turn),
        results(&[(&written, "first"), (&written, "second")]),
    ];
    let (sent, body) = sent(wire, history, true);
    assert_eq!(
        statement(&sent),
        vec![((1, 0), under("call_0")), ((1, 1), under("call_1"))]
    );
    assert_eq!(
        W::items(&body),
        (ids(&["call_0", "call_1"]), ids(&["call_0", "call_1"]))
    );
}

#[test]
fn calls_sharing_an_id_in_one_turn_are_stated_apart() {
    assert_shared_id_in_one_turn(&messages());
    assert_shared_id_in_one_turn(&responses());
}

/// The answering results of a shared id on Messages, whole: `first`
/// answers the first call, `second` the second.
#[test]
fn results_of_a_shared_id_carry_their_own_answers_on_messages() {
    let call = |x: i64| AssistantContent::tool_call("call_0", add(), json!({ "x": x }));
    let written = CallId::from_wire("call_0");
    let history = vec![
        Message::user("hi"),
        Message::Assistant(AssistantMessage::new(vec![call(1), call(2)])),
        results(&[(&written, "first"), (&written, "second")]),
    ];
    let (_, body) = sent(&messages(), history, true);
    assert_eq!(
        body["messages"].get(2),
        Some(&json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_0", "content": [{"type": "text", "text": "first"}]},
                {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "second"}]}
            ]
        }))
    );
}

/// Two calls the caller wrote under one rig-issued id are not renamed:
/// both are stated under the one alias, and the body holds two call items
/// and two results under it, which a check of distinct ids refuses.
fn assert_shared_local_id<W: Case>(wire: &W) {
    let local = CallId::Local(LocalCallId::new());
    let call = |x: i64| {
        AssistantContent::ToolCall(ToolCall::new(
            local.clone(),
            ToolFunction::new(add(), json!({ "x": x })),
        ))
    };
    let history = vec![
        Message::user("hi"),
        Message::Assistant(AssistantMessage::new(vec![call(1), call(2)])),
        results(&[(&local, "first"), (&local, "second")]),
    ];
    let (sent, body) = sent(wire, history, true);
    assert_eq!(
        statement(&sent),
        vec![((1, 0), under("tool-0")), ((1, 1), under("tool-0"))]
    );
    assert_eq!(
        W::items(&body),
        (ids(&["tool-0", "tool-0"]), ids(&["tool-0", "tool-0"]))
    );
}

#[test]
fn calls_written_under_one_rig_issued_id_are_stated_under_one_alias() {
    assert_shared_local_id(&messages());
    assert_shared_local_id(&responses());
}

/// A small deterministic generator: a linear congruential sequence.
struct Draw(u64);

impl Draw {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// A history of one to three turns, each from the target model or another,
/// with one or two calls whose ids are drawn from `call_0`, `call_1` and
/// none (an id rig issues), every call answered; the next user message
/// follows the last results as a message of its own or not. Then whether
/// the request declares tools.
fn drawn<W: Case>(wire: &W, draw: &mut Draw) -> (Vec<Message>, bool) {
    let other = wire.other();
    let mut history = vec![Message::user("hi")];
    for _ in 0..=draw.below(3) {
        let calls: Vec<(&str, i64)> = (0..=draw.below(2))
            .map(|x| {
                let id = ["call_0", "call_1", ""]
                    .get(draw.below(3) as usize)
                    .copied()
                    .unwrap_or_default();
                (id, x as i64)
            })
            .collect();
        let turn = if draw.below(2) == 0 {
            fold(wire, &calls)
        } else {
            fold(&other, &calls)
        };
        let results = answers(&turn);
        history.extend([Message::Assistant(turn), results]);
    }
    if draw.below(2) == 0 {
        history.push(Message::user("next"));
    }
    (history, draw.below(5) != 0)
}

/// Every call of `history`, by position.
fn written(history: &[Message]) -> Vec<(usize, usize)> {
    let mut calls = Vec::new();
    for (at, message) in history.iter().enumerate() {
        if let Message::Assistant(turn) = message {
            calls.extend(
                (turn.content.iter().enumerate())
                    .filter(|(_, block)| matches!(block, AssistantContent::ToolCall(_)))
                    .map(|(block, _)| (at, block)),
            );
        }
    }
    calls
}

fn assert_statement_is_the_body<W: Case>(wire: &W, seed: u64) {
    let mut draw = Draw(seed);
    for case in 0..100 {
        let (history, tools) = drawn(wire, &mut draw);
        let (sent, body) = sent(wire, history.clone(), tools);
        let positions: Vec<(usize, usize)> = sent.calls().map(|(at, _)| at).collect();
        assert_eq!(
            positions,
            written(&history),
            "case {case}: every call is stated, at its position: {history:?}"
        );
        let (calls, results) = W::items(&body);
        let stated = stated(&sent);
        assert_eq!(
            stated, calls,
            "case {case}, tools {tools}: the stated ids are the body's: {history:?}"
        );
        assert!(
            results.iter().all(|id| stated.contains(id)),
            "case {case}: every result answers a stated id: {results:?}, {stated:?}"
        );
    }
}

#[test]
fn the_stated_ids_are_the_bodys_on_drawn_histories() {
    assert_statement_is_the_body(&messages(), 1);
    assert_statement_is_the_body(&responses(), 2);
    assert_statement_is_the_body(&codex(), 3);
}

/// Stating the ids changes nothing about the request: `prepare` and
/// `prepare_with_sent_ids` give equal prepared requests that encode to
/// equal bodies.
fn assert_statement_changes_nothing<W: Case>(wire: &W, seed: u64) {
    use crate::wire::Operation;
    let mut draw = Draw(seed);
    for case in 0..100 {
        let (history, tools) = drawn(wire, &mut draw);
        let plain = Completion::prepare(request(history.clone(), tools), &wire.describe())
            .expect("the request prepares");
        let (stated, _) =
            Completion::prepare_with_sent_ids(request(history, tools), &wire.describe())
                .expect("the request prepares");
        assert_eq!(plain, stated, "case {case}: the prepared requests");
        assert_eq!(
            body(wire, plain),
            body(wire, stated),
            "case {case}: the encoded bodies"
        );
    }
}

#[test]
fn stating_the_ids_changes_nothing_sent() {
    assert_statement_changes_nothing(&messages(), 1);
    assert_statement_changes_nothing(&responses(), 2);
    assert_statement_changes_nothing(&codex(), 3);
}
