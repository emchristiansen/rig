//! The OpenAI Responses wire's history suite, and the fixture its dialects
//! (ChatGPT, Copilot's Responses route, xAI) share: replies are the API's
//! own response objects and SSE events, and the body is the JSON the wire
//! sends.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{
    AssistantContent, AssistantMessage, CarriedPayload, CarriedTextContent, ProviderItem,
    Reasoning, Text, ToolCall, ToolFunction, ToolName,
};
use rig_core::operation::Completion;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::streaming::IncompleteTerminal;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Encoded, Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablated, Ablation, CallShape, Ending, HistoryFixture, Refusal, RefusedArguments, Rng, Shape,
    http_body, replies,
};
use serde_json::{Value, json};

/// One Responses wire or dialect: how to build it for a model, and the
/// models its replies come from.
pub struct ResponsesHistory<W> {
    /// The wire addressing a model.
    pub wire: fn(&str) -> W,
    /// The model the replies come from.
    pub model: &'static str,
    /// Another model on the same wire.
    pub other_model: &'static str,
    /// A model on the wire that reads no images.
    pub text_only_model: Option<&'static str>,
    /// Whether the dialect reads item events that carry no numeric
    /// `output_index`; the strict reader of the others refuses them.
    pub reads_indexless: bool,
    /// Whether the dialect reads a whole reply from a buffered SSE body
    /// (ChatGPT), whose frame decode errors say so.
    pub buffered: bool,
    /// Whether the dialect's strict options publish calls only at the
    /// terminal (G207), so a stream cut before its end publishes none; xAI's
    /// publish each call when it is done.
    pub calls_at_terminal: bool,
    /// The wire set to accept a streamed `response.incomplete`, which it
    /// refuses by default (G193).
    pub accepting: fn(W) -> W,
}

/// A JSON value that is not an object, as call argument text: the strict
/// reader keeps it exactly beside empty arguments (G209/G210).
fn non_object(arguments: &str) -> bool {
    serde_json::from_str::<Value>(arguments).is_ok_and(|value| !value.is_object())
}

fn message(id: &str, text: &str, extra: Value) -> Value {
    let mut item = json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [{
                "type": "url_citation",
                "url": "https://example.com",
                "start_index": 0,
                "end_index": 1,
                "title": "Example",
            }],
            "logprobs": [],
        }],
    });
    if let (Some(item), Value::Object(extra)) = (item.as_object_mut(), extra) {
        item.extend(extra);
    }
    item
}

fn reasoning(id: &str, summary: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [{"type": "summary_text", "text": summary}],
        "encrypted_content": format!("ciphertext-of-{id}"),
    })
}

fn call(id: &str, call_id: &str, arguments: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": "completed",
    })
}

fn output(shape: Shape) -> Vec<Value> {
    match shape {
        Shape::Rich => vec![
            reasoning("rs_1", "Plan the lookup."),
            message("msg_1", "Looking it up.", json!({"phase": "commentary"})),
            call("fc_1", "call_1", r#"{"q":"rig"}"#),
            json!({
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "rig"},
            }),
        ],
        Shape::Interleaved => vec![
            reasoning("rs_1", "First."),
            message("msg_1", "Between.", json!({})),
            reasoning("rs_2", "Second."),
            call("fc_1", "call_1", r#"{"q":"rig"}"#),
        ],
        Shape::Unknown => vec![
            json!({"type": "x_rig_invented", "id": "x_1", "payload": {"n": 1}}),
            message("msg_1", "Noted.", json!({"x_rig_field": true})),
        ],
    }
}

/// The response object a reply of `output` ends with.
fn response(output: &[Value], status: Value) -> Value {
    let mut response = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "the-model",
        "output": output,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 30,
        },
    });
    if let (Some(fields), Value::Object(status)) = (response.as_object_mut(), status) {
        fields.extend(status);
    }
    response
}

fn completed() -> Value {
    json!({"status": "completed"})
}

fn whole(output: &[Value], status: Value) -> Vec<WireFrame> {
    vec![WireFrame::Text(response(output, status).to_string())]
}

/// `item` as `output_item.added` states it: its text not yet streamed.
fn added(item: &Value) -> Value {
    let mut item = item.clone();
    if let Some(fields) = item.as_object_mut() {
        match fields.get("type").and_then(Value::as_str) {
            Some("message") => {
                fields.insert("content".to_owned(), json!([]));
                fields.insert("status".to_owned(), json!("in_progress"));
            }
            Some("reasoning") => {
                fields.insert("summary".to_owned(), json!([]));
                fields.shift_remove("encrypted_content");
            }
            Some("function_call") => {
                fields.insert("arguments".to_owned(), json!(""));
                fields.insert("status".to_owned(), json!("in_progress"));
            }
            _ => {
                fields.insert("status".to_owned(), json!("in_progress"));
            }
        }
    }
    item
}

/// The stream that restates `output`: each item added, its text streamed
/// in two deltas, then done, and the terminal response.
fn streamed(output: &[Value], status: Value) -> Vec<WireFrame> {
    let mut events = vec![json!({
        "type": "response.created",
        "response": {"id": "resp_1", "object": "response", "created_at": 1_700_000_000,
            "model": "the-model", "status": "in_progress", "output": []},
    })];
    for (index, item) in output.iter().enumerate() {
        events.push(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": added(item),
        }));
        let (kind, text) = match item.get("type").and_then(Value::as_str) {
            Some("message") => (
                "response.output_text.delta",
                item.pointer("/content/0/text").and_then(Value::as_str),
            ),
            Some("reasoning") => (
                "response.reasoning_summary_text.delta",
                item.pointer("/summary/0/text").and_then(Value::as_str),
            ),
            Some("function_call") => (
                "response.function_call_arguments.delta",
                item.get("arguments").and_then(Value::as_str),
            ),
            _ => ("", None),
        };
        if let Some(text) = text {
            let middle = text.char_indices().nth(text.chars().count() / 2);
            let (head, tail) = text.split_at(middle.map_or(text.len(), |(at, _)| at));
            for delta in [head, tail] {
                events.push(json!({
                    "type": kind,
                    "output_index": index,
                    "item_id": item.get("id"),
                    "content_index": 0,
                    "summary_index": 0,
                    "delta": delta,
                }));
            }
        }
        events.push(json!({
            "type": "response.output_item.done",
            "output_index": index,
            "item": item,
        }));
    }
    let terminal = match status.get("status").and_then(Value::as_str) {
        Some("incomplete") => "response.incomplete",
        _ => "response.completed",
    };
    events.push(json!({"type": terminal, "response": response(output, status)}));
    self::events(&events)
}

/// One frame per event, each numbered in order by its `sequence_number`,
/// which the strict reader requires of every event.
/// A stream of one done message ending on a `response.incomplete` whose
/// response states no `status`, with `extra` fields.
fn statusless_incomplete(extra: Value) -> Vec<WireFrame> {
    let text = [message("msg_1", "Done.", json!({}))];
    events(&[
        json!({"type": "response.output_item.done", "output_index": 0, "item": text[0]}),
        json!({"type": "response.incomplete", "response": response(&text, extra)}),
    ])
}

fn events(events: &[Value]) -> Vec<WireFrame> {
    events
        .iter()
        .enumerate()
        .map(|(number, event)| {
            let mut event = event.clone();
            event["sequence_number"] = json!(number);
            WireFrame::Text(event.to_string())
        })
        .collect()
}

fn reply(output: &[Value], status: Value, mode: Mode) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => whole(output, status),
        Mode::Streaming => streamed(output, status),
    }
}

/// A rich whole reply as a provider echoes it, request config included.
fn document() -> Value {
    let mut document = response(&output(Shape::Rich), completed());
    if let Some(fields) = document.as_object_mut() {
        fields.extend(
            json!({
                "error": null,
                "incomplete_details": null,
                "instructions": "Be brief.",
                "max_output_tokens": null,
                "parallel_tool_calls": true,
                "previous_response_id": null,
                "reasoning": {"effort": "medium", "summary": "auto"},
                "service_tier": "default",
                "store": false,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "auto",
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "description": null,
                    "parameters": {"type": "object", "properties": {}},
                    "strict": null,
                }],
                "top_p": 1.0,
                "truncation": "disabled",
                "user": null,
                "metadata": {},
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
        );
    }
    document
}

impl<W> HistoryFixture for ResponsesHistory<W>
where
    W: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame>,
{
    type Wire = W;

    fn wire(&self, model: &str) -> W {
        (self.wire)(model)
    }

    fn model(&self) -> &'static str {
        self.model
    }

    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        Some(serde_json::json!({
            "type": "function", "name": tool.name.as_str(),
            "description": tool.description, "parameters": tool.parameters,
        }))
    }

    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        Some(replies::responses_spec(rng))
    }

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<WireFrame>> {
        let (whole, streamed) = replies::responses_build(spec, self.reads_indexless);
        Some(replies::Frames {
            whole: replies::texts(whole),
            streamed: replies::texts(streamed),
        })
    }

    fn other_model(&self) -> &'static str {
        self.other_model
    }

    fn text_only_model(&self) -> Option<&'static str> {
        self.text_only_model
    }

    fn body(&self, wire: &W, request: CompletionRequest, mode: Mode) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(&output(shape), completed(), mode))
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(
            &[call("fc_1", "call_1", arguments)],
            completed(),
            mode,
        ))
    }

    /// Argument text that is not JSON fails the call with its malformed
    /// input, whole or streamed.
    fn refused_arguments(&self, arguments: &str, _mode: Mode) -> Option<RefusedArguments> {
        serde_json::from_str::<Value>(arguments)
            .is_err()
            .then_some(RefusedArguments::MalformedToolInput)
    }

    /// `null`, a JSON string and an array keep their exact text beside
    /// empty arguments (G209/G210; R §7).
    fn kept_arguments(&self, arguments: &str) -> Option<(Value, Option<String>)> {
        non_object(arguments).then(|| (json!({}), Some(arguments.to_owned())))
    }

    /// A response without a `status`, whole or as a streamed
    /// `response.incomplete`, fails its frame's decode in the strict reader
    /// (G139/G191; R §2, §6). The ChatGPT buffered body names the event it
    /// could not read instead of the missing field.
    fn refused_finish(&self, name: &str) -> Option<Refusal> {
        let streamed = match name {
            "no status" => {
                let frame = response(&[message("msg_1", "Done.", json!({}))], json!({}));
                let buffered = if self.buffered {
                    "invalid JSON frame in buffered Responses SSE body: "
                } else {
                    ""
                };
                return Some(Refusal::CorruptFrame(format!(
                    "frame failed to decode: {buffered}missing field `status`; frame: {frame}"
                )));
            }
            "streamed incomplete at the output cap, without a status" => statusless_incomplete(
                json!({"incomplete_details": {"reason": "max_output_tokens"}}),
            ),
            "streamed incomplete without a status or a reason" => statusless_incomplete(json!({})),
            _ => return None,
        };
        let Some(WireFrame::Text(frame)) = streamed.last() else {
            unreachable!("the stream ends on its text terminal");
        };
        let error = if self.buffered {
            "malformed `response.incomplete` event in buffered Responses SSE body"
        } else {
            "missing field `status`"
        };
        Some(Refusal::CorruptFrame(format!(
            "`response.incomplete` frame failed to decode: {error}; frame: {frame}"
        )))
    }

    /// The invented field of the known message is kept for inspection only
    /// (A25/A27/A35; R §8): the invented item replays exactly, the replayed
    /// message has no `x_rig_field`, and the decoded text reports it.
    fn inspected_field(&self, turn: &AssistantMessage, body: &Value) -> Option<Vec<String>> {
        let mut problems = Vec::new();
        let input = body["input"].as_array().cloned().unwrap_or_default();
        let invented = json!({"type": "x_rig_invented", "id": "x_1", "payload": {"n": 1}});
        if !input.contains(&invented) {
            problems.push(format!("the body replays {invented} exactly: {body}"));
        }
        let messages: Vec<&Value> = input
            .iter()
            .filter(|item| item["type"] == "message" && item["role"] == "assistant")
            .collect();
        match messages.as_slice() {
            [message] if message.get("x_rig_field").is_none() => {}
            messages => problems.push(format!(
                "one replayed assistant message, without `x_rig_field`: {messages:?}"
            )),
        }
        let unknown: Vec<_> = turn
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::CarriedText(text) => Some(text.unknown_message_fields()),
                _ => None,
            })
            .collect();
        let expected = json!({"x_rig_field": true});
        if unknown != [expected.as_object()] {
            problems.push(format!(
                "the decoded text reports {expected} as its unknown message fields: {unknown:?}"
            ));
        }
        Some(problems)
    }

    /// Calls defer to the terminal (G207), reasoning closes late, and
    /// carried blocks are built at the terminal, so the cut reply keeps
    /// canonical blocks (R §10): the third is the hosted search, since no
    /// call is published. xAI's options publish each call when it is done,
    /// so its third is the call.
    fn cut_blocks(&self) -> Option<Vec<AssistantContent>> {
        let third = if self.calls_at_terminal {
            AssistantContent::ProviderItem(ProviderItem {
                payload: CarriedPayload::new(
                    json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                           "action": {"type": "search", "query": "rig"}}),
                    None,
                ),
            })
        } else {
            AssistantContent::ToolCall(ToolCall::from_wire(
                "call_1",
                ToolFunction::new(
                    ToolName::new("lookup").expect("a tool name"),
                    json!({"q": "rig"}),
                ),
            ))
        };
        Some(vec![
            AssistantContent::Reasoning(Reasoning::new("Plan the lookup.")),
            AssistantContent::Text(Text::new("Looking it up.")),
            third,
        ])
    }

    fn cuts_publish_no_calls(&self) -> Option<&'static str> {
        self.calls_at_terminal.then_some(
            "strict options publish calls only at the terminal (G207), so a cut stream publishes none",
        )
    }

    /// A streamed `response.incomplete` fails the stream by default with
    /// exactly its frame (G193; R §2).
    fn refused_incomplete(&self, streamed: &[WireFrame]) -> Option<(String, W)> {
        let body = streamed.iter().find_map(|frame| match frame {
            WireFrame::Text(text)
                if serde_json::from_str::<Value>(text)
                    .is_ok_and(|event| event["type"] == "response.incomplete") =>
            {
                Some(text.clone())
            }
            _ => None,
        })?;
        Some((body, (self.accepting)((self.wire)(self.model))))
    }

    /// Every status the API documents, every incomplete reason, and a body
    /// that states no status, which the strict reader refuses, as it
    /// refuses a streamed `response.incomplete` without one. A call
    /// announced and never done fails the turn.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        let text = [message("msg_1", "Done.", json!({}))];
        let calling = [call("fc_1", "call_1", "{}")];
        let incomplete = |reason: Value| json!({"status": "incomplete", "incomplete_details": {"reason": reason}});
        let failed = |status: &str| json!({"status": status, "error": {"code": "server_error", "message": "boom"}});
        vec![
            ("completed", whole(&text, completed()), Ending::Success),
            (
                "completed with a call",
                whole(&calling, completed()),
                Ending::Success,
            ),
            (
                "incomplete: max_output_tokens",
                whole(&text, incomplete(json!("max_output_tokens"))),
                Ending::Success,
            ),
            (
                "incomplete: content_filter",
                whole(&text, incomplete(json!("content_filter"))),
                Ending::Failure,
            ),
            (
                "incomplete: an unknown reason",
                whole(&text, incomplete(json!("max_tool_calls"))),
                Ending::Failure,
            ),
            (
                "incomplete without a reason",
                whole(
                    &text,
                    json!({"status": "incomplete", "incomplete_details": null}),
                ),
                Ending::Failure,
            ),
            ("failed", whole(&text, failed("failed")), Ending::Failure),
            (
                "cancelled",
                whole(&text, failed("cancelled")),
                Ending::Failure,
            ),
            (
                "queued",
                whole(&[], json!({"status": "queued"})),
                Ending::Failure,
            ),
            (
                "in_progress",
                whole(&text, json!({"status": "in_progress"})),
                Ending::Failure,
            ),
            (
                "an unknown status",
                whole(&text, json!({"status": "paused"})),
                Ending::Failure,
            ),
            ("no status", whole(&text, json!({})), Ending::Failure),
            (
                "streamed incomplete at the output cap, without a status",
                statusless_incomplete(
                    json!({"incomplete_details": {"reason": "max_output_tokens"}}),
                ),
                Ending::Failure,
            ),
            (
                "streamed incomplete without a status or a reason",
                statusless_incomplete(json!({})),
                Ending::Failure,
            ),
            (
                "a call announced and never done",
                events(&[
                    json!({"type": "response.output_item.added", "output_index": 0, "item": added(&calling[0])}),
                    json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"}),
                    json!({"type": "response.completed", "response": response(&[], completed())}),
                ]),
                Ending::Failure,
            ),
        ]
    }

    /// The item alone, as the `output` of a whole reply.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let item = block.native_item()?.clone();
        let response = rig_history_conformance::decode(
            &self.wire(self.model),
            &CompletionRequest::new("restate"),
            Mode::Unary,
            whole(&[item], completed()),
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Calls a stream tells apart by item, whatever their output index
    /// says; a whole reply's are its `output` list. A dialect whose strict
    /// reader refuses an item event without a numeric `output_index` has no
    /// stream that leaves it out or sends it `null`.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let calls = [
            json!({"type": "function_call", "id": "fc_a1", "call_id": "a1", "name": "weather",
                   "arguments": r#"{"city":"Paris"}"#, "status": "completed"}),
            json!({"type": "function_call", "id": "fc_b2", "call_id": "b2", "name": "weather",
                   "arguments": r#"{"city":"Rome"}"#, "status": "completed"}),
        ];
        let index = |at: usize| match shape {
            CallShape::Indexless => None,
            CallShape::NullIndex => Some(Value::Null),
            CallShape::ReusedIndex => Some(json!(0)),
            CallShape::WholeList => Some(json!(at)),
        };
        match (shape, mode) {
            (CallShape::WholeList, Mode::Unary) => Some(whole(&calls, completed())),
            (CallShape::Indexless | CallShape::NullIndex, Mode::Streaming)
                if !self.reads_indexless =>
            {
                None
            }
            (CallShape::WholeList, Mode::Streaming) | (_, Mode::Unary) => None,
            (_, Mode::Streaming) => {
                let mut stream = Vec::new();
                for (at, call) in calls.iter().enumerate() {
                    for event in [
                        json!({"type": "response.output_item.added", "item": added(call)}),
                        json!({"type": "response.function_call_arguments.delta", "item_id": call["id"], "delta": call["arguments"]}),
                        json!({"type": "response.output_item.done", "item": call}),
                    ] {
                        let mut event = event;
                        if let Some(index) = index(at) {
                            event["output_index"] = index;
                        }
                        stream.push(event);
                    }
                }
                stream.push(json!({"type": "response.completed", "response": response(&calls, completed())}));
                Some(events(&stream))
            }
        }
    }

    /// A completed response with no output: an empty turn, as pi keeps it.
    fn empty_reply(&self, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(&[], completed(), mode))
    }

    /// The stream's `error` event.
    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(
            json!({"type": "error", "code": "server_error", "message": "boom"}).to_string(),
        ))
    }

    /// The interleaved reply includes its carried reasoning and known text part.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        let interleaved = match turn.content.as_slice() {
            [
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_),
                text,
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_),
                AssistantContent::ToolCall(_),
            ] => {
                matches!(text, AssistantContent::Text(_))
                    || matches!(text, AssistantContent::CarriedText(text)
                    if matches!(&text.content, CarriedTextContent::Text(_)))
            }
            _ => false,
        };
        let replayed: Vec<Value> = turn
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Opaque(opaque) if opaque.replay => Some(opaque.item.clone()),
                AssistantContent::CarriedReasoning(reasoning) if interleaved => {
                    Some(reasoning.payload.item.clone())
                }
                AssistantContent::CarriedText(text)
                    if interleaved && matches!(&text.content, CarriedTextContent::Text(_)) =>
                {
                    Some(text.payload.item.clone())
                }
                block => block.native_item().cloned(),
            })
            .collect();
        if interleaved {
            assert_eq!(
                replayed.len(),
                4,
                "the interleaved reply supplies all four outbound items: {turn:?} gives {replayed:?}"
            );
        }
        replayed
    }

    /// Every field is ablated; the strict reader's refusals are
    /// [`Self::ablation_refusal`]'s.
    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: document(),
            required: &[],
            frames: |document| vec![WireFrame::Text(document.to_string())],
        })
    }

    /// The strict reader (G139/G191; R §6) refuses each change of
    /// [`STRICT_FIELDS`]: the document fails its decode as a whole reply and,
    /// holding `"error": null`, falls back to an error envelope, so the
    /// reply fails with the document as the provider's body. A null tool
    /// `description` is read as absent (Q9).
    fn ablation_refusal(
        &self,
        pointer: &str,
        change: Ablated,
        document: &Value,
    ) -> Option<Refusal> {
        STRICT_FIELDS
            .iter()
            .any(|(field, refused)| {
                *field == pointer
                    && refused[match change {
                        Ablated::Removed => 0,
                        Ablated::Null => 1,
                        Ablated::Retyped => 2,
                    }]
            })
            .then(|| Refusal::ProviderResponse {
                status: None,
                body: document.to_string(),
            })
    }
}

/// The ablation document's fields whose removal, `null` and retyping
/// (in that order) the strict reader refuses (`reader.rs` field sets, F's
/// typed reader; G139/G191); every other change decodes.
const STRICT_FIELDS: &[(&str, [bool; 3])] = &[
    ("/id", [true, true, true]),
    ("/object", [true, true, true]),
    ("/created_at", [true, true, true]),
    ("/model", [true, true, true]),
    ("/output", [false, true, true]),
    ("/output/0/id", [true, true, true]),
    ("/output/0/summary", [false, true, true]),
    ("/output/0/summary/0/type", [true, true, true]),
    ("/output/0/summary/0/text", [true, true, true]),
    ("/output/0/encrypted_content", [false, false, true]),
    ("/output/1/id", [true, true, true]),
    ("/output/1/role", [true, true, true]),
    ("/output/1/status", [true, true, true]),
    ("/output/1/content", [true, true, true]),
    ("/output/1/content/0/type", [true, true, true]),
    ("/output/1/content/0/text", [true, true, true]),
    ("/output/1/phase", [false, false, true]),
    ("/output/2/id", [false, true, true]),
    ("/output/2/call_id", [true, true, true]),
    ("/output/2/name", [true, true, true]),
    ("/output/2/arguments", [true, true, true]),
    ("/output/2/status", [true, true, true]),
    ("/usage", [false, false, true]),
    ("/usage/input_tokens", [true, true, true]),
    ("/usage/input_tokens_details", [false, false, true]),
    (
        "/usage/input_tokens_details/cached_tokens",
        [false, false, true],
    ),
    ("/usage/output_tokens", [true, true, true]),
    ("/usage/output_tokens_details", [false, false, true]),
    (
        "/usage/output_tokens_details/reasoning_tokens",
        [true, true, true],
    ),
    ("/usage/total_tokens", [true, true, true]),
    ("/status", [true, true, true]),
    ("/error", [false, false, true]),
    ("/incomplete_details", [false, false, true]),
    ("/instructions", [false, false, true]),
    ("/max_output_tokens", [false, false, true]),
    ("/tools", [false, true, true]),
    ("/tools/0/type", [true, true, true]),
    ("/tools/0/name", [false, true, true]),
    // A null description reads as absent (Q9, pinned in
    // `streaming/reader/tests.rs`).
    ("/tools/0/description", [false, false, true]),
];

/// The official OpenAI dialect.
pub fn openai(model: &str) -> Responses {
    Responses::new(OpenAIConfig::new("test-key"), model)
}

rig_history_conformance::history_conformance_suite! {
    wire: "openai_responses",
    fixture: ResponsesHistory {
        wire: openai,
        model: "gpt-5.4",
        other_model: "gpt-5.2",
        text_only_model: Some("o3-mini"),
        reads_indexless: false,
        buffered: false,
        calls_at_terminal: true,
        accepting: |wire| wire.with_streamed_incomplete(IncompleteTerminal::Accept),
    },
}
