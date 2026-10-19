//! Original native Ollama streaming grammar over recorded response bytes.

use std::collections::HashSet;

use futures::StreamExt;
use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, FinishReason, ToolDefinition};
use crate::driver::Model;
use crate::message::{
    AssistantContent, AssistantMessage, CallId, LocalCallId, Message, Reasoning, ToolCall,
    ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::providers::ollama::{OllamaConfig, chat::Chat};
use crate::streaming::{Item, Relayed, StreamEvent};
use crate::test_utils::raw_parity::recorded_reply;
use crate::test_utils::streaming_conformance::assert_valid_event_stream;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient, json_body};
use crate::wire::{Mode, Wire};

const MODEL: &str = "qwen3:4b";

fn native() -> Chat {
    OllamaConfig::new().native_completion(MODEL)
}

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the original tool name is nonempty")
}

fn alpha() -> ToolDefinition {
    ToolDefinition::new(
        name("lookup_harbor_label"),
        "Return the alpha signal marker.",
        json!({"type":"object","properties":{},"required":[]}),
    )
}

fn beta() -> ToolDefinition {
    ToolDefinition::new(
        name("lookup_orchard_label"),
        "Return the beta signal marker.",
        json!({"type":"object","properties":{},"required":[]}),
    )
}

fn adder() -> ToolDefinition {
    ToolDefinition::new(
        name("add"),
        "Add x and y together",
        json!({
            "type":"object",
            "properties": {
                "x":{"type":"number","description":"The first number to add"},
                "y":{"type":"number","description":"The second number to add"}
            },
            "required":["x","y"]
        }),
    )
}

struct StreamRun {
    text: String,
    reasoning_blocks: Vec<Reasoning>,
    reasoning_delta: String,
    tool_calls: Vec<ToolCall>,
    done_count: usize,
    choice: Vec<AssistantContent>,
    response: CompletionResponse,
}

async fn drain_stream(request: CompletionRequest, cassette: &str) -> StreamRun {
    let received = recorded_reply(cassette, 1).expect("the original recorded reply body");
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/x-ndjson"),
    );
    let model = Model::new(
        native(),
        SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(received, headers)]),
    );
    let mut stream = model
        .stream(request)
        .expect("stream should start")
        .into_relay();
    let mut text = String::new();
    let mut reasoning_blocks = Vec::new();
    let mut reasoning_delta = String::new();
    let mut tool_calls = Vec::new();
    let mut raw_items = Vec::new();
    let mut done_count = 0;
    let mut response = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be ok") {
            Relayed::Item(item) => {
                raw_items.push(Ok(item.clone()));
                match item {
                    Item::Event(StreamEvent::Text { text: fragment, .. }) => {
                        text.push_str(&fragment)
                    }
                    Item::Event(StreamEvent::Reasoning { text, .. }) => {
                        reasoning_delta.push_str(&text)
                    }
                    Item::Event(StreamEvent::End {
                        content: AssistantContent::Reasoning(reasoning),
                        ..
                    }) => reasoning_blocks.push(reasoning),
                    Item::Event(StreamEvent::End {
                        content: AssistantContent::ToolCall(call),
                        ..
                    }) => tool_calls.push(call),
                    _ => {}
                }
            }
            Relayed::Done(terminal) => {
                done_count += 1;
                response = Some(*terminal);
            }
            Relayed::Origin(_) => {}
        }
    }
    let response = response.expect("aggregated stream should retain the terminal record");
    let choice = response.choice.clone();
    assert_valid_event_stream(&raw_items, &choice);
    StreamRun {
        text,
        reasoning_blocks,
        reasoning_delta,
        tool_calls,
        done_count,
        choice,
        response,
    }
}

fn assert_terminal(run: &StreamRun, expected_finish: FinishReason) {
    assert_eq!(
        run.done_count, 1,
        "stream should yield exactly one terminal record"
    );
    assert_eq!(
        run.response.finish_reason(),
        Some(expected_finish),
        "unexpected finish reason"
    );
    assert!(
        run.response.usage.total_tokens.is_some_and(|n| n > 0),
        "terminal record should carry non-zero usage, got {:?}",
        run.response.usage
    );
}

fn assert_received_call(call: &ToolCall, received: Value) {
    let provider = call
        .id
        .provider()
        .expect("the daemon-issued call id must be preserved");
    let native = call
        .native
        .as_ref()
        .expect("the actual decoder retains the received call");
    assert_eq!(
        native.item, received,
        "the original whole received call remains native evidence"
    );
    assert_eq!(
        native.item["id"].as_str(),
        Some(provider.as_str()),
        "the durable id adopts the independently received daemon id"
    );
    assert!(
        call.function.invalid_arguments.is_none(),
        "arguments must assemble uncorrupted"
    );
    assert!(
        native.item["function"]["arguments"].is_object(),
        "the received arguments must be an object"
    );
    assert_eq!(
        call.function.arguments_value(),
        native.item["function"]["arguments"]
    );
    assert_eq!(
        call.function.name.as_str(),
        native.item["function"]["name"]
            .as_str()
            .expect("the received function name")
    );
}

fn assert_current_body(request: &CompletionRequest, expected: Value) -> Value {
    let encoded = native()
        .encode(request.clone(), Mode::Streaming)
        .expect("the original caller encodes natively");
    assert!(encoded.request.uri().path().ends_with("/api/chat"));
    let actual = json_body(&encoded.request);
    assert_eq!(actual, expected, "the whole current native request body");
    actual
}

fn expected_thinking_and_tool_call() -> Value {
    // G39 omits the original empty options object.
    json!({
        "messages": [
            {
                "content": "You must call the requested tool before writing any normal text. After the tool result is available, do not call any more tools and answer in one short sentence that includes the exact tool output.",
                "role": "system"
            },
            {
                "content": "Call `lookup_harbor_label` exactly once before answering. After the tool result is available, answer in one short sentence that includes the exact tool output.",
                "role": "user"
            }
        ],
        "model": "qwen3:4b",
        "stream": true,
        "think": true,
        "tools": [
            {
                "function": {
                    "description": "Return the alpha signal marker.",
                    "name": "lookup_harbor_label",
                    "parameters": {
                        "properties": {},
                        "required": [],
                        "type": "object"
                    }
                },
                "type": "function"
            }
        ]
    })
}

fn expected_parallel_tool_calls() -> Value {
    // G39 omits the original empty options object.
    json!({
        "messages": [
            {
                "content": "You are a precise assistant. When tools are available, you must use them instead of guessing. Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. Never call the same tool twice once you already have its result.",
                "role": "system"
            },
            {
                "content": "Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them together in this single reply, before writing any text. Emit the two tool calls in one turn — do not wait for results between them.",
                "role": "user"
            }
        ],
        "model": "qwen3:4b",
        "stream": true,
        "think": false,
        "tools": [
            {
                "function": {
                    "description": "Return the alpha signal marker.",
                    "name": "lookup_harbor_label",
                    "parameters": {
                        "properties": {},
                        "required": [],
                        "type": "object"
                    }
                },
                "type": "function"
            },
            {
                "function": {
                    "description": "Return the beta signal marker.",
                    "name": "lookup_orchard_label",
                    "parameters": {
                        "properties": {},
                        "required": [],
                        "type": "object"
                    }
                },
                "type": "function"
            }
        ]
    })
}

fn expected_same_tool_twice() -> Value {
    // G39 omits the original empty options object.
    json!({
        "messages": [
            {
                "content": "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself.",
                "role": "system"
            },
            {
                "content": "/no_think Use the `add` tool twice in this single reply, before any text: first add 2 and 3, then add 10 and 20. Emit both tool calls together in this one turn — do not wait for results between them, and do not compute the sums yourself.",
                "role": "user"
            }
        ],
        "model": "qwen3:4b",
        "stream": true,
        "think": false,
        "tools": [
            {
                "function": {
                    "description": "Add x and y together",
                    "name": "add",
                    "parameters": {
                        "properties": {
                            "x": {
                                "description": "The first number to add",
                                "type": "number"
                            },
                            "y": {
                                "description": "The second number to add",
                                "type": "number"
                            }
                        },
                        "required": [
                            "x",
                            "y"
                        ],
                        "type": "object"
                    }
                },
                "type": "function"
            }
        ]
    })
}

fn expected_chat_sourced_history_replay() -> Value {
    // G39 omits the original empty options object. G41 emits aliases; Ollama-2 omits the legacy call type key.
    json!({
        "messages": [
            {
                "content": "You are a calculator assistant. Report tool results faithfully.",
                "role": "system"
            },
            {
                "content": "/no_think Use the add tool to compute 2 + 3, then state the result.",
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": {
                                "x": 2,
                                "y": 3
                            },
                            "name": "add"
                        },
                        "id": "tool-0"
                    }
                ]
            },
            {
                "content": "5",
                "role": "tool",
                "tool_name": "add",
                "tool_call_id": "tool-0"
            },
            {
                "content": "/no_think State the final result in one short sentence.",
                "role": "user"
            }
        ],
        "model": "qwen3:4b",
        "stream": true,
        "think": false,
        "tools": [
            {
                "function": {
                    "description": "Add x and y together",
                    "name": "add",
                    "parameters": {
                        "properties": {
                            "x": {
                                "description": "The first number to add",
                                "type": "number"
                            },
                            "y": {
                                "description": "The second number to add",
                                "type": "number"
                            }
                        },
                        "required": [
                            "x",
                            "y"
                        ],
                        "type": "object"
                    }
                },
                "type": "function"
            }
        ]
    })
}

#[tokio::test]
async fn thinking_and_tool_call_in_one_stream_native_original() {
    let request = CompletionRequest::new("Call `lookup_harbor_label` exactly once before answering. After the tool result is available, answer in one short sentence that includes the exact tool output.")
        .preamble("You must call the requested tool before writing any normal text. After the tool result is available, do not call any more tools and answer in one short sentence that includes the exact tool output.").tool(alpha()).additional_params(json!({"think":true}));
    assert_current_body(&request, expected_thinking_and_tool_call());
    let run = drain_stream(request, include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../rig-cassette/fixtures/cassettes/ollama/streaming_grammar/thinking_and_tool_call_native_original.yaml"))).await;
    assert_terminal(&run, FinishReason::ToolCalls);
    assert!(
        !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
        "think:true should surface reasoning on the stream"
    );
    let streamed = run
        .tool_calls
        .iter()
        .find(|call| call.function.name.as_str() == "lookup_harbor_label")
        .expect("stream should yield the lookup_harbor_label call");
    assert!(
        run.choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(_))),
        "aggregated choice should keep the reasoning part, got {:?}",
        run.choice
    );
    let aggregated = run
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call)
                if call.function.name.as_str() == "lookup_harbor_label" =>
            {
                Some(call)
            }
            _ => None,
        })
        .expect("aggregated choice should keep the tool call");
    assert_eq!(aggregated.id, streamed.id, "id should aggregate unchanged");
    assert_received_call(
        streamed,
        json!({"id": "call_4p4xs3px", "function": {"index": 0, "name": "lookup_harbor_label", "arguments": {}}}),
    );
}

#[tokio::test]
async fn parallel_id_less_tool_calls_stay_distinct() {
    let request = CompletionRequest::new("Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them together in this single reply, before writing any text. Emit the two tool calls in one turn — do not wait for results between them.").preamble("You are a precise assistant. When tools are available, you must use them instead of guessing. Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. Never call the same tool twice once you already have its result.")
        .tool(alpha()).tool(beta()).additional_params(json!({"think":false}));
    assert_current_body(&request, expected_parallel_tool_calls());
    let run = drain_stream(
        request,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rig-cassette/fixtures/cassettes/ollama/streaming_grammar/parallel_tool_calls.yaml"
        )),
    )
    .await;
    assert_terminal(&run, FinishReason::ToolCalls);
    let aggregated: Vec<&ToolCall> = run
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    for (name, received) in [
        (
            "lookup_harbor_label",
            json!({"id": "call_dzvxo0hm", "function": {"index": 0, "name": "lookup_harbor_label", "arguments": {}}}),
        ),
        (
            "lookup_orchard_label",
            json!({"id": "call_0yo19v2u", "function": {"index": 1, "name": "lookup_orchard_label", "arguments": {}}}),
        ),
    ] {
        let streamed = run
            .tool_calls
            .iter()
            .find(|call| call.function.name.as_str() == name)
            .unwrap_or_else(|| panic!("stream should yield a {name} call"));
        let aggregated_call = aggregated
            .iter()
            .find(|call| call.function.name.as_str() == name)
            .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
        assert_eq!(
            aggregated_call.id, streamed.id,
            "{name} id should aggregate"
        );
        assert_received_call(streamed, received);
    }
    assert!(run.tool_calls.len() >= 2, "turn should stream both calls");
    assert_eq!(
        aggregated.len(),
        run.tool_calls.len(),
        "every streamed call should survive as its own aggregated part"
    );
    let distinct_ids: HashSet<&CallId> = run.tool_calls.iter().map(|call| &call.id).collect();
    assert_eq!(
        distinct_ids.len(),
        run.tool_calls.len(),
        "daemon-issued call ids must be preserved distinct"
    );
}

#[tokio::test]
async fn same_tool_called_twice_in_one_turn_stays_distinct() {
    let request = CompletionRequest::new("/no_think Use the `add` tool twice in this single reply, before any text: first add 2 and 3, then add 10 and 20. Emit both tool calls together in this one turn — do not wait for results between them, and do not compute the sums yourself.").preamble("You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself.")
        .tool(adder()).additional_params(json!({"think":false}));
    assert_current_body(&request, expected_same_tool_twice());
    let run = drain_stream(
        request,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rig-cassette/fixtures/cassettes/ollama/streaming_grammar/same_tool_twice.yaml"
        )),
    )
    .await;
    assert_terminal(&run, FinishReason::ToolCalls);
    let add_calls: Vec<&ToolCall> = run
        .tool_calls
        .iter()
        .filter(|call| call.function.name.as_str() == "add")
        .collect();
    assert!(
        add_calls.len() >= 2,
        "the turn should stream (at least) two `add` calls, got {:?}",
        run.tool_calls
    );
    let received = [
        json!({"id": "call_049zr2vz", "function": {"index": 0, "name": "add", "arguments": {"x": 2, "y": 3}}}),
        json!({"id": "call_56dg87tf", "function": {"index": 1, "name": "add", "arguments": {"x": 10, "y": 20}}}),
        json!({"id": "call_pnqmxufh", "function": {"index": 2, "name": "add", "arguments": {"x": 2, "y": 3}}}),
        json!({"id": "call_wio0ofrj", "function": {"index": 3, "name": "add", "arguments": {"x": 10, "y": 20}}}),
    ];
    assert_eq!(
        add_calls.len(),
        received.len(),
        "all four actual received calls survive"
    );
    for (call, received) in add_calls.iter().zip(received) {
        assert_received_call(call, received);
    }
    let argument_sets: HashSet<String> = add_calls
        .iter()
        .map(|call| call.function.arguments_value().to_string())
        .collect();
    assert!(
        argument_sets.len() >= 2,
        "the two same-name calls must keep distinct argument payloads, got {argument_sets:?}"
    );
    let distinct_ids: HashSet<&CallId> = add_calls.iter().map(|call| &call.id).collect();
    assert_eq!(
        distinct_ids.len(),
        add_calls.len(),
        "same-name calls must keep their distinct daemon-issued ids"
    );
    let aggregated_adds = run.choice.iter().filter(|content|
        matches!(content, AssistantContent::ToolCall(call) if call.function.name.as_str() == "add")).count();
    assert_eq!(
        aggregated_adds,
        add_calls.len(),
        "every streamed same-name call must survive as its own aggregated part"
    );
}

#[tokio::test]
async fn chat_sourced_history_replays_the_tool_name_not_the_identifier() {
    let call = ToolCall::new(
        CallId::from(LocalCallId::new()),
        ToolFunction::new(name("add"), json!({"x":2,"y":3})),
    );
    assert!(call.id.is_local());
    assert!(call.id.provider().is_none());
    assert!(call.native.is_none());
    let local_spelling = call.id.wire().into_owned();
    let result = call.result(vec![ToolResultContent::text("5")]);
    assert_eq!(result.call, call.id);
    let history = vec![
        Message::user("/no_think Use the add tool to compute 2 + 3, then state the result."),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(result)],
        },
    ];
    let request = CompletionRequest::new("/no_think State the final result in one short sentence.")
        .preamble("You are a calculator assistant. Report tool results faithfully.")
        .tool(adder())
        .messages(history)
        .additional_params(json!({"think":false}));
    let expected = expected_chat_sourced_history_replay();
    let actual = assert_current_body(&request, expected);
    assert_eq!(actual["messages"][2]["tool_calls"][0]["id"], "tool-0");
    assert_eq!(actual["messages"][3]["tool_name"], "add");
    assert_eq!(actual["messages"][3]["tool_call_id"], "tool-0");
    assert!(
        !actual.to_string().contains(&local_spelling),
        "the genuine local custody stays off the wire"
    );
    let run = drain_stream(request, include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../rig-cassette/fixtures/cassettes/ollama/streaming_grammar/chat_sourced_history_replay.yaml"))).await;
    assert!(
        run.text.contains('5'),
        "the model should answer from the replayed tool result, got {:?}",
        run.text
    );
}
