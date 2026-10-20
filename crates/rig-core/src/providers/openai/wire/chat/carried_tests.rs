//! Fork Chat contracts migrated onto the actual upstream wire and driver.

use super::*;
use crate::message::{AssistantMessage, DocumentData};
use crate::message::{
    CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart, CustomToolCall, ProviderItem,
    ToolName, UnreplayableProviderItem, UnrepresentableOpaqueContent, UnrepresentableToolCall,
};
use crate::providers::openai::wire::{Dialect, GROQ, OPENAI};
use crate::test_utils::history::decode;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient, json_body};
use crate::wire::Operation;
use bytes::Bytes;
use futures::StreamExt;

fn wire() -> Chat {
    OpenAIConfig::new("sk-test")
        .with_dialect(&OPENAI)
        .chat("gpt-4.1-nano")
}
fn prompt(text: &str) -> CompletionRequest {
    CompletionRequest::new(text).temperature(0.0).max_tokens(16)
}
fn canonical(choice: &[AssistantContent]) -> Vec<AssistantContent> {
    choice.iter().map(AssistantContent::canonical).collect()
}
fn empty_turn_body(finish: Option<&str>) -> String {
    json!({"id":"chatcmpl-empty", "object":"chat.completion", "created":0,
      "model":"gpt-4.1-nano", "choices":[{"index":0,"finish_reason":finish,
      "message":{"role":"assistant","content":""}}],
      "usage":{"prompt_tokens":900,"completion_tokens":32,"total_tokens":932}})
    .to_string()
}

fn history(block: AssistantContent) -> CompletionRequest {
    let mut request = prompt("next");
    request.chat_history = vec![
        Message::user("first"),
        Message::Assistant(AssistantMessage {
            content: vec![block],
            ..Default::default()
        }),
        Message::user("next"),
    ];
    request
}

fn whole_frame(message: Value, finish: &str) -> WireFrame {
    WireFrame::Text(
        json!({"object":"chat.completion", "id":"c", "model":"anthropic/claude-test",
        "choices":[{"index":0,"message":message,"finish_reason":finish}]})
        .to_string(),
    )
}

#[test]
fn the_mode_decides_the_reply_framing() {
    assert_eq!(
        wire()
            .encode(prompt("hi"), Mode::Unary)
            .expect("encodes")
            .framing,
        Framing::Whole
    );
    assert_eq!(
        wire()
            .encode(prompt("hi"), Mode::Streaming)
            .expect("encodes")
            .framing,
        Framing::Sse
    );
}

#[test]
fn the_output_cap_spelling_follows_the_model_family() {
    for (model, key, absent) in [
        ("gpt-5.2", "max_completion_tokens", "max_tokens"),
        ("o3-mini", "max_completion_tokens", "max_tokens"),
        ("gpt-4.1-nano", "max_tokens", "max_completion_tokens"),
    ] {
        let body = json_body(
            &OpenAIConfig::new("k")
                .chat(model)
                .encode(prompt("hi"), Mode::Unary)
                .expect("encodes")
                .request,
        );
        assert!(body.get(key).is_some(), "{model}: {body}");
        assert!(body.get(absent).is_none(), "{model}: {body}");
    }
    let body = json_body(
        &OpenAIConfig::with_key(&GROQ, "k")
            .chat("gpt-5.2")
            .encode(prompt("hi"), Mode::Unary)
            .expect("Groq encodes")
            .request,
    );
    assert!(body.get("max_tokens").is_some());
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn openrouter_refuses_a_document_that_is_only_a_file_id() {
    let request = || {
        let mut request = prompt("read this");
        request.chat_history = vec![Message::User {
            content: vec![UserContent::Document(crate::message::Document {
                data: DocumentData::File(Source::FileId("file-abc".to_owned())),
                ..Default::default()
            })],
        }];
        request
    };
    let error = OpenAIConfig::with_key(&super::super::OPENROUTER, "k")
        .chat("openai/gpt-4o")
        .encode(request(), Mode::Unary)
        .expect_err("direct encoding refuses a provider file id");
    assert!(
        error
            .to_string()
            .contains("Provider file IDs are not supported for OpenRouter document inputs")
    );
}

#[test]
fn openai_accepts_a_document_that_is_only_a_file_id() {
    let request = || {
        let mut request = prompt("read this");
        request.chat_history = vec![Message::User {
            content: vec![UserContent::Document(crate::message::Document {
                data: DocumentData::File(Source::FileId("file-abc".to_owned())),
                ..Default::default()
            })],
        }];
        request
    };
    assert!(wire().encode(request(), Mode::Unary).is_ok());
}

#[tokio::test]
async fn a_truncated_stream_yields_no_terminal_record() {
    let bytes=Bytes::from_static(b"data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n");
    let model = crate::driver::Model::new(wire(), MockStreamingClient { sse_bytes: bytes });
    let mut stream = model.stream(prompt("hi")).expect("stream");
    while stream.next().await.is_some() {}
    assert!(
        matches!(stream.finish().await, Err(ProviderError::Truncated)),
        "the canonical U driver reports truncation without manufacturing a response"
    );
}

#[test]
fn groq_native_tools_preserve_functions_and_compound_options() {
    let mut request = prompt("search");
    request.tools.push(crate::completion::ToolDefinition::new(
        ToolName::new("lookup").expect("name"),
        "lookup",
        json!({"type":"object"}),
    ));
    request.additional_params = Some(json!({"tools":[{"type":"browser_search"},
        {"type":"browser_search"},{"type":"code_interpreter"}],
        "compound_custom":{"enabled_tools":[{"type":"browser_search","held":true}],"limit":3}}));
    let body = json_body(
        &OpenAIConfig::with_key(&GROQ, "k")
            .chat("compound")
            .encode(request, Mode::Unary)
            .expect("encodes")
            .request,
    );
    assert_eq!(body["tools"].as_array().expect("functions").len(), 1);
    assert_eq!(body["tools"][0]["function"]["name"], "lookup");
    assert_eq!(
        body["compound_custom"],
        json!({"enabled_tools":[
        {"type":"browser_search","held":true},{"type":"code_interpreter"}],"limit":3})
    );
}

#[test]
fn groq_native_tools_refuse_a_nonarray() {
    for raw in [Value::Null, json!({"type":"browser_search"})] {
        let mut request = prompt("search");
        request.additional_params = Some(json!({"tools":raw}));
        assert!(
            OpenAIConfig::with_key(&GROQ, "k")
                .chat("compound")
                .encode(request, Mode::Unary)
                .is_err()
        );
    }
}

#[test]
fn hyperbolic_removes_tool_remnants_and_retains_image_arrays() {
    let image = json!({"type":"image_url","image_url":{"url":"https://x.invalid/a.png"}});
    let mut body = json!({"messages":[{"role":"tool","content":"answer"},
        {"role":"assistant","content":null,"tool_calls":[{}]},
        {"role":"user","content":[{"type":"text","text":"look"},image]},
        {"role":"assistant","content":"answer","tool_calls":[{}],"reasoning_content":"private"}]});
    rewrite_body(BodyRewrite::Hyperbolic, body.as_object_mut().expect("map")).expect("rewrite");
    assert_eq!(body["messages"].as_array().expect("messages").len(), 2);
    assert_eq!(body["messages"][0]["content"][1], image);
    assert!(body["messages"][1].get("tool_calls").is_none());
    assert!(body["messages"][1].get("reasoning_content").is_none());
}

#[test]
fn openrouter_cache_marks_only_the_last_system_part() {
    let mut request = prompt("next");
    request.chat_history.insert(
        0,
        Message::System {
            content: "rules".to_owned(),
        },
    );
    let chat = OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("openai/gpt-4o");
    let plain = json_body(
        &chat
            .encode(request.clone(), Mode::Unary)
            .expect("plain")
            .request,
    );
    let cached = json_body(
        &chat
            .with_prompt_caching()
            .encode(request, Mode::Unary)
            .expect("cached")
            .request,
    );
    assert!(
        plain["messages"][0]["content"][0]
            .get("cache_control")
            .is_none()
    );
    assert_eq!(
        cached["messages"][0]["content"][0]["cache_control"],
        json!({"type":"ephemeral"})
    );
    assert!(
        cached["messages"][1]["content"][0]
            .get("cache_control")
            .is_none()
    );
}

#[test]
fn explicit_custom_calls_are_typed_json_only_refusals() {
    let call = CustomToolCall::new(
        CallId::from_wire("c"),
        ToolName::new("grep").expect("name"),
        "{ raw input",
    );
    let error = wire()
        .encode(history(AssistantContent::CustomToolCall(call)), Mode::Unary)
        .expect_err("refused");
    let error: ProviderError = error.into();
    assert!(matches!(error,ProviderError::Request(ref error)
        if matches!(error.downcast_ref::<UnrepresentableToolCall>(),Some(UnrepresentableToolCall::CustomCall{..}))));
}

#[test]
fn qualified_functions_keep_the_exact_namespace_in_the_refusal() {
    let call = ToolCall::new(
        CallId::from_wire("c"),
        crate::message::ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
    )
    .with_namespace(Some(String::new()));
    let error: ProviderError = wire()
        .encode(history(AssistantContent::ToolCall(call)), Mode::Unary)
        .expect_err("qualified refusal")
        .into();
    assert!(
        matches!(error,ProviderError::Request(ref error) if matches!(
        error.downcast_ref::<UnrepresentableToolCall>(),Some(UnrepresentableToolCall::Namespace{namespace,..}) if namespace.is_empty()))
    );
}

#[test]
fn calls_and_results_use_original_coordinates_after_message_splitting() {
    let local = ToolCall::new(
        CallId::from_wire(""),
        crate::message::ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
    );
    let held = ToolCall::new(
        CallId::from_wire("tool-0"),
        crate::message::ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
    );
    let mut request = prompt("next");
    request.chat_history = vec![
        Message::user("first"),
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::reasoning("plan"),
                AssistantContent::ToolCall(local.clone()),
            ],
            ..Default::default()
        }),
        Message::User {
            content: vec![
                UserContent::text("before"),
                UserContent::ToolResult(local.result(vec![ToolResultContent::text("one")])),
                UserContent::text("after"),
            ],
        },
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(local.clone())],
            ..Default::default()
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                local.result(vec![ToolResultContent::text("two")]),
            )],
        },
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(held)],
            ..Default::default()
        }),
    ];
    let body = json_body(
        &wire()
            .encode(request, Mode::Unary)
            .expect("occurrence plan")
            .request,
    );
    let messages = body["messages"].as_array().expect("messages");
    let calls: Vec<_> = messages
        .iter()
        .filter_map(|message| message.pointer("/tool_calls/0/id").and_then(Value::as_str))
        .collect();
    let results: Vec<_> = messages
        .iter()
        .filter_map(|message| message.get("tool_call_id").and_then(Value::as_str))
        .collect();
    assert_eq!(
        calls,
        ["tool-1", "tool-2", "tool-0"],
        "each completed local occurrence has its own alias"
    );
    assert_eq!(
        results,
        ["tool-1", "tool-2"],
        "split result messages use their original content coordinates"
    );
}

#[tokio::test]
async fn a_corrupt_body_that_is_not_utf8_keeps_its_exact_bytes() {
    let body: &[u8] = b"{\"object\":\"chat.completion\",\"choices\":7,\"note\":\"\xff\"}";
    let error =
        crate::driver::Model::new(wire(), RecordingHttpClient::new(Bytes::from_static(body)))
            .call(prompt("hello"))
            .await
            .expect_err("malformed body");
    let ProviderError::CorruptFrame(corrupt) = &error else {
        panic!("corrupt frame: {error:?}");
    };
    assert_eq!(corrupt.event_type(), Some("chat.completion"));
    assert_eq!(corrupt.frame(), None);
    assert_eq!(corrupt.frame_bytes(), Some(body));
    assert_eq!(
        corrupt.evidence(),
        &crate::error::FrameEvidence::Bytes(body.to_vec())
    );
}

#[test]
fn carried_provider_items_are_refused_for_every_issuer_before_adaptation() {
    for provider in [Some("openai"), Some("foreign"), None] {
        let item = ProviderItem {
            payload: CarriedPayload::new(
                json!({"type":"web_search_call","detail":7}),
                provider.map(str::to_owned),
            ),
        };
        let request = history(AssistantContent::ProviderItem(item));
        let error = wire()
            .encode(request.clone(), Mode::Unary)
            .expect_err("direct refusal");
        assert!(
            error
                .to_string()
                .contains("cannot replay the `web_search_call` item"),
            "{provider:?}: {error}"
        );
        let error =
            Completion::prepare(request, &wire().describe()).expect_err("preflight refusal");
        assert!(
            matches!(error,ProviderError::Request(ref error) if error.downcast_ref::<UnreplayableProviderItem>().is_some())
        );
    }
}

#[test]
fn opaque_nested_reasoning_is_refused_before_adaptation() {
    let block = CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
            json!({"type":"future","x":1}),
            None,
        ))],
        CarriedPayload::new(json!({}), None),
    );
    let error = Completion::prepare(
        history(AssistantContent::CarriedReasoning(block)),
        &wire().describe(),
    )
    .expect_err("nested refusal");
    assert!(
        matches!(error,ProviderError::Request(ref error) if error.downcast_ref::<UnrepresentableOpaqueContent>().is_some())
    );
}

#[test]
fn openrouter_gateway_details_keep_distinct_blocks_signatures_and_upstream_issuer() {
    let chat = OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("anthropic/claude-test");
    let details = json!([{"type":"reasoning.text","id":"r1","text":"first","signature":"sig"},
        {"type":"reasoning.summary","id":"r2","summary":"second"},
        {"type":"reasoning.encrypted","id":"r3","data":"cipher"}]);
    let reply = decode(
        &chat,
        Mode::Unary,
        [whole_frame(
            json!({"role":"assistant","reasoning":"duplicate",
        "reasoning_details":details,"content":"answer"}),
            "stop",
        )],
    )
    .expect("decodes");
    let blocks: Vec<_> = reply
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::CarriedReasoning(block) => Some(block),
            _ => None,
        })
        .collect();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].id.as_deref(), Some("r1"));
    assert_eq!(blocks[0].provider(), Some("anthropic"));
    assert_eq!(
        blocks[0].content,
        vec![CarriedReasoningPart::Text {
            text: "first".to_owned(),
            signature: Some("sig".to_owned())
        }]
    );
    assert_eq!(blocks[2].payload.item, details[2]);
    assert!(
        !reply
            .choice
            .iter()
            .any(|block| matches!(block, AssistantContent::Reasoning(_)))
    );
    assert_eq!(
        reply.choice.last().map(AssistantContent::canonical),
        Some(AssistantContent::text("answer"))
    );
}

#[test]
fn openrouter_carried_reasoning_replays_every_part_with_ordered_indices() {
    let chat = OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("anthropic/claude-test");
    let block = CarriedReasoning::new(
        vec![
            CarriedReasoningPart::Summary("a".to_owned()),
            CarriedReasoningPart::Text {
                text: "b".to_owned(),
                signature: Some("sig".to_owned()),
            },
            CarriedReasoningPart::Encrypted("c".to_owned()),
        ],
        CarriedPayload::new(json!({}), Some("anthropic".to_owned())),
    )
    .with_id(Some("r".to_owned()));
    let body = json_body(
        &chat
            .encode(
                history(AssistantContent::CarriedReasoning(block)),
                Mode::Unary,
            )
            .expect("encodes")
            .request,
    );
    assert_eq!(
        body["messages"][1]["reasoning_details"],
        json!([
        {"type":"reasoning.summary","id":"r","format":null,"index":0,"summary":"a"},
        {"type":"reasoning.text","id":"r","format":null,"index":1,"text":"b","signature":"sig"},
        {"type":"reasoning.encrypted","id":"r","format":null,"index":2,"data":"c"}])
    );
}

#[test]
fn openrouter_signed_plaintext_survives_canonicalizing_and_editing_before_replay() {
    let chat = OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("anthropic/claude-test");
    let message = json!({"role":"assistant","reasoning":"thinking","content":"answer",
        "reasoning_details":[{"type":"reasoning.text","text":"","signature":"sig-f","index":0}]});
    let reply = decode(&chat, Mode::Unary, [whole_frame(message.clone(), "stop")])
        .expect("signed plaintext");
    let mut block = reply.choice.first().expect("reasoning first").canonical();
    let AssistantContent::CarriedReasoning(reasoning) = &mut block else {
        panic!("carried signed plaintext: {block:?}");
    };
    assert_eq!(reasoning.payload.item, message);
    assert_eq!(reasoning.provider(), Some("anthropic"));
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "thinking".to_owned(),
            signature: Some("sig-f".to_owned())
        }]
    );
    let CarriedReasoningPart::Text { text, .. } = &mut reasoning.content[0] else {
        panic!("text");
    };
    *text = "edited".to_owned();
    let body = json_body(
        &chat
            .encode(history(block), Mode::Unary)
            .expect("replay")
            .request,
    );
    assert_eq!(
        body["messages"][1]["reasoning_details"][0]["text"],
        "edited"
    );
    assert_eq!(
        body["messages"][1]["reasoning_details"][0]["signature"],
        "sig-f"
    );
}

#[tokio::test]
async fn openrouter_late_signature_finishes_the_original_part_without_repeating_text() {
    use crate::streaming::{Item, PartKind, StreamEvent};
    let deltas = [
        json!({"reasoning":"thinking"}),
        json!({"content":"answer"}),
        json!({"reasoning_details":[{"type":"reasoning.text","text":"","signature":"sig-late","index":0}]}),
        json!({}),
    ];
    let mut bytes = String::new();
    for (index, delta) in deltas.iter().enumerate() {
        let finish = if index == 3 {
            json!("stop")
        } else {
            Value::Null
        };
        let frame = json!({"object":"chat.completion.chunk","model":"anthropic/claude-test",
            "choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
        bytes.push_str(&format!("data: {frame}\n\n"));
    }
    bytes.push_str("data: [DONE]\n\n");
    let model = crate::driver::Model::new(
        OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("anthropic/claude-test"),
        MockStreamingClient {
            sse_bytes: Bytes::from(bytes),
        },
    );
    let mut stream = model.stream(prompt("think")).expect("stream");
    let mut starts = Vec::new();
    let mut deltas = Vec::new();
    let mut ends = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("event") {
            Item::Event(StreamEvent::Start {
                part,
                kind: PartKind::Reasoning,
                ..
            }) => starts.push(part),
            Item::Event(StreamEvent::Reasoning { part, text }) => deltas.push((part, text)),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(block),
                reasoning_authority: _,
            }) => ends.push((part, block)),
            _ => {}
        }
    }
    assert_eq!(starts.len(), 1);
    assert_eq!(deltas, vec![(starts[0], "thinking".to_owned())]);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0, starts[0]);
    assert_eq!(
        ends[0].1.content,
        vec![CarriedReasoningPart::Text {
            text: "thinking".to_owned(),
            signature: Some("sig-late".to_owned())
        }]
    );
    let response = stream.finish().await.expect("terminal");
    assert!(matches!(
        response.choice.first(),
        Some(AssistantContent::CarriedReasoning(_))
    ));
    assert_eq!(response.choice.len(), 2);
}

#[test]
fn unknown_finish_is_exact_and_wire_terminal_stays_successful() {
    let reply = decode(
        &wire(),
        Mode::Unary,
        [whole_frame(
            json!({"role":"assistant","content":"answer"}),
            "bespoke_reason",
        )],
    )
    .expect("wire succeeds");
    assert_eq!(
        reply.finish_reason(),
        Some(FinishReason::Other("bespoke_reason".to_owned()))
    );
    // U's new caller-facing stop classification is deliberately unchanged.
    assert!(!reply.choice.is_empty());
}

#[test]
fn an_explicit_empty_reasoning_field_keeps_the_fork_precedence() {
    let reply = decode(
        &wire(),
        Mode::Streaming,
        [
            WireFrame::Text(
                json!({"object":"chat.completion.chunk",
        "choices":[{"index":0,"delta":{"reasoning_content":"","reasoning":"alternate",
            "content":"answer"},"finish_reason":"stop"}]})
                .to_string(),
            ),
            WireFrame::Text("[DONE]".to_owned()),
        ],
    )
    .expect("turn");
    assert_eq!(canonical(&reply.choice), [AssistantContent::text("answer")]);
}

#[test]
fn projected_usage_is_exact_in_payload_order_before_the_verdict() {
    use crate::observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::with_capacity(32));
    let context = AdapterContext::new(log.clone(), Subject::default(), "chat-usage");
    let request = http::Request::post("https://x.invalid/chat/completions")
        .body(())
        .expect("request");
    let mut attempt = context
        .attempt_for(&request, "/chat/completions")
        .expect("attempt");
    for payload in [
        json!({"usage":{"prompt_tokens":5,"completion_tokens":null,"total_tokens":8,"prompt_tokens_details":{}}}),
        json!({"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"reasoning_tokens":2}},
            "choices":[{"finish_reason":"stop"}]}),
    ] {
        attempt.project(|sink| {
            ChatDecoder::project(&serde_json::to_vec(&payload).expect("json"), sink)
        });
    }
    let events: Vec<_> = log
        .trace()
        .observations
        .into_iter()
        .filter_map(|fact| match fact.action {
            Action::Adapter { observation } => Some(observation.event),
            _ => None,
        })
        .collect();
    let snapshots: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AdapterEvent::Usage { usage } => Some(usage),
            _ => None,
        })
        .collect();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].input_tokens, Some(5));
    assert_eq!(snapshots[0].output_tokens, None);
    assert_eq!(snapshots[0].cached_input_tokens, None);
    assert_eq!(snapshots[1].reasoning_tokens, Some(2));
    assert_eq!(snapshots[1].total_tokens, Some(8));
    let usage_at = events
        .iter()
        .rposition(|event| matches!(event, AdapterEvent::Usage { .. }))
        .expect("usage");
    let verdict_at = events
        .iter()
        .position(|event| matches!(event, AdapterEvent::Provider { .. }))
        .expect("verdict");
    assert!(usage_at < verdict_at);
    assert_eq!(
        wire().describe().usage_observation,
        UsageObservation::PayloadSnapshots
    );
}

#[test]
fn the_fork_observed_payload_shape_gates_every_projection() {
    use crate::observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject};
    let events = |payload: Value| {
        let log = std::sync::Arc::new(ObservationLog::with_capacity(16));
        let context = AdapterContext::new(log.clone(), Subject::default(), "chat-shape");
        let request = http::Request::post("https://x.invalid/chat/completions")
            .body(())
            .expect("request");
        let mut attempt = context
            .attempt_for(&request, "/chat/completions")
            .expect("attempt");
        attempt.project(|sink| {
            ChatDecoder::project(&serde_json::to_vec(&payload).expect("json"), sink)
        });
        log.trace()
            .observations
            .into_iter()
            .filter_map(|fact| match fact.action {
                Action::Adapter { observation }
                    if matches!(
                        observation.event,
                        AdapterEvent::Usage { .. }
                            | AdapterEvent::Provider { .. }
                            | AdapterEvent::ErrorEnvelope { .. }
                    ) =>
                {
                    Some(observation.event)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let valid = json!({"id":"reply","model":"actual", "usage":{"prompt_tokens":5},
        "choices":[{"finish_reason":"stop"}],
        "error":{"code":429,"type":"rate_limit","message":"retry"}});
    for (field, malformed) in [
        ("choices", Value::Null),
        ("choices", json!(7)),
        ("choices", json!([{"finish_reason":7}])),
        ("id", json!({})),
        ("model", json!(false)),
        ("usage", json!("five")),
        (
            "usage",
            json!({"prompt_tokens":5,"prompt_tokens_details":7}),
        ),
        (
            "usage",
            json!({"prompt_tokens":5,"completion_tokens_details":"details"}),
        ),
        ("error", json!("retry")),
        ("error", json!({"message":false})),
        ("error", json!({"type":[]})),
        ("error", json!({"status":7})),
        ("error", json!({"type":"one","status":"two"})),
    ] {
        let mut payload = valid.clone();
        payload[field] = malformed;
        assert!(
            events(payload.clone()).is_empty(),
            "whole projection is gated: {payload}"
        );
    }
    for choices in [None, Some(json!([]))] {
        let mut payload = json!({"usage":{"prompt_tokens":5}});
        if let Some(choices) = choices {
            payload["choices"] = choices;
        }
        let observed = events(payload);
        assert!(
            matches!(observed.as_slice(), [AdapterEvent::Usage { usage }] if usage.input_tokens == Some(5))
        );
    }
    let observed = events(json!({"usage":{"prompt_tokens":"5","completion_tokens":-1,
        "total_tokens":0,"prompt_tokens_details":{"cached_tokens":{}},
        "completion_tokens_details":{"reasoning_tokens":null}}}));
    assert!(
        matches!(observed.as_slice(), [AdapterEvent::Usage { usage }]
        if usage.input_tokens.is_none() && usage.output_tokens.is_none()
        && usage.total_tokens == Some(0) && usage.cached_input_tokens.is_none()
        && usage.reasoning_tokens.is_none())
    );
    let observed = events(json!({"choices":[
        {"index":1,"usage":{"prompt_tokens":99}},
        {"index":0,"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}
    ]}));
    assert!(
        matches!(observed.as_slice(), [AdapterEvent::Usage { usage }]
        if usage.input_tokens == Some(7) && usage.output_tokens == Some(2)
        && usage.total_tokens == Some(9))
    );
    let observed = events(json!({"usage":{"prompt_tokens":5},
        "choices":[{"usage":{"prompt_tokens":99}}]}));
    assert!(
        matches!(observed.as_slice(), [AdapterEvent::Usage { usage }] if usage.input_tokens == Some(5))
    );
    let observed = events(valid);
    assert!(matches!(
        observed.as_slice(),
        [
            AdapterEvent::Usage { .. },
            AdapterEvent::Provider { .. },
            AdapterEvent::ErrorEnvelope { .. }
        ]
    ));
}

#[tokio::test]
async fn complete_single_chunk_calls_close_before_the_stream_terminal() {
    use crate::streaming::{Item, StreamEvent};
    for dialect in [&GROQ, &super::super::DEEPSEEK, &super::super::MISTRAL] {
        let frame = json!({"object":"chat.completion.chunk","choices":[{"index":0,"delta":{
            "tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]});
        let body = format!("data: {frame}\n\n");
        let model = crate::driver::Model::new(
            OpenAIConfig::with_key(dialect, "k").chat("m"),
            MockStreamingClient {
                sse_bytes: Bytes::from(body),
            },
        );
        let mut stream = model.stream(prompt("lookup")).expect("stream");
        let mut closed = false;
        while let Some(item) = stream.next().await {
            if matches!(
                item,
                Ok(Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                }))
            ) {
                closed = true;
                break;
            }
        }
        assert!(
            closed,
            "{}: the complete call closes before EOF",
            dialect.name
        );
        assert!(
            stream.finish().await.is_err(),
            "EOF did not manufacture a terminal reply"
        );
    }
}

#[tokio::test]
async fn a_single_chunk_probe_closes_only_the_call_in_that_fragment() {
    use crate::streaming::{Item, StreamEvent};
    let frames = [
        json!({"tool_calls":[{"index":0,"id":"fragmented","type":"function","function":{"name":"lookup"}}]}),
        json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}),
        json!({"tool_calls":[{"index":1,"id":"whole","type":"function","function":{"name":"lookup","arguments":"{}"}}]}),
    ];
    let body: String = frames
        .into_iter()
        .map(|delta| {
            let frame =
                json!({"object":"chat.completion.chunk","choices":[{"index":0,"delta":delta}]});
            format!("data: {frame}\n\n")
        })
        .collect();
    let model = crate::driver::Model::new(
        OpenAIConfig::with_key(&GROQ, "k").chat("m"),
        MockStreamingClient {
            sse_bytes: Bytes::from(body),
        },
    );
    let mut stream = model.stream(prompt("lookup")).expect("stream");
    let first_closed = loop {
        match stream.next().await {
            Some(Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            }))) => break call,
            Some(Ok(_)) => {}
            other => panic!("a whole call closes before EOF: {other:?}"),
        }
    };
    assert_eq!(
        first_closed.id.provider().map(|id| id.as_str()),
        Some("whole")
    );
    assert!(
        stream.finish().await.is_err(),
        "EOF is not a provider terminal"
    );
}

#[tokio::test]
async fn length_discards_only_unusable_function_calls_on_their_original_parts() {
    use crate::streaming::{DiscardReason, Item, PartKind, StreamEvent, Transcript};
    for raw in ["", r#"{"note":"unfinished"#] {
        let frame = json!({"object":"chat.completion.chunk","id":"cut-turn","model":"gpt-4.1-nano",
            "choices":[{"index":0,"delta":{"content":"beside the calls","tool_calls":[
                {"index":0,"id":"cut","type":"function","function":{"name":"lookup","arguments":raw}},
                {"index":1,"id":"whole","type":"function","function":{"name":"lookup","arguments":"{\"note\":\"kept\"}"}}]},
                "finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}});
        let model = crate::driver::Model::new(
            wire(),
            MockStreamingClient {
                sse_bytes: Bytes::from(format!("data: {frame}\n\ndata: [DONE]\n\n")),
            },
        );
        let mut stream = model.stream(prompt("lookup")).expect("stream");
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item.expect("event"));
        }
        let starts: Vec<_> = items
            .iter()
            .filter_map(|item| {
                if let Item::Event(StreamEvent::Start {
                    part,
                    kind: PartKind::ToolCall,
                    ..
                }) = item
                {
                    Some(*part)
                } else {
                    None
                }
            })
            .collect();
        let discarded: Vec<_> = items
            .iter()
            .filter_map(|item| {
                if let Item::Event(StreamEvent::Discard { part, reason }) = item {
                    Some((*part, *reason))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(starts.len(), 2, "{raw:?}");
        assert_eq!(
            discarded,
            vec![(starts[0], DiscardReason::Length)],
            "{raw:?}"
        );
        let arguments: Vec<_> = items
            .iter()
            .filter_map(|item| {
                if let Item::Event(StreamEvent::Arguments { part, json }) = item {
                    Some((*part, json.as_str()))
                } else {
                    None
                }
            })
            .filter(|(part, _)| *part == starts[0])
            .collect();
        assert_eq!(
            arguments,
            if raw.is_empty() {
                vec![]
            } else {
                vec![(starts[0], raw)]
            },
            "fragments survive"
        );
        assert!(
            !items.iter().any(|item| matches!(item,
            Item::Event(StreamEvent::End{part,..}) if *part==starts[0])),
            "no fabricated completed cut call"
        );
        Transcript::parse(serde_json::to_value(&items).expect("items"))
            .expect("discard closes the original ToolCall part");
        let response = stream
            .finish()
            .await
            .expect("explicit Length is a successful wire terminal");
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1, "{raw:?}");
        assert_eq!(calls[0].id.provider().map(|id| id.as_str()), Some("whole"));
        assert_eq!(
            calls[0].function.arguments,
            json!({"note":"kept"}).as_object().expect("object").clone()
        );
        assert!(
            calls[0].native.is_some(),
            "the completed U-native call retains its item"
        );
        assert!(
            response
                .choice
                .iter()
                .any(|block| block.canonical() == AssistantContent::text("beside the calls"))
        );
        assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        assert_eq!(response.usage.total_tokens, Some(7));
    }
}

#[test]
fn length_keeps_the_original_upstream_custom_call_path() {
    let frame = WireFrame::Text(
        json!({"object":"chat.completion.chunk","choices":[{
        "index":0,"delta":{"tool_calls":[{"index":0,"id":"custom","type":"custom",
            "custom":{"name":"grep","input":"{ raw"}}]},"finish_reason":"length"}]})
        .to_string(),
    );
    let response = decode(
        &wire(),
        Mode::Streaming,
        [frame, WireFrame::Text("[DONE]".to_owned())],
    )
    .expect("U native custom input is not a function argument cut");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("legacy U custom call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({"input":"{ raw"}));
    assert_eq!(call.native.as_ref().expect("native").item["type"], "custom");
}

#[tokio::test]
async fn completed_malformed_streamed_input_keeps_its_actual_identity_and_raw_diagnostic() {
    use crate::error::{ErrorDetail, ErrorKind};
    use crate::streaming::{Item, StreamEvent};
    let raw = r#"{"note":"unfinished"#;
    for finish in ["tool_calls", "stop"] {
        let frame = json!({"object":"chat.completion.chunk","choices":[{"index":0,
            "delta":{"tool_calls":[{"index":0,"id":"bad-provider-id","type":"function",
                "function":{"name":"lookup","arguments":raw}}]},"finish_reason":finish}]});
        let model = crate::driver::Model::new(
            wire(),
            MockStreamingClient {
                sse_bytes: Bytes::from(format!("data: {frame}\n\ndata: [DONE]\n\n")),
            },
        );
        let mut stream = model.stream(prompt("lookup")).expect("stream");
        let mut events = Vec::new();
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => events.push(item),
                Err(error) => errors.push(error),
            }
        }
        let [error] = errors.as_slice() else {
            panic!("one typed defect: {errors:?}");
        };
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::Response);
        let Some(ErrorDetail::MalformedToolInput(detail)) = report.detail else {
            panic!("typed malformed tool input: {report:?}");
        };
        assert_eq!(detail.id, CallId::from_wire("bad-provider-id"));
        assert_eq!(
            detail
                .provider
                .as_ref()
                .expect("provider evidence")
                .call_id
                .as_str(),
            "bad-provider-id"
        );
        assert_eq!(
            detail.provider.as_ref().expect("provider evidence").item_id,
            None
        );
        assert_eq!(detail.name, "lookup");
        assert_eq!(detail.raw, raw);
        assert!(!detail.error.is_empty());
        assert!(
            events.iter().any(
                |item| matches!(item,Item::Event(StreamEvent::Arguments{json,..}) if json==raw)
            ),
            "the received prefix was emitted before the error"
        );
        assert!(
            !events.iter().any(|item| matches!(
                item,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                }) | Item::Event(StreamEvent::Discard { .. })
            )),
            "a malformed ordinary finish fabricates no closure"
        );
        let final_error = stream
            .finish()
            .await
            .expect_err("the wire terminal is a defect");
        assert_eq!(final_error.report(), error.report());
    }
}

#[tokio::test]
async fn delivered_function_calls_flush_before_a_terminal_error_without_native_proof() {
    use crate::streaming::{Item, StreamEvent};
    for provider_error in [false, true] {
        let mut calls = vec![json!({"index":0,"id":"delivered","type":"function",
            "function":{"name":"lookup","arguments":"{}"}})];
        if provider_error {
            calls.push(json!({"index":1,"id":"unfinished","type":"function",
                "function":{"name":"lookup","arguments":"{\"q\":"}}));
        }
        let frame = json!({"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":calls}}]});
        let mut bytes = format!("data: {frame}\n\n");
        if provider_error {
            bytes.push_str(
                "data: {\"error\":{\"message\":\"blocked\",\"type\":\"provider_fault\"}}\n\n",
            );
        }
        let model = crate::driver::Model::new(
            wire(),
            MockStreamingClient {
                sse_bytes: Bytes::from(bytes),
            },
        );
        let mut stream = model.stream(prompt("lookup")).expect("stream");
        let mut ended = Vec::new();
        let error = loop {
            match stream.next().await {
                Some(Ok(Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(call),
                    ..
                }))) => ended.push(call),
                Some(Ok(_)) => {}
                Some(Err(error)) => break error,
                None => panic!("an unfinished/error reply must fail"),
            }
        };
        let [call] = ended.as_slice() else {
            panic!("one complete call precedes the first error: {ended:?}");
        };
        assert_eq!(call.id.provider().map(|id| id.as_str()), Some("delivered"));
        assert!(
            call.native.is_none(),
            "syntax-complete input did not invent a provider assertion"
        );
        if provider_error {
            assert!(
                matches!(error, ProviderError::ProviderResponse(_)),
                "the original provider envelope stays terminal"
            );
        } else {
            assert!(
                matches!(error, ProviderError::Truncated),
                "EOF did not invent a terminal"
            );
        }
    }
}

#[test]
fn paired_identity_is_a_typed_encode_refusal_before_body_creation() {
    use super::super::{Dialect, Identity, MissingCallerIdentity};
    let mut dialect = Dialect::gateway("requires-identity", "https://x.invalid/v1", "KEY");
    dialect.quirks.identity = Some(Identity {
        originator_env: "TEST_ORIGINATOR",
        user_agent_env: "TEST_USER_AGENT",
        session_ids: false,
    });
    let chat = OpenAIConfig::with_key(&dialect, "k").chat("m");
    let error: ProviderError = chat
        .encode(prompt("hi"), Mode::Unary)
        .expect_err("identity missing")
        .into();
    assert!(
        matches!(error,ProviderError::Request(ref error) if matches!(error.downcast_ref::<MissingCallerIdentity>(),Some(MissingCallerIdentity::Unset{dialect:"requires-identity"})))
    );
}

#[test]
fn budget_cut_drops_empty_arguments_but_retains_compound_defects() {
    let mut frame = json!({"object":"chat.completion","choices":[{"index":0,"finish_reason":"length",
        "message":{"role":"assistant","content":"text","tool_calls":[
            {"id":"empty","type":"function","function":{"name":"f","arguments":""}},
            {"id":"whole","type":"function","function":{"name":"f","arguments":"{}"}}]}}]});
    let source = frame.to_string();
    carried::prepare_whole_calls(&mut frame, &source, &Quirks::openai()).expect("budget tolerance");
    assert_eq!(
        frame["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("calls")
            .len(),
        1
    );
    assert_eq!(
        frame["choices"][0]["message"]["tool_calls"][0]["id"],
        "whole"
    );
    let mut bad = json!({"object":"chat.completion","choices":[{"finish_reason":"length","message":{
        "tool_calls":[{"id":"bad","type":"not_a_real_tool_type","function":{"name":"f","arguments":"{\"x\":"}}]}}]});
    let source = bad.to_string();
    let error = carried::prepare_whole_calls(&mut bad, &source, &Quirks::openai())
        .expect_err("compound defect");
    assert_eq!(error.frame(), Some(source.as_str()));
}

#[tokio::test]
async fn a_unary_reply_with_reasoning_folds_like_the_stream_of_the_same_turn() {
    use crate::message::Reasoning;

    const UNARY: &str = concat!(
        r#"{"object":"chat.completion","id":"chatcmpl-1","model":"m","#,
        r#""choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","#,
        r#""reasoning_content":"let me think","content":"the answer"}}],"#,
        r#""usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#,
    );
    const STREAM: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"the answer\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n",
        "data: [DONE]\n\n",
    );

    let buffered = crate::driver::Model::new(wire(), RecordingHttpClient::new(UNARY))
        .call(prompt("think then answer"))
        .await
        .expect("the unary reply decodes without tripping the sequence law");

    let streaming = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(STREAM.as_bytes()),
        },
    );
    let mut response = streaming
        .stream(prompt("think then answer"))
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    // Reasoning first, then the visible text — one emitter, one order.
    let reasoning_text = |choice: &[AssistantContent]| match choice.first() {
        Some(AssistantContent::Reasoning(Reasoning { text, .. })) => Some(format!("{text:?}")),
        _ => None,
    };
    assert!(
        reasoning_text(&buffered.choice).is_some(),
        "the unary reply's reasoning is a reasoning block: {:?}",
        buffered.choice
    );
    assert_eq!(
        reasoning_text(&buffered.choice),
        reasoning_text(&streamed.choice),
        "the two shapes of one turn produce the same reasoning block"
    );
    assert_eq!(canonical(&buffered.choice), canonical(&streamed.choice));
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(
        buffered.choice.last().map(AssistantContent::canonical),
        Some(AssistantContent::text("the answer"))
    );
}

#[tokio::test]
async fn a_dialect_that_streams_a_message_per_chunk_is_still_streaming() {
    use crate::providers::openai::wire::PERPLEXITY;

    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},",
        "\"message\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"pong\"},",
        "\"message\":{\"role\":\"assistant\",\"content\":\"pong\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.done\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},",
        "\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\",\"content\":\"pong\"}}],",
        "\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1,\"total_tokens\":8}}\n\n",
    );

    let wire = OpenAIConfig::new("pplx")
        .with_dialect(&PERPLEXITY)
        .chat("sonar");
    let mut response = crate::driver::Model::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    )
    .stream(prompt("ping"))
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    // The whole turn, not just its first frame.
    assert_eq!(
        folded.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("pong"))
    );
    assert_eq!(folded.usage.input_tokens, Some(7));
    assert_eq!(folded.usage.total_tokens, Some(8));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Stop));
    // The body's own id, never the transport request id.
    assert_eq!(folded.response_id(), Some("78385058-uuid"));

    // The terminator's envelope is the one the record carries: last frame
    // wins, which is what `AdditionalParams::merge` does for a scalar.
    assert_eq!(folded.raw["id"], "78385058-uuid");
    assert_eq!(folded.raw["usage"]["total_tokens"], 8);
    // The final provider object tag survives without normalization.
    assert_eq!(folded.raw["object"], "chat.completion.done");
}

#[tokio::test]
async fn a_tool_call_cut_mid_arguments_drops_only_itself() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-cut","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","#,
        r#""content":"noting both","tool_calls":["#,
        r#"{"id":"call_whole","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"first\"}"}},"#,
        r#"{"id":"call_cut","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"The"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let folded = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record two notes."))
        .await
        .expect(
            "a body cut mid-arguments must still decode: erroring discards the turn's \
             usage, id, finish reason and every other part over one unusable fragment",
        );

    let calls: Vec<_> = folded
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    let [call] = calls.as_slice() else {
        panic!(
            "exactly the complete call survives — a half-parsed one must never reach \
             the caller: {:?}",
            folded.choice
        );
    };
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_whole"));
    assert_eq!(call.function.name, "record");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"note": "first"})
    );

    // The rest of the turn, which erroring would have thrown away.
    assert!(
        folded
            .choice
            .iter()
            .any(|item| item.canonical() == AssistantContent::text("noting both")),
        "the text delivered beside the calls survives: {:?}",
        folded.choice
    );
    assert_eq!(folded.usage.output_tokens, Some(20));
    assert_eq!(folded.usage.input_tokens, Some(165));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Length));
}

#[tokio::test]
async fn malformed_arguments_on_a_completed_tool_turn_stay_a_decode_error() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-bad","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","#,
        r#""content":null,"tool_calls":["#,
        r#"{"id":"call_bad","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"The"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let error = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record a note."))
        .await
        .expect_err("a completed tool-call turn with malformed arguments is a defect");
    assert!(
        matches!(&error, ProviderError::CorruptFrame(corrupt) if corrupt.frame() == Some(BODY)),
        "the malformed payload must stay loud rather than be dropped: {error:?}"
    );
}

#[tokio::test]
async fn valid_arguments_survive_a_length_truncated_turn() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-odd","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","#,
        r#""content":null,"tool_calls":["#,
        r#"{"id":"call_odd","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"unexpected\": 1}"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let folded = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record a note."))
        .await
        .expect("valid arguments decode");

    let Some(AssistantContent::ToolCall(call)) = folded.choice.first() else {
        panic!(
            "a parseable call is kept even under a truncated turn: {:?}",
            folded.choice
        );
    };
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_odd"));
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"unexpected": 1})
    );
    assert_eq!(folded.finish_reason(), Some(FinishReason::Length));
}

#[tokio::test]
async fn an_empty_turn_the_provider_cut_short_keeps_its_reason_and_usage() {
    for (reason, expected) in [
        ("length", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
    ] {
        let response = crate::driver::Model::new(
            wire(),
            RecordingHttpClient::new(empty_turn_body(Some(reason))),
        )
        .call(prompt("ask"))
        .await
        .unwrap_or_else(|error| panic!("`{reason}` is a cut-short turn, not a defect: {error}"));

        assert!(
            response.choice.is_empty(),
            "{reason}: {:?}",
            response.choice
        );
        assert_eq!(response.finish_reason(), Some(expected), "{reason}");
        // Raising would have thrown these away, which is the whole reason
        // the cut-short turn is kept.
        assert_eq!(response.usage.input_tokens, Some(900), "{reason}");
        assert_eq!(response.usage.output_tokens, Some(32), "{reason}");
        assert_eq!(response.usage.total_tokens, Some(932), "{reason}");
    }
}

#[tokio::test]
async fn an_empty_turn_that_ran_to_completion_is_a_provider_defect() {
    for reason in [
        Some("stop"),
        Some("tool_calls"),
        Some("bespoke_reason"),
        None,
    ] {
        let folded =
            crate::driver::Model::new(wire(), RecordingHttpClient::new(empty_turn_body(reason)))
                .call(prompt("ask"))
                .await;

        let Err(ProviderError::Response(message)) = &folded else {
            panic!("`{reason:?}` does not license an empty turn: {folded:?}");
        };
        assert_eq!(message, crate::message::EMPTY_RESPONSE_ERROR, "{reason:?}");
    }
}

#[test]
fn the_mistral_body_rebuilds_content_as_its_own_chunks() {
    use crate::message::{Document, DocumentMediaType, Image, Message, UserContent};
    use crate::providers::openai::wire::MISTRAL;

    let encode = |content: Vec<UserContent>| {
        let mut request = prompt("look at this");
        request.chat_history = vec![Message::User { content }];
        OpenAIConfig::new("k")
            .with_dialect(&MISTRAL)
            .chat("mistral-small-latest")
            .encode(request, Mode::Unary)
            .map(|encoded| {
                let [http_request] = std::slice::from_ref(&encoded.request) else {
                    panic!("one request")
                };
                let crate::wire::Body::Bytes(bytes) = http_request.body() else {
                    panic!("bytes")
                };
                serde_json::from_slice::<serde_json::Value>(bytes).expect("JSON")
            })
    };

    // A lone text part is the plain string the shared encoder sends.
    let body = encode(vec![UserContent::text("just words")]).expect("encodes");
    assert_eq!(body["messages"][0]["content"], "just words");

    // A text-only array stays an array of text chunks, as upstream sends it,
    // rather than being flattened into one string.
    let body = encode(vec![
        UserContent::text("first"),
        UserContent::text("second"),
    ])
    .expect("encodes");
    assert_eq!(
        body["messages"][0]["content"],
        json!([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}])
    );

    // An image rides as Mistral's own `image_url` chunk, in an array \u2014 not
    // flattened away.
    let body = encode(vec![
        UserContent::text("describe"),
        UserContent::Image(Image {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.png".to_owned()),
            media_type: None,
            detail: None,
            ..Default::default()
        }),
    ])
    .expect("encodes");
    let parts = body["messages"][0]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content is a chunk array: {}", body["messages"][0]));
    assert_eq!(parts.len(), 2, "the image survives: {parts:?}");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");

    // A part of a type Mistral has no chunk for is forwarded verbatim, as
    // upstream forwards it, rather than refused or removed.
    let body = encode(vec![
        UserContent::text("watch"),
        UserContent::Video(crate::message::Video {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.mp4".to_owned()),
            media_type: None,
            ..Default::default()
        }),
    ])
    .expect("an unknown part type is forwarded, not refused");
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "watch"},
            {"type": "video_url", "video_url": {"url": "https://x.invalid/a.mp4"}}
        ])
    );

    // A document with inline bytes becomes `document_url`, carrying its name
    // in Mistral's own optional field.
    let body = encode(vec![UserContent::Document(Document {
        data: DocumentData::File(crate::message::DocumentSourceKind::Base64(
            "ZGF0YQ==".to_owned(),
        )),
        media_type: Some(DocumentMediaType::PDF),
        ..Default::default()
    })])
    .expect("encodes");
    let parts = body["messages"][0]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content is a chunk array: {}", body["messages"][0]));
    assert_eq!(parts[0]["type"], "document_url");
    assert!(
        parts[0].get("document_url").is_some(),
        "the payload is under Mistral's own key: {parts:?}"
    );

    // And no other dialect rebuilds content this way.
    let mut request = prompt("describe");
    request.chat_history = vec![Message::User {
        content: vec![UserContent::Image(Image {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.png".to_owned()),
            media_type: None,
            detail: None,
            ..Default::default()
        })],
    }];
    let openai = wire().encode(request, Mode::Unary).expect("encodes");
    let [http_request] = std::slice::from_ref(&openai.request) else {
        panic!("one request")
    };
    let crate::wire::Body::Bytes(bytes) = http_request.body() else {
        panic!("bytes")
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert_eq!(body["messages"][0]["content"][0]["type"], "image_url");
    assert!(
        body["messages"][0]["content"][0]["image_url"]["url"].is_string(),
        "OpenAI keeps its own nesting: {body}"
    );
    assert_eq!(
        body["messages"][0]["content"][0]["image_url"]["detail"], "auto",
        "the carried OpenRouter detail rewrite does not change OpenAI"
    );
}

#[test]
fn groq_replays_reasoning_turns_without_reasoning_content() {
    use crate::message::{Message, Reasoning};
    use crate::providers::openai::wire::DEEPSEEK;

    let history = || {
        let mut request = prompt("and then?");
        request.chat_history = vec![
            Message::user("think first"),
            Message::Assistant(AssistantMessage {
                content: vec![
                    AssistantContent::Reasoning(Reasoning::new("private chain")),
                    AssistantContent::text("visible answer"),
                ],
                ..Default::default()
            }),
            Message::user("and then?"),
        ];
        request
    };
    let assistant = |dialect: &Dialect| {
        let encoded = OpenAIConfig::new("k")
            .with_dialect(dialect)
            .chat("m")
            .encode(history(), Mode::Unary)
            .expect("encodes");
        let [request] = std::slice::from_ref(&encoded.request) else {
            panic!("a chat turn is one request");
        };
        let crate::wire::Body::Bytes(bytes) = request.body() else {
            panic!("a chat request body is bytes");
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        body["messages"][1].clone()
    };

    let groq = assistant(&GROQ);
    assert_eq!(groq["role"], "assistant");
    assert_eq!(groq["content"][0]["text"], "visible answer");
    assert!(groq.get("reasoning_content").is_none(), "{groq}");

    let deepseek = assistant(&DEEPSEEK);
    assert_eq!(deepseek["reasoning_content"], "private chain");
}

#[derive(Clone, Default)]
struct RotatingSource {
    current: std::sync::Arc<std::sync::Mutex<Option<crate::wire::Credential>>>,
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl RotatingSource {
    fn holding(credential: crate::wire::Credential) -> Self {
        let source = Self::default();
        source.rotate(credential);
        source
    }

    fn rotate(&self, credential: crate::wire::Credential) {
        *self.current.lock().expect("unpoisoned") = Some(credential);
    }

    fn reads(&self) -> usize {
        self.reads.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl crate::wire::CredentialSource for RotatingSource {
    fn current(
        &self,
    ) -> crate::wasm_compat::WasmBoxedFuture<
        '_,
        Result<crate::wire::Credential, crate::wire::CredentialSourceError>,
    > {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let current = self.current.lock().expect("unpoisoned").clone();
        Box::pin(async move { current.ok_or_else(|| "the token owner is offline".into()) })
    }
}

const CHAT_REPLY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

fn header<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .map(|value| value.to_str().expect("ascii header"))
}

/// Each request reads the source afresh and sends what it holds then, in
/// place of the static key: a rotation between two requests reaches the
/// second. The source is authoritative for the account: a credential naming
/// none removes the configuration's static account header.
#[tokio::test]
async fn a_credential_source_is_read_for_every_request_and_replaces_the_static_key() {
    let source = RotatingSource::holding(
        crate::wire::Credential::new("token-1").with_account_id("account-1"),
    );
    let provider = OpenAIConfig::new("static-key")
        .with_account_id("static-account")
        .with_credential_source(source.clone());
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let bound = crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone());

    bound
        .call(prompt("one"))
        .await
        .expect("the first reply decodes");
    source.rotate(crate::wire::Credential::new("token-2"));
    bound
        .call(prompt("two"))
        .await
        .expect("the second reply decodes");

    assert_eq!(source.reads(), 2, "one read per request");
    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        header(&requests[0].headers, "authorization"),
        Some("Bearer token-1")
    );
    assert_eq!(
        header(&requests[0].headers, "chatgpt-account-id"),
        Some("account-1")
    );
    assert_eq!(
        header(&requests[1].headers, "authorization"),
        Some("Bearer token-2")
    );
    assert_eq!(
        header(&requests[1].headers, "chatgpt-account-id"),
        None,
        "a credential naming no account sends none, whatever is configured"
    );
    for request in &requests {
        assert_eq!(request.headers.get_all("authorization").iter().count(), 1);
    }
}

/// An `api-key` dialect gets the source's token in `api-key`, not a bearer.
#[tokio::test]
async fn a_credential_source_fills_the_api_key_header_its_auth_names() {
    let source = RotatingSource::holding(crate::wire::Credential::new("token-1"));
    let provider = OpenAIConfig::new("static-key")
        .with_auth(crate::providers::openai::wire::Auth::ApiKeyHeader)
        .with_credential_source(source.clone());
    let http = RecordingHttpClient::new(CHAT_REPLY);
    crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone())
        .call(prompt("one"))
        .await
        .expect("the reply decodes");

    let requests = http.requests();
    assert_eq!(header(&requests[0].headers, "api-key"), Some("token-1"));
    assert_eq!(header(&requests[0].headers, "authorization"), None);
}

/// A source that cannot supply a credential refuses the request by name,
/// keeping its own error whole, and nothing reaches the transport.
#[tokio::test]
async fn a_failing_credential_source_refuses_before_anything_is_sent() {
    let provider =
        OpenAIConfig::new("static-key").with_credential_source(RotatingSource::default());
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let error = crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone())
        .call(prompt("one"))
        .await
        .expect_err("the source has no credential");

    assert!(http.requests().is_empty(), "nothing was sent");
    let crate::error::ProviderError::Request(inner) = &error else {
        panic!("expected a request refusal, got {error:?}");
    };
    let refusal = inner
        .downcast_ref::<crate::wire::CredentialUnavailable>()
        .unwrap_or_else(|| panic!("expected CredentialUnavailable, got {inner}"));
    assert!(refusal.to_string().contains("the token owner is offline"));
    assert_eq!(
        std::error::Error::source(refusal)
            .map(ToString::to_string)
            .as_deref(),
        Some("the token owner is offline")
    );
}

/// A streamed request reads the source once, when it is sent, and the
/// transport receives the source's credential.
#[tokio::test]
async fn a_streamed_request_reads_the_credential_source_when_it_is_sent() {
    let source = RotatingSource::holding(
        crate::wire::Credential::new("token-1").with_account_id("account-1"),
    );
    let provider = OpenAIConfig::new("static-key").with_credential_source(source.clone());
    let http = crate::test_utils::CapturingStreamingClient {
        sse_bytes: Bytes::from_static(b"data: [DONE]\n\n"),
        ..Default::default()
    };
    let mut response = crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone())
        .stream(prompt("one"))
        .expect("the stream opens");
    while response.next().await.is_some() {}

    assert_eq!(source.reads(), 1);
    let requests = http.requests.lock().expect("unpoisoned").clone();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        header(&requests[0], "authorization"),
        Some("Bearer token-1")
    );
    assert_eq!(
        header(&requests[0], "chatgpt-account-id"),
        Some("account-1")
    );
}

/// A failing source refuses a streamed request exactly as it refuses a unary
/// one: the same named refusal, reported the same way, and nothing sent.
#[tokio::test]
async fn a_failing_credential_source_refuses_a_stream_as_it_refuses_a_unary_request() {
    let provider =
        OpenAIConfig::new("static-key").with_credential_source(RotatingSource::default());
    let unary = crate::driver::Model::new(
        Chat::new(provider.clone(), "gpt-4o"),
        RecordingHttpClient::new(CHAT_REPLY),
    )
    .call(prompt("one"))
    .await
    .expect_err("the source has no credential");

    let http = crate::test_utils::CapturingStreamingClient::default();
    let mut response = crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone())
        .stream(prompt("one"))
        .expect("the stream opens");
    let mut errors = Vec::new();
    while let Some(item) = response.next().await {
        if let Err(error) = item {
            errors.push(error);
        }
    }

    assert!(
        http.requests.lock().expect("unpoisoned").is_empty(),
        "nothing was sent"
    );
    let [streamed] = errors.as_slice() else {
        panic!("expected exactly one refusal, got {errors:?}");
    };
    assert_eq!(streamed.to_string(), unary.to_string());
}

/// A credential a source stamps at send time gets exactly the treatment a
/// statically configured key gets in observation: the same scenario, once
/// with each, where the provider's error echoes the credential back, records
/// the same error envelope, and neither credential appears in the trace.
#[tokio::test]
async fn a_source_credential_is_kept_out_of_observed_diagnostics_as_a_static_key_is() {
    let mut envelopes = Vec::new();
    let cases = [
        (
            "static",
            OpenAIConfig::new("static-secret-key"),
            "static-secret-key",
        ),
        (
            "source",
            OpenAIConfig::new("static-key").with_credential_source(RotatingSource::holding(
                crate::wire::Credential::new("source-secret-token"),
            )),
            "source-secret-token",
        ),
    ];
    for (label, provider, token) in cases {
        let http = RecordingHttpClient::new(CHAT_REPLY);
        http.set_response(crate::test_utils::MockHttpResponse::ErrorResponse(
            http::StatusCode::UNAUTHORIZED,
            Bytes::from(format!(
                r#"{{"error":{{"message":"invalid token {token}","type":"invalid_request_error"}}}}"#
            )),
        ));
        let log = std::sync::Arc::new(crate::observe::ObservationLog::with_capacity(64));
        let context = crate::observe::AdapterContext::new(
            log.clone(),
            crate::observe::Subject::default(),
            "credential-test".to_owned(),
        );
        let _ = crate::driver::Model::new(Chat::new(provider, "gpt-4o"), http.clone())
            .call_observed(prompt("one"), context)
            .await;

        let bearer = format!("Bearer {token}");
        assert_eq!(
            header(&http.requests()[0].headers, "authorization"),
            Some(bearer.as_str()),
            "{label}: the credential reached the transport"
        );
        let trace = format!("{:?}", log.trace());
        assert!(!trace.contains(token), "{label}: {trace}");
        let envelope = trace
            .find("ErrorEnvelope")
            .map(|start| {
                let rest = &trace[start..];
                rest[..rest.find('}').map_or(rest.len(), |end| end + 1)].to_owned()
            })
            .unwrap_or_else(|| {
                panic!("{label}: the provider's error envelope was observed: {trace}")
            });
        envelopes.push(envelope);
    }
    assert_eq!(
        envelopes[0], envelopes[1],
        "a source credential and a static key are observed alike"
    );
}

#[test]
fn presealed_groq_native_tools_keep_the_full_merged_body_in_both_modes() {
    let chat = OpenAIConfig::with_key(&GROQ, "k").chat("compound");
    let mut request = CompletionRequest::new("canonical prompt");
    request.tools.push(crate::completion::ToolDefinition::new(
        ToolName::new("lookup").expect("name"),
        "lookup",
        json!({"type":"object"}),
    ));
    request.additional_params = Some(json!({
        "messages":[
            {"role":"assistant","content":"before","reasoning_content":"drop"},
            {"role":"user","content":"raw search"}
        ],
        "tools":[{"type":"browser_search"},{"type":"browser_search"},
            {"type":"code_interpreter"}],
        "compound_custom":{"enabled_tools":[{"type":"browser_search","held":true}],
            "limit":3},
        "raw_marker":"kept",
        "stream_options":{"include_usage":false,"raw_option":7}
    }));
    for mode in [Mode::Unary, Mode::Streaming] {
        let encoded = chat.encode(request.clone(), mode).expect("encodes");
        let mut expected = json!({
            "model":"compound",
            "messages":[{"role":"assistant","content":"before"},
                {"role":"user","content":"raw search"}],
            "tools":[{"type":"function","function":{"name":"lookup",
                "description":"lookup","parameters":{"type":"object"}}}],
            "compound_custom":{"enabled_tools":[
                {"type":"browser_search","held":true},{"type":"code_interpreter"}],
                "limit":3},
            "raw_marker":"kept",
            "stream_options":{"include_usage":false,"raw_option":7}
        });
        if mode == Mode::Streaming {
            expected["stream"] = json!(true);
        }
        assert_eq!(json_body(&encoded.request), expected, "{mode:?}");
        assert_eq!(
            encoded.framing,
            match mode {
                Mode::Unary => Framing::Whole,
                Mode::Streaming => Framing::Sse,
            }
        );
        assert_eq!(encoded.route, Some("/chat/completions"));
    }
}

#[test]
fn presealed_groq_native_tools_keep_absent_empty_and_null_distinct() {
    let chat = OpenAIConfig::with_key(&GROQ, "k").chat("compound");
    for mode in [Mode::Unary, Mode::Streaming] {
        for (raw, compound) in [
            (json!({}), None),
            (json!({"tools":[]}), Some(json!({"enabled_tools":[]}))),
            (
                json!({"tools":[{"type":"browser_search"}]}),
                Some(json!({"enabled_tools":[{"type":"browser_search"}]})),
            ),
        ] {
            let mut request = CompletionRequest::new("search");
            request.additional_params = Some(raw);
            let mut expected = json!({"model":"compound",
                "messages":[{"role":"user","content":"search"}]});
            if let Some(compound) = compound {
                expected["compound_custom"] = compound;
            }
            if mode == Mode::Streaming {
                expected["stream"] = json!(true);
                expected["stream_options"] = json!({"include_usage":true});
            }
            let encoded = chat.encode(request, mode).expect("encodes");
            assert_eq!(json_body(&encoded.request), expected, "{mode:?}");
        }
        let mut request = CompletionRequest::new("search");
        request.additional_params = Some(json!({"tools":null}));
        let error = chat
            .encode(request, mode)
            .expect_err("explicit null is refused");
        assert_eq!(
            error.to_string(),
            "RequestError: Groq `additional_params.tools` must be an array of native tool objects"
        );
        assert!(error.unsupported_option().is_none());
    }
}

#[test]
fn presealed_openrouter_cache_marks_the_raw_system_override_in_both_modes() {
    let chat = OpenAIConfig::with_key(&super::super::OPENROUTER, "k").chat("openai/gpt-4o");
    for (system, cached_system) in [
        (
            json!("raw rules"),
            json!([{"type":"text","text":"raw rules",
            "cache_control":{"type":"ephemeral"}}]),
        ),
        (
            json!([
                {"type":"text","text":"first","cache_control":{"type":"preserved"}},
                {"type":"text","text":"last","cache_control":{"type":"long","ttl":"1h"}}
            ]),
            json!([
                {"type":"text","text":"first","cache_control":{"type":"preserved"}},
                {"type":"text","text":"last","cache_control":{"type":"ephemeral"}}
            ]),
        ),
    ] {
        for caching in [false, true] {
            for mode in [Mode::Unary, Mode::Streaming] {
                let mut request = CompletionRequest::new("canonical prompt");
                request.chat_history.insert(
                    0,
                    Message::System {
                        content: "canonical rules".to_owned(),
                    },
                );
                let raw_messages = json!([
                    {"role":"system","content":system},
                    {"role":"system","content":"second system"},
                    {"role":"assistant","content":"before","reasoning_content":"plan"},
                    {"role":"user","content":[{"type":"text","text":"raw user"},
                        {"type":"image_url","image_url":{"url":"https://x.invalid/a.png",
                            "detail":"high"}}]}
                ]);
                request.additional_params = Some(json!({"messages":raw_messages,
                    "raw_marker":"kept"}));
                let selected = if caching {
                    chat.clone().with_prompt_caching()
                } else {
                    chat.clone()
                };
                let encoded = selected.encode(request, mode).expect("encodes");
                let mut expected = json!({"model":"openai/gpt-4o", "messages":[
                    {"role":"system","content":if caching { &cached_system } else { &system }},
                    {"role":"system","content":"second system"},
                    {"role":"assistant","content":"before","reasoning":"plan"},
                    {"role":"user","content":[{"type":"text","text":"raw user"},
                        {"type":"image_url","image_url":{"url":"https://x.invalid/a.png"}}]}
                ], "raw_marker":"kept"});
                if mode == Mode::Streaming {
                    expected["stream"] = json!(true);
                }
                assert_eq!(json_body(&encoded.request), expected, "{caching}/{mode:?}");
                assert_eq!(
                    encoded.framing,
                    match mode {
                        Mode::Unary => Framing::Whole,
                        Mode::Streaming => Framing::Sse,
                    }
                );
            }
        }
    }
}

#[test]
fn presealed_openrouter_cache_flag_is_limited_to_its_dialect_and_system_content() {
    let router = OpenAIConfig::with_key(&super::super::OPENROUTER, "k")
        .chat("openai/gpt-4o")
        .with_prompt_caching();
    for (chat, messages) in [
        (
            router.clone(),
            json!([{"role":"user","content":"no system"}]),
        ),
        (
            router.clone(),
            json!([{"role":"system","content":null},
            {"role":"user","content":"null system"}]),
        ),
        (
            router,
            json!([{"role":"system","content":[]},
            {"role":"user","content":"empty system"}]),
        ),
    ] {
        assert_presealed_chat_cache_flag_body(chat, messages);
    }
}

#[test]
fn presealed_openai_cache_flag_is_limited_to_its_dialect_and_system_content() {
    let openai = OpenAIConfig::new("k").chat("gpt-4o").with_prompt_caching();
    assert_presealed_chat_cache_flag_body(
        openai,
        json!([{"role":"system","content":"plain rules"},
        {"role":"user","content":"openai"}]),
    );
}

fn assert_presealed_chat_cache_flag_body(chat: Chat, messages: Value) {
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("canonical prompt");
        request.additional_params = Some(json!({"messages":messages}));
        let encoded = chat.encode(request, mode).expect("encodes");
        let mut expected = json!({"model":chat.model,"messages":messages});
        if mode == Mode::Streaming {
            expected["stream"] = json!(true);
            if chat.provider.dialect.quirks.stream_include_usage {
                expected["stream_options"] = json!({"include_usage":true});
            }
        }
        assert_eq!(json_body(&encoded.request), expected, "{mode:?}");
    }
}

#[test]
fn presealed_chat_direct_headers_are_built_once_and_keep_the_same_final_body() {
    assert_direct_headers_preserve_final_body(OpenAIConfig::with_key(&GROQ, "k").chat("compound"));
}

#[test]
fn openrouter_presealed_chat_direct_headers_are_built_once_and_keep_the_same_final_body() {
    assert_direct_headers_preserve_final_body(
        OpenAIConfig::with_key(&super::super::OPENROUTER, "k")
            .chat("openai/gpt-4o")
            .with_prompt_caching(),
    );
}

fn assert_direct_headers_preserve_final_body(chat: Chat) {
    let raw = json!({"messages":[{"role":"system","content":"rules"},
        {"role":"user","content":"search"}],
        "tools":[{"type":"browser_search"}],
        "compound_custom":{"limit":3}});
    let mut request = CompletionRequest::new("canonical prompt");
    request.additional_params = Some(raw.clone());
    for mode in [Mode::Unary, Mode::Streaming] {
        let calls = std::cell::Cell::new(0);
        let direct = chat
            .encode_with_headers(request.clone(), mode, |provider, original, builder| {
                calls.set(calls.get() + 1);
                assert_eq!(provider, &chat.provider);
                assert_eq!(original.additional_params.as_ref(), Some(&raw));
                Ok(provider
                    .completion_headers(original, builder)?
                    .header("x-g233-stage", "direct"))
            })
            .expect("direct headers encode");
        let ordinary = chat.encode(request.clone(), mode).expect("ordinary encode");
        assert_eq!(calls.get(), 1);
        assert_eq!(
            header(direct.request.headers(), "x-g233-stage"),
            Some("direct")
        );
        assert_eq!(header(ordinary.request.headers(), "x-g233-stage"), None);
        assert_eq!(
            header(direct.request.headers(), "authorization"),
            Some("Bearer k")
        );
        assert_eq!(
            header(direct.request.headers(), "content-type"),
            Some("application/json")
        );
        assert_eq!(direct.request.method(), ordinary.request.method());
        assert_eq!(direct.request.uri(), ordinary.request.uri());
        assert_eq!(direct.route, ordinary.route);
        assert_eq!(direct.framing, ordinary.framing);
        assert_eq!(direct.request_id_header, ordinary.request_id_header);
        assert_eq!(
            direct.response_header_prefix,
            ordinary.response_header_prefix
        );
        assert_eq!(json_body(&direct.request), json_body(&ordinary.request));
    }
}

#[test]
fn presealed_chat_catalog_policy_checks_the_effective_model_and_keeps_raw_overrides() {
    assert_catalog_policy_preserves_raw_overrides(
        OpenAIConfig::with_key(&GROQ, "k").chat("compound"),
    );
}

#[test]
fn openrouter_presealed_chat_catalog_policy_checks_the_effective_model_and_keeps_raw_overrides() {
    assert_catalog_policy_preserves_raw_overrides(
        OpenAIConfig::with_key(&super::super::OPENROUTER, "k")
            .chat("openai/gpt-4o")
            .with_prompt_caching(),
    );
}

fn assert_catalog_policy_preserves_raw_overrides(chat: Chat) {
    use crate::completion::OnUnsupported;
    let model = crate::providers::openai::GPT_6_SOL;
    let rules = json!([{"role":"system","content":"rules"},
        {"role":"user","content":"search"}]);
    for mode in [Mode::Unary, Mode::Streaming] {
        for raw_top_p in [false, true] {
            let mut raw = json!({"model":model,"messages":rules,
                "tools":[{"type":"browser_search"}],
                "compound_custom":{"limit":3}});
            if raw_top_p {
                raw["top_p"] = json!(0.7);
            }
            let mut request = CompletionRequest::new("canonical prompt").top_p(0.9);
            request.additional_params = Some(raw);
            let error = chat
                .encode(request.clone().on_unsupported(OnUnsupported::Error), mode)
                .expect_err("the effective model refuses sampling");
            let refusal = error.unsupported_option().expect("typed catalog refusal");
            assert_eq!(refusal.option, "top_p");
            let encoded = chat
                .encode(request.on_unsupported(OnUnsupported::Ignore), mode)
                .expect("Ignore encodes");
            let mut expected = json!({"model":model,"messages":rules,
                "compound_custom":{"limit":3}});
            if chat.provider.dialect.quirks.rewrite == BodyRewrite::GroqCompoundTools {
                expected["compound_custom"]["enabled_tools"] = json!([{"type":"browser_search"}]);
            } else {
                expected["tools"] = json!([{"type":"browser_search"}]);
                expected["messages"][0]["content"] = json!([
                    {"type":"text","text":"rules","cache_control":{"type":"ephemeral"}}
                ]);
            }
            if raw_top_p {
                expected["top_p"] = json!(0.7);
            }
            if mode == Mode::Streaming {
                expected["stream"] = json!(true);
                if chat.provider.dialect.quirks.stream_include_usage {
                    expected["stream_options"] = json!({"include_usage":true});
                }
            }
            assert_eq!(
                json_body(&encoded.request),
                expected,
                "{raw_top_p}/{mode:?}"
            );
        }
    }
}

#[test]
fn presealed_groq_null_tools_refuse_before_an_independent_catalog_defect() {
    use crate::completion::OnUnsupported;
    let chat = OpenAIConfig::with_key(&GROQ, "k").chat("compound");
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("search")
            .top_p(0.9)
            .on_unsupported(OnUnsupported::Error);
        request.additional_params = Some(json!({"model":crate::providers::openai::GPT_6_SOL,
            "tools":null}));
        let error = chat
            .encode(request, mode)
            .expect_err("null tools are refused preseal");
        assert_eq!(
            error.to_string(),
            "RequestError: Groq `additional_params.tools` must be an array of native tool objects"
        );
        assert!(error.unsupported_option().is_none());
    }
}
