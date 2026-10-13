use super::super::completion::{CLAUDE_OPUS_4_8, CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6};
use super::*;
use crate::completion::CompletionRequest;
use crate::completion::Message as RigMessage;
use crate::completion::request::Document as RigDocument;
use crate::driver::{Decoded, decode_events};
use crate::message::{
    AssistantContent, DocumentRange, Opaque, Reasoning, Source, SourceLocation, StopReason,
};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::streaming::StreamEvent;
use crate::wire::Mode;

/// A fresh decoder labelled the way the [`Messages`](super::super::wire::Messages)
/// wire labels Anthropic proper.
fn adapter() -> MessagesDecoder {
    MessagesDecoder::new(false)
}

/// Classify an actual wire frame through the provider parser.
fn classified(frame: &str) -> MessagesEvent {
    let WireEvent::Known(event) = adapter().classify(WireFrame::Text(frame.into())) else {
        panic!("frame must classify Known: {frame}");
    };
    event
}

/// Decode known events through the shared output and EOF path.
fn decode(events: impl IntoIterator<Item = MessagesEvent>) -> Decoded<Completion> {
    decode_events!(MessagesDecoder::new(false), "anthropic", events)
}

/// The response the stream of `frames` folds into on the Anthropic wire.
fn streamed(frames: &[Value]) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
        .completion(super::super::completion::CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
}

/// The frames of one content block: its start, its deltas and its stop.
fn block(index: usize, start: Value, deltas: &[Value]) -> Vec<Value> {
    std::iter::once(json!({"type": "content_block_start", "index": index, "content_block": start}))
        .chain(
            deltas.iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ),
        )
        .chain([json!({"type": "content_block_stop", "index": index})])
        .collect()
}

/// A stream of `blocks` ending with `stop_reason`.
fn reply(blocks: Vec<Vec<Value>>, stop_reason: &str) -> Vec<Value> {
    std::iter::once(json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": super::super::completion::CLAUDE_SONNET_4_6,
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    }}))
    .chain(blocks.into_iter().flatten())
    .chain([json!({"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": 5}})])
    .collect()
}

/// The preserved provider item of one finalized block.
fn item(content: &crate::message::AssistantContent) -> Option<&Value> {
    match content {
        crate::message::AssistantContent::Opaque(opaque) => Some(&opaque.item),
        crate::message::AssistantContent::CarriedText(text) => Some(&text.payload.item),
        content => content.native_item(),
    }
}

/// The streaming request body the [`Messages`](super::super::wire::Messages)
/// wire encodes — the shared typed request plus the streaming-only patches.
///
/// Read back off the encoded HTTP request rather than rebuilt here: the
/// wire's `encode` is the only statement of that body now, so a cell that
/// pins the body pins the thing that runs.
fn built_streaming_body(
    model: &str,
    request: CompletionRequest,
    strict_tools: bool,
) -> Result<Value, ProviderError> {
    use crate::wire::{Body, Mode, Wire};

    let wire =
        crate::providers::anthropic::wire::AnthropicConfig::new("test-key").completion(model);
    let wire = if strict_tools {
        wire.with_strict_tools()
    } else {
        wire
    };
    let request = <Completion as crate::wire::Operation>::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Streaming)?;
    match encoded.request.body() {
        Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        Body::Multipart(_) => Err(ProviderError::request("the Messages endpoint takes JSON")),
    }
}

#[test]
fn test_streaming_tool_build_marks_final_combined_tool() {
    let request = CompletionRequest::new(RigMessage::user("Hi"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("rig_tool").expect("tool name"),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }])
        .additional_params(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }]
        }));
    let wire = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
        .completion(CLAUDE_SONNET_4_6)
        .with_prompt_caching();
    let body = streamed_body(&wire, request);
    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

/// The streaming body `wire` encodes for `request`.
fn streamed_body(
    wire: &crate::providers::anthropic::Messages,
    request: CompletionRequest,
) -> Value {
    use crate::wire::Wire;
    crate::test_utils::json_body(
        &wire
            .encode(request, Mode::Streaming)
            .expect("the request encodes")
            .request,
    )
}

#[test]
fn streaming_request_keeps_documents_after_leading_system_messages() {
    let request = CompletionRequest::from(vec![
        RigMessage::system("System prompt"),
        RigMessage::assistant("Earlier assistant turn"),
        RigMessage::system("Mid-conversation instruction"),
        RigMessage::user("Prompt"),
    ])
    .max_tokens(64)
    .documents(vec![RigDocument {
        id: "doc1".to_string(),
        text: "Document text.".to_string(),
        additional_props: Default::default(),
    }]);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert_eq!(body["system"][0]["text"], "System prompt");
    assert_eq!(body["system"][1]["text"], "Mid-conversation instruction");
    let messages = body["messages"]
        .as_array()
        .expect("messages should be array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert!(
        messages[0].to_string().contains("<file id: doc1>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
}

#[test]
fn streaming_body_is_blocking_body_plus_stream_flag_and_carries_output_schema() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "title": "WeatherResponse",
        "type": "object",
        "properties": { "city": { "type": "string" } }
    }))
    .expect("schema should deserialize");

    let request = CompletionRequest::from(vec![
        RigMessage::system("You are helpful"),
        RigMessage::user("What's the weather?"),
    ])
    .temperature(0.5)
    .max_tokens(64)
    .output_schema(schema);

    let streaming_body = built_streaming_body(CLAUDE_OPUS_4_8, request.clone(), false)
        .expect("streaming request body should build");

    // The streaming endpoint flag is set.
    assert_eq!(streaming_body["stream"], serde_json::Value::Bool(true));

    // Regression: `output_schema` now reaches the streaming wire as
    // `output_config` (the hand-rolled body dropped it entirely, so this
    // assertion would have failed before the typed-request unification).
    assert_eq!(
        streaming_body["output_config"]["format"]["type"],
        "json_schema"
    );
    assert!(
        streaming_body["output_config"]["format"]["schema"].is_object(),
        "streaming body must carry the structured-output schema: {streaming_body}"
    );

    // Unification invariant: the streaming body is exactly the blocking body
    // (built via the same typed request) plus `stream: true`. Pins the two
    // wire formats together so a future edit can't reintroduce drift.
    let wire = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
        .completion(CLAUDE_OPUS_4_8);
    let mut expected = crate::test_utils::json_body(
        &crate::wire::Wire::encode(&wire, request, Mode::Unary)
            .expect("blocking request body should build")
            .request,
    );
    expected
        .as_object_mut()
        .expect("body is an object")
        .insert("stream".to_string(), serde_json::Value::Bool(true));

    assert_eq!(streaming_body, expected);
}

#[test]
fn streaming_body_keeps_explicit_tool_choice_auto_when_tools_present_but_unset() {
    let request = CompletionRequest::new(RigMessage::user("Add 2 and 3"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("add").expect("tool name"),
            description: "Add x and y".into(),
            parameters: json!({"type":"object", "properties": {"x": { "type": "integer" }}}),
        }]);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    // Tools advertised + `tool_choice` unset must still carry the explicit
    // `auto` the streaming wire format has always sent (parity with recorded
    // fixtures), even though the blocking typed request omits it.
    assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
    assert!(body["tools"].is_array());
}

#[test]
fn streaming_body_applies_strict_tool_opt_in() {
    let request = CompletionRequest::new(RigMessage::user("Look this up"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("lookup").expect("tool name"),
            description: "Look up a value".into(),
            parameters: json!({"type":"object", "properties": {"query": { "type": "string" }}, "required": ["query"]}),
        }]);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, true)
        .expect("streaming request body should build");

    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(
        body["tools"][0]["input_schema"]["additionalProperties"],
        false
    );
    assert_eq!(
        body["tools"][0]["input_schema"]["required"],
        json!(["query"])
    );
}

#[test]
fn streaming_body_drops_tool_choice_when_no_tools_are_advertised() {
    // The typed request serializes a caller-set `tool_choice` regardless of
    // whether tools are present, but the streaming path has always emitted
    // `tool_choice` *only* alongside a non-empty tool set (Anthropic rejects it
    // otherwise). A `tool_choice` set with no tools must not reach the wire.
    let request = CompletionRequest::new(RigMessage::user("Hi"))
        .max_tokens(64)
        .tool_choice(crate::message::ToolChoice::Auto);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert!(
        body.get("tool_choice").is_none(),
        "tool_choice must be omitted when no tools are advertised: {body}"
    );
    assert!(body.get("tools").is_none());
}

#[test]
fn test_streaming_prompt_cache_control_uses_raw_top_level_ttl() {
    let request = CompletionRequest::from(vec![
        RigMessage::system("System prompt"),
        RigMessage::user("Hi"),
    ])
    .max_tokens(64)
    .tools(vec![crate::completion::ToolDefinition {
        name: crate::message::ToolName::new("rig_tool").expect("tool name"),
        description: "Rig tool".into(),
        parameters: json!({"type":"object", "properties": {}}),
    }])
    .additional_params(json!({"cache_control":{"type":"ephemeral", "ttl":"1h"}}));
    let wire = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
        .completion(CLAUDE_OPUS_4_8)
        .with_prompt_caching();
    let body = streamed_body(&wire, request);
    assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        body["cache_control"],
        json!({"type":"ephemeral", "ttl":"1h"})
    );
}

#[test]
fn test_thinking_delta_deserialization() {
    let event = classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this..."}}"#,
    );
    assert_eq!(event.fields["delta"]["type"], "thinking_delta");
    assert_eq!(
        event.fields["delta"]["thinking"],
        "Let me think about this..."
    );
}

#[test]
fn test_signature_delta_deserialization() {
    let event = classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123def456"}}"#,
    );
    assert_eq!(event.fields["delta"]["type"], "signature_delta");
    assert_eq!(event.fields["delta"]["signature"], "abc123def456");
}

#[test]
fn test_thinking_delta_streaming_event_deserialization() {
    let event = classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"First, I need to understand the problem."}}"#,
    );
    assert_eq!(event.fields["type"], "content_block_delta");
    assert_eq!(event.fields["index"], 0);
    assert_eq!(event.fields["delta"]["type"], "thinking_delta");
    assert_eq!(
        event.fields["delta"]["thinking"],
        "First, I need to understand the problem."
    );
}

#[test]
fn test_signature_delta_streaming_event_deserialization() {
    let event = classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"ErUBCkYICBgCIkCaGbqC85F4"}}"#,
    );
    assert_eq!(event.fields["type"], "content_block_delta");
    assert_eq!(event.fields["index"], 0);
    assert_eq!(event.fields["delta"]["type"], "signature_delta");
    assert_eq!(
        event.fields["delta"]["signature"],
        "ErUBCkYICBgCIkCaGbqC85F4"
    );
}

#[test]
fn test_handle_thinking_delta_event() {
    let decoded = decode([classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Analyzing the request..."}}"#,
    )]);
    let events = decoded.events();
    assert!(matches!(
        events[0],
        StreamEvent::Start {
            kind: crate::streaming::PartKind::Reasoning,
            ..
        }
    ));
    assert!(
        matches!(events[1], StreamEvent::Reasoning { text, .. } if text == "Analyzing the request...")
    );
    assert_eq!(events[0].part(), events[1].part());
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::End {
            content: crate::message::AssistantContent::Reasoning(_),
            ..
        }
    )));
}

#[test]
fn test_handle_signature_delta_event() {
    let decoded = decode([
        classified(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"test_signature"}}"#,
        ),
        classified(r#"{"type":"content_block_stop","index":0}"#),
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}"#,
        ),
    ]);
    assert!(
        !decoded
            .events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Reasoning { .. }))
    );
    let ended = decoded.ended();
    assert_eq!(ended.len(), 1);
    assert_eq!(
        item(&ended[0]).expect("signed item")["signature"],
        "test_signature"
    );
}

#[test]
fn test_handle_redacted_thinking_content_block_start_event() {
    let frames = reply(
        vec![block(
            0,
            json!({"type":"redacted_thinking","data":"redacted_blob"}),
            &[],
        )],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let ended = decoded.ended();
    let [crate::message::AssistantContent::Reasoning(reasoning)] = ended.as_slice() else {
        panic!("one redacted block: {ended:?}");
    };
    assert!(reasoning.redacted);
    assert_eq!(
        item(&ended[0]),
        Some(&json!({"type":"redacted_thinking","data":"redacted_blob"}))
    );
    let events = decoded.events();
    assert!(matches!(
        events[0],
        StreamEvent::Start {
            kind: crate::streaming::PartKind::Reasoning,
            ..
        }
    ));
    assert!(matches!(events[1], StreamEvent::End { .. }));
    assert_eq!(events[0].part(), events[1].part());
}

/// The adaptive-thinking wire shape, exactly as recorded in
/// `crates/rig-cassette/fixtures/cassettes/anthropic/opus_4_7/messages_adaptive_thinking_streaming_smoke.yaml`:
/// `content_block_start` opens the block with an EMPTY `thinking` and an
/// EMPTY `signature`, a `signature_delta` carries the whole signature, and
/// no `thinking_delta` ever arrives. The block's only content is its
/// signature, and it must survive `content_block_stop`.
#[test]
fn signature_only_thinking_block_survives_content_block_stop() {
    let frames = reply(
        vec![block(
            0,
            json!({"type":"thinking", "thinking":"", "signature":""}),
            &[json!({"type":"signature_delta", "signature":"the_whole_signature"})],
        )],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    assert!(
        !decoded
            .events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Reasoning { .. }))
    );
    let ended = decoded.ended();
    assert_eq!(ended.len(), 1, "signature-only thinking must survive");
    assert_eq!(
        item(&ended[0]).expect("signed reasoning")["signature"],
        "the_whole_signature"
    );
}

/// Forward compat: a block that delivers its whole signature on
/// `content_block_start` and sends no `signature_delta` keeps it.
#[test]
fn signature_delivered_only_on_content_block_start_is_kept() {
    let frames = reply(
        vec![block(
            0,
            json!({"type":"thinking", "thinking":"", "signature":"up_front_signature"}),
            &[],
        )],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    assert!(
        !decoded
            .events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Reasoning { .. }))
    );
    let ended = decoded.ended();
    assert_eq!(ended.len(), 1, "signature-only thinking must survive");
    assert_eq!(
        item(&ended[0]).expect("signed reasoning")["signature"],
        "up_front_signature"
    );
}

/// The opening `signature` is a fallback, never a prefix the deltas
/// extend: a delta-bearing block must publish exactly what the deltas
/// assembled, or the value replayed to Anthropic is corrupt.
#[test]
fn signature_deltas_supersede_the_opening_signature() {
    let frames = reply(
        vec![block(
            0,
            json!({"type":"thinking", "thinking":"", "signature":"opening"}),
            &[
                json!({"type":"signature_delta", "signature":"delta_"}),
                json!({"type":"signature_delta", "signature":"assembled"}),
            ],
        )],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    assert!(
        !decoded
            .events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Reasoning { .. }))
    );
    let ended = decoded.ended();
    assert_eq!(ended.len(), 1, "signature-only thinking must survive");
    assert_eq!(
        item(&ended[0]).expect("signed reasoning")["signature"],
        "delta_assembled"
    );
}

/// `content_block_start` can carry the block's opening text; discarding it
/// would truncate the block the accumulator assembles from deltas.
#[test]
fn thinking_block_start_text_streams_as_the_first_delta() {
    let frames = reply(
        vec![block(
            2,
            json!({"type":"thinking","thinking":"opening "}),
            &[json!({"type":"thinking_delta","thinking":"rest"})],
        )],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let events = decoded.events();
    assert!(matches!(
        events[0],
        StreamEvent::Start {
            kind: crate::streaming::PartKind::Reasoning,
            ..
        }
    ));
    assert!(matches!(events[1], StreamEvent::Reasoning { text, .. } if text == "opening "));
    assert!(matches!(events[2], StreamEvent::Reasoning { text, .. } if text == "rest"));
    assert_eq!(events[0].part(), events[1].part());
    assert_eq!(events[1].part(), events[2].part());
    let ended = decoded.ended();
    let [crate::message::AssistantContent::Reasoning(reasoning)] = ended.as_slice() else {
        panic!("one reasoning block: {ended:?}");
    };
    assert_eq!(reasoning.text, "opening rest");
    assert!(
        item(&ended[0]).is_none(),
        "unsigned reasoning is canonical only"
    );
}

/// A block with neither text nor signature carries nothing to replay.
#[test]
fn wholly_empty_thinking_block_is_dropped() {
    let frames = reply(
        vec![block(0, json!({"type":"thinking","thinking":""}), &[])],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    assert!(decoded.ended().is_empty());
    assert!(
        decoded
            .outcome
            .expect("completed empty turn")
            .choice
            .is_empty()
    );
}

#[test]
fn test_handle_text_delta_event() {
    let decoded = decode([classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, world!"}}"#,
    )]);
    let events = decoded.events();
    assert!(matches!(
        events[0],
        StreamEvent::Start {
            kind: crate::streaming::PartKind::Text,
            ..
        }
    ));
    assert!(matches!(events[1], StreamEvent::Text { text, .. } if text == "Hello, world!"));
    assert_eq!(events[0].part(), events[1].part());
}

#[test]
fn test_handle_text_block_start_event() {
    let decoded = decode([classified(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#,
    )]);
    let events = decoded.events();
    assert!(matches!(
        events.as_slice(),
        [StreamEvent::Start {
            kind: crate::streaming::PartKind::Text,
            source_order: None,
            ..
        }]
    ));
}

#[test]
fn test_thinking_delta_does_not_interfere_with_tool_calls() {
    let decoded = decode([
        classified(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_123","name":"f","input":{}}}"#,
        ),
        classified(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Thinking while tool is active..."}}"#,
        ),
        classified(r#"{"type":"content_block_stop","index":1}"#),
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":0}}"#,
        ),
    ]);
    let events = decoded.events();
    assert!(events.iter().any(|event| matches!(event, StreamEvent::Reasoning { text, .. } if text == "Thinking while tool is active...")));
    assert!(decoded.ended().iter().any(|content| matches!(content, crate::message::AssistantContent::ToolCall(call) if call.id == crate::message::CallId::from_wire("tool_123"))));
    assert!(decoded.outcome.is_ok());
}

#[test]
fn test_handle_input_json_delta_event() {
    let decoded = decode([
        classified(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_123","name":"f","input":{}}}"#,
        ),
        classified(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"arg\":\"value"}}"#,
        ),
    ]);
    let events = decoded.events();
    assert!(matches!(
        events[0],
        StreamEvent::Start {
            kind: crate::streaming::PartKind::ToolCall,
            ..
        }
    ));
    assert!(matches!(events[1], StreamEvent::Arguments { json, .. } if json == "{\"arg\":\"value"));
    assert_eq!(events[0].part(), events[1].part());
    assert!(decoded.outcome.is_err());
}

#[test]
fn test_tool_call_accumulation_with_multiple_deltas() {
    let fragments = ["{\"location\":", "\"Paris\",", "\"temp\":\"20C\"}"];
    let frames = reply(
        vec![block(
            0,
            json!({"type":"tool_use","id":"tool_123","name":"f","input":{}}),
            &fragments
                .iter()
                .map(|fragment| json!({"type":"input_json_delta","partial_json":fragment}))
                .collect::<Vec<_>>(),
        )],
        "tool_use",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let events = decoded.events();
    let arguments = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(arguments, fragments);
    let ended = decoded.ended();
    let [crate::message::AssistantContent::ToolCall(call)] = ended.as_slice() else {
        panic!("one completed call: {ended:?}");
    };
    assert_eq!(call.id, crate::message::CallId::from_wire("tool_123"));
    assert_eq!(
        call.function.arguments_value(),
        json!({"location":"Paris","temp":"20C"})
    );
    assert!(decoded.outcome.is_ok());
}

#[test]
fn test_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "Example",
                    "start_char_index": 0,
                    "end_char_index": 20
                }
            }
        }"#;

    let event = classified(json);
    assert_eq!(event.kind(), "content_block_delta");
    assert_eq!(event.fields["index"], 0);
    let citation = &event.fields["delta"]["citation"];
    assert_eq!(event.fields["delta"]["type"], "citations_delta");
    assert_eq!(citation["type"], "char_location");
    assert_eq!(citation["start_char_index"], 0);
    assert_eq!(citation["end_char_index"], 20);
}

#[test]
fn test_search_result_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "search_result_location",
                    "cited_text": "API requests require a key.",
                    "source": "https://docs.example.com/api-reference",
                    "title": "API Reference",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1
                }
            }
        }"#;

    let event = classified(json);
    assert_eq!(event.kind(), "content_block_delta");
    assert_eq!(event.fields["delta"]["type"], "citations_delta");
    let citation = &event.fields["delta"]["citation"];
    assert_eq!(citation["type"], "search_result_location");
    assert_eq!(citation["search_result_index"], 0);
    assert_eq!(citation["start_block_index"], 0);
    assert_eq!(citation["end_block_index"], 1);
}

#[test]
fn test_web_search_result_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

    let event = classified(json);
    assert_eq!(event.kind(), "content_block_delta");
    assert_eq!(event.fields["delta"]["type"], "citations_delta");
    let citation = &event.fields["delta"]["citation"];
    assert_eq!(citation["type"], "web_search_result_location");
    assert_eq!(citation["url"], "https://example.com/shannon");
    assert_eq!(citation["encrypted_index"], "encrypted-reference");
}

#[test]
fn test_web_search_result_citations_delta_allows_null_title() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": null,
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

    let event = classified(json);
    assert_eq!(event.kind(), "content_block_delta");
    assert_eq!(event.fields["delta"]["type"], "citations_delta");
    let citation = &event.fields["delta"]["citation"];
    assert_eq!(citation["type"], "web_search_result_location");
    assert!(citation["title"].is_null());
}

#[test]
fn test_text_content_block_start_allows_null_citations() {
    // The Anthropic Messages API emits an explicit `"citations": null` on the
    // first text `content_block_start` event. `#[serde(default)]` alone covers
    // a missing field but not an explicit null, so this must deserialize to an
    // empty citation list rather than failing the whole stream (see #1971).
    let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "",
                "citations": null
            }
        }"#;

    let event = classified(json);
    assert_eq!(event.kind(), "content_block_start");
    assert_eq!(event.fields["content_block"]["type"], "text");
    assert_eq!(event.fields["content_block"]["text"], "");
    assert!(event.fields["content_block"]["citations"].is_null());
    let decoded = decode([
        event,
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}"#,
        ),
    ]);
    assert!(decoded.outcome.is_ok());
    assert!(decoded.ended().is_empty());
}

#[test]
fn test_web_search_content_block_start_events_deserialize() {
    let server_tool_use = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            }
        }"#;
    let event = classified(server_tool_use);
    assert_eq!(event.kind(), "content_block_start");
    let content = &event.fields["content_block"];
    assert_eq!(content["type"], "server_tool_use");
    assert_eq!(content["id"], "srvtoolu_01");
    assert_eq!(content["name"], "web_search");
    assert_eq!(content["input"]["query"], "claude shannon birth date");

    let web_search_tool_result = r#"{
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [{
                    "type": "web_search_result",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_content": "encrypted-content"
                }]
            }
        }"#;
    let event = classified(web_search_tool_result);
    assert_eq!(event.kind(), "content_block_start");
    let content = &event.fields["content_block"];
    assert_eq!(content["type"], "web_search_tool_result");
    assert_eq!(content["tool_use_id"], "srvtoolu_01");
    assert_eq!(
        content["content"][0]["encrypted_content"],
        "encrypted-content"
    );
}

#[test]
fn test_code_execution_tool_result_block_is_preserved() {
    let content = json!({"type":"code_execution_tool_result","tool_use_id":"srvtoolu_01","content":{"type":"code_execution_result","return_code":0,"stdout":"42\n","stderr":"","content":[]}});
    let decoded = decode(
        reply(vec![block(1, content.clone(), &[])], "end_turn")
            .iter()
            .map(|frame| classified(&frame.to_string())),
    );
    let ended = decoded.ended();
    assert_eq!(ended.len(), 1);
    assert_eq!(item(&ended[0]), Some(&content));
    assert_eq!(
        item(&ended[0]).expect("raw hosted result")["type"],
        "code_execution_tool_result"
    );
    assert_eq!(
        item(&ended[0]).expect("raw hosted result")["content"]["stdout"],
        "42\n"
    );
}

#[test]
fn test_streaming_web_search_blocks_are_preserved_on_final_choice() {
    let citation = json!({"type":"web_search_result_location","cited_text":"Claude Shannon was born on April 30, 1916.","url":"https://example.com/shannon","title":"Claude Shannon","encrypted_index":"encrypted-index"});
    let frames = reply(
        vec![
            block(
                0,
                json!({"type":"server_tool_use","id":"srvtoolu_01","name":"web_search","input":null}),
                &[
                    json!({"type":"input_json_delta","partial_json":r#"{"query":"claude shannon birth date"}"#}),
                ],
            ),
            block(
                1,
                json!({"type":"web_search_tool_result","tool_use_id":"srvtoolu_01","content":[{"type":"web_search_result","url":"https://example.com/shannon","title":"Claude Shannon","encrypted_content":"encrypted-content"}]}),
                &[],
            ),
            block(
                2,
                json!({"type":"text","text":""}),
                &[
                    json!({"type":"text_delta","text":"Claude Shannon was born on April 30, 1916."}),
                    json!({"type":"citations_delta","citation":citation}),
                ],
            ),
        ],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let choice_items = decoded.ended();
    assert_eq!(choice_items.len(), 3);
    assert!(
        choice_items
            .iter()
            .all(|item| !matches!(item, crate::message::AssistantContent::ToolCall(_))),
        "provider-owned web-search blocks must not become Rig client tool calls"
    );
    let raw = item(&choice_items[0]).expect("raw server tool use");
    assert_eq!(raw["type"], "server_tool_use");
    assert_eq!(raw["input"]["query"], "claude shannon birth date");
    assert_eq!(
        item(&choice_items[1]).expect("raw web search result")["content"][0]["encrypted_content"],
        "encrypted-content"
    );
    let crate::message::AssistantContent::Text(answer) = &choice_items[2] else {
        panic!("expected answer text");
    };
    assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
    assert_eq!(
        item(&choice_items[2]).expect("native citations")["citations"],
        json!([citation])
    );
    assert_eq!(answer.citations().len(), 1);
    assert!(
        !decoded
            .events()
            .iter()
            .any(|event| matches!(event, StreamEvent::Arguments { .. })),
        "hosted input must not emit client tool-call deltas"
    );
}

#[test]
fn test_handle_citations_delta_event_preserves_metadata() {
    let citation = json!({"type":"char_location","cited_text":"The grass is green.","document_index":0,"document_title":"Example","start_char_index":0,"end_char_index":20});
    let frames = reply(
        vec![vec![
            json!({"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":citation}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The grass is green."}}),
            json!({"type":"content_block_stop","index":0}),
        ]],
        "end_turn",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let ended = decoded.ended();
    assert_eq!(
        item(&ended[0]).expect("citation metadata")["citations"][0]["type"],
        "char_location"
    );
    assert_eq!(
        item(&ended[0]).expect("citation metadata")["citations"],
        json!([citation])
    );
}

#[test]
fn test_streaming_citation_deltas_are_preserved_on_final_text() {
    let citation = json!({"type":"char_location","cited_text":"The grass is green.","document_index":0,"document_title":"Example","start_char_index":0,"end_char_index":20});
    let response = streamed(&reply(
        vec![block(
            0,
            json!({"type":"text","text":""}),
            &[
                json!({"type":"text_delta","text":"the grass is green"}),
                json!({"type":"citations_delta","citation":citation}),
            ],
        )],
        "end_turn",
    ))
    .expect("completed text");
    let crate::message::AssistantContent::Text(text) = &response.choice[0] else {
        panic!("expected accumulated text item");
    };
    assert_eq!(text.text, "the grass is green");
    assert_eq!(
        item(&response.choice[0]).expect("native citations")["citations"],
        json!([citation])
    );
}

/// The `#[serde(other)]` policy fallbacks are gone: classification is the
/// only policy site. An unmodeled *top-level* event type is `Unknown`
/// (driver: warn + skip); a `ping` is Known; and a known tag whose payload
/// this client cannot decode is `Corrupt`, never silently demoted to an
/// ignorable unknown. An unmodeled *nested* delta type is the one carved
/// exception (Anthropic's versioning policy reserves the right to add
/// them): it decodes to a known Messages event with an unmodelled delta and stays a Known
/// no-op — see the dedicated tests below.
#[test]
fn classify_dispatches_on_the_known_event_list() {
    let adapter = adapter();

    let frame = WireFrame::Text(r#"{"type":"something_new_from_anthropic","field":"x"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Unknown { event_type, .. }
            if event_type == "something_new_from_anthropic"
    ));

    let frame = WireFrame::Text(r#"{"type":"ping"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Known(event) if event.fields["type"] == "ping"
    ));

    let frame = WireFrame::Text("{not json".into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Forward compat: a novel nested delta type Anthropic ships tomorrow
/// must not corrupt the whole `content_block_delta` frame — it decodes
/// to a known Messages event with an unmodelled delta and interprets as a warned no-op, so the
/// stream continues.
#[test]
fn novel_nested_delta_type_is_a_known_noop() {
    let classifier = adapter();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"banana_delta","x":1}}"#.into(),
    );
    let crate::wire::WireEvent::Known(event) = classifier.classify(frame)
    else {
        panic!("a novel nested delta type must stay a Known event");
    };

    let decoded = decode([
        event,
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}"#,
        ),
    ]);
    assert!(decoded.outcome.is_ok());
    assert!(
        decoded.items.is_empty(),
        "an unmodeled nested delta is a no-op"
    );
}

/// Anthropic reports the per-TTL `cache_creation` split on
/// `message_start` only; the terminal `message_delta` usage omits it. The
/// adapter must carry it onto the terminal record. Unit-tested (not a
/// cassette) because the carry-forward is internal adapter state — the
/// wire evidence lives in the recorded `prompt_caching/matrix_*` streaming
/// cassettes, whose `message_start` frames hold the split.
#[test]
fn per_ttl_cache_creation_split_carries_from_message_start_to_terminal() {
    let start = json!({"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":9366,"ephemeral_5m_input_tokens":336}}}});
    let delta = json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7,"input_tokens":3,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0}});
    let terminal = streamed(&[start, delta]).expect("terminal message_delta must yield a response");
    let split = &terminal.raw["usage"]["cache_creation"];
    assert!(
        !split.is_null(),
        "terminal usage must carry the message_start cache_creation split"
    );
    assert_eq!(split["ephemeral_1h_input_tokens"], 9366);
    assert_eq!(split["ephemeral_5m_input_tokens"], 336);
    assert_eq!(terminal.raw["usage"]["cache_creation_input_tokens"], 9702);
}

/// A `content_block_delta` whose `delta` omits `type` is malformed, not
/// novel: silently skipping it would turn a compat gateway's untagged
/// text delta into a successful *empty* completion. It classifies
/// `Corrupt`, surfacing in-band while the stream keeps consuming
/// (#2258 B5).
#[test]
fn delta_missing_its_type_is_corrupt_not_skipped() {
    let adapter = adapter();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"text":"hello"}}"#.into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Policy preserved: a *known* nested delta tag with a defective payload
/// is a data-level defect, not an unmodeled delta — the frame classifies
/// `Corrupt` instead of degrading to an `Unknown` no-op.
#[test]
fn known_nested_delta_tag_with_defective_payload_is_corrupt() {
    let adapter = adapter();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":42}}"#
            .into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Anthropic's top-level `{"type":"error"}` envelope (e.g.
/// `overloaded_error`) is a Known event that surfaces as a provider error
/// carrying the envelope verbatim — never a warn-skipped unknown — and,
/// since no `message_delta` follows, the stream ends with no terminal
/// record.
///
/// Byte-equality is the assertion, and the frame carries the top-level
/// `request_id` recorded replies carry: an envelope re-encoded from the
/// fields this client models loses every sibling key and normalizes the
/// order, which is the provider's body rendered rather than preserved.
#[test]
fn top_level_error_event_surfaces_as_a_provider_error() {
    const ENVELOPE: &str = r#"{"error":{"message":"Overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let classifier = adapter();
    let crate::wire::WireEvent::Known(event) =
        classifier.classify(WireFrame::Text(ENVELOPE.into()))
    else {
        panic!("the error envelope must classify as a Known event");
    };

    let decoded = decode([event]);
    assert_eq!(
        decoded.items.len(),
        1,
        "the error envelope maps to one error item"
    );
    let Some(Err(error)) = decoded.items.into_iter().last() else {
        panic!("the error envelope must surface as an Err item");
    };
    assert_eq!(error.provider_response_body(), Some(ENVELOPE));
    assert_eq!(
        decoded.outcome.unwrap_err().provider_response_body(),
        Some(ENVELOPE)
    );
}

/// Bedrock-compat quirk: `message_start` without a message body is a
/// Known no-op, not a corrupt frame.
#[test]
fn message_start_with_null_message_is_a_known_noop() {
    let classifier = adapter();
    let frame = WireFrame::Text(r#"{"type":"message_start","message":null}"#.into());
    let crate::wire::WireEvent::Known(event) = classifier.classify(frame)
    else {
        panic!("null-message message_start must stay a known event");
    };

    let decoded = decode([
        event,
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}"#,
        ),
    ]);
    assert!(decoded.outcome.is_ok());
    assert!(
        decoded.items.is_empty(),
        "a message-less message_start is a no-op"
    );
}

#[test]
fn terminal_record_normalizes_stop_reason_usage_and_metadata() {
    let frames = [
        json!({"type":"message_start","message":{"id":"msg_1","model":CLAUDE_OPUS_4_8,"role":"assistant","content":[],"usage":{"input_tokens":0,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}),
        json!({"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"input_tokens":3,"output_tokens":5,"cache_read_input_tokens":2}}),
    ];
    let terminal = streamed(&frames).expect("expected a terminal record");
    assert_eq!(terminal.provider(), "anthropic");
    assert_eq!(
        terminal
            .message_id
            .as_ref()
            .map(crate::completion::ProviderMessageId::as_str),
        Some("msg_1")
    );
    assert_eq!(terminal.model(), Some(CLAUDE_OPUS_4_8));
    assert_eq!(
        terminal.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(terminal.usage.input_tokens, Some(3));
    assert_eq!(terminal.usage.output_tokens, Some(5));
    assert_eq!(terminal.usage.cached_input_tokens, Some(2));
    assert_eq!(terminal.usage.total_tokens, Some(10));
}

#[test]
fn terminal_record_upgrades_end_turn_to_tool_calls_after_a_streamed_tool_call() {
    let frames = reply(
        vec![block(
            0,
            json!({"type":"tool_use","id":"toolu_1","name":"add","input":{"x":1}}),
            &[],
        )],
        "end_turn",
    );
    let terminal = streamed(&frames).expect("expected a terminal record");
    assert_eq!(
        terminal.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[test]
fn unknown_stop_reason_survives_onto_the_terminal_record() {
    let terminal = streamed(&[json!({"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":0}})])
        .expect("expected a terminal record");
    assert_eq!(
        terminal.finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "pause_turn".to_owned()
        ))
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission;

/// A `tool_use` block whose wire id is empty is keyed by a minted key, not
/// the empty string: two such blocks in one stream stay distinct, their
/// deltas and stops follow the same key, and nothing panics.
#[test]
fn an_empty_tool_use_id_is_minted_not_keyed_on_the_empty_string() {
    let frames = reply(
        (0..2)
            .map(|index| {
                block(
                    index,
                    json!({"type":"tool_use","id":"","name":"add","input":{}}),
                    &[json!({"type":"input_json_delta","partial_json":"{\"x\":1}"})],
                )
            })
            .collect(),
        "tool_use",
    );
    let decoded = decode(frames.iter().map(|frame| classified(&frame.to_string())));
    let ended = decoded.ended();
    let calls = ended
        .iter()
        .filter_map(|content| match content {
            crate::message::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    for call in &calls {
        assert!(
            matches!(call.id, crate::message::CallId::Local(_)),
            "{:?}",
            call.id
        );
        assert_eq!(call.function.arguments_value(), json!({"x":1}));
    }
    assert_ne!(
        calls[0].id, calls[1].id,
        "each id-less call is its own block"
    );
    let events = decoded.events();
    for part in 0..2 {
        let for_part = events
            .iter()
            .filter(|event| event.part().index() == part)
            .collect::<Vec<_>>();
        assert!(matches!(for_part.first(), Some(StreamEvent::Start { .. })));
        assert!(for_part.iter().any(
            |event| matches!(event, StreamEvent::Arguments { json, .. } if json == "{\"x\":1}")
        ));
        assert!(matches!(for_part.last(), Some(StreamEvent::End { .. })));
    }
    assert!(decoded.outcome.is_ok());
}

/// The Messages projection, driven through [`crate::driver`].
///
/// The projector is reached as [`crate::wire::Decoder::project`], which the
/// driver calls on every raw payload — the rejection body included — so
/// these cells exercise the wire and the driver together rather than the
/// projector in isolation. That is the only way the *closure* facts
/// (`Started`, `Response`, `Finished`) are observable at all: they belong to
/// the attempt, not to the payload.
mod projection;

const PROGRESS_UPDATE_TEXT: &str = "Confirmed the retry path never refreshes the expired token. Editing auth.py to add the refresh call.";

#[test]
fn thinking_assembles_its_text_and_signature_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "thinking", "thinking": "", "signature": "open-"}),
            &[
                json!({"type": "thinking_delta", "thinking": "Let me "}),
                json!({"type": "thinking_delta", "thinking": "think."}),
                json!({"type": "signature_delta", "signature": "sig_"}),
                json!({"type": "signature_delta", "signature": "end"}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "Let me think.");
    assert!(!reasoning.redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "thinking", "thinking": "Let me think.", "signature": "sig_end"}))
    );
}

#[test]
fn redacted_thinking_is_a_redacted_block_holding_its_payload() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "redacted_thinking", "data": "redacted_blob"}),
            &[],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(Reasoning { text, redacted, .. })] =
        response.choice.as_slice()
    else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert!(text.is_empty() && *redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "redacted_thinking", "data": "redacted_blob"}))
    );
}

#[test]
fn a_call_assembles_its_input_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {},
                "caller": {"type": "direct"}}),
            &[
                json!({"type": "input_json_delta", "partial_json": "{\"location\":"}),
                json!({"type": "input_json_delta", "partial_json": "\"Paris\"}"}),
            ],
        )],
        "tool_use",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.id.to_string(), "toolu_1");
    assert_eq!(
        call.function.arguments_value(),
        json!({"location": "Paris"})
    );
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "tool_use", "id": "toolu_1", "name": "lookup",
            "input": {"location": "Paris"}, "caller": {"type": "direct"}})
        )
    );
    assert_eq!(response.stop(), StopReason::ToolUse);
}

#[test]
fn a_stream_without_its_stop_reason_never_ends_successfully() {
    let mut frames = reply(
        vec![block(0, json!({"type": "text", "text": "partial"}), &[])],
        "end_turn",
    );
    if let Some(delta) = frames.last_mut() {
        delta["delta"]["stop_reason"] = Value::Null;
    }
    frames.push(json!({"type": "message_stop"}));
    assert!(matches!(streamed(&frames), Err(ProviderError::Truncated)));
}

#[test]
fn input_to_a_text_block_fails_the_reply() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "text", "text": ""}),
            &[json!({"type": "input_json_delta", "partial_json": "{}"})],
        )],
        "end_turn",
    );
    assert!(streamed(&frames).is_err());
}

#[test]
fn citation_deltas_land_on_the_text_item() {
    let citation = json!({"type": "char_location", "cited_text": "The grass is green.",
        "document_index": 0, "document_title": "Example", "start_char_index": 0,
        "end_char_index": 20});
    for opening in [json!(null), json!([])] {
        let frames = reply(
            vec![block(
                0,
                json!({"type": "text", "text": "", "citations": opening}),
                &[
                    json!({"type": "citations_delta", "citation": citation}),
                    json!({"type": "text_delta", "text": "the grass is green"}),
                ],
            )],
            "end_turn",
        );
        let response = streamed(&frames).expect("the reply folds");
        let [AssistantContent::Text(text)] = response.choice.as_slice() else {
            panic!("one text block: {:?}", response.choice);
        };
        assert_eq!(text.text, "the grass is green");
        assert_eq!(
            response.choice[0]
                .native_item()
                .and_then(|item| item.get("citations")),
            Some(&json!([citation]))
        );
    }
}

#[test]
fn a_fallback_block_is_kept_first_and_an_error_after_output() {
    let fallback = json!({"type": "fallback", "model": "claude-opus-4-8"});
    let text = json!({"type": "text", "text": ""});
    let leading = streamed(&reply(
        vec![
            block(0, fallback.clone(), &[]),
            block(
                1,
                text.clone(),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("a leading fallback is legal");
    assert!(matches!(
        leading.choice.first(),
        Some(AssistantContent::Opaque(Opaque { replay: false, .. }))
    ));
    let late = streamed(&reply(
        vec![
            block(0, text, &[json!({"type": "text_delta", "text": "hi"})]),
            block(1, fallback, &[]),
        ],
        "end_turn",
    ));
    assert!(late.is_err(), "{late:?}");
}

#[test]
fn cache_usage_from_message_start_survives_output_only_terminal_delta() {
    let frames = [
        r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":6,"cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":1}}}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#,
    ];
    let decoded = crate::driver::feed_frames!(
        MessagesDecoder::new(false),
        document::Message::default(),
        "anthropic",
        frames.map(|frame| WireFrame::Text(frame.to_owned()))
    );
    let response = decoded.outcome.expect("the message_delta ends the reply");
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.cache_creation_input_tokens, Some(4));
    assert_eq!(response.usage.cached_input_tokens, Some(6));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(23));
    let usage = &response.raw["usage"];
    assert_eq!(usage["cache_creation_input_tokens"], 4);
    assert_eq!(usage["cache_read_input_tokens"], 6);
    assert_eq!(usage["cache_creation"]["ephemeral_1h_input_tokens"], 3);
}

#[test]
fn a_delta_without_its_type_or_text_fails_the_reply() {
    for delta in [
        json!({"text": "hello"}),
        json!({"type": "text_delta", "text": 42}),
    ] {
        let frames = reply(
            vec![block(
                0,
                json!({"type": "text", "text": ""}),
                std::slice::from_ref(&delta),
            )],
            "end_turn",
        );
        assert!(streamed(&frames).is_err(), "{delta}");
    }
}

#[test]
fn every_messages_event_has_a_sample() {
    let samples: Vec<MessagesEvent> = [
        r#"{"type":"message_start","message":null}"#,
        r#"{"type":"message","id":"msg_1","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#,
        r#"{"type":"message_stop"}"#,
        r#"{"type":"ping"}"#,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    ]
    .into_iter()
    .map(classified)
    .collect();
    let index = |event: &MessagesEvent| {
        KNOWN_EVENT_TYPES
            .iter()
            .position(|kind| *kind == event.kind())
            .unwrap_or(usize::MAX)
    };
    crate::test_utils::history::assert_every_variant(&samples, index, KNOWN_EVENT_TYPES.len());
}

#[test]
fn an_invented_item_and_field_survive_decode_and_same_model_replay() {
    use crate::wire::{Operation, Wire};

    let invented = json!({"type": "frobnicate", "payload": {"x": 1}});
    let text = json!({"type": "text", "text": "hi", "sparkle": true});
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let whole = json!({
        "type": "message", "id": "msg_1", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [invented, text], "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    });
    let unary = crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Unary,
        [WireFrame::Text(whole.to_string())],
        whole.clone(),
    )
    .expect("the whole reply folds");
    let stream = streamed(&reply(
        vec![
            block(0, invented.clone(), &[]),
            block(
                1,
                json!({"type": "text", "text": "", "sparkle": true}),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("the stream folds");
    assert_eq!(unary.message(), stream.message());

    for response in [unary, stream] {
        let turn = response.message().expect("an assistant turn");
        let request = CompletionRequest::from(vec![
            RigMessage::user("hello"),
            turn,
            RigMessage::user("again"),
        ]);
        let request = Completion::prepare(request, &wire.describe()).expect("the history adapts");
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the request encodes");
        let body = crate::test_utils::json_body(&encoded.request);
        assert_eq!(body["messages"][1]["content"], json!([invented, text]));
    }
}

fn whole(message: Value) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Unary,
        [WireFrame::Text(message.to_string())],
        Value::Null,
    )
}

#[test]
fn a_stop_reason_states_every_open_block_complete() {
    let without_stops = |reason: &str, input: &str| {
        let mut frames = reply(Vec::new(), reason);
        let end = frames.pop().expect("the message_delta");
        frames.extend([
            json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"}}),
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_a", "name": "f", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": input}}),
            end,
        ]);
        frames
    };
    let response = streamed(&without_stops("tool_use", "{\"x\":1}")).expect("the reply folds");
    assert_eq!(response.stop(), StopReason::ToolUse);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "thinking", "thinking": "plan", "signature": "sig"}))
    );
    assert_eq!(
        item(&response.choice[1]),
        Some(&json!({"type": "tool_use", "id": "toolu_a", "name": "f", "input": {"x": 1}}))
    );

    let cut = streamed(&without_stops("max_tokens", "{\"x\":")).expect("the reply folds");
    assert_eq!(cut.stop(), StopReason::Length);
    assert!(item(&cut.choice[0]).is_some(), "{:?}", cut.choice);
    assert!(item(&cut.choice[1]).is_none(), "{:?}", cut.choice);
    let unfinished = streamed(&without_stops("tool_use", "{\"x\":")).expect("the reply folds");
    assert!(unfinished.stop().is_failure(), "{:?}", unfinished.stop());
}

#[test]
fn a_delta_for_a_block_that_never_started_opens_it() {
    let frames = reply(
        vec![vec![
            json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ]],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply decodes");
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hi"),
        "{:?}",
        response.choice
    );
}

fn progress_update_content(update_text: &str) -> Value {
    json!([
        {"type": "thinking", "thinking": "", "signature": "EqMBCkYICxIM-reasoning"},
        {"type": "thinking", "thinking": update_text, "signature": "Es8CCkYICxIM-update"},
        {
            "type": "tool_use",
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "name": "edit_file",
            "input": {"path": "auth.py", "content": "..."}
        }
    ])
}

fn progress_update_reply(update_text: &str, mode: Mode) -> crate::completion::CompletionResponse {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    let frames: Vec<Value> = match mode {
        Mode::Unary => vec![json!({
            "type": "message", "id": "msg_progress", "model": CLAUDE_OPUS_5_5,
            "role": "assistant", "content": progress_update_content(update_text),
            "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 120, "output_tokens": 80}
        })],
        _ => {
            let thinking = |index, text: &str, signature: &str| {
                block(
                    index,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                    &[
                        json!({"type": "thinking_delta", "thinking": text}),
                        json!({"type": "signature_delta", "signature": signature}),
                    ],
                )
            };
            std::iter::once(json!({"type": "message_start", "message": {
                "type": "message", "id": "msg_progress", "model": CLAUDE_OPUS_5_5,
                "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 120, "output_tokens": 1}
            }}))
            .chain(thinking(0, "", "EqMBCkYICxIM-reasoning"))
            .chain(thinking(1, update_text, "Es8CCkYICxIM-update"))
            .chain(block(
                2,
                json!({"type": "tool_use", "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    "name": "edit_file", "input": {}}),
                &[json!({"type": "input_json_delta",
                    "partial_json": "{\"path\": \"auth.py\", \"content\": \"...\"}"})],
            ))
            .chain([
                json!({"type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                    "usage": {"output_tokens": 80}}),
                json!({"type": "message_stop"}),
            ])
            .collect()
        }
    };
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
    .expect("the reply folds")
}

#[test]
fn progress_update_blocks_survive_decoding_and_replay_under_every_display() {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    for update_text in ["", PROGRESS_UPDATE_TEXT] {
        for mode in [Mode::Unary, Mode::Streaming] {
            let response = progress_update_reply(update_text, mode);
            let items: Vec<Value> = response.choice.iter().filter_map(item).cloned().collect();
            assert_eq!(
                Value::Array(items),
                progress_update_content(update_text),
                "{mode:?}: each block decodes as received"
            );
            let turn = response.message().expect("an assistant turn");
            let mut request = CompletionRequest::from(vec![
                RigMessage::user("The login test fails after an hour of uptime."),
                turn,
            ]);
            request.chat_history.push(RigMessage::tool_result(
                crate::message::CallId::from_wire("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
                crate::message::ToolName::new("edit_file").expect("a tool name"),
                "saved",
            ));
            request.tools = vec![crate::completion::ToolDefinition {
                name: crate::message::ToolName::new("edit_file").expect("a tool name"),
                description: "Edit a file.".to_owned(),
                parameters: json!({"type": "object", "properties": {}}),
            }];
            let request =
                Completion::prepare(request, &wire.describe()).expect("the history adapts");
            let encoded = wire
                .encode(request, Mode::Unary)
                .expect("the request encodes");
            let body = crate::test_utils::json_body(&encoded.request);
            assert_eq!(
                body["messages"][1]["content"],
                progress_update_content(update_text),
                "the {mode:?} turn replays exactly as received"
            );
        }
    }
}

fn cited_blocks() -> Vec<(&'static str, Value)> {
    vec![
        (
            "The grass is green.",
            json!({"type": "char_location", "cited_text": "The grass is green.",
                "document_index": 0, "document_title": "Lawn", "start_char_index": 0,
                "end_char_index": 20, "file_id": "file_1"}),
        ),
        (
            "Pages two and three.",
            json!({"type": "page_location", "cited_text": "Two. Three.",
                "document_index": 1, "document_title": null, "start_page_number": 2,
                "end_page_number": 4}),
        ),
        (
            "Block one.",
            json!({"type": "content_block_location", "cited_text": "One.",
                "document_index": 2, "document_title": "Blocks", "start_block_index": 1,
                "end_block_index": 2}),
        ),
        (
            "Café opens at nine.",
            json!({"type": "search_result_location", "cited_text": "Opens 9am.",
                "source": "https://example.com/cafe", "title": "Café", "search_result_index": 3,
                "start_block_index": 0, "end_block_index": 1}),
        ),
        (
            "Rust 2.0 shipped.",
            json!({"type": "web_search_result_location", "cited_text": "Rust 2.0 is out.",
                "url": "https://example.com/rust", "title": "Rust", "encrypted_index": "Eo8B"}),
        ),
        (
            "Unknown.",
            json!({"type": "frobnicate_location", "cited_text": "?"}),
        ),
    ]
}

fn expected_sources() -> Vec<Vec<Source>> {
    let document = |index, within| SourceLocation::Document {
        index: Some(index),
        id: None,
        within: Some(within),
    };
    vec![
        vec![
            Source::new(SourceLocation::Document {
                index: Some(0),
                id: Some("file_1".to_owned()),
                within: Some(DocumentRange::Chars(0..20)),
            })
            .title("Lawn")
            .cited_text("The grass is green."),
        ],
        vec![Source::new(document(1, DocumentRange::Pages(2..4))).cited_text("Two. Three.")],
        vec![
            Source::new(document(2, DocumentRange::Blocks(1..2)))
                .title("Blocks")
                .cited_text("One."),
        ],
        vec![
            Source::new(SourceLocation::SearchResult {
                index: 3,
                source: "https://example.com/cafe".to_owned(),
                blocks: Some(0..1),
            })
            .title("Café")
            .cited_text("Opens 9am."),
        ],
        vec![
            Source::new(SourceLocation::Url {
                url: "https://example.com/rust".to_owned(),
            })
            .title("Rust")
            .cited_text("Rust 2.0 is out."),
        ],
        vec![],
    ]
}

fn citations_of(
    response: &crate::completion::CompletionResponse,
) -> Vec<(String, Vec<(Option<std::ops::Range<usize>>, Vec<Source>)>)> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some((
                text.text.clone(),
                text.citations()
                    .iter()
                    .map(|citation| {
                        (
                            citation.span.map(|span| span.range()),
                            citation.sources.clone(),
                        )
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

#[test]
fn every_citation_kind_resolves_the_same_unary_and_streamed() {
    use crate::providers::anthropic::wire::{ANTHROPIC, MINIMAX, MOONSHOT, XIAOMIMIMO, ZAI};

    let blocks = cited_blocks();
    let expected: Vec<_> = blocks
        .iter()
        .zip(expected_sources())
        .map(|((text, _), sources)| {
            let cited = (!sources.is_empty()).then_some((None, sources));
            ((*text).to_owned(), cited.into_iter().collect::<Vec<_>>())
        })
        .collect();
    let frames = reply(
        blocks
            .iter()
            .enumerate()
            .map(|(index, (text, citation))| {
                block(
                    index,
                    json!({"type": "text", "text": "", "citations": null}),
                    &[
                        json!({"type": "citations_delta", "citation": citation}),
                        json!({"type": "text_delta", "text": text}),
                    ],
                )
            })
            .collect(),
        "end_turn",
    );
    for (dialect, model) in [
        (&ANTHROPIC, CLAUDE_SONNET_4_6),
        (&ZAI, "glm-5"),
        (&MINIMAX, "MiniMax-M2.7"),
        (&MOONSHOT, "kimi-k3"),
        (&XIAOMIMIMO, "mimo-v2.5"),
    ] {
        let whole = json!({
            "type": "message", "id": "msg_1", "role": "assistant", "model": model,
            "content": blocks.iter().map(|(text, citation)| json!({
                "type": "text", "text": text, "citations": [citation]
            })).collect::<Vec<_>>(),
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 1}
        });
        let wire = AnthropicConfig::with_key(dialect, "test-key").completion(model);
        let decoded = |mode, frames: Vec<WireFrame>| {
            crate::test_utils::decode_reply(
                &wire,
                &CompletionRequest::new("hello"),
                mode,
                frames,
                whole.clone(),
            )
            .expect("the reply folds")
        };
        let unary = decoded(Mode::Unary, vec![WireFrame::Text(whole.to_string())]);
        let stream = decoded(
            Mode::Streaming,
            frames
                .iter()
                .map(|frame| WireFrame::Text(frame.to_string()))
                .collect(),
        );
        assert_eq!(citations_of(&unary), expected, "{}", dialect.name);
        assert_eq!(citations_of(&stream), expected, "{}", dialect.name);
        for response in [&unary, &stream] {
            for (content, (_, citation)) in response.choice.iter().zip(&blocks) {
                assert_eq!(
                    content.native_item().and_then(|item| item.get("citations")),
                    Some(&json!([citation])),
                    "{}",
                    dialect.name
                );
            }
        }
    }
}

#[test]
fn an_empty_reply_folds_the_same_whole_or_streamed() {
    for reason in [
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "model_context_window_exceeded",
        "tool_use",
        "pause_turn",
        "refusal",
        "sensitive",
    ] {
        let streamed = streamed(&reply(vec![], reason));
        let whole = whole(
            json!({"type":"message", "id":"msg_1", "role":"assistant", "model":CLAUDE_SONNET_4_6,
            "content":[], "stop_reason":reason, "stop_sequence":null, "usage":{"input_tokens":3,"output_tokens":5}}),
        );
        assert!(streamed.is_ok(), "{reason}: {streamed:?}");
        if reason == "end_turn" {
            assert_eq!(
                whole.expect("legal empty end_turn").stop(),
                streamed.unwrap().stop()
            );
        } else {
            let error = whole.expect_err("protected unary empty rejection");
            assert_eq!(
                error.to_string(),
                ProviderError::Response(crate::message::EMPTY_RESPONSE_ERROR.to_owned())
                    .to_string()
            );
        }
    }
}

#[test]
fn a_citation_without_its_source_stays_native() {
    for citation in [
        json!({"type":"web_search_result_location", "cited_text":"x"}),
        json!({"type":"search_result_location", "cited_text":"x", "search_result_index":0}),
        json!("not an object"),
    ] {
        let event = json!({"type":"content_block_delta", "index":0, "delta":{"type":"citations_delta", "citation":citation}});
        if citation.is_object() {
            assert!(matches!(
                adapter().classify(WireFrame::Text(event.to_string())),
                WireEvent::Corrupt(_)
            ));
            continue;
        }
        let response = streamed(&reply(
            vec![block(
                0,
                json!({"type":"text", "text":"Claim."}),
                &[json!({"type":"citations_delta", "citation":citation})],
            )],
            "end_turn",
        ))
        .expect("unknown citation survives");
        let [AssistantContent::Text(text)] = response.choice.as_slice() else {
            panic!("one text");
        };
        assert!(text.citations().is_empty());
        assert_eq!(
            response.choice[0]
                .native_item()
                .and_then(|item| item.get("citations")),
            Some(&json!([citation]))
        );
    }
}

#[test]
fn a_citation_without_its_passage_or_title_still_cites_its_source() {
    let citation = json!({"type":"web_search_result_location", "url":"https://example.com/rust"});
    for event in [
        json!({"type":"content_block_start", "index":0, "content_block":{
        "type":"text", "text":"", "citations":[citation]}}),
        json!({"type":"content_block_delta", "index":0, "delta":{"type":"citations_delta", "citation":citation}}),
    ] {
        assert!(matches!(
            adapter().classify(WireFrame::Text(event.to_string())),
            WireEvent::Corrupt(_)
        ));
    }
}

#[test]
fn a_length_terminal_keeps_a_valid_open_call_unfinished() {
    let mut frames = reply(Vec::new(), "max_tokens");
    let end = frames.pop().expect("the genuine terminal");
    frames.extend([
        json!({"type":"content_block_start","index":0,
            "content_block":{"type":"tool_use","id":"call_1","name":"add","input":{}}}),
        json!({"type":"content_block_delta","index":0,
            "delta":{"type":"input_json_delta","partial_json":"{\"x\":1,\"y\":2}"}}),
        end,
    ]);
    let response = streamed(&frames).expect("the received turn folds");
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    let error = "the provider ended the reply with a tool call it never finished";
    assert_eq!(response.error.as_deref(), Some(error));
    assert_eq!(response.stop(), StopReason::Error(error.to_owned()));
    let call = response.tool_calls().next().expect("the incomplete call is kept");
    assert_eq!(call.function.arguments_value(), json!({"x":1,"y":2}));
    assert!(call.native.is_none());
}

#[test]
fn a_length_terminal_retains_an_explicitly_finished_valid_call() {
    let response = streamed(&reply(
        vec![block(
            0,
            json!({"type":"tool_use","id":"call_1","name":"add","input":{}}),
            &[json!({"type":"input_json_delta","partial_json":"{\"x\":1,\"y\":2}"})],
        )],
        "max_tokens",
    ))
    .expect("the complete call folds");
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    assert_eq!(response.error, None);
    assert_eq!(response.stop(), StopReason::Length);
    let call = response.tool_calls().next().expect("the completed call is kept");
    assert_eq!(call.function.arguments_value(), json!({"x":1,"y":2}));
    assert!(call.native.is_some());
}
