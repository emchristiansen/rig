use super::*;
use crate::error::ErrorDetail;
use crate::message::AssistantContent;
use crate::test_utils::decode_reply as decode;
use crate::wire::Mode;

fn wire() -> super::super::Interactions {
    crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro")
}

fn frame(value: Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

fn start(index: usize, step: Value) -> WireFrame {
    frame(json!({"event_type":"step.start", "index":index, "step":step}))
}

fn completed() -> WireFrame {
    frame(
        json!({"event_type":"interaction.completed", "interaction":{"id":"int_1", "status":"completed"}}),
    )
}

#[test]
fn a_model_output_step_yields_every_convertible_item() {
    let response = decode(
        &wire(),
        &crate::completion::CompletionRequest::new("restate"),
        Mode::Streaming,
        [
            start(
                0,
                json!({"type":"model_output", "content":[
                    {"type":"text", "text":"answer: "},
                    {"type":"function_call", "name":"add", "arguments":{"x":1}, "id":"fc_9"},
                ]}),
            ),
            completed(),
        ],
        Value::Null,
    )
    .expect("known text and call survive");
    assert_eq!(response.text(), "answer: ");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name.as_str(), "add");
    assert_eq!(calls[0].function.arguments_value(), json!({"x":1}));
    assert_eq!(calls[0].id.provider().map(|id| id.as_str()), Some("fc_9"));
    assert!(
        response
            .choice
            .iter()
            .filter(|content| matches!(content, AssistantContent::ToolCall(_)))
            .all(|content| content.native_item().is_none())
    );
}

#[test]
fn announced_call_finishes_at_genuine_completion_without_a_step_stop() {
    let response = decode(
        &wire(),
        &crate::completion::CompletionRequest::new("restate"),
        Mode::Streaming,
        [
            start(
                9,
                json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":1}}),
            ),
            completed(),
        ],
        Value::Null,
    )
    .expect("genuine completion finalizes the announced call");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments_value(), json!({"x":1}));
    assert!(response.choice[0].native_item().is_none());
}

#[test]
fn announce_arguments_never_concatenate_with_fragments() {
    let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        start(1, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":999}})),
        frame(json!({"event_type":"step.delta", "index":1, "delta":{"type":"arguments_delta", "arguments":"{\"x\":1}"}})),
        completed(),
    ], Value::Null).expect("fragments supersede the announcement");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments_value(), json!({"x":1}));
    assert!(response.choice[0].native_item().is_none());
}

#[test]
fn malformed_arguments_at_completed_retain_the_exact_typed_diagnostic() {
    const RAW: &str = "{\"x\":";
    let error = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        start(1, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":999}})),
        frame(json!({"event_type":"step.delta", "index":1, "delta":{"type":"arguments_delta", "arguments":RAW}})),
        completed(),
    ], Value::Null).expect_err("completed malformed arguments fail");
    let Some(ErrorDetail::MalformedToolInput(detail)) = error.report().detail else {
        panic!("exact malformed-input diagnostic: {error:?}");
    };
    assert_eq!(detail.name, "add");
    assert_eq!(detail.raw, RAW);
    assert_eq!(
        detail.error,
        crate::json_utils::parse_tool_arguments(RAW)
            .expect_err("malformed fixture")
            .to_string()
    );
    assert_eq!(detail.id.provider().map(|id| id.as_str()), Some("fc_1"));
    assert_eq!(
        detail
            .provider
            .as_ref()
            .map(|evidence| evidence.call_id.as_str()),
        Some("fc_1")
    );
}

#[test]
fn unknown_nested_content_remains_an_upstream_opaque_item() {
    let unknown = json!({"type":"future_output", "payload":{"exact":true}});
    let response = decode(
        &wire(),
        &crate::completion::CompletionRequest::new("restate"),
        Mode::Streaming,
        [
            start(
                0,
                json!({"type":"model_output", "content":[unknown.clone()]}),
            ),
            frame(json!({"event_type":"step.stop", "index":0})),
            completed(),
        ],
        Value::Null,
    )
    .expect("unknown content remains opaque");
    assert!(matches!(&response.choice[0], AssistantContent::Opaque(item)
        if item.item == json!({"type":"model_output", "content":[unknown]})));
}

#[test]
fn empty_argument_delta_supersedes_announced_arguments_at_step_stop() {
    for fragment in ["", " "] {
        let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
            start(1, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":999}})),
            frame(json!({"event_type":"step.delta", "index":1, "delta":{"type":"arguments_delta", "arguments":fragment}})),
            frame(json!({"event_type":"step.stop", "index":1})),
            completed(),
        ], Value::Null).expect("an empty real fragment replaces the announcement");
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments_value(), json!({}));
    }
}

#[test]
fn missing_argument_delta_preserves_the_announcement_at_step_stop() {
    for delta in [
        json!({"type":"arguments_delta"}),
        json!({"type":"arguments_delta", "arguments":null}),
    ] {
        let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
            start(1, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":999}})),
            frame(json!({"event_type":"step.delta", "index":1, "delta":delta})),
            frame(json!({"event_type":"step.stop", "index":1})),
            completed(),
        ], Value::Null).expect("no argument fragment retains the announcement");
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments_value(), json!({"x":999}));
    }
}

#[test]
fn thought_signature_completes_the_accumulated_reasoning_block() {
    let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        frame(json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_summary", "content":{"type":"text", "text":"think1 "}}})),
        frame(json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_summary", "content":{"type":"text", "text":"think2"}}})),
        frame(json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_signature", "signature":"sig-abc"}})),
        start(1, json!({"type":"model_output", "content":[{"type":"text", "text":"answer"}]})),
        frame(json!({"event_type":"step.stop", "index":1})),
        completed(),
    ], Value::Null).expect("signed reasoning and the later answer survive");
    let reasoning: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::CarriedReasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(
        reasoning[0].content,
        [CarriedReasoningPart::Text {
            text: "think1 think2".to_owned(),
            signature: Some("sig-abc".to_owned()),
        }]
    );
    assert_eq!(
        reasoning[0].payload.provider.as_deref(),
        Some(PROVIDER_NAME)
    );
    assert_eq!(response.text(), "answer");
}

#[test]
fn signature_only_thought_still_carries_the_signature() {
    for signature in ["sig-only", ""] {
        let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
            frame(json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_signature", "signature":signature}})),
            completed(),
        ], Value::Null).expect("a signature without summary text survives");
        assert_eq!(response.choice.len(), 1);
        assert!(
            matches!(&response.choice[0], AssistantContent::CarriedReasoning(reasoning)
            if reasoning.content == [CarriedReasoningPart::Text { text: String::new(), signature: Some(signature.to_owned()) }])
        );
    }
}

#[test]
fn nested_thought_and_image_keep_their_protected_canonical_meaning() {
    let image =
        json!({"type":"image", "data":"aW1n", "mime_type":"image/png", "extra":{"exact":true}});
    let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        start(0, json!({"type":"model_output", "content":[
            {"type":"thought", "signature":"sig", "summary":[{"type":"text", "text":"plan"}]},
            {"type":"text", "text":"answer"}, image.clone(),
        ]})),
        frame(json!({"event_type":"step.stop", "index":0})),
        completed(),
    ], Value::Null).expect("nested known parts preserve their protected representations");
    assert_eq!(response.choice.len(), 3);
    assert!(
        matches!(&response.choice[0], AssistantContent::CarriedReasoning(reasoning)
        if reasoning.display_text() == "plan")
    );
    assert_eq!(response.text(), "answer");
    assert!(
        matches!(&response.choice[2], AssistantContent::CarriedText(text)
        if text.payload.item == image
            && text.payload.provider.as_deref() == Some(PROVIDER_NAME)
            && text.payload.position == CarriedPosition::MessageContent { item_index: 0, part_index: 2 }
            && text.known_text_projection() == Some(""))
    );
}

#[test]
fn model_output_items_do_not_collide_with_later_provider_step_indices() {
    let response = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        start(0, json!({"type":"model_output", "content":[
            {"type":"text", "text":"a"}, {"type":"text", "text":"b"}, {"type":"text", "text":"c"},
        ]})),
        start(1, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":1}})),
        frame(json!({"event_type":"step.stop", "index":0})),
        completed(),
    ], Value::Null).expect("provider indices and writer slots are distinct");
    assert_eq!(response.text(), "abc");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.provider().map(|id| id.as_str()), Some("fc_1"));
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn signed_reasoning_survives_eof_without_a_response_terminal() {
    use crate::driver::Model;
    use crate::streaming::{Item, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;
    let frames = [
        json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_summary", "content":{"type":"text", "text":"think1 "}}}),
        json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_summary", "content":{"type":"text", "text":"think2"}}}),
        json!({"event_type":"step.delta", "index":0, "delta":{"type":"thought_signature", "signature":"sig-abc"}}),
    ];
    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    );
    let model = Model::new(wire(), MockStreamingClient { sse_bytes });
    let mut stream = model
        .stream(crate::completion::CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut signed = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("signed thought deltas are valid") {
            Item::Event(StreamEvent::End {
                content: AssistantContent::CarriedReasoning(reasoning),
                reasoning_authority,
                ..
            }) => {
                assert_eq!(
                    reasoning_authority, None,
                    "G12(a): Gemini retains default absent authority; the Partial probe was preparation-added"
                );
                signed.push(reasoning);
            }
            Item::Event(
                StreamEvent::Start { .. }
                | StreamEvent::Resume { .. }
                | StreamEvent::Text { .. }
                | StreamEvent::Reasoning { .. }
                | StreamEvent::Arguments { .. }
                | StreamEvent::Input { .. }
                | StreamEvent::End { .. }
                | StreamEvent::Discard { .. },
            )
            | Item::Unknown(_) => {}
        }
    }
    assert_eq!(signed.len(), 1);
    assert_eq!(
        signed[0].content,
        [CarriedReasoningPart::Text {
            text: "think1 think2".to_owned(),
            signature: Some("sig-abc".to_owned()),
        }]
    );
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[test]
fn a_later_named_start_retains_arguments_from_the_same_provider_step() {
    let error = decode(&wire(), &crate::completion::CompletionRequest::new("restate"), Mode::Streaming, [
        frame(json!({"event_type":"step.delta", "index":7, "delta":{"type":"arguments_delta", "arguments":"{\"x\":1}"}})),
        start(7, json!({"type":"function_call", "id":"fc_1", "name":"add", "arguments":{"x":999}})),
        completed(),
    ], Value::Null).expect_err("G13: a later named start cannot authorize the earlier orphan fragment");
    let ProviderError::Response(message) = &error else {
        panic!("G13: exact Response refusal carrier: {error:?}");
    };
    assert_eq!(
        message,
        r#"interactions: arguments fragment at index 7 has no open call; received {"type":"arguments_delta","arguments":"{\"x\":1}"}"#
    );
    assert_eq!(error.kind(), crate::error::ErrorKind::Response);
    assert!(!error.is_retryable());
    assert!(
        std::error::Error::source(&error).is_none(),
        "G13: received typed form is lost; no recovery or success parity is claimed"
    );
}
