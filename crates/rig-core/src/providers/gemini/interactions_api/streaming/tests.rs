use super::*;
use crate::test_utils::history::assert_every_variant;
use crate::wire::Mode;
use serde_json::json;

/// The request every stream test below sends. The decoder is what they
/// exercise, so the request only has to be well-formed.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn interactions_request() -> crate::completion::CompletionRequest {
    crate::completion::CompletionRequest::new("hello")
}

/// The Interactions wire bound to a transport answering with `frames` as
/// one SSE body.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn interactions_stream(
    frames: &[&str],
) -> crate::Model<
    crate::providers::gemini::interactions_api::Interactions,
    crate::test_utils::MockStreamingClient,
> {
    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    crate::Model::new(
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro"),
        crate::test_utils::MockStreamingClient { sse_bytes },
    )
}

#[test]
fn test_streaming_completion_response_has_model_version() {
    let response = decode_interactions(
        Mode::Streaming,
        [
            delta(0, json!({"type":"text", "text":"Hello"})),
            stop(0),
            frame(json!({
                "event_type":"interaction.completed",
                "interaction":{
                    "id":"int_1", "status":"completed",
                    "model":"gemini-2.5-pro-preview-05-06"
                }
            })),
        ],
    )
    .expect("G10: genuine completed interaction supplies the reported model");

    assert_eq!(response.model(), Some("gemini-2.5-pro-preview-05-06"));
    let json = serde_json::to_string(&response).unwrap();
    let deserialized: crate::completion::CompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.model(), Some("gemini-2.5-pro-preview-05-06"));
}

#[test]
fn test_content_delta_text_event() {
    use crate::streaming::{Item, PartKind, StreamEvent};
    use crate::wire::{Decoder, Wire, WireEvent};

    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "text",
            "text": "Hello"
        }
    });
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro");
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, crate::wire::Mode::Streaming),
    ));
    let mut decoder = wire.decoder_in(crate::wire::Mode::Streaming);
    let WireEvent::Known(event) = decoder.classify(WireFrame::Text(event_json.to_string())) else {
        panic!("expected step delta");
    };
    let _ = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect("parts should exist");
    let shared = shared.lock().expect("reply state");
    let items: Vec<_> = shared.items.iter().collect();
    let [
        Ok(Item::Event(StreamEvent::Start {
            part: start,
            kind: PartKind::Text,
            ..
        })),
        Ok(Item::Event(StreamEvent::Text {
            part: text_part,
            text,
        })),
    ] = items.as_slice()
    else {
        panic!("unexpected text-only events: {items:?}");
    };
    assert_eq!(Some(text.as_str()), Some("Hello"));
    assert_eq!(start, text_part);
    // The exact text-only sequence above admits no tool event.
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    use crate::streaming::{Item, StreamEvent};
    use futures::StreamExt;

    // Content deltas then EOF without `interaction.completed`: the
    // truncated stream must deliver its content but never a synthesized
    // terminal record.
    let model = interactions_stream(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
    ]);
    let mut stream = model
        .stream(interactions_request())
        .expect("stream should open");
    let mut texts = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("stream item should be Ok")
        {
            texts.push(text);
        }
    }
    assert_eq!(texts, ["hi"]);
    assert_eq!(
        stream.partial().finish_reason(),
        None,
        "EOF without interaction.completed must not synthesize a terminal record"
    );
    assert!(matches!(
        stream.finish().await,
        Err(crate::error::ProviderError::Truncated)
    ));
}

/// Drive Interactions SSE frames through the full normalized path and
/// collect what the consumer sees, in order.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
async fn drive_frames(
    frames: &[&str],
) -> (
    Vec<Result<crate::streaming::Item<crate::streaming::StreamEvent>, crate::error::ProviderError>>,
    crate::streaming::CompletionStream,
) {
    use futures::StreamExt;
    let model = interactions_stream(frames);
    let mut stream = model
        .stream(interactions_request())
        .expect("stream should open");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    (items, stream)
}
/// A `model_output` step interleaving text and a function call in one
/// step's `content`: every convertible item must surface, in wire
/// order. `find_map` kept only the first — a `function_call` following
/// text in the same step silently vanished.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_model_output_step_yields_every_convertible_item() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":0,"step":{"type":"model_output","content":[{"type":"text","text":"answer: "},{"type":"function_call","name":"add","arguments":{"x":1},"id":"fc_9"}]}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for item in &items {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text.clone()),
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => {
                calls.push(tool_call.clone());
            }
            _ => {}
        }
    }
    assert_eq!(texts, ["answer: "], "the text survives, got {items:?}");
    assert_eq!(
        calls.len(),
        1,
        "the function_call after text must also survive, got {items:?}"
    );
    let call = calls.first().expect("one call");
    assert_eq!(call.function.name.as_str(), "add");
    assert_eq!(call.function.arguments_value(), serde_json::json!({"x": 1}));
}

/// A `step.start` that announces non-empty arguments AND fragments the
/// real payload across `arguments_delta` events: the deltas are the
/// arguments. Concatenating the announce payload with the fragments
/// yields `{..}{..}` — unparseable under the step's Error policy, so
/// the call was lost outright.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_never_concatenate_with_fragments() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":1},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced-then-fragmented call must survive, got {items:?}"
    );
    assert_eq!(
        tool_calls
            .first()
            .expect("one call")
            .function
            .arguments_value(),
        serde_json::json!({"x": 1}),
        "streamed fragments are the arguments; the announce payload is not prepended"
    );
}

/// A partial announce with NO fragments: the announce payload is the
/// only arguments the wire sent, so it finalizes the call
/// (replace-if-no-deltas).
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_finalize_a_call_with_no_fragments() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":7},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(tool_calls.len(), 1, "got {items:?}");
    assert_eq!(
        tool_calls
            .first()
            .expect("one call")
            .function
            .arguments_value(),
        serde_json::json!({"x": 7})
    );
}

/// Interactions retains its `fc_…` handle as one current Provider call ID.
/// G9 retires the unavailable independent item-ID absence observation;
/// the labelled Native pin separately checks the actual completed call.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_streamed_call_carries_a_single_wire_identity() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => Some(tool_call),
            _ => None,
        })
        .collect();
    let tool_call = tool_calls.first().expect("one call");
    let provider = tool_call.id.provider().expect("the wire issued an id");
    assert_eq!(provider.as_str(), "fc_1");
    assert_eq!(
        tool_call
            .native
            .as_ref()
            .expect("actual decoder retains the completed call")
            .item,
        json!({"arguments":{"x":1}, "id":"fc_1", "name":"add", "type":"function_call"}),
        "G9: actual Native from the received start and argument fragments; independent of the retired item-ID absence predicate"
    );
}

/// A `step.stop` that never arrives must not lose the call: the wire
/// announced it (`step.start`), streamed its full arguments
/// (`arguments_delta`), and proved the turn finished
/// (`interaction.completed`). Before this fix the assembly stayed open,
/// `finish` never ran (terminal return), and the accumulator's
/// end-of-stream clear dropped the whole call — the agent then treated
/// a tool-calling turn as plain text.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_missing_step_stop_does_not_lose_the_announced_call() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"get_weather","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"city\":\"Paris\"}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            })) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced call must survive the missing step.stop, got {items:?}"
    );
    let tool_call = tool_calls.first().expect("one call");
    assert_eq!(tool_call.function.name.as_str(), "get_weather");
    assert_eq!(
        tool_call.function.arguments_value(),
        serde_json::json!({"city": "Paris"}),
        "the streamed argument fragments finalize the call"
    );
    assert_eq!(tool_call.id.provider().map(|id| id.as_str()), Some("fc_1"));

    // The turn completed normally: the terminal record survives too.
    let partial = stream.partial();
    let aggregated_calls = partial
        .choice
        .iter()
        .filter(|content| matches!(content, crate::message::AssistantContent::ToolCall(_)))
        .count();
    assert_eq!(
        aggregated_calls, 1,
        "the call reaches the aggregated choice"
    );
    assert!(
        stream.finish().await.is_ok(),
        "the genuine terminal record survives"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn provider_error_event_ends_the_stream_without_draining_later_frames() {
    use crate::streaming::{Item, StreamEvent};

    // A provider `error` event, then more frames: well-formed content, an
    // unknown frame, and a terminal `interaction.completed`. The error
    // must be the LAST item — the driver stops reading (`is_finished`),
    // so nothing after it is interpreted or passed through as `Unknown`.
    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
        r#"{"event_type":"error","error":{"code":"internal","message":"boom"}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"dead"}}"#,
        r#"{"event_type":"something.future","payload":{"x":1}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let error_position = items
        .iter()
        .position(|item| item.is_err())
        .expect("the provider error must reach the consumer");
    assert_eq!(
        error_position,
        items.len() - 1,
        "the in-band error must end the stream: no later text, Unknown passthrough, or terminal; got {items:?}"
    );
    assert!(
        items.iter().any(|item| matches!(
            item,
            Ok(Item::Event(StreamEvent::Text { text, .. })) if text == "hi"
        )),
        "content before the error must survive"
    );
    assert_eq!(
        items
            .iter()
            .filter_map(|item| match item {
                Ok(Item::Event(StreamEvent::Text { text, .. })) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        ["hi"]
    );
    assert_eq!(stream.partial().finish_reason(), None);
    assert!(
        stream.finish().await.is_err(),
        "the failed stream has no terminal response"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn thought_signature_completes_the_accumulated_reasoning_block() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    // Text-then-signature: the signed block must restate the full
    // accumulated thought text and carry the signature; the aggregated
    // choice keeps it (superseding the deltas), alongside the later text.
    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think1 "}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think2"}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-abc"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = items
        .iter()
        .find_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::CarriedReasoning(reasoning),
                ..
            })) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("the signature must yield a completed Reasoning block");
    assert_eq!(
        signed.content,
        vec![crate::message::CarriedReasoningPart::Text {
            text: "think1 think2".to_string(),
            signature: Some("sig-abc".to_string()),
        }],
        "the signed block must restate the accumulated text with the signature"
    );

    // The aggregated choice keeps exactly one reasoning part carrying the
    // signature — the signed restatement superseded the deltas.
    let partial = stream.partial();
    let choice = &partial.choice;
    let aggregated: Vec<_> = choice
        .iter()
        .filter_map(|content| match content {
            crate::message::AssistantContent::CarriedReasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(aggregated.len(), 1, "got {choice:?}");
    assert_eq!(
        aggregated.first().map(|r| r.content.clone()),
        Some(signed.content)
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn signature_only_thought_still_carries_the_signature() {
    use crate::message::AssistantContent;
    use crate::streaming::{Item, StreamEvent};

    // Signature with no preceding thought-summary text: the signature is
    // the provider's replay-validated payload and must still survive as a
    // signed (empty-text) Reasoning block.
    let (items, _stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-only"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = items
        .iter()
        .find_map(|item| match item {
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::CarriedReasoning(reasoning),
                ..
            })) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("a signature-only block must still yield a signed Reasoning");
    assert_eq!(
        signed.content,
        vec![crate::message::CarriedReasoningPart::Text {
            text: String::new(),
            signature: Some("sig-only".to_string()),
        }]
    );
}

#[test]
fn test_content_delta_function_call_event() {
    use crate::streaming::{Item, PartKind, StreamEvent};
    use crate::wire::{Decoder, Wire, WireEvent};

    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "function_call",
            "name": "get_weather",
            "arguments": {"location": "Paris"},
            "id": "call-1"
        }
    });
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro");
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let mut decoder = wire.decoder_in(Mode::Streaming);
    let WireEvent::Known(event) = decoder.classify(WireFrame::Text(event_json.to_string())) else {
        panic!("expected step delta");
    };
    let _ = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect("parts should exist");
    {
        let state = shared.lock().expect("reply state");
        let items: Vec<_> = state.items.iter().collect();
        let [
            Ok(Item::Event(StreamEvent::Start {
                part: start_id,
                source_order: None,
                kind: PartKind::ToolCall,
                name: Some(start_name),
                namespace: None,
                tool_slot_state: None,
            })),
            Ok(Item::Event(StreamEvent::Arguments {
                part: arguments_id,
                json: arguments,
            })),
            Ok(Item::Event(StreamEvent::End {
                part: end_id,
                content: AssistantContent::ToolCall(call),
                reasoning_authority: None,
            })),
        ] = items.as_slice()
        else {
            panic!("unexpected tool events: {items:?}");
        };
        assert_eq!(start_id, end_id);
        assert_eq!(arguments_id, end_id);
        assert_eq!(start_name.as_str(), "get_weather");
        assert_eq!(call.function.name.as_str(), "get_weather");
        assert_eq!(
            call.id
                .provider()
                .map(crate::message::ProviderCallId::as_str),
            Some("call-1")
        );
        assert_eq!(call.native, None);
        assert_eq!(
            serde_json::from_str::<Value>(arguments).unwrap(),
            json!({"location":"Paris"})
        );
        assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
        // The exact sequence has no text and no manufactured response terminal.
        assert!(state.end.is_none());
    }
    // stop(None) was already a no-op; it cannot close this completed Part twice.
    let WireEvent::Known(stop) =
        decoder.classify(frame(json!({"event_type":"step.stop","index":0})))
    else {
        panic!("expected step stop");
    };
    let _ = decoder
        .decode(stop, crate::wire::Out::new(&shared))
        .expect("existing empty stop");
    let state = shared.lock().expect("reply state");
    assert_eq!(state.items.len(), 3);
    assert!(state.end.is_none());
}

use super::super::{Interactions, create_request_body};
use crate::message::{AssistantContent, Message};
use crate::streaming::{Item, StreamEvent};

fn wire() -> Interactions {
    crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro")
}

fn frame(value: serde_json::Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

fn classify_interactions_frame(data: &str) -> WireEvent<InteractionsEvent> {
    InteractionsDecoder::default().classify(WireFrame::Text(data.to_owned()))
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn an_in_band_error_with_a_status_code_is_classified_like_the_rest_stream()
-> anyhow::Result<()> {
    use futures::StreamExt;

    for code in [429_u16, 503] {
        let frame = json!({
            "event_type": "error",
            "error": { "code": code, "message": "The model is overloaded.", "status": "UNAVAILABLE" }
        })
        .to_string();
        let model = crate::driver::Model::new(
            wire(),
            crate::test_utils::MockStreamingClient {
                sse_bytes: bytes::Bytes::from(format!("data: {frame}\n\n")),
            },
        );
        let mut stream = model.stream(crate::completion::CompletionRequest::new("hello"))?;
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            if let Err(error) = item {
                errors.push(error);
            }
        }
        anyhow::ensure!(errors.len() == 1, "one terminal failure: {errors:?}");
        anyhow::ensure!(
            errors[0].is_retryable(),
            "code {code} must be retryable: {:?}",
            errors[0]
        );
    }
    Ok(())
}

#[test]
fn a_tagged_frame_never_passes_for_a_whole_interaction() {
    let event = classify_interactions_frame(r#"{"event_type":"step.stop","status":"completed"}"#);
    assert!(matches!(event, WireEvent::Known(InteractionsEvent::Sse(_))));
    let outcome = crate::test_utils::decode_reply(
        &wire(),
        &crate::completion::CompletionRequest::new("restate"),
        Mode::Streaming,
        [frame(
            json!({"event_type": "step.stop", "status": "completed"}),
        )],
        Value::Null,
    );
    assert!(outcome.is_err(), "{outcome:?}");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentKind {
    Text,
    /// A recognized image, retained only for inspection.
    Image,
    /// Audio, video, a document, or a type rig does not know.
    Other,
}

fn whole(steps: Vec<serde_json::Value>) -> WireFrame {
    frame(json!({"id": "int_1", "status": "completed", "steps": steps}))
}

fn start(index: usize, step: serde_json::Value) -> WireFrame {
    frame(json!({"event_type": "step.start", "index": index, "step": step}))
}

fn delta(index: usize, delta: serde_json::Value) -> WireFrame {
    frame(json!({"event_type": "step.delta", "index": index, "delta": delta}))
}

fn stop(index: usize) -> WireFrame {
    frame(json!({"event_type": "step.stop", "index": index}))
}

fn completed() -> WireFrame {
    frame(json!({
        "event_type": "interaction.completed",
        "interaction": {"id": "int_1", "status": "completed"},
    }))
}

fn replayed(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    let turn = response.message().expect("a turn");
    let tools = match &turn {
        Message::Assistant(turn) => turn
            .tool_calls()
            .map(|call| crate::completion::ToolDefinition {
                name: call.function.name.clone(),
                description: "A tool.".to_owned(),
                parameters: json!({"type": "object", "properties": {}}),
            })
            .collect(),
        _ => Vec::new(),
    };
    let mut request =
        crate::completion::CompletionRequest::from(vec![Message::user("hello"), turn]);
    request.tools = tools;
    let request = <Completion as crate::wire::Operation>::prepare(
        request,
        &crate::wire::Wire::describe(&wire()),
    )
    .expect("the request is valid");
    let body = create_request_body(&wire(), &request, None).expect("the request builds");
    let mut steps = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    steps.remove(0);
    steps.retain(|step| step["type"] != "function_result");
    steps
}

fn content_kind(content: &serde_json::Value) -> ContentKind {
    match content.get("type").and_then(serde_json::Value::as_str) {
        Some("text") => ContentKind::Text,
        Some("image") => ContentKind::Image,
        _ => ContentKind::Other,
    }
}

fn decode_interactions(
    mode: Mode,
    frames: impl IntoIterator<Item = WireFrame>,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    crate::test_utils::decode_reply(
        &wire(),
        &crate::completion::CompletionRequest::new("restate"),
        mode,
        frames,
        Value::Null,
    )
}

#[test]
fn a_whole_interaction_and_its_stream_agree() {
    let thought = json!({"type": "thought", "signature": "sig", "summary": [{"type": "text", "text": "plan"}]});
    let search = json!({"type": "google_search_call", "id": "call_1", "signature": "s", "arguments": {"queries": ["euro 2024"]}, "search_type": "web_search"});
    let output = json!({"type": "model_output", "content": [{"type": "text", "text": "Spain won.", "annotations": [{"start_index": 0, "end_index": 5, "source": "https://example.com"}]}]});
    let unary = decode_interactions(
        Mode::Unary,
        [whole(vec![thought.clone(), search.clone(), output])],
    )
    .expect("whole interaction decodes");
    let streamed = decode_interactions(Mode::Streaming, [
            start(0, json!({"type": "thought"})),
            delta(
                0,
                json!({"type": "thought_summary", "content": {"type": "text", "text": "plan"}}),
            ),
            delta(0, json!({"type": "thought_signature", "signature": "sig"})),
            stop(0),
            start(
                1,
                json!({"type": "google_search_call", "id": "call_1", "signature": ""}),
            ),
            delta(1, search),
            stop(1),
            start(2, json!({"type": "model_output"})),
            delta(
                2,
                json!({"type": "text", "text": "Spain", "annotations": [{"start_index": 0, "end_index": 5, "source": "https://example.com"}]}),
            ),
            delta(2, json!({"type": "text", "text": " won."})),
            stop(2),
            completed(),
        ]).expect("streamed interaction decodes");
    for response in [&unary, &streamed] {
        let Some(AssistantContent::CarriedReasoning(part)) = response.choice.first() else {
            panic!("thought remains carried reasoning");
        };
        assert_eq!(part.payload.item, thought);
        assert_eq!(part.payload.provider.as_deref(), Some("gcp.gemini"));
        assert_eq!(
            serde_json::to_value(part.payload.position).unwrap(),
            json!({"kind":"whole"})
        );
    }
    assert_eq!(
        unary.message(),
        streamed.message(),
        "a whole reply and its restatement as a stream fold into the same turn"
    );
}

#[test]
fn a_thought_start_with_its_summary_agrees_with_the_whole() {
    let thought = json!({"type": "thought", "signature": "sig", "summary": [{"type": "text", "text": "plan"}]});
    let unary = decode_interactions(Mode::Unary, [whole(vec![thought.clone()])])
        .expect("whole thought decodes");
    let streamed = decode_interactions(
        Mode::Streaming,
        [start(0, thought.clone()), stop(0), completed()],
    )
    .expect("streamed thought decodes");
    for response in [&unary, &streamed] {
        let [AssistantContent::CarriedReasoning(part)] = response.choice.as_slice() else {
            panic!("one carried thought");
        };
        assert_eq!(part.payload.item, thought);
        assert_eq!(part.payload.provider.as_deref(), Some("gcp.gemini"));
        assert_eq!(
            serde_json::to_value(part.payload.position).unwrap(),
            json!({"kind":"whole"})
        );
    }
    assert_eq!(
        unary.message(),
        streamed.message(),
        "a whole reply and its restatement as a stream fold into the same turn"
    );
}

#[test]
fn every_content_kind_decodes() {
    #[deny(clippy::wildcard_enum_match_arm)]
    fn variant_index(kind: ContentKind) -> usize {
        match kind {
            ContentKind::Text => 0,
            ContentKind::Image => 1,
            ContentKind::Other => 2,
        }
    }
    let samples = [
        json!({"type": "text", "text": "hello"}),
        json!({"type": "image", "uri": "https://example.com/a.png", "mime_type": "image/png"}),
        json!({"type": "image", "mime_type": "image/png"}),
        json!({"type": "audio", "data": "YQ==", "mime_type": "audio/wav"}),
        json!({"type": "video", "uri": "https://example.com/a.mp4"}),
        json!({"type": "document", "data": "YQ==", "mime_type": "application/pdf"}),
        json!({"type": "x_rig_invented"}),
    ];
    assert_every_variant(&samples, |item| variant_index(content_kind(item)), 3);
    for item in &samples {
        let step = json!({"type": "model_output", "content": [item]});
        let response = decode_interactions(Mode::Unary, [whole(vec![step.clone()])])
            .expect("the sample decodes");
        let block = response.choice.first();
        let expected = match (content_kind(item), item.get("uri").or(item.get("data"))) {
            (ContentKind::Text, _) => matches!(block, Some(AssistantContent::Text(_))),
            // CF005: every recognized model-output image is inspection-only.
            (ContentKind::Image, _) => matches!(block, Some(AssistantContent::CarriedText(part))
                if matches!(&part.content, CarriedTextContent::InspectionOnlyMedia(text) if text.is_empty())
                && part.payload.item == *item
                && part.payload.provider.as_deref() == Some("gcp.gemini")
                && part.payload.position == (CarriedPosition::MessageContent { item_index: 0, part_index: 0 })),
            (ContentKind::Other, _) => {
                matches!(block, Some(AssistantContent::Opaque(opaque)) if opaque.replay && opaque.item == step)
            }
        };
        assert!(expected, "{item} decoded to {block:?}");
        match content_kind(item) {
            ContentKind::Image => assert_eq!(
                replayed(&response),
                [json!({"type":"model_output", "content":[{"type":"text", "text":""}]})],
                "CF005: no model-output image raw value is replayed"
            ),
            ContentKind::Text | ContentKind::Other => assert_eq!(replayed(&response), [step]),
        }
    }
}

#[test]
fn annotations_cite_their_text_item_in_both_modes() {
    let text = "Spain won 2\u{2013}1 in Berlin.";
    let at = |part: &str| text.find(part).expect("the text holds it");
    let span = |part: &str| json!({"start_index": at(part), "end_index": at(part) + part.len()});
    let annotation = |part: &str, fields: serde_json::Value| {
        let mut annotation = span(part);
        if let (Some(annotation), serde_json::Value::Object(fields)) =
            (annotation.as_object_mut(), fields)
        {
            annotation.extend(fields);
        }
        annotation
    };
    let annotations = json!([
        annotation(
            "in Berlin.",
            json!({"type": "url_citation", "url": "https://example.com", "title": "example.com"})
        ),
        annotation(
            "Spain won",
            json!({"type": "file_citation", "document_uri": "files/report", "file_name": "report.pdf", "page_number": 3})
        ),
        annotation(
            "Berlin",
            json!({"type": "place_citation", "place_id": "places/berlin", "name": "Berlin"})
        ),
        annotation("Spain", json!({"type": "word_info"})),
    ]);
    let output = json!({"type": "model_output", "content": [{"type": "text", "text": text, "annotations": annotations}]});
    let streamed = [
        start(0, json!({"type": "model_output"})),
        delta(0, json!({"type": "text", "text": &text[..at("1 in")]})),
        delta(0, json!({"type": "text", "text": &text[at("1 in")..]})),
        delta(
            0,
            json!({"type": "text_annotation_delta", "annotations": annotations}),
        ),
        stop(0),
        completed(),
    ];
    let unary = decode_interactions(Mode::Unary, [whole(vec![output.clone()])])
        .expect("whole annotated text decodes");
    let response = decode_interactions(Mode::Streaming, streamed).expect("the stream decodes");
    assert_eq!(
        unary.message(),
        response.message(),
        "a whole reply and its restatement as a stream fold into the same turn"
    );
    let [AssistantContent::Text(block)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    let cited: Vec<_> = block
        .citations()
        .iter()
        .map(|citation| (block.cited(citation), citation.sources.clone()))
        .collect();
    let url = Source::new(SourceLocation::Url {
        url: "https://example.com".to_owned(),
    })
    .title("example.com");
    let file = Source::new(SourceLocation::Document {
        index: None,
        id: Some("files/report".to_owned()),
        within: Some(DocumentRange::Pages(3..4)),
    })
    .title("report.pdf");
    let place = Source::new(SourceLocation::Document {
        index: None,
        id: Some("places/berlin".to_owned()),
        within: None,
    })
    .title("Berlin");
    assert_eq!(
        cited,
        [
            (Some("in Berlin."), vec![url]),
            (Some("Spain won"), vec![file]),
            (Some("Berlin"), vec![place]),
        ]
    );
    assert_eq!(replayed(&response), [output]);
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_late_arguments_delta_cannot_reopen_a_full_known_call() {
    let (items, stream) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"location\":\"Rome\"}"}}"#,
        r#"{"event_type":"step.stop","index":0}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ]).await;
    let [
        Ok(Item::Event(StreamEvent::Start {
            part: start,
            kind: crate::streaming::PartKind::ToolCall,
            ..
        })),
        Ok(Item::Event(StreamEvent::Arguments {
            part: arguments,
            json,
        })),
        Ok(Item::Event(StreamEvent::End {
            part: end,
            content: AssistantContent::ToolCall(call),
            ..
        })),
    ] = items.as_slice()
    else {
        panic!("the finalized call must remain the only event sequence: {items:?}");
    };
    assert_eq!(start, arguments);
    assert_eq!(start, end);
    assert_eq!(
        call.id
            .provider()
            .map(crate::message::ProviderCallId::as_str),
        Some("call-1")
    );
    assert_eq!(call.function.name.as_str(), "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(json).unwrap(),
        json!({"location":"Paris"})
    );
    assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
    assert_eq!(call.native, None);
    let response = stream
        .finish()
        .await
        .expect("the actual completed event finishes the reply");
    assert_eq!(
        response.choice,
        vec![AssistantContent::ToolCall(call.clone())]
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

#[test]
fn cf_full_delta_partial_start_keeps_one_part_until_stop() {
    use crate::wire::{Decoder, Wire, WireEvent};

    let wire = wire();
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let mut decoder = wire.decoder_in(Mode::Streaming);
    for event in [
        json!({"event_type":"step.start","index":0,"step":{"type":"function_call","name":"get_weather","id":"call-1"}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"function_call","name":"get_weather","id":"call-1","arguments":{"location":"Paris"}}}),
    ] {
        let WireEvent::Known(event) = decoder.classify(frame(event)) else {
            panic!("known call frame");
        };
        let _ = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect("the existing open slot is updated");
        let state = shared.lock().expect("reply state");
        assert_eq!(state.items.len(), 1);
        assert!(matches!(
            state.items.front(),
            Some(Ok(Item::Event(StreamEvent::Start {
                kind: crate::streaming::PartKind::ToolCall,
                ..
            })))
        ));
        assert!(state.end.is_none());
    }
    // Named intentional difference: F emitted another same-wire-ID pair;
    // this canonical slot stays open until its genuine step stop.
    let WireEvent::Known(stop) =
        decoder.classify(frame(json!({"event_type":"step.stop","index":0})))
    else {
        panic!("known stop");
    };
    let _ = decoder
        .decode(stop, crate::wire::Out::new(&shared))
        .expect("the stop closes the same slot");
    let state = shared.lock().expect("reply state");
    let items: Vec<_> = state.items.iter().collect();
    let [
        Ok(Item::Event(StreamEvent::Start { part: start, .. })),
        Ok(Item::Event(StreamEvent::Arguments {
            part: arguments,
            json,
        })),
        Ok(Item::Event(StreamEvent::End {
            part: end,
            content: AssistantContent::ToolCall(call),
            ..
        })),
    ] = items.as_slice()
    else {
        panic!("one Part only: {items:?}");
    };
    assert_eq!(start, arguments);
    assert_eq!(start, end);
    assert_eq!(
        call.id
            .provider()
            .map(crate::message::ProviderCallId::as_str),
        Some("call-1")
    );
    assert_eq!(call.function.name.as_str(), "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(json).unwrap(),
        json!({"location":"Paris"})
    );
    assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
    assert!(state.end.is_none());
}

#[test]
fn a_full_kind_call_without_an_id_keeps_the_completed_hook_refusal() {
    use crate::wire::{Decoder, WireEvent};

    let wire = wire();
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let mut decoder = InteractionsDecoder::default().in_mode(Mode::Streaming);
    let WireEvent::Known(full) = decoder.classify(frame(json!({
        "event_type":"step.delta","index":0,"delta":{
            "type":"function_call","name":"get_weather","arguments":{"location":"Paris"}
        }
    }))) else {
        panic!("known full-kind frame");
    };
    let _ = decoder
        .decode(full, crate::wire::Out::new(&shared))
        .expect("no new early missing-ID refusal");
    let Some(Open::Call { block, .. }) = decoder.steps.get(&0) else {
        panic!("the existing pending slot");
    };
    let expected = format!(
        "the reply finalized item {block} at a provider terminal without a named function call and retained ID"
    );
    assert!(!decoder.call_ids_by_step.contains_key(&0));
    assert!(decoder.closed_full_calls.is_empty());
    let WireEvent::Known(completed) = decoder.classify(frame(json!({
        "event_type":"interaction.completed","interaction":{"status":"completed"}
    }))) else {
        panic!("known completed frame");
    };
    let error = decoder
        .decode(completed, crate::wire::Out::new(&shared))
        .expect_err("the existing retained-ID gate refuses completion");
    assert_eq!(error.kind(), crate::error::ErrorKind::Response);
    assert!(!error.is_retryable());
    assert!(matches!(&error, ProviderError::Response(reason) if reason == &expected));
    let state = shared.lock().expect("reply state");
    assert!(state.items.iter().all(|item| !matches!(
        item,
        Ok(Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(_),
            ..
        }))
    )));
    assert!(state.end.is_none());
}

#[test]
fn cf_full_kind_response_text_refusal_intentionally_loses_typed_value_form() {
    use crate::wire::{Decoder, Wire, WireEvent};

    // Intentional difference from F: this full-kind path refuses nonobjects.
    // The existing Response carrier retains the exact values only as text.
    for (arguments, expected_json) in [
        (json!(12345), "12345"),
        (json!(-1), "-1"),
        (Value::Null, "null"),
        (json!(true), "true"),
        (json!(false), "false"),
        (json!([1, 2]), "[1,2]"),
        (json!("quoted"), "\"quoted\""),
        (
            json!("{\"location\":\"Paris\"}"),
            "\"{\\\"location\\\":\\\"Paris\\\"}\"",
        ),
    ] {
        let wire = wire();
        let request = crate::completion::CompletionRequest::new("hello");
        let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
            crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
        ));
        let mut decoder = wire.decoder_in(Mode::Streaming);
        let event_json = json!({"event_type":"step.delta","index":0,"delta":{
            "type":"function_call","name":"get_weather","id":"call-1","arguments":arguments
        }});
        let WireEvent::Known(event) = decoder.classify(frame(event_json.clone())) else {
            panic!("known full-kind event");
        };
        let error = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect_err("the full-kind nonobject is intentionally refused");
        let expected = format!(
            "interactions full-kind function call arguments are not a JSON object: call_id=call-1 arguments={expected_json}"
        );
        assert!(matches!(&error, ProviderError::Response(message) if message == &expected));
        assert_eq!(error.to_string(), format!("ResponseError: {expected}"));
        assert_eq!(error.kind(), crate::error::ErrorKind::Response);
        assert!(!error.is_retryable());
        assert!(
            std::error::Error::source(&error).is_none(),
            "no typed Value or call-ID source survives this carrier"
        );
        assert_eq!(event_json["delta"]["arguments"].to_string(), expected_json);
        let state = shared.lock().expect("reply state");
        assert!(state.items.is_empty());
        assert!(state.end.is_none());
    }
}

#[test]
fn the_full_kind_refusal_does_not_change_announced_values_on_step_start() {
    use crate::wire::{Decoder, Wire, WireEvent};

    let wire = wire();
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let mut decoder = wire.decoder_in(Mode::Streaming);
    for event in [
        json!({"event_type":"step.start","index":0,"step":{
            "type":"function_call","name":"get_weather","id":"call-1","arguments":12345
        }}),
        json!({"event_type":"step.stop","index":0}),
    ] {
        let WireEvent::Known(event) = decoder.classify(frame(event)) else {
            panic!("known step event");
        };
        let _ = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect("the other announced-Value path stays unchanged");
    }
    let state = shared.lock().expect("reply state");
    let items: Vec<_> = state.items.iter().collect();
    let [
        Ok(Item::Event(StreamEvent::Start { part: start, .. })),
        Ok(Item::Event(StreamEvent::Arguments {
            part: arguments,
            json,
        })),
        Ok(Item::Event(StreamEvent::End {
            part: end,
            content: AssistantContent::ToolCall(call),
            ..
        })),
    ] = items.as_slice()
    else {
        panic!("the original announced projection: {items:?}");
    };
    assert_eq!(start, arguments);
    assert_eq!(start, end);
    assert_eq!(
        call.id
            .provider()
            .map(crate::message::ProviderCallId::as_str),
        Some("call-1")
    );
    assert_eq!(call.function.arguments_value(), json!({}));
    assert_eq!(call.function.invalid_arguments.as_deref(), Some("12345"));
    assert_eq!(*json, "{}");
    assert!(state.end.is_none());
}

fn closed_full_prefix(
    events: &[Value],
) -> (
    InteractionsDecoder,
    std::sync::Mutex<crate::wire::Shared<Completion>>,
) {
    use crate::wire::{Decoder, WireEvent};
    let wire = wire();
    let request = crate::completion::CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let mut decoder = InteractionsDecoder::default().in_mode(Mode::Streaming);
    for event in events {
        let WireEvent::Known(event) = decoder.classify(frame(event.clone())) else {
            panic!("known received frame");
        };
        let _ = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect("valid prefix");
    }
    (decoder, shared)
}

fn closed_full_prior_events(
    shared: &std::sync::Mutex<crate::wire::Shared<Completion>>,
) -> Vec<StreamEvent> {
    shared
        .lock()
        .expect("reply state")
        .items
        .iter()
        .map(|item| match item {
            Ok(Item::Event(event)) => event.clone(),
            other => panic!("only genuine prefix events: {other:?}"),
        })
        .collect()
}

// CLOSED-STEP-REFUSAL-TEXT-TYPED-FORM-LOSS: these deliberate F divergences
// retain received JSON only as text in Response, with no typed delta source.
#[test]
fn closed_full_refusals_keep_exact_received_text_and_prior_part_events() {
    use crate::wire::{Decoder, WireEvent};
    use std::error::Error as _;
    let full: Value = serde_json::from_str(
        r#"{"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}"#,
    ).expect("ordered received delta");
    for (index, received, expected) in [
        (
            0,
            full.clone(),
            r#"interactions: second full function call for closed call call-1; received {"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}"#,
        ),
        (
            7,
            full.clone(),
            r#"interactions: second full function call for closed call call-1; received {"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}"#,
        ),
        (
            0,
            json!({"type":"text","text":"after"}),
            r#"interactions: text fragment at index 0 belongs to closed call call-1; received {"type":"text","text":"after"}"#,
        ),
        (
            1,
            json!({"type":"arguments_delta","arguments":"{\"late\":true}"}),
            r#"interactions: arguments fragment at index 1 has no open call; received {"type":"arguments_delta","arguments":"{\"late\":true}"}"#,
        ),
    ] {
        let (mut decoder, shared) = closed_full_prefix(&[
            json!({"event_type":"step.delta","index":0,"delta":full.clone()}),
        ]);
        let prior = closed_full_prior_events(&shared);
        let [
            StreamEvent::Start {
                part: start,
                kind: crate::streaming::PartKind::ToolCall,
                ..
            },
            StreamEvent::Arguments {
                part: arguments,
                json: raw,
            },
            StreamEvent::End {
                part: end,
                content: AssistantContent::ToolCall(call),
                ..
            },
        ] = prior.as_slice()
        else {
            panic!("one finalized Part: {prior:?}");
        };
        assert_eq!(start, arguments);
        assert_eq!(start, end);
        assert_eq!(
            call.id
                .provider()
                .map(crate::message::ProviderCallId::as_str),
            Some("call-1")
        );
        assert_eq!(call.function.name.as_str(), "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(raw).expect("arguments"),
            json!({"location":"Paris"})
        );
        assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
        assert_eq!(call.native, None);
        let ids = decoder.call_ids_by_step.clone();
        let closed = decoder.closed_full_calls.clone();
        let received_copy = received.clone();
        let WireEvent::Known(event) = decoder.classify(frame(
            json!({"event_type":"step.delta","index":index,"delta":received}),
        )) else {
            panic!("known received refusal frame");
        };
        let error = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect_err("named closed-step refusal");
        let ProviderError::Response(message) = &error else {
            panic!("Response carrier: {error:?}");
        };
        assert_eq!(message, expected);
        assert!(message.ends_with(&serde_json::to_string(&received_copy).expect("compact delta")));
        assert_eq!(error.kind(), crate::error::ErrorKind::Response);
        assert!(!error.is_retryable());
        assert!(
            error.source().is_none(),
            "typed received form is deliberately lost"
        );
        assert_eq!(closed_full_prior_events(&shared), prior);
        assert_eq!(decoder.call_ids_by_step, ids);
        assert_eq!(decoder.closed_full_calls, closed);
        assert!(decoder.steps.is_empty());
        assert!(shared.lock().expect("reply state").end.is_none());
    }
}

// Legitimacy of a provider's orphan/resumed fragment remains UNKNOWN.
// This pins the admitted no-open-call/no-association refusal, not live traffic.
#[test]
fn an_orphan_resumed_arguments_fragment_refuses_without_creating_a_call_or_id() {
    use crate::wire::{Decoder, WireEvent};
    let (mut decoder, shared) = closed_full_prefix(&[]);
    let received = json!({"type":"arguments_delta","arguments":"{\"late\":true}"});
    let WireEvent::Known(event) = decoder.classify(frame(
        json!({"event_type":"step.delta","index":3,"delta":received}),
    )) else {
        panic!("known arguments frame");
    };
    let error = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect_err("the received fragment has neither prerequisite");
    let ProviderError::Response(message) = error else {
        panic!("Response carrier: {error:?}");
    };
    assert_eq!(
        message,
        r#"interactions: arguments fragment at index 3 has no open call; received {"type":"arguments_delta","arguments":"{\"late\":true}"}"#
    );
    assert!(decoder.steps.is_empty());
    assert!(decoder.call_ids_by_step.is_empty());
    assert!(decoder.closed_full_calls.is_empty());
    assert!(decoder.full_updates_by_step.is_empty());
    assert!(closed_full_prior_events(&shared).is_empty());
}

#[test]
fn an_announced_full_updated_call_closes_only_at_successful_stop_and_ignores_late_arguments() {
    use crate::wire::{Decoder, WireEvent};
    let id = crate::message::ProviderCallId::new("call-1").expect("genuine id");
    let (mut decoder, shared) = closed_full_prefix(&[
        json!({"event_type":"step.start","index":0,"step":{"type":"function_call","name":"get_weather","id":"call-1"}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"function_call","name":"get_weather","id":"call-1","arguments":{"location":"Paris"}}}),
    ]);
    assert!(!decoder.closed_full_calls.contains(&id));
    assert_eq!(decoder.full_updates_by_step.get(&0), Some(&id));
    let WireEvent::Known(event) =
        decoder.classify(frame(json!({"event_type":"step.stop","index":0})))
    else {
        panic!("known stop");
    };
    let _ = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect("existing stop finalizer succeeds");
    assert!(decoder.closed_full_calls.contains(&id));
    assert!(decoder.full_updates_by_step.is_empty());
    let prior = closed_full_prior_events(&shared);
    let [
        StreamEvent::Start { part: start, .. },
        StreamEvent::Arguments {
            part: arguments,
            json: raw,
        },
        StreamEvent::End {
            part: end,
            content: AssistantContent::ToolCall(call),
            ..
        },
    ] = prior.as_slice()
    else {
        panic!("same Part closes: {prior:?}");
    };
    assert_eq!(start, arguments);
    assert_eq!(start, end);
    assert_eq!(call.id.provider(), Some(&id));
    assert_eq!(call.function.name.as_str(), "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(raw).expect("arguments"),
        json!({"location":"Paris"})
    );
    assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
    for received in [
        json!({"type":"arguments_delta","arguments":"{\"location\":\"Rome\"}"}),
        json!({"type":"arguments_delta","arguments":null}),
    ] {
        let WireEvent::Known(event) = decoder.classify(frame(
            json!({"event_type":"step.delta","index":0,"delta":received}),
        )) else {
            panic!("known late fragment");
        };
        let _ = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect("existing closed-ID ignore");
        assert_eq!(closed_full_prior_events(&shared), prior);
        assert!(decoder.steps.is_empty());
    }
    // The same closed-step refusals also apply after announced/full stop.
    let ids = decoder.call_ids_by_step.clone();
    let closed = decoder.closed_full_calls.clone();
    let full_updates = decoder.full_updates_by_step.clone();
    let full: Value = serde_json::from_str(
        r#"{"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}"#,
    ).expect("ordered received delta");
    for (received, expected) in [
        (
            full,
            r#"interactions: second full function call for closed call call-1; received {"type":"function_call","name":"get_weather","arguments":{"location":"Paris"},"id":"call-1"}"#,
        ),
        (
            json!({"type":"text","text":"after"}),
            r#"interactions: text fragment at index 0 belongs to closed call call-1; received {"type":"text","text":"after"}"#,
        ),
    ] {
        let received_copy = received.clone();
        let WireEvent::Known(event) = decoder.classify(frame(
            json!({"event_type":"step.delta","index":0,"delta":received}),
        )) else {
            panic!("known received refusal frame");
        };
        let error = decoder
            .decode(event, crate::wire::Out::new(&shared))
            .expect_err("closed announced/full-updated call refuses");
        let ProviderError::Response(message) = &error else {
            panic!("Response carrier: {error:?}");
        };
        assert_eq!(message, expected);
        assert!(message.ends_with(&serde_json::to_string(&received_copy).expect("compact delta")));
        assert_eq!(error.kind(), crate::error::ErrorKind::Response);
        assert!(!error.is_retryable());
        assert!(
            std::error::Error::source(&error).is_none(),
            "typed received form is deliberately lost"
        );
        assert_eq!(closed_full_prior_events(&shared), prior);
        assert_eq!(decoder.call_ids_by_step, ids);
        assert_eq!(decoder.closed_full_calls, closed);
        assert_eq!(decoder.full_updates_by_step, full_updates);
        assert!(decoder.steps.is_empty());
        assert!(shared.lock().expect("reply state").end.is_none());
    }
    assert!(shared.lock().expect("reply state").end.is_none());
}

#[test]
fn a_failed_stop_does_not_mark_a_full_updated_call_closed() {
    use crate::wire::{Decoder, WireEvent};
    let id = crate::message::ProviderCallId::new("call-1").expect("genuine id");
    let (mut decoder, shared) = closed_full_prefix(&[
        json!({"event_type":"step.start","index":0,"step":{"type":"function_call","name":"get_weather","id":"call-1"}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"function_call","name":"get_weather","id":"call-1","arguments":{"location":"Paris"}}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{not-json"}}),
    ]);
    let prior = closed_full_prior_events(&shared);
    let WireEvent::Known(event) =
        decoder.classify(frame(json!({"event_type":"step.stop","index":0})))
    else {
        panic!("known stop");
    };
    let error = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect_err("failed existing finalizer");
    assert_eq!(error.kind(), crate::error::ErrorKind::Response);
    assert!(!decoder.closed_full_calls.contains(&id));
    assert!(decoder.full_updates_by_step.is_empty());
    assert_eq!(closed_full_prior_events(&shared), prior);
    assert!(shared.lock().expect("reply state").end.is_none());
}

#[test]
fn ordinary_associated_partial_arguments_and_open_missing_id_partials_are_unchanged() {
    use crate::wire::{Decoder, WireEvent};
    let id = crate::message::ProviderCallId::new("call-1").expect("genuine id");
    let (mut decoder, shared) = closed_full_prefix(&[
        json!({"event_type":"step.start","index":0,"step":{"type":"function_call","name":"get_weather","id":"call-1"}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"location\":\"Paris\"}"}}),
        json!({"event_type":"step.stop","index":0}),
    ]);
    assert!(
        !decoder.closed_full_calls.contains(&id),
        "no general all-call closure"
    );
    assert!(decoder.full_updates_by_step.is_empty());
    assert_eq!(decoder.call_ids_by_step.get(&0), Some(&id));
    let events = closed_full_prior_events(&shared);
    let Some(StreamEvent::End {
        content: AssistantContent::ToolCall(call),
        ..
    }) = events.last()
    else {
        panic!("ordinary partial completes: {events:?}");
    };
    assert_eq!(call.id.provider(), Some(&id));
    assert_eq!(call.function.arguments_value(), json!({"location":"Paris"}));
    // Stopped ordinary association stays outside the full-updated closure scope.
    // Its existing implicit-start behavior is pinned, not provider legitimacy.
    let WireEvent::Known(event) = decoder.classify(frame(json!({
        "event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"after\":true}"},
    }))) else { panic!("known ordinary associated fragment"); };
    let _ = decoder
        .decode(event, crate::wire::Out::new(&shared))
        .expect("existing default subject unchanged");
    assert!(matches!(decoder.steps.get(&0), Some(Open::Call { .. })));
    assert_eq!(decoder.call_ids_by_step.get(&0), Some(&id));
    assert!(!decoder.closed_full_calls.contains(&id));

    let (mut pending, pending_shared) = closed_full_prefix(&[
        json!({"event_type":"step.start","index":1,"step":{"type":"function_call","name":"get_weather"}}),
        json!({"event_type":"step.delta","index":1,"delta":{"type":"arguments_delta","arguments":"{\"location\":\"Paris\"}"}}),
    ]);
    assert!(matches!(pending.steps.get(&1), Some(Open::Call { .. })));
    assert!(!pending.call_ids_by_step.contains_key(&1));
    let prior = closed_full_prior_events(&pending_shared);
    let WireEvent::Known(event) = pending.classify(frame(json!({
        "event_type":"step.delta","index":1,"delta":{"type":"arguments_delta","arguments":" "},
    }))) else {
        panic!("known open partial fragment");
    };
    let _ = pending
        .decode(event, crate::wire::Out::new(&pending_shared))
        .expect("existing open partial unchanged");
    assert!(!pending.call_ids_by_step.contains_key(&1));
    let after = closed_full_prior_events(&pending_shared);
    assert_eq!(after.len(), prior.len() + 1);
    let (last, prefix) = after.split_last().expect("one new whitespace delta");
    assert_eq!(prefix, prior.as_slice());
    let [
        StreamEvent::Start { part: start, .. },
        StreamEvent::Arguments {
            part: arguments, ..
        },
    ] = prior.as_slice()
    else {
        panic!("existing partial Part: {prior:?}");
    };
    assert_eq!(start, arguments);
    let StreamEvent::Arguments { part, json } = last else {
        panic!("actual whitespace Arguments: {last:?}");
    };
    assert_eq!(part, start);
    assert_eq!(json, " ");
    assert!(pending.closed_full_calls.is_empty());
}
