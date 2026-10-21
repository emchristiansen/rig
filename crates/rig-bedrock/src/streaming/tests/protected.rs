use super::*;
use rig_core::error::{ErrorDetail, ErrorKind};

fn announced(index: usize, id: &str, name: &str, input: &str) -> [Value; 2] {
    [
        start(index, json!({"toolUse": {"toolUseId": id, "name": name}})),
        delta(index, json!({"toolUse": {"input": input}})),
    ]
}

#[test]
fn message_stop_flushes_stragglers_missing_a_block_stop() {
    let mut events = Vec::from(announced(0, "call_a", "get_weather", "{\"location\":\"Paris\"}"));
    events.extend(announced(1, "call_b", "get_time", "{\"zone\":\"UTC\"}"));
    events.extend(ended("tool_use"));
    let response = streamed(CLAUDE, events).expect("calls finalize at ToolUse");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.provider().map(|id| id.as_str()), Some("call_a"));
    assert_eq!(calls[1].id.provider().map(|id| id.as_str()), Some("call_b"));
    assert_eq!(calls[0].function.name.as_str(), "get_weather");
    assert_eq!(calls[0].function.arguments_value(), json!({"location": "Paris"}));
    assert_eq!(calls[1].function.name.as_str(), "get_time");
    assert_eq!(calls[1].function.arguments_value(), json!({"zone": "UTC"}));
    assert!(calls.iter().all(|call| call.native.is_none()));
    assert_eq!(response.stop(), StopReason::ToolUse);
}

#[test]
fn malformed_tool_json_surfaces_an_error_item() {
    let raw = "{\"location\": not-json";
    let mut events = Vec::from(announced(0, "call_a", "get_weather", raw));
    events.push(stop(0));
    events.extend(ended("tool_use"));
    let error = streamed(CLAUDE, events).expect_err("malformed input refuses finalization");
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::Response);
    assert!(report.message.contains("get_weather"));
    let Some(ErrorDetail::MalformedToolInput(detail)) = report.detail else {
        panic!("typed malformed-input diagnostic");
    };
    assert_eq!(detail.name, "get_weather");
    assert_eq!(detail.raw, raw);
    assert!(!detail.error.is_empty());
    assert_eq!(detail.provider.as_ref().map(|e| e.call_id.as_str()), Some("call_a"));
}

#[test]
fn malformed_tool_use_terminal_preserves_exact_input_and_name() {
    let raw = "{\"location\": not-json";
    let mut events = Vec::from(announced(0, "call_a", "get_weather", raw));
    events.extend(ended("tool_use"));
    let report = streamed(CLAUDE, events).expect_err("terminal does not invent a valid call").report();
    assert_eq!(report.kind, ErrorKind::Response);
    assert!(report.message.contains("get_weather"));
    let Some(ErrorDetail::MalformedToolInput(detail)) = report.detail else {
        panic!("typed malformed-input diagnostic");
    };
    assert_eq!(detail.name, "get_weather");
    assert_eq!(detail.raw, raw);
    assert_eq!(detail.provider.as_ref().map(|e| e.call_id.as_str()), Some("call_a"));
}

#[test]
fn non_tool_use_terminal_leaves_missing_block_stop_partial() {
    for reason in ["end_turn", "max_tokens", "guardrail_intervened"] {
        let mut events = Vec::from(announced(0, "call_a", "get_weather", "{}"));
        events.extend(ended(reason));
        let response = streamed(CLAUDE, events).expect("terminal reason survives");
        assert_eq!(response.tool_calls().count(), 0);
    }
}

#[test]
fn bedrock_usage_from_streaming_response() {
    let usage = super::super::usage(&json!({
        "inputTokens": 448, "outputTokens": 68, "totalTokens": 516,
        "cacheReadInputTokens": 80, "cacheWriteInputTokens": 20,
    }));
    assert_eq!(usage.input_tokens, Some(448));
    assert_eq!(usage.output_tokens, Some(68));
    assert_eq!(usage.total_tokens, Some(516));
    assert_eq!(usage.cached_input_tokens, Some(80));
    assert_eq!(usage.cache_creation_input_tokens, Some(20));
    assert_eq!(usage.tool_use_prompt_tokens, None);
    assert_eq!(usage.reasoning_tokens, None);
}

fn carried_parts(response: &CompletionResponse) -> Vec<CarriedReasoningPart> {
    response.choice.iter().flat_map(|content| match content {
        AssistantContent::CarriedReasoning(reasoning) => reasoning.content.clone(),
        other => panic!("expected preserved reasoning, got {other:?}"),
    }).collect()
}

#[test]
fn redacted_reasoning_is_a_sibling_of_the_open_thinking_block() {
    const REDACTED_BLOB: &[u8] = b"\x00opaque-stream-ciphertext\xff";
    let mut events = vec![
        delta(0, json!({"reasoningContent":{"text":"visible thinking"}})),
        delta(0, json!({"reasoningContent":{"signature":"sig_1"}})),
        delta(0, json!({"reasoningContent":{"redactedContent": BASE64_STANDARD.encode(REDACTED_BLOB)}})),
        stop(0),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(CLAUDE, events).expect("reasoning survives");
    assert_eq!(carried_parts(&response), vec![
        CarriedReasoningPart::Text { text:"visible thinking".into(), signature:Some("sig_1".into()) },
        CarriedReasoningPart::Redacted { data:BASE64_STANDARD.encode(REDACTED_BLOB) },
    ]);
    assert_eq!(response.choice.len(), 2);
    assert_eq!(response.stop(), StopReason::EndTurn);
}

#[test]
fn signature_only_thinking_block_still_reaches_the_consumer() {
    let mut events = vec![delta(0, json!({"reasoningContent":{"signature":"sig-only"}})), stop(0)];
    events.extend(ended("end_turn"));
    assert_eq!(carried_parts(&streamed(CLAUDE, events).expect("signature survives")), vec![
        CarriedReasoningPart::Text { text:String::new(), signature:Some("sig-only".into()) },
    ]);
}

#[test]
fn an_empty_thinking_block_emits_no_reasoning_part() {
    let mut events = vec![delta(0, json!({"reasoningContent":{"text":""}})), stop(0)];
    events.extend(ended("end_turn"));
    assert!(streamed(CLAUDE, events).expect("terminal survives").choice.is_empty());
}

#[test]
fn reasoning_issuer_follows_the_model() {
    for (model, issuer) in [(CLAUDE, "anthropic"), (NOVA, "aws_bedrock")] {
        let response = whole(model, vec![reasoning("thinking", Some("sig"))], "end_turn");
        let AssistantContent::CarriedReasoning(reasoning) = &response.choice[0] else { panic!("carried reasoning"); };
        assert_eq!(reasoning.provider(), Some(issuer));
        assert_eq!(reasoning.content, vec![CarriedReasoningPart::Text { text:"thinking".into(), signature:Some("sig".into()) }]);
    }
}

#[derive(Clone)]
struct ReplyFrames(Vec<ConverseFrame>);

impl rig_core::driver::Transport<Converse> for ReplyFrames {
    fn send(&self, _payload: crate::completion::ConverseRequest, _exchange: rig_core::driver::Exchange) -> rig_core::driver::Opening<ConverseFrame> {
        rig_core::driver::Opening::ready(rig_core::driver::Opened::new(
            futures::stream::iter(self.0.clone().into_iter().map(Ok)),
        ))
    }
}

#[tokio::test]
async fn reasoning_issuer_uses_request_override_in_both_directions() {
    for (configured, requested, issuer) in [(NOVA, CLAUDE, "anthropic"), (CLAUDE, NOVA, "aws_bedrock")] {
        let model = rig_core::driver::Model::new(Converse::new(configured), ReplyFrames(vec![
            ConverseFrame::Whole(document(vec![reasoning("thinking", Some("sig"))], "end_turn")),
        ]));
        let mut request = rig_core::completion::CompletionRequest::new("hello");
        request.model = Some(requested.to_owned());
        let response = model.call(request).await.expect("local frame decodes");
        let AssistantContent::CarriedReasoning(reasoning) = &response.choice[0] else { panic!("carried reasoning"); };
        assert_eq!(reasoning.provider(), Some(issuer));
    }
}

#[tokio::test]
async fn original_invalid_user_image_refuses_before_adaptation() {
    let model = rig_core::driver::Model::new(Converse::new("meta.llama3-1-8b-instruct-v1:0"), ReplyFrames(Vec::new()));
    let image = rig_core::message::Image {
        data: DocumentSourceKind::Url("https://example.com/image.png".into()),
        media_type: Some(ImageMediaType::PNG),
        ..rig_core::message::Image::default()
    };
    let request = rig_core::completion::CompletionRequest::from(vec![rig_core::message::Message::User {
        content:vec![rig_core::message::UserContent::Image(image)],
    }]);
    let error = model.call(request).await.expect_err("original URL image must refuse");
    assert_eq!(error.kind(), ErrorKind::Request);
    assert!(error.to_string().contains("Only base64 encoded strings are allowed for image input on AWS Bedrock"));
}
