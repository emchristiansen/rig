use super::*;
use crate::completion::GenerateContent;
use rig_core::message::AssistantContent;
use rig_core::test_utils::history::decode;

fn response(parts: Vec<proto::Part>, reason: i32) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content { parts, role: "model".into() }),
            finish_reason: reason,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn text(text: &str) -> proto::Part {
    proto::Part { data: Some(proto::part::Data::Text(text.into())), ..Default::default() }
}

fn wire() -> GenerateContent { GenerateContent::new("gemini-2.5-flash") }

#[test]
fn sdk_raw_retains_snake_case_numeric_enums_and_zero_usage() {
    let mut reply = response(vec![text("answer")], proto::candidate::FinishReason::Stop as i32);
    reply.usage_metadata = Some(proto::UsageMetadata::default());
    reply.response_id = "response".into();
    let normalized = decode(&wire(), Mode::Unary, [reply.clone()]).expect("typed reply decodes");
    assert_eq!(normalized.raw, raw::response(&reply));
    assert_eq!(normalized.raw["candidates"][0]["finish_reason"], 1);
    assert_eq!(normalized.raw["response_id"], "response");
    let restored: proto::GenerateContentResponse = raw::read_response(&normalized.raw).expect("SDK roundtrip");
    assert_eq!(restored, reply);
    assert_eq!(normalized.usage, Usage {
        input_tokens: Some(0), output_tokens: Some(0), total_tokens: Some(0),
        cached_input_tokens: Some(0), ..Usage::default()
    });
}

#[test]
fn unary_images_are_canonical_and_stream_images_are_skipped() {
    let image = proto::Part { data: Some(proto::part::Data::InlineData(proto::Blob {
        mime_type: "image/png".into(), data: vec![1, 2, 3],
    })), ..Default::default() };
    let reply = response(vec![image.clone(), text("answer")], 1);
    let unary = decode(&wire(), Mode::Unary, [reply.clone()]).expect("unary image decodes");
    assert!(matches!(&unary.choice[0], AssistantContent::Image(image)
        if image.media_type == Some(rig_core::message::ImageMediaType::PNG)));
    let streamed = decode(&wire(), Mode::Streaming, [reply]).expect("stream skips image");
    assert_eq!(streamed.choice.len(), 1);
    assert_eq!(streamed.text(), "answer");
    let bad = proto::Part { data: Some(proto::part::Data::InlineData(proto::Blob {
        mime_type: "not/a-type".into(), data: vec![],
    })), ..Default::default() };
    let error = decode(&wire(), Mode::Unary, [response(vec![bad], 1)]).expect_err("unsupported unary MIME");
    assert!(matches!(error, ProviderError::Response(message) if message == "Unsupported media type None"));
}

#[test]
fn unary_candidate_and_content_diagnostics_are_exact() {
    let error = decode(&wire(), Mode::Unary, [proto::GenerateContentResponse::default()]).expect_err("candidate required");
    assert!(matches!(error, ProviderError::Response(message) if message == "No response candidates in response"));
    let reply = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate { finish_reason: 1, ..Default::default() }],
        ..Default::default()
    };
    let error = decode(&wire(), Mode::Unary, [reply]).expect_err("content required");
    assert!(matches!(error, ProviderError::Response(message) if message == "Gemini candidate missing content (finish_reason=1)"));
}

#[test]
fn unary_missing_function_handles_are_stable_and_count_functions_only() {
    let call = |id: Option<&str>, name: &str| proto::Part {
        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
            id: id.unwrap_or_default().to_owned(), name: name.into(), args: None,
        })), ..Default::default()
    };
    let reply = response(vec![text("before"), call(None, "first"), text("between"),
        call(Some("tool-0"), "second"), call(Some(""), "third")], 1);
    let first = decode(&wire(), Mode::Unary, [reply.clone()]).expect("call reply");
    let second = decode(&wire(), Mode::Unary, [reply]).expect("same call reply");
    assert_eq!(first.choice, second.choice);
    let calls: Vec<_> = first.choice.iter().filter_map(|part| match part {
        AssistantContent::ToolCall(call) => Some(call), _ => None,
    }).collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].id, CallId::Local(LocalCallId::for_gemini_grpc_response_position(0)));
    assert_eq!(calls[1].id.provider().map(|id| id.as_str()), Some("tool-0"));
    assert_eq!(calls[2].id, CallId::Local(LocalCallId::for_gemini_grpc_response_position(2)));
    assert!(calls.iter().all(|call| call.item_id.is_none()));
}

#[test]
fn a_streams_raw_is_its_genuine_terminal_sdk_frame() {
    let first = response(vec![text("hi")], 0);
    let terminal = response(vec![text("!")], 1);
    let normalized = decode(&wire(), Mode::Streaming, [first, terminal.clone()]).expect("terminal completes stream");
    assert_eq!(normalized.text(), "hi!");
    assert_eq!(normalized.raw, raw::response(&terminal));
}

#[test]
fn protobuf_tool_protocol_failure_keeps_the_exact_finish_message() {
    let mut reply = response(vec![text("unemitted")], proto::candidate::FinishReason::MalformedFunctionCall as i32);
    reply.candidates[0].finish_message = Some("broken call".into());
    for mode in [Mode::Unary, Mode::Streaming] {
        let error = decode(&wire(), mode, [reply.clone()]).expect_err("tool protocol fails");
        assert!(matches!(error, ProviderError::Response(message)
            if message == "Gemini stopped with finish_reason=MALFORMED_FUNCTION_CALL: broken call"));
    }
}
