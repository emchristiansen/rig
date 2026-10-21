use super::*;
use rig_core::message::{CarriedPayload, CarriedReasoning, CarriedReasoningPart, LocalCallId};

#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let generated = ToolCall::new(
        CallId::Local(LocalCallId::new()),
        ToolFunction::new(name("test"), json!({})),
    );
    let real = call("tool-0", "test", json!({}));
    let assistant = |call: &ToolCall| Message::Assistant(AssistantMessage::new(vec![
        AssistantContent::ToolCall(call.clone()),
    ]));
    let result = |call: &ToolCall| Message::User { content: vec![
        UserContent::ToolResult(call.result(vec![ToolResultContent::text("done")])),
    ] };
    let body = unprepared(vec![
        Message::system("system"),
        assistant(&generated),
        assistant(&real),
        result(&real),
        Message::assistant("intervening text"),
        result(&generated),
        assistant(&generated),
        result(&generated),
    ]).expect("the private encoder plans original coordinates");
    let parts = body["messages"].as_array().expect("messages").iter()
        .flat_map(|message| message["content"].as_array().expect("content"));
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for part in parts {
        if let Some(id) = part.pointer("/toolUse/toolUseId").and_then(Value::as_str) {
            calls.push(id.to_owned());
        }
        if let Some(id) = part.pointer("/toolResult/toolUseId").and_then(Value::as_str) {
            results.push(id.to_owned());
        }
    }
    assert_eq!(calls.len(), 3);
    assert_eq!(results, [calls[1].clone(), calls[0].clone(), calls[2].clone()]);
    assert_eq!(calls[1], "tool-0");
    assert_eq!(calls.iter().collect::<std::collections::HashSet<_>>().len(), 3);
}

#[test]
fn test_system_prompt_includes_system_history() {
    let body = unprepared(vec![
        Message::system("History system instruction"),
        Message::user("test"),
        Message::system("Later system instruction"),
    ]).expect("the private encoder hoists system blocks");
    assert_eq!(body["system"], json!([
        {"text": "History system instruction"},
        {"text": "Later system instruction"},
    ]));
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["messages"][0]["role"], "user");
}

#[test]
fn test_messages_skip_cache_point_when_history_contains_reasoning() {
    let reasoning = CarriedReasoning::new(
        vec![CarriedReasoningPart::Text { text: "thinking".into(), signature: Some("sig".into()) }],
        CarriedPayload::new(json!({"reasoningContent": {"reasoningText": {"text": "thinking", "signature": "sig"}}}), Some("anthropic".into())),
    );
    let request = CompletionRequest::from(vec![
        Message::system("System prompt"),
        Message::user("user prompt"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedReasoning(reasoning)])),
        Message::user("follow up"),
    ]).cache(CacheRetention::Short);
    let wire = Converse::new(CLAUDE);
    let body = serde_json::to_value(body(&wire, &request, CLAUDE).expect("reasoning encodes")).expect("body");
    assert_eq!(body["system"][1], json!({"cachePoint": {"type": "default"}}));
    let last = body["messages"].as_array().expect("messages").last().expect("last message");
    assert!(!last["content"].as_array().expect("content").iter().any(|part| part.get("cachePoint").is_some()),
        "message-level cache point should be skipped when chat history contains reasoning");
}

#[test]
fn document_names_are_deterministic_and_unique_within_a_request() {
    let document = || UserContent::Document(Document {
        data: DocumentData::Text("A repeated attachment.".into()),
        media_type: Some(DocumentMediaType::TXT), additional_params: None,
    });
    let history = vec![Message::User { content: vec![document(), document(), UserContent::text("question")] }];
    let first = unprepared(history.clone()).expect("documents encode");
    let second = unprepared(history).expect("documents encode again");
    assert_eq!(first, second);
    let names: Vec<_> = first["messages"][0]["content"].as_array().expect("content").iter()
        .filter_map(|block| block.pointer("/document/name").and_then(Value::as_str)).collect();
    assert_eq!(names.len(), 2);
    assert!(names.iter().all(|name| name.starts_with("document-")));
    assert_eq!(names[1], format!("{}-2", names[0]));
}

#[test]
fn test_document_to_aws_document() {
    let document = Document { data: DocumentData::Text("document_data".into()), media_type: Some(DocumentMediaType::PDF), additional_params: None };
    let block = super::super::document(&document).expect("document bytes encode");
    assert_eq!(block["format"], "pdf");
    assert_eq!(block["source"]["bytes"], BASE64_STANDARD.encode(b"document_data"));
    assert!(block["name"].as_str().is_some_and(|name| name.starts_with("document-")));
}

#[test]
fn test_unsupported_document_to_aws_document() {
    let document = Document { data: DocumentData::File(DocumentSourceKind::Base64("data".into())), media_type: Some(DocumentMediaType::Javascript), additional_params: None };
    let error = super::super::document(&document).expect_err("unsupported declared type");
    assert!(error.to_string().contains("Unsupported media type application/x-javascript"));
}

#[test]
fn test_unsupported_image_to_aws_image() {
    let image = Image { data: DocumentSourceKind::Base64(BASE64_STANDARD.encode("img_data")), media_type: Some(rig_core::message::ImageMediaType::HEIC), ..Image::default() };
    let error = super::super::image(&image).expect_err("unsupported image format");
    assert!(error.to_string().contains("Unsupported format image/heic"));
}

#[test]
fn message_to_aws_message() {
    let body = unprepared(vec![Message::user("text")]).expect("text encodes");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], json!([{ "text": "text" }]));
}

#[test]
fn user_content_to_aws_content_block() {
    let body = unprepared(vec![Message::user("txt")]).expect("text encodes");
    assert_eq!(body["messages"][0]["content"], json!([{ "text": "txt" }]));
}

#[test]
fn rig_tool_text_to_aws_tool() {
    let call = call("123", "lookup", json!({}));
    let body = unprepared(vec![
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())])),
        Message::User { content: vec![UserContent::ToolResult(call.result(vec![ToolResultContent::text("42")]))] },
    ]).expect("text result encodes");
    assert_eq!(body["messages"][1]["content"][0]["toolResult"]["content"], json!([{ "text": "42" }]));
}

#[test]
fn rig_tool_json_maps_to_native_aws_json() {
    let expected = json!({ "answer": -3, "exact": true });
    let call = call("123", "lookup", json!({}));
    let body = unprepared(vec![
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())])),
        Message::User { content: vec![UserContent::ToolResult(call.result(vec![ToolResultContent::Json { value: expected.clone() }]))] },
    ]).expect("JSON result encodes");
    assert_eq!(body["messages"][1]["content"][0]["toolResult"]["content"][0]["json"], expected);
}

#[test]
fn rig_tool_non_object_json_is_wrapped_for_bedrock() {
    for value in [Value::Null, json!(true), json!(-3), json!("literal text"), json!([1, 2, 3])] {
        let call = call("123", "lookup", json!({}));
        let body = unprepared(vec![
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())])),
            Message::User { content: vec![UserContent::ToolResult(call.result(vec![ToolResultContent::Json { value: value.clone() }]))] },
        ]).expect("JSON result encodes");
        assert_eq!(body["messages"][1]["content"][0]["toolResult"]["content"][0]["json"], json!({ "result": value }));
    }
}
