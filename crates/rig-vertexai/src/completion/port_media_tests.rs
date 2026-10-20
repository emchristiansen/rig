use super::*;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart, DocumentSourceKind, Image, ImageMediaType, Message, Text, ToolName,
    ToolResult, ToolResultContent, UserContent,
};
use rig_core::wire::Operation as _;
use serde_json::{Value, json};

fn name(value: &str) -> ToolName {
    ToolName::new(value).expect("the fixture name is nonempty")
}

fn direct_content(message: Message) -> Result<vertexai::model::Content, ProviderError> {
    let mut request = CompletionRequest::new("test");
    request.chat_history = vec![message];
    let encoded = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)
        .map_err(ProviderError::from)?;
    encoded
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::request("no content"))
}

fn assistant(parts: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage::new(parts))
}

fn carried_thought(text: &str, signature: &str) -> AssistantContent {
    AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::Text {
            text: text.to_owned(),
            signature: Some(signature.to_owned()),
        }],
        CarriedPayload::new(Value::Null, None),
    ))
}

#[test]
fn test_assistant_image_history_rejects_invalid_or_unsupported_input() {
    let cases = [
        (
            AssistantContent::image_base64(BASE64.encode([1]), None, None),
            "Media type",
        ),
        (
            AssistantContent::image_base64(BASE64.encode([1]), Some(ImageMediaType::GIF), None),
            "Unsupported",
        ),
        (
            AssistantContent::image_base64("not valid base64", Some(ImageMediaType::PNG), None),
            "Invalid base64",
        ),
    ];

    for (image, expected_message) in cases {
        let result: Result<vertexai::model::Content, ProviderError> =
            direct_content(assistant(vec![image]));
        let Err(error) = result else {
            panic!("invalid assistant image must fail")
        };
        assert!(matches!(error, ProviderError::Request(_)));
        assert!(error.to_string().contains(expected_message));
    }
}

/// G107 input exception: the signature comes from the real received SDK call,
/// not the retired caller-owned ToolCall signature field.
#[test]
fn test_assistant_tool_call_echoes_thought_signature() {
    let raw = b"\x00\x01\x02thinking-sig\xff";
    let call: vertexai::model::FunctionCall = serde_json::from_value(json!({
        "name": "add", "args": {"x": 5},
    }))
    .expect("the SDK call fixture reads");
    let part = vertexai::model::Part::new()
        .set_function_call(call)
        .set_thought_signature(raw.to_vec());
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new()
            .set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts([part.clone()]),
            )
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop),
    ]);
    let response = crate::types::completion_response::tests::complete(received)
        .expect("the actual SDK call decodes");
    let [block @ AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one real received function call is required");
    };
    assert_eq!(call.function.name.as_str(), "add");
    assert_eq!(call.function.arguments_value(), json!({"x": 5}));
    assert_eq!(
        block.native_item(),
        Some(&serde_json::to_value(&part).expect("part serializes"))
    );
    let content = direct_content(response.message().expect("the real turn supplies history"))
        .expect("the received call replays");
    assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
}

/// G107 input/stage exception: the original malformed call signature has no
/// canonical setter or SDK byte representation, so the existing private
/// pre-seal adjustment receives that exact string instead.
#[test]
fn test_assistant_tool_call_malformed_signature_is_dropped_not_fatal() {
    let source = json!({"role":"model","parts":[{
        "functionCall":{"name":"add","args":{"x":5}},
        "thoughtSignature":"!!! not base64 !!!",
    }]});
    let mut body = source.clone();
    standard_signatures(&mut body);
    let content: vertexai::model::Content = serde_json::from_value(body)
        .expect("malformed signature should not fail the private adjustment");
    assert_eq!(content.parts.len(), 1);
    assert!(content.parts[0].thought_signature.is_empty());
    assert!(content.parts[0].function_call().is_some());
    let call = content.parts[0]
        .function_call()
        .expect("the original call stays");
    assert_eq!(call.name, "add");
    assert_eq!(
        serde_json::to_value(&call.args).expect("args serialize"),
        json!({"x":5})
    );
    assert_eq!(
        source["parts"][0]["thoughtSignature"],
        json!("!!! not base64 !!!")
    );
}

#[test]
fn test_assistant_reasoning_echoes_thought_signature() {
    let raw = b"\x00\x01\x02thinking-text-sig\xff";
    let content = direct_content(assistant(vec![carried_thought(
        "thinking text",
        &BASE64.encode(raw),
    )]))
    .expect("known carried reasoning encodes");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(
        content.parts[0].text().map(String::as_str),
        Some("thinking text")
    );
    assert!(content.parts[0].thought);
    assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
}

#[test]
fn test_assistant_reasoning_malformed_signature_is_dropped_not_fatal() {
    let source = carried_thought("thinking text", "!!! not base64 !!!");
    let content = direct_content(assistant(vec![source.clone()]))
        .expect("malformed signature should not fail the conversion");
    assert_eq!(content.parts.len(), 1);
    assert!(content.parts[0].thought);
    assert!(content.parts[0].thought_signature.is_empty());
    let AssistantContent::CarriedReasoning(reasoning) = source else {
        unreachable!()
    };
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "thinking text".to_owned(),
            signature: Some("!!! not base64 !!!".to_owned()),
        }]
    );
}

#[test]
fn test_user_tool_result_message_conversion() {
    // Vertex results are id-less on the wire: a minted correlation
    // handle, no provider id, and the required executed-tool name that
    // becomes `functionResponse.name`.
    let tool_result = ToolResult {
        call: CallId::from_wire(""),
        name: name("add"),
        answers: rig_core::message::AnsweredToolCall::Function,
        is_error: false,
        content: vec![ToolResultContent::Text(Text::new("8".to_string()))],
    };

    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(tool_result)],
    };

    let vertex_content: Result<vertexai::model::Content, _> = direct_content(message);

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "user");
    assert_eq!(content.parts.len(), 1);

    let function_response = content.parts[0].function_response();
    assert!(function_response.is_some());
    let function_response = function_response.unwrap();
    assert_eq!(function_response.name.as_str(), "add");
    assert_eq!(
        function_response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::Value::String("8".to_string()))
    );
}

#[test]
fn structured_tool_result_stays_structured_at_the_vertex_boundary() {
    let value = serde_json::json!({ "answer": 8 });
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("lookup"),
            answers: rig_core::message::AnsweredToolCall::Function,
            is_error: false,
            content: vec![ToolResultContent::json(value.clone())],
        })],
    };

    let content: vertexai::model::Content =
        direct_content(message).expect("tool result should convert");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&value)
    );
}

#[test]
fn mixed_tool_result_preserves_structured_and_media_order() {
    let content = vec![
        ToolResultContent::text("before"),
        ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::JPEG), None),
        ToolResultContent::json(serde_json::json!({ "after": true })),
        ToolResultContent::image_url("gs://bucket/result.png", Some(ImageMediaType::PNG), None),
    ];
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("inspect"),
            answers: rig_core::message::AnsweredToolCall::Function,
            is_error: false,
            content,
        })],
    };

    let content: vertexai::model::Content =
        direct_content(message).expect("mixed tool result should convert");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!([
            "before",
            { "$ref": "rig_tool_result_image_0" },
            { "after": true },
            { "$ref": "rig_tool_result_image_1" }
        ]))
    );
    assert_eq!(response.parts.len(), 2);
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("first media part")
            .data
            .as_ref(),
        &[1, 2, 3]
    );
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("first media part")
            .display_name,
        "rig_tool_result_image_0"
    );
    assert_eq!(
        response.parts[1]
            .file_data()
            .expect("second media part")
            .file_uri,
        "gs://bucket/result.png"
    );
    assert_eq!(
        response.parts[1]
            .file_data()
            .expect("second media part")
            .display_name,
        "rig_tool_result_image_1"
    );
}

#[test]
fn tool_result_image_refs_avoid_names_reserved_by_structured_json() {
    let content = vec![
        ToolResultContent::json(serde_json::json!({
            "literal": { "$ref": "rig_tool_result_image_0" }
        })),
        ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::PNG), None),
    ];
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("inspect"),
            answers: rig_core::message::AnsweredToolCall::Function,
            is_error: false,
            content,
        })],
    };

    let content: vertexai::model::Content =
        direct_content(message).expect("colliding ref should be avoided");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!([
            { "literal": { "$ref": "rig_tool_result_image_0" } },
            { "$ref": "rig_tool_result_image_1" }
        ]))
    );
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("image part")
            .display_name,
        "rig_tool_result_image_1"
    );
}

#[test]
fn unsupported_tool_result_image_media_type_is_rejected_locally() {
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("inspect"),
            answers: rig_core::message::AnsweredToolCall::Function,
            is_error: false,
            content: vec![ToolResultContent::image_raw(
                vec![1, 2, 3],
                Some(ImageMediaType::GIF),
                None,
            )],
        })],
    };

    let error = direct_content(message)
        .expect_err("unsupported media must fail before the provider request");
    assert!(
        error.to_string().contains("expected JPEG, PNG, or WEBP"),
        "unexpected conversion error: {error}"
    );
}

#[test]
fn original_assistant_image_refusals_run_before_preparation_adapts_them() {
    let cases = [
        (
            AssistantContent::image_base64(BASE64.encode([1]), None, None),
            "Media type",
        ),
        (
            AssistantContent::image_base64(BASE64.encode([1]), Some(ImageMediaType::GIF), None),
            "Unsupported",
        ),
        (
            AssistantContent::image_base64("not valid base64", Some(ImageMediaType::PNG), None),
            "Invalid base64",
        ),
    ];
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    for (image, expected) in cases {
        let request = CompletionRequest::from(vec![
            Message::user("before"),
            assistant(vec![image]),
            Message::user("after"),
        ]);
        let error = Completion::prepare(request, &wire.describe())
            .expect_err("original image refuses before adaptation");
        assert!(matches!(error, ProviderError::Request(_)));
        assert!(error.to_string().contains(expected));
    }
}

/// G106 explicitly restores the Base64-only caller form: Raw and URL are
/// refused as canonical assistant inputs, in both direct and prepared paths.
#[test]
fn caller_raw_and_url_assistant_images_are_not_the_vertex_inline_form() {
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    for data in [
        DocumentSourceKind::Raw(vec![1]),
        DocumentSourceKind::Url("gs://bucket/image.png".to_owned()),
    ] {
        let image = AssistantContent::Image(Image {
            data,
            media_type: Some(ImageMediaType::PNG),
            ..Image::default()
        });
        let mut request = CompletionRequest::new("test");
        request.chat_history = vec![
            Message::user("before"),
            assistant(vec![image]),
            Message::user("after"),
        ];
        let direct = ProviderError::from(
            wire.encode(request.clone(), Mode::Unary)
                .expect_err("caller source refuses"),
        );
        let ProviderError::Request(direct) = direct else {
            panic!("direct Request refusal")
        };
        assert_eq!(
            direct.to_string(),
            "Vertex AI assistant images must use base64 data"
        );
        let prepared = Completion::prepare(request, &wire.describe())
            .expect_err("caller source refuses before adaptation");
        let ProviderError::Request(prepared) = prepared else {
            panic!("prepared Request refusal")
        };
        assert_eq!(
            prepared.to_string(),
            "Vertex AI assistant images must use base64 data"
        );
    }
}

#[test]
fn all_five_canonical_assistant_image_types_keep_their_inline_bytes() {
    for media_type in [
        ImageMediaType::JPEG,
        ImageMediaType::PNG,
        ImageMediaType::WEBP,
        ImageMediaType::HEIC,
        ImageMediaType::HEIF,
    ] {
        let content = direct_content(assistant(vec![AssistantContent::image_base64(
            BASE64.encode([1, 2, 3]),
            Some(media_type.clone()),
            None,
        )]))
        .expect("the canonical inline type encodes");
        let blob = content.parts[0].inline_data().expect("inline bytes");
        assert_eq!(
            blob.mime_type,
            rig_core::message::MimeType::to_mime_type(&media_type)
        );
        assert_eq!(blob.data.as_ref(), &[1, 2, 3]);
    }
}

#[test]
fn actual_received_native_image_keeps_its_part_under_the_private_validator() {
    let part = vertexai::model::Part::new().set_inline_data(
        vertexai::model::Blob::new()
            .set_mime_type("image/png")
            .set_data(vec![0, 1, 2, 255]),
    );
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new()
            .set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts([part.clone()]),
            )
            .set_finish_reason(vertexai::model::candidate::FinishReason::Stop),
    ]);
    let response = crate::types::completion_response::tests::complete(received)
        .expect("the native image decodes");
    let [block @ AssistantContent::Image(_)] = response.choice.as_slice() else {
        panic!("one genuine image")
    };
    assert_eq!(
        block.native_item(),
        Some(&serde_json::to_value(&part).expect("part serializes"))
    );
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let request = Completion::prepare(
        CompletionRequest::from(vec![
            Message::user("before"),
            response.message().expect("real history"),
            Message::user("after"),
        ]),
        &wire.describe(),
    )
    .expect("native custody prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("native custody encodes");
    assert_eq!(encoded.contents[1].parts, [part]);
}

#[test]
fn existing_signature_adjustment_keeps_standard_and_url_safe_bytes() {
    let bytes = [251, 255];
    for signature in [BASE64.encode(bytes), URL_SAFE.encode(bytes)] {
        let source =
            json!({"parts":[{"text":"thinking text","thought":true,"thoughtSignature":signature}]});
        let mut adjusted = source.clone();
        standard_signatures(&mut adjusted);
        let content: vertexai::model::Content =
            serde_json::from_value(adjusted).expect("valid bytes read through SDK");
        assert_eq!(
            content.parts[0].thought_signature.as_ref(),
            bytes.as_slice()
        );
        assert_eq!(source["parts"][0]["thoughtSignature"], json!(signature));
    }
}

#[test]
fn literal_raw_signature_after_merge_is_not_the_private_adjustment_input() {
    let mut request = CompletionRequest::new("test");
    let literal = json!({"contents":[{"role":"model","parts":[{
        "text":"thinking text","thought":true,"thoughtSignature":"!!! not base64 !!!"
    }]}]});
    request.additional_params = Some(literal.clone());
    let source = request.clone();
    let error = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)
        .expect_err("post-merge raw keeps the literal and SDK refuses it");
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::Request(_)
    ));
    assert_eq!(source.additional_params, Some(literal));
}

fn failed(content: Vec<ToolResultContent>) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(""),
            name: name("inspect"),
            answers: rig_core::message::AnsweredToolCall::Function,
            is_error: true,
            content,
        })],
    }
}

/// G120 keeps the existing failed-result path and its success-key quirks.
#[test]
fn failed_results_keep_current_error_keys_and_image_only_empty_output_quirks() {
    for value in [json!("8"), json!({"answer":8})] {
        let content = if value.is_string() {
            vec![ToolResultContent::text("8")]
        } else {
            vec![ToolResultContent::json(value.clone())]
        };
        let body = direct_content(failed(content)).expect("failed value encodes");
        let response = body.parts[0]
            .function_response()
            .expect("function response");
        assert_eq!(
            response.response.as_ref().and_then(|r| r.get("error")),
            Some(&value)
        );
        assert!(
            response
                .response
                .as_ref()
                .is_some_and(|r| r.get("output").is_none())
        );
    }
    let image = direct_content(failed(vec![ToolResultContent::image_base64(
        BASE64.encode([1, 2, 3]),
        Some(ImageMediaType::PNG),
        None,
    )]))
    .expect("failed Base64 image keeps the current path");
    let response = image.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response.response.as_ref().and_then(|r| r.get("output")),
        Some(&json!({"$ref":"rig_tool_result_image_0"}))
    );
    assert!(
        response
            .response
            .as_ref()
            .is_some_and(|r| r.get("error").is_none())
    );
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("image")
            .data
            .as_ref(),
        &[1, 2, 3]
    );
    let empty = direct_content(failed(vec![])).expect("empty failed result keeps current callback");
    let response = empty.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response.response.as_ref().and_then(|r| r.get("output")),
        Some(&json!([]))
    );
    assert!(
        response
            .response
            .as_ref()
            .is_some_and(|r| r.get("error").is_none())
    );
}

#[test]
fn failed_raw_and_url_images_keep_the_existing_typed_refusal() {
    for (content, expected) in [
        (
            ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::PNG), None),
            "Unsupported image source kind for tool results",
        ),
        (
            ToolResultContent::image_url("gs://bucket/result.png", Some(ImageMediaType::PNG), None),
            "Gemini tool result images must use base64 inline data; URL-backed images are not supported",
        ),
    ] {
        let error = direct_content(failed(vec![content])).expect_err("failed source refusal stays");
        let ProviderError::Request(source) = error else {
            panic!("existing typed Request refusal")
        };
        let Some(rig_core::message::MessageError::ConversionError(reason)) =
            source.downcast_ref::<rig_core::message::MessageError>()
        else {
            panic!("the existing MessageError source must stay typed")
        };
        assert_eq!(reason, expected);
    }
}

/// Vertex AI's v1 `FunctionResponse` declares no `id`: a provider handle,
/// like a local identity, never reaches the wire, on success or failure.
#[test]
fn function_responses_send_no_call_id_for_provider_or_local_identities() {
    for call in [CallId::from_wire("call_1"), CallId::from_wire("")] {
        for is_error in [false, true] {
            let body = direct_content(Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: call.clone(),
                    name: name("add"),
                    answers: rig_core::message::AnsweredToolCall::Function,
                    content: vec![ToolResultContent::text("8")],
                    is_error,
                })],
            })
            .expect("the result encodes");
            let encoded = serde_json::to_value(&body).expect("SDK content serializes");
            assert_eq!(
                encoded.pointer("/parts/0/functionResponse/id"),
                None,
                "{call:?} is_error={is_error}"
            );
            assert_eq!(
                encoded
                    .pointer("/parts/0/functionResponse/name")
                    .and_then(Value::as_str),
                Some("add")
            );
        }
    }
}
