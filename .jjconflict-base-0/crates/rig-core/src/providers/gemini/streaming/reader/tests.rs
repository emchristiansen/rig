use super::*;
use serde_json::json;

#[test]
fn test_usage_metadata_deserializes_without_total_token_count() {
    // Gemini's proto3-JSON encoding omits fields whose value is the default (0),
    // so `totalTokenCount` is absent on short/empty/blocked generations.
    let usage: Usage =
        serde_json::from_str(r#"{"promptTokenCount": 12}"#).expect("should deserialize");
    assert_eq!(usage.total_token_count, 0);
    assert_eq!(usage.prompt_token_count, 12);
}

#[test]
fn test_generate_content_response_deserializes_without_candidates_or_response_id_private_reader_defaults()
 {
    let body = json!({"promptFeedback":{"blockReason":"SAFETY"}});
    let response: Reply =
        serde_json::from_value(body).expect("blocked prompt response should deserialize");
    assert!(response.response_id.is_empty());
    assert!(response.candidates.is_empty());
}

#[test]
fn test_modality_token_count_deserializes_without_zero_token_count() {
    let count: ModalityCount = serde_json::from_value(json!({
        "modality": "TEXT"
    }))
    .expect("zero tokenCount may be omitted");

    assert_eq!(count.token_count, 0);
}

#[test]
fn test_response_metadata_repeated_fields_deserialize_when_omitted() {
    let citation_metadata: Citations =
        serde_json::from_value(json!({})).expect("empty citation metadata should deserialize");
    assert!(citation_metadata.citation_sources.is_empty());

    let logprobs: Logprobs =
        serde_json::from_value(json!({})).expect("empty logprobs result should deserialize");
    assert!(logprobs.top_candidates.is_empty());
    assert_eq!(logprobs.log_probability_sum, None);
    assert!(logprobs.chosen_candidates.is_empty());

    let top_candidate: TopCandidate =
        serde_json::from_value(json!({})).expect("empty top candidate should deserialize");
    assert!(top_candidate.candidates.is_empty());
}

#[test]
fn test_logprobs_result_deserializes_official_json_field_names() {
    let logprobs: Logprobs = serde_json::from_value(json!({
        "topCandidates": [
            {
                "candidates": [
                    {
                        "token": "Hello",
                        "tokenId": 123,
                        "logProbability": -0.1
                    },
                    {
                        "token": "Hi",
                        "tokenId": 124,
                        "logProbability": -1.25
                    }
                ]
            }
        ],
        "logProbabilitySum": -0.1,
        "chosenCandidates": [
            {
                "token": "Hello",
                "tokenId": 123,
                "logProbability": -0.1
            }
        ]
    }))
    .expect("official Gemini logprobs result should deserialize");

    assert_eq!(logprobs.top_candidates.len(), 1);
    assert_eq!(logprobs.top_candidates[0].candidates.len(), 2);
    assert_eq!(
        logprobs.top_candidates[0].candidates[0].token.as_deref(),
        Some("Hello")
    );
    assert_eq!(logprobs.top_candidates[0].candidates[0].token_id, Some(123));
    assert_eq!(
        logprobs.top_candidates[0].candidates[0].log_probability,
        Some(-0.1)
    );
    assert_eq!(logprobs.log_probability_sum, Some(-0.1));
    assert_eq!(logprobs.chosen_candidates.len(), 1);
    assert_eq!(
        logprobs.chosen_candidates[0].token.as_deref(),
        Some("Hello")
    );
    assert_eq!(logprobs.chosen_candidates[0].token_id, Some(123));
    assert_eq!(logprobs.chosen_candidates[0].log_probability, Some(-0.1));
}

#[test]
fn test_unknown_block_reason_deserializes_verbatim() {
    // Same contract for prompt feedback: a new block reason must not fail
    // the payload, and the spelling is preserved.
    let feedback: PromptFeedback = serde_json::from_value(json!({
        "blockReason": "BLOCK_REASON_FUTURE"
    }))
    .expect("unknown block reason should deserialize");
    assert!(matches!(
        feedback.block_reason,
        Some(BlockReason::Unknown(ref s)) if s == "BLOCK_REASON_FUTURE"
    ));
}

#[test]
fn test_finish_reason_wire_spelling_matches_serde() {
    for (reason, wire) in [
        (
            FinishReason::FinishReasonUnspecified,
            "FINISH_REASON_UNSPECIFIED",
        ),
        (FinishReason::Stop, "STOP"),
        (FinishReason::MaxTokens, "MAX_TOKENS"),
        (FinishReason::Safety, "SAFETY"),
        (FinishReason::Recitation, "RECITATION"),
        (FinishReason::Language, "LANGUAGE"),
        (FinishReason::Other, "OTHER"),
        (FinishReason::Blocklist, "BLOCKLIST"),
        (FinishReason::ProhibitedContent, "PROHIBITED_CONTENT"),
        (FinishReason::Spii, "SPII"),
        (
            FinishReason::MalformedFunctionCall,
            "MALFORMED_FUNCTION_CALL",
        ),
        (FinishReason::UnexpectedToolCall, "UNEXPECTED_TOOL_CALL"),
        (
            FinishReason::MissingThoughtSignature,
            "MISSING_THOUGHT_SIGNATURE",
        ),
        (FinishReason::TooManyToolCalls, "TOO_MANY_TOOL_CALLS"),
        (FinishReason::MalformedResponse, "MALFORMED_RESPONSE"),
    ] {
        let serialized = serde_json::to_value(&reason).expect("reason should serialize");
        assert_eq!(serialized, json!(wire));
        assert_eq!(finish_reason(&serialized).as_deref(), Some(wire));
    }
}

#[test]
fn test_unknown_finish_reason_round_trips_verbatim() {
    let reason: FinishReason = serde_json::from_value(json!("FINISH_REASON_FUTURE"))
        .expect("unknown finish reason should deserialize");
    assert!(matches!(&reason, FinishReason::Unknown(s) if s == "FINISH_REASON_FUTURE"));
    let serialized = serde_json::to_value(&reason).expect("reason should serialize");
    assert_eq!(
        finish_reason(&serialized).as_deref(),
        Some("FINISH_REASON_FUTURE")
    );
    assert_eq!(serialized, json!("FINISH_REASON_FUTURE"));
    let wire = finish_reason(&serialized).expect("finish reason present");
    assert_eq!(
        Some(crate::providers::gemini::completion::map_google_finish_reason(&wire)),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn test_streaming_candidate_with_unknown_finish_reason_stays_parseable() {
    // An unknown reason leaves the terminal candidate deserializable.
    let candidate: Candidate = serde_json::from_value(json!({
        "content": {
            "parts": [{"text": "done"}],
            "role": "model"
        },
        "finishReason": "FINISH_REASON_FUTURE"
    }))
    .expect("unknown finish reason should not fail the chunk");

    let reason = candidate.finish_reason.expect("finish reason present");
    let serialized = serde_json::to_value(reason).expect("reason should serialize");
    let wire = finish_reason(&serialized).expect("finish reason present");
    assert_eq!(
        Some(crate::providers::gemini::completion::map_google_finish_reason(&wire)),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[test]
fn test_deserialize_message_user() {
    let raw_message = r#"{
            "parts": [
                {"text": "Hello, world!"},
                {"inlineData": {"mimeType": "image/png", "data": "base64encodeddata"}},
                {"functionCall": {"name": "test_function", "args": {"arg1": "value1"}}},
                {"functionResponse": {"name": "test_function", "response": {"result": "success"}}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "http://example.com/file.pdf"}},
                {"executableCode": {"code": "print('Hello, world!')", "language": "PYTHON"}},
                {"codeExecutionResult": {"output": "Hello, world!", "outcome": "OUTCOME_OK"}}
            ],
            "role": "user"
        }"#;
    let document: Value = serde_json::from_str(raw_message).expect("message JSON");
    let reply = json!({"candidates":[{"content": document.clone()}]});
    read(reply.as_object().expect("reply object")).expect("modeled message fields");
    let content: Content = serde_json::from_value(document).expect("message content");
    assert!(matches!(content.role, Some(Role::User)));
    assert_eq!(content.parts.len(), 7);

    let parts: Vec<Value> = content.parts.into_iter().collect();
    if select(parts[0].as_object().expect("text part object")) == Some(SelectedKind::Text) {
        let text: String = serde_json::from_value(parts[0]["text"].clone()).expect("text");
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text part");
    }

    if select(parts[1].as_object().expect("inline data part object"))
        == Some(SelectedKind::InlineData)
    {
        let inline_data: InlineData =
            serde_json::from_value(parts[1]["inlineData"].clone()).expect("inline data");
        assert_eq!(inline_data.mime_type, "image/png");
        assert_eq!(inline_data.data, "base64encodeddata");
    } else {
        panic!("Expected inline data part");
    }

    if select(parts[2].as_object().expect("function call part object"))
        == Some(SelectedKind::FunctionCall)
    {
        let function_call: FunctionCall =
            serde_json::from_value(parts[2]["functionCall"].clone()).expect("function call");
        assert_eq!(function_call.name, "test_function");
        assert_eq!(
            function_call
                .args
                .as_object()
                .expect("arguments object")
                .get("arg1")
                .expect("arg1"),
            "value1"
        );
    } else {
        panic!("Expected function call part");
    }

    if select(parts[3].as_object().expect("function response part object"))
        == Some(SelectedKind::FunctionResponse)
    {
        let function_response: FunctionResponse =
            serde_json::from_value(parts[3]["functionResponse"].clone())
                .expect("function response");
        assert_eq!(function_response.name, "test_function");
        assert_eq!(
            function_response
                .response
                .as_ref()
                .expect("response")
                .get("result")
                .expect("result"),
            "success"
        );
    } else {
        panic!("Expected function response part");
    }

    if select(parts[4].as_object().expect("file data part object")) == Some(SelectedKind::FileData)
    {
        let file_data: FileData =
            serde_json::from_value(parts[4]["fileData"].clone()).expect("file data");
        assert_eq!(
            file_data.mime_type.as_ref().expect("media type"),
            "application/pdf"
        );
        assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
    } else {
        panic!("Expected file data part");
    }

    if select(parts[5].as_object().expect("executable code part object"))
        == Some(SelectedKind::ExecutableCode)
    {
        let executable_code: HostedCode =
            serde_json::from_value(parts[5]["executableCode"].clone()).expect("executable code");
        assert_eq!(executable_code.code, "print('Hello, world!')");
    } else {
        panic!("Expected executable code part");
    }

    if select(
        parts[6]
            .as_object()
            .expect("code execution result part object"),
    ) == Some(SelectedKind::CodeExecutionResult)
    {
        let code_execution_result: HostedResult =
            serde_json::from_value(parts[6]["codeExecutionResult"].clone())
                .expect("code execution result");
        assert_eq!(
            code_execution_result.output.expect("output"),
            "Hello, world!"
        );
    } else {
        panic!("Expected code execution result part");
    }
}

#[test]
fn test_deserialize_message_model() {
    let json_data = json!({
        "parts": [{"text": "Hello, user!"}],
        "role": "model"
    });
    let reply = json!({"candidates":[{"content": json_data.clone()}]});
    read(reply.as_object().expect("reply object")).expect("modeled message fields");
    let content: Content = serde_json::from_value(json_data).expect("message content");
    assert!(matches!(content.role, Some(Role::Model)));
    assert_eq!(content.parts.len(), 1);
    if let Some(part) = content.parts.first()
        && select(part.as_object().expect("text part object")) == Some(SelectedKind::Text)
    {
        let text: String = serde_json::from_value(part["text"].clone()).expect("text");
        assert_eq!(text, "Hello, user!");
    } else {
        panic!("Expected text part");
    }
}

// G16: unknown classification is private; exact labels stay in real wire evidence.
#[test]
fn future_metadata_labels_keep_their_exact_string_values() {
    use crate::wire::{Mode, WireFrame};
    let body = json!({
        "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "trafficType": "ON_DEMAND_PRIORITY",
            "promptTokensDetails": [{"modality": "HOLOGRAM", "tokenCount": 3}],
            "promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5
        }
    });
    let wire = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-2.5-flash");
    let response = crate::test_utils::history::decode(
        &wire,
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
    ).expect("future metadata labels do not reject the real reply");
    let raw_usage = &response.raw["usageMetadata"];
    assert_eq!(raw_usage["trafficType"], "ON_DEMAND_PRIORITY");
    assert_eq!(raw_usage["promptTokensDetails"][0]["modality"], "HOLOGRAM");
    let usage: Usage = serde_json::from_value(raw_usage.clone())
        .expect("future metadata labels do not reject valid usage");
    assert!(matches!(usage.traffic_type, Some(Traffic::Unknown(()))));
    let detail = &usage.prompt_tokens_details.as_ref().expect("details")[0];
    assert!(matches!(detail.modality, Modality::Unknown(())));
    assert_eq!(detail.token_count, 3);
    assert_eq!(
        (
            usage.prompt_token_count,
            usage.candidates_token_count,
            usage.total_token_count
        ),
        (3, Some(2), 5)
    );
}

// G2 broadens label strings only; modelled non-string fields still refuse.
#[test]
fn future_metadata_label_acceptance_does_not_admit_non_strings() {
    for value in [json!(12), json!(true), json!([]), json!({})] {
        let traffic = json!({"usageMetadata": {"trafficType": value.clone()}});
        assert!(read(traffic.as_object().expect("reply object")).is_err());
        let modality = json!({"usageMetadata": {
            "promptTokensDetails": [{"modality": value, "tokenCount": 3}]
        }});
        assert!(read(modality.as_object().expect("reply object")).is_err());
    }
}
