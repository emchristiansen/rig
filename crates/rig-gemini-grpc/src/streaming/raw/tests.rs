use super::*;

fn legacy(data: Value) -> Value {
    json!({
        "candidates": [{
            "content": {"parts": [{"data": data, "thought": false,
                "thought_signature": [0, 255, 17], "part_metadata": null}], "role": "model"},
            "finish_reason": 1, "index": 0, "finish_message": "done"
        }],
        "prompt_feedback": {},
        "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 20,
            "total_token_count": 30, "cached_content_token_count": 4},
        "model_version": "gemini-2.5-flash", "response_id": "resp-grpc-1"
    })
}

#[test]
fn literal_sdk_capture_preserves_protobuf_kind_and_bytes_without_rest_fallback() {
    let raw = legacy(json!({"FunctionCall": {"name": "lookup", "id": "tool-0", "args": {"fields": {
        "nothing": {"kind": {"NullValue": 0}},
        "number": {"kind": {"NumberValue": 2.0}},
        "string": {"kind": {"StringValue": "literal"}},
        "bool": {"kind": {"BoolValue": true}},
        "object": {"kind": {"StructValue": {"fields": {"child": {"kind": null}}}}},
        "list": {"kind": {"ListValue": {"values": [{"kind": {"StringValue": "item"}}]}}}
    }}}}));
    let sdk = read_response(&raw).expect("legacy SDK capture reads");
    let call = sdk.candidates[0].content.as_ref().expect("content").parts[0].data.as_ref().expect("part data");
    assert!(matches!(call, proto::part::Data::FunctionCall(call)
        if call.name == "lookup" && call.id == "tool-0"));
    let capture = response(&sdk);
    assert_eq!(capture["candidates"][0]["content"]["parts"][0]["data"], raw["candidates"][0]["content"]["parts"][0]["data"]);
    assert_eq!(capture["candidates"][0]["content"]["parts"][0]["thought_signature"], json!([0, 255, 17]));
    assert_eq!(capture["candidates"][0]["finish_reason"], 1);
    assert_eq!(capture["usage_metadata"]["cached_content_token_count"], 4);
    assert_eq!(capture["response_id"], "resp-grpc-1");
    assert_eq!(read_response(&capture).expect("current SDK capture reads"), sdk);
    assert!(read_response(&json!({"candidates": [], "modelVersion": "model", "responseId": "id"})).is_err());
}

#[test]
fn all_known_legacy_code_languages_read_and_project_numeric_enums() {
    for (name, number) in [("LANGUAGE_UNSPECIFIED", 0), ("PYTHON", 1)] {
        let sdk = read_response(&legacy(json!({"ExecutableCode": {"language": name, "code": "print(1)"}})))
            .expect("known old language reads");
        let capture = response(&sdk);
        assert_eq!(capture.pointer("/candidates/0/content/parts/0/data/ExecutableCode/language"), Some(&json!(number)));
        assert_eq!(read_response(&capture).expect("numeric language reads"), sdk);
    }
    let error = read_response(&legacy(json!({"ExecutableCode": {"language": "RUST", "code": "fn main() {}"}})))
        .expect_err("unknown old language is not UNSPECIFIED");
    assert_eq!(error.to_string(), "unknown SDK code language `RUST`");
}

#[test]
fn all_known_legacy_code_outcomes_read_and_project_numeric_enums() {
    for (name, number) in [("OUTCOME_UNSPECIFIED", 0), ("OUTCOME_OK", 1),
        ("OUTCOME_FAILED", 2), ("OUTCOME_DEADLINE_EXCEEDED", 3)] {
        let sdk = read_response(&legacy(json!({"CodeExecutionResult": {"outcome": name, "output": "output"}})))
            .expect("known old outcome reads");
        let capture = response(&sdk);
        assert_eq!(capture.pointer("/candidates/0/content/parts/0/data/CodeExecutionResult/outcome"), Some(&json!(number)));
        assert_eq!(read_response(&capture).expect("numeric outcome reads"), sdk);
    }
    let error = read_response(&legacy(json!({"CodeExecutionResult": {"outcome": "UNKNOWN_FAILURE", "output": "output"}})))
        .expect_err("unknown old outcome is not UNSPECIFIED");
    assert_eq!(error.to_string(), "unknown SDK code outcome `UNKNOWN_FAILURE`");
}

#[test]
fn unknown_numeric_code_enums_survive_without_defaulting() {
    for (kind, field) in [("ExecutableCode", "language"), ("CodeExecutionResult", "outcome")] {
        let mut body = json!({"code": "code", "output": "output"});
        body[field] = json!(731);
        let mut tagged = Map::new(); tagged.insert(kind.to_owned(), body);
        let sdk = read_response(&legacy(Value::Object(tagged))).expect("SDK stores unknown numeric enum");
        let capture = response(&sdk);
        assert_eq!(capture["candidates"][0]["content"]["parts"][0]["data"][kind][field], 731);
        assert_eq!(read_response(&capture).expect("numeric enum retained"), sdk);
    }
}

#[test]
fn selected_sdk_oneof_and_required_field_failures_are_typed() {
    for data in [json!({"Text": "first", "InlineData": {"mime_type": "image/png", "data": []}}),
        json!({"Unknown": {}}), json!({"FunctionCall": {"name": "lookup", "id": "tool-0", "args": {"fields": {
            "bad": {"kind": {"NumberValue": "not a number"}}
        }}}}), json!({"InlineData": {"mime_type": "image/png", "data": [256]}})] {
        assert!(read_response(&legacy(data)).is_err());
    }
}

#[test]
fn current_sdk_reply_fields_survive_capture_and_extras_inspection() {
    use rig_core::completion::ReplyExtras;
    let sdk = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content { parts: vec![proto::Part {
                data: Some(proto::part::Data::Text("answer".into())),
                metadata: Some(proto::part::Metadata::VideoMetadata(proto::VideoMetadata {
                    start_offset: Some(prost_types::Duration { seconds: 1, nanos: 2 }),
                    end_offset: Some(prost_types::Duration { seconds: 3, nanos: 4 }), fps: 24.0,
                })), ..Default::default()
            }], role: "model".into() }),
            finish_reason: 1,
            safety_ratings: vec![proto::SafetyRating { category: 7, probability: 1, blocked: false }],
            citation_metadata: Some(proto::CitationMetadata { citation_sources: vec![proto::CitationSource {
                start_index: Some(0), end_index: Some(6), uri: Some("https://example.com".into()), license: Some("license".into()),
            }] }),
            grounding_metadata: Some(proto::GroundingMetadata {
                grounding_chunks: vec![proto::GroundingChunk {
                    chunk_type: Some(proto::grounding_chunk::ChunkType::RetrievedContext(proto::grounding_chunk::RetrievedContext {
                        uri: Some("https://example.com".into()), title: Some("title".into()), text: Some("answer".into()),
                    })),
                }],
                grounding_supports: vec![proto::GroundingSupport {
                    segment: Some(proto::Segment { part_index: 0, start_index: 0, end_index: 6, text: "answer".into() }),
                    grounding_chunk_indices: vec![0], confidence_scores: vec![0.75],
                }],
            }),
            logprobs_result: Some(proto::LogprobsResult {
                log_probability_sum: Some(-0.5),
                top_candidates: vec![proto::logprobs_result::TopCandidates { candidates: vec![proto::logprobs_result::Candidate {
                    token: Some("answer".into()), token_id: Some(42), log_probability: Some(-0.5),
                }] }],
                chosen_candidates: vec![proto::logprobs_result::Candidate { token: Some("answer".into()), token_id: Some(42), log_probability: Some(-0.5) }],
            }), avg_logprobs: -0.5, ..Default::default()
        }],
        usage_metadata: Some(proto::UsageMetadata { prompt_tokens_details: vec![proto::ModalityTokenCount { modality: 1, token_count: 4 }], ..Default::default() }),
        prompt_feedback: Some(proto::PromptFeedback { block_reason: 0, safety_ratings: vec![] }),
        model_version: "model".into(), response_id: "response".into(),
    };
    let capture = response(&sdk);
    assert_eq!(read_response(&capture).expect("whole SDK fields read"), sdk);
    assert_eq!(capture.pointer("/candidates/0/grounding_metadata/grounding_chunks/0/chunk_type/RetrievedContext/text"), Some(&json!("answer")));
    let extras = crate::extension::GeminiGrpcExtras::from_reply(&rig_core::message::Api::from_static("gemini.generate_content"), &capture)
        .expect("SDK raw inspects through the existing REST extras reader");
    assert_eq!(extras.model_version.as_deref(), Some("model"));
    assert_eq!(extras.response_id.as_deref(), Some("response"));
    assert!(extras.citation_metadata.is_some());
    assert!(extras.logprobs_result.is_some());
    assert_eq!(extras.prompt_tokens_details.expect("prompt token details")[0].token_count, Some(4));
}

#[test]
fn all_unchanged_sdk_part_variants_keep_their_payload_schema() {
    for data in [
        json!({"Text": "answer"}),
        json!({"InlineData": {"mime_type": "image/png", "data": [0, 1, 255]}}),
        json!({"FileData": {"mime_type": "text/plain", "file_uri": "gs://bucket/file"}}),
        json!({"FunctionCall": {"name": "call", "args": null, "id": "tool-0"}}),
        json!({"FunctionResponse": {"name": "call", "response": {"fields": {}}, "id": "tool-0"}}),
    ] {
        let sdk = read_response(&legacy(data.clone())).expect("protected SDK payload reads");
        let capture = response(&sdk);
        let actual = &capture["candidates"][0]["content"]["parts"][0]["data"];
        if let Some(result) = data.get("FunctionResponse") {
            assert_eq!(actual["FunctionResponse"]["name"], result["name"]);
            assert_eq!(actual["FunctionResponse"]["response"], result["response"]);
            assert_eq!(actual["FunctionResponse"]["id"], result["id"]);
        } else { assert_eq!(actual, &data); }
        assert_eq!(read_response(&capture).expect("current SDK capture reads"), sdk);
    }
}

#[test]
fn protected_capture_missing_current_sdk_fields_uses_only_their_schema_defaults() {
    let raw = legacy(json!({"FunctionResponse": {
        "name": "lookup", "response": {"fields": {}}, "id": "tool-0"
    }}));
    let sdk = read_response(&raw).expect("protected capture reads without added fields");
    let candidate = &sdk.candidates[0];
    assert!(candidate.safety_ratings.is_empty());
    assert!(candidate.citation_metadata.is_none());
    assert_eq!(candidate.token_count, 0);
    assert!(candidate.grounding_metadata.is_none());
    assert_eq!(candidate.avg_logprobs, 0.0);
    assert!(candidate.logprobs_result.is_none());
    let part = &candidate.content.as_ref().expect("content").parts[0];
    assert!(part.metadata.is_none());
    assert_eq!(part.thought_signature, [0, 255, 17]);
    let Some(proto::part::Data::FunctionResponse(result)) = &part.data else {
        panic!("expected protected function response");
    };
    assert_eq!(result.name, "lookup");
    assert_eq!(result.id, "tool-0");
    assert_eq!(result.response, Some(prost_types::Struct { fields: Default::default() }));
    assert!(!result.will_continue);
    assert!(result.scheduling.is_none());
    assert!(result.parts.is_empty());
    let feedback = sdk.prompt_feedback.as_ref().expect("protected empty feedback");
    assert_eq!(feedback.block_reason, 0);
    assert!(feedback.safety_ratings.is_empty());
    let usage = sdk.usage_metadata.as_ref().expect("protected usage");
    assert_eq!((usage.prompt_token_count, usage.candidates_token_count,
        usage.total_token_count, usage.cached_content_token_count), (10, 20, 30, 4));
    assert!(usage.prompt_tokens_details.is_empty());
    assert!(usage.cache_tokens_details.is_empty());
    assert!(usage.candidates_tokens_details.is_empty());
    assert_eq!(usage.tool_use_prompt_token_count, 0);
    assert!(usage.tool_use_prompt_tokens_details.is_empty());
    assert_eq!(usage.thoughts_token_count, 0);
    assert_eq!(read_response(&response(&sdk)).expect("current capture roundtrip"), sdk);
}

#[test]
fn present_malformed_current_sdk_fields_do_not_become_legacy_defaults() {
    let original = legacy(json!({"FunctionResponse": {
        "name": "lookup", "response": {"fields": {}}, "id": "tool-0"
    }}));
    for (parent, field, invalid) in [
        ("/candidates/0", "safety_ratings", json!({})),
        ("/candidates/0", "citation_metadata", json!(false)),
        ("/candidates/0", "token_count", json!("bad")),
        ("/candidates/0", "grounding_metadata", json!([])),
        ("/candidates/0", "avg_logprobs", json!("bad")),
        ("/candidates/0", "logprobs_result", json!(false)),
        ("/candidates/0/content/parts/0", "metadata", json!({"VideoMetadata": false})),
        ("/candidates/0/content/parts/0/data/FunctionResponse", "will_continue", json!("bad")),
        ("/candidates/0/content/parts/0/data/FunctionResponse", "scheduling", json!("bad")),
        ("/candidates/0/content/parts/0/data/FunctionResponse", "parts", json!({})),
        ("/prompt_feedback", "block_reason", json!("bad")),
        ("/prompt_feedback", "safety_ratings", json!({})),
        ("/usage_metadata", "prompt_tokens_details", json!({})),
        ("/usage_metadata", "cache_tokens_details", json!({})),
        ("/usage_metadata", "candidates_tokens_details", json!({})),
        ("/usage_metadata", "tool_use_prompt_token_count", json!("bad")),
        ("/usage_metadata", "tool_use_prompt_tokens_details", json!({})),
        ("/usage_metadata", "thoughts_token_count", json!("bad")),
    ] {
        let mut raw = original.clone();
        raw.pointer_mut(parent).expect("fixture parent")[field] = invalid;
        let error: serde_json::Error = read_response(&raw).expect_err(field);
        assert!(!error.to_string().is_empty(), "{parent}/{field}");
    }
    for field in ["prompt_token_count", "candidates_token_count", "total_token_count",
        "cached_content_token_count"] {
        let mut raw = original.clone();
        raw["usage_metadata"].as_object_mut().expect("usage object").remove(field);
        assert_eq!(read_response(&raw).expect_err("protected counter remains required").to_string(),
            format!("missing SDK field `{field}`"));
    }
}
