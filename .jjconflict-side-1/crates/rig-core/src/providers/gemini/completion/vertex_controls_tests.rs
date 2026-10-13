use serde_json::{Map, Value, json};

use super::{FinalBody, GenerateContent, RawAt, Rewrite, base, request_body,
    request_body_for_vertex, request_params};
use crate::completion::{CompletionRequest, GenerationOptions};
use crate::providers::gemini::GeminiConfig;

const MODEL: &str = "gemini-2.5-flash";

fn target() -> GenerateContent {
    GenerateContent::new(GeminiConfig::new("test-key"), MODEL)
}

fn adjust(body: &mut Map<String, Value>) {
    body.insert("model".to_owned(), json!(MODEL));
    body.insert("transportField".to_owned(), json!({"kept": true}));
}

fn vertex(request: &CompletionRequest) -> FinalBody {
    request_body_for_vertex(request, &target(), MODEL, None, adjust).expect("actual Vertex wrapper")
}

fn bytes(body: FinalBody) -> Vec<u8> {
    let crate::wire::Body::Bytes(bytes) = body.into_body() else { panic!("compact JSON bytes") };
    bytes
}

#[test]
fn actual_vertex_wrapper_overrides_only_explicit_controls_and_candidate_count() {
    let request = CompletionRequest::new("hi").temperature(0.2).max_tokens(64)
        .options(GenerationOptions::default().top_p(0.5).seed(7))
        .additional_params(json!({
            "generationConfig": {"temperature": 0.9, "maxOutputTokens": 900,
                "candidateCount": 2, "topP": 0.9, "topK": 40, "extension": [null, true]},
            "transportField": {"raw": true}
        }));
    let body = vertex(&request);
    assert_eq!(body.get("generationConfig"), Some(&json!({
        "temperature": 0.2, "maxOutputTokens": 64, "candidateCount": 1,
        "topP": 0.9, "topK": 40, "seed": 7, "extension": [null, true]
    })));
    assert_eq!(body.get("transportField"), Some(&json!({"kept": true, "raw": true})));
    assert_eq!(body.get("model"), Some(&json!(MODEL)));
    assert_eq!(body.pointer("/contents/0/parts/0/text"), Some(&json!("hi")));
}

#[test]
fn no_explicit_vertex_controls_keep_other_merged_values() {
    let request = CompletionRequest::new("hi")
        .options(GenerationOptions::default().seed(7).stop(["END"]))
        .additional_params(json!({"generationConfig": {
            "candidateCount": 2, "temperature": 0.9, "maxOutputTokens": 900,
            "responseMimeType": "text/plain", "seed": 9, "topK": 3,
            "extension": {"kept": [1, null]}
        }}));
    assert_eq!(vertex(&request).get("generationConfig"), Some(&json!({
        "candidateCount": 1, "temperature": 0.9, "maxOutputTokens": 900,
        "responseMimeType": "text/plain", "seed": 9, "topK": 3,
        "stopSequences": ["END"], "extension": {"kept": [1, null]}
    })));
    for raw in [json!({}), json!({"generationConfig": null})] {
        let request = CompletionRequest::new("hi").additional_params(raw);
        assert_eq!(vertex(&request).get("generationConfig"), Some(&json!({"candidateCount": 1})));
    }
}

#[test]
fn vertex_does_not_select_generate_content_schema_reapplication() {
    let schema: schemars::Schema = serde_json::from_value(json!({"type":"object"})).expect("schema");
    let request = CompletionRequest::new("hi").output_schema(schema.clone())
        .additional_params(json!({"generationConfig": {
            "responseMimeType": "text/plain", "responseJsonSchema": {"type":"number"}
        }}));
    let vertex = vertex(&request);
    assert_eq!(vertex.pointer("/generationConfig/responseMimeType"), Some(&json!("text/plain")));
    assert_eq!(vertex.pointer("/generationConfig/responseJsonSchema"), Some(&json!({"type":"number"})));
    assert_eq!(vertex.pointer("/generationConfig/candidateCount"), Some(&json!(1)));
    let regular = request_body(&request, &target(), MODEL, None, adjust).expect("regular wrapper");
    assert_eq!(regular.pointer("/generationConfig/responseMimeType"), Some(&json!("application/json")));
    assert_eq!(regular.pointer("/generationConfig/responseJsonSchema"), Some(&schema.to_value()));
    assert_eq!(regular.pointer("/generationConfig/candidateCount"), None);
}

#[test]
fn regular_generate_content_keeps_pre_refactor_request_bytes() {
    for raw in [json!({}), json!({"generationConfig": null}), json!({"generationConfig": {
        "candidateCount": 2, "temperature": 0.9, "maxOutputTokens": 900, "topK": 40
    }})] {
        let request = CompletionRequest::new("hi").temperature(0.2).max_tokens(64)
            .options(GenerationOptions::default().seed(7)).additional_params(raw);
        let target = target();
        // Exact private base/adjust/merge sequence from the approved clean
        // provider preimage, before extracting adjusted_base.
        let previous = request_params(&target, &request, |input| {
            let mut body = base(&request, &target, MODEL, input, super::declaration, super::ReasoningScope::Unchanged)?;
            adjust(&mut body);
            Ok(body)
        }, RawAt::Top, &[Rewrite::GeminiCachedContent(None), Rewrite::GeminiTypedControls])
            .expect("pre-refactor regular request");
        let actual = request_body(&request, &target, MODEL, None, adjust).expect("regular request");
        assert_eq!(bytes(actual), bytes(previous));
    }
    let request = CompletionRequest::new("hi");
    let regular = request_body(&request, &target(), MODEL, None, |_| {}).expect("default regular request");
    assert_eq!(bytes(regular), json!({
        "contents": [{"parts":[{"text":"hi", "thought":false}], "role":"user"}],
        "generationConfig": null, "safetySettings": null,
        "toolConfig": null, "systemInstruction": null
    }).to_string().into_bytes());
}
