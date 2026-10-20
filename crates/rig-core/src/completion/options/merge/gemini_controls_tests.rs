use serde_json::{Value, json};

use super::{
    FinalBody, GenerationConfigKind, GenerationConfigNotObject,
    InteractionsGenerationConfigNotObject, RawAt, Rewrite, request_params,
};
use crate::completion::{CompletionRequest, GenerationOptions};
use crate::error::{EncodeError, ProviderError};
use crate::message::{ToolChoice, ToolName};
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::completion::{GenerateContent, UnsupportedVertexCandidateCount};
use crate::providers::gemini::interactions_api::Interactions;

fn merged(request: &CompletionRequest, base: Value, rewrites: &[Rewrite]) -> FinalBody {
    let target = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3-flash-preview");
    request_params(
        &target,
        request,
        |_| Ok(base.as_object().expect("an encoded base object").clone()),
        RawAt::Top,
        rewrites,
    )
    .expect("the local request merge succeeds")
}

fn typed(request: &CompletionRequest, base: Value) -> FinalBody {
    merged(
        request,
        base,
        &[
            Rewrite::GeminiCachedContent(None),
            Rewrite::GeminiTypedControls,
        ],
    )
}

#[test]
fn vertex_controls_preserve_single_candidate_and_do_not_apply_schema() {
    let target = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3-flash-preview");
    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .output_schema(
            serde_json::from_value::<schemars::Schema>(json!({"type":"object"})).expect("schema"),
        )
        .additional_params(json!({"generationConfig": {
            "temperature":0.9, "maxOutputTokens":900, "candidateCount":1,
            "responseMimeType":"text/plain", "responseJsonSchema":{"type":"number"},
            "unmodeled":{"kept":true}
        }}));
    let body = crate::providers::gemini::completion::request_body_for_vertex(
        &request,
        &target,
        "gemini-3-flash-preview",
        None,
        |_| {},
    )
    .expect("Vertex request body");
    assert_eq!(
        body.pointer("/generationConfig/temperature"),
        Some(&json!(0.2))
    );
    assert_eq!(
        body.pointer("/generationConfig/maxOutputTokens"),
        Some(&json!(64))
    );
    assert_eq!(
        body.pointer("/generationConfig/candidateCount"),
        Some(&json!(1))
    );
    assert_eq!(
        body.pointer("/generationConfig/responseMimeType"),
        Some(&json!("text/plain"))
    );
    assert_eq!(
        body.pointer("/generationConfig/responseJsonSchema"),
        Some(&json!({"type":"number"}))
    );
    assert_eq!(
        body.pointer("/generationConfig/unmodeled"),
        Some(&json!({"kept":true}))
    );
}

#[test]
fn vertex_without_legacy_controls_still_requests_one_candidate() {
    let target = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3-flash-preview");
    let request = CompletionRequest::new("hi")
        .additional_params(json!({"generationConfig":{"candidateCount":4,"temperature":0.9}}));
    let error = crate::providers::gemini::completion::request_body_for_vertex(
        &request,
        &target,
        "gemini-3-flash-preview",
        None,
        |_| {},
    )
    .expect_err("a raw count above 1 is refused, not lowered to 1");
    let ProviderError::Request(source) = ProviderError::from(error) else {
        panic!("a request refusal")
    };
    let refusal = source
        .downcast_ref::<UnsupportedVertexCandidateCount>()
        .expect("the refusal is typed");
    assert_eq!(
        (refusal.spelling, &refusal.count),
        ("candidateCount", &json!(4))
    );
    let request = CompletionRequest::new("hi")
        .additional_params(json!({"generationConfig":{"temperature":0.9}}));
    let body = crate::providers::gemini::completion::request_body_for_vertex(
        &request,
        &target,
        "gemini-3-flash-preview",
        None,
        |_| {},
    )
    .expect("Vertex request body");
    assert_eq!(
        body.pointer("/generationConfig/candidateCount"),
        Some(&json!(1))
    );
    assert_eq!(
        body.pointer("/generationConfig/temperature"),
        Some(&json!(0.9))
    );
    assert_eq!(body.pointer("/generationConfig/responseJsonSchema"), None);
}

#[test]
fn explicit_temperature_beats_raw_without_changing_other_fields() {
    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .options(GenerationOptions::default().top_p(0.5).seed(7))
        .additional_params(json!({
            "generationConfig": {"temperature": 0.9, "topP": 0.9, "topK": 3},
            "providerExtension": {"kept": true}
        }));
    let body = typed(
        &request,
        json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"temperature": 0.2, "candidateCount": 1}
        }),
    );
    assert_eq!(
        body.pointer("/generationConfig/temperature"),
        Some(&json!(0.2))
    );
    assert_eq!(body.pointer("/generationConfig/topK"), Some(&json!(3)));
    assert_eq!(body.pointer("/generationConfig/topP"), Some(&json!(0.9)));
    assert_eq!(body.pointer("/generationConfig/seed"), Some(&json!(7)));
    assert_eq!(
        body.pointer("/generationConfig/candidateCount"),
        Some(&json!(1))
    );
    assert_eq!(body.pointer("/providerExtension/kept"), Some(&json!(true)));
    assert_eq!(body.pointer("/contents/0/parts/0/text"), Some(&json!("hi")));
    assert_eq!(body.pointer("/generationConfig/maxOutputTokens"), None);
    assert_eq!(body.pointer("/generationConfig/responseMimeType"), None);
}

#[test]
fn explicit_cap_and_schema_beat_conflicting_raw_fields() {
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    });
    let request = CompletionRequest::new("hi")
        .max_tokens(64)
        .output_schema(serde_json::from_value::<schemars::Schema>(schema.clone()).expect("schema"))
        .additional_params(json!({"generationConfig": {
            "maxOutputTokens": 900,
            "responseMimeType": "text/plain",
            "responseJsonSchema": {"type": "number", "title": "Raw schema"},
            "temperature": 0.9,
            "responseModalities": ["TEXT"],
            "unmodeled": {"kept": [1, null]}
        }}));
    let body = typed(
        &request,
        json!({"generationConfig": {"maxOutputTokens": 64}}),
    );
    assert_eq!(
        body.pointer("/generationConfig/maxOutputTokens"),
        Some(&json!(64))
    );
    assert_eq!(
        body.pointer("/generationConfig/responseMimeType"),
        Some(&json!("application/json"))
    );
    assert_eq!(
        body.pointer("/generationConfig/responseJsonSchema"),
        Some(&schema)
    );
    assert_eq!(
        body.pointer("/generationConfig/temperature"),
        Some(&json!(0.9))
    );
    assert_eq!(
        body.pointer("/generationConfig/responseModalities"),
        Some(&json!(["TEXT"]))
    );
    assert_eq!(
        body.pointer("/generationConfig/unmodeled"),
        Some(&json!({"kept": [1, null]}))
    );
}

#[test]
fn absent_legacy_controls_leave_raw_values_and_absent_config_unchanged() {
    for config in [
        json!({"temperature": 0.9, "maxOutputTokens": 900, "responseMimeType": "text/plain"}),
        json!("opaque raw config"),
        json!([1, {"kept": true}]),
    ] {
        let raw = json!({"generationConfig": config, "extension": [null, true]});
        let request = CompletionRequest::new("hi").additional_params(raw.clone());
        let body = typed(&request, json!({}));
        let crate::wire::Body::Bytes(bytes) = body.clone().into_body() else {
            panic!("the final body encodes to JSON bytes");
        };
        assert_eq!(bytes, raw.to_string().into_bytes());
        assert_eq!(body.deserialize::<Value>().expect("final body value"), raw);
    }
    let body = typed(&CompletionRequest::new("hi"), json!({"contents": []}));
    assert_eq!(body.get("generationConfig"), None);
    assert_eq!(
        body.deserialize::<Value>().expect("final body value"),
        json!({"contents": []})
    );
}

#[test]
fn generation_options_only_keep_upstream_raw_precedence() {
    let request = CompletionRequest::new("hi")
        .options(
            GenerationOptions::default()
                .top_p(0.5)
                .seed(7)
                .stop(["END"]),
        )
        .additional_params(json!({"generationConfig": {"topP": 0.9, "topK": 3}}));
    let base = json!({"contents": []});
    let unchanged = merged(
        &request,
        base.clone(),
        &[Rewrite::GeminiCachedContent(None)],
    );
    let rewritten = typed(&request, base);
    assert_eq!(rewritten, unchanged);
    assert_eq!(
        rewritten.get("generationConfig"),
        Some(&json!({
            "topP": 0.9, "seed": 7, "stopSequences": ["END"], "topK": 3
        }))
    );
}

#[test]
fn typed_controls_still_work_with_raw_null_config() {
    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .options(GenerationOptions::default().seed(7))
        .additional_params(json!({"generationConfig": null}));
    let body = typed(&request, json!({"generationConfig": {"temperature": 0.2}}));
    assert_eq!(
        body.get("generationConfig"),
        Some(&json!({
            "temperature": 0.2, "maxOutputTokens": 64, "seed": 7
        }))
    );
}

#[test]
fn explicit_controls_refuse_non_object_configs_with_the_concrete_json_kind() {
    for (config, kind) in [
        (json!(true), GenerationConfigKind::Boolean),
        (json!(9), GenerationConfigKind::Number),
        (json!("raw scalar"), GenerationConfigKind::String),
        (json!([1, null]), GenerationConfigKind::Array),
    ] {
        let request = CompletionRequest::new("hi")
            .temperature(0.2)
            .additional_params(json!({"generationConfig": config}));
        let target = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3-flash-preview");
        let error = request_params(
            &target,
            &request,
            |_| Ok(serde_json::Map::new()),
            RawAt::Top,
            &[
                Rewrite::GeminiCachedContent(None),
                Rewrite::GeminiTypedControls,
            ],
        )
        .expect_err("an explicit control cannot replace a malformed config");
        let ProviderError::Request(source) = ProviderError::from(error) else {
            panic!("a typed request error");
        };
        assert_eq!(
            source.downcast_ref::<GenerationConfigNotObject>(),
            Some(&GenerationConfigNotObject { kind })
        );
    }
}

fn interactions_merged(
    request: &CompletionRequest,
    base: Value,
    rewrites: &[Rewrite],
) -> Result<FinalBody, EncodeError> {
    let target = Interactions::new(GeminiConfig::new("test-key"), "gemini-3-flash-preview");
    request_params(
        &target,
        request,
        |_| {
            Ok(base
                .as_object()
                .expect("an encoded Interactions base")
                .clone())
        },
        RawAt::Top,
        rewrites,
    )
}

fn interactions_typed(request: &CompletionRequest, base: Value) -> FinalBody {
    interactions_merged(request, base, &[Rewrite::GeminiInteractionsTypedControls])
        .expect("the local Interactions merge succeeds")
}

#[test]
fn interactions_legacy_controls_override_raw_and_leave_other_options_merged() {
    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .tool_choice(ToolChoice::Required)
        .options(GenerationOptions::default().seed(7).stop(["END"]))
        .additional_params(json!({
            "generation_config": {
                "temperature": 0.9, "max_output_tokens": 900, "tool_choice": "auto",
                "seed": 9, "unmodeled": {"kept": [null, 1]}
            },
            "provider_extension": {"kept": true}
        }));
    let body = interactions_typed(
        &request,
        json!({"model":"gemini-3-flash-preview", "input":"hi"}),
    );
    assert_eq!(
        body.get("generation_config"),
        Some(&json!({
            "seed":9, "stop_sequences":["END"], "temperature":0.2,
            "max_output_tokens":64, "tool_choice":"any", "unmodeled":{"kept":[null,1]}
        }))
    );
    assert_eq!(body.get("input"), Some(&json!("hi")));
    assert_eq!(body.pointer("/provider_extension/kept"), Some(&json!(true)));
    assert_eq!(body.get("generationConfig"), None);
}

#[test]
fn interactions_tool_choice_uses_the_existing_provider_json_conversion() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("any")),
        (
            ToolChoice::Specific {
                function_names: vec![
                    ToolName::new(" alpha ").expect("literal tool name"),
                    ToolName::new("beta").expect("literal tool name"),
                ],
            },
            json!({"allowed_tools":{"mode":"validated", "tools":[" alpha ", "beta"]}}),
        ),
    ] {
        let request = CompletionRequest::new("hi")
            .tool_choice(choice)
            .additional_params(
                json!({"generation_config":{"tool_choice":"raw choice", "temperature":0.9}}),
            );
        let body = interactions_typed(&request, json!({}));
        assert_eq!(
            body.pointer("/generation_config/tool_choice"),
            Some(&expected)
        );
        assert_eq!(
            body.pointer("/generation_config/temperature"),
            Some(&json!(0.9))
        );
        assert_eq!(body.pointer("/generation_config/max_output_tokens"), None);
    }
}

#[test]
fn interactions_without_legacy_controls_preserve_raw_bytes_and_ignore_schema_control() {
    let schema: schemars::Schema =
        serde_json::from_value(json!({"type":"object"})).expect("schema");
    for config in [
        json!({"temperature":0.9}),
        json!("raw config"),
        json!([1, null]),
        Value::Null,
    ] {
        let raw = json!({"generation_config":config, "provider_extension":[null, true]});
        let request = CompletionRequest::new("hi")
            .output_schema(schema.clone())
            .additional_params(raw.clone());
        let body = interactions_typed(&request, json!({}));
        let crate::wire::Body::Bytes(bytes) = body.into_body() else {
            panic!("JSON body bytes");
        };
        assert_eq!(bytes, raw.to_string().into_bytes());
    }
    let body = interactions_typed(&CompletionRequest::new("hi"), json!({"input":"hi"}));
    assert_eq!(body.get("generation_config"), None);
}

#[test]
fn interactions_generation_options_only_keep_upstream_raw_precedence() {
    let request = CompletionRequest::new("hi")
        .options(GenerationOptions::default().seed(7).stop(["END"]))
        .additional_params(json!({"generation_config":{"seed":9, "extension":true}}));
    let base = json!({"input":"hi"});
    let old = interactions_merged(&request, base.clone(), &[]).expect("existing merge");
    let new = interactions_typed(&request, base);
    assert_eq!(new, old);
    assert_eq!(
        new.get("generation_config"),
        Some(&json!({
            "seed":9, "stop_sequences":["END"], "extension":true
        }))
    );
}

#[test]
fn interactions_null_or_absent_config_accepts_a_tool_choice_only_control() {
    for raw in [json!({}), json!({"generation_config":null})] {
        let request = CompletionRequest::new("hi")
            .tool_choice(ToolChoice::Required)
            .additional_params(raw);
        let body = interactions_typed(&request, json!({"input":"hi"}));
        assert_eq!(
            body.get("generation_config"),
            Some(&json!({"tool_choice":"any"}))
        );
        assert_eq!(body.get("generationConfig"), None);
    }
}

#[test]
fn interactions_explicit_controls_keep_concrete_shape_errors() {
    for (config, kind) in [
        (json!(false), GenerationConfigKind::Boolean),
        (json!(9), GenerationConfigKind::Number),
        (json!("raw scalar"), GenerationConfigKind::String),
        (json!([1, null]), GenerationConfigKind::Array),
    ] {
        let request = CompletionRequest::new("hi")
            .tool_choice(ToolChoice::Required)
            .additional_params(json!({"generation_config":config}));
        let error = interactions_merged(
            &request,
            json!({}),
            &[Rewrite::GeminiInteractionsTypedControls],
        )
        .expect_err("an explicit control cannot replace the malformed config");
        let ProviderError::Request(source) = ProviderError::from(error) else {
            panic!("a typed request error");
        };
        assert_eq!(
            source.downcast_ref::<InteractionsGenerationConfigNotObject>(),
            Some(&InteractionsGenerationConfigNotObject { kind })
        );
    }
}
