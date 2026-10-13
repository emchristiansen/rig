//! Copilot's wires, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::CompletionRequest;
use crate::message::Message;
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::Wire;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use bytes::Bytes;

/// One recorded interaction's request or reply body, read out of a cassette.
///
/// The format is a `when:`/`then:` document whose bodies are single-quoted
/// scalars (`''` for a quote). Parsed here rather than with a YAML
/// dependency, and never written.
fn cassette_body(path: &str, section: &str) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes/copilot")
        .join(path);
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let section = text
        .split_once(&format!("\n{section}:"))
        .unwrap_or_else(|| panic!("{} should record a {section} section", file.display()))
        .1;
    section
        .split_once("  body: ")
        .unwrap_or_else(|| panic!("{} should record a {section} body", file.display()))
        .1
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .trim_start_matches('\'')
        .trim_end_matches('\'')
        .replace("''", "'")
}

/// A Copilot addressed with a token that carries no `proxy-ep=` segment, so
/// the base URL is the default one.
fn copilot() -> CopilotConfig {
    CopilotConfig::new("tid=copilot-session-token")
}

fn prompt() -> CompletionRequest {
    CompletionRequest::new("say hi")
}

/// The first text block of a folded turn.
fn text_of(response: &crate::completion::CompletionResponse) -> Option<String> {
    response.choice.iter().find_map(|content| match content {
        crate::message::AssistantContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

/// The request one `encode` produced, for the envelope assertions.
fn encoded(
    wire: &impl Wire<Op = Completion, Payload = crate::wire::Encoded>,
) -> http::Request<Body> {
    wire.encode(prompt(), Mode::Unary)
        .expect("the request encodes")
        .request
}

fn assert_same_requests(mut direct: http::Request<Body>, mut catalog: http::Request<Body>) {
    assert_eq!(direct.uri(), catalog.uri());
    // Each encode creates its own transport request id.
    for request in [&mut direct, &mut catalog] {
        assert!(
            !request
                .headers_mut()
                .remove("x-request-id")
                .unwrap()
                .is_empty()
        );
    }
    assert_eq!(direct.headers(), catalog.headers());
    let (Body::Bytes(direct), Body::Bytes(catalog)) = (direct.into_body(), catalog.into_body())
    else {
        panic!("JSON request bodies")
    };
    assert_eq!(direct, catalog);
}

#[test]
fn a_manual_copilot_wrapper_keeps_its_envelope_after_deserialization() {
    let provider = OpenAIConfig::new("manual-token");
    for shared in [
        provider.chat("model").into(),
        provider.responses("model").into(),
    ] {
        let wire = CopilotWire {
            wire: shared,
            intent: CopilotIntent::Edits,
        };
        let mut reloaded: CopilotWire =
            serde_json::from_value(serde_json::to_value(&wire).unwrap()).unwrap();
        match &mut reloaded.wire {
            OpenAiWire::Chat(wire) => wire.provider.api_key = "manual-token".into(),
            OpenAiWire::Responses(wire) => wire.provider.api_key = "manual-token".into(),
        }
        for wire in [&wire, &reloaded] {
            let request = encoded(wire);
            assert_eq!(request.headers()["copilot-integration-id"], "vscode-chat");
            assert_eq!(request.headers()["openai-intent"], "conversation-edits");
            assert_eq!(
                request.headers()[http::header::AUTHORIZATION],
                "Bearer manual-token"
            );
            assert_eq!(
                request
                    .headers()
                    .get_all(http::header::AUTHORIZATION)
                    .iter()
                    .count(),
                1
            );
        }
    }
}

/// A Copilot envelope around a ChatGPT wire is refused before anything is
/// sent, even with the caller's exact identity: the envelope would replace
/// its `user-agent`, so the request could not carry that identity exactly.
#[test]
fn a_copilot_wrapper_around_the_chatgpt_dialect_is_refused() {
    use crate::providers::chatgpt;

    let provider = OpenAIConfig::with_key(&chatgpt::DIALECT, "token")
        .with_caller_identity(crate::test_utils::test_caller_identity());
    for shared in [
        provider.chat("model").into(),
        provider.responses("model").into(),
    ] {
        let wire = CopilotWire {
            wire: shared,
            intent: CopilotIntent::default(),
        };
        for mode in [Mode::Unary, Mode::Streaming] {
            let error = crate::error::ProviderError::from(
                wire.encode(prompt(), mode)
                    .expect_err("the envelope would replace the caller's identity"),
            );
            match error {
                crate::error::ProviderError::Request(inner) => assert_eq!(
                    inner.downcast_ref::<CopilotEnvelopeOverCallerIdentity>(),
                    Some(&CopilotEnvelopeOverCallerIdentity { dialect: "chatgpt" })
                ),
                other => panic!("expected a request refusal, got {other:?}"),
            }
        }
    }
}

// ── routing ─────────────────────────────────────────────────────────────

/// The intent is a per-turn header, so it lives on the wire and survives
/// whichever route was chosen.
#[test]
fn the_intent_is_a_wire_option_on_both_routes() {
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let wire = copilot().completion(model).with_edits_intent();
        assert_eq!(wire.intent(), CopilotIntent::Edits);
        assert_eq!(
            encoded(&wire)
                .headers()
                .get("openai-intent")
                .map(|value| value.as_bytes()),
            Some(&b"conversation-edits"[..]),
            "{model}"
        );
    }
}

/// Exercise the registry's erased handler against recorded Copilot replies,
/// including the actual outbound request; no live credentials are needed.
async fn registry_request(
    reference: crate::providers::registry::ProviderRef,
    cassette: &str,
) -> crate::test_utils::CapturedHttpRequest {
    use crate::effect::{EffectId, EffectKind};
    use crate::http_client::DynHttpClient;
    use crate::serve::Dispatch;

    let transport = RecordingHttpClient::new(Bytes::from(cassette_body(cassette, "then")));
    let handler = reference
        .config("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
        .completion_handler(
            "copilot",
            reference.model(),
            DynHttpClient::new(transport.clone()),
        );
    let mut request = prompt();
    request.chat_history.insert(0, Message::system("be brief"));
    handler
        .handle(
            EffectKind::Completion {
                request,
                stream: false,
            },
            Dispatch::new(EffectId::from_raw(1), false),
        )
        .await
        .into_outcome()
        .await
        .expect("the registry handler folds the recorded Copilot reply");
    let mut requests = transport.requests();
    assert_eq!(requests.len(), 1);
    requests.remove(0)
}

/// The registry must select the same routes and editor envelope as the
/// dedicated provider. Recorded replies catch a wrong decoder as well.
#[tokio::test]
async fn registry_copilot_preserves_model_routing_and_editor_envelope() {
    use crate::providers::registry::ProviderRef;

    for (model, path, cassette) in [
        (
            super::super::GPT_4O,
            "/chat/completions",
            "agent/completion_smoke.yaml",
        ),
        (
            super::super::GPT_5_3_CODEX,
            "/responses",
            "routing/codex_models_route_through_responses.yaml",
        ),
    ] {
        let reference = ProviderRef::parse(&format!("copilot/openai:{model}"))
            .expect("a registered Copilot selection");
        let request = registry_request(reference, cassette).await;
        assert_eq!(
            request.uri,
            format!("https://api.individual.githubcopilot.com{path}"),
            "the session endpoint and model route must both survive materialization"
        );
        assert_eq!(request.headers["copilot-integration-id"], "vscode-chat");
        assert_eq!(
            request.headers["editor-version"],
            super::super::EDITOR_VERSION
        );
        assert_eq!(request.headers["openai-intent"], "conversation-panel");
        assert_eq!(request.headers["x-initiator"], "user");
        assert_eq!(
            request.headers[http::header::AUTHORIZATION],
            "Bearer tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2"
        );
        if path == "/responses" {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).expect("JSON request");
            assert!(body.get("instructions").is_none());
            assert_eq!(body["input"][0]["role"], "system");
        }
    }
}

/// Explicit registry configuration overrides the model's default route and
/// instruction placement, while keeping the gateway's editor envelope.
#[tokio::test]
async fn registry_copilot_preserves_explicit_configuration_after_reload() {
    use crate::providers::openai::Route;
    use crate::providers::registry::{ProviderConfig, ProviderRef};

    let reference = ProviderRef::configured(
        ProviderConfig::OpenAi(
            OpenAIConfig::with_key(&DIALECT, "")
                .with_base_url("https://gateway.invalid/copilot")
                .with_route(Route::Responses)
                .with_system_instructions_placement(SystemInstructionsPlacement::Instructions),
        ),
        super::super::GPT_4O,
    )
    .unwrap();
    let saved = serde_json::to_string(&reference).expect("configuration serializes");
    let loaded = serde_json::from_str(&saved).expect("configuration reloads");
    let request =
        registry_request(loaded, "routing/codex_models_route_through_responses.yaml").await;
    assert_eq!(request.uri, "https://gateway.invalid/copilot/responses");
    assert_eq!(request.headers["copilot-integration-id"], "vscode-chat");
    assert_eq!(request.headers["openai-intent"], "conversation-panel");
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("JSON request");
    assert_eq!(body["instructions"], "be brief");
    assert_eq!(body["model"], super::super::GPT_4O);
}

// ── the two completion routes, folded from recorded replies ─────────────

/// The chat route's recorded turn folds to the normalized response.
#[tokio::test]
async fn the_chat_route_folds_its_recorded_turn() {
    let body = cassette_body("agent/completion_smoke.yaml", "then");
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_4O),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded chat body folds");

    assert_eq!(response.provider(), PROVIDER_NAME);
    assert!(
        text_of(&response)
            .is_some_and(|text| text.contains("Rust is a systems programming language")),
        "{:?}",
        response.choice
    );
    assert_eq!(response.usage.input_tokens, Some(38));
}

/// The Responses route's recorded turn folds to the normalized response —
/// same operation, same fold, a different wire underneath.
#[tokio::test]
async fn the_responses_route_folds_its_recorded_turn() {
    let body = cassette_body("routing/codex_models_route_through_responses.yaml", "then");
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded responses body folds");

    assert_eq!(response.provider(), PROVIDER_NAME);
    assert_eq!(response.model(), Some(super::super::GPT_5_3_CODEX));
    assert!(
        text_of(&response).is_some_and(|text| text.contains("Refactoring is the process")),
        "{:?}",
        response.choice
    );
    assert_eq!(response.response_id(), Some("resp_REDACTED_1"));
}

/// Copilot's Responses route answers a tool-calling turn with a
/// *contentless* reasoning item — empty `content`, empty `summary`, no
/// encrypted payload, just an id. The next turn has to replay it verbatim:
/// `crates/rig-cassette/fixtures/cassettes/copilot/typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml`
/// records `{"id":"id_REDACTED_1","summary":[],"type":"reasoning"}` ahead of
/// the tool result in its second request, so the block has to survive the
/// fold or the replayed history is missing an item the provider sent.
#[tokio::test]
async fn a_contentless_reasoning_item_survives_the_fold() {
    let body = cassette_body(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip.yaml",
        "then",
    );
    let response = crate::driver::Model::new(
        copilot().completion(super::super::GPT_5_3_CODEX),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .call(prompt())
    .await
    .expect("the recorded responses body folds");

    let reasoning = response
        .choice
        .iter()
        .find(|content| matches!(content, crate::message::AssistantContent::Reasoning(_)))
        .unwrap_or_else(|| panic!("the turn's reasoning item survives: {:?}", response.choice));
    assert_eq!(
        reasoning.native_item(),
        Some(&serde_json::json!({
            "content": [],
            "id": "id_REDACTED_1",
            "summary": [],
            "type": "reasoning"
        }))
    );
}

// ── the modality wires ──────────────────────────────────────────────────

/// The embeddings reply folds with the request's inputs joined back on, and
/// a reply without a usage block reports no counter rather than failing:
/// Copilot's multi-vendor route omits it.
#[tokio::test]
async fn the_embeddings_wire_folds_its_recorded_reply() {
    let body = cassette_body("embeddings/embeddings_smoke.yaml", "then");
    let documents = vec![
        "Rust values memory safety and predictable performance.".to_owned(),
        "Streaming responses arrive incrementally instead of all at once.".to_owned(),
        "Embeddings turn text into numeric vectors for similarity search.".to_owned(),
    ];
    let bound = crate::driver::Model::new(
        copilot().embedding(super::super::TEXT_EMBEDDING_3_SMALL, None),
        RecordingHttpClient::new(Bytes::from(body)),
    );
    assert_eq!(
        bound.capabilities().ndims,
        1536,
        "the width defaults from the model"
    );
    assert_eq!(bound.capabilities().max_documents, 1024);

    let response = bound
        .call(documents.clone())
        .await
        .expect("the recorded embeddings body folds");
    assert_eq!(response.provider, PROVIDER_NAME);
    assert_eq!(response.embeddings.len(), documents.len());
    assert_eq!(response.embeddings[0].document, documents[0]);
    assert!(!response.embeddings[0].vec.is_empty());
}

/// The recorded embeddings request carries the width resolved from the model
/// identifier: the client layer defaulted it, and the cassette records
/// `"dimensions":1536` for a caller who named none.
#[test]
fn the_embeddings_request_sends_the_resolved_width() {
    let wire = copilot().embedding(super::super::TEXT_EMBEDDING_3_SMALL, None);
    let encoded = wire
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(body["dimensions"], serde_json::json!(1536));
    assert_eq!(body["model"], serde_json::json!("text-embedding-3-small"));

    // The legacy Ada model accepts no width at all.
    let ada = copilot().embedding(super::super::TEXT_EMBEDDING_ADA_002, None);
    let encoded = ada
        .encode(vec!["one".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert!(body.get("dimensions").is_none(), "{body}");
}

/// Copilot's catalogue names the vendor behind each model and nests the
/// modality under `capabilities.type`, which is what distinguishes it from
/// the OpenAI-shaped listing.
#[tokio::test]
async fn the_model_listing_folds_its_recorded_catalogue() {
    let body = cassette_body("models/list_models_smoke.yaml", "then");
    let models = crate::driver::Model::new(
        copilot().models(),
        RecordingHttpClient::new(Bytes::from(body)),
    )
    .list()
    .await
    .expect("the recorded catalogue folds");

    let opus = models
        .iter()
        .find(|model| model.id == "claude-opus-4.7")
        .expect("the recorded catalogue names claude-opus-4.7");
    assert_eq!(opus.name.as_deref(), Some("Claude Opus 4.7"));
    assert_eq!(opus.r#type.as_deref(), Some("chat"));
}

// ── the configuration is storable data ──────────────────────────────────

/// A wire is data a host may store in a scene, a component or a config
/// file, so the credential must not travel with it.
#[test]
fn a_serialized_config_carries_no_credential() {
    a_config_reloads_without_its_credential(
        &CopilotConfig::new("tid=super-secret-session-token"),
        "super-secret",
        |copilot| &copilot.api_key,
    );

    // And the wires built from it, whichever route.
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let json = serde_json::to_string(
            &CopilotConfig::new("tid=super-secret-session-token").completion(model),
        )
        .expect("the wire serializes");
        assert!(!json.contains("super-secret"), "{model}: {json}");
    }
}

/// Persistence must refuse callbacks it cannot restore; this is a local data contract.
#[test]
fn named_dialect_persistence_rejects_replaced_hook_definitions() {
    // Even a copied callback set is not the registered static definition.
    // Function addresses cannot establish whether a custom definition is reloadable.
    static REPLACEMENT: DialectHooks = DialectHooks {
        default_endpoint: HOOKS.default_endpoint,
        model_route: HOOKS.model_route,
        completion_envelope: HOOKS.completion_envelope,
        modality_envelope: None,
    };
    let changed = Dialect {
        quirks: Quirks {
            hooks: Some(&REPLACEMENT),
            ..DIALECT.quirks
        },
        ..DIALECT
    };
    assert_ne!(changed, DIALECT);
    assert!(serde_json::to_value(changed).is_err());
    let restored: Dialect = serde_json::from_value(serde_json::to_value(DIALECT).unwrap()).unwrap();
    assert_eq!(restored, DIALECT);
    assert!(std::ptr::eq(restored.quirks.hooks.unwrap(), &HOOKS));
}

/// Synthetic credentials and explicit hosts exercise construction precedence, not server behavior.
/// Recorded reply normalization is covered by the registry and direct-route tests above.
#[test]
fn configured_outbound_endpoints_remain_explicit_after_rotation() {
    use crate::providers::registry::{ProviderConfig, ProviderId, ProviderRef};

    let keys = [
        "tid=1;proxy-ep=proxy.individual.githubcopilot.com;",
        "tid=2;proxy-ep=proxy.business.githubcopilot.com;",
    ];
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        for base in [
            "https://api.githubcopilot.com",
            "https://api.individual.githubcopilot.com",
            "https://explicit.invalid/copilot",
        ] {
            for route in [Route::Chat, Route::Responses] {
                let reference = ProviderRef::configured(
                    ProviderConfig::OpenAi(
                        OpenAIConfig::with_key(&DIALECT, keys[0])
                            .with_base_url(base)
                            .with_route(route),
                    ),
                    model,
                )
                .unwrap();
                let restored: ProviderRef =
                    serde_json::from_value(serde_json::to_value(reference).unwrap()).unwrap();
                for key in keys {
                    let ProviderConfig::OpenAi(provider) = restored.config(key) else {
                        panic!("OpenAI family")
                    };
                    let request = encoded(&provider.completion(model));
                    let path = match route {
                        Route::Chat => "/chat/completions",
                        Route::Responses => "/responses",
                    };
                    assert_eq!(request.uri().to_string(), format!("{base}{path}"));
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        format!("Bearer {key}")
                    );
                    assert_eq!(request.headers()["copilot-integration-id"], "vscode-chat");
                    assert_same_requests(
                        encoded(&CopilotWire {
                            wire: CopilotConfig::new(key)
                                .with_base_url(base)
                                .openai()
                                .with_route(route)
                                .completion(model),
                            intent: CopilotIntent::default(),
                        }),
                        request,
                    );
                }
            }
        }
        let registered =
            ProviderRef::registered(ProviderId::resolve("copilot").unwrap(), model).unwrap();
        for (key, host) in keys.into_iter().zip([
            "api.individual.githubcopilot.com",
            "api.business.githubcopilot.com",
        ]) {
            let ProviderConfig::OpenAi(provider) = registered.config(key) else {
                panic!("OpenAI family")
            };
            let request = encoded(&provider.completion(model));
            assert_eq!(request.uri().host(), Some(host));
            assert_same_requests(encoded(&CopilotConfig::new(key).completion(model)), request);
        }
    }
}

/// The local envelope depends on input history before either codec consumes it; no reply is needed.
#[test]
fn both_completion_envelopes_see_the_original_vision_and_assistant_history() {
    use crate::message::{DocumentSourceKind, Image, UserContent};
    let mut request = prompt();
    request.chat_history = vec![
        Message::assistant("send an image"),
        Message::User {
            content: vec![UserContent::Image(Image {
                data: DocumentSourceKind::Url("https://image.invalid/example.png".into()),
                ..Image::default()
            })],
        },
    ];
    for model in [super::super::GPT_4O, super::super::GPT_5_3_CODEX] {
        let direct = copilot().completion(model).with_edits_intent();
        let generic = copilot().openai().completion(model);
        for mode in [Mode::Unary, Mode::Streaming] {
            let direct = direct.encode(request.clone(), mode).unwrap();
            let generic = generic.encode(request.clone(), mode).unwrap();
            for (encoded, intent) in [
                (direct, "conversation-edits"),
                (generic, "conversation-panel"),
            ] {
                let request = &encoded.request;
                assert_eq!(request.headers()["x-initiator"], "agent");
                assert_eq!(request.headers()["copilot-vision-request"], "true");
                assert_eq!(request.headers()["openai-intent"], intent);
            }
        }
    }
}

/// The incomplete-terminal opt-in reaches the Responses route's wire and
/// leaves the chat route untouched; the default refuses.
#[test]
fn the_streamed_incomplete_opt_in_is_a_responses_route_option() {
    let copilot = copilot();
    let OpenAiWire::Responses(default) = copilot.completion(super::super::GPT_5_3_CODEX).wire
    else {
        panic!("a Codex model is served by /responses");
    };
    assert_eq!(default.streamed_incomplete, IncompleteTerminal::Refuse);

    let OpenAiWire::Responses(opted_in) = copilot
        .completion(super::super::GPT_5_3_CODEX)
        .with_streamed_incomplete(IncompleteTerminal::Accept)
        .wire
    else {
        panic!("a Codex model is served by /responses");
    };
    assert_eq!(opted_in.streamed_incomplete, IncompleteTerminal::Accept);

    let chat = copilot.completion(super::super::GPT_4O);
    assert_eq!(
        chat.clone()
            .with_streamed_incomplete(IncompleteTerminal::Accept)
            .wire,
        chat.wire
    );
}

/// Copilot's Responses route states strict tools by default. That applies to
/// the typed tools Rig builds and to a declared function's own `strict` and
/// schema; a declared Responses-lite namespace reaches the body exactly as the
/// caller wrote it, empty description included and no `strict` added.
#[test]
fn the_responses_route_sends_declared_tools_as_written_under_its_strict_default() {
    let namespace = serde_json::json!({
        "type": "namespace",
        "name": "functions",
        "description": "",
        "tools": [{
            "type": "function",
            "name": "exec_command",
            "description": "",
            "strict": false,
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
        }]
    });
    let function = serde_json::json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    });
    let request = CompletionRequest {
        tools: vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("lookup").expect("a nonempty tool name"),
            description: "Look something up".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }),
        }],
        additional_params: Some(serde_json::json!({
            "tools": [namespace.clone(), function.clone()]
        })),
        ..prompt()
    };

    let wire = copilot().completion(super::super::GPT_5_3_CODEX);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let request = encoded.request;
    assert_eq!(request.uri().path(), "/responses");
    let Body::Bytes(bytes) = request.into_body() else {
        panic!("a Responses body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("the body is JSON");
    let tools = body["tools"]
        .as_array()
        .expect("tools serialize as an array");

    assert_eq!(
        tools.len(),
        3,
        "typed, then the two declarations: {tools:?}"
    );
    assert_eq!(tools[0]["name"], "lookup");
    assert_eq!(tools[0]["strict"], serde_json::Value::Bool(true));
    assert_eq!(
        tools[1], namespace,
        "the declared namespace is sent verbatim"
    );
    // Strict mode's transformation of the declared function, computed by the
    // typed shape that defines it: `strict: true` and the sanitized schema.
    let strict = crate::providers::openai::responses_api::ResponsesToolDefinition::function(
        "post_to_bus",
        "",
        function["parameters"].clone(),
    )
    .with_strict();
    let mut expected = function;
    expected["strict"] = serde_json::Value::Bool(true);
    expected["parameters"] = strict.parameters;
    assert_eq!(
        expected["parameters"]["additionalProperties"],
        serde_json::Value::Bool(false)
    );
    assert_eq!(
        tools[2], expected,
        "a declared function takes exactly strict mode's transformation"
    );
}

// ── credential source through the wrapper ──────────────────────────────

/// A source counting its reads and supplying `source-token`, or failing.
#[derive(Clone, Default)]
struct SourceForCopilot {
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fails: bool,
}

impl crate::wire::CredentialSource for SourceForCopilot {
    fn current(
        &self,
    ) -> crate::wasm_compat::WasmBoxedFuture<
        '_,
        Result<crate::wire::Credential, crate::wire::CredentialSourceError>,
    > {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let fails = self.fails;
        Box::pin(async move {
            if fails {
                Err("the token owner is offline".into())
            } else {
                Ok(crate::wire::Credential::new("source-token"))
            }
        })
    }
}

fn copilot_over_a_source(source: SourceForCopilot) -> CopilotWire {
    CopilotWire {
        wire: crate::providers::openai::OpenAIConfig::new("static-key")
            .with_credential_source(source)
            .completion("gpt-4o"),
        intent: CopilotIntent::default(),
    }
}

/// The wrapper forwards the contained wire's credential source: the driver
/// reads it once and the transport receives its token, not the static key.
#[tokio::test]
async fn the_copilot_wrapper_forwards_its_wires_credential_source() {
    let source = SourceForCopilot::default();
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    // Whether the canned reply decodes is beside the point: the request has
    // already reached the transport.
    let _ = crate::driver::Model::new(copilot_over_a_source(source.clone()), http.clone())
        .call(crate::completion::CompletionRequest {
            model: None,
            chat_history: vec![crate::message::Message::user("hi")],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
            ..crate::completion::CompletionRequest::new("hi")
        })
        .await;

    assert_eq!(source.reads.load(std::sync::atomic::Ordering::SeqCst), 1);
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].headers["authorization"], "Bearer source-token");
}

/// A failing source refuses through the wrapper, and nothing is sent.
#[tokio::test]
async fn a_failing_credential_source_refuses_through_the_copilot_wrapper() {
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let error = crate::driver::Model::new(
        copilot_over_a_source(SourceForCopilot {
            fails: true,
            ..SourceForCopilot::default()
        }),
        http.clone(),
    )
    .call(crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
        ..crate::completion::CompletionRequest::new("hi")
    })
    .await
    .expect_err("the source has no credential");

    assert!(http.requests().is_empty(), "nothing was sent");
    let crate::error::ProviderError::Request(inner) = &error else {
        panic!("expected a request refusal, got {error:?}");
    };
    assert!(
        inner
            .downcast_ref::<crate::wire::CredentialUnavailable>()
            .is_some()
    );
}

// Endpoint intent survives the in-memory credential exchange independently of its value.
fn recorded_copilot_exchange() -> (
    RecordingHttpClient,
    super::super::auth::Authenticator,
) {
    use super::super::auth::{AuthSource, Authenticator, DeviceCodeHandler};

    let http = RecordingHttpClient::new(
        r#"{"token":"tid=synthetic-exchanged-session","endpoints":{"api":"https://api.enterprise.githubcopilot.com"}}"#,
    );
    let auth = Authenticator::new(
        AuthSource::GitHubAccessToken("synthetic-bootstrap".into()),
        None,
        None,
        DeviceCodeHandler::default(),
        false,
    );
    (http, auth)
}

fn assert_one_recorded_copilot_exchange(http: &RecordingHttpClient) {
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].uri, "https://api.github.com/copilot_internal/v2/token");
    assert_eq!(requests[0].headers[http::header::AUTHORIZATION], "token synthetic-bootstrap");
    assert!(requests[0].body.is_empty());
}

#[tokio::test]
async fn an_explicit_default_copilot_root_survives_the_recorded_exchange() {
    let (http, auth) = recorded_copilot_exchange();
    let config = CopilotConfig::new("synthetic-old-session")
        .with_base_url(super::super::GITHUB_COPILOT_API_BASE_URL);
    assert!(config.base_url_explicit);
    let client = config.connect(http.clone()).authenticate(&auth).await.unwrap();
    assert_eq!(client.config().base_url, super::super::GITHUB_COPILOT_API_BASE_URL);
    assert!(client.config().base_url_explicit);
    assert_eq!(client.config().api_key.expose(), "tid=synthetic-exchanged-session");
    assert_one_recorded_copilot_exchange(&http);
}

#[tokio::test]
async fn a_nondefault_copilot_root_keeps_the_legacy_inference_after_exchange() {
    for marked in [false, true] {
        let (http, auth) = recorded_copilot_exchange();
        let mut config = CopilotConfig::new("synthetic-old-session");
        if marked {
            config = config.with_base_url("https://explicit.invalid/copilot");
        } else {
            config.base_url = "https://explicit.invalid/copilot".into();
        }
        assert_eq!(config.base_url_explicit, marked);
        let client = config.connect(http.clone()).authenticate(&auth).await.unwrap();
        assert_eq!(client.config().base_url, "https://explicit.invalid/copilot");
        assert!(client.config().base_url_explicit);
        assert_one_recorded_copilot_exchange(&http);
    }
}

#[tokio::test]
async fn unpinned_default_and_token_derived_roots_take_the_exchanged_root() {
    for old_token in [
        "synthetic-old-session",
        "tid=synthetic-old;proxy-ep=proxy.individual.githubcopilot.com;",
    ] {
        let (http, auth) = recorded_copilot_exchange();
        let config = CopilotConfig::new(old_token);
        assert!(!config.base_url_explicit);
        let client = config.connect(http.clone()).authenticate(&auth).await.unwrap();
        assert_eq!(client.config().base_url, "https://api.enterprise.githubcopilot.com");
        assert!(!client.config().base_url_explicit);
        assert_one_recorded_copilot_exchange(&http);
    }
}

#[tokio::test]
async fn a_serialized_explicit_copilot_root_stays_pinned_after_exchange() {
    for root in [
        super::super::GITHUB_COPILOT_API_BASE_URL,
        "https://explicit.invalid/copilot",
    ] {
        let (http, auth) = recorded_copilot_exchange();
        let config = CopilotConfig::new("synthetic-old-session").with_base_url(root);
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["base_url_explicit"], serde_json::json!(true));
        let restored: CopilotConfig = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(serde_json::to_value(&restored).unwrap(), encoded);
        assert!(restored.base_url_explicit);
        assert_eq!(restored.base_url, root);
        let client = restored.connect(http.clone()).authenticate(&auth).await.unwrap();
        assert_eq!(client.config().base_url, root);
        assert!(client.config().base_url_explicit);
        assert_one_recorded_copilot_exchange(&http);
    }
}

#[tokio::test]
async fn legacy_two_field_equal_default_configs_have_no_explicitness_evidence() {
    let original = CopilotConfig::new("synthetic-old-session");
    let legacy = serde_json::to_value(&original).unwrap();
    assert_eq!(legacy.as_object().unwrap().len(), 2);
    assert!(legacy.get("base_url_explicit").is_none());
    let restored: CopilotConfig = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), legacy);
    assert!(!restored.base_url_explicit);
    let (http, auth) = recorded_copilot_exchange();
    let client = restored.connect(http.clone()).authenticate(&auth).await.unwrap();
    assert_eq!(client.config().base_url, "https://api.enterprise.githubcopilot.com");
    assert!(!client.config().base_url_explicit);
    assert_one_recorded_copilot_exchange(&http);
}

#[test]
fn a_context_derived_copilot_root_is_not_an_explicit_caller_pin() {
    let context = super::super::auth::AuthContext {
        api_key: "synthetic-context-session".into(),
        api_base: Some("https://api.enterprise.githubcopilot.com".into()),
    };
    let config = CopilotConfig::from_auth(&context);
    assert_eq!(config.base_url, "https://api.enterprise.githubcopilot.com");
    assert!(!config.base_url_explicit);
    assert!(serde_json::to_value(config).unwrap().get("base_url_explicit").is_none());
}
