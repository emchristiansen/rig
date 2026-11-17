//! The request bytes a Codex call sends over HTTP, through Rig's driver and
//! [`CodexHttpClient`].

use super::*;
use crate::responses_api::fidelity_fixtures::*;
use crate::wire::CallerIdentity;
use http::StatusCode;
use rig_core::Model;
use rig_core::completion::CompletionRequest;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Body, Mode, Wire};
use rig_http::test_utils::{MockHttpResponse, SequencedHttpClient};

fn provider() -> OpenAIConfig {
    OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token")
}

fn codex(wire: Responses) -> CodexResponses {
    CodexResponses::new(wire)
        .expect("the ChatGPT dialect speaks the Codex contract")
        .with_caller_identity(
            CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
                .expect("a valid caller identity"),
        )
}

fn chatgpt() -> CodexResponses {
    codex(Responses::new(provider(), "gpt-5.4"))
}

/// The body `wire` encodes `request` to in `mode`.
fn encoded(wire: &CodexResponses, request: CompletionRequest, mode: Mode) -> String {
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let Body::Bytes(body) = encoded.request.body() else {
        panic!("a Responses body is bytes");
    };
    String::from_utf8(body.to_vec()).expect("the body is UTF-8")
}

/// The body of the one HTTP request a call of `request` on `wire` sends.
async fn sent(wire: &CodexResponses, request: CompletionRequest) -> String {
    let transport = SequencedHttpClient::new([MockHttpResponse::ErrorResponse(
        StatusCode::BAD_REQUEST,
        Bytes::from_static(b"{}"),
    )]);
    let model = Model::new(
        wire.clone(),
        CodexHttpClient::new(transport.clone(), wire.clone()),
    );
    model
        .call(request)
        .await
        .expect_err("the scripted reply refuses the request");
    let mut requests = transport.requests();
    assert_eq!(requests.len(), 1, "one call, one request: {requests:?}");
    String::from_utf8(requests.remove(0).body.to_vec()).expect("the body is UTF-8")
}

fn muninn_configured() -> CodexResponses {
    codex(Responses::new(provider().with_instructions(""), "gpt-5.4"))
}

/// `body`'s `tools`, each as the bytes it was written with.
fn raw_tools(body: &str) -> Vec<String> {
    let tools: Vec<Box<serde_json::value::RawValue>> =
        serde_json::from_str(raw_fields(body)["tools"].get()).expect("`tools` is an array");
    tools.iter().map(|tool| tool.get().to_owned()).collect()
}

fn value(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("JSON")
}

/// The refusal of `request` on `wire`, as the caller of a model sees it, and
/// proof that nothing was sent.
async fn refused(wire: &CodexResponses, request: CompletionRequest) -> String {
    let transport = SequencedHttpClient::new([MockHttpResponse::ErrorResponse(
        StatusCode::BAD_REQUEST,
        Bytes::from_static(b"{}"),
    )]);
    let model = Model::new(
        wire.clone(),
        CodexHttpClient::new(transport.clone(), wire.clone()),
    );
    let error = model
        .call(request)
        .await
        .expect_err("the request is refused");
    assert!(
        transport.requests().is_empty(),
        "a refused request sends nothing: {error:?}"
    );
    format!("{error} / {error:?}")
}

/// The Codex lane's request is the caller's fields and nothing else, in
/// either mode and through the HTTP client: byte for byte in a build that
/// sorts keys, and the same JSON value with `input` last in any build.
#[tokio::test]
async fn the_codex_lane_sends_the_pinned_body() {
    let wire = muninn_configured();
    let bodies = [
        encoded(&wire, muninn_shaped_request(), Mode::Unary),
        encoded(&wire, muninn_shaped_request(), Mode::Streaming),
        sent(&wire, muninn_shaped_request()).await,
    ];
    for body in &bodies {
        assert_eq!(value(body), value(MUNINN_SHAPED_BODY_SORTED));
        assert!(
            body.ends_with(&format!(r#""input":{}}}"#, raw_fields(body)["input"].get())),
            "`input` is the last field: {body}"
        );
        if !preserves_order() {
            assert_eq!(body, MUNINN_SHAPED_BODY_SORTED);
        }
    }
    assert_eq!(bodies[0], bodies[1], "one body in either mode");
    assert_eq!(
        bodies[0], bodies[2],
        "the HTTP client sends the wire's body"
    );
}

/// The conversation identity on the wire states the same cache key and
/// metadata a caller would: the body is the same bytes either way.
#[tokio::test]
async fn an_identity_on_the_wire_writes_the_bytes_the_caller_would() {
    let mut request = muninn_shaped_request();
    request.additional_params = Some(serde_json::json!({
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "store": false,
    }));
    let identified = muninn_configured()
        .with_codex_identity(
            crate::responses_api::codex_identity::CodexIdentity::from_ids("session-7", "thread-7")
                .expect("header-safe ids"),
        )
        .expect("a Codex wire");
    let body = sent(&identified, request).await;
    assert_eq!(
        body,
        sent(&muninn_configured(), muninn_shaped_request()).await
    );
}

/// Declared tools reach the HTTP body as the bytes of the JSON values the
/// caller wrote, after the typed tools and ahead of the wire's own: empty
/// descriptions, an explicit `strict: false`, an explicit null and members
/// Rig's tool type does not model all survive.
#[tokio::test]
async fn declared_tools_reach_the_http_body_as_the_caller_wrote_them() {
    let wire = codex(Responses::new(provider(), "gpt-5.4").with_tool(
        rig_core::providers::openai::responses_api::ResponsesToolDefinition::hosted("web_search"),
    ));
    let declared = [namespace(), declared_function(), declared_hosted()];
    let body = sent(
        &wire,
        turn_declaring(serde_json::Value::Array(declared.to_vec())),
    )
    .await;

    let tools = raw_tools(&body);
    assert_eq!(
        tools.len(),
        5,
        "typed, three declared, one default: {tools:?}"
    );
    assert_eq!(value(&tools[0])["name"], "lookup");
    assert_eq!(
        value(&tools[0])["strict"],
        false,
        "a typed tool always states `strict`"
    );
    assert_eq!(
        tools[1..4],
        declared.iter().map(written).collect::<Vec<_>>(),
        "declared tools must reach the provider as the bytes of the values declared"
    );
    assert_eq!(value(&tools[4]), serde_json::json!({"type": "web_search"}));

    for mode in [Mode::Unary, Mode::Streaming] {
        assert_eq!(
            encoded(
                &wire,
                turn_declaring(serde_json::Value::Array(declared.to_vec())),
                mode
            ),
            body,
            "{mode:?}"
        );
    }
}

/// Strict mode changes a declared function by its own transformation only
/// (`strict` and the sanitized `parameters`), and changes neither a
/// namespace, nor its member, nor a hosted tool.
#[tokio::test]
async fn strict_tools_apply_only_their_own_transformation_to_declared_tools() {
    let wire = codex(Responses::new(provider(), "gpt-5.4").with_strict_tools());
    let declared = serde_json::json!([namespace(), declared_function(), declared_hosted()]);
    let tools = raw_tools(&sent(&wire, turn_declaring(declared)).await);

    assert_eq!(tools.len(), 4);
    assert_eq!(value(&tools[0])["strict"], true, "the typed tool is strict");
    assert_eq!(tools[1], written(&namespace()));
    assert_eq!(tools[3], written(&declared_hosted()));

    let rewritten = value(&tools[2]);
    assert_eq!(rewritten["strict"], true);
    assert_eq!(
        rewritten["parameters"],
        serde_json::json!({
            "type": "object",
            "properties": {"body": {"type": "string"}},
            "required": ["body"],
            "additionalProperties": false
        })
    );
    let mut untouched = rewritten.as_object().expect("an object").clone();
    let mut expected = declared_function().as_object().expect("an object").clone();
    for changed in ["strict", "parameters"] {
        rig_core::json_utils::remove_preserving_order(&mut untouched, changed);
        rig_core::json_utils::remove_preserving_order(&mut expected, changed);
    }
    assert_eq!(
        untouched, expected,
        "strict mode must leave the empty description, the explicit null and \
         false, and the unmodelled member exactly as declared"
    );
    assert!(
        tools[2].contains(r#""x_vendor_hint":null"#),
        "the explicit null is written: {}",
        tools[2]
    );
}

/// The Codex lane under strict mode sends a declared namespace alone exactly
/// as written.
#[tokio::test]
async fn the_codex_lane_sends_a_declared_namespace_verbatim_under_strict_mode() {
    let wire = codex(Responses::new(provider(), "gpt-5.4").with_strict_tools());
    let mut request = prompt();
    request.additional_params = Some(serde_json::json!({ "tools": [namespace()] }));
    let body = sent(&wire, request).await;
    assert_eq!(
        raw_fields(&body)["tools"].get(),
        format!("[{}]", written(&namespace()))
    );
}

/// A `tools` payload that is not a list of tools is refused, and nothing is
/// sent.
#[tokio::test]
async fn a_malformed_tools_payload_is_still_refused() {
    for payload in [
        serde_json::json!({"type": "function"}),
        serde_json::json!(["not a tool object"]),
        serde_json::json!([{"name": "missing_type"}]),
    ] {
        let mut request = turn_declaring(payload.clone());
        request.tools.clear();
        let refusal = refused(&chatgpt(), request).await;
        assert!(refusal.contains("tools"), "{payload}: {refusal}");
    }
}

/// The ciphertext `include` is stated alongside `reasoning` and never
/// forced onto a turn without it; a caller's own `include` is sent as
/// written either way.
#[tokio::test]
async fn the_reasoning_include_is_stated_only_alongside_reasoning() {
    let with = |params: serde_json::Value| {
        let mut request = prompt();
        request.additional_params = Some(params);
        request
    };
    let wire = chatgpt();

    let plain = sent(&wire, prompt()).await;
    assert!(
        !raw_fields(&plain).contains_key("include"),
        "no include is forced onto a turn without reasoning: {plain}"
    );

    let reasoning = sent(
        &wire,
        with(serde_json::json!({"reasoning": {"effort": "low"}})),
    )
    .await;
    assert_eq!(
        raw_fields(&reasoning)["include"].get(),
        r#"["reasoning.encrypted_content"]"#
    );

    let own = sent(
        &wire,
        with(serde_json::json!({"include": ["file_search_call.results"]})),
    )
    .await;
    assert_eq!(
        raw_fields(&own)["include"].get(),
        r#"["file_search_call.results"]"#,
        "the caller's own include, and nothing added without reasoning"
    );

    let asked = sent(
        &wire,
        with(serde_json::json!({"include": ["reasoning.encrypted_content"]})),
    )
    .await;
    assert_eq!(
        raw_fields(&asked)["include"].get(),
        r#"["reasoning.encrypted_content"]"#,
        "a caller who asks for the ciphertext gets it stated"
    );

    let both = sent(
        &wire,
        with(serde_json::json!({
            "include": ["file_search_call.results"],
            "reasoning": {"effort": "low"}
        })),
    )
    .await;
    assert_eq!(
        raw_fields(&both)["include"].get(),
        r#"["file_search_call.results","reasoning.encrypted_content"]"#
    );
}

/// The Codex contract states `store: false`, the dialect's default
/// instructions, and no sampling or output-cap control the caller did not
/// set.
#[tokio::test]
async fn the_chatgpt_dialect_states_store_false_and_clears_nothing() {
    let body = sent(&chatgpt(), prompt()).await;
    let fields = raw_fields(&body);
    let names: Vec<&str> = fields.keys().map(String::as_str).collect();
    assert_eq!(names, ["input", "instructions", "model", "store", "stream"]);
    assert_eq!(fields["store"].get(), "false");
    assert_eq!(
        fields["instructions"].get(),
        r#""You are ChatGPT, a helpful AI assistant.""#
    );
}

/// Controls the Codex contract takes are sent as the caller wrote them.
#[tokio::test]
async fn the_chatgpt_dialect_sends_the_controls_the_caller_set() {
    let controls = serde_json::json!({
        "background": false,
        "metadata": { "k": "v" },
        "parallel_tool_calls": true,
        "service_tier": "priority",
        "text": { "format": { "type": "text" } },
        "user": "user-1",
    });
    let mut request = prompt();
    request.additional_params = Some(controls.clone());
    let body = sent(&chatgpt(), request).await;
    let fields = raw_fields(&body);
    for (name, control) in controls.as_object().expect("an object") {
        assert_eq!(fields[name].get(), written(control), "{name}");
    }
}

/// A control the Codex contract does not take is refused by name before
/// anything is sent, over HTTP as on the WebSocket.
#[tokio::test]
async fn the_chatgpt_dialect_refuses_caller_set_controls_by_name() {
    let with = |params: serde_json::Value| {
        let mut request = prompt();
        request.additional_params = Some(params);
        request
    };
    let mut temperature = prompt();
    temperature.temperature = Some(0.25);
    let mut max_tokens = prompt();
    max_tokens.max_tokens = Some(64);
    for (request, spelling) in [
        (with(serde_json::json!({ "top_p": 0.5 })), "`top_p`"),
        (temperature, "`temperature`"),
        (max_tokens, "`max_output_tokens`"),
        (with(serde_json::json!({ "store": true })), "`store: true`"),
    ] {
        let refusal = refused(&chatgpt(), request).await;
        assert!(
            refusal.contains(spelling)
                && refusal.contains("Codex Responses contract")
                && refusal.contains("this adapter refuses"),
            "{spelling}: {refusal}"
        );
    }
}

/// The body Rig's own Responses wire encodes `request` to.
fn rig_encoded(wire: &CodexResponses, request: CompletionRequest) -> String {
    let encoded = wire
        .wire
        .encode(request, Mode::Streaming)
        .expect("Rig encodes the request");
    let Body::Bytes(body) = encoded.request.body() else {
        panic!("a Responses body is bytes");
    };
    String::from_utf8(body.to_vec()).expect("the body is UTF-8")
}

/// A custom call's answer is sent as the 0.42-era fork sent it: its texts
/// and JSON joined into one string, with the name of the tool that ran.
/// Rig v0.46.0 writes the same result as an array of parts with no name.
/// Every other item of the request keeps the bytes Rig wrote.
#[tokio::test]
async fn a_custom_calls_answer_is_one_string_naming_the_tool_that_ran() {
    use rig_core::message::ToolResultContent;
    let request = answering(
        custom_call_item(),
        vec![
            ToolResultContent::text("Done: "),
            ToolResultContent::Json {
                value: serde_json::json!({"files": 1}),
            },
            ToolResultContent::text("."),
        ],
    );
    let wire = chatgpt();
    let rig = raw_input(&rig_encoded(&wire, request.clone()));
    assert_eq!(
        value(&rig[2]),
        serde_json::json!({
            "type": "custom_tool_call_output", "call_id": "call_c",
            "output": [
                {"type": "input_text", "text": "Done: "},
                {"type": "input_text", "text": "{\"files\":1}"},
                {"type": "input_text", "text": "."},
            ]
        }),
        "what Rig v0.46.0 writes"
    );

    for body in [
        encoded(&wire, request.clone(), Mode::Streaming),
        sent(&wire, request.clone()).await,
    ] {
        let input = raw_input(&body);
        assert_eq!(input.len(), 3, "{body}");
        assert_eq!(
            input[..2],
            rig[..2],
            "the other items are as Rig wrote them"
        );
        assert_eq!(value(&input[1]), custom_call_item());
        assert_eq!(
            value(&input[2]),
            serde_json::json!({
                "type": "custom_tool_call_output", "call_id": "call_c",
                "name": "apply_patch", "output": "Done: {\"files\":1}."
            })
        );
        if !preserves_order() {
            assert_eq!(
                input[2],
                r#"{"call_id":"call_c","name":"apply_patch","output":"Done: {\"files\":1}.","type":"custom_tool_call_output"}"#
            );
        }
    }

    // One text is already one string; the answer gains only its name.
    let single = answering(custom_call_item(), vec![ToolResultContent::text("ok")]);
    assert_eq!(
        value(&raw_input(&sent(&wire, single).await)[2]),
        serde_json::json!({
            "type": "custom_tool_call_output", "call_id": "call_c",
            "name": "apply_patch", "output": "ok"
        })
    );
}

/// The body Rig's own Responses wire encodes `request` to once the request
/// is prepared for it, as Rig's driver prepares it.
fn rig_prepared(wire: &CodexResponses, request: CompletionRequest) -> String {
    use rig_core::wire::Operation as _;
    let request = rig_core::operation::Completion::prepare(request, &wire.wire.describe())
        .expect("Rig prepares the request");
    rig_encoded(wire, request)
}

/// An image Rig v0.46.0 would send as a part of a custom call's answer is
/// refused before anything is sent, as the 0.42-era fork refused it.
#[tokio::test]
async fn an_image_in_a_custom_calls_answer_is_refused_and_nothing_is_sent() {
    use rig_core::message::ToolResultContent;
    let wire = chatgpt();
    for image in images_rig_sends_in_a_result() {
        let request = answering(
            custom_call_item(),
            vec![
                ToolResultContent::text("see "),
                ToolResultContent::Image(image),
            ],
        );
        let rig = raw_input(&rig_prepared(&wire, request.clone()));
        assert_eq!(
            value(&rig[2])["output"][1]["type"],
            "input_image",
            "Rig v0.46.0 sends the image: {rig:?}"
        );

        let refusal = refused(&wire, request).await;
        assert!(refusal.contains(CUSTOM_IMAGE_REFUSAL), "{refusal}");
        assert!(refusal.contains("call_c"), "{refusal}");
    }
}

/// An image Rig v0.46.0 does not send on this contract never reaches the
/// wire: Rig's preparation of the request puts its own words in the image's
/// place, and the answer is sent saying so. The 0.42-era fork refused the
/// request. A caller that must refuse every image in a custom call's answer
/// refuses it before it calls the model.
#[tokio::test]
async fn an_image_rig_omits_from_a_custom_calls_answer_is_sent_as_rigs_words() {
    use rig_core::message::ToolResultContent;
    let wire = chatgpt();
    let request = answering(
        custom_call_item(),
        vec![
            ToolResultContent::text("see "),
            ToolResultContent::Image(image_rig_omits()),
        ],
    );
    assert_eq!(
        value(&raw_input(&sent(&wire, request.clone()).await)[2]),
        serde_json::json!({
            "type": "custom_tool_call_output", "call_id": "call_c", "name": "apply_patch",
            "output": format!("see {TOOL_IMAGE_OMITTED}"),
        })
    );

    // The wire refuses the image when it is given the request unprepared.
    let refusal = wire
        .encode(request, Mode::Streaming)
        .expect_err("the wire sees the image");
    assert!(
        format!("{refusal}").contains(CUSTOM_IMAGE_REFUSAL),
        "{refusal}"
    );
}

/// A function call's answer is not a custom call's: it is sent as Rig wrote
/// it, an image among its parts and no name.
#[tokio::test]
async fn a_function_calls_answer_is_sent_as_rig_wrote_it() {
    use rig_core::message::ToolResultContent;
    let wire = chatgpt();
    for image in images_rig_sends_in_a_result()
        .into_iter()
        .chain([image_rig_omits()])
    {
        let request = answering(
            function_call_item(),
            vec![
                ToolResultContent::text("see "),
                ToolResultContent::Image(image),
            ],
        );
        let rig = raw_input(&rig_prepared(&wire, request.clone()));
        let input = raw_input(&sent(&wire, request).await);
        assert_eq!(input, rig);
        let answer = value(&input[2]);
        assert_eq!(answer["type"], "function_call_output");
        assert!(answer["output"].is_array(), "{answer}");
        assert!(answer.get("name").is_none(), "{answer}");
    }
}
