//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are the two cells of
//! `crates/rig-cassette/fixtures/cassettes/anthropic/raw_completion_parity_matrix/` — the same turn
//! recorded both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::driver::{feed, settle};
use crate::message::AssistantContent;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Body, Framing, Mode, Operation, Shared, Wire, WireFrame};
use std::sync::Mutex;

/// `text_turn_parity.yaml`'s reply body, verbatim.
const UNARY: &str = r#"{"content":[{"text":"parity probe","type":"text"}],"id":"msg_REDACTED_1","model":"claude-haiku-4-5-20251001","role":"assistant","stop_details":null,"stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"inference_geo":"not_available","input_tokens":14,"output_tokens":6,"service_tier":"standard"}}"#;

/// `streamed_text_turn_parity.yaml`'s reply body, verbatim.
const STREAMED: &str = concat!(
    "event: message_start\n",
    r#"data: {"message":{"content":[],"id":"msg_REDACTED_1","model":"claude-haiku-4-5-20251001","role":"assistant","stop_details":null,"stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"inference_geo":"not_available","input_tokens":14,"output_tokens":1,"service_tier":"standard"}},"type":"message_start"}"#,
    "\n\nevent: content_block_start\n",
    r#"data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}"#,
    "\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n",
    "event: content_block_delta\n",
    r#"data: {"delta":{"text":"p","type":"text_delta"},"index":0,"type":"content_block_delta"}"#,
    "\n\nevent: content_block_delta\n",
    r#"data: {"delta":{"text":"arity probe","type":"text_delta"},"index":0,"type":"content_block_delta"}"#,
    "\n\nevent: content_block_stop\n",
    r#"data: {"index":0,"type":"content_block_stop"}"#,
    "\n\nevent: message_delta\n",
    r#"data: {"delta":{"stop_details":null,"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":14,"output_tokens":6}}"#,
    "\n\n",
);

fn wire() -> Messages {
    AnthropicConfig::new("sk-test").completion("claude-haiku-4-5")
}

fn request() -> CompletionRequest {
    CompletionRequest::new("Reply with exactly: parity probe").max_tokens(32)
}

/// Fold a recorded reply body through the wire's own decoder, read in the
/// mode the driver would have read it in — which is also what decides the
/// framing, exactly as `encode` decides it.
fn fold(body: &str, mode: Mode) -> crate::completion::CompletionResponse {
    let wire = wire();
    let frames = match mode {
        Mode::Streaming => Framing::Sse.split(body.as_bytes()),
        Mode::Unary => vec![WireFrame::Text(body.to_owned())],
    };
    let shared = Mutex::new(Shared::new(crate::operation::Turn::relayed("anthropic")));
    let fed = feed(
        &mut wire.decoder_in(mode),
        Some(wire.reassembler()),
        &shared,
        frames,
    );
    let decoded = settle(
        shared,
        fed,
        crate::wire::Reply {
            provider: "anthropic".to_owned(),
            raw: serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
            provider_request_id: Some("req_REDACTED_1".to_owned()),
            response_headers: Default::default(),
        },
    );
    for item in &decoded.items {
        assert!(
            item.is_ok(),
            "the recorded reply decodes without an in-band error: {item:?}"
        );
    }
    decoded.outcome.expect("the fold produces a response")
}

#[test]
fn the_same_turn_folds_identically_whether_it_was_buffered_or_streamed() {
    let buffered = fold(UNARY, Mode::Unary);
    let streamed = fold(STREAMED, Mode::Streaming);

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.model(), streamed.model());
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.message_id, streamed.message_id);
    assert_eq!(buffered.provider_request_id, streamed.provider_request_id);
    assert_eq!(
        buffered.choice.first().and_then(AssistantContent::as_text),
        Some("parity probe")
    );
    assert_eq!(buffered.usage.output_tokens, Some(6));
    assert_eq!(buffered.usage.input_tokens, Some(14));
}

#[test]
fn the_buffered_reply_is_one_frame_whose_terminal_is_unconditional() {
    let wire = wire();
    let shared = Mutex::new(Shared::new(crate::operation::Turn::relayed("anthropic")));
    let fed = feed(
        &mut wire.decoder_in(Mode::Unary),
        Some(wire.reassembler()),
        &shared,
        [
            WireFrame::Text(UNARY.to_owned()),
            WireFrame::Text("{not json".to_owned()),
        ],
    );
    let decoded = settle(
        shared,
        fed,
        crate::wire::Reply {
            provider: "anthropic".to_owned(),
            raw: serde_json::Value::Null,
            provider_request_id: None,
            response_headers: Default::default(),
        },
    );
    assert!(
        decoded.outcome.is_ok(),
        "a whole message is a complete turn, so it always ends in a terminal"
    );
    assert!(
        decoded.items.iter().all(Result::is_ok),
        "the terminal stops the driver before the corrupt second frame"
    );
}

#[test]
fn the_streaming_request_asks_for_a_stream_and_the_unary_one_does_not() {
    let unary = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(unary.framing, Framing::Whole);
    let unary_body = body_of(&unary);
    assert_eq!(unary_body.get("stream"), None);

    let streaming = wire()
        .encode(request(), Mode::Streaming)
        .expect("the request encodes");
    assert_eq!(streaming.framing, Framing::Sse);
    assert_eq!(
        body_of(&streaming).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn the_request_carries_the_key_version_and_endpoint() {
    let encoded = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;
    assert_eq!(request.uri().path(), "/v1/messages");
    assert_eq!(
        request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok()),
        Some("sk-test")
    );
    assert_eq!(
        request
            .headers()
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok()),
        Some(super::super::completion::ANTHROPIC_VERSION_LATEST)
    );
    assert_eq!(encoded.request_id_header, Some("request-id"));
}

#[test]
fn private_headers_preserve_custom_version_and_caller_beta_order() {
    let provider = AnthropicConfig::new("sk-test-exact")
        .with_version("2023-01-01")
        .with_beta("first")
        .with_beta("first")
        .with_beta("second");
    let request = provider
        .headers(http::Request::get("https://example.invalid/v1/models"))
        .body(())
        .expect("the private headers build");
    assert_eq!(request.headers()["x-api-key"], "sk-test-exact");
    assert_eq!(request.headers()["anthropic-version"], "2023-01-01");
    assert_eq!(request.headers()["anthropic-beta"], "first,first,second");
}

fn body_of(encoded: &Encoded) -> serde_json::Value {
    let request = &encoded.request;
    match request.body() {
        Body::Bytes(bytes) => {
            serde_json::from_slice(bytes).expect("the body is the JSON the wire built")
        }
        Body::Multipart(_) => panic!("the Messages endpoint takes JSON"),
    }
}

#[test]
fn a_serialized_provider_never_carries_its_key() {
    let provider =
        AnthropicConfig::new("sk-live-do-not-leak").with_beta("prompt-caching-2024-07-31");
    a_config_reloads_without_its_credential(&provider, "sk-live-do-not-leak", |provider| {
        &provider.api_key
    });

    let wire = provider.completion("claude-haiku-4-5");
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(!json.contains("sk-live-do-not-leak"));
    let restored: Messages = serde_json::from_str(&json).expect("the wire round-trips");
    assert_eq!(restored.model, "claude-haiku-4-5");
    assert_eq!(restored.provider.dialect, ANTHROPIC);
    assert!(restored.provider.api_key.is_empty());
}

#[test]
fn a_dialect_round_trips_by_name_and_rejects_an_unknown_one() {
    for dialect in [ANTHROPIC, ZAI, MINIMAX, MOONSHOT, XIAOMIMIMO] {
        let json = serde_json::to_string(&dialect).expect("a dialect serializes");
        assert_eq!(json, format!("\"{}\"", dialect.name));
        let restored: Dialect = serde_json::from_str(&json).expect("a dialect round-trips");
        assert_eq!(restored, dialect);
    }
    assert!(serde_json::from_str::<Dialect>("\"not-a-provider\"").is_err());
}

#[test]
fn a_gateway_defaults_max_tokens_to_its_one_documented_ceiling() {
    assert_eq!(
        ANTHROPIC.default_max_tokens("claude-haiku-4-5"),
        Some(64_000)
    );
    assert_eq!(ANTHROPIC.default_max_tokens("some-unknown-model"), None);
    // A gateway documents one ceiling rather than per-model limits, so an
    // unrecognized model still gets a usable default.
    assert_eq!(ZAI.default_max_tokens("some-unknown-model"), Some(4096));
    // `strict_tool_schemas` is a const quirk of a const dialect, so the
    // gateway's disagreement with Anthropic is a compile-time fact, not a
    // runtime one.
    const _: () = assert!(!ZAI.quirks.strict_tool_schemas);
    const _: () = assert!(ANTHROPIC.quirks.strict_tool_schemas);
}

#[test]
fn a_base_url_that_already_names_the_endpoint_is_trimmed() {
    for pasted in [
        "https://example.invalid/v1/messages",
        "https://example.invalid/messages",
        "https://example.invalid/v1",
        "https://example.invalid/",
    ] {
        assert_eq!(normalize_base_url(pasted), "https://example.invalid");
    }
}

/// Each dialect reads images on its documented vision models only, in user
/// turns and tool results; no Messages model reads assistant images.
#[test]
fn each_dialect_reads_images_on_its_vision_models() {
    use crate::completion::ReplayTarget;

    for (dialect, model, images) in [
        (&ANTHROPIC, "claude-sonnet-4-6", true),
        (&ZAI, "glm-4.6", false),
        (&ZAI, "glm-4.5-air", false),
        (&ZAI, "glm-4.5v", true),
        (&ZAI, "glm-4.6v-flash", true),
        (&ZAI, "glm-5v-turbo", true),
        (&MOONSHOT, "kimi-k2-thinking", false),
        (&MOONSHOT, "kimi-k2-0905-preview", false),
        (&MOONSHOT, "moonshot-v1-8k", false),
        (&MOONSHOT, "moonshot-v1-8k-vision-preview", true),
        (&MOONSHOT, "kimi-k2.6", true),
        (&MOONSHOT, "kimi-k3", true),
        (&MINIMAX, "MiniMax-M2.7", false),
        (&MINIMAX, "MiniMax-M3", true),
        (&XIAOMIMIMO, "mimo-v2-flash", false),
        (&XIAOMIMIMO, "mimo-v2-pro", false),
        (&XIAOMIMIMO, "mimo-v2.5-pro", false),
        (&XIAOMIMIMO, "mimo-v2-omni", true),
        (&XIAOMIMIMO, "mimo-v2.5", true),
        (&XIAOMIMIMO, "mimo-v2.6-flash", true),
    ] {
        // The model the request addresses decides, not the wire's own.
        let accepts = AnthropicConfig::with_key(dialect, "sk-test")
            .completion("another-model")
            .accepts(model);
        assert_eq!(accepts.user_images, images, "{model}");
        assert_eq!(accepts.tool_result_images, images, "{model}");
        assert!(!accepts.assistant_images, "{model}");
        assert!(accepts.tools, "{model}");
    }
}

/// Every Messages-format dialect asks for tool input as it is written by
/// default, as pi does for every provider it does not know to reject it.
#[test]
fn every_dialect_defaults_to_eager_tool_input() {
    for quirks in [Quirks::anthropic(), Quirks::gateway()] {
        assert_eq!(quirks.tool_input_streaming, ToolInputStreaming::Eager);
    }
    for dialect in all() {
        assert_eq!(
            dialect.quirks.tool_input_streaming,
            ToolInputStreaming::Eager,
            "{}",
            dialect.name
        );
    }
    assert_eq!(Quirks::gateway().max_tokens, MaxTokens::Fixed(4096));
}

/// The path `config` posts a completion to.
fn messages_path(config: AnthropicConfig) -> String {
    let wire = config.completion("a-model");
    let mut request = CompletionRequest::new("hello");
    request.max_tokens = Some(64);
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    encoded.request.uri().path().to_owned()
}

/// Every preset posts to exactly one `/v1/messages`.
#[test]
fn every_dialect_posts_to_one_v1_messages() {
    for dialect in all() {
        let path = messages_path(AnthropicConfig::with_key(dialect, "sk-test"));
        assert!(path.ends_with("/v1/messages"), "{}: {path}", dialect.name);
        assert_eq!(path.matches("/v1").count(), 1, "{}: {path}", dialect.name);
    }
}

/// A dialect whose default base URL names the endpoint is normalized like
/// a caller-supplied one.
#[test]
fn a_dialect_base_url_is_normalized() {
    let dialect = compatible(
        "custom",
        "https://gateway.example/anthropic/v1/",
        "CUSTOM_API_KEY",
        None,
    );
    let config = AnthropicConfig::with_key(&dialect, "sk-test");
    assert_eq!(config.base_url, "https://gateway.example/anthropic");
    assert_eq!(messages_path(config), "/anthropic/v1/messages");
}
