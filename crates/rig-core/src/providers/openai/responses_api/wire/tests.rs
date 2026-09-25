//! The Responses wire, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::{CompletionModel, CompletionRequest};
use crate::driver::Bound;
use crate::message::{self, Message};
use crate::providers::chatgpt::DIALECT as CHATGPT;
use crate::providers::openai::responses_api::openapi_schema::{self, Schema};
use crate::providers::xai::DIALECT as XAI;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Body, Mode};
use bytes::Bytes;
use futures::StreamExt;

// ── the cassettes, as bytes ─────────────────────────────────────────────

/// One recorded interaction's reply body, read out of a cassette.
///
/// The format is one or more `when:`/`then:` documents; the reply body is
/// either a single-quoted scalar (a JSON body, with `''` for a quote) or a
/// `|+` literal block (an SSE body). Parsed here rather than with a YAML
/// dependency, and never written.
fn cassette_body(path: &str) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes")
        .join(path);
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let reply = text
        .split_once("\nthen:")
        .unwrap_or_else(|| panic!("{} should record a reply", file.display()))
        .1;
    let body = reply
        .split_once("  body: ")
        .unwrap_or_else(|| panic!("{} should record a reply body", file.display()))
        .1;
    match body.strip_prefix("|+\n") {
        // A literal block: four-space-indented lines up to the first line
        // that is neither blank nor indented.
        Some(block) => {
            let mut out = String::new();
            for line in block.lines() {
                match line.strip_prefix("    ") {
                    Some(line) => out.push_str(line),
                    None if line.trim().is_empty() => {}
                    None => break,
                }
                out.push('\n');
            }
            out
        }
        // A single-quoted scalar on one line.
        None => body
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .trim_start_matches('\'')
            .trim_end_matches('\'')
            .replace("''", "'"),
    }
}

/// The unary body of the turn a recorded SSE body streams: the response
/// object the provider itself restates on `response.completed`, which is
/// byte-for-byte the shape its unary endpoint answers with.
fn terminal_response_body(sse: &str) -> String {
    let event = sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .find(|event| {
            matches!(
                event.get("type").and_then(serde_json::Value::as_str),
                Some("response.completed") | Some("response.incomplete")
            )
        })
        .expect("the recorded stream ends with a terminal response event");
    event
        .get("response")
        .expect("a terminal event carries its response object")
        .to_string()
}

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("say hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Fold a recorded unary body through the wire, as [`crate::driver::call`]
/// does.
async fn folded_unary(wire: Responses, body: &str) -> completion::CompletionResponse {
    Bound::new(wire, RecordingHttpClient::new(Bytes::from(body.to_owned())))
        .completion(prompt())
        .await
        .expect("the recorded body folds")
}

/// Fold a recorded SSE body through the wire, as [`crate::driver::stream`]
/// does, draining every event first.
async fn folded_stream(wire: Responses, body: &str) -> completion::CompletionResponse {
    let bound = Bound::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from(body.to_owned()),
        },
    );
    let mut response = bound.stream(prompt()).await.expect("the stream opens");
    while response.next().await.is_some() {}
    response
        .finish()
        .expect("the stream produced a terminal record")
}

fn openai() -> Responses {
    OpenAI::new("test-key").responses("gpt-4o")
}

// ── the property the model exists for ───────────────────────────────────

/// The recorded stream of one turn and that same turn's unary body — the
/// response object its `response.completed` event restates — fold to one
/// answer.
#[tokio::test]
async fn a_unary_body_and_the_stream_of_the_same_turn_fold_alike() {
    let sse = cassette_body("openai/response_identity/responses_streaming_carries_identity.yaml");
    let unary = terminal_response_body(&sse);

    let buffered = folded_unary(openai(), &unary).await;
    let streamed = folded_stream(openai(), &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(buffered.message_id, streamed.message_id);
    assert_eq!(buffered.response_id, streamed.response_id);
    assert_eq!(
        text_of(&buffered),
        Some("stream identity probe".to_owned()),
        "the recorded turn's text must survive both paths"
    );
}

/// A recorded tool turn: the call the provider restates in its unary body is
/// the call its stream assembles from fragments.
#[tokio::test]
async fn a_unary_tool_turn_and_its_stream_fold_alike() {
    let sse = cassette_body("openai/streaming_grammar/tool_then_followup_text.yaml");
    let unary = terminal_response_body(&sse);

    let buffered = folded_unary(openai(), &unary).await;
    let streamed = folded_stream(openai(), &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert!(
        buffered
            .choice
            .iter()
            .any(|content| matches!(content, message::AssistantContent::ToolCall(_))),
        "the recorded turn calls a tool: {:?}",
        buffered.choice
    );
}

/// ChatGPT answers a unary request with a replayed event stream, so the same
/// recorded bytes go through `call` and through `stream`; both fold to the
/// same turn.
#[tokio::test]
async fn a_chatgpt_replayed_body_folds_the_same_unary_and_streamed() {
    let sse = cassette_body("chatgpt/codex_tool_args/zero_argument_tool_call_nonstreaming.yaml");
    let wire = OpenAI::with_key(&CHATGPT, "test-token").responses("gpt-5.4");

    let buffered = folded_unary(wire.clone(), &sse).await;
    let streamed = folded_stream(wire, &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.provider, "chatgpt");
    assert!(
        buffered
            .choice
            .iter()
            .any(|content| matches!(content, message::AssistantContent::ToolCall(_))),
        "the recorded turn calls a tool: {:?}",
        buffered.choice
    );
}

fn text_of(response: &completion::CompletionResponse) -> Option<String> {
    response.choice.iter().find_map(|content| match content {
        message::AssistantContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

// ── what each dialect sends ─────────────────────────────────────────────

fn encoded_body(wire: &Responses, mode: Mode) -> serde_json::Value {
    encoded_body_of(wire, prompt(), mode)
}

/// One request's body, for a turn other than the bare [`prompt`].
fn encoded_body_of(wire: &Responses, request: CompletionRequest, mode: Mode) -> serde_json::Value {
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let request = encoded
        .requests
        .first()
        .expect("a Responses request is one request");
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    serde_json::from_slice(body).expect("the body is JSON")
}

/// The bare [`prompt`] with a history of its own.
fn turn(chat_history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        chat_history,
        ..prompt()
    }
}

fn image_message(text: &str) -> Message {
    Message::User {
        content: vec![
            message::UserContent::text(text),
            message::UserContent::Image(message::Image {
                data: message::DocumentSourceKind::Url("https://example.test/image.png".to_owned()),
                media_type: None,
                detail: None,
                additional_params: None,
            }),
        ],
    }
}

fn assert_string_map(value: &serde_json::Value, field: &str) {
    let values = value
        .as_object()
        .unwrap_or_else(|| panic!("{field} must be an object: {value}"));
    assert!(
        values.values().all(serde_json::Value::is_string),
        "every {field} value must be a string: {value}"
    );
}

fn chatgpt() -> Responses {
    OpenAI::with_key(&CHATGPT, "test-token").responses("gpt-5.4")
}

#[test]
fn a_streamed_request_asks_for_a_stream_and_a_unary_one_does_not() {
    let wire = openai();
    assert_eq!(
        encoded_body(&wire, Mode::Streaming).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(encoded_body(&wire, Mode::Unary).get("stream"), None);
    assert_eq!(
        wire.encode(prompt(), Mode::Unary)
            .expect("the request encodes")
            .framing,
        Framing::Whole
    );
}

/// ChatGPT's gateway answers with an event stream whatever was asked for, so
/// even a unary call asks for one and accepts a reply that names no content
/// type.
#[test]
fn the_chatgpt_dialect_always_streams_and_relaxes_the_content_type() {
    let wire = chatgpt();
    let encoded = wire
        .encode(prompt(), Mode::Unary)
        .expect("the request encodes");

    assert_eq!(encoded.framing, Framing::Sse);
    assert!(encoded.relaxed_content_type);
    assert_eq!(
        encoded_body(&wire, Mode::Unary).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
}

/// The Codex contract states `store: false` and otherwise sends the request
/// the caller built: no sampling, metadata or structured-output control is
/// cleared, and the encrypted-reasoning include is not forced onto a request
/// that carries no reasoning.
#[test]
fn the_chatgpt_dialect_states_store_false_and_clears_nothing() {
    let wire = chatgpt();
    let body = encoded_body(&wire, Mode::Unary);

    assert_eq!(body.get("temperature"), None);
    assert_eq!(body.get("max_output_tokens"), None);
    assert_eq!(body.get("top_p"), None);
    assert_eq!(body.get("store"), Some(&serde_json::Value::Bool(false)));
    assert_eq!(body.get("include"), None);
    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some("You are ChatGPT, a helpful AI assistant.")
    );
}

/// The Codex request Muninn's HTTP lane builds, in the typed shape it builds
/// it, with neither `top_p` nor `temperature` set: cache identity
/// (`prompt_cache_key` and `client_metadata` session/thread ids) and reasoning
/// egress (`reasoning`, `include`, `store`), over a history that opens with a
/// system message.
fn muninn_shaped_codex_request() -> CompletionRequest {
    use crate::providers::openai::responses_api::{
        AdditionalParameters, Include, Reasoning, ReasoningEffort,
    };
    let params = AdditionalParameters {
        prompt_cache_key: Some("thread-7".to_string()),
        client_metadata: std::collections::BTreeMap::from([
            ("session_id".to_string(), "session-7".to_string()),
            ("thread_id".to_string(), "thread-7".to_string()),
        ]),
        reasoning: Some(Reasoning::new().with_effort(ReasoningEffort::High)),
        include: Some(vec![Include::ReasoningEncryptedContent]),
        store: Some(false),
        ..Default::default()
    };
    CompletionRequest {
        chat_history: vec![Message::system("Be brief."), Message::user("say hi")],
        additional_params: Some(serde_json::to_value(params).expect("params serialize")),
        ..prompt()
    }
}

/// The ChatGPT wire configured as Muninn configures its client: explicit empty
/// default instructions (`default_instructions("")` at the fork).
fn muninn_configured_chatgpt() -> Responses {
    OpenAI::with_key(&CHATGPT, "test-token")
        .with_instructions("")
        .responses("gpt-5.4")
}

/// The exact request bytes one encode sends.
fn encoded_bytes(wire: &Responses, request: CompletionRequest, mode: Mode) -> String {
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let request = encoded
        .requests
        .first()
        .expect("a Responses request is one request");
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    String::from_utf8(body.to_vec()).expect("the body is UTF-8")
}

/// Pinned FINAL outbound bytes of the Codex HTTP lane with `top_p` and
/// `temperature` unset: byte-identical to what fork revision `fa1430e4`'s
/// `chatgpt::ResponsesCompletionModel::create_request` emitted for the same
/// request and client configuration. That revision serialized the same
/// `CompletionRequest`/`AdditionalParameters` field order, lifted every
/// system message into `instructions` (merged after its empty default
/// instructions, which leaves the preamble alone), forced `stream: true`,
/// kept the caller's `include` (its reasoning gate was already satisfied)
/// and `store: false`, and wrote `prompt_cache_key` and `client_metadata` as
/// top-level members. No `top_p` or `temperature` key is emitted.
///
/// The input item's key order is not fixed by either revision: `InputItem`
/// serializes through `serde_json::Value` (identical code at `fa1430e4`), so
/// its keys follow the build's `serde_json/preserve_order` feature, which a
/// workspace-wide build unifies on and a `-p rig-core` build leaves off. Both
/// spellings are pinned, and the probe selects the one this build produces.
#[test]
fn the_codex_lane_sends_the_fork_bytes_when_sampling_is_unset() {
    let preserve_order = serde_json::to_string(&serde_json::json!({ "b": 0, "a": 0 }))
        .expect("serializes")
        == r#"{"b":0,"a":0}"#;
    let input = if preserve_order {
        r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"say hi"}]}],"#
    } else {
        r#"{"input":[{"content":[{"text":"say hi","type":"input_text"}],"role":"user","type":"message"}],"#
    };
    let expected = [
        input,
        r#""model":"gpt-5.4","instructions":"Be brief.","stream":true,"#,
        r#""include":["reasoning.encrypted_content"],"prompt_cache_key":"thread-7","#,
        r#""reasoning":{"effort":"high"},"store":false,"#,
        r#""client_metadata":{"session_id":"session-7","thread_id":"thread-7"}}"#,
    ]
    .concat();
    for mode in [Mode::Unary, Mode::Streaming] {
        let bytes = encoded_bytes(
            &muninn_configured_chatgpt(),
            muninn_shaped_codex_request(),
            mode,
        );
        assert_eq!(bytes, expected, "{mode:?}");
    }
}

/// With reasoning but no explicit include, the Codex request gets the
/// encrypted-reasoning include by the same rule as every other dialect.
#[test]
fn the_chatgpt_dialect_includes_encrypted_reasoning_alongside_reasoning() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({ "reasoning": { "effort": "low" } })),
        ..prompt()
    };
    let body = encoded_body_of(&chatgpt(), request, Mode::Unary);
    assert_eq!(
        body.get("include"),
        Some(&serde_json::json!(["reasoning.encrypted_content"]))
    );
}

/// Controls upstream used to clear silently, other than the refused sampling
/// controls, now reach the Codex wire as set.
#[test]
fn the_chatgpt_dialect_sends_the_controls_it_used_to_clear() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "background": false,
            "metadata": { "k": "v" },
            "parallel_tool_calls": true,
            "service_tier": "priority",
            "text": { "format": { "type": "text" } },
            "user": "user-1",
        })),
        ..prompt()
    };
    let body = encoded_body_of(&chatgpt(), request, Mode::Unary);
    assert_eq!(body["background"], serde_json::json!(false));
    assert_eq!(body["metadata"], serde_json::json!({ "k": "v" }));
    assert_eq!(body["parallel_tool_calls"], serde_json::json!(true));
    assert_eq!(body["service_tier"], serde_json::json!("priority"));
    assert_eq!(
        body["text"],
        serde_json::json!({ "format": { "type": "text" } })
    );
    assert_eq!(body["user"], serde_json::json!("user-1"));
}

/// The control a Codex encode refused, recovered by type from the request
/// error, with the error's rendering.
fn refused_codex_control(request: CompletionRequest) -> (UnsupportedCodexControl, String) {
    let error = chatgpt()
        .encode(request, Mode::Unary)
        .expect_err("the Codex contract must refuse this control");
    let crate::error::ProviderError::Request(source) = crate::error::ProviderError::from(error)
    else {
        panic!("an encode refusal is a request error");
    };
    let control = *source
        .downcast_ref::<UnsupportedCodexControl>()
        .unwrap_or_else(|| panic!("expected a named Codex refusal, got {source}"));
    (control, source.to_string())
}

/// A caller-set control this adapter does not send on the Codex contract is
/// refused by name, never cleared, and the request is never built: `top_p`,
/// `temperature`, an output-token cap, and `store: true`. Each refusal names
/// the control and the Codex contract, as this adapter's refusal.
#[test]
fn the_chatgpt_dialect_refuses_caller_set_controls_by_name() {
    let cases = [
        (
            CompletionRequest {
                additional_params: Some(serde_json::json!({ "top_p": 0.5 })),
                ..prompt()
            },
            UnsupportedCodexControl::TopP,
            "`top_p`",
        ),
        (
            CompletionRequest {
                temperature: Some(0.25),
                ..prompt()
            },
            UnsupportedCodexControl::Temperature,
            "`temperature`",
        ),
        (
            CompletionRequest {
                max_tokens: Some(64),
                ..prompt()
            },
            UnsupportedCodexControl::MaxOutputTokens,
            "`max_output_tokens`",
        ),
        (
            CompletionRequest {
                additional_params: Some(serde_json::json!({ "store": true })),
                ..prompt()
            },
            UnsupportedCodexControl::Store,
            "`store: true`",
        ),
    ];
    for (request, control, spelling) in cases {
        let (refused, rendered) = refused_codex_control(request);
        assert_eq!(refused, control);
        assert!(
            rendered.contains(spelling)
                && rendered.contains("Codex Responses contract")
                && rendered.starts_with("this adapter refuses"),
            "{control:?}: {rendered}"
        );
    }
}

/// The generic OpenAI Responses contract is unchanged: it sends `top_p` and
/// `temperature` as set.
#[test]
fn the_generic_contract_sends_top_p_and_temperature() {
    let request = CompletionRequest {
        temperature: Some(0.25),
        additional_params: Some(serde_json::json!({ "top_p": 0.5 })),
        ..prompt()
    };
    let body = encoded_body_of(&openai(), request, Mode::Unary);
    assert_eq!(body["temperature"], serde_json::json!(0.25));
    assert_eq!(body["top_p"], serde_json::json!(0.5));
}

/// The incomplete-terminal policy is decoder configuration: it never reaches
/// the request body.
#[test]
fn the_incomplete_policy_stays_off_the_wire() {
    let wire = openai().with_streamed_incomplete(IncompleteTerminal::Accept);
    let accepting = encoded_body(&wire, Mode::Streaming);
    let refusing = encoded_body(&openai(), Mode::Streaming);
    assert_eq!(accepting, refusing);
}

/// The gateway's own instructions lead, the caller's follow: a backend that
/// expects instructions of its own gets them ahead of the turn's preamble
/// rather than instead of it.
#[test]
fn the_chatgpt_dialect_merges_its_instructions_ahead_of_the_callers() {
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![
            Message::system("Respond tersely."),
            Message::user("say hi"),
        ]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some("You are ChatGPT, a helpful AI assistant.\n\nRespond tersely.")
    );
}

/// ...and they are not stated twice when the caller's preamble already
/// carries them, which is what a replayed conversation's history looks like.
#[test]
fn the_chatgpt_dialect_does_not_repeat_instructions_the_caller_already_carries() {
    let carried = "You are ChatGPT, a helpful AI assistant.\n\nRespond tersely.";
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![Message::system(carried), Message::user("say hi")]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some(carried)
    );
}

/// This gateway rejects the `system` role in `input` outright, so every
/// system message is lifted — the leading run *and* the mid-conversation
/// ones — leaving only the non-system turns as input items.
#[test]
fn the_chatgpt_dialect_lifts_every_system_message_into_instructions() {
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![
            Message::system("System one"),
            Message::user("hi"),
            Message::system("Mid-conversation instruction"),
            Message::user("again"),
        ]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some(
            "You are ChatGPT, a helpful AI assistant.\n\nSystem one\n\nMid-conversation instruction"
        )
    );
    assert_eq!(
        body.get("input")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len),
        Some(2),
        "only the two user turns remain as input: {body}"
    );
}

/// A turn whose terminal event restates the assembled output.
const CHATGPT_ASSEMBLED_OUTPUT: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}

data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[{"type":"message","id":"msg_chatgpt_raw","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"hi"}]}],"tools":[]}}

data: [DONE]"#;

/// The same turn with an empty terminal `output`: the deltas are the only
/// place the content exists. A recorded shape, not a synthetic one.
const CHATGPT_EMPTY_OUTPUT: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}

data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}

data: [DONE]"#;

/// A gateway that answers every request with an event stream sends no reply
/// document of its own, so the terminal `response.completed` *is* the
/// document: `raw` must be that object — deserializable back into the wire
/// type and re-serializing value-equal, carrying the fields rig does not
/// normalize (`service_tier`) — whether or not its `output` restates the
/// turn, and the choice comes from the deltas either way.
#[tokio::test]
async fn a_chatgpt_reply_captures_the_terminal_response_object_as_raw() {
    for (body, case) in [
        (CHATGPT_ASSEMBLED_OUTPUT, "assembled output"),
        (CHATGPT_EMPTY_OUTPUT, "empty output"),
    ] {
        let response = folded_unary(chatgpt(), body).await;

        let typed: crate::providers::openai::responses_api::CompletionResponse =
            serde_json::from_value(response.raw.clone())
                .expect("raw must deserialize back into the wire type");
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            response.raw,
            "{case}: the capture must be exactly what the wire type serializes to"
        );
        assert_eq!(response.raw["service_tier"], "default", "{case}");
        assert_eq!(typed.id, "resp_chatgpt_raw", "{case}");

        assert_eq!(
            response.choice,
            vec![message::AssistantContent::text("hi")],
            "{case}: the deltas are the content"
        );
        assert_eq!(response.usage.total_tokens, Some(2), "{case}");
        assert_eq!(
            response.identity().response_id.as_deref(),
            Some("resp_chatgpt_raw"),
            "{case}"
        );
    }
}

/// A Codex turn whose terminal restates an echoed `top_p` in a shape that does
/// not fit its typed field. The typed projection drops only that key.
const CHATGPT_MISTYPED_TOP_P: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}

data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","top_p":{"value":0.95},"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}

data: [DONE]"#;

/// Lenient projection of echoed metadata is confined to the typed view: the
/// captured `raw` keeps the provider's own value of a key the projection could
/// not type, rather than discarding it so the typed key looks valid. The
/// authoritative fields stay strict (see
/// `known_terminal_with_malformed_usage_surfaces_error_without_terminal`).
#[tokio::test]
async fn a_mistyped_echoed_metadata_key_survives_in_the_captured_raw() {
    let response = folded_unary(chatgpt(), CHATGPT_MISTYPED_TOP_P).await;

    assert_eq!(response.raw["top_p"], serde_json::json!({ "value": 0.95 }));
    assert_eq!(response.raw["service_tier"], "default");
    let typed: crate::providers::openai::responses_api::CompletionResponse =
        serde_json::from_value(response.raw.clone()).expect("raw still decodes");
    assert_eq!(typed.additional_parameters.top_p, None);
    assert_eq!(response.usage.total_tokens, Some(2));
}

fn xai() -> Responses {
    OpenAI::with_key(&XAI, "test-key").responses("grok-4")
}

/// xAI's endpoint lives under `/v1` and rejects top-level `instructions`, so
/// every system message — the leading run and the mid-conversation ones —
/// stays in `input` where it was, and the turn's documents follow the
/// preamble as the shared history conversion places them.
#[test]
fn the_xai_dialect_keeps_every_system_message_in_input() {
    let wire = xai();
    let encoded = wire
        .encode(prompt(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.requests.first().expect("one request").uri(),
        "https://api.x.ai/v1/responses"
    );

    let body = encoded_body_of(
        &wire,
        CompletionRequest {
            documents: vec![crate::completion::Document {
                id: "doc_1".to_owned(),
                text: "Definition of glarb-glarb: an ancient tool.".to_owned(),
                additional_props: Default::default(),
            }],
            ..turn(vec![
                Message::system("System prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("What is glarb-glarb?"),
            ])
        },
        Mode::Unary,
    );

    assert_eq!(body.get("instructions"), None, "{body}");
    let input = body["input"].as_array().expect("input is an array");
    let roles: Vec<_> = input
        .iter()
        .map(|item| item["role"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "system", "user"],
        "{body}"
    );
    assert_eq!(
        input
            .iter()
            .filter(|item| item.to_string().contains("<file id: doc_1>"))
            .count(),
        1,
        "the document rides one user item, after the preamble: {body}"
    );
    assert!(input[1].to_string().contains("<file id: doc_1>"), "{body}");
}

/// A user turn interleaving text and a tool result keeps its order on the
/// wire: text before the result is one message item, the result is a
/// `function_call_output` under its call id, text after is another message.
#[test]
fn the_xai_dialect_folds_tool_results_between_user_text_in_order() {
    let body = encoded_body_of(
        &xai(),
        turn(vec![Message::User {
            content: vec![
                message::UserContent::text("before"),
                message::UserContent::tool_result_with_call_id(
                    "result-id",
                    "call-id".to_owned(),
                    "tool",
                    vec![message::ToolResultContent::json(
                        serde_json::json!({ "ok": true }),
                    )],
                ),
                message::UserContent::text("after"),
            ],
        }]),
        Mode::Unary,
    );

    let input = body["input"].as_array().expect("input is an array");
    assert_eq!(input.len(), 3, "{body}");
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"][0]["text"], "before");
    assert_eq!(input[1]["type"], "function_call_output");
    assert_eq!(input[1]["call_id"], "call-id");
    assert_eq!(input[1]["output"], r#"{"ok":true}"#);
    assert_eq!(input[2]["type"], "message");
    assert_eq!(input[2]["content"][0]["text"], "after");
}

/// A replayed reasoning turn goes back under the id the wire issued, its
/// summary as `summary` and its opaque block as the one `encrypted_content`
/// — never as summary text — ahead of the tool call it preceded.
#[test]
fn the_xai_dialect_replays_reasoning_by_wire_id_with_its_encrypted_payload() {
    let body = encoded_body_of(
        &xai(),
        turn(vec![
            Message::user("Use the tool."),
            Message::Assistant {
                id: Some("msg_1".to_owned()),
                content: vec![
                    message::AssistantContent::Reasoning(message::Reasoning {
                        provider: None,
                        id: Some("rs_1".to_owned()),
                        content: vec![
                            message::ReasoningContent::Summary("explain".to_owned()),
                            message::ReasoningContent::Redacted {
                                data: "opaque-redacted".to_owned(),
                            },
                        ],
                    }),
                    message::AssistantContent::tool_call(
                        "call_1",
                        "my_tool",
                        serde_json::json!({"arg": "value"}),
                    ),
                ],
            },
        ]),
        Mode::Unary,
    );

    let input = body["input"].as_array().expect("input is an array");
    assert_eq!(input.len(), 3, "{body}");
    let reasoning = &input[1];
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(reasoning["id"], "rs_1");
    assert_eq!(
        reasoning["summary"],
        serde_json::json!([{"type": "summary_text", "text": "explain"}])
    );
    assert_eq!(reasoning["encrypted_content"], "opaque-redacted");
    assert_eq!(reasoning.get("content"), None, "{reasoning}");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[2]["name"], "my_tool");
}

/// A success carrying the provider's error envelope instead of a response is
/// the provider's failure, not a decode defect — on the dialects that answer
/// that way.
#[tokio::test]
async fn an_error_envelope_on_a_success_fails_the_xai_call() {
    let error = Bound::new(
        xai(),
        RecordingHttpClient::new(Bytes::from_static(
            br#"{"error":{"message":"no capacity","code":"overloaded"}}"#,
        )),
    )
    .completion(prompt())
    .await
    .expect_err("an error envelope fails the call");

    assert!(
        error.to_string().contains("no capacity"),
        "the provider's own message must survive: {error}"
    );
}

// ── declared tools on the outgoing body ─────────────────────────────────

/// The Codex Responses-lite `functions` namespace: its description is the
/// empty string, and neither it nor its member is a typed Rig tool.
fn responses_lite_namespace() -> serde_json::Value {
    serde_json::json!({
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
    })
}

/// A declared function whose `strict` is an explicit `false` and whose
/// description is empty.
fn declared_function() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    })
}

fn lookup_tool() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: "lookup".to_string(),
        description: "Look something up".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}}
        }),
    }
}

/// A turn carrying one typed tool and `declared` in `additional_params["tools"]`.
fn turn_declaring(declared: serde_json::Value) -> CompletionRequest {
    CompletionRequest {
        tools: vec![lookup_tool()],
        additional_params: Some(serde_json::json!({ "tools": declared })),
        ..prompt()
    }
}

/// The body's `tools` array, as sent.
fn sent_tools(wire: &Responses, request: CompletionRequest) -> Vec<serde_json::Value> {
    encoded_body_of(wire, request, Mode::Unary)["tools"]
        .as_array()
        .expect("tools serialize as an array")
        .clone()
}

/// On the HTTP body, declared tools are the JSON values the caller wrote, in
/// place between the typed request tools and the wire's default tools. Typed
/// tools keep their typed serialization, `strict: false` included.
#[test]
fn declared_tools_reach_the_http_body_as_the_caller_wrote_them() {
    let wire = openai().with_tool(ResponsesToolDefinition::hosted("web_search"));
    let declared = serde_json::json!([responses_lite_namespace(), declared_function()]);

    let tools = sent_tools(&wire, turn_declaring(declared.clone()));

    assert_eq!(
        tools.len(),
        4,
        "typed, two declared, one default: {tools:?}"
    );
    assert_eq!(tools[0]["name"], "lookup");
    assert_eq!(
        tools[0]["strict"],
        serde_json::Value::Bool(false),
        "a typed tool always states `strict`"
    );
    assert_eq!(
        serde_json::Value::Array(tools[1..3].to_vec()),
        declared,
        "declared tools must reach the provider as the same JSON values, including \
         the empty descriptions and the explicit `strict: false`"
    );
    assert_eq!(tools[3]["type"], "web_search");
}

/// Under strict mode a declared namespace is sent verbatim: no `strict` is
/// added to it or to its member, and the empty description survives. Only
/// the typed tool and the declared top-level function take strict mode.
#[test]
fn a_declared_namespace_reaches_the_http_body_verbatim_under_strict_mode() {
    let wire = openai().with_strict_tools();
    let declared = serde_json::json!([responses_lite_namespace(), declared_function()]);

    let tools = sent_tools(&wire, turn_declaring(declared));

    assert_eq!(tools.len(), 3);
    assert_eq!(tools[0]["strict"], serde_json::Value::Bool(true));
    assert_eq!(tools[1], responses_lite_namespace());
    assert_eq!(tools[2]["strict"], serde_json::Value::Bool(true));
    assert_eq!(tools[2]["description"], "");
}

/// The Codex contract, with strict tools, sends a declared Responses-lite
/// namespace exactly as written on the HTTP lane.
#[test]
fn the_codex_lane_sends_a_declared_namespace_verbatim_under_strict_mode() {
    let wire = chatgpt().with_strict_tools();
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "tools": [responses_lite_namespace()]
        })),
        ..prompt()
    };

    let tools = sent_tools(&wire, request);

    assert_eq!(tools, vec![responses_lite_namespace()]);
}

// ── the Codex conversation identity on HTTP ──────────────────────────────

fn derived_identity() -> super::super::codex_identity::CodexIdentity {
    super::super::codex_identity::CodexIdentity::from_ids("session-derived", "thread-derived")
        .expect("header-safe ids")
}

fn encoded_request(wire: &Responses, mode: Mode) -> http::Request<Body> {
    let mut encoded = wire.encode(prompt(), mode).expect("the request encodes");
    encoded.requests.remove(0)
}

fn header_names(request: &http::Request<Body>) -> Vec<&str> {
    let mut names: Vec<&str> = request
        .headers()
        .keys()
        .map(http::HeaderName::as_str)
        .collect();
    names.sort_unstable();
    names
}

/// With a Codex identity, every HTTP request, unary or streamed, names the
/// conversation with the caller's ids, exactly as given: dashed `session-id`
/// and `thread-id`, `x-client-request-id`, and the body's `prompt_cache_key`
/// and `client_metadata`. The per-request `session_id` is gone, and two
/// requests carry the same identity.
#[test]
fn a_codex_identity_names_every_http_request_with_the_callers_ids() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    for mode in [Mode::Unary, Mode::Streaming, Mode::Unary] {
        let request = encoded_request(&wire, mode);
        assert_eq!(
            header_names(&request),
            [
                "authorization",
                "content-type",
                "originator",
                "session-id",
                "thread-id",
                "user-agent",
                "x-client-request-id"
            ],
            "{mode:?}"
        );
        assert_eq!(request.headers()["session-id"], "session-derived");
        assert_eq!(request.headers()["thread-id"], "thread-derived");
        assert_eq!(request.headers()["x-client-request-id"], "thread-derived");
        let Body::Bytes(body) = request.body() else {
            panic!("a Responses body is bytes");
        };
        let body: serde_json::Value = serde_json::from_slice(body).expect("the body is JSON");
        assert_eq!(body["prompt_cache_key"], "thread-derived");
        assert_eq!(
            body["client_metadata"],
            serde_json::json!({"session_id": "session-derived", "thread_id": "thread-derived"})
        );
    }
}

/// Lite is opt-in: Standard mode retains the existing top-level tools shape
/// and does not add the internal HTTP marker.
#[test]
fn standard_request_shape_keeps_the_existing_codex_wire_body() {
    let namespace = responses_lite_namespace();
    let encoded = encoded_request(&chatgpt(), Mode::Unary);
    assert!(
        encoded
            .headers()
            .get(super::super::responses_lite::HTTP_HEADER)
            .is_none()
    );
    let tools = sent_tools(
        &chatgpt(),
        turn_declaring(serde_json::json!([namespace.clone()])),
    );
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "lookup");
    assert_eq!(tools[1], namespace);
}

#[test]
fn exact_standard_http_create_response_satisfies_the_named_public_schema_view() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let mut request = turn_declaring(serde_json::json!([responses_lite_namespace()]));
    request.chat_history = vec![image_message("describe")];
    let body = encoded_body_of(&wire, request, Mode::Streaming);

    openapi_schema::assert_valid(Schema::HttpCreateResponse, &body);
    assert_string_map(&body["client_metadata"], "client_metadata");

    let mut bad_role = body.clone();
    bad_role["input"][0]["role"] = serde_json::json!("tool");
    openapi_schema::assert_invalid(Schema::HttpCreateResponse, &bad_role);
    let mut bad_content = body.clone();
    bad_content["input"][0]["content"] = serde_json::json!(42);
    openapi_schema::assert_invalid(Schema::HttpCreateResponse, &bad_content);
    let mut bad_tools = body.clone();
    bad_tools["tools"][0]["name"] = serde_json::json!(42);
    openapi_schema::assert_invalid(Schema::HttpCreateResponse, &bad_tools);
    let mut bad_image = body;
    let image = bad_image["input"]
        .as_array_mut()
        .expect("input is an array")
        .iter_mut()
        .flat_map(|item| item["content"].as_array_mut().into_iter().flatten())
        .find(|content| content["type"] == "input_image")
        .expect("the emitted request carries its image");
    image["detail"] = serde_json::json!("maximum");
    openapi_schema::assert_invalid(Schema::HttpCreateResponse, &bad_image);
}

/// HTTP Lite uses the wire identity for its deterministic prefix, carries the
/// marker only as an HTTP header, and removes top-level tools/instructions.
#[test]
fn responses_lite_http_uses_the_stable_identity_and_http_only_marker() {
    let wire = OpenAI::with_key(&CHATGPT, "test-token")
        .with_instructions("provider base")
        .responses("gpt-5.4")
        .with_responses_lite()
        .expect("the ChatGPT dialect supports Lite")
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let mut encoded = wire
        .encode(
            turn(vec![Message::system("caller base"), image_message("hello")]),
            Mode::Streaming,
        )
        .expect("the Lite request encodes");
    let request = encoded.requests.remove(0);

    assert_eq!(
        request.headers()[super::super::responses_lite::HTTP_HEADER],
        "true"
    );
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(body).expect("the body is JSON");
    assert_eq!(body["input"][0]["type"], "additional_tools");
    assert_eq!(body["input"][0]["role"], "developer");
    assert_eq!(body["input"][1]["type"], "message");
    assert_eq!(body["input"][1]["role"], "developer");
    assert_eq!(
        body["input"][1]["internal_chat_message_metadata_passthrough"]["content_item_kinds"],
        serde_json::json!(["model.base_instructions"])
    );
    assert_eq!(
        body["input"][1]["content"][0]["text"],
        "provider base\n\ncaller base"
    );
    assert_eq!(
        body["input"][1]["id"], "msg_3333557a-9135-5aeb-b8f2-d31c4d718f4b",
        "the developer id hashes the exact merged instruction bytes"
    );
    assert!(body.get("tools").is_none());
    assert!(body.get("instructions").is_none());
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["reasoning"]["context"], "all_turns");
    // Lite created `reasoning`, so the shared include rule follows it.
    assert_eq!(
        body["include"],
        serde_json::json!(["reasoning.encrypted_content"])
    );
    assert_eq!(body["client_metadata"]["thread_id"], "thread-derived");
    assert!(body["input"][2]["content"][1].get("detail").is_none());
    openapi_schema::assert_valid(Schema::LiteHttpCreateResponse, &body);
    assert_string_map(&body["client_metadata"], "client_metadata");

    #[cfg(feature = "websocket")]
    {
        let handshake = derived_identity()
            .handshake_request(&wire)
            .expect("the handshake builds");
        assert!(
            handshake
                .headers()
                .get(super::super::responses_lite::HTTP_HEADER)
                .is_none(),
            "the HTTP Lite header must not reach the websocket handshake"
        );
    }
}

fn lite_refusal(error: EncodeError) -> super::super::responses_lite::ResponsesLiteError {
    let crate::error::ProviderError::Request(reason) = crate::error::ProviderError::from(error)
    else {
        panic!("a Lite encode refusal is a request error");
    };
    reason
        .downcast::<super::super::responses_lite::ResponsesLiteError>()
        .map(|error| *error)
        .unwrap_or_else(|reason| panic!("expected a named Lite refusal, got {reason}"))
}

/// Builder validation rejects non-Codex selection, while encode-time
/// validation catches serde/public-field construction and missing identity.
#[test]
fn responses_lite_selection_is_revalidated_at_encode_time() {
    use super::super::responses_lite::{CodexRequestShape, ResponsesLiteError};

    assert!(matches!(
        openai().with_responses_lite(),
        Err(ResponsesLiteError::ResponsesLiteRequiresCodex { dialect: "openai" })
    ));

    let mut by_field = openai();
    by_field.codex_request_shape = CodexRequestShape::ResponsesLite;
    let mut serialized = serde_json::to_value(openai()).expect("the wire serializes");
    serialized["codex_request_shape"] = serde_json::json!("responses_lite");
    let by_serde: Responses = serde_json::from_value(serialized).expect("the wire deserializes");
    for wire in [by_field, by_serde] {
        let error = wire
            .encode(prompt(), Mode::Unary)
            .expect_err("a non-Codex Lite wire is refused");
        assert!(matches!(
            lite_refusal(error),
            ResponsesLiteError::ResponsesLiteRequiresCodex { dialect: "openai" }
        ));
    }

    let missing_identity = chatgpt()
        .with_responses_lite()
        .expect("the ChatGPT dialect supports Lite")
        .encode(prompt(), Mode::Unary)
        .expect_err("HTTP Lite needs a stable identity");
    assert!(matches!(
        lite_refusal(missing_identity),
        ResponsesLiteError::ResponsesLiteRequiresIdentity
    ));

    let input_system = chatgpt()
        .with_system_instructions_as_messages()
        .with_responses_lite()
        .expect("the ChatGPT dialect supports Lite")
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract")
        .encode(prompt(), Mode::Unary)
        .expect_err("Lite refuses InputSystemMessages");
    assert!(matches!(
        lite_refusal(input_system),
        ResponsesLiteError::InputSystemMessages
    ));
}

/// Without an identity the ChatGPT dialect keeps sending a fresh
/// `session_id` per request and no dashed identity.
#[test]
fn without_a_codex_identity_each_request_keeps_its_own_session_id() {
    let first = encoded_request(&chatgpt(), Mode::Unary);
    let second = encoded_request(&chatgpt(), Mode::Unary);
    assert!(first.headers().get("session-id").is_none());
    assert!(first.headers().get("thread-id").is_none());
    assert_ne!(
        first.headers()["session_id"],
        second.headers()["session_id"]
    );
}

/// A websocket handshake and an HTTP request carrying one identity name the
/// conversation with the same three headers.
#[cfg(feature = "websocket")]
#[test]
fn http_and_the_websocket_handshake_carry_one_identity_alike() {
    let identity = derived_identity();
    let wire = chatgpt()
        .with_codex_identity(identity.clone())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let http = encoded_request(&wire, Mode::Streaming);
    let handshake = identity
        .handshake_request(&wire)
        .expect("the handshake builds");
    for name in ["session-id", "thread-id", "x-client-request-id"] {
        assert_eq!(http.headers()[name], handshake.headers()[name], "{name}");
    }
}

#[test]
fn a_codex_identity_is_refused_on_a_wire_that_does_not_speak_the_codex_contract() {
    let error = openai()
        .with_codex_identity(derived_identity())
        .expect_err("the OpenAI dialect is not Codex");
    assert_eq!(error.dialect, "openai");
}

/// Caller-derived ids are sent verbatim, so an id no header can carry, or an
/// empty one, is refused by name, both when built and when deserialized.
#[test]
fn a_codex_identity_refuses_ids_no_header_can_carry() {
    use super::super::codex_identity::CodexIdentity;
    let empty = CodexIdentity::from_ids("", "thread").expect_err("empty");
    assert_eq!(empty.field, "session_id");
    let broken = CodexIdentity::from_ids("session", "line\nbreak").expect_err("not header-safe");
    assert_eq!(broken.field, "thread_id");

    let identity = derived_identity();
    let json = serde_json::to_value(&identity).expect("serializes");
    assert_eq!(
        json,
        serde_json::json!({"session_id": "session-derived", "thread_id": "thread-derived"})
    );
    assert_eq!(
        serde_json::from_value::<CodexIdentity>(json).expect("deserializes"),
        identity
    );
    assert!(
        serde_json::from_value::<CodexIdentity>(
            serde_json::json!({"session_id": "", "thread_id": "t"})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<CodexIdentity>(
            serde_json::json!({"session_id": "s", "thread_id": "line\nbreak"})
        )
        .is_err(),
        "an id no header can carry is refused on the way in, too"
    );
}

/// An identity that reaches a non-Codex wire other than through
/// `with_codex_identity`, by field assignment or by serde, is refused by name
/// when the request is encoded, so nothing is stamped or sent.
#[test]
fn a_codex_identity_on_a_non_codex_wire_is_refused_when_encoding() {
    use super::super::codex_identity::NotACodexWire;
    let mut by_field = openai();
    by_field.codex_identity = Some(derived_identity());
    let mut json = serde_json::to_value(openai()).expect("serializes");
    json["codex_identity"] = serde_json::to_value(derived_identity()).expect("serializes");
    let by_serde: Responses = serde_json::from_value(json).expect("deserializes");

    for wire in [by_field, by_serde] {
        let Err(error) = wire.encode(prompt(), Mode::Unary) else {
            panic!("an OpenAI wire does not speak the Codex contract");
        };
        let crate::error::ProviderError::Request(reason) = crate::error::ProviderError::from(error)
        else {
            panic!("an encode refusal is a request error");
        };
        assert_eq!(
            reason.downcast_ref::<NotACodexWire>(),
            Some(&NotACodexWire { dialect: "openai" })
        );
    }
}

/// Over HTTP, as over the websocket, a request's own cache key and metadata
/// keys are kept; the identity fills only what is missing.
#[test]
fn a_callers_own_cache_key_and_metadata_are_kept_over_http() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let mut request = prompt();
    request.additional_params = Some(serde_json::json!({
        "prompt_cache_key": "caller-key",
        "client_metadata": {"session_id": "caller-session"},
    }));
    let mut encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let request = encoded.requests.remove(0);
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(body).expect("the body is JSON");
    assert_eq!(body["prompt_cache_key"], "caller-key");
    assert_eq!(
        body["client_metadata"],
        serde_json::json!({"session_id": "caller-session", "thread_id": "thread-derived"})
    );
}

// ── provider items at encode ──────────────────────────────────────────────

fn with_provider_item(provider: Option<&str>) -> CompletionRequest {
    turn(vec![
        Message::user("hi"),
        Message::Assistant {
            id: Some("msg_1".to_owned()),
            content: vec![message::AssistantContent::ProviderItem(
                message::ProviderItem {
                    item: serde_json::json!({"type": "web_search_call", "id": "ws_1"}),
                    provider: provider.map(str::to_owned),
                },
            )],
        },
        Message::user("again"),
    ])
}

/// The Responses wire replays an item it issued, or one of unknown
/// provenance, verbatim, and refuses by name an item another issuer
/// produced.
#[test]
fn the_responses_wire_replays_its_own_and_unknown_provider_items_and_refuses_others() {
    for provider in [Some("openai"), None] {
        let body = encoded_body_of(&openai(), with_provider_item(provider), Mode::Unary);
        let input = body["input"].as_array().expect("input items");
        assert!(
            input.contains(&serde_json::json!({"type": "web_search_call", "id": "ws_1"})),
            "{provider:?}: {body}"
        );
    }
    let error = openai()
        .encode(with_provider_item(Some("anthropic")), Mode::Unary)
        .expect_err("another issuer's item cannot be replayed");
    let crate::error::ProviderError::Request(inner) = crate::error::ProviderError::from(error)
    else {
        panic!("expected a request refusal");
    };
    let refusal = inner
        .downcast_ref::<message::UnreplayableProviderItem>()
        .unwrap_or_else(|| panic!("expected UnreplayableProviderItem, got {inner}"));
    assert_eq!(refusal.issuer.as_deref(), Some("anthropic"));
    assert_eq!(refusal.item_type.as_deref(), Some("web_search_call"));
}
