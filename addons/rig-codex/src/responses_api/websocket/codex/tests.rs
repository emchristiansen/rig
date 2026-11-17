use super::*;
use crate::responses_api::request_headers::RequestHeaders;
use crate::responses_api::streaming::ItemChunkKind;
use crate::responses_api::websocket::test_connection::Script;
use crate::responses_api::{
    CompletionResponse, IncompleteDetailsReason, OutputTokensDetails, ResponseObject,
    ResponsesUsage, SystemInstructionsPlacement, input_items,
};
use crate::wire::{CallerIdentity, MissingCallerIdentity};
use rig_core::message::{CallId, ProviderCallId, ToolName};
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use serde_json::json;
use std::sync::Mutex;

const ACCESS_TOKEN: &str = "test-token";
const ACCOUNT_ID: &str = "acct-123";

/// The caller identity these tests send: no `version`.
fn test_caller_identity() -> CallerIdentity {
    CallerIdentity::new("rig-codex-tests", "rig-codex-tests/0.0 (test)", None)
        .expect("a valid caller identity")
}

/// A ChatGPT provider with the test credential and account.
fn provider() -> OpenAIConfig {
    OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN).with_account_id(ACCOUNT_ID)
}

/// Codex settings over `provider`, with no caller identity.
fn anonymous_wire(provider: OpenAIConfig) -> CodexResponses {
    CodexResponses::new(Responses::new(provider, chatgpt::GPT_5_3_CODEX))
        .expect("the ChatGPT dialect speaks the Codex contract")
}

fn codex_wire() -> CodexResponses {
    anonymous_wire(provider()).with_caller_identity(test_caller_identity())
}

/// The conversation identity a test names when it does not care which.
fn identity() -> CodexIdentity {
    CodexIdentity::from_ids("session-1", "thread-1").expect("header-safe ids")
}

fn session_over(script: &Script) -> CodexWebSocketSession {
    CodexWebSocketSession::from_connection(codex_wire(), identity(), script.connection(), None)
        .expect("the ChatGPT dialect speaks the Codex contract")
}

fn user_request(text: &str) -> CompletionRequest {
    CompletionRequest::new(Message::user(text))
}

/// A request whose whole history is `messages`.
fn request_of(messages: Vec<Message>) -> CompletionRequest {
    let mut request = user_request("unused");
    request.chat_history = messages.into_iter().map(Arc::new).collect();
    request
}

fn delta(text: &str) -> InputDelta {
    let items = input_items(&codex_wire().wire, user_request(text))
        .expect("a user message converts into input items");
    InputDelta::new(items).expect("the delta carries an item")
}

/// A full turn read to its end: what these tests used the pin's
/// `completion()` for. It returns the turn's terminal event.
trait CompleteTurn {
    async fn completion(
        &mut self,
        request: CompletionRequest,
    ) -> Result<ResponsesWebSocketEvent, CodexSessionError>;
}

impl CompleteTurn for CodexWebSocketSession {
    async fn completion(
        &mut self,
        request: CompletionRequest,
    ) -> Result<ResponsesWebSocketEvent, CodexSessionError> {
        self.send(request).await?;
        loop {
            let event = self.next_event().await?;
            if event.is_terminal() {
                return Ok(event);
            }
        }
    }
}

fn response(response_id: &str, status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: response_id.to_string(),
        object: ResponseObject::Response,
        provider_request_id: None,
        created_at: 0,
        status,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: chatgpt::GPT_5_3_CODEX.to_string(),
        usage: Some(ResponsesUsage {
            input_tokens: 1,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 3,
        }),
        output: Vec::new(),
        tools: Vec::new(),
        additional_parameters: Default::default(),
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
    }
}

fn terminal_event(kind: &str, response: CompletionResponse) -> String {
    json!({
        "type": kind,
        "sequence_number": 9,
        "response": serde_json::to_value(response).expect("response should serialize"),
    })
    .to_string()
}

fn completed(response_id: &str) -> String {
    terminal_event(
        "response.completed",
        response(response_id, ResponseStatus::Completed),
    )
}

fn failed(response_id: &str) -> String {
    terminal_event(
        "response.failed",
        response(response_id, ResponseStatus::Failed),
    )
}

fn incomplete(response_id: &str) -> String {
    let mut response = response(response_id, ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    terminal_event("response.incomplete", response)
}

fn text_delta(text: &str) -> String {
    json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": text,
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1,
    })
    .to_string()
}

/// Read events until the in-flight turn ends.
async fn finish_turn(session: &mut CodexWebSocketSession) -> ResponsesWebSocketEvent {
    loop {
        let event = session.next_event().await.expect("event should arrive");
        if event.is_terminal() {
            return event;
        }
    }
}

/// The named refusal inside a refused incremental send.
fn refusal(error: &ProviderError) -> IncrementalSendRefused {
    match error {
        ProviderError::Request(inner) => inner
            .downcast_ref::<IncrementalSendRefused>()
            .cloned()
            .unwrap_or_else(|| panic!("expected an incremental refusal, got {inner}")),
        other => panic!("expected a named request refusal, got {other:?}"),
    }
}
// --- identity ---------------------------------------------------------------

/// The handshake is exactly the Codex set: the wire's credential, its caller
/// identity (the same `originator` and `user-agent` its HTTP requests carry),
/// the account, the dashed identity headers, the correlation header and the
/// beta opt-in : nothing else (the backend adds the websocket handshake
/// headers itself). Strict, so a header added to the HTTP set later cannot
/// reach the handshake unnoticed.
#[test]
fn the_handshake_carries_exactly_the_codex_identity_headers() {
    let identity = identity();
    let wire = codex_wire();
    let caller = test_caller_identity();
    let (originator, user_agent) = (
        caller.originator().to_owned(),
        caller.user_agent().to_owned(),
    );
    let request = identity
        .handshake_request(&wire)
        .expect("handshake request should build");

    assert_eq!(request.method(), http::Method::GET);
    assert_eq!(
        request.uri(),
        "wss://chatgpt.com/backend-api/codex/responses"
    );

    let mut headers: Vec<(String, String)> = request
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                value.to_str().expect("ascii header").to_owned(),
            )
        })
        .collect();
    headers.sort();
    let mut expected = vec![
        ("authorization".to_owned(), format!("Bearer {ACCESS_TOKEN}")),
        ("chatgpt-account-id".to_owned(), ACCOUNT_ID.to_owned()),
        ("originator".to_owned(), originator),
        ("user-agent".to_owned(), user_agent),
        (
            "openai-beta".to_owned(),
            "responses_websockets=2026-02-06".to_owned(),
        ),
        ("session-id".to_owned(), identity.session_id().to_owned()),
        ("thread-id".to_owned(), identity.thread_id().to_owned()),
        (
            "x-client-request-id".to_owned(),
            identity.thread_id().to_owned(),
        ),
    ];
    expected.sort();
    assert_eq!(headers, expected);
}

/// Without the caller's exact identity the handshake is refused before it is
/// sent, rather than identifying as Rig.
#[test]
fn the_handshake_refuses_without_the_caller_s_exact_identity() {
    let error = identity()
        .handshake_request(&anonymous_wire(provider()))
        .expect_err("no exact identity, no handshake");
    match ProviderError::from(error) {
        ProviderError::Request(inner) => assert_eq!(
            inner.downcast_ref::<MissingCallerIdentity>(),
            Some(&MissingCallerIdentity { dialect: "chatgpt" })
        ),
        other => panic!("expected a request refusal, got {other:?}"),
    }
}

/// A Codex session over a connection the caller opened still refuses to
/// send without the caller's exact identity on its wire, before any frame is
/// written.
#[tokio::test]
async fn a_codex_session_over_an_open_connection_refuses_to_send_without_the_caller_s_identity() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = CodexWebSocketSession::from_connection(
        anonymous_wire(OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN)),
        identity(),
        script.connection(),
        None,
    )
    .expect("the ChatGPT dialect speaks the Codex contract");
    match session
        .send(user_request("hello"))
        .await
        .expect_err("no exact identity, no frame")
    {
        ProviderError::Request(inner) => assert_eq!(
            inner.downcast_ref::<MissingCallerIdentity>(),
            Some(&MissingCallerIdentity { dialect: "chatgpt" })
        ),
        other => panic!("expected a request refusal, got {other:?}"),
    }
    assert!(script.sent().is_empty(), "nothing was written");
}

#[test]
fn the_handshake_omits_the_account_header_without_an_account() {
    let wire = anonymous_wire(OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN))
        .with_caller_identity(test_caller_identity());
    let request = identity()
        .handshake_request(&wire)
        .expect("handshake request should build");
    assert!(request.headers().get(CHATGPT_ACCOUNT_ID_HEADER).is_none());
}

#[test]
fn identities_key_the_cache_by_thread() {
    let identity = identity();
    assert_ne!(identity.session_id(), identity.thread_id());
    assert_eq!(identity.prompt_cache_key(), identity.thread_id());
}

#[test]
fn a_non_codex_wire_is_refused_by_name() {
    let openai = Responses::new(OpenAIConfig::new("key"), "gpt-5.4");
    assert_eq!(
        CodexResponses::new(openai.clone()).expect_err("an OpenAI wire is not a Codex wire"),
        NotACodexWire { dialect: "openai" }
    );
    // The fields are public, so a wire can still be put together by hand.
    let wire = CodexResponses {
        wire: openai,
        ..codex_wire()
    };
    let Err(error) = CodexWebSocketSessionBuilder::new(wire.clone()) else {
        panic!("an OpenAI wire does not speak the Codex contract");
    };
    assert_eq!(error, NotACodexWire { dialect: "openai" });
    assert!(
        CodexWebSocketSession::from_connection(wire, identity(), Script::new().connection(), None)
            .is_err()
    );
}

// --- root sends ---------------------------------------------------------------

/// Root full sends never chain, and every frame carries the same identity.
#[tokio::test]
async fn root_sends_never_chain_and_carry_one_identity() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("first"))
        .await
        .expect("first turn should complete");
    session
        .completion(user_request("second"))
        .await
        .expect("second turn should complete");
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    let identity = session.identity().clone();
    for frame in script.sent_json() {
        assert_eq!(frame["type"], "response.create");
        assert!(
            frame.get("previous_response_id").is_none(),
            "a root full send must not chain, got {frame}"
        );
        assert_eq!(frame["prompt_cache_key"], identity.thread_id());
        assert_eq!(
            frame["client_metadata"],
            json!({
                "session_id": identity.session_id(),
                "thread_id": identity.thread_id(),
            })
        );
        // The official client states `stream: true` on every frame; the
        // SSE-only `background` stays off the wire.
        assert_eq!(frame["stream"], json!(true), "got {frame}");
        assert!(frame.get("background").is_none(), "got {frame}");
        assert!(frame.get("generate").is_none(), "got {frame}");
    }
}

#[tokio::test]
async fn a_caller_supplied_cache_key_wins_and_the_metadata_is_still_stamped() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    let mut request = user_request("hello");
    request.additional_params = Some(json!({ "prompt_cache_key": "caller-key" }));
    session.send(request).await.expect("send should succeed");

    let frame = &script.sent_json()[0];
    assert_eq!(frame["prompt_cache_key"], "caller-key");
    assert_eq!(
        frame["client_metadata"]["thread_id"],
        session.identity().thread_id()
    );
}

#[tokio::test]
async fn warmup_is_a_root_send_with_generate_false() {
    let script = Script::new().turn([completed("resp_warm")]);
    let mut session = session_over(&script);

    let response_id = session
        .warmup(user_request("prewarm"))
        .await
        .expect("warmup should complete");
    assert_eq!(response_id, "resp_warm");

    let frame = &script.sent_json()[0];
    assert_eq!(frame["generate"], false);
    assert!(frame.get("previous_response_id").is_none());
}

// --- Codex shaping on the websocket lane ---------------------------------------

/// The `response.create` frame the adapter actually writes carries the Codex
/// request shaping: `store: false` stated, the encrypted-reasoning `include`
/// following the rule every dialect shares (absent without `reasoning`,
/// requested alongside it), no sampling or output-cap key when the caller set
/// none, and the cache identity stamped from the session's one identity.
#[tokio::test]
async fn the_response_create_frame_carries_the_codex_shaping() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("plain"))
        .await
        .expect("a plain turn should complete");
    let mut reasoning = user_request("reasoned");
    reasoning.additional_params = Some(json!({ "reasoning": { "effort": "low" } }));
    session
        .completion(reasoning)
        .await
        .expect("a reasoning turn should complete");

    let identity = session.identity().clone();
    let frames = script.sent_json();
    assert_eq!(frames.len(), 2, "one frame per turn: {frames:?}");
    for frame in &frames {
        assert_eq!(frame["type"], "response.create");
        assert_eq!(frame["store"], json!(false), "store is stated: {frame}");
        assert_eq!(frame["stream"], json!(true), "stream is stated: {frame}");
        for unset in ["top_p", "temperature", "max_output_tokens"] {
            assert!(
                frame.get(unset).is_none(),
                "no `{unset}` key when the caller set none: {frame}"
            );
        }
        assert_eq!(frame["prompt_cache_key"], identity.prompt_cache_key());
        assert_eq!(
            frame["client_metadata"],
            json!({
                "session_id": identity.session_id(),
                "thread_id": identity.thread_id(),
            })
        );
    }
    assert_eq!(
        frames[1]["include"],
        json!(["reasoning.encrypted_content"]),
        "the encrypted-reasoning include rides alongside reasoning"
    );
}

/// The pin stated the encrypted-reasoning `include` only alongside
/// `reasoning`, as the official client does.
///
/// Not provided on v0.46.0: Rig's encoder states the include on every Codex
/// request, because the contract stores nothing. The session sends what the
/// encoder wrote.
#[tokio::test]
#[ignore = "v0.46.0's encoder states the encrypted-reasoning include on every Codex request"]
async fn no_include_is_forced_onto_a_turn_without_reasoning() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);
    session
        .completion(user_request("plain"))
        .await
        .expect("a plain turn should complete");
    let frames = script.sent_json();
    assert!(
        frames[0].get("include").is_none(),
        "no include is forced onto a turn without reasoning: {}",
        frames[0]
    );
}

/// A caller-set `top_p` (and likewise `temperature`) is refused by name on the
/// websocket lane before any frame is written, and the refusal leaves the
/// session free to send the next turn.
#[tokio::test]
async fn a_caller_set_sampling_control_is_refused_by_name_before_anything_is_sent() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    let mut top_p = user_request("hello");
    top_p.additional_params = Some(json!({ "top_p": 0.5 }));
    let mut temperature = user_request("hello");
    temperature.temperature = Some(0.25);

    for (request, control) in [
        (top_p, UnsupportedCodexControl::TopP),
        (temperature, UnsupportedCodexControl::Temperature),
    ] {
        let error = session
            .send(request)
            .await
            .expect_err("the Codex contract refuses a caller-set sampling control");
        let ProviderError::Request(source) = &error else {
            panic!("{control:?}: an encode refusal is a request error, got {error:?}");
        };
        assert_eq!(
            source.downcast_ref::<UnsupportedCodexControl>(),
            Some(&control),
            "expected the named refusal, got {source}"
        );
        assert!(
            script.sent().is_empty(),
            "{control:?}: nothing is written before the refusal"
        );
    }

    session
        .completion(user_request("after"))
        .await
        .expect("a refused send leaves the session free to send");
    assert_eq!(script.sent().len(), 1);
}

// --- incremental sends --------------------------------------------------------

/// A delta chains the tip, carries only the new input, and reuses every other
/// field of the captured root envelope, including the one identity.
#[tokio::test]
async fn an_incremental_send_chains_the_tip_and_reuses_the_captured_envelope() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT_CONTEXT_MARKER"))
        .await
        .expect("root turn should complete");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    session
        .send_incremental(delta("FIRST_DELTA_MARKER"))
        .await
        .expect("incremental turn should send");
    finish_turn(&mut session).await;
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    // The captured envelope survives a completed continuation.
    session
        .send_incremental(delta("SECOND_DELTA_MARKER"))
        .await
        .expect("a second incremental turn should send");
    finish_turn(&mut session).await;
    assert_eq!(session.previous_response_id(), Some("resp_3"));

    let frames = script.sent_json();
    let root = frames[0].as_object().expect("root frame is an object");
    for (index, expected_tip, marker) in [
        (1, "resp_1", "FIRST_DELTA_MARKER"),
        (2, "resp_2", "SECOND_DELTA_MARKER"),
    ] {
        let frame = frames[index].as_object().expect("delta frame is an object");
        assert_eq!(frame["type"], "response.create");
        assert_eq!(frame["previous_response_id"], expected_tip);

        let input = serde_json::to_string(&frame["input"]).expect("input serializes");
        assert!(
            input.contains(marker),
            "the delta is the input, got {input}"
        );
        assert!(
            !input.contains("ROOT_CONTEXT_MARKER"),
            "a delta must not replay the root turn, got {input}"
        );

        // Every other field is the root envelope's, byte for byte.
        let mut rest = frame.clone();
        remove_preserving_order(&mut rest, "input");
        remove_preserving_order(&mut rest, "previous_response_id");
        let mut root_rest = root.clone();
        remove_preserving_order(&mut root_rest, "input");
        assert_eq!(
            rest, root_rest,
            "the delta must reuse the captured envelope"
        );
    }
}

#[tokio::test]
async fn an_incremental_send_without_a_completed_turn_is_refused_and_writes_nothing() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    let error = session
        .send_incremental(delta("PREMATURE_DELTA"))
        .await
        .expect_err("there is no tip to continue");
    assert_eq!(refusal(&error), IncrementalSendRefused::NoCompletedTurn);
    assert!(script.sent().is_empty(), "a refused delta writes nothing");

    session
        .completion(user_request("ROOT_AFTER_REFUSAL"))
        .await
        .expect("a root send still works");
    let sent = script.sent();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].contains("ROOT_AFTER_REFUSAL"));
}

#[tokio::test]
async fn an_incremental_send_while_a_turn_is_in_flight_is_refused() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    session
        .send(user_request("ROOT"))
        .await
        .expect("root turn should send");
    let error = session
        .send_incremental(delta("DELTA"))
        .await
        .expect_err("one turn at a time");
    assert!(
        error.to_string().contains("already in flight"),
        "got {error}"
    );
    assert_eq!(script.sent().len(), 1);

    finish_turn(&mut session).await;
}

#[tokio::test]
async fn a_second_root_send_while_a_turn_is_in_flight_is_refused() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    session
        .send(user_request("ROOT"))
        .await
        .expect("root turn should send");
    session
        .send(user_request("ROOT_AGAIN"))
        .await
        .expect_err("single in-flight custody");
    assert_eq!(script.sent().len(), 1);
}

#[tokio::test]
async fn an_incremental_send_after_a_failed_turn_is_refused() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([failed("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    session
        .send_incremental(delta("FIRST_DELTA"))
        .await
        .expect("incremental turn should send");
    finish_turn(&mut session).await;
    assert_eq!(session.previous_response_id(), None);

    let error = session
        .send_incremental(delta("SECOND_DELTA"))
        .await
        .expect_err("a failed turn leaves no tip");
    assert_eq!(
        refusal(&error),
        IncrementalSendRefused::LastTurnNotCompleted
    );
    assert_eq!(script.sent().len(), 2);
}

/// A Codex-wrapped error event that reports an HTTP status still ends the turn
/// without a tip: the status it now carries changes nothing about incremental
/// eligibility, so the next delta is refused by name.
#[tokio::test]
async fn an_incremental_send_after_a_status_bearing_error_event_is_refused() {
    let wrapped_error = json!({
        "type": "error",
        "status": 429,
        "error": {"type": "usage_limit_reached", "message": "The usage limit has been reached"},
        "headers": {"x-codex-primary-used-percent": "100.0"}
    })
    .to_string();
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([wrapped_error]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    session
        .send_incremental(delta("FIRST_DELTA"))
        .await
        .expect("incremental turn should send");
    let ResponsesWebSocketEvent::Error(event) = finish_turn(&mut session).await else {
        panic!("the wrapped error ends the turn");
    };
    assert_eq!(event.status, Some(429));
    assert_eq!(session.previous_response_id(), None);

    let error = session
        .send_incremental(delta("SECOND_DELTA"))
        .await
        .expect_err("an error terminal leaves no tip");
    assert_eq!(
        refusal(&error),
        IncrementalSendRefused::LastTurnNotCompleted
    );
    assert_eq!(script.sent().len(), 2);
}

#[tokio::test]
async fn clearing_the_tip_refuses_incremental_sends_until_a_root_send() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    session.clear_previous_response_id();
    let error = session
        .send_incremental(delta("DELTA"))
        .await
        .expect_err("the tip was cleared");
    assert_eq!(refusal(&error), IncrementalSendRefused::TipCleared);

    session
        .completion(user_request("ROOT_AGAIN"))
        .await
        .expect("a root send re-establishes a tip");
    session
        .send_incremental(delta("DELTA"))
        .await
        .expect("the new tip is eligible");
    finish_turn(&mut session).await;
    assert_eq!(script.sent_json()[2]["previous_response_id"], "resp_2");
}

/// Clearing the tip while a turn is in flight drops that turn's envelope: its
/// completion does not make it eligible behind the caller's back.
#[tokio::test]
async fn clearing_the_tip_mid_turn_survives_the_turns_completion() {
    let script = Script::new().turn([completed("resp_1")]);
    let mut session = session_over(&script);

    session
        .send(user_request("ROOT"))
        .await
        .expect("root turn should send");
    session.clear_previous_response_id();
    finish_turn(&mut session).await;

    let error = session
        .send_incremental(delta("DELTA"))
        .await
        .expect_err("the cleared turn is not an eligible tip");
    assert_eq!(refusal(&error), IncrementalSendRefused::TipCleared);
}

// --- incomplete turns (B7, and B3's websocket side) ---------------------------

/// An incomplete turn is a truthful terminal: its partial output survives and
/// its finish reason says why it stopped. Its response ID is kept as evidence,
/// but it is not an eligible tip, so a delta is refused by name until a full
/// send establishes one : after which deltas continue normally.
#[tokio::test]
async fn an_incomplete_turn_is_a_truthful_terminal_that_blocks_deltas_until_a_full_send() {
    let script = Script::new()
        .turn([text_delta("partial"), incomplete("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);

    session
        .send(user_request("ROOT"))
        .await
        .expect("the root turn sends");
    let ResponsesWebSocketEvent::Item(partial) = session
        .next_event()
        .await
        .expect("the streamed partial output survives")
    else {
        panic!("the partial output is an item event");
    };
    assert!(
        matches!(&partial.data, ItemChunkKind::OutputTextDelta(text) if text.delta == "partial"),
        "the streamed partial output survives, got {partial:?}"
    );
    let ResponsesWebSocketEvent::Response(truncated) = session
        .next_event()
        .await
        .expect("an incomplete turn is a terminal, not a failure")
    else {
        panic!("an incomplete turn ends with its response");
    };
    assert!(matches!(
        truncated.kind,
        ResponseChunkKind::ResponseIncomplete
    ));
    assert_eq!(
        truncated
            .response
            .incomplete_details
            .as_ref()
            .map(|details| details.reason.as_str()),
        Some("max_output_tokens"),
        "the terminal says why the turn stopped"
    );
    assert_eq!(truncated.response.id, "resp_1");
    assert_eq!(
        session.previous_response_id(),
        Some("resp_1"),
        "the incomplete response's ID is kept as terminal evidence"
    );

    let error = session
        .send_incremental(delta("DELTA_AFTER_INCOMPLETE"))
        .await
        .expect_err("an incomplete turn is not an eligible tip");
    assert_eq!(
        refusal(&error),
        IncrementalSendRefused::LastTurnIncomplete {
            response_id: Some("resp_1".to_string())
        }
    );
    assert!(error.to_string().contains("incomplete"), "got {error}");
    assert_eq!(script.sent().len(), 1, "the refused delta wrote nothing");

    // Recovery: a full send establishes an eligible tip again.
    session
        .completion(user_request("ROOT_RECOVERY"))
        .await
        .expect("the recovery root should complete");
    let recovery = &script.sent_json()[1];
    assert!(
        recovery.get("previous_response_id").is_none(),
        "the recovery is a root send, got {recovery}"
    );
    session
        .send_incremental(delta("DELTA_AFTER_RECOVERY"))
        .await
        .expect("the recovered tip is eligible");
    finish_turn(&mut session).await;
    assert_eq!(script.sent_json()[2]["previous_response_id"], "resp_2");
    assert_eq!(session.previous_response_id(), Some("resp_3"));
}

/// The same holds when the truncated turn was itself a continuation: the
/// envelope it continued is no longer eligible.
#[tokio::test]
async fn an_incremental_turn_that_ends_incomplete_blocks_further_deltas() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([incomplete("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    session
        .send_incremental(delta("DELTA"))
        .await
        .expect("incremental turn should send");
    let terminal = finish_turn(&mut session).await;
    assert_eq!(terminal.response_id(), Some("resp_2"));
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    let error = session
        .send_incremental(delta("NEXT_DELTA"))
        .await
        .expect_err("an incomplete continuation is not an eligible tip");
    assert_eq!(
        refusal(&error),
        IncrementalSendRefused::LastTurnIncomplete {
            response_id: Some("resp_2".to_string())
        }
    );
}

/// `response.done` carrying an incomplete status settles the same way.
#[tokio::test]
async fn a_done_event_reporting_incomplete_blocks_deltas_too() {
    let script = Script::new().turn([completed("resp_1")]).turn([json!({
        "type": "response.done",
        "response": { "id": "resp_2", "status": "incomplete" },
    })
    .to_string()]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    session
        .send_incremental(delta("DELTA"))
        .await
        .expect("incremental turn should send");
    finish_turn(&mut session).await;

    let error = session
        .send_incremental(delta("NEXT_DELTA"))
        .await
        .expect_err("an incomplete done is not an eligible tip");
    assert_eq!(
        refusal(&error),
        IncrementalSendRefused::LastTurnIncomplete {
            response_id: Some("resp_2".to_string())
        }
    );
}

// --- keepalive through the adapter ------------------------------------------

/// Servicing the idle socket between turns neither disturbs the tip nor the
/// continuation's eligibility, and hands back what it could not model.
#[tokio::test]
async fn keepalive_between_turns_keeps_the_continuation_eligible() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("ROOT"))
        .await
        .expect("root turn should complete");
    script.arrive([crate::ws_client::Frame::Ping(bytes::Bytes::new())]);
    script.arrive_text([
        json!({ "type": "codex.rate_limits", "primary": { "used_percent": 1 } }).to_string(),
        json!({
            "type": "response.done",
            "response": { "id": "resp_1", "status": "completed" },
        })
        .to_string(),
    ]);

    let (recovered, ending) = session.keepalive().await.into_parts();
    assert!(ending.is_none(), "the drain should service the socket");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].kind, "codex.rate_limits");
    assert_eq!(script.pongs_flushed(), 1);

    session
        .send_incremental(delta("DELTA"))
        .await
        .expect("the tip is still eligible after a keepalive");
    finish_turn(&mut session).await;
    assert_eq!(script.sent_json()[1]["previous_response_id"], "resp_1");
}

/// The Codex websocket lane, with strict tools, sends a declared
/// Responses-lite `functions` namespace on the `response.create` frame
/// exactly as written: the empty description survives and no `strict` is
/// added to the namespace or its member.
///
/// Not provided on v0.46.0: Rig's encoder reads each declared tool into its
/// own tool type and writes it back, which leaves an empty namespace
/// `description` out. The session sends what the encoder wrote.
#[tokio::test]
#[ignore = "v0.46.0's encoder rewrites declared tools and drops an empty namespace description"]
async fn a_declared_namespace_reaches_the_codex_frame_verbatim_under_strict_mode() {
    let namespace = json!({
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
    let script = Script::new().turn([completed("resp_1")]);
    let mut wire = codex_wire();
    wire.wire = wire.wire.with_strict_tools();
    let mut session =
        CodexWebSocketSession::from_connection(wire, identity(), script.connection(), None)
            .expect("the ChatGPT dialect speaks the Codex contract");

    let mut request = user_request("hello");
    request.additional_params = Some(json!({ "tools": [namespace.clone()] }));
    session
        .completion(request)
        .await
        .expect("the turn should complete");

    let frames = script.sent_json();
    assert_eq!(frames.len(), 1, "one frame for the turn: {frames:?}");
    assert_eq!(frames[0]["type"], "response.create");
    assert_eq!(frames[0]["tools"], json!([namespace]));
}

/// A websocket backend that records each handshake it is asked to open and
/// answers with a scripted connection.
#[derive(Clone, Default)]
struct HandshakeRecorder {
    handshakes: Arc<Mutex<Vec<http::HeaderMap>>>,
}

impl WebSocketClientExt for HandshakeRecorder {
    fn connect(
        &self,
        request: http_client::Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        self.handshakes
            .lock()
            .expect("unpoisoned")
            .push(request.headers().clone());
        let connection = Script::new().connection();
        async move { Ok(connection) }
    }
}

// --- caller-derived identity -------------------------------------------------

/// A caller-derived identity reaches the websocket verbatim: the handshake
/// headers when the wire names it, and every frame's cache key and metadata.
#[tokio::test]
async fn a_caller_derived_identity_names_the_handshake_and_every_frame() {
    let identity =
        CodexIdentity::from_ids("session-derived", "thread-derived").expect("header-safe ids");
    let wire = codex_wire()
        .with_codex_identity(identity.clone())
        .expect("the ChatGPT dialect speaks the Codex contract");

    let backend = HandshakeRecorder::default();
    let session = CodexWebSocketSessionBuilder::new(wire.clone())
        .expect("the ChatGPT dialect speaks the Codex contract")
        .connect_with(&backend)
        .await
        .expect("the scripted connection opens");
    assert_eq!(
        session.identity(),
        &identity,
        "the wire's identity is the session's"
    );
    let handshake = backend.handshakes.lock().expect("unpoisoned")[0].clone();
    assert_eq!(handshake["session-id"], "session-derived");
    assert_eq!(handshake["thread-id"], "thread-derived");
    assert_eq!(handshake["x-client-request-id"], "thread-derived");

    let script = Script::new().turn([completed("resp_1")]);
    let mut session =
        CodexWebSocketSession::from_connection(wire, identity, script.connection(), None)
            .expect("the ChatGPT dialect speaks the Codex contract");
    session
        .send(user_request("hello"))
        .await
        .expect("send should succeed");
    let frame = &script.sent_json()[0];
    assert_eq!(frame["prompt_cache_key"], "thread-derived");
    assert_eq!(
        frame["client_metadata"],
        json!({"session_id": "session-derived", "thread_id": "thread-derived"})
    );
}

/// `with_identity` overrides the wire's identity for this session only: the
/// handshake and the session carry the override, while the wire's own HTTP
/// requests keep the wire's identity.
#[tokio::test]
async fn with_identity_overrides_the_wire_identity_for_the_session_only() {
    let on_wire = CodexIdentity::from_ids("session-wire", "thread-wire").expect("header-safe ids");
    let override_identity =
        CodexIdentity::from_ids("session-override", "thread-override").expect("header-safe ids");
    let wire = codex_wire()
        .with_codex_identity(on_wire)
        .expect("the ChatGPT dialect speaks the Codex contract");
    let backend = HandshakeRecorder::default();

    let session = CodexWebSocketSessionBuilder::new(wire.clone())
        .expect("the ChatGPT dialect speaks the Codex contract")
        .with_identity(override_identity.clone())
        .connect_with(&backend)
        .await
        .expect("the scripted connection opens");

    assert_eq!(session.identity(), &override_identity);
    let handshakes = backend.handshakes.lock().expect("unpoisoned").clone();
    assert_eq!(handshakes.len(), 1);
    assert_eq!(handshakes[0]["session-id"], "session-override");
    assert_eq!(handshakes[0]["thread-id"], "thread-override");
    assert_eq!(handshakes[0]["x-client-request-id"], "thread-override");

    // The headers the wire's HTTP requests are sent with.
    let mut http = http::HeaderMap::new();
    wire.stamp_headers(wire.codex_identity.as_ref(), &mut http)
        .expect("the wire's headers stamp");
    assert_eq!(http["session-id"], "session-wire");
    assert_eq!(http["thread-id"], "thread-wire");

    let script = Script::new();
    let mut overridden =
        CodexWebSocketSession::from_connection(wire, override_identity, script.connection(), None)
            .expect("the ChatGPT dialect speaks the Codex contract");
    overridden
        .send(user_request("hello"))
        .await
        .expect("the frame sends");
    let frame = &script.sent_json()[0];
    assert_eq!(frame["prompt_cache_key"], "thread-override");
    assert_eq!(frame["client_metadata"]["thread_id"], "thread-override");
}

// ── the caller-supplied identity mechanisms ─────────────────────────────

/// A caller identity, the `version` header and caller request headers.
fn official_wire() -> CodexResponses {
    anonymous_wire(provider())
        .with_caller_identity(
            CallerIdentity::new(
                "client_exec",
                "client_exec/1.2.3 (Linux 6.18; x86_64) xterm (client_exec; 1.2.3)",
                Some("1.2.3".to_owned()),
            )
            .expect("a valid caller identity"),
        )
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-window-id", "thread-1:0")
                .expect("a valid header")
                .with("x-codex-beta-features", "remote_compaction_v2")
                .expect("a valid header"),
        )
}

/// The handshake carries the caller identity exactly as given, its `version`
/// header, and the wire's request headers, beside the Codex set, and nothing
/// else.
#[test]
fn the_handshake_carries_the_caller_identity_its_version_and_the_request_headers() {
    let identity = identity();
    let request = identity
        .handshake_request(&official_wire())
        .expect("handshake request should build");
    let mut headers: Vec<(String, String)> = request
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                value.to_str().expect("ascii header").to_owned(),
            )
        })
        .collect();
    headers.sort();
    let mut expected = vec![
        ("authorization".to_owned(), format!("Bearer {ACCESS_TOKEN}")),
        ("chatgpt-account-id".to_owned(), ACCOUNT_ID.to_owned()),
        ("originator".to_owned(), "client_exec".to_owned()),
        (
            "user-agent".to_owned(),
            "client_exec/1.2.3 (Linux 6.18; x86_64) xterm (client_exec; 1.2.3)".to_owned(),
        ),
        ("version".to_owned(), "1.2.3".to_owned()),
        (
            "openai-beta".to_owned(),
            "responses_websockets=2026-02-06".to_owned(),
        ),
        ("session-id".to_owned(), identity.session_id().to_owned()),
        ("thread-id".to_owned(), identity.thread_id().to_owned()),
        (
            "x-client-request-id".to_owned(),
            identity.thread_id().to_owned(),
        ),
        ("x-codex-window-id".to_owned(), "thread-1:0".to_owned()),
        (
            "x-codex-beta-features".to_owned(),
            "remote_compaction_v2".to_owned(),
        ),
    ];
    expected.sort();
    assert_eq!(headers, expected);
}

/// Frame metadata reaches every frame, full and incremental, over the
/// envelope's own value for the key; a changed value reaches the next frame
/// and a removed key stops.
#[tokio::test]
async fn frame_metadata_is_stamped_on_every_frame_over_the_envelopes_values() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);
    session
        .set_frame_metadata("x-codex-turn-state", "state-1")
        .expect("a caller key");

    let mut request = user_request("root");
    request.additional_params = Some(json!({
        "client_metadata": { "x-codex-turn-state": "stale", "turn_id": "turn-1" }
    }));
    session
        .completion(request)
        .await
        .expect("root turn should complete");

    session
        .set_frame_metadata("x-codex-turn-state", "state-2")
        .expect("a caller key");
    session
        .send_incremental(delta("first delta"))
        .await
        .expect("incremental turn should send");
    finish_turn(&mut session).await;

    assert_eq!(
        session
            .remove_frame_metadata("x-codex-turn-state")
            .as_deref(),
        Some("state-2")
    );
    assert!(session.frame_metadata().is_empty());
    session
        .send_incremental(delta("second delta"))
        .await
        .expect("a second incremental turn should send");
    finish_turn(&mut session).await;

    let identity = session.identity().clone();
    let frames = script.sent_json();
    assert_eq!(
        frames[0]["client_metadata"],
        json!({
            "session_id": identity.session_id(),
            "thread_id": identity.thread_id(),
            "turn_id": "turn-1",
            "x-codex-turn-state": "state-1",
        }),
        "the frame metadata wins over the request's own value"
    );
    assert_eq!(
        frames[1]["client_metadata"]["x-codex-turn-state"], "state-2",
        "an incremental frame carries the current value, not the captured envelope's"
    );
    assert_eq!(
        frames[2]["client_metadata"]["x-codex-turn-state"], "stale",
        "once the key is removed, a frame carries the request's own value: frame \
         metadata is stamped per frame, never into the captured envelope"
    );
}

/// A key the session stamps itself cannot be set as frame metadata.
#[test]
fn a_frame_metadata_key_the_session_stamps_itself_is_refused() {
    let mut session = session_over(&Script::new());
    for key in [
        SESSION_ID_METADATA_KEY,
        THREAD_ID_METADATA_KEY,
        "ws_request_header_x_openai_internal_codex_responses_lite",
        WS_STREAM_REQUEST_START_MS_METADATA_KEY,
    ] {
        assert_eq!(
            session.set_frame_metadata(key, "value"),
            Err(ReservedFrameMetadataKey {
                key: key.to_owned()
            })
        );
        assert!(
            CodexWebSocketSessionBuilder::new(codex_wire())
                .expect("a Codex wire")
                .with_frame_metadata(key, "value")
                .is_err(),
            "the builder refuses `{key}` too"
        );
    }
    assert!(session.frame_metadata().is_empty());
}

/// A clock the test controls: each read returns the next instant.
struct SteppingClock(std::sync::atomic::AtomicU64);

impl FrameClock for SteppingClock {
    fn unix_millis(&self) -> u64 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}

/// With a clock, every frame carries the instant it was encoded at, read
/// from the clock once per frame; without one, no instant is stamped.
#[tokio::test]
async fn the_send_instant_is_read_from_the_injected_clock_once_per_frame() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);
    let clock = Arc::new(SteppingClock(std::sync::atomic::AtomicU64::new(
        1_790_000_000_000,
    )));
    session.set_request_start_ms_stamp(Some(clock.clone()));
    session
        .completion(user_request("root"))
        .await
        .expect("root turn should complete");
    session
        .send_incremental(delta("delta"))
        .await
        .expect("incremental turn should send");
    finish_turn(&mut session).await;
    session.set_request_start_ms_stamp(None);
    session
        .send_incremental(delta("unstamped delta"))
        .await
        .expect("a second incremental turn should send");
    finish_turn(&mut session).await;

    let frames = script.sent_json();
    assert_eq!(
        frames[0]["client_metadata"][WS_STREAM_REQUEST_START_MS_METADATA_KEY],
        "1790000000000"
    );
    assert_eq!(
        frames[1]["client_metadata"][WS_STREAM_REQUEST_START_MS_METADATA_KEY],
        "1790000000001"
    );
    assert!(
        frames[2]["client_metadata"]
            .get(WS_STREAM_REQUEST_START_MS_METADATA_KEY)
            .is_none(),
        "without a clock no instant is stamped, got {}",
        frames[2]
    );
    assert_eq!(
        clock.0.load(std::sync::atomic::Ordering::SeqCst),
        1_790_000_000_002,
        "the clock is read once per stamped frame"
    );
}

/// The builder's clock reaches the session it connects.
#[tokio::test]
async fn the_builders_clock_and_frame_metadata_reach_the_connected_session() {
    let script = Script::new().turn([completed("resp_1")]);
    let backend = ScriptBackend(script.clone());
    let mut session = CodexWebSocketSessionBuilder::new(codex_wire())
        .expect("a Codex wire")
        .with_identity(identity())
        .with_frame_metadata("x-codex-turn-state", "state-1")
        .expect("a caller key")
        .with_request_start_ms_stamp(Arc::new(SteppingClock(std::sync::atomic::AtomicU64::new(
            7,
        ))))
        .connect_with(&backend)
        .await
        .expect("connects");
    session
        .completion(user_request("root"))
        .await
        .expect("root turn should complete");
    let frame = &script.sent_json()[0];
    assert_eq!(frame["client_metadata"]["x-codex-turn-state"], "state-1");
    assert_eq!(
        frame["client_metadata"][WS_STREAM_REQUEST_START_MS_METADATA_KEY],
        "7"
    );
}

/// A backend that hands out the script's connection.
#[derive(Clone)]
struct ScriptBackend(Script);

impl WebSocketClientExt for ScriptBackend {
    fn connect(
        &self,
        _request: http_client::Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let connection = self.0.connection();
        async move { Ok(connection) }
    }
}

// ── the Codex session-start prewarm ─────────────────────────────────────

/// The session's standing request with no conversation yet: its
/// instructions and nothing else, as Codex's startup prewarm builds it
/// (`build_prompt(Vec::new(), …)`).
fn prewarm_request() -> CompletionRequest {
    request_of(vec![Message::system("Standing instructions.")])
}

/// Codex 0.155.1 opens a fresh session with a prewarm: a `generate: false`
/// `response.create` with empty input and the prewarm's metadata, waited on
/// until it completes. The first turn then continues it on the same
/// connection: `previous_response_id` names the prewarm's response, the input
/// is the turn's items alone, and the turn's own metadata replaces the
/// prewarm's. The session reproduces that sequence with `warmup`, the frame
/// metadata and `send_incremental`, and nothing else in the frames changes.
#[tokio::test]
async fn the_codex_prewarm_sequence_is_a_warmup_then_an_incremental_first_turn() {
    let script = Script::new()
        .turn([turn_state_event("ts-prewarm"), completed("resp_prewarm")])
        .turn([completed("resp_turn_1")]);
    let mut session = session_over(&script);

    session
        .set_frame_metadata("turn_id", "")
        .expect("a caller key");
    session
        .set_frame_metadata("x-codex-turn-metadata", r#"{"request_kind":"prewarm"}"#)
        .expect("a caller key");
    let prewarm_id = session
        .warmup(prewarm_request())
        .await
        .expect("the prewarm completes");
    assert_eq!(prewarm_id, "resp_prewarm");
    // The prewarm's turn-state token belongs to the first turn, as it does in
    // the official client.
    let turn_state = session
        .take_received_turn_state()
        .expect("the prewarm handed over a turn-state token");
    assert_eq!(turn_state, "ts-prewarm");
    session
        .set_frame_metadata(TURN_STATE_METADATA_KEY, turn_state)
        .expect("a caller key");

    session
        .set_frame_metadata("turn_id", "turn-1")
        .expect("a caller key");
    session
        .set_frame_metadata("x-codex-turn-metadata", r#"{"request_kind":"turn"}"#)
        .expect("a caller key");
    session
        .send_incremental(delta("FIRST_TURN_MARKER"))
        .await
        .expect("the first turn continues the prewarm");
    finish_turn(&mut session).await;

    let frames = script.sent_json();
    assert_eq!(frames.len(), 2);
    let (prewarm, turn) = (&frames[0], &frames[1]);

    assert_eq!(prewarm["generate"], false);
    assert!(prewarm.get("previous_response_id").is_none());
    assert_eq!(prewarm["input"], json!([]), "the prewarm carries no input");
    assert_eq!(prewarm["client_metadata"]["turn_id"], "");
    assert_eq!(
        prewarm["client_metadata"]["x-codex-turn-metadata"],
        r#"{"request_kind":"prewarm"}"#
    );

    assert!(turn.get("generate").is_none(), "the first turn generates");
    assert_eq!(turn["previous_response_id"], "resp_prewarm");
    let input = serde_json::to_string(&turn["input"]).expect("input serializes");
    assert!(input.contains("FIRST_TURN_MARKER"), "got {input}");
    assert_eq!(turn["client_metadata"]["turn_id"], "turn-1");
    assert_eq!(
        turn["client_metadata"][TURN_STATE_METADATA_KEY],
        "ts-prewarm"
    );
    assert!(
        prewarm["client_metadata"]
            .get(TURN_STATE_METADATA_KEY)
            .is_none(),
        "the prewarm had no token to send"
    );
    assert_eq!(
        turn["client_metadata"]["x-codex-turn-metadata"],
        r#"{"request_kind":"turn"}"#
    );

    // Everything but the input, the chaining, `generate` and the per-turn
    // metadata is the prewarm's request: the properties Codex requires to
    // match before it continues a prewarm.
    let strip = |frame: &serde_json::Value| {
        let mut frame = frame.as_object().expect("a frame is an object").clone();
        for key in [
            "input",
            "previous_response_id",
            "generate",
            "client_metadata",
        ] {
            remove_preserving_order(&mut frame, key);
        }
        frame
    };
    assert_eq!(strip(turn), strip(prewarm));
}

/// Only a frame that generates nothing may carry no input: a generating send
/// of the same request is refused before anything is written.
#[tokio::test]
async fn a_generating_send_without_input_is_still_refused() {
    let script = Script::new();
    let mut session = session_over(&script);
    let refused = session
        .send(prewarm_request())
        .await
        .expect_err("a generating request needs input");
    assert!(
        refused.to_string().contains("at least one non-system item"),
        "got {refused}"
    );
    assert!(script.sent_json().is_empty(), "nothing was written");
}

/// A `response.metadata` event handing over a turn-state token.
fn turn_state_event(token: &str) -> String {
    json!({
        "type": "response.metadata",
        "headers": { TURN_STATE_METADATA_KEY: token },
    })
    .to_string()
}

/// The session records the first turn-state token its turns are handed,
/// whichever call reads the event, and keeps it until it is taken: a later
/// token is ignored, a name in another case still matches, and an array
/// value gives its first string, as the official client reads them.
#[tokio::test]
async fn the_first_turn_state_token_is_recorded_until_taken() {
    let script = Script::new()
        .turn([
            turn_state_event("ts-1"),
            turn_state_event("ts-ignored"),
            completed("resp_1"),
        ])
        .turn([
            json!({
                "type": "response.metadata",
                "headers": { "X-Codex-Turn-State": ["ts-2", "ts-other"] },
            })
            .to_string(),
            completed("resp_2"),
        ])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);
    assert_eq!(session.received_turn_state(), None);

    session
        .completion(user_request("first"))
        .await
        .expect("the first turn completes");
    assert_eq!(session.received_turn_state(), Some("ts-1"));
    assert_eq!(session.take_received_turn_state().as_deref(), Some("ts-1"));
    assert_eq!(session.received_turn_state(), None);

    session
        .completion(user_request("second"))
        .await
        .expect("the second turn completes");
    assert_eq!(session.take_received_turn_state().as_deref(), Some("ts-2"));

    session
        .completion(user_request("third"))
        .await
        .expect("the third turn completes");
    assert_eq!(
        session.received_turn_state(),
        None,
        "a turn without a token records none"
    );
}

/// A ChatGPT wire that lifts only the leading run of system messages into
/// `instructions`, as a caller that keeps mid-conversation system messages in
/// place configures it. It carries none of the dialect's default
/// instructions, so a frame's `instructions` are exactly what the placement
/// lifted.
fn leading_instructions_wire() -> CodexResponses {
    let mut provider = provider();
    provider.instructions = None;
    let mut wire = anonymous_wire(provider).with_caller_identity(test_caller_identity());
    wire.wire = wire
        .wire
        .with_system_instructions_placement(SystemInstructionsPlacement::Instructions);
    wire
}

/// Under the leading-run placement a request of only instructions keeps them
/// in `input` when it generates, so it is not empty. The prewarm generates
/// nothing, so it lifts them as Codex's prewarm sends them: `instructions`
/// with an empty input. The first turn continues it with the same
/// `instructions`, as a delta on the prewarm.
#[tokio::test]
async fn the_prewarm_lifts_its_instructions_under_the_leading_run_placement() {
    let script = Script::new()
        .turn([completed("resp_prewarm")])
        .turn([completed("resp_turn_1")]);
    let mut session = CodexWebSocketSession::from_connection(
        leading_instructions_wire(),
        identity(),
        script.connection(),
        None,
    )
    .expect("the ChatGPT dialect speaks the Codex contract");

    session
        .warmup(prewarm_request())
        .await
        .expect("the prewarm completes");
    session
        .send_incremental(delta("FIRST_TURN_MARKER"))
        .await
        .expect("the first turn continues the prewarm");
    finish_turn(&mut session).await;

    let frames = script.sent_json();
    assert_eq!(frames.len(), 2);
    let (prewarm, turn) = (&frames[0], &frames[1]);
    assert_eq!(prewarm["generate"], false);
    assert_eq!(prewarm["input"], json!([]), "got {prewarm}");
    assert_eq!(prewarm["instructions"], "Standing instructions.");
    assert_eq!(turn["previous_response_id"], "resp_prewarm");
    assert_eq!(turn["instructions"], "Standing instructions.");
    let turn_input = serde_json::to_string(&turn["input"]).expect("input serializes");
    assert!(turn_input.contains("FIRST_TURN_MARKER"), "got {turn_input}");
    assert!(
        !turn_input.contains("Standing instructions."),
        "got {turn_input}"
    );
}

/// Generating requests are unchanged by the prewarm's lift: under the
/// leading-run placement a request of only system messages still sends them
/// in `input` with no `instructions`, and a request whose system messages
/// open a conversation still lifts only that leading run.
#[tokio::test]
async fn generating_requests_keep_the_leading_run_placement() {
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")]);
    let mut session = CodexWebSocketSession::from_connection(
        leading_instructions_wire(),
        identity(),
        script.connection(),
        None,
    )
    .expect("the ChatGPT dialect speaks the Codex contract");

    session
        .completion(prewarm_request())
        .await
        .expect("a generating request of only system messages is sent");
    let conversation = request_of(vec![
        Message::system("Standing instructions."),
        Message::user("hello"),
        Message::system("MID_CONVERSATION_NOTE"),
    ]);
    session
        .completion(conversation)
        .await
        .expect("the conversation turn completes");

    let frames = script.sent_json();
    assert_eq!(frames.len(), 2);
    let (system_only, conversation) = (&frames[0], &frames[1]);

    assert!(system_only.get("generate").is_none());
    assert!(
        system_only.get("instructions").is_none(),
        "got {system_only}"
    );
    let system_only_input = serde_json::to_string(&system_only["input"]).expect("input serializes");
    assert!(
        system_only_input.contains("Standing instructions."),
        "got {system_only_input}"
    );

    assert_eq!(conversation["instructions"], "Standing instructions.");
    let conversation_input =
        serde_json::to_string(&conversation["input"]).expect("input serializes");
    assert!(
        !conversation_input.contains("Standing instructions."),
        "got {conversation_input}"
    );
    assert!(
        conversation_input.contains("MID_CONVERSATION_NOTE"),
        "got {conversation_input}"
    );
}

/// A warmup may carry no input, but not no messages: a request without even
/// instructions is refused before anything is written.
#[tokio::test]
async fn a_warmup_without_any_message_is_refused_and_writes_nothing() {
    let script = Script::new();
    let mut session = session_over(&script);
    let empty = request_of(Vec::new());
    let refused = session
        .warmup(empty)
        .await
        .expect_err("a warmup needs at least its instructions");
    assert!(
        refused.to_string().contains("empty chat history"),
        "got {refused}"
    );
    assert!(script.sent_json().is_empty(), "nothing was written");
}

// ── the session's surface in one pass, and what the public API reaches ──

/// A wire with a `version` in its caller identity and none of the dialect's
/// default instructions, so a frame's `instructions` are the request's own.
fn versioned_wire() -> CodexResponses {
    let mut provider = provider();
    // The gateway's default instructions are not the caller's.
    provider.instructions = None;
    anonymous_wire(provider).with_caller_identity(
        CallerIdentity::new(
            "codex_exec",
            "codex_exec/0.1 (test)",
            Some("0.1.0".to_owned()),
        )
        .expect("a valid caller identity"),
    )
}

/// A backend that hands out one scripted connection and keeps the handshake.
#[derive(Clone)]
struct KeepingBackend {
    script: Script,
    handshakes: Arc<Mutex<Vec<http_client::Request<NoBody>>>>,
}

impl WebSocketClientExt for KeepingBackend {
    async fn connect(
        &self,
        request: http_client::Request<NoBody>,
        _options: ConnectOptions,
    ) -> http_client::Result<BoxedWebSocketConnection> {
        self.handshakes.lock().expect("lock").push(request);
        Ok(self.script.connection())
    }
}

struct FixedClock(u64);

impl FrameClock for FixedClock {
    fn unix_millis(&self) -> u64 {
        self.0
    }
}

fn response_event(kind: &str, response_id: &str, status: &str) -> String {
    json!({
        "type": kind,
        "sequence_number": 1,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": 0,
            "status": status,
            "model": chatgpt::GPT_5_3_CODEX,
            "output": [],
        },
    })
    .to_string()
}

fn metadata_event(turn_state: &str) -> String {
    json!({
        "type": "response.metadata",
        "headers": {"X-Codex-Turn-State": turn_state},
    })
    .to_string()
}

/// The whole surface in one pass: handshake, a full send, typed events, the
/// turn-state token, an incremental send, an idle keepalive and a close.
#[tokio::test]
async fn a_session_runs_a_full_turn_a_delta_a_keepalive_and_closes() {
    let script = Script::new()
        .turn([
            response_event("response.created", "resp_1", "in_progress"),
            metadata_event("ts-1"),
            text_delta("hi"),
            response_event("response.completed", "resp_1", "completed"),
        ])
        .turn([response_event("response.completed", "resp_2", "completed")]);
    let backend = KeepingBackend {
        script: script.clone(),
        handshakes: Arc::default(),
    };
    let wire = versioned_wire()
        .with_codex_identity(identity())
        .expect("a Codex wire");
    let mut session = CodexWebSocketSessionBuilder::new(wire.clone())
        .expect("a Codex wire")
        .with_request_start_ms_stamp(Arc::new(FixedClock(1_700_000_000_000)))
        .connect_with(&backend)
        .await
        .expect("connects");

    let handshake = backend
        .handshakes
        .lock()
        .expect("lock")
        .first()
        .expect("one handshake")
        .clone();
    assert_eq!(
        handshake.uri().to_string(),
        "wss://chatgpt.com/backend-api/codex/responses"
    );
    let header = |name: &str| {
        handshake
            .headers()
            .get(name)
            .map(|value| value.to_str().expect("ascii").to_owned())
    };
    assert_eq!(
        header("authorization").as_deref(),
        Some("Bearer test-token")
    );
    assert_eq!(header("chatgpt-account-id").as_deref(), Some(ACCOUNT_ID));
    assert_eq!(header("originator").as_deref(), Some("codex_exec"));
    assert_eq!(
        header("user-agent").as_deref(),
        Some("codex_exec/0.1 (test)")
    );
    assert_eq!(header("version").as_deref(), Some("0.1.0"));
    assert_eq!(header("session-id").as_deref(), Some("session-1"));
    assert_eq!(header("thread-id").as_deref(), Some("thread-1"));
    assert_eq!(header("x-client-request-id").as_deref(), Some("thread-1"));
    assert_eq!(
        header("openai-beta").as_deref(),
        Some(RESPONSES_WEBSOCKETS_BETA_VALUE)
    );
    assert_eq!(handshake.headers().len(), 9, "{:?}", handshake.headers());

    session
        .send(CompletionRequest::new("say hi").preamble("be brief"))
        .await
        .expect("sends");
    let sent = script.sent_json();
    let frame = sent.first().expect("one frame");
    assert_eq!(frame["type"], "response.create");
    assert_eq!(frame["model"], chatgpt::GPT_5_3_CODEX);
    assert_eq!(frame["stream"], true);
    assert_eq!(frame["store"], false);
    assert_eq!(frame["instructions"], "be brief");
    assert_eq!(frame["prompt_cache_key"], "thread-1");
    assert_eq!(frame["client_metadata"]["session_id"], "session-1");
    assert_eq!(frame["client_metadata"]["thread_id"], "thread-1");
    assert_eq!(
        frame["client_metadata"][WS_STREAM_REQUEST_START_MS_METADATA_KEY],
        "1700000000000"
    );
    assert!(frame.get("previous_response_id").is_none(), "{frame}");
    assert!(frame.get("background").is_none(), "{frame}");
    assert_eq!(frame["input"].as_array().map(Vec::len), Some(1));
    // The conversation is the frame's last field.
    let text = script.sent().first().expect("one frame").clone();
    assert!(text.starts_with(r#"{"type":"response.create","#), "{text}");
    assert!(text.ends_with("}]}"), "{text}");
    let input_at = text.find(r#""input":"#).expect("input");
    for key in [
        r#""client_metadata":"#,
        r#""prompt_cache_key":"#,
        r#""model":"#,
    ] {
        assert!(text.find(key).expect("field") < input_at, "{key} in {text}");
    }

    assert!(matches!(
        session.next_event().await.expect("created"),
        ResponsesWebSocketEvent::Response(chunk)
            if matches!(chunk.kind, ResponseChunkKind::ResponseCreated)
    ));
    let ResponsesWebSocketEvent::Unknown(metadata) = session.next_event().await.expect("metadata")
    else {
        panic!("response.metadata is unmodelled");
    };
    assert_eq!(metadata.kind, "response.metadata");
    assert_eq!(session.received_turn_state(), Some("ts-1"));
    assert!(matches!(
        session.next_event().await.expect("delta"),
        ResponsesWebSocketEvent::Item(_)
    ));
    let terminal = session.next_event().await.expect("completed");
    assert!(terminal.is_terminal());
    assert_eq!(session.previous_response_id(), Some("resp_1"));
    assert_eq!(session.take_received_turn_state().as_deref(), Some("ts-1"));

    // Idle: a ping, an unmodelled frame and the trailing `response.done`.
    script.arrive([
        Frame::Ping(bytes::Bytes::from_static(b"p")),
        Frame::Text(json!({"type": "codex.rate_limits", "limit": 1}).to_string()),
        Frame::Text(
            json!({"type": "response.done", "response": {"id": "resp_1", "status": "completed"}})
                .to_string(),
        ),
    ]);
    let (recovered, ending) = session.keepalive().await.into_parts();
    assert!(ending.is_none(), "{ending:?}");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].kind, "codex.rate_limits");
    assert_eq!(script.pongs_flushed(), 1);

    // A delta-only history: a tool result whose call the request does not
    // carry, projected through Rig's own encoder.
    session
        .set_frame_metadata(TURN_STATE_METADATA_KEY, "ts-1")
        .expect("not reserved");
    let call = CallId::from(ProviderCallId::new("call_1").expect("a call id"));
    let tail = CompletionRequest::new(rig_core::completion::Message::tool_result(
        call,
        ToolName::new("lookup").expect("a tool name"),
        "42",
    ))
    .additional_params(json!({"previous_response_id": "resp_1"}));
    let items = input_items(&wire.wire, tail).expect("a delta-only history encodes");
    assert_eq!(
        items,
        vec![json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42",
            "status": "completed",
        })]
    );
    session
        .send_incremental(InputDelta::new(items.clone()).expect("non-empty"))
        .await
        .expect("sends the delta");
    let sent = script.sent_json();
    let delta = sent.get(1).expect("a second frame");
    assert_eq!(delta["previous_response_id"], "resp_1");
    assert_eq!(delta["input"], json!(items));
    assert_eq!(delta["instructions"], "be brief");
    assert_eq!(delta["client_metadata"][TURN_STATE_METADATA_KEY], "ts-1");
    assert!(session.next_event().await.expect("completed").is_terminal());
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    session.clear_previous_response_id();
    assert_eq!(session.previous_response_id(), None);
    session.close().await.expect("closes");
    assert!(script.closed());
}

/// The Codex prewarm: instructions and tools, no conversation.
#[tokio::test]
async fn an_input_free_warmup_sends_an_empty_input() {
    let script = Script::new().turn([response_event("response.completed", "resp_w", "completed")]);
    let mut session = CodexWebSocketSession::from_connection(
        versioned_wire(),
        identity(),
        script.connection(),
        None,
    )
    .expect("a Codex wire");
    let mut request = CompletionRequest::new("placeholder").preamble("be brief");
    request.chat_history.pop();
    session
        .send_with_options(request, ResponsesWebSocketCreateOptions::warmup())
        .await
        .expect("an input-free warmup sends");
    let sent = script.sent_json();
    let frame = sent.first().expect("one frame");
    assert_eq!(frame["generate"], false);
    assert_eq!(frame["input"], json!([]));
    assert_eq!(frame["instructions"], "be brief");
    assert!(session.next_event().await.expect("completed").is_terminal());
    assert_eq!(session.previous_response_id(), Some("resp_w"));
    session.close().await.expect("closes");
}

/// A frame that fails to decode reaches the caller with the frame itself.
#[tokio::test]
async fn a_corrupt_frame_reaches_the_caller_as_received() {
    let payload = r#"{"type":"response.completed","sequence_number":1,"response":{"id":7}}"#;
    let script = Script::new().turn([payload.to_owned()]);
    let mut session = CodexWebSocketSession::from_connection(
        versioned_wire(),
        identity(),
        script.connection(),
        None,
    )
    .expect("a Codex wire");
    session
        .send(CompletionRequest::new("say hi"))
        .await
        .expect("sends");
    let error = session
        .next_event()
        .await
        .expect_err("the frame is corrupt");
    let corrupt = error.corrupt_frame().expect("typed evidence");
    assert_eq!(corrupt.event_type(), Some("response.completed"));
    assert_eq!(corrupt.frame(), Some(payload));

    use crate::error::ProviderErrorExt as _;
    let provider = ProviderError::from(error);
    assert_eq!(
        provider.corrupt_frame().and_then(|corrupt| corrupt.frame()),
        Some(payload),
        "the frame survives the conversion to a provider error"
    );
}

/// Frames in hand decode through Rig's public reply decoder.
#[test]
fn a_recorded_turn_decodes_through_the_public_reply_decoder() {
    use rig_core::wire::{Mode, WireFrame, decode_reply};

    let wire = versioned_wire().wire;
    let request = CompletionRequest::new("say hi");
    let frames = [
        response_event("response.created", "resp_1", "in_progress"),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 2,
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []},
        })
        .to_string(),
        text_delta("hi"),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            },
        })
        .to_string(),
        response_event("response.completed", "resp_1", "completed"),
    ];
    let response = decode_reply(
        &wire,
        &request,
        Mode::Streaming,
        frames.into_iter().map(WireFrame::Text),
    )
    .expect("the turn decodes");
    assert_eq!(response.raw["id"], "resp_1");
}

/// A `response.done` whose present structural field is malformed is a
/// corrupt frame: it settles nothing, leaves no tip and fails the session.
#[tokio::test]
async fn malformed_done_does_not_settle_a_successful_response() {
    let script = Script::new().turn([json!({
        "type":"response.done", "response":{"id":"bad_tip","status":"completed","output":42}
    })
    .to_string()]);
    let mut session = session_over(&script);
    session
        .send(user_request("hello"))
        .await
        .expect("request sends");
    assert!(session.next_event().await.is_err());
    assert_eq!(session.previous_response_id(), None);
    assert!(matches!(session.continuation, Continuation::Blocked(_)));
    assert!(session.send(user_request("again")).await.is_err());
}

/// Everything a session writes is a function of what it was given. Two
/// sessions built from the same wire, conversation identity, frame metadata
/// and clock write the same handshake and the same frames, byte for byte:
/// nothing the session writes is drawn at random or read from a clock of its
/// own.
#[tokio::test]
async fn a_session_writes_the_same_bytes_for_the_same_inputs() {
    async fn run() -> (Vec<(String, String)>, Vec<String>) {
        let script = Script::new()
            .turn([turn_state_event("ts-prewarm"), completed("resp_1")])
            .turn([completed("resp_2")])
            .turn([completed("resp_3")]);
        let backend = KeepingBackend {
            script: script.clone(),
            handshakes: Arc::default(),
        };
        let wire = official_wire()
            .with_codex_identity(identity())
            .expect("the ChatGPT dialect speaks the Codex contract");
        let mut session = CodexWebSocketSessionBuilder::new(wire)
            .expect("a Codex wire")
            .with_frame_metadata("turn_id", "turn-1")
            .expect("a caller key")
            .with_request_start_ms_stamp(Arc::new(SteppingClock(
                std::sync::atomic::AtomicU64::new(1_790_000_000_000),
            )))
            .connect_with(&backend)
            .await
            .expect("connects");

        session
            .warmup(prewarm_request())
            .await
            .expect("the prewarm completes");
        let turn_state = session
            .take_received_turn_state()
            .expect("the prewarm handed over a turn-state token");
        session
            .set_frame_metadata(TURN_STATE_METADATA_KEY, turn_state)
            .expect("a caller key");
        session
            .send_incremental(delta("FIRST_TURN"))
            .await
            .expect("the first turn continues the prewarm");
        finish_turn(&mut session).await;
        session
            .completion(user_request("ROOT"))
            .await
            .expect("a root turn completes");

        let handshakes = backend.handshakes.lock().expect("lock").clone();
        assert_eq!(handshakes.len(), 1);
        let mut handshake: Vec<(String, String)> = handshakes[0]
            .headers()
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_owned(),
                    value.to_str().expect("ascii header").to_owned(),
                )
            })
            .collect();
        handshake.sort();
        (handshake, script.sent())
    }

    let first = run().await;
    let second = run().await;
    assert_eq!(first.1.len(), 3, "a prewarm, a delta and a root send");
    assert_eq!(first, second);
}

/// A binary frame that is not UTF-8 cannot be JSON. It fails the turn as a
/// corrupt frame carrying its bytes exactly as received, mid-turn and during
/// an idle drain alike, and the bytes survive the conversion to a provider
/// error.
#[tokio::test]
async fn a_binary_frame_that_is_not_utf8_carries_its_bytes_as_received() {
    use crate::error::{CorruptFrameDetail, FrameEvidence, ProviderErrorExt as _};

    let received = vec![0x7b, 0xff, 0xfe, 0x7d];
    let frame = || Frame::Binary(bytes::Bytes::from(vec![0x7b, 0xff, 0xfe, 0x7d]));

    let script = Script::new();
    let mut session = session_over(&script);
    session
        .send(user_request("hello"))
        .await
        .expect("request sends");
    script.arrive([frame()]);
    let error = session
        .next_event()
        .await
        .expect_err("bytes that are not UTF-8 are not a frame");
    let corrupt = error.corrupt_frame().expect("typed evidence");
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(received.clone()));
    assert_eq!(corrupt.frame_bytes(), Some(received.as_slice()));
    assert_eq!(corrupt.frame(), None, "there is no exact text");
    assert_eq!(
        CorruptFrameDetail::from(corrupt).evidence,
        FrameEvidence::Bytes(received.clone())
    );
    assert!(
        session.send(user_request("again")).await.is_err(),
        "the session is failed"
    );
    let provider = ProviderError::from(error);
    assert_eq!(
        provider
            .corrupt_frame()
            .and_then(|corrupt| corrupt.frame_bytes()),
        Some(received.as_slice())
    );

    // The same frame while idle ends the drain with the same evidence.
    let script = Script::new();
    let mut idle = session_over(&script);
    script.arrive_text([json!({ "type": "codex.rate_limits" }).to_string()]);
    script.arrive([frame()]);
    let (recovered, ending) = idle.keepalive().await.into_parts();
    assert_eq!(recovered.len(), 1, "what preceded the frame is handed back");
    let ending = ending.expect("the drain failed");
    assert_eq!(
        ending.corrupt_frame().map(|corrupt| corrupt.evidence()),
        Some(&FrameEvidence::Bytes(received))
    );
}
