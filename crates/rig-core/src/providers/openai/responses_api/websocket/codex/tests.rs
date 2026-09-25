use super::*;
use crate::completion::{AssistantContent, FinishReason, Message};
use crate::providers::chatgpt;
use crate::providers::openai::OpenAI;
use crate::providers::openai::responses_api::websocket::test_connection::Script;
use crate::providers::openai::responses_api::{
    CompletionResponse, IncompleteDetailsReason, InputItem, OutputTokensDetails, ResponseObject,
    ResponseStatus, ResponsesUsage,
};
use serde_json::json;

const ACCESS_TOKEN: &str = "test-token";
const ACCOUNT_ID: &str = "acct-123";

fn codex_wire() -> Responses {
    OpenAI::with_key(&chatgpt::DIALECT, ACCESS_TOKEN)
        .with_account_id(ACCOUNT_ID)
        .responses(chatgpt::GPT_5_3_CODEX)
}

fn session_over(script: &Script) -> CodexWebSocketSession {
    CodexWebSocketSession::from_connection(
        codex_wire(),
        CodexIdentity::generate(),
        script.connection(),
        None,
    )
    .expect("the ChatGPT dialect speaks the Codex contract")
}

fn user_request(text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest {
        model: None,
        chat_history: vec![Message::user(text)],
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

fn delta(text: &str) -> InputDelta {
    let items = Vec::<InputItem>::try_from(Message::user(text))
        .expect("a user message converts into input items");
    InputDelta::new(items).expect("the delta carries an item")
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

/// The handshake is exactly the Codex set: the wire's credential, the account,
/// the dashed identity headers, the correlation header and the beta opt-in —
/// nothing else (the backend adds the websocket handshake headers itself).
#[test]
fn the_handshake_carries_exactly_the_codex_identity_headers() {
    let identity = CodexIdentity::generate();
    let request = identity
        .handshake_request(&codex_wire())
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

#[test]
fn the_handshake_omits_the_account_header_without_an_account() {
    let wire = OpenAI::with_key(&chatgpt::DIALECT, ACCESS_TOKEN).responses(chatgpt::GPT_5_3_CODEX);
    let request = CodexIdentity::generate()
        .handshake_request(&wire)
        .expect("handshake request should build");
    assert!(request.headers().get(CHATGPT_ACCOUNT_ID_HEADER).is_none());
}

#[test]
fn identities_are_opaque_distinct_and_key_the_cache_by_thread() {
    let identity = CodexIdentity::generate();
    assert_eq!(identity.session_id().len(), 21);
    assert_eq!(identity.thread_id().len(), 21);
    assert_ne!(identity.session_id(), identity.thread_id());
    assert_eq!(identity.prompt_cache_key(), identity.thread_id());
    assert_ne!(CodexIdentity::generate(), identity);
}

#[test]
fn a_non_codex_wire_is_refused_by_name() {
    let wire = OpenAI::new("key").responses("gpt-5.4");
    let Err(error) = CodexWebSocketSessionBuilder::new(wire.clone()) else {
        panic!("an OpenAI wire does not speak the Codex contract");
    };
    assert_eq!(error, NotACodexWire { dialect: "openai" });
    assert!(
        CodexWebSocketSession::from_connection(
            wire,
            CodexIdentity::generate(),
            Script::new().connection(),
            None
        )
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

    let first = session
        .completion(user_request("first"))
        .await
        .expect("first turn should complete");
    assert_eq!(first.provider, chatgpt::PROVIDER_NAME);
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
        // Websocket mode is event-driven: the SSE-only flags stay off the wire.
        assert!(frame.get("stream").is_none(), "got {frame}");
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
    assert!(
        frames[0].get("include").is_none(),
        "no include is forced onto a turn without reasoning: {}",
        frames[0]
    );
    assert_eq!(
        frames[1]["include"],
        json!(["reasoning.encrypted_content"]),
        "the encrypted-reasoning include rides alongside reasoning"
    );
}

/// A caller-set `top_p` (and likewise `temperature`) is refused by name on the
/// websocket lane before any frame is written, and the refusal leaves the
/// session free to send the next turn.
#[tokio::test]
async fn a_caller_set_sampling_control_is_refused_by_name_before_anything_is_sent() {
    use crate::providers::openai::responses_api::wire::UnsupportedCodexControl;

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
        rest.remove("input");
        rest.remove("previous_response_id");
        let mut root_rest = root.clone();
        root_rest.remove("input");
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
/// send establishes one — after which deltas continue normally.
#[tokio::test]
async fn an_incomplete_turn_is_a_truthful_terminal_that_blocks_deltas_until_a_full_send() {
    let script = Script::new()
        .turn([text_delta("partial"), incomplete("resp_1")])
        .turn([completed("resp_2")])
        .turn([completed("resp_3")]);
    let mut session = session_over(&script);

    let truncated = session
        .completion(user_request("ROOT"))
        .await
        .expect("an incomplete turn is a terminal, not a failure");
    assert_eq!(truncated.finish_reason(), Some(FinishReason::Length));
    assert!(
        matches!(truncated.choice.first(), Some(AssistantContent::Text(text)) if text.text == "partial"),
        "the streamed partial output survives, got {:?}",
        truncated.choice
    );
    assert_eq!(truncated.response_id.as_deref(), Some("resp_1"));
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
#[tokio::test]
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
    let mut session = CodexWebSocketSession::from_connection(
        codex_wire().with_strict_tools(),
        CodexIdentity::generate(),
        script.connection(),
        None,
    )
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
