use super::test_connection::Script;
use super::*;
use crate::http_client::{HeaderMap, StatusCode};
use crate::providers::openai::OpenAI;
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ResponseError, ResponseObject, ResponsesUsage,
};
use crate::ws_client::CloseFrame;
use futures::FutureExt as _;
use serde_json::json;

/// The wire a session is opened over.
fn test_wire(base_url: &str) -> Responses {
    OpenAI::new("test-key")
        .with_base_url(base_url)
        .responses("gpt-5.4")
}

/// The shape a rejected upgrade reaches this module in: the backend has
/// already read the status, headers and body off the refusing HTTP
/// response.
fn rejection(
    status: u16,
    request_id: Option<&str>,
    body: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> http_client::Error {
    let mut headers = HeaderMap::new();
    if let Some(request_id) = request_id {
        headers.insert(
            "x-request-id",
            request_id.parse().expect("header value should be valid"),
        );
    }
    for (name, value) in extra_headers {
        headers.insert(
            http::HeaderName::from_bytes(name.as_bytes()).expect("header name should be valid"),
            value.parse().expect("header value should be valid"),
        );
    }
    http_client::Error::non_success_with_details(
        StatusCode::from_u16(status).expect("status should be valid"),
        headers,
        body.unwrap_or_default().to_string(),
    )
}

/// The live shape, recorded in
/// `websocket_error_identity_matrix/handshake_rejection_carries_status_body_and_request_id`.
const REJECTION_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***-key.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#;

#[test]
fn websocket_provider_error_preserves_status_body_and_request_id() {
    let error = websocket_provider_error(rejection(
        401,
        Some("req_websocket_1"),
        Some(REJECTION_BODY),
        &[],
    ));

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::UNAUTHORIZED)
    );
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
    assert_eq!(error.provider_request_id(), Some("req_websocket_1"));
    assert_eq!(
        error
            .provider_response_json()
            .expect("body should be valid JSON")
            .expect("parsed JSON should be present")["error"]["code"],
        "invalid_api_key"
    );
}

/// The id is optional everywhere else in this crate and is optional here:
/// its absence must not cost the status or the body.
#[test]
fn websocket_provider_error_without_a_request_id_keeps_the_rest() {
    let error = websocket_provider_error(rejection(401, None, Some(REJECTION_BODY), &[]));

    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::UNAUTHORIZED)
    );
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
    assert_eq!(error.provider_request_id(), None);
}

#[test]
fn websocket_provider_error_treats_an_empty_request_id_as_absent() {
    let error = websocket_provider_error(rejection(401, Some(""), Some(REJECTION_BODY), &[]));

    assert_eq!(error.provider_request_id(), None);
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
}

#[test]
fn websocket_provider_error_without_a_body_keeps_the_status() {
    let error = websocket_provider_error(rejection(503, Some("req_websocket_2"), None, &[]));

    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_websocket_2"));
    // An empty preserved body is `Some("")`, not `None`: the provider
    // answered, it just said nothing.
    assert_eq!(error.provider_response_body(), Some(""));
}

/// Every status a refused upgrade can carry, **including 2xx and 3xx**:
/// tungstenite raises `Error::Http` for any non-101 response, and
/// `connect_async` does not follow redirects, so a `200` or a `302` reaches
/// this mapping exactly as a `401` does. Classification here follows the
/// call path, not the status class, so those two must survive too.
#[test]
fn websocket_provider_error_preserves_every_rejection_status() {
    for status in [200u16, 302, 400, 401, 403, 404, 429, 500, 503] {
        let error = websocket_provider_error(rejection(status, None, Some("{}"), &[]));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::from_u16(status).expect("status should be valid")),
            "status {status} should survive"
        );
    }
}

/// A `429` upgrade carries the same rate-limit metadata its HTTP twin
/// does, and a caller that has to back off needs it (rig#2210).
#[test]
fn websocket_provider_error_preserves_the_rejections_headers() {
    let error = websocket_provider_error(rejection(
        429,
        Some("req_websocket_3"),
        Some("{}"),
        &[("retry-after", "20"), ("x-ratelimit-remaining", "0")],
    ));

    let headers = error
        .provider_response_headers()
        .expect("headers should be preserved");
    assert_eq!(
        headers.get("retry-after").and_then(|v| v.to_str().ok()),
        Some("20")
    );
    assert_eq!(
        headers
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0")
    );
    // The id is read before the map is consumed.
    assert_eq!(error.provider_request_id(), Some("req_websocket_3"));
}

/// A failure that never reached the provider has no response to preserve
/// and stays a transport error, with the transport table's retryability.
#[test]
fn websocket_provider_error_leaves_a_transport_failure_alone() {
    let error = websocket_provider_error(http_client::Error::StreamEnded);

    assert!(matches!(error, ProviderError::Http(_)));
    assert!(error.is_retryable());
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_request_id(), None);
}

/// The regression this mapping exists for: a rejection must not flatten to
/// its display string (rig#2314, rig#2315).
#[test]
fn websocket_provider_error_no_longer_flattens_a_rejection_to_a_string() {
    let error = websocket_provider_error(rejection(
        401,
        Some("req_websocket_4"),
        Some(REJECTION_BODY),
        &[],
    ));

    assert!(
        error.provider_response_body().is_some(),
        "the provider's own body must survive, not just a display string"
    );
}

#[test]
fn websocket_error_event_preserves_provider_payload_as_json() {
    let mut extra = Map::new();
    extra.insert(
        "type".to_string(),
        Value::String("invalid_request_error".to_string()),
    );
    let event = ResponsesWebSocketErrorEvent {
        kind: ResponsesWebSocketErrorEventKind::Error,
        error: ResponsesWebSocketErrorPayload {
            code: Some("rate_limit_exceeded".to_string()),
            message: Some("slow down".to_string()),
            extra,
        },
        status: None,
        headers: None,
        extra: Map::new(),
    };

    let err = provider_error_from_event(&event);

    // No HTTP status on the websocket stream, and the raw payload round-trips
    // through provider_response_json() (code + message + extra all preserved).
    assert_eq!(err.provider_response_status(), None);
    let json = err
        .provider_response_json()
        .expect("preserved body should be valid JSON")
        .expect("provider response body should be present");
    assert_eq!(json["error"]["code"], "rate_limit_exceeded");
    assert_eq!(json["error"]["message"], "slow down");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

/// The Codex backend's wrapped failure: a top-level `status`, the `error`
/// object, and a `headers` map carrying `x-codex-*` rate-limit values. The
/// event decodes whole, and the error it raises carries that status and those
/// headers, so it classifies as the same failure over HTTP would.
#[test]
fn a_codex_wrapped_error_event_keeps_its_status_and_headers() {
    let payload = json!({
        "type": "error",
        "status": 429,
        "error": {
            "type": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "resets_at": 1738888888
        },
        "headers": {
            "x-codex-primary-used-percent": "100.0",
            "x-codex-primary-window-minutes": 15
        }
    })
    .to_string();
    let Some(ResponsesWebSocketEvent::Error(event)) =
        parse_server_event(&payload).expect("the event parses")
    else {
        panic!("a wrapped error decodes as an error event");
    };
    assert_eq!(event.status, Some(429));

    let err = provider_error_from_event(&event);
    assert_eq!(
        err.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    let headers = err
        .provider_response_headers()
        .expect("the event's headers are attached");
    assert_eq!(headers["x-codex-primary-used-percent"], "100.0");
    assert_eq!(headers["x-codex-primary-window-minutes"], "15");
    let body = err
        .provider_response_json()
        .expect("preserved body should be valid JSON")
        .expect("provider response body should be present");
    assert_eq!(body["status"], 429);
    assert_eq!(body["error"]["type"], "usage_limit_reached");
    assert_eq!(body["error"]["resets_at"], 1738888888);
    assert_eq!(body["headers"]["x-codex-primary-window-minutes"], 15);
}

/// A header the typed map cannot carry, a name with a space or a value with
/// a control character, is left out of the error's headers but kept in the
/// preserved body, which carries the whole event.
#[test]
fn an_error_event_header_no_header_map_can_carry_stays_in_the_body_only() {
    let payload = json!({
        "type": "error",
        "status": 429,
        "headers": {
            "not a header name": "kept-in-body",
            "x-control-value": "line\nbreak",
            "x-codex-primary-used-percent": "100.0"
        }
    })
    .to_string();
    let Some(ResponsesWebSocketEvent::Error(event)) =
        parse_server_event(&payload).expect("the event parses")
    else {
        panic!("a wrapped error decodes as an error event");
    };
    let err = provider_error_from_event(&event);
    let headers = err
        .provider_response_headers()
        .expect("the event's headers are attached");
    assert_eq!(headers.len(), 1);
    assert_eq!(headers["x-codex-primary-used-percent"], "100.0");
    let body = err
        .provider_response_json()
        .expect("preserved body should be valid JSON")
        .expect("provider response body should be present");
    assert_eq!(body["headers"]["not a header name"], "kept-in-body");
    assert_eq!(body["headers"]["x-control-value"], "line\nbreak");
}

/// A reported status that no HTTP status can be is a malformed known field:
/// the event is refused with a typed decode error rather than read as an
/// error without a status.
#[test]
fn an_error_event_with_an_impossible_status_is_refused() {
    for (field, status) in [("status", 0), ("status", 1000), ("status_code", 42)] {
        let payload = json!({"type": "error", field: status}).to_string();
        let error = parse_server_event(&payload).expect_err("not an HTTP status");
        assert!(
            error.to_string().contains("is not an HTTP status"),
            "{field}={status}: {error}"
        );
    }
    let payload = json!({"type": "error", "status": null}).to_string();
    let Some(ResponsesWebSocketEvent::Error(event)) =
        parse_server_event(&payload).expect("a null status is no status")
    else {
        panic!("an error event with a null status decodes");
    };
    assert_eq!(event.status, None);
}

/// The `status_code` spelling is read as the status, an event without an
/// `error` object still decodes, and unmodelled top-level fields are kept
/// rather than dropped.
#[test]
fn an_error_event_without_an_error_object_decodes_and_keeps_its_fields() {
    let payload = json!({
        "type": "error",
        "status_code": 503,
        "request_id": "req_1"
    })
    .to_string();
    let Some(ResponsesWebSocketEvent::Error(event)) =
        parse_server_event(&payload).expect("the event parses")
    else {
        panic!("an error event without an error object still decodes");
    };
    assert_eq!(event.status, Some(503));
    assert!(event.error.is_empty());
    assert_eq!(event.extra["request_id"], "req_1");
    let body = provider_error_from_event(&event)
        .provider_response_json()
        .expect("preserved body should be valid JSON")
        .expect("provider response body should be present");
    assert_eq!(
        body,
        json!({"type": "error", "status": 503, "request_id": "req_1"})
    );
}

/// An explicit `"error": null` decodes as an empty payload, exactly as a
/// missing `error` does, so a wrapped error event is never lost to a decode
/// failure.
#[test]
fn an_error_event_with_a_null_error_decodes_as_an_empty_payload() {
    let payload = json!({ "type": "error", "status": 429, "error": null }).to_string();
    let Some(ResponsesWebSocketEvent::Error(event)) =
        parse_server_event(&payload).expect("the event parses")
    else {
        panic!("an error event with a null error still decodes");
    };
    assert_eq!(event.status, Some(429));
    assert!(event.error.is_empty());
    assert!(
        event.extra.is_empty(),
        "the null error is not kept as an extra"
    );
}

/// The flattened `extra` cannot swallow the tag: an event whose `type` is not
/// `error` does not decode as an error event.
#[test]
fn an_event_with_another_type_does_not_decode_as_an_error_event() {
    let decoded = serde_json::from_value::<ResponsesWebSocketErrorEvent>(json!({
        "type": "response.completed",
        "status": 429,
    }));
    assert!(decoded.is_err(), "the tag is still checked: {decoded:?}");
}

fn sample_response(status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: "resp_123".to_string(),
        object: ResponseObject::Response,
        provider_request_id: None,
        created_at: 0,
        status,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: "gpt-5.4".to_string(),
        usage: Some(ResponsesUsage {
            input_tokens: 1,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: Some(
                crate::providers::openai::responses_api::OutputTokensDetails {
                    reasoning_tokens: 0,
                },
            ),
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

#[test]
fn warmup_options_serialize_generate_false() {
    let options = ResponsesWebSocketCreateOptions::warmup();
    let json = serde_json::to_value(options).expect("options should serialize");

    assert_eq!(json, json!({ "generate": false }));
}

/// The handshake request carries the endpoint path and the wire's own auth
/// headers, on the websocket scheme.
#[test]
fn websocket_request_targets_the_responses_endpoint_with_the_wires_headers() {
    let request =
        websocket_request(&test_wire("https://api.openai.com/v1")).expect("request should build");

    assert_eq!(request.uri(), "wss://api.openai.com/v1/responses");
    assert_eq!(
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-key")
    );
}

#[test]
fn websocket_request_rejects_an_unsupported_base_url_scheme() {
    let error = websocket_request(&test_wire("ftp://api.openai.com/v1"))
        .expect_err("ftp is not a websocket base");
    assert!(
        error.to_string().contains("ftp"),
        "the error should name the scheme, got {error}"
    );
    // Building the handshake request is request building, not transport.
    let error = ProviderError::from(error);
    assert!(matches!(error, ProviderError::Request(_)), "{error:?}");
    assert!(!error.is_retryable());
}

#[test]
fn parse_done_event_exposes_response_id() {
    let payload = json!({
        "type": "response.done",
        "response": {
            "id": "resp_done_1",
            "status": "completed"
        }
    });

    let event = parse_server_event(&payload.to_string())
        .expect("done event should deserialize")
        .expect("done event should not be skipped");

    assert!(matches!(
        event,
        ResponsesWebSocketEvent::Done(ResponsesWebSocketDoneEvent { .. })
    ));
    assert_eq!(event.response_id(), Some("resp_done_1"));
    assert!(event.is_terminal());
}

#[test]
fn parse_response_completed_event_is_terminal() {
    let payload = json!({
        "type": "response.completed",
        "sequence_number": 12,
        "response": {
            "id": "resp_completed_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "usage": null,
            "output": [],
            "tools": []
        }
    });

    let event = parse_server_event(&payload.to_string())
        .expect("response event should deserialize")
        .expect("response event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Response(_)));
    assert!(event.is_terminal());
    assert_eq!(event.response_id(), Some("resp_completed_1"));
}

#[test]
fn parse_live_output_item_added_event() {
    let payload = json!({
        "type": "response.output_item.added",
        "item": {
            "id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
            "type": "message",
            "status": "in_progress",
            "content": [],
            "role": "assistant"
        },
        "output_index": 0,
        "sequence_number": 2
    });

    let event = parse_server_event(&payload.to_string())
        .expect("output item event should parse")
        .expect("output item event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_live_content_part_added_event() {
    let payload = json!({
        "type": "response.content_part.added",
        "content_index": 0,
        "item_id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
        "output_index": 0,
        "part": {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": ""
        },
        "sequence_number": 3
    });

    let event = parse_server_event(&payload.to_string())
        .expect("content part event should parse")
        .expect("content part event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_live_output_text_delta_event() {
    let payload = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "Web",
        "item_id": "msg_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
        "logprobs": [],
        "obfuscation": "2YYErYq7jkqqM",
        "output_index": 0,
        "sequence_number": 4
    });

    let event = parse_server_event(&payload.to_string())
        .expect("output text delta event should parse")
        .expect("output text delta event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_reasoning_text_delta_event_is_item() {
    let payload = json!({
        "type": "response.reasoning_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "thinking",
    });

    let event = parse_server_event(&payload.to_string())
        .expect("reasoning delta should parse")
        .expect("reasoning delta should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    assert!(!event.is_terminal());
}

#[test]
fn unknown_event_type_is_forwarded_raw() {
    let payload = json!({
        "type": "response.some_future_event",
        "data": "hello"
    });

    let result = parse_server_event(&payload.to_string()).expect("unknown event should not error");
    // Semantically skipped, but carried verbatim with its tag so the streaming
    // surface can yield it on the `StreamEvent::Unknown` passthrough and a
    // consumer failing closed can name it.
    match result {
        Some(ResponsesWebSocketEvent::Unknown(event)) => {
            assert_eq!(event.kind, "response.some_future_event");
            assert_eq!(event.payload, payload.into());
        }
        other => panic!("expected the raw Unknown passthrough event, got {other:?}"),
    }
    assert!(
        !parse_server_event(&json!({ "type": "codex.rate_limits" }).to_string())
            .expect("an unknown namespace should not error")
            .expect("an unknown event is not skipped")
            .is_terminal(),
        "an unmodelled event ends nothing"
    );
}

#[test]
fn malformed_known_event_returns_error() {
    let payload = json!({
        "type": "response.completed"
    });

    let error =
        parse_server_event(&payload.to_string()).expect_err("malformed known event should error");
    assert!(
        error.to_string().contains("StreamingCompletionChunk"),
        "expected strict decode failure, got {error}"
    );
}

#[test]
fn terminal_response_requires_completed_status() {
    let completed = terminal_response_result(sample_response(ResponseStatus::Completed))
        .expect("completed response should succeed");
    assert_eq!(completed.id, "resp_123");

    let failed = terminal_response_result(sample_response(ResponseStatus::Failed))
        .expect_err("failed response should error");
    assert!(failed.to_string().contains("failed response"));
}

#[test]
fn terminal_failed_response_with_error_preserves_raw_payload() {
    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: Some("server_error".to_string()),
        message: "the model failed to generate a response".to_string(),
    });

    let Err(err) = terminal_response_result(response) else {
        panic!("failed response with an error object should fail")
    };

    // The full failed-response envelope is preserved as a ProviderResponse with
    // no HTTP status (the websocket stream carries none), so the raw JSON parses
    // back with the provider error nested under `error` — proving the whole
    // envelope is kept, not just the error object.
    assert_eq!(err.provider_response_status(), None);

    let json = err
        .provider_response_json()
        .expect("preserved body should parse as JSON")
        .expect("preserved body should not be empty");
    assert_eq!(
        json["error"]["message"],
        "the model failed to generate a response"
    );
    assert_eq!(json["error"]["code"], "server_error");
}

#[test]
fn terminal_failed_response_without_error_is_rig_diagnostic() {
    let Err(err) = terminal_response_result(sample_response(ResponseStatus::Failed)) else {
        panic!("failed response should fail")
    };

    // No provider error object, so this is a Rig-authored diagnostic and exposes
    // no preserved provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert!(err.to_string().contains("failed response"));
}

/// An incomplete terminal is a success, not a failure: the partial output
/// and usage are kept and normalization maps the status downstream.
#[test]
fn terminal_incomplete_response_is_a_terminal_success() {
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });

    let response = terminal_response_result(response).expect("incomplete is a terminal");
    assert!(matches!(response.status, ResponseStatus::Incomplete));
}

/// A close frame mid-turn is an error naming the peer's reason; a keepalive
/// is skipped without ending the turn.
#[test]
fn websocket_frame_to_text_maps_control_frames() {
    assert_eq!(
        websocket_frame_to_text(Frame::Text("{}".to_string())).expect("text frame"),
        Some("{}".to_string())
    );
    assert_eq!(
        websocket_frame_to_text(Frame::Ping(bytes::Bytes::new())).expect("ping is skipped"),
        None
    );

    let error = websocket_frame_to_text(Frame::Close(Some(CloseFrame {
        code: 1011,
        reason: "server restarting".to_string(),
    })))
    .expect_err("a close frame ends the turn");
    assert!(
        error.to_string().contains("server restarting"),
        "the peer's reason should surface, got {error}"
    );

    let error = websocket_frame_to_text(Frame::Close(None))
        .expect_err("a reasonless close still ends the turn");
    assert!(error.to_string().contains("without a close reason"));
}

// ---------------------------------------------------------------------------
// Idle keepalive: servicing, custody of recovered frames, and its bounds.
// ---------------------------------------------------------------------------

fn user_request(text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest {
        model: None,
        chat_history: vec![completion::Message::user(text)],
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

fn completed_event(response_id: &str) -> String {
    json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": serde_json::to_value(CompletionResponse {
            id: response_id.to_string(),
            ..sample_response(ResponseStatus::Completed)
        })
        .expect("response should serialize"),
    })
    .to_string()
}

fn done_event(response_id: &str) -> String {
    json!({
        "type": "response.done",
        "response": { "id": response_id, "status": "completed" },
    })
    .to_string()
}

fn unknown_event(kind: &str, seen: &str) -> String {
    json!({ "type": kind, "seen": seen }).to_string()
}

fn stray_delta() -> String {
    json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "stray",
        "item_id": "msg_stray",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    })
    .to_string()
}

fn session_over(script: &Script) -> ResponsesWebSocketSession {
    ResponsesWebSocketSession::from_connection(
        test_wire("https://api.openai.com/v1"),
        script.connection(),
        None,
    )
}

/// The frames of a drain that must have serviced the socket cleanly.
fn serviced(drain: KeepaliveDrain, context: &str) -> Vec<UnrecognizedEvent> {
    let (recovered, ending) = drain.into_parts();
    if let Some(error) = ending {
        panic!("{context}, but the drain failed: {error}");
    }
    recovered
}

/// The frames a failing drain still recovered, and the failure itself. Both
/// halves, always: that is the property under test.
fn failed(drain: KeepaliveDrain, context: &str) -> (Vec<UnrecognizedEvent>, ProviderError) {
    let (recovered, ending) = drain.into_parts();
    let Some(error) = ending else {
        panic!("{context}, but the drain reported success");
    };
    (recovered, error)
}

fn kinds(events: &[UnrecognizedEvent]) -> Vec<&str> {
    events.iter().map(|event| event.kind.as_str()).collect()
}

#[tokio::test]
async fn keepalive_returns_every_unknown_event_in_arrival_order_and_consumes_the_trailing_done() {
    let script = Script::new()
        .turn([completed_event("resp_1")])
        .turn([completed_event("resp_2")]);
    let mut session = session_over(&script);

    session
        .completion(user_request("first"))
        .await
        .expect("first turn should complete");

    // The trailing `done` sits between unmodelled frames: the drain must
    // neither reorder them nor let its `done` handling drop the one after it.
    script.arrive_text([
        unknown_event("codex.rate_limits", "first"),
        unknown_event("responsesapi.websocket_timing", "second"),
        done_event("resp_1"),
        unknown_event("response.some_future_event", "last"),
    ]);

    let drained = serviced(
        session.keepalive().await,
        "unknown idle events must not fail the session",
    );
    assert_eq!(
        kinds(&drained),
        vec![
            "codex.rate_limits",
            "responsesapi.websocket_timing",
            "response.some_future_event",
        ]
    );
    assert_eq!(
        drained[2].payload.value()["seen"],
        "last",
        "the complete parsed value is returned, not just the kind"
    );
    assert_eq!(script.unread(), 0, "the trailing done was consumed");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    // The session is still usable and still chains the untouched tip.
    session
        .completion(user_request("second"))
        .await
        .expect("second turn should complete after the drain");
    assert!(
        script.sent()[1].contains("\"previous_response_id\":\"resp_1\""),
        "expected the tip to survive the drain, got {}",
        script.sent()[1]
    );
}

#[tokio::test]
async fn keepalive_answers_an_idle_ping_with_a_flushed_pong() {
    let script = Script::new();
    let mut session = session_over(&script);

    script.arrive([Frame::Ping(bytes::Bytes::from_static(b"are you there"))]);
    let drained = serviced(
        session.keepalive().await,
        "keepalive should service the server ping",
    );

    assert!(drained.is_empty());
    assert_eq!(script.unread(), 0, "the ping was read");
    assert_eq!(script.pongs_flushed(), 1, "the owed pong reached the wire");
    // Servicing a ping must not invent or advance a live tip.
    assert_eq!(session.previous_response_id(), None);
}

#[tokio::test]
async fn keepalive_recovers_frames_consumed_before_the_stream_ended() {
    let script = Script::new();
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("response.some_future_event", "before close")]);
    script.end_stream();

    let (recovered, error) = failed(
        session.keepalive().await,
        "a socket that ends mid-drain must report the failure",
    );
    assert_eq!(kinds(&recovered), vec!["response.some_future_event"]);
    assert!(
        error.to_string().contains("closed during idle keepalive"),
        "got {error}"
    );

    let closed = session
        .send(user_request("after close"))
        .await
        .expect_err("a closed session refuses a send");
    assert!(
        closed.to_string().contains("session is closed"),
        "got {closed}"
    );
}

#[tokio::test]
async fn keepalive_stops_at_the_frame_budget_and_keeps_what_it_consumed() {
    let script = Script::new();
    let mut session = session_over(&script);

    // One more than the budget, so the cap stops the drain rather than the
    // supply running out.
    script
        .arrive_text((0..=MAX_KEEPALIVE_DRAIN_FRAMES).map(|_| json!({ "type": "z" }).to_string()));

    let (recovered, error) = failed(
        session.keepalive().await,
        "exceeding the frame budget must be reported",
    );
    assert_eq!(recovered.len(), MAX_KEEPALIVE_DRAIN_FRAMES);
    assert!(error.to_string().contains("buffered frames"), "got {error}");
    assert_eq!(script.unread(), 1, "nothing past the budget was consumed");

    // A socket that cannot be read to a known state is not serviceable: the
    // session failed, and a later drain is the ordinary no-op.
    let after = serviced(
        session.keepalive().await,
        "a failed session drains as a no-op",
    );
    assert!(after.is_empty());
    assert_eq!(script.unread(), 1);
}

#[tokio::test]
async fn keepalive_fails_loud_on_an_idle_modelled_event_and_returns_what_preceded_it() {
    let script = Script::new();
    let mut session = session_over(&script);

    script.arrive_text([
        unknown_event("codex.rate_limits", "before the stray delta"),
        stray_delta(),
    ]);

    let (recovered, error) = failed(
        session.keepalive().await,
        "an idle data frame should fail keepalive",
    );
    assert_eq!(kinds(&recovered), vec!["codex.rate_limits"]);
    assert!(
        error.to_string().contains("unexpected server event"),
        "got {error}"
    );

    let closed = session
        .send(user_request("after failure"))
        .await
        .expect_err("the session is unusable after a failed keepalive");
    assert!(
        closed.to_string().contains("session is closed"),
        "got {closed}"
    );
}

#[tokio::test]
async fn keepalive_is_a_noop_while_a_turn_is_in_flight() {
    let script = Script::new().turn([completed_event("resp_in_flight")]);
    let mut session = session_over(&script);

    session
        .send(user_request("hello"))
        .await
        .expect("request should send");
    let drained = serviced(
        session.keepalive().await,
        "keepalive should be a no-op while in flight",
    );
    assert!(drained.is_empty());
    assert_eq!(
        script.unread(),
        1,
        "reading here would steal the in-flight turn's own event"
    );
    assert_eq!(script.flushes(), 0);

    let event = session
        .next_event()
        .await
        .expect("the in-flight turn is still readable");
    assert_eq!(event.response_id(), Some("resp_in_flight"));
}

#[tokio::test]
async fn keepalive_is_a_noop_after_close() {
    let script = Script::new();
    let mut session = session_over(&script);

    session.close().await.expect("close should succeed");
    script.arrive_text([unknown_event("codex.rate_limits", "after close")]);
    let drained = serviced(
        session.keepalive().await,
        "keepalive should be a no-op after close",
    );
    assert!(drained.is_empty());
    assert_eq!(script.unread(), 1);
    assert!(script.closed());
}

/// A backend without the ready-read capability is refused by name, and since
/// nothing was read, the session stays usable.
#[tokio::test]
async fn keepalive_names_a_backend_without_ready_reads_and_leaves_the_session_open() {
    let script = Script::new()
        .without_recv_ready()
        .turn([completed_event("resp_1")]);
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("codex.rate_limits", "unread")]);
    let (recovered, error) = failed(
        session.keepalive().await,
        "a backend that cannot read only arrived frames cannot be drained",
    );
    assert!(recovered.is_empty());
    match &error {
        ProviderError::Request(inner) => assert_eq!(
            inner.downcast_ref::<UnsupportedCapability>(),
            Some(&UnsupportedCapability {
                capability: "recv_ready"
            })
        ),
        other => panic!("expected a named request refusal, got {other:?}"),
    }
    assert_eq!(script.unread(), 1, "nothing was consumed");

    session
        .send(user_request("still usable"))
        .await
        .expect("the session is still open");
}

#[tokio::test]
async fn keepalive_names_a_backend_without_flush() {
    let script = Script::new().without_flush();
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("codex.rate_limits", "recovered")]);
    let (recovered, error) = failed(
        session.keepalive().await,
        "a backend that cannot flush cannot promise the pongs went out",
    );
    assert_eq!(kinds(&recovered), vec!["codex.rate_limits"]);
    assert!(error.to_string().contains("`flush`"), "got {error}");
}

#[tokio::test]
async fn a_failed_flush_returns_the_recovered_frames_and_fails_the_session() {
    let script = Script::new();
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("codex.rate_limits", "recovered")]);
    script.set_flush_fails(true);
    let (recovered, _error) = failed(session.keepalive().await, "the flush failed");
    assert_eq!(kinds(&recovered), vec!["codex.rate_limits"]);

    session
        .send(user_request("after a failed flush"))
        .await
        .expect_err("an unflushable socket is not serviceable");
}

/// The one suspension is bounded inside the drain: a flush that never
/// finishes returns the recovered frames beside a typed failure after
/// [`KEEPALIVE_FLUSH_TIMEOUT`], without any bound from the caller.
#[tokio::test]
async fn a_stalled_flush_is_bounded_internally_and_keeps_the_recovered_frames() {
    let script = Script::new();
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("codex.rate_limits", "recovered")]);
    script.set_flush_stalls(true);
    let (recovered, error) = failed(session.keepalive().await, "the flush stalled");
    assert_eq!(kinds(&recovered), vec!["codex.rate_limits"]);
    assert!(
        error.to_string().contains("Timed out flushing"),
        "got {error}"
    );
}

/// Dropping a drain midway cannot destroy what it consumed: the frames are in
/// the session's custody, a send is refused until they are collected, and the
/// next drain returns them first, in order.
#[tokio::test]
async fn a_dropped_keepalive_leaves_its_frames_in_the_session() {
    let script = Script::new().turn([completed_event("resp_1")]);
    let mut session = session_over(&script);

    script.arrive_text([unknown_event("codex.rate_limits", "first")]);
    script.set_flush_stalls(true);
    // The reads complete on the first poll; the flush does not, so this
    // drops the drain at its one suspension point.
    assert!(session.keepalive().now_or_never().is_none());

    let refused = session
        .send(user_request("too early"))
        .await
        .expect_err("a send must wait for the recovered frames to be collected");
    match &refused {
        ProviderError::Request(inner) => assert_eq!(
            inner.downcast_ref::<UncollectedRecoveredFrames>(),
            Some(&UncollectedRecoveredFrames { count: 1 })
        ),
        other => panic!("expected a named request refusal, got {other:?}"),
    }
    assert!(script.sent().is_empty(), "the refused send wrote nothing");

    script.set_flush_stalls(false);
    script.arrive_text([unknown_event("response.some_future_event", "second")]);
    let drained = serviced(session.keepalive().await, "the next drain succeeds");
    assert_eq!(
        kinds(&drained),
        vec!["codex.rate_limits", "response.some_future_event"]
    );

    session
        .completion(user_request("now"))
        .await
        .expect("the session sends once the frames are collected");
}

/// An unknown frame mid-turn reaches the caller before the terminal event.
#[tokio::test]
async fn an_unknown_event_mid_turn_is_returned_with_its_tag() {
    let script = Script::new().turn([
        unknown_event("codex.rate_limits", "mid-turn"),
        completed_event("resp_1"),
    ]);
    let mut session = session_over(&script);

    session
        .send(user_request("hello"))
        .await
        .expect("request should send");
    match session.next_event().await.expect("first event") {
        ResponsesWebSocketEvent::Unknown(event) => assert_eq!(event.kind, "codex.rate_limits"),
        other => panic!("expected the unknown event first, got {other:?}"),
    }
    assert!(
        session
            .next_event()
            .await
            .expect("terminal event")
            .is_terminal()
    );
}

// ---------------------------------------------------------------------------
// Declared tools on the `response.create` frame.
// ---------------------------------------------------------------------------

/// A tool declared through `additional_params["tools"]` reaches the
/// `response.create` frame as the same JSON values the caller wrote, on a
/// strict wire: the empty namespace description survives, the namespace and
/// its member gain no `strict`, and the declared function keeps its empty
/// description while taking strict mode's own transformation. A Responses-lite
/// client declares exactly such a namespace, and a provider that requires
/// `description` refuses the frame if it is dropped.
#[tokio::test]
async fn a_declared_tool_reaches_the_response_create_frame_as_written() {
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
    let function = json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    });
    let script = Script::new().turn([completed_event("resp_1")]);
    let mut session = ResponsesWebSocketSession::from_connection(
        test_wire("https://api.openai.com/v1").with_strict_tools(),
        script.connection(),
        None,
    );

    let mut request = user_request("hello");
    request.additional_params = Some(json!({ "tools": [namespace.clone(), function.clone()] }));
    session
        .completion(request)
        .await
        .expect("the turn should complete");

    let frames = script.sent_json();
    assert_eq!(frames.len(), 1, "one frame for the turn: {frames:?}");
    let frame = &frames[0];
    assert_eq!(frame["type"], json!("response.create"));
    let tools = frame["tools"].as_array().expect("the frame carries tools");
    assert_eq!(tools.len(), 2);
    assert_eq!(
        tools[0], namespace,
        "the namespace reaches the frame verbatim"
    );
    let mut untouched = tools[1].as_object().expect("a tool is an object").clone();
    assert_eq!(untouched.remove("strict"), Some(json!(true)));
    untouched.remove("parameters");
    let mut expected = function.as_object().expect("fixture is an object").clone();
    expected.remove("strict");
    expected.remove("parameters");
    assert_eq!(
        untouched, expected,
        "every other declared member is as written"
    );
}

// --- credential source on the ordinary Responses websocket -----------------

/// A websocket backend recording each handshake it is asked to open.
#[derive(Clone, Default)]
struct HandshakeRecorder {
    handshakes: std::sync::Arc<std::sync::Mutex<Vec<HeaderMap>>>,
}

impl crate::ws_client::WebSocketClientExt for HandshakeRecorder {
    fn connect(
        &self,
        request: crate::http_client::Request<crate::http_client::NoBody>,
        _options: crate::ws_client::ConnectOptions,
    ) -> impl std::future::Future<
        Output = crate::http_client::Result<crate::ws_client::BoxedWebSocketConnection>,
    > + crate::wasm_compat::WasmCompatSend {
        self.handshakes
            .lock()
            .expect("unpoisoned")
            .push(request.headers().clone());
        let connection = Script::new().connection();
        async move { Ok(connection) }
    }
}

/// A source counting its reads, supplying `source-token-<n>` and no account,
/// or failing when `fails`.
#[derive(Clone, Default)]
struct CountingSource {
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fails: bool,
}

impl crate::wire::CredentialSource for CountingSource {
    fn current(
        &self,
    ) -> crate::wasm_compat::WasmBoxedFuture<
        '_,
        Result<crate::wire::Credential, crate::wire::CredentialSourceError>,
    > {
        let read = self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let fails = self.fails;
        Box::pin(async move {
            if fails {
                Err("the token owner is offline".into())
            } else {
                Ok(crate::wire::Credential::new(format!("source-token-{read}")))
            }
        })
    }
}

fn wire_with_source(source: CountingSource) -> Responses {
    OpenAI::new("static-key")
        .with_account_id("static-account")
        .with_credential_source(source)
        .responses("gpt-5.4")
}

/// The ordinary Responses websocket reads the source once per connect and
/// opens with what it supplied, in place of the static key and account.
#[tokio::test]
async fn an_ordinary_websocket_connect_reads_the_credential_source_once() {
    let source = CountingSource::default();
    let backend = HandshakeRecorder::default();
    for _ in 0..2 {
        ResponsesWebSocketSessionBuilder::new(wire_with_source(source.clone()))
            .connect_with(&backend)
            .await
            .expect("the scripted connection opens");
    }

    assert_eq!(source.reads.load(std::sync::atomic::Ordering::SeqCst), 2);
    let handshakes = backend.handshakes.lock().expect("unpoisoned").clone();
    for (index, handshake) in handshakes.iter().enumerate() {
        assert_eq!(
            handshake["authorization"],
            format!("Bearer source-token-{}", index + 1).as_str()
        );
        assert!(
            handshake.get("chatgpt-account-id").is_none(),
            "the source names no account, so the static one is not sent"
        );
    }
}

/// A source that fails refuses the connect by name, and the backend is never
/// asked to open anything.
#[tokio::test]
async fn a_failing_credential_source_refuses_an_ordinary_websocket_connect() {
    let backend = HandshakeRecorder::default();
    let result = ResponsesWebSocketSessionBuilder::new(wire_with_source(CountingSource {
        fails: true,
        ..CountingSource::default()
    }))
    .connect_with(&backend)
    .await;
    let Err(ProviderError::Request(inner)) = result else {
        panic!("expected a request refusal");
    };
    assert!(
        inner
            .downcast_ref::<crate::wire::CredentialUnavailable>()
            .is_some()
    );
    assert!(backend.handshakes.lock().expect("unpoisoned").is_empty());
}
