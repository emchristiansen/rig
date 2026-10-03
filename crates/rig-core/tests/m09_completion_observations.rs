//! Deterministic boundary tests for the opt-in observed Responses operation.
//!
//! These are synthetic transport fixtures, not recorded provider traffic: they
//! exercise first-poll/Drop timing, exact supplied strings, and deliberately
//! malformed or rejected replies that a cassette cannot reliably reproduce.
#![cfg(all(feature = "completion-observations", not(target_family = "wasm")))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bytes::Bytes;
use futures::{StreamExt, poll};
use http::{HeaderMap, HeaderValue, Request, Response, StatusCode};
use rig_core::{
    client::CompletionClient,
    completion::{CompletionError, CompletionModel},
    http_client::{
        self, HttpClientExt, LazyBody, MultipartForm, StreamingResponse, sse::BoxedStream,
    },
    providers::{
        chatgpt::{self, ChatGPTAuth},
        live_support::{
            CallerIdentity,
            error::{ErrorDetail, ProviderError},
        },
        openai::responses_api::{
            observation::{
                ExistingDiagnosticBasis, Observation, ObservationDispatch, ObservationHandle,
                ObservationProvenance, ObservationStage, ObservedFailureKind,
                ObservedRequestContext, ObservedResponsesResult, ObservedResponsesStream,
            },
            observed_types::{AssistantContent, FinishReason},
            streaming::observed::ObservedEvent,
        },
    },
    streaming::RawStreamingChoice,
    wasm_compat::WasmCompatSend,
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
};

#[derive(Clone, Debug, Default)]
struct ScriptedHttp {
    script: Arc<Script>,
    sends: Arc<AtomicUsize>,
    body_polls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<Request<Bytes>>>>,
}

#[derive(Debug)]
struct Script {
    status: StatusCode,
    headers: HeaderMap,
    frames: Vec<String>,
    pending_connect: bool,
    pending_tail: bool,
    tail_error: bool,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            frames: Vec::new(),
            pending_connect: false,
            pending_tail: false,
            tail_error: false,
        }
    }
}

impl ScriptedHttp {
    fn new(script: Script) -> Self {
        Self {
            script: Arc::new(script),
            ..Self::default()
        }
    }

    fn frames(frames: Vec<String>) -> Self {
        Self::new(Script {
            frames,
            ..Script::default()
        })
    }

    fn sent_request(&self) -> Request<Bytes> {
        self.requests
            .lock()
            .expect("request ledger")
            .first()
            .expect("one request")
            .clone()
    }
}

impl HttpClientExt for ScriptedHttp {
    fn send<T, U>(
        &self,
        _request: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::InvalidStatusCode(
            StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart<U>(
        &self,
        _request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::InvalidStatusCode(
            StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let request = request.map(Into::into);
        let this = self.clone();
        async move {
            this.sends.fetch_add(1, Ordering::SeqCst);
            this.requests.lock().expect("request ledger").push(request);
            if this.script.pending_connect {
                return std::future::pending().await;
            }
            let mut chunks: VecDeque<_> = this
                .script
                .frames
                .iter()
                .map(|frame| Bytes::from(format!("data: {frame}\n\n")))
                .collect();
            let pending_tail = this.script.pending_tail;
            let mut tail_error = this.script.tail_error;
            let polls = this.body_polls.clone();
            let body: BoxedStream = Box::pin(futures::stream::poll_fn(move |_| {
                polls.fetch_add(1, Ordering::SeqCst);
                match chunks.pop_front() {
                    Some(chunk) => Poll::Ready(Some(Ok(chunk))),
                    None if tail_error => {
                        tail_error = false;
                        Poll::Ready(Some(Err(http_client::Error::StreamEnded)))
                    }
                    None if pending_tail => Poll::Pending,
                    None => Poll::Ready(None),
                }
            }));
            let mut response = Response::new(body);
            *response.status_mut() = this.script.status;
            *response.headers_mut() = this.script.headers.clone();
            Ok(response)
        }
    }
}

type Model = chatgpt::ResponsesCompletionModel<ScriptedHttp>;

fn model(http: ScriptedHttp) -> Model {
    chatgpt::Client::builder()
        .api_key(ChatGPTAuth::AccessToken {
            access_token: "synthetic-baseline-token".into(),
            account_id: Some("synthetic-baseline-account".into()),
        })
        .base_url("https://baseline.invalid/backend")
        .default_instructions("baseline defaults")
        .originator("baseline-originator")
        .user_agent("baseline-agent")
        .allow_device_flow(false)
        .http_client(http)
        .build()
        .expect("synthetic client")
        .completion_model("requested-model")
}

fn context() -> ObservedRequestContext {
    ObservedRequestContext {
        access_token: "synthetic-observed-token".into(),
        account_id: Some("synthetic-observed-account".into()),
        base_url: "https://observed.invalid/backend".into(),
        caller_identity: CallerIdentity::new(
            "observed-originator",
            "observed-agent",
            Some("fixture-v1".into()),
        )
        .expect("synthetic caller identity"),
    }
}

async fn open(http: ScriptedHttp) -> (ObservationHandle, ObservedResponsesStream) {
    let model = model(http);
    let request = model
        .completion_request("hello")
        .preamble("caller preamble".into())
        .build();
    let (handle, writer) = Observation::prepare();
    let stream = model
        .raw_stream_observed(request, context(), writer)
        .await
        .expect("observed open");
    (handle, stream)
}

async fn finish(
    mut stream: ObservedResponsesStream,
) -> (Vec<ObservedEvent>, ObservedResponsesResult) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("successful observed item"));
    }
    (
        events,
        stream.finish().expect("successful native finalization"),
    )
}

fn response(output: Value) -> Value {
    json!({
        "id": "resp_fixture", "object": "response", "created_at": 7,
        "status": "completed", "model": "reported-model", "output": output,
    })
}

fn terminal(response: Value) -> String {
    json!({"type": "response.completed", "sequence_number": 4, "response": response}).to_string()
}

fn message(text: &str) -> Value {
    json!({
        "type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    })
}

fn whole_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("x-request-id", HeaderValue::from_static("request-fixture"));
    headers.append("x-codex-limit", HeaderValue::from_static("first"));
    headers.append("x-codex-limit", HeaderValue::from_static("second"));
    headers.insert(
        "x-codex-bytes",
        HeaderValue::from_bytes(b"a\xffz").expect("opaque header bytes"),
    );
    headers.append("x-unprojected", HeaderValue::from_static("one"));
    headers.append(
        "x-unprojected",
        HeaderValue::from_bytes(b"two\xfe").expect("opaque header bytes"),
    );
    headers
}

/// An unpolled future owns the writer but cannot start the mock HTTP future.
#[tokio::test]
async fn dropping_an_unpolled_opening_future_cancels_without_dispatch() {
    let http = ScriptedHttp::default();
    let model = model(http.clone());
    let request = model.completion_request("hello").build();
    let (handle, writer) = Observation::prepare();
    let opening = model.raw_stream_observed(request, context(), writer);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Prepared);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert_eq!(view.provenance, ObservationProvenance::SuppliedSse);
        assert!(view.committed_prefix.is_empty());
    });
    drop(opening);
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.reply_head.is_none());
    });
}

/// A pending send proves the mark is made at first poll, independently of a reply.
#[tokio::test]
async fn first_stream_poll_marks_attempt_and_dropping_next_does_not_cancel() {
    let http = ScriptedHttp::new(Script {
        pending_connect: true,
        ..Script::default()
    });
    let (handle, mut stream) = open(http.clone()).await;
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Prepared);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
    });
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
    let mut next = Box::pin(stream.next());
    assert!(poll!(next.as_mut()).is_pending());
    drop(next);
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Running);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert!(view.committed_prefix.is_empty());
        assert!(view.reply_head.is_none());
    });
    drop(stream);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
    });
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
}

/// JSON parsing cannot preserve duplicate keys; the ledger must keep the source string.
#[tokio::test]
async fn original_duplicate_key_prefix_survives_the_whole_stream() {
    let original =
        "{ \"type\" : \"response.future_fixture\", \"same\":1, \"same\":2, \"text\":\"a\\nb\" }"
            .to_owned();
    let http = ScriptedHttp::new(Script {
        frames: vec![original.clone()],
        pending_tail: true,
        ..Script::default()
    });
    let (handle, mut stream) = open(http.clone()).await;
    let stream_handle = stream.observation();
    assert!(handle.same_operation(&stream_handle));
    let (other, other_writer) = Observation::prepare();
    assert!(!handle.same_operation(&other));
    drop(other_writer);
    match stream
        .next()
        .await
        .expect("unknown frame")
        .expect("valid unknown frame")
    {
        ObservedEvent::Unknown { event_type, value } => {
            assert_eq!(event_type, "response.future_fixture");
            assert_eq!(value.get("same"), Some(&json!(2)));
        }
        event => assert!(
            matches!(&event, ObservedEvent::Unknown { .. }),
            "unexpected event: {event:?}"
        ),
    }
    let mut next = Box::pin(stream.next());
    assert!(poll!(next.as_mut()).is_pending());
    drop(next);
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::Running));
    drop(stream);
    drop(stream_handle);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.committed_prefix, [original]);
        assert!(view.capture_fault.is_none());
    });
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
}

/// Terminal-only output and headers must survive without any preceding delta.
#[tokio::test]
async fn terminal_only_output_preserves_opaque_parts_usage_and_the_whole_head() {
    let opaque_part = json!({"type": "future_part", "bytes": [1, 2], "keep": null});
    let provider_item =
        json!({"type": "future_tool_result", "id": "opaque_1", "payload": {"exact": true}});
    let mut output_message = message("terminal answer");
    output_message
        .as_object_mut()
        .expect("message object")
        .insert("phase".into(), json!("final_answer"));
    output_message
        .get_mut("content")
        .expect("message content field")
        .as_array_mut()
        .expect("message content")
        .push(opaque_part.clone());
    let mut body = response(json!([output_message, provider_item.clone()]));
    body.as_object_mut().expect("response object").insert("usage".into(), json!({"input_tokens": 5, "output_tokens": 2, "total_tokens": 7, "input_tokens_details": {}}));
    let original = terminal(body.clone());
    let headers = whole_headers();
    let http = ScriptedHttp::new(Script {
        frames: vec![original.clone(), "[DONE]".into()],
        headers: headers.clone(),
        ..Script::default()
    });
    let (handle, stream) = open(http.clone()).await;
    let (events, result) = finish(stream).await;
    assert!(handle.same_operation(&result.observation));
    let response = &result.body.response;
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    assert_eq!(response.provider, "chatgpt");
    assert_eq!(response.model.as_deref(), Some("reported-model"));
    assert_eq!(response.message_id.as_deref(), Some("msg_fixture"));
    assert_eq!(response.response_id.as_deref(), Some("resp_fixture"));
    assert_eq!(
        response.provider_request_id.as_deref(),
        Some("request-fixture")
    );
    assert_eq!(
        response
            .provider_response_headers
            .get("x-codex-limit")
            .expect("projected x-codex-limit"),
        "first, second"
    );
    assert_eq!(
        response
            .provider_response_headers
            .get("x-codex-bytes")
            .expect("projected x-codex-bytes"),
        "a\u{fffd}z"
    );
    assert_eq!(response.provider_response_headers.len(), 2);
    assert_eq!(
        response.raw,
        json!({
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7, "input_tokens_details": {}},
            "status": "completed", "message_id": "msg_fixture",
            "response_id": "resp_fixture", "model": "reported-model",
        })
    );
    let native = serde_json::to_value(&result.body.native).expect("native response JSON");
    assert_eq!(
        native.pointer("/output/0/content/0/text"),
        Some(&json!("terminal answer"))
    );
    assert_eq!(
        native.pointer("/output/0/phase"),
        Some(&json!("final_answer"))
    );
    assert_eq!(native.pointer("/output/0/content/1"), Some(&opaque_part));
    assert_eq!(native.pointer("/output/1"), Some(&provider_item));
    assert_eq!(
        format!("{:?}", response.usage),
        "Usage { input_tokens: Some(5), output_tokens: Some(2), total_tokens: Some(7), cached_input_tokens: None, cache_creation_input_tokens: None, tool_use_prompt_tokens: None, reasoning_tokens: None }"
    );
    assert_eq!(response.choice.len(), 3);
    match response.choice.first().expect("first output") {
        AssistantContent::Text(text) => {
            assert_eq!(text.text, "terminal answer");
            assert_eq!(
                text.additional_params
                    .as_ref()
                    .expect("phase metadata")
                    .get("openai_responses")
                    .and_then(|extras| extras.get("phase")),
                Some(&json!("final_answer"))
            );
        }
        other => assert!(
            matches!(other, AssistantContent::Text(_)),
            "unexpected first output: {other:?}"
        ),
    }
    match response.choice.get(1).expect("opaque part") {
        AssistantContent::Text(text) => {
            assert!(text.text.is_empty());
            assert_eq!(
                text.additional_params
                    .as_ref()
                    .expect("opaque metadata")
                    .get("openai_responses_part")
                    .and_then(|part| part.get("value")),
                Some(&opaque_part)
            );
        }
        other => assert!(
            matches!(other, AssistantContent::Text(_)),
            "unexpected opaque part: {other:?}"
        ),
    }
    match response.choice.get(2).expect("provider item") {
        AssistantContent::ProviderItem(item) => {
            assert_eq!(item.item, provider_item);
            assert_eq!(item.provider.as_deref(), Some("chatgpt"));
        }
        other => assert!(
            matches!(other, AssistantContent::ProviderItem(_)),
            "unexpected provider item: {other:?}"
        ),
    }
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                ObservedEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>(),
        "terminal answer"
    );
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeSucceeded);
        assert_eq!(view.committed_prefix, [original, "[DONE]".into()]);
        let head = view.reply_head.expect("actual HTTP head");
        assert_eq!(head.status, StatusCode::OK);
        assert_eq!(head.headers, headers);
        assert!(head.accepted_sse);
    });
    // A caller's later text-only admission can reject output without owning
    // any capability that changes Rig's already successful native stage.
    let caller_admission: Result<(), &str> = if response
        .choice
        .iter()
        .any(|part| matches!(part, AssistantContent::ProviderItem(_)))
    {
        Err("caller accepts text only")
    } else {
        Ok(())
    };
    assert_eq!(caller_admission, Err("caller accepts text only"));
    drop(result);
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::NativeSucceeded));
    let sent = http.sent_request();
    let request_body: Value = serde_json::from_slice(sent.body()).expect("request JSON");
    assert_eq!(
        request_body.get("instructions"),
        Some(&json!("caller preamble"))
    );
    assert_eq!(
        sent.uri().to_string(),
        "https://observed.invalid/backend/responses"
    );
    assert_eq!(
        sent.headers()
            .get("authorization")
            .expect("sent authorization header"),
        "Bearer synthetic-observed-token"
    );
    assert_eq!(
        sent.headers()
            .get("chatgpt-account-id")
            .expect("sent chatgpt-account-id header"),
        "synthetic-observed-account"
    );
    assert_eq!(
        sent.headers()
            .get("originator")
            .expect("sent originator header"),
        "observed-originator"
    );
    assert_eq!(
        sent.headers()
            .get("user-agent")
            .expect("sent user-agent header"),
        "observed-agent"
    );
    assert_eq!(
        sent.headers().get("version").expect("sent version header"),
        "fixture-v1"
    );
}

/// Missing usage and a reported cached zero are distinct provider facts.
#[tokio::test]
async fn missing_usage_output_and_optional_cached_counts_decode_without_defaults() {
    for (usage, expected_input, expected_cached) in [
        (None, None, None),
        (
            Some(json!({"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})),
            Some(3),
            None,
        ),
        (
            Some(
                json!({"input_tokens": 3, "output_tokens": 1, "total_tokens": 4, "input_tokens_details": {}}),
            ),
            Some(3),
            None,
        ),
        (
            Some(
                json!({"input_tokens": 3, "output_tokens": 1, "total_tokens": 4, "input_tokens_details": {"cached_tokens": 0}}),
            ),
            Some(3),
            Some(0),
        ),
    ] {
        let mut body = response(json!([]));
        body.as_object_mut()
            .expect("response object")
            .remove("output");
        if let Some(usage) = usage {
            body.as_object_mut()
                .expect("response object")
                .insert("usage".into(), usage);
        }
        let (handle, stream) = open(ScriptedHttp::frames(vec![terminal(body)])).await;
        let (_, result) = finish(stream).await;
        assert!(result.body.response.choice.is_empty());
        assert_eq!(result.body.response.usage.input_tokens, expected_input);
        assert_eq!(
            result.body.response.usage.cached_input_tokens,
            expected_cached
        );
        handle.inspect(|view| assert_eq!(view.stage, ObservationStage::NativeSucceeded));
    }
}

/// Codex's omitted envelope indices are repaired only after original capture.
#[tokio::test]
async fn missing_codex_indices_repair_without_replacing_the_original_payload() {
    let delta =
        r#"{ "type":"response.output_text.delta", "item_id":"msg_fixture", "delta":"hello" }"#
            .to_owned();
    let done = terminal(response(json!([message("hello")])));
    let (handle, stream) = open(ScriptedHttp::frames(vec![delta.clone(), done.clone()])).await;
    let (events, result) = finish(stream).await;
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                ObservedEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>(),
        "hello"
    );
    assert!(
        matches!(result.body.response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello")
    );
    handle.inspect(|view| assert_eq!(view.committed_prefix, [delta.clone(), done.clone()]));

    // The ordinary live adapter retains its baseline strict envelope decoder.
    let baseline_model = model(ScriptedHttp::frames(vec![delta, done]));
    let request = baseline_model.completion_request("hello").build();
    let mut baseline = baseline_model
        .raw_stream(request)
        .await
        .expect("baseline open");
    let failure = baseline
        .next()
        .await
        .expect("strict envelope rejection")
        .expect_err("baseline must not gain observed envelope repair");
    assert!(matches!(failure, CompletionError::JsonError(_)));
}

/// Finalization without a genuine terminal differs from a completed malformed tool item.
#[tokio::test]
async fn missing_terminal_is_provider_failure_but_completed_malformed_tool_is_stream_failure() {
    let (handle, mut stream) = open(ScriptedHttp::default()).await;
    assert!(stream.next().await.is_none());
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization));
    let failure = stream.finish().expect_err("no genuine terminal");
    assert_eq!(failure.kind, ObservedFailureKind::Provider);
    assert!(handle.same_operation(&failure.observation));
    match &failure.diagnostic_basis {
        ExistingDiagnosticBasis::Provider(error) => assert_eq!(
            format!("{error:?}"),
            "Response(\"provider stream ended without a terminal record; treating the turn as truncated\")"
        ),
        other => assert!(
            matches!(other, ExistingDiagnosticBasis::Provider(_)),
            "wrong missing-terminal basis: {other:?}"
        ),
    }
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::NativeFailed));

    let tool = json!({
        "type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
        "name": "broken", "arguments": "{", "status": "completed",
    });
    let item_done = json!({
        "type": "response.output_item.done", "output_index": 0,
        "sequence_number": 3, "item": tool.clone(),
    })
    .to_string();
    let terminal = terminal(response(json!([tool])));
    let originals = vec![item_done, terminal];
    let (handle, mut stream) = open(ScriptedHttp::frames(originals.clone())).await;
    let failure = loop {
        if let Err(failure) = stream
            .next()
            .await
            .expect("malformed completed tool must fail as a stream item")
        {
            break failure;
        }
    };
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    assert!(handle.same_operation(&failure.observation));
    match &failure.diagnostic_basis {
        ExistingDiagnosticBasis::Stream(report) => match report.detail.as_ref() {
            Some(ErrorDetail::MalformedToolInput(detail)) => {
                assert_eq!(detail.name, "broken");
                assert_eq!(detail.raw, "{");
                assert_eq!(detail.id.explicit(), Some("call_fixture"));
                assert_eq!(
                    detail
                        .provider
                        .as_ref()
                        .expect("provider identity")
                        .item_id
                        .as_deref(),
                    Some("fc_fixture")
                );
            }
            other => assert!(
                matches!(other, Some(ErrorDetail::MalformedToolInput(_))),
                "wrong malformed tool detail: {other:?}"
            ),
        },
        other => assert!(
            matches!(other, ExistingDiagnosticBasis::Stream(_)),
            "wrong malformed tool basis: {other:?}"
        ),
    }
    assert_eq!(
        stream.finish().expect_err("retained stream failure").kind,
        ObservedFailureKind::Stream
    );
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, originals);
        assert!(matches!(
            view.native_cause,
            Some(ExistingDiagnosticBasis::Stream(_))
        ));
    });
}

/// Completed malformed tools take precedence over a subsequent transport failure.
#[tokio::test]
async fn pending_malformed_tool_precedes_transport_failure() {
    let item_done = json!({
        "type": "response.output_item.done", "output_index": 0,
        "sequence_number": 3,
        "item": {
            "type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
            "name": "broken", "arguments": "{", "status": "completed",
        },
    })
    .to_string();
    let transport_report = ProviderError::from(http_client::Error::StreamEnded).report();
    for malformed in [false, true] {
        let originals = if malformed {
            vec![item_done.clone()]
        } else {
            Vec::new()
        };
        let http = ScriptedHttp::new(Script {
            frames: originals.clone(),
            tail_error: true,
            ..Script::default()
        });
        let (handle, mut stream) = open(http.clone()).await;
        let failure = stream
            .next()
            .await
            .expect("first item is the native failure")
            .expect_err("completed malformed tool or transport failure");
        assert_eq!(failure.kind, ObservedFailureKind::Stream);
        assert!(handle.same_operation(&failure.observation));
        let report = match &failure.diagnostic_basis {
            ExistingDiagnosticBasis::Stream(report) => Some(report),
            _ => None,
        }
        .expect("stream failure retains the native report");
        if malformed {
            assert_ne!(report.as_ref(), &transport_report);
            assert!(matches!(
                report.detail.as_ref(),
                Some(ErrorDetail::MalformedToolInput(detail))
                    if detail.name == "broken"
                        && detail.raw == "{"
                        && detail.id.explicit() == Some("call_fixture")
                        && detail.provider.as_ref().and_then(|provider| provider.item_id.as_deref())
                            == Some("fc_fixture")
            ));
        } else {
            assert_eq!(report.as_ref(), &transport_report);
        }
        assert_eq!(http.body_polls.load(Ordering::SeqCst), originals.len() + 1);
        assert!(stream.next().await.is_none());
        let retained = stream.finish().expect_err("retained first failure");
        assert_eq!(retained.kind, ObservedFailureKind::Stream);
        assert!(handle.same_operation(&retained.observation));
        assert!(matches!(
            &retained.diagnostic_basis,
            ExistingDiagnosticBasis::Stream(retained_report) if retained_report == report
        ));
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeFailed);
            assert_eq!(view.committed_prefix, originals);
            assert!(matches!(
                view.native_cause,
                Some(ExistingDiagnosticBasis::Stream(retained_report)) if retained_report == report
            ));
        });
        assert_eq!(http.body_polls.load(Ordering::SeqCst), originals.len() + 1);
    }
}

/// A whole response ends the stream without polling any following source data.
#[tokio::test]
async fn whole_response_finishes_without_reading_the_tail() {
    for (pending_tail, tail_error, unread_frame) in [
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (true, true, true),
    ] {
        let body = response(json!([message("whole answer")]));
        let original = body.to_string();
        let mut frames = vec![original.clone()];
        if unread_frame {
            frames.push(r#"{ "type": "future.unread", "value": "untouched" }"#.into());
        }
        let http = ScriptedHttp::new(Script {
            frames,
            pending_tail,
            tail_error,
            ..Script::default()
        });
        let (handle, mut stream) = open(http.clone()).await;
        let mut next = Box::pin(stream.next());
        assert!(matches!(
            poll!(next.as_mut()),
            Poll::Ready(Some(Ok(ObservedEvent::TextDelta { text }))) if text == "whole answer"
        ));
        drop(next);
        let mut next = Box::pin(stream.next());
        assert!(matches!(poll!(next.as_mut()), Poll::Ready(None)));
        drop(next);
        assert_eq!(http.body_polls.load(Ordering::SeqCst), 1);
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization);
            assert_eq!(view.committed_prefix, [original.clone()]);
        });
        let result = stream.finish().expect("whole response native finalization");
        assert!(handle.same_operation(&result.observation));
        assert_eq!(result.body.native.id, "resp_fixture");
        assert_eq!(
            result.body.response.raw,
            json!({"status": "completed", "message_id": "msg_fixture",
                "response_id": "resp_fixture", "model": "reported-model"})
        );
        let native = serde_json::to_value(&result.body.native).expect("native whole response JSON");
        assert_eq!(
            native.pointer("/output/0/content/0/text"),
            Some(&json!("whole answer"))
        );
        assert!(matches!(
            result.body.response.choice.as_slice(),
            [AssistantContent::Text(text)] if text.text == "whole answer"
        ));
        assert_eq!(
            result.body.response.finish_reason(),
            Some(&FinishReason::Stop)
        );
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeSucceeded);
            assert_eq!(view.committed_prefix, [original]);
            assert!(view.native_cause.is_none());
        });
        assert_eq!(http.body_polls.load(Ordering::SeqCst), 1);
    }
}

/// Whole-response completion still flushes an earlier completed malformed tool.
#[tokio::test]
async fn pending_malformed_tool_is_not_lost_before_a_whole_response() {
    let item_done = json!({
        "type": "response.output_item.done", "output_index": 0,
        "sequence_number": 3,
        "item": {
            "type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
            "name": "broken", "arguments": "{", "status": "completed",
        },
    })
    .to_string();
    let originals = vec![item_done, response(json!([])).to_string()];
    let http = ScriptedHttp::new(Script {
        frames: originals.clone(),
        pending_tail: true,
        tail_error: true,
        ..Script::default()
    });
    let (handle, mut stream) = open(http.clone()).await;
    let mut next = Box::pin(stream.next());
    let polled = poll!(next.as_mut());
    drop(next);
    assert!(matches!(&polled, Poll::Ready(Some(Err(_)))));
    let failure = match polled {
        Poll::Ready(Some(Err(failure))) => Some(failure),
        _ => None,
    }
    .expect("whole response must immediately flush the malformed tool");
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    assert!(handle.same_operation(&failure.observation));
    assert!(matches!(
        &failure.diagnostic_basis,
        ExistingDiagnosticBasis::Stream(report)
            if matches!(report.detail.as_ref(), Some(ErrorDetail::MalformedToolInput(detail))
                if detail.name == "broken" && detail.raw == "{"
                    && detail.id.explicit() == Some("call_fixture")
                    && detail.provider.as_ref().and_then(|provider| provider.item_id.as_deref())
                        == Some("fc_fixture"))
    ));
    assert_eq!(http.body_polls.load(Ordering::SeqCst), 2);
    let mut next = Box::pin(stream.next());
    assert!(matches!(poll!(next.as_mut()), Poll::Ready(None)));
    drop(next);
    let retained = stream
        .finish()
        .expect_err("retained malformed tool failure");
    assert_eq!(retained.kind, ObservedFailureKind::Stream);
    assert!(handle.same_operation(&retained.observation));
    assert!(matches!(
        (&retained.diagnostic_basis, &failure.diagnostic_basis),
        (ExistingDiagnosticBasis::Stream(retained_report), ExistingDiagnosticBasis::Stream(report))
            if retained_report == report
    ));
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, originals);
        assert!(matches!(
            (view.native_cause, &failure.diagnostic_basis),
            (Some(ExistingDiagnosticBasis::Stream(retained_report)), ExistingDiagnosticBasis::Stream(report))
                if retained_report == report
        ));
    });
    assert_eq!(http.body_polls.load(Ordering::SeqCst), 2);
}

/// Whole-response text remains visible before a later pending-tool failure.
#[tokio::test]
async fn whole_response_text_precedes_a_pending_malformed_tool_failure() {
    let item_done = json!({
        "type": "response.output_item.done", "output_index": 1,
        "sequence_number": 3,
        "item": {
            "type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
            "name": "broken", "arguments": "{", "status": "completed",
        },
    })
    .to_string();
    let originals = vec![
        item_done,
        response(json!([message("whole answer")])).to_string(),
    ];
    let http = ScriptedHttp::new(Script {
        frames: originals.clone(),
        pending_tail: true,
        tail_error: true,
        ..Script::default()
    });
    let (handle, mut stream) = open(http.clone()).await;
    assert!(handle.same_operation(&stream.observation()));
    let mut next = Box::pin(stream.next());
    assert!(matches!(
        poll!(next.as_mut()),
        Poll::Ready(Some(Ok(ObservedEvent::TextDelta { text }))) if text == "whole answer"
    ));
    drop(next);
    assert_eq!(http.body_polls.load(Ordering::SeqCst), 2);
    handle.inspect(|view| assert_eq!(view.committed_prefix, originals));

    let mut next = Box::pin(stream.next());
    let polled = poll!(next.as_mut());
    drop(next);
    assert!(matches!(&polled, Poll::Ready(Some(Err(_)))));
    let failure = match polled {
        Poll::Ready(Some(Err(failure))) => Some(failure),
        _ => None,
    }
    .expect("malformed tool failure follows the whole-response text immediately");
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    assert!(handle.same_operation(&failure.observation));
    assert!(matches!(
        &failure.diagnostic_basis,
        ExistingDiagnosticBasis::Stream(report)
            if matches!(report.detail.as_ref(), Some(ErrorDetail::MalformedToolInput(detail))
                if detail.name == "broken" && detail.raw == "{"
                    && detail.id.explicit() == Some("call_fixture")
                    && detail.provider.as_ref().and_then(|provider| provider.item_id.as_deref())
                        == Some("fc_fixture"))
    ));
    assert_eq!(http.body_polls.load(Ordering::SeqCst), 2);
    let mut next = Box::pin(stream.next());
    assert!(matches!(poll!(next.as_mut()), Poll::Ready(None)));
    drop(next);
    let retained = stream
        .finish()
        .expect_err("retained malformed tool failure after delivered text");
    assert_eq!(retained.kind, ObservedFailureKind::Stream);
    assert!(handle.same_operation(&retained.observation));
    assert!(matches!(
        (&retained.diagnostic_basis, &failure.diagnostic_basis),
        (ExistingDiagnosticBasis::Stream(retained_report), ExistingDiagnosticBasis::Stream(report))
            if retained_report == report
    ));
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, originals);
        assert!(matches!(
            (view.native_cause, &failure.diagnostic_basis),
            (Some(ExistingDiagnosticBasis::Stream(retained_report)), ExistingDiagnosticBasis::Stream(report))
                if retained_report == report
        ));
    });
    assert_eq!(http.body_polls.load(Ordering::SeqCst), 2);
}

/// Selected terminal snapshots hydrate text and opaque items, not known tools.
/// A malformed argument string stays raw data without an item-done declaration.
#[tokio::test]
async fn terminal_only_known_tools_remain_raw_without_normalized_tool_calls() {
    let body = response(json!([
        {
            "type": "function_call", "id": "fc_terminal", "call_id": "call_terminal",
            "name": "raw_function", "arguments": "{", "status": "completed",
        },
        {
            "type": "custom_tool_call", "id": "ctc_terminal", "call_id": "custom_terminal",
            "name": "raw_custom", "input": "verbatim custom input", "status": "completed",
        },
    ]));
    let original = terminal(body.clone());
    let (handle, stream) = open(ScriptedHttp::frames(vec![original.clone()])).await;
    let (events, result) = finish(stream).await;
    assert!(events.is_empty());
    assert!(result.body.response.choice.is_empty());
    assert_eq!(
        result.body.response.finish_reason(),
        Some(&FinishReason::Stop)
    );
    assert_eq!(
        result.body.response.raw,
        json!({
            "status": "completed", "response_id": "resp_fixture", "model": "reported-model",
        })
    );
    let native = serde_json::to_value(&result.body.native).expect("native tool response JSON");
    assert_eq!(
        native.pointer("/output/0/type"),
        Some(&json!("function_call"))
    );
    assert_eq!(native.pointer("/output/0/arguments"), Some(&json!("{")));
    assert_eq!(
        native.pointer("/output/1/type"),
        Some(&json!("custom_tool_call"))
    );
    assert_eq!(
        native.pointer("/output/1/input"),
        Some(&json!("verbatim custom input"))
    );
    assert!(handle.same_operation(&result.observation));
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeSucceeded);
        assert_eq!(view.committed_prefix, [original]);
        assert!(view.native_cause.is_none());
    });
}

/// Rig's response Debug is the basis carried by four caller rejection variants.
/// These fixtures do not instantiate the outer consumer error or test persistence.
/// The metadata and field order below come from selected 149ffec91da3f74b6b52c7fd117d58e8a8b5551a:
/// completion/request.rs::CompletionResponse and responses_api/streaming.rs's
/// StreamingCompletionResponse/terminal_record. Choice Debug is carried from the
/// typed normalized output; its fixture content is checked separately, so this
/// is not an independent golden for every nested content type's Debug spelling.
#[tokio::test]
async fn caller_rejection_diagnostic_bases_match_selected_terminal_metadata_debug() {
    // Actual consumer limit at muninn 418e8162f6ab,
    // observatory/companion/src/internal/backend.rs:38 (owner-verified).
    const MAX_ANSWER_BYTES: usize = 64 * 1024;
    for case in ["NonStop", "UnexpectedOutput", "NoText", "AnswerTooLarge"] {
        let has_usage = matches!(case, "NonStop" | "AnswerTooLarge");
        let text = if case == "AnswerTooLarge" {
            "x".repeat(MAX_ANSWER_BYTES + 1)
        } else {
            "diagnostic answer".to_owned()
        };
        let opaque = json!({"type": "future_tool_result", "id": "opaque_diagnostic",
            "payload": {"keep": null, "exact": [1, 2]}});
        let output = match case {
            "NoText" => json!([]),
            "UnexpectedOutput" => json!([message(&text), opaque.clone()]),
            _ => json!([message(&text)]),
        };
        let mut body = response(output);
        let fields = body.as_object_mut().expect("diagnostic response object");
        fields.insert(
            "reasoning".into(),
            json!({
                "context": "diagnostic-context", "future": {"keep": null},
            }),
        );
        fields.insert(
            "future_response".into(),
            json!({"original": [1, null, {"exact": true}]}),
        );
        if has_usage {
            fields.insert(
                "usage".into(),
                json!({
                    "input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }),
            );
        }
        if case == "NonStop" {
            fields.insert("status".into(), json!("cancelled"));
            fields.insert(
                "incomplete_details".into(),
                json!({"reason": "caller_cancelled"}),
            );
        }
        let original = format!(
            " {{ \"type\":\"response.completed\", \"sequence_number\":4, \"response\":{body} }} "
        );
        let headers = HeaderMap::from_iter([
            (
                http::header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            ),
            (
                http::header::HeaderName::from_static("x-request-id"),
                HeaderValue::from_static("diagnostic-request"),
            ),
            (
                http::header::HeaderName::from_static("x-codex-fixture"),
                HeaderValue::from_static("captured"),
            ),
        ]);
        let (handle, stream) = open(ScriptedHttp::new(Script {
            frames: vec![original.clone()],
            headers,
            ..Script::default()
        }))
        .await;
        let (_, result) = finish(stream).await;
        let actual = &result.body.response;
        assert!(handle.same_operation(&result.observation), "{case}");
        let delivered_text = actual
            .choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>();
        match case {
            "NoText" => {
                assert!(actual.choice.is_empty());
                assert!(delivered_text.is_empty());
            }
            "AnswerTooLarge" => {
                assert_eq!(delivered_text, text);
                assert_eq!(delivered_text.len(), 65_537);
                assert!(delivered_text.len() > MAX_ANSWER_BYTES);
                assert_eq!(actual.choice.len(), 1);
            }
            "UnexpectedOutput" => {
                assert_eq!(delivered_text, text);
                assert_eq!(actual.choice.len(), 2);
                assert!(
                    matches!(actual.choice.get(1), Some(AssistantContent::ProviderItem(item))
                    if item.item == opaque && item.provider.as_deref() == Some("chatgpt"))
                );
            }
            _ => {
                assert_eq!(case, "NonStop");
                assert_eq!(delivered_text, text);
                assert_eq!(actual.choice.len(), 1);
            }
        }

        // Independently spell the selected narrow terminal record. In selected S,
        // driver.rs stamps the HTTP ID after terminal_record serializes raw, so
        // provider_request_id is present on the normalized response only.
        let mut expected_raw = json!({
            "reasoning_metadata": {"context": "diagnostic-context", "future": {"keep": null}},
            "reasoning_context": "diagnostic-context", "status": "completed",
            "response_id": "resp_fixture", "model": "reported-model",
        });
        let expected_fields = expected_raw
            .as_object_mut()
            .expect("selected terminal metadata");
        let expected_message_id = (case != "NoText").then_some("msg_fixture");
        if let Some(message_id) = expected_message_id {
            expected_fields.insert("message_id".into(), json!(message_id));
        }
        if has_usage {
            expected_fields.insert(
                "usage".into(),
                json!({
                    "input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }),
            );
        }
        let expected_finish_reason = if case == "NonStop" {
            expected_fields.insert("status".into(), json!("cancelled"));
            expected_fields.insert(
                "incomplete_details".into(),
                json!({"reason": "caller_cancelled"}),
            );
            Some(FinishReason::Other("cancelled".into()))
        } else {
            Some(FinishReason::Stop)
        };
        let expected_usage_debug = if has_usage {
            "Usage { input_tokens: Some(11), output_tokens: Some(7), total_tokens: Some(18), cached_input_tokens: Some(0), cache_creation_input_tokens: None, tool_use_prompt_tokens: None, reasoning_tokens: Some(2) }"
        } else {
            "Usage { input_tokens: None, output_tokens: None, total_tokens: None, cached_input_tokens: None, cache_creation_input_tokens: None, tool_use_prompt_tokens: None, reasoning_tokens: None }"
        };
        assert_eq!(actual.raw, expected_raw, "{case} selected raw metadata");
        assert!(actual.raw.get("output").is_none(), "{case}");
        assert!(actual.raw.get("future_response").is_none(), "{case}");
        assert!(actual.raw.get("provider_request_id").is_none(), "{case}");
        assert_eq!(
            actual.finish_reason(),
            expected_finish_reason.as_ref(),
            "{case}"
        );
        let expected_debug = format!(
            concat!(
                "CompletionResponse {{ choice: {:?}, usage: {}, message_id: {:?}, ",
                "response_id: Some(\"resp_fixture\"), provider_request_id: Some(\"diagnostic-request\"), ",
                "provider_response_headers: {{\"x-codex-fixture\": \"captured\"}}, ",
                "finish_reason: {:?}, provider: \"chatgpt\", model: Some(\"reported-model\"), raw: {:?} }}"
            ),
            actual.choice,
            expected_usage_debug,
            expected_message_id,
            expected_finish_reason,
            expected_raw,
        );
        assert_eq!(
            format!("{actual:?}"),
            expected_debug,
            "{case} complete response Debug"
        );
        let native = serde_json::to_value(&result.body.native).expect("native diagnostic response");
        if case == "NoText" {
            assert_eq!(native.get("output"), Some(&json!([])));
        } else {
            assert_eq!(
                native.pointer("/output/0/content/0/text"),
                Some(&json!(text))
            );
        }
        if case == "UnexpectedOutput" {
            assert_eq!(native.pointer("/output/1"), Some(&opaque));
        }
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeSucceeded, "{case}");
            assert_eq!(view.committed_prefix, [original.clone()], "{case}");
            assert!(view.native_cause.is_none());
        });
        drop(result);
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeSucceeded, "{case}");
            assert_eq!(view.committed_prefix, [original], "{case}");
        });
    }
}

/// Later terminal frames preserve an earlier optional incomplete reason.
#[tokio::test]
async fn repeated_terminals_keep_the_retained_incomplete_reason() {
    for whole_last in [false, true] {
        let mut first = response(json!([]));
        first["status"] = json!("incomplete");
        first["incomplete_details"] = json!({"reason": "max_output_tokens"});
        let mut last = response(json!([]));
        last["status"] = json!("incomplete");
        let originals = vec![
            terminal(first),
            if whole_last {
                last.to_string()
            } else {
                terminal(last)
            },
        ];
        let http = ScriptedHttp::new(Script {
            frames: originals.clone(),
            pending_tail: whole_last,
            tail_error: whole_last,
            ..Script::default()
        });
        let (handle, mut stream) = open(http.clone()).await;
        let mut next = Box::pin(stream.next());
        assert!(matches!(poll!(next.as_mut()), Poll::Ready(None)));
        drop(next);
        let expected_polls = if whole_last { 2 } else { 3 };
        assert_eq!(http.body_polls.load(Ordering::SeqCst), expected_polls);
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization);
            assert_eq!(view.committed_prefix, originals);
        });
        let result = stream.finish().expect("retained terminal finalization");
        assert!(handle.same_operation(&result.observation));
        assert!(result.body.native.incomplete_details.is_none());
        assert_eq!(
            result.body.response.finish_reason(),
            Some(&FinishReason::Length)
        );
        let expected_raw = json!({
            "incomplete_details": {"reason": "max_output_tokens"},
            "status": "incomplete", "response_id": "resp_fixture", "model": "reported-model",
        });
        assert_eq!(result.body.response.raw, expected_raw);
        let expected_debug = format!(
            concat!(
                "CompletionResponse {{ choice: [], usage: Usage {{ input_tokens: None, ",
                "output_tokens: None, total_tokens: None, cached_input_tokens: None, ",
                "cache_creation_input_tokens: None, tool_use_prompt_tokens: None, reasoning_tokens: None }}, ",
                "message_id: None, response_id: Some(\"resp_fixture\"), provider_request_id: None, ",
                "provider_response_headers: {{}}, finish_reason: Some(Length), ",
                "provider: \"chatgpt\", model: Some(\"reported-model\"), raw: {:?} }}"
            ),
            expected_raw
        );
        assert_eq!(format!("{:?}", result.body.response), expected_debug);
        drop(result);
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeSucceeded);
            assert_eq!(view.committed_prefix, originals);
            assert!(view.native_cause.is_none());
        });
        assert_eq!(http.body_polls.load(Ordering::SeqCst), expected_polls);
    }
}

/// The feature flag must not route ordinary callers through the observed decoder.
#[tokio::test]
async fn feature_on_unobserved_raw_stream_keeps_baseline_instructions_and_content() {
    let delta = json!({"type": "response.output_text.delta", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "sequence_number": 1, "delta": "baseline answer"}).to_string();
    let mut body = response(json!([]));
    body.as_object_mut().expect("response object").insert(
        "usage".into(),
        json!({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
    );
    let http = ScriptedHttp::frames(vec![delta, terminal(body)]);
    let model = model(http.clone());
    let (unrelated, writer) = Observation::prepare();
    let request = model
        .completion_request("hello")
        .preamble("caller preamble".into())
        .build();
    let mut stream = model.raw_stream(request).await.expect("baseline open");
    let mut text = String::new();
    let mut final_usage = None;
    while let Some(item) = stream.next().await {
        match item.expect("baseline item") {
            RawStreamingChoice::Message(delta) => text.push_str(&delta),
            RawStreamingChoice::FinalResponse(response) => {
                final_usage = Some(response.usage.total_tokens)
            }
            _ => {}
        }
    }
    assert_eq!(text, "baseline answer");
    assert_eq!(final_usage, Some(3));
    assert_eq!(http.sends.load(Ordering::SeqCst), 1);
    let sent = http.sent_request();
    let body: Value = serde_json::from_slice(sent.body()).expect("baseline request JSON");
    assert_eq!(
        body.get("instructions"),
        Some(&json!("baseline defaults\n\ncaller preamble"))
    );
    assert_eq!(body.get("model"), Some(&json!("requested-model")));
    assert_eq!(body.get("stream"), Some(&json!(true)));
    assert_eq!(body.get("store"), Some(&json!(false)));
    assert_eq!(
        sent.uri().to_string(),
        "https://baseline.invalid/backend/responses"
    );
    assert_eq!(
        sent.headers()
            .get("authorization")
            .expect("sent authorization header"),
        "Bearer synthetic-baseline-token"
    );
    assert_eq!(
        sent.headers()
            .get("originator")
            .expect("sent originator header"),
        "baseline-originator"
    );
    unrelated.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Prepared);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.committed_prefix.is_empty());
        assert!(view.reply_head.is_none());
    });
    drop(writer);
}

/// The baseline SSE check accepts 200 only and never polls rejected reply bodies.
/// Expected diagnostics start with the same baseline status-only error; they
/// intentionally make no selected-version claim about 201 or 204 acceptance.
#[tokio::test]
async fn returned_201_204_and_non_success_statuses_keep_baseline_rejection_diagnostics() {
    for status in [
        StatusCode::CREATED,
        StatusCode::NO_CONTENT,
        StatusCode::BAD_REQUEST,
        StatusCode::UNAUTHORIZED,
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::INTERNAL_SERVER_ERROR,
    ] {
        let headers = whole_headers();
        let script = || Script {
            status,
            headers: headers.clone(),
            frames: vec![r#"{"error":{"message":"rejected body must stay unread"}}"#.into()],
            ..Script::default()
        };
        let baseline_http = ScriptedHttp::new(script());
        let model = model(baseline_http.clone());
        let request = model.completion_request("hello").build();
        let mut baseline = model.raw_stream(request).await.expect("lazy baseline open");
        let actual = baseline
            .next()
            .await
            .expect("baseline rejection")
            .expect_err("baseline status rejection");
        let expected = CompletionError::from(http_client::Error::InvalidStatusCode(status));
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(
            format!("{actual:?}"),
            format!("HttpError(InvalidStatusCode({}))", status.as_u16())
        );
        assert_eq!(baseline_http.body_polls.load(Ordering::SeqCst), 0);
        drop(baseline);

        let observed_http = ScriptedHttp::new(script());
        let (handle, mut observed) = open(observed_http.clone()).await;
        let failure = observed
            .next()
            .await
            .expect("observed rejection")
            .expect_err("observed status rejection");
        assert_eq!(failure.kind, ObservedFailureKind::Stream);
        assert!(handle.same_operation(&failure.observation));
        let expected = ProviderError::from(http_client::Error::InvalidStatusCode(status)).report();
        assert!(
            matches!(
                &failure.diagnostic_basis,
                ExistingDiagnosticBasis::Stream(_)
            ),
            "wrong rejected-status basis: {:?}",
            failure.diagnostic_basis
        );
        let report = match &failure.diagnostic_basis {
            ExistingDiagnosticBasis::Stream(report) => Some(report),
            _ => None,
        }
        .expect("rejected-status stream basis");
        assert_eq!(report.as_ref(), &expected);
        assert_eq!(format!("{report:?}"), format!("{expected:?}"));
        if status == StatusCode::CREATED {
            assert_eq!(
                format!("{report:?}"),
                "ErrorReport { kind: Http, retryable: false, message: \"HttpError: Invalid status code: 201 Created\", code: None, http_status: Some(201), refusal: false, source_chain: [], request_id: None, provider_response: None, detail: None }"
            );
        }
        if status == StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(
                format!("{report:?}"),
                "ErrorReport { kind: Http, retryable: true, message: \"HttpError: Invalid status code: 429 Too Many Requests\", code: None, http_status: Some(429), refusal: false, source_chain: [], request_id: None, provider_response: None, detail: None }"
            );
        }
        assert!(report.provider_response.is_none());
        assert!(report.request_id.is_none());
        assert_eq!(observed_http.body_polls.load(Ordering::SeqCst), 0);
        assert_eq!(observed_http.sends.load(Ordering::SeqCst), 1);
        handle.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeFailed);
            assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
            assert!(view.committed_prefix.is_empty());
            let head = view.reply_head.expect("returned rejection head");
            assert_eq!(head.status, status);
            assert_eq!(head.headers, headers);
            assert!(!head.accepted_sse);
        });
        assert_eq!(
            observed.finish().expect_err("retained rejection").kind,
            ObservedFailureKind::Stream
        );
    }
}
