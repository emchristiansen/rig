use super::*;

#[test]
fn rejection_and_unpolled_drop_have_distinct_stages_and_no_dispatch() {
    let (rejected, writer) = Observation::prepare();
    writer.reject_before_dispatch();
    rejected.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::CallerRejectedBeforeDispatch);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.committed_prefix.is_empty());
    });
    let (cancelled, writer) = Observation::prepare();
    drop(writer);
    cancelled.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
    });
}

#[test]
fn original_prefix_and_send_attempt_survive_owner_drop() {
    let (handle, mut writer) = Observation::prepare();
    let retained = handle.clone();
    assert!(retained.same_operation(&writer.handle()));
    handle.mark_send_attempt_started();
    let original = "  {\"delta\":\"first\",\"delta\":\"second\"}  \n".to_owned();
    assert!(writer.append(original.clone()).is_ok());
    drop(writer);
    retained.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert_eq!(view.committed_prefix, &[original]);
        assert!(view.capture_fault.is_none());
    });
    let (other, _writer) = Observation::prepare();
    assert!(!retained.same_operation(&other));
}

#[test]
fn poisoned_append_retains_prefix_and_whole_uncommitted_string() {
    let (handle, mut writer) = Observation::prepare();
    handle.mark_send_attempt_started();
    assert!(writer.append("prefix".into()).is_ok());
    // Real allocation failure, induced as a deterministic capacity overflow.
    let mut allocation = Vec::<String>::new();
    let cause = match allocation.try_reserve(usize::MAX) {
        Err(cause) => cause,
        Ok(()) => panic!("an impossible String-vector capacity must fail"),
    };
    handle.lock().fault = Some(CaptureFault {
        cause: CaptureCause::Allocation(cause),
        offending_string: None,
    });
    let offending = "{\"type\":\"error\",\"error\":\"whole supplied cause\"}".to_owned();
    assert!(writer.append(offending.clone()).is_err());
    let failure = ObservedFailure::capture_only(&mut writer);
    assert_eq!(failure.kind, ObservedFailureKind::CaptureOnly);
    assert!(failure.observation.same_operation(&handle));
    assert!(matches!(
        failure.diagnostic_basis,
        ExistingDiagnosticBasis::NoExistingNativeCause
    ));
    assert!(writer.succeeded().is_err());
    drop(writer);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, &["prefix".to_owned()]);
        let Some(fault) = view.capture_fault else {
            panic!("actual capture fault must remain")
        };
        assert_eq!(fault.offending_string.as_ref(), Some(&offending));
        assert!(view.native_cause.is_none());
    });
}

#[test]
fn finalization_and_caller_admission_have_separate_lifetimes() {
    let (handle, mut writer) = Observation::prepare();
    handle.mark_send_attempt_started();
    writer.awaiting_finalization();
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization));
    assert!(writer.succeeded().is_ok());
    drop(writer);
    // A caller retaining this handle may reject its returned answer separately.
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::NativeSucceeded));
}

#[test]
fn fakes_are_explicitly_synthetic() {
    let (handle, writer) = Observation::prepare_synthetic();
    handle.inspect(|view| {
        assert_eq!(view.provenance, ObservationProvenance::Synthetic);
        assert!(view.committed_prefix.is_empty());
    });
    drop(writer);
}

#[test]
fn poisoned_inspection_keeps_actual_attempt_and_uncommitted_evidence() {
    let (handle, mut writer) = Observation::prepare();
    assert!(writer.append("committed prefix".into()).is_ok());
    let panic = std::panic::catch_unwind(|| {
        handle.inspect(|_| panic!("fixture visitor panic while holding the ledger"));
    });
    assert!(panic.is_err());
    // The send-future poll is a fact even when an earlier visitor poisoned capture.
    handle.mark_send_attempt_started();
    let original = "  {\"type\":\"error\",\"whole\":\"supplied\"}  ".to_owned();
    assert!(writer.append(original.clone()).is_err());
    let failure = ObservedFailure::capture_only(&mut writer);
    assert!(failure.observation.same_operation(&handle));
    drop(writer);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert_eq!(view.committed_prefix, &["committed prefix".to_owned()]);
        let fault = view.capture_fault.expect("actual mutex poison remains");
        assert!(matches!(fault.cause, CaptureCause::LockPoisoned));
        assert_eq!(fault.offending_string.as_ref(), Some(&original));
        assert!(view.native_cause.is_none());
    });
}

// ── an observed stream over Rig's driver ────────────────────────────────

use crate::wire::CallerIdentity;
use futures::StreamExt;
use http::StatusCode;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_http::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde_json::json;

fn wire() -> CodexResponses {
    CodexResponses::new(Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        "gpt-5.4",
    ))
    .expect("the ChatGPT dialect speaks the Codex contract")
    .with_caller_identity(
        CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
            .expect("a valid caller identity"),
    )
}

fn response(status: &str, output: Value) -> Value {
    json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": status,
        "model": "gpt-5.4", "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    })
}

fn message(text: &str) -> Value {
    json!({
        "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "annotations": [], "text": text}]
    })
}

/// The data strings of a turn that answers `hello` in two deltas.
fn turn_data() -> Vec<String> {
    let delta = |text: &str, sequence: u64| {
        json!({
            "type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
            "content_index": 0, "sequence_number": sequence, "delta": text,
        })
        .to_string()
    };
    vec![
        json!({"type": "response.created", "sequence_number": 0, "response": response("in_progress", json!([]))}).to_string(),
        json!({"type": "response.output_item.added", "output_index": 0, "sequence_number": 1,
            "item": {"type": "message", "id": "msg_1", "status": "in_progress", "role": "assistant", "content": []}}).to_string(),
        // Spacing inside the data is the provider's and is kept.
        format!("{}  ", delta("hel", 2)),
        delta("lo", 3),
        json!({"type": "response.output_item.done", "output_index": 0, "sequence_number": 4, "item": message("hello")}).to_string(),
        json!({"type": "response.completed", "sequence_number": 5, "response": response("completed", json!([message("hello")]))}).to_string(),
    ]
}

/// A server-sent-event body of `data` strings, one event each.
fn sse(data: &[String]) -> Bytes {
    let mut body = String::new();
    for data in data {
        body.push_str("data: ");
        body.push_str(data);
        body.push_str("\n\n");
    }
    Bytes::from(body)
}

fn replying(body: Bytes) -> SequencedHttpClient {
    SequencedHttpClient::new([MockHttpResponse::success_typed(body, "text/event-stream")])
}

async fn drain(
    stream: &mut ObservedResponsesStream,
) -> Vec<Result<ObservedEvent, ObservedFailure>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}

/// An observed stream sends the wire's ordinary Codex request, commits each
/// event's data exactly as supplied before Rig decodes it, stops at the
/// terminal, and finishes with Rig's response and the provider's.
#[tokio::test]
async fn an_observed_stream_commits_every_event_as_supplied_and_finishes_with_rigs_response() {
    let data = turn_data();
    let mut reply = data.clone();
    // Whatever follows the terminal is neither read nor captured.
    reply.push(json!({"type": "response.created", "sequence_number": 6, "response": response("in_progress", json!([]))}).to_string());
    let transport = replying(sse(&reply));
    let (observation, writer) = Observation::prepare();
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Prepared);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
    });

    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    assert!(stream.observation().same_operation(&observation));
    assert!(
        transport.requests().is_empty(),
        "nothing is sent until polled"
    );

    let texts: Vec<String> = drain(&mut stream)
        .await
        .into_iter()
        .map(|item| match item.expect("the turn has no failure") {
            ObservedEvent::TextDelta { text } => text,
            other @ ObservedEvent::Unknown { .. } => panic!("no unknown frame was sent: {other:?}"),
        })
        .collect();
    assert_eq!(texts, ["hel", "lo"]);
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization);
    });

    let result = stream.finish().await.expect("the turn finishes");
    assert!(result.observation.same_operation(&observation));
    assert_eq!(result.body.response.text(), "hello");
    assert_eq!(result.body.response.raw["id"], "resp_1");
    assert_eq!(result.body.native.id, "resp_1");
    assert_eq!(
        result
            .body
            .native
            .usage
            .as_ref()
            .map(|usage| usage.total_tokens),
        Some(15)
    );

    observation.inspect(|view| {
        assert_eq!(view.provenance, ObservationProvenance::SuppliedSse);
        assert_eq!(view.stage, ObservationStage::NativeSucceeded);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert_eq!(view.committed_prefix, data.as_slice());
        assert!(view.capture_fault.is_none());
        assert!(view.native_cause.is_none());
        let head = view.reply_head.expect("the reply's head");
        assert_eq!(head.status, StatusCode::OK);
        assert!(head.accepted_sse);
        assert_eq!(
            head.headers[http::header::CONTENT_TYPE],
            "text/event-stream"
        );
    });

    let requests = transport.requests();
    assert_eq!(requests.len(), 1, "one request: {requests:?}");
    assert_eq!(requests[0].headers["originator"], "client_exec");
    assert_eq!(
        requests[0].headers["user-agent"],
        "client_exec/1.2.3 (test)"
    );
}

/// A request the Codex wire refuses is a provider failure, and nothing is
/// sent.
#[tokio::test]
async fn a_request_that_cannot_be_encoded_is_a_provider_failure_and_sends_nothing() {
    let transport = replying(sse(&turn_data()));
    let (observation, writer) = Observation::prepare();
    let mut request = CompletionRequest::new("say hi");
    request.temperature = Some(0.25);

    let Err(failure) = wire().raw_stream_observed(&transport, request, writer) else {
        panic!("a caller-set temperature is refused");
    };
    assert_eq!(failure.kind, ObservedFailureKind::Provider);
    assert!(failure.observation.same_operation(&observation));
    assert!(matches!(
        failure.diagnostic_basis,
        ExistingDiagnosticBasis::Provider(_)
    ));
    assert!(transport.requests().is_empty());
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.committed_prefix.is_empty());
        assert!(view.native_error.is_some());
    });
}

/// A reply the provider refuses is a stream failure: its head is recorded,
/// none of its body is captured, and finishing returns the same failure.
#[tokio::test]
async fn a_refused_reply_is_a_stream_failure_with_its_head_and_no_captured_strings() {
    let transport = SequencedHttpClient::new([MockHttpResponse::ErrorResponse(
        StatusCode::TOO_MANY_REQUESTS,
        Bytes::from_static(br#"{"error":{"message":"quota exceeded"}}"#),
    )]);
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");

    let items = drain(&mut stream).await;
    let [Err(failure)] = items.as_slice() else {
        panic!("exactly one failure, got {items:?}");
    };
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    assert!(failure.observation.same_operation(&observation));
    let finished = stream.finish().await.expect_err("the stream failed");
    assert_eq!(finished.kind, ObservedFailureKind::Stream);

    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert!(view.committed_prefix.is_empty());
        assert!(view.capture_fault.is_none());
        assert!(matches!(
            view.native_cause,
            Some(ExistingDiagnosticBasis::Stream(_))
        ));
        let error = view.native_error.expect("Rig's own error is kept");
        assert!(error.to_string().contains("quota exceeded"), "{error}");
        let head = view.reply_head.expect("the refused reply's head");
        assert_eq!(head.status, StatusCode::TOO_MANY_REQUESTS);
        assert!(!head.accepted_sse);
    });
}

/// A frame of a modelled type that fails to decode is committed, exactly as
/// supplied, before it fails the stream.
#[tokio::test]
async fn a_corrupt_frame_is_committed_before_it_fails_the_stream() {
    let malformed = crate::responses_api::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES[0].frame;
    let data = vec![turn_data().remove(0), malformed.to_owned()];
    let transport = replying(sse(&data));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");

    let items = drain(&mut stream).await;
    let Some(Err(failure)) = items.last() else {
        panic!("the stream ends in a failure, got {items:?}");
    };
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, data.as_slice());
        let last = view.committed_prefix.last().expect("the malformed frame");
        let corrupt = decode_frame(last).expect_err("the kept frame is the corrupt one");
        assert_eq!(corrupt.frame(), Some(malformed));
    });
}

/// The ledger holds exactly the frames Rig's decoder was given. A frame the
/// decoder refuses is the ledger's last string, whatever followed it in the
/// chunk that carried it: the reply is handed on an event at a time, and
/// the driver stops reading at the refused frame. So no frame after the
/// failure is committed, a terminal event least of all.
#[tokio::test]
async fn the_ledger_of_a_failed_call_ends_with_the_frame_the_decoder_refused() {
    let refused = "{nope}".to_owned();
    let mut data = turn_data();
    data.insert(1, refused.clone());
    let transport = replying(sse(&data));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");

    let items = drain(&mut stream).await;
    let [Err(failure)] = items.as_slice() else {
        panic!("the stream yields the failure alone, got {items:?}");
    };
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(
            view.committed_prefix,
            [data[0].clone(), refused.clone()],
            "the frames the decoder was given, ending with the one it refused"
        );
    });
}

/// A bounded ledger keeps a reply that fits whole, to the byte, and fails a
/// call whose reply does not: the string that would pass the limit is not
/// committed, the strings before it stay as they were, none is dropped or
/// shortened, and the fault names the limit and holds the string that did
/// not fit.
#[tokio::test]
async fn a_bounded_ledger_keeps_the_whole_reply_or_fails_the_call() {
    let data = turn_data();
    let whole: usize = data.iter().map(String::len).sum();
    let bounded = |limit: usize| {
        Observation::prepare_bounded(NonZeroUsize::new(limit).expect("a non-zero limit"))
    };

    let transport = replying(sse(&data));
    let (observation, writer) = bounded(whole);
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    let items = drain(&mut stream).await;
    assert!(items.iter().all(Result::is_ok), "the reply fits: {items:?}");
    stream.finish().await.expect("the call succeeds");
    observation.inspect(|view| {
        assert_eq!(view.committed_prefix, data.as_slice());
        assert!(view.capture_fault.is_none());
    });

    let transport = replying(sse(&data));
    let (observation, writer) = bounded(whole - 1);
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    let items = drain(&mut stream).await;
    let Some(Err(failure)) = items.last() else {
        panic!("the stream ends in a failure, got {items:?}");
    };
    assert_eq!(failure.kind, ObservedFailureKind::CaptureOnly);
    let (terminal, before) = data.split_last().expect("the terminal event");
    let held: usize = before.iter().map(String::len).sum();
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(
            view.committed_prefix, before,
            "every string that fitted, whole"
        );
        let fault = view.capture_fault.expect("the limit is a capture fault");
        assert!(
            matches!(
                fault.cause,
                CaptureCause::LimitExceeded { limit, committed, offered }
                    if limit == whole - 1 && committed == held && offered == terminal.len()
            ),
            "the fault states the limit: {:?}",
            fault.cause
        );
        assert_eq!(fault.offending_string.as_ref(), Some(terminal));
    });
}

/// The observed body hands the reply on with its bytes unchanged and in
/// order, in pieces that each complete at most one event, and commits an
/// event before its piece is handed on. It ends with the piece that
/// completes the first terminal: bytes after it in the same chunk are not
/// handed on.
#[tokio::test]
async fn the_observed_body_hands_the_reply_on_an_event_at_a_time() {
    let data = turn_data();
    let mut reply = sse(&data).to_vec();
    let through_terminal = reply.len();
    reply.extend_from_slice(b"data: {\"type\":\"response.done\"}\n\n");
    // Chunks that split an event, hold several events, and run past the
    // terminal.
    let cut = [7, reply.len() / 3, reply.len() / 2];
    let mut chunks = Vec::new();
    let mut from = 0;
    for at in cut {
        chunks.push(reply[from..at].to_vec());
        from = at;
    }
    chunks.push(reply[from..].to_vec());

    let (observation, _writer) = Observation::prepare();
    let mut body = observed_body(chunks, &observation);
    let mut handed_on = Vec::new();
    let mut committed_before = 0;
    while let Some(piece) = body.next().await {
        handed_on.extend_from_slice(&piece.expect("reads"));
        let committed = observation.inspect(|view| view.committed_prefix.len());
        assert!(
            committed <= committed_before + 1,
            "a piece completes at most one event"
        );
        let cut_so_far = Framing::Sse
            .framer()
            .map(|mut framer| framer.push(&handed_on).count());
        assert_eq!(
            cut_so_far,
            Some(committed),
            "what is committed is what the bytes handed on complete"
        );
        committed_before = committed;
    }
    assert_eq!(
        handed_on,
        reply[..through_terminal],
        "the reply's bytes through the terminal, and none after it"
    );
    observation.inspect(|view| assert_eq!(view.committed_prefix, data.as_slice()));
}

/// Dropping the stream before its end records a cancelled operation and
/// keeps what was committed.
#[tokio::test]
async fn dropping_the_stream_before_its_end_records_a_cancelled_operation() {
    let transport = replying(sse(&turn_data()));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    let first = stream.next().await.expect("a first item");
    assert!(matches!(first, Ok(ObservedEvent::TextDelta { ref text }) if text == "hel"));
    drop(stream);

    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert!(!view.committed_prefix.is_empty());
        assert!(view.capture_fault.is_none());
    });
}

/// A frame of a type Rig does not model is delivered whole and committed.
#[tokio::test]
async fn an_unmodelled_frame_is_delivered_whole_and_committed() {
    let unknown = json!({"type": "codex.rate_limits", "plan_type": "pro"});
    let mut data = turn_data();
    data.insert(1, unknown.to_string());
    let transport = replying(sse(&data));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");

    let items = drain(&mut stream).await;
    let unknowns: Vec<(&str, &Value)> = items
        .iter()
        .filter_map(|item| match item {
            Ok(ObservedEvent::Unknown { event_type, value }) => Some((event_type.as_str(), value)),
            _ => None,
        })
        .collect();
    assert_eq!(unknowns, [("codex.rate_limits", &unknown)]);
    stream.finish().await.expect("the turn finishes");
    observation.inspect(|view| assert_eq!(view.committed_prefix, data.as_slice()));
}

/// A server-sent event that is not UTF-8 fails the stream and is not
/// captured, not even as replacement text.
#[tokio::test]
async fn a_frame_that_is_not_utf8_fails_the_stream_uncaptured() {
    let first = turn_data().remove(0);
    let mut body = sse(std::slice::from_ref(&first)).to_vec();
    body.extend_from_slice(
        b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"\xff\xfe\"}\n\n",
    );
    let transport = replying(Bytes::from(body));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");

    let items = drain(&mut stream).await;
    assert!(
        matches!(items.last(), Some(Err(failure)) if failure.kind == ObservedFailureKind::Stream),
        "{items:?}"
    );
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, std::slice::from_ref(&first));
    });
}

/// The observed body of a reply delivered in `chunks`.
fn observed_body(chunks: Vec<Vec<u8>>, observation: &ObservationHandle) -> ObservedBody {
    let chunks = chunks
        .into_iter()
        .map(|chunk| Ok::<_, http_client::Error>(Bytes::from(chunk)));
    ObservedBody {
        pending: Bytes::new(),
        body: Box::pin(futures::stream::iter(chunks)),
        framer: Framing::Sse
            .framer()
            .expect("server-sent events are framed"),
        partial: Vec::new(),
        observation: observation.clone(),
        ended: false,
    }
}

/// A character split across two transport chunks is committed whole, and
/// the bytes handed on are the transport's own chunks.
#[tokio::test]
async fn a_character_split_across_chunks_is_committed_whole() {
    let data = r#"{"type":"codex.note","text":"café ☕"}"#;
    let event = format!("data: {data}\n\n").into_bytes();
    // Cut inside the two-byte `é` and inside the three-byte `☕`.
    let first_cut = event.iter().position(|byte| *byte == 0xc3).expect("é") + 1;
    let second_cut = event.iter().position(|byte| *byte == 0xe2).expect("☕") + 2;
    let chunks = vec![
        event[..first_cut].to_vec(),
        event[first_cut..second_cut].to_vec(),
        event[second_cut..].to_vec(),
    ];
    let (observation, _writer) = Observation::prepare();
    let handed_on: Vec<Bytes> = observed_body(chunks.clone(), &observation)
        .map(|chunk| chunk.expect("no failure"))
        .collect()
        .await;
    assert_eq!(
        handed_on,
        chunks.into_iter().map(Bytes::from).collect::<Vec<_>>()
    );
    observation.inspect(|view| assert_eq!(view.committed_prefix, [data.to_owned()]));
}

/// The body ends with the chunk that completes the terminal event: a later
/// chunk is never read.
#[tokio::test]
async fn the_body_ends_with_the_chunk_that_completes_the_terminal() {
    let data = turn_data();
    let terminal = data.last().expect("the terminal event").clone();
    let chunks = vec![
        sse(std::slice::from_ref(&terminal)).to_vec(),
        sse(std::slice::from_ref(&data[0])).to_vec(),
    ];
    let (observation, _writer) = Observation::prepare();
    let handed_on: Vec<Bytes> = observed_body(chunks.clone(), &observation)
        .map(|chunk| chunk.expect("no failure"))
        .collect()
        .await;
    assert_eq!(handed_on, [Bytes::from(chunks[0].clone())]);
    observation.inspect(|view| assert_eq!(view.committed_prefix, [terminal]));
}

/// One ordinary (unobserved) stream of the same request on the same wire,
/// read to its end: what it sent and how it ended.
async fn ordinary(
    transport: &SequencedHttpClient,
) -> Result<rig_core::completion::CompletionResponse, ProviderError> {
    let wire = wire();
    let model = Model::new(wire.clone(), CodexHttpClient::new(transport.clone(), wire));
    let mut stream = model.stream(CompletionRequest::new("say hi"))?;
    while let Some(item) = stream.next().await {
        item?;
    }
    stream.finish().await
}

/// An observed stream sends exactly the request an ordinary stream of the
/// same wire sends: same target, headers and body.
#[tokio::test]
async fn observed_and_ordinary_streams_send_the_same_request() {
    let observed_transport = replying(sse(&turn_data()));
    let (_observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(
            &observed_transport,
            CompletionRequest::new("say hi"),
            writer,
        )
        .expect("the request encodes");
    drain(&mut stream).await;
    stream.finish().await.expect("the turn finishes");

    let ordinary_transport = replying(sse(&turn_data()));
    ordinary(&ordinary_transport)
        .await
        .expect("the turn finishes");

    let observed = observed_transport.requests();
    let plain = ordinary_transport.requests();
    assert_eq!(observed.len(), 1);
    assert_eq!(plain.len(), 1);
    assert_eq!(observed[0].uri, plain[0].uri);
    assert_eq!(observed[0].body, plain[0].body);
    // Rig draws its own session id per request when the wire names no
    // conversation; every other header is the same.
    let headers = |request: &rig_http::test_utils::CapturedHttpRequest| {
        let mut headers: Vec<(String, Vec<u8>)> = request
            .headers
            .iter()
            .filter(|(name, _)| !matches!(name.as_str(), "session-id" | "session_id"))
            .map(|(name, value)| (name.as_str().to_owned(), value.as_bytes().to_vec()))
            .collect();
        headers.sort();
        headers
    };
    assert_eq!(headers(&observed[0]), headers(&plain[0]));
}

/// A refused reply fails an observed stream with the error an ordinary
/// stream reports for it.
#[tokio::test]
async fn refused_statuses_report_what_the_ordinary_stream_reports() {
    for status in [
        StatusCode::BAD_REQUEST,
        StatusCode::UNAUTHORIZED,
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::INTERNAL_SERVER_ERROR,
    ] {
        let refusing = || {
            SequencedHttpClient::new([MockHttpResponse::ErrorResponse(
                status,
                Bytes::from_static(br#"{"error":{"message":"refused","code":"some_code"}}"#),
            )])
        };
        let expected = ordinary(&refusing())
            .await
            .expect_err("the reply is refused");

        let (observation, writer) = Observation::prepare();
        let mut stream = wire()
            .raw_stream_observed(&refusing(), CompletionRequest::new("say hi"), writer)
            .expect("the request encodes");
        let items = drain(&mut stream).await;
        assert!(matches!(items.as_slice(), [Err(_)]), "{status}: {items:?}");
        observation.inspect(|view| {
            let error = view.native_error.expect("Rig's error is kept");
            assert_eq!(error.to_string(), expected.to_string(), "{status}");
            assert_eq!(error.report(), expected.report(), "{status}");
        });
    }
}

/// A stream dropped before it is polled sent nothing and is cancelled.
#[tokio::test]
async fn dropping_an_unpolled_observed_stream_cancels_without_dispatch() {
    let transport = replying(sse(&turn_data()));
    let (observation, writer) = Observation::prepare();
    let stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    drop(stream);
    assert!(transport.requests().is_empty());
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.committed_prefix.is_empty());
        assert!(view.reply_head.is_none());
    });
}

/// Whatever shares the terminal's chunk after it is ignored: more events,
/// a second terminal, or bytes that are not UTF-8. The turn finishes and
/// the ledger ends at the first terminal.
#[tokio::test]
async fn the_first_terminal_ends_the_turn_whatever_follows_it() {
    let data = turn_data();
    let later_terminal = json!({"type": "response.failed", "sequence_number": 6, "response": response("failed", json!([]))}).to_string();
    for tail in [
        sse(std::slice::from_ref(&data[0])).to_vec(),
        sse(std::slice::from_ref(&later_terminal)).to_vec(),
        b"data: \xff\xfe\n\n".to_vec(),
    ] {
        let mut body = sse(&data).to_vec();
        body.extend_from_slice(&tail);
        let transport = replying(Bytes::from(body));
        let (observation, writer) = Observation::prepare();
        let mut stream = wire()
            .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
            .expect("the request encodes");
        let items = drain(&mut stream).await;
        assert!(items.iter().all(Result::is_ok), "{items:?}");
        let result = stream.finish().await.expect("the turn finishes");
        assert_eq!(result.body.response.text(), "hello");
        observation.inspect(|view| {
            assert_eq!(view.stage, ObservationStage::NativeSucceeded);
            assert_eq!(view.committed_prefix, data.as_slice());
        });
    }
}

/// A reply that ends without a terminal event fails the stream after its
/// delivered text, as Rig's own stream does (the fork reported it at
/// finalization instead), and what was committed stays.
#[tokio::test]
async fn a_reply_without_a_terminal_is_a_failure_that_keeps_its_prefix() {
    let mut data = turn_data();
    data.pop();
    let transport = replying(sse(&data));
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    let items = drain(&mut stream).await;
    let Some(Err(failure)) = items.last() else {
        panic!("the stream's last item is the failure, got {items:?}");
    };
    assert_eq!(failure.kind, ObservedFailureKind::Stream);
    assert_eq!(items.iter().filter(|item| item.is_ok()).count(), 2);
    let finished = stream.finish().await.expect_err("no terminal, no response");
    assert_eq!(finished.kind, ObservedFailureKind::Stream);
    observation.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, data.as_slice());
        assert!(view.native_error.is_some());
    });
}

/// The reply's head is kept whole: repeated headers and values that are
/// not UTF-8 included.
#[tokio::test]
async fn the_reply_head_is_kept_whole() {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    headers.append("x-codex-note", http::HeaderValue::from_static("one"));
    headers.append("x-codex-note", http::HeaderValue::from_static("two"));
    headers.insert(
        "x-opaque",
        http::HeaderValue::from_bytes(b"\xff\xfe").expect("an opaque header value"),
    );
    let transport = SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        sse(&turn_data()),
        headers.clone(),
    )]);
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(&transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    drain(&mut stream).await;
    stream.finish().await.expect("the turn finishes");
    observation.inspect(|view| {
        let head = view.reply_head.expect("the reply's head");
        assert!(head.accepted_sse);
        assert_eq!(head.headers, headers);
    });
}

/// A success that names no content type, as the Codex gateway replies.
fn replying_without_a_content_type(body: Bytes) -> SequencedHttpClient {
    SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        body,
        http::HeaderMap::new(),
    )])
}

/// One observed call of `transport`: the strings it committed and what it
/// finished with.
async fn observed_call(
    transport: &SequencedHttpClient,
) -> (
    Vec<String>,
    Result<ObservedResponsesResult, ObservedFailure>,
    bool,
) {
    let (observation, writer) = Observation::prepare();
    let mut stream = wire()
        .raw_stream_observed(transport, CompletionRequest::new("say hi"), writer)
        .expect("the request encodes");
    drain(&mut stream).await;
    let result = stream.finish().await;
    let (prefix, accepted) = observation.inspect(|view| {
        (
            view.committed_prefix.to_vec(),
            view.reply_head.is_some_and(|head| head.accepted_sse),
        )
    });
    (prefix, result, accepted)
}

/// A reply that names no content type is the Codex gateway's ordinary reply
/// and Rig's Codex wire reads it. It is captured exactly as a reply that
/// names `text/event-stream` is: the same strings, the same response.
#[tokio::test]
async fn a_reply_that_names_no_content_type_is_captured_as_a_typed_one_is() {
    let data = turn_data();
    let (typed_prefix, typed, typed_accepted) = observed_call(&replying(sse(&data))).await;
    let (bare_prefix, bare, bare_accepted) =
        observed_call(&replying_without_a_content_type(sse(&data))).await;

    assert!(typed_accepted && bare_accepted);
    assert_eq!(typed_prefix, data);
    assert_eq!(bare_prefix, data, "a headerless reply commits every event");
    let (typed, bare) = (typed.expect("finishes"), bare.expect("finishes"));
    assert_eq!(bare.body.response.text(), "hello");
    assert_eq!(bare.body.response.raw, typed.body.response.raw);
    assert_eq!(bare.body.native.id, typed.body.native.id);
}

/// The safeguards hold for a reply that names no content type: bytes that
/// are not UTF-8 fail it uncaptured, and reading ends at the first terminal.
#[tokio::test]
async fn a_headerless_reply_keeps_the_utf8_and_terminal_rules() {
    let data = turn_data();
    let mut invalid = sse(&data[..1]).to_vec();
    invalid.extend_from_slice(
        b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"\xff\xfe\"}\n\n",
    );
    let (prefix, result, _) =
        observed_call(&replying_without_a_content_type(Bytes::from(invalid))).await;
    assert_eq!(prefix, data[..1]);
    result.expect_err("a reply that is not UTF-8 fails");

    let mut reply = data.clone();
    reply.push(json!({"type": "response.created", "sequence_number": 6, "response": response("in_progress", json!([]))}).to_string());
    let (prefix, result, _) = observed_call(&replying_without_a_content_type(sse(&reply))).await;
    assert_eq!(prefix, data, "nothing after the terminal is committed");
    result.expect("the turn finishes");
}

/// The strings an observed call committed are the frames Rig's decoder
/// consumed: folded by Rig's public reply decoder they give the response
/// the call returned. The reply carries what a transport adds and the
/// decoder never sees: comment lines, heartbeat events, `event:` and `id:`
/// fields, data split over lines, and an event after the terminal.
#[tokio::test]
async fn the_committed_strings_fold_to_the_response_the_call_returned() {
    use rig_core::wire::{Mode, WireFrame, decode_reply};
    let data = turn_data();
    let mut body = String::from(": connected\n\n");
    let mut expected = Vec::new();
    for (index, event) in data.iter().enumerate() {
        let (head, tail) = event.split_once(',').expect("an object with two members");
        body.push_str(&format!(
            "event: message\nid: {index}\ndata: {head},\ndata: {tail}\n\n"
        ));
        body.push_str("data:  \n\n: ping\n\n");
        expected.push(format!("{head},\n{tail}"));
    }
    body.push_str("data: {\"type\":\"response.created\",\"sequence_number\":9}\n\n");

    let (prefix, result, _) =
        observed_call(&replying_without_a_content_type(Bytes::from(body))).await;
    assert_eq!(
        prefix, expected,
        "the joined data of each event, and nothing else"
    );

    let live = result.expect("the turn finishes").body.response;
    let folded = decode_reply(
        &wire(),
        &CompletionRequest::new("say hi"),
        Mode::Streaming,
        prefix.into_iter().map(WireFrame::Text),
    )
    .expect("the committed strings fold");
    assert_eq!(folded.choice, live.choice);
    assert_eq!(folded.raw, live.raw);
    assert_eq!(folded.usage, live.usage);
    assert_eq!(folded.finish_reason(), live.finish_reason());
    assert_eq!(
        serde_json::to_value(folded.message().expect("a message")).expect("serializes"),
        serde_json::to_value(live.message().expect("a message")).expect("serializes"),
    );
}
