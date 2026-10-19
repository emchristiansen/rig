use super::*;
use crate::completion::{
    CompletionRequest, CompletionResponse, FinishReason, ProviderResponseHeaders, Usage,
};
use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::ProviderError;
use crate::message::{AssistantContent, CarriedReasoningPart};
use crate::operation::Finish;
use crate::test_utils::{MockCompletionModel, MockFrame, MockScript, MockStreamEvent};
use futures::StreamExt;
use serde_json::{Value, json};
use std::task::Poll;
use std::time::Duration;

const TEST_PROVIDER: &str = "test-provider";

fn model(events: impl IntoIterator<Item = MockStreamEvent>) -> MockCompletionModel {
    let mut model =
        MockCompletionModel::from_stream_turns([events.into_iter().collect::<Vec<_>>()]);
    model.wire = MockScript::new(TEST_PROVIDER);
    model
}

fn stream_of(events: impl IntoIterator<Item = MockStreamEvent>) -> CompletionStream {
    model(events)
        .stream(CompletionRequest::new("streaming consumer"))
        .expect("fixture stream opens")
}

fn total(total_tokens: u64) -> MockStreamEvent {
    MockStreamEvent::final_response(Usage::new().total_tokens(total_tokens))
}

fn signed(id: &str, text: &str, signature: &str) -> MockStreamEvent {
    MockStreamEvent::Reasoning {
        id: id.to_owned(),
        content: CarriedReasoningPart::Text {
            text: text.to_owned(),
            signature: Some(signature.to_owned()),
        },
    }
}

async fn drain(stream: &mut CompletionStream) {
    while let Some(item) = stream.next().await {
        item.expect("fixture item");
    }
}

// Only the fixture transport supplies reply evidence. Model's real reader,
// decoder, reassembler and Turn still produce every returned response.
#[derive(Clone)]
struct CapturedFixture {
    events: Vec<MockStreamEvent>,
    document: Option<Value>,
    headers: ProviderResponseHeaders,
    delay: bool,
    request_id: Option<String>,
}

impl Transport<MockScript> for CapturedFixture {
    fn send(&self, _: CompletionRequest, _: Exchange) -> Opening<MockFrame> {
        let events = self.events.clone();
        let delay = self.delay;
        let frames = async_stream::stream! {
            for event in events {
                yield Ok(MockFrame::Event(event));
                if delay { tokio::time::sleep(Duration::from_millis(50)).await; }
            }
        };
        let mut opened = Opened::new(frames)
            .with_captured_response_headers(self.headers.clone())
            .with_request_id(self.request_id.clone());
        if let Some(document) = &self.document {
            opened = opened.with_document(document.clone());
        }
        Opening::ready(opened)
    }
}

fn captured(
    events: Vec<MockStreamEvent>,
    document: Option<Value>,
    headers: ProviderResponseHeaders,
    delay: bool,
) -> CompletionStream {
    Model::new(
        MockScript::new(TEST_PROVIDER),
        CapturedFixture {
            events,
            document,
            headers,
            delay,
            request_id: None,
        },
    )
    .stream(CompletionRequest::new("captured fixture"))
    .expect("fixture stream opens")
}

fn create_mock_stream() -> CompletionStream {
    captured(
        vec![
            MockStreamEvent::text("hello 1"),
            MockStreamEvent::text("hello 2"),
            MockStreamEvent::text("hello 3"),
            total(15),
        ],
        None,
        ProviderResponseHeaders::new(),
        true,
    )
}

#[tokio::test]
async fn finish_derives_usage_from_final_response() {
    let mut stream = create_mock_stream();
    drain(&mut stream).await;
    assert_eq!(stream.partial().usage.total_tokens, Some(15));
    let response = stream
        .finish()
        .await
        .expect("the stream produced a terminal record");
    assert_eq!(response.usage.total_tokens, Some(15));
    assert_eq!(response.provider(), TEST_PROVIDER);
}

#[tokio::test]
async fn a_stream_without_a_terminal_record_names_its_provider_and_refuses_to_fold() {
    let mut stream = stream_of([MockStreamEvent::text("truncated")]);
    while stream.next().await.is_some() {}
    assert!(lock(&stream.shared).end.is_none());
    assert_eq!(stream.partial().provider(), TEST_PROVIDER);
    assert_eq!(stream.partial().usage, Usage::default());
    let error = stream
        .finish()
        .await
        .expect_err("a stream without a terminal record has no response to fold");
    // G160: typed truncation survives; only the former substring diagnostic is retired.
    assert!(matches!(&error, ProviderError::Truncated));
    assert_eq!(
        error.to_string(),
        "ResponseError: the reply ended before the provider ended it"
    );
}

#[tokio::test]
async fn a_stream_that_errors_mid_stream_keeps_content_and_omits_the_terminal() {
    let mut stream = stream_of([
        MockStreamEvent::text("partial"),
        MockStreamEvent::error("connection reset"),
    ]);
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if item.is_err() {
            saw_error = true;
        }
    }
    assert!(saw_error, "the mid-stream error must be forwarded");
    assert!(lock(&stream.shared).end.is_none());
    assert_eq!(
        stream.partial().choice.first(),
        Some(&AssistantContent::text("partial"))
    );
}

#[tokio::test]
async fn a_stop_that_carried_a_tool_call_is_upgraded_to_tool_calls() {
    let mut stream = stream_of([
        MockStreamEvent::tool_call("call_1", "lookup", json!({})),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Stop),
            ..Finish::default()
        }),
    ]);
    drain(&mut stream).await;
    assert_eq!(
        stream.partial().finish_reason(),
        Some(FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn a_stop_without_tool_calls_is_left_alone() {
    let mut stream = stream_of([
        MockStreamEvent::text("done"),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Stop),
            ..Finish::default()
        }),
    ]);
    drain(&mut stream).await;
    assert_eq!(stream.partial().finish_reason(), Some(FinishReason::Stop));
}

#[tokio::test]
async fn finish_reason_is_reconciled_with_raw_attached() {
    let raw = serde_json::to_value(Usage::default()).expect("serialize usage");
    let mut stream = captured(
        vec![
            MockStreamEvent::tool_call("call_1", "lookup", json!({})),
            MockStreamEvent::FinalResponse(Finish {
                reason: Some(FinishReason::Stop),
                ..Finish::default()
            }),
        ],
        Some(raw.clone()),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("genuine provider end");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(response.raw, raw);
}

#[tokio::test]
async fn usage_is_unreported_before_final_response() {
    let stream = stream_of([MockStreamEvent::text("no final response")]);
    assert!(!stream.partial().usage.is_reported());
}

#[tokio::test]
async fn test_stream_cancellation() {
    let mut stream = create_mock_stream();
    let mut chunk_count = 0;
    while let Some(chunk) = stream.next().await {
        match chunk.expect("fixture event") {
            Item::Event(
                StreamEvent::Text { .. }
                | StreamEvent::Reasoning { .. }
                | StreamEvent::Arguments { .. }
                | StreamEvent::Input { .. },
            )
            | Item::Unknown(_) => chunk_count += 1,
            _ => {}
        }
        if chunk_count >= 2 {
            stream.cancel();
            break;
        }
    }
    let next_chunk = stream.next().await;
    assert!(
        next_chunk.is_none(),
        "Expected no further chunks after cancellation, got {next_chunk:?}"
    );
}

#[tokio::test]
async fn test_stream_pause_resume() {
    let stream = create_mock_stream();
    stream.pause();
    assert!(stream.is_paused());
    stream.resume();
    assert!(!stream.is_paused());
}

#[tokio::test]
async fn a_paused_stream_parks_until_resume_instead_of_busy_waking() {
    let stream = stream_of([MockStreamEvent::text("hello")]);
    let resume = stream.pause_control();
    stream.pause();
    let mut task = tokio_test::task::spawn(stream);
    assert!(
        task.poll_next().is_pending(),
        "a paused stream yields nothing"
    );
    assert!(
        !task.is_woken(),
        "a paused stream must idle, not re-wake itself"
    );
    resume.resume();
    assert!(task.is_woken(), "resuming must wake the parked stream");
    // Current Start is visible before the original hello text.
    assert!(matches!(
        task.poll_next(),
        Poll::Ready(Some(Ok(Item::Event(StreamEvent::Start { .. }))))
    ));
    assert!(
        matches!(task.poll_next(), Poll::Ready(Some(Ok(Item::Event(StreamEvent::Text { text, .. })))) if text == "hello")
    );
}

#[tokio::test]
async fn cancelling_a_paused_stream_terminates_instead_of_deadlocking() {
    let mut stream = create_mock_stream();
    stream.pause();
    stream.cancel();
    assert!(
        !stream.is_paused(),
        "cancel must lift the pause so the termination is observable"
    );
    assert!(
        stream.next().await.is_none(),
        "a cancelled stream terminates"
    );
}

#[tokio::test]
async fn re_polling_a_drained_stream_preserves_the_aggregated_choice() {
    let mut stream = create_mock_stream();
    drain(&mut stream).await;
    let drained = stream.partial().choice;
    assert_eq!(
        drained,
        vec![AssistantContent::text("hello 1hello 2hello 3")]
    );
    for _ in 0..3 {
        assert!(
            stream.next().await.is_none(),
            "a drained stream stays drained"
        );
    }
    assert_eq!(
        stream.partial().choice,
        drained,
        "re-polling must not disturb the aggregated choice"
    );
    let response = stream
        .finish()
        .await
        .expect("the stream produced a terminal record");
    assert_eq!(response.choice, drained);
}

#[tokio::test]
async fn a_provider_error_mentioning_aborted_reaches_the_consumer() {
    let mut stream = stream_of([
        MockStreamEvent::text("partial"),
        MockStreamEvent::error("upstream aborted the request"),
    ]);
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            errors.push(err.to_string());
        }
    }
    assert_eq!(errors.len(), 1, "the error must not be swallowed");
    assert!(errors[0].contains("upstream aborted the request"));
    assert_eq!(
        stream.partial().choice.first(),
        Some(&AssistantContent::text("partial"))
    );
    assert!(lock(&stream.shared).end.is_none());
}

#[tokio::test]
async fn a_full_tool_call_correlates_with_the_deltas_of_the_same_id() {
    let mut stream = stream_of([
        MockStreamEvent::tool_call_name_delta("tc1", "add"),
        MockStreamEvent::tool_call_arguments_delta("tc1", "{\"x\":1}"),
        MockStreamEvent::tool_call("tc1", "add", json!({"x":1})),
        MockStreamEvent::tool_call_end("tc1"),
        total(1),
    ]);
    let mut delta_ids = Vec::new();
    let mut completed_ids = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            Item::Event(StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name: Some(_),
                ..
            })
            | Item::Event(StreamEvent::Arguments { part, .. }) => delta_ids.push(part),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(_),
                ..
            }) => completed_ids.push(part),
            _ => {}
        }
    }
    assert_eq!(delta_ids.len(), 2);
    assert_eq!(delta_ids[0], delta_ids[1], "one call, one block id");
    assert_eq!(
        completed_ids,
        vec![delta_ids[0]],
        "the completed call must carry the id its deltas published"
    );
    let partial = stream.partial();
    let tool_calls: Vec<_> = partial
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(tool_calls.len(), 1, "got {:?}", partial.choice);
}

#[tokio::test]
async fn test_stream_aggregates_reasoning_content() {
    let mut stream = stream_of([
        signed("rs_1", "step one", "sig_1"),
        MockStreamEvent::text("final answer"),
        total(5),
    ]);
    drain(&mut stream).await;
    assert!(stream.partial().choice.iter().any(|item| matches!(item,
    AssistantContent::CarriedReasoning(reasoning) if reasoning.id.as_deref() == Some("rs_1")
        && reasoning.content.first() == Some(&CarriedReasoningPart::Text {
            text: "step one".to_owned(), signature: Some("sig_1".to_owned()),
        }))));
}

#[tokio::test]
async fn full_reasoning_block_supersedes_its_accumulated_deltas() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("rs_1", "partial "),
        signed("rs_1", "the complete chain", "sig_1"),
        total(2),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    let reasoning: Vec<_> = choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::CarriedReasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning.len(), 1, "got {choice:?}");
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning[0].content.first(),
        Some(&CarriedReasoningPart::Text {
            text: "the complete chain".to_owned(),
            signature: Some("sig_1".to_owned()),
        })
    );
}

#[tokio::test]
async fn test_stream_reasoning_only_does_not_inject_empty_text() {
    let mut stream = stream_of([
        MockStreamEvent::Reasoning {
            id: "rs_only".to_owned(),
            content: CarriedReasoningPart::Summary("hidden summary".to_owned()),
        },
        total(2),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 1);
    assert!(
        matches!(choice.first(), Some(AssistantContent::CarriedReasoning(reasoning)) if reasoning.id.as_deref() == Some("rs_only"))
    );
}

#[tokio::test]
async fn unknown_choice_reaches_consumer_but_not_aggregated_choice() {
    let unknown = json!({"type":"web_search_call","id":"ws_1","status":"completed"});
    let mut stream = stream_of([
        MockStreamEvent::unknown(unknown.clone()),
        MockStreamEvent::text("done"),
        total(1),
    ]);
    let mut consumer_unknown = None;
    let mut consumer_text = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            Item::Unknown(value) => consumer_unknown = Some(value),
            Item::Event(StreamEvent::Text { text, .. }) => consumer_text.push_str(&text),
            _ => {}
        }
    }
    assert_eq!(consumer_unknown.as_ref(), Some(&unknown.into()));
    assert_eq!(consumer_text, "done");
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 1);
    assert!(matches!(choice.first(), Some(AssistantContent::Text(text)) if text.text == "done"));
}

#[tokio::test]
async fn test_stream_keeps_non_contiguous_text_chunks_split_by_tool_call() {
    let mut stream = stream_of([
        MockStreamEvent::text("first"),
        MockStreamEvent::tool_call("tool_split", "mock_tool", json!({"arg":"x"})),
        MockStreamEvent::text("second"),
        total(3),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 3);
    assert!(matches!(choice.first(), Some(AssistantContent::Text(text)) if text.text == "first"));
    assert!(
        matches!(choice.get(1), Some(AssistantContent::ToolCall(call)) if call.id.provider().map(|id| id.as_str()) == Some("tool_split"))
    );
    assert!(matches!(choice.get(2), Some(AssistantContent::Text(text)) if text.text == "second"));
}

#[tokio::test]
async fn test_stream_preserves_text_additional_params() {
    let mut stream = stream_of([
        MockStreamEvent::text_start("block-0", None),
        MockStreamEvent::text("first"),
        MockStreamEvent::text_additional_params(
            json!({"citations":[{"type":"char_location","cited_text":"First citation.","document_index":0,"start_char_index":0,"end_char_index":15}]}),
        ),
        MockStreamEvent::text_additional_params(
            json!({"citations":[{"type":"char_location","cited_text":"Second citation.","document_index":0,"start_char_index":16,"end_char_index":32}]}),
        ),
        MockStreamEvent::text_start("block-1", Some(json!({"block":2}))),
        MockStreamEvent::text("second"),
        total(3),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 2);
    let Some(AssistantContent::Text(first)) = choice.first() else {
        panic!("expected first text item with metadata")
    };
    let first_params = choice[0]
        .native_item()
        .expect("decoded first text metadata");
    assert_eq!(first.text, "first");
    assert_eq!(
        first_params["citations"]
            .as_array()
            .expect("citations should be an array")
            .len(),
        2
    );
    let Some(AssistantContent::Text(second)) = choice.get(1) else {
        panic!("expected second text item with metadata")
    };
    let second_params = choice[1]
        .native_item()
        .expect("decoded second text metadata");
    assert_eq!(second.text, "second");
    assert_eq!(second_params["block"], 2);
}

#[tokio::test]
async fn captured_response_headers_round_trip_and_are_omitted_when_empty() {
    let headers: ProviderResponseHeaders = [
        ("x-codex-credits-balance".to_owned(), "10, 20".to_owned()),
        ("x-codex-primary-used-percent".to_owned(), "12.5".to_owned()),
    ]
    .into_iter()
    .collect();
    let mut stream = captured(
        vec![MockStreamEvent::final_response_with_default_usage()],
        Some(json!({})),
        headers.clone(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("real decoded response");
    let encoded = serde_json::to_value(&response).expect("the response serializes");
    assert_eq!(
        encoded["provider_response_headers"],
        json!({"x-codex-credits-balance":"10, 20","x-codex-primary-used-percent":"12.5"})
    );
    let back: CompletionResponse =
        serde_json::from_value(encoded).expect("the response deserializes");
    assert_eq!(back.provider_response_headers, headers);
    let mut stream = captured(
        vec![MockStreamEvent::final_response_with_default_usage()],
        Some(json!({})),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let bare = stream.finish().await.expect("real decoded response");
    let encoded = serde_json::to_value(bare).expect("the response serializes");
    assert!(
        encoded.get("provider_response_headers").is_none(),
        "{encoded}"
    );
}

fn u_stream_of(events: Vec<MockStreamEvent>) -> CompletionStream {
    MockCompletionModel::from_stream_turns([events])
        .stream("hi")
        .expect("the stream opens")
}

/// Every item of `stream`, read to its end.
async fn items_of(stream: &mut CompletionStream) -> Vec<Result<Item<StreamEvent>, ProviderError>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}

#[tokio::test]
async fn an_error_is_the_last_item_and_finish_returns_it() {
    let mut stream = u_stream_of(vec![
        MockStreamEvent::text("kept"),
        MockStreamEvent::error("aborted by the provider"),
        MockStreamEvent::text("never read"),
    ]);
    let items = items_of(&mut stream).await;
    let Some(Err(yielded)) = items.last() else {
        panic!("the error is the last item: {items:?}");
    };
    assert!(yielded.to_string().contains("aborted"), "{yielded}");
    let partial = stream.partial();
    assert_eq!(partial.text(), "kept", "the text taken so far stays");
    let finished = stream.finish().await.expect_err("the error stands");
    assert_eq!(finished.to_string(), yielded.to_string());
}

/// A reply the provider did not end is a failed turn by construction: its
/// message is never replayed, whatever it holds.
#[tokio::test]
async fn a_partial_turn_after_an_error_is_marked_failed() {
    use crate::message::{Message, StopReason};
    let mut stream = u_stream_of(vec![
        MockStreamEvent::text("answer"),
        MockStreamEvent::tool_call("call_1", "lookup", serde_json::json!({"q": 1})),
        MockStreamEvent::error("mid-stream failure"),
    ]);
    let _ = items_of(&mut stream).await;
    let Some(Message::Assistant(turn)) = stream.partial().message() else {
        panic!("the partial turn has content");
    };
    assert!(
        matches!(&turn.stop, Some(StopReason::Error(error)) if error.contains("mid-stream failure")),
        "{:?}",
        turn.stop
    );
}

// G63: current unsigned helper close, with the exact original received values.
#[tokio::test]
async fn unsigned_reasoning_restatement_keeps_the_delta_across_interleaved_output() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("rs_1", "partial "),
        MockStreamEvent::tool_call("call_1", "probe", json!({})),
        MockStreamEvent::reasoning("the full block").with_reasoning_id("rs_1"),
        total(2),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    let reasoning: Vec<_> = choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning.len(), 1, "one unsigned helper slot: {choice:?}");
    assert_eq!(reasoning[0].text, "partial ");
    assert_eq!(choice.len(), 2);
    assert!(matches!(&choice[0], AssistantContent::Reasoning(_)));
    assert!(matches!(&choice[1], AssistantContent::ToolCall(call)
        if call.id.provider().map(|id| id.as_str()) == Some("call_1")
            && call.function.name.as_str() == "probe" && call.function.arguments_value() == json!({})));
}

// G64: the current Mock close returns its typed refusal, never an empty End.
#[tokio::test]
async fn malformed_mock_end_refuses_after_the_prior_completed_call() {
    const RAW: &str = "{\"x\":";
    let mut stream = stream_of([
        MockStreamEvent::tool_call_name_delta("ok", "add"),
        MockStreamEvent::tool_call_arguments_delta("ok", "{\"x\":1}"),
        MockStreamEvent::tool_call_end("ok"),
        MockStreamEvent::tool_call_name_delta("truncated", "add"),
        MockStreamEvent::tool_call_arguments_delta("truncated", RAW),
        MockStreamEvent::tool_call_end("truncated"),
        total(1),
    ]);
    let mut completed = Vec::new();
    let error = loop {
        match stream.next().await.expect("typed error must arrive") {
            Err(error) => break error,
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            })) => completed.push(call),
            Ok(_) => {}
        }
    };
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].id.provider().map(|id| id.as_str()), Some("ok"));
    assert_eq!(completed[0].function.name.as_str(), "add");
    assert_eq!(completed[0].function.arguments_value(), json!({"x":1}));
    let report = error.report();
    assert_eq!(report.kind, crate::error::ErrorKind::Response);
    assert!(!report.retryable);
    let Some(crate::error::ErrorDetail::MalformedToolInput(input)) = &report.detail else {
        panic!("existing typed malformed detail")
    };
    assert_eq!(input.name, "add");
    assert_eq!(input.raw, RAW);
    assert_eq!(input.id.provider().map(|id| id.as_str()), Some("truncated"));
    let provider = input.provider.as_ref().expect("real received correlation");
    assert_eq!(provider.call_id.as_str(), "truncated");
    assert!(provider.item_id.is_none());
    let expected_parse_error = crate::json_utils::parse_tool_arguments(RAW)
        .expect_err("the exact raw input is malformed")
        .to_string();
    assert_eq!(input.error, expected_parse_error);
    assert_eq!(
        report.message,
        format!("tool call `add` arrived with malformed JSON input: {expected_parse_error}")
    );
    assert_eq!(
        stream.partial().choice.len(),
        1,
        "only the earlier completed call remains"
    );
    assert!(
        matches!(&stream.partial().choice[0], AssistantContent::ToolCall(call) if call.id == completed[0].id)
    );
    assert!(stream.next().await.is_none());
    assert_eq!(
        stream
            .finish()
            .await
            .expect_err("the typed failure remains terminal")
            .report(),
        report
    );
}

#[derive(Clone)]
struct FrameChannel {
    frames: Arc<
        Mutex<Option<futures::channel::mpsc::UnboundedReceiver<Result<MockFrame, ProviderError>>>>,
    >,
}

impl Transport<MockScript> for FrameChannel {
    fn send(&self, _: CompletionRequest, _: Exchange) -> Opening<MockFrame> {
        match lock(&self.frames).take() {
            Some(frames) => Opening::ready(Opened::new(frames)),
            None => Opening::failed(ProviderError::Provider(
                "fixture channel was already consumed".to_owned(),
            )),
        }
    }
}

fn channel_stream() -> (
    futures::channel::mpsc::UnboundedSender<Result<MockFrame, ProviderError>>,
    CompletionStream,
) {
    let (sender, receiver) = futures::channel::mpsc::unbounded();
    let model = Model::new(
        MockScript::new(TEST_PROVIDER),
        FrameChannel {
            frames: Arc::new(Mutex::new(Some(receiver))),
        },
    );
    (
        sender,
        model
            .stream("streaming consumer")
            .expect("actual channel stream opens"),
    )
}

fn colliding_fixture(explicit: bool) -> MockStreamEvent {
    if explicit {
        MockStreamEvent::tool_call("tool-0", "explicit", json!({})).with_call_id("tool-0")
    } else {
        // The existing mock's tool-0 spelling is its unnamed assembly marker.
        // The actual fold, never this test, issues the LocalCallId.
        MockStreamEvent::tool_call("tool-0", "generated", json!({}))
    }
}

async fn receive_call_without_lookahead(stream: &mut CompletionStream) -> crate::message::ToolCall {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream
                .next()
                .await
                .expect("event before EOF")
                .expect("valid event");
            let encoded = serde_json::to_string(&event).expect("serialize event");
            let decoded: Item<StreamEvent> =
                serde_json::from_str(&encoded).expect("deserialize event");
            assert_eq!(decoded, event);
            if let Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            }) = decoded
            {
                break call;
            }
        }
    })
    .await
    .expect("completed call must not wait for later frames")
}

fn assert_fixture_identity(call: &crate::message::ToolCall, explicit: bool) {
    if explicit {
        assert_eq!(call.id.provider().map(|id| id.as_str()), Some("tool-0"));
        assert_eq!(call.function.name.as_str(), "explicit");
    } else {
        assert!(
            call.id.is_local(),
            "the genuine fold issues an id for the unnamed fixture"
        );
        assert_eq!(call.function.name.as_str(), "generated");
    }
    assert_eq!(call.function.arguments_value(), json!({}));
}

// G65: the original ordinary Response leaf is fatal, without replacing it.
#[tokio::test]
async fn ordinary_response_probe_is_terminal_after_either_first_call() {
    use futures::FutureExt;
    for explicit_first in [false, true] {
        let (sender, mut stream) = channel_stream();
        sender
            .unbounded_send(Ok(MockFrame::Event(colliding_fixture(explicit_first))))
            .expect("send first received frame");
        let first = receive_call_without_lookahead(&mut stream).await;
        assert_fixture_identity(&first, explicit_first);
        assert_eq!(stream.partial().choice.len(), 1);
        assert!(
            stream.next().now_or_never().is_none(),
            "no extra provider frames exist"
        );
        sender
            .unbounded_send(Err(ProviderError::Response("recoverable probe".to_owned())))
            .expect("send original error");
        let Some(Some(Err(error))) = stream.next().now_or_never() else {
            panic!("error delivery must not wait for EOF")
        };
        assert!(
            matches!(&error, ProviderError::Response(message) if message == "recoverable probe")
        );
        let report = error.report();
        // Frames after the fatal leaf must never produce another canonical call.
        let _ = sender.unbounded_send(Ok(MockFrame::Event(colliding_fixture(!explicit_first))));
        assert!(stream.next().await.is_none());
        assert_eq!(stream.partial().choice.len(), 1);
        assert!(
            matches!(&stream.partial().choice[0], AssistantContent::ToolCall(call) if call.id == first.id)
        );
        assert_eq!(
            stream
                .finish()
                .await
                .expect_err("the original ordinary Response is fatal")
                .report(),
            report
        );
    }
}

// G65 clean control: both original orders remain independently observable.
async fn clean_colliding_call_order(explicit_first: bool) {
    use futures::FutureExt;
    let (sender, mut stream) = channel_stream();
    let mut calls = Vec::new();
    for (position, explicit) in [explicit_first, !explicit_first].into_iter().enumerate() {
        sender
            .unbounded_send(Ok(MockFrame::Event(colliding_fixture(explicit))))
            .expect("send received call");
        let call = receive_call_without_lookahead(&mut stream).await;
        assert_fixture_identity(&call, explicit);
        assert_eq!(
            stream.partial().choice.len(),
            position + 1,
            "partial snapshot retains each completed call"
        );
        assert!(
            stream.next().now_or_never().is_none(),
            "no extra provider frames exist"
        );
        calls.push(call);
    }
    sender
        .unbounded_send(Ok(MockFrame::Event(total(1))))
        .expect("send final");
    drop(sender);
    drain(&mut stream).await;
    let response = stream.finish().await.expect("clean genuine end");
    assert_eq!(response.choice.len(), 2);
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].id, calls[1].id);
    for (part, call) in response.choice.iter().zip(&calls) {
        assert!(matches!(part, AssistantContent::ToolCall(ended) if ended.id == call.id));
    }
}

#[tokio::test]
async fn clean_explicit_first_collision_keeps_both_id_kinds_without_lookahead() {
    clean_colliding_call_order(true).await;
}

#[tokio::test]
async fn clean_generated_first_collision_keeps_both_id_kinds_without_lookahead() {
    clean_colliding_call_order(false).await;
}

// G74: actual decoded response serde preserves the current contract.

#[tokio::test]
async fn deserializing_response_filters_empty_identifiers() {
    let mut stream = captured(
        vec![MockStreamEvent::final_response_with_default_usage()],
        Some(json!({})),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("actual decoded response");
    let mut encoded = serde_json::to_value(response).expect("serialize actual current response");
    encoded["message_id"] = json!("");
    encoded["origin"]["response_id"] = json!("");
    encoded["origin"]["response_model"] = json!("");
    let decoded: CompletionResponse =
        serde_json::from_value(encoded).expect("deserialize normalized response");
    assert_eq!(decoded.message_id, None);
    assert_eq!(decoded.response_id(), None);
    assert_eq!(decoded.model(), None);
}

#[tokio::test]
async fn response_raw_round_trips_through_serde_mirror() {
    let payload = json!({"usage":{"total_tokens":8},"provider_only":"kept"});
    let mut stream = captured(
        vec![MockStreamEvent::FinalResponse(Finish {
            message_id: crate::completion::ProviderMessageId::new("msg_123"),
            ..Finish::default()
        })],
        Some(payload.clone()),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("actual decoded response");
    let encoded = serde_json::to_value(&response).expect("serialize");
    assert_eq!(encoded["raw"], payload);
    let decoded: CompletionResponse = serde_json::from_value(encoded.clone()).expect("deserialize");
    assert_eq!(decoded.raw, payload);
    assert_eq!(decoded, response);
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );
    let mut without_raw = encoded;
    without_raw
        .as_object_mut()
        .expect("response object")
        .shift_remove("raw");
    let error = serde_json::from_value::<CompletionResponse>(without_raw)
        .expect_err("a response without raw is refused");
    assert!(error.to_string().contains("raw"), "{error}");
}

#[tokio::test]
async fn response_serde_round_trip_is_identity() {
    let mut stream = captured(
        vec![MockStreamEvent::FinalResponse(Finish {
            usage: Usage {
                input_tokens: Some(4),
                output_tokens: Some(6),
                total_tokens: Some(10),
                cached_input_tokens: Some(1),
                cache_creation_input_tokens: Some(2),
                tool_use_prompt_tokens: Some(3),
                reasoning_tokens: Some(4),
                cost: None,
            },
            reason: Some(FinishReason::Stop),
            message_id: crate::completion::ProviderMessageId::new("msg_123"),
            response_id: Some("resp_456".to_owned()),
            model: Some("provider-model-v2".to_owned()),
            ..Finish::default()
        })],
        Some(json!({})),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("actual decoded response");
    let encoded = serde_json::to_value(&response).expect("serialize response");
    let decoded: CompletionResponse = serde_json::from_value(encoded.clone()).expect("deserialize");
    assert_eq!(decoded, response);
    assert_eq!(
        serde_json::to_value(decoded).expect("re-serialize"),
        encoded
    );
}

// G73: the current relay carries the single genuine Done.
#[tokio::test]
async fn a_long_run_of_non_yielding_events_does_not_grow_the_stack() {
    let mut events = Vec::new();
    for n in 0..50_000u32 {
        events.push(MockStreamEvent::message_id(format!("msg_{n}")));
    }
    events.push(MockStreamEvent::text("done"));
    events.push(total(1));
    for _ in 0..50_000u32 {
        events.push(total(99));
    }
    let mut relay = stream_of(events).into_relay();
    let mut texts = Vec::new();
    let mut terminals = 0;
    let mut completed = None;
    while let Some(item) = relay.next().await {
        match item.expect("stream item") {
            Relayed::Item(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Relayed::Done(response) => {
                terminals += 1;
                completed = Some(response);
            }
            _ => {}
        }
    }
    assert_eq!(texts, vec!["done".to_owned()]);
    assert_eq!(
        terminals, 1,
        "the first terminal latched; duplicates never yield"
    );
    let response = completed.expect("one actual terminal response");
    assert_eq!(response.usage.total_tokens, Some(1));
    assert_eq!(
        response.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_49999")
    );
}

fn identity_stream(events: Vec<MockStreamEvent>, request_id: Option<&str>) -> CompletionStream {
    Model::new(
        MockScript::new(TEST_PROVIDER),
        CapturedFixture {
            events,
            document: None,
            headers: ProviderResponseHeaders::new(),
            delay: false,
            request_id: request_id.map(str::to_owned),
        },
    )
    .stream("identity fixture")
    .expect("actual identity reply opens")
}

// G73: Finish supplies genuine message/response identity; Opened supplies request identity.
#[tokio::test]
async fn stream_identity_falls_back_to_the_real_finish_and_opened_metadata() {
    let mut stream = identity_stream(
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::FinalResponse(Finish {
                usage: Usage::new().total_tokens(1),
                message_id: crate::completion::ProviderMessageId::new("msg_terminal"),
                response_id: Some("resp_1".to_owned()),
                ..Finish::default()
            }),
        ],
        Some("req_1"),
    );
    drain(&mut stream).await;
    assert_eq!(
        stream.partial().identity(),
        crate::completion::ResponseIdentity {
            message_id: crate::completion::ProviderMessageId::new("msg_terminal"),
            response_id: Some("resp_1".to_owned()),
            provider_request_id: Some("req_1".to_owned()),
        }
    );
}

#[tokio::test]
async fn stream_identity_prefers_an_explicit_message_id_event() {
    let mut stream = identity_stream(
        vec![
            MockStreamEvent::message_id("msg_event"),
            MockStreamEvent::text("done"),
            MockStreamEvent::FinalResponse(Finish {
                usage: Usage::new().total_tokens(1),
                message_id: crate::completion::ProviderMessageId::new("msg_terminal"),
                response_id: Some("resp_1".to_owned()),
                ..Finish::default()
            }),
        ],
        None,
    );
    drain(&mut stream).await;
    assert_eq!(
        stream.partial().identity(),
        crate::completion::ResponseIdentity {
            message_id: crate::completion::ProviderMessageId::new("msg_event"),
            response_id: Some("resp_1".to_owned()),
            provider_request_id: None,
        }
    );
}

#[tokio::test]
async fn finish_carries_the_actual_opened_request_id() {
    let mut stream = identity_stream(
        vec![
            MockStreamEvent::text("hi"),
            MockStreamEvent::FinalResponse(Finish {
                response_id: Some("resp_1".to_owned()),
                ..Finish::default()
            }),
        ],
        Some("req_transport_1"),
    );
    drain(&mut stream).await;
    let response = stream
        .finish()
        .await
        .expect("the stream produced a terminal record");
    assert_eq!(response.response_id(), Some("resp_1"));
    assert_eq!(
        response.provider_request_id.as_deref(),
        Some("req_transport_1")
    );
}

// G74: current response and Item serde retain normalized values and unknown evidence.
#[tokio::test]
async fn decoded_response_round_trips_and_unknown_item_fidelity_is_preserved() {
    let mut stream = captured(
        vec![MockStreamEvent::FinalResponse(Finish {
            usage: Usage {
                input_tokens: Some(4),
                output_tokens: Some(6),
                total_tokens: Some(10),
                cached_input_tokens: Some(1),
                cache_creation_input_tokens: Some(2),
                tool_use_prompt_tokens: Some(3),
                reasoning_tokens: Some(4),
                cost: None,
            },
            reason: Some(FinishReason::Other("future_reason".to_owned())),
            message_id: crate::completion::ProviderMessageId::new("msg_123"),
            model: Some("provider-model-v2".to_owned()),
            ..Finish::default()
        })],
        Some(json!({})),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("actual decoded response");
    let encoded = serde_json::to_value(&response).expect("serialize actual response");
    let decoded: CompletionResponse =
        serde_json::from_value(encoded).expect("deserialize actual response");
    assert_eq!(decoded, response);
    assert_eq!(
        decoded.finish_reason(),
        Some(FinishReason::Other("future_reason".to_owned()))
    );
    assert_eq!(
        decoded.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_123")
    );
    assert_eq!(decoded.model(), Some("provider-model-v2"));
    assert_eq!(
        decoded.usage,
        Usage {
            input_tokens: Some(4),
            output_tokens: Some(6),
            total_tokens: Some(10),
            cached_input_tokens: Some(1),
            cache_creation_input_tokens: Some(2),
            tool_use_prompt_tokens: Some(3),
            reasoning_tokens: Some(4),
            cost: None,
        }
    );
    let provider_item =
        json!({"provider_native_event":"future_terminal","usage":{"total_tokens":10}});
    let mut stream = stream_of([
        MockStreamEvent::unknown(provider_item.clone()),
        MockStreamEvent::final_response_with_default_usage(),
    ]);
    let unknown = stream
        .next()
        .await
        .expect("actual unknown item")
        .expect("the unknown is delivered");
    assert!(matches!(&unknown, Item::Unknown(value) if value.value() == &provider_item));
    let encoded = serde_json::to_value(&unknown).expect("serialize unknown item");
    assert_eq!(encoded["item"], json!("unknown"));
    assert_eq!(encoded["value"], provider_item);
    let decoded: Item<StreamEvent> =
        serde_json::from_value(encoded).expect("deserialize unknown item");
    assert_eq!(decoded, unknown);
    assert!(
        serde_json::from_value::<StreamEvent>(provider_item).is_err(),
        "an untagged provider payload is not a stream event"
    );
}

#[tokio::test]
async fn the_decoded_response_captures_the_providers_exact_raw_terminal() {
    let usage = Usage::new()
        .input_tokens(3)
        .output_tokens(5)
        .total_tokens(8);
    let raw = json!({"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8},"provider_only":"kept"});
    let mut stream = captured(
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(usage),
        ],
        Some(raw.clone()),
        ProviderResponseHeaders::new(),
        false,
    );
    drain(&mut stream).await;
    let response = stream.finish().await.expect("actual decoded response");
    assert_eq!(response.raw["provider_only"], "kept");
    assert_eq!(response.raw, raw);
    assert_eq!(response.usage.total_tokens, Some(8));
    assert_eq!(response.provider(), TEST_PROVIDER);
    assert_eq!(response.finish_reason(), None);
}

// Only positive provider-ID evidence is read from the actual decoded native item.
fn actual_reasoning_id(content: &AssistantContent) -> Option<&str> {
    match content {
        AssistantContent::Reasoning(_) => content.native_item()?.get("id")?.as_str(),
        AssistantContent::CarriedReasoning(reasoning) => reasoning.id.as_deref(),
        _ => None,
    }
}

#[tokio::test]
async fn full_reasoning_block_with_a_different_received_id_appends() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("rs_1", "first item deltas"),
        MockStreamEvent::reasoning("a different item").with_reasoning_id("rs_2"),
        total(2),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    let ids: Vec<_> = choice.iter().filter_map(actual_reasoning_id).collect();
    assert_eq!(ids, vec!["rs_1", "rs_2"]);
    assert_eq!(choice.len(), 2);
    assert!(
        matches!(&choice[0], AssistantContent::Reasoning(reasoning) if reasoning.text == "first item deltas")
    );
    assert!(
        matches!(&choice[1], AssistantContent::Reasoning(reasoning) if reasoning.text == "a different item")
    );
}

// G75: current complete closes publish End; the old silent-false predicate is lost.
#[tokio::test]
async fn current_reasoning_close_emits_the_completed_end() {
    for through_terminal in [false, true] {
        let mut events = vec![MockStreamEvent::reasoning_delta_with_id(
            "block-0",
            "unsigned thoughts",
        )];
        if !through_terminal {
            events
                .push(MockStreamEvent::reasoning("unsigned thoughts").with_reasoning_id("block-0"));
        }
        events.push(total(2));
        let mut stream = stream_of(events);
        let mut completed = Vec::new();
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) = item.expect("stream item")
            {
                completed.push(reasoning);
            }
        }
        assert_eq!(completed.len(), 1, "a current complete close publishes End");
        assert_eq!(completed[0].text, "unsigned thoughts");
    }
}

#[tokio::test]
async fn completed_unsigned_reasoning_restates_the_actual_delta_part() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("block-0", "unsigned thoughts"),
        MockStreamEvent::reasoning("unsigned thoughts").with_reasoning_id("block-0"),
        total(2),
    ]);
    let mut deltas = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            Item::Event(StreamEvent::Reasoning { part, .. }) => deltas.push(part),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) => completed.push((part, reasoning)),
            _ => {}
        }
    }
    let (part, reasoning) = completed.first().expect("one completed block");
    assert_eq!(
        Some(part),
        deltas.first(),
        "the completed block restates its actual delta Part"
    );
    assert_eq!(reasoning.text, "unsigned thoughts");
}

#[tokio::test]
async fn completed_signed_reasoning_keeps_part_and_received_id_distinct() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("rs_1", "signed thoughts"),
        signed("rs_1", "signed thoughts", "sig_1"),
        total(2),
    ]);
    let mut deltas = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            Item::Event(StreamEvent::Reasoning { part, .. }) => deltas.push(part),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(reasoning),
                ..
            }) => completed.push((part, reasoning)),
            _ => {}
        }
    }
    let (part, reasoning) = completed.first().expect("one completed block");
    assert_eq!(Some(part), deltas.first());
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "signed thoughts".to_owned(),
            signature: Some("sig_1".to_owned())
        }]
    );
}

#[tokio::test]
async fn late_signed_whole_restatement_keeps_the_original_delta_part() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("reasoning-0", "hidden thoughts"),
        MockStreamEvent::text("visible"),
        signed("reasoning-0", "hidden thoughts", "sig_late"),
        total(2),
    ]);
    let mut deltas = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            Item::Event(StreamEvent::Reasoning { part, .. }) => deltas.push(part),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(reasoning),
                ..
            }) => completed.push((part, reasoning)),
            _ => {}
        }
    }
    assert_eq!(completed.len(), 1, "one signed completion, no duplicate");
    let (part, reasoning) = &completed[0];
    assert_eq!(
        Some(part),
        deltas.first(),
        "the signed completion restates the actual delta Part"
    );
    assert!(reasoning.content.iter().any(|content| matches!(content, CarriedReasoningPart::Text { signature:Some(sig),.. } if sig == "sig_late")));
    assert_eq!(reasoning.display_text(), "hidden thoughts");
}

#[tokio::test]
async fn empty_signed_reasoning_sibling_opens_a_new_part_after_close() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("reasoning-0", "part one"),
        MockStreamEvent::reasoning("part one").with_reasoning_id("reasoning-0"),
        signed("reasoning-0", "", "sig2"),
        total(2),
    ]);
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End { part, content, .. }) = item.expect("stream item") {
            if matches!(
                content,
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_)
            ) {
                completed.push((part, content));
            }
        }
    }
    assert_eq!(completed.len(), 2, "two distinct parts complete");
    assert_ne!(completed[0].0, completed[1].0);
    assert!(
        matches!(&completed[0].1,AssistantContent::Reasoning(reasoning) if reasoning.text == "part one")
    );
    assert!(
        matches!(&completed[1].1,AssistantContent::CarriedReasoning(reasoning)
        if reasoning.content == vec![CarriedReasoningPart::Text { text:String::new(),signature:Some("sig2".to_owned()) }])
    );
    assert_eq!(
        stream.partial().choice,
        completed
            .into_iter()
            .map(|(_, content)| content)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn reused_reasoning_helper_key_opens_distinct_parts_after_end() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("reasoning-0", "first part"),
        MockStreamEvent::reasoning("first part").with_reasoning_id("reasoning-0"),
        MockStreamEvent::reasoning_delta_with_id("reasoning-0", "second part"),
        MockStreamEvent::reasoning("second part").with_reasoning_id("reasoning-0"),
        total(2),
    ]);
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            part,
            content: AssistantContent::Reasoning(reasoning),
            ..
        }) = item.expect("stream item")
        {
            completed.push((part, reasoning));
        }
    }
    assert_eq!(completed.len(), 2, "two parts under the reused helper key");
    assert_ne!(completed[0].0, completed[1].0);
    assert_eq!(completed[0].1.text, "first part");
    assert_eq!(completed[1].1.text, "second part");
    assert_eq!(
        stream.partial().choice.len(),
        2,
        "two distinct parts in the choice"
    );
}

#[tokio::test]
async fn whole_reasoning_keeps_distinct_parts_and_genuine_received_ids() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning("first").with_reasoning_id("rs_1"),
        MockStreamEvent::reasoning("second").with_reasoning_id("rs_2"),
        total(2),
    ]);
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End { part, content, .. }) = item.expect("stream item") {
            if let Some(id) = actual_reasoning_id(&content) {
                completed.push((part, id.to_owned()));
            }
        }
    }
    assert_eq!(completed.len(), 2);
    assert_ne!(completed[0].0, completed[1].0);
    assert_eq!(
        completed
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["rs_1", "rs_2"]
    );
}

// G76: the current helper treats reasoning-0 as unnamed, without clobbering rs_1.
#[tokio::test]
async fn unnamed_reasoning_helper_item_does_not_clobber_a_genuine_id_item() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning_delta_with_id("rs_1", "identified deltas"),
        MockStreamEvent::reasoning("anonymous block").with_reasoning_id("reasoning-0"),
        total(2),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 2);
    assert_eq!(actual_reasoning_id(&choice[0]), Some("rs_1"));
    assert!(
        matches!(&choice[0],AssistantContent::Reasoning(reasoning) if reasoning.text == "identified deltas")
    );
    assert!(
        matches!(&choice[1],AssistantContent::Reasoning(reasoning) if reasoning.text == "anonymous block")
    );
}

#[tokio::test]
async fn stream_aggregates_assistant_items_in_current_start_order() {
    let mut stream = stream_of([
        MockStreamEvent::reasoning("chain-of-thought").with_reasoning_id("rs_interleaved"),
        MockStreamEvent::text("final-text"),
        MockStreamEvent::tool_call("tool_1", "mock_tool", json!({"arg":1})),
        total(3),
    ]);
    drain(&mut stream).await;
    let choice = stream.partial().choice;
    assert_eq!(choice.len(), 3);
    assert_eq!(actual_reasoning_id(&choice[0]), Some("rs_interleaved"));
    assert!(
        matches!(&choice[0],AssistantContent::Reasoning(reasoning) if reasoning.text == "chain-of-thought")
    );
    assert!(matches!(&choice[1],AssistantContent::Text(text) if text.text == "final-text"));
    assert!(
        matches!(&choice[2],AssistantContent::ToolCall(call) if call.id.provider().map(|id|id.as_str()) == Some("tool_1"))
    );
    assert!(
        matches!(&choice[2],AssistantContent::ToolCall(call) if call.function.name.as_str() == "mock_tool" && call.function.arguments_value() == json!({"arg":1}))
    );
}
