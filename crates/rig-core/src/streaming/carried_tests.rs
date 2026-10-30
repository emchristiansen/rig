//! Public delivery and cancellation contracts over the branded completion driver.
//!
//! These fixtures exercise consumer polling, relaying and source lifetime. They
//! have no provider dialect or request bytes for a cassette to replay.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

use futures::task::{ArcWake, waker};
use futures::{Stream, StreamExt};
use serde_json::{Value, json};

use super::{CompletionStream, Item, PartKind, StreamEvent, Streamed};
use crate::completion::{Accepts, CompletionRequest, FinishReason, ReplayTarget, Usage};
use crate::driver::{Exchange, Model, Opened, Opening, Transport, lock};
use crate::error::{
    EncodeError, ErrorDetail, ErrorKind, ErrorReport, FrameEvidence, MalformedToolInput,
    ProviderError,
};
use crate::message::{Api, AssistantContent, CallId};
use crate::operation::{Block, Completion, Finish};
use crate::wire::document::{Reassemble, Serves};
use crate::wire::{
    Decoder, Descriptor, Flow, IncompleteEof, Mode, Out, Wire, WireEvent, WireFrame,
};

const MALFORMED_DELTA: &str = " \n{\"type\":\"delta\", \"future\":{\"exact\":\"kept\\t\"}}\t ";
const TEXT: &str = r#"{"type":"delta","text":"kept"}"#;
const UNREAD_TEXT: &str = r#"{"type":"delta","text":"unread"}"#;
const STOP: &str = r#"{"type":"stop","usage":7,"response_id":"reply_first"}"#;
const LATER_STOP: &str = r#"{"type":"stop","usage":99,"response_id":"reply_later"}"#;
const ABORTED_ERROR: &str =
    r#"{"type":"error","message":"upstream aborted the request; CorruptFrameError"}"#;
const QUEUED_TEXT: &str = r#"{"type":"batch","first":"kept","second":"queued","terminal":false}"#;
const QUEUED_END: &str = r#"{"type":"batch","first":"kept","second":"queued","terminal":true}"#;

#[derive(Clone, Copy, Debug)]
struct StreamWire;

impl ReplayTarget for StreamWire {
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> Api {
        Api::from_static("stream-delivery.chat")
    }

    fn provider(&self) -> &str {
        "stream-delivery"
    }

    fn model(&self) -> &str {
        "stream-delivery-model"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
}

impl Wire for StreamWire {
    type Op = Completion;
    type Payload = ();
    type Frame = WireFrame;
    type Decoder<'id> = StreamDecoder<'id>;
    type Reassembler = StreamDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider())
            .model(self.model())
            .replay(self)
    }

    fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<(), EncodeError> {
        Ok(())
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        StreamDecoder::default()
    }
}

#[derive(Clone, Copy)]
enum EofOutcome {
    More,
    Truncated,
}

#[derive(Clone, Copy)]
struct IncompleteWire {
    base: StreamWire,
    eof: EofOutcome,
}

impl IncompleteWire {
    fn new(eof: EofOutcome) -> Self {
        Self {
            base: StreamWire,
            eof,
        }
    }
}

impl Wire for IncompleteWire {
    type Op = Completion;
    type Payload = ();
    type Frame = WireFrame;
    type Decoder<'id> = IncompleteDecoder<'id>;
    type Reassembler = StreamDocument;

    fn describe(&self) -> Descriptor<'_> {
        self.base.describe()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<(), EncodeError> {
        self.base.encode(request, mode)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        IncompleteDecoder {
            base: self.base.decoder(),
            buffered: Vec::new(),
            eof: self.eof,
        }
    }
}

struct IncompleteDecoder<'id> {
    base: StreamDecoder<'id>,
    buffered: Vec<String>,
    eof: EofOutcome,
}

impl<'id> Decoder<'id, Completion> for IncompleteDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        self.base.classify(frame)
    }

    fn decode(
        &mut self,
        event: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            Frame::Delta { text } => {
                self.buffered.push(text);
                Ok(Flow::More)
            }
            event => {
                for text in self.buffered.drain(..) {
                    out.run(Block::Text, &text)?;
                }
                self.base.decode(event, out)
            }
        }
    }

    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        for text in self.buffered.drain(..) {
            out.run(Block::Text, &text)?;
        }
        match self.eof {
            EofOutcome::More => Ok(Flow::More),
            EofOutcome::Truncated => Err(ProviderError::Truncated),
        }
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        IncompleteEof::KeepPartial
    }
}

#[derive(Clone, Copy)]
struct DefaultIncompleteWire(IncompleteWire);

impl Wire for DefaultIncompleteWire {
    type Op = Completion;
    type Payload = ();
    type Frame = WireFrame;
    type Decoder<'id> = DefaultIncompleteDecoder<'id>;
    type Reassembler = StreamDocument;

    fn describe(&self) -> Descriptor<'_> {
        self.0.describe()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<(), EncodeError> {
        self.0.encode(request, mode)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DefaultIncompleteDecoder(self.0.decoder())
    }
}

struct DefaultIncompleteDecoder<'id>(IncompleteDecoder<'id>);

impl<'id> Decoder<'id, Completion> for DefaultIncompleteDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        self.0.classify(frame)
    }

    fn decode(&mut self, event: Frame, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.0.decode(event, out)
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        self.0.eof(out)
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    Delta {
        text: String,
    },
    Batch {
        first: String,
        second: String,
        terminal: bool,
    },
    Stop {
        usage: u64,
        response_id: String,
    },
    Error {
        message: String,
    },
}

#[derive(Default)]
struct StreamDecoder<'id> {
    brand: PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion> for StreamDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "delta" | "batch" | "stop" | "error")
        })
    }

    fn decode(
        &mut self,
        event: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            Frame::Delta { text } => {
                out.run(Block::Text, &text)?;
            }
            Frame::Batch {
                first,
                second,
                terminal,
            } => {
                out.run(Block::Text, &first)?;
                out.run(Block::Text, &second)?;
                if terminal {
                    return Self::end(out, 7, "reply_first".to_owned());
                }
            }
            Frame::Stop { usage, response_id } => return Self::end(out, usage, response_id),
            Frame::Error { message } => return Err(ProviderError::Provider(message)),
        }
        Ok(Flow::More)
    }
}

impl<'id> StreamDecoder<'id> {
    fn end(
        mut out: Out<'id, Completion>,
        output_tokens: u64,
        response_id: String,
    ) -> Result<Flow, ProviderError> {
        out.end_run()?;
        Ok(out.end(Finish {
            usage: Usage {
                output_tokens: Some(output_tokens),
                ..Usage::default()
            },
            reason: Some(FinishReason::Stop),
            response_id: Some(response_id),
            model: Some("stream-delivery-model".to_owned()),
            ..Finish::default()
        }))
    }
}

#[derive(Default)]
struct StreamDocument(Vec<Value>);

impl Serves<Completion> for StreamDocument {}

impl Reassemble<WireFrame> for StreamDocument {
    fn absorb(&mut self, frame: &WireFrame) {
        if let Ok(value) = serde_json::from_str(&frame.as_str()) {
            self.0.push(value);
        }
    }

    fn finish(self) -> Value {
        Value::Array(self.0)
    }
}

#[derive(Clone, Default)]
struct SourceState {
    polls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}

#[derive(Clone)]
struct Script {
    frames: Arc<Vec<Result<String, ProviderError>>>,
    state: SourceState,
}

impl Script {
    fn text(frames: &[&str]) -> Self {
        Self::new(frames.iter().map(|frame| Ok((*frame).to_owned())).collect())
    }

    fn new(frames: Vec<Result<String, ProviderError>>) -> Self {
        Self {
            frames: Arc::new(frames),
            state: SourceState::default(),
        }
    }
}

impl Transport<StreamWire> for Script {
    fn send(&self, _payload: (), _exchange: Exchange) -> Opening<WireFrame> {
        let script = self.clone();
        Opening::new(async move {
            Ok(Opened::new(FrameSource {
                frames: script.frames.iter().cloned().collect(),
                state: script.state,
            }))
        })
    }
}

impl Transport<IncompleteWire> for Script {
    fn send(&self, payload: (), exchange: Exchange) -> Opening<WireFrame> {
        <Self as Transport<StreamWire>>::send(self, payload, exchange)
    }
}

impl Transport<DefaultIncompleteWire> for Script {
    fn send(&self, payload: (), exchange: Exchange) -> Opening<WireFrame> {
        <Self as Transport<StreamWire>>::send(self, payload, exchange)
    }
}

struct FrameSource {
    frames: VecDeque<Result<String, ProviderError>>,
    state: SourceState,
}

impl Stream for FrameSource {
    type Item = Result<WireFrame, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.state.polls.fetch_add(1, Ordering::SeqCst);
        Poll::Ready(
            this.frames
                .pop_front()
                .map(|frame| frame.map(WireFrame::Text)),
        )
    }
}

impl Drop for FrameSource {
    fn drop(&mut self) {
        self.state.dropped.store(true, Ordering::SeqCst);
    }
}

fn stream_of(script: Script) -> CompletionStream {
    Model::new(StreamWire, script)
        .stream(CompletionRequest::new("test delivery"))
        .expect("the fixture request opens")
}

fn assert_corruption(report: &ErrorReport) {
    assert_eq!(report.kind, ErrorKind::Json);
    assert!(!report.retryable);
    let Some(ErrorDetail::CorruptFrame(detail)) = &report.detail else {
        panic!("the typed corrupt-frame detail must survive: {report:?}");
    };
    assert_eq!(detail.event_type.as_deref(), Some("delta"));
    assert_eq!(
        detail.evidence,
        FrameEvidence::Text(MALFORMED_DELTA.to_owned())
    );
    assert!(!detail.error.is_empty());
}

async fn take_text(stream: &mut CompletionStream, expected: &str) {
    assert!(matches!(
        stream.next().await,
        Some(Ok(Item::Event(StreamEvent::Start {
            kind: PartKind::Text,
            ..
        })))
    ));
    assert!(matches!(
        stream.next().await,
        Some(Ok(Item::Event(StreamEvent::Text { text, .. }))) if text == expected
    ));
}

#[tokio::test]
async fn a_corrupt_known_frame_keeps_consuming_to_the_first_genuine_end() {
    let script = Script::text(&[MALFORMED_DELTA, TEXT, STOP, UNREAD_TEXT, LATER_STOP]);
    let state = script.state.clone();
    let mut stream = stream_of(script);
    let error = stream
        .next()
        .await
        .expect("the corrupt frame is delivered")
        .expect_err("a known delta missing text is corrupt");
    assert!(matches!(&error, ProviderError::CorruptFrame(_)));
    let corrupt = error
        .corrupt_frame()
        .expect("typed received-frame evidence");
    assert_eq!(corrupt.frame(), Some(MALFORMED_DELTA));
    assert_eq!(corrupt.frame_bytes(), Some(MALFORMED_DELTA.as_bytes()));
    assert_corruption(&error.report());
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
    assert!(!state.dropped.load(Ordering::SeqCst));

    take_text(&mut stream, "kept").await;
    assert!(matches!(
        stream.next().await,
        Some(Ok(Item::Event(StreamEvent::End { content, .. })))
            if content == AssistantContent::text("kept")
    ));
    assert!(stream.next().await.is_none());
    assert_eq!(state.polls.load(Ordering::SeqCst), 3);
    assert!(state.dropped.load(Ordering::SeqCst));
    let partial = stream.partial();
    assert_eq!(partial.choice, [AssistantContent::text("kept")]);
    assert_eq!(partial.usage.output_tokens, Some(7));
    assert_eq!(partial.response_id(), Some("reply_first"));
    assert!(partial.error.is_none());
    assert!(partial.aborted.is_none());
    let response = stream
        .finish()
        .await
        .expect("the drained reply has a genuine end");
    assert_eq!(response, partial);
    assert_eq!(
        response.raw,
        json!([
            serde_json::from_str::<Value>(MALFORMED_DELTA)
                .expect("the malformed typed frame is JSON"),
            serde_json::from_str::<Value>(TEXT).expect("text frame"),
            serde_json::from_str::<Value>(STOP).expect("first terminal frame"),
        ])
    );
}

#[tokio::test]
async fn a_relay_keeps_corruption_detail_and_later_content_through_the_bus() {
    let source = stream_of(Script::text(&[MALFORMED_DELTA, TEXT, STOP]));
    let reports = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&reports);
    let bus = source.into_relay().map(move |item| {
        item.map_err(|report| {
            assert_corruption(&report);
            let bytes = serde_json::to_vec(&report).expect("serialize the actual relay error");
            let decoded: ErrorReport =
                serde_json::from_slice(&bytes).expect("read the relay error");
            assert_eq!(decoded, report);
            lock(&captured).push(report);
            decoded
        })
    });
    let mut stream = Streamed::relay("delivery-relay", Box::pin(bus));
    let error = stream
        .next()
        .await
        .expect("the relay surfaces the corruption")
        .expect_err("the raw frame is corrupt");
    assert!(matches!(&error, ProviderError::Relayed(_)));
    assert_corruption(&error.report());
    assert_eq!(lock(&reports).as_slice(), &[error.report()]);
    take_text(&mut stream, "kept").await;
    assert!(matches!(
        stream.next().await,
        Some(Ok(Item::Event(StreamEvent::End { content, .. })))
            if content == AssistantContent::text("kept")
    ));
    assert!(stream.next().await.is_none());
    let response = stream
        .finish()
        .await
        .expect("the relay carries the genuine response");
    assert_eq!(response.choice, [AssistantContent::text("kept")]);
    assert_eq!(response.origin.provider, "stream-delivery");
    assert_eq!(response.usage.output_tokens, Some(7));
    assert_eq!(response.response_id(), Some("reply_first"));
    assert!(response.error.is_none());
    assert!(response.aborted.is_none());
}

#[tokio::test]
async fn unread_finish_and_unary_call_fail_at_the_first_corruption() {
    for unary in [false, true] {
        let script = Script::text(&[MALFORMED_DELTA, TEXT, STOP]);
        let state = script.state.clone();
        let model = Model::new(StreamWire, script);
        let result = if unary {
            model.call(CompletionRequest::new("test delivery")).await
        } else {
            model
                .stream(CompletionRequest::new("test delivery"))
                .expect("the fixture stream opens")
                .finish()
                .await
        };
        let error = result.expect_err("finish refuses the first unread error");
        assert!(matches!(&error, ProviderError::CorruptFrame(_)));
        assert_corruption(&error.report());
        assert_eq!(state.polls.load(Ordering::SeqCst), 1, "unary={unary}");
        assert!(state.dropped.load(Ordering::SeqCst), "unary={unary}");
    }
}

#[tokio::test]
async fn an_ordinary_provider_error_mentioning_aborted_is_terminal() {
    let script = Script::text(&[TEXT, ABORTED_ERROR, UNREAD_TEXT, STOP]);
    let state = script.state.clone();
    let mut stream = stream_of(script);
    take_text(&mut stream, "kept").await;
    let error = stream
        .next()
        .await
        .expect("provider failure")
        .expect_err("terminal error");
    assert!(matches!(&error, ProviderError::Provider(message) if message.contains("aborted")));
    assert!(error.report().detail.is_none());
    assert!(stream.next().await.is_none());
    assert_eq!(state.polls.load(Ordering::SeqCst), 2);
    assert!(state.dropped.load(Ordering::SeqCst));
    let partial = stream.partial();
    assert_eq!(partial.text(), "kept");
    assert_eq!(partial.error.as_deref(), Some(error.to_string().as_str()));
    assert_eq!(partial.response_id(), None);
    assert_eq!(partial.usage, Usage::default());
    let finished = stream
        .finish()
        .await
        .expect_err("the terminal error remains latched");
    assert_eq!(finished.report(), error.report());
}

#[tokio::test]
async fn ordinary_json_retryable_and_other_detail_errors_do_not_enable_recovery() {
    let parser =
        serde_json::from_str::<Value>("{").expect_err("a parser error without frame custody");
    let report = ErrorReport::new(
        ErrorKind::Json,
        "aborted; CorruptFrameError: ordinary failure",
    )
    .with_retryable(true);
    let tool_error = ErrorReport::new(ErrorKind::Response, "malformed complete tool input")
        .with_detail(ErrorDetail::MalformedToolInput(MalformedToolInput {
            name: "lookup".to_owned(),
            id: CallId::from_wire("call_malformed"),
            provider: None,
            raw: "{".to_owned(),
            error: "incomplete JSON".to_owned(),
        }));
    for error in [
        ProviderError::Json(Arc::new(parser)),
        ProviderError::Relayed(Box::new(report)),
        ProviderError::Relayed(Box::new(tool_error)),
    ] {
        let expected = error.report();
        let script = Script::new(vec![
            Ok(TEXT.to_owned()),
            Err(error),
            Ok(UNREAD_TEXT.to_owned()),
            Ok(STOP.to_owned()),
        ]);
        let state = script.state.clone();
        let mut stream = stream_of(script);
        take_text(&mut stream, "kept").await;
        let failure = stream
            .next()
            .await
            .expect("ordinary error")
            .expect_err("terminal error");
        assert_eq!(failure.report(), expected);
        assert!(stream.next().await.is_none());
        assert_eq!(state.polls.load(Ordering::SeqCst), 2);
        assert!(state.dropped.load(Ordering::SeqCst));
        assert_eq!(stream.partial().text(), "kept");
        assert_eq!(
            stream.finish().await.expect_err("failure stays").report(),
            expected
        );
    }
}

#[tokio::test]
async fn cancel_drops_the_source_and_discards_items_the_consumer_never_took() {
    let script = Script::text(&[QUEUED_TEXT, UNREAD_TEXT, STOP]);
    let state = script.state.clone();
    let mut stream = stream_of(script);
    take_text(&mut stream, "kept").await;
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
    assert!(!state.dropped.load(Ordering::SeqCst));
    let delivered = stream.partial();
    assert_eq!(delivered.text(), "kept");
    stream.cancel();
    assert!(
        state.dropped.load(Ordering::SeqCst),
        "cancel drops before another poll"
    );
    for _ in 0..3 {
        assert!(stream.next().await.is_none());
        assert_eq!(stream.partial(), delivered);
    }
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
    assert_eq!(delivered.response_id(), None);
    assert_eq!(delivered.usage, Usage::default());
    assert!(delivered.finish_reason().is_none());
    assert!(delivered.error.is_none());
    assert!(delivered.aborted.is_some());
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[tokio::test]
async fn cancel_preserves_an_end_already_recorded_without_delivering_queued_content() {
    let script = Script::text(&[QUEUED_END, UNREAD_TEXT, LATER_STOP]);
    let state = script.state.clone();
    let mut stream = stream_of(script);
    take_text(&mut stream, "kept").await;
    let delivered = stream.partial();
    assert_eq!(delivered.text(), "kept");
    assert_eq!(delivered.response_id(), Some("reply_first"));
    assert_eq!(delivered.usage.output_tokens, Some(7));
    assert_eq!(delivered.finish_reason(), Some(FinishReason::Stop));
    assert!(delivered.error.is_none());
    assert!(delivered.aborted.is_none());
    assert!(!state.dropped.load(Ordering::SeqCst));
    stream.cancel();
    assert!(state.dropped.load(Ordering::SeqCst));
    assert!(stream.next().await.is_none());
    assert_eq!(stream.partial(), delivered);
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
}

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl ArcWake for WakeCount {
    fn wake_by_ref(state: &Arc<Self>) {
        state.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn pause_parks_without_busy_wakes_and_a_clone_resumes_then_cancel_unparks() {
    let script = Script::text(&[TEXT, STOP]);
    let state = script.state.clone();
    let mut stream = stream_of(script);
    let resume = stream.pause_control();
    let wakes = Arc::new(WakeCount::default());
    let waker = waker(Arc::clone(&wakes));
    let mut cx = Context::from_waker(&waker);
    stream.pause();
    assert!(resume.is_paused());
    for _ in 0..2 {
        assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
    }
    assert_eq!(
        wakes.0.load(Ordering::SeqCst),
        0,
        "pause must not busy-wake"
    );
    assert_eq!(
        state.polls.load(Ordering::SeqCst),
        0,
        "pause must not poll the source"
    );
    resume.resume();
    assert!(!stream.is_paused());
    assert_eq!(
        wakes.0.load(Ordering::SeqCst),
        1,
        "the clone wakes the parked consumer"
    );
    assert!(matches!(
        Pin::new(&mut stream).poll_next(&mut cx),
        Poll::Ready(Some(Ok(Item::Event(StreamEvent::Start {
            kind: PartKind::Text,
            ..
        }))))
    ));
    assert!(matches!(
        Pin::new(&mut stream).poll_next(&mut cx),
        Poll::Ready(Some(Ok(Item::Event(StreamEvent::Text { text, .. })))) if text == "kept"
    ));
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
    let delivered = stream.partial();
    stream.pause();
    assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    stream.cancel();
    assert!(!resume.is_paused(), "paused cancel must lift pause");
    assert!(
        wakes.0.load(Ordering::SeqCst) > 1,
        "cancel wakes the parked consumer"
    );
    assert!(state.dropped.load(Ordering::SeqCst));
    assert!(matches!(
        Pin::new(&mut stream).poll_next(&mut cx),
        Poll::Ready(None)
    ));
    assert_eq!(stream.partial(), delivered);
    assert_eq!(state.polls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn repeated_eof_polls_keep_the_drained_fold_and_terminal_unchanged() {
    let mut stream = stream_of(Script::text(&[TEXT, STOP]));
    while let Some(item) = stream.next().await {
        item.expect("a valid fixture stream");
    }
    let delivered = stream.partial();
    assert_eq!(delivered.choice, [AssistantContent::text("kept")]);
    assert_eq!(delivered.usage.output_tokens, Some(7));
    assert_eq!(delivered.response_id(), Some("reply_first"));
    for _ in 0..3 {
        assert!(
            stream.next().await.is_none(),
            "a drained stream stays drained"
        );
        assert_eq!(
            stream.partial(),
            delivered,
            "re-polling preserves the whole folded view"
        );
    }
    assert_eq!(
        stream.finish().await.expect("the genuine end remains"),
        delivered
    );
}

#[tokio::test]
async fn opted_in_clean_eof_delivers_buffered_text_without_an_error_or_end() {
    let script = Script::text(&[TEXT]);
    let state = script.state.clone();
    let mut stream = Model::new(IncompleteWire::new(EofOutcome::More), script)
        .stream(CompletionRequest::new("test delivery"))
        .expect("the fixture stream opens");

    take_text(&mut stream, "kept").await;
    assert_eq!(
        state.polls.load(Ordering::SeqCst),
        2,
        "the text is queued only when the source reaches EOF"
    );
    assert!(stream.next().await.is_none(), "EOF emits no error or End");
    assert!(state.dropped.load(Ordering::SeqCst));
    let partial = stream.partial();
    assert_eq!(partial.choice, [AssistantContent::text("kept")]);
    assert!(partial.error.is_none());
    assert!(partial.finish_reason().is_none());
    assert_eq!(partial.response_id(), None);
    assert_eq!(partial.message_id, None);
    assert_eq!(partial.usage, Usage::default());
    {
        let shared = lock(&stream.shared);
        assert!(shared.end.is_none(), "EOF supplies no provider-end proof");
        assert!(shared.response.is_none(), "EOF supplies no response");
        assert_eq!(
            shared.raw,
            Some(json!([
                serde_json::from_str::<Value>(TEXT).expect("the fixture text is JSON")
            ]))
        );
    }
    for _ in 0..3 {
        assert!(stream.next().await.is_none());
        assert_eq!(stream.partial(), partial);
    }
    assert_eq!(state.polls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[tokio::test]
async fn opted_in_clean_eof_is_still_truncated_for_a_unary_call() {
    let script = Script::text(&[TEXT]);
    let state = script.state.clone();
    let error = Model::new(IncompleteWire::new(EofOutcome::More), script)
        .call(CompletionRequest::new("test delivery"))
        .await
        .expect_err("a partial EOF cannot complete a unary reply");
    assert!(matches!(error, ProviderError::Truncated));
    assert_eq!(state.polls.load(Ordering::SeqCst), 2);
    assert!(state.dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn the_default_decoder_still_refuses_streaming_eof_without_an_end() {
    let immediate = Script::text(&[TEXT]);
    let immediate_state = immediate.state.clone();
    let buffered = Script::text(&[TEXT]);
    let buffered_state = buffered.state.clone();
    let buffered_stream = Model::new(
        DefaultIncompleteWire(IncompleteWire::new(EofOutcome::More)),
        buffered,
    )
    .stream(CompletionRequest::new("test delivery"))
    .expect("the fixture stream opens");
    for (state, mut stream) in [
        (immediate_state, stream_of(immediate)),
        (buffered_state, buffered_stream),
    ] {
        take_text(&mut stream, "kept").await;
        let error = stream
            .next()
            .await
            .expect("default EOF surfaces a failure")
            .expect_err("the default policy refuses an unfinished reply");
        assert!(matches!(error, ProviderError::Truncated));
        assert!(stream.next().await.is_none());
        assert_eq!(stream.partial().text(), "kept");
        assert!(stream.partial().error.is_some());
        assert!(lock(&stream.shared).end.is_none());
        assert!(lock(&stream.shared).response.is_none());
        assert_eq!(state.polls.load(Ordering::SeqCst), 2);
        assert!(state.dropped.load(Ordering::SeqCst));
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
    }
}

#[tokio::test]
async fn an_actual_eof_truncation_is_not_suppressed_by_the_partial_policy() {
    let script = Script::text(&[TEXT]);
    let state = script.state.clone();
    let mut stream = Model::new(IncompleteWire::new(EofOutcome::Truncated), script)
        .stream(CompletionRequest::new("test delivery"))
        .expect("the fixture stream opens");
    take_text(&mut stream, "kept").await;
    let error = stream
        .next()
        .await
        .expect("the actual EOF error is delivered")
        .expect_err("KeepPartial applies only to Ok(More)");
    assert!(matches!(error, ProviderError::Truncated));
    assert!(stream.next().await.is_none());
    assert_eq!(stream.partial().text(), "kept");
    assert!(stream.partial().error.is_some());
    assert!(lock(&stream.shared).end.is_none());
    assert!(lock(&stream.shared).response.is_none());
    assert_eq!(state.polls.load(Ordering::SeqCst), 2);
    assert!(state.dropped.load(Ordering::SeqCst));
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}
