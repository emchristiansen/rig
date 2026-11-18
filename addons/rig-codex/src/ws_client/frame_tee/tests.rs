use super::*;
use crate::error::{CorruptFrame, FrameEvidence};
use crate::responses_api::streaming::decode_frame;
use rig_http::http_client::Error;
use rig_http::ws_client::WebSocketConnection as _;
use std::collections::VecDeque;

/// A connection that hands out queued frames and records what it is sent.
#[derive(Default)]
struct Queue {
    inbound: VecDeque<Frame>,
    sent: Vec<Frame>,
    flushed: usize,
    closed: bool,
}

impl Queue {
    fn of<I: IntoIterator<Item = Frame>>(frames: I) -> Self {
        Self {
            inbound: frames.into_iter().collect(),
            ..Self::default()
        }
    }
}

impl rig_http::ws_client::WebSocketConnection for Queue {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        self.sent.push(frame);
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(std::future::ready(Ok(self.inbound.pop_front())))
    }

    fn close(&mut self, _frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        self.closed = true;
        Box::pin(std::future::ready(Ok(())))
    }
}

impl WebSocketConnection for Queue {
    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        let ready = match self.inbound.pop_front() {
            Some(frame) => ReadyFrame::Frame(frame),
            None => ReadyFrame::Empty,
        };
        Box::pin(std::future::ready(Ok(ready)))
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        self.flushed += 1;
        Box::pin(std::future::ready(Ok(())))
    }
}

fn capacity(frames: usize) -> NonZeroUsize {
    NonZeroUsize::new(frames).expect("a non-zero capacity")
}

fn text(frame: &str) -> Frame {
    Frame::Text(frame.to_owned())
}

/// Every frame a read returns is kept exactly as delivered, in arrival
/// order, whichever read returned it; the reader sees the same frames.
#[tokio::test]
async fn every_frame_read_is_kept_as_delivered_in_arrival_order() {
    let irregular = r#"{ "type" : "codex.rate_limits",  "seen":1 }"#;
    let frames = [
        text(irregular),
        Frame::Ping(bytes::Bytes::from_static(b"p")),
        Frame::Binary(bytes::Bytes::from_static(&[0xff, 0x00])),
        text("{}"),
    ];
    let mut tee = FrameTee::new(Queue::of(frames.clone()), capacity(8));
    let kept = tee.frames();
    assert_eq!(kept.capacity(), capacity(8));
    assert!(kept.last().is_none());

    assert_eq!(tee.recv().await.expect("reads"), Some(frames[0].clone()));
    assert_eq!(kept.last(), Some(frames[0].clone()));
    assert_eq!(
        tee.recv_ready().await.expect("reads"),
        ReadyFrame::Frame(frames[1].clone())
    );
    assert_eq!(tee.recv().await.expect("reads"), Some(frames[2].clone()));
    assert_eq!(
        tee.recv_ready().await.expect("reads"),
        ReadyFrame::Frame(frames[3].clone())
    );
    // Nothing more arrived: neither read keeps anything.
    assert_eq!(tee.recv_ready().await.expect("reads"), ReadyFrame::Empty);
    assert_eq!(tee.recv().await.expect("reads"), None);

    assert_eq!(kept.snapshot(), frames);
    assert_eq!(kept.take(), frames, "taking hands the same frames over");
    assert!(kept.snapshot().is_empty(), "and removes them from the tee");
}

/// The tee holds its capacity of most recent frames and no more.
#[tokio::test]
async fn the_oldest_frame_is_dropped_at_capacity() {
    let mut tee = FrameTee::new(Queue::of([text("1"), text("2"), text("3")]), capacity(2));
    let kept = tee.frames();
    for _ in 0..3 {
        tee.recv().await.expect("reads");
    }
    assert_eq!(kept.snapshot(), [text("2"), text("3")]);
}

/// Writes, flushes and closes pass through to the connection, and a frame
/// written is not kept.
#[tokio::test]
async fn writes_flushes_and_closes_pass_through_unkept() {
    let mut tee = FrameTee::new(Queue::default(), capacity(2));
    let kept = tee.frames();
    tee.send(text("out")).await.expect("writes");
    tee.flush().await.expect("flushes");
    tee.close(None).await.expect("closes");
    assert_eq!(tee.inner().sent, [text("out")]);
    assert_eq!(tee.inner().flushed, 1);
    assert!(tee.inner().closed);
    assert!(kept.snapshot().is_empty());
}

/// A tee under a boxed connection still answers through its handle, and a
/// backend without the idle capabilities still refuses them by name.
#[tokio::test]
async fn a_boxed_tee_keeps_frames_and_forwards_missing_capabilities() {
    struct Plain(Queue);

    impl rig_http::ws_client::WebSocketConnection for Plain {
        fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
            self.0.send(frame)
        }

        fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
            rig_http::ws_client::WebSocketConnection::recv(&mut self.0)
        }

        fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
            self.0.close(frame)
        }
    }

    impl WebSocketConnection for Plain {}

    let tee = FrameTee::new(Plain(Queue::of([text("only")])), capacity(1));
    let kept = tee.frames();
    let mut boxed: crate::ws_client::BoxedWebSocketConnection = Box::new(tee);
    assert_eq!(boxed.recv().await.expect("reads"), Some(text("only")));
    assert_eq!(kept.last(), Some(text("only")));
    let refused: Error = boxed
        .recv_ready()
        .await
        .expect_err("the backend has no ready read");
    assert!(crate::ws_client::UnsupportedCapability::is(
        &refused,
        "recv_ready"
    ));
    let refused = boxed.flush().await.expect_err("the backend has no flush");
    assert!(crate::ws_client::UnsupportedCapability::is(
        &refused, "flush"
    ));
}

/// The use the tee exists for: a decoder reports only its decode error, and
/// the kept frame supplies the frame as received.
#[tokio::test]
async fn a_kept_frame_pairs_a_decode_failure_with_the_frame_as_received() {
    let malformed = r#"{"type" : "response.output_text.delta",  "delta": 42}"#;
    let mut tee = FrameTee::new(Queue::of([text(malformed)]), capacity(1));
    let kept = tee.frames();

    let Some(Frame::Text(received)) = tee.recv().await.expect("reads") else {
        panic!("a text frame arrives");
    };
    // What a decoder that kept nothing would report.
    let decode_error = decode_frame(&received)
        .expect_err("the frame is malformed")
        .without_frame();
    assert_eq!(decode_error.frame(), None);

    let evidence = kept
        .last()
        .as_ref()
        .and_then(FrameEvidence::of_frame)
        .expect("the frame was kept");
    assert_eq!(evidence, FrameEvidence::Text(malformed.to_owned()));
    let FrameEvidence::Text(frame) = evidence else {
        panic!("a text frame is text evidence");
    };
    let corrupt = CorruptFrame::text(
        decode_error.event_type().map(str::to_owned),
        frame,
        serde_json::from_str::<serde_json::Value>("{").expect_err("an error to carry"),
    );
    assert_eq!(corrupt.frame(), Some(malformed));
    assert_eq!(corrupt.event_type(), Some("response.output_text.delta"));
}
