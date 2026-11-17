use super::*;

fn capacity(frames: usize) -> NonZeroUsize {
    NonZeroUsize::new(frames).expect("a non-zero capacity")
}

/// The store holds its capacity of most recent frames, in arrival order, and
/// every handle sees the same frames.
#[test]
fn the_most_recent_frames_are_kept_in_arrival_order_up_to_capacity() {
    let frames = RetainedFrames::<u8>::new(capacity(2));
    let handle = frames.clone();
    assert_eq!(handle.capacity(), capacity(2));
    assert_eq!(handle.last(), None);
    for frame in [1, 2, 3] {
        frames.keep(frame);
    }
    assert_eq!(handle.snapshot(), [2, 3], "the oldest frame was dropped");
    assert_eq!(handle.last(), Some(3));
    assert_eq!(handle.take(), [2, 3], "taking hands the same frames over");
    assert!(frames.snapshot().is_empty(), "and removes them");
    frames.keep(4);
    assert_eq!(handle.snapshot(), [4]);
}

/// Evidence is the frame's text when it is text, its bytes when it is not,
/// and nothing for a websocket frame that carries no payload.
#[test]
fn a_frames_evidence_is_its_text_or_its_bytes() {
    assert_eq!(
        FrameEvidence::of_frame(&Frame::Text("{ }".to_owned())),
        Some(FrameEvidence::Text("{ }".to_owned()))
    );
    assert_eq!(
        FrameEvidence::of_frame(&Frame::Binary(bytes::Bytes::from_static(b"{ }"))),
        Some(FrameEvidence::Text("{ }".to_owned()))
    );
    assert_eq!(
        FrameEvidence::of_frame(&Frame::Binary(bytes::Bytes::from_static(&[0xff, 0x7b]))),
        Some(FrameEvidence::Bytes(vec![0xff, 0x7b]))
    );
    for empty in [
        Frame::Ping(bytes::Bytes::new()),
        Frame::Pong(bytes::Bytes::new()),
        Frame::Close(None),
    ] {
        assert_eq!(FrameEvidence::of_frame(&empty), None, "{empty:?}");
    }

    assert_eq!(
        FrameEvidence::of_wire_frame(WireFrame::Text("{ }".to_owned()), true),
        FrameEvidence::Text("{ }".to_owned())
    );
    assert_eq!(
        FrameEvidence::of_wire_frame(WireFrame::Text("{ \u{fffd} }".to_owned()), false),
        FrameEvidence::Decoded("{ \u{fffd} }".to_owned()),
        "text cut from a reply that was not all UTF-8 is not exact"
    );
    assert_eq!(
        FrameEvidence::of_wire_frame(WireFrame::Bytes(vec![0xff, 0x7b]), false),
        FrameEvidence::Bytes(vec![0xff, 0x7b])
    );
}
