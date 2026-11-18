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

/// A reply kept whole is held beyond the capacity, every frame of it, with
/// older frames dropped down to capacity; the next frame kept one at a time
/// brings the store back within its capacity.
#[test]
fn a_reply_kept_whole_exceeds_the_capacity_until_the_next_frame_is_kept() {
    let frames = RetainedFrames::<u32>::new(capacity(3));
    frames.keep(1);
    frames.keep(2);
    frames.keep_whole([10]);
    assert_eq!(
        frames.snapshot(),
        [1, 2, 10],
        "a whole reply smaller than the capacity leaves older frames their room"
    );
    frames.keep_whole(100..10_100);
    let held = frames.snapshot();
    assert_eq!(
        held.len(),
        10_000,
        "every frame of the reply, and no older one"
    );
    assert_eq!((held.first(), held.last()), (Some(&100), Some(&10_099)));
    assert_eq!(frames.capacity(), capacity(3));

    frames.keep(7);
    assert_eq!(
        frames.snapshot(),
        [10_098, 10_099, 7],
        "back within the capacity with the next frame"
    );

    let one = RetainedFrames::<u32>::new(capacity(1));
    one.keep_whole(0..10_000);
    one.keep(7);
    assert_eq!(one.snapshot(), [7]);
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
