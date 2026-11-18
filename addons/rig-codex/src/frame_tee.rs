//! The frames a transport wrapper kept while handing them on.
//!
//! Rig's decoders report a frame they cannot decode by its decode error, and
//! the frame itself is gone by then. A tee keeps the most recent frames it
//! handed to Rig, exactly as the transport delivered them, so the caller can
//! pair a decode failure with the frame that caused it.
//! [`FrameTee`](crate::ws_client::FrameTee) wraps a websocket connection and
//! [`HttpFrameTee`](crate::http_client::HttpFrameTee) an HTTP client; both
//! answer through a [`RetainedFrames`] handle.
//!
//! ```
//! use rig_codex::error::FrameEvidence;
//! use rig_codex::frame_tee::RetainedFrames;
//! use rig_codex::ws_client::Frame;
//!
//! fn evidence(frames: &RetainedFrames<Frame>) -> Option<FrameEvidence> {
//!     frames.last().as_ref().and_then(FrameEvidence::of_frame)
//! }
//! # let _ = evidence;
//! ```

use crate::error::FrameEvidence;
use crate::ws_client::Frame;
use rig_http::http_client::framing::WireFrame;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// The frames a tee has kept, shared with the tee.
///
/// Of frames read one at a time, a websocket's or a streamed HTTP reply's,
/// it holds at most the tee's capacity: when another frame would exceed it,
/// the oldest kept frame is dropped.
///
/// A reply delivered whole is the one exception. A unary HTTP body reaches
/// Rig all at once, and nothing says which of its frames a later failure is
/// about, so [`HttpFrameTee`](crate::http_client::HttpFrameTee) keeps every
/// frame of the body, however many, and the store holds them until
/// [`take`](Self::take) or the next frame kept, which brings it back within
/// its capacity. The capacity therefore bounds the store for a caller that
/// streams its replies, and does not for one that reads a reply whole.
#[derive(Debug)]
pub struct RetainedFrames<F> {
    capacity: NonZeroUsize,
    frames: Arc<Mutex<VecDeque<F>>>,
}

impl<F> Clone for RetainedFrames<F> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            frames: Arc::clone(&self.frames),
        }
    }
}

impl<F> RetainedFrames<F> {
    /// An empty store whose capacity is `capacity` frames.
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            frames: Arc::default(),
        }
    }

    /// The most frames kept at once of those read one at a time. The frames
    /// of a reply delivered whole are kept together beyond it; see the type.
    #[must_use]
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// The kept frames in arrival order, removed from the tee.
    #[must_use]
    pub fn take(&self) -> Vec<F> {
        self.lock().drain(..).collect()
    }

    /// Keep `frame` as the most recent, dropping the oldest frames down to
    /// capacity: one at capacity, and after a reply kept whole, all that it
    /// held beyond it, with the room they took.
    pub(crate) fn keep(&self, frame: F) {
        let mut frames = self.lock();
        let room = self.capacity.get() - 1;
        let excess = frames.len().saturating_sub(room);
        frames.drain(..excess);
        if excess > 1 {
            frames.shrink_to(self.capacity.get());
        }
        frames.push_back(frame);
    }

    /// Keep `body`, every frame of a reply delivered whole, as the most
    /// recent. Older frames are dropped down to capacity; no frame of `body`
    /// is, whatever the capacity.
    pub(crate) fn keep_whole(&self, body: impl IntoIterator<Item = F>) {
        let body: Vec<F> = body.into_iter().collect();
        let mut frames = self.lock();
        let older = self.capacity.get().saturating_sub(body.len());
        let excess = frames.len().saturating_sub(older);
        frames.drain(..excess);
        frames.extend(body);
    }

    /// The frame queue. A panic in another holder leaves the queue as that
    /// holder left it, which is still a queue of whole frames.
    fn lock(&self) -> MutexGuard<'_, VecDeque<F>> {
        self.frames.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<F: Clone> RetainedFrames<F> {
    /// The most recent frame, when one is kept.
    #[must_use]
    pub fn last(&self) -> Option<F> {
        self.lock().back().cloned()
    }

    /// The kept frames in arrival order, left in place.
    #[must_use]
    pub fn snapshot(&self) -> Vec<F> {
        self.lock().iter().cloned().collect()
    }
}

/// The length of the first line of `bytes` with its line end: through a
/// line feed, a carriage return and the line feed after it, or a carriage
/// return alone. All of `bytes` when it holds no line end.
///
/// Rig's incremental framers complete a frame only at a line end, so a
/// wrapper that feeds a chunk to one line by line finds the byte where each
/// frame ends, and can hand the chunk on a frame at a time.
pub(crate) fn first_line(bytes: &[u8]) -> usize {
    match bytes.iter().position(|byte| matches!(byte, b'\n' | b'\r')) {
        Some(at) if bytes.get(at) == Some(&b'\r') && bytes.get(at + 1) == Some(&b'\n') => at + 2,
        Some(at) => at + 1,
        None => bytes.len(),
    }
}

impl FrameEvidence {
    /// The evidence a websocket frame is of itself: its text or its bytes,
    /// exactly as received. `None` for a ping, a pong or a close, which carry
    /// no protocol payload.
    #[must_use]
    pub fn of_frame(frame: &Frame) -> Option<Self> {
        match frame {
            Frame::Text(text) => Some(Self::Text(text.clone())),
            Frame::Binary(bytes) => Some(match std::str::from_utf8(bytes) {
                Ok(text) => Self::Text(text.to_owned()),
                Err(_) => Self::Bytes(bytes.to_vec()),
            }),
            Frame::Ping(_) | Frame::Pong(_) | Frame::Close(_) => None,
        }
    }
}

impl FrameEvidence {
    /// The evidence an HTTP reply's frame is of itself, as Rig's framer cut
    /// it. `exact` says whether every byte of the reply up to and including
    /// the chunk that completed the frame was valid UTF-8: Rig's framer
    /// reads text with each invalid sequence replaced, so text cut from a
    /// reply that was not is [`Self::Decoded`], never [`Self::Text`].
    pub(crate) fn of_wire_frame(frame: WireFrame, exact: bool) -> Self {
        match frame {
            WireFrame::Text(text) if exact => Self::Text(text),
            WireFrame::Text(text) => Self::Decoded(text),
            WireFrame::Bytes(bytes) => Self::Bytes(bytes),
        }
    }
}

#[cfg(test)]
mod tests;
