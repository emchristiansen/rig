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
/// Holds at most the tee's capacity: when another frame would exceed it, the
/// oldest kept frame is dropped.
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
    /// An empty store of at most `capacity` frames.
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            frames: Arc::default(),
        }
    }

    /// The most frames kept at once.
    #[must_use]
    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    /// The kept frames in arrival order, removed from the tee.
    #[must_use]
    pub fn take(&self) -> Vec<F> {
        self.lock().drain(..).collect()
    }

    /// Keep `frame` as the most recent, dropping the oldest at capacity.
    pub(crate) fn keep(&self, frame: F) {
        let mut frames = self.lock();
        if frames.len() >= self.capacity.get() {
            frames.pop_front();
        }
        frames.push_back(frame);
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

/// The evidence an HTTP reply's frame is of itself: its text or its bytes,
/// exactly as Rig's framer cut it.
impl From<&WireFrame> for FrameEvidence {
    fn from(frame: &WireFrame) -> Self {
        match frame {
            WireFrame::Text(text) => Self::Text(text.clone()),
            WireFrame::Bytes(bytes) => Self::Bytes(bytes.clone()),
        }
    }
}

#[cfg(test)]
mod tests;
