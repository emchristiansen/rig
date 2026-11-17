//! A connection wrapper that keeps the frames it hands on.
//!
//! [`FrameTee`] wraps a websocket connection and keeps the most recent frames
//! it read, exactly as the backend delivered them, in a
//! [`RetainedFrames`] store. It works under Rig's own websocket session and
//! under this crate's.
//!
//! ```
//! use rig_codex::ws_client::FrameTee;
//! use std::num::NonZeroUsize;
//!
//! fn tee<C>(connection: C) -> FrameTee<C> {
//!     FrameTee::new(connection, NonZeroUsize::MIN)
//! }
//! # let _ = tee::<()>;
//! ```

use super::{CloseFrame, Frame, ReadyFrame, WebSocketConnection};
use crate::frame_tee::RetainedFrames;
use rig_http::http_client::Result;
use rig_http::wasm_compat::WasmBoxedFuture;
use std::num::NonZeroUsize;

/// A connection that keeps the most recent frames it read from `inner`.
///
/// Every frame a read returns is kept as it was delivered, control frames
/// included, before the caller of the read sees it. Frames written are not
/// kept. Reads, writes and closes are otherwise `inner`'s own.
#[derive(Debug)]
pub struct FrameTee<C> {
    inner: C,
    frames: RetainedFrames<Frame>,
}

impl<C> FrameTee<C> {
    /// Read through `inner`, keeping the `capacity` most recent frames.
    pub fn new(inner: C, capacity: NonZeroUsize) -> Self {
        Self {
            inner,
            frames: RetainedFrames::new(capacity),
        }
    }

    /// The kept frames. The handle stays valid after the tee is boxed into a
    /// session.
    #[must_use]
    pub fn frames(&self) -> RetainedFrames<Frame> {
        self.frames.clone()
    }

    /// The connection frames are read from.
    pub fn inner(&self) -> &C {
        &self.inner
    }
}

impl<C: rig_http::ws_client::WebSocketConnection> rig_http::ws_client::WebSocketConnection
    for FrameTee<C>
{
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        self.inner.send(frame)
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(async move {
            let frame = self.inner.recv().await?;
            if let Some(frame) = &frame {
                self.frames.keep(frame.clone());
            }
            Ok(frame)
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        self.inner.close(frame)
    }
}

impl<C: WebSocketConnection> WebSocketConnection for FrameTee<C> {
    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        Box::pin(async move {
            let ready = self.inner.recv_ready().await?;
            if let ReadyFrame::Frame(frame) = &ready {
                self.frames.keep(frame.clone());
            }
            Ok(ready)
        })
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests;
