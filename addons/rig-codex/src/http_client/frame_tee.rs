//! An HTTP client wrapper that keeps the frames of the replies it hands on.
//!
//! Rig's driver cuts a reply's bytes into frames and gives each to its
//! decoder, which reports a frame it cannot decode by its decode error alone.
//! [`HttpFrameTee`] wraps the client the driver sends through and cuts the
//! same bytes with the same framer, keeping the most recent frames in a
//! [`RetainedFrames`] store: the data of each server-sent event of a streamed
//! reply, or a whole unary body.
//!
//! A kept frame says how faithful it is ([`FrameEvidence`]). Rig's framer
//! reads text with each invalid UTF-8 sequence replaced, so a text frame is
//! [`FrameEvidence::Text`], the event's data as the provider sent it, only
//! while every byte of the reply so far was valid UTF-8. A text frame cut
//! from the chunk holding the reply's first invalid byte, or from any chunk
//! after it, is [`FrameEvidence::Decoded`]. Name a failure with
//! [`decode_frame`](crate::responses_api::streaming::decode_frame) on a
//! `Text` frame only.
//!
//! A streamed reply is handed on in pieces that each complete at most one
//! frame, so the driver has decoded a frame before the next is cut. The
//! store then never holds more than its capacity of that reply's frames, and
//! when Rig's decoder refuses a frame, that frame is the newest one kept
//! ([`RetainedFrames::last`]), at any capacity.
//!
//! A unary reply arrives whole. Its frames are cut together, nothing says
//! which of them a failure is about, and so all of them are kept, however
//! many and whatever the capacity, until they are taken or the next reply's
//! first frame is kept. The capacity bounds what is kept of streamed
//! replies only; a caller that reads replies whole keeps each reply's
//! frames whole.
//!
//! ```
//! use rig_codex::http_client::HttpFrameTee;
//! use rig_http::http_client::framing::Framing;
//! use std::num::NonZeroUsize;
//!
//! fn tee<H>(http: H) -> HttpFrameTee<H> {
//!     HttpFrameTee::new(http, Framing::Sse, NonZeroUsize::MIN)
//! }
//! # let _ = tee::<()>;
//! ```

use crate::error::FrameEvidence;
use crate::frame_tee::{RetainedFrames, first_line};
use bytes::Bytes;
use futures::Stream;
use rig_http::http_client::framing::{Framer, Framing, WireFrame};
use rig_http::http_client::{
    BoxedStream, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::task::{Context, Poll};

/// An [`HttpClientExt`] that keeps the most recent frames of the replies
/// `inner` returns.
///
/// A reply is cut as Rig's driver cuts it for a wire with the same
/// [`Framing`]: a streamed reply as its bytes arrive, a unary body as a
/// whole. The bytes handed on are `inner`'s own, unchanged and in order; a
/// streamed chunk that completes more than one frame is handed on as
/// several chunks, one per frame. Requests are not kept.
///
/// A streamed reply's frames are kept one at a time, so the store never
/// holds more than its capacity of them, and when Rig's decoder refuses a
/// frame, that frame is the newest one kept ([`RetainedFrames::last`]), at
/// any capacity. A unary reply arrives whole: nothing says which of its
/// frames a failure is about, so all of them are kept, however many and
/// whatever the capacity, until they are taken or the next reply's first
/// frame is kept.
#[derive(Clone, Debug)]
pub struct HttpFrameTee<H> {
    inner: H,
    framing: Framing,
    frames: RetainedFrames<FrameEvidence>,
}

impl<H> HttpFrameTee<H> {
    /// Send through `inner`, keeping the frames of its replies, cut by
    /// `framing`: the framing of the wire whose requests go through this
    /// client ([`Framing::Sse`] for a Responses wire). Of a streamed reply
    /// the `capacity` most recent frames are kept; of a unary reply, every
    /// frame.
    pub fn new(inner: H, framing: Framing, capacity: NonZeroUsize) -> Self {
        Self {
            inner,
            framing,
            frames: RetainedFrames::new(capacity),
        }
    }

    /// The kept frames. The handle stays valid after the client is moved
    /// into a model.
    #[must_use]
    pub fn frames(&self) -> RetainedFrames<FrameEvidence> {
        self.frames.clone()
    }

    /// The transport requests are sent through.
    pub fn inner(&self) -> &H {
        &self.inner
    }
}

impl<H: HttpClientExt> HttpClientExt for HttpFrameTee<H> {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let sending = self.inner.send::<T, Bytes>(req);
        let frames = self.frames.clone();
        let framing = self.framing;
        async move {
            let (parts, body) = sending.await?.into_parts();
            let body: LazyBody<U> = Box::pin(async move {
                let bytes = body.await?;
                let exact = std::str::from_utf8(&bytes).is_ok();
                frames.keep_whole(
                    framing
                        .split(&bytes)
                        .into_iter()
                        .map(|frame| FrameEvidence::of_wire_frame(frame, exact)),
                );
                Ok(U::from(bytes))
            });
            Ok(Response::from_parts(parts, body))
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let sending = self.inner.send_streaming(req);
        let frames = self.frames.clone();
        let framing = self.framing;
        async move {
            let (parts, body) = sending.await?.into_parts();
            let body: BoxedStream = Box::pin(TeedBody {
                body,
                pending: Bytes::new(),
                cut: Cut::of(framing),
                utf8: Utf8Watch::default(),
                frames,
            });
            Ok(Response::from_parts(parts, body))
        }
    }
}

/// How a streamed body's chunks become frames.
enum Cut {
    /// Frame by frame, as the chunks arrive.
    Framed(Framer),
    /// The whole body is one frame, known when the stream ends.
    Whole(Vec<u8>),
    /// The stream ended and its last frame was kept.
    Ended,
}

impl Cut {
    fn of(framing: Framing) -> Self {
        match framing.framer() {
            Some(framer) => Self::Framed(framer),
            None => Self::Whole(Vec::new()),
        }
    }
}

/// Whether every byte of a body so far is valid UTF-8, across its chunks.
#[derive(Default)]
struct Utf8Watch {
    /// The bytes of a character the last chunk left unfinished.
    partial: Vec<u8>,
    /// An invalid sequence was seen. It stays seen: Rig's framer has read
    /// replacement text from there on, and nothing after is vouched for.
    invalid: bool,
}

impl Utf8Watch {
    fn feed(&mut self, chunk: &[u8]) {
        if self.invalid {
            return;
        }
        let mut bytes = std::mem::take(&mut self.partial);
        bytes.extend_from_slice(chunk);
        match std::str::from_utf8(&bytes) {
            Ok(_) => {}
            Err(error) if error.error_len().is_some() => self.invalid = true,
            Err(error) => bytes.drain(..error.valid_up_to()).for_each(drop),
        }
        if !self.invalid && std::str::from_utf8(&bytes).is_err() {
            self.partial = bytes;
        }
    }

    /// Whether the text cut so far is the body's own text.
    fn exact(&self) -> bool {
        !self.invalid
    }

    /// Whether the body, now ended, was valid UTF-8 to its last byte.
    fn exact_at_end(&self) -> bool {
        !self.invalid && self.partial.is_empty()
    }
}

/// A streamed body that keeps each frame its bytes complete, and hands the
/// bytes on in pieces that each complete at most one frame.
///
/// Rig's driver decodes every frame of a chunk before it asks for the next
/// chunk. So a frame is kept, and its piece handed on, only after the driver
/// has decoded every frame before it: when the decoder refuses a frame and
/// the driver stops reading, the newest kept frame is the refused one, and
/// the store never holds more than its capacity of this reply's frames.
struct TeedBody {
    body: BoxedStream,
    /// What is left of the transport's last chunk, not yet handed on.
    pending: Bytes,
    cut: Cut,
    utf8: Utf8Watch,
    frames: RetainedFrames<FrameEvidence>,
}

impl TeedBody {
    /// Take the next piece of the pending chunk: its bytes through the line
    /// that completes its first frame, or all of it when it completes none.
    /// The frame, when there is one, is kept.
    fn next_piece(&mut self) -> Bytes {
        let chunk = std::mem::take(&mut self.pending);
        let framer = match &mut self.cut {
            Cut::Framed(framer) => framer,
            Cut::Whole(body) => {
                body.extend_from_slice(&chunk);
                return chunk;
            }
            Cut::Ended => return chunk,
        };
        let exact = self.utf8.exact();
        let mut end = 0;
        // Both framers complete a frame only at a line end, so a frame's
        // last byte is found by feeding the chunk line by line.
        while let Some(rest) = chunk.get(end..).filter(|rest| !rest.is_empty()) {
            let line = rest.get(..first_line(rest)).unwrap_or(rest);
            end += line.len();
            let mut completed = false;
            for frame in framer.push(line) {
                self.frames.keep(FrameEvidence::of_wire_frame(frame, exact));
                completed = true;
            }
            if completed {
                break;
            }
        }
        self.pending = chunk.slice(end..);
        chunk.slice(..end)
    }
}

impl Stream for TeedBody {
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.pending.is_empty() {
            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    // Exactness is judged a transport chunk at a time: text
                    // cut from the chunk holding the reply's first invalid
                    // byte is not vouched for, wherever in it the byte is.
                    if matches!(this.cut, Cut::Framed(_)) {
                        this.utf8.feed(&chunk);
                    }
                    this.pending = chunk;
                }
                Poll::Ready(None) => {
                    match std::mem::replace(&mut this.cut, Cut::Ended) {
                        Cut::Framed(mut framer) => {
                            if let Some(frame) = framer.finish() {
                                let exact = this.utf8.exact_at_end();
                                this.frames.keep(FrameEvidence::of_wire_frame(frame, exact));
                            }
                        }
                        Cut::Whole(body) if !body.is_empty() => {
                            // A whole body is text only when all of it is UTF-8.
                            this.frames.keep(FrameEvidence::of_wire_frame(
                                WireFrame::from_bytes(body),
                                true,
                            ));
                        }
                        Cut::Whole(_) | Cut::Ended => {}
                    }
                    return Poll::Ready(None);
                }
                // A transport error ends nothing the framer has not already cut.
                other @ (Poll::Ready(Some(Err(_))) | Poll::Pending) => return other,
            }
        }
        Poll::Ready(Some(Ok(this.next_piece())))
    }
}

#[cfg(test)]
mod tests;
