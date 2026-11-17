//! An HTTP client wrapper that keeps the frames of the replies it hands on.
//!
//! Rig's driver cuts a reply's bytes into frames and gives each to its
//! decoder, which reports a frame it cannot decode by its decode error alone.
//! [`HttpFrameTee`] wraps the client the driver sends through and cuts the
//! same bytes with the same framer, keeping the most recent frames in a
//! [`RetainedFrames`] store: the data of each server-sent event of a streamed
//! reply, or a whole unary body.
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

use crate::frame_tee::RetainedFrames;
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
/// [`Framing`]: a streamed reply chunk by chunk, a unary body as a whole.
/// The bytes handed on are `inner`'s own, unchanged. Requests are not kept.
#[derive(Clone, Debug)]
pub struct HttpFrameTee<H> {
    inner: H,
    framing: Framing,
    frames: RetainedFrames<WireFrame>,
}

impl<H> HttpFrameTee<H> {
    /// Send through `inner`, keeping the `capacity` most recent frames of its
    /// replies, cut by `framing`: the framing of the wire whose requests go
    /// through this client ([`Framing::Sse`] for a Responses wire).
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
    pub fn frames(&self) -> RetainedFrames<WireFrame> {
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
                for frame in framing.split(&bytes) {
                    frames.keep(frame);
                }
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
                cut: Cut::of(framing),
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

/// A streamed body that keeps each frame its chunks complete.
struct TeedBody {
    body: BoxedStream,
    cut: Cut,
    frames: RetainedFrames<WireFrame>,
}

impl Stream for TeedBody {
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let next = this.body.as_mut().poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(chunk))) => match &mut this.cut {
                Cut::Framed(framer) => {
                    for frame in framer.push(chunk) {
                        this.frames.keep(frame);
                    }
                }
                Cut::Whole(body) => body.extend_from_slice(chunk),
                Cut::Ended => {}
            },
            Poll::Ready(None) => match std::mem::replace(&mut this.cut, Cut::Ended) {
                Cut::Framed(mut framer) => {
                    if let Some(frame) = framer.finish() {
                        this.frames.keep(frame);
                    }
                }
                Cut::Whole(body) if !body.is_empty() => {
                    this.frames.keep(WireFrame::from_bytes(body));
                }
                Cut::Whole(_) | Cut::Ended => {}
            },
            // A transport error ends nothing the framer has not already cut.
            Poll::Ready(Some(Err(_))) | Poll::Pending => {}
        }
        next
    }
}

#[cfg(test)]
mod tests;
