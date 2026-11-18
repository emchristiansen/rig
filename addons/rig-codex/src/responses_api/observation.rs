//! Caller-owned observations of one Codex Responses operation.
//!
//! The ledger contains the original strings supplied by the SSE source, before
//! interpretation. It is an in-memory prefix, not a record of unread transport
//! bytes. Keeping a handle retains that prefix after the owning future is dropped.
//!
//! [`CodexResponses::raw_stream_observed`] opens one observed stream. The
//! request is sent and the reply read by Rig's own driver, through a
//! transport wrapper that commits the data of each server-sent event to the
//! ledger before Rig's decoder sees it. The stream's events and its final
//! response are Rig's; the ledger is the evidence they were read from.
//!
//! ```
//! use rig_codex::responses_api::observation::{Observation, ObservationStage};
//!
//! let (observation, writer) = Observation::prepare();
//! // A caller that refuses its own input before sending says so.
//! writer.reject_before_dispatch();
//! observation.inspect(|view| {
//!     assert_eq!(view.stage, ObservationStage::CallerRejectedBeforeDispatch);
//!     assert!(view.committed_prefix.is_empty());
//! });
//! ```

use crate::frame_tee::driver_streams;
use crate::http_client::CodexHttpClient;
use crate::responses_api::streaming::{
    DecodedFrame, ResponseChunkKind, StreamingCompletionChunk, decode_frame,
};
use crate::responses_api::wire::CodexResponses;
use bytes::Bytes;
use futures::Stream;
use rig_core::Model;
use rig_core::completion::CompletionRequest;
use rig_core::error::{ErrorReport, ProviderError};
use rig_core::streaming::{CompletionStream, Item, StreamEvent};
use rig_http::http_client::framing::{Framer, Framing, WireFrame};
use rig_http::http_client::{
    self, BoxedStream, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;
use serde_json::Value;
use std::collections::TryReserveError;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

/// How this operation acquired its evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationProvenance {
    /// Original strings supplied by the existing SSE source.
    SuppliedSse,
    /// A caller's explicitly synthetic operation; it supplies no wire evidence.
    Synthetic,
}

/// Native operation state, independent of whether the caller later accepts the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationStage {
    /// Prepared, with no send attempt yet.
    Prepared,
    /// The send attempt has started.
    Running,
    /// Source EOF or a retained terminal ended reading; finalization is pending.
    AwaitingNativeFinalization,
    /// Native finalization succeeded with healthy capture.
    NativeSucceeded,
    /// Native processing or capture failed.
    NativeFailed,
    /// The caller explicitly rejected its input before transfer to Rig.
    CallerRejectedBeforeDispatch,
    /// The whole owning future or stream was dropped before a terminal state.
    Cancelled,
}

impl ObservationStage {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::NativeSucceeded
                | Self::NativeFailed
                | Self::CallerRejectedBeforeDispatch
                | Self::Cancelled
        )
    }
}

/// A local send-attempt boundary; this does not certify provider receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationDispatch {
    /// The HTTP response future has not been polled.
    NotDispatched,
    /// The same HTTP response future was polled at least once.
    SendAttemptStarted,
}

/// An actual recoverable capture fault.
#[derive(Debug, thiserror::Error)]
pub enum CaptureCause {
    /// Reserving space for one whole supplied string failed.
    #[error("observation entry allocation failed: {0}")]
    Allocation(#[source] TryReserveError),
    /// A panic while the ledger was locked poisoned that lock.
    #[error("observation ledger lock is poisoned")]
    LockPoisoned,
}

/// Capture fault and the whole supplied string that could not be committed.
#[derive(Debug)]
pub struct CaptureFault {
    /// The real storage or locking failure.
    pub cause: CaptureCause,
    /// Original uncommitted data, if the failure happened at an append.
    pub offending_string: Option<String>,
}

#[derive(Debug)]
struct Ledger {
    provenance: ObservationProvenance,
    stage: ObservationStage,
    dispatch: ObservationDispatch,
    committed: Vec<String>,
    fault: Option<CaptureFault>,
    native_cause: Option<ExistingDiagnosticBasis>,
    native_error: Option<Arc<ProviderError>>,
    reply_head: Option<ObservedReplyHead>,
}

/// Whole HTTP head returned to the observed SSE source.
///
/// A refused reply's body is kept in the native error. The reqwest non-2xx
/// path reads it without a size bound. For `Ok(non-200)` replies, the driver
/// collects at most 1 MiB or 4096 chunks.
#[derive(Clone, Debug)]
pub struct ObservedReplyHead {
    /// Actual returned status.
    pub status: http::StatusCode,
    /// Whole headers, including repeated and non-UTF8 values.
    pub headers: http::HeaderMap,
    /// Whether the existing SSE response validation accepted this reply.
    pub accepted_sse: bool,
}

/// A consistent borrowed ledger view; no event is polled to obtain it.
#[derive(Debug)]
pub struct ObservationView<'a> {
    /// Provenance of this operation.
    pub provenance: ObservationProvenance,
    /// Native stage, separate from caller acceptance.
    pub stage: ObservationStage,
    /// Local transport-attempt status, even for an empty prefix.
    pub dispatch: ObservationDispatch,
    /// Ordered whole original supplied strings.
    pub committed_prefix: &'a [String],
    /// Capture is healthy exactly when this is `None`.
    pub capture_fault: Option<&'a CaptureFault>,
    /// Available native cause, without a recursive handle inside the ledger.
    pub native_cause: Option<&'a ExistingDiagnosticBasis>,
    /// The original typed error of a provider failure, including one Rig converted into an item report.
    pub native_error: Option<&'a ProviderError>,
    /// Actual HTTP head, when a response reached the SSE source.
    pub reply_head: Option<&'a ObservedReplyHead>,
}

/// Synchronous preparation of one ledger and its exclusive write capability.
pub struct Observation;

impl Observation {
    /// Prepare without validating input, opening HTTP or polling a future.
    pub fn prepare() -> (ObservationHandle, ObservationWriter) {
        Self::with_provenance(ObservationProvenance::SuppliedSse)
    }

    /// Prepare explicitly synthetic custody without fabricating SSE strings.
    pub fn prepare_synthetic() -> (ObservationHandle, ObservationWriter) {
        Self::with_provenance(ObservationProvenance::Synthetic)
    }

    fn with_provenance(
        provenance: ObservationProvenance,
    ) -> (ObservationHandle, ObservationWriter) {
        let handle = ObservationHandle(Arc::new(Mutex::new(Ledger {
            provenance,
            stage: ObservationStage::Prepared,
            dispatch: ObservationDispatch::NotDispatched,
            committed: Vec::new(),
            fault: None,
            native_cause: None,
            native_error: None,
            reply_head: None,
        })));
        (handle.clone(), ObservationWriter { handle })
    }
}

/// Shareable read custody of one operation, independent of its future's lifetime.
#[derive(Clone, Debug)]
pub struct ObservationHandle(Arc<Mutex<Ledger>>);

impl ObservationHandle {
    /// Inspect a consistent borrowed view synchronously.
    ///
    /// A visitor must not re-enter this same handle. A visitor panic poisons
    /// capture, which is visible to the next inspection or native operation.
    pub fn inspect<T>(&self, visitor: impl FnOnce(ObservationView<'_>) -> T) -> T {
        let ledger = self.lock();
        visitor(ObservationView {
            provenance: ledger.provenance,
            stage: ledger.stage,
            dispatch: ledger.dispatch,
            committed_prefix: &ledger.committed,
            capture_fault: ledger.fault.as_ref(),
            native_cause: ledger.native_cause.as_ref(),
            native_error: ledger.native_error.as_deref(),
            reply_head: ledger.reply_head.as_ref(),
        })
    }

    /// Compare actual ledger identity, rather than provider or generated IDs.
    pub fn same_operation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn lock(&self) -> MutexGuard<'_, Ledger> {
        match self.0.lock() {
            Ok(ledger) => ledger,
            Err(poison) => {
                let mut ledger = poison.into_inner();
                if ledger.fault.is_none() {
                    ledger.fault = Some(CaptureFault {
                        cause: CaptureCause::LockPoisoned,
                        offending_string: None,
                    });
                    if !ledger.stage.is_terminal() {
                        ledger.stage = ObservationStage::NativeFailed;
                    }
                }
                ledger
            }
        }
    }

    pub(crate) fn mark_send_attempt_started(&self) {
        let mut ledger = self.lock();
        ledger.dispatch = ObservationDispatch::SendAttemptStarted;
        if !ledger.stage.is_terminal() {
            ledger.stage = ObservationStage::Running;
        }
    }

    pub(crate) fn record_reply_head(
        &self,
        status: http::StatusCode,
        headers: &http::HeaderMap,
        accepted_sse: bool,
    ) {
        self.lock().reply_head = Some(ObservedReplyHead {
            status,
            headers: headers.clone(),
            accepted_sse,
        });
    }

    /// Commit one whole supplied string, or keep it as the string a capture
    /// fault could not commit.
    pub(crate) fn append(&self, original: String) -> Result<(), ()> {
        let mut ledger = self.lock();
        if ledger.fault.is_some() {
            if let Some(fault) = ledger.fault.as_mut()
                && fault.offending_string.is_none()
            {
                fault.offending_string = Some(original);
            }
            return Err(());
        }
        if let Err(cause) = ledger.committed.try_reserve(1) {
            ledger.fault = Some(CaptureFault {
                cause: CaptureCause::Allocation(cause),
                offending_string: Some(original),
            });
            ledger.stage = ObservationStage::NativeFailed;
            return Err(());
        }
        ledger.committed.push(original);
        Ok(())
    }

    fn has_capture_fault(&self) -> bool {
        self.lock().fault.is_some()
    }
}

/// The exclusive capability moved into one prepared future and then its stream.
///
/// Dropping this capability changes only the ledger; it performs no HTTP work,
/// draining, retry or background task.
#[derive(Debug)]
pub struct ObservationWriter {
    handle: ObservationHandle,
}

impl ObservationWriter {
    /// Obtain read custody of the same operation.
    pub fn handle(&self) -> ObservationHandle {
        self.handle.clone()
    }

    /// Record explicit caller rejection before handing this writer to Rig.
    pub fn reject_before_dispatch(self) {
        let mut ledger = self.handle.lock();
        if ledger.stage == ObservationStage::Prepared
            && ledger.dispatch == ObservationDispatch::NotDispatched
        {
            ledger.stage = ObservationStage::CallerRejectedBeforeDispatch;
        }
    }

    /// Commit through the writer, as the ledger's own tests do; the
    /// transport wrapper commits through the handle.
    #[cfg(test)]
    pub(crate) fn append(&mut self, original: String) -> Result<(), ObservationHandle> {
        self.handle
            .append(original)
            .map_err(|()| self.handle.clone())
    }

    pub(crate) fn awaiting_finalization(&mut self) {
        self.transition(ObservationStage::AwaitingNativeFinalization);
    }

    pub(crate) fn failed(&mut self) {
        self.transition(ObservationStage::NativeFailed);
    }

    fn failed_with(&mut self, cause: ExistingDiagnosticBasis) {
        let mut ledger = self.handle.lock();
        ledger.native_cause = Some(cause);
        if !ledger.stage.is_terminal() {
            ledger.stage = ObservationStage::NativeFailed;
        }
    }

    pub(crate) fn succeeded(&mut self) -> Result<(), ObservationHandle> {
        let mut ledger = self.handle.lock();
        if ledger.fault.is_some() {
            return Err(self.handle.clone());
        }
        if !ledger.stage.is_terminal() {
            ledger.stage = ObservationStage::NativeSucceeded;
        }
        Ok(())
    }

    fn transition(&mut self, stage: ObservationStage) {
        let mut ledger = self.handle.lock();
        if !ledger.stage.is_terminal() {
            ledger.stage = stage;
        }
    }
}

impl Drop for ObservationWriter {
    fn drop(&mut self) {
        self.transition(ObservationStage::Cancelled);
    }
}

/// Original diagnostic content, independent of newly retained evidence.
#[derive(Clone, Debug)]
pub enum ExistingDiagnosticBasis {
    /// No native error had materialized when capture failed.
    NoExistingNativeCause,
    /// Original opening or finalization error.
    Provider(Arc<ProviderError>),
    /// Original item error report.
    Stream(Arc<ErrorReport>),
}

/// The existing caller category of an observed native failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedFailureKind {
    /// Opening or finalization failed.
    Provider,
    /// Stream consumption failed.
    Stream,
    /// Capture failed before a native cause was available.
    CaptureOnly,
}

/// A failure with separate diagnostic content and read custody of the same ledger.
#[derive(Clone, Debug)]
pub struct ObservedFailure {
    /// Existing caller category, or the explicitly new capture-only path.
    pub kind: ObservedFailureKind,
    /// Original diagnostics; do not persist this wrapper's enriched Debug.
    pub diagnostic_basis: ExistingDiagnosticBasis,
    /// Full evidence retained independently of diagnostic rendering.
    pub observation: ObservationHandle,
}

impl ObservedFailure {
    pub(crate) fn provider(error: ProviderError, writer: &mut ObservationWriter) -> Self {
        let error = Arc::new(error);
        writer.handle.lock().native_error = Some(error.clone());
        let basis = ExistingDiagnosticBasis::Provider(error);
        writer.failed_with(basis.clone());
        Self {
            kind: ObservedFailureKind::Provider,
            diagnostic_basis: basis,
            observation: writer.handle(),
        }
    }

    fn report(error: ErrorReport, writer: &mut ObservationWriter) -> Self {
        let basis = ExistingDiagnosticBasis::Stream(Arc::new(error));
        writer.failed_with(basis.clone());
        Self {
            kind: ObservedFailureKind::Stream,
            diagnostic_basis: basis,
            observation: writer.handle(),
        }
    }

    fn report_provider(error: ProviderError, writer: &mut ObservationWriter) -> Self {
        let report = error.report();
        writer.handle.lock().native_error = Some(Arc::new(error));
        Self::report(report, writer)
    }

    fn capture_only(writer: &mut ObservationWriter) -> Self {
        writer.failed();
        Self {
            kind: ObservedFailureKind::CaptureOnly,
            diagnostic_basis: ExistingDiagnosticBasis::NoExistingNativeCause,
            observation: writer.handle(),
        }
    }
}

/// Incremental caller-visible information; original payloads live on the handle.
#[derive(Clone, Debug)]
pub enum ObservedEvent {
    /// Newly delivered text.
    TextDelta {
        /// The fragment, as Rig's decoder delivered it.
        text: String,
    },
    /// A forward-compatible frame, not normalized assistant output.
    Unknown {
        /// The frame's `type`; empty when it states none.
        event_type: String,
        /// The whole frame.
        value: Value,
    },
}

/// Rig's response and the provider's own response object, typed.
///
/// `response` is what [`Model::call`] returns for the same reply. `native`
/// is the provider's terminal response object, which `response.raw` holds,
/// read into this crate's typed Responses data. The observation handle
/// separately retains the original provider strings.
#[derive(Clone, Debug)]
pub struct ObservedResponsesResultBody {
    /// Rig's folded response.
    pub response: rig_core::completion::CompletionResponse,
    /// The provider's terminal response object.
    pub native: crate::responses_api::CompletionResponse,
}

/// Response views accompanied by custody of their original operation.
#[derive(Clone, Debug)]
pub struct ObservedResponsesResult {
    /// Rig's response and the provider-native response.
    pub body: ObservedResponsesResultBody,
    /// The same ledger installed by the caller before polling.
    pub observation: ObservationHandle,
}

/// One observed stream. Dropping a borrowed `next()` future does not drop its writer.
pub struct ObservedResponsesStream {
    stream: CompletionStream,
    writer: ObservationWriter,
    ended: bool,
    failure: Option<ObservedFailure>,
}

impl ObservedResponsesStream {
    fn new(stream: CompletionStream, writer: ObservationWriter) -> Self {
        Self {
            stream,
            writer,
            ended: false,
            failure: None,
        }
    }

    /// Read custody of this stream's operation.
    pub fn observation(&self) -> ObservationHandle {
        self.writer.handle()
    }

    /// Finalize: Rig reads what remains of the reply and folds it. A failure
    /// the stream already yielded is the result; a reply the provider did not
    /// end is Rig's truncation error.
    pub async fn finish(self) -> Result<ObservedResponsesResult, ObservedFailure> {
        let Self {
            stream,
            mut writer,
            failure,
            ..
        } = self;
        if let Some(failure) = failure {
            return Err(failure);
        }
        let response = match stream.finish().await {
            Ok(response) => response,
            Err(_) if writer.handle.has_capture_fault() => {
                return Err(ObservedFailure::capture_only(&mut writer));
            }
            Err(error) => return Err(ObservedFailure::provider(error, &mut writer)),
        };
        let native = match serde_json::from_value(response.raw.clone()) {
            Ok(native) => native,
            Err(error) => {
                return Err(ObservedFailure::provider(
                    ProviderError::Json(Arc::new(error)),
                    &mut writer,
                ));
            }
        };
        if writer.succeeded().is_err() {
            return Err(ObservedFailure::capture_only(&mut writer));
        }
        Ok(ObservedResponsesResult {
            body: ObservedResponsesResultBody { response, native },
            observation: writer.handle(),
        })
    }

    /// The failure `error` is: the capture's own when the ledger could not
    /// keep a supplied string, and Rig's otherwise.
    fn failure_of(&mut self, error: ProviderError) -> ObservedFailure {
        if self.writer.handle.has_capture_fault() {
            ObservedFailure::capture_only(&mut self.writer)
        } else {
            ObservedFailure::report_provider(error, &mut self.writer)
        }
    }
}

impl Stream for ObservedResponsesStream {
    type Item = Result<ObservedEvent, ObservedFailure>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.ended {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.stream).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.ended = true;
                    this.writer.awaiting_finalization();
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Ok(Item::Event(StreamEvent::Text { text, .. })))) => {
                    return Poll::Ready(Some(Ok(ObservedEvent::TextDelta { text })));
                }
                Poll::Ready(Some(Ok(Item::Event(_)))) => {}
                Poll::Ready(Some(Ok(Item::Unknown(payload)))) => {
                    let value = payload.value().clone();
                    let event_type = value
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    return Poll::Ready(Some(Ok(ObservedEvent::Unknown { event_type, value })));
                }
                Poll::Ready(Some(Err(error))) => {
                    let failure = this.failure_of(error);
                    this.ended = true;
                    this.failure = Some(failure.clone());
                    return Poll::Ready(Some(Err(failure)));
                }
            }
        }
    }
}

impl CodexResponses {
    /// Open one observed stream of `request` over `http`.
    ///
    /// The request is this wire's ordinary streamed request, sent by Rig's
    /// driver through [`CodexHttpClient`]: the same encoding, caller
    /// identity and headers as a [`Model`] of this wire. Each server-sent
    /// event's data is committed to the ledger as it arrives, before Rig
    /// decodes it. Reading stops after the first terminal response event;
    /// later bytes are neither read nor captured. A frame that is not UTF-8
    /// fails the stream and is not captured.
    ///
    /// The returned stream owns `writer`, so dropping it before a terminal
    /// state records the operation as cancelled. A request that cannot be
    /// encoded is an [`ObservedFailureKind::Provider`] failure and nothing is
    /// sent; the send and the reply are read lazily, on the first poll, and
    /// a failure there is a stream item.
    pub fn raw_stream_observed<H>(
        &self,
        http: &H,
        request: CompletionRequest,
        mut writer: ObservationWriter,
    ) -> Result<ObservedResponsesStream, ObservedFailure>
    where
        H: HttpClientExt + Clone + 'static,
    {
        let observing = ObservingHttpClient {
            inner: http.clone(),
            observation: writer.handle(),
        };
        let model = Model::new(self.clone(), CodexHttpClient::new(observing, self.clone()));
        match model.stream(request) {
            Ok(stream) => Ok(ObservedResponsesStream::new(stream, writer)),
            Err(error) => Err(ObservedFailure::provider(error, &mut writer)),
        }
    }
}

/// Why the transport wrapper ended a reply it was observing.
#[derive(Debug, thiserror::Error)]
enum ObservedReplyError {
    /// The ledger could not keep a supplied string; the fault is on the
    /// ledger.
    #[error("the observation ledger could not keep a server-sent event")]
    Capture,
    /// A frame was not UTF-8.
    #[error("a server-sent event was not UTF-8; it was not captured")]
    NotUtf8,
}

/// An [`HttpClientExt`] that records one operation's send attempt and reply
/// head, and commits each server-sent event of its streamed reply.
#[derive(Clone, Debug)]
struct ObservingHttpClient<H> {
    inner: H,
    observation: ObservationHandle,
}

/// Whether Rig's driver reads this reply as server-sent events on the
/// Codex contract: a reply of status 200, the one status the driver streams,
/// whose content type is `text/event-stream`, or which names no content
/// type, which the Codex gateway sends and Rig's Codex wire accepts
/// (`Encoded::with_relaxed_content_type`).
fn accepted_sse(status: http::StatusCode, headers: &http::HeaderMap) -> bool {
    driver_streams(status)
        && headers.get(http::header::CONTENT_TYPE).is_none_or(|value| {
            value
                .to_str()
                .ok()
                .and_then(|value| value.split(';').next())
                .is_some_and(|essence| essence.trim().eq_ignore_ascii_case("text/event-stream"))
        })
}

impl<H: HttpClientExt> HttpClientExt for ObservingHttpClient<H> {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send(req)
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let sending = self.inner.send_streaming(req);
        let observation = self.observation.clone();
        async move {
            // The response future is about to be polled for the first time.
            observation.mark_send_attempt_started();
            let (parts, body) = sending.await?.into_parts();
            let accepted = accepted_sse(parts.status, &parts.headers);
            observation.record_reply_head(parts.status, &parts.headers, accepted);
            // A refused reply's body is kept in Rig's error, not here.
            let body: BoxedStream = match Framing::Sse.framer() {
                Some(framer) if accepted => Box::pin(ObservedBody {
                    body,
                    framer,
                    partial: Vec::new(),
                    observation,
                    ended: false,
                }),
                _ => body,
            };
            Ok(Response::from_parts(parts, body))
        }
    }
}

/// A streamed body that commits each server-sent event its chunks complete,
/// and ends after the first terminal response event.
struct ObservedBody {
    body: BoxedStream,
    framer: Framer,
    /// The bytes of a UTF-8 sequence the last chunk left unfinished.
    partial: Vec<u8>,
    observation: ObservationHandle,
    ended: bool,
}

impl ObservedBody {
    /// End the reply with `error`.
    fn fail(&mut self, error: ObservedReplyError) -> Poll<Option<http_client::Result<Bytes>>> {
        self.ended = true;
        Poll::Ready(Some(Err(http_client::Error::Instance(Box::new(error)))))
    }
}

/// Whether `data` is a terminal response event.
fn is_terminal(data: &str) -> bool {
    matches!(
        decode_frame(data),
        Ok(DecodedFrame::Chunk(StreamingCompletionChunk::Response(chunk)))
            if matches!(
                chunk.kind,
                ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseFailed
                    | ResponseChunkKind::ResponseIncomplete
            )
    )
}

impl Stream for ObservedBody {
    type Item = http_client::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        let next = this.body.as_mut().poll_next(cx);
        if let Poll::Ready(Some(Ok(chunk))) = &next {
            // The framer reads bytes that are not UTF-8 as replacement
            // text, so only whole UTF-8 reaches it: a sequence a chunk
            // leaves unfinished waits for the next, and an invalid one
            // fails the reply once the events before it are committed.
            let mut bytes = std::mem::take(&mut this.partial);
            bytes.extend_from_slice(chunk);
            let (whole, invalid) = match std::str::from_utf8(&bytes) {
                Ok(_) => (bytes.len(), false),
                Err(error) => (error.valid_up_to(), error.error_len().is_some()),
            };
            let (text, rest) = bytes.split_at(whole);
            let frames: Vec<WireFrame> = this.framer.push(text).collect();
            for frame in frames {
                let WireFrame::Text(data) = frame else {
                    return this.fail(ObservedReplyError::NotUtf8);
                };
                let terminal = is_terminal(&data);
                // Commit the complete source string before Rig classifies it.
                if this.observation.append(data).is_err() {
                    return this.fail(ObservedReplyError::Capture);
                }
                if terminal {
                    // The chunk that completed the terminal is handed on;
                    // nothing after it is read.
                    this.ended = true;
                    return next;
                }
            }
            if invalid {
                return this.fail(ObservedReplyError::NotUtf8);
            }
            this.partial = rest.to_vec();
        }
        next
    }
}

#[cfg(test)]
mod tests;
