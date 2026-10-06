//! Caller-owned observations of one Responses operation.
//!
//! The ledger contains the original strings supplied by the SSE source, before
//! interpretation. It is an in-memory prefix, not a record of unread transport
//! bytes. Keeping a handle retains that prefix after the owning future is dropped.

use crate::error::{ErrorReport, ProviderError};
use std::collections::TryReserveError;
use std::sync::{Arc, Mutex, MutexGuard};

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
    /// Source EOF or a whole-response terminal ended reading; finalization is pending.
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
/// A refused reply's body is not kept here: the native error carries it, read
/// to a bounded length as the driver reads every refused reply.
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

    fn response_metadata(&self) -> (Option<String>, std::collections::BTreeMap<String, String>) {
        self.inspect(|view| {
            let Some(head) = view.reply_head.filter(|head| head.accepted_sse) else {
                return (None, Default::default());
            };
            let request_id = head
                .headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            let mut headers = std::collections::BTreeMap::<String, String>::new();
            for (name, value) in &head.headers {
                if name.as_str().starts_with("x-codex-") {
                    let value = String::from_utf8_lossy(value.as_bytes());
                    headers
                        .entry(name.as_str().to_owned())
                        .and_modify(|existing| {
                            existing.push_str(", ");
                            existing.push_str(&value);
                        })
                        .or_insert_with(|| value.into_owned());
                }
            }
            (request_id, headers)
        })
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

    pub(crate) fn append(&mut self, original: String) -> Result<(), ObservationHandle> {
        let mut ledger = self.handle.lock();
        if ledger.fault.is_some() {
            if let Some(fault) = ledger.fault.as_mut() {
                if fault.offending_string.is_none() {
                    fault.offending_string = Some(original);
                }
            }
            return Err(self.handle.clone());
        }
        if let Err(cause) = ledger.committed.try_reserve(1) {
            ledger.fault = Some(CaptureFault {
                cause: CaptureCause::Allocation(cause),
                offending_string: Some(original),
            });
            ledger.stage = ObservationStage::NativeFailed;
            return Err(self.handle.clone());
        }
        ledger.committed.push(original);
        Ok(())
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
        let error = error.with_provider_request_id(writer.handle.response_metadata().0);
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
        let error = error.with_provider_request_id(writer.handle.response_metadata().0);
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

use super::streaming::observed::{
    ObservedAssembler, ObservedEvent, ObservedInterpretationError, ObservedResponsesResultBody,
};
use crate::driver::observed::ObservedSourceEvent;
use futures::Stream;
use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type ObservedSource =
    futures::stream::BoxStream<'static, Result<ObservedSourceEvent, ProviderError>>;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type ObservedSource =
    futures::stream::LocalBoxStream<'static, Result<ObservedSourceEvent, ProviderError>>;

/// Response views accompanied by custody of their original operation.
#[derive(Clone, Debug)]
pub struct ObservedResponsesResult {
    /// Normalized response and provider-native response.
    pub body: ObservedResponsesResultBody,
    /// The same ledger installed by the caller before polling.
    pub observation: ObservationHandle,
}

/// One observed stream. Dropping a borrowed `next()` future does not drop its writer.
pub struct ObservedResponsesStream {
    source: ObservedSource,
    assembler: Option<ObservedAssembler>,
    writer: ObservationWriter,
    ready: VecDeque<Result<ObservedEvent, ObservedFailure>>,
    ended: bool,
    failure: Option<ObservedFailure>,
}

impl ObservedResponsesStream {
    pub(crate) fn new(source: ObservedSource, writer: ObservationWriter, provider: &str) -> Self {
        Self {
            source,
            assembler: Some(ObservedAssembler::new(provider)),
            writer,
            ready: VecDeque::new(),
            ended: false,
            failure: None,
        }
    }

    /// Read custody of this stream's operation.
    pub fn observation(&self) -> ObservationHandle {
        self.writer.handle()
    }

    /// Finalize after source EOF or a whole-response terminal, without more I/O.
    pub fn finish(mut self) -> Result<ObservedResponsesResult, ObservedFailure> {
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        if !self.ended {
            return Err(ObservedFailure::provider(ProviderError::Response("provider stream ended without a terminal record; treating the turn as truncated".into()), &mut self.writer));
        }
        let Some(assembler) = self.assembler.take() else {
            return Err(ObservedFailure::provider(ProviderError::Response("provider stream ended without a terminal record; treating the turn as truncated".into()), &mut self.writer));
        };
        let body = match assembler.finish() {
            Ok(body) => body,
            Err(ObservedInterpretationError::NativeProvider(error)) => {
                return Err(ObservedFailure::provider(error, &mut self.writer));
            }
            Err(ObservedInterpretationError::NativeReport(report)) => {
                return Err(ObservedFailure::report(report, &mut self.writer));
            }
        };
        if self.writer.succeeded().is_err() {
            return Err(ObservedFailure::capture_only(&mut self.writer));
        }
        Ok(ObservedResponsesResult {
            body,
            observation: self.writer.handle(),
        })
    }

    fn fail_item(
        &mut self,
        failure: ObservedFailure,
    ) -> Poll<Option<Result<ObservedEvent, ObservedFailure>>> {
        self.ended = true;
        self.failure = Some(failure.clone());
        self.ready.push_back(Err(failure));
        Poll::Ready(self.ready.pop_front())
    }
}

impl Stream for ObservedResponsesStream {
    type Item = Result<ObservedEvent, ObservedFailure>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.ready.pop_front() {
                return Poll::Ready(Some(event));
            }
            if self.ended {
                return Poll::Ready(None);
            }
            match self.source.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    let folded = match self.assembler.as_mut() {
                        Some(assembler) => assembler.eof(),
                        None => Ok(Vec::new()),
                    };
                    match folded {
                        Ok(events) => self.ready.extend(events.into_iter().map(Ok)),
                        Err(ObservedInterpretationError::NativeProvider(error)) => {
                            let failure = ObservedFailure::report_provider(error, &mut self.writer);
                            return self.fail_item(failure);
                        }
                        Err(ObservedInterpretationError::NativeReport(report)) => {
                            let failure = ObservedFailure::report(report, &mut self.writer);
                            return self.fail_item(failure);
                        }
                    }
                    self.ended = true;
                    self.writer.awaiting_finalization();
                    if let Some(event) = self.ready.pop_front() {
                        return Poll::Ready(Some(event));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(error))) => {
                    let flushed = match self.assembler.as_mut() {
                        Some(assembler) => assembler.flush_before_terminal_error(),
                        None => Ok(()),
                    };
                    match flushed {
                        Err(ObservedInterpretationError::NativeProvider(error)) => {
                            let failure = ObservedFailure::report_provider(error, &mut self.writer);
                            return self.fail_item(failure);
                        }
                        Err(ObservedInterpretationError::NativeReport(report)) => {
                            let failure = ObservedFailure::report(report, &mut self.writer);
                            return self.fail_item(failure);
                        }
                        Ok(()) => {}
                    }
                    let failure = ObservedFailure::report_provider(error, &mut self.writer);
                    return self.fail_item(failure);
                }
                Poll::Ready(Some(Ok(ObservedSourceEvent::Open))) => {
                    let (request_id, headers) = self.writer.handle.response_metadata();
                    if let Some(assembler) = self.assembler.as_mut() {
                        assembler.set_response_metadata(request_id, headers);
                    }
                }
                Poll::Ready(Some(Ok(ObservedSourceEvent::Message(data)))) => {
                    // Commit the complete source String before any classification.
                    if self.writer.append(data).is_err() {
                        let failure = ObservedFailure::capture_only(&mut self.writer);
                        return self.fail_item(failure);
                    }
                    let handle = self.writer.handle();
                    let Some(assembler) = self.assembler.as_mut() else {
                        continue;
                    };
                    let mut events = Vec::new();
                    let interpreted = handle.inspect(|view| match view.committed_prefix.last() {
                        Some(original) => assembler.push(original, &mut events),
                        None => Ok(()),
                    });
                    self.ready.extend(events.into_iter().map(Ok));
                    match interpreted {
                        Ok(()) => {
                            if self
                                .assembler
                                .as_ref()
                                .is_some_and(|assembler| assembler.whole_response_finished())
                            {
                                self.ended = true;
                                self.writer.awaiting_finalization();
                            }
                        }
                        Err(ObservedInterpretationError::NativeProvider(error)) => {
                            let failure = ObservedFailure::report_provider(error, &mut self.writer);
                            return self.fail_item(failure);
                        }
                        Err(ObservedInterpretationError::NativeReport(report)) => {
                            let failure = ObservedFailure::report(report, &mut self.writer);
                            return self.fail_item(failure);
                        }
                    }
                }
            }
        }
    }
}

impl super::wire::Responses {
    /// Open one observed stream of `request` over `http`.
    ///
    /// The request is this wire's ordinary streamed request: the same
    /// encoding, provider credential, caller identity and headers as
    /// [`driver::stream`](crate::driver::stream). The returned stream owns
    /// `writer`, so dropping it before a terminal state records the operation
    /// as cancelled. A request that cannot be encoded is a
    /// [`ObservedFailureKind::Provider`] failure and nothing is sent; the
    /// credential read, the send and the reply are read lazily, on the first
    /// poll, and a failure there is a stream item.
    pub fn raw_stream_observed<H>(
        &self,
        http: &H,
        request: crate::completion::CompletionRequest,
        mut writer: ObservationWriter,
    ) -> Result<ObservedResponsesStream, ObservedFailure>
    where
        H: crate::http_client::HttpClientExt + Clone + 'static,
    {
        use crate::wire::Wire as _;

        match crate::driver::observed::observed_sse(self, http, request, writer.handle()) {
            Ok(source) => Ok(ObservedResponsesStream::new(
                Box::pin(source),
                writer,
                self.name(),
            )),
            Err(error) => Err(ObservedFailure::provider(error, &mut writer)),
        }
    }
}

#[cfg(test)]
mod tests;
