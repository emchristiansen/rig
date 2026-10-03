//! Shared provider-local diagnostics for Live and observed Responses operations.
//!
//! These leaves retain the selected provider diagnostic structure independently
//! of observation custody. Baseline capability errors remain unchanged.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::http_client;
use http::StatusCode;

/// A boxed request-building failure.
#[cfg(not(target_family = "wasm"))]
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
/// A boxed request-building failure on WebAssembly.
#[cfg(target_family = "wasm")]
pub type BoxError = Box<dyn std::error::Error + 'static>;

/// Classifications used by the admitted provider operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Transport failure.
    Http,
    /// JSON encoding or decoding failure.
    Json,
    /// Invalid URL.
    Url,
    /// Request construction failure.
    Request,
    /// A decoded response does not answer the request.
    Response,
    /// Provider failure without a retained response.
    Provider,
    /// Provider failure with a retained response.
    ProviderResponse,
}

/// A retained provider diagnostic, separate from observation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorReport {
    /// Normalized classification.
    pub kind: ErrorKind,
    /// Whether the same operation may reasonably be retried.
    pub retryable: bool,
    /// Human-readable description (the source's `Display`).
    pub message: String,
    /// A provider- or tool-specific machine code, when one was reported:
    /// for a provider's reply, the transport's own code when it gave one
    /// apart from the body, else the code the body names
    /// ([`body_code`]).
    pub code: Option<String>,
    /// The HTTP status, when the failure had one.
    pub http_status: Option<u16>,
    /// The failure was an intentional refusal rather than a fault.
    pub refusal: bool,
    /// `Display` of each `source()` link, outermost first.
    pub source_chain: Vec<String>,
    /// The provider's request id, when the failure had a response that
    /// carried one.
    pub request_id: Option<String>,
    /// Preserved provider failure response, including available status, body,
    /// headers, and request ID.
    pub provider_response: Option<ProviderResponseError>,
    /// Structured diagnostic for failures a consumer may want to *route*
    /// rather than only display. Absent for the common case; see
    /// [`ErrorDetail`] for what each variant carries and why.
    pub detail: Option<ErrorDetail>,
}

/// Optional structured recovery diagnostic supplementing a report's kind and message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorDetail {
    /// A tool block declared complete contained invalid JSON, rather than
    /// truncated input. Carries exact accumulated arguments and the parser
    /// error for recovery or replay.
    #[cfg(feature = "completion-observations")]
    MalformedToolInput(MalformedToolInput),
    /// A frame of the provider's reply failed to decode. Carries what was
    /// kept of the frame, labelled by how faithful it is, so a consumer can
    /// keep it as evidence.
    CorruptFrame(CorruptFrameDetail),
}

/// The payload of [`ErrorDetail::CorruptFrame`]: a [`CorruptFrame`] in wire
/// form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptFrameDetail {
    /// The frame's discriminator value (for a Responses frame, its `type`),
    /// when the classifier read a string one.
    pub event_type: Option<String>,
    /// What was kept of the frame, and how faithful it is.
    pub evidence: FrameEvidence,
    /// Why the frame failed to decode.
    pub error: String,
}

impl From<&CorruptFrame> for CorruptFrameDetail {
    fn from(corrupt: &CorruptFrame) -> Self {
        Self {
            event_type: corrupt.event_type.clone(),
            evidence: corrupt.evidence.clone(),
            error: corrupt.error.to_string(),
        }
    }
}

/// The payload of [`ErrorDetail::MalformedToolInput`].
#[cfg(feature = "completion-observations")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedToolInput {
    /// The tool the model named.
    pub name: String,
    /// Durable correlation ID retained from the call for recovery actions,
    /// including tool results and rollback.
    pub id: crate::providers::openai::responses_api::observed_types::ToolCallId,
    /// The provider's own call id(s), when the wire supplied any.
    pub provider: Option<crate::providers::openai::responses_api::observed_types::ProviderCallId>,
    /// The raw argument text, byte-for-byte as accumulated.
    pub raw: String,
    /// The JSON parser's description of what was wrong.
    pub error: String,
}

impl ErrorReport {
    /// Build a report of `kind` with `message` and no other metadata.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            retryable: false,
            message: message.into(),
            code: None,
            http_status: None,
            refusal: false,
            source_chain: Vec::new(),
            request_id: None,
            provider_response: None,
            detail: None,
        }
    }

    /// Set `retryable`.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// Set the machine code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Set the HTTP status.
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Mark the report as an intentional refusal.
    pub fn refused(mut self) -> Self {
        self.refusal = true;
        self
    }

    /// Attach the provider's request id.
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Attach a typed diagnostic.
    pub fn with_detail(mut self, detail: ErrorDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Whether the same operation may reasonably be retried.
    pub const fn is_retryable(&self) -> bool {
        self.retryable
    }
}

impl ErrorReport {
    /// The provider response body preserved on this report, if any.
    pub fn provider_response_body(&self) -> Option<&str> {
        self.provider_response
            .as_ref()
            .map(|response| response.body.as_str())
    }

    /// The preserved provider response body parsed as JSON, when present.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        response_json(self.provider_response_body())
    }

    /// The preserved provider response headers, if any.
    pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
        self.provider_response
            .as_ref()
            .and_then(|response| response.headers.as_ref())
    }

    /// The HTTP status this report carries, as a status code.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        self.http_status
            .and_then(|status| http::StatusCode::from_u16(status).ok())
    }

    /// The provider's transport request id, if the failure carried one.
    pub fn provider_request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ErrorReport {}

/// A failed frame decode with its original evidence.
#[derive(Debug)]
pub struct CorruptFrame {
    event_type: Option<String>,
    evidence: FrameEvidence,
    error: serde_json::Error,
}

/// What a [`CorruptFrame`] kept of the frame, and how faithful it is.
///
/// Only [`Self::Text`] and [`Self::Bytes`] are the frame exactly as received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "frame", rename_all = "snake_case")]
pub enum FrameEvidence {
    /// The frame's text, exactly as received.
    Text(String),
    /// The frame's bytes, exactly as received. They are not valid UTF-8, so
    /// there is no exact text.
    Bytes(Vec<u8>),
    /// Text decoded from bytes that were not all valid UTF-8, each invalid
    /// sequence replaced with U+FFFD before this client saw the frame, as the
    /// `text/event-stream` grammar requires. Not the frame as received; the
    /// original bytes were not kept.
    Decoded(String),
    /// No frame: the transport delivered an already-decoded event.
    Absent,
}

impl CorruptFrame {
    /// A text frame that failed to decode, with the discriminator value the
    /// classifier read from it, if any.
    pub(crate) fn text(
        event_type: Option<String>,
        frame: impl Into<String>,
        error: serde_json::Error,
    ) -> Self {
        Self {
            event_type,
            evidence: FrameEvidence::Text(frame.into()),
            error,
        }
    }

    /// A byte frame that failed to decode.
    pub(crate) fn bytes(bytes: Vec<u8>, error: serde_json::Error) -> Self {
        Self {
            event_type: None,
            evidence: FrameEvidence::Bytes(bytes),
            error,
        }
    }

    /// A decode failure with no frame: a transport that delivers
    /// already-decoded events.
    pub(crate) fn absent(error: serde_json::Error) -> Self {
        Self {
            event_type: None,
            evidence: FrameEvidence::Absent,
            error,
        }
    }

    /// The same frame with `error` in place of the decode error.
    pub(crate) fn with_error(self, error: serde_json::Error) -> Self {
        Self { error, ..self }
    }

    /// Relabel text a decoder read from a replacement-decoded payload: it is
    /// not the frame as received.
    pub(crate) fn mark_replacement_decoded(&mut self) {
        if let FrameEvidence::Text(text) = &mut self.evidence {
            self.evidence = FrameEvidence::Decoded(std::mem::take(text));
        }
    }

    /// Replace what a decoder read lossily from `bytes`, the frame as
    /// received, with those bytes.
    pub(crate) fn set_received_bytes(&mut self, bytes: &[u8]) {
        if !matches!(self.evidence, FrameEvidence::Absent) {
            self.evidence = FrameEvidence::Bytes(bytes.to_vec());
        }
    }

    /// The frame's discriminator value (for a Responses frame, its `type`),
    /// when the classifier read a string one.
    pub fn event_type(&self) -> Option<&str> {
        self.event_type.as_deref()
    }

    /// What was kept of the frame, and how faithful it is.
    pub fn evidence(&self) -> &FrameEvidence {
        &self.evidence
    }

    /// The frame's text exactly as received. `None` when there is no exact
    /// text: see [`Self::evidence`].
    pub fn frame(&self) -> Option<&str> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The frame's bytes exactly as received, text or not. `None` when they
    /// were not kept: see [`Self::evidence`].
    pub fn frame_bytes(&self) -> Option<&[u8]> {
        match &self.evidence {
            FrameEvidence::Text(text) => Some(text.as_bytes()),
            FrameEvidence::Bytes(bytes) => Some(bytes),
            FrameEvidence::Decoded(_) | FrameEvidence::Absent => None,
        }
    }

    /// Why the frame failed to decode.
    pub fn error(&self) -> &serde_json::Error {
        &self.error
    }
}

impl fmt::Display for CorruptFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.event_type {
            Some(event_type) => write!(f, "`{event_type}` frame failed to decode: {}", self.error)?,
            None => write!(f, "frame failed to decode: {}", self.error)?,
        }
        match &self.evidence {
            FrameEvidence::Text(text) => write!(f, "; frame: {text}"),
            FrameEvidence::Bytes(bytes) => {
                write!(f, "; frame bytes (not UTF-8): {}", bytes.escape_ascii())
            }
            FrameEvidence::Decoded(text) => {
                write!(f, "; frame decoded with U+FFFD replacement: {text}")
            }
            FrameEvidence::Absent => Ok(()),
        }
    }
}

impl std::error::Error for CorruptFrame {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// A preserved provider reply with its available transport metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponseError {
    /// HTTP status of the provider response, when it was captured alongside the body.
    pub status: Option<StatusCode>,
    /// Raw response body as returned by the provider.
    pub body: String,
    /// Transport request ID from response headers or SDK metadata, when captured.
    pub provider_request_id: Option<String>,
    /// Captured response headers, including rate-limit metadata. `None` means
    /// not captured, rather than an empty header set.
    pub headers: Option<http::HeaderMap>,
    /// The provider's own machine-readable code for the failure, when the
    /// transport reported one apart from the body: a gRPC status code name
    /// (`UNAVAILABLE`), an AWS exception type (`ThrottlingException`).
    /// `None` when the reply carried only a status and a body.
    pub code: Option<String>,
    /// Transport retry verdict used when status is absent or successful.
    /// A non-success HTTP status takes precedence; refusals are never retryable.
    pub transient: Option<bool>,
    /// Whether this error represents a content refusal. Refusals are never
    /// retryable; refusal content in a successful model answer is separate.
    pub refusal: bool,
    /// The provider and request path the reply answered, when the operation
    /// records them for diagnostics.
    pub route: Option<String>,
}

impl ProviderResponseError {
    /// Preserve a provider error response captured with its HTTP status.
    pub fn new(status: StatusCode, body: impl Into<String>) -> Self {
        Self {
            status: Some(status),
            body: body.into(),
            provider_request_id: None,
            headers: None,
            code: None,
            transient: None,
            refusal: false,
            route: None,
        }
    }

    /// Preserve a provider error body that has no HTTP status (gRPC / SDK
    /// transports).
    pub fn without_status(body: impl Into<String>) -> Self {
        Self {
            status: None,
            body: body.into(),
            provider_request_id: None,
            headers: None,
            code: None,
            transient: None,
            refusal: false,
            route: None,
        }
    }

    /// Mark the reply as the provider's verdict on the content: a refusal,
    /// final, never retried.
    pub fn with_refusal(mut self, refusal: bool) -> Self {
        self.refusal = refusal;
        self
    }

    /// Attach the HTTP status a transport reported beside a reply that was
    /// first preserved without one (an SDK that hands back the raw HTTP
    /// response next to its typed exception). A status already set is kept.
    pub fn with_status(mut self, status: Option<StatusCode>) -> Self {
        if self.status.is_none() {
            self.status = status;
        }
        self
    }

    /// Attach the provider's own machine-readable code for the failure.
    pub fn with_code(mut self, code: Option<String>) -> Self {
        self.code = code.filter(|code| !code.is_empty());
        self
    }

    /// Replaces the transport retry verdict. Used for absent or successful
    /// HTTP statuses unless the response is a refusal.
    pub fn with_transient(mut self, transient: Option<bool>) -> Self {
        self.transient = transient;
        self
    }

    /// Returns false for refusals. Otherwise classifies non-success HTTP statuses
    /// through [`retryable_status`], or uses `transient` for absent
    /// or successful statuses. Missing verdicts default to false.
    pub fn is_retryable(&self) -> bool {
        if self.refusal {
            return false;
        }
        match self.status {
            Some(status) if !status.is_success() => retryable_status(Some(status.as_u16())),
            _ => self.transient.unwrap_or(false),
        }
    }

    /// Returns the explicit code, falling back to nonempty string fields
    /// `error.code`, `error.status`, then `error.type` in the JSON body.
    pub fn machine_code(&self) -> Option<String> {
        self.code.clone().or_else(|| body_code(&self.body))
    }

    /// Attach the transport request id the failed response reported.
    pub fn with_provider_request_id(mut self, request_id: Option<String>) -> Self {
        self.provider_request_id = request_id.filter(|id| !id.is_empty());
        self
    }

    /// Replaces captured response headers, including rate-limit metadata.
    pub fn with_headers(mut self, headers: Option<http::HeaderMap>) -> Self {
        self.headers = headers;
        self
    }
}

impl std::fmt::Display for ProviderResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "status {status}: {}", self.body)?,
            None => write!(f, "{}", self.body)?,
        }
        // The id support asks for belongs in the message a caller logs.
        if let Some(request_id) = &self.provider_request_id {
            write!(f, " (request id: {request_id})")?;
        }
        if let Some(route) = &self.route {
            write!(f, " [{route}]")?;
        }
        Ok(())
    }
}

impl std::error::Error for ProviderResponseError {}

/// Returns the first nonempty string at `error.code`, `error.status`, or
/// `error.type`, in that order. Invalid JSON, non-string fields, and missing
/// envelopes yield no code.
pub fn body_code(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    ["code", "status", "type"].iter().find_map(|field| {
        error
            .get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|code| !code.is_empty())
            .map(str::to_owned)
    })
}

/// Parses an optional response body as JSON.
///
/// Returns:
/// - `Ok(Some(value))` when a body is present and valid JSON.
/// - `Ok(None)` when the body is absent or empty.
/// - `Err(error)` when a body is present but isn't valid JSON.
fn response_json(body: Option<&str>) -> Result<Option<serde_json::Value>, serde_json::Error> {
    body.filter(|body| !body.is_empty())
        .map(serde_json::from_str)
        .transpose()
}

/// A failure in the admitted Live or observed Responses provider operation.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// A transport failure that produced no provider reply: a reset
    /// connection, a timeout, an unreadable response. Also retains baseline
    /// status-only errors, which supply no body for [`Self::ProviderResponse`].
    #[error("HttpError: {0}")]
    Http(http_client::Error),
    /// JSON serialization or deserialization failed.
    #[error("JsonError: {0}")]
    Json(#[from] serde_json::Error),
    /// A frame of the provider's reply failed to decode. What was kept of the
    /// frame travels with it; see [`CorruptFrame`].
    #[error("CorruptFrameError: {0}")]
    CorruptFrame(#[source] CorruptFrame),
    /// A URL could not be parsed.
    #[error("UrlError: {0}")]
    Url(#[from] url::ParseError),
    /// The request could not be built.
    #[error("RequestError: {0}")]
    Request(#[from] BoxError),
    /// The reply decoded but does not answer the request.
    #[error("ResponseError: {0}")]
    Response(String),
    /// The provider reported a failure without a preserved reply.
    #[error("ProviderError: {0}")]
    Provider(String),
    /// The provider's reply, preserved: a non-success status with its body, a
    /// 2xx error envelope, or a non-HTTP transport's error payload.
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(ProviderResponseError),
    /// The provider rejected the configured credentials with 401 or 403.
    #[error("invalid authentication: {0}")]
    InvalidAuthentication(ProviderResponseError),
}

impl ProviderError {
    /// Preserves the status and verbatim body as [`Self::ProviderResponse`],
    /// including error envelopes returned with 2xx statuses.
    pub fn from_http_response(status: http::StatusCode, body: impl Into<String>) -> Self {
        Self::ProviderResponse(ProviderResponseError::new(status, body))
    }

    /// Preserves a verbatim provider error body with no HTTP status as
    /// [`Self::ProviderResponse`].
    pub fn from_provider_body(body: impl Into<String>) -> Self {
        Self::ProviderResponse(ProviderResponseError::without_status(body))
    }

    /// Converts a non-success reply the transport reported as an error to
    /// [`Self::ProviderResponse`], keeping its status, body, and headers.
    /// Other transport errors become [`Self::Http`].
    pub fn from_transport_error(error: http_client::Error) -> Self {
        match error {
            http_client::Error::InvalidStatusCodeWithDetails {
                status,
                body,
                headers,
            } => Self::from_http_response(status, body).with_response_headers(Some(*headers)),
            http_client::Error::InvalidStatusCodeWithMessage(status, body) => {
                Self::from_http_response(status, body)
            }
            // A legacy status-only error has no body to put in a response leaf.
            other => Self::Http(other),
        }
    }

    /// The classification this error reports as.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Http(_) => ErrorKind::Http,
            Self::Json(_) | Self::CorruptFrame(_) => ErrorKind::Json,
            Self::Url(_) => ErrorKind::Url,
            Self::Request(_) => ErrorKind::Request,
            Self::Response(_) => ErrorKind::Response,
            Self::Provider(_) => ErrorKind::Provider,
            Self::ProviderResponse(_) | Self::InvalidAuthentication(_) => {
                ErrorKind::ProviderResponse
            }
        }
    }

    /// Classifies transport failures with [`transient_transport`] and
    /// preserved replies with [`ProviderResponseError::is_retryable`]. Every
    /// other failure, including rejected credentials, is not
    /// retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => transient_transport(error),
            Self::ProviderResponse(response) => response.is_retryable(),
            _ => false,
        }
    }

    /// The provider's preserved reply, when this error carries one.
    pub fn provider_response(&self) -> Option<&ProviderResponseError> {
        match self {
            Self::ProviderResponse(response) | Self::InvalidAuthentication(response) => {
                Some(response)
            }
            _ => None,
        }
    }

    /// The undecodable frame and what was kept of it, when this error is
    /// one.
    pub fn corrupt_frame(&self) -> Option<&CorruptFrame> {
        match self {
            Self::CorruptFrame(corrupt) => Some(corrupt),
            _ => None,
        }
    }

    /// The preserved reply's body. An empty body returns `Some("")`, while
    /// [`Self::provider_response_json`] maps it to `Ok(None)`.
    pub fn provider_response_body(&self) -> Option<&str> {
        self.provider_response()
            .map(|response| response.body.as_str())
    }

    /// Parses the preserved reply's body as JSON: `Ok(None)` when there is no
    /// body or it is empty, `Err` when it is not valid JSON.
    pub fn provider_response_json(&self) -> Result<Option<serde_json::Value>, serde_json::Error> {
        response_json(self.provider_response_body())
    }

    /// The preserved reply's HTTP status. It may be 2xx for an error envelope.
    pub fn provider_response_status(&self) -> Option<http::StatusCode> {
        self.provider_response()
            .and_then(|response| response.status)
            .or_else(|| match self {
                Self::Http(error) => error.non_success_status(),
                _ => None,
            })
    }

    /// The provider's transport request ID, when the reply carried one.
    pub fn provider_request_id(&self) -> Option<&str> {
        self.provider_response()
            .and_then(|response| response.provider_request_id.as_deref())
    }

    /// The preserved reply's headers; `None` means the transport supplied none.
    pub fn provider_response_headers(&self) -> Option<&http::HeaderMap> {
        self.provider_response()
            .and_then(|response| response.headers.as_ref())
    }

    /// Fills an absent request ID on the preserved reply, ignoring empty
    /// strings.
    pub fn with_provider_request_id(self, request_id: Option<String>) -> Self {
        self.map_response(|response| match response.provider_request_id {
            Some(_) => response,
            None => response.with_provider_request_id(request_id),
        })
    }

    /// Fills absent headers on the preserved reply.
    pub fn with_response_headers(self, headers: Option<http::HeaderMap>) -> Self {
        self.map_response(|response| match (&response.headers, headers) {
            (None, Some(headers)) => response.with_headers(Some(headers)),
            _ => response,
        })
    }

    /// Attaches the HTTP status a transport reported beside a reply preserved
    /// without one, so it classifies by status. A captured status is kept.
    pub fn with_provider_status(self, status: Option<http::StatusCode>) -> Self {
        self.map_response(|response| response.with_status(status))
    }

    /// Attaches the provider's machine-readable code for the failure, such as
    /// a gRPC status name or an AWS exception type.
    pub fn with_provider_code(self, code: Option<String>) -> Self {
        self.map_response(|response| response.with_code(code))
    }

    /// Replaces the preserved reply's transport retry verdict, used when its
    /// status is absent or successful.
    pub fn with_transient(self, transient: Option<bool>) -> Self {
        self.map_response(|response| response.with_transient(transient))
    }

    /// The retained diagnostic report of this error.
    pub fn report(&self) -> ErrorReport {
        ErrorReport::from(self)
    }

    fn map_response(
        self,
        map: impl FnOnce(ProviderResponseError) -> ProviderResponseError,
    ) -> Self {
        match self {
            Self::ProviderResponse(response) => Self::ProviderResponse(map(response)),
            Self::InvalidAuthentication(response) => Self::InvalidAuthentication(map(response)),
            other => other,
        }
    }
}

impl From<http_client::Error> for ProviderError {
    fn from(error: http_client::Error) -> Self {
        Self::from_transport_error(error)
    }
}

/// A request-construction failure that retains its typed cause.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct EncodeError(ProviderError);

impl EncodeError {
    /// A request that could not be built, for the given reason.
    pub fn request(reason: impl Into<BoxError>) -> Self {
        Self(ProviderError::Request(reason.into()))
    }
}

impl From<EncodeError> for ProviderError {
    fn from(error: EncodeError) -> Self {
        debug_assert_eq!(error.0.kind(), ErrorKind::Request);
        error.0
    }
}

impl From<http::Error> for EncodeError {
    fn from(error: http::Error) -> Self {
        Self::request(error)
    }
}

impl From<serde_json::Error> for EncodeError {
    fn from(error: serde_json::Error) -> Self {
        Self::request(error)
    }
}

impl From<BoxError> for EncodeError {
    fn from(error: BoxError) -> Self {
        Self(ProviderError::Request(error))
    }
}
impl From<http::Error> for ProviderError {
    fn from(error: http::Error) -> Self {
        Self::Request(Box::new(error))
    }
}
impl From<super::identity::InvalidCodexIdentity> for ProviderError {
    fn from(error: super::identity::InvalidCodexIdentity) -> Self {
        Self::Request(Box::new(error))
    }
}

impl From<&ProviderError> for ErrorReport {
    fn from(error: &ProviderError) -> Self {
        let response = error.provider_response();
        ErrorReport {
            kind: error.kind(),
            retryable: error.is_retryable(),
            message: error.to_string(),
            code: response.and_then(ProviderResponseError::machine_code),
            http_status: error
                .provider_response_status()
                .map(|status| status.as_u16()),
            refusal: response.is_some_and(|response| response.refusal),
            source_chain: source_chain(error),
            request_id: response.and_then(|response| response.provider_request_id.clone()),
            provider_response: response.cloned(),
            detail: error
                .corrupt_frame()
                .map(|corrupt| ErrorDetail::CorruptFrame(corrupt.into())),
        }
    }
}

impl From<ProviderError> for ErrorReport {
    fn from(error: ProviderError) -> Self {
        Self::from(&error)
    }
}

/// Public spelling shared by the Live and observation entry points.
pub type LiveProviderError = ProviderError;
/// Public spelling for Live reply evidence.
pub type LiveResponseError = ProviderResponseError;

/// Selected status retry policy.
pub const fn retryable_status(status: Option<u16>) -> bool {
    matches!(status, Some(408 | 425 | 429 | 500..=599))
}

/// Selected transport retry policy, including baseline legacy status variants.
pub fn transient_transport(error: &http_client::Error) -> bool {
    match error {
        http_client::Error::StreamEnded | http_client::Error::Instance(_) => true,
        http_client::Error::InvalidStatusCode(status)
        | http_client::Error::InvalidStatusCodeWithMessage(status, _)
        | http_client::Error::InvalidStatusCodeWithDetails { status, .. } => {
            retryable_status(Some(status.as_u16()))
        }
        _ => false,
    }
}

fn source_chain(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = error.source();
    while let Some(source) = current {
        chain.push(source.to_string());
        current = source.source();
    }
    chain
}

/// Read a nonempty request id from the supplied headers.
pub(crate) fn request_id_from_headers(
    headers: &http::HeaderMap,
    name: Option<&str>,
) -> Option<String> {
    headers
        .get(name?)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests;
