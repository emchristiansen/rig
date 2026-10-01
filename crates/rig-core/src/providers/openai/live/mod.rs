//! GPT-Live sessions over the public Live API, with a WebRTC transport.
//!
//! [`PublicLiveSessions::create_session`] posts the caller's SDP offer and a
//! [`SessionConfig`] to `POST /live/sessions` and returns the session id and
//! SDP answer, or a [`LiveApiError`] that tells rejected credentials and a
//! reached spend limit apart from other failures. The caller's WebRTC peer carries the
//! audio and the `oai-events` data channel, whose messages are
//! [`ServerEvent`]s and [`ClientEvent`]s.
//!
//! With Responses delegation the provider runs the backend model, and the
//! caller executes its function tools: a [`FunctionCallCollector`] fed the
//! server events reports when a backend response's calls need outputs.
//!
//! ```no_run
//! use rig_core::providers::openai::{OpenAI, live::{
//!     Delegation, PublicLiveSessions, ResponsesDelegation, SessionConfig, Voice,
//! }};
//!
//! # async fn example(
//! #     http: &impl rig_core::http_client::HttpClientExt,
//! #     offer_sdp: &str,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! let sessions = PublicLiveSessions::new(OpenAI::new("api-key"))?;
//! let session = SessionConfig::new()
//!     .with_instructions("Be concise.")
//!     .with_voice(Voice::Marin)
//!     .with_delegation(Delegation::Responses(ResponsesDelegation::new("gpt-6-luna")));
//! let created = sessions.create_session(http, offer_sdp, &session).await?;
//! println!("{} answered with {} bytes of SDP", created.session_id, created.answer_sdp.len());
//! # Ok(())
//! # }
//! ```

mod create;
mod events;
mod function_calls;
mod session;

pub use create::{
    ApiErrorDetail, CreatedSession, EmptyOffer, LiveApiError, LiveErrorReply, NotOfficialOpenAi,
    PublicLiveSessions,
};
pub use events::{
    Acknowledgement, Appended, BackendEvent, ClientEvent, CloseReason, ContextAppend,
    ContextWindow, DelegationCreated, DelegationInfo, DelegationTarget, ErrorEvent, FunctionCall,
    ResponseEvent, ResponseOutcome, ServerEvent, SessionClosed, SessionResource, SessionSnapshot,
    SessionUsage, TranscriptDelta, UnknownEvent, UsageUpdated,
};
pub use function_calls::{
    CallsUpdate, FunctionCallCollector, MismatchedOutputs, PendingFunctionCalls,
};
pub use session::{
    Delegation, DelegationTool, FunctionTool, InitialItem, InvalidMaxOutputTokens, MaxOutputTokens,
    Reasoning, ReasoningEffort, ReasoningSummary, ResponsesDelegation, ResponsesDelegationUpdate,
    ResponsesSettings, ServiceTier, SessionConfig, TextSettings, ToolChoice, Verbosity, Voice,
};

/// `gpt-live-1`, the Live model.
pub const GPT_LIVE_1: &str = "gpt-live-1";

/// The session-creation path, relative to the OpenAI base URL.
pub const LIVE_SESSIONS_PATH: &str = "live/sessions";
