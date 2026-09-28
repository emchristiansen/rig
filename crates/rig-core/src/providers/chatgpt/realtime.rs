//! GPT-Live realtime calls over the ChatGPT subscription backend.
//!
//! A call joins three channels. [`LiveCalls::create_call`] posts the caller's
//! WebRTC offer and a [`SessionConfig`] to the Codex backend and returns the
//! answer SDP and the call id. The caller's WebRTC peer carries Opus audio and
//! the `oai-events` data channel. The control socket
//! ([`LiveCalls::connect_control`], feature `websocket`) carries typed
//! [`ServerEvent`]s and [`ClientEvent`]s for the same call.
//!
//! This is the alpha, Codex-only protocol (`openai-alpha: quicksilver=v2`), not
//! the public `/v1/live/sessions` API. Every request carries the provider's
//! credential, account id and exact caller identity, as the ChatGPT dialect's
//! other requests do.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::{self, realtime::{LiveCalls, SessionConfig}};
//! use rig_core::providers::openai::OpenAI;
//!
//! # async fn example(
//! #     http: &impl rig_core::http_client::HttpClientExt,
//! #     identity: rig_core::providers::openai::wire::CallerIdentity,
//! #     offer_sdp: &str,
//! # ) -> Result<(), rig_core::error::ProviderError> {
//! let provider = OpenAI::with_key(&chatgpt::DIALECT, "access-token")
//!     .with_caller_identity(identity)
//!     .with_account_id("account-id");
//! let calls = LiveCalls::new(provider)?;
//! let call = calls
//!     .create_call(http, offer_sdp, &SessionConfig::new("Answer briefly."))
//!     .await?;
//! println!("{} answered with {} bytes of SDP", call.call_id, call.answer_sdp.len());
//! # Ok(())
//! # }
//! ```

mod call;
#[cfg(feature = "websocket")]
mod control;
mod events;
mod session;

pub use call::{CallId, InvalidCallId, LiveCalls, NotTheCodexBackend, RealtimeCall};
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub use control::ControlSocket;
pub use events::{
    ClientEvent, ContentPart, ContextChannel, ContextChunk, DelegationContextAppended,
    DelegationCreated, DelegationItem, ErrorDetail, ErrorEvent, InputAudioAppend, OutputAudioDelta,
    Role, ServerEvent, SessionStarted, StartedSession, TranscriptAdded, TranscriptItem, Turn,
    TurnDelta, TurnEvent, UnknownEvent, Usage, UsageLimit, UsageUpdated, context_chunks,
};
pub use session::{InitialItem, SessionConfig, Voice};

/// `gpt-live-1-codex`, the GPT-Live model the Codex backend serves.
pub const GPT_LIVE_1_CODEX: &str = "gpt-live-1-codex";

/// The alpha opt-in header every request of this protocol carries.
pub const OPENAI_ALPHA_HEADER: &str = "openai-alpha";

/// The [`OPENAI_ALPHA_HEADER`] value selecting the frameless GPT-Live protocol.
pub const QUICKSILVER_V2: &str = "quicksilver=v2";

/// The header carrying the realtime session id.
pub const X_SESSION_ID_HEADER: &str = "x-session-id";

/// The call-creation path, relative to the ChatGPT Codex base URL.
pub const REALTIME_CALLS_PATH: &str = "realtime/calls";

/// The query every call creation carries.
pub const REALTIME_CALLS_QUERY: &str = "intent=quicksilver&architecture=avas";

/// The control socket base URL; the call id is its last path segment.
pub const CONTROL_SOCKET_BASE_URL: &str = "wss://api.openai.com/v1/live";

/// The most bytes of text one context append carries.
pub const CONTEXT_APPEND_MAX_BYTES: usize = 500;

/// The label of the ordered WebRTC data channel the caller's peer opens.
pub const EVENTS_DATA_CHANNEL: &str = "oai-events";
