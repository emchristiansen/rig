#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! GPT-Live calls for Rig, written against `rig-core`'s public API.
//!
//! [`openai`] creates sessions on the public OpenAI Live API with an API-key
//! provider; [`chatgpt`] creates realtime calls on the ChatGPT subscription's
//! Codex backend and, with the `websocket` feature, joins their control
//! socket. Both hand back an SDP answer for the caller's WebRTC peer, which
//! `rig-webrtc` provides on native targets. [`support`] holds the identities,
//! request headers, send-time credentials and errors both share.
//!
//! ```no_run
//! use rig_core::providers::openai::OpenAI;
//! use rig_live::openai::{PublicLiveSessions, SessionConfig};
//!
//! # async fn example(
//! #     http: &impl rig_core::http_client::HttpClientExt,
//! #     offer_sdp: &str,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! let sessions = PublicLiveSessions::new(OpenAI::new("api-key"))?;
//! let created = sessions
//!     .create_session(http, offer_sdp, &SessionConfig::new().with_instructions("Be concise."))
//!     .await?;
//! println!("{} answered", created.session_id);
//! # Ok(())
//! # }
//! ```

pub mod chatgpt;
mod driver;
mod headers;
pub mod openai;
pub mod support;
