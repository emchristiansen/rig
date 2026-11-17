#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! Codex subscription support for Rig, built on Rig's public API.
//!
//! [`responses_api`] holds the typed Responses data, the Codex wire
//! ([`CodexResponses`](responses_api::wire::CodexResponses)) with its
//! conversation identity, request headers and request compression, and the
//! Codex websocket session. The Codex wire is the wire a model encodes
//! with: it sends the request's declared tools and its `include` as the
//! caller wrote them. [`http_client`] sends that wire's requests with its
//! exact headers and compression, and refuses a body some other wire wrote,
//! [`ws_client`] names the websocket connection an idle session services,
//! [`error`] the evidence kept of a frame that failed to decode, and
//! [`frame_tee`] the frames a transport wrapper kept while handing them to
//! Rig.
//!
//! ```
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::providers::openai::responses_api::wire::Responses;
//!
//! let wire = OpenAIConfig::with_key(&chatgpt::DIALECT, "access-token");
//! let wire = CodexResponses::new(Responses::new(wire, "gpt-5.4"))?;
//! // Over HTTP, with `http` a transport:
//! // rig_core::Model::new(wire.clone(), CodexHttpClient::new(http, wire))
//! # let _ = wire;
//! # Ok::<(), rig_codex::responses_api::codex_identity::NotACodexWire>(())
//! ```

pub mod error;
pub mod frame_tee;
pub mod http_client;
pub mod responses_api;
#[cfg(all(feature = "tungstenite", not(target_family = "wasm")))]
#[cfg_attr(docsrs, doc(cfg(feature = "tungstenite")))]
pub mod tungstenite;
pub mod wire;
pub mod ws_client;
