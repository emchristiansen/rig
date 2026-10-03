#![cfg_attr(docsrs, feature(doc_cfg))]
//! The WebRTC peer of a GPT-Live realtime call.
//!
//! A [`LivePeer`] is the offering side of a call created with
//! `rig_core::providers::chatgpt::realtime::LiveCalls`: one sendrecv Opus track
//! (48 kHz, payload type 111) and the ordered `oai-events` data channel. It
//! sends Opus packets the caller already encoded and hands back the Opus
//! packets and data-channel messages it receives, as [`PeerEvent`]s, through
//! a bounded queue that reports dropped events as [`PeerEvent::Lost`]. It
//! decodes no audio; the control socket also carries `ServerEvent::OutputAudioDelta`
//! events containing base64 audio.
//!
//! The WebRTC stack enables rustls's ring provider. The `rig-core` `rustls`
//! feature enables aws-lc-rs for `rig_core::http_client::ReqwestClient`, which
//! is reqwest's `Client`. With both linked, rustls cannot select a provider
//! automatically. The binary must depend on `rustls` 0.23
//! with its `aws_lc_rs` feature and install that provider before it makes a
//! connection, as the first lines of the example do. The README lists the
//! dependencies and the peer's delivery and lifetime contract.
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use rig_core::providers::chatgpt::realtime;
//! use rig_core::providers::live_support::{LiveConfiguration, CallerIdentity};
//! use rig_webrtc::{LivePeer, OpusPacket, PeerEvent};
//!
//! # async fn example(identity: CallerIdentity) -> Result<(), Box<dyn std::error::Error>> {
//! rustls::crypto::aws_lc_rs::default_provider()
//!     .install_default()
//!     .map_err(|_| "another rustls provider is already installed")?;
//!
//! let provider = LiveConfiguration::subscription("access-token")
//!     .with_account_id("account-id")
//!     .with_caller_identity(identity);
//! let calls = realtime::LiveCalls::new(provider)?;
//! let peer = LivePeer::builder().build().await?;
//! let offer_sdp = peer.offer().await?;
//! let session = realtime::SessionConfig::new("Answer briefly.");
//! let call = calls
//!     .create_call(&rig_core::http_client::ReqwestClient::builder().build()?, &offer_sdp, &session)
//!     .await?;
//! peer.apply_answer(call.answer_sdp).await?;
//! let mut control = calls
//!     .connect_control(&rig_tungstenite::TungsteniteClient::new(), call.call_id)
//!     .await?;
//!
//! // Media and control run side by side; the peer is shared by reference.
//! let media = async {
//!     peer.send_opus(&OpusPacket::new(vec![0xf8, 0xff, 0xfe], Duration::from_millis(20)))
//!         .await?;
//!     while let Some(event) = peer.next_event().await {
//!         if let PeerEvent::Lost(lost) = event {
//!             eprintln!("the consumer fell behind: {lost:?}");
//!         }
//!     }
//!     Ok::<_, Box<dyn std::error::Error>>(())
//! };
//! let control_loop = async {
//!     while let Some(event) = control.next_event().await? {
//!         if let realtime::ServerEvent::DelegationCreated(created) = event {
//!             control
//!                 .append_delegation_context(&created.item.id, None, "All services are healthy.")
//!                 .await?;
//!             break;
//!         }
//!     }
//!     control.close().await?;
//!     peer.close().await?;
//!     Ok::<_, Box<dyn std::error::Error>>(())
//! };
//! tokio::try_join!(media, control_loop)?;
//! # Ok(())
//! # }
//! ```

#[cfg(target_family = "wasm")]
compile_error!(
    "rig-webrtc is native-only (webrtc-rs binds UDP and TCP sockets on tokio); a browser peer \
     uses the browser's own RTCPeerConnection."
);

#[cfg(not(target_family = "wasm"))]
mod peer;

#[cfg(not(target_family = "wasm"))]
pub use peer::{
    LivePeer, LivePeerBuilder, LostEvents, OpusPacket, PeerError, PeerEvent, ReceivedOpus,
};
#[cfg(not(target_family = "wasm"))]
pub use webrtc::peer_connection::RTCPeerConnectionState as PeerConnectionState;

/// The RTP payload type of the Opus track.
pub const OPUS_PAYLOAD_TYPE: u8 = 111;

/// The Opus RTP clock rate, in hertz.
pub const OPUS_CLOCK_RATE: u32 = 48_000;
