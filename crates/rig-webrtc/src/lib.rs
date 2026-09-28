#![cfg_attr(docsrs, feature(doc_cfg))]
//! The WebRTC peer of a GPT-Live realtime call.
//!
//! A [`LivePeer`] is the offering side of a call created with
//! `rig_core::providers::chatgpt::realtime::LiveCalls`: one sendrecv Opus track
//! (48 kHz, payload type 111) and the ordered `oai-events` data channel. It
//! sends Opus packets the caller already encoded and hands back the Opus
//! packets and data-channel messages it receives, as [`PeerEvent`]s. It
//! decodes no audio; the control socket also carries the model's audio as
//! PCM.
//!
//! ```no_run
//! use rig_webrtc::{LivePeer, OpusPacket};
//! use std::time::Duration;
//!
//! # async fn example(answer_sdp: String) -> Result<(), rig_webrtc::PeerError> {
//! let mut peer = LivePeer::builder().build().await?;
//! let offer_sdp = peer.offer().await?;
//! // Create the call with `offer_sdp`; apply the answer it returns.
//! peer.apply_answer(answer_sdp).await?;
//! peer.send_opus(&OpusPacket::new(vec![0xf8, 0xff, 0xfe], Duration::from_millis(20)))
//!     .await?;
//! while let Some(event) = peer.next_event().await {
//!     println!("{event:?}");
//! }
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
pub use peer::{LivePeer, LivePeerBuilder, OpusPacket, PeerError, PeerEvent, ReceivedOpus};
#[cfg(not(target_family = "wasm"))]
pub use webrtc::peer_connection::RTCPeerConnectionState as PeerConnectionState;

/// The RTP payload type of the Opus track.
pub const OPUS_PAYLOAD_TYPE: u8 = 111;

/// The Opus RTP clock rate, in hertz.
pub const OPUS_CLOCK_RATE: u32 = 48_000;
