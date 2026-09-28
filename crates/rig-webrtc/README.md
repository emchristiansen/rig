# rig-webrtc

The WebRTC peer for [Rig](https://crates.io/crates/rig)'s GPT-Live realtime calls over a ChatGPT subscription.

`rig-core` owns the call protocol: `rig_core::providers::chatgpt::realtime` creates the call from an SDP offer and joins its control socket. This crate owns only the media: a `LivePeer` offers one sendrecv Opus track (48 kHz, payload type 111) and the ordered `oai-events` data channel, sends Opus packets the caller already encoded, and hands back received Opus packets and data-channel messages. It decodes no audio.

It uses [webrtc-rs](https://crates.io/crates/webrtc) `=0.20.3`, the version the Codex voice host pins, and is native-only. Through the `rig` facade it is `rig::webrtc`, behind the `webrtc` feature.

```rust
use rig_webrtc::{LivePeer, OpusPacket};
use std::time::Duration;

let mut peer = LivePeer::builder().build().await?;
let offer_sdp = peer.offer().await?;
// Create the call with `offer_sdp`, then apply the answer it returns.
peer.apply_answer(answer_sdp).await?;
peer.send_opus(&OpusPacket::new(opus_frame, Duration::from_millis(20))).await?;
while let Some(event) = peer.next_event().await {
    // PeerEvent::Audio, PeerEvent::Text, ...
}
```
