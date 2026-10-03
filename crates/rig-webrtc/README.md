# rig-webrtc

The WebRTC peer for [Rig](https://crates.io/crates/rig)'s GPT-Live realtime calls over a ChatGPT subscription.

`rig-core` owns the call protocol: `rig_core::providers::chatgpt::realtime` creates the call from an SDP offer and joins its control socket. This crate owns only the media: a `LivePeer` offers one sendrecv Opus track (48 kHz, payload type 111) and the ordered `oai-events` data channel, sends Opus packets the caller already encoded, and hands back received Opus packets and data-channel messages. It decodes no audio.

It uses [webrtc-rs](https://crates.io/crates/webrtc) `=0.20.3`, the version the Codex voice host pins, and is native-only. Through the `rig` facade it is `rig::webrtc`, behind the `webrtc` feature.

## Setup: choose a rustls provider

A call needs HTTPS to create it and a secure websocket for its control socket. The WebRTC stack enables rustls's ring provider. The `rig-core` `rustls` feature enables aws-lc-rs for `rig_core::http_client::ReqwestClient`, which is reqwest's `Client`. With both linked, rustls cannot select a provider automatically. The binary, not a library, owns that process-wide choice: install one provider before the first connection.

```toml
[dependencies]
rig-core = { version = "0.41", features = ["live-websocket", "rustls"] }
rig-tungstenite = "0.41"
rig-webrtc = "0.41"
rustls = { version = "0.23", default-features = false, features = ["aws_lc_rs", "std"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
rustls::crypto::aws_lc_rs::default_provider()
    .install_default()
    .map_err(|_| "another rustls provider is already installed")?;
```

The test `tests/tls_provider.rs` runs this setup and the composition below against local servers, and checks that both the HTTPS request and the secure websocket reach their TLS handshake without panicking.

## A call

```rust
use std::time::Duration;

use rig_core::providers::chatgpt::realtime;
use rig_core::providers::live_support::{LiveConfiguration, CallerIdentity};
use rig_webrtc::{LivePeer, OpusPacket, PeerEvent};

async fn call(identity: CallerIdentity) -> Result<(), Box<dyn std::error::Error>> {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| "another rustls provider is already installed")?;

    let provider = LiveConfiguration::subscription("access-token")
        .with_account_id("account-id")
        .with_caller_identity(identity);
    let calls = realtime::LiveCalls::new(provider)?;
    let peer = LivePeer::builder().build().await?;
    let offer_sdp = peer.offer().await?;
    let session = realtime::SessionConfig::new("Answer briefly.");
    let call = calls
        .create_call(&rig_core::http_client::ReqwestClient::builder().build()?, &offer_sdp, &session)
        .await?;
    peer.apply_answer(call.answer_sdp).await?;
    let mut control = calls
        .connect_control(&rig_tungstenite::TungsteniteClient::new(), call.call_id)
        .await?;

    // Media and control run side by side; the peer is shared by reference.
    let media = async {
        peer.send_opus(&OpusPacket::new(vec![0xf8, 0xff, 0xfe], Duration::from_millis(20)))
            .await?;
        while let Some(event) = peer.next_event().await {
            if let PeerEvent::Lost(lost) = event {
                eprintln!("the consumer fell behind: {lost:?}");
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    };
    let control_loop = async {
        while let Some(event) = control.next_event().await? {
            if let realtime::ServerEvent::DelegationCreated(created) = event {
                control
                    .append_delegation_context(&created.item.id, None, "All services are healthy.")
                    .await?;
                break;
            }
        }
        control.close().await?;
        peer.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    };
    tokio::try_join!(media, control_loop)?;
    Ok(())
}
```

## Isolated local peers

`LivePeerBuilder::with_multicast_dns_disabled()` opts out of multicast DNS
candidate resolution and the stack's multicast socket. Production defaults
are unchanged. This option is useful with numeric ICE candidates in an
isolated local peer; browser peers that advertise only mDNS candidate names
cannot connect through those names when resolution is disabled.

For a UDP-only loopback peer, use all four settings together:

```rust
let builder = rig_webrtc::LivePeer::builder()
    .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
    .with_tcp_addrs(Vec::new())
    .with_loopback_candidates(true)
    .with_multicast_dns_disabled();
```

The offline test answerer uses the same address restriction and disables mDNS
in its own setting engine. Local HTTP mock clients use `no_proxy()` so proxy
environment settings cannot redirect their requests outside loopback.

## Contract

**Sharing.** Every `LivePeer` method takes `&self`, and the peer is `Send + Sync`. One task can wait in `next_event` while others send audio or text, through shared references (`tokio::try_join!`) or an `Arc<LivePeer>`. Concurrent `next_event` calls take turns, and each event goes to one of them.

**Ending.** `next_event` returns `None` once the peer has ended and its queued events are taken. The peer ends when `close` completes or the connection reports `Closed` or `Failed`. A closed data channel alone does not end it; audio may still arrive. When a remote stops sending without a closing signal, the connection reports `Disconnected` after about 5 seconds and `Failed` after about 30 seconds of silence. A connection that never forms reports `Failed` after about 30 seconds of checking.

**Release.** `close` stops the WebRTC driver and releases its sockets. Dropping a peer without `close`, including one dropped by `?` after a failed negotiation, and dropping a `build` future partway, closes the connection in a task on the runtime that built the peer. If that runtime has shut down, its tasks, the driver among them, are already gone.

**Delivery limits.** Received audio and data-channel messages share one bounded queue (256 events by default, `LivePeerBuilder::with_event_queue`). The peer's readers never wait for the consumer: while the queue is full, new events are dropped, and the stream reports them in place as one `PeerEvent::Lost` with counts by kind, followed by the events that fit afterward. The data channel is ordered and reliable on the wire, but a consumer that falls behind loses messages here. A `Lost` whose `messages` count is not zero means server events are missing, so the application can no longer trust its view of the conversation and should end or resynchronize the call. webrtc-rs also buffers up to 256 messages per data channel and 256 packets per track before this queue, and drops overflow with only a log line; because the peer's readers never wait, that happens only if their tasks are starved of CPU time.
