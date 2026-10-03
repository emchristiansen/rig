//! An in-process answering peer for loopback calls.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use rig_webrtc::{LivePeer, LivePeerBuilder, OPUS_CLOCK_RATE, OPUS_PAYLOAD_TYPE, PeerEvent};
use rtc::rtp_transceiver::rtp_sender::{
    RTCRtpCodec, RTCRtpCodecParameters, RTCRtpCodingParameters, RTCRtpEncodingParameters,
    RtpCodecKind,
};
use tokio::sync::mpsc;
use webrtc::data_channel::{DataChannel, DataChannelEvent};
use webrtc::media_stream::MediaStreamTrack;
use webrtc::media_stream::track_local::static_sample::TrackLocalStaticSample;
use webrtc::media_stream::track_remote::{TrackRemote, TrackRemoteEvent};
use webrtc::peer_connection::{
    MediaEngine, PeerConnection, PeerConnectionBuilder, PeerConnectionEventHandler,
    RTCIceGatheringState, RTCSessionDescription, SettingEngine,
};

pub const WAIT: Duration = Duration::from_secs(20);
pub const ANSWERER_SSRC: u32 = 4242;

/// What the answering peer saw.
pub enum Seen {
    /// An Opus payload from the caller's track.
    Audio(Vec<u8>),
    /// The caller's event channel.
    Channel(Arc<dyn DataChannel>),
    /// A message on that channel.
    Text(String),
}

struct Answerer {
    gathered: Arc<tokio::sync::Notify>,
    seen: mpsc::UnboundedSender<Seen>,
}

impl PeerConnectionEventHandler for Answerer {
    fn on_ice_gathering_state_change<'a, 'async_trait>(
        &'a self,
        state: RTCIceGatheringState,
    ) -> BoxFuture<'async_trait, ()>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            if state == RTCIceGatheringState::Complete {
                self.gathered.notify_one();
            }
        })
    }

    fn on_track<'a, 'async_trait>(
        &'a self,
        track: Arc<dyn TrackRemote>,
    ) -> BoxFuture<'async_trait, ()>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        let seen = self.seen.clone();
        Box::pin(async move {
            tokio::spawn(async move {
                while let Some(event) = track.poll().await {
                    if let TrackRemoteEvent::OnRtpPacket(packet) = event {
                        let _ = seen.send(Seen::Audio(packet.payload.to_vec()));
                    }
                }
            });
        })
    }

    fn on_data_channel<'a, 'async_trait>(
        &'a self,
        channel: Arc<dyn DataChannel>,
    ) -> BoxFuture<'async_trait, ()>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        let seen = self.seen.clone();
        Box::pin(async move {
            let _ = seen.send(Seen::Channel(channel.clone()));
            tokio::spawn(async move {
                while let Some(event) = channel.poll().await {
                    if let DataChannelEvent::OnMessage(message) = event {
                        let _ = seen.send(Seen::Text(
                            String::from_utf8_lossy(&message.data).into_owned(),
                        ));
                    }
                }
            });
        })
    }
}

pub fn opus_codec() -> RTCRtpCodec {
    RTCRtpCodec {
        mime_type: "audio/opus".into(),
        clock_rate: OPUS_CLOCK_RATE,
        channels: 2,
        sdp_fmtp_line: "minptime=10;useinbandfec=1".into(),
        rtcp_feedback: vec![],
    }
}

pub async fn next_seen(seen: &mut mpsc::UnboundedReceiver<Seen>, what: &str) -> Seen {
    tokio::time::timeout(WAIT, seen.recv())
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
        .unwrap_or_else(|| panic!("the answerer stopped before {what}"))
}

/// The answering peer of a loopback call, with its own Opus track.
pub struct AnsweringPeer {
    pub connection: Arc<dyn PeerConnection>,
    pub track: Arc<TrackLocalStaticSample>,
    pub seen: mpsc::UnboundedReceiver<Seen>,
    pub answer: String,
}

/// Answer `offer` on 127.0.0.1, returning once candidate gathering completes.
pub async fn answer(offer: String) -> AnsweringPeer {
    install_crypto_provider();
    // The answering side: one Opus track of its own, loopback candidates.
    let mut media = MediaEngine::default();
    media
        .register_codec(
            RTCRtpCodecParameters {
                rtp_codec: opus_codec(),
                payload_type: OPUS_PAYLOAD_TYPE,
            },
            RtpCodecKind::Audio,
        )
        .expect("opus registers");
    let mut settings = SettingEngine::default();
    settings.set_include_loopback_candidate(true);
    settings.set_multicast_dns_mode(rtc::ice::mdns::MulticastDnsMode::Disabled);
    let gathered = Arc::new(tokio::sync::Notify::new());
    let (seen_tx, seen) = mpsc::unbounded_channel();
    let answerer: Arc<dyn PeerConnection> = Arc::new(
        PeerConnectionBuilder::new()
            .with_media_engine(media)
            .with_setting_engine(settings)
            .with_handler(Arc::new(Answerer {
                gathered: gathered.clone(),
                seen: seen_tx,
            }))
            .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
            .with_tcp_addrs(Vec::new())
            .build()
            .await
            .expect("the answerer builds"),
    );
    let track = Arc::new(
        TrackLocalStaticSample::new(MediaStreamTrack::new(
            "answer".into(),
            "answer-audio".into(),
            "speaker".into(),
            RtpCodecKind::Audio,
            vec![RTCRtpEncodingParameters {
                rtp_coding_parameters: RTCRtpCodingParameters {
                    ssrc: Some(ANSWERER_SSRC),
                    ..Default::default()
                },
                codec: opus_codec(),
                active: true,
                ..Default::default()
            }],
        ))
        .expect("the answer track builds"),
    );
    answerer
        .add_track(track.clone())
        .await
        .expect("the answer track attaches");
    answerer
        .set_remote_description(RTCSessionDescription::offer(offer).expect("an offer"))
        .await
        .expect("the offer applies");
    let answer = answerer.create_answer(None).await.expect("an answer");
    answerer
        .set_local_description(answer)
        .await
        .expect("the answer applies locally");
    tokio::time::timeout(WAIT, gathered.notified())
        .await
        .expect("the answerer gathers");
    let answer = answerer
        .local_description()
        .await
        .expect("a local answer")
        .sdp;

    AnsweringPeer {
        connection: answerer,
        track,
        seen,
        answer,
    }
}

/// A builder for a peer that gathers only UDP on 127.0.0.1:`port`.
pub fn loopback_builder(port: u16) -> LivePeerBuilder {
    install_crypto_provider();
    LivePeer::builder()
        .with_udp_addrs(vec![format!("127.0.0.1:{port}")])
        .with_tcp_addrs(Vec::new())
        .with_loopback_candidates(true)
        .with_multicast_dns_disabled()
}

/// Build `builder`'s peer and connect it to an in-process answering peer,
/// returning both and the answerer's end of the event channel.
pub async fn connect(builder: LivePeerBuilder) -> (LivePeer, AnsweringPeer, Arc<dyn DataChannel>) {
    let peer = builder.build().await.expect("the live peer builds");
    let offer = peer.offer().await.expect("an offer");
    let mut answering = answer(offer).await;
    peer.apply_answer(answering.answer.clone())
        .await
        .expect("the call connects");
    let Seen::Channel(channel) = next_seen(&mut answering.seen, "the event channel").await else {
        panic!("the channel arrives first");
    };
    (peer, answering, channel)
}

/// The next peer event that `matches`, skipping others.
pub async fn next_peer_event(peer: &LivePeer, matches: impl Fn(&PeerEvent) -> bool) -> PeerEvent {
    tokio::time::timeout(WAIT, async {
        loop {
            match peer.next_event().await {
                Some(event) if matches(&event) => return event,
                Some(_) => continue,
                None => panic!("the peer's events ended"),
            }
        }
    })
    .await
    .expect("the expected peer event arrives")
}

/// Install aws-lc-rs as the process-wide rustls provider, once, exactly as
/// the crate documentation tells a binary to.
pub fn install_crypto_provider() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        // Another test in the same binary may have installed it first.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}
