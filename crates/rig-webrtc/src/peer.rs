//! The offering peer: an Opus track, the event data channel, and the events
//! they deliver.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;
use rig_core::providers::chatgpt::realtime::EVENTS_DATA_CHANNEL;
use rtc::media::Sample;
use rtc::rtp_transceiver::rtp_sender::{
    RTCRtpCodec, RTCRtpCodecParameters, RTCRtpCodingParameters, RTCRtpEncodingParameters,
    RtpCodecKind,
};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinHandle;
use webrtc::data_channel::{DataChannel, DataChannelEvent, RTCDataChannelInit};
use webrtc::media_stream::MediaStreamTrack;
use webrtc::media_stream::track_local::static_sample::TrackLocalStaticSample;
use webrtc::media_stream::track_remote::{TrackRemote, TrackRemoteEvent};
use webrtc::peer_connection::{
    MediaEngine, PeerConnection, PeerConnectionBuilder, PeerConnectionEventHandler,
    RTCIceCandidateInit, RTCIceGatheringState, RTCPeerConnectionState, RTCSessionDescription,
    SettingEngine,
};

use crate::{OPUS_CLOCK_RATE, OPUS_PAYLOAD_TYPE};

/// The most remote candidates an answer may carry.
const MAX_REMOTE_CANDIDATES: usize = 32;

/// How many events wait for the caller before the peer's readers pause.
const EVENT_QUEUE: usize = 256;

/// A peer failure.
#[derive(Debug, thiserror::Error)]
pub enum PeerError {
    /// The WebRTC stack refused an operation.
    #[error("webrtc: {0}")]
    Webrtc(#[from] webrtc::error::Error),
    /// Negotiation did not finish within the builder's timeout.
    #[error("timed out {0}")]
    TimedOut(&'static str),
    /// The answer carries more than 32 ICE candidates.
    #[error("the answer carries more than {MAX_REMOTE_CANDIDATES} ICE candidates")]
    TooManyCandidates,
    /// The answer carries a candidate that does not parse.
    #[error("the answer carries an invalid ICE candidate: {0}")]
    InvalidCandidate(String),
    /// The event data channel closed before it opened.
    #[error("the event data channel closed before it opened")]
    ChannelClosed,
    /// The peer has no local description after gathering.
    #[error("the peer has no local description")]
    MissingOffer,
}

/// One encoded Opus packet to send, and how much audio it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpusPacket {
    /// The Opus packet, exactly as encoded.
    pub data: Bytes,
    /// The audio it holds; it advances the RTP timestamp.
    pub duration: Duration,
}

impl OpusPacket {
    /// A packet holding `duration` of audio.
    pub fn new(data: impl Into<Bytes>, duration: Duration) -> Self {
        Self {
            data: data.into(),
            duration,
        }
    }
}

/// One Opus packet received from the remote track, with its RTP header
/// fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReceivedOpus {
    /// The Opus packet.
    pub payload: Bytes,
    /// The RTP sequence number.
    pub sequence_number: u16,
    /// The RTP timestamp, at [`OPUS_CLOCK_RATE`].
    pub timestamp: u32,
    /// The sending stream.
    pub ssrc: u32,
}

/// Something the peer received or observed.
#[derive(Clone, Debug, PartialEq)]
pub enum PeerEvent {
    /// An Opus packet from the remote track.
    Audio(ReceivedOpus),
    /// A text message on the event data channel: one JSON server event.
    Text(String),
    /// A binary message on the event data channel.
    Binary(Bytes),
    /// The event data channel opened.
    ChannelOpen,
    /// The event data channel closed.
    ChannelClosed,
    /// The peer connection changed state.
    Connection(RTCPeerConnectionState),
}

/// Configures a [`LivePeer`].
#[derive(Clone, Debug)]
pub struct LivePeerBuilder {
    udp_addrs: Vec<String>,
    tcp_addrs: Vec<String>,
    loopback_candidates: bool,
    timeout: Duration,
}

impl Default for LivePeerBuilder {
    fn default() -> Self {
        Self {
            udp_addrs: vec!["0.0.0.0:0".to_owned(), "[::]:0".to_owned()],
            tcp_addrs: vec!["0.0.0.0:0".to_owned(), "[::]:0".to_owned()],
            loopback_candidates: false,
            timeout: Duration::from_secs(15),
        }
    }
}

impl LivePeerBuilder {
    /// Gather UDP candidates on `addrs` instead of every IPv4 and IPv6
    /// interface.
    #[must_use]
    pub fn with_udp_addrs(mut self, addrs: Vec<String>) -> Self {
        self.udp_addrs = addrs;
        self
    }

    /// Gather TCP candidates on `addrs` instead of every IPv4 and IPv6
    /// interface; empty gathers none.
    #[must_use]
    pub fn with_tcp_addrs(mut self, addrs: Vec<String>) -> Self {
        self.tcp_addrs = addrs;
        self
    }

    /// Gather loopback candidates, which are skipped by default.
    #[must_use]
    pub fn with_loopback_candidates(mut self, include: bool) -> Self {
        self.loopback_candidates = include;
        self
    }

    /// Bound candidate gathering, and the wait for the event channel after an
    /// answer, by `timeout` (15 seconds by default). ICE keeps checking
    /// candidates for as long.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build the peer: its Opus track and its ordered event data channel.
    /// Needs a running tokio runtime.
    pub async fn build(self) -> Result<LivePeer, PeerError> {
        let ssrc = fastrand::u32(..);
        let codec = RTCRtpCodec {
            mime_type: "audio/opus".into(),
            clock_rate: OPUS_CLOCK_RATE,
            channels: 2,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".into(),
            rtcp_feedback: vec![],
        };
        let mut media = MediaEngine::default();
        media.register_codec(
            RTCRtpCodecParameters {
                rtp_codec: codec.clone(),
                payload_type: OPUS_PAYLOAD_TYPE,
            },
            RtpCodecKind::Audio,
        )?;
        let track = Arc::new(TrackLocalStaticSample::new(MediaStreamTrack::new(
            "realtime".into(),
            format!("audio-{ssrc}"),
            "microphone".into(),
            RtpCodecKind::Audio,
            vec![RTCRtpEncodingParameters {
                rtp_coding_parameters: RTCRtpCodingParameters {
                    ssrc: Some(ssrc),
                    ..Default::default()
                },
                codec,
                active: true,
                ..Default::default()
            }],
        ))?);

        let check_interval = Duration::from_millis(200);
        let attempts = u16::try_from(self.timeout.as_millis() / check_interval.as_millis())
            .unwrap_or(u16::MAX)
            .max(1);
        let mut settings = SettingEngine::default();
        settings.set_ice_connection_attempts(Some(check_interval), Some(attempts));
        settings.set_include_loopback_candidate(self.loopback_candidates);

        let (events, receiver) = mpsc::channel(EVENT_QUEUE);
        let gathered = Arc::new(Notify::new());
        let connection: Arc<dyn PeerConnection> = Arc::new(
            PeerConnectionBuilder::new()
                .with_media_engine(media)
                .with_setting_engine(settings)
                .with_handler(Arc::new(Handler {
                    gathered: gathered.clone(),
                    events: events.clone(),
                }))
                .with_udp_addrs(self.udp_addrs)
                .with_tcp_addrs(self.tcp_addrs)
                .build()
                .await?,
        );
        let channel = match async {
            connection.add_track(track.clone()).await?;
            connection
                .create_data_channel(
                    EVENTS_DATA_CHANNEL,
                    Some(RTCDataChannelInit {
                        ordered: true,
                        ..Default::default()
                    }),
                )
                .await
        }
        .await
        {
            Ok(channel) => channel,
            Err(error) => {
                let _ = connection.close().await;
                return Err(error.into());
            }
        };
        let (open_sender, open) = watch::channel(false);
        let observer = tokio::spawn(observe_channel(channel.clone(), open_sender, events));
        Ok(LivePeer {
            connection,
            track,
            ssrc,
            channel,
            gathered,
            open,
            events: receiver,
            observer,
            timeout: self.timeout,
        })
    }
}

/// The offering WebRTC peer of one GPT-Live call.
///
/// Send and receive may run concurrently: sending takes `&self`, and only
/// [`Self::next_event`] takes `&mut self`.
pub struct LivePeer {
    connection: Arc<dyn PeerConnection>,
    track: Arc<TrackLocalStaticSample>,
    ssrc: u32,
    channel: Arc<dyn DataChannel>,
    gathered: Arc<Notify>,
    open: watch::Receiver<bool>,
    events: mpsc::Receiver<PeerEvent>,
    observer: JoinHandle<()>,
    timeout: Duration,
}

impl LivePeer {
    /// A builder with the defaults: every IPv4 and IPv6 interface for UDP and
    /// TCP, no loopback candidates, a 15 second timeout.
    #[must_use]
    pub fn builder() -> LivePeerBuilder {
        LivePeerBuilder::default()
    }

    /// Create the offer, wait until candidate gathering completes, and
    /// return the offer SDP with every candidate in it.
    pub async fn offer(&self) -> Result<String, PeerError> {
        tokio::time::timeout(self.timeout, async {
            let offer = self.connection.create_offer(None).await?;
            self.connection.set_local_description(offer).await?;
            self.gathered.notified().await;
            self.connection
                .local_description()
                .await
                .map(|offer| offer.sdp)
                .ok_or(PeerError::MissingOffer)
        })
        .await
        .map_err(|_| PeerError::TimedOut("gathering ICE candidates"))?
    }

    /// Apply the call's answer and wait until the event data channel opens.
    ///
    /// The answer's candidates are added explicitly after it is applied,
    /// which the async driver needs to start TCP dialing. An answer with more
    /// than 32 candidates, or one that does not parse, is refused before it
    /// changes the peer.
    pub async fn apply_answer(&self, sdp: String) -> Result<(), PeerError> {
        let answer = RTCSessionDescription::answer(sdp)?;
        let candidates = answer_candidates(&answer)?;
        tokio::time::timeout(self.timeout, async {
            self.connection.set_remote_description(answer).await?;
            for candidate in candidates {
                self.connection.add_ice_candidate(candidate).await?;
            }
            self.open
                .clone()
                .wait_for(|open| *open)
                .await
                .map(|_| ())
                .map_err(|_| PeerError::ChannelClosed)
        })
        .await
        .map_err(|_| PeerError::TimedOut("waiting for the event data channel to open"))?
    }

    /// Send one Opus packet on the audio track.
    pub async fn send_opus(&self, packet: &OpusPacket) -> Result<(), PeerError> {
        let sample = Sample {
            data: packet.data.clone(),
            duration: packet.duration,
            ..Default::default()
        };
        self.track
            .write_sample(self.ssrc, OPUS_PAYLOAD_TYPE, &sample, &[])
            .await?;
        Ok(())
    }

    /// Send one text message on the event data channel.
    pub async fn send_text(&self, text: &str) -> Result<(), PeerError> {
        self.channel.send_text(text).await?;
        Ok(())
    }

    /// The next event, in arrival order per source. `None` once the peer is
    /// closed and every event has been taken.
    pub async fn next_event(&mut self) -> Option<PeerEvent> {
        self.events.recv().await
    }

    /// Close the peer connection.
    pub async fn close(self) -> Result<(), PeerError> {
        self.observer.abort();
        self.connection.close().await?;
        Ok(())
    }
}

impl Drop for LivePeer {
    fn drop(&mut self) {
        self.observer.abort();
    }
}

impl std::fmt::Debug for LivePeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LivePeer")
            .field("ssrc", &self.ssrc)
            .field("open", &*self.open.borrow())
            .finish_non_exhaustive()
    }
}

/// The answer's component-1 candidates, deduplicated, counting every
/// occurrence toward the limit.
fn answer_candidates(
    answer: &RTCSessionDescription,
) -> Result<Vec<RTCIceCandidateInit>, PeerError> {
    let parsed = answer.unmarshal()?;
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (index, attribute) in parsed
        .media_descriptions
        .iter()
        .flat_map(|media| &media.attributes)
        .filter(|attribute| attribute.key == "candidate")
        .enumerate()
    {
        if index >= MAX_REMOTE_CANDIDATES {
            return Err(PeerError::TooManyCandidates);
        }
        let candidate = attribute
            .value
            .as_deref()
            .ok_or_else(|| PeerError::InvalidCandidate(String::new()))?;
        let parsed = rtc::ice::candidate::unmarshal_candidate(candidate)
            .map_err(|_| PeerError::InvalidCandidate(candidate.to_owned()))?;
        if parsed.component() == 1 && seen.insert(candidate) {
            candidates.push(RTCIceCandidateInit {
                candidate: format!("candidate:{candidate}"),
                ..Default::default()
            });
        }
    }
    Ok(candidates)
}

/// Forward the event channel's lifecycle and messages until it closes.
async fn observe_channel(
    channel: Arc<dyn DataChannel>,
    open: watch::Sender<bool>,
    events: mpsc::Sender<PeerEvent>,
) {
    while let Some(event) = channel.poll().await {
        let forwarded = match event {
            DataChannelEvent::OnOpen => {
                open.send_replace(true);
                PeerEvent::ChannelOpen
            }
            DataChannelEvent::OnMessage(message) if message.is_string => {
                PeerEvent::Text(String::from_utf8_lossy(&message.data).into_owned())
            }
            DataChannelEvent::OnMessage(message) => PeerEvent::Binary(message.data.freeze()),
            DataChannelEvent::OnError | DataChannelEvent::OnClosing | DataChannelEvent::OnClose => {
                break;
            }
            DataChannelEvent::OnBufferedAmountLow | DataChannelEvent::OnBufferedAmountHigh => {
                continue;
            }
        };
        if events.send(forwarded).await.is_err() {
            return;
        }
    }
    open.send_replace(false);
    let _ = events.send(PeerEvent::ChannelClosed).await;
}

/// Forward a remote track's Opus packets until it ends.
async fn forward_track(track: Arc<dyn TrackRemote>, events: mpsc::Sender<PeerEvent>) {
    while let Some(event) = track.poll().await {
        match event {
            TrackRemoteEvent::OnRtpPacket(packet)
                if packet.header.payload_type == OPUS_PAYLOAD_TYPE =>
            {
                let audio = PeerEvent::Audio(ReceivedOpus {
                    payload: packet.payload,
                    sequence_number: packet.header.sequence_number,
                    timestamp: packet.header.timestamp,
                    ssrc: packet.header.ssrc,
                });
                if events.send(audio).await.is_err() {
                    return;
                }
            }
            TrackRemoteEvent::OnEnded => return,
            _ => {}
        }
    }
}

struct Handler {
    gathered: Arc<Notify>,
    events: mpsc::Sender<PeerEvent>,
}

// The upstream trait uses async-trait's boxed-future signature. Its callbacks
// run inside the connection's driver loop, so none of them waits on the caller.
impl PeerConnectionEventHandler for Handler {
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

    fn on_connection_state_change<'a, 'async_trait>(
        &'a self,
        state: RTCPeerConnectionState,
    ) -> BoxFuture<'async_trait, ()>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            // A full queue drops the state change rather than stall the driver.
            let _ = self.events.try_send(PeerEvent::Connection(state));
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
        Box::pin(async move {
            tokio::spawn(forward_track(track, self.events.clone()));
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
        Box::pin(async move {
            // Only the locally created event channel is used.
            let _ = channel.close().await;
        })
    }
}
