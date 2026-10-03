//! The offering peer: an Opus track, the event data channel, and the events
//! they deliver.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;
use rig_core::providers::chatgpt::realtime::EVENTS_DATA_CHANNEL;
use rtc::media::Sample;
use rtc::rtp_transceiver::rtp_sender::{
    RTCRtpCodec, RTCRtpCodecParameters, RTCRtpCodingParameters, RTCRtpEncodingParameters,
    RtpCodecKind,
};
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
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

/// How many events wait for the caller by default before new ones are
/// dropped.
const DEFAULT_EVENT_QUEUE: NonZeroUsize = match NonZeroUsize::new(256) {
    Some(capacity) => capacity,
    None => NonZeroUsize::MIN,
};

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
    /// The runtime shut down while the peer was being built.
    #[error("the runtime shut down while the peer was being built")]
    RuntimeShutDown,
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

/// How many events the peer dropped because its event queue was full, by
/// kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LostEvents {
    /// [`PeerEvent::Audio`] packets.
    pub audio: u64,
    /// [`PeerEvent::Text`] and [`PeerEvent::Binary`] data-channel messages.
    pub messages: u64,
    /// [`PeerEvent::ChannelOpen`], [`PeerEvent::ChannelClosed`] and
    /// [`PeerEvent::Connection`] events.
    pub lifecycle: u64,
}

impl LostEvents {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    fn count(&mut self, event: &PeerEvent) {
        let counter = match event {
            PeerEvent::Audio(_) => &mut self.audio,
            PeerEvent::Text(_) | PeerEvent::Binary(_) => &mut self.messages,
            PeerEvent::ChannelOpen | PeerEvent::ChannelClosed | PeerEvent::Connection(_) => {
                &mut self.lifecycle
            }
            PeerEvent::Lost(_) => return,
        };
        *counter = counter.saturating_add(1);
    }
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
    /// The event data channel closed. Audio may still arrive; the event
    /// stream ends only when the whole peer does.
    ChannelClosed,
    /// The peer connection changed state.
    Connection(RTCPeerConnectionState),
    /// Events were dropped at this point in the stream because the queue was
    /// full. The events before it and after it were delivered. A nonzero
    /// [`LostEvents::messages`] means server events are missing, so the
    /// caller's view of the conversation is incomplete.
    Lost(LostEvents),
}

/// Configures a [`LivePeer`].
#[derive(Clone, Debug)]
pub struct LivePeerBuilder {
    udp_addrs: Vec<String>,
    tcp_addrs: Vec<String>,
    loopback_candidates: bool,
    multicast_dns_disabled: bool,
    timeout: Duration,
    event_queue: NonZeroUsize,
}

impl Default for LivePeerBuilder {
    fn default() -> Self {
        Self {
            udp_addrs: vec!["0.0.0.0:0".to_owned(), "[::]:0".to_owned()],
            tcp_addrs: vec!["0.0.0.0:0".to_owned(), "[::]:0".to_owned()],
            loopback_candidates: false,
            multicast_dns_disabled: false,
            timeout: Duration::from_secs(15),
            event_queue: DEFAULT_EVENT_QUEUE,
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

    /// Disable multicast DNS candidate resolution and its multicast socket.
    ///
    /// Opt in when every remote ICE candidate is a numeric address, such as
    /// an isolated loopback test. Browser peers may advertise only mDNS names;
    /// those candidates cannot be resolved with this option. The default keeps
    /// the WebRTC stack's multicast DNS behavior.
    #[must_use]
    pub fn with_multicast_dns_disabled(mut self) -> Self {
        self.multicast_dns_disabled = true;
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

    /// Hold at most `capacity` undelivered events (256 by default). While
    /// the queue is full, new events are dropped and reported as one
    /// [`PeerEvent::Lost`].
    #[must_use]
    pub fn with_event_queue(mut self, capacity: NonZeroUsize) -> Self {
        self.event_queue = capacity;
        self
    }

    /// Build the peer: its Opus track and its ordered event data channel.
    /// Needs a running tokio runtime.
    ///
    /// The construction runs as its own task, so dropping this future part
    /// way still closes whatever connection it created.
    pub async fn build(self) -> Result<LivePeer, PeerError> {
        match tokio::spawn(self.construct()).await {
            Ok(built) => built,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(_) => Err(PeerError::RuntimeShutDown),
        }
    }

    async fn construct(self) -> Result<LivePeer, PeerError> {
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
        if self.multicast_dns_disabled {
            settings.set_multicast_dns_mode(rtc::ice::mdns::MulticastDnsMode::Disabled);
        }

        let (sink, events) = event_queue(self.event_queue);
        let ended = Arc::new(watch::Sender::new(false));
        let gathered = Arc::new(Notify::new());
        let owner = ConnectionOwner {
            connection: Arc::new(
                PeerConnectionBuilder::new()
                    .with_media_engine(media)
                    .with_setting_engine(settings)
                    .with_handler(Arc::new(Handler {
                        gathered: gathered.clone(),
                        events: sink.clone(),
                        ended: ended.clone(),
                    }))
                    .with_udp_addrs(self.udp_addrs)
                    .with_tcp_addrs(self.tcp_addrs)
                    .build()
                    .await?,
            ),
            runtime: tokio::runtime::Handle::current(),
            ended,
            closed: AtomicBool::new(false),
        };
        owner.connection.add_track(track.clone()).await?;
        let channel = owner
            .connection
            .create_data_channel(
                EVENTS_DATA_CHANNEL,
                Some(RTCDataChannelInit {
                    ordered: true,
                    ..Default::default()
                }),
            )
            .await?;
        let (open_sender, open) = watch::channel(false);
        let observer = tokio::spawn(observe_channel(channel.clone(), open_sender, sink));
        Ok(LivePeer {
            owner,
            track,
            ssrc,
            channel,
            gathered,
            open,
            events: tokio::sync::Mutex::new(events),
            observer,
            timeout: self.timeout,
        })
    }
}

/// The offering WebRTC peer of one GPT-Live call.
///
/// Every method takes `&self`, so one task can wait in [`Self::next_event`]
/// while another sends, for example through an `Arc<LivePeer>` or
/// `tokio::join!`. Concurrent [`Self::next_event`] calls take turns, and each
/// event goes to one of them.
///
/// Dropping the peer without [`Self::close`] still closes the connection, in
/// a task on the runtime that built it, which stops the WebRTC driver and
/// releases its sockets.
pub struct LivePeer {
    owner: ConnectionOwner,
    track: Arc<TrackLocalStaticSample>,
    ssrc: u32,
    channel: Arc<dyn DataChannel>,
    gathered: Arc<Notify>,
    open: watch::Receiver<bool>,
    events: tokio::sync::Mutex<EventQueue>,
    observer: JoinHandle<()>,
    timeout: Duration,
}

impl LivePeer {
    /// A builder with the defaults: every IPv4 and IPv6 interface for UDP and
    /// TCP, no loopback candidates, a 15 second timeout, 256 queued events.
    #[must_use]
    pub fn builder() -> LivePeerBuilder {
        LivePeerBuilder::default()
    }

    /// Create the offer, wait until candidate gathering completes, and
    /// return the offer SDP with every candidate in it.
    pub async fn offer(&self) -> Result<String, PeerError> {
        let connection = &self.owner.connection;
        tokio::time::timeout(self.timeout, async {
            let offer = connection.create_offer(None).await?;
            connection.set_local_description(offer).await?;
            self.gathered.notified().await;
            connection
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
        let connection = &self.owner.connection;
        tokio::time::timeout(self.timeout, async {
            connection.set_remote_description(answer).await?;
            for candidate in candidates {
                connection.add_ice_candidate(candidate).await?;
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

    /// The next event, in arrival order per source.
    ///
    /// `None` once the peer has ended and every queued event has been
    /// taken. The peer ends when [`Self::close`] completes or the connection
    /// reaches [`RTCPeerConnectionState::Closed`] or
    /// [`RTCPeerConnectionState::Failed`]; a closed data channel alone does
    /// not end it. When a remote stops sending without a closing signal,
    /// the connection reports `Disconnected` after about 5 seconds and
    /// `Failed` after about 30 seconds of silence. A connection that never
    /// forms reports `Failed` after about 30 seconds of checking. Cancel-safe:
    /// dropping the future loses no event.
    pub async fn next_event(&self) -> Option<PeerEvent> {
        let mut events = self.events.lock().await;
        let mut ended = self.owner.ended.subscribe();
        loop {
            if *ended.borrow_and_update() {
                events.receiver.close();
            }
            match events.take_ready() {
                Ready::Event(event) => return Some(event),
                Ready::Ended => return None,
                Ready::Empty => {}
            }
            if events.receiver.is_closed() {
                // Ended: only sends already under way can still arrive.
                if let Some(event) = events.receiver.recv().await {
                    return Some(event);
                }
                continue;
            }
            tokio::select! {
                biased;
                event = events.receiver.recv() => {
                    if let Some(event) = event {
                        return Some(event);
                    }
                }
                _ = ended.wait_for(|ended| *ended) => {}
            }
        }
    }

    /// Close the peer connection, which stops the WebRTC driver and releases
    /// its sockets, and end the event stream once the queued events are
    /// taken. A later call closes again, which the WebRTC stack treats as
    /// done.
    pub async fn close(&self) -> Result<(), PeerError> {
        self.owner.connection.close().await?;
        self.observer.abort();
        self.owner.closed.store(true, Ordering::Release);
        self.owner.ended.send_replace(true);
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
            .field("ended", &*self.owner.ended.borrow())
            .finish_non_exhaustive()
    }
}

/// Owns the connection: dropping it without a completed close closes the
/// connection in a task, because the upstream peer detaches its driver on
/// drop and the driver keeps the sockets.
struct ConnectionOwner {
    connection: Arc<dyn PeerConnection>,
    runtime: tokio::runtime::Handle,
    /// Whether the peer has ended; it also stops the audio forwarders.
    ended: Arc<watch::Sender<bool>>,
    closed: AtomicBool,
}

impl Drop for ConnectionOwner {
    fn drop(&mut self) {
        self.ended.send_replace(true);
        if !self.closed.load(Ordering::Acquire) {
            let connection = self.connection.clone();
            // A runtime that has shut down drops the task, and with it the
            // driver it ran.
            drop(self.runtime.spawn(async move {
                let _ = connection.close().await;
            }));
        }
    }
}

/// A bounded event queue whose producers never wait: an event that does not
/// fit is counted, and the count is queued as [`PeerEvent::Lost`] ahead of
/// the next event that fits.
fn event_queue(capacity: NonZeroUsize) -> (EventSink, EventQueue) {
    let (sender, receiver) = mpsc::channel(capacity.get());
    let lost = Arc::new(Mutex::new(LostEvents::default()));
    (
        EventSink {
            sender,
            lost: lost.clone(),
        },
        EventQueue { receiver, lost },
    )
}

fn lock(lost: &Mutex<LostEvents>) -> MutexGuard<'_, LostEvents> {
    lost.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Clone)]
struct EventSink {
    sender: mpsc::Sender<PeerEvent>,
    lost: Arc<Mutex<LostEvents>>,
}

impl EventSink {
    /// Queue `event`, or count it as lost. The lock orders every push
    /// against the consumer's check for pending losses.
    fn push(&self, event: PeerEvent) {
        let mut lost = lock(&self.lost);
        if !lost.is_empty() {
            match self.sender.try_send(PeerEvent::Lost(*lost)) {
                Ok(()) => *lost = LostEvents::default(),
                Err(TrySendError::Full(_)) => {
                    lost.count(&event);
                    return;
                }
                Err(TrySendError::Closed(_)) => return,
            }
        }
        if let Err(TrySendError::Full(event)) = self.sender.try_send(event) {
            lost.count(&event);
        }
    }
}

struct EventQueue {
    receiver: mpsc::Receiver<PeerEvent>,
    lost: Arc<Mutex<LostEvents>>,
}

enum Ready {
    Event(PeerEvent),
    Empty,
    Ended,
}

impl EventQueue {
    /// The next queued event, or the pending losses once the queue is
    /// empty, without waiting.
    fn take_ready(&mut self) -> Ready {
        let mut lost = lock(&self.lost);
        match self.receiver.try_recv() {
            Ok(event) => Ready::Event(event),
            Err(_) if !lost.is_empty() => Ready::Event(PeerEvent::Lost(std::mem::take(&mut *lost))),
            Err(TryRecvError::Empty) => Ready::Empty,
            Err(TryRecvError::Disconnected) => Ready::Ended,
        }
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
    events: EventSink,
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
        events.push(forwarded);
    }
    open.send_replace(false);
    events.push(PeerEvent::ChannelClosed);
}

/// Forward a remote track's Opus packets until it ends or the peer does.
async fn forward_track(
    track: Arc<dyn TrackRemote>,
    events: EventSink,
    mut ended: watch::Receiver<bool>,
) {
    loop {
        let event = tokio::select! {
            event = track.poll() => event,
            _ = ended.wait_for(|ended| *ended) => return,
        };
        match event {
            Some(TrackRemoteEvent::OnRtpPacket(packet))
                if packet.header.payload_type == OPUS_PAYLOAD_TYPE =>
            {
                events.push(PeerEvent::Audio(ReceivedOpus {
                    payload: packet.payload,
                    sequence_number: packet.header.sequence_number,
                    timestamp: packet.header.timestamp,
                    ssrc: packet.header.ssrc,
                }));
            }
            Some(TrackRemoteEvent::OnEnded) | None => return,
            Some(_) => {}
        }
    }
}

struct Handler {
    gathered: Arc<Notify>,
    events: EventSink,
    ended: Arc<watch::Sender<bool>>,
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
            self.events.push(PeerEvent::Connection(state));
            if matches!(
                state,
                RTCPeerConnectionState::Closed | RTCPeerConnectionState::Failed
            ) {
                self.ended.send_replace(true);
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
        Box::pin(async move {
            tokio::spawn(forward_track(
                track,
                self.events.clone(),
                self.ended.subscribe(),
            ));
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
