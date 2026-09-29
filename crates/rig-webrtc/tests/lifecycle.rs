//! How a [`LivePeer`] ends: dropping it releases its socket, its event stream
//! ends after the peer does, and a paused consumer is told what it lost.
//! Sockets are observed passively in `/proc/net/udp`, so the checks never
//! compete with the peer for its port. No network beyond 127.0.0.1 is used.

#![cfg(target_os = "linux")]
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

mod common;

use std::num::NonZeroUsize;
use std::time::Duration;

use common::{WAIT, connect, loopback_builder};
use rig_webrtc::{LostEvents, PeerConnectionState, PeerEvent, PeerError};

/// A UDP port on 127.0.0.1 that nothing holds right now.
fn free_udp_port() -> u16 {
    std::net::UdpSocket::bind("127.0.0.1:0")
        .expect("a local port")
        .local_addr()
        .expect("its address")
        .port()
}

/// Whether a UDP socket is bound to 127.0.0.1:`port`.
fn udp_port_bound(port: u16) -> bool {
    let table = std::fs::read_to_string("/proc/net/udp").expect("the UDP socket table");
    let local = format!("0100007F:{port:04X}");
    table
        .lines()
        .skip(1)
        .any(|line| line.split_whitespace().nth(1) == Some(local.as_str()))
}

/// Take events until the stream ends, which must happen within the bound.
async fn drain(peer: &rig_webrtc::LivePeer) -> Vec<PeerEvent> {
    let started = std::time::Instant::now();
    let mut events = Vec::new();
    loop {
        match tokio::time::timeout(WAIT.saturating_sub(started.elapsed()), peer.next_event()).await {
            Ok(Some(event)) => {
                eprintln!("DIAG {:?} {event:?}", started.elapsed());
                events.push(event);
            }
            Ok(None) => return events,
            Err(_) => panic!("the event stream did not end: {events:?}"),
        }
    }
}

/// A runtime of its own, so that its live tasks are the peer's alone.
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("a runtime")
}

/// Wait until every task on `runtime` has finished, the WebRTC driver
/// included, then check that `port` is free again.
fn assert_everything_released(runtime: &tokio::runtime::Runtime, port: u16) {
    let metrics = runtime.metrics();
    runtime.block_on(async {
        tokio::time::timeout(WAIT, async {
            while metrics.num_alive_tasks() > 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{} tasks still running after {WAIT:?}",
                metrics.num_alive_tasks()
            )
        });
    });
    assert!(!udp_port_bound(port), "127.0.0.1:{port} is released");
}

/// An answer from a peer that has already closed.
fn stale_answer(offer: String) -> String {
    runtime().block_on(async {
        let answering = common::answer(offer).await;
        answering
            .connection
            .close()
            .await
            .expect("the answerer closes");
        answering.answer
    })
}

#[test]
fn dropping_a_peer_without_close_releases_its_driver_and_socket() {
    let runtime = runtime();
    let port = free_udp_port();
    runtime.block_on(async {
        let peer = loopback_builder(port).build().await.expect("builds");
        peer.offer().await.expect("an offer");
        assert!(udp_port_bound(port), "the peer holds its socket");
        drop(peer);
    });
    assert_everything_released(&runtime, port);
}

#[test]
fn a_failed_negotiation_that_drops_the_peer_releases_its_driver_and_socket() {
    async fn negotiate(port: u16) -> Result<(), PeerError> {
        let peer = loopback_builder(port)
            .with_timeout(Duration::from_secs(1))
            .build()
            .await?;
        let offer = peer.offer().await?;
        // The answerer is gone, so the event channel never opens.
        let answer = tokio::task::spawn_blocking(move || stale_answer(offer))
            .await
            .expect("the answerer ran");
        peer.apply_answer(answer).await
    }
    let runtime = runtime();
    let port = free_udp_port();
    let outcome = runtime.block_on(negotiate(port));
    assert!(
        matches!(outcome, Err(PeerError::TimedOut(_))),
        "{outcome:?}"
    );
    assert_everything_released(&runtime, port);
}

#[test]
fn a_build_dropped_part_way_releases_its_driver_and_socket() {
    let runtime = runtime();
    let port = free_udp_port();
    // One poll starts the construction; dropping the future then abandons it.
    let abandoned = runtime.block_on(async {
        tokio::time::timeout(Duration::ZERO, loopback_builder(port).build()).await
    });
    assert!(abandoned.is_err(), "the build did not finish in one poll");
    assert_everything_released(&runtime, port);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn after_close_the_queued_events_drain_and_the_stream_ends() {
    let (peer, answering, _channel) = connect(loopback_builder(0)).await;
    // Nothing has been taken yet: the connection's own events are queued.
    peer.close().await.expect("closes");
    let events = drain(&peer).await;
    assert!(
        events.contains(&PeerEvent::Connection(PeerConnectionState::Connected)),
        "{events:?}"
    );
    assert_eq!(peer.next_event().await, None, "the end is final");
    peer.close().await.expect("a second close is harmless");
    answering.connection.close().await.expect("the answerer closes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_stream_ends_after_the_remote_peer_closes() {
    let (peer, answering, _channel) = connect(loopback_builder(0)).await;
    answering.connection.close().await.expect("the answerer closes");
    let events = drain(&peer).await;
    let last_state = events.iter().rev().find_map(|event| match event {
        PeerEvent::Connection(state) => Some(*state),
        _ => None,
    });
    assert!(
        matches!(
            last_state,
            Some(PeerConnectionState::Closed | PeerConnectionState::Failed)
        ),
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_stream_ends_when_the_connection_fails() {
    let peer = loopback_builder(0)
        .with_timeout(Duration::from_secs(1))
        .build()
        .await
        .expect("builds");
    let offer = peer.offer().await.expect("an offer");
    let answering = common::answer(offer).await;
    answering.connection.close().await.expect("the answerer closes");
    let applied = peer.apply_answer(answering.answer).await;
    assert!(matches!(applied, Err(PeerError::TimedOut(_))), "{applied:?}");
    let events = drain(&peer).await;
    assert_eq!(
        events.last(),
        Some(&PeerEvent::Connection(PeerConnectionState::Failed)),
        "{events:?}"
    );
}

/// With the consumer paused, data-channel messages beyond the queue are
/// dropped, and the stream says so in place: the queued messages, then one
/// `Lost` counting the rest. Every sent message is accounted for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_paused_consumer_is_told_how_many_messages_it_lost() {
    const QUEUE: usize = 8;
    const SENT: usize = 64;
    let queue = NonZeroUsize::new(QUEUE).expect("nonzero");
    let (peer, answering, channel) =
        connect(loopback_builder(0).with_event_queue(queue)).await;
    // Take the connection's own events, so the queue starts empty.
    loop {
        match common::next_peer_event(&peer, |_| true).await {
            PeerEvent::ChannelOpen | PeerEvent::Lost(_) => break,
            _ => {}
        }
    }

    for index in 0..SENT {
        channel
            .send_text(&format!("m{index}"))
            .await
            .expect("the answerer sends");
    }
    // Every message is acknowledged, so the peer's stack has received all
    // of them while nobody was reading.
    tokio::time::timeout(WAIT, async {
        while channel.outstanding_bytes().await.expect("the channel is open") > 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the messages are acknowledged");

    let mut texts = Vec::new();
    let mut lost = Vec::new();
    tokio::time::timeout(WAIT, async {
        while texts.len() + lost.iter().map(|lost: &LostEvents| lost.messages as usize).sum::<usize>()
            < SENT
        {
            match peer.next_event().await.expect("the stream continues") {
                PeerEvent::Text(text) => texts.push(text),
                PeerEvent::Lost(counts) => {
                    assert_eq!(counts.audio, 0);
                    lost.push(counts);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    })
    .await
    .expect("every message is delivered or counted as lost");

    // The queued messages come first, in order; a count of the ones that did
    // not fit follows them; whatever the stack still held arrives after it.
    let expected: Vec<String> = (0..QUEUE).map(|index| format!("m{index}")).collect();
    assert_eq!(texts[..QUEUE], expected[..], "{texts:?}");
    assert!(lost.first().is_some_and(|lost| lost.messages > 0), "{lost:?}");
    let indices: Vec<usize> = texts
        .iter()
        .map(|text| text[1..].parse().expect("an index"))
        .collect();
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]), "{indices:?}");

    peer.close().await.expect("closes");
    answering.connection.close().await.expect("the answerer closes");
}
