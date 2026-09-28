//! A [`LivePeer`] negotiates with an in-process answering peer over
//! loopback and exchanges Opus packets and event-channel messages both ways.
//! No network beyond 127.0.0.1 is used.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

mod common;

use std::time::Duration;

use common::{ANSWERER_SSRC, Seen, WAIT, answer, next_seen};
use rig_webrtc::{LivePeer, OPUS_PAYLOAD_TYPE, OpusPacket, PeerEvent};
use rtc::media::Sample;

async fn next_peer_event(peer: &mut LivePeer, matches: impl Fn(&PeerEvent) -> bool) -> PeerEvent {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn opus_and_event_messages_cross_a_loopback_call_both_ways() {
    let mut peer = LivePeer::builder()
        .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
        .with_tcp_addrs(Vec::new())
        .with_loopback_candidates(true)
        .build()
        .await
        .expect("the live peer builds");
    let offer = peer.offer().await.expect("an offer");
    assert!(offer.contains("a=rtpmap:111 opus/48000/2"), "{offer}");
    assert!(offer.contains("webrtc-datachannel"), "{offer}");
    assert!(offer.contains("a=sendrecv"), "{offer}");

    let common::AnsweringPeer {
        connection: answerer,
        track: reply_track,
        mut seen,
        answer,
    } = answer(offer).await;
    peer.apply_answer(answer).await.expect("the call connects");

    // Caller to answerer: Opus packets and one event message.
    let Seen::Channel(channel) = next_seen(&mut seen, "the event channel").await else {
        panic!("the channel arrives first");
    };
    assert_eq!(channel.label().await.expect("a label"), "oai-events");
    assert!(channel.ordered().await.expect("ordered"));
    peer.send_text(r#"{"type":"session.close"}"#)
        .await
        .expect("the message is sent");
    let sent = [vec![0xf8, 0xff, 0xfe], vec![0xf8, 0x01, 0x02, 0x03]];
    let mut received_audio = Vec::new();
    let mut received_text = None;
    for attempt in 0..100 {
        let packet = &sent[attempt % sent.len()];
        peer.send_opus(&OpusPacket::new(packet.clone(), Duration::from_millis(20)))
            .await
            .expect("the packet is sent");
        tokio::time::sleep(Duration::from_millis(20)).await;
        while let Ok(event) = seen.try_recv() {
            match event {
                Seen::Audio(payload) => received_audio.push(payload),
                Seen::Text(text) => received_text = Some(text),
                Seen::Channel(_) => {}
            }
        }
        if received_audio.len() >= 2 && received_text.is_some() {
            break;
        }
    }
    assert_eq!(
        received_text.as_deref(),
        Some(r#"{"type":"session.close"}"#)
    );
    assert!(
        received_audio.iter().all(|payload| sent.contains(payload)),
        "only sent packets arrive, intact: {received_audio:?}"
    );
    assert!(received_audio.len() >= 2, "{received_audio:?}");

    // Answerer to caller: an event message and Opus packets.
    channel
        .send_text(r#"{"type":"turn.done"}"#)
        .await
        .expect("the answerer sends a message");
    let text = next_peer_event(&mut peer, |event| matches!(event, PeerEvent::Text(_))).await;
    assert_eq!(text, PeerEvent::Text(r#"{"type":"turn.done"}"#.to_owned()));

    let writer = tokio::spawn(async move {
        for _ in 0..100 {
            let sample = Sample {
                data: bytes::Bytes::from_static(&[0xfc, 0x7a, 0x55]),
                duration: Duration::from_millis(20),
                ..Default::default()
            };
            if reply_track
                .write_sample(ANSWERER_SSRC, OPUS_PAYLOAD_TYPE, &sample, &[])
                .await
                .is_err()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let audio = next_peer_event(&mut peer, |event| matches!(event, PeerEvent::Audio(_))).await;
    let PeerEvent::Audio(audio) = audio else {
        panic!("audio");
    };
    assert_eq!(audio.payload.as_ref(), [0xfc, 0x7a, 0x55]);
    assert_eq!(audio.ssrc, ANSWERER_SSRC);
    writer.abort();

    peer.close().await.expect("the peer closes");
    answerer.close().await.expect("the answerer closes");
}
