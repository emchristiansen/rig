//! One live GPT-Live call over a ChatGPT subscription: create the call from a
//! real offer, connect the peer, join the control socket, stream Opus silence
//! and read events until the session starts and reports usage, then close.
//!
//! Ignored, and inert unless `RIG_LIVE_ONE_LIVE=1`. It reads the credential
//! and the caller identity from the environment:
//! `CHATGPT_ACCESS_TOKEN`, `CHATGPT_ACCOUNT_ID`, `CHATGPT_ORIGINATOR`,
//! `CHATGPT_USER_AGENT` and, optionally, `CHATGPT_VERSION`. The session is
//! closed within `RIG_LIVE_ONE_SECONDS` (default 20, at most 55).

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use rig_core::providers::chatgpt::{self, realtime};
use rig_core::providers::openai::OpenAI;
use rig_core::providers::openai::wire::CallerIdentity;
use rig_webrtc::{LivePeer, OpusPacket};

/// One 20 ms Opus frame of silence.
const OPUS_SILENCE: [u8; 3] = [0xf8, 0xff, 0xfe];

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("`{name}` must be set for the live call"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "calls the live ChatGPT realtime backend; set RIG_LIVE_ONE_LIVE=1 and the CHATGPT_* credential variables"]
async fn a_live_call_starts_reports_usage_and_closes() {
    if std::env::var("RIG_LIVE_ONE_LIVE").as_deref() != Ok("1") {
        eprintln!("skipped: RIG_LIVE_ONE_LIVE is not 1");
        return;
    }
    let seconds: u64 = std::env::var("RIG_LIVE_ONE_SECONDS")
        .ok()
        .map(|value| value.parse().expect("RIG_LIVE_ONE_SECONDS is a number"))
        .unwrap_or(20)
        .min(55);
    let identity = CallerIdentity::new(
        env("CHATGPT_ORIGINATOR"),
        env("CHATGPT_USER_AGENT"),
        std::env::var("CHATGPT_VERSION").ok(),
    )
    .expect("a valid caller identity");
    let provider = OpenAI::with_key(&chatgpt::DIALECT, env("CHATGPT_ACCESS_TOKEN"))
        .with_account_id(env("CHATGPT_ACCOUNT_ID"))
        .with_caller_identity(identity);
    let calls = realtime::LiveCalls::new(provider).expect("the Codex backend");

    let peer = LivePeer::builder().build().await.expect("the peer builds");
    let offer = peer.offer().await.expect("an offer");
    let call = calls
        .create_call(
            &rig_reqwest::ReqwestClient::default(),
            &offer,
            &realtime::SessionConfig::new(
                "You are a brief test companion. Say nothing unless asked.",
            ),
        )
        .await
        .expect("the call is created");
    peer.apply_answer(call.answer_sdp.clone())
        .await
        .expect("the peer connects");
    let mut control = calls
        .connect_control(
            &rig_tungstenite::TungsteniteClient::new(),
            call.call_id.clone(),
        )
        .await
        .expect("the control socket opens");

    let silence = tokio::spawn(async move {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        while Instant::now() < deadline {
            let packet = OpusPacket::new(OPUS_SILENCE.to_vec(), Duration::from_millis(20));
            if peer.send_opus(&packet).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        peer
    });

    let opened = Instant::now();
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut started = None;
    let mut usage = None;
    while Instant::now() < deadline && (started.is_none() || usage.is_none()) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Ok(event) = tokio::time::timeout(remaining, control.next_event()).await else {
            break;
        };
        let event = event.expect("the control socket reads");
        eprintln!(
            "live event at {:?}: {}",
            opened.elapsed(),
            match &event {
                Some(event) => format!("{event:?}").chars().take(160).collect::<String>(),
                None => "control socket closed".to_string(),
            }
        );
        match event {
            Some(realtime::ServerEvent::SessionStarted(event)) => started = Some(event.session.id),
            Some(realtime::ServerEvent::UsageUpdated(event)) => {
                usage = Some(event.usage.audio_duration_ms)
            }
            Some(realtime::ServerEvent::Error(error)) => panic!("the server reported {error:?}"),
            Some(realtime::ServerEvent::Unknown(unknown)) => {
                eprintln!("unmodelled event `{}`", unknown.kind)
            }
            Some(_) => {}
            None => break,
        }
    }
    control.close().await.expect("the session closes");
    let peer = silence.await.expect("the sender stops");
    eprintln!("live peer after {:?}: {peer:?}", opened.elapsed());
    peer.close().await.expect("the peer closes");

    assert_eq!(started.as_deref(), Some(call.call_id.as_str()));
    assert!(usage.is_some(), "no usage within {seconds} s");
}
