//! The composition the crate documentation shows, run offline: a binary that
//! links `rig-webrtc` with `rig-reqwest`'s `rustls` feature has two rustls
//! providers, installs one exactly as the documentation does, and then
//! reaches TLS on both the call-creation request and the control socket
//! without panicking. The servers read the client's first TLS record and hang
//! up, so this proves the TLS client was built and started its handshake,
//! not that a handshake completes.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

mod common;

use std::net::SocketAddr;

use rig_core::providers::chatgpt::{self, realtime};
use rig_core::providers::openai::OpenAI;
use rig_core::providers::openai::wire::CallerIdentity;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// The content type of a TLS handshake record, which starts a ClientHello.
const TLS_HANDSHAKE: u8 = 0x16;

/// Accept one connection, return the first byte the client sends, and hang
/// up.
async fn first_byte_server() -> (SocketAddr, JoinHandle<u8>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a local port");
    let addr = listener.local_addr().expect("its address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("one connection");
        socket.read_u8().await.expect("the client speaks first")
    });
    (addr, server)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_documented_composition_reaches_tls_without_panicking() {
    // The documented setup, verbatim. This binary installs nothing else.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("no other rustls provider is installed");

    let (https, call_server) = first_byte_server().await;
    let (wss, control_server) = first_byte_server().await;
    let identity =
        CallerIdentity::new("codex_cli_rs", "codex_cli_rs/0.155.1", None).expect("an identity");
    let provider = OpenAI::with_key(&chatgpt::DIALECT, "access-token")
        .with_base_url(format!("https://{https}/backend-api/codex"))
        .with_account_id("account-id")
        .with_caller_identity(identity);
    let calls = realtime::LiveCalls::new(provider)
        .expect("the Codex backend")
        .with_control_base_url(format!("wss://{wss}/v1/live"));

    let peer = common::loopback_builder(0)
        .build()
        .await
        .expect("the peer builds");
    let offer_sdp = peer.offer().await.expect("an offer");
    let created = tokio::time::timeout(
        common::WAIT,
        calls.create_call(
            &rig_reqwest::ReqwestClient::default(),
            &offer_sdp,
            &realtime::SessionConfig::new("Answer briefly."),
        ),
    )
    .await
    .expect("the request ends when the server hangs up");
    assert!(created.is_err(), "the server hung up: {created:?}");
    let control = tokio::time::timeout(
        common::WAIT,
        calls.connect_control(
            &rig_tungstenite::TungsteniteClient::new(),
            realtime::CallId::new("rtc_u2_offline").expect("a call id"),
        ),
    )
    .await
    .expect("the connect ends when the server hangs up");
    assert!(control.is_err(), "the server hung up");

    assert_eq!(call_server.await.expect("served"), TLS_HANDSHAKE);
    assert_eq!(control_server.await.expect("served"), TLS_HANDSHAKE);
    peer.close().await.expect("the peer closes");
}
