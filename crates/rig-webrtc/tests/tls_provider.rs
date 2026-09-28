//! A secure websocket connect must not panic in a binary that links both
//! rustls providers, once the process installs one explicitly.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use rig_core::http_client::{NoBody, Request};
use rig_core::ws_client::{ConnectOptions, WebSocketClientExt};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wss_connect_to_a_silent_local_port_fails_without_panicking() {
    common::install_crypto_provider();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a local port");
    let port = listener.local_addr().expect("its address").port();
    let held = tokio::spawn(async move {
        // Accept and hold the socket without answering, so the client gets as
        // far as building its TLS configuration and then times out.
        let (socket, _) = listener.accept().await.expect("one connection");
        tokio::time::sleep(Duration::from_secs(5)).await;
        drop(socket);
    });
    let request = Request::builder()
        .uri(format!("wss://127.0.0.1:{port}/"))
        .body(NoBody)
        .expect("a request");
    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        rig_tungstenite::TungsteniteClient::new().connect(request, ConnectOptions::default()),
    )
    .await;
    assert!(
        !matches!(outcome, Ok(Ok(_))),
        "a silent peer cannot complete a TLS websocket handshake"
    );
    held.abort();
}
