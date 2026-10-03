//! The bundled backend must work when the caller has no tokio runtime.
//!
//! Bevy task pools, smol and `futures::executor` are the cases this exists for.
//! A websocket differs from a unary request in living long enough that the
//! socket cannot simply be driven per-call: it moves onto the fallback runtime
//! as an actor, and the caller polls only `futures` channels. That is invisible
//! in an ordinary tokio test, so this drives a whole session — connect, send,
//! receive, close — with `futures::executor::block_on` and no tokio runtime on
//! the calling thread.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use rig_core::http_client::{NoBody, Request};
use rig_core::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt as _,
};
use rig_tungstenite::TungsteniteClient;
use std::sync::mpsc;
use std::time::Duration;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        std::time::Duration::from_secs(10),
        future,
    ))
    .expect("client operation deadline")
}

/// Serve one websocket turn on its own tokio runtime, on its own thread: the
/// server needs a reactor even though the client under test must not have one.
///
/// With `events` empty the server accepts the turn and then goes quiet, which
/// is what an event timeout has to survive.
fn serve_one_turn(events: Vec<String>) -> String {
    serve_one_turn_after(None, events)
}

/// Hold events until the caller releases them, so a canceled read cannot
/// race a sleeping server waking up on a loaded machine.
fn serve_one_turn_after(
    release: Option<futures::channel::oneshot::Receiver<()>>,
    events: Vec<String>,
) -> String {
    use futures::{SinkExt, StreamExt};

    let (address_tx, address_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime should build");
        runtime.block_on(async move {
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address should send");

                let (stream, _) = listener.accept().await.expect("accept");
                let mut socket = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");

                let request = socket
                    .next()
                    .await
                    .expect("request should arrive")
                    .expect("request should be valid");
                assert!(
                    request
                        .into_text()
                        .expect("request should be text")
                        .contains("\"type\":\"response.create\""),
                    "the session should open the turn with response.create"
                );

                if let Some(release) = release {
                    release.await.expect("release events");
                }

                for event in events {
                    socket
                        .send(tokio_tungstenite::tungstenite::Message::text(event))
                        .await
                        .expect("event should send");
                }

                // Wait for the client's close handshake so the assertion below is
                // about a completed round trip, not a race.
                while let Some(Ok(message)) = socket.next().await {
                    if message.is_close() {
                        break;
                    }
                }
            })
            .await;
        });
    });

    let address = address_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("server should report its address");
    format!("ws://{address}/v1")
}

async fn connect(url: &str) -> BoxedWebSocketConnection {
    TungsteniteClient::new()
        .connect(
            Request::builder().uri(url).body(NoBody).expect("request"),
            ConnectOptions::new(),
        )
        .await
        .expect("connect without a caller runtime")
}

#[test]
fn a_raw_round_trip_runs_without_a_tokio_runtime() {
    let events = vec!["first event".to_owned(), "second event".to_owned()];
    let url = serve_one_turn(events.clone());
    assert!(tokio::runtime::Handle::try_current().is_err());
    block_on(async move {
        let mut connection = connect(&url).await;
        connection
            .send(Frame::Text(r#"{"type":"response.create"}"#.to_owned()))
            .await
            .expect("send");
        for expected in events {
            assert_eq!(
                connection.recv().await.expect("receive"),
                Some(Frame::Text(expected))
            );
        }
        connection.close(None).await.expect("close");
    });
}

/// A canceled pending receive must leave the actor able to accept close.
#[test]
fn a_receive_timeout_still_allows_close_without_a_tokio_runtime() {
    let url = serve_one_turn(Vec::new());
    assert!(tokio::runtime::Handle::try_current().is_err());
    block_on(async move {
        let mut connection = connect(&url).await;
        connection
            .send(Frame::Text(r#"{"type":"response.create"}"#.to_owned()))
            .await
            .expect("send");
        let timeout =
            rig_core::wasm_compat::timeout(Duration::from_millis(50), connection.recv()).await;
        assert!(timeout.is_err(), "the peer deliberately sends no frame");
        rig_core::wasm_compat::timeout(Duration::from_secs(5), connection.close(None))
            .await
            .expect("close must not hang after receive cancellation")
            .expect("close");
    });
}

/// Only cancellation before the server releases a frame is proved here.
/// Cancellation after the actor successfully enqueues its reply can lose it.
#[test]
fn cancellation_before_a_frame_exists_preserves_the_later_frame() {
    let (release, released) = futures::channel::oneshot::channel();
    let url = serve_one_turn_after(Some(released), vec!["kept".to_owned()]);
    assert!(tokio::runtime::Handle::try_current().is_err());
    block_on(async move {
        let mut connection = connect(&url).await;
        connection
            .send(Frame::Text(r#"{"type":"response.create"}"#.to_owned()))
            .await
            .expect("send");
        let cancelled =
            rig_core::wasm_compat::timeout(Duration::from_millis(20), connection.recv()).await;
        assert!(cancelled.is_err());
        release.send(()).expect("release only after cancellation");
        let event = rig_core::wasm_compat::timeout(Duration::from_secs(5), connection.recv())
            .await
            .expect("receive deadline")
            .expect("receive");
        assert_eq!(event, Some(Frame::Text("kept".to_owned())));
        connection.close(None).await.expect("close");
    });
}
