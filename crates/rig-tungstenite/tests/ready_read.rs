//! Ready reads and flushing through the backend's off-runtime connection actor.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use rig_http::http_client::{NoBody, Request};
use rig_http::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, ReadyFrame, WebSocketClientExt as _,
    WebSocketConnection as _,
};
use rig_tungstenite::TungsteniteClient;
use std::sync::mpsc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

const DEADLINE: Duration = Duration::from_secs(10);

async fn next_nonempty_ready(connection: &mut BoxedWebSocketConnection) -> ReadyFrame {
    loop {
        match connection
            .recv_ready()
            .await
            .expect("read the actor's queue")
        {
            ReadyFrame::Empty => {
                rig_http::wasm_compat::sleep(Duration::from_millis(1)).await;
            }
            ready => return ready,
        }
    }
}

#[test]
fn forwarded_ready_reads_preserve_frames_flush_pongs_and_report_the_end() {
    let (address_tx, address_rx) = mpsc::channel();
    let (release, released) = futures::channel::oneshot::channel();
    let server = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build the server runtime");
        runtime.block_on(async move {
            tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind a loopback listener");
                address_tx
                    .send(listener.local_addr().expect("listener address"))
                    .expect("report the listener address");
                let (stream, _) = listener.accept().await.expect("accept the client");
                let mut peer = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("complete the handshake");
                released
                    .await
                    .expect("release the peer after the empty read");
                for message in [
                    Message::text("first"),
                    Message::Binary(Bytes::from_static(b"second")),
                    Message::Ping(Bytes::from_static(b"actor-ping")),
                ] {
                    peer.send(message).await.expect("send the ordered frames");
                }
                assert_eq!(
                    peer.next()
                        .await
                        .expect("the peer should receive the pong")
                        .expect("the pong should be valid"),
                    Message::Pong(Bytes::from_static(b"actor-ping")),
                );
                peer.send(Message::Close(None))
                    .await
                    .expect("start the close handshake");
                assert_eq!(
                    peer.next()
                        .await
                        .expect("the peer should receive the close acknowledgment")
                        .expect("the close acknowledgment should be valid"),
                    Message::Close(None),
                );
                assert!(
                    peer.next().await.is_none(),
                    "the close handshake should end"
                );
            })
            .await
            .expect("the server workflow should finish");
        });
    });
    let address = address_rx
        .recv_timeout(DEADLINE)
        .expect("the server should report its address");
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "the caller must use the actual off-runtime connection actor",
    );

    futures::executor::block_on(async {
        rig_http::wasm_compat::timeout(DEADLINE, async {
            let request = Request::builder()
                .uri(format!("ws://{address}/"))
                .body(NoBody)
                .expect("build the handshake request");
            let mut connection = TungsteniteClient::new()
                .connect(request, ConnectOptions::new().with_timeout(Some(DEADLINE)))
                .await
                .expect("connect through the fallback runtime");
            assert_eq!(
                connection
                    .recv_ready()
                    .await
                    .expect("read before peer data"),
                ReadyFrame::Empty,
            );
            release.send(()).expect("allow the peer to send");
            for frame in [
                Frame::Text("first".to_string()),
                Frame::Binary(Bytes::from_static(b"second")),
                Frame::Ping(Bytes::from_static(b"actor-ping")),
            ] {
                assert_eq!(
                    next_nonempty_ready(&mut connection).await,
                    ReadyFrame::Frame(frame),
                );
            }
            connection.flush().await.expect("flush through the actor");
            assert_eq!(
                next_nonempty_ready(&mut connection).await,
                ReadyFrame::Frame(Frame::Close(None)),
            );
            connection.flush().await.expect("flush the close reply");
            assert_eq!(
                next_nonempty_ready(&mut connection).await,
                ReadyFrame::Ended
            );
            assert_eq!(
                connection.recv_ready().await.expect("read the ended actor"),
                ReadyFrame::Ended,
            );
        })
        .await
        .expect("the client workflow should finish");
    });
    server.join().expect("the server assertions should pass");
}
