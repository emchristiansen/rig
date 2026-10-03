//! Ready reads and automatic pong replies on direct and forwarded transports.
//! These local raw-socket tests do not assert canonical Responses semantics.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use futures::{SinkExt, StreamExt};
use rig_core::http_client::{NoBody, Request};
use rig_core::ws_client::{ConnectOptions, Frame, ReadyFrame, WebSocketClientExt as _};
use rig_tungstenite::TungsteniteClient;
use std::sync::mpsc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

fn serve_idle_window() -> (String, mpsc::Receiver<()>) {
    let (address_tx, address_rx) = mpsc::channel();
    let (pong_tx, pong_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime");
        runtime.block_on(async move {
            tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address send");
                let (stream, _) = listener.accept().await.expect("accept");
                let mut socket = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");
                assert_eq!(
                    socket.next().await.expect("first").expect("valid"),
                    Message::text("first")
                );
                for message in [
                    Message::text("event one"),
                    Message::text("event two"),
                    Message::Ping(b"idle".to_vec().into()),
                ] {
                    socket.send(message).await.expect("send");
                }
                assert_eq!(
                    socket.next().await.expect("pong").expect("valid"),
                    Message::Pong(b"idle".to_vec().into())
                );
                pong_tx.send(()).expect("pong report");
                assert_eq!(
                    socket.next().await.expect("second").expect("valid"),
                    Message::text("second")
                );
                socket
                    .send(Message::text("after idle"))
                    .await
                    .expect("reply");
                while let Some(Ok(message)) = socket.next().await {
                    if message.is_close() {
                        break;
                    }
                }
            })
            .await
            .expect("server deadline");
        });
    });
    let address = address_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("server address");
    (format!("ws://{address}/"), pong_rx)
}

async fn exercise_idle_window(url: String, pong_rx: mpsc::Receiver<()>) {
    let mut connection = TungsteniteClient::new()
        .connect(
            Request::builder().uri(url).body(NoBody).expect("request"),
            ConnectOptions::new(),
        )
        .await
        .expect("connect");
    connection
        .send(Frame::Text("first".to_owned()))
        .await
        .expect("send");
    let mut recovered = Vec::new();
    let mut ponged = false;
    loop {
        loop {
            match connection.recv_ready().await.expect("ready read") {
                ReadyFrame::Frame(frame) => recovered.push(frame),
                ReadyFrame::Empty => break,
                ReadyFrame::Ended => panic!("peer ended while idle"),
            }
        }
        connection.flush().await.expect("flush automatic pong");
        if ponged {
            break;
        }
        ponged = pong_rx.try_recv().is_ok();
        if !ponged {
            let _ = rig_core::wasm_compat::timeout(
                Duration::from_millis(10),
                futures::future::pending::<()>(),
            )
            .await;
        }
    }
    assert_eq!(
        recovered,
        vec![
            Frame::Text("event one".to_owned()),
            Frame::Text("event two".to_owned()),
            Frame::Ping(b"idle".to_vec().into())
        ]
    );
    connection
        .send(Frame::Text("second".to_owned()))
        .await
        .expect("send after idle");
    assert_eq!(
        connection.recv().await.expect("reply"),
        Some(Frame::Text("after idle".to_owned()))
    );
    connection.close(None).await.expect("close");
}

#[tokio::test]
async fn an_idle_drain_services_a_direct_connection() {
    let (url, pong) = serve_idle_window();
    tokio::time::timeout(Duration::from_secs(20), exercise_idle_window(url, pong))
        .await
        .expect("idle deadline");
}

#[test]
fn an_idle_drain_services_a_forwarded_connection() {
    let (url, pong) = serve_idle_window();
    assert!(tokio::runtime::Handle::try_current().is_err());
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        Duration::from_secs(20),
        exercise_idle_window(url, pong),
    ))
    .expect("idle deadline");
}

/// Caller-supplied Live headers reach the real handshake without alteration.
#[tokio::test]
async fn the_raw_handshake_preserves_supplied_identity_headers() {
    use tokio_tungstenite::tungstenite::handshake::server::{
        ErrorResponse, Request as UpgradeRequest, Response,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut captured = None;
        let callback =
            |request: &UpgradeRequest, response: Response| -> Result<Response, ErrorResponse> {
                captured = Some(request.headers().clone());
                Ok(response)
            };
        let mut socket = tokio_tungstenite::accept_hdr_async(stream, callback)
            .await
            .expect("upgrade");
        while let Some(Ok(message)) = socket.next().await {
            if message.is_close() {
                break;
            }
        }
        captured.expect("handshake headers")
    });
    let supplied = [
        ("authorization", "Bearer test-token"),
        ("chatgpt-account-id", "acct-123"),
        ("originator", "test-caller"),
        ("user-agent", "test-caller/1"),
        ("version", "1"),
        ("session-id", "session-1"),
        ("thread-id", "thread-1"),
        ("x-session-id", "session-1"),
    ];
    let mut request = Request::builder().uri(format!("ws://{address}/v1/live/call-1"));
    for (name, value) in supplied {
        request = request.header(name, value);
    }
    let mut connection = TungsteniteClient::new()
        .connect(
            request.body(NoBody).expect("request"),
            ConnectOptions::new(),
        )
        .await
        .expect("connect");
    connection.close(None).await.expect("close");
    let headers = server.await.expect("server");
    for (name, value) in supplied {
        assert_eq!(headers[name], value, "{name}");
    }
}
