use super::*;
use bytes::Bytes;
use futures::FutureExt as _;
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(10);

async fn loopback_sockets() -> (Socket, WebSocketStream<TcpStream>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback listener");
    let address = listener.local_addr().expect("listener address");
    let (client, accepted) = tokio::join!(TcpStream::connect(address), listener.accept());
    let client = client.expect("connect to the listener");
    let (server, _) = accepted.expect("accept the client");
    let client = WebSocketStream::from_raw_socket(
        MaybeTlsStream::Plain(client),
        tungstenite::protocol::Role::Client,
        None,
    )
    .await;
    let server =
        WebSocketStream::from_raw_socket(server, tungstenite::protocol::Role::Server, None).await;
    (client, server)
}

async fn wait_for_socket_bytes(socket: &Socket, byte_count: usize) {
    let MaybeTlsStream::Plain(stream) = socket.get_ref() else {
        panic!("the loopback socket should be plain TCP");
    };
    let mut bytes = vec![0; byte_count];
    loop {
        let available = stream.peek(&mut bytes).await.expect("peek at socket input");
        assert!(available > 0, "the peer should remain open");
        if available == byte_count {
            return;
        }
        tokio::task::yield_now().await;
    }
}

async fn buffer_peer_frames(
    socket: &Socket,
    peer: &mut WebSocketStream<TcpStream>,
    frames: &[Frame],
) {
    let mut byte_count = 0;
    for frame in frames {
        let message = into_message(frame.clone());
        assert!(message.len() < 126, "these test frames use short headers");
        // A short, unmasked server frame has a two-byte header.
        byte_count += message.len() + 2;
        peer.feed(message).await.expect("queue the peer's frame");
    }
    peer.flush().await.expect("flush the peer's frames");
    wait_for_socket_bytes(socket, byte_count).await;
}

/// Every frame the protocol carries must survive a round trip through
/// tungstenite's own representation, or a session sees the wrong event.
#[test]
fn frames_round_trip_through_tungstenite() {
    let cases = [
        Frame::Text("{\"type\":\"response.create\"}".to_string()),
        Frame::Binary(Bytes::from_static(b"\x00\x01")),
        Frame::Ping(Bytes::from_static(b"ping")),
        Frame::Pong(Bytes::new()),
        Frame::Close(Some(CloseFrame {
            code: 1000,
            reason: "done".to_string(),
        })),
        Frame::Close(None),
    ];

    for frame in cases {
        assert_eq!(
            from_message(into_message(frame.clone())),
            Some(frame.clone()),
            "frame should round-trip"
        );
    }
}

/// A raw frame is not protocol payload: it must be skipped, not handed on
/// as bytes a session would try to parse.
#[test]
fn a_raw_frame_carries_no_protocol_payload() {
    let raw = Message::Frame(tungstenite::protocol::frame::Frame::message(
        Bytes::from_static(b"raw"),
        tungstenite::protocol::frame::coding::OpCode::Data(
            tungstenite::protocol::frame::coding::Data::Binary,
        ),
        true,
    ));

    assert_eq!(from_message(raw), None);
}

/// Releasing the last command handle ends the actor, so a connection whose
/// owner is gone never keeps its socket. The owner's drop also aborts the
/// actor, and that abort can land first, so this drives the actor directly.
#[tokio::test]
async fn the_actor_ends_when_its_last_handle_is_released() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback listener");
    let address = listener.local_addr().expect("listener address");
    let (client, accepted) = tokio::join!(TcpStream::connect(address), listener.accept());
    let client = client.expect("connect to the listener");
    let _server = accepted.expect("accept the client");
    let socket = WebSocketStream::from_raw_socket(
        MaybeTlsStream::Plain(client),
        tungstenite::protocol::Role::Client,
        None,
    )
    .await;
    let (commands, requests) = futures::channel::mpsc::channel::<Command>(1);
    drop(commands);

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_actor(socket, requests),
    )
    .await
    .expect("the actor should end once no handle remains");
}

#[tokio::test]
async fn a_direct_ready_read_is_empty_on_its_first_poll_before_the_peer_sends() {
    tokio::time::timeout(DEADLINE, async {
        let (socket, _peer) = loopback_sockets().await;
        let mut connection = DirectConnection::new(socket);
        assert_eq!(
            connection
                .recv_ready()
                .now_or_never()
                .expect("a ready read should complete on its first poll")
                .expect("read the idle socket"),
            ReadyFrame::Empty,
        );
    })
    .await
    .expect("the idle read should finish without waiting for the peer");
}

#[tokio::test]
async fn direct_ready_reads_preserve_frame_order_and_report_the_end() {
    tokio::time::timeout(DEADLINE, async {
        let (socket, mut peer) = loopback_sockets().await;
        let expected = [
            Frame::Text("first".to_string()),
            Frame::Binary(Bytes::from_static(b"second")),
            Frame::Close(None),
        ];
        buffer_peer_frames(&socket, &mut peer, &expected).await;
        let mut connection = DirectConnection::new(socket);
        for frame in expected {
            assert_eq!(
                connection
                    .recv_ready()
                    .now_or_never()
                    .expect("a buffered read should complete on its first poll")
                    .expect("read the buffered frame"),
                ReadyFrame::Frame(frame),
            );
        }
        connection.flush().await.expect("flush the close reply");
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
        drop(peer);
        loop {
            let answer = connection
                .recv_ready()
                .now_or_never()
                .expect("an ending read should complete on its first poll")
                .expect("read the ending socket");
            match answer {
                ReadyFrame::Ended => break,
                ReadyFrame::Empty => tokio::task::yield_now().await,
                ReadyFrame::Frame(frame) => panic!("unexpected frame after close: {frame:?}"),
            }
        }
        for _ in 0..2 {
            assert_eq!(
                connection
                    .recv_ready()
                    .now_or_never()
                    .expect("an ended read should complete on its first poll")
                    .expect("read the ended socket"),
                ReadyFrame::Ended,
            );
        }
    })
    .await
    .expect("the ordered reads and close handshake should finish");
}

#[tokio::test]
async fn direct_flush_delivers_the_automatic_pong_after_a_ready_read() {
    tokio::time::timeout(DEADLINE, async {
        let (socket, mut peer) = loopback_sockets().await;
        let payload = Bytes::from_static(b"ready-ping");
        buffer_peer_frames(&socket, &mut peer, &[Frame::Ping(payload.clone())]).await;
        let mut connection = DirectConnection::new(socket);
        assert_eq!(
            connection
                .recv_ready()
                .now_or_never()
                .expect("the ping read should complete on its first poll")
                .expect("read the ping"),
            ReadyFrame::Frame(Frame::Ping(payload.clone())),
        );
        // No later read or send may incidentally flush the automatic reply.
        connection.flush().await.expect("flush the queued pong");
        assert_eq!(
            peer.next()
                .await
                .expect("the peer should receive the pong")
                .expect("the pong should be valid"),
            Message::Pong(payload),
        );
    })
    .await
    .expect("the peer should receive the flushed pong");
}

#[derive(Clone, Copy)]
enum ReceiveMode {
    Ordinary,
    Ready,
}

async fn enqueue_cancelled_read(
    commands: &mut futures::channel::mpsc::Sender<Command>,
    mode: ReceiveMode,
) {
    let command = match mode {
        ReceiveMode::Ordinary => {
            let (reply, answer) = futures::channel::oneshot::channel();
            drop(answer);
            Command::Recv(reply)
        }
        ReceiveMode::Ready => {
            let (reply, answer) = futures::channel::oneshot::channel();
            drop(answer);
            Command::RecvReady(reply)
        }
    };
    commands
        .send(command)
        .await
        .expect("enqueue a cancelled read");
}

#[tokio::test]
async fn cancelled_actor_replies_keep_the_queued_frame_at_the_front() {
    tokio::time::timeout(DEADLINE, async {
        for mode in [ReceiveMode::Ordinary, ReceiveMode::Ready] {
            let (socket, mut peer) = loopback_sockets().await;
            let expected = [
                Frame::Text("kept".to_string()),
                Frame::Binary(Bytes::from_static(b"after")),
            ];
            buffer_peer_frames(&socket, &mut peer, &expected).await;
            let (mut commands, requests) = futures::channel::mpsc::channel::<Command>(1);
            let mut actor = Box::pin(run_actor(socket, requests));
            // Drive all ready input into the real actor before it sees a read.
            assert!(futures::poll!(actor.as_mut()).is_pending());
            enqueue_cancelled_read(&mut commands, mode).await;
            assert!(futures::poll!(actor.as_mut()).is_pending());

            for frame in expected {
                let (reply, answer) = futures::channel::oneshot::channel();
                commands
                    .send(Command::RecvReady(reply))
                    .await
                    .expect("request the retained frame");
                assert!(futures::poll!(actor.as_mut()).is_pending());
                assert_eq!(
                    answer
                        .now_or_never()
                        .expect("the actor should have answered")
                        .expect("the actor should retain the reply channel")
                        .expect("the retained frame should be valid"),
                    ReadyFrame::Frame(frame),
                );
            }
            drop(commands);
            actor.await;
        }
    })
    .await
    .expect("cancelled replies should leave both frames readable");
}

#[tokio::test]
async fn cancelled_actor_replies_keep_the_queued_protocol_error() {
    use tokio::io::AsyncWriteExt as _;

    tokio::time::timeout(DEADLINE, async {
        for mode in [ReceiveMode::Ordinary, ReceiveMode::Ready] {
            let (socket, mut peer) = loopback_sockets().await;
            // Opcode 3 is reserved, so this produces a concrete protocol error.
            peer.get_mut()
                .write_all(&[0x83, 0x00])
                .await
                .expect("send an invalid opcode");
            wait_for_socket_bytes(&socket, 2).await;
            let (mut commands, requests) = futures::channel::mpsc::channel::<Command>(1);
            let mut actor = Box::pin(run_actor(socket, requests));
            assert!(futures::poll!(actor.as_mut()).is_pending());
            enqueue_cancelled_read(&mut commands, mode).await;
            assert!(futures::poll!(actor.as_mut()).is_pending());

            let (reply, answer) = futures::channel::oneshot::channel();
            commands
                .send(Command::RecvReady(reply))
                .await
                .expect("request the retained error");
            assert!(futures::poll!(actor.as_mut()).is_pending());
            let error = answer
                .now_or_never()
                .expect("the actor should have answered")
                .expect("the actor should retain the reply channel")
                .expect_err("the cancelled reply should preserve the protocol error");
            let Error::Instance(error) = error else {
                panic!("a protocol error should preserve its concrete type");
            };
            assert!(matches!(
                error.downcast_ref::<tungstenite::Error>(),
                Some(tungstenite::Error::Protocol(
                    tungstenite::error::ProtocolError::InvalidOpcode(3)
                )),
            ));

            let (reply, answer) = futures::channel::oneshot::channel();
            commands
                .send(Command::RecvReady(reply))
                .await
                .expect("request the end after the error");
            assert!(futures::poll!(actor.as_mut()).is_pending());
            assert_eq!(
                answer
                    .now_or_never()
                    .expect("the actor should have answered")
                    .expect("the actor should retain the reply channel")
                    .expect("the error should be consumed exactly once"),
                ReadyFrame::Ended,
            );
            drop(commands);
            actor.await;
        }
    })
    .await
    .expect("cancelled replies should leave the protocol error readable");
}
