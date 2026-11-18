//! The whole call sequence of the live test, offline: `LiveCalls` creates the
//! call over the bundled reqwest transport against a local HTTP server, whose
//! answer comes from an in-process answering peer; the `LivePeer` connects
//! over loopback; the control socket joins a local websocket server over
//! the bundled tungstenite backend and answers a recorded delegation.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

mod common;

use std::collections::BTreeSet;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_live::chatgpt as realtime;
use rig_live::support::{CallerIdentity, CodexIdentity, RequestHeaders};
use rig_webrtc::{LivePeer, PeerEvent};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

const CALL_ID: &str = "rtc_u2_offline";

/// What a server received: request line, every header as received with a
/// lowercase name, body.
#[derive(Debug, Default)]
struct Received {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Received {
    /// The one value received under `name`; a missing or repeated header
    /// fails the test.
    fn single(&self, name: &str) -> &str {
        let values: Vec<&str> = self
            .headers
            .iter()
            .filter(|(received, _)| received == name)
            .map(|(_, value)| value.as_str())
            .collect();
        let [value] = values.as_slice() else {
            panic!("expected one `{name}`, received {values:?}");
        };
        value
    }

    /// The names received, once each.
    fn names(&self) -> BTreeSet<&str> {
        self.headers.iter().map(|(name, _)| name.as_str()).collect()
    }
}

/// Read one HTTP/1.1 request: its request line, its headers, and the body
/// its `content-length` states, which is empty when it states none.
async fn read_request(stream: &mut TcpStream) -> Received {
    let mut buffer = Vec::new();
    let head_end = loop {
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk).await.expect("reads");
        assert!(read > 0, "the request ended early");
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let head = String::from_utf8(buffer[..head_end].to_vec()).expect("an ASCII head");
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().expect("a request line").split(' ');
    let mut received = Received {
        method: request_line.next().expect("a method").to_owned(),
        target: request_line.next().expect("a target").to_owned(),
        ..Received::default()
    };
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':').expect("a header");
        received
            .headers
            .push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
    let length: usize = received
        .headers
        .iter()
        .find(|(name, _)| name == "content-length")
        .map_or(0, |(_, value)| value.parse().expect("a length"));
    let mut body = buffer[head_end..].to_vec();
    while body.len() < length {
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk).await.expect("reads");
        assert!(read > 0, "the body ended early");
        body.extend_from_slice(&chunk[..read]);
    }
    received.body = body;
    received
}

/// Serve one HTTP/1.1 request: record it, answer the offer in its body with
/// an in-process peer, and reply 201 with the answer and a `Location`.
async fn serve_call(listener: TcpListener) -> (Received, common::AnsweringPeer) {
    let (mut stream, _) = listener.accept().await.expect("a connection");
    let received = read_request(&mut stream).await;

    let request: serde_json::Value = serde_json::from_slice(&received.body).expect("JSON");
    let answering = common::answer(request["sdp"].as_str().expect("an offer").to_owned()).await;
    let reply = format!(
        "HTTP/1.1 201 Created\r\ncontent-type: text/plain\r\nlocation: /v1/realtime/calls/{CALL_ID}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        answering.answer.len(),
        answering.answer
    );
    stream.write_all(reply.as_bytes()).await.expect("replies");
    stream.shutdown().await.expect("closes");
    (received, answering)
}

/// Serve one control socket: record the handshake, send the recorded
/// delegation, acknowledge the answer, and return every client frame.
async fn serve_control(listener: TcpListener) -> (Received, Vec<serde_json::Value>) {
    let (stream, _) = listener.accept().await.expect("a connection");
    let mut handshake = Received::default();
    let callback = |request: &Request, response: Response| {
        handshake.method = request.method().to_string();
        handshake.target = request.uri().to_string();
        for (name, value) in request.headers() {
            handshake.headers.push((
                name.as_str().to_owned(),
                value.to_str().expect("ascii").to_owned(),
            ));
        }
        Ok(response)
    };
    let mut socket = tokio_tungstenite::accept_hdr_async(stream, callback)
        .await
        .expect("the upgrade completes");
    for event in [
        format!(r#"{{"type":"session.started","session":{{"id":"{CALL_ID}","status":"active","expires_at":1790626280}}}}"#),
        r#"{"type":"delegation.created","item":{"id":"item_1","type":"delegation","content":[{"type":"input_text","text":"What is the status?"}],"handoff_id":"handoff_1","target":"client","user_bidi_turn_id":"turn_1"},"offset_ms":5000}"#.to_owned(),
    ] {
        socket.send(Message::text(event)).await.expect("sends");
    }
    let mut client = Vec::new();
    while let Some(message) = socket.next().await {
        match message.expect("reads") {
            Message::Text(text) => {
                let event: serde_json::Value = serde_json::from_str(&text).expect("JSON");
                if event["type"] == "delegation.context.append" {
                    socket
                        .send(Message::text(
                            r#"{"type":"delegation.context.appended","delegation_item_id":"item_1","start_ms":5200,"end_ms":5400}"#,
                        ))
                        .await
                        .expect("sends");
                }
                client.push(event);
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    (handshake, client)
}

/// Serve one HTTP/1.1 request: record it and reply 204.
async fn serve_bare(listener: TcpListener) -> Received {
    let (mut stream, _) = listener.accept().await.expect("a connection");
    let received = read_request(&mut stream).await;
    stream
        .write_all(b"HTTP/1.1 204 No Content\r\nconnection: close\r\n\r\n")
        .await
        .expect("replies");
    stream.shutdown().await.expect("closes");
    received
}

/// The `accept-encoding` value `transport` adds by itself to a request that
/// names none, or `None` when it adds no such header: what a local server
/// receives for a bare `GET` this crate had no part in.
///
/// reqwest adds the header when it was compiled with `gzip`, `brotli`,
/// `zstd` or `deflate`, and Cargo compiles one reqwest for every package it
/// builds together, so whether a default client sends it depends on which
/// other packages are in the build. The tests ask the transport instead of
/// assuming either answer.
async fn automatic_accept_encoding(transport: &rig_reqwest::ReqwestClient) -> Option<String> {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("an address");
    let server = tokio::spawn(serve_bare(listener));
    transport
        .inner()
        .expect("reqwest built the client")
        .get(format!("http://{addr}/"))
        .send()
        .await
        .expect("the bare request is answered");
    let received = server.await.expect("the bare server finishes");
    let values: Vec<&str> = received
        .headers
        .iter()
        .filter(|(name, _)| name == "accept-encoding")
        .map(|(_, value)| value.as_str())
        .collect();
    match values.as_slice() {
        [] => None,
        [value] => Some((*value).to_owned()),
        repeated => panic!("the transport repeated `accept-encoding`: {repeated:?}"),
    }
}

/// A reqwest transport that asks for no compressed response in any build:
/// the four switches exist whether or not the features they turn off do.
fn transport_without_compression() -> rig_reqwest::ReqwestClient {
    rig_reqwest::reqwest::Client::builder()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .build()
        .expect("reqwest builds its client")
        .into()
}

/// The headers the protocol and the caller put on the call request and on
/// the control handshake, in the order the Codex client sends them.
const EXPECTED_CALL_HEADERS: [(&str, &str); 10] = [
    ("authorization", "Bearer test-token"),
    ("chatgpt-account-id", "acct-123"),
    ("openai-alpha", "quicksilver=v2"),
    ("originator", "codex_cli_rs"),
    (
        "user-agent",
        "codex_cli_rs/0.155.1 (NixOS 26.05; x86_64) unknown",
    ),
    ("version", "0.155.1"),
    ("session-id", "session-1"),
    ("thread-id", "thread-1"),
    ("x-session-id", "session-1"),
    ("x-codex-turn-metadata", r#"{"thread_source":"user"}"#),
];

/// The headers reqwest and the request's JSON body add to every call
/// request, beside [`EXPECTED_CALL_HEADERS`].
const TRANSPORT_CALL_HEADERS: [&str; 4] = ["content-type", "content-length", "host", "accept"];

/// The calls of the test's caller against the HTTP server at `http_addr`,
/// with the control socket at `control_base_url`.
fn live_calls(
    http_addr: std::net::SocketAddr,
    control_base_url: String,
    transport: rig_reqwest::ReqwestClient,
) -> realtime::LiveCalls {
    let identity = CallerIdentity::new(
        "codex_cli_rs",
        "codex_cli_rs/0.155.1 (NixOS 26.05; x86_64) unknown",
        Some("0.155.1".to_owned()),
    )
    .expect("an identity");
    let provider = OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token")
        .with_base_url(format!("http://{http_addr}/backend-api/codex"))
        .with_account_id("acct-123")
        .connect(transport);
    let conversation = CodexIdentity::from_ids("session-1", "thread-1").expect("ids");
    realtime::LiveCalls::new(provider, conversation)
        .expect("the Codex backend")
        .with_caller_identity(identity)
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-turn-metadata", r#"{"thread_source":"user"}"#)
                .expect("a valid header"),
        )
        .expect("not a Live header")
        .with_control_base_url(control_base_url)
}

/// A peer offering over loopback.
async fn offering_peer() -> LivePeer {
    LivePeer::builder()
        .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
        .with_tcp_addrs(Vec::new())
        .with_loopback_candidates(true)
        .build()
        .await
        .expect("the peer builds")
}

/// What the call server received is the call request: the method, target
/// and every header the protocol, the caller and the JSON body own, by
/// value.
fn assert_call_request(received: &Received, http_addr: std::net::SocketAddr) {
    assert_eq!(received.method, "POST");
    assert_eq!(
        received.target,
        "/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas"
    );
    for (name, value) in EXPECTED_CALL_HEADERS {
        assert_eq!(received.single(name), value, "{name}");
    }
    assert_eq!(received.single("content-type"), "application/json");
    assert_eq!(received.single("accept"), "*/*");
    assert_eq!(received.single("host"), http_addr.to_string());
}

/// Over a transport that asks for no compression, the call request names
/// exactly the protocol's, the caller's and the transport's fixed headers,
/// in every build.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_call_over_a_transport_without_compression_names_a_fixed_set_of_headers() {
    let transport = transport_without_compression();
    assert_eq!(automatic_accept_encoding(&transport).await, None);

    let http = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let http_addr = http.local_addr().expect("an address");
    let call_server = tokio::spawn(serve_call(http));
    let calls = live_calls(
        http_addr,
        "ws://127.0.0.1:9/v1/live".to_owned(),
        transport.clone(),
    );
    let peer = offering_peer().await;
    let offer = peer.offer().await.expect("an offer");
    calls
        .create_call(
            &transport,
            &offer,
            &realtime::SessionConfig::new("Answer briefly."),
        )
        .await
        .expect("the call is created");
    let (received, answering) = call_server.await.expect("the call server finishes");

    assert_call_request(&received, http_addr);
    let call_names: BTreeSet<&str> = EXPECTED_CALL_HEADERS
        .iter()
        .map(|(name, _)| *name)
        .chain(TRANSPORT_CALL_HEADERS)
        .collect();
    assert_eq!(received.names(), call_names);
    assert_eq!(
        received.headers.len(),
        call_names.len(),
        "a repeated header"
    );

    peer.close().await.expect("the peer closes");
    answering
        .connection
        .close()
        .await
        .expect("the answerer closes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_call_runs_end_to_end_against_local_servers() {
    let http = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let ws = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let http_addr = http.local_addr().expect("an address");
    let ws_addr = ws.local_addr().expect("an address");
    let call_server = tokio::spawn(serve_call(http));
    let control_server = tokio::spawn(serve_control(ws));

    // The client a caller gets by default. Whether it adds
    // `accept-encoding` is the build's, so the transport is asked first.
    let transport = rig_reqwest::ReqwestClient::default();
    let automatic = automatic_accept_encoding(&transport).await;
    let calls = live_calls(
        http_addr,
        format!("ws://{ws_addr}/v1/live"),
        transport.clone(),
    );

    let peer = offering_peer().await;
    let offer = peer.offer().await.expect("an offer");
    let call = calls
        .create_call(
            &transport,
            &offer,
            &realtime::SessionConfig::new("Answer briefly."),
        )
        .await
        .expect("the call is created");
    assert_eq!(call.call_id.as_str(), CALL_ID);
    let (received, answering) = call_server.await.expect("the call server finishes");
    assert_eq!(call.answer_sdp, answering.answer);

    assert_call_request(&received, http_addr);
    // Beside the fixed headers, exactly what the transport adds by itself:
    // `accept-encoding` with the transport's own value when this build's
    // reqwest negotiates compression, and no such header when it does not.
    let call_names: BTreeSet<&str> = EXPECTED_CALL_HEADERS
        .iter()
        .map(|(name, _)| *name)
        .chain(TRANSPORT_CALL_HEADERS)
        .chain(automatic.as_ref().map(|_| "accept-encoding"))
        .collect();
    assert_eq!(received.names(), call_names);
    if let Some(automatic) = &automatic {
        assert_eq!(received.single("accept-encoding"), automatic);
    }
    assert_eq!(
        received.headers.len(),
        call_names.len(),
        "a repeated header"
    );
    let body: serde_json::Value = serde_json::from_slice(&received.body).expect("JSON");
    assert_eq!(body["sdp"], offer);
    assert_eq!(body["session"]["model"], "gpt-live-1-codex");

    peer.apply_answer(call.answer_sdp.clone())
        .await
        .expect("the peer connects");
    let mut answering_peer = answering;
    let common::Seen::Channel(channel) =
        common::next_seen(&mut answering_peer.seen, "the event channel").await
    else {
        panic!("the channel arrives first");
    };
    channel
        .send_text(r#"{"type":"session.started"}"#)
        .await
        .expect("the answerer sends");
    let over_channel = tokio::time::timeout(common::WAIT, async {
        loop {
            if let Some(PeerEvent::Text(text)) = peer.next_event().await {
                return text;
            }
        }
    })
    .await
    .expect("the data channel carries events");
    assert_eq!(over_channel, r#"{"type":"session.started"}"#);

    let mut control = calls
        .connect_control(
            &rig_tungstenite::TungsteniteClient::new(),
            call.call_id.clone(),
        )
        .await
        .expect("the control socket opens");
    let mut delegation = None;
    let mut acknowledged = false;
    while !acknowledged {
        let event = tokio::time::timeout(Duration::from_secs(10), control.next_event())
            .await
            .expect("an event in time")
            .expect("reads")
            .expect("the socket stays open");
        match event {
            realtime::ServerEvent::SessionStarted(started) => {
                assert_eq!(started.session.id, CALL_ID);
            }
            realtime::ServerEvent::DelegationCreated(created) => {
                control
                    .append_delegation_context(
                        &created.item.id,
                        Some(realtime::ContextChannel::Speakable),
                        "All services are healthy.",
                    )
                    .await
                    .expect("answers");
                delegation = Some(created.item.id);
            }
            realtime::ServerEvent::DelegationContextAppended(appended) => {
                assert_eq!(Some(appended.delegation_item_id), delegation);
                acknowledged = true;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    control.close().await.expect("the session closes");
    let (handshake, client) = control_server.await.expect("the control server finishes");

    assert_eq!(handshake.method, "GET");
    assert_eq!(handshake.target, format!("/v1/live/{CALL_ID}"));
    for (name, value) in EXPECTED_CALL_HEADERS {
        assert_eq!(handshake.single(name), value, "{name}");
    }
    assert_eq!(handshake.single("host"), ws_addr.to_string());
    assert_eq!(handshake.single("connection"), "Upgrade");
    assert_eq!(handshake.single("upgrade"), "websocket");
    assert_eq!(handshake.single("sec-websocket-version"), "13");
    assert!(!handshake.single("sec-websocket-key").is_empty());
    let handshake_names: BTreeSet<&str> = EXPECTED_CALL_HEADERS
        .iter()
        .map(|(name, _)| *name)
        .chain([
            "host",
            "connection",
            "upgrade",
            "sec-websocket-version",
            "sec-websocket-key",
        ])
        .collect();
    assert_eq!(handshake.names(), handshake_names);
    assert_eq!(
        handshake.headers.len(),
        handshake_names.len(),
        "a repeated header"
    );
    assert_eq!(
        client,
        [
            serde_json::json!({
                "type": "delegation.context.append",
                "delegation_item_id": "item_1",
                "channel": "speakable",
                "content": [{"type": "input_text", "text": "All services are healthy."}]
            }),
            serde_json::json!({"type": "session.close"}),
        ]
    );

    peer.close().await.expect("the peer closes");
    answering_peer
        .connection
        .close()
        .await
        .expect("the answerer closes");
}
