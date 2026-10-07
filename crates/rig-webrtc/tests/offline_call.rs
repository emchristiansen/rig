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
use rig_core::providers::chatgpt::{self, realtime};
use rig_core::providers::openai::OpenAI;
use rig_core::providers::openai::responses_api::codex_identity::CodexIdentity;
use rig_core::providers::openai::responses_api::request_headers::RequestHeaders;
use rig_core::providers::openai::wire::CallerIdentity;
use rig_webrtc::{LivePeer, PeerEvent};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
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

/// Serve one HTTP/1.1 request: record it, answer the offer in its body with
/// an in-process peer, and reply 201 with the answer and a `Location`.
async fn serve_call(listener: TcpListener) -> (Received, common::AnsweringPeer) {
    let (mut stream, _) = listener.accept().await.expect("a connection");
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
    let length: usize = received.single("content-length").parse().expect("a length");
    let mut body = buffer[head_end..].to_vec();
    while body.len() < length {
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk).await.expect("reads");
        body.extend_from_slice(&chunk[..read]);
    }
    received.body = body;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_call_runs_end_to_end_against_local_servers() {
    let http = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let ws = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let http_addr = http.local_addr().expect("an address");
    let ws_addr = ws.local_addr().expect("an address");
    let call_server = tokio::spawn(serve_call(http));
    let control_server = tokio::spawn(serve_control(ws));

    let identity = CallerIdentity::new(
        "codex_cli_rs",
        "codex_cli_rs/0.155.1 (NixOS 26.05; x86_64) unknown",
        Some("0.155.1".to_owned()),
    )
    .expect("an identity");
    let provider = OpenAI::with_key(&chatgpt::DIALECT, "test-token")
        .with_base_url(format!("http://{http_addr}/backend-api/codex"))
        .with_account_id("acct-123")
        .with_caller_identity(identity);
    let calls = realtime::LiveCalls::new(provider)
        .expect("the Codex backend")
        .with_identity(CodexIdentity::from_ids("session-1", "thread-1").expect("ids"))
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-turn-metadata", r#"{"thread_source":"user"}"#)
                .expect("a valid header"),
        )
        .expect("not a Live header")
        .with_control_base_url(format!("ws://{ws_addr}/v1/live"));

    let peer = LivePeer::builder()
        .with_udp_addrs(vec!["127.0.0.1:0".to_owned()])
        .with_tcp_addrs(Vec::new())
        .with_loopback_candidates(true)
        .build()
        .await
        .expect("the peer builds");
    let offer = peer.offer().await.expect("an offer");
    let call = calls
        .create_call(
            &rig_reqwest::ReqwestClient::default(),
            &offer,
            &realtime::SessionConfig::new("Answer briefly."),
        )
        .await
        .expect("the call is created");
    assert_eq!(call.call_id.as_str(), CALL_ID);
    let (received, answering) = call_server.await.expect("the call server finishes");
    assert_eq!(call.answer_sdp, answering.answer);

    assert_eq!(received.method, "POST");
    assert_eq!(
        received.target,
        "/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas"
    );
    let expected_call_headers = [
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
    for (name, value) in expected_call_headers {
        assert_eq!(received.single(name), value, "{name}");
    }
    assert_eq!(received.single("content-type"), "application/json");
    assert_eq!(received.single("accept"), "*/*");
    assert_eq!(received.single("host"), http_addr.to_string());
    let call_names: BTreeSet<&str> = expected_call_headers
        .iter()
        .map(|(name, _)| *name)
        .chain(["content-type", "content-length", "host", "accept"])
        .collect();
    assert_eq!(received.names(), call_names);
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
    for (name, value) in expected_call_headers {
        assert_eq!(handshake.single(name), value, "{name}");
    }
    assert_eq!(handshake.single("host"), ws_addr.to_string());
    assert_eq!(handshake.single("connection"), "Upgrade");
    assert_eq!(handshake.single("upgrade"), "websocket");
    assert_eq!(handshake.single("sec-websocket-version"), "13");
    assert!(!handshake.single("sec-websocket-key").is_empty());
    let handshake_names: BTreeSet<&str> = expected_call_headers
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
