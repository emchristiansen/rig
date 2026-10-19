//! Wire conformance for `openai_responses_websocket` over the bundled tungstenite
//! connection and a local server, followed by the existing Responses decoder.
//!
//! The frames are the shared OpenAI Responses fixture's, re-wrapped as one
//! JSON websocket message per SSE `data:` line. The wire events are identical
//! across the two transports, only the framing differs. All generic scenarios
//! run below `ResponsesWebSocketSession`: its public events do not retain the
//! full lifecycle envelope needed to replay the original payload.
//!
//! The corruption cases exercise decoder-stage recovery through a genuine
//! terminal. They do not assert public-session recovery: that session aborts on
//! corrupt frames, as its separate protocol tests require.

#![cfg(not(target_family = "wasm"))]
use futures::{SinkExt, StreamExt};
use rig_core::error::ProviderError;
use rig_core::http_client::{NoBody, Request};
use rig_core::test_utils::streaming_conformance::{
    self as conformance, fixtures::openai_responses,
};
use rig_core::ws_client::{
    BoxedWebSocketConnection, ConnectOptions, Frame, WebSocketClientExt as _,
};

use rig_tungstenite::TungsteniteClient;
use rig_tungstenite::tokio_tungstenite::{accept_async, tungstenite::Message};
use std::time::Duration;
use tokio::net::TcpListener;

const DEADLINE: Duration = Duration::from_secs(10);

/// Lower the fixture's byte frames onto ws text messages (one per `data:`
/// line); an `Err` chunk truncates the script and marks an abrupt abort.
fn ws_script(chunks: conformance::WireChunks) -> Result<(Vec<String>, bool), ProviderError> {
    let mut messages = Vec::new();
    for chunk in chunks {
        match chunk {
            Ok(frame) => {
                let bytes = frame.as_bytes().cloned().ok_or_else(|| {
                    ProviderError::Provider(
                        "typed-event frame fed to the websocket driver".to_string(),
                    )
                })?;
                let text = std::str::from_utf8(&bytes).map_err(|error| {
                    ProviderError::Provider(format!("non-UTF-8 fixture frame: {error}"))
                })?;
                messages.extend(
                    text.lines()
                        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                        .filter(|data| !data.is_empty() && *data != "[DONE]")
                        .map(ToOwned::to_owned),
                );
            }
            // A scripted transport failure: everything after it is undeliverable.
            Err(_) => return Ok((messages, true)),
        }
    }
    Ok((messages, false))
}

/// Serve one websocket turn: upgrade, read the `response.create` request,
/// send the scripted messages, then end the connection — abruptly (no close
/// handshake, the client observes a transport reset) when `abort`, cleanly
/// otherwise.
fn spawn_server(listener: TcpListener, messages: Vec<String>, abort: bool) {
    tokio::spawn(async move {
        let _ = tokio::time::timeout(DEADLINE, async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut socket) = accept_async(stream).await else {
                return;
            };
            let _ = socket.next().await;
            for message in messages {
                if socket.send(Message::text(message)).await.is_err() {
                    return;
                }
            }
            if abort {
                drop(socket);
            } else {
                let _ = socket.close(None).await;
            }
        })
        .await;
    });
}

/// Reframe the received text payloads for the shared Responses decoder driver,
/// preserving corrupt payloads and the connection's actual transport error.
async fn drain_openai_responses_websocket_frames(
    mut connection: BoxedWebSocketConnection,
) -> Result<conformance::DrainedStream, ProviderError> {
    let mut chunks = Vec::new();
    loop {
        match connection.recv().await {
            Ok(Some(Frame::Text(payload))) => {
                chunks.push(Ok(conformance::WireInput::Bytes(bytes::Bytes::from(
                    format!("data: {payload}\n\n"),
                ))));
            }
            Ok(Some(Frame::Close(_))) | Ok(None) => break,
            Ok(Some(frame)) => {
                return Err(ProviderError::Provider(format!(
                    "expected a scripted websocket text frame, received {frame:?}"
                )));
            }
            Err(error) => {
                chunks.push(Err(error));
                break;
            }
        }
    }
    openai_responses::driver().drive(chunks).await
}

fn driver() -> conformance::WireDriver {
    conformance::WireDriver::new("openai-responses-websocket", |chunks| {
        Box::pin(async move {
            tokio::time::timeout(DEADLINE, async move {
                let (messages, abort) = ws_script(chunks)?;
                let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|error| {
                    ProviderError::Provider(format!("listener bind failed: {error}"))
                })?;
                let address = listener.local_addr().map_err(|error| {
                    ProviderError::Provider(format!("listener address failed: {error}"))
                })?;
                spawn_server(listener, messages, abort);

                let request = Request::builder()
                    .uri(format!("ws://{address}/v1/responses"))
                    .body(NoBody)?;
                let mut connection = TungsteniteClient::new()
                    .connect(request, ConnectOptions::new().with_timeout(Some(DEADLINE)))
                    .await?;
                connection
                    .send(Frame::Text(
                        serde_json::json!({
                            "type": "response.create",
                            "model": "gpt-5.4",
                            "input": "hello",
                        })
                        .to_string(),
                    ))
                    .await?;

                drain_openai_responses_websocket_frames(connection).await
            })
            .await
            .map_err(|error| {
                ProviderError::Provider(format!("websocket conformance timed out: {error}"))
            })?
        })
    })
}

fn fixture() -> conformance::ProviderWireFixture {
    conformance::ProviderWireFixture {
        driver: driver(),
        ..openai_responses::fixture()
    }
}

pub mod openai_responses_websocket_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "openai_responses_websocket",
        fixture: fixture(),
        manifest: [partial_tool_args, zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, refusal],
    }
}

/// Compile-linked manifest of the wire families this binary covers.
///
/// This suite lives outside the `rig` facade's `core` test binary, so the
/// workspace registry cannot link it: it lists `openai_responses_websocket` in
/// `OUT_OF_BINARY_FAMILIES` and relies on the "Test out-of-facade streaming
/// conformance and structural guards" CI step to execute this binary. The test
/// below keeps the family name honest at the definition site, which is the
/// direction the registry loses for out-of-binary suites (#2258 F3).
const SUITE_FAMILIES: &[&str] = &[openai_responses_websocket_suite::WIRE_FAMILY];

#[test]
fn suite_families_are_registered_wire_families() {
    for family in SUITE_FAMILIES {
        assert!(
            rig_core::test_utils::streaming_conformance::WIRE_FAMILIES.contains(family),
            "suite names wire family {family:?}, absent from WIRE_FAMILIES"
        );
    }
}
