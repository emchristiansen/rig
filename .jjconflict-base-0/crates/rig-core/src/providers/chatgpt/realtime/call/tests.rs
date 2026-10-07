//! Call creation against an in-process HTTP double that records the whole
//! request and answers as the backend did for the live probe. No cassette:
//! recording one needs a live WebRTC offer, and the probe's replies are
//! reproduced here from its log (status 201, `text/plain`, `Location`).

use super::*;
use crate::http_client::{HttpClientExt, LazyBody};
use crate::providers::chatgpt;
use crate::wasm_compat::WasmCompatSend;
use bytes::Bytes;
use serde_json::json;
use std::sync::{Arc, Mutex};

const ACCESS_TOKEN: &str = "test-token";
const ACCOUNT_ID: &str = "acct-123";
const OFFER: &str = "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n";
const ANSWER: &str = "v=0\r\no=- 2 2 IN IP4 0.0.0.0\r\n";

/// A request as the double received it.
#[derive(Clone, Debug)]
struct Captured {
    method: http::Method,
    uri: http::Uri,
    headers: http::HeaderMap,
    body: Bytes,
}

/// An HTTP double answering every request with one scripted reply.
#[derive(Clone)]
struct Backend {
    requests: Arc<Mutex<Vec<Captured>>>,
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: Bytes,
    rejection_at: RejectionAt,
}

#[derive(Clone, Copy)]
enum RejectionAt {
    Response,
    Send,
    Body,
    BodyTransport,
}

impl Backend {
    fn created(location: &str) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("text/plain"),
        );
        headers.insert(
            http::header::LOCATION,
            http::HeaderValue::from_str(location).expect("a header value"),
        );
        Self {
            requests: Arc::default(),
            status: http::StatusCode::CREATED,
            headers,
            body: Bytes::from_static(ANSWER.as_bytes()),
            rejection_at: RejectionAt::Response,
        }
    }

    fn failing(status: http::StatusCode, body: &'static str, rejection_at: RejectionAt) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("req-9"));
        Self {
            requests: Arc::default(),
            status,
            headers,
            body: Bytes::from_static(body.as_bytes()),
            rejection_at,
        }
    }

    fn requests(&self) -> Vec<Captured> {
        self.requests.lock().expect("unpoisoned").clone()
    }
}

impl HttpClientExt for Backend {
    fn send<T, U>(
        &self,
        req: http::Request<T>,
    ) -> impl Future<Output = http_client::Result<http::Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let (parts, body) = req.into_parts();
        self.requests.lock().expect("unpoisoned").push(Captured {
            method: parts.method,
            uri: parts.uri,
            headers: parts.headers,
            body: body.into(),
        });
        let reply = self.clone();
        async move {
            if matches!(reply.rejection_at, RejectionAt::Send) && !reply.status.is_success() {
                return Err(http_client::Error::non_success_with_details(
                    reply.status,
                    reply.headers,
                    String::from_utf8_lossy(&reply.body).into_owned(),
                ));
            }
            let response_status = reply.status;
            let response_headers = reply.headers.clone();
            let body: LazyBody<U> = Box::pin(async move {
                if matches!(reply.rejection_at, RejectionAt::BodyTransport) {
                    return Err(http_client::Error::StreamEnded);
                }
                if matches!(reply.rejection_at, RejectionAt::Body) && !reply.status.is_success() {
                    return Err(http_client::Error::non_success_with_details(
                        reply.status,
                        reply.headers,
                        String::from_utf8_lossy(&reply.body).into_owned(),
                    ));
                }
                Ok(U::from(reply.body))
            });
            let mut response = http::Response::builder().status(response_status);
            if let Some(headers) = response.headers_mut() {
                *headers = response_headers;
            }
            response.body(body).map_err(http_client::Error::Protocol)
        }
    }

    fn send_multipart<U>(
        &self,
        _req: http::Request<http_client::MultipartForm>,
    ) -> impl Future<Output = http_client::Result<http::Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        _req: http::Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        std::future::ready(Err(http_client::Error::StreamEnded))
    }
}

#[tokio::test]
async fn a_response_less_body_failure_stays_a_transport_error() {
    let mut backend = Backend::created("/v1/realtime/calls/rtc_test");
    backend.rejection_at = RejectionAt::BodyTransport;
    let error = calls()
        .create_call(&backend, OFFER, &SessionConfig::new("x"))
        .await
        .expect_err("body read failed");
    assert!(matches!(error, ProviderError::Http(_)), "got {error:?}");
    assert_eq!(error.provider_response_status(), None);
}

/// The identity the Codex client stamps: `originator`, its user agent and
/// its version, all three exactly as given.
fn provider() -> OpenAI {
    OpenAI::with_key(&chatgpt::DIALECT, ACCESS_TOKEN)
        .with_account_id(ACCOUNT_ID)
        .with_caller_identity(
            crate::providers::openai::wire::CallerIdentity::new(
                "codex_cli_rs",
                "codex_cli_rs/0.155.1 (NixOS 26.05; x86_64) unknown",
                Some("0.155.1".to_owned()),
            )
            .expect("a valid caller identity"),
        )
}

fn calls() -> LiveCalls {
    LiveCalls::new(provider())
        .expect("the ChatGPT dialect is the Codex backend")
        .with_identity(CodexIdentity::from_ids("session-1", "thread-1").expect("valid ids"))
}

fn sorted_headers(headers: &http::HeaderMap) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                value.to_str().expect("ascii header").to_owned(),
            )
        })
        .collect();
    headers.sort();
    headers
}

/// The headers every request of a call carries, besides content type.
fn call_headers() -> Vec<(String, String)> {
    [
        ("authorization", format!("Bearer {ACCESS_TOKEN}")),
        ("chatgpt-account-id", ACCOUNT_ID.to_owned()),
        ("openai-alpha", "quicksilver=v2".to_owned()),
        ("originator", "codex_cli_rs".to_owned()),
        (
            "user-agent",
            "codex_cli_rs/0.155.1 (NixOS 26.05; x86_64) unknown".to_owned(),
        ),
        ("version", "0.155.1".to_owned()),
        ("session-id", "session-1".to_owned()),
        ("thread-id", "thread-1".to_owned()),
        ("x-session-id", "session-1".to_owned()),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect()
}

/// The request is exactly the probe's: the method, the Codex path and query,
/// every header and nothing else, and the `{sdp, session}` body. The 201
/// reply yields the SDP body and the call id from `Location`.
#[tokio::test]
async fn a_call_is_created_exactly_as_the_probe_created_it() {
    let backend = Backend::created("/v1/realtime/calls/rtc_u2_ETA6qp3oN2ZjwM8wdXUrx6g4jZ3pN2Zx");
    let session = SessionConfig::new("Answer briefly.");
    let call = calls()
        .create_call(&backend, OFFER, &session)
        .await
        .expect("the call is created");

    assert_eq!(call.answer_sdp, ANSWER);
    assert_eq!(
        call.call_id.as_str(),
        "rtc_u2_ETA6qp3oN2ZjwM8wdXUrx6g4jZ3pN2Zx"
    );

    let requests = backend.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method, http::Method::POST);
    assert_eq!(request.uri.scheme_str(), Some("https"));
    assert_eq!(request.uri.host(), Some("chatgpt.com"));
    assert_eq!(request.uri.path(), "/backend-api/codex/realtime/calls");
    assert_eq!(
        request.uri.query(),
        Some("intent=quicksilver&architecture=avas")
    );

    let mut expected = call_headers();
    expected.push(("content-type".to_owned(), "application/json".to_owned()));
    expected.sort();
    assert_eq!(sorted_headers(&request.headers), expected);

    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("a JSON body");
    assert_eq!(
        body,
        json!({
            "sdp": OFFER,
            "session": {
                "model": "gpt-live-1-codex",
                "instructions": "Answer briefly.",
                "audio": {"output": {"voice": "cove"}},
                "delegation": {"type": "client"}
            }
        })
    );
}

/// A non-success reply keeps its status, body and request id, whether the
/// transport hands it back as a response or reports it as an error.
#[tokio::test]
async fn a_rejected_call_keeps_its_status_and_body() {
    for rejection_at in [RejectionAt::Response, RejectionAt::Send, RejectionAt::Body] {
        let backend = Backend::failing(
            http::StatusCode::FORBIDDEN,
            "{\"detail\":\"no\"}",
            rejection_at,
        );
        let error = calls()
            .create_call(&backend, OFFER, &SessionConfig::new("x"))
            .await
            .expect_err("refused");
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::FORBIDDEN)
        );
        assert_eq!(error.provider_response_body(), Some("{\"detail\":\"no\"}"));
        assert_eq!(error.provider_request_id(), Some("req-9"));
        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get("x-request-id")),
            Some(&http::HeaderValue::from_static("req-9"))
        );
    }
}

/// A success without a call id in `Location` is a response error.
#[tokio::test]
async fn a_created_call_without_a_call_id_is_refused() {
    let backend = Backend::created("/v1/realtime/calls/");
    let error = calls()
        .create_call(&backend, OFFER, &SessionConfig::new("x"))
        .await
        .expect_err("no call id");
    assert!(matches!(error, ProviderError::Response(_)), "got {error:?}");
}

/// Call ids are the last matching `Location` segment, as the Codex client
/// reads them: `rtc_…` or a dashed UUID, ignoring the query.
#[test]
fn call_ids_are_read_from_location_like_the_codex_client() {
    let read = |location: &str| CallId::from_location(location).map(|id| id.as_str().to_owned());
    assert_eq!(
        read("/v1/realtime/calls/rtc_abc?x=1").as_deref(),
        Some("rtc_abc")
    );
    assert_eq!(read("rtc_abc/extra").as_deref(), Some("rtc_abc"));
    assert_eq!(
        read("/calls/0123abcd-0123-4567-89ab-0123456789ab").as_deref(),
        Some("0123abcd-0123-4567-89ab-0123456789ab")
    );
    assert_eq!(read("/calls/rtc_"), None);
    assert_eq!(read("/calls/not-a-call"), None);
    assert!(CallId::new("rtc_1").is_ok());
    assert_eq!(
        CallId::new("call_1"),
        Err(InvalidCallId("call_1".to_owned()))
    );
}

/// The control handshake goes to the public API host with the call id as
/// its last segment, and carries the call's headers without a content type.
#[test]
fn the_control_handshake_carries_the_call_headers() {
    let call_id = CallId::new("rtc_u2_test").expect("a call id");
    let request = calls().control_request(&call_id).expect("builds");
    assert_eq!(request.method(), http::Method::GET);
    assert_eq!(request.uri(), "wss://api.openai.com/v1/live/rtc_u2_test");
    let mut expected = call_headers();
    expected.sort();
    assert_eq!(sorted_headers(request.headers()), expected);

    let local = calls().with_control_base_url("ws://127.0.0.1:9/v1/live/");
    assert_eq!(
        local.control_request(&call_id).expect("builds").uri(),
        "ws://127.0.0.1:9/v1/live/rtc_u2_test"
    );
}

/// Without the caller's exact identity nothing is built, and a non-Codex
/// dialect is refused by name.
#[test]
fn calls_need_the_codex_backend_and_the_callers_identity() {
    let anonymous = LiveCalls::new(OpenAI::with_key(&chatgpt::DIALECT, ACCESS_TOKEN))
        .expect("the ChatGPT dialect is the Codex backend");
    assert!(
        anonymous
            .call_request(OFFER, &SessionConfig::new("x"))
            .is_err()
    );
    assert!(
        anonymous
            .control_request(&CallId::new("rtc_1").expect("an id"))
            .is_err()
    );

    let refused = LiveCalls::new(OpenAI::new("sk-test")).expect_err("not the Codex backend");
    assert_eq!(refused.dialect, "openai");
}

/// A separate `x-session-id` replaces only that header.
#[test]
fn the_realtime_session_id_can_differ_from_the_session_id() {
    let calls = calls()
        .with_realtime_session_id("realtime-7")
        .expect("a valid id");
    let request = calls
        .call_request(OFFER, &SessionConfig::new("x"))
        .expect("builds");
    assert_eq!(request.headers()["x-session-id"], "realtime-7");
    assert_eq!(request.headers()["session-id"], "session-1");
    assert!(calls.clone().with_realtime_session_id("").is_err());
}

/// A credential source replaces the static token and account, read once
/// per call.
#[tokio::test]
async fn a_credential_source_is_read_at_send_time() {
    struct Source;
    impl crate::wire::CredentialSource for Source {
        fn current(
            &self,
        ) -> crate::wasm_compat::WasmBoxedFuture<
            '_,
            Result<crate::wire::Credential, crate::wire::CredentialSourceError>,
        > {
            Box::pin(async {
                Ok(
                    crate::wire::Credential::new("rotated-token")
                        .with_account_id("rotated-account"),
                )
            })
        }
    }
    let calls = LiveCalls::new(provider().with_credential_source(Source)).expect("Codex");
    let backend = Backend::created("/calls/rtc_1");
    calls
        .create_call(&backend, OFFER, &SessionConfig::new("x"))
        .await
        .expect("created");
    let headers = &backend.requests()[0].headers;
    assert_eq!(headers["authorization"], "Bearer rotated-token");
    assert_eq!(headers["chatgpt-account-id"], "rotated-account");
    assert_eq!(headers.get_all("authorization").iter().count(), 1);
}

/// The caller's headers follow the conversation identity on call creation
/// and on the control handshake, exactly as given.
#[test]
fn the_callers_headers_reach_every_request_of_a_call() {
    let thread_source = r#"{"thread_source":"user"}"#;
    let calls = calls()
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-turn-metadata", thread_source)
                .expect("a valid header"),
        )
        .expect("not a header this protocol owns");
    let mut expected = call_headers();
    expected.push(("x-codex-turn-metadata".to_owned(), thread_source.to_owned()));

    let control = calls
        .control_request(&CallId::new("rtc_1").expect("an id"))
        .expect("builds");
    let mut control_expected = expected.clone();
    control_expected.sort();
    assert_eq!(sorted_headers(control.headers()), control_expected);

    let creation = calls
        .call_request(OFFER, &SessionConfig::new("x"))
        .expect("builds");
    expected.push(("content-type".to_owned(), "application/json".to_owned()));
    expected.sort();
    assert_eq!(sorted_headers(creation.headers()), expected);
}

/// The headers this protocol sends itself cannot be supplied.
#[test]
fn the_protocols_own_headers_cannot_be_supplied() {
    for name in [
        "openai-alpha",
        "OpenAI-Alpha",
        "x-session-id",
        "X-Session-Id",
    ] {
        let headers = RequestHeaders::new()
            .with(name, "caller-value")
            .expect("not reserved for Responses");
        let refused = calls()
            .with_request_headers(headers)
            .expect_err("owned by this protocol");
        assert_eq!(
            refused,
            InvalidRequestHeader::Reserved {
                name: name.to_ascii_lowercase()
            }
        );
    }
}

/// The headers the transport writes itself cannot be supplied either, in any
/// case.
#[test]
fn the_transports_own_headers_cannot_be_supplied() {
    for name in [
        "Host",
        "Connection",
        "Upgrade",
        "Sec-WebSocket-Key",
        "Sec-WebSocket-Version",
        "Content-Length",
        "Accept",
    ] {
        assert_eq!(
            RequestHeaders::new().with(name, "caller-value"),
            Err(InvalidRequestHeader::Reserved {
                name: name.to_ascii_lowercase()
            }),
            "{name}"
        );
    }
}

/// A credential source re-stamps the credential and account without
/// touching the caller's headers.
#[tokio::test]
async fn a_credential_source_keeps_the_callers_headers() {
    struct Source;
    impl crate::wire::CredentialSource for Source {
        fn current(
            &self,
        ) -> crate::wasm_compat::WasmBoxedFuture<
            '_,
            Result<crate::wire::Credential, crate::wire::CredentialSourceError>,
        > {
            Box::pin(async {
                Ok(
                    crate::wire::Credential::new("rotated-token")
                        .with_account_id("rotated-account"),
                )
            })
        }
    }
    let calls = LiveCalls::new(provider().with_credential_source(Source))
        .expect("Codex")
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-turn-metadata", r#"{"thread_source":"user"}"#)
                .expect("a valid header"),
        )
        .expect("not a header this protocol owns");
    let backend = Backend::created("/calls/rtc_1");
    calls
        .create_call(&backend, OFFER, &SessionConfig::new("x"))
        .await
        .expect("created");
    let headers = &backend.requests()[0].headers;
    let all = |name: &str| -> Vec<&str> {
        headers
            .get_all(name)
            .iter()
            .map(|value| value.to_str().expect("ascii"))
            .collect()
    };
    assert_eq!(
        all("x-codex-turn-metadata"),
        [r#"{"thread_source":"user"}"#]
    );
    assert_eq!(all("authorization"), ["Bearer rotated-token"]);
    assert_eq!(all("chatgpt-account-id"), ["rotated-account"]);
}
