//! Public Live session creation against an in-process HTTP double that records
//! every request and answers with one scripted reply. No cassette: recording
//! a creation needs a live WebRTC offer and a billed session. The success
//! body is the Live API reference's example; the error bodies use OpenAI's
//! `{"error": {...}}` envelope with the spend-limit code the spend-limits
//! guide names.

use super::*;
use crate::providers::live_support::LiveConfiguration;
use crate::wasm_compat::WasmCompatSend;
use std::sync::{Arc, Mutex};

const OFFER: &str = "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n";
const CREATED: &str =
    r#"{"session":{"id":"live_123"},"transport":{"type":"webrtc","sdp":"<SDP answer>"}}"#;
const SPEND_LIMIT: &str = r#"{"error":{"message":"You exceeded your hard spend limit.","type":"insufficient_quota","param":null,"code":"project_spend_limit_exceeded"}}"#;

/// Where the double reports its scripted reply.
#[derive(Clone, Copy)]
enum Delivery {
    /// As an ordinary response.
    Reply,
    /// As a transport error carrying the reply, from `send`.
    SendError,
    /// As a response whose body read then fails.
    BodyTransport,
    /// As a send that fails before any reply.
    SendTransport,
    /// Legacy error without headers, retaining text.
    LegacyMessage,
    /// Legacy error without headers or body.
    LegacyStatus,
}

/// An HTTP double answering every request with one scripted reply.
#[derive(Clone)]
struct Backend {
    requests: Arc<Mutex<Vec<http::Request<Bytes>>>>,
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: Bytes,
    delivery: Delivery,
}

impl Backend {
    fn new(status: http::StatusCode, body: &'static str, delivery: Delivery) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("req_42"));
        Self {
            requests: Arc::default(),
            status,
            headers,
            body: Bytes::from_static(body.as_bytes()),
            delivery,
        }
    }

    fn sent(&self) -> usize {
        self.requests.lock().expect("unpoisoned").len()
    }

    fn request_header(&self, name: &str) -> Option<http::HeaderValue> {
        self.requests
            .lock()
            .expect("unpoisoned")
            .first()
            .and_then(|request| request.headers().get(name).cloned())
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
        self.requests
            .lock()
            .expect("unpoisoned")
            .push(http::Request::from_parts(parts, body.into()));
        let reply = self.clone();
        async move {
            if matches!(reply.delivery, Delivery::SendTransport) {
                return Err(http_client::Error::StreamEnded);
            }
            if matches!(reply.delivery, Delivery::LegacyStatus) { return Err(http_client::Error::InvalidStatusCode(reply.status)); }
            if matches!(reply.delivery, Delivery::LegacyMessage) { return Err(http_client::Error::InvalidStatusCodeWithMessage(reply.status, String::from_utf8_lossy(&reply.body).into_owned())); }
            if matches!(reply.delivery, Delivery::SendError) {
                return Err(http_client::Error::InvalidStatusCodeWithDetails { status: reply.status, headers: Box::new(reply.headers), body: String::from_utf8_lossy(&reply.body).into_owned() });
            }
            let status = reply.status;
            let headers = reply.headers.clone();
            let body: LazyBody<U> = Box::pin(async move {
                if matches!(reply.delivery, Delivery::BodyTransport) {
                    return Err(http_client::Error::StreamEnded);
                }
                Ok(U::from(reply.body))
            });
            let mut response = http::Response::builder().status(status);
            if let Some(response_headers) = response.headers_mut() {
                *response_headers = headers;
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

fn sessions() -> PublicLiveSessions {
    PublicLiveSessions::new(LiveConfiguration::public("sk-test")).expect("the OpenAI dialect")
}

fn session() -> PublicSessionConfig {
    PublicSessionConfig::new().with_instructions("Be concise.")
}

#[tokio::test]
async fn a_created_session_decodes() {
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::Reply);
    let created = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect("a created session");
    assert_eq!(
        created,
        CreatedSession {
            session_id: "live_123".to_owned(),
            answer_sdp: "<SDP answer>".to_owned(),
        }
    );
    assert_eq!(backend.sent(), 1);
    assert_eq!(
        backend.request_header("authorization"),
        Some(http::HeaderValue::from_static("Bearer sk-test"))
    );
}

#[tokio::test]
async fn invalid_static_access_sends_nothing() {
    let sessions = PublicLiveSessions::new(LiveConfiguration::public("invalid\ntoken")).expect("public");
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::Reply);
    let error=sessions.create_session(&backend,OFFER,&session()).await.expect_err("invalid header");
    assert!(matches!(error,LiveApiError::Request(_)));
    assert_eq!(backend.sent(),0);
}

#[tokio::test]
async fn a_rejected_reply_keeps_its_request_id_however_it_arrives() {
    let body = r#"{"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#;
    for delivery in [Delivery::Reply, Delivery::SendError] {
        let backend = Backend::new(http::StatusCode::UNAUTHORIZED, body, delivery);
        let error = sessions()
            .create_session(&backend, OFFER, &session())
            .await
            .expect_err("a rejected key");
        let LiveApiError::Authentication(reply) = &error else {
            panic!("expected an authentication failure, got {error:?}");
        };
        assert_eq!(reply.reply.head.request_id.as_deref(), Some("req_42"));
        assert_eq!(reply.reply.body.bytes(), Some(body.as_bytes()));
        assert!(error.is_terminal());
    }

    let backend = Backend::new(
        http::StatusCode::BAD_REQUEST,
        r#"{"error":{"message":"Bad.","type":"invalid_request_error","code":"unknown_parameter"}}"#,
        Delivery::SendError,
    );
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("a bad request");
    let LiveApiError::Rejected(reply) = &error else {
        panic!("expected a rejection, got {error:?}");
    };
    assert_eq!(reply.reply.head.request_id.as_deref(), Some("req_42"));
}

#[tokio::test]
async fn a_spend_limit_429_stays_terminal_however_it_arrives() {
    for delivery in [Delivery::Reply, Delivery::SendError] {
        let backend = Backend::new(http::StatusCode::TOO_MANY_REQUESTS, SPEND_LIMIT, delivery);
        let error = sessions()
            .create_session(&backend, OFFER, &session())
            .await
            .expect_err("a reached spend limit");
        let LiveApiError::SpendLimit(reply) = &error else {
            panic!("expected a spend limit, got {error:?}");
        };
        assert_eq!(reply.reply.head.request_id.as_deref(), Some("req_42"));
        assert!(error.is_terminal());
        assert!(!error.is_retryable());
    }
}

#[tokio::test]
async fn a_send_without_a_reply_is_a_transport_failure() {
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::SendTransport);
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("the send failed");
    let LiveApiError::Transport(inner) = &error else {
        panic!("expected a transport failure, got {error:?}");
    };
    assert!(matches!(inner, ProviderError::Http(_)), "got {inner:?}");
    assert!(error.head().is_none());
    assert!(error.is_retryable());
    assert!(!error.is_terminal());
}

#[tokio::test]
async fn an_unreadable_created_body_is_an_unknown_outcome() {
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::BodyTransport);
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("the body read failed");
    let LiveApiError::OutcomeUnknown { head, read_error } = &error else {
        panic!("expected an unknown outcome, got {error:?}");
    };
    assert_eq!(head.status, http::StatusCode::CREATED);
    assert_eq!(head.request_id.as_deref(), Some("req_42"));
    assert!(matches!(read_error, ProviderError::Http(_)));
    assert!(!error.is_retryable());
    assert!(!error.is_terminal());
}

#[tokio::test]
async fn a_rejected_credential_with_an_unreadable_body_stays_terminal() {
    for status in [http::StatusCode::UNAUTHORIZED, http::StatusCode::FORBIDDEN] {
        let backend = Backend::new(status, "", Delivery::BodyTransport);
        let error = sessions()
            .create_session(&backend, OFFER, &session())
            .await
            .expect_err("a rejected credential");
        let LiveApiError::Authentication(reply) = &error else {
            panic!("expected an authentication failure, got {error:?}");
        };
        assert_eq!(reply.reply.head.status, status);
        assert_eq!(reply.reply.head.request_id.as_deref(), Some("req_42"));
        assert!(matches!(reply.reply.body, ReplyBody::Unreadable(_)));
        assert!(reply.error.is_none());
        assert!(error.is_terminal());
        assert!(!error.is_retryable());
    }
}

#[tokio::test]
async fn a_429_with_an_unreadable_body_is_not_a_spend_limit() {
    let backend = Backend::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        SPEND_LIMIT,
        Delivery::BodyTransport,
    );
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("a 429");
    let LiveApiError::Rejected(reply) = &error else {
        panic!("expected a rejection with an unknown code, got {error:?}");
    };
    assert_eq!(reply.reply.head.status, http::StatusCode::TOO_MANY_REQUESTS);
    assert!(reply.error.is_none());
    assert!(matches!(reply.reply.body, ReplyBody::Unreadable(_)));
    assert!(!error.is_terminal());
}

#[tokio::test]
async fn a_malformed_success_is_not_a_transport_failure() {
    let backend = Backend::new(
        http::StatusCode::CREATED,
        r#"{"session":{"id":"live_123"},"transport":{"type":"sip"}}"#,
        Delivery::Reply,
    );
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("no SDP answer");
    assert!(matches!(error, LiveApiError::Malformed(_)), "got {error:?}");
    assert!(!error.is_retryable());
}

/// Legacy transports report only what they actually supplied; status still
/// decides authentication and a spend limit still requires an observed code.
#[tokio::test]
async fn legacy_replies_preserve_absence_and_terminal_classification() {
    for delivery in [Delivery::LegacyMessage, Delivery::LegacyStatus] {
        let backend=Backend::new(http::StatusCode::UNAUTHORIZED,"rejected",delivery);
        let error=sessions().create_session(&backend,OFFER,&session()).await.expect_err("rejected");
        let LiveApiError::Authentication(reply)=&error else { panic!("expected authentication, {error:?}") };
        assert!(!reply.reply.head.headers_available);
        assert!(reply.reply.head.headers.is_empty());
        assert!(reply.reply.head.request_id.is_none());
        match delivery {
            Delivery::LegacyStatus=>assert!(matches!(reply.reply.body,ReplyBody::Unavailable)),
            Delivery::LegacyMessage=>assert_eq!(reply.reply.body.bytes(),Some(&b"rejected"[..])),
            _=>unreachable!(),
        }
        assert!(error.is_terminal());
    }
    let backend=Backend::new(http::StatusCode::TOO_MANY_REQUESTS,SPEND_LIMIT,Delivery::LegacyStatus);
    let error=sessions().create_session(&backend,OFFER,&session()).await.expect_err("rejected");
    assert!(matches!(error,LiveApiError::Rejected(_)));
    let backend=Backend::new(http::StatusCode::TOO_MANY_REQUESTS,SPEND_LIMIT,Delivery::LegacyMessage);
    let error=sessions().create_session(&backend,OFFER,&session()).await.expect_err("rejected");
    assert!(matches!(error,LiveApiError::SpendLimit(_)));
}
