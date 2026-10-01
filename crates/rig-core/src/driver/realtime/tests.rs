//! Public Live session creation against an in-process HTTP double that records
//! every request and answers with one scripted reply. No cassette: recording
//! a creation needs a live WebRTC offer and a billed session. The success
//! body is the Live API reference's example; the error bodies use OpenAI's
//! `{"error": {...}}` envelope with the spend-limit code the spend-limits
//! guide names.

use super::*;
use crate::providers::openai::OpenAI;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use crate::wire::{Credential, CredentialSource, CredentialSourceError};
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
    /// As a body read that fails with no reply.
    BodyTransport,
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
            if matches!(reply.delivery, Delivery::SendError) {
                return Err(http_client::Error::non_success_with_details(
                    reply.status,
                    reply.headers,
                    String::from_utf8_lossy(&reply.body).into_owned(),
                ));
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
    PublicLiveSessions::new(OpenAI::new("sk-test")).expect("the OpenAI dialect")
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
async fn a_credential_source_failure_sends_nothing() {
    struct Locked;
    impl CredentialSource for Locked {
        fn current(&self) -> WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>> {
            Box::pin(async { Err("the vault is locked".into()) })
        }
    }
    let sessions = PublicLiveSessions::new(OpenAI::new("sk-test").with_credential_source(Locked))
        .expect("the OpenAI dialect");
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::Reply);
    let error = sessions
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("no credential");
    assert!(matches!(error, LiveApiError::Request(_)), "got {error:?}");
    assert_eq!(backend.sent(), 0);
    assert!(!error.is_terminal());
}

#[tokio::test]
async fn a_credential_source_is_read_at_send_time() {
    struct Rotated;
    impl CredentialSource for Rotated {
        fn current(&self) -> WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>> {
            Box::pin(async { Ok(Credential::new("sk-rotated")) })
        }
    }
    let sessions = PublicLiveSessions::new(OpenAI::new("sk-test").with_credential_source(Rotated))
        .expect("the OpenAI dialect");
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::Reply);
    sessions
        .create_session(&backend, OFFER, &session())
        .await
        .expect("a created session");
    assert_eq!(
        backend.request_header("authorization"),
        Some(http::HeaderValue::from_static("Bearer sk-rotated"))
    );
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
        assert_eq!(
            reply.response.provider_request_id.as_deref(),
            Some("req_42")
        );
        assert_eq!(reply.response.body, body);
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
    assert_eq!(
        reply.response.provider_request_id.as_deref(),
        Some("req_42")
    );
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
        assert_eq!(
            reply.response.provider_request_id.as_deref(),
            Some("req_42")
        );
        assert!(error.is_terminal());
        assert!(!error.is_retryable());
    }
}

#[tokio::test]
async fn a_body_read_without_a_reply_is_a_transport_failure() {
    let backend = Backend::new(http::StatusCode::CREATED, CREATED, Delivery::BodyTransport);
    let error = sessions()
        .create_session(&backend, OFFER, &session())
        .await
        .expect_err("the body read failed");
    let LiveApiError::Transport(inner) = &error else {
        panic!("expected a transport failure, got {error:?}");
    };
    assert!(matches!(inner, ProviderError::Http(_)), "got {inner:?}");
    assert!(error.is_retryable());
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
