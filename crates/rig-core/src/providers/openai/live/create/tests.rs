//! Session-creation requests and reply decoding. These are unit tests, not
//! cassette tests: recording a creation needs a live WebRTC offer and a
//! billed session. The request and success shapes are the Live API
//! reference's examples. The error bodies use OpenAI's `{"error": {...}}`
//! envelope with the spend-limit codes the spend-limits guide names; their
//! other fields are illustrative.

use super::*;
use crate::providers::chatgpt;
use serde_json::json;

const API_KEY: &str = "sk-test";
const OFFER: &str = "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n";

fn sessions() -> PublicLiveSessions {
    PublicLiveSessions::new(OpenAI::new(API_KEY)).expect("the OpenAI dialect")
}

fn headers_with_request_id() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", http::HeaderValue::from_static("req_42"));
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    headers
}

#[test]
fn the_request_matches_the_reference_example() {
    let session =
        SessionConfig::new().with_instructions("Be concise. Ask for clarification when needed.");
    let request = sessions()
        .create_request("<SDP offer>", &session)
        .expect("a request");

    assert_eq!(request.method(), http::Method::POST);
    assert_eq!(
        request.uri().to_string(),
        "https://api.openai.com/v1/live/sessions"
    );
    assert_eq!(
        request.headers().get(http::header::AUTHORIZATION),
        Some(&http::HeaderValue::from_static("Bearer sk-test"))
    );
    assert_eq!(
        request.headers().get(http::header::CONTENT_TYPE),
        Some(&http::HeaderValue::from_static("application/json"))
    );
    let body: serde_json::Value = serde_json::from_slice(request.body()).expect("JSON");
    assert_eq!(
        body,
        json!({
            "session": {
                "model": "gpt-live-1",
                "instructions": "Be concise. Ask for clarification when needed."
            },
            "transport": {"type": "webrtc", "sdp": "<SDP offer>"}
        })
    );
}

#[test]
fn the_offer_is_sent_verbatim_under_a_custom_base_url() {
    let sessions =
        PublicLiveSessions::new(OpenAI::new(API_KEY).with_base_url("http://localhost:8080/v1/"))
            .expect("the OpenAI dialect");
    let request = sessions
        .create_request(OFFER, &SessionConfig::new())
        .expect("a request");
    assert_eq!(
        request.uri().to_string(),
        "http://localhost:8080/v1/live/sessions"
    );
    let body: serde_json::Value = serde_json::from_slice(request.body()).expect("JSON");
    assert_eq!(body["transport"]["sdp"], json!(OFFER));
}

#[test]
fn an_empty_offer_is_refused() {
    let error = sessions()
        .create_request("", &SessionConfig::new())
        .expect_err("an empty offer");
    assert_eq!(
        ProviderError::from(error).to_string(),
        "RequestError: the SDP offer is empty"
    );
}

#[test]
fn the_codex_backend_is_refused() {
    let provider = OpenAI::with_key(&chatgpt::DIALECT, "access-token");
    let error = PublicLiveSessions::new(provider).expect_err("the Codex dialect");
    assert_eq!(
        error,
        NotOfficialOpenAi {
            dialect: chatgpt::DIALECT.name
        }
    );
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::Request(_)
    ));
}

#[test]
fn another_openai_shaped_gateway_is_refused() {
    let provider = OpenAI::with_key(&crate::providers::openai::wire::GROQ, "gsk-test");
    let error = PublicLiveSessions::new(provider).expect_err("the Groq dialect");
    assert_eq!(error, NotOfficialOpenAi { dialect: "groq" });
}

#[test]
fn the_official_openai_dialect_is_accepted() {
    assert!(PublicLiveSessions::new(OpenAI::new(API_KEY)).is_ok());
}

#[test]
fn the_dialect_supplies_the_request_id_header_and_no_stamp_by_default() {
    let sessions = sessions();
    assert_eq!(sessions.request_id_header(), Some("x-request-id"));
    assert!(sessions.credential_stamp().is_none());
}

#[test]
fn the_reference_201_body_decodes() {
    let body = br#"{
  "session": {
    "id": "live_123"
  },
  "transport": {
    "type": "webrtc",
    "sdp": "<SDP answer>"
  }
}"#;
    let created = sessions()
        .decode_reply(http::StatusCode::CREATED, &http::HeaderMap::new(), body)
        .expect("a created session");
    assert_eq!(
        created,
        CreatedSession {
            session_id: "live_123".to_owned(),
            answer_sdp: "<SDP answer>".to_owned(),
        }
    );
}

#[test]
fn a_success_without_a_webrtc_answer_is_malformed() {
    let sip = br#"{"session":{"id":"live_123"},"transport":{"type":"sip"}}"#;
    let error = sessions()
        .decode_reply(http::StatusCode::CREATED, &http::HeaderMap::new(), sip)
        .expect_err("no SDP answer");
    let LiveApiError::Malformed(ProviderError::Response(message)) = &error else {
        panic!("expected a malformed reply, got {error:?}");
    };
    assert!(message.contains(r#""type":"sip""#), "{message}");
    assert!(!error.is_terminal());
    assert!(!error.is_retryable());

    let not_json = b"<html>oops</html>";
    let error = sessions()
        .decode_reply(http::StatusCode::CREATED, &http::HeaderMap::new(), not_json)
        .expect_err("not JSON");
    let LiveApiError::Malformed(ProviderError::Response(message)) = &error else {
        panic!("expected a malformed reply, got {error:?}");
    };
    assert!(message.ends_with("body: <html>oops</html>"), "{message}");
}

#[test]
fn a_401_is_a_terminal_authentication_failure() {
    let body = r#"{"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#;
    let error = sessions()
        .decode_reply(
            http::StatusCode::UNAUTHORIZED,
            &headers_with_request_id(),
            body.as_bytes(),
        )
        .expect_err("a rejected key");
    let LiveApiError::Authentication(reply) = &error else {
        panic!("expected an authentication failure, got {error:?}");
    };
    assert_eq!(
        reply.error.as_ref().and_then(|error| error.code.as_deref()),
        Some("invalid_api_key")
    );
    assert_eq!(reply.response.status, Some(http::StatusCode::UNAUTHORIZED));
    assert_eq!(reply.response.body, body);
    assert_eq!(
        reply.response.provider_request_id.as_deref(),
        Some("req_42")
    );
    assert!(reply.response.headers.is_some());
    assert!(error.is_terminal());
    assert!(!error.is_retryable());

    let provider_error = ProviderError::from(error);
    assert!(matches!(
        provider_error,
        ProviderError::InvalidAuthentication(_)
    ));
    assert_eq!(provider_error.provider_request_id(), Some("req_42"));
}

#[test]
fn a_403_is_a_terminal_authentication_failure() {
    let body = br#"{"error":{"message":"Project does not have access to model gpt-live-1.","type":"invalid_request_error","code":"model_not_found"}}"#;
    let error = sessions()
        .decode_reply(http::StatusCode::FORBIDDEN, &http::HeaderMap::new(), body)
        .expect_err("a forbidden key");
    assert!(matches!(error, LiveApiError::Authentication(_)));
    assert!(error.is_terminal());
}

#[test]
fn a_429_spend_limit_is_terminal() {
    for code in [
        "project_spend_limit_exceeded",
        "organization_spend_limit_exceeded",
    ] {
        let body = format!(
            r#"{{"error":{{"message":"You exceeded your hard spend limit.","type":"insufficient_quota","param":null,"code":"{code}"}}}}"#
        );
        let error = sessions()
            .decode_reply(
                http::StatusCode::TOO_MANY_REQUESTS,
                &headers_with_request_id(),
                body.as_bytes(),
            )
            .expect_err("a reached spend limit");
        let LiveApiError::SpendLimit(reply) = &error else {
            panic!("expected a spend limit, got {error:?}");
        };
        assert_eq!(
            reply.error.as_ref().and_then(|error| error.code.as_deref()),
            Some(code)
        );
        assert_eq!(reply.response.body, body);
        assert!(error.is_terminal(), "{code}");
        assert!(!error.is_retryable(), "{code}");
    }
}

#[test]
fn a_429_rate_limit_is_retryable_not_terminal() {
    let body = br#"{"error":{"message":"Rate limit reached.","type":"requests","param":null,"code":"rate_limit_exceeded"}}"#;
    let error = sessions()
        .decode_reply(
            http::StatusCode::TOO_MANY_REQUESTS,
            &http::HeaderMap::new(),
            body,
        )
        .expect_err("a rate limit");
    assert!(matches!(error, LiveApiError::Rejected(_)));
    assert!(!error.is_terminal());
    assert!(error.is_retryable());
}

#[test]
fn other_failures_are_rejections_keeping_the_reply() {
    let bad_request = br#"{"error":{"message":"Unknown parameter: 'session.voice'.","type":"invalid_request_error","param":"session.voice","code":"unknown_parameter"}}"#;
    let error = sessions()
        .decode_reply(
            http::StatusCode::BAD_REQUEST,
            &http::HeaderMap::new(),
            bad_request,
        )
        .expect_err("a bad request");
    let LiveApiError::Rejected(reply) = &error else {
        panic!("expected a rejection, got {error:?}");
    };
    let detail = reply.error.as_ref().expect("an error object");
    assert_eq!(detail.param.as_deref(), Some("session.voice"));
    assert_eq!(detail.kind.as_deref(), Some("invalid_request_error"));
    assert!(!error.is_terminal());
    assert!(!error.is_retryable());

    let error = sessions()
        .decode_reply(
            http::StatusCode::BAD_GATEWAY,
            &http::HeaderMap::new(),
            b"upstream unavailable",
        )
        .expect_err("a gateway failure");
    let LiveApiError::Rejected(reply) = &error else {
        panic!("expected a rejection, got {error:?}");
    };
    assert_eq!(reply.error, None);
    assert_eq!(reply.response.body, "upstream unavailable");
    assert!(error.is_retryable());
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::ProviderResponse(_)
    ));
}
