//! Static request construction and protocol refusal, without a transport.
use super::*;
use crate::providers::{
    chatgpt::realtime::{LiveCalls, SessionConfig},
    openai::live::PublicLiveSessions,
};
#[test]
fn subscription_requires_exact_identity_and_backend_is_independent_of_url() {
    let config =
        LiveConfiguration::subscription("token").with_base_url("https://api.openai.com/v1");
    assert!(PublicLiveSessions::new(config.clone()).is_err());
    let calls = LiveCalls::new(config).unwrap();
    let error = calls
        .call_request("offer", &SessionConfig::new("instructions"))
        .unwrap_err();
    assert!(error.to_string().contains("exact originator"));
    assert!(
        LiveCalls::new(
            LiveConfiguration::public("token")
                .with_base_url("https://chatgpt.com/backend-api/codex")
        )
        .is_err()
    );
}
#[test]
fn optional_account_and_version_are_not_synthesized() {
    let config = LiveConfiguration::subscription("token")
        .with_caller_identity(CallerIdentity::new("caller", "agent", None).unwrap());
    let calls = LiveCalls::new(config).unwrap();
    let request = calls
        .call_request("offer", &SessionConfig::new("instructions"))
        .unwrap();
    assert!(request.headers().get("chatgpt-account-id").is_none());
    assert!(request.headers().get("version").is_none());
    assert_eq!(
        request.headers()["session-id"],
        calls.identity().session_id()
    );
    assert_eq!(request.headers()["thread-id"], calls.identity().thread_id());
    assert_eq!(
        request.headers()["x-session-id"],
        calls.identity().session_id()
    );
    assert!(request.headers().get("session_id").is_none());
    assert!(request.headers().get("x-client-request-id").is_none());
}
