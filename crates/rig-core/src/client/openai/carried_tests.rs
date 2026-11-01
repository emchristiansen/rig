//! One-shot credential attachment and authenticator cache ownership.

#[cfg(not(target_family = "wasm"))]
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(target_family = "wasm"))]
use std::task::Poll;

#[cfg(not(target_family = "wasm"))]
use bytes::Bytes;
#[cfg(target_family = "wasm")]
use futures::FutureExt;
use rig_http::test_utils::RecordingHttpClient;
#[cfg(not(target_family = "wasm"))]
use rig_http::test_utils::{MockHttpResponse, SequencedHttpClient};

use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::error::ProviderError;
#[cfg(not(target_family = "wasm"))]
use crate::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use crate::providers::chatgpt::{self, auth};
use crate::providers::copilot;
use crate::providers::copilot::wire::{CopilotEnvelopeOverCallerIdentity, CopilotWire};
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::wire::{CallerIdentity, ConfiguredIdentity, MissingCallerIdentity};
use crate::wasm_compat::WasmBoxedFuture;
#[cfg(not(target_family = "wasm"))]
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Credential, CredentialSource, CredentialSourceError};

const CHAT_REPLY: &str = r#"{"object":"chat.completion","id":"reply_auth","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"kept"},"finish_reason":"stop"}]}"#;

fn request_identity() -> CallerIdentity {
    CallerIdentity::new("request-owner", "request-agent/2.0", Some("2.0".to_owned()))
        .expect("the synthetic request identity is valid")
}

fn refresh_identity() -> CallerIdentity {
    CallerIdentity::new("refresh-owner", "refresh-agent/1.0", Some("1.0".to_owned()))
        .expect("the synthetic refresh identity is valid")
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn authenticate_refreshes_as_a_and_sends_as_b_retaining_a_missing_account() {
    let dir = assert_fs::TempDir::new().expect("synthetic cache directory");
    let path = dir.path().join("auth.json");
    std::fs::write(
        &path,
        serde_json::json!({
            "access_token": "synthetic-expired",
            "refresh_token": "synthetic-refresh",
            "expires_at": 0,
            "account_id": "cached-account"
        })
        .to_string(),
    )
    .expect("write the synthetic expired cache");
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"access_token":"synthetic-resolved"}"#),
        MockHttpResponse::success(CHAT_REPLY),
    ]);
    let identity = request_identity();
    let authenticator = auth::Authenticator::new(
        auth::AuthSource::OAuth {
            identity: refresh_identity(),
        },
        Some(path),
        auth::DeviceCodeHandler::default(),
        false,
    );
    let client = OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-configured")
        .with_caller_identity(identity.clone())
        .with_account_id("configured-account")
        .connect(http.clone())
        .authenticate(&authenticator)
        .await
        .expect("resolve the synthetic refresh");
    assert_eq!(client.config().api_key.expose(), "synthetic-resolved");
    assert_eq!(
        client.config().identity,
        Some(ConfiguredIdentity::Exact(identity))
    );
    assert_eq!(
        client.config().account_id.as_deref(),
        Some("configured-account")
    );
    assert!(client.config().credential_source.is_none());
    assert_eq!(http.requests().len(), 1, "authentication resolves once");

    let response = client
        .chat("fixture-model")
        .call(CompletionRequest::new("test credential attachment"))
        .await
        .expect("the synthetic completion succeeds");
    assert_eq!(response.text(), "kept");
    let requests = http.requests();
    let [refresh, completion] = requests.as_slice() else {
        panic!("one refresh and one completion required: {requests:?}");
    };
    assert_eq!(refresh.uri, "https://auth.openai.com/oauth/token");
    let mut names: Vec<_> = refresh
        .headers
        .keys()
        .map(http::HeaderName::as_str)
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["content-type", "originator", "user-agent"]);
    assert_eq!(refresh.headers["content-type"], "application/json");
    assert_eq!(refresh.headers["originator"], "refresh-owner");
    assert_eq!(refresh.headers["user-agent"], "refresh-agent/1.0");
    assert_eq!(refresh.body.as_ref(), br#"{"client_id":"app_EMoamEEZ73f0CkXaXp7hrann","grant_type":"refresh_token","refresh_token":"synthetic-refresh"}"#);
    assert_eq!(
        completion.uri,
        "https://chatgpt.com/backend-api/codex/chat/completions"
    );
    assert_eq!(
        completion.headers["authorization"],
        "Bearer synthetic-resolved"
    );
    assert_eq!(completion.headers["originator"], "request-owner");
    assert_eq!(completion.headers["user-agent"], "request-agent/2.0");
    assert_eq!(completion.headers["version"], "2.0");
    assert_eq!(
        completion.headers["chatgpt-account-id"],
        "configured-account"
    );
    for name in [
        "authorization",
        "originator",
        "user-agent",
        "version",
        "chatgpt-account-id",
    ] {
        assert_eq!(completion.headers.get_all(name).iter().count(), 1, "{name}");
    }
    assert_eq!(http.remaining_responses(), 0);
}

struct CurrentCredential(Arc<AtomicUsize>);

impl CredentialSource for CurrentCredential {
    fn current(&self) -> WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(Credential::new("synthetic-current").with_account_id("source-account"))
        })
    }
}

#[tokio::test]
async fn authenticate_keeps_the_source_handle_and_defers_reading_it_until_send() {
    let reads = Arc::new(AtomicUsize::new(0));
    let config = OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-configured")
        .with_caller_identity(request_identity())
        .with_account_id("configured-account")
        .with_credential_source(CurrentCredential(Arc::clone(&reads)));
    let before = config.clone();
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let auth = auth::Authenticator::new(
        auth::AuthSource::AccessToken {
            access_token: "synthetic-resolved".to_owned(),
            account_id: None,
        },
        None,
        auth::DeviceCodeHandler::default(),
        false,
    );
    let client = config
        .connect(http.clone())
        .authenticate(&auth)
        .await
        .expect("explicit token");
    assert_eq!(client.config().credential_source, before.credential_source);
    assert_eq!(client.config().identity, before.identity);
    assert_eq!(client.config().account_id, before.account_id);
    assert_eq!(client.config().api_key.expose(), "synthetic-resolved");
    assert_eq!(reads.load(Ordering::SeqCst), 0);
    assert!(http.requests().is_empty());
    client
        .chat("fixture-model")
        .call(CompletionRequest::new("test source ownership"))
        .await
        .expect("the current source supplies the request credential");
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    let requests = http.requests();
    let [request] = requests.as_slice() else {
        panic!("one completion required: {requests:?}");
    };
    assert_eq!(request.headers["authorization"], "Bearer synthetic-current");
    assert_eq!(request.headers["chatgpt-account-id"], "source-account");
    assert_eq!(request.headers["originator"], "request-owner");
    assert_eq!(request.headers["user-agent"], "request-agent/2.0");
    assert_eq!(request.headers["version"], "2.0");
}

#[tokio::test]
async fn an_attached_token_does_not_supply_a_missing_request_identity() {
    let http = RecordingHttpClient::new(CHAT_REPLY);
    let authenticator = auth::Authenticator::new(
        auth::AuthSource::AccessToken {
            access_token: "synthetic-resolved".to_owned(),
            account_id: None,
        },
        None,
        auth::DeviceCodeHandler::default(),
        false,
    );
    let client = OpenAIConfig::with_key(&chatgpt::DIALECT, "")
        .connect(http.clone())
        .authenticate(&authenticator)
        .await
        .expect("explicit token");
    let error = client
        .chat("fixture-model")
        .call(CompletionRequest::new("identity required"))
        .await
        .expect_err("credential resolution cannot create a caller identity");
    let ProviderError::Request(source) = error else {
        panic!("a typed request refusal is required");
    };
    assert_eq!(
        source.downcast_ref::<MissingCallerIdentity>(),
        Some(&MissingCallerIdentity::Unset { dialect: "chatgpt" })
    );
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn a_copilot_envelope_still_refuses_an_exact_identity_dialect_before_send() {
    for identity in [None, Some(request_identity())] {
        let mut config = OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-token");
        config.identity = identity.map(ConfiguredIdentity::Exact);
        let wire = CopilotWire {
            wire: config.completion("fixture-model"),
            intent: copilot::CopilotIntent::default(),
        };
        let http = RecordingHttpClient::new(CHAT_REPLY);
        let error = Model::new(wire, http.clone())
            .call(CompletionRequest::new("identity required"))
            .await
            .expect_err("the wrapper cannot replace this dialect's user-agent");
        let ProviderError::Request(source) = error else {
            panic!("a typed request refusal is required");
        };
        assert_eq!(
            source.downcast_ref::<CopilotEnvelopeOverCallerIdentity>(),
            Some(&CopilotEnvelopeOverCallerIdentity { dialect: "chatgpt" })
        );
        assert!(http.requests().is_empty());
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
struct YieldingHttp(RecordingHttpClient);

#[cfg(not(target_family = "wasm"))]
impl HttpClientExt for YieldingHttp {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let reply = self.0.send(req);
        async move {
            let mut yielded = false;
            futures::future::poll_fn(move |cx| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            })
            .await;
            reply.await
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.0.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.0.send_streaming(req)
    }
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn cloned_chatgpt_authenticators_lock_before_cache_read_and_hold_across_refresh() {
    use base64::Engine;
    use base64::prelude::BASE64_URL_SAFE_NO_PAD;

    let dir = assert_fs::TempDir::new().expect("synthetic cache directory");
    let path = dir.path().join("auth.json");
    std::fs::write(
        &path,
        r#"{"refresh_token":"synthetic-refresh","expires_at":0}"#,
    )
    .expect("synthetic cache");
    let token = format!(
        "synthetic.{}.synthetic",
        BASE64_URL_SAFE_NO_PAD.encode(serde_json::json!({"exp": i64::MAX}).to_string())
    );
    let recorded = RecordingHttpClient::new(serde_json::json!({"access_token": token}).to_string());
    let http = YieldingHttp(recorded.clone());
    let first = auth::Authenticator::new(
        auth::AuthSource::OAuth {
            identity: refresh_identity(),
        },
        Some(path.clone()),
        auth::DeviceCodeHandler::default(),
        false,
    );
    let second = first.clone();
    let (a, b) = futures::join!(first.auth_context(&http), second.auth_context(&http));
    assert_eq!(a.expect("first refresh").access_token.expose(), token);
    assert_eq!(
        b.expect("clone reuses refresh").access_token.expose(),
        token
    );
    assert_eq!(
        recorded.requests().len(),
        1,
        "the clone reads the cache after the first refresh"
    );
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("persisted cache")).expect("cache JSON");
    assert_eq!(persisted["access_token"], token);
    assert_eq!(persisted["refresh_token"], "synthetic-refresh");
    assert_eq!(persisted["expires_at"], i64::MAX);
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn cloned_copilot_authenticators_lock_before_cache_read_and_hold_across_exchange() {
    let dir = assert_fs::TempDir::new().expect("synthetic cache directory");
    let access = dir.path().join("access-token");
    let key = dir.path().join("api-key.json");
    std::fs::write(&access, "synthetic-bootstrap").expect("synthetic bootstrap cache");
    let recorded = RecordingHttpClient::new(serde_json::json!({"token": "synthetic-copilot", "expires_at": i64::MAX, "endpoints": {"api": "https://synthetic.example.invalid"}}).to_string());
    let http = YieldingHttp(recorded.clone());
    let first = copilot::auth::Authenticator::new(
        copilot::auth::AuthSource::OAuth,
        Some(access),
        Some(key.clone()),
        auth::DeviceCodeHandler::default(),
        false,
    );
    let second = first.clone();
    let (a, b) = futures::join!(first.auth_context(&http), second.auth_context(&http));
    for context in [
        a.expect("first exchange"),
        b.expect("clone reuses exchange"),
    ] {
        assert_eq!(context.api_key.expose(), "synthetic-copilot");
        assert_eq!(
            context.api_base.as_deref(),
            Some("https://synthetic.example.invalid")
        );
    }
    assert_eq!(
        recorded.requests().len(),
        1,
        "the clone reads the cache after the first exchange"
    );
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(key).expect("persisted key")).expect("cache JSON");
    assert_eq!(persisted["token"], "synthetic-copilot");
    assert_eq!(persisted["expires_at"], i64::MAX);
    assert!(persisted["bootstrap_token_fingerprint"].as_str().is_some());
}

#[cfg(target_family = "wasm")]
#[test]
fn chatgpt_oauth_remains_a_wasm_refusal_with_an_exact_identity() {
    let http = RecordingHttpClient::new("");
    let auth = auth::Authenticator::new(
        auth::AuthSource::OAuth {
            identity: refresh_identity(),
        },
        None,
        auth::DeviceCodeHandler::default(),
        false,
    );
    let error = auth
        .auth_context(&http)
        .now_or_never()
        .expect("the refusal is immediate")
        .expect_err("OAuth is unsupported");
    assert_eq!(
        error.to_string(),
        "ChatGPT OAuth is not supported on wasm targets"
    );
    assert!(http.requests().is_empty());
}

#[cfg(target_family = "wasm")]
#[test]
fn copilot_oauth_remains_a_wasm_refusal() {
    let http = RecordingHttpClient::new("");
    let auth = copilot::auth::Authenticator::new(
        copilot::auth::AuthSource::OAuth,
        None,
        None,
        auth::DeviceCodeHandler::default(),
        false,
    );
    let error = auth
        .auth_context(&http)
        .now_or_never()
        .expect("the refusal is immediate")
        .expect_err("OAuth is unsupported");
    assert_eq!(
        error.to_string(),
        "GitHub Copilot OAuth is not supported on wasm targets"
    );
    assert!(http.requests().is_empty());
}
