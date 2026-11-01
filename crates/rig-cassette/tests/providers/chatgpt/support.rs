use bytes::Bytes;
use rig::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig::providers::chatgpt;
use rig::providers::openai::OpenAIConfig;
use rig::providers::openai::wire::CallerIdentity;
use rig::wasm_compat::WasmCompatSend;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient, test_caller_identity};
use rig_test_support::cassette_models::OpenAiModels;
use std::collections::BTreeSet;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

/// The caller identity a ChatGPT cassette sends. Recording reads it from
/// `CHATGPT_ORIGINATOR` and `CHATGPT_USER_AGENT`, so what reaches the live
/// service is the recorder's exact identity and never one Rig makes up;
/// replay sends a placeholder that never leaves the machine.
pub(super) fn cassette_identity(cassette: &ProviderCassette) -> CallerIdentity {
    CallerIdentity::new(
        cassette.api_key("CHATGPT_ORIGINATOR"),
        cassette.api_key("CHATGPT_USER_AGENT"),
        None,
    )
    .expect("the cassette identity should be header-safe")
}

async fn chatgpt_cassette_with_default_instructions(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "chatgpt",
        spec,
        "https://chatgpt.com/backend-api/codex",
    )
    .await;
    let client =
        OpenAIConfig::with_key(&chatgpt::DIALECT, cassette.api_key("CHATGPT_ACCESS_TOKEN"))
            .with_caller_identity(cassette_identity(&cassette))
            .with_account_id(cassette.api_key("CHATGPT_ACCOUNT_ID"))
            .with_base_url(cassette.base_url())
            .with_instructions(default_instructions);

    (cassette, client)
}

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    chatgpt_cassette_with_default_instructions(spec, "").await
}

pub(super) async fn with_chatgpt_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = chatgpt_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

/// Adds method capture to the existing ordered request/response transport.
#[derive(Clone)]
struct CurrentCodexCapture {
    http: SequencedHttpClient,
    methods: Arc<Mutex<Vec<http::Method>>>,
}

impl HttpClientExt for CurrentCodexCapture {
    fn send<T, U>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let mut methods = self
            .methods
            .lock()
            .expect("the method capture is available");
        methods.push(request.method().clone());
        let reply = self.http.send(request);
        drop(methods);
        reply
    }

    fn send_multipart<U>(
        &self,
        request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let mut methods = self
            .methods
            .lock()
            .expect("the method capture is available");
        methods.push(request.method().clone());
        let reply = self.http.send_multipart(request);
        drop(methods);
        reply
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let mut methods = self
            .methods
            .lock()
            .expect("the method capture is available");
        methods.push(request.method().clone());
        let reply = self.http.send_streaming(request);
        drop(methods);
        reply
    }
}

/// Check current Codex requests while decoding every original reply in order.
/// This proves no classic recorded-request replay or live chunk segmentation.
pub(super) async fn with_chatgpt_current_outbound<F, Fut>(
    scenario: &str,
    recorded: Vec<(String, String)>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let paths = crate::cassettes::recorded_request_paths("chatgpt", scenario);
    let queries = crate::cassettes::recorded_request_query_pairs("chatgpt", scenario);
    let headers = crate::cassettes::recorded_request_header_pairs("chatgpt", scenario);
    assert!(
        !recorded.is_empty(),
        "{scenario}: recorded replies are required"
    );
    assert_eq!(paths.len(), recorded.len(), "{scenario}: path count");
    assert_eq!(queries.len(), recorded.len(), "{scenario}: query count");
    assert_eq!(headers.len(), recorded.len(), "{scenario}: header count");

    let http = CurrentCodexCapture {
        http: SequencedHttpClient::new(
            recorded
                .iter()
                .map(|(_, reply)| MockHttpResponse::success(reply.clone().into_bytes())),
        ),
        methods: Arc::new(Mutex::new(Vec::new())),
    };
    let base_url = "https://chatgpt.com/backend-api/codex";
    let caller = test_caller_identity();
    let config = OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-test-key")
        .with_base_url(base_url)
        .with_caller_identity(caller.clone())
        .with_account_id("synthetic-account")
        .with_instructions("");
    test_body(OpenAiModels::new(config, http.clone())).await;

    let sent = http.http.requests();
    let methods = http
        .methods
        .lock()
        .expect("the method capture is available");
    assert_eq!(sent.len(), recorded.len(), "{scenario}: request count");
    assert_eq!(methods.len(), sent.len(), "{scenario}: method count");
    assert_eq!(
        http.http.remaining_responses(),
        0,
        "{scenario}: unused replies"
    );
    let mut sessions = BTreeSet::new();

    for (index, ((original, _), request)) in recorded.iter().zip(&sent).enumerate() {
        assert_eq!(
            methods[index],
            http::Method::POST,
            "{scenario}/{index}: method"
        );
        assert_eq!(
            paths[index], "/backend-api/codex/responses",
            "{scenario}/{index}: recorded path"
        );
        assert!(
            queries[index].is_empty(),
            "{scenario}/{index}: recorded query must be empty"
        );
        assert_eq!(
            request.uri,
            format!("{base_url}/responses"),
            "{scenario}/{index}: complete URI"
        );

        let mut expected_headers = http::HeaderMap::new();
        for (name, value) in &headers[index] {
            expected_headers.append(
                http::header::HeaderName::from_bytes(name.as_bytes())
                    .expect("the recorded header name is valid"),
                http::HeaderValue::from_str(value).expect("the recorded header value is valid"),
            );
        }
        let mut recorded_names: Vec<_> = expected_headers
            .keys()
            .map(http::header::HeaderName::as_str)
            .collect();
        recorded_names.sort_unstable();
        assert_eq!(
            recorded_names,
            ["accept", "content-type"],
            "{scenario}/{index}: recorded header keys"
        );
        expected_headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer synthetic-test-key"),
        );
        expected_headers.insert(
            "chatgpt-account-id",
            http::HeaderValue::from_static("synthetic-account"),
        );
        expected_headers.insert(
            "originator",
            http::HeaderValue::from_str(caller.originator()).expect("the test originator is valid"),
        );
        expected_headers.insert(
            "user-agent",
            http::HeaderValue::from_str(caller.user_agent()).expect("the test user agent is valid"),
        );
        assert_eq!(
            caller.version(),
            None,
            "the test caller has no version header"
        );
        let session = request
            .headers
            .get("session_id")
            .expect("the current request has a session header");
        let session_text = session
            .to_str()
            .expect("the current session header is text");
        assert!(
            session_text.len() == 21
                && session_text
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'),
            "{scenario}/{index}: the current session has the generated ID shape"
        );
        assert!(
            sessions.insert(session_text.to_owned()),
            "{scenario}/{index}: the current session is fresh"
        );
        expected_headers.insert("session_id", session.clone());
        let mut names: Vec<_> = request
            .headers
            .keys()
            .map(http::header::HeaderName::as_str)
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "accept",
                "authorization",
                "chatgpt-account-id",
                "content-type",
                "originator",
                "session_id",
                "user-agent"
            ],
            "{scenario}/{index}: complete current header keys"
        );
        assert_eq!(
            request.headers, expected_headers,
            "{scenario}/{index}: complete current headers"
        );

        let mut expected: serde_json::Value =
            serde_json::from_str(original).expect("the original request is JSON");
        let instructions = expected
            .as_object_mut()
            .expect("the original request is an object")
            .shift_remove("instructions");
        let Some(serde_json::Value::String(text)) = instructions else {
            panic!("{scenario}/{index}: the original instructions must be a present string");
        };
        let input = expected["input"]
            .as_array_mut()
            .expect("the original input is an array");
        if !text.is_empty() {
            input.insert(0, serde_json::json!({
                "type":"message", "role":"developer",
                "content":[{"type":"input_text", "text":text}],
                "internal_chat_message_metadata_passthrough":{"content_item_kinds":["model.base_instructions"]},
            }));
        }
        let actual: serde_json::Value =
            serde_json::from_slice(&request.body).expect("the captured current request is JSON");
        assert_eq!(
            actual, expected,
            "{scenario}/{index}: only the adopted instruction carrier may differ"
        );
    }
}

/// The `include` every Codex request in these cassettes was recorded with.
///
/// Responses requests include `reasoning.encrypted_content` for non-null
/// reasoning or `store:false`, unless the selected model is known not to reason.
/// These tests state the recorded include explicitly to pin their request bytes.
pub(super) fn recorded_include() -> serde_json::Value {
    serde_json::json!({ "include": ["reasoning.encrypted_content"] })
}
