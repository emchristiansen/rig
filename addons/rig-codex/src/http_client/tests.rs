use super::*;
use crate::responses_api::codex_identity::CodexIdentity;
use crate::responses_api::request_headers::RequestHeaders;
use crate::wire::{CallerIdentity, MissingCallerIdentity};
use http::StatusCode;
use rig_core::Model;
use rig_core::completion::CompletionRequest;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_http::http_client::Error;
use rig_http::test_utils::{CapturedHttpRequest, MockHttpResponse, SequencedHttpClient};
use serde_json::json;

const ACCESS_TOKEN: &str = "test-token";

fn prompt() -> CompletionRequest {
    CompletionRequest::new("say hi")
}

fn anonymous() -> CodexResponses {
    CodexResponses::new(Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN),
        "gpt-5.4",
    ))
    .expect("the ChatGPT dialect speaks the Codex contract")
}

fn chatgpt() -> CodexResponses {
    anonymous().with_caller_identity(
        CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
            .expect("a valid caller identity"),
    )
}

fn derived_identity() -> CodexIdentity {
    CodexIdentity::from_ids("session-derived", "thread-derived").expect("header-safe ids")
}

/// A transport that keeps every request and refuses each with a 400, so a
/// call through it ends after one request.
fn refusing() -> SequencedHttpClient {
    SequencedHttpClient::new((0..4).map(|_| {
        MockHttpResponse::ErrorResponse(StatusCode::BAD_REQUEST, Bytes::from_static(b"{}"))
    }))
}

/// The one HTTP request Rig's own driver sends for `request` on `wire`,
/// through the Codex client: the request as it reaches the transport.
async fn sent_request(wire: &CodexResponses, request: CompletionRequest) -> CapturedHttpRequest {
    let transport = refusing();
    let model = Model::new(
        wire.wire.clone(),
        CodexHttpClient::new(transport.clone(), wire.clone()),
    );
    model
        .call(request)
        .await
        .expect_err("the scripted reply refuses the request");
    let mut requests = transport.requests();
    assert_eq!(requests.len(), 1, "one call, one request: {requests:?}");
    requests.remove(0)
}

async fn sent(wire: &CodexResponses) -> CapturedHttpRequest {
    sent_request(wire, prompt()).await
}

fn header_names(request: &CapturedHttpRequest) -> Vec<&str> {
    let mut names: Vec<&str> = request
        .headers
        .keys()
        .map(http::HeaderName::as_str)
        .collect();
    names.sort_unstable();
    names
}

fn body_json(request: &CapturedHttpRequest) -> Value {
    serde_json::from_slice(&request.body).expect("the body is JSON")
}

/// A request sent straight through the client, without Rig's driver.
async fn through(wire: &CodexResponses, request: Request<Bytes>) -> Result<CapturedHttpRequest> {
    let transport = refusing();
    let client = CodexHttpClient::new(transport.clone(), wire.clone());
    client.send::<Bytes, Bytes>(request).await?;
    let mut requests = transport.requests();
    assert_eq!(requests.len(), 1);
    Ok(requests.remove(0))
}

fn json_post() -> http::request::Builder {
    Request::builder()
        .method(Method::POST)
        .uri("https://chatgpt.com/backend-api/codex/responses")
        .header(CONTENT_TYPE, "application/json")
}

/// With a Codex identity, every HTTP request names the conversation with the
/// caller's ids, exactly as given: dashed `session-id` and `thread-id`,
/// `x-client-request-id`, and the body's `prompt_cache_key` and
/// `client_metadata`. Two requests carry the same identity.
#[tokio::test]
async fn a_codex_identity_names_every_http_request_with_the_callers_ids() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    for _ in 0..2 {
        let request = sent(&wire).await;
        assert_eq!(
            header_names(&request),
            [
                // Rig's transport asks for the event stream.
                "accept",
                "authorization",
                "content-type",
                "originator",
                "session-id",
                "thread-id",
                "user-agent",
                "x-client-request-id"
            ]
        );
        assert_eq!(request.headers["accept"], "text/event-stream");
        assert_eq!(request.headers["authorization"], "Bearer test-token");
        assert_eq!(request.headers["originator"], "client_exec");
        assert_eq!(request.headers["user-agent"], "client_exec/1.2.3 (test)");
        assert_eq!(request.headers["session-id"], "session-derived");
        assert_eq!(request.headers["thread-id"], "thread-derived");
        assert_eq!(request.headers["x-client-request-id"], "thread-derived");
        let body = body_json(&request);
        assert_eq!(body["prompt_cache_key"], "thread-derived");
        assert_eq!(
            body["client_metadata"],
            json!({"session_id": "session-derived", "thread_id": "thread-derived"})
        );
    }
}

/// Without a conversation identity the request keeps the session id Rig's
/// own encoder gives it, a fresh one per request, and no thread or
/// correlation header.
#[tokio::test]
async fn without_a_codex_identity_each_request_keeps_rigs_own_session_id() {
    let first = sent(&chatgpt()).await;
    let second = sent(&chatgpt()).await;
    assert!(first.headers.get("thread-id").is_none());
    assert!(first.headers.get("x-client-request-id").is_none());
    assert_ne!(first.headers["session-id"], second.headers["session-id"]);
    assert!(body_json(&first).get("client_metadata").is_none());
}

/// What reaches the transport is a function of the wire and the request: two
/// sends of one request on a wire that names its conversation are identical,
/// header for header and byte for byte, although Rig's encoder gave each its
/// own session id on the way in.
#[tokio::test]
async fn a_request_on_an_identified_wire_is_a_function_of_its_inputs() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract")
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-window-id", "thread-1:0")
                .expect("a valid header"),
        );
    let first = sent(&wire).await;
    let second = sent(&wire).await;
    assert_eq!(first, second);
}

/// A websocket handshake and an HTTP request carrying one identity name the
/// conversation with the same three headers.
#[tokio::test]
async fn http_and_the_websocket_handshake_carry_one_identity_alike() {
    let identity = derived_identity();
    let wire = chatgpt()
        .with_codex_identity(identity.clone())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let http = sent(&wire).await;
    let handshake = identity
        .handshake_request(&wire)
        .expect("the handshake builds");
    for name in ["session-id", "thread-id", "x-client-request-id"] {
        assert_eq!(http.headers[name], handshake.headers()[name], "{name}");
    }
}

/// Over HTTP, as over the websocket, a request's own cache key and metadata
/// keys are kept; the identity fills only what is missing.
#[tokio::test]
async fn a_callers_own_cache_key_and_metadata_are_kept_over_http() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("the ChatGPT dialect speaks the Codex contract");
    let request = prompt().additional_params(json!({
        "prompt_cache_key": "caller-key",
        "client_metadata": {"session_id": "caller-session"},
    }));
    let body = body_json(&sent_request(&wire, request).await);
    assert_eq!(body["prompt_cache_key"], "caller-key");
    assert_eq!(
        body["client_metadata"],
        json!({"session_id": "caller-session", "thread_id": "thread-derived"})
    );
}

/// A wire's request headers reach its HTTP request exactly as given, beside
/// the headers Rig sends itself; the default wire sends none.
#[tokio::test]
async fn request_headers_reach_the_http_request_beside_rigs_own() {
    let headers = RequestHeaders::new()
        .with("x-codex-window-id", "thread-1:0")
        .expect("a valid header")
        .with("x-codex-turn-state", "state-1")
        .expect("a valid header");
    let identity = CodexIdentity::from_ids("thread-1", "thread-1").expect("valid ids");
    let wire = chatgpt()
        .with_codex_identity(identity)
        .expect("a Codex wire")
        .with_request_headers(headers);
    let request = sent(&wire).await;
    let sent_headers = &request.headers;
    assert_eq!(sent_headers["x-codex-window-id"], "thread-1:0");
    assert_eq!(sent_headers["x-codex-turn-state"], "state-1");
    assert_eq!(sent_headers["session-id"], "thread-1");
    assert_eq!(sent_headers["thread-id"], "thread-1");
    assert_eq!(sent_headers["x-client-request-id"], "thread-1");
    assert!(sent_headers.get("session_id").is_none());
    for name in [
        "x-codex-window-id",
        "x-codex-turn-state",
        "originator",
        "user-agent",
    ] {
        assert_eq!(
            sent_headers.get_all(name).iter().count(),
            1,
            "`{name}` is sent once"
        );
    }

    let plain = sent(&chatgpt()).await;
    assert!(plain.headers.get("x-codex-window-id").is_none());
}

/// Without the caller's exact identity nothing reaches the transport, and the
/// refusal keeps its type inside the transport error.
#[tokio::test]
async fn a_wire_without_a_caller_identity_sends_nothing() {
    let transport = refusing();
    let wire = anonymous();
    let client = CodexHttpClient::new(transport.clone(), wire.clone());
    let request = json_post().body(Bytes::from_static(b"{}")).expect("builds");
    let Err(error) = client.send::<Bytes, Bytes>(request).await else {
        panic!("no caller identity, no request");
    };
    match error {
        Error::Instance(inner) => assert_eq!(
            inner.downcast_ref::<MissingCallerIdentity>(),
            Some(&MissingCallerIdentity { dialect: "chatgpt" })
        ),
        other => panic!("expected the typed refusal, got {other:?}"),
    }
    assert!(transport.requests().is_empty());
    assert_eq!(client.wire(), &wire);
    assert!(client.inner().requests().is_empty());

    // The driver reports the same refusal as a failed call.
    let model = Model::new(
        wire.wire.clone(),
        CodexHttpClient::new(transport.clone(), wire),
    );
    model
        .call(prompt())
        .await
        .expect_err("no caller identity, no call");
    assert!(transport.requests().is_empty());
}

/// The header set that leaves is exact whatever the request carried in: the
/// caller identity replaces another's, a `version` the caller does not name
/// is removed, and the conversation's identity replaces a session id under
/// either spelling.
#[tokio::test]
async fn the_headers_that_leave_are_exactly_the_wires() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("a Codex wire");
    let request = json_post()
        .header("authorization", "Bearer test-token")
        .header("originator", "rig")
        .header("user-agent", "rig/0 (linux x86_64; rig)")
        .header("version", "0.46.0")
        .header("session_id", "per-request")
        .header("session-id", "per-request")
        .body(Bytes::from_static(b"{}"))
        .expect("builds");
    let sent = through(&wire, request).await.expect("sends");
    let mut headers: Vec<(&str, &str)> = sent
        .headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.to_str().expect("ascii header")))
        .collect();
    headers.sort_unstable();
    assert_eq!(
        headers,
        [
            ("authorization", "Bearer test-token"),
            ("content-type", "application/json"),
            ("originator", "client_exec"),
            ("session-id", "session-derived"),
            ("thread-id", "thread-derived"),
            ("user-agent", "client_exec/1.2.3 (test)"),
            ("x-client-request-id", "thread-derived"),
        ]
    );
}

/// The body keeps the encoder's order with the conversation last: the
/// identity is added ahead of `input`, and `max_output_tokens` stays after
/// it.
#[tokio::test]
async fn the_identity_is_stamped_ahead_of_the_conversation() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("a Codex wire");
    let body =
        br#"{"model":"gpt-5.4","stream":true,"input":[{"role":"user"}],"max_output_tokens":64}"#;
    let request = json_post()
        .header(CONTENT_LENGTH, body.len())
        .body(Bytes::from_static(body))
        .expect("builds");
    let sent = through(&wire, request).await.expect("sends");
    let text = std::str::from_utf8(&sent.body).expect("the body is JSON text");
    assert!(
        text.ends_with(r#""input":[{"role":"user"}],"max_output_tokens":64}"#),
        "{text}"
    );
    let input_at = text.find(r#""input":"#).expect("input");
    for key in [
        r#""client_metadata":"#,
        r#""prompt_cache_key":"#,
        r#""model":"#,
    ] {
        assert!(text.find(key).expect("field") < input_at, "{key} in {text}");
    }
    assert_eq!(
        sent.headers[CONTENT_LENGTH],
        sent.body.len().to_string().as_str(),
        "a stated length is the length of the body as it leaves"
    );
}

/// A request that is not a JSON `POST` keeps its body as it is, and still
/// leaves with the wire's headers.
#[tokio::test]
async fn a_request_without_a_json_body_keeps_its_body() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("a Codex wire");
    let get = Request::builder()
        .method(Method::GET)
        .uri("https://chatgpt.com/backend-api/codex/models")
        .body(Bytes::new())
        .expect("builds");
    let sent = through(&wire, get).await.expect("sends");
    assert!(sent.body.is_empty());
    assert_eq!(sent.headers["originator"], "client_exec");
    assert_eq!(sent.headers["session-id"], "session-derived");

    let text = Request::builder()
        .method(Method::POST)
        .uri("https://chatgpt.com/backend-api/codex/other")
        .header(CONTENT_TYPE, "text/plain")
        .body(Bytes::from_static(b"not json"))
        .expect("builds");
    let sent = through(&wire, text).await.expect("sends");
    assert_eq!(sent.body, Bytes::from_static(b"not json"));
}

/// A JSON body the identity cannot be stamped into is refused before
/// anything is sent, by a typed error.
#[tokio::test]
async fn a_body_the_identity_cannot_be_stamped_into_is_refused() {
    let wire = chatgpt()
        .with_codex_identity(derived_identity())
        .expect("a Codex wire");
    for (body, expected) in [
        (&b"[1,2]"[..], "a Codex request body must be a JSON object"),
        (
            &b"{not json"[..],
            "a Codex request body must be a JSON object",
        ),
        (
            &br#"{"client_metadata":"text"}"#[..],
            "`client_metadata` must be a JSON object",
        ),
    ] {
        let request = json_post()
            .body(Bytes::copy_from_slice(body))
            .expect("builds");
        let Err(Error::Instance(inner)) = through(&wire, request).await else {
            panic!("{expected}: the body must be refused");
        };
        let error = inner
            .downcast_ref::<CodexRequestError>()
            .expect("the typed refusal");
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[cfg(feature = "request-compression")]
mod compression {
    use super::*;
    use crate::responses_api::request_compression::RequestCompression;

    fn encoding(request: &CapturedHttpRequest) -> Option<&str> {
        request
            .headers
            .get(CONTENT_ENCODING)
            .map(|value| value.to_str().expect("ascii header"))
    }

    /// A zstd wire sends exactly the plain wire's body, zstd-compressed, and
    /// names the encoding.
    #[tokio::test]
    async fn a_zstd_wire_sends_the_plain_body_compressed_and_names_the_encoding() {
        // One conversation identity on both wires, so the bodies are the
        // same request.
        let plain_wire = chatgpt()
            .with_codex_identity(derived_identity())
            .expect("a Codex wire");
        let plain = sent(&plain_wire).await;
        assert_eq!(encoding(&plain), None);

        let compressed = sent(&plain_wire.with_request_compression(RequestCompression::Zstd)).await;
        assert_eq!(encoding(&compressed), Some("zstd"));
        assert_ne!(compressed.body, plain.body, "the body is compressed");
        assert_eq!(
            zstd::stream::decode_all(compressed.body.as_ref()).expect("a zstd frame"),
            plain.body.to_vec(),
            "decompressing gives back the plain body, byte for byte"
        );
        assert_eq!(
            compressed.body.to_vec(),
            zstd::stream::encode_all(plain.body.as_ref(), 3).expect("compresses"),
            "compressed at level 3, as the Codex client does"
        );
    }

    /// The default sends bodies as they are, and stays off the serialized
    /// wire.
    #[tokio::test]
    async fn the_default_sends_bodies_as_they_are() {
        let wire = chatgpt();
        assert!(wire.request_compression.is_none());
        let json = serde_json::to_value(&wire).expect("serializes");
        assert!(json.get("request_compression").is_none(), "got {json}");
        let request = sent(&wire).await;
        assert_eq!(encoding(&request), None);
        assert!(body_json(&request).is_object(), "the body is plain JSON");

        let zstd =
            serde_json::to_value(chatgpt().with_request_compression(RequestCompression::Zstd))
                .expect("serializes");
        assert_eq!(zstd["request_compression"], "zstd");
    }

    /// Compression applies to the body as sent, after the Codex identity is
    /// stamped into it: the decompressed body carries the identity.
    #[tokio::test]
    async fn a_codex_identity_is_stamped_before_the_body_is_compressed() {
        let identity = CodexIdentity::from_ids("thread-1", "thread-1").expect("valid ids");
        let wire = chatgpt()
            .with_codex_identity(identity)
            .expect("a Codex wire")
            .with_request_compression(RequestCompression::Zstd);
        let request = sent(&wire).await;
        assert_eq!(encoding(&request), Some("zstd"));
        let body: Value = serde_json::from_slice(
            &zstd::stream::decode_all(request.body.as_ref()).expect("a zstd frame"),
        )
        .expect("the body is JSON");
        assert_eq!(body["prompt_cache_key"], "thread-1");
        assert_eq!(body["client_metadata"]["session_id"], "thread-1");
        assert_eq!(body["client_metadata"]["thread_id"], "thread-1");
    }

    /// A stated length is the compressed body's.
    #[tokio::test]
    async fn a_stated_length_is_the_compressed_bodys() {
        let wire = chatgpt().with_request_compression(RequestCompression::Zstd);
        let body = br#"{"model":"gpt-5.4","input":[]}"#;
        let request = json_post()
            .header(CONTENT_LENGTH, body.len())
            .body(Bytes::from_static(body))
            .expect("builds");
        let sent = through(&wire, request).await.expect("sends");
        assert_eq!(
            sent.headers[CONTENT_LENGTH],
            sent.body.len().to_string().as_str()
        );
        assert_eq!(
            zstd::stream::decode_all(sent.body.as_ref()).expect("a zstd frame"),
            body.to_vec(),
            "with no identity to stamp, the body is compressed as it came"
        );
    }
}
