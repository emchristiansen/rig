//! ChatGPT Responses WebSocket backend.
//!
//! Plugs the ChatGPT subscription provider into the generic OpenAI Responses
//! WebSocket session through the [`ResponsesWebSocketBackend`] seam: this backend
//! supplies ChatGPT's base URL, reuses the provider's Codex request shaping, and
//! builds handshake headers from the async ChatGPT auth context.
//!
//! It emits the source-derived Codex WebSocket handshake headers
//! (`Authorization`, the optional `ChatGPT-Account-Id`, the dashed `session-id`
//! and `thread-id`, `x-client-request-id`, and the `OpenAI-Beta` beta opt-in) and
//! stamps the Codex cache identity (`prompt_cache_key` and `client_metadata`) onto
//! the request body as typed top-level fields. It does not chain turns via
//! `previous_response_id` (Codex runs full-replay), and it deliberately does not
//! fake optional host/reconnect/attestation headers (`x-oai-attestation`,
//! `x-codex-turn-state`, and similar), which the live handshake tolerates absent.

use super::{Client, ResponsesCompletionModel};
use crate::OneOrMany;
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::providers::openai::responses_api::CompletionRequest as ResponsesRequest;
use crate::providers::openai::responses_api::InputItem;
use crate::providers::openai::responses_api::websocket::{
    ResponsesWebSocketBackend, ResponsesWebSocketSession,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::fmt::Debug;

/// The handshake header carrying the ChatGPT subscription account id.
///
/// HTTP header names are case-insensitive, so this canonical spelling is for
/// source consistency; it is lowercased on the wire like the existing HTTP path.
const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The dashed session identity header used by the Codex WebSocket transport.
const SESSION_ID_HEADER: &str = "session-id";

/// The dashed thread identity header used by the Codex WebSocket transport.
const THREAD_ID_HEADER: &str = "thread-id";

/// The per-request correlation header; Codex sets it to the thread id.
const X_CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// The beta opt-in header enabling Codex's Responses WebSocket protocol.
const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";

/// The Codex Responses WebSocket beta value (mirrors the Codex client).
const RESPONSES_WEBSOCKETS_BETA_VALUE: &str = "responses_websockets=2026-02-06";

/// The `client_metadata` key carrying the session identifier (Codex spelling).
const SESSION_ID_METADATA_KEY: &str = "session_id";

/// The `client_metadata` key carrying the thread identifier (Codex spelling).
const THREAD_ID_METADATA_KEY: &str = "thread_id";

/// The stable Codex cache/correlation identity for a WebSocket session.
///
/// Codex derives `prompt_cache_key` from the thread id and correlates a turn via
/// the dashed `session-id`/`thread-id` headers plus `client_metadata`. Rig has no
/// installation/window concept, so this carries only the session and thread
/// identifiers — generated once per session and stable across its turns so cache
/// routing stays sticky. The identifiers use `nanoid`, matching the existing
/// ChatGPT HTTP path's `session_id` generation, since the server treats them as
/// opaque correlation strings.
#[derive(Clone, Debug)]
struct CodexCacheIdentity {
    session_id: String,
    thread_id: String,
}

impl CodexCacheIdentity {
    fn new() -> Self {
        Self {
            session_id: nanoid::nanoid!(),
            thread_id: nanoid::nanoid!(),
        }
    }

    /// The cache-routing affinity key. Codex defaults this to the thread id.
    fn prompt_cache_key(&self) -> &str {
        &self.thread_id
    }
}

/// A [`ResponsesWebSocketBackend`] backed by the ChatGPT subscription provider.
///
/// Wraps a ChatGPT [`ResponsesCompletionModel`] so the session reaches ChatGPT's
/// Codex backend through the model's configured client and shapes requests with
/// the provider's existing Codex request conversion.
pub struct ChatGptWsBackend<H = reqwest::Client> {
    model: ResponsesCompletionModel<H>,
    identity: CodexCacheIdentity,
}

impl<H> ChatGptWsBackend<H> {
    pub(crate) fn new(model: ResponsesCompletionModel<H>) -> Self {
        Self {
            model,
            identity: CodexCacheIdentity::new(),
        }
    }
}

impl<H> ResponsesWebSocketBackend for ChatGptWsBackend<H>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    fn base_url(&self) -> &str {
        self.model.client.base_url()
    }

    fn shape_request(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<ResponsesRequest, CompletionError> {
        let mut request = self.model.create_request(request)?;
        let params = &mut request.additional_parameters;

        // Codex carries the cache key and correlation metadata as typed top-level
        // fields (the placement validated against the live server), so they ride
        // along with the flattened request body rather than being stamped in by
        // hand. A caller-supplied value wins.
        if params.prompt_cache_key.is_none() {
            params.prompt_cache_key = Some(self.identity.prompt_cache_key().to_string());
        }
        params
            .client_metadata
            .entry(SESSION_ID_METADATA_KEY.to_string())
            .or_insert_with(|| self.identity.session_id.clone());
        params
            .client_metadata
            .entry(THREAD_ID_METADATA_KEY.to_string())
            .or_insert_with(|| self.identity.thread_id.clone());

        Ok(request)
    }

    async fn handshake_headers(&self) -> Result<http::HeaderMap, CompletionError> {
        // ChatGPT credentials are refreshed asynchronously, so the handshake
        // headers are derived from the live auth context rather than the client's
        // static provider headers.
        let context = self
            .model
            .client
            .ext()
            .auth
            .auth_context()
            .await
            .map_err(|err| CompletionError::ProviderError(err.to_string()))?;

        let mut headers = http::HeaderMap::new();

        let authorization = format!("Bearer {}", context.access_token)
            .parse()
            .map_err(|err| {
                CompletionError::ProviderError(format!(
                    "Invalid ChatGPT authorization header value: {err}"
                ))
            })?;
        headers.insert(http::header::AUTHORIZATION, authorization);

        if let Some(account_id) = &context.account_id {
            let name = http::HeaderName::from_bytes(CHATGPT_ACCOUNT_ID_HEADER.as_bytes()).map_err(
                |err| {
                    CompletionError::ProviderError(format!(
                        "Invalid ChatGPT account id header name: {err}"
                    ))
                },
            )?;
            let value = account_id.parse().map_err(|err| {
                CompletionError::ProviderError(format!(
                    "Invalid ChatGPT account id header value: {err}"
                ))
            })?;
            headers.insert(name, value);
        }

        // The source-derived Codex identity headers: the dashed `session-id` and
        // `thread-id`, the `x-client-request-id` correlation header (set to the
        // thread id, matching Codex), and the `OpenAI-Beta` protocol opt-in.
        insert_header(&mut headers, SESSION_ID_HEADER, &self.identity.session_id)?;
        insert_header(&mut headers, THREAD_ID_HEADER, &self.identity.thread_id)?;
        insert_header(
            &mut headers,
            X_CLIENT_REQUEST_ID_HEADER,
            &self.identity.thread_id,
        )?;
        insert_header(
            &mut headers,
            OPENAI_BETA_HEADER,
            RESPONSES_WEBSOCKETS_BETA_VALUE,
        )?;

        Ok(headers)
    }

    fn chains_previous_response_id(&self) -> bool {
        // Rig drives Codex as full-replay turns, so it does not chain via
        // `previous_response_id`.
        false
    }
}

impl<H> ResponsesWebSocketSession<ChatGptWsBackend<H>>
where
    Client<H>: HttpClientExt + Clone + Debug + 'static,
    H: Clone + Default + Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Continues the current live tip with a forward-only incremental Codex turn.
    ///
    /// Codex runs full-replay, so `send` never chains. This instead sends exactly
    /// `delta` as the new input while injecting the live `previous_response_id`
    /// captured from the last completed `send`, reusing that turn's non-input
    /// envelope/config (model, instructions, tools, reasoning/include, and the
    /// Codex cache identity). It errors rather than falling back to full replay
    /// when no completed `send` has established a tip and envelope, and a terminal
    /// failure clears the tip so the next call fails until a fresh `send` re-roots
    /// the session. Changing any non-input configuration likewise requires a fresh
    /// `send`; an incremental turn cannot reconfigure the chain.
    pub async fn send_incremental(
        &mut self,
        delta: OneOrMany<InputItem>,
    ) -> Result<(), CompletionError> {
        self.send_incremental_frame(delta).await
    }
}

/// Inserts a static-named handshake header, surfacing an invalid name or value as
/// a provider error rather than silently dropping it.
fn insert_header(
    headers: &mut http::HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), CompletionError> {
    let name = http::HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
        CompletionError::ProviderError(format!("Invalid Codex websocket header name: {err}"))
    })?;
    let value = value.parse().map_err(|err| {
        CompletionError::ProviderError(format!("Invalid Codex websocket header value: {err}"))
    })?;
    headers.insert(name, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::chatgpt::{ChatGPTAuth, Client as ChatGptClient, GPT_5_3_CODEX};

    fn access_token_model(base_url: &str) -> ResponsesCompletionModel {
        let client = ChatGptClient::builder()
            .api_key(ChatGPTAuth::AccessToken {
                access_token: "test-token".to_string(),
                account_id: Some("acct-123".to_string()),
            })
            .base_url(base_url)
            .build()
            .expect("client should build");
        ResponsesCompletionModel::new(client, GPT_5_3_CODEX)
    }

    fn user_turn(text: &str) -> crate::completion::CompletionRequest {
        crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one(crate::completion::Message::user(text)),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    fn completed_response_event(response_id: &str) -> String {
        serde_json::json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "usage": null,
                "output": [],
                "tools": []
            }
        })
        .to_string()
    }

    fn failed_response_event(response_id: &str) -> String {
        serde_json::json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 0,
                "status": "failed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "usage": null,
                "output": [],
                "tools": []
            }
        })
        .to_string()
    }

    fn delta_turn(text: &str) -> crate::OneOrMany<InputItem> {
        let items = Vec::<InputItem>::try_from(crate::completion::Message::user(text))
            .expect("user message should convert into input items");
        crate::OneOrMany::many(items).expect("delta should contain at least one item")
    }

    #[test]
    fn chatgpt_ws_backend_does_not_chain() {
        let backend =
            ChatGptWsBackend::new(access_token_model("https://chatgpt.com/backend-api/codex"));
        assert!(!backend.chains_previous_response_id());
        assert!(backend.base_url().contains("chatgpt.com"));
    }

    #[tokio::test]
    async fn handshake_sends_authorization_and_account_id() {
        use std::sync::{Arc, Mutex};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_hdr_async;
        use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let captured: Arc<Mutex<Option<http::HeaderMap>>> = Arc::new(Mutex::new(None));
        let captured_server = captured.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            // `accept_hdr_async` exposes the upgrade request to the callback so the
            // handshake headers can be captured; `accept_async` would not.
            let callback = move |request: &Request,
                                 response: Response|
                  -> Result<Response, ErrorResponse> {
                *captured_server.lock().expect("capture lock") = Some(request.headers().clone());
                Ok(response)
            };
            let _socket = accept_hdr_async(stream, callback)
                .await
                .expect("server should upgrade websocket");
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        server.await.expect("server task should finish");

        let headers = captured
            .lock()
            .expect("capture lock")
            .take()
            .expect("handshake headers should be captured");
        assert_eq!(
            headers
                .get(http::header::AUTHORIZATION)
                .expect("authorization header")
                .to_str()
                .expect("authorization header should be ascii"),
            "Bearer test-token"
        );
        assert_eq!(
            headers
                .get(CHATGPT_ACCOUNT_ID_HEADER)
                .expect("account id header")
                .to_str()
                .expect("account id header should be ascii"),
            "acct-123"
        );

        // The dashed identity headers and beta opt-in are present, and the
        // correlation header mirrors the thread id, matching the Codex client.
        let session_id = headers
            .get(SESSION_ID_HEADER)
            .expect("session-id header")
            .to_str()
            .expect("session-id header should be ascii");
        assert!(!session_id.is_empty(), "session-id should be non-empty");
        let thread_id = headers
            .get(THREAD_ID_HEADER)
            .expect("thread-id header")
            .to_str()
            .expect("thread-id header should be ascii");
        assert!(!thread_id.is_empty(), "thread-id should be non-empty");
        assert_eq!(
            headers
                .get(X_CLIENT_REQUEST_ID_HEADER)
                .expect("x-client-request-id header")
                .to_str()
                .expect("x-client-request-id header should be ascii"),
            thread_id,
            "x-client-request-id should mirror the thread id"
        );
        assert_eq!(
            headers
                .get(OPENAI_BETA_HEADER)
                .expect("OpenAI-Beta header")
                .to_str()
                .expect("OpenAI-Beta header should be ascii"),
            RESPONSES_WEBSOCKETS_BETA_VALUE
        );

        drop(session);
    }

    #[test]
    fn shape_request_stamps_cache_identity_as_top_level_fields() {
        let backend =
            ChatGptWsBackend::new(access_token_model("https://chatgpt.com/backend-api/codex"));
        let request = backend
            .shape_request(user_turn("hello"))
            .expect("request should shape");

        // The cache key defaults to the thread id, and the session/thread land in
        // client_metadata under Codex's key spellings.
        assert_eq!(
            request.additional_parameters.prompt_cache_key.as_deref(),
            Some(backend.identity.thread_id.as_str())
        );
        assert_eq!(
            request
                .additional_parameters
                .client_metadata
                .get(SESSION_ID_METADATA_KEY)
                .map(String::as_str),
            Some(backend.identity.session_id.as_str())
        );
        assert_eq!(
            request
                .additional_parameters
                .client_metadata
                .get(THREAD_ID_METADATA_KEY)
                .map(String::as_str),
            Some(backend.identity.thread_id.as_str())
        );

        // The fields serialize at the top level of the request body (the placement
        // validated against the live server), not nested under another object.
        let body = serde_json::to_value(&request).expect("request should serialize");
        assert_eq!(
            body.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some(backend.identity.thread_id.as_str())
        );
        assert!(
            body.get("client_metadata")
                .and_then(|v| v.get(THREAD_ID_METADATA_KEY))
                .is_some(),
            "client_metadata should be a top-level object, got {body}"
        );
    }

    #[tokio::test]
    async fn chatgpt_ws_emits_response_create_and_omits_previous_response_id() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // Two completed turns: Codex runs full-replay, so neither outbound
            // payload may carry `previous_response_id`, even after turn 1 completes.
            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                let payload = request.into_text().expect("request should be text");
                assert!(
                    payload.contains("\"type\":\"response.create\""),
                    "expected response.create envelope, got {payload}"
                );
                assert!(
                    !payload.contains("previous_response_id"),
                    "Codex replay mode must not chain previous_response_id, got {payload}"
                );

                socket
                    .send(Message::text(completed_response_event(response_id)))
                    .await
                    .expect("completed event should send");
            }
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        for turn in ["first", "second"] {
            session
                .send(user_turn(turn))
                .await
                .expect("turn should send");
            loop {
                let event = session.next_event().await.expect("event should arrive");
                if event.is_terminal() {
                    break;
                }
            }
        }

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn chatgpt_ws_warmup_emits_generate_false() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request.into_text().expect("request should be text");
            assert!(
                payload.contains("\"type\":\"response.create\""),
                "expected response.create envelope, got {payload}"
            );
            assert!(
                payload.contains("\"generate\":false"),
                "expected warmup to serialize generate:false, got {payload}"
            );

            socket
                .send(Message::text(completed_response_event("resp_warmup")))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        let response_id = session
            .warmup(user_turn("prewarm"))
            .await
            .expect("warmup should complete");
        assert_eq!(response_id, "resp_warmup");

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn chatgpt_ws_send_incremental_injects_tip_and_reuses_envelope() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let mut frames = Vec::new();
            for response_id in ["resp_1", "resp_2"] {
                let request = socket
                    .next()
                    .await
                    .expect("request should exist")
                    .expect("request should be valid");
                frames.push(
                    request
                        .into_text()
                        .expect("request should be text")
                        .to_string(),
                );
                socket
                    .send(Message::text(completed_response_event(response_id)))
                    .await
                    .expect("completed event should send");
            }
            frames
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        // The root full-replay turn establishes the live tip and captures envelope.
        session
            .send(user_turn("ROOT_CONTEXT_MARKER"))
            .await
            .expect("root turn should send");
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        // A forward-only incremental continuation of the established live tip.
        session
            .send_incremental(delta_turn("DELTA_QUESTION_MARKER"))
            .await
            .expect("incremental turn should send");
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }
        // The live tip advances to the incremental response.
        assert_eq!(session.previous_response_id(), Some("resp_2"));

        let frames = server.await.expect("server task should finish");
        let root: serde_json::Value =
            serde_json::from_str(&frames[0]).expect("root frame should parse");
        let incremental: serde_json::Value =
            serde_json::from_str(&frames[1]).expect("incremental frame should parse");

        // The root replay frame does not chain; the incremental frame injects the
        // live tip even though the Codex backend reports chains == false.
        assert!(
            root.get("previous_response_id").is_none(),
            "root replay frame must not chain, got {root}"
        );
        assert_eq!(
            incremental.get("type").and_then(|v| v.as_str()),
            Some("response.create")
        );
        assert_eq!(
            incremental
                .get("previous_response_id")
                .and_then(|v| v.as_str()),
            Some("resp_1"),
            "incremental frame must inject the current live tip"
        );

        // The incremental input is exactly the delta, not a replay of the root turn.
        assert!(
            frames[1].contains("DELTA_QUESTION_MARKER"),
            "incremental input should carry the delta, got {}",
            frames[1]
        );
        assert!(
            !frames[1].contains("ROOT_CONTEXT_MARKER"),
            "incremental input must not replay the root turn, got {}",
            frames[1]
        );

        // The incremental frame reuses the captured non-input envelope/config.
        assert_eq!(
            incremental.get("model"),
            root.get("model"),
            "incremental should reuse the captured model"
        );
        assert_eq!(
            incremental.get("prompt_cache_key"),
            root.get("prompt_cache_key"),
            "incremental should reuse the captured cache key"
        );
        assert_eq!(
            incremental.get("client_metadata"),
            root.get("client_metadata"),
            "incremental should reuse the captured client metadata"
        );
    }

    #[tokio::test]
    async fn chatgpt_ws_send_incremental_without_tip_errors_and_emits_no_frame() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // The first and only frame the server sees must be the root send(); a
            // failed incremental must not have emitted anything ahead of it.
            let request = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            let payload = request
                .into_text()
                .expect("request should be text")
                .to_string();
            socket
                .send(Message::text(completed_response_event("resp_1")))
                .await
                .expect("completed event should send");
            payload
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        // No completed send() yet: the incremental must fail without a frame.
        let error = session
            .send_incremental(delta_turn("PREMATURE_DELTA"))
            .await
            .expect_err("incremental without a tip should fail");
        assert!(
            matches!(error, CompletionError::ProviderError(_)),
            "expected ProviderError, got {error:?}"
        );

        // A subsequent real send() must be the first frame the server receives.
        session
            .send(user_turn("ROOT_AFTER_FAILURE"))
            .await
            .expect("root turn should send");
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }

        let first_frame = server.await.expect("server task should finish");
        assert!(
            first_frame.contains("ROOT_AFTER_FAILURE"),
            "the first frame should be the root send, got {first_frame}"
        );
        assert!(
            !first_frame.contains("PREMATURE_DELTA"),
            "a failed incremental must not emit a frame, got {first_frame}"
        );
    }

    #[tokio::test]
    async fn chatgpt_ws_send_incremental_after_terminal_failure_errors() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            // Turn 1: the root send() completes and establishes the tip.
            let _ = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            socket
                .send(Message::text(completed_response_event("resp_1")))
                .await
                .expect("completed event should send");

            // Turn 2: the incremental is answered with a terminal failure.
            let _ = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            socket
                .send(Message::text(failed_response_event("resp_2")))
                .await
                .expect("failed event should send");
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        session
            .send(user_turn("ROOT"))
            .await
            .expect("root turn should send");
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }
        assert_eq!(session.previous_response_id(), Some("resp_1"));

        session
            .send_incremental(delta_turn("FIRST_DELTA"))
            .await
            .expect("incremental turn should send");
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }
        // The terminal failure cleared the tip.
        assert_eq!(session.previous_response_id(), None);

        // The next incremental fails until a fresh send() re-roots the session.
        let error = session
            .send_incremental(delta_turn("SECOND_DELTA"))
            .await
            .expect_err("incremental after a terminal failure should fail");
        assert!(
            matches!(error, CompletionError::ProviderError(_)),
            "expected ProviderError, got {error:?}"
        );

        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn chatgpt_ws_send_incremental_rejected_while_in_flight() {
        use futures::{SinkExt, StreamExt};
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::Message;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("server should upgrade websocket");

            let _ = socket
                .next()
                .await
                .expect("request should exist")
                .expect("request should be valid");
            socket
                .send(Message::text(completed_response_event("resp_1")))
                .await
                .expect("completed event should send");
        });

        let base_url = format!("http://{address}/backend-api/codex");
        let mut session = access_token_model(&base_url)
            .websocket()
            .connect()
            .await
            .expect("session should connect");

        session
            .send(user_turn("ROOT"))
            .await
            .expect("root turn should send");

        // A response is now in flight; an incremental must be rejected.
        let error = session
            .send_incremental(delta_turn("DELTA"))
            .await
            .expect_err("incremental while a turn is in flight should fail");
        assert!(
            matches!(error, CompletionError::ProviderError(_)),
            "expected ProviderError, got {error:?}"
        );

        // Drain the in-flight turn so the session and server settle cleanly.
        loop {
            let event = session.next_event().await.expect("event should arrive");
            if event.is_terminal() {
                break;
            }
        }
        server.await.expect("server task should finish");
    }
}
