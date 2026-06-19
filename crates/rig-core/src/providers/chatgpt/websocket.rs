//! ChatGPT Responses WebSocket backend.
//!
//! Plugs the ChatGPT subscription provider into the generic OpenAI Responses
//! WebSocket session through the [`ResponsesWebSocketBackend`] seam: this backend
//! supplies ChatGPT's base URL, reuses the provider's Codex request shaping, and
//! builds handshake headers from the async ChatGPT auth context.
//!
//! This is an intentionally minimal first phase. It emits only the
//! auth-context-derived handshake headers (`Authorization` and the optional
//! `ChatGPT-Account-Id`) and does not chain turns via `previous_response_id`. The
//! dashed identity headers, the `OpenAI-Beta` header, FedRAMP/JWT behavior, the
//! cache identity, and request-body stamping are deferred to later phases, so this
//! backend is not yet a live-acceptable Codex WebSocket implementation.

use super::{Client, ResponsesCompletionModel};
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::providers::openai::responses_api::CompletionRequest as ResponsesRequest;
use crate::providers::openai::responses_api::websocket::ResponsesWebSocketBackend;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::fmt::Debug;

/// The handshake header carrying the ChatGPT subscription account id.
///
/// HTTP header names are case-insensitive, so this canonical spelling is for
/// source consistency; it is lowercased on the wire like the existing HTTP path.
const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// A [`ResponsesWebSocketBackend`] backed by the ChatGPT subscription provider.
///
/// Wraps a ChatGPT [`ResponsesCompletionModel`] so the session reaches ChatGPT's
/// Codex backend through the model's configured client and shapes requests with
/// the provider's existing Codex request conversion.
pub struct ChatGptWsBackend<H = reqwest::Client> {
    model: ResponsesCompletionModel<H>,
}

impl<H> ChatGptWsBackend<H> {
    pub(crate) fn new(model: ResponsesCompletionModel<H>) -> Self {
        Self { model }
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
        self.model.create_request(request)
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

        Ok(headers)
    }

    fn chains_previous_response_id(&self) -> bool {
        // Rig drives Codex as full-replay turns, so it does not chain via
        // `previous_response_id`.
        false
    }
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

        drop(session);
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
}
