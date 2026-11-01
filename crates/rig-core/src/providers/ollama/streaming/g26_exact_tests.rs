use crate::completion::{CompletionRequest, CompletionResponse, FinishReason};
use crate::driver::Model;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::providers::ollama::{ALL_MINILM, LLAMA3_2, OllamaConfig};
use crate::streaming::{Item, Relayed, StreamEvent};
use crate::test_utils::{MockStreamingClient, RecordingHttpClient, json_body};
use crate::wire::{Mode, Wire};
use futures::StreamExt;
use serde_json::{Value, json};

fn native() -> crate::providers::ollama::chat::Chat {
    OllamaConfig::new().native_completion(LLAMA3_2)
}

#[test]
fn test_completion_request_num_predict_from_additional_params_wins() {
    let mut request = CompletionRequest::new("Hello!");
    request.max_tokens = Some(1024);
    request.additional_params = Some(json!({"num_predict":42}));
    let serialized = json_body(
        &native()
            .encode(request, Mode::Unary)
            .expect("the native request encodes")
            .request,
    );
    assert_eq!(serialized["options"], json!({"num_predict":42}));
    assert_eq!(serialized.get("max_tokens"), None);
}

#[test]
fn test_completion_request_num_predict_without_additional_params() {
    let mut request = CompletionRequest::new("Hello!");
    request.temperature = Some(0.7);
    request.max_tokens = Some(1024);
    let serialized = json_body(
        &native()
            .encode(request, Mode::Unary)
            .expect("the native request encodes")
            .request,
    );
    assert_eq!(
        serialized["options"],
        json!({"temperature":0.7,"num_predict":1024})
    );
    assert_eq!(serialized.get("max_tokens"), None);
    assert_eq!(serialized.get("temperature"), None);
}

#[tokio::test]
async fn malformed_line_is_surfaced_and_the_terminal_still_arrives() {
    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        "{not json\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":" there"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
    );
    let model = Model::new(
        native(),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("the native stream opens")
        .into_relay();
    let mut texts = Vec::new();
    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => texts.push(text),
            Ok(Relayed::Done(response)) => terminal = Some(response),
            Ok(_) => {}
            Err(_) => saw_error = true,
        }
    }
    assert_eq!(texts, ["hi", " there"]);
    assert!(saw_error, "the malformed line reaches the consumer");
    let terminal = terminal.expect("the genuine done record still arrives");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[tokio::test]
async fn content_after_the_done_record_is_not_yielded() {
    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":"stray"},"done":false}"#,
        "\n",
    );
    let model = Model::new(
        native(),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("the native stream opens")
        .into_relay();
    let mut texts = Vec::new();
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.expect("the stream item is Ok") {
            Relayed::Item(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Relayed::Done(response) => {
                assert!(terminal.is_none(), "the terminal is delivered exactly once");
                terminal = Some(response);
            }
            Relayed::Origin(_)
            | Relayed::Item(Item::Event(StreamEvent::Start { .. } | StreamEvent::End { .. })) => {}
            other => panic!("unexpected stream item: {other:?}"),
        }
    }
    assert_eq!(texts, ["hi"], "content after done is not yielded");
    let terminal = terminal.expect("the done record delivers the terminal");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let error = Model::new(native(), http_client)
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("non-success status fails");
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn embeddings_non_success_preserves_status_and_body() {
    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = Model::new(OllamaConfig::new().embedding(ALL_MINILM, None), http_client);
    let error = model
        .call(vec!["hello".to_string()])
        .await
        .expect_err("non-success status fails");
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
    const BODY: &str = r#"{
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {"role": "assistant", "content": "hello"},
            "done": true,
            "done_reason": "stop",
            "total_duration": 5043500667,
            "load_duration": 5025959,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 325953000,
            "eval_count": 5,
            "eval_duration": 4709213000
        }"#;
    let model = Model::new(native(), RecordingHttpClient::new(BODY));
    let response = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("completion");
    let raw = &response.raw;
    assert_eq!(
        *raw,
        serde_json::from_str::<Value>(BODY).expect("the original body is JSON"),
        "the capture is the reply document verbatim"
    );
    let serialized = serde_json::to_vec(&response).expect("the actual current response serializes");
    let loaded: CompletionResponse =
        serde_json::from_slice(&serialized).expect("the actual current response roundtrips");
    assert_eq!(loaded, response);
    assert_eq!(loaded.raw, *raw);
    assert_eq!(raw["total_duration"], 5_043_500_667_u64);
    assert_eq!(raw["eval_duration"], 4_709_213_000_u64);
    assert_eq!(raw["done_reason"].as_str(), Some("stop"));

    let renormalized = Model::new(native(), RecordingHttpClient::new(raw.to_string()))
        .call(CompletionRequest::new("hello"))
        .await
        .expect("the actual decoder refolds the capture");
    assert_eq!(response.identity(), renormalized.identity());
    assert_eq!(response.finish_reason(), renormalized.finish_reason());
    assert_eq!(response.model(), renormalized.model());
    assert_eq!(response.usage, renormalized.usage);
    assert_eq!(response.choice, renormalized.choice);
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.model(), Some("llama3.2"));
    assert_eq!(response.usage.total_tokens, Some(31));
}

fn original_unrepresentable_turns() -> [(crate::message::Message, &'static str); 2] {
    use crate::message::{
        AssistantMessage, CallId, CustomToolCall, Message, ToolCall, ToolFunction, ToolName,
    };
    let name = |value: &str| ToolName::new(value).expect("the original nonempty name");
    [
        (
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                ToolCall::new(
                    CallId::from_wire("call_1"),
                    ToolFunction::new(name("add"), json!({"x":1}))
                        .with_namespace(Some("math".to_owned())),
                ),
            )])),
            "namespace",
        ),
        (
            Message::Assistant(AssistantMessage::new(vec![
                AssistantContent::CustomToolCall(CustomToolCall::new(
                    CallId::from_wire("call_2"),
                    name("apply_patch"),
                    r#"{"x": 1}"#,
                )),
            ])),
            "custom",
        ),
    ]
}

fn assert_original_typed_refusal(error: &ProviderError, kind: &str, wire: &str) {
    use crate::message::UnrepresentableToolCall;
    let ProviderError::Request(reason) = error else {
        panic!("a request refusal, got {error:?}");
    };
    let refusal = reason
        .downcast_ref::<UnrepresentableToolCall>()
        .expect("the complete typed refusal survives");
    match (kind, refusal) {
        (
            "namespace",
            UnrepresentableToolCall::Namespace {
                wire: got,
                name,
                namespace,
            },
        ) => {
            assert_eq!(
                (*got, name.as_str(), namespace.as_str()),
                (wire, "add", "math")
            );
        }
        ("custom", UnrepresentableToolCall::CustomCall { wire: got, name }) => {
            assert_eq!((*got, name.as_str()), (wire, "apply_patch"));
        }
        _ => panic!("expected the {kind} refusal on {wire}, got {refusal:?}"),
    }
}

#[test]
fn ollama_chat_refuses_namespaced_and_custom_calls() {
    for (turn, kind) in original_unrepresentable_turns() {
        let request = CompletionRequest::new(turn);
        let error = ProviderError::from(
            native()
                .encode(request, Mode::Unary)
                .expect_err("refused, not dropped"),
        );
        assert_original_typed_refusal(&error, kind, "Ollama native chat");
    }
}
