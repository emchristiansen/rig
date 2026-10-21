use super::*;
use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::RecordingHttpClient;

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

#[tokio::test]
async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
    let normalize = |body: String| async {
        Model::new(
            OllamaConfig::new().completion(MODEL),
            RecordingHttpClient::new(body),
        )
        .call(CompletionRequest::new("hello"))
        .await
        .expect("completion")
    };
    let response = normalize(BODY.to_owned()).await;
    let raw = &response.raw;
    assert_eq!(
        *raw,
        serde_json::from_str::<Value>(BODY).expect("recorded body"),
        "the capture is the reply document verbatim"
    );
    let typed: ChatRecord = serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(
        Value::Object(typed.0.clone()),
        *raw,
        "the actual raw-map wire type preserves every field"
    );
    assert_eq!(
        typed.0.get("total_duration"),
        Some(&json!(5_043_500_667u64))
    );
    assert_eq!(typed.0.get("eval_duration"), Some(&json!(4_709_213_000u64)));
    assert_eq!(raw["total_duration"], 5_043_500_667u64);
    assert_eq!(raw["done_reason"], "stop");
    let renormalized = normalize(raw.to_string()).await;
    assert_eq!(response.identity(), renormalized.identity());
    assert_eq!(response.finish_reason(), renormalized.finish_reason());
    assert_eq!(response.model(), renormalized.model());
    assert_eq!(response.usage, renormalized.usage);
    assert_eq!(response.choice, renormalized.choice);
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.model(), Some("llama3.2"));
    assert_eq!(response.usage.total_tokens, Some(31));
}
