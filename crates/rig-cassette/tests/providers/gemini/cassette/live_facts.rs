//! Live facts behind two Gemini decisions Rig makes without code:
//!
//! | # | Cell | Fact |
//! |---|------|------|
//! | 1 | `media_resolution/per_part_low` | The Developer API honours a part's own `mediaResolution`: the same image costs fewer image tokens at `MEDIA_RESOLUTION_LOW` than at the default. Rig sends no per-part setting on any wire; the vendored gRPC `Part` has no field for one. |
//! | 2 | `stream_function_call_arguments/developer_api` | A request that sets `streamFunctionCallArguments` still gets each call's arguments whole from the Developer API, never as `partialArgs`. Vertex AI documents `partialArgs` for the flag; Rig's Vertex wire is unary, and the Vertex streaming form is not verified here. |
//!
//! Cell 1 sends raw requests, through the session's client: Rig's images
//! carry no per-part settings. Cell 2's recording sends the declaration as
//! `parametersJsonSchema`, which the port sends as `parameters`, so in replay
//! it is an exact named request-mismatch witness; its checks of the recorded
//! exchanges still run on the fixture.

use rig::completion::{CompletionRequest, Effort, ToolDefinition};
use rig::message::ToolName;
use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
use serde_json::{Value, json};

use super::super::request_witness_support::{
    NamedCase, model_stream_result, with_gemini_request_witness_cassette,
};
use super::super::support::with_gemini_cassette;

/// A one-pixel PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// The image tokens a `generateContent` reply reports.
fn image_tokens(reply: &Value) -> u64 {
    reply["usageMetadata"]["promptTokensDetails"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|detail| detail["modality"] == "IMAGE")
        .and_then(|detail| detail["tokenCount"].as_u64())
        .unwrap_or_else(|| panic!("an image token count: {reply}"))
}

#[tokio::test]
async fn per_part_media_resolution() {
    with_gemini_cassette("media_resolution/per_part_low", |models| async move {
        let client = rig_test_support::cassettes::local_reqwest();
        let http = client.inner().expect("the session's reqwest client");
        let url = format!(
            "{}/v1beta/models/{GEMINI_3_FLASH_PREVIEW}:generateContent?key={}",
            models.config.base_url.trim_end_matches('/'),
            models.config.api_key.expose()
        );
        let mut tokens = Vec::new();
        for resolution in [None, Some("MEDIA_RESOLUTION_LOW")] {
            let mut image = json!({"inlineData": {"mimeType": "image/png", "data": PNG}});
            if let Some(level) = resolution {
                image["mediaResolution"] = json!({"level": level});
            }
            let body = json!({
                "contents": [{"role": "user", "parts": [image, {"text": "Say ok."}]}],
                "generationConfig": {"maxOutputTokens": 200}
            });
            let response = http
                .post(&url)
                .header("content-type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .expect("the raw request is sent");
            assert_eq!(response.status().as_u16(), 200, "Gemini accepts the part");
            let reply: Value = response.json().await.expect("a JSON reply");
            tokens.push(image_tokens(&reply));
        }
        assert!(
            tokens[1] < tokens[0],
            "MEDIA_RESOLUTION_LOW costs fewer image tokens than the default: {tokens:?}"
        );
    })
    .await;
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent, value unchanged, as `parameters`, and replay refuses interaction
/// 0. Every recorded-exchange check of the fact still runs on the fixture:
/// the recorded request asks for streamed arguments, and the recorded stream
/// carries the call without `partialArgs` or `willContinue`. Named losses,
/// read from the refused stream: a stream without error items that ends
/// cleanly, and its lookup call's `q` arriving whole (more than ten words).
#[tokio::test]
async fn stream_function_call_arguments() -> anyhow::Result<()> {
    const SCENARIO: &str = "stream_function_call_arguments/developer_api";
    with_gemini_request_witness_cassette(
        "stream_function_call_arguments/developer_api",
        NamedCase::StreamFunctionCallArguments,
        |models, _clock| async move {
            let request = CompletionRequest::new(
                "Call lookup with q set to a 40 word sentence about harbors.",
            )
            .tool(ToolDefinition::new(
                ToolName::new("lookup").expect("a tool name"),
                "Looks a sentence up.",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            ))
            .max_tokens(4000)
            .reasoning(Effort::Low)
            .additional_params(json!({
                "toolConfig": {"functionCallingConfig": {
                    "mode": "ANY",
                    "streamFunctionCallArguments": true
                }}
            }));
            model_stream_result(models.completion(GEMINI_3_FLASH_PREVIEW).stream(request)).await
        },
    )
    .await?;

    let root = crate::cassettes::cassette_root();
    let request = rig_cassette::http::recorded_json_request(&root, "gemini", SCENARIO);
    anyhow::ensure!(
        request["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"] == true,
        "the recorded request asks for streamed arguments: {request}"
    );
    let frames = rig_cassette::http::recorded_sse_json_frames(&root, "gemini", SCENARIO);
    let text = serde_json::to_string(&frames).expect("frames serialize");
    anyhow::ensure!(
        !text.contains("partialArgs") && !text.contains("willContinue"),
        "the Developer API sends no partial arguments"
    );
    anyhow::ensure!(
        text.contains("functionCall"),
        "the recorded stream carries the call"
    );
    Ok(())
}
