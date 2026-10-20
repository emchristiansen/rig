//! Gemini streaming regression for multimodal tool results in chat history.

use rig::message::{ImageMediaType, Message, ToolResultContent};
use rig::providers::gemini;
use rig::tool::{Tool, ToolOutput};
use serde_json::json;

use super::super::request_witness_support::{
    NamedCase, stream_result, with_gemini_request_witness_cassette,
};

const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";
const MULTIMODAL_FUNCTION_RESPONSE_MODEL: &str = gemini::completion::GEMINI_3_FLASH_PREVIEW;

fn streaming_tool_params() -> serde_json::Value {
    json!({ "generationConfig": {} })
}

#[derive(Debug)]
struct HybridImageTool;

#[derive(Debug, thiserror::Error)]
#[error("hybrid image tool error")]
struct HybridImageToolError;

impl Tool for HybridImageTool {
    const NAME: &'static str = "render_reference_image";
    type Error = HybridImageToolError;
    type Args = serde_json::Value;
    type Output = ToolOutput;

    fn description(&self) -> String {
        "Return a reference image the assistant must inspect before answering.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let mut content = vec![ToolResultContent::json(json!({
            "instruction": "Use the image part to answer the user's question."
        }))];
        content.push(ToolResultContent::image_base64(
            RED_PIXEL_PNG_BASE64,
            Some(ImageMediaType::PNG),
            None,
        ));
        Ok(ToolOutput::content(content).expect("fixture content is non-empty"))
    }
}

/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the only
/// request difference is that declaration 0's recorded `parametersJsonSchema`
/// is sent as `parameters`, as named in `NamedCase::HybridToolResultImage`,
/// and replay refuses interaction 0, before the tool runs. Named losses: a
/// non-empty final response, and the final history keeping the assistant's
/// render_reference_image call and its hybrid tool result as `[json, image]`
/// with the instruction JSON and the PNG base64 payload intact.
#[tokio::test]
async fn streaming_history_preserves_hybrid_tool_result_image_parts() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "streaming_multimodal_tool_results/streaming_history_preserves_hybrid_tool_result_image_parts",
        NamedCase::HybridToolResultImage,
        |client, _clock| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(MULTIMODAL_FUNCTION_RESPONSE_MODEL))
                    .preamble(
                        "You are a precise assistant. Call `render_reference_image` exactly once \
                         before answering. After the tool result arrives, do not call any more \
                         tools. Answer in one short sentence.",
                    )
                    .tool(HybridImageTool)
                    .additional_params(streaming_tool_params())
                    .build();

            let empty_history: &[Message] = &[];
            let mut stream = agent
                .prompt(
                    "Use the tool once, then answer with the dominant color in the returned image.",
                )
                .history(empty_history)
                .max_turns(4)
                .stream();
            let result = stream_result(&mut stream).await;
            drop(stream);
            drop(agent);
            result
        },
    )
    .await
}
