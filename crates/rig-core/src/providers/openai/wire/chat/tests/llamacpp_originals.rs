//! Original llama.cpp history subjects at independent current test stages.
//!
//! Direct LLAMACPP Chat encoding retains the original request predicates.
//! The actual Model reads each original recorded response to the same caller
//! request. Its preparation still merges user turns and relabels no-tools
//! calls/results as text, so recorded-response coupling is not retained.
//! System history keeps the original role-order predicate while the encoder
//! uses the upstream bare assistant string in place of the original recorded
//! assistant text-part array.

use serde_json::json;

use crate::completion::CompletionRequest;
use crate::driver::Model;
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Message, ToolName, ToolResult,
    ToolResultContent, UserContent,
};
use crate::providers::openai::wire::{LLAMACPP, OpenAIConfig};
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::{Mode, Wire};

use super::{recorded, recorded_json};

const CASSETTE_MODEL: &str = "Qwen3-1.7B-Q4_K_M";
const NO_THINK: &str = "/no_think ";

fn lookup_call_turn(id: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
        id,
        ToolName::new("lookup").expect("tool name"),
        json!({}),
    )]))
}

/// Join assistant text blocks with newlines, returning none for empty text.
fn assistant_text_response(choice: &[AssistantContent]) -> Option<String> {
    let response = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if response.is_empty() {
        None
    } else {
        Some(response)
    }
}

#[tokio::test]
async fn a_system_message_plus_history_keeps_its_order() {
    const FIXTURE: &str = "../llamacpp/content_matrix/system_plus_history.yaml";
    let wire = OpenAIConfig::with_key(&LLAMACPP, "").chat(CASSETTE_MODEL);
    let request = CompletionRequest::new(format!("{NO_THINK}And what was the first one?"))
        .preamble("You are a ledger. Answer with the requested codeword only.".to_string())
        .messages(vec![
            Message::User {
                content: vec![UserContent::text("Codeword one is heliotrope.")],
            },
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::text("Noted.")])),
            Message::User {
                content: vec![UserContent::text("Codeword two is quicksilver.")],
            },
        ])
        .max_tokens(256);
    let encoded = wire
        .encode(request.clone(), Mode::Unary)
        .expect("the original system-history request encodes");
    let actual_request = json_body(&encoded.request);

    for request in [recorded_json("when", FIXTURE), actual_request] {
        let roles = request["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|message| message["role"].as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec![
                "system".to_string(),
                "user".to_string(),
                "assistant".to_string(),
                "user".to_string(),
                "user".to_string(),
            ],
            "the preamble leads and the history keeps its order: {roles:?}"
        );
    }

    let response = Model::new(wire, RecordingHttpClient::new(recorded("then", FIXTURE)))
        .call(request)
        .await
        .expect("a system message plus history should be accepted");
    let text = assistant_text_response(&response.choice).unwrap_or_default();
    assert!(
        text.to_ascii_lowercase().contains("heliotrope"),
        "the model must be able to reach the first turn: {text:?}"
    );
}

#[tokio::test]
async fn a_very_long_tool_output_survives_the_round_trip() {
    const FIXTURE: &str = "../llamacpp/content_matrix/long_tool_output.yaml";
    const NEEDLE: &str = "ZANZIBAR-9317";
    let long_output = format!(
        "{}END OF DUMP. The code is {NEEDLE}.",
        "lorem ipsum dolor sit amet. ".repeat(290)
    );
    let wire = OpenAIConfig::with_key(&LLAMACPP, "").chat(CASSETTE_MODEL);
    let request = CompletionRequest::new(Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("call_long"),
            name: ToolName::new("dump").expect("tool name"),
            answers: crate::message::AnsweredToolCall::Function,
            is_error: false,
            content: vec![ToolResultContent::text(long_output)],
        })],
    })
    .preamble("The tool result ends with a code. Reply with only that code.".to_string())
    .messages(vec![
        Message::User {
            content: vec![UserContent::text("What code does the dump end with?")],
        },
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
            "call_long",
            ToolName::new("dump").expect("tool name"),
            json!({}),
        )])),
    ])
    .max_tokens(256);
    let encoded = wire
        .encode(request.clone(), Mode::Unary)
        .expect("the original long-output request encodes");
    let actual_request = json_body(&encoded.request);

    for request in [recorded_json("when", FIXTURE), actual_request] {
        let serialized = request["messages"].to_string();
        assert!(
            serialized.len() > 8_000,
            "the payload must really be large for this cell to test anything, was {}",
            serialized.len()
        );
        assert!(
            serialized.contains(NEEDLE),
            "the tail of the payload must reach the wire uncut"
        );
    }

    let response = Model::new(wire, RecordingHttpClient::new(recorded("then", FIXTURE)))
        .call(request)
        .await
        .expect("a long tool result should be accepted");
    let text = assistant_text_response(&response.choice).unwrap_or_default();
    assert!(
        text.contains(NEEDLE),
        "the needle at the end of an 8 KiB payload must reach the model: {text:?}"
    );
}

#[tokio::test]
async fn a_tool_result_carrying_text_reaches_the_model() {
    const FIXTURE: &str = "../llamacpp/tool_matrix/tool_result_text.yaml";
    let wire = OpenAIConfig::with_key(&LLAMACPP, "").chat(CASSETTE_MODEL);
    let request = CompletionRequest::new(Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("call_text"),
            name: ToolName::new("lookup").expect("tool name"),
            answers: crate::message::AnsweredToolCall::Function,
            is_error: false,
            content: vec![ToolResultContent::text("the codeword is heliotrope")],
        })],
    })
    .preamble(
        "Answer using only the tool result you were given. \
                         Repeat the codeword verbatim."
            .to_string(),
    )
    .messages(vec![
        Message::User {
            content: vec![UserContent::text("What is the codeword?")],
        },
        lookup_call_turn("call_text"),
    ])
    .max_tokens(512);
    let encoded = wire
        .encode(request.clone(), Mode::Unary)
        .expect("the original text-result request encodes");
    let actual_request = json_body(&encoded.request);

    for request in [recorded_json("when", FIXTURE), actual_request] {
        let roles = request["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|message| message["role"].as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>();
        assert!(
            roles.contains(&"tool".to_string()),
            "the result must ride a `role: \"tool\"` message: {roles:?}"
        );
    }

    let response = Model::new(wire, RecordingHttpClient::new(recorded("then", FIXTURE)))
        .call(request)
        .await
        .expect("a text tool result should be accepted");
    let text = assistant_text_response(&response.choice).unwrap_or_default();
    assert!(
        text.to_ascii_lowercase().contains("heliotrope"),
        "the tool result's text must reach the model: {text:?}"
    );
}

#[tokio::test]
async fn a_tool_result_carrying_json_reaches_the_model() {
    const FIXTURE: &str = "../llamacpp/tool_matrix/tool_result_json.yaml";
    let wire = OpenAIConfig::with_key(&LLAMACPP, "").chat(CASSETTE_MODEL);
    let request = CompletionRequest::new(Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("call_json"),
            name: ToolName::new("lookup").expect("tool name"),
            answers: crate::message::AnsweredToolCall::Function,
            is_error: false,
            content: vec![ToolResultContent::text(
                json!({ "codeword": "heliotrope", "confidence": 0.99 }).to_string(),
            )],
        })],
    })
    .preamble(
        "Answer using only the JSON tool result you were given. \
                         Report the codeword verbatim."
            .to_string(),
    )
    .messages(vec![
        Message::User {
            content: vec![UserContent::text("What is the codeword?")],
        },
        lookup_call_turn("call_json"),
    ])
    .max_tokens(512);
    let encoded = wire
        .encode(request.clone(), Mode::Unary)
        .expect("the original JSON-result request encodes");
    let actual_request = json_body(&encoded.request);

    for request in [recorded_json("when", FIXTURE), actual_request] {
        let tool_message = request["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .find(|message| message["role"] == json!("tool"))
            .expect("a tool message");
        let serialized = tool_message["content"].to_string();
        assert!(
            serialized.contains("codeword") && serialized.contains("heliotrope"),
            "the document must survive serialization into the tool message: {serialized}"
        );
    }

    let response = Model::new(wire, RecordingHttpClient::new(recorded("then", FIXTURE)))
        .call(request)
        .await
        .expect("a JSON tool result should be accepted");
    let text = assistant_text_response(&response.choice).unwrap_or_default();
    assert!(
        text.to_ascii_lowercase().contains("heliotrope"),
        "the JSON result's contents must reach the model: {text:?}"
    );
}
