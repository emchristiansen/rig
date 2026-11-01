//! The per-provider gate on images in `role:"tool"` messages.
//!
//! Measured, not assumed. Official OpenAI Chat Completions answers 400 on
//! `gpt-4o`/`gpt-4o-mini` (*"Image URLs are only allowed for messages with
//! role 'user'"*) and, on `gpt-5` … `gpt-5.5`, answers 200 with the image
//! discarded and the model describing what it never received. llama.cpp
//! (`b1-6d05498`, Qwen3-VL-2B) delivers it: a magenta/green/yellow square
//! handed back through a tool is named correctly 3/3, matching a control
//! that sends the same bytes in a `user` message.
//!
//! These are unit tests rather than cassettes because the behaviour under
//! test is a *local* refusal — no request is made, so there is nothing to
//! record. The provider-side facts they encode are pinned by recorded cells
//! in the llamacpp and openai suites.

use super::*;
use crate::message;
use crate::providers::openai::wire::{Chat, OPENAI, OpenAIConfig, Quirks};
use crate::wire::{Mode, Wire};

fn params(
    supports_image_tool_results: bool,
    content: Vec<message::ToolResultContent>,
) -> (Chat, crate::completion::CompletionRequest) {
    let dialect = OPENAI.with_quirks(Quirks {
        supports_image_tool_results,
        ..Quirks::openai()
    });
    let wire = OpenAIConfig::new("test-key")
        .with_dialect(&dialect)
        .chat("test-model");
    let request = crate::completion::CompletionRequest::new(message::Message::User {
        content: vec![message::UserContent::tool_result(
            message::CallId::from_wire("call_1"),
            message::ToolName::new("view_file").expect("the fixture tool name is nonempty"),
            content,
        )],
    });
    (wire, request)
}

fn encode(
    (wire, request): (Chat, crate::completion::CompletionRequest),
) -> Result<crate::wire::Encoded, crate::error::EncodeError> {
    wire.encode(request, Mode::Unary)
}

fn image() -> message::ToolResultContent {
    message::ToolResultContent::image_base64(
        "iVBORw0KGgo=",
        Some(message::ImageMediaType::PNG),
        None,
    )
}

/// A provider that cannot carry the image refuses locally rather than
/// flattening it away or letting the provider answer.
#[test]
fn an_image_tool_result_is_refused_when_the_provider_cannot_carry_it() {
    let error = encode(params(false, vec![image()]))
        .expect_err("a provider without the capability must refuse");
    let message = error.to_string();
    assert!(
        message.contains("does not accept an image in a tool result"),
        "{message}"
    );
    assert!(
        message.contains("Responses API"),
        "the refusal should name the surface that works: {message}"
    );
}

/// The same request is built when the provider does carry it.
#[test]
fn an_image_tool_result_is_sent_when_the_provider_supports_it() {
    let request =
        encode(params(true, vec![image()])).expect("a capable provider should accept the image");
    let wire = crate::test_utils::json_body(&request.request)["messages"].clone();
    let content = &wire[0]["content"];
    assert_eq!(content[0]["type"], "image_url", "{wire}");
    assert!(
        content[0]["image_url"]["url"]
            .as_str()
            .is_some_and(|u| u.starts_with("data:image/png;base64,")),
        "{wire}"
    );
}

/// An image forces array form even when the provider flattens text results,
/// because a string has nowhere to put it.
#[test]
fn an_image_forces_array_content_even_when_flattening_is_configured() {
    let request = encode(params(true, vec![image()])).expect("build");
    let wire = crate::test_utils::json_body(&request.request)["messages"].clone();
    assert!(
        wire[0]["content"].is_array(),
        "image results must stay an array: {wire}"
    );
}

/// Text-only results are untouched by the gate, on both settings.
#[test]
fn a_text_tool_result_is_unaffected_by_the_gate() {
    for supports in [false, true] {
        let request = encode(params(
            supports,
            vec![message::ToolResultContent::text("ok")],
        ))
        .unwrap_or_else(|e| panic!("text results must always build (supports={supports}): {e}"));
        let wire = crate::test_utils::json_body(&request.request)["messages"].clone();
        assert_eq!(wire[0]["content"], "ok", "supports={supports}: {wire}");
    }
}

/// A mixed result is refused as a whole rather than silently losing its
/// image half.
#[test]
fn a_mixed_text_and_image_result_is_refused_rather_than_partly_sent() {
    let error = encode(params(
        false,
        vec![
            message::ToolResultContent::text("here is the file"),
            image(),
        ],
    ))
    .expect_err("the image half cannot be dropped silently");
    assert!(
        error.to_string().contains("does not accept an image"),
        "{error}"
    );
}

/// And the image shape round-trips.
#[test]
fn an_image_part_round_trips_through_serde() {
    let parsed: Message = serde_json::from_str(
        r#"{"role":"tool","tool_call_id":"c1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}"#,
    )
    .expect("an image part should deserialize");
    let Message::ToolResult { content, .. } = parsed else {
        panic!("expected a tool result");
    };
    assert!(content.has_image());
    let wire = serde_json::to_value(&content).expect("serialize");
    assert_eq!(wire[0]["type"], "image_url");
}

/// The adopted direct image gate also covers the shared header encoder.
#[test]
fn the_adopted_direct_image_gate_also_refuses_encode_with_headers() {
    let (wire, request) = params(false, vec![image()]);
    let error = wire
        .encode_with_headers(request, Mode::Unary, OpenAIConfig::completion_headers)
        .expect_err("a provider without the capability must refuse");
    let message = error.to_string();
    assert!(
        message.contains("does not accept an image in a tool result"),
        "{message}"
    );
    assert!(
        message.contains("Responses API"),
        "the refusal should name the surface that works: {message}"
    );
}
