//! Genuine terminal proof, exact identity, metadata, usage and incomplete policy.

use super::helpers::*;
use super::super::{IncompleteTerminal, ResponsesStreamOptions};
use crate::completion::{CompletionRequest, FinishReason};
use crate::providers::openai::wire::OpenAIConfig;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::{Mode, Wire, WireFrame};
use futures::StreamExt;
use serde_json::json;

fn message_output_item(id: &str, text: &str) -> serde_json::Value {
    message_item(id, json!([{"type":"output_text","annotations":[],"text":text}]))
}

// F streaming/tests.rs2235
#[test]
fn terminal_body_message_text_merges_when_no_delta_delivered_it() {
    let mut raw_response = sample_response("completed");
    raw_response["output"] = json!([message_output_item("msg_body_1", "from body")]);
    let completed = json!({"type":"response.completed","sequence_number":1,"response":raw_response});
    let response = folded_body_with(ResponsesStreamOptions::strict(), &[completed]).unwrap();
    assert_eq!(folded_texts(&response.choice), ["from body"], "body-only text reaches the choice once");
    assert_eq!(response.message_id.as_ref().map(|id| id.as_str()), Some("msg_body_1"));
}

// F streaming/tests.rs2264
#[test]
fn terminal_body_message_text_restating_a_delta_is_not_duplicated() {
    let text_delta = json!({"type":"response.output_text.delta","item_id":"msg_body_1",
        "output_index":0,"content_index":0,"sequence_number":1,"delta":"from body"});
    let mut raw_response = sample_response("completed");
    raw_response["output"] = json!([message_output_item("msg_body_1", "from body")]);
    let completed = json!({"type":"response.completed","sequence_number":2,"response":raw_response});
    let response = folded_body_with(ResponsesStreamOptions::strict(), &[text_delta, completed]).unwrap();
    assert_eq!(folded_texts(&response.choice), ["from body"], "terminal restatement adds no duplicate delta-built text");
}

// F streaming/tests.rs2336
#[tokio::test]
async fn response_completed_chunk_populates_final_usage() {
    let mut response = sample_response("completed");
    response["usage"] = json!({"input_tokens":10,"input_tokens_details":null,
        "output_tokens":5,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":15});
    let event = json!({"type":"response.completed","sequence_number":1,"response":response});
    let response = finish_events(&[event], IncompleteTerminal::Refuse).await;
    assert_eq!(response.raw["usage"]["input_tokens"], 10);
    assert_eq!(response.raw["usage"]["output_tokens"], 5);
    assert_eq!(response.raw["usage"]["total_tokens"], 15);
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

// F streaming/tests.rs2370
#[tokio::test]
async fn response_completed_chunk_tolerates_object_shaped_top_p() {
    let mut response = sample_response("completed");
    response["top_p"] = json!({"value":0.95});
    response["usage"] = json!({"input_tokens":10,"output_tokens":5,"total_tokens":15});
    let event = json!({"type":"response.completed","sequence_number":1,"response":response});
    let response = finish_events(&[event], IncompleteTerminal::Refuse).await;
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.total_tokens, Some(15));
    assert_eq!(response.raw["top_p"], json!({"value":0.95}));
}

// F streaming/tests.rs2394
#[tokio::test]
async fn response_completed_chunk_populates_reasoning_metadata_and_context() {
    let metadata = json!({"context":"all_turns","effort":"ultra","summary":null,"future_control":true});
    let mut response = sample_response("completed");
    response["reasoning"] = metadata.clone();
    let event = json!({"type":"response.completed","sequence_number":1,"response":response});
    let response = finish_events(&[event], IncompleteTerminal::Refuse).await;
    assert_eq!(response.raw["reasoning"]["context"].as_str(), Some("all_turns"));
    assert_eq!(response.raw["reasoning"], metadata);
}

// F streaming/tests.rs2415
#[tokio::test]
async fn terminal_record_normalizes_into_the_stream_final() {
    let mut response = sample_response("completed");
    response["usage"] = json!({"input_tokens":10,"input_tokens_details":null,
        "output_tokens":5,"output_tokens_details":null,"total_tokens":15});
    response["output"] = json!([message_output_item("msg_stream_1", "hi")]);
    let event = json!({"type":"response.completed","sequence_number":1,"response":response});
    let response = finish_events(&[event], IncompleteTerminal::Refuse).await;
    assert_eq!(response.provider(), "openai");
    assert_eq!(response.model(), Some("gpt-5.4"));
    assert_eq!(response.response_id(), Some("resp_123"));
    assert_eq!(response.message_id.as_ref().map(|id| id.as_str()), Some("msg_stream_1"));
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

// F streaming/tests.rs2455: keep explicit finish_reason, not just stop()==ToolUse.
#[tokio::test]
async fn terminal_record_reports_tool_calls_when_the_stream_called_a_tool() {
    let done = json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,
        "item":{"type":"function_call","id":"fc_123","arguments":"{}","call_id":"call_123",
            "name":"example_tool","status":"completed"}});
    let completed = json!({"type":"response.completed","sequence_number":2,"response":sample_response("completed")});
    let response = finish_events(&[done, completed], IncompleteTerminal::Refuse).await;
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

// F streaming/tests.rs2499: deleted terminal DTO fields now pin actual native response.
#[test]
fn terminal_record_preserves_an_unknown_incomplete_reason() {
    let mut raw = sample_response("incomplete");
    raw["incomplete_details"] = json!({"reason":"MAX_TOOL_CALLS"});
    raw["output"] = json!([message_output_item("msg_1", "")]);
    let event = json!({"type":"response.incomplete","sequence_number":1,"response":raw});
    let response = folded_body_with(ResponsesStreamOptions::strict().with_incomplete(IncompleteTerminal::Accept), &[event]).unwrap();
    assert_eq!(response.finish_reason(), Some(FinishReason::Other("MAX_TOOL_CALLS".into())));
    assert_eq!(response.message_id.as_ref().map(|id| id.as_str()), Some("msg_1"));
    assert_eq!(response.model(), Some("gpt-5.4"));
    assert_eq!(response.raw, raw);
}

// F streaming/tests.rs2895
#[tokio::test]
async fn a_streamed_incomplete_terminal_errors_by_default() {
    let mut stream = responses_stream_of(&incomplete_turn_events()).await;
    let mut error = None;
    while let Some(item) = stream.next().await {
        if let Err(err) = item { error = Some(err); break; }
    }
    let error = error.expect("the incomplete terminal must surface an error");
    let rendered = error.to_string();
    assert!(rendered.contains("response.incomplete") && rendered.contains("max_output_tokens"), "{rendered}");
    assert!(stream.partial().response_id().is_none(), "a refused terminal proves no completed response");
    assert!(stream.finish().await.is_err());
}

// F streaming/tests.rs2921
#[tokio::test]
async fn a_streamed_incomplete_terminal_is_accepted_with_the_opt_in() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4").with_streamed_incomplete(IncompleteTerminal::Accept);
    let sse: String = incomplete_turn_events().iter().map(|event| format!("data: {event}\n\n")).collect();
    let mut stream = crate::driver::Model::new(wire, MockStreamingClient { sse_bytes: sse.into() })
        .stream(CompletionRequest::new("hello")).unwrap();
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: delta, .. }) = item.expect("accepted incomplete does not error") {
            text.push_str(&delta);
        }
    }
    assert_eq!(text, "partial");
    assert_eq!(stream.finish().await.expect("genuine accepted terminal").finish_reason(), Some(FinishReason::Length));
}

// F streaming/tests.rs2958
#[test]
fn a_unary_reply_accepts_an_incomplete_terminal() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let decoded = crate::driver::feed_frames!(wire.decoder_in(Mode::Unary), "openai",
        incomplete_turn_events().into_iter().map(|event| WireFrame::Text(event.to_string())));
    let response = decoded.outcome.expect("unary incomplete is not an error");
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
}

// F streaming/tests.rs2984
#[test]
fn the_decoder_api_lets_a_session_accept_an_incomplete_terminal() {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let decoded = crate::driver::feed_frames!(wire.decoder_with_incomplete(IncompleteTerminal::Accept), "openai",
        incomplete_turn_events().into_iter().map(|event| WireFrame::Text(event.to_string())));
    assert!(decoded.items.iter().all(Result::is_ok), "{:?}", decoded.items);
    let response = decoded.outcome.expect("the accepting decoder establishes a genuine terminal");
    assert_eq!(response.response_id(), Some("resp_123"));
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
}
