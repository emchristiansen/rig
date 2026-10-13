//! Native paused custody through the real Messages parser, decoder and stream.

use crate::completion::{CompletionRequest, FinishReason, ToolDefinition};
use crate::driver::Model;
use crate::message::{AssistantContent, CarriedPosition, Message, StopReason, ToolName};
use crate::operation::Completion;
use crate::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use crate::providers::anthropic::wire::{AnthropicConfig, Messages, ZAI};
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::{MockStreamingClient, json_body};
use crate::wire::{Mode, Operation, Wire, WireFrame};
use futures::StreamExt;
use serde_json::{Value, json};

fn wire() -> Messages { AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6) }
fn hosted() -> Value {
    json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rig"}})
}
fn unary(reason: &str) -> Value {
    json!({"type":"message","id":"msg_1","model":CLAUDE_SONNET_4_6,"role":"assistant",
        "stop_reason":reason,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1},
        "content":[{"type":"text","text":"Searching."},hosted()]})
}
fn frames(reason: &str, stopped_text: bool, cited: bool) -> Vec<Value> {
    let mut frames = vec![
        json!({"type":"message_start","message":{"id":"msg_1","model":CLAUDE_SONNET_4_6,
            "role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Searching."}}),
    ];
    if cited { frames.push(json!({"type":"content_block_delta","index":0,"delta":{
        "type":"citations_delta","citation":{"type":"web_search_result_location",
            "url":"https://example.com/search","title":"Search","cited_text":"Searching."}}})); }
    if stopped_text { frames.push(json!({"type":"content_block_stop","index":0})); }
    frames.extend([
        json!({"type":"content_block_start","index":1,"content_block":hosted()}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":reason,"stop_sequence":null},
            "usage":{"output_tokens":1}}),
    ]);
    frames
}
fn decode(wire: &Messages, mode: Mode, frames: &[Value]) -> crate::completion::CompletionResponse {
    crate::test_utils::decode_reply(wire, &CompletionRequest::new("search"), mode,
        frames.iter().map(|frame| WireFrame::Text(frame.to_string())), Value::Null)
        .expect("real native response")
}
fn assert_paused(response: &crate::completion::CompletionResponse) {
    assert_eq!(response.finish_reason(), Some(FinishReason::Other("pause_turn".to_owned())));
    assert_eq!(response.stop(), StopReason::Error("Provider finish_reason: pause_turn".to_owned()));
    assert_eq!(response.origin.provider, crate::providers::anthropic::PROVIDER_NAME);
    assert_eq!(response.response_id(), Some("msg_1"));
    assert_eq!(response.model(), Some(CLAUDE_SONNET_4_6));
    assert_eq!(response.usage.input_tokens, Some(1));
    assert_eq!(response.usage.output_tokens, Some(1));
    assert_eq!(response.choice.len(), 2);
    let AssistantContent::CarriedText(text) = &response.choice[0] else { panic!("paused text custody"); };
    assert_eq!(text.text(), "Searching.");
    assert_eq!(text.payload.provider.as_deref(), Some(crate::providers::anthropic::PROVIDER_NAME));
    assert_eq!(text.payload.position, CarriedPosition::MessageContent { item_index:0,part_index:0 });
    assert_eq!(text.payload.item["type"], "text");
    assert_eq!(text.payload.item["text"], "Searching.");
    let AssistantContent::CarriedText(server) = &response.choice[1] else { panic!("server custody"); };
    assert_eq!(server.payload.item, hosted());
    assert!(response.tool_calls().next().is_none(), "no client-tool dispatch is introduced");
    let target = wire();
    let request = CompletionRequest::from(vec![Message::user("search"),
        response.message().expect("genuine paused message")]).max_tokens(64)
        .tools(vec![ToolDefinition::new(ToolName::new("add").expect("name"),
            "add description", json!({"type":"object","properties":{}}))]);
    let prepared = Completion::prepare(request, &target.describe()).expect("original paused custody prepares");
    let encoded = target.encode(prepared, Mode::Unary).expect("paused continuation encodes");
    assert_eq!(json_body(&encoded.request)["messages"][1]["content"],
        json!([{"type":"text","text":"Searching."},hosted()]));
}

#[test]
fn original_paused_unary_values_keep_both_received_blocks_and_other_finish() {
    let original = unary("pause_turn");
    let response = decode(&wire(), Mode::Unary, &[original.clone()]);
    assert_eq!(response.raw, original);
    assert_paused(&response);
    let AssistantContent::CarriedText(text) = &response.choice[0] else { panic!("known text"); };
    assert_eq!(text.payload.item, original["content"][0]);
}

#[test]
fn unread_stream_end_changes_queued_text_custody_without_duplicate_output() {
    let response = decode(&wire(), Mode::Streaming, &frames("pause_turn", true, false));
    assert_paused(&response);
    assert_eq!(response.raw, unary("pause_turn"));
}

#[tokio::test]
async fn delivered_text_end_is_unchanged_while_terminal_custody_is_carried() {
    let frames = frames("pause_turn", true, false);
    let bytes = frames.iter().map(|frame| format!("data: {frame}\n\n")).collect::<String>();
    let mut stream = Model::new(wire(), MockStreamingClient { sse_bytes: bytes::Bytes::from(bytes) })
        .stream(CompletionRequest::new("search")).expect("real native stream");
    let mut delivered = Vec::new();
    let text_part;
    loop {
        let item = stream.next().await.expect("text boundary arrives").expect("native item");
        let ended = match &item {
            Item::Event(StreamEvent::End { part, content: AssistantContent::Text(text), .. }) => {
                assert_eq!(text.text(), "Searching."); Some(*part)
            }
            _ => None,
        };
        delivered.push(item);
        if let Some(part) = ended { text_part = part; break; }
    }
    assert!(matches!(&stream.partial().choice[0], AssistantContent::Text(_)));
    while let Some(item) = stream.next().await { delivered.push(item.expect("remaining native item")); }
    let response = stream.finish().await.expect("actual paused end");
    assert_paused(&response);
    // G151 named observable divergence: delivered End stays ordinary Text,
    // while later folded/terminal custody is carried on the same Part.
    let text_ends = delivered.iter().filter_map(|item| match item {
        Item::Event(StreamEvent::End { part, content, .. }) if *part == text_part => Some(content),
        _ => None,
    }).collect::<Vec<_>>();
    assert!(matches!(text_ends.as_slice(), [AssistantContent::Text(text)] if text.text() == "Searching."));
    assert_eq!(delivered.iter().filter(|item| matches!(item,
        Item::Event(StreamEvent::Start { part, .. }) if *part == text_part)).count(), 1);
    assert_eq!(response.raw, unary("pause_turn"));
}

#[test]
fn paused_open_text_uses_existing_finish_custody_with_exact_received_citations() {
    let stopped = decode(&wire(), Mode::Streaming, &frames("pause_turn", true, true));
    let open = decode(&wire(), Mode::Streaming, &frames("pause_turn", false, true));
    for response in [&stopped, &open] { assert_paused(response); }
    let (AssistantContent::CarriedText(stopped), AssistantContent::CarriedText(open)) =
        (&stopped.choice[0], &open.choice[0]) else { panic!("actual carried text"); };
    assert_eq!(open.payload, stopped.payload);
    assert_eq!(open.citations(), stopped.citations());
    assert_eq!(open.citations().len(), 1);
    assert_eq!(open.payload.item["citations"], json!([{"type":"web_search_result_location",
        "url":"https://example.com/search","title":"Search","cited_text":"Searching."}]));
}

#[test]
fn nonpaused_and_compatible_text_preserve_their_existing_native_categories() {
    for reason in ["end_turn", "stop_sequence", "tool_use", "max_tokens"] {
        let response = decode(&wire(), Mode::Streaming, &frames(reason, true, false));
        assert!(matches!(&response.choice[0], AssistantContent::Text(_)));
        assert_eq!(response.choice[0].native_item().expect("actual ordinary native")["text"], "Searching.");
    }
    let compatible = AnthropicConfig::with_key(&ZAI, "test-key").completion("glm-5");
    let response = decode(&compatible, Mode::Streaming, &frames("pause_turn", true, false));
    assert!(matches!(&response.choice[0], AssistantContent::Text(_)));
    assert_eq!(response.origin.provider, crate::providers::zai::PROVIDER_NAME);
    assert_eq!(response.finish_reason(), Some(FinishReason::Other("pause_turn".to_owned())));
}

#[test]
fn blank_native_paused_text_keeps_the_original_empty_and_whitespace_views() {
    for text in ["", " "] {
        let response = decode(&wire(), Mode::Unary, &[json!({"type":"message","id":"msg_1",
            "model":CLAUDE_SONNET_4_6,"role":"assistant","stop_reason":"pause_turn",
            "stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1},
            "content":[{"type":"text","text":text},hosted()]})]);
        assert!(response.choice.iter().all(|part| !matches!(part,
            AssistantContent::CarriedText(text) if text.known_text_projection().is_some())));
        if text.is_empty() { assert_eq!(response.choice.len(), 1); }
        else { assert!(matches!(&response.choice[0], AssistantContent::Text(value) if value.text() == text)); }
        assert_eq!(response.raw["content"][0]["text"], text);
        assert_eq!(response.finish_reason(), Some(FinishReason::Other("pause_turn".to_owned())));
    }
}
