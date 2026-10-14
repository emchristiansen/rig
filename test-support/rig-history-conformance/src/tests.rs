//! The private failed-cut predicate observes carried custody, not request admission.

use super::*;
use rig_core::providers::anthropic::{AnthropicConfig, completion::CLAUDE_SONNET_4_6};
use rig_core::wire::WireFrame;
use serde_json::json;

fn decoded_hosted_cut() -> CompletionResponse {
    // Newly authored synthetic received frames use the real Messages decoder/fold.
    let wire = AnthropicConfig::new("synthetic-test-key").completion(CLAUDE_SONNET_4_6);
    let frames = [
        json!({"type":"message_start", "message":{"type":"message", "id":"msg_observer",
            "role":"assistant", "model":CLAUDE_SONNET_4_6, "content":[], "stop_reason":null,
            "stop_sequence":null, "usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start", "index":0, "content_block":{
            "type":"server_tool_use", "id":"srv_observer", "name":"web_search", "input":{"query":"synthetic observer"}}}),
        json!({"type":"content_block_stop", "index":0}),
    ];
    let (response, ended) = partial(
        &wire,
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
    );
    assert!(!ended);
    assert!(matches!(response.stop(), StopReason::Aborted(_)));
    let [AssistantContent::CarriedText(part)] = response.choice.as_slice() else {
        panic!("actual carried hosted custody");
    };
    assert_eq!(part.payload.provider.as_deref(), Some("anthropic"));
    assert_eq!(part.payload.item, frames[1]["content_block"]);
    assert_eq!(
        part.payload.position,
        rig_core::message::CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 0
        }
    );
    response
}

#[test]
fn h07_decoded_failed_hosted_cut_is_retained_as_carried_custody() {
    let response = decoded_hosted_cut();
    let before = response.clone();
    let source_serde = serde_json::to_vec(&response).expect("whole received custody");
    let wire = AnthropicConfig::new("synthetic-test-key").completion(CLAUDE_SONNET_4_6);
    let history = vec![
        Message::user("q"),
        response.message().expect("real failed message"),
        Message::user("next"),
    ];
    let history_before = history.clone();
    let describe = wire.describe();
    let adapted = rig_core::completion::history::adapt(
        &history,
        describe.replay.expect("actual Messages target"),
    );
    assert!(
        adapted
            .iter()
            .any(|message| matches!(message, Message::Assistant(_)))
    );
    assert!(
        adapted.iter().all(failed_cut_custody),
        "the exact private predicate used by H07 admits custody"
    );
    assert_eq!(history, history_before);
    assert_eq!(response, before);
    assert_eq!(
        serde_json::to_vec(&response).expect("received after"),
        source_serde
    );
}

#[test]
fn h07_synthetic_ordinary_leaks_and_nonfailed_carried_heads_fail_the_actual_predicate() {
    let response = decoded_hosted_cut();
    let before = response.clone();
    let source_serde = serde_json::to_vec(&response).expect("whole received custody");
    let wire = AnthropicConfig::new("synthetic-test-key").completion(CLAUDE_SONNET_4_6);
    let raw = json!({"type":"message", "id":"msg_ordinary_control", "model":CLAUDE_SONNET_4_6,
    "role":"assistant", "stop_reason":"tool_use", "stop_sequence":null,
    "usage":{"input_tokens":1,"output_tokens":3}, "content":[
        {"type":"text", "text":"synthetic native text"},
        {"type":"thinking", "thinking":"synthetic ordinary thought", "signature":"synthetic signature"},
        {"type":"tool_use", "id":"toolu_observer", "name":"lookup", "input":{}}
    ]});
    let ordinary = decode(
        &wire,
        &CompletionRequest::new("synthetic fixture"),
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
    )
    .expect("genuine decoder constructs ordinary/native blocks");
    assert_eq!(ordinary.choice.len(), 3);
    assert!(
        ordinary
            .choice
            .iter()
            .all(|part| part.native_item().is_some())
    );
    let ordinary_before = ordinary.clone();
    for leak in ordinary
        .choice
        .iter()
        .cloned()
        .chain([AssistantContent::text("synthetic canonical text leak")])
    {
        // Synthetic observer input deliberately appends an ordinary block to
        // a clone of a genuinely received failed head, without changing its source.
        let mut turn = response.head().with_content(response.choice.clone());
        turn.content.push(leak);
        let history = vec![
            Message::user("q"),
            Message::Assistant(turn),
            Message::user("next"),
        ];
        let before = serde_json::to_vec(&history).expect("leaked input serde");
        assert!(
            !history.iter().all(failed_cut_custody),
            "the actual H07 predicate detects the ordinary leak"
        );
        assert_eq!(
            serde_json::to_vec(&history).expect("leaked input after"),
            before
        );
    }
    for stop in [None, Some(StopReason::Stop)] {
        let mut turn = response.head().with_content(response.choice.clone());
        turn.stop = stop; // Labelled synthetic failure-metadata boundary.
        assert!(!failed_cut_custody(&Message::Assistant(turn)));
    }
    assert_eq!(ordinary, ordinary_before);
    assert_eq!(response, before);
    assert_eq!(
        serde_json::to_vec(&response).expect("source after"),
        source_serde
    );
}
