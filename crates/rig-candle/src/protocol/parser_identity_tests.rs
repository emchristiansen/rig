use super::*;

#[test]
fn thinking_and_interleaved_text_do_not_change_inbound_call_ordinals() {
    let request = CompletionRequest::new("calculate");
    let raw = concat!(
        "<think>private</think>before",
        r#"<tool_call>{"name":"calculate","arguments":{"value":1}}</tool_call>"#,
        "between",
        r#"<tool_call>{"id":"tool-0","name":"lookup","arguments":{}}</tool_call>"#,
        "after",
        r#"<tool_call>{"name":"calculate","arguments":{"value":2}}</tool_call>"#,
        "last",
    );
    let first = parse_assistant(raw, &request, ConversationProtocol::Qwen3).expect("valid envelope");
    let again = parse_assistant(raw, &request, ConversationProtocol::Qwen3).expect("same envelope");
    assert_eq!(first.items, again.items);
    assert_eq!(first.visible_text, "before between after last");
    assert_eq!(first.items.len(), 8);
    assert!(matches!(&first.items[0], AssistantContent::Reasoning(reasoning) if reasoning.text == "private"));
    for (at, expected) in [(1, "before"), (3, " between"), (5, " after"), (7, " last")] {
        assert!(matches!(&first.items[at], AssistantContent::Text(text) if text.text == expected));
    }
    let calls = first.items.iter().filter_map(|part| match part {
        AssistantContent::ToolCall(call) => Some(call),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].id, CallId::Local(LocalCallId::for_candle_response_position(0)));
    assert_eq!(calls[1].id, CallId::from_wire("tool-0"));
    assert_eq!(calls[2].id, CallId::Local(LocalCallId::for_candle_response_position(2)));
    assert_eq!(calls[0].function.arguments_value(), serde_json::json!({"value":1}));
    assert_eq!(calls[1].function.arguments_value(), serde_json::json!({}));
    assert_eq!(calls[2].function.arguments_value(), serde_json::json!({"value":2}));
    assert!(calls.iter().all(|call| call.item_id.is_none()));
}
