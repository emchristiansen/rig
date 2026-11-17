//! What a caller of Rig's own fold gets for the Responses bodies this
//! crate's typed data describes.

use rig_core::completion::{AssistantContent, CompletionRequest};
use rig_core::message::{Message, ToolCall};
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Mode, WireFrame, decode_reply};
use serde_json::{Value, json};
use std::sync::Arc;

/// `body`, a whole Responses reply, folded by Rig as a unary call's reply.
fn folded(body: Value) -> rig_core::completion::CompletionResponse {
    let wire = Responses::new(OpenAIConfig::new("sk-test"), "gpt-5.4");
    decode_reply(
        &wire,
        &CompletionRequest::new("say hi"),
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
    )
    .expect("the reply folds")
}

fn reply(extra: Value, output: Value) -> Value {
    let mut body = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": output,
        "tools": []
    });
    for (key, value) in extra.as_object().expect("an object") {
        body[key] = value.clone();
    }
    body
}

fn message() -> Value {
    json!({
        "type": "message", "id": "msg_123", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "annotations": [], "text": "done"}]
    })
}

fn call(name: &str, extra: Value) -> Value {
    let mut item = json!({"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": name, "status": "completed"});
    for (key, value) in extra.as_object().expect("an object") {
        item[key] = value.clone();
    }
    item
}

/// The one tool call a reply folds to.
fn only_call(response: &rig_core::completion::CompletionResponse) -> &ToolCall {
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("expected one tool call, got {:?}", response.choice);
    };
    call
}

/// A top-level reasoning string folds to a reasoning block ahead of the
/// message's text.
#[test]
fn a_top_level_reasoning_string_folds_to_reasoning_ahead_of_the_text() {
    let response = folded(reply(
        json!({"reasoning": "thinking through the answer"}),
        json!([message()]),
    ));
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!("expected reasoning then text, got {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "thinking through the answer");
    assert_eq!(text.text, "done");
}

/// A reply that is reasoning alone folds to that one block.
#[test]
fn a_reasoning_only_reply_folds_to_one_reasoning_block() {
    let response = folded(reply(json!({"reasoning": "thinking only"}), json!([])));
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("expected one reasoning block, got {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "thinking only");
}

/// A reasoning configuration object is not reasoning text: the reply folds
/// to its message alone.
#[test]
fn a_reasoning_configuration_is_not_folded_as_text() {
    let response = folded(reply(
        json!({"reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": null}}),
        json!([message()]),
    ));
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("expected the text alone, got {:?}", response.choice);
    };
    assert_eq!(text.text, "done");
}

/// A custom tool call folds to a call whose input is reachable, and keeps
/// the provider's item whole, with no namespace invented.
#[test]
fn a_custom_tool_call_folds_with_its_input_and_its_item() {
    let item = json!({
        "type": "custom_tool_call", "id": "ctc_9", "call_id": "call_9",
        "name": "grammar_tool", "input": "SELECT 1", "status": "completed",
    });
    let response = folded(reply(json!({}), json!([item])));
    let call = only_call(&response);
    assert_eq!(call.function.name.as_str(), "grammar_tool");
    assert_eq!(call.function.arguments.get("input"), Some(&json!("SELECT 1")));
    assert_eq!(call.function.invalid_arguments, None);
    let native = call.native.as_ref().expect("the provider's item is kept");
    assert_eq!(native.item, item);
    assert_eq!(native.item.get("namespace"), None);
}

/// A namespaced function call keeps its namespace in the provider's item,
/// and an unqualified one gains none.
#[test]
fn a_function_calls_namespace_is_kept_in_its_item() {
    let namespaced = call("search", json!({"namespace": "web", "arguments": "{\"q\":\"rig\"}"}));
    let response = folded(reply(json!({}), json!([namespaced])));
    let folded_call = only_call(&response);
    assert_eq!(folded_call.function.name.as_str(), "search");
    assert_eq!(folded_call.function.arguments.get("q"), Some(&json!("rig")));
    let native = folded_call.native.as_ref().expect("the provider's item");
    assert_eq!(native.item["namespace"], "web");

    let unqualified = call("search", json!({"arguments": "{\"q\":\"rig\"}"}));
    let response = folded(reply(json!({}), json!([unqualified])));
    let native = only_call(&response).native.as_ref().expect("the item");
    assert_eq!(native.item.get("namespace"), None);
}

/// A call whose arguments are not a JSON object is delivered, marked: Rig
/// v0.46.0 does not refuse it (as the fork did when the provider called the
/// item complete) or drop it (as the fork did otherwise). The arguments as
/// the model sent them are in `invalid_arguments` and in the provider's
/// item, whatever the item's status. A caller that must not run such a call
/// refuses on the mark.
#[test]
fn a_call_with_malformed_arguments_is_delivered_marked_with_the_raw_text() {
    for status in ["completed", "incomplete"] {
        let malformed = call("add", json!({"arguments": "{\"x\":481", "status": status}));
        let response = folded(reply(json!({}), json!([malformed.clone()])));
        let folded_call = only_call(&response);
        assert_eq!(
            folded_call.function.invalid_arguments.as_deref(),
            Some("{\"x\":481"),
            "{status}: the mark carries the arguments as sent"
        );
        let native = folded_call.native.as_ref().expect("the provider's item");
        assert_eq!(native.item, malformed, "{status}");
    }
}

// ── a streamed turn ─────────────────────────────────────────────────────

fn chatgpt_wire() -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&rig_core::providers::chatgpt::DIALECT, "test-token"),
        "gpt-5.4",
    )
}

/// A streamed turn's frames: `response.created`, then each of `items` added
/// and done, then `terminal` restating them all.
fn streamed_frames(items: &[Value], terminal: &str, extra: Value) -> Vec<WireFrame> {
    let response = |status: &str, output: Value| {
        let mut body = json!({
            "id": "resp_1", "object": "response", "created_at": 0, "status": status,
            "model": "gpt-5.4", "output": output,
        });
        for (key, value) in extra.as_object().expect("an object") {
            body[key] = value.clone();
        }
        body
    };
    let mut sequence = 0;
    let mut frame = |mut event: Value| {
        event["sequence_number"] = json!(sequence);
        sequence += 1;
        WireFrame::Text(event.to_string())
    };
    let mut frames = vec![frame(
        json!({"type": "response.created", "response": response("in_progress", json!([]))}),
    )];
    for (index, item) in items.iter().enumerate() {
        frames.push(frame(
            json!({"type": "response.output_item.added", "output_index": index, "item": item}),
        ));
        frames.push(frame(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        ));
    }
    let status = terminal.strip_prefix("response.").expect("a response event");
    frames.push(frame(
        json!({"type": terminal, "response": response(status, json!(items))}),
    ));
    frames
}

/// The provider items of a turn that reasons, searches, answers with a
/// refusal beside its text, and makes a custom call, a namespaced call and
/// a call with malformed arguments.
fn turn_items() -> [Value; 6] {
    [
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "step 1"}], "encrypted_content": "enc_blob"}),
        json!({"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "rig"}}),
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "annotations": [], "text": "hello"}, {"type": "refusal", "refusal": "no"}]}),
        json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c", "name": "apply_patch", "namespace": "editor", "input": "{\"looks\":\"like json\"}", "status": "completed"}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search", "namespace": "web", "arguments": "{\"q\":\"rig\"}", "status": "completed"}),
        json!({"type": "function_call", "id": "fc_2", "call_id": "call_m", "name": "add", "arguments": "{\"x\":481", "status": "completed"}),
    ]
}

fn streamed_turn() -> rig_core::completion::CompletionResponse {
    decode_reply(
        &chatgpt_wire(),
        &CompletionRequest::new("say hi"),
        Mode::Streaming,
        streamed_frames(&turn_items(), "response.completed", json!({})),
    )
    .expect("the turn folds")
}

/// The request that follows `turn`: its history, an answer to each call,
/// and the next user message.
fn following(turn: &rig_core::completion::CompletionResponse) -> CompletionRequest {
    let mut next = CompletionRequest::new("say hi");
    let mut history = next.chat_history.clone();
    history.push(Arc::new(turn.message().expect("the turn's message")));
    for call in turn.tool_calls() {
        history.push(Arc::new(Message::tool_result(
            call.id.clone(),
            call.function.name.clone(),
            "ok",
        )));
    }
    history.push(Arc::new(Message::user("next")));
    next.chat_history = history;
    next
}

/// A streamed turn folds to one block per provider item, each keeping its
/// item whole: reasoning with its ciphertext, an item Rig does not model as
/// an opaque block, a custom call with its input and namespace, and a
/// namespaced call. A message's refusal part is folded into the block's
/// text; the item says which part was the refusal.
#[test]
fn a_streamed_turn_folds_to_blocks_that_keep_their_provider_items() {
    let turn = streamed_turn();
    let items = turn_items();
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Opaque(search),
        AssistantContent::Text(text),
        AssistantContent::ToolCall(custom),
        AssistantContent::ToolCall(namespaced),
        AssistantContent::ToolCall(malformed),
    ] = turn.choice.as_slice()
    else {
        panic!("expected one block per item, got {:?}", turn.choice);
    };
    let item_of = |native: &Option<rig_core::message::Native>| {
        native.as_ref().expect("the provider's item").item.clone()
    };
    assert_eq!(reasoning.text, "step 1");
    assert_eq!(item_of(&reasoning.native), items[0]);
    assert_eq!(search.item, items[1]);
    assert_eq!(text.text, "hellono");
    assert_eq!(item_of(&text.native), items[2]);
    assert_eq!(custom.function.name.as_str(), "apply_patch");
    assert_eq!(
        custom.function.arguments.get("input"),
        Some(&json!("{\"looks\":\"like json\"}")),
        "a custom call's input stays the text the model wrote"
    );
    assert_eq!(item_of(&custom.native), items[3]);
    assert_eq!(namespaced.function.name.as_str(), "search");
    assert_eq!(item_of(&namespaced.native), items[4]);
    assert_eq!(
        malformed.function.invalid_arguments.as_deref(),
        Some("{\"x\":481")
    );
    assert_eq!(item_of(&malformed.native), items[5]);
}

/// On a request that declares tools, the turn replays as the provider's own
/// items, each exactly as it arrived (ciphertext, namespaces, the refusal
/// part and the malformed arguments included), followed by each call's
/// answer in its own item kind.
#[test]
fn a_turn_replays_as_the_providers_own_items_on_a_request_that_declares_tools() {
    let mut next = following(&streamed_turn());
    next.additional_params = Some(json!({
        "tools": [crate::responses_api::fidelity_fixtures::namespace()]
    }));
    let input = crate::responses_api::input_items(&chatgpt_wire(), next).expect("encodes");

    let mut expected = vec![json!({
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "say hi"}]
    })];
    expected.extend(turn_items());
    expected.extend([
        // Rig v0.46.0 states no `name` on a custom call's output; the fork
        // stated the tool's name there.
        json!({"type": "custom_tool_call_output", "call_id": "call_c", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
        json!({"type": "function_call_output", "call_id": "call_m", "output": "ok", "status": "completed"}),
        json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]}),
    ]);
    assert_eq!(input, expected);
}

/// On a request that declares no tools, Rig v0.46.0 replays the turn's
/// calls and their answers as text, where the fork replayed the items. A
/// caller that replays tool history declares its tools.
#[test]
fn a_request_that_declares_no_tools_replays_calls_as_text() {
    let input = crate::responses_api::input_items(&chatgpt_wire(), following(&streamed_turn()))
        .expect("encodes");
    let kinds: Vec<&str> = input
        .iter()
        .map(|item| item["type"].as_str().expect("a typed item"))
        .collect();
    assert_eq!(
        kinds,
        [
            "message",
            "reasoning",
            "web_search_call",
            "message",
            "message",
            "message",
            "message",
            "message",
            "message",
            "message",
            "message"
        ],
        "{input:?}"
    );
    assert_eq!(
        input[4]["content"][0]["text"],
        "[called tool apply_patch with {\"input\":\"{\\\"looks\\\":\\\"like json\\\"}\"}]"
    );
    assert_eq!(input[7]["content"][0]["text"], "[tool apply_patch result] ok");
}

/// A turn the provider ends `incomplete` is a reply, not an error: its
/// partial text is kept and its finish reason says the output was cut.
#[test]
fn a_streamed_incomplete_terminal_is_a_reply_with_its_partial_text() {
    let partial = [json!({
        "type": "message", "id": "msg_1", "status": "incomplete", "role": "assistant",
        "content": [{"type": "output_text", "annotations": [], "text": "partial"}]
    })];
    let turn = decode_reply(
        &chatgpt_wire(),
        &CompletionRequest::new("say hi"),
        Mode::Streaming,
        streamed_frames(
            &partial,
            "response.incomplete",
            json!({"incomplete_details": {"reason": "max_output_tokens"}}),
        ),
    )
    .expect("an incomplete turn folds");
    assert_eq!(turn.text(), "partial");
    assert_eq!(
        turn.finish_reason(),
        Some(rig_core::completion::FinishReason::Length)
    );
    assert_eq!(turn.response_id(), Some("resp_1"));
}
