//! A reasoning item that ends its turn: what the next request replays for
//! it on the Codex contract, where this fork sends it, and on OpenAI's own,
//! where upstream Rig's rule holds and it is not sent. Every turn is folded
//! from a reply body by Rig's own decoder on the wire it is replayed on, and
//! prepared and encoded as a request is.

use super::fold_contract_tests::{chatgpt_wire, following, reply};
use rig_core::completion::{AssistantContent, CompletionRequest, CompletionResponse};
use rig_core::message::{Message, Text};
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Mode, WireFrame, decode_reply};
use serde_json::{Value, json};
use std::sync::Arc;

/// OpenAI's own Responses dialect, at the model [`chatgpt_wire`] addresses.
fn openai_wire() -> Responses {
    Responses::new(OpenAIConfig::new("sk-test"), "gpt-5.4")
}

/// A reply whose output is `output`, folded as a unary body on `wire`, so
/// the turn's origin is the model the next request goes to.
fn folded_on(wire: &Responses, output: Value) -> CompletionResponse {
    decode_reply(
        wire,
        &CompletionRequest::new("say hi"),
        Mode::Unary,
        [WireFrame::Text(
            reply(json!({"id": "resp_1"}), output).to_string(),
        )],
    )
    .expect("the reply folds")
}

/// The `input` the request `next` encodes to on `wire`.
fn sent(wire: &Responses, next: CompletionRequest) -> Vec<Value> {
    crate::responses_api::input_items(wire, next).expect("encodes")
}

/// `next` with a tool declared, so calls replay as items.
fn declaring_tools(mut next: CompletionRequest) -> CompletionRequest {
    next.additional_params = Some(json!({
        "tools": [crate::responses_api::fidelity_fixtures::namespace()]
    }));
    next
}

/// The request that follows `turn` with its assistant message's blocks
/// replaced by `content`, the turn's origin and stop kept.
fn following_with(turn: &CompletionResponse, content: Vec<AssistantContent>) -> CompletionRequest {
    let mut next = following(turn);
    let Some(Message::Assistant(mut message)) = turn.message() else {
        panic!("the turn folds to an assistant message");
    };
    message.content = content;
    let slot = next
        .chat_history
        .iter_mut()
        .find(|message| matches!(***message, Message::Assistant(_)))
        .expect("the history holds the turn");
    *slot = Arc::new(Message::Assistant(message));
    next
}

fn user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

fn reasoning(id: &str, text: &str, ciphertext: &str) -> Value {
    json!({
        "type": "reasoning", "id": id,
        "summary": [{"type": "summary_text", "text": text}],
        "encrypted_content": ciphertext
    })
}

fn message(text: &str) -> Value {
    json!({
        "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "annotations": [], "text": text}]
    })
}

fn call() -> Value {
    json!({
        "type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search",
        "arguments": "{\"q\":\"rig\"}", "status": "completed"
    })
}

/// On the Codex contract, a turn that holds one reasoning item alone
/// replays as that item, whole, between the two user messages.
#[test]
fn codex_replays_a_reasoning_only_turn_as_its_item() {
    let wire = chatgpt_wire();
    let turn = folded_on(&wire, json!([reasoning("rs_1", "step 1", "enc_blob")]));
    let [AssistantContent::Reasoning(_)] = turn.choice.as_slice() else {
        panic!("expected one reasoning block, got {:?}", turn.choice);
    };

    assert_eq!(
        sent(&wire, following(&turn)),
        [
            user("say hi"),
            json!({
                "type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "step 1"}],
                "encrypted_content": "enc_blob"
            }),
            user("next"),
        ]
    );
}

/// On OpenAI's own contract the same turn keeps upstream Rig's rule: the
/// reasoning item has no item after it, so it is not sent, and the turn,
/// left with nothing, is not either.
#[test]
fn openai_does_not_replay_a_reasoning_only_turn() {
    let wire = openai_wire();
    let turn = folded_on(&wire, json!([reasoning("rs_1", "step 1", "enc_blob")]));

    assert_eq!(
        sent(&wire, following(&turn)),
        [user("say hi"), user("next")]
    );
}

/// On the Codex contract, two reasoning items alone in their turn both
/// go back, in order.
#[test]
fn codex_replays_two_reasoning_items_alone_in_order() {
    let wire = chatgpt_wire();
    let turn = folded_on(
        &wire,
        json!([
            reasoning("rs_1", "step 1", "enc_1"),
            reasoning("rs_2", "step 2", "enc_2")
        ]),
    );

    assert_eq!(
        sent(&wire, following(&turn)),
        [
            user("say hi"),
            json!({
                "type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "step 1"}],
                "encrypted_content": "enc_1"
            }),
            json!({
                "type": "reasoning", "id": "rs_2",
                "summary": [{"type": "summary_text", "text": "step 2"}],
                "encrypted_content": "enc_2"
            }),
            user("next"),
        ]
    );
}

/// On the Codex contract, a message then a reasoning item go back as the
/// message, then the reasoning.
#[test]
fn codex_replays_text_then_reasoning() {
    let wire = chatgpt_wire();
    let turn = folded_on(
        &wire,
        json!([message("hello"), reasoning("rs_1", "step 1", "enc_blob")]),
    );

    assert_eq!(
        sent(&wire, following(&turn)),
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": "hello"}]
            }),
            json!({
                "type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "step 1"}],
                "encrypted_content": "enc_blob"
            }),
            user("next"),
        ]
    );
}

/// On the Codex contract, reasoning before a function call in a request
/// that declares no tools: the call and its answer go back as text, so the
/// reasoning, whose next block exists but is not replayed as its item, is
/// not sent.
#[test]
fn codex_does_not_send_reasoning_before_a_call_narrated_as_text() {
    let wire = chatgpt_wire();
    let turn = folded_on(
        &wire,
        json!([reasoning("rs_1", "step 1", "enc_blob"), call()]),
    );

    assert_eq!(
        sent(&wire, following(&turn)),
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "[called tool search with {\"q\":\"rig\"}]"}]
            }),
            user("[tool search result] ok"),
            user("next"),
        ]
    );
}

/// On the Codex contract, reasoning before a text block replaced by one
/// with no provider item keeps upstream Rig's rule: the text is rebuilt
/// under an id Rig makes up, and the reasoning is not sent.
#[test]
fn codex_does_not_send_reasoning_before_a_block_with_no_item() {
    let wire = chatgpt_wire();
    let turn = folded_on(
        &wire,
        json!([reasoning("rs_1", "step 1", "enc_blob"), message("hello")]),
    );
    let [
        reasoning_block @ AssistantContent::Reasoning(_),
        AssistantContent::Text(_),
    ] = turn.choice.as_slice()
    else {
        panic!("expected reasoning then text, got {:?}", turn.choice);
    };
    let next = following_with(
        &turn,
        vec![
            reasoning_block.clone(),
            AssistantContent::Text(Text::new("hello")),
        ],
    );

    assert_eq!(
        sent(&wire, next),
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "hello"}]
            }),
            user("next"),
        ]
    );
}

/// On the Codex contract, reasoning then a function call with its
/// answer, on a request that declares tools, is unchanged from upstream
/// Rig: the reasoning, the call and its output.
#[test]
fn codex_replays_reasoning_a_call_and_its_output() {
    let wire = chatgpt_wire();
    let turn = folded_on(
        &wire,
        json!([reasoning("rs_1", "step 1", "enc_blob"), call()]),
    );

    assert_eq!(
        sent(&wire, declaring_tools(following(&turn))),
        [
            user("say hi"),
            json!({
                "type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "step 1"}],
                "encrypted_content": "enc_blob"
            }),
            json!({
                "type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search",
                "arguments": "{\"q\":\"rig\"}", "status": "completed"
            }),
            json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
            user("next"),
        ]
    );
}

/// A reasoning item followed by a message whose text is empty: the
/// folded text block holds its item, so it is not blank, goes back as that
/// item, and the reasoning before it goes too, on both contracts.
#[test]
fn reasoning_before_an_empty_message_item_goes_with_it_on_both_contracts() {
    for wire in [chatgpt_wire(), openai_wire()] {
        let turn = folded_on(
            &wire,
            json!([reasoning("rs_1", "step 1", "enc_blob"), message("")]),
        );

        assert_eq!(
            sent(&wire, following(&turn)),
            [
                user("say hi"),
                json!({
                    "type": "reasoning", "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "step 1"}],
                    "encrypted_content": "enc_blob"
                }),
                json!({
                    "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "annotations": [], "text": ""}]
                }),
                user("next"),
            ],
            "{}",
            wire.provider.dialect.name
        );
    }
}

/// A reasoning item followed by a blank text block with no item: the
/// blank block is dropped, but the reasoning is not the last block of its
/// turn as written, so on both contracts it keeps upstream Rig's rule and is
/// not sent, and the turn, left with nothing, is not either. Judged after
/// the blank block is removed, the reasoning would end the turn and go back
/// on the Codex contract.
#[test]
fn reasoning_before_a_blank_block_with_no_item_is_not_sent_on_either_contract() {
    for wire in [chatgpt_wire(), openai_wire()] {
        let turn = folded_on(&wire, json!([reasoning("rs_1", "step 1", "enc_blob")]));
        let [reasoning_block @ AssistantContent::Reasoning(_)] = turn.choice.as_slice() else {
            panic!("expected one reasoning block, got {:?}", turn.choice);
        };
        let next = following_with(
            &turn,
            vec![
                reasoning_block.clone(),
                AssistantContent::Text(Text::new("")),
            ],
        );

        assert_eq!(
            sent(&wire, next),
            [user("say hi"), user("next")],
            "{}",
            wire.provider.dialect.name
        );
    }
}
