//! A turn a client sends back (the provider's output items, verbatim),
//! folded with Rig's own decoder as a unary reply body: what it becomes,
//! and what the next request replays for it.

use super::fold_contract_tests::{
    chatgpt_wire, folded, following, reply, streamed_turn, turn_items,
};
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::wire::{Mode, WireFrame, decode_reply};
use serde_json::{Value, json};

/// The client's six items as a unary reply body of the streamed turn's
/// response.
fn echoed_body() -> Value {
    reply(json!({"id": "resp_1"}), json!(turn_items()))
}

/// `body` folded as a unary reply on the wire the next request is sent on.
fn folded_on_the_codex_wire(body: Value) -> CompletionResponse {
    decode_reply(
        &chatgpt_wire(),
        &CompletionRequest::new("say hi"),
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
    )
    .expect("the echoed turn folds")
}

fn declaring_tools(mut next: CompletionRequest) -> CompletionRequest {
    next.additional_params = Some(json!({
        "tools": [crate::responses_api::fidelity_fixtures::namespace()]
    }));
    next
}

fn user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

/// The answers [`following`] gives the turn's three calls.
fn answers() -> [Value; 3] {
    [
        json!({"type": "custom_tool_call_output", "call_id": "call_c", "name": "apply_patch", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
        json!({"type": "function_call_output", "call_id": "call_m", "output": "ok", "status": "completed"}),
    ]
}

/// Folded on the wire it is replayed on (ChatGPT's Responses, the request's
/// model), an echoed turn replays, on a request that declares tools, as
/// exactly the six items the client sent, item for item.
#[test]
fn an_echoed_turn_folded_on_the_codex_wire_replays_as_the_clients_items() {
    let turn = folded_on_the_codex_wire(echoed_body());
    let input =
        crate::responses_api::input_items(&chatgpt_wire(), declaring_tools(following(&turn)))
            .expect("encodes");

    assert_eq!(input[0], user("say hi"));
    assert_eq!(
        input[1..7],
        turn_items(),
        "the client's items, item for item"
    );
    assert_eq!(input[7..10], answers());
    assert_eq!(input[10..], [user("next")]);
}

/// The echoed turn folded as a unary body of the same response is the
/// message the streamed fold gives for the same items: same blocks, same
/// items, same origin, same stop.
#[test]
fn an_echoed_turn_folds_to_the_streamed_turns_message() {
    let echoed = folded_on_the_codex_wire(echoed_body());
    assert_eq!(echoed.message(), streamed_turn().message());
    assert_eq!(echoed.choice, streamed_turn().choice);
}

/// The message names the response the body states: an echo folded in a
/// body of another response id has the same blocks, and a message that
/// differs from the streamed one by that id alone.
#[test]
fn an_echoed_turn_in_another_responses_body_differs_by_its_response_id_alone() {
    use rig_core::message::Message;
    let echoed = folded_on_the_codex_wire(reply(json!({}), json!(turn_items())));
    assert_eq!(echoed.choice, streamed_turn().choice);
    let (Some(Message::Assistant(echoed)), Some(Message::Assistant(mut streamed))) =
        (echoed.message(), streamed_turn().message())
    else {
        panic!("both fold to an assistant message");
    };
    assert_ne!(
        Message::Assistant(echoed.clone()),
        Message::Assistant(streamed.clone())
    );
    let origin = streamed.origin.as_mut().expect("an origin");
    assert_eq!(origin.response_id.as_deref(), Some("resp_1"));
    origin.response_id = Some("resp_123".to_owned());
    assert_eq!(echoed, streamed);
}

/// Folded on another wire (OpenAI's own Responses, as `folded` does), the
/// echoed turn's origin names the `openai` provider, so the Codex request
/// treats it as another provider's turn and replays it rebuilt: the
/// reasoning becomes assistant text and loses its ciphertext and id, the
/// search item is dropped, the refusal part is merged into the text under
/// an id Rig makes up, the custom call becomes a function call of
/// `{"input": ...}`, the malformed arguments come back as a well-formed
/// object, and every item id, namespace and status is gone.
#[test]
fn an_echoed_turn_folded_on_another_wire_replays_rebuilt_as_another_providers() {
    let turn = folded(echoed_body());
    let origin = turn
        .message()
        .and_then(|message| match message {
            rig_core::message::Message::Assistant(message) => message.origin,
            _ => None,
        })
        .expect("the fold names its origin");
    assert_eq!(origin.provider, "openai");
    assert_eq!(origin.api.as_str(), "openai.responses");

    let input =
        crate::responses_api::input_items(&chatgpt_wire(), declaring_tools(following(&turn)))
            .expect("encodes");
    assert_eq!(
        input,
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "step 1"}]
            }),
            json!({
                "type": "message", "id": "msg_rig_1_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "hellono"}]
            }),
            json!({
                "type": "function_call", "call_id": "call_c", "name": "apply_patch",
                "arguments": "{\"input\":\"{\\\"looks\\\":\\\"like json\\\"}\"}"
            }),
            json!({"type": "function_call", "call_id": "call_f", "name": "search", "arguments": "{\"q\":\"rig\"}"}),
            json!({"type": "function_call", "call_id": "call_m", "name": "add", "arguments": "{\"x\":481}"}),
            json!({"type": "function_call_output", "call_id": "call_c", "output": "ok", "status": "completed"}),
            json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
            json!({"type": "function_call_output", "call_id": "call_m", "output": "ok", "status": "completed"}),
            user("next"),
        ]
    );
}
