//! Rig's serde form of a message and of a whole request is a fixed point:
//! serialized, parsed and serialized again it is the same bytes and the
//! same value, and the re-read request replays to the same input items.
//! Each case states, too, what the request replays.

use super::fold_contract_tests::{chatgpt_wire, following, streamed_turn, turn_items};
use crate::responses_api::fidelity_fixtures::{
    answering, function_call_item, images_rig_sends_in_a_result,
};
use rig_core::completion::{AssistantContent, CompletionRequest};
use rig_core::message::{
    AssistantMessage, CallId, Document, DocumentData, DocumentMediaType, DocumentSourceKind,
    Message, Opaque, Origin, ProviderCallId, ToolName, ToolResultContent, UserContent,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::fmt::Debug;
use std::sync::Arc;

/// `value` serialized, parsed and serialized again gives the same bytes,
/// and the parsed value equals `value`.
fn assert_fixed_point<T>(value: &T, what: &str)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let first = serde_json::to_string(value).expect("serializes");
    let read: T = serde_json::from_str(&first)
        .unwrap_or_else(|error| panic!("{what}: does not parse back: {error}: {first}"));
    let second = serde_json::to_string(&read).expect("serializes again");
    assert_eq!(first, second, "{what}: the bytes");
    assert_eq!(&read, value, "{what}: the value");
}

/// The input items `request` replays as on the Codex wire.
fn replay(request: CompletionRequest) -> Vec<Value> {
    crate::responses_api::input_items(&chatgpt_wire(), request).expect("encodes")
}

/// Every message of `request`'s history and the whole request are fixed
/// points, and the request re-read whole, or rebuilt from its history read
/// back one message at a time (as a store holds it), replays to the same
/// input items as `request`. Returns those items.
fn assert_request_fixed_point(request: &CompletionRequest, case: &str) -> Vec<Value> {
    for (index, message) in request.chat_history.iter().enumerate() {
        assert_fixed_point(
            message.as_ref(),
            &format!("{case}: history message {index}"),
        );
    }
    assert_fixed_point(request, &format!("{case}: the request"));

    let reread: CompletionRequest =
        serde_json::from_str(&serde_json::to_string(request).expect("serializes")).expect("parses");
    let mut rebuilt = request.clone();
    rebuilt.chat_history = request
        .chat_history
        .iter()
        .map(|message| {
            let stored = serde_json::to_string(message.as_ref()).expect("serializes");
            Arc::new(serde_json::from_str::<Message>(&stored).expect("parses"))
        })
        .collect();

    let original = replay(request.clone());
    assert_eq!(
        replay(reread),
        original,
        "{case}: the re-read request's replay"
    );
    assert_eq!(
        replay(rebuilt),
        original,
        "{case}: the re-read history's replay"
    );
    original
}

fn declaring_tools(mut request: CompletionRequest) -> CompletionRequest {
    request.additional_params = Some(json!({
        "tools": [crate::responses_api::fidelity_fixtures::namespace()]
    }));
    request
}

/// A request whose history is `history`, declaring tools.
fn with_history(history: Vec<Message>) -> CompletionRequest {
    let mut request = CompletionRequest::new("say hi");
    request.chat_history = history.into_iter().map(Arc::new).collect();
    declaring_tools(request)
}

fn user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

fn user_message(content: Vec<UserContent>) -> Message {
    Message::User { content }
}

// ── the streamed turn (F1) ──────────────────────────────────────────────

/// The streamed turn's message, every message of the request that follows
/// it, and that whole request are fixed points, and the re-read request
/// replays the provider's own items.
#[test]
fn a_streamed_turn_and_the_request_after_it_are_fixed_points() {
    let turn = streamed_turn();
    let message = turn.message().expect("the turn's message");
    assert_fixed_point(&message, "the streamed turn's message");

    let replayed = assert_request_fixed_point(&declaring_tools(following(&turn)), "F1");
    let mut expected = vec![user("say hi")];
    expected.extend(turn_items());
    expected.extend([
        json!({"type": "custom_tool_call_output", "call_id": "call_c", "name": "apply_patch", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
        json!({"type": "function_call_output", "call_id": "call_m", "output": "ok", "status": "completed"}),
        user("next"),
    ]);
    assert_eq!(replayed, expected);
}

// ── user images and documents ───────────────────────────────────────────

/// A user message of an image by URL and text is a fixed point. It replays
/// as two user messages, one per part: the image (detail `auto` stated)
/// and the text.
#[test]
fn a_user_image_by_url_is_a_fixed_point() {
    let [url, _] = images_rig_sends_in_a_result();
    let request = with_history(vec![user_message(vec![
        UserContent::Image(url),
        UserContent::text("look"),
    ])]);
    assert_eq!(
        assert_request_fixed_point(&request, "a user image by URL"),
        [
            json!({"type": "message", "role": "user", "content": [{"type": "input_image", "detail": "auto", "image_url": "https://example.test/diff.png"}]}),
            user("look"),
        ]
    );
}

/// A user message of a typed base64 image and text is a fixed point. It
/// replays as two user messages, the image as a data URL.
#[test]
fn a_user_image_in_typed_base64_is_a_fixed_point() {
    let [_, base64] = images_rig_sends_in_a_result();
    let request = with_history(vec![user_message(vec![
        UserContent::Image(base64),
        UserContent::text("look"),
    ])]);
    assert_eq!(
        assert_request_fixed_point(&request, "a user image in base64"),
        [
            json!({"type": "message", "role": "user", "content": [{"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,aGk="}]}),
            user("look"),
        ]
    );
}

/// A user message of a text document and text is a fixed point. The
/// document replays as plain input text, its media type not stated, in a
/// user message of its own.
#[test]
fn a_user_text_document_is_a_fixed_point() {
    let request = with_history(vec![user_message(vec![
        UserContent::Document(Document {
            data: DocumentData::Text("doc text".to_owned()),
            media_type: Some(DocumentMediaType::TXT),
            additional_params: None,
        }),
        UserContent::text("read"),
    ])]);
    assert_eq!(
        assert_request_fixed_point(&request, "a user text document"),
        [user("doc text"), user("read")]
    );
}

/// A user message of a base64 PDF and text is a fixed point. The PDF
/// replays as an input file under a file name Rig makes up
/// (`document.pdf`), in a user message of its own.
#[test]
fn a_user_pdf_document_is_a_fixed_point() {
    let request = with_history(vec![user_message(vec![
        UserContent::Document(Document {
            data: DocumentData::File(DocumentSourceKind::Base64("JVBERi0=".to_owned())),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        }),
        UserContent::text("read"),
    ])]);
    assert_eq!(
        assert_request_fixed_point(&request, "a user PDF document"),
        [
            json!({"type": "message", "role": "user", "content": [{"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}]}),
            user("read"),
        ]
    );
}

// ── tool results ────────────────────────────────────────────────────────

/// A request answering a function call with text is a fixed point, and
/// replays the call's item and a text output.
#[test]
fn a_tool_result_of_text_is_a_fixed_point() {
    let request = answering(
        function_call_item(),
        vec![ToolResultContent::text("found it")],
    );
    assert_eq!(
        assert_request_fixed_point(&request, "a tool result of text"),
        [
            user("say hi"),
            function_call_item(),
            json!({"type": "function_call_output", "call_id": "call_f", "output": "found it", "status": "completed"}),
        ]
    );
}

/// A request answering a function call with each image Rig sends in a
/// result is a fixed point, and replays the image as the output's one
/// part (detail `auto` stated).
#[test]
fn a_tool_result_of_an_image_is_a_fixed_point() {
    let [url, base64] = images_rig_sends_in_a_result();
    for (case, image, sent) in [
        (
            "a tool result of an image by URL",
            url,
            "https://example.test/diff.png",
        ),
        (
            "a tool result of an image in base64",
            base64,
            "data:image/png;base64,aGk=",
        ),
    ] {
        let request = answering(function_call_item(), vec![ToolResultContent::Image(image)]);
        assert_eq!(
            assert_request_fixed_point(&request, case),
            [
                user("say hi"),
                function_call_item(),
                json!({
                    "type": "function_call_output", "call_id": "call_f", "status": "completed",
                    "output": [{"type": "input_image", "detail": "auto", "image_url": sent}]
                }),
            ],
            "{case}"
        );
    }
}

// ── another provider's turn ─────────────────────────────────────────────

/// A turn of Anthropic's origin, its blocks carrying Anthropic's items:
/// signed thinking, text, a tool use, and a server tool use kept opaque.
fn anthropic_turn() -> AssistantMessage {
    AssistantMessage::new(vec![
        AssistantContent::reasoning("plan")
            .with_native(json!({"type": "thinking", "thinking": "plan", "signature": "c2ln"})),
        AssistantContent::text("hi")
            .with_native(json!({"type": "text", "text": "hi", "citations": []})),
        AssistantContent::tool_call(
            "toolu_1",
            ToolName::new("search").expect("a tool name"),
            json!({"q": "rig"}),
        )
        .with_native(json!({"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "rig"}})),
        AssistantContent::Opaque(Opaque {
            item: json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "rig"}}),
            replay: true,
        }),
    ])
    .with_origin(Origin::new(
        "anthropic.messages",
        "anthropic",
        "claude-sonnet-4-5",
    ))
}

/// A history holding Anthropic's turn is a fixed point, its items and
/// origin kept. On the Codex wire the turn replays rebuilt from its
/// canonical fields: the thinking becomes assistant text with its
/// signature dropped, the text is rebuilt under an id Rig makes up, the
/// tool use becomes a function call with no item id, and the opaque server
/// tool use is dropped.
#[test]
fn another_providers_turn_is_a_fixed_point_and_replays_rebuilt() {
    let request = with_history(vec![
        Message::user("say hi"),
        Message::Assistant(anthropic_turn()),
        Message::tool_result(
            CallId::from(ProviderCallId::new("toolu_1").expect("a call id")),
            ToolName::new("search").expect("a tool name"),
            "ok",
        ),
        Message::user("next"),
    ]);
    let replayed = assert_request_fixed_point(&request, "an Anthropic turn");
    assert_eq!(
        replayed,
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "plan"}]
            }),
            json!({
                "type": "message", "id": "msg_rig_1_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "hi"}]
            }),
            json!({"type": "function_call", "call_id": "toolu_1", "name": "search", "arguments": "{\"q\":\"rig\"}"}),
            json!({"type": "function_call_output", "call_id": "toolu_1", "output": "ok", "status": "completed"}),
            user("next"),
        ]
    );
    assert!(
        !replayed
            .iter()
            .any(|item| item["type"] == "server_tool_use" || item.get("signature").is_some()),
        "the opaque item and the signature are not sent"
    );
}

/// The streamed Codex turn with its origin naming another provider of the
/// same API (`openai`, not `chatgpt`) is a fixed point. On the Codex wire
/// it replays rebuilt, as another provider's: reasoning as assistant text
/// without its ciphertext, the opaque search item dropped, the refusal part
/// merged into the text, the custom call as a function call of
/// `{"input": ...}`, the malformed arguments as a well-formed object, and
/// no item id, namespace or status the provider sent.
#[test]
fn a_codex_turn_of_another_origin_is_a_fixed_point_and_replays_rebuilt() {
    let turn = streamed_turn();
    let Some(Message::Assistant(mut other)) = turn.message() else {
        panic!("the turn's message");
    };
    other.origin.as_mut().expect("an origin").provider = "openai".to_owned();
    let mut history = vec![Message::user("say hi"), Message::Assistant(other)];
    history.extend(
        turn.tool_calls()
            .map(|call| Message::tool_result(call.id.clone(), call.function.name.clone(), "ok")),
    );
    history.push(Message::user("next"));

    let replayed = assert_request_fixed_point(&with_history(history), "a Codex turn of openai");
    assert_eq!(
        replayed,
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

/// A hand-built turn with no origin, holding text and an opaque Codex
/// search item, is a fixed point. It replays its text rebuilt under an id
/// Rig makes up and drops the opaque item, though the item is the Codex
/// wire's own kind and marked to replay.
#[test]
fn a_turn_of_no_origin_is_a_fixed_point_and_drops_its_opaque_item() {
    let handbuilt = AssistantMessage::new(vec![
        AssistantContent::text("hi"),
        AssistantContent::Opaque(Opaque {
            item: turn_items()[1].clone(),
            replay: true,
        }),
    ]);
    let request = with_history(vec![
        Message::user("say hi"),
        Message::Assistant(handbuilt),
        Message::user("next"),
    ]);
    assert_eq!(
        assert_request_fixed_point(&request, "a turn of no origin"),
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "hi"}]
            }),
            user("next"),
        ]
    );
}
