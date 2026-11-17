//! Requests whose bytes on the Codex contract are pinned, for the HTTP and
//! WebSocket tests alike.

use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{Message, ToolName};
use serde_json::value::RawValue;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A Responses `functions` namespace as a caller declares it: empty
/// descriptions and an explicit `strict: false`.
pub(crate) fn namespace() -> Value {
    json!({
        "type": "namespace",
        "name": "functions",
        "description": "",
        "tools": [{
            "type": "function",
            "name": "exec_command",
            "description": "",
            "strict": false,
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
        }]
    })
}

/// A function as a caller declares it, with members Rig's tool type does
/// not model.
pub(crate) fn declared_function() -> Value {
    json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "defer_loading": false,
        "x_vendor_hint": null,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    })
}

/// A hosted tool as a caller declares it.
pub(crate) fn declared_hosted() -> Value {
    json!({"type": "web_search", "search_context_size": "low"})
}

/// A typed tool, which Rig writes ahead of the declared ones.
pub(crate) fn lookup_tool() -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new("lookup").expect("a tool name"),
        "Look something up.",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )
}

/// One user turn.
pub(crate) fn prompt() -> CompletionRequest {
    CompletionRequest::new(Message::user("say hi"))
}

/// One user turn with the typed `lookup` tool and `declared` as its
/// declared tools.
pub(crate) fn turn_declaring(declared: Value) -> CompletionRequest {
    let mut request = prompt();
    request.tools = vec![Arc::new(lookup_tool())];
    request.additional_params = Some(json!({ "tools": declared }));
    request
}

/// The request Muninn's Codex lane sends: instructions, one user turn,
/// reasoning with its ciphertext, and the conversation's cache key and
/// metadata stated by the caller.
pub(crate) fn muninn_shaped_request() -> CompletionRequest {
    let mut request = prompt();
    request.chat_history = vec![
        Arc::new(Message::system("Be brief.")),
        Arc::new(Message::user("say hi")),
    ];
    request.additional_params = Some(json!({
        "prompt_cache_key": "thread-7",
        "client_metadata": {"session_id": "session-7", "thread_id": "thread-7"},
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "store": false,
    }));
    request
}

/// Whether this build's `serde_json` keeps an object's keys in insertion
/// order (`preserve_order`) or sorts them.
pub(crate) fn preserves_order() -> bool {
    serde_json::to_string(&json!({ "b": 0, "a": 0 })).expect("serializes") == r#"{"b":0,"a":0}"#
}

/// The top-level fields of the JSON object `body`, each as the bytes it was
/// written with.
pub(crate) fn raw_fields(body: &str) -> BTreeMap<String, Box<RawValue>> {
    serde_json::from_str(body).expect("the body is a JSON object")
}

/// The bytes `value` is sent as: its serialization in this build.
pub(crate) fn written(value: &Value) -> String {
    serde_json::to_string(value).expect("serializes")
}

/// The whole body of the request Muninn's Codex lane sends, in a build whose
/// `serde_json` sorts keys (Muninn's): every field is the caller's, `input`
/// is last, and nothing else is stated.
pub(crate) const MUNINN_SHAPED_BODY_SORTED: &str = concat!(
    r#"{"client_metadata":{"session_id":"session-7","thread_id":"thread-7"},"#,
    r#""include":["reasoning.encrypted_content"],"instructions":"Be brief.","#,
    r#""model":"gpt-5.4","prompt_cache_key":"thread-7","reasoning":{"effort":"high"},"#,
    r#""store":false,"stream":true,"#,
    r#""input":[{"content":[{"text":"say hi","type":"input_text"}],"role":"user","type":"message"}]}"#,
);
