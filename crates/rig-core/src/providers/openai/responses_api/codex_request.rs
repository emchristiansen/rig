//! Standard Codex instruction input and routing-first serialization.

use serde::ser::{Error, SerializeMap};
use serde::{Serialize, Serializer};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::completion::options::FinalBody;
use crate::error::EncodeError;

use super::codex_identity::CodexIdentity;

/// Move the effective instructions after all option layers have merged.
pub(crate) fn shape_body(
    body: &mut Map<String, Value>,
    identity: Option<&CodexIdentity>,
) -> Result<(), EncodeError> {
    let text = match body.shift_remove("instructions") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::String(text)) if text.is_empty() => return Ok(()),
        Some(Value::String(text)) => text,
        Some(_) => {
            return Err(EncodeError::request(
                "Standard Codex instructions must be a string or null",
            ));
        }
    };
    let input = body.get_mut("input").ok_or_else(|| {
        EncodeError::request("Standard Codex instructions require an input array or string")
    })?;
    match input {
        Value::Array(items) => items.insert(0, base_instructions(&text, identity)),
        Value::String(user) => {
            *input = json!([
                base_instructions(&text, identity),
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user}]},
            ]);
        }
        _ => {
            return Err(EncodeError::request(
                "Standard Codex instructions require an input array or string",
            ));
        }
    }
    Ok(())
}

fn base_instructions(text: &str, identity: Option<&CodexIdentity>) -> Value {
    let mut item = json!({
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
        "internal_chat_message_metadata_passthrough": {
            "content_item_kinds": ["model.base_instructions"],
        },
    });
    if let Some(identity) = identity {
        let namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, identity.thread_id().as_bytes());
        let id = Uuid::new_v5(&namespace, text.as_bytes());
        item["id"] = Value::String(format!("msg_{id}"));
    }
    item
}

/// Serialize existing routing fields ahead of the potentially large input.
pub(super) struct RoutingFirst<'a>(pub(super) &'a FinalBody);

impl Serialize for RoutingFirst<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let body = self
            .0
            .deserialize::<Map<String, Value>>()
            .map_err(S::Error::custom)?;
        let routing = ["model", "stream", "service_tier"];
        let mut output = serializer.serialize_map(Some(body.len()))?;
        for key in routing {
            if let Some(value) = body.get(key) {
                output.serialize_entry(key, value)?;
            }
        }
        for (key, value) in &body {
            if !routing.contains(&key.as_str()) {
                output.serialize_entry(key, value)?;
            }
        }
        output.end()
    }
}
