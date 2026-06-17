//! Typed per-request ChatGPT/Codex cache identity.
//!
//! Callers attach a [`ChatGptCacheIdentity`] to a
//! [`crate::completion::CompletionRequest`] via [`ChatGptRequestExt::with_cache_identity`].
//! Identity is carried inside `additional_params` under a private sentinel key so
//! it survives request cloning and builder merge semantics. The ChatGPT provider
//! consumes and strips the sentinel before serializing the outbound body — the
//! key is never sent on the wire.
//!
//! When identity is supplied, the provider stamps:
//! - headers `session-id`, `thread-id`, `x-client-request-id` (dashed),
//!   plus the existing `session_id` (underscore) header replaced with the
//!   stable identity value;
//! - body `prompt_cache_key = identity.prompt_cache_key.unwrap_or(thread_id)`;
//! - body `client_metadata` with protected core keys `session_id` /
//!   `thread_id` and any `extra_client_metadata`.
//!
//! When identity is absent, current provider behavior is preserved.

use crate::completion;
use crate::json_utils;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Private sentinel key for internal carriage inside `additional_params`.
/// Stripped before the outbound body is serialized.
pub(crate) const CHATGPT_CACHE_IDENTITY_KEY: &str = "__rig_chatgpt_cache_identity__";

pub(crate) const PROMPT_CACHE_KEY_FIELD: &str = "prompt_cache_key";
pub(crate) const CLIENT_METADATA_FIELD: &str = "client_metadata";
pub(crate) const SESSION_ID_KEY: &str = "session_id";
pub(crate) const THREAD_ID_KEY: &str = "thread_id";

/// Typed cache identity for a ChatGPT/Codex completion request.
///
/// `session_id` and `thread_id` are the two stable identifiers Codex's prompt
/// cache keys against. `prompt_cache_key` defaults to `thread_id` at the wire
/// boundary unless overridden here. `extra_client_metadata` is merged into the
/// outbound `client_metadata` object; protected core keys (`session_id`,
/// `thread_id`) take precedence and cannot be silently overridden.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatGptCacheIdentity {
    pub session_id: String,
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_client_metadata: BTreeMap<String, String>,
}

/// Extension trait for attaching a typed cache identity to a completion request.
pub trait ChatGptRequestExt {
    /// Attach a typed cache identity to this request. Overwrites any previously
    /// attached identity. The identity is carried internally and consumed by the
    /// ChatGPT provider; it is never exposed on the outbound wire under this
    /// sentinel key.
    fn with_cache_identity(self, identity: ChatGptCacheIdentity) -> Self;
}

impl ChatGptRequestExt for completion::CompletionRequest {
    fn with_cache_identity(mut self, identity: ChatGptCacheIdentity) -> Self {
        let identity_value = serde_json::to_value(&identity)
            .expect("ChatGptCacheIdentity serialization is infallible");
        let mut overlay = serde_json::Map::new();
        overlay.insert(
            CHATGPT_CACHE_IDENTITY_KEY.to_string(),
            identity_value,
        );
        let overlay = Value::Object(overlay);
        self.additional_params = Some(match self.additional_params.take() {
            Some(existing) => json_utils::merge(existing, overlay),
            None => overlay,
        });
        self
    }
}

/// Extract and remove the typed cache identity from `additional_params`, if any.
///
/// Used by the ChatGPT provider on the outbound path to consume the sentinel
/// before serializing the wire body.
pub(crate) fn extract_chatgpt_cache_identity(
    params: &mut Option<Value>,
) -> Option<ChatGptCacheIdentity> {
    let Some(Value::Object(map)) = params.as_mut() else {
        return None;
    };
    let raw = map.remove(CHATGPT_CACHE_IDENTITY_KEY)?;
    let identity = serde_json::from_value(raw).ok()?;
    if map.is_empty() {
        *params = None;
    }
    Some(identity)
}

/// Stamp the wire body with `prompt_cache_key` and `client_metadata` derived
/// from the typed cache identity. Mutates the body in place; expects an
/// `Object` root (the OpenAI Responses request body). Protected core keys
/// (`session_id`, `thread_id`) cannot be silently overridden by
/// `extra_client_metadata`.
pub(crate) fn stamp_body_with_cache_identity(
    body: &mut Value,
    identity: &ChatGptCacheIdentity,
) {
    let Some(map) = body.as_object_mut() else {
        return;
    };
    let prompt_cache_key = identity
        .prompt_cache_key
        .clone()
        .unwrap_or_else(|| identity.thread_id.clone());
    map.insert(
        PROMPT_CACHE_KEY_FIELD.to_string(),
        Value::String(prompt_cache_key),
    );

    let mut metadata = serde_json::Map::new();
    for (key, value) in &identity.extra_client_metadata {
        if key == SESSION_ID_KEY || key == THREAD_ID_KEY {
            continue;
        }
        metadata.insert(key.clone(), Value::String(value.clone()));
    }
    metadata.insert(
        SESSION_ID_KEY.to_string(),
        Value::String(identity.session_id.clone()),
    );
    metadata.insert(
        THREAD_ID_KEY.to_string(),
        Value::String(identity.thread_id.clone()),
    );
    map.insert(CLIENT_METADATA_FIELD.to_string(), Value::Object(metadata));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::completion::CompletionRequest;
    use crate::message::Message;

    fn empty_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::user("hi")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    fn sample_identity() -> ChatGptCacheIdentity {
        ChatGptCacheIdentity {
            session_id: "sess-123".to_string(),
            thread_id: "thr-456".to_string(),
            prompt_cache_key: None,
            extra_client_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn with_cache_identity_writes_sentinel() {
        let req = empty_request().with_cache_identity(sample_identity());
        let params = req.additional_params.expect("sentinel must be present");
        let obj = params.as_object().expect("additional_params must be an object");
        assert!(obj.contains_key(CHATGPT_CACHE_IDENTITY_KEY));
    }

    #[test]
    fn extract_round_trips() {
        let req = empty_request().with_cache_identity(sample_identity());
        let mut params = req.additional_params;
        let extracted = extract_chatgpt_cache_identity(&mut params)
            .expect("identity must round-trip out");
        assert_eq!(extracted.session_id, "sess-123");
        assert_eq!(extracted.thread_id, "thr-456");
        assert!(params.is_none(), "sentinel-only map must collapse to None");
    }

    #[test]
    fn stamp_body_writes_prompt_cache_key_and_client_metadata() {
        let mut body = serde_json::json!({"model": "gpt-5"});
        stamp_body_with_cache_identity(&mut body, &sample_identity());
        let obj = body.as_object().expect("object root");
        assert_eq!(
            obj.get(PROMPT_CACHE_KEY_FIELD),
            Some(&Value::String("thr-456".to_string())),
            "prompt_cache_key defaults to thread_id"
        );
        let metadata = obj
            .get(CLIENT_METADATA_FIELD)
            .and_then(|v| v.as_object())
            .expect("client_metadata is an object");
        assert_eq!(
            metadata.get(SESSION_ID_KEY),
            Some(&Value::String("sess-123".to_string()))
        );
        assert_eq!(
            metadata.get(THREAD_ID_KEY),
            Some(&Value::String("thr-456".to_string()))
        );
    }

    #[test]
    fn stamp_body_uses_explicit_prompt_cache_key_override() {
        let identity = ChatGptCacheIdentity {
            prompt_cache_key: Some("override-key".to_string()),
            ..sample_identity()
        };
        let mut body = serde_json::json!({});
        stamp_body_with_cache_identity(&mut body, &identity);
        assert_eq!(
            body.get(PROMPT_CACHE_KEY_FIELD),
            Some(&Value::String("override-key".to_string()))
        );
    }

    #[test]
    fn stamp_body_protects_core_metadata_keys_from_extras() {
        let mut extras = BTreeMap::new();
        extras.insert(SESSION_ID_KEY.to_string(), "evil-session".to_string());
        extras.insert(THREAD_ID_KEY.to_string(), "evil-thread".to_string());
        extras.insert("custom".to_string(), "kept".to_string());
        let identity = ChatGptCacheIdentity {
            extra_client_metadata: extras,
            ..sample_identity()
        };
        let mut body = serde_json::json!({});
        stamp_body_with_cache_identity(&mut body, &identity);
        let metadata = body
            .get(CLIENT_METADATA_FIELD)
            .and_then(|v| v.as_object())
            .expect("client_metadata is an object");
        assert_eq!(
            metadata.get(SESSION_ID_KEY),
            Some(&Value::String("sess-123".to_string())),
            "core session_id must not be overridden by extras"
        );
        assert_eq!(
            metadata.get(THREAD_ID_KEY),
            Some(&Value::String("thr-456".to_string())),
            "core thread_id must not be overridden by extras"
        );
        assert_eq!(
            metadata.get("custom"),
            Some(&Value::String("kept".to_string())),
            "unrelated extras are preserved"
        );
    }

    #[test]
    fn extract_preserves_unrelated_keys() {
        let mut req = empty_request().with_cache_identity(sample_identity());
        let params_obj = req
            .additional_params
            .as_mut()
            .and_then(|v| v.as_object_mut())
            .expect("must be object");
        params_obj.insert("unrelated".to_string(), Value::from(42));
        let mut params = req.additional_params;
        let _ = extract_chatgpt_cache_identity(&mut params)
            .expect("identity must round-trip out");
        let remaining = params.expect("unrelated keys must remain");
        let remaining_obj = remaining.as_object().expect("must be object");
        assert!(!remaining_obj.contains_key(CHATGPT_CACHE_IDENTITY_KEY));
        assert_eq!(remaining_obj.get("unrelated"), Some(&Value::from(42)));
    }
}
