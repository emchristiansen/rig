use super::*;
use crate::responses_api::wire::CodexRequestError;
use serde_json::json;

fn derived_identity() -> CodexIdentity {
    CodexIdentity::from_ids("session-derived", "thread-derived").expect("header-safe ids")
}

fn object(value: serde_json::Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("expected an object, got {other}"),
    }
}

/// Caller-derived ids are sent verbatim, so an id no header can carry, or an
/// empty one, is refused by name, both when built and when deserialized.
#[test]
fn a_codex_identity_refuses_ids_no_header_can_carry() {
    let empty = CodexIdentity::from_ids("", "thread").expect_err("empty");
    assert_eq!(empty.field, "session_id");
    let broken = CodexIdentity::from_ids("session", "line\nbreak").expect_err("not header-safe");
    assert_eq!(broken.field, "thread_id");

    let identity = derived_identity();
    let json = serde_json::to_value(&identity).expect("serializes");
    assert_eq!(
        json,
        serde_json::json!({"session_id": "session-derived", "thread_id": "thread-derived"})
    );
    assert_eq!(
        serde_json::from_value::<CodexIdentity>(json).expect("deserializes"),
        identity
    );
    assert!(
        serde_json::from_value::<CodexIdentity>(
            serde_json::json!({"session_id": "", "thread_id": "t"})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<CodexIdentity>(
            serde_json::json!({"session_id": "s", "thread_id": "line\nbreak"})
        )
        .is_err(),
        "an id no header can carry is refused on the way in, too"
    );
}

/// The identity names a conversation with three headers: the dashed session
/// and thread ids, and the thread id again as the correlation id.
#[test]
fn the_identity_headers_are_the_dashed_pair_and_the_correlation_id() {
    let identity = derived_identity();
    assert_eq!(
        identity.headers(),
        [
            (SESSION_ID_HEADER, "session-derived"),
            (THREAD_ID_HEADER, "thread-derived"),
            (X_CLIENT_REQUEST_ID_HEADER, "thread-derived"),
        ]
    );
    assert_eq!(identity.prompt_cache_key(), identity.thread_id());
}

/// Stamping fills the cache key and both metadata ids into a body that has
/// none.
#[test]
fn stamping_names_the_conversation_in_a_body_that_does_not() {
    let mut body = object(json!({"model": "gpt-5.4"}));
    derived_identity().stamp(&mut body).expect("stamps");
    assert_eq!(
        Value::Object(body),
        json!({
            "model": "gpt-5.4",
            "prompt_cache_key": "thread-derived",
            "client_metadata": {"session_id": "session-derived", "thread_id": "thread-derived"},
        })
    );
}

/// A request's own cache key and metadata keys are kept; the identity fills
/// only what is missing.
#[test]
fn stamping_keeps_what_the_request_already_says() {
    let mut body = object(json!({
        "prompt_cache_key": "caller-key",
        "client_metadata": {"session_id": "caller-session", "turn_id": "turn-1"},
    }));
    derived_identity().stamp(&mut body).expect("stamps");
    assert_eq!(
        Value::Object(body),
        json!({
            "prompt_cache_key": "caller-key",
            "client_metadata": {
                "session_id": "caller-session",
                "thread_id": "thread-derived",
                "turn_id": "turn-1",
            },
        })
    );
}

/// Metadata that is not an object cannot carry the ids, and is refused with
/// the value found rather than replaced.
#[test]
fn stamping_refuses_metadata_that_is_not_an_object() {
    let mut body = object(json!({"client_metadata": "not a map"}));
    let error = derived_identity()
        .stamp(&mut body)
        .expect_err("a string cannot carry the ids");
    let CodexRequestError::ClientMetadata { found } = error else {
        panic!("expected the metadata refusal, got {error:?}");
    };
    assert_eq!(found, json!("not a map"));
    assert_eq!(
        body["client_metadata"], "not a map",
        "the body is untouched"
    );
}

/// A session without an identity is refused by a named error.
#[test]
fn a_missing_identity_is_a_request_refusal() {
    let error = ProviderError::from(MissingCodexIdentity);
    let ProviderError::Request(reason) = &error else {
        panic!("expected a request refusal, got {error:?}");
    };
    assert_eq!(
        reason.downcast_ref::<MissingCodexIdentity>(),
        Some(&MissingCodexIdentity)
    );
    assert!(!error.is_retryable());
}
