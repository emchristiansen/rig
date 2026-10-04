//! Pure validation and request stamping: no provider call or cassette is needed.
use super::*;
#[test]
fn caller_values_are_preserved_and_partial_identity_cannot_deserialize() {
    let identity =
        CallerIdentity::new("caller", "caller/1 (host)", Some("version-1".into())).unwrap();
    let request = identity
        .stamp(http::Request::get("https://example.invalid"))
        .body(())
        .unwrap();
    assert_eq!(request.headers()["originator"], "caller");
    assert_eq!(request.headers()["user-agent"], "caller/1 (host)");
    assert_eq!(request.headers()["version"], "version-1");
    assert!(serde_json::from_str::<CallerIdentity>(r#"{"originator":"caller"}"#).is_err());
    assert!(
        serde_json::from_str::<CallerIdentity>(r#"{"originator":"caller","user_agent":""}"#)
            .is_err()
    );
    assert!(CallerIdentity::new("caller", "user\nagent", None).is_err());
    assert!(CallerIdentity::new("caller", "agent", Some("".into())).is_err());
}
#[test]
fn identity_round_trip_validates_and_preserves_supplied_ids() {
    let identity = CodexIdentity::from_ids("chosen-session", "chosen-thread").unwrap();
    let encoded = serde_json::to_string(&identity).unwrap();
    assert_eq!(
        serde_json::from_str::<CodexIdentity>(&encoded).unwrap(),
        identity
    );
    assert!(
        serde_json::from_str::<CodexIdentity>(r#"{"session_id":"","thread_id":"thread"}"#).is_err()
    );
    assert!(CodexIdentity::from_ids("session", "thread\r\n").is_err());
}

/// Codex ff6aec96 protocol/src/thread_id.rs:30 and session/session.rs:894–914.
#[test]
fn generated_root_identity_uses_one_uuid_v7() {
    let generated = CodexIdentity::generate();
    let thread = uuid::Uuid::parse_str(generated.thread_id()).expect("a UUID");
    assert_eq!(thread.get_version_num(), 7);
    assert_eq!(thread.get_variant(), uuid::Variant::RFC4122);
    assert_eq!(generated.session_id(), generated.thread_id());
    assert_eq!(generated.thread_id(), thread.hyphenated().to_string());
    let encoded = serde_json::to_string(&generated).expect("serializes");
    assert_eq!(
        serde_json::from_str::<CodexIdentity>(&encoded).expect("valid identity"),
        generated
    );
}
