use super::*;

fn client_exec_identity(version: Option<&str>) -> CallerIdentity {
    CallerIdentity::new(
        "client_exec",
        "client_exec/1.2.3 (Linux 6.18; x86_64) xterm (client_exec; 1.2.3)",
        version.map(str::to_owned),
    )
    .expect("a valid identity")
}

/// An identity holds its values exactly as given.
#[test]
fn a_caller_identity_holds_its_values_exactly_as_given() {
    let identity = client_exec_identity(Some("1.2.3"));
    assert_eq!(identity.originator(), "client_exec");
    assert_eq!(
        identity.user_agent(),
        "client_exec/1.2.3 (Linux 6.18; x86_64) xterm (client_exec; 1.2.3)"
    );
    assert_eq!(identity.version(), Some("1.2.3"));
    assert_eq!(client_exec_identity(None).version(), None);
}

/// A value no request could carry cannot become a caller identity, by the
/// constructor or by deserialization, and the refusal names which value it
/// was.
#[test]
fn a_caller_identity_no_request_could_carry_cannot_be_built() {
    let refused = |originator: &str, user_agent: &str, version: Option<&str>| {
        CallerIdentity::new(originator, user_agent, version.map(str::to_owned))
            .expect_err("an invalid value is refused")
    };
    assert_eq!(
        refused("", "ua", None),
        InvalidCallerIdentity {
            field: "originator",
            value: String::new()
        }
    );
    assert_eq!(
        refused("client_exec", "line\nbreak", None),
        InvalidCallerIdentity {
            field: "user_agent",
            value: "line\nbreak".to_owned()
        }
    );
    assert_eq!(
        refused("client_exec", "ua", Some("")),
        InvalidCallerIdentity {
            field: "version",
            value: String::new()
        }
    );

    for json in [
        serde_json::json!({"originator": "", "user_agent": "ua"}),
        serde_json::json!({"originator": "o", "user_agent": "ua", "version": "1\n2"}),
    ] {
        assert!(
            serde_json::from_value::<CallerIdentity>(json.clone()).is_err(),
            "{json} must not decode"
        );
    }
}

/// A serialized identity without a version still decodes, and a version is
/// written only when there is one.
#[test]
fn a_caller_identity_serializes_its_version_only_when_it_has_one() {
    let old: CallerIdentity =
        serde_json::from_str(r#"{"originator":"rig","user_agent":"rig/0 (linux x86_64; rig)"}"#)
            .expect("an identity serialized before `version` existed decodes");
    assert_eq!(old.version(), None);
    assert_eq!(
        serde_json::to_value(&old).expect("serializes"),
        serde_json::json!({"originator": "rig", "user_agent": "rig/0 (linux x86_64; rig)"})
    );
    assert_eq!(
        serde_json::to_value(client_exec_identity(Some("1.2.3"))).expect("serializes")["version"],
        "1.2.3"
    );
    assert!(
        serde_json::from_value::<CallerIdentity>(
            serde_json::json!({"originator": "o", "user_agent": "ua", "extra": 1})
        )
        .is_err(),
        "unknown fields are still refused"
    );
}

/// The refusal for a missing identity is a request error under either error
/// type, and keeps its own type.
#[test]
fn a_missing_caller_identity_is_a_typed_request_refusal() {
    let missing = MissingCallerIdentity { dialect: "chatgpt" };
    for error in [
        ProviderError::from(missing.clone()),
        ProviderError::from(EncodeError::from(missing.clone())),
    ] {
        let ProviderError::Request(reason) = &error else {
            panic!("expected a request refusal, got {error:?}");
        };
        assert_eq!(
            reason.downcast_ref::<MissingCallerIdentity>(),
            Some(&missing)
        );
        assert!(!error.is_retryable());
    }
    assert!(missing.to_string().contains("`chatgpt`"), "{missing}");
}
