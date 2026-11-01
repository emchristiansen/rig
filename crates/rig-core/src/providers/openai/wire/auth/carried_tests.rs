//! Pure identity and credential-placement contracts; no provider response or environment mutation.

use super::super::Quirks;
use super::*;
use crate::wire::{Credential, CredentialSourceError};

const IDENTITY: Identity = Identity {
    originator_env: "PORT_TEST_ORIGINATOR",
    user_agent_env: "PORT_TEST_USER_AGENT",
    session_ids: false,
};

fn required_dialect() -> Dialect {
    Dialect::gateway("requires-exact", "https://example.invalid", "PORT_TEST_KEY").with_quirks(
        Quirks {
            identity: Some(IDENTITY),
            ..Quirks::openai()
        },
    )
}

/// Identity validation and persistence are local value contracts, not provider replies.
#[test]
fn exact_identity_round_trips_and_stamps_values_verbatim() {
    let identity = CallerIdentity::new("caller", "caller/1.2 (custom)", Some("1.2".into()))
        .expect("valid header values");
    let config = OpenAIConfig::new("key").with_caller_identity(identity.clone());
    let value = serde_json::to_value(&config).expect("config serializes");
    assert_eq!(value["identity"]["kind"], "exact");
    assert_eq!(value["identity"]["version"], "1.2");
    let restored: OpenAIConfig = serde_json::from_value(value).expect("config deserializes");
    assert_eq!(
        restored.identity,
        Some(ConfiguredIdentity::Exact(identity.clone()))
    );
    let request = restored
        .headers(http::Request::builder())
        .expect("exact identity sends")
        .body(())
        .expect("valid headers");
    assert_eq!(request.headers()["originator"], identity.originator());
    assert_eq!(
        request.headers()[http::header::USER_AGENT],
        identity.user_agent()
    );
    assert_eq!(
        request.headers()[VERSION_HEADER],
        identity.version().expect("version present")
    );
    let client_request = identity
        .stamp_client(http::Request::builder())
        .body(())
        .expect("client headers are valid");
    assert!(!client_request.headers().contains_key(VERSION_HEADER));
}

/// A serialized originator must remain originator-only, never become exact identity.
#[test]
fn configured_identity_preserves_its_kind_and_rejects_legacy_untagged_values() {
    let config = OpenAIConfig::new("key").with_originator("caller");
    let mut value = serde_json::to_value(&config).expect("config serializes");
    assert_eq!(value["identity"]["kind"], "originator");
    let restored: OpenAIConfig =
        serde_json::from_value(value.clone()).expect("config deserializes");
    assert_eq!(restored.identity, config.identity);
    value["identity"] = serde_json::json!({"originator": "caller", "user_agent": "rig/derived"});
    assert!(serde_json::from_value::<OpenAIConfig>(value).is_err());
    let exact = CallerIdentity::new("caller", "caller-agent", None).expect("valid identity");
    let serialized = serde_json::to_value(exact).expect("identity serializes");
    assert!(serialized.get("version").is_none());
}

/// Every identity entry point refuses the same empty or unsafe value locally.
#[test]
fn invalid_identity_retains_the_exact_rejected_field_and_value() {
    for (field, originator, user_agent, version) in [
        ("originator", "", "agent", None),
        ("originator", "caller\r\nextra: value", "agent", None),
        ("user_agent", "caller", "", None),
        ("user_agent", "caller", "agent\n", None),
        ("version", "caller", "agent", Some("")),
        ("version", "caller", "agent", Some("1\n")),
    ] {
        let error = CallerIdentity::new(originator, user_agent, version.map(str::to_owned))
            .expect_err("invalid identity refuses");
        let rejected = match field {
            "originator" => originator,
            "user_agent" => user_agent,
            "version" => version.expect("version case"),
            _ => unreachable!("table only names identity fields"),
        };
        assert_eq!(error.field, field);
        assert_eq!(error.value, rejected);
        let value = serde_json::json!({
            "originator": originator, "user_agent": user_agent, "version": version,
        });
        assert!(serde_json::from_value::<CallerIdentity>(value).is_err());
    }
    assert!(
        serde_json::from_value::<CallerIdentity>(serde_json::json!({
            "originator": "caller", "user_agent": "agent", "extra": "unrecognized",
        }))
        .is_err()
    );
}

/// Injected lookup exercises the actual paired-variable resolver without process-global env.
#[test]
fn paired_environment_identity_is_exact_or_absent_and_never_half_derived() {
    for (originator, user_agent) in [(None, None), (Some(""), Some(""))] {
        assert_eq!(
            identity_from_env(IDENTITY, |name| Ok(if name == IDENTITY.originator_env {
                originator.map(str::to_owned)
            } else {
                user_agent.map(str::to_owned)
            }))
            .expect("both absent"),
            None
        );
    }
    let identity = identity_from_env(IDENTITY, |name| {
        Ok(Some(
            if name == IDENTITY.originator_env {
                "caller"
            } else {
                "actual-agent/9"
            }
            .to_owned(),
        ))
    })
    .expect("paired values")
    .expect("identity set");
    assert_eq!(identity.originator(), "caller");
    assert_eq!(identity.user_agent(), "actual-agent/9");
    assert_eq!(identity.version(), None);
    for (originator, user_agent, missing) in [
        (Some("caller"), None, IDENTITY.user_agent_env),
        (Some("caller"), Some(""), IDENTITY.user_agent_env),
        (None, Some("agent"), IDENTITY.originator_env),
        (Some(""), Some("agent"), IDENTITY.originator_env),
    ] {
        let error = identity_from_env(IDENTITY, |name| {
            Ok(if name == IDENTITY.originator_env {
                originator.map(str::to_owned)
            } else {
                user_agent.map(str::to_owned)
            })
        })
        .expect_err("half identity refuses");
        assert!(matches!(error, EnvError::Invalid { name, .. } if name == missing));
    }
    for invalid_field in [IDENTITY.originator_env, IDENTITY.user_agent_env] {
        let error = identity_from_env(IDENTITY, |name| {
            Ok(Some(
                if name == invalid_field {
                    "invalid\nvalue"
                } else {
                    "valid"
                }
                .to_owned(),
            ))
        })
        .expect_err("unsafe env value refuses");
        assert!(matches!(error, EnvError::Invalid { name, .. } if name == invalid_field));
        let error = identity_from_env(IDENTITY, |name| {
            if name == invalid_field {
                Err(EnvError::Invalid {
                    name,
                    detail: "lookup failure".to_owned(),
                })
            } else {
                Ok(Some("valid".to_owned()))
            }
        })
        .expect_err("lookup failure propagates");
        assert!(
            matches!(error, EnvError::Invalid { name, detail } if name == invalid_field && detail == "lookup failure")
        );
    }
}

/// Missing identity is refused before a request exists and survives typed error conversion.
#[test]
fn required_identity_has_no_default_and_refusals_remain_downcastable() {
    let config = OpenAIConfig::with_key(&required_dialect(), "key");
    assert!(config.identity.is_none());
    let unset = config
        .headers(http::Request::builder())
        .expect_err("no identity refuses");
    assert_eq!(
        unset,
        MissingCallerIdentity::Unset {
            dialect: "requires-exact"
        }
    );
    let originator = config.clone().with_originator("caller");
    let refusal = originator
        .headers(http::Request::builder())
        .expect_err("derived identity refuses");
    assert_eq!(
        refusal,
        MissingCallerIdentity::OriginatorOnly {
            dialect: "requires-exact",
            originator: "caller".into(),
        }
    );
    let direct: ProviderError = refusal.clone().into();
    let encoded: EncodeError = refusal.clone().into();
    for error in [direct, ProviderError::from(encoded)] {
        assert_eq!(error.kind(), crate::error::ErrorKind::Request);
        match error {
            ProviderError::Request(source) => assert_eq!(
                source.downcast_ref::<MissingCallerIdentity>(),
                Some(&refusal)
            ),
            other => panic!("wrong error variant: {other:?}"),
        }
    }
    let exact = CallerIdentity::new("caller", "exact-agent", None).expect("valid identity");
    assert!(
        config
            .with_caller_identity(exact)
            .headers(http::Request::builder())
            .is_ok()
    );
    let optional = OpenAIConfig::new("key")
        .with_originator("caller")
        .headers(http::Request::builder())
        .expect("originator-only allowed")
        .body(())
        .expect("headers valid");
    assert_eq!(optional.headers()["originator"], "caller");
    assert_eq!(
        optional.headers()[http::header::USER_AGENT],
        default_user_agent("caller")
    );
}

struct FixedSource;

impl CredentialSource for FixedSource {
    fn current(
        &self,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, Result<Credential, CredentialSourceError>> {
        Box::pin(async { Ok(Credential::new("source-token")) })
    }
}

/// Source identity and placement are data contracts; these tests never request a credential.
#[test]
fn credential_source_survives_dialect_changes_and_owns_token_and_account_headers() {
    let config = OpenAIConfig::new("construction-token")
        .with_account_id("construction-account")
        .with_credential_source(FixedSource);
    let source = config.credential_source.clone();
    let switched = config.clone().with_dialect(&super::super::AZURE);
    assert_eq!(switched.credential_source, source);
    assert_eq!(switched.auth, Auth::ApiKeyHeader);
    let json = serde_json::to_value(&config).expect("config serializes");
    assert!(json.get("credential_source").is_none());
    let restored: OpenAIConfig = serde_json::from_value(json).expect("config deserializes");
    assert!(restored.credential_source.is_none());
    for (auth, placement) in [
        (Auth::Bearer, TokenPlacement::Bearer),
        (Auth::OptionalBearer, TokenPlacement::OptionalBearer),
        (
            Auth::ApiKeyHeader,
            TokenPlacement::Header(http::HeaderName::from_static(API_KEY_HEADER)),
        ),
    ] {
        let stamp = config
            .clone()
            .with_auth(auth)
            .credential_stamp()
            .expect("source configured");
        assert_eq!(Some(stamp.source), source);
        assert_eq!(stamp.placement.token, placement);
        assert_eq!(
            stamp.placement.account,
            Some(http::HeaderName::from_static(ACCOUNT_ID_HEADER))
        );
        let mut headers = config
            .clone()
            .with_auth(auth)
            .headers(http::Request::builder())
            .expect("identity optional")
            .body(())
            .expect("headers valid")
            .into_parts()
            .0
            .headers;
        headers.insert("unrelated", http::HeaderValue::from_static("kept"));
        stamp
            .placement
            .apply(&mut headers, &Credential::new("replacement"))
            .expect("valid credential");
        assert!(!headers.contains_key(ACCOUNT_ID_HEADER));
        assert_eq!(headers["unrelated"], "kept");
        let token_header = if auth == Auth::ApiKeyHeader {
            API_KEY_HEADER
        } else {
            "authorization"
        };
        let expected = if auth == Auth::ApiKeyHeader {
            "replacement"
        } else {
            "Bearer replacement"
        };
        assert_eq!(headers[token_header], expected);
        stamp
            .placement
            .apply(
                &mut headers,
                &Credential::new("next").with_account_id("source-account"),
            )
            .expect("valid credential with account");
        assert_eq!(headers[ACCOUNT_ID_HEADER], "source-account");
    }
}
