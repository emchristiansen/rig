use super::*;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use serde_json::json;

const ACCESS_TOKEN: &str = "test-token";

fn openai() -> Responses {
    Responses::new(OpenAIConfig::new("sk-test"), "gpt-5.4")
}

fn chatgpt_wire() -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN).with_account_id("acct-123"),
        chatgpt::GPT_5_3_CODEX,
    )
}

fn anonymous() -> CodexResponses {
    CodexResponses::new(chatgpt_wire()).expect("the ChatGPT dialect speaks the Codex contract")
}

fn caller(version: Option<&str>) -> CallerIdentity {
    CallerIdentity::new(
        "client_exec",
        "client_exec/1.2.3 (test)",
        version.map(str::to_owned),
    )
    .expect("a valid caller identity")
}

fn derived_identity() -> CodexIdentity {
    CodexIdentity::from_ids("session-derived", "thread-derived").expect("header-safe ids")
}

/// A wire read back from its serialized form and given its credential again:
/// Rig's `Secret` never serializes its value.
fn reloaded(json: serde_json::Value) -> CodexResponses {
    let mut wire: CodexResponses = serde_json::from_value(json).expect("deserializes");
    assert!(
        wire.wire.provider.api_key.is_empty(),
        "the credential is not in the serialized wire"
    );
    wire.wire.provider.api_key = ACCESS_TOKEN.into();
    wire
}

fn sorted(headers: &HeaderMap) -> Vec<(&str, &str)> {
    let mut sorted: Vec<(&str, &str)> = headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.to_str().expect("ascii header")))
        .collect();
    sorted.sort_unstable();
    sorted
}

#[test]
fn a_codex_identity_is_refused_on_a_wire_that_does_not_speak_the_codex_contract() {
    let error = CodexResponses::new(openai()).expect_err("the OpenAI dialect is not Codex");
    assert_eq!(error.dialect, "openai");

    // The fields are public, so an identity can still reach such a wire.
    let by_field = CodexResponses {
        wire: openai(),
        ..anonymous()
    };
    let error = by_field
        .clone()
        .with_codex_identity(derived_identity())
        .expect_err("the OpenAI dialect is not Codex");
    assert_eq!(error.dialect, "openai");
}

/// An identity that reaches a non-Codex wire other than through
/// `with_codex_identity`, by field assignment or by serde, is refused by name
/// when its headers are stamped, so nothing is stamped or sent.
#[test]
fn a_codex_identity_on_a_non_codex_wire_is_refused_when_stamping() {
    let by_field = CodexResponses {
        wire: openai(),
        codex_identity: Some(derived_identity()),
        ..anonymous().with_caller_identity(caller(None))
    };
    let mut json =
        serde_json::to_value(anonymous().with_caller_identity(caller(None))).expect("serializes");
    json["wire"] = serde_json::to_value(openai()).expect("serializes");
    json["codex_identity"] = serde_json::to_value(derived_identity()).expect("serializes");
    let by_serde: CodexResponses = serde_json::from_value(json).expect("deserializes");

    for wire in [by_field, by_serde] {
        let mut headers = HeaderMap::new();
        let error = wire
            .stamp_headers(wire.codex_identity.as_ref(), &mut headers)
            .expect_err("an OpenAI wire does not speak the Codex contract");
        assert!(headers.is_empty(), "nothing is stamped");
        let ProviderError::Request(reason) = ProviderError::from(error) else {
            panic!("a refusal is a request error");
        };
        assert_eq!(
            reason.downcast_ref::<NotACodexWire>(),
            Some(&NotACodexWire { dialect: "openai" })
        );
    }
}

/// Stamping adds every request header to a request, beside the caller
/// identity, and nothing else.
#[test]
fn stamping_adds_every_header() {
    let wire = anonymous()
        .with_caller_identity(caller(None))
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-window-id", "thread-1:0")
                .expect("a valid header")
                .with("x-codex-routing-hint", "model=gpt")
                .expect("a valid header"),
        );
    let mut headers = HeaderMap::new();
    wire.stamp_headers(None, &mut headers).expect("stamps");
    assert_eq!(headers["x-codex-window-id"], "thread-1:0");
    assert_eq!(headers["x-codex-routing-hint"], "model=gpt");
    assert_eq!(
        sorted(&headers),
        [
            ("originator", "client_exec"),
            ("user-agent", "client_exec/1.2.3 (test)"),
            ("x-codex-routing-hint", "model=gpt"),
            ("x-codex-window-id", "thread-1:0"),
        ]
    );
}

/// The stamped set is exact: the caller identity replaces another's, a
/// `version` the caller does not name is removed, and the conversation's
/// identity replaces any session id the request had under either spelling.
#[test]
fn stamping_replaces_what_another_source_put_under_the_same_names() {
    let theirs = || {
        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("authorization", "Bearer test-token"),
            ("originator", "rig"),
            ("user-agent", "rig/0 (linux x86_64; rig)"),
            ("version", "0.46.0"),
            ("session_id", "per-request"),
            ("session-id", "per-request"),
        ] {
            set(&mut headers, name, value).expect("a valid header");
        }
        headers
    };

    let mut unversioned = theirs();
    anonymous()
        .with_caller_identity(caller(None))
        .stamp_headers(Some(&derived_identity()), &mut unversioned)
        .expect("stamps");
    assert_eq!(
        sorted(&unversioned),
        [
            ("authorization", "Bearer test-token"),
            ("originator", "client_exec"),
            ("session-id", "session-derived"),
            ("thread-id", "thread-derived"),
            ("user-agent", "client_exec/1.2.3 (test)"),
            ("x-client-request-id", "thread-derived"),
        ]
    );

    let mut versioned = theirs();
    anonymous()
        .with_caller_identity(caller(Some("1.2.3")))
        .stamp_headers(None, &mut versioned)
        .expect("stamps");
    assert_eq!(versioned[VERSION_HEADER], "1.2.3");
    assert_eq!(
        versioned["session_id"], "per-request",
        "without a conversation identity the request's own session id stays"
    );
    for name in ["originator", "user-agent", "version"] {
        assert_eq!(versioned.get_all(name).iter().count(), 1, "{name}");
    }
}

/// Without a caller identity nothing is stamped.
#[test]
fn stamping_refuses_without_a_caller_identity() {
    let wire = anonymous();
    assert_eq!(
        wire.sendable_identity(),
        Err(MissingCallerIdentity { dialect: "chatgpt" })
    );
    let mut headers = HeaderMap::new();
    let error = wire
        .stamp_headers(Some(&derived_identity()), &mut headers)
        .expect_err("no caller identity");
    assert!(matches!(error, CodexRequestError::MissingCallerIdentity(_)));
    assert!(headers.is_empty());

    let wire = wire.with_caller_identity(caller(None));
    assert_eq!(wire.sendable_identity(), Ok(&caller(None)));
}

/// The credential headers are the ones the dialect's authentication sends,
/// with the account when the wire names one.
#[test]
fn the_credential_stamp_is_the_dialects_own() {
    let mut headers = HeaderMap::new();
    anonymous().stamp_credential(&mut headers).expect("stamps");
    assert_eq!(
        sorted(&headers),
        [
            ("authorization", "Bearer test-token"),
            ("chatgpt-account-id", "acct-123"),
        ]
    );

    let mut api_key = chatgpt_wire();
    api_key.provider.auth = Auth::ApiKeyHeader;
    api_key.provider.account_id = None;
    let mut headers = HeaderMap::new();
    CodexResponses::new(api_key)
        .expect("a Codex wire")
        .stamp_credential(&mut headers)
        .expect("stamps");
    assert_eq!(sorted(&headers), [("api-key", "test-token")]);

    let mut keyless = Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, ""),
        chatgpt::GPT_5_3_CODEX,
    );
    keyless.provider.auth = Auth::OptionalBearer;
    let mut headers = HeaderMap::new();
    CodexResponses::new(keyless)
        .expect("a Codex wire")
        .stamp_credential(&mut headers)
        .expect("stamps");
    assert!(headers.is_empty(), "no key, no credential header");
}

/// The defaults stay off the serialized wire, and a wire with every setting
/// round-trips.
#[test]
fn the_settings_serialize_only_when_set_and_round_trip() {
    let plain = serde_json::to_value(anonymous()).expect("serializes");
    assert_eq!(
        plain
            .as_object()
            .expect("an object")
            .keys()
            .collect::<Vec<_>>(),
        ["wire"],
        "got {plain}"
    );
    assert_eq!(reloaded(plain), anonymous());

    let full = anonymous()
        .with_caller_identity(caller(Some("1.2.3")))
        .with_codex_identity(derived_identity())
        .expect("a Codex wire")
        .with_request_headers(
            RequestHeaders::new()
                .with("x-codex-window-id", "thread-1:0")
                .expect("a valid header"),
        );
    let json = serde_json::to_value(&full).expect("serializes");
    assert_eq!(
        json["caller_identity"],
        json!({"originator": "client_exec", "user_agent": "client_exec/1.2.3 (test)", "version": "1.2.3"})
    );
    assert_eq!(
        json["codex_identity"],
        json!({"session_id": "session-derived", "thread_id": "thread-derived"})
    );
    assert_eq!(
        json["request_headers"],
        json!([["x-codex-window-id", "thread-1:0"]])
    );
    assert!(json.get("request_compression").is_none(), "got {json}");
    assert_eq!(reloaded(json), full);
}

/// The wire's `Debug` shows the credential in the typed secret's own form,
/// never as its value.
#[test]
fn debug_shows_the_credential_in_the_secrets_own_form() {
    let wire = anonymous().with_caller_identity(caller(None));
    let shown = format!("{wire:?}");
    assert!(shown.contains("client_exec"), "{shown}");
    assert!(
        !shown.contains(ACCESS_TOKEN),
        "the credential's value is not in the Debug form"
    );
    assert_eq!(
        format!("{:?}", wire.wire.provider.api_key),
        format!("{:?}", rig_core::wire::Secret::from(ACCESS_TOKEN)),
        "the field's Debug is the secret type's own"
    );
}

/// The control check reports the first refused control in a fixed order, and
/// nothing for a request that sets none.
#[test]
fn an_unsupported_control_is_named_in_a_fixed_order() {
    let request = || CompletionRequest::new("hello");
    assert_eq!(UnsupportedCodexControl::in_request(&request()), None);
    assert_eq!(
        UnsupportedCodexControl::in_request(
            &request().additional_params(json!({"store": false, "top_p": null}))
        ),
        None,
        "`store: false` is what the contract states, and a null is not a value"
    );

    let mut all = request().additional_params(json!({"top_p": 0.5, "store": true}));
    all.max_tokens = Some(64);
    all.temperature = Some(0.25);
    assert_eq!(
        UnsupportedCodexControl::in_request(&all),
        Some(UnsupportedCodexControl::MaxOutputTokens)
    );
    all.max_tokens = None;
    assert_eq!(
        UnsupportedCodexControl::in_request(&all),
        Some(UnsupportedCodexControl::Temperature)
    );
    all.temperature = None;
    assert_eq!(
        UnsupportedCodexControl::in_request(&all),
        Some(UnsupportedCodexControl::TopP)
    );
    assert_eq!(
        UnsupportedCodexControl::in_request(&request().additional_params(json!({"store": true}))),
        Some(UnsupportedCodexControl::Store)
    );

    let error = ProviderError::from(UnsupportedCodexControl::TopP);
    let ProviderError::Request(reason) = &error else {
        panic!("a refused control is a request error, got {error:?}");
    };
    assert_eq!(
        reason.downcast_ref::<UnsupportedCodexControl>(),
        Some(&UnsupportedCodexControl::TopP)
    );
}

/// A serialized wire whose caller identity no request could carry does not
/// decode.
#[test]
fn a_serialized_wire_with_an_invalid_caller_identity_does_not_decode() {
    let mut json =
        serde_json::to_value(anonymous().with_caller_identity(caller(None))).expect("serializes");
    assert!(
        serde_json::from_value::<CodexResponses>(json.clone()).is_ok(),
        "the wire as serialized decodes, so the refusal below is the identity's"
    );
    json["caller_identity"]["originator"] = json!("");
    assert!(serde_json::from_value::<CodexResponses>(json).is_err());
}
