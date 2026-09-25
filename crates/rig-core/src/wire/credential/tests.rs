//! Byte-level tests of [`CredentialPlacement::apply`], the one header write
//! every send-time transport shares.

use super::{Credential, CredentialPlacement, TokenPlacement};

fn account_header() -> http::HeaderName {
    http::HeaderName::from_static("chatgpt-account-id")
}

fn encoded_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::AUTHORIZATION,
        http::HeaderValue::from_static("Bearer static-key"),
    );
    headers.insert(
        account_header(),
        http::HeaderValue::from_static("static-account"),
    );
    headers.insert("x-unrelated", http::HeaderValue::from_static("kept"));
    headers
}

fn written(headers: &http::HeaderMap) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect();
    pairs.sort();
    pairs
}

fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = expected
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    pairs.sort();
    pairs
}

#[test]
fn a_bearer_credential_replaces_the_encoded_token_and_account() {
    let placement = CredentialPlacement {
        token: TokenPlacement::Bearer,
        account: Some(account_header()),
    };
    let mut headers = encoded_headers();
    placement
        .apply(
            &mut headers,
            &Credential::new("source-token").with_account_id("source-account"),
        )
        .expect("both values fit a header");
    assert_eq!(
        written(&headers),
        pairs(&[
            ("authorization", "Bearer source-token"),
            ("chatgpt-account-id", "source-account"),
            ("x-unrelated", "kept"),
        ])
    );
}

#[test]
fn a_credential_naming_no_account_removes_the_encoded_account() {
    let placement = CredentialPlacement {
        token: TokenPlacement::Bearer,
        account: Some(account_header()),
    };
    let mut headers = encoded_headers();
    placement
        .apply(&mut headers, &Credential::new("source-token"))
        .expect("the token fits a header");
    assert_eq!(
        written(&headers),
        pairs(&[
            ("authorization", "Bearer source-token"),
            ("x-unrelated", "kept"),
        ])
    );
}

#[test]
fn an_empty_optional_bearer_sends_no_credential_header() {
    let placement = CredentialPlacement {
        token: TokenPlacement::OptionalBearer,
        account: None,
    };
    let mut headers = encoded_headers();
    placement
        .apply(&mut headers, &Credential::new(""))
        .expect("nothing to write");
    assert_eq!(
        written(&headers),
        pairs(&[
            ("chatgpt-account-id", "static-account"),
            ("x-unrelated", "kept"),
        ]),
        "no account header is named, so the encoded one is left alone"
    );

    let mut headers = encoded_headers();
    placement
        .apply(&mut headers, &Credential::new("source-token"))
        .expect("the token fits a header");
    assert_eq!(
        headers.get(http::header::AUTHORIZATION),
        Some(&http::HeaderValue::from_static("Bearer source-token"))
    );
}

#[test]
fn a_named_header_carries_the_token_alone() {
    let placement = CredentialPlacement {
        token: TokenPlacement::Header(http::HeaderName::from_static("api-key")),
        account: None,
    };
    let mut headers = http::HeaderMap::new();
    headers.insert("api-key", http::HeaderValue::from_static("static-key"));
    placement
        .apply(&mut headers, &Credential::new("source-token"))
        .expect("the token fits a header");
    assert_eq!(written(&headers), pairs(&[("api-key", "source-token")]));
}

#[test]
fn a_value_no_header_can_carry_is_refused_without_naming_it() {
    let placement = CredentialPlacement {
        token: TokenPlacement::Bearer,
        account: Some(account_header()),
    };
    for credential in [
        Credential::new("line\nbreak-secret"),
        Credential::new("fine").with_account_id("line\nbreak-secret"),
    ] {
        let error = placement
            .apply(&mut encoded_headers(), &credential)
            .expect_err("a newline cannot be sent in a header");
        let rendered = format!("{error} {error:?}");
        assert!(
            !rendered.contains("break-secret"),
            "the refusal names the problem, never the value: {rendered}"
        );
    }
}
