use super::*;

/// A set keeps each header exactly as given, lowercasing only the name, in
/// the order the caller added them.
#[test]
fn a_set_keeps_its_headers_in_order_with_lowercase_names() {
    let headers = RequestHeaders::new()
        .with("X-Codex-Window-Id", "thread-1:0")
        .expect("a valid header")
        .with("x-codex-beta-features", "remote_compaction_v2")
        .expect("a valid header");
    assert_eq!(
        headers.iter().collect::<Vec<_>>(),
        [
            ("x-codex-window-id", "thread-1:0"),
            ("x-codex-beta-features", "remote_compaction_v2"),
        ]
    );
    assert!(!headers.is_empty());
    assert!(RequestHeaders::new().is_empty());
}

/// Every header Rig sends from its own source is refused, whatever its case,
/// so a caller can neither send one twice nor contradict its owner.
#[test]
fn a_header_rig_sends_itself_is_refused_by_name() {
    for name in RESERVED {
        let refused = RequestHeaders::new()
            .with(name.to_ascii_uppercase(), "value")
            .expect_err("a reserved header is refused");
        assert_eq!(
            refused,
            InvalidRequestHeader::Reserved {
                name: (*name).to_owned()
            }
        );
    }
    // The ones named by the identity rulings, spelled out.
    for name in [
        "originator",
        "user-agent",
        "version",
        "session-id",
        "thread-id",
        "x-client-request-id",
        "session_id",
        "authorization",
        "chatgpt-account-id",
        "openai-beta",
        "x-openai-internal-codex-responses-lite",
        "content-encoding",
    ] {
        assert!(
            RESERVED.contains(&name),
            "`{name}` must be one of the reserved headers"
        );
    }
}

#[test]
fn a_repeated_header_is_refused() {
    let refused = RequestHeaders::new()
        .with("x-codex-routing-hint", "model=a")
        .expect("a valid header")
        .with("X-Codex-Routing-Hint", "model=b")
        .expect_err("the same header twice is refused");
    assert_eq!(
        refused,
        InvalidRequestHeader::Duplicate {
            name: "x-codex-routing-hint".to_owned()
        }
    );
}

#[test]
fn an_invalid_name_or_value_is_refused() {
    assert_eq!(
        RequestHeaders::new().with("bad name", "value"),
        Err(InvalidRequestHeader::Name {
            name: "bad name".to_owned()
        })
    );
    for value in ["", "line\nbreak"] {
        assert_eq!(
            RequestHeaders::new().with("x-codex-window-id", value),
            Err(InvalidRequestHeader::Value {
                name: "x-codex-window-id".to_owned(),
                value: value.to_owned(),
            })
        );
    }
}

/// The serialized form round-trips, and is validated on the way back in.
#[test]
fn the_serialized_form_round_trips_and_is_validated() {
    let headers = RequestHeaders::new()
        .with("x-codex-window-id", "thread-1:0")
        .expect("a valid header");
    let json = serde_json::to_value(&headers).expect("serializes");
    assert_eq!(
        json,
        serde_json::json!([["x-codex-window-id", "thread-1:0"]])
    );
    assert_eq!(
        serde_json::from_value::<RequestHeaders>(json).expect("deserializes"),
        headers
    );
    assert!(
        serde_json::from_value::<RequestHeaders>(serde_json::json!([["originator", "x"]])).is_err(),
        "a reserved header cannot enter through serde"
    );
}

/// Stamping adds every header in order to a request.
#[test]
fn stamping_adds_every_header() {
    let headers = RequestHeaders::new()
        .with("x-codex-window-id", "thread-1:0")
        .expect("a valid header")
        .with("x-codex-routing-hint", "model=gpt")
        .expect("a valid header");
    let request = headers
        .stamp(http::Request::get("https://example.invalid/"))
        .body(())
        .expect("builds");
    assert_eq!(request.headers()["x-codex-window-id"], "thread-1:0");
    assert_eq!(request.headers()["x-codex-routing-hint"], "model=gpt");
    assert_eq!(request.headers().len(), 2);
}
