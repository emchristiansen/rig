#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::RequestBody;
use serde_json::{Value, json};

/// The fields' names in the order `body` writes them.
fn written_order(body: &RequestBody) -> Vec<String> {
    let text = String::from_utf8(body.to_vec().expect("serializes")).expect("is UTF-8");
    let value: Vec<(String, Value)> = serde_json::from_str::<super::Fields>(&text)
        .expect("parses back")
        .0
        .into_iter()
        .map(|(key, raw)| (key, serde_json::from_str(raw.get()).expect("is JSON")))
        .collect();
    value.into_iter().map(|(key, _)| key).collect()
}

#[test]
fn an_untouched_field_keeps_the_bytes_it_was_read_with() {
    // Number spelling, nested key order and escapes that a round trip
    // through `Value` would rewrite.
    let nested = r#"{"zeta":1.0,"alpha":1e2,"text":"aé\/b","list":[ 1,2 ]}"#;
    let text = format!(r#"{{"model":"m","tools":{nested},"input":[]}}"#);
    let mut body = RequestBody::parse(text.as_bytes()).expect("parses");
    body.insert("stream", &true).expect("inserts");

    assert_eq!(body.raw("tools").expect("has tools").get(), nested);
    let written = String::from_utf8(body.to_vec().expect("serializes")).expect("is UTF-8");
    assert!(written.contains(nested), "{written}");
}

#[test]
fn the_conversation_and_the_output_cap_are_written_last() {
    let body =
        RequestBody::parse(br#"{"input":[],"max_output_tokens":5,"model":"m","stream":true}"#)
            .expect("parses");
    let order = written_order(&body);
    assert_eq!(
        order.get(order.len() - 2..).expect("has two fields"),
        ["input", "max_output_tokens"]
    );
}

#[test]
fn a_replaced_field_keeps_its_place_and_a_removed_one_leaves_the_rest_in_order() {
    let mut body = RequestBody::parse(br#"{"a":1,"b":2,"c":3,"input":[]}"#).expect("parses");
    let before = written_order(&body);
    body.insert("b", &json!({"replaced": true}))
        .expect("inserts");
    assert_eq!(written_order(&body), before);
    assert_eq!(
        body.get::<Value>("b").expect("has b").expect("is JSON"),
        json!({"replaced": true})
    );

    assert_eq!(
        body.remove("b").expect("had b").get(),
        r#"{"replaced":true}"#
    );
    assert!(!body.contains("b"));
    assert_eq!(written_order(&body), ["a", "c", "input"]);
    assert_eq!(body.len(), 3);
}

#[test]
fn a_body_that_is_not_an_object_is_refused() {
    assert!(RequestBody::parse(b"[1,2]").is_err());
    assert!(RequestBody::parse(b"\"text\"").is_err());
}

#[test]
fn equal_bodies_have_the_same_fields_and_bytes() {
    let one = RequestBody::parse(br#"{"a":1,"input":[]}"#).expect("parses");
    let same = RequestBody::parse(br#"{"a":1,"input":[]}"#).expect("parses");
    let respelled = RequestBody::parse(br#"{"a":1.0,"input":[]}"#).expect("parses");
    assert_eq!(one, same);
    assert_ne!(one, respelled);
}
