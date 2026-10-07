use super::{captured_response_headers, request_id_from_headers};
use crate::completion::ProviderResponseHeaders;
use http::{HeaderMap, HeaderName, HeaderValue};

#[test]
fn request_id_requires_a_named_present_nonempty_header() {
    let mut headers = HeaderMap::new();
    assert_eq!(
        request_id_from_headers(&headers, Some("x-request-id")),
        None
    );

    headers.insert("x-request-id", HeaderValue::from_static("request-42"));
    assert_eq!(request_id_from_headers(&headers, None), None);
    assert_eq!(request_id_from_headers(&headers, Some("request-id")), None);
    assert_eq!(
        request_id_from_headers(&headers, Some("X-Request-Id")),
        Some("request-42".to_owned())
    );

    headers.insert("x-request-id", HeaderValue::from_static(""));
    assert_eq!(
        request_id_from_headers(&headers, Some("x-request-id")),
        None
    );
}

#[test]
fn request_id_keeps_surrounding_and_whitespace_only_values() {
    let mut headers = HeaderMap::new();
    for value in [" \trequest-42\t ", " \t "] {
        headers.insert(
            "x-request-id",
            HeaderValue::from_bytes(value.as_bytes()).expect("valid whitespace in a header"),
        );
        assert_eq!(
            request_id_from_headers(&headers, Some("x-request-id")),
            Some(value.to_owned())
        );
    }
}

#[test]
fn request_id_rejects_values_that_header_to_str_cannot_read() {
    let mut headers = HeaderMap::new();
    for value in [b"request-\xff".as_slice(), "request-café".as_bytes()] {
        headers.insert(
            "x-request-id",
            HeaderValue::from_bytes(value).expect("opaque bytes are valid in a header"),
        );
        assert_eq!(
            request_id_from_headers(&headers, Some("x-request-id")),
            None
        );
    }
}

#[test]
fn request_id_reads_only_the_first_repeated_value() {
    let mut headers = HeaderMap::new();
    headers.append("x-request-id", HeaderValue::from_static("first"));
    headers.append("x-request-id", HeaderValue::from_static("second"));
    assert_eq!(
        request_id_from_headers(&headers, Some("x-request-id")),
        Some("first".to_owned())
    );

    for first in [b"".as_slice(), b"\xff".as_slice()] {
        let mut headers = HeaderMap::new();
        headers.append(
            "x-request-id",
            HeaderValue::from_bytes(first).expect("empty and opaque header values are valid"),
        );
        headers.append("x-request-id", HeaderValue::from_static("later-valid"));
        assert_eq!(
            request_id_from_headers(&headers, Some("x-request-id")),
            None
        );
    }
}

#[test]
fn captured_headers_use_the_exact_prefix_of_lowercase_names() {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_bytes(b"X-RateLimit-Remaining").expect("valid mixed-case header name"),
        HeaderValue::from_static("17"),
    );
    headers.insert("x-ratelimit-reset", HeaderValue::from_static("soon"));
    headers.insert("x-request-id", HeaderValue::from_static("request-42"));
    headers.insert("other-x-ratelimit-limit", HeaderValue::from_static("100"));

    assert_eq!(
        captured_response_headers(&headers, None),
        ProviderResponseHeaders::new()
    );
    assert_eq!(
        captured_response_headers(&headers, Some("x-ratelimit-")),
        ProviderResponseHeaders::from([
            ("x-ratelimit-remaining".to_owned(), "17".to_owned()),
            ("x-ratelimit-reset".to_owned(), "soon".to_owned()),
        ])
    );
    for prefix in ["X-RateLimit-", "unreported-"] {
        assert_eq!(
            captured_response_headers(&headers, Some(prefix)),
            ProviderResponseHeaders::new()
        );
    }
    assert_eq!(
        captured_response_headers(&headers, Some("")),
        ProviderResponseHeaders::from([
            ("x-ratelimit-remaining".to_owned(), "17".to_owned()),
            ("x-ratelimit-reset".to_owned(), "soon".to_owned()),
            ("x-request-id".to_owned(), "request-42".to_owned()),
            ("other-x-ratelimit-limit".to_owned(), "100".to_owned()),
        ])
    );
}

#[test]
fn captured_repeated_headers_join_each_values_arrival_order_exactly() {
    let mut headers = HeaderMap::new();
    headers.append("x-limit-a", HeaderValue::from_static(" second "));
    headers.append("x-limit-b", HeaderValue::from_static("b-first"));
    headers.append("x-limit-a", HeaderValue::from_static(""));
    headers.append("x-limit-b", HeaderValue::from_static("b-second"));
    headers.append("x-limit-a", HeaderValue::from_static("first\t"));
    headers.append("x-limit-a", HeaderValue::from_static(" second "));

    assert_eq!(
        captured_response_headers(&headers, Some("x-limit-")),
        ProviderResponseHeaders::from([
            (
                "x-limit-a".to_owned(),
                " second , , first\t,  second ".to_owned()
            ),
            ("x-limit-b".to_owned(), "b-first, b-second".to_owned()),
        ])
    );
}

#[test]
fn captured_headers_preserve_empty_whitespace_and_valid_utf8_values() {
    let mut headers = HeaderMap::new();
    for (name, value) in [
        ("x-limit-empty", ""),
        ("x-limit-space", " \t "),
        ("x-limit-value", " \t17\t "),
        ("x-limit-unicode", " café "),
    ] {
        headers.insert(
            name,
            HeaderValue::from_bytes(value.as_bytes()).expect("valid header bytes"),
        );
    }
    assert_eq!(
        captured_response_headers(&headers, Some("x-limit-")),
        ProviderResponseHeaders::from([
            ("x-limit-empty".to_owned(), "".to_owned()),
            ("x-limit-space".to_owned(), " \t ".to_owned()),
            ("x-limit-value".to_owned(), " \t17\t ".to_owned()),
            ("x-limit-unicode".to_owned(), " café ".to_owned()),
        ])
    );
}

#[test]
fn captured_invalid_utf8_replaces_only_invalid_sequences_before_joining() {
    let mut headers = HeaderMap::new();
    headers.append(
        "x-limit-opaque",
        HeaderValue::from_bytes(b" \xff\xfe\t").expect("opaque bytes are valid in a header"),
    );
    headers.append(
        "x-limit-opaque",
        HeaderValue::from_bytes(" café ".as_bytes()).expect("UTF-8 bytes are valid in a header"),
    );
    headers.append("x-limit-opaque", HeaderValue::from_static(""));
    assert_eq!(
        captured_response_headers(&headers, Some("x-limit-")),
        ProviderResponseHeaders::from([(
            "x-limit-opaque".to_owned(),
            " \u{fffd}\u{fffd}\t,  café , ".to_owned(),
        )])
    );
}
