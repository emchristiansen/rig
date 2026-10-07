//! Contracts of the capturing transport double, independent of provider traffic.

use super::CapturingStreamingClient;
use crate::http_client::{DynHttpClient, HttpClientExt, MultipartForm, Request};
use bytes::Bytes;
use futures::{StreamExt, executor::block_on};
use http::{HeaderMap, HeaderValue, StatusCode};

fn request(headers: HeaderMap) -> Request<Bytes> {
    let mut request = Request::new(Bytes::from_static(b"request body"));
    *request.headers_mut() = headers;
    request
}

#[test]
fn capture_order_is_call_order_even_when_futures_are_polled_in_reverse() {
    let client = CapturingStreamingClient {
        sse_bytes: Bytes::from_static(b"data: reply\n\n"),
        ..Default::default()
    };
    let clone = client.clone();
    let mut first_headers = HeaderMap::new();
    first_headers.append("x-repeat", HeaderValue::from_static(" first "));
    first_headers.append("x-repeat", HeaderValue::from_static(""));
    first_headers.append(
        "x-repeat",
        HeaderValue::from_bytes(b"\xff").expect("opaque header"),
    );
    first_headers.insert("x-only-first", HeaderValue::from_static("kept"));
    let second_headers = HeaderMap::new();

    let first = client.send_streaming(request(first_headers.clone()));
    let second = clone.send_streaming(request(second_headers.clone()));
    assert_eq!(
        *client.requests.lock().expect("capture log"),
        [first_headers.clone(), second_headers.clone()],
        "capture precedes polling and clones share the log"
    );
    let captured = client.requests.lock().expect("capture log").clone();
    assert_eq!(
        captured[0]
            .get_all("x-repeat")
            .iter()
            .map(HeaderValue::as_bytes)
            .collect::<Vec<_>>(),
        [b" first ".as_slice(), b"".as_slice(), b"\xff".as_slice()]
    );
    assert!(!captured[1].contains_key("x-only-first"));

    let second_response = block_on(second).expect("second response");
    let first_response = block_on(first).expect("first response");
    for response in [second_response, first_response] {
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().len(), 1);
        assert_eq!(
            response.headers()[http::header::CONTENT_TYPE],
            "text/event-stream"
        );
        let mut body = response.into_body();
        assert_eq!(
            block_on(body.next())
                .expect("one chunk")
                .expect("successful chunk"),
            client.sse_bytes
        );
        assert!(block_on(body.next()).is_none());
    }
    assert_eq!(
        *client.requests.lock().expect("capture log"),
        [first_headers, second_headers],
        "polling does not capture twice or reorder requests"
    );

    let discarded = clone.send_streaming(request(HeaderMap::new()));
    drop(discarded);
    assert_eq!(client.requests.lock().expect("capture log").len(), 3);
}

#[test]
fn the_default_client_yields_one_empty_chunk_through_the_erased_transport() {
    let client = CapturingStreamingClient::default();
    let erased = DynHttpClient::new(client.clone());
    let response = block_on(erased.send_streaming(request(HeaderMap::new()))).expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().len(), 1);
    assert_eq!(
        response.headers()[http::header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert!(!response.headers().contains_key("x-request-id"));
    let mut body = response.into_body();
    assert_eq!(
        block_on(body.next())
            .expect("empty chunk exists")
            .expect("successful chunk"),
        Bytes::new()
    );
    assert!(block_on(body.next()).is_none());
    assert!(block_on(body.next()).is_none());
    assert_eq!(
        *client.requests.lock().expect("capture log"),
        [HeaderMap::new()]
    );
}

#[test]
fn unsupported_unary_surfaces_preserve_the_failure_details_and_do_not_capture() {
    let client = CapturingStreamingClient::default();
    let unary = block_on(client.send::<_, Bytes>(request(HeaderMap::new())))
        .err()
        .expect("unary refused");
    let multipart_request = Request::new(MultipartForm::new().text("field", "value"));
    let multipart = block_on(client.send_multipart::<Bytes>(multipart_request))
        .err()
        .expect("multipart refused");
    for error in [unary, multipart] {
        assert_eq!(
            error.non_success_status(),
            Some(StatusCode::NOT_IMPLEMENTED)
        );
        assert_eq!(error.non_success_body(), Some(""));
        assert!(error.non_success_headers().is_some_and(HeaderMap::is_empty));
    }
    assert!(client.requests.lock().expect("capture log").is_empty());
}
