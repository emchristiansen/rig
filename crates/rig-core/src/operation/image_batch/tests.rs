use super::*;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde_json::json;

fn png(tail: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(tail);
    bytes
}

fn model(http: SequencedHttpClient) -> Model<ImageEmbeddings, SequencedHttpClient> {
    Model::new(
        crate::providers::cohere::CohereConfig::new("cohere-test-key").image_embedding(),
        http,
    )
}

fn image_reply(first: f64) -> MockHttpResponse {
    MockHttpResponse::success(format!(
        r#"{{"id":"img-{first}","embeddings":{{"float":[[{first},1.0]]}},"meta":{{"api_version":{{"version":"1"}},"billed_units":{{"search_units":0,"classifications":0,"images":1}}}}}}"#
    ))
}

#[tokio::test]
async fn an_image_batch_folds_its_replies_in_input_order() {
    let http = SequencedHttpClient::new([image_reply(0.5), image_reply(0.75)]);
    let response = model(http.clone())
        .embed_images_response(CohereImageBatch::new(png(b"first"), [png(b"second")]))
        .await
        .expect("both replies decode");

    assert_eq!(http.requests().len(), 2);
    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| embedding.vec.as_slice())
            .collect::<Vec<_>>(),
        vec![[0.5, 1.0].as_slice(), [0.75, 1.0].as_slice()]
    );
    assert_eq!(
        response.embeddings[0].document,
        crate::embeddings::image_document(&png(b"first"))
    );
    assert_ne!(
        response.embeddings[0].document,
        response.embeddings[1].document
    );
    assert_eq!(response.usage.input_tokens, None);
    let pages: Vec<crate::providers::cohere::embeddings::ImageEmbeddingResponse> =
        serde_json::from_value(response.raw.clone()).expect("raw is the per-image array");
    assert_eq!(pages.len(), 2);
    assert_eq!(
        pages
            .iter()
            .map(|page| page.id.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("img-0.5"), Some("img-0.75")]
    );
    assert!(
        pages
            .iter()
            .all(|page| { page.meta.as_ref().map(|meta| meta.billed_units.images) == Some(1) })
    );
    assert_eq!(response.response_id.as_deref(), Some("img-0.5"));
    assert_eq!(response.usage, crate::completion::Usage::default());
    let requests = http.requests();
    for (request, image) in requests.iter().zip([png(b"first"), png(b"second")]) {
        let body: serde_json::Value = serde_json::from_slice(&request.body).expect("JSON request");
        assert!(request.uri.ends_with("/v1/embed"));
        assert_eq!(body["model"], "embed-english-v3.0");
        assert_eq!(body["input_type"], "image");
        assert_eq!(body["embedding_types"], json!(["float"]));
        assert_eq!(body["images"].as_array().expect("images").len(), 1);
        use base64::Engine as _;
        assert_eq!(
            body["images"][0],
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(image)
            )
        );
    }
}

#[tokio::test]
async fn an_invalid_second_image_sends_no_request() {
    let http = SequencedHttpClient::new([image_reply(0.5), image_reply(0.75)]);
    let error = model(http.clone())
        .embed_images_response(CohereImageBatch::new(
            png(b"first"),
            [b"not an image".to_vec()],
        ))
        .await
        .expect_err("the entire batch preflights");
    assert!(matches!(error, ProviderError::Request(_)));
    assert!(http.requests().is_empty());
    assert_eq!(http.remaining_responses(), 2);
}

#[tokio::test]
async fn a_failed_second_image_discards_successes_and_sends_no_later_request() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("17"));
    let body = r#"{"message":"second image rejected"}"#;
    let http = SequencedHttpClient::new([
        image_reply(0.5),
        MockHttpResponse::error_with_headers(http::StatusCode::TOO_MANY_REQUESTS, body, headers),
        image_reply(0.75),
    ]);
    let error = model(http.clone())
        .embed_images_response(CohereImageBatch::new(
            png(b"first"),
            [png(b"second"), png(b"third")],
        ))
        .await
        .expect_err("no partial successful response is returned");
    assert_eq!(http.requests().len(), 2);
    assert_eq!(http.remaining_responses(), 1);
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after")),
        Some(&http::HeaderValue::from_static("17"))
    );
}

#[tokio::test]
async fn a_second_page_decoder_failure_stops_the_batch() {
    let http = SequencedHttpClient::new([
        image_reply(0.5),
        MockHttpResponse::success(r#"{"message":"bad page"}"#),
        image_reply(0.75),
    ]);
    let error = model(http.clone())
        .embed_images_response(CohereImageBatch::new(
            png(b"first"),
            [png(b"second"), png(b"third")],
        ))
        .await
        .expect_err("the decoder error aborts the batch");
    assert_eq!(http.requests().len(), 2);
    assert_eq!(http.remaining_responses(), 1);
    assert_eq!(
        error.provider_response_body(),
        Some(r#"{"message":"bad page"}"#)
    );
    assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
}

#[tokio::test]
async fn one_image_matches_the_single_exchange_and_keeps_a_bare_raw_page() {
    let image = png(b"only");
    let single_http = SequencedHttpClient::new([image_reply(0.5)]);
    let single = model(single_http.clone())
        .call(vec![image.clone()])
        .await
        .expect("single reply");
    let batch_http = SequencedHttpClient::new([image_reply(0.5)]);
    let batched = model(batch_http.clone())
        .embed_images_response(CohereImageBatch::new(image, []))
        .await
        .expect("batch reply");
    assert_eq!(single_http.requests(), batch_http.requests());
    assert_eq!(
        serde_json::to_value(single).expect("single JSON"),
        serde_json::to_value(batched).expect("batch JSON")
    );
    assert_eq!(batch_http.requests().len(), 1);
}

#[tokio::test]
async fn token_usage_sums_only_reported_token_counters() {
    let page = |id: &str, billed: serde_json::Value| {
        MockHttpResponse::success(
            json!({
                "id": id,
                "embeddings": {"float": [[0.5, 1.0]]},
                "meta": {"api_version": {"version":"1"}, "billed_units": billed},
            })
            .to_string(),
        )
    };
    let http = SequencedHttpClient::new([
        page("first", json!({"input_tokens":7,"images":1})),
        page("second", json!({"output_tokens":3,"images":1})),
        page("third", json!({"images":1})),
    ]);
    let response = model(http)
        .embed_images_response(CohereImageBatch::new(
            png(b"first"),
            [png(b"second"), png(b"third")],
        ))
        .await
        .expect("all pages decode");
    assert_eq!(response.usage.input_tokens, Some(7));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(10));
    assert_eq!(response.usage.cost, None);
    assert_eq!(response.response_id.as_deref(), Some("first"));
}

#[test]
fn an_empty_ordered_batch_is_refused() {
    assert!(CohereImageBatch::try_from(Vec::new()).is_err());
}
