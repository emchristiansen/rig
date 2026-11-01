use super::{HelixDBFilter, HelixDBVectorStore, QueryResult};
use rig_core::vector_store::{VectorStoreError, request::VectorSearchRequest};
use serde_json::json;

fn hit(id: &str, similarity: f64, payload: &str) -> QueryResult {
    QueryResult {
        id: id.to_owned(),
        score: 1.0 - similarity,
        doc: "source text".to_owned(),
        json_payload: payload.to_owned(),
    }
}

/// Exercises shared response selection without a HelixDB or embedding provider.
#[test]
fn shared_selection_applies_payload_filter_and_optional_threshold() -> anyhow::Result<()> {
    let req = VectorSearchRequest::builder()
        .query("query")
        .samples(5)
        .threshold(0.75)
        .filter(HelixDBFilter::Eq("kind".to_owned(), json!("keep")))
        .build();
    let selected = HelixDBVectorStore::select_results(
        vec![
            hit("best", 1.0, r#"{"kind":"keep"}"#),
            hit("wrong-kind", 1.0, r#"{"kind":"drop"}"#),
            hit("below", 0.5, r#"{"kind":"keep"}"#),
            hit("boundary", 0.75, r#"{"kind":"keep"}"#),
        ],
        &req,
    )?;
    let ids: Vec<_> = selected.iter().map(|result| result.id.as_str()).collect();
    anyhow::ensure!(
        ids == ["best", "boundary"],
        "selected rows changed: {ids:?}"
    );
    Ok(())
}

/// An absent threshold adds no local cutoff, including for negative similarity.
#[test]
fn shared_selection_without_threshold_keeps_negative_similarity() -> anyhow::Result<()> {
    let req = VectorSearchRequest::builder()
        .query("query")
        .samples(1)
        .filter(HelixDBFilter::Eq("kind".to_owned(), json!("keep")))
        .build();
    let selected = HelixDBVectorStore::select_results(
        vec![hit("negative", -0.5, r#"{"kind":"keep"}"#)],
        &req,
    )?;
    anyhow::ensure!(selected.len() == 1, "negative similarity was discarded");
    anyhow::ensure!(selected[0].id == "negative", "selected id changed");
    Ok(())
}

/// A present filter cannot be evaluated against a malformed JSON payload.
#[test]
fn shared_selection_rejects_malformed_payload_when_filter_is_present() {
    let req = VectorSearchRequest::builder()
        .query("query")
        .samples(1)
        .filter(HelixDBFilter::Eq("kind".to_owned(), json!("keep")))
        .build();
    let selected =
        HelixDBVectorStore::select_results(vec![hit("malformed", 1.0, "not JSON")], &req);
    assert!(matches!(selected, Err(VectorStoreError::JsonError(_))));
}

/// ID-only searches need not decode payloads when no filter was requested.
#[test]
fn shared_selection_without_filter_preserves_malformed_payload() -> anyhow::Result<()> {
    let req = VectorSearchRequest::builder()
        .query("query")
        .samples(1)
        .build();
    let selected =
        HelixDBVectorStore::select_results(vec![hit("malformed", -0.5, "not JSON")], &req)?;
    anyhow::ensure!(selected.len() == 1, "unfiltered row was discarded");
    anyhow::ensure!(selected[0].json_payload == "not JSON", "payload changed");
    Ok(())
}

/// Threshold rejection happens before payload decoding for filter evaluation.
#[test]
fn shared_selection_checks_threshold_before_filter_payload() -> anyhow::Result<()> {
    let req = VectorSearchRequest::builder()
        .query("query")
        .samples(1)
        .threshold(0.75)
        .filter(HelixDBFilter::Eq("kind".to_owned(), json!("keep")))
        .build();
    let selected = HelixDBVectorStore::select_results(vec![hit("below", 0.5, "not JSON")], &req)?;
    anyhow::ensure!(selected.is_empty(), "below-threshold row was retained");
    Ok(())
}
