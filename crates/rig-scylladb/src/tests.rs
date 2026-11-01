use super::{CqlValue, ScyllaDbVectorStore, ScyllaSearchFilter, SearchFilter, VectorStoreError};

/// Exercises the local scoring boundary without a ScyllaDB or embedding provider.
#[test]
fn cosine_similarity_rejects_unequal_lengths_in_both_orders() {
    for (left, right) in [
        (&[1.0, 1.0][..], &[1.0, 1.0, 1.0][..]),
        (&[1.0, 1.0, 1.0][..], &[1.0, 1.0][..]),
        (&[][..], &[0.0][..]),
        (&[0.0][..], &[][..]),
    ] {
        let result = ScyllaDbVectorStore::cosine_similarity(left, right);
        assert!(matches!(
            result,
            Err(VectorStoreError::DatastoreError(error))
                if error.to_string()
                    == "ScyllaDB search cosine similarity dimension mismatch (vector values not shown)"
        ));
    }
}

/// Equal-length inputs retain the cosine formula, including opposite directions.
#[test]
fn cosine_similarity_preserves_equal_length_scores() -> anyhow::Result<()> {
    for (left, right, expected) in [
        ([1.0, 2.0], [1.0, 2.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 2.0], [-1.0, -2.0], -1.0),
        ([1.0, 2.0], [2.0, 1.0], 0.8),
    ] {
        let score = ScyllaDbVectorStore::cosine_similarity(&left, &right)?;
        anyhow::ensure!((score - expected).abs() < 1e-6, "cosine score changed");
    }
    Ok(())
}

/// Equal-length zero-norm inputs, including empty vectors, retain zero similarity.
#[test]
fn cosine_similarity_preserves_zero_norm_scores() -> anyhow::Result<()> {
    for (left, right) in [
        (&[0.0, 0.0][..], &[1.0, 2.0][..]),
        (&[1.0, 2.0][..], &[0.0, 0.0][..]),
        (&[0.0, 0.0][..], &[0.0, 0.0][..]),
        (&[][..], &[][..]),
    ] {
        let score = ScyllaDbVectorStore::cosine_similarity(left, right)?;
        anyhow::ensure!(score == 0.0, "zero-norm cosine score changed");
    }
    Ok(())
}

/// CQL binds positionally, so the rendered condition must carry exactly one
/// `?` per parameter — including `IN`, which renders one per value.
#[test]
fn every_parameterised_operator_uses_question_mark_placeholders() {
    let filter = ScyllaSearchFilter::gte("price", CqlValue::BigInt(5))
        .and(ScyllaSearchFilter::member(
            "id",
            vec![CqlValue::BigInt(1), CqlValue::BigInt(2)],
        ))
        .or(ScyllaSearchFilter::ne("kind", CqlValue::Text("veg".to_string())).not());

    assert_eq!(
        filter.condition(),
        "((price >= ?) AND (id IN (?, ?))) OR (NOT (kind != ?))"
    );
    assert_eq!(filter.condition().matches('?').count(), 4);
    assert_eq!(filter.params().len(), 4);
}
