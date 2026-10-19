use super::{Mem, SurrealDistanceFunction, SurrealSearchFilter, SurrealVectorStore};
use rig_core::{
    embeddings::Embedding,
    vector_store::{
        VectorStoreIndex,
        request::{DynamicSearchFilter, Filter},
    },
};
use serde_json::json;
use surrealdb::{Surreal, engine::local::Db};

/// A deterministic embedding runtime: every text embeds to three zeros.
#[derive(Clone)]
struct MockEmbeddingModel;

impl MockEmbeddingModel {
    fn model() -> rig_core::Model<rig_core::driver::Local<rig_core::operation::Embedding>, Self> {
        rig_core::Model::new(
            rig_core::driver::Local::new("mock")
                .with_capabilities(rig_core::wire::Capabilities::embedding(4, 3)),
            Self,
        )
    }
}

impl rig_core::driver::Transport<rig_core::driver::Local<rig_core::operation::Embedding>>
    for MockEmbeddingModel
{
    fn send(
        &self,
        texts: Vec<String>,
        _exchange: rig_core::driver::Exchange,
    ) -> rig_core::driver::Opening<rig_core::driver::Step<rig_core::operation::Embedding>> {
        let response = rig_core::embeddings::EmbeddingResponse {
            provider: "mock".into(),
            ..rig_core::embeddings::EmbeddingResponse::new(
                texts
                    .into_iter()
                    .map(|text| Embedding {
                        document: text,
                        vec: vec![0.0, 0.0, 0.0],
                    })
                    .collect(),
            )
        };
        rig_core::driver::Opening::ready(rig_core::driver::Opened::new(futures::stream::iter([
            Ok(rig_core::driver::Step::End(response)),
        ])))
    }
}

fn assert_query_direction(
    store: &SurrealVectorStore<Db>,
    function: &str,
    comparison: &str,
    order: &str,
) {
    for with_document in [false, true] {
        let projection = if with_document {
            "id , document , embedded_text,"
        } else {
            "id ,"
        };
        let expected = format!(
            "SELECT {projection} {function}($vec, embedding) as distance \
             from type::table($tablename) \
             where {function}($vec, embedding) {comparison} $threshold AND $filter \
             order by distance {order} LIMIT $limit"
        );
        let query = store
            .search_query(with_document, Some(0.5))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(query, expected);
    }
}

/// Checks generated queries without connecting to a database or embedding provider.
#[test]
fn explicit_distance_queries_use_maximum_threshold_and_ascending_order() {
    for (function, sql_function) in [
        (SurrealDistanceFunction::Knn, "vector::distance::knn"),
        (
            SurrealDistanceFunction::Hamming,
            "vector::distance::hamming",
        ),
        (
            SurrealDistanceFunction::Euclidean,
            "vector::distance::euclidean",
        ),
    ] {
        let store = SurrealVectorStore::new(
            MockEmbeddingModel::model(),
            Surreal::<Db>::init(),
            None,
            function,
        );
        assert_query_direction(&store, sql_function, "<=", "asc");
    }
}

/// An absent distance threshold leaves both projections free of a score cutoff.
#[test]
fn absent_distance_threshold_keeps_positive_distance_candidates() {
    for (function, sql_function) in [
        (SurrealDistanceFunction::Knn, "vector::distance::knn"),
        (
            SurrealDistanceFunction::Hamming,
            "vector::distance::hamming",
        ),
        (
            SurrealDistanceFunction::Euclidean,
            "vector::distance::euclidean",
        ),
    ] {
        let store = SurrealVectorStore::new(
            MockEmbeddingModel::model(),
            Surreal::<Db>::init(),
            None,
            function,
        );
        for with_document in [false, true] {
            let projection = if with_document {
                "id , document , embedded_text,"
            } else {
                "id ,"
            };
            let expected = format!(
                "SELECT {projection} {sql_function}($vec, embedding) as distance \
                 from type::table($tablename) where $filter \
                 order by distance asc LIMIT $limit"
            );
            let query = store
                .search_query(with_document, None)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(query, expected);
        }
    }
}

/// Checks the default similarity query without a database or embedding provider.
#[test]
fn default_cosine_query_keeps_minimum_threshold_and_descending_order() {
    let store =
        SurrealVectorStore::with_defaults(MockEmbeddingModel::model(), Surreal::<Db>::init());
    assert_query_direction(&store, "vector::similarity::cosine", ">=", "desc");
    for with_document in [false, true] {
        assert_eq!(
            store.search_query(with_document, None),
            store.search_query(with_document, Some(0.0))
        );
    }
}

/// Checks the other admitted similarity measure without contacting a database.
#[test]
fn jaccard_query_keeps_minimum_threshold_and_descending_order() {
    let store = SurrealVectorStore::new(
        MockEmbeddingModel::model(),
        Surreal::<Db>::init(),
        None,
        SurrealDistanceFunction::Jaccard,
    );
    assert_query_direction(&store, "vector::similarity::jaccard", ">=", "desc");
    for with_document in [false, true] {
        assert_eq!(
            store.search_query(with_document, None),
            store.search_query(with_document, Some(0.0))
        );
    }
}

#[allow(clippy::panic)]
#[test]
fn filter_from_json_preserves_nested_values() {
    let filter = match SurrealSearchFilter::from_dynamic_filter(Filter::Eq(
        "metadata".to_string(),
        json!({
            "name": "rig",
            "flags": { "native": true },
            "tags": ["surreal", "json"]
        }),
    )) {
        Ok(filter) => filter,
        Err(err) => panic!("unexpected surreal filter conversion failure: {err}"),
    };

    let sql = filter.to_string();

    assert!(sql.starts_with("metadata = {"));
    assert!(sql.contains("name: 'rig'"));
    assert!(sql.contains("flags: { native: true }"));
    assert!(sql.contains("tags: ['surreal', 'json']"));
}

#[allow(clippy::panic)]
#[tokio::test]
async fn surreal_vector_store_supports_type_erased_queries() {
    fn assert_dyn<T: VectorStoreIndex + Send + Sync + 'static>(_: T) {}

    let surreal = match Surreal::new::<Mem>(()).await {
        Ok(surreal) => surreal,
        Err(err) => panic!("failed to create in-memory surreal client: {err}"),
    };
    let vector_store = SurrealVectorStore::with_defaults(MockEmbeddingModel::model(), surreal);

    assert_dyn(vector_store);
}
