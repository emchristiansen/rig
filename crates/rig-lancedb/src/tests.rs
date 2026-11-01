use serde_json::{Value, json};

use super::{search_id_result, search_result};

#[test]
fn search_mappers_use_lancedb_distance_column() {
    let row = json!({"id": "record-1", "_distance": 0.375, "distance": 99.0});
    let document = search_result::<Value>(row.clone(), "id", 0).unwrap();
    let id = search_id_result(&row, "id");

    assert_eq!(document.score, 0.375);
    assert_eq!(id.score, document.score);
    assert_eq!(document.id, "record-1");
    assert_eq!(id.id, document.id);
    assert_eq!(document.document, row);
}

#[test]
fn search_mappers_preserve_missing_distance_and_id_fallbacks() {
    for row in [json!({}), json!({"_distance": null, "id": 7})] {
        let document = search_result::<Value>(row.clone(), "id", 3).unwrap();
        let id = search_id_result(&row, "id");

        assert_eq!(document.score, 0.0);
        assert_eq!(id.score, 0.0);
        assert_eq!(document.id, "unknown3");
        assert_eq!(id.id, "");
        assert_eq!(document.document, row);
    }
}
