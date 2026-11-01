use rig_core::effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome};

use super::EffectLog;

fn record(id: u64) -> EffectRecord {
    EffectRecord {
        stream_origin: None,
        tool_output: None,
        parent: None,
        scope: None,
        id: EffectId::from_raw(id),
        key: HandlerKey::from("model"),
        kind: EffectKind::Custom {
            kind: std::sync::Arc::from("test"),
            payload: serde_json::json!({"ask": id}),
        },
        outcome: Ok(Outcome::Custom {
            payload: serde_json::json!({"n": id}),
        }),
        events: None,
    }
}

fn records() -> Vec<EffectRecord> {
    vec![record(1), record(2)]
}

/// Converting a vector or collecting an iterator builds the log
/// `from_records` builds, header included; an owned log yields its records
/// in order.
#[test]
fn conversions_build_and_yield_the_records_of_from_records() {
    let expected = EffectLog::from_records(records());
    let from_vec = EffectLog::from(records());
    let collected: EffectLog = records().into_iter().collect();
    for log in [&from_vec, &collected] {
        assert_eq!(log.header, expected.header);
        assert_eq!(
            serde_json::to_value(&log.records).expect("serializes"),
            serde_json::to_value(&expected.records).expect("serializes")
        );
    }
    let ids: Vec<EffectId> = IntoIterator::into_iter(from_vec)
        .map(|record| record.id)
        .collect();
    assert_eq!(ids, vec![EffectId::from_raw(1), EffectId::from_raw(2)]);
}
