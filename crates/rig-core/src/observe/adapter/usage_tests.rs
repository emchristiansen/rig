use super::{
    AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope, AdapterEvent,
    AdapterObservation, AdapterSlot, AdapterUsage, AdapterVerdict,
};
use crate::completion::Usage;
use crate::error::ProviderError;
use crate::observe::{Action, ObservationLog, Subject};
use std::sync::Arc;

fn installed() -> (Arc<ObservationLog>, AdapterSlot) {
    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::scoped("usage-run"), "usage-call");
    let slot = AdapterSlot::default();
    slot.install(context.attempt_for(&http::Request::new(()), "/usage"));
    (log, slot)
}

fn facts(log: &ObservationLog) -> Vec<AdapterObservation> {
    log.trace()
        .observations
        .into_iter()
        .map(|fact| {
            assert_eq!(fact.subject, Subject::scoped("usage-run"));
            let Action::Adapter { observation } = fact.action else {
                panic!("an adapter observation");
            };
            assert_eq!(observation.operation, "usage-call");
            assert_eq!(observation.attempt, Some(1));
            observation
        })
        .collect()
}

fn snapshot() -> AdapterUsage {
    AdapterUsage {
        input_tokens: Some(7),
        output_tokens: Some(0),
        total_tokens: None,
        cached_input_tokens: Some(3),
        reasoning_tokens: None,
        tool_input_tokens: Some(0),
    }
}

#[test]
fn usage_snapshot_precedes_its_provider_verdict_and_survives_terminal_without_normalized_usage() {
    let (log, slot) = installed();
    let usage = snapshot();
    let verdict = AdapterVerdict {
        finish_reason: Some("STOP".to_owned()),
        model: Some("reported-model".to_owned()),
        ..AdapterVerdict::default()
    };
    slot.project(|sink| {
        sink.usage_snapshot(usage.clone());
        sink.provider(verdict.clone(), Some("response-only".to_owned()));
    });
    let before_end = facts(&log);
    assert_eq!(before_end.len(), 3);
    assert_eq!(
        before_end[1].event,
        AdapterEvent::Usage {
            usage: usage.clone()
        }
    );
    assert_eq!(before_end[2].event, AdapterEvent::Provider { verdict });
    assert_eq!(
        before_end[2]
            .analysis
            .as_ref()
            .and_then(|analysis| analysis.response_id.as_deref()),
        Some("response-only")
    );
    slot.terminal(None);
    drop(slot);
    let completed = facts(&log);
    assert_eq!(&completed[..3], before_end.as_slice());
    assert_eq!(
        completed[3].event,
        AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        }
    );
    assert_eq!(
        completed
            .iter()
            .filter(|fact| matches!(fact.event, AdapterEvent::Usage { .. }))
            .count(),
        1
    );
}

#[test]
fn usage_snapshot_precedes_its_error_envelope_and_retains_the_original_tuple_after_failure() {
    let (log, slot) = installed();
    let usage = snapshot();
    let envelope = AdapterErrorEnvelope {
        code: Some("capacity".to_owned()),
        status: Some("UNAVAILABLE".to_owned()),
        message: Some("ordinary failure".to_owned()),
    };
    slot.project(|sink| {
        sink.usage_snapshot(usage.clone());
        sink.error(envelope.clone());
    });
    slot.fail(&ProviderError::from_http_response(
        http::StatusCode::SERVICE_UNAVAILABLE,
        "ordinary failure",
    ));
    drop(slot);
    let recorded = facts(&log);
    assert_eq!(recorded[1].event, AdapterEvent::Usage { usage });
    assert_eq!(
        recorded[2].event,
        AdapterEvent::ErrorEnvelope { error: envelope }
    );
    assert_eq!(recorded[3].event, AdapterEvent::Response { status: 503 });
    assert_eq!(
        recorded[4].event,
        AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".to_owned(),
                status: Some(503),
                retryable: true,
            }
        }
    );
    assert_eq!(recorded.len(), 5);
}

#[test]
fn snapshots_remain_observable_when_attempts_end_at_eof_partial_frame_error_or_drop() {
    for ending in [
        AdapterEnding::Eof { after: 2 },
        AdapterEnding::PartialFrame {
            byte_count: 13,
            after: 2,
        },
        AdapterEnding::Error {
            boundary: AdapterErrorBoundary::Request,
            kind: "request".to_owned(),
            status: None,
            retryable: false,
        },
        AdapterEnding::Dropped,
    ] {
        let (log, slot) = installed();
        slot.project(|sink| sink.usage_snapshot(snapshot()));
        match &ending {
            AdapterEnding::Eof { after } => slot.eof(*after),
            AdapterEnding::PartialFrame { after, .. } => {
                slot.bytes(b"data: partial");
                slot.eof(*after);
            }
            AdapterEnding::Error { .. } => {
                slot.fail(&ProviderError::request("ordinary request failure"))
            }
            AdapterEnding::Dropped => {}
            _ => unreachable!("the fixture lists incomplete endings"),
        }
        drop(slot);
        let recorded = facts(&log);
        assert_eq!(recorded.len(), 3);
        assert_eq!(recorded[1].event, AdapterEvent::Usage { usage: snapshot() });
        assert_eq!(recorded[2].event, AdapterEvent::Finished { ending });
        let trace = log.trace();
        let encoded = serde_json::to_string(&trace).expect("serialize preserved usage fact");
        assert_eq!(
            serde_json::from_str::<crate::observe::ObservationTrace>(&encoded)
                .expect("read usage trace"),
            trace
        );
    }
}

#[test]
fn absent_and_zero_counters_remain_distinct_exact_snapshots_and_total_stays_none() {
    let zero = Usage {
        input_tokens: Some(0),
        output_tokens: Some(0),
        total_tokens: Some(99),
        cached_input_tokens: Some(0),
        reasoning_tokens: Some(0),
        tool_use_prompt_tokens: Some(0),
        ..Usage::default()
    };
    let zero_snapshot = AdapterUsage {
        input_tokens: Some(0),
        output_tokens: Some(0),
        total_tokens: None,
        cached_input_tokens: Some(0),
        reasoning_tokens: Some(0),
        tool_input_tokens: Some(0),
    };
    assert_eq!(AdapterUsage::from(&zero), zero_snapshot);
    let expected = [AdapterUsage::default(), zero_snapshot, snapshot()];
    let (log, slot) = installed();
    for usage in &expected {
        slot.project(|sink| sink.usage_snapshot(usage.clone()));
    }
    drop(slot);
    let recorded = facts(&log);
    let usages: Vec<_> = recorded
        .iter()
        .filter_map(|fact| match &fact.event {
            AdapterEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(usages, expected);
    assert!(usages.iter().all(|usage| usage.total_tokens.is_none()));
    assert_ne!(usages[0], usages[1]);
    assert_eq!(
        serde_json::to_value(&usages).expect("serialize exact sparse tuples"),
        serde_json::json!([
            {"input_tokens":null,"output_tokens":null,"total_tokens":null,"cached_input_tokens":null,"reasoning_tokens":null,"tool_input_tokens":null},
            {"input_tokens":0,"output_tokens":0,"total_tokens":null,"cached_input_tokens":0,"reasoning_tokens":0,"tool_input_tokens":0},
            {"input_tokens":7,"output_tokens":0,"total_tokens":null,"cached_input_tokens":3,"reasoning_tokens":null,"tool_input_tokens":0},
        ])
    );
}
