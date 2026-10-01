//! The function-call collector over decoded server events. These are unit
//! tests, not cassette tests: the collector is internal bookkeeping over
//! data channel events, which the HTTP cassette harness does not record. The
//! nested events follow the Responses streaming shapes the delegation guide
//! names.

use super::*;
use serde_json::{Value, json};

fn wrap(delegation_id: Option<&str>, nested: Value) -> ServerEvent {
    let payload = json!({
        "type": "response.event",
        "event_id": "evt",
        "delegation_id": delegation_id,
        "event": nested,
    });
    ServerEvent::parse(&payload.to_string()).expect("a server event")
}

fn created(delegation_id: Option<&str>, response_id: &str) -> ServerEvent {
    wrap(
        delegation_id,
        json!({"type": "response.created", "sequence_number": 0,
               "response": {"id": response_id, "output": [], "tools": [], "instructions": null}}),
    )
}

fn ended(delegation_id: Option<&str>, kind: &str, response_id: &str) -> ServerEvent {
    wrap(
        delegation_id,
        json!({"type": kind, "sequence_number": 9,
               "response": {"id": response_id, "output": [], "tools": [], "instructions": null}}),
    )
}

fn call_done(delegation_id: Option<&str>, call_id: &str, name: &str) -> ServerEvent {
    wrap(
        delegation_id,
        json!({"type": "response.output_item.done", "output_index": 0, "sequence_number": 4,
               "item": {"type": "function_call", "id": format!("fc_{call_id}"),
                        "call_id": call_id, "name": name, "arguments": "{}",
                        "status": "completed"}}),
    )
}

fn call(call_id: &str, name: &str) -> FunctionCall {
    FunctionCall {
        call_id: call_id.to_owned(),
        name: name.to_owned(),
        arguments: "{}".to_owned(),
        id: Some(format!("fc_{call_id}")),
    }
}

fn observe_all(collector: &mut FunctionCallCollector, events: &[ServerEvent]) -> Vec<CallsUpdate> {
    events
        .iter()
        .filter_map(|event| collector.observe(event))
        .collect()
}

#[test]
fn two_calls_then_completion_ask_for_both_outputs_then_continue() {
    let mut collector = FunctionCallCollector::new();
    let updates = observe_all(
        &mut collector,
        &[
            created(Some("del_1"), "resp_1"),
            call_done(Some("del_1"), "call_a", "lookup"),
            wrap(
                Some("del_1"),
                json!({"type": "response.function_call_arguments.done", "item_id": "fc_call_b"}),
            ),
            call_done(Some("del_1"), "call_b", "book"),
        ],
    );
    assert!(updates.is_empty());
    assert!(collector.is_collecting());

    let update = collector.observe(&ended(Some("del_1"), "response.completed", "resp_1"));
    let pending = PendingFunctionCalls {
        delegation_id: Some("del_1".to_owned()),
        response_id: "resp_1".to_owned(),
        calls: vec![call("call_a", "lookup"), call("call_b", "book")],
    };
    assert_eq!(update, Some(CallsUpdate::Submit(pending.clone())));
    assert!(!collector.is_collecting());

    let events = pending
        .submit([("call_b", "booked"), ("call_a", "found")])
        .expect("every call answered once");
    assert_eq!(
        events,
        vec![
            ClientEvent::FunctionCallOutput {
                call_id: "call_a".to_owned(),
                output: "found".to_owned(),
                event_id: None,
            },
            ClientEvent::FunctionCallOutput {
                call_id: "call_b".to_owned(),
                output: "booked".to_owned(),
                event_id: None,
            },
            ClientEvent::ResponseCreate { event_id: None },
        ]
    );
}

#[test]
fn a_response_without_calls_just_ends() {
    let mut collector = FunctionCallCollector::new();
    assert_eq!(collector.observe(&created(Some("del_1"), "resp_1")), None);
    assert_eq!(
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_1")),
        Some(CallsUpdate::Ended {
            delegation_id: Some("del_1".to_owned()),
            response_id: "resp_1".to_owned(),
            outcome: ResponseOutcome::Completed,
        })
    );
}

#[test]
fn calls_before_response_created_belong_to_that_response() {
    let mut collector = FunctionCallCollector::new();
    let updates = observe_all(
        &mut collector,
        &[
            call_done(Some("del_1"), "call_a", "lookup"),
            created(Some("del_1"), "resp_1"),
            call_done(Some("del_1"), "call_b", "book"),
        ],
    );
    assert!(updates.is_empty());
    let Some(CallsUpdate::Submit(pending)) =
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_1"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(pending.response_id, "resp_1");
    assert_eq!(
        pending.calls,
        vec![call("call_a", "lookup"), call("call_b", "book")]
    );
}

#[test]
fn a_completion_without_response_created_still_collects_its_calls() {
    let mut collector = FunctionCallCollector::new();
    collector.observe(&call_done(None, "call_a", "lookup"));
    let Some(CallsUpdate::Submit(pending)) =
        collector.observe(&ended(None, "response.completed", "resp_9"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(pending.delegation_id, None);
    assert_eq!(pending.response_id, "resp_9");
    assert_eq!(pending.calls, vec![call("call_a", "lookup")]);

    assert_eq!(
        collector.observe(&ended(None, "response.completed", "resp_unseen")),
        Some(CallsUpdate::Ended {
            delegation_id: None,
            response_id: "resp_unseen".to_owned(),
            outcome: ResponseOutcome::Completed,
        })
    );
}

#[test]
fn a_failed_or_incomplete_response_abandons_its_calls() {
    for (kind, outcome) in [
        ("response.failed", ResponseOutcome::Failed),
        ("response.incomplete", ResponseOutcome::Incomplete),
    ] {
        let mut collector = FunctionCallCollector::new();
        collector.observe(&created(Some("del_1"), "resp_1"));
        collector.observe(&call_done(Some("del_1"), "call_a", "lookup"));
        assert_eq!(
            collector.observe(&ended(Some("del_1"), kind, "resp_1")),
            Some(CallsUpdate::Abandoned {
                pending: PendingFunctionCalls {
                    delegation_id: Some("del_1".to_owned()),
                    response_id: "resp_1".to_owned(),
                    calls: vec![call("call_a", "lookup")],
                },
                outcome,
            })
        );
        assert!(!collector.is_collecting());
    }
}

#[test]
fn a_repeated_call_is_kept_once() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("del_1"), "resp_1"),
            call_done(Some("del_1"), "call_a", "lookup"),
            call_done(Some("del_1"), "call_a", "lookup"),
        ],
    );
    let Some(CallsUpdate::Submit(pending)) =
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_1"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(pending.calls, vec![call("call_a", "lookup")]);
}

#[test]
fn interleaved_delegations_keep_their_calls_apart() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("del_1"), "resp_1"),
            created(Some("del_2"), "resp_2"),
            call_done(Some("del_2"), "call_x", "weather"),
            call_done(Some("del_1"), "call_a", "lookup"),
        ],
    );
    let Some(CallsUpdate::Submit(second)) =
        collector.observe(&ended(Some("del_2"), "response.completed", "resp_2"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(second.calls, vec![call("call_x", "weather")]);
    let Some(CallsUpdate::Submit(first)) =
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_1"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(first.calls, vec![call("call_a", "lookup")]);
}

#[test]
fn a_continued_response_collects_its_own_calls() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("del_1"), "resp_1"),
            call_done(Some("del_1"), "call_a", "lookup"),
        ],
    );
    assert!(matches!(
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_1")),
        Some(CallsUpdate::Submit(_))
    ));
    observe_all(
        &mut collector,
        &[
            created(Some("del_1"), "resp_2"),
            call_done(Some("del_1"), "call_b", "book"),
        ],
    );
    let Some(CallsUpdate::Submit(pending)) =
        collector.observe(&ended(Some("del_1"), "response.completed", "resp_2"))
    else {
        panic!("expected calls to submit");
    };
    assert_eq!(pending.calls, vec![call("call_b", "book")]);
}

#[test]
fn other_events_change_nothing() {
    let mut collector = FunctionCallCollector::new();
    let transcript = ServerEvent::parse(
        r#"{"type":"session.output_transcript.delta","event_id":"e","delta":"Hi","start_ms":1,"end_ms":2}"#,
    )
    .expect("a server event");
    assert_eq!(collector.observe(&transcript), None);
    assert_eq!(
        collector.observe(&wrap(
            Some("del_1"),
            json!({"type": "response.output_text.delta", "delta": "x"})
        )),
        None
    );
    assert!(!collector.is_collecting());
}

#[test]
fn outputs_must_answer_every_call_exactly_once() {
    let pending = PendingFunctionCalls {
        delegation_id: Some("del_1".to_owned()),
        response_id: "resp_1".to_owned(),
        calls: vec![call("call_a", "lookup"), call("call_b", "book")],
    };
    assert_eq!(
        pending.submit([("call_a", "found")]),
        Err(MismatchedOutputs {
            missing: vec!["call_b".to_owned()],
            unknown: Vec::new(),
            repeated: Vec::new(),
        })
    );
    assert_eq!(
        pending.submit([
            ("call_a", "found"),
            ("call_b", "booked"),
            ("call_z", "stray"),
            ("call_a", "again"),
        ]),
        Err(MismatchedOutputs {
            missing: Vec::new(),
            unknown: vec!["call_z".to_owned()],
            repeated: vec!["call_a".to_owned()],
        })
    );
    assert_eq!(
        pending.submit(Vec::<(String, String)>::new()),
        Err(MismatchedOutputs {
            missing: vec!["call_a".to_owned(), "call_b".to_owned()],
            unknown: Vec::new(),
            repeated: Vec::new(),
        })
    );
}

#[test]
fn an_uncorrelated_call_with_one_open_response_belongs_to_it() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[created(Some("d1"), "r1"), call_done(None, "c1", "lookup")],
    );
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r1")),
        Some(CallsUpdate::Submit(PendingFunctionCalls {
            delegation_id: Some("d1".to_owned()),
            response_id: "r1".to_owned(),
            calls: vec![call("c1", "lookup")],
        }))
    );
    assert!(!collector.is_collecting());
}

#[test]
fn correlation_present_on_only_some_events_still_resolves() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[created(None, "r1"), call_done(Some("d1"), "c1", "lookup")],
    );
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r1")),
        Some(CallsUpdate::Submit(PendingFunctionCalls {
            delegation_id: Some("d1".to_owned()),
            response_id: "r1".to_owned(),
            calls: vec![call("c1", "lookup")],
        }))
    );

    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[call_done(None, "c2", "book"), created(Some("d2"), "r2")],
    );
    assert_eq!(
        collector.observe(&ended(None, "response.completed", "r2")),
        Some(CallsUpdate::Submit(PendingFunctionCalls {
            delegation_id: Some("d2".to_owned()),
            response_id: "r2".to_owned(),
            calls: vec![call("c2", "book")],
        }))
    );
}

#[test]
fn an_uncorrelated_call_with_two_open_responses_stays_unresolved() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("d1"), "r1"),
            created(Some("d2"), "r2"),
            call_done(None, "c1", "lookup"),
        ],
    );
    assert!(collector.is_collecting());
    assert_eq!(
        collector.unresolved_calls().cloned().collect::<Vec<_>>(),
        vec![call("c1", "lookup")]
    );
    for (delegation, response) in [("d1", "r1"), ("d2", "r2")] {
        assert_eq!(
            collector.observe(&ended(Some(delegation), "response.completed", response)),
            Some(CallsUpdate::Unresolved {
                delegation_id: Some(delegation.to_owned()),
                response_id: response.to_owned(),
                outcome: ResponseOutcome::Completed,
                owned: Vec::new(),
                uncertain: vec![call("c1", "lookup")],
            })
        );
    }
    assert_eq!(collector.take_unresolved(), vec![call("c1", "lookup")]);
    assert!(!collector.is_collecting());
}

#[test]
fn two_open_responses_of_one_delegation_leave_its_call_unresolved() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("d1"), "r1"),
            call_done(Some("d1"), "a", "lookup"),
            created(Some("d1"), "r2"),
            created(Some("d2"), "r3"),
            call_done(Some("d1"), "b", "book"),
            call_done(Some("d2"), "x", "weather"),
        ],
    );
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r1")),
        Some(CallsUpdate::Unresolved {
            delegation_id: Some("d1".to_owned()),
            response_id: "r1".to_owned(),
            outcome: ResponseOutcome::Completed,
            owned: vec![call("a", "lookup")],
            uncertain: vec![call("b", "book")],
        })
    );
    assert_eq!(
        collector.observe(&ended(Some("d2"), "response.completed", "r3")),
        Some(CallsUpdate::Submit(PendingFunctionCalls {
            delegation_id: Some("d2".to_owned()),
            response_id: "r3".to_owned(),
            calls: vec![call("x", "weather")],
        }))
    );
    assert!(matches!(
        collector.observe(&ended(Some("d1"), "response.completed", "r2")),
        Some(CallsUpdate::Unresolved { uncertain, .. }) if uncertain == vec![call("b", "book")]
    ));
}

#[test]
fn a_known_correlation_is_never_claimed_by_another_delegation() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            call_done(None, "c0", "lookup"),
            call_done(Some("d1"), "c1", "book"),
            created(Some("d2"), "r2"),
        ],
    );
    assert_eq!(
        collector.observe(&ended(Some("d2"), "response.completed", "r2")),
        Some(CallsUpdate::Unresolved {
            delegation_id: Some("d2".to_owned()),
            response_id: "r2".to_owned(),
            outcome: ResponseOutcome::Completed,
            owned: Vec::new(),
            uncertain: vec![call("c0", "lookup")],
        })
    );
    assert_eq!(
        collector.unresolved_calls().cloned().collect::<Vec<_>>(),
        vec![call("c0", "lookup")]
    );

    observe_all(&mut collector, &[created(Some("d1"), "r1")]);
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r1")),
        Some(CallsUpdate::Submit(PendingFunctionCalls {
            delegation_id: Some("d1".to_owned()),
            response_id: "r1".to_owned(),
            calls: vec![call("c1", "book")],
        }))
    );
}

#[test]
fn an_unannounced_response_reports_only_retrievable_unresolved_calls() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            call_done(Some("d1"), "c1", "lookup"),
            call_done(Some("d2"), "c2", "book"),
        ],
    );
    let both = vec![call("c1", "lookup"), call("c2", "book")];
    assert_eq!(
        collector.observe(&ended(None, "response.completed", "r1")),
        Some(CallsUpdate::Unresolved {
            delegation_id: None,
            response_id: "r1".to_owned(),
            outcome: ResponseOutcome::Completed,
            owned: Vec::new(),
            uncertain: both.clone(),
        })
    );
    assert_eq!(
        collector.unresolved_calls().cloned().collect::<Vec<_>>(),
        both
    );

    // Reported calls are not attributed again before they are drained.
    observe_all(&mut collector, &[created(Some("d1"), "r2")]);
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r2")),
        Some(CallsUpdate::Ended {
            delegation_id: Some("d1".to_owned()),
            response_id: "r2".to_owned(),
            outcome: ResponseOutcome::Completed,
        })
    );

    assert_eq!(collector.take_unresolved(), both);
    assert_eq!(collector.unresolved_calls().count(), 0);
    assert!(!collector.is_collecting());
    assert_eq!(
        collector.observe(&ended(None, "response.completed", "r3")),
        Some(CallsUpdate::Ended {
            delegation_id: None,
            response_id: "r3".to_owned(),
            outcome: ResponseOutcome::Completed,
        })
    );
}

#[test]
fn a_drained_ambiguous_call_is_not_reported_again() {
    let mut collector = FunctionCallCollector::new();
    observe_all(
        &mut collector,
        &[
            created(Some("d1"), "r1"),
            created(Some("d2"), "r2"),
            call_done(None, "c1", "lookup"),
        ],
    );
    assert_eq!(collector.take_unresolved(), vec![call("c1", "lookup")]);
    assert_eq!(
        collector.observe(&ended(Some("d1"), "response.completed", "r1")),
        Some(CallsUpdate::Ended {
            delegation_id: Some("d1".to_owned()),
            response_id: "r1".to_owned(),
            outcome: ResponseOutcome::Completed,
        })
    );
}

#[test]
fn a_response_learns_its_delegation_from_a_call_it_owns() {
    let mut collector = FunctionCallCollector::new();
    let updates = observe_all(
        &mut collector,
        &[
            created(None, "r1"),
            call_done(Some("d1"), "c1", "lookup"),
            created(Some("d2"), "r2"),
            call_done(Some("d2"), "c2", "book"),
            ended(Some("d1"), "response.completed", "r1"),
            ended(Some("d2"), "response.completed", "r2"),
        ],
    );
    assert_eq!(
        updates,
        vec![
            CallsUpdate::Submit(PendingFunctionCalls {
                delegation_id: Some("d1".to_owned()),
                response_id: "r1".to_owned(),
                calls: vec![call("c1", "lookup")],
            }),
            CallsUpdate::Submit(PendingFunctionCalls {
                delegation_id: Some("d2".to_owned()),
                response_id: "r2".to_owned(),
                calls: vec![call("c2", "book")],
            }),
        ]
    );
    assert_eq!(collector.unresolved_calls().count(), 0);
}
