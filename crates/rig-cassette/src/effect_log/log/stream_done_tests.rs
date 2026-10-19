use super::*;
use futures::StreamExt;
use rig_core::{
    completion::FinishReason,
    error::{ErrorDetail, MalformedToolInput},
    message::{Opaque, Origin, ProviderItem, ToolCall, ToolFunction, ToolName},
    serve::{Dispatch, Recorder, Serve},
    streaming::{Item, Relayed, Transcript},
};
use serde_json::json;

fn response(label: &str) -> CompletionResponse {
    let mut origin = Origin::new("done.api", "done-provider", "requested-model");
    origin.response_id = Some(format!("response-{label}"));
    origin.response_model = Some(format!("reported-{label}"));
    let mut usage = Usage::default();
    usage.input_tokens = Some(3);
    usage.output_tokens = Some(5);
    usage.total_tokens = Some(8);
    usage.cached_input_tokens = Some(1);
    usage.cache_creation_input_tokens = Some(1);
    usage.tool_use_prompt_tokens = Some(1);
    usage.reasoning_tokens = Some(2);
    let mut response = CompletionResponse::new(
        vec![AssistantContent::text(label)],
        usage,
        origin,
        json!({"label": label, "nested": {"local": "opaque"}}),
    )
    .with_finish_reason(FinishReason::Length)
    .accept_unknown_finish_reasons(true);
    response.message_id = rig_core::completion::ProviderMessageId::new(format!("message-{label}"));
    response.provider_request_id = Some(format!("request-{label}"));
    response
        .provider_response_headers
        .insert("x-done".into(), label.into());
    response.error = Some(format!("provider-state-{label}"));
    response.aborted = Some(format!("caller-state-{label}"));
    response
}

fn begin(recorder: &EffectLogRecorder, id: EffectId) {
    recorder.begin(
        id,
        "done-model".into(),
        EffectKind::Completion {
            request: request(),
            stream: true,
        },
        rig_core::serve::Origin::default(),
    );
}

fn kept_log() -> EffectLog {
    let recorder = EffectLogRecorder::keeping_stream_events();
    let id = EffectId::from_raw(7);
    begin(&recorder, id);
    let first = ErrorReport::new(ErrorKind::Response, "first");
    recorder.stream_error(id, &first);
    recorder.resolve(id, Err(first));
    recorder.stream_done(id, &response("first response"));
    recorder.event(id, &Item::Unknown(json!({"received": "between"}).into()));
    recorder.stream_error(id, &ErrorReport::new(ErrorKind::Provider, "late"));
    recorder.stream_done(id, &response("second response"));
    recorder.log()
}

#[tokio::test]
async fn positioned_done_responses_replay_with_events_and_errors_in_both_lookup_modes()
-> TestResult<()> {
    let log = kept_log();
    let restored: EffectLog = serde_json::from_value(serde_json::to_value(&log)?)?;
    assert_eq!(restored.header, log.header);
    let id = log.records[0].id;
    let expected = vec![
        Err(ErrorReport::new(ErrorKind::Response, "first")),
        Ok(Relayed::Done(Box::new(response("first response")))),
        Ok(Relayed::Item(Item::Unknown(
            json!({"received": "between"}).into(),
        ))),
        Err(ErrorReport::new(ErrorKind::Provider, "late")),
        Ok(Relayed::Done(Box::new(response("second response")))),
    ];
    assert_eq!(
        log.header.stream_errors[&id]
            .iter()
            .map(|e| e.item)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );
    assert_eq!(
        log.header.stream_done[&id]
            .iter()
            .map(|d| d.item)
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
    for replay in [
        EffectLogReplayer::for_key(&restored, &log.records[0].key)?,
        EffectLogReplayer::for_key_by_id(&restored, &log.records[0].key)?,
    ] {
        let items = replay
            .serve(log.records[0].kind.clone(), Dispatch::new(id, true))
            .await
            .into_stream()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(items, expected);
    }
    assert_eq!(
        log.records[0].outcome.as_ref().err(),
        Some(&ErrorReport::new(ErrorKind::Response, "first"))
    );
    Ok(())
}

#[test]
fn done_metadata_follows_snapshots_checkpoint_tails_take_and_discard() -> TestResult<()> {
    let recorder = EffectLogRecorder::keeping_stream_events();
    let id = EffectId::from_raw(7);
    let pending = EffectId::from_raw(8);
    begin(&recorder, id);
    begin(&recorder, pending);
    let complete = response("complete");
    recorder.stream_done(id, &complete);
    recorder.resolve(id, Ok(Outcome::Completion(complete.clone())));
    recorder.stream_error(id, &ErrorReport::new(ErrorKind::Provider, "late"));
    recorder.stream_done(pending, &response("pending"));
    let snapshot = recorder.log();
    assert_eq!(snapshot.records.len(), 1);
    assert_eq!(snapshot.header.stream_done[&id][0].response, complete);
    assert!(!snapshot.header.stream_done.contains_key(&pending));
    assert!(recorder.header().stream_done.contains_key(&pending));
    let (checkpoint, tail) = snapshot.checkpoint(0, ());
    assert_eq!(
        EffectLog::from_checkpoint(&checkpoint, tail)?.header,
        snapshot.header
    );
    let (_, empty) = snapshot.checkpoint(1, ());
    assert!(empty.header.stream_done.is_empty() && empty.header.stream_errors.is_empty());
    let taken = recorder.take();
    assert_eq!(taken.header, snapshot.header);
    assert!(!recorder.header().stream_done.contains_key(&id));
    assert!(!recorder.header().stream_errors.contains_key(&id));
    assert_eq!(
        recorder.header().stream_done[&pending][0].response,
        response("pending")
    );
    recorder.stream_done(id, &response("after take"));
    recorder.stream_error(id, &ErrorReport::new(ErrorKind::Provider, "after take"));
    assert!(!recorder.header().stream_done.contains_key(&id));
    assert!(!recorder.header().stream_errors.contains_key(&id));
    recorder.discard(pending);
    recorder.stream_done(pending, &response("after discard"));
    recorder.stream_error(
        pending,
        &ErrorReport::new(ErrorKind::Provider, "after discard"),
    );
    assert!(recorder.header().stream_done.is_empty() && recorder.header().stream_errors.is_empty());
    assert!(recorder.log().records.is_empty());
    Ok(())
}

#[test]
fn taking_a_resolved_stream_keeps_only_observations_available_then() {
    let recorder = EffectLogRecorder::keeping_stream_events();
    let id = EffectId::from_raw(1);
    begin(&recorder, id);
    let first = ErrorReport::new(ErrorKind::Response, "first");
    recorder.stream_error(id, &first);
    recorder.resolve(id, Err(first.clone()));
    let taken = recorder.take();
    assert_eq!(taken.records[0].outcome.as_ref().err(), Some(&first));
    assert!(taken.header.stream_done.is_empty());
    recorder.stream_done(id, &response("later"));
    recorder.stream_error(id, &ErrorReport::new(ErrorKind::Provider, "later"));
    assert!(recorder.log().records.is_empty());
    assert!(recorder.header().stream_done.is_empty() && recorder.header().stream_errors.is_empty());
}

#[test]
fn kept_stream_positions_require_matching_records_and_distinct_ordered_positions() -> TestResult<()>
{
    let log = kept_log();
    EffectLogReplayer::check_header(&log)?;
    let id = log.records[0].id;
    for case in 0..9 {
        let mut invalid = log.clone();
        match case {
            0 => invalid.records.clear(),
            1 => invalid.records[0].events = None,
            2 => {
                invalid.records[0].kind = EffectKind::Completion {
                    request: request(),
                    stream: false,
                }
            }
            3 => invalid.header.stream_done.get_mut(&id).unwrap()[0].item = usize::MAX,
            4 => invalid.header.stream_done.get_mut(&id).unwrap()[1].item = 1,
            5 => invalid.header.stream_errors.get_mut(&id).unwrap()[1].item = 0,
            6 => invalid.header.stream_done.get_mut(&id).unwrap()[0].item = 0,
            7 => invalid.header.stream_done.get_mut(&id).unwrap().reverse(),
            8 => invalid.header.stream_errors.get_mut(&id).unwrap().reverse(),
            _ => unreachable!(),
        }
        let error =
            EffectLogReplayer::check_header(&invalid).expect_err("invalid kept-stream metadata");
        assert_eq!(error.kind, ErrorKind::Divergence, "case {case}");
    }
    Ok(())
}

#[tokio::test]
async fn absent_done_metadata_keeps_legacy_success_and_error_inference() -> TestResult<()> {
    let mut log = kept_log();
    let id = log.records[0].id;
    log.records[0].events = Some(Transcript::from_items(vec![])?);
    log.records[0].outcome = Ok(Outcome::Completion(response("legacy")));
    log.header.stream_done.clear();
    log.header.stream_errors.insert(
        id,
        vec![RecordedStreamError {
            item: 0,
            error: ErrorReport::new(ErrorKind::Provider, "late"),
        }],
    );
    let serialized = serde_json::to_value(&log)?;
    assert!(serialized.pointer("/header/stream_done").is_none());
    let restored: EffectLog = serde_json::from_value(serialized)?;
    let replay = EffectLogReplayer::for_key(&restored, &log.records[0].key)?;
    let items = replay
        .serve(log.records[0].kind.clone(), Dispatch::new(id, true))
        .await
        .into_stream()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(
        items,
        vec![
            Ok(Relayed::Done(Box::new(response("legacy")))),
            Err(ErrorReport::new(ErrorKind::Provider, "late")),
        ]
    );
    Ok(())
}

#[test]
fn canonical_local_ids_changes_only_typed_done_response_leaves() -> TestResult<()> {
    let (request, ids) = fresh_local_correlation_request()?;
    let Message::Assistant(turn) = &request.chat_history[0] else {
        panic!("assistant fixture");
    };
    let text = ids[0].wire().into_owned();
    let opaque = json!({"local": text, "nested": {"local": ids[1].wire()}, "text": format!("before-{text}-after")});
    let mut complete = response("local");
    complete.choice = turn.content.clone();
    complete
        .choice
        .push(AssistantContent::ToolCall(ToolCall::from_wire(
            text.clone(),
            ToolFunction::new(ToolName::new("provider")?, opaque.clone()),
        )));
    complete
        .choice
        .push(AssistantContent::text(text.clone()).with_native(opaque.clone()));
    complete.choice.push(AssistantContent::Opaque(Opaque {
        item: opaque.clone(),
        replay: true,
    }));
    complete
        .choice
        .push(AssistantContent::ProviderItem(ProviderItem::new(
            opaque.clone(),
            "done-provider",
        )));
    complete.raw = opaque.clone();
    let mut error = ErrorReport::new(ErrorKind::Json, text.clone());
    error.detail = Some(ErrorDetail::MalformedToolInput(MalformedToolInput {
        name: text.clone(),
        id: ids[0].clone(),
        provider: None,
        raw: opaque.to_string(),
        error: text,
    }));
    let id = EffectId::from_raw(7);
    let mut log = EffectLog::from_records(vec![EffectRecord {
        id,
        key: "done-model".into(),
        kind: EffectKind::Completion {
            request,
            stream: true,
        },
        outcome: Err(error.clone()),
        events: Some(Transcript::from_items(vec![Item::Unknown(opaque.into())])?),
        tool_output: None,
        stream_origin: None,
        parent: None,
        scope: None,
    }]);
    log.header
        .stream_errors
        .insert(id, vec![RecordedStreamError { item: 0, error }]);
    log.header.stream_done.insert(
        id,
        vec![
            RecordedStreamDone {
                item: 1,
                response: complete.clone(),
            },
            RecordedStreamDone {
                item: 2,
                response: complete,
            },
        ],
    );
    let original = serde_json::to_value(log)?;
    let mut expected = original.clone();
    for (canonical, paths) in [
        (
            "00000000-0000-4000-8000-000000000000",
            vec![
                "/header/stream_errors/7/0/error/detail/id/local",
                "/header/stream_done/7/0/response/choice/0/id/local",
                "/header/stream_done/7/1/response/choice/0/id/local",
                "/records/0/kind/request/chat_history/0/content/0/id/local",
                "/records/0/kind/request/chat_history/1/content/1/call/local",
                "/records/0/outcome/Err/detail/id/local",
            ],
        ),
        (
            "00000000-0000-4000-8000-000000000001",
            vec![
                "/header/stream_done/7/0/response/choice/1/id/local",
                "/header/stream_done/7/1/response/choice/1/id/local",
                "/records/0/kind/request/chat_history/0/content/1/id/local",
                "/records/0/kind/request/chat_history/1/content/0/call/local",
            ],
        ),
    ] {
        for path in paths {
            *expected
                .pointer_mut(path)
                .unwrap_or_else(|| panic!("missing typed location {path}")) = canonical.into();
        }
    }
    let normalized = canonical_local_ids(original);
    assert_eq!(
        normalized, expected,
        "every nonidentity value remains exact"
    );
    let _: EffectLog = serde_json::from_value(normalized.clone())?;
    assert_eq!(canonical_local_ids(normalized.clone()), normalized);
    Ok(())
}
