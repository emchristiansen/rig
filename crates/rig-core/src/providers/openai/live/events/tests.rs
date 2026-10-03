//! Server event decoding and client event serialization. These are unit
//! tests, not cassette tests: the events travel on a WebRTC data channel,
//! which the HTTP cassette harness does not record. Server samples are the
//! Live API reference's examples unless a test says otherwise; client
//! expectations are the reference and delegation guide examples.

use super::*;
use crate::providers::openai::live::{MaxOutputTokens, ResponsesSettings};
use serde_json::json;

fn parse(payload: &str) -> ServerEvent {
    ServerEvent::parse(payload).expect("a server event")
}

fn to_json(event: &ClientEvent) -> Value {
    serde_json::to_value(event).expect("serializable")
}

#[test]
fn session_started_decodes_the_reference_example() {
    let event = parse(
        r#"{
  "type": "session.started",
  "event_id": "evt_started_001",
  "client_event_id": "evt_start_001",
  "session": {
    "id": "live_abc123",
    "model": "gpt-live-1",
    "status": "active",
    "expires_at": 1788555600,
    "instructions": "Help the caller plan a restaurant reservation. Confirm details before booking.",
    "input": [],
    "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
    "delegation": {"type": "client"}
  }
}"#,
    );
    let ServerEvent::SessionStarted(started) = event else {
        panic!("expected session.started, got {event:?}");
    };
    assert_eq!(started.event_id.as_deref(), Some("evt_started_001"));
    assert_eq!(started.client_event_id.as_deref(), Some("evt_start_001"));
    assert_eq!(started.session.id, "live_abc123");
    assert_eq!(started.session.model.as_deref(), Some("gpt-live-1"));
    assert_eq!(started.session.status.as_deref(), Some("active"));
    assert_eq!(started.session.expires_at, Some(1_788_555_600));
    assert_eq!(
        started.session.extra.get("delegation"),
        Some(&json!({"type": "client"}))
    );
    assert_eq!(
        started.session.extra.get("audio"),
        Some(
            &json!({"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}})
        )
    );
}

#[test]
fn session_updated_keeps_the_resolved_delegation() {
    let event = parse(
        r#"{
  "type": "session.updated",
  "event_id": "evt_updated_001",
  "client_event_id": "evt_update_001",
  "session": {
    "id": "live_def456",
    "model": "gpt-live-1",
    "status": "active",
    "expires_at": 1788555600,
    "instructions": "Help the caller plan a restaurant reservation. Confirm details before booking.",
    "input": [],
    "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
    "delegation": {
      "type": "responses",
      "responses": {
        "model": "gpt-6-astra",
        "instructions": "Check restaurant availability. Ask before confirming a booking.",
        "max_output_tokens": 1024,
        "tools": []
      }
    }
  }
}"#,
    );
    let ServerEvent::SessionUpdated(updated) = event else {
        panic!("expected session.updated, got {event:?}");
    };
    assert_eq!(updated.session.id, "live_def456");
    assert_eq!(
        updated.session.extra["delegation"]["responses"]["model"],
        json!("gpt-6-astra")
    );
}

#[test]
fn session_closed_decodes_reason_usage_and_snapshot() {
    let event = parse(
        r#"{
  "type": "session.closed",
  "event_id": "evt_closed_001",
  "client_event_id": "evt_close_001",
  "reason": "close_requested",
  "session": {
    "id": "live_abc123",
    "model": "gpt-live-1",
    "status": "active",
    "expires_at": 1788555600,
    "instructions": "Help the caller plan a restaurant reservation. Confirm details before booking.",
    "input": [],
    "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
    "delegation": {"type": "client"}
  },
  "usage": {"seconds": 45.8}
}"#,
    );
    let ServerEvent::SessionClosed(closed) = event else {
        panic!("expected session.closed, got {event:?}");
    };
    assert_eq!(closed.reason, CloseReason::CloseRequested);
    assert!((closed.usage.seconds - 45.8).abs() < f64::EPSILON);
    assert_eq!(
        closed.session.as_ref().map(|session| session.id.as_str()),
        Some("live_abc123")
    );
    assert_eq!(closed.client_event_id.as_deref(), Some("evt_close_001"));
}

#[test]
fn close_reasons_decode_with_unknown_ones_kept() {
    for (wire, reason) in [
        ("close_requested", CloseReason::CloseRequested),
        ("expired", CloseReason::Expired),
        ("content", CloseReason::Content),
        ("remote_hangup", CloseReason::RemoteHangup),
        ("connection_lost", CloseReason::ConnectionLost),
        ("billing", CloseReason::Other("billing".to_owned())),
    ] {
        let event = parse(&format!(
            r#"{{"type":"session.closed","event_id":"e","reason":"{wire}","usage":{{"seconds":1}}}}"#
        ));
        let ServerEvent::SessionClosed(closed) = event else {
            panic!("expected session.closed, got {event:?}");
        };
        assert_eq!(closed.reason, reason);
        assert_eq!(closed.session, None);
    }
}

#[test]
fn transcript_deltas_decode_both_directions() {
    let input = parse(
        r#"{
  "type": "session.input_transcript.delta",
  "event_id": "evt_input_transcript_001",
  "delta": "A table for two at seven, please.",
  "start_ms": 1600,
  "end_ms": 3400
}"#,
    );
    let ServerEvent::InputTranscriptDelta(delta) = input else {
        panic!("expected an input transcript delta, got {input:?}");
    };
    assert_eq!(delta.delta, "A table for two at seven, please.");
    assert_eq!((delta.start_ms, delta.end_ms), (1600, 3400));

    let output = parse(
        r#"{
  "type": "session.output_transcript.delta",
  "event_id": "evt_output_transcript_001",
  "delta": "Would you like me to reserve that table?",
  "start_ms": 5400,
  "end_ms": 7200
}"#,
    );
    let ServerEvent::OutputTranscriptDelta(delta) = output else {
        panic!("expected an output transcript delta, got {output:?}");
    };
    assert_eq!(delta.delta, "Would you like me to reserve that table?");
    assert_eq!((delta.start_ms, delta.end_ms), (5400, 7200));
    assert_eq!(delta.event_id.as_deref(), Some("evt_output_transcript_001"));
}

#[test]
fn delegation_created_decodes_client_and_responses_targets() {
    let client = parse(
        r#"{
  "type": "session.delegation.created",
  "event_id": "evt_delegation_001",
  "offset_ms": 3600,
  "delegation": {"id": "del_abc123", "type": "delegation", "target": "client"}
}"#,
    );
    let ServerEvent::DelegationCreated(created) = client else {
        panic!("expected session.delegation.created, got {client:?}");
    };
    assert_eq!(created.offset_ms, 3600);
    assert_eq!(created.delegation.id, "del_abc123");
    assert_eq!(created.delegation.target, DelegationTarget::Client);
    assert_eq!(created.delegation.response_id, None);
    assert_eq!(
        created.delegation.extra.get("type"),
        Some(&json!("delegation"))
    );

    // Shaped by the reference schema: a Responses delegation names its response.
    let responses = parse(
        r#"{"type":"session.delegation.created","event_id":"evt_delegation_002","offset_ms":4100,
            "delegation":{"id":"del_resp1","type":"delegation","target":"responses","response_id":"resp_1"}}"#,
    );
    let ServerEvent::DelegationCreated(created) = responses else {
        panic!("expected session.delegation.created, got {responses:?}");
    };
    assert_eq!(created.delegation.target, DelegationTarget::Responses);
    assert_eq!(created.delegation.response_id.as_deref(), Some("resp_1"));
}

#[test]
fn a_nested_text_delta_is_kept_whole() {
    let payload = r#"{
  "type": "response.event",
  "event_id": "evt_response_002",
  "delegation_id": "del_responses123",
  "event": {
    "type": "response.output_text.delta",
    "item_id": "msg_abc123",
    "output_index": 0,
    "content_index": 0,
    "delta": "An outdoor table is available at 7 PM.",
    "sequence_number": 3,
    "logprobs": []
  }
}"#;
    let event = parse(payload);
    let ServerEvent::ResponseEvent(response) = event else {
        panic!("expected response.event, got {event:?}");
    };
    assert_eq!(response.delegation_id.as_deref(), Some("del_responses123"));
    assert_eq!(
        response.backend,
        BackendEvent::Other("response.output_text.delta".to_owned())
    );
    let whole: Value = serde_json::from_str(payload).expect("JSON");
    assert_eq!(response.event, whole["event"]);
}

// The nested Responses events below follow the Responses streaming event
// shapes the delegation guide names; the reference shows only a text delta.
#[test]
fn nested_lifecycle_and_function_call_events_are_typed() {
    let created = parse(
        r#"{"type":"response.event","event_id":"e1","delegation_id":"del_1",
            "event":{"type":"response.created","sequence_number":0,
                     "response":{"id":"resp_1","status":"in_progress","output":[],"tools":[],"instructions":null}}}"#,
    );
    let ServerEvent::ResponseEvent(created) = created else {
        panic!("expected response.event, got {created:?}");
    };
    assert_eq!(
        created.backend,
        BackendEvent::ResponseCreated {
            response_id: "resp_1".to_owned()
        }
    );

    let call = parse(
        r#"{"type":"response.event","event_id":"e2","delegation_id":"del_1",
            "event":{"type":"response.output_item.done","output_index":0,"sequence_number":4,
                     "item":{"type":"function_call","id":"fc_1","call_id":"call_123",
                             "name":"book_table","arguments":"{\"party\":2}","status":"completed"}}}"#,
    );
    let ServerEvent::ResponseEvent(call) = call else {
        panic!("expected response.event, got {call:?}");
    };
    assert_eq!(
        call.backend,
        BackendEvent::FunctionCallDone(FunctionCall {
            call_id: "call_123".to_owned(),
            name: "book_table".to_owned(),
            arguments: r#"{"party":2}"#.to_owned(),
            id: Some("fc_1".to_owned()),
        })
    );

    let message = parse(
        r#"{"type":"response.event","event_id":"e3","delegation_id":"del_1",
            "event":{"type":"response.output_item.done","output_index":1,"sequence_number":7,
                     "item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}}"#,
    );
    let ServerEvent::ResponseEvent(message) = message else {
        panic!("expected response.event, got {message:?}");
    };
    assert_eq!(
        message.backend,
        BackendEvent::Other("response.output_item.done".to_owned())
    );

    for (kind, outcome) in [
        ("response.completed", ResponseOutcome::Completed),
        ("response.failed", ResponseOutcome::Failed),
        ("response.incomplete", ResponseOutcome::Incomplete),
    ] {
        let ended = parse(&format!(
            r#"{{"type":"response.event","event_id":"e4","delegation_id":null,
                "event":{{"type":"{kind}","sequence_number":9,
                          "response":{{"id":"resp_1","output":[],"tools":[],"instructions":null}}}}}}"#
        ));
        let ServerEvent::ResponseEvent(ended) = ended else {
            panic!("expected response.event, got {ended:?}");
        };
        assert_eq!(ended.delegation_id, None);
        assert_eq!(
            ended.backend,
            BackendEvent::ResponseEnded {
                response_id: "resp_1".to_owned(),
                outcome,
            }
        );
    }
}

#[test]
fn a_malformed_nested_function_call_is_a_corrupt_frame() {
    let payload = r#"{"type":"response.event","event_id":"e","event":{"type":"response.output_item.done","item":{"type":"function_call","name":"book_table","arguments":"{}"}}}"#;
    let error = ServerEvent::parse(payload).expect_err("no call_id");
    let ProviderError::CorruptFrame(corrupt) = error else {
        panic!("expected a corrupt frame, got {error:?}");
    };
    assert_eq!(corrupt.event_type(), Some("response.event"));
    assert_eq!(corrupt.frame(), Some(payload));

    let untyped = r#"{"type":"response.event","event_id":"e","event":{"delta":"x"}}"#;
    assert!(matches!(
        ServerEvent::parse(untyped),
        Err(ProviderError::CorruptFrame(_))
    ));
}

#[test]
fn usage_updated_decodes_seconds_and_context_window() {
    let event = parse(
        r#"{
  "type": "session.usage.updated",
  "event_id": "evt_usage_001",
  "usage": {"seconds": 32.5},
  "context_window": {"usage_ratio": 0.12}
}"#,
    );
    let ServerEvent::UsageUpdated(usage) = event else {
        panic!("expected session.usage.updated, got {event:?}");
    };
    assert!((usage.usage.seconds - 32.5).abs() < f64::EPSILON);
    let ratio = usage.context_window.map(|window| window.usage_ratio);
    assert!(ratio.is_some_and(|ratio| (ratio - 0.12).abs() < f64::EPSILON));

    let without_window =
        parse(r#"{"type":"session.usage.updated","event_id":"e","usage":{"seconds":3}}"#);
    let ServerEvent::UsageUpdated(usage) = without_window else {
        panic!("expected session.usage.updated, got {without_window:?}");
    };
    assert_eq!(usage.context_window, None);
}

#[test]
fn append_acknowledgements_decode() {
    let cases = [
        (
            r#"{"type":"session.instructions.appended","event_id":"evt_instructions_002","client_event_id":"evt_instructions_001","start_ms":1200,"end_ms":1400}"#,
            "evt_instructions_001",
            (1200, 1400),
        ),
        (
            r#"{"type":"session.thinking.appended","event_id":"evt_thinking_002","client_event_id":"evt_thinking_001","start_ms":4600,"end_ms":4800}"#,
            "evt_thinking_001",
            (4600, 4800),
        ),
        (
            r#"{"type":"session.commentary.appended","event_id":"evt_commentary_002","client_event_id":"evt_commentary_001","start_ms":5200,"end_ms":5400}"#,
            "evt_commentary_001",
            (5200, 5400),
        ),
    ];
    for (payload, client_event_id, span) in cases {
        let event = parse(payload);
        let appended = match &event {
            ServerEvent::InstructionsAppended(appended)
            | ServerEvent::ThinkingAppended(appended)
            | ServerEvent::CommentaryAppended(appended) => appended,
            _ => panic!("expected an append acknowledgement, got {event:?}"),
        };
        assert_eq!(appended.client_event_id.as_deref(), Some(client_event_id));
        assert_eq!((appended.start_ms, appended.end_ms), span);
    }
    assert!(matches!(
        parse(r#"{"type":"session.thinking.appended","event_id":"e","start_ms":1,"end_ms":1}"#),
        ServerEvent::ThinkingAppended(_)
    ));
}

#[test]
fn mute_acknowledgements_decode() {
    let muted = parse(
        r#"{"type":"session.input_audio.muted","event_id":"evt_muted_001","client_event_id":"evt_mute_001"}"#,
    );
    let ServerEvent::InputAudioMuted(muted) = muted else {
        panic!("expected session.input_audio.muted, got {muted:?}");
    };
    assert_eq!(muted.client_event_id.as_deref(), Some("evt_mute_001"));

    let unmuted = parse(
        r#"{"type":"session.input_audio.unmuted","event_id":"evt_unmuted_001","client_event_id":"evt_unmute_001"}"#,
    );
    let ServerEvent::InputAudioUnmuted(unmuted) = unmuted else {
        panic!("expected session.input_audio.unmuted, got {unmuted:?}");
    };
    assert_eq!(unmuted.event_id.as_deref(), Some("evt_unmuted_001"));
}

#[test]
fn error_decodes_the_reference_example() {
    let event = parse(
        r#"{
  "type": "error",
  "event_id": "evt_error_001",
  "error": {
    "type": "invalid_request_error",
    "code": "unknown_parameter",
    "message": "Unknown parameter: 'session.voice'.",
    "param": "session.voice",
    "client_event_id": "evt_invalid_001"
  }
}"#,
    );
    let ServerEvent::Error(error) = event else {
        panic!("expected error, got {event:?}");
    };
    assert_eq!(error.error.code.as_deref(), Some("unknown_parameter"));
    assert_eq!(error.error.kind.as_deref(), Some("invalid_request_error"));
    assert_eq!(error.error.param.as_deref(), Some("session.voice"));
    assert_eq!(error.client_event_id, None);
    assert_eq!(error.caused_by(), Some("evt_invalid_001"));

    let outer = parse(
        r#"{"type":"error","event_id":"e","client_event_id":"evt_outer","error":{"message":"bad"}}"#,
    );
    let ServerEvent::Error(error) = outer else {
        panic!("expected error, got {outer:?}");
    };
    assert_eq!(error.caused_by(), Some("evt_outer"));
}

#[test]
fn unknown_events_are_kept_whole() {
    let payload = r#"{
  "type": "info",
  "event_id": "evt_info_001",
  "code": "data_channel_permissions",
  "message": "The frontend data channel is configured with restricted event permissions."
}"#;
    let event = parse(payload);
    let ServerEvent::Unknown(unknown) = event else {
        panic!("expected an unknown event, got {event:?}");
    };
    assert_eq!(unknown.kind, "info");
    assert_eq!(
        unknown.raw,
        serde_json::from_str::<Value>(payload).expect("JSON")
    );
}

#[test]
fn unmodelled_fields_are_kept_in_extra() {
    let event = parse(
        r#"{"type":"session.output_transcript.delta","event_id":"e","delta":"Hi","start_ms":1,"end_ms":2,"item_id":"it_9"}"#,
    );
    let ServerEvent::OutputTranscriptDelta(delta) = event else {
        panic!("expected an output transcript delta, got {event:?}");
    };
    assert_eq!(delta.extra.get("item_id"), Some(&json!("it_9")));
}

#[test]
fn malformed_payloads_are_corrupt_frames() {
    for payload in [
        "not json",
        r#"["session.started"]"#,
        r#"{"event_id":"e"}"#,
        r#"{"type":7}"#,
    ] {
        let error = ServerEvent::parse(payload).expect_err(payload);
        let ProviderError::CorruptFrame(corrupt) = error else {
            panic!("expected a corrupt frame for {payload}, got {error:?}");
        };
        assert_eq!(corrupt.event_type(), None);
        assert_eq!(corrupt.frame(), Some(payload));
    }
    let missing_delta =
        r#"{"type":"session.input_transcript.delta","event_id":"e","start_ms":1,"end_ms":2}"#;
    let error = ServerEvent::parse(missing_delta).expect_err("no delta");
    let ProviderError::CorruptFrame(corrupt) = error else {
        panic!("expected a corrupt frame, got {error:?}");
    };
    assert_eq!(corrupt.event_type(), Some("session.input_transcript.delta"));
}

#[test]
fn session_update_serializes_the_reference_example() {
    let event = ClientEvent::SessionUpdate {
        responses: ResponsesDelegationUpdate {
            model: None,
            settings: ResponsesSettings {
                instructions: Some(
                    "Check restaurant availability. Ask before confirming a booking.".to_owned(),
                ),
                max_output_tokens: Some(MaxOutputTokens::new(1024).expect("at least 16")),
                ..ResponsesSettings::default()
            },
        },
        event_id: Some("evt_update_001".to_owned()),
    };
    assert_eq!(
        to_json(&event),
        json!({
            "type": "session.update",
            "event_id": "evt_update_001",
            "session": {
                "delegation": {
                    "type": "responses",
                    "responses": {
                        "instructions": "Check restaurant availability. Ask before confirming a booking.",
                        "max_output_tokens": 1024
                    }
                }
            }
        })
    );
}

#[test]
fn context_appends_serialize_the_reference_examples() {
    let instructions = ClientEvent::InstructionsAppend(
        ContextAppend::session("The caller prefers outdoor seating.")
            .with_event_id("evt_instructions_001"),
    );
    assert_eq!(
        to_json(&instructions),
        json!({
            "type": "session.instructions.append",
            "event_id": "evt_instructions_001",
            "delegation_id": null,
            "content": "The caller prefers outdoor seating."
        })
    );
    let thinking = ClientEvent::ThinkingAppend(
        ContextAppend::for_delegation(
            "Checking availability for two guests at 7 PM.",
            "del_abc123",
        )
        .with_event_id("evt_thinking_001"),
    );
    assert_eq!(
        to_json(&thinking),
        json!({
            "type": "session.thinking.append",
            "event_id": "evt_thinking_001",
            "delegation_id": "del_abc123",
            "content": "Checking availability for two guests at 7 PM."
        })
    );
    let commentary = ClientEvent::CommentaryAppend(
        ContextAppend::for_delegation(
            "There is an outdoor table for two at 7 PM. Ask whether to reserve it.",
            "del_abc123",
        )
        .with_event_id("evt_commentary_001"),
    );
    assert_eq!(
        to_json(&commentary),
        json!({
            "type": "session.commentary.append",
            "event_id": "evt_commentary_001",
            "delegation_id": "del_abc123",
            "content": "There is an outdoor table for two at 7 PM. Ask whether to reserve it."
        })
    );
    assert_eq!(
        to_json(&ClientEvent::ThinkingAppend(ContextAppend::session(
            "quiet"
        ))),
        json!({"type": "session.thinking.append", "content": "quiet", "delegation_id": null})
    );
}

#[test]
fn function_output_and_continue_serialize_the_delegation_guide_examples() {
    let output = ClientEvent::FunctionCallOutput {
        call_id: "call_123".to_owned(),
        output: r#"{"status":"confirmed","order_id":"order_123"}"#.to_owned(),
        event_id: Some("tool_result_1".to_owned()),
    };
    assert_eq!(
        to_json(&output),
        json!({
            "type": "response.item.create",
            "event_id": "tool_result_1",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "{\"status\":\"confirmed\",\"order_id\":\"order_123\"}"
            }
        })
    );
    let create = ClientEvent::ResponseCreate {
        event_id: Some("continue_1".to_owned()),
    };
    assert_eq!(
        to_json(&create),
        json!({"type": "response.create", "event_id": "continue_1"})
    );
    assert_eq!(
        to_json(&ClientEvent::ResponseCreate { event_id: None }),
        json!({"type": "response.create"})
    );
}

#[test]
fn mute_unmute_and_close_serialize_the_reference_examples() {
    assert_eq!(
        to_json(&ClientEvent::MuteInputAudio {
            event_id: Some("evt_mute_001".to_owned())
        }),
        json!({"type": "session.input_audio.mute", "event_id": "evt_mute_001"})
    );
    assert_eq!(
        to_json(&ClientEvent::UnmuteInputAudio {
            event_id: Some("evt_unmute_001".to_owned())
        }),
        json!({"type": "session.input_audio.unmute", "event_id": "evt_unmute_001"})
    );
    assert_eq!(
        to_json(&ClientEvent::SessionClose {
            event_id: Some("evt_close_001".to_owned())
        }),
        json!({"type": "session.close", "event_id": "evt_close_001"})
    );
    assert_eq!(
        to_json(&ClientEvent::SessionClose { event_id: None }),
        json!({"type": "session.close"})
    );
}
