//! Server events replay the control-socket frames a live probe recorded
//! (`tests/fixtures/chatgpt-realtime`, audio truncated to 64 base64
//! characters). The control socket is not HTTP, so the cassette engine does
//! not record it; these fixtures are its recorded evidence.

use super::*;
use serde_json::json;

const FIXTURES: [&str; 4] = [
    "s1-hello",
    "s2-codeword",
    "s3-delegation",
    "s4-large-context",
];

/// One recorded frame: seconds since the probe started, who sent it, and
/// the event object.
#[derive(serde::Deserialize)]
struct Recorded {
    #[allow(dead_code)]
    t: f64,
    direction: String,
    event: Value,
}

fn recorded(name: &str) -> Vec<Recorded> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chatgpt-realtime")
        .join(format!("{name}.control-socket.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()));
    serde_json::from_str(&text).expect("a fixture is a JSON array of recorded frames")
}

fn server_events(name: &str) -> Vec<ServerEvent> {
    recorded(name)
        .into_iter()
        .filter(|frame| frame.direction == "server")
        .map(|frame| {
            let payload = frame.event.to_string();
            ServerEvent::parse(&payload)
                .unwrap_or_else(|error| panic!("{name}: {payload} should decode: {error}"))
        })
        .collect()
}

/// Every recorded server event decodes to a modelled variant.
#[test]
fn every_recorded_server_event_decodes_to_a_modelled_variant() {
    let mut total = 0;
    for name in FIXTURES {
        for event in server_events(name) {
            total += 1;
            if let ServerEvent::Unknown(unknown) = &event {
                panic!("{name}: `{}` decoded as unknown", unknown.kind);
            }
        }
    }
    assert_eq!(total, 1294, "all four recordings replayed");
}

/// The first recording's facts: the session id is the call id, both
/// transcripts arrive, turns end with their final text, audio keeps its
/// timing, and usage reports the billed audio.
#[test]
fn the_hello_recording_carries_its_transcripts_turns_and_usage() {
    let events = server_events("s1-hello");

    let ServerEvent::SessionStarted(started) = &events[0] else {
        panic!("the first event starts the session, got {:?}", events[0]);
    };
    assert_eq!(
        started.session.id,
        "rtc_u2_ETA6qp3oN2ZjwM8wdXUrx6g4jZ3pN2Zx"
    );
    assert_eq!(started.session.status.as_deref(), Some("active"));
    assert_eq!(started.session.expires_at, Some(1_790_626_280));

    let heard: String = events
        .iter()
        .filter_map(|event| match event {
            ServerEvent::InputTranscriptAdded(added) => Some(added.item.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(heard, " Hello, can you hear me");
    let said: String = events
        .iter()
        .filter_map(|event| match event {
            ServerEvent::OutputTranscriptAdded(added) => Some(added.item.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(said, " Yep, loud and clear.");

    let done: Vec<(Role, &str)> = events
        .iter()
        .filter_map(|event| match event {
            ServerEvent::TurnDone(done) => {
                Some((done.turn.role.clone(), done.turn.transcript.as_str()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        done,
        [
            (Role::User, " Hello, can you hear me"),
            (Role::Assistant, " Yep, loud and clear."),
        ]
    );

    let first_audio = events.iter().find_map(|event| match event {
        ServerEvent::OutputAudioDelta(delta) => Some(delta),
        _ => None,
    });
    let first_audio = first_audio.expect("the recording has output audio");
    assert_eq!(
        (first_audio.start_ms, first_audio.end_ms),
        (Some(200), Some(400))
    );

    let usage: Vec<&UsageUpdated> = events
        .iter()
        .filter_map(|event| match event {
            ServerEvent::UsageUpdated(usage) => Some(usage),
            _ => None,
        })
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].usage.audio_duration_ms, 14_400);
    assert!(usage[0].usage.backend_model_usage.is_empty());
    let limit = usage[0].usage_limit.as_ref().expect("a usage limit object");
    assert_eq!((&limit.status, &limit.reset_seconds), (&None, &None));
}

/// The delegation recording: GPT-Live hands the question to the client, the
/// probe's answer is exactly what [`ClientEvent`] builds, and the server
/// acknowledges it against the same item.
#[test]
fn the_delegation_recording_round_trips_the_client_answer() {
    let frames = recorded("s3-delegation");
    let events = server_events("s3-delegation");
    let created = events
        .iter()
        .find_map(|event| match event {
            ServerEvent::DelegationCreated(created) => Some(created),
            _ => None,
        })
        .expect("the recording delegates once");
    assert_eq!(created.item.id, "item_ETA8G52EuzeNlxyQbOYZn");
    assert_eq!(created.item.kind, "delegation");
    assert_eq!(created.item.target, "client");
    assert_eq!(
        created.item.text(),
        "Please check the latest status of the operator line for me"
    );
    assert_eq!(created.item.handoff_id.as_deref(), Some("handoff_1"));
    assert_eq!(
        created.item.user_bidi_turn_id.as_deref(),
        Some("turn_ETA8EysfTCa00LPsqAFaK")
    );
    assert_eq!(created.offset_ms, Some(5_000));

    let sent: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame.direction == "client")
        .map(|frame| &frame.event)
        .collect();
    let built = ClientEvent::delegation_context_append(
        &created.item.id,
        Some(ContextChannel::Speakable),
        "<backend answer>",
    );
    assert_eq!(built.len(), 1);
    assert_eq!(
        sent,
        [&serde_json::to_value(&built[0]).expect("serializes")]
    );

    let appended = events
        .iter()
        .find_map(|event| match event {
            ServerEvent::DelegationContextAppended(appended) => Some(appended),
            _ => None,
        })
        .expect("the server acknowledges the answer");
    assert_eq!(appended.delegation_item_id, created.item.id);

    let usage: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            ServerEvent::UsageUpdated(usage) => Some(usage.usage.audio_duration_ms),
            _ => None,
        })
        .collect();
    assert_eq!(usage, [14_200, 29_400]);
}

/// An unmodelled type is kept whole; a modelled type with a malformed field,
/// or a payload with no string `type`, is a corrupt frame.
#[test]
fn unknown_types_are_kept_and_malformed_known_types_are_refused() {
    let event =
        ServerEvent::parse(r#"{"type":"session.closed","reason":"requested"}"#).expect("decodes");
    assert_eq!(
        event,
        ServerEvent::Unknown(UnknownEvent {
            kind: "session.closed".to_owned(),
            raw: json!({"type": "session.closed", "reason": "requested"}),
        })
    );

    for payload in [
        r#"{"type":"turn.done","turn":{"id":1}}"#,
        r#"{"audio":"AAAA"}"#,
        r#"{"type":7}"#,
        "not json",
    ] {
        let error = ServerEvent::parse(payload).expect_err("refused");
        assert!(
            error.corrupt_frame().is_some(),
            "{payload} should be a corrupt frame, got {error:?}"
        );
    }
}

/// Both error spellings yield their message, and extras survive.
#[test]
fn error_events_expose_their_message() {
    let ServerEvent::Error(nested) = ServerEvent::parse(
        r#"{"type":"error","event_id":"e1","error":{"code":"bad","message":"nope","type":"invalid_request_error"}}"#,
    )
    .expect("decodes") else {
        panic!("an error event");
    };
    assert_eq!(nested.message(), Some("nope"));
    assert_eq!(nested.extra["event_id"], "e1");
    let detail = nested.error.as_ref().expect("an error object");
    assert_eq!(detail.code.as_deref(), Some("bad"));
    assert_eq!(detail.kind.as_deref(), Some("invalid_request_error"));

    let ServerEvent::Error(flat) =
        ServerEvent::parse(r#"{"type":"error","message":"flat"}"#).expect("decodes")
    else {
        panic!("an error event");
    };
    assert_eq!(flat.message(), Some("flat"));
}

/// Chunks hold at most 500 bytes, never split a character, and join back to
/// the text; short and empty texts are one chunk, as in the Codex client.
#[test]
fn context_chunks_split_on_character_boundaries() {
    assert_eq!(
        context_chunks("")
            .iter()
            .map(ContextChunk::as_str)
            .collect::<Vec<_>>(),
        [""]
    );
    let exact = "a".repeat(CONTEXT_APPEND_MAX_BYTES);
    assert_eq!(context_chunks(&exact).len(), 1);

    let text = format!("{}é{}", "a".repeat(499), "ü".repeat(400));
    let chunks = context_chunks(&text);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.as_str().len() <= CONTEXT_APPEND_MAX_BYTES)
    );
    assert_eq!(
        chunks[0].as_str().len(),
        499,
        "the two-byte `é` moves whole"
    );
    assert_eq!(
        chunks.iter().map(ContextChunk::as_str).collect::<String>(),
        text
    );
    assert_eq!(chunks.len(), 3);
}

/// Client events serialize to the Codex client's wire shapes, one event per
/// chunk, with `channel` omitted when unset.
#[test]
fn client_events_serialize_like_the_codex_client() {
    let long = "x".repeat(1_200);
    let events = ClientEvent::session_context_append(None, &long);
    assert_eq!(events.len(), 3);
    assert_eq!(
        serde_json::to_value(&events[2]).expect("serializes"),
        json!({"type": "session.context.append", "content": [{"type": "input_text", "text": "x".repeat(200)}]})
    );
    let commentary =
        ClientEvent::delegation_context_append("item_1", Some(ContextChannel::Commentary), "hi");
    assert_eq!(
        serde_json::to_value(&commentary[0]).expect("serializes"),
        json!({
            "type": "delegation.context.append",
            "delegation_item_id": "item_1",
            "channel": "commentary",
            "content": [{"type": "input_text", "text": "hi"}]
        })
    );
    assert_eq!(
        serde_json::to_value(ClientEvent::SessionClose).expect("serializes"),
        json!({"type": "session.close"})
    );
}
