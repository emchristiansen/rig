//! The frames a session returns for a turn: each frame's text beside its
//! event, and what a turn's texts fold to through Rig's reply decoder.

use super::*;
use crate::ws_client::Frame;
use rig_core::wire::{Mode, WireFrame, decode_reply};

/// The provider's message item of the turn below.
const MESSAGE_ITEM: &str = r#"{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"héllo","annotations":[]}]}"#;

/// A turn's frames, spelled as no serializer here would spell them: spaces,
/// a `type` that is not the first key, and an escape where a character
/// would do. A frame returned with any other spelling was rewritten.
fn turn_texts() -> Vec<String> {
    let response = |status: &str, output: &str| {
        format!(
            r#"{{"id":"resp_1", "object":"response", "created_at":0, "status":"{status}", "model":"gpt-5.4", "output":[{output}]}}"#
        )
    };
    vec![
        format!(
            r#"{{ "response": {}, "sequence_number": 0, "type": "response.created" }}"#,
            response("in_progress", "")
        ),
        r#"{"headers":{"X-Codex-Turn-State":"ts_1"},"type":"response.metadata"}"#.to_owned(),
        r#"{"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#.to_owned(),
        r#"{"type":"response.output_text.delta","content_index":0,"delta":"héllo","item_id":"msg_1","logprobs":[],"output_index":0,"sequence_number":2}"#.to_owned(),
        format!(
            r#"{{"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{MESSAGE_ITEM}}}"#
        ),
        format!(
            r#"{{"type":"response.completed","sequence_number":4,"response":{}}}"#,
            response("completed", MESSAGE_ITEM)
        ),
    ]
}

/// Read the turn in flight to its end, keeping every frame.
async fn frames_of_turn(session: &mut CodexWebSocketSession) -> Vec<TurnFrame> {
    let mut frames = Vec::new();
    loop {
        let frame = session.next_frame().await.expect("the turn's frames");
        let terminal = frame.event.is_terminal();
        frames.push(frame);
        if terminal {
            return frames;
        }
    }
}

fn texts(frames: &[TurnFrame]) -> Vec<&str> {
    frames.iter().map(|frame| frame.text.as_str()).collect()
}

/// A turn's frames are returned with the text the server sent, in order,
/// each beside the event it decodes to. A ping, and the `response.done`
/// trailing the turn before, are not frames of the turn. A binary frame
/// whose bytes are UTF-8 is returned as that text.
#[tokio::test]
async fn a_turns_frames_are_returned_as_the_server_sent_them() {
    let sent = turn_texts();
    let script = Script::new()
        .turn([completed("resp_0")])
        .turn(Vec::<String>::new());
    let mut session = session_over(&script);
    session
        .send(user_request("first"))
        .await
        .expect("the first turn sends");
    finish_turn(&mut session).await;

    session
        .send(user_request("second"))
        .await
        .expect("the second turn sends");
    let mut arriving = vec![
        Frame::Text(
            json!({"type": "response.done", "response": {"id": "resp_0", "status": "completed"}})
                .to_string(),
        ),
        Frame::Ping(bytes::Bytes::from_static(b"p")),
    ];
    for (index, text) in sent.iter().enumerate() {
        arriving.push(if index == 3 {
            Frame::Binary(bytes::Bytes::from(text.clone().into_bytes()))
        } else {
            Frame::Text(text.clone())
        });
        arriving.push(Frame::Pong(bytes::Bytes::new()));
    }
    script.arrive(arriving);

    let frames = frames_of_turn(&mut session).await;
    assert_eq!(
        texts(&frames),
        sent.iter().map(String::as_str).collect::<Vec<_>>(),
        "the turn's own frames, as sent, and no other frame"
    );
    let kinds: Vec<&str> = frames
        .iter()
        .map(|frame| match &frame.event {
            ResponsesWebSocketEvent::Response(_) => "response",
            ResponsesWebSocketEvent::Item(_) => "item",
            ResponsesWebSocketEvent::Error(_) => "error",
            ResponsesWebSocketEvent::Done(_) => "done",
            ResponsesWebSocketEvent::Unknown(_) => "unknown",
        })
        .collect();
    assert_eq!(
        kinds,
        ["response", "unknown", "item", "item", "item", "response"]
    );
    assert_eq!(session.received_turn_state(), Some("ts_1"));
    assert_eq!(session.previous_response_id(), Some("resp_1"));
}

/// The texts of a turn fold, through Rig's reply decoder on the session's
/// own wire, to the turn: its response id and its message, whose block
/// keeps the provider's item. A frame of a type Rig does not model, here
/// the Codex `response.metadata`, changes nothing in the fold.
#[tokio::test]
async fn a_turns_texts_fold_to_the_turn() {
    let sent = turn_texts();
    let script = Script::new().turn(sent.clone());
    let mut session = session_over(&script);
    let request = user_request("say hi");
    session.send(request.clone()).await.expect("the turn sends");
    let frames = frames_of_turn(&mut session).await;

    let fold = |texts: Vec<String>| {
        decode_reply(
            &codex_wire(),
            &request,
            Mode::Streaming,
            texts.into_iter().map(WireFrame::Text),
        )
    };
    let turn = fold(frames.iter().map(|frame| frame.text.clone()).collect())
        .expect("the turn's texts fold");
    assert_eq!(turn.response_id(), Some("resp_1"));
    let message = turn.message().expect("the turn's message");
    let written = serde_json::to_value(&message).expect("serializes");
    assert_eq!(written["content"].as_array().map(Vec::len), Some(1));
    assert_eq!(written["content"][0]["text"], "héllo");
    assert_eq!(
        written["content"][0]["native"]["item"],
        serde_json::from_str::<Value>(MESSAGE_ITEM).expect("the item is JSON"),
        "the block keeps the provider's item"
    );
    assert_eq!(written["origin"]["response_id"], "resp_1");

    let mut without_metadata = sent;
    without_metadata.remove(1);
    let plain = fold(without_metadata).expect("the same turn without the metadata frame");
    assert_eq!(plain.message(), Some(message));
    assert_eq!(plain.raw, turn.raw);
}

/// A turn the server ends with its own `error` event: the event's frame is
/// the last text of the turn, and the fold of the turn's texts fails with
/// that frame's text rather than yielding a reply. The fold does not read
/// the event's status or error type; the event the session returned beside
/// the frame does.
#[tokio::test]
async fn a_turn_ended_by_an_error_event_keeps_that_frame_and_does_not_fold() {
    let error = r#"{"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached"}}"#;
    let mut sent = turn_texts();
    sent.truncate(4);
    sent.push(error.to_owned());
    let script = Script::new().turn(sent.clone());
    let mut session = session_over(&script);
    let request = user_request("say hi");
    session.send(request.clone()).await.expect("the turn sends");
    let frames = frames_of_turn(&mut session).await;
    assert_eq!(
        texts(&frames),
        sent.iter().map(String::as_str).collect::<Vec<_>>()
    );

    let folded = decode_reply(
        &codex_wire(),
        &request,
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.text.clone())),
    );
    let ProviderError::ProviderResponse(refused) =
        folded.expect_err("an error event is not a reply")
    else {
        panic!("the fold fails with the provider's error event");
    };
    assert_eq!(refused.body, error, "the event as sent");
    assert_eq!(
        (refused.status, refused.code.as_deref(), refused.transient),
        (None, None, None),
        "the fold reads neither the event's status nor its error type"
    );
    let ResponsesWebSocketEvent::Error(event) = &frames[frames.len() - 1].event else {
        panic!("the turn's last event is the error");
    };
    assert_eq!(event.status, Some(429), "the session's event states them");
}

/// A turn the server ends with `response.done` and no `response.completed`:
/// the session ends the turn there and the frame is the turn's last text,
/// but Rig's reply decoder does not take `response.done` as the end of a
/// reply, so the turn's texts do not fold: the fold reports a reply cut
/// short. A caller that needs the folded turn has none for such a turn.
#[tokio::test]
async fn a_turn_ended_by_done_alone_is_returned_whole_and_does_not_fold() {
    let mut sent = turn_texts();
    let completed = sent.pop().expect("the terminal");
    sent.push(completed.replace("response.completed", "response.done"));
    let script = Script::new().turn(sent.clone());
    let mut session = session_over(&script);
    let request = user_request("say hi");
    session.send(request.clone()).await.expect("the turn sends");
    let frames = frames_of_turn(&mut session).await;
    assert_eq!(
        texts(&frames),
        sent.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert!(matches!(
        frames[frames.len() - 1].event,
        ResponsesWebSocketEvent::Done(_)
    ));
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    let folded = decode_reply(
        &codex_wire(),
        &request,
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.text.clone())),
    );
    assert!(
        matches!(folded, Err(ProviderError::Truncated)),
        "the fold does not end at `response.done`: {:?}",
        folded.map(|turn| turn.raw)
    );
}
