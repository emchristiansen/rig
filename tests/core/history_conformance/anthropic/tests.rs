//! The unchanged Anthropic rich fixture under its typed Messages boundary.

use super::*;
use rig_core::error::ErrorDetail;

#[derive(Clone, Copy, Debug)]
enum Outcome {
    Accepted,
    AcceptedMissingFinish,
    CorruptFrame,
}

pub(super) fn anthropic_h05_preserves_the_typed_boundary(fixture: &MessagesHistory) {
    let ablation = fixture
        .ablation()
        .expect("the unchanged original ablation fixture");
    let wire = fixture.wire(fixture.model);
    let request = CompletionRequest::new("restate");
    let original = decode(
        &wire,
        &request,
        Mode::Unary,
        (ablation.frames)(ablation.document.clone()),
    )
    .expect("the original whole reply decodes");
    assert!(!original.stop().is_failure());
    let cases = [
        (
            "/id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/role",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/model",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/0/thinking",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/0/signature",
            [Outcome::Accepted, Outcome::Accepted, Outcome::CorruptFrame],
        ),
        (
            "/content/1/data",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/name",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/input",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/2/input/query",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/3/tool_use_id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/3/content",
            [Outcome::CorruptFrame, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/3/content/0/type",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/3/content/0/url",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/3/content/0/title",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/3/content/0/encrypted_content",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/4/text",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/5/id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/5/name",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/5/input",
            [Outcome::CorruptFrame, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/5/input/q",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/stop_reason",
            [
                Outcome::AcceptedMissingFinish,
                Outcome::AcceptedMissingFinish,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/stop_sequence",
            [Outcome::Accepted, Outcome::Accepted, Outcome::CorruptFrame],
        ),
        (
            "/usage",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/usage/input_tokens",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/usage/output_tokens",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
    ];
    for (pointer, expected) in cases {
        let original = ablation
            .document
            .pointer(pointer)
            .expect("the original field");
        let retyped = match original {
            Value::String(_) => Value::from(7),
            Value::Number(_) => Value::from("7"),
            Value::Bool(_) => Value::from("true"),
            Value::Array(_) => json!({}),
            Value::Object(_) => json!([]),
            Value::Null => Value::from(false),
        };
        for (change, replacement, expected) in [
            ("removed", None, expected[0]),
            ("null", Some(Value::Null), expected[1]),
            ("retyped", Some(retyped), expected[2]),
        ] {
            let mut document = ablation.document.clone();
            if let Some(value) = replacement {
                *document.pointer_mut(pointer).expect("the original field") = value;
            } else {
                let (parent, key) = pointer.rsplit_once('/').expect("a field pointer");
                let key = key.replace("~1", "/").replace("~0", "~");
                document
                    .pointer_mut(parent)
                    .and_then(Value::as_object_mut)
                    .expect("the original parent")
                    .shift_remove(&key);
            }
            let result = decode(&wire, &request, Mode::Unary, (ablation.frames)(document));
            match expected {
                Outcome::CorruptFrame => {
                    let error = result.expect_err("the invalid typed field must refuse");
                    assert!(
                        matches!(error.report().detail, Some(ErrorDetail::CorruptFrame(_))),
                        "{pointer} {change}: {error}"
                    );
                }
                Outcome::Accepted => {
                    let response =
                        result.unwrap_or_else(|error| panic!("{pointer} {change}: {error}"));
                    assert!(!response.stop().is_failure(), "{pointer} {change}");
                }
                Outcome::AcceptedMissingFinish => {
                    let response =
                        result.unwrap_or_else(|error| panic!("{pointer} {change}: {error}"));
                    assert!(response.stop().is_failure(), "{pointer} {change}");
                    assert_eq!(
                        response.error.as_deref(),
                        Some("the provider ended the reply without a finish reason")
                    );
                }
            }
        }
    }
    let mut without_finish = ablation.document;
    without_finish
        .as_object_mut()
        .expect("the original object")
        .shift_remove("stop_reason");
    assert!(
        decode(
            &wire,
            &request,
            Mode::Unary,
            (ablation.frames)(without_finish)
        )
        .map_or(true, |response| response.stop().is_failure())
    );
}

/// Temporary evidence using the unchanged rich fixture and existing cut/encode boundaries.
#[test]
fn r002_cut_hosted_request_evidence() {
    let fixture = &ANTHROPIC_HISTORY;
    let wire = fixture.wire(fixture.model);
    let frames = fixture
        .reply(Shape::Rich, Mode::Streaming)
        .expect("the original rich stream");
    let expected = json!({"type":"server_tool_use","id":"srvtoolu_1",
        "name":"web_search","input":{"query":"rig"}});
    for cut in 1..frames.len() {
        let (response, ended) = rig_history_conformance::partial(
            &wire,
            Mode::Streaming,
            frames.iter().take(cut).cloned(),
        );
        let Some(carried) = response.choice.iter().find_map(|block| match block {
            AssistantContent::CarriedText(part)
                if part.payload.item_type() == Some("server_tool_use") =>
            {
                Some(part)
            }
            _ => None,
        }) else {
            continue;
        };
        assert!(!ended);
        assert_eq!(carried.payload.item, expected);
        assert_eq!(carried.payload.provider.as_deref(), Some("anthropic"));
        assert_eq!(
            carried.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 2
            }
        );
        assert_eq!(
            carried.content,
            rig_core::message::CarriedTextContent::Opaque(String::new())
        );
        assert_eq!(response.tool_calls().count(), 0);
        let turn = response.message().expect("the received partial turn");
        assert!(matches!(&turn, Message::Assistant(turn)
            if matches!(turn.stop, Some(rig_core::message::StopReason::Aborted(_)))));
        let history = vec![Message::user("q"), turn, Message::user("next")];
        let describe = wire.describe();
        let adapted = rig_core::completion::history::adapt(
            &history,
            describe.replay.expect("the same real Messages target"),
        );
        let mut request = CompletionRequest::new("next");
        request.chat_history = adapted;
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the exact next history encodes");
        let rig_core::wire::Body::Bytes(bytes) = encoded.request.body() else {
            panic!("the native Messages request is JSON bytes");
        };
        eprintln!("R002_CUT_COUNT: {cut}");
        eprintln!(
            "R002_ACTUAL_ENCODED_BODY: {}",
            std::str::from_utf8(bytes).expect("the encoder's JSON is UTF-8")
        );
        let body: Value = serde_json::from_slice(bytes).expect("the real encoded request body");
        assert_eq!(
            body["messages"],
            json!([
                {"role":"user","content":[{"type":"text","text":"q"}]},
                {"role":"assistant","content":[expected]},
                {"role":"user","content":[{"type":"text","text":"next"}]},
            ])
        );
        return;
    }
    panic!("the original rich prefix must expose the real carried hosted use");
}
/// Permanent policy control at the exact original rich-fixture cut 10.
#[test]
fn hc001_original_cut10_refuses_native_and_prepared_model_before_transport() {
    let fixture = &ANTHROPIC_HISTORY;
    let wire = fixture.wire(fixture.model);
    let frames = fixture
        .reply(Shape::Rich, Mode::Streaming)
        .expect("unchanged original rich stream");
    let frame_bytes: Vec<_> = frames
        .iter()
        .map(|frame| match frame {
            WireFrame::Text(text) => text.clone(),
            _ => panic!("the original rich frames are text"),
        })
        .collect();
    let (response, ended) =
        rig_history_conformance::partial(&wire, Mode::Streaming, frames.iter().take(10).cloned());
    assert!(!ended);
    let expected = json!({"type":"server_tool_use", "id":"srvtoolu_1",
        "name":"web_search", "input":{"query":"rig"}});
    let part = response
        .choice
        .iter()
        .find_map(|block| match block {
            AssistantContent::CarriedText(part)
                if part.payload.item_type() == Some("server_tool_use") =>
            {
                Some(part)
            }
            _ => None,
        })
        .expect("exact cut 10 closes the genuine hosted use");
    assert_eq!(part.payload.item, expected);
    assert_eq!(part.payload.provider.as_deref(), Some("anthropic"));
    assert_eq!(
        part.payload.position,
        rig_core::message::CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 2
        }
    );
    assert_eq!(
        part.content,
        rig_core::message::CarriedTextContent::Opaque(String::new())
    );
    assert_eq!(response.tool_calls().count(), 0);
    let received_before = response.clone();
    let received_serde = serde_json::to_vec(&response).expect("whole received response serde");
    let history = vec![
        Message::user("q"),
        response.message().expect("genuine failed turn"),
        Message::user("next"),
    ];
    let history_before = history.clone();
    let history_serde = serde_json::to_vec(&history).expect("whole original history serde");
    let describe = wire.describe();
    let adapted =
        rig_core::completion::history::adapt(&history, describe.replay.expect("real target"));
    assert!(matches!(&adapted[1], Message::Assistant(turn)
        if matches!(turn.stop, Some(rig_core::message::StopReason::Aborted(_)))
            && turn.content.len() == 1
            && matches!(&turn.content[0], AssistantContent::CarriedText(part) if part.payload.item == expected)));
    let request = CompletionRequest::from(adapted.clone());
    let direct = wire
        .encode(request, Mode::Unary)
        .expect_err("the exact original carrier now refuses at native encode");
    let expected_error = "Cannot continue Anthropic Messages: an aborted turn's unresolved server call `srvtoolu_1` must be resolved or removed from history before continuing";
    let rig_core::error::ProviderError::Request(cause) =
        rig_core::error::ProviderError::from(direct)
    else {
        panic!("native Request refusal");
    };
    assert_eq!(cause.to_string(), expected_error);
    let client = rig_core::test_utils::RecordingHttpClient::new("no transport reply may be read");
    let model = rig_core::driver::Model::new(wire, client.clone());
    let failure = match model.stream(CompletionRequest::from(history.clone())) {
        Err(error) => error,
        Ok(_) => panic!("Model must refuse during preparation before opening transport"),
    };
    let rig_core::error::ProviderError::Request(cause) = failure else {
        panic!("prepared Model Request refusal");
    };
    assert_eq!(cause.to_string(), expected_error);
    assert!(client.requests().is_empty());
    assert_eq!(response, received_before);
    assert_eq!(
        serde_json::to_vec(&response).expect("source after"),
        received_serde
    );
    assert_eq!(history, history_before);
    assert_eq!(
        serde_json::to_vec(&history).expect("history after"),
        history_serde
    );
    assert_eq!(
        frames
            .iter()
            .map(|frame| match frame {
                WireFrame::Text(text) => text.clone(),
                _ => panic!("original text frame"),
            })
            .collect::<Vec<_>>(),
        frame_bytes
    );
}
