//! Steps the API restates or extends out of the ordinary order: a call
//! started twice, JSON argument fragments, thoughts a signature already
//! closed, unknown thought deltas, place citations, starts that state no
//! step, hand-built events without their object, and EOF in each mode.

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::AssistantContent;
use crate::wire::Wire as _;

fn interactions() -> super::super::Interactions {
    crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro")
}

/// Decode `frames` as one reply folded as `mode` folds it.
fn decode(
    mode: Mode,
    frames: impl IntoIterator<Item = Value>,
) -> Result<CompletionResponse, ProviderError> {
    crate::test_utils::decode_reply(
        &interactions(),
        &CompletionRequest::new("restate"),
        mode,
        frames
            .into_iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
}

fn whole(steps: Vec<Value>) -> Value {
    json!({"id": "int_1", "status": "completed", "steps": steps})
}

fn start(index: usize, step: Value) -> Value {
    json!({"event_type": "step.start", "index": index, "step": step})
}

fn delta(index: usize, delta: Value) -> Value {
    json!({"event_type": "step.delta", "index": index, "delta": delta})
}

fn stop(index: usize) -> Value {
    json!({"event_type": "step.stop", "index": index})
}

fn completed() -> Value {
    json!({
        "event_type": "interaction.completed",
        "interaction": {"id": "int_1", "status": "completed"},
    })
}

/// The one tool call `response` holds, as `(provider id, name, arguments)`.
fn only_call(response: &CompletionResponse) -> (Option<&str>, &str, Value) {
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    (
        call.id
            .provider()
            .map(crate::message::ProviderCallId::as_str),
        call.function.name.as_str(),
        call.function.arguments_value(),
    )
}

/// A second `step.start` for a call step still open restates that call:
/// it keeps its block and takes the restated arguments and item.
#[test]
fn a_restarted_call_step_restates_the_open_call() {
    let call = json!({"type": "function_call", "id": "fc_1", "name": "add", "arguments": {"x": 2}});
    let streamed = decode(
        Mode::Streaming,
        [
            start(
                0,
                json!({"type": "function_call", "id": "fc_1", "name": "add"}),
            ),
            start(0, call.clone()),
            stop(0),
            completed(),
        ],
    )
    .expect("the restarted call decodes");
    assert_eq!(only_call(&streamed), (Some("fc_1"), "add", json!({"x": 2})));
    let unary = decode(Mode::Unary, [whole(vec![call])]).expect("the whole call decodes");
    assert_eq!(
        unary.message(),
        streamed.message(),
        "a whole reply and its restatement as a stream fold into the same turn"
    );
}

/// An `arguments_delta` whose arguments are JSON rather than text streams
/// that JSON's text, which the stop then parses as the arguments.
#[test]
fn a_json_arguments_fragment_streams_its_text() {
    let streamed = decode(
        Mode::Streaming,
        [
            start(
                0,
                json!({"type": "function_call", "id": "fc_1", "name": "add"}),
            ),
            delta(0, json!({"type": "arguments_delta", "arguments": {"x": 3}})),
            stop(0),
            completed(),
        ],
    )
    .expect("the fragmented call decodes");
    assert_eq!(only_call(&streamed), (Some("fc_1"), "add", json!({"x": 3})));
    let unary = decode(
        Mode::Unary,
        [whole(vec![
            json!({"type": "function_call", "id": "fc_1", "name": "add", "arguments": {"x": 3}}),
        ])],
    )
    .expect("the whole call decodes");
    assert_eq!(
        unary.message(),
        streamed.message(),
        "a whole reply and its restatement as a stream fold into the same turn"
    );
}

/// A summary arriving for a thought its signature already closed opens a
/// new thought of its own rather than reopening the closed one.
#[test]
fn a_summary_after_a_signed_thought_opens_another_thought() {
    let signed = json!({"type": "thought", "signature": "sig", "summary": [{"type": "text", "text": "plan"}]});
    let response = decode(
        Mode::Streaming,
        [
            start(0, signed.clone()),
            delta(
                0,
                json!({"type": "thought_summary", "content": {"type": "text", "text": "more"}}),
            ),
            stop(0),
            completed(),
        ],
    )
    .expect("both thoughts decode");
    let [
        AssistantContent::CarriedReasoning(first),
        AssistantContent::CarriedReasoning(second),
    ] = response.choice.as_slice()
    else {
        panic!("two carried thoughts: {:?}", response.choice);
    };
    assert_eq!(
        first.content,
        vec![CarriedReasoningPart::Text {
            text: "plan".to_owned(),
            signature: Some("sig".to_owned()),
        }]
    );
    assert_eq!(first.payload.item, signed);
    assert_eq!(
        second.content,
        vec![CarriedReasoningPart::Text {
            text: "more".to_owned(),
            signature: None,
        }]
    );
    assert_eq!(
        second.payload.item,
        json!({"type": "thought", "summary": [{"type": "text", "text": "more"}]})
    );
}

/// A delta of another kind merges into a thought still open; once a
/// signature closed the thought, it changes nothing.
#[test]
fn another_thought_delta_merges_only_into_an_open_thought() {
    let response = decode(
        Mode::Streaming,
        [
            start(
                0,
                json!({"type": "thought", "summary": [{"type": "text", "text": "plan"}]}),
            ),
            delta(0, json!({"type": "thought_note", "note": "kept"})),
            stop(0),
            start(1, json!({"type": "thought", "signature": "sig"})),
            delta(1, json!({"type": "thought_note", "note": "dropped"})),
            stop(1),
            completed(),
        ],
    )
    .expect("both thoughts decode");
    let [
        AssistantContent::CarriedReasoning(open),
        AssistantContent::CarriedReasoning(signed),
    ] = response.choice.as_slice()
    else {
        panic!("two carried thoughts: {:?}", response.choice);
    };
    assert_eq!(
        open.payload.item,
        json!({"type": "thought", "summary": [{"type": "text", "text": "plan"}], "note": "kept"})
    );
    assert_eq!(
        signed.payload.item,
        json!({"type": "thought", "signature": "sig"})
    );
}

/// A place citation cites its URL when it states one; one stating
/// neither a URL nor a place id cites nothing.
#[test]
fn a_place_citation_cites_its_url_or_nothing() {
    let output = json!({"type": "model_output", "content": [{
        "type": "text",
        "text": "Berlin is a city.",
        "annotations": [
            {"type": "place_citation", "url": "https://maps.example/berlin", "place_id": "places/berlin", "name": "Berlin", "start_index": 0, "end_index": 6},
            {"type": "place_citation", "name": "Nowhere", "start_index": 0, "end_index": 6},
        ],
    }]});
    let response =
        decode(Mode::Unary, [whole(vec![output])]).expect("whole annotated text decodes");
    let [AssistantContent::Text(block)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    let cited: Vec<_> = block
        .citations()
        .iter()
        .map(|citation| (block.cited(citation), citation.sources.clone()))
        .collect();
    let url = Source::new(SourceLocation::Url {
        url: "https://maps.example/berlin".to_owned(),
    })
    .title("Berlin");
    assert_eq!(cited, [(Some("Berlin"), vec![url])]);
}

/// A `step.start` stating no step decodes to nothing: the deltas that
/// follow open the step as they would without it.
#[test]
fn a_start_stating_no_step_leaves_its_deltas_to_open_it() {
    let frames = [
        delta(0, json!({"type": "text", "text": "hi"})),
        stop(0),
        completed(),
    ];
    let bare = decode(
        Mode::Streaming,
        std::iter::once(json!({"event_type": "step.start", "index": 0})).chain(frames.clone()),
    )
    .expect("the reply with a bare start decodes");
    let plain = decode(Mode::Streaming, frames).expect("the reply decodes");
    let [AssistantContent::Text(text)] = bare.choice.as_slice() else {
        panic!("one text block: {:?}", bare.choice);
    };
    assert_eq!(text.text, "hi");
    assert_eq!(bare.message(), plain.message());
}

/// The fields of a hand-built event.
fn object(fields: Value) -> serde_json::Map<String, Value> {
    let Value::Object(fields) = fields else {
        panic!("event fields are an object: {fields}");
    };
    fields
}

/// Decode one SSE event built by hand, bypassing the classifier.
fn decode_event(event_type: &str, fields: Value) -> Result<Flow, ProviderError> {
    let wire = interactions();
    let request = CompletionRequest::new("hello");
    let shared = std::sync::Mutex::new(crate::wire::Shared::<Completion>::new(
        crate::test_utils::fold_for(&request, &wire, Mode::Streaming),
    ));
    let event = InteractionsEvent::Sse(SseEvent {
        event_type: event_type.to_owned(),
        fields: object(fields),
    });
    wire.decoder_in(Mode::Streaming)
        .decode(event, Out::new(&shared))
}

/// The classifier refuses a start or delta event without its object; a
/// decoder handed such an event directly refuses it with the same reason.
#[test]
fn a_hand_built_event_without_its_object_is_refused() {
    let no_step = "an Interactions `step.start` event carries no `step` object";
    let no_delta = "an Interactions `step.delta` event carries no `delta` object";
    for (event_type, fields, reason) in [
        ("step.start", json!({"index": 0, "step": "text"}), no_step),
        ("step.start", json!({"index": 0, "step": null}), no_step),
        ("step.delta", json!({"index": 0, "delta": "text"}), no_delta),
        ("step.delta", json!({"index": 0}), no_delta),
    ] {
        match decode_event(event_type, fields.clone()) {
            Err(ProviderError::Response(refusal)) => assert_eq!(refusal, reason, "{fields}"),
            Err(other) => panic!("{event_type} {fields}: {other:?}"),
            Ok(flow) => panic!("{event_type} {fields} decoded: {flow:?}"),
        }
    }
}

/// Only a streaming decoder keeps partial content at EOF; a unary reply
/// that ends before its interaction is truncated.
#[test]
fn only_a_streaming_decoder_keeps_partial_content_at_eof() {
    assert_eq!(
        interactions().decoder_in(Mode::Unary).incomplete_eof(),
        IncompleteEof::Refuse
    );
    assert_eq!(
        interactions().decoder_in(Mode::Streaming).incomplete_eof(),
        IncompleteEof::KeepPartial
    );
    let outcome = decode(
        Mode::Unary,
        [delta(0, json!({"type": "text", "text": "hi"}))],
    );
    assert!(
        matches!(outcome, Err(ProviderError::Truncated)),
        "{outcome:?}"
    );
}
