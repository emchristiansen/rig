use super::{ResponsesDecoder, ResponsesStreamOptions};
use crate::completion::CompletionRequest;
use crate::driver::Decoded;
use crate::message::{AssistantContent, Message};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::streaming::{PartKind, SequenceError, StreamEvent, Transcript};
use crate::wire::{Mode, Operation, Wire, WireFrame};
use serde_json::{Value, json};

fn delta(sequence: u64, item: &str, output: u64, part: u64, text: &str) -> Value {
    json!({"type":"response.output_text.delta","sequence_number":sequence,"item_id":item,
        "output_index":output,"content_index":part,"delta":text})
}

fn message(text: &str) -> Value {
    json!({"type":"message","id":"msg_1","role":"assistant","status":"completed",
        "content":[{"type":"output_text","text":text,"annotations":[]}]})
}

fn terminal(output: Value) -> Value {
    json!({"type":"response.completed","sequence_number":20,
        "response":{"id":"resp_1","object":"response","created_at":1,"model":"gpt-5.4",
            "status":"completed","output":output}})
}

fn decode(frames: &[Value]) -> Decoded<Completion> {
    crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        "openai",
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string()))
    )
}

fn transcript(result: &Decoded<Completion>, closed: bool) -> Transcript {
    let items = result
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect::<Vec<_>>();
    let encoded = serde_json::to_value(items).expect("actual decoder stream");
    if closed {
        Transcript::parse(encoded).expect("actual Parts each close once")
    } else {
        Transcript::parse_prefix(encoded).expect("actual partial stream")
    }
}

fn activation_counts(result: &Decoded<Completion>) -> (usize, usize, usize, Vec<String>) {
    let transcript = transcript(result, false);
    let part = transcript
        .events()
        .find(|event| {
            matches!(
                event,
                StreamEvent::Start {
                    kind: PartKind::Text,
                    ..
                }
            )
        })
        .expect("real first text Part")
        .part();
    let events = transcript
        .events()
        .filter(|event| event.part() == part)
        .collect::<Vec<_>>();
    let starts = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::Start { .. }))
        .count();
    let resumes = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::Resume { .. }))
        .count();
    let ends = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::End { .. }))
        .count();
    let fragments = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    (starts, resumes, ends, fragments)
}

#[test]
fn actual_reasoning_interleave_has_start_plus_resume_and_one_canonical_text_part() {
    let reasoning = json!({"type":"response.reasoning_summary_text.delta","sequence_number":1,
        "item_id":"rs_2","output_index":1,"summary_index":0,"delta":"because"});
    let result = decode(&[
        delta(0, "msg_1", 0, 0, "hello "),
        reasoning,
        delta(2, "msg_1", 0, 0, "world"),
        terminal(json!([])),
    ]);
    assert_eq!(
        activation_counts(&result),
        (1, 1, 1, vec!["hello ".to_owned(), "world".to_owned()])
    );
    transcript(&result, true);
    let response = result.outcome.expect("genuine terminal");
    let texts = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            AssistantContent::CarriedText(text) => Some(text.text()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(texts, vec!["hello world"]);
    assert!(response.choice.iter().any(|content| matches!(
        content,
        AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_)
    )));
}

#[test]
fn returning_after_function_added_or_arguments_delta_uses_the_same_live_text_part() {
    for interleave in [
        json!({"type":"response.output_item.added","sequence_number":1,"output_index":1,
            "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","status":"in_progress","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","sequence_number":1,"output_index":1,"item_id":"fc_1","delta":"{"}),
        delta(1, "msg_2", 2, 0, "other"),
    ] {
        let result = decode(&[
            delta(0, "msg_1", 0, 0, "hello "),
            interleave,
            delta(2, "msg_1", 0, 0, "world"),
        ]);
        let (starts, resumes, _, fragments) = activation_counts(&result);
        assert_eq!((starts, resumes), (1, 1));
        assert_eq!(fragments, vec!["hello ", "world"]);
        assert!(result.outcome.is_err(), "no synthetic terminal");
    }
}

#[test]
fn unchanged_done_and_terminal_snapshots_have_one_start_no_resume_no_duplicate_fragments_and_one_close()
 {
    let item = message("A");
    let done = json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item.clone()});
    let result = decode(&[
        delta(0, "msg_1", 0, 0, "A"),
        done,
        terminal(json!([item.clone()])),
    ]);
    assert_eq!(activation_counts(&result), (1, 0, 1, vec!["A".to_owned()]));
    transcript(&result, true);
    let response = result.outcome.expect("genuine done and terminal");
    assert_eq!(response.choice.len(), 1);
    let mut request = CompletionRequest::from(vec![
        Message::user("first"),
        response.message().expect("canonical message"),
        Message::user("next"),
    ]);
    request.model = Some("different-model".to_owned());
    let target = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let prepared =
        Completion::prepare(request, &target.describe()).expect("actual history preflight");
    let encoded = target
        .encode(prepared, Mode::Unary)
        .expect("actual canonical request replay");
    let body = crate::test_utils::json_body(&encoded.request);
    let replay = body["input"]
        .as_array()
        .expect("actual input")
        .iter()
        .filter(|input| input["role"] == "assistant")
        .collect::<Vec<_>>();
    assert_eq!(
        replay,
        vec![&item],
        "approved explicit close preserves exact canonical replay"
    );
}

#[test]
fn resumed_text_prefix_still_requires_a_real_close() {
    let second = delta(1, "msg_2", 1, 0, "other");
    let result = decode(&[
        delta(0, "msg_1", 0, 0, "hello "),
        second,
        delta(2, "msg_1", 0, 0, "world"),
    ]);
    assert!(result.outcome.is_err());
    let transcript = transcript(&result, false);
    assert_eq!(activation_counts(&result).0, 1);
    assert_eq!(activation_counts(&result).1, 1);
    // The decoder flush may honestly close partial observed text at EOF;
    // inspect the actual live event prefix ending at its last text fragment.
    let mut encoded = serde_json::to_value(&transcript).expect("actual flushed events");
    let items = encoded.as_array_mut().expect("items");
    let last_text = items
        .iter()
        .rposition(|item| item["value"]["event"] == "text")
        .expect("actual resumed text");
    items.truncate(last_text + 1);
    Transcript::parse_prefix(encoded.clone()).expect("live partial prefix");
    assert!(matches!(
        Transcript::parse(encoded),
        Err(SequenceError::Unclosed(_))
    ));
}

#[test]
fn changed_snapshot_metadata_or_phase_reactivates_but_identical_or_empty_metadata_does_not() {
    let mut initial = message("A");
    initial["phase"] = json!("analysis");
    let mut metadata = initial.clone();
    metadata["content"][0]["annotations"] =
        json!([{"type":"url_citation","url":"https://example.test/source","title":"source"}]);
    let mut phase = initial.clone();
    phase["phase"] = json!("final_answer");
    let mut empty = initial.clone();
    empty["content"][0]["future_null"] = Value::Null;
    empty["content"][0]["future_array"] = json!([]);
    empty["content"][0]["future_object"] = json!({});
    for (snapshot, resumes) in [(initial.clone(), 0), (metadata, 1), (phase, 1), (empty, 0)] {
        let added = json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":initial.clone()});
        let reasoning = json!({"type":"response.reasoning_summary_text.delta","sequence_number":1,
            "item_id":"rs_2","output_index":1,"summary_index":0,"delta":"because"});
        let done = json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":snapshot.clone()});
        let result = decode(&[added, reasoning, done, terminal(json!([snapshot.clone()]))]);
        assert_eq!(
            activation_counts(&result),
            (1, resumes, 1, vec!["A".to_owned()])
        );
        transcript(&result, true);
        let response = result
            .outcome
            .expect("genuine changed snapshot and terminal");
        let carried = response
            .choice
            .iter()
            .find_map(|content| match content {
                AssistantContent::CarriedText(text) => Some(text),
                _ => None,
            })
            .expect("actual carried known part");
        assert_eq!(
            carried.payload.item, snapshot["content"][0],
            "activation projection leaves exact raw siblings intact"
        );
        assert_eq!(carried.phase.as_deref(), snapshot["phase"].as_str());
        assert_eq!(carried.text(), "A");
    }
}
