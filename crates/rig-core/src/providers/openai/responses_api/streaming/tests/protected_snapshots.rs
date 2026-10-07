//! Protected complete/partial snapshots at canonical replay boundaries.

use super::helpers::*;
use super::super::ResponsesStreamOptions;
use crate::message::{AssistantContent, CarriedPayload, CarriedPosition, CarriedReasoningPart};
use serde_json::json;

// F streaming/tests.rs:4046
#[test]
fn opaque_message_snapshots_preserve_parts_and_replace_raw_metadata_atomically() {
    let initial = json!({"type":"future_message","payload":{"list":[1],"old":true}});
    let final_part = json!({"type":"future_message","payload":{"list":[2]}});
    let item = opaque_message_fixture(json!([
        {"type":"output_text","text":"A"},final_part,{"type":"output_text","text":"B"}
    ]));
    let events = vec![
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":1,"part":initial}),
        json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":2,"part":final_part}),
        json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":3,"part":final_part}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":2,"sequence_number":4,"delta":"B"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":5,"item":item}),
        completed_with_output(json!([item])),
    ];
    let response = folded_body_with(ResponsesStreamOptions::strict(), &events).unwrap();
    assert_eq!(response.choice.len(), 3);
    assert_eq!(
        replay_opaque_choice(response.choice)[0]["content"],
        item["content"]
    );
}

// F streaming/tests.rs:4070
#[test]
fn opaque_message_late_part_retains_payload_in_source_order() {
    // The part first seen in the terminal folds at its content coordinate,
    // between the parts that streamed before it.
    let unknown = json!({"type":"future_message","payload":{"list":[1]}});
    let item = opaque_message_fixture(json!([
        {"type":"output_text","text":"A"},unknown,{"type":"output_text","text":"B"}
    ]));
    let events = vec![
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":2,"sequence_number":1,"delta":"B"}),
        completed_with_output(json!([item])),
    ];
    let response = folded_body_with(ResponsesStreamOptions::strict(), &events).unwrap();
    assert_eq!(
        replay_opaque_choice(response.choice)[0]["content"],
        item["content"]
    );
}

// F streaming/tests.rs:4090
#[test]
fn opaque_reasoning_part_events_survive_missing_done_and_truncation_with_known_fragments() {
    let unknown = json!({"type":"future_summary","payload":{"list":[1]}});
    let events = vec![
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_opaque","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}),
        json!({"type":"response.reasoning_summary_part.added","item_id":"rs_opaque","output_index":0,"summary_index":1,"sequence_number":1,"part":unknown}),
        json!({"type":"response.reasoning_summary_part.done","item_id":"rs_opaque","output_index":0,"summary_index":1,"sequence_number":2,"part":unknown}),
    ];
    for terminal in [false, true] {
        let mut frames = events.clone();
        if terminal {
            frames.push(completed_with_output(json!([])));
        }
        let response = folded_body_with(ResponsesStreamOptions::strict(), &frames).unwrap();
        let [AssistantContent::CarriedReasoning(reasoning)] = response.choice.as_slice() else {
            panic!("one reasoning item: {:?}", response.choice);
        };
        assert_eq!(
            reasoning.content,
            vec![
                CarriedReasoningPart::Summary("known".into()),
                CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(unknown.clone(), Some("openai".into())).with_position(CarriedPosition::ReasoningSummary { part_index: 1 }))
            ]
        );
        let replay = replay_opaque_choice(response.choice);
        assert_eq!(replay[0]["id"], "rs_opaque");
        assert_eq!(
            replay[0]["summary"],
            json!([{"type":"summary_text","text":"known"},unknown])
        );
    }
}

// F streaming/tests.rs:4123
#[test]
fn opaque_reasoning_item_and_terminal_snapshots_emit_one_complete_restatement() {
    let unknown = json!({"type":"future_content","text":"opaque","list":[1]});
    let item = json!({"type":"reasoning","id":"rs_opaque","summary":[{"type":"summary_text","text":"known"}],"content":[unknown,{"type":"reasoning_text","text":"visible"}]});
    let events = vec![
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}),
        completed_with_output(json!([item])),
    ];
    let response = folded_body_with(ResponsesStreamOptions::strict(), &events).unwrap();
    assert_eq!(response.choice.len(), 1);
    let replay = replay_opaque_choice(response.choice);
    assert_eq!(replay[0]["summary"], item["summary"]);
    assert_eq!(replay[0]["content"], item["content"]);
}

// F streaming/tests.rs:4140
#[test]
fn opaque_siblings_do_not_duplicate_annotations_when_text_snapshots_grow() {
    let unknown = json!({"type":"future_message","value":[1]});
    let annotations =
        json!([{"type":"url_citation","url":"https://example.com","start_index":0,"end_index":1}]);
    let item = opaque_message_fixture(json!([
        {"type":"output_text","text":"AB","annotations":annotations},unknown
    ]));
    let terminal = opaque_message_fixture(json!([
        {"type":"output_text","text":"ABC","annotations":annotations},unknown
    ]));
    let events = vec![
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"AB","annotations":annotations}}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}),
        completed_with_output(json!([terminal])),
    ];
    let response = folded_body_with(ResponsesStreamOptions::strict(), &events).unwrap();
    let replay = replay_opaque_choice(response.choice);
    assert_eq!(replay[0]["content"], terminal["content"]);
}

// F streaming/tests.rs:4162
#[test]
fn opaque_message_full_snapshot_annotations_append_only_the_new_suffix() {
    let a = json!({"url":"a","range":[0,1]});
    let b = json!({"url":"b","range":[1,2]});
    let opaque = json!({"type":"future_part","payload":[1]});
    let item =
        opaque_message_fixture(json!([{"type":"output_text","text":"A","annotations":[a]},opaque]));
    let terminal = opaque_message_fixture(
        json!([{"type":"output_text","text":"AB","annotations":[a,b]},opaque]),
    );
    let response = folded_body_with(ResponsesStreamOptions::strict(),&[
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":0,"item":item}),
        completed_with_output(json!([terminal])),
    ]).unwrap();
    assert_eq!(
        replay_opaque_choice(response.choice)[0]["content"],
        terminal["content"]
    );
}

// F streaming/tests.rs:4437
#[test]
fn reasoning_completion_keeps_authoritative_ciphertext_and_signature() {
    let item = |cipher: &str, signature: &str| json!({"type":"reasoning","id":"rs_cipher","summary":[],"encrypted_content":cipher,"signature":signature});
    let provisional = item("provisional", "provisional-signature");
    let done = item("done", "done-signature");
    let mut terminal = item("terminal", "terminal-signature");
    let opaque = json!({"type":"future_summary","value":[1]});
    terminal["summary"] = json!([opaque]);
    let response = folded_body_with(ResponsesStreamOptions::strict(),&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":provisional}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":done}),
        completed_with_output(json!([terminal])),
    ]).unwrap();
    let replay = replay_opaque_choice(response.choice);
    assert_eq!(replay[0]["encrypted_content"], "done");
    assert_eq!(replay[0]["signature"], "done-signature");
    assert_eq!(replay[0]["summary"], json!([opaque]));
}

// F streaming/tests.rs:4793
#[test]
fn a_streamed_message_takes_no_phase_from_its_snapshots_while_unary_keeps_it() {
    // Known issue: the stream keeps ca81's replay bytes, so a streamed
    // message replays without the phase a unary reply of it carries.
    let mut item = opaque_message_fixture(json!([{"type":"output_text","text":"A"}]));
    item["phase"] = json!("final_answer");
    let streamed = folded_body_with(ResponsesStreamOptions::strict(), &[
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        completed_with_output(json!([item])),
    ])
    .unwrap();
    let replay = replay_opaque_choice(streamed.choice);
    assert_eq!(
        replay[0]["content"],
        json!([{"type":"output_text","text":"A"}])
    );
    assert!(replay[0].get("phase").is_none(), "{replay}");

    // A message only the terminal states keeps its phase, as at ca81.
    let terminal_only = folded_body_with(
        ResponsesStreamOptions::strict(),
        &[completed_with_output(json!([item]))],
    )
    .unwrap();
    assert_eq!(
        replay_opaque_choice(terminal_only.choice)[0]["phase"],
        "final_answer"
    );

    let unary = replay_opaque_choice(unary_choice(json!([item])));
    assert_eq!(unary[0]["phase"], "final_answer");
}

// F streaming/tests.rs:4828
#[test]
fn reasoning_completion_keeps_its_authoritative_provider_id() {
    let item =
        |id: &str| json!({"type":"reasoning","id":id,"summary":[],"encrypted_content":"cipher"});
    let response = folded_body_with(ResponsesStreamOptions::strict(), &[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":item("rs_added")}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item("rs_done")}),
        completed_with_output(json!([item("rs_terminal")])),
    ])
    .unwrap();
    assert_eq!(replay_opaque_choice(response.choice)[0]["id"], "rs_done");
}

// F streaming/tests.rs:4841
#[test]
fn reasoning_first_published_at_the_flush_keeps_its_output_slot() {
    // Slot 0's unknown part waits for its parent kind until the terminal says
    // reasoning; slot 1's text streamed meanwhile. The deferred close carries
    // slot 0's coordinate, so the fold matches the unary order.
    let unknown = json!({"type":"future_content","payload":[1]});
    let reasoning = json!({"type":"reasoning","id":"rs_late","summary":[],"content":[unknown]});
    let message = opaque_message_fixture(json!([{"type":"output_text","text":"answer"}]));
    let streamed = folded_body_with(ResponsesStreamOptions::strict(), &[
        json!({"type":"response.content_part.added","item_id":"rs_late","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":1,"content_index":0,"sequence_number":1,"delta":"answer"}),
        completed_with_output(json!([reasoning, message])),
    ])
    .unwrap()
    .choice;
    let unary = unary_choice(json!([reasoning, message]));
    assert!(
        matches!(&streamed[0], AssistantContent::CarriedReasoning(r) if r.content == vec![CarriedReasoningPart::OpaqueContent(CarriedPayload::new(unknown.clone(), Some("openai".into())).with_position(CarriedPosition::ReasoningContent { part_index: 0 }))]),
        "{streamed:?}"
    );
    assert!(
        matches!(&streamed[1], AssistantContent::CarriedText(t) if t.text() == "answer"),
        "{streamed:?}"
    );
    assert_eq!(streamed.len(), 2);
    assert_eq!(streamed, unary);
}

// F streaming/tests.rs:4871
#[test]
fn an_empty_ciphertext_at_done_does_not_block_the_terminal_ciphertext() {
    let item = |cipher: &str| json!({"type":"reasoning","id":"rs_empty","summary":[],"encrypted_content":cipher});
    let response = folded_body_with(ResponsesStreamOptions::strict(), &[
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":0,"item":item("")}),
        completed_with_output(json!([item("real")])),
    ])
    .unwrap();
    assert_eq!(
        replay_opaque_choice(response.choice)[0]["encrypted_content"],
        "real"
    );
}


