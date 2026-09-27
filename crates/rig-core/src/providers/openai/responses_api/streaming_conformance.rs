//! Public-schema checks for existing Responses streaming fixtures.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::openapi_schema::{self, Schema};
use super::streaming::{StreamingCompletionChunk, classify_responses_frame};
use crate::providers::internal::wire::WireEvent;
use crate::test_utils::streaming_conformance::{WireInput, fixtures};

fn event(frame: &WireInput) -> Value {
    let bytes = frame
        .as_bytes()
        .expect("Responses fixtures use byte frames");
    let text = std::str::from_utf8(bytes).expect("Responses SSE fixtures are UTF-8");
    let data = text
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("a Responses fixture has an SSE data line");
    serde_json::from_str(data).expect("a selected Responses fixture is JSON")
}

#[test]
fn pinned_public_openapi_fixture_has_the_recorded_bytes_and_provenance() {
    let document = openapi_schema::document();
    assert_eq!(openapi_schema::bytes().len(), 3_489_474);
    assert_eq!(
        format!("{:x}", Sha256::digest(openapi_schema::bytes())),
        "0282b31f42995cdad456d15054bdb118d8cae368a0892e517d4557a7b2fe82dd"
    );
    assert_eq!(document["openapi"], "3.1.0");
    assert_eq!(document["info"]["version"], "2.3.0");
    assert_eq!(document["info"]["license"]["identifier"], "MIT");
}

#[test]
fn selected_existing_responses_stream_fixtures_satisfy_the_public_event_union() {
    let fixture = fixtures::openai_responses::fixture();
    let mut frames = Vec::new();
    frames.extend(fixture.tool_call_frames.iter());
    frames.extend(
        fixture
            .partial_tool_call_frames
            .as_ref()
            .expect("Responses has partial tool frames"),
    );
    frames.extend(
        &fixture
            .refusal
            .as_ref()
            .expect("Responses has refusal frames")
            .frames,
    );
    frames.push(
        fixture
            .unknown_event_frame
            .as_ref()
            .expect("the Rig-unknown web-search event is present"),
    );
    let incomplete = fixtures::openai_responses::incomplete_mid_tool_call_frames();
    frames.extend(incomplete.iter().take(5));

    for frame in frames {
        openapi_schema::assert_valid(Schema::ResponseStreamEvent, &event(frame));
    }
}

#[test]
fn public_known_web_search_event_can_remain_unknown_to_rig() {
    let fixture = fixtures::openai_responses::fixture();
    let event = event(
        fixture
            .unknown_event_frame
            .as_ref()
            .expect("the public web-search event is present"),
    );
    openapi_schema::assert_valid(Schema::ResponseStreamEvent, &event);

    assert!(matches!(
        classify_responses_frame(&event.to_string()),
        WireEvent::<StreamingCompletionChunk>::Unknown { event_type, .. }
            if event_type == "response.web_search_call.searching"
    ));
}

#[test]
fn future_event_fails_the_public_union_but_remains_an_unknown_rig_event() {
    let event = json!({
        "type": "response.future.delta",
        "sequence_number": 1,
        "delta": "future"
    });
    openapi_schema::assert_invalid(Schema::ResponseStreamEvent, &event);

    assert!(matches!(
        classify_responses_frame(&event.to_string()),
        WireEvent::<StreamingCompletionChunk>::Unknown { event_type, .. }
            if event_type == "response.future.delta"
    ));
}

#[test]
fn malformed_known_fixture_fails_both_the_public_union_and_rig_decode() {
    let fixture = fixtures::openai_responses::fixture();
    let event = event(
        fixture
            .defective_known_frame
            .as_ref()
            .expect("the malformed known event is present"),
    );
    openapi_schema::assert_invalid(Schema::ResponseStreamEvent, &event);
    assert!(matches!(
        classify_responses_frame(&event.to_string()),
        WireEvent::<StreamingCompletionChunk>::Corrupt(_)
    ));
}

/// The public Responses WebSocket union has no `response.done` event. Its
/// similarly named Realtime event is a different API and cannot validate the
/// sparse terminal that the Codex subscription transport sends.
#[test]
fn sparse_responses_websocket_done_has_no_public_responses_schema() {
    let event = json!({
        "type": "response.done",
        "response": {"id": "resp_1", "status": "completed"}
    });
    openapi_schema::assert_invalid(Schema::ResponseStreamEvent, &event);
    let server_union = &openapi_schema::document()["components"]["schemas"]["ResponsesServerEvent"];
    let serialized =
        serde_json::to_string(server_union).expect("the public server union serializes");
    assert!(!serialized.contains("response.done"));
}
