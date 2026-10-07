//! Protected raw custom input, namespaces and authoritative argument verdicts.

use super::helpers::*;
use super::super::ResponsesStreamOptions;
use crate::message::AssistantContent;
use serde_json::json;

// F streaming/tests.rs:2765
#[test]
fn a_custom_tool_call_item_folds_into_a_custom_call_with_raw_input() {
    let item = json!({
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_custom",
        "name": "apply_patch",
        "namespace": "editor",
        "input": "{\"looks\":\"like json\"}",
        "status": "completed",
    });
    let events = [
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": item,
        }),
        completed_with_output(json!([item])),
    ];

    for options in [
        ResponsesStreamOptions::strict(),
        ResponsesStreamOptions::strict_with_immediate_tool_calls(),
    ] {
        let response = folded_body_with(options, &events).expect("the custom call folds");
        let [AssistantContent::CustomToolCall(call)] = response.choice.as_slice() else {
            panic!(
                "expected exactly one custom call, got {:?}",
                response.choice
            );
        };
        assert_eq!(call.name.as_str(), "apply_patch");
        assert_eq!(call.namespace.as_deref(), Some("editor"));
        assert_eq!(call.input, "{\"looks\":\"like json\"}");
        assert_eq!(call.id.provider().expect("provider correlation ID").as_str(), "call_custom");
        assert_eq!(call.native.as_ref().expect("whole custom item is native").item["id"], "ctc_1");
    }
}

// F streaming/tests.rs:2810
#[test]
fn a_namespaced_function_call_keeps_its_namespace_streamed_and_drained() {
    let added = json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_ns",
            "call_id": "call_ns",
            "name": "search",
            "namespace": "web",
            "arguments": "",
            "status": "in_progress",
        },
    });
    let delta = json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_ns",
        "output_index": 0,
        "sequence_number": 2,
        "delta": "{\"q\":\"rig\"}",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 3,
        "item": {
            "type": "function_call",
            "id": "fc_ns",
            "call_id": "call_ns",
            "name": "search",
            "namespace": "web",
            "arguments": "{\"q\":\"rig\"}",
            "status": "completed",
        },
    });
    let terminal = completed_with_output(json!([]));

    for (case, events) in [
        (
            "done item",
            vec![added.clone(), delta.clone(), done, terminal.clone()],
        ),
        ("terminal drain", vec![added, delta, terminal]),
    ] {
        let response = folded_body_with(ResponsesStreamOptions::strict(), &events)
            .unwrap_or_else(|error| panic!("{case}: the call folds: {error}"));
        let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
            panic!(
                "{case}: expected one function call, got {:?}",
                response.choice
            );
        };
        assert_eq!(call.function.name.as_str(), "search", "{case}");
        assert_eq!(call.function.namespace.as_deref(), Some("web"), "{case}");
        assert_eq!(call.function.arguments_value(), json!({"q": "rig"}), "{case}");
    }
}

// F streaming/tests.rs:3155
#[tokio::test]
async fn a_streamed_asserted_call_with_unparseable_arguments_is_refused_by_name() {
    for status in ["completed", "in_progress"] {
        let (calls, errors) =
            delivered_calls_and_errors(&[malformed_function_call_done(status, 1)]).await;
        assert!(
            calls.is_empty(),
            "{status}: no call is delivered: {calls:?}"
        );
        let [error] = errors.as_slice() else {
            panic!("{status}: exactly one refusal, got {errors:?}");
        };
        assert!(
            error
                .message
                .starts_with("tool call `add` arrived with malformed JSON input"),
            "{status}: the refusal names the call: {error:?}"
        );
        assert!(
            matches!(
                &error.detail,
                Some(crate::error::ErrorDetail::MalformedToolInput(detail))
                    if detail.name.as_str() == "add" && detail.raw == "{\"x\":481"
            ),
            "{status}: the refusal keeps the raw restatement for recovery: {error:?}"
        );
    }
}

// F streaming/tests.rs:3186
#[tokio::test]
async fn a_streamed_unasserted_call_with_unparseable_arguments_is_dropped() {
    let (calls, errors) =
        delivered_calls_and_errors(&[malformed_function_call_done("incomplete", 1)]).await;
    assert!(calls.is_empty(), "the call is dropped: {calls:?}");
    assert!(errors.is_empty(), "dropping is not an error: {errors:?}");
}

// F streaming/tests.rs:3196
#[tokio::test]
async fn streamed_fragments_repeating_a_malformed_restatement_take_its_verdict() {
    let (calls, errors) = delivered_calls_and_errors(&[
        argument_delta("{\"x\":481"),
        malformed_function_call_done("completed", 2),
    ])
    .await;
    assert!(calls.is_empty(), "no call is delivered: {calls:?}");
    assert!(
        matches!(errors.as_slice(), [error]
            if error.message.starts_with("tool call `add` arrived with malformed JSON input")),
        "the asserted call is refused by name: {errors:?}"
    );

    let (calls, errors) = delivered_calls_and_errors(&[
        argument_delta("{\"x\":481"),
        malformed_function_call_done("incomplete", 2),
    ])
    .await;
    assert!(
        calls.is_empty(),
        "the unasserted call is dropped: {calls:?}"
    );
    assert!(errors.is_empty(), "dropping is not an error: {errors:?}");
}

// F streaming/tests.rs:3225
#[tokio::test]
async fn a_parsing_delta_cannot_rescue_a_malformed_restatement() {
    for delta in ["   ", "{}"] {
        let (calls, errors) = delivered_calls_and_errors(&[
            argument_delta(delta),
            malformed_function_call_done("completed", 2),
        ])
        .await;
        assert!(
            calls.is_empty(),
            "{delta:?}: a parseable delta does not make a malformed restatement executable: \
             {calls:?}"
        );
        assert!(
            matches!(errors.as_slice(), [error]
                if error.message.starts_with("tool call `add` arrived with malformed JSON input")),
            "{delta:?}: the asserted call is refused by name: {errors:?}"
        );

        let (calls, errors) = delivered_calls_and_errors(&[
            argument_delta(delta),
            malformed_function_call_done("incomplete", 2),
        ])
        .await;
        assert!(
            calls.is_empty(),
            "{delta:?}: the unasserted call is dropped, never fabricated: {calls:?}"
        );
        assert!(
            errors.is_empty(),
            "{delta:?}: dropping is not an error: {errors:?}"
        );
    }
}

// F streaming/tests.rs:3262
#[tokio::test]
async fn streaming_and_unary_refuse_an_asserted_malformed_call_alike() {
    let (_, errors) =
        delivered_calls_and_errors(&[malformed_function_call_done("completed", 1)]).await;
    let [streamed] = errors.as_slice() else {
        panic!("exactly one streamed refusal, got {errors:?}");
    };

    let mut response = sample_response("completed");
    response["output"] = json!([malformed_function_call_done("completed", 1)["item"]]);
    let unary = crate::driver::feed_frames!(
        super::super::ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()),
        "openai",
        [crate::wire::WireFrame::Text(response.to_string())]
    ).outcome.expect_err("the same real decoder refuses the unary malformed call");
    assert_eq!(streamed.message, unary.report().message);
    assert!(matches!(unary.report().detail, Some(crate::error::ErrorDetail::MalformedToolInput(_))));
}

// F streaming/tests.rs:3996
#[test]
fn dropped_ordered_call_leaves_no_placeholder() {
    let provider = json!({
        "type": "web_search_call", "id": "ws_1", "status": "completed",
        "action": {"query": "rig"}
    });
    let incomplete = json!({
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "lookup", "arguments": "{", "status": "incomplete"
    });
    let events = [
        json!({
            "type": "response.output_item.added", "output_index": 1,
            "sequence_number": 1, "item": incomplete
        }),
        json!({
            "type": "response.function_call_arguments.delta", "item_id": "fc_1",
            "output_index": 1, "sequence_number": 2, "delta": "{"
        }),
        json!({
            "type": "response.output_item.done", "output_index": 0,
            "sequence_number": 3, "item": provider
        }),
        json!({
            "type": "response.output_item.done", "output_index": 1,
            "sequence_number": 4, "item": incomplete
        }),
        completed_with_output(json!([provider, incomplete])),
    ];

    let choice = folded_body_with(ResponsesStreamOptions::strict(), &events)
        .expect("the incomplete call is dropped")
        .choice;
    assert_eq!(response_choice_labels(&choice), ["provider"]);
}


