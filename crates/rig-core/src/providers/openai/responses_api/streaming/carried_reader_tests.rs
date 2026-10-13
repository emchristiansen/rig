//! Responses reader regressions through the real driver and history encoder.

use crate::completion::CompletionRequest;
use crate::message::{AssistantContent, CarriedPosition, Message};
use crate::operation::Completion;
use crate::providers::openai::wire::OpenAIConfig;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};
use crate::wire::{Mode, Operation, Wire};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Cassette {
    then: RecordedResponse,
}

#[derive(Deserialize)]
struct RecordedResponse {
    status: u16,
    header: Vec<RecordedHeader>,
    body: String,
}

#[derive(Deserialize)]
struct RecordedHeader {
    name: String,
    value: String,
}

/// The unchanged recorded reply must survive actual decoding, canonical
/// history adaptation, and request encoding as the exact compaction item.
#[tokio::test]
async fn recorded_compaction_decodes_as_a_carried_item_and_replays_exactly() {
    const FIXTURE: &[u8] = include_bytes!(
        "../../../../../../rig-cassette/fixtures/cassettes/openai/stateless_replay_matrix/compaction_item_decodes_on_the_response.yaml"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURE)),
        "76f309a28c4140364ad5a37ca43e89b1dcb79cec205edaa068c43049b98f1218"
    );
    let cassette: Cassette = serde_yaml::from_slice(FIXTURE).expect("recorded YAML fixture");
    assert_eq!(cassette.then.status, 200);
    let raw: Value = serde_json::from_str(&cassette.then.body).expect("recorded response JSON");
    let compaction = raw["output"][0].clone();
    assert_eq!(compaction["type"], "compaction");
    let mut headers = http::HeaderMap::new();
    for header in cassette.then.header {
        headers.append(
            http::HeaderName::from_bytes(header.name.as_bytes()).expect("recorded header name"),
            http::HeaderValue::from_str(&header.value).expect("recorded header value"),
        );
    }
    let http = SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        cassette.then.body.into(),
        headers,
    )]);
    let wire = OpenAIConfig::new("fixture-key").responses("gpt-5.6-sol");
    let response = crate::driver::Model::new(wire, http.clone())
        .call(CompletionRequest::new(
            "Remember the codeword ALPHA-17. Reply exactly: ACK-1",
        ))
        .await
        .expect("the actual Responses decoder reads the recorded reply");
    assert_eq!(response.raw, raw);
    assert_eq!(response.text(), "ACK-1");
    assert_eq!(response.provider_request_id.as_deref(), Some("req_REDACTED_1"));
    assert_eq!(response.response_id(), raw["id"].as_str());
    assert_eq!(response.message_id.as_ref().map(|id| id.as_str()), raw["output"][1]["id"].as_str());
    assert_eq!(response.usage.input_tokens, Some(21));
    assert_eq!(response.usage.output_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(28));
    assert_eq!(response.usage.cached_input_tokens, Some(0));
    assert_eq!(response.usage.cache_creation_input_tokens, Some(0));
    assert_eq!(response.usage.reasoning_tokens, Some(0));
    let [
        AssistantContent::ProviderItem(item),
        AssistantContent::CarriedText(_),
    ] = response.choice.as_slice()
    else {
        panic!("compaction then text, got {:?}", response.choice);
    };
    assert_eq!(item.payload.item, compaction);
    assert_eq!(item.payload.provider.as_deref(), Some("openai"));
    let history = response.message().expect("canonical assistant history");
    let Message::Assistant(turn) = &history else {
        panic!("assistant history");
    };
    assert_eq!(turn.content, response.choice);

    // Carried replay follows its own issuer contract across model changes.
    let destination = OpenAIConfig::new("fixture-key").responses("different-model");
    let request = CompletionRequest::new("Repeat the codeword").message(history);
    let request =
        Completion::prepare(request, &destination.describe()).expect("real history adaptation");
    let encoded = destination
        .encode(request, Mode::Unary)
        .expect("real Responses encoder");
    let body = crate::test_utils::json_body(&encoded.request);
    let replayed = body["input"]
        .as_array()
        .expect("request input")
        .iter()
        .find(|item| item["type"] == "compaction")
        .expect("compaction is replayed");
    assert_eq!(replayed, &compaction);
    assert_eq!(
        serde_json::to_string(replayed).expect("encoded item"),
        serde_json::to_string(&compaction).expect("recorded item")
    );
    assert_eq!(http.requests().len(), 1);
    assert_eq!(http.remaining_responses(), 0);
}

use super::{
    IncompleteTerminal, ResponsesDecoder, ResponsesStreamOptions, classify_responses_payload,
};
use crate::error::{ErrorDetail, FrameEvidence, ProviderError};
use crate::streaming::{DiscardReason, Item, PartKind, SourceOrder, StreamEvent, Transcript};
use crate::test_utils::MockStreamingClient;
use crate::wire::{Decoder, WireEvent, WireFrame};
use futures::StreamExt;
use serde_json::json;

fn response(output: Value) -> Value {
    json!({"id":"resp_1", "object":"response", "created_at":1, "status":"completed",
        "model":"model_1", "output":output,
        "usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}})
}

fn terminal(output: Value) -> Value {
    json!({"type":"response.completed","sequence_number":99,"response":response(output)})
}

fn message(parts: Value) -> Value {
    json!({"type":"message","id":"msg_1","role":"assistant","status":"completed","content":parts})
}

fn done(index: u64, item: Value) -> Value {
    json!({"type":"response.output_item.done","sequence_number":2,"output_index":index,"item":item})
}

fn delta(index: u64, part: u64, text: &str) -> Value {
    json!({"type":"response.output_text.delta","sequence_number":1,"output_index":index,
        "content_index":part,"item_id":"msg_1","delta":text})
}

fn function(status: &str, arguments: &str, namespace: Option<&str>) -> Value {
    let mut item = json!({"type":"function_call","id":"fc_1","call_id":"call_1",
        "name":"add","arguments":arguments,"status":status});
    if let Some(namespace) = namespace {
        item["namespace"] = json!(namespace);
    }
    item
}

fn argument_delta(arguments: &str) -> Value {
    json!({"type":"response.function_call_arguments.delta","sequence_number":1,
        "output_index":1,"item_id":"fc_1","delta":arguments})
}

fn decoded(
    options: ResponsesStreamOptions,
    frames: &[Value],
) -> crate::driver::Decoded<Completion> {
    crate::driver::feed_frames!(
        ResponsesDecoder::new("openai", options),
        super::document::Response::with_strict_reader(false),
        "openai",
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string()))
    )
}

fn streamed(
    frames: &[Value],
    incomplete: IncompleteTerminal,
) -> crate::streaming::CompletionStream {
    let sse: String = frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect();
    let wire = OpenAIConfig::new("fixture-key")
        .responses("model_1")
        .with_streamed_incomplete(incomplete);
    crate::driver::Model::new(
        wire,
        MockStreamingClient {
            sse_bytes: sse.into(),
        },
    )
    .stream(CompletionRequest::new("fixture"))
    .expect("actual streaming request")
}

#[test]
fn strict_known_shapes_keep_received_corruption_and_cannot_fall_back_to_a_whole_body() {
    let malformed = [
        r#"{"type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"text":"done-shaped"}"#,
        r#"{"type":"response.output_text.done","sequence_number":1,"output_index":0,"content_index":0,"delta":"delta-shaped"}"#,
        r#"{"type":"response.reasoning_summary_text.delta","sequence_number":1,"output_index":0,"summary_index":0,"text":"done-shaped"}"#,
        r#"{"type":"response.refusal.delta","sequence_number":1,"output_index":0,"content_index":0,"delta":7}"#,
        r#"{"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc","call_id":"call","name":"add","arguments":{},"status":"completed"}}"#,
        r#"{"type":"response.output_text.delta","type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"delta":"duplicate"}"#,
        r#"{"type":"response.output_text.delta","output":[],"error":{"message":"not a rescue"},"delta":"missing indices"}"#,
    ];
    for raw in malformed {
        let WireEvent::Corrupt(corrupt) = classify_responses_payload(raw) else {
            panic!("known malformed frame must stay corrupt: {raw}");
        };
        assert_eq!(corrupt.frame(), Some(raw));
        assert_eq!(corrupt.frame_bytes(), Some(raw.as_bytes()));
    }
    let mut invalid_usage = terminal(json!([]));
    invalid_usage["response"]["usage"]["output_tokens"] = json!("five");
    assert!(matches!(
        classify_responses_payload(&invalid_usage.to_string()),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn original_custom_input_delta_and_done_tags_pass_through_verbatim() {
    for raw in [
        json!({"type":"response.custom_tool_call_input.delta","delta":"literal\n{not json}","extra":null}),
        json!({"type":"response.custom_tool_call_input.done","input":"  exact  ","output_index":"unknown stays untyped"}),
        json!({"type":"response.queued","future":{"value":false}}),
    ] {
        let WireEvent::Unknown { event_type, value } = classify_responses_payload(&raw.to_string())
        else {
            panic!("protected unknown event: {raw}");
        };
        assert_eq!(event_type, raw["type"].as_str().unwrap());
        assert_eq!(value.value(), &raw);
    }
}

#[test]
fn error_events_and_success_body_error_envelopes_keep_the_exact_provider_body() {
    for raw in [
        " {\"type\":\"error\",\"error\":{\"message\":\"failed\"},\"extra\":null} ",
        " {\"error\":{\"code\":\"bad\",\"message\":\"failure\"},\"output\":[],\"status\":\"completed\"} ",
    ] {
        assert!(
            matches!(classify_responses_payload(raw), WireEvent::Known(super::ResponsesEvent::Failure(body)) if body == raw)
        );
    }
}

#[test]
fn envelope_repair_fills_absence_and_keeps_invalid_supplied_indices_and_original_evidence() {
    let decoder =
        ResponsesDecoder::new("chatgpt", ResponsesStreamOptions::strict()).with_envelope_repair();
    let raw = r#"{"type":"response.output_text.delta","delta":"kept"}"#;
    assert!(matches!(
        decoder.classify(WireFrame::Text(raw.to_owned())),
        WireEvent::Known(_)
    ));
    let malformed = r#"{"type":"response.output_text.delta","output_index":null,"delta":"kept"}"#;
    let WireEvent::Corrupt(corrupt) = decoder.classify(WireFrame::Text(malformed.to_owned()))
    else {
        panic!("repair must not replace supplied null");
    };
    assert_eq!(corrupt.frame(), Some(malformed));
    assert!(
        corrupt
            .error()
            .to_string()
            .contains("malformed `response.output_text.delta` event")
    );
}

#[test]
fn raw_custom_calls_and_exact_namespace_remain_distinct_from_function_arguments() {
    for namespace in [None, Some(""), Some("functions")] {
        let mut custom = json!({"type":"custom_tool_call","id":"ctc_1","call_id":"call_custom",
            "name":"grammar","input":" \n{literal input}\t","status":"completed"});
        if let Some(namespace) = namespace {
            custom["namespace"] = json!(namespace);
        }
        let result = decoded(
            ResponsesStreamOptions::strict(),
            &[done(0, custom.clone()), terminal(json!([custom]))],
        );
        assert!(result.events().iter().any(|event| matches!(event, StreamEvent::Start { kind: PartKind::CustomToolCall, namespace: actual, .. } if actual.as_deref() == namespace)));
        let response = result.outcome.expect("custom call decodes");
        let [AssistantContent::CustomToolCall(call)] = response.choice.as_slice() else {
            panic!("custom call, no wrapper");
        };
        assert_eq!(call.input, " \n{literal input}\t");
        assert_eq!(call.namespace.as_deref(), namespace);
        assert_eq!(call.id.provider().unwrap().as_str(), "call_custom");
    }
}

#[test]
fn an_announced_function_keeps_present_empty_or_named_namespace_on_start_and_end() {
    for namespace in [None, Some(""), Some("functions")] {
        let item = function("completed", "{\"x\":481}", namespace);
        let added = json!({"type":"response.output_item.added","sequence_number":0,"output_index":1,"item":item});
        let result = decoded(
            ResponsesStreamOptions::strict(),
            &[
                added,
                argument_delta("{\"x\":481}"),
                done(1, item.clone()),
                terminal(json!([item])),
            ],
        );
        assert!(result.events().iter().any(|event| matches!(event, StreamEvent::Start { kind: PartKind::ToolCall, namespace: actual, .. } if actual.as_deref() == namespace)));
        let response = result.outcome.expect("function call");
        let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
            panic!("function call");
        };
        assert_eq!(call.function.namespace.as_deref(), namespace);
        assert_eq!(call.function.arguments_value(), json!({"x":481}));
    }
}

#[test]
fn carried_items_stamp_only_the_final_fold_and_keep_the_exact_done_item() {
    let item = json!({"type":"future_provider_item","id":"future_1","extra":[null,{"raw":true}]});
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[done(7, item.clone()), terminal(json!([]))],
    );
    let ended = result.ended();
    let [AssistantContent::ProviderItem(observed)] = ended.as_slice() else {
        panic!("carried end");
    };
    assert_eq!(observed.payload.provider, None);
    assert_eq!(observed.payload.item, item);
    let response = result.outcome.expect("genuine end");
    let [AssistantContent::ProviderItem(folded)] = response.choice.as_slice() else {
        panic!("folded carried item");
    };
    assert_eq!(folded.payload.provider.as_deref(), Some("openai"));
    assert_eq!(folded.payload.item, item);
    assert_eq!(response.response_id(), Some("resp_1"));
}

#[test]
fn primary_output_and_secondary_content_order_survive_late_nested_parts() {
    let item = message(json!([
        {"type":"output_text","text":"first","logprobs":[],"annotations":[{"future":true}]},
        {"type":"future_content","bytes":"unchanged","extra":null},
        {"type":"refusal","refusal":"third"}
    ]));
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[
            delta(4, 2, "third"),
            done(4, item.clone()),
            terminal(json!([])),
        ],
    );
    let starts: Vec<_> = result
        .events()
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Start { source_order, .. } => *source_order,
            _ => None,
        })
        .collect();
    assert!(starts.contains(&SourceOrder::new(4, 0)));
    assert!(starts.contains(&SourceOrder::new(4, 1)));
    assert!(starts.contains(&SourceOrder::new(4, 2)));
    let response = result.outcome.expect("ordered nested content");
    let [
        AssistantContent::CarriedText(first),
        AssistantContent::CarriedText(opaque),
        AssistantContent::CarriedText(refusal),
    ] = response.choice.as_slice()
    else {
        panic!("ordered carried parts");
    };
    assert_eq!(first.content.text(), "first");
    assert_eq!(opaque.payload.item, item["content"][1]);
    assert_eq!(refusal.content.text(), "third");
    assert_eq!(
        refusal.payload.position,
        CarriedPosition::MessageContent {
            item_index: 4,
            part_index: 2
        }
    );
}

#[test]
fn unbound_unknown_parts_remain_unknown_and_known_opaque_kind_conflicts_refuse_the_turn() {
    let opaque = json!({"type":"future_content","raw":[1,null]});
    let part = json!({"type":"response.content_part.done","sequence_number":1,"output_index":2,"content_index":0,"item_id":"unbound","part":opaque});
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[part.clone(), terminal(json!([]))],
    );
    assert!(
        result
            .items
            .iter()
            .any(|item| matches!(item, Ok(Item::Unknown(value)) if value.value() == &opaque))
    );
    assert!(
        result
            .outcome
            .expect("unknown is not fabricated content")
            .choice
            .is_empty()
    );
    let conflict = done(
        2,
        json!({"type":"message","id":"unbound","role":"assistant","status":"completed","content":[{"type":"output_text","text":"contradicts"}]}),
    );
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[part, conflict, terminal(json!([]))],
    );
    assert!(
        matches!(result.outcome, Err(ProviderError::Response(message)) if message.contains("conflicting Responses content part kind"))
    );
}

#[test]
fn incomplete_terminal_refusal_acceptance_and_unary_acceptance_are_independent() {
    let mut incomplete = response(json!([message(
        json!([{"type":"output_text","text":"partial"}])
    )]));
    incomplete["status"] = json!("incomplete");
    incomplete["incomplete_details"] = json!({"reason":"max_output_tokens"});
    let frame = json!({"type":"response.incomplete","sequence_number":2,"response":incomplete});
    let refused = decoded(ResponsesStreamOptions::strict(), &[frame.clone()]);
    assert!(refused.outcome.is_err());
    assert!(refused.ended().is_empty());
    let accepted = decoded(
        ResponsesStreamOptions::strict().with_incomplete(IncompleteTerminal::Accept),
        &[frame],
    );
    let response = accepted.outcome.expect("opted in partial terminal");
    assert_eq!(response.text(), "partial");
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(response.usage.output_tokens, Some(5));
    assert!(response.error.is_none());
    let unary = decoded(ResponsesStreamOptions::strict(), &[incomplete]);
    assert_eq!(
        unary
            .outcome
            .expect("whole reply independent of SSE default")
            .text(),
        "partial"
    );
}

#[tokio::test]
async fn malformed_known_frame_is_delivered_then_later_text_and_genuine_end_drain_successfully() {
    let malformed = json!({"type":"response.output_text.delta","sequence_number":0,"output_index":0,"content_index":0,"delta":17});
    let mut stream = streamed(
        &[
            malformed.clone(),
            delta(0, 0, "kept"),
            terminal(json!([message(
                json!([{"type":"output_text","text":"kept"}])
            )])),
            delta(0, 0, "unread"),
        ],
        IncompleteTerminal::Refuse,
    );
    let error = stream
        .next()
        .await
        .expect("corruption delivered")
        .expect_err("known wrong field type");
    let report = error.report();
    let Some(ErrorDetail::CorruptFrame(detail)) = report.detail else {
        panic!("typed corrupt frame");
    };
    assert_eq!(detail.evidence, FrameEvidence::Text(malformed.to_string()));
    assert_eq!(
        detail.event_type.as_deref(),
        Some("response.output_text.delta")
    );
    while let Some(item) = stream.next().await {
        item.expect("later valid frames");
    }
    let response = stream.finish().await.expect("drained successful terminal");
    assert_eq!(response.text(), "kept");
    assert_eq!(response.response_id(), Some("resp_1"));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(
        response.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_1")
    );
}

#[tokio::test]
async fn clean_stream_eof_keeps_delivered_partial_without_inventing_terminal_metadata() {
    let mut stream = streamed(&[delta(0, 0, "partial")], IncompleteTerminal::Refuse);
    while let Some(item) = stream.next().await {
        item.expect("clean partial EOF");
    }
    let partial = stream.partial();
    assert_eq!(partial.text(), "partial");
    assert!(partial.response_id().is_none());
    assert!(partial.finish_reason().is_none());
    assert!(partial.error.is_none());
    assert!(partial.message_id.is_none());
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[test]
fn malformed_authoritative_calls_drop_only_when_incomplete_and_keep_exact_typed_refusals_otherwise()
{
    for prefix in [None, Some("{\"x\":481"), Some("{}"), Some("   ")] {
        for status in ["incomplete", "completed", "in_progress"] {
            let item = function(status, "{\"x\":481", None);
            let mut frames = Vec::new();
            if let Some(prefix) = prefix {
                frames.push(argument_delta(prefix));
            }
            frames.push(done(1, item));
            frames.push(terminal(json!([])));
            let result = decoded(ResponsesStreamOptions::strict(), &frames);
            assert!(
                result
                    .ended()
                    .iter()
                    .all(|content| !matches!(content, AssistantContent::ToolCall(_)))
            );
            if status == "incomplete" {
                let items: Vec<_> = result
                    .items
                    .iter()
                    .map(|item| item.clone().expect("drop is not an error"))
                    .collect();
                Transcript::parse(serde_json::to_value(&items).expect("transcript JSON"))
                    .expect("discard closes the same Part");
                assert!(
                    result
                        .outcome
                        .expect("genuine end after drop")
                        .choice
                        .is_empty()
                );
                assert!(items.iter().any(|item| matches!(
                    item,
                    Item::Event(StreamEvent::Discard {
                        reason: DiscardReason::Incomplete,
                        ..
                    })
                )));
            } else {
                let error = result
                    .outcome
                    .expect_err("asserted malformed input is refused");
                let report = error.report();
                assert!(
                    report
                        .message
                        .starts_with("tool call `add` arrived with malformed JSON input")
                );
                let Some(ErrorDetail::MalformedToolInput(detail)) = report.detail else {
                    panic!("typed malformed input");
                };
                assert_eq!(detail.raw, "{\"x\":481");
                assert_eq!(detail.provider.as_ref().unwrap().call_id.as_str(), "call_1");
                assert_eq!(
                    detail.provider.as_ref().unwrap().item_id.as_deref(),
                    Some("fc_1")
                );
                assert_eq!(detail.id.provider().unwrap().as_str(), "call_1");
            }
        }
    }
}

#[test]
fn dropped_ordered_call_leaves_no_placeholder_and_closes_its_original_part() {
    let provider =
        json!({"type":"web_search_call","id":"ws_1","status":"completed","action":{"query":"rig"}});
    let incomplete = function("incomplete", "{", None);
    let added = json!({"type":"response.output_item.added","output_index":1,"sequence_number":1,"item":incomplete});
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[
            added,
            argument_delta("{"),
            done(0, provider.clone()),
            done(1, incomplete.clone()),
            terminal(json!([provider.clone(), incomplete])),
        ],
    );
    let items: Vec<_> = result
        .items
        .iter()
        .map(|item| item.clone().expect("no error"))
        .collect();
    let transcript = Transcript::parse(serde_json::to_value(&items).expect("transcript"))
        .expect("closed discard transcript");
    let start = transcript
        .events()
        .find_map(|event| match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                ..
            } => Some(*part),
            _ => None,
        })
        .expect("call actually started");
    assert!(transcript.events().any(|event| matches!(event, StreamEvent::Discard { part, reason: DiscardReason::Incomplete } if *part == start)));
    let response = result.outcome.expect("genuine terminal");
    let [AssistantContent::ProviderItem(item)] = response.choice.as_slice() else {
        panic!("one provider item; no dropped placeholder");
    };
    assert_eq!(item.payload.item, provider);
}

#[test]
fn completed_calls_flush_before_a_failed_provider_terminal_without_terminal_metadata() {
    let item = function("completed", "{}", Some(""));
    let mut failed = response(json!([]));
    failed["status"] = json!("failed");
    failed["error"] = json!({"code":"failure","message":"exact failure"});
    let frame = json!({"type":"response.failed","sequence_number":3,"response":failed});
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[done(1, item), frame.clone()],
    );
    assert!(matches!(
        result.ended().as_slice(),
        [AssistantContent::ToolCall(_)]
    ));
    let error = result.outcome.expect_err("provider failure");
    assert_eq!(
        error.provider_response_body(),
        Some(frame.to_string().as_str())
    );
}

#[test]
fn completed_reasoning_cipher_is_authoritative_and_terminal_fills_only_missing_fields() {
    let first = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"summary"}],
        "content":[{"type":"future_reasoning","payload":[null,7]},{"type":"reasoning_text","text":"content"}],
        "encrypted_content":"first-cipher","signature":"first-signature","status":"completed"});
    let mut last = first.clone();
    last["encrypted_content"] = json!("later-cipher");
    last["signature"] = json!("later-signature");
    let result = decoded(
        ResponsesStreamOptions::strict(),
        &[done(0, first.clone()), terminal(json!([last]))],
    );
    let response = result.outcome.expect("reasoning restatement");
    let [AssistantContent::CarriedReasoning(reasoning)] = response.choice.as_slice() else {
        panic!("carried reasoning");
    };
    assert_eq!(reasoning.payload.item["encrypted_content"], "first-cipher");
    assert_eq!(reasoning.payload.item["signature"], "first-signature");
    assert_eq!(reasoning.payload.item["content"][0], first["content"][0]);
    assert_eq!(reasoning.payload.provider.as_deref(), Some("openai"));
    assert_eq!(reasoning.display_text(), "summary\ncontent");
    assert_eq!(response.text(), "");
}

#[test]
fn malformed_never_done_arguments_drop_only_at_a_genuine_terminal() {
    for status in ["completed", "incomplete"] {
        let added = json!({"type":"response.output_item.added","sequence_number":0,
            "output_index":1,"item":function("in_progress", "", Some("functions"))});
        let mut end = terminal(json!([]));
        end["type"] = json!(format!("response.{status}"));
        end["response"]["status"] = json!(status);
        if status == "incomplete" {
            end["response"]["incomplete_details"] = json!({"reason":"max_output_tokens"});
        }
        let result = decoded(
            ResponsesStreamOptions::strict().with_incomplete(IncompleteTerminal::Accept),
            &[added, argument_delta("{\"city\":\"Paris\""), end],
        );
        let items: Vec<_> = result.items.iter()
            .map(|item| item.clone().expect("unasserted malformed fragments drop without error"))
            .collect();
        let transcript = Transcript::parse(serde_json::to_value(&items).expect("transcript"))
            .expect("the started Part closes by discard");
        let part = transcript.events().find_map(|event| match event {
            StreamEvent::Start { part, kind: PartKind::ToolCall, .. } => Some(*part),
            _ => None,
        }).expect("actual fragment assembly started");
        assert!(transcript.events().any(|event| matches!(event,
            StreamEvent::Discard { part: discarded, reason: DiscardReason::UnfinishedAtTerminal }
                if *discarded == part)));
        assert!(result.ended().is_empty());
        let response = result.outcome.expect("genuine terminal after drop");
        assert!(response.choice.is_empty());
        assert_eq!(response.response_id(), Some("resp_1"));
    }
}

/// F accepts every successfully parsed JSON value at a never-done terminal.
/// Replay must retain scalar/array/string/null rather than replace it with {}.
#[tokio::test]
async fn valid_never_done_json_values_replay_exactly_without_discard() {
    for (raw, expected) in [
        ("{\"city\":\"Paris\"}", "{\"city\":\"Paris\"}"),
        ("481", "481"),
        ("[481,null]", "[481,null]"),
        ("\"quoted\"", "\"quoted\""),
        ("\"{\\\"city\\\":\\\"Paris\\\"}\"", "\"{\\\"city\\\":\\\"Paris\\\"}\""),
        ("null", "null"),
        (" \t null \n", "null"),
        ("   ", "{}"),
    ] {
        let added = json!({"type":"response.output_item.added","sequence_number":0,
            "output_index":1,"item":function("in_progress", "", Some(""))});
        let mut stream = streamed(&[added, argument_delta(raw), terminal(json!([]))],
            IncompleteTerminal::Refuse);
        while let Some(item) = stream.next().await {
            let item = item.expect("valid fragments complete at terminal");
            assert!(!matches!(item, Item::Event(StreamEvent::Discard { .. })));
        }
        let response = stream.finish().await.expect("actual Model's genuine terminal");
        let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
            panic!("one valid function call for {raw}");
        };
        assert_eq!(call.id.provider().unwrap().as_str(), "call_1");
        assert_eq!(call.function.name.as_str(), "add");
        assert_eq!(call.function.namespace.as_deref(), Some(""));
        let wire = OpenAIConfig::new("fixture-key").responses("different-model");
        let original_request = CompletionRequest::new("continue")
            .message(response.message().expect("canonical call history"));
        let original = Completion::prepare(original_request.clone(), &wire.describe())
            .expect("original no-tools history adaptation");
        let original = wire.encode(original, Mode::Unary)
            .expect("original no-tools request");
        let original_body = crate::test_utils::json_body(&original.request);
        let original_input = original_body["input"].as_array().expect("original input");
        assert!(original_input.iter().all(|item| item["type"] != "function_call"));
        let original_assistant = original_input.iter()
            .find(|item| item["role"] == "assistant").expect("original textual call");
        let expected_no_tools = if raw == "{\"city\":\"Paris\"}" {
            r#"[called tool add with {"city":"Paris"}]"#
        } else {
            "[called tool add with {}]"
        };
        assert_eq!(original_assistant["content"][0]["text"], expected_no_tools);

        // The replay control explicitly declares its tool; no-tools stays textual above.
        let request = original_request.tool(crate::completion::ToolDefinition::new(
            crate::message::ToolName::new("add").expect("declared call name"),
            "add",
            json!({}),
        ));
        let request = Completion::prepare(request, &wire.describe()).expect("history adaptation");
        let encoded = wire.encode(request, Mode::Unary).expect("actual Responses encoder");
        let body = crate::test_utils::json_body(&encoded.request);
        let call = body["input"].as_array().expect("input").iter()
            .find(|item| item["type"] == "function_call").expect("replayed call");
        assert_eq!(call["arguments"], expected, "exact F JSON value for {raw}");
        assert_eq!(call["namespace"], "");
    }
}

#[tokio::test]
async fn fragment_only_clean_eof_leaves_call_open_without_end_discard_or_response() {
    let added = json!({"type":"response.output_item.added","sequence_number":0,
        "output_index":1,"item":function("in_progress", "", None)});
    let mut stream = streamed(&[added, argument_delta("{\"city\":\"Paris\"")],
        IncompleteTerminal::Refuse);
    let mut started = false;
    while let Some(item) = stream.next().await {
        let item = item.expect("clean EOF is not a provider failure");
        match item {
            Item::Event(StreamEvent::Start { kind: PartKind::ToolCall, .. }) => started = true,
            Item::Event(StreamEvent::End { .. } | StreamEvent::Discard { .. }) => {
                panic!("EOF cannot close an unasserted call")
            }
            _ => {}
        }
    }
    assert!(started);
    let partial = stream.partial();
    assert!(partial.choice.is_empty(), "F drops open tool inputs from canonical choice");
    assert!(partial.response_id().is_none());
    assert!(partial.finish_reason().is_none());
    assert!(partial.message_id.is_none());
    assert!(partial.error.is_none());
    assert!(matches!(stream.finish().await, Err(ProviderError::Truncated)));
}

#[test]
fn reasoning_string_content_and_typed_text_restatements_share_the_same_kind() {
    let first = json!({"type":"reasoning","id":"rs_1","summary":[],
        "content":["kept"],"status":"completed"});
    let last = json!({"type":"reasoning","id":"rs_1","summary":[],
        "content":[{"type":"reasoning_text","text":"kept"}],"status":"completed"});
    let result = decoded(ResponsesStreamOptions::strict(),
        &[done(0, first), terminal(json!([last]))]);
    let response = result.outcome.expect("both F known-text forms are valid");
    let [AssistantContent::CarriedReasoning(reasoning)] = response.choice.as_slice() else {
        panic!("carried reasoning content");
    };
    assert_eq!(reasoning.display_text(), "kept");
    assert_eq!(reasoning.payload.provider.as_deref(), Some("openai"));
}

#[test]
fn reasoning_part_and_done_restatements_do_not_emit_live_reasoning_deltas() {
    for observed in [
        json!({"type":"response.content_part.added","sequence_number":1,"output_index":0,
            "content_index":0,"item_id":"rs_1","part":{"type":"reasoning_text","text":"kept"}}),
        json!({"type":"response.reasoning_text.done","sequence_number":1,"output_index":0,
            "content_index":0,"item_id":"rs_1","text":"kept"}),
        json!({"type":"response.reasoning_summary_text.done","sequence_number":1,"output_index":0,
            "summary_index":0,"item_id":"rs_1","text":"kept"}),
    ] {
        let result = decoded(ResponsesStreamOptions::strict(), &[observed]);
        assert!(result.events().iter().all(|event|
            !matches!(event, StreamEvent::Reasoning { .. })));
        assert!(result.ended().is_empty(), "unasserted known snapshots do not complete a block");
        assert!(matches!(result.outcome, Err(ProviderError::Truncated)));
    }
}

#[tokio::test]
async fn malformed_known_terminal_cannot_take_raw_custody_before_a_genuine_terminal() {
    let mut malformed = terminal(json!([]));
    malformed["response"]["usage"]["output_tokens"] = json!("not a counter");
    let genuine = terminal(json!([message(json!([{"type":"output_text","text":"kept"}]))]));
    let mut stream = streamed(&[malformed.clone(), delta(0, 0, "kept"), genuine.clone()],
        IncompleteTerminal::Refuse);
    let error = stream.next().await.expect("corruption delivered").expect_err("typed malformed terminal");
    let Some(ErrorDetail::CorruptFrame(detail)) = error.report().detail else {
        panic!("typed corruption detail");
    };
    assert_eq!(detail.evidence, FrameEvidence::Text(malformed.to_string()));
    while let Some(item) = stream.next().await {
        item.expect("later genuine terminal");
    }
    let response = stream.finish().await.expect("delivered corruption was drained");
    assert_eq!(response.raw, genuine["response"]);
    assert_eq!(response.text(), "kept");
    assert_eq!(response.usage.output_tokens, Some(5));
}

#[test]
fn strict_document_repair_matches_decoder_and_default_document_stays_lenient() {
    use crate::wire::document::Reassemble;
    let mut envelope_less = terminal(json!([]));
    envelope_less.as_object_mut().unwrap().remove("sequence_number");
    let frame = WireFrame::Text(envelope_less.to_string());
    let mut strict = super::document::Response::with_strict_reader(false);
    strict.absorb(&frame);
    assert_eq!(strict.finish(), Value::Null);
    let mut repaired = super::document::Response::with_strict_reader(true);
    repaired.absorb(&frame);
    assert_eq!(repaired.finish(), envelope_less["response"]);
    let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
        .with_envelope_repair();
    assert!(matches!(decoder.classify(frame), WireEvent::Known(_)));

    // U's existing sparse raw-document fixtures continue to use Default.
    let sparse = json!({"type":"response.completed","response":{"id":"resp_sparse","output":[]}});
    let mut default = super::document::Response::default();
    default.absorb(&WireFrame::Text(sparse.to_string()));
    let mut expected = sparse["response"].clone();
    expected["status"] = json!("completed");
    assert_eq!(default.finish(), expected);
}

#[tokio::test]
async fn partial_eof_stamps_only_folded_carried_issuer_and_keeps_delivered_ends_unknown() {
    let item = json!({"type":"future_item","id":"future_1","payload":[null,7]});
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[{"type":"future_summary","raw":true}],
        "content":[{"type":"reasoning_text","text":"kept"}],"status":"completed"});
    let mut stream = streamed(&[done(0, item.clone()), done(1, reasoning)], IncompleteTerminal::Refuse);
    let mut ended_provider = false;
    let mut ended_reasoning = false;
    while let Some(item) = stream.next().await {
        match item.expect("clean partial EOF") {
            Item::Event(StreamEvent::End { content: AssistantContent::ProviderItem(observed), .. }) => {
                assert_eq!(observed.payload.provider, None);
                ended_provider = true;
            }
            Item::Event(StreamEvent::End { content: AssistantContent::CarriedReasoning(observed), .. }) => {
                assert_eq!(observed.payload.provider, None);
                ended_reasoning = true;
            }
            _ => {}
        }
    }
    assert!(ended_provider && ended_reasoning);
    let partial = stream.partial();
    let [AssistantContent::ProviderItem(folded), AssistantContent::CarriedReasoning(reasoning)] = partial.choice.as_slice() else {
        panic!("provider item then reasoning survive partial EOF");
    };
    assert_eq!(folded.payload.item, item);
    assert_eq!(folded.payload.provider.as_deref(), Some("openai"));
    assert_eq!(reasoning.payload.provider.as_deref(), Some("openai"));
    let crate::message::CarriedReasoningPart::OpaqueSummary(opaque) = &reasoning.content[0] else {
        panic!("unknown summary is retained");
    };
    assert_eq!(opaque.provider.as_deref(), Some("openai"));
    assert!(partial.response_id().is_none());
    assert!(partial.finish_reason().is_none());
    assert!(partial.error.is_none());
    assert!(matches!(stream.finish().await, Err(ProviderError::Truncated)));
}

#[test]
fn repaired_lifecycle_preserves_every_genuine_sibling_in_the_raw_document() {
    use crate::wire::document::Reassemble;
    let frame = json!({"type":"response.completed","response":response(json!([])),
        "output_index":17,"content_index":23,"summary_index":29,
        "future":{"keep":[null,7]}});
    let mut repaired = super::document::Response::with_strict_reader(true);
    repaired.absorb(&WireFrame::Text(frame.to_string()));
    let mut expected = frame["response"].clone();
    for field in ["output_index", "content_index", "summary_index", "future"] {
        expected[field] = frame[field].clone();
    }
    assert_eq!(repaired.finish(), expected);
}

#[test]
fn repaired_known_events_synthesize_only_their_required_coordinates_and_keep_raw() {
    let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
        .with_envelope_repair();
    for (received, fields) in [
        (json!({"type":"response.completed","response":response(json!([]))}),
            vec!["sequence_number"]),
        (json!({"type":"response.output_item.done","item":message(json!([]))}),
            vec!["sequence_number", "output_index"]),
        (json!({"type":"response.output_text.delta","delta":"kept"}),
            vec!["sequence_number", "output_index", "content_index"]),
        (json!({"type":"response.reasoning_summary_text.delta","delta":"thought"}),
            vec!["sequence_number", "output_index", "summary_index"]),
        (json!({"type":"response.reasoning_text.delta","delta":"thought"}),
            vec!["sequence_number", "output_index"]),
    ] {
        let raw = received.to_string();
        let mut expected = received;
        for field in fields {
            expected[field] = json!(0);
        }
        let WireEvent::Known(super::ResponsesEvent::Frame { frame, raw: captured, .. }) =
            decoder.classify(WireFrame::Text(raw.clone())) else {
                panic!("only missing required bookkeeping is repaired");
            };
        assert_eq!(frame, expected);
        assert_eq!(captured, raw);
    }
}
