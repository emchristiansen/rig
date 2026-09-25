use super::{
    ContentPartChunkPart, ItemChunk, ItemChunkKind, RawChoiceAccumulator, ResponsesDecoder,
    ResponsesStreamOptions, StreamingCompletionChunk, classify_responses_frame,
    reasoning_from_done_item,
};
use crate::completion::CompletionModel;
use crate::driver::{Bound, WireDriver};
use crate::error::ProviderError;
use crate::error::{ErrorKind, ErrorReport};
use crate::message::{AssistantContent, ReasoningContent};
use crate::operation::AdapterOutput;
use crate::operation::Completion;
use crate::providers::internal::openai_chat_completions_compatible::test_support::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::internal::wire::WireEvent;
use crate::providers::openai::OpenAI;
use crate::providers::openai::responses_api::{
    AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
    ReasoningSummary, ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
};
use crate::streaming::{BlockClose, BlockId, BlockKind, Delta, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::WireFrame;
use crate::wire::{Fold, Operation, Reply};
use futures::StreamExt;
use serde_json::{self, json};

#[test]
fn known_event_cannot_fall_back_to_a_whole_response() {
    let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
    let mut body = json!({
        "id":"resp_1", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-test"
    });
    assert!(matches!(
        decoder.classify_payload(&body.to_string()),
        WireEvent::Known(super::ResponsesEvent::Whole(_))
    ));
    body["type"] = json!("response.completed");
    assert!(matches!(
        decoder.classify_payload(&body.to_string()),
        WireEvent::Corrupt(_)
    ));
    // Even a duplicate discriminator must not make body decoding rescue the
    // event's invalid envelope.
    let duplicate = format!("{{\"type\":\"future.event\",{}", &body.to_string()[1..]);
    assert!(matches!(
        decoder.classify_payload(&duplicate),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn classify_known_event_decodes() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "hi",
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Known(StreamingCompletionChunk::Delta(_))
    ));
}

#[test]
fn classify_unknown_event_type_is_unknown() {
    let frame = json!({
        "type": "response.web_search_call.searching",
        "output_index": 0,
        "sequence_number": 1,
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Unknown { event_type, .. } if event_type == "response.web_search_call.searching"
    ));
}

/// #2258 G4: `response.reasoning_text.done` terminates every raw-reasoning
/// block on all three Responses surfaces. It used to be absent from the
/// known-event set, so each block logged a spurious "unknown event" warn
/// and passed through as `Unknown`.
///
/// Both halves of the fix are asserted here, because either alone is a
/// regression: the tag must be KNOWN (no `Unknown`), and `ItemChunkKind`
/// must carry a variant for it (no `Corrupt`, which is what naming the tag
/// without the variant would have produced — strictly worse than the warn).
///
/// No recorded cassette contains this event; the wire shape is the
/// Responses spec's, so this unit test is the pin.
#[test]
fn classify_reasoning_text_done_is_known_and_decodes() {
    let frame = json!({
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 7,
        "text": "the model's raw chain of thought",
    })
    .to_string();

    let event = classify_responses_frame(&frame);
    assert!(
        !matches!(event, WireEvent::Unknown { .. }),
        "the tag must be in the known-event set: {event:?}"
    );
    assert!(
        !matches!(event, WireEvent::Corrupt(_)),
        "a known tag with no matching ItemChunkKind variant decodes to Corrupt, which the \
             driver surfaces as an in-band Err — worse than the warn it replaced: {event:?}"
    );
    assert!(matches!(
        event,
        WireEvent::Known(StreamingCompletionChunk::Delta(chunk))
            if matches!(chunk.data, ItemChunkKind::ReasoningTextDone(_))
    ));
}

/// The done event restates text the deltas already streamed, so it must be
/// a no-op: replaying it would double every raw-reasoning block.
#[test]
fn reasoning_text_done_emits_nothing() {
    let mut accumulator = RawChoiceAccumulator::new("openai", None);
    let chunk: ItemChunk = serde_json::from_value(json!({
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 7,
        "text": "the model's raw chain of thought",
    }))
    .expect("reasoning text done event should deserialize");

    let mut emitted = AdapterOutput::new();
    accumulator.decode_item_chunk(chunk, ResponsesStreamOptions::strict(), &mut emitted);
    assert!(
        emitted.is_empty(),
        "the done restatement must not re-emit the reasoning text: {emitted:?}"
    );
}

#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(
        classify_responses_frame("{not json"),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn classify_known_event_with_defective_payload_is_corrupt() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": 42,
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

// The P2 probe shape from `rig-2257-code-review-findings-34ee8ba5.md`: a
// known part tag whose payload is schema-defective must classify as
// `Corrupt`, not slide into the unknown-part catch-all.
#[test]
fn classify_defective_known_content_part_is_corrupt() {
    let frame = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": {"type": "output_text", "text": 42},
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn content_part_known_tag_decodes() {
    let part: ContentPartChunkPart =
        serde_json::from_value(json!({"type": "output_text", "text": "hi"})).unwrap();
    assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
}

#[test]
fn content_part_known_tag_with_defective_payload_errors() {
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": "output_text", "text": 42}));
    assert!(result.is_err());
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": "summary_text", "text": 42}));
    assert!(result.is_err());
}

// A non-string `type` is a data-level defect of the tagged shape, never a
// skippable unknown part (#2258 F8).
#[test]
fn content_part_non_string_type_errors() {
    let result = serde_json::from_value::<ContentPartChunkPart>(json!({"type": 42, "text": "hi"}));
    assert!(result.is_err());
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": null, "text": "hi"}));
    assert!(result.is_err());
}

// Pins the documented duplicate-key edge (#2258 F8): `serde_json::Value`
// keeps the last duplicate key, so the hand dispatch resolves on the LAST
// `type` — unlike a derived internally-tagged enum, which takes the first.
#[test]
fn content_part_duplicate_type_key_dispatches_on_the_last_occurrence() {
    let part: ContentPartChunkPart =
        serde_json::from_str(r#"{"type":"bogus","type":"output_text","text":"hi"}"#).unwrap();
    assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
}

// `refusal` and `reasoning_text` part tags are not in the modeled set:
// they must stay skippable no-ops (the content arrives via the
// corresponding delta events), round-tripping the value verbatim.
#[test]
fn content_part_unknown_tag_is_preserved_verbatim() {
    let wire = json!({"type": "future_refusal", "refusal": "no"});
    let part: ContentPartChunkPart = serde_json::from_value(wire.clone()).unwrap();
    let ContentPartChunkPart::Unknown(value) = &part else {
        panic!("unmodeled part tag must fall back to Unknown");
    };
    assert_eq!(value, &wire);
    assert_eq!(serde_json::to_value(&part).unwrap(), wire);
}

fn sample_response(status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: "resp_123".to_string(),
        object: ResponseObject::Response,
        provider_request_id: None,
        created_at: 0,
        status,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: "gpt-5.4".to_string(),
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
        usage: None,
        output: Vec::new(),
        tools: Vec::new(),
        additional_parameters: AdditionalParameters::default(),
    }
}

/// The OpenAI Responses stream one scripted transport yields: the live
/// loop, over a socket that answers from a script.
async fn responses_stream<H: crate::driver::Socket>(
    http: H,
) -> crate::streaming::StreamingCompletionResponse {
    let model = Bound::new(OpenAI::new("test-key").responses("gpt-5.4"), http);
    let request = model.completion_request("hello").build();
    model.stream(request).await.expect("stream should start")
}

/// The same, for a body scripted as JSON events.
async fn responses_stream_of(
    events: &[serde_json::Value],
) -> crate::streaming::StreamingCompletionResponse {
    responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(events),
    })
    .await
}

/// The events one buffered Responses SSE body decodes to.
///
/// The buffered twin of a stream: the SAME decoder the live loop runs,
/// driven by the SAME [`WireDriver`], without a socket. There is no stream
/// to carry `Err` items here, so the first data error fails the whole
/// decode rather than returning a silently partial completion.
fn stream_events_from_sse_body(
    provider: &str,
    body: &str,
    initial_usage: Option<ResponsesUsage>,
) -> Result<Vec<StreamEvent>, ProviderError> {
    let mut driver: WireDriver<Completion, _> = WireDriver::new(
        ResponsesDecoder::new(provider, ResponsesStreamOptions::strict())
            .with_envelope_repair()
            .with_initial_usage(initial_usage),
    );
    let mut events = Vec::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        driver.push(WireFrame::Text(data.to_owned()));
        for item in driver.drain() {
            events.push(item?);
        }
    }
    driver.finish();
    for item in driver.drain() {
        events.push(item?);
    }
    Ok(events)
}

/// The response a decoded event sequence folds to: the operation's own
/// fold — the one [`crate::driver::stream`] drains into — closed with the
/// reply document the turn came from.
fn folded_stream_events(
    provider: &str,
    events: Vec<StreamEvent>,
    raw_response: &CompletionResponse,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let reply = Reply {
        provider: provider.to_owned(),
        raw: serde_json::to_value(raw_response)?,
        provider_request_id: raw_response.provider_request_id.clone(),
        response_headers: Default::default(),
    };
    let mut fold = <Completion as Operation>::Fold::default();
    for event in events {
        fold.absorb(event)?;
    }
    fold.finish(reply)
}

async fn first_error_from_event(event: serde_json::Value) -> ErrorReport {
    let mut stream = responses_stream_of(&[event]).await;

    stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider error")
}

/// The provider-native terminal record, recovered from the serialized
/// `StreamFinal::raw` the stream's terminal carries.
async fn final_response_from_event(event: serde_json::Value) -> super::StreamingCompletionResponse {
    let mut stream = responses_stream_of(&[event]).await;

    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(response) = item.expect("completed stream should not error") {
            return serde_json::from_value(response.raw)
                .expect("the raw terminal record is the provider's own type");
        }
    }

    panic!("stream should yield a final response");
}

/// The normalized terminal record, as `stream` exposes it.
async fn stream_final_from_event(event: serde_json::Value) -> crate::streaming::StreamFinal {
    let mut stream = responses_stream_of(&[event]).await;

    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(response) = item.expect("completed stream should not error") {
            return response;
        }
    }

    panic!("stream should yield a final response");
}

/// Drain a stream whose provider fully delivered one tool call before a
/// terminal error: the call's block events (its start, then the end
/// carrying the completed call) come first, then the error, then nothing.
async fn flushed_tool_call_then_error(
    stream: &mut crate::streaming::StreamingCompletionResponse,
) -> (crate::message::ToolCall, ErrorReport) {
    let mut tool_call = None;
    let err = loop {
        match stream
            .next()
            .await
            .expect("stream should yield the flushed tool call, then the error")
        {
            Ok(StreamEvent::BlockStart {
                kind: BlockKind::ToolCall,
                ..
            }) => {}
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(call)),
                ..
            }) => tool_call = Some(call),
            Ok(other) => panic!("expected the flushed tool call first, got {other:?}"),
            Err(err) => break err,
        }
    };
    let tool_call = tool_call.expect("the flushed tool call must precede the terminal error");
    assert!(
        stream.next().await.is_none(),
        "nothing may follow the terminal error"
    );
    (tool_call, err)
}

#[test]
fn a_buffered_body_preserves_its_error_payloads() {
    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: Some("server_error".to_string()),
        message: "response failed".to_string(),
    });
    let events = [
        json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        }),
        json!({
            "type": "error",
            "error": {
                "message": "boom",
                "code": "server_error",
                "type": "server_error"
            }
        }),
    ];

    for event in events {
        let payload = serde_json::to_string(&event).expect("event should serialize");
        let body = format!("data: {payload}\n");
        let err = stream_events_from_sse_body("ChatGPT", &body, None)
            .expect_err("error payload should surface as provider response");

        assert!(matches!(
            err,
            crate::error::ProviderError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(payload.as_str()));
    }
}

#[test]
fn reasoning_done_item_fuses_summary_content_and_encrypted_into_one_end() {
    let summary = vec![
        ReasoningSummary::SummaryText {
            text: "step 1".to_string(),
        },
        ReasoningSummary::SummaryText {
            text: "step 2".to_string(),
        },
    ];
    let content = vec!["private reasoning".to_string()];
    let reasoning = reasoning_from_done_item(
        Some("rs_1"),
        summary,
        content.into_iter().map(Into::into).collect(),
        Some("enc_blob".to_string()),
        None,
    );

    // ONE restatement carrying every block in wire field order — never a
    // block per entry, which made siblings under one `rs_*` id.
    let Some(reasoning) = reasoning else {
        panic!("expected one wire-sent reasoning restatement");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.content,
        vec![
            ReasoningContent::Summary("step 1".to_string()),
            ReasoningContent::Summary("step 2".to_string()),
            ReasoningContent::Text {
                text: "private reasoning".to_string(),
                signature: None,
            },
            ReasoningContent::Encrypted("enc_blob".to_string()),
        ]
    );
}

#[test]
fn reasoning_output_item_done_emits_reasoning_text_content() {
    let body = format!(
        "data: {}\n",
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_text_1",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "visible reasoning" }],
                "status": "completed"
            },
        })
    );

    let events =
        stream_events_from_sse_body("openai", &body, None).expect("sse body should decode");

    // The done item opens its block under the item's `rs_*` id and closes
    // it with one wire-sent end restatement whose single block is the
    // reasoning text.
    assert!(matches!(
        events.first(),
        Some(StreamEvent::BlockStart {
            id,
            kind: BlockKind::Reasoning {
                provider_id: Some(provider_id)
            },
        }) if id == &BlockId::wire("rs_text_1") && provider_id == "rs_text_1"
    ));
    assert!(matches!(
        events.get(1),
        Some(StreamEvent::BlockEnd {
            id,
            end: BlockClose::Reasoning {
                reasoning: Some(reasoning),
                wire_sent: true,
                ..
            },
            ..
        }) if id == &BlockId::wire("rs_text_1")
            && reasoning.content
                == vec![ReasoningContent::Text {
                    text: "visible reasoning".to_string(),
                    signature: None,
                }]
    ));
}

/// Envelope-less replay shape (ChatGPT bodies): an id-less summary
/// delta mints an Output-kind key, the done item restates the whole
/// block under the SAME adopted minted key, and visible text follows.
/// The driver's boundary law must treat the same-key whole block as a
/// close — this exact body used to abort every debug build
/// (sequence-law O1, Responses variant).
#[test]
fn envelope_less_reasoning_then_text_decodes_without_violation() {
    let body = format!(
        "data: {}\ndata: {}\ndata: {}\n",
        json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "reasoning",
                "id": "",
                "summary": [{ "type": "summary_text", "text": "thinking, complete" }],
                "status": "completed"
            },
        }),
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "the answer",
        }),
    );

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("sse body should decode without a sequence-law violation");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "the answer"
    )));
}

#[test]
fn reasoning_text_delta_emits_reasoning_delta() {
    let body = format!(
        "data: {}\n",
        json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_delta_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        })
    );

    let events =
        stream_events_from_sse_body("openai", &body, None).expect("sse body should decode");

    // The first delta for an unseen id opens its block, carrying the
    // wire's `rs_*` id as the durable provider id.
    assert!(matches!(
        events.first(),
        Some(StreamEvent::BlockStart {
            id,
            kind: BlockKind::Reasoning {
                provider_id: Some(provider_id)
            },
        }) if id == &BlockId::wire("rs_delta_1") && provider_id == "rs_delta_1"
    ));
    assert!(matches!(
        events.get(1),
        Some(StreamEvent::BlockDelta { id, delta: Delta::Reasoning { text } })
            if id == &BlockId::wire("rs_delta_1") && text == "thinking"
    ));
}

#[test]
fn unknown_output_item_surfaces_as_raw_unknown_choice() {
    // A hosted-tool item (web_search_call) arriving on
    // `response.output_item.done` must surface to stream consumers as
    // `StreamEvent::Unknown` carrying the verbatim item, mirroring how
    // the non-streaming decode preserves it on `CompletionResponse.output`.
    let item = json!({
        "type": "web_search_call",
        "id": "ws_001",
        "status": "completed",
        "action": { "type": "search", "queries": ["rig framework"] },
    });
    let body = format!(
        "data: {}\n",
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": item,
        })
    );

    let events =
        stream_events_from_sse_body("openai", &body, None).expect("sse body should decode");

    let unknown = events.iter().find_map(|event| match event {
        StreamEvent::Unknown(value) => Some(value),
        _ => None,
    });
    assert_eq!(
        unknown,
        Some(&item.into()),
        "the raw web_search_call item should reach the consumer verbatim",
    );
}

#[test]
fn reasoning_done_item_without_encrypted_emits_summary_only() {
    let summary = vec![ReasoningSummary::SummaryText {
        text: "only summary".to_string(),
    }];
    let reasoning = reasoning_from_done_item(Some("rs_2"), summary, Vec::new(), None, None);

    let Some(reasoning) = reasoning else {
        panic!("expected one reasoning restatement");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_2"));
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Summary("only summary".to_string())]
    );
}

#[test]
fn empty_encrypted_reasoning_is_not_emitted() {
    let content = vec!["visible reasoning".to_string()];

    let reasoning = reasoning_from_done_item(
        Some("rs_1"),
        Vec::new(),
        content.into_iter().map(Into::into).collect(),
        Some(String::new()),
        None,
    );

    let Some(reasoning) = reasoning else {
        panic!("expected one reasoning restatement");
    };
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Text {
            text: "visible reasoning".to_string(),
            signature: None,
        }],
        "an empty encrypted payload contributes no block"
    );

    // An entirely empty done item says nothing at the boundary.
    assert!(
        reasoning_from_done_item(
            Some("rs_1"),
            Vec::new(),
            Vec::new(),
            Some(String::new()),
            None
        )
        .is_none()
    );
}

#[test]
fn content_part_added_deserializes_snake_case_part_type() {
    let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 3,
        "part": {
            "type": "output_text",
            "text": "hello"
        }
    }))
    .expect("content part event should deserialize");

    assert!(matches!(
        chunk,
        StreamingCompletionChunk::Delta(chunk)
            if matches!(
                chunk.data,
                ItemChunkKind::ContentPartAdded(_)
            )
    ));
}

#[test]
fn content_part_done_deserializes_snake_case_part_type() {
    let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 4,
        "part": {
            "type": "summary_text",
            "text": "done"
        }
    }))
    .expect("content part done event should deserialize");

    assert!(matches!(
        chunk,
        StreamingCompletionChunk::Delta(chunk)
            if matches!(
                chunk.data,
                ItemChunkKind::ContentPartDone(_)
            )
    ));
}

#[test]
fn reasoning_summary_part_added_deserializes_snake_case_part_type() {
    let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
        "type": "response.reasoning_summary_part.added",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 5,
        "part": {
            "type": "summary_text",
            "text": "step 1"
        }
    }))
    .expect("reasoning summary part event should deserialize");

    assert!(matches!(
        chunk,
        StreamingCompletionChunk::Delta(chunk)
            if matches!(
                chunk.data,
                ItemChunkKind::ReasoningSummaryPartAdded(_)
            )
    ));
}

#[test]
fn reasoning_summary_part_done_deserializes_snake_case_part_type() {
    let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
        "type": "response.reasoning_summary_part.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 6,
        "part": {
            "type": "summary_text",
            "text": "step 2"
        }
    }))
    .expect("reasoning summary part done event should deserialize");

    assert!(matches!(
        chunk,
        StreamingCompletionChunk::Delta(chunk)
            if matches!(
                chunk.data,
                ItemChunkKind::ReasoningSummaryPartDone(_)
            )
    ));
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: Some(String::new()),
        message: "maximum context length exceeded".to_string(),
    });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("maximum context length exceeded")
    }));
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: Some("context_length_exceeded".to_string()),
        message: "maximum context length exceeded".to_string(),
    });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed")
            && body.contains("context_length_exceeded")
            && body.contains("maximum context length exceeded")
    }));
}

#[tokio::test]
async fn an_opted_in_response_incomplete_chunk_is_a_terminal_with_mapped_finish_reason() {
    let text_delta = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "partial",
        "item_id": "msg_incomplete_1",
        "output_index": 0,
        "sequence_number": 1,
    });

    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    response.usage = Some(ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 5,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 0,
        }),
        total_tokens: 15,
    });

    let incomplete = json!({
        "type": "response.incomplete",
        "sequence_number": 2,
        "response": response,
    });

    // HTTP SSE refuses an incomplete terminal unless the caller opts in; this
    // is the opted-in half (the default half is
    // `a_streamed_incomplete_terminal_errors_by_default`).
    let model = Bound::new(
        OpenAI::new("test-key")
            .responses("gpt-5.4")
            .with_streamed_incomplete(super::IncompleteTerminal::Accept),
        MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[text_delta, incomplete]),
        },
    );
    let mut stream = model
        .stream(model.completion_request("hello").build())
        .await
        .expect("stream should start");

    let mut text = String::new();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item.expect("an opted-in incomplete stream should not error") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: delta },
                ..
            } => text.push_str(&delta),
            StreamEvent::Final(response) => final_response = Some(response),
            _ => {}
        }
    }

    // The partial output survives, and the terminal record maps the
    // incomplete status to the same finish reason as the unary path.
    assert_eq!(text, "partial");
    let final_response = final_response.expect("stream should yield a final response");
    assert_eq!(
        final_response.finish_reason,
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(final_response.usage.input_tokens, Some(10));
    assert_eq!(final_response.usage.output_tokens, Some(5));
    assert_eq!(final_response.usage.total_tokens, Some(15));
}

/// A multi-block reasoning done item (summaries + `encrypted_content`)
/// aggregates as exactly ONE reasoning part carrying every block in wire
/// order — never sibling parts sharing one `rs_*` id, which would replay
/// as duplicate reasoning input items carrying the identical id on the
/// next request.
#[tokio::test]
async fn multi_block_reasoning_done_item_yields_one_part() {
    let reasoning_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [
                {"type": "summary_text", "text": "step 1"},
                {"type": "summary_text", "text": "step 2"}
            ],
            "content": [],
            "encrypted_content": "enc_blob"
        }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[reasoning_done, completed]),
    })
    .await;

    let mut completed_reasoning = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamEvent::BlockEnd {
            block: Some(AssistantContent::Reasoning(reasoning)),
            ..
        } = item.expect("stream items should be ok")
        {
            completed_reasoning.push(reasoning);
        }
    }

    assert_eq!(
        completed_reasoning.len(),
        1,
        "one done item must complete exactly one reasoning part, got {completed_reasoning:?}"
    );
    let reasoning = completed_reasoning.first().expect("one part");
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.content,
        vec![
            ReasoningContent::Summary("step 1".to_string()),
            ReasoningContent::Summary("step 2".to_string()),
            ReasoningContent::Encrypted("enc_blob".to_string()),
        ],
        "every block survives, in wire order, inside the one part"
    );

    // The aggregated choice replays as exactly one reasoning input item.
    let choice = stream.snapshot();
    let reasoning_parts = choice
        .iter()
        .filter(|content| matches!(content, crate::message::AssistantContent::Reasoning(_)))
        .count();
    assert_eq!(
        reasoning_parts, 1,
        "history must carry one reasoning part per rs_* id, got {choice:?}"
    );
}

/// A `response.failed` after a fully-delivered tool call: the tool call is
/// content and flushes first, the terminal error follows, and nothing
/// (least of all a terminal record) comes after it.
#[tokio::test]
async fn response_failed_flushes_delivered_tool_calls_before_the_error() {
    let tool_call_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });

    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: Some("server_error".to_string()),
        message: "response stream failed".to_string(),
    });

    let failed = json!({
        "type": "response.failed",
        "sequence_number": 2,
        "response": response,
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
    })
    .await;

    // The flushed call (its block start and its completed end) precedes
    // the terminal error.
    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    // The correlator drives rig's id; the item id rides on `provider`.
    assert_eq!(tool_call.id.explicit(), Some("call_123"));
    let provider = tool_call.provider.as_ref().expect("provider ids are kept");
    assert_eq!(provider.call_id, "call_123");
    assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
    assert_eq!(tool_call.function.name, "example_tool");

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("response stream failed")
    }));
    assert!(
        stream.next().await.is_none(),
        "stream should terminate immediately after the terminal error"
    );
    assert!(stream.response.is_none());
}

/// Same ordering for a transport failure: fully-delivered tool call, then
/// the error, then the end — with no terminal record.
#[tokio::test]
async fn transport_error_flushes_delivered_tool_calls_before_the_error() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let tool_call_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([tool_call_done.to_string()])),
        Err(crate::http_client::Error::non_success_with_details(
            http::StatusCode::BAD_GATEWAY,
            http::HeaderMap::new(),
            r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
        )),
    ];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    assert_eq!(tool_call.id.explicit(), Some("call_123"));
    let provider = tool_call.provider.as_ref().expect("provider ids are kept");
    assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
    assert_eq!(
        err.http_status,
        Some(http::StatusCode::BAD_GATEWAY.as_u16())
    );

    assert!(
        stream.next().await.is_none(),
        "nothing may follow the terminal error"
    );
    assert!(stream.response.is_none());
}

/// A known terminal event with a data-level defect (malformed `usage`) is
/// a corrupt frame, not silent truncation: the error surfaces and, since
/// the terminal itself failed to parse, no terminal record is emitted.
#[tokio::test]
async fn known_terminal_with_malformed_usage_surfaces_error_without_terminal() {
    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": sample_response(ResponseStatus::Completed),
    });
    event["response"]["usage"] = json!("banana");

    let mut stream = responses_stream_of(&[event]).await;

    let mut saw_error = false;
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Final(_)) => saw_final = true,
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(err) => {
                assert!(
                    err.kind == ErrorKind::Json,
                    "expected a parse error item, got {err:?}"
                );
                saw_error = true;
            }
        }
    }

    assert!(saw_error, "the corrupt terminal must surface as an error");
    assert!(
        !saw_final,
        "a terminal that failed to parse must not produce a terminal record"
    );
    assert!(stream.response.is_none());
}

/// An invented event type stays skippable for forward compatibility; a
/// later genuine terminal still completes the stream.
#[tokio::test]
async fn unknown_event_type_is_skipped_and_stream_completes() {
    let unknown = json!({
        "type": "response.rocket_launch",
        "payload": { "count": 3 }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[unknown, completed]),
    })
    .await;

    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(_) = item.expect("unknown event types must not surface as errors")
        {
            saw_final = true;
        }
    }
    assert!(
        saw_final,
        "the genuine terminal must still complete the stream"
    );
}

#[tokio::test]
async fn refusal_content_part_frames_are_no_ops_and_refusal_text_streams() {
    // A refusal turn emits `response.content_part.added/.done` with a
    // `refusal` part — a shape outside the modeled text parts — followed
    // by the refusal text via `response.refusal.delta`. The part frames
    // must parse as no-ops (never error items); the deltas carry the
    // content.
    let part_added = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": { "type": "refusal", "refusal": "" }
    });
    let refusal_delta = json!({
        "type": "response.refusal.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 2,
        "delta": "I can't help with that."
    });
    let part_done = json!({
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 3,
        "part": { "type": "refusal", "refusal": "I can't help with that." }
    });
    let reasoning_part = json!({
        "type": "response.content_part.added",
        "item_id": "rs_1",
        "output_index": 1,
        "content_index": 0,
        "sequence_number": 4,
        "part": { "type": "reasoning_text", "text": "" }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 5,
        "response": sample_response(ResponseStatus::Completed),
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[
            part_added,
            refusal_delta,
            part_done,
            reasoning_part,
            completed,
        ]),
    })
    .await;

    let mut texts = Vec::new();
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        match item.expect("content-part frames must not surface as errors") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => texts.push(text),
            StreamEvent::Final(_) => saw_final = true,
            _ => {}
        }
    }

    assert_eq!(texts, ["I can't help with that."]);
    assert!(saw_final, "the terminal must still arrive");
}

#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::test_utils::MockStreamingClient;

    // Deltas then EOF without `response.completed`: the accumulator's
    // `saw_terminal` gate must withhold the terminal record rather than
    // present the truncated turn as a successful completion.
    let deltas = [
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hel"
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "lo"
        }),
    ];

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&deltas),
    })
    .await;

    let mut texts = Vec::new();
    let mut saw_terminal = false;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => texts.push(text),
            StreamEvent::Final(_) => saw_terminal = true,
            _ => {}
        }
    }

    assert_eq!(texts, ["hel", "lo"]);
    assert!(
        !saw_terminal,
        "EOF without response.completed must not synthesize a terminal record"
    );
    assert!(stream.response.is_none());
}

#[tokio::test]
async fn streaming_error_event_preserves_full_payload_in_live_loop() {
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::test_utils::MockStreamingClient;

    let payload = json!({
        "type": "error",
        "error": {
            "message": "boom",
            "code": "server_error",
            "type": "server_error"
        }
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[payload]),
    })
    .await;

    let err = stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider response error");
    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(
        err.provider_response_body()
            .is_some_and(|body| { body.contains("\"type\":\"error\"") && body.contains("boom") })
    );
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after error event"
    );
}

#[tokio::test]
async fn streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;

    let body = r#"{"error":{"message":"quota exceeded"}}"#;
    let mut stream = responses_stream(HttpErrorStreamingClient::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        body,
    ))
    .await;

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("HTTP non-success should surface as a stream error");
    assert_eq!(
        err.http_status,
        Some(http::StatusCode::TOO_MANY_REQUESTS.as_u16())
    );
    assert_eq!(err.provider_response_body(), Some(body));
    assert_eq!(
        err.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({"error": {"message": "quota exceeded"}}))
    );
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after HTTP non-success"
    );
}

/// The buffered unary path has no stream to carry error items, so a
/// corrupt known frame fails the whole decode — even when a valid terminal
/// follows — instead of returning a silently partial completion.
#[test]
fn corrupt_known_frame_fails_the_buffered_body() {
    let corrupt = json!({
        "type": "response.output_text.delta",
        "delta": 42
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });
    let body = format!("data: {corrupt}\ndata: {completed}\n");

    let err = stream_events_from_sse_body("openai", &body, None)
        .expect_err("a corrupt known frame must fail the buffered decode");
    assert!(
        err.to_string().contains("response.output_text.delta"),
        "the error should name the malformed event, got: {err}"
    );

    // Syntactically invalid JSON fails too.
    let body = format!("data: {{not json\ndata: {completed}\n");
    stream_events_from_sse_body("openai", &body, None)
        .expect_err("invalid JSON must fail the buffered decode");

    // Unknown event types stay skippable.
    let unknown = json!({ "type": "response.rocket_launch", "count": 3 });
    let body = format!("data: {unknown}\ndata: {completed}\n");
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("unknown event types must stay skippable");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Final(_))),
        "the genuine terminal must still be recorded"
    );
}

/// Envelope-less frames (ChatGPT's replayed bodies) are repaired and fed
/// through the same typed interpreter as the live loop, so the buffered
/// path agrees with the live path's semantics.
#[test]
fn envelope_less_frames_repair_onto_the_shared_interpreter() {
    let completed = json!({
        "type": "response.completed",
        "response": sample_response(ResponseStatus::Completed),
    });

    // A ChatGPT-style text delta with no envelope bookkeeping fields.
    let body = format!(
        "data: {}\ndata: {completed}\n",
        json!({ "type": "response.output_text.delta", "delta": "hi" })
    );
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("an envelope-less delta must repair and decode");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "hi"
    )));

    // Live-path parity, pinned: a function-call-arguments delta with no
    // `item_id` is keyed by the minted slot identity (the repair injects
    // `output_index: 0`), matching the live loop — it must flow into
    // assembly instead of vanishing.
    let body = format!(
        "data: {}\ndata: {completed}\n",
        json!({ "type": "response.function_call_arguments.delta", "delta": "{}" })
    );
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("an id-less args delta must repair and decode");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { id, delta: Delta::ToolArguments { .. } }
            if id == &crate::streaming::MintKind::Output.for_wire_index(0)
    )));

    // Live-path parity, pinned: an envelope-less bookkeeping event whose
    // data is intact (`.done` events) is a no-op, not an error as the old
    // salvage made it.
    let body = format!(
        "data: {}\ndata: {completed}\n",
        json!({ "type": "response.output_text.done", "text": "hi" })
    );
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("an envelope-less done event must repair to the live no-op");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::Final(_)))
    );

    // An envelope-less reasoning summary delta keys by the repaired
    // `output_index`, matching the live derivation.
    let body = format!(
        "data: {}\ndata: {completed}\n",
        json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" })
    );
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("an envelope-less summary delta must repair and decode");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { id, delta: Delta::Reasoning { text } }
            if id == &crate::streaming::MintKind::Output.for_wire_index(0) && text == "think"
    )));
}

/// The `max_output_tokens`-mid-tool-call shape: `arguments_delta`
/// frames stream partial JSON and the done item restates the same
/// truncated bytes (unparseable). Re-emitting the restatement as
/// another raw delta put the partial JSON in the buffer TWICE — a
/// delta-reassembling consumer rendered it twice and the bytes were
/// double-charged against the accumulation bound. Fragments seen →
/// the buffer already holds the bytes; only a fragment-less done item
/// (pure replay of a truncated restatement) still routes its raw
/// string through the buffer at all (#2258 P3).
#[test]
fn an_unparseable_restatement_is_not_reemitted_over_streamed_fragments() {
    let delta = json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "delta": "{\"x\":481",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 2,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":481",
            "status": "incomplete"
        },
    });
    let body = format!(
        "data: {delta}
data: {done}
"
    );

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("the truncated shape must decode");
    let raw_fragments: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockDelta {
                delta: Delta::ToolArguments { arguments },
                ..
            } => Some(arguments.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        raw_fragments,
        vec!["{\"x\":481"],
        "the streamed fragment is buffered once; the restatement adds nothing"
    );
}

/// The pure-replay half of the same policy: a truncated restatement
/// with NO preceding fragments must still reach the buffer (else the
/// bytes never arrive and the truncation policy has nothing to judge).
#[test]
fn a_fragmentless_unparseable_restatement_still_reaches_the_buffer() {
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":481",
            "status": "incomplete"
        },
    });
    let body = format!(
        "data: {done}
"
    );

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("the replayed truncated shape must decode");
    let raw_fragments = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: Delta::ToolArguments { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(raw_fragments, 1, "the raw bytes must reach the buffer once");
}

/// A slot mixing id-bearing and id-less reasoning frames (gateways and
/// ChatGPT's envelope-less replay bodies omit the id on a subset of a
/// slot's events) must key every frame — and the done item — by ONE
/// slot identity, the same discipline `tool_slots` applies. Per-event
/// resolution split the slot into `Wire("rs_1")` and `Minted(Output, 0)`,
/// and the done item superseded only one of them: the other survived as
/// an orphaned partial part carrying the same provider id.
#[tokio::test]
async fn mixed_id_and_id_less_reasoning_frames_share_one_slot_key() {
    let with_id = json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 1,
        "delta": "s1 ",
    });
    let id_less = json!({
        "type": "response.reasoning_summary_text.delta",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 2,
        "delta": "s2",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 3,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "s1 s2"}],
            "content": [],
            "status": "completed",
        },
    });
    let completed = json!({
        "type": "response.completed",
        "response": sample_response(ResponseStatus::Completed),
    });
    let body = format!("data: {with_id}\ndata: {id_less}\ndata: {done}\ndata: {completed}\n");

    let raw_choices =
        stream_events_from_sse_body("openai", &body, None).expect("the mixed slot must decode");
    let mut keys = std::collections::HashSet::new();
    for event in &raw_choices {
        match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { .. },
            }
            | StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { .. },
            }
            | StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning { .. },
                ..
            } => {
                keys.insert(id.clone());
            }
            _ => {}
        }
    }
    assert_eq!(
        keys.len(),
        1,
        "one slot, one assembly key — got {keys:?} across {raw_choices:?}"
    );

    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("the mixed slot should normalize");
    let reasoning_parts = response
        .choice
        .iter()
        .filter(|content| matches!(content, crate::completion::AssistantContent::Reasoning(_)))
        .count();
    assert_eq!(
        reasoning_parts, 1,
        "the done item supersedes the one delta-built part; nothing orphans"
    );
}

/// #2258 F3: an id-less reasoning delta is keyed by the minted
/// `output-{index}` identity, and the slot's `output_item.done` full block
/// (which always carries the real `rs_*` id) must adopt that minted
/// identity — otherwise the restated summary appends beside the
/// delta-built part and duplicates it. This is the ChatGPT envelope-less
/// replay shape: the repair injects `output_index: 0` into the delta while
/// the done item arrives envelope-full.
#[tokio::test]
async fn envelope_less_reasoning_deltas_are_superseded_by_their_done_item() {
    let delta = json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 2,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "think"}],
            "content": [],
            "status": "completed",
        },
    });
    let completed = json!({
        "type": "response.completed",
        "response": sample_response(ResponseStatus::Completed),
    });
    let body = format!("data: {delta}\ndata: {done}\ndata: {completed}\n");

    let raw_choices = stream_events_from_sse_body("openai", &body, None)
        .expect("the envelope-less reasoning replay must decode");
    // The done item's restatement shares the minted per-slot identity.
    assert!(raw_choices.iter().any(|event| matches!(
        event,
        StreamEvent::BlockEnd {
            id,
            end: BlockClose::Reasoning { reasoning: Some(_), .. },
            ..
        } if id == &crate::streaming::MintKind::Output.for_wire_index(0)
    )));

    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("chatgpt", raw_choices, &raw_response)
        .expect("replay should normalize");

    let reasoning: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning.len(),
        1,
        "deltas and their full block must collapse to one reasoning item: {reasoning:?}"
    );
    let occurrences = reasoning
        .iter()
        .flat_map(|item| item.content.iter())
        .filter(|content| match content {
            ReasoningContent::Summary(text) | ReasoningContent::Text { text, .. } => {
                text.contains("think")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        occurrences, 1,
        "the restated summary must supersede its deltas, not duplicate them"
    );
}

/// #2258 P2: text deltas for one message item interleaved with reasoning
/// must aggregate as ONE text part. Interleaving reasoning closes the open
/// text block downstream, so the adapter must re-emit `TextStart` with the
/// same item id when the item's text resumes — the accumulator's keyed
/// reactivation then reopens the block instead of minting a sibling.
#[tokio::test]
async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
    let events = [
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hello "
        }),
        json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_2",
            "output_index": 1,
            "summary_index": 0,
            "sequence_number": 2,
            "delta": "because"
        }),
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "world"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body = events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect::<String>();

    let raw_choices = stream_events_from_sse_body("openai", &body, None)
        .expect("the interleaved stream must decode");
    // The resumed item re-announces its block: two text `BlockStart { msg_1 }`.
    let starts = raw_choices
        .iter()
        .filter(|event| {
            matches!(
                event,
                StreamEvent::BlockStart { id, kind: BlockKind::Text { .. } }
                    if id == &BlockId::wire("msg_1")
            )
        })
        .count();
    assert_eq!(
        starts, 2,
        "returning to the same item must re-emit its text BlockStart: {raw_choices:?}"
    );

    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("replay should normalize");
    let texts: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        ["hello world"],
        "same-item text must aggregate as one part around the reasoning"
    );
    assert!(
        response
            .choice
            .iter()
            .any(|content| matches!(content, crate::completion::AssistantContent::Reasoning(_))),
        "the interleaved reasoning must survive"
    );
}

/// #2258 P3: two parallel function calls whose events all lack `fc_*` ids
/// must not share the `""` assembly key — each slot gets a minted
/// `output-{index}` identity shared by its added/delta/done events, so two
/// distinct calls assemble.
/// A slot whose `added` event carries a real `fc_*` id but whose later
/// args delta arrives id-less must keep ONE assembly key: slot-scoped
/// identity (the bridge) makes event-scoped key-splitting
/// unrepresentable, and the finalized call reports the wire id.
#[tokio::test]
async fn mixed_id_and_id_less_events_share_one_slot_key() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_real",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        // Id-less delta for the same slot: must resolve to the slot's
        // established key, not mint a second identity.
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "function_call",
                "id": "fc_real",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "{\"x\":1}",
                "status": "completed",
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body = events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect::<String>();

    let raw_choices = stream_events_from_sse_body("openai", &body, None)
        .expect("the mixed-id stream must decode");

    // Every tool event (block start, name delta, args delta, end) carries
    // the slot's single key — no fragment dangles under a second identity.
    let mut keys: Vec<BlockId> = raw_choices
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::ToolCall,
            }
            | StreamEvent::BlockDelta {
                id,
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
            }
            | StreamEvent::BlockEnd {
                id,
                end: BlockClose::ToolCall(_),
                ..
            } => Some(id.clone()),
            _ => None,
        })
        .collect();
    keys.dedup();
    assert_eq!(
        keys,
        [BlockId::wire("fc_real")],
        "one slot, one assembly key"
    );

    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("replay should normalize");
    let call = response
        .choice
        .iter()
        .find_map(|content| match content {
            crate::completion::AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("the call finalizes");
    assert_eq!(call.function.name, "tool_a");
    assert_eq!(call.function.arguments, serde_json::json!({"x": 1}));
}

#[tokio::test]
async fn parallel_id_less_function_calls_assemble_distinctly() {
    let call_item = |name: &str, call_id: &str, arguments: &str| {
        json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        })
    };
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": call_item("tool_a", "call_a", ""),
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "sequence_number": 2,
            "item": call_item("tool_b", "call_b", ""),
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 3,
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "sequence_number": 4,
            "delta": "{\"y\":2}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 5,
            "item": call_item("tool_a", "call_a", "{\"x\":1}"),
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "sequence_number": 6,
            "item": call_item("tool_b", "call_b", "{\"y\":2}"),
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 7,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body = events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect::<String>();

    let raw_choices = stream_events_from_sse_body("openai", &body, None)
        .expect("the id-less parallel-call stream must decode");
    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("replay should normalize");

    let mut calls: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::ToolCall(call) => Some((
                call.function.name.clone(),
                call.function.arguments.to_string(),
            )),
            _ => None,
        })
        .collect();
    calls.sort();
    assert_eq!(
        calls,
        [
            ("tool_a".to_owned(), json!({"x": 1}).to_string()),
            ("tool_b".to_owned(), json!({"y": 2}).to_string()),
        ],
        "each id-less slot must assemble its own call"
    );
}

/// A lost `output_item.done` frame followed by a healthy
/// `response.completed` must not discard the call as truncation: the
/// provider proved the turn ended, so the still-open slot closes at the
/// terminal and finalizes from its streamed fragments — with the full
/// dual-wire identity the added event announced. The same
/// terminal-drain the sibling adapters ship (Interactions at
/// `interaction.completed`, chat-compat at `finish_reason`).
#[tokio::test]
async fn a_lost_done_frame_does_not_discard_a_provider_completed_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"city\":\"Paris\"}"
        }),
        // The output_item.done frame is lost; the terminal still arrives.
        json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body = events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect::<String>();

    let raw_choices =
        stream_events_from_sse_body("openai", &body, None).expect("the stream must decode");
    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("replay should normalize");

    let calls: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "the provider-completed call must survive");
    let call = calls[0];
    assert_eq!(call.function.name, "get_weather");
    assert_eq!(call.function.arguments, json!({"city": "Paris"}));
    let provider = call.provider.as_ref().expect("the wire issued ids");
    assert_eq!(provider.call_id, "call_abc");
    assert_eq!(provider.item_id.as_deref(), Some("fc_1"));
}

/// #2258 P3: id-less argument fragments must surface as deltas (keyed by
/// the minted slot identity) rather than vanish; when the stream truncates
/// before the authoritative `output_item.done` restatement, the settled
/// truncation policy still applies — partial arguments never fabricate a
/// call.
#[tokio::test]
async fn id_less_args_deltas_surface_and_truncation_fabricates_no_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"loc\":"
        }),
    ];
    let body = events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect::<String>();

    let raw_choices = stream_events_from_sse_body("openai", &body, None)
        .expect("the truncated id-less stream must decode");
    // The fragment flowed into assembly under the minted identity.
    assert!(
        raw_choices.iter().any(|event| matches!(
            event,
            StreamEvent::BlockDelta {
                id,
                delta: Delta::ToolArguments { arguments },
            } if id == &crate::streaming::MintKind::Output.for_wire_index(0) && arguments == "{\"loc\":"
        )),
        "the id-less args fragment must surface as a delta: {raw_choices:?}"
    );

    // No done restatement arrived: the truncation policy withholds the
    // call rather than fabricating one from partial arguments.
    let raw_response = sample_response(ResponseStatus::Completed);
    let response = folded_stream_events("openai", raw_choices, &raw_response)
        .expect("the fold should not error");
    assert!(
        response.choice.is_empty(),
        "partial arguments must not fabricate a call: {:?}",
        response.choice
    );
}

#[test]
fn refusal_content_part_frames_do_not_fail_the_buffered_body() {
    // The ChatGPT buffered route replays recorded SSE bodies; a refusal
    // turn's `content_part` frames (an unmodeled `refusal` part) must not
    // fail the whole completion — the refusal text arrives via the
    // modeled `response.refusal.delta`.
    let part_added = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": { "type": "refusal", "refusal": "" }
    });
    let refusal_delta = json!({
        "type": "response.refusal.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 2,
        "delta": "no"
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": sample_response(ResponseStatus::Completed),
    });
    let body = format!(
        "data: {part_added}
data: {refusal_delta}
data: {completed}
"
    );

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("refusal content-part frames must not fail the buffered decode");
    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "no"
        )),
        "the refusal text must be delivered"
    );
}

/// One `message` output item, as a terminal response body states it.
fn message_output_item(id: &str, text: &str) -> crate::providers::openai::responses_api::Output {
    serde_json::from_value(json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "annotations": [], "text": text }],
    }))
    .expect("output message should deserialize")
}

/// The visible text parts of a folded choice, in order.
fn choice_text_parts(response: &crate::completion::CompletionResponse) -> Vec<String> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// The terminal restates the whole turn, and its message text IS the turn's
/// answer when nothing else stated it: a gateway that answers a unary call
/// with a replayed event stream can deliver a message only inside
/// `response.completed`'s `output` — no `output_text.delta`, no
/// `output_item.done` for it — so dropping that text loses the reply
/// entirely.
///
/// Driven on the plain `openai` provider: a body-only terminal is a shape
/// any Responses dialect can send, so the merge is no dialect's quirk.
#[test]
fn terminal_body_message_text_merges_when_no_delta_delivered_it() {
    let mut raw_response = sample_response(ResponseStatus::Completed);
    raw_response.output = vec![message_output_item("msg_body_1", "from body")];
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": raw_response,
    });
    let body = format!("data: {completed}\n");

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("a body-only terminal must decode");
    let response =
        folded_stream_events("openai", events, &raw_response).expect("the fold should not error");

    assert_eq!(
        choice_text_parts(&response),
        ["from body"],
        "text stated only in the terminal body must reach the choice once"
    );
    assert_eq!(response.message_id.as_deref(), Some("msg_body_1"));
}

/// The other half of that boundary: a terminal restating text the deltas
/// already delivered adds nothing. The merge publishes the terminal's
/// content only where no delta delivered it, so an ungated merge — or one
/// keyed on a slot the delta never opened — would state one turn's answer
/// twice.
#[test]
fn terminal_body_message_text_restating_a_delta_is_not_duplicated() {
    let text_delta = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_body_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "from body",
    });
    let mut raw_response = sample_response(ResponseStatus::Completed);
    raw_response.output = vec![message_output_item("msg_body_1", "from body")];
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": raw_response,
    });
    let body = format!("data: {text_delta}\ndata: {completed}\n");

    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("a restating terminal must decode");
    let response =
        folded_stream_events("openai", events, &raw_response).expect("the fold should not error");

    assert_eq!(
        choice_text_parts(&response),
        ["from body"],
        "the terminal's restatement must not duplicate the delta-built text"
    );
}

#[test]
fn streaming_error_event_preserves_full_payload() {
    let payload = r#"{"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
    let body = format!("data: {payload}\n");

    let err = stream_events_from_sse_body("openai", &body, None)
        .expect_err("error event should surface as a provider response error");

    assert_eq!(err.provider_response_status(), None);
    assert_eq!(err.provider_response_body(), Some(payload));
    let json = err
        .provider_response_json()
        .expect("raw body should be valid JSON")
        .expect("parsed JSON should be present");
    assert_eq!(json["error"]["code"], "server_error");
}

#[tokio::test]
async fn streaming_non_http_transport_error_stays_a_transport_error() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let chunks = vec![Err(crate::http_client::Error::InvalidContentType(
        http::HeaderValue::from_static("application/json"),
    ))];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("non-HTTP transport failure should surface as a transport error");
    assert_eq!(
        err.to_string(),
        "HttpError: Invalid content type was returned: \"application/json\""
    );
    assert_eq!(err.kind, ErrorKind::Http);
    // A response-less transport failure has no provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert_eq!(err.http_status, None);
}

#[tokio::test]
async fn response_completed_chunk_populates_final_usage() {
    let mut response = sample_response(ResponseStatus::Completed);
    response.usage = Some(ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 5,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 0,
        }),
        total_tokens: 15,
    });

    let event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });

    let usage = final_response_from_event(event)
        .await
        .usage
        .expect("the terminal carries usage");
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

/// The terminal `response.completed` frame carries usage. An object-shaped
/// `top_p` echoed on that frame (MiniMax-style endpoints, rig#2483) or a
/// numeric one buffered under `serde_json/arbitrary_precision` (rig#2493)
/// must not turn the terminal into a parse error — that both fails the turn
/// and loses the usage. Contrast `known_terminal_with_malformed_usage_*`:
/// a defect in a field rig reads is still an error.
#[tokio::test]
async fn response_completed_chunk_tolerates_object_shaped_top_p() {
    let mut response = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("sample response serializes");
    response["top_p"] = json!({ "value": 0.95 });
    response["usage"] = json!({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15
    });
    let event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });

    let usage = final_response_from_event(event)
        .await
        .usage
        .expect("the terminal carries usage");
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.total_tokens, 15);
}

#[tokio::test]
async fn response_completed_chunk_populates_reasoning_metadata_and_context() {
    let response = sample_response(ResponseStatus::Completed);
    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });
    let metadata = json!({
        "context": "all_turns",
        "effort": "ultra",
        "summary": null,
        "future_control": true
    });
    event["response"]["reasoning"] = metadata.clone();

    let response = final_response_from_event(event).await;
    assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
}

#[tokio::test]
async fn terminal_record_normalizes_into_the_stream_final() {
    let mut response = sample_response(ResponseStatus::Completed);
    response.usage = Some(ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 5,
        output_tokens_details: None,
        total_tokens: 15,
    });

    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });
    event["response"]["output"] = json!([{
        "type": "message",
        "id": "msg_stream_1",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "annotations": [], "text": "hi" }]
    }]);

    let final_response = stream_final_from_event(event).await;

    assert_eq!(final_response.provider, "openai");
    assert_eq!(final_response.model.as_deref(), Some("gpt-5.4"));
    // The assistant message ID (`msg_...`), never the response ID
    // (`resp_123`) that the same event carries.
    assert_eq!(final_response.message_id.as_deref(), Some("msg_stream_1"));
    assert_eq!(
        final_response.finish_reason,
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(final_response.usage.input_tokens, Some(10));
    assert_eq!(final_response.usage.output_tokens, Some(5));
    assert_eq!(final_response.usage.total_tokens, Some(15));
}

#[tokio::test]
async fn terminal_record_reports_tool_calls_when_the_stream_called_a_tool() {
    let tool_call_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[tool_call_done, completed]),
    })
    .await;

    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(response) = item.expect("completed stream should not error") {
            final_response = Some(response);
        }
    }

    // `completed` is reconciled up to `ToolCalls` by
    // `StreamingCompletionResponse`, using the call the stream actually
    // emitted.
    assert_eq!(
        final_response
            .expect("stream should yield a final response")
            .finish_reason,
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[test]
fn terminal_record_preserves_an_unknown_incomplete_reason() {
    let response = super::StreamingCompletionResponse {
        status: Some(ResponseStatus::Incomplete),
        incomplete_details: Some(IncompleteDetailsReason {
            reason: "MAX_TOOL_CALLS".to_string(),
        }),
        model: Some("gpt-5.4".to_string()),
        message_id: Some("msg_1".to_string()),
        ..super::StreamingCompletionResponse::new(None)
    };

    let final_response = super::terminal_record("openai", false, response.clone())
        .expect("the native record serializes");

    assert_eq!(
        final_response.finish_reason,
        Some(crate::completion::FinishReason::Other(
            "MAX_TOOL_CALLS".to_string()
        ))
    );
    assert_eq!(final_response.message_id.as_deref(), Some("msg_1"));
    assert_eq!(final_response.model.as_deref(), Some("gpt-5.4"));
    // The native record rides on `raw`, exactly as its wire type serializes.
    assert_eq!(
        final_response.raw,
        serde_json::to_value(&response).expect("serialize")
    );
}

#[tokio::test]
async fn done_sentinel_is_ignored_without_debug_parse_noise() {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("log buffer mutex should not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut response = sample_response(ResponseStatus::Completed);
    response.usage = Some(ResponsesUsage {
        input_tokens: 4,
        input_tokens_details: None,
        output_tokens: 2,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 0,
        }),
        total_tokens: 6,
    });

    // Scoped-subscriber tests must not run concurrently; see
    // `test_utils::scoped_tracing_subscriber_guard`.
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .without_time()
        .with_writer({
            let captured = captured.clone();
            move || SharedWriter(captured.clone())
        })
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&json!({
                "type": "response.completed",
                "sequence_number": 1,
                "response": response,
            }))
            .expect("response event should serialize")
        )),
    })
    .await;

    let mut final_usage = None;
    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(response) = item.expect("stream should complete successfully") {
            final_usage = Some(response.usage);
        }
    }

    let usage = final_usage.expect("expected final response");
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(6));

    let logs = String::from_utf8(
        captured
            .lock()
            .expect("log buffer mutex should not be poisoned")
            .clone(),
    )
    .expect("captured logs should be valid UTF-8");
    assert!(
        !logs.contains("Couldn't deserialize SSE data as StreamingCompletionChunk"),
        "expected [DONE] to bypass the parse-failure debug path, logs were: {logs}"
    );
}

#[tokio::test]
async fn malformed_frame_surfaces_error_and_stream_still_completes() {
    let delta = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "hello",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            delta.to_string(),
            "{not valid json".to_string(),
            completed.to_string(),
        ]),
    };
    let mut stream = responses_stream(http_client).await;

    let mut text = String::new();
    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            }) => text.push_str(&chunk),
            Ok(StreamEvent::Final(final_response)) => {
                terminal = Some(final_response);
            }
            // The item's text block opens under its `msg_*` id before the
            // first fragment.
            Ok(StreamEvent::BlockStart {
                kind: BlockKind::Text { .. },
                ..
            }) => {}
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(err) => {
                assert!(
                    err.kind == ErrorKind::Json,
                    "expected a JSON parse error item, got {err:?}"
                );
                saw_error = true;
            }
        }
    }

    // The malformed frame is surfaced as an error item, and the content
    // and genuine terminal on either side of it both still arrive.
    assert_eq!(text, "hello");
    assert!(saw_error, "malformed frame should surface an error item");
    assert!(
        terminal.is_some(),
        "stream should still emit its terminal record"
    );
}

/// An item id the wire left empty identifies nothing: a text delta under
/// `"item_id": ""` opens no block of its own (the text still streams), a
/// message item with `"id": ""` starts no message block, and neither
/// panics on the empty-id assertion.
#[test]
fn empty_item_ids_identify_nothing_and_do_not_panic() {
    let body = format!(
        "data: {}\ndata: {}\n",
        json!({
            "type": "response.output_text.delta",
            "item_id": "",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "still text",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "message",
                "id": "",
                "role": "assistant",
                "status": "completed",
                "content": []
            },
        }),
    );
    let events = stream_events_from_sse_body("openai", &body, None)
        .expect("an empty id is not a decode failure");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "still text"
    )));
    assert!(
        events.iter().all(|event| !matches!(
            event,
            StreamEvent::BlockStart { id: crate::streaming::BlockId::Wire(id), .. } if id.is_empty()
        )),
        "no block is keyed on the empty string: {events:?}"
    );
}

// ── custom calls, namespaces, incomplete policy, summary identity, native
//    evidence ─────────────────────────────────────────────────────────────

/// The folded choice of a buffered SSE body, decoded with the given options.
fn folded_body_with(
    options: ResponsesStreamOptions,
    events: &[serde_json::Value],
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let mut driver: WireDriver<Completion, _> =
        WireDriver::new(ResponsesDecoder::new("openai", options));
    let mut decoded = Vec::new();
    for event in events {
        driver.push(WireFrame::Text(event.to_string()));
        for item in driver.drain() {
            decoded.push(item?);
        }
    }
    driver.finish();
    for item in driver.drain() {
        decoded.push(item?);
    }
    folded_stream_events(
        "openai",
        decoded,
        &sample_response(ResponseStatus::Completed),
    )
}

fn completed_with_output(output: serde_json::Value) -> serde_json::Value {
    let mut response =
        serde_json::to_value(sample_response(ResponseStatus::Completed)).expect("serializes");
    response["output"] = output;
    json!({
        "type": "response.completed",
        "sequence_number": 9,
        "response": response,
    })
}

/// A `custom_tool_call` item decodes into a typed custom call: its raw input
/// verbatim (even though it parses as JSON), its namespace, and both provider
/// identifiers. Never a function call, and never dropped.
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
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.namespace.as_deref(), Some("editor"));
        assert_eq!(call.input, "{\"looks\":\"like json\"}");
        let provider = call.provider.as_ref().expect("the wire issued ids");
        assert_eq!(provider.call_id, "call_custom");
        assert_eq!(provider.item_id.as_deref(), Some("ctc_1"));
    }
}

/// A namespaced function call keeps its namespace on the streamed path,
/// whether the call closes at its `output_item.done` or, with that frame lost,
/// at the terminal drain (which reads the namespace `output_item.added`
/// announced).
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
        assert_eq!(call.function.name, "search", "{case}");
        assert_eq!(call.function.namespace.as_deref(), Some("web"), "{case}");
        assert_eq!(call.function.arguments, json!({"q": "rig"}), "{case}");
    }
}

fn incomplete_turn_events() -> [serde_json::Value; 2] {
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    [
        json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "partial",
            "item_id": "msg_incomplete_1",
            "output_index": 0,
            "sequence_number": 1,
        }),
        json!({
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": response,
        }),
    ]
}

/// HTTP SSE is strict by default: a terminal `response.incomplete` ends the
/// stream with an error carrying the provider's event, and no terminal record
/// presents the partial turn as completed.
#[tokio::test]
async fn a_streamed_incomplete_terminal_errors_by_default() {
    let mut stream = responses_stream_of(&incomplete_turn_events()).await;
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Final(_)) => {
                panic!("an incomplete turn must not produce a final record")
            }
            Ok(_) => {}
            Err(err) => {
                error = Some(err);
                break;
            }
        }
    }
    let error = error.expect("the incomplete terminal must surface an error");
    let rendered = error.to_string();
    assert!(
        rendered.contains("response.incomplete") && rendered.contains("max_output_tokens"),
        "the error must carry the provider's terminal event: {rendered}"
    );
}

/// The per-request opt-in lives on the wire: with it, the same turn keeps its
/// partial output and ends with the truthful `Length` finish reason.
#[tokio::test]
async fn a_streamed_incomplete_terminal_is_accepted_with_the_opt_in() {
    let wire = OpenAI::new("test-key")
        .responses("gpt-5.4")
        .with_streamed_incomplete(super::IncompleteTerminal::Accept);
    let model = Bound::new(
        wire,
        MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&incomplete_turn_events()),
        },
    );
    let mut stream = model
        .stream(model.completion_request("hello").build())
        .await
        .expect("stream should start");
    let mut text = String::new();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item.expect("an accepted incomplete stream must not error") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: delta },
                ..
            } => text.push_str(&delta),
            StreamEvent::Final(response) => final_response = Some(response),
            _ => {}
        }
    }
    assert_eq!(text, "partial");
    assert_eq!(
        final_response.expect("a final record").finish_reason,
        Some(crate::completion::FinishReason::Length)
    );
}

/// A unary reply accepts an incomplete terminal whatever the wire's streamed
/// policy, including when the gateway answers the unary call with an event
/// stream: the decoder the wire hands a unary call accepts it.
#[test]
fn a_unary_reply_accepts_an_incomplete_terminal() {
    use crate::wire::{Mode, Wire};

    let wire = OpenAI::new("test-key").responses("gpt-5.4");
    let mut driver: WireDriver<Completion, _> = WireDriver::new(wire.decoder(Mode::Unary));
    let mut decoded = Vec::new();
    for event in incomplete_turn_events() {
        driver.push(WireFrame::Text(event.to_string()));
        decoded.extend(driver.drain());
    }
    driver.finish();
    decoded.extend(driver.drain());
    let decoded = decoded
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("a unary incomplete reply is not an error");
    let finish = decoded.iter().find_map(|event| match event {
        StreamEvent::Final(response) => Some(response.finish_reason.clone()),
        _ => None,
    });
    assert_eq!(finish, Some(Some(crate::completion::FinishReason::Length)));
}

/// A websocket-style caller picks the accepting policy explicitly on the
/// decoder API, independently of the wire's HTTP SSE default.
#[test]
fn the_decoder_api_lets_a_session_accept_an_incomplete_terminal() {
    let wire = OpenAI::new("test-key").responses("gpt-5.4");
    let mut driver: WireDriver<Completion, _> =
        WireDriver::new(wire.decoder_with_incomplete(super::IncompleteTerminal::Accept));
    let mut decoded = Vec::new();
    for event in incomplete_turn_events() {
        driver.push(WireFrame::Text(event.to_string()));
        decoded.extend(driver.drain());
    }
    driver.finish();
    decoded.extend(driver.drain());
    assert!(
        decoded.iter().all(Result::is_ok),
        "an accepted incomplete terminal decodes without error: {decoded:?}"
    );
    assert!(
        decoded
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final(_)))),
        "an accepted incomplete terminal produces the final record"
    );
}

/// The summary delta and done events are distinct shapes: a delta event that
/// carries `text` instead of `delta` is a defect of a known event, not a
/// fragment, and the done event's `text` is not read as a `delta`.
#[test]
fn summary_delta_and_done_keep_their_own_members() {
    let delta_with_text = json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 1,
        "text": "wrong member",
    });
    assert!(
        matches!(
            classify_responses_frame(&delta_with_text.to_string()),
            WireEvent::Corrupt(_)
        ),
        "a delta without `delta` must not decode"
    );

    let done_with_delta = json!({
        "type": "response.reasoning_summary_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 2,
        "delta": "wrong member",
    });
    assert!(
        matches!(
            classify_responses_frame(&done_with_delta.to_string()),
            WireEvent::Corrupt(_)
        ),
        "a done without `text` must not decode"
    );

    let done = json!({
        "type": "response.reasoning_summary_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "sequence_number": 3,
        "text": "the whole summary",
    });
    let WireEvent::Known(StreamingCompletionChunk::Delta(ItemChunk {
        data: ItemChunkKind::ReasoningSummaryTextDone(chunk),
        ..
    })) = classify_responses_frame(&done.to_string())
    else {
        panic!("a well-formed done event decodes");
    };
    assert_eq!(chunk.text, "the whole summary");
}

/// `response.function_call_arguments.done` carries its arguments in the same
/// type as the `output_item.done` item, so the two observations reconcile
/// through the one classification.
#[test]
fn arguments_done_and_the_done_item_reconcile_through_one_type() {
    let args_done = json!({
        "type": "response.function_call_arguments.done",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "arguments": "{\"a\": 1}",
    });
    let WireEvent::Known(StreamingCompletionChunk::Delta(ItemChunk {
        data: ItemChunkKind::FunctionCallArgsDone(chunk),
        ..
    })) = classify_responses_frame(&args_done.to_string())
    else {
        panic!("the args-done event decodes");
    };
    let item: crate::providers::openai::responses_api::OutputFunctionCall =
        serde_json::from_value(json!({
            "id": "fc_1",
            "call_id": "call_1",
            "name": "f",
            "arguments": "{\"a\":1}",
            "status": "completed",
        }))
        .expect("the done item decodes");
    assert_eq!(
        chunk.arguments.reconcile(&item.arguments),
        Ok(json!({"a": 1}))
    );
    // Serialization keeps the stringified spelling the provider sent.
    assert_eq!(
        serde_json::to_value(&chunk).expect("serializes")["arguments"],
        json!("{\"a\": 1}")
    );
}

/// A function-call `output_item.done` whose restated arguments do not
/// parse, under the given item status.
fn malformed_function_call_done(status: &str, sequence_number: u64) -> serde_json::Value {
    json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": sequence_number,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":481",
            "status": status
        },
    })
}

/// An argument delta for the slot [`malformed_function_call_done`] closes.
fn argument_delta(delta: &str) -> serde_json::Value {
    json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "delta": delta,
    })
}

/// Drain a scripted Responses stream, returning the tool calls it delivered
/// and every error it surfaced, in order.
async fn delivered_calls_and_errors(
    events: &[serde_json::Value],
) -> (Vec<crate::message::ToolCall>, Vec<ErrorReport>) {
    let mut stream = responses_stream_of(events).await;
    let mut calls = Vec::new();
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(call)),
                ..
            }) => calls.push(call),
            Ok(_) => {}
            Err(error) => errors.push(error),
        }
    }
    (calls, errors)
}

/// The streamed half of the asserted-status policy: a call the provider
/// states `completed` or `in_progress` whose arguments do not parse is refused
/// by name, never silently dropped and never delivered.
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
                    if detail.name == "add" && detail.raw == "{\"x\":481"
            ),
            "{status}: the refusal keeps the raw restatement for recovery: {error:?}"
        );
    }
}

/// The unasserted half: a call the provider marks `incomplete` is dropped
/// without an error, as before.
#[tokio::test]
async fn a_streamed_unasserted_call_with_unparseable_arguments_is_dropped() {
    let (calls, errors) =
        delivered_calls_and_errors(&[malformed_function_call_done("incomplete", 1)]).await;
    assert!(calls.is_empty(), "the call is dropped: {calls:?}");
    assert!(errors.is_empty(), "dropping is not an error: {errors:?}");
}

/// Fragments that stream the same malformed bytes the restatement repeats
/// reach the same verdict: refused when asserted, dropped when not.
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

/// A delta whose bytes parse (whitespace, which is a parameterless `{}`, or a
/// complete `{}`) must not answer for a restatement that does not: the call
/// is never delivered with arguments the provider did not assert.
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

/// The streamed and unary surfaces refuse the same asserted call in the same
/// words, so an operator reading logs cannot tell them apart.
#[tokio::test]
async fn streaming_and_unary_refuse_an_asserted_malformed_call_alike() {
    let (_, errors) =
        delivered_calls_and_errors(&[malformed_function_call_done("completed", 1)]).await;
    let [streamed] = errors.as_slice() else {
        panic!("exactly one streamed refusal, got {errors:?}");
    };

    let mut response = sample_response(ResponseStatus::Completed);
    response.output = vec![
        serde_json::from_value(malformed_function_call_done("completed", 1)["item"].clone())
            .expect("the function-call item decodes"),
    ];
    let unary = super::super::wire::fold_body("openai", response)
        .expect_err("the unary surface refuses the same call");
    let ProviderError::Response(unary) = unary else {
        panic!("the unary refusal is a response error, got {unary:?}");
    };
    assert_eq!(streamed.message, unary);
}

/// Refusal deltas stream into their own text block per content part, marked
/// as a refusal and keyed by a deterministic stream-order minted id; output
/// text of the same item reopens its own block after a refusal part, so
/// nothing is merged across the two.
#[test]
fn refusal_parts_stream_into_their_own_marked_blocks() {
    let delta = |kind: &str, content_index: u64, sequence_number: u64, text: &str| {
        json!({
            "type": kind,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": content_index,
            "sequence_number": sequence_number,
            "delta": text
        })
    };
    let events = [
        delta("response.output_text.delta", 0, 1, "Hello"),
        delta("response.refusal.delta", 1, 2, "I can't"),
        delta("response.refusal.delta", 1, 3, " do that"),
        delta("response.output_text.delta", 0, 4, " there"),
        delta("response.refusal.delta", 2, 5, "Nor that"),
        json!({
            "type": "response.completed",
            "sequence_number": 6,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    let decoded = stream_events_from_sse_body("openai", &body, None).expect("the body decodes");

    let refusal_starts: Vec<BlockId> = decoded
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Text { additional_params },
            } if super::super::is_refusal(additional_params.as_ref()) => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        refusal_starts,
        [
            BlockId::minted(crate::streaming::MintKind::Refusal, 0),
            BlockId::minted(crate::streaming::MintKind::Refusal, 1),
        ],
        "one marked block per refusal part, minted in stream order"
    );

    let response = folded_stream_events(
        "openai",
        decoded,
        &sample_response(ResponseStatus::Completed),
    )
    .expect("the stream folds");
    let parts: Vec<(String, bool)> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Text(text) => (
                text.text.clone(),
                super::super::is_refusal(text.additional_params.as_ref()),
            ),
            other => panic!("expected text blocks, got {other:?}"),
        })
        .collect();
    assert_eq!(
        parts,
        [
            ("Hello there".to_string(), false),
            ("I can't do that".to_string(), true),
            ("Nor that".to_string(), true),
        ]
    );
}

/// A message item's `done` ends a refusal part's block even though the event
/// it emits leaves anonymous text open: an id-less output-text delta after it
/// opens a text block of its own instead of extending the refusal.
#[test]
fn an_id_less_text_delta_after_a_message_done_never_extends_a_refusal() {
    let events = [
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "No"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "refusal", "refusal": "No" }]
            }
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "Hello"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let body: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    let decoded = stream_events_from_sse_body("openai", &body, None).expect("the body decodes");
    let response = folded_stream_events(
        "openai",
        decoded,
        &sample_response(ResponseStatus::Completed),
    )
    .expect("the stream folds");
    let parts: Vec<(String, bool)> = response
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Text(text) => (
                text.text.clone(),
                super::super::is_refusal(text.additional_params.as_ref()),
            ),
            other => panic!("expected text blocks, got {other:?}"),
        })
        .collect();
    assert_eq!(
        parts,
        [("No".to_string(), true), ("Hello".to_string(), false)]
    );
}

fn opaque_message_fixture(parts: serde_json::Value) -> serde_json::Value {
    json!({"type":"message","id":"msg_opaque","role":"assistant","status":"completed","content":parts})
}

fn replay_opaque_choice(choice: Vec<AssistantContent>) -> serde_json::Value {
    let input: Vec<super::super::InputItem> = crate::message::Message::Assistant {
        id: Some("msg_opaque".into()),
        content: choice,
    }
    .try_into()
    .expect("same-wire replay");
    serde_json::to_value(input).expect("input JSON")
}

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

#[test]
fn opaque_message_late_part_retains_payload_with_standalone_arrival_order() {
    // SourceOrder integration in the composed branch restores [A, unknown, B].
    // This standalone decoder's fold registers parts on first delivery.
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
        json!([
            {"type":"output_text","text":"A"},{"type":"output_text","text":"B"},unknown
        ])
    );
}

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
        let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
            panic!("one reasoning item: {:?}", response.choice);
        };
        assert_eq!(
            reasoning.content,
            vec![
                ReasoningContent::Summary("known".into()),
                ReasoningContent::OpaqueSummary(unknown.clone())
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

#[test]
fn complete_snapshot_replacement_resets_only_owned_extras_contiguously() {
    use crate::message::AdditionalParams;
    let before = AdditionalParams::new(
        json!({"openai_responses":{"annotations":["a","b"],"old":true}})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let after = AdditionalParams::new(
        json!({"openai_responses":{"annotations":["replacement"]}})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut out = AdapterOutput::new();
    out.text_start(BlockId::wire("msg_opaque"), None);
    out.text_meta(
        AdditionalParams::new(
            json!({"unrelated":{"list":[1]},"openai_responses_part":{"kind":"refusal"}})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap(),
    );
    out.text_meta(before.clone());
    super::message_snapshot_metadata(Some(&before), Some(&after), Some("final_answer"), &mut out);
    let events: Vec<_> = out.into_items().into_iter().map(Result::unwrap).collect();
    assert!(
        matches!(&events[events.len()-2],StreamEvent::BlockDelta {delta:Delta::TextMeta {additional_params},..} if additional_params.get("openai_responses") == Some(&json!(null)))
    );
    assert!(
        matches!(&events[events.len()-1],StreamEvent::BlockDelta {delta:Delta::TextMeta {additional_params},..} if additional_params.get("openai_responses") == Some(&json!({"annotations":["replacement"],"phase":"final_answer"})))
    );
    let mut folded = crate::streaming::BlockAccumulator::new();
    for event in &events {
        folded.apply(event).unwrap();
    }
    let choice = folded.finish();
    let AssistantContent::Text(text) = &choice[0] else {
        panic!("text")
    };
    let params = text.additional_params.as_ref().unwrap();
    assert_eq!(params.get("unrelated"), Some(&json!({"list":[1]})));
    assert_eq!(
        params.get("openai_responses_part"),
        Some(&json!({"kind":"refusal"}))
    );
    assert_eq!(
        params.get("openai_responses"),
        Some(&json!({"annotations":["replacement"],"phase":"final_answer"}))
    );
}

#[test]
fn complete_snapshot_removing_all_extras_emits_empty_replacement() {
    use crate::message::AdditionalParams;
    let before = AdditionalParams::new(
        json!({"openai_responses":{"annotations":["a"]}})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut out = AdapterOutput::new();
    out.text_start(BlockId::wire("msg_empty_extras"), None);
    out.text_meta(before.clone());
    super::message_snapshot_metadata(Some(&before), None, None, &mut out);
    let events: Vec<_> = out.into_items().into_iter().map(Result::unwrap).collect();
    assert!(
        matches!(&events[events.len()-2],StreamEvent::BlockDelta {delta:Delta::TextMeta {additional_params},..} if additional_params.get("openai_responses") == Some(&json!(null)))
    );
    assert!(
        matches!(&events[events.len()-1],StreamEvent::BlockDelta {delta:Delta::TextMeta {additional_params},..} if additional_params.get("openai_responses") == Some(&json!({})))
    );
    let mut folded = crate::streaming::BlockAccumulator::new();
    for event in &events {
        folded.apply(event).unwrap();
    }
    let choice = folded.finish();
    let AssistantContent::Text(text) = &choice[0] else {
        panic!("text")
    };
    assert_eq!(
        text.additional_params
            .as_ref()
            .unwrap()
            .get("openai_responses"),
        Some(&json!({}))
    );
}

#[test]
fn reasoning_done_waits_for_late_terminal_parts_without_losing_its_position() {
    let summary = json!({"type":"future_summary","payload":[1]});
    let content = json!({"type":"future_content","payload":{"a":true}});
    let done = json!({"type":"reasoning","id":"rs_late","summary":[{"type":"summary_text","text":"known"}],"content":[],"signature":"signature"});
    let terminal_item = json!({"type":"reasoning","id":"rs_late","summary":[{"type":"summary_text","text":"known"},summary],"content":[content],"signature":"signature"});
    for with_delta in [false, true] {
        let mut driver: WireDriver<Completion, _> = WireDriver::new(ResponsesDecoder::new(
            "openai",
            ResponsesStreamOptions::strict(),
        ));
        let mut events = Vec::new();
        if with_delta {
            driver.push(WireFrame::Text(json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}).to_string()));
            events.extend(driver.drain().map(Result::unwrap));
            assert!(events.iter().any(|event| matches!(event,StreamEvent::BlockDelta {delta:Delta::Reasoning {text},..} if text == "known")));
        }
        for sequence_number in [1, 2] {
            driver.push(WireFrame::Text(json!({"type":"response.output_item.done","output_index":0,"sequence_number":sequence_number,"item":done}).to_string()));
            events.extend(driver.drain().map(Result::unwrap));
        }
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockStart {
                kind: BlockKind::Reasoning { .. },
                ..
            }
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockEnd {
                end: BlockClose::Reasoning { .. },
                ..
            }
        )));
        driver.push(WireFrame::Text(json!({"type":"response.output_text.delta","item_id":"msg_late","output_index":1,"content_index":0,"sequence_number":3,"delta":"answer"}).to_string()));
        events.extend(driver.drain().map(Result::unwrap));
        driver.push(WireFrame::Text(
            completed_with_output(json!([terminal_item])).to_string(),
        ));
        events.extend(driver.drain().map(Result::unwrap));
        driver.finish();
        events.extend(driver.drain().map(Result::unwrap));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    StreamEvent::BlockEnd {
                        end: BlockClose::Reasoning {
                            wire_sent: true,
                            ..
                        },
                        ..
                    }
                ))
                .count(),
            1
        );
        let folded = folded_stream_events(
            "openai",
            events,
            &sample_response(ResponseStatus::Completed),
        )
        .unwrap();
        assert!(matches!(&folded.choice[0], AssistantContent::Reasoning(_)));
        assert!(matches!(&folded.choice[1],AssistantContent::Text(text) if text.text == "answer"));
        let replay = replay_opaque_choice(folded.choice);
        assert_eq!(replay[0]["summary"], terminal_item["summary"]);
        assert_eq!(replay[0]["content"], terminal_item["content"]);
        assert_eq!(replay[0]["signature"], "signature");
    }
    let whole: super::super::Output = serde_json::from_value(terminal_item.clone()).unwrap();
    let replay = replay_opaque_choice(super::super::tests::folded_choice(vec![whole]));
    assert_eq!(replay[0]["summary"], terminal_item["summary"]);
    assert_eq!(replay[0]["content"], terminal_item["content"]);
}

#[test]
fn opaque_reasoning_partial_flush_keeps_content_and_truthful_wire_end() {
    let raw = json!({"type":"future_summary","payload":[1]});
    let item = json!({"type":"reasoning","id":"rs_partial","summary":[{"type":"summary_text","text":"known"},raw]});
    for with_done in [false, true] {
        for ending in ["eof", "error", "incomplete", "failed"] {
            let mut driver: WireDriver<Completion, _> = WireDriver::new(ResponsesDecoder::new(
                "openai",
                ResponsesStreamOptions::strict(),
            ));
            let mut frames = vec![
                json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_partial","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}),
                json!({"type":"response.reasoning_summary_part.done","item_id":"rs_partial","output_index":0,"summary_index":1,"sequence_number":1,"part":raw}),
            ];
            if with_done {
                frames.push(json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}));
            }
            if ending == "error" {
                frames.push(json!({"error":{"message":"failure","type":"server_error"}}));
            }
            if matches!(ending, "incomplete" | "failed") {
                let mut terminal = completed_with_output(json!([item]));
                terminal["type"] = json!(format!("response.{ending}"));
                terminal["response"]["status"] = json!(ending);
                frames.push(terminal);
            }
            let mut events = Vec::new();
            let mut errors = 0;
            for frame in frames {
                driver.push(WireFrame::Text(frame.to_string()));
                for event in driver.drain() {
                    match event {
                        Ok(event) => events.push(event),
                        Err(_) => errors += 1,
                    }
                }
            }
            driver.finish();
            for event in driver.drain() {
                match event {
                    Ok(event) => events.push(event),
                    Err(_) => errors += 1,
                }
            }
            assert_eq!(errors, usize::from(ending != "eof"), "{ending}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Final(_)))
            );
            let closes: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    StreamEvent::BlockEnd {
                        end:
                            BlockClose::Reasoning {
                                reasoning: Some(reasoning),
                                wire_sent,
                                ..
                            },
                        ..
                    } => Some((reasoning, *wire_sent)),
                    _ => None,
                })
                .collect();
            assert_eq!(closes.len(), 1, "{ending}");
            assert_eq!(
                closes[0].1,
                with_done || matches!(ending, "incomplete" | "failed"),
                "{ending}"
            );
            assert_eq!(
                closes[0].0.content,
                vec![
                    ReasoningContent::Summary("known".into()),
                    ReasoningContent::OpaqueSummary(raw.clone())
                ]
            );
        }
    }
}

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

#[test]
fn unchanged_message_snapshots_do_not_reopen_text_blocks() {
    let item = opaque_message_fixture(json!([{"type":"output_text","text":"A"}]));
    let frames = [
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        completed_with_output(json!([item])),
    ];
    let body = frames
        .iter()
        .map(|frame| format!("data: {frame}\n"))
        .collect::<String>();
    let events = stream_events_from_sse_body("openai", &body, None).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                StreamEvent::BlockStart {
                    kind: BlockKind::Text { .. },
                    ..
                }
            ))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                StreamEvent::BlockEnd {
                    end: BlockClose::Text,
                    ..
                }
            ))
            .count(),
        0
    );
}

/// Drive raw frames through a strict decoder, keeping events and errors apart.
fn driven_frames(frames: &[serde_json::Value]) -> (Vec<StreamEvent>, Vec<ProviderError>) {
    let mut driver: WireDriver<Completion, _> = WireDriver::new(ResponsesDecoder::new(
        "openai",
        ResponsesStreamOptions::strict(),
    ));
    let mut events = Vec::new();
    let mut errors = Vec::new();
    let mut keep = |item: Result<StreamEvent, ProviderError>| match item {
        Ok(event) => events.push(event),
        Err(error) => errors.push(error),
    };
    for frame in frames {
        driver.push(WireFrame::Text(frame.to_string()));
        driver.drain().for_each(&mut keep);
    }
    driver.finish();
    driver.drain().for_each(&mut keep);
    (events, errors)
}

fn unknown_payloads(events: &[StreamEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Unknown(payload) => Some(payload.value().clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn an_unknown_part_before_parent_evidence_joins_the_item_that_evidence_names() {
    let unknown = json!({"type":"future_part","payload":{"list":[1]}});
    // The item ID spells like reasoning for both parents; only typed evidence
    // binds the kind. The evidence items once carried other ids (msg_opaque,
    // rs_opaque); since round-2 H5 a waiting part binds only to the item its
    // event named, so the part event and its item agree here, as in an honest
    // stream. A part naming another item is pinned separately
    // (a_waiting_part_whose_item_never_arrives_leaves_raw).
    let part = json!({"type":"response.content_part.added","item_id":"rs_spelling_only","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    let message = |content| json!({"type":"message","id":"rs_spelling_only","role":"assistant","status":"completed","content":content});
    let reasoning = |content| json!({"type":"reasoning","id":"rs_spelling_only","summary":[],"content":content});
    for parent in ["message", "reasoning"] {
        let item = |content| {
            if parent == "message" {
                message(content)
            } else {
                reasoning(content)
            }
        };
        for evidence in ["added", "done", "terminal"] {
            let mut frames = vec![part.clone()];
            match evidence {
                "added" => frames.push(json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":item(json!([]))})),
                "done" => frames.push(json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item(json!([]))})),
                _ => {}
            }
            frames.push(completed_with_output(if evidence == "terminal" {
                json!([item(json!([unknown]))])
            } else {
                json!([])
            }));
            let (events, errors) = driven_frames(&frames);
            assert!(errors.is_empty(), "{parent}/{evidence}: {errors:?}");
            assert!(unknown_payloads(&events).is_empty(), "{parent}/{evidence}");
            let folded = folded_stream_events(
                "openai",
                events,
                &sample_response(ResponseStatus::Completed),
            )
            .unwrap();
            assert_eq!(
                folded.choice.len(),
                1,
                "{parent}/{evidence}: {:?}",
                folded.choice
            );
            if parent == "reasoning" {
                let AssistantContent::Reasoning(reasoning) = &folded.choice[0] else {
                    panic!(
                        "{evidence}: reasoning, not a fabricated message: {:?}",
                        folded.choice
                    );
                };
                assert_eq!(
                    reasoning.content,
                    vec![ReasoningContent::OpaqueContent(unknown.clone())]
                );
                assert_eq!(
                    replay_opaque_choice(folded.choice)[0]["content"],
                    json!([unknown])
                );
            } else {
                assert!(
                    matches!(&folded.choice[0], AssistantContent::Text(_)),
                    "{evidence}"
                );
                assert_eq!(
                    replay_opaque_choice(folded.choice)[0]["content"],
                    json!([unknown])
                );
            }
        }
    }
}

#[test]
fn a_known_typed_part_binds_waiting_unknown_siblings_in_content_order() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let frames = [
        json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":1,"sequence_number":1,"delta":"B"}),
        completed_with_output(json!([])),
    ];
    let (events, errors) = driven_frames(&frames);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    assert_eq!(
        replay_opaque_choice(folded.choice)[0]["content"],
        json!([unknown, {"type":"output_text","text":"B"}])
    );
}

#[test]
fn unresolved_unknown_parts_leave_raw_before_the_flush_without_a_fabricated_choice() {
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let parts = [
        json!({"type":"response.content_part.added","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        // An identical restatement collapses; a distinct one is kept.
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":1,"part":first}),
        json!({"type":"response.content_part.done","item_id":"item_0","output_index":0,"content_index":0,"sequence_number":2,"part":second}),
    ];
    for ending in ["eof", "error", "failed", "incomplete", "completed"] {
        let mut frames = parts.to_vec();
        match ending {
            "error" => frames.push(json!({"error":{"message":"failure","type":"server_error"}})),
            "failed" | "incomplete" => {
                let mut terminal = completed_with_output(json!([]));
                terminal["type"] = json!(format!("response.{ending}"));
                terminal["response"]["status"] = json!(ending);
                frames.push(terminal);
            }
            "completed" => frames.push(completed_with_output(json!([]))),
            _ => {}
        }
        let (events, errors) = driven_frames(&frames);
        assert_eq!(
            errors.len(),
            usize::from(!matches!(ending, "eof" | "completed")),
            "{ending}"
        );
        assert_eq!(
            unknown_payloads(&events),
            vec![first.clone(), second.clone()],
            "{ending}"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::BlockStart { .. })),
            "{ending}: no message or reasoning is fabricated: {events:?}"
        );
        if ending == "completed" {
            let unknown_at = events
                .iter()
                .rposition(|event| matches!(event, StreamEvent::Unknown(_)))
                .unwrap();
            let final_at = events
                .iter()
                .position(|event| matches!(event, StreamEvent::Final(_)))
                .unwrap();
            assert!(unknown_at < final_at);
            let folded = folded_stream_events(
                "openai",
                events,
                &sample_response(ResponseStatus::Completed),
            )
            .unwrap();
            assert!(folded.choice.is_empty(), "{:?}", folded.choice);
        }
    }
}

#[test]
fn a_part_restated_as_the_other_kind_is_refused_before_any_mutation() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let opaque_first = json!({"type":"response.content_part.added","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"part":unknown});
    let text_first = json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"});
    let added = json!({"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":opaque_message_fixture(json!([]))});
    let cases = [
        // opaque, then a known text delta
        (
            vec![
                added.clone(),
                opaque_first.clone(),
                json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":2,"delta":"late"}),
            ],
            true,
        ),
        // opaque, then a known output_text part
        (
            vec![
                added.clone(),
                opaque_first.clone(),
                json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":2,"part":{"type":"output_text","text":"late"}}),
            ],
            true,
        ),
        // opaque, then a terminal stating known text at that index
        (
            vec![
                added.clone(),
                opaque_first.clone(),
                completed_with_output(json!([opaque_message_fixture(
                    json!([{"type":"output_text","text":"late"}])
                )])),
            ],
            true,
        ),
        // known text, then an opaque part at that index
        (
            vec![
                text_first.clone(),
                json!({"type":"response.content_part.done","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":1,"part":unknown}),
            ],
            false,
        ),
    ];
    for (index, (frames, stored_opaque)) in cases.into_iter().enumerate() {
        let (events, errors) = driven_frames(&frames);
        assert_eq!(errors.len(), 1, "case {index}: {errors:?}");
        let message = errors[0].to_string();
        assert!(
            message.contains("conflicting Responses content part kind"),
            "case {index}: {message}"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::Final(_))),
            "case {index}"
        );
        let texts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let markers: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::BlockDelta {
                    delta: Delta::TextMeta { additional_params },
                    ..
                }
                | StreamEvent::BlockStart {
                    kind:
                        BlockKind::Text {
                            additional_params: Some(additional_params),
                        },
                    ..
                } => additional_params.get("openai_responses_part").cloned(),
                _ => None,
            })
            .collect();
        if stored_opaque {
            assert!(
                texts.is_empty(),
                "case {index}: no text appended to the opaque part: {texts:?}"
            );
            assert_eq!(
                markers,
                vec![json!({"kind":"opaque","value":unknown})],
                "case {index}"
            );
        } else {
            assert_eq!(texts, ["A"], "case {index}");
            assert!(
                markers.is_empty(),
                "case {index}: the text part gains no opaque marker"
            );
        }
    }
}

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

    let whole: super::super::Output = serde_json::from_value(item.clone()).unwrap();
    let unary = replay_opaque_choice(super::super::tests::folded_choice(vec![whole]));
    assert_eq!(unary[0]["phase"], "final_answer");
}

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

#[test]
fn repeated_reasoning_snapshots_publish_exactly_one_close() {
    let unknown = json!({"type":"future_content","list":[1]});
    let item = json!({"type":"reasoning","id":"rs_once","summary":[{"type":"summary_text","text":"known"}],"content":[unknown]});
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":2,"item":item}),
        completed_with_output(json!([item])),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let closes = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                StreamEvent::BlockEnd {
                    end: BlockClose::Reasoning { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(closes, 1);
}

#[test]
fn a_terminal_restating_known_text_as_opaque_is_refused() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_opaque","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        completed_with_output(json!([opaque_message_fixture(json!([unknown]))])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind")
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Final(_)))
    );
    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta { delta: Delta::TextMeta { additional_params }, .. }
            if additional_params.get("openai_responses_part").is_some()
    )));
}

#[test]
fn an_empty_done_after_reasoning_deltas_keeps_the_streamed_reasoning() {
    // The empty restatement neither erases the deltas nor cancels their one
    // close; the streamed reasoning still reaches the folded choice.
    let empty = json!({"type":"reasoning","id":"rs_deltas","summary":[],"content":[]});
    let (events, errors) = driven_frames(&[
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_deltas","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"}),
        json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":empty}),
        completed_with_output(json!([empty])),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    let [AssistantContent::Reasoning(reasoning)] = folded.choice.as_slice() else {
        panic!("one reasoning item: {:?}", folded.choice);
    };
    assert_eq!(reasoning.display_text(), "known");
}

#[test]
fn an_opened_reasoning_block_closes_exactly_once_whatever_its_restatements() {
    // Deltas open the block; empty restatements keep it eligible for its one
    // close, which carries the streamed text, on every flush path.
    let empty = json!({"type":"reasoning","id":"rs_deltas","summary":[],"content":[]});
    let delta = json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_deltas","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"});
    let done = json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":empty});
    let mut failed_response = sample_response(ResponseStatus::Failed);
    failed_response.error = Some(ResponseError {
        code: Some("server_error".to_string()),
        message: "boom".to_string(),
    });
    let failed = json!({"type":"response.failed","sequence_number":2,"response":failed_response});
    // Each close with whether the wire completed the item: a done was seen.
    let closes = |events: &[StreamEvent]| -> Vec<(String, bool)> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::BlockEnd {
                    id,
                    end:
                        BlockClose::Reasoning {
                            reasoning,
                            wire_sent,
                            ..
                        },
                    ..
                } => {
                    assert_eq!(*id, BlockId::wire("rs_deltas".to_owned()));
                    Some((
                        reasoning
                            .as_ref()
                            .map(|reasoning| reasoning.display_text())
                            .unwrap_or_default(),
                        *wire_sent,
                    ))
                }
                _ => None,
            })
            .collect()
    };
    for (path, frames, wire_sent) in [
        (
            "terminal",
            vec![
                delta.clone(),
                done.clone(),
                completed_with_output(json!([empty])),
            ],
            true,
        ),
        ("partial eof", vec![delta.clone(), done.clone()], true),
        ("deltas only, eof", vec![delta.clone()], false),
        (
            "error flush",
            vec![delta.clone(), done.clone(), failed.clone()],
            true,
        ),
    ] {
        let (events, _) = driven_frames(&frames);
        assert_eq!(
            closes(&events),
            [("known".to_owned(), wire_sent)],
            "{path}: {events:?}"
        );
    }
}

fn message_item(id: &str, content: serde_json::Value) -> serde_json::Value {
    json!({"type":"message","id":id,"role":"assistant","status":"completed","content":content})
}

fn folded_texts(choice: &[AssistantContent]) -> Vec<&str> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_terminal_restating_a_message_at_another_position_does_not_repeat_its_text() {
    // Envelope repair gives index-less deltas output_index 0; the terminal
    // states the same message at position 1.
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[]});
    let message = message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]));
    for id_bearing in [true, false] {
        let mut delta = json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"});
        if id_bearing {
            delta["item_id"] = json!("msg_1");
        }
        let (events, errors) =
            driven_frames(&[delta, completed_with_output(json!([reasoning, message]))]);
        assert!(errors.is_empty(), "{errors:?}");
        let folded = folded_stream_events(
            "openai",
            events,
            &sample_response(ResponseStatus::Completed),
        )
        .unwrap();
        assert_eq!(
            folded_texts(&folded.choice),
            ["Hello"],
            "id_bearing={id_bearing}"
        );
    }
}

#[test]
fn an_added_position_owns_its_id_against_later_repaired_deltas() {
    let message = message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]));
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":0,"item":message_item("msg_1", json!([]))}),
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1,"delta":"Hello"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":2,"item":message}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message
        ])),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    assert_eq!(folded_texts(&folded.choice), ["Hello"]);
}

#[test]
fn an_added_message_never_takes_over_an_id_less_slot() {
    // msg_b is announced at its own slot; its annotations must not land on
    // the text an id-less delta streamed into slot 0.
    let annotated = json!([{"type":"output_text","text":"B","annotations":[{"url":"b"}]}]);
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([]))}),
        json!({"type":"response.output_text.delta","item_id":"msg_b","output_index":1,"content_index":0,"sequence_number":2,"delta":"B"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":3,"item":message_item("msg_b", annotated.clone())}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_b", annotated)
        ])),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    assert_eq!(folded_texts(&folded.choice), ["A", "B"]);
    let AssistantContent::Text(a) = &folded.choice[0] else {
        panic!("text first: {:?}", folded.choice);
    };
    assert!(
        a.additional_params
            .as_ref()
            .and_then(|params| params.get("openai_responses"))
            .is_none(),
        "{a:?}"
    );
}

#[test]
fn an_opaque_terminal_part_elsewhere_never_marks_repaired_streamed_text() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_2", json!([unknown]))
        ])),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    let hello = folded
        .choice
        .iter()
        .find_map(|part| match part {
            AssistantContent::Text(text) if text.text == "Hello" => Some(text),
            _ => None,
        })
        .expect("the streamed text");
    assert!(
        hello
            .additional_params
            .as_ref()
            .and_then(|params| params.get("openai_responses_part"))
            .is_none(),
        "{hello:?}"
    );
    assert!(
        folded
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::Text(text)
        if text.additional_params.as_ref().and_then(|params| params.get("openai_responses_part"))
            == Some(&json!({"kind":"opaque","value":unknown}))))
    );
}

#[test]
fn a_reasoning_slot_restated_as_the_other_kind_is_refused() {
    let unknown = json!({"type":"future_summary","payload":[1]});
    let known_delta = json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":0,"delta":"known"});
    let unknown_part = |kind: &str| json!({"type":format!("response.reasoning_summary_part.{kind}"),"item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":1,"part":unknown});
    let known_done = json!({"type":"response.reasoning_summary_text.done","item_id":"rs_k","output_index":0,"summary_index":0,"sequence_number":2,"text":"known"});
    let known_snapshot = json!({"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"type":"reasoning","id":"rs_k","summary":[{"type":"summary_text","text":"known"}]}});
    let undecided = json!({"type":"response.content_part.added","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"future_content","payload":[2]}});
    let reasoning_delta = json!({"type":"response.reasoning_text.delta","item_id":"rs_k","output_index":0,"content_index":0,"sequence_number":1,"delta":"text"});
    let cases = [
        vec![known_delta.clone(), unknown_part("done")],
        vec![unknown_part("added"), known_done],
        vec![unknown_part("done"), known_snapshot],
        vec![undecided, reasoning_delta],
    ];
    for (index, frames) in cases.into_iter().enumerate() {
        let (events, errors) = driven_frames(&frames);
        assert_eq!(errors.len(), 1, "case {index}: {errors:?}");
        assert!(
            errors[0]
                .to_string()
                .contains("conflicting Responses content part kind"),
            "case {index}: {}",
            errors[0]
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::Final(_))),
            "case {index}"
        );
    }
}

#[test]
fn known_text_over_a_waiting_opaque_part_is_refused_before_binding() {
    let unknown = json!({"type":"future_part","payload":[1]});
    let (events, errors) = driven_frames(&[
        json!({"type":"response.content_part.added","item_id":"msg_w","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_w","output_index":0,"content_index":0,"sequence_number":1,"delta":"late"}),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind")
    );
    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockDelta {
            delta: Delta::Text { .. },
            ..
        }
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::BlockStart { .. }))
    );
    // The waiting part still leaves raw rather than vanishing.
    assert_eq!(unknown_payloads(&events), vec![unknown]);
}

#[test]
fn a_content_part_without_a_type_tag_is_a_malformed_frame() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.content_part.added","item_id":"msg_t","output_index":0,"content_index":0,"sequence_number":0,"part":{"text":"x"}}),
    ]);
    assert!(
        matches!(errors.as_slice(), [ProviderError::Json(_)]),
        "{errors:?}"
    );
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn responses_replay_refuses_text_beside_an_opaque_marker() {
    let mut text = super::super::text_block(super::super::AssistantContent::Unknown(
        json!({"type":"future_part","payload":[1]}),
    ));
    text.text = "replacement".into();
    for id in [Some("msg_opaque".to_owned()), None] {
        let result: Result<Vec<super::super::InputItem>, _> = crate::message::Message::Assistant {
            id: id.clone(),
            content: vec![AssistantContent::Text(text.clone())],
        }
        .try_into();
        let error = result.expect_err("text beside an opaque marker cannot replay");
        assert!(
            error
                .to_string()
                .contains("an opaque Responses message part carries non-empty text"),
            "id={id:?}: {error}"
        );
    }
}

#[test]
fn reasoning_text_content_parts_are_known_reasoning_content() {
    // OpenRouter announces reasoning content with a content part before its
    // reasoning_text deltas; it is known content, not an opaque part.
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":{"type":"reasoning","id":"rs_or","summary":[]}}),
        json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":""}}),
        json!({"type":"response.reasoning_text.delta","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":2,"delta":"think"}),
        json!({"type":"response.reasoning_text.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":3,"text":"think"}),
        json!({"type":"response.content_part.done","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":4,"part":{"type":"reasoning_text","text":"think"}}),
        completed_with_output(
            json!([{"type":"reasoning","id":"rs_or","summary":[],"content":[{"type":"reasoning_text","text":"think"}]}]),
        ),
    ]);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events,
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    let [AssistantContent::Reasoning(reasoning)] = folded.choice.as_slice() else {
        panic!("one reasoning item: {:?}", folded.choice);
    };
    assert_eq!(reasoning.display_text(), "think");
    assert!(!reasoning.has_opaque_parts());

    let (_, errors) = driven_frames(&[
        json!({"type":"response.content_part.added","item_id":"rs_or","output_index":0,"content_index":0,"sequence_number":0,"part":{"type":"reasoning_text"}}),
    ]);
    assert!(
        !errors.is_empty(),
        "a known tag with a missing text is malformed"
    );
}

#[test]
fn a_reasoning_text_part_inside_a_message_is_refused() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":message_item("msg_r", json!([]))}),
        json!({"type":"response.content_part.added","item_id":"msg_r","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":"think"}}),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses parent kind"),
        "{}",
        errors[0]
    );
    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::BlockStart {
            kind: BlockKind::Reasoning { .. },
            ..
        }
    )));
}

fn reasoning_items(choice: &[AssistantContent]) -> Vec<&crate::message::Reasoning> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect()
}

fn fold_driven(frames: &[serde_json::Value]) -> (Vec<AssistantContent>, Vec<StreamEvent>) {
    let (events, errors) = driven_frames(frames);
    assert!(errors.is_empty(), "{errors:?}");
    let folded = folded_stream_events(
        "openai",
        events.clone(),
        &sample_response(ResponseStatus::Completed),
    )
    .unwrap();
    (folded.choice, events)
}

#[test]
fn a_done_only_second_message_keeps_its_own_text() {
    // Id-less deltas fill slot 0; a different message arrives only as its
    // done at slot 1. It does not restate slot 0, so it keeps its position.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"A"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"output_text","text":"B"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_b", json!([{"type":"output_text","text":"B"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", "B"]);
}

#[test]
fn a_done_only_message_extending_streamed_text_is_a_different_message() {
    // Only an exact restatement takes over an id-less slot: "Yes, and more"
    // is not "Yes" restated.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Yes"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"output_text","text":"Yes, and more"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"Yes"}])),
            message_item(
                "msg_b",
                json!([{"type":"output_text","text":"Yes, and more"}])
            )
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["Yes", "Yes, and more"]);
}

#[test]
fn a_done_only_refusal_never_reopens_an_id_less_refusal() {
    let (choice, events) = fold_driven(&[
        json!({"type":"response.refusal.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"no"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"refusal","refusal":"other"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"refusal","refusal":"no"}])),
            message_item("msg_b", json!([{"type":"refusal","refusal":"other"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["no", "other"]);
    // Every refusal block carries its marker from its start.
    for event in &events {
        if let StreamEvent::BlockStart {
            kind: BlockKind::Text { additional_params },
            ..
        } = event
        {
            assert!(additional_params.is_some(), "{events:?}");
        }
    }
}

#[test]
fn a_content_part_done_restating_an_id_less_slot_does_not_repeat_its_text() {
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Hello","annotations":[]}}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello"]);
}

#[test]
fn a_content_part_added_never_takes_over_an_id_less_slot() {
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.added","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Hello","annotations":[]}}),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", "Hello"]);
}

#[test]
fn a_terminal_refusal_still_closes_later_reasoning() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"summary_index":0,"sequence_number":1,"delta":"think"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think"}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind at output 0"),
        "{errors:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning { .. },
                ..
            } if *id == BlockId::wire("rs_1".to_owned())
        )),
        "{events:?}"
    );
}

#[test]
fn an_empty_signature_is_no_more_content_than_empty_ciphertext() {
    let done_with = |field: &str| {
        let mut item = json!({"type":"reasoning","id":"rs_1","summary":[]});
        item[field] = json!("");
        driven_frames(&[
            json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":0,"delta":"t"}),
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":item}),
        ])
    };
    let signature = done_with("signature");
    let ciphertext = done_with("encrypted_content");
    assert!(signature.1.is_empty() && ciphertext.1.is_empty());
    assert_eq!(format!("{:?}", signature.0), format!("{:?}", ciphertext.0));
}

#[test]
fn a_waiting_part_binds_only_to_the_item_its_event_named() {
    // Repair puts msg_1's unknown part at slot 0, where the terminal's
    // reasoning sits; the part belongs to msg_1 alone.
    let unknown = json!({"type":"future_part","payload":[1]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([unknown]))
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    assert!(reasoning[0].content.is_empty(), "{choice:?}");
    let opaque: Vec<_> = choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => {
                super::super::opaque_message_part(text.additional_params.as_ref())
            }
            _ => None,
        })
        .collect();
    assert_eq!(opaque, [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn a_waiting_part_whose_item_never_arrives_leaves_raw() {
    let unknown = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]})
        ])),
    ]);
    assert!(
        reasoning_items(&choice)
            .iter()
            .all(|reasoning| reasoning.content.is_empty()),
        "{choice:?}"
    );
    assert_eq!(unknown_payloads(&events), vec![unknown]);
}

#[test]
fn distinct_waiting_observations_bind_to_reasoning_last_writer_wins() {
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.done","output_index":0,"content_index":0,"sequence_number":1,"part":second}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":2,"delta":"t"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"t"}]})
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    let opaque: Vec<_> = reasoning[0]
        .content
        .iter()
        .filter_map(|content| match content {
            ReasoningContent::OpaqueContent(value) => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(opaque, [&second], "{choice:?}");
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn reasoning_text_added_text_stays_off_the_live_deltas() {
    // As with summary parts, the text a content_part.added carries is not a
    // live reasoning delta.
    let (_, events) = fold_driven(&[
        json!({"type":"response.output_item.added","output_index":0,"sequence_number":0,"item":{"type":"reasoning","id":"rs_x","summary":[]}}),
        json!({"type":"response.content_part.added","item_id":"rs_x","output_index":0,"content_index":0,"sequence_number":1,"part":{"type":"reasoning_text","text":"x"}}),
    ]);
    assert!(
        !events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockDelta {
                delta: Delta::Reasoning { .. },
                ..
            }
        )),
        "{events:?}"
    );
}

fn opaque_texts(choice: &[AssistantContent]) -> Vec<&serde_json::Value> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => {
                super::super::opaque_message_part(text.additional_params.as_ref())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn another_items_waiting_part_never_blocks_a_known_delta() {
    // Repair puts msg_z's unknown part at msg_a's position; it can never
    // bind to msg_a, so msg_a's text is no kind contradiction.
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":1,"delta":"A"}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", ""]);
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn another_items_waiting_part_never_blocks_reasoning_text() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":1,"delta":"r"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"r"}]}),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 1, "{choice:?}");
    assert_eq!(reasoning[0].display_text(), "r");
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn a_terminal_binds_another_items_waiting_part_once_to_its_item() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_z","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"A"}])),
            message_item("msg_z", json!([unknown]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["A", ""]);
    assert_eq!(opaque_texts(&choice), [&unknown]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn a_waiting_part_of_the_same_item_still_refuses_a_contradicting_kind() {
    let unknown = json!({"type":"future_part","payload":[9]});
    let (_, errors) = driven_frames(&[
        json!({"type":"response.content_part.added","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        json!({"type":"response.output_text.delta","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":1,"delta":"A"}),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind"),
        "{errors:?}"
    );
}

#[test]
fn a_reasoning_items_waiting_part_moves_to_its_own_slot() {
    // Repair puts rs_1's unknown part at rs_0's position; it binds once to
    // rs_1 when rs_1 establishes its slot, and rs_0 stays empty.
    let unknown = json!({"type":"future_part","payload":[4]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":0,"part":unknown}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_0","summary":[]}),
            json!({"type":"reasoning","id":"rs_1","summary":[],"content":[]})
        ])),
    ]);
    let reasoning = reasoning_items(&choice);
    assert_eq!(reasoning.len(), 2, "{choice:?}");
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_0"));
    assert!(reasoning[0].content.is_empty(), "{choice:?}");
    assert_eq!(reasoning[1].id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning[1].content,
        vec![ReasoningContent::OpaqueContent(unknown.clone())]
    );
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}

#[test]
fn known_issue_an_added_position_duplicates_repaired_id_less_text() {
    // Known issue: an empty `added` binds msg_1 to position 1, so the
    // terminal's exact restatement never takes over the slot that repaired
    // id-less deltas filled, and the text appears twice. A fix must change
    // this pin deliberately.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_item.added","output_index":1,"sequence_number":0,"item":message_item("msg_1", json!([]))}),
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":1,"delta":"Hello"}),
        completed_with_output(json!([
            json!({"type":"reasoning","id":"rs_1","summary":[]}),
            message_item("msg_1", json!([{"type":"output_text","text":"Hello"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", "Hello"]);
}

#[test]
fn a_terminal_refusal_still_starts_and_closes_terminal_only_reasoning() {
    let (events, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_t","summary":[{"type":"summary_text","text":"t"}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    let rs_t = BlockId::wire("rs_t".to_owned());
    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockStart { id, .. } if *id == rs_t
        )),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning { reasoning: Some(reasoning), .. },
                ..
            } if *id == rs_t && reasoning.display_text() == "t"
        )),
        "{events:?}"
    );
}

#[test]
fn the_first_terminal_refusal_stands_over_a_later_reasoning_conflict() {
    let (_, errors) = driven_frames(&[
        json!({"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hi"}),
        json!({"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":1,"summary_index":0,"sequence_number":1,"delta":"think"}),
        completed_with_output(json!([
            message_item("msg_1", json!([{"type":"future_part","payload":[1]}])),
            json!({"type":"reasoning","id":"rs_1","summary":[{"type":"future_summary","payload":[2]}]})
        ])),
    ]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("conflicting Responses content part kind at output 0, content 0"),
        "{errors:?}"
    );
}

#[test]
fn an_empty_signature_on_an_id_less_first_done_publishes_nothing() {
    // No prior slot and no provider id: an empty signature, like empty
    // ciphertext, is no content, so the done opens no reasoning block.
    for field in ["signature", "encrypted_content"] {
        let mut item = json!({"type":"reasoning","id":"","summary":[]});
        item[field] = json!("");
        let (events, errors) = driven_frames(&[
            json!({"type":"response.output_item.done","output_index":0,"sequence_number":0,"item":item}),
        ]);
        assert!(errors.is_empty(), "{field}: {errors:?}");
        assert!(
            !events.iter().any(|event| matches!(
                event,
                StreamEvent::BlockStart { .. } | StreamEvent::BlockEnd { .. }
            )),
            "{field}: {events:?}"
        );
    }
}

#[test]
fn equal_text_of_another_kind_is_not_a_restatement() {
    // "no" streamed id-less as text, then a done-only refusal "no": not the
    // same part, so it keeps its own position.
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"no"}),
        json!({"type":"response.output_item.done","output_index":1,"sequence_number":1,"item":message_item("msg_b", json!([{"type":"refusal","refusal":"no"}]))}),
        completed_with_output(json!([
            message_item("msg_a", json!([{"type":"output_text","text":"no"}])),
            message_item("msg_b", json!([{"type":"refusal","refusal":"no"}]))
        ])),
    ]);
    assert_eq!(folded_texts(&choice), ["no", "no"]);
}

#[test]
fn a_non_matching_content_part_done_keeps_its_own_position() {
    let (choice, _) = fold_driven(&[
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"sequence_number":0,"delta":"Hello"}),
        json!({"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":1,"part":{"type":"output_text","text":"Other","annotations":[]}}),
    ]);
    assert_eq!(folded_texts(&choice), ["Hello", "Other"]);
}

#[test]
fn two_items_waiting_parts_at_one_position_each_bind_to_their_own_item() {
    let first = json!({"type":"future_part","payload":[1]});
    let second = json!({"type":"future_part","payload":[2]});
    let (choice, events) = fold_driven(&[
        json!({"type":"response.content_part.added","item_id":"msg_a","output_index":0,"content_index":0,"sequence_number":0,"part":first}),
        json!({"type":"response.content_part.added","item_id":"msg_b","output_index":0,"content_index":0,"sequence_number":1,"part":second}),
        completed_with_output(json!([
            message_item("msg_a", json!([first])),
            message_item("msg_b", json!([second]))
        ])),
    ]);
    assert_eq!(opaque_texts(&choice), [&first, &second]);
    assert!(unknown_payloads(&events).is_empty(), "{events:?}");
}
