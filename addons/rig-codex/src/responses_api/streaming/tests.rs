use super::*;
use crate::error::FrameEvidence;
use crate::responses_api::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES;
use serde_json::json;

/// An event of a modelled type is never read as anything else: a whole
/// response wearing a known event's `type` is corrupt, and a repeated
/// discriminator does not rescue it.
#[test]
fn known_event_cannot_fall_back_to_a_whole_response() {
    let mut body = json!({
        "id":"resp_1", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-test"
    });
    body["type"] = json!("response.completed");
    assert!(decode_frame(&body.to_string()).is_err());
    // Even a duplicate discriminator must not make body decoding rescue the
    // event's invalid envelope.
    let duplicate = format!("{{\"type\":\"future.event\",{}", &body.to_string()[1..]);
    assert!(decode_frame(&duplicate).is_err());
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
        decode_frame(&frame),
        Ok(DecodedFrame::Chunk(StreamingCompletionChunk::Delta(_)))
    ));
}

#[test]
fn classify_unknown_event_type_is_unknown() {
    let value = json!({
        "type": "response.web_search_call.searching",
        "output_index": 0,
        "sequence_number": 1,
    });
    let DecodedFrame::Unknown {
        event_type,
        payload,
    } = decode_frame(&value.to_string()).expect("an unmodelled event is not corrupt")
    else {
        panic!("an unmodelled event is unknown");
    };
    assert_eq!(event_type, "response.web_search_call.searching");
    assert_eq!(payload.value(), &value, "the whole payload is kept");
}

/// `response.reasoning_text.done` terminates every raw-reasoning block. The
/// tag must be known (no `Unknown`), and `ItemChunkKind` must carry a variant
/// for it (no corrupt frame, which is what naming the tag without the variant
/// would produce).
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

    let event = decode_frame(&frame);
    assert!(
        !matches!(event, Ok(DecodedFrame::Unknown { .. })),
        "the tag must be in the known-event set: {event:?}"
    );
    assert!(
        event.is_ok(),
        "a known tag with no matching ItemChunkKind variant decodes to a corrupt frame: {event:?}"
    );
    assert!(matches!(
        event,
        Ok(DecodedFrame::Chunk(StreamingCompletionChunk::Delta(chunk)))
            if matches!(chunk.data, ItemChunkKind::ReasoningTextDone(_))
    ));
}

#[test]
fn classify_invalid_json_is_corrupt() {
    let corrupt = decode_frame("{not json").expect_err("not JSON");
    assert_eq!(corrupt.event_type(), None);
    assert_eq!(corrupt.frame(), Some("{not json"));
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
    let corrupt = decode_frame(&frame).expect_err("a defective known event");
    assert_eq!(corrupt.event_type(), Some("response.output_text.delta"));
    assert_eq!(corrupt.frame(), Some(frame.as_str()));
}

// A known part tag whose payload is schema-defective must be corrupt, not
// slide into the unknown-part catch-all.
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
    assert!(decode_frame(&frame).is_err());
}

/// Each malformed frame of a modelled type is refused with its `type` and the
/// frame exactly as received: the irregular spacing and key order prove it
/// was not re-serialized.
#[test]
fn a_malformed_known_frame_is_corrupt_and_carries_the_frame_as_received() {
    for malformed in MALFORMED_KNOWN_FRAMES {
        let corrupt = decode_frame(malformed.frame)
            .expect_err(&format!("{}: the frame must not decode", malformed.case));
        assert_eq!(
            corrupt.event_type(),
            Some(malformed.event_type),
            "{}",
            malformed.case
        );
        assert_eq!(corrupt.frame(), Some(malformed.frame), "{}", malformed.case);
        assert_eq!(
            corrupt.evidence(),
            &FrameEvidence::Text(malformed.frame.to_owned()),
            "{}",
            malformed.case
        );
    }
}

/// A content part without a `type` tag is a malformed frame, not an unknown
/// part.
#[test]
fn a_content_part_without_a_type_tag_is_a_malformed_frame() {
    let frame = json!({"type":"response.content_part.added","item_id":"msg_t","output_index":0,"content_index":0,"sequence_number":0,"part":{"text":"x"}}).to_string();
    let corrupt = decode_frame(&frame).expect_err("a part without a type");
    assert_eq!(corrupt.event_type(), Some("response.content_part.added"));
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
// skippable unknown part.
#[test]
fn content_part_non_string_type_errors() {
    let result = serde_json::from_value::<ContentPartChunkPart>(json!({"type": 42, "text": "hi"}));
    assert!(result.is_err());
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": null, "text": "hi"}));
    assert!(result.is_err());
}

// `serde_json::Value` keeps the last duplicate key, so the hand dispatch
// resolves on the last `type`, unlike a derived internally-tagged enum, which
// takes the first.
#[test]
fn content_part_duplicate_type_key_dispatches_on_the_last_occurrence() {
    let part: ContentPartChunkPart =
        serde_json::from_str(r#"{"type":"bogus","type":"output_text","text":"hi"}"#).unwrap();
    assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
}

// A part tag outside the modelled set stays a skippable no-op, round-tripping
// the value verbatim.
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
            "text": "step 1"
        }
    }))
    .expect("reasoning summary part event should deserialize");

    assert!(matches!(
        chunk,
        StreamingCompletionChunk::Delta(chunk)
            if matches!(
                chunk.data,
                ItemChunkKind::ReasoningSummaryPartDone(_)
            )
    ));
}

fn completed_event() -> serde_json::Value {
    json!({
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.4",
            "reasoning": {"context": "all_turns", "effort": "high"},
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 12
            },
            "output": [{
                "type": "message",
                "id": "msg_9",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": "done"}]
            }],
            "tools": []
        }
    })
}

/// A frame Rig hands over as JSON reads into the same typed value its text
/// does, and an unmodelled one keeps its whole value.
#[test]
fn a_json_valued_frame_reads_into_its_typed_chunk() {
    let event = completed_event();
    let DecodedFrame::Chunk(StreamingCompletionChunk::Response(chunk)) =
        DecodedFrame::try_from(&event).expect("a modelled event decodes")
    else {
        panic!("a lifecycle event is a response chunk");
    };
    assert!(matches!(chunk.kind, ResponseChunkKind::ResponseCompleted));
    assert_eq!(chunk.response.id, "resp_1");
    assert_eq!(chunk.sequence_number, 9);

    let unknown =
        json!({"type": "codex.rate_limits", "primary": {"used_percent": 1.5}, "extra": [1, null]});
    let DecodedFrame::Unknown {
        event_type,
        payload,
    } = DecodedFrame::try_from(&unknown).expect("an unmodelled event is not corrupt")
    else {
        panic!("an unmodelled event is unknown");
    };
    assert_eq!(event_type, "codex.rate_limits");
    assert_eq!(payload.value(), &unknown);
}

/// A JSON-valued frame that fails to decode says so without claiming a frame
/// text it never had: the caller still holds the value.
#[test]
fn a_corrupt_json_valued_frame_carries_no_invented_frame_text() {
    let malformed = json!({"type": "response.output_text.delta", "delta": 42});
    let corrupt = DecodedFrame::try_from(&malformed).expect_err("a defective known event");
    assert_eq!(corrupt.event_type(), Some("response.output_text.delta"));
    assert_eq!(corrupt.evidence(), &FrameEvidence::Absent);
    assert_eq!(corrupt.frame(), None);
}

/// The terminal record is read off the response a terminal event carries.
#[test]
fn the_terminal_record_is_the_terminal_responses_own() {
    let DecodedFrame::Chunk(StreamingCompletionChunk::Response(chunk)) =
        DecodedFrame::try_from(&completed_event()).expect("a modelled event decodes")
    else {
        panic!("a lifecycle event is a response chunk");
    };
    let record = StreamingCompletionResponse::from(&chunk.response);
    assert_eq!(record.response_id.as_deref(), Some("resp_1"));
    assert_eq!(record.model.as_deref(), Some("gpt-5.4"));
    assert_eq!(record.message_id.as_deref(), Some("msg_9"));
    assert!(matches!(record.status, Some(ResponseStatus::Completed)));
    assert_eq!(record.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(
        record.reasoning_metadata.as_ref(),
        json!({"context": "all_turns", "effort": "high"}).as_object()
    );
    let usage = record.usage.expect("the terminal carried usage");
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.total_tokens, 12);
    assert_eq!(
        usage
            .output_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens),
        Some(3)
    );
    assert!(record.incomplete_details.is_none());
    assert!(record.provider_request_id.is_none());
}

/// A summary delta carries `delta` and a summary done carries `text`: each
/// event refuses the other's member.
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
        decode_frame(&delta_with_text.to_string()).is_err(),
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
        decode_frame(&done_with_delta.to_string()).is_err(),
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
    let Ok(DecodedFrame::Chunk(StreamingCompletionChunk::Delta(ItemChunk {
        data: ItemChunkKind::ReasoningSummaryTextDone(chunk),
        ..
    }))) = decode_frame(&done.to_string())
    else {
        panic!("a well-formed done event decodes");
    };
    assert_eq!(chunk.text, "the whole summary");
}

/// The arguments an args-done event states and the arguments its done item
/// states are one type, and reconcile to one value however each was spaced.
#[test]
fn arguments_done_and_the_done_item_reconcile_through_one_type() {
    let args_done = json!({
        "type": "response.function_call_arguments.done",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "arguments": "{\"a\": 1}",
    });
    let Ok(DecodedFrame::Chunk(StreamingCompletionChunk::Delta(ItemChunk {
        data: ItemChunkKind::FunctionCallArgsDone(chunk),
        ..
    }))) = decode_frame(&args_done.to_string())
    else {
        panic!("the args-done event decodes");
    };
    let item: crate::responses_api::OutputFunctionCall = serde_json::from_value(json!({
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
