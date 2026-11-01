use super::super::{
    ResponsesDecoder, ResponsesEvent, ResponsesStreamOptions, classify_responses_payload,
    classify_websocket_payload, document::Response,
};
use super::{websocket_done, websocket_response};
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};
use serde_json::{Value, json};

#[test]
fn a_response_error_without_a_code_decodes_and_stays_absent() {
    let error: super::WebSocketErrorFields =
        serde_json::from_value(json!({"message": "overloaded"}))
            .expect("an error without a code decodes");
    assert_eq!(error._code, None);
    assert_eq!(error._message, "overloaded");
    assert_eq!(
        serde_json::to_value(&error).expect("the error serializes"),
        json!({"message": "overloaded"}),
    );
}

#[test]
fn a_bare_response_error_is_a_corrupt_frame_on_the_default_wire() {
    const RAW: &str = r#"{"message":"overloaded"}"#;
    let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
    for event in [
        classify_responses_payload(RAW),
        decoder.classify(WireFrame::Text(RAW.to_owned())),
    ] {
        let error =
            crate::driver::triage(event).expect_err("the bare error is not a wire response");
        let crate::error::ProviderError::CorruptFrame(corrupt) = error else {
            panic!("a wire-shape refusal retains CorruptFrame, not a provider failure");
        };
        assert_eq!(corrupt.frame(), Some(RAW));
        assert_eq!(corrupt.frame_bytes(), Some(RAW.as_bytes()));
        assert_eq!(
            corrupt.evidence(),
            &crate::error::FrameEvidence::Text(RAW.to_owned()),
        );
        assert_eq!(corrupt.event_type(), None);
    }
}

#[test]
fn sparse_done_accepts_full_field_subsets_and_keeps_unknown_evidence() {
    let optional = [
        ("object", json!("response")),
        ("created_at", json!(0)),
        ("model", json!("gpt-test")),
    ];
    for mask in 0..8 {
        let mut response = json!({
            "id":"resp_sparse", "status":"future_status",
            "reasoning":42, "parallel_tool_calls":"future metadata",
            "headers":{"x-provider-evidence":["first",null,"last"]},
            "future":{"nested":[1,null,"kept"]}
        });
        for (index, (key, value)) in optional.iter().enumerate() {
            if mask & (1 << index) != 0 {
                response[*key] = value.clone();
            }
        }
        let evidence = response.clone();
        assert!(websocket_done(&response).is_ok(), "{response}");
        assert_eq!(response, evidence);
    }
}

#[test]
fn sparse_done_requires_string_id_and_status_without_filling_absent_fields() {
    for response in [
        Value::Null,
        json!(42),
        json!([]),
        json!({}),
        json!({"id":"r"}),
        json!({"status":"completed"}),
        json!({"id":null,"status":"completed"}),
        json!({"id":"r","status":42}),
    ] {
        assert!(websocket_done(&response).is_err(), "{response}");
    }
}

#[test]
fn sparse_done_error_code_is_optional_or_null_but_still_typed() {
    for error in [
        Value::Null,
        json!({"message":"failed"}),
        json!({"code":null,"message":"failed"}),
        json!({"code":"future_error","message":"failed","future":[false,null]}),
    ] {
        let response = json!({"id":"r","status":"failed","error":error});
        assert!(websocket_done(&response).is_ok(), "{response}");
    }
    for error in [
        json!({}),
        json!({"message":null}),
        json!({"message":42}),
        json!({"code":42,"message":"failed"}),
        json!({"code":false,"message":"failed"}),
    ] {
        let response = json!({"id":"r","status":"failed","error":error});
        assert!(websocket_done(&response).is_err(), "{response}");
    }
}

#[test]
fn sparse_and_full_done_refuse_malformed_present_structural_fields() {
    for (field, value) in [
        ("object", json!(null)),
        ("object", json!("not_response")),
        ("created_at", json!(null)),
        ("created_at", json!(-1)),
        ("model", json!(null)),
        ("model", json!(42)),
        ("output", json!(null)),
        ("output", json!([{"type":"message","content":42}])),
        ("tools", json!(42)),
        ("tools", json!([{"type":"function","description":null}])),
        ("tools", json!([{"type":"function","strict":"yes"}])),
        ("usage", json!({})),
        (
            "usage",
            json!({"input_tokens":1,"output_tokens":2,"total_tokens":3,
            "input_tokens_details":{"cached_tokens":-1}}),
        ),
        (
            "usage",
            json!({"input_tokens":1,"output_tokens":2,"total_tokens":3,
            "output_tokens_details":{}}),
        ),
        ("incomplete_details", json!({"reason":42})),
        ("instructions", json!([])),
        ("max_output_tokens", json!(-1)),
    ] {
        for full in [false, true] {
            let mut response = if full {
                json!({"id":"r","status":"completed","object":"response",
                    "created_at":0,"model":"gpt-test"})
            } else {
                json!({"id":"r","status":"completed"})
            };
            response[field] = value.clone();
            assert!(websocket_done(&response).is_err(), "{response}");
        }
    }
}

#[test]
fn sparse_done_reuses_known_output_shapes_without_reinterpreting_unknown_items() {
    let valid = json!({"id":"r","status":"completed","output":[
        {"type":"message","id":"m","role":"assistant","status":"future_status",
            "content":[{"type":"output_text","text":"answer","future":null},
                {"type":"future_part","payload":[false,null]}]},
        {"type":"function_call","call_id":"c","name":"lookup","arguments":"{}",
            "status":"completed"},
        {"type":"custom_tool_call","call_id":"c2","name":"custom","input":"raw"},
        {"type":"reasoning","id":"reason","summary":[{"type":"summary_text","text":"s"}],
            "content":["first",{"type":"reasoning_text","text":"second"},
                {"type":"future_reasoning","payload":null}]},
        {"type":"future_output","payload":[1,null]}, null, 42, {"type":false}
    ],"tools":[{"type":"function","strict":null}],"usage":null});
    let evidence = valid.clone();
    assert!(websocket_done(&valid).is_ok());
    assert_eq!(valid, evidence);

    for output in [
        json!({"type":"message","id":"m","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":42}]}),
        json!({"type":"function_call","call_id":"c","name":"lookup","arguments":{},
            "status":"completed"}),
        json!({"type":"custom_tool_call","call_id":"c","name":"custom","input":42}),
        json!({"type":"reasoning","id":"reason","summary":[{"type":"summary_text","text":42}]}),
        json!({"type":"reasoning","id":"reason","content":null}),
    ] {
        let response = json!({"id":"r","status":"completed","output":[output]});
        assert!(websocket_done(&response).is_err(), "{response}");
    }
}

#[test]
fn websocket_lifecycle_requires_the_full_root_while_done_accepts_sparse_subsets() {
    let optional = [
        ("object", json!("response")),
        ("created_at", json!(0)),
        ("model", json!("gpt-test")),
    ];
    for mask in 0..8 {
        let mut response = json!({"id":"r","status":"completed"});
        for (index, (key, value)) in optional.iter().enumerate() {
            if mask & (1 << index) != 0 {
                response[*key] = value.clone();
            }
        }
        assert!(websocket_done(&response).is_ok());
        assert_eq!(
            websocket_response(&response).is_ok(),
            mask == 7,
            "{response}"
        );
    }
}

#[test]
fn websocket_lifecycle_preserves_optional_code_and_original_tool_description_contract() {
    for error in [
        json!({"message":"failed"}),
        json!({"code":null,"message":"failed"}),
        json!({"code":"future_error","message":"failed"}),
    ] {
        let response = json!({"id":"r","status":"failed","object":"response",
            "created_at":0,"model":"gpt-test","error":error,
            "reasoning":42,"future":{"raw":[null,false]}});
        let evidence = response.clone();
        assert!(websocket_response(&response).is_ok(), "{response}");
        assert_eq!(response, evidence);
    }
    let invalid_description = json!({"id":"r","status":"completed","object":"response",
        "created_at":0,"model":"gpt-test","tools":[{"type":"function","description":null}]});
    assert!(websocket_response(&invalid_description).is_err());
}

/// The strict SSE reader accepts `"description": null` on a whole response's
/// tool and keeps the body exactly. This departs from F, whose typed reader
/// refused it; the departure was accepted under mec-rig-port-answers-1 Q9.
/// A present description must still be a string or null, and the WebSocket
/// reader keeps refusing null (see the tests beside this one).
#[test]
fn the_strict_reader_accepts_a_null_tool_description() {
    let response = json!({"id":"r","object":"response","created_at":0,
        "status":"completed","model":"gpt-test","output":[],
        "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},
            "strict":false,"description":null}]});
    assert!(super::whole(&response).is_ok());
    let raw = response.to_string();
    assert!(matches!(classify_responses_payload(&raw),
        WireEvent::Known(ResponsesEvent::Whole(retained)) if retained == response));
    let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict());
    assert!(matches!(decoder.classify(WireFrame::Text(raw)),
        WireEvent::Known(ResponsesEvent::Whole(retained)) if retained == response));
    let frame = json!({"type":"response.completed","sequence_number":0,"response":response});
    assert!(matches!(
        classify_responses_payload(&frame.to_string()),
        WireEvent::Known(_)
    ));
    let mut numeric = response.clone();
    numeric["tools"][0]["description"] = json!(7);
    assert!(super::whole(&numeric).is_err());
}

#[test]
fn websocket_lifecycle_optionality_does_not_relax_the_sse_reader() {
    let response = json!({"id":"r","status":"failed","object":"response",
        "created_at":0,"model":"gpt-test","error":{"message":"failed"}});
    assert!(websocket_response(&response).is_ok());
    assert!(super::whole(&response).is_err());
}

#[test]
fn websocket_parser_decoder_and_document_accept_the_same_optional_code_lifecycle() {
    for error in [
        json!({"message":"kept"}),
        json!({"code":null,"message":"kept"}),
    ] {
        let response = json!({"id":"r","object":"response","created_at":0,
            "status":"completed","model":"gpt-test","error":error,
            "headers":{"future":[null,false]},"future":{"exact":42}});
        let frame = json!({"type":"response.completed","sequence_number":7,
            "item_id":42,"response":response});
        let raw = frame.to_string();
        assert!(matches!(classify_websocket_payload(&raw),
            WireEvent::Known(ResponsesEvent::Frame { frame: retained, raw: retained_raw, .. })
            if retained == frame && retained_raw == raw));
        let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
            .with_websocket_reader();
        assert!(matches!(decoder.classify(WireFrame::Text(raw.clone())),
            WireEvent::Known(ResponsesEvent::Frame { frame: retained, .. }) if retained == frame));
        let mut document = Response::with_websocket_reader();
        document.absorb(&WireFrame::Text(raw.clone()));
        let rebuilt = document.finish();
        assert_eq!(rebuilt["error"], response["error"]);
        assert_eq!(rebuilt["headers"], response["headers"]);
        assert_eq!(rebuilt["future"], response["future"]);
        assert_eq!(rebuilt["item_id"], json!(42));
        assert!(matches!(
            classify_responses_payload(&raw),
            WireEvent::Corrupt(_)
        ));
        let mut sse_document = Response::with_strict_reader(false);
        sse_document.absorb(&WireFrame::Text(raw));
        assert_eq!(sse_document.finish(), Value::Null);
    }
}

#[test]
fn websocket_policy_never_repairs_a_missing_or_invalid_sequence() {
    for sequence in [None, Some(Value::Null), Some(json!(-1)), Some(json!("7"))] {
        let mut frame = json!({"type":"response.completed","response":{
            "id":"r","object":"response","created_at":0,"status":"completed",
            "model":"gpt-test"}});
        if let Some(sequence) = sequence {
            frame["sequence_number"] = sequence;
        }
        let raw = frame.to_string();
        assert!(matches!(
            classify_websocket_payload(&raw),
            WireEvent::Corrupt(_)
        ));
        let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
            .with_envelope_repair()
            .with_websocket_reader()
            .with_envelope_repair();
        assert!(matches!(
            decoder.classify(WireFrame::Text(raw.clone())),
            WireEvent::Corrupt(_)
        ));
        let mut document = Response::with_websocket_reader();
        document.absorb(&WireFrame::Text(raw));
        assert_eq!(document.finish(), Value::Null);
    }
}

#[test]
fn websocket_policy_refuses_null_tool_descriptions_and_sparse_collector_bodies() {
    let sparse = json!({"id":"r","status":"completed"});
    assert!(websocket_done(&sparse).is_ok());
    let full = json!({"id":"r","object":"response","created_at":0,
        "status":"completed","model":"gpt-test","error":{"message":"kept"}});
    for response in [sparse, {
        let mut invalid = full.clone();
        invalid["tools"] = json!([{"type":"function","description":null}]);
        invalid
    }] {
        for frame in [
            response.clone(),
            json!({"type":"response.completed",
            "sequence_number":0,"response":response}),
        ] {
            let raw = frame.to_string();
            assert!(matches!(
                classify_websocket_payload(&raw),
                WireEvent::Corrupt(_)
            ));
            let decoder = ResponsesDecoder::new("openai", ResponsesStreamOptions::strict())
                .with_websocket_reader();
            assert!(matches!(
                decoder.classify(WireFrame::Text(raw.clone())),
                WireEvent::Corrupt(_)
            ));
            let mut document = Response::with_websocket_reader();
            document.absorb(&WireFrame::Text(raw));
            assert_eq!(document.finish(), Value::Null);
        }
    }
    let raw = full.to_string();
    assert!(matches!(classify_websocket_payload(&raw),
        WireEvent::Known(ResponsesEvent::Whole(retained)) if retained == full));
    let decoder =
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()).with_websocket_reader();
    assert!(matches!(decoder.classify(WireFrame::Text(raw.clone())),
        WireEvent::Known(ResponsesEvent::Whole(retained)) if retained == full));
    let mut document = Response::with_websocket_reader();
    document.absorb(&WireFrame::Text(raw));
    assert_eq!(document.finish(), full);
}

#[test]
fn websocket_policy_keeps_unknown_frames_and_strict_known_non_lifecycle_shapes() {
    let unknown = json!({"type":"future.event","sequence_number":"future shape",
        "payload":{"exact":[null,false,42]}});
    let raw = unknown.to_string();
    assert!(matches!(classify_websocket_payload(&raw),
        WireEvent::Unknown { event_type, value } if event_type == "future.event"
            && value.value() == &unknown));
    let decoder =
        ResponsesDecoder::new("openai", ResponsesStreamOptions::strict()).with_websocket_reader();
    assert!(matches!(decoder.classify(WireFrame::Text(raw)),
        WireEvent::Unknown { value, .. } if value.value() == &unknown));
    let valid = json!({"type":"response.output_text.delta","sequence_number":0,
        "output_index":0,"content_index":0,"delta":"known"});
    assert!(matches!(
        classify_websocket_payload(&valid.to_string()),
        WireEvent::Known(_)
    ));
    let mut invalid = valid;
    invalid["delta"] = json!(42);
    assert!(matches!(
        classify_websocket_payload(&invalid.to_string()),
        WireEvent::Corrupt(_)
    ));
    assert!(matches!(
        decoder.classify(WireFrame::Text(invalid.to_string())),
        WireEvent::Corrupt(_)
    ));
}
