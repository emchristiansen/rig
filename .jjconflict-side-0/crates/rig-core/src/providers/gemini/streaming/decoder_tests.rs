use super::*;
use crate::completion::{CompletionRequest, CompletionResponse, FinishReason};
use crate::providers::gemini::GeminiConfig;
use serde_json::json;

fn wire() -> super::super::completion::GenerateContent {
    GeminiConfig::new("test-key").completion("gemini-2.5-flash")
}

#[test]
fn recognized_hosted_code_stream_parts_keep_raw_without_becoming_answers() {
    for part in [
        json!({"executableCode":{"language":"PYTHON","code":"print(1)"}}),
        json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}}),
    ] {
        let body = json!({"candidates":[{"content":{"parts":[part.clone()],"role":"model"},"finishReason":"STOP"}]});
        let response = crate::test_utils::history::decode(&wire(), Mode::Streaming,
            [WireFrame::Text(body.to_string())]).expect("a genuine stream terminal survives");
        assert!(response.choice.is_empty());
        assert_eq!(response.text(), "");
        assert_eq!(response.raw["candidates"][0]["content"]["parts"][0], part);
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    }
}

#[test]
fn hosted_code_schema_keeps_required_fields_and_exact_enums() {
    for part in [
        json!({"executableCode":{"language":"RUST","code":"x"}}),
        json!({"executableCode":{"language":"PYTHON"}}),
        json!({"codeExecutionResult":{"outcome":"OK"}}),
        json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":1}}),
    ] {
        let frame = WireFrame::Text(json!({"candidates":[{"content":{"parts":[part]}}]}).to_string());
        assert!(matches!(GenerateContentDecoder::default().in_mode(Mode::Streaming).classify(frame), WireEvent::Corrupt(_)));
    }
    for outcome in ["OUTCOME_UNSPECIFIED","OUTCOME_OK","OUTCOME_FAILED","OUTCOME_DEADLINE_EXCEEDED"] {
        let frame = WireFrame::Text(json!({"candidates":[{"content":{"parts":[{"codeExecutionResult":{"outcome":outcome}}]}}]}).to_string());
        assert!(matches!(GenerateContentDecoder::default().in_mode(Mode::Streaming).classify(frame), WireEvent::Known(_)));
    }
}

fn decoded(reply: Value) -> Result<CompletionResponse, ProviderError> {
    crate::test_utils::history::decode(&wire(), Mode::Unary, [WireFrame::Text(reply.to_string())])
}

#[test]
fn tool_protocol_finish_reasons_preserve_exact_error_names_and_details() {
    for (wire_name, name, detail) in [
        ("MALFORMED_FUNCTION_CALL", "MalformedFunctionCall", "malformed function call"),
        ("UNEXPECTED_TOOL_CALL", "UnexpectedToolCall", "unexpected tool call"),
        ("MISSING_THOUGHT_SIGNATURE", "MissingThoughtSignature", "missing thought signature for tool call"),
        ("TOO_MANY_TOOL_CALLS", "TooManyToolCalls", "too many tool calls in response"),
        ("MALFORMED_RESPONSE", "MalformedResponse", "malformed response from provider"),
    ] {
        let error = decoded(json!({"candidates": [{"finishReason": wire_name, "finishMessage": detail}]}))
            .expect_err("a tool protocol failure is an error");
        assert!(matches!(error, ProviderError::Response(message)
            if message == format!("Gemini stopped with finish_reason={name}: {detail}")));
    }
    let error = decoded(json!({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}))
        .expect_err("missing detail retains the protected diagnostic");
    assert!(matches!(error, ProviderError::Response(message)
        if message == "Gemini stopped with finish_reason=MalformedFunctionCall: no finish message provided"));
}

#[test]
fn empty_whole_replies_require_a_truncating_finish_reason() {
    for reason in [None, Some("STOP"), Some("FINISH_REASON_UNSPECIFIED")] {
        let mut candidate = json!({"content": {"parts": [], "role": "model"}});
        if let Some(reason) = reason {
            candidate["finishReason"] = Value::from(reason);
        }
        let error = decoded(json!({"candidates": [candidate], "usageMetadata": {"promptTokenCount": 9}}))
            .expect_err("an empty successful whole reply is a defect");
        assert!(matches!(error, ProviderError::Response(message)
            if message == crate::message::EMPTY_RESPONSE_ERROR));
    }
    let error = decoded(json!({"candidates": [], "usageMetadata": {"promptTokenCount": 9}}))
        .expect_err("a silent whole reply is a defect");
    assert!(error.to_string().contains(crate::message::EMPTY_RESPONSE_ERROR));
    for (reason, normalized) in [("MAX_TOKENS", FinishReason::Length), ("SAFETY", FinishReason::ContentFilter)] {
        let response = decoded(json!({
            "candidates": [{"finishReason": reason, "index": 0}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 32, "totalTokenCount": 41}
        })).expect("a truncating terminal legalizes an empty choice");
        assert!(response.choice.is_empty());
        assert_eq!(response.finish_reason(), Some(normalized));
        assert_eq!(response.usage.output_tokens, Some(32));
    }
}

#[test]
fn unspecified_provider_terminal_is_retained_without_inventing_a_normalized_reason() {
    let response = decoded(json!({"candidates": [{
        "content": {"parts": [{"text": "hi"}], "role": "model"},
        "finishReason": "FINISH_REASON_UNSPECIFIED"
    }]})).expect("a real terminal can have no normalized reason");
    assert_eq!(response.text(), "hi");
    assert_eq!(response.finish_reason(), None);
}

#[test]
fn answer_parts_in_one_document_stay_distinct() {
    use crate::message::AssistantContent;
    let response = decoded(json!({"candidates": [{
        "content": {"parts": [{"text":"first"}, {"text":"second"},
            {"text":"", "thoughtSignature":"sig-9"}], "role":"model"},
        "finishReason":"STOP"
    }]})).expect("answer parts decode");
    let texts: Vec<_> = response.choice.iter().map(|part| match part {
        AssistantContent::Text(text) => (text.text.clone(), None),
        AssistantContent::CarriedText(text) => (text.text().to_owned(),
            text.signed_answer().map(|answer| answer.signature.as_str().to_owned())),
        other => panic!("answer text only: {other:?}"),
    }).collect();
    assert_eq!(texts, [
        ("first".to_owned(), None), ("second".to_owned(), None),
        (String::new(), Some("sig-9".to_owned())),
    ]);
}

#[test]
fn a_trailing_thought_signature_stays_on_the_part_that_carried_it() {
    use crate::message::AssistantContent;
    let buffered = decoded(json!({"candidates":[{"content":{"parts":[
        {"text":"289", "thoughtSignature":"signature_REDACTED_1"}
    ]}, "finishReason":"STOP"}]})).expect("whole answer decodes");
    let frames = [
        json!({"candidates":[{"content":{"parts":[{"text":"289"}]}}]}),
        json!({"candidates":[{"content":{"parts":[{"text":"", "thoughtSignature":"signature_REDACTED_1"}]}, "finishReason":"STOP"}]}),
    ].map(|frame| WireFrame::Text(frame.to_string()));
    let streamed = crate::test_utils::history::decode(&wire(), Mode::Streaming, frames).expect("stream decodes");
    assert_eq!(buffered.choice.len(), 1);
    assert_eq!(streamed.choice.len(), 2);
    assert!(matches!(&buffered.choice[0], AssistantContent::CarriedText(text)
        if text.text() == "289" && text.signed_answer().is_some_and(|answer| answer.signature.as_str() == "signature_REDACTED_1")));
    assert!(matches!(&streamed.choice[0], AssistantContent::Text(text) if text.text == "289"));
    assert!(matches!(&streamed.choice[1], AssistantContent::CarriedText(text)
        if text.text().is_empty() && text.signed_answer().is_some_and(|answer| answer.signature.as_str() == "signature_REDACTED_1")));
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
}

#[test]
fn an_inline_data_part_survives_as_a_raw_content_block() {
    use crate::message::AssistantContent;
    let response = decoded(json!({"candidates":[{"content":{"parts":[
        {"inlineData":{"data":"iVBORw0KGgoAAAANSUhEUg==", "mimeType":"image/png"}, "wire_extra":{"keep":true}}
    ]}, "finishReason":"STOP"}]})).expect("raw media is retained");
    let AssistantContent::CarriedText(part) = &response.choice[0] else {
        panic!("inspection-only carried text");
    };
    assert_eq!(part.payload.item.pointer("/inlineData/mimeType").and_then(Value::as_str), Some("image/png"));
    assert_eq!(part.payload.item.pointer("/inlineData/data").and_then(Value::as_str), Some("iVBORw0KGgoAAAANSUhEUg=="));
    assert_eq!(part.payload.item["wire_extra"], json!({"keep":true}));
    assert_eq!(part.known_text_projection(), Some(""));
}

#[test]
fn a_signed_answer_text_round_trips_and_replays_edits_on_another_model() {
    use crate::message::{AssistantContent, AssistantMessage, Message};
    use crate::wire::Wire;
    let response = decoded(json!({"candidates":[{"content":{"parts":[
        {"text":"the answer", "thoughtSignature":"c2lnbmVk", "wire_extra":"history only"}
    ]}, "finishReason":"STOP"}]})).expect("signed answer decodes");
    let message = Message::Assistant(AssistantMessage::new(response.choice.clone()));
    let json = serde_json::to_string(&message).expect("message serializes");
    let mut loaded: Message = serde_json::from_str(&json).expect("message loads");
    assert_eq!(loaded, message);
    let Message::Assistant(turn) = &mut loaded else { panic!("assistant turn"); };
    let AssistantContent::CarriedText(text) = &mut turn.content[0] else { panic!("signed text"); };
    *text.text_mut() = "edited answer".to_owned();
    let request = CompletionRequest::from(vec![Message::user("q"), loaded, Message::user("again")]);
    let wire = GeminiConfig::new("test-key").completion("gemini-2.0-flash");
    let encoded = wire.encode(request, Mode::Unary).expect("known text and signature replay");
    let body = crate::test_utils::json_body(&encoded.request);
    let part = &body["contents"][1]["parts"][0];
    assert_eq!(part["text"], "edited answer");
    assert_eq!(part["thoughtSignature"], "c2lnbmVk");
    assert_eq!(part["thought"], false);
    assert!(part.get("wire_extra").is_none());
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission;

#[test]
fn a_thought_signature_closes_its_own_carried_reasoning_part() {
    use crate::message::{AssistantContent, CarriedReasoningPart};
    let frame = |part: Value, ended: bool| {
        let mut candidate = json!({"content":{"parts":[part],"role":"model"}});
        if ended { candidate["finishReason"] = json!("STOP"); }
        WireFrame::Text(json!({"candidates":[candidate]}).to_string())
    };
    let response = crate::test_utils::history::decode(&wire(), Mode::Streaming, [
        frame(json!({"thought":true,"text":"think "}), false),
        frame(json!({"thought":true,"text":"first","thoughtSignature":"sig-1"}), false),
        frame(json!({"thought":true,"text":"second","thoughtSignature":"sig-2"}), true),
    ]).expect("signed reasoning parts survive");
    let parts: Vec<_> = response.choice.iter().map(|content| {
        let AssistantContent::CarriedReasoning(reasoning) = content else { panic!("carried reasoning: {content:?}"); };
        assert_eq!(reasoning.provider(), Some(super::super::completion::PROVIDER_NAME));
        reasoning.content.clone()
    }).collect();
    assert_eq!(parts, vec![
        vec![CarriedReasoningPart::Text { text:"think first".into(), signature:Some("sig-1".into()) }],
        vec![CarriedReasoningPart::Text { text:"second".into(), signature:Some("sig-2".into()) }],
    ]);
}

#[test]
fn signature_only_thought_replays_its_exact_empty_signature() {
    use crate::message::{AssistantContent, CarriedReasoningPart, Message};
    let response = decoded(json!({"candidates":[{"content":{"role":"model","parts":[
        {"thought":true,"text":"","thoughtSignature":""},
    ]},"finishReason":"STOP"}]})).expect("an explicit signature retains the thought");
    let AssistantContent::CarriedReasoning(reasoning) = &response.choice[0] else { panic!("carried reasoning"); };
    assert_eq!(reasoning.content, vec![CarriedReasoningPart::Text { text:String::new(), signature:Some(String::new()) }]);
    let request = CompletionRequest::from(vec![Message::user("question"), response.message().expect("a thought turn")]);
    let encoded = crate::wire::Wire::encode(&wire(), request, Mode::Unary).expect("known signature replays");
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(body["contents"][1]["parts"][0], json!({"thought":true,"text":"","thoughtSignature":""}));
}

#[test]
fn unary_hosted_code_only_requires_a_truncating_finish_reason() {
    for part in [
        json!({"executableCode":{"language":"PYTHON","code":"print(1)"}}),
        json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}}),
    ] {
        for reason in [None, Some("STOP")] {
            let mut candidate = json!({"content":{"parts":[part.clone()],"role":"model"}});
            if let Some(reason) = reason { candidate["finishReason"] = json!(reason); }
            let error = decoded(json!({"candidates":[candidate]})).expect_err("hosted code alone is no answer");
            assert!(matches!(error, ProviderError::Response(message) if message == crate::message::EMPTY_RESPONSE_ERROR));
        }
        for reason in ["MAX_TOKENS", "SAFETY"] {
            let body = json!({"candidates":[{"content":{"parts":[part.clone()],"role":"model"},"finishReason":reason}],
                "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}});
            let response = decoded(body.clone()).expect("empty truncation retains evidence");
            assert!(response.choice.is_empty());
            assert_eq!(response.usage.input_tokens, Some(3));
            assert_eq!(response.usage.output_tokens, Some(1));
            assert_eq!(response.usage.total_tokens, Some(4));
            assert_eq!(response.raw, body);
        }
    }
}

#[test]
fn unary_answer_sibling_keeps_hosted_code_only_in_raw() {
    let code = json!({"executableCode":{"language":"PYTHON","code":"print(1)"}});
    let body = json!({"candidates":[{"content":{"parts":[{"text":"answer"},code],"role":"model"},"finishReason":"STOP"}]});
    let response = decoded(body.clone()).expect("answer sibling survives");
    assert_eq!(response.text(), "answer");
    assert_eq!(response.choice.len(), 1);
    assert_eq!(response.raw, body);
}

#[test]
fn malformed_hosted_code_is_rejected_by_the_whole_document_reader() {
    let body = json!({"candidates":[{"content":{"parts":[{"text":"must not emit"},
        {"executableCode":{"language":"RUST","code":"invalid language"}}],"role":"model"},"finishReason":"STOP"}]});
    assert!(matches!(GenerateContentDecoder::default().in_mode(Mode::Unary)
        .classify(WireFrame::Text(body.to_string())), WireEvent::Corrupt(_)));
    let error = decoded(body).expect_err("known malformed code prevents sibling normalization");
    assert!(matches!(error.report().detail, Some(crate::error::ErrorDetail::CorruptFrame(_))));
}

#[test]
fn skipped_hosted_code_does_not_split_unsigned_answer_text() {
    let parts = json!([
        {"text":"A"}, {"executableCode":{"language":"PYTHON","code":"print(1)"}}, {"text":"B"},
    ]);
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(json!({"candidates":[{"content":{"role":"model","parts":parts},"finishReason":"STOP"}]}).to_string())])
            .expect("code does not close the active text run");
        assert_eq!(response.text(), "AB");
        assert_eq!(response.choice.len(), 1);
        assert_eq!(response.raw["candidates"][0]["content"]["parts"], parts);
    }
}

#[test]
fn modeled_metadata_is_read_before_any_candidate_emission() {
    let valid_candidate = json!({"content":{"parts":[{"text":"must not emit"}]},"finishReason":"STOP"});
    let invalid_candidates = [
        json!({"tokenCount":"1"}),
        json!({"avgLogprobs":"0.5"}),
        json!({"logprobsResult":{"chosenCandidates":[{"tokenId":"1"}]}}),
        json!({"citationMetadata":{"citationSources":[{"startIndex":"0"}]}}),
        json!({"safetyRatings":[{"category":"HARM_CATEGORY_SAFETY","probability":1}]}),
        json!({"content":{"role":"assistant","parts":[{"text":"x"}]}}),
        json!({"finishMessage":1}),
    ];
    for invalid in invalid_candidates {
        let body = json!({"candidates":[valid_candidate.clone(), invalid]});
        assert!(matches!(GenerateContentDecoder::default().in_mode(Mode::Streaming)
            .classify(WireFrame::Text(body.to_string())), WireEvent::Corrupt(_)));
    }
    for metadata in [
        json!({"promptTokenCount":"1"}),
        json!({"cachedContentTokenCount":1.5}),
        json!({"promptTokensDetails":[{"modality":"UNKNOWN_KIND","tokenCount":1}]}),
        json!({"trafficType":"NEW_TRAFFIC"}),
    ] {
        let body = json!({"candidates":[valid_candidate.clone()],"usageMetadata":metadata});
        assert!(matches!(GenerateContentDecoder::default().in_mode(Mode::Streaming)
            .classify(WireFrame::Text(body.to_string())), WireEvent::Corrupt(_)));
    }
}

#[test]
fn known_enum_maps_and_future_string_values_keep_their_wire_spelling() {
    for reason in [json!({"MAX_TOKENS":null}), json!("FUTURE_FINISH")] {
        let body = json!({"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":reason}]});
        let response = decoded(body.clone()).expect("the protected enum reader accepts this representation");
        let expected = if reason.is_object() { FinishReason::Length } else { FinishReason::Other("FUTURE_FINISH".into()) };
        assert_eq!(response.finish_reason(), Some(expected));
        assert_eq!(response.raw, body);
    }
    let body = json!({"candidates":[{"finishReason":{"MALFORMED_FUNCTION_CALL":null},"finishMessage":"bad"}]});
    let error = decoded(body).expect_err("a known enum map still names the same tool protocol failure");
    assert!(matches!(error, ProviderError::Response(message)
        if message == "Gemini stopped with finish_reason=MalformedFunctionCall: bad"));
    let body = json!({"promptFeedback":{"blockReason":{"SAFETY":null},"safetyRatings":[{
        "category":{"HARM_CATEGORY_HARASSMENT":null},"probability":{"HIGH":null}
    }]}});
    let error = decoded(body).expect_err("blocked prompt preserves typed enum spellings");
    assert!(error.to_string().contains("block_reason=SAFETY"));
    assert!(error.to_string().contains("HARM_CATEGORY_HARASSMENT=HIGH"));
    let frame = WireFrame::Text(json!({"candidates":[{"content":{"parts":[{"text":"answer"}]},
        "finishReason":"FUTURE_FINISH","safetyRatings":[{"category":"FUTURE_CATEGORY","probability":"FUTURE_PROBABILITY"}]}]}).to_string());
    assert!(matches!(GenerateContentDecoder::default().classify(frame), WireEvent::Known(_)));
}

#[test]
fn hosted_code_private_reader_accepts_exact_unit_enum_maps() {
    let body = json!({"candidates":[{"content":{"parts":[
        {"text":"answer"},
        {"executableCode":{"language":{"PYTHON":null},"code":"print(1)"}},
        {"codeExecutionResult":{"outcome":{"OUTCOME_OK":null},"output":"1"}}
    ]},"finishReason":"STOP"}]});
    let response = decoded(body.clone()).expect("unit maps retain the protected reader schema");
    assert_eq!(response.text(), "answer");
    assert_eq!(response.choice.len(), 1);
    assert_eq!(response.raw, body);
}

#[test]
fn the_first_recognized_part_key_selects_validation_and_projection() {
    const TEXT_FIRST: &str = r#"{"candidates":[{"content":{"parts":[{"text":"answer","executableCode":{"language":"RUST","code":"x"}}]},"finishReason":"STOP"}]}"#;
    const CODE_FIRST: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"RUST","code":"x"},"text":"answer"}]},"finishReason":"STOP"}]}"#;
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(TEXT_FIRST.to_owned())]).expect("unselected malformed code is inspection data");
        assert_eq!(response.text(), "answer");
        assert_eq!(response.choice.len(), 1);
        assert_eq!(response.raw["candidates"][0]["content"]["parts"][0]["executableCode"],
            json!({"language":"RUST","code":"x"}));
        assert!(matches!(GenerateContentDecoder::default().in_mode(mode)
            .classify(WireFrame::Text(CODE_FIRST.to_owned())), WireEvent::Corrupt(_)));
        let error = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(CODE_FIRST.to_owned())]).expect_err("the selected malformed code prevents emission");
        assert!(matches!(error.report().detail, Some(crate::error::ErrorDetail::CorruptFrame(_))));
    }
    const VALID_CODE_FIRST: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"x"},"text":"unselected"}]},"finishReason":"MAX_TOKENS"}]}"#;
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(VALID_CODE_FIRST.to_owned())]).expect("selected code permits an empty length choice");
        assert!(response.choice.is_empty());
        assert_eq!(response.raw["candidates"][0]["content"]["parts"][0]["text"], "unselected");
    }
}

#[test]
fn known_nonanswer_parts_refuse_instead_of_becoming_opaque_answers() {
    for (kind, value) in [
        ("fileData", json!({"fileUri":"gs://bucket/file.txt","mimeType":"text/plain"})),
        ("functionResponse", json!({"name":"lookup","response":{"output":"value"}})),
    ] {
        let mut part = Map::new();
        part.insert(kind.to_owned(), value);
        let body = json!({"candidates":[{"content":{"parts":[Value::Object(part)]},"finishReason":"STOP"}]});
        for mode in [Mode::Unary, Mode::Streaming] {
            let error = crate::test_utils::history::decode(&wire(), mode,
                [WireFrame::Text(body.to_string())]).expect_err("selected known nonanswer kind is a protected response error");
            assert!(matches!(error, ProviderError::Response(message)
                if message == format!("Gemini response part kind {kind} carries no assistant content rig can account for")));
        }
    }
}

#[test]
fn duplicate_text_keys_use_the_upstream_maps_last_value_in_both_modes() {
    const BODY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"first","text":"second"}]},"finishReason":"STOP"}]}"#;
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(BODY.to_owned())]).expect("duplicate names follow the upstream value map");
        assert_eq!(response.text(), "second");
        assert_eq!(response.choice.len(), 1);
        assert_eq!(response.raw["candidates"][0]["content"]["parts"][0]["text"], "second");
    }
}

#[test]
fn selected_unsigned_text_keeps_siblings_only_for_inspection_and_projects_real_requests() {
    use crate::message::{AssistantContent, CarriedPosition, CarriedTextContent, Message};
    use crate::wire::{Operation, Wire};
    const BODY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"answer","executableCode":{"language":"RUST","code":"x"},"arbitrary":{"retained":true}}]},"finishReason":"STOP"}]}"#;
    let raw: Value = serde_json::from_str(BODY).expect("literal reply");
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&wire(), mode,
            [WireFrame::Text(BODY.to_owned())]).expect("selected text decodes");
        assert_eq!(response.raw, raw);
        let message = response.message().expect("genuine decoded turn");
        let loaded: Message = serde_json::from_value(serde_json::to_value(&message)
            .expect("history serializes")).expect("history loads");
        assert_eq!(loaded, message);
        let Message::Assistant(turn) = &loaded else { panic!("assistant turn"); };
        let [AssistantContent::CarriedText(part)] = turn.content.as_slice() else {
            panic!("one selected carried text: {:?}", turn.content);
        };
        assert!(matches!(&part.content, CarriedTextContent::InspectionOnlyText(text) if text == "answer"));
        assert_eq!(part.payload.item, raw["candidates"][0]["content"]["parts"][0]);
        assert_eq!(part.payload.provider.as_deref(), Some(super::super::completion::PROVIDER_NAME));
        assert_eq!(part.payload.position, CarriedPosition::MessageContent { item_index: 0, part_index: 0 });
        assert_eq!(part.signed_answer(), None);
        assert!(response.choice[0].native_item().is_none());
        for edited in [false, true] {
            let mut message = loaded.clone();
            if edited {
                let Message::Assistant(turn) = &mut message else { panic!("assistant turn"); };
                let AssistantContent::CarriedText(part) = &mut turn.content[0] else { panic!("carried text"); };
                *part.text_mut() = "edited".into();
                assert_eq!(part.payload.item, raw["candidates"][0]["content"]["parts"][0]);
            }
            let history = vec![Message::user("before"), message, Message::user("continue")];
            let text = if edited { "edited" } else { "answer" };
            for model in ["gemini-2.5-flash", "gemini-2.0-flash"] {
                let target = GeminiConfig::new("test-key").completion(model);
                let prepared = Completion::prepare(CompletionRequest::from(history.clone()),
                    &target.describe()).expect("original carried history prepares");
                let encoded = target.encode(prepared, Mode::Unary).expect("known text encodes");
                let body = crate::test_utils::json_body(&encoded.request);
                assert_eq!(body["contents"][1]["parts"], json!([{"text":text}]));
                assert!(!body.to_string().contains("executableCode"));
                assert!(!body.to_string().contains("arbitrary"));
            }
            let target = crate::providers::anthropic::wire::AnthropicConfig::new("test-key")
                .completion(crate::providers::anthropic::completion::CLAUDE_SONNET_4_6);
            let prepared = Completion::prepare(CompletionRequest::from(history),
                &target.describe()).expect("foreign known text prepares");
            let encoded = target.encode(prepared, Mode::Unary).expect("foreign known text encodes");
            let body = crate::test_utils::json_body(&encoded.request);
            assert_eq!(body["messages"][1]["content"], json!([{"type":"text","text":text}]));
            assert!(!body.to_string().contains("executableCode"));
            assert!(!body.to_string().contains("arbitrary"));
        }
    }
}
