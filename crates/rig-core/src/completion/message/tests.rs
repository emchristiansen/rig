use serde::{Deserialize, Serialize};

use super::{Message, Text, ToolResultContent};

mod vec_content_serde {
    use super::super::{AssistantContent, Message, UserContent};

    #[test]
    fn message_content_still_serializes_as_a_plain_sequence() {
        // The removed container serialized as a bare sequence, which is why
        // this migration changes no persisted history and no recorded
        // provider fixture. Pin the wire shape so that stays true.
        let message = Message::User {
            content: vec![UserContent::text("hi")],
        };
        let json = serde_json::to_value(&message).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            })
        );
    }

    #[test]
    fn message_content_round_trips_byte_identically() {
        let message = Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::text("hello"),
        ]));
        let encoded = serde_json::to_string(&message).expect("serialize");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(
            serde_json::to_string(&decoded).expect("re-serialize"),
            encoded
        );
    }

    #[test]
    fn an_empty_content_array_deserializes() {
        // The request boundary rejects it when it is sent, not serde.
        for message in [
            serde_json::json!({"role": "user", "content": []}),
            serde_json::json!({"role": "assistant", "content": []}),
        ] {
            let decoded = serde_json::from_value::<Message>(message.clone()).expect("parses");
            assert_eq!(serde_json::to_value(&decoded).expect("serializes"), message);
        }
    }
}

#[test]
fn system_message_constructor_and_serde_roundtrip() {
    let message = Message::system("You are concise.");

    match &message {
        Message::System { content } => assert_eq!(content, "You are concise."),
        _ => panic!("Expected system message"),
    }

    let json = serde_json::to_string(&message).expect("serialize");
    let roundtrip: Message = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtrip, message);
}

#[test]
fn current_schema_tool_call_json_round_trips_without_provider_promotion() {
    // A call without a provider ID remains local through the round trip.
    let call = super::ToolCall::from_wire(
        "",
        super::ToolFunction::new(
            super::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    assert!(call.id.is_local());

    let json = serde_json::to_value(&call).expect("serialize");
    assert!(json.get("call_id").is_none());
    let roundtrip: super::ToolCall = serde_json::from_value(json).expect("deserialize");
    assert!(roundtrip.id.provider().is_none());
    assert_eq!(roundtrip, call);
}

#[test]
fn legacy_call_id_key_cannot_recover_an_untagged_identity() {
    let legacy = serde_json::json!({
        "id": "fc_123",
        "call_id": "call_abc",
        "function": {"name": "add", "arguments": {"x": 1}},
    });
    assert!(serde_json::from_value::<super::ToolCall>(legacy).is_err());
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ExecutorLikeResponse {
    output: serde_json::Value,
    logs: Vec<String>,
    execution_time_ms: u64,
}

#[test]
fn tool_result_content_decodes_structured_and_legacy_json() {
    let response = ExecutorLikeResponse {
        output: serde_json::json!({"answer": 42}),
        logs: vec!["computed".to_string()],
        execution_time_ms: 7,
    };
    let value = serde_json::to_value(&response).expect("serialize response");

    let structured = ToolResultContent::json(value.clone());
    assert_eq!(structured.as_json(), Some(&value));
    assert_eq!(structured.as_text(), None);
    assert_eq!(
        structured
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode structured response"),
        response
    );

    let legacy_json = value.to_string();
    let legacy_text = ToolResultContent::Text(Text::new(legacy_json.clone()));
    assert_eq!(legacy_text.as_text(), Some(legacy_json.as_str()));
    assert_eq!(legacy_text.as_json(), None);
    assert_eq!(
        legacy_text
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode legacy response"),
        response
    );

    let image = ToolResultContent::image_url("https://example.com/result.png", None, None);
    let image_error = image.deserialize_json::<ExecutorLikeResponse>();
    assert!(image_error.is_err());
    if let Err(error) = image_error {
        assert_eq!(
            error.to_string(),
            "cannot decode image tool-result content as JSON"
        );
    }
}

/// The shapes persisted before calls had one id no longer parse.
#[test]
fn the_legacy_tool_call_id_shape_does_not_parse() {
    let legacy = serde_json::json!({
        "id": {"origin": "explicit", "id": "call_1"},
        "function": {"name": "add", "arguments": {}},
    });
    assert!(serde_json::from_value::<super::ToolCall>(legacy).is_err());
    assert!(serde_json::from_str::<super::CallId>(r#""call_1""#).is_err());
}

#[test]
fn a_stored_call_with_any_arguments_shape_loads() {
    let stored = serde_json::json!({"name": "search", "arguments": null});
    let function: super::ToolFunction =
        serde_json::from_value(stored).expect("a stored call loads");
    assert_eq!(function.arguments_value(), serde_json::json!({}));
    let stored = serde_json::json!({"name": "search", "arguments": "{\"q\":1}"});
    let function: super::ToolFunction =
        serde_json::from_value(stored).expect("a stored call loads");
    assert_eq!(function.arguments_value(), serde_json::json!({"q": 1}));
}

#[test]
fn an_unanswered_turn_fails_its_run_only_when_cut_or_failed() {
    use super::{AssistantContent, StopReason, turn_failure};
    use crate::completion::FinishReason;
    let empty: Vec<AssistantContent> = vec![AssistantContent::reasoning("thinking")];
    let answered = vec![AssistantContent::text("hi")];
    let failed = StopReason::Error("refused".into());
    assert_eq!(
        turn_failure(
            &empty,
            Some(&StopReason::Length),
            Some(&FinishReason::Length)
        ),
        Some(FinishReason::Length.no_answer_message())
    );
    assert!(turn_failure(&empty, Some(&failed), None).is_some_and(|m| m.contains("refused")));
    assert_eq!(
        turn_failure(&empty, Some(&StopReason::Stop), Some(&FinishReason::Stop)),
        None
    );
    assert_eq!(
        turn_failure(
            &answered,
            Some(&StopReason::Length),
            Some(&FinishReason::Length)
        ),
        None
    );
}

/// The STOP policy applies only to nonempty choices of exactly empty
/// inspection-media carriers, independently of their preserved payloads.
#[test]
fn only_empty_inspection_media_with_matching_stop_fails_its_run() {
    use super::{
        AssistantContent, CarriedPayload, CarriedText, CarriedTextContent, DocumentSourceKind,
        Image, ImageMediaType, StopReason, turn_failure,
    };
    use crate::completion::FinishReason;

    let carried = |content| {
        AssistantContent::CarriedText(CarriedText::new(
            content,
            CarriedPayload::new(
                serde_json::json!({"thought": true}),
                Some("vertexai".into()),
            ),
        ))
    };
    let media = carried(CarriedTextContent::InspectionOnlyMedia(String::new()));
    let message = "the provider completed a turn containing only inspection media and no answer";
    for choice in [vec![media.clone()], vec![media.clone(), media.clone()]] {
        assert_eq!(
            turn_failure(&choice, Some(&StopReason::Stop), Some(&FinishReason::Stop)),
            Some(message.to_owned()),
        );
    }
    let image = AssistantContent::Image(Image {
        data: DocumentSourceKind::Base64("AQID".into()),
        media_type: Some(ImageMediaType::PNG),
        detail: None,
        native: None,
    });
    for choice in [
        vec![],
        vec![AssistantContent::reasoning("thinking")],
        vec![AssistantContent::text("")],
        vec![carried(CarriedTextContent::InspectionOnlyText(
            String::new(),
        ))],
        vec![carried(CarriedTextContent::Opaque(String::new()))],
        vec![carried(CarriedTextContent::InspectionOnlyMedia(" ".into()))],
        vec![carried(CarriedTextContent::InspectionOnlyMedia(
            "edited".into(),
        ))],
        vec![media.clone(), AssistantContent::reasoning("thinking")],
        vec![media.clone(), AssistantContent::text("answer")],
        vec![media.clone(), image],
    ] {
        assert_eq!(
            turn_failure(&choice, Some(&StopReason::Stop), Some(&FinishReason::Stop)),
            None,
            "{choice:?}",
        );
    }
    for (stop, finish) in [
        (None, None),
        (Some(StopReason::Stop), None),
        (None, Some(FinishReason::Stop)),
        (Some(StopReason::Length), Some(FinishReason::Stop)),
        (Some(StopReason::Stop), Some(FinishReason::ToolCalls)),
        (Some(StopReason::ToolUse), Some(FinishReason::ToolCalls)),
        (
            Some(StopReason::Aborted("caller stopped".into())),
            Some(FinishReason::Stop),
        ),
    ] {
        assert_eq!(
            turn_failure(&[media.clone()], stop.as_ref(), finish.as_ref()),
            None
        );
    }
    assert_eq!(
        turn_failure(
            &[media.clone()],
            Some(&StopReason::Stop),
            Some(&FinishReason::Length)
        ),
        Some(FinishReason::Length.no_answer_message()),
    );
    assert_eq!(
        turn_failure(
            &[media.clone()],
            Some(&StopReason::Length),
            Some(&FinishReason::Length)
        ),
        Some(FinishReason::Length.no_answer_message()),
    );
    assert_eq!(
        turn_failure(
            &[media],
            Some(&StopReason::Error("failed".into())),
            Some(&FinishReason::Stop)
        ),
        Some("the provider failed the turn without an answer: failed".to_owned()),
    );
}

/// A turn the provider failed fails its run even when it holds an answer:
/// replay leaves it out, so the caller must not take it as a success.
#[test]
fn a_failed_turn_with_an_answer_fails_its_run() {
    use super::{AssistantContent, StopReason, turn_failure};
    use crate::completion::FinishReason;
    let answered = vec![AssistantContent::text("hi")];
    let failed = StopReason::Error("Provider finish_reason: weird".into());
    let message = turn_failure(
        &answered,
        Some(&failed),
        Some(&FinishReason::Other("weird".into())),
    )
    .expect("the run fails");
    assert_eq!(
        message,
        "the provider failed the turn: Provider finish_reason: weird"
    );
    let filtered = StopReason::Error("Provider finish_reason: content_filter".into());
    assert!(
        turn_failure(
            &answered,
            Some(&filtered),
            Some(&FinishReason::ContentFilter)
        )
        .is_some_and(|message| message.contains("content_filter"))
    );
    let aborted = StopReason::Aborted("the caller stopped".into());
    assert_eq!(turn_failure(&answered, Some(&aborted), None), None);
}

/// A turn that will not replay runs none of its tool calls, whatever else
/// it holds; a call in a turn the token limit ended is finished and runs.
#[test]
fn a_failed_turn_with_calls_fails_its_run() {
    use super::{AssistantContent, StopReason, ToolName, turn_failure};
    use crate::completion::FinishReason;
    let call = AssistantContent::tool_call(
        "call_1",
        ToolName::new("lookup").expect("a tool name"),
        serde_json::json!({}),
    );
    let turn = vec![AssistantContent::text("checking"), call];
    for stop in [
        StopReason::Error("Provider finish_reason: pause".into()),
        StopReason::Aborted("the stream ended".into()),
    ] {
        let message = turn_failure(&turn, Some(&stop), None).expect("the run fails");
        assert!(message.contains("none of its tool calls ran"), "{message}");
    }
    for (stop, finish) in [
        (StopReason::Length, FinishReason::Length),
        (StopReason::ToolUse, FinishReason::ToolCalls),
    ] {
        assert_eq!(turn_failure(&turn, Some(&stop), Some(&finish)), None);
    }
}

/// Redacted reasoning without a current provider item has nothing to send
/// (round-5 F7); with one, it replays.
#[test]
fn redacted_reasoning_without_its_item_is_blank() {
    use super::{AssistantContent, Reasoning};
    let redacted = AssistantContent::Reasoning(Reasoning {
        text: "kept".to_owned(),
        redacted: true,
        native: None,
    });
    assert!(redacted.is_blank());
    let with_item =
        redacted.with_native(serde_json::json!({"type": "redacted_thinking", "data": "x"}));
    assert!(!with_item.is_blank());
}

/// A document's text and its file keep the JSON they have always had: a
/// file is its source's spelling, text is `text`, and the old `string`
/// spelling loads as text for a document and as base64 data for media.
#[test]
fn document_data_keeps_its_json_and_reads_the_string_spelling() {
    use super::{DocumentData, DocumentSourceKind};
    use serde_json::json;

    let file = DocumentData::File(DocumentSourceKind::Base64("JVBERi0=".into()));
    let spelled = json!({"type": "base64", "value": "JVBERi0="});
    assert_eq!(serde_json::to_value(&file).ok(), Some(spelled.clone()));
    assert_eq!(
        serde_json::from_value::<DocumentData>(spelled).ok(),
        Some(file)
    );
    for source in [
        DocumentSourceKind::Url("https://example.com/a.pdf".into()),
        DocumentSourceKind::FileId("file_1".into()),
        DocumentSourceKind::Raw(vec![1, 2]),
        DocumentSourceKind::Unknown,
    ] {
        let as_source = serde_json::to_value(&source).ok();
        let data = DocumentData::from(source);
        let as_data = serde_json::to_value(&data).ok();
        assert_eq!(as_data, as_source);
        let back = as_data.and_then(|value| serde_json::from_value::<DocumentData>(value).ok());
        assert_eq!(back, Some(data));
    }

    let text = DocumentData::Text("notes".into());
    let spelled = json!({"type": "text", "value": "notes"});
    assert_eq!(serde_json::to_value(&text).ok(), Some(spelled.clone()));
    assert_eq!(
        serde_json::from_value::<DocumentData>(spelled).ok(),
        Some(text.clone())
    );

    let legacy = json!({"type": "string", "value": "notes"});
    assert_eq!(
        serde_json::from_value::<DocumentData>(legacy.clone()).ok(),
        Some(text)
    );
    assert_eq!(
        serde_json::from_value::<DocumentSourceKind>(legacy).ok(),
        Some(DocumentSourceKind::Base64("notes".into()))
    );
    assert_eq!(
        DocumentData::default(),
        DocumentData::File(DocumentSourceKind::Unknown)
    );
}

/// Namespaces, custom calls, and the call kind a result answers — the typed
/// tool-call surface — pinned as serialized bytes, so serde drift in the
/// canonical format is a visible test failure.
mod typed_tool_calls {
    use super::super::{
        AnsweredToolCall, AssistantContent, CustomToolCall, ToolCall, ToolFunction, ToolResult,
        ToolResultContent, UndispatchableToolCall, UnrepresentableToolCall, UserContent,
        json_only_wire_tool_call, name_dispatchable_call,
    };

    const FUNCTION_CALL: &str = r#"{"type":"toolcall","id":{"provider":"call_1"},"function":{"name":"add","arguments":{"x":1}},"native":{"item":{"call_id":"call_1","id":"fc_1"},"fingerprint":"5e62f7d16726783d"}}"#;
    const NAMESPACED_CALL: &str = r#"{"type":"toolcall","id":{"provider":"call_1"},"function":{"name":"add","namespace":"math","arguments":{"x":1}},"native":{"item":{"call_id":"call_1","id":"fc_1"},"fingerprint":"13a06c1314125750"}}"#;
    const CUSTOM_CALL: &str = r#"{"type":"customtoolcall","id":{"provider":"call_2"},"name":"apply_patch","namespace":"repo","input":"*** Begin Patch\n{\"x\": 1}","native":{"item":{"call_id":"call_2","id":"ctc_2"},"fingerprint":"02487528e677fca4"}}"#;
    const FUNCTION_RESULT: &str = r#"{"type":"toolresult","call":{"provider":"call_1"},"name":"add","answers":"function","content":[{"type":"text","text":"2"}]}"#;
    const CUSTOM_RESULT: &str = r#"{"type":"toolresult","call":{"provider":"call_2"},"name":"apply_patch","answers":"custom","content":[{"type":"text","text":"applied"}]}"#;

    fn function_call() -> AssistantContent {
        AssistantContent::tool_call(
            "call_1",
            super::super::ToolName::new("add").expect("tool name"),
            serde_json::json!({"x": 1}),
        )
        .with_native(serde_json::json!({"call_id":"call_1","id":"fc_1"}))
    }

    fn namespaced_call() -> AssistantContent {
        AssistantContent::ToolCall(
            ToolCall::from_wire(
                "call_1",
                ToolFunction::new(
                    super::super::ToolName::new("add").expect("tool name"),
                    serde_json::json!({"x": 1}),
                ),
            )
            .with_namespace(Some("math".to_owned())),
        )
        .with_native(serde_json::json!({"call_id":"call_1","id":"fc_1"}))
    }

    /// Input that happens to be valid JSON after a prefix line: it must stay
    /// the verbatim string it arrived as.
    fn custom_call() -> CustomToolCall {
        let content = AssistantContent::custom_tool_call(
            "call_2",
            super::super::ToolName::new("apply_patch").expect("tool name"),
            Some("repo".to_owned()),
            "*** Begin Patch\n{\"x\": 1}",
        )
        .with_native(serde_json::json!({"call_id":"call_2","id":"ctc_2"}));
        let AssistantContent::CustomToolCall(call) = content else {
            unreachable!()
        };
        call
    }

    fn assert_pinned<T>(value: &T, pinned: &str)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        assert_eq!(serde_json::to_string(value).expect("serialize"), pinned);
        let decoded: T = serde_json::from_str(pinned).expect("the pinned bytes decode");
        assert_eq!(&decoded, value, "the pinned bytes decode to the same value");
    }

    #[test]
    fn a_function_call_keeps_the_upstream_bytes() {
        assert_pinned(&function_call(), FUNCTION_CALL);
    }

    #[test]
    fn a_namespaced_call_carries_its_namespace_beside_the_name() {
        assert_pinned(&namespaced_call(), NAMESPACED_CALL);
    }

    #[test]
    fn a_custom_call_is_its_own_variant_with_verbatim_input() {
        let call = AssistantContent::CustomToolCall(custom_call());
        assert_pinned(&call, CUSTOM_CALL);
        let AssistantContent::CustomToolCall(decoded) =
            serde_json::from_str::<AssistantContent>(CUSTOM_CALL).expect("decode")
        else {
            panic!("a custom call decodes as the custom variant");
        };
        assert_eq!(decoded.input, "*** Begin Patch\n{\"x\": 1}");
    }

    #[test]
    fn a_custom_call_whose_input_is_json_is_still_a_custom_call() {
        let call = AssistantContent::custom_tool_call(
            "call_3",
            super::super::ToolName::new("t").expect("tool name"),
            None,
            r#"{"a":1}"#,
        );
        let json = serde_json::to_value(&call).expect("serialize");
        assert_eq!(json["type"], "customtoolcall");
        assert_eq!(
            json["input"],
            serde_json::json!(r#"{"a":1}"#),
            "a string, not an object"
        );
        let decoded: AssistantContent = serde_json::from_value(json).expect("decode");
        assert_eq!(decoded, call);
    }

    #[test]
    fn results_state_the_kind_they_answer() {
        let AssistantContent::ToolCall(call) = function_call() else {
            unreachable!()
        };
        let function = UserContent::ToolResult(call.result(vec![ToolResultContent::text("2")]));
        assert_pinned(&function, FUNCTION_RESULT);

        let custom =
            UserContent::ToolResult(custom_call().result(vec![ToolResultContent::text("applied")]));
        assert_pinned(&custom, CUSTOM_RESULT);
        assert_eq!(
            custom,
            UserContent::ToolResult(ToolResult {
                call: super::super::CallId::from_wire("call_2"),
                name: super::super::ToolName::new("apply_patch").expect("tool name"),
                answers: AnsweredToolCall::Custom,
                content: vec![ToolResultContent::text("applied")],
                is_error: false,
            }),
            "the relay constructor states what the call-derived one derives"
        );
    }

    #[test]
    fn a_result_that_does_not_say_what_it_answers_is_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(FUNCTION_RESULT).expect("json");
        json.as_object_mut()
            .expect("object")
            .shift_remove("answers");
        let error =
            serde_json::from_value::<UserContent>(json.clone()).expect_err("`answers` is required");
        assert!(error.to_string().contains("answers"), "{error}");
        json.as_object_mut().expect("object").shift_remove("type");
        assert!(serde_json::from_value::<ToolResult>(json).is_err());
    }

    #[test]
    fn the_answered_kind_derives_from_the_held_call() {
        let AssistantContent::ToolCall(call) = function_call() else {
            unreachable!()
        };
        let UserContent::ToolResult(result) =
            UserContent::ToolResult(call.result(vec![ToolResultContent::text("2")]))
        else {
            unreachable!()
        };
        assert_eq!(result.answers, AnsweredToolCall::Function);
        assert_eq!(result.call, call.id);
        assert_eq!(custom_call().answered_by(), AnsweredToolCall::Custom);
    }

    #[test]
    fn an_absent_namespace_is_omitted_and_every_present_spelling_is_kept() {
        let bare = ToolFunction::new(
            super::super::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        );
        let json = serde_json::to_value(&bare).expect("serialize");
        assert!(json.get("namespace").is_none(), "{json}");
        for spelling in ["", "functions", "math"] {
            let qualified = bare.clone().with_namespace(Some(spelling.to_owned()));
            let json = serde_json::to_value(&qualified).expect("serialize");
            assert_eq!(
                json["namespace"], spelling,
                "no spelling normalizes to absence"
            );
            assert_eq!(
                serde_json::from_value::<ToolFunction>(json).expect("decode"),
                qualified
            );
        }
    }

    #[test]
    fn json_only_wires_refuse_custom_and_namespaced_calls_by_name() {
        let AssistantContent::ToolCall(plain) = function_call() else {
            unreachable!()
        };
        assert_eq!(
            json_only_wire_tool_call("Test Wire", &function_call()),
            Ok(Some(&plain))
        );
        assert_eq!(
            json_only_wire_tool_call("Test Wire", &AssistantContent::text("hi")),
            Ok(None)
        );
        // `""` and `"functions"` are refused too: no JSON-only wire here has
        // evidence that it treats either as its default namespace.
        for spelling in ["", "functions", "math"] {
            let call =
                AssistantContent::ToolCall(plain.clone().with_namespace(Some(spelling.to_owned())));
            assert_eq!(
                json_only_wire_tool_call("Test Wire", &call),
                Err(UnrepresentableToolCall::Namespace {
                    wire: "Test Wire",
                    name: super::super::ToolName::new("add").expect("tool name"),
                    namespace: spelling.to_owned(),
                })
            );
        }
        let refusal = json_only_wire_tool_call(
            "Test Wire",
            &AssistantContent::CustomToolCall(custom_call()),
        )
        .expect_err("a custom call is refused");
        assert_eq!(
            refusal.to_string(),
            "Test Wire carries JSON tool arguments only, so custom tool call `apply_patch` and its raw input cannot be sent on it"
        );
        // Refusals are request failures in the shared error vocabulary.
        let provider: crate::error::ProviderError = crate::error::EncodeError::from(refusal).into();
        assert_eq!(provider.kind(), crate::error::ErrorKind::Request);
    }

    #[test]
    fn name_keyed_dispatch_refuses_namespaced_and_custom_calls() {
        let AssistantContent::ToolCall(plain) = function_call() else {
            unreachable!()
        };
        assert_eq!(name_dispatchable_call(&function_call()), Ok(Some(&plain)));
        assert_eq!(
            name_dispatchable_call(&namespaced_call()),
            Err(UndispatchableToolCall::Namespaced {
                id: plain.id.clone(),
                name: super::super::ToolName::new("add").expect("tool name"),
                namespace: "math".to_owned(),
            })
        );
        let json_input = AssistantContent::custom_tool_call(
            "call_3",
            super::super::ToolName::new("t").expect("tool name"),
            None,
            r#"{"a":1}"#,
        );
        assert!(
            matches!(
                name_dispatchable_call(&json_input),
                Err(UndispatchableToolCall::Custom { .. })
            ),
            "custom input that parses as JSON is still refused"
        );
        assert_eq!(
            super::super::first_undispatchable_call(&[
                AssistantContent::text("x"),
                function_call(),
                AssistantContent::CustomToolCall(custom_call()),
            ]),
            Some(UndispatchableToolCall::Custom {
                id: custom_call().id,
                name: super::super::ToolName::new("apply_patch").expect("tool name"),
                namespace: Some("repo".to_owned()),
            })
        );
    }

    #[test]
    fn missing_ids_are_minted_across_both_call_kinds() {
        let content = [
            AssistantContent::ToolCall(ToolCall::from_wire(
                "",
                ToolFunction::new(
                    super::super::ToolName::new("a").expect("tool name"),
                    serde_json::json!({}),
                ),
            )),
            AssistantContent::custom_tool_call(
                "",
                super::super::ToolName::new("b").expect("tool name"),
                None,
                "raw",
            ),
        ];
        let ids: Vec<_> = content
            .iter()
            .map(|item| match item {
                AssistantContent::ToolCall(call) => call.id.clone(),
                AssistantContent::CustomToolCall(call) => call.id.clone(),
                _ => unreachable!(),
            })
            .collect();
        assert!(ids.iter().all(super::super::CallId::is_local));
        assert_ne!(ids[0], ids[1], "two id-less calls never share a handle");
    }
}
