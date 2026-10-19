use super::*;
use crate::{
    error::ErrorKind,
    operation::Finish,
    streaming::{Item, StreamEvent},
};
use futures::StreamExt;

#[tokio::test]
async fn completion_consumes_scripted_turns_and_records_requests() {
    let model = MockCompletionModel::from_turns([
        MockTurn::text("first").with_message_id("msg_1"),
        MockTurn::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
            .with_call_id("call_1"),
    ]);

    let first = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("first scripted turn should succeed");
    assert_eq!(
        first.message_id.as_ref().map(|id| id.as_str()),
        Some("msg_1")
    );
    assert!(matches!(
        first.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "first"
    ));

    let second = model
        .call(CompletionRequest::new("use a tool"))
        .await
        .expect("second scripted turn should succeed");
    assert!(matches!(
        second.choice.first(),
        Some(AssistantContent::ToolCall(tool_call))
            if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
                && tool_call.native.is_none()
                && tool_call.function.name.as_str() == "calculator"
                && tool_call.function.arguments_value() == serde_json::json!({"x": 1})
    ));

    assert_eq!(model.request_count(), 2);
    assert_eq!(model.requests().len(), 2);
}

/// The mock behaves like a real seam: a scripted raw payload rides on the
/// normalized response unconditionally, and a turn that scripted none
/// carries the scripted turn itself serialized — the mock's own document,
/// the same capture every real adapter performs.
#[tokio::test]
async fn completion_attaches_scripted_raw_and_its_own_turn_when_unscripted() {
    let payload = serde_json::json!({"provider_only": "kept", "id": "resp_1"});
    let unscripted_turn = MockTurn::text("second");
    let expected_unscripted = unscripted_turn
        .raw()
        .expect("a scripted turn has a document");
    let model = MockCompletionModel::from_turns([
        MockTurn::text("first").with_raw(payload.clone()),
        unscripted_turn,
    ]);

    let scripted = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("first scripted turn should succeed");
    assert_eq!(scripted.raw, payload);

    let unscripted = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("second scripted turn should succeed");
    assert_eq!(unscripted.raw, expected_unscripted);
    assert_eq!(
        unscripted.raw["choice"][0]["text"],
        serde_json::json!("second"),
        "the mock's document is the turn it was scripted with"
    );

    assert_eq!(model.requests().len(), 2);
}

/// The streaming half of the same contract: the mock's document is its
/// scripted end, so the response's `raw` is that end serialized.
#[tokio::test]
async fn stream_terminal_raw_is_the_scripted_terminal_serialized() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response(Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
            ..Usage::default()
        }),
    ]]);

    let mut stream = model
        .stream(CompletionRequest::new("hello"))
        .expect("stream should open");
    while stream.next().await.is_some() {}
    let response = stream.finish().await.expect("the reply ended");
    let typed: Finish = serde_json::from_value(response.raw.clone()).expect("the end");
    assert_eq!(typed.usage.total_tokens, Some(3));
    assert_eq!(
        typed,
        super::super::streaming::mock_final(typed.usage),
        "the capture is the scripted end"
    );
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        response.raw,
        "the capture must be exactly what the scripted terminal serializes to"
    );
    assert_eq!(response.usage.total_tokens, Some(3));
}

#[tokio::test]
async fn missing_completion_turn_returns_provider_error() {
    let model = MockCompletionModel::from_turns([]);

    let err = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect_err("missing turn should error");

    assert!(matches!(
        err,
        ProviderError::Provider(message)
            if message.contains("no scripted completion turn")
    ));
}

#[tokio::test]
async fn stream_yields_scripted_events_and_records_requests() {
    // CFMOCK-002: the genuine provider correlator is the single canonical ID.
    // Both F's whole-call restatement and U's delta-only close remain covered.
    for close in [
        MockStreamEvent::tool_call("call_1", "calculator", serde_json::json!({"x": 1}))
            .with_call_id("call_1"),
        MockStreamEvent::tool_call_end("call_1"),
    ] {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::message_id("msg_stream"),
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("call_1", "{\"x\":1}"),
            close,
            MockStreamEvent::final_response_with_total_tokens(7),
        ]]);
        let mut stream = model
            .stream(CompletionRequest::new("stream"))
            .expect("stream should be created");
        let mut text = String::new();
        let mut name = None;
        let mut arguments = None;
        let mut call = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream event should succeed") {
                Item::Event(StreamEvent::Text { text: chunk, .. }) => text.push_str(&chunk),
                Item::Event(StreamEvent::Start {
                    kind: crate::streaming::PartKind::ToolCall,
                    name: emitted_name,
                    ..
                }) => name = emitted_name,
                Item::Event(StreamEvent::Arguments { json, .. }) => arguments = Some(json),
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(tool_call),
                    ..
                }) => call = Some(tool_call),
                _ => {}
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(name.as_ref().map(|name| name.as_str()), Some("calculator"));
        assert_eq!(arguments.as_deref(), Some("{\"x\":1}"));
        let call = call.expect("the call ended");
        assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_1"));
        assert!(call.native.is_none());
        assert_eq!(call.function.name, "calculator");
        assert_eq!(call.function.arguments_value(), serde_json::json!({"x": 1}));
        let response = stream.finish().await.expect("the reply ended");
        assert_eq!(response.usage.total_tokens, Some(7));
        assert_eq!(
            response.message_id.as_ref().map(|id| id.as_str()),
            Some("msg_stream")
        );
        assert_eq!(model.request_count(), 1);
    }
}

#[tokio::test]
async fn stream_error_event_is_returned() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error("boom")]]);
    let mut stream = model
        .stream(CompletionRequest::new("stream"))
        .expect("stream should be created");

    let err = stream
        .next()
        .await
        .expect("stream should yield one event")
        .expect_err("scripted event should error");

    assert_eq!(err.kind(), ErrorKind::Provider);
    assert_eq!(err.to_string(), "ProviderError: boom");
}

#[test]
fn scripted_null_stays_distinct_from_an_unscripted_document_after_round_trip() {
    let unscripted = MockTurn::text("hello");
    let scripted = unscripted.clone().with_raw(serde_json::Value::Null);
    for turn in [unscripted, scripted] {
        let json = serde_json::to_string(&turn).expect("turn serializes");
        let restored: MockTurn = serde_json::from_str(&json).expect("turn deserializes");
        assert_eq!(
            restored.raw().expect("a document"),
            turn.raw().expect("a document")
        );
        assert_eq!(restored, turn);
    }
}

#[test]
fn a_script_is_serde_in_and_serde_out() {
    let turns = vec![
        MockTurn::text("hello"),
        MockTurn::tool_call("tc1", "add", serde_json::json!({"x": 1})),
        MockTurn::text("scripted").with_raw(serde_json::json!({"id": "resp_1"})),
        MockTurn::error("boom"),
    ];
    let json = serde_json::to_string(&turns).expect("turns serialize");
    let restored: Vec<MockTurn> = serde_json::from_str(&json).expect("turns deserialize");
    assert_eq!(restored, turns);
    assert_eq!(
        restored[2].raw().expect("a document"),
        serde_json::json!({"id": "resp_1"}),
        "a scripted document survives the round trip"
    );

    let model = MockCompletionModel::from_turns(restored);
    assert_eq!(model.script(), turns);
    assert_eq!(model.stream_script(), Vec::<Vec<MockStreamEvent>>::new());

    let stream_turns = vec![vec![
        MockStreamEvent::Text("hi".into()),
        MockStreamEvent::FinalResponse(super::super::streaming::mock_final(Usage::default())),
    ]];
    let json = serde_json::to_string(&stream_turns).expect("stream turns serialize");
    let restored: Vec<Vec<MockStreamEvent>> =
        serde_json::from_str(&json).expect("stream turns deserialize");
    assert_eq!(restored, stream_turns);
    let model = MockCompletionModel::from_stream_turns(restored);
    assert_eq!(model.stream_script(), stream_turns);
    assert!(model.script().is_empty());
}

#[tokio::test]
async fn completion_keeps_the_headers_captured_by_opened() {
    let headers = crate::completion::ProviderResponseHeaders::from([
        ("x-provider-proof".to_owned(), "kept".to_owned()),
        ("x-request-id".to_owned(), "req_1".to_owned()),
    ]);
    let turn = MockTurn::text("answer").with_provider_response_headers(headers.clone());
    let expected_raw = turn.raw().expect("a scripted document");
    let model = MockCompletionModel::from_turns([turn]);
    let response = model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("reply");
    assert_eq!(response.provider_response_headers, headers);
    assert_eq!(response.raw, expected_raw);
}

async fn collect_reasoning_script(
    model: MockCompletionModel,
) -> (Vec<Item<StreamEvent>>, CompletionResponse) {
    let mut stream = model
        .stream(CompletionRequest::new("reasoning"))
        .expect("stream");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("scripted item"));
    }
    let response = stream.finish().await.expect("scripted completion");
    assert!(response.error.is_none());
    (items, response)
}

#[tokio::test]
async fn scripted_full_reasoning_block_supersedes_its_accumulated_deltas() {
    use crate::message::{CarriedPosition, CarriedReasoningPart};
    let whole = MockStreamEvent::Reasoning {
        id: "rs_1".to_owned(),
        content: CarriedReasoningPart::Text {
            text: "the complete chain".to_owned(),
            signature: Some("sig_1".to_owned()),
        },
    };
    let raw = serde_json::to_value(&whole).expect("received script record");
    let (items, response) =
        collect_reasoning_script(MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::reasoning_delta_with_id("rs_1", "partial "),
            whole,
            MockStreamEvent::final_response_with_total_tokens(2),
        ]]))
        .await;
    let fragments: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Reasoning { part, text }) => Some((*part, text.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].1, "partial ");
    let ends: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(reasoning),
                reasoning_authority,
            }) => Some((*part, reasoning, reasoning_authority)),
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0, fragments[0].0);
    assert_eq!(ends[0].2, &None);
    let reasoning = ends[0].1;
    assert_eq!(response.choice.len(), 1);
    assert_eq!(
        response.choice.first(),
        Some(&AssistantContent::CarriedReasoning(reasoning.clone()))
    );
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "the complete chain".to_owned(),
            signature: Some("sig_1".to_owned()),
        }]
    );
    assert_eq!(reasoning.payload.item, raw);
    assert_eq!(reasoning.payload.provider.as_deref(), Some("mock"));
    assert_eq!(reasoning.payload.position, CarriedPosition::Whole);
    assert_eq!(response.usage.total_tokens, Some(2));
}

#[tokio::test]
async fn structured_reasoning_retains_every_part_and_nested_provenance() {
    use crate::message::{CarriedPayload, CarriedPosition, CarriedReasoningPart};
    let parts = [
        CarriedReasoningPart::Text {
            text: "same".to_owned(),
            signature: Some("signed".to_owned()),
        },
        CarriedReasoningPart::Summary("summary".to_owned()),
        CarriedReasoningPart::Encrypted("ciphertext".to_owned()),
        CarriedReasoningPart::Redacted {
            data: "withheld".to_owned(),
        },
        CarriedReasoningPart::OpaqueSummary(
            CarriedPayload::new(
                serde_json::json!({"type":"future_summary","future":7}),
                Some("gemini".to_owned()),
            )
            .with_position(CarriedPosition::ReasoningSummary { part_index: 7 }),
        ),
        CarriedReasoningPart::OpaqueContent(
            CarriedPayload::new(
                serde_json::json!({"type":"future_content","future":9}),
                Some("anthropic".to_owned()),
            )
            .with_position(CarriedPosition::ReasoningContent { part_index: 3 }),
        ),
    ];
    for content in parts {
        let event = MockStreamEvent::Reasoning {
            id: "rs_1".to_owned(),
            content: content.clone(),
        };
        let raw = serde_json::to_value(&event).expect("script record");
        assert_eq!(
            serde_json::from_value::<MockStreamEvent>(raw.clone()).expect("script read"),
            event
        );
        let (items, response) = collect_reasoning_script(MockCompletionModel::from_stream_turns([
            vec![event, MockStreamEvent::final_response_with_default_usage()],
        ]))
        .await;
        assert_eq!(response.choice.len(), 1);
        let Some(AssistantContent::CarriedReasoning(reasoning)) = response.choice.first() else {
            panic!(
                "the structured part must remain carried: {:?}",
                response.choice
            );
        };
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert_eq!(reasoning.content, vec![content]);
        assert_eq!(reasoning.payload.item, raw);
        assert_eq!(reasoning.payload.provider.as_deref(), Some("mock"));
        assert_eq!(reasoning.payload.position, CarriedPosition::Whole);
        let authorities: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End {
                    content: AssistantContent::CarriedReasoning(_),
                    reasoning_authority,
                    ..
                }) => Some(reasoning_authority),
                _ => None,
            })
            .collect();
        assert_eq!(authorities, vec![&None]);
    }
}

#[tokio::test]
async fn same_text_signed_reasoning_closes_the_original_part() {
    use crate::message::CarriedReasoningPart;
    let (items, response) =
        collect_reasoning_script(MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::reasoning_delta_with_id("rs_1", "same"),
            MockStreamEvent::Reasoning {
                id: "rs_1".to_owned(),
                content: CarriedReasoningPart::Text {
                    text: "same".to_owned(),
                    signature: Some("sig_1".to_owned()),
                },
            },
            MockStreamEvent::final_response_with_default_usage(),
        ]]))
        .await;
    let starts: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::Reasoning,
                ..
            }) => Some(*part),
            _ => None,
        })
        .collect();
    let ends: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(_),
                reasoning_authority,
                ..
            }) => {
                assert_eq!(reasoning_authority, &None);
                Some(*part)
            }
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 1);
    assert_eq!(ends, starts);
    let Some(AssistantContent::CarriedReasoning(reasoning)) = response.choice.first() else {
        panic!("carried reasoning")
    };
    assert_eq!(response.choice.len(), 1);
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "same".to_owned(),
            signature: Some("sig_1".to_owned()),
        }]
    );
}

#[tokio::test]
async fn signed_restatement_keeps_its_slot_across_an_interleaved_call() {
    use crate::message::CarriedReasoningPart;
    let (items, response) =
        collect_reasoning_script(MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::reasoning_delta_with_id("rs_1", "partial "),
            MockStreamEvent::tool_call("call_1", "calculator", serde_json::json!({"x":1})),
            MockStreamEvent::Reasoning {
                id: "rs_1".to_owned(),
                content: CarriedReasoningPart::Text {
                    text: "the complete chain".to_owned(),
                    signature: Some("sig_1".to_owned()),
                },
            },
            MockStreamEvent::final_response_with_default_usage(),
        ]]))
        .await;
    assert_eq!(response.choice.len(), 2);
    assert!(
        matches!(&response.choice[0], AssistantContent::CarriedReasoning(reasoning)
        if reasoning.id.as_deref() == Some("rs_1") && reasoning.content == vec![CarriedReasoningPart::Text {
            text: "the complete chain".to_owned(), signature: Some("sig_1".to_owned()),
        }])
    );
    assert!(
        matches!(&response.choice[1], AssistantContent::ToolCall(call)
        if call.id.provider().map(|id| id.as_str()) == Some("call_1")
            && call.function.name.as_str() == "calculator" && call.function.arguments_value() == serde_json::json!({"x":1}))
    );
    let starts: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::Start { part, kind, .. }) => Some((*part, *kind)),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].1, crate::streaming::PartKind::Reasoning);
    assert_eq!(starts[1].1, crate::streaming::PartKind::ToolCall);
    let reasoning_ends: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::CarriedReasoning(_),
                reasoning_authority,
                ..
            }) => {
                assert_eq!(reasoning_authority, &None);
                Some(*part)
            }
            _ => None,
        })
        .collect();
    assert_eq!(reasoning_ends, vec![starts[0].0]);
}

#[tokio::test]
async fn named_anthropic_routing_still_marks_reasoning_as_mock() {
    use crate::message::{CarriedPosition, CarriedReasoningPart};
    let event = MockStreamEvent::Reasoning {
        id: "rs_1".to_owned(),
        content: CarriedReasoningPart::Summary("summary".to_owned()),
    };
    let raw = serde_json::to_value(&event).expect("script record");
    let mut model = MockCompletionModel::from_stream_turns([vec![
        event,
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    model.wire = MockScript::new("anthropic");
    assert_eq!(model.wire.name, "anthropic");
    let (_, response) = collect_reasoning_script(model).await;
    let Some(AssistantContent::CarriedReasoning(reasoning)) = response.choice.first() else {
        panic!("carried reasoning")
    };
    assert_eq!(reasoning.payload.provider.as_deref(), Some("mock"));
    assert_eq!(reasoning.payload.position, CarriedPosition::Whole);
    assert_eq!(reasoning.payload.item, raw);
}

#[tokio::test]
async fn unsigned_reasoning_helpers_keep_the_upstream_delta_close() {
    let event = MockStreamEvent::reasoning("supplied whole").with_reasoning_id("rs_1");
    let (_, response) = collect_reasoning_script(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::reasoning_delta_with_id("rs_1", "partial"),
        event,
        MockStreamEvent::final_response_with_default_usage(),
    ]]))
    .await;
    assert_eq!(response.choice.len(), 1);
    assert!(
        matches!(response.choice.first(), Some(AssistantContent::Reasoning(reasoning))
        if reasoning.text == "partial" && !reasoning.redacted)
    );
    let (_, response) = collect_reasoning_script(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::reasoning("standalone"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]))
    .await;
    assert!(
        matches!(response.choice.first(), Some(AssistantContent::Reasoning(reasoning))
        if reasoning.text == "standalone" && !reasoning.redacted)
    );
}

// MOCK-R1-004: F's absent mock message ID is an explicit JSON null.
// This pins the unscripted raw and script serde field, not a provider decoder.
#[test]
fn absent_mock_message_id_keeps_its_explicit_null_in_raw_and_script() {
    let turn = MockTurn::text("x");
    let raw = turn.raw().expect("unscripted mock document");
    assert_eq!(raw.get("message_id"), Some(&serde_json::Value::Null));
    let script = serde_json::to_value(&turn).expect("turn serializes");
    assert_eq!(
        script["response"]["Ok"].get("message_id"),
        Some(&serde_json::Value::Null)
    );
    let restored: MockTurn =
        serde_json::from_value(script.clone()).expect("explicit null ID decodes");
    assert_eq!(restored, turn);
    assert_eq!(
        restored
            .raw()
            .expect("restored mock document")
            .get("message_id"),
        Some(&serde_json::Value::Null)
    );
    assert_eq!(
        serde_json::to_value(&restored).expect("restored turn serializes"),
        script
    );
}

// CFMOCK-003 records the request after the real Model.call preparation.
#[tokio::test]
async fn completion_records_prepared_merged_user_history_and_consumes_one_turn() {
    use crate::message::{Message, UserContent};

    let original_script = vec![MockTurn::text("first"), MockTurn::text("second")];
    let model = MockCompletionModel::from_turns(original_script.clone());
    assert!(matches!(model.requests().as_slice(), []));
    assert_eq!(model.request_count(), 0);
    assert_eq!(model.script(), original_script);

    let request = CompletionRequest::from(vec![
        Message::User {
            content: vec![UserContent::text("A")],
        },
        Message::User {
            content: vec![UserContent::text("B")],
        },
    ]);
    model
        .call(request)
        .await
        .expect("the prepared request should succeed");

    assert_eq!(model.request_count(), 1);
    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].model.as_deref(), Some(MOCK_MODEL));
    assert_eq!(
        requests[0].chat_history,
        vec![Message::User {
            content: vec![UserContent::text("A"), UserContent::text("B")],
        }]
    );
    assert_eq!(model.script(), vec![original_script[1].clone()]);
}

// CFMOCK-003: invalid caller content fails real preparation before transport.
#[tokio::test]
async fn completion_refuses_empty_user_before_recording_or_consuming_script() {
    use crate::message::Message;

    let original_script = vec![MockTurn::text("first"), MockTurn::text("second")];
    let model = MockCompletionModel::from_turns(original_script.clone());
    assert!(matches!(model.requests().as_slice(), []));
    assert_eq!(model.request_count(), 0);
    assert_eq!(model.script(), original_script);

    let request = CompletionRequest::from(vec![Message::User { content: vec![] }]);
    let error = model
        .call(request)
        .await
        .expect_err("empty User content must be refused");
    assert!(matches!(error, ProviderError::Request(_)));
    assert!(matches!(model.requests().as_slice(), []));
    assert_eq!(model.request_count(), 0);
    assert_eq!(model.script(), original_script);
}

// MOCK-R1-001a: final object and original delivered fragments share one Part.
#[tokio::test]
async fn whole_object_restates_the_original_call_part() {
    for fragmented in [false, true] {
        let mut events = Vec::new();
        if fragmented {
            events.push(MockStreamEvent::tool_call_name_delta(
                "call_1",
                "calculator",
            ));
            events.push(MockStreamEvent::tool_call_arguments_delta(
                "call_1",
                "{\"x\":1}",
            ));
        }
        events.push(MockStreamEvent::tool_call(
            "call_1",
            "calculator",
            serde_json::json!({"x":2}),
        ));
        events.push(MockStreamEvent::final_response_with_total_tokens(2));
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("object"))
            .expect("stream opens");
        let mut started = None;
        let mut argument_parts = Vec::new();
        let mut delivered = Vec::new();
        let mut ended = None;
        while let Some(item) = stream.next().await {
            match item.expect("object stream succeeds") {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: crate::streaming::PartKind::ToolCall,
                    ..
                }) => started = Some(part),
                Item::Event(StreamEvent::Arguments { part, json }) => {
                    argument_parts.push(part);
                    delivered.push(json);
                }
                Item::Event(StreamEvent::End {
                    part,
                    content: AssistantContent::ToolCall(call),
                    ..
                }) => {
                    ended = Some(part);
                    assert_eq!(call.function.arguments_value(), serde_json::json!({"x":2}));
                    assert!(call.function.invalid_arguments.is_none());
                }
                _ => {}
            }
        }
        assert!(started.is_some());
        assert_eq!(ended, started);
        assert_eq!(argument_parts, vec![started.expect("actual call Part")]);
        assert_eq!(
            delivered,
            vec![(if fragmented { "{\"x\":1}" } else { "{\"x\":2}" }).to_owned()]
        );
        let response = stream.finish().await.expect("genuine terminal succeeds");
        assert_eq!(response.choice.len(), 1);
        let AssistantContent::ToolCall(call) = &response.choice[0] else {
            panic!("completed call");
        };
        assert_eq!(call.function.arguments_value(), serde_json::json!({"x":2}));
        assert_eq!(response.usage.total_tokens, Some(2));
    }
}

// MOCK-R1-001b: both changing-name subjects pin the admitted canonical name.
#[tokio::test]
async fn changing_tool_names_keep_the_first_nonempty_canonical_name() {
    for whole in [false, true] {
        let close = if whole {
            MockStreamEvent::tool_call("call_1", "new", serde_json::json!({}))
        } else {
            MockStreamEvent::tool_call_end("call_1")
        };
        let mut events = vec![MockStreamEvent::tool_call_name_delta("call_1", "old")];
        if !whole {
            events.push(MockStreamEvent::tool_call_name_delta("call_1", "new"));
        }
        events.push(close);
        events.push(MockStreamEvent::final_response_with_total_tokens(2));
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("names"))
            .expect("stream opens");
        let mut names = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("name stream succeeds") {
                Item::Event(StreamEvent::Start {
                    name: Some(name),
                    kind: crate::streaming::PartKind::ToolCall,
                    ..
                }) => names.push(name.as_str().to_owned()),
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(call),
                    ..
                }) => assert_eq!(call.function.name.as_str(), "old"),
                _ => {}
            }
        }
        assert_eq!(names, vec!["old"]);
        let response = stream.finish().await.expect("genuine terminal succeeds");
        let AssistantContent::ToolCall(call) = &response.choice[0] else {
            panic!("completed call");
        };
        assert_eq!(call.function.name.as_str(), "old");
    }
}

// MOCK-R1-002: completed, never-opened and id-less no-open ends allocate nothing.
#[tokio::test]
async fn payloadless_ends_without_an_open_call_preserve_terminal_success() {
    for (events, expected) in [
        (
            vec![
                MockStreamEvent::tool_call("call_1", "calculator", serde_json::json!({"x":1})),
                MockStreamEvent::tool_call_end("call_1"),
            ],
            1,
        ),
        (vec![MockStreamEvent::tool_call_end("never_opened")], 0),
        (vec![MockStreamEvent::tool_call_end("")], 0),
    ] {
        let mut events = events;
        events.push(MockStreamEvent::final_response_with_total_tokens(2));
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("close"))
            .expect("stream opens");
        let mut starts = 0;
        let mut calls = 0;
        while let Some(item) = stream.next().await {
            match item.expect("no-open close succeeds") {
                Item::Event(StreamEvent::Start {
                    kind: crate::streaming::PartKind::ToolCall,
                    ..
                }) => starts += 1,
                Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                }) => calls += 1,
                _ => {}
            }
        }
        assert_eq!(starts, expected);
        assert_eq!(calls, expected);
        let response = stream.finish().await.expect("terminal survives");
        assert_eq!(response.choice.len(), expected);
        assert_eq!(response.usage.total_tokens, Some(2));
    }
}

// MOCK-R1-002: neither a genuine reopened assembly nor a second whole is suppressed.
#[tokio::test]
async fn payloadless_close_keeps_reopened_calls_and_repeated_whole_calls() {
    for fragments in [false, true] {
        let mut events = vec![MockStreamEvent::tool_call(
            "call_1",
            "calculator",
            serde_json::json!({"x":1}),
        )];
        if fragments {
            events.extend([
                MockStreamEvent::tool_call_name_delta("call_1", "probe"),
                MockStreamEvent::tool_call_arguments_delta("call_1", "{}"),
                MockStreamEvent::tool_call_end("call_1"),
            ]);
        } else {
            events.push(MockStreamEvent::tool_call(
                "call_1",
                "probe",
                serde_json::json!({}),
            ));
        }
        events.push(MockStreamEvent::final_response_with_total_tokens(2));
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("reopen"))
            .expect("stream opens");
        let mut names = Vec::new();
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            }) = item.expect("reopened calls succeed")
            {
                names.push(call.function.name.as_str().to_owned());
            }
        }
        assert_eq!(names, vec!["calculator", "probe"]);
        let response = stream.finish().await.expect("terminal succeeds");
        assert_eq!(response.choice.len(), 2);
    }
}

// MOCK-R1-002/five-2: one private empty-key assembly interleaves a genuine keyed call.
#[tokio::test]
async fn empty_key_call_fragments_keep_one_part_across_keyed_interleaving() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("", "old"),
        MockStreamEvent::tool_call_arguments_delta("", "{\"x\":"),
        MockStreamEvent::tool_call_name_delta("call_2", "probe"),
        MockStreamEvent::tool_call_arguments_delta("call_2", "{}"),
        MockStreamEvent::tool_call_name_delta("", "new"),
        MockStreamEvent::tool_call_arguments_delta("", "1}"),
        MockStreamEvent::tool_call_end(""),
        MockStreamEvent::tool_call_end("call_2"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("empty key"))
        .expect("stream opens");
    let mut starts = Vec::new();
    let mut argument_parts = Vec::new();
    let mut ends = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("interleaving succeeds") {
            Item::Event(StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::ToolCall,
                ..
            }) => starts.push(part),
            Item::Event(StreamEvent::Arguments { part, .. }) => argument_parts.push(part),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(_),
                ..
            }) => ends.push(part),
            _ => {}
        }
    }
    assert_eq!(starts.len(), 2);
    assert_eq!(ends, starts);
    assert_eq!(argument_parts, vec![starts[0], starts[1], starts[0]]);
    let response = stream.finish().await.expect("terminal succeeds");
    assert_eq!(response.choice.len(), 2);
    let AssistantContent::ToolCall(empty) = &response.choice[0] else {
        panic!("empty-key call");
    };
    assert_eq!(empty.function.name.as_str(), "old");
    assert_eq!(empty.function.arguments_value(), serde_json::json!({"x":1}));
    assert!(empty.id.provider().is_none());
    let AssistantContent::ToolCall(keyed) = &response.choice[1] else {
        panic!("keyed call");
    };
    assert_eq!(keyed.id.provider().map(|id| id.as_str()), Some("call_2"));
}

// MOCK-R1-003: repeated starts remain distinct, including interleaved content order.
#[tokio::test]
async fn repeated_text_start_keys_keep_distinct_parts_in_content_order() {
    for interleaved in [false, true] {
        let mut events = vec![
            MockStreamEvent::text_start("txt_1", None),
            MockStreamEvent::text("a"),
        ];
        if interleaved {
            events.push(MockStreamEvent::tool_call(
                "call_1",
                "probe",
                serde_json::json!({}),
            ));
        }
        events.extend([
            MockStreamEvent::text_start("txt_1", None),
            MockStreamEvent::text("b"),
            MockStreamEvent::final_response_with_total_tokens(2),
        ]);
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("starts"))
            .expect("stream opens");
        let mut text_parts = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("text starts succeed") {
                Item::Event(StreamEvent::Text { part, text }) => {
                    text_parts.push(part);
                    assert!(text == "a" || text == "b");
                }
                Item::Event(StreamEvent::Resume { .. }) => {
                    panic!("mock selects no Responses resume")
                }
                _ => {}
            }
        }
        assert_eq!(text_parts.len(), 2);
        assert_ne!(text_parts[0], text_parts[1]);
        let response = stream.finish().await.expect("terminal succeeds");
        assert_eq!(response.choice.len(), if interleaved { 3 } else { 2 });
        let AssistantContent::Text(first) = &response.choice[0] else {
            panic!("first text");
        };
        assert_eq!(first.text, "a");
        if interleaved {
            assert!(matches!(&response.choice[1], AssistantContent::ToolCall(_)));
        }
        let AssistantContent::Text(second) = response.choice.last().expect("second text") else {
            panic!("second text");
        };
        assert_eq!(second.text, "b");
    }
}

// MOCK-R1-005: exact bounded assembly, typed correlation, no End, terminal divergence.
#[tokio::test]
async fn malformed_payloadless_close_keeps_raw_correlation_and_ends_the_stream() {
    const RAW: &str = "{\"x\": 2, \"y\": \x01";
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("call_1", RAW),
        MockStreamEvent::tool_call_end("call_1"),
        MockStreamEvent::text("after"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("malformed"))
        .expect("stream opens");
    let mut calls = 0;
    let error = loop {
        match stream.next().await.expect("typed error must arrive") {
            Err(error) => break error,
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            })) => calls += 1,
            Ok(_) => {}
        }
    };
    assert_eq!(calls, 0);
    let ProviderError::Relayed(report) = error else {
        panic!("typed relayed report");
    };
    assert_eq!(report.kind, ErrorKind::Response);
    let Some(crate::error::ErrorDetail::MalformedToolInput(input)) = &report.detail else {
        panic!("malformed detail");
    };
    assert_eq!(input.name, "calculator");
    assert_eq!(input.raw, RAW);
    assert_eq!(input.id.provider().map(|id| id.as_str()), Some("call_1"));
    let provider = input
        .provider
        .as_ref()
        .expect("received provider correlation");
    assert_eq!(provider.call_id.as_str(), "call_1");
    assert!(provider.item_id.is_none());
    assert!(!input.error.is_empty());
    assert!(stream.next().await.is_none());
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Relayed(_))
    ));
}

// MOCK-R1-001a/005: valid whole object before close wins over malformed fragments.
#[tokio::test]
async fn valid_whole_object_supersedes_malformed_fragments_before_close() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "not-json"),
        MockStreamEvent::tool_call("call_1", "calculator", serde_json::json!({"x":2})),
        MockStreamEvent::tool_call_end("call_1"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("supersede"))
        .expect("stream opens");
    let mut calls = 0;
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(call),
            ..
        }) = item.expect("whole supersession succeeds")
        {
            calls += 1;
            assert_eq!(call.function.arguments_value(), serde_json::json!({"x":2}));
            assert!(call.function.invalid_arguments.is_none());
        }
    }
    assert_eq!(calls, 1);
    assert_eq!(
        stream
            .finish()
            .await
            .expect("terminal succeeds")
            .choice
            .len(),
        1
    );
}

// MOCK-R1-005/five-2: source-only controls at the exact private F bound.
#[tokio::test]
async fn bounded_call_arguments_accept_under_and_exactly_at_the_f_bound() {
    const LIMIT: usize = 32 * 1024 * 1024;
    for length in [LIMIT - 1, LIMIT] {
        let raw = format!("{{\"x\":1}}{}", " ".repeat(length - 7));
        assert_eq!(raw.len(), length);
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("call_1", raw),
            MockStreamEvent::tool_call_end("call_1"),
            MockStreamEvent::final_response_with_total_tokens(2),
        ]]);
        let mut stream = model
            .stream(CompletionRequest::new("bound"))
            .expect("stream opens");
        while let Some(item) = stream.next().await {
            item.expect("within-bound arguments succeed");
        }
        let response = stream.finish().await.expect("terminal succeeds");
        let AssistantContent::ToolCall(call) = &response.choice[0] else {
            panic!("completed call");
        };
        assert_eq!(call.function.arguments_value(), serde_json::json!({"x":1}));
        assert!(call.function.invalid_arguments.is_none());
    }
}

// MOCK-R1-005/five-2: overflow refuses the whole fragment and keeps F raw evidence.
#[tokio::test]
async fn over_bound_close_reports_the_exact_retained_assembly_and_correlation() {
    const LIMIT: usize = 32 * 1024 * 1024;
    for prefix in [Some("{\"x\":1}"), None] {
        let mut events = vec![MockStreamEvent::tool_call_name_delta(
            "call_1",
            "calculator",
        )];
        if let Some(prefix) = prefix {
            events.push(MockStreamEvent::tool_call_arguments_delta("call_1", prefix));
        }
        events.extend([
            MockStreamEvent::tool_call_arguments_delta("call_1", " ".repeat(LIMIT + 1)),
            MockStreamEvent::tool_call_end("call_1"),
            MockStreamEvent::final_response_with_total_tokens(2),
        ]);
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("overflow"))
            .expect("stream opens");
        let error = loop {
            match stream.next().await.expect("typed overflow error arrives") {
                Err(error) => break error,
                Ok(Item::Event(StreamEvent::End {
                    content: AssistantContent::ToolCall(_),
                    ..
                })) => panic!("no completed overflow call"),
                Ok(_) => {}
            }
        };
        let ProviderError::Relayed(report) = error else {
            panic!("typed relayed report");
        };
        assert_eq!(report.kind, ErrorKind::Response);
        let Some(crate::error::ErrorDetail::MalformedToolInput(input)) = &report.detail else {
            panic!("malformed detail");
        };
        assert_eq!(input.raw, prefix.unwrap_or(""));
        assert_eq!(input.id.provider().map(|id| id.as_str()), Some("call_1"));
        assert_eq!(
            input
                .provider
                .as_ref()
                .expect("received correlation")
                .call_id
                .as_str(),
            "call_1"
        );
        assert_eq!(
            input.error,
            "tool-call input exceeded the accumulation bound"
        );
        assert!(stream.next().await.is_none());
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Relayed(_))
        ));
    }
}

// MOCK-R1-005/five-2: F's literal-null placeholder yields to real object fragments.
#[tokio::test]
async fn null_placeholder_is_replaced_by_the_real_argument_assembly() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "null"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "{\"x\":1}"),
        MockStreamEvent::tool_call_end("call_1"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("placeholder"))
        .expect("stream opens");
    let mut delivered = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Arguments { json, .. }) =
            item.expect("normalized assembly succeeds")
        {
            delivered.push(json);
        }
    }
    assert_eq!(delivered, vec!["{\"x\":1}"]);
    let response = stream.finish().await.expect("terminal succeeds");
    let AssistantContent::ToolCall(call) = &response.choice[0] else {
        panic!("completed call");
    };
    assert_eq!(call.function.arguments_value(), serde_json::json!({"x":1}));
    assert!(call.function.invalid_arguments.is_none());
}

// MOCK-R1-006: both metadata entry sites retain F arrays/objects/scalars/atomic marker.
#[tokio::test]
async fn text_metadata_composes_f_values_at_both_private_sites() {
    let first = serde_json::json!({
        "citations":[{"n":1}], "extra":{"a":1}, "scalar":7, "string":"old", "type":"old",
        "openai_responses_part":{"kind":"opaque","value":{"old":1},"old_only":true},
        "nested":{"openai_responses_part":{"kind":"opaque","value":{"old":1}}},
    });
    let second = serde_json::json!({
        "citations":[{"n":2}], "extra":{"b":2}, "scalar":false, "string":"new", "type":"new",
        "openai_responses_part":{"kind":"opaque","value":{"new":2}},
        "nested":{"openai_responses_part":{"kind":"opaque","value":{"new":2}}},
    });
    let expected = serde_json::json!({
        "citations":[{"n":1},{"n":2}], "extra":{"a":1,"b":2}, "scalar":false, "string":"new", "type":"new",
        "openai_responses_part":{"kind":"opaque","value":{"new":2}},
        "nested":{"openai_responses_part":{"kind":"opaque","value":{"old":1,"new":2}}},
    });
    for start in [false, true] {
        let mut events = if start {
            vec![
                MockStreamEvent::text_start("txt_1", Some(first.clone())),
                MockStreamEvent::text("x"),
            ]
        } else {
            vec![
                MockStreamEvent::text("x"),
                MockStreamEvent::text_additional_params(first.clone()),
            ]
        };
        events.extend([
            MockStreamEvent::text_additional_params(second.clone()),
            MockStreamEvent::final_response_with_total_tokens(2),
        ]);
        let model = MockCompletionModel::from_stream_turns([events]);
        let mut stream = model
            .stream(CompletionRequest::new("metadata"))
            .expect("stream opens");
        while let Some(item) = stream.next().await {
            item.expect("metadata succeeds");
        }
        let response = stream.finish().await.expect("terminal succeeds");
        assert_eq!(response.choice.len(), 1);
        let AssistantContent::Text(text) = &response.choice[0] else {
            panic!("text");
        };
        assert_eq!(text.text, "x");
        assert_eq!(response.choice[0].native_item(), Some(&expected));
    }
}

// MOCK-R1-006: only incoming top-level opaque/value-present markers are atomic.
#[tokio::test]
async fn metadata_marker_controls_and_repeated_starts_keep_their_exact_scope() {
    for (incoming, expected) in [
        (
            serde_json::json!({"kind":"refusal","value":{"new":2}}),
            serde_json::json!({"kind":"refusal","value":{"old":1,"new":2},"old_only":true}),
        ),
        (
            serde_json::json!({"kind":"opaque","new_only":true}),
            serde_json::json!({"kind":"opaque","value":{"old":1},"old_only":true,"new_only":true}),
        ),
        (
            serde_json::json!({"kind":"opaque","value":null}),
            serde_json::json!({"kind":"opaque","value":null}),
        ),
    ] {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text_start(
                "txt_1",
                Some(
                    serde_json::json!({"openai_responses_part":{"kind":"opaque","value":{"old":1},"old_only":true}}),
                ),
            ),
            MockStreamEvent::text("a"),
            MockStreamEvent::text_additional_params(
                serde_json::json!({"openai_responses_part":incoming}),
            ),
            MockStreamEvent::text_start("txt_1", Some(serde_json::json!({"only_second":true}))),
            MockStreamEvent::text("b"),
            MockStreamEvent::final_response_with_total_tokens(2),
        ]]);
        let mut stream = model
            .stream(CompletionRequest::new("marker"))
            .expect("stream opens");
        while let Some(item) = stream.next().await {
            item.expect("marker controls succeed");
        }
        let response = stream.finish().await.expect("terminal succeeds");
        assert_eq!(response.choice.len(), 2);
        assert_eq!(
            response.choice[0].native_item(),
            Some(&serde_json::json!({"openai_responses_part":expected}))
        );
        assert_eq!(
            response.choice[1].native_item(),
            Some(&serde_json::json!({"only_second":true}))
        );
    }
}

// MOCK-R1-001a/prod-copy-2: a changed genuine ID takes over a new real Part.
#[tokio::test]
async fn genuine_id_takeover_restates_the_object_on_the_new_part() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("assembly_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("assembly_1", "{\"x\":"),
        MockStreamEvent::tool_call("assembly_1", "calculator", serde_json::json!({"x":2}))
            .with_call_id("different_1"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("takeover"))
        .expect("stream opens");
    let mut starts = Vec::new();
    let mut arguments = Vec::new();
    let mut ends = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("the real takeover succeeds") {
            Item::Event(StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::ToolCall,
                ..
            }) => {
                if starts.len() == 1 {
                    assert_eq!(
                        ends.len(),
                        1,
                        "the prior partial closes before the new Part starts"
                    );
                }
                starts.push(part);
            }
            Item::Event(StreamEvent::Arguments { part, json }) => arguments.push((part, json)),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(call),
                ..
            }) => {
                ends.push((part, call));
            }
            Item::Event(StreamEvent::Discard { .. }) => {
                panic!("actual incomplete takeover closes with End, not Discard");
            }
            _ => {}
        }
    }
    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0], starts[1]);
    assert_eq!(ends.len(), 2);
    assert_eq!(ends[0].0, starts[0]);
    assert_eq!(ends[1].0, starts[1]);
    assert_eq!(arguments, vec![(starts[0], "{\"x\":".to_owned())]);
    assert_eq!(
        ends[0].1.id.provider().map(|id| id.as_str()),
        Some("assembly_1")
    );
    assert_eq!(ends[0].1.function.name.as_str(), "calculator");
    assert_eq!(ends[0].1.function.arguments_value(), serde_json::json!({}));
    assert_eq!(
        ends[0].1.function.invalid_arguments.as_deref(),
        Some("{\"x\":")
    );
    assert_eq!(
        ends[1].1.id.provider().map(|id| id.as_str()),
        Some("different_1")
    );
    assert_eq!(ends[1].1.function.name.as_str(), "calculator");
    assert_eq!(
        ends[1].1.function.arguments_value(),
        serde_json::json!({"x":2})
    );
    assert!(ends[1].1.function.invalid_arguments.is_none());
    let response = stream.finish().await.expect("genuine terminal succeeds");
    assert_eq!(
        response.choice,
        vec![
            AssistantContent::ToolCall(ends[0].1.clone()),
            AssistantContent::ToolCall(ends[1].1.clone()),
        ]
    );
    assert_eq!(response.usage.total_tokens, Some(2));
}

// MOCK-R1-001a/005/prod-copy-2: authoritative whole retains the sticky overflow issue.
#[tokio::test]
async fn whole_object_after_overflow_keeps_the_same_part_and_sticky_invalid_marker() {
    const LIMIT: usize = 32 * 1024 * 1024;
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call_name_delta("call_1", "calculator"),
        MockStreamEvent::tool_call_arguments_delta("call_1", "{\"x\":1}"),
        MockStreamEvent::tool_call_arguments_delta("call_1", " ".repeat(LIMIT + 1)),
        MockStreamEvent::tool_call("call_1", "calculator", serde_json::json!({"x":2})),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut stream = model
        .stream(CompletionRequest::new("overflow then whole"))
        .expect("stream opens");
    let mut starts = Vec::new();
    let mut arguments = Vec::new();
    let mut ends = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("the adopted whole-object restatement succeeds") {
            Item::Event(StreamEvent::Start {
                part,
                kind: crate::streaming::PartKind::ToolCall,
                ..
            }) => starts.push(part),
            Item::Event(StreamEvent::Arguments { part, json }) => arguments.push((part, json)),
            Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(call),
                ..
            }) => ends.push((part, call)),
            Item::Event(StreamEvent::Discard { .. }) => {
                panic!("the authoritative object closes its original Part")
            }
            _ => {}
        }
    }
    assert_eq!(starts.len(), 1);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].0, starts[0]);
    assert_eq!(arguments, vec![(starts[0], "{\"x\":1}".to_owned())]);
    let call = &ends[0].1;
    assert_eq!(call.id.provider().map(|id| id.as_str()), Some("call_1"));
    assert_eq!(call.function.name.as_str(), "calculator");
    assert_eq!(call.function.arguments_value(), serde_json::json!({"x":2}));
    let authoritative_raw = serde_json::json!({"x":2}).to_string();
    assert_eq!(
        call.function.invalid_arguments.as_deref(),
        Some(authoritative_raw.as_str())
    );
    let response = stream.finish().await.expect("genuine terminal succeeds");
    assert_eq!(
        response.choice,
        vec![AssistantContent::ToolCall(call.clone())]
    );
    assert_eq!(response.usage.total_tokens, Some(2));
}
