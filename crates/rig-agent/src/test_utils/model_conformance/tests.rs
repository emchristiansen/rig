use super::*;
use crate::{
    completion::Usage,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn, mock_final},
};
use rig_core::message::{ToolCall, ToolFunction};

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(
            rig_core::message::ToolName::new(name.to_string()).expect("tool name"),
            arguments,
        ),
    ))
}

fn usage(input: u64, output: u64) -> Usage {
    Usage::new()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(input + output)
}

fn fixture_contract(condition: bool, details: &str) -> Result<(), ScenarioError> {
    if condition {
        Ok(())
    } else {
        Err(ScenarioError::contract("test_fixture", details))
    }
}

#[tokio::test]
async fn parallel_contract_validates_batch_and_correlation() -> Result<(), ScenarioError> {
    let first = MockTurn::from_contents([
        tool_call("call_add", "add", serde_json::json!({"x": 3, "y": 4})),
        tool_call(
            "call_subtract",
            "subtract",
            serde_json::json!({"x": 10, "y": 2}),
        ),
    ]);
    let report = parallel_tools(
        MockCompletionModel::from_turns([first, MockTurn::text("7 and 8")]),
        |builder| builder,
        Some(1),
    )
    .await?;
    fixture_contract(report.tool_calls == 2, "parallel tool-call count")?;
    fixture_contract(report.history_messages >= 4, "parallel history length")?;
    Ok(())
}

#[tokio::test]
async fn zero_argument_and_output_serialization_contracts_pass() -> Result<(), ScenarioError> {
    let zero = zero_argument_tool(
        MockCompletionModel::from_turns([
            MockTurn::tool_call("ping_call", "ping", serde_json::json!({})),
            MockTurn::text(PING_OUTPUT),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(zero.tool_calls == 1, "zero-argument call count")?;

    let first = MockTurn::from_contents([
        tool_call("motto_call", "fetch_motto", serde_json::json!({})),
        tool_call("config_call", "fetch_config", serde_json::json!({})),
    ]);
    let serialized = tool_output_serialization(
        MockCompletionModel::from_turns([first, MockTurn::text("summary")]),
        |builder| builder,
    )
    .await?;
    fixture_contract(serialized.tool_calls == 2, "serialized-output call count")?;
    Ok(())
}

#[tokio::test]
async fn complex_arguments_preserve_nested_unicode_and_escapes() -> Result<(), ScenarioError> {
    let arguments = serde_json::json!({
        "profile": {"name": "Zoë \"Z\"", "tags": ["rust", "東京"]},
        "mode": "careful",
        "note": "line one\nline two",
        "quote": "path C:\\tmp and \"quoted\""
    });
    let report = complex_tool_arguments(
        MockCompletionModel::from_turns([
            MockTurn::tool_call("profile_call", "store_profile", arguments),
            MockTurn::text("stored"),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "complex-argument call count")?;
    Ok(())
}

#[tokio::test]
async fn extraction_contract_requires_fields_and_usage() -> Result<(), ScenarioError> {
    let report = structured_extraction(
        MockCompletionModel::from_turns([MockTurn::tool_call(
            "submit_call",
            "submit",
            serde_json::json!({
                "first_name": "Ada",
                "last_name": "Lovelace",
                "job": "mathematician"
            }),
        )
        .with_usage(usage(20, 5))]),
        None,
    )
    .await?;
    fixture_contract(report.prompt_tokens == 20, "extraction input usage")?;
    fixture_contract(report.generated_tokens == 5, "extraction output usage")?;
    Ok(())
}

#[tokio::test]
async fn streaming_contract_checks_events_history_and_usage() -> Result<(), ScenarioError> {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("add_call", "add", serde_json::json!({"a": 17, "b": 25})),
            MockStreamEvent::FinalResponse(mock_final(usage(10, 2))),
        ],
        vec![
            MockStreamEvent::text("42"),
            MockStreamEvent::FinalResponse(mock_final(usage(14, 1))),
        ],
    ]);
    let report = streaming_tool(model, |builder| builder).await?;
    fixture_contract(report.prompt_tokens == 24, "streaming input usage")?;
    fixture_contract(report.generated_tokens == 3, "streaming output usage")?;
    Ok(())
}

#[tokio::test]
async fn invalid_recovery_paths_do_not_execute_tools() -> Result<(), ScenarioError> {
    let report = invalid_tool_recovery(
        MockCompletionModel::from_turns([MockTurn::tool_call(
            "invalid-add",
            "add",
            serde_json::json!({ "x": 2, "y": 3 }),
        )]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "recovery source call count")?;
    Ok(())
}

#[tokio::test]
async fn hook_rewrites_chain_and_request_patch_is_turn_local() -> Result<(), ScenarioError> {
    let report = hook_rewrites_and_request_patch(
        MockCompletionModel::from_turns([
            MockTurn::tool_call("hook-add", "add", serde_json::json!({ "x": 1, "y": 1 })),
            MockTurn::text("[portable-redacted]"),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "hook execution count")?;
    Ok(())
}

#[tokio::test]
async fn cancellation_and_max_turn_controls_retain_diagnostics() -> Result<(), ScenarioError> {
    let report = cancellation_and_max_turns(
        MockCompletionModel::from_turns([
            MockTurn::tool_call("cancel-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
            MockTurn::tool_call("budget-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 2, "run-control execution count")?;
    Ok(())
}

#[test]
fn typed_validators_reject_bad_structured_output_and_protocol_leaks() {
    let invalid = decode_structured_output::<ConfigOutput>("invalid_json", "not json");
    assert!(matches!(invalid, Err(ScenarioError::Contract { .. })));

    let messages = vec![Message::Assistant(
        rig_core::message::AssistantMessage::new(vec![AssistantContent::text(
            "visible <tool_call>",
        )]),
    )];
    let hygiene = validate_protocol_hygiene(
        "protocol_hygiene",
        "visible <tool_call>",
        &messages,
        &["<tool_call>"],
    );
    assert!(matches!(hygiene, Err(ScenarioError::Contract { .. })));
}

#[test]
fn invalid_tool_diagnostics_require_rejected_call_history() {
    let history = vec![Message::Assistant(
        rig_core::message::AssistantMessage::new(vec![tool_call(
            "bad_call",
            "missing",
            serde_json::json!({"value": 1}),
        )]),
    )];
    let error = PromptError::UnknownToolCall {
        tool_name: "missing".to_string(),
        available_tools: vec!["add".to_string()],
        allowed_tools: Vec::new(),
        chat_history: history,
    };
    assert!(validate_unknown_tool_failure(&error, "missing", &[]).is_ok());
    assert!(validate_unknown_tool_failure(&error, "other", &[]).is_err());
}

fn ag010_responses_body(answer: Option<&str>) -> serde_json::Value {
    // Synthetic required envelope fields drive the existing Responses decoder.
    let output = answer.map_or_else(Vec::new, |answer| {
        vec![serde_json::json!({
            "type":"message", "id":"msg_ag010", "role":"assistant",
            "status":"completed", "phase":"final_answer",
            "content":[{"type":"output_text", "text":answer, "annotations":[]}]
        })]
    });
    serde_json::json!({
        "id":"resp_ag010", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-4o-mini", "tools":[], "output":output,
        "usage":{"input_tokens":4, "output_tokens":2, "total_tokens":6},
        "synthetic_raw_only":{"text":"London"}
    })
}

fn ag010_responses_sse(body: &serde_json::Value, with_text: bool) -> String {
    let mut frames = Vec::new();
    if with_text {
        for (sequence, delta) in [(1, "Pa"), (2, "ris")] {
            frames.push(serde_json::json!({
                "type":"response.output_text.delta", "sequence_number":sequence,
                "output_index":0, "content_index":0, "item_id":"msg_ag010", "delta":delta
            }));
        }
        frames.push(serde_json::json!({
            "type":"response.output_item.done", "sequence_number":3, "output_index":0,
            "item":body["output"][0]
        }));
    }
    frames.push(serde_json::json!({
        "type":"response.completed", "sequence_number":4, "response":body
    }));
    frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect()
}

/// Synthetic conformance control: decoding, not cassette traffic, establishes
/// that a phased carried answer reaches the generic buffered text consumer.
#[tokio::test]
async fn decoded_carried_only_paris_matches_actual_stream_text() -> anyhow::Result<()> {
    use rig_core::providers::openai::{OpenAIConfig, responses_api::wire::Responses};
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

    let body = ag010_responses_body(Some("Paris"));
    let http = SequencedHttpClient::new([
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(ag010_responses_sse(&body, true), "text/event-stream"),
    ]);
    let model = rig_core::Model::new(
        Responses::new(OpenAIConfig::new("synthetic-test-key"), "gpt-4o-mini"),
        http.clone(),
    );
    let buffered = model
        .call(CompletionRequest::new("probe the synthetic answer"))
        .await?;
    assert_eq!(buffered.raw, body);
    assert!(
        matches!(buffered.choice.as_slice(), [AssistantContent::CarriedText(text)]
        if text.text() == "Paris" && text.phase.as_deref() == Some("final_answer")
            && text.payload.item == body["output"][0]["content"][0]
            && text.payload.provider.as_deref() == Some("openai"))
    );
    assert_eq!(buffered.text(), "Paris");
    assert_eq!(buffered.usage.input_tokens, Some(4));
    assert_eq!(buffered.usage.output_tokens, Some(2));
    assert_eq!(buffered.usage.total_tokens, Some(6));

    let report = buffered_streaming_text_parity(model, |request| request).await?;
    assert_eq!(report.name, "buffered_streaming_text_parity");
    assert_eq!(report.response, "Paris");
    assert_eq!(report.prompt_tokens, 4);
    assert_eq!(report.generated_tokens, 2);
    assert_eq!(report.tool_calls, 0);
    assert_eq!(report.history_messages, 0);
    assert_eq!(http.requests().len(), 3);
    assert_eq!(http.remaining_responses(), 0);
    Ok(())
}

fn ag010_gemini_body(signed_prefix: bool) -> serde_json::Value {
    // Both parts are received data; only the signed prefix becomes carried.
    let parts = if signed_prefix {
        serde_json::json!([
            {"text":"Pa", "thoughtSignature":"synthetic-signed-pa", "futureField":{"text":"London"}},
            {"text":"ris"}
        ])
    } else {
        serde_json::json!([{"text":"Paris"}])
    };
    serde_json::json!({
        "responseId":"resp_ag010_gemini", "modelVersion":"gemini-2.5-flash",
        "candidates":[{"index":0, "content":{"role":"model", "parts":parts}, "finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":4, "candidatesTokenCount":2, "totalTokenCount":6}
    })
}

/// Synthetic mixed-shape control uses Gemini's existing signed/plain decoder
/// path; no ordinary Responses part is invented for the phased control.
#[tokio::test]
async fn decoded_mixed_carried_and_plain_paris_matches_actual_stream_text() -> anyhow::Result<()> {
    use rig_core::providers::gemini::{GeminiConfig, completion::GenerateContent};
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

    let body = ag010_gemini_body(true);
    let http = SequencedHttpClient::new([
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(format!("data: {body}\n\n"), "text/event-stream"),
    ]);
    let model = rig_core::Model::new(
        GenerateContent::new(GeminiConfig::new("synthetic-test-key"), "gemini-2.5-flash"),
        http.clone(),
    );
    let buffered = model
        .call(CompletionRequest::new("probe the synthetic answer"))
        .await?;
    assert_eq!(buffered.raw, body);
    assert!(matches!(buffered.choice.as_slice(),
        [AssistantContent::CarriedText(carried), AssistantContent::Text(plain)]
        if carried.text() == "Pa" && plain.text == "ris"
            && carried.payload.item == body["candidates"][0]["content"]["parts"][0]
            && carried.payload.provider.as_deref() == Some("gcp.gemini")));
    let AssistantContent::CarriedText(carried) = &buffered.choice[0] else {
        anyhow::bail!("the signed received prefix must be carried");
    };
    assert_eq!(
        carried
            .signed_answer()
            .map(|answer| answer.signature.as_str()),
        Some("synthetic-signed-pa")
    );
    assert_eq!(buffered.text(), "Paris");
    assert_eq!(buffered.usage.input_tokens, Some(4));
    assert_eq!(buffered.usage.output_tokens, Some(2));
    assert_eq!(buffered.usage.total_tokens, Some(6));

    let report = buffered_streaming_text_parity(model, |request| request).await?;
    assert_eq!(report.response, "Paris");
    assert_eq!(report.prompt_tokens, 4);
    assert_eq!(report.generated_tokens, 2);
    assert_eq!(http.requests().len(), 3);
    assert_eq!(http.remaining_responses(), 0);
    Ok(())
}

/// Synthetic ordinary-text control keeps the existing Paris normalization
/// and reported-usage premise on a real decoder with no carried answer.
#[tokio::test]
async fn decoded_plain_paris_keeps_the_existing_conformance_premise() -> anyhow::Result<()> {
    use rig_core::providers::gemini::{GeminiConfig, completion::GenerateContent};
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

    let body = ag010_gemini_body(false);
    let http = SequencedHttpClient::new([
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(format!("data: {body}\n\n"), "text/event-stream"),
    ]);
    let model = rig_core::Model::new(
        GenerateContent::new(GeminiConfig::new("synthetic-test-key"), "gemini-2.5-flash"),
        http.clone(),
    );
    let buffered = model
        .call(CompletionRequest::new("probe the synthetic answer"))
        .await?;
    assert_eq!(buffered.raw, body);
    assert!(
        matches!(buffered.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "Paris")
    );
    assert_eq!(buffered.text(), "Paris");
    let report = buffered_streaming_text_parity(model, |request| request).await?;
    assert_eq!(report.response, "Paris");
    assert_eq!(report.prompt_tokens, 4);
    assert_eq!(report.generated_tokens, 2);
    assert_eq!(http.requests().len(), 3);
    assert_eq!(http.remaining_responses(), 0);
    Ok(())
}

/// Synthetic empty control proves raw-only text is not promoted and an
/// empty canonical answer fails the unchanged Paris premise.
#[tokio::test]
async fn an_empty_decoded_answer_stays_empty_and_refuses_the_paris_premise() -> anyhow::Result<()> {
    use rig_core::providers::openai::{OpenAIConfig, responses_api::wire::Responses};
    use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

    let mut body = ag010_responses_body(None);
    body["synthetic_raw_only"]["text"] = serde_json::json!("Paris");
    let http = SequencedHttpClient::new([
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(body.to_string(), "application/json"),
        MockHttpResponse::success_typed(ag010_responses_sse(&body, false), "text/event-stream"),
    ]);
    let model = rig_core::Model::new(
        Responses::new(OpenAIConfig::new("synthetic-test-key"), "gpt-4o-mini"),
        http.clone(),
    );
    let buffered = model
        .call(CompletionRequest::new("probe the synthetic empty answer"))
        .await?;
    assert_eq!(buffered.raw, body);
    assert!(buffered.choice.is_empty());
    assert_eq!(buffered.text(), "");
    assert!(buffered.usage.is_reported());
    let error = buffered_streaming_text_parity(model, |request| request).await;
    let Err(ScenarioError::Contract { scenario, details }) = error else {
        anyhow::bail!("empty canonical text must fail the existing Paris premise");
    };
    assert_eq!(scenario, "buffered_streaming_text_parity");
    assert!(details.starts_with("buffered=\"\", streamed=\"\", buffered_usage="));
    assert_eq!(http.requests().len(), 3);
    assert_eq!(http.remaining_responses(), 0);
    Ok(())
}
