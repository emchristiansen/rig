use super::*;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_test_support::cassette_models::OpenAiModels;

/// Separate scripted-adapter evidence; the original cassette still matches requests.
#[tokio::test]
async fn three_turn_recorded_replies_capture_current_requests_without_matching() {
    let scenario = "streaming_grammar/three_turn_tool_session";
    let replies = crate::cassettes::recorded_statuses_and_bodies("openai", scenario);
    let recorded_headers = crate::cassettes::recorded_response_header_pairs("openai", scenario);
    assert_eq!(replies.len(), 3);
    assert_eq!(recorded_headers.len(), 3);
    let responses = replies
        .into_iter()
        .zip(recorded_headers)
        .map(|((status, body), pairs)| {
            assert_eq!(status, 200);
            let mut headers = http::HeaderMap::new();
            for (name, value) in pairs {
                headers.append(
                    http::HeaderName::from_bytes(name.as_bytes()).expect("recorded header name"),
                    http::HeaderValue::from_str(&value).expect("recorded header value"),
                );
            }
            MockHttpResponse::SuccessWithHeaders(body.into(), headers)
        });
    let http = SequencedHttpClient::new(responses);
    // The same replay-only dummy key used by the cassette server.
    let client = OpenAiModels::new(openai::OpenAIConfig::new("[REDACTED]"), http.clone());
    let directory = std::path::PathBuf::from(
        std::env::var_os("RIG_THREE_TURN_CAPTURE_DIR").expect("the assigned capture directory"),
    );
    let capture_requests = || {
        for (index, request) in http.requests().iter().enumerate() {
            std::fs::write(
                directory.join(format!("REQUEST-{index}.json")),
                &request.body,
            )
            .expect("retain the complete actual request");
        }
    };
    let model = client.completion(openai::GPT_5_6);
    // A trivial tool turn skips the reasoning item entirely on the
    // wire (verified via a direct probe, even at medium effort); a
    // math sub-task at high effort reliably yields the `rs_*`
    // reasoning item the provenance assertions need.
    const FIRST_TURN_PROMPT: &str = "First work out how many positive integers n < 90 are divisible by 7 \
         (do this carefully). Then call `lookup_harbor_label` exactly once. After \
         the tool result arrives, answer with the count and the exact tool output.";
    let generation = effort(Effort::High);
    let provider = openai_options(
        OpenAiOptions::new()
            .store(false)
            .include([Include::ReasoningEncryptedContent]),
    );

    // Turn 1: forced tool work with encrypted reasoning.
    let request = CompletionRequest::new(FIRST_TURN_PROMPT)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
        .tool(rig::tool::tool_definition(&AlphaSignal))
        .options(generation.clone())
        .provider_options(provider.clone());
    let first = drain_stream(model.stream(request).expect("stream should start")).await;
    capture_requests();
    assert_terminal(&first, FinishReason::ToolCalls);

    // The reasoning item the wire produced carries a real `rs_*` id;
    // the id is derived from the recorded turn, never minted.
    let reasoning_ids: Vec<&str> = first
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => {
                reasoning.native.as_ref()?.item["id"].as_str()
            }
            AssistantContent::CarriedReasoning(reasoning) => reasoning.id.as_deref(),
            _ => None,
        })
        .collect();
    assert!(
        !reasoning_ids.is_empty(),
        "store:false encrypted turn should aggregate a reasoning part with an id, got {:?}",
        first.choice
    );
    for id in &reasoning_ids {
        assert!(
            id.starts_with("rs_"),
            "reasoning part should carry the wire's rs_* id, got {id}"
        );
    }
    let tool_call = first
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) if call.function.name == "lookup_harbor_label" => {
                Some(call.clone())
            }
            _ => None,
        })
        .expect("aggregated first turn should contain the lookup_harbor_label call");

    // Turn 2: the full aggregated choice — reasoning items with their
    // recorded rs_* ids included — goes back through the provenance
    // gate together with the tool result.
    let first_assistant = Message::Assistant(first.head().with_content(first.choice.clone()));
    let tool_result = Message::from(UserContent::tool_result(
        tool_call.id.clone(),
        tool_call.function.name.clone(),
        vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
    ));
    let second_request = CompletionRequest::new(
        "Answer in one short sentence that includes the exact tool output. \
             Do not call any tools.",
    )
    .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
    .messages(vec![
        Message::user(FIRST_TURN_PROMPT),
        first_assistant.clone(),
        tool_result.clone(),
    ])
    .options(generation.clone())
    .provider_options(provider.clone());
    let second = drain_stream(
        model
            .stream(second_request)
            .expect("second-turn stream should start"),
    )
    .await;
    capture_requests();
    assert_terminal(&second, FinishReason::Stop);
    assert!(
        second.text.contains(ALPHA_SIGNAL_OUTPUT),
        "second turn should use the tool result, got {:?}",
        second.text
    );

    // Turn 3: both prior assistant turns' rs_* items replay together.
    let second_assistant = Message::Assistant(second.head().with_content(second.choice.clone()));
    let third_request = CompletionRequest::new(
        "Repeat the exact tool output one more time, alone on a single line.",
    )
    .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
    .messages(vec![
        Message::user(FIRST_TURN_PROMPT),
        first_assistant,
        tool_result,
        Message::user(
            "Answer in one short sentence that includes the exact tool output. \
                 Do not call any tools.",
        ),
        second_assistant,
    ])
    .options(generation)
    .provider_options(provider);
    let third = drain_stream(
        model
            .stream(third_request)
            .expect("third-turn stream should start"),
    )
    .await;
    capture_requests();
    assert_terminal(&third, FinishReason::Stop);
    assert!(
        third.text.contains(ALPHA_SIGNAL_OUTPUT),
        "third turn should still carry the tool output, got {:?}",
        third.text
    );
    assert!(
        third.tool_calls.is_empty(),
        "third turn should not call tools"
    );
    assert_eq!(http.requests().len(), 3);
    assert_eq!(http.remaining_responses(), 0);
}
