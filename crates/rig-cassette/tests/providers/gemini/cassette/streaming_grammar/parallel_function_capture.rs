//! Evidence from the recorded parallel-call reply on the current request path.

use super::*;
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_test_support::cassette_models::GeminiModels;

#[tokio::test]
async fn capture_parallel_function_calls_on_current_request_with_recorded_reply() {
    const SCENARIO: &str = "streaming_grammar/parallel_function_calls";
    let replies = crate::cassettes::recorded_statuses_and_bodies("gemini", SCENARIO);
    let headers = crate::cassettes::recorded_response_header_pairs("gemini", SCENARIO);
    assert_eq!(replies.len(), 1, "one recorded exchange");
    assert_eq!(headers.len(), replies.len());
    let responses = replies
        .into_iter()
        .zip(headers)
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
    let directory = std::path::PathBuf::from(
        std::env::var_os("RIG_GEMINI_PARALLEL_CAPTURE_DIR")
            .expect("the assigned capture directory"),
    );
    let client = GeminiModels::new(gemini::GeminiConfig::new("[REDACTED]"), http.clone());
    let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
    let request = CompletionRequest::new(
        "Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them \
             together in this single reply, before writing any text. Emit the two \
             tool calls in one turn - do not wait for results between them.",
    )
    .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
    .tool(rig::tool::tool_definition(&AlphaSignal))
    .tool(rig::tool::tool_definition(&BetaSignal))
    .reasoning(rig::completion::Reasoning::Off);
    let run = drain_stream(model.stream(request).expect("stream should start")).await;
    let actual = http.requests();
    assert_eq!(actual.len(), 1, "one real model request");
    std::fs::write(directory.join("REQUEST-0.json"), &actual[0].body)
        .expect("retain the complete actual request");
    assert_eq!(
        http.remaining_responses(),
        0,
        "the original reply is consumed"
    );

    assert_terminal(&run, FinishReason::ToolCalls);
    let aggregated: Vec<&ToolCall> = run
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    for name in ["lookup_harbor_label", "lookup_orchard_label"] {
        let streamed = run
            .tool_calls
            .iter()
            .find(|call| call.function.name == name)
            .unwrap_or_else(|| panic!("stream should yield a {name} call"));
        let aggregated_call = aggregated
            .iter()
            .find(|call| call.function.name == name)
            .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
        // IDs derived from the recorded turn, never minted literally.
        assert_eq!(
            aggregated_call.id, streamed.id,
            "{name} id should aggregate"
        );
    }
    assert_eq!(
        aggregated.len(),
        2,
        "aggregated choice should contain exactly the two parallel calls"
    );
    // The recorded turn carries no wire ids, and rig no longer
    // fabricates durable identifiers from the wire (not from an
    // index, not from the tool name): both calls surface with a
    // minted id and no provider id, and stay distinct as parts —
    // two calls, in wire order, uncorrupted.
    assert!(aggregated[0].id.provider().is_none());
    assert!(aggregated[1].id.provider().is_none());
    assert!(aggregated[0].id.is_local());
    assert!(aggregated[1].id.is_local());
    assert_ne!(
        aggregated[0].id, aggregated[1].id,
        "each id-less call mints a unique durable id"
    );
    assert_ne!(
        aggregated[0].function.name, aggregated[1].function.name,
        "the two parallel calls stay distinct parts"
    );
}
