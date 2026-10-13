//! G224: existing selected decoder hooks are reached through real Models.
//! The scripted transport supplies exact strict frames; it sends no HTTP.

use super::*;
use crate::completion::{CompletionResponse, FinishReason};
use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::message::AssistantContent;
use crate::providers::openai::responses_api::streaming::IncompleteTerminal;
use crate::streaming::{Item, ReasoningAuthority, StreamEvent};
use futures::StreamExt;
use serde_json::{Value, json};

const TRANSPORT_ERROR: &str = "G224 terminal transport failure";

#[derive(Clone)]
struct Frames {
    frames: Vec<Value>,
    terminal_error: bool,
}

impl<W> Transport<W> for Frames
where
    W: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame>,
{
    fn send(&self, _request: Encoded, _exchange: Exchange) -> Opening<WireFrame> {
        let mut frames = self.frames.iter()
            .map(|frame| Ok(WireFrame::Text(frame.to_string())))
            .collect::<Vec<_>>();
        if self.terminal_error {
            frames.push(Err(ProviderError::Response(TRANSPORT_ERROR.to_owned())));
        }
        Opening::ready(Opened::new(futures::stream::iter(frames)))
    }
}

struct Observed {
    items: Vec<Item<StreamEvent>>,
    errors: Vec<ProviderError>,
    ends_at_errors: Vec<usize>,
    partial: CompletionResponse,
    finish: Result<CompletionResponse, ProviderError>,
}

async fn streamed<W>(wire: W, frames: Vec<Value>, terminal_error: bool) -> Observed
where
    W: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame>,
{
    let model = Model::new(wire, Frames { frames, terminal_error });
    let mut stream = model.stream(CompletionRequest::new("facade probe"))
        .expect("the real Model encodes the request");
    let mut items = Vec::new();
    let mut errors = Vec::new();
    let mut ends_at_errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(item) => items.push(item),
            Err(error) => {
                ends_at_errors.push(items.iter().filter(|item|
                    matches!(item, Item::Event(StreamEvent::End { .. }))).count());
                errors.push(error);
            }
        }
    }
    let partial = stream.partial();
    let finish = stream.finish().await;
    Observed { items, errors, ends_at_errors, partial, finish }
}

fn direct_chat() -> Chat {
    OpenAIConfig::new("sk-test").chat("gpt-4o-mini")
}

fn direct_responses() -> Responses {
    OpenAIConfig::new("sk-test").responses("gpt-5.2")
}

fn response_text_delta() -> Value {
    json!({"type":"response.output_text.delta","sequence_number":0,
        "output_index":0,"content_index":0,"item_id":"msg_facade","delta":"partial"})
}

fn response_incomplete() -> Value {
    json!({"type":"response.incomplete","sequence_number":1,"response":{
        "id":"resp_facade","object":"response","created_at":0,"model":"gpt-5.2",
        "status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
        "output":[{"type":"message","id":"msg_facade","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":"partial","annotations":[]}]}],
        "usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
    }})
}

fn response_reasoning_delta() -> Value {
    json!({"type":"response.reasoning_summary_text.delta","sequence_number":0,
        "output_index":0,"summary_index":0,"item_id":"rs_1","delta":"thought"})
}

fn assert_response_equal(actual: &CompletionResponse, expected: &CompletionResponse) {
    assert_eq!(actual, expected);
}

fn assert_partial_text(response: &CompletionResponse) {
    assert_eq!(response.choice.len(), 1);
    match &response.choice[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "partial"),
        AssistantContent::CarriedText(text) => assert_eq!(text.content,
            crate::message::CarriedTextContent::Text("partial".to_owned())),
        other => panic!("expected the actual known output text, got {other:?}"),
    }
}

#[tokio::test]
async fn facade_forwards_responses_mode_and_explicit_incomplete_policy() {
    let frames = vec![response_text_delta(), response_incomplete()];
    // The same strict tagged frames deliberately enter a real unary Model
    // and a streaming Model, so this probes decoder_in, not an HTTP claim.
    for policy in [IncompleteTerminal::Refuse, IncompleteTerminal::Accept] {
        let direct = direct_responses().with_streamed_incomplete(policy);
        let own = Model::new(direct.clone(), Frames { frames: frames.clone(), terminal_error: false })
            .call(CompletionRequest::new("facade probe")).await
            .expect("unary incomplete is accepted independently of streamed policy");
        let routed = Model::new(OpenAiWire::Responses(direct.clone()),
            Frames { frames: frames.clone(), terminal_error: false })
            .call(CompletionRequest::new("facade probe")).await
            .expect("the facade forwards unary mode");
        assert_response_equal(&routed, &own);
        assert_partial_text(&routed);
        assert_eq!(routed.finish_reason(), Some(FinishReason::Length));
        assert_eq!(routed.raw, response_incomplete()["response"]);

        let own = streamed(direct.clone(), frames.clone(), false).await;
        let routed = streamed(OpenAiWire::Responses(direct), frames.clone(), false).await;
        assert_eq!(routed.items, own.items);
        assert_response_equal(&routed.partial, &own.partial);
        if policy == IncompleteTerminal::Accept {
            assert!(own.errors.is_empty() && routed.errors.is_empty());
            let own = own.finish.expect("direct explicit Accept keeps a genuine terminal");
            let routed = routed.finish.expect("facade explicit Accept keeps the terminal");
            assert_response_equal(&routed, &own);
            assert_partial_text(&routed);
            assert_eq!(routed.finish_reason(), Some(FinishReason::Length));
            assert_eq!(routed.raw, response_incomplete()["response"]);
        } else {
            let raw = response_incomplete().to_string();
            for observed in [own, routed] {
                assert_eq!(observed.errors.len(), 1);
                assert_eq!(observed.errors[0].provider_response_body(), Some(raw.as_str()));
                let error = observed.finish.expect_err("default streamed Refuse remains a failure");
                assert_eq!(error.provider_response_body(), Some(raw.as_str()));
                assert_partial_text(&observed.partial);
            }
        }
    }
}

#[tokio::test]
async fn facade_forwards_chat_unary_empty_response_mode_without_changing_stream_truncation() {
    // No frame arrived. Only the real decoder's selected mode distinguishes
    // an empty unary Response error from a cut-off streamed reply.
    let own = Model::new(direct_chat(), Frames { frames: vec![], terminal_error: false })
        .call(CompletionRequest::new("facade probe")).await.expect_err("direct empty unary");
    let routed = Model::new(OpenAiWire::Chat(direct_chat()), Frames { frames: vec![], terminal_error: false })
        .call(CompletionRequest::new("facade probe")).await.expect_err("facade empty unary");
    for error in [own, routed] {
        assert!(matches!(error, ProviderError::Response(ref message)
            if message == crate::message::EMPTY_RESPONSE_ERROR));
    }
    let own = streamed(direct_chat(), vec![], false).await;
    let routed = streamed(OpenAiWire::Chat(direct_chat()), vec![], false).await;
    assert_eq!(routed.items, own.items);
    for observed in [own, routed] {
        assert!(observed.items.is_empty());
        assert_eq!(observed.errors.len(), 1);
        assert!(matches!(observed.errors[0], ProviderError::Truncated));
        assert!(matches!(observed.finish, Err(ProviderError::Truncated)));
    }
}

#[tokio::test]
async fn facade_flushes_delivered_responses_reasoning_before_the_original_transport_error() {
    let frames = vec![response_reasoning_delta()];
    let own = streamed(direct_responses(), frames.clone(), true).await;
    let routed = streamed(OpenAiWire::Responses(direct_responses()), frames, true).await;
    assert_eq!(routed.items, own.items);
    assert_response_equal(&routed.partial, &own.partial);
    for observed in [own, routed] {
        assert_eq!(observed.errors.len(), 1);
        assert_eq!(observed.ends_at_errors, vec![1], "delivered End precedes transport failure");
        assert!(matches!(&observed.errors[0], ProviderError::Response(message) if message == TRANSPORT_ERROR));
        assert!(matches!(&observed.finish, Err(ProviderError::Response(message)) if message == TRANSPORT_ERROR));
        let ends = observed.items.iter().filter_map(|item| match item {
            Item::Event(StreamEvent::End { content, reasoning_authority, .. }) => Some((content, reasoning_authority)),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        let (AssistantContent::CarriedReasoning(reasoning), authority) = ends[0] else {
            panic!("the actual flushed reasoning carrier");
        };
        assert_eq!(*authority, Some(ReasoningAuthority::Partial));
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert_eq!(reasoning.display_text(), "thought");
        assert_eq!(reasoning.payload.item["summary"], json!([{"type":"summary_text","text":"thought"}]));
    }
}

#[tokio::test]
async fn facade_flushes_delivered_chat_call_before_the_original_transport_error() {
    let frames = vec![json!({"id":"chat_facade","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini",
        "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",
            "function":{"name":"add","arguments":"{\"x\":1}"}}]},"finish_reason":null}]})];
    let own = streamed(direct_chat(), frames.clone(), true).await;
    let routed = streamed(OpenAiWire::Chat(direct_chat()), frames, true).await;
    assert_eq!(routed.items, own.items);
    assert_response_equal(&routed.partial, &own.partial);
    for observed in [own, routed] {
        assert_eq!(observed.errors.len(), 1);
        assert_eq!(observed.ends_at_errors, vec![1]);
        assert!(matches!(&observed.errors[0], ProviderError::Response(message) if message == TRANSPORT_ERROR));
        assert!(matches!(&observed.finish, Err(ProviderError::Response(message)) if message == TRANSPORT_ERROR));
        let ends = observed.items.iter().filter_map(|item| match item {
            Item::Event(StreamEvent::End { content: AssistantContent::ToolCall(call), .. }) => Some(call),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].id.provider().expect("received call").as_str(), "call_1");
        assert_eq!(ends[0].function.name.as_str(), "add");
        assert_eq!(ends[0].function.arguments_value(), json!({"x":1}));
        assert_eq!(ends[0].function.invalid_arguments, None);
        assert!(ends[0].native.is_none(), "delivered JSON is not native terminal authority");
    }
}

#[tokio::test]
async fn facade_keeps_responses_partial_eof_polling_without_fabricating_a_finish() {
    let frames = vec![response_reasoning_delta()];
    let own = streamed(direct_responses(), frames.clone(), false).await;
    let routed = streamed(OpenAiWire::Responses(direct_responses()), frames, false).await;
    assert_eq!(routed.items, own.items);
    assert_response_equal(&routed.partial, &own.partial);
    for observed in [own, routed] {
        assert!(observed.errors.is_empty(), "KeepPartial EOF ends polling silently");
        assert!(observed.ends_at_errors.is_empty());
        assert!(matches!(observed.finish, Err(ProviderError::Truncated)), "no genuine provider end arrived");
        let ends = observed.items.iter().filter_map(|item| match item {
            Item::Event(StreamEvent::End { content, reasoning_authority, .. }) => Some((content, reasoning_authority)),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        let (AssistantContent::CarriedReasoning(reasoning), authority) = ends[0] else {
            panic!("the actual EOF End retains its carried reasoning evidence");
        };
        assert_eq!(*authority, Some(ReasoningAuthority::Partial));
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert_eq!(reasoning.display_text(), "thought");
        assert_eq!(reasoning.payload.item["summary"], json!([{"type":"summary_text","text":"thought"}]));
        // Removing unfinished Native does not convert an independent carrier.
        let [AssistantContent::CarriedReasoning(partial)] = observed.partial.choice.as_slice() else {
            panic!("the partial retains actual independent carried reasoning");
        };
        assert_eq!(partial.id.as_deref(), Some("rs_1"));
        assert_eq!(partial.display_text(), "thought");
        assert_eq!(partial.payload.item["summary"], json!([{"type":"summary_text","text":"thought"}]));
        assert_eq!(partial.payload.provider.as_deref(), Some(crate::providers::openai::wire::OPENAI.name));
        assert!(observed.partial.choice[0].native_item().is_none());
        assert_eq!(observed.partial.finish_reason(), None);
    }
}
