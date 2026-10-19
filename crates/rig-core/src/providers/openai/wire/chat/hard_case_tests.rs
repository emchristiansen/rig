//! The chat stream's hard cases, each a reply the shared pending-call buffer
//! must assemble: ids and names that arrive late or never, a reused index,
//! interleaved indices, whole calls in one chunk, `null` placeholders, a
//! length cut inside a call, late and repeated reasoning signatures, and a
//! provider id sent twice.

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::*;
use crate::completion::CompletionResponse;
use crate::message::{AssistantContent, CallId};
use crate::providers::openai::wire::{Dialect, OPENAI, OPENROUTER, OpenAIConfig};
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::test_utils::MockStreamingClient;

fn wire(dialect: &'static Dialect) -> Chat {
    OpenAIConfig::new("sk-test")
        .with_dialect(dialect)
        .chat("gpt-4.1-nano")
}

/// One `chat.completion.chunk` as an SSE event.
fn chunk(delta: Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": "chatcmpl-hard",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    });
    format!("data: {chunk}\n\n")
}

/// A `tool_calls` fragment; an absent field is the wire's `null`.
fn call(index: u32, id: Option<&str>, name: Option<&str>, arguments: Option<&str>) -> String {
    chunk(
        json!({"tool_calls": [{
            "index": index,
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }]}),
        None,
    )
}

fn text(text: &str) -> String {
    chunk(json!({"content": text}), None)
}

fn finish(reason: &str) -> String {
    chunk(json!({}), Some(reason))
}

const DONE: &str = "data: [DONE]\n\n";

/// Stream `chunks` through `dialect`'s chat wire: the items, then what
/// `finish` returns.
async fn stream(
    dialect: &'static Dialect,
    chunks: &[String],
) -> (
    Vec<Result<Item<StreamEvent>, ProviderError>>,
    Result<CompletionResponse, ProviderError>,
) {
    let model = crate::driver::Model::new(
        wire(dialect),
        MockStreamingClient {
            sse_bytes: Bytes::from(chunks.concat()),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hard case"))
        .expect("the stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    (items, stream.finish().await)
}

fn signature(detail: &str) -> String {
    chunk(
        json!({"reasoning_details": [{
            "type": "reasoning.text",
            "text": "",
            "signature": detail,
            "index": 0,
        }]}),
        None,
    )
}

/// A message's reasoning is one block, as pi keeps it, so a signature that
/// arrives after the answer joins the reasoning before it, and the message
/// sent back holds the reasoning and the signature as one message did.
#[tokio::test]
async fn openrouter_a_late_signature_joins_the_reasoning_before_the_answer() {
    let (_, response) = stream(
        &OPENROUTER,
        &[
            chunk(json!({"reasoning": "thinking"}), None),
            text("answer"),
            signature("sig-late"),
            finish("stop"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [
        AssistantContent::CarriedReasoning(first),
        AssistantContent::Text(answer),
    ] = response.choice.as_slice()
    else {
        panic!("the reasoning and the answer: {:?}", response.choice);
    };
    assert_eq!(first.display_text(), "thinking");
    assert_eq!(
        first.content,
        vec![crate::message::CarriedReasoningPart::Text {
            text: "thinking".to_owned(),
            signature: Some("sig-late".to_owned())
        }]
    );
    assert_eq!(answer.text, "answer");
    let signed =
        json!([{"type": "reasoning.text", "text": "", "signature": "sig-late", "index": 0}]);
    assert_eq!(first.payload.item["reasoning_details"], signed);
    let turn = crate::message::AssistantMessage {
        content: response.choice.clone(),
        ..response.head()
    };
    let sent = super::tests::replayed(&wire(&OPENROUTER), turn);
    assert_eq!(sent["reasoning_details"][0]["text"], "thinking");
    assert_eq!(sent["reasoning_details"][0]["signature"], "sig-late");
}

/// Reasoning that arrives after the answer is still the turn's first block,
/// while the events it streams keep their arrival order.
#[tokio::test]
async fn openrouter_late_reasoning_is_stored_first_and_streamed_in_arrival_order() {
    let (items, response) = stream(
        &OPENROUTER,
        &[
            text("answer"),
            chunk(json!({"reasoning": "late"}), None),
            finish("stop"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    assert_eq!(
        response
            .choice
            .iter()
            .map(AssistantContent::canonical)
            .collect::<Vec<_>>(),
        [
            AssistantContent::reasoning("late"),
            AssistantContent::text("answer")
        ]
    );
    let starts: Vec<&PartKind> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::Start { kind, .. })) => Some(kind),
            _ => None,
        })
        .collect();
    assert!(
        matches!(starts.as_slice(), [PartKind::Text, PartKind::Reasoning]),
        "{starts:?}"
    );
}

/// pi's merge: a second signature for the same detail fills nothing, as the
/// first already signed it.
#[tokio::test]
async fn openrouter_a_second_signature_for_one_detail_keeps_the_first() {
    let (_, response) = stream(
        &OPENROUTER,
        &[
            chunk(json!({"reasoning": "thinking"}), None),
            signature("sig-1"),
            signature("sig-2"),
            text("answer"),
            finish("stop"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [
        AssistantContent::CarriedReasoning(only),
        AssistantContent::Text(_),
    ] = response.choice.as_slice()
    else {
        panic!("one carried reasoning part: {:?}", response.choice);
    };
    assert_eq!(
        only.payload.item["reasoning_details"],
        json!([{"type": "reasoning.text", "text": "", "signature": "sig-1", "index": 0}])
    );
    assert_eq!(
        only.content,
        vec![crate::message::CarriedReasoningPart::Text {
            text: "thinking".to_owned(),
            signature: Some("sig-1".to_owned())
        }]
    );
}

#[tokio::test]
async fn a_provider_id_sent_twice_is_renamed() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, Some("call_same"), Some("add"), Some(r#"{"a":1}"#)),
            call(1, Some("call_same"), Some("add"), Some(r#"{"a":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("a duplicate id does not fail the reply");
    let ids: Vec<&CallId> = response.tool_calls().map(|call| &call.id).collect();
    let [first, second] = ids.as_slice() else {
        panic!("both calls are kept: {ids:?}");
    };
    assert_eq!(**first, CallId::from_wire("call_same"));
    assert!(matches!(second, CallId::Local(_)), "{second:?}");
}
