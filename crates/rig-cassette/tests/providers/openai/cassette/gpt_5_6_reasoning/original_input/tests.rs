use super::super::FIVE_TURN_PROMPTS;
use super::*;
use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::driver::Model;
use rig::operation::Completion;
use rig::providers::openai::{self, OpenAIConfig, responses_api::wire::Responses};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig::wire::{Body, Mode, Operation, Wire};
use serde_json::{Value, json};

#[tokio::test]
async fn sparse_history_input_preserves_the_genuine_stored_turn() {
    let scenario = "gpt_5_6_reasoning/five_turn_streaming_metadata_roundtrip";
    let bodies = crate::cassettes::recorded_interaction_bodies("openai", scenario);
    assert_eq!(bodies.len(), 5);
    let first = bodies.first().expect("the original first interaction");
    let second = bodies.get(1).expect("the original second interaction");
    let statuses = crate::cassettes::recorded_statuses_and_bodies("openai", scenario);
    assert_eq!(statuses.first().expect("the original status").0, 200);
    let recorded_headers = crate::cassettes::recorded_response_header_pairs("openai", scenario);
    let mut headers = http::HeaderMap::new();
    for (name, value) in recorded_headers
        .first()
        .expect("the original reply headers")
    {
        headers.append(
            http::HeaderName::from_bytes(name.as_bytes()).expect("recorded header name"),
            http::HeaderValue::from_str(value).expect("recorded header value"),
        );
    }
    let http = SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        first.1.clone().into(),
        headers,
    )]);
    let wire = Responses::new(
        OpenAIConfig::new("original-input-control"),
        openai::GPT_5_6_SOL,
    );
    let model = Model::new(wire.clone(), http);
    let first_user = Message::user(FIVE_TURN_PROMPTS.first().expect("first prompt").0);
    let mut stream = model
        .stream(
            CompletionRequest::new(first_user.clone()).additional_params(json!({"reasoning": {
                "context":"all_turns", "effort":"low", "mode":"pro"
            }})),
        )
        .expect("the original first response opens");
    while let Some(item) = stream.next().await {
        item.expect("the genuine recorded first stream");
    }
    let final_response = stream
        .finish()
        .await
        .expect("the actual fold returns the genuine response");
    let turn = StoredStreamingTurn {
        user: first_user,
        assistant: Message::Assistant(
            final_response
                .head()
                .with_content(final_response.choice.clone()),
        ),
        final_response,
    };
    let before = serde_json::to_value(&turn).expect("the whole genuine turn serializes");
    let Message::Assistant(genuine_head) = &turn.assistant else {
        panic!("the genuine stored assistant");
    };
    let [AssistantContent::CarriedText(source)] = genuine_head.content.as_slice() else {
        panic!("the original first response has one known carried text part");
    };
    assert_eq!(source.content, CarriedTextContent::Text("ACK-1".into()));
    assert_eq!(
        source.item_id.as_deref(),
        Some("msg_0f74e25a348a327b006ab396ab6fcc87d08f10b30219276532")
    );
    assert_eq!(source.phase.as_deref(), Some("final_answer"));
    assert_eq!(
        source.payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: 0
        }
    );
    assert!(source.payload.provider.is_some());
    assert_eq!(
        source.payload.item,
        json!({
            "type":"output_text", "text":"ACK-1", "annotations":[], "logprobs":[]
        })
    );

    let outgoing = assistant(&turn);
    let Message::Assistant(projected_head) = &outgoing else {
        panic!("the fresh outgoing assistant");
    };
    assert_eq!(projected_head.origin, genuine_head.origin);
    assert_eq!(projected_head.stop, genuine_head.stop);
    let [AssistantContent::CarriedText(projected)] = projected_head.content.as_slice() else {
        panic!("one freshly authored known-text part");
    };
    assert_eq!(projected.content, source.content);
    assert_eq!(projected.item_id, source.item_id);
    assert_eq!(projected.payload.position, source.payload.position);
    assert!(projected.phase.is_none());
    assert!(projected.payload.provider.is_none());
    assert!(projected.unknown_message_fields().is_none());
    assert!(
        projected_head
            .content
            .first()
            .expect("projected known text")
            .native_item()
            .is_none()
    );
    let mut expected = serde_json::to_value(&turn.assistant).expect("the genuine head serializes");
    expected["content"] = json!([{
        "type":"carried_text", "content":{"type":"text", "text":"ACK-1"},
        "payload":{
            "item":{"type":"output_text", "text":"ACK-1"},
            "position":{"kind":"message_content", "item_index":0, "part_index":0}
        },
        "item_id":"msg_0f74e25a348a327b006ab396ab6fcc87d08f10b30219276532"
    }]);
    assert_eq!(
        serde_json::to_value(&outgoing).expect("full projected shape"),
        expected
    );
    assert_eq!(
        serde_json::to_value(&turn).expect("the genuine stored data remain serializable"),
        before
    );

    let request = CompletionRequest::new(FIVE_TURN_PROMPTS.get(1).expect("second prompt").0)
        .messages([turn.user.clone(), outgoing])
        .additional_params(json!({"reasoning": {
            "context":"all_turns", "effort":"low", "mode":"pro"
        }}));
    let prepared =
        Completion::prepare(request, &wire.describe()).expect("the original sparse input prepares");
    let encoded = wire
        .encode(prepared, Mode::Streaming)
        .expect("the encoder must accept the sparse original input");
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("the original Responses request is a byte body");
    };
    let actual: Value = serde_json::from_slice(bytes).expect("the outgoing request JSON");
    let original: Value =
        serde_json::from_str(&second.0).expect("the full original second request JSON");
    assert_eq!(actual, original);
    assert_eq!(
        serde_json::to_value(&turn).expect("encoding preserves genuine stored data"),
        before
    );
}
