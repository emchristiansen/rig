use serde_json::{Value, json};

use super::websocket::{InputDelta, InputDeltaItem, WebSocketCreate, WebSocketCreateInput};
use super::{DeltaContext, InputRequirement, codex_identity::CodexIdentity, wire::Responses};
use crate::completion::{CompletionRequest, options::FinalBody};
use crate::message::{
    AssistantContent, AssistantMessage, ImageDetail, Message, Opaque, Reasoning, ToolCall,
    ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::providers::openai::{OpenAIConfig, wire::CallerIdentity};

fn full(target: &Responses, request: &CompletionRequest) -> (FinalBody, DeltaContext) {
    let identity = target.codex_identity.as_ref();
    target
        .prepare_websocket_request_with_identity(
            request,
            identity,
            InputRequirement::NonEmpty,
            WebSocketCreate {
                input: WebSocketCreateInput::Full,
                stamps: Default::default(),
                identity: identity.cloned(),
                lite: identity.is_some() && target.codex_request_shape.is_lite(),
                automatic_previous: None,
                generate: None,
            },
        )
        .expect("the full request captures its lowering context")
}

fn lite() -> Responses {
    let mut provider = OpenAIConfig::with_key(&crate::providers::chatgpt::DIALECT, "test-key")
        .with_caller_identity(
            CallerIdentity::new("fixture-origin", "fixture-agent/1.0", None)
                .expect("caller identity is header-safe"),
        );
    provider.instructions = None;
    provider
        .responses("wire-default-model")
        .with_codex_identity(
            CodexIdentity::from_ids("session-a", "thread-a").expect("identity is header-safe"),
        )
        .expect("the Codex wire accepts its identity")
        .with_responses_lite()
        .expect("the Codex wire supports Lite")
}

fn call(name: &str) -> ToolCall {
    ToolCall::from_wire(
        "",
        ToolFunction::new(
            ToolName::new(name).expect("the tool name is nonempty"),
            json!({"input": "run"}),
        ),
    )
}

#[test]
fn lite_delta_lowers_canonical_image_detail_and_preserves_identical_raw_input() {
    let target = lite();
    let (_, context) = full(&target, &CompletionRequest::new("full turn"));
    let raw = json!({
        "type": "message", "role": "user",
        "content": [{
            "type": "input_image", "image_url": "https://example.test/image.png",
            "detail": "high"
        }],
    });
    let delta = InputDelta::new(vec![
        InputDeltaItem::Canonical(Message::User {
            content: vec![UserContent::image_url(
                "https://example.test/image.png",
                None,
                Some(ImageDetail::High),
            )],
        }),
        InputDeltaItem::Raw(raw.clone()),
    ])
    .expect("the source delta is nonempty");
    let items = target
        .lower_websocket_delta(delta, &context)
        .expect("the mixed delta lowers")
        .into_vec();
    let mut canonical = raw.clone();
    canonical["content"][0]
        .as_object_mut()
        .expect("the image part is an object")
        .shift_remove("detail");
    assert_eq!(items, vec![canonical, raw]);
}

#[test]
fn delta_uses_the_captured_model_custom_tools_and_stateless_context_before_lite() {
    let target = lite();
    let mut request = CompletionRequest::new("full turn");
    request.model = Some("request-model".to_owned());
    request.additional_params = Some(json!({
        "model": "resolved-model",
        "tools": [{"type": "custom", "name": "execute", "format": {"type": "text"}}]
    }));
    let (body, context) = full(&target, &request);
    assert_eq!(body.get("model"), Some(&json!("request-model")));
    assert_eq!(context.model, "request-model");
    assert!(context.custom.contains("execute"));
    assert!(context.stateless);
    assert!(
        body.get("tools").is_none(),
        "Lite moved the tools into input"
    );

    let reasoning = AssistantContent::Reasoning(Reasoning::new("unresolvable reasoning"))
        .with_native(json!({
            "type": "reasoning", "id": "rs_no_ciphertext",
            "summary": [{"type": "summary_text", "text": "unresolvable reasoning"}]
        }));
    let delta = InputDelta::one(InputDeltaItem::Canonical(Message::Assistant(
        AssistantMessage::new(vec![reasoning, AssistantContent::ToolCall(call("execute"))]),
    )));
    let items = target
        .lower_websocket_delta(delta, &context)
        .expect("the canonical call uses the captured custom declaration")
        .into_vec();
    assert_eq!(
        items,
        vec![json!({
            "type": "custom_tool_call", "call_id": "tool-0", "name": "execute", "input": "run"
        })]
    );
}

#[test]
fn typed_selected_model_is_captured_instead_of_the_wire_default() {
    let target = lite();
    let mut request = CompletionRequest::new("full turn");
    request.model = Some("typed-selected-model".to_owned());
    let (body, context) = full(&target, &request);
    assert_eq!(body.get("model"), Some(&json!("typed-selected-model")));
    assert_eq!(context.model, "typed-selected-model");
}

#[test]
fn one_delta_batch_keeps_assistant_positions_call_pairing_raw_order_and_system_input() {
    let target = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let (body, context) = full(
        &target,
        &CompletionRequest::new("full turn").preamble("captured instructions"),
    );
    let first = call("function");
    let second = call("function");
    let raw_first = json!({"type": "future_item", "id": "literal", "extra": [null, 1]});
    let raw_second = json!(["literal", {"nested": null}]);
    let delta = InputDelta::new(vec![
        InputDeltaItem::Canonical(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("first"),
            AssistantContent::ToolCall(first.clone()),
        ]))),
        InputDeltaItem::Raw(raw_first.clone()),
        InputDeltaItem::Canonical(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("second"),
            AssistantContent::ToolCall(second.clone()),
        ]))),
        InputDeltaItem::Raw(raw_second.clone()),
        InputDeltaItem::Canonical(Message::tool_results(vec![
            second.result(vec![ToolResultContent::text("second result")]),
            first.result(vec![ToolResultContent::text("first result")]),
        ])),
        InputDeltaItem::Canonical(Message::system("delta system")),
    ])
    .expect("the source delta is nonempty");
    let items = target
        .lower_websocket_delta(delta, &context)
        .expect("one batch pairs every canonical call and result")
        .into_vec();
    assert_eq!(items.len(), 9);
    assert_eq!(items[0]["id"], "msg_rig_0");
    assert_eq!(items[0]["content"][0]["text"], "first");
    assert_eq!(items[1]["call_id"], "tool-0");
    assert_eq!(items[2], raw_first);
    assert_eq!(items[3]["id"], "msg_rig_1");
    assert_eq!(items[3]["content"][0]["text"], "second");
    assert_eq!(items[4]["call_id"], "tool-1");
    assert_eq!(items[5], raw_second);
    assert_eq!(items[6]["call_id"], "tool-1");
    assert_eq!(items[6]["output"], "second result");
    assert_eq!(items[7]["call_id"], "tool-0");
    assert_eq!(items[7]["output"], "first result");
    assert_eq!(
        items[8],
        json!({
            "type": "message", "role": "system",
            "content": [{"type": "input_text", "text": "delta system"}]
        })
    );
    assert_eq!(
        body.get("instructions"),
        Some(&json!("captured instructions"))
    );
}

#[test]
fn delta_refuses_zero_wire_output_and_retains_literal_raw_values() {
    let target = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let (_, context) = full(&target, &CompletionRequest::new("full turn"));
    for message in [
        Message::user(" \n\t"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::Opaque(
            Opaque {
                item: json!({"type": "inspection_only", "data": "kept for inspection"}),
                replay: false,
            },
        )])),
    ] {
        let error = target
            .lower_websocket_delta(
                InputDelta::one(InputDeltaItem::Canonical(message)),
                &context,
            )
            .expect_err("a nonempty source may still produce no wire input");
        assert!(error.to_string().contains("incremental delta"));
    }
    let delta = InputDelta::new(vec![
        InputDeltaItem::Canonical(Message::user(" ")),
        InputDeltaItem::Raw(Value::Null),
    ])
    .expect("the source delta is nonempty");
    assert_eq!(
        target
            .lower_websocket_delta(delta, &context)
            .expect("a literal raw item remains a wire item")
            .into_vec(),
        vec![Value::Null]
    );
}
