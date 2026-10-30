//! A full WebSocket frame captures its delta context from the merged upper
//! layers, and refuses what cannot seed or lower one.
#![cfg(feature = "websocket")]

use std::collections::HashSet;

use serde_json::{Value, json};

use super::websocket::{
    InputDelta, InputDeltaItem, WebSocketCreate, WebSocketCreateInput, WireDelta,
};
use super::{DeltaContext, InputRequirement, wire::Responses};
use crate::completion::{CompletionRequest, ProviderOptions, options::FinalBody};
use crate::error::EncodeError;
use crate::message::{DocumentSourceKind, Image, Message, UserContent};
use crate::providers::openai::OpenAIConfig;

fn target() -> Responses {
    OpenAIConfig::new("test-key").responses("gpt-5.4")
}

fn create(input: WebSocketCreateInput) -> WebSocketCreate {
    WebSocketCreate {
        input,
        stamps: Default::default(),
        identity: None,
        lite: false,
        automatic_previous: None,
        generate: None,
    }
}

fn prepare(request: &CompletionRequest) -> Result<(FinalBody, DeltaContext), EncodeError> {
    target().prepare_websocket_request_with_identity(
        request,
        None,
        InputRequirement::NonEmpty,
        create(WebSocketCreateInput::Full),
    )
}

/// The full turn with `fields` as the OpenAI provider options' shared section.
fn with_provider_fields(fields: Value) -> CompletionRequest {
    let mut request = CompletionRequest::new("full turn");
    request.provider_options = serde_json::from_value::<ProviderOptions>(json!({
        (crate::providers::openai::PROVIDER_NAME): {"*": fields}
    }))
    .expect("an object of object sections");
    request
}

#[test]
fn a_delta_create_cannot_seed_a_delta_context() {
    let request = CompletionRequest::new("full turn");
    let (envelope, _) = prepare(&request).expect("the full request captures its context");
    let error = target()
        .prepare_websocket_request_with_identity(
            &request,
            None,
            InputRequirement::NonEmpty,
            create(WebSocketCreateInput::Delta {
                envelope,
                input: WireDelta::new(vec![
                    json!({"type": "message", "role": "user", "content": "next"}),
                ])
                .expect("the wire delta is nonempty"),
                previous_response_id: "resp_1".to_owned(),
            }),
        )
        .expect_err("only a full request lowers the context a delta reuses");
    assert_eq!(
        error.to_string(),
        "RequestError: a websocket delta context must be captured from a full request"
    );
}

#[test]
fn a_provider_option_model_is_the_captured_model() {
    let (body, context) = prepare(&with_provider_fields(json!({"model": "provider-model"})))
        .expect("a string model seeds the context");
    assert_eq!(body.get("model"), Some(&json!("provider-model")));
    assert_eq!(context.model, "provider-model");
}

#[test]
fn a_provider_option_model_that_is_not_a_string_is_refused() {
    let error = prepare(&with_provider_fields(json!({"model": 7})))
        .expect_err("a delta context names its model as a string");
    assert_eq!(
        error.to_string(),
        "RequestError: the full websocket request's model must be a string"
    );
}

#[test]
fn provider_option_tools_name_the_captured_custom_tools() {
    let tools = json!([
        {"type": "custom", "name": "lookup", "format": {"type": "text"}},
        {
            "type": "function", "name": "post_to_bus", "description": "", "strict": false,
            "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
        }
    ]);
    let (body, context) = prepare(&with_provider_fields(json!({"tools": tools.clone()})))
        .expect("declared provider tools seed the context");
    assert_eq!(body.get("tools"), Some(&tools));
    assert_eq!(context.custom, HashSet::from(["lookup".to_owned()]));
}

#[test]
fn provider_option_tools_that_are_not_an_array_are_refused() {
    let error = prepare(&with_provider_fields(json!({"tools": {"type": "custom"}})))
        .expect_err("the context reads only declared tool arrays");
    assert_eq!(
        error.to_string(),
        "RequestError: `tools` must be a JSON array of tool declarations"
    );
}

#[test]
fn a_canonical_delta_message_the_wire_cannot_carry_is_refused() {
    let (_, context) = prepare(&CompletionRequest::new("full turn"))
        .expect("the full request captures its context");
    let delta = InputDelta::one(InputDeltaItem::Canonical(Message::User {
        content: vec![UserContent::Image(Image {
            data: DocumentSourceKind::Unknown,
            ..Default::default()
        })],
    }));
    let error = target()
        .lower_websocket_delta(delta, &context)
        .expect_err("an unsourced image has no Responses input part");
    assert_eq!(
        error.to_string(),
        "RequestError: the Responses API cannot carry this image; prepare the request first"
    );
}
