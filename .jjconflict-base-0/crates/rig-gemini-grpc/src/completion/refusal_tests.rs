use super::*;
use message::{
    AssistantContent, AssistantMessage, CallId, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, CustomToolCall, Message, ProviderItem,
    ToolCall, ToolFunction, ToolName, UnreplayableProviderItem, UnrepresentableToolCall,
};
use rig_core::completion::ToolDefinition;
use serde_json::{Value, json};

fn request_error(request: CompletionRequest) -> ProviderError {
    GenerateContent::new("gemini-test")
        .encode(request, Mode::Unary)
        .expect_err("the request must be refused")
        .into()
}

fn with_parts(parts: Vec<AssistantContent>) -> CompletionRequest {
    CompletionRequest::from(vec![Message::Assistant(AssistantMessage::new(parts))])
}

#[test]
fn an_unflattenable_tool_schema_is_a_request_failure() {
    let parameters = json!({
        "type": "object",
        "$defs": 5,
        "properties": {"a": {"$ref": "#/$defs/x"}},
    });
    let mut request = CompletionRequest::new("unused");
    request.tools = vec![ToolDefinition {
        name: ToolName::new("schema_tool").expect("fixture name"),
        description: String::new(),
        parameters,
    }];
    let error = request_error(request);
    assert!(matches!(error, ProviderError::Request(_)), "{error:?}");
    assert_eq!(
        error.to_string(),
        "RequestError: Tool 'schema_tool' could not be converted to a schema: $defs must be an object"
    );
}

// Independent provider item IDs fc_1/ctc_1 have no current input field.
fn unrepresentable_calls() -> [(AssistantContent, &'static str); 2] {
    [
        (
            AssistantContent::ToolCall(
                ToolCall::from_wire(
                    "call_1",
                    ToolFunction::new(ToolName::new("add").expect("fixture name"), json!({"x": 1})),
                )
                .with_namespace(Some("math".to_owned())),
            ),
            "cannot represent tool namespaces, so call `add` qualified by namespace `math`",
        ),
        (
            AssistantContent::CustomToolCall(CustomToolCall::new(
                CallId::from_wire("call_2"),
                ToolName::new("apply_patch").expect("fixture name"),
                r#"{"x": 1}"#,
            )),
            "carries JSON tool arguments only, so custom tool call `apply_patch`",
        ),
    ]
}

#[test]
fn grpc_refuses_namespaced_and_custom_calls() {
    for (content, refusal) in unrepresentable_calls() {
        let error = request_error(with_parts(vec![content]));
        assert!(
            matches!(&error, ProviderError::Request(_)),
            "a request refusal, got {error:?}"
        );
        let reason = error.to_string();
        assert!(
            reason.starts_with("RequestError: Gemini generateContent "),
            "{reason}"
        );
        assert!(reason.contains(refusal), "{reason}");
        let typed = match &error {
            ProviderError::Request(inner) => inner.downcast_ref::<UnrepresentableToolCall>(),
            _ => None,
        }
        .expect("the concrete refusal is kept");
        match typed {
            UnrepresentableToolCall::Namespace {
                wire,
                name,
                namespace,
            } => assert_eq!(
                (*wire, name.as_str(), namespace.as_str()),
                ("Gemini generateContent", "add", "math"),
            ),
            UnrepresentableToolCall::CustomCall { wire, name } => assert_eq!(
                (*wire, name.as_str()),
                ("Gemini generateContent", "apply_patch"),
            ),
        }
    }
}

#[test]
fn grpc_refuses_provider_items_by_name_for_known_and_unknown_issuers() {
    for issuer in [Some("openai"), None] {
        let content = AssistantContent::ProviderItem(ProviderItem {
            payload: CarriedPayload::new(
                json!({"type": "web_search_call", "id": "search_1"}),
                issuer.map(str::to_owned),
            ),
        });
        let error = request_error(with_parts(vec![content]));
        let refusal = match &error {
            ProviderError::Request(inner) => inner.downcast_ref::<UnreplayableProviderItem>(),
            _ => None,
        }
        .expect("a request refusal that retains its shared concrete type");
        assert_eq!(refusal.wire, "Gemini GenerateContent");
        assert_eq!(refusal.issuer.as_deref(), issuer);
        assert_eq!(refusal.item_type.as_deref(), Some("web_search_call"));
    }
}

#[test]
fn opaque_responses_parts_are_refused_before_request_content_is_discarded() {
    let raw = json!({"type":"future_part","payload":{"list":[1,2]}});
    let opaque = AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(raw.clone(), None),
    ));
    let reasoning = |part, id: Option<&str>| {
        let mut carried = CarriedReasoning::new(
            vec![part],
            CarriedPayload::new(Value::Null, Some("openai".to_owned())),
        );
        carried.id = id.map(str::to_owned);
        AssistantContent::CarriedReasoning(carried)
    };
    for part in [
        opaque,
        reasoning(
            CarriedReasoningPart::OpaqueSummary(CarriedPayload::new(
                raw.clone(),
                Some("openai".to_owned()),
            )),
            None,
        ),
        reasoning(
            CarriedReasoningPart::OpaqueContent(CarriedPayload::new(
                raw,
                Some("openai".to_owned()),
            )),
            Some("rs_1"),
        ),
    ] {
        let error = request_error(with_parts(vec![AssistantContent::text("known"), part]));
        assert!(matches!(error, ProviderError::Request(_)));
        assert!(error.to_string().contains("opaque"));
    }
}
