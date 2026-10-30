use super::*;
use anyhow::{Context, ensure};
use message::{
    AssistantContent, AssistantMessage, CallId, CarriedPayload, CarriedReasoning,
    CarriedReasoningPart, CarriedText, CarriedTextContent, CustomToolCall, Message, ProviderItem,
    ToolCall, ToolFunction, ToolName, UnreplayableProviderItem, UnrepresentableToolCall,
};
use rig_core::completion::ToolDefinition;
use serde_json::{Value, json};

fn request_error(request: CompletionRequest) -> anyhow::Result<ProviderError> {
    match GenerateContent::new("gemini-test").encode(request, Mode::Unary) {
        Ok(request) => anyhow::bail!("the request must be refused, got: {request:?}"),
        Err(error) => Ok(error.into()),
    }
}

fn with_parts(parts: Vec<AssistantContent>) -> CompletionRequest {
    CompletionRequest::from(vec![Message::Assistant(AssistantMessage::new(parts))])
}

#[test]
fn an_unflattenable_tool_schema_is_a_request_failure() -> anyhow::Result<()> {
    let parameters = json!({
        "type": "object",
        "$defs": 5,
        "properties": {"a": {"$ref": "#/$defs/x"}},
    });
    let mut request = CompletionRequest::new("unused");
    request.tools = vec![ToolDefinition {
        name: ToolName::new("schema_tool").context("fixture name")?,
        description: String::new(),
        parameters,
    }];
    let error = request_error(request)?;
    ensure!(matches!(error, ProviderError::Request(_)), "{error:?}");
    ensure!(
        error.to_string()
            == "RequestError: Tool 'schema_tool' could not be converted to a schema: $defs must be an object",
        "schema refusal differs: {error:?}"
    );
    Ok(())
}

// Independent provider item IDs fc_1/ctc_1 have no current input field.
fn unrepresentable_calls() -> anyhow::Result<[(AssistantContent, &'static str); 2]> {
    Ok([
        (
            AssistantContent::ToolCall(
                ToolCall::from_wire(
                    "call_1",
                    ToolFunction::new(
                        ToolName::new("add").context("fixture name")?,
                        json!({"x": 1}),
                    ),
                )
                .with_namespace(Some("math".to_owned())),
            ),
            "cannot represent tool namespaces, so call `add` qualified by namespace `math`",
        ),
        (
            AssistantContent::CustomToolCall(CustomToolCall::new(
                CallId::from_wire("call_2"),
                ToolName::new("apply_patch").context("fixture name")?,
                r#"{"x": 1}"#,
            )),
            "carries JSON tool arguments only, so custom tool call `apply_patch`",
        ),
    ])
}

#[test]
fn grpc_refuses_namespaced_and_custom_calls() -> anyhow::Result<()> {
    for (content, refusal) in unrepresentable_calls()? {
        let error = request_error(with_parts(vec![content]))?;
        ensure!(
            matches!(&error, ProviderError::Request(_)),
            "a request refusal, got {error:?}"
        );
        let reason = error.to_string();
        ensure!(
            reason.starts_with("RequestError: Gemini generateContent "),
            "{reason}"
        );
        ensure!(reason.contains(refusal), "{reason}");
        let typed = match &error {
            ProviderError::Request(inner) => inner.downcast_ref::<UnrepresentableToolCall>(),
            _ => None,
        }
        .with_context(|| format!("the concrete refusal is kept: {error:?}"))?;
        match typed {
            UnrepresentableToolCall::Namespace {
                wire,
                name,
                namespace,
            } => ensure!(
                (*wire, name.as_str(), namespace.as_str())
                    == ("Gemini generateContent", "add", "math"),
                "namespace refusal differs: {typed:?}"
            ),
            UnrepresentableToolCall::CustomCall { wire, name } => ensure!(
                (*wire, name.as_str()) == ("Gemini generateContent", "apply_patch"),
                "custom call refusal differs: {typed:?}"
            ),
        }
    }
    Ok(())
}

#[test]
fn grpc_refuses_provider_items_by_name_for_known_and_unknown_issuers() -> anyhow::Result<()> {
    for issuer in [Some("openai"), None] {
        let content = AssistantContent::ProviderItem(ProviderItem {
            payload: CarriedPayload::new(
                json!({"type": "web_search_call", "id": "search_1"}),
                issuer.map(str::to_owned),
            ),
        });
        let error = request_error(with_parts(vec![content]))?;
        let refusal = match &error {
            ProviderError::Request(inner) => inner.downcast_ref::<UnreplayableProviderItem>(),
            _ => None,
        }
        .with_context(|| {
            format!("a request refusal that retains its shared concrete type: {error:?}")
        })?;
        ensure!(refusal.wire == "Gemini GenerateContent", "{refusal:?}");
        ensure!(refusal.issuer.as_deref() == issuer, "{refusal:?}");
        ensure!(
            refusal.item_type.as_deref() == Some("web_search_call"),
            "{refusal:?}"
        );
    }
    Ok(())
}

#[test]
fn opaque_responses_parts_are_refused_before_request_content_is_discarded() -> anyhow::Result<()> {
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
        let error = request_error(with_parts(vec![AssistantContent::text("known"), part]))?;
        ensure!(matches!(error, ProviderError::Request(_)), "{error:?}");
        ensure!(error.to_string().contains("opaque"), "{error:?}");
    }
    Ok(())
}
