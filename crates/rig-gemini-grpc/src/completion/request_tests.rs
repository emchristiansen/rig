use super::*;
use crate::proto;
use anyhow::{Context, ensure};
use message::{
    AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName,
    ToolResultContent, UserContent,
};
use rig_core::completion::ToolDefinition;

fn create_grpc_request(
    model: &str,
    request: CompletionRequest,
) -> Result<GenerateContentRequest, EncodeError> {
    GenerateContent::new(model).encode(request, Mode::Unary)
}

#[test]
fn create_grpc_request_sends_the_executed_name_not_an_identifier() -> anyhow::Result<()> {
    let call = |wire_id: &str, name: &str| -> anyhow::Result<Message> {
        Ok(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::ToolCall(ToolCall::from_wire(
                wire_id,
                ToolFunction::new(
                    ToolName::new(name).context("fixture name")?,
                    serde_json::json!({}),
                ),
            )),
        ])))
    };
    let result = |wire_id: &str, name: &str| -> anyhow::Result<Message> {
        let source = ToolCall::from_wire(
            wire_id,
            ToolFunction::new(
                ToolName::new(name).context("fixture name")?,
                serde_json::json!({}),
            ),
        );
        Ok(Message::User {
            content: vec![UserContent::ToolResult(
                source.result(vec![ToolResultContent::text("out")]),
            )],
        })
    };
    let mut request = CompletionRequest::new("unused");
    request.chat_history = vec![
        call("call_1", "add")?,
        result("call_1", "sum")?,
        call("call_abc", "get_weather")?,
        result("call_abc", "get_weather")?,
    ];
    let req = create_grpc_request("gemini-2.5-flash", request).context("request build")?;
    let responses: Vec<(&str, &str)> = req
        .contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| match &part.data {
            Some(proto::part::Data::FunctionResponse(fr)) => {
                Some((fr.id.as_str(), fr.name.as_str()))
            }
            _ => None,
        })
        .collect();
    let expected = vec![("call_1", "sum"), ("call_abc", "get_weather")];
    ensure!(
        responses == expected,
        "function responses differ: actual {responses:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn create_grpc_request_populates_tool_parameters() -> anyhow::Result<()> {
    let tool = ToolDefinition {
        name: ToolName::new("get_weather").context("fixture name")?,
        description: "Look up the current weather for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    };
    let mut request = CompletionRequest::new("forecast in Berlin?");
    request.tools = vec![tool];
    let req = create_grpc_request("gemini-2.5-flash", request).context("request build")?;
    ensure!(
        req.tools.len() == 1,
        "tool count differs: actual {:?}, expected 1",
        req.tools.len()
    );
    let tool = req
        .tools
        .first()
        .with_context(|| format!("tool entry is absent: {req:?}"))?;
    let decl = tool
        .function_declarations
        .first()
        .with_context(|| format!("function declaration is absent: {tool:?}"))?;
    ensure!(
        decl.name == "get_weather",
        "function name differs: actual {:?}, expected {:?}",
        decl.name,
        "get_weather"
    );
    let params = decl
        .parameters
        .as_ref()
        .with_context(|| format!("parameters are absent: {decl:?}"))?;
    ensure!(
        params.r#type == proto::Type::Object as i32,
        "parameter type differs: actual {:?}, expected {:?}",
        params.r#type,
        proto::Type::Object as i32
    );
    let expected = vec!["city".to_string()];
    ensure!(
        params.required == expected,
        "required parameters differ: actual {:?}, expected {expected:?}",
        params.required
    );
    ensure!(
        params.properties.contains_key("city"),
        "city property is absent: {:?}",
        params.properties
    );
    Ok(())
}
