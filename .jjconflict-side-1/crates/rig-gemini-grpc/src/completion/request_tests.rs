use super::*;
use crate::proto;
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
fn create_grpc_request_sends_the_executed_name_not_an_identifier() {
    let call = |wire_id: &str, name: &str| {
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            ToolCall::from_wire(
                wire_id,
                ToolFunction::new(
                    ToolName::new(name).expect("fixture name"),
                    serde_json::json!({}),
                ),
            ),
        )]))
    };
    let result = |wire_id: &str, name: &str| {
        let source = ToolCall::from_wire(
            wire_id,
            ToolFunction::new(
                ToolName::new(name).expect("fixture name"),
                serde_json::json!({}),
            ),
        );
        Message::User {
            content: vec![UserContent::ToolResult(
                source.result(vec![ToolResultContent::text("out")]),
            )],
        }
    };
    let mut request = CompletionRequest::new("unused");
    request.chat_history = vec![
        call("call_1", "add"),
        result("call_1", "sum"),
        call("call_abc", "get_weather"),
        result("call_abc", "get_weather"),
    ];
    let req = create_grpc_request("gemini-2.5-flash", request).expect("request build");
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
    assert_eq!(
        responses,
        vec![("call_1", "sum"), ("call_abc", "get_weather")]
    );
}

#[test]
fn create_grpc_request_populates_tool_parameters() {
    let tool = ToolDefinition {
        name: ToolName::new("get_weather").expect("fixture name"),
        description: "Look up the current weather for a city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    };
    let mut request = CompletionRequest::new("forecast in Berlin?");
    request.tools = vec![tool];
    let req = create_grpc_request("gemini-2.5-flash", request).expect("request build");
    assert_eq!(req.tools.len(), 1);
    let tool = req.tools.first().expect("tool entry");
    let decl = tool
        .function_declarations
        .first()
        .expect("function declaration");
    assert_eq!(decl.name, "get_weather");
    let params = decl.parameters.as_ref().expect("parameters populated");
    assert_eq!(params.r#type, proto::Type::Object as i32);
    assert_eq!(params.required, vec!["city".to_string()]);
    assert!(params.properties.contains_key("city"));
}
