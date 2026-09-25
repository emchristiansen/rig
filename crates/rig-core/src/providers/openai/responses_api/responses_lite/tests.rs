use crate::message::ImageDetail;

use super::*;
use crate::providers::openai::completion::ToolChoice as OpenAIToolChoice;
use crate::providers::openai::responses_api::openapi_schema::{self, Schema};
use crate::providers::openai::responses_api::{
    AdditionalParameters, DeclaredResponsesTool, ReasoningSummaryLevel, ResponsesRequestTool,
    ResponsesToolDefinition, ToolResult, ToolStatus,
};

fn identity(thread_id: &str) -> CodexIdentity {
    CodexIdentity::from_ids("session-a", thread_id).expect("the fixture ids are header-safe")
}

fn request(instructions: Option<&str>, tools: Vec<ResponsesRequestTool>) -> CompletionRequest {
    CompletionRequest {
        input: vec![InputItem::user_content(UserContent::InputText {
            text: "hello".to_owned(),
        })],
        model: "gpt-6-sol".to_owned(),
        instructions: instructions.map(str::to_owned),
        max_output_tokens: None,
        stream: Some(true),
        temperature: None,
        tool_choice: None,
        tools,
        additional_parameters: AdditionalParameters::default(),
    }
}

fn shaped_json(
    instructions: Option<&str>,
    tools: Vec<ResponsesRequestTool>,
    thread_id: &str,
) -> Value {
    let mut request = request(instructions, tools);
    shape_request(&mut request, &identity(thread_id)).expect("the fixture can be shaped");
    serde_json::to_value(request).expect("the shaped request serializes")
}

fn prefix_ids(body: &Value) -> (String, Option<String>) {
    let input = body["input"].as_array().expect("input is an array");
    let tools_id = input[0]["id"]
        .as_str()
        .expect("the tools prefix has an id")
        .to_owned();
    let message_id = input
        .get(1)
        .filter(|item| item["role"] == "developer")
        .and_then(|item| item["id"].as_str())
        .map(str::to_owned);
    (tools_id, message_id)
}

#[test]
fn exact_prefix_ids_are_stable_and_change_only_with_their_inputs() {
    let base = shaped_json(Some("base"), Vec::new(), "thread-a");
    assert_eq!(
        prefix_ids(&base),
        (
            "at_855a7933-349f-53b2-bfd4-b4cfff60aee4".to_owned(),
            Some("msg_b16266e1-b752-5add-9796-139b2fb17ae0".to_owned())
        )
    );
    assert_eq!(base["input"][0]["tools"], serde_json::json!([]));
    assert_eq!(base["input"][0]["role"], "developer");
    assert_eq!(base["input"][1]["type"], "message");
    assert_eq!(base["input"][1]["role"], "developer");
    assert_eq!(
        base["input"][1]["internal_chat_message_metadata_passthrough"]["content_item_kinds"],
        serde_json::json!(["model.base_instructions"])
    );

    let changed_instructions = shaped_json(Some("changed"), Vec::new(), "thread-a");
    assert_eq!(
        prefix_ids(&changed_instructions),
        (
            "at_855a7933-349f-53b2-bfd4-b4cfff60aee4".to_owned(),
            Some("msg_42d747b4-6a03-5bab-8fc7-24659a1fea99".to_owned())
        )
    );

    let changed_tools = shaped_json(
        Some("base"),
        vec![ResponsesToolDefinition::web_search().into()],
        "thread-a",
    );
    assert_eq!(
        prefix_ids(&changed_tools),
        (
            "at_d122644b-6449-51ae-9e51-3decd45e8ec0".to_owned(),
            Some("msg_b16266e1-b752-5add-9796-139b2fb17ae0".to_owned())
        )
    );

    let changed_thread = shaped_json(Some("base"), Vec::new(), "thread-b");
    assert_eq!(
        prefix_ids(&changed_thread),
        (
            "at_6cd5a590-6e19-583d-94d6-ef0445c7f05d".to_owned(),
            Some("msg_74be36a7-1d02-510c-a0b4-4f6e9c8517ac".to_owned())
        )
    );
}

#[test]
fn plain_developer_message_keeps_its_standard_mode_wire_shape() {
    let value = serde_json::json!({
        "role": "developer",
        "content": "ordinary developer instruction"
    });
    let decoded: Message =
        serde_json::from_value(value.clone()).expect("the legacy developer alias still decodes");

    assert_eq!(
        serde_json::to_value(decoded).expect("the developer item serializes"),
        serde_json::json!({
            "role": "developer",
            "content": [{"type":"input_text","text":"ordinary developer instruction"}]
        })
    );
}

#[test]
fn named_developer_message_keeps_its_name_when_typed_and_reencoded() {
    let value = serde_json::json!({
        "role": "developer",
        "content": "ordinary developer instruction",
        "name": "policy"
    });
    let decoded: Message =
        serde_json::from_value(value).expect("the named developer message decodes");

    assert_eq!(
        serde_json::to_value(decoded).expect("the named developer message reencodes"),
        serde_json::json!({
            "role": "developer",
            "content": [{"type":"input_text","text":"ordinary developer instruction"}],
            "name": "policy"
        })
    );
}

#[test]
fn non_message_input_keeps_accepting_an_explicit_null_role() {
    let value = serde_json::json!({
        "type": "function_call",
        "role": null,
        "id": "fc_1",
        "arguments": "{}",
        "call_id": "call_1",
        "name": "lookup",
        "status": "completed"
    });
    let decoded: InputItem =
        serde_json::from_value(value).expect("an explicit null role remains optional");
    let encoded = serde_json::to_value(decoded).expect("the function call re-serializes");

    assert_eq!(encoded["type"], "function_call");
    assert_eq!(encoded.get("role"), None);
}

#[test]
fn generated_lite_prefix_round_trips_without_accepting_a_conflicting_role() {
    let value = shaped_json(Some("base"), Vec::new(), "thread-a");
    let decoded: CompletionRequest =
        serde_json::from_value(value.clone()).expect("the generated Lite request decodes");
    for item in &decoded.input {
        let encoded = serde_json::to_string(item).expect("the input item serializes");
        if encoded.contains("\"role\"") {
            assert_eq!(
                encoded.matches("\"role\"").count(),
                1,
                "an input item must not serialize duplicate role keys: {encoded}"
            );
        }
    }
    assert_eq!(
        serde_json::to_value(decoded).expect("the decoded Lite request serializes"),
        value
    );

    let mut conflicting = value;
    conflicting["input"][0]["role"] = Value::String("user".to_owned());
    assert!(
        serde_json::from_value::<CompletionRequest>(conflicting).is_err(),
        "an additional_tools prefix cannot silently normalize a conflicting role"
    );
}

#[test]
fn public_input_item_ambiguity_and_the_named_validation_view_are_pinned() {
    let message = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}]
    });

    openapi_schema::assert_valid(Schema::EasyInputMessage, &message);
    openapi_schema::assert_valid(Schema::Item, &message);
    openapi_schema::assert_invalid(Schema::PublicInputItem, &message);
    openapi_schema::assert_valid(Schema::InputItemView, &message);

    let mut bad_role = message.clone();
    bad_role["role"] = serde_json::json!("tool");
    openapi_schema::assert_invalid(Schema::InputItemView, &bad_role);
    let mut bad_content = message;
    bad_content["content"] = serde_json::json!(42);
    openapi_schema::assert_invalid(Schema::InputItemView, &bad_content);
}

#[test]
fn lite_additional_tools_satisfies_the_unmodified_public_component_and_union() {
    let shaped = shaped_json(
        None,
        vec![
            ResponsesToolDefinition::function(
                "lookup",
                "Lookup a value",
                serde_json::json!({"type":"object","properties":{},"required":[]}),
            )
            .into(),
        ],
        "thread-a",
    );
    let additional_tools = &shaped["input"][0];

    openapi_schema::assert_valid(Schema::AdditionalToolsItem, additional_tools);
    openapi_schema::assert_valid(Schema::PublicInputItem, additional_tools);
    openapi_schema::assert_valid(Schema::LiteInputItemView, additional_tools);

    let mut bad_role = additional_tools.clone();
    bad_role["role"] = serde_json::json!("user");
    openapi_schema::assert_invalid(Schema::AdditionalToolsItem, &bad_role);
    openapi_schema::assert_invalid(Schema::LiteInputItemView, &bad_role);

    let mut bad_tools = additional_tools.clone();
    bad_tools["tools"] = serde_json::json!([{"type":"function","name":42}]);
    openapi_schema::assert_invalid(Schema::AdditionalToolsItem, &bad_tools);
    openapi_schema::assert_invalid(Schema::LiteInputItemView, &bad_tools);
}

#[test]
fn lite_image_view_relaxes_only_the_required_detail_member() {
    let image = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_image",
            "image_url": "https://example.test/image.png"
        }]
    });
    openapi_schema::assert_invalid(Schema::InputItemView, &image);
    openapi_schema::assert_valid(Schema::LiteInputItemView, &image);

    let mut with_detail = image.clone();
    with_detail["content"][0]["detail"] = serde_json::json!("high");
    openapi_schema::assert_valid(Schema::InputItemView, &with_detail);

    let mut bad_detail = image;
    bad_detail["content"][0]["detail"] = serde_json::json!("maximum");
    openapi_schema::assert_invalid(Schema::LiteInputItemView, &bad_detail);
}

fn declared(value: Value) -> ResponsesRequestTool {
    ResponsesRequestTool::Declared(
        DeclaredResponsesTool::from_value(value).expect("the declared tool is valid"),
    )
}

#[test]
fn tool_folding_keeps_order_and_the_last_nonblank_functions_description() {
    let tools = vec![
        declared(serde_json::json!({"type":"web_search","x":"first"})),
        declared(serde_json::json!({
            "type":"namespace",
            "name":"functions",
            "description":"first description",
            "tools":[{"type":"function","name":"nested_one","sentinel":1}]
        })),
        declared(serde_json::json!({"type":"file_search","x":"middle"})),
        declared(
            serde_json::json!({"type":"custom","name":"top_custom","format":{"type":"grammar"}}),
        ),
        declared(serde_json::json!({
            "type":"namespace",
            "name":"functions",
            "description":"   ",
            "tools":[{"type":"custom","name":"nested_two","sentinel":2}]
        })),
        declared(serde_json::json!({
            "type":"namespace",
            "name":"functions",
            "description":"last description",
            "tools":[{"type":"function","name":"nested_three","sentinel":3}]
        })),
        declared(serde_json::json!({"type":"image_generation","x":"last"})),
    ];

    let body = shaped_json(None, tools, "thread-a");
    let folded = body["input"][0]["tools"]
        .as_array()
        .expect("additional tools are an array");
    assert_eq!(folded.len(), 4);
    assert_eq!(folded[0]["type"], "web_search");
    assert_eq!(folded[1]["type"], "namespace");
    assert_eq!(folded[1]["description"], "last description");
    assert_eq!(folded[1]["tools"][0]["name"], "nested_one");
    assert_eq!(folded[1]["tools"][1]["name"], "top_custom");
    assert_eq!(folded[1]["tools"][2]["name"], "nested_two");
    assert_eq!(folded[1]["tools"][3]["name"], "nested_three");
    assert_eq!(folded[2]["type"], "file_search");
    assert_eq!(folded[3]["type"], "image_generation");
    assert_eq!(folded[0]["x"], "first");
    assert_eq!(folded[3]["x"], "last");
    assert!(body.get("tools").is_none());
    assert!(body.get("instructions").is_none());
}

#[test]
fn malformed_functions_namespace_is_refused_instead_of_discarded() {
    let mut request = request(
        None,
        vec![declared(serde_json::json!({
            "type": "namespace",
            "name": "functions",
            "description": "kept if valid",
            "sentinel": "must not vanish"
        }))],
    );

    assert!(matches!(
        shape_request(&mut request, &identity("thread-a")),
        Err(ResponsesLiteError::MalformedFunctionsNamespace { field: "tools" })
    ));
}

#[test]
fn absent_lite_controls_are_defaulted_and_matching_values_are_retained() {
    let body = shaped_json(None, Vec::new(), "thread-a");
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["reasoning"]["context"], "all_turns");

    let mut request = request(None, Vec::new());
    request.additional_parameters.parallel_tool_calls = Some(false);
    request.tool_choice = Some(ToolChoice::Mode(OpenAIToolChoice::Auto));
    request.additional_parameters.reasoning = Some(
        Reasoning::new()
            .with_context(ReasoningContext::AllTurns)
            .with_summary_level(ReasoningSummaryLevel::Detailed),
    );
    shape_request(&mut request, &identity("thread-a")).expect("matching controls are valid");
    let body = serde_json::to_value(request).expect("the request serializes");
    assert_eq!(body["reasoning"]["summary"], "detailed");
}

#[test]
fn explicit_conflicting_lite_controls_are_refused_by_name() {
    let mut parallel = request(None, Vec::new());
    parallel.additional_parameters.parallel_tool_calls = Some(true);
    assert!(matches!(
        shape_request(&mut parallel, &identity("thread-a")),
        Err(ResponsesLiteError::ParallelToolCalls)
    ));

    let mut choice = request(None, Vec::new());
    choice.tool_choice = Some(ToolChoice::Mode(OpenAIToolChoice::Required));
    assert!(matches!(
        shape_request(&mut choice, &identity("thread-a")),
        Err(ResponsesLiteError::ToolChoice)
    ));

    let mut context = request(None, Vec::new());
    context.additional_parameters.reasoning =
        Some(Reasoning::new().with_context(ReasoningContext::CurrentTurn));
    assert!(matches!(
        shape_request(&mut context, &identity("thread-a")),
        Err(ResponsesLiteError::ReasoningContext)
    ));
}

#[test]
fn full_and_delta_image_details_are_omitted_in_every_supported_location() {
    let image_message = InputItem::user_content(UserContent::InputImage {
        image_url: "https://example.test/image.png".to_owned(),
        detail: Some(ImageDetail::High),
    });
    let function_output = InputItem {
        role: None,
        input: InputContent::FunctionCallOutput(ToolResult {
            call_id: "call-1".to_owned(),
            output: ToolResultOutput::Content(vec![ToolResultOutputContent::InputImage {
                image_url: Some("https://example.test/result.png".to_owned()),
                file_id: None,
                detail: Some(ImageDetail::Low),
            }]),
            status: ToolStatus::Completed,
        }),
    };

    let mut full = request(None, Vec::new());
    full.input = vec![image_message.clone(), function_output.clone()];
    shape_request(&mut full, &identity("thread-a")).expect("the full request shapes");
    let full = serde_json::to_value(full).expect("the full request serializes");
    assert!(full["input"][1]["content"][0].get("detail").is_none());
    assert!(full["input"][2]["output"][0].get("detail").is_none());

    let mut delta = vec![image_message, function_output];
    shape_delta(&mut delta);
    let delta = serde_json::to_value(delta).expect("the delta serializes");
    assert!(delta[0]["content"][0].get("detail").is_none());
    assert!(delta[1]["output"][0].get("detail").is_none());
}
