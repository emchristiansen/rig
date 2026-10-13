use crate::message::ImageDetail;

use super::*;
use crate::providers::openai::responses_api::openapi_schema::{self, Schema};
use crate::providers::openai::responses_api::{
    DeclaredResponsesTool, ResponsesRequestTool, ResponsesToolDefinition,
};

fn identity(thread_id: &str) -> CodexIdentity {
    CodexIdentity::from_ids("session-a", thread_id).expect("the fixture ids are header-safe")
}

fn request(instructions: Option<&str>, tools: Vec<ResponsesRequestTool>) -> Map<String, Value> {
    let mut body = serde_json::json!({
        "input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],
        "model": "gpt-6-sol",
        "stream": true,
    })
    .as_object()
    .expect("fixture body is an object")
    .clone();
    if let Some(instructions) = instructions {
        body.insert(
            "instructions".to_owned(),
            Value::String(instructions.to_owned()),
        );
    }
    if !tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            serde_json::to_value(tools).expect("actual tools serialize"),
        );
    }
    body
}

fn shaped_json(
    instructions: Option<&str>,
    tools: Vec<ResponsesRequestTool>,
    thread_id: &str,
) -> Value {
    let mut body = request(instructions, tools);
    shape_body(&mut body, &identity(thread_id)).expect("the fixture can be shaped");
    Value::Object(body)
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
    // A non-empty tools array hashes its transmitted JSON, whose key order
    // depends on whether `serde_json/preserve_order` is unified into the
    // build (the workspace root enables it), so the exact UUID is not a
    // constant here. Pin the derivation instead: the ID is the UUIDv5 of the
    // transmitted tools bytes in the thread's namespace, as `openai/codex`
    // derives it.
    let thread_namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"thread-a");
    let transmitted_tools =
        serde_json::to_vec(&changed_tools["input"][0]["tools"]).expect("tools serialize");
    let (changed_tools_id, changed_tools_message_id) = prefix_ids(&changed_tools);
    assert_eq!(
        changed_tools_id,
        format!("at_{}", Uuid::new_v5(&thread_namespace, &transmitted_tools))
    );
    assert_ne!(changed_tools_id, prefix_ids(&base).0);
    assert_eq!(
        changed_tools_message_id,
        Some("msg_b16266e1-b752-5add-9796-139b2fb17ae0".to_owned())
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

fn request_with_raw_item(item: Value) -> crate::completion::CompletionRequest {
    use crate::message::{AssistantContent, AssistantMessage, CarriedPayload, Message, ProviderItem};
    crate::completion::CompletionRequest::new("hello").message(Message::Assistant(
        AssistantMessage::new(vec![AssistantContent::ProviderItem(ProviderItem {
            payload: CarriedPayload::new(item, None),
        })]),
    ))
}

#[test]
fn ordinary_developer_items_keep_their_standard_mode_wire_shape() {
    let expected = serde_json::json!({
        "role": "developer",
        "content": [{"type":"input_text","text":"ordinary developer instruction"}]
    });
    let mut target = lite_wire();
    target.codex_request_shape = CodexRequestShape::Standard;
    let body = encoded_body(&target, request_with_raw_item(expected.clone()));
    assert_eq!(body["input"][0], expected);

    // The old DTO's string alias is retired. A raw item was always an
    // independent reader path, and its string stays exact on that path.
    let raw = serde_json::json!({
        "role": "developer", "content": "ordinary developer instruction"
    });
    let body = encoded_body(&target, request_with_raw_item(raw.clone()));
    assert_eq!(body["input"][0], raw);
}

#[test]
fn named_developer_items_keep_their_name_at_the_real_request_boundary() {
    let expected = serde_json::json!({
        "role": "developer",
        "content": [{"type":"input_text","text":"ordinary developer instruction"}],
        "name": "policy"
    });
    let mut target = lite_wire();
    target.codex_request_shape = CodexRequestShape::Standard;
    let body = encoded_body(&target, request_with_raw_item(expected.clone()));
    assert_eq!(body["input"][0], expected);
    assert_eq!(body["input"][0]["name"], "policy");
}

#[test]
fn canonical_function_calls_omit_role_and_raw_null_role_items_remain_exact() {
    use crate::message::{AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName};
    let call = ToolCall::from_wire("call_1", ToolFunction::new(
        ToolName::new("lookup").expect("nonempty name"), serde_json::json!({}),
    ));
    let mut target = lite_wire();
    target.codex_request_shape = CodexRequestShape::Standard;
    let request = crate::completion::CompletionRequest::new("hello").message(Message::Assistant(
        AssistantMessage::new(vec![AssistantContent::ToolCall(call)]),
    ));
    let body = encoded_body(&target, request);
    assert_eq!(body["input"][0]["type"], "function_call");
    assert_eq!(body["input"][0].get("role"), None);
    assert_eq!(body["input"][0]["call_id"], "call_1");
    assert_eq!(body["input"][0]["name"], "lookup");
    assert_eq!(body["input"][0]["arguments"], "{}");

    let raw = serde_json::json!({
        "type": "function_call", "role": null, "id": "fc_1",
        "arguments": "{}", "call_id": "call_1", "name": "lookup", "status": "completed"
    });
    let body = encoded_body(&target, request_with_raw_item(raw.clone()));
    assert_eq!(body["input"][0], raw);
    assert_eq!(body["input"][0].get("role"), Some(&Value::Null));
}

#[test]
fn generated_lite_prefix_has_exact_bytes_and_its_schema_refuses_conflicting_role() {
    let target = lite_wire();
    let body = encoded_body(&target, crate::completion::CompletionRequest::new("hello").preamble("base"));
    let expected = shaped_json(Some("base"), Vec::new(), "thread-a");
    assert_eq!(&body["input"].as_array().expect("encoded input")[..2],
        &expected["input"].as_array().expect("expected input")[..2]);
    assert_eq!(prefix_ids(&body), prefix_ids(&expected));
    assert_eq!(body["input"][1]["internal_chat_message_metadata_passthrough"]["content_item_kinds"],
        serde_json::json!(["model.base_instructions"]));
    for item in body["input"].as_array().expect("input array") {
        let encoded = serde_json::to_string(item).expect("actual input item serializes");
        if encoded.contains("\"role\"") {
            assert_eq!(encoded.matches("\"role\"").count(), 1,
                "an input item must not serialize duplicate role keys: {encoded}");
        }
    }
    let serialized = serde_json::to_vec(&body).expect("actual request JSON bytes");
    assert_eq!(serde_json::from_slice::<Value>(&serialized).expect("JSON bytes read back"), body);
    openapi_schema::assert_valid(Schema::AdditionalToolsItem, &body["input"][0]);
    openapi_schema::assert_valid(Schema::LiteInputItemView, &body["input"][0]);
    let mut conflicting = body["input"][0].clone();
    conflicting["role"] = Value::String("user".to_owned());
    openapi_schema::assert_invalid(Schema::AdditionalToolsItem, &conflicting);
    openapi_schema::assert_invalid(Schema::LiteInputItemView, &conflicting);
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
            "sentinel": "refused with its envelope, never dropped silently"
        }))],
    );

    assert!(matches!(
        shape_body(&mut request, &identity("thread-a")),
        Err(ResponsesLiteError::MalformedFunctionsNamespace { field: "tools" })
    ));
}

#[test]
fn a_valid_functions_namespace_loses_its_extra_members_when_folded() {
    let tools = vec![declared(serde_json::json!({
        "type": "namespace",
        "name": "functions",
        "description": "kept",
        "sentinel": "dropped by the fold",
        "tools": [{"type": "function", "name": "only"}]
    }))];

    let body = shaped_json(None, tools, "thread-a");
    let folded = body["input"][0]["tools"]
        .as_array()
        .expect("additional tools are an array");
    assert_eq!(folded.len(), 1);
    assert_eq!(folded[0]["description"], "kept");
    assert_eq!(folded[0]["tools"][0]["name"], "only");
    assert!(folded[0].get("sentinel").is_none(), "{folded:?}");
}

#[test]
fn a_functions_namespace_that_folds_no_functions_is_removed() {
    let tools = vec![
        declared(serde_json::json!({"type": "web_search"})),
        declared(serde_json::json!({
            "type": "namespace",
            "name": "functions",
            "description": "empty",
            "tools": []
        })),
    ];

    let body = shaped_json(None, tools, "thread-a");
    let folded = body["input"][0]["tools"]
        .as_array()
        .expect("additional tools are an array");
    assert_eq!(folded.len(), 1, "{folded:?}");
    assert_eq!(folded[0]["type"], "web_search");
}

#[test]
fn a_functions_namespace_with_a_non_string_description_is_refused() {
    // The typed declared-tool constructor already refuses this shape, so the
    // fold's own refusal is a second line; pin it on the fold directly.
    let namespace = serde_json::json!({
        "type": "namespace",
        "name": "functions",
        "description": 7,
        "tools": [{"type": "function", "name": "only"}]
    });
    assert!(DeclaredResponsesTool::from_value(namespace.clone()).is_err());
    assert!(matches!(
        super::fold_tools(vec![namespace]),
        Err(ResponsesLiteError::MalformedFunctionsNamespace {
            field: "description"
        })
    ));
}

#[test]
fn absent_lite_controls_are_defaulted_and_matching_values_are_retained() {
    let body = shaped_json(None, Vec::new(), "thread-a");
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["reasoning"]["context"], "all_turns");

    let mut request = request(None, Vec::new());
    request.insert("parallel_tool_calls".to_owned(), serde_json::json!(false));
    request.insert("tool_choice".to_owned(), serde_json::json!("auto"));
    request.insert(
        "reasoning".to_owned(),
        serde_json::json!({"context":"all_turns","summary":"detailed"}),
    );
    shape_body(&mut request, &identity("thread-a")).expect("matching controls are valid");
    let body = serde_json::to_value(request).expect("the request serializes");
    assert_eq!(body["reasoning"]["summary"], "detailed");
}

#[test]
fn explicit_conflicting_lite_controls_are_refused_by_name() {
    let mut parallel = request(None, Vec::new());
    parallel.insert("parallel_tool_calls".to_owned(), serde_json::json!(true));
    assert!(matches!(
        shape_body(&mut parallel, &identity("thread-a")),
        Err(ResponsesLiteError::ParallelToolCalls)
    ));

    let mut choice = request(None, Vec::new());
    choice.insert("tool_choice".to_owned(), serde_json::json!("required"));
    assert!(matches!(
        shape_body(&mut choice, &identity("thread-a")),
        Err(ResponsesLiteError::ToolChoice)
    ));

    let mut context = request(None, Vec::new());
    context.insert(
        "reasoning".to_owned(),
        serde_json::json!({"context":"current_turn"}),
    );
    assert!(matches!(
        shape_body(&mut context, &identity("thread-a")),
        Err(ResponsesLiteError::ReasoningContext)
    ));
}

fn lite_wire() -> super::super::wire::Responses {
    let mut config = crate::providers::openai::OpenAIConfig::with_key(
        &crate::providers::chatgpt::DIALECT,
        "test-key",
    );
    config = config.with_caller_identity(
        crate::providers::openai::wire::CallerIdentity::new("fixture-origin", "fixture-agent/1.0", None)
            .expect("caller headers are valid"),
    );
    config.instructions = None;
    super::super::wire::Responses::new(config, "gpt-6-sol")
        .with_codex_identity(identity("thread-a"))
        .expect("Codex accepts its identity")
        .with_responses_lite()
        .expect("Codex supports Lite")
}

fn encoded_body(
    target: &super::super::wire::Responses,
    request: crate::completion::CompletionRequest,
) -> Value {
    use crate::wire::Wire;
    let encoded = target
        .encode(request, crate::wire::Mode::Unary)
        .expect("real Responses request encodes");
    crate::test_utils::json_body(&encoded.request)
}

#[test]
fn full_and_delta_image_details_are_omitted_in_every_supported_location() {
    use crate::message::{
        AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName,
        ToolResultContent, UserContent,
    };
    let image_message = Message::User {
        content: vec![UserContent::image_url(
            "https://example.test/image.png",
            None,
            Some(ImageDetail::High),
        )],
    };
    let call = ToolCall::from_wire(
        "call-1",
        ToolFunction::new(
            ToolName::new("lookup").expect("nonempty tool"),
            serde_json::json!({}),
        ),
    );
    let function_output =
        Message::tool_results(vec![call.result(vec![ToolResultContent::image_url(
            "https://example.test/result.png",
            None,
            Some(ImageDetail::Low),
        )])]);
    let history = vec![
        image_message,
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )])),
        function_output,
    ];
    let target = lite_wire();
    let request = crate::completion::CompletionRequest::from(history.clone());
    let full = target
        .responses_request(
            &request,
            super::super::Delivery::Http(crate::wire::Mode::Unary),
        )
        .expect("the actual full request shapes");
    let full = serde_json::to_value(full).expect("the full request serializes");
    assert!(full["input"][1]["content"][0].get("detail").is_none());
    assert!(full["input"][3]["output"][0].get("detail").is_none());
    assert_eq!(
        full["input"][1]["content"][0]["image_url"],
        "https://example.test/image.png"
    );
    assert_eq!(
        full["input"][3]["output"][0]["image_url"],
        "https://example.test/result.png"
    );
    assert_eq!(full["input"][3]["call_id"], "call-1");
    assert_eq!(full["input"][3]["status"], "completed");

    // Incremental input uses the same typed converter before the captured
    // envelope supplies its existing Lite prefix and controls.
    let delta = super::super::input(
        &history,
        &target,
        &target.model,
        &mut super::super::Custom {
            tools: Default::default(),
            calls: Default::default(),
        },
        true,
        &mut Vec::new(),
        None,
    )
    .expect("actual typed delta input converts");
    assert!(delta[0]["content"][0].get("detail").is_none());
    assert!(delta[2]["output"][0].get("detail").is_none());
    assert_eq!(
        &full["input"].as_array().expect("input array")[1..],
        delta.as_slice()
    );

    let mut standard = target.clone();
    standard.codex_request_shape = CodexRequestShape::Standard;
    let input = super::super::input(
        &history,
        &standard,
        &standard.model,
        &mut super::super::Custom {
            tools: Default::default(),
            calls: Default::default(),
        },
        true,
        &mut Vec::new(),
        None,
    )
    .expect("standard image input converts");
    assert_eq!(input[0]["content"][0]["detail"], "high");
    assert_eq!(input[2]["output"][0]["detail"], "low");
}

#[test]
fn raw_whole_items_keep_image_lookalikes_verbatim_in_full_and_websocket_bodies() {
    use crate::message::{
        AssistantContent, AssistantMessage, CarriedPayload, Message, ProviderItem,
    };
    let raw = vec![
        serde_json::json!({"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.test/image.png","detail":"high","future":{"detail":"nested"}}],"future_item":true}),
        serde_json::json!({"type":"function_call_output","call_id":"call-1","output":[{"type":"input_image","image_url":"https://example.test/result.png","detail":"low"}],"status":"completed"}),
    ];
    let target = lite_wire();
    let carried = crate::completion::CompletionRequest::from(vec![
        Message::user("hello"),
        Message::Assistant(AssistantMessage::new(
            raw.iter()
                .cloned()
                .map(|item| {
                    AssistantContent::ProviderItem(ProviderItem {
                        payload: CarriedPayload::new(item, None),
                    })
                })
                .collect(),
        )),
    ]);
    for delivery in [
        super::super::Delivery::Http(crate::wire::Mode::Unary),
        super::super::Delivery::WebSocket,
    ] {
        let request = &carried;
        let body = target
            .responses_request_with_identity(
                request,
                delivery,
                target.codex_identity.as_ref(),
                super::super::InputRequirement::NonEmpty,
            )
            .expect("the real typed/full request boundary preserves raw items");
        let input = body
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(&input[input.len() - raw.len()..], raw.as_slice());
        for item in &raw {
            assert!(input.contains(item));
        }
    }
    let encoded = encoded_body(&target, carried);
    let input = encoded["input"].as_array().expect("encoded input array");
    assert_eq!(&input[input.len() - raw.len()..], raw.as_slice());
}

#[test]
fn malformed_merged_body_members_are_typed_failures_without_fabricated_defaults() {
    for (field, invalid, diagnostic) in [
        (
            "tools",
            serde_json::json!({}),
            "the Responses tools collection did not serialize as an array",
        ),
        (
            "tools",
            Value::Null,
            "the Responses tools collection did not serialize as an array",
        ),
        (
            "input",
            serde_json::json!("untyped prompt"),
            "the Responses input collection did not serialize as an array",
        ),
        (
            "instructions",
            serde_json::json!(7),
            "the Responses instructions did not serialize as a string",
        ),
    ] {
        let mut body = request(None, Vec::new());
        body.insert(field.to_owned(), invalid.clone());
        let error = shape_body(&mut body, &identity("thread-a"))
            .expect_err("malformed merged member refuses");
        assert!(matches!(error, ResponsesLiteError::ToolSerialization(_)));
        assert_eq!(
            error.to_string(),
            format!("Responses Lite could not serialize the effective tools: {diagnostic}")
        );
        assert_eq!(body.get(field), Some(&invalid));
    }
}

#[test]
fn whitespace_instructions_and_unknown_body_fields_keep_exact_values() {
    let instructions = "  base\n\n ";
    let mut body = request(Some(instructions), Vec::new());
    body.insert(
        "future".to_owned(),
        serde_json::json!({"exact":[null,"  untouched "]}),
    );
    let original_input = body["input"].clone();
    shape_body(&mut body, &identity("thread-a")).expect("shape exact instruction bytes");
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input[1]["content"][0]["text"], instructions);
    assert_eq!(
        input[1]["id"],
        format!(
            "msg_{}",
            Uuid::new_v5(
                &Uuid::new_v5(&Uuid::NAMESPACE_OID, b"thread-a"),
                instructions.as_bytes()
            )
        )
    );
    assert_eq!(input[2], original_input[0]);
    assert_eq!(
        body["future"],
        serde_json::json!({"exact":[null,"  untouched "]})
    );
    let empty = shaped_json(Some(""), Vec::new(), "thread-a");
    assert_eq!(empty["input"].as_array().expect("input array").len(), 2);
    assert!(empty.get("instructions").is_none());
}

#[cfg(feature = "websocket")]
#[test]
fn websocket_marker_merges_exact_metadata_and_refuses_a_non_object() {
    let mut body =
        serde_json::json!({"client_metadata":{"session_id":"session-a","future":[null,1]}})
            .as_object()
            .expect("object")
            .clone();
    stamp_websocket_marker(&mut body).expect("existing metadata accepts marker");
    assert_eq!(
        body["client_metadata"],
        serde_json::json!({
            "session_id":"session-a", "future":[null,1],
            "ws_request_header_x_openai_internal_codex_responses_lite":"true",
        })
    );
    let mut invalid = serde_json::json!({"client_metadata":17})
        .as_object()
        .expect("object")
        .clone();
    let error = stamp_websocket_marker(&mut invalid).expect_err("non-object metadata refuses");
    assert_eq!(
        error.to_string(),
        "RequestError: the Codex request's `client_metadata` must be a JSON object to carry the Responses Lite marker, got 17"
    );
    assert_eq!(invalid["client_metadata"], 17);
}
