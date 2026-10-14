use super::*;
use crate::completion::CacheRetention;
use crate::error::ProviderError;
use crate::message::EMPTY_RESPONSE_ERROR;
use crate::operation::Completion;
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::test_utils::json_body;
use crate::wire::WireFrame;
use crate::wire::{Operation, Wire};
use serde_json::json;

/// The wire settings a request body is built with.
struct Params<'a> {
    model: &'a str,
    request: CompletionRequest,
    prompt_caching: bool,
    /// The request's cache retention.
    cache: Option<crate::completion::CacheRetention>,
    static_prefix_cache_ttl: Option<CacheTtl>,
}

/// The body `params` builds, its documents placed as `prepare` places them.
fn request_body(params: Params<'_>) -> Result<Value, EncodeError> {
    request_body_with(params, false)
}

/// [`request_body`], with strict tools when `strict`.
fn request_body_with(params: Params<'_>, strict: bool) -> Result<Value, EncodeError> {
    let mut wire = AnthropicConfig::new("test-key").completion(params.model);
    wire.prompt_caching = params.prompt_caching;
    wire.static_prefix_cache_ttl = params.static_prefix_cache_ttl;
    wire.strict_tools = strict;
    use crate::wire::{Operation, Wire};
    // The request as production encodes it: prepared, so adapted.
    let mut request = params.request;
    if let Some(cache) = params.cache {
        request.options = request.options.cache(cache);
    }
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .map_err(|error| EncodeError::request(error.to_string()))?;
    Ok(serde_json::to_value(body(&wire, &request, Mode::Unary)?)?)
}

/// `message` alone on the wire.
fn convert(message: message::Message) -> Result<Option<Value>, EncodeError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let history = [message];
    let ids =
        WireIds::for_target(&history, &wire, CLAUDE_SONNET_4_6).map_err(EncodeError::request)?;
    history
        .iter()
        .enumerate()
        .map(|(index, message)| message_json(index, message, &wire, &ids))
        .next()
        .unwrap_or(Ok(None))
}

/// The user message holding `part`, as the encoder converts it.
fn user_wire(part: message::UserContent) -> Result<Option<Value>, EncodeError> {
    convert(message::Message::User {
        content: vec![part],
    })
}

/// An assistant turn of `blocks` on the wire.
fn assistant_wire(blocks: Vec<message::AssistantContent>) -> Vec<Value> {
    convert(message::Message::Assistant(message::AssistantMessage::new(
        blocks,
    )))
    .expect("the turn converts")
    .and_then(|message| message["content"].as_array().cloned())
    .unwrap_or_default()
}

/// The one-turn request every reply below is folded against.
fn hello_request() -> CompletionRequest {
    completion_request_with_history(vec![message::Message::user("hello")], None)
}

/// Fold a Messages reply body through the wire's own decoder — the one
/// mapping there is, now that the duplicate `normalize` is gone.
///
/// `POST /v1/messages` answers with a whole `message` object, which the
/// decoder treats as a frame like any other, so a cell that used to call
/// `normalize` on a typed response states the same thing by folding the
/// body the provider would have sent. `type` is injected when the cell
/// built the typed response directly, since that type does not model the
/// tag. Whatever the decoder or the fold rejects surfaces here as the
/// error the driver would report.
fn fold_reply(body: &serde_json::Value) -> Result<completion::CompletionResponse, ProviderError> {
    let mut body = body.clone();
    if let Some(map) = body.as_object_mut() {
        map.entry("type").or_insert_with(|| json!("message"));
    }
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        crate::wire::Mode::Unary,
        [WireFrame::Text(body.to_string())],
        body,
    )
}

#[test]
fn current_model_default_max_tokens_match_anthropic_limits() {
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_FABLE_5_1),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_FABLE_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_SONNET_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_8), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_7), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_6), Some(128_000));
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_SONNET_4_6),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_HAIKU_4_5), Some(64_000));
}

#[test]
fn unknown_model_has_no_documented_default_max_tokens() {
    assert_eq!(default_max_tokens_for_model("claude-unknown"), None);
}

#[test]
fn test_deserialize_message() {
    let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

    let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                }
            ]
        }
        "#;

    let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "15 degrees"
                }
            ]
        }
        "#;

    let first: Value = serde_json::from_str(assistant_message_json).unwrap();
    let second: Value = serde_json::from_str(assistant_message_json2).unwrap();
    let user: Value = serde_json::from_str(user_message_json).unwrap();
    let text = "\n\nHello there, how may I assist you today?";
    let assistant = convert(message::Message::assistant(text)).unwrap().unwrap();
    assert_eq!(assistant["role"], first["role"]);
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["content"][0]["type"], "text");
    assert_eq!(assistant["content"][0]["text"], first["content"]);
    assert_eq!(assistant["content"][0]["text"], text);
    assert!(assistant["content"][0].get("citations").is_none());
    assert!(assistant["content"][0].get("cache_control").is_none());

    let name = message::ToolName::new("get_weather").unwrap();
    let call = message::ToolCall::from_wire(
        "toolu_01A09q90qw90lq917835lq9",
        message::ToolFunction::new(name, json!({"location":"San Francisco, CA"})),
    );
    let assistant = convert(message::Message::Assistant(message::AssistantMessage::new(
        vec![
            message::AssistantContent::text(text),
            message::AssistantContent::ToolCall(call.clone()),
        ],
    )))
    .unwrap()
    .unwrap();
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["content"].as_array().unwrap().len(), 2);
    assert_eq!(assistant["content"], second["content"]);
    assert_eq!(
        assistant["content"][1]["id"],
        "toolu_01A09q90qw90lq917835lq9"
    );
    assert_eq!(assistant["content"][1]["name"], "get_weather");
    assert_eq!(
        assistant["content"][1]["input"],
        json!({"location":"San Francisco, CA"})
    );
    for part in assistant["content"].as_array().unwrap() {
        assert_eq!(validated_content(&part.to_string()), *part);
    }

    let image = message::Image {
        data: message::DocumentSourceKind::base64("/9j/4AAQSkZJRg..."),
        media_type: Some(message::ImageMediaType::JPEG),
        ..Default::default()
    };
    let result = call.result(vec![message::ToolResultContent::text("15 degrees")]);
    let user_wire = convert(message::Message::User {
        content: vec![
            message::UserContent::Image(image),
            message::UserContent::text("What is in this image?"),
            message::UserContent::ToolResult(result),
        ],
    })
    .unwrap()
    .unwrap();
    assert_eq!(user_wire["role"], "user");
    let content = user_wire["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0], user["content"][0]);
    assert_eq!(content[0]["source"]["data"], "/9j/4AAQSkZJRg...");
    assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
    assert_eq!(content[1], user["content"][1]);
    assert_eq!(content[1]["text"], "What is in this image?");
    assert_eq!(content[2]["tool_use_id"], "toolu_01A09q90qw90lq917835lq9");
    assert_eq!(
        content[2]["content"][0]["text"],
        user["content"][2]["content"]
    );
    assert_eq!(content[2]["content"][0]["text"], "15 degrees");
    assert!(content[2].get("is_error").is_none());
    for part in content {
        assert_eq!(validated_content(&part.to_string()), *part);
    }
}

#[test]
fn test_cache_control_serialization() {
    let system = json!({"type": "text", "text": "You are a helpful assistant.",
        "cache_control": ephemeral(None)});
    let json = serde_json::to_string(&system).unwrap();
    assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    assert!(json.contains(r#""type":"text""#));

    let system_no_cache = text("Hello");
    let json_no_cache = serde_json::to_string(&system_no_cache).unwrap();
    assert!(!json_no_cache.contains("cache_control"));

    let mut content = text("Test message");
    content["cache_control"] = ephemeral(None);
    let json_content = serde_json::to_string(&content).unwrap();
    assert!(json_content.contains(r#""cache_control":{"type":"ephemeral"}"#));

    let mut system_vec = vec![text("System prompt")];
    let mut messages = vec![
        json!({"role": "user", "content": [text("First message")]}),
        json!({"role": "assistant", "content": [text("Response")]}),
    ];
    let mut wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    wire.prompt_caching = true;
    apply_cache_control(&wire, false, None, &mut system_vec, &mut messages, &mut []).unwrap();
    assert!(system_vec[0].get("cache_control").is_some());
    for content in messages[0]["content"].as_array().unwrap() {
        assert!(content.get("cache_control").is_none());
    }
    for content in messages[1]["content"].as_array().unwrap() {
        assert!(content.get("cache_control").is_some());
    }
}

fn generic_tool(name: &str) -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: crate::message::ToolName::new(name).expect("tool name"),
        description: format!("{name} description"),
        parameters: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn completion_request_with_tools(
    tools: Vec<completion::ToolDefinition>,
    additional_params: Option<serde_json::Value>,
) -> CompletionRequest {
    CompletionRequest::from(vec![
        message::Message::system("System prompt"),
        message::Message::from("Hello"),
    ])
    .max_tokens(64)
    .tools(tools)
    .additional_params(additional_params)
}

fn completion_request_with_history(
    chat_history: Vec<message::Message>,
    preamble: Option<String>,
) -> CompletionRequest {
    CompletionRequest::from(
        preamble
            .map(message::Message::system)
            .into_iter()
            .chain(chat_history)
            .collect::<Vec<_>>(),
    )
    .max_tokens(64)
}

#[test]
fn rig_tools_are_non_strict_by_default() {
    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let request = request_body(Params {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

/// Anthropic-compatible gateways do not necessarily implement Anthropic's
/// constrained tool schemas, so asking for strict tools on one leaves the
/// tool unchanged. The policy is the dialect's data, not a trait hook, so
/// this is asserted through the wire a gateway builds.
#[test]
fn strict_tool_hook_is_a_noop_for_anthropic_compatible_gateways() {
    use crate::wire::{Mode, Wire};

    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let encoded = crate::providers::anthropic::wire::AnthropicConfig::with_key(
        &crate::providers::anthropic::wire::ZAI,
        "k",
    )
    .completion("some-model")
    .with_strict_tools()
    .encode(request, Mode::Unary)
    .expect("the request encodes");
    let value = json_body(&encoded.request);

    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

#[test]
fn strict_tools_opt_in_marks_and_sanitizes_rig_tools_only() {
    let mut tool = generic_tool("lookup");
    tool.parameters = json!({
        "type": "object",
        "additionalProperties": true,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "maxLength": 20,
                "pattern": "^[a-z]+$",
                "format": "uuid"
            },
            "kind": {
                "type": "string",
                "const": "lookup"
            },
            "legacy_filter": {
                "$ref": "#/definitions/LegacyFilter"
            },
            "options": {
                "type": "object",
                "additionalProperties": true,
                "properties": {
                    "limit": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 100,
                        "format": "uint32"
                    }
                }
            }
        },
        "definitions": {
            "LegacyFilter": {
                "type": "object",
                "properties": {
                    "term": { "type": "string" }
                }
            }
        },
        "required": ["query"]
    });
    let request = completion_request_with_tools(
        vec![tool],
        Some(json!({
            "tools": [{
                "type": "mcp_toolset",
                "name": "remote_tools"
            }]
        })),
    );
    let request = request_body_with(
        Params {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            cache: None,
            static_prefix_cache_ttl: None,
        },
        true,
    )
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let rig_tool = &value["tools"][0];
    assert_eq!(rig_tool["strict"], true);
    assert_eq!(rig_tool["input_schema"]["additionalProperties"], false);
    let required = rig_tool["input_schema"]["required"]
        .as_array()
        .expect("strict object schema should list required properties");
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert_eq!(
        rig_tool["input_schema"]["properties"]["options"]["additionalProperties"],
        false
    );
    assert!(
        rig_tool["input_schema"]["properties"]["options"]
            .get("required")
            .is_none()
    );
    let query = &rig_tool["input_schema"]["properties"]["query"];
    assert_eq!(query["format"], "uuid");
    for keyword in ["minLength", "maxLength", "pattern"] {
        assert!(query.get(keyword).is_none());
    }
    let query_description = query["description"]
        .as_str()
        .expect("unsupported string constraints should become guidance");
    for guidance in ["minLength: 2", "maxLength: 20", "pattern: ^[a-z]+$"] {
        assert!(query_description.contains(guidance));
    }
    assert_eq!(
        rig_tool["input_schema"]["properties"]["kind"]["const"],
        "lookup"
    );
    assert_eq!(
        rig_tool["input_schema"]["properties"]["legacy_filter"]["$ref"],
        "#/definitions/LegacyFilter"
    );
    assert_eq!(
        rig_tool["input_schema"]["definitions"]["LegacyFilter"]["additionalProperties"],
        false
    );
    let limit = &rig_tool["input_schema"]["properties"]["options"]["properties"]["limit"];
    assert!(limit.get("format").is_none());
    assert!(
        ["minimum", "maximum"]
            .into_iter()
            .all(|keyword| limit.get(keyword).is_none())
    );
    let limit_description = limit["description"]
        .as_str()
        .expect("unsupported numeric constraints should become guidance");
    for guidance in ["minimum: 1", "maximum: 100", "format: uint32"] {
        assert!(limit_description.contains(guidance));
    }

    let provider_tool = &value["tools"][1];
    assert_eq!(provider_tool["type"], "mcp_toolset");
    assert!(provider_tool.get("strict").is_none());
}

fn system_has_cache_control(value: &serde_json::Value) -> bool {
    value["system"]
        .as_array()
        .and_then(|blocks| blocks.last())
        .and_then(|block| block.get("cache_control"))
        .is_some()
}

fn last_message_has_cache_control(value: &serde_json::Value) -> bool {
    value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .and_then(|content| content.get("cache_control"))
        .is_some()
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        Some("Top-level instruction.".to_string()),
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Top-level instruction.");
    assert_eq!(value["system"][1]["text"], "Global history instruction.");

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "From now on, require explicit type annotations."
    );
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message_before_assistant_turn() {
    let request = completion_request_with_history(
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::assistant("I will enforce explicit type annotations."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(messages[2]["role"], "assistant");
    assert!(value.get("system").is_none());
}

#[test]
fn opus_4_8_hoists_leading_system_message_when_documents_are_present() {
    let mut request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::assistant("Acknowledged."),
            message::Message::System {
                content: "Mid-conversation instruction.".to_string(),
            },
            message::Message::user("Answer from the document."),
        ],
        None,
    );
    request.documents = vec![completion::Document {
        id: "doc".to_string(),
        text: "Document context.".to_string(),
        additional_props: Default::default(),
    }];

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Global history instruction.");
    assert_eq!(value["system"][1]["text"], "Mid-conversation instruction.");

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert!(
        messages[0].to_string().contains("<file id: doc>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .all(|message| message["role"].as_str() != Some("system"))
    );
}

/// The canonical history turn produced by a real Messages reply.
fn decoded_assistant(model: &str, content: Vec<Value>) -> message::Message {
    fold_reply(&json!({"id":"msg_history", "model":model, "role":"assistant", "content":content, "stop_reason":"end_turn", "usage":{"input_tokens":1,"output_tokens":1}}))
        .expect("provider content decodes").message().expect("an assistant turn")
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_result() {
    let request = completion_request_with_history(
        vec![
            decoded_assistant(
                CLAUDE_OPUS_4_8,
                vec![
                    json!({"type":"server_tool_use","id":"srvtoolu_01","name":"web_search","input":{"query":"clear daytime sky color"}}),
                    json!({"type":"web_search_tool_result","tool_use_id":"srvtoolu_01","content":{"type":"web_search_tool_result_error","error_code":"unavailable"}}),
                ],
            ),
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[0]["content"][1]["type"], "web_search_tool_result");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

#[test]
fn foreign_annotated_empty_text_produces_no_anthropic_block() {
    let foreign_annotated_empty = message::AssistantContent::text("")
        .with_native(
            json!({"type":"output_text","text":"","annotations":[{"type":"url_citation"}]}),
        )
        .canonical();
    let values = assistant_wire(vec![foreign_annotated_empty.clone()]);
    assert!(
        values.is_empty(),
        "a foreign-annotated empty block must produce no Anthropic content"
    );
    let values = assistant_wire(vec![
        foreign_annotated_empty,
        message::AssistantContent::text("real answer"),
    ]);
    assert_eq!(values.len(), 1, "only the real block survives");
    assert_eq!(values[0], json!({"type":"text","text":"real answer"}));
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_use() {
    let request = completion_request_with_history(
        vec![
            decoded_assistant(
                CLAUDE_OPUS_4_8,
                vec![
                    json!({"type":"server_tool_use","id":"srvtoolu_01","name":"web_search","input":{"query":"clear daytime sky color"}}),
                ],
            ),
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

#[test]
fn opus_4_8_hoists_system_message_in_invalid_mid_conversation_position() {
    let request = completion_request_with_history(
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::user("Now review this other file."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(
        value["system"][0]["text"],
        "From now on, require explicit type annotations."
    );

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn older_anthropic_models_hoist_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_7,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(
        value["system"][0]["text"],
        "From now on, require explicit type annotations."
    );

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_tool_definition_cache_control_serialization() {
    // G124: original DTO serde is now tested at the actual native RAW-tool
    // request stage; original tool values and both cache predicates remain.
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let encode_tool = |tool: Value| {
        let original = tool.clone();
        let request = CompletionRequest {
            additional_params: Some(json!({"tools": [tool]})),
            ..CompletionRequest::new("test")
        };
        let encoded = wire.encode(request, Mode::Unary)
            .expect("the exact original raw tool encodes on native Anthropic");
        let body = json_body(&encoded.request);
        let tools = body["tools"].as_array().expect("the actual encoded tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0], original);
        tools[0].clone()
    };
    let tool = json!({
        "name": "cached_tool",
        "description": "Cached tool",
        "input_schema": {"type": "object"},
        "strict": false,
        "cache_control": {"type": "ephemeral"}
    });

    let value = encode_tool(tool);
    assert_eq!(value["cache_control"]["type"], "ephemeral");

    let tool_without_cache = json!({
        "name": "uncached_tool",
        "description": "Uncached tool",
        "input_schema": {"type": "object"},
        "strict": false
    });

    let value = encode_tool(tool_without_cache);
    assert!(value.get("cache_control").is_none());
}

#[test]
fn test_apply_tool_cache_control_marks_only_final_tool() {
    let mut tools = vec![
        json!({
            "name": "first_tool",
            "description": "First tool",
            "input_schema": {"type": "object"}
        }),
        json!({
            "name": "second_tool",
            "description": "Second tool",
            "input_schema": {"type": "object"}
        }),
    ];

    // G125: existing private placement helper; the old signature and only
    // its caller-visible remaining==3 view are retired, not reconstructed.
    let wire = AnthropicConfig::new("test-key")
        .completion(CLAUDE_SONNET_4_6)
        .with_prompt_caching();
    apply_cache_control(&wire, false, None, &mut [], &mut [], &mut tools).unwrap();

    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_skips_final_deferred_tool_in_request() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_tool",
                    "description": "Deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["name"], "regular_tool");
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["name"], "deferred_tool");
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_existing_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
}

#[test]
fn test_prompt_caching_all_deferred_tools_do_not_receive_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_deferred_tool",
                    "description": "First deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                },
                {
                    "name": "second_deferred_tool",
                    "description": "Second deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_earlier_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "earlier_tool",
                    "description": "Earlier tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "later_tool",
                    "description": "Later tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_deferred_marker_does_not_suppress_loaded_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_cached_tool",
                    "description": "Deferred cached tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_errors_when_tool_cache_control_ttl_order_is_invalid() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_caching_preserves_valid_mixed_ttl_tool_cache_controls() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert!(tools[1]["cache_control"].get("ttl").is_none());
}

#[test]
fn test_prompt_caching_preserves_deferred_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "deferred_cached_tool",
                "description": "Deferred cached tool",
                "input_schema": {"type": "object"},
                "defer_loading": true,
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_budget_preserves_three_tool_markers_and_skips_message() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert!(system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_errors_when_explicit_tool_markers_exceed_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fifth_cached_tool",
                    "description": "Fifth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_prompt_caching_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_prompt_caching_replaces_null_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "final_tool",
                "description": "Final tool",
                "input_schema": {"type": "object"},
                "cache_control": null
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_ignores_null_tool_cache_control_when_budgeting() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_null_tool",
                    "description": "First null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "second_null_tool",
                    "description": "Second null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "third_null_tool",
                    "description": "Third null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "fourth_null_tool",
                    "description": "Fourth null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
    assert!(tools[2].get("cache_control").is_none());
    assert!(tools[3].get("cache_control").is_none());
    assert_eq!(tools[4]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_preserves_non_null_provider_tool_cache_control_escape_hatch() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "provider_specific"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "provider_specific");
}

#[test]
fn test_prompt_caching_automatic_mode_uses_reduced_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(!system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_automatic_mode_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_automatic_caching_errors_when_explicit_tool_markers_exhaust_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_automatic_caching_1h_errors_with_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_and_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_and_raw_top_level_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "test"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "test");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_uses_raw_top_level_cache_control_ttl() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "raw-cache-control"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "raw-cache-control");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_with_manual_caching_splits_prefix_and_tail() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The tail keeps the 5-minute default: a marker with no `ttl` field.
    let tail_cache_control = value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .map(|block| &block["cache_control"])
        .unwrap();
    assert_eq!(tail_cache_control["type"], "ephemeral");
    assert!(tail_cache_control.get("ttl").is_none());
    assert!(value.get("cache_control").is_none());
}

#[test]
fn test_static_prefix_ttl_with_automatic_caching_marks_prefix_only() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The moving tail breakpoint is Anthropic's top-level one at the
    // 5-minute default; no explicit message marker exists.
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(value["cache_control"].get("ttl").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_alone_marks_prefix_without_tail_or_top_level() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert!(value.get("cache_control").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_five_minutes_with_automatic_1h_errors_client_side() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let error = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl"),
        "error should name the knob: {message}"
    );
    assert!(
        message.contains("CacheRetention::Long"),
        "error should name the conflicting knob: {message}"
    );
}

#[test]
fn test_static_prefix_ttl_five_minutes_matches_automatic_default_ttl() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    // 5m prefix + 5m (default) top-level is uniform, not an inversion.
    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    // The explicit knob serializes an explicit `"5m"`, equivalent to the
    // omitted-`ttl` default.
    assert_eq!(tools[0]["cache_control"]["ttl"], "5m");
}

#[test]
fn test_static_prefix_ttl_preserves_marker_budget_arithmetic() {
    // Automatic mode reserves one marker for the top-level breakpoint; the
    // static-prefix knob spends from the same remaining budget as manual
    // caching does — two markers (final tool + system), no more.
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Short),
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let marker_count = value["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|tool| !tool["cache_control"].is_null())
        .count()
        + value["system"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| !block["cache_control"].is_null())
            .count()
        + usize::from(!value["cache_control"].is_null());
    assert_eq!(marker_count, 3);
    assert!(marker_count <= MAX_CACHE_CONTROL_MARKERS);
}

#[test]
fn test_usage_parses_per_ttl_cache_creation_breakdown() {
    let usage = json!({"input_tokens":3,"cache_read_input_tokens":0,"cache_creation_input_tokens":9677,"cache_creation":{"ephemeral_5m_input_tokens":9677,"ephemeral_1h_input_tokens":0,"ephemeral_24h_input_tokens":0},"output_tokens":7});
    let response = fold_reply(&json!({"id":"msg_usage","model":CLAUDE_SONNET_4_6,"role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":usage})).expect("usage decodes");
    assert_eq!(response.usage.cache_creation_input_tokens, Some(9677));
    assert_eq!(
        response.raw["usage"]["cache_creation"]["ephemeral_5m_input_tokens"],
        9677
    );
    assert_eq!(
        response.raw["usage"]["cache_creation"]["ephemeral_1h_input_tokens"],
        0
    );
}

#[test]
fn test_usage_without_cache_creation_breakdown_parses_as_none() {
    let response = fold_reply(&json!({"id":"msg_usage","model":CLAUDE_SONNET_4_6,"role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":7}})).expect("usage decodes");
    assert!(response.raw["usage"].get("cache_creation").is_none());
}

#[test]
fn test_raw_top_level_automatic_caching_reduces_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"},
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_raw_top_level_automatic_caching_1h_errors_after_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_typed_automatic_caching_ttl_errors_on_conflicting_raw_top_level_ttl() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"}
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("conflicts with the typed automatic caching TTL")
    );
}

#[test]
fn test_prompt_caching_marks_final_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("first_tool"), generic_tool("second_tool")],
        None,
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_marks_final_additional_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("rig_tool")],
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }],
            "metadata": {"source": "test"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["metadata"]["source"], "test");
}

#[test]
fn test_prompt_caching_without_tools_omits_tools() {
    let request = completion_request_with_tools(Vec::new(), None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("tools").is_none());
}

/// The original content bytes validated by the decoder's known-event parser.
fn validated_content(raw: &str) -> Value {
    let frame = format!(r#"{{"type":"content_block_start","index":0,"content_block":{raw}}}"#);
    let event: super::super::streaming::MessagesEvent = serde_json::from_str(&frame).unwrap();
    event.fields["content_block"].clone()
}

fn document_wire(data: DocumentData, media_type: Option<DocumentMediaType>) -> Value {
    let value = user_wire(message::UserContent::Document(message::Document {
        data,
        media_type,
        additional_params: None,
    }))
    .unwrap()
    .unwrap();
    value["content"][0].clone()
}

#[test]
fn test_plaintext_document_serialization() {
    let json = document_wire(
        DocumentData::Text("Hello, world!".into()),
        Some(DocumentMediaType::TXT),
    );
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "text");
    assert_eq!(json["source"]["media_type"], "text/plain");
    assert_eq!(json["source"]["data"], "Hello, world!");
}

#[test]
fn test_plaintext_document_deserialization() {
    let content = validated_content(
        r#"{"type":"document","source":{"type":"text","media_type":"text/plain","data":"Hello, world!"}}"#,
    );
    assert_eq!(
        content["source"],
        json!({"type":"text","data":"Hello, world!","media_type":"text/plain"})
    );
    assert!(content.get("cache_control").is_none());
}

#[test]
fn test_base64_pdf_document_serialization() {
    let json = document_wire(
        DocumentData::File(DocumentSourceKind::Base64("base64data".into())),
        Some(DocumentMediaType::PDF),
    );
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "base64");
    assert_eq!(json["source"]["media_type"], "application/pdf");
    assert_eq!(json["source"]["data"], "base64data");
}

#[test]
fn test_base64_pdf_document_deserialization() {
    let content = validated_content(
        r#"{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"base64data"}}"#,
    );
    assert_eq!(
        content["source"],
        json!({"type":"base64","data":"base64data","media_type":"application/pdf"})
    );
}

#[test]
fn test_file_id_document_serialization() {
    let json = document_wire(
        DocumentData::File(DocumentSourceKind::FileId("file_abc".into())),
        None,
    );
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "file");
    assert_eq!(json["source"]["file_id"], "file_abc");
}

#[test]
fn test_file_id_document_deserialization() {
    let content =
        validated_content(r#"{"type":"document","source":{"type":"file","file_id":"file_abc"}}"#);
    assert_eq!(
        content["source"],
        json!({"type":"file","file_id":"file_abc"})
    );
}

#[test]
fn test_file_id_rig_to_anthropic_conversion() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::File(DocumentSourceKind::FileId("file_abc".into())),
        media_type: None,
        additional_params: None,
    }))
    .unwrap()
    .unwrap();
    assert_eq!(value["role"], "user");
    assert_eq!(
        value["content"][0]["source"],
        json!({"type":"file","file_id":"file_abc"})
    );
}

#[test]
fn test_plaintext_rig_to_anthropic_conversion() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::Text("Some plain text content".into()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    }))
    .unwrap()
    .unwrap();
    assert_eq!(value["role"], "user");
    assert_eq!(
        value["content"][0]["source"],
        json!({"type":"text","data":"Some plain text content","media_type":"text/plain"})
    );
}

#[test]
fn test_unsupported_document_type_returns_error() {
    let result = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::Text("data".into()),
        media_type: Some(DocumentMediaType::HTML),
        additional_params: None,
    }));
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Anthropic only supports PDF and plain text documents"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_plaintext_document_url_source_returns_error() {
    let result = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::File(DocumentSourceKind::Url(
            "https://example.com/doc.txt".into(),
        )),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    }));
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Only string or base64 data is supported for plain text documents"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_plaintext_document_with_cache_control() {
    let request = completion_request_with_history(
        vec![message::Message::User {
            content: vec![message::UserContent::Document(message::Document {
                data: DocumentData::Text("cached text".into()),
                media_type: Some(DocumentMediaType::TXT),
                additional_params: None,
            })],
        }],
        None,
    );
    let body = request_body(Params {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    let json = &body["messages"][0]["content"][0];
    assert_eq!(json["source"]["type"], "text");
    assert_eq!(json["source"]["media_type"], "text/plain");
    assert_eq!(json["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_message_with_plaintext_document_deserialization() {
    let original = json!({"role":"user","content":[
        {"type":"document","source":{"type":"text","media_type":"text/plain","data":"Hello from a text file"}},
        {"type":"text","text":"Summarize this document."}
    ]});
    let mut envelope = original;
    envelope["type"] = json!("message");
    envelope["id"] = json!("msg_document");
    envelope["model"] = json!(CLAUDE_SONNET_4_6);
    envelope["usage"] = json!({"input_tokens": 1, "output_tokens": 0});
    let parsed: super::super::streaming::MessagesEvent =
        serde_json::from_str(&envelope.to_string()).unwrap();
    let content = parsed.fields["content"].as_array().unwrap();
    assert_eq!(parsed.fields["role"], "user");
    assert_eq!(content.len(), 2);
    assert_eq!(
        content[0]["source"],
        json!({"type":"text","data":"Hello from a text file","media_type":"text/plain"})
    );
    assert_eq!(content[1]["text"], "Summarize this document.");
}

#[test]
fn test_assistant_reasoning_multiblock_to_anthropic_content() {
    let reasoning = message::CarriedReasoning::new(
        vec![
            message::CarriedReasoningPart::Text {
                text: "step one".to_owned(),
                signature: Some("sig-1".to_owned()),
            },
            message::CarriedReasoningPart::Summary("summary".to_owned()),
            message::CarriedReasoningPart::Text {
                text: "step two".to_owned(),
                signature: Some("sig-2".to_owned()),
            },
            message::CarriedReasoningPart::Redacted {
                data: "redacted block".to_owned(),
            },
        ],
        message::CarriedPayload::new(Value::Null, None),
    );
    let converted = convert(message::Message::Assistant(message::AssistantMessage::new(
        vec![message::AssistantContent::CarriedReasoning(reasoning)],
    )))
    .expect("convert assistant message")
    .expect("one assistant turn");
    assert_eq!(converted["role"], "assistant");
    let content = converted["content"].as_array().expect("content blocks");
    assert_eq!(content.len(), 4);
    assert_eq!(
        content[0],
        json!({"type":"thinking","thinking":"step one","signature":"sig-1"})
    );
    assert_eq!(content[1], json!({"type":"thinking","thinking":"summary"}));
    assert_eq!(
        content[2],
        json!({"type":"thinking","thinking":"step two","signature":"sig-2"})
    );
    assert_eq!(
        content[3],
        json!({"type":"redacted_thinking","data":"redacted block"})
    );
}

#[test]
fn test_assistant_encrypted_reasoning_maps_to_redacted_thinking() {
    let reasoning = message::CarriedReasoning::new(
        vec![message::CarriedReasoningPart::Encrypted(
            "ciphertext".to_owned(),
        )],
        message::CarriedPayload::new(Value::Null, None),
    );
    let converted = assistant_wire(vec![message::AssistantContent::CarriedReasoning(reasoning)]);
    assert_eq!(converted.len(), 1);
    assert_eq!(
        converted[0],
        json!({"type":"redacted_thinking","data":"ciphertext"})
    );
}

/// An empty wire reply with the given terminal.
fn empty_response_with(stop_reason: Option<&str>, stop_sequence: Option<&str>) -> Value {
    json!({
        "content": [], "id": "msg_123", "model": CLAUDE_SONNET_4_6,
        "role": "assistant", "stop_reason": stop_reason, "stop_sequence": stop_sequence,
        "usage": {"input_tokens": 7, "output_tokens": 2}
    })
}

#[test]
fn empty_end_turn_response_normalizes_to_an_empty_choice() {
    let parsed = fold_reply(&empty_response_with(Some("end_turn"), None))
        .expect("empty end_turn should not error");
    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.provider(), "anthropic");
    assert_eq!(
        parsed
            .message_id
            .as_ref()
            .map(completion::ProviderMessageId::as_str),
        Some("msg_123")
    );
    assert_eq!(parsed.model(), Some(CLAUDE_SONNET_4_6));
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn empty_response_outside_the_legal_terminals_still_errors() {
    for (stop_reason, stop_sequence) in [
        (Some("tool_use"), None),
        (Some("max_tokens"), None),
        (Some("refusal"), None),
        (Some("pause_turn"), None),
        (None, None),
        (Some("stop_sequence"), None),
        (Some("max_tokens"), Some("alpha")),
    ] {
        let err = fold_reply(&empty_response_with(stop_reason, stop_sequence)).expect_err(
            &format!("empty {stop_reason:?} response should remain an error"),
        );
        assert!(matches!(err, ProviderError::Response(message) if message == EMPTY_RESPONSE_ERROR));
    }
}

#[test]
fn empty_stop_sequence_response_naming_its_sequence_is_a_completed_turn() {
    let parsed = fold_reply(&empty_response_with(Some("stop_sequence"), Some("alpha")))
        .expect("a completed stop-sequence turn must not fold into an error");
    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn stop_reason_maps_onto_the_normalized_vocabulary() {
    for (reason, expected) in [
        ("end_turn", completion::FinishReason::Stop),
        ("stop_sequence", completion::FinishReason::Stop),
        ("max_tokens", completion::FinishReason::Length),
        ("tool_use", completion::FinishReason::ToolCalls),
        ("refusal", completion::FinishReason::ContentFilter),
    ] {
        let mut response = empty_response_with(Some(reason), None);
        response["content"] = json!([{"type": "text", "text": "answer"}]);
        let parsed = fold_reply(&response).expect("the response folds");
        assert_eq!(parsed.finish_reason(), Some(expected));
    }
}

#[test]
fn unknown_stop_reason_is_preserved_verbatim() {
    for reason in ["pause_turn", "model_context_window_exceeded"] {
        let mut response = empty_response_with(Some(reason), None);
        response["content"] = json!([{"type": "text", "text": "answer"}]);
        let parsed = fold_reply(&response).expect("the response folds");
        assert_eq!(
            parsed.finish_reason(),
            Some(completion::FinishReason::Other(reason.to_owned()))
        );
    }
}

#[test]
fn end_turn_with_a_tool_call_is_reconciled_to_tool_calls() {
    let mut response = empty_response_with(Some("end_turn"), None);
    response["content"] =
        json!([{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 1}}]);
    let parsed = fold_reply(&response).expect("tool-use response should fold");
    assert_eq!(
        parsed.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

#[test]
fn test_tool_result_content_in_message_roundtrip() {
    let message_json = r#"{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the screenshot:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo..."
                            }
                        }
                    ]
                }
            ]
        }"#;

    let fixture: Value = serde_json::from_str(message_json).unwrap();
    let call = message::ToolCall::from_wire(
        "toolu_01A09q90qw90lq917835lq9",
        message::ToolFunction::new(message::ToolName::new("screenshot").unwrap(), json!({})),
    );
    let image = message::Image {
        data: message::DocumentSourceKind::base64("iVBORw0KGgo..."),
        media_type: Some(message::ImageMediaType::PNG),
        ..Default::default()
    };
    let result = call.result(vec![
        message::ToolResultContent::text("Here is the screenshot:"),
        message::ToolResultContent::Image(image),
    ]);
    let serialized = user_wire(message::UserContent::ToolResult(result))
        .unwrap()
        .unwrap();
    assert_eq!(serialized, fixture);
    assert_eq!(
        validated_content(&serialized["content"][0].to_string()),
        serialized["content"][0]
    );

    let tool_result = &serialized["content"][0];
    assert_eq!(tool_result["type"], "tool_result");

    let image_content = &tool_result["content"][1];
    assert_eq!(image_content["type"], "image");
    assert_eq!(image_content["source"]["type"], "base64");
    assert_eq!(image_content["source"]["media_type"], "image/png");
    assert_eq!(image_content["source"]["data"], "iVBORw0KGgo...");
}

// -------------------------------------------------------------------
// Citations (#1767)
// -------------------------------------------------------------------

#[test]
fn document_serializes_citations_and_metadata() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::Text("hello".into()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: Some(json!({"title":"My Doc", "citations":{"enabled":true}})),
    }))
    .unwrap()
    .unwrap();
    let value = &value["content"][0];
    assert_eq!(value["citations"]["enabled"], true);
    assert_eq!(value["title"], "My Doc");
    assert!(
        value.get("context").is_none(),
        "context should be skipped when None"
    );
}

#[test]
fn text_serializes_without_citations_when_empty() {
    let values = assistant_wire(vec![message::AssistantContent::text("hello")]);
    assert!(
        values[0].get("citations").is_none(),
        "empty citations vec must be skipped"
    );
}

#[test]
fn text_deserializes_char_location_citation() {
    let value = json!({
        "type": "text",
        "text": "the grass is green",
        "citations": [{
            "type": "char_location",
            "cited_text": "The grass is green.",
            "document_index": 0,
            "document_title": "Example",
            "start_char_index": 0,
            "end_char_index": 20
        }]
    });
    let parsed = validated_content(&value.to_string());
    assert_eq!(parsed["type"], "text");
    let citations = parsed["citations"].as_array().unwrap();
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["type"], "char_location");
    assert_eq!(citations[0]["start_char_index"], 0);
    assert_eq!(citations[0]["end_char_index"], 20);
}

#[test]
fn text_deserializes_search_result_location_citation() {
    let value = json!({
        "type": "text",
        "text": "API keys are required.",
        "citations": [{
            "type": "search_result_location",
            "cited_text": "All API requests must include an API key.",
            "source": "https://docs.example.com/api-reference",
            "title": "API Reference",
            "search_result_index": 0,
            "start_block_index": 0,
            "end_block_index": 1
        }]
    });

    let parsed = validated_content(&value.to_string());
    assert_eq!(parsed["type"], "text");
    let citation = &parsed["citations"][0];
    assert_eq!(citation["type"], "search_result_location");
    assert_eq!(citation["source"], "https://docs.example.com/api-reference");
    assert_eq!(citation["title"], "API Reference");
    assert_eq!(citation["search_result_index"], 0);
    assert_eq!(citation["start_block_index"], 0);
    assert_eq!(citation["end_block_index"], 1);
}

#[test]
fn text_deserializes_web_search_result_location_citation() {
    let value = json!({
        "type": "text",
        "text": "Claude Shannon worked at Bell Labs.",
        "citations": [{
            "type": "web_search_result_location",
            "cited_text": "Claude Shannon was a mathematician.",
            "url": "https://example.com/shannon",
            "title": "Claude Shannon",
            "encrypted_index": "encrypted-reference"
        }]
    });

    let parsed = validated_content(&value.to_string());
    assert_eq!(parsed["type"], "text");
    let citation = &parsed["citations"][0];
    assert_eq!(citation["type"], "web_search_result_location");
    assert_eq!(citation["url"], "https://example.com/shannon");
    assert_eq!(citation["title"], "Claude Shannon");
    assert_eq!(citation["encrypted_index"], "encrypted-reference");
}

#[test]
fn text_deserializes_web_search_result_location_citation_with_null_title() {
    let value = json!({
        "type": "text",
        "text": "Claude Shannon worked at Bell Labs.",
        "citations": [{
            "type": "web_search_result_location",
            "cited_text": "Claude Shannon was a mathematician.",
            "url": "https://example.com/shannon",
            "title": null,
            "encrypted_index": "encrypted-reference"
        }]
    });

    let parsed = validated_content(&value.to_string());
    assert_eq!(parsed["type"], "text");
    let citation = &parsed["citations"][0];
    assert_eq!(citation["type"], "web_search_result_location");
    assert!(citation["title"].is_null());
    let serialized = serde_json::to_value(citation).unwrap();
    assert!(serialized.get("title").is_some());
    assert!(serialized["title"].is_null());
}

#[test]
fn web_search_response_preserves_raw_blocks_and_citations() {
    let value = json!({
        "id": "msg_web_search",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        },
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/shannon",
                        "title": "Claude Shannon",
                        "encrypted_content": "encrypted-content",
                        "page_age": "April 30, 2025"
                    }
                ]
            },
            {
                "type": "text",
                "text": "Claude Shannon was born on April 30, 1916.",
                "citations": [{
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was born on April 30, 1916.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-index"
                }]
            }
        ]
    });

    let converted = fold_reply(&value).expect("the hosted-tool reply folds");
    assert_eq!(converted.choice.len(), 3);
    let message::AssistantContent::CarriedText(server_tool_use) = &converted.choice[0] else {
        panic!("expected carried server_tool_use");
    };
    assert_eq!(server_tool_use.text(), "");
    assert_eq!(
        server_tool_use.payload.provider.as_deref(),
        Some(super::super::PROVIDER_NAME)
    );
    assert_eq!(server_tool_use.payload.item["type"], "server_tool_use");
    assert_eq!(server_tool_use.payload.item, value["content"][0]);
    let message::AssistantContent::CarriedText(web_search_result) = &converted.choice[1] else {
        panic!("expected carried web_search_tool_result");
    };
    assert_eq!(
        web_search_result.payload.item["content"][0]["encrypted_content"],
        "encrypted-content"
    );
    assert_eq!(web_search_result.payload.item, value["content"][1]);
    let message::AssistantContent::Text(answer) = &converted.choice[2] else {
        panic!("expected text answer");
    };
    assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
    let native = converted.choice[2]
        .native_item()
        .expect("the exact citation wire item survives");
    assert_eq!(native["citations"][0]["type"], "web_search_result_location");
    assert_eq!(native["citations"][0]["encrypted_index"], "encrypted-index");
    let round_trip = convert(converted.message().expect("an assistant turn"))
        .unwrap()
        .unwrap();
    assert_eq!(round_trip["content"], value["content"]);
    assert_eq!(round_trip["content"][0]["id"], "srvtoolu_01");
    assert_eq!(round_trip["content"][0]["name"], "web_search");
    assert_eq!(
        round_trip["content"][0]["input"]["query"],
        "claude shannon birth date"
    );
    assert_eq!(round_trip["content"][1]["tool_use_id"], "srvtoolu_01");
    assert_eq!(
        round_trip["content"][1]["content"][0]["encrypted_content"],
        "encrypted-content"
    );
}

#[test]
fn web_search_tool_result_error_object_is_preserved_raw() {
    let value = json!({
        "id": "msg_web_search_error",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2
        },
        "content": [{
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded"
            }
        }]
    });

    let converted = fold_reply(&value).expect("the reply folds");
    assert_eq!(converted.choice.len(), 1);
    let message::AssistantContent::CarriedText(web_search_result) = &converted.choice[0] else {
        panic!("expected carried web_search_tool_result");
    };
    let raw_content = &web_search_result.payload.item;
    assert_eq!(raw_content["type"], "web_search_tool_result");
    assert_eq!(raw_content["content"]["error_code"], "max_uses_exceeded");
    assert_eq!(
        raw_content["content"]["type"],
        "web_search_tool_result_error"
    );
    let round_trip = convert(converted.message().expect("an assistant turn"))
        .unwrap()
        .unwrap();
    assert_eq!(round_trip["content"], value["content"]);
    assert_eq!(round_trip["content"][0]["tool_use_id"], "srvtoolu_01");
    assert_eq!(
        round_trip["content"][0]["content"]["error_code"],
        "max_uses_exceeded"
    );
}

#[test]
fn code_execution_tool_result_variants_deserialize() {
    let normal = json!({"type":"code_execution_tool_result", "tool_use_id":"srvtoolu_normal",
        "content":{"type":"code_execution_result", "return_code":0, "stdout":"42\n", "stderr":"", "content":[]}});
    let parsed = validated_content(&normal.to_string());
    assert_eq!(parsed, normal);
    assert_eq!(parsed["tool_use_id"], "srvtoolu_normal");
    assert_eq!(parsed["content"]["type"], "code_execution_result");
    assert_eq!(parsed["content"]["stdout"], "42\n");
    let encrypted = json!({"type":"code_execution_tool_result", "tool_use_id":"srvtoolu_encrypted",
        "content":{"type":"encrypted_code_execution_result", "return_code":1, "stderr":"failure",
            "encrypted_stdout":"encrypted-output", "content":[]}});
    let parsed = validated_content(&encrypted.to_string());
    assert_eq!(parsed, encrypted);
    assert_eq!(parsed["tool_use_id"], "srvtoolu_encrypted");
    assert_eq!(parsed["content"]["type"], "encrypted_code_execution_result");
    assert_eq!(parsed["content"]["encrypted_stdout"], "encrypted-output");
}

#[test]
fn code_execution_tool_result_is_preserved_and_round_trips() {
    let raw_block = json!({
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": {
            "type": "code_execution_result",
            "return_code": 0,
            "stdout": "42\n",
            "stderr": "",
            "content": []
        }
    });
    let value = json!({
        "id": "msg_code_execution",
        "model": CLAUDE_OPUS_4_8,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        },
        "content": [raw_block]
    });

    let converted = fold_reply(&value).expect("the reply folds");
    assert_eq!(converted.choice.len(), 1);
    let message::AssistantContent::CarriedText(code_execution_result) = &converted.choice[0] else {
        panic!("expected carried code_execution_tool_result");
    };
    assert_eq!(code_execution_result.payload.item, raw_block);
    let round_trip = convert(converted.message().expect("an assistant turn"))
        .unwrap()
        .unwrap();
    assert_eq!(round_trip["content"][0], raw_block);
    assert_eq!(round_trip["content"][0]["tool_use_id"], "srvtoolu_01");
    assert_eq!(
        round_trip["content"][0]["content"]["type"],
        "code_execution_result"
    );
    assert_eq!(round_trip["content"][0]["content"]["stdout"], "42\n");
}

#[test]
fn text_deserializes_unknown_citation_without_failing() {
    let value = json!({
        "type": "text",
        "text": "future citation",
        "citations": [{
            "type": "future_location",
            "cited_text": "future text",
            "new_field": "kept"
        }]
    });

    let parsed = validated_content(&value.to_string());
    assert_eq!(parsed["type"], "text");
    let citation = &parsed["citations"][0];
    assert_eq!(citation["type"], "future_location");
    assert_eq!(citation["new_field"], "kept");
}

#[test]
fn page_location_citation_roundtrips() {
    let citation = json!({"type":"page_location", "cited_text":"Water is essential for life.",
        "document_index":1, "document_title":"PDF Doc", "start_page_number":5, "end_page_number":6});
    let value = serde_json::to_value(&citation).unwrap();
    assert_eq!(value["type"], "page_location");
    assert_eq!(value["start_page_number"], 5);
    let parsed = validated_content(
        &json!({"type":"text", "text":"Water is essential for life.", "citations":[value]})
            .to_string(),
    );
    assert_eq!(parsed["citations"][0], citation);
}

#[test]
fn content_block_location_citation_roundtrips() {
    let citation = json!({"type":"content_block_location", "cited_text":"These are important findings.",
        "document_index":2, "start_block_index":0, "end_block_index":1});
    let value = serde_json::to_value(&citation).unwrap();
    assert_eq!(value["type"], "content_block_location");
    assert!(value.get("document_title").is_none());
    let parsed = validated_content(
        &json!({"type":"text", "text":"These are important findings.", "citations":[value]})
            .to_string(),
    );
    assert_eq!(parsed["citations"][0], citation);
}

#[test]
fn anthropic_citations_extracts_from_additional_params() {
    let response = fold_reply(&json!({"id":"msg_citation","model":CLAUDE_SONNET_4_6,"role":"assistant","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"the grass is green","citations":[{"type":"char_location","cited_text":"The grass is green.","document_index":0,"start_char_index":0,"end_char_index":20}]}]})).expect("cited text decodes");
    let message::AssistantContent::Text(text) = &response.choice[0] else {
        panic!("text");
    };
    assert_eq!(text.citations().len(), 1);
}

#[test]
fn anthropic_citations_returns_empty_when_absent() {
    let text = message::Text::new("hello");
    assert!(text.citations().is_empty());
}

#[test]
fn assistant_text_citations_survive_anthropic_request_conversion() {
    let citation = json!({"type":"char_location","cited_text":"The grass is green.","document_index":0,"start_char_index":0,"end_char_index":20});
    let text = message::AssistantContent::text("the grass is green")
        .with_native(json!({"type":"text","text":"the grass is green","citations":[citation]}));
    let parts = assistant_wire(vec![text]);
    assert_eq!(parts[0]["text"], "the grass is green");
    assert_eq!(parts[0]["citations"], json!([citation]));
}

#[test]
fn assistant_text_invalid_known_citations_are_rejected_for_anthropic_request_conversion() {
    let text = message::AssistantContent::text("bad citation").with_native(json!({"type":"text","text":"bad citation","citations":[{"type":"char_location","cited_text":"bad"}]}));
    let result = convert(message::Message::Assistant(message::AssistantMessage::new(
        vec![text],
    )));
    assert!(
        result.is_err(),
        "invalid Anthropic citation metadata should not be silently dropped"
    );
}

#[test]
fn document_additional_params_forward_to_anthropic_document() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: DocumentData::Text("Hello world.".into()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: Some(
            json!({"title":"Doc1", "context":"ctx", "citations":{"enabled":true}}),
        ),
    }))
    .unwrap()
    .unwrap();
    let block = &value["content"][0];
    assert_eq!(block["type"], "document");
    assert_eq!(block["title"], "Doc1");
    assert_eq!(block["context"], "ctx");
    assert_eq!(block["citations"], json!({"enabled":true}));
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire, http)
        .call(hello_request())
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_2xx_error_envelope_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    // Anthropic answers an in-band failure with its top-level error
    // envelope, and it can arrive under a 200: the decoder models `error`
    // as an event of the Messages wire, so the envelope routes through the
    // same `ProviderResponse` funnel as a rejected status rather than
    // reading as a corrupt frame.
    //
    // The body is asserted byte-for-byte, in the key order Anthropic's
    // recorded replies use, and carries the top-level `request_id` the
    // envelope type does not model: a preserved reply that Rig rebuilt from
    // the fields it happens to parse is not the provider's reply. The
    // status is the driver's — a success at the transport layer is still a
    // status the caller must see.
    let body = r#"{"error":{"message":"model overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let http = RecordingHttpClient::new(body); // 200 OK
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire, http)
        .call(hello_request())
        .await
        .expect_err("completion should fail with provider error envelope");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
}

#[tokio::test]
async fn completion_streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;
    use futures::StreamExt;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = HttpErrorStreamingClient::new(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let stream = crate::driver::Model::new(wire, http)
        .stream(hello_request())
        .expect("the streamed request encodes");
    let mut stream = Box::pin(stream);

    // The transport failure surfaces as the first error item yielded by the stream.
    let error = loop {
        match stream.next().await {
            Some(Ok(_)) => continue,
            Some(Err(error)) => break error,
            None => panic!("stream ended without yielding the transport error"),
        }
    };

    // A rejected SSE handshake is the provider's reply, classified like the
    // unary driver's and the in-band envelopes': one funnel.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));

    // The transport failure ends the stream: nothing may follow it that
    // would read as a successfully completed turn.
    assert!(stream.next().await.is_none());
}

#[test]
fn coerce_tool_input_normalizes_non_object_arguments() {
    let name = message::ToolName::new("lookup").expect("tool name");
    for (input, expected) in [
        (json!({"q":"rust","n":3}), json!({"q":"rust","n":3})),
        (json!("{\"q\":\"rust\"}"), json!({"q":"rust"})),
        (json!("not json"), json!({})),
        (json!("[1,2,3]"), json!({})),
        (Value::Null, json!({})),
        (json!([1, 2, 3]), json!({})),
        (json!(42), json!({})),
        (json!(true), json!({})),
    ] {
        let call =
            message::ToolCall::from_wire("call_1", message::ToolFunction::new(name.clone(), input));
        let parts = assistant_wire(vec![message::AssistantContent::ToolCall(call)]);
        assert_eq!(parts[0]["input"], expected);
    }
}

// Regression test for issue #1429: PR #1431 added the `DocumentSource::Url`
// wire variant and response-side parsing, but the request-side
// `UserContent::Document` conversion still rejected URL-backed PDFs even
// though the Anthropic Messages API supports
// `"source": {"type": "url", ...}` for PDFs.
// The media type is optional because Anthropic's URL source is implicitly a
// PDF and does not include a media-type field on the wire.
//
// See <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
// for URL-sourced PDF documents.
#[test]
fn url_pdf_with_or_without_media_type_converts_to_url_document_source() {
    let pdf_url = "https://example.com/resume.pdf";

    for media_type in [Some(message::DocumentMediaType::PDF), None] {
        let msg = message::Message::User {
            content: vec![message::UserContent::document_url(pdf_url, media_type)],
        };

        let json = convert(msg)
            .expect("URL PDF should convert")
            .expect("one user turn");

        assert_eq!(
            json.pointer("/content/0/source"),
            Some(&json!({ "type": "url", "url": pdf_url })),
            "URL PDF should map to a url document source: {json:#}"
        );
    }
}

/// Raw-capture tests: `CompletionResponse::raw` driven end to end over a
/// mock transport that hands back a Messages body *and* a `request-id`
/// response header. `raw` is the verbatim reply document — not a
/// re-serialization of the parsed type — so it answers what the normalized
/// response does not, and the transport id the driver read off the headers
/// reaches the normalized response rather than the document.
/// `with_error_response_headers` with `200 OK` is the one unary double
/// that carries response headers.
mod raw_capture;

/// Synthetic transcript tests required-ID request correlation without a paid call.
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let name = message::ToolName::new("test").expect("tool name");
    let id = message::CallId::Local(message::LocalCallId::new());
    let local = message::ToolCall::new(id, message::ToolFunction::new(name.clone(), json!({})));
    let genuine =
        message::ToolCall::from_wire("tool-0", message::ToolFunction::new(name, json!({})));
    let result = |call: &message::ToolCall| {
        let mut result = call.result(vec![]);
        result.name = message::ToolName::new("possibly_repaired").expect("tool name");
        message::Message::User {
            content: vec![message::UserContent::ToolResult(result)],
        }
    };
    let assistant = |call: &message::ToolCall| {
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ]))
    };
    let request = CompletionRequest::from(vec![
        message::Message::system("system"),
        assistant(&local),
        assistant(&genuine),
        result(&genuine),
        message::Message::assistant("intervening text"),
        result(&local),
        assistant(&local),
        result(&local),
    ])
    .max_tokens(128);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let value = serde_json::to_value(
        body(&wire, &request, Mode::Unary).expect("the private request encodes"),
    )
    .unwrap();
    let content = value["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .flat_map(|message| message["content"].as_array().expect("content"));
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for part in content {
        match part["type"].as_str() {
            Some("tool_use") => calls.push(part["id"].as_str().expect("call ID").to_owned()),
            Some("tool_result") => results.push(
                part["tool_use_id"]
                    .as_str()
                    .expect("answered ID")
                    .to_owned(),
            ),
            _ => {}
        }
    }
    assert_eq!(calls.len(), 3, "{value}");
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()],
        "{value}"
    );
    assert_eq!(calls[1], "tool-0");
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "{value}"
    );
}

/// Hosted handles remain exact while fresh local calls receive a distinct wire alias.
#[test]
fn request_reserves_raw_server_tool_handles_without_rewriting_them() {
    for hosted_id in ["srvtoolu_real", "tool-0"] {
        let raw_call = json!({"type":"server_tool_use","id":hosted_id,"name":"web_search","input":{"query":"test"}});
        let raw_result =
            json!({"type":"web_search_tool_result","tool_use_id":hosted_id,"content":[]});
        let local = message::ToolCall::new(
            message::CallId::Local(message::LocalCallId::new()),
            message::ToolFunction::new(
                message::ToolName::new("local").expect("tool name"),
                json!({}),
            ),
        );
        assert!(local.id.is_local());
        let message::Message::Assistant(mut turn) = decoded_assistant(
            CLAUDE_SONNET_4_6,
            vec![raw_call.clone(), raw_result.clone()],
        ) else {
            panic!("assistant");
        };
        turn.content
            .insert(0, message::AssistantContent::ToolCall(local.clone()));
        let history = vec![
            message::Message::Assistant(turn),
            message::Message::User {
                content: vec![message::UserContent::ToolResult(local.result(vec![]))],
            },
        ];
        let mut request = completion_request_with_history(history, None);
        request.tools = vec![generic_tool("local")];
        let error = request_body(Params {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            cache: None,
            static_prefix_cache_ttl: None,
        })
        .expect_err("G152: the original empty Local result is refused");
        let error = ProviderError::from(error);
        assert!(matches!(&error, ProviderError::Request(_)));
        assert!(error.to_string().contains(
            "tool result for `local` at index 0 of the user message at index 1 has no content"
        ), "{error}");
    }
}

#[test]
fn messages_refuses_namespaced_and_custom_calls() {
    let name = message::ToolName::new("tool").expect("tool name");
    let mut function = message::ToolFunction::new(name.clone(), json!({}));
    function.namespace = Some("".to_owned());
    let namespaced =
        message::AssistantContent::ToolCall(message::ToolCall::from_wire("call_1", function));
    let custom = message::AssistantContent::custom_tool_call(
        "call_2",
        name,
        Some("ns".to_owned()),
        "{raw input}",
    );
    for (content, custom) in [(namespaced, false), (custom, true)] {
        let error: ProviderError = convert(message::Message::Assistant(
            message::AssistantMessage::new(vec![content]),
        ))
        .expect_err("refused, not dropped")
        .into();
        let ProviderError::Request(cause) = error else {
            panic!("the refusal is a typed request failure");
        };
        let cause = cause
            .downcast_ref::<message::UnrepresentableToolCall>()
            .expect("the typed cause survives");
        if custom {
            assert!(
                matches!(cause, message::UnrepresentableToolCall::CustomCall { wire, name }
                if *wire == ANTHROPIC_MESSAGES_WIRE && name.as_str() == "tool")
            );
        } else {
            assert!(
                matches!(cause, message::UnrepresentableToolCall::Namespace { wire, name, namespace }
                if *wire == ANTHROPIC_MESSAGES_WIRE && name.as_str() == "tool" && namespace.is_empty())
            );
        }
    }
}

#[test]
fn messages_refuses_carried_provider_items_before_adapting_failed_history() {
    use crate::wire::{Operation, Wire};
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    for issuer in [None, Some("anthropic"), Some("foreign/Responses")] {
        let item = message::ProviderItem {
            payload: message::CarriedPayload::new(
                json!({"type":"future_item", "payload": [1, 2]}),
                issuer.map(str::to_owned),
            ),
        };
        let turn =
            message::AssistantMessage::new(vec![message::AssistantContent::ProviderItem(item)])
                .with_stop(message::StopReason::Error("failed turn".to_owned()));
        let request = CompletionRequest::from(vec![
            message::Message::Assistant(turn),
            message::Message::user("continue"),
        ]);
        let error = crate::operation::Completion::prepare(request, &wire.describe())
            .expect_err("the original failed turn's item must be refused before adaptation");
        let ProviderError::Request(cause) = error else {
            panic!("the refusal is a typed request failure");
        };
        let cause = cause
            .downcast_ref::<message::UnreplayableProviderItem>()
            .expect("the typed cause survives");
        assert_eq!(cause.issuer.as_deref(), issuer);
        assert_eq!(cause.item_type.as_deref(), Some("future_item"));
    }
}

#[test]
fn messages_refuses_foreign_or_edited_carried_hosted_blocks_before_adaptation() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    for (issuer, text) in [
        (None, ""),
        (Some("foreign/Responses"), ""),
        (Some(super::super::PROVIDER_NAME), "an edit"),
    ] {
        let payload = message::CarriedPayload::new(
            json!({"type":"server_tool_use", "id":"srvtoolu_01",
            "name":"web_search", "input":{"query":"test"}}),
            issuer.map(str::to_owned),
        );
        let part = message::CarriedText::new(
            message::CarriedTextContent::Opaque(text.to_owned()),
            payload,
        );
        let turn =
            message::AssistantMessage::new(vec![message::AssistantContent::CarriedText(part)])
                .with_stop(message::StopReason::Error("failed turn".to_owned()));
        let request = CompletionRequest::from(vec![
            message::Message::Assistant(turn),
            message::Message::user("continue"),
        ]);
        let error = Completion::prepare(request, &wire.describe())
            .expect_err("original carried part is refused");
        let ProviderError::Request(cause) = error else {
            panic!("typed request error");
        };
        assert!(
            cause
                .downcast_ref::<message::UnrepresentableOpaqueContent>()
                .is_some()
        );
    }
}

#[test]
fn current_provider_items_replay_verbatim() {
    let thinking = json!({"type": "thinking", "thinking": "step one", "signature": "sig-1"});
    let redacted = json!({"type": "redacted_thinking", "data": "redacted block"});
    let call = json!({"type": "tool_use", "id": "toolu_1", "name": "add",
        "input": {"x": 1}, "caller": {"type": "direct"}});
    let text = json!({"type": "text", "text": "Two.", "citations": [{"type": "char_location",
        "cited_text": "2", "document_index": 0, "start_char_index": 0, "end_char_index": 1}]});
    let add = message::ToolName::new("add").expect("tool name");
    let wire = assistant_wire(vec![
        message::AssistantContent::reasoning("step one").with_native(thinking.clone()),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        })
        .with_native(redacted.clone()),
        message::AssistantContent::tool_call("toolu_1", add, json!({"x": 1}))
            .with_native(call.clone()),
        message::AssistantContent::text("Two.").with_native(text.clone()),
    ]);
    assert_eq!(wire, [thinking, redacted, call, text]);
}

#[test]
fn blocks_without_a_current_item_rebuild_as_pi_does() {
    let unsigned = message::AssistantContent::reasoning("thought");
    let redacted = message::AssistantContent::Reasoning(message::Reasoning {
        redacted: true,
        ..message::Reasoning::default()
    });
    let mut edited = message::AssistantContent::text("before")
        .with_native(json!({"type": "text", "text": "before", "citations": []}));
    if let message::AssistantContent::Text(text) = &mut edited {
        text.text = "after".to_owned();
    }
    let wire = assistant_wire(vec![unsigned, redacted, edited]);
    assert_eq!(
        wire,
        [
            json!({"type": "text", "text": "thought"}),
            json!({"type": "text", "text": "after"})
        ]
    );
}

#[test]
fn a_tool_use_whose_input_is_not_an_object_keeps_its_call() {
    let value = json!({
        "id": "msg_bad", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": [1]}]
    });
    let response = fold_reply(&value).expect("the reply folds");
    let [message::AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({}));
    assert_eq!(call.function.invalid_arguments.as_deref(), Some("[1]"));
    assert!(response.choice[0].native_item().is_none());
}

#[test]
fn an_assistant_image_is_sent_as_its_placeholder() {
    let image = message::AssistantContent::Image(message::Image {
        data: message::DocumentSourceKind::base64("aGk="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    });
    assert_eq!(
        assistant_wire(vec![image]),
        [json!({"type": "text", "text": crate::completion::history::ASSISTANT_IMAGE_OMITTED})]
    );
}

#[test]
fn encodes_states_exactly_the_media_the_encoder_carries() {
    use crate::completion::{Media, Place, ReplayTarget};
    use message::{
        Audio, AudioMediaType, Document, DocumentMediaType as Doc, DocumentSourceKind as Source,
        Image, ImageMediaType, Video, VideoMediaType,
    };

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let image = |data: Source, media_type: Option<ImageMediaType>| Image {
        data,
        media_type,
        ..Image::default()
    };
    let document = |data: Source, media_type: Option<Doc>| Document {
        data: data.into(),
        media_type,
        additional_params: None,
    };
    let text = |text: &str, media_type: Option<Doc>| Document {
        data: message::DocumentData::Text(text.into()),
        media_type,
        additional_params: None,
    };
    let images = [
        (
            image(Source::base64("aGk="), Some(ImageMediaType::PNG)),
            true,
        ),
        (image(Source::base64("aGk="), None), false),
        (
            image(Source::base64("aGk="), Some(ImageMediaType::HEIC)),
            false,
        ),
        (image(Source::url("https://example.com/a.png"), None), true),
        (image(Source::file_id("file_image"), None), true),
    ];
    for (image, carried) in images {
        let in_result = message::UserContent::ToolResult(message::ToolResult {
            call: message::CallId::from_wire("toolu_1"),
            name: message::ToolName::new("shot").expect("tool name"),
            answers: message::AnsweredToolCall::Function,
            content: vec![message::ToolResultContent::Image(image.clone())],
            is_error: false,
        });
        for (place, part) in [
            (Place::User, message::UserContent::Image(image.clone())),
            (Place::ToolResult, in_result),
        ] {
            assert_eq!(
                wire.encodes(CLAUDE_SONNET_4_6, Media::Image(&image, place)),
                carried,
                "{image:?} at {place:?}"
            );
            assert_eq!(user_wire(part).is_ok(), carried, "{image:?} at {place:?}");
        }
        assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Image(&image, Place::Assistant)));
    }
    let documents = [
        (document(Source::base64("JVBERi0="), Some(Doc::PDF)), true),
        (
            document(Source::url("https://example.com/a.pdf"), Some(Doc::PDF)),
            true,
        ),
        (
            document(Source::url("https://example.com/a.pdf"), None),
            true,
        ),
        (
            document(Source::url("https://example.com/a.txt"), Some(Doc::TXT)),
            false,
        ),
        (document(Source::file_id("file_doc"), None), true),
        (text("plain", Some(Doc::TXT)), true),
        (text("plain", None), true),
        (document(Source::base64("cGxhaW4="), Some(Doc::TXT)), true),
        (
            document(Source::base64("cmlnLG1hdHJpeAo="), Some(Doc::CSV)),
            false,
        ),
        (text("<p>hi</p>", Some(Doc::HTML)), false),
        (document(Source::base64("//79"), Some(Doc::CSV)), false),
        (document(Source::base64("cGxhaW4="), None), false),
    ];
    for (document, carried) in documents {
        assert_eq!(
            wire.encodes(CLAUDE_SONNET_4_6, Media::Document(&document)),
            carried,
            "{document:?}"
        );
        assert_eq!(
            user_wire(message::UserContent::Document(document.clone())).is_ok(),
            carried,
            "{document:?}"
        );
    }
    let audio = Audio {
        data: Source::base64("SUQz"),
        media_type: Some(AudioMediaType::MP3),
    };
    let video = Video {
        data: Source::url("https://example.com/a.mp4"),
        media_type: Some(VideoMediaType::MP4),
        additional_params: None,
    };
    assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Audio(&audio)));
    assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Video(&video)));
    assert!(user_wire(message::UserContent::Audio(audio)).is_err());
    assert!(user_wire(message::UserContent::Video(video)).is_err());
}

fn prepared_body(history: Vec<message::Message>) -> serde_json::Value {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let mut request = completion_request_with_history(history, None);
    request.tools = vec![generic_tool("add")];
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(
        &wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
    )
}

fn container_turn() -> message::AssistantMessage {
    let reply = json!({
        "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "container": {"id": "container_1", "expires_at": "2026-10-03T00:00:00Z"},
        "content": [
            {"type": "fallback", "model": CLAUDE_SONNET_4_6},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 1},
                "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_1"}}
        ]
    });
    let message::Message::Assistant(turn) = fold_reply(&reply)
        .expect("the reply folds")
        .message()
        .expect("a turn")
    else {
        panic!("an assistant turn");
    };
    turn
}

fn answered(turn: message::AssistantMessage) -> Vec<message::Message> {
    vec![
        message::Message::user("add"),
        message::Message::Assistant(turn),
        message::Message::tool_result(
            message::CallId::from_wire("toolu_1"),
            message::ToolName::new("add").expect("tool name"),
            "2",
        ),
    ]
}

#[test]
fn the_container_survives_a_dropped_block_and_an_edited_call() {
    let body = prepared_body(answered(container_turn()));
    assert_eq!(body["container"], json!("container_1"));

    let mut edited = container_turn();
    for block in &mut edited.content {
        if let message::AssistantContent::ToolCall(call) = block {
            call.function.arguments = json!({"x": 5}).as_object().cloned().unwrap_or_default();
        }
    }
    let body = prepared_body(answered(edited));
    assert_eq!(body["container"], json!("container_1"));
}

#[test]
fn an_edited_call_keeps_its_caller() {
    let mut turn = container_turn();
    for block in &mut turn.content {
        if let message::AssistantContent::ToolCall(call) = block {
            call.function.arguments = json!({"x": 5}).as_object().cloned().unwrap_or_default();
        }
    }
    let body = prepared_body(answered(turn));
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 5},
            "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_1"}})
    );
}

#[test]
fn a_paused_turn_resumes_with_its_running_server_tool_use() {
    let reply = json!({
        "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "pause_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {"type": "text", "text": "Searching."},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                "input": {"query": "rig"}}
        ]
    });
    let turn = fold_reply(&reply)
        .expect("the reply folds")
        .message()
        .expect("a turn");
    let body = prepared_body(vec![message::Message::user("search"), turn]);
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
            "input": {"query": "rig"}}),
        "{body:#}"
    );
}

#[test]
fn tool_results_lead_a_merged_user_message() {
    let call = message::AssistantContent::tool_call(
        "toolu_1",
        message::ToolName::new("add").expect("tool name"),
        json!({"x": 1}),
    );
    let body = prepared_body(vec![
        message::Message::user("add"),
        message::Message::Assistant(message::AssistantMessage::new(vec![call])),
        message::Message::user("also this"),
        message::Message::User {
            content: vec![
                message::UserContent::text("and this"),
                message::UserContent::tool_result(
                    message::CallId::from_wire("toolu_1"),
                    message::ToolName::new("add").expect("tool name"),
                    vec![message::ToolResultContent::text("2")],
                ),
            ],
        },
    ]);
    let content = &body["messages"][2]["content"];
    assert_eq!(content[0]["type"], json!("tool_result"), "{body:#}");
    assert_eq!(content[0]["tool_use_id"], json!("toolu_1"));
    assert_eq!(content[1]["text"], json!("also this"));
    assert_eq!(content[2]["text"], json!("and this"));
}

fn folded_turn(
    wire: &Messages,
    request: &CompletionRequest,
    reply: serde_json::Value,
) -> message::AssistantMessage {
    let mut reply = reply;
    if let Some(map) = reply.as_object_mut() {
        map.entry("type").or_insert_with(|| json!("message"));
        map.entry("id").or_insert_with(|| json!("msg_1"));
        map.entry("role").or_insert_with(|| json!("assistant"));
        map.entry("usage")
            .or_insert_with(|| json!({"input_tokens": 1, "output_tokens": 1}));
    }
    let response = crate::test_utils::decode_reply(
        wire,
        request,
        crate::wire::Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply,
    )
    .expect("the reply folds");
    let message::Message::Assistant(turn) = response.message().expect("a turn") else {
        panic!("an assistant turn: {:?}", response.choice);
    };
    turn
}

fn sent_with(
    wire: &Messages,
    history: Vec<message::Message>,
    tools: Vec<completion::ToolDefinition>,
) -> serde_json::Value {
    use crate::wire::{Operation, Wire};
    let mut request = completion_request_with_history(history, None);
    request.tools = tools;
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(
        &wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
    )
}

#[test]
fn an_idless_call_replays_under_the_id_its_result_gets() {
    for dialect in [&super::super::wire::ANTHROPIC, &super::super::wire::MINIMAX] {
        let wire = AnthropicConfig::with_key(dialect, "k").completion(CLAUDE_SONNET_4_6);
        for call in [
            json!({"type": "tool_use", "id": "", "name": "add", "input": {"x": 1}}),
            json!({"type": "tool_use", "name": "add", "input": {"x": 1}}),
        ] {
            if call.get("id").is_none() {
                let body = json!({"type":"message", "id":"msg_1", "role":"assistant",
                    "model":CLAUDE_SONNET_4_6, "stop_reason":"tool_use", "usage":{"input_tokens":1,"output_tokens":1}, "content":[call]});
                let event: Result<super::super::streaming::MessagesEvent, _> =
                    serde_json::from_value(body);
                assert!(
                    event.is_err(),
                    "the protected known tool-use id is mandatory"
                );
                continue;
            }
            let turn = folded_turn(
                &wire,
                &hello_request(),
                json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "tool_use", "content": [call]}),
            );
            let id = turn.tool_calls().next().expect("a call").id.clone();
            assert!(id.is_local(), "{id:?}");
            let body = sent_with(
                &wire,
                vec![
                    message::Message::user("add"),
                    message::Message::Assistant(turn),
                    message::Message::tool_result(
                        id,
                        message::ToolName::new("add").expect("name"),
                        "2",
                    ),
                ],
                vec![generic_tool("add")],
            );
            let use_id = &body["messages"][1]["content"][0]["id"];
            assert_eq!(
                use_id, &body["messages"][2]["content"][0]["tool_use_id"],
                "{body}"
            );
            assert_eq!(use_id, "tool-0", "{body}");
        }
    }
}

#[test]
fn a_binding_model_replays_its_thinking_and_asks_for_drop_block() {
    use crate::wire::{Operation, Wire};
    let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "sig_opus"});
    let drop_block = json!({"prefix_mismatch_behavior": "drop_block"});
    for (model, binds) in [(CLAUDE_OPUS_5_5, true), (CLAUDE_SONNET_4_6, false)] {
        let wire = AnthropicConfig::new("test-key").completion(model);
        let made = hello_request().tools(vec![generic_tool("add")]);
        let turn = folded_turn(
            &wire,
            &made,
            json!({"model": model, "stop_reason": "end_turn", "content": [
                thinking.clone(), {"type": "text", "text": "done"}]}),
        );
        let history = vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ];
        let changed = sent_with(
            &wire,
            history.clone(),
            vec![generic_tool("add"), generic_tool("mul")],
        );
        assert_eq!(
            changed["messages"][1]["content"][0], thinking,
            "{model}: {changed}"
        );
        let expected = binds.then(|| json!({"type": "adaptive", "block_binding": drop_block}));
        assert_eq!(
            changed.get("thinking"),
            expected.as_ref(),
            "{model}: {changed}"
        );

        let mut request = completion_request_with_history(history, None);
        request.tools = vec![generic_tool("add")];
        let request = crate::operation::Completion::prepare(request, &wire.describe())
            .expect("the request prepares");
        let encoded = wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes");
        let beta = encoded
            .request
            .headers()
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok());
        assert_eq!(
            beta,
            binds.then_some("thinking-binding-controls-2026-08-01")
        );
    }
}

#[test]
fn drop_block_joins_the_callers_thinking() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    for (thinking, sent) in [
        (
            json!({"type": "adaptive", "display": "summarized"}),
            json!({"type": "adaptive", "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"}}),
        ),
        (json!({"type": "disabled"}), json!({"type": "disabled"})),
    ] {
        let mut request = completion_request_with_history(vec![message::Message::user("q")], None);
        request.additional_params = Some(json!({ "thinking": thinking }));
        let body = super::body(&wire, &request, crate::wire::Mode::Unary).expect("encodes");
        assert_eq!(body.get("thinking"), Some(&sent));
    }
}

#[test]
fn unsigned_thinking_keeps_its_item_only_where_the_dialect_takes_it() {
    let unsigned = json!({"type": "thinking", "thinking": "plan", "signature": ""});
    for (dialect, model, kept) in [
        (&super::super::wire::ANTHROPIC, CLAUDE_SONNET_4_6, false),
        (&super::super::wire::MOONSHOT, "kimi-k2.6", true),
    ] {
        let wire = AnthropicConfig::with_key(dialect, "k").completion(model);
        let turn = folded_turn(
            &wire,
            &hello_request(),
            json!({"model": model, "stop_reason": "max_tokens", "content": [unsigned.clone()]}),
        );
        assert_eq!(turn.content[0].native_item().is_some(), kept, "{model}");
        let body = sent_with(
            &wire,
            vec![
                message::Message::user("go"),
                message::Message::Assistant(turn),
                message::Message::user("next"),
            ],
            Vec::new(),
        );
        let sent = &body["messages"][1]["content"][0];
        if kept {
            assert_eq!(sent, &unsigned);
        } else {
            assert_eq!(sent, &json!({"type": "text", "text": "plan"}));
        }
    }
}

#[test]
fn a_kept_tool_use_item_always_states_an_object_input() {
    let wire = AnthropicConfig::with_key(&super::super::wire::ZAI, "k").completion("glm-4.6");
    for (call, item) in [
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add"}),
            Some(json!({"type": "tool_use", "id": "call_1", "name": "add", "input": {}})),
        ),
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add", "input": null}),
            Some(json!({"type": "tool_use", "id": "call_1", "name": "add", "input": {}})),
        ),
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add", "input": "{\"x\":1}"}),
            None,
        ),
    ] {
        if call.get("input").is_none() {
            let event: Result<super::super::streaming::MessagesEvent, _> =
                serde_json::from_value(json!({
                "type":"content_block_start", "index":0, "content_block":call}));
            assert!(
                event.is_err(),
                "the protected known tool-use input is mandatory"
            );
            continue;
        }
        let turn = folded_turn(
            &wire,
            &hello_request(),
            json!({"model": "glm-4.6", "stop_reason": "tool_use", "content": [call]}),
        );
        assert_eq!(turn.content[0].native_item(), item.as_ref());
        let body = sent_with(
            &wire,
            vec![
                message::Message::user("add"),
                message::Message::Assistant(turn),
                message::Message::tool_result(
                    message::CallId::from_wire("call_1"),
                    message::ToolName::new("add").expect("name"),
                    "2",
                ),
            ],
            vec![generic_tool("add")],
        );
        assert!(
            body["messages"][1]["content"][0]["input"].is_object(),
            "{body}"
        );
    }
}

#[test]
fn a_blank_text_block_keeps_no_item_and_is_never_sent() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let turn = folded_turn(
        &wire,
        &hello_request(),
        json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "end_turn", "content": [
            {"type": "text", "text": "  "}, {"type": "text", "text": "real answer"}]}),
    );
    assert!(turn.content[0].native_item().is_none());
    let body = sent_with(
        &wire,
        vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        Vec::new(),
    );
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "real answer"}])
    );
}

#[test]
fn context_binding_reads_every_spelling_of_a_claude_model() {
    for model in [
        "claude-opus-5-5",
        "claude-opus-5-5-20260101",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-sonnet-5-5",
        "anthropic/claude-opus-5.5",
        "claude-opus-5.5",
        "anthropic.claude-opus-5-5-v1:0",
        "us.anthropic.claude-opus-5-5-20260101-v1:0",
        "us.anthropic.claude-opus-5",
    ] {
        assert!(binds_context(model), "{model}");
    }
    for model in [
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "gpt-5",
    ] {
        assert!(!binds_context(model), "{model}");
    }
}

#[test]
fn thinking_without_a_binding_replays_another_contexts_turn_as_text() {
    let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "sig"});
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_5_5);
    let made = hello_request().tools(vec![generic_tool("add")]);
    let turn = folded_turn(
        &wire,
        &made,
        json!({"model": CLAUDE_SONNET_5_5, "stop_reason": "end_turn", "content": [
            thinking.clone(), {"type": "text", "text": "done"}]}),
    );
    let history = vec![
        message::Message::user("go"),
        message::Message::Assistant(turn),
        message::Message::user("next"),
    ];
    let mut request = completion_request_with_history(history, None);
    request.tools = vec![generic_tool("add"), generic_tool("mul")];
    request.additional_params = Some(json!({"thinking": {"type": "between_tools"}}));
    let value = request_body(Params {
        model: CLAUDE_SONNET_5_5,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .expect("encodes");
    assert_eq!(
        value["thinking"],
        json!({"type": "between_tools"}),
        "{value}"
    );
    assert_eq!(
        value["messages"][1]["content"][0],
        json!({"type": "text", "text": "plan"}),
        "{value}"
    );
}

fn messages_violations(body: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    for (i, m) in messages.iter().enumerate() {
        let content = m["content"].as_array().cloned().unwrap_or_default();
        if content.is_empty() {
            out.push(format!("message {i} has empty content"));
        }
        for b in &content {
            if b["type"] == "text" && b["text"].as_str().is_some_and(|t| t.trim().is_empty()) {
                out.push(format!("message {i} has blank text"));
            }
        }
        // A run of system messages shares one slot, which Anthropic takes.
        if m["role"] == "system" {
            let prev = messages[..i]
                .iter()
                .rev()
                .find(|message| message["role"] != "system")
                .map(|message| message["role"].clone());
            let next = messages[i + 1..]
                .iter()
                .find(|message| message["role"] != "system")
                .map(|message| message["role"].clone());
            if prev != Some(json!("user")) || !(next.is_none() || next == Some(json!("assistant")))
            {
                out.push(format!("system at {i} between {prev:?} and {next:?}"));
            }
        }
        if m["role"] == "assistant" {
            let ids: Vec<String> = content
                .iter()
                .filter(|b| b["type"] == "tool_use")
                .filter_map(|b| b["id"].as_str().map(str::to_owned))
                .collect();
            if ids.is_empty() {
                continue;
            }
            let Some(next) = messages.get(i + 1) else {
                out.push(format!("tool_use at {i} ends the request"));
                continue;
            };
            let next_content = next["content"].as_array().cloned().unwrap_or_default();
            let leading: Vec<String> = next_content
                .iter()
                .take_while(|b| b["type"] == "tool_result")
                .filter_map(|b| b["tool_use_id"].as_str().map(str::to_owned))
                .collect();
            if next["role"] != "user" || ids.iter().any(|id| !leading.contains(id)) {
                out.push(format!(
                    "tool_use at {i} not answered first by next message"
                ));
            }
        }
    }
    out
}

#[test]
fn adversarial_histories_encode_to_requests_anthropic_takes() {
    use crate::message::{
        AssistantMessage, CallId, Origin, StopReason, ToolCall, ToolFunction, ToolName,
    };
    let other = Some(Origin::new("openai.chat", "openai", "gpt-4.1"));
    let asst = |content: Vec<AssistantContent>, stop: StopReason| {
        Message::Assistant(
            AssistantMessage::new(content)
                .with_origin(other.clone().expect("origin"))
                .with_stop(stop),
        )
    };
    let call = |id: &str| {
        ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
        )
    };
    let result = |id: &str| Message::User {
        content: vec![UserContent::ToolResult(
            call(id).result(vec![ToolResultContent::text("r")]),
        )],
    };
    let text = |t: &str| AssistantContent::text(t);
    let tc = |id: &str| AssistantContent::ToolCall(call(id));
    let cases: Vec<(&str, Vec<Message>)> = vec![
        (
            "system between assistants",
            vec![
                Message::user("a"),
                asst(vec![text("one")], StopReason::Stop),
                Message::system("mid"),
                asst(vec![text("two")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "system while a call waits, then text",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::ToolUse),
                Message::system("mid"),
                Message::User {
                    content: vec![
                        UserContent::text("note"),
                        UserContent::ToolResult(
                            call("x").result(vec![ToolResultContent::text("r")]),
                        ),
                    ],
                },
                Message::user("q"),
            ],
        ),
        (
            "blank user after a call",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::ToolUse),
                Message::user("  "),
                result("x"),
            ],
        ),
        (
            "two systems around a user",
            vec![
                Message::system("s0"),
                Message::user("a"),
                Message::system("s1"),
                Message::user("b"),
                Message::system("s2"),
                asst(vec![text("ok")], StopReason::Stop),
                Message::user("end"),
            ],
        ),
        (
            "system before results of an aborted turn",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::Aborted("cut".into())),
                Message::system("mid"),
                result("x"),
                Message::user("q"),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for model in [CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6] {
        for (label, history) in &cases {
            let mut request = CompletionRequest::from(history.clone()).max_tokens(64);
            request.tools = vec![completion::ToolDefinition {
                name: ToolName::new("lookup").expect("name"),
                description: "d".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }];
            let body = match request_body(Params {
                model,
                request,
                prompt_caching: false,
                cache: None,
                static_prefix_cache_ttl: None,
            }) {
                Ok(body) => body,
                Err(error) => {
                    failures.push(format!("{model} {label}: {error}"));
                    continue;
                }
            };
            let v = messages_violations(&body);
            if !v.is_empty() {
                failures.push(format!("{model} {label}: {v:?}\n  {}", body["messages"]));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn tool_input_streaming_sent(
    wire: &crate::providers::anthropic::wire::Messages,
    mode: Mode,
    tools: Vec<completion::ToolDefinition>,
) -> (Vec<Option<Value>>, Option<String>) {
    use crate::wire::Wire;
    let request = completion_request_with_tools(
        tools,
        Some(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})),
    );
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let body = json_body(&encoded.request);
    let eager = body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| tool.get("eager_input_streaming").cloned())
                .collect()
        })
        .unwrap_or_default();
    let beta = encoded
        .request
        .headers()
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    (eager, beta)
}

#[test]
fn a_request_with_tools_asks_for_eager_tool_input_on_every_dialect() {
    use crate::providers::anthropic::wire::{ANTHROPIC, MINIMAX, MOONSHOT, XIAOMIMIMO, ZAI};
    for dialect in [ANTHROPIC, ZAI, MINIMAX, MOONSHOT, XIAOMIMIMO] {
        let wire = AnthropicConfig::with_key(&dialect, "k").completion("some-model");
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Streaming, vec![generic_tool("lookup")]),
            (vec![Some(json!(true)), None], None),
            "{}",
            dialect.name
        );
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Unary, vec![generic_tool("lookup")]),
            (vec![Some(json!(true)), None], None),
            "{}",
            dialect.name
        );
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Streaming, Vec::new()),
            (vec![None], None),
            "{}",
            dialect.name
        );
    }
}

#[test]
fn tool_input_streaming_falls_back_to_the_beta_flag_or_off() {
    use crate::providers::anthropic::wire::ToolInputStreaming;
    let provider = AnthropicConfig::new("k").with_beta("files-api-2025-04-14");
    let beta = provider
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::BetaHeader);
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Streaming, vec![generic_tool("lookup")]),
        (
            vec![None, None],
            Some("files-api-2025-04-14,fine-grained-tool-streaming-2025-05-14".to_owned())
        )
    );
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Unary, vec![generic_tool("lookup")]),
        (
            vec![None, None],
            Some("files-api-2025-04-14,fine-grained-tool-streaming-2025-05-14".to_owned())
        )
    );
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Streaming, Vec::new()),
        (vec![None], Some("files-api-2025-04-14".to_owned()))
    );
    let off = provider
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::Off);
    assert_eq!(
        tool_input_streaming_sent(&off, Mode::Streaming, vec![generic_tool("lookup")]),
        (vec![None, None], Some("files-api-2025-04-14".to_owned()))
    );
}

#[test]
fn a_wire_serialized_without_tool_input_streaming_reloads_eager() {
    use crate::providers::anthropic::wire::{Messages, ToolInputStreaming};
    let wire = AnthropicConfig::new("k")
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::Off);
    let mut json = serde_json::to_value(&wire).expect("the wire serializes");
    assert_eq!(json["tool_input_streaming"], json!("off"));
    json.as_object_mut()
        .map(|wire| wire.shift_remove("tool_input_streaming"));
    let restored: Messages = serde_json::from_value(json).expect("the wire reloads");
    assert_eq!(restored.tool_input_streaming, ToolInputStreaming::Eager);
}

#[test]
fn tool_input_streaming_reads_its_names_and_sends_the_beta_flag_once() {
    use crate::providers::anthropic::wire::ToolInputStreaming;
    for (name, streaming) in [
        ("eager", ToolInputStreaming::Eager),
        ("beta_header", ToolInputStreaming::BetaHeader),
        ("off", ToolInputStreaming::Off),
    ] {
        assert_eq!(
            serde_json::from_value::<ToolInputStreaming>(json!(name)).ok(),
            Some(streaming)
        );
        assert_eq!(serde_json::to_value(streaming).ok(), Some(json!(name)));
        assert!(!format!("{streaming:?}").is_empty());
    }
    for unknown in [json!("fine_grained"), json!(1)] {
        assert!(serde_json::from_value::<ToolInputStreaming>(unknown).is_err());
    }
    let wire = AnthropicConfig::new("k")
        .with_beta("fine-grained-tool-streaming-2025-05-14")
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::BetaHeader);
    assert_eq!(
        tool_input_streaming_sent(&wire, Mode::Streaming, vec![generic_tool("lookup")]).1,
        Some("fine-grained-tool-streaming-2025-05-14".to_owned())
    );
}

#[test]
fn an_invalid_raw_top_level_cache_marker_is_refused() {
    for marker in [
        json!({"type": "persistent"}),
        json!({"type": "ephemeral", "ttl": "1d"}),
    ] {
        let request = completion_request_with_tools(
            vec![generic_tool("cached_tool")],
            Some(json!({ "cache_control": marker })),
        );
        let error = request_body(Params {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            cache: None,
            static_prefix_cache_ttl: None,
        })
        .expect_err("refused");
        assert!(
            error
                .to_string()
                .contains("additional_params.cache_control"),
            "{error}"
        );
    }
}

#[test]
fn a_raw_top_level_marker_beats_the_cache_option() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({"cache_control":{"type":"ephemeral", "ttl":"5m"}})),
    );
    let error = request_body(Params {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .expect_err("protected TTL conflict");
    assert!(
        error
            .to_string()
            .contains("conflicts with the typed automatic caching TTL")
    );
}

#[test]
fn a_text_family_document_is_sent_as_its_text() {
    use message::{DocumentMediaType as Doc, DocumentSourceKind as Source};

    for (data, media_type, text) in [
        (
            Source::base64("cmlnLG1hdHJpeAoxLDIK").into(),
            Some(Doc::CSV),
            "rig,matrix\n1,2\n",
        ),
        (
            Source::base64("cGxhaW4gdGV4dA==").into(),
            Some(Doc::TXT),
            "plain text",
        ),
        (
            message::DocumentData::Text("# notes".into()),
            Some(Doc::MARKDOWN),
            "# notes",
        ),
        (
            message::DocumentData::Text("untyped".into()),
            None,
            "untyped",
        ),
    ] {
        if media_type
            .as_ref()
            .is_some_and(|kind| !matches!(kind, Doc::TXT | Doc::PDF))
        {
            let error = user_wire(message::UserContent::Document(message::Document {
                data,
                media_type,
                additional_params: None,
            }))
            .expect_err("declared non-PDF/non-TXT media is refused");
            assert!(
                error
                    .to_string()
                    .contains("Anthropic only supports PDF and plain text documents")
            );
            continue;
        }
        let converted = user_wire(message::UserContent::Document(message::Document {
            data,
            media_type,
            additional_params: None,
        }))
        .expect("the document converts")
        .expect("a block");
        let value = serde_json::to_value(&converted).expect("the message serializes");
        assert_eq!(
            value["content"][0]["source"],
            json!({"type": "text", "media_type": "text/plain", "data": text})
        );
    }
}

#[test]
fn a_server_tool_result_never_replays_without_its_use() {
    let wire = AnthropicConfig::with_key(&super::super::wire::ZAI, "test-key")
        .completion(CLAUDE_SONNET_4_6);
    let result = json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "url": "https://x", "title": "x",
            "encrypted_content": "e"}]});
    let mut turn = folded_turn(
        &wire,
        &hello_request(),
        json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "end_turn", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                "input": {"query": "rig"}},
            result.clone(),
            {"type": "text", "text": "done"}]}),
    );
    turn.content.remove(0);
    let body = sent_with(
        &wire,
        vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        Vec::new(),
    );
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "done"}]),
        "{body}"
    );
}

#[test]
fn opus_4_8_preserves_system_message_after_a_code_execution_result_in_a_container() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_OPUS_4_8, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "container": {"id": "container_1", "expires_at": "2026-10-03T00:00:00Z"},
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "code_execution",
                "input": {"code": "print(1)"}},
            {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_1",
                "content": {"type": "code_execution_result", "stdout": "1\n", "stderr": "",
                    "return_code": 0, "content": []}}
        ]
    });
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_4_8);
    let turn = crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        crate::wire::Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds")
    .message()
    .expect("a turn");
    let request = completion_request_with_history(
        vec![
            message::Message::user("run it"),
            turn,
            message::Message::System {
                content: "Answer in Spanish.".to_string(),
            },
        ],
        None,
    );
    let value = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    assert_eq!(value["system"][0]["text"], "Answer in Spanish.");
    assert_eq!(value["messages"].as_array().unwrap().len(), 2);
    assert_eq!(value["messages"][1]["role"], "assistant");
    assert_eq!(value["container"], json!("container_1"));
}

#[test]
fn sonnet_5_5_defers_a_system_message_past_tool_results() {
    let lookup = || message::ToolName::new("lookup").expect("tool name");
    let tool_call = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::tool_call("toolu_1", lookup(), json!({})),
    ]));
    let tool_result =
        message::Message::tool_result(message::CallId::from_wire("toolu_1"), lookup(), "ok");
    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            tool_call,
            message::Message::system("Answer in Spanish."),
            tool_result,
        ],
    );
    // G153: the held system now occupies the valid slot after the tool results.
    assert!(value.get("system").is_none());
    assert_eq!(value["messages"][3], json!({
        "role":"system", "content":[{"type":"text","text":"Answer in Spanish."}]
    }));
    let roles: Vec<_> = value["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].clone())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "system"]);

    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            message::Message::assistant("Done."),
            message::Message::system("Answer in Spanish."),
        ],
    );
    assert_eq!(value["system"][0]["text"], "Answer in Spanish.");
}

fn encode_mid_conversation_history(
    model: &str,
    history: Vec<message::Message>,
) -> serde_json::Value {
    let request = request_body(Params {
        model,
        request: completion_request_with_history(history, None),
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    serde_json::to_value(request).unwrap()
}

#[test]
fn carried_hosted_results_replay_without_an_origin_or_a_matching_use() {
    let raw = json!({"type":"web_search_tool_result", "tool_use_id":"srvtoolu_01",
        "content":{"type":"web_search_tool_result_error", "error_code":"max_uses_exceeded"}});
    let mut turn = match decoded_assistant(CLAUDE_OPUS_4_8, vec![raw.clone()]) {
        message::Message::Assistant(turn) => turn,
        _ => panic!("assistant turn"),
    };
    turn.origin = None;
    let body = prepared_body(vec![
        message::Message::user("search"),
        message::Message::Assistant(turn),
        message::Message::user("next"),
    ]);
    assert_eq!(body["messages"][1]["content"], json!([raw]));
}

// G152: a separate nonempty result control retains every original alias/raw predicate.
#[test]
fn nonempty_results_preserve_raw_server_tool_handle_reservations() {
    for hosted_id in ["srvtoolu_real", "tool-0"] {
        let raw_call = json!({"type":"server_tool_use","id":hosted_id,"name":"web_search","input":{"query":"test"}});
        let raw_result =
            json!({"type":"web_search_tool_result","tool_use_id":hosted_id,"content":[]});
        let local = message::ToolCall::new(
            message::CallId::Local(message::LocalCallId::new()),
            message::ToolFunction::new(
                message::ToolName::new("local").expect("tool name"),
                json!({}),
            ),
        );
        assert!(local.id.is_local());
        let message::Message::Assistant(mut turn) = decoded_assistant(
            CLAUDE_SONNET_4_6,
            vec![raw_call.clone(), raw_result.clone()],
        ) else {
            panic!("assistant");
        };
        turn.content
            .insert(0, message::AssistantContent::ToolCall(local.clone()));
        let history = vec![
            message::Message::Assistant(turn),
            message::Message::User {
                content: vec![message::UserContent::ToolResult(local.result(vec![message::ToolResultContent::text("ok")]))],
            },
        ];
        let mut request = completion_request_with_history(history, None);
        request.tools = vec![generic_tool("local")];
        let value = request_body(Params {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            cache: None,
            static_prefix_cache_ttl: None,
        })
        .expect("reserved raw handles");
        let content = &value["messages"][0]["content"];
        assert_ne!(content[0]["id"], hosted_id);
        assert_eq!(
            content[0]["id"],
            value["messages"][1]["content"][0]["tool_use_id"]
        );
        assert_eq!(content[1], raw_call);
        assert_eq!(content[2], raw_result);
        if hosted_id == "tool-0" {
            assert_eq!(content[0]["id"], "tool-1");
        }
    }
}

// G140: foreign received signature data is never outbound Anthropic authority.
#[test]
fn foreign_signed_answer_projects_only_its_known_text() {
    let source = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let response = crate::test_utils::decode_reply(&source, &CompletionRequest::new("restate"),
        Mode::Unary, [WireFrame::Text(json!({"candidates":[{
            "content":{"parts":[{"text":"the answer","thoughtSignature":"c2lnbmVkLWFuc3dlcg=="}],"role":"model"},
            "finishReason":"STOP"
        }]}).to_string())], Value::Null).expect("genuine signed answer");
    let request = CompletionRequest::from(vec![message::Message::user("q"),
        message::Message::Assistant(message::AssistantMessage::new(response.choice)),
        message::Message::user("again")]).max_tokens(16);
    let target = AnthropicConfig::new("sk-test").completion("claude-haiku-4-5");
    let encoded = target.encode(request.clone(), Mode::Unary).expect("direct known text");
    let direct = json_body(&encoded.request);
    let prepared = Completion::prepare(request, &target.describe()).expect("original carried preflight");
    let encoded = target.encode(prepared, Mode::Unary).expect("prepared known text");
    for body in [direct, json_body(&encoded.request)] {
        assert_eq!(body["messages"][1]["content"], json!([{"type":"text","text":"the answer"}]));
        assert!(!body.to_string().contains("thoughtSignature"));
        assert!(!body.to_string().contains("c2lnbmVkLWFuc3dlcg"));
    }
}

#[test]
fn foreign_inspection_text_edits_do_not_replay_payload_siblings() {
    const BODY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"answer","executableCode":{"language":"RUST","code":"x"},"arbitrary":{"retained":true}}]},"finishReason":"STOP"}]}"#;
    let source = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let raw: Value = serde_json::from_str(BODY).expect("exact original fixture");
    for mode in [Mode::Unary, Mode::Streaming] {
        let response = crate::test_utils::history::decode(&source, mode,
            [WireFrame::Text(BODY.to_owned())]).expect("known inspection text");
        assert_eq!(response.raw, raw);
        let received = response.message().expect("genuine turn");
        for edited in [false, true] {
            let mut received = received.clone();
            let message::Message::Assistant(turn) = &mut received else { panic!("assistant"); };
            let message::AssistantContent::CarriedText(part) = &mut turn.content[0] else { panic!("carried"); };
            if edited { *part.text_mut() = "edited".to_owned(); }
            assert_eq!(part.payload.item, raw["candidates"][0]["content"]["parts"][0]);
            let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
            let request = CompletionRequest::from(vec![message::Message::user("before"),
                received, message::Message::user("continue")]).max_tokens(64);
            let prepared = Completion::prepare(request, &target.describe()).expect("original known text prepares");
            let encoded = target.encode(prepared, Mode::Unary).expect("known text encodes");
            let body = json_body(&encoded.request);
            assert_eq!(body["messages"][1]["content"], json!([{
                "type":"text","text":if edited { "edited" } else { "answer" }
            }]));
            assert!(!body.to_string().contains("executableCode"));
            assert!(!body.to_string().contains("arbitrary"));
        }
    }
}

#[test]
fn foreign_opaque_carried_content_still_refuses_before_lowering() {
    use crate::completion::ReplayTarget;
    let part = message::CarriedText::new(message::CarriedTextContent::Opaque(String::new()),
        message::CarriedPayload::new(json!({"unknown":"opaque"}), Some(crate::providers::gemini::completion::PROVIDER_NAME.to_owned())));
    let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    assert!(target.check_carried_text(&part).is_err());
    assert!(assistant_parts(&message::AssistantContent::CarriedText(part), &target, None).is_err());
}


/// A real signed Gemini thought and a real Responses ciphertext item.
fn g226_received_carried_reasoning() -> Vec<(message::Message, &'static str, Value)> {
    let signed_part = json!({"text":"signed thought","thought":true,"thoughtSignature":"c2ln"});
    let signed_raw = json!({"candidates":[{
        "content":{"role":"model","parts":[signed_part]},"finishReason":"STOP",
    }]});
    let gemini = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let signed = crate::test_utils::history::decode(&gemini, Mode::Unary,
        [WireFrame::Text(signed_raw.to_string())]).expect("real signed thought decoder");
    assert_eq!(signed.raw, signed_raw);
    let [message::AssistantContent::CarriedReasoning(reasoning)] = signed.choice.as_slice()
        else { panic!("one received carried signed thought"); };
    assert_eq!(reasoning.payload.item, signed_part);
    assert_eq!(reasoning.provider(), Some(crate::providers::gemini::completion::PROVIDER_NAME));
    assert_eq!(reasoning.content, vec![message::CarriedReasoningPart::Text {
        text:"signed thought".to_owned(), signature:Some("c2ln".to_owned()),
    }]);

    let cipher_item = json!({"type":"reasoning","id":"rs_g226","summary":[],
        "encrypted_content":"ciphertext","status":"completed"});
    let cipher_raw = json!({"id":"resp_g226","object":"response","created_at":0,
        "status":"completed","model":"gpt-5.4","output":[cipher_item],"tools":[]});
    let responses = crate::providers::openai::OpenAIConfig::new("test-key").responses("gpt-5.4");
    let cipher = crate::test_utils::history::decode(&responses, Mode::Unary,
        [WireFrame::Text(cipher_raw.to_string())]).expect("real ciphertext decoder");
    assert_eq!(cipher.raw, cipher_raw);
    let [message::AssistantContent::CarriedReasoning(reasoning)] = cipher.choice.as_slice()
        else { panic!("one received carried ciphertext"); };
    assert_eq!(reasoning.payload.item, cipher_item);
    assert_eq!(reasoning.provider(), Some("openai"));
    assert_eq!(reasoning.id.as_deref(), Some("rs_g226"));
    assert_eq!(reasoning.content, vec![message::CarriedReasoningPart::Encrypted("ciphertext".to_owned())]);
    vec![
        (signed.message().expect("genuine signed turn"), crate::providers::gemini::completion::PROVIDER_NAME,
            json!([{"type":"thinking","thinking":"signed thought","signature":"c2ln"}])),
        (cipher.message().expect("genuine ciphertext turn"), "openai",
            json!([{"type":"redacted_thinking","data":"ciphertext"}])),
    ]
}

fn g226_messages_target(issuer: &'static str) -> Messages {
    let dialect = super::super::wire::compatible(
        issuer, "https://messages.invalid", "G226_TEST_KEY", None,
    );
    AnthropicConfig::with_key(&dialect, "test-key").completion("fixture-model")
}

fn g226_request(turn: message::Message) -> CompletionRequest {
    CompletionRequest::from(vec![message::Message::user("before"), turn,
        message::Message::user("again")]).max_tokens(64)
}

fn g226_encode(target: &Messages, request: CompletionRequest, prepared: bool) -> Value {
    let request = if prepared {
        Completion::prepare(request, &target.describe()).expect("real carried preflight and adaptation")
    } else { request };
    json_body(&target.encode(request, Mode::Unary).expect("real Messages encoding").request)
}

#[test]
fn g226_known_received_reasoning_is_scoped_to_the_actual_messages_dialect() {
    for (received, issuer, parts) in g226_received_carried_reasoning() {
        let before = received.clone();
        let matching = g226_messages_target(issuer);
        let foreign = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
        for prepared in [false, true] {
            let accepted = g226_encode(&matching, g226_request(received.clone()), prepared);
            assert_eq!(accepted["messages"], json!([
                {"role":"user","content":[{"type":"text","text":"before"}]},
                {"role":"assistant","content":parts},
                {"role":"user","content":[{"type":"text","text":"again"}]},
            ]));
            let omitted = g226_encode(&foreign, g226_request(received.clone()), prepared);
            assert_eq!(omitted["messages"], json!([
                {"role":"user","content":[{"type":"text","text":"before"}]},
                {"role":"user","content":[{"type":"text","text":"again"}]},
            ]));
        }
        assert_eq!(received, before, "encoding never edits received evidence");
    }
}

#[test]
fn g226_unknown_issuer_keeps_signed_and_ciphertext_parts() {
    let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    for (received, _, parts) in g226_received_carried_reasoning() {
        let message::Message::Assistant(mut turn) = received else { panic!("assistant"); };
        let message::AssistantContent::CarriedReasoning(reasoning) = &mut turn.content[0]
            else { panic!("carried"); };
        let raw = reasoning.payload.item.clone();
        // This is the explicit unknown-provenance input, not a fabricated issuer.
        reasoning.payload.provider = None;
        let unknown = message::Message::Assistant(turn);
        let before = unknown.clone();
        for prepared in [false, true] {
            let body = g226_encode(&target, g226_request(unknown.clone()), prepared);
            assert_eq!(body["messages"][1]["content"], parts);
        }
        let message::Message::Assistant(turn) = &unknown else { panic!("assistant"); };
        let message::AssistantContent::CarriedReasoning(reasoning) = &turn.content[0]
            else { panic!("carried"); };
        assert_eq!(reasoning.payload.item, raw);
        assert_eq!(unknown, before);
    }
}

#[test]
fn g226_foreign_reasoning_keeps_other_blocks_and_original_tool_coordinates() {
    let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    for (received, _, _) in g226_received_carried_reasoning() {
        let message::Message::Assistant(mut turn) = received else { panic!("assistant"); };
        let id: message::CallId = message::LocalCallId::new().into();
        let name = message::ToolName::new("lookup").expect("tool name");
        turn.content.push(message::AssistantContent::ToolCall(message::ToolCall::new(
            id.clone(), message::ToolFunction::new(name.clone(), json!({"q":"x"})),
        )));
        turn.content.push(message::AssistantContent::text("kept"));
        let request = CompletionRequest::from(vec![
            message::Message::user("before"), message::Message::Assistant(turn),
            message::Message::User {content:vec![message::UserContent::tool_result(
                id, name, vec![message::ToolResultContent::text("done")],
            )]},
        ]).tools(vec![crate::completion::ToolDefinition::new(
            message::ToolName::new("lookup").expect("declared tool name"),
            "lookup",
            json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}),
        )]).max_tokens(64);
        for prepared in [false, true] {
            let request = if prepared {
                Completion::prepare(request.clone(), &target.describe()).expect("prepared mixed turn")
            } else { request.clone() };
            let ids = WireIds::for_target(&request.chat_history, &target, CLAUDE_SONNET_4_6)
                .expect("original coordinate plan");
            let planned = ids.of(1, 1).expect("call remains at original content index one").to_owned();
            let body = g226_encode(&target, request, false);
            assert_eq!(body["messages"][1]["content"], json!([
                {"type":"tool_use","id":planned,"name":"lookup","input":{"q":"x"}},
                {"type":"text","text":"kept"},
            ]));
            assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], planned);
        }
    }
}

#[test]
fn g226_ordinary_reasoning_and_genuine_native_anthropic_parts_stay_unchanged() {
    let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let ordinary = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::Reasoning(message::Reasoning::new("ordinary")),
    ]));
    for prepared in [false, true] {
        let body = g226_encode(&target, g226_request(ordinary.clone()), prepared);
        assert_eq!(body["messages"][1]["content"], json!([{"type":"text","text":"ordinary"}]));
    }
    let content = json!([
        {"type":"thinking","thinking":"native thought","signature":"native-signature"},
        {"type":"redacted_thinking","data":"native-ciphertext"},
    ]);
    let raw = json!({"type":"message","id":"msg_g226","role":"assistant",
        "model":CLAUDE_SONNET_4_6,"content":content,"stop_reason":"end_turn",
        "stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}});
    let response = crate::test_utils::history::decode(&target, Mode::Unary,
        [WireFrame::Text(raw.to_string())]).expect("real Anthropic decoder");
    assert_eq!(response.raw, raw);
    assert!(response.choice.iter().all(|part| matches!(part, message::AssistantContent::Reasoning(_))));
    assert_eq!(response.choice[0].native_item(), Some(&content[0]));
    assert_eq!(response.choice[1].native_item(), Some(&content[1]));
    let received = response.message().expect("genuine native turn");
    let before = received.clone();
    for prepared in [false, true] {
        let body = g226_encode(&target, g226_request(received.clone()), prepared);
        assert_eq!(body["messages"][1]["content"], content);
    }
    assert_eq!(received, before);
}

#[test]
fn g226_opaque_reasoning_is_refused_before_foreign_omission() {
    let target = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let raw = json!({"type":"future_content","payload":[1,2]});
    for issuer in [None, Some("anthropic"), Some("foreign/Responses")] {
        for summary in [false, true] {
            let payload = message::CarriedPayload::new(raw.clone(), issuer.map(str::to_owned));
            let part = if summary {
                message::CarriedReasoningPart::OpaqueSummary(payload)
            } else { message::CarriedReasoningPart::OpaqueContent(payload) };
            let reasoning = message::CarriedReasoning::new(vec![part],
                message::CarriedPayload::new(json!({"type":"reasoning","content":[raw]}),
                    issuer.map(str::to_owned)));
            let request = g226_request(message::Message::Assistant(message::AssistantMessage::new(
                vec![message::AssistantContent::CarriedReasoning(reasoning)],
            )));
            let prepared = Completion::prepare(request.clone(), &target.describe())
                .expect_err("opaque preflight still refuses");
            let direct: ProviderError = target.encode(request, Mode::Unary)
                .expect_err("direct opaque lowering still refuses").into();
            for error in [prepared, direct] {
                let ProviderError::Request(source) = error else {panic!("typed Request");};
                let source = source.downcast_ref::<message::UnrepresentableOpaqueContent>()
                    .expect("existing typed opaque source");
                assert_eq!(source.wire, "Anthropic Messages");
            }
        }
    }
}
// HC001 synthetic received variants use the actual Messages decoder and fold.
const HC001_REFUSAL: &str = "Cannot continue Anthropic Messages: an aborted turn's unresolved server call `srv_hc001` must be resolved or removed from history before continuing";

fn hc001_wire() -> Messages {
    AnthropicConfig::new("synthetic-test-key").completion(CLAUDE_SONNET_4_6)
}

fn hc001_use(id: &str) -> Value {
    json!({"type":"server_tool_use", "id":id, "name":"web_search", "input":{"query":"synthetic query"}})
}

fn hc001_result(id: &str) -> Value {
    json!({"type":"web_search_tool_result", "tool_use_id":id,
        "content":[{"type":"web_search_result", "url":"https://example.com/hc001",
            "title":"Synthetic received result", "encrypted_content":" exact encrypted result\n "}]})
}

fn hc001_prefix(
    wire: &Messages,
    blocks: &[Value],
    close_last: bool,
) -> completion::CompletionResponse {
    let mut frames = vec![json!({"type":"message_start", "message":{
        "type":"message", "id":"msg_hc001", "role":"assistant", "model":wire.model,
        "content":[], "stop_reason":null, "stop_sequence":null,
        "usage":{"input_tokens":3,"output_tokens":0}}})];
    for (index, block) in blocks.iter().enumerate() {
        frames.push(json!({"type":"content_block_start", "index":index, "content_block":block}));
        if close_last || index + 1 < blocks.len() {
            frames.push(json!({"type":"content_block_stop", "index":index}));
        }
    }
    let (response, ended) = crate::test_utils::history_conformance::partial(
        wire,
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
    );
    assert!(!ended, "the synthetic prefix has no provider end");
    assert!(matches!(response.stop(), message::StopReason::Aborted(_)));
    assert_eq!(response.raw["content"], json!(blocks));
    response
}

fn hc001_complete(
    wire: &Messages,
    blocks: Vec<Value>,
    reason: &str,
) -> completion::CompletionResponse {
    let raw = json!({"type":"message", "id":"msg_hc001_complete", "role":"assistant",
        "model":wire.model, "content":blocks, "stop_reason":reason, "stop_sequence":null,
        "usage":{"input_tokens":3,"output_tokens":5}});
    crate::test_utils::decode_reply(
        wire,
        &CompletionRequest::new("synthetic fixture"),
        Mode::Unary,
        [WireFrame::Text(raw.to_string())],
        raw,
    )
    .expect("actual Messages decoder and fold")
}

fn hc001_request(
    source: &completion::CompletionResponse,
    next: Option<message::Message>,
) -> CompletionRequest {
    let mut history = vec![message::Message::user("q")];
    if let Some(turn) = source.message() {
        history.push(turn);
    }
    history.extend(next);
    CompletionRequest::from(history).max_tokens(64)
}

// Tuple return permits custody assertions without a Result-returning assertion helper.
fn hc001_encode(
    wire: &Messages,
    request: &CompletionRequest,
    sources: &[&completion::CompletionResponse],
    prepared: bool,
) -> (Result<Value, ProviderError>, ()) {
    let history = request.chat_history.clone();
    let history_serde = serde_json::to_vec(&history).expect("complete history serde");
    let raw_options = request.additional_params.clone();
    let before: Vec<_> = sources
        .iter()
        .map(|source| {
            (
                (**source).clone(),
                serde_json::to_vec(*source).expect("complete received response serde"),
            )
        })
        .collect();
    let result = if prepared {
        Completion::prepare(request.clone(), &wire.describe())
    } else {
        Ok(request.clone())
    }
    .and_then(|request| {
        wire.encode(request, Mode::Unary)
            .map(|encoded| json_body(&encoded.request))
            .map_err(ProviderError::from)
    });
    assert_eq!(request.chat_history, history);
    assert_eq!(
        serde_json::to_vec(&request.chat_history).expect("history after"),
        history_serde
    );
    assert_eq!(request.additional_params, raw_options);
    for (source, (saved, serde)) in sources.iter().zip(before) {
        assert_eq!(
            **source, saved,
            "whole genuine received response/raw/choice stays exact"
        );
        assert_eq!(serde_json::to_vec(*source).expect("source after"), serde);
    }
    (result, ())
}

fn hc001_refused(result: Result<Value, ProviderError>) {
    let error = result.expect_err("unresolved aborted hosted occurrence refuses");
    let ProviderError::Request(source) = error else {
        panic!("native Request refusal");
    };
    assert_eq!(source.to_string(), HC001_REFUSAL);
}

fn hc001_body(messages: Vec<Value>) -> Value {
    json!({"model":CLAUDE_SONNET_4_6, "max_tokens":64, "messages":messages})
}

fn hc001_user(text: &str) -> Value {
    json!({"role":"user", "content":[{"type":"text","text":text}]})
}
fn hc001_assistant(parts: Vec<Value>) -> Value {
    json!({"role":"assistant", "content":parts})
}

#[test]
fn hc001_closed_aborted_use_refuses_native_encode_and_prepared_model_without_transport() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    let request = hc001_request(&source, Some(message::Message::user("next")));
    for prepared in [false, true] {
        hc001_refused(hc001_encode(&wire, &request, &[&source], prepared).0);
    }
    let before = source.clone();
    let history = request.chat_history.clone();
    let client = crate::test_utils::RecordingHttpClient::new("no transport reply may be read");
    let model = crate::driver::Model::new(wire, client.clone());
    let error = match model.prepare_exchange(request.clone(), Mode::Unary) {
        Err(error) => error,
        Ok(_) => panic!("actual Model preparation must refuse before opening transport"),
    };
    let ProviderError::Request(cause) = error else {
        panic!("Model preserves the Request refusal");
    };
    assert_eq!(cause.to_string(), HC001_REFUSAL);
    assert!(client.requests().is_empty());
    assert_eq!(request.chat_history, history);
    assert_eq!(source, before);
}

#[test]
fn hc001_same_and_later_assistant_results_admit_exact_received_bodies() {
    let wire = hc001_wire();
    let used = hc001_use("srv_hc001");
    let result = hc001_result("srv_hc001");
    let same = hc001_prefix(&wire, &[used.clone(), result.clone()], true);
    let request = hc001_request(&same, Some(message::Message::user("next")));
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(&wire, &request, &[&same], prepared)
                .0
                .expect("same-response pairing"),
            hc001_body(vec![
                hc001_user("q"),
                hc001_assistant(vec![used.clone(), result.clone()]),
                hc001_user("next")
            ])
        );
    }
    let first = hc001_prefix(&wire, &[used.clone()], true);
    let later = hc001_complete(&wire, vec![result.clone()], "end_turn");
    let request = CompletionRequest::from(vec![
        message::Message::user("q"),
        first.message().expect("first received turn"),
        later.message().expect("later received turn"),
        message::Message::user("next"),
    ])
    .max_tokens(64);
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(&wire, &request, &[&first, &later], prepared)
                .0
                .expect("later-response pairing"),
            hc001_body(vec![
                hc001_user("q"),
                hc001_assistant(vec![used.clone()]),
                hc001_assistant(vec![result.clone()]),
                hc001_user("next")
            ])
        );
    }
}

#[test]
fn hc001_wrong_previous_or_reused_id_results_never_certify_an_older_occurrence() {
    let wire = hc001_wire();
    for blocks in [
        vec![hc001_use("srv_hc001"), hc001_result("srv_other")],
        vec![hc001_result("srv_hc001"), hc001_use("srv_hc001")],
        vec![
            hc001_use("srv_hc001"),
            hc001_use("srv_hc001"),
            hc001_result("srv_hc001"),
        ],
        vec![hc001_use("srv_hc001"), hc001_result("")],
    ] {
        let source = hc001_prefix(&wire, &blocks, true);
        let request = hc001_request(&source, Some(message::Message::user("next")));
        for prepared in [false, true] {
            hc001_refused(hc001_encode(&wire, &request, &[&source], prepared).0);
        }
    }
    let first = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    let reused = hc001_complete(
        &wire,
        vec![hc001_use("srv_hc001"), hc001_result("srv_hc001")],
        "end_turn",
    );
    let request = CompletionRequest::from(vec![
        message::Message::user("q"),
        first.message().expect("first occurrence"),
        reused.message().expect("new occurrence"),
        message::Message::user("next"),
    ])
    .max_tokens(64);
    for prepared in [false, true] {
        hc001_refused(hc001_encode(&wire, &request, &[&first, &reused], prepared).0);
    }
}

#[test]
fn hc001_results_only_continuation_allows_a_later_hosted_result_before_new_input() {
    let wire = hc001_wire();
    let client_raw = json!({"type":"tool_use", "id":"toolu_hc001", "name":"lookup", "input":{"q":"synthetic query"}});
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001"), client_raw.clone()], true);
    let call = source
        .tool_calls()
        .next()
        .expect("real ordinary client call");
    let output = message::Message::User {
        content: vec![message::UserContent::ToolResult(
            call.result(vec![message::ToolResultContent::text("client result")]),
        )],
    };
    let later = hc001_complete(&wire, vec![hc001_result("srv_hc001")], "end_turn");
    let request = CompletionRequest::from(vec![
        message::Message::user("q"),
        source.message().expect("mixed source"),
        output,
        later.message().expect("later hosted result"),
        message::Message::user("next"),
    ])
    .max_tokens(64);
    // Direct native encoding keeps this received client block and result. Shared
    // failed-turn adaptation has its separate existing client-call disposition.
    assert_eq!(
        hc001_encode(&wire, &request, &[&source, &later], false)
            .0
            .expect("mixed continuation"),
        hc001_body(vec![
            hc001_user("q"),
            hc001_assistant(vec![hc001_use("srv_hc001"), client_raw]),
            json!({"role":"user", "content":[{"type":"tool_result", "tool_use_id":"toolu_hc001",
                "content":[{"type":"text","text":"client result"}]}]}),
            hc001_assistant(vec![hc001_result("srv_hc001")]),
            hc001_user("next")
        ])
    );
    assert_eq!(
        hc001_encode(&wire, &request, &[&source, &later], true)
            .0
            .expect("prepared continuation"),
        hc001_body(vec![
            hc001_user("q"),
            hc001_assistant(vec![hc001_use("srv_hc001")]),
            hc001_assistant(vec![hc001_result("srv_hc001")]),
            hc001_user("next")
        ])
    );
    let no_result = hc001_request(
        &source,
        Some(message::Message::User {
            content: vec![message::UserContent::ToolResult(
                call.result(vec![message::ToolResultContent::text("client result")]),
            )],
        }),
    );
    let direct = hc001_encode(&wire, &no_result, &[&source], false)
        .0
        .expect("results-only continuation stays open");
    assert_eq!(
        direct,
        hc001_body(vec![
            hc001_user("q"),
            hc001_assistant(vec![
                hc001_use("srv_hc001"),
                json!({"type":"tool_use", "id":"toolu_hc001", "name":"lookup", "input":{"q":"synthetic query"}})
            ]),
            json!({"role":"user", "content":[{"type":"tool_result", "tool_use_id":"toolu_hc001",
            "content":[{"type":"text","text":"client result"}]}]})
        ])
    );
}

#[test]
fn hc001_no_new_input_and_wire_omitted_blank_user_text_stay_open() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    for next in [
        None,
        Some(message::Message::user("")),
        Some(message::Message::user(" \n\t ")),
    ] {
        let request = hc001_request(&source, next);
        for prepared in [false, true] {
            assert_eq!(
                hc001_encode(&wire, &request, &[&source], prepared)
                    .0
                    .expect("no ordinary user boundary"),
                hc001_body(vec![
                    hc001_user("q"),
                    hc001_assistant(vec![hc001_use("srv_hc001")])
                ])
            );
        }
    }
}

#[test]
fn hc001_nonblank_text_and_non_tool_result_media_are_new_input_boundaries() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    for input in [
        message::UserContent::text(" next "),
        message::UserContent::Image(message::Image {
            data: message::DocumentSourceKind::url("https://example.com/synthetic.png"),
            media_type: None,
            detail: None,
            native: None,
        }),
    ] {
        let request = hc001_request(
            &source,
            Some(message::Message::User {
                content: vec![input],
            }),
        );
        for prepared in [false, true] {
            hc001_refused(hc001_encode(&wire, &request, &[&source], prepared).0);
        }
    }
}

#[test]
fn hc001_non_aborted_stops_and_an_open_hosted_input_preserve_the_existing_routes() {
    let wire = hc001_wire();
    for reason in [
        "end_turn",
        "pause_turn",
        "max_tokens",
        "tool_use",
        "refusal",
    ] {
        let source = hc001_complete(&wire, vec![hc001_use("srv_hc001")], reason);
        let request = hc001_request(&source, Some(message::Message::user("next")));
        for prepared in [false, true] {
            assert_eq!(
                hc001_encode(&wire, &request, &[&source], prepared)
                    .0
                    .expect("non-Aborted carrier route"),
                hc001_body(vec![
                    hc001_user("q"),
                    hc001_assistant(vec![hc001_use("srv_hc001")]),
                    hc001_user("next")
                ])
            );
        }
    }
    let open = hc001_prefix(&wire, &[hc001_use("srv_hc001")], false);
    assert!(
        open.choice.is_empty(),
        "actual decoder withholds the open protected use"
    );
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(
                &wire,
                &hc001_request(&open, Some(message::Message::user("next"))),
                &[&open],
                prepared
            )
            .0
            .expect("open source route"),
            hc001_body(vec![hc001_user("q"), hc001_user("next")])
        );
    }
}

#[test]
fn hc001_ordinary_client_calls_and_non_hosted_carried_text_do_not_become_server_uses() {
    let wire = hc001_wire();
    let client = json!({"type":"tool_use", "id":"toolu_hc001", "name":"lookup", "input":{}});
    let source = hc001_prefix(&wire, &[client.clone()], true);
    let request = hc001_request(&source, Some(message::Message::user("next")));
    assert_eq!(
        hc001_encode(&wire, &request, &[&source], false)
            .0
            .expect("ordinary native client route"),
        hc001_body(vec![
            hc001_user("q"),
            hc001_assistant(vec![client]),
            hc001_user("next")
        ])
    );
    assert_eq!(
        hc001_encode(&wire, &request, &[&source], true)
            .0
            .expect("shared failed client omission unchanged"),
        hc001_body(vec![hc001_user("q"), hc001_user("next")])
    );
    let text_source = hc001_complete(
        &wire,
        vec![json!({"type":"text", "text":"synthetic known text"})],
        "pause_turn",
    );
    let mut request = hc001_request(&text_source, Some(message::Message::user("next")));
    // Labelled synthetic stop boundary on genuinely decoded inspection text.
    let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
        panic!("real head clone");
    };
    turn.stop = Some(message::StopReason::Aborted(
        "synthetic stop boundary".to_owned(),
    ));
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(&wire, &request, &[&text_source], prepared)
                .0
                .expect("other carried class"),
            hc001_body(vec![
                hc001_user("q"),
                hc001_assistant(vec![json!({"type":"text","text":"synthetic known text"})]),
                hc001_user("next")
            ])
        );
    }
}

#[test]
fn hc001_compatible_dialects_keep_original_hosted_shapes_and_proper_carrier_refusals() {
    for dialect in [
        &super::super::wire::MOONSHOT,
        &super::super::wire::ZAI,
        &super::super::wire::MINIMAX,
        &super::super::wire::XIAOMIMIMO,
    ] {
        let wire =
            AnthropicConfig::with_key(dialect, "synthetic-test-key").completion(CLAUDE_SONNET_4_6);
        let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
        let request = hc001_request(&source, Some(message::Message::user("next")));
        assert_eq!(
            hc001_encode(&wire, &request, &[&source], false)
                .0
                .expect("compatible direct native route"),
            hc001_body(vec![
                hc001_user("q"),
                hc001_assistant(vec![hc001_use("srv_hc001")]),
                hc001_user("next")
            ])
        );
        assert_eq!(
            hc001_encode(&wire, &request, &[&source], true)
                .0
                .expect("compatible failed Opaque omission unchanged"),
            hc001_body(vec![hc001_user("q"), hc001_user("next")])
        );
        let proper = hc001_prefix(&hc001_wire(), &[hc001_use("srv_hc001")], true);
        let error = hc001_encode(
            &wire,
            &hc001_request(&proper, Some(message::Message::user("next"))),
            &[&proper],
            false,
        )
        .0
        .expect_err("foreign proper carrier still refuses");
        let ProviderError::Request(cause) = error else {
            panic!("existing carried shape refusal");
        };
        assert!(
            cause
                .downcast_ref::<message::UnrepresentableOpaqueContent>()
                .is_some()
        );
    }
}

#[test]
fn hc001_raw_admission_and_messages_override_cannot_bypass_targeted_canonical_refusal() {
    let wire = hc001_wire();
    let source = hc001_prefix(
        &wire,
        &[hc001_use("srv_hc001"), hc001_result("srv_hc001")],
        true,
    );
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    request.additional_params = Some(json!({"metadata":{"user_id":"synthetic exact metadata"},
        "temperature":0.125, "messages":[{"role":"user","content":"synthetic raw replacement"}]}));
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(&wire, &request, &[&source], prepared)
                .0
                .expect("existing admitted raw merge"),
            json!({"model":CLAUDE_SONNET_4_6, "max_tokens":64, "temperature":0.125,
                "metadata":{"user_id":"synthetic exact metadata"},
                "messages":[{"role":"user","content":"synthetic raw replacement"}]})
        );
    }
    let unresolved = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    request.chat_history =
        hc001_request(&unresolved, Some(message::Message::user("next"))).chat_history;
    for prepared in [false, true] {
        hc001_refused(hc001_encode(&wire, &request, &[&unresolved], prepared).0);
    }
    // Explicit scope limitation: raw-only server history has no canonical
    // Aborted occurrence and is not certified by this policy.
    let mut raw_only = CompletionRequest::new("canonical unrelated prompt").max_tokens(64);
    let messages = json!([
        hc001_user("q"),
        hc001_assistant(vec![hc001_use("srv_hc001")]),
        hc001_user("next")
    ]);
    raw_only.additional_params = Some(json!({"messages":messages}));
    assert_eq!(
        hc001_encode(&wire, &raw_only, &[], false)
            .0
            .expect("raw-only path remains outside policy"),
        json!({"model":CLAUDE_SONNET_4_6,"max_tokens":64,"messages":messages})
    );
}

#[test]
fn hc001_original_shape_errors_precede_policy_and_policy_precedes_budget_and_cache_errors() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    for boundary in ["foreign", "edited", "malformed_result"] {
        let mut request = hc001_request(&source, Some(message::Message::user("next")));
        let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
            panic!("genuine head clone");
        };
        let mut extra = turn.content[0].clone();
        let AssistantContent::CarriedText(part) = &mut extra else {
            panic!("actual received carrier");
        };
        match boundary {
            "foreign" => part.payload.provider = Some("synthetic foreign clone".to_owned()),
            "edited" => {
                part.content = message::CarriedTextContent::Opaque("synthetic edit".to_owned())
            }
            "malformed_result" => {
                part.payload.item =
                    json!({"type":"web_search_tool_result","tool_use_id":7,"content":[]})
            }
            _ => unreachable!("listed synthetic boundary"),
        }
        turn.content.push(extra);
        let error = hc001_encode(&wire, &request, &[&source], false)
            .0
            .expect_err("shape refusal wins");
        let ProviderError::Request(cause) = error else {
            panic!("typed original shape refusal");
        };
        assert!(
            cause
                .downcast_ref::<message::UnrepresentableOpaqueContent>()
                .is_some()
        );
    }
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    request.max_tokens = None;
    let mut no_budget = wire.clone();
    no_budget.default_max_tokens = None;
    hc001_refused(hc001_encode(&no_budget, &request, &[&source], false).0);
    request.chat_history = vec![message::Message::user("unrelated")];
    let error = hc001_encode(&no_budget, &request, &[&source], false)
        .0
        .expect_err("existing budget refusal");
    let ProviderError::Request(cause) = error else {
        panic!("budget Request error");
    };
    assert_eq!(cause.to_string(), "`max_tokens` must be set for Anthropic");
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    request.options.cache = Some(completion::CacheRetention::Long);
    request.additional_params = Some(json!({"cache_control":{"type":"ephemeral","ttl":"5m"}}));
    hc001_refused(hc001_encode(&wire, &request, &[&source], false).0);
    request.chat_history = vec![message::Message::user("unrelated")];
    let error = hc001_encode(&wire, &request, &[&source], false)
        .0
        .expect_err("existing cache conflict");
    let ProviderError::Request(cause) = error else {
        panic!("cache Request error");
    };
    assert_eq!(
        cause.to_string(),
        "Raw top-level cache_control conflicts with the typed automatic caching TTL"
    );
}

#[test]
fn hc001_opaque_raw_sibling_and_empty_use_id_never_acquire_pairing_authority() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
        panic!("genuine head clone");
    };
    // Synthetic raw sibling with no carried issuer does not certify the use.
    turn.content.push(AssistantContent::Opaque(message::Opaque {
        item: hc001_result("srv_hc001"),
        replay: true,
    }));
    hc001_refused(hc001_encode(&wire, &request, &[&source], false).0);
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
        panic!("genuine head clone");
    };
    let AssistantContent::CarriedText(part) = &mut turn.content[0] else {
        panic!("actual carrier clone");
    };
    part.payload.item["id"] = json!("");
    for prepared in [false, true] {
        assert_eq!(
            hc001_encode(&wire, &request, &[&source], prepared)
                .0
                .expect("empty ID stays on old route"),
            hc001_body(vec![
                hc001_user("q"),
                hc001_assistant(vec![hc001_use("")]),
                hc001_user("next")
            ])
        );
    }
}

#[test]
fn hc001_previous_response_and_result_after_new_input_do_not_certify_completion() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    let result = hc001_complete(&wire, vec![hc001_result("srv_hc001")], "end_turn");
    for history in [
        vec![
            message::Message::user("q"),
            result.message().expect("earlier result"),
            source.message().expect("later use"),
            message::Message::user("next"),
        ],
        vec![
            message::Message::user("q"),
            source.message().expect("earlier use"),
            message::Message::user("next"),
            result.message().expect("result after boundary"),
        ],
    ] {
        let request = CompletionRequest::from(history).max_tokens(64);
        for prepared in [false, true] {
            hc001_refused(hc001_encode(&wire, &request, &[&source, &result], prepared).0);
        }
    }
}

#[test]
fn hc001_provider_document_and_json_call_checks_keep_priority_over_hosted_policy() {
    let wire = hc001_wire();
    let source = hc001_prefix(&wire, &[hc001_use("srv_hc001")], true);
    let mut request = hc001_request(&source, Some(message::Message::user("next")));
    let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
        panic!("genuine head clone");
    };
    let AssistantContent::CarriedText(part) = &turn.content[0] else {
        panic!("received hosted payload");
    };
    // Synthetic category clone retains the actual payload/issuer; no issuer is minted.
    let payload = part.payload.clone();
    turn.content
        .push(AssistantContent::ProviderItem(message::ProviderItem {
            payload,
        }));
    let error = hc001_encode(&wire, &request, &[&source], false)
        .0
        .expect_err("provider-item refusal wins");
    let ProviderError::Request(cause) = error else {
        panic!("existing provider Request refusal");
    };
    assert!(
        cause
            .downcast_ref::<message::UnreplayableProviderItem>()
            .is_some()
    );
    let mut request = hc001_request(
        &source,
        Some(message::Message::User {
            content: vec![message::UserContent::Document(message::Document {
                data: message::DocumentData::File(message::DocumentSourceKind::url(
                    "https://example.com/synthetic.txt",
                )),
                media_type: Some(message::DocumentMediaType::TXT),
                additional_params: None,
            })],
        }),
    );
    let error = hc001_encode(&wire, &request, &[&source], false)
        .0
        .expect_err("document check wins");
    let ProviderError::Request(cause) = error else {
        panic!("existing document Request refusal");
    };
    assert_eq!(
        cause.to_string(),
        "Only string or base64 data is supported for plain text documents"
    );
    let client = hc001_complete(
        &wire,
        vec![json!({"type":"tool_use","id":"toolu_hc001","name":"lookup","input":{}})],
        "tool_use",
    );
    let mut call = client.choice[0].clone();
    let AssistantContent::ToolCall(tool) = &mut call else {
        panic!("actual decoded client call");
    };
    tool.function.namespace = Some("synthetic namespace clone".to_owned());
    request.chat_history =
        hc001_request(&source, Some(message::Message::user("next"))).chat_history;
    let message::Message::Assistant(turn) = &mut request.chat_history[1] else {
        panic!("genuine head clone");
    };
    turn.content.push(call);
    let error = hc001_encode(&wire, &request, &[&source, &client], false)
        .0
        .expect_err("JSON call shape wins");
    let ProviderError::Request(cause) = error else {
        panic!("existing call-kind Request refusal");
    };
    assert!(
        matches!(cause.downcast_ref::<message::UnrepresentableToolCall>(),
        Some(message::UnrepresentableToolCall::Namespace { wire, name, namespace })
            if *wire == ANTHROPIC_MESSAGES_WIRE && name.as_str() == "lookup" && namespace == "synthetic namespace clone")
    );
}
