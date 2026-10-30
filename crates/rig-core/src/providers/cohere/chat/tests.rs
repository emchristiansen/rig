use std::collections::HashMap;

use super::*;
use crate::completion::{CompletionResponse, ReplayTarget, ToolDefinition};
use crate::message::{
    CallId, DocumentMediaType, Image, ImageMediaType, Reasoning, Text, ToolCall, ToolFunction,
    ToolName,
};
use crate::providers::cohere::{ChatRoute, CohereChat, CohereConfig};
use crate::test_utils::json_body;
use crate::wire::{Operation, WireFrame};

const MODEL: &str = "command-a-03-2025";

fn chat(route: ChatRoute) -> CohereChat {
    CohereConfig::new("key")
        .with_base_url("http://127.0.0.1:9")
        .completion(MODEL)
        .with_route(route)
}

fn native() -> NativeChat {
    NativeChat::new(CohereConfig::new("key"), MODEL)
}

/// What `wire` sends for `request`, prepared as the driver prepares it:
/// its path and body.
fn sent<W: Wire<Op = Completion, Payload = Encoded>>(
    wire: &W,
    request: CompletionRequest,
    mode: Mode,
) -> (String, Value) {
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire.encode(request, mode).expect("the request encodes");
    (
        encoded.request.uri().path().to_owned(),
        json_body(&encoded.request),
    )
}

fn documents() -> Vec<Document> {
    vec![
        Document {
            id: "harbor-1".to_owned(),
            text: "Beacon amber-73 is at Dock Seven.".to_owned(),
            additional_props: HashMap::from([
                ("source".to_owned(), "registry".to_owned()),
                ("region".to_owned(), "north".to_owned()),
            ]),
        },
        Document {
            id: String::new(),
            text: "Beacon violet-19 is at Dock Three.".to_owned(),
            additional_props: HashMap::new(),
        },
    ]
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new(name).expect("a tool name"),
        "looks things up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )
}

/// Documents go as Cohere `documents`, each under its id (or a positional
/// one) with its metadata and text sorted under `data`, and never into the
/// history.
#[test]
fn documents_are_sent_as_cohere_documents() {
    let mut request = CompletionRequest::new("Which dock?");
    request.documents = documents();
    let (path, body) = sent(&native(), request, Mode::Unary);
    assert_eq!(path, "/v2/chat");
    assert_eq!(
        body,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Which dock?"}]}],
            "documents": [
                {"id": "harbor-1", "data": {"region": "north", "source": "registry",
                    "text": "Beacon amber-73 is at Dock Seven."}},
                {"id": "doc_1", "data": {"text": "Beacon violet-19 is at Dock Three."}},
            ],
        })
    );
    assert_eq!(
        serde_json::to_string(&body["documents"][0]["data"]).expect("serializes"),
        r#"{"region":"north","source":"registry","text":"Beacon amber-73 is at Dock Seven."}"#
    );
}

/// The automatic route sends a request with documents to the native API
/// and any other to the Compatibility API; the explicit routes override it.
#[test]
fn the_route_follows_the_request_and_the_setting() {
    let with_documents = || {
        let mut request = CompletionRequest::new("Which dock?");
        request.documents = documents();
        request
    };
    let (path, body) = sent(&chat(ChatRoute::Auto), with_documents(), Mode::Streaming);
    assert_eq!(path, "/v2/chat");
    assert_eq!(body["stream"], true);
    assert_eq!(body["documents"][0]["id"], "harbor-1");
    let (path, _) = sent(
        &chat(ChatRoute::Auto),
        CompletionRequest::new("hi"),
        Mode::Unary,
    );
    assert_eq!(path, "/compatibility/v1/chat/completions");
    let (path, body) = sent(
        &chat(ChatRoute::Native),
        CompletionRequest::new("hi"),
        Mode::Unary,
    );
    assert_eq!(path, "/v2/chat");
    assert!(body.get("documents").is_none(), "{body}");
    let (path, body) = sent(
        &chat(ChatRoute::Compatibility),
        with_documents(),
        Mode::Unary,
    );
    assert_eq!(path, "/compatibility/v1/chat/completions");
    assert!(body.get("documents").is_none(), "{body}");
    assert!(
        body["messages"].to_string().contains("Dock Seven"),
        "the Compatibility API gets documents as text: {body}"
    );
    let default = CohereConfig::new("key").completion("command-a-03-2025");
    assert_eq!(default.route, ChatRoute::Auto);
    let (path, _) = sent(&default, with_documents(), Mode::Unary);
    assert_eq!(
        path, "/v2/chat",
        "default documents keep their native meaning"
    );
}

/// Tools, the tool choice, strict tools, the output schema and sampling
/// settings land where Cohere reads them, and `additional_params` override
/// them last.
#[test]
fn request_settings_land_where_cohere_reads_them() {
    let mut request = CompletionRequest::new("hi")
        .tool(tool("lookup"))
        .tool(tool("other"))
        .temperature(0.2)
        .max_tokens(64);
    request.tool_choice = Some(ToolChoice::Specific {
        function_names: vec![ToolName::new("lookup").expect("a tool name")],
    });
    request.output_schema = Some(schemars::json_schema!({"type": "object"}));
    request.additional_params = Some(json!({"seed": 7, "temperature": 0.5,
        "tools": [{"type": "function", "function": {"name": "raw", "parameters": {}}}]}));
    let (_, body) = sent(&native().with_strict_tools(), request, Mode::Unary);
    assert_eq!(
        body["tools"],
        json!([
            {"type": "function", "function": {"name": "lookup", "description": "looks things up",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
        ])
    );
    assert_eq!(body["tool_choice"], "REQUIRED");
    assert_eq!(body["strict_tools"], true);
    assert_eq!(
        body["response_format"],
        json!({"type": "json_object", "schema": {"type": "object"}})
    );
    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["seed"], 7);

    for (choice, sent_choice) in [
        (ToolChoice::Auto, None),
        (ToolChoice::None, Some("NONE")),
        (ToolChoice::Required, Some("REQUIRED")),
    ] {
        let mut request = CompletionRequest::new("hi").tool(tool("lookup"));
        request.tool_choice = Some(choice);
        let (_, body) = sent(&native(), request, Mode::Unary);
        assert_eq!(body.get("tool_choice").and_then(Value::as_str), sent_choice);
    }
    let mut toolless = CompletionRequest::new("hi");
    toolless.tool_choice = Some(ToolChoice::Required);
    let error = native()
        .encode(toolless, Mode::Unary)
        .expect_err("Required needs tools");
    let error = crate::error::ProviderError::from(error);
    assert_eq!(error.kind(), crate::error::ErrorKind::Request);
}

/// Malformed `additional_params` and a history with nothing to send are
/// request errors.
#[test]
fn malformed_requests_are_rejected() {
    for params in [json!([1]), json!({"tools": {"type": "function"}})] {
        let mut request = CompletionRequest::new("hi");
        request.additional_params = Some(params);
        assert!(native().encode(request, Mode::Unary).is_err());
    }
    let mut empty = CompletionRequest::new("hi");
    empty.chat_history.clear();
    assert!(native().encode(empty, Mode::Unary).is_err());
}

/// User text, images and text documents are content parts, and a result
/// answers its call in a `tool` message, its parts joined into one text.
#[test]
fn user_content_and_results_are_cohere_messages() {
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("lookup").expect("a tool name"),
            json!({"q": "rig"}),
        ),
    );
    let image = |data: Source| {
        UserContent::Image(Image {
            data,
            media_type: Some(ImageMediaType::PNG),
            ..Image::default()
        })
    };
    let history = vec![
        Message::system("be brief"),
        Message::User {
            content: vec![
                UserContent::text("look"),
                image(Source::base64("aGk=")),
                image(Source::Url("https://rig.rs/a.png".to_owned())),
                UserContent::document_text("notes", Some(DocumentMediaType::TXT)),
            ],
        },
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(call.clone())],
            origin: None,
            stop: None,
        }),
        Message::User {
            content: vec![
                UserContent::ToolResult(call.result(vec![
                    ToolResultContent::text("found"),
                    ToolResultContent::Json {
                        value: json!({"n": 1}),
                    },
                ])),
                UserContent::text("thanks"),
            ],
        },
    ];
    let wire = NativeChat::new(CohereConfig::new("key"), "command-a-vision-07-2025");
    let mut request = CompletionRequest::new("next").tool(tool("lookup"));
    request.chat_history.splice(0..0, history);
    let (_, body) = sent(&wire, request, Mode::Unary);
    assert_eq!(
        body["messages"],
        json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                {"type": "image_url", "image_url": {"url": "https://rig.rs/a.png"}},
                {"type": "text", "text": "notes"},
            ]},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"rig\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "found\n{\"n\":1}"},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "thanks"},
                {"type": "text", "text": "next"},
            ]},
        ])
    );
}

/// Content replay leaves out never reaches the encoder: it errors rather
/// than guessing.
#[test]
fn unsendable_parts_are_errors() {
    for part in [
        UserContent::Image(Image {
            data: Source::base64("aGk="),
            ..Image::default()
        }),
        UserContent::document_base64("JVBERi0=", Some(DocumentMediaType::PDF)),
        UserContent::audio_base64("aGk=", None),
        UserContent::video_base64("aGk=", None),
    ] {
        let request = CompletionRequest::from(vec![Message::User {
            content: vec![part.clone()],
        }]);
        assert!(native().encode(request, Mode::Unary).is_err(), "{part:?}");
    }
}

/// A native reply holding thinking, a cited answer, a tool plan and a call.
fn native_reply() -> Value {
    json!({
        "id": "resp_1",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan it"},
                {"type": "text", "text": "Dock Seven."},
            ],
            "tool_plan": "I will look it up.",
            "tool_calls": [{"id": "lookup_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"dock\"}"}}],
            "citations": [
                {"start": 0, "end": 10, "text": "Dock Seven", "type": "TEXT_CONTENT",
                    "content_index": 1,
                    "sources": [{"type": "document", "id": "harbor-1",
                        "document": {"id": "harbor-1", "text": "Beacon amber-73 is at Dock Seven."}}]},
                {"start": 0, "end": 4, "text": "look", "type": "PLAN",
                    "sources": [{"type": "document", "id": "harbor-1"}]},
            ],
        },
        "finish_reason": "TOOL_CALL",
        "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
    })
}

/// The response `wire` folds `reply` into for `request`.
fn decoded(wire: &CohereChat, request: &CompletionRequest, reply: &Value) -> CompletionResponse {
    let request =
        Completion::prepare(request.clone(), &wire.describe()).expect("the request prepares");
    crate::test_utils::decode_reply(
        wire,
        &request,
        Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds")
}

/// A history holding `turn`, answered, and a next question.
fn continued(turn: &CompletionResponse) -> CompletionRequest {
    let call = turn
        .choice
        .iter()
        .find_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("the turn holds a call");
    let mut request = CompletionRequest::new("And then?").tool(tool("lookup"));
    request.chat_history.splice(
        0..0,
        [
            Message::user("Which dock?"),
            turn.message().expect("an assistant turn"),
            Message::User {
                content: vec![UserContent::ToolResult(
                    call.result(vec![ToolResultContent::text("Dock Seven")]),
                )],
            },
        ],
    );
    request
}

/// A native turn replays on the native API with its thinking, tool plan,
/// call and citations, each citation pointed at the part it cites; on the
/// Compatibility API it replays from its canonical fields only.
#[test]
fn a_native_turn_replays_natively_and_canonically_elsewhere() {
    let mut first = CompletionRequest::new("Which dock?").tool(tool("lookup"));
    first.documents = documents();
    let turn = decoded(&chat(ChatRoute::Auto), &first, &native_reply());
    assert_eq!(turn.origin.api.as_str(), API);

    let (_, body) = sent(&chat(ChatRoute::Native), continued(&turn), Mode::Unary);
    let assistant = &body["messages"][1];
    assert_eq!(
        assistant,
        &json!({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan it"},
                {"type": "text", "text": "Dock Seven."},
            ],
            "tool_plan": "I will look it up.",
            "tool_calls": [{"id": "lookup_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"dock\"}"}}],
            "citations": [
                {"start": 0, "end": 4, "text": "look", "type": "PLAN",
                    "sources": [{"type": "document", "id": "harbor-1"}]},
                {"start": 0, "end": 10, "text": "Dock Seven", "type": "TEXT_CONTENT",
                    "content_index": 1,
                    "sources": [{"type": "document", "id": "harbor-1",
                        "document": {"id": "harbor-1", "text": "Beacon amber-73 is at Dock Seven."}}]},
            ],
        })
    );
    assert_eq!(body["messages"][2]["tool_call_id"], "lookup_1");

    let (path, body) = sent(&chat(ChatRoute::Auto), continued(&turn), Mode::Unary);
    assert_eq!(path, "/compatibility/v1/chat/completions");
    let assistant = &body["messages"][1];
    let sent = assistant.to_string();
    assert!(
        !sent.contains("citations") && !sent.contains("tool_plan"),
        "{assistant}"
    );
    assert!(sent.contains("Dock Seven."), "{assistant}");
    assert_eq!(assistant["tool_calls"][0]["id"], "lookup_1");
    assert_eq!(body["messages"][2]["tool_call_id"], "lookup_1");
}

/// A Compatibility API turn replays on the native API from its canonical
/// fields: its Chat items never reach the native request.
#[test]
fn a_compatibility_turn_replays_canonically_on_the_native_api() {
    let reply = json!({
        "id": "chatcmpl_1", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "Looking.", "reasoning_content": "think",
            "tool_calls": [{"id": "lookup_2", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"dock\"}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    });
    let first = CompletionRequest::new("Which dock?").tool(tool("lookup"));
    let turn = decoded(&chat(ChatRoute::Auto), &first, &reply);
    assert_eq!(turn.origin.api.as_str(), "openai.chat");

    let mut next = continued(&turn);
    next.documents = documents();
    let (path, body) = sent(&chat(ChatRoute::Auto), next, Mode::Unary);
    assert_eq!(path, "/v2/chat");
    assert_eq!(
        body["messages"][1],
        json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "think"},
                {"type": "text", "text": "Looking."},
            ],
            "tool_calls": [{"id": "lookup_2", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"dock\"}"}}],
        })
    );
}

/// An edited tool plan stays a plan, canonical reasoning is thinking, and
/// an unknown part that replays goes back as it came.
#[test]
fn edited_and_canonical_reasoning_keep_their_kind() {
    let edited = AssistantContent::Reasoning(Reasoning::new("new plan")).with_native(json!({
        "type": PLAN, PLAN: "old plan"
    }));
    let AssistantContent::Reasoning(reasoning) = &edited else {
        panic!("a reasoning block");
    };
    let mut stale = reasoning.clone();
    stale.text = "newer plan".to_owned();
    let stale = AssistantContent::Reasoning(stale);
    let turn = AssistantMessage {
        content: vec![
            stale,
            AssistantContent::Reasoning(Reasoning::new("musing")),
            AssistantContent::Text(Text::new("")),
            AssistantContent::Opaque(crate::message::Opaque {
                item: json!({"type": "x"}),
                replay: true,
            }),
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("c1"),
                ToolFunction::new(ToolName::new("lookup").expect("a tool name"), json!({})),
            )),
        ],
        origin: Some(crate::message::Origin::new(API, "cohere", MODEL)),
        stop: None,
    };
    let ids = WireIds::for_target(&[Message::Assistant(turn.clone())], &native(), MODEL)
        .expect("the call history plans");
    let message = native()
        .assistant(&turn, 0, &ids)
        .expect("the turn encodes")
        .expect("the turn sends");
    assert_eq!(message["tool_plan"], "newer plan");
    assert_eq!(
        message["content"],
        json!([{"type": "thinking", "thinking": "musing"}, {"type": "x"}])
    );
    let empty = AssistantMessage {
        content: vec![AssistantContent::Text(Text::new(""))],
        origin: None,
        stop: None,
    };
    assert!(
        native()
            .assistant(&empty, 0, &WireIds::default())
            .expect("the empty turn encodes")
            .is_none()
    );
}

#[test]
fn carried_known_content_uses_canonical_text_without_issuer_gating() {
    use crate::message::{
        CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText, CarriedTextContent,
    };

    let turn = AssistantMessage {
        content: vec![
            AssistantContent::CarriedText(CarriedText::new(
                CarriedTextContent::InspectionOnlyMedia("answer".to_owned()),
                CarriedPayload::new(
                    json!({"unknown": "inspection data"}),
                    Some("foreign".to_owned()),
                ),
            )),
            AssistantContent::CarriedReasoning(CarriedReasoning::new(
                vec![
                    CarriedReasoningPart::Summary("first".to_owned()),
                    CarriedReasoningPart::Text {
                        text: "second".to_owned(),
                        signature: None,
                    },
                ],
                CarriedPayload::new(
                    json!({"unknown": "reasoning data"}),
                    Some("foreign".to_owned()),
                ),
            )),
        ],
        origin: None,
        stop: None,
    };
    let message = native()
        .assistant(&turn, 0, &WireIds::default())
        .expect("known content encodes")
        .expect("known content sends");
    assert_eq!(
        message,
        json!({"role":"assistant", "content":[
            {"type":"text", "text":"answer"},
            {"type":"thinking", "thinking":"first\nsecond"}
        ]})
    );
}

/// What replay reads of the native target.
#[test]
fn the_native_target_states_what_it_carries() {
    use crate::completion::{Media, Place};
    let wire = native();
    assert_eq!(wire.api().as_str(), API);
    assert_eq!(ReplayTarget::provider(&wire), "cohere");
    assert_eq!(ReplayTarget::model(&wire), MODEL);
    assert!(!wire.accepts(MODEL).user_images);
    assert!(wire.accepts("command-a-vision-07-2025").user_images);
    assert!(wire.takes_documents());
    assert_eq!(wire.call_id_slot(), Some("/id"));
    let url = Image {
        data: Source::Url("https://rig.rs/a.png".to_owned()),
        ..Image::default()
    };
    let untyped = Image {
        data: Source::base64("aGk="),
        ..Image::default()
    };
    let raw = Image {
        data: Source::Raw(vec![1]),
        ..Image::default()
    };
    assert!(wire.encodes(MODEL, Media::Image(&url, Place::User)));
    assert!(!wire.encodes(MODEL, Media::Image(&untyped, Place::User)));
    assert!(!wire.encodes(MODEL, Media::Image(&raw, Place::User)));
    assert!(!wire.encodes(MODEL, Media::Image(&url, Place::ToolResult)));
    let text = crate::message::Document {
        data: crate::message::DocumentData::Text("notes".to_owned()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    };
    let pdf = crate::message::Document {
        data: Source::base64("JVBERi0=").into(),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    };
    assert!(wire.encodes(MODEL, Media::Document(&text)));
    assert!(!wire.encodes(MODEL, Media::Document(&pdf)));
    assert!(wire.identity(&json!({"type": "thinking"})).is_empty());
    assert_eq!(
        wire.identity(&json!({"type": PLAN})),
        Map::from_iter([("type".to_owned(), json!(PLAN))])
    );
    let plan = AssistantContent::Reasoning(Reasoning::new("plan"))
        .with_native(json!({"type": PLAN, PLAN: "plan"}));
    assert!(!wire.sends_alone(&plan));
    assert!(wire.sends_alone(&AssistantContent::Reasoning(Reasoning::new("think"))));
    assert!(!wire.sends_alone(&AssistantContent::text("")));
    assert!(wire.sends_alone(&AssistantContent::text("hi")));
    assert!(!wire.sends_alone(&AssistantContent::Image(url)));
    let opaque =
        |item: Value| AssistantContent::Opaque(crate::message::Opaque { item, replay: true });
    assert!(wire.sends_alone(&opaque(json!({"type": "x"}))));
    assert!(!wire.sends_alone(&opaque(json!({"x": 1}))));
}

/// The routing wire answers replay as its Compatibility API when asked as a
/// whole, and routes each request.
#[test]
fn the_routing_wire_routes_each_request() {
    let wire = chat(ChatRoute::Auto);
    assert_eq!(wire.api().as_str(), "openai.chat");
    assert_eq!(ReplayTarget::provider(&wire), "cohere");
    assert_eq!(ReplayTarget::model(&wire), MODEL);
    assert!(!wire.accepts(MODEL).user_images);
    let mut request = CompletionRequest::new("hi");
    let routed = |wire: &CohereChat, request: &CompletionRequest| {
        wire.route(request).map(|target| target.api())
    };
    assert_eq!(
        routed(&wire, &request).map(|api| api.as_str().to_owned()),
        Some("openai.chat".to_owned())
    );
    request.documents = documents();
    assert_eq!(
        routed(&wire, &request).map(|api| api.as_str().to_owned()),
        Some(API.to_owned())
    );
    let strict = wire.with_strict_tools();
    assert!(strict.compatibility_api.strict_tools && strict.native_api.strict_tools);
}

/// What the native wire answers for `reasoning` on `model`: the body's
/// `thinking`, or the error.
fn native_thinking(
    model: &str,
    reasoning: crate::completion::Reasoning,
) -> Result<Option<Value>, String> {
    let wire = NativeChat::new(CohereConfig::new("key"), model);
    let request = CompletionRequest::new("hi").reasoning(reasoning);
    let encoded = Completion::prepare(request, &wire.describe())
        .map_err(|error| error.to_string())
        .and_then(|request| {
            wire.encode(request, Mode::Unary)
                .map_err(|error| error.to_string())
        })?;
    Ok(json_body(&encoded.request).get("thinking").cloned())
}

/// Whether a model thinks is its catalog entry's `reasoning`: Command A
/// Plus thinks by default, so `Off` turns thinking off, and a listed model
/// that does not think refuses an effort and a budget. An unlisted id
/// thinks by default when its name says `reasoning`, and is sent an effort.
#[test]
fn native_thinking_follows_the_catalog() {
    use crate::completion::{Effort, Reasoning as Thinking};
    assert_eq!(
        native_thinking("command-a-plus-05-2026", Thinking::Off),
        Ok(Some(json!({"type": "disabled"})))
    );
    assert_eq!(
        native_thinking("command-a-reasoning-08-2025", Thinking::Off),
        Ok(Some(json!({"type": "disabled"})))
    );
    assert_eq!(
        native_thinking(
            "command-a-reasoning-08-2025",
            Thinking::Effort(Effort::High)
        ),
        Ok(Some(json!({"type": "enabled"})))
    );
    assert_eq!(
        native_thinking(
            "command-a-reasoning-08-2025",
            Thinking::Budget { tokens: 256 }
        ),
        Ok(Some(json!({"type": "enabled", "token_budget": 256})))
    );
    for reasoning in [
        Thinking::Effort(Effort::High),
        Thinking::Budget { tokens: 256 },
    ] {
        let refused = native_thinking("command-r7b-12-2024", reasoning);
        assert!(
            refused
                .as_ref()
                .is_err_and(|error| error.contains("`reasoning` is not supported")),
            "{reasoning:?}: {refused:?}"
        );
    }
    assert_eq!(
        native_thinking("command-r7b-12-2024", Thinking::Off),
        Ok(None)
    );
    assert_eq!(
        native_thinking("command-z-reasoning-unlisted", Thinking::Off),
        Ok(Some(json!({"type": "disabled"}))),
        "an unlisted id thinks by its name"
    );
    assert_eq!(
        native_thinking("command-z-unlisted", Thinking::Effort(Effort::High)),
        Ok(Some(json!({"type": "enabled"}))),
        "an unlisted id is sent an effort"
    );
}

/// The Compatibility route refuses `high` effort on a model that does not
/// think, by the same catalog rule.
#[test]
fn compatibility_effort_follows_the_catalog() {
    use crate::completion::Effort;
    let effort = |model: &str| {
        let wire = CohereConfig::new("key")
            .completion(model)
            .with_route(ChatRoute::Compatibility);
        let request = CompletionRequest::new("hi").reasoning(Effort::High);
        Completion::prepare(request, &wire.describe())
            .map_err(|error| error.to_string())
            .and_then(|request| {
                wire.encode(request, Mode::Unary)
                    .map(|encoded| json_body(&encoded.request))
                    .map_err(|error| error.to_string())
            })
    };
    let body = effort("command-a-reasoning-08-2025").expect("a model that thinks");
    assert_eq!(body["reasoning_effort"], "high");
    let error = effort("command-r7b-12-2024").expect_err("a model that does not think");
    assert!(error.contains("`reasoning`"), "{error}");
}

// G28 corrects U's broadening of Specific to unselected raw functions.
#[test]
fn specific_tool_choice_filters_typed_and_raw_functions() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("hi")
            .tool(tool("lookup"))
            .tool(tool("other"));
        request.tool_choice = Some(ToolChoice::Specific {
            function_names: vec![
                ToolName::new("lookup").unwrap(),
                ToolName::new("raw_selected").unwrap(),
            ],
        });
        let raw_selected = json!({"type":"function", "function":{"name":"raw_selected", "parameters":{}, "future":7}});
        request.additional_params = Some(json!({"tools":[
            raw_selected.clone(),
            {"type":"function", "function":{"name":"raw_other", "parameters":{}}}
        ], "seed":11}));
        let (_, body) = sent(&native(), request, mode);
        assert_eq!(
            body["tools"],
            json!([
                {"type":"function", "function":{"name":"lookup", "description":"looks things up", "parameters":{"type":"object", "properties":{"q":{"type":"string"}}}}},
                raw_selected,
            ])
        );
        assert_eq!(body["tool_choice"], "REQUIRED");
        assert_eq!(body["seed"], 11);
    }
}

#[test]
fn specific_tool_choice_refuses_unidentifiable_raw_tools() {
    for raw_tool in [
        json!({"type":"function", "function":{"parameters":{}}}),
        json!({"type":"function", "function":{"name":""}}),
        json!({"type":"function", "function":{"name":7}}),
        json!({"type":"custom", "name":"lookup"}),
        Value::Null,
    ] {
        let mut request = CompletionRequest::new("hi").tool(tool("lookup"));
        request.tool_choice = Some(ToolChoice::Specific {
            function_names: vec![ToolName::new("lookup").unwrap()],
        });
        request.additional_params = Some(json!({"tools":[raw_tool]}));
        let error = native()
            .encode(request, Mode::Unary)
            .expect_err("raw tool cannot carry Specific");
        let error = crate::error::ProviderError::from(error);
        assert_eq!(error.kind(), crate::error::ErrorKind::Request);
    }
}

#[test]
fn specific_tool_choice_refuses_an_empty_selected_set() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("hi").tool(tool("other"));
        request.tool_choice = Some(ToolChoice::Specific {
            function_names: vec![ToolName::new("lookup").unwrap()],
        });
        request.additional_params =
            Some(json!({"tools":[{"type":"function", "function":{"name":"raw_other"}}]}));
        let error = native()
            .encode(request, mode)
            .expect_err("Specific selected no effective tool");
        let error = crate::error::ProviderError::from(error);
        assert_eq!(error.kind(), crate::error::ErrorKind::Request);
    }
}

// G29 keeps the requirement effective, including tools supplied as raw JSON.
#[test]
fn required_tool_choice_refuses_no_tools_and_accepts_raw_tools() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("hi");
        request.tool_choice = Some(ToolChoice::Required);
        let error = native()
            .encode(request.clone(), mode)
            .expect_err("Required has no tools");
        let error = crate::error::ProviderError::from(error);
        assert_eq!(error.kind(), crate::error::ErrorKind::Request);
        let raw_tool = json!({"type":"function", "function":{"name":"lookup", "parameters":{}}});
        request.additional_params = Some(json!({"tools":[raw_tool.clone()]}));
        let (_, body) = sent(&native(), request, mode);
        assert_eq!(body["tools"], json!([raw_tool]));
        assert_eq!(body["tool_choice"], "REQUIRED");
    }
}

// G32 fails conflicting mandatory instructions before the raw layer merges.
#[test]
fn mandatory_tool_choice_refuses_conflicting_raw_override() {
    for choice in [
        ToolChoice::Required,
        ToolChoice::Specific {
            function_names: vec![ToolName::new("lookup").unwrap()],
        },
    ] {
        for raw_choice in [
            json!("NONE"),
            json!("AUTO"),
            Value::Null,
            json!(7),
            json!({"type":"REQUIRED"}),
        ] {
            let mut request = CompletionRequest::new("hi").tool(tool("lookup"));
            request.tool_choice = Some(choice.clone());
            request.additional_params = Some(json!({"tool_choice":raw_choice}));
            let error = native()
                .encode(request, Mode::Unary)
                .expect_err("mandatory instructions conflict");
            let error = crate::error::ProviderError::from(error);
            assert_eq!(error.kind(), crate::error::ErrorKind::Request);
        }
        let mut request = CompletionRequest::new("hi").tool(tool("lookup"));
        request.tool_choice = Some(choice);
        request.additional_params = Some(json!({"tool_choice":"REQUIRED"}));
        assert_eq!(
            sent(&native(), request, Mode::Unary).1["tool_choice"],
            "REQUIRED"
        );
    }
    for choice in [ToolChoice::Auto, ToolChoice::None] {
        let mut request = CompletionRequest::new("hi").tool(tool("lookup"));
        request.tool_choice = Some(choice);
        request.additional_params = Some(json!({"tool_choice":"raw precedence"}));
        assert_eq!(
            sent(&native(), request, Mode::Unary).1["tool_choice"],
            "raw precedence"
        );
    }
}

// G37's source condition: no documents means the very same Compatibility
// encoder, option mapper and replay target, with no route-specific shaping.
#[test]
fn default_document_free_requests_equal_compatibility_bytes() {
    for mode in [Mode::Unary, Mode::Streaming] {
        for choice in [
            None,
            Some(ToolChoice::Auto),
            Some(ToolChoice::None),
            Some(ToolChoice::Required),
            Some(ToolChoice::Specific {
                function_names: vec![ToolName::new("lookup").unwrap()],
            }),
        ] {
            let default = CohereConfig::new("key")
                .completion(MODEL)
                .with_strict_tools();
            let explicit = default.clone().with_route(ChatRoute::Compatibility);
            let mut request = CompletionRequest::new("hi")
                .tool(tool("lookup"))
                .temperature(0.2)
                .max_tokens(64);
            request.tool_choice = choice;
            request.additional_params = Some(json!({"seed":7, "metadata":{"kept":[null,1]}}));
            let prepared_default = Completion::prepare(request.clone(), &default.describe())
                .expect("default prepares");
            let prepared_explicit =
                Completion::prepare(request, &explicit.describe()).expect("explicit prepares");
            let automatic = default
                .encode(prepared_default, mode)
                .expect("default encodes");
            let compatibility = explicit
                .encode(prepared_explicit, mode)
                .expect("explicit encodes");
            assert_eq!(automatic.request.method(), compatibility.request.method());
            assert_eq!(automatic.request.uri(), compatibility.request.uri());
            assert_eq!(automatic.request.headers(), compatibility.request.headers());
            let (crate::wire::Body::Bytes(actual), crate::wire::Body::Bytes(expected)) =
                (automatic.request.body(), compatibility.request.body())
            else {
                panic!("chat bodies are bytes");
            };
            assert_eq!(actual, expected);
            assert_eq!(automatic.framing, compatibility.framing);
            assert_eq!(automatic.route, compatibility.route);
            assert_eq!(automatic.request_id_header, compatibility.request_id_header);
            assert_eq!(
                automatic.response_header_prefix,
                compatibility.response_header_prefix
            );
            assert_eq!(
                automatic.relaxed_content_type,
                compatibility.relaxed_content_type
            );
        }
    }
}

#[test]
fn default_document_requests_keep_native_structured_documents() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let default = CohereConfig::new("key").completion(MODEL);
        assert_eq!(default.route, ChatRoute::Auto);
        assert_eq!(ChatRoute::default(), ChatRoute::Auto);
        let mut request = CompletionRequest::new("Which dock?");
        request.documents = documents();
        let (path, body) = sent(&default, request, mode);
        assert_eq!(path, "/v2/chat");
        assert_eq!(
            body["documents"],
            json!([
                {"id":"harbor-1", "data":{"region":"north", "source":"registry", "text":"Beacon amber-73 is at Dock Seven."}},
                {"id":"doc_1", "data":{"text":"Beacon violet-19 is at Dock Three."}},
            ])
        );
        assert_eq!(
            body["messages"],
            json!([{"role":"user", "content":[{"type":"text", "text":"Which dock?"}]}])
        );
        if mode == Mode::Streaming {
            assert_eq!(body["stream"], true);
        } else {
            assert!(body.get("stream").is_none());
        }
    }
}

/// Carried text whose known text is empty and carried reasoning with no
/// visible text send nothing, so a turn holding only them is no message.
#[test]
fn empty_carried_parts_are_skipped() {
    use crate::message::{
        CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText, CarriedTextContent,
    };
    let turn = AssistantMessage {
        content: vec![
            AssistantContent::CarriedText(CarriedText::new(
                CarriedTextContent::Text(String::new()),
                CarriedPayload::new(json!({"type": "output_text"}), Some("foreign".to_owned())),
            )),
            AssistantContent::CarriedReasoning(CarriedReasoning::new(
                vec![CarriedReasoningPart::Encrypted("sealed".to_owned())],
                CarriedPayload::new(json!({"type": "reasoning"}), Some("foreign".to_owned())),
            )),
        ],
        origin: None,
        stop: None,
    };
    assert_eq!(
        native()
            .assistant(&turn, 0, &WireIds::default())
            .expect("empty carried parts encode"),
        None
    );
}

/// A whole provider item cannot be replayed on the native API: the encoder
/// refuses it with the item's issuer and type.
#[test]
fn a_provider_item_is_refused() {
    use crate::message::{ProviderItem, UnreplayableProviderItem};
    use std::error::Error as _;
    let turn = AssistantMessage {
        content: vec![AssistantContent::ProviderItem(ProviderItem::new(
            json!({"type": "future_item", "id": "item_1"}),
            "openai",
        ))],
        origin: None,
        stop: None,
    };
    let error = crate::error::ProviderError::from(
        native()
            .assistant(&turn, 0, &WireIds::default())
            .expect_err("a provider item is refused"),
    );
    assert_eq!(error.kind(), crate::error::ErrorKind::Request);
    assert_eq!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<UnreplayableProviderItem>()),
        Some(&UnreplayableProviderItem {
            wire: "Cohere native chat".to_owned(),
            issuer: Some("openai".to_owned()),
            item_type: Some("future_item".to_owned()),
        })
    );
}

/// A current item that is not a JSON object carries no fields to keep, so
/// the block is sent as its canonical part; a current call item whose
/// `function` is not an object gets one holding the canonical name and
/// arguments.
#[test]
fn malformed_current_items_fall_back_to_canonical_parts() {
    let call = AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("c1"),
        ToolFunction::new(
            ToolName::new("lookup").expect("a tool name"),
            json!({"q": "x"}),
        ),
    ))
    .with_native(json!({"type": "function", "function": "broken", "index": 3}));
    let turn = AssistantMessage {
        content: vec![AssistantContent::text("hi").with_native(json!("raw")), call],
        origin: None,
        stop: None,
    };
    let ids = WireIds::for_target(&[Message::Assistant(turn.clone())], &native(), MODEL)
        .expect("the call history plans");
    assert_eq!(
        native()
            .assistant(&turn, 0, &ids)
            .expect("the turn encodes"),
        Some(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": [{"type": "function", "index": 3, "id": "c1",
                "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}}],
        }))
    );
}

/// An image in an unprepared tool result is left out of the `tool`
/// message, which carries text only.
#[test]
fn an_image_in_a_tool_result_is_left_out() {
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("lookup").expect("a tool name"),
            json!({"q": "rig"}),
        ),
    );
    let mut request = CompletionRequest::new("unused");
    request.chat_history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(call.clone())],
            origin: None,
            stop: None,
        }),
        Message::User {
            content: vec![UserContent::ToolResult(call.result(vec![
                ToolResultContent::text("found"),
                ToolResultContent::Image(Image {
                    data: Source::base64("aGk="),
                    media_type: Some(ImageMediaType::PNG),
                    ..Image::default()
                }),
            ]))],
        },
    ];
    let encoded = native()
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        json_body(&encoded.request)["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"rig\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "found"},
            ]},
        ])
    );
}

/// Carried text sends alone when its known text is not empty, carried
/// reasoning when its visible text is not empty, and a provider item never.
#[test]
fn carried_parts_send_alone_only_with_text() {
    use crate::message::{
        CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText, CarriedTextContent,
        ProviderItem,
    };
    let wire = native();
    let payload = || CarriedPayload::new(json!({"type": "x"}), Some("foreign".to_owned()));
    let text = |content| AssistantContent::CarriedText(CarriedText::new(content, payload()));
    let reasoning =
        |part| AssistantContent::CarriedReasoning(CarriedReasoning::new(vec![part], payload()));
    assert!(wire.sends_alone(&text(CarriedTextContent::Text("hi".to_owned()))));
    assert!(!wire.sends_alone(&text(CarriedTextContent::Text(String::new()))));
    assert!(!wire.sends_alone(&text(CarriedTextContent::Opaque("hi".to_owned()))));
    assert!(wire.sends_alone(&reasoning(CarriedReasoningPart::Summary("hi".to_owned()))));
    assert!(
        !wire.sends_alone(&reasoning(CarriedReasoningPart::Encrypted(
            "sealed".to_owned()
        )))
    );
    assert!(
        !wire.sends_alone(&AssistantContent::ProviderItem(ProviderItem::new(
            json!({"type": "x"}),
            "foreign",
        )))
    );
}

/// What the native wire maps `reasoning`, `cache`, `top_p` and `stop` to,
/// with every other option unset.
fn native_options(
    model: &str,
    reasoning: crate::completion::Reasoning,
    cache: crate::completion::CacheRetention,
    top_p: f64,
    stop: &[String],
) -> crate::completion::options::OptionMap {
    let wire = NativeChat::new(CohereConfig::new("key"), model);
    ReplayTarget::map_options(
        &wire,
        &CompletionRequest::new("hi"),
        crate::completion::options::OptionFields {
            reasoning: Some(&reasoning),
            cache: Some(&cache),
            service_tier: None,
            verbosity: None,
            parallel_tool_calls: None,
            top_p: Some(top_p),
            seed: None,
            stop,
        },
    )
}

/// The native API refuses an effort other than `high`, any prompt cache
/// retention, a `p` outside 0.01 to 0.99 and more than five stop
/// sequences, and honours `cache: None` by sending nothing.
#[test]
fn native_options_refuse_what_cohere_cannot_take() {
    use crate::completion::options::{Mapping, OptionMap};
    use crate::completion::{CacheRetention, Effort, Reasoning as Thinking};
    const THINKS: &str = "command-a-reasoning-08-2025";
    let six: Vec<String> = (1..=6).map(|n| format!("S{n}")).collect();
    let five = &six[..5];
    assert_eq!(
        native_options(
            THINKS,
            Thinking::Effort(Effort::Low),
            CacheRetention::Short,
            1.0,
            &six
        ),
        OptionMap {
            reasoning: Mapping::Unsupported(
                "Cohere takes thinking on or a token budget, not `low`".to_owned()
            ),
            cache: Mapping::Unsupported("Cohere has no prompt cache".to_owned()),
            service_tier: Mapping::Nothing,
            verbosity: Mapping::Nothing,
            parallel_tool_calls: Mapping::Nothing,
            top_p: Mapping::Unsupported("Cohere takes `p` from 0.01 to 0.99".to_owned()),
            seed: Mapping::Nothing,
            stop: Mapping::Unsupported("Cohere takes at most 5 stop sequences".to_owned()),
        }
    );
    assert_eq!(
        native_options(
            THINKS,
            Thinking::Effort(Effort::XHigh),
            CacheRetention::None,
            0.0,
            five
        ),
        OptionMap {
            reasoning: Mapping::Unsupported(
                "Cohere takes thinking on or a token budget, not `xhigh`".to_owned()
            ),
            cache: Mapping::Omit("Cohere does not cache prompts"),
            service_tier: Mapping::Nothing,
            verbosity: Mapping::Nothing,
            parallel_tool_calls: Mapping::Nothing,
            top_p: Mapping::Unsupported("Cohere takes `p` from 0.01 to 0.99".to_owned()),
            seed: Mapping::Nothing,
            stop: Mapping::Send(json!({"stop_sequences": ["S1", "S2", "S3", "S4", "S5"]})),
        }
    );
    assert_eq!(
        native_options(
            THINKS,
            Thinking::Effort(Effort::High),
            CacheRetention::Long,
            0.5,
            five
        )
        .cache,
        Mapping::Unsupported("Cohere has no prompt cache".to_owned())
    );
}

mod g89_pair_request_tests;
