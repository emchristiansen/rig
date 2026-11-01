//! Current-operation ports of the 28 original G97–G105 Bedrock subjects.
//! Constructor, view and policy exceptions are recorded per original row in the receipt.
use super::*;
use crate::completion::{ConverseFrame, PROVIDER_NAME};
use rig_core::completion::{CompletionResponse, FinishReason, OnUnsupported};
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantMessage, CallId, CarriedPayload, CarriedReasoning, CarriedReasoningPart, CarriedText,
    CarriedTextContent, ToolCall, ToolFunction, ToolName, UnrepresentableOpaqueContent,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};

const CLAUDE: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
const NOVA: &str = "amazon.nova-pro-v1:0";
const REDACTED_BLOB: &[u8] = b"\x00\x01opaque-ciphertext\xff";

fn received(model: &str, content: Vec<Value>, usage: Option<Value>) -> CompletionResponse {
    let mut document = json!({
        "output":{"message":{"role":"assistant","content":content}},
        "stopReason":"end_turn",
    });
    if let Some(usage) = usage {
        document["usage"] = usage;
    }
    rig_core::test_utils::history::decode(
        &Converse::new(model),
        Mode::Unary,
        [ConverseFrame::Whole(document)],
    )
    .expect("the original received content decodes")
}

fn signed(text: &str, signature: Option<&str>) -> Value {
    crate::streaming::tests::reasoning(text, signature)
}

fn redacted(bytes: &[u8]) -> Value {
    json!({"reasoningContent":{"redactedContent":BASE64_STANDARD.encode(bytes)}})
}

fn known(parts: Vec<CarriedReasoningPart>) -> AssistantContent {
    AssistantContent::CarriedReasoning(CarriedReasoning::new(
        parts,
        CarriedPayload::new(Value::Null, None),
    ))
}

fn text_part(text: &str, signature: Option<&str>) -> CarriedReasoningPart {
    CarriedReasoningPart::Text {
        text: text.to_owned(),
        signature: signature.map(str::to_owned),
    }
}

fn lower(model: &str, block: &AssistantContent) -> Option<Value> {
    let wire = Converse::new(model);
    assistant(block, &wire, wire.family(model), None).expect("the original known content lowers")
}

fn turn(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage::new(content))
}

fn tool(id: CallId, name: &str, arguments: Value) -> ToolCall {
    ToolCall::new(
        id,
        ToolFunction::new(ToolName::new(name).expect("tool name"), arguments),
    )
}

fn direct(wire: &Converse, request: CompletionRequest) -> Result<Value, ProviderError> {
    let encoded = wire.encode(request, Mode::Unary)?;
    serde_json::to_value(encoded.body).map_err(ProviderError::from)
}

fn prepared(wire: &Converse, request: CompletionRequest) -> Result<Value, ProviderError> {
    let request = Completion::prepare(request, &wire.describe())?;
    direct(wire, request)
}

fn response_history(response: &CompletionResponse) -> Vec<Message> {
    vec![
        Message::user("hi"),
        response.message().expect("a genuinely decoded turn"),
        Message::user("again"),
    ]
}

fn reasoning_blocks(body: &Value) -> Vec<&Value> {
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .filter(|block| block.get("reasoningContent").is_some())
        .collect()
}

#[test]
fn provider_response_ext_response_id_is_none() {
    let out = received("addressed-model", vec![json!({"text":"x"})], None);
    assert_eq!(out.text(), "x");
    assert!(!out.usage.is_reported());
    assert!(out.response_id().is_none());
    assert!(
        out.model().is_none(),
        "model() reports provider metadata, not the addressed model"
    );
    assert_eq!(out.origin.model, "addressed-model");
}

#[test]
fn aws_converse_output_round_trips_through_serde_json_value() {
    let document = json!({
        "output":{"message":{"role":"assistant","content":[{"text":"hello"}]}},
        "stopReason":"end_turn", "usage":{"inputTokens":3,"outputTokens":1,"totalTokens":4},
        "metrics":{"latencyMs":42}, "additionalModelResponseFields":{"stop_sequence":"alpha"},
    });
    let original = rig_core::test_utils::history::decode(
        &Converse::new(""),
        Mode::Unary,
        [ConverseFrame::Whole(document.clone())],
    )
    .expect("the real Converse fold decodes the mirror fixture");
    assert_eq!(original.raw, document);
    assert_eq!(original.raw["metrics"]["latencyMs"], 42);
    assert_eq!(
        original.raw["additionalModelResponseFields"]["stop_sequence"],
        "alpha"
    );
    assert!(original.raw.get("trace").is_none());
    let value = serde_json::to_value(&original).expect("current response serializes");
    let restored: CompletionResponse =
        serde_json::from_value(value.clone()).expect("current response restores");
    assert_eq!(
        serde_json::to_value(&restored).expect("re-serialize"),
        value
    );
    assert_eq!(restored.raw, original.raw);
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(restored.text(), "hello");
    assert_eq!(restored.usage.input_tokens, Some(3));
    assert_eq!(restored.usage.output_tokens, Some(1));
    assert_eq!(restored.usage.total_tokens, Some(4));
    assert_eq!(restored.finish_reason(), Some(FinishReason::Stop));
}

#[test]
fn tool_use_echo_preserves_the_provider_call_identity() {
    // G99: the independent minted-handle leg is unavailable; provider custody stays exact.
    let call = tool(CallId::from_wire("call_abc"), "add", json!({"x":1}));
    assert_eq!(call.function.namespace, None);
    let request = CompletionRequest::from(vec![turn(vec![AssistantContent::ToolCall(call)])]);
    let body = direct(&Converse::new(NOVA), request).expect("the direct call encodes");
    assert_eq!(
        body["messages"][0]["content"][0]["toolUse"],
        json!({"toolUseId":"call_abc","name":"add","input":{"x":1}})
    );
}

#[test]
fn tool_use_echo_uses_the_current_local_alias() {
    // G99: a genuine Local identity uses the existing request-local alias, not minted-handle.
    let call = tool(CallId::from_wire(""), "add", json!({"x":1}));
    assert!(call.id.provider().is_none());
    assert_eq!(call.function.namespace, None);
    let request = CompletionRequest::from(vec![turn(vec![AssistantContent::ToolCall(call)])]);
    let body = direct(&Converse::new(NOVA), request).expect("a local call encodes");
    assert_eq!(
        body["messages"][0]["content"][0]["toolUse"],
        json!({"toolUseId":"tool-0","name":"add","input":{"x":1}})
    );
}

#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    let generated = tool(CallId::from_wire(""), "test", json!({}));
    let real = tool(CallId::from_wire("tool-0"), "test", json!({}));
    let call = |call: &ToolCall| turn(vec![AssistantContent::ToolCall(call.clone())]);
    let result = |call: &ToolCall| Message::User {
        content: vec![UserContent::ToolResult(
            call.result(vec![ToolResultContent::text("done")]),
        )],
    };
    let request = CompletionRequest::from(vec![
        Message::system("system"),
        call(&generated),
        call(&real),
        result(&real),
        Message::assistant("intervening text"),
        result(&generated),
        call(&generated),
        result(&generated),
    ]);
    assert_eq!(request.chat_history.len(), 8);
    let body = direct(&Converse::new(NOVA), request)
        .expect("the full original eight-message history encodes directly");
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for block in body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .flat_map(|message| message["content"].as_array().expect("content"))
    {
        if let Some(call) = block.get("toolUse") {
            assert_eq!(call["name"], "test");
            assert_eq!(call["input"], json!({}));
            calls.push(call["toolUseId"].as_str().expect("call ID").to_owned());
        }
        if let Some(result) = block.get("toolResult") {
            assert_eq!(result["content"], json!([{"text":"done"}]));
            results.push(result["toolUseId"].as_str().expect("result ID").to_owned());
        }
    }
    assert_eq!(calls.len(), 3);
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()]
    );
    assert_eq!(calls[1], "tool-0");
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3
    );
    assert!(
        body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .any(|message| message["content"] == json!([{"text":"intervening text"}]))
    );
    assert_eq!(body["system"], json!([{"text":"system"}]));
}

#[test]
fn aws_reasoning_content_to_assistant_content_without_signature() {
    let response = received(NOVA, vec![signed("This is my reasoning", None)], None);
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("reasoning");
    };
    assert_eq!(reasoning.text, "This is my reasoning");
    assert!(
        reasoning.native.is_none(),
        "unsigned received reasoning has no signature item"
    );
    assert!(!reasoning.redacted);
}

#[test]
fn aws_reasoning_content_to_assistant_content_with_signature() {
    let original = signed(
        "This is my reasoning with signature",
        Some("test_signature_123"),
    );
    let response = received(NOVA, vec![original.clone()], None);
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("reasoning");
    };
    assert_eq!(reasoning.text, "This is my reasoning with signature");
    assert_eq!(
        reasoning
            .native
            .as_ref()
            .expect("received signed item")
            .item,
        original
    );
    assert_eq!(lower(NOVA, &response.choice[0]), Some(original));
}

#[test]
fn rig_reasoning_to_aws_content_block_without_signature() {
    let block = AssistantContent::reasoning("My reasoning content");
    assert_eq!(
        lower(NOVA, &block),
        Some(signed("My reasoning content", None))
    );
    assert_eq!(
        lower(CLAUDE, &block),
        Some(json!({"text":"My reasoning content"}))
    );
}

#[test]
fn rig_reasoning_to_aws_content_block_with_signature() {
    // G100 constructor exception: signature custody originates on the received wire.
    let original = signed("My reasoning content", Some("sig_abc_123"));
    let response = received(NOVA, vec![original.clone()], None);
    assert_eq!(lower(NOVA, &response.choice[0]), Some(original));
}

#[test]
fn rig_reasoning_with_multiple_strings_to_aws_content_block() {
    let block = known(vec![
        text_part("First part", None),
        text_part(" Second part", None),
        text_part(" Third part", None),
    ]);
    assert_eq!(
        lower(NOVA, &block),
        Some(signed("First part\n Second part\n Third part", None))
    );
}

#[test]
fn rig_reasoning_with_empty_text_and_signature_is_converted() {
    let original = signed("", Some("sig_empty_text"));
    let response = received(NOVA, vec![original.clone()], None);
    assert_eq!(response.choice.len(), 1);
    assert_eq!(lower(NOVA, &response.choice[0]), Some(original));
}

#[test]
fn empty_unsigned_reasoning_is_omitted_without_error() {
    // G101: the old Provider error becomes current omission, preserving the empty input.
    assert_eq!(lower(NOVA, &AssistantContent::reasoning("")), None);
}

#[test]
fn known_signed_parts_project_joined_text_without_signatures() {
    // G101: both exact caller signature inputs remain; only their visible text is emitted.
    let parts = vec![
        text_part("part one", Some("sig_1")),
        text_part("part two", Some("sig_2")),
    ];
    let block = known(parts.clone());
    let AssistantContent::CarriedReasoning(input) = &block else {
        panic!("carried input");
    };
    assert_eq!(input.content, parts);
    assert_eq!(
        lower(NOVA, &block),
        Some(signed("part one\npart two", None))
    );
}

#[test]
fn aws_redacted_reasoning_content_becomes_redacted_reasoning_not_an_error() {
    let original = redacted(REDACTED_BLOB);
    let response = received(NOVA, vec![original.clone()], None);
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("redacted reasoning");
    };
    assert!(reasoning.redacted);
    assert_eq!(reasoning.text, "");
    assert_eq!(
        reasoning
            .native
            .as_ref()
            .expect("received redacted item")
            .item,
        original
    );
}

#[test]
fn caller_redacted_parts_omit_while_received_redacted_parts_replay() {
    // G100: caller-carried redacted data omits; genuinely received Native replays separately.
    let caller = known(vec![CarriedReasoningPart::Redacted {
        data: BASE64_STANDARD.encode(REDACTED_BLOB),
    }]);
    assert_eq!(lower(NOVA, &caller), None);
    let original = redacted(REDACTED_BLOB);
    let response = received(NOVA, vec![original.clone()], None);
    assert_eq!(lower(NOVA, &response.choice[0]), Some(original));
}

#[test]
fn redacted_reasoning_round_trips_byte_for_byte() {
    let response = received(NOVA, vec![redacted(REDACTED_BLOB)], None);
    let outbound = lower(NOVA, &response.choice[0]).expect("received redacted Native replays");
    let bytes = BASE64_STANDARD
        .decode(
            outbound["reasoningContent"]["redactedContent"]
                .as_str()
                .expect("blob"),
        )
        .expect("base64");
    assert_eq!(bytes, REDACTED_BLOB);
}

#[test]
fn foreign_encrypted_reasoning_is_dropped_never_shipped_as_redacted() {
    let block = known(vec![CarriedReasoningPart::Encrypted(
        BASE64_STANDARD.encode(REDACTED_BLOB),
    )]);
    assert_eq!(lower(NOVA, &block), None);
}

#[test]
fn multiple_caller_redacted_payloads_are_omitted() {
    // G101: both original caller blobs omit rather than acquiring Native replay authority.
    let block = known(vec![
        CarriedReasoningPart::Redacted {
            data: BASE64_STANDARD.encode(REDACTED_BLOB),
        },
        CarriedReasoningPart::Redacted {
            data: BASE64_STANDARD.encode(b"second"),
        },
    ]);
    assert_eq!(lower(NOVA, &block), None);
}

#[test]
fn foreign_encrypted_reasoning_never_fails_the_request() {
    assert_eq!(
        lower(
            NOVA,
            &known(vec![CarriedReasoningPart::Encrypted(
                "gAAAA-non_base64-token_".to_owned()
            )])
        ),
        None
    );
}

#[test]
fn non_base64_redacted_reasoning_is_dropped_rather_than_sent_corrupt() {
    // G101: omission proves no base64 validation, because all caller Redacted parts omit.
    assert_eq!(
        lower(
            NOVA,
            &known(vec![CarriedReasoningPart::Redacted {
                data: "not*valid*base64".to_owned()
            }])
        ),
        None
    );
}

#[test]
fn bedrock_reasoning_keeps_current_model_and_context_custody() {
    // G102: model/context custody replaces issuer-only filtering, with all original names retained.
    for model in [
        "anthropic.claude-sonnet-4-6-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-7-v1:0",
        "amazon.nova-pro-v1:0",
        "deepseek.r1-v1:0",
    ] {
        let blocks = ["anthropic", "gcp.gemini", PROVIDER_NAME]
            .into_iter()
            .map(|issuer| {
                AssistantContent::CarriedReasoning(CarriedReasoning::new(
                    vec![text_part("thinking", Some(&format!("{issuer}-signature")))],
                    CarriedPayload::new(Value::Null, Some(issuer.to_owned())),
                ))
            })
            .chain([AssistantContent::text("hello")])
            .collect::<Vec<_>>();
        let request = CompletionRequest::from(vec![Message::user("hi"), turn(blocks)]);
        let body = direct(&Converse::new(model), request)
            .expect("all known caller inputs encode by current family");
        let content = body["messages"][1]["content"]
            .as_array()
            .expect("assistant content");
        assert_eq!(content.len(), 4);
        for block in &content[..3] {
            assert_eq!(
                block,
                &if Converse::new(model).family(model) == Family::Claude {
                    json!({"text":"thinking"})
                } else {
                    signed("thinking", None)
                }
            );
        }
        assert_eq!(content[3], json!({"text":"hello"}));
        let response = received(model, vec![signed("thinking", Some("sig"))], None);
        assert_eq!(response.origin.provider, PROVIDER_NAME);
        assert_eq!(response.origin.model, model);
    }
}

#[test]
fn decoded_bedrock_reasoning_records_the_models_issuer_and_replays_to_it() {
    let content = vec![
        signed("thinking", Some("sig")),
        redacted(b"redacted"),
        json!({"text":"hello"}),
    ];
    let response = received(CLAUDE, content.clone(), None);
    let nova = received(NOVA, content, None);
    assert_eq!(response.origin.provider, PROVIDER_NAME);
    assert_eq!(nova.origin.provider, PROVIDER_NAME);
    for received in [&response, &nova] {
        assert_eq!(
            received
                .choice
                .iter()
                .filter(|block| matches!(block, AssistantContent::Reasoning(_)))
                .count(),
            2
        );
    }
    let history = response_history(&response);
    let body = prepared(
        &Converse::new(CLAUDE),
        CompletionRequest::from(history.clone()),
    )
    .expect("same-model/context genuine Native replays");
    assert_eq!(reasoning_blocks(&body).len(), 2);
    assert_eq!(
        reasoning_blocks(&body),
        [&signed("thinking", Some("sig")), &redacted(b"redacted")]
    );
    let cross = prepared(&Converse::new(NOVA), CompletionRequest::from(history))
        .expect("cross-model known projection encodes");
    assert!(reasoning_blocks(&cross).is_empty());
    assert_eq!(
        cross["messages"][1]["content"],
        json!([{"text":"thinking"},{"text":"hello"}])
    );
    assert!(
        !cross
            .to_string()
            .contains(&BASE64_STANDARD.encode(b"redacted"))
    );
    assert!(!cross.to_string().contains("signature"));
}

#[test]
fn opaque_responses_parts_are_refused_before_request_content_is_discarded() {
    let raw = json!({"type":"future_part","payload":{"list":[1,2]}});
    let payload = CarriedPayload::new(raw.clone(), Some("openai".to_owned()));
    let text = AssistantContent::CarriedText(CarriedText::new(
        CarriedTextContent::Opaque(String::new()),
        CarriedPayload::new(raw, None),
    ));
    let summary = AssistantContent::CarriedReasoning(CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueSummary(payload.clone())],
        payload.clone(),
    ));
    let mut reasoning = CarriedReasoning::new(
        vec![CarriedReasoningPart::OpaqueContent(payload.clone())],
        payload,
    );
    reasoning.id = Some("rs_1".to_owned());
    for part in [text, summary, AssistantContent::CarriedReasoning(reasoning)] {
        let request =
            CompletionRequest::from(vec![turn(vec![AssistantContent::text("known"), part])]);
        let wire = Converse::new("anthropic.claude");
        let error = direct(&wire, request.clone()).expect_err("opaque content refused directly");
        let ProviderError::Request(reason) = &error else {
            panic!("Request refusal: {error:?}");
        };
        assert_eq!(
            reason.downcast_ref::<UnrepresentableOpaqueContent>(),
            Some(&UnrepresentableOpaqueContent::new(PROVIDER_NAME))
        );
        assert!(error.to_string().contains("opaque"));
        let error = Completion::prepare(request, &wire.describe())
            .expect_err("opaque content refused before shaping");
        assert!(matches!(error, ProviderError::Request(_)));
        assert!(error.to_string().contains("opaque"));
    }
}

#[test]
fn test_system_prompt_appends_cache_point_when_prompt_caching_enabled() {
    let mut request = CompletionRequest::new("test").cache(CacheRetention::Short);
    request
        .chat_history
        .insert(0, Message::system("System prompt"));
    assert!(request.model.is_none());
    let body = direct(&Converse::new(CLAUDE), request).expect("explicit Short caches on Claude");
    assert_eq!(body["system"].as_array().map(Vec::len), Some(2));
    assert_eq!(body["system"][0], json!({"text":"System prompt"}));
    assert!(body["system"][1].get("cachePoint").is_some());
}

#[test]
fn test_messages_append_cache_point_when_prompt_caching_enabled() {
    let request = CompletionRequest::new("test").cache(CacheRetention::Short);
    assert!(request.model.is_none());
    let body = direct(&Converse::new(CLAUDE), request).expect("explicit Short caches on Claude");
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(
        body["messages"][0]["content"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(body["messages"][0]["content"][0], json!({"text":"test"}));
    assert!(
        body["messages"][0]["content"][1]
            .get("cachePoint")
            .is_some()
    );
}

#[test]
fn encoded_reasoning_refuses_or_skips_cache_under_the_explicit_policy() {
    // G104: genuine same-model signature plus Short/Error versus Short/Ignore, not Boolean parity.
    let response = received(CLAUDE, vec![signed("thinking", Some("sig"))], None);
    let history = vec![
        Message::user("user prompt"),
        response.message().expect("actual signed turn"),
        Message::user("follow up"),
    ];
    let request = CompletionRequest::from(history).cache(CacheRetention::Short);
    let wire = Converse::new(CLAUDE);
    let error = direct(&wire, request.clone())
        .expect_err("default Error refuses cache after encoded reasoning");
    assert!(
        matches!(error, ProviderError::UnsupportedOption(ref option) if option.option == "cache")
    );
    let body = direct(&wire, request.on_unsupported(OnUnsupported::Ignore))
        .expect("Ignore skips only the unsupported message checkpoint");
    assert!(
        body.get("system").is_none(),
        "the original system-only path is absent independently"
    );
    assert_eq!(reasoning_blocks(&body), [&signed("thinking", Some("sig"))]);
    assert_eq!(
        body["messages"][0]["content"],
        json!([{"text":"user prompt"}])
    );
    assert_eq!(
        body["messages"][2]["content"],
        json!([{"text":"follow up"}])
    );
    assert!(
        !body["messages"]
            .as_array()
            .expect("messages")
            .last()
            .expect("last message")["content"]
            .as_array()
            .expect("content")
            .iter()
            .any(|block| block.get("cachePoint").is_some())
    );
}

fn document_request(prompt_document: bool) -> CompletionRequest {
    let mut request = CompletionRequest::new("test");
    request.chat_history = vec![
        Message::system("Answer with the exact token from the document only."),
        Message::assistant("Acknowledged."),
    ];
    if prompt_document {
        request.chat_history.push(Message::User {
            content: vec![
                UserContent::document_text("A repeated attachment.", Some(DocumentMediaType::TXT)),
                UserContent::document_text("A repeated attachment.", Some(DocumentMediaType::TXT)),
                UserContent::text("According to the document, what is the ordering token?"),
            ],
        });
    } else {
        request.chat_history.push(Message::user(
            "According to the document, what is the ordering token?",
        ));
    }
    request.documents = vec![rig_core::completion::Document {
        id: "ordering-note".to_owned(),
        text: "The ordering token is violet-needle.".to_owned(),
        additional_props: Default::default(),
    }];
    request
}

#[test]
fn documents_are_prepended_before_history() {
    let body = prepared(&Converse::new(CLAUDE), document_request(false))
        .expect("original text document request encodes");
    assert!(body["system"].to_string().contains("exact token"));
    let messages = body["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert!(
        messages[0]["content"]
            .as_array()
            .expect("content")
            .iter()
            .any(|block| block.get("document").is_some())
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
}

#[test]
fn document_names_are_deterministic_and_unique_within_a_request() {
    let wire = Converse::new(CLAUDE);
    let first = prepared(&wire, document_request(true)).expect("documents encode");
    let second = prepared(&wire, document_request(true)).expect("documents encode again");
    assert_eq!(first, second);
    let names = first["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .flat_map(|message| message["content"].as_array().expect("content"))
        .filter_map(|block| block.pointer("/document/name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(names.len(), 3);
    assert_eq!(
        names.iter().collect::<std::collections::HashSet<_>>().len(),
        3
    );
    assert!(names.iter().all(|name| name.starts_with("document-")));
    assert_eq!(names[2], format!("{}-2", names[1]));
}
