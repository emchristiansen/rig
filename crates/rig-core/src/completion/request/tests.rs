use super::{CompletionResponse, FinishReason, Usage};
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::{http_client, provider_response};

mod message_content_validation {
    use super::super::CompletionRequest;
    use crate::message::{AssistantContent, AssistantMessage, Message, UserContent};

    fn request(chat_history: Vec<Message>) -> CompletionRequest {
        CompletionRequest::from(chat_history)
    }

    #[test]
    fn a_populated_history_passes() {
        let request = request(vec![Message::user("hello")]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn an_empty_history_is_rejected() {
        // `chat_history` was non-empty by construction until the container
        // was removed; the field is public, so this is now constructible.
        let error = request(Vec::new())
            .validate_message_content()
            .expect_err("an empty history must not reach a provider");
        assert!(
            error.to_string().contains("empty chat history"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn an_empty_user_message_is_rejected_by_index_and_role() {
        let error = request(vec![
            Message::user("hello"),
            Message::User {
                content: Vec::new(),
            },
        ])
        .validate_message_content()
        .expect_err("an empty user message must not reach a provider");
        let message = error.to_string();
        assert!(message.contains("user message at index 1"), "{message}");
    }

    #[test]
    fn an_empty_assistant_message_is_rejected_by_index_and_role() {
        let error = request(vec![Message::Assistant(AssistantMessage::new(Vec::new()))])
            .validate_message_content()
            .expect_err("an empty assistant message must not reach a provider");
        let message = error.to_string();
        assert!(
            message.contains("assistant message at index 0"),
            "{message}"
        );
    }

    #[test]
    fn an_empty_system_message_is_not_rejected() {
        // System content is a `String` and always has been, so the removed
        // container never constrained it. Rejecting an empty one would be a
        // new restriction, and would break a history carrying a
        // conditionally built preamble that resolved to `""`.
        let request = request(vec![
            Message::System {
                content: String::new(),
            },
            Message::user("hello"),
        ]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn a_block_less_tool_result_is_rejected_naming_the_tool() {
        use crate::message::{CallId, ToolName, ToolResult, ToolResultContent};
        // `ToolResult::content` was non-empty by construction until the
        // container was removed; the message around it has one item, so the
        // message-level check alone would let this reach the wire as
        // `"content": []`.
        let error = request(vec![
            Message::user("hello"),
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: CallId::from_wire("call_1"),
                    name: ToolName::new("lookup").expect("tool name"),
                    answers: crate::message::AnsweredToolCall::Function,
                    content: Vec::<ToolResultContent>::new(),
                    is_error: false,
                })],
            },
        ])
        .validate_message_content()
        .expect_err("a block-less tool result must not reach a provider");
        let message = error.to_string();
        assert!(message.contains("`lookup`"), "{message}");
        assert!(message.contains("index 0"), "{message}");
        assert!(message.contains("user message at index 1"), "{message}");
    }

    #[test]
    fn a_tool_result_with_one_empty_string_block_is_accepted() {
        use crate::message::{CallId, ToolName, ToolResult, ToolResultContent};
        // The guard is on the cardinality of the block list, not on the
        // blocks' content. A tool that legitimately returned an empty
        // string produces one block and must still send — this pins the
        // "no blocks" / "no content" distinction so a future tightening
        // pass cannot collapse it.
        let request = request(vec![Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("call_1"),
                name: ToolName::new("lookup").expect("tool name"),
                answers: crate::message::AnsweredToolCall::Function,
                content: vec![ToolResultContent::text("")],
                is_error: false,
            })],
        }]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn the_legacy_fabricated_sentinel_still_passes() {
        // Histories persisted before message content became a `Vec` encode a
        // content-less assistant turn as a single empty text part. That is a
        // one-element list, so it validates — the rule is block count, not
        // block content — and, being caller-supplied history, it is never
        // filtered: it goes to the wire as-is (where some providers reject
        // it). Callers migrating pre-`Vec` histories drop such turns
        // themselves; see MIGRATING.
        let request = request(vec![
            Message::user("hello"),
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::text("")])),
            Message::User {
                content: vec![UserContent::text("and again")],
            },
        ]);
        assert!(request.validate_message_content().is_ok());
    }
}

/// An empty conversation or message parses, and the request boundary
/// rejects it by role and index. Tool-result block cardinality is checked
/// separately by the carried validation tests.
mod empty_lists_parse_and_are_rejected_when_sent {
    use crate::message::Message;
    use crate::test_utils::{MockCompletionModel, MockTurn};
    use serde_json::json;

    #[test]
    fn an_empty_history_parses() {
        let mut request =
            serde_json::to_value(super::CompletionRequest::new("hello")).expect("serializes");
        request["chat_history"] = json!([]);
        let request = serde_json::from_value::<super::CompletionRequest>(request)
            .expect("an empty history parses");
        assert!(request.chat_history.is_empty());
    }

    /// Deserializing `[]` is not where the rule lives: the message reads and
    /// writes the same JSON, and sending it is what fails.
    #[tokio::test]
    async fn an_empty_content_list_round_trips_and_is_rejected_when_sent() {
        let json = json!({"role": "user", "content": []});
        let message = serde_json::from_value::<Message>(json.clone()).expect("parses");
        assert_eq!(serde_json::to_value(&message).expect("serializes"), json);

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let error = model
            .call(super::CompletionRequest::new(message))
            .await
            .expect_err("sending it is rejected");
        assert!(
            error
                .to_string()
                .contains("user message at index 0 has no content"),
            "{error}"
        );
        assert_eq!(model.request_count(), 0);
    }
}

mod carried_document_requests {
    use super::*;

    fn document_part(id: &str, text: &str) -> UserContent {
        UserContent::document_text(
            test_document(id, text).to_string(),
            Some(DocumentMediaType::TXT),
        )
    }

    fn is_document_message(message: &Message, expected_id: &str) -> bool {
        let Message::User { content } = message else {
            return false;
        };
        content.iter().any(|content| {
            matches!(
                content,
                UserContent::Document(document)
                    if matches!(&document.data, crate::message::DocumentData::Text(text)
                        if text.contains(&format!("<file id: {expected_id}>")))
            )
        })
    }

    #[test]
    fn message_telemetry_includes_normalized_documents() {
        let request = CompletionRequest::new("prompt")
            .preamble("system")
            .message(Message::user("history"))
            .document(test_document("doc1", "static context secret"));
        let messages = request.messages_for_telemetry();
        // Upstream joins document blocks to the first existing user message.
        assert_eq!(messages.len(), 3);
        assert_eq!(
            messages,
            [
                Message::system("system"),
                Message::User {
                    content: vec![
                        document_part("doc1", "static context secret"),
                        UserContent::text("history"),
                    ]
                },
                Message::user("prompt"),
            ]
        );
        assert!(is_document_message(&messages[1], "doc1"));
        assert_eq!(messages, request.chat_history_with_documents());
    }

    #[test]
    fn test_document_display_without_metadata() {
        let doc = test_document("123", "This is a test document.");
        assert_eq!(
            format!("{doc}"),
            "<file id: 123>\nThis is a test document.\n</file>\n"
        );
    }

    #[test]
    fn test_normalize_documents_with_documents() {
        let request = CompletionRequest::new("What is the capital of France?").documents([
            test_document("doc1", "Document 1 text."),
            test_document("doc2", "Document 2 text."),
        ]);
        assert_eq!(
            request.normalized_documents(),
            Some(Message::User {
                content: vec![
                    UserContent::document_text(
                        "<file id: doc1>\nDocument 1 text.\n</file>\n",
                        Some(DocumentMediaType::TXT)
                    ),
                    UserContent::document_text(
                        "<file id: doc2>\nDocument 2 text.\n</file>\n",
                        Some(DocumentMediaType::TXT)
                    ),
                ]
            })
        );
    }

    #[test]
    fn test_normalize_documents_without_documents() {
        assert_eq!(
            CompletionRequest::new("What is the capital of France?").normalized_documents(),
            None
        );
    }

    #[test]
    fn preamble_builder_funnels_to_system_message() {
        let request = CompletionRequest::new(Message::user("Prompt"))
            .preamble("System prompt")
            .message(Message::user("History"));
        assert_eq!(
            request.chat_history,
            [
                Message::system("System prompt"),
                Message::user("History"),
                Message::user("Prompt"),
            ]
        );
    }

    #[test]
    fn build_places_documents_after_preamble_system_message() {
        let request = CompletionRequest::new(Message::user("Prompt"))
            .preamble("System prompt")
            .document(test_document("doc1", "Document text."));
        assert_eq!(request.documents.len(), 1);
        let history = request.chat_history_with_documents();
        assert_eq!(history.len(), 2);
        assert_eq!(
            history,
            [
                Message::system("System prompt"),
                Message::User {
                    content: vec![
                        document_part("doc1", "Document text."),
                        UserContent::text("Prompt"),
                    ]
                },
            ]
        );
        assert!(is_document_message(&history[1], "doc1"));
    }

    #[test]
    fn build_places_documents_after_leading_system_messages_before_prior_history() {
        let request = CompletionRequest::new(Message::user("Prompt"))
            .message(Message::system("System one"))
            .message(Message::system("System two"))
            .message(Message::user("Earlier user turn"))
            .message(Message::assistant("Earlier assistant turn"))
            .document(test_document("doc1", "Document text."));
        let history = request.chat_history_with_documents();
        assert_eq!(history.len(), 5);
        assert_eq!(
            history,
            [
                Message::system("System one"),
                Message::system("System two"),
                Message::User {
                    content: vec![
                        document_part("doc1", "Document text."),
                        UserContent::text("Earlier user turn")
                    ]
                },
                Message::assistant("Earlier assistant turn"),
                Message::user("Prompt"),
            ]
        );
        assert!(is_document_message(&history[2], "doc1"));
    }

    #[test]
    fn build_without_documents_keeps_message_order_unchanged() {
        let request = CompletionRequest::new(Message::user("Prompt"))
            .message(Message::system("System prompt"))
            .message(Message::user("Earlier user turn"));
        assert_eq!(
            request.chat_history,
            [
                Message::system("System prompt"),
                Message::user("Earlier user turn"),
                Message::user("Prompt"),
            ]
        );
    }

    #[test]
    fn chat_history_with_documents_places_documents_after_leading_system_messages() {
        let request = CompletionRequest::from(vec![
            Message::system("System prompt"),
            Message::assistant("Earlier assistant turn"),
            Message::user("Earlier user turn"),
            Message::user("Prompt"),
        ])
        .document(test_document("doc1", "Document text."));
        assert_eq!(request.documents.len(), 1);
        let history = request.chat_history_with_documents();
        assert_eq!(history.len(), 5);
        assert_eq!(
            history,
            [
                Message::system("System prompt"),
                Message::User {
                    content: vec![document_part("doc1", "Document text.")]
                },
                Message::assistant("Earlier assistant turn"),
                Message::user("Earlier user turn"),
                Message::user("Prompt"),
            ]
        );
        assert!(is_document_message(&history[1], "doc1"));
    }

    #[test]
    fn chat_history_with_documents_places_documents_before_mid_conversation_system_messages() {
        let request = CompletionRequest::from(vec![
            Message::system("Leading system prompt"),
            Message::assistant("Earlier assistant turn"),
            Message::system("Mid-conversation instruction"),
            Message::user("Prompt"),
        ])
        .document(test_document("doc1", "Document text."));
        let history = request.chat_history_with_documents();
        assert_eq!(history.len(), 5);
        assert_eq!(
            history,
            [
                Message::system("Leading system prompt"),
                Message::User {
                    content: vec![document_part("doc1", "Document text.")]
                },
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("Prompt"),
            ]
        );
        assert!(is_document_message(&history[1], "doc1"));
    }

    #[test]
    fn chat_history_with_documents_does_not_duplicate_documents() {
        let request = CompletionRequest::from(vec![
            Message::system("System prompt"),
            Message::user("Earlier user turn"),
            Message::assistant("Earlier assistant turn"),
            Message::user("Prompt"),
        ])
        .document(test_document("doc1", "Document text."));
        let history = request.chat_history_with_documents();
        assert_eq!(
            history
                .iter()
                .filter(|message| is_document_message(message, "doc1"))
                .count(),
            1
        );
        let count = history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter(|content| matches!(content, UserContent::Document(_)))
            .count();
        assert_eq!(count, 1);
        assert_eq!(request.chat_history_with_documents(), history);
    }
}

fn tool_call_choice() -> Vec<AssistantContent> {
    vec![AssistantContent::tool_call(
        "call_1",
        crate::message::ToolName::new("lookup").expect("tool name"),
        serde_json::json!({"query": "rig"}),
    )]
}

#[test]
fn normalized_response_round_trips_through_serde() {
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(2),
                total_tokens: Some(5),
                cached_input_tokens: Some(1),
                cache_creation_input_tokens: Some(0),
                tool_use_prompt_tokens: Some(0),
                reasoning_tokens: Some(1),
                cost: None,
            },
            crate::message::Origin::new("test.api", "example", ""),
            serde_json::json!({}),
        )
        .with_message_id(ProviderMessageId::new("msg_123").expect("message id"))
        .with_finish_reason(FinishReason::Stop);
        response.origin.response_model = Some("provider-model-v2".into());
        response
    };

    let encoded = serde_json::to_value(&response).expect("serialize response");
    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded.clone()).expect("deserialize");

    assert_eq!(
        serde_json::to_value(decoded).expect("re-serialize"),
        encoded
    );
}

/// Serde must not be a back door around `reconcile_with_output`: a
/// persisted `"stop"` next to a tool-call choice deserializes as
/// `ToolCalls`, exactly as if it had gone through the setter.
#[test]
fn deserializing_stop_with_a_tool_call_reconciles_to_tool_calls() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    ))
    .expect("serialize response");
    encoded["finish_reason"] = serde_json::json!("stop");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Serde must not be a back door around the empty-string filtering either:
/// a persisted `""` identifier deserializes as `None`.
#[test]
fn deserializing_empty_identifiers_yields_none() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    ))
    .expect("serialize response");
    encoded["message_id"] = serde_json::json!("");
    encoded["origin"]["response_id"] = serde_json::json!("");
    encoded["origin"]["response_model"] = serde_json::json!("");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.message_id, None);
    assert_eq!(decoded.response_id(), None);
    assert_eq!(decoded.model(), None);
}

#[test]
fn unknown_finish_reason_survives_a_serde_round_trip_verbatim() {
    let reason = FinishReason::Other("provider_specific_stop".to_owned());
    let encoded = serde_json::to_string(&reason).expect("serialize");
    let decoded = serde_json::from_str::<FinishReason>(&encoded).expect("deserialize");

    assert_eq!(decoded, reason);
}

#[test]
fn stop_with_a_tool_call_reconciles_to_tool_calls() {
    let response = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    )
    .with_finish_reason(FinishReason::Stop);

    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

/// The `Option` setter is what provider conversions actually reach for, so
/// it must reconcile identically — a provider holding an `Option` must not
/// have to choose between ergonomics and correctness.
#[test]
fn optional_setter_reconciles_exactly_like_the_plain_setter() {
    let via_option = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    )
    .with_optional_finish_reason(Some(FinishReason::Stop));
    let via_plain = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    )
    .with_finish_reason(FinishReason::Stop);

    assert_eq!(via_option.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(via_option.finish_reason(), via_plain.finish_reason());
}

#[test]
fn reconciliation_only_upgrades_a_natural_stop() {
    // A truncated tool call is still a truncation; a filtered one is still
    // filtered. Overriding either would lose why the turn actually ended.
    for reason in [
        FinishReason::Length,
        FinishReason::ContentFilter,
        FinishReason::Other("provider_specific".to_owned()),
    ] {
        let response = CompletionResponse::new(
            tool_call_choice(),
            Usage::default(),
            crate::message::Origin::new("test.api", "example", ""),
            serde_json::json!({}),
        )
        .with_finish_reason(reason.clone());

        assert_eq!(response.finish_reason(), Some(reason));
    }
}

#[test]
fn reconciliation_leaves_a_stop_without_tool_calls_alone() {
    let response = CompletionResponse::new(
        vec![AssistantContent::text("done")],
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    )
    .with_finish_reason(FinishReason::Stop);

    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
}

#[test]
fn provider_capabilities_are_externally_configurable_from_default() {
    let capabilities = ProviderCapabilities::default().with_native_output_tool_composition(true);

    assert!(capabilities.composes_native_output_with_tools);
    assert!(!ProviderCapabilities::new().composes_native_output_with_tools);
    assert_eq!(ProviderCapabilities::new(), ProviderCapabilities::default());
}

#[test]
fn usage_is_reported_when_any_counter_is_present() {
    use super::Usage;

    assert!(!Usage::default().is_reported());

    let zero = Usage {
        reasoning_tokens: Some(0),
        ..Usage::default()
    };
    assert!(zero.is_reported(), "a reported zero is still a report");
}

#[test]
fn usage_sum_keeps_a_reported_counter_when_the_other_side_is_absent() {
    use super::Usage;

    let reported = Usage {
        input_tokens: Some(3),
        output_tokens: Some(0),
        ..Usage::default()
    };
    let sum = Usage::default() + reported + reported;
    assert_eq!(sum.input_tokens, Some(6));
    assert_eq!(sum.output_tokens, Some(0));
    assert_eq!(sum.total_tokens, None);
}

use super::*;

#[test]
fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
    let default_request = CompletionRequest::new("completion prompt");
    assert!(!default_request.record_telemetry_content);

    let default_json = serde_json::to_value(&default_request).expect("serialize request");
    assert!(
        default_json.get("record_telemetry_content").is_none(),
        "safe default should not serialize the telemetry opt-in field"
    );
    let default_roundtrip: CompletionRequest =
        serde_json::from_value(default_json).expect("deserialize default request");
    assert!(!default_roundtrip.record_telemetry_content);

    let opt_in_request = CompletionRequest::new("completion prompt").record_content_telemetry(true);
    assert!(opt_in_request.record_telemetry_content);

    let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
    assert!(
        opt_in_json.get("record_telemetry_content").is_none(),
        "local telemetry policy must not be serialized into provider requests"
    );
    let without_field: CompletionRequest =
        serde_json::from_value(opt_in_json).expect("deserialize a request without the field");
    assert!(
        !without_field.record_telemetry_content,
        "missing field should deserialize to the safe default"
    );
}

/// The deserialization mirror carries `raw`: a response with a captured
/// payload survives serialize → deserialize with the payload intact, and a
/// response written without the field is refused rather than loaded with
/// `raw` invented.
#[test]
fn normalized_response_raw_round_trips_through_serde_mirror() {
    let payload = serde_json::json!({
        "id": "chatcmpl-1",
        "system_fingerprint": "fp_abc",
        "choices": [{"finish_reason": "stop"}]
    });
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::default(),
            crate::message::Origin::new("test.api", "example", ""),
            payload.clone(),
        );
        response.origin.response_id = Some("chatcmpl-1".into());
        response
    };

    let encoded = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(encoded["raw"], payload);
    let decoded: CompletionResponse =
        serde_json::from_value(encoded.clone()).expect("deserialize response");
    assert_eq!(decoded.raw, payload);
    assert_eq!(decoded.response_id(), Some("chatcmpl-1"));
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );

    let without_raw = serde_json::json!({
        "choice": [{"type": "text", "text": "hello"}],
        "usage": serde_json::to_value(Usage::default()).unwrap(),
        "origin": {"api": "test.api", "provider": "example", "model": ""}
    });
    let error = serde_json::from_value::<CompletionResponse>(without_raw)
        .expect_err("a response without `raw` is refused");
    assert!(error.to_string().contains("raw"), "{error}");
}

fn test_document(id: &str, text: &str) -> Document {
    Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

#[test]
fn test_document_display_with_metadata() {
    let mut additional_props = HashMap::new();
    additional_props.insert("author".to_string(), "John Doe".to_string());
    additional_props.insert("length".to_string(), "42".to_string());

    let doc = Document {
        id: "123".to_string(),
        text: "This is a test document.".to_string(),
        additional_props,
    };

    let expected = concat!(
        "<file id: 123>\n",
        "<metadata author: \"John Doe\" length: \"42\" />\n",
        "This is a test document.\n",
        "</file>\n"
    );
    assert_eq!(format!("{doc}"), expected);
}

#[test]
fn documents_join_the_first_user_message_so_roles_alternate() {
    let request = CompletionRequest::new(Message::user("Prompt"))
        .message(Message::system("System prompt"))
        .message(Message::user("Earlier user turn"))
        .document(test_document("doc1", "Document text."));
    let history = request.chat_history_with_documents();
    let roles: Vec<&str> = history
        .iter()
        .map(|message| match message {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant(_) => "assistant",
        })
        .collect();
    assert_eq!(roles, ["system", "user", "user"]);
    let Message::User { content } = &history[1] else {
        panic!("the first user message: {history:?}");
    };
    assert!(matches!(content.first(), Some(UserContent::Document(_))));
    assert!(
        content.iter().any(
            |part| matches!(part, UserContent::Text(text) if text.text == "Earlier user turn")
        )
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"code":"rate_limit","message":"slow down"}}"#;
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("fixture body should parse as valid JSON"),
        Some(serde_json::json!({
            "error": {
                "code": "rate_limit",
                "message": "slow down"
            }
        }))
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("provider exploded".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("provider exploded"));
    assert_eq!(error.provider_response_status(), None);
    assert!(error.provider_response_json().is_err());
}

#[test]
fn completion_error_provider_response_helpers_with_http_non_success_body_and_status() {
    let body = r#"{"error":{"type":"invalid_request","message":"bad request"}}"#;
    let error = ProviderError::from_transport_error(http_client::Error::non_success_with_details(
        http::StatusCode::BAD_REQUEST,
        http::HeaderMap::new(),
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": {
                "type": "invalid_request",
                "message": "bad request"
            }
        }))
    );
}

mod additional_params_precedence {
    use super::super::shadowed_typed_fields;
    use serde_json::json;

    /// A passthrough key names a typed field only when that field is set too;
    /// an unset typed field is not shadowed, and a non-object passthrough
    /// shadows nothing.
    #[test]
    fn shadowed_keys_are_the_intersection_with_set_typed_fields() {
        let params = json!({"temperature": 0.9, "top_p": 0.5, "max_tokens": 10});
        let shadowed = shadowed_typed_fields(
            Some(&params),
            &[
                ("temperature", true),
                ("max_tokens", false),
                ("model", true),
            ],
        );
        assert_eq!(shadowed, ["temperature"]);
        assert!(shadowed_typed_fields(Some(&json!([1])), &[("temperature", true)]).is_empty());
        assert!(shadowed_typed_fields(None, &[("temperature", true)]).is_empty());
    }

    /// A second `additional_params(Some(..))` merges into the first rather
    /// than replacing it; `None` clears, like every other folded setter.
    #[test]
    fn additional_params_merges_and_none_clears() {
        let request = crate::completion::CompletionRequest::new("p")
            .additional_params(json!({"a": 1}))
            .additional_params(json!({"b": 2}))
            .temperature(None);
        assert_eq!(request.additional_params, Some(json!({"a": 1, "b": 2})));
        assert_eq!(request.temperature, None);
        let cleared = crate::completion::CompletionRequest::new("p")
            .additional_params(json!({"a": 1}))
            .additional_params(None)
            .additional_params(json!({"b": 2}));
        assert_eq!(cleared.additional_params, Some(json!({"b": 2})));
    }
}

/// An unknown finish reason fails the turn by default, and replay leaves the
/// turn out. A request that accepts unknown reasons gets a response that
/// stops normally, and the turn replays: the response's stop, the run rule
/// and replay read the one answer.
mod unknown_finish_reasons {
    use crate::completion::message::turn_failure;
    use crate::completion::{CompletionRequest, FinishReason};
    use crate::message::{AssistantContent, Message, StopReason};
    use crate::test_utils::{MockCompletionModel, MockTurn};

    fn weird(turn: MockTurn) -> MockTurn {
        turn.with_finish_reason(FinishReason::Other("weird".to_owned()))
    }

    async fn turn(accept: bool, turn: MockTurn) -> (Message, Option<String>, bool) {
        let model = MockCompletionModel::from_turns([turn, MockTurn::text("next")]);
        let response = model
            .call(CompletionRequest::new("hi").accept_unknown_finish_reasons(accept))
            .await
            .expect("the reply folds");
        assert_eq!(response.accepts_unknown_finish_reasons(), accept);
        let head = response.head();
        let failure = turn_failure(
            &response.choice,
            head.stop.as_ref(),
            response.finish_reason().as_ref(),
        );
        let message = response.message().expect("the turn has content");
        model
            .call(CompletionRequest::new("again").messages([Message::user("hi"), message.clone()]))
            .await
            .expect("the follow-up folds");
        let replayed = model.requests()[1]
            .chat_history
            .iter()
            .any(|message| matches!(message, Message::Assistant(_)));
        (message, failure, replayed)
    }

    #[tokio::test]
    async fn a_text_turn_fails_and_is_left_out_by_default() {
        let (message, failure, replayed) = turn(false, weird(MockTurn::text("answer"))).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(
            turn.stop,
            Some(StopReason::Error(
                "Provider finish_reason: weird".to_owned()
            ))
        );
        assert!(failure.is_some_and(|failure| failure.contains("weird")));
        assert!(!replayed, "replay leaves the failed turn out");
    }

    #[tokio::test]
    async fn an_accepted_text_turn_succeeds_and_replays() {
        let (message, failure, replayed) = turn(true, weird(MockTurn::text("answer"))).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(turn.stop, Some(StopReason::Stop));
        assert_eq!(failure, None);
        assert!(replayed, "replay keeps the accepted turn");
    }

    #[tokio::test]
    async fn a_turn_with_calls_runs_them_only_when_accepted() {
        let call = || {
            weird(MockTurn::tool_call(
                "call_1",
                "lookup",
                serde_json::json!({}),
            ))
        };
        let (_, failure, _) = turn(false, call()).await;
        assert!(failure.is_some_and(|failure| failure.contains("none of its tool calls ran")));
        let (message, failure, _) = turn(true, call()).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(turn.stop, Some(StopReason::ToolUse));
        assert_eq!(failure, None);
    }

    /// Accepting unknown reasons never accepts filtered content or a failure
    /// the provider reported.
    #[test]
    fn filtered_content_and_reported_failures_still_fail() {
        let accepted = |reason: FinishReason| {
            super::CompletionResponse::new(
                vec![AssistantContent::text("partial")],
                super::Usage::default(),
                crate::message::Origin::new("mock", "mock", "mock"),
                serde_json::Value::Null,
            )
            .with_finish_reason(reason)
            .accept_unknown_finish_reasons(true)
        };
        assert!(accepted(FinishReason::ContentFilter).stop().is_failure());
        let mut reported = accepted(FinishReason::Other("weird".to_owned()));
        reported.error = Some("refused".to_owned());
        assert_eq!(reported.stop(), StopReason::Error("refused".to_owned()));
        assert_eq!(
            accepted(FinishReason::Other("weird".to_owned())).stop(),
            StopReason::Stop
        );
    }

    /// The choice travels with the response and the request, and neither
    /// writes it when it is off.
    #[test]
    fn the_choice_round_trips_and_is_omitted_when_off() {
        let request = CompletionRequest::new("hi").accept_unknown_finish_reasons(true);
        let json = serde_json::to_value(&request).expect("serializes");
        assert_eq!(json["accept_unknown_finish_reasons"], true);
        let back: CompletionRequest = serde_json::from_value(json).expect("parses");
        assert!(back.accept_unknown_finish_reasons);
        let off = serde_json::to_value(CompletionRequest::new("hi")).expect("serializes");
        assert!(off.get("accept_unknown_finish_reasons").is_none());

        let response = super::CompletionResponse::new(
            Vec::new(),
            super::Usage::default(),
            crate::message::Origin::new("mock", "mock", "mock"),
            serde_json::Value::Null,
        );
        let off = serde_json::to_value(&response).expect("serializes");
        assert!(off.get("accepts_unknown_finish_reasons").is_none());
        let on =
            serde_json::to_value(response.accept_unknown_finish_reasons(true)).expect("serializes");
        let back: super::CompletionResponse = serde_json::from_value(on).expect("parses");
        assert!(back.accepts_unknown_finish_reasons());
    }
}

/// The fields added for options and cost are absent from the serialized
/// form while empty, so stored requests, usage and recordings keep their
/// bytes; set, they round-trip.
mod empty_new_fields_are_not_serialized {
    use super::{CompletionRequest, Usage};
    use crate::completion::{Cost, Effort, GenerationOptions};
    use serde_json::json;

    #[test]
    fn a_request_without_options_has_no_options_key() {
        let value = serde_json::to_value(CompletionRequest::new("hi")).expect("serializes");
        assert!(value.get("options").is_none(), "{value}");

        let request = CompletionRequest::new("hi")
            .options(GenerationOptions::default().reasoning(Effort::High));
        let value = serde_json::to_value(&request).expect("serializes");
        assert_eq!(value["options"]["reasoning"], json!({"effort": "high"}));
        let back = serde_json::from_value::<CompletionRequest>(value).expect("deserializes");
        assert_eq!(back.options, request.options);
    }

    #[test]
    fn usage_without_cost_keeps_its_keys() {
        let usage = Usage::new()
            .input_tokens(4)
            .output_tokens(2)
            .total_tokens(6);
        assert_eq!(
            serde_json::to_value(usage).expect("serializes"),
            json!({"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
        );
        assert_eq!(
            serde_json::to_value(Usage::default()).expect("serializes"),
            json!({})
        );

        let priced = usage.cost(Cost::from_total(0.5));
        let value = serde_json::to_value(priced).expect("serializes");
        assert_eq!(value["cost"]["total"], json!(0.5));
        assert_eq!(
            serde_json::from_value::<Usage>(value).expect("deserializes"),
            priced
        );
    }

    #[test]
    fn a_cost_from_parts_totals_them() {
        let cost = Cost::from_parts(1.0, 2.0, 0.25, 0.5);
        assert_eq!(cost.total, 3.75);
        assert_eq!(
            (cost.input, cost.output, cost.cache_read, cost.cache_write),
            (Some(1.0), Some(2.0), Some(0.25), Some(0.5))
        );
    }

    #[test]
    fn a_cost_from_its_total_knows_no_part() {
        let cost = Cost::from_total(1.5);
        assert_eq!(cost.total, 1.5);
        assert_eq!(
            (cost.input, cost.output, cost.cache_read, cost.cache_write),
            (None, None, None, None)
        );
    }

    #[test]
    fn cost_part_setters_leave_the_total() {
        let cost = Cost::from_total(1.0)
            .input(0.25)
            .output(0.5)
            .cache_read(0.125)
            .cache_write(0.0625);
        assert_eq!(cost.total, 1.0);
        assert_eq!(
            (cost.input, cost.output, cost.cache_read, cost.cache_write),
            (Some(0.25), Some(0.5), Some(0.125), Some(0.0625))
        );
        assert_eq!(cost.input(None).input, None);
    }

    /// The total always sums; a part sums only when both sides know it.
    #[test]
    fn summing_a_known_part_with_an_unknown_one_is_unknown() {
        let split = Cost::from_parts(1.0, 2.0, 0.25, 0.5);
        let both = split + split;
        assert_eq!(both, Cost::from_parts(2.0, 4.0, 0.5, 1.0));

        let mixed = split + Cost::from_total(1.0);
        assert_eq!(mixed.total, 4.75);
        assert_eq!(
            (
                mixed.input,
                mixed.output,
                mixed.cache_read,
                mixed.cache_write
            ),
            (None, None, None, None)
        );

        let partial = Cost::from_total(1.0).input(0.5) + split;
        assert_eq!(partial.input, Some(1.5));
        assert_eq!(partial.output, None);
        assert_eq!(partial.total, 4.75);
    }

    #[test]
    fn unknown_cost_parts_are_absent_on_the_wire() {
        let total = serde_json::to_value(Cost::from_total(0.5)).expect("serializes");
        assert_eq!(total, json!({ "total": 0.5 }));
        assert_eq!(
            serde_json::from_value::<Cost>(total).expect("deserializes"),
            Cost::from_total(0.5)
        );

        let split = Cost::from_parts(1.0, 2.0, 0.0, 0.0);
        let value = serde_json::to_value(split).expect("serializes");
        assert_eq!(
            value,
            json!({ "input": 1.0, "output": 2.0, "cache_read": 0.0, "cache_write": 0.0, "total": 3.0 })
        );
        assert_eq!(
            serde_json::from_value::<Cost>(value).expect("deserializes"),
            split
        );
    }

    #[test]
    fn token_sums_are_unchanged_when_no_turn_has_a_cost() {
        let first = Usage::new().input_tokens(3).cached_input_tokens(1);
        let second = Usage::new().input_tokens(5).output_tokens(2);
        let sum = first + second;
        assert_eq!(sum.input_tokens, Some(8));
        assert_eq!(sum.cached_input_tokens, Some(1));
        assert_eq!(sum.output_tokens, Some(2));
        assert_eq!(sum.cost, None);
        assert_eq!(Usage::default() + Usage::default(), Usage::default());
    }
}

/// The generation-option shortcuts write the field their long form writes.
mod generation_option_shortcuts {
    use super::CompletionRequest;
    use crate::completion::{
        CacheRetention, Effort, GenerationOptions, OnUnsupported, Reasoning, ServiceTier, Verbosity,
    };

    /// The request as JSON; a request that does not serialize fails the
    /// test, so two failures never compare equal.
    fn json(request: &CompletionRequest) -> serde_json::Value {
        serde_json::to_value(request).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn each_shortcut_equals_its_long_form() {
        type Pair = (
            fn(CompletionRequest) -> CompletionRequest,
            fn(GenerationOptions) -> GenerationOptions,
        );
        let pairs: [Pair; 10] = [
            (|r| r.reasoning(Effort::High), |o| o.reasoning(Effort::High)),
            (
                |r| r.reasoning(Reasoning::Budget { tokens: 512 }),
                |o| o.reasoning(Reasoning::Budget { tokens: 512 }),
            ),
            (
                |r| r.cache(CacheRetention::Long),
                |o| o.cache(CacheRetention::Long),
            ),
            (
                |r| r.service_tier(ServiceTier::Flex),
                |o| o.service_tier(ServiceTier::Flex),
            ),
            (
                |r| r.verbosity(Verbosity::Low),
                |o| o.verbosity(Verbosity::Low),
            ),
            (
                |r| r.parallel_tool_calls(false),
                |o| o.parallel_tool_calls(false),
            ),
            (|r| r.top_p(0.9), |o| o.top_p(0.9)),
            (|r| r.seed(7), |o| o.seed(7)),
            (|r| r.stop(["END", "STOP"]), |o| o.stop(["END", "STOP"])),
            (
                |r| r.on_unsupported(OnUnsupported::Ignore),
                |o| o.on_unsupported(OnUnsupported::Ignore),
            ),
        ];
        for (short, long) in pairs {
            let short = short(CompletionRequest::new("hi"));
            let long = CompletionRequest::new("hi").options(long(GenerationOptions::new()));
            assert!(!short.options.is_default());
            assert_eq!(short.options, long.options);
            assert_eq!(json(&short), json(&long));
        }
    }

    #[test]
    fn shortcuts_keep_the_other_fields() {
        let request = CompletionRequest::new("hi")
            .reasoning(Effort::Low)
            .cache(CacheRetention::Short)
            .service_tier(ServiceTier::Priority)
            .verbosity(Verbosity::High)
            .parallel_tool_calls(true)
            .top_p(0.5)
            .seed(3)
            .stop(["x"])
            .on_unsupported(OnUnsupported::Error);
        let options = GenerationOptions::new()
            .reasoning(Effort::Low)
            .cache(CacheRetention::Short)
            .service_tier(ServiceTier::Priority)
            .verbosity(Verbosity::High)
            .parallel_tool_calls(true)
            .top_p(0.5)
            .seed(3)
            .stop(["x"])
            .on_unsupported(OnUnsupported::Error);
        assert_eq!(request.options, options);
        assert_eq!(
            json(&request),
            json(&CompletionRequest::new("hi").options(options))
        );
    }

    #[test]
    fn calls_apply_in_order() {
        let shared = GenerationOptions::new().reasoning(Effort::High).seed(1);
        // `options` after a shortcut replaces every field, the shortcut's too.
        let request = CompletionRequest::new("hi")
            .seed(7)
            .top_p(0.2)
            .options(shared.clone());
        assert_eq!(request.options, shared);
        // A shortcut after `options` sets its one field on top.
        let request = CompletionRequest::new("hi").options(shared.clone()).seed(7);
        assert_eq!(request.options, shared.seed(7));
    }

    #[test]
    fn new_is_the_default() {
        assert_eq!(GenerationOptions::new(), GenerationOptions::default());
        assert!(GenerationOptions::new().is_default());
    }
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_status() {
    let body = r#"{"error":{"message":"too many requests"}}"#;
    let error = ProviderError::ProviderResponse(provider_response::ProviderResponseError::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
}

#[test]
fn completion_error_provider_error_is_not_a_provider_response() {
    // `ProviderError` also carries Rig-generated diagnostics, so the helpers
    // must not report its string as a provider response body.
    let error = ProviderError::Provider("stream transport failed".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("no body is not an error"),
        None
    );
}

#[test]
fn completion_error_provider_response_helpers_with_unrelated_variant() {
    let error = ProviderError::Response("failed to parse provider response".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("no body is not an error"),
        None
    );
}

#[test]
fn provider_response_json_returns_none_for_empty_preserved_body() {
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(String::new()),
    );

    assert_eq!(error.provider_response_body(), Some(""));
    assert_eq!(
        error
            .provider_response_json()
            .expect("empty body is not a JSON parse error"),
        None
    );
}
