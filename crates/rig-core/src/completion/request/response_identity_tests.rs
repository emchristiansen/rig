use super::*;

/// `provider_request_id` is skipped when `None`, so a response written
/// without one loads with the field `None` (rig#2265).
#[test]
fn completion_response_without_request_id_deserializes() {
    let response: CompletionResponse = serde_json::from_str(
        r#"{"choice": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "cached_input_tokens": 0, "cache_creation_input_tokens": 0,
                          "reasoning_tokens": 0},
                "origin": {"api": "test.api", "provider": "test", "model": ""},
                "raw": null}"#,
    )
    .expect("a CompletionResponse without a request id should load");
    assert_eq!(response.provider_request_id, None);
    assert_eq!(response.identity(), ResponseIdentity::default());
}

/// The identity accessor keeps message, response and transport IDs distinct.
#[test]
fn identity_accessor_mirrors_flat_fields() {
    let mut origin = crate::message::Origin::new("test.api", "test", "requested-model");
    origin.response_id = Some("resp_1".to_owned());
    let mut response = CompletionResponse::new(
        vec![crate::completion::AssistantContent::text("hi")],
        Usage::default(),
        origin,
        serde_json::json!({}),
    )
    .with_message_id(ProviderMessageId::new("msg_1").expect("message id"));
    response.provider_request_id = Some("req_1".to_owned());
    assert_eq!(
        response.identity(),
        ResponseIdentity {
            message_id: ProviderMessageId::new("msg_1"),
            response_id: Some("resp_1".to_owned()),
            provider_request_id: Some("req_1".to_owned()),
        }
    );
    let encoded = serde_json::to_value(&response).expect("serialize all IDs");
    let decoded = serde_json::from_value::<CompletionResponse>(encoded).expect("read all IDs");
    assert_eq!(
        decoded.message_id.as_ref().map(ProviderMessageId::as_str),
        Some("msg_1")
    );
    assert_eq!(decoded.response_id(), Some("resp_1"));
    assert_eq!(decoded.provider_request_id.as_deref(), Some("req_1"));
    assert_eq!(decoded.identity(), response.identity());
}

#[test]
fn an_absent_message_id_keeps_the_exact_upstream_identity_serialization() {
    let identity = ResponseIdentity {
        message_id: None,
        response_id: Some("resp_1".to_owned()),
        provider_request_id: Some("req_1".to_owned()),
    };
    assert_eq!(
        serde_json::to_string(&identity).expect("serialize identity"),
        r#"{"response_id":"resp_1","provider_request_id":"req_1"}"#
    );
    let decoded = serde_json::from_str::<ResponseIdentity>(
        r#"{"response_id":"resp_1","provider_request_id":"req_1"}"#,
    )
    .expect("read upstream identity");
    assert_eq!(decoded, identity);
    assert_eq!(
        serde_json::to_string(&ResponseIdentity::default()).expect("empty identity"),
        "{}"
    );
}

#[test]
fn response_and_message_ids_never_fill_in_for_each_other() {
    let mut origin = crate::message::Origin::new("test.api", "test", "requested-model");
    origin.response_id = Some("resp_only".to_owned());
    let mut response_only = CompletionResponse::new(
        vec![crate::message::AssistantContent::text("hi")],
        Usage::default(),
        origin,
        serde_json::json!({"message_id": "raw_message", "id": "raw_id"}),
    );
    response_only.provider_request_id = Some("req_only".to_owned());
    let encoded = serde_json::to_value(&response_only).expect("serialize response-only ID");
    assert!(encoded.get("message_id").is_none());
    let decoded = serde_json::from_value::<CompletionResponse>(encoded.clone())
        .expect("read response-only ID");
    assert_eq!(decoded.message_id, None);
    assert_eq!(decoded.response_id(), Some("resp_only"));
    assert_eq!(decoded.provider_request_id.as_deref(), Some("req_only"));
    assert_eq!(
        serde_json::to_value(decoded).expect("serialize again"),
        encoded
    );

    let message_only = CompletionResponse::new(
        vec![crate::message::AssistantContent::text("hi")],
        Usage::default(),
        crate::message::Origin::new("test.api", "test", "requested-model"),
        serde_json::json!({"response_id": "raw_response", "id": "raw_id"}),
    )
    .with_message_id(ProviderMessageId::new("msg_only").expect("message id"));
    let encoded = serde_json::to_value(&message_only).expect("serialize message-only ID");
    assert_eq!(encoded["message_id"], "msg_only");
    assert!(encoded["origin"].get("response_id").is_none());
    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("read message-only ID");
    assert_eq!(
        decoded.message_id.as_ref().map(ProviderMessageId::as_str),
        Some("msg_only")
    );
    assert_eq!(decoded.response_id(), None);
    assert_eq!(decoded.provider_request_id, None);
    assert_eq!(decoded.identity().response_id, None);
    let cleared = decoded.with_optional_message_id(None);
    assert_eq!(cleared.message_id, None);
    assert_eq!(cleared.response_id(), None);
    assert!(
        serde_json::to_value(cleared)
            .expect("serialize cleared ID")
            .get("message_id")
            .is_none()
    );
}

#[test]
fn provider_message_ids_are_nonempty_and_keep_their_exact_wire_spelling() {
    assert_eq!(ProviderMessageId::new(""), None);
    assert!(ProviderMessageId::try_from(String::new()).is_err());
    assert!(serde_json::from_str::<ProviderMessageId>(r#""""#).is_err());
    let id = ProviderMessageId::new(" \tmsg_1 ").expect("nonempty whitespace is preserved");
    assert_eq!(id.as_str(), " \tmsg_1 ");
    assert_eq!(id.to_string(), " \tmsg_1 ");
    let encoded = serde_json::to_string(&id).expect("serialize typed ID");
    assert_eq!(encoded, r#"" \tmsg_1 ""#);
    assert_eq!(
        serde_json::from_str::<ProviderMessageId>(&encoded).expect("read typed ID"),
        id
    );
}
