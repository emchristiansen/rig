use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use crate::error::ProviderError;
use crate::message::{AssistantContent, Message, ToolName, ToolResultContent, UserContent};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Mode, Operation, Wire, WireFrame};

fn response(reasoning: Value) -> CompletionResponse {
    let body = json!({
        "id":"resp_1", "object":"response", "created_at":0,
        "status":"completed", "model":"gpt-5.4", "output":[
            reasoning,
            {"type":"function_call", "id":"fc_1", "call_id":"call_1",
             "name":"f", "arguments":"{}", "status":"completed"}
        ]
    });
    crate::test_utils::history::decode(
        &OpenAIConfig::new("test-key").responses("gpt-5.4"),
        Mode::Unary,
        [WireFrame::Text(body.to_string())],
    )
    .expect("the genuine reply fixture decodes")
}

fn known(ciphertext: Option<&str>) -> Value {
    let mut item = json!({"type":"reasoning", "id":"rs_1",
        "summary":[{"type":"summary_text", "text":"plan"}]});
    if let Some(ciphertext) = ciphertext {
        item["encrypted_content"] = json!(ciphertext);
    }
    item
}

fn input(response: &CompletionResponse, stateless: bool) -> Result<Vec<Value>, ProviderError> {
    let wire = OpenAIConfig::new("test-key").responses("gpt-5.4");
    let results = response
        .tool_calls()
        .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")])))
        .collect();
    let mut request = CompletionRequest::from(vec![
        Message::user("q"),
        response
            .message()
            .expect("the decoded reply is an assistant"),
        Message::User { content: results },
    ])
    .tools(vec![ToolDefinition {
        name: ToolName::new("f").expect("tool name"),
        description: "A tool".to_owned(),
        parameters: json!({"type":"object"}),
    }]);
    request.additional_params = Some(json!({"store":!stateless}));
    let prepared = Completion::prepare(request, &wire.describe())?;
    let encoded = wire
        .encode(prepared, Mode::Unary)
        .map_err(ProviderError::from)?;
    Ok(crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .expect("encoded input array")
        .clone())
}

#[test]
fn stateless_known_carried_without_ciphertext_omits_reasoning_and_rebuilds_its_partner() {
    for ciphertext in [None, Some("")] {
        let response = response(known(ciphertext));
        let input = input(&response, true).expect("stateless request encodes");
        assert_eq!(
            input,
            vec![
                json!({"type":"message", "role":"user", "content":[{"type":"input_text","text":"q"}]}),
                json!({"type":"function_call", "call_id":"call_1", "name":"f", "arguments":"{}"}),
                json!({"type":"function_call_output", "call_id":"call_1", "output":"ok", "status":"completed"}),
            ]
        );
    }
}

#[test]
fn stored_known_carried_without_ciphertext_retains_reasoning_and_partner_identity() {
    let response = response(known(None));
    let input = input(&response, false).expect("stored request encodes");
    assert_eq!(input.len(), 4);
    assert_eq!(input[1], known(None));
    assert_eq!(input[2], response.raw["output"][1]);
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["call_id"], "call_1");
}

#[test]
fn stateless_known_carried_with_ciphertext_retains_reasoning_and_partner_identity() {
    let response = response(known(Some("ENC")));
    let input = input(&response, true).expect("resolvable stateless request encodes");
    assert_eq!(input.len(), 4);
    assert_eq!(input[1], known(Some("ENC")));
    assert_eq!(input[2], response.raw["output"][1]);
}

#[test]
fn stateless_opaque_carried_keeps_its_existing_received_replay() {
    let reasoning = json!({"type":"reasoning", "id":"rs_1",
        "summary":[{"type":"future_summary", "payload":{"n":1}}]});
    let response = response(reasoning.clone());
    let input = input(&response, true).expect("same-issuer opaque replay is unchanged");
    assert_eq!(input.len(), 4);
    assert_eq!(input[1], reasoning);
    assert_eq!(input[2], response.raw["output"][1]);
}

#[test]
fn stateless_foreign_opaque_carried_keeps_its_existing_typed_refusal() {
    let mut response = response(json!({"type":"reasoning", "id":"rs_1",
        "summary":[{"type":"future_summary", "payload":{"n":1}}]}));
    let Some(AssistantContent::CarriedReasoning(reasoning)) = response.choice.first_mut() else {
        panic!("the actual unknown summary is carried");
    };
    // This is an explicitly foreign caller input, not received issuer evidence.
    reasoning.payload.provider = Some("foreign".to_owned());
    let error = input(&response, true).expect_err("foreign opaque custody refuses");
    let ProviderError::Request(source) = error else {
        panic!("the existing refusal is Request: {error:?}");
    };
    assert!(
        source
            .downcast_ref::<crate::message::UnrepresentableOpaqueContent>()
            .is_some()
    );
}
