use rig::message::Message;
use rig::providers::{chatgpt, openai::OpenAIConfig};
use rig_core::test_utils::{RecordingHttpClient, test_caller_identity};
use rig_test_support::cassette_models::OpenAiModels;

use super::recorded_include;
use crate::cassettes::recorded_interaction_bodies;

/// Captures the current outbound instruction/history shape using the original SSE.
/// Classic recorded-request replay and provider acceptance remain unproven.
#[tokio::test]
async fn explicit_history_is_lifted_on_the_current_agent_boundary() {
    let recorded = recorded_interaction_bodies(
        "chatgpt",
        "codex_behaviors/explicit_preamble_and_mid_conversation_system_messages_are_instructions",
    );
    assert_eq!(
        recorded.len(),
        1,
        "the original scenario records one interaction"
    );
    let (original_request, received_sse) = recorded
        .into_iter()
        .next()
        .expect("the original scenario contains its one interaction");
    assert_eq!(
        received_sse.len(),
        7373,
        "the original received SSE stays unchanged"
    );
    let http = RecordingHttpClient::new(received_sse.into_bytes());
    let caller_identity = test_caller_identity();
    let config = OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-test-key")
        .with_caller_identity(caller_identity.clone())
        .with_account_id("synthetic-account")
        .with_instructions("");
    let client = OpenAiModels::new(config, http.clone());
    let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
        .preamble("You are a concise assistant.")
        .additional_params(recorded_include())
        .build();
    let mut history = vec![
        Message::user("Hello!"),
        Message::assistant("Hi! How can I help you today?"),
        Message::system("The user's codename is FALCON-9. Always refer to the user by codename."),
    ];

    let result = agent
        .chat("What is my codename?", &mut history)
        .await
        .expect("the agent should decode the original received SSE");

    assert!(
        result.output().contains("FALCON-9"),
        "the original recorded answer must retain the codename, got {result:?}"
    );

    let sent = http.requests();
    assert_eq!(
        sent.len(),
        1,
        "the actual agent chat sends exactly one request"
    );
    assert_eq!(sent[0].headers["originator"], caller_identity.originator());
    assert_eq!(sent[0].headers["user-agent"], caller_identity.user_agent());
    assert_eq!(
        sent[0]
            .headers
            .get("version")
            .map(|value| value.to_str().expect("the version header is text")),
        caller_identity.version()
    );
    assert_eq!(sent[0].headers["chatgpt-account-id"], "synthetic-account");
    let actual: serde_json::Value = serde_json::from_slice(sent[0].body.as_ref())
        .expect("the captured whole request body is JSON");
    let mut expected: serde_json::Value =
        serde_json::from_str(&original_request).expect("the original whole request body is JSON");
    assert_eq!(
        expected["input"][1],
        serde_json::json!({
            "type":"message", "role":"assistant", "content":"Hi! How can I help you today?",
        })
    );
    // Current canonical history sends an assistant message item instead of bare text.
    expected["input"][1]["content"] = serde_json::json!([
        {"type":"output_text", "text":"Hi! How can I help you today?", "annotations":[]},
    ]);
    expected["input"][1]["id"] = serde_json::json!("msg_rig_2");
    expected["input"][1]["status"] = serde_json::json!("completed");
    let instructions = "You are a concise assistant.\n\nThe user's codename is FALCON-9. Always refer to the user by codename.";
    assert_eq!(
        expected
            .as_object_mut()
            .expect("the original request is an object")
            .shift_remove("instructions"),
        Some(serde_json::json!(instructions))
    );
    expected["input"]
        .as_array_mut()
        .expect("the original input is an array")
        .insert(
            0,
            serde_json::json!({
                "type":"message", "role":"developer",
                "content":[{"type":"input_text", "text":instructions}],
                "internal_chat_message_metadata_passthrough":{
                    "content_item_kinds":["model.base_instructions"],
                },
            }),
        );
    assert_eq!(
        actual, expected,
        "only the adopted instruction carrier and named assistant item representation differ"
    );
    assert!(actual.get("instructions").is_none());
    assert_eq!(
        actual["input"][0]["content"][0]["text"].as_str(),
        Some(instructions),
    );
    assert!(
        actual["input"]
            .as_array()
            .expect("the actual input is an array")
            .iter()
            .all(|item| item["role"].as_str() != Some("system")),
        "every original system message is lifted into the developer instructions"
    );
}
