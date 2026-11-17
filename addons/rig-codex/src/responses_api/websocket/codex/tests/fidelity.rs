//! The request bytes a Codex session writes, on every path that sends a
//! `response.create` frame: a full send, a send with options, a warmup and
//! an incremental send.

use super::*;
use crate::responses_api::fidelity_fixtures::*;

fn caller() -> CallerIdentity {
    CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
        .expect("a valid caller identity")
}

/// A session on `wire`, named `session-7`/`thread-7`, over `script`.
fn session_on(wire: Responses, script: &Script) -> CodexWebSocketSession {
    let wire = CodexResponses::new(wire)
        .expect("the ChatGPT dialect speaks the Codex contract")
        .with_caller_identity(caller());
    let identity = CodexIdentity::from_ids("session-7", "thread-7").expect("header-safe ids");
    CodexWebSocketSession::from_connection(wire, identity, script.connection(), None)
        .expect("the ChatGPT dialect speaks the Codex contract")
}

fn plain_wire() -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN),
        "gpt-5.4",
    )
}

/// Read the turn in flight to its end.
async fn finish(session: &mut CodexWebSocketSession) {
    loop {
        let event = session.next_event().await.expect("the turn's events");
        if event.is_terminal() {
            return;
        }
    }
}

/// One frame from each path that sends `request`: a full send, a send with
/// warmup options, a warmup, an incremental send continuing the warmup, and
/// an incremental send continuing a full send.
async fn frames_of_every_send_path(wire: Responses, request: CompletionRequest) -> Vec<String> {
    let script = Script::new()
        .turn([completed("resp_full")])
        .turn([completed("resp_delta")])
        .turn([completed("resp_options")])
        .turn([completed("resp_warm")])
        .turn([completed("resp_first")]);
    let mut session = session_on(wire, &script);

    session.send(request.clone()).await.expect("a full send");
    finish(&mut session).await;
    session
        .send_incremental(delta("next"))
        .await
        .expect("a delta on the full send");
    finish(&mut session).await;
    session
        .send_with_options(request.clone(), ResponsesWebSocketCreateOptions::warmup())
        .await
        .expect("a send with options");
    finish(&mut session).await;
    session.warmup(request).await.expect("a warmup");
    session
        .send_incremental(delta("first"))
        .await
        .expect("a delta on the warmup");
    finish(&mut session).await;

    let frames = script.sent();
    assert_eq!(frames.len(), 5, "one frame per send: {frames:?}");
    frames
}

/// A full send of the request Muninn's Codex lane makes is the HTTP body
/// with the frame type ahead of it: byte for byte in a build that sorts
/// keys, and the same JSON value in any build. The session's identity states
/// the cache key and metadata the caller would.
#[tokio::test]
async fn a_full_send_writes_the_pinned_frame() {
    let wire = Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, ACCESS_TOKEN).with_instructions(""),
        "gpt-5.4",
    );
    let mut unnamed = muninn_shaped_request();
    unnamed.additional_params = Some(json!({
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "store": false,
    }));

    for request in [muninn_shaped_request(), unnamed] {
        let script = Script::new().turn([completed("resp_1")]);
        let mut session = session_on(wire.clone(), &script);
        session.send(request).await.expect("a full send");
        let frame = script.sent().remove(0);

        let mut expected: Value =
            serde_json::from_str(MUNINN_SHAPED_BODY_SORTED).expect("the pinned body");
        expected["type"] = json!("response.create");
        assert_eq!(
            serde_json::from_str::<Value>(&frame).expect("a JSON frame"),
            expected
        );
        if !preserves_order() {
            assert_eq!(
                frame,
                format!(
                    r#"{{"type":"response.create",{}"#,
                    &MUNINN_SHAPED_BODY_SORTED[1..]
                )
            );
        }
    }
}

/// Every path that sends a frame writes the declared tools as the bytes of
/// the JSON values the caller wrote, after the typed tool.
#[tokio::test]
async fn every_send_path_writes_declared_tools_as_the_caller_wrote_them() {
    let declared = [namespace(), declared_function(), declared_hosted()];
    let frames = frames_of_every_send_path(
        plain_wire(),
        turn_declaring(Value::Array(declared.to_vec())),
    )
    .await;

    for frame in &frames {
        let tools: Vec<Box<RawValue>> =
            serde_json::from_str(raw_fields(frame)["tools"].get()).expect("`tools` is an array");
        let tools: Vec<&str> = tools.iter().map(|tool| tool.get()).collect();
        assert_eq!(tools.len(), 4, "typed and three declared: {frame}");
        assert_eq!(
            serde_json::from_str::<Value>(tools[0]).expect("JSON")["name"],
            "lookup"
        );
        assert_eq!(
            tools[1..],
            declared.iter().map(written).collect::<Vec<_>>(),
            "{frame}"
        );
    }
}

/// Under strict mode every path sends a declared namespace exactly as
/// written, and a declared function changed only by strict mode's own
/// transformation.
#[tokio::test]
async fn every_send_path_keeps_declared_tools_under_strict_mode() {
    let mut request = user_request("hello");
    request.additional_params = Some(json!({ "tools": [namespace(), declared_function()] }));
    let frames = frames_of_every_send_path(plain_wire().with_strict_tools(), request).await;

    let mut strict_function = declared_function();
    strict_function["strict"] = json!(true);
    strict_function["parameters"] = json!({
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
        "additionalProperties": false
    });
    let expected = format!("[{},{}]", written(&namespace()), written(&strict_function));
    for frame in &frames {
        assert_eq!(raw_fields(frame)["tools"].get(), expected, "{frame}");
    }
}

/// No path forces the ciphertext `include` onto a turn without reasoning,
/// and every path states it alongside reasoning.
#[tokio::test]
async fn no_send_path_forces_the_reasoning_include() {
    for frame in frames_of_every_send_path(plain_wire(), user_request("plain")).await {
        assert!(
            !raw_fields(&frame).contains_key("include"),
            "no include is forced onto a turn without reasoning: {frame}"
        );
    }

    let mut reasoning = user_request("think");
    reasoning.additional_params = Some(json!({ "reasoning": { "effort": "low" } }));
    for frame in frames_of_every_send_path(plain_wire(), reasoning).await {
        assert_eq!(
            raw_fields(&frame)["include"].get(),
            r#"["reasoning.encrypted_content"]"#,
            "{frame}"
        );
    }

    let mut own = user_request("search");
    own.additional_params = Some(json!({ "include": ["file_search_call.results"] }));
    for frame in frames_of_every_send_path(plain_wire(), own).await {
        assert_eq!(
            raw_fields(&frame)["include"].get(),
            r#"["file_search_call.results"]"#,
            "{frame}"
        );
    }
}
