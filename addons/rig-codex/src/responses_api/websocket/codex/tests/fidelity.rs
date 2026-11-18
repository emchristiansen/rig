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

    // The members strict mode adds, in the order it adds them: a build that
    // keeps insertion order sends them so.
    let mut strict_function = declared_function();
    strict_function["strict"] = json!(true);
    strict_function["parameters"] = json!({
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "additionalProperties": false,
        "required": ["body"]
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

/// Every path that sends a request sends a custom call's answer as one
/// string naming the tool that ran, and so does the projection a caller
/// builds an incremental turn's items with.
#[tokio::test]
async fn every_send_path_sends_a_custom_calls_answer_as_one_named_string() {
    use rig_core::message::ToolResultContent;
    let content = || {
        vec![
            ToolResultContent::text("Done: "),
            ToolResultContent::Json {
                value: json!({"files": 1}),
            },
        ]
    };
    // The wire states the name after the members Rig wrote: a build that
    // keeps insertion order sends them so.
    let answer = json!({
        "type": "custom_tool_call_output", "call_id": "call_c",
        "output": "Done: {\"files\":1}", "name": "apply_patch"
    });

    let frames =
        frames_of_every_send_path(plain_wire(), answering(custom_call_item(), content())).await;
    for frame in [&frames[0], &frames[2], &frames[3]] {
        let input = raw_input(frame);
        assert_eq!(
            serde_json::from_str::<Value>(&input[2]).expect("JSON"),
            answer,
            "{frame}"
        );
        if !preserves_order() {
            assert_eq!(
                input[2],
                r#"{"call_id":"call_c","name":"apply_patch","output":"Done: {\"files\":1}","type":"custom_tool_call_output"}"#
            );
        }
    }

    // The answer alone, appended to stored state.
    let items = input_items(&plain_wire(), answering_stored_custom_call(content()))
        .expect("the appended answer encodes");
    assert_eq!(items, vec![answer.clone()]);
    let script = Script::new()
        .turn([completed("resp_1")])
        .turn([completed("resp_2")]);
    let mut session = session_on(plain_wire(), &script);
    session
        .send(user_request("patch it"))
        .await
        .expect("a send");
    finish(&mut session).await;
    session
        .send_incremental(InputDelta::new(items).expect("non-empty"))
        .await
        .expect("an incremental send");
    let delta = &script.sent()[1];
    assert_eq!(raw_input(delta), [written(&answer)], "{delta}");
}

/// No path sends an image Rig v0.46.0 would send as a part of a custom
/// call's answer: a full send, a send with options and a warmup refuse the
/// request and write no frame, and the projection of an incremental turn's
/// items refuses it.
#[tokio::test]
async fn no_send_path_sends_an_image_in_a_custom_calls_answer() {
    use rig_core::message::ToolResultContent;
    for image in images_rig_sends_in_a_result() {
        let content = || {
            vec![
                ToolResultContent::text("see "),
                ToolResultContent::Image(image.clone()),
            ]
        };
        let request = answering(custom_call_item(), content());
        let script = Script::new();
        let mut session = session_on(plain_wire(), &script);

        let full = session
            .send(request.clone())
            .await
            .expect_err("a full send");
        let with_options = session
            .send_with_options(request.clone(), ResponsesWebSocketCreateOptions::warmup())
            .await
            .expect_err("a send with options");
        let warmup = session.warmup(request).await.expect_err("a warmup");
        let projection = input_items(&plain_wire(), answering_stored_custom_call(content()))
            .expect_err("the projection of appended items");
        for refusal in [
            format!("{full}"),
            format!("{with_options}"),
            format!("{warmup}"),
            format!("{projection}"),
        ] {
            assert!(refusal.contains(CUSTOM_IMAGE_REFUSAL), "{refusal}");
        }
        assert!(
            script.sent().is_empty(),
            "nothing is written: {:?}",
            script.sent()
        );
    }
}

/// No path sends a request that states a control the Codex contract does
/// not take, whichever way the request states it.
#[tokio::test]
async fn no_send_path_sends_a_refused_control() {
    for (name, request) in requests_stating_a_refused_control() {
        let script = Script::new();
        let mut session = session_on(plain_wire(), &script);
        let full = session
            .send(request.clone())
            .await
            .expect_err("a full send");
        let with_options = session
            .send_with_options(request.clone(), ResponsesWebSocketCreateOptions::warmup())
            .await
            .expect_err("a send with options");
        let warmup = session.warmup(request).await.expect_err("a warmup");
        for refusal in [
            format!("{full}"),
            format!("{with_options}"),
            format!("{warmup}"),
        ] {
            assert!(refusal.contains(&format!("`{name}")), "{name}: {refusal}");
        }
        assert!(script.sent().is_empty(), "{name}: {:?}", script.sent());
    }
}

/// Every path sends a request whose later layer takes a refused control
/// back out, and the request's own `include` from whichever layer states
/// it; an incremental send restates the full send's `include`.
#[tokio::test]
async fn every_send_path_honours_the_layers_of_a_request() {
    for request in requests_overriding_a_refused_control() {
        for frame in frames_of_every_send_path(plain_wire(), request).await {
            let frame: Value = serde_json::from_str(&frame).expect("a JSON frame");
            assert_eq!(frame["store"], false, "{frame}");
        }
    }
    for (request, include) in requests_and_their_include() {
        let frames = frames_of_every_send_path(plain_wire(), request.clone()).await;
        for frame in &frames {
            let frame: Value = serde_json::from_str(frame).expect("a JSON frame");
            assert_eq!(
                frame.get("include"),
                include.as_ref(),
                "{request:?}: {frame}"
            );
        }
    }
}

/// A send whose future is dropped while the connection is still writing
/// leaves a turn in flight that the session knows it cannot vouch for: it
/// does not present an idle socket. Every send is refused by name and
/// writes nothing, the reply is read if the backend sends one, and the
/// session takes sends again once that turn ends.
#[tokio::test]
async fn a_send_dropped_while_writing_keeps_custody_of_its_turn() {
    use futures::FutureExt as _;
    for incremental in [false, true] {
        let script = if incremental {
            Script::new().turn([completed("resp_first")])
        } else {
            Script::new()
        }
        .turn([completed("resp_uncertain")])
        .turn([completed("resp_after")]);
        let mut session = session_on(plain_wire(), &script);
        if incremental {
            session.send(user_request("first")).await.expect("a send");
            finish(&mut session).await;
        }

        // The connection takes the frame and does not finish writing it.
        script.set_send_stalls(true);
        let written_before = script.sent().len();
        let dropped = if incremental {
            session.send_incremental(delta("next")).now_or_never()
        } else {
            session.send(user_request("first")).now_or_never()
        };
        assert!(dropped.is_none(), "the send is still pending when dropped");
        script.set_send_stalls(false);
        assert_eq!(script.sent().len(), written_before + 1, "the frame left");
        assert!(session.send_unconfirmed());

        // Not idle: no second request goes out behind the uncertain one.
        let refused = [
            session
                .send(user_request("second"))
                .await
                .expect_err("a send"),
            session
                .send_incremental(delta("second"))
                .await
                .expect_err("an incremental send"),
            session
                .warmup(user_request("second"))
                .await
                .expect_err("a warmup")
                .into(),
        ];
        for error in &refused {
            let ProviderError::Request(reason) = error else {
                panic!("a refused send is a request error, got {error:?}");
            };
            assert_eq!(
                reason.downcast_ref::<UnconfirmedSend>(),
                Some(&UnconfirmedSend),
                "{error}"
            );
        }
        assert_eq!(
            script.sent().len(),
            written_before + 1,
            "nothing was written"
        );

        // The backend did take the frame, and its reply is this turn's.
        let ending = loop {
            let event = session
                .next_event()
                .await
                .expect("the uncertain turn's events");
            if event.is_terminal() {
                break event;
            }
        };
        assert!(
            format!("{ending:?}").contains("resp_uncertain"),
            "{ending:?}"
        );
        assert!(!session.send_unconfirmed());
        session
            .send(user_request("after"))
            .await
            .expect("the session takes sends again");
        assert_eq!(script.sent().len(), written_before + 2);
    }
}

/// When the backend never answers an uncertain send, the caller closes the
/// session and sends again on a new one. Nothing requires knowing whether
/// the first request arrived: it may be delivered twice.
#[tokio::test]
async fn an_uncertain_send_is_resent_on_a_new_session() {
    use futures::FutureExt as _;
    let script = Script::new();
    let mut session = session_on(plain_wire(), &script);
    script.set_send_stalls(true);
    assert!(session.send(user_request("ask")).now_or_never().is_none());
    assert!(session.send_unconfirmed());
    assert!(
        session.next_event().now_or_never().is_none(),
        "no reply has come: the read waits rather than report an idle session"
    );
    session.close().await.expect("the session closes");

    let again = Script::new().turn([completed("resp_again")]);
    let mut session = session_on(plain_wire(), &again);
    session.send(user_request("ask")).await.expect("the resend");
    finish(&mut session).await;
    assert_eq!(
        script.sent(),
        again.sent(),
        "the same request left on both sessions"
    );
}
