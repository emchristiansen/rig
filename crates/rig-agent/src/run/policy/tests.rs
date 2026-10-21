use super::*;

/// Every decision type a host may cache in serializable state round-trips.
#[test]
fn decision_types_round_trip_through_serde() {
    for action in [
        InvalidToolCallAction::fail(),
        InvalidToolCallAction::retry("try add"),
        InvalidToolCallAction::repair("add"),
        InvalidToolCallAction::skip("nope"),
        InvalidToolCallAction::stop("done"),
    ] {
        let json = serde_json::to_string(&action).expect("serialize action");
        assert_eq!(
            serde_json::from_str::<InvalidToolCallAction>(&json).expect("deserialize action"),
            action
        );
    }

    let retry = RetryRequest::Feedback("again".to_string());
    let json = serde_json::to_string(&retry).expect("serialize retry");
    assert_eq!(
        serde_json::from_str::<RetryRequest>(&json).expect("deserialize retry"),
        retry
    );
}

/// The malformed-arguments reason carries the parser's error, or names the
/// JSON value that is not an object.
#[test]
fn malformed_arguments_name_what_the_parser_rejected() {
    let InvalidToolCallReason::MalformedArguments { error } =
        InvalidToolCallReason::malformed_arguments("{\"x\":")
    else {
        panic!("a malformed-arguments reason");
    };
    assert!(error.contains("line 1 column"), "{error}");
    for (raw, kind) in [
        ("null", "null"),
        ("true", "a boolean"),
        ("1", "a number"),
        ("\"x\"", "a string"),
        ("[1]", "an array"),
    ] {
        assert_eq!(
            InvalidToolCallReason::malformed_arguments(raw),
            InvalidToolCallReason::MalformedArguments {
                error: format!("expected a JSON object, found {kind}")
            }
        );
    }
    assert_eq!(json_kind(&serde_json::json!({})), "an object");
}

#[test]
fn reasons_round_trip_through_serde() {
    for reason in [
        InvalidToolCallReason::UnknownTool,
        InvalidToolCallReason::DisallowedByToolChoice,
        InvalidToolCallReason::malformed_arguments("{"),
    ] {
        let json = serde_json::to_value(&reason).expect("serialize reason");
        assert!(json["reason"].is_string(), "{json}");
        assert_eq!(
            serde_json::from_value::<InvalidToolCallReason>(json).expect("deserialize reason"),
            reason
        );
    }
}

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn resolve_specific_rejects_missing_tools() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("missing").expect("tool name")],
    };

    let err = TurnPolicy::resolve(executable, Some(choice), None, None)
        .expect_err("missing specific tool should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("missing")
                && err.to_string().contains("add")
    ));
}

#[test]
fn resolve_specific_rejects_empty_names() {
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![],
    };

    let err = TurnPolicy::resolve(executable, Some(choice), None, None)
        .expect_err("empty specific tool choice should fail before provider request");

    assert!(matches!(
        err,
        PrepareError::Request(err)
            if err.to_string().contains("requires at least one function name")
    ));
}

#[test]
fn required_with_no_advertised_tool_is_local_error() {
    let empty = tool_names(&[]);
    let err = TurnPolicy::resolve(empty, Some(ToolChoice::Required), None, None)
        .expect_err("Required with no advertised tool must fail locally");
    assert!(matches!(
        err,
        PrepareError::Request(err) if err.to_string().contains("Required")
    ));
}

#[test]
fn required_with_active_tools_filter_names_the_filter_in_the_error() {
    let empty = tool_names(&[]);
    let err = TurnPolicy::resolve(
        empty,
        Some(ToolChoice::Required),
        None,
        Some(&tool_names(&["add"])),
    )
    .expect_err("Required after active_tools filtered everything must fail locally");
    let msg = err.to_string();
    assert!(
        msg.contains("active_tools"),
        "error should name active_tools: {msg}"
    );
    assert!(
        msg.contains("RequestPatch"),
        "error should suggest RequestPatch: {msg}"
    );
}

#[test]
fn specific_typo_is_not_blamed_on_active_tools() {
    // Specific names a tool that never existed (a typo), even though an
    // active_tools filter was applied. The error must NOT blame active_tools,
    // because the filter never had that tool to drop.
    let executable = tool_names(&["add"]);
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("nonexistent").expect("tool name")],
    };
    let err = TurnPolicy::resolve(executable, Some(choice), None, Some(&tool_names(&["add"])))
        .expect_err("Specific naming a non-existent tool must fail locally");
    let msg = err.to_string();
    assert!(msg.contains("nonexistent"), "error names the typo: {msg}");
    assert!(
        !msg.contains("active_tools"),
        "a plain typo must not be blamed on active_tools: {msg}"
    );
}

#[test]
fn the_allowed_set_follows_the_choice() {
    let none = TurnPolicy::new(tool_names(&["add"]), Some(ToolChoice::None), None).expect("none");
    assert!(none.forbids_calls());
    assert!(!none.allows("add"));

    let auto = TurnPolicy::new(tool_names(&["add"]), None, Some("final".into())).expect("auto");
    assert!(!auto.forbids_calls());
    assert!(auto.allows("add") && auto.allows("final"));
    assert!(!auto.executable().contains("final"));
}

/// A persisted policy stores no allowed set: it is re-derived from the
/// choice, and a choice that cannot be honored is refused on load.
#[test]
fn a_persisted_policy_re_derives_its_allowed_set() {
    let choice = ToolChoice::Specific {
        function_names: vec![rig_core::message::ToolName::new("add").expect("tool name")],
    };
    let policy = TurnPolicy::new(tool_names(&["add", "sub"]), Some(choice), None).expect("policy");
    let json = serde_json::to_value(&policy).expect("serialize");
    assert!(json.get("allowed").is_none(), "{json}");
    let restored: TurnPolicy = serde_json::from_value(json.clone()).expect("deserialize");
    assert_eq!(restored, policy);
    assert_eq!(restored.allowed(), &tool_names(&["add"]));

    let mut stale = json;
    stale["executable"] = serde_json::json!(["sub"]);
    let error = serde_json::from_value::<TurnPolicy>(stale).expect_err("unadvertised choice");
    assert!(error.to_string().contains("add"), "{error}");
}

#[test]
fn the_context_reports_the_policy_it_was_built_from() {
    let policy = TurnPolicy::new(tool_names(&["add"]), Some(ToolChoice::None), None).expect("none");
    let call = rig_core::message::ToolCall::from_wire(
        "c1",
        rig_core::message::ToolFunction::new(
            rig_core::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let context =
        policy.invalid_call_context(&call, None, Vec::new(), false, policy.name_reason(&call));
    assert_eq!(context.tool_choice, Some(ToolChoice::None));
    assert_eq!(context.available_tools, vec!["add".to_string()]);
    assert!(context.allowed_tools.is_empty());
    assert_eq!(
        context.reason,
        InvalidToolCallReason::DisallowedByToolChoice
    );
}


#[test]
fn diagnostic_messages_keep_exact_message_serde_bytes() {
    let call = rig_core::message::ToolCall::from_wire(
        "real-call", rig_core::message::ToolFunction::new(
            ToolName::new("missing").expect("name"), serde_json::json!({"x": 1}),
        ),
    );
    let messages = vec![
        Message::user("question"),
        Message::Assistant(rig_core::message::AssistantMessage::new(vec![
            rig_core::message::AssistantContent::ToolCall(call),
        ])),
    ];
    let history: Vec<_> = messages.iter().cloned().map(DiagnosticHistoryEntry::Message).collect();
    let bytes = serde_json::to_string(&messages).expect("canonical history");
    assert_eq!(bytes, r#"[{"role":"user","content":[{"type":"text","text":"question"}]},{"role":"assistant","content":[{"type":"toolcall","id":{"provider":"real-call"},"function":{"name":"missing","arguments":{"x":1}}}]}]"#);
    assert_eq!(serde_json::to_string(&history).expect("diagnostic history"), bytes);
    let restored: Vec<DiagnosticHistoryEntry> = serde_json::from_str(&bytes).expect("message history");
    for (entry, message) in restored.iter().zip(&messages) {
        let DiagnosticHistoryEntry::Message(actual) = entry else { panic!("ordinary message") };
        assert_eq!(actual, message);
    }
}

#[test]
fn rejected_history_round_trips_without_becoming_a_message() {
    let id = rig_core::message::CallId::from_wire("real-call");
    for raw in ["null", "true", "42", "\"answer\"", "[1,2]", "{\"x\":1}"] {
        let call = DiagnosticCall::early_name(id.clone(), "missing".into(), raw.into());
        assert_eq!(call.arguments, DiagnosticArguments::Parsed(serde_json::from_str(raw).expect("JSON")));
        let entry = DiagnosticHistoryEntry::RejectedCall(call.clone());
        let bytes = serde_json::to_string(&entry).expect("entry");
        assert!(bytes.starts_with("{\"$rig_rejected_call\":"), "{bytes}");
        assert!(serde_json::from_str::<Message>(&bytes).is_err());
        let restored: DiagnosticHistoryEntry = serde_json::from_str(&bytes).expect("entry");
        let DiagnosticHistoryEntry::RejectedCall(actual) = restored else { panic!("rejected call") };
        assert_eq!(actual, call);
        assert_eq!(actual.raw.as_deref(), Some(raw));
        assert_eq!(actual.id, id);
    }
}

#[test]
fn diagnostic_parse_states_distinguish_blank_invalid_and_malformed() {
    let id = rig_core::message::CallId::from_wire("real-call");
    let blank = DiagnosticCall::early_name(id.clone(), "missing".into(), " \t".into());
    assert_eq!(blank.arguments, DiagnosticArguments::Blank);
    assert_eq!(blank.raw.as_deref(), Some(" \t"));
    let invalid = DiagnosticCall::early_name(id.clone(), "missing".into(), "{\"x\":".into());
    let DiagnosticArguments::Invalid { error } = &invalid.arguments else { panic!("invalid prefix") };
    assert_eq!(error, &serde_json::from_str::<serde_json::Value>("{\"x\":").expect_err("invalid JSON").to_string());
    let detail = rig_core::error::MalformedToolInput {
        name: "missing".into(), id,
        provider: Some(rig_core::error::ProviderCallEvidence {
            call_id: rig_core::message::ProviderCallId::new("real-call").expect("provider id"),
            item_id: Some("real-item".into()),
        }),
        raw: "[1,2]".into(), error: "literal decoder reason".into(),
    };
    let malformed = DiagnosticCall::malformed(&detail);
    assert_eq!(malformed.arguments, DiagnosticArguments::Malformed { error: detail.error.clone() });
    assert_eq!(malformed.provider, detail.provider);
    assert_eq!(malformed.raw.as_deref(), Some("[1,2]"));
    let bytes = serde_json::to_string(&malformed).expect("diagnostic");
    assert_eq!(serde_json::from_str::<DiagnosticCall>(&bytes).expect("diagnostic"), malformed);
}

#[test]
fn rejected_history_tag_cannot_accept_extra_message_fields() {
    let call = DiagnosticCall::early_name(rig_core::message::CallId::from_wire("real-call"), "missing".into(), "null".into());
    let value = serde_json::json!({"$rig_rejected_call":call,"role":"user","content":[]});
    assert!(serde_json::from_value::<DiagnosticHistoryEntry>(value).is_err());
}
