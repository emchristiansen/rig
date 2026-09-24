use serde::{Deserialize, Serialize};

use super::{AdditionalParams, Message, Reasoning, ReasoningContent, Text, ToolResultContent};

mod vec_content_serde {
    use super::super::{AssistantContent, Message, UserContent};

    #[test]
    fn message_content_still_serializes_as_a_plain_sequence() {
        // The removed container serialized as a bare sequence, which is why
        // this migration changes no persisted history and no recorded
        // provider fixture. Pin the wire shape so that stays true.
        let message = Message::User {
            content: vec![UserContent::text("hi")],
        };
        let json = serde_json::to_value(&message).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            })
        );
    }

    #[test]
    fn message_content_round_trips_byte_identically() {
        let message = Message::Assistant {
            id: Some("msg_1".to_owned()),
            content: vec![AssistantContent::text("hello")],
        };
        let encoded = serde_json::to_string(&message).expect("serialize");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(
            serde_json::to_string(&decoded).expect("re-serialize"),
            encoded
        );
    }

    #[test]
    fn an_empty_content_array_now_deserializes() {
        // The container's `Deserialize` implemented only `visit_seq` and
        // rejected `[]`. That is the single input whose behaviour this
        // migration changes: it was an error, and it is now an empty list.
        let message: Message =
            serde_json::from_value(serde_json::json!({"role": "user", "content": []}))
                .expect("an empty content list is representable now");
        let Message::User { content } = message else {
            panic!("expected a user message");
        };
        assert!(content.is_empty());
    }
}

#[test]
fn reasoning_constructors_and_accessors_work() {
    let single = Reasoning::new("think");
    assert_eq!(single.first_text(), Some("think"));
    assert_eq!(single.first_signature(), None);

    let signed = Reasoning::new_with_signature("signed", Some("sig-1".to_string()));
    assert_eq!(signed.first_text(), Some("signed"));
    assert_eq!(signed.first_signature(), Some("sig-1"));

    let multi = Reasoning::multi(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(multi.display_text(), "a\nb");
    assert_eq!(multi.first_text(), Some("a"));

    let redacted = Reasoning::redacted("redacted-value");
    assert_eq!(redacted.display_text(), "redacted-value");
    assert_eq!(redacted.first_text(), None);

    let encrypted = Reasoning::encrypted("enc");
    assert_eq!(encrypted.encrypted_content(), Some("enc"));
    assert_eq!(encrypted.display_text(), "");

    let summaries = Reasoning::summaries(vec!["s1".to_string(), "s2".to_string()]);
    assert_eq!(summaries.display_text(), "s1\ns2");
    assert_eq!(summaries.encrypted_content(), None);
}

#[test]
fn reasoning_content_serde_roundtrip() {
    let variants = vec![
        ReasoningContent::Text {
            text: "plain".to_string(),
            signature: Some("sig".to_string()),
        },
        ReasoningContent::Encrypted("opaque".to_string()),
        ReasoningContent::Redacted {
            data: "redacted".to_string(),
        },
        ReasoningContent::Summary("summary".to_string()),
    ];

    for variant in variants {
        let json = serde_json::to_string(&variant).expect("serialize");
        let roundtrip: ReasoningContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, variant);
    }
}

#[test]
fn system_message_constructor_and_serde_roundtrip() {
    let message = Message::system("You are concise.");

    match &message {
        Message::System { content } => assert_eq!(content, "You are concise."),
        _ => panic!("Expected system message"),
    }

    let json = serde_json::to_string(&message).expect("serialize");
    let roundtrip: Message = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtrip, message);
}

#[test]
fn current_schema_tool_call_json_round_trips_without_provider_promotion() {
    // A minted handle with no provider must stay provider-less —
    // nothing in the round trip may invent provider provenance.
    let call = super::ToolCall::new(
        super::ToolCallId::new("minted-handle").expect("non-empty"),
        super::ToolFunction {
            name: "add".to_string(),
            namespace: None,
            arguments: serde_json::json!({}),
        },
    );

    let json = serde_json::to_value(&call).expect("serialize");
    assert!(json.get("call_id").is_none());
    let roundtrip: super::ToolCall = serde_json::from_value(json).expect("deserialize");
    assert_eq!(roundtrip.provider, None);
    assert_eq!(roundtrip, call);
}

#[test]
fn empty_params_canonicalize_to_none_in_both_serde_directions() {
    // The `AdditionalParams` contract, pinned where it lives. One
    // fixture, every direction: canonicalization, round-trip, tolerance,
    // and rejection.

    // An explicit `{}` or `null` decodes as `None` exactly like an
    // absent field.
    for empty_spelling in [serde_json::json!({}), serde_json::Value::Null] {
        let text: Text = serde_json::from_value(
            serde_json::json!({"text": "x", "additional_params": empty_spelling}),
        )
        .expect("deserialize");
        assert_eq!(text.additional_params, None);
    }

    // Data survives a round trip value-identically, and `Some` params
    // always carry data — `AdditionalParams` has no empty value, so the
    // old uncanonicalized-`Some({})` hazard is unrepresentable rather
    // than tolerated.
    let text: Text = serde_json::from_value(
        serde_json::json!({"text": "x", "additional_params": {"citations": [1]}}),
    )
    .expect("deserialize");
    assert_eq!(
        text.additional_params,
        AdditionalParams::from_entries([("citations", serde_json::json!([1]))])
    );
    assert_eq!(
        text.additional_params
            .as_ref()
            .and_then(|params| params.get("citations")),
        Some(&serde_json::json!([1]))
    );
    let round: Text = serde_json::from_value(serde_json::to_value(&text).expect("serialize"))
        .expect("round trip");
    assert_eq!(round, text);

    // The empty map canonicalizes to `None` at the constructor, so it
    // never reaches serialization at all.
    assert_eq!(AdditionalParams::new(serde_json::Map::new()), None);
    assert_eq!(
        AdditionalParams::try_from_value(serde_json::json!({})).expect("object"),
        None
    );

    // An unknown key on the block itself is tolerated and dropped —
    // never an error, never captured into params — so histories written
    // by a newer rig (or 0.41 flattened extras that were never
    // re-nested) still load; MIGRATING's strict-decode recipe is the
    // opt-in detector for the dropped keys.
    let tolerant: Text = serde_json::from_value(
        serde_json::json!({"text": "x", "citations": ["stray"], "future_field": 1}),
    )
    .expect("unknown keys on a block must not fail the decode");
    assert_eq!(tolerant.text, "x");
    assert_eq!(tolerant.additional_params, None);

    // Extras are a keyed namespace: a non-object carrier (the shape a
    // mis-firing migration script writes) is malformed data and fails
    // loudly instead of loading as a phantom annotation no extractor
    // can read.
    for malformed in [serde_json::json!([]), serde_json::json!("title")] {
        let err = serde_json::from_value::<Text>(
            serde_json::json!({"text": "x", "additional_params": malformed}),
        )
        .expect_err("non-object params must be a decode error");
        assert!(
            err.to_string().contains("must be a JSON object"),
            "unexpected error: {err}"
        );
        assert!(
            AdditionalParams::try_from_value(serde_json::json!([])).is_err(),
            "try_from_value must hand a non-object back, not swallow it"
        );
    }
}

#[test]
fn round_trip_diff_recipe_detects_every_dropped_key() {
    // Pins MIGRATING's opt-in verification recipe: the runtime load
    // path tolerates unknown keys (see the tolerance case in
    // `empty_params_canonicalize_to_none_in_both_serde_directions`),
    // and a migration script detects what tolerance dropped by loading,
    // re-serializing, and asking `keys_lost_in_round_trip` — a
    // serde_ignored-based recipe cannot serve here, because the
    // internally tagged enums buffer their content and hide ignored
    // keys from its callback.
    let migrated = serde_json::json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "cited", "citations": ["not re-nested"]},
            {"type": "text", "text": "clean",
             "additional_params": {"citations": ["re-nested"]}},
        ],
    });
    let loaded: Message =
        serde_json::from_value(migrated.clone()).expect("tolerant decode must succeed");
    let reserialized = serde_json::to_value(&loaded).expect("serialize");
    assert_eq!(
        super::keys_lost_in_round_trip(&migrated, &reserialized),
        vec!["content.0.citations".to_string()],
        "every dropped key must be reported by path, and only dropped keys \
             — writer-added defaults are not differences"
    );

    // A fully re-nested history survives whole: the recipe's success
    // condition is an empty list. MIGRATING's blessed
    // `"additional_params": {}` spelling canonicalizes to absence and
    // must not read as a loss.
    let clean = serde_json::json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "clean",
             "additional_params": {"citations": ["re-nested"]}},
            {"type": "text", "text": "mechanically migrated",
             "additional_params": {}},
        ],
    });
    let loaded: Message = serde_json::from_value(clean.clone()).expect("decode");
    let reserialized = serde_json::to_value(&loaded).expect("serialize");
    assert_eq!(
        super::keys_lost_in_round_trip(&clean, &reserialized),
        Vec::<String>::new(),
        "clean history must survive the round trip whole"
    );
}

#[test]
fn legacy_call_id_key_cannot_recover_an_untagged_identity() {
    let legacy = serde_json::json!({
        "id": "fc_123",
        "call_id": "call_abc",
        "function": {"name": "add", "arguments": {"x": 1}},
    });
    assert!(serde_json::from_value::<super::ToolCall>(legacy).is_err());
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ExecutorLikeResponse {
    output: serde_json::Value,
    logs: Vec<String>,
    execution_time_ms: u64,
}

#[test]
fn tool_result_content_decodes_structured_and_legacy_json() {
    let response = ExecutorLikeResponse {
        output: serde_json::json!({"answer": 42}),
        logs: vec!["computed".to_string()],
        execution_time_ms: 7,
    };
    let value = serde_json::to_value(&response).expect("serialize response");

    let structured = ToolResultContent::json(value.clone());
    assert_eq!(structured.as_json(), Some(&value));
    assert_eq!(structured.as_text(), None);
    assert_eq!(
        structured
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode structured response"),
        response
    );

    let legacy_json = value.to_string();
    let legacy_text = ToolResultContent::Text(Text::new(legacy_json.clone()));
    assert_eq!(legacy_text.as_text(), Some(legacy_json.as_str()));
    assert_eq!(legacy_text.as_json(), None);
    assert_eq!(
        legacy_text
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode legacy response"),
        response
    );

    let image = ToolResultContent::image_url("https://example.com/result.png", None, None);
    let image_error = image.deserialize_json::<ExecutorLikeResponse>();
    assert!(image_error.is_err());
    if let Err(error) = image_error {
        assert_eq!(
            error.to_string(),
            "cannot decode image tool-result content as JSON"
        );
    }
}

/// Generated positions occupy a namespace disjoint from explicit handles.
#[test]
fn missing_call_id_normalization_separates_namespaces_and_preserves_metadata() {
    use super::{AssistantContent, ToolCall, ToolFunction, normalize_missing_tool_call_ids};
    use serde_json::json;
    let mut content = vec![
        AssistantContent::text("not a call position"),
        AssistantContent::ToolCall(
            ToolCall::from_wire("", ToolFunction::new("same".into(), json!({"n":1})))
                .with_signature(Some("signed".into()))
                .with_additional_params(Some(json!({"opaque":true}))),
        ),
        AssistantContent::tool_call("tool-0", "same", json!({"n":2})),
        AssistantContent::tool_call("tool-1", "same", json!({"n":3})),
        AssistantContent::tool_call("", "same", json!({"n":4})),
        AssistantContent::tool_call("tool-0", "same", json!({"n":5})),
    ];
    normalize_missing_tool_call_ids(&mut content);
    let calls: Vec<_> = content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.iter().map(|call| call.id.clone()).collect::<Vec<_>>(),
        [
            super::ToolCallId::minted(0),
            super::ToolCallId::new("tool-0").expect("explicit"),
            super::ToolCallId::new("tool-1").expect("explicit"),
            super::ToolCallId::minted(3),
            super::ToolCallId::new("tool-0").expect("explicit"),
        ]
    );
    assert!(calls[0].provider.is_none());
    assert_eq!(calls[0].signature.as_deref(), Some("signed"));
    assert_eq!(calls[0].additional_params, Some(json!({"opaque":true})));
    assert_eq!(calls[1].provider.as_ref().unwrap().call_id, "tool-0");
    let first = content.clone();
    normalize_missing_tool_call_ids(&mut content);
    assert_eq!(content, first, "normalization is stable when repeated");
}

/// Internal identity law: provider text must not impersonate a generated key.
#[test]
fn typed_tool_identity_separates_generated_and_explicit_keys() {
    use super::ToolCallId;
    let generated = ToolCallId::minted(0);
    let explicit = ToolCallId::new("tool-0").expect("nonempty explicit ID");
    assert_ne!(generated, explicit);
    let keys = std::collections::HashSet::from([generated.clone(), explicit.clone()]);
    assert_eq!(keys.len(), 2);
    let generated_json = serde_json::to_value(&generated).expect("serialize generated ID");
    let explicit_json = serde_json::to_value(&explicit).expect("serialize explicit ID");
    assert_ne!(generated_json, explicit_json);
    for id in [generated, explicit] {
        let json = serde_json::to_string(&id).expect("serialize ID");
        assert_eq!(
            serde_json::from_str::<ToolCallId>(&json).expect("decode ID"),
            id
        );
    }
}

/// Assembly keys with equal display text still name different generated origins.
#[test]
fn typed_tool_identity_preserves_the_assembly_key_discriminant() {
    use super::ToolCallId;
    use crate::streaming::{BlockId, SyntheticIds};
    assert_ne!(
        ToolCallId::from_block(&BlockId::wire("tool-0")),
        ToolCallId::from_block(&SyntheticIds::tool().mint()),
    );
}

/// Legacy untagged IDs cannot tell generated handles from explicit handles.
#[test]
fn typed_tool_identity_rejects_legacy_untagged_serialization() {
    assert!(serde_json::from_str::<super::ToolCallId>(r#""tool-0""#).is_err());
}

/// Namespaces, custom calls, and the call kind a result answers — the typed
/// tool-call surface — pinned as serialized bytes, so serde drift in the
/// canonical format is a visible test failure.
mod typed_tool_calls {
    use super::super::{
        AnsweredToolCall, AssistantContent, CustomToolCall, ToolCall, ToolFunction, ToolResult,
        ToolResultContent, UndispatchableToolCall, UnrepresentableToolCall, UserContent,
        json_only_wire_tool_call, name_dispatchable_call,
    };

    const FUNCTION_CALL: &str = r#"{"type":"toolcall","id":{"origin":"explicit","id":"call_1"},"provider":{"call_id":"call_1","item_id":"fc_1"},"function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null}"#;
    const NAMESPACED_CALL: &str = r#"{"type":"toolcall","id":{"origin":"explicit","id":"call_1"},"provider":{"call_id":"call_1","item_id":"fc_1"},"function":{"name":"add","namespace":"math","arguments":{"x":1}},"signature":null,"additional_params":null}"#;
    const CUSTOM_CALL: &str = r#"{"type":"customtoolcall","id":{"origin":"explicit","id":"call_2"},"provider":{"call_id":"call_2","item_id":"ctc_2"},"name":"apply_patch","namespace":"repo","input":"*** Begin Patch\n{\"x\": 1}"}"#;
    const FUNCTION_RESULT: &str = r#"{"type":"toolresult","call":{"origin":"explicit","id":"call_1"},"provider":{"call_id":"call_1","item_id":"fc_1"},"name":"add","answers":"function","content":[{"type":"text","text":"2"}]}"#;
    const CUSTOM_RESULT: &str = r#"{"type":"toolresult","call":{"origin":"explicit","id":"call_2"},"provider":{"call_id":"call_2","item_id":"ctc_2"},"name":"apply_patch","answers":"custom","content":[{"type":"text","text":"applied"}]}"#;

    fn function_call() -> AssistantContent {
        AssistantContent::tool_call_with_call_id(
            "fc_1",
            "call_1".to_owned(),
            "add",
            serde_json::json!({"x": 1}),
        )
    }

    fn namespaced_call() -> AssistantContent {
        AssistantContent::tool_call_with_namespace(
            "fc_1",
            "call_1".to_owned(),
            "add",
            Some("math".to_owned()),
            serde_json::json!({"x": 1}),
        )
    }

    /// Input that happens to be valid JSON after a prefix line: it must stay
    /// the verbatim string it arrived as.
    fn custom_call() -> CustomToolCall {
        CustomToolCall::from_dual_wire(
            "ctc_2",
            "call_2",
            "apply_patch",
            Some("repo".to_owned()),
            "*** Begin Patch\n{\"x\": 1}",
        )
    }

    fn assert_pinned<T>(value: &T, pinned: &str)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        assert_eq!(serde_json::to_string(value).expect("serialize"), pinned);
        let decoded: T = serde_json::from_str(pinned).expect("the pinned bytes decode");
        assert_eq!(&decoded, value, "the pinned bytes decode to the same value");
    }

    #[test]
    fn a_function_call_keeps_the_upstream_bytes() {
        assert_pinned(&function_call(), FUNCTION_CALL);
    }

    #[test]
    fn a_namespaced_call_carries_its_namespace_beside_the_name() {
        assert_pinned(&namespaced_call(), NAMESPACED_CALL);
    }

    #[test]
    fn a_custom_call_is_its_own_variant_with_verbatim_input() {
        let call = AssistantContent::CustomToolCall(custom_call());
        assert_pinned(&call, CUSTOM_CALL);
        let AssistantContent::CustomToolCall(decoded) =
            serde_json::from_str::<AssistantContent>(CUSTOM_CALL).expect("decode")
        else {
            panic!("a custom call decodes as the custom variant");
        };
        assert_eq!(decoded.input, "*** Begin Patch\n{\"x\": 1}");
    }

    #[test]
    fn a_custom_call_whose_input_is_json_is_still_a_custom_call() {
        let call = AssistantContent::custom_tool_call("ctc_3", "call_3", "t", None, r#"{"a":1}"#);
        let json = serde_json::to_value(&call).expect("serialize");
        assert_eq!(json["type"], "customtoolcall");
        assert_eq!(
            json["input"],
            serde_json::json!(r#"{"a":1}"#),
            "a string, not an object"
        );
        let decoded: AssistantContent = serde_json::from_value(json).expect("decode");
        assert_eq!(decoded, call);
    }

    #[test]
    fn results_state_the_kind_they_answer() {
        let function = UserContent::tool_result_with_call_id(
            "fc_1",
            "call_1",
            "add",
            vec![ToolResultContent::text("2")],
        );
        assert_pinned(&function, FUNCTION_RESULT);

        let custom = UserContent::tool_result_for_custom_call(
            &custom_call(),
            vec![ToolResultContent::text("applied")],
        );
        assert_pinned(&custom, CUSTOM_RESULT);
        assert_eq!(
            custom,
            UserContent::tool_result_answering(
                "ctc_2",
                "call_2",
                "apply_patch",
                AnsweredToolCall::Custom,
                vec![ToolResultContent::text("applied")],
            ),
            "the relay constructor states what the call-derived one derives"
        );
    }

    #[test]
    fn a_result_that_does_not_say_what_it_answers_is_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(FUNCTION_RESULT).expect("json");
        json.as_object_mut().expect("object").remove("answers");
        let error =
            serde_json::from_value::<UserContent>(json.clone()).expect_err("`answers` is required");
        assert!(error.to_string().contains("answers"), "{error}");
        json.as_object_mut().expect("object").remove("type");
        assert!(serde_json::from_value::<ToolResult>(json).is_err());
    }

    #[test]
    fn the_answered_kind_derives_from_the_held_call() {
        let AssistantContent::ToolCall(call) = function_call() else {
            unreachable!()
        };
        let UserContent::ToolResult(result) =
            UserContent::tool_result_for_call(&call, vec![ToolResultContent::text("2")])
        else {
            unreachable!()
        };
        assert_eq!(result.answers, AnsweredToolCall::Function);
        assert_eq!(result.call, call.id);
        assert_eq!(custom_call().answered_by(), AnsweredToolCall::Custom);
    }

    #[test]
    fn an_absent_namespace_is_omitted_and_every_present_spelling_is_kept() {
        let bare = ToolFunction::new("add".to_owned(), serde_json::json!({}));
        let json = serde_json::to_value(&bare).expect("serialize");
        assert!(json.get("namespace").is_none(), "{json}");
        for spelling in ["", "functions", "math"] {
            let qualified = bare.clone().with_namespace(Some(spelling.to_owned()));
            let json = serde_json::to_value(&qualified).expect("serialize");
            assert_eq!(
                json["namespace"], spelling,
                "no spelling normalizes to absence"
            );
            assert_eq!(
                serde_json::from_value::<ToolFunction>(json).expect("decode"),
                qualified
            );
        }
    }

    #[test]
    fn json_only_wires_refuse_custom_and_namespaced_calls_by_name() {
        let AssistantContent::ToolCall(plain) = function_call() else {
            unreachable!()
        };
        assert_eq!(
            json_only_wire_tool_call("Test Wire", &function_call()),
            Ok(Some(&plain))
        );
        assert_eq!(
            json_only_wire_tool_call("Test Wire", &AssistantContent::text("hi")),
            Ok(None)
        );
        // `""` and `"functions"` are refused too: no JSON-only wire here has
        // evidence that it treats either as its default namespace.
        for spelling in ["", "functions", "math"] {
            let call =
                AssistantContent::ToolCall(plain.clone().with_namespace(Some(spelling.to_owned())));
            assert_eq!(
                json_only_wire_tool_call("Test Wire", &call),
                Err(UnrepresentableToolCall::Namespace {
                    wire: "Test Wire",
                    name: "add".to_owned(),
                    namespace: spelling.to_owned(),
                })
            );
        }
        let refusal = json_only_wire_tool_call(
            "Test Wire",
            &AssistantContent::CustomToolCall(custom_call()),
        )
        .expect_err("a custom call is refused");
        assert_eq!(
            refusal.to_string(),
            "Test Wire carries JSON tool arguments only, so custom tool call `apply_patch` and its raw input cannot be sent on it"
        );
        // Refusals are request failures in the shared error vocabulary.
        let provider: crate::error::ProviderError = crate::error::EncodeError::from(refusal).into();
        assert_eq!(provider.kind(), crate::error::ErrorKind::Request);
    }

    #[test]
    fn name_keyed_dispatch_refuses_namespaced_and_custom_calls() {
        let AssistantContent::ToolCall(plain) = function_call() else {
            unreachable!()
        };
        assert_eq!(name_dispatchable_call(&function_call()), Ok(Some(&plain)));
        assert_eq!(
            name_dispatchable_call(&namespaced_call()),
            Err(UndispatchableToolCall::Namespaced {
                id: plain.id.clone(),
                name: "add".to_owned(),
                namespace: "math".to_owned(),
            })
        );
        let json_input =
            AssistantContent::custom_tool_call("ctc_3", "call_3", "t", None, r#"{"a":1}"#);
        assert!(
            matches!(
                name_dispatchable_call(&json_input),
                Err(UndispatchableToolCall::Custom { .. })
            ),
            "custom input that parses as JSON is still refused"
        );
        assert_eq!(
            super::super::first_undispatchable_call(&[
                AssistantContent::text("x"),
                function_call(),
                AssistantContent::CustomToolCall(custom_call()),
            ]),
            Some(UndispatchableToolCall::Custom {
                id: custom_call().id,
                name: "apply_patch".to_owned(),
                namespace: Some("repo".to_owned()),
            })
        );
    }

    #[test]
    fn missing_ids_are_minted_across_both_call_kinds() {
        let mut content = vec![
            AssistantContent::ToolCall(ToolCall::from_wire(
                "",
                ToolFunction::new("a".to_owned(), serde_json::json!({})),
            )),
            AssistantContent::custom_tool_call("", "", "b", None, "raw"),
        ];
        super::super::normalize_missing_tool_call_ids(&mut content);
        let ids: Vec<_> = content
            .iter()
            .map(|item| match item {
                AssistantContent::ToolCall(call) => call.id.clone(),
                AssistantContent::CustomToolCall(call) => call.id.clone(),
                _ => unreachable!(),
            })
            .collect();
        assert_ne!(ids[0], ids[1], "two id-less calls never share a handle");
    }
}
