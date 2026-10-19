use super::*;
use crate::completion::{Accepts, ReplayTarget};
use crate::message::{
    AnsweredToolCall, Api, AssistantMessage, CustomToolCall, LocalCallId, ProviderItem, ToolCall,
    ToolFunction, ToolName, ToolResult, ToolResultContent,
};
use serde_json::json;

#[derive(Debug)]
struct Target;

impl ReplayTarget for Target {
    fn api(&self) -> Api {
        Api::from_static("interactions.test")
    }
    fn provider(&self) -> &str {
        "test"
    }
    fn model(&self) -> &str {
        "model"
    }
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }
    fn hosted_pair(
        &self,
        item: &serde_json::Value,
    ) -> Option<(crate::completion::history::Pairing, String)> {
        match item.get("type")?.as_str()? {
            "hosted" => Some((
                crate::completion::history::Pairing::Use,
                item.get("id")?.as_str()?.to_owned(),
            )),
            _ => None,
        }
    }
}

fn local() -> CallId {
    CallId::from(LocalCallId::new())
}

fn call(id: CallId) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        ToolCall::new(
            id,
            ToolFunction::new(ToolName::new("test").expect("tool name"), json!({})),
        ),
    )]))
}

fn result_part(id: CallId) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: id,
        name: ToolName::new("get_weather").expect("tool name"),
        answers: AnsweredToolCall::Function,
        content: vec![ToolResultContent::text(" exact result ")],
        is_error: false,
    })
}

fn result(id: CallId) -> Message {
    Message::User {
        content: vec![result_part(id)],
    }
}

#[test]
fn standalone_local_result_has_one_deterministic_alias_at_its_original_coordinate() {
    let id = local();
    let history = vec![
        Message::system("context"),
        Message::User {
            content: vec![
                UserContent::text("before"),
                result_part(id.clone()),
                UserContent::text("after"),
            ],
        },
    ];
    let original = serde_json::to_value(&history).expect("original canonical history");
    let first =
        WireIds::for_interactions(&history, &Target, "model").expect("standalone local result");
    let second =
        WireIds::for_interactions(&history, &Target, "model").expect("same deterministic plan");
    assert_eq!(first.of(1, 1), Some("tool-0"));
    assert_eq!(first.spell(1, 1).expect("result occurrence"), "tool-0");
    assert_eq!(second.of(1, 1), first.of(1, 1));
    assert_eq!(first.of(1, 0), None);
    assert_eq!(first.of(1, 2), None);
    assert!(
        id.provider().is_none(),
        "planning never gives a local identity provider provenance"
    );
    assert_eq!(
        serde_json::to_value(&history).expect("unchanged canonical history"),
        original
    );
    // The actual Interactions helper owns the emitted function_result.name proof.
}

#[test]
fn ordinary_required_id_target_still_refuses_the_same_standalone_local_result() {
    let history = vec![Message::user("before"), result(local())];
    let error =
        WireIds::for_target(&history, &Target, "model").expect_err("strict local orphan policy");
    assert_eq!(
        error,
        WireIdError::OrphanResult {
            message: 1,
            content: 0
        }
    );
    assert_eq!(
        error.to_string(),
        "local-only orphan tool result at message 1, content 0"
    );
    assert_eq!(
        WireIds::for_interactions(&history, &Target, "model")
            .expect("explicit Interactions policy")
            .of(1, 0),
        Some("tool-0")
    );
}

#[test]
fn later_genuine_and_hosted_handles_reserve_standalone_aliases_before_allocation() {
    let id = local();
    let history = vec![
        result(id.clone()),
        result(CallId::from_wire("tool-0")),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ProviderItem(
            ProviderItem::new(
                json!({"type":"hosted","id":"tool-1","future":[null,true]}),
                "provider",
            ),
        )])),
        call(id.clone()),
        result(id),
    ];
    let original = history.clone();
    let ids =
        WireIds::for_interactions(&history, &Target, "model").expect("reserve all future handles");
    assert_eq!(ids.of(0, 0), Some("tool-2"));
    assert_eq!(ids.of(1, 0), Some("tool-0"));
    assert_eq!(ids.of(2, 0), None);
    assert_eq!(ids.of(3, 0), Some("tool-3"));
    assert_eq!(ids.of(4, 0), ids.of(3, 0));
    assert_eq!(history, original);
}

#[test]
fn standalone_then_function_custom_and_reused_call_occurrences_keep_reversed_pairing() {
    let standalone = local();
    let function = ToolCall::new(
        local(),
        ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
    );
    let custom = CustomToolCall::new(local(), ToolName::new("patch").expect("name"), " raw ");
    let history = vec![
        result(standalone.clone()),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("before"),
            AssistantContent::ToolCall(function.clone()),
            AssistantContent::text("between"),
            AssistantContent::CustomToolCall(custom.clone()),
        ])),
        Message::tool_results(vec![custom.result(vec![]), function.result(vec![])]),
        call(standalone.clone()),
        result(standalone),
    ];
    let ids = WireIds::for_interactions(&history, &Target, "model")
        .expect("chronological mixed occurrences");
    assert_eq!(ids.of(0, 0), Some("tool-0"));
    assert_eq!(ids.of(1, 1), Some("tool-1"));
    assert_eq!(ids.of(1, 3), Some("tool-2"));
    assert_eq!(ids.of(2, 0), ids.of(1, 3));
    assert_eq!(ids.of(2, 1), ids.of(1, 1));
    assert_eq!(ids.of(3, 0), Some("tool-3"));
    assert_eq!(ids.of(4, 0), ids.of(3, 0));
    assert_eq!(ids.of(1, 0), None);
    assert_eq!(ids.of(1, 2), None);
}

#[test]
fn standalone_result_consumption_and_unanswered_call_reuse_keep_existing_typed_errors() {
    let id = local();
    let duplicate = vec![
        result(id.clone()),
        Message::User {
            content: vec![
                UserContent::text("before duplicate"),
                result_part(id.clone()),
            ],
        },
    ];
    let error = WireIds::for_interactions(&duplicate, &Target, "model")
        .expect_err("standalone result consumed once");
    assert_eq!(
        error,
        WireIdError::DuplicateResult {
            message: 1,
            content: 1
        }
    );
    assert_eq!(
        error.to_string(),
        "duplicate tool result at message 1, content 1"
    );
    let unanswered = vec![result(id.clone()), call(id.clone()), call(id.clone())];
    assert_eq!(
        WireIds::for_interactions(&unanswered, &Target, "model")
            .expect_err("unanswered second call"),
        WireIdError::Ambiguous {
            message: 2,
            content: 0
        }
    );
    let completed = vec![
        result(id.clone()),
        call(id.clone()),
        result(id.clone()),
        result(id),
    ];
    assert_eq!(
        WireIds::for_interactions(&completed, &Target, "model")
            .expect_err("completed result cannot repeat"),
        WireIdError::DuplicateResult {
            message: 3,
            content: 0
        }
    );
}

#[test]
fn genuine_orphan_spelling_and_equal_looking_typed_identity_remain_distinct() {
    let id = local();
    let genuine = CallId::from_wire(id.wire().into_owned());
    let history = vec![
        result(id.clone()),
        result(genuine.clone()),
        call(id.clone()),
        result(id),
    ];
    let ids = WireIds::for_interactions(&history, &Target, "model")
        .expect("typed identities stay distinct");
    assert_eq!(ids.of(0, 0), Some("tool-0"));
    assert_eq!(
        ids.spell(1, 0).expect("provider orphan exact"),
        genuine.wire()
    );
    assert_eq!(ids.of(2, 0), Some("tool-1"));
    assert_eq!(ids.of(3, 0), ids.of(2, 0));
    let provider_history = vec![result(CallId::from_wire(" exact / provider "))];
    for ids in [
        WireIds::for_target(&provider_history, &Target, "model"),
        WireIds::for_interactions(&provider_history, &Target, "model"),
    ] {
        assert_eq!(
            ids.expect("provider orphan allowed").of(0, 0),
            Some(" exact / provider ")
        );
    }
}

#[test]
fn conversations_without_standalone_local_results_keep_the_same_required_id_bytes() {
    let id = local();
    let history = vec![
        call(id.clone()),
        result(id.clone()),
        call(id.clone()),
        result(id),
        result(CallId::from_wire("tool-0")),
    ];
    let strict =
        WireIds::for_target(&history, &Target, "model").expect("ordinary valid conversation");
    let interactions =
        WireIds::for_interactions(&history, &Target, "model").expect("same ordinary conversation");
    for (message, expected) in [
        (0, "tool-1"),
        (1, "tool-1"),
        (2, "tool-2"),
        (3, "tool-2"),
        (4, "tool-0"),
    ] {
        assert_eq!(strict.of(message, 0), Some(expected));
        assert_eq!(interactions.of(message, 0), strict.of(message, 0));
    }
}

#[derive(Debug)]
struct NormalizedTarget;

impl ReplayTarget for NormalizedTarget {
    fn api(&self) -> Api {
        Target.api()
    }
    fn provider(&self) -> &str {
        Target.provider()
    }
    fn model(&self) -> &str {
        Target.model()
    }
    fn accepts(&self, model: &str) -> Accepts {
        Target.accepts(model)
    }
    fn map_options(
        &self,
        request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        Target.map_options(request, fields)
    }
    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        source: Option<&crate::message::Origin>,
    ) -> String {
        assert_eq!(model, "requested-model");
        assert!(source.is_none());
        format!("n_{id}")
    }
}

#[test]
fn standalone_and_call_aliases_use_the_same_target_normalization_and_future_reservation() {
    let id = local();
    let paired = local();
    let history = vec![
        result(id),
        call(paired.clone()),
        result(paired),
        result(CallId::from_wire("n_tool-0")),
    ];
    let ids = WireIds::for_interactions(&history, &NormalizedTarget, "requested-model")
        .expect("shared normalized allocator");
    assert_eq!(ids.of(0, 0), Some("n_tool-1"));
    assert_eq!(ids.of(1, 0), Some("n_tool-2"));
    assert_eq!(ids.of(2, 0), ids.of(1, 0));
    assert_eq!(ids.of(3, 0), Some("n_tool-0"));
    let paired = local();
    let ordinary = vec![call(paired.clone()), result(paired)];
    for ids in [
        WireIds::for_target(&ordinary, &NormalizedTarget, "requested-model"),
        WireIds::for_interactions(&ordinary, &NormalizedTarget, "requested-model"),
    ] {
        let ids = ids.expect("ordinary single occurrence");
        assert_eq!(ids.of(0, 0), Some("n_tool-0"));
        assert_eq!(ids.of(1, 0), ids.of(0, 0));
    }
}
