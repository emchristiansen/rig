//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover their use.

use super::*;
use crate::completion::{Accepts, ReplayTarget};
use crate::message::{Api, CustomToolCall, ToolCall, ToolFunction, ToolName, ToolResult};

#[derive(Debug)]
struct Target;

impl ReplayTarget for Target {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> Api {
        Api::from_static("test.api")
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
}

fn call(id: CallId) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(vec![
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(
                crate::message::ToolName::new("test").expect("tool name"),
                serde_json::json!({}),
            ),
        )),
    ]))
}

fn result(id: CallId) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: id,
            name: crate::message::ToolName::new("possibly_repaired").expect("tool name"),
            answers: crate::message::AnsweredToolCall::Function,
            content: vec![crate::message::ToolResultContent::text("")],
        })],
    }
}

#[test]
fn a_rig_issued_id_is_one_alias_that_no_provider_id_takes() {
    let issued = CallId::from_wire("");
    let later = CallId::from_wire("");
    let history = vec![
        call(CallId::from_wire("tool-0")),
        call(issued.clone()),
        result(issued.clone()),
        call(later.clone()),
    ];
    let ids = WireIds::for_target(&history, &Target, "model").expect("valid history");
    assert_eq!(ids.of(0, 0), Some("tool-0"));
    assert_eq!(ids.of(1, 0), Some("tool-1"));
    assert_eq!(ids.of(3, 0), Some("tool-2"));
}

#[test]
fn custom_calls_take_aliases_in_call_order_before_reversed_results() {
    let custom = CustomToolCall::new(
        CallId::from_wire(""),
        ToolName::new("patch").expect("tool name"),
        "raw",
    );
    let function = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(
            ToolName::new("test").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let provider = CustomToolCall::new(
        CallId::from_wire("tool-0"),
        ToolName::new("patch").expect("tool name"),
        "later",
    );
    let custom_result = custom.result(vec![]);
    let function_result = function.result(vec![]);
    let history = vec![
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::text("before"),
            AssistantContent::CustomToolCall(custom.clone()),
            AssistantContent::text("between"),
            AssistantContent::ToolCall(function.clone()),
        ])),
        Message::User {
            content: vec![
                UserContent::ToolResult(function_result.clone()),
                UserContent::ToolResult(custom_result.clone()),
            ],
        },
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::CustomToolCall(provider.clone()),
        ])),
    ];
    let before = serde_json::to_value(&history).expect("serialize the history");
    let ids = WireIds::for_target(&history, &Target, "model").expect("valid history");
    assert_eq!(ids.of(0, 1), Some("tool-1"));
    assert_eq!(ids.of(0, 3), Some("tool-2"));
    assert_eq!(ids.of(2, 0), Some("tool-0"));
    assert_eq!(ids.of(1, 1), ids.of(0, 1));
    assert_eq!(ids.of(1, 0), ids.of(0, 3));
    assert_eq!(ids.spell(1, 1).expect("planned custom result"), "tool-1");
    assert_eq!(ids.spell(1, 0).expect("planned function result"), "tool-2");
    assert_eq!(
        before,
        serde_json::to_value(&history).expect("serialize the unchanged history"),
    );
    let repeated = WireIds::for_target(&history, &Target, "model").expect("valid history");
    for (message, content) in [(0, 1), (0, 3), (2, 0), (1, 0), (1, 1)] {
        assert_eq!(ids.of(message, content), repeated.of(message, content));
    }
}

#[test]
fn a_custom_call_without_a_result_still_reserves_its_ordered_alias() {
    let unanswered = CustomToolCall::new(
        CallId::from_wire(""),
        ToolName::new("patch").expect("tool name"),
        "raw",
    );
    let later = CallId::from_wire("");
    let provider_custom = CustomToolCall::new(
        CallId::from_wire("tool-0"),
        ToolName::new("patch").expect("tool name"),
        "later",
    );
    let provider_function = CallId::from_wire("tool-2");
    let history = vec![
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::CustomToolCall(unanswered.clone()),
        ])),
        call(later.clone()),
        result(later.clone()),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::CustomToolCall(provider_custom.clone()),
        ])),
        call(provider_function.clone()),
    ];
    let ids = WireIds::for_target(&history, &Target, "model").expect("valid history");
    assert_eq!(ids.of(0, 0), Some("tool-1"));
    assert_eq!(ids.of(1, 0), Some("tool-3"));
    assert_eq!(ids.of(3, 0), Some("tool-0"));
    assert_eq!(ids.of(4, 0), Some("tool-2"));
    let aliases = [ids.of(0, 0), ids.of(1, 0), ids.of(3, 0), ids.of(4, 0)];
    assert_eq!(aliases.into_iter().collect::<HashSet<_>>().len(), 4);
}

#[derive(Debug)]
struct HostedTarget;

impl ReplayTarget for HostedTarget {
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
    fn hosted_pair(
        &self,
        item: &serde_json::Value,
    ) -> Option<(crate::completion::history::Pairing, String)> {
        use crate::completion::history::Pairing;
        match item.get("type")?.as_str()? {
            "server_tool_use" => Some((Pairing::Use, item.get("id")?.as_str()?.to_owned())),
            "web_search_tool_result" => Some((
                Pairing::Result,
                item.get("tool_use_id")?.as_str()?.to_owned(),
            )),
            _ => None,
        }
    }
}

fn carried_hosted() -> Message {
    use crate::message::{CarriedPayload, CarriedText, CarriedTextContent};
    Message::Assistant(crate::message::AssistantMessage::new(vec![
        AssistantContent::CarriedText(CarriedText::new(
            CarriedTextContent::Opaque(String::new()),
            CarriedPayload::new(
                serde_json::json!({"type":"server_tool_use","id":"tool-0","input":{"query":"exact"}}),
                Some("anthropic".to_owned()),
            ),
        )),
        AssistantContent::CarriedText(CarriedText::new(
            CarriedTextContent::Opaque(String::new()),
            CarriedPayload::new(
                serde_json::json!({"type":"web_search_tool_result","tool_use_id":"tool-0","content":[]}),
                Some("anthropic".to_owned()),
            ),
        )),
    ]))
}

#[test]
fn local_aliases_reserve_carried_hosted_handles_and_leave_payloads_exact() {
    let local = CallId::from(crate::message::LocalCallId::new());
    let history = vec![call(local.clone()), carried_hosted(), result(local.clone())];
    let before = serde_json::to_value(&history).expect("serialize the original history");
    let ids = WireIds::for_target(&history, &HostedTarget, "model").expect("valid history");
    assert_eq!(ids.of(0, 0), Some("tool-1"));
    assert_eq!(ids.spell(0, 0).expect("planned local call"), "tool-1");
    assert_eq!(
        serde_json::to_value(&history).expect("serialize exact payloads"),
        before
    );
    assert_eq!(
        WireIds::for_target(&history, &Target, "model")
            .expect("valid history")
            .of(0, 0),
        Some("tool-0")
    );
}

#[test]
fn carried_hosted_reservation_never_rewrites_a_provider_issued_id() {
    let provider = CallId::from_wire("tool-0");
    let history = vec![
        call(provider.clone()),
        carried_hosted(),
        result(provider.clone()),
    ];
    let ids = WireIds::for_target(&history, &HostedTarget, "model").expect("valid history");
    assert_eq!(ids.of(0, 0), Some("tool-0"));
    assert_eq!(ids.spell(0, 0).expect("planned provider call"), "tool-0");
}

/// The protected adapter fixture's three calls, reversed results across text,
/// and later reuse of the same local identity have distinct occurrence slots.
#[test]
fn completed_local_reuse_keeps_three_distinct_calls_and_reversed_result_pairing() {
    let local = CallId::from(crate::message::LocalCallId::new());
    let genuine = CallId::from_wire(local.wire().into_owned());
    let history = vec![
        Message::system("system"),
        call(local.clone()),
        call(genuine.clone()),
        result(genuine.clone()),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::text("between results"),
        ])),
        result(local.clone()),
        call(local.clone()),
        result(local.clone()),
    ];
    let before = serde_json::to_value(&history).expect("original history");
    let ids = WireIds::for_target(&history, &Target, "model").expect("completed reuse is valid");
    let calls = [
        ids.spell(1, 0).expect("first local"),
        ids.spell(2, 0).expect("genuine"),
        ids.spell(6, 0).expect("reused local"),
    ];
    assert_eq!(calls[0], "tool-0");
    assert_eq!(calls[1], genuine.wire());
    assert_eq!(calls[2], "tool-1");
    assert_eq!(calls.iter().collect::<HashSet<_>>().len(), 3);
    assert_eq!(
        [
            ids.spell(3, 0).expect("genuine result first"),
            ids.spell(5, 0).expect("first local result second"),
            ids.spell(7, 0).expect("reused local result"),
        ],
        [calls[1].clone(), calls[0].clone(), calls[2].clone()]
    );
    assert_eq!(ids.of(4, 0), None);
    assert_eq!(
        serde_json::to_value(&history).expect("unchanged history"),
        before
    );
    let repeated = WireIds::for_target(&history, &Target, "model").expect("deterministic plan");
    for message in [1, 2, 3, 5, 6, 7] {
        assert_eq!(ids.of(message, 0), repeated.of(message, 0));
    }
}

#[test]
fn local_reuse_after_a_custom_result_gets_a_new_occurrence_alias() {
    let local = CallId::from(crate::message::LocalCallId::new());
    let custom = CustomToolCall::new(
        local.clone(),
        ToolName::new("patch").expect("name"),
        " raw ",
    );
    let history = vec![
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::CustomToolCall(custom.clone()),
        ])),
        Message::tool_results(vec![
            custom.result(vec![crate::message::ToolResultContent::text("ok")]),
        ]),
        call(local.clone()),
        result(local),
    ];
    let ids = WireIds::for_target(&history, &Target, "model").expect("completed custom occurrence");
    assert_eq!(ids.of(0, 0), Some("tool-0"));
    assert_eq!(ids.of(1, 0), ids.of(0, 0));
    assert_eq!(ids.of(2, 0), Some("tool-1"));
    assert_eq!(ids.of(3, 0), ids.of(2, 0));
}

#[test]
fn unanswered_local_and_provider_reuse_refuse_at_the_second_call_coordinate() {
    for id in [
        CallId::from_wire(""),
        CallId::from_wire("exact-provider-id"),
    ] {
        let history = vec![
            call(id.clone()),
            Message::Assistant(crate::message::AssistantMessage::new(vec![
                AssistantContent::text("before"),
                AssistantContent::CustomToolCall(CustomToolCall::new(
                    id,
                    ToolName::new("patch").expect("name"),
                    "raw",
                )),
            ])),
        ];
        let error = WireIds::for_target(&history, &Target, "model").expect_err("unanswered reuse");
        assert_eq!(
            error,
            WireIdError::Ambiguous {
                message: 1,
                content: 1
            }
        );
        assert_eq!(
            error.to_string(),
            "ambiguous tool call identity at message 1, content 1"
        );
        let encoded = crate::error::EncodeError::request(error);
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&encoded);
        let mut found = false;
        while let Some(error) = source {
            found |= error.downcast_ref::<WireIdError>().is_some();
            source = error.source();
        }
        assert!(found, "the request error retains the typed pairing source");
    }
}

#[test]
fn provider_orphans_and_completed_provider_reuse_keep_exact_handles() {
    let provider = CallId::from_wire("exact / provider handle");
    let local = CallId::from_wire("");
    let history = vec![
        result(provider.clone()),
        call(local.clone()),
        result(local),
        call(provider.clone()),
        result(provider.clone()),
        call(provider.clone()),
        result(provider),
    ];
    let ids =
        WireIds::for_target(&history, &Target, "model").expect("provider-held partial history");
    for message in [0, 3, 4, 5, 6] {
        assert_eq!(ids.of(message, 0), Some("exact / provider handle"));
    }
    assert_eq!(ids.of(1, 0), Some("tool-0"));
    assert_eq!(ids.of(2, 0), ids.of(1, 0));
}

#[test]
fn local_orphans_and_repeated_results_refuse_with_exact_coordinates() {
    let local = CallId::from_wire("");
    let error = WireIds::for_target(
        &[result(local.clone()), call(local.clone())],
        &Target,
        "model",
    )
    .expect_err("a local-only orphan cannot be paired");
    assert_eq!(
        error,
        WireIdError::OrphanResult {
            message: 0,
            content: 0
        }
    );
    assert_eq!(
        error.to_string(),
        "local-only orphan tool result at message 0, content 0"
    );
    let provider_equal = CallId::from_wire(local.wire().into_owned());
    let stale = vec![
        call(local.clone()),
        result(local.clone()),
        call(provider_equal),
        result(local.clone()),
    ];
    assert_eq!(
        WireIds::for_target(&stale, &Target, "model")
            .expect_err("local identity never consumes the provider variant"),
        WireIdError::DuplicateResult {
            message: 3,
            content: 0
        },
    );
    for id in [local, CallId::from_wire("provider-exact")] {
        let history = vec![call(id.clone()), result(id.clone()), result(id)];
        let error = WireIds::for_target(&history, &Target, "model").expect_err("duplicate result");
        assert_eq!(
            error,
            WireIdError::DuplicateResult {
                message: 2,
                content: 0
            }
        );
        assert_eq!(
            error.to_string(),
            "duplicate tool result at message 2, content 0"
        );
    }
    let provider = CallId::from_wire("provider-orphan");
    let error = WireIds::for_target(
        &[result(provider.clone()), result(provider)],
        &Target,
        "model",
    )
    .expect_err("provider orphan may be consumed only once");
    assert_eq!(
        error,
        WireIdError::DuplicateResult {
            message: 1,
            content: 0
        }
    );
}

#[test]
fn unplanned_coordinates_refuse_without_guessing_an_identity() {
    let ids = WireIds::default();
    assert_eq!(ids.of(4, 7), None);
    let error = ids
        .spell(4, 7)
        .expect_err("an empty plan has no occurrence");
    assert_eq!(
        error,
        WireIdError::MissingOccurrence {
            message: 4,
            content: 7
        }
    );
    assert_eq!(
        error.to_string(),
        "missing planned call at message 4, content 7"
    );
}

#[test]
fn future_whole_and_reasoning_hosted_handles_reserve_every_local_occurrence() {
    use crate::message::{
        CarriedPayload, CarriedPosition, CarriedReasoning, CarriedReasoningPart, ProviderItem,
    };
    let local = CallId::from_wire("");
    let hosted =
        |id: &str| serde_json::json!({"type":"server_tool_use","id":id,"input":{"exact":" raw "}});
    let history = vec![
        call(local.clone()),
        result(local.clone()),
        call(local.clone()),
        result(local),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::Opaque(crate::message::Opaque {
                item: hosted("tool-4"),
                replay: true,
            }),
            AssistantContent::ProviderItem(ProviderItem::new(hosted("tool-0"), "anthropic")),
            AssistantContent::CarriedReasoning(CarriedReasoning::new(
                vec![
                    CarriedReasoningPart::OpaqueSummary(
                        CarriedPayload::new(hosted("tool-1"), None)
                            .with_position(CarriedPosition::ReasoningSummary { part_index: 3 }),
                    ),
                    CarriedReasoningPart::OpaqueContent(
                        CarriedPayload::new(hosted("tool-2"), Some("anthropic".to_owned()))
                            .with_position(CarriedPosition::ReasoningContent { part_index: 7 }),
                    ),
                ],
                CarriedPayload::new(hosted("tool-3"), Some("anthropic".to_owned())),
            )),
        ])),
    ];
    let before = serde_json::to_value(&history).expect("original hosted payloads");
    let ids = WireIds::for_target(&history, &HostedTarget, "model").expect("hosted reservations");
    assert_eq!(ids.of(0, 0), Some("tool-5"));
    assert_eq!(ids.of(1, 0), ids.of(0, 0));
    assert_eq!(ids.of(2, 0), Some("tool-6"));
    assert_eq!(ids.of(3, 0), ids.of(2, 0));
    assert_eq!(
        serde_json::to_value(&history).expect("unchanged payloads"),
        before
    );
}

#[derive(Debug)]
struct NormalizingTarget;

impl ReplayTarget for NormalizingTarget {
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
        _model: &str,
        _source: Option<&crate::message::Origin>,
    ) -> String {
        format!("normalized_{id}")
    }
}

#[test]
fn single_occurrence_normalization_and_future_provider_reservation_keep_upstream_bytes() {
    let local = CallId::from_wire("");
    let plain = vec![call(local.clone()), result(local.clone())];
    let ids = WireIds::for_target(&plain, &NormalizingTarget, "model").expect("single occurrence");
    assert_eq!(ids.of(0, 0), Some("normalized_tool-0"));
    assert_eq!(ids.of(1, 0), ids.of(0, 0));
    let mut reserved = plain;
    reserved.push(result(CallId::from_wire("normalized_tool-0")));
    let ids =
        WireIds::for_target(&reserved, &NormalizingTarget, "model").expect("future genuine orphan");
    assert_eq!(ids.of(0, 0), Some("normalized_tool-1"));
    assert_eq!(ids.of(1, 0), ids.of(0, 0));
    assert_eq!(ids.of(2, 0), Some("normalized_tool-0"));
}
