//! A custom call sent as text, optional call-id omission that finds no
//! field to remove, and the Responses pairing of carried reasoning with the
//! part after it.

use std::borrow::Cow;

use serde_json::{Value, json};

use super::{Accepts, OptionalCallId, Replay, ReplayTarget, Request, adapt, adapt_for};
use crate::message::{
    Api, AssistantContent, AssistantMessage, CallId, CarriedPayload, CarriedPosition,
    CarriedReasoning, CarriedReasoningPart, CarriedText, CarriedTextContent, CustomToolCall,
    LocalCallId, Message, Origin, StopReason, Text, ToolCall, ToolFunction, ToolName,
    ToolResultContent, UserContent,
};
use crate::providers::openai::wire::OPENAI;

fn turn(origin: Option<Origin>, content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        origin,
        stop: Some(StopReason::Stop),
    })
}

/// A target with an optional call-id field at the JSON pointer it holds.
#[derive(Debug)]
struct Slotted(&'static str);

impl ReplayTarget for Slotted {
    fn api(&self) -> Api {
        Api::from_static("optional.calls")
    }

    fn provider(&self) -> &str {
        "optional-provider"
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

    fn call_id_slot(&self) -> Option<&'static str> {
        Some(self.0)
    }
}

#[test]
fn a_request_without_tools_gets_a_custom_call_as_text_with_its_raw_input() {
    let call = CustomToolCall::new(
        CallId::from_wire("custom_1"),
        ToolName::new("apply_patch").expect("a name"),
        "*** Begin Patch\nraw {not json}",
    );
    let history = vec![
        Message::user("q"),
        turn(None, vec![AssistantContent::CustomToolCall(call.clone())]),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("patched")]),
            )],
        },
    ];
    assert_eq!(
        adapt_for(
            &history,
            &Slotted("/call_id"),
            &Request {
                tools: false,
                ..Request::default()
            }
        ),
        vec![
            Message::user("q"),
            turn(
                None,
                vec![AssistantContent::Text(Text::new(
                    "[called tool apply_patch with *** Begin Patch\nraw {not json}]"
                ))]
            ),
            Message::User {
                content: vec![UserContent::text("[tool apply_patch result] patched")],
            },
        ]
    );
}

/// A current item keeps every field when the optional slot names no object
/// field in it: a pointer without a parent, or a parent that is missing or
/// not an object.
#[test]
fn optional_omission_leaves_a_current_item_whole_when_the_slot_resolves_to_no_field() {
    let native = json!({"id": "leftover", "routing": 7,
        "function": {"name": "lookup", "arguments": {"x": 1}}});
    for slot in ["id", "", "/missing/id", "/routing/id"] {
        let block = AssistantContent::ToolCall(ToolCall::new(
            CallId::Local(LocalCallId::new()),
            ToolFunction::new(ToolName::new("lookup").expect("a name"), json!({"x": 1})),
        ))
        .with_native(native.clone());
        assert_eq!(
            block.replay_optional_call_id(&Slotted(slot), OptionalCallId::OmitLocal),
            Replay::Item(Cow::Owned(native.clone())),
            "{slot}"
        );
        assert_eq!(block.native_item(), Some(&native), "{slot}");
    }
}

#[test]
fn optional_omission_returns_a_non_call_rebuild_unchanged() {
    let text = AssistantContent::text("plain");
    assert_eq!(
        text.replay_optional_call_id(&Slotted("/id"), OptionalCallId::OmitLocal),
        Replay::Rebuild
    );
}

/// Direct OpenAI Responses as replay sees it, with a chosen answer for
/// whether a reasoning item needs the item after it.
#[derive(Debug)]
struct DirectResponses {
    needs_next: bool,
}

impl ReplayTarget for DirectResponses {
    fn api(&self) -> Api {
        Api::from_static("openai.responses")
    }

    fn provider(&self) -> &str {
        OPENAI.name
    }

    fn model(&self) -> &str {
        "gpt-5"
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

    fn needs_next(&self, item: &Value) -> bool {
        self.needs_next && item.get("type").and_then(Value::as_str) == Some("reasoning")
    }
}

fn own() -> Option<Origin> {
    Some(Origin::new("openai.responses", OPENAI.name, "gpt-5"))
}

/// Reasoning item `rs_1` whose item holds the ciphertext `cipher`.
fn reasoning(parts: Vec<CarriedReasoningPart>) -> AssistantContent {
    AssistantContent::CarriedReasoning(
        CarriedReasoning::new(
            parts,
            CarriedPayload::new(
                json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher"}),
                Some(OPENAI.name.to_owned()),
            ),
        )
        .with_id(Some("rs_1".to_owned())),
    )
}

fn answer(position: CarriedPosition) -> AssistantContent {
    let mut text = CarriedText::new(
        CarriedTextContent::Text("answer".to_owned()),
        CarriedPayload::new(
            json!({"type": "output_text", "text": "answer", "annotations": []}),
            Some(OPENAI.name.to_owned()),
        )
        .with_position(position),
    );
    text.item_id = Some("msg_1".to_owned());
    AssistantContent::CarriedText(text)
}

#[test]
fn carried_reasoning_the_target_does_not_pair_stays_without_a_partner() {
    let history = vec![
        Message::user("q"),
        turn(
            own(),
            vec![reasoning(vec![CarriedReasoningPart::Encrypted(
                "cipher".to_owned(),
            )])],
        ),
    ];
    assert_eq!(
        adapt(&history, &DirectResponses { needs_next: false }),
        history
    );
    assert_eq!(
        adapt(&history, &DirectResponses { needs_next: true }),
        vec![Message::user("q")],
        "the same reasoning paired by the target goes without its partner"
    );
}

/// The pairing reads the first ciphertext part, past visible parts.
#[test]
fn carried_reasoning_pairs_on_its_first_ciphertext_part_after_visible_parts() {
    let target = DirectResponses { needs_next: true };
    let visible = || {
        vec![
            CarriedReasoningPart::Summary("summary".to_owned()),
            CarriedReasoningPart::Text {
                text: "thought".to_owned(),
                signature: None,
            },
        ]
    };
    let with = |last: CarriedReasoningPart| {
        let mut parts = visible();
        parts.push(last);
        vec![Message::user("q"), turn(own(), vec![reasoning(parts)])]
    };
    assert_eq!(
        adapt(
            &with(CarriedReasoningPart::Encrypted("cipher".to_owned())),
            &target
        ),
        vec![Message::user("q")],
        "the selected ciphertext is the item's, so it needs its missing partner"
    );
    for last in [
        CarriedReasoningPart::Encrypted("other".to_owned()),
        CarriedReasoningPart::Redacted {
            data: "withheld".to_owned(),
        },
    ] {
        let history = with(last);
        assert_eq!(adapt(&history, &target), history);
    }
}

/// Only a message-content part keeps the reasoning before it paired.
#[test]
fn carried_text_outside_message_content_does_not_partner_carried_reasoning() {
    let target = DirectResponses { needs_next: true };
    let cipher = || reasoning(vec![CarriedReasoningPart::Encrypted("cipher".to_owned())]);
    let kept = vec![
        Message::user("q"),
        turn(
            own(),
            vec![
                cipher(),
                answer(CarriedPosition::MessageContent {
                    item_index: 1,
                    part_index: 0,
                }),
            ],
        ),
    ];
    assert_eq!(adapt(&kept, &target), kept);
    for position in [
        CarriedPosition::Whole,
        CarriedPosition::ReasoningSummary { part_index: 0 },
    ] {
        let history = vec![
            Message::user("q"),
            turn(own(), vec![cipher(), answer(position)]),
        ];
        assert_eq!(
            adapt(&history, &target),
            vec![Message::user("q"), turn(own(), vec![answer(position)])]
        );
    }
}
