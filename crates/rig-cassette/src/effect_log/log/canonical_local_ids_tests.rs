use super::TestResult;
use rig_core::{
    completion::{CompletionRequest, CompletionResponse, Usage},
    effect::{EffectId, EffectKind, EffectRecord, HandlerKey, MemoryOp, MemoryOutcome, Outcome},
    error::{ErrorDetail, ErrorKind, ErrorReport, MalformedToolInput, ProviderCallEvidence},
    message::{
        AssistantContent, AssistantMessage, CallId, CustomToolCall, Message, Opaque, Origin,
        ProviderCallId, ProviderItem, ToolCall, ToolFunction, ToolName, ToolResultContent,
        UserContent,
    },
    streaming::Transcript,
};
use serde_json::{Value, json};

use super::{EffectLog, RecordedStreamError, canonical_local_ids};

const FIRST: &[&str] = &[
    "/records/0/kind/request/chat_history/1/content/0/id",
    "/records/0/kind/request/chat_history/2/content/1/call",
    "/records/0/outcome/Ok/choice/0/id",
    "/records/0/events/2/value/content/id",
    "/records/1/kind/op/messages/1/content/0/id",
    "/records/1/kind/op/messages/2/content/1/call",
    "/records/1/outcome/Ok/messages/1/content/0/id",
    "/records/1/outcome/Ok/messages/2/content/1/call",
    "/header/stream_errors/1/0/error/detail/id",
];
const SECOND: &[&str] = &[
    "/records/0/kind/request/chat_history/1/content/1/id",
    "/records/0/kind/request/chat_history/2/content/0/call",
    "/records/0/outcome/Ok/choice/1/id",
    "/records/0/events/5/value/content/id",
    "/records/1/kind/op/messages/1/content/1/id",
    "/records/1/kind/op/messages/2/content/0/call",
    "/records/1/outcome/Ok/messages/1/content/1/id",
    "/records/1/outcome/Ok/messages/2/content/0/call",
    "/records/4/outcome/Err/detail/id",
];

fn typed_log() -> TestResult<EffectLog> {
    let first = CallId::from_wire("");
    let second = CallId::from_wire("");
    let text = first.wire().into_owned();
    let opaque = json!({"local": text, "nested": {"local": second.wire()},
        "text": format!("before-{text}-after")});
    let function = ToolCall::new(
        first.clone(),
        ToolFunction::new(ToolName::new("function")?, opaque.clone()),
    );
    let custom = CustomToolCall::new(second.clone(), ToolName::new("custom")?, text.clone());
    let provider = ToolCall::from_wire(
        text.clone(),
        ToolFunction::new(ToolName::new("provider")?, json!({})),
    );
    let choice = vec![
        AssistantContent::ToolCall(function.clone()).with_native(opaque.clone()),
        AssistantContent::CustomToolCall(custom.clone()).with_native(opaque.clone()),
        AssistantContent::ToolCall(provider.clone()),
        AssistantContent::text(text.clone()).with_native(opaque.clone()),
        AssistantContent::Opaque(Opaque {
            item: opaque.clone(),
            replay: true,
        }),
        AssistantContent::ProviderItem(ProviderItem::new(opaque.clone(), "test")),
    ];
    let messages = vec![
        Message::System {
            content: text.clone(),
        },
        Message::Assistant(AssistantMessage::new(choice.clone())),
        Message::User {
            content: vec![
                UserContent::ToolResult(custom.result(vec![ToolResultContent::text(text.clone())])),
                UserContent::ToolResult(
                    function.result(vec![ToolResultContent::text(text.clone())]),
                ),
                UserContent::ToolResult(
                    provider.result(vec![ToolResultContent::text(text.clone())]),
                ),
            ],
        },
    ];
    let mut request = CompletionRequest::from(messages.clone());
    request.model = Some(text.clone());
    request.additional_params = Some(opaque.clone());
    request.output_schema = Some(serde_json::from_value(json!({
        "type": "object", "description": text, "x-data": opaque,
    }))?);
    let events = Transcript::parse(json!([
        {"item": "event", "value": {"event": "start", "part": 0,
            "kind": "tool_call", "name": function.function.name}},
        {"item": "event", "value": {"event": "arguments", "part": 0,
            "json": text}},
        {"item": "event", "value": {"event": "end", "part": 0,
            "content": choice[0]}},
        {"item": "event", "value": {"event": "start", "part": 1,
            "kind": "custom_tool_call", "name": custom.name}},
        {"item": "event", "value": {"event": "input", "part": 1,
            "input": text}},
        {"item": "event", "value": {"event": "end", "part": 1,
            "content": choice[1]}},
        {"item": "unknown", "value": opaque},
    ]))?;
    let error = |id: CallId| -> TestResult<ErrorReport> {
        let mut error = ErrorReport::new(ErrorKind::Json, text.clone());
        error.source_chain = vec![format!("source-{text}")];
        error.detail = Some(ErrorDetail::MalformedToolInput(MalformedToolInput {
            name: text.clone(),
            id,
            provider: Some(ProviderCallEvidence {
                call_id: ProviderCallId::try_from(text.clone())?,
                item_id: Some(text.clone()),
            }),
            raw: opaque.to_string(),
            error: text.clone(),
        }));
        Ok(error)
    };
    let record = |id, kind, outcome| EffectRecord {
        id: EffectId::from_raw(id),
        key: HandlerKey::from(format!("test/{id}")),
        kind,
        outcome,
        events: None,
        tool_output: None,
        stream_origin: None,
        parent: None,
        scope: None,
    };
    let mut completion = record(
        1,
        EffectKind::Completion {
            request,
            stream: true,
        },
        Ok(Outcome::Completion(CompletionResponse::new(
            choice,
            Usage::default(),
            Origin::new("test.api", "test", text.clone()),
            opaque.clone(),
        ))),
    );
    completion.events = Some(events);
    completion.tool_output = Some(serde_json::from_value(json!({"published": opaque}))?);
    let mut log = EffectLog::from_records(vec![
        completion,
        record(
            2,
            EffectKind::Memory {
                op: MemoryOp::Append {
                    conversation: text.clone().into(),
                    messages: messages.clone(),
                },
            },
            Ok(Outcome::Memory(MemoryOutcome::Loaded { messages })),
        ),
        record(
            3,
            EffectKind::ToolCall {
                name: text.clone(),
                args: opaque.to_string(),
            },
            Ok(Outcome::ToolResult {
                result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::json(
                    opaque.clone(),
                )),
            }),
        ),
        record(
            4,
            EffectKind::Custom {
                kind: text.clone().into(),
                payload: opaque.clone(),
            },
            Ok(Outcome::Custom {
                payload: opaque.clone(),
            }),
        ),
        record(
            5,
            EffectKind::ToolCall {
                name: "invalid".into(),
                args: text.clone(),
            },
            Err(error(second)?),
        ),
    ]);
    log.header.stream_errors.insert(
        EffectId::from_raw(1),
        vec![RecordedStreamError {
            item: 7,
            error: error(first)?,
        }],
    );
    Ok(log)
}

fn check_only_locations(original: Value, first: &[&str], second: &[&str]) -> TestResult<Value> {
    let normalized = canonical_local_ids(original.clone());
    let mut expected = original.clone();
    let mut renamed = Vec::new();
    for locations in [first, second] {
        let path = locations
            .first()
            .ok_or_else(|| anyhow::anyhow!("fixture has no identity locations: {locations:?}"))?;
        let canonical = normalized
            .pointer(path)
            .ok_or_else(|| anyhow::anyhow!("missing normalized identity {path}: {normalized:?}"))?
            .clone();
        let id: CallId = serde_json::from_value(canonical.clone())?;
        if !id.is_local() {
            return Err(anyhow::anyhow!("local identity changed namespace: {id:?}").into());
        }
        if Some(&canonical) == original.pointer(path) {
            return Err(anyhow::anyhow!(
                "local identity was not normalized at {path}: {canonical:?}"
            )
            .into());
        }
        for path in locations {
            let Some(value) = expected.pointer_mut(path) else {
                return Err(
                    anyhow::anyhow!("missing typed fixture location {path}: {expected:?}").into(),
                );
            };
            *value = canonical.clone();
        }
        renamed.push(canonical);
    }
    let first = renamed
        .first()
        .ok_or_else(|| anyhow::anyhow!("first renamed identity missing: {renamed:?}"))?;
    let second = renamed
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("second renamed identity missing: {renamed:?}"))?;
    if first == second {
        return Err(anyhow::anyhow!("distinct local identities collapsed: {renamed:?}").into());
    }
    if normalized != expected {
        return Err(anyhow::anyhow!(
            "canonicalization changed data outside typed local identity leaves: actual {normalized:?}, expected {expected:?}"
        ).into());
    }
    let again = canonical_local_ids(normalized.clone());
    if again != normalized {
        return Err(anyhow::anyhow!(
            "canonicalization is not idempotent: actual {again:?}, expected {normalized:?}"
        )
        .into());
    }
    Ok(normalized)
}

#[test]
fn canonical_local_ids_preserves_untyped_and_custom_data() -> TestResult<()> {
    for value in [
        json!({"local": "red"}),
        json!({"local": "7ecaf57b-5bbb-4b9e-a440-63738a5939ec"}),
        json!([{"local": "red"}, "before-red-after"]),
        serde_json::to_value(EffectKind::Custom {
            kind: "opaque".into(),
            payload: json!({"local": "red"}),
        })?,
        serde_json::to_value(EffectKind::Custom {
            kind: "opaque".into(),
            payload: json!({"local": "7ecaf57b-5bbb-4b9e-a440-63738a5939ec"}),
        })?,
    ] {
        let normalized = canonical_local_ids(value.clone());
        if normalized != value {
            return Err(anyhow::anyhow!(
                "opaque local-looking JSON changed: actual {normalized:?}, expected {value:?}"
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn canonical_local_ids_changes_only_typed_completion_leaves() -> TestResult<()> {
    let log = typed_log()?;
    let prefix = "/records/0/kind";
    let first: Vec<_> = FIRST
        .iter()
        .filter_map(|path| path.strip_prefix(prefix))
        .collect();
    let second: Vec<_> = SECOND
        .iter()
        .filter_map(|path| path.strip_prefix(prefix))
        .collect();
    let normalized =
        check_only_locations(serde_json::to_value(&log.records[0].kind)?, &first, &second)?;
    let _: EffectKind = serde_json::from_value(normalized)?;
    Ok(())
}

#[test]
fn canonical_local_ids_preserves_bijection_in_whole_logs() -> TestResult<()> {
    let normalized = check_only_locations(serde_json::to_value(typed_log()?)?, FIRST, SECOND)?;
    let _: EffectLog = serde_json::from_value(normalized)?;
    Ok(())
}

/// A struct also deserializes from the sequence of its field values, so
/// `[header, records]` is a typed log. Its local identities' locations are
/// object paths, which do not resolve in a sequence; each is skipped and the
/// value comes back unchanged.
#[test]
fn canonical_local_ids_skips_locations_absent_from_sequence_form_logs() -> TestResult<()> {
    let log = typed_log()?;
    let object = serde_json::to_value(&log)?;
    if canonical_local_ids(object.clone()) == object {
        return Err(
            anyhow::anyhow!("fixture has no typed local identity to locate: {object:?}").into(),
        );
    }
    let sequence = json!([
        serde_json::to_value(&log.header)?,
        serde_json::to_value(&log.records)?,
    ]);
    let from_sequence: EffectLog = serde_json::from_value(sequence.clone())?;
    let from_object: EffectLog = serde_json::from_value(object)?;
    if serde_json::to_value(&from_sequence)? != serde_json::to_value(&from_object)? {
        return Err(anyhow::anyhow!(
            "sequence form read as another log: actual {from_sequence:?}, expected {from_object:?}"
        )
        .into());
    }
    let normalized = canonical_local_ids(sequence.clone());
    if normalized != sequence {
        return Err(anyhow::anyhow!(
            "unresolvable locations changed the sequence form: actual {normalized:?}, expected {sequence:?}"
        )
        .into());
    }
    Ok(())
}

#[test]
fn canonical_local_ids_preserves_bijection_in_record_arrays() -> TestResult<()> {
    let first: Vec<_> = FIRST
        .iter()
        .filter_map(|path| path.strip_prefix("/records"))
        .collect();
    let second: Vec<_> = SECOND
        .iter()
        .filter_map(|path| path.strip_prefix("/records"))
        .collect();
    let normalized =
        check_only_locations(serde_json::to_value(typed_log()?.records)?, &first, &second)?;
    let _: Vec<EffectRecord> = serde_json::from_value(normalized)?;
    Ok(())
}
