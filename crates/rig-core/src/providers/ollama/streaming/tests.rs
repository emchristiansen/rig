//! Decoder tests. Each states a reply shape Ollama's `/api/chat` documents
//! (`docs/api.md`) or one the inline `<think>` split must survive; the
//! recorded replies are pinned by the `ollama` cassettes.

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::providers::ollama::OllamaConfig;
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::wire::{Call, Mode, Operation, Shared, Wire};
use serde_json::json;

const MODEL: &str = "llama3.2";

#[test]
fn missing_tool_ids_are_distinct_stable_and_collision_free_in_responses() {
    let wire = json!({
        "model": "test", "created_at": "2024-01-01T00:00:00Z", "done": true,
        "message": {"role":"assistant", "content":"", "tool_calls":[
            {"function":{"name":"same","arguments":{"value":1}}},
            {"id":"tool-0","function":{"name":"same","arguments":{"value":2}}},
            {"function":{"name":"same","arguments":{"value":3}}}
        ]}
    });
    let normalize = || decode("test", Mode::Unary, std::slice::from_ref(&wire)).expect("response decodes");
    let first = normalize();
    assert_eq!(serde_json::to_value(&first.choice).expect("choice"),
        serde_json::to_value(normalize().choice).expect("choice"));
    let calls: Vec<_> = first.tool_calls().collect();
    assert_eq!(calls.iter().map(|call| &call.id).collect::<std::collections::HashSet<_>>().len(), 3);
    assert!(calls[0].id.provider().is_none());
    assert_eq!(calls[1].id.provider().map(|id| id.as_str()), Some("tool-0"));
    assert!(calls[2].id.provider().is_none());
}

fn frames(records: &[Value]) -> Vec<WireFrame> {
    records
        .iter()
        .map(|record| WireFrame::Text(record.to_string()))
        .collect()
}

/// One streamed record carrying `message`.
fn record(model: &str, message: Value) -> Value {
    json!({"model": model, "created_at": "2026-10-05T00:00:00Z", "message": message, "done": false})
}

/// The record that ends a reply on `reason`.
fn done(model: &str, reason: &str) -> Value {
    json!({
        "model": model,
        "created_at": "2026-10-05T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": true,
        "done_reason": reason,
        "total_duration": 100,
        "prompt_eval_count": 12,
        "prompt_eval_cached_count": 4,
        "eval_count": 7,
    })
}

/// A whole reply: `message`, ended on `reason`.
fn whole(model: &str, message: Value, reason: &str) -> Value {
    let mut reply = done(model, reason);
    reply["message"] = message;
    reply
}

fn decode(model: &str, mode: Mode, records: &[Value]) -> Result<CompletionResponse, ProviderError> {
    let wire = OllamaConfig::new().native_completion(model);
    crate::test_utils::history::decode(&wire, mode, frames(records))
}

/// The text and reasoning blocks of `response`, in order.
fn blocks(response: &CompletionResponse) -> Vec<(&'static str, String)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(("text", text.text.clone())),
            AssistantContent::Reasoning(reasoning) => Some(("reasoning", reasoning.text.clone())),
            _ => None,
        })
        .collect()
}

/// The events the consumer has seen after each of `records` is decoded,
/// before the reply ends: what streaming shows as it goes.
fn events_after_each(model: &str, records: &[Value]) -> Vec<Vec<StreamEvent>> {
    let wire = OllamaConfig::new().native_completion(model);
    let describe = wire.describe();
    let fold = Completion::fold(
        &CompletionRequest::new("hi"),
        &mut Call::new(&describe, Mode::Streaming),
    );
    let shared = std::sync::Mutex::new(Shared::new(fold));
    let mut decoder = wire.decoder_in(Mode::Streaming);
    let mut seen = Vec::new();
    let mut taken = Vec::new();
    for frame in frames(records) {
        crate::driver::step(
            &mut decoder,
            None::<&mut crate::wire::document::Unreassembled>,
            &shared,
            frame,
            None,
        )
        .map(drop)
        .expect("the record decodes");
        let mut shared = shared.lock().expect("the reply is not poisoned");
        while let Some(item) = shared.take() {
            if let Ok(Item::Event(event)) = item {
                taken.push(event);
            }
        }
        seen.push(taken.clone());
    }
    seen
}

/// The text a list of events streamed, by kind.
fn streamed(events: &[StreamEvent]) -> (String, String) {
    let (mut text, mut reasoning) = (String::new(), String::new());
    for event in events {
        match event {
            StreamEvent::Text { text: fragment, .. } => text.push_str(fragment),
            StreamEvent::Reasoning { text: fragment, .. } => reasoning.push_str(fragment),
            _ => {}
        }
    }
    (text, reasoning)
}

/// The native record has no discriminator. Invalid modeled fields and
/// unrelated JSON surface as corrupt frames.
#[test]
fn a_line_is_a_record_unknown_or_corrupt() {
    let decoder = ChatDecoder::default();
    let line = |text: &str| decoder.classify(WireFrame::Text(text.to_owned()));
    assert!(matches!(
        line(&record(MODEL, json!({"role": "assistant", "content": "hi"})).to_string()),
        WireEvent::Known(_)
    ));
    assert!(matches!(line("{not json"), WireEvent::Corrupt(_)));
    assert!(matches!(line(r#"{"done": 42}"#), WireEvent::Corrupt(_)));
    assert!(matches!(line(r#"{"status": "ok"}"#), WireEvent::Corrupt(_)));
    assert!(matches!(line("[1]"), WireEvent::Corrupt(_)));
    let mistyped = decode(MODEL, Mode::Streaming, &[json!({"done": 42, "message": 7})]);
    assert!(
        matches!(mistyped, Err(ProviderError::CorruptFrame(_))),
        "{mistyped:?}"
    );
}

/// A streamed reply's records join into one text block, and the `done`
/// record ends it with its reason, counters and model.
#[test]
fn ndjson_records_stream_into_one_answer() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "The sky "})),
        record(MODEL, json!({"role": "assistant", "content": "is blue."})),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(blocks(&response), [("text", "The sky is blue.".to_owned())]);
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.input_tokens, Some(12));
    assert_eq!(response.usage.output_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(19));
    assert_eq!(response.usage.cached_input_tokens, Some(4));
    assert_eq!(response.model(), Some(MODEL));
    // A stream's raw is the body a unary call returns.
    assert_eq!(
        response.raw,
        whole(
            MODEL,
            json!({"role": "assistant", "content": "The sky is blue."}),
            "stop"
        )
    );
}

/// The reply's `thinking` is reasoning, held before the answer.
#[test]
fn thinking_is_reasoning() {
    let records = [
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "thinking": "Rayleigh "}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "thinking": "scattering."}),
        ),
        record(MODEL, json!({"role": "assistant", "content": "Blue."})),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "Rayleigh scattering.".to_owned()),
            ("text", "Blue.".to_owned())
        ]
    );
}

/// Ollama sends each call whole, its arguments an object. Each call starts
/// and streams its arguments through the shared writer, keeps the daemon's
/// id, and a `stop` reply that called a tool ends on `ToolCalls`.
#[test]
fn ndjson_tool_calls_stream_through_the_writer() {
    let call = |id: &str, city: &str| json!({"id": id, "function": {"index": 0, "name": "weather", "arguments": {"city": city}}});
    let records = [
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "tool_calls": [call("call_a", "Paris")]}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "tool_calls": [call("call_b", "Rome")]}),
        ),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.wire(), "call_a");
    assert_eq!(calls[1].id.wire(), "call_b");
    assert_eq!(calls[0].function.name.as_str(), "weather");
    assert_eq!(
        calls[1].function.arguments.get("city"),
        Some(&json!("Rome"))
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));

    let events = events_after_each(MODEL, &records);
    let first = &events[0];
    assert!(first.iter().any(|event| matches!(
        event,
        StreamEvent::Start { kind: PartKind::ToolCall, name: Some(name), .. } if name.as_str() == "weather"
    )));
    assert!(first.iter().any(|event| matches!(
        event,
        StreamEvent::Arguments { json, .. } if json == r#"{"city":"Paris"}"#
    )));
}

/// Text and absent arguments decode; a malformed call surfaces an error.
#[test]
fn call_arguments_as_text_or_absent() {
    let reply = whole(
        MODEL,
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "a", "arguments": "{\"x\":1}"}},
            {"function": {"name": "b"}},
        ]}),
        "stop",
    );
    let response = decode(MODEL, Mode::Unary, &[reply]).expect("the reply decodes");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.arguments.get("x"), Some(&json!(1)));
    assert!(calls[1].function.arguments.is_empty());

    // A call that is not an object names no tool, and is dropped too.
    let reply = whole(
        MODEL,
        json!({"role": "assistant", "tool_calls": ["call"]}),
        "stop",
    );
    let error = decode(MODEL, Mode::Unary, &[reply]).expect_err("the malformed call is refused");
    assert!(matches!(error, ProviderError::CorruptFrame(_)));
}

/// `done_reason` maps to the finish reason: `stop`, `length`, and anything
/// else carried as Ollama spells it.
#[test]
fn done_reason_is_the_finish_reason() {
    for (reason, expected) in [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("unload", FinishReason::Other("unload".to_owned())),
    ] {
        let reply = whole(
            MODEL,
            json!({"role": "assistant", "content": "partial"}),
            reason,
        );
        let response = decode(MODEL, Mode::Unary, std::slice::from_ref(&reply));
        let finish = match response {
            Ok(response) => response.finish_reason(),
            // An unknown reason fails the turn.
            Err(_) => Some(FinishReason::Other(reason.to_owned())),
        };
        assert_eq!(finish, Some(expected), "{reason}");
    }
}

/// An in-band `error` record fails the reply, and a stream that ends
/// before its `done` record is truncated.
#[test]
fn an_error_record_fails_and_a_cut_stream_is_truncated() {
    let error = decode(
        MODEL,
        Mode::Streaming,
        &[
            record(MODEL, json!({"role": "assistant", "content": "par"})),
            json!({"error": "model runner has unexpectedly stopped"}),
        ],
    )
    .expect_err("the error fails the reply");
    assert!(
        error.to_string().contains("unexpectedly stopped"),
        "{error}"
    );

    let cut = decode(
        MODEL,
        Mode::Streaming,
        &[record(
            MODEL,
            json!({"role": "assistant", "content": "par"}),
        )],
    )
    .expect_err("no done record");
    assert!(matches!(cut, ProviderError::Truncated), "{cut:?}");
}

/// Terminal content splits closed legacy reasoning; nonterminal character
/// fragments preserve their literal content.
#[test]
fn inline_think_splits_the_same_whole_and_streamed() {
    let qwen = "qwen3:4b";
    let cases: [(&str, &str, &str, Vec<(&'static str, &str)>); 9] = [
        (
            "terminated",
            MODEL,
            "<think>private</think>\n\nThe answer.",
            vec![("reasoning", "private"), ("text", "The answer.")],
        ),
        (
            "after leading whitespace",
            MODEL,
            "\n <think>\nprivate\n</think>\nThe answer.",
            vec![("reasoning", "private"), ("text", "The answer.")],
        ),
        (
            "unterminated remains visible",
            MODEL,
            "<think>never closed",
            vec![("text", "<think>never closed")],
        ),
        (
            "absent",
            MODEL,
            "Just an answer.",
            vec![("text", "Just an answer.")],
        ),
        (
            "Qwen3 permits its prefilled closing boundary",
            qwen,
            "private\n</think>\n\nThe answer.",
            vec![("reasoning", "private"), ("text", "The answer.")],
        ),
        (
            "not at the start stays text",
            MODEL,
            "Use <think>x</think> tags.",
            vec![("text", "Use <think>x</think> tags.")],
        ),
        (
            "an empty block leaves only the answer",
            MODEL,
            "<think>\n\n</think>\n\nThe answer.",
            vec![("text", "The answer.")],
        ),
        (
            "a partial close tag inside the block is reasoning",
            MODEL,
            "<think>a </th b</think>c",
            vec![("reasoning", "a </th b"), ("text", "c")],
        ),
        (
            "a prefix of the opening tag is text",
            MODEL,
            "<thin ice",
            vec![("text", "<thin ice")],
        ),
    ];
    for (case, model, content, expected) in cases {
        let expected: Vec<(&str, String)> = expected
            .into_iter()
            .map(|(kind, text)| (kind, text.to_owned()))
            .collect();
        let reply = whole(
            model,
            json!({"role": "assistant", "content": content}),
            "stop",
        );
        let response = decode(model, Mode::Unary, &[reply]).expect("the reply decodes");
        assert_eq!(blocks(&response), expected, "{case}, whole");

        let mut records: Vec<Value> = content
            .chars()
            .map(|ch| {
                record(
                    model,
                    json!({"role": "assistant", "content": ch.to_string()}),
                )
            })
            .collect();
        records.push(done(model, "stop"));
        let response = decode(model, Mode::Streaming, &records).expect("the reply decodes");
        assert_eq!(
            blocks(&response),
            [("text", content.to_owned())],
            "{case}, streamed"
        );
    }
}

/// A reply with a native `thinking` field splits nothing out of its
/// content, whole or streamed.
#[test]
fn native_thinking_turns_the_split_off() {
    let content = "<think>shown</think> as written";
    let reply = whole(
        "qwen3:4b",
        json!({"role": "assistant", "content": content, "thinking": "native"}),
        "stop",
    );
    let response = decode("qwen3:4b", Mode::Unary, &[reply]).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "native".to_owned()),
            ("text", content.to_owned())
        ]
    );

    let records = [
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "", "thinking": "native"}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "<think>shown"}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "</think> as written"}),
        ),
        done("qwen3:4b", "stop"),
    ];
    let response = decode("qwen3:4b", Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "native".to_owned()),
            ("text", content.to_owned())
        ]
    );
    // The content streams as it arrives.
    let events = events_after_each("qwen3:4b", &records);
    assert_eq!(streamed(&events[1]).0, "<think>shown");
}

/// A reply that does not open with `<think>` streams as text record by
/// record, with nothing held, whatever the model's name.
#[test]
fn a_reply_without_a_think_tag_streams_record_by_record() {
    let coder = "qwen3-coder:30b";
    let records = [
        record(coder, json!({"role": "assistant", "content": "def"})),
        record(coder, json!({"role": "assistant", "content": " add"})),
        record(coder, json!({"role": "assistant", "content": "(a, b):"})),
        done(coder, "stop"),
    ];
    let events = events_after_each(coder, &records);
    assert_eq!(streamed(&events[0]), ("def".to_owned(), String::new()));
    assert_eq!(streamed(&events[1]).0, "def add");
    assert_eq!(streamed(&events[2]).0, "def add(a, b):");
}

/// Content fragments stay literal before and after a closing marker.
#[test]
fn reasoning_streams_before_the_block_closes() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "<thi"})),
        record(
            MODEL,
            json!({"role": "assistant", "content": "nk>step one"}),
        ),
        record(MODEL, json!({"role": "assistant", "content": ", step two"})),
        record(MODEL, json!({"role": "assistant", "content": "</think>"})),
        record(MODEL, json!({"role": "assistant", "content": "\n\n"})),
        record(MODEL, json!({"role": "assistant", "content": "Done."})),
        done(MODEL, "stop"),
    ];
    let events = events_after_each(MODEL, &records);
    assert_eq!(streamed(&events[0]), ("<thi".to_owned(), String::new()));
    assert_eq!(
        streamed(&events[1]),
        ("<think>step one".to_owned(), String::new())
    );
    assert_eq!(
        streamed(&events[2]),
        ("<think>step one, step two".to_owned(), String::new())
    );
    assert_eq!(
        streamed(&events[4]).0,
        "<think>step one, step two</think>\n\n"
    );
    assert_eq!(
        streamed(&events[5]).0,
        "<think>step one, step two</think>\n\nDone."
    );
}

/// A closing marker split across records remains literal content.
#[test]
fn a_close_tag_split_across_records_is_found() {
    let records = [
        record(
            MODEL,
            json!({"role": "assistant", "content": "<think>plan</th"}),
        ),
        record(MODEL, json!({"role": "assistant", "content": "ink>Answer"})),
        done(MODEL, "stop"),
    ];
    let events = events_after_each(MODEL, &records);
    assert_eq!(
        streamed(&events[0]),
        ("<think>plan</th".to_owned(), String::new())
    );
    assert_eq!(
        streamed(&events[1]),
        ("<think>plan</think>Answer".to_owned(), String::new())
    );
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [("text", "<think>plan</think>Answer".to_owned())]
    );
}

/// An unterminated marker stays text. A call ends the preceding text run.
#[test]
fn an_unterminated_block_is_reasoning_and_a_call_ends_the_text() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "<think>"})),
        record(MODEL, json!({"role": "assistant", "content": "the answer"})),
        done(MODEL, "length"),
    ];
    let events = events_after_each(MODEL, &records);
    assert_eq!(
        streamed(&events[1]),
        ("<think>the answer".to_owned(), String::new())
    );
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [("text", "<think>the answer".to_owned())]
    );

    let records = [
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "Calling."}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "lookup", "arguments": {}}}
            ]}),
        ),
        done("qwen3:4b", "stop"),
    ];
    assert_eq!(
        streamed(&events_after_each("qwen3:4b", &records)[0]).0,
        "Calling."
    );
    let response = decode("qwen3:4b", Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(blocks(&response), [("text", "Calling.".to_owned())]);
    assert_eq!(response.tool_calls().count(), 1);
}

/// Observation reads the reason and the model off the `done` record, and
/// an in-band error's message. Usage is the driver's to observe.
#[test]
fn the_projection_reads_verdict_and_error() {
    use crate::observe::{AdapterContext, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    let mut attempt = context
        .attempt_for(&http::Request::new(()), "/api/chat")
        .expect("an attempt starts");
    for payload in [
        done(MODEL, "stop").to_string(),
        json!({"error": "out of memory"}).to_string(),
        "not json".to_owned(),
    ] {
        attempt.project(|sink| ChatDecoder::project(payload.as_bytes(), sink));
    }
    drop(attempt);
    let observed = log.trace();
    let text = format!("{observed:?}");
    assert!(!text.contains("input_tokens"), "{text}");
    assert!(text.contains("\"stop\""), "{text}");
    assert!(text.contains("out of memory"), "{text}");
}

/// The document the native reassembler rebuilds from `records`.
fn rebuilt(records: &[Value]) -> Value {
    let mut document = document::ChatResponse::default();
    for frame in frames(records) {
        crate::wire::document::Reassemble::absorb(&mut document, &frame);
    }
    crate::wire::document::Reassemble::finish(document)
}

/// A stream rebuilds the `/api/chat` body: `content` and `thinking` join,
/// calls, images and log probabilities collect, and the `done` record's
/// reason, counts and durations stand.
#[test]
fn a_stream_rebuilds_the_unary_body() {
    let logprob = |token: &str| json!({"token": token, "logprob": -0.25});
    let mut first = record(
        MODEL,
        json!({"role": "assistant", "content": "", "thinking": "Add "}),
    );
    first["logprobs"] = json!([logprob("Add")]);
    let mut second = record(
        MODEL,
        json!({"role": "assistant", "content": "", "thinking": "them."}),
    );
    second["logprobs"] = json!([logprob("them")]);
    let call =
        |id: &str| json!({"id": id, "function": {"name": "add", "arguments": {"a": 2, "b": 3}}});
    let records = [
        first,
        second,
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "tool_calls": [call("call_1")]}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": "Adding", "tool_calls": [call("call_2")]}),
        ),
        record(MODEL, json!({"role": "assistant", "content": " now."})),
        done(MODEL, "stop"),
    ];
    let mut expected = whole(
        MODEL,
        json!({
            "role": "assistant",
            "content": "Adding now.",
            "thinking": "Add them.",
            "tool_calls": [call("call_1"), call("call_2")],
        }),
        "stop",
    );
    expected["logprobs"] = json!([logprob("Add"), logprob("them")]);
    assert_eq!(rebuilt(&records), expected);

    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(response.raw, expected);
}

/// A reply that fails in band keeps its `error` beside what arrived, and a
/// stream cut before `done` rebuilds what arrived.
#[test]
fn a_failed_or_cut_stream_rebuilds_what_arrived() {
    let started = record(MODEL, json!({"role": "assistant", "content": "par"}));
    let failed = json!({"error": "model runner has unexpectedly stopped"});
    let mut expected = started.clone();
    expected["error"] = failed["error"].clone();
    assert_eq!(rebuilt(&[started.clone(), failed]), expected);
    assert_eq!(rebuilt(std::slice::from_ref(&started)), started);
    assert_eq!(rebuilt(&[]), Value::Null);
}

/// Ollama reports no total, so the observed usage carries none: the
/// observation is the decoder's normalized `Finish` usage, observed once
/// by the driver, never a second parse that sums the counters.
#[tokio::test]
async fn the_observed_usage_does_not_invent_a_total() {
    let body = br#"{"model":"m","created_at":"2026-10-05T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":4}"#;
    let (observed, usage) = crate::driver::tests::observed_usage(
        OllamaConfig::new().native_completion(MODEL),
        crate::test_utils::RecordingHttpClient::new(&body[..]),
        Mode::Unary,
    )
    .await;
    let totals: Vec<Option<u64>> = observed.iter().map(|usage| usage.total_tokens).collect();
    assert_eq!(totals, [None], "Ollama reports no total_tokens");
    assert_eq!(observed, [crate::wire::AdapterUsage::from(&usage)]);
}

mod protected;

mod raw_capture;
