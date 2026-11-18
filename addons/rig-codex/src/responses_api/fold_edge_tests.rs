//! What a block of a streamed turn carries, and what the next request
//! replays for it, when the stream is not the clean case: an item that
//! never got its done event, a done item that disagrees with what streamed,
//! a stream cut before its terminal, and a block the caller edited.

use super::fold_contract_tests::{chatgpt_wire, following, streamed_turn, turn_items};
use rig_core::ProviderError;
use rig_core::completion::{AssistantContent, CompletionRequest, CompletionResponse};
use rig_core::message::{Message, Native, Reasoning, StopReason, Text, ToolCall};
use rig_core::wire::{Mode, WireFrame, decode_reply};
use serde_json::{Value, json};
use std::sync::Arc;

/// `events`, each numbered in sequence, as the frames of one stream.
fn frames(events: Vec<Value>) -> Vec<WireFrame> {
    events
        .into_iter()
        .enumerate()
        .map(|(sequence, mut event)| {
            event["sequence_number"] = json!(sequence);
            WireFrame::Text(event.to_string())
        })
        .collect()
}

fn response(status: &str, output: Value) -> Value {
    json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": status,
        "model": "gpt-5.4", "output": output,
    })
}

/// The provider's finished items of a turn that reasons, answers, and makes
/// a namespaced custom call and a namespaced function call.
fn finished() -> [Value; 4] {
    [
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "step 1"}], "encrypted_content": "enc_blob"}),
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "annotations": [], "text": "hello"}]}),
        json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c", "name": "apply_patch", "namespace": "editor", "input": "*** Begin Patch", "status": "completed"}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search", "namespace": "web", "arguments": "{\"q\":\"rig\"}", "status": "completed"}),
    ]
}

/// The same items as their done events state them when they disagree with
/// what streamed: other summary text, other text, other input, other
/// arguments.
fn disagreeing() -> [Value; 4] {
    [
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "step one"}], "encrypted_content": "enc_blob"}),
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "annotations": [], "text": "goodbye"}]}),
        json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c", "name": "apply_patch", "namespace": "editor", "input": "*** End Patch", "status": "completed"}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search", "namespace": "web", "arguments": "{\"q\":\"other\"}", "status": "completed"}),
    ]
}

/// The items as `response.output_item.added` announces them, empty.
fn announced() -> [Value; 4] {
    [
        json!({"type": "reasoning", "id": "rs_1", "summary": []}),
        json!({"type": "message", "id": "msg_1", "status": "in_progress", "role": "assistant", "content": []}),
        json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c", "name": "apply_patch", "namespace": "editor", "input": "", "status": "in_progress"}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_f", "name": "search", "namespace": "web", "arguments": "", "status": "in_progress"}),
    ]
}

/// The deltas that stream the item at `index` of [`finished`], in two
/// pieces.
fn deltas(index: usize) -> Vec<Value> {
    let (kind, item, part, pieces): (&str, &str, Option<&str>, [&str; 2]) = match index {
        0 => (
            "response.reasoning_summary_text.delta",
            "rs_1",
            Some("summary_index"),
            ["step ", "1"],
        ),
        1 => (
            "response.output_text.delta",
            "msg_1",
            Some("content_index"),
            ["hel", "lo"],
        ),
        2 => (
            "response.custom_tool_call_input.delta",
            "ctc_1",
            None,
            ["*** Begin", " Patch"],
        ),
        _ => (
            "response.function_call_arguments.delta",
            "fc_1",
            None,
            ["{\"q\":", "\"rig\"}"],
        ),
    };
    pieces
        .into_iter()
        .map(|delta| {
            let mut event =
                json!({"type": kind, "item_id": item, "output_index": index, "delta": delta});
            if let Some(part) = part {
                event[part] = json!(0);
            }
            event
        })
        .collect()
}

/// A stream of the first `count` items: each announced and streamed by its
/// deltas, then given `done` as its done item when there is one; then
/// `response.completed` with `output` when there is a terminal.
fn stream(count: usize, done: Option<&[Value; 4]>, output: Option<Value>) -> Vec<WireFrame> {
    let mut events =
        vec![json!({"type": "response.created", "response": response("in_progress", json!([]))})];
    for (index, added) in announced().iter().take(count).enumerate() {
        events.push(
            json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        );
        events.extend(deltas(index));
        if let Some(done) = done {
            events.push(json!({"type": "response.output_item.done", "output_index": index, "item": done[index]}));
        }
    }
    if let Some(output) = output {
        events
            .push(json!({"type": "response.completed", "response": response("completed", output)}));
    }
    frames(events)
}

fn fold(frames: Vec<WireFrame>) -> Result<CompletionResponse, ProviderError> {
    decode_reply(
        &chatgpt_wire(),
        &CompletionRequest::new("say hi"),
        Mode::Streaming,
        frames,
    )
}

/// The input items the request after `turn` sends for it, on a request that
/// declares tools.
fn replayed(next: CompletionRequest) -> Vec<Value> {
    let mut next = next;
    next.additional_params = Some(json!({
        "tools": [crate::responses_api::fidelity_fixtures::namespace()]
    }));
    crate::responses_api::input_items(&chatgpt_wire(), next).expect("encodes")
}

fn user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

/// The answers [`following`] gives the custom call and the function call.
fn answers() -> [Value; 2] {
    [
        json!({"type": "custom_tool_call_output", "call_id": "call_c", "name": "apply_patch", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
    ]
}

/// The four blocks of a folded turn of [`finished`]'s kinds.
fn blocks(turn: &CompletionResponse) -> (&Reasoning, &Text, &ToolCall, &ToolCall) {
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Text(text),
        AssistantContent::ToolCall(custom),
        AssistantContent::ToolCall(function),
    ] = turn.choice.as_slice()
    else {
        panic!(
            "expected reasoning, text and two calls, got {:?}",
            turn.choice
        );
    };
    (reasoning, text, custom, function)
}

fn item_of(native: &Option<Native>) -> Option<&Value> {
    native.as_ref().map(|native| &native.item)
}

fn stop_of(turn: &CompletionResponse) -> Option<StopReason> {
    let Some(Message::Assistant(message)) = turn.message() else {
        panic!("the turn has an assistant message");
    };
    message.stop
}

// ── a. no done event; the terminal restates the item ────────────────────

/// An item streamed with no done event, which the terminal restates, ends
/// with the terminal's item as its provider item, for every kind; the
/// modelled fields agree with it, and the turn replays as those items.
#[test]
fn an_item_never_done_takes_the_terminals_restatement_as_its_item() {
    let items = finished();
    let turn = fold(stream(4, None, Some(json!(items)))).expect("the turn folds");
    let (reasoning, text, custom, function) = blocks(&turn);

    assert_eq!(item_of(&reasoning.native), Some(&items[0]), "reasoning");
    assert_eq!(reasoning.text, "step 1");
    assert_eq!(
        reasoning
            .native
            .as_ref()
            .and_then(|native| native.reasoning_blob())
            .map(|blob| blob.value),
        Some("enc_blob"),
        "the ciphertext is the terminal's"
    );
    assert_eq!(item_of(&text.native), Some(&items[1]), "message");
    assert_eq!(text.text, "hello");
    assert_eq!(item_of(&custom.native), Some(&items[2]), "custom call");
    assert_eq!(
        custom.function.arguments.get("input"),
        Some(&json!("*** Begin Patch"))
    );
    assert_eq!(item_of(&function.native), Some(&items[3]), "function call");
    assert_eq!(function.function.arguments.get("q"), Some(&json!("rig")));
    assert_eq!(stop_of(&turn), Some(StopReason::ToolUse));

    let mut expected = vec![user("say hi")];
    expected.extend(items);
    expected.extend(answers());
    expected.push(user("next"));
    assert_eq!(replayed(following(&turn)), expected);
}

// ── b. no done event; the terminal omits the item ───────────────────────

/// Items streamed with no done event, which the terminal leaves out, carry
/// no provider item, of any kind. When they include calls, Rig marks the
/// turn an error ("a tool call it never finished") while its finish reason
/// still says it called tools, and the next request replays neither the
/// turn nor the answers to its calls: only the user messages remain.
#[test]
fn calls_never_done_and_left_out_of_the_terminal_drop_the_turn_and_its_answers() {
    let turn = fold(stream(4, None, Some(json!([])))).expect("the turn folds");
    let (reasoning, text, custom, function) = blocks(&turn);

    assert_eq!(reasoning.native, None, "reasoning: no item, no ciphertext");
    assert_eq!(reasoning.text, "step 1");
    assert_eq!(text.native, None, "message: no item");
    assert_eq!(text.text, "hello");
    assert_eq!(custom.native, None, "custom call: no item, no namespace");
    assert_eq!(
        custom.function.arguments.get("input"),
        Some(&json!("*** Begin Patch"))
    );
    assert_eq!(
        function.native, None,
        "function call: no item, no namespace"
    );
    assert_eq!(function.function.arguments.get("q"), Some(&json!("rig")));
    assert_eq!(
        stop_of(&turn),
        Some(StopReason::Error(
            "the provider ended the reply with a tool call it never finished".to_owned()
        ))
    );
    assert_eq!(
        turn.finish_reason(),
        Some(rig_core::completion::FinishReason::ToolCalls)
    );

    assert_eq!(
        replayed(following(&turn)),
        [user("say hi"), user("next")],
        "the errored turn and the answers to its calls are left out"
    );
}

/// Text and reasoning streamed with no done event, which the terminal
/// leaves out, carry no provider item. The turn ends as a stop; it replays
/// its text as a message rebuilt under an id Rig makes up (`msg_rig_1`, not
/// the provider's `msg_1`), and its reasoning not at all.
#[test]
fn text_and_reasoning_never_done_and_left_out_replay_as_rebuilt_text_alone() {
    let turn = fold(stream(2, None, Some(json!([])))).expect("the turn folds");
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Text(text),
    ] = turn.choice.as_slice()
    else {
        panic!("expected reasoning then text, got {:?}", turn.choice);
    };
    assert_eq!(reasoning.native, None);
    assert_eq!(reasoning.text, "step 1");
    assert_eq!(text.native, None);
    assert_eq!(text.text, "hello");
    assert_eq!(stop_of(&turn), Some(StopReason::Stop));

    assert_eq!(
        replayed(following(&turn)),
        [
            user("say hi"),
            json!({
                "type": "message", "id": "msg_rig_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "text": "hello"}]
            }),
            user("next"),
        ]
    );
}

// ── c. a done item that disagrees with what streamed ────────────────────

/// Where a done item states text, input or arguments that the deltas also
/// stated, differently, the done item wins, for every kind: the block's
/// modelled field holds the done item's value, the block's item is the done
/// item, and that is what replays. What streamed is gone. This is the rule
/// for a field both filled; a done item that leaves a field empty is not
/// tested here, and Rig fills an empty done message from the streamed text.
#[test]
fn a_done_item_that_disagrees_with_the_deltas_wins_in_the_field_and_the_item() {
    let done = disagreeing();
    let turn = fold(stream(4, Some(&done), Some(json!(done)))).expect("the turn folds");
    let (reasoning, text, custom, function) = blocks(&turn);

    assert_eq!(reasoning.text, "step one", "streamed `step 1`");
    assert_eq!(item_of(&reasoning.native), Some(&done[0]));
    assert_eq!(text.text, "goodbye", "streamed `hello`");
    assert_eq!(item_of(&text.native), Some(&done[1]));
    assert_eq!(
        custom.function.arguments.get("input"),
        Some(&json!("*** End Patch")),
        "streamed `*** Begin Patch`"
    );
    assert_eq!(item_of(&custom.native), Some(&done[2]));
    assert_eq!(
        function.function.arguments.get("q"),
        Some(&json!("other")),
        "streamed `rig`"
    );
    assert_eq!(function.function.invalid_arguments, None);
    assert_eq!(item_of(&function.native), Some(&done[3]));

    let mut expected = vec![user("say hi")];
    expected.extend(done);
    expected.extend(answers());
    expected.push(user("next"));
    assert_eq!(replayed(following(&turn)), expected);
}

/// After a done item that disagrees with the deltas, a terminal that
/// restates what streamed changes none of the fields the done item filled:
/// the done item stays in the field, the item and the replay. A terminal
/// can still supply what a done item lacks, reasoning ciphertext for one,
/// which is not tested here.
#[test]
fn a_terminal_restating_the_deltas_does_not_override_a_disagreeing_done_item() {
    let done = disagreeing();
    let turn = fold(stream(4, Some(&done), Some(json!(finished())))).expect("the turn folds");
    let (reasoning, text, custom, function) = blocks(&turn);

    assert_eq!(reasoning.text, "step one");
    assert_eq!(item_of(&reasoning.native), Some(&done[0]));
    assert_eq!(text.text, "goodbye");
    assert_eq!(item_of(&text.native), Some(&done[1]));
    assert_eq!(
        custom.function.arguments.get("input"),
        Some(&json!("*** End Patch"))
    );
    assert_eq!(item_of(&custom.native), Some(&done[2]));
    assert_eq!(function.function.arguments.get("q"), Some(&json!("other")));
    assert_eq!(item_of(&function.native), Some(&done[3]));

    let mut expected = vec![user("say hi")];
    expected.extend(done);
    expected.extend(answers());
    expected.push(user("next"));
    assert_eq!(replayed(following(&turn)), expected);
}

// ── d. a stream cut before its terminal ─────────────────────────────────

/// A stream cut before any terminal gives no message at all: the fold is
/// the error `Truncated`, whether or not its items were done. Nothing of
/// the turn reaches a caller of the fold.
#[test]
fn a_stream_cut_before_its_terminal_folds_to_truncated() {
    for (case, done) in [("never done", None), ("each done", Some(&finished()))] {
        let error = fold(stream(4, done, None)).expect_err(case);
        assert!(
            matches!(error, ProviderError::Truncated),
            "{case}: {error:?}"
        );
        assert_eq!(
            error.to_string(),
            "ResponseError: the reply ended before the provider ended it",
            "{case}"
        );
    }
}

// ── e. an edited block ──────────────────────────────────────────────────

/// What the request after the clean streamed turn replays when `edit` is
/// applied to each block of the turn first, and the edited turn's blocks.
fn replayed_after(edit: impl Fn(&mut AssistantContent)) -> (Vec<AssistantContent>, Vec<Value>) {
    let turn = streamed_turn();
    let mut next = following(&turn);
    let Message::Assistant(mut assistant) = (*next.chat_history[1]).clone() else {
        panic!("the turn follows the prompt");
    };
    assistant.content.iter_mut().for_each(edit);
    let blocks = assistant.content.clone();
    next.chat_history[1] = Arc::new(Message::Assistant(assistant));
    (blocks, replayed(next))
}

/// The replay of the clean streamed turn with `edited` in place of the item
/// at `index` of [`turn_items`].
fn clean_replay_with(index: usize, edited: Value) -> Vec<Value> {
    let mut items = turn_items();
    items[index] = edited;
    let mut expected = vec![user("say hi")];
    expected.extend(items);
    expected.extend([
        json!({"type": "custom_tool_call_output", "call_id": "call_c", "name": "apply_patch", "output": "ok"}),
        json!({"type": "function_call_output", "call_id": "call_f", "output": "ok", "status": "completed"}),
        json!({"type": "function_call_output", "call_id": "call_m", "output": "ok", "status": "completed"}),
        user("next"),
    ]);
    expected
}

/// An edited block keeps its stale item, which is no longer current.
fn assert_stale(block: &AssistantContent, item: &Value) {
    assert_eq!(block.native_item(), None, "the edit makes the item stale");
    let native = match block {
        AssistantContent::Text(text) => &text.native,
        AssistantContent::Reasoning(reasoning) => &reasoning.native,
        AssistantContent::ToolCall(call) => &call.native,
        other => panic!("no item slot: {other:?}"),
    };
    assert_eq!(
        item_of(native),
        Some(item),
        "the stale item is kept as it was"
    );
}

/// An edited text block replays rebuilt under its item's id: its one text
/// part is the edited text, and the refusal part the provider sent is gone.
/// The id is the item's; the role and status are written anew by Rig
/// (`assistant`, `completed`), which here are also what the item said, so
/// this test does not show a differing status surviving. Every neighbour
/// replays verbatim.
#[test]
fn an_edited_text_block_replays_rebuilt_under_its_id_without_its_refusal_part() {
    let (blocks, replay) = replayed_after(|block| {
        if let AssistantContent::Text(text) = block {
            text.text = "edited".to_owned();
        }
    });
    assert_stale(&blocks[2], &turn_items()[2]);
    assert_eq!(
        replay,
        clean_replay_with(
            2,
            json!({
                "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "text": "edited"}]
            })
        )
    );
}

/// A function call whose arguments are edited replays rebuilt under its
/// item's id and call id with the edited arguments; its namespace (`web`)
/// and its status are gone. Every neighbour replays verbatim.
#[test]
fn an_edited_function_call_replays_rebuilt_without_its_namespace_or_status() {
    let (blocks, replay) = replayed_after(|block| {
        if let AssistantContent::ToolCall(call) = block
            && call.function.name.as_str() == "search"
        {
            call.function
                .arguments
                .insert("q".to_owned(), json!("edited"));
        }
    });
    assert_stale(&blocks[4], &turn_items()[4]);
    assert_eq!(
        replay,
        clean_replay_with(
            4,
            json!({
                "type": "function_call", "id": "fc_1", "call_id": "call_f",
                "name": "search", "arguments": "{\"q\":\"edited\"}"
            })
        )
    );
}

/// A custom call whose input is edited replays rebuilt as a custom call
/// under its item's id and call id with the edited input; its namespace
/// (`editor`) and its status are gone. Every neighbour replays verbatim.
#[test]
fn an_edited_custom_call_replays_rebuilt_without_its_namespace_or_status() {
    let (blocks, replay) = replayed_after(|block| {
        if let AssistantContent::ToolCall(call) = block
            && call.function.name.as_str() == "apply_patch"
        {
            call.function
                .arguments
                .insert("input".to_owned(), json!("*** edited"));
        }
    });
    assert_stale(&blocks[3], &turn_items()[3]);
    assert_eq!(
        replay,
        clean_replay_with(
            3,
            json!({
                "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c",
                "name": "apply_patch", "input": "*** edited"
            })
        )
    );
}

/// Reasoning whose text is edited replays rebuilt under its item's id with
/// the edited text as its summary, and still carries the original
/// ciphertext: the new summary is sent beside the ciphertext that came with
/// the old one. The fixture's ciphertext is a placeholder string, so
/// nothing here shows how a provider treats the pair. Every neighbour
/// replays verbatim.
#[test]
fn edited_reasoning_replays_its_new_summary_beside_the_original_ciphertext() {
    let (blocks, replay) = replayed_after(|block| {
        if let AssistantContent::Reasoning(reasoning) = block {
            reasoning.text = "edited reasoning".to_owned();
        }
    });
    assert_stale(&blocks[0], &turn_items()[0]);
    assert_eq!(
        replay,
        clean_replay_with(
            0,
            json!({
                "type": "reasoning", "id": "rs_1", "encrypted_content": "enc_blob",
                "summary": [{"type": "summary_text", "text": "edited reasoning"}]
            })
        )
    );
}

/// With no edit, the same construction replays every item verbatim, so the
/// differences above are the edits'.
#[test]
fn an_unedited_turn_replays_every_item_verbatim() {
    let (blocks, replay) = replayed_after(|_| {});
    assert!(
        blocks
            .iter()
            .filter(|block| !matches!(block, AssistantContent::Opaque(_)))
            .all(|block| block.native_item().is_some())
    );
    assert_eq!(replay, clean_replay_with(0, turn_items()[0].clone()));
}
