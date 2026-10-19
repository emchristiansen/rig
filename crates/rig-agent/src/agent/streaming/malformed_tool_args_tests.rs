//! A completed tool call whose arguments are not a JSON object never reaches the tool.
//! The tool step offers it to the invalid-call hook, then answers it with a
//! tool result so the model can call again. A model that keeps sending
//! malformed arguments is answered until `max_turns`, or stopped after
//! `max_consecutive_malformed_tool_calls` consecutive turns when that is set.
//! Original fragmented malformed inputs are strict refusals before the
//! invalid-call hook. Blocking policy controls retain their assertions;
//! separately named completed-response Streaming controls observe those
//! policies without changing the fragmented fixture producer.

use std::sync::{Arc, Mutex};

use super::MultiTurnStreamItem;
use crate::AgentRun;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::{
    AgentBuilder, AgentRunner, InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason,
};
use crate::run::{
    AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, TurnPolicy, UnhandledInvalidToolCall,
};
use crate::test_utils::{MockAddTool, MockCompletionModel, MockStreamEvent, MockTurn};
use futures::StreamExt;
use rig_core::completion::{CompletionRequest, Usage};
use rig_core::message::{
    AssistantContent, AssistantMessage, Message, ToolCall, ToolChoice, ToolFunction, ToolName,
    ToolResult, ToolResultContent, UserContent,
};
use serde_json::json;

const RAW: &str = "{\"x\": 2, \"y\": \x01";

/// One scripted model turn.
#[derive(Clone, Copy)]
enum Turn {
    /// A call to `add` whose arguments are [`RAW`].
    Malformed,
    /// A call to `add` with valid arguments.
    WellFormed,
    /// The final answer, `recovered`.
    Answer,
}

#[derive(Clone, Copy, Debug)]
enum Surface {
    Blocking,
    Streamed,
}

const SURFACES: [Surface; 2] = [Surface::Blocking, Surface::Streamed];

fn call_id(turn: usize) -> String {
    format!("tool_call_{turn}")
}

fn add() -> ToolName {
    ToolName::new("add").unwrap_or_else(|error| panic!("{error}"))
}

fn model(surface: Surface, script: &[Turn]) -> MockCompletionModel {
    match surface {
        Surface::Blocking => MockCompletionModel::from_turns(script.iter().enumerate().map(
            |(index, turn)| match turn {
                Turn::Malformed => MockTurn::from_content(AssistantContent::ToolCall(
                    ToolCall::from_wire(call_id(index + 1), ToolFunction::parse(add(), RAW)),
                )),
                Turn::WellFormed => {
                    MockTurn::tool_call(call_id(index + 1), "add", json!({"x": 1, "y": 2}))
                }
                Turn::Answer => MockTurn::text("recovered"),
            },
        )),
        Surface::Streamed => MockCompletionModel::from_stream_turns(script.iter().enumerate().map(
            |(index, turn)| {
                let id = call_id(index + 1);
                let mut events = match turn {
                    Turn::Malformed => vec![
                        MockStreamEvent::tool_call_name_delta(&id, "add"),
                        MockStreamEvent::tool_call_arguments_delta(&id, RAW),
                        MockStreamEvent::tool_call_end(&id),
                    ],
                    Turn::WellFormed => {
                        vec![MockStreamEvent::tool_call(
                            &id,
                            "add",
                            json!({"x": 1, "y": 2}),
                        )]
                    }
                    Turn::Answer => vec![MockStreamEvent::text("recovered")],
                };
                events.push(MockStreamEvent::final_response_with_total_tokens(4));
                events
            },
        )),
    }
}

/// Answers every invalid call with `action` and records what it saw. With a
/// `choice`, it also patches every turn's tool choice to it.
#[derive(Clone, Default)]
struct Decide {
    action: Option<InvalidToolCallAction>,
    choice: Option<ToolChoice>,
    seen: Arc<Mutex<Vec<InvalidToolCallContext>>>,
}

impl Decide {
    fn new(action: Option<InvalidToolCallAction>) -> Self {
        Self {
            action,
            choice: None,
            seen: Arc::default(),
        }
    }

    fn patching(mut self, choice: ToolChoice) -> Self {
        self.choice = Some(choice);
        self
    }

    fn seen(&self) -> Vec<InvalidToolCallContext> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl AgentHook for Decide {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::CompletionCallEvent<'_>,
    ) -> crate::agent::CompletionCallAction {
        match &self.choice {
            Some(choice) => crate::agent::CompletionCallAction::patch(
                crate::agent::RequestPatch::new().tool_choice(choice.clone()),
            ),
            None => crate::agent::CompletionCallAction::Continue,
        }
    }

    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(context.clone());
        self.action.clone()
    }
}

/// The run's output, or its error as text.
async fn drive(surface: Surface, runner: AgentRunner) -> Result<String, String> {
    match surface {
        Surface::Blocking => runner
            .run()
            .await
            .map(|response| response.output())
            .map_err(|error| error.to_string()),
        Surface::Streamed => {
            let mut stream = runner.stream();
            let mut output = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        output = Some(response.output());
                    }
                    Ok(_) => {}
                    Err(error) => return Err(error.to_string()),
                }
            }
            output.ok_or_else(|| "no final response".to_owned())
        }
    }
}

/// Run `script` on `surface` with `configure` applied to the runner.
async fn run(
    surface: Surface,
    script: &[Turn],
    configure: impl FnOnce(AgentRunner) -> AgentRunner,
) -> (Result<String, String>, Vec<CompletionRequest>) {
    let model = model(surface, script);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let runner = configure(agent.prompt("add 2 and something").max_turns(10));
    let outcome = match surface {
        Surface::Blocking => drive(surface, runner).await,
        Surface::Streamed => drive_fragmented_fixture(runner).await,
    };
    (outcome, recorded.requests())
}

/// The result answering the call of model turn `turn` in `history`.
fn answer(history: &[Message], turn: usize) -> Option<&ToolResult> {
    let id = call_id(turn);
    history.iter().find_map(|message| match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::ToolResult(result)
                if result.call.provider().map(|id| id.as_str()) == Some(id.as_str()) =>
            {
                Some(result)
            }
            _ => None,
        }),
        Message::System { .. } | Message::Assistant(_) => None,
    })
}

fn answer_text(requests: &[CompletionRequest], turn: usize) -> String {
    let request = requests
        .get(turn)
        .unwrap_or_else(|| panic!("request {} was made", turn + 1));
    let result = answer(&request.chat_history, turn)
        .unwrap_or_else(|| panic!("call {turn} is answered: {:?}", request.chat_history));
    assert!(result.is_error, "the answer is an error result: {result:?}");
    result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => text.text.clone(),
            other => panic!("a text answer, not {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn by_default_a_malformed_call_is_answered_with_feedback_and_the_run_continues() {
    for surface in [Surface::Blocking] {
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| r).await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(
            requests.len(),
            2,
            "{surface:?}: the model gets another turn"
        );
        let text = answer_text(&requests, 1);
        assert!(
            text.starts_with("The arguments for tool `add` are not a JSON object: ")
                && text.contains(RAW),
            "{surface:?}: the answer names the problem and the arguments: {text}"
        );
    }
}

#[tokio::test]
async fn the_hook_sees_the_reason_and_the_raw_arguments() {
    for surface in [Surface::Blocking] {
        let hook = Decide::new(None);
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        let seen = hook.seen();
        let [context] = seen.as_slice() else {
            panic!("{surface:?}: the hook is consulted once, saw {seen:?}");
        };
        assert_eq!(context.tool_name, "add");
        assert_eq!(context.args.as_deref(), Some(RAW));
        assert!(
            matches!(
                &context.reason,
                InvalidToolCallReason::MalformedArguments { error } if !error.is_empty()
            ),
            "{surface:?}: {:?}",
            context.reason
        );
        assert_eq!(
            context
                .tool_call_id
                .as_ref()
                .and_then(|id| id.provider())
                .map(|id| id.as_str()),
            Some("tool_call_1")
        );
        assert_eq!(context.available_tools, ["add"]);
        assert_eq!(context.allowed_tools, ["add"]);
        assert_eq!(
            context.is_streaming,
            matches!(surface, Surface::Streamed),
            "{surface:?}"
        );
        assert!(
            context.chat_history.iter().any(|message| matches!(
                message,
                Message::Assistant(AssistantMessage { content, .. })
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            )),
            "{surface:?}: the diagnostic history holds the committed turn"
        );
        // No action leaves the default feedback.
        assert!(answer_text(&requests, 1).contains("not a JSON object"));
    }
}

#[tokio::test]
async fn retry_answers_with_the_hooks_feedback() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::retry(
            "arguments were not JSON; try again",
        )));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
            continue;
        }
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(
            answer_text(&requests, 1),
            "arguments were not JSON; try again"
        );
    }
}

#[tokio::test]
async fn skip_answers_with_a_skipped_result() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::skip("add: skipped")));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
            continue;
        }
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(answer_text(&requests, 1), "add: skipped");
    }
}

#[tokio::test]
async fn stop_ends_the_run_with_the_hooks_reason() {
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::stop("operator halted")));
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
            continue;
        }
        let error = outcome.expect_err("stop ends the run");
        assert!(error.contains("operator halted"), "{surface:?}: {error}");
        assert_eq!(requests.len(), 1, "{surface:?}");
    }
}

/// `Fail` ends the run naming the tool and the parse error, and `Repair` is
/// refused the same way: renaming the tool cannot fix its arguments.
#[tokio::test]
async fn fail_and_repair_end_the_run_naming_the_tool_and_the_parse_error() {
    for action in [
        InvalidToolCallAction::fail(),
        InvalidToolCallAction::repair("add"),
    ] {
        for surface in SURFACES {
            let hook = Decide::new(Some(action.clone()));
            let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
            if matches!(surface, Surface::Streamed) {
                assert!(outcome.is_err(), "the original fatal fragment is refused");
                assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
                assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
                continue;
            }
            let error = outcome.expect_err("the run fails");
            assert!(
                error.contains("tool `add` was called with arguments that are not a JSON object: ")
                    && error.contains("at line 1 column"),
                "{surface:?} {action:?}: {error}"
            );
            assert_eq!(requests.len(), 1, "{surface:?} {action:?}");
        }
    }
}

#[tokio::test]
async fn without_a_limit_malformed_arguments_are_answered_until_max_turns() {
    let script = [Turn::Malformed; 10];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| r).await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        let error = outcome.expect_err("the turn budget ends the run");
        assert!(
            error.contains("reached the max turns limit of 10"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 10, "{surface:?}");
        for turn in 1..10 {
            assert!(
                answer_text(&requests, turn).contains("not a JSON object"),
                "{surface:?}: call {turn} is answered with feedback"
            );
        }
    }
}

#[tokio::test]
async fn a_model_that_keeps_sending_malformed_arguments_stops_after_the_limit() {
    let script = [Turn::Malformed; 8];
    for surface in SURFACES {
        let hook = Decide::new(Some(InvalidToolCallAction::retry("again")));
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .add_hook(hook.clone())
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
            continue;
        }
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("tool `add` was called with arguments that are not a JSON object on 4 consecutive turns")
                && !error.contains("max turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(
            requests.len(),
            4,
            "{surface:?}: three retries, then the limit"
        );
        assert_eq!(hook.seen().len(), 3, "{surface:?}: hook retries count");
    }
}

#[tokio::test]
async fn the_limit_is_configurable() {
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &[Turn::Malformed; 3], |r| {
            r.max_consecutive_malformed_tool_calls(1)
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 2 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 2, "{surface:?}");
    }
}

/// `None` clears a limit set earlier, on the runner and on the run.
#[tokio::test]
async fn none_clears_the_limit() {
    let script = [
        Turn::Malformed,
        Turn::Malformed,
        Turn::Malformed,
        Turn::Answer,
    ];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(1)
                .max_consecutive_malformed_tool_calls(None)
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
    let run = AgentRun::new("add")
        .max_consecutive_malformed_tool_calls(1)
        .max_consecutive_malformed_tool_calls(None);
    let json = serde_json::to_value(&run).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(json["max_consecutive_malformed_tool_calls"], json!(null));
}

#[tokio::test]
async fn a_well_formed_call_between_malformed_ones_resets_the_count() {
    use Turn::{Answer, Malformed, WellFormed};
    let script = [
        Malformed, Malformed, Malformed, WellFormed, Malformed, Malformed, Malformed, Answer,
    ];
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 8, "{surface:?}");
        assert!(!answer(&requests[4].chat_history, 4).is_some_and(|result| result.is_error));
    }
}

/// `Ignore` governs unknown and disallowed names only; it does not lift a
/// malformed-arguments limit.
#[tokio::test]
async fn ignore_does_not_lift_the_limit() {
    let mut script = vec![Turn::Malformed; 6];
    script.push(Turn::Answer);
    for surface in SURFACES {
        let (outcome, requests) = run(surface, &script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
}

/// A run persisted after two malformed turns resumes with its count and
/// limit: two more malformed turns pass its limit of three.
#[tokio::test]
async fn the_count_survives_a_serialize_and_resume() {
    let mut run = AgentRun::new("add 2 and something")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(3);
    for turn in 1..=2 {
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallModel { .. })
        ));
        let call = ToolCall::from_wire(call_id(turn), ToolFunction::parse(add(), RAW));
        let outcome = run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![AssistantContent::ToolCall(call.clone())],
            Usage::default(),
            TurnPolicy::new(["add".to_owned()].into(), None, None).expect("policy"),
            json!({}),
        ));
        assert!(matches!(outcome, Ok(ModelTurnOutcome::Continue { .. })));
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallTools { .. })
        ));
        run.tool_results(vec![UserContent::ToolResult(
            call.error_result(vec![ToolResultContent::text("not a JSON object")]),
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }
    let saved = serde_json::to_string(&run).unwrap_or_else(|error| panic!("{error}"));

    for surface in SURFACES {
        let restored: AgentRun =
            serde_json::from_str(&saved).unwrap_or_else(|error| panic!("{error}"));
        let model = model(surface, &[Turn::Malformed, Turn::Malformed, Turn::Answer]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let outcome = match surface {
            Surface::Blocking => drive(surface, agent.resume(restored)).await,
            Surface::Streamed => drive_fragmented_fixture(agent.resume(restored)).await,
        };
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                recorded.request_count(),
                1,
                "no recovery request on the fragmented stage"
            );
            continue;
        }
        let error = outcome.expect_err("the persisted count reaches the limit");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(recorded.request_count(), 2, "{surface:?}");
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct Answer {
    answer: String,
}

/// Malformed arguments to the structured-output tool keep their own
/// reprompt budget: the invalid-call hook is not consulted.
#[tokio::test]
async fn the_structured_output_path_is_unchanged() {
    for surface in SURFACES {
        let output_raw = "{\"answer\":";
        let model = match surface {
            Surface::Blocking => MockCompletionModel::from_turns([
                MockTurn::from_content(AssistantContent::ToolCall(ToolCall::from_wire(
                    "out1",
                    ToolFunction::parse(
                        ToolName::new("final_result").unwrap_or_else(|error| panic!("{error}")),
                        output_raw,
                    ),
                ))),
                MockTurn::tool_call("out2", "final_result", json!({"answer": "done"})),
            ]),
            Surface::Streamed => MockCompletionModel::from_stream_turns([
                vec![
                    MockStreamEvent::tool_call_name_delta("out1", "final_result"),
                    MockStreamEvent::tool_call_arguments_delta("out1", output_raw),
                    MockStreamEvent::tool_call_end("out1"),
                    MockStreamEvent::final_response_with_total_tokens(4),
                ],
                vec![
                    MockStreamEvent::tool_call("out2", "final_result", json!({"answer": "done"})),
                    MockStreamEvent::final_response_with_total_tokens(4),
                ],
            ]),
        };
        let recorded = model.clone();
        let hook = Decide::new(Some(InvalidToolCallAction::fail()));
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .build();
        let runner = agent
            .prompt("answer")
            .max_turns(3)
            .max_consecutive_malformed_tool_calls(0)
            .add_hook(hook.clone());
        let output = drive(surface, runner)
            .await
            .unwrap_or_else(|error| panic!("{surface:?}: {error}"));
        assert!(output.contains("done"), "{surface:?}: {output}");
        assert!(
            hook.seen().is_empty(),
            "{surface:?}: the hook is not consulted"
        );
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2, "{surface:?}");
        let reprompt = serde_json::to_string(&requests[1].chat_history)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            reprompt.contains("not a JSON object"),
            "{surface:?}: {reprompt}"
        );
    }
}

/// The malformed-arguments context, built while the call's tools are
/// pending, reports the tool choice the turn was sent with.
#[tokio::test]
async fn the_malformed_call_context_reports_the_patched_tool_choice() {
    for surface in SURFACES {
        let hook = Decide::new(None).patching(ToolChoice::Required);
        let (outcome, requests) = run(surface, &[Turn::Malformed, Turn::Answer], |r| {
            r.add_hook(hook.clone())
        })
        .await;
        if matches!(surface, Surface::Streamed) {
            assert!(outcome.is_err(), "the original fatal fragment is refused");
            assert_eq!(
                requests.len(),
                1,
                "no recovery request on the fragmented stage"
            );
            assert!(
                hook.seen().is_empty(),
                "no completed malformed call reached the hook"
            );
            continue;
        }
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        let choices: Vec<_> = hook.seen().into_iter().map(|c| c.tool_choice).collect();
        assert_eq!(choices, [Some(ToolChoice::Required)], "{surface:?}");
    }
}


/// The actual default engine must not execute even a tool that accepts {}.
/// The initial turn is the real Anthropic decoder's completed malformed call;
/// only the safe recovery answer uses the existing scripted model.
#[tokio::test]
async fn anthropic_first_fragment_overflow_never_executes_a_tool_at_the_default_retry_cap() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use crate::tool::{Tool, ToolContext};
    use rig_core::completion::FinishReason;
    use rig_core::message::StopReason;
    use rig_core::providers::anthropic::{Anthropic, completion::CLAUDE_SONNET_4_6};
    use rig_core::wire::{Decoder, Mode, Wire, WireEvent, WireFrame};

    struct CountingAdd(Arc<AtomicUsize>);
    impl Tool for CountingAdd {
        const NAME: &'static str = "add";
        // Accept {}, so deserialization cannot falsely hide tool execution.
        type Args = serde_json::Value;
        type Output = usize;
        type Error = crate::test_utils::MockToolError;
        fn description(&self) -> String {
            "Count actual add tool executions".to_owned()
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type":"object","properties":{}})
        }
        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.0.fetch_add(1, SeqCst);
            Ok(0)
        }
    }

    let wire = Anthropic::new("test-key").completion(CLAUDE_SONNET_4_6).wire;
    let oversized = "x".repeat(32 * 1024 * 1024 + 1);
    let block = json!({"type":"tool_use","id":"call_1","name":"add","input":{}});
    let delta = json!({"type":"content_block_delta","index":0,
        "delta":{"type":"input_json_delta","partial_json":oversized}});
    let raw_delta = delta.to_string();
    let WireEvent::Known(classified) = wire
        .decoder_in(Mode::Streaming)
        .classify(WireFrame::Text(raw_delta.clone()))
    else {
        panic!("the actual reader accepts the complete received delta");
    };
    assert_eq!(classified.fields, delta);
    assert_eq!(classified.fields["delta"]["partial_json"].as_str(), Some(oversized.as_str()));
    let frames = vec![
        json!({"type":"message_start","message":{
            "id":"msg_g217","type":"message","role":"assistant",
            "model":CLAUDE_SONNET_4_6,"content":[],"stop_reason":null,
            "stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}
        }}).to_string(),
        json!({"type":"content_block_start","index":0,"content_block":block}).to_string(),
        raw_delta,
        json!({"type":"content_block_stop","index":0}).to_string(),
        json!({"type":"message_delta",
            "delta":{"stop_reason":"tool_use","stop_sequence":null},
            "usage":{"output_tokens":0}}).to_string(),
    ];
    let response = rig_core::test_utils::history::decode(
        &wire, Mode::Streaming, frames.into_iter().map(WireFrame::Text),
    ).expect("the received block stop and tool_use terminal complete the turn");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(response.stop(), StopReason::ToolUse);
    assert_eq!(response.error, None);
    assert_eq!(response.origin.provider, rig_core::providers::anthropic::PROVIDER_NAME);
    assert_eq!(response.raw["content"], json!([block]));
    let calls = response.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.provider().expect("the received ID survives").as_str(), "call_1");
    assert_eq!(calls[0].function.name.as_str(), "add");
    assert_eq!(calls[0].function.namespace, None);
    assert_eq!(calls[0].function.arguments_value(), json!({}));
    assert_eq!(calls[0].function.invalid_arguments.as_deref(), Some(""));
    assert!(calls[0].native.is_none(), "the actual malformed stop clears Native evidence");

    for surface in SURFACES {
        let executions = Arc::new(AtomicUsize::new(0));
        let recovery = model(surface, &[Turn::Answer]);
        let recorded = recovery.clone();
        let agent = AgentBuilder::new(recovery)
            .tool(CountingAdd(executions.clone()))
            .build();
        let mut run = AgentRun::new("add things").max_turns(3);
        assert_eq!(serde_json::to_value(&run).expect("serialize run")["max_consecutive_malformed_tool_calls"], json!(null));
        let AgentRunStep::CallModel { turn: turn_number, .. } = run.next_step().expect("initial model step") else {
            panic!("the initial model call is pending");
        };
        let definitions = agent.tool_definitions(None).await.expect("real registered tool definitions");
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "add");
        run.advertise_tools(turn_number, definitions);
        let turn = ModelTurn::new(
            response.head(), response.choice.clone(), response.usage,
            TurnPolicy::new(["add".to_owned()].into(), None, None).expect("add policy"),
            response.raw.clone(),
        ).with_finish_reason(response.finish_reason());
        assert!(matches!(run.model_response(turn), Ok(ModelTurnOutcome::Continue { .. })));
        // No malformed retry-cap override and no invalid-call hook: resume
        // executes the real shared default engine's pending tool step.
        let outcome = drive(surface, agent.resume(run)).await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(executions.load(SeqCst), 0, "{surface:?}: no tool body executed during the whole run");
        let requests = recorded.requests();
        assert_eq!(requests.len(), 1, "{surface:?}: only the safe recovery model call is scripted");
        let results = requests[0].chat_history.iter().flat_map(|message| match message {
            Message::User { content } => content.as_slice(),
            Message::System { .. } | Message::Assistant(_) => &[],
        }).filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(result),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(results.len(), 1, "{surface:?}: the pending malformed call is actually answered");
        let result = results[0];
        assert_eq!(result.call.provider().expect("received call ID").as_str(), "call_1");
        assert_eq!(result.name.as_str(), "add");
        assert!(result.is_error, "{surface:?}: the default handler answers malformed input");
        let feedback = result.content.iter().map(|content| match content {
            ToolResultContent::Text(text) => text.text.as_str(),
            other => panic!("expected default text feedback, got {other:?}"),
        }).collect::<String>();
        assert_eq!(feedback,
            "The arguments for tool `add` are not a JSON object: . \
             Call the tool again with a JSON object as its arguments.");
    }
}

// G234 originally separated two subjects. The additional completed-response
// controls below preserve the other policies on that same received-turn stage.
// model() keeps every original fragmented fixture unchanged.
#[derive(Clone)]
struct CompletedFrameTransport {
    turns: Arc<Mutex<std::collections::VecDeque<rig_core::completion::CompletionResponse>>>,
    requests: Arc<Mutex<Vec<(rig_core::wire::Mode, CompletionRequest)>>>,
}

impl rig_core::driver::Transport<rig_core::test_utils::MockScript> for CompletedFrameTransport {
    fn send(
        &self,
        request: CompletionRequest,
        exchange: rig_core::driver::Exchange,
    ) -> rig_core::driver::Opening<rig_core::test_utils::MockFrame> {
        use rig_core::driver::{Opened, Opening};
        use rig_core::error::ProviderError;
        use rig_core::test_utils::MockFrame;
        if exchange.mode != rig_core::wire::Mode::Streaming {
            return Opening::failed(ProviderError::Provider(
                "G234 completed-response transport is a labelled streaming fixture".to_owned(),
            ));
        }
        self.requests.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((exchange.mode, request));
        let response = self.turns.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front();
        let Some(response) = response else {
            return Opening::failed(ProviderError::Provider("G234 fixture has no completed response left".to_owned()));
        };
        let document = response.raw.clone();
        let request_id = response.provider_request_id.clone();
        let headers = response.provider_response_headers.clone();
        Opening::ready(
            Opened::new(futures::stream::iter([Ok(MockFrame::Response(Box::new(response)))]))
                .with_document(document)
                .with_request_id(request_id)
                .with_captured_response_headers(headers),
        )
    }
}

/// Produce actual typed completions through the existing unary mock Model,
/// then deliver those completions as MockFrame::Response in Streaming mode.
/// No Origin/Native or success terminal is constructed by this fixture.
async fn run_completed_frame_stage(
    configure: impl FnOnce(AgentRunner) -> AgentRunner,
) -> (Result<String, String>, Vec<CompletionRequest>) {
    let unary = model(Surface::Blocking, &[Turn::Malformed, Turn::Answer]);
    let mut turns = std::collections::VecDeque::new();
    for _ in 0..2 {
        let response = unary.call(CompletionRequest::new("add 2 and something")).await
            .expect("the actual unary mock produces the completed typed fixture");
        assert_eq!(response.origin.provider, rig_core::test_utils::MOCK_PROVIDER);
        if turns.is_empty() {
            let calls = response.tool_calls().collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id.provider().expect("actual received call ID").as_str(), "tool_call_1");
            assert_eq!(calls[0].function.name.as_str(), "add");
            assert_eq!(calls[0].function.invalid_arguments.as_deref(), Some(RAW));
            assert!(calls[0].native.is_none());
        }
        turns.push_back(response);
    }
    let transport = CompletedFrameTransport {
        turns: Arc::new(Mutex::new(turns)),
        requests: Arc::default(),
    };
    let recorded = transport.clone();
    let completed = rig_core::driver::Model::new(unary.wire.clone(), transport);
    let agent = AgentBuilder::new(completed).tool(MockAddTool).build();
    let outcome = drive(Surface::Streamed,
        configure(agent.prompt("add 2 and something").max_turns(10)),
    ).await;
    let captured = recorded.requests.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    assert!(captured.iter().all(|(mode, _)| *mode == rig_core::wire::Mode::Streaming));
    (outcome, captured.into_iter().map(|(_, request)| request).collect())
}

/// G234 retains the exact original fragmented Streamed script as refusal,
/// separate from completed-turn malformed feedback/hook consumption.
async fn assert_original_fragmented_stage_refusal(with_hook: bool) {
    use rig_core::error::{ErrorDetail, ErrorKind};
    use rig_core::streaming::{Item, StreamEvent};
    let fragmented = model(Surface::Streamed, &[Turn::Malformed, Turn::Answer]);
    let recorded = fragmented.clone();
    let hook = Decide::new(None);
    let agent = AgentBuilder::new(fragmented).tool(MockAddTool).build();
    let mut runner = agent.prompt("add 2 and something").max_turns(10);
    if with_hook {
        runner = runner.add_hook(hook.clone());
    }
    let mut stream = runner.stream();
    let mut call_ends = 0;
    let mut completed_calls = 0;
    let mut final_answers = 0;
    let mut refusal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(_), ..
            }))) => call_ends += 1,
            Ok(MultiTurnStreamItem::CompletionCall(_)) => completed_calls += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => final_answers += 1,
            Ok(_) => {},
            Err(error) => { refusal = Some(error); break; },
        }
    }
    assert!(stream.next().await.is_none());
    assert_eq!(call_ends, 0);
    assert_eq!(completed_calls, 0, "the original producer never reaches its scripted terminal");
    assert_eq!(final_answers, 0);
    let crate::run::PromptError::Report(report) = refusal.expect("exact fragmented RAW is refused") else {
        panic!("the existing relayed malformed detail remains typed at the agent boundary");
    };
    assert_eq!(report.kind, ErrorKind::Response);
    assert_eq!(report.message, "tool call `add` arrived with malformed JSON input: expected value at line 1 column 15");
    let Some(ErrorDetail::MalformedToolInput(input)) = report.detail else { panic!("actual malformed detail"); };
    assert_eq!(input.name.as_str(), "add");
    assert_eq!(input.raw, RAW);
    assert_eq!(input.error, "expected value at line 1 column 15");
    assert_eq!(input.id.provider().expect("actual call ID").as_str(), "tool_call_1");
    let provider = input.provider.expect("the exact received correlation is retained");
    assert_eq!(provider.call_id.as_str(), "tool_call_1");
    assert_eq!(provider.item_id, None);
    assert_eq!(recorded.requests().len(), 1, "no recovery request was made on this original stage");
    assert!(hook.seen().is_empty(), "no completed malformed call reached the invalid-call hook");
}

#[tokio::test]
async fn the_original_default_fragmented_malformed_stream_is_a_strict_refusal() {
    assert_original_fragmented_stage_refusal(false).await;
}

#[tokio::test]
async fn the_original_hook_fragmented_malformed_stream_is_a_strict_refusal() {
    assert_original_fragmented_stage_refusal(true).await;
}

/// G234 input-stage exception: a real completed typed malformed turn is
/// streamed through MockFrame::Response; no fragmented-input parity claim.
#[tokio::test]
async fn completed_response_streaming_preserves_default_malformed_feedback_and_recovery() {
    let surface = Surface::Streamed;
    let (outcome, requests) = run_completed_frame_stage(|r| r).await;
    assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
    assert_eq!(requests.len(), 2, "{surface:?}: the model gets another turn");
    let text = answer_text(&requests, 1);
    assert!(
        text.starts_with("The arguments for tool `add` are not a JSON object: ") && text.contains(RAW),
        "{surface:?}: the answer names the problem and the arguments: {text}",
    );
}

/// G234 completed-response stage retains every original hook predicate and
/// independently pins the existing Required request patch in this new control.
#[tokio::test]
async fn completed_response_streaming_preserves_malformed_hook_context_and_patched_choice() {
    let surface = Surface::Streamed;
    for patched_choice in [None, Some(ToolChoice::Required)] {
        let mut hook = Decide::new(None);
        if let Some(choice) = patched_choice.clone() {
            hook = hook.patching(choice);
        }
        let (outcome, requests) = run_completed_frame_stage(|r| r.add_hook(hook.clone())).await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 2, "the actual completed stage reaches the recovery request");
        let seen = hook.seen();
        let [context] = seen.as_slice() else { panic!("{surface:?}: the hook is consulted once, saw {seen:?}"); };
        assert_eq!(context.tool_name, "add");
        assert_eq!(context.args.as_deref(), Some(RAW));
        assert!(matches!(&context.reason, InvalidToolCallReason::MalformedArguments { error } if !error.is_empty()),
            "{surface:?}: {:?}", context.reason);
        assert_eq!(context.tool_call_id.as_ref().and_then(|id| id.provider()).map(|id| id.as_str()), Some("tool_call_1"));
        assert_eq!(context.available_tools, ["add"]);
        assert_eq!(context.allowed_tools, ["add"]);
        assert_eq!(context.is_streaming, matches!(surface, Surface::Streamed), "{surface:?}");
        assert!(context.chat_history.iter().any(|message| matches!(message,
            Message::Assistant(AssistantMessage { content, .. }) if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
        )), "{surface:?}: the diagnostic history holds the committed turn");
        assert!(answer_text(&requests, 1).contains("not a JSON object"));
        assert_eq!(context.tool_choice, patched_choice);
        assert!(requests.iter().all(|request| request.tool_choice == patched_choice));
    }
}

// Positive fragmented-stream observer conjunctions beyond the original two
// G234 subjects are intentionally lost; these are completed-response controls.
// They establish neither legacy-producer nor full Streaming parity.

/// The original fatal fragment never reaches a completed turn or policy hook.
async fn drive_fragmented_fixture(runner: AgentRunner) -> Result<String, String> {
    use rig_core::error::{ErrorDetail, ErrorKind};
    use rig_core::streaming::{Item, StreamEvent};
    let mut stream = runner.stream();
    let mut call_ends = 0;
    let mut completed_calls = 0;
    let mut final_answers = 0;
    let mut refusal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            }))) => call_ends += 1,
            Ok(MultiTurnStreamItem::CompletionCall(_)) => completed_calls += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => final_answers += 1,
            Ok(_) => {}
            Err(error) => {
                refusal = Some(error);
                break;
            }
        }
    }
    assert!(stream.next().await.is_none());
    assert_eq!(call_ends, 0);
    assert_eq!(
        completed_calls, 0,
        "the original producer never reaches its scripted terminal"
    );
    assert_eq!(final_answers, 0);
    let error = refusal.expect("the original fragmented RAW is refused");
    let message = error.to_string();
    let crate::run::PromptError::Report(report) = error else {
        panic!("the fragmented malformed detail stays typed at the agent boundary");
    };
    assert_eq!(report.kind, ErrorKind::Response);
    assert_eq!(
        report.message,
        "tool call `add` arrived with malformed JSON input: expected value at line 1 column 15"
    );
    let Some(ErrorDetail::MalformedToolInput(input)) = report.detail else {
        panic!("actual malformed detail");
    };
    assert_eq!(input.name.as_str(), "add");
    assert_eq!(input.raw, RAW);
    assert_eq!(input.error, "expected value at line 1 column 15");
    assert_eq!(
        input
            .id
            .provider()
            .expect("actual received call ID")
            .as_str(),
        "tool_call_1"
    );
    let provider = input
        .provider
        .expect("the received correlation is retained");
    assert_eq!(provider.call_id.as_str(), "tool_call_1");
    assert_eq!(provider.item_id, None);
    Err(message)
}

impl CompletedFrameTransport {
    fn request_count(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// Build the original scripted unary completions before delivering their frames.
async fn completed_frame_model(
    script: &[Turn],
) -> (
    rig_core::driver::Model<rig_core::test_utils::MockScript, CompletedFrameTransport>,
    CompletedFrameTransport,
) {
    let unary = model(Surface::Blocking, script);
    let mut turns = std::collections::VecDeque::new();
    for (index, turn) in script.iter().enumerate() {
        let response = unary
            .call(CompletionRequest::new("add 2 and something"))
            .await
            .expect("the unary mock produces the actual completed typed fixture");
        assert_eq!(
            response.origin.provider,
            rig_core::test_utils::MOCK_PROVIDER
        );
        if matches!(turn, Turn::Malformed) {
            let calls = response.tool_calls().collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            assert_eq!(
                calls[0]
                    .id
                    .provider()
                    .expect("actual received call ID")
                    .as_str(),
                call_id(index + 1)
            );
            assert_eq!(calls[0].function.name.as_str(), "add");
            assert_eq!(calls[0].function.invalid_arguments.as_deref(), Some(RAW));
            assert!(calls[0].native.is_none());
        }
        turns.push_back(response);
    }
    let transport = CompletedFrameTransport {
        turns: Arc::new(Mutex::new(turns)),
        requests: Arc::default(),
    };
    let recorded = transport.clone();
    (
        rig_core::driver::Model::new(unary.wire.clone(), transport),
        recorded,
    )
}

async fn run_script_completed_frame_stage(
    script: &[Turn],
    configure: impl FnOnce(AgentRunner) -> AgentRunner,
) -> (Result<String, String>, Vec<CompletionRequest>) {
    let (completed, recorded) = completed_frame_model(script).await;
    let agent = AgentBuilder::new(completed).tool(MockAddTool).build();
    let outcome = drive(
        Surface::Streamed,
        configure(agent.prompt("add 2 and something").max_turns(10)),
    )
    .await;
    let captured = recorded
        .requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(
        captured
            .iter()
            .all(|(mode, _)| *mode == rig_core::wire::Mode::Streaming)
    );
    (
        outcome,
        captured.into_iter().map(|(_, request)| request).collect(),
    )
}

#[tokio::test]
async fn completed_response_streaming_keeps_retry_answers_with_the_hooks_feedback() {
    for surface in [Surface::Streamed] {
        let hook = Decide::new(Some(InvalidToolCallAction::retry(
            "arguments were not JSON; try again",
        )));
        let (outcome, requests) =
            run_script_completed_frame_stage(&[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(
            answer_text(&requests, 1),
            "arguments were not JSON; try again"
        );
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_skip_answers_with_a_skipped_result() {
    for surface in [Surface::Streamed] {
        let hook = Decide::new(Some(InvalidToolCallAction::skip("add: skipped")));
        let (outcome, requests) =
            run_script_completed_frame_stage(&[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(answer_text(&requests, 1), "add: skipped");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_stop_ends_the_run_with_the_hooks_reason() {
    for surface in [Surface::Streamed] {
        let hook = Decide::new(Some(InvalidToolCallAction::stop("operator halted")));
        let (outcome, requests) =
            run_script_completed_frame_stage(&[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
        let error = outcome.expect_err("stop ends the run");
        assert!(error.contains("operator halted"), "{surface:?}: {error}");
        assert_eq!(requests.len(), 1, "{surface:?}");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_fail_and_repair_end_the_run_naming_the_tool_and_the_parse_error()
 {
    for action in [
        InvalidToolCallAction::fail(),
        InvalidToolCallAction::repair("add"),
    ] {
        for surface in [Surface::Streamed] {
            let hook = Decide::new(Some(action.clone()));
            let (outcome, requests) =
                run_script_completed_frame_stage(&[Turn::Malformed, Turn::Answer], |r| {
                    r.add_hook(hook.clone())
                })
                .await;
            let error = outcome.expect_err("the run fails");
            assert!(
                error.contains("tool `add` was called with arguments that are not a JSON object: ")
                    && error.contains("at line 1 column"),
                "{surface:?} {action:?}: {error}"
            );
            assert_eq!(requests.len(), 1, "{surface:?} {action:?}");
        }
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_without_a_limit_malformed_arguments_are_answered_until_max_turns()
 {
    let script = [Turn::Malformed; 10];
    for surface in [Surface::Streamed] {
        let (outcome, requests) = run_script_completed_frame_stage(&script, |r| r).await;
        let error = outcome.expect_err("the turn budget ends the run");
        assert!(
            error.contains("reached the max turns limit of 10"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 10, "{surface:?}");
        for turn in 1..10 {
            assert!(
                answer_text(&requests, turn).contains("not a JSON object"),
                "{surface:?}: call {turn} is answered with feedback"
            );
        }
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_a_model_that_keeps_sending_malformed_arguments_stops_after_the_limit()
 {
    let script = [Turn::Malformed; 8];
    for surface in [Surface::Streamed] {
        let hook = Decide::new(Some(InvalidToolCallAction::retry("again")));
        let (outcome, requests) = run_script_completed_frame_stage(&script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .add_hook(hook.clone())
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("tool `add` was called with arguments that are not a JSON object on 4 consecutive turns")
                && !error.contains("max turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(
            requests.len(),
            4,
            "{surface:?}: three retries, then the limit"
        );
        assert_eq!(hook.seen().len(), 3, "{surface:?}: hook retries count");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_the_limit_is_configurable() {
    for surface in [Surface::Streamed] {
        let (outcome, requests) = run_script_completed_frame_stage(&[Turn::Malformed; 3], |r| {
            r.max_consecutive_malformed_tool_calls(1)
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 2 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 2, "{surface:?}");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_none_clears_the_limit() {
    let script = [
        Turn::Malformed,
        Turn::Malformed,
        Turn::Malformed,
        Turn::Answer,
    ];
    for surface in [Surface::Streamed] {
        let (outcome, requests) = run_script_completed_frame_stage(&script, |r| {
            r.max_consecutive_malformed_tool_calls(1)
                .max_consecutive_malformed_tool_calls(None)
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
    let run = AgentRun::new("add")
        .max_consecutive_malformed_tool_calls(1)
        .max_consecutive_malformed_tool_calls(None);
    let json = serde_json::to_value(&run).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(json["max_consecutive_malformed_tool_calls"], json!(null));
}

#[tokio::test]
async fn completed_response_streaming_keeps_a_well_formed_call_between_malformed_ones_resets_the_count()
 {
    use Turn::{Answer, Malformed, WellFormed};
    let script = [
        Malformed, Malformed, Malformed, WellFormed, Malformed, Malformed, Malformed, Answer,
    ];
    for surface in [Surface::Streamed] {
        let (outcome, requests) = run_script_completed_frame_stage(&script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
        })
        .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        assert_eq!(requests.len(), 8, "{surface:?}");
        assert!(!answer(&requests[4].chat_history, 4).is_some_and(|result| result.is_error));
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_ignore_does_not_lift_the_limit() {
    let mut script = vec![Turn::Malformed; 6];
    script.push(Turn::Answer);
    for surface in [Surface::Streamed] {
        let (outcome, requests) = run_script_completed_frame_stage(&script, |r| {
            r.max_consecutive_malformed_tool_calls(3)
                .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        })
        .await;
        let error = outcome.expect_err("the limit ends the run");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(requests.len(), 4, "{surface:?}");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_the_count_survives_a_serialize_and_resume() {
    let mut run = AgentRun::new("add 2 and something")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(3);
    for turn in 1..=2 {
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallModel { .. })
        ));
        let call = ToolCall::from_wire(call_id(turn), ToolFunction::parse(add(), RAW));
        let outcome = run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![AssistantContent::ToolCall(call.clone())],
            Usage::default(),
            TurnPolicy::new(["add".to_owned()].into(), None, None).expect("policy"),
            json!({}),
        ));
        assert!(matches!(outcome, Ok(ModelTurnOutcome::Continue { .. })));
        assert!(matches!(
            run.next_step(),
            Ok(AgentRunStep::CallTools { .. })
        ));
        run.tool_results(vec![UserContent::ToolResult(
            call.error_result(vec![ToolResultContent::text("not a JSON object")]),
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }
    let saved = serde_json::to_string(&run).unwrap_or_else(|error| panic!("{error}"));

    for surface in [Surface::Streamed] {
        let restored: AgentRun =
            serde_json::from_str(&saved).unwrap_or_else(|error| panic!("{error}"));
        let (model, recorded) =
            completed_frame_model(&[Turn::Malformed, Turn::Malformed, Turn::Answer]).await;
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let error = drive(surface, agent.resume(restored))
            .await
            .expect_err("the persisted count reaches the limit");
        assert!(
            error.contains("on 4 consecutive turns"),
            "{surface:?}: {error}"
        );
        assert_eq!(recorded.request_count(), 2, "{surface:?}");
    }
}

#[tokio::test]
async fn completed_response_streaming_keeps_the_malformed_call_context_reports_the_patched_tool_choice()
 {
    for surface in [Surface::Streamed] {
        let hook = Decide::new(None).patching(ToolChoice::Required);
        let (outcome, _) =
            run_script_completed_frame_stage(&[Turn::Malformed, Turn::Answer], |r| {
                r.add_hook(hook.clone())
            })
            .await;
        assert_eq!(outcome.as_deref(), Ok("recovered"), "{surface:?}");
        let choices: Vec<_> = hook.seen().into_iter().map(|c| c.tool_choice).collect();
        assert_eq!(choices, [Some(ToolChoice::Required)], "{surface:?}");
    }
}
