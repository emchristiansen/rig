use super::*;
use futures::StreamExt as _;
use rig_agent::agent::{AgentBuilder, MultiTurnStreamItem};
use rig_agent::run::{
    AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PromptError, TurnPolicy,
};
use rig_core::completion::FinishReason;
use rig_core::message::{CarriedPosition, CarriedTextContent, StopReason};
use rig_core::streaming::{Item, StreamEvent};

const INSPECTION_FAILURE: &str =
    "the provider completed a turn containing only inspection media and no answer";

/// Scripted SDK input isolates an actual STOP from the original GM9
/// no-terminal input. Vertex has no cassette transport for this SDK boundary.
#[test]
fn stopped_thought_png_stays_inspectable_but_fails_ordinary_agent() {
    let received =
        create_parts_response([inline_data_part("image/png", vec![1, 2, 3]).set_thought(true)]);
    let expected_raw = Value::Object(rest_chunk(&received).expect("the SDK frame transcodes"));
    assert_eq!(expected_raw["candidates"][0]["finishReason"], json!("STOP"));
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received),
    );
    futures::executor::block_on(async {
        let response = model
            .call(CompletionRequest::new("hello"))
            .await
            .expect("the direct Model remains successful for inspection");
        assert_eq!(response.raw, expected_raw);
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.stop(), StopReason::Stop);
        assert_eq!(response.text(), "");
        let [AssistantContent::CarriedText(media)] = response.choice.as_slice() else {
            panic!("one inspection carrier: {:?}", response.choice);
        };
        assert_eq!(
            media.content,
            CarriedTextContent::InspectionOnlyMedia(String::new())
        );
        assert_eq!(media.known_text_projection(), Some(""));
        assert!(media.signed_answer().is_none());
        assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            media.payload.position,
            CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0,
            }
        );
        assert_eq!(
            media.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][0]
        );
        assert_eq!(media.payload.item["thought"], json!(true));
        assert_eq!(
            media.payload.item["inlineData"]["mimeType"],
            json!("image/png")
        );
        assert_eq!(media.payload.item["inlineData"]["data"], json!("AQID"));
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                media.payload.item["inlineData"]["data"]
                    .as_str()
                    .expect("SDK base64 string")
            )
            .expect("SDK base64 bytes"),
            vec![1, 2, 3]
        );
        assert_eq!(response.choice[0].native_item(), None);
        let message = response
            .message()
            .expect("the stopped inspection response retains history");
        assert!(
            matches!(message, rig_core::message::Message::Assistant(ref turn)
            if turn.content == response.choice && turn.stop == Some(StopReason::Stop))
        );

        let mut run = AgentRun::new("hello");
        assert!(matches!(
            run.next_step().expect("the agent requests the model"),
            AgentRunStep::CallModel { turn: 1, .. }
        ));
        let policy = TurnPolicy::new(Default::default(), None, None).expect("ordinary policy");
        assert!(matches!(
            run.model_response(ModelTurn::from_policy(&response, policy))
                .expect("the decoded response is ingested"),
            ModelTurnOutcome::Continue { .. }
        ));
        assert_eq!(run.accepted_turn_choice(), Some(response.choice.clone()));
        let error = run
            .next_step()
            .expect_err("inspection media cannot finalize an ordinary answer");
        assert!(
            matches!(error, PromptError::Provider(ProviderError::Response(ref message))
            if message == INSPECTION_FAILURE),
            "{error:?}"
        );
        let [call] = run.completion_calls() else {
            panic!("one completed provider call: {:?}", run.completion_calls());
        };
        assert_eq!(call.raw, expected_raw);
        assert_eq!(call.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw, expected_raw);
        assert_eq!(response.stop(), StopReason::Stop);

        let agent = AgentBuilder::new(model).build();
        let error = agent
            .prompt("hello")
            .run()
            .await
            .expect_err("the ordinary public agent also rejects the empty inspection answer");
        assert!(
            matches!(error, PromptError::Provider(ProviderError::Response(ref message))
            if message == INSPECTION_FAILURE),
            "{error:?}"
        );
    });
}

/// The existing SDK Reply transport makes the inspection event and real
/// STOP deterministic without a provider call or cassette mutation.
#[test]
fn stopped_thought_png_stream_preserves_inspection_before_agent_failure() {
    let received =
        create_parts_response([inline_data_part("image/png", vec![1, 2, 3]).set_thought(true)]);
    let expected_raw = Value::Object(rest_chunk(&received).expect("the SDK frame transcodes"));
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received),
    );
    futures::executor::block_on(async {
        let response = model
            .call(CompletionRequest::new("hello"))
            .await
            .expect("the direct Model remains successful for inspection");
        let agent = AgentBuilder::new(model).build();
        let mut stream = agent.prompt("hello").stream();
        let mut ended = Vec::new();
        let mut completed = Vec::new();
        let mut failures = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                    part,
                    content,
                    ..
                }))) => {
                    assert_eq!(failures, 0, "inspection precedes failure");
                    ended.push((part.index(), content.canonical()));
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(
                    StreamEvent::Discard { .. },
                ))) => {
                    panic!("the inspection part must remain available");
                }
                Ok(MultiTurnStreamItem::CompletionCall(call)) => {
                    assert_eq!(failures, 0, "completion accounting precedes failure");
                    completed.push(call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    panic!("inspection-only STOP cannot succeed: {response:?}");
                }
                Ok(_) => {}
                Err(error) => {
                    assert!(
                        matches!(error, PromptError::Provider(ProviderError::Response(ref message))
                        if message == INSPECTION_FAILURE),
                        "{error:?}"
                    );
                    failures += 1;
                }
            }
        }
        assert_eq!(failures, 1);
        assert_eq!(ended, vec![(0, response.choice[0].clone())]);
        let [call] = completed.as_slice() else {
            panic!("one successful low-level completion: {completed:?}");
        };
        assert_eq!(call.raw, expected_raw);
        assert_eq!(call.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw, expected_raw);
        assert_eq!(response.stop(), StopReason::Stop);
    });
}

/// Scripted SDK controls pin the existing mixed G94, typed-image, and
/// reasoning-only STOP behavior at the agent boundary without recording.
#[test]
fn stopped_mixed_inspection_images_and_reasoning_keep_ordinary_success() {
    let cases = [
        (
            vec![
                vertexai::model::Part::new().set_text("before"),
                inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
                inline_data_part("audio/wav", vec![0]),
                inline_data_part("image/gif", vec![0]).set_thought_signature(vec![1, 2, 3]),
                vertexai::model::Part::new().set_text("after"),
            ],
            "beforeafter",
        ),
        (
            vec![
                inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
                inline_data_part("image/png", vec![1, 2, 3]),
            ],
            "",
        ),
        (
            vec![
                vertexai::model::Part::new()
                    .set_text("thinking")
                    .set_thought(true),
            ],
            "",
        ),
    ];
    futures::executor::block_on(async {
        for (parts, text) in cases {
            let model = Model::new(
                crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
                Reply(create_parts_response(parts)),
            );
            let response = model
                .call(CompletionRequest::new("hello"))
                .await
                .expect("the control response decodes");
            assert_eq!(response.stop(), StopReason::Stop);
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            let mut run = AgentRun::new("hello");
            assert!(matches!(
                run.next_step().expect("the control requests the model"),
                AgentRunStep::CallModel { .. }
            ));
            let policy = TurnPolicy::new(Default::default(), None, None).expect("ordinary policy");
            assert!(matches!(
                run.model_response(ModelTurn::from_policy(&response, policy))
                    .expect("the control response is ingested"),
                ModelTurnOutcome::Continue { .. }
            ));
            let AgentRunStep::Done(output) = run.next_step().expect("the control stays successful")
            else {
                panic!("the control completes without another model or tool request");
            };
            assert_eq!(output.output(), text);
        }
    });
}
