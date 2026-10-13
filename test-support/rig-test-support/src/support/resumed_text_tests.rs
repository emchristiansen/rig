    use super::*;
    use bytes::Bytes;
    use rig_core::completion::CompletionRequest;
    use rig_core::driver::Model;
    use rig_core::providers::openai::OpenAIConfig;
    use rig_core::providers::openai::wire::CallerIdentity;
    use rig_core::streaming::Transcript;
    use rig_core::test_utils::MockStreamingClient;

    fn interleaved_text_stream() -> CompletionStream {
        let response = json!({"id":"resp_resume", "object":"response", "created_at":0,
            "status":"completed", "model":"gpt-5.4", "tools":[], "output":[
                {"type":"message", "id":"msg_a", "role":"assistant", "status":"completed",
                    "content":[{"type":"output_text", "text":"AC"}]},
                {"type":"message", "id":"msg_b", "role":"assistant", "status":"completed",
                    "content":[{"type":"output_text", "text":"B"}]}
            ]});
        let frames = [
            json!({"type":"response.output_text.delta", "sequence_number":0,
                "output_index":0, "content_index":0, "item_id":"msg_a", "delta":"A"}),
            json!({"type":"response.output_text.delta", "sequence_number":1,
                "output_index":1, "content_index":0, "item_id":"msg_b", "delta":"B"}),
            json!({"type":"response.output_text.delta", "sequence_number":2,
                "output_index":0, "content_index":0, "item_id":"msg_a", "delta":"C"}),
            json!({"type":"response.completed", "sequence_number":3, "response":response}),
        ];
        let sse: String = frames.iter().map(|frame| format!("data: {frame}\n\n")).collect();
        let identity = CallerIdentity::new("stream-consumer-test", "stream-consumer-test/1", None)
            .expect("explicit test identity");
        let model = Model::new(
            rig_core::providers::openai::responses_api::wire::Responses::new(
                OpenAIConfig::new("test-key").with_caller_identity(identity), "gpt-5.4",
            ),
            MockStreamingClient { sse_bytes: Bytes::from(sse) },
        );
        model.stream(CompletionRequest::new("fixture")).expect("actual mock stream opens")
    }

    #[tokio::test]
    async fn raw_collector_keeps_arrival_text_without_counting_resume_as_content() {
        let observation = collect_raw_stream_observation(interleaved_text_stream()).await;
        assert_eq!(observation.text, "ABC");
        assert_eq!(observation.events, ["text", "text", "text", "final"]);
        assert!(observation.errors.is_empty());
        assert!(observation.tool_calls.is_empty());
        assert!(observation.got_final);

        let updates = crate::updates::collect_updates(interleaved_text_stream()).await;
        let transcript = Transcript::parse(
            serde_json::to_value(&updates.items).expect("actual writer items serialize"),
        ).expect("actual resumed writer transcript closes");
        let resumed: Vec<_> = transcript.events().filter_map(|event| {
            if let StreamEvent::Resume { part } = event { Some(*part) } else { None }
        }).collect();
        assert_eq!(resumed.len(), 1);
        let part = resumed[0];
        assert_eq!(part.index(), 0);
        assert_eq!(transcript.events().filter(|event| {
            matches!(event, StreamEvent::Start { part: own, .. } if *own == part)
        }).count(), 1);
        assert_eq!(transcript.events().filter(|event| {
            matches!(event, StreamEvent::End { part: own, .. } if *own == part)
        }).count(), 1);
        let response = updates.response.expect("actual stream terminal");
        assert_eq!(response.choice.len(), 2);
        let texts: Vec<_> = response.choice.iter().map(|content| {
            match content {
                AssistantContent::Text(text) => text.text.as_str(),
                other => panic!("ordinary decoded text stays text: {other:?}"),
            }
        }).collect();
        assert_eq!(texts, ["AC", "B"]);
    }
