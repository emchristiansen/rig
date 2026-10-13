use super::*;
use base64::Engine as _;
use futures::StreamExt;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{AssistantContent, CarriedReasoningPart, Message};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::wire::{Mode, Operation as _, Wire as _, WireFrame};

type TestResult = Result<(), Box<dyn std::error::Error>>;
const THOUGHT_SIGNATURE: &[u8] = b"\x00\x01\x02thinking-text-sig\xff";
const ANSWER_SIGNATURE: &[u8] = b"signature";

fn fixture(
    signed: bool,
) -> (
    vertexai::model::GenerateContentResponse,
    Vec<vertexai::model::Part>,
) {
    let mut thought = vertexai::model::Part::new()
        .set_text("thinking text")
        .set_thought(true);
    if signed {
        thought = thought.set_thought_signature(THOUGHT_SIGNATURE.to_vec());
    }
    let parts = vec![
        thought,
        vertexai::model::Part::new()
            .set_text("4")
            .set_thought_signature(ANSWER_SIGNATURE.to_vec()),
    ];
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts.clone());
    let candidate = vertexai::model::Candidate::new()
        .set_content(content)
        .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
    let response = vertexai::model::GenerateContentResponse::new()
        .set_candidates([candidate])
        .set_model_version("gemini-2.5-flash-001")
        .set_response_id("resp-vertex-1");
    (response, parts)
}

#[derive(Clone)]
struct SingleReply(vertexai::model::GenerateContentResponse);

impl Transport<crate::completion::GenerateContent> for SingleReply {
    fn send(
        &self,
        _request: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

fn assert_carried(
    response: &CompletionResponse,
    parts: &[vertexai::model::Part],
    signed: bool,
    issuer: &str,
) -> TestResult {
    let [
        AssistantContent::CarriedReasoning(reasoning),
        AssistantContent::CarriedText(answer),
    ] = response.choice.as_slice()
    else {
        return Err("expected received reasoning and signed answer".into());
    };
    assert_eq!(reasoning.payload.provider.as_deref(), Some(issuer));
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "thinking text".to_owned(),
            signature: signed.then(|| base64::prelude::BASE64_STANDARD.encode(THOUGHT_SIGNATURE)),
        }]
    );
    assert_eq!(
        reasoning.payload.item,
        serde_json::to_value(parts.first().ok_or("thought missing")?)?
    );
    assert_eq!(answer.payload.provider.as_deref(), Some(issuer));
    assert_eq!(
        answer.payload.item,
        serde_json::to_value(parts.get(1).ok_or("answer missing")?)?
    );
    let known = answer.signed_answer().ok_or("signed answer lost")?;
    assert_eq!(known.text, "4");
    assert_eq!(
        known.signature.as_str(),
        base64::prelude::BASE64_STANDARD.encode(ANSWER_SIGNATURE)
    );
    assert_eq!(response.text(), "4");
    assert_eq!(response.model(), Some("gemini-2.5-flash-001"));
    assert_eq!(response.response_id(), Some("resp-vertex-1"));
    assert_eq!(
        response.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
    Ok(())
}

fn assert_vertex_history_replays(
    response: &CompletionResponse,
    parts: &[vertexai::model::Part],
) -> TestResult {
    assert_eq!(response.origin.provider, PROVIDER_NAME);
    assert_eq!(response.origin.api.as_str(), "vertexai.generate_content");
    assert_eq!(response.origin.model, crate::completion::GEMINI_2_5_FLASH);
    let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
    let history = vec![
        Message::user("What is 2 + 2?"),
        response.message().ok_or("received turn missing")?,
        Message::user("And 3 + 3?"),
    ];
    let request = rig_core::operation::Completion::prepare(
        CompletionRequest::from(history),
        &wire.describe(),
    )?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let replayed = encoded
        .contents
        .get(1)
        .ok_or("received assistant turn lost")?;
    assert_eq!(replayed.role, "model");
    assert_eq!(replayed.parts.as_slice(), parts);
    Ok(())
}

#[tokio::test]
async fn unary_vertex_carried_parts_keep_their_actual_issuer_and_replay() -> TestResult {
    for signed in [false, true] {
        let (received, parts) = fixture(signed);
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            SingleReply(received),
        );
        let response = model.call(CompletionRequest::new("hello")).await?;
        assert_carried(&response, &parts, signed, PROVIDER_NAME)?;
        assert_vertex_history_replays(&response, &parts)?;
    }
    Ok(())
}

#[tokio::test]
async fn streamed_unary_vertex_carried_events_and_response_keep_their_actual_issuer() -> TestResult
{
    for signed in [false, true] {
        let (received, parts) = fixture(signed);
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            SingleReply(received),
        );
        let mut stream = model.stream(CompletionRequest::new("hello"))?;
        let mut ends = Vec::new();
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::End { content, .. }) = item? {
                ends.push(content);
            }
        }
        let response = stream.finish().await?;
        assert_eq!(ends, response.choice);
        assert_carried(&response, &parts, signed, PROVIDER_NAME)?;
        assert_vertex_history_replays(&response, &parts)?;
    }
    Ok(())
}

#[test]
fn ordinary_gemini_decoder_keeps_its_existing_carried_issuer() -> TestResult {
    for signed in [false, true] {
        let (received, parts) = fixture(signed);
        let wire = rig_core::providers::gemini::GeminiConfig::new("test-key")
            .completion(crate::completion::GEMINI_2_5_FLASH);
        let frame = WireFrame::Text(Value::Object(rest_chunk(&received)?).to_string());
        let response = rig_core::test_utils::history_conformance::decode(
            &wire,
            &CompletionRequest::new("hello"),
            Mode::Unary,
            [frame],
        )?;
        assert_carried(
            &response,
            &parts,
            signed,
            rig_core::providers::gemini::completion::PROVIDER_NAME,
        )?;
        assert_eq!(
            response.origin.provider,
            rig_core::providers::gemini::completion::PROVIDER_NAME
        );
    }
    Ok(())
}
