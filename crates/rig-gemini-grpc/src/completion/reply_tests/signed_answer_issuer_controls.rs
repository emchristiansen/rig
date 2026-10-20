//! Real REST/protobuf received answers replay through the selected GenerateContent issuer.

use crate::{
    completion::{GEMINI_2_5_FLASH, GenerateContent},
    proto,
};
use anyhow::{Context, bail, ensure};
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::Model;
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantContent, AssistantMessage, CarriedPayload, CarriedText, CarriedTextContent,
    GeminiSignedAnswer, GeminiThoughtSignature, Message, SourceLocation,
    UnrepresentableOpaqueContent,
};
use rig_core::operation::Completion;
use rig_core::providers::gemini::{GeminiConfig, completion::PROVIDER_NAME as ISSUER};
use rig_core::wire::{Mode, Operation as _, Wire as _};
use serde_json::{Value, json};

const SIGNATURE: &str = "c2lnbmF0dXJl";
fn history(message: Message) -> CompletionRequest {
    CompletionRequest::from(vec![Message::user("q"), message, Message::user("next")])
}
fn grpc_response() -> anyhow::Result<CompletionResponse> {
    let received = proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                role: "model".to_owned(),
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("4".to_owned())),
                    thought_signature: b"signature".to_vec(),
                    ..Default::default()
                }],
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            citation_metadata: Some(proto::CitationMetadata {
                citation_sources: vec![proto::CitationSource {
                    start_index: Some(0),
                    end_index: Some(1),
                    uri: Some("https://example.com/arith".to_owned()),
                    license: Some("test-license".to_owned()),
                }],
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let raw = crate::rest::to_rest(&received)?;
    let response = crate::completion::tests::complete(received)?;
    ensure!(
        response.raw == raw,
        "received raw differs: {:?}, {raw:?}",
        response.raw
    );
    assert_received(&response)?;
    Ok(response)
}
fn rest_response() -> anyhow::Result<CompletionResponse> {
    let raw = json!({"candidates":[{"content":{"role":"model","parts":[
        {"text":"4","thoughtSignature":SIGNATURE}
    ]},"finishReason":"STOP","citationMetadata":{"citationSources":[{
        "startIndex":0,"endIndex":1,"uri":"https://example.com/arith"
    }]}}]});
    let response = futures::executor::block_on(
        Model::new(
            rig_core::providers::gemini::completion::GenerateContent::new(
                GeminiConfig::new("test-key"),
                GEMINI_2_5_FLASH,
            ),
            rig_core::test_utils::RecordingHttpClient::new(serde_json::to_vec(&raw)?),
        )
        .call(CompletionRequest::new("q")),
    )?;
    ensure!(
        response.raw == raw,
        "received raw differs: {:?}, {raw:?}",
        response.raw
    );
    assert_received(&response)?;
    Ok(response)
}
fn assert_received(response: &CompletionResponse) -> anyhow::Result<()> {
    let [AssistantContent::CarriedText(text)] = response.choice.as_slice() else {
        bail!("one real signed answer: {:?}", response.choice);
    };
    ensure!(
        text.payload.provider.as_deref() == Some(ISSUER),
        "expected issuer {ISSUER}, got {:?}",
        text.payload.provider
    );
    ensure!(
        text.payload.item == json!({"text":"4","thoughtSignature":SIGNATURE}),
        "expected the complete received signed-answer item, got {:?}",
        text.payload.item
    );
    ensure!(
        text.signed_answer()
            .context("the received text is a signed answer")?
            .text
            == "4",
        "expected signed answer text 4, got {text:?}"
    );
    ensure!(
        text.signed_answer()
            .context("the received text is a signed answer")?
            .signature
            .as_str()
            == SIGNATURE,
        "expected signature {SIGNATURE}, got {text:?}"
    );
    ensure!(
        text.citations().len() == 1,
        "expected one citation, got {:?}",
        text.citations()
    );
    ensure!(
        text.citations()
            .first()
            .context("the signed answer has a citation")?
            .span
            .as_ref()
            .context("the citation has a span")?
            .range()
            == (0..1),
        "expected citation span 0..1, got {:?}",
        text.citations()
    );
    let source = text
        .citations()
        .first()
        .context("the signed answer has a citation")?
        .sources
        .first()
        .context("the citation has a source")?;
    let expected_source = SourceLocation::Url {
        url: "https://example.com/arith".to_owned(),
    };
    ensure!(
        source.location == expected_source,
        "expected the citation URL https://example.com/arith, got {:?}",
        source.location
    );
    Ok(())
}
fn encode_grpc(
    message: Message,
    expected: Option<&[AssistantContent]>,
) -> anyhow::Result<proto::GenerateContentRequest> {
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let prepared = Completion::prepare(history(message), &wire.describe())?;
    if let Some(expected) = expected {
        let Some(Message::Assistant(turn)) = prepared.chat_history.get(1) else {
            bail!("answer turn retained: {:?}", prepared.chat_history);
        };
        ensure!(
            turn.content.as_slice() == expected,
            "received raw/content/citations do not change: {:?}, {expected:?}",
            turn.content
        );
    }
    Ok(wire.encode(prepared, Mode::Unary)?)
}
fn assert_grpc_answer(request: &proto::GenerateContentRequest) -> anyhow::Result<()> {
    ensure!(
        request.contents.len() == 3,
        "expected three contents, got {:?}",
        request.contents
    );
    let answer = request
        .contents
        .get(1)
        .context("the request has an answer turn")?;
    ensure!(
        answer.role == "model",
        "expected model role, got {:?}",
        answer.role
    );
    ensure!(
        answer.parts.len() == 1,
        "expected one answer part, got {:?}",
        answer.parts
    );
    let part = answer.parts.first().context("the answer has a part")?;
    ensure!(
        matches!(&part.data, Some(proto::part::Data::Text(text)) if text=="4"),
        "expected answer text 4, got {:?}",
        part.data
    );
    ensure!(!part.thought, "the answer is not a thought: {part:?}");
    ensure!(
        part.thought_signature == b"signature".to_vec(),
        "expected signature bytes {:?}, got {:?}",
        b"signature".to_vec(),
        part.thought_signature
    );
    Ok(())
}

#[test]
fn g227_actual_grpc_answer_replays_to_grpc_using_selected_issuer_not_transport_name()
-> anyhow::Result<()> {
    let response = grpc_response()?;
    ensure!(
        GenerateContent::new(GEMINI_2_5_FLASH).describe().name != ISSUER,
        "the protobuf transport name and shared selected issuer differ: {:?}, {ISSUER:?}",
        GenerateContent::new(GEMINI_2_5_FLASH).describe().name
    );
    let before_raw = response.raw.clone();
    let before_choice = response.choice.clone();
    let request = encode_grpc(
        response
            .message()
            .context("the received response has a message")?,
        Some(&before_choice),
    )?;
    assert_grpc_answer(&request)?;
    ensure!(
        response.raw == before_raw,
        "raw changed: {:?}, {before_raw:?}",
        response.raw
    );
    ensure!(
        response.choice == before_choice,
        "choice changed: {:?}, {before_choice:?}",
        response.choice
    );
    Ok(())
}

#[test]
fn g227_actual_grpc_answer_replays_to_rest_with_genuine_shared_issuer() -> anyhow::Result<()> {
    let response = grpc_response()?;
    let before = response.choice.clone();
    let raw = response.raw.clone();
    let wire = rig_core::providers::gemini::completion::GenerateContent::new(
        GeminiConfig::new("test-key"),
        GEMINI_2_5_FLASH,
    );
    let prepared = Completion::prepare(
        history(
            response
                .message()
                .context("the received response has a message")?,
        ),
        &wire.describe(),
    )?;
    let Some(Message::Assistant(turn)) = prepared.chat_history.get(1) else {
        bail!("answer retained: {:?}", prepared.chat_history);
    };
    ensure!(
        turn.content == before,
        "answer content changed: {:?}, {before:?}",
        turn.content
    );
    let encoded = wire.encode(prepared, Mode::Unary)?;
    let rig_core::wire::Body::Bytes(bytes) = encoded.request.body() else {
        bail!("actual REST byte body, got {:?}", encoded.request.body());
    };
    let body: Value = serde_json::from_slice(bytes)?;
    let contents = body
        .get("contents")
        .and_then(Value::as_array)
        .context("the REST body has a contents array")?;
    ensure!(
        contents.len() == 3,
        "expected three REST contents, got {contents:?}"
    );
    let parts = contents
        .get(1)
        .and_then(|content| content.get("parts"))
        .context("the REST answer has parts")?;
    ensure!(
        *parts == json!([{"text":"4","thought":false,"thoughtSignature":SIGNATURE}]),
        "expected the full signed REST answer parts, got {parts:?}"
    );
    ensure!(
        response.raw == raw,
        "raw changed: {:?}, {raw:?}",
        response.raw
    );
    ensure!(
        response.choice == before,
        "choice changed: {:?}, {before:?}",
        response.choice
    );
    Ok(())
}

#[test]
fn g227_actual_rest_answer_replays_to_grpc_with_genuine_shared_issuer() -> anyhow::Result<()> {
    let response = rest_response()?;
    let before = response.choice.clone();
    let raw = response.raw.clone();
    assert_grpc_answer(&encode_grpc(
        response
            .message()
            .context("the received response has a message")?,
        Some(&before),
    )?)?;
    ensure!(
        response.raw == raw,
        "raw changed: {:?}, {raw:?}",
        response.raw
    );
    ensure!(
        response.choice == before,
        "choice changed: {:?}, {before:?}",
        response.choice
    );
    Ok(())
}

#[test]
fn g227_grpc_unknown_issuer_and_opaque_canonical_controls_keep_existing_routes()
-> anyhow::Result<()> {
    // Explicit unissued canonical input; no received provenance is changed.
    let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(
        CarriedText::new(
            CarriedTextContent::GeminiSignedAnswer(GeminiSignedAnswer::new(
                "4",
                GeminiThoughtSignature::new(SIGNATURE),
            )),
            CarriedPayload::new(Value::Null, None),
        ),
    )]));
    assert_grpc_answer(&encode_grpc(message, None)?)?;
    let opaque = Message::Assistant(AssistantMessage::new(vec![AssistantContent::CarriedText(
        CarriedText::new(
            CarriedTextContent::Opaque(String::new()),
            CarriedPayload::new(json!({"futureField":{"n":1}}), None),
        ),
    )]));
    let wire = GenerateContent::new(GEMINI_2_5_FLASH);
    let error = match Completion::prepare(history(opaque), &wire.describe()) {
        Err(error) => error,
        Ok(prepared) => bail!("opaque remains refused, got {prepared:?}"),
    };
    ensure!(
        matches!(&error,ProviderError::Request(inner)
        if inner.downcast_ref::<UnrepresentableOpaqueContent>().is_some_and(|error|error.wire==wire.describe().name)),
        "expected the selected wire's typed opaque-content refusal, got {error:?}"
    );
    Ok(())
}
