//! The real protobuf adapter assigns pending signatures to media custody.
//! These complete replies never place a late signature after a typed image.

use crate::proto;
use anyhow::{Context, ensure};
use rig_core::completion::{CompletionResponse, FinishReason};
use rig_core::message::{AssistantContent, CarriedPosition, CarriedTextContent};
use serde_json::{Value, json};

fn signature(bytes: &[u8]) -> proto::Part {
    proto::Part {
        thought_signature: bytes.to_vec(),
        ..Default::default()
    }
}
fn media() -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::InlineData(proto::Blob {
            mime_type: "image/png".to_owned(),
            data: vec![0],
        })),
        ..Default::default()
    }
}
fn answer() -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text("answer".to_owned())),
        ..Default::default()
    }
}
fn reply(parts: Vec<proto::Part>) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                role: "model".to_owned(),
                parts,
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        ..Default::default()
    }
}
fn assert_media_answer(
    response: &CompletionResponse,
    raw: &Value,
    signature: Option<&str>,
    index: u64,
) -> anyhow::Result<()> {
    let [
        AssistantContent::CarriedText(image),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        anyhow::bail!("real GrpcAdapter must retain inspection media and unsigned answer");
    };
    let mut expected = json!({"inlineData":{"mimeType":"image/png","data":"AA=="}});
    if let Some(signature) = signature {
        let Some(fields) = expected.as_object_mut() else {
            anyhow::bail!("inspection media expected an object: {expected:?}");
        };
        fields.insert("thoughtSignature".to_owned(), json!(signature));
    }
    ensure!(
        image.content == CarriedTextContent::InspectionOnlyMedia(String::new()),
        "inspection media category differs: {image:?}"
    );
    ensure!(
        image.payload.item == expected,
        "inspection media differs: {image:?}"
    );
    ensure!(
        image.payload.provider.as_deref()
            == Some(rig_core::providers::gemini::completion::PROVIDER_NAME),
        "inspection media provider differs: {image:?}"
    );
    ensure!(
        image.payload.position
            == CarriedPosition::MessageContent {
                item_index: 0,
                part_index: index
            },
        "inspection media position differs: {image:?}"
    );
    ensure!(
        image.signed_answer().is_none(),
        "media gained a signed answer: {image:?}"
    );
    ensure!(text.text == "answer", "unsigned answer differs: {text:?}");
    let answer = response
        .choice
        .get(1)
        .with_context(|| format!("second response content is missing: {:?}", response.choice))?;
    ensure!(
        answer.native_item() == Some(&json!({"text":"answer"})),
        "unsigned answer native item differs: {answer:?}"
    );
    let answer_text = response.text();
    ensure!(
        answer_text == "answer",
        "response text differs: {answer_text:?}"
    );
    let finish = response.finish_reason();
    ensure!(
        finish == Some(FinishReason::Stop),
        "finish reason differs: {finish:?}"
    );
    ensure!(
        response.raw == *raw,
        "the actual unary protobuf document remains exact: actual {:?}, expected {raw:?}",
        response.raw
    );
    ensure!(
        !response
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::Image(_))),
        "unexpected typed image: {:?}",
        response.choice
    );
    Ok(())
}

#[test]
fn g218_grpc_leading_signature_belongs_to_inline_media() -> anyhow::Result<()> {
    let received = reply(vec![signature(b"sig"), media(), answer()]);
    let raw = crate::rest::to_rest(&received).context("actual proto3 JSON")?;
    ensure!(
        *raw.get("candidates")
            .and_then(|value| value.get(0))
            .and_then(|value| value.get("content"))
            .and_then(|value| value.get("parts"))
            .unwrap_or(&Value::Null)
            == json!([
                {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}}, {"text":"answer"}
            ]),
        "protobuf signature/media parts differ: {raw:?}"
    );
    let response = crate::completion::tests::complete(received)
        .context("real unary protobuf Model/GrpcAdapter/fold")?;
    assert_media_answer(&response, &raw, Some("c2ln"), 1)?;
    Ok(())
}

#[test]
fn g218_grpc_bare_signature_after_signed_inspection_targets_that_media_slot() -> anyhow::Result<()>
{
    let received = reply(vec![
        signature(b"sig"),
        media(),
        signature(b"tail"),
        answer(),
    ]);
    let raw = crate::rest::to_rest(&received).context("actual proto3 JSON")?;
    ensure!(
        *raw.get("candidates")
            .and_then(|value| value.get(0))
            .and_then(|value| value.get("content"))
            .and_then(|value| value.get("parts"))
            .unwrap_or(&Value::Null)
            == json!([
                {"thoughtSignature":"c2ln"}, {"inlineData":{"mimeType":"image/png","data":"AA=="}},
                {"thoughtSignature":"dGFpbA=="}, {"text":"answer"}
            ]),
        "protobuf signature/media parts differ: {raw:?}"
    );
    let response = crate::completion::tests::complete(received).context("real protobuf adapter")?;
    assert_media_answer(&response, &raw, Some("dGFpbA=="), 1)?;
    Ok(())
}

#[test]
fn g218_grpc_unsigned_media_keeps_its_existing_inspection_category() -> anyhow::Result<()> {
    let received = reply(vec![media(), answer()]);
    let raw = crate::rest::to_rest(&received).context("actual proto3 JSON")?;
    let response =
        crate::completion::tests::complete(received).context("real unsigned protobuf reply")?;
    assert_media_answer(&response, &raw, None, 0)?;
    Ok(())
}
