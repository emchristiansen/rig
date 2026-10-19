//! Late bare signatures settle the reserved SDK media slot before later content.
//! Multi-frame transports here are synthetic adapter controls, not network streams.

use super::super::{PROVIDER_NAME, rest_chunk};
use futures::{FutureExt as _, StreamExt as _};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantContent, CarriedPosition, CarriedTextContent, DocumentSourceKind, ImageMediaType,
};
use rig_core::streaming::{Item, PartKind, StreamEvent};
use serde_json::{Value, json};

type Frame = vertexai::model::GenerateContentResponse;
type TestResult = anyhow::Result<()>;

fn media(mime: &str) -> Value {
    json!({"inlineData":{"mimeType":mime,"data":"AA=="}})
}
fn signature(value: &str) -> Value {
    json!({"thoughtSignature":value})
}
fn body(parts: Vec<Value>, finish: Option<&str>) -> Value {
    let mut candidate = json!({"content":{"role":"model","parts":parts}});
    if let Some(finish) = finish {
        candidate["finishReason"] = json!(finish);
    }
    json!({"candidates":[candidate]})
}
fn frame(parts: Vec<Value>, finish: Option<&str>) -> Result<Frame, serde_json::Error> {
    // Actual SDK serde construction of new synthetic received fixtures only.
    serde_json::from_value(body(parts, finish))
}

#[derive(Clone)]
struct Scripted {
    frames: Vec<Result<Frame, ProviderError>>,
    document: Option<Value>,
}
impl Transport<crate::completion::GenerateContent> for Scripted {
    fn send(
        &self,
        _request: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<Frame> {
        let opened = Opened::new(futures::stream::iter(self.frames.clone()));
        Opening::ready(match &self.document {
            Some(raw) => opened.with_document(raw.clone()),
            None => opened,
        })
    }
}
fn model(script: Scripted) -> Model<crate::completion::GenerateContent, Scripted> {
    Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        script,
    )
}
struct Read {
    items: Vec<Result<Item<StreamEvent>, ProviderError>>,
    partial: CompletionResponse,
    outcome: Result<CompletionResponse, ProviderError>,
}
fn read(script: Scripted) -> Result<Read, ProviderError> {
    futures::executor::block_on(async move {
        let mut stream = model(script).stream(CompletionRequest::new("inspect"))?;
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        let partial = stream.partial();
        let outcome = stream.finish().await;
        Ok(Read {
            items,
            partial,
            outcome,
        })
    })
}
fn whole(received: Frame) -> Result<Scripted, ProviderError> {
    let document = Value::Object(rest_chunk(&received)?);
    Ok(Scripted {
        frames: vec![Ok(received)],
        document: Some(document),
    })
}
fn ends(read: &Read) -> Vec<&AssistantContent> {
    read.items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { content, .. })) => Some(content),
            _ => None,
        })
        .collect()
}
fn assert_order(read: &Read, kinds: &[PartKind]) {
    let starts: Vec<_> = read
        .items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::Start {
                part,
                source_order,
                kind,
                ..
            })) => Some((part.index(), *source_order, *kind)),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts,
        kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| (index, None, *kind))
            .collect::<Vec<_>>(),
        "genuine Part positions in SDK arrival order"
    );
    let positions: Vec<_> = read
        .items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { part, .. })) => Some(part.index()),
            _ => None,
        })
        .collect();
    assert_eq!(positions, (0..kinds.len()).collect::<Vec<_>>());
}
fn assert_inspection(content: &AssistantContent, mime: &str, sig: &str, position: u64) {
    let AssistantContent::CarriedText(image) = content else {
        panic!("late signed image must be the real inspection carrier");
    };
    assert_eq!(
        image.content,
        CarriedTextContent::InspectionOnlyMedia(String::new())
    );
    let mut expected = media(mime);
    expected["thoughtSignature"] = json!(sig);
    assert_eq!(image.payload.item, expected);
    assert_eq!(image.payload.provider.as_deref(), Some(PROVIDER_NAME));
    assert_eq!(
        image.payload.position,
        CarriedPosition::MessageContent {
            item_index: 0,
            part_index: position,
        }
    );
    assert!(image.signed_answer().is_none());
    assert!(content.native_item().is_none());
}
fn assert_image(content: &AssistantContent, mime: &str, expected: ImageMediaType) {
    let AssistantContent::Image(image) = content else {
        panic!("unsigned selected image");
    };
    assert_eq!(image.media_type, Some(expected));
    assert_eq!(image.data, DocumentSourceKind::Base64("AA==".to_owned()));
    assert_eq!(content.native_item(), Some(&media(mime)));
}
fn assert_unsigned_answer(content: &AssistantContent, text: &str) {
    let AssistantContent::Text(answer) = content else {
        panic!("unsigned canonical answer");
    };
    assert_eq!(answer.text, text);
    assert_eq!(content.native_item(), Some(&json!({"text":text})));
}

#[test]
fn g237_sdk_same_chunk_late_signature_keeps_media_not_the_next_answer() -> TestResult {
    let received = frame(
        vec![
            media("image/png"),
            signature("c2ln"),
            json!({"text":"answer"}),
        ],
        Some("STOP"),
    )?;
    let script = whole(received)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured SDK document"))?;
    anyhow::ensure!(
        raw == body(
            vec![
                media("image/png"),
                signature("c2ln"),
                json!({"text":"answer"})
            ],
            Some("STOP")
        ),
        "captured SDK document changed"
    );
    let unary =
        futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    let response = observed.outcome?;
    anyhow::ensure!(
        response.choice == unary.choice,
        "stream and unary choices differ"
    );
    anyhow::ensure!(response.raw == raw, "stream raw document changed");
    anyhow::ensure!(unary.raw == raw, "unary raw document changed");
    assert_inspection(&response.choice[0], "image/png", "c2ln", 0);
    assert_unsigned_answer(&response.choice[1], "answer");
    anyhow::ensure!(
        response.finish_reason() == Some(FinishReason::Stop),
        "stop finish reason changed"
    );
    Ok(())
}

#[test]
fn g237_sdk_split_chunks_retain_original_media_position_and_reconstructed_raw() -> TestResult {
    let script = Scripted {
        frames: vec![
            Ok(frame(vec![media("image/png")], None)?),
            Ok(frame(vec![signature("c2ln")], None)?),
            Ok(frame(vec![json!({"text":"answer"})], Some("STOP"))?),
        ],
        document: None,
    };
    let unary =
        futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    let response = observed.outcome?;
    anyhow::ensure!(
        response.choice == unary.choice,
        "stream and unary choices differ"
    );
    let mut signed = media("image/png");
    signed["thoughtSignature"] = json!("c2ln");
    let expected = body(vec![signed, json!({"text":"answer"})], Some("STOP"));
    anyhow::ensure!(
        response.raw == expected,
        "stream reconstructed raw document changed"
    );
    anyhow::ensure!(
        unary.raw == expected,
        "unary reconstructed raw document changed"
    );
    assert_inspection(&response.choice[0], "image/png", "c2ln", 0);
    assert_unsigned_answer(&response.choice[1], "answer");
    Ok(())
}

#[test]
fn g237_adjacent_signature_metadata_keeps_the_last_signature_until_content() -> TestResult {
    let received = frame(
        vec![
            media("image/png"),
            signature("c2ln"),
            signature("dGFpbA=="),
            json!({"text":"answer"}),
        ],
        Some("STOP"),
    )?;
    let script = whole(received)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured document"))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    let response = observed.outcome?;
    assert_inspection(&response.choice[0], "image/png", "dGFpbA==", 0);
    assert_unsigned_answer(&response.choice[1], "answer");
    anyhow::ensure!(response.raw == raw, "stream raw document changed");
    anyhow::ensure!(
        response.raw["candidates"][0]["content"]["parts"]
            == json!([
                media("image/png"),signature("c2ln"),signature("dGFpbA=="),{"text":"answer"}
            ]),
        "captured SDK parts changed"
    );
    Ok(())
}

#[test]
fn g237_all_five_unsigned_sdk_mimes_flush_on_next_content_or_eof() -> TestResult {
    for (mime, kind) in [
        ("image/jpeg", ImageMediaType::JPEG),
        ("image/png", ImageMediaType::PNG),
        ("image/webp", ImageMediaType::WEBP),
        ("image/heic", ImageMediaType::HEIC),
        ("image/heif", ImageMediaType::HEIF),
    ] {
        for with_answer in [false, true] {
            let mut parts = vec![media(mime)];
            if with_answer {
                parts.push(json!({"text":"answer"}));
            }
            let script = whole(frame(parts, Some("STOP"))?)?;
            let raw = script
                .document
                .clone()
                .ok_or_else(|| anyhow::anyhow!("captured document"))?;
            let observed = read(script)?;
            let mut kinds = vec![PartKind::Image];
            if with_answer {
                kinds.push(PartKind::Text);
            }
            assert_order(&observed, &kinds);
            let response = observed.outcome?;
            anyhow::ensure!(
                response.choice.len() == if with_answer { 2 } else { 1 },
                "choice count changed"
            );
            assert_image(&response.choice[0], mime, kind.clone());
            if with_answer {
                assert_unsigned_answer(&response.choice[1], "answer");
            }
            anyhow::ensure!(response.raw == raw, "stream raw document changed");
            anyhow::ensure!(
                response.finish_reason() == Some(FinishReason::Stop),
                "stop finish reason changed"
            );
        }
    }
    Ok(())
}

#[test]
fn g237_all_five_late_signed_sdk_mimes_flush_at_eof_as_exact_inspection() -> TestResult {
    for mime in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ] {
        let script = whole(frame(vec![media(mime), signature("c2ln")], Some("STOP"))?)?;
        let raw = script
            .document
            .clone()
            .ok_or_else(|| anyhow::anyhow!("captured document"))?;
        let observed = read(script)?;
        assert_order(&observed, &[PartKind::Text]);
        let response = observed.outcome?;
        anyhow::ensure!(response.choice.len() == 1, "expected one choice");
        assert_inspection(&response.choice[0], mime, "c2ln", 0);
        anyhow::ensure!(response.raw == raw, "stream raw document changed");
        anyhow::ensure!(
            response.finish_reason() == Some(FinishReason::Stop),
            "stop finish reason changed"
        );
    }
    Ok(())
}

#[test]
fn g237_pending_media_preserves_real_part_order_between_surrounding_content() -> TestResult {
    let script = whole(frame(
        vec![
            json!({"text":"before"}),
            media("image/png"),
            signature("c2ln"),
            json!({"text":"answer"}),
            media("image/webp"),
            json!({"text":"after"}),
        ],
        Some("STOP"),
    )?)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured document"))?;
    let observed = read(script)?;
    assert_order(
        &observed,
        &[
            PartKind::Text,
            PartKind::Text,
            PartKind::Text,
            PartKind::Image,
            PartKind::Text,
        ],
    );
    let response = observed.outcome?;
    anyhow::ensure!(response.choice.len() == 5, "expected five choices");
    assert_unsigned_answer(&response.choice[0], "before");
    assert_inspection(&response.choice[1], "image/png", "c2ln", 1);
    assert_unsigned_answer(&response.choice[2], "answer");
    assert_image(&response.choice[3], "image/webp", ImageMediaType::WEBP);
    assert_unsigned_answer(&response.choice[4], "after");
    anyhow::ensure!(response.raw == raw, "stream raw document changed");
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_original_transport_error() -> TestResult {
    for signed in [false, true] {
        let mut frames = vec![Ok(frame(vec![media("image/png")], None)?)];
        if signed {
            frames.push(Ok(frame(vec![signature("c2ln")], None)?));
        }
        frames.push(Err(ProviderError::Response(
            "g218-late scripted transport failure".to_owned(),
        )));
        let observed = read(Scripted {
            frames,
            document: None,
        })?;
        assert_order(
            &observed,
            &[if signed {
                PartKind::Text
            } else {
                PartKind::Image
            }],
        );
        let parts = ends(&observed);
        anyhow::ensure!(parts.len() == 1, "expected one ended part");
        if signed {
            assert_inspection(parts[0], "image/png", "c2ln", 0);
        } else {
            assert_image(parts[0], "image/png", ImageMediaType::PNG);
        }
        anyhow::ensure!(
            matches!(observed.items.last(),Some(Err(ProviderError::Response(message)))
            if message=="g218-late scripted transport failure"),
            "transport error changed"
        );
        let mut expected = media("image/png");
        if signed {
            expected["thoughtSignature"] = json!("c2ln");
        }
        anyhow::ensure!(
            observed.partial.raw == body(vec![expected], None),
            "partial raw document changed"
        );
        anyhow::ensure!(
            observed.partial.choice.len() == 1,
            "expected one partial choice"
        );
        anyhow::ensure!(
            matches!(&observed.outcome,Err(ProviderError::Response(message))
            if message=="g218-late scripted transport failure"),
            "transport outcome error changed"
        );
    }
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_eof_truncation_without_inventing_stop() -> TestResult {
    for signed in [false, true] {
        let mut frames = vec![Ok(frame(vec![media("image/png")], None)?)];
        if signed {
            frames.push(Ok(frame(vec![signature("c2ln")], None)?));
        }
        let observed = read(Scripted {
            frames,
            document: None,
        })?;
        assert_order(
            &observed,
            &[if signed {
                PartKind::Text
            } else {
                PartKind::Image
            }],
        );
        let parts = ends(&observed);
        anyhow::ensure!(parts.len() == 1, "expected one ended part");
        if signed {
            assert_inspection(parts[0], "image/png", "c2ln", 0);
        } else {
            assert_image(parts[0], "image/png", ImageMediaType::PNG);
        }
        anyhow::ensure!(
            matches!(observed.items.last(), Some(Err(ProviderError::Truncated))),
            "expected final truncated error"
        );
        anyhow::ensure!(
            matches!(observed.outcome, Err(ProviderError::Truncated)),
            "expected truncated outcome"
        );
        anyhow::ensure!(
            observed.partial.finish_reason().is_none(),
            "partial finish reason changed"
        );
        let mut expected = media("image/png");
        if signed {
            expected["thoughtSignature"] = json!("c2ln");
        }
        anyhow::ensure!(
            observed.partial.raw == body(vec![expected], None),
            "partial raw document changed"
        );
    }
    Ok(())
}

#[test]
fn g237_pending_media_flushes_before_known_sdk_tool_protocol_error() -> TestResult {
    for signed in [false, true] {
        let mut frames = vec![Ok(frame(vec![media("image/png")], None)?)];
        if signed {
            frames.push(Ok(frame(vec![signature("c2ln")], None)?));
        }
        let failed: Frame = serde_json::from_value(json!({"candidates":[{
            "finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"
        }]}))?;
        frames.push(Ok(failed));
        let observed = read(Scripted {
            frames,
            document: None,
        })?;
        assert_order(
            &observed,
            &[if signed {
                PartKind::Text
            } else {
                PartKind::Image
            }],
        );
        let parts = ends(&observed);
        anyhow::ensure!(parts.len() == 1, "expected one ended part");
        if signed {
            assert_inspection(parts[0], "image/png", "c2ln", 0);
        } else {
            assert_image(parts[0], "image/png", ImageMediaType::PNG);
        }
        let expected = "Gemini stopped with finish_reason=MalformedFunctionCall: bad call";
        anyhow::ensure!(
            matches!(observed.items.last(),Some(Err(ProviderError::Response(message))) if message==expected),
            "response error changed"
        );
        anyhow::ensure!(
            matches!(&observed.outcome,Err(ProviderError::Response(message)) if message==expected),
            "response outcome error changed"
        );
        let mut part = media("image/png");
        if signed {
            part["thoughtSignature"] = json!("c2ln");
        }
        anyhow::ensure!(
            observed.partial.raw
                == json!({"candidates":[{
                    "content":{"role":"model","parts":[part]},
                    "finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"
                }]}),
            "partial raw document changed"
        );
    }
    Ok(())
}

#[test]
fn g237_next_refused_file_content_cannot_discard_preceding_pending_media() -> TestResult {
    let file = json!({"fileData":{"mimeType":"image/png","fileUri":"gs://image"}});
    let script = whole(frame(vec![media("image/png"), file.clone()], Some("STOP"))?)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured document"))?;
    let observed = read(script)?;
    assert_order(&observed, &[PartKind::Image]);
    assert_image(ends(&observed)[0], "image/png", ImageMediaType::PNG);
    let expected =
        "Gemini response part kind fileData carries no assistant content rig can account for";
    anyhow::ensure!(
        matches!(observed.items.last(),Some(Err(ProviderError::Response(message))) if message==expected),
        "response error changed"
    );
    anyhow::ensure!(
        matches!(observed.outcome,Err(ProviderError::Response(message)) if message==expected),
        "response error changed"
    );
    anyhow::ensure!(observed.partial.raw == raw, "partial raw document changed");
    anyhow::ensure!(
        raw["candidates"][0]["content"]["parts"][1] == file,
        "refused file payload changed"
    );
    Ok(())
}

#[test]
fn g237_unselected_sdk_inline_media_keeps_immediate_inspection_and_exact_fields() -> TestResult {
    let mut thought = media("image/png");
    thought["thought"] = json!(true);
    let mut signed = media("image/png");
    signed["thoughtSignature"] = json!("c2ln");
    let mut extra = media("image/png");
    extra["futurePart"] = json!({"n":1});
    let mut nested = media("image/png");
    nested["inlineData"]["displayName"] = json!("original.png");
    for part in [
        thought,
        signed,
        extra,
        nested,
        media("audio/wav"),
        media("image/gif"),
        media("image/svg+xml"),
    ] {
        let script = whole(frame(
            vec![part.clone(), json!({"text":"answer"})],
            Some("STOP"),
        )?)?;
        let raw = script
            .document
            .clone()
            .ok_or_else(|| anyhow::anyhow!("captured document"))?;
        anyhow::ensure!(
            raw["candidates"][0]["content"]["parts"][0] == part,
            "inspection payload changed"
        );
        let observed = read(script)?;
        assert_order(&observed, &[PartKind::Text, PartKind::Text]);
        let response = observed.outcome?;
        let AssistantContent::CarriedText(image) = &response.choice[0] else {
            anyhow::bail!("inspection path unchanged");
        };
        anyhow::ensure!(
            image.content == CarriedTextContent::InspectionOnlyMedia(String::new()),
            "inspection content changed"
        );
        anyhow::ensure!(image.payload.item == part, "inspection payload changed");
        anyhow::ensure!(
            image.payload.provider.as_deref() == Some(PROVIDER_NAME),
            "inspection issuer changed"
        );
        anyhow::ensure!(
            image.payload.position
                == CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index: 0
                },
            "inspection position changed"
        );
        assert_unsigned_answer(&response.choice[1], "answer");
        anyhow::ensure!(response.raw == raw, "stream raw document changed");
    }
    Ok(())
}

#[derive(Clone)]
struct MetadataOnly(Vec<Frame>);
impl Transport<crate::completion::GenerateContent> for MetadataOnly {
    fn send(
        &self,
        _request: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<Frame> {
        Opening::ready(Opened::new(
            futures::stream::iter(self.0.clone().into_iter().map(Ok))
                .chain(futures::stream::pending::<Result<Frame, ProviderError>>()),
        ))
    }
}

#[test]
fn g237_metadata_only_frames_do_not_publish_pending_media_before_content() -> TestResult {
    let received = MetadataOnly(vec![
        frame(vec![media("image/png")], None)?,
        frame(vec![signature("c2ln")], None)?,
        frame(vec![signature("dGFpbA==")], None)?,
    ]);
    let mut stream = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        received,
    )
    .stream(CompletionRequest::new("inspect"))?;
    anyhow::ensure!(
        stream.next().now_or_never().is_none(),
        "actual driver consumed metadata and awaits next content without an emitted item"
    );
    anyhow::ensure!(
        stream.partial().choice.is_empty(),
        "pending media was published"
    );
    // Existing cancellation discards undelivered items. No parity with the
    // old immediate-image delivery is asserted for this deliberate delay.
    stream.cancel();
    anyhow::ensure!(
        stream.partial().choice.is_empty(),
        "pending media was published"
    );
    anyhow::ensure!(
        matches!(
            futures::executor::block_on(stream.finish()),
            Err(ProviderError::Truncated)
        ),
        "expected cancelled stream truncation"
    );
    Ok(())
}

#[test]
fn gm1b_sdk_empty_texts_preserve_pending_signed_media_in_both_modes() -> TestResult {
    let parts = vec![
        media("image/png"),
        json!({"text":""}),
        json!({"text":""}),
        signature("c2ln"),
        json!({"text":"answer"}),
    ];
    let script = whole(frame(parts.clone(), Some("STOP"))?)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured SDK document"))?;
    anyhow::ensure!(
        raw == body(parts.clone(), Some("STOP")),
        "captured SDK document changed"
    );
    anyhow::ensure!(
        raw["candidates"][0]["content"]["parts"] == json!(parts),
        "captured SDK parts changed"
    );
    let unary =
        futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    anyhow::ensure!(
        observed.items.iter().all(Result::is_ok),
        "stream emitted an error"
    );
    assert_order(&observed, &[PartKind::Text, PartKind::Text]);
    anyhow::ensure!(ends(&observed).len() == 2, "expected two ended parts");
    assert_inspection(ends(&observed)[0], "image/png", "c2ln", 0);
    assert_unsigned_answer(ends(&observed)[1], "answer");
    let response = observed.outcome?;
    anyhow::ensure!(
        response.choice == unary.choice,
        "stream and unary choices differ"
    );
    for response in [&unary, &response] {
        anyhow::ensure!(response.choice.len() == 2, "expected two choices");
        assert_inspection(&response.choice[0], "image/png", "c2ln", 0);
        assert_unsigned_answer(&response.choice[1], "answer");
        anyhow::ensure!(response.raw == raw, "stream raw document changed");
        anyhow::ensure!(
            response.finish_reason() == Some(FinishReason::Stop),
            "stop finish reason changed"
        );
        anyhow::ensure!(response.text() == "answer", "answer text changed");
        anyhow::ensure!(
            !response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::Image(_))),
            "signed media became an image"
        );
    }
    Ok(())
}

/// Synthetic SDK frames exercise adapter boundaries, not network streaming.
#[test]
fn gm1b_synthetic_sdk_frames_keep_empty_texts_before_late_signature() -> TestResult {
    for grouped in [true, false] {
        let empty_parts = vec![json!({"text":""}), json!({"text":""})];
        let mut frames = vec![Ok(frame(vec![media("image/png")], None)?)];
        if grouped {
            let received = frame(empty_parts.clone(), None)?;
            anyhow::ensure!(
                Value::Object(rest_chunk(&received)?) == body(empty_parts, None),
                "grouped empty SDK parts changed"
            );
            frames.push(Ok(received));
        } else {
            for part in empty_parts {
                let received = frame(vec![part.clone()], None)?;
                anyhow::ensure!(
                    Value::Object(rest_chunk(&received)?) == body(vec![part], None),
                    "split empty SDK part changed"
                );
                frames.push(Ok(received));
            }
        }
        frames.push(Ok(frame(vec![signature("c2ln")], None)?));
        frames.push(Ok(frame(vec![json!({"text":"answer"})], Some("STOP"))?));
        let script = Scripted {
            frames,
            document: None,
        };
        let unary = futures::executor::block_on(
            model(script.clone()).call(CompletionRequest::new("inspect")),
        )?;
        let observed = read(script)?;
        anyhow::ensure!(
            observed.items.iter().all(Result::is_ok),
            "stream emitted an error"
        );
        assert_order(&observed, &[PartKind::Text, PartKind::Text]);
        let response = observed.outcome?;
        anyhow::ensure!(
            response.choice == unary.choice,
            "stream and unary choices differ"
        );
        let mut signed = media("image/png");
        signed["thoughtSignature"] = json!("c2ln");
        let expected = body(vec![signed, json!({"text":"answer"})], Some("STOP"));
        for response in [&unary, &response] {
            anyhow::ensure!(response.choice.len() == 2, "expected two choices");
            assert_inspection(&response.choice[0], "image/png", "c2ln", 0);
            assert_unsigned_answer(&response.choice[1], "answer");
            anyhow::ensure!(
                response.raw == expected,
                "the existing split-frame reassembler drops empty text and joins bare signature metadata"
            );
            anyhow::ensure!(
                response.finish_reason() == Some(FinishReason::Stop),
                "stop finish reason changed"
            );
        }
    }
    Ok(())
}

#[test]
fn gm1b_unsigned_media_stays_typed_across_empty_texts_until_content_or_eof() -> TestResult {
    for with_answer in [true, false] {
        let mut parts = vec![media("image/png"), json!({"text":""}), json!({"text":""})];
        if with_answer {
            parts.push(json!({"text":"answer"}));
        }
        let script = whole(frame(parts.clone(), Some("STOP"))?)?;
        let raw = script
            .document
            .clone()
            .ok_or_else(|| anyhow::anyhow!("captured SDK document"))?;
        anyhow::ensure!(
            raw == body(parts, Some("STOP")),
            "captured SDK document changed"
        );
        let unary = futures::executor::block_on(
            model(script.clone()).call(CompletionRequest::new("inspect")),
        )?;
        let observed = read(script)?;
        anyhow::ensure!(
            observed.items.iter().all(Result::is_ok),
            "stream emitted an error"
        );
        let kinds = if with_answer {
            vec![PartKind::Image, PartKind::Text]
        } else {
            vec![PartKind::Image]
        };
        assert_order(&observed, &kinds);
        let response = observed.outcome?;
        anyhow::ensure!(
            response.choice == unary.choice,
            "stream and unary choices differ"
        );
        for response in [&unary, &response] {
            anyhow::ensure!(
                response.choice.len() == if with_answer { 2 } else { 1 },
                "choice count changed"
            );
            assert_image(&response.choice[0], "image/png", ImageMediaType::PNG);
            if with_answer {
                assert_unsigned_answer(&response.choice[1], "answer");
            }
            anyhow::ensure!(response.raw == raw, "stream raw document changed");
            anyhow::ensure!(
                response.finish_reason() == Some(FinishReason::Stop),
                "stop finish reason changed"
            );
        }
    }
    // Synthetic pending SDK input proves empty parts do not publish the image.
    let received = MetadataOnly(vec![frame(
        vec![media("image/png"), json!({"text":""}), json!({"text":""})],
        None,
    )?]);
    let mut stream = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        received,
    )
    .stream(CompletionRequest::new("inspect"))?;
    anyhow::ensure!(
        stream.next().now_or_never().is_none(),
        "empty unsigned parts are not a content boundary"
    );
    anyhow::ensure!(
        stream.partial().choice.is_empty(),
        "pending media was published"
    );
    stream.cancel();
    anyhow::ensure!(
        stream.partial().choice.is_empty(),
        "pending media was published"
    );
    anyhow::ensure!(
        matches!(
            futures::executor::block_on(stream.finish()),
            Err(ProviderError::Truncated)
        ),
        "expected cancelled stream truncation"
    );
    Ok(())
}

#[test]
fn gm1b_signed_empty_text_remains_a_material_boundary_with_original_position() -> TestResult {
    let signed_empty = json!({"text":"", "thoughtSignature":"c2ln"});
    let parts = vec![
        media("image/png"),
        json!({"text":""}),
        json!({"text":""}),
        signed_empty.clone(),
        json!({"text":"answer"}),
    ];
    let script = whole(frame(parts.clone(), Some("STOP"))?)?;
    let raw = script
        .document
        .clone()
        .ok_or_else(|| anyhow::anyhow!("captured SDK document"))?;
    anyhow::ensure!(
        raw == body(parts, Some("STOP")),
        "captured SDK document changed"
    );
    let unary =
        futures::executor::block_on(model(script.clone()).call(CompletionRequest::new("inspect")))?;
    let observed = read(script)?;
    anyhow::ensure!(
        observed.items.iter().all(Result::is_ok),
        "stream emitted an error"
    );
    assert_order(
        &observed,
        &[PartKind::Image, PartKind::Text, PartKind::Text],
    );
    let response = observed.outcome?;
    anyhow::ensure!(
        response.choice == unary.choice,
        "stream and unary choices differ"
    );
    for response in [&unary, &response] {
        anyhow::ensure!(response.choice.len() == 3, "expected three choices");
        assert_image(&response.choice[0], "image/png", ImageMediaType::PNG);
        let AssistantContent::CarriedText(empty) = &response.choice[1] else {
            anyhow::bail!("signed empty text is a real separate answer part");
        };
        let CarriedTextContent::GeminiSignedAnswer(answer) = &empty.content else {
            anyhow::bail!("the signed empty part keeps its known signed answer category");
        };
        anyhow::ensure!(answer.text.is_empty(), "signed empty answer text changed");
        anyhow::ensure!(
            answer.signature.as_str() == "c2ln",
            "empty answer signature changed"
        );
        anyhow::ensure!(
            empty.payload.item == signed_empty,
            "signed empty payload changed"
        );
        anyhow::ensure!(
            empty.payload.provider.as_deref() == Some(PROVIDER_NAME),
            "signed empty issuer changed"
        );
        anyhow::ensure!(
            empty.payload.position
                == CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index: 3
                },
            "signed empty position changed"
        );
        anyhow::ensure!(
            response.choice[1].native_item().is_none(),
            "signed empty answer gained a native item"
        );
        assert_unsigned_answer(&response.choice[2], "answer");
        anyhow::ensure!(response.raw == raw, "stream raw document changed");
        anyhow::ensure!(
            response.finish_reason() == Some(FinishReason::Stop),
            "stop finish reason changed"
        );
        anyhow::ensure!(response.text() == "answer", "answer text changed");
    }
    Ok(())
}
