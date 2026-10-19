use super::*;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};

/// Answers every request with one scripted SDK reply.
#[derive(Clone)]
pub(crate) struct Reply(vertexai::model::GenerateContentResponse);

impl Transport<crate::completion::GenerateContent> for Reply {
    fn send(
        &self,
        _payload: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        Opening::ready(Opened::new(futures::stream::iter([Ok(self.0.clone())])))
    }
}

/// The content a turn holding `choice` replays as, encoded for the model
/// the replies come from.
pub(crate) fn replay(
    choice: Vec<rig_core::message::AssistantContent>,
) -> Result<vertexai::model::Content, rig_core::error::EncodeError> {
    use rig_core::wire::{Mode, Wire};
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![rig_core::message::Message::from(choice)];
    let request = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH)
        .encode(request, Mode::Unary)?;
    request
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| rig_core::error::EncodeError::request("no content"))
}

/// The reply as the unary endpoint answers it.
pub(crate) trait Complete {
    fn complete(self) -> Result<CompletionResponse, ProviderError>;
}

impl Complete for vertexai::model::GenerateContentResponse {
    fn complete(self) -> Result<CompletionResponse, ProviderError> {
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            Reply(self),
        );
        futures::executor::block_on(model.call(CompletionRequest::new("hello")))
    }
}

pub(crate) fn complete(
    response: vertexai::model::GenerateContentResponse,
) -> Result<CompletionResponse, ProviderError> {
    response.complete()
}
use rig_core::message::{AssistantContent, Text};
use serde_json::json;

/// `candidate` as a finished one: Vertex states why every unary reply ended.
fn finished(candidate: vertexai::model::Candidate) -> vertexai::model::Candidate {
    candidate.set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
}

fn create_text_response(text: &str) -> vertexai::model::GenerateContentResponse {
    let part = vertexai::model::Part::new().set_text(text.to_string());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new()
        .set_content(content)
        .set_finish_reason(vertexai::model::candidate::FinishReason::Stop);
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
}

fn create_parts_response(
    parts: impl IntoIterator<Item = vertexai::model::Part>,
) -> vertexai::model::GenerateContentResponse {
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = finished(vertexai::model::Candidate::new().set_content(content));
    vertexai::model::GenerateContentResponse::new().set_candidates([candidate])
}

fn inline_data_part(mime_type: &str, data: Vec<u8>) -> vertexai::model::Part {
    vertexai::model::Part::new().set_inline_data(
        vertexai::model::Blob::new()
            .set_mime_type(mime_type)
            .set_data(data),
    )
}

#[test]
fn test_text_response_conversion() {
    let vertex_output = create_text_response("Hello, world!");
    let completion_response: Result<CompletionResponse, _> = vertex_output.complete();

    assert!(completion_response.is_ok());
    let response = completion_response.unwrap();
    assert_eq!(
        response
            .choice
            .iter()
            .map(AssistantContent::canonical)
            .collect::<Vec<_>>(),
        vec![AssistantContent::Text(Text::new(
            "Hello, world!".to_string()
        ))]
    );
}

/// G94: thought images, audio and signed images retain exact inspection
/// payloads between the surrounding text. Their carried category gives those
/// payloads no outbound replay authority.
#[test]
fn unselected_inline_parts_keep_exact_inspection_evidence() {
    use base64::Engine as _;
    use rig_core::completion::FinishReason;
    use rig_core::message::{CarriedPosition, CarriedTextContent, StopReason};

    let parts = [
        vertexai::model::Part::new().set_text("before"),
        inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
        inline_data_part("audio/wav", vec![0]),
        inline_data_part("image/gif", vec![0]).set_thought_signature(vec![1, 2, 3]),
        vertexai::model::Part::new().set_text("after"),
    ];
    let received = create_parts_response(parts.clone());
    let wire_raw = serde_json::Value::Object(
        rest_chunk(&received).expect("the received SDK reply has a REST wire frame"),
    );
    let response = complete(received).expect("the reply decodes");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.stop(), StopReason::Stop);
    assert_eq!(response.raw, wire_raw);
    assert_eq!(
        response.raw.pointer("/candidates/0/content/parts"),
        Some(&serde_json::to_value(&parts).expect("the received parts serialize")),
    );

    let [
        AssistantContent::Text(before),
        AssistantContent::CarriedText(thought_image),
        AssistantContent::CarriedText(audio),
        AssistantContent::CarriedText(signed_image),
        AssistantContent::Text(after),
    ] = response.choice.as_slice()
    else {
        panic!("the received five parts must stay text/inspection/inspection/inspection/text");
    };
    assert_eq!(before.text, "before");
    assert_eq!(after.text, "after");
    for (index, carried, mime, bytes) in [
        (1, thought_image, "image/png", vec![1, 2, 3]),
        (2, audio, "audio/wav", vec![0]),
        (3, signed_image, "image/gif", vec![0]),
    ] {
        assert!(matches!(&carried.content,
            CarriedTextContent::InspectionOnlyMedia(text) if text.is_empty()));
        assert_eq!(carried.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            carried.payload.position,
            CarriedPosition::MessageContent {
                item_index: 0,
                part_index: index as u64
            }
        );
        assert_eq!(
            carried.payload.item,
            serde_json::to_value(&parts[index]).expect("the received part serializes")
        );
        assert_eq!(response.choice[index].native_item(), None);
        assert_eq!(carried.payload.item["inlineData"]["mimeType"], json!(mime));
        let data = carried.payload.item["inlineData"]["data"]
            .as_str()
            .expect("the received inline bytes remain base64");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .expect("the SDK bytes retain valid base64"),
            bytes
        );
    }
    assert_eq!(thought_image.payload.item["thought"], json!(true));
    let signature = signed_image.payload.item["thoughtSignature"]
        .as_str()
        .expect("the received signature is retained");
    assert_eq!(signature, "AQID");
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(signature)
            .expect("the SDK signature retains valid base64"),
        vec![1, 2, 3]
    );
}

/// G94: completed history projects these three inspection carriers to empty
/// text; the retained media and signature never reach the Vertex request.
#[test]
fn prepared_history_omits_inspection_media_and_keeps_surrounding_text() {
    use rig_core::completion::FinishReason;
    use rig_core::message::{Message, StopReason};
    use rig_core::wire::{Mode, Operation as _, Wire as _};

    let parts = [
        vertexai::model::Part::new().set_text("before"),
        inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
        inline_data_part("audio/wav", vec![0]),
        inline_data_part("image/gif", vec![0]).set_thought_signature(vec![1, 2, 3]),
        vertexai::model::Part::new().set_text("after"),
    ];
    let response = complete(create_parts_response(parts.clone())).expect("the reply decodes");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.stop(), StopReason::Stop);
    let retained_raw = response.raw.clone();
    let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
    let request = rig_core::operation::Completion::prepare(
        CompletionRequest::from(vec![
            Message::user("hello"),
            response
                .message()
                .expect("the completed reply supplies its own history message"),
            Message::user("next"),
        ]),
        &wire.describe(),
    )
    .expect("the completed history prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the prepared history encodes");
    assert_eq!(encoded.contents.len(), 3);
    assert_eq!(encoded.contents[0].role, "user");
    assert_eq!(encoded.contents[2].role, "user");
    let model = &encoded.contents[1];
    assert_eq!(model.role, "model");
    // These are empty known-text projections, not replayed media payloads.
    let empty = || vertexai::model::Part::new().set_text("").set_thought(false);
    assert_eq!(
        model.parts,
        [
            parts[0].clone(),
            empty(),
            empty(),
            empty(),
            parts[4].clone()
        ]
    );
    assert_eq!(model.parts[0].text(), Some(&"before".to_owned()));
    assert_eq!(model.parts[4].text(), Some(&"after".to_owned()));
    for part in &model.parts[1..4] {
        assert_eq!(part.text(), Some(&String::new()));
        assert!(part.inline_data().is_none());
        assert!(part.file_data().is_none());
        assert!(part.thought_signature.is_empty());
    }
    assert_eq!(response.raw, retained_raw);
}

/// The load-bearing property behind `CompletionResponse::raw` for Vertex
/// AI: the captured value is `serde_json::to_value(&vertexai::model::GenerateContentResponse)`
/// — the SDK response as `raw_completion` returns it — and a consumer must
/// be able to read it back as the same type and get the same JSON.
/// Vertex has no cassette harness, so this is the unit-form pin: the
/// serde derives on the newtype delegate to the SDK's own (camelCase)
/// wire encoding (camelCase keys, enums as proto numbers), and fields
/// rig never normalizes (`modelVersion` is mapped, but a candidate's
/// `safetyRatings` and `avgLogprobs` are not)
/// survive both directions. Normalizing the restored value agrees with
/// normalizing the original.
#[test]
fn vertex_generate_content_output_round_trips_through_serde_json_value() {
    let part = vertexai::model::Part::new().set_text("hello".to_string());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new()
        .set_content(content)
        .set_finish_reason(vertexai::model::candidate::FinishReason::Stop)
        .set_avg_logprobs(-0.25)
        .set_safety_ratings([vertexai::model::SafetyRating::new()
            .set_category(vertexai::model::HarmCategory::Harassment)
            .set_probability(vertexai::model::safety_rating::HarmProbability::Negligible)]);
    let usage_metadata = vertexai::model::generate_content_response::UsageMetadata::new()
        .set_prompt_token_count(10)
        .set_candidates_token_count(20)
        .set_total_token_count(30);
    let response = vertexai::model::GenerateContentResponse::new()
        .set_candidates([candidate])
        .set_model_version("gemini-2.5-flash-001")
        .set_response_id("resp-vertex-1")
        .set_usage_metadata(usage_metadata);
    let raw = response;

    let value = serde_json::to_value(&raw).expect("serialize");
    assert_eq!(value["modelVersion"], "gemini-2.5-flash-001");
    assert_eq!(value["candidates"][0]["avgLogprobs"], -0.25);
    // The SDK encodes enums as their proto numbers, not their names —
    // `HARM_CATEGORY_HARASSMENT` is 3, `STOP` is 1 — so that is what the
    // capture carries; a consumer decodes it through the SDK's own enum.
    assert_eq!(value["candidates"][0]["safetyRatings"][0]["category"], 3);
    assert_eq!(value["candidates"][0]["finishReason"], 1);

    let back: vertexai::model::GenerateContentResponse =
        serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&back).expect("re-serialize"),
        value,
        "the capture must read back into GenerateContentResponse and re-serialize identically"
    );
    assert_eq!(back, raw);

    let original: CompletionResponse = raw.clone().complete().expect("original converts");
    // The candidate's metadata stays in the response's raw document.
    let native = original.raw.pointer("/candidates/0");
    assert_eq!(
        native.map(|native| &native["avgLogprobs"]),
        Some(&json!(-0.25))
    );
    // The turn reads the REST JSON, which spells enums by name.
    assert_eq!(
        native.map(|native| &native["safetyRatings"][0]["category"]),
        Some(&json!("HARM_CATEGORY_HARASSMENT"))
    );
    assert_eq!(
        original.raw.pointer("/candidates/0/finishReason"),
        Some(&json!("STOP"))
    );
    let restored: CompletionResponse = back.complete().expect("restored converts");
    assert_eq!(restored.identity(), original.identity());
    assert_eq!(restored.finish_reason(), original.finish_reason());
    assert_eq!(restored.model(), original.model());
    assert_eq!(restored.usage, original.usage);
    assert_eq!(restored.choice, original.choice);
    assert_eq!(restored.model(), Some("gemini-2.5-flash-001"));
    assert_eq!(
        restored.identity().response_id.as_deref(),
        Some("resp-vertex-1")
    );
}

/// A reply that arrives in several responses rebuilds the one response
/// Vertex AI would have sent whole, and a single response is its own REST
/// JSON. A unit test: no Vertex AI recording exists.
#[test]
fn several_responses_rebuild_the_whole_one() {
    use rig_core::wire::document::Reassemble;

    let mut first = create_text_response("Hel");
    for candidate in &mut first.candidates {
        candidate.finish_reason = vertexai::model::candidate::FinishReason::Unspecified;
    }
    let rest = create_text_response("lo").set_model_version("gemini-2.5-flash-001");
    let mut document = VertexDocument::default();
    document.absorb(&first);
    document.absorb(&rest);
    let whole = create_text_response("Hello").set_model_version("gemini-2.5-flash-001");
    assert_eq!(
        document.finish(),
        Value::Object(rest_chunk(&whole).expect("the response transcodes"))
    );

    let mut single = VertexDocument::default();
    single.absorb(&whole);
    assert_eq!(
        single.finish(),
        complete(whole).expect("the reply decodes").raw
    );
}

/// A hand-built reply, as no Vertex AI recording carries grounding: each
/// grounding segment cites the bytes of its part with its web sources and
/// scores, and each `citations` entry, whose offsets Vertex AI states no
/// unit for, cites the text block whole. The citations never reach the
/// replayed part.
#[test]
fn grounding_and_citations_cite_the_answer() {
    const ANSWER: &str = "Spain won 2\u{2013}1. Oyarzabal scored late.";
    let quoted = "Oyarzabal scored late.";
    let at = ANSWER.find(quoted).expect("the answer holds it");
    let response: vertexai::model::GenerateContentResponse = serde_json::from_value(json!({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": ANSWER}]},
            "finishReason": 1,
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": "https://example.com/a", "title": "example.com", "domain": "example.com"}}],
                "groundingSupports": [{
                    "segment": {"startIndex": at, "endIndex": at + quoted.len(), "text": quoted},
                    "groundingChunkIndices": [0],
                    "confidenceScores": [0.5],
                }],
            },
            "citationMetadata": {"citations": [{"startIndex": 1, "endIndex": 4, "uri": "https://example.com/c", "title": "C"}]},
        }],
    }))
    .expect("an SDK reply");
    let response = response.complete().expect("the reply decodes");
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    let cited: Vec<_> = text
        .citations()
        .iter()
        .map(|citation| {
            let [source] = &citation.sources[..] else {
                panic!("one source: {citation:?}");
            };
            let rig_core::message::SourceLocation::Url { url } = &source.location else {
                panic!("a web source: {source:?}");
            };
            (
                citation.span.as_ref().and_then(|_| text.cited(citation)),
                url.as_str(),
                source.title.as_deref(),
                source.confidence,
            )
        })
        .collect();
    assert_eq!(
        cited,
        [
            (
                Some(quoted),
                "https://example.com/a",
                Some("example.com"),
                Some(0.5)
            ),
            (None, "https://example.com/c", Some("C"), None),
        ]
    );
    let replayed = replay(response.choice.clone()).expect("the turn replays");
    assert_eq!(
        replayed.parts,
        [vertexai::model::Part::new().set_text(ANSWER.to_owned())]
    );
}

mod signature_media_controls;

mod late_media_controls;

/// The original call signature is observed on End Native; the failed partial strips it.
#[test]
fn gm9_current_signed_call_end_keeps_original_binary_signature() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let raw = b"\x00\x01\x02thinking-sig\xff";
    let function_call = vertexai::model::FunctionCall::new()
        .set_name("add".to_string())
        .set_args(serde_json::Map::new());
    let part = vertexai::model::Part::new()
        .set_function_call(function_call)
        .set_thought_signature(raw.to_vec());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let starts: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: rig_core::streaming::PartKind::ToolCall,
                    name,
                    ..
                }) => Some((*part, name.as_ref().map(|name| name.as_str()))),
                _ => None,
            })
            .collect();
        let arguments: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Arguments { part, json }) => Some((*part, json)),
                _ => None,
            })
            .collect();
        let calls: Vec<_> = ended
            .iter()
            .filter_map(|(part, content)| match content {
                AssistantContent::ToolCall(call) => Some((*part, call)),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(
            starts.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
            vec![Some("add")]
        );
        assert_eq!(arguments.len(), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls
                .iter()
                .map(|(_, call)| &call.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1
        );
        let expected_arguments = [json!({})];
        for (ordinal, ((start, _), ((argument_part, json), (end, call)))) in
            starts.iter().zip(arguments.iter().zip(&calls)).enumerate()
        {
            assert_eq!(start, argument_part);
            assert_eq!(start, end);
            assert_eq!(*json, &expected_arguments[ordinal].to_string());
            assert_eq!(
                Value::Object(call.function.arguments.clone()),
                expected_arguments[ordinal]
            );
            assert_eq!(Some(call.function.name.as_str()), starts[ordinal].1);
            assert!(matches!(&call.id, rig_core::message::CallId::Local(_)));
            assert_eq!(call.id.provider(), None);
            let AssistantContent::ToolCall(partial_call) = &partial.choice[end.index()] else {
                panic!(
                    "the matching partial Part is a tool call: {:?}",
                    partial.choice
                );
            };
            assert_eq!(call.id, partial_call.id);
            assert_eq!(call.function, partial_call.function);
            assert!(partial_call.native.is_none());
        }
        assert_eq!(partial.choice.len(), 1);
        let (part, content) = ended.first().expect("the signed call ends before EOF");
        assert_eq!(part.index(), 0);
        assert_eq!(
            content.native_item(),
            Some(&expected_raw["candidates"][0]["content"]["parts"][0])
        );
        let native = content
            .native_item()
            .expect("the actual End retains the call's Native");
        let encoded = base64::Engine::encode(&base64::prelude::BASE64_STANDARD, raw);
        assert_eq!(native["thoughtSignature"], json!(encoded));
        assert_eq!(
            expected_raw["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
            json!(encoded)
        );
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                native["thoughtSignature"]
                    .as_str()
                    .expect("the signature is a string")
            )
            .expect("the binary signature is base64"),
            raw.to_vec(),
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// The original signed thought is observed as carried reasoning before EOF failure.
#[test]
fn gm9_current_signed_thought_partial_keeps_original_binary_signature() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let raw = b"\x00\x01\x02thinking-text-sig\xff";
    let part = vertexai::model::Part::new()
        .set_text("thinking text".to_string())
        .set_thought(true)
        .set_thought_signature(raw.to_vec());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [AssistantContent::CarriedReasoning(reasoning)] = partial.choice.as_slice() else {
            panic!(
                "the original signed thought is carried reasoning: {:?}",
                partial.choice
            );
        };
        let encoded = base64::Engine::encode(&base64::prelude::BASE64_STANDARD, raw);
        assert_eq!(reasoning.display_text(), "thinking text");
        assert_eq!(
            reasoning.content,
            vec![rig_core::message::CarriedReasoningPart::Text {
                text: "thinking text".to_owned(),
                signature: Some(encoded.clone()),
            }]
        );
        assert_eq!(reasoning.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            reasoning.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0
            }
        );
        assert_eq!(
            reasoning.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][0]
        );
        assert_eq!(reasoning.payload.item["thoughtSignature"], json!(encoded));
        assert_eq!(reasoning.payload.item["thought"], json!(true));
        assert_eq!(
            base64::Engine::decode(&base64::prelude::BASE64_STANDARD, &encoded)
                .expect("the original thought signature is base64"),
            raw.to_vec(),
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_singleton_png_flushes_at_eof_before_truncation() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let raw = vec![0, 1, 2, 255];
    let parts = [inline_data_part("image/png", raw.clone())];
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
    let (prefix, ended, prefix_items) = rig_core::test_utils::history_conformance::cut(
        &wire,
        rig_core::wire::Mode::Streaming,
        [received.clone()],
    );
    assert!(!ended);
    assert!(
        prefix.choice.is_empty(),
        "the singleton image remains pending before actual EOF"
    );
    assert!(!prefix_items.iter().any(|item| matches!(
        item,
        Item::Event(StreamEvent::End {
            content: AssistantContent::Image(_),
            ..
        })
    )));
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [AssistantContent::Image(image)] = partial.choice.as_slice() else {
            panic!(
                "actual EOF flush must deliver the singleton PNG: {:?}",
                partial.choice
            );
        };
        assert_eq!(
            image.data,
            rig_core::message::DocumentSourceKind::Base64("AAEC/w==".to_owned())
        );
        assert_eq!(
            image.media_type,
            Some(rig_core::message::ImageMediaType::PNG)
        );
        // mec-rig-gm9-1 selects the current None; the original Some(Auto) is not restored.
        assert_eq!(image.detail, None);
        assert!(image.native.is_none());
        let (_, content) = ended
            .first()
            .expect("EOF emits the pending image End before Truncated");
        assert_eq!(
            content.native_item(),
            Some(&expected_raw["candidates"][0]["content"]["parts"][0])
        );
        assert_eq!(
            base64::Engine::decode(&base64::prelude::BASE64_STANDARD, "AAEC/w==")
                .expect("the actual image bytes are base64"),
            raw
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_mixed_jpeg_partial_keeps_original_order() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let raw = vec![1, 2, 3];
    let parts = [
        vertexai::model::Part::new().set_text("before"),
        inline_data_part("image/jpeg", raw.clone()),
        vertexai::model::Part::new().set_text("after"),
    ];
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [
            AssistantContent::Text(before),
            AssistantContent::Image(image),
            AssistantContent::Text(after),
        ] = partial.choice.as_slice()
        else {
            panic!(
                "the original text/JPEG/text order remains exact: {:?}",
                partial.choice
            );
        };
        assert_eq!(before.text, "before");
        assert_eq!(after.text, "after");
        assert_eq!(
            image.data,
            rig_core::message::DocumentSourceKind::Base64("AQID".to_owned())
        );
        assert_eq!(
            image.media_type,
            Some(rig_core::message::ImageMediaType::JPEG)
        );
        assert_eq!(
            base64::Engine::decode(&base64::prelude::BASE64_STANDARD, "AQID")
                .expect("the original JPEG bytes are base64"),
            raw
        );
        assert_eq!(
            ended[1].1.native_item(),
            Some(&expected_raw["candidates"][0]["content"]["parts"][1])
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_thought_png_keeps_inspection_between_original_text() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let parts = [
        vertexai::model::Part::new().set_text("before"),
        inline_data_part("image/png", vec![1, 2, 3]).set_thought(true),
        vertexai::model::Part::new().set_text("after"),
    ];
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [
            AssistantContent::Text(before),
            AssistantContent::CarriedText(media),
            AssistantContent::Text(after),
        ] = partial.choice.as_slice()
        else {
            panic!(
                "original before/inspection/after order: {:?}",
                partial.choice
            );
        };
        assert_eq!(before.text, "before");
        assert_eq!(after.text, "after");
        assert_eq!(partial.text(), "beforeafter");
        assert_eq!(
            media.content,
            rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
        );
        assert_eq!(media.known_text_projection(), Some(""));
        assert!(media.signed_answer().is_none());
        assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            media.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 1
            }
        );
        assert_eq!(
            media.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][1]
        );
        assert_eq!(
            media.payload.item["inlineData"]["mimeType"],
            json!("image/png")
        );
        assert_eq!(media.payload.item["inlineData"]["data"], json!("AQID"));
        assert_eq!(partial.choice[1].native_item(), None);
        assert!(
            !partial
                .choice
                .iter()
                .any(|part| matches!(part, AssistantContent::Image(_)))
        );
        assert_eq!(media.payload.item["thought"], json!(true));
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                media.payload.item["inlineData"]["data"]
                    .as_str()
                    .expect("the received media is a string")
            )
            .expect("the received media is base64"),
            vec![1, 2, 3],
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_thought_png_only_is_inspection_before_truncation() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let parts = [inline_data_part("image/png", vec![1, 2, 3]).set_thought(true)];
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [AssistantContent::CarriedText(media)] = partial.choice.as_slice() else {
            panic!("one exact inspection carrier: {:?}", partial.choice);
        };
        assert_eq!(partial.text(), "");
        assert_eq!(
            media.content,
            rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
        );
        assert_eq!(media.known_text_projection(), Some(""));
        assert!(media.signed_answer().is_none());
        assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            media.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0
            }
        );
        assert_eq!(
            media.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][0]
        );
        assert_eq!(
            media.payload.item["inlineData"]["mimeType"],
            json!("image/png")
        );
        assert_eq!(media.payload.item["inlineData"]["data"], json!("AQID"));
        assert_eq!(partial.choice[0].native_item(), None);
        assert!(
            !partial
                .choice
                .iter()
                .any(|part| matches!(part, AssistantContent::Image(_)))
        );
        assert_eq!(media.payload.item["thought"], json!(true));
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                media.payload.item["inlineData"]["data"]
                    .as_str()
                    .expect("the received media is a string")
            )
            .expect("the received media is base64"),
            vec![1, 2, 3],
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_audio_and_non_image_inputs_are_inspection_before_truncation() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    for mime_type in ["audio/wav", "application/pdf", "application/octet-stream"] {
        let parts = [inline_data_part(mime_type, vec![0])];
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts(parts);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        let expected_raw =
            Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
        assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            Reply(received.clone()),
        );
        futures::executor::block_on(async {
            let mut stream = model
                .stream(CompletionRequest::new("hello"))
                .expect("the unchanged no-terminal SDK input opens");
            let mut items = Vec::new();
            let mut saw_truncated = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(item) => {
                        assert!(!saw_truncated, "no content follows the terminal failure");
                        items.push(item);
                    }
                    Err(error) => {
                        assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                        assert!(!saw_truncated, "the original input fails once");
                        saw_truncated = true;
                    }
                }
            }
            assert!(saw_truncated, "the original input has no provider end");
            let partial = stream.partial();
            assert_eq!(partial.finish_reason(), None);
            assert!(matches!(partial.stop(), StopReason::Error(_)));
            assert_eq!(partial.raw, expected_raw);
            assert!(
                items.iter().all(|item| matches!(item, Item::Event(_))),
                "{items:?}"
            );
            assert!(
                !items
                    .iter()
                    .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
            );
            let ended: Vec<_> = items
                .iter()
                .filter_map(|item| match item {
                    Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                ended
                    .iter()
                    .map(|(part, _)| part.index())
                    .collect::<Vec<_>>(),
                (0..partial.choice.len()).collect::<Vec<_>>(),
            );
            assert_eq!(
                ended
                    .iter()
                    .map(|(_, content)| content.canonical())
                    .collect::<Vec<_>>(),
                partial.choice,
            );
            let [AssistantContent::CarriedText(media)] = partial.choice.as_slice() else {
                panic!("one exact inspection carrier: {:?}", partial.choice);
            };
            assert_eq!(partial.text(), "");
            assert_eq!(
                media.content,
                rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
            );
            assert_eq!(media.known_text_projection(), Some(""));
            assert!(media.signed_answer().is_none());
            assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
            assert_eq!(
                media.payload.position,
                rig_core::message::CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index: 0
                }
            );
            assert_eq!(
                media.payload.item,
                expected_raw["candidates"][0]["content"]["parts"][0]
            );
            assert_eq!(
                media.payload.item["inlineData"]["mimeType"],
                json!(mime_type)
            );
            assert_eq!(media.payload.item["inlineData"]["data"], json!("AA=="));
            assert_eq!(partial.choice[0].native_item(), None);
            assert!(
                !partial
                    .choice
                    .iter()
                    .any(|part| matches!(part, AssistantContent::Image(_)))
            );
            assert_eq!(
                base64::Engine::decode(
                    &base64::prelude::BASE64_STANDARD,
                    media.payload.item["inlineData"]["data"]
                        .as_str()
                        .expect("the received media is a string")
                )
                .expect("the received media is base64"),
                vec![0],
            );
            assert!(matches!(
                stream.finish().await,
                Err(ProviderError::Truncated)
            ));
            assert!(matches!(
                model.call(CompletionRequest::new("hello")).await,
                Err(ProviderError::Truncated)
            ));
        });
    }
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_gif_and_svg_inputs_are_inspection_before_truncation() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    for mime_type in ["image/gif", "image/svg+xml"] {
        let parts = [inline_data_part(mime_type, vec![0])];
        let content = vertexai::model::Content::new()
            .set_role("model")
            .set_parts(parts);
        let candidate = vertexai::model::Candidate::new().set_content(content);
        let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
        let expected_raw =
            Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
        assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
        let model = Model::new(
            crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
            Reply(received.clone()),
        );
        futures::executor::block_on(async {
            let mut stream = model
                .stream(CompletionRequest::new("hello"))
                .expect("the unchanged no-terminal SDK input opens");
            let mut items = Vec::new();
            let mut saw_truncated = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(item) => {
                        assert!(!saw_truncated, "no content follows the terminal failure");
                        items.push(item);
                    }
                    Err(error) => {
                        assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                        assert!(!saw_truncated, "the original input fails once");
                        saw_truncated = true;
                    }
                }
            }
            assert!(saw_truncated, "the original input has no provider end");
            let partial = stream.partial();
            assert_eq!(partial.finish_reason(), None);
            assert!(matches!(partial.stop(), StopReason::Error(_)));
            assert_eq!(partial.raw, expected_raw);
            assert!(
                items.iter().all(|item| matches!(item, Item::Event(_))),
                "{items:?}"
            );
            assert!(
                !items
                    .iter()
                    .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
            );
            let ended: Vec<_> = items
                .iter()
                .filter_map(|item| match item {
                    Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                ended
                    .iter()
                    .map(|(part, _)| part.index())
                    .collect::<Vec<_>>(),
                (0..partial.choice.len()).collect::<Vec<_>>(),
            );
            assert_eq!(
                ended
                    .iter()
                    .map(|(_, content)| content.canonical())
                    .collect::<Vec<_>>(),
                partial.choice,
            );
            let [AssistantContent::CarriedText(media)] = partial.choice.as_slice() else {
                panic!("one exact inspection carrier: {:?}", partial.choice);
            };
            assert_eq!(partial.text(), "");
            assert_eq!(
                media.content,
                rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
            );
            assert_eq!(media.known_text_projection(), Some(""));
            assert!(media.signed_answer().is_none());
            assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
            assert_eq!(
                media.payload.position,
                rig_core::message::CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index: 0
                }
            );
            assert_eq!(
                media.payload.item,
                expected_raw["candidates"][0]["content"]["parts"][0]
            );
            assert_eq!(
                media.payload.item["inlineData"]["mimeType"],
                json!(mime_type)
            );
            assert_eq!(media.payload.item["inlineData"]["data"], json!("AA=="));
            assert_eq!(partial.choice[0].native_item(), None);
            assert!(
                !partial
                    .choice
                    .iter()
                    .any(|part| matches!(part, AssistantContent::Image(_)))
            );
            assert_eq!(
                base64::Engine::decode(
                    &base64::prelude::BASE64_STANDARD,
                    media.payload.item["inlineData"]["data"]
                        .as_str()
                        .expect("the received media is a string")
                )
                .expect("the received media is base64"),
                vec![0],
            );
            assert!(matches!(
                stream.finish().await,
                Err(ProviderError::Truncated)
            ));
            assert!(matches!(
                model.call(CompletionRequest::new("hello")).await,
                Err(ProviderError::Truncated)
            ));
        });
    }
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_signed_png_retains_inspection_signature_before_truncation() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let part = inline_data_part("image/png", vec![0]).set_thought_signature(vec![1, 2, 3]);
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [AssistantContent::CarriedText(media)] = partial.choice.as_slice() else {
            panic!("one exact inspection carrier: {:?}", partial.choice);
        };
        assert_eq!(partial.text(), "");
        assert_eq!(
            media.content,
            rig_core::message::CarriedTextContent::InspectionOnlyMedia(String::new())
        );
        assert_eq!(media.known_text_projection(), Some(""));
        assert!(media.signed_answer().is_none());
        assert_eq!(media.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            media.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0
            }
        );
        assert_eq!(
            media.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][0]
        );
        assert_eq!(
            media.payload.item["inlineData"]["mimeType"],
            json!("image/png")
        );
        assert_eq!(media.payload.item["inlineData"]["data"], json!("AA=="));
        assert_eq!(partial.choice[0].native_item(), None);
        assert!(
            !partial
                .choice
                .iter()
                .any(|part| matches!(part, AssistantContent::Image(_)))
        );
        assert_eq!(media.payload.item["thoughtSignature"], json!("AQID"));
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                media.payload.item["thoughtSignature"]
                    .as_str()
                    .expect("the received signature is a string")
            )
            .expect("the received signature is base64"),
            vec![1, 2, 3],
        );
        assert_eq!(
            base64::Engine::decode(
                &base64::prelude::BASE64_STANDARD,
                media.payload.item["inlineData"]["data"]
                    .as_str()
                    .expect("the received media is a string")
            )
            .expect("the received media is base64"),
            vec![0],
        );
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_missing_ids_are_distinct_local_calls_in_original_order() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let parts = (0..3).map(|i| {
        vertexai::model::Part::new().set_function_call(
            vertexai::model::FunctionCall::new()
                .set_name("same")
                .set_args(
                    serde_json::json!({"n": i})
                        .as_object()
                        .expect("the original argument literal is an object")
                        .clone(),
                ),
        )
    });
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts(parts);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [
            AssistantContent::ToolCall(first),
            AssistantContent::ToolCall(second),
            AssistantContent::ToolCall(third),
        ] = partial.choice.as_slice()
        else {
            panic!(
                "three original calls, without discarded or extra parts: {:?}",
                partial.choice
            );
        };
        assert_eq!(
            [
                first.function.name.as_str(),
                second.function.name.as_str(),
                third.function.name.as_str()
            ],
            ["same", "same", "same"]
        );
        let starts: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: rig_core::streaming::PartKind::ToolCall,
                    name,
                    ..
                }) => Some((*part, name.as_ref().map(|name| name.as_str()))),
                _ => None,
            })
            .collect();
        let arguments: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Arguments { part, json }) => Some((*part, json)),
                _ => None,
            })
            .collect();
        let calls: Vec<_> = ended
            .iter()
            .filter_map(|(part, content)| match content {
                AssistantContent::ToolCall(call) => Some((*part, call)),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            starts.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
            vec![Some("same"), Some("same"), Some("same")]
        );
        assert_eq!(arguments.len(), 3);
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls
                .iter()
                .map(|(_, call)| &call.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        let expected_arguments = [json!({"n": 0}), json!({"n": 1}), json!({"n": 2})];
        for (ordinal, ((start, _), ((argument_part, json), (end, call)))) in
            starts.iter().zip(arguments.iter().zip(&calls)).enumerate()
        {
            assert_eq!(start, argument_part);
            assert_eq!(start, end);
            assert_eq!(*json, &expected_arguments[ordinal].to_string());
            assert_eq!(
                Value::Object(call.function.arguments.clone()),
                expected_arguments[ordinal]
            );
            assert_eq!(Some(call.function.name.as_str()), starts[ordinal].1);
            assert!(matches!(&call.id, rig_core::message::CallId::Local(_)));
            assert_eq!(call.id.provider(), None);
            let AssistantContent::ToolCall(partial_call) = &partial.choice[end.index()] else {
                panic!(
                    "the matching partial Part is a tool call: {:?}",
                    partial.choice
                );
            };
            assert_eq!(call.id, partial_call.id);
            assert_eq!(call.function, partial_call.function);
            assert!(partial_call.native.is_none());
        }
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// Received leaves of the original no-terminal SDK input, observed before failed finish.
#[test]
fn gm9_current_interleaved_calls_correlate_by_part_and_keep_local_ids() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};

    let call = || {
        let function_call = vertexai::model::FunctionCall::new()
            .set_name("same".to_string())
            .set_args(serde_json::Map::new());
        vertexai::model::Part::new().set_function_call(function_call)
    };
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([
            call(),
            vertexai::model::Part::new().set_text("between".to_string()),
            call(),
            call(),
        ]);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [
            AssistantContent::ToolCall(first),
            AssistantContent::Text(between),
            AssistantContent::ToolCall(second),
            AssistantContent::ToolCall(third),
        ] = partial.choice.as_slice()
        else {
            panic!(
                "the original call/text/call/call sequence is retained: {:?}",
                partial.choice
            );
        };
        assert_eq!(between.text, "between");
        assert_eq!(
            [
                first.function.name.as_str(),
                second.function.name.as_str(),
                third.function.name.as_str()
            ],
            ["same", "same", "same"]
        );
        let starts: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Start {
                    part,
                    kind: rig_core::streaming::PartKind::ToolCall,
                    name,
                    ..
                }) => Some((*part, name.as_ref().map(|name| name.as_str()))),
                _ => None,
            })
            .collect();
        let arguments: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::Arguments { part, json }) => Some((*part, json)),
                _ => None,
            })
            .collect();
        let calls: Vec<_> = ended
            .iter()
            .filter_map(|(part, content)| match content {
                AssistantContent::ToolCall(call) => Some((*part, call)),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        assert_eq!(
            starts.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
            vec![Some("same"), Some("same"), Some("same")]
        );
        assert_eq!(arguments.len(), 3);
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls
                .iter()
                .map(|(_, call)| &call.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        let expected_arguments = [json!({}), json!({}), json!({})];
        for (ordinal, ((start, _), ((argument_part, json), (end, call)))) in
            starts.iter().zip(arguments.iter().zip(&calls)).enumerate()
        {
            assert_eq!(start, argument_part);
            assert_eq!(start, end);
            assert_eq!(*json, &expected_arguments[ordinal].to_string());
            assert_eq!(
                Value::Object(call.function.arguments.clone()),
                expected_arguments[ordinal]
            );
            assert_eq!(Some(call.function.name.as_str()), starts[ordinal].1);
            assert!(matches!(&call.id, rig_core::message::CallId::Local(_)));
            assert_eq!(call.id.provider(), None);
            let AssistantContent::ToolCall(partial_call) = &partial.choice[end.index()] else {
                panic!(
                    "the matching partial Part is a tool call: {:?}",
                    partial.choice
                );
            };
            assert_eq!(call.id, partial_call.id);
            assert_eq!(call.function, partial_call.function);
            assert!(partial_call.native.is_none());
        }
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

/// The original signed answer replays from its actual failed partial with that failure intact.
#[test]
fn gm9_current_failed_signed_answer_message_prepares_and_replays_exact_part() {
    use futures::StreamExt as _;
    use rig_core::message::StopReason;
    use rig_core::streaming::{Item, StreamEvent};
    use rig_core::wire::{Mode, Operation as _, Wire as _};

    let raw = b"\x00\x01answer-sig\xff";
    let part = vertexai::model::Part::new()
        .set_text("the answer".to_string())
        .set_thought_signature(raw.to_vec());
    let content = vertexai::model::Content::new()
        .set_role("model")
        .set_parts([part]);
    let candidate = vertexai::model::Candidate::new().set_content(content);
    let received = vertexai::model::GenerateContentResponse::new().set_candidates([candidate]);
    let expected_raw =
        Value::Object(rest_chunk(&received).expect("the original SDK frame transcodes"));
    assert!(expected_raw.pointer("/candidates/0/finishReason").is_none());
    let model = Model::new(
        crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH),
        Reply(received.clone()),
    );
    futures::executor::block_on(async {
        let mut stream = model
            .stream(CompletionRequest::new("hello"))
            .expect("the unchanged no-terminal SDK input opens");
        let mut items = Vec::new();
        let mut saw_truncated = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    assert!(!saw_truncated, "no content follows the terminal failure");
                    items.push(item);
                }
                Err(error) => {
                    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
                    assert!(!saw_truncated, "the original input fails once");
                    saw_truncated = true;
                }
            }
        }
        assert!(saw_truncated, "the original input has no provider end");
        let partial = stream.partial();
        assert_eq!(partial.finish_reason(), None);
        assert!(matches!(partial.stop(), StopReason::Error(_)));
        assert_eq!(partial.raw, expected_raw);
        assert!(
            items.iter().all(|item| matches!(item, Item::Event(_))),
            "{items:?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| matches!(item, Item::Event(StreamEvent::Discard { .. })))
        );
        let ended: Vec<_> = items
            .iter()
            .filter_map(|item| match item {
                Item::Event(StreamEvent::End { part, content, .. }) => Some((*part, content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended
                .iter()
                .map(|(part, _)| part.index())
                .collect::<Vec<_>>(),
            (0..partial.choice.len()).collect::<Vec<_>>(),
        );
        assert_eq!(
            ended
                .iter()
                .map(|(_, content)| content.canonical())
                .collect::<Vec<_>>(),
            partial.choice,
        );
        let [AssistantContent::CarriedText(answer)] = partial.choice.as_slice() else {
            panic!(
                "the original signed answer stays carried: {:?}",
                partial.choice
            );
        };
        let signed = answer
            .signed_answer()
            .expect("the actual partial retains its signed-answer category");
        assert_eq!(signed.text, "the answer");
        assert_eq!(signed.signature.as_str(), "AAFhbnN3ZXItc2ln/w==");
        assert_eq!(
            base64::Engine::decode(&base64::prelude::BASE64_STANDARD, signed.signature.as_str())
                .expect("the original answer signature is base64"),
            raw.to_vec(),
        );
        assert_eq!(answer.payload.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(
            answer.payload.position,
            rig_core::message::CarriedPosition::MessageContent {
                item_index: 0,
                part_index: 0
            }
        );
        assert_eq!(
            answer.payload.item,
            expected_raw["candidates"][0]["content"]["parts"][0]
        );
        assert_eq!(partial.text(), "the answer");
        let failed_stop = partial.stop();
        let message = partial
            .message()
            .expect("the actual failed partial has its own history message");
        let rig_core::message::Message::Assistant(turn) = &message else {
            panic!("the partial's own message is assistant history");
        };
        assert_eq!(turn.stop.as_ref(), Some(&failed_stop));
        assert_eq!(turn.content, partial.choice);
        let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
        let request = rig_core::operation::Completion::prepare(
            CompletionRequest::from(vec![
                rig_core::message::Message::user("hello"),
                message,
                rig_core::message::Message::user("next"),
            ]),
            &wire.describe(),
        )
        .expect("the actual failed-partial history prepares");
        let [_, rig_core::message::Message::Assistant(prepared_turn), _] =
            request.chat_history.as_slice()
        else {
            panic!(
                "prepare retains the failed carried turn: {:?}",
                request.chat_history
            );
        };
        assert_eq!(prepared_turn.stop.as_ref(), Some(&failed_stop));
        assert_eq!(prepared_turn.content, partial.choice);
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the actual prepared history encodes for Vertex");
        assert_eq!(encoded.contents.len(), 3);
        assert_eq!(encoded.contents[0].role, "user");
        assert_eq!(encoded.contents[2].role, "user");
        let model_content = &encoded.contents[1];
        assert_eq!(model_content.role, "model");
        let [replayed] = model_content.parts.as_slice() else {
            panic!(
                "exactly one original signed answer part: {:?}",
                model_content.parts
            );
        };
        assert_eq!(replayed.text().map(String::as_str), Some("the answer"));
        assert_eq!(replayed.thought_signature.to_vec(), raw.to_vec());
        assert!(!replayed.thought);
        assert_eq!(
            model_content.parts,
            [vertexai::model::Part::new()
                .set_text("the answer".to_string())
                .set_thought_signature(raw.to_vec())
                .set_thought(false)]
        );
        assert_eq!(partial.stop(), failed_stop);
        assert_eq!(partial.raw, expected_raw);
        assert!(matches!(
            stream.finish().await,
            Err(ProviderError::Truncated)
        ));
        assert!(matches!(
            model.call(CompletionRequest::new("hello")).await,
            Err(ProviderError::Truncated)
        ));
    });
}

mod inspection_only_stop_controls;
