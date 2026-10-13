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

    let [AssistantContent::Text(before), AssistantContent::CarriedText(thought_image),
        AssistantContent::CarriedText(audio), AssistantContent::CarriedText(signed_image),
        AssistantContent::Text(after)] = response.choice.as_slice()
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
        assert_eq!(carried.payload.position,
            CarriedPosition::MessageContent { item_index: 0, part_index: index as u64 });
        assert_eq!(carried.payload.item,
            serde_json::to_value(&parts[index]).expect("the received part serializes"));
        assert_eq!(response.choice[index].native_item(), None);
        assert_eq!(carried.payload.item["inlineData"]["mimeType"], json!(mime));
        let data = carried.payload.item["inlineData"]["data"]
            .as_str().expect("the received inline bytes remain base64");
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(data)
            .expect("the SDK bytes retain valid base64"), bytes);
    }
    assert_eq!(thought_image.payload.item["thought"], json!(true));
    let signature = signed_image.payload.item["thoughtSignature"]
        .as_str().expect("the received signature is retained");
    assert_eq!(signature, "AQID");
    assert_eq!(base64::engine::general_purpose::STANDARD.decode(signature)
        .expect("the SDK signature retains valid base64"), vec![1, 2, 3]);
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
            response.message().expect("the completed reply supplies its own history message"),
            Message::user("next"),
        ]),
        &wire.describe(),
    ).expect("the completed history prepares");
    let encoded = wire.encode(request, Mode::Unary).expect("the prepared history encodes");
    assert_eq!(encoded.contents.len(), 3);
    assert_eq!(encoded.contents[0].role, "user");
    assert_eq!(encoded.contents[2].role, "user");
    let model = &encoded.contents[1];
    assert_eq!(model.role, "model");
    // These are empty known-text projections, not replayed media payloads.
    let empty = || vertexai::model::Part::new().set_text("").set_thought(false);
    assert_eq!(model.parts, [parts[0].clone(), empty(), empty(), empty(), parts[4].clone()]);
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
