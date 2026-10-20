//! The history conformance suite for the Gemini gRPC wire. Replies are
//! protobuf messages built from the REST JSON the REST suite uses, so both
//! wires are checked against the same documents; the request body is the
//! protobuf request's REST JSON.

#![allow(clippy::expect_used)]

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::wire::{Mode, Wire};
use rig_gemini_grpc::completion::GenerateContent;
use rig_gemini_grpc::proto::{self, GenerateContentResponse};
use rig_gemini_grpc::rest::{from_rest, to_rest};
use rig_history_conformance::{Ablation, CallShape, Ending, HistoryFixture, Rng, Shape, replies};
use serde_json::{Value, json};

struct GrpcHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply holding `parts`, ending on `finish` when it is set, as the
/// protobuf message its REST JSON describes.
fn chunk(parts: Value, finish: Option<Value>) -> Value {
    let mut candidate = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let (Some(finish), Some(candidate)) = (finish, candidate.as_object_mut()) {
        candidate.insert("finishReason".to_owned(), finish);
    }
    json!({
        "candidates": [candidate],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": MODEL,
        "responseId": "resp_1",
    })
}

fn frame(document: Value) -> GenerateContentResponse {
    from_rest(document).expect("the document is a GenerateContentResponse")
}

fn stop() -> Option<Value> {
    Some(json!("STOP"))
}

/// The parts of `shape`, whole, and split into the chunks a stream sends.
fn parts(shape: Shape) -> Option<(Value, Vec<Value>)> {
    Some(match shape {
        Shape::Rich => (
            json!([
                { "text": "plan the lookup", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ=" },
                { "text": "looking it up" },
                { "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } },
                { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                {
                    "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" },
                    "thoughtSignature": "c2lnLWNhbGw="
                },
            ]),
            vec![
                json!([{ "text": "plan the ", "thought": true }]),
                json!([{ "text": "lookup", "thought": true, "thoughtSignature": "c2lnLXRob3VnaHQ=" }]),
                json!([{ "text": "looking " }]),
                json!([{ "text": "it up" }]),
                json!([
                    { "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } },
                    { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                ]),
                json!([{
                    "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" },
                    "thoughtSignature": "c2lnLWNhbGw="
                }]),
            ],
        ),
        Shape::Interleaved => (
            json!([
                { "text": "first", "thought": true },
                { "text": "between" },
                { "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" },
                { "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" } },
            ]),
            vec![
                json!([{ "text": "first", "thought": true }]),
                json!([{ "text": "between" }]),
                json!([{ "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" }]),
                json!([{ "functionCall": { "name": "lookup", "args": { "q": "rig" }, "id": "call_1" } }]),
            ],
        ),
        // Protobuf decoding drops fields and part kinds its schema does not
        // declare, so an invented item or field never reaches the decoder.
        Shape::Unknown => return None,
    })
}

impl HistoryFixture for GrpcHistory {
    type Wire = GenerateContent;

    fn wire(&self, model: &str) -> GenerateContent {
        GenerateContent::new(model)
    }

    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        Some(serde_json::json!({"functionDeclarations": [{
            "name": tool.name.as_str(), "description": tool.description,
            "parametersJsonSchema": tool.parameters,
        }]}))
    }

    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        Some(replies::gemini_spec(rng, true))
    }

    fn reply_frames(
        &self,
        spec: &replies::Spec,
    ) -> Option<replies::Frames<GenerateContentResponse>> {
        let (whole, streamed) = replies::gemini_build(MODEL, spec, true);
        Some(replies::Frames {
            whole: whole.into_iter().map(frame).collect(),
            streamed: streamed.into_iter().map(frame).collect(),
        })
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "gemini-2.5-flash"
    }

    /// The protobuf request as its REST JSON; its `model` names the model.
    fn body(
        &self,
        wire: &GenerateContent,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        Ok(to_rest(&wire.encode(request, mode)?)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let (whole, streamed) = parts(shape)?;
        Some(match mode {
            Mode::Unary => vec![frame(chunk(whole, stop()))],
            Mode::Streaming => {
                let last = streamed.len().saturating_sub(1);
                streamed
                    .into_iter()
                    .enumerate()
                    .map(|(at, parts)| frame(chunk(parts, if at == last { stop() } else { None })))
                    .collect()
            }
        })
    }

    /// `args` is a `google.protobuf.Struct`: only an object, or no
    /// arguments at all, can arrive.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let call = match serde_json::from_str::<Value>(arguments).ok()? {
            Value::Null => json!({ "name": "lookup" }),
            args @ Value::Object(_) => json!({ "name": "lookup", "args": args }),
            _ => return None,
        };
        Some(vec![frame(chunk(
            json!([{ "functionCall": call }]),
            stop(),
        ))])
    }

    /// Every finish reason the proto lists, and a number it does not,
    /// which is how a reason newer than the proto arrives.
    fn finishes(&self) -> Vec<(&'static str, Vec<GenerateContentResponse>, Ending)> {
        let reply =
            |finish: Value| vec![frame(chunk(json!([{ "text": "partial" }]), Some(finish)))];
        let mut finishes = vec![
            ("STOP", reply(json!("STOP")), Ending::Success),
            ("MAX_TOKENS", reply(json!("MAX_TOKENS")), Ending::Success),
        ];
        for name in [
            "SAFETY",
            "RECITATION",
            "OTHER",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
        ] {
            assert!(
                proto::candidate::FinishReason::from_str_name(name).is_some(),
                "the proto lists {name}"
            );
            finishes.push((name, reply(json!(name)), Ending::Failure));
        }
        finishes.push(("an unlisted number", reply(json!(99)), Ending::Failure));
        finishes
    }

    fn ablation(&self) -> Option<Ablation<GenerateContentResponse>> {
        Some(Ablation {
            document: chunk(parts(Shape::Rich)?.0, stop()),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/content/parts/4/functionCall/name",
                "/candidates/0/finishReason",
            ],
            frames: |mut document| {
                if let Some((whole, _)) = parts(Shape::Rich) {
                    fitted(&mut document, &chunk(whole, stop()));
                }
                vec![frame(document)]
            },
        })
    }

    /// A part is its own whole reply.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let part = block.native_item()?.clone();
        let response = rig_core::test_utils::history::decode(
            &self.wire(MODEL),
            Mode::Unary,
            [frame(chunk(json!([part]), stop()))],
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Gemini sends whole calls: one reply listing both, or one chunk each.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        let first = json!({ "functionCall": { "name": "weather", "args": { "city": "Paris" }, "id": "a1" } });
        let second = json!({ "functionCall": { "name": "weather", "args": { "city": "Rome" }, "id": "b2" } });
        match (shape, mode) {
            (CallShape::WholeList, _) => Some(vec![frame(chunk(json!([first, second]), stop()))]),
            (CallShape::Indexless, Mode::Streaming) => Some(vec![
                frame(chunk(json!([first]), None)),
                frame(chunk(json!([second]), stop())),
            ]),
            _ => None,
        }
    }

    fn empty_reply(&self, _mode: Mode) -> Option<Vec<GenerateContentResponse>> {
        Some(vec![frame(chunk(json!([]), stop()))])
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/candidates/0/finishReason")
    }

    /// A part's index within the chunk that carried it
    /// (mec-rig-h03-carried-position-1).
    fn transport_local(&self) -> &'static [&'static str] {
        &["/payload/position/part_index"]
    }

    /// A signature on a trailing empty part is an empty signed block of its
    /// own (AG §2); the stream wraps the REST decoder.
    fn splits_trailing_signature(&self) -> bool {
        true
    }

    /// Each replayed provider part as REST JSON, including interleaved reasoning.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        let interleaved = matches!(
            turn.content.as_slice(),
            [
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_),
                AssistantContent::Text(_),
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_),
                AssistantContent::ToolCall(_),
            ]
        );
        let replayed: Vec<Value> = turn
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Opaque(opaque) if opaque.replay => Some(opaque.item.clone()),
                AssistantContent::CarriedReasoning(reasoning) if interleaved => {
                    Some(reasoning.payload.item.clone())
                }
                block => block.native_item().cloned(),
            })
            .collect();
        if interleaved {
            assert_eq!(
                replayed.len(),
                4,
                "the interleaved reply supplies all four outbound items: {turn:?} gives {replayed:?}"
            );
        }
        replayed
    }
}

/// `value` with each field and element whose JSON type differs from the
/// same one in `typed` left out. A reply frame is a typed message, so a
/// field of another type never arrives; it arrives unset.
fn fitted(value: &mut Value, typed: &Value) {
    let fits = |value: &mut Value, typed: Option<&Value>| match typed {
        Some(typed) if std::mem::discriminant(value) != std::mem::discriminant(typed) => false,
        Some(typed) => {
            fitted(value, typed);
            true
        }
        None => true,
    };
    match (value, typed) {
        (Value::Object(fields), Value::Object(types)) => {
            fields.retain(|key, field| fits(field, types.get(key)));
        }
        (Value::Array(values), Value::Array(types)) => {
            let mut types = types.iter();
            values.retain_mut(|value| fits(value, types.next()));
        }
        _ => {}
    }
}

#[test]
fn h05_grpc_missing_function_call_name_is_refused() {
    use rig_core::error::ProviderError;
    use rig_core::message::EmptyToolName;
    use rig_core::test_utils::history::decode;

    let fixture = GrpcHistory;
    let mut ablation = fixture
        .ablation()
        .expect("the gRPC ablation fixture exists");
    let call = ablation
        .document
        .pointer_mut("/candidates/0/content/parts/4/functionCall")
        .and_then(Value::as_object_mut)
        .expect("the original ablation fixture has the fifth part's function call");
    assert_eq!(call.shift_remove("name"), Some(json!("lookup")));
    let result = decode(
        &fixture.wire(fixture.model()),
        Mode::Unary,
        (ablation.frames)(ablation.document),
    );
    assert!(
        matches!(result, Err(ProviderError::Request(ref error))
            if error.downcast_ref::<EmptyToolName>() == Some(&EmptyToolName)
                && error.to_string() == EmptyToolName.to_string()),
        "the unchanged removed-name ablation reports the exact empty tool name error: {result:?}"
    );
}

/// Whole and streamed turns keep their provider-frame coordinates. Rich
/// PNG and Interleaved second-thought comparisons remain separate.
fn h03_grpc_stream_equals_whole(fixture: &GrpcHistory) {
    use rig_core::message::{CarriedPosition, CarriedTextContent};
    use rig_core::test_utils::history::{assert_restated_agrees, decode};

    let wire = fixture.wire(fixture.model());
    for shape in [Shape::Rich, Shape::Interleaved, Shape::Unknown] {
        let (Some(whole), Some(streamed)) = (
            fixture.reply(shape, Mode::Unary),
            fixture.reply(shape, Mode::Streaming),
        ) else {
            assert!(shape != Shape::Rich, "every wire supplies the rich reply");
            continue;
        };
        if shape == Shape::Interleaved {
            h03_grpc_interleaved_signed_reasoning(&wire, whole, streamed);
            continue;
        }
        if shape != Shape::Rich {
            assert_restated_agrees(&wire, whole, streamed);
            continue;
        }

        let png = json!({ "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } });
        let positions = |frames: &[GenerateContentResponse]| {
            let mut positions = Vec::new();
            for (frame_index, frame) in frames.iter().enumerate() {
                let document = to_rest(frame).expect("the actual fixture frame transcodes");
                let parts = document
                    .pointer("/candidates/0/content/parts")
                    .and_then(Value::as_array)
                    .expect("the actual fixture frame has parts");
                for (part_index, part) in parts.iter().enumerate() {
                    if part == &png {
                        positions.push((frame_index, part_index));
                    }
                }
            }
            positions
        };
        let whole_positions = positions(&whole);
        let streamed_positions = positions(&streamed);
        assert_eq!(
            whole_positions.len(),
            1,
            "one exact PNG in the whole fixture: {whole_positions:?}"
        );
        let (whole_frame_index, whole_part_index) = whole_positions
            .first()
            .expect("the unique whole PNG position exists");
        assert_eq!(
            streamed_positions.len(),
            1,
            "one exact PNG in the streamed fixture: {streamed_positions:?}"
        );
        let (stream_frame_index, stream_part_index) = streamed_positions
            .first()
            .expect("the unique streamed PNG position exists");
        assert_eq!(*whole_frame_index, 0);
        assert_eq!(*whole_part_index, 2);
        assert_eq!(*stream_part_index, 0);
        let whole_frame = to_rest(
            whole
                .get(*whole_frame_index)
                .expect("the located whole frame exists"),
        )
        .expect("the whole PNG frame transcodes");
        let stream_frame = to_rest(
            streamed
                .get(*stream_frame_index)
                .expect("the located streamed frame exists"),
        )
        .expect("the streamed PNG frame transcodes");
        let whole_part = whole_frame
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .and_then(|parts| parts.get(*whole_part_index))
            .expect("the whole PNG position exists");
        let stream_part = stream_frame
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .and_then(|parts| parts.get(*stream_part_index))
            .expect("the streamed PNG position exists");
        assert_eq!(whole_part, &png);
        assert_eq!(stream_part, &png);

        let unary = decode(&wire, Mode::Unary, whole).expect("the whole Rich reply decodes");
        let stream =
            decode(&wire, Mode::Streaming, streamed).expect("the streamed Rich reply decodes");
        assert!(
            matches!(unary.choice.get(2), Some(AssistantContent::CarriedText(_))),
            "the whole Rich PNG is carried at content index 2: {:?}",
            unary.choice
        );
        let whole_media = unary
            .choice
            .get(2)
            .and_then(|content| match content {
                AssistantContent::CarriedText(media) => Some(media),
                _ => None,
            })
            .expect("the checked whole carried media exists");
        assert!(
            matches!(stream.choice.get(2), Some(AssistantContent::CarriedText(_))),
            "the streamed Rich PNG is carried at content index 2: {:?}",
            stream.choice
        );
        let stream_media = stream
            .choice
            .get(2)
            .and_then(|content| match content {
                AssistantContent::CarriedText(media) => Some(media),
                _ => None,
            })
            .expect("the checked stream carried media exists");
        for (media, fixture_part, part_index) in [
            (whole_media, whole_part, *whole_part_index),
            (stream_media, stream_part, *stream_part_index),
        ] {
            assert_eq!(
                media.content,
                CarriedTextContent::InspectionOnlyMedia(String::new())
            );
            assert_eq!(media.payload.item, *fixture_part);
            assert_eq!(
                media.payload.provider.as_deref(),
                Some(rig_core::providers::gemini::PROVIDER_NAME)
            );
            assert_eq!(
                media.payload.position,
                CarriedPosition::MessageContent {
                    item_index: 0,
                    part_index: u64::try_from(part_index)
                        .expect("the fixture index fits a carried position"),
                }
            );
        }
        assert_eq!(whole_media.content, stream_media.content);
        assert_eq!(whole_media.payload.item, stream_media.payload.item);
        assert_eq!(whole_media.payload.provider, stream_media.payload.provider);

        const PART_INDEX: &str = "/content/2/payload/position/part_index";
        let whole_turn = serde_json::to_value(unary.message().expect("the whole Rich turn exists"))
            .expect("the whole Rich turn serializes");
        let mut streamed_turn =
            serde_json::to_value(stream.message().expect("the streamed Rich turn exists"))
                .expect("the streamed Rich turn serializes");
        assert_eq!(
            whole_turn.pointer(PART_INDEX),
            Some(&json!(whole_part_index))
        );
        assert_eq!(
            streamed_turn.pointer(PART_INDEX),
            Some(&json!(stream_part_index))
        );
        *streamed_turn
            .pointer_mut(PART_INDEX)
            .expect("the exact streamed coordinate exists") = whole_turn
            .pointer(PART_INDEX)
            .expect("the exact whole coordinate exists")
            .clone();
        assert_eq!(
            whole_turn, streamed_turn,
            "the Rich turns agree everywhere except the PNG's transport-local part_index"
        );
    }
    // Both empty reply modes report the existing exact no-message error.
    if let (Some(whole), Some(streamed)) = (
        fixture.empty_reply(Mode::Unary),
        fixture.empty_reply(Mode::Streaming),
    ) {
        let error = decode(&wire, Mode::Unary, whole)
            .expect_err("the unchanged unary empty fixture has no message or tool call");
        assert!(
            matches!(error, rig_core::error::ProviderError::Response(ref message)
                if message == rig_core::message::EMPTY_RESPONSE_ERROR),
            "the unary empty reply reports the exact no-message response error: {error:?}"
        );
        let error = decode(&wire, Mode::Streaming, streamed)
            .expect_err("the unchanged streamed empty fixture has no message or tool call");
        assert!(
            matches!(error, rig_core::error::ProviderError::Response(ref message)
                if message == rig_core::message::EMPTY_RESPONSE_ERROR),
            "the streamed empty reply reports the exact no-message response error: {error:?}"
        );
    }
}

/// The Interleaved second thought retains its position in its actual
/// signed provider part, independently of the unchanged first thought.
fn h03_grpc_interleaved_signed_reasoning(
    wire: &GenerateContent,
    whole: Vec<GenerateContentResponse>,
    streamed: Vec<GenerateContentResponse>,
) {
    use rig_core::message::{CarriedPosition, CarriedReasoningPart};
    use rig_core::test_utils::history::decode;

    let signed = json!({
        "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA=="
    });
    let positions = |frames: &[GenerateContentResponse]| {
        let mut positions = Vec::new();
        for (frame_index, frame) in frames.iter().enumerate() {
            let document = to_rest(frame).expect("the actual Interleaved frame transcodes");
            let parts = document
                .pointer("/candidates/0/content/parts")
                .and_then(Value::as_array)
                .expect("the actual Interleaved frame has parts");
            for (part_index, part) in parts.iter().enumerate() {
                if part == &signed {
                    positions.push((frame_index, part_index));
                }
            }
        }
        positions
    };
    let whole_positions = positions(&whole);
    let streamed_positions = positions(&streamed);
    assert_eq!(
        whole_positions.len(),
        1,
        "one exact signed second thought in the whole fixture: {whole_positions:?}"
    );
    let (whole_frame_index, whole_part_index) = whole_positions
        .first()
        .expect("the unique whole signed second thought position exists");
    assert_eq!(
        streamed_positions.len(),
        1,
        "one exact signed second thought in the streamed fixture: {streamed_positions:?}"
    );
    let (stream_frame_index, stream_part_index) = streamed_positions
        .first()
        .expect("the unique streamed signed second thought position exists");
    assert_eq!(*whole_frame_index, 0);
    assert_eq!(*whole_part_index, 2);
    assert_eq!(*stream_frame_index, 2);
    assert_eq!(*stream_part_index, 0);
    let whole_frame = to_rest(
        whole
            .get(*whole_frame_index)
            .expect("the located whole frame exists"),
    )
    .expect("the whole signed thought frame transcodes");
    let stream_frame = to_rest(
        streamed
            .get(*stream_frame_index)
            .expect("the located streamed frame exists"),
    )
    .expect("the streamed signed thought frame transcodes");
    let whole_part = whole_frame
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .and_then(|parts| parts.get(*whole_part_index))
        .expect("the whole signed thought position exists");
    let stream_part = stream_frame
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .and_then(|parts| parts.get(*stream_part_index))
        .expect("the streamed signed thought position exists");
    assert_eq!(whole_part, &signed);
    assert_eq!(stream_part, &signed);

    let unary = decode(wire, Mode::Unary, whole).expect("the whole Interleaved reply decodes");
    let stream =
        decode(wire, Mode::Streaming, streamed).expect("the streamed Interleaved reply decodes");
    assert!(
        matches!(
            unary.choice.get(2),
            Some(AssistantContent::CarriedReasoning(_))
        ),
        "the whole second thought is carried at content index 2: {:?}",
        unary.choice
    );
    let whole_thought = unary
        .choice
        .get(2)
        .and_then(|content| match content {
            AssistantContent::CarriedReasoning(thought) => Some(thought),
            _ => None,
        })
        .expect("the checked whole carried thought exists");
    assert!(
        matches!(
            stream.choice.get(2),
            Some(AssistantContent::CarriedReasoning(_))
        ),
        "the streamed second thought is carried at content index 2: {:?}",
        stream.choice
    );
    let stream_thought = stream
        .choice
        .get(2)
        .and_then(|content| match content {
            AssistantContent::CarriedReasoning(thought) => Some(thought),
            _ => None,
        })
        .expect("the checked stream carried thought exists");
    for (thought, fixture_part, part_index) in [
        (whole_thought, whole_part, *whole_part_index),
        (stream_thought, stream_part, *stream_part_index),
    ] {
        assert_eq!(
            thought.content,
            vec![CarriedReasoningPart::Text {
                text: "second".to_owned(),
                signature: Some("c2lnLXNlY29uZA==".to_owned()),
            }]
        );
        assert_eq!(thought.payload.item, *fixture_part);
        assert_eq!(
            thought.payload.provider.as_deref(),
            Some(rig_core::providers::gemini::PROVIDER_NAME)
        );
        assert_eq!(
            thought.payload.position,
            CarriedPosition::MessageContent {
                item_index: 0,
                part_index: u64::try_from(part_index)
                    .expect("the signed thought fixture index fits a carried position"),
            }
        );
    }
    assert_eq!(whole_thought.content, stream_thought.content);
    assert_eq!(whole_thought.payload.item, stream_thought.payload.item);
    assert_eq!(
        whole_thought.payload.provider,
        stream_thought.payload.provider
    );

    const PART_INDEX: &str = "/content/2/payload/position/part_index";
    let whole_turn =
        serde_json::to_value(unary.message().expect("the whole Interleaved turn exists"))
            .expect("the whole Interleaved turn serializes");
    let mut streamed_turn = serde_json::to_value(
        stream
            .message()
            .expect("the streamed Interleaved turn exists"),
    )
    .expect("the streamed Interleaved turn serializes");
    assert_eq!(
        whole_turn.pointer(PART_INDEX),
        Some(&json!(whole_part_index))
    );
    assert_eq!(
        streamed_turn.pointer(PART_INDEX),
        Some(&json!(stream_part_index))
    );
    let first_whole = whole_turn
        .pointer("/content/0")
        .expect("the first whole thought exists");
    let first_streamed = streamed_turn
        .pointer("/content/0")
        .expect("the first streamed thought exists");
    assert_eq!(
        first_whole, first_streamed,
        "the first thought stays strictly equal"
    );
    *streamed_turn
        .pointer_mut(PART_INDEX)
        .expect("the exact streamed signed thought coordinate exists") = whole_turn
        .pointer(PART_INDEX)
        .expect("the exact whole signed thought coordinate exists")
        .clone();
    assert_eq!(
        whole_turn, streamed_turn,
        "the Interleaved turns agree except the second thought's transport-local part_index"
    );
}

fn h07_grpc_failed_turns(fixture: &GrpcHistory) {
    use rig_core::error::ProviderError;
    use rig_core::message::{Message, StopReason};
    use rig_history_conformance::{decode, partial};

    let failed_cut_custody = |message: &Message| match message {
        Message::Assistant(turn) => {
            turn.stop.as_ref().is_some_and(StopReason::is_failure)
                && turn.content.iter().all(AssistantContent::is_carried)
        }
        Message::User { .. } | Message::System { .. } => true,
    };
    let wire = fixture.wire(fixture.model());
    let finishes = fixture.finishes();
    assert!(!finishes.is_empty(), "the wire documents its finishes");
    for (name, frames, ending) in finishes {
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            frames,
        );
        let protocol_error = match name {
            "MALFORMED_FUNCTION_CALL" => Some(
                "Gemini stopped with finish_reason=MalformedFunctionCall: no finish message provided",
            ),
            "UNEXPECTED_TOOL_CALL" => Some(
                "Gemini stopped with finish_reason=UnexpectedToolCall: no finish message provided",
            ),
            "TOO_MANY_TOOL_CALLS" => Some(
                "Gemini stopped with finish_reason=TooManyToolCalls: no finish message provided",
            ),
            _ => None,
        };
        if let Some(expected) = protocol_error {
            assert_eq!(ending, Ending::Failure);
            assert!(
                matches!(response, Err(ProviderError::Response(ref message)) if message == expected),
                "the unchanged `{name}` finish reports its exact response error: {response:?}"
            );
            continue;
        }
        let response = response.expect("the documented finish reply decodes");
        let failed = response.stop().is_failure();
        assert_eq!(
            failed,
            ending == Ending::Failure,
            "`{name}` ends as {:?}",
            response.stop()
        );
    }
    let Some(count) = fixture
        .reply(Shape::Rich, Mode::Streaming)
        .map(|frames| frames.len())
    else {
        return;
    };
    for cut in 1..count {
        let frames = fixture
            .reply(Shape::Rich, Mode::Streaming)
            .unwrap_or_default();
        let (response, ended) = partial(&wire, Mode::Streaming, frames.into_iter().take(cut));
        if ended {
            continue;
        }
        let Some(Message::Assistant(turn)) = response.message() else {
            continue;
        };
        assert!(
            turn.stop.as_ref().is_some_and(StopReason::is_failure),
            "a reply cut after {cut} frames is a failed turn: {:?}",
            turn.stop
        );
        let history = vec![
            Message::user("q"),
            Message::Assistant(turn),
            Message::user("next"),
        ];
        let describe = wire.describe();
        let replay = describe.replay.expect("a replay target");
        let adapted = rig_core::completion::history::adapt(&history, replay);
        assert!(
            adapted.iter().all(failed_cut_custody),
            "a failed cut retains only carried content in failed assistant turns: {adapted:?}"
        );
    }
    if let (Some(error), Some(mut frames)) = (
        fixture.error_frame(),
        fixture.reply(Shape::Rich, Mode::Streaming),
    ) {
        frames.truncate(frames.len().saturating_sub(1).max(1));
        frames.push(error);
        let (response, _) = partial(&wire, Mode::Streaming, frames);
        assert!(
            response.stop().is_failure(),
            "an in-band error fails the turn: {:?}",
            response.stop()
        );
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "gemini_grpc",
    fixture: GrpcHistory,
    h03_stream_equals_whole: h03_grpc_stream_equals_whole,
    h07_failed_turns: h07_grpc_failed_turns,
}
