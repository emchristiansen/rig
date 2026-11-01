//! The Gemini GenerateContent REST wire's history suite: unary
//! `generateContent` bodies and `streamGenerateContent` SSE chunks, each
//! part a block whose provider item is the part as Gemini sent it.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{
    AssistantContent, AssistantMessage, CarriedTextContent, DocumentSourceKind, UserContent,
};
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::gemini::completion::GenerateContent;
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablated, Ablation, CallShape, Ending, HistoryFixture, MediaPart, Refusal, RefusedArguments,
    Rng, Shape, decode, http_body, replies,
};
use serde_json::{Value, json};

/// Every `finishReason` Gemini and Vertex AI document that ends a turn as a
/// failure: content filters, recitation, unsupported language, image
/// failures, tool-protocol failures and the unused zero value.
pub const FAILURE_FINISHES: &[&str] = &[
    "FINISH_REASON_UNSPECIFIED",
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "OTHER",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_OTHER",
    "NO_IMAGE",
    "IMAGE_RECITATION",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "MISSING_THOUGHT_SIGNATURE",
    "MALFORMED_RESPONSE",
    "ESCALATION",
    "PUP_LIMITED_DISABLED",
    "MODEL_ARMOR",
];

/// Two calls to `weather`, `a1` and `b2`, as `functionCall` parts.
pub fn weather_calls() -> [Value; 2] {
    [
        json!({ "functionCall": { "name": "weather", "args": { "city": "Paris" }, "id": "a1" } }),
        json!({ "functionCall": { "name": "weather", "args": { "city": "Rome" }, "id": "b2" } }),
    ]
}

/// The error a Gemini tool-protocol finish fails the reply with: a
/// malformed or unexpected call, too many calls, a missing thought
/// signature or a malformed response.
/// What breaks inspection-only custody of the invented field in `turn`, the
/// decoded unknown-item reply, and `body`, its same-model replay, where the
/// known text replays as exactly `text` (row H4).
pub fn inspected_field(turn: &AssistantMessage, body: &Value, text: Value) -> Vec<String> {
    let mut problems = Vec::new();
    let kept = turn.content.iter().any(|block| {
        matches!(block, AssistantContent::CarriedText(carried)
            if carried.content == CarriedTextContent::InspectionOnlyText("noted".to_owned())
                && carried.payload.item.get("x_rig_field").is_some())
    });
    if !kept {
        problems.push(format!(
            "the turn keeps `noted` as inspection-only text whose part holds `x_rig_field`: {:?}",
            turn.content
        ));
    }
    let parts: Vec<&Value> = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|content| content["role"] == "model")
        .map(|content| &content["parts"])
        .collect();
    let expected = json!([text, {"x_rig_invented": {"id": "x_1"}}]);
    if parts != [&expected] {
        problems.push(format!(
            "the model turn replays exactly {expected}: {parts:?}"
        ));
    }
    problems
}

pub fn tool_protocol_error(finish: &str) -> Option<String> {
    let reason = match finish {
        "MALFORMED_FUNCTION_CALL" => "MalformedFunctionCall",
        "UNEXPECTED_TOOL_CALL" => "UnexpectedToolCall",
        "TOO_MANY_TOOL_CALLS" => "TooManyToolCalls",
        "MISSING_THOUGHT_SIGNATURE" => "MissingThoughtSignature",
        "MALFORMED_RESPONSE" => "MalformedResponse",
        _ => return None,
    };
    Some(format!(
        "Gemini stopped with finish_reason={reason}: no finish message provided"
    ))
}

/// Each field of the rich reply the typed reader refuses when it is
/// removed, `null` or retyped, in that order.
const TYPED_BOUNDARY: &[(&str, [bool; 3])] = &[
    ("/candidates/0/content/role", [false, false, true]),
    ("/candidates/0/content/parts/0/text", [false, true, true]),
    (
        "/candidates/0/content/parts/0/thought",
        [false, false, true],
    ),
    (
        "/candidates/0/content/parts/0/thoughtSignature",
        [false, false, true],
    ),
    ("/candidates/0/content/parts/1/text", [false, true, true]),
    (
        "/candidates/0/content/parts/2/inlineData",
        [false, true, true],
    ),
    (
        "/candidates/0/content/parts/2/inlineData/mimeType",
        [true, true, true],
    ),
    (
        "/candidates/0/content/parts/2/inlineData/data",
        [true, true, true],
    ),
    (
        "/candidates/0/content/parts/3/executableCode",
        [false, true, true],
    ),
    (
        "/candidates/0/content/parts/3/executableCode/language",
        [true, true, true],
    ),
    (
        "/candidates/0/content/parts/3/executableCode/code",
        [true, true, true],
    ),
    (
        "/candidates/0/content/parts/4/functionCall",
        [false, true, true],
    ),
    (
        "/candidates/0/content/parts/4/functionCall/name",
        [true, true, true],
    ),
    (
        "/candidates/0/content/parts/4/functionCall/args",
        [false, true, true],
    ),
    (
        "/candidates/0/content/parts/4/functionCall/id",
        [false, false, true],
    ),
    (
        "/candidates/0/content/parts/4/thoughtSignature",
        [false, false, true],
    ),
    ("/candidates/0/index", [false, false, true]),
    ("/usageMetadata", [false, false, true]),
    ("/usageMetadata/promptTokenCount", [false, true, true]),
    ("/usageMetadata/candidatesTokenCount", [false, false, true]),
    ("/usageMetadata/totalTokenCount", [false, true, true]),
    ("/modelVersion", [false, false, true]),
    ("/responseId", [false, true, true]),
];

pub struct GeminiRestHistory;

const MODEL: &str = "gemini-3-flash-preview";

/// A reply document holding `parts`, ending on `finish` when it is set.
fn chunk(parts: Value, finish: Option<&str>) -> Value {
    let mut candidate = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let (Some(finish), Some(candidate)) = (finish, candidate.as_object_mut()) {
        candidate.insert("finishReason".to_owned(), json!(finish));
    }
    json!({
        "candidates": [candidate],
        "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 5, "totalTokenCount": 12 },
        "modelVersion": MODEL,
        "responseId": "resp_1",
    })
}

fn frames(documents: impl IntoIterator<Item = Value>) -> Vec<WireFrame> {
    documents
        .into_iter()
        .map(|document| WireFrame::Text(document.to_string()))
        .collect()
}

/// The parts of `shape`, whole, and the same parts split into the chunks a
/// stream sends them in.
fn parts(shape: Shape) -> (Value, Vec<Value>) {
    match shape {
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
        Shape::Unknown => (
            json!([
                { "text": "noted", "x_rig_field": true },
                { "x_rig_invented": { "id": "x_1" } },
            ]),
            vec![
                json!([{ "text": "noted", "x_rig_field": true }]),
                json!([{ "x_rig_invented": { "id": "x_1" } }]),
            ],
        ),
    }
}

fn reply_of(shape: Shape, mode: Mode) -> Vec<WireFrame> {
    let (whole, streamed) = parts(shape);
    match mode {
        Mode::Unary => frames([chunk(whole, Some("STOP"))]),
        Mode::Streaming => {
            let last = streamed.len().saturating_sub(1);
            frames(
                streamed
                    .into_iter()
                    .enumerate()
                    .map(|(at, parts)| chunk(parts, (at == last).then_some("STOP"))),
            )
        }
    }
}

impl HistoryFixture for GeminiRestHistory {
    type Wire = GenerateContent;

    fn wire(&self, model: &str) -> GenerateContent {
        GenerateContent::new(GeminiConfig::new("test-key"), model)
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

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<WireFrame>> {
        let (whole, streamed) = replies::gemini_build(self.model(), spec, true);
        Some(replies::Frames {
            whole: replies::values(whole),
            streamed: replies::values(streamed),
        })
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "gemini-2.5-flash"
    }

    fn body(
        &self,
        wire: &GenerateContent,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply_of(shape, mode))
    }

    /// Gemini sends `args` as JSON, so only argument text that is JSON can
    /// arrive; truncated or non-JSON text cannot be expressed on this wire.
    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<WireFrame>> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        Some(frames([chunk(
            json!([{ "functionCall": { "name": "lookup", "args": args } }]),
            Some("STOP"),
        )]))
    }

    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        let reply = |finish: &str| frames([chunk(json!([{ "text": "partial" }]), Some(finish))]);
        [("STOP", Ending::Success), ("MAX_TOKENS", Ending::Success)]
            .into_iter()
            .chain(
                FAILURE_FINISHES
                    .iter()
                    .map(|finish| (*finish, Ending::Failure)),
            )
            .chain([("A_REASON_FROM_TOMORROW", Ending::Failure)])
            .map(|(finish, ending)| (finish, reply(finish), ending))
            .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: chunk(parts(Shape::Rich).0, Some("STOP")),
            required: &[
                "/candidates",
                "/candidates/0/content",
                "/candidates/0/content/parts",
                "/candidates/0/finishReason",
            ],
            frames: |document| frames([document]),
        })
    }

    /// A part is its own whole reply.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let part = block.native_item()?.clone();
        let wire = self.wire(MODEL);
        let response = rig_core::test_utils::history::decode(
            &wire,
            Mode::Unary,
            frames([chunk(json!([part]), Some("STOP"))]),
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Gemini sends whole calls: one reply listing both, or one chunk each.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let [first, second] = weather_calls();
        match (shape, mode) {
            (CallShape::WholeList, _) => {
                Some(frames([chunk(json!([first, second]), Some("STOP"))]))
            }
            (CallShape::Indexless, Mode::Streaming) => Some(frames([
                chunk(json!([first]), None),
                chunk(json!([second]), Some("STOP")),
            ])),
            _ => None,
        }
    }

    fn empty_reply(&self, _mode: Mode) -> Option<Vec<WireFrame>> {
        Some(frames([chunk(json!([]), Some("STOP"))]))
    }

    /// A whole reply with no part is refused; a stream that ended cleanly
    /// is a successful empty turn.
    fn refuses_empty_reply(&self, mode: Mode) -> bool {
        mode == Mode::Unary
    }

    fn refuses_empty_whole(&self) -> bool {
        true
    }

    /// The typed reader takes `args` only as an object: any other JSON
    /// fails the frame.
    fn refused_arguments(&self, arguments: &str, _mode: Mode) -> Option<RefusedArguments> {
        let found = match serde_json::from_str::<Value>(arguments).ok()? {
            Value::Object(_) => return None,
            Value::Null => "null".to_owned(),
            Value::String(text) => format!("string {}", json!(text)),
            Value::Array(_) => "sequence".to_owned(),
            other => panic!("no H6 case sends `{other}`"),
        };
        Some(RefusedArguments::CorruptFrame {
            event_type: None,
            error: format!("invalid type: {found}, expected a map"),
        })
    }

    fn ablation_refusal(
        &self,
        pointer: &str,
        change: Ablated,
        _document: &Value,
    ) -> Option<Refusal> {
        TYPED_BOUNDARY
            .iter()
            .any(|(field, refused)| {
                *field == pointer
                    && refused[match change {
                        Ablated::Removed => 0,
                        Ablated::Null => 1,
                        Ablated::Retyped => 2,
                    }]
            })
            .then_some(Refusal::UndecodedFrame)
    }

    /// Gemini's tool-protocol finishes fail the reply.
    fn refused_finish(&self, name: &str) -> Option<Refusal> {
        tool_protocol_error(name).map(Refusal::Response)
    }

    /// Unsigned text with an invented sibling field is inspection-only text
    /// (A48; AG §8): the turn keeps the whole part, and replay sends the
    /// known text alone beside the invented part.
    fn inspected_field(&self, turn: &AssistantMessage, body: &Value) -> Option<Vec<String>> {
        Some(inspected_field(
            turn,
            body,
            json!({"text": "noted", "thought": false}),
        ))
    }

    /// A part's index within the chunk that carried it
    /// (mec-rig-h03-carried-position-1).
    fn transport_local(&self) -> &'static [&'static str] {
        &["/payload/position/part_index"]
    }

    /// A signature on a trailing empty part is an empty signed block of its
    /// own (AG §2).
    fn splits_trailing_signature(&self) -> bool {
        true
    }

    /// A document of no declared type, and a tool-result image named by
    /// URL, are refused before adaptation.
    fn original_media_preflight(&self, _model: &str, part: MediaPart<'_>) -> Option<String> {
        let refused = match part {
            MediaPart::User(UserContent::Document(document)) if document.media_type.is_none() => {
                "A mime type is required for document inputs to Gemini"
            }
            MediaPart::ResultImage(image) if matches!(image.data, DocumentSourceKind::Url(_)) => {
                "Message conversion error: Gemini tool result images must use base64 inline \
                 data; URL-backed images are not supported"
            }
            _ => return None,
        };
        Some(refused.to_owned())
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/candidates/0/finishReason")
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

    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(
            json!({ "error": { "code": 503, "message": "overloaded", "status": "UNAVAILABLE" } })
                .to_string(),
        ))
    }
}

/// H3 on REST: whole and streamed turns agree except where a carried part
/// sits in its own frame. A carried part's position keeps its
/// transport-local `part_index`: the rich PNG (inspection-only media) and
/// the interleaved second thought are part 2 of the whole reply and part 0
/// of the stream chunk that holds each.
fn h03_gemini_rest_stream_equals_whole(fixture: &GeminiRestHistory) {
    let wire = fixture.wire(MODEL);
    let turn = |shape, mode| {
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            mode,
            reply_of(shape, mode),
        )
        .unwrap_or_else(|error| panic!("{shape:?} in {mode:?} decodes: {error}"));
        serde_json::to_value(response.message()).expect("a turn serializes")
    };
    let carried = [
        (
            Shape::Rich,
            json!({"type": "inspection_only_media", "text": ""}),
            json!({ "inlineData": { "mimeType": "image/png", "data": "aW1hZ2U=" } }),
        ),
        (
            Shape::Interleaved,
            json!([{"type": "text", "content": {"text": "second", "signature": "c2lnLXNlY29uZA=="}}]),
            json!({ "text": "second", "thought": true, "thoughtSignature": "c2lnLXNlY29uZA==" }),
        ),
    ];
    for (shape, content, item) in carried {
        const PART_INDEX: &str = "/content/2/payload/position/part_index";
        let whole = turn(shape, Mode::Unary);
        let mut streamed = turn(shape, Mode::Streaming);
        for (mode, turn, part_index) in [(Mode::Unary, &whole, 2), (Mode::Streaming, &streamed, 0)]
        {
            assert_eq!(
                (
                    turn.pointer("/content/2/content"),
                    turn.pointer("/content/2/payload/item"),
                    turn.pointer(PART_INDEX),
                ),
                (Some(&content), Some(&item), Some(&json!(part_index))),
                "{shape:?} in {mode:?}: block 2 is carried at its own part index: {turn}"
            );
        }
        if let Some(index) = streamed.pointer_mut(PART_INDEX) {
            *index = json!(2);
        }
        assert_eq!(
            whole, streamed,
            "{shape:?}: the turns agree everywhere but block 2's transport-local part index"
        );
    }
    rig_core::test_utils::history::assert_restated_agrees(
        &wire,
        reply_of(Shape::Unknown, Mode::Unary),
        reply_of(Shape::Unknown, Mode::Streaming),
    );
    rig_history_conformance::h03_empty_reply(fixture);
}

rig_history_conformance::history_conformance_suite! {
    wire: "gemini_rest",
    fixture: GeminiRestHistory,
    h03_stream_equals_whole: h03_gemini_rest_stream_equals_whole,
}
