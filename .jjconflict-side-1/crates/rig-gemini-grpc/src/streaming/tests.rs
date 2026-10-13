use super::*;
use crate::completion::tests::{complete, stream_from_events};
use base64::Engine as _;
use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, CarriedReasoning, CarriedReasoningPart};
use rig_core::streaming::{Item, StreamEvent};

fn thought_part(text: &str, signature: &[u8]) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text(text.to_string())),
        thought: true,
        thought_signature: signature.to_vec(),
        ..Default::default()
    }
}

fn response(parts: Vec<proto::Part>, finish_reason: i32) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts,
                role: "model".to_string(),
            }),
            finish_reason,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Drive protobuf events through the full normalized path and collect the
/// Reasoning blocks the consumer sees.
async fn reasoning_blocks(events: Vec<proto::GenerateContentResponse>) -> Vec<CarriedReasoning> {
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut blocks = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::CarriedReasoning(reasoning),
            ..
        }) = item.expect("stream item should be ok")
        {
            blocks.push(reasoning);
        }
    }
    blocks
}

// A signed thought part reaches the stream as completed carried reasoning
// that restates the accumulated text with the wire signature in base64.
#[tokio::test]
async fn signed_thought_part_restates_accumulated_text_with_signature() {
    let signature_bytes = b"opaque-signature".as_slice();
    let events = vec![
        response(vec![thought_part("think1 ", b"")], 0),
        response(
            vec![thought_part("think2", signature_bytes)],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("the signed part must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![CarriedReasoningPart::Text {
            text: "think1 think2".to_string(),
            // The expected encoding is the unary path's: standard base64
            // over the wire's signature bytes.
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

// The wire's real signed shape: the signature rides a trailing EMPTY
// thought part. It must still emit a signed block so the signature
// survives into chat history (signature-only case).
#[tokio::test]
async fn signature_on_empty_trailer_part_still_carries_the_signature() {
    let signature_bytes = b"trailer-signature".as_slice();
    let events = vec![
        response(vec![thought_part("thinking...", b"")], 0),
        response(
            vec![thought_part("", signature_bytes)],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("the signed trailer must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![CarriedReasoningPart::Text {
            text: "thinking...".to_string(),
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

// Signature with no thought text anywhere in the stream: the signed block
// still surfaces (empty text) rather than dropping the signature.
#[tokio::test]
async fn signature_without_any_thought_text_still_surfaces() {
    let signature_bytes = b"lone-signature".as_slice();
    let events = vec![response(
        vec![thought_part("", signature_bytes)],
        proto::candidate::FinishReason::Stop as i32,
    )];

    let blocks = reasoning_blocks(events).await;
    let signed = blocks
        .last()
        .expect("a lone signature must yield a Reasoning block");
    assert_eq!(
        signed.content,
        vec![CarriedReasoningPart::Text {
            text: String::new(),
            signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
        }]
    );
}

fn function_call_part(name: &str, id: &str) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
            name: name.to_string(),
            args: None,
            id: id.to_string(),
        })),
        ..Default::default()
    }
}

// Two id-less calls to the same tool remain distinct, with real Local IDs.
#[tokio::test]
async fn two_id_less_function_calls_stay_distinct() {
    let events = vec![response(
        vec![
            function_call_part("get_weather", ""),
            function_call_part("get_weather", ""),
        ],
        proto::candidate::FinishReason::Stop as i32,
    )];

    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut correlators = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::End {
            content: AssistantContent::ToolCall(tool_call),
            ..
        }) = item.expect("stream item should be ok")
        {
            assert_eq!(tool_call.function.name.as_str(), "get_weather");
            assert!(tool_call.id.is_local(), "the provider sent no call id");
            correlators.push(tool_call.id.clone());
        }
    }

    assert_eq!(correlators.len(), 2, "correlators: {correlators:?}");
    assert_ne!(
        correlators.first(),
        correlators.last(),
        "each call needs its own correlator"
    );
}

// ---- #2258 H4: tool-protocol finish reasons must fail the turn ----

fn failed_response(
    reason: proto::candidate::FinishReason,
    finish_message: Option<&str>,
) -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![],
                role: "model".to_string(),
            }),
            finish_reason: reason as i32,
            finish_message: finish_message.map(str::to_owned),
            ..Default::default()
        }],
        ..Default::default()
    }
}

struct Drained {
    errors: Vec<String>,
    reached_terminal: bool,
    text: String,
}

async fn drain(events: Vec<proto::GenerateContentResponse>) -> Drained {
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    let mut drained = Drained {
        errors: Vec::new(),
        reached_terminal: false,
        text: String::new(),
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => drained.text.push_str(&text),
            Ok(_) => {}
            Err(error) => drained.errors.push(error.to_string()),
        }
    }
    match stream.finish().await {
        Ok(_) => drained.reached_terminal = true,
        Err(error) => {
            let expected = error.to_string();
            assert_eq!(
                drained.errors.last().map(String::as_str),
                Some(expected.as_str()),
                "the separate finish retains the stream's failure"
            );
        }
    }

    drained
}

// An aborted tool protocol fails instead of completing the turn.
#[tokio::test]
async fn malformed_function_call_fails_the_stream_with_no_terminal() {
    let drained = drain(vec![failed_response(
        proto::candidate::FinishReason::MalformedFunctionCall,
        Some("could not parse the function call"),
    )])
    .await;

    assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
    let error = drained.errors.first().expect("one error");
    assert!(
        error.contains("MalformedFunctionCall")
            && error.contains("could not parse the function call"),
        "error should name the reason and carry finish_message: {error}"
    );
    assert!(
        !drained.reached_terminal,
        "a failed turn must not synthesize a terminal record"
    );
}

#[tokio::test]
async fn unexpected_and_too_many_tool_calls_also_fail_the_stream() {
    for reason in [
        proto::candidate::FinishReason::UnexpectedToolCall,
        proto::candidate::FinishReason::TooManyToolCalls,
    ] {
        let drained = drain(vec![failed_response(reason, None)]).await;
        assert_eq!(
            drained.errors.len(),
            1,
            "{} should fail the stream",
            reason.as_str_name()
        );
        assert!(!drained.reached_terminal);
    }
}

// The stream latches the failure, so a later genuine terminal cannot
// turn the aborted protocol into a completed response.
#[tokio::test]
async fn frames_after_a_tool_protocol_failure_are_not_interpreted() {
    let drained = drain(vec![
        failed_response(proto::candidate::FinishReason::MalformedFunctionCall, None),
        response(
            vec![proto::Part {
                data: Some(proto::part::Data::Text("recovered?".to_string())),
                ..Default::default()
            }],
            proto::candidate::FinishReason::Stop as i32,
        ),
    ])
    .await;

    assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
    assert!(drained.text.is_empty(), "text: {:?}", drained.text);
    assert!(!drained.reached_terminal);
}

// Ordinary terminals are untouched by the new gate.
#[tokio::test]
async fn non_tool_protocol_finish_reasons_still_complete_the_turn() {
    let drained = drain(vec![response(
        vec![proto::Part {
            data: Some(proto::part::Data::Text("done".to_string())),
            ..Default::default()
        }],
        proto::candidate::FinishReason::Stop as i32,
    )])
    .await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert_eq!(drained.text, "done");
    assert!(drained.reached_terminal);
}

// The unary path routes through the same helper, so the two surfaces
// report an aborted tool protocol with the same message.
#[tokio::test]
async fn unary_and_streaming_report_the_same_tool_protocol_error() {
    let response = failed_response(
        proto::candidate::FinishReason::TooManyToolCalls,
        Some("budget exhausted"),
    );
    let drained = drain(vec![response.clone()]).await;
    assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
    assert!(!drained.reached_terminal);
    let expected = drained.errors.first().expect("the streaming decoder fails");
    assert_eq!(
        expected.as_str(),
        "ResponseError: Gemini stopped with finish_reason=TooManyToolCalls: budget exhausted",
    );

    match complete(response) {
        Err(err) => assert_eq!(err.to_string(), *expected),
        Ok(_) => panic!("the unary path must fail on a tool-protocol finish reason"),
    }
}

// The streaming path maps both the initial `stream_generate_content` RPC
// failure and any per-item iteration error through `rpc_error`. Pin that the
// mapping preserves the provider's status text and exposes no HTTP status.
#[test]
fn stream_rpc_error_preserves_status_text_without_http_status() {
    let status = tonic::Status::unavailable("boom");
    let expected = status.to_string();

    let err = super::super::completion::rpc_error(&status);

    assert_eq!(err.provider_response_body(), Some(expected.as_str()));
    assert_eq!(err.provider_response_status(), None);
}

fn text_part(text: &str) -> proto::Part {
    proto::Part {
        data: Some(proto::part::Data::Text(text.to_string())),
        ..Default::default()
    }
}

/// The terminal frame of a stream: the last `GenerateContentResponse`,
/// carrying usage, a finish reason and the response id, so the stream
/// ends with a fully populated terminal record.
fn terminal_frame() -> proto::GenerateContentResponse {
    proto::GenerateContentResponse {
        candidates: vec![proto::Candidate {
            content: Some(proto::Content {
                parts: vec![proto::Part {
                    data: Some(proto::part::Data::Text("!".to_string())),
                    ..Default::default()
                }],
                role: "model".to_string(),
            }),
            finish_reason: proto::candidate::FinishReason::Stop as i32,
            ..Default::default()
        }],
        usage_metadata: Some(proto::UsageMetadata {
            prompt_token_count: 3,
            candidates_token_count: 2,
            total_token_count: 5,
            cached_content_token_count: 0,
            ..Default::default()
        }),
        model_version: "gemini-2.5-flash".to_string(),
        response_id: "resp-grpc-stream".to_string(),
        ..Default::default()
    }
}

/// Drive the endpoint's protobuf frames through its actual decoder and fold.
async fn normalized_terminal(events: Vec<proto::GenerateContentResponse>) -> CompletionResponse {
    let mut stream = stream_from_events(events.into_iter().map(Ok).collect());
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    stream
        .finish()
        .await
        .expect("the stream must end with a terminal record")
}

/// The original two-frame sequence yields the whole unary-equivalent raw
/// response. The final-frame-only capture and its DTO view were retired.
#[tokio::test]
async fn stream_from_events_terminal_carries_raw() {
    let mut stream = stream_from_events(
        vec![response(vec![text_part("hi")], 0), terminal_frame()]
            .into_iter()
            .map(Ok)
            .collect(),
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let terminal = stream.finish().await.expect("terminal record");
    let mut whole = terminal_frame();
    whole
        .candidates
        .first_mut()
        .expect("the fixture has a candidate")
        .content = Some(proto::Content {
        parts: vec![text_part("hi!")],
        role: "model".to_owned(),
    });

    let raw = &terminal.raw;
    assert_eq!(
        *raw,
        crate::rest::to_rest(&whole).expect("whole response transcodes")
    );
    let typed: proto::GenerateContentResponse =
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(typed, whole);
    assert_eq!(terminal.usage.total_tokens, Some(5));
}

/// The same original two-frame sequence captures the whole response in
/// REST JSON. Its typed round trip and real decoder preserve terminal metadata.
#[tokio::test]
async fn terminal_raw_round_trips_into_the_terminal_type() {
    let terminal =
        normalized_terminal(vec![response(vec![text_part("hi")], 0), terminal_frame()]).await;
    let mut whole = terminal_frame();
    whole
        .candidates
        .first_mut()
        .expect("the fixture has a candidate")
        .content = Some(proto::Content {
        parts: vec![text_part("hi!")],
        role: "model".to_owned(),
    });

    let raw = &terminal.raw;
    assert_eq!(
        *raw,
        crate::rest::to_rest(&whole).expect("whole response transcodes")
    );
    let typed: proto::GenerateContentResponse =
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(typed, whole);
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&serde_json::json!("STOP"))
    );

    // Feeding the capture back through the same pipeline tells the same
    // story as the terminal the stream produced.
    assert_eq!(
        typed
            .candidates
            .first()
            .expect("terminal candidate")
            .finish_reason,
        proto::candidate::FinishReason::Stop as i32,
    );
    let renormalized = normalized_terminal(vec![typed]).await;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason(), renormalized.finish_reason());
    assert_eq!(terminal.model(), renormalized.model());
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(
        terminal.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
    assert_eq!(terminal.model(), Some("gemini-2.5-flash"));
    assert_eq!(
        terminal.identity().response_id.as_deref(),
        Some("resp-grpc-stream")
    );
}

/// The original text-only sequence keeps its actual gRPC provider Origin.
/// It has no independent terminal reasoning-issuer view in the current API.
#[tokio::test]
async fn the_stream_names_the_gemini_service_as_reasoning_issuer() {
    let terminal =
        normalized_terminal(vec![response(vec![text_part("hi")], 0), terminal_frame()]).await;
    assert_eq!(
        terminal.origin.provider,
        super::super::completion::PROVIDER_NAME
    );
}

/// A separately decoded reasoning fixture exposes its real per-item issuer.
#[tokio::test]
async fn decoded_carried_reasoning_names_the_gemini_service_as_issuer() {
    let terminal = normalized_terminal(vec![response(
        vec![thought_part("issuer thought", b"issuer-signature")],
        proto::candidate::FinishReason::Stop as i32,
    )])
    .await;
    let reasoning = match terminal.choice.as_slice() {
        [AssistantContent::CarriedReasoning(reasoning)] => reasoning,
        other => panic!("one decoded reasoning part: {other:?}"),
    };
    assert_eq!(
        reasoning.payload.provider.as_deref(),
        Some(rig_core::providers::gemini::completion::PROVIDER_NAME),
    );
    assert_eq!(reasoning.provider(), Some("gcp.gemini"));
    assert_eq!(
        reasoning.content,
        vec![CarriedReasoningPart::Text {
            text: "issuer thought".to_owned(),
            signature: Some(base64::engine::general_purpose::STANDARD.encode(b"issuer-signature")),
        }]
    );
}

/// The streamed twin of a signed answer: the text, then an empty part
/// carrying the signature. The signature stays on its own empty part.
#[tokio::test]
async fn a_trailing_signed_part_keeps_its_signature_on_its_own_text() {
    let mut signed = text_part("");
    signed.thought_signature = b"sig".to_vec();
    let mut stream = stream_from_events(
        vec![
            response(vec![text_part("289")], 0),
            response(vec![signed], 0),
            terminal_frame(),
        ]
        .into_iter()
        .map(Ok)
        .collect(),
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let choice = stream.finish().await.expect("terminal").choice;
    let texts: Vec<(String, bool)> = choice
        .iter()
        .map(|part| match part {
            rig_core::message::AssistantContent::Text(text) => (text.text.clone(), false),
            AssistantContent::CarriedText(text) => {
                (text.text().to_owned(), text.signed_answer().is_some())
            }
            other => panic!("answer text only: {other:?}"),
        })
        .collect();
    // The terminal frame's own text starts after the signed part.
    assert_eq!(
        texts,
        [
            ("289".to_owned(), false),
            (String::new(), true),
            ("!".to_owned(), false)
        ]
    );
}

/// A streamed reply's `raw` is the unary message its chunks add up to:
/// the text parts joined, the terminal chunk's finish, usage and ids. It
/// reads back into that prost message, equals the `raw` of the same reply
/// served unary, and re-normalizing it reproduces every normalized field.
#[tokio::test]
async fn streamed_raw_is_the_unary_message_the_chunks_add_up_to() {
    let terminal =
        normalized_terminal(vec![response(vec![text_part("hi")], 0), terminal_frame()]).await;
    let mut whole = terminal_frame();
    for candidate in &mut whole.candidates {
        candidate.content = Some(proto::Content {
            parts: vec![text_part("hi!")],
            role: "model".to_string(),
        });
    }

    let raw = &terminal.raw;
    let typed: proto::GenerateContentResponse =
        crate::rest::from_rest(raw.clone()).expect("raw must read back");
    assert_eq!(typed, whole);
    assert_eq!(
        *raw,
        complete(whole.clone())
            .expect("the unary reply decodes")
            .raw,
        "the streamed raw is the unary raw of the same reply"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&serde_json::json!("STOP"))
    );

    let renormalized = normalized_terminal(vec![typed]).await;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason(), renormalized.finish_reason());
    assert_eq!(terminal.model(), renormalized.model());
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(
        terminal.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
    assert_eq!(terminal.model(), Some("gemini-2.5-flash"));
    assert_eq!(
        terminal.identity().response_id.as_deref(),
        Some("resp-grpc-stream")
    );
}

/// #2475: a blocked prompt's only reply carries `promptFeedback` and no
/// candidate. It is the provider's refusal, naming the reason and ratings,
/// as on REST, never a truncated reply.
#[test]
fn a_blocked_prompt_is_a_refusal_on_grpc() {
    let blocked = proto::GenerateContentResponse {
        prompt_feedback: Some(proto::PromptFeedback {
            block_reason: proto::prompt_feedback::BlockReason::Safety as i32,
            safety_ratings: vec![proto::SafetyRating {
                category: proto::HarmCategory::HateSpeech as i32,
                probability: proto::safety_rating::HarmProbability::High as i32,
                blocked: true,
            }],
        }),
        ..Default::default()
    };
    let error = complete(blocked).expect_err("a blocked prompt is no answer");
    assert!(error.report().refusal, "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("block_reason=SAFETY")
            && message.contains("HARM_CATEGORY_HATE_SPEECH=HIGH"),
        "{message}"
    );
}

/// A part of a kind the proto does not declare arrives with no data, its
/// signature alone. The turn replays to the same model without it, never
/// as a part with no data.
#[test]
fn a_part_of_an_undeclared_kind_never_replays_without_data() {
    use rig_core::message::Message;
    let signed = proto::Part {
        data: None,
        thought_signature: b"sig".to_vec(),
        ..Default::default()
    };
    let response = complete(response(
        vec![signed, text_part("answer")],
        proto::candidate::FinishReason::Stop as i32,
    ))
    .expect("the reply decodes");
    let history = vec![
        Message::user("q"),
        Message::Assistant(response.continued(response.choice.clone())),
    ];
    let wire = crate::completion::GenerateContent::new(crate::completion::GEMINI_2_5_FLASH);
    let mut request = rig_core::completion::CompletionRequest::new("next");
    request.chat_history = rig_core::completion::adapt(&history, &wire);
    let request = rig_core::wire::Wire::encode(&wire, request, rig_core::wire::Mode::Unary)
        .expect("the history encodes");
    let parts: Vec<_> = request.contents.iter().flat_map(|c| &c.parts).collect();
    assert!(parts.iter().all(|part| part.data.is_some()), "{parts:?}");
    assert!(
        parts
            .iter()
            .any(|part| part.data == Some(proto::part::Data::Text("answer".to_owned()))),
        "{parts:?}"
    );
}

/// The cited text and sources of each citation on the reply's text blocks.
fn citations(response: &CompletionResponse) -> Vec<(String, Vec<String>, Vec<Option<f32>>)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            rig_core::message::AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            text.citations().iter().map(|citation| {
                let urls = citation
                    .sources
                    .iter()
                    .map(|source| match &source.location {
                        rig_core::message::SourceLocation::Url { url } => url.clone(),
                        other => format!("{other:?}"),
                    });
                let scores = citation.sources.iter().map(|source| source.confidence);
                (
                    text.cited(citation).unwrap_or_default().to_owned(),
                    urls.collect(),
                    scores.collect(),
                )
            })
        })
        .collect()
}

/// Hand-built replies, as no recording carries grounding: the proto keeps
/// the candidate's grounding and recitation metadata, so a segment counted
/// in bytes past a multi-byte dash cites its text and sources, unary or
/// streamed with the metadata on the last chunk.
#[tokio::test]
async fn grounding_and_recitations_cite_the_answer_unary_or_streamed() {
    const ANSWER: &str = "Spain won 2\u{2013}1. Oyarzabal scored late.";
    let quoted = "Oyarzabal scored late.";
    let at = ANSWER.find(quoted).expect("the answer holds it");
    let text = |text: &str| proto::Part {
        data: Some(proto::part::Data::Text(text.to_owned())),
        ..Default::default()
    };
    let web = |uri: &str| proto::GroundingChunk {
        chunk_type: Some(proto::grounding_chunk::ChunkType::Web(
            proto::grounding_chunk::Web {
                uri: Some(uri.to_owned()),
                title: Some("example".to_owned()),
            },
        )),
    };
    let metadata = |mut response: proto::GenerateContentResponse| {
        let candidate = response.candidates.first_mut().expect("a candidate");
        candidate.grounding_metadata = Some(proto::GroundingMetadata {
            grounding_chunks: vec![web("https://example.com/a"), web("https://example.com/b")],
            grounding_supports: vec![proto::GroundingSupport {
                segment: Some(proto::Segment {
                    part_index: 0,
                    start_index: at as i32,
                    end_index: (at + quoted.len()) as i32,
                    text: quoted.to_owned(),
                }),
                grounding_chunk_indices: vec![1, 0],
                confidence_scores: vec![0.5, 0.25],
            }],
        });
        candidate.citation_metadata = Some(proto::CitationMetadata {
            citation_sources: vec![proto::CitationSource {
                start_index: None,
                end_index: Some(9),
                uri: Some("https://example.com/c".to_owned()),
                license: None,
            }],
        });
        response
    };
    let expected = vec![
        (
            quoted.to_owned(),
            vec![
                "https://example.com/b".to_owned(),
                "https://example.com/a".to_owned(),
            ],
            vec![Some(0.5), Some(0.25)],
        ),
        (
            "Spain won".to_owned(),
            vec!["https://example.com/c".to_owned()],
            vec![None],
        ),
    ];

    let unary = complete(metadata(response(vec![text(ANSWER)], 1))).expect("the reply decodes");
    assert_eq!(citations(&unary), expected);

    let (head, tail) = ANSWER.split_at(at);
    let mut stream = stream_from_events(vec![
        Ok(response(vec![text(head)], 0)),
        Ok(metadata(response(vec![text(tail)], 1))),
    ]);
    while stream.next().await.is_some() {}
    let streamed = stream.finish().await.expect("the stream completes");
    assert_eq!(citations(&streamed), expected);
}
