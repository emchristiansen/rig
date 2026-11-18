use super::*;
use crate::error::FrameEvidence;
use crate::http_client::CodexHttpClient;
use crate::responses_api::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES;
use crate::responses_api::streaming::decode_frame;
use crate::responses_api::wire::CodexResponses;
use crate::wire::CallerIdentity;
use futures::StreamExt as _;
use rig_core::Model;
use rig_core::ProviderError;
use rig_core::completion::CompletionRequest;
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_http::http_client::Error;
use rig_http::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};

fn capacity(frames: usize) -> NonZeroUsize {
    NonZeroUsize::new(frames).expect("a non-zero capacity")
}

fn chunk(bytes: &'static [u8]) -> Result<Bytes> {
    Ok(Bytes::from_static(bytes))
}

fn get() -> Request<Bytes> {
    Request::builder()
        .uri("https://example.invalid/responses")
        .body(Bytes::new())
        .expect("builds")
}

fn text(frame: &str) -> FrameEvidence {
    FrameEvidence::Text(frame.to_owned())
}

/// Read a streamed reply to its end, returning its chunks as handed on.
async fn drained(response: StreamingResponse) -> Vec<Result<Bytes>> {
    response.into_body().collect().await
}

/// Each server-sent event's data is kept exactly as sent, whatever chunk
/// boundaries it crossed; a heartbeat is not a frame; and the bytes handed on
/// are the transport's own, a chunk that completes two frames as two pieces.
#[tokio::test]
async fn a_streamed_reply_keeps_each_events_data_as_sent() {
    let chunks = [
        &b"data: {\"a\" : 1,  \"b\":[ ]}\n\nda"[..],
        &b"ta: second\n\ndata:  \n\n"[..],
        &b"data: [DONE]\n\n"[..],
    ];
    let transport = SequencedStreamingHttpClient::new(chunks.iter().map(|c| chunk(c)).collect());
    let tee = HttpFrameTee::new(transport, Framing::Sse, capacity(8));
    let kept = tee.frames();

    let response = tee.send_streaming(get()).await.expect("the reply opens");
    assert!(
        kept.snapshot().is_empty(),
        "nothing is kept before it is read"
    );
    let handed_on: Vec<Bytes> = drained(response)
        .await
        .into_iter()
        .map(|chunk| chunk.expect("a chunk"))
        .collect();
    assert_eq!(
        handed_on.concat(),
        chunks.concat(),
        "the bytes handed on are the transport's own"
    );
    assert_eq!(
        handed_on,
        [
            &b"data: {\"a\" : 1,  \"b\":[ ]}\n\n"[..],
            &b"da"[..],
            &b"ta: second\n\n"[..],
            &b"data:  \n\n"[..],
            &b"data: [DONE]\n\n"[..],
        ]
        .map(Bytes::from_static),
        "a piece ends where a frame does, or where the transport's chunk does"
    );
    assert_eq!(
        kept.snapshot(),
        [
            text("{\"a\" : 1,  \"b\":[ ]}"),
            text("second"),
            text("[DONE]")
        ]
    );
    assert_eq!(kept.last(), Some(text("[DONE]")));
}

/// The tee never holds more than its capacity of frames, however many one
/// chunk completes: a chunk is handed on a frame at a time.
#[tokio::test]
async fn a_streamed_reply_keeps_only_its_most_recent_frames() {
    let transport = SequencedStreamingHttpClient::new(vec![
        chunk(b"data: 1\n\n"),
        chunk(b"data: 2\n\n"),
        chunk(b"data: 3\n\n"),
    ]);
    let tee = HttpFrameTee::new(transport, Framing::Sse, capacity(2));
    let kept = tee.frames();
    drained(tee.send_streaming(get()).await.expect("the reply opens")).await;
    assert_eq!(kept.take(), [text("2"), text("3")]);
    assert_eq!(kept.capacity(), capacity(2));

    // One chunk that completes three frames arrives as three pieces, and
    // the store holds two frames at most after each.
    let transport = SequencedStreamingHttpClient::new(vec![
        chunk(b"data: 1\n\ndata: 2\n\ndata: 3\n\n"),
        chunk(b"data: 4\n\n"),
    ]);
    let tee = HttpFrameTee::new(transport, Framing::Sse, capacity(2));
    let kept = tee.frames();
    let mut body = tee
        .send_streaming(get())
        .await
        .expect("the reply opens")
        .into_body();
    let mut seen = Vec::new();
    while let Some(piece) = body.next().await {
        seen.push((piece.expect("reads"), kept.snapshot()));
    }
    assert_eq!(
        seen,
        [
            (Bytes::from_static(b"data: 1\n\n"), vec![text("1")]),
            (
                Bytes::from_static(b"data: 2\n\n"),
                vec![text("1"), text("2")]
            ),
            (
                Bytes::from_static(b"data: 3\n\n"),
                vec![text("2"), text("3")]
            ),
            (
                Bytes::from_static(b"data: 4\n\n"),
                vec![text("3"), text("4")]
            ),
        ]
    );
}

/// A reply is handed on with its bytes unchanged and in order, in pieces
/// that each complete at most one frame, whatever the transport's chunks:
/// several events in one chunk, an event across chunks, heartbeats and
/// comments that are no frame, and each spelling of a line end. The frames
/// kept are the frames Rig's framer cuts from the whole reply.
#[tokio::test]
async fn a_reply_is_handed_on_a_frame_at_a_time_with_its_bytes_unchanged() {
    let chunks: [&'static [u8]; 6] = [
        b": comment\n\ndata: 1\n\ndata:\n\ndata: 2\r\n\r\ndata: 3\r\rda",
        b"ta: 4\nid: 9\n",
        b"\ndata: 5\r",
        b"\n\r",
        b"\nevent: x\ndata: 6\ndata: 7\n\ndata: 8\n\n",
        b"data: unfinished",
    ];
    let whole: Vec<u8> = chunks.concat();
    let expected: Vec<FrameEvidence> = Framing::Sse
        .split(&whole)
        .into_iter()
        .map(|frame| text(&frame.as_str()))
        .collect();
    assert_eq!(expected.len(), 7, "the fixture's frames: {expected:?}");

    for frames in [1, 3, 64] {
        let transport =
            SequencedStreamingHttpClient::new(chunks.iter().map(|bytes| chunk(bytes)).collect());
        let tee = HttpFrameTee::new(transport, Framing::Sse, capacity(frames));
        let kept = tee.frames();
        let mut body = tee
            .send_streaming(get())
            .await
            .expect("the reply opens")
            .into_body();
        let mut handed_on = Vec::new();
        let mut all = Vec::new();
        while let Some(piece) = body.next().await {
            handed_on.extend_from_slice(&piece.expect("reads"));
            let now = kept.snapshot();
            assert!(now.len() <= frames, "capacity {frames}: holds {now:?}");
            let newest = now.last().cloned();
            if newest.is_some() && newest.as_ref() != all.last() {
                all.extend(newest);
            }
            let cut_so_far = Framing::Sse
                .framer()
                .map(|mut framer| framer.push(&handed_on).count());
            assert_eq!(
                cut_so_far,
                Some(all.len()),
                "capacity {frames}: each piece completes at most one frame"
            );
        }
        assert_eq!(handed_on, whole, "capacity {frames}: the bytes handed on");
        assert_eq!(
            all, expected,
            "capacity {frames}: the frames kept, in order"
        );
    }
}

/// A transport failure mid-reply is handed on as it is, and the frames cut
/// before it stay kept.
#[tokio::test]
async fn a_failed_stream_keeps_the_frames_cut_before_it_failed() {
    let transport = SequencedStreamingHttpClient::new(vec![
        chunk(b"data: first\n\ndata: unfin"),
        Err(Error::StreamEnded),
    ]);
    let tee = HttpFrameTee::new(transport, Framing::Sse, capacity(4));
    let kept = tee.frames();
    let chunks = drained(tee.send_streaming(get()).await.expect("the reply opens")).await;
    assert!(matches!(
        chunks.as_slice(),
        [Ok(first), Ok(rest), Err(Error::StreamEnded)]
            if first == &b"data: first\n\n"[..] && rest == &b"data: unfin"[..]
    ));
    assert_eq!(
        kept.snapshot(),
        [text("first")],
        "an event that never ended is not a frame"
    );
}

/// A streamed reply that is one whole frame is kept once the stream ends,
/// and a line-delimited one line by line, its unterminated last line too.
#[tokio::test]
async fn a_streamed_reply_is_cut_by_the_wires_framing() {
    let whole = SequencedStreamingHttpClient::new(vec![chunk(b"{\"a\":"), chunk(b" 1}")]);
    let tee = HttpFrameTee::new(whole, Framing::Whole, capacity(4));
    let kept = tee.frames();
    drained(tee.send_streaming(get()).await.expect("the reply opens")).await;
    assert_eq!(kept.snapshot(), [text("{\"a\": 1}")]);

    let lines = SequencedStreamingHttpClient::new(vec![chunk(b"{\"n\":1}\n{\"n\""), chunk(b":2}")]);
    let tee = HttpFrameTee::new(lines, Framing::Ndjson, capacity(4));
    let kept = tee.frames();
    drained(tee.send_streaming(get()).await.expect("the reply opens")).await;
    assert_eq!(kept.snapshot(), [text("{\"n\":1}"), text("{\"n\":2}")]);
}

/// A unary body is cut as the driver cuts it: whole, or event by event when
/// the wire frames its replies as server-sent events. The body handed on is
/// the transport's.
#[tokio::test]
async fn a_unary_reply_is_kept_once_its_body_is_read() {
    let body = b"{ \"id\" : \"resp_1\" }";
    let tee = HttpFrameTee::new(
        RecordingHttpClient::new(&body[..]),
        Framing::Whole,
        capacity(4),
    );
    let kept = tee.frames();
    let response = tee.send::<Bytes, Bytes>(get()).await.expect("answers");
    assert!(
        kept.snapshot().is_empty(),
        "nothing is kept before it is read"
    );
    let handed_on = response.into_body().await.expect("the body reads");
    assert_eq!(handed_on, Bytes::from_static(body));
    assert_eq!(kept.snapshot(), [text("{ \"id\" : \"resp_1\" }")]);
    assert_eq!(tee.inner().requests().len(), 1);

    let events = b"data: one\n\ndata: two\n\n";
    let tee = HttpFrameTee::new(
        RecordingHttpClient::new(&events[..]),
        Framing::Sse,
        capacity(4),
    );
    let kept = tee.frames();
    let response = tee.send::<Bytes, Bytes>(get()).await.expect("answers");
    response.into_body().await.expect("the body reads");
    assert_eq!(kept.snapshot(), [text("one"), text("two")]);

    let not_text = HttpFrameTee::new(
        RecordingHttpClient::new(&[0xff, 0x7b][..]),
        Framing::Whole,
        capacity(1),
    );
    let kept = not_text.frames();
    not_text
        .send::<Bytes, Bytes>(get())
        .await
        .expect("answers")
        .into_body()
        .await
        .expect("the body reads");
    assert_eq!(kept.last(), Some(FrameEvidence::Bytes(vec![0xff, 0x7b])));
}

fn codex_wire() -> CodexResponses {
    CodexResponses::new(Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        "gpt-5.4",
    ))
    .expect("the ChatGPT dialect speaks the Codex contract")
    .with_caller_identity(
        CallerIdentity::new("client_exec", "client_exec/1.2.3 (test)", None)
            .expect("a valid caller identity"),
    )
}

const CREATED: &str = r#"{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":0,"status":"in_progress","model":"gpt-5.4","output":[]}}"#;

/// A reply whose second event is a frame of a modelled type that fails its
/// typed decoding, written with irregular spacing.
fn malformed_reply() -> Bytes {
    Bytes::from(format!(
        "data: {CREATED}\n\ndata: {}\n\n",
        MALFORMED_KNOWN_FRAMES[0].frame
    ))
}

/// The kept frames hold the malformed one exactly as the provider sent it,
/// and this crate's decoder names it with the frame as received.
fn assert_the_malformed_frame_was_kept(frames: &[FrameEvidence]) {
    let malformed = &MALFORMED_KNOWN_FRAMES[0];
    assert_eq!(frames, [text(CREATED), text(malformed.frame)]);
    let refused: Vec<_> = frames
        .iter()
        .filter_map(|frame| match frame {
            FrameEvidence::Text(frame) => decode_frame(frame).err(),
            other => panic!("every kept frame is exact text, got {other:?}"),
        })
        .collect();
    let [corrupt] = refused.as_slice() else {
        panic!("exactly one kept frame fails to decode, got {refused:?}");
    };
    assert_eq!(corrupt.event_type(), Some(malformed.event_type));
    assert_eq!(corrupt.frame(), Some(malformed.frame));
    assert_eq!(frames[1], FrameEvidence::Text(malformed.frame.to_owned()));
}

/// The use the tee exists for, through Rig's own driver: a streamed reply
/// carrying a frame Rig's decoder refuses fails the stream, and the tee
/// holds that frame exactly as the provider sent it.
#[tokio::test]
async fn a_frame_rigs_decoder_refuses_is_kept_as_the_provider_sent_it() {
    let wire = codex_wire();
    let tee = HttpFrameTee::new(
        SequencedStreamingHttpClient::new(vec![Ok(malformed_reply())]),
        Framing::Sse,
        capacity(8),
    );
    let kept = tee.frames();
    let model = Model::new(wire.clone(), CodexHttpClient::new(tee, wire));
    let mut stream = model
        .stream(CompletionRequest::new("say hi"))
        .expect("the request encodes");
    let mut failed = false;
    while let Some(event) = stream.next().await {
        failed |= event.is_err();
    }
    assert!(
        failed || stream.finish().await.is_err(),
        "a malformed frame of a modelled type fails the stream"
    );
    assert_the_malformed_frame_was_kept(&kept.snapshot());
}

/// A unary body arrives whole, so its frames are cut together and all of
/// them are kept, whatever the capacity: nothing says which one a failure
/// would be about. They are held until taken.
#[tokio::test]
async fn a_unary_reply_keeps_every_frame_of_its_body_whatever_the_capacity() {
    let mut events = String::new();
    for index in 0..1000 {
        events.push_str(&format!("data: {index}\n\n"));
    }
    let tee = HttpFrameTee::new(
        RecordingHttpClient::new(Bytes::from(events)),
        Framing::Sse,
        capacity(2),
    );
    let kept = tee.frames();
    let response = tee.send::<Bytes, Bytes>(get()).await.expect("answers");
    response.into_body().await.expect("the body reads");
    let frames = kept.snapshot();
    assert_eq!(
        frames.len(),
        1000,
        "every frame of the body, beyond capacity 2"
    );
    assert_eq!(frames.first(), Some(&text("0")));
    assert_eq!(frames.last(), Some(&text("999")));
    assert_eq!(kept.take().len(), 1000);
    assert!(kept.snapshot().is_empty(), "taken, the store holds nothing");
}

/// The same holds for a call the driver reads as one body: the refused
/// frame is among the kept frames wherever in the body it stands.
#[tokio::test]
async fn a_frame_rigs_decoder_refuses_in_a_unary_call_is_kept_too() {
    let wire = codex_wire();
    let tee = HttpFrameTee::new(
        RecordingHttpClient::new(malformed_reply()),
        Framing::Sse,
        capacity(8),
    );
    let kept = tee.frames();
    let model = Model::new(wire.clone(), CodexHttpClient::new(tee, wire));
    model
        .call(CompletionRequest::new("say hi"))
        .await
        .expect_err("a malformed frame of a modelled type fails the call");
    assert_the_malformed_frame_was_kept(&kept.snapshot());
}

/// A reply Rig's decoder refuses, drained through Rig's own driver over a
/// tee of `capacity` frames. Returns what the tee holds when the failure
/// has surfaced.
async fn kept_after_a_failed_stream(reply: Vec<u8>, frames: usize) -> Vec<FrameEvidence> {
    let wire = codex_wire();
    let tee = HttpFrameTee::new(
        SequencedStreamingHttpClient::new(vec![Ok(Bytes::from(reply))]),
        Framing::Sse,
        capacity(frames),
    );
    let kept = tee.frames();
    let model = Model::new(wire.clone(), CodexHttpClient::new(tee, wire));
    let mut stream = model
        .stream(CompletionRequest::new("say hi"))
        .expect("the request encodes");
    let mut failed = false;
    while let Some(event) = stream.next().await {
        failed |= event.is_err();
    }
    assert!(failed || stream.finish().await.is_err(), "the stream fails");
    kept.snapshot()
}

/// Text Rig's framer read from bytes that were not UTF-8 is kept as
/// replacement-decoded, never as the frame as received; a frame that holds
/// a genuine U+FFFD the provider sent is exact text.
#[tokio::test]
async fn replacement_decoded_text_is_not_kept_as_exact() {
    let malformed = MALFORMED_KNOWN_FRAMES[0].frame;
    let broken = malformed.replacen('{', "{\"x\":\"\u{1}\",", 1);
    assert!(
        broken.contains('\u{1}'),
        "a marker byte to corrupt: {broken}"
    );

    // The provider's own U+FFFD, valid UTF-8.
    let genuine = broken.replace('\u{1}', "\u{fffd}");
    let kept = kept_after_a_failed_stream(
        format!("data: {CREATED}\n\ndata: {genuine}\n\n").into_bytes(),
        8,
    )
    .await;
    assert_eq!(kept, [text(CREATED), text(&genuine)]);

    // The same event with the byte 0xff where the character was.
    let mut invalid = format!("data: {CREATED}\n\ndata: {broken}\n\n").into_bytes();
    let at = invalid
        .iter()
        .position(|byte| *byte == 1)
        .expect("the marker");
    invalid[at] = 0xff;
    let kept = kept_after_a_failed_stream(invalid, 8).await;
    assert_eq!(
        kept,
        [
            FrameEvidence::Decoded(CREATED.to_owned()),
            FrameEvidence::Decoded(genuine.clone()),
        ],
        "one chunk held the invalid byte, so nothing cut from it is exact; the text is \
         what Rig read, a replacement character where the byte was"
    );

    // A frame whose chunk arrived before the invalid byte stays exact.
    let wire = codex_wire();
    let mut second = format!("data: {broken}\n\n").into_bytes();
    let at = second
        .iter()
        .position(|byte| *byte == 1)
        .expect("the marker");
    second[at] = 0xff;
    let tee = HttpFrameTee::new(
        SequencedStreamingHttpClient::new(vec![
            Ok(Bytes::from(format!("data: {CREATED}\n\n"))),
            Ok(Bytes::from(second)),
        ]),
        Framing::Sse,
        capacity(8),
    );
    let kept = tee.frames();
    let model = Model::new(wire.clone(), CodexHttpClient::new(tee, wire));
    let mut stream = model
        .stream(CompletionRequest::new("say hi"))
        .expect("the request encodes");
    while stream.next().await.is_some() {}
    assert_eq!(
        kept.snapshot(),
        [text(CREATED), FrameEvidence::Decoded(genuine)]
    );
}

/// A frame Rig's decoder refuses is the newest kept frame when the failure
/// surfaces, however many frames followed it in the chunk that carried it
/// and whatever the capacity: the driver stops reading at the refused
/// frame, and no frame after it was cut. The store holds no more than its
/// capacity throughout.
#[tokio::test]
async fn the_refused_frame_is_the_newest_kept_whatever_follows_it_in_its_chunk() {
    // Not JSON, so Rig's decoder refuses it where it stands; a frame Rig
    // can read but this crate's typed decoding cannot would instead fail
    // the reply at its end, after every later frame was read.
    let refused = r#"{nope}"#;
    let trailing = r#"{"type":"future.event"}"#;
    let mut reply = format!("data: {CREATED}\n\ndata: {refused}\n\n");
    for _ in 0..1000 {
        reply.push_str(&format!("data: {trailing}\n\n"));
    }
    for frames in [1, 8] {
        let wire = codex_wire();
        let tee = HttpFrameTee::new(
            SequencedStreamingHttpClient::new(vec![Ok(Bytes::from(reply.clone()))]),
            Framing::Sse,
            capacity(frames),
        );
        let kept = tee.frames();
        let model = Model::new(wire.clone(), CodexHttpClient::new(tee, wire));
        let mut stream = model
            .stream(CompletionRequest::new("say hi"))
            .expect("the request encodes");
        let mut failure = None;
        while let Some(event) = stream.next().await {
            assert!(kept.snapshot().len() <= frames, "capacity {frames}");
            if let Err(error) = event {
                failure = Some(error);
            }
        }
        let failure = match failure {
            Some(failure) => failure,
            None => stream.finish().await.expect_err("the stream fails"),
        };
        assert!(
            matches!(failure, ProviderError::Json(_)),
            "capacity {frames}: the decoder refused the frame itself: {failure:?}"
        );
        assert_eq!(kept.last(), Some(text(refused)), "capacity {frames}");
        let expected = if frames == 1 {
            vec![text(refused)]
        } else {
            vec![text(CREATED), text(refused)]
        };
        assert_eq!(kept.snapshot(), expected, "capacity {frames}");
    }
}
