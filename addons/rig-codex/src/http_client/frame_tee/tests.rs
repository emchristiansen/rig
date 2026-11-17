use super::*;
use crate::error::FrameEvidence;
use crate::http_client::CodexHttpClient;
use crate::responses_api::corrupt_frame_fixtures::MALFORMED_KNOWN_FRAMES;
use crate::responses_api::streaming::decode_frame;
use crate::responses_api::wire::CodexResponses;
use crate::wire::CallerIdentity;
use futures::StreamExt as _;
use rig_core::Model;
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
/// are the transport's own.
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
        handed_on,
        chunks.map(Bytes::from_static),
        "the bytes handed on are the transport's own"
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

/// The tee holds its capacity of most recent frames, and every frame of the
/// chunk it last handed on even when that is more.
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

    // One chunk that completes three frames: all three are kept while it
    // is the chunk in hand, and the next chunk brings the store back to
    // capacity.
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
    body.next().await.expect("a chunk").expect("reads");
    assert_eq!(kept.snapshot(), [text("1"), text("2"), text("3")]);
    body.next().await.expect("a chunk").expect("reads");
    assert_eq!(kept.snapshot(), [text("3"), text("4")]);
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
        [Ok(_), Err(Error::StreamEnded)]
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

/// The same holds for a call the driver reads as one body.
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

/// A frame Rig's decoder refuses is still kept when the failure surfaces,
/// however many frames followed it in the chunk that carried it and
/// whatever the capacity: the frames of the chunk being handed on are never
/// dropped for capacity.
#[tokio::test]
async fn the_refused_frame_is_kept_when_frames_follow_it_in_its_chunk() {
    let malformed = MALFORMED_KNOWN_FRAMES[0].frame;
    let trailing = r#"{"type":"future.event"}"#;
    for frames in [1, 8] {
        let mut reply = format!("data: {CREATED}\n\ndata: {malformed}\n\n");
        for _ in 0..9 {
            reply.push_str(&format!("data: {trailing}\n\n"));
        }
        let kept = kept_after_a_failed_stream(reply.into_bytes(), frames).await;
        assert_eq!(kept.len(), 11, "capacity {frames}: the whole chunk is kept");
        let refused: Vec<_> = kept
            .iter()
            .filter_map(|frame| match frame {
                FrameEvidence::Text(frame) => decode_frame(frame).err(),
                other => panic!("every kept frame is exact text, got {other:?}"),
            })
            .collect();
        let [corrupt] = refused.as_slice() else {
            panic!("capacity {frames}: one kept frame fails to decode, got {refused:?}");
        };
        assert_eq!(corrupt.frame(), Some(malformed));
    }
}
