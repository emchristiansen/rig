//! The driver's own tests, over a fake wire and a fake socket.
//!
//! These are the tests that used to live per provider or in
//! `http_client/sse/tests.rs`: the preserved non-success response facts, the
//! observation fact order, the connect-error-is-the-only-item rule, paging,
//! and the property the whole model exists for — a unary reply and a
//! streamed reply of the same content fold to the same response.

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;

use super::{Model, Transport};
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::framing::Framing;
use crate::message::{CallId, ToolName};
use crate::model::{ModelInfo, ModelList};
use crate::observe::{AdapterContext, AdapterUsage, AdapterVerdict, ObservationLog, Subject};
use crate::operation::{Block, Completion, Finish, ModelListing, ModelPage};
use crate::streaming::Streamed;
use crate::test_utils::{
    HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient, NonSuccessStreamingClient,
    RecordingHttpClient, SequencedHttpClient, SequencedStreamingHttpClient, TraceCapture,
};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, IncompleteEof, Mode, ObservationSink, Out,
    UsageObservation, Wire, WireEvent, WireFrame,
};

/// Fold one reply of `wire` over `http`.
pub(crate) async fn call<W, H>(
    wire: &W,
    http: &H,
    request: crate::wire::Request<W>,
    context: Option<AdapterContext>,
) -> Result<crate::wire::Response<W>, ProviderError>
where
    W: Wire,
    H: Transport<W>,
{
    let model = Model::new(wire.clone(), http.clone());
    match context {
        Some(context) => model.call_observed(request, context).await,
        None => model.call(request).await,
    }
}

/// One streamed reply of `wire` over `http`.
pub(crate) fn stream<W, H>(
    wire: &W,
    http: &H,
    request: crate::wire::Request<W>,
    context: Option<AdapterContext>,
) -> Result<Streamed<W::Op>, ProviderError>
where
    W: Wire,
    H: Transport<W>,
{
    let model = Model::new(wire.clone(), http.clone());
    match context {
        Some(context) => model.stream_observed(request, context),
        None => model.stream(request),
    }
}

/// One observed reply of `wire` over `http` in `mode`: the usage
/// observations it emitted, in order, and the response's usage.
///
/// # Panics
///
/// When the reply does not fold.
pub(crate) async fn observed_usage<W, T>(
    wire: W,
    http: T,
    mode: crate::wire::Mode,
) -> (Vec<crate::observe::AdapterUsage>, crate::completion::Usage)
where
    W: crate::wire::Wire<Op = crate::operation::Completion>,
    T: Transport<W>,
{
    use crate::observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    let model = Model::new(wire, http);
    let request = crate::completion::CompletionRequest::new("hi");
    let response = match mode {
        crate::wire::Mode::Unary => model.call_observed(request, context).await,
        crate::wire::Mode::Streaming => match model.stream_observed(request, context) {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        },
    }
    .unwrap_or_else(|error| panic!("the reply folds: {error}"));
    let observed = log
        .trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => match &observation.event {
                AdapterEvent::Usage { usage } => Some(usage.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    (observed, response.usage)
}

/// A wire whose reply is either a whole message (`{"text":…}`) or a stream
/// of the same shape's deltas — the Anthropic/OpenAI split, in miniature.
#[derive(Clone, Debug, PartialEq)]
struct Echo {
    framing: Framing,
    request_id_header: Option<&'static str>,
    response_header_prefix: Option<&'static str>,
    usage_observation: UsageObservation,
    relaxed_content_type: bool,
}

impl Echo {
    fn unary() -> Self {
        Self {
            framing: Framing::Whole,
            request_id_header: Some("request-id"),
            response_header_prefix: None,
            usage_observation: UsageObservation::FinalNormalized,
            relaxed_content_type: false,
        }
    }

    fn streaming() -> Self {
        Self {
            framing: Framing::Sse,
            request_id_header: Some("request-id"),
            response_header_prefix: None,
            usage_observation: UsageObservation::FinalNormalized,
            relaxed_content_type: false,
        }
    }

    /// A wire whose *unary* reply is an event stream — the Responses
    /// endpoint's shape on a dialect that always streams.
    fn sse_unary() -> Self {
        Self::streaming()
    }

    fn capturing(mut self, prefix: &'static str) -> Self {
        self.response_header_prefix = Some(prefix);
        self
    }

    /// The legacy payload-snapshot profile; normalized fixtures keep the default.
    fn snapshots(mut self) -> Self {
        self.usage_observation = UsageObservation::PayloadSnapshots;
        self
    }

    fn relaxed(mut self) -> Self {
        self.relaxed_content_type = true;
        self
    }
}

/// One frame of the fake wire.
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    /// The whole message: the unary reply's shape.
    Message { text: String, usage: Usage },
    /// One text delta.
    Delta { text: String },
    /// The provider's own end of turn.
    Stop { usage: Usage },
    /// A whole tool call whose block the provider declared complete, ending
    /// the turn when it carries usage.
    Tool {
        name: String,
        arguments: String,
        usage: Option<Usage>,
    },
    /// The provider's in-band failure, ending the stream without an end.
    Error { message: String },
}

#[derive(Clone, Copy, serde::Deserialize)]
struct Usage {
    output_tokens: u64,
}

#[derive(Default)]
struct EchoDecoder<'id> {
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion> for EchoDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "message" | "delta" | "stop" | "tool" | "error")
        })
    }

    fn decode(
        &mut self,
        event: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            // A whole message is the stream's text and end in one frame.
            Frame::Message { text, usage } => {
                out.run(Block::Text, &text)?;
                return self.end(out, usage);
            }
            Frame::Delta { text } => {
                out.run(Block::Text, &text)?;
            }
            Frame::Stop { usage } => return self.end(out, usage),
            Frame::Error { message } => return Err(ProviderError::Provider(message)),
            Frame::Tool {
                name,
                arguments,
                usage,
            } => {
                out.end_run()?;
                let name = ToolName::new(name)
                    .map_err(|error| ProviderError::Response(error.to_string()))?;
                let index = out.fresh_index();
                out.whole(
                    index,
                    Block::Call {
                        id: CallId::from_wire("call_1"),
                        name,
                        namespace: None,
                    },
                    serde_json::Value::Null,
                    &arguments,
                )?;
                if let Some(usage) = usage {
                    return self.end(out, usage);
                }
            }
        }
        Ok(Flow::More)
    }
}

impl<'id> EchoDecoder<'id> {
    fn end(&mut self, mut out: Out<'id, Completion>, usage: Usage) -> Result<Flow, ProviderError> {
        out.end_run()?;
        Ok(out.end(Finish {
            usage: crate::completion::Usage {
                output_tokens: Some(usage.output_tokens),
                ..crate::completion::Usage::default()
            },
            model: Some("echo-1".to_owned()),
            ..Finish::default()
        }))
    }
}

impl EchoDecoder<'_> {
    fn project_snapshots(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(payload) else {
            return;
        };
        if let Some(tokens) = value
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
        {
            sink.usage_snapshot(AdapterUsage {
                output_tokens: Some(tokens),
                ..AdapterUsage::default()
            });
        }
        sink.provider(
            AdapterVerdict {
                model: Some(sink.scrub("echo-1")),
                ..AdapterVerdict::default()
            },
            None,
        );
    }

    fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        if serde_json::from_slice::<serde_json::Value>(payload).is_err() {
            return;
        }
        sink.provider(
            AdapterVerdict {
                model: Some(sink.scrub("echo-1")),
                ..AdapterVerdict::default()
            },
            None,
        );
    }
}

impl crate::completion::ReplayTarget for Echo {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("echo.chat")
    }

    fn provider(&self) -> &str {
        "echo"
    }

    fn model(&self) -> &str {
        "echo-1"
    }

    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts::ALL
    }
}

/// The unary `message` document the fake wire's frames add up to: delta
/// texts append, a stop gives the usage, a whole message or call is kept.
#[derive(Default)]
struct EchoDocument(Option<serde_json::Map<String, serde_json::Value>>);

impl crate::wire::document::Serves<Completion> for EchoDocument {}

impl crate::wire::document::Reassemble<WireFrame> for EchoDocument {
    fn absorb(&mut self, frame: &WireFrame) {
        let Ok(serde_json::Value::Object(frame)) = serde_json::from_str(&frame.as_str()) else {
            return;
        };
        let document = self.0.get_or_insert_with(|| {
            serde_json::Map::from_iter([
                ("type".to_owned(), json!("message")),
                ("text".to_owned(), json!("")),
            ])
        });
        match frame.get("type").and_then(serde_json::Value::as_str) {
            Some("message") => *document = frame,
            Some("delta") => {
                let text = frame.get("text").and_then(serde_json::Value::as_str);
                if let (Some(serde_json::Value::String(held)), Some(text)) =
                    (document.get_mut("text"), text)
                {
                    held.push_str(text);
                }
            }
            Some("stop" | "tool") => {
                for (key, value) in frame {
                    if key != "type" {
                        document.insert(key, value);
                    }
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> serde_json::Value {
        self.0
            .map_or(serde_json::Value::Null, serde_json::Value::Object)
    }
}

impl Wire for Echo {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = EchoDecoder<'id>;
    type Reassembler = EchoDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("echo")
            .model("echo-1")
            .replay(self)
            .usage_observation(self.usage_observation)
    }

    fn encode(&self, request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": request.chat_history.len(),
        }))?;
        let request =
            http::Request::post("https://echo.invalid/v1/messages").body(Body::Bytes(body))?;
        let encoded = Encoded::new(request, self.framing)
            .with_request_id_header(self.request_id_header)
            .with_captured_response_headers(self.response_header_prefix)
            .with_projection(match self.usage_observation {
                UsageObservation::FinalNormalized => EchoDecoder::project,
                UsageObservation::PayloadSnapshots => EchoDecoder::project_snapshots,
            });
        Ok(if self.relaxed_content_type {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EchoDecoder::default()
    }
}

fn prompt() -> CompletionRequest {
    CompletionRequest::new("say hi")
}

const UNARY_BODY: &str = r#"{"type":"message","text":"hi there","usage":{"output_tokens":3}}"#;
const STREAM_BODY: &str = concat!(
    "data: {\"type\":\"delta\",\"text\":\"hi \"}\n\n",
    "data: {\"type\":\"delta\",\"text\":\"there\"}\n\n",
    "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":3}}\n\n",
);

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_response() {
    let unary = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let buffered = unary.call(prompt()).await.expect("the reply decodes");

    let streaming = Model::new(
        Echo::streaming(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
        },
    );
    let mut response = streaming.stream(prompt()).expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.model(), streamed.model());
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(
        buffered.choice.first().and_then(|block| match block {
            crate::message::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
        Some("hi there")
    );
}

#[tokio::test]
async fn a_unary_reply_carries_its_body_as_raw() {
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let response = bound.call(prompt()).await.expect("the reply decodes");
    assert_eq!(
        response.raw.pointer("/text").and_then(|text| text.as_str()),
        Some("hi there")
    );
}

#[tokio::test]
async fn a_unary_reply_that_is_not_the_event_stream_it_asked_for_fails_the_call() {
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        r#"{"error":"this endpoint speaks JSON"}"#,
        "application/json",
    )]);
    let error = call(&Echo::sse_unary(), &http, prompt(), None)
        .await
        .expect_err("a JSON reply to an SSE wire is the wrong endpoint");
    assert!(
        error.to_string().contains("content type"),
        "the error names what was wrong: {error}"
    );
}

#[tokio::test]
async fn a_relaxed_wire_still_reads_a_unary_reply_that_names_no_content_type() {
    let http =
        SequencedHttpClient::new([MockHttpResponse::success(format!("data: {UNARY_BODY}\n\n"))]);
    let response = call(&Echo::sse_unary().relaxed(), &http, prompt(), None)
        .await
        .expect("the reply decodes");
    assert_eq!(text_of(&response), Some("hi there"));
}

#[tokio::test]
async fn a_unary_reply_projects_the_same_facts_a_streamed_one_does() {
    let body = format!("data: {UNARY_BODY}\n\n");
    let (unary_log, unary_context) = observed();
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        body.clone(),
        "text/event-stream",
    )]);
    call(
        &Echo::sse_unary().snapshots(),
        &http,
        prompt(),
        Some(unary_context),
    )
    .await
    .expect("the reply decodes");

    let (streamed_log, streamed_context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    let frames = stream(
        &Echo::streaming().snapshots(),
        &http,
        prompt(),
        Some(streamed_context),
    )
    .expect("the stream opens");
    let _: Vec<_> = frames.collect().await;

    let facts = |log: &ObservationLog| {
        events(log)
            .into_iter()
            .filter(|event| event != "finished")
            .collect::<Vec<_>>()
    };
    assert_eq!(facts(&unary_log), facts(&streamed_log));
    assert!(
        facts(&unary_log).contains(&"usage".to_owned()),
        "the projection ran: {:?}",
        facts(&unary_log)
    );
}

#[tokio::test]
async fn a_transport_reported_failure_preserves_status_headers_and_request_id() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    headers.insert("request-id", http::HeaderValue::from_static("req_1"));
    let http = RecordingHttpClient::with_error_headers(
        http::StatusCode::TOO_MANY_REQUESTS,
        "slow down",
        headers,
    );
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a 429 fails the call");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_request_id(), Some("req_1"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("7")
    );
}

#[tokio::test]
async fn a_non_success_response_preserves_status_headers_and_request_id() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("11"));
    headers.insert("request-id", http::HeaderValue::from_static("req_2"));
    let http = SequencedHttpClient::new([MockHttpResponse::error_with_headers(
        http::StatusCode::SERVICE_UNAVAILABLE,
        "down",
        headers,
    )]);
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a 503 fails the call");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_2"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("11")
    );
}

#[tokio::test]
async fn an_undecodable_body_fails_the_call_as_a_json_error() {
    let http = RecordingHttpClient::new("not json at all");
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a non-JSON reply fails the call");
    let ProviderError::CorruptFrame(corrupt) = &error else {
        panic!("expected a corrupt frame, got {error:?}");
    };
    assert_eq!(corrupt.frame(), Some("not json at all"));
    assert_eq!(error.kind(), crate::error::ErrorKind::Json);
}

#[tokio::test]
async fn a_reply_with_no_frames_folds_to_an_empty_response_with_no_terminal() {
    // CFDRIVER-003: zero-frame Echo refuses a call without a genuine end.
    let http = RecordingHttpClient::new("");
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("an empty body names no provider end");
    assert!(matches!(error, ProviderError::Truncated));

    // Preserve the three original response predicates on the actual unary partial.
    let mut reply = Model::new(Echo::unary(), http)
        .open(prompt(), Mode::Unary, None)
        .expect("the unary request encodes");
    while reply.next().await.is_some() {}
    let response = reply.partial();
    assert!(response.choice.is_empty());
    assert_eq!(response.finish_reason(), None);
    assert_eq!(response.usage, crate::completion::Usage::default());
    assert!(matches!(
        reply.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[tokio::test]
async fn a_connect_failure_is_the_streams_only_item() {
    let http = HttpErrorStreamingClient::new(http::StatusCode::UNAUTHORIZED, "{\"error\":\"no\"}");
    let frames = stream(&Echo::streaming(), &http, prompt(), None)
        .expect("opening the stream is not an encode error");
    let items: Vec<_> = frames.collect().await;
    assert_eq!(items.len(), 1);
    let error = items
        .into_iter()
        .next()
        .expect("one item")
        .expect_err("the item is the connect failure");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED)
    );
}

#[tokio::test]
async fn a_non_success_streaming_response_is_rejected_as_the_streams_only_item() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("13"));
    headers.insert("request-id", http::HeaderValue::from_static("req_3"));
    let http = NonSuccessStreamingClient {
        status: http::StatusCode::SERVICE_UNAVAILABLE,
        headers,
        body: Bytes::from_static(b"{\"error\":\"down\"}"),
    };
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    assert_eq!(items.len(), 1, "nothing follows the rejection: {items:?}");
    let error = items
        .into_iter()
        .next()
        .expect("one item")
        .expect_err("the item is the rejected reply");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_3"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("13")
    );
    assert!(
        error
            .provider_response_body()
            .is_some_and(|body| body.contains("down")),
        "the reply body is the error: {error:?}"
    );
}

#[tokio::test]
async fn a_transport_failure_mid_stream_ends_it_without_a_terminal() {
    let http = SequencedStreamingHttpClient::new(vec![
        Ok(Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n",
        )),
        Err(crate::http_client::Error::StreamEnded),
    ]);
    // CFDRIVER-001: provider end is proved by finish, not a Final item.
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = (&mut frames).collect().await;
    assert!(
        frames.finish().await.is_err(),
        "truncation never fabricates a terminal"
    );
    assert!(
        items.last().is_some_and(|item| item.is_err()),
        "the transport error is the last item"
    );
}

#[tokio::test]
async fn a_corrupt_frame_surfaces_in_band_and_the_stream_keeps_consuming() {
    let body = concat!(
        "data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n",
        "data: {\"type\":\"delta\"}\n\n",
        "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n",
    );
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    // CFDRIVER-001: a genuine end is separate from content-part events.
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = (&mut frames).collect().await;
    assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);
    frames
        .finish()
        .await
        .expect("a genuine terminal after a corrupt frame still completes the stream");
}

#[tokio::test]
async fn a_heartbeat_frame_never_reaches_the_decoder() {
    let body = concat!(
        ": keep-alive\n\n",
        "data:  \n\n",
        "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":0}}\n\n",
    );
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    // CFDRIVER-001: the provider end yields no content item.
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = (&mut frames).collect().await;
    assert!(items.iter().all(|item| item.is_ok()));
    assert_eq!(
        items.len(),
        0,
        "the provider end is separate from content items"
    );
    frames
        .finish()
        .await
        .expect("the genuine stop completes the reply");
}

#[tokio::test]
async fn a_streams_terminal_carries_the_transport_request_id() {
    // CFDRIVER-001: request-id metadata is on the canonical partial response.
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    // This mock reports no request-id. The original absence predicate remains
    // weak: it does not itself require a genuine provider end.
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let _: Vec<_> = (&mut frames).collect().await;
    let terminal = frames.partial();
    assert_eq!(terminal.provider_request_id.as_deref(), None);

    // New closure pin, stronger than the original optional request-ID predicate.
    frames
        .finish()
        .await
        .expect("the stream produced a genuine provider end");
}

/// A transport whose reply sends `frames`, then fails.
#[derive(Clone)]
struct Broken(Vec<&'static str>);

impl Transport<Echo> for Broken {
    fn send(&self, _payload: Encoded, _exchange: super::Exchange) -> super::Opening<WireFrame> {
        let frames = self
            .0
            .iter()
            .map(|frame| Ok(WireFrame::Text((*frame).to_owned())))
            .chain([Err(ProviderError::Response(
                "the connection dropped".to_owned(),
            ))])
            .collect::<Vec<_>>();
        super::Opening::ready(super::Opened::new(futures::stream::iter(frames)))
    }
}

#[tokio::test]
async fn a_streamed_reply_records_its_document_whole_cut_or_failed() {
    let delta = r#"{"type":"delta","text":"hi "}"#;
    let whole = MockStreamingClient {
        sse_bytes: Bytes::from(format!(
            "data: {delta}\n\ndata: {{\"type\":\"delta\",\"text\":\"there\"}}\n\n\
             data: {{\"type\":\"stop\",\"usage\":{{\"output_tokens\":3}}}}\n\n"
        )),
    };
    let response = Model::new(Echo::streaming(), whole)
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await
        .expect("the reply ends");
    assert_eq!(
        response.raw,
        serde_json::from_str::<serde_json::Value>(UNARY_BODY).expect("the unary body is JSON"),
        "the stream rebuilds the unary document"
    );

    let so_far = json!({"type": "message", "text": "hi "});
    let cut = MockStreamingClient {
        sse_bytes: Bytes::from(format!("data: {delta}\n\n")),
    };
    let mut stream = Model::new(Echo::streaming(), cut)
        .stream(prompt())
        .expect("the stream opens");
    while stream.next().await.is_some() {}
    assert_eq!(
        stream.partial().raw,
        so_far,
        "a cut stream keeps the document so far"
    );

    let mut stream = Model::new(Echo::streaming(), Broken(vec![delta]))
        .stream(prompt())
        .expect("the stream opens");
    let items: Vec<_> = (&mut stream).collect().await;
    assert!(
        items.last().is_some_and(Result::is_err),
        "the transport failure ends the stream: {items:?}"
    );
    assert_eq!(
        stream.partial().raw,
        so_far,
        "a failed stream keeps the document so far"
    );
}

#[tokio::test]
async fn a_stream_the_caller_stopped_polling_records_no_raw() {
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi \"}\n\n\
              data: {\"type\":\"delta\",\"text\":\"there\"}\n\n",
        ),
    };
    let mut stream = Model::new(Echo::streaming(), http)
        .stream(prompt())
        .expect("the stream opens");
    let _first = stream.next().await;
    assert!(stream.partial().raw.is_null());
}

fn observed() -> (Arc<ObservationLog>, AdapterContext) {
    let log = Arc::new(ObservationLog::with_capacity(64));
    let context = AdapterContext::new(log.clone(), Subject::default(), "driver-test".to_owned());
    (log, context)
}

/// The names of the adapter events a trace recorded, in order.
fn events(log: &ObservationLog) -> Vec<String> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            crate::observe::Action::Adapter { observation } => Some(
                serde_json::to_value(&observation.event)
                    .ok()?
                    .get("event")?
                    .as_str()?
                    .to_owned(),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_streams_observation_facts_arrive_in_the_old_order() {
    let (log, context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    let frames = stream(
        &Echo::streaming().snapshots(),
        &http,
        prompt(),
        Some(context),
    )
    .expect("the stream opens");
    let _: Vec<_> = frames.collect().await;
    assert_eq!(
        events(&log),
        vec![
            "started", "response",
            // Each frame's payload is projected before it is classified.
            "provider", "provider", "usage", "provider", "finished",
        ]
    );
}

#[tokio::test]
async fn a_unary_call_closes_its_attempt_as_decoded() {
    let (log, context) = observed();
    let http = RecordingHttpClient::new(UNARY_BODY);
    call(&Echo::unary().snapshots(), &http, prompt(), Some(context))
        .await
        .expect("the reply decodes");
    assert_eq!(
        events(&log),
        vec!["started", "response", "usage", "provider", "finished"]
    );
}

#[tokio::test]
async fn a_malformed_tool_input_is_kept_with_its_raw_text() {
    let http = RecordingHttpClient::new(
        r#"{"type":"tool","name":"add","arguments":"{not json","usage":{"output_tokens":1}}"#,
    );
    let response = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect("a malformed call does not fail the reply");
    let call_of = |response: &crate::completion::CompletionResponse| {
        response
            .tool_calls()
            .map(|call| {
                (
                    call.function.arguments_value(),
                    call.function.invalid_arguments.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        call_of(&response),
        [(serde_json::json!({}), Some("{not json".to_owned()))]
    );

    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"tool\",\"name\":\"add\",\"arguments\":\"{not json\"}\n\n\
              data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n",
        ),
    };
    let streamed = Model::new(Echo::streaming(), http)
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await
        .expect("a malformed call does not fail the stream");
    assert_eq!(call_of(&streamed), call_of(&response));
}

#[tokio::test]
async fn a_failed_unary_call_projects_the_reply_it_failed_on() {
    // F payload snapshots and U normalized finish are explicit separate profiles.
    {
        let (log, context) = observed();
        let http = RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            r#"{"usage":{"output_tokens":0}}"#,
        );
        call(&Echo::unary().snapshots(), &http, prompt(), Some(context))
            .await
            .expect_err("a 400 fails the call");
        let events = events(&log);
        assert!(
            events.contains(&"usage".to_owned()),
            "the failed reply's facts are still projected: {events:?}"
        );
        assert_eq!(events.last().map(String::as_str), Some("finished"));
    }
    {
        let (log, context) = observed();
        let http = RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            r#"{"usage":{"output_tokens":0}}"#,
        );
        call(&Echo::unary(), &http, prompt(), Some(context))
            .await
            .expect_err("a 400 fails the call");
        let events = events(&log);
        assert!(
            events.contains(&"provider".to_owned()),
            "the failed reply's facts are still projected: {events:?}"
        );
        assert_eq!(events.last().map(String::as_str), Some("finished"));
    }
}

#[tokio::test]
async fn a_reply_observes_its_finish_usage_once() {
    let streamed = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n\
              data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":3}}\n\n",
        ),
    };
    for (observed, usage) in [
        crate::driver::tests::observed_usage(
            Echo::unary(),
            RecordingHttpClient::new(UNARY_BODY),
            Mode::Unary,
        )
        .await,
        crate::driver::tests::observed_usage(Echo::streaming(), streamed, Mode::Streaming).await,
    ] {
        let expected = crate::observe::AdapterUsage {
            output_tokens: Some(3),
            ..Default::default()
        };
        assert_eq!(observed, std::slice::from_ref(&expected));
        assert_eq!(expected, crate::observe::AdapterUsage::from(&usage));
    }
}

#[tokio::test]
async fn a_failed_stream_observes_no_usage() {
    use crate::observe::{Action, AdapterEnding, AdapterEvent};
    // Each frame carries usage the old projector would have reported.
    let usage_frame =
        "data: {\"type\":\"delta\",\"text\":\"hi\",\"usage\":{\"output_tokens\":3}}\n\n";
    let in_band =
        format!("{usage_frame}data: {{\"type\":\"error\",\"message\":\"overloaded\"}}\n\n");
    for (sse, ended) in [
        (
            in_band,
            (|ending: &AdapterEnding| matches!(ending, AdapterEnding::Error { .. }))
                as fn(&AdapterEnding) -> bool,
        ),
        (usage_frame.to_owned(), |ending| {
            matches!(ending, AdapterEnding::Eof { after: 1 })
        }),
    ] {
        let (log, context) = observed();
        let http = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
        };
        let result = match Model::new(Echo::streaming(), http).stream_observed(prompt(), context) {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        };
        assert!(result.is_err(), "the stream fails");
        let adapter: Vec<AdapterEvent> = log
            .trace()
            .observations
            .iter()
            .filter_map(|observation| match &observation.action {
                Action::Adapter { observation } => Some(observation.event.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !adapter
                .iter()
                .any(|event| matches!(event, AdapterEvent::Usage { .. })),
            "a failed stream observes no usage: {adapter:?}"
        );
        let endings: Vec<&AdapterEnding> = adapter
            .iter()
            .filter_map(|event| match event {
                AdapterEvent::Finished { ending } => Some(ending),
                _ => None,
            })
            .collect();
        assert!(
            matches!(endings.as_slice(), [ending] if ended(ending)),
            "the attempt closes once, on the failure: {endings:?}"
        );
    }
}

/// The same fake model listing, one canonical ModelPage per reply.
#[derive(Clone, Debug, PartialEq)]
struct Catalogue;

#[derive(Default)]
struct CatalogueDecoder;

#[derive(serde::Deserialize)]
struct Page {
    data: Vec<String>,
    #[serde(default)]
    next: Option<String>,
}

impl<'id> Decoder<'id, ModelListing> for CatalogueDecoder {
    type Event = Page;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_untyped_line(frame.as_str().as_bytes())
    }

    fn decode(&mut self, page: Page, out: Out<'id, ModelListing>) -> Result<Flow, ProviderError> {
        Ok(out.end(ModelPage {
            models: ModelList::new(page.data.into_iter().map(ModelInfo::from_id).collect()),
            next: page.next,
        }))
    }
}

impl Wire for Catalogue {
    type Op = ModelListing;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = CatalogueDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("echo")
    }

    fn encode(&self, request: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let uri = match request {
            Some(cursor) => format!("https://echo.invalid/v1/models?after={cursor}"),
            None => "https://echo.invalid/v1/models".to_owned(),
        };
        let request = http::Request::get(uri).body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CatalogueDecoder
    }
}

#[tokio::test]
async fn a_paged_listing_follows_every_continuation() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a","b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["c"]}"#),
    ]);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("both pages decode");
    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
    let paths: Vec<_> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert_eq!(
        paths,
        vec![
            "https://echo.invalid/v1/models".to_owned(),
            "https://echo.invalid/v1/models?after=b".to_owned(),
        ]
    );
}

#[tokio::test]
async fn a_listing_that_repeats_its_cursor_stops_after_the_repeated_page() {
    // The second page names the cursor that fetched it, so the next request
    // would be identical to the one just answered (rig#2334).
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["never"]}"#),
    ]);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("the fetched pages decode");
    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        http.requests().len(),
        2,
        "the repeated cursor is not re-sent"
    );
}

#[tokio::test]
async fn a_listing_whose_cursor_keeps_changing_stops_at_the_page_ceiling() {
    let pages = (0..super::MAX_CONTINUATION_PAGES + 5).map(|page| {
        MockHttpResponse::success(format!(r#"{{"data":["m{page}"],"next":"c{}"}}"#, page + 1))
    });
    let http = SequencedHttpClient::new(pages);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("the fetched pages decode");
    assert_eq!(models.len(), super::MAX_CONTINUATION_PAGES);
    assert_eq!(http.requests().len(), super::MAX_CONTINUATION_PAGES);
}

#[test]
fn the_completion_operation_names_its_span_by_mode() {
    // CFDRIVER-004: the actual default is Chat in both modes; streaming is a flag.
    // The removed telemetry(bool)/ChatStreaming equality is explicitly changed.
    use tracing::subscriber::with_default;

    for streaming in [false, true] {
        let capture = TraceCapture::default();
        let wire = if streaming {
            Echo::streaming()
        } else {
            Echo::unary()
        };
        let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
            if streaming { STREAM_BODY } else { UNARY_BODY },
            if streaming {
                "text/event-stream"
            } else {
                "application/json"
            },
        )]);
        let model = Model::new(wire, http);
        with_default(capture.subscriber(), || {
            if streaming {
                futures::executor::block_on(
                    model.stream(prompt()).expect("the stream opens").finish(),
                )
            } else {
                futures::executor::block_on(model.call(prompt()))
            }
        })
        .expect("the reply folds");
        assert_eq!(
            capture.values_of("gen_ai.operation.name"),
            vec![json!("chat")]
        );
        assert_eq!(
            capture.values_of("gen_ai.request.stream"),
            vec![json!(streaming)]
        );
    }
}

#[tokio::test]
async fn the_driver_records_the_folded_responses_metadata() {
    // Both original telemetry observers retain their exact span predicates.
    {
        use tracing::subscriber::with_default;
        use tracing_subscriber::layer::SubscriberExt;

        let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
        let layer = RecordFields {
            recorded: recorded.clone(),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
        let response = with_default(subscriber, || {
            futures::executor::block_on(bound.call(prompt()))
        })
        .expect("the reply decodes");
        assert_eq!(response.model(), Some("echo-1"));
        let recorded = recorded.lock().expect("no panic held the lock").clone();
        assert!(
            recorded
                .iter()
                .any(|(field, value)| field == "gen_ai.response.model" && value == "echo-1"),
            "expected the response model on the span: {recorded:?}"
        );
        assert!(
            recorded
                .iter()
                .any(|(field, _)| field == "gen_ai.usage.output_tokens"),
            "expected usage on the span: {recorded:?}"
        );
    }
    {
        use tracing::subscriber::with_default;

        let capture = TraceCapture::default();
        let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
        let response = with_default(capture.subscriber(), || {
            futures::executor::block_on(bound.call(prompt()))
        })
        .expect("the reply decodes");
        assert_eq!(response.model(), Some("echo-1"));
        assert!(
            capture
                .values_of("gen_ai.response.model")
                .contains(&json!("echo-1")),
            "expected the response model on the span"
        );
        assert!(
            !capture.values_of("gen_ai.usage.output_tokens").is_empty(),
            "expected usage on the span"
        );
    }
}

#[test]
fn a_streamed_embedding_records_its_response_on_the_span() {
    use tracing::subscriber::with_default;

    const REPLY: &str = r#"{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#;
    let capture = TraceCapture::default();
    let wire = crate::providers::openai::wire::OpenAIConfig::new("sk-test")
        .embedding("text-embedding-3-small", None);
    let model = Model::new(
        wire,
        SequencedHttpClient::new([MockHttpResponse::success(REPLY)]),
    );
    with_default(capture.subscriber(), || {
        let stream = model
            .stream(vec!["a".to_owned()])
            .expect("the embedding opens");
        futures::executor::block_on(stream.finish()).expect("the reply folds")
    });
    assert!(
        capture
            .values_of("gen_ai.response.model")
            .contains(&json!("text-embedding-3-small")),
        "expected the response model on the span"
    );
}

#[tokio::test]
async fn the_span_names_the_requests_model_override_not_the_wires() {
    // CFDRIVER-004: keep the exact override predicate on the actual span.
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::SubscriberExt;

    let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let layer = RecordFields {
        recorded: recorded.clone(),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let request = CompletionRequest {
        model: Some("echo-override".to_owned()),
        ..prompt()
    };
    with_default(subscriber, || {
        futures::executor::block_on(bound.call(request))
    })
    .expect("the reply decodes");
    let recorded = recorded.lock().expect("no panic held the lock").clone();
    assert!(
        recorded
            .iter()
            .any(|(field, value)| field == "gen_ai.request.model" && value == "echo-override"),
        "expected the override on the span: {recorded:?}"
    );
}

/// Captures every field recorded on a span, so the driver's telemetry is
/// asserted through `tracing` rather than through its own call sites.
struct RecordFields {
    recorded: Arc<std::sync::Mutex<Vec<(String, String)>>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RecordFields {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        attrs.record(&mut Visit(&mut recorded));
    }

    fn on_record(
        &self,
        _id: &tracing::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        values.record(&mut Visit(&mut recorded));
    }
}

struct Visit<'a>(&'a mut Vec<(String, String)>);
impl tracing::field::Visit for Visit<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_owned(), value.to_owned()));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.push((field.name().to_owned(), value.to_string()));
    }
}

/// The text a folded response carries, for the assertions below.
fn text_of(response: &crate::completion::CompletionResponse) -> Option<&str> {
    response.choice.first().and_then(|block| match block {
        crate::message::AssistantContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    })
}

/// The existing fake mode guard, expressed through the published EOF policy.
#[derive(Clone, Debug, PartialEq)]
struct Guarded(Framing);

struct GuardedDecoder {
    whole: bool,
}

impl<'id> Decoder<'id, Completion> for GuardedDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<()> {
        WireEvent::Known(())
    }

    fn decode(&mut self, _event: (), _out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Ok(Flow::More)
    }

    fn eof(&mut self, _out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        if self.whole {
            Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ))
        } else {
            Ok(Flow::More)
        }
    }

    fn incomplete_eof(&self) -> IncompleteEof {
        if self.whole {
            IncompleteEof::Refuse
        } else {
            IncompleteEof::KeepPartial
        }
    }
}

impl crate::completion::ReplayTarget for Guarded {
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("guarded")
    }

    fn provider(&self) -> &str {
        "guarded"
    }
    fn model(&self) -> &str {
        ""
    }
    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts::ALL
    }
}

impl Wire for Guarded {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = GuardedDecoder;
    type Reassembler = EchoDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("guarded").replay(self)
    }

    fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::post("https://echo.invalid/v1/messages")
            .body(Body::Bytes(Vec::new()))?;
        Ok(Encoded::new(request, self.0))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        GuardedDecoder { whole: false }
    }

    fn decoder_in<'id>(&self, mode: Mode) -> Self::Decoder<'id> {
        GuardedDecoder {
            whole: mode == Mode::Unary,
        }
    }
}

#[tokio::test]
async fn the_mode_a_decoder_is_built_for_decides_what_its_eof_means() {
    // CFDRIVER-003: only Guarded uses the explicit silent partial streaming EOF.
    let unary = Model::new(Guarded(Framing::Whole), RecordingHttpClient::new("{}"));
    let error = unary
        .call(prompt())
        .await
        .expect_err("a whole reply that delivered nothing is not an answer");
    assert!(
        matches!(&error, ProviderError::Response(message)
            if message == crate::message::EMPTY_RESPONSE_ERROR),
        "expected the empty-reply error, got {error:?}"
    );

    let streaming = Model::new(
        Guarded(Framing::Sse),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(b"data: {}\n\n"),
        },
    );
    let mut response = streaming.stream(prompt()).expect("the stream opens");
    let mut errors = Vec::new();
    while let Some(item) = response.next().await {
        if let Err(error) = item {
            errors.push(error);
        }
    }
    assert!(
        errors.is_empty(),
        "a streamed reply's EOF is truncation — reported by carrying no \
         terminal record, not by an error: {errors:?}"
    );
    // New end-custody pin: the silent EOF never fabricates a provider end.
    assert!(matches!(
        response.finish().await,
        Err(ProviderError::Truncated)
    ));
}

#[test]
fn a_stream_the_driver_cannot_send_is_a_request_failure() {
    // CFDRIVER-001: only the unrepresentable Encoded.requests Batch branch is retired.
    // CFDRIVER-002: Multipart fails as the sole in-band item, with no HTTP send.
    #[derive(Clone, Debug)]
    struct Multipart;
    impl crate::completion::ReplayTarget for Multipart {
        fn map_options(
            &self,
            _request: &crate::completion::CompletionRequest,
            fields: crate::completion::options::OptionFields<'_>,
        ) -> crate::completion::options::OptionMap {
            crate::test_utils::refuse_options(fields)
        }

        fn api(&self) -> crate::message::Api {
            crate::message::Api::from_static("echo.chat")
        }
        fn provider(&self) -> &str {
            "echo"
        }
        fn model(&self) -> &str {
            ""
        }

        fn accepts(&self, _model: &str) -> crate::completion::Accepts {
            crate::completion::Accepts::ALL
        }
    }
    impl Wire for Multipart {
        type Op = Completion;
        type Payload = Encoded;
        type Frame = WireFrame;
        type Decoder<'id> = EchoDecoder<'id>;
        type Reassembler = EchoDocument;
        fn describe(&self) -> Descriptor<'_> {
            Descriptor::new("echo").replay(self)
        }
        fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
            let request = http::Request::post("https://echo.invalid/a")
                .body(Body::Multipart(crate::http_client::MultipartForm::new()))?;
            Ok(Encoded::new(request, Framing::Sse))
        }
        fn decoder<'id>(&self) -> Self::Decoder<'id> {
            EchoDecoder::default()
        }
    }

    let http = crate::test_utils::RecordingHttpClient::new("");
    let items: Vec<_> = futures::executor::block_on(
        stream(&Multipart, &http, prompt(), None)
            .expect("the request encodes")
            .collect::<Vec<_>>(),
    );
    let [Err(error)] = items.as_slice() else {
        panic!("the failure is the stream's only item: {items:?}");
    };
    assert_eq!(
        error.to_string(),
        "RequestError: a multipart request cannot open a streamed reply"
    );
    assert_eq!(error.kind(), crate::error::ErrorKind::Request);
    assert_eq!(
        error.boundary(),
        crate::observe::AdapterErrorBoundary::Request
    );
    assert!(!error.is_retryable());
    assert!(http.requests().is_empty(), "nothing was sent");
}

/// A successful reply's headers: three `x-codex-*` ones — one repeated, one
/// not valid UTF-8 — beside headers no dialect captures here.
fn codex_reply_headers(content_type: &'static str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static(content_type),
    );
    headers.insert("request-id", http::HeaderValue::from_static("req_9"));
    headers.insert(
        "x-codex-primary-used-percent",
        http::HeaderValue::from_static("12.5"),
    );
    headers.append(
        "x-codex-credits-balance",
        http::HeaderValue::from_static("10"),
    );
    headers.append(
        "x-codex-credits-balance",
        http::HeaderValue::from_static("20"),
    );
    headers.insert(
        "x-codex-promo-message",
        http::HeaderValue::from_bytes(b"caf\xe9").expect("an opaque header value"),
    );
    headers
}

/// What a dialect capturing `x-codex-` keeps from [`codex_reply_headers`]:
/// only those headers, a repeated one joined in arrival order, and a value
/// that is not UTF-8 converted lossily.
fn captured_codex_headers() -> crate::completion::ProviderResponseHeaders {
    [
        ("x-codex-credits-balance", "10, 20"),
        ("x-codex-primary-used-percent", "12.5"),
        ("x-codex-promo-message", "caf\u{FFFD}"),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .collect()
}

#[tokio::test]
async fn a_unary_reply_keeps_the_success_headers_its_dialect_captures() {
    let http = RecordingHttpClient::new(UNARY_BODY);
    http.set_response(MockHttpResponse::SuccessWithHeaders(
        Bytes::from_static(UNARY_BODY.as_bytes()),
        codex_reply_headers("application/json"),
    ));
    let bound = Model::new(Echo::unary().capturing("x-codex-"), http.clone());
    let response = bound.call(prompt()).await.expect("the reply decodes");
    assert_eq!(response.provider_response_headers, captured_codex_headers());

    // A wire that captures nothing keeps nothing from the same reply.
    let plain = Model::new(Echo::unary(), http);
    let response = plain.call(prompt()).await.expect("the reply decodes");
    assert!(response.provider_response_headers.is_empty());
}

#[tokio::test]
async fn a_streamed_reply_keeps_the_success_headers_its_dialect_captures() {
    // CFDRIVER-001: successful finish exposes the original exact header map.
    let http = NonSuccessStreamingClient {
        status: http::StatusCode::OK,
        headers: codex_reply_headers("text/event-stream"),
        body: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    let frames = stream(
        &Echo::streaming().capturing("x-codex-"),
        &http,
        prompt(),
        None,
    )
    .expect("the stream opens");
    let terminal = frames
        .finish()
        .await
        .expect("the stream ends with a terminal record");
    assert_eq!(terminal.provider_response_headers, captured_codex_headers());

    // The same genuine provider end carries no captured headers without a prefix.
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let terminal = frames
        .finish()
        .await
        .expect("the stream ends with a terminal record");
    assert!(terminal.provider_response_headers.is_empty());
}
