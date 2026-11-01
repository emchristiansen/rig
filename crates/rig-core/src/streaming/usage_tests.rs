//! Usage observation through the real HTTP transport and branded completion driver.
//! The fixture protocol is local; it establishes no provider-dialect behavior.

use crate::completion::{Accepts, CompletionRequest, FinishReason, ReplayTarget, Usage};
use crate::driver::Model;
use crate::error::{EncodeError, ProviderError};
use crate::message::Api;
use crate::observe::{
    Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, ObservationTrace, Subject,
};
use crate::operation::{Block, Completion, Finish};
use crate::test_utils::MockStreamingClient;
use crate::wire::document::{Reassemble, Serves};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, IncompleteEof, Mode, ObservationSink, Out,
    UsageObservation, Wire, WireEvent, WireFrame,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct UsageWire {
    policy: UsageObservation,
    keep_partial: bool,
}

impl ReplayTarget for UsageWire {
    fn api(&self) -> Api {
        Api::from_static("usage-fixture.chat")
    }
    fn provider(&self) -> &str {
        "usage-fixture"
    }
    fn model(&self) -> &str {
        "usage-fixture-model"
    }
    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }
}

impl Wire for UsageWire {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = UsageDecoder<'id>;
    type Reassembler = UsageDocument;

    fn describe(&self) -> Descriptor<'_> {
        let descriptor = Descriptor::new(self.provider())
            .model(self.model())
            .replay(self);
        match self.policy {
            UsageObservation::FinalNormalized => descriptor,
            UsageObservation::PayloadSnapshots => descriptor.usage_observation(self.policy),
        }
    }
    fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::builder()
            .method(http::Method::POST)
            .uri("https://usage.invalid/completion")
            .body(Body::empty())?;
        let project = match self.policy {
            UsageObservation::FinalNormalized => project_verdicts,
            UsageObservation::PayloadSnapshots => project_snapshots,
        };
        Ok(Encoded::new(request, Framing::Sse)
            .with_route(Some("/completion"))
            .with_projection(project))
    }
    fn decoder<'id>(&self) -> UsageDecoder<'id> {
        UsageDecoder {
            keep_partial: self.keep_partial,
            brand: PhantomData,
        }
    }
}

fn project_verdicts(payload: &[u8], sink: &mut ObservationSink<'_>) {
    let doc: Value = serde_json::from_slice(payload).expect("a complete fixture payload");
    project_metadata(&doc, sink);
}

fn project_snapshots(payload: &[u8], sink: &mut ObservationSink<'_>) {
    let doc: Value = serde_json::from_slice(payload).expect("a complete fixture payload");
    if let Some(snapshot) = doc.get("snapshot") {
        sink.usage_snapshot(serde_json::from_value(snapshot.clone()).expect("typed fixture usage"));
    }
    project_metadata(&doc, sink);
}

fn project_metadata(doc: &Value, sink: &mut ObservationSink<'_>) {
    if let Some(finish) = doc.get("finish").and_then(Value::as_str) {
        sink.provider(
            AdapterVerdict {
                finish_reason: Some(finish.to_owned()),
                ..AdapterVerdict::default()
            },
            None,
        );
    }
    if doc.get("error").is_some() {
        sink.error(AdapterErrorEnvelope {
            code: Some("fixture-error".to_owned()),
            status: None,
            message: Some("fixture failure".to_owned()),
        });
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    Snapshot { text: String },
    Stop { normalized: Usage },
    Error { error: Value },
}

struct UsageDecoder<'id> {
    keep_partial: bool,
    brand: PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion> for UsageDecoder<'id> {
    type Event = Frame;
    fn classify(&self, frame: WireFrame) -> WireEvent<Frame> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "snapshot" | "stop" | "error")
        })
    }
    fn decode(
        &mut self,
        event: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            Frame::Snapshot { text } => {
                out.run(Block::Text, &text)?;
                Ok(Flow::More)
            }
            Frame::Stop { normalized } => {
                out.end_run()?;
                Ok(out.end(Finish {
                    usage: normalized,
                    reason: Some(FinishReason::Stop),
                    ..Finish::default()
                }))
            }
            Frame::Error { error } => {
                out.end_run()?;
                Err(ProviderError::from_provider_body(
                    json!({"error":error}).to_string(),
                ))
            }
        }
    }
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        out.end_run()?;
        Ok(Flow::More)
    }
    fn incomplete_eof(&self) -> IncompleteEof {
        if self.keep_partial {
            IncompleteEof::KeepPartial
        } else {
            IncompleteEof::Refuse
        }
    }
}

#[derive(Default)]
struct UsageDocument(Vec<Value>);
impl Serves<Completion> for UsageDocument {}
impl Reassemble<WireFrame> for UsageDocument {
    fn absorb(&mut self, frame: &WireFrame) {
        self.0
            .push(serde_json::from_str(&frame.as_str()).expect("fixture frame is JSON"));
    }
    fn finish(self) -> Value {
        Value::Array(self.0)
    }
}

fn first_snapshot() -> AdapterUsage {
    AdapterUsage {
        input_tokens: Some(11),
        output_tokens: Some(0),
        total_tokens: None,
        cached_input_tokens: Some(3),
        reasoning_tokens: None,
        tool_input_tokens: Some(0),
    }
}
fn second_snapshot() -> AdapterUsage {
    AdapterUsage {
        input_tokens: None,
        output_tokens: Some(17),
        total_tokens: None,
        cached_input_tokens: None,
        reasoning_tokens: Some(5),
        tool_input_tokens: None,
    }
}
fn normalized_usage() -> Usage {
    Usage {
        input_tokens: Some(100),
        output_tokens: Some(999),
        total_tokens: Some(1099),
        cached_input_tokens: Some(70),
        reasoning_tokens: Some(80),
        tool_use_prompt_tokens: Some(90),
        ..Usage::default()
    }
}
fn first_payload() -> Value {
    json!({"type":"snapshot", "snapshot":first_snapshot(), "text":"kept", "finish":"WORKING"})
}
fn stop_payload() -> Value {
    json!({"type":"stop", "snapshot":second_snapshot(), "normalized":normalized_usage(), "finish":"STOP"})
}
fn event(payload: Value) -> String {
    format!("data: {payload}\n\n")
}
fn model(
    policy: UsageObservation,
    keep_partial: bool,
    body: String,
) -> Model<UsageWire, MockStreamingClient> {
    Model::new(
        UsageWire {
            policy,
            keep_partial,
        },
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(body),
        },
    )
}
fn context(log: Arc<ObservationLog>) -> AdapterContext {
    AdapterContext::new(log, Subject::scoped("usage-stream"), "usage-operation")
}
fn events(trace: &ObservationTrace) -> Vec<AdapterEvent> {
    trace
        .observations
        .iter()
        .map(|fact| {
            assert_eq!(fact.subject, Subject::scoped("usage-stream"));
            let Action::Adapter { observation } = &fact.action else {
                panic!("adapter fact");
            };
            assert_eq!(observation.operation, "usage-operation");
            assert_eq!(observation.attempt, Some(1));
            observation.event.clone()
        })
        .collect()
}
fn usages(events: &[AdapterEvent]) -> Vec<AdapterUsage> {
    events
        .iter()
        .filter_map(|event| match event {
            AdapterEvent::Usage { usage } => Some(usage.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn usage_policy_defaults_to_the_existing_normalized_end_and_opt_in_is_explicit() {
    assert_eq!(
        UsageObservation::default(),
        UsageObservation::FinalNormalized
    );
    let descriptor = Descriptor::new("existing-wire").model("existing-model");
    assert_eq!(
        descriptor.usage_observation,
        UsageObservation::FinalNormalized
    );
    let opted_in = descriptor
        .clone()
        .usage_observation(UsageObservation::PayloadSnapshots);
    assert_eq!(
        opted_in.usage_observation,
        UsageObservation::PayloadSnapshots
    );
    assert_eq!(opted_in.name, descriptor.name);
    assert_eq!(opted_in.model, descriptor.model);
    assert_eq!(
        UsageWire {
            policy: UsageObservation::FinalNormalized,
            keep_partial: false
        }
        .describe()
        .usage_observation,
        UsageObservation::FinalNormalized
    );
}

#[tokio::test]
async fn opted_in_payload_snapshots_survive_actual_projection_and_do_not_duplicate_normalized_end()
{
    let body = format!("{}{}", event(first_payload()), event(stop_payload()));
    let model = model(UsageObservation::PayloadSnapshots, false, body);
    let log = Arc::new(ObservationLog::default());
    let mut stream = model
        .stream_observed(CompletionRequest::new("usage test"), context(log.clone()))
        .expect("prepare fixture request");
    assert!(
        log.is_empty(),
        "the unpolled HTTP transport has not observed a send"
    );
    while let Some(item) = stream.next().await {
        item.expect("actual decoder event");
    }
    let response = stream.finish().await.expect("provider terminal is real");
    assert_eq!(response.text(), "kept");
    assert_eq!(
        response.usage,
        normalized_usage(),
        "observation policy changes no returned usage"
    );
    let recorded = events(&log.trace());
    assert_eq!(
        recorded,
        vec![
            AdapterEvent::Started {
                method: "POST".to_owned(),
                route: "/completion".to_owned()
            },
            AdapterEvent::Response { status: 200 },
            AdapterEvent::Usage {
                usage: first_snapshot()
            },
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("WORKING".to_owned()),
                    ..AdapterVerdict::default()
                }
            },
            AdapterEvent::Usage {
                usage: second_snapshot()
            },
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("STOP".to_owned()),
                    ..AdapterVerdict::default()
                }
            },
            AdapterEvent::Finished {
                ending: AdapterEnding::Terminal
            },
        ]
    );
    assert_eq!(usages(&recorded), [first_snapshot(), second_snapshot()]);
    assert!(!usages(&recorded).contains(&AdapterUsage::from(&normalized_usage())));
}

#[tokio::test]
async fn unchanged_wire_emits_exact_final_normalized_usage_once_without_payload_snapshots() {
    let body = format!("{}{}", event(first_payload()), event(stop_payload()));
    let model = model(UsageObservation::FinalNormalized, false, body);
    let log = Arc::new(ObservationLog::default());
    let response = model
        .stream_observed(CompletionRequest::new("usage test"), context(log.clone()))
        .expect("prepare fixture request")
        .finish()
        .await
        .expect("actual normalized reply");
    assert_eq!(response.usage, normalized_usage());
    assert_eq!(response.text(), "kept");
    let recorded = events(&log.trace());
    let normalized = AdapterUsage::from(&normalized_usage());
    assert_eq!(
        normalized,
        AdapterUsage {
            input_tokens: Some(100),
            output_tokens: Some(999),
            total_tokens: None,
            cached_input_tokens: Some(70),
            reasoning_tokens: Some(80),
            tool_input_tokens: Some(90)
        }
    );
    assert_eq!(
        recorded,
        vec![
            AdapterEvent::Started {
                method: "POST".to_owned(),
                route: "/completion".to_owned()
            },
            AdapterEvent::Response { status: 200 },
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("WORKING".to_owned()),
                    ..AdapterVerdict::default()
                }
            },
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("STOP".to_owned()),
                    ..AdapterVerdict::default()
                }
            },
            AdapterEvent::Usage {
                usage: normalized.clone()
            },
            AdapterEvent::Finished {
                ending: AdapterEnding::Terminal
            },
        ]
    );
    assert_eq!(usages(&recorded), [normalized]);
}

#[tokio::test]
async fn snapshots_survive_actual_error_eof_partial_eof_and_cancel_without_inventing_terminal_usage()
 {
    for (suffix, keep_partial, cancel, ending) in [
        (
            event(
                json!({"type":"error", "snapshot":second_snapshot(), "error":{"code":"fixture-error", "message":"fixture failure"}}),
            ),
            false,
            false,
            AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".to_owned(),
                status: None,
                retryable: false,
            },
        ),
        (String::new(), false, false, AdapterEnding::Eof { after: 1 }),
        (
            "data: partial".to_owned(),
            false,
            false,
            AdapterEnding::PartialFrame {
                byte_count: 13,
                after: 1,
            },
        ),
        (
            "data: partial".to_owned(),
            true,
            false,
            AdapterEnding::PartialFrame {
                byte_count: 13,
                after: 1,
            },
        ),
        (event(stop_payload()), false, true, AdapterEnding::Dropped),
    ] {
        let body = format!("{}{suffix}", event(first_payload()));
        let model = model(UsageObservation::PayloadSnapshots, keep_partial, body);
        let log = Arc::new(ObservationLog::default());
        let mut stream = model
            .stream_observed(CompletionRequest::new("usage test"), context(log.clone()))
            .expect("prepare fixture request");
        if cancel {
            stream
                .next()
                .await
                .expect("first projected payload yields a start")
                .expect("valid start");
            assert_eq!(usages(&events(&log.trace())), [first_snapshot()]);
            stream.cancel();
        } else {
            while stream.next().await.is_some() {}
        }
        let partial = stream.partial();
        assert_eq!(
            partial.usage,
            Usage::default(),
            "no provider end means no normalized usage was invented"
        );
        assert!(partial.finish_reason().is_none());
        if !cancel {
            assert_eq!(partial.text(), "kept");
        }
        let error = stream
            .finish()
            .await
            .expect_err("the attempt has no genuine provider end");
        match &ending {
            AdapterEnding::Error { .. } => assert!(error.provider_response().is_some()),
            _ => assert!(matches!(error, ProviderError::Truncated)),
        }
        let recorded = events(&log.trace());
        let expected = if matches!(ending, AdapterEnding::Error { .. }) {
            vec![first_snapshot(), second_snapshot()]
        } else {
            vec![first_snapshot()]
        };
        assert_eq!(usages(&recorded), expected);
        assert_eq!(
            recorded.last(),
            Some(&AdapterEvent::Finished {
                ending: ending.clone()
            })
        );
        assert_eq!(
            recorded
                .iter()
                .filter(|event| matches!(event, AdapterEvent::Finished { .. }))
                .count(),
            1
        );
        assert!(!recorded.iter().any(|event| matches!(
            event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Terminal
            }
        )));
        if let Some(at) = recorded
            .iter()
            .position(|event| matches!(event, AdapterEvent::ErrorEnvelope { .. }))
        {
            assert_eq!(
                recorded[at - 1],
                AdapterEvent::Usage {
                    usage: second_snapshot()
                }
            );
        }
        if let AdapterEnding::PartialFrame { byte_count, after } = ending {
            assert!(recorded.contains(&AdapterEvent::TransportEof {
                after,
                partial_bytes: byte_count
            }));
        }
    }
}
