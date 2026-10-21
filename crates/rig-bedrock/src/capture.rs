//! The HTTP bodies of one Converse call, at the SDK's boundary: the request
//! body this crate built goes out in place of the SDK's serialization, and
//! a successful reply's body is read here rather than by the SDK, so a
//! reply is the JSON Bedrock sent rather than a re-serialization of the
//! SDK's types. A unary body is one JSON document; a stream's body is
//! event-stream messages, each carrying one event's or exception's JSON.

use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_runtime_api::box_error::BoxError;
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::{
    BeforeDeserializationInterceptorContextMut, BeforeTransmitInterceptorContextMut,
};
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::config_bag::ConfigBag;
use aws_smithy_types::event_stream::Message;
use bytes::{Bytes, BytesMut};
use http_body::Body;
use rig_core::error::{CorruptFrame, ProviderError};
use serde_json::Value;

/// An interceptor that sends `body` and takes a successful reply's body
/// from the SDK.
#[derive(Clone, Debug)]
pub(crate) struct Capture {
    body: Bytes,
    reply: Arc<Mutex<Option<SdkBody>>>,
}

impl Capture {
    /// A capture that sends `body` as the request's JSON.
    pub(crate) fn new(body: Vec<u8>) -> Self {
        Self {
            body: Bytes::from(body),
            reply: Arc::default(),
        }
    }

    /// The successful reply's body, once.
    pub(crate) fn reply(&self) -> Option<SdkBody> {
        self.reply
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}

/// The next bytes of `body`, as they arrive.
pub(crate) async fn chunk(body: &mut SdkBody) -> Option<Result<Bytes, ProviderError>> {
    let frame = std::future::poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await?;
    Some(match frame {
        Ok(frame) => Ok(frame.into_data().unwrap_or_default()),
        Err(error) => Err(ProviderError::Http(
            rig_core::http_client::Error::instance(std::io::Error::other(error)).into(),
        )),
    })
}

impl Intercept for Capture {
    fn name(&self) -> &'static str {
        "rig_bedrock::Capture"
    }

    /// Before the first attempt, so every attempt signs and sends this body.
    fn modify_before_retry_loop(
        &self,
        context: &mut BeforeTransmitInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        let request = context.request_mut();
        request
            .headers_mut()
            .insert("content-length", self.body.len().to_string());
        *request.body_mut() = SdkBody::from(self.body.clone());
        Ok(())
    }

    /// A success's body is this crate's to read; the SDK reads an error's.
    fn modify_before_deserialization(
        &self,
        context: &mut BeforeDeserializationInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        let response = context.response_mut();
        if response.status().is_success() {
            let body = std::mem::replace(response.body_mut(), SdkBody::empty());
            *self.reply.lock().unwrap_or_else(PoisonError::into_inner) = Some(body);
        }
        Ok(())
    }
}

/// Reads a stream's bytes into its messages' JSON.
#[derive(Default)]
pub(crate) struct Events {
    buffer: BytesMut,
    received: BytesMut,
    decoder: MessageFrameDecoder,
    stopped: bool,
}

impl Events {
    /// The JSON of each event or exception `bytes` completes, as
    /// `{"<type>": <payload>}`.
    pub(crate) fn read(&mut self, bytes: &[u8]) -> Vec<Result<Value, CorruptFrame>> {
        if self.stopped {
            return Vec::new();
        }
        self.buffer.extend_from_slice(bytes);
        self.received.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            // The SDK retains a consumed prelude across incomplete reads.
            // Keep the received frame independently before it consumes bytes.
            let received = self.received_frame();
            match self.decoder.decode_frame(&mut self.buffer) {
                Ok(DecodedFrame::Complete(message)) => {
                    let consumed = self.received.len() - self.buffer.len();
                    let _ = self.received.split_to(consumed);
                    match event(&message) {
                        Ok(Some(event)) => events.push(Ok(event)),
                        Ok(None) => {}
                        Err(error) => {
                            events.push(Err(CorruptFrame::received_bytes(received, error)));
                            self.stopped = true;
                            break;
                        }
                    }
                }
                Ok(DecodedFrame::Incomplete) => break,
                Err(error) => {
                    events.push(Err(CorruptFrame::received_bytes(received, error)));
                    self.stopped = true;
                    break;
                }
            }
        }
        events
    }

    fn received_frame(&self) -> Vec<u8> {
        let length = self.received.get(..4).and_then(|bytes| {
            let prefix: [u8; 4] = bytes.try_into().ok()?;
            usize::try_from(u32::from_be_bytes(prefix)).ok()
        });
        let frame = match length {
            Some(length) if length >= 12 && length <= self.received.len() => {
                &self.received[..length]
            }
            _ => &self.received,
        };
        frame.to_vec()
    }
}

fn event(message: &Message) -> Result<Option<Value>, serde_json::Error> {
    let header = |name: &str| {
        message
            .headers()
            .iter()
            .find(|header| header.name().as_str() == name)
            .and_then(|header| header.value().as_string().ok())
            .map(|value| value.as_str().to_owned())
    };
    let kind = match header(":message-type").as_deref() {
        Some("event") => header(":event-type"),
        Some("exception") => header(":exception-type"),
        _ => return Ok(None),
    };
    let Some(kind) = kind else { return Ok(None); };
    let payload = serde_json::from_slice(message.payload())?;
    Ok(Some(Value::Object(serde_json::Map::from_iter([(kind, payload)]))))
}

#[cfg(test)]
mod tests;
