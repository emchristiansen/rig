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
use rig_core::error::ProviderError;
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
    failed: bool,
}

impl Events {
    /// The JSON of each event or exception `bytes` completes, followed by
    /// one fatal error if its JSON or event-stream framing is malformed.
    pub(crate) fn read(&mut self, bytes: &[u8]) -> Vec<Result<Value, ProviderError>> {
        if self.failed {
            return Vec::new();
        }
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            // Probe borrowed bytes so an incomplete or malformed frame keeps
            // its original prelude and payload until validation succeeds.
            let mut view = self.buffer.as_ref();
            match MessageFrameDecoder::default().decode_frame(&mut view) {
                Ok(DecodedFrame::Complete(message)) => {
                    let consumed = self.buffer.len() - view.len();
                    match event(&message) {
                        Ok(value) => {
                            if let Some(value) = value {
                                events.push(Ok(value));
                            }
                            let _ = self.buffer.split_to(consumed);
                        }
                        Err(error) => {
                            events.push(Err(ProviderError::Response(format!(
                                "Bedrock Converse event-stream JSON payload failed: {error}; payload_bytes={:?}; unvalidated_wire_bytes={:?}",
                                message.payload().as_ref(),
                                self.buffer.as_ref(),
                            ))));
                            self.failed = true;
                            break;
                        }
                    }
                }
                Ok(DecodedFrame::Incomplete) => break,
                Err(error) => {
                    events.push(Err(ProviderError::Response(format!(
                        "Bedrock Converse event-stream framing failed: {error}; unvalidated_wire_bytes={:?}",
                        self.buffer.as_ref(),
                    ))));
                    self.failed = true;
                    break;
                }
            }
        }
        events
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
    let Some(message_type) = header(":message-type") else {
        return Ok(None);
    };
    let kind = match message_type.as_str() {
        "event" => header(":event-type"),
        "exception" => header(":exception-type"),
        _ => return Ok(None),
    };
    let Some(kind) = kind else {
        return Ok(None);
    };
    let payload = serde_json::from_slice(message.payload())?;
    Ok(Some(Value::Object(serde_json::Map::from_iter([(kind, payload)]))))
}

#[cfg(test)]
mod tests;
