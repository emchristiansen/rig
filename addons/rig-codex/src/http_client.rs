//! An HTTP client that sends a Codex wire's requests as the Codex backend
//! expects them.
//!
//! [`CodexResponses`] is the wire: it encodes the request with the caller's
//! declared tools and `include` as written and the conversation identity in
//! the body. [`CodexHttpClient`] wraps the transport that sends it and, as
//! the last hop, makes the request carry exactly the wire's caller identity,
//! its conversation identity in headers, its request headers and its body
//! compression. It sends a JSON body only when that wire encoded it, and a
//! wire with no caller identity sends nothing. [`HttpFrameTee`] is a second wrapper, which
//! keeps the frames of the replies it hands on.
//!
//! ```
//! use rig_codex::http_client::CodexHttpClient;
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::providers::openai::responses_api::wire::Responses;
//!
//! # fn example<H: rig_core::http_client::HttpClientExt>(
//! #     http: H,
//! # ) -> Result<(), rig_codex::responses_api::codex_identity::NotACodexWire> {
//! let wire = OpenAIConfig::with_key(&chatgpt::DIALECT, "access-token");
//! let wire = CodexResponses::new(Responses::new(wire, "gpt-5.4"))?;
//! let model = rig_core::Model::new(wire.clone(), CodexHttpClient::new(http, wire));
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod frame_tee;

pub use frame_tee::HttpFrameTee;

use crate::responses_api::wire::{CodexEncoded, CodexRequestError, CodexResponses};
use bytes::Bytes;
use http::header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE};
use http::{HeaderValue, Method};
use rig_http::http_client::{
    HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;

/// An [`HttpClientExt`] that applies a Codex wire's settings to every request
/// before `inner` sends it.
#[derive(Clone, Debug)]
pub struct CodexHttpClient<H> {
    inner: H,
    wire: CodexResponses,
}

impl<H> CodexHttpClient<H> {
    /// Send through `inner` with `wire`'s caller identity, conversation
    /// identity, request headers and request compression.
    pub fn new(inner: H, wire: CodexResponses) -> Self {
        Self { inner, wire }
    }

    /// The Codex wire whose settings every request is sent with.
    pub fn wire(&self) -> &CodexResponses {
        &self.wire
    }

    /// The transport requests are sent through.
    pub fn inner(&self) -> &H {
        &self.inner
    }

    /// The request as it leaves: `request` with the wire's exact headers,
    /// and, for a JSON `POST`, the body compressed. A JSON `POST` whose body
    /// the Codex wire did not encode is refused.
    fn prepare<T: Into<Bytes>>(
        &self,
        request: Request<T>,
    ) -> std::result::Result<Request<Bytes>, CodexRequestError> {
        let (mut parts, body) = request.into_parts();
        let mut body: Bytes = body.into();
        self.wire
            .stamp_headers(self.wire.codex_identity.as_ref(), &mut parts.headers)?;

        let json_post = parts.method == Method::POST
            && !body.is_empty()
            && parts
                .headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("application/json"));
        if json_post {
            // The body's fidelity and identity are the wire's work; a body some
            // other wire wrote has neither.
            if parts.extensions.get::<CodexEncoded>().is_none() {
                return Err(CodexRequestError::NotEncodedByCodexWire);
            }
            let (encoding, encoded) = self.wire.request_compression.apply(body.to_vec())?;
            if let Some(encoding) = encoding {
                parts
                    .headers
                    .insert(CONTENT_ENCODING, HeaderValue::from_static(encoding));
            }
            body = encoded.into();
            // A stated length is the length of the body as it now is.
            if parts.headers.contains_key(CONTENT_LENGTH) {
                parts
                    .headers
                    .insert(CONTENT_LENGTH, HeaderValue::from(body.len()));
            }
        }
        Ok(Request::from_parts(parts, body))
    }
}

impl<H: HttpClientExt> HttpClientExt for CodexHttpClient<H> {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let sending = self
            .prepare(req)
            .map(|request| self.inner.send::<Bytes, U>(request));
        async move { sending?.await }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let (mut parts, form) = req.into_parts();
        let sending = self
            .wire
            .stamp_headers(self.wire.codex_identity.as_ref(), &mut parts.headers)
            .map(|()| {
                self.inner
                    .send_multipart::<U>(Request::from_parts(parts, form))
            });
        async move { sending?.await }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let sending = self
            .prepare(req)
            .map(|request| self.inner.send_streaming::<Bytes>(request));
        async move { sending?.await }
    }
}

#[cfg(test)]
mod fidelity_tests;
#[cfg(test)]
mod tests;
