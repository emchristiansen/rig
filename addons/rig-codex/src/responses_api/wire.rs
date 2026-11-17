//! The Codex settings a Responses wire is sent with.
//!
//! Rig's [`Responses`] wire encodes the request. [`CodexResponses`] pairs it
//! with what the Codex backend needs beside it: the caller's exact identity,
//! the conversation's [`CodexIdentity`], caller-supplied
//! [`RequestHeaders`] and [`RequestCompression`]. It is itself a [`Wire`]:
//! it sends Rig's encoding with the request's declared tools and `include` as
//! the caller wrote them and the conversation identity in the body. A
//! [`CodexHttpClient`](crate::http_client::CodexHttpClient) adds the headers
//! and compression to every HTTP request, and a
//! [`CodexWebSocketSession`](super::websocket::codex::CodexWebSocketSession)
//! shapes the same body into its frames.
//!
//! ```
//! use rig_codex::responses_api::wire::CodexResponses;
//! use rig_core::providers::chatgpt;
//! use rig_core::providers::openai::OpenAIConfig;
//! use rig_core::providers::openai::responses_api::wire::Responses;
//!
//! let wire = OpenAIConfig::with_key(&chatgpt::DIALECT, "access-token");
//! let wire = CodexResponses::new(Responses::new(wire, "gpt-5.4"))?;
//! assert!(wire.codex_identity.is_none());
//! # Ok::<(), rig_codex::responses_api::codex_identity::NotACodexWire>(())
//! ```

use super::body::RequestBody;
use super::codex_identity::{CodexIdentity, NotACodexWire, require_codex};
use super::request_compression::{RequestCompression, RequestCompressionError};
use super::request_headers::RequestHeaders;
use super::ResponsesToolDefinition;
use crate::wire::{CallerIdentity, MissingCallerIdentity, ORIGINATOR_HEADER, VERSION_HEADER};
use http::{HeaderMap, HeaderName, HeaderValue};
use rig_core::completion::CompletionRequest;
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::providers::openai::wire::Auth;
use rig_core::wire::{Body, Descriptor, Encoded, Mode, Wire, WireFrame};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::{Map, Value};

/// The `include` entry asking for reasoning ciphertext.
const REASONING_CIPHERTEXT: &str = "reasoning.encrypted_content";

/// Marks an HTTP request whose body [`CodexResponses`] encoded, so a
/// [`CodexHttpClient`](crate::http_client::CodexHttpClient) can refuse a body
/// some other wire wrote.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CodexEncoded;

/// The header naming the account a credential belongs to.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// The underscore session header some dialects send per request. A Codex
/// identity replaces it with the dashed pair.
const SESSION_ID_PER_REQUEST_HEADER: &str = "session_id";

/// The header [`Auth::ApiKeyHeader`] sends the credential in.
const API_KEY_HEADER: &str = "api-key";

/// A Responses wire with the Codex settings its requests are sent with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodexResponses {
    /// The Responses wire that encodes each request.
    pub wire: Responses,
    /// The caller identity every request carries. `None` refuses every send
    /// with [`MissingCallerIdentity`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller_identity: Option<CallerIdentity>,
    /// The conversation identity stamped on every HTTP request, and the one a
    /// websocket session built from this wire carries. `None` leaves HTTP
    /// requests as Rig encodes them, and a websocket session then takes its
    /// identity from
    /// [`with_identity`](super::websocket::codex::CodexWebSocketSessionBuilder::with_identity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_identity: Option<CodexIdentity>,
    /// Caller-supplied headers added to every HTTP request and every
    /// websocket handshake.
    #[serde(default, skip_serializing_if = "RequestHeaders::is_empty")]
    pub request_headers: RequestHeaders,
    /// How HTTP request bodies are encoded. Sent as they are by default.
    #[serde(default, skip_serializing_if = "RequestCompression::is_none")]
    pub request_compression: RequestCompression,
}

/// A Codex request that could not be given its headers or body fields.
#[derive(Debug, thiserror::Error)]
pub enum CodexRequestError {
    /// The wire's dialect does not speak the Codex contract.
    #[error(transparent)]
    NotACodexWire(#[from] NotACodexWire),
    /// No caller identity is configured.
    #[error(transparent)]
    MissingCallerIdentity(#[from] MissingCallerIdentity),
    /// A header value the request cannot carry.
    #[error("the `{name}` header cannot carry its value: {source}")]
    Header {
        /// The header's name.
        name: String,
        /// Why the value was refused.
        #[source]
        source: http::header::InvalidHeaderValue,
    },
    /// A caller-supplied header whose name is not a header name.
    #[error("`{name}` is not a valid HTTP header name")]
    HeaderName {
        /// The name as supplied.
        name: String,
    },
    /// The request body is not a JSON object.
    #[error("a Codex request body must be a JSON object: {0}")]
    Body(#[source] serde_json::Error),
    /// The body's `client_metadata` is not a JSON object.
    #[error(
        "the Codex request's `client_metadata` must be a JSON object to carry the session identity, got {found}"
    )]
    ClientMetadata {
        /// The value found in its place.
        found: serde_json::Value,
    },
    /// The body could not be compressed.
    #[error(transparent)]
    Compression(#[from] RequestCompressionError),
    /// Rig's encoding does not hold the request's declared tools where the
    /// request puts them, so they cannot be sent as the caller wrote them.
    #[error(
        "Rig's encoding of the request has {found} tools; the request's {typed} typed tools, \
         {declared} declared tools and the wire's {default} default tools make {expected}",
        expected = .typed + .declared + .default
    )]
    ToolCount {
        /// The request's typed tools (`CompletionRequest::tools`).
        typed: usize,
        /// The tools declared in `additional_params["tools"]`.
        declared: usize,
        /// The wire's own default tools.
        default: usize,
        /// The tools in Rig's encoding.
        found: usize,
    },
    /// A tool in Rig's encoding is not the declared tool at its place.
    #[error(
        "Rig's encoding of declared tool {index} is {found}, which is not the declaration's \
         typed form {expected}"
    )]
    DeclaredTool {
        /// The tool's position among the declared tools.
        index: usize,
        /// The declaration's typed form, as Rig's encoder writes it.
        expected: Value,
        /// What Rig's encoding holds there.
        found: Value,
    },
    /// Rig's encoding states an `include` other than the request's own with
    /// the reasoning ciphertext added.
    #[error(
        "Rig's encoding states `include` {found}, which is not the request's own {asked} \
         with the reasoning ciphertext added"
    )]
    Include {
        /// The request's own `include`, `null` when it sets none.
        asked: Value,
        /// The `include` in Rig's encoding.
        found: Value,
    },
    /// A JSON `POST` whose body [`CodexResponses`] did not encode.
    #[error(
        "a Codex HTTP client sends only bodies a `CodexResponses` wire encoded; send the \
         request through `CodexResponses` as the wire"
    )]
    NotEncodedByCodexWire,
}

// Each conversion hands on a refusal that has its own type as that type, so a
// caller can downcast to it: `MissingCallerIdentity` stays one however it
// reached the caller.
impl From<CodexRequestError> for ProviderError {
    fn from(error: CodexRequestError) -> Self {
        match error {
            CodexRequestError::NotACodexWire(error) => ProviderError::request(error),
            CodexRequestError::MissingCallerIdentity(error) => ProviderError::request(error),
            CodexRequestError::Compression(error) => ProviderError::request(error),
            other => ProviderError::request(other),
        }
    }
}

impl From<CodexRequestError> for EncodeError {
    fn from(error: CodexRequestError) -> Self {
        match error {
            CodexRequestError::NotACodexWire(error) => EncodeError::request(error),
            CodexRequestError::MissingCallerIdentity(error) => EncodeError::request(error),
            CodexRequestError::Compression(error) => EncodeError::request(error),
            other => EncodeError::request(other),
        }
    }
}

impl From<CodexRequestError> for rig_http::http_client::Error {
    fn from(error: CodexRequestError) -> Self {
        use rig_http::http_client::Error;
        match error {
            CodexRequestError::NotACodexWire(error) => Error::instance(error),
            CodexRequestError::MissingCallerIdentity(error) => Error::instance(error),
            CodexRequestError::Compression(error) => Error::instance(error),
            other => Error::instance(other),
        }
    }
}

/// A request control this crate refuses to send on the Codex contract.
///
/// Rig's encoder leaves a caller-set output-token cap and temperature off a
/// Codex request without saying so. [`CodexResponses`], over HTTP and in a
/// [`CodexWebSocketSession`](super::websocket::codex::CodexWebSocketSession),
/// refuses each by name instead, before anything is sent, so a caller learns
/// that the control it set would not have reached the backend. This states
/// adapter policy, not a rule the backend is known to enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnsupportedCodexControl {
    /// A caller-set output-token cap (`max_output_tokens`).
    #[error(
        "this adapter refuses a caller-set `max_output_tokens` on the Codex Responses \
         contract; leave the output-token cap unset for this dialect"
    )]
    MaxOutputTokens,
    /// A caller-set `store: true`; `store: false` is what this adapter states
    /// for the Codex contract.
    #[error(
        "this adapter refuses a caller-set `store: true` on the Codex Responses contract; \
         leave `store` unset or send `false`"
    )]
    Store,
    /// A caller-set nucleus-sampling value (`top_p`).
    #[error(
        "this adapter refuses a caller-set `top_p` on the Codex Responses contract; \
         leave `top_p` unset for this dialect"
    )]
    TopP,
    /// A caller-set sampling temperature (`temperature`).
    #[error(
        "this adapter refuses a caller-set `temperature` on the Codex Responses contract; \
         leave `temperature` unset for this dialect"
    )]
    Temperature,
}

impl UnsupportedCodexControl {
    /// The first control `request` sets that the Codex contract does not
    /// take, in the order output-token cap, temperature, `top_p`, `store`.
    #[must_use]
    pub fn in_request(request: &CompletionRequest) -> Option<Self> {
        let param = |name: &str| {
            request
                .additional_params
                .as_ref()
                .and_then(|params| params.get(name))
                .filter(|value| !value.is_null())
        };
        if request.max_tokens.is_some() {
            Some(Self::MaxOutputTokens)
        } else if request.temperature.is_some() {
            Some(Self::Temperature)
        } else if param("top_p").is_some() {
            Some(Self::TopP)
        } else if param("store") == Some(&serde_json::Value::Bool(true)) {
            Some(Self::Store)
        } else {
            None
        }
    }
}

impl From<UnsupportedCodexControl> for ProviderError {
    fn from(error: UnsupportedCodexControl) -> Self {
        ProviderError::request(error)
    }
}

impl CodexResponses {
    /// Codex settings for `wire`, with no caller identity, no conversation
    /// identity, no request headers and no compression. Refuses a wire whose
    /// dialect does not speak the Codex contract.
    pub fn new(wire: Responses) -> Result<Self, NotACodexWire> {
        require_codex(&wire)?;
        Ok(Self {
            wire,
            caller_identity: None,
            codex_identity: None,
            request_headers: RequestHeaders::new(),
            request_compression: RequestCompression::default(),
        })
    }

    /// Send every request with exactly `identity`'s `originator`,
    /// `user-agent` and optional `version` headers.
    #[must_use]
    pub fn with_caller_identity(mut self, identity: CallerIdentity) -> Self {
        self.caller_identity = Some(identity);
        self
    }

    /// Name one conversation on every HTTP request and on the websocket
    /// session built from this wire. Refuses a wire whose dialect does not
    /// speak the Codex contract.
    pub fn with_codex_identity(mut self, identity: CodexIdentity) -> Result<Self, NotACodexWire> {
        require_codex(&self.wire)?;
        self.codex_identity = Some(identity);
        Ok(self)
    }

    /// Add `headers` to every HTTP request and every websocket handshake.
    #[must_use]
    pub fn with_request_headers(mut self, headers: RequestHeaders) -> Self {
        self.request_headers = headers;
        self
    }

    /// Encode every HTTP request body with `compression`.
    #[must_use]
    pub fn with_request_compression(mut self, compression: RequestCompression) -> Self {
        self.request_compression = compression;
        self
    }

    /// The caller identity a request is sent with, or the refusal when the
    /// wire has none.
    pub fn sendable_identity(&self) -> Result<&CallerIdentity, MissingCallerIdentity> {
        self.caller_identity.as_ref().ok_or(MissingCallerIdentity {
            dialect: self.wire.provider.dialect.name,
        })
    }

    /// Set the credential and account headers Rig's own encoder sends, for a
    /// request this crate builds itself.
    pub(crate) fn stamp_credential(
        &self,
        headers: &mut HeaderMap,
    ) -> Result<(), CodexRequestError> {
        let provider = &self.wire.provider;
        let bearer = || format!("Bearer {}", provider.api_key.expose());
        match provider.auth {
            Auth::Bearer => set(headers, http::header::AUTHORIZATION.as_str(), &bearer())?,
            Auth::OptionalBearer if provider.api_key.is_empty() => {}
            Auth::OptionalBearer => set(headers, http::header::AUTHORIZATION.as_str(), &bearer())?,
            Auth::ApiKeyHeader => set(headers, API_KEY_HEADER, provider.api_key.expose())?,
        }
        if let Some(account_id) = &provider.account_id {
            set(headers, CHATGPT_ACCOUNT_ID_HEADER, account_id)?;
        }
        Ok(())
    }

    /// Make `headers` carry exactly this wire's caller identity, the Codex
    /// identity `identity` when there is one, and the caller's request
    /// headers, replacing any value another source put under those names.
    pub(crate) fn stamp_headers(
        &self,
        identity: Option<&CodexIdentity>,
        headers: &mut HeaderMap,
    ) -> Result<(), CodexRequestError> {
        require_codex(&self.wire)?;
        let caller = self.sendable_identity()?;
        set(headers, ORIGINATOR_HEADER, caller.originator())?;
        set(
            headers,
            http::header::USER_AGENT.as_str(),
            caller.user_agent(),
        )?;
        match caller.version() {
            Some(version) => set(headers, VERSION_HEADER, version)?,
            None => {
                headers.remove(VERSION_HEADER);
            }
        }
        if let Some(identity) = identity {
            // The conversation's identity replaces a per-request session id.
            headers.remove(SESSION_ID_PER_REQUEST_HEADER);
            for (name, value) in identity.headers() {
                set(headers, name, value)?;
            }
        }
        for (name, value) in self.request_headers.iter() {
            set(headers, name, value)?;
        }
        Ok(())
    }
}

impl CodexResponses {
    /// Rig's encoding of `request` in `mode`, and its body as this crate
    /// sends it: the request's declared tools and its `include` as the
    /// caller wrote them ([`CallerFields::restore`]). `request` must already
    /// be prepared for the wire, as Rig's driver prepares it. Refuses a
    /// control the Codex contract does not take ([`UnsupportedCodexControl`])
    /// before encoding.
    pub(crate) fn encode_body(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<(Encoded, RequestBody), EncodeError> {
        if let Some(control) = UnsupportedCodexControl::in_request(&request) {
            return Err(EncodeError::request(control));
        }
        let asked = CallerFields::of(&request);
        let encoded = self.wire.encode(request, mode)?;
        let Body::Bytes(bytes) = encoded.request.body() else {
            return Err(EncodeError::request(
                "a Responses request body must be JSON bytes",
            ));
        };
        let mut body = RequestBody::parse(bytes).map_err(CodexRequestError::Body)?;
        asked.restore(&self.wire, &mut body)?;
        Ok((encoded, body))
    }
}

/// What the caller wrote into a request where Rig's encoder writes something
/// else on the Codex contract.
///
/// Rig reads each tool declared in `additional_params["tools"]` into its own
/// tool type and writes it back, which drops an empty `description` or
/// `name`, adds `strict` to a function and reorders keys. It also asks every
/// Codex request for the reasoning ciphertext. This crate sends a declared
/// tool as the JSON value the caller wrote, changed only by strict mode's own
/// transformation of a function (its `strict` and sanitized `parameters`),
/// and adds the ciphertext `include` only when the request reasons, as every
/// other Responses dialect does. Every other field is Rig's.
struct CallerFields {
    /// The number of typed tools (`CompletionRequest::tools`), which Rig
    /// writes ahead of the declared ones.
    typed_tools: usize,
    /// The tools in `additional_params["tools"]`.
    declared_tools: Vec<Value>,
    /// `additional_params["include"]`, when the request sets one.
    include: Option<Value>,
}

impl CallerFields {
    fn of(request: &CompletionRequest) -> Self {
        let param = |name: &str| {
            request
                .additional_params
                .as_ref()
                .and_then(|params| params.get(name))
                .filter(|value| !value.is_null())
        };
        Self {
            typed_tools: request.tools.len(),
            declared_tools: param("tools")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            include: param("include").cloned(),
        }
    }

    /// Put the declared tools and the `include` back into `body`, Rig's
    /// encoding of the request on `wire`. Each edit first checks that `body`
    /// holds what Rig writes for the request, and refuses otherwise.
    fn restore(&self, wire: &Responses, body: &mut RequestBody) -> Result<(), CodexRequestError> {
        self.restore_tools(wire, body)?;
        self.restore_include(body)
    }

    /// Replace Rig's writing of each declared tool with the declaration.
    fn restore_tools(&self, wire: &Responses, body: &mut RequestBody) -> Result<(), CodexRequestError> {
        if self.declared_tools.is_empty() {
            return Ok(());
        }
        let mut tools: Vec<Box<RawValue>> = match body.get("tools") {
            Some(tools) => tools.map_err(CodexRequestError::Body)?,
            None => Vec::new(),
        };
        let count = CodexRequestError::ToolCount {
            typed: self.typed_tools,
            declared: self.declared_tools.len(),
            default: wire.tools.len(),
            found: tools.len(),
        };
        if tools.len() != self.typed_tools + self.declared_tools.len() + wire.tools.len() {
            return Err(count);
        }
        let encoded = tools
            .get_mut(self.typed_tools..self.typed_tools + self.declared_tools.len())
            .ok_or(count)?;
        for (index, (declared, encoded)) in
            self.declared_tools.iter().zip(encoded.iter_mut()).enumerate()
        {
            let restored = declared_tool(wire, index, declared, encoded)?;
            *encoded = serde_json::value::to_raw_value(&restored).map_err(CodexRequestError::Body)?;
        }
        body.insert("tools", &tools).map_err(CodexRequestError::Body)
    }

    /// Take the ciphertext `include` Rig adds back out, unless the request
    /// asked for it or reasons.
    fn restore_include(&self, body: &mut RequestBody) -> Result<(), CodexRequestError> {
        let asked_ciphertext = self
            .include
            .as_ref()
            .and_then(Value::as_array)
            .is_some_and(|include| include.iter().any(|item| item == REASONING_CIPHERTEXT));
        if asked_ciphertext || body.contains("reasoning") {
            return Ok(());
        }
        let Some(found) = body.get::<Value>("include") else {
            return Ok(());
        };
        let found = found.map_err(CodexRequestError::Body)?;
        let mismatch = || CodexRequestError::Include {
            asked: self.include.clone().unwrap_or(Value::Null),
            found: found.clone(),
        };
        // Rig appends the ciphertext to the request's own `include`, or
        // states it alone when the request sets none.
        let Some((added, own)) = found.as_array().and_then(|items| items.split_last()) else {
            return Err(mismatch());
        };
        if added != REASONING_CIPHERTEXT {
            return Err(mismatch());
        }
        match &self.include {
            None if own.is_empty() => {
                body.remove("include");
                Ok(())
            }
            Some(Value::Array(asked)) if asked.as_slice() == own => {
                // The request's own entries, as Rig wrote them.
                let mut entries: Vec<Box<RawValue>> = body
                    .get("include")
                    .ok_or_else(mismatch)?
                    .map_err(CodexRequestError::Body)?;
                entries.pop();
                body.insert("include", &entries)
                    .map_err(CodexRequestError::Body)
            }
            _ => Err(mismatch()),
        }
    }
}

/// The declared tool `declared` as it is sent: the declaration, with strict
/// mode's `strict` and sanitized `parameters` when the wire makes tools
/// strict and the tool is a function. `encoded` is Rig's writing of it, which
/// must be the declaration's typed form.
fn declared_tool(
    wire: &Responses,
    index: usize,
    declared: &Value,
    encoded: &RawValue,
) -> Result<Value, CodexRequestError> {
    let found: Value = serde_json::from_str(encoded.get()).map_err(CodexRequestError::Body)?;
    let typed: ResponsesToolDefinition =
        serde_json::from_value(declared.clone()).map_err(CodexRequestError::Body)?;
    let sent = if wire.strict_tools {
        typed.clone().with_strict()
    } else {
        typed.clone()
    };
    let expected = serde_json::to_value(&sent).map_err(CodexRequestError::Body)?;
    if expected != found {
        return Err(CodexRequestError::DeclaredTool {
            index,
            expected,
            found,
        });
    }
    let Value::Object(mut restored) = declared.clone() else {
        return Err(CodexRequestError::DeclaredTool {
            index,
            expected,
            found,
        });
    };
    if sent.strict != typed.strict {
        restored.insert("strict".to_owned(), Value::Bool(sent.strict));
    }
    if sent.parameters != typed.parameters {
        restored.insert("parameters".to_owned(), sent.parameters);
    }
    Ok(Value::Object(restored))
}

/// Rig's Responses wire with the request fidelity and conversation identity
/// of the Codex contract.
///
/// The body is Rig's, with the request's declared tools and `include` as
/// the caller wrote them ([`CallerFields`]) and the conversation identity
/// stamped in, and a caller-set control the contract does not take refused
/// ([`UnsupportedCodexControl`]). A wire with no caller identity refuses to
/// encode. Headers and compression are applied by
/// [`CodexHttpClient`](crate::http_client::CodexHttpClient), which sends only
/// bodies this wire encoded. Decoding is Rig's.
impl Wire for CodexResponses {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Event = <Responses as Wire>::Event;
    type Decoder<'id> = <Responses as Wire>::Decoder<'id>;
    type Reassembler = <Responses as Wire>::Reassembler;

    fn describe(&self) -> Descriptor<'_> {
        self.wire.describe()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        require_codex(&self.wire).map_err(CodexRequestError::NotACodexWire)?;
        self.sendable_identity()
            .map_err(CodexRequestError::MissingCallerIdentity)?;
        let (mut encoded, mut body) = self.encode_body(request, mode)?;
        if let Some(identity) = &self.codex_identity {
            identity.stamp(&mut body)?;
        }
        let bytes = body.to_vec().map_err(CodexRequestError::Body)?;
        *encoded.request.body_mut() = Body::Bytes(bytes);
        encoded.request.extensions_mut().insert(CodexEncoded);
        Ok(encoded)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.wire.decoder()
    }

    fn reassembler(&self) -> Self::Reassembler {
        self.wire.reassembler()
    }
}

/// Set `name` to `value` in `headers`, replacing any value it had.
pub(crate) fn set(
    headers: &mut HeaderMap,
    name: &str,
    value: &str,
) -> Result<(), CodexRequestError> {
    let header =
        HeaderName::from_bytes(name.as_bytes()).map_err(|_| CodexRequestError::HeaderName {
            name: name.to_owned(),
        })?;
    let value = HeaderValue::from_str(value).map_err(|source| CodexRequestError::Header {
        name: name.to_owned(),
        source,
    })?;
    headers.insert(header, value);
    Ok(())
}

#[cfg(test)]
mod tests;
