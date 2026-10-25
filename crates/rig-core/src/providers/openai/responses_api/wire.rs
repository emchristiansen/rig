//! Responses endpoint encoding and observation for configured OpenAI dialects.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").responses("gpt-5.2");
//! ```

#[cfg(feature = "completion-observations")]
pub(super) mod observed;

use crate::completion::{self, ProviderCapabilities};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::observe::ObservedError;
use crate::operation::Completion;
use crate::providers::openai::wire::OpenAIConfig;
pub(crate) use crate::providers::openai::wire::ResponsesContract;
use crate::wire::{
    AdapterUsage, AdapterVerdict, Capabilities, Descriptor, Encoded, Framing, Mode,
    ObservationSink, UsageObservation, Wire,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::request_compression::{BodyOrder, RequestCompression};
use super::request_headers::RequestHeaders;
use super::responses_lite::{CodexRequestShape, ResponsesLiteError};
use super::streaming::{IncompleteTerminal, ResponsesDecoder, ResponsesStreamOptions};

const SESSION_ID_PER_REQUEST_HEADER: &str = "session_id";
use super::{ResponsesToolDefinition, SystemInstructionsPlacement};

/// The Responses wire: `POST /responses`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Responses {
    /// The provider this wire speaks to.
    pub provider: OpenAIConfig,
    /// The model to address.
    pub model: String,
    /// Tools added to every request from this wire.
    pub tools: Vec<ResponsesToolDefinition>,
    /// Whether Rig-generated tools request OpenAI's strict validation.
    pub strict_tools: bool,
    /// Where this wire puts Rig's system instructions. Defaults to the
    /// dialect's placement.
    pub system_instructions: SystemInstructionsPlacement,
    /// How a streamed (SSE) reply's terminal `response.incomplete` is taken:
    /// refused by default, accepted with its truthful finish reason when the
    /// caller opts in. A unary reply always accepts it. This is decoder
    /// policy, never a request parameter, so it does not reach the wire.
    #[serde(default)]
    pub streamed_incomplete: IncompleteTerminal,
    /// The Codex conversation identity every HTTP request from this wire
    /// carries, when set (see [`Self::with_codex_identity`]). `None` sends a
    /// fresh `session_id` per request, as the dialect asks.
    ///
    /// An identity names ONE conversation: every HTTP request, and every
    /// websocket session built from this wire value, shares its cache
    /// affinity. A caller running several conversations gives each its own
    /// wire value with its own identity ([`CodexIdentity::from_ids`] or
    /// [`CodexIdentity::generate`]).
    ///
    /// [`CodexIdentity::from_ids`]: super::codex_identity::CodexIdentity::from_ids
    /// [`CodexIdentity::generate`]: super::codex_identity::CodexIdentity::generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_identity: Option<super::codex_identity::CodexIdentity>,
    /// Whether this wire emits the ordinary Responses envelope or the Codex
    /// Responses Lite developer prefix. Standard is the serialized default.
    #[serde(default, skip_serializing_if = "CodexRequestShape::is_standard")]
    pub codex_request_shape: CodexRequestShape,
    /// Caller-supplied headers added to every HTTP request and every
    /// websocket handshake this wire sends (see [`Self::with_request_headers`]).
    /// Empty by default.
    #[serde(default, skip_serializing_if = "RequestHeaders::is_empty")]
    pub request_headers: RequestHeaders,
    /// How this wire encodes its HTTP request bodies (see
    /// [`Self::with_request_compression`]). Sent as they are by default.
    #[serde(default, skip_serializing_if = "RequestCompression::is_none")]
    pub request_compression: RequestCompression,
}

impl Responses {
    pub(crate) fn encode_with_headers(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
        headers: impl FnOnce(
            &OpenAIConfig,
            &completion::CompletionRequest,
            http::request::Builder,
        ) -> Result<http::request::Builder, EncodeError>,
    ) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks.responses;
        // The codex gateway only ever answers with an event stream, and
        // names no content type on it. It is asked for one whatever the
        // caller wanted: the reply is framed the same way either way, and
        // the driver folds it.
        let codex = quirks.contract == ResponsesContract::Codex;
        let streaming = matches!(mode, Mode::Streaming) || codex;
        let mut builder = self.request_headers.stamp(headers(
            &self.provider,
            &request,
            http::Request::post(self.provider.uri(quirks.path, None)),
        )?);
        let mode = if streaming {
            Mode::Streaming
        } else {
            Mode::Unary
        };
        let body = self.responses_request(&request, super::Delivery::Http(mode))?;
        if self.codex_request_shape.is_lite() {
            builder = builder.header(super::responses_lite::HTTP_HEADER, "true");
        }
        if let Some(identity) = &self.codex_identity {
            super::codex_identity::require_codex(self).map_err(EncodeError::request)?;
            if let Some(headers) = builder.headers_mut() {
                headers.remove(SESSION_ID_PER_REQUEST_HEADER);
            }
            builder = identity.stamp_headers(builder);
        }
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Responses completion request",
            &body,
        );
        let order = if codex && self.codex_request_shape.is_standard() {
            BodyOrder::RoutingFirst
        } else {
            BodyOrder::Original
        };
        let (builder, body) = self.request_compression.apply(builder, body, order)?;
        let request = builder
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(body.into_body())?;

        let framing = if streaming {
            Framing::Sse
        } else {
            Framing::Whole
        };
        let encoded = Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header)
            .with_captured_response_headers(self.provider.dialect.response_header_prefix)
            .with_route(Some(self.provider.dialect.quirks.responses.path))
            .with_projection(project_payload);
        Ok(if codex {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    /// Name the Codex conversation every HTTP request from this wire belongs
    /// to. Each request then carries `identity`'s dashed `session-id` and
    /// `thread-id` headers, `x-client-request-id`, and its `prompt_cache_key`
    /// and `client_metadata` body fields, exactly as a Codex websocket session
    /// with the same identity does, instead of a fresh `session_id` header.
    /// A request's own `prompt_cache_key` or metadata keys are kept. Refuses a
    /// wire whose dialect does not speak the Codex contract.
    pub fn with_codex_identity(
        mut self,
        identity: super::codex_identity::CodexIdentity,
    ) -> Result<Self, super::codex_identity::NotACodexWire> {
        super::codex_identity::require_codex(&self)?;
        self.codex_identity = Some(identity);
        Ok(self)
    }

    /// Create a wire with the provider's instruction placement and dialect's
    /// strict-tool default, without additional tools.
    pub fn new(provider: OpenAIConfig, model: impl Into<String>) -> Self {
        Self {
            system_instructions: provider.system_instructions_placement(),
            strict_tools: provider.dialect.quirks.responses.strict_tools_by_default,
            provider,
            model: model.into(),
            tools: Vec::new(),
            streamed_incomplete: IncompleteTerminal::default(),
            codex_identity: None,
            codex_request_shape: CodexRequestShape::default(),
            request_headers: RequestHeaders::new(),
            request_compression: RequestCompression::default(),
        }
    }

    /// Encode every HTTP request body this wire sends as `compression` says;
    /// see [`RequestCompression`]. Websocket frames are unaffected.
    #[must_use]
    pub fn with_request_compression(mut self, compression: RequestCompression) -> Self {
        self.request_compression = compression;
        self
    }

    /// Add `headers` to every HTTP request and every websocket handshake this
    /// wire sends, exactly as given, beside the headers Rig sends itself,
    /// replacing any set given before. For headers a gateway asks for that
    /// Rig does not model, such as the Codex backend's `x-codex-*` request
    /// headers; [`RequestHeaders`] refuses the ones Rig sends from another
    /// source, so the position of a caller header carries no meaning.
    ///
    /// A wire is a cheap value: a header whose value changes per request
    /// (a turn's metadata, say) is a per-request wire. The same set goes on
    /// HTTP requests and on handshakes alike, so a header only one transport
    /// should carry needs a wire of its own for that transport: the Codex
    /// backend's `x-codex-turn-state`, for one, travels as an HTTP header but
    /// never on a handshake.
    #[must_use]
    pub fn with_request_headers(mut self, headers: RequestHeaders) -> Self {
        self.request_headers = headers;
        self
    }

    /// Emit the Codex Responses Lite request shape.
    ///
    /// The wire still needs a stable [`CodexIdentity`](super::codex_identity::CodexIdentity)
    /// before HTTP encoding. A Codex WebSocket builder supplies its selected
    /// session identity when it prepares each frame. Returns
    /// [`ResponsesLiteError::ResponsesLiteRequiresCodex`] for another dialect.
    pub fn with_responses_lite(mut self) -> Result<Self, ResponsesLiteError> {
        super::responses_lite::validate_codex(
            self.provider.dialect.quirks.responses.contract,
            self.provider.dialect.name,
        )?;
        self.codex_request_shape = CodexRequestShape::ResponsesLite;
        Ok(self)
    }

    /// Select how a streamed reply's terminal `response.incomplete` is taken.
    ///
    /// [`IncompleteTerminal::Refuse`] (the default) ends the stream with an
    /// error carrying the provider's terminal event;
    /// [`IncompleteTerminal::Accept`] opts this wire's requests into partial
    /// success: the partial output and usage are kept and the turn ends with
    /// the finish reason the provider's `incomplete_details` states. A wire
    /// is a cheap value, so a per-request choice is a per-request wire.
    pub fn with_streamed_incomplete(mut self, incomplete: IncompleteTerminal) -> Self {
        self.streamed_incomplete = incomplete;
        self
    }

    /// The decoder for one reply, taking a terminal
    /// `response.incomplete` as `incomplete` says rather than as this wire's
    /// transport default.
    ///
    /// For a transport whose contract differs from HTTP's, such as a
    /// websocket session, which accepts an incomplete terminal.
    pub fn decoder_with_incomplete(&self, incomplete: IncompleteTerminal) -> ResponsesDecoder {
        let quirks = &self.provider.dialect.quirks.responses;
        let options = if quirks.contract == ResponsesContract::Xai {
            // xAI answers a 200 with its error envelope, and the same
            // gateway publishes a finished call at its `output_item.done`.
            ResponsesStreamOptions::strict_with_immediate_tool_calls()
        } else {
            ResponsesStreamOptions::strict()
        }
        .with_incomplete(incomplete);
        let mut decoder = ResponsesDecoder::new(self.provider.dialect.name, options);
        if quirks.contract == ResponsesContract::Codex {
            // The codex gateway's replayed frames may omit their envelope
            // bookkeeping; elsewhere an envelope-less frame is a defect
            // worth surfacing rather than salvaging.
            decoder = decoder.with_envelope_repair();
        }
        if self.provider.dialect.quirks.upstream_reasoning_issuer {
            decoder = decoder.with_upstream_reasoning_issuer();
        }
        decoder
    }

    /// Sanitize function schemas for strict mode and send `strict: true`.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Add a tool to every request from this wire.
    pub fn with_tool(mut self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add tools to every request from this wire.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Put Rig's system instructions somewhere other than the dialect's
    /// default placement.
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.system_instructions = placement;
        self
    }

    /// Send Rig's system instructions as `system` messages in `input`, for a
    /// backend that rejects or ignores top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }
}

/// A caller-set request control this adapter refuses on the Codex Responses
/// contract.
///
/// Refused by name at encode rather than cleared: a control the caller set and
/// the wire silently dropped would make the request that left differ from the
/// one the caller built, with nothing to say so. Only a value the caller set is
/// refused; an unset control emits no field and raises nothing. Every control
/// not named here is sent as the caller set it. The refusal is this adapter's
/// policy for the Codex contract; it is not a statement about what the Codex
/// backend itself accepts.
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

/// Apply the Codex contract to the final merged body without clearing caller fields.
pub(crate) fn shape_codex_body(body: &mut Map<String, Value>) -> Result<(), EncodeError> {
    // These two controls were recognized by the fork's additional-parameter
    // decoder, which ran before any valid-value Codex refusal.
    let decode_error = |error| {
        EncodeError::request(format!(
            "Invalid OpenAI Responses additional_params payload: {error}"
        ))
    };
    let top_p =
        serde_json::from_value::<Option<f64>>(body.get("top_p").cloned().unwrap_or(Value::Null))
            .map_err(decode_error)?;
    let store =
        serde_json::from_value::<Option<bool>>(body.get("store").cloned().unwrap_or(Value::Null))
            .map_err(decode_error)?;
    if body
        .get("max_output_tokens")
        .is_some_and(|value| !value.is_null())
    {
        return Err(EncodeError::request(
            UnsupportedCodexControl::MaxOutputTokens,
        ));
    }
    if body
        .get("temperature")
        .is_some_and(|value| !value.is_null())
    {
        return Err(EncodeError::request(UnsupportedCodexControl::Temperature));
    }
    if top_p.is_some() {
        return Err(EncodeError::request(UnsupportedCodexControl::TopP));
    }
    match store {
        Some(true) => return Err(EncodeError::request(UnsupportedCodexControl::Store)),
        Some(false) => {}
        None => {
            body.insert("store".to_owned(), Value::Bool(false));
        }
    }
    Ok(())
}

impl Wire for Responses {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ResponsesDecoder;
    type Reassembler = super::streaming::document::Response;

    /// The xAI contract does not compose native structured output with tools.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(
                    self.provider.dialect.quirks.responses.contract != ResponsesContract::Xai,
                ),
            ))
            .replay(self)
            .usage_observation(UsageObservation::PayloadSnapshots)
    }

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAIConfig::completion_headers)
    }

    fn reassembler(&self) -> Self::Reassembler {
        super::streaming::document::Response::with_strict_reader(
            self.provider.dialect.quirks.responses.contract == ResponsesContract::Codex,
        )
    }

    fn credential_stamp(&self) -> Option<crate::wire::CredentialStamp> {
        self.provider.credential_stamp()
    }

    fn decoder_in<'id>(&self, mode: Mode) -> Self::Decoder<'id> {
        let incomplete = match mode {
            Mode::Unary => IncompleteTerminal::Accept,
            Mode::Streaming => self.streamed_incomplete,
        };
        self.decoder_with_incomplete(incomplete)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.decoder_with_incomplete(self.streamed_incomplete)
    }
}

impl crate::completion::ReplayTarget for Responses {
    fn check_carried_text(
        &self,
        _part: &crate::message::CarriedText,
    ) -> Result<(), crate::message::UnrepresentableOpaqueContent> {
        Ok(())
    }

    // The effective request model selects reasoning issuers at final encoding.
    fn check_carried_reasoning(
        &self,
        _part: &crate::message::CarriedReasoning,
    ) -> Result<(), crate::message::UnrepresentableOpaqueContent> {
        Ok(())
    }

    /// Carried items keep the fork's per-item issuer rule, independently of
    /// Origin and the requested model's native replay rules. The configured
    /// wire's accepted issuer families are the same ones the fork checked.
    fn check_provider_item(
        &self,
        item: &crate::message::ProviderItem,
    ) -> Result<(), crate::message::UnreplayableProviderItem> {
        let issuers =
            crate::providers::openai::wire::replay_issuers(&self.provider.dialect, &self.model);
        if issuers.iter().any(|issuer| item.replayable_to(issuer)) {
            Ok(())
        } else {
            Err(crate::message::UnreplayableProviderItem::new(
                self.provider.dialect.name,
                item,
            ))
        }
    }

    /// Section 6.3 of the typed-options design, by dialect.
    fn map_options(
        &self,
        request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::providers::openai::options::responses_options(self, request, fields)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("openai.responses")
    }

    fn provider(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// The wire's own tools are declared beside the request's.
    fn declares_tools(&self, request: &completion::CompletionRequest) -> bool {
        !self.tools.is_empty() || crate::completion::history::declares_tools(request)
    }

    /// A call item names its id in `call_id`.
    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/call_id")
    }

    /// An edited block keeps its item's `id` and `type`, a message its
    /// `phase`, and reasoning its ciphertext, so the items after it stay
    /// paired (pi's text signature is `{id, phase}`).
    fn identity(&self, item: &Value) -> Map<String, Value> {
        let keys: &[&str] = match item.str("type") {
            Some("message") => &["type", "id", "phase"],
            Some("reasoning") => &["type", "id", "encrypted_content"],
            Some("function_call" | "custom_tool_call") => &["type", "id"],
            _ => &[],
        };
        keys.iter()
            .filter_map(|key| Some(((*key).to_owned(), item.get(*key)?.clone())))
            .collect()
    }

    /// A reasoning item goes only with the item it preceded.
    fn needs_next(&self, item: &Value) -> bool {
        item.str("type") == Some("reasoning")
    }

    /// Responses reads images in user input and in function outputs, never
    /// in assistant messages, and only on a model with vision input. Every
    /// documented model calls tools except `o1-mini` and `o1-preview`.
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        let images = reads_images(self.provider.dialect.quirks.responses.contract, model);
        let model = model.to_ascii_lowercase();
        crate::completion::Accepts {
            user_images: images,
            assistant_images: false,
            tool_result_images: images,
            tools: !(model.starts_with("o1-mini") || model.starts_with("o1-preview")),
        }
    }

    /// Responses carries user and tool-result images as data URLs, URLs or
    /// file ids, documents as files or text, and no audio, video or
    /// assistant image. File ids need a dialect that resolves them, and
    /// xAI's `input_image` takes none.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::{Media, Place};
        use crate::message::{DocumentData, DocumentSourceKind as Source};
        let quirks = &self.provider.dialect.quirks;
        let file_ids = quirks.accepts_file_ids;
        match media {
            Media::Image(_, Place::Assistant) | Media::Audio(_) | Media::Video(_) => false,
            Media::Image(image, _) => {
                super::image_part(image, self.codex_request_shape.is_lite()).is_some()
                    && (!matches!(image.data, Source::FileId(_))
                        || file_ids && quirks.responses.contract != ResponsesContract::Xai)
            }
            Media::Document(document) => {
                super::document_part(document).is_some()
                    && (file_ids || !matches!(document.data, DocumentData::File(Source::FileId(_))))
            }
        }
    }

    /// A request naming `previous_response_id` or a `conversation`
    /// continues state the provider stores, which holds the calls its first
    /// results answer.
    fn continues_stored(&self, request: &completion::CompletionRequest) -> bool {
        ["previous_response_id", "conversation"].iter().any(|key| {
            crate::completion::options::param(self, request, key)
                .is_some_and(|value| !value.is_null())
        })
    }

    /// pi's `normalizeIdPart`: characters outside `[a-zA-Z0-9_-]` become
    /// `_`, the id is cut to 64 characters and loses its trailing `_`.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _source: Option<&crate::message::Origin>,
    ) -> String {
        use crate::providers::internal::wire_ids::{legal_call_id, short_hash};
        let legal = legal_call_id(id, 64);
        match legal.trim_end_matches('_') {
            // An id with nothing legal in it still names its call.
            "" => short_hash(id),
            trimmed => trimmed.to_owned(),
        }
    }
}

/// Whether `model` reads images, past a `vendor/` prefix: its catalog
/// entry's input, or for a model the catalog does not list its vendor's
/// documented text-only models. An unknown model reads them.
fn reads_images(contract: ResponsesContract, model: &str) -> bool {
    let model = model.rsplit('/').next().unwrap_or_default();
    match contract {
        ResponsesContract::Xai => crate::catalog::reads_images_or(
            crate::providers::xai::DIALECT.name,
            model,
            crate::providers::xai::reads_images,
        ),
        ResponsesContract::OpenAi | ResponsesContract::Codex => crate::catalog::reads_images_or(
            crate::providers::openai::wire::OPENAI.name,
            model,
            crate::providers::openai::reads_images,
        ),
    }
}

/// The facts a Responses payload carries before normalization discards
/// them: the verdict, the model, the response id, usage and any error envelope.
///
/// The unary reply is the response object itself; a stream event wraps that
/// object under `response` (`response.created`, `.completed`, `.failed`,
/// `.incomplete`) or, for `error`, carries the envelope's fields itself.
pub(crate) fn project_payload(payload: &[u8], sink: &mut ObservationSink<'_>) {
    let Ok(payload) = serde_json::from_slice::<Payload>(payload) else {
        return;
    };
    if payload.kind.as_deref() == Some("error") {
        payload
            .unwrapped
            .error
            .unwrap_or(PayloadError {
                code: payload.code,
                kind: None,
                message: payload.message,
            })
            .emit(sink);
        return;
    }
    let object = payload.response.unwrap_or(payload.unwrapped);
    if let Some(usage) = object.usage {
        sink.usage_snapshot(AdapterUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            cached_input_tokens: usage
                .input_tokens_details
                .and_then(|details| details.cached_tokens),
            reasoning_tokens: usage
                .output_tokens_details
                .and_then(|details| details.reasoning_tokens),
            tool_input_tokens: None,
        });
    }
    let verdict = AdapterVerdict {
        finish_reason: object
            .status
            .filter(|status| status != "in_progress" && status != "queued")
            .map(|value| sink.scrub(&value)),
        block_reason: None,
        detail: object
            .incomplete_details
            .and_then(|details| details.reason)
            .map(|value| sink.scrub(&value)),
        model: object.model.map(|value| sink.scrub(&value)),
    };
    let response_id = object.id.map(|value| sink.scrub(&value));
    sink.provider(verdict, response_id);
    if let Some(error) = object.error {
        error.emit(sink);
    }
}

#[derive(Default, Deserialize)]
struct PayloadTokenDetails {
    #[serde(default, deserialize_with = "payload_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "payload_count")]
    reasoning_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct PayloadUsage {
    #[serde(default, deserialize_with = "payload_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "payload_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "payload_count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<PayloadTokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<PayloadTokenDetails>,
}

fn payload_count<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u64>, D::Error> {
    Ok(Value::deserialize(deserializer)?.as_u64())
}

#[derive(Deserialize)]
struct PayloadIncompleteDetails {
    reason: Option<String>,
}

#[derive(Deserialize)]
struct PayloadError {
    code: Option<Value>,
    #[serde(rename = "type", alias = "status")]
    kind: Option<String>,
    message: Option<String>,
}

impl PayloadError {
    fn emit(self, sink: &mut ObservationSink<'_>) {
        ObservedError {
            code: self.code,
            kind: self.kind,
            message: self.message,
        }
        .emit(sink);
    }
}

/// Private observation read shape; malformed known fields abort projection.
#[derive(Default, Deserialize)]
struct PayloadResponseObject {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<PayloadIncompleteDetails>,
    usage: Option<PayloadUsage>,
    error: Option<PayloadError>,
}

#[derive(Deserialize)]
struct Payload {
    #[serde(rename = "type")]
    kind: Option<String>,
    response: Option<PayloadResponseObject>,
    #[serde(flatten)]
    unwrapped: PayloadResponseObject,
    code: Option<Value>,
    message: Option<String>,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod carried_tests;
