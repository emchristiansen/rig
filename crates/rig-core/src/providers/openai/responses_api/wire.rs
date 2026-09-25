//! Responses endpoint encoding and observation for configured OpenAI dialects.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").responses("gpt-5.2");
//! ```

use crate::completion::{self, ProviderCapabilities};
use crate::error::EncodeError;
use crate::observe::ObservedError;
use crate::operation::Completion;
use crate::providers::openai::wire::{OpenAI, ResponsesContract};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Encoded, Framing, Mode, ObservationSink, Wire,
};
use serde::{Deserialize, Serialize};

use super::streaming::{IncompleteTerminal, ResponsesDecoder, ResponsesStreamOptions};
use super::{
    CompletionRequest, ResponsesRequestParams, ResponsesToolDefinition, SystemInstructionsPlacement,
};

/// The Responses wire: `POST /responses`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Responses {
    /// The provider this wire speaks to.
    pub provider: OpenAI,
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
}

impl Responses {
    pub(crate) fn encode_with_headers(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
        headers: impl FnOnce(
            &OpenAI,
            &completion::CompletionRequest,
            http::request::Builder,
        ) -> http::request::Builder,
    ) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks.responses;
        // The codex gateway only ever answers with an event stream, and
        // names no content type on it. It is asked for one whatever the
        // caller wanted: the reply is framed the same way either way, and
        // the driver folds it.
        let codex = quirks.contract == ResponsesContract::Codex;
        let streaming = matches!(mode, Mode::Streaming) || codex;
        let builder = headers(
            &self.provider,
            &request,
            http::Request::post(self.provider.uri(quirks.path, None)),
        );
        let request = self.responses_request(request, streaming)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Responses completion request",
            &request,
        );
        let body = serde_json::to_vec(&request)?;

        let request = builder
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(body))?;

        let framing = if streaming {
            Framing::Sse
        } else {
            Framing::Whole
        };
        let encoded = Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header);
        Ok(if codex {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    /// Create a wire with the provider's instruction placement and dialect's
    /// strict-tool default, without additional tools.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            system_instructions: provider.system_instructions_placement(),
            strict_tools: provider.dialect.quirks.responses.strict_tools_by_default,
            provider,
            model: model.into(),
            tools: Vec::new(),
            streamed_incomplete: IncompleteTerminal::default(),
        }
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

    /// The Responses request this wire sends, before serialization.
    pub(crate) fn responses_request(
        &self,
        request: completion::CompletionRequest,
        streaming: bool,
    ) -> Result<CompletionRequest, EncodeError> {
        let quirks = &self.provider.dialect.quirks.responses;
        let mut request = CompletionRequest::try_from(ResponsesRequestParams {
            model: self.model.clone(),
            request,
            system_instructions_placement: self.system_instructions,
        })?;
        request.tools.extend(self.tools.clone());
        if self.strict_tools {
            request.tools = request
                .tools
                .into_iter()
                .map(ResponsesToolDefinition::normalize)
                .collect();
        }
        if let Some(instructions) = &self.provider.instructions {
            request.instructions = Some(merge_instructions(
                instructions,
                request.instructions.as_deref(),
            ));
        }
        if quirks.contract == ResponsesContract::Codex {
            shape_codex_request(&mut request).map_err(EncodeError::request)?;
        }
        request.stream = streaming.then_some(true);
        Ok(request)
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

/// Shape a request for the Codex contract: state `store: false` when the
/// caller left it unset, and refuse by name each caller-set control this
/// adapter does not send on the contract ([`UnsupportedCodexControl`]).
///
/// Nothing else is rewritten. Cache identity (`prompt_cache_key`,
/// `client_metadata`), metadata, service tier, text format, parallel tool
/// calls and user reach the wire as set, and the encrypted-reasoning include
/// follows the rule every dialect shares (requested alongside `reasoning`)
/// rather than being forced on. An unset refused control emits no field, so a
/// request that sets none of them is sent as built.
fn shape_codex_request(request: &mut CompletionRequest) -> Result<(), UnsupportedCodexControl> {
    if request.max_output_tokens.is_some() {
        return Err(UnsupportedCodexControl::MaxOutputTokens);
    }
    if request.temperature.is_some() {
        return Err(UnsupportedCodexControl::Temperature);
    }
    if request.additional_parameters.top_p.is_some() {
        return Err(UnsupportedCodexControl::TopP);
    }
    match request.additional_parameters.store {
        Some(true) => return Err(UnsupportedCodexControl::Store),
        Some(false) => {}
        // `store: false` is the one value the gateway wants stated.
        None => request.additional_parameters.store = Some(false),
    }
    Ok(())
}

/// Merge a gateway's own instructions ahead of the caller's preamble,
/// without repeating them when the preamble already carries them.
fn merge_instructions(instructions: &str, existing: Option<&str>) -> String {
    match existing.map(str::trim).filter(|value| !value.is_empty()) {
        Some(existing) if existing.contains(instructions) => existing.to_owned(),
        Some(existing) => format!("{instructions}\n\n{existing}"),
        None => instructions.to_owned(),
    }
}

impl Wire for Responses {
    type Op = Completion;
    type Decoder = ResponsesDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn replay_issuers(&self, model: Option<&str>) -> Vec<String> {
        crate::providers::openai::wire::replay_issuers(
            &self.provider.dialect,
            model.unwrap_or(&self.model),
        )
    }

    fn route(&self) -> Option<&str> {
        Some(self.provider.dialect.quirks.responses.path)
    }

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAI::completion_headers)
    }

    fn decoder(&self, mode: Mode) -> ResponsesDecoder {
        // The contract is per transport: a unary reply (including a gateway
        // that answers a unary call with an event stream) accepts an
        // incomplete terminal; a streamed one takes the wire's policy.
        let incomplete = match mode {
            Mode::Unary => IncompleteTerminal::Accept,
            Mode::Streaming => self.streamed_incomplete,
        };
        self.decoder_with_incomplete(incomplete)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // The xAI contract does not compose native structured output with tools.
        ProviderCapabilities::default().with_native_output_tool_composition(
            self.provider.dialect.quirks.responses.contract != ResponsesContract::Xai,
        )
    }
}

/// Normalize a whole Responses body through the decoder and completion fold.
/// Return serialization, decoder, or fold errors without performing I/O.
#[cfg(any(test, feature = "websocket"))]
pub(crate) fn fold_body(
    provider: &str,
    response: super::CompletionResponse,
) -> Result<completion::CompletionResponse, crate::error::ProviderError> {
    use super::streaming::ResponsesEvent;
    use crate::operation::AdapterOutput;
    use crate::wire::{Decoder, Fold, Operation, Reply, Sink};

    let reply = Reply {
        provider: provider.to_owned(),
        raw: serde_json::to_value(&response)?,
        provider_request_id: response.provider_request_id.clone(),
    };
    // A whole body is a unary reply, whose contract accepts an incomplete
    // terminal.
    let mut decoder = ResponsesDecoder::new(
        provider,
        ResponsesStreamOptions::strict().with_incomplete(IncompleteTerminal::Accept),
    );
    let mut out = AdapterOutput::new();
    decoder.interpret(ResponsesEvent::Whole(Box::new(response)), &mut out);

    let mut fold = <Completion as Operation>::Fold::default();
    for item in Sink::<Completion>::drain(&mut out) {
        fold.absorb(item?)?;
    }
    fold.finish(reply)
}

#[derive(Default, Deserialize)]
struct TokenDetails {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    reasoning_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<TokenDetails>,
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: Option<String>,
}

/// The response object, whether it arrived nested under a stream event's
/// `response` or as the unary reply itself. Every field is optional: this is
/// the observation parse, and a payload that carries none of them projects
/// nothing rather than failing.
#[derive(Default, Deserialize)]
struct ResponseObject {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<Usage>,
    error: Option<ObservedError>,
}

#[derive(Deserialize)]
struct Payload {
    #[serde(rename = "type")]
    kind: Option<String>,
    response: Option<ResponseObject>,
    /// The unary reply *is* the response object, so the same fields are
    /// read at the top level rather than declared a second time.
    #[serde(flatten)]
    unwrapped: ResponseObject,
    // The stream `error` event's own fields.
    code: Option<serde_json::Value>,
    message: Option<String>,
}

/// The facts a Responses payload carries before normalization discards
/// them: the verdict, the model, the response id, the usage and any error
/// envelope.
///
/// The unary reply is the response object itself; a stream event wraps that
/// object under `response` (`response.created`, `.completed`, `.failed`,
/// `.incomplete`) or, for `error`, carries the envelope's fields itself.
pub(crate) fn project_payload(payload: &[u8], sink: &mut dyn ObservationSink) {
    let Ok(payload) = serde_json::from_slice::<Payload>(payload) else {
        return;
    };
    if payload.kind.as_deref() == Some("error") {
        // The event carries its envelope either nested under `error` or as
        // its own top-level fields; the nested form names the error type.
        payload
            .unwrapped
            .error
            .unwrap_or(ObservedError {
                code: payload.code,
                kind: None,
                message: payload.message,
            })
            .emit(sink);
        return;
    }
    let object = payload.response.unwrap_or(payload.unwrapped);
    if let Some(usage) = object.usage {
        sink.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage.input_tokens_details.and_then(|d| d.cached_tokens),
                reasoning_tokens: usage.output_tokens_details.and_then(|d| d.reasoning_tokens),
                tool_input_tokens: None,
            },
        });
    }
    // `status` is the provider's verdict; `in_progress` on a stream's
    // opening event is not one yet, so it is left out of the projection.
    let finish_reason = object
        .status
        .filter(|status| status != "in_progress" && status != "queued");
    let verdict = AdapterVerdict {
        finish_reason: finish_reason.map(|value| sink.scrub(&value)),
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

#[cfg(test)]
mod tests;
