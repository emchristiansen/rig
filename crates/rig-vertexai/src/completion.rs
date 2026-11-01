//! The Vertex AI `GenerateContent` completion wire and model identifiers.
//! A streamed call re-emits the unary reply: this integration has no
//! streaming RPC.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_vertexai::{VertexAi, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let model = VertexAi::from_env()?.completion(GEMINI_2_5_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use super::VertexAi;
use crate::types::completion_response::{PROVIDER_NAME, VertexDecoder, rest_chunk};
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::{CompletionRequest, Replay};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion as rest;
use rig_core::wire::{Descriptor, Mode, Wire};

pub use rig_core::providers::gemini::completion::GEMINI_2_5_FLASH;

/// The `GenerateContent` endpoint for one model.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateContent {
    pub model: String,
}

impl GenerateContent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    /// The SDK request. Its `model` is the model id the request addresses;
    /// the transport qualifies it with the project and location.
    type Payload = vertexai::model::GenerateContentRequest;
    type Frame = vertexai::model::GenerateContentResponse;
    type Decoder<'id> = VertexDecoder;
    type Reassembler = crate::types::completion_response::VertexDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    /// The REST wire's request, read into the SDK's types: every field the
    /// shared encoder builds reaches Vertex AI. The SDK's spellings of the
    /// contents and its `model` key are part of the wire's own encoding, so
    /// `additional_params` merges over them.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<vertexai::model::GenerateContentRequest, EncodeError> {
        if request
            .temperature
            .is_some_and(|temperature| !temperature.is_finite())
        {
            return Err(EncodeError::request("Vertex AI temperature must be finite"));
        }
        check_original_assistant_images(&request, self)?;
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let body = rest::request_body_for_vertex(&request, self, &model, None, |body| {
            if let Some(system) = body
                .get_mut("systemInstruction")
                .and_then(serde_json::Value::as_object_mut)
            {
                system.insert("role".to_owned(), "user".into());
            }
            if !request.tools.is_empty() && request.tool_choice.is_none() {
                body.insert(
                    "toolConfig".to_owned(),
                    serde_json::json!({"functionCallingConfig": {"mode": "AUTO"}}),
                );
            }
            let contents = body
                .get_mut("contents")
                .and_then(serde_json::Value::as_array_mut);
            let mut images = 0;
            for content in contents.into_iter().flatten() {
                standard_signatures(content);
                referenced_media(content, &mut images);
            }
            body.insert("model".to_owned(), model.clone().into());
        })?;
        let encoded: vertexai::model::GenerateContentRequest = body.deserialize()?;
        if let Some(config) = &encoded.generation_config {
            if config
                .response_modalities
                .contains(&vertexai::model::generation_config::Modality::Audio)
            {
                return Err(EncodeError::request(
                    "responseModalities AUDIO is unsupported because Rig cannot represent assistant audio responses",
                ));
            }
            if let Some(thinking) = &config.thinking_config
                && thinking.thinking_budget.is_some()
                && thinking.thinking_level.is_some()
            {
                return Err(EncodeError::request(
                    "thinking_budget and thinking_level cannot both be set",
                ));
            }
        }
        Ok(encoded)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VertexDecoder::default()
    }
}

/// Check only caller/rebuilt assistant images. A current Native item's exact
/// received bytes remain under the existing replay custody.
fn check_original_assistant_images(
    request: &CompletionRequest,
    target: &GenerateContent,
) -> Result<(), EncodeError> {
    for message in &request.chat_history {
        if let rig_core::message::Message::Assistant(turn) = message {
            for block in &turn.content {
                if let rig_core::message::AssistantContent::Image(image) = block
                    && !matches!(block.replay(target, None), Replay::Item(_))
                {
                    check_assistant_image(image)?;
                }
            }
        }
    }
    Ok(())
}

/// Vertex's private canonical assistant-image form is typed inline base64.
fn check_assistant_image(image: &rig_core::message::Image) -> Result<(), EncodeError> {
    use rig_core::message::{DocumentSourceKind, ImageMediaType};
    let media_type = image.media_type.as_ref().ok_or_else(|| {
        EncodeError::request("Media type for assistant image is required for Vertex AI")
    })?;
    if !matches!(
        media_type,
        ImageMediaType::JPEG
            | ImageMediaType::PNG
            | ImageMediaType::WEBP
            | ImageMediaType::HEIC
            | ImageMediaType::HEIF
    ) {
        return Err(EncodeError::request(format!(
            "Unsupported Vertex AI assistant image media type {media_type:?}"
        )));
    }
    let DocumentSourceKind::Base64(data) = &image.data else {
        return Err(EncodeError::request(
            "Vertex AI assistant images must use base64 data",
        ));
    };
    BASE64.decode(data).map(|_| ()).map_err(|error| {
        EncodeError::request(format!("Invalid base64 assistant image data: {error}"))
    })
}

/// Respell every thought signature in `content` in standard base64, the
/// only alphabet the SDK's byte fields read. Google's placeholder signature
/// is URL-safe base64; Vertex AI reads the bytes either spelling decodes
/// to, so both name the same signature.
fn standard_signatures(content: &mut serde_json::Value) {
    let parts = content
        .get_mut("parts")
        .and_then(serde_json::Value::as_array_mut);
    for part in parts.into_iter().flatten() {
        let invalid = part
            .get("thoughtSignature")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|text| BASE64.decode(text).is_err() && URL_SAFE.decode(text).is_err());
        if invalid {
            if let Some(fields) = part.as_object_mut() {
                fields.shift_remove("thoughtSignature");
            }
        } else if let Some(signature) = part.get_mut("thoughtSignature")
            && let Some(text) = signature.as_str()
            && BASE64.decode(text).is_err()
            && let Ok(bytes) = URL_SAFE.decode(text)
        {
            *signature = serde_json::Value::String(BASE64.encode(bytes));
        }
    }
}

/// Give every function response in `content` that states no `response` one
/// naming its media parts, which Vertex AI requires: each part gets a
/// `displayName` and the response refers to it by `$ref`, as Google's
/// multimodal function responses spell it. `images` numbers the names
/// across the request.
fn referenced_media(content: &mut serde_json::Value, images: &mut usize) {
    let parts = content
        .get_mut("parts")
        .and_then(serde_json::Value::as_array_mut);
    for part in parts.into_iter().flatten() {
        let Some(response) = part
            .get_mut("functionResponse")
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };
        if response.contains_key("response") {
            continue;
        }
        let mut refs = Vec::new();
        let media = response
            .get_mut("parts")
            .and_then(serde_json::Value::as_array_mut);
        for media in media.into_iter().flatten() {
            let key = ["inlineData", "fileData"]
                .into_iter()
                .find(|key| media.get(*key).is_some());
            let blob = key
                .and_then(|key| media.get_mut(key))
                .and_then(serde_json::Value::as_object_mut);
            if let Some(blob) = blob {
                let name = format!("rig_tool_result_image_{images}");
                *images += 1;
                blob.insert("displayName".to_owned(), name.clone().into());
                refs.push(serde_json::json!({ "$ref": name }));
            }
        }
        let output = match refs.len() {
            1 => refs.pop().unwrap_or_default(),
            _ => serde_json::Value::Array(refs),
        };
        response.insert(
            "response".to_owned(),
            serde_json::json!({ "output": output }),
        );
    }
}

impl rig_core::completion::ReplayTarget for GenerateContent {
    /// The GenerateContent mapping, for Vertex AI's tiers.
    fn map_options(
        &self,
        request: &rig_core::completion::CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        rest::generate_content_options(model, rest::Route::Vertex, fields)
    }

    // There is no dedicated assistant-image preflight hook. Use this existing
    // hook after unchanged original-document validation, before adaptation.
    fn check_documents(&self, request: &CompletionRequest) -> Result<(), EncodeError> {
        for message in &request.chat_history {
            if let rig_core::message::Message::User { content } = message {
                for part in content {
                    if let rig_core::message::UserContent::Document(document) = part {
                        self.check_document(document)?;
                    }
                }
            }
        }
        check_original_assistant_images(request, self)
    }

    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("vertexai.generate_content")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// What the model reads, as on every GenerateContent wire.
    fn accepts(&self, model: &str) -> rig_core::completion::Accepts {
        rest::accepts(model)
    }

    /// The media Vertex AI takes: what the Gemini API takes, and image URLs inside
    /// function responses too.
    fn encodes(&self, _model: &str, media: rig_core::completion::Media<'_>) -> bool {
        rest::encodes(media, true)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&rig_core::message::Origin>,
    ) -> String {
        rest::normalize_tool_call_id(model, id)
    }

    /// Gemini takes system text only in `systemInstruction`: later system
    /// messages fold into the leading one, as pi's `collapseSystemMessages`.
    fn later_system(&self, _model: &str) -> rig_core::completion::LaterSystem {
        rig_core::completion::LaterSystem::Leading
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        rest::CALL_ID_SLOT
    }

    /// Tools in `additional_params` or a cached content count, as on the
    /// REST wire.
    fn declares_tools(&self, request: &rig_core::completion::CompletionRequest) -> bool {
        rest::declares_tools(self, request)
    }
}

/// Both modes send the unary RPC; a streamed call re-emits its reply.
impl Transport<GenerateContent> for VertexAi {
    fn send(
        &self,
        mut request: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        request.model = format!(
            "projects/{}/locations/{}/publishers/google/models/{}",
            self.project(),
            self.location(),
            request.model
        );
        let client = self.clone();
        Opening::new(async move {
            let service = match client.inner().await {
                Ok(service) => service,
                Err(error) => return Err(ProviderError::request(error)),
            };
            match service
                .generate_content()
                .with_request(request)
                .send()
                .await
            {
                Ok(response) => {
                    tracing::debug!(
                        target: "rig_core::vertexai",
                        "Vertex AI completion response: {response:?}"
                    );
                    // The reply's REST JSON is the response's `raw`, in both
                    // modes. One that does not transcode fails as its
                    // decoding would.
                    let document = serde_json::Value::Object(rest_chunk(&response)?);
                    Ok(Opened::new(futures::stream::iter([Ok(response)])).with_document(document))
                }
                Err(error) => Ok(Opened::failed(rpc_error(&error))),
            }
        })
    }
}

/// Preserves SDK error display text, HTTP status, RPC code, and retry hints.
/// Transport errors also use the provider-body representation.
fn rpc_error(error: &google_cloud_aiplatform_v1::Error) -> ProviderError {
    let status = error
        .http_status_code()
        .and_then(|code| rig_core::http_client::StatusCode::from_u16(code).ok());
    let code = error.status().map(|status| status.code.name());
    // SDK transport classifications supply retry hints when no RPC code
    // exists; an RPC code decides through the one provider code table.
    let transient = (code.is_none()
        && (error.is_transport() || error.is_io() || error.is_timeout() || error.is_connect()))
    .then_some(true);
    ProviderError::from_provider_body(error.to_string())
        .with_provider_status(status)
        .with_provider_code(code.map(str::to_owned))
        .with_transient(transient)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

#[cfg(test)]
mod reasoning_guard_tests;

#[cfg(test)]
mod port_mechanical_tests;

#[cfg(test)]
mod port_controls_tests;

#[cfg(test)]
mod port_media_tests;
