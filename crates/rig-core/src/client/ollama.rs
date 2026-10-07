//! The Ollama client: an [`OllamaConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;

use crate::providers::ollama::{self, Embeddings, OllamaConfig};
use crate::providers::openai::wire::Chat;

http_client!(
    /// An Ollama daemon: its [`OllamaConfig`] on a transport. Every model it
    /// builds sends through that transport.
    Ollama,
    OllamaConfig
);

impl Ollama {
    /// The local daemon, unauthenticated, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    #[allow(
        clippy::new_without_default,
        reason = "the configuration is the default; a client also names its transport"
    )]
    pub fn new() -> Self {
        OllamaConfig::new().client()
    }

    /// The daemon `OLLAMA_API_BASE_URL` names, with the token
    /// `OLLAMA_API_KEY` carries, on the shared reqwest client. Both are
    /// optional.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(OllamaConfig::from_env()?.client())
    }

    /// The chat model for `model`, on the daemon's native `/api/chat`.
    /// It takes `think`, `keep_alive` and model `options` such as `num_ctx`.
    pub fn completion(&self, model: impl Into<String>) -> Model<ollama::Chat> {
        self.model(self.config.completion(model))
    }

    /// The explicit OpenAI-compatible `/v1/chat/completions` model.
    /// It takes `keep_alive` and `reasoning_effort`; native `num_ctx` and
    /// `options` are refused as on the upstream compatibility route.
    pub fn compatible_completion(&self, model: impl Into<String>) -> Model<Chat> {
        self.model(self.config.compatible_completion(model))
    }

    /// The chat model for `model`, on the daemon's native `/api/chat`, which
    /// takes `think`, `keep_alive` and model `options` such as `num_ctx`.
    pub fn native_completion(&self, model: impl Into<String>) -> Model<ollama::Chat> {
        self.model(self.config.native_completion(model))
    }

    /// The embedding model for `model`. `ndims` is the width it reports,
    /// defaulting to the model's known width.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The models the daemon serves.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).list().await
    }
}
