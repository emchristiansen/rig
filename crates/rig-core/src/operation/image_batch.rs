//! Cohere image embedding batches over single-image exchanges.
//!
//! Every image is validated and encoded before the first request opens.
//! Replies are read in input order, and the first failure returns no batch
//! response and prevents every later send.

use crate::driver::{Model, Transport};
use crate::embeddings::EmbeddingResponse;
use crate::error::{EncodeError, ProviderError};
use crate::observe::AdapterContext;
use crate::providers::cohere::ImageEmbeddings;
use crate::wasm_compat::WasmCompatSend;
use crate::wire::Mode;

/// A nonempty ordered batch of image file bytes for Cohere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CohereImageBatch {
    first: Vec<u8>,
    rest: Vec<Vec<u8>>,
}

impl CohereImageBatch {
    /// A batch beginning with `first`, followed by `rest` in order.
    /// Image formats and sizes are checked before any request is sent.
    pub fn new(first: Vec<u8>, rest: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            first,
            rest: rest.into_iter().collect(),
        }
    }
}

impl TryFrom<Vec<Vec<u8>>> for CohereImageBatch {
    type Error = EncodeError;

    fn try_from(images: Vec<Vec<u8>>) -> Result<Self, Self::Error> {
        let mut images = images.into_iter();
        let first = images.next().ok_or_else(|| {
            EncodeError::request("Cohere image embedding batches require at least one image")
        })?;
        Ok(Self::new(first, images))
    }
}

impl<T> Model<ImageEmbeddings, T>
where
    T: Transport<ImageEmbeddings>,
{
    /// Embed every image after validating and encoding the whole batch.
    /// The first failed exchange returns its error and stops all later sends.
    pub fn embed_images_response(
        &self,
        batch: CohereImageBatch,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, ProviderError>>
    + WasmCompatSend
    + 'static {
        self.image_batch(batch, None)
    }

    /// [`Self::embed_images_response`], observing each single-image exchange.
    pub fn embed_images_response_observed(
        &self,
        batch: CohereImageBatch,
        observation: AdapterContext,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, ProviderError>>
    + WasmCompatSend
    + 'static {
        self.image_batch(batch, Some(observation))
    }

    fn image_batch(
        &self,
        batch: CohereImageBatch,
        observation: Option<AdapterContext>,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, ProviderError>>
    + WasmCompatSend
    + 'static {
        let model = self.clone();
        async move {
            let first = model.prepare_exchange(vec![batch.first], Mode::Unary)?;
            let rest = batch
                .rest
                .into_iter()
                .map(|image| model.prepare_exchange(vec![image], Mode::Unary))
                .collect::<Result<Vec<_>, _>>()?;
            let mut response = model
                .open_prepared(first, observation.clone())
                .finish()
                .await?;
            let mut pages = vec![std::mem::take(&mut response.raw)];
            for exchange in rest {
                let page = model
                    .open_prepared(exchange, observation.clone())
                    .finish()
                    .await?;
                response.embeddings.extend(page.embeddings);
                response.usage += page.usage;
                response.provider_request_id = page.provider_request_id;
                pages.push(page.raw);
            }
            response.raw = if pages.len() == 1 {
                pages.remove(0)
            } else {
                serde_json::Value::Array(pages)
            };
            Ok(response)
        }
    }
}

#[cfg(test)]
mod tests;
