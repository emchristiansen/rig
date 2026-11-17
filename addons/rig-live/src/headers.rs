//! The request and reply headers both protocols share: the provider's
//! static credential, its send-time credential placement, and the reply's
//! request id.

use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::wire::Auth;

use crate::support::{
    CredentialPlacement, CredentialSourceHandle, CredentialStamp, TokenPlacement,
};

/// The header `Auth::ApiKeyHeader` sends the credential in.
const API_KEY_HEADER: &str = "api-key";

/// The header naming the account a credential belongs to.
pub(crate) const ACCOUNT_ID_HEADER: &str = "chatgpt-account-id";

/// Add `config`'s static credential with the header its [`Auth`] names.
pub(crate) fn authenticate(
    config: &OpenAIConfig,
    builder: http::request::Builder,
) -> http::request::Builder {
    match config.auth {
        Auth::Bearer => builder.header(
            http::header::AUTHORIZATION,
            format!("Bearer {}", config.api_key.expose()),
        ),
        Auth::OptionalBearer if config.api_key.is_empty() => builder,
        Auth::OptionalBearer => builder.header(
            http::header::AUTHORIZATION,
            format!("Bearer {}", config.api_key.expose()),
        ),
        Auth::ApiKeyHeader => builder.header(API_KEY_HEADER, config.api_key.expose()),
    }
}

/// The send-time credential of `source` for `config`: its token goes in the
/// header `config`'s [`Auth`] names, and it is authoritative for
/// `ChatGPT-Account-Id`.
pub(crate) fn credential_stamp(
    config: &OpenAIConfig,
    source: &CredentialSourceHandle,
) -> CredentialStamp {
    CredentialStamp {
        source: source.clone(),
        placement: CredentialPlacement {
            token: match config.auth {
                Auth::Bearer => TokenPlacement::Bearer,
                Auth::OptionalBearer => TokenPlacement::OptionalBearer,
                Auth::ApiKeyHeader => {
                    TokenPlacement::Header(http::HeaderName::from_static(API_KEY_HEADER))
                }
            },
            account: Some(http::HeaderName::from_static(ACCOUNT_ID_HEADER)),
        },
    }
}

/// The provider's request id, when it names such a header and the reply
/// carries a non-empty value.
pub(crate) fn request_id_from(headers: &http::HeaderMap, header: Option<&str>) -> Option<String> {
    header.and_then(|header| {
        headers
            .get(header)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}
