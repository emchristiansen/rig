//! Adversarial handle round-trips on Gemini: see
//! `rig_test_support::history_survival::adversarial`.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::request_witness_support::{
    NamedCase, model_result, with_gemini_request_witness_cassette,
};
use crate::history_survival::adversarial;
use crate::history_survival::adversarial::Hop;

/// Second foreign hop: Anthropic then Responses history continues on Gemini.
///
/// Exact named mismatch witness (mec-rig-132-dispositions-1 item 5,
/// mec-rig-schema-route-1 item 1(b)): the recording is unchanged; the hop's
/// request is the shared driver's (`adversarial::request` over
/// `Hop::Gemini.history()`, no parameters, 4096 tokens), and its only
/// difference is the one named in `NamedCase::ThreeProviderRoundTrip`;
/// replay refuses interaction 0. The recorded round-trip rule
/// (`assert_round_trip_recorded`) still runs on the fixture. Named loss: the
/// Gemini hop answering with non-empty text.
#[tokio::test]
async fn three_provider_round_trip() -> anyhow::Result<()> {
    with_gemini_request_witness_cassette(
        "adversarial/three_provider_round_trip",
        NamedCase::ThreeProviderRoundTrip,
        |client, _clock| async move {
            let request = adversarial::request(Hop::Gemini.history(), None, 4096);
            let model: rig::DynModel<rig::operation::Completion> =
                client.completion(GEMINI_3_FLASH_PREVIEW).into();
            Ok(model_result(model.call(request).await))
        },
    )
    .await?;
    adversarial::assert_round_trip_recorded(Hop::Gemini);
    Ok(())
}
