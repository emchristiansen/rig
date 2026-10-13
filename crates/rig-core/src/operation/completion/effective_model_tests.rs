use super::Completion;
use crate::Model;
use crate::completion::CompletionRequest;
use crate::driver::decode_with;
use crate::error::ProviderError;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Call, Mode, Operation, Wire};

const NOVA: &str = "amazon.nova-pro-v1:0";
const CLAUDE: &str = "anthropic.claude-3-5-sonnet-20241022-v2:0";

fn getter_from_model_fold(configured: &str, requested: Option<&str>, expected: &str) {
    // The real Model holds the configuration; its transport is never opened.
    // No Bedrock/Gemini helper acceptance or issuer policy is exercised here.
    let model = Model::new(OpenAIConfig::new("test-key").responses(configured), ());
    let descriptor = model.wire.describe();
    assert_eq!(descriptor.model, Some(configured));
    for mode in [Mode::Unary, Mode::Streaming] {
        let mut request = CompletionRequest::new("question");
        request.model = requested.map(str::to_owned);
        let prepared = Completion::prepare(request, &descriptor).expect("actual request preflight");
        assert_eq!(prepared.model.as_deref(), Some(expected));
        let mut call = Call::new(&descriptor, mode);
        let fold = Completion::fold(&prepared, &mut call);
        let decoded = decode_with::<Completion>(fold, descriptor.name, |reply| {
            let out = reply.out();
            let before = out.lock().fold.origin.clone();
            let issuer = out.lock().fold.carried_issuer.clone();
            assert_eq!(out.effective_model(), expected);
            let mut owned = out.effective_model();
            owned.push_str("caller-owned edit");
            assert_ne!(owned, expected);
            assert_eq!(out.effective_model(), expected, "editing the owned clone cannot change the fold");
            let shared = out.lock();
            assert_eq!(shared.fold.origin, before);
            assert_eq!(shared.fold.carried_issuer, issuer);
            assert!(shared.items.is_empty(), "reading the model emits no event or provider item");
            Err(ProviderError::Truncated)
        });
        assert!(matches!(decoded.outcome, Err(ProviderError::Truncated)), "no response terminal is fabricated");
        assert!(decoded.items.iter().all(Result::is_err), "the getter yields no stream event");
    }
}

#[test]
fn configured_nova_request_claude_reads_the_request_override() {
    getter_from_model_fold(NOVA, Some(CLAUDE), CLAUDE);
}

#[test]
fn configured_claude_request_nova_reads_the_inverse_request_override() {
    getter_from_model_fold(CLAUDE, Some(NOVA), NOVA);
}

#[test]
fn ordinary_configured_model_and_identical_override_keep_the_exact_existing_model() {
    for configured in ["ordinary-model", "exact-model spelling/with spaces"] {
        getter_from_model_fold(configured, None, configured);
        getter_from_model_fold(configured, Some(configured), configured);
    }
}
