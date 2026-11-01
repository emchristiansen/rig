//! Every bank reply through its provider's real decoder. A runtime scenario
//! runs once, on one wire, so the decode of each provider's reply shapes is
//! pinned here instead: each reply is decoded by the model of its provider
//! and encoder, and what the decoder made of it must agree with what the
//! bank read off the bytes: the tools it calls, how the turn ended, and a
//! status error for a rejection. Bedrock answers through the AWS SDK, not
//! an HTTP client the bank can stand in for, and has no decoder here.
//!
//! Every reply of a provider `sweeps!` names must decode. The coverage gate
//! reads that list and counts those replies as recordings of their reply
//! shapes, so a cassette holding only a decoded reply shape can go.
//! OpenRouter-origin replies run separately, including its Messages gateway
//! recording stored in the Anthropic bank.
//!
//! A reply the port refuses by contract is named in [`REFUSED`] with its
//! exact refusal; it must be refused with exactly that, and any other reply
//! that is refused still disagrees.

use rig::completion::{AssistantContent, FinishReason};
use rig::message::EMPTY_RESPONSE_ERROR;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::providers::cohere::CohereConfig;
use rig::providers::copilot::CopilotConfig;
use rig::providers::gemini::GeminiConfig;
use rig::providers::ollama::OllamaConfig;
use rig::providers::openai::wire::{
    DEEPSEEK, DOUBLEWORD, Dialect, GROQ, LLAMACPP, MISTRAL, OPENROUTER, OpenAIConfig, PERPLEXITY,
    VENICE,
};
use rig::{ProviderError, ProviderResponseError};
use rig_test_support::bank::{self, Ending, Entry};
use rig_test_support::cassette_models::{
    AnthropicModels, CohereModels, CopilotModels, GeminiModels, OllamaModels,
};

const KEY: &str = "bank";
const BASE_URL: &str = "http://bank.invalid";

/// What the decoder made of one reply.
#[derive(Debug)]
pub(crate) struct Decoded {
    pub(crate) finish: Option<FinishReason>,
    pub(crate) text: String,
    /// The distinct names of the tools the turn calls, sorted.
    pub(crate) calls: Vec<String>,
    /// The decoder's error, when it refused the reply.
    pub(crate) error: Option<ProviderError>,
    /// The response's raw provider document; `Null` when it was refused.
    pub(crate) raw: serde_json::Value,
}

async fn call<W, T>(model: rig::driver::Model<W, T>, streamed: bool) -> Decoded
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let response = if streamed {
        match model.stream("Reply.") {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        }
    } else {
        model.call("Reply.").await
    };
    match response {
        Ok(response) => {
            let mut calls: Vec<String> = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => {
                        Some(call.function.name.as_str().to_owned())
                    }
                    _ => None,
                })
                .collect();
            calls.sort();
            calls.dedup();
            Decoded {
                finish: response.finish_reason(),
                text: response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect(),
                calls,
                error: None,
                raw: response.raw,
            }
        }
        Err(error) => Decoded {
            finish: None,
            text: String::new(),
            calls: Vec::new(),
            error: Some(error),
            raw: serde_json::Value::Null,
        },
    }
}

pub(crate) fn openai_family(provider: &str) -> Option<OpenAIConfig> {
    let dialect: Option<&Dialect> = match provider {
        "deepseek" => Some(&DEEPSEEK),
        "doubleword" => Some(&DOUBLEWORD),
        "groq" => Some(&GROQ),
        "llamacpp" => Some(&LLAMACPP),
        "mistral" => Some(&MISTRAL),
        "openrouter" => Some(&OPENROUTER),
        "perplexity" => Some(&PERPLEXITY),
        "venice" => Some(&VENICE),
        "xai" => Some(&rig::providers::xai::DIALECT),
        "chatgpt" => Some(&rig::providers::chatgpt::DIALECT),
        "openai" | "mistralrs" => None,
        _ => return None,
    };
    let mut config = match dialect {
        Some(dialect) => OpenAIConfig::with_key(dialect, KEY),
        None => OpenAIConfig::new(KEY),
    };
    // The ChatGPT dialect refuses to send without the caller's exact
    // identity; the harness supplies the plainly-test one.
    if provider == "chatgpt" {
        config = config.with_caller_identity(rig::test_utils::test_caller_identity());
    }
    Some(config.with_base_url(BASE_URL))
}

/// Bind `$model` to `$entry`'s provider and encoder model over `$http` and
/// evaluate `$body` with it: `Some` of the body, or `None` when the bank has
/// no model for that provider and encoder. A macro, because each arm's model
/// is of its own type.
macro_rules! with_model {
    ($entry:expr, $http:expr, |$model:ident| $body:expr) => {{
        let entry: &rig_test_support::bank::Entry = $entry;
        let http: rig::http_client::DynHttpClient = $http;
        let encoder = entry.encoder.as_str();
        match entry.provider.as_str() {
            "anthropic" if encoder.ends_with("/messages") => {
                let $model = $crate::decode::anthropic_model(http);
                Some($body)
            }
            "gemini" if encoder.ends_with("/interactions") => {
                let $model =
                    $crate::decode::gemini_model(http).interactions($crate::decode::GEMINI);
                Some($body)
            }
            "gemini" => {
                let $model = $crate::decode::gemini_model(http).completion($crate::decode::GEMINI);
                Some($body)
            }
            "cohere" => {
                let $model = $crate::decode::cohere_model(http);
                Some($body)
            }
            "ollama" if encoder.ends_with("/api/chat") => {
                let $model = $crate::decode::ollama_native_model(http);
                Some($body)
            }
            "ollama" => {
                let $model = $crate::decode::ollama_model(http);
                Some($body)
            }
            "copilot" => {
                let $model = $crate::decode::copilot_model(http, encoder);
                Some($body)
            }
            provider => match $crate::decode::openai_family(provider) {
                Some(config) if encoder.ends_with("/chat/completions") => {
                    let $model = rig_test_support::cassette_models::OpenAiModels::new(config, http)
                        .chat("model");
                    Some($body)
                }
                Some(config) if encoder.ends_with("/responses") => {
                    let $model = rig_test_support::cassette_models::OpenAiModels::new(config, http)
                        .responses("model");
                    Some($body)
                }
                _ => None,
            },
        }
    }};
}
pub(crate) use with_model;

pub(crate) const GEMINI: &str = rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

pub(crate) fn anthropic_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::anthropic::wire::Messages> {
    AnthropicModels::new(AnthropicConfig::new(KEY).with_base_url(BASE_URL), http)
        .completion("claude-haiku-4-5-20251001")
}

pub(crate) fn gemini_model(http: rig::http_client::DynHttpClient) -> GeminiModels {
    GeminiModels::new(GeminiConfig::new(KEY).with_base_url(BASE_URL), http)
}

pub(crate) fn cohere_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::cohere::CohereChat> {
    CohereModels::new(CohereConfig::new(KEY).with_base_url(BASE_URL), http)
        .completion("command-a-03-2025")
}

pub(crate) fn ollama_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::openai::wire::Chat> {
    OllamaModels::new(OllamaConfig::new().with_base_url(BASE_URL), http).completion("qwen3:4b")
}

pub(crate) fn ollama_native_model(
    http: rig::http_client::DynHttpClient,
) -> rig::Model<rig::providers::ollama::Chat> {
    OllamaModels::new(OllamaConfig::new().with_base_url(BASE_URL), http)
        .native_completion("qwen3:4b")
}

pub(crate) fn copilot_model(
    http: rig::http_client::DynHttpClient,
    encoder: &str,
) -> rig::Model<rig::providers::copilot::wire::CopilotWire> {
    let model = if encoder.ends_with("/responses") {
        "gpt-5.3-codex"
    } else {
        "gpt-4o"
    };
    CopilotModels::new(CopilotConfig::new(KEY).with_base_url(BASE_URL), http).completion(model)
}

/// `entry` through its provider's decoder; `None` when the bank has no
/// decoder for its provider and encoder.
pub(crate) async fn decode(entry: &Entry) -> Option<Decoded> {
    let streamed = entry.streamed();
    with_model!(entry, bank::client(std::slice::from_ref(entry)), |model| {
        call(model, streamed).await
    })
}

/// OpenRouter recordings include the Messages gateway stored under Anthropic.
pub(crate) fn openrouter_entry(entry: &Entry) -> bool {
    entry.provider == "openrouter"
        || entry.source.split('#').next()
            == Some("anthropic/streaming/gateway_message_delta_metadata.yaml")
}

/// One function call the port closed with the typed
/// [`DiscardReason::Length`]: the tool its Part started under and the
/// argument fragments it had delivered, joined.
#[derive(Debug, PartialEq, Eq)]
struct Discarded {
    name: String,
    arguments: String,
}

/// The calls `wire` discards with reason Length when it decodes `frames` in
/// `mode`, in stream order. A discard under any other reason is returned as
/// an error: this sweep subtracts only the Chat Length contract.
fn length_discards<W>(
    wire: &W,
    mode: rig::wire::Mode,
    frames: Vec<rig::wire::WireFrame>,
) -> Result<Vec<Discarded>, String>
where
    W: rig::wire::Wire<Op = rig::operation::Completion, Frame = rig::wire::WireFrame>,
{
    use rig::streaming::{DiscardReason, Item, PartKind, StreamEvent};
    let (_, _, items) = rig_core::test_utils::history_conformance::cut(wire, mode, frames);
    let mut names = std::collections::HashMap::new();
    let mut arguments: std::collections::HashMap<_, String> = std::collections::HashMap::new();
    let mut discarded = Vec::new();
    for item in items {
        match item {
            Item::Event(StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name,
                ..
            }) => {
                names.insert(part, name.map(|name| name.as_str().to_owned()));
            }
            Item::Event(StreamEvent::Arguments { part, json }) => {
                arguments.entry(part).or_default().push_str(&json);
            }
            Item::Event(StreamEvent::Discard {
                part,
                reason: DiscardReason::Length,
            }) => {
                let Some(Some(name)) = names.get(&part) else {
                    return Err(format!("a Length discard of unnamed part {part:?}"));
                };
                discarded.push(Discarded {
                    name: name.clone(),
                    arguments: arguments.remove(&part).unwrap_or_default(),
                });
            }
            Item::Event(StreamEvent::Discard { part, reason }) => {
                return Err(format!("part {part:?} discarded with reason {reason:?}"));
            }
            _ => {}
        }
    }
    Ok(discarded)
}

/// The function calls a Chat Completions reply's bytes carry on choice 0, in
/// wire order: each call's name and its argument text, joined across stream
/// fragments by `index`. Custom calls carry no function arguments and are
/// left out.
fn chat_calls(entry: &Entry) -> Vec<(String, String)> {
    let body = entry.body();
    chat_document_calls(&String::from_utf8_lossy(&body), entry.streamed())
}

/// [`chat_calls`] of a Chat body's text: a stream of `data:` chunks when
/// `streamed`, else one `chat.completion` document.
fn chat_document_calls(text: &str, streamed: bool) -> Vec<(String, String)> {
    let mut calls: Vec<(String, String)> = Vec::new();
    let mut absorb = |call: &serde_json::Value, position: usize| {
        let Some(function) = call.get("function") else {
            return;
        };
        if calls.len() <= position {
            calls.resize(position + 1, (String::new(), String::new()));
        }
        let slot = &mut calls[position];
        if let Some(name) = function["name"].as_str()
            && slot.0.is_empty()
        {
            name.clone_into(&mut slot.0);
        }
        if let Some(arguments) = function["arguments"].as_str() {
            slot.1.push_str(arguments);
        }
    };
    let choice_zero = |reply: &serde_json::Value| -> Option<serde_json::Value> {
        reply["choices"]
            .as_array()?
            .iter()
            .find(|choice| choice["index"].as_u64().unwrap_or(0) == 0)
            .cloned()
    };
    if streamed {
        for line in text.lines() {
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let Some(choice) = choice_zero(&chunk) else {
                continue;
            };
            for (order, call) in choice["delta"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let index = call["index"].as_u64().map_or(order, |index| index as usize);
                absorb(call, index);
            }
        }
    } else if let Ok(reply) = serde_json::from_str::<serde_json::Value>(text)
        && let Some(choice) = choice_zero(&reply)
    {
        for (position, call) in choice["message"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            absorb(call, position);
        }
    }
    calls.retain(|(name, _)| !name.is_empty());
    calls
}

/// Whether a Chat call's argument text, as the bytes carry it, was cut:
/// empty, or not a JSON value.
fn cut_arguments(arguments: &str) -> bool {
    arguments.trim().is_empty() || serde_json::from_str::<serde_json::Value>(arguments).is_err()
}

/// The calls a Chat Completions entry ending at Length must decode to, under
/// coord-a-rig-port-discard-15: a call the bytes cut (empty or non-JSON
/// arguments) is discarded and every other call is kept. `Ok(None)` for any
/// other entry, whose calls the bank states as they are. `Err` when what the
/// port discarded is not exactly the bytes' cut calls:
///
/// * a stream closes each discarded call's Part with the typed `Discard`
///   reason Length, its delivered `Arguments` fragments visible;
/// * a unary reply has no event surface: the port drops a cut call from the
///   whole document before decoding it (`prepare_whole_calls`), so no Part,
///   and no typed `Discard`, ever exists for it. Its delivered arguments stay
///   visible only in the response's raw document, which must still carry
///   each cut call verbatim while the decoded calls leave it out. Should a
///   unary decode ever emit typed Length discards, they are held to the
///   stream's standard instead.
async fn calls_after_length_discards(
    entry: &Entry,
    decoded: &Decoded,
) -> Result<Option<Vec<String>>, String> {
    if entry.ending() != Ending::Length || !entry.encoder.ends_with("/chat/completions") {
        return Ok(None);
    }
    let (mode, framing) = if entry.streamed() {
        (rig::wire::Mode::Streaming, rig::wire::Framing::Sse)
    } else {
        (rig::wire::Mode::Unary, rig::wire::Framing::Whole)
    };
    let frames = framing.split(&entry.body());
    let discarded = with_model!(entry, bank::client(std::slice::from_ref(entry)), |model| {
        length_discards(&model.wire, mode, frames)
    })
    .ok_or_else(|| format!("no decoder for `{}`", entry.encoder))??;
    let calls = chat_calls(entry);
    let mut read: Vec<String> = calls.iter().map(|(name, _)| name.clone()).collect();
    read.sort();
    read.dedup();
    if read != entry.calls {
        return Err(format!(
            "the Chat bytes carry calls {read:?}, the bank read {:?}",
            entry.calls
        ));
    }
    let (cut, kept): (Vec<_>, Vec<_>) = calls
        .into_iter()
        .partition(|(_, arguments)| cut_arguments(arguments));
    let cut: Vec<Discarded> = cut
        .into_iter()
        .map(|(name, arguments)| Discarded { name, arguments })
        .collect();
    let discarded = if entry.streamed() || !discarded.is_empty() {
        discarded
    } else {
        let raw = chat_document_calls(&decoded.raw.to_string(), false);
        cut.iter()
            .filter(|call| raw.contains(&(call.name.clone(), call.arguments.clone())))
            .map(|call| Discarded {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect()
    };
    if discarded != cut {
        return Err(format!(
            "Length discards {discarded:?}, the bytes cut {cut:?}"
        ));
    }
    let mut kept: Vec<String> = kept.into_iter().map(|(name, _)| name).collect();
    kept.sort();
    kept.dedup();
    Ok(Some(kept))
}

/// How the port refuses a reply by contract.
#[derive(Debug)]
enum Refusal {
    /// G193 (coord-a-rig-g191-g195-1): a default Responses stream refuses a
    /// terminal `response.incomplete` as the provider's reply. The refusal
    /// is [`ProviderError::ProviderResponse`] whose body is that frame's data,
    /// verbatim, with no status and the reply's request id.
    IncompleteTerminal { request_id: &'static str },
    /// [`ProviderError::Response`] with exactly this text.
    Response(&'static str),
}

/// The replies the port refuses by contract, by bank source, each with its
/// exact refusal:
///
/// * the three Responses streams ending in `response.incomplete` (G193);
/// * Gemini's `MALFORMED_FUNCTION_CALL` finish, a REST tool-protocol finish
///   refused with its finish message (mec-rig-78-four-subjects-1 D,
///   mec-rig-132-dispositions-2);
/// * llama.cpp's empty unary Chat reply, refused with
///   [`EMPTY_RESPONSE_ERROR`] (the listed empty-unary contract).
const REFUSED: &[(&str, Refusal)] = &[
    (
        "openai/reasoning_matrix_responses/capped_streamed.yaml#0",
        Refusal::IncompleteTerminal {
            request_id: "req_754c1faf9b714535a2165baef9aa63e3",
        },
    ),
    (
        "openai/truncated_turn_matrix/responses_streaming_reasoning_budget_exhausted.yaml#0",
        Refusal::IncompleteTerminal {
            request_id: "req_acf96114e4b341969e8df1c5a0eae76b",
        },
    ),
    (
        "openai/streaming_grammar/incomplete_mid_tool_call.yaml#0",
        Refusal::IncompleteTerminal {
            request_id: "req_1eb9969a53024964a71f2b0820075968",
        },
    ),
    (
        "gemini/long_loop_matrix/output_cap_midway.yaml#0",
        Refusal::Response(
            "Gemini stopped with finish_reason=MalformedFunctionCall: Malformed function call: tool",
        ),
    ),
    (
        "llamacpp/content_matrix/empty_answer_with_stop.yaml#0",
        Refusal::Response(EMPTY_RESPONSE_ERROR),
    ),
];

/// The data of the one `response.incomplete` event among a Responses
/// stream's `frames`, verbatim. `Err` unless exactly one frame is that
/// event.
pub(crate) fn incomplete_frame_data(frames: &[String]) -> Result<String, String> {
    let data: Vec<&str> = frames
        .iter()
        .filter_map(|frame| {
            let line = frame.lines().find_map(|line| line.strip_prefix("data:"))?;
            let line = line.strip_prefix(' ').unwrap_or(line);
            let event: serde_json::Value = serde_json::from_str(line).ok()?;
            (event["type"] == "response.incomplete").then_some(line)
        })
        .collect();
    match data.as_slice() {
        [one] => Ok((*one).to_owned()),
        other => Err(format!(
            "{} `response.incomplete` frames, not one",
            other.len()
        )),
    }
}

/// The provider reply a default Responses stream refuses its
/// `response.incomplete` terminal with (G193): that frame's `body`, no
/// status, the reply's request id, nothing else.
pub(crate) fn incomplete_refusal(body: String, request_id: Option<&str>) -> ProviderResponseError {
    ProviderResponseError {
        status: None,
        body,
        provider_request_id: request_id.map(str::to_owned),
        headers: None,
        code: None,
        transient: None,
        refusal: false,
        route: None,
    }
}

/// Whether `error` is exactly the contract's `refusal` of `entry`; `Err`
/// names how it differs.
fn check_refusal(entry: &Entry, refusal: &Refusal, error: &ProviderError) -> Result<(), String> {
    match (refusal, error) {
        (Refusal::IncompleteTerminal { request_id }, ProviderError::ProviderResponse(response)) => {
            let frames = rig_test_support::stream_faults::sse_frames(&String::from_utf8_lossy(
                &entry.body(),
            ));
            let expected = incomplete_refusal(incomplete_frame_data(&frames)?, Some(request_id));
            if *response == expected {
                Ok(())
            } else {
                Err(format!(
                    "refused with {response:?}, the contract refuses with {expected:?}"
                ))
            }
        }
        (Refusal::Response(expected), ProviderError::Response(message)) if message == expected => {
            Ok(())
        }
        (refusal, error) => Err(format!(
            "refused with {error:?}, the contract refuses with {refusal:?}"
        )),
    }
}

/// The finish reason a decoded turn of `ending` must report.
fn expected(ending: Ending, calls: &[String]) -> Option<FinishReason> {
    match ending {
        Ending::Stop if calls.is_empty() => Some(FinishReason::Stop),
        Ending::Stop | Ending::Tool => Some(FinishReason::ToolCalls),
        Ending::Length => Some(FinishReason::Length),
        Ending::Filtered => Some(FinishReason::ContentFilter),
        Ending::Other => None,
    }
}

/// One origin partition of a provider's bank, decoded and compared with what
/// the bank read off its bytes. Returns how many replies were decoded.
async fn sweep(provider: &str, openrouter_only: bool) -> usize {
    let mut decoded_count = 0;
    let mut failures = Vec::new();
    let mut refused = Vec::new();
    for entry in bank::entries(provider).iter() {
        if openrouter_entry(entry) != openrouter_only {
            continue;
        }
        let Some(decoded) = decode(entry).await else {
            failures.push(format!(
                "{}: no decoder for `{}`",
                entry.source, entry.encoder
            ));
            continue;
        };
        decoded_count += 1;
        let status = entry.then.status;
        if !(200..300).contains(&status) {
            if decoded.error.is_none() {
                failures.push(format!(
                    "{}: status {status} decoded as a reply",
                    entry.source
                ));
            }
            continue;
        }
        if let Some((source, refusal)) = REFUSED.iter().find(|(source, _)| *source == entry.source)
        {
            refused.push(*source);
            let verdict = match &decoded.error {
                Some(error) => check_refusal(entry, refusal, error),
                None => Err(format!("decoded, the contract refuses it with {refusal:?}")),
            };
            if let Err(problem) = verdict {
                failures.push(format!("{}: {problem}", entry.source));
            }
            continue;
        }
        if let Some(error) = &decoded.error {
            failures.push(format!("{}: {error}", entry.source));
            continue;
        }
        // A Chat call the bytes cut at Length is discarded, never decoded
        // into a call; the bank reads every call the bytes carry.
        match calls_after_length_discards(entry, &decoded).await {
            Ok(Some(kept)) if decoded.calls != kept => failures.push(format!(
                "{}: calls {:?}, the bank read {:?}, of which {kept:?} survive its Length cut",
                entry.source, decoded.calls, entry.calls
            )),
            Ok(Some(_)) => {}
            Ok(None) if decoded.calls != entry.calls => failures.push(format!(
                "{}: calls {:?}, the bank read {:?}",
                entry.source, decoded.calls, entry.calls
            )),
            Ok(None) => {}
            Err(error) => {
                failures.push(format!("{}: {error}", entry.source));
                continue;
            }
        }
        if let Some(finish) = expected(entry.ending(), &entry.calls)
            && decoded.finish.as_ref() != Some(&finish)
        {
            failures.push(format!(
                "{}: finish {:?}, the bank read {:?}",
                entry.source, decoded.finish, entry.ends
            ));
        }
    }
    // A row this sweep never reaches names a reply the bank no longer holds.
    if !openrouter_only {
        let prefix = format!("{provider}/");
        failures.extend(
            REFUSED
                .iter()
                .filter(|(source, _)| source.starts_with(&prefix) && !refused.contains(source))
                .map(|(source, _)| format!("{source}: listed in REFUSED, not in the bank")),
        );
    }
    assert!(
        failures.is_empty(),
        "{provider}: {} of {decoded_count} replies disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
    decoded_count
}

macro_rules! sweeps {
    ($($provider:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $provider() {
                assert!(
                    sweep(stringify!($provider), stringify!($provider) == "openrouter").await > 0,
                    "the bank decodes some reply"
                );
            }
        )*
    };
}

sweeps!(
    anthropic, chatgpt, cohere, copilot, deepseek, doubleword, gemini, groq, llamacpp, mistral,
    mistralrs, ollama, openai, openrouter, perplexity, venice, xai,
);

#[tokio::test(flavor = "current_thread")]
async fn openrouter_anthropic_gateway() {
    assert!(
        sweep("anthropic", true).await > 0,
        "the bank decodes some reply"
    );
}
