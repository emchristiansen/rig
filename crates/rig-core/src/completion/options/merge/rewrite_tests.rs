//! The rewrites no built-in wire selects, the accepting arms of the stream
//! usage, Anthropic cache-control and WebSocket-create writes, the
//! `Mapping::Place` contract refusal and `FinalBody::pointer`.

use serde_json::{Map, Value, json};

use super::{FinalBody, RawAt, Rewrite, check_anthropic_cache_control, request_params};
use crate::completion::options::{Mapping, OptionFields, OptionMap};
use crate::completion::{
    Accepts, CacheRetention, CompletionRequest, GenerationOptions, OnUnsupported, ReplayTarget,
};
use crate::message::Api;

/// A target whose answer for each option is the function it holds.
#[derive(Debug)]
struct Fake(fn(OptionFields<'_>) -> OptionMap);

impl ReplayTarget for Fake {
    fn api(&self) -> Api {
        Api::from_static("fake.chat")
    }

    fn provider(&self) -> &str {
        "fake"
    }

    fn model(&self) -> &str {
        "fake-1"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn map_options(&self, _request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
        (self.0)(fields)
    }
}

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(map) = value else {
        panic!("a JSON object literal: {value}");
    };
    map
}

/// `base` merged for a request that sets no option, then `rewrites`.
fn merged(base: Value, rewrites: &[Rewrite]) -> FinalBody {
    request_params(
        &Fake(crate::test_utils::refuse_options),
        &CompletionRequest::new("hi"),
        |_| Ok(object(base)),
        RawAt::Top,
        rewrites,
    )
    .expect("the local request merge succeeds")
}

#[test]
fn codex_store_sends_store_false_over_any_stored_value() {
    for base in [
        json!({"model": "gpt-5"}),
        json!({"model": "gpt-5", "store": true}),
        json!({"model": "gpt-5", "store": null}),
    ] {
        assert_eq!(
            merged(base.clone(), &[Rewrite::CodexStore]),
            FinalBody(object(json!({"model": "gpt-5", "store": false}))),
            "{base}"
        );
    }
}

#[test]
fn reasoning_ciphertext_asks_for_the_ciphertext_only_when_the_body_reasons_or_stores_nothing() {
    const CIPHERTEXT: &str = "reasoning.encrypted_content";
    for (always, base, expected) in [
        (false, json!({"model": "m"}), json!({"model": "m"})),
        (
            false,
            json!({"model": "m", "store": true}),
            json!({"model": "m", "store": true}),
        ),
        (
            false,
            json!({"model": "m", "reasoning": {"effort": "low"}}),
            json!({"model": "m", "reasoning": {"effort": "low"}, "include": [CIPHERTEXT]}),
        ),
        // Unlike `ResponsesReasoningCiphertext`, a present null asks too.
        (
            false,
            json!({"model": "m", "reasoning": null}),
            json!({"model": "m", "reasoning": null, "include": [CIPHERTEXT]}),
        ),
        (
            false,
            json!({"model": "m", "store": false}),
            json!({"model": "m", "store": false, "include": [CIPHERTEXT]}),
        ),
        (
            true,
            json!({"model": "m", "include": ["file_search_call.results"]}),
            json!({"model": "m", "include": ["file_search_call.results", CIPHERTEXT]}),
        ),
        (
            true,
            json!({"model": "m", "include": [CIPHERTEXT]}),
            json!({"model": "m", "include": [CIPHERTEXT]}),
        ),
    ] {
        assert_eq!(
            merged(base.clone(), &[Rewrite::ReasoningCiphertext(always)]),
            FinalBody(object(expected)),
            "always={always} base={base}"
        );
    }
}

#[test]
fn stream_usage_leaves_stream_options_that_are_not_an_object_as_written() {
    for stream_options in [json!(true), json!("verbatim"), json!([1])] {
        let base = json!({"model": "m", "stream_options": stream_options});
        assert_eq!(
            merged(base.clone(), &[Rewrite::StreamUsage]),
            FinalBody(object(base))
        );
    }
}

/// `seed` answered with `Mapping::Place`, every other set option refused.
fn places_seed(fields: OptionFields<'_>) -> OptionMap {
    let seed = fields.seed;
    OptionMap {
        seed: Mapping::of(seed, |_| Mapping::Place),
        ..crate::test_utils::refuse_options(fields)
    }
}

/// Only `cache` places markers, so a wire that answers `Place` for another
/// option breaks the mapping contract under either policy.
#[test]
fn a_place_answer_for_an_option_other_than_cache_is_a_contract_error() {
    for policy in [OnUnsupported::Error, OnUnsupported::Ignore] {
        let request = CompletionRequest::new("hi")
            .options(GenerationOptions::default().seed(7).on_unsupported(policy));
        let error = request_params(
            &Fake(places_seed),
            &request,
            |_| Ok(Map::new()),
            RawAt::Top,
            &[],
        )
        .expect_err("only cache places markers");
        assert!(error.unsupported_option().is_none(), "{error}");
        assert_eq!(
            error.to_string(),
            "RequestError: fake answered `Mapping::Place` for the option `seed`; \
             only `cache` places markers"
        );
    }
}

#[test]
fn a_raw_anthropic_marker_with_the_long_ttl_agrees_with_long_retention() {
    let long = |marker: Value| {
        CompletionRequest::new("hi")
            .options(GenerationOptions::default().cache(CacheRetention::Long))
            .additional_params(json!({"cache_control": marker}))
    };
    check_anthropic_cache_control(&long(json!({"type": "ephemeral", "ttl": "1h"})))
        .expect("a one-hour raw marker matches long retention");
    let error = check_anthropic_cache_control(&long(json!({"type": "ephemeral", "ttl": "5m"})))
        .expect_err("a five-minute raw marker conflicts with long retention");
    assert_eq!(
        error.to_string(),
        "RequestError: Raw top-level cache_control conflicts with the typed automatic caching TTL"
    );
}

#[test]
fn final_body_pointer_unescapes_its_top_level_key_and_reads_nested_paths() {
    let body = FinalBody(object(json!({
        "a/b": {"c": 1},
        "m~n": 2,
        "top": {"x": [10, 20]},
    })));
    assert_eq!(body.pointer("/a~1b/c"), Some(&json!(1)));
    assert_eq!(body.pointer("/m~0n"), Some(&json!(2)));
    assert_eq!(body.pointer("/top"), Some(&json!({"x": [10, 20]})));
    assert_eq!(body.pointer("/top/x/1"), Some(&json!(20)));
    assert_eq!(body.pointer("/top/missing"), None);
    assert_eq!(body.pointer("/missing/x"), None);
    assert_eq!(body.pointer("top"), None);
    assert_eq!(body.pointer(""), None);
}

/// A full create frame that is the only rewrite is merged as a full
/// request, and a caller's own `previous_response_id` wins over the
/// automatic one.
#[cfg(feature = "websocket")]
#[test]
fn a_lone_full_websocket_create_keeps_the_callers_previous_response() {
    use crate::providers::openai::responses_api::websocket::{
        WebSocketCreate, WebSocketCreateInput,
    };
    let create = WebSocketCreate {
        input: WebSocketCreateInput::Full,
        stamps: Map::new(),
        identity: None,
        lite: false,
        automatic_previous: Some("resp_automatic".to_owned()),
        generate: None,
    };
    assert_eq!(
        merged(
            json!({
                "model": "gpt-5",
                "previous_response_id": "resp_caller",
                "stream": true,
                "background": false,
            }),
            &[Rewrite::ResponsesWebSocketCreate(Box::new(create))],
        ),
        FinalBody(object(json!({
            "model": "gpt-5",
            "previous_response_id": "resp_caller",
            "type": "response.create",
        })))
    );
}
