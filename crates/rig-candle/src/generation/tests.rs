use super::*;
use rig_core::message::Message;

/// Pure parse/emission regression: no model artifact or provider call is needed.
#[test]
fn parsed_missing_ids_keep_their_identity_and_provenance_through_stream_emission() {
    let request = CompletionRequest::from(vec![Message::user("tools")]);
    let raw = r#"<tool_call>{"name":"same","arguments":{"n":1}}</tool_call><tool_call>{"id":"tool-0","name":"same","arguments":{"n":2}}</tool_call><tool_call>{"name":"same","arguments":{"n":3}}</tool_call>"#;
    let parsed = crate::protocol::parse_assistant(raw, &request, ConversationProtocol::Qwen3)
        .expect("valid parsed tool event");
    let expected = parsed.items.clone();
    let mut emitted = Vec::new();
    emit_parsed_items(parsed.items, |event| {
        let GenerationEvent::ToolCall(call) = event else {
            return Err(CandleError::Inference("expected only tool calls".into()));
        };
        emitted.push(rig_core::message::AssistantContent::ToolCall(call));
        Ok(())
    })
    .expect("valid parsed tool event");
    assert_eq!(emitted.len(), 3);
    // The call that named its id keeps it; the others keep the ids rig
    // issued when parsing.
    let ids: Vec<_> = emitted
        .iter()
        .filter_map(|content| match content {
            rig_core::message::AssistantContent::ToolCall(call) => Some(call.id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 3, "{ids:?}");
    assert_eq!(
        ids.get(1).map(ToString::to_string).as_deref(),
        Some("tool-0")
    );
    assert!(
        [ids.first(), ids.get(2)]
            .iter()
            .all(|id| id.is_some_and(|id| id.is_local() && id.provider().is_none()))
    );
    assert_eq!(
        serde_json::to_value(emitted).expect("valid parsed tool event"),
        serde_json::to_value(expected).expect("valid parsed tool event")
    );
}

#[test]
fn sampling_without_top_k_or_top_p_samples_the_whole_distribution() {
    let config = GenerationConfig {
        temperature: 0.7,
        top_k: None,
        top_p: None,
        ..GenerationConfig::default()
    };
    assert_eq!(sampling(&config), Sampling::All { temperature: 0.7 });
}

/// A byte-fallback tokenizer: id 0 is the lone byte `0xC3`, the first half
/// of `é`, and `<|begin_of_text|>` is its only special token.
fn byte_fallback_tokenizer() -> Tokenizer {
    let vocab: tokenizers::models::bpe::Vocab =
        [("<0xC3>".to_string(), 0), ("<0xA9>".to_string(), 1)]
            .into_iter()
            .collect();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .byte_fallback(true)
        .build()
        .expect("a byte-fallback BPE model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(
        tokenizers::decoders::byte_fallback::ByteFallback::default(),
    ));
    tokenizer.add_special_tokens(&[tokenizers::AddedToken::from(crate::BEGIN_OF_TEXT, true)]);
    tokenizer
}

#[test]
fn incremental_decoder_flushes_a_trailing_partial_character_once() {
    let tokenizer = byte_fallback_tokenizer();
    let mut decoder = IncrementalTextDecoder::new(&tokenizer);
    assert_eq!(decoder.push(0).expect("a byte token decodes"), None);
    assert_eq!(
        decoder.finish().expect("the complete decode").as_deref(),
        Some("\u{FFFD}")
    );
    assert_eq!(decoder.finish().expect("an idempotent finish"), None);
    assert_eq!(decoder.text(), "\u{FFFD}");
}

#[test]
fn parsed_reasoning_and_text_are_emitted_in_order() {
    let request = CompletionRequest::from(vec![Message::user("plan")]);
    let parsed = crate::protocol::parse_assistant(
        "<think>plan</think>answer",
        &request,
        ConversationProtocol::Qwen3,
    )
    .expect("valid reasoning and text");
    let mut emitted = Vec::new();
    emit_parsed_items(parsed.items, |event| {
        emitted.push(event);
        Ok(())
    })
    .expect("every parsed item is emitted");
    assert_eq!(
        emitted,
        vec![
            GenerationEvent::Reasoning(Reasoning::new("plan")),
            GenerationEvent::Text("answer".to_owned()),
        ]
    );
}

/// The Qwen3 parser never produces these blocks, so only a direct call
/// reaches the refusals.
#[test]
fn emission_refuses_blocks_the_local_parser_never_produces() {
    for (item, reason) in [
        (
            AssistantContent::ProviderItem(rig_core::message::ProviderItem::new(
                serde_json::json!({"type": "web_search_call"}),
                "openai",
            )),
            "the local output parser produced a provider item, which it never emits",
        ),
        (
            AssistantContent::CustomToolCall(rig_core::message::CustomToolCall::new(
                rig_core::message::CallId::from_wire("call_1"),
                rig_core::message::ToolName::new("apply_patch").expect("tool name"),
                "*** Begin Patch",
            )),
            "the local output parser produced a custom tool call, which it never emits",
        ),
        (
            AssistantContent::Image(rig_core::message::Image::default()),
            "text-only Qwen output parser produced non-text content",
        ),
    ] {
        let mut emitted = Vec::new();
        let error = emit_parsed_items(vec![AssistantContent::text("before"), item], |event| {
            emitted.push(event);
            Ok(())
        })
        .expect_err("refused, not dropped");
        assert!(
            matches!(&error, CandleError::Inference(message) if message == reason),
            "{error:?}"
        );
        assert_eq!(
            error.to_string(),
            format!("Candle inference failed: {reason}")
        );
        assert_eq!(emitted, vec![GenerationEvent::Text("before".to_owned())]);
    }
}

/// An all-zero safetensors tensor for the tiny checkpoint below.
#[cfg(not(target_family = "wasm"))]
struct ZeroTensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[cfg(not(target_family = "wasm"))]
impl safetensors::tensor::View for ZeroTensor {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        safetensors::tensor::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(not(target_family = "wasm"))]
fn zeros(shape: &[usize]) -> ZeroTensor {
    ZeroTensor {
        shape: shape.to_vec(),
        bytes: vec![0; shape.iter().product::<usize>() * 4],
    }
}

/// A one-layer, eight-token Llama 3 checkpoint built in memory, the same
/// shape the model tests load. Its weights are all zero, so every logit ties
/// and greedy sampling always picks token 0; each test chooses what token 0
/// decodes to by swapping the tokenizer.
#[cfg(not(target_family = "wasm"))]
fn tiny_llama() -> LoadedModel {
    use tokenizers::AddedToken;
    use tokenizers::models::wordlevel::WordLevel;

    let config = br#"{
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 0.00001,
        "max_position_embeddings": 128,
        "bos_token_id": 2,
        "eos_token_id": [1, 3],
        "tie_word_embeddings": false
    }"#
    .to_vec();
    let vocab = [
        ("<unk>".to_string(), 0),
        (crate::END_OF_TURN.to_string(), 1),
        (crate::BEGIN_OF_TEXT.to_string(), 2),
        ("<eos>".to_string(), 3),
        (crate::START_HEADER.to_string(), 4),
        (crate::END_HEADER.to_string(), 5),
        ("assistant".to_string(), 6),
        ("hello".to_string(), 7),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("a word-level model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from(crate::END_OF_TURN, true),
        AddedToken::from(crate::BEGIN_OF_TEXT, true),
        AddedToken::from(crate::START_HEADER, true),
        AddedToken::from(crate::END_HEADER, true),
    ]);
    let tensors = vec![
        ("model.embed_tokens.weight".to_string(), zeros(&[8, 4])),
        ("model.norm.weight".to_string(), zeros(&[4])),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            zeros(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            zeros(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            zeros(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            zeros(&[4, 4]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            zeros(&[8, 4]),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            zeros(&[8, 4]),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            zeros(&[4, 8]),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            zeros(&[4]),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            zeros(&[4]),
        ),
        ("lm_head.weight".to_string(), zeros(&[8, 4])),
    ];
    let data = crate::ModelData {
        config,
        tokenizer: tokenizer
            .to_string(false)
            .expect("a serializable tokenizer")
            .into_bytes(),
        weights: safetensors::tensor::serialize(tensors, None).expect("a safetensors checkpoint"),
    };
    let generation = GenerationConfig {
        temperature: 0.0,
        max_tokens: 1,
        ..GenerationConfig::default()
    };
    crate::loader::load_model(data, generation, 1).expect("the tiny checkpoint loads")
}

/// A tokenizer whose token 0 decodes to `generated` and which encodes any
/// prompt as the single unknown token 1.
#[cfg(not(target_family = "wasm"))]
fn generating(generated: &str) -> Tokenizer {
    let vocab = [(generated.to_owned(), 0), ("<unk>".to_owned(), 1)]
        .into_iter()
        .collect();
    Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .expect("a word-level model"),
    )
}

/// The tiny checkpoint rendered and parsed as Qwen3, generating `generated`.
#[cfg(not(target_family = "wasm"))]
fn tiny_qwen3(generated: &str) -> LoadedModel {
    let mut loaded = tiny_llama();
    loaded.profile.definition = crate::profile::definition_for(
        ConversationProtocol::Qwen3,
        crate::profile::ArtifactFormat::Gguf,
    )
    .expect("the Qwen3 profile");
    loaded.tokenizer = generating(generated);
    loaded
}

#[cfg(not(target_family = "wasm"))]
fn payload(request: CompletionRequest) -> CandleRequest {
    use rig_core::wire::Wire as _;
    crate::Generation {
        model: "tiny".to_owned(),
        protocol: ConversationProtocol::Qwen3,
    }
    .encode(request, rig_core::wire::Mode::Unary)
    .expect("the generation wire encodes the request")
}

#[cfg(not(target_family = "wasm"))]
fn hello() -> CandleRequest {
    payload(CompletionRequest::from(vec![Message::user("hello")]))
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn qwen3_streaming_emits_the_parsed_turn() {
    let loaded = tiny_qwen3("<think>plan</think>answer");
    let mut events = Vec::new();
    let response = stream_generate(&loaded, &hello(), &CancellationSignal::default(), |event| {
        events.push(event);
        Ok(())
    })
    .expect("the turn streams");
    assert_eq!(
        events,
        vec![
            GenerationEvent::Reasoning(Reasoning::new("plan")),
            GenerationEvent::Text("answer".to_owned()),
        ]
    );
    assert_eq!(response.text, "answer");
    assert_eq!(response.prompt_tokens, 1);
    assert_eq!(response.generated_tokens, 1);
    assert_eq!(response.finish_reason, FinishReason::MaxTokens);
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn qwen3_malformed_output_is_refused_by_unary_and_streaming_generation() {
    let loaded = tiny_qwen3("<think>unfinished");
    let reason = "unterminated `<think>` reasoning block";
    assert!(matches!(
        infer(&loaded, &hello(), &CancellationSignal::default()),
        Err(CandleError::MalformedToolCall(actual)) if actual == reason
    ));
    let mut events = Vec::new();
    let streamed = stream_generate(&loaded, &hello(), &CancellationSignal::default(), |event| {
        events.push(event);
        Ok(())
    });
    assert!(
        matches!(&streamed, Err(CandleError::MalformedToolCall(actual)) if actual == reason),
        "{streamed:?}"
    );
    assert!(events.is_empty(), "{events:?}");
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn hosted_tools_are_refused_before_the_prompt_renders() {
    let loaded = tiny_llama();
    let mut request = CompletionRequest::from(vec![Message::user("hello")]);
    request.additional_params = Some(serde_json::json!({"tools": [{"type": "web_search"}]}));
    let result = generate(
        &loaded,
        &payload(request),
        &CancellationSignal::default(),
        |_| Ok(()),
    );
    assert!(
        matches!(
            &result,
            Err(CandleError::UnsupportedFeature(feature)) if feature == "provider-native hosted tools"
        ),
        "{result:?}"
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn a_prompt_that_encodes_to_no_tokens_is_refused() {
    let mut loaded = tiny_llama();
    // An empty BPE vocabulary with no unknown token drops every character.
    loaded.tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let result = generate(
        &loaded,
        &hello(),
        &CancellationSignal::default(),
        |_| Ok(()),
    );
    assert!(
        matches!(
            &result,
            Err(CandleError::TokenizerEncoding(reason))
                if reason == "the rendered prompt produced no tokens"
        ),
        "{result:?}"
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn a_trailing_partial_character_is_delivered_when_generation_finishes() {
    let mut loaded = tiny_llama();
    // The Llama 3 prompt encodes as `<|begin_of_text|>` alone; the model then
    // samples the lone byte 0xC3, which only the final decode can render.
    loaded.tokenizer = byte_fallback_tokenizer();
    let mut fragments = Vec::new();
    let response = generate(
        &loaded,
        &hello(),
        &CancellationSignal::default(),
        |fragment| {
            fragments.push(fragment);
            Ok(())
        },
    )
    .expect("generation finishes");
    assert_eq!(fragments, vec!["\u{FFFD}".to_owned()]);
    assert_eq!(response.text, "\u{FFFD}");
    assert_eq!(response.prompt_tokens, 1);
    assert_eq!(response.generated_tokens, 1);
    assert_eq!(response.finish_reason, FinishReason::MaxTokens);
}
