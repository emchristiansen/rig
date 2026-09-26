use super::*;
use crate::completion::{CompletionRequest, Message};
use crate::providers::chatgpt;
use crate::providers::openai::OpenAI;
use crate::providers::openai::responses_api::wire::Responses;
use crate::wire::{Body, Mode, Wire};

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("say hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn chatgpt() -> Responses {
    OpenAI::with_key(&chatgpt::DIALECT, "test-token").responses("gpt-5.4")
}

/// The one request an encode sends: its `Content-Encoding`, if any, and its
/// body bytes.
fn sent(wire: &Responses) -> (Option<String>, Vec<u8>) {
    let encoded = wire.encode(prompt(), Mode::Streaming).expect("encodes");
    let request = encoded
        .requests
        .first()
        .expect("a Responses request is one request");
    let encoding = request
        .headers()
        .get(http::header::CONTENT_ENCODING)
        .map(|value| value.to_str().expect("ascii header").to_owned());
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    (encoding, body.to_vec())
}

/// A zstd wire sends exactly the plain wire's body, zstd-compressed, and
/// names the encoding.
#[test]
fn a_zstd_wire_sends_the_plain_body_compressed_and_names_the_encoding() {
    let (plain_encoding, plain) = sent(&chatgpt());
    assert_eq!(plain_encoding, None);

    let (encoding, compressed) =
        sent(&chatgpt().with_request_compression(RequestCompression::Zstd));
    assert_eq!(encoding.as_deref(), Some("zstd"));
    assert_ne!(compressed, plain, "the body is compressed");
    assert_eq!(
        zstd::stream::decode_all(compressed.as_slice()).expect("a zstd frame"),
        plain,
        "decompressing gives back the plain body, byte for byte"
    );
    assert_eq!(
        compressed,
        zstd::stream::encode_all(plain.as_slice(), 3).expect("compresses"),
        "compressed at level 3, as the Codex client does"
    );
}

/// The default sends bodies as they are, and stays off the serialized wire.
#[test]
fn the_default_sends_bodies_as_they_are() {
    let wire = chatgpt();
    assert!(wire.request_compression.is_none());
    let json = serde_json::to_value(&wire).expect("serializes");
    assert!(json.get("request_compression").is_none(), "got {json}");

    let zstd = serde_json::to_value(chatgpt().with_request_compression(RequestCompression::Zstd))
        .expect("serializes");
    assert_eq!(zstd["request_compression"], "zstd");
}
