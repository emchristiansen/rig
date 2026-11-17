use super::*;

/// Zstd names its encoding and compresses at level 3, as the Codex client
/// does; decompressing gives back the body byte for byte.
#[test]
fn zstd_compresses_at_level_three_and_names_the_encoding() {
    let plain = br#"{"model":"gpt-5.4","input":[{"role":"user","content":"say hi"}]}"#.to_vec();
    let (encoding, compressed) = RequestCompression::Zstd
        .apply(plain.clone())
        .expect("compresses");
    assert_eq!(encoding, Some("zstd"));
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

/// The default sends a body as it is and names no encoding.
#[test]
fn the_default_leaves_the_body_as_it_is() {
    let plain = b"{}".to_vec();
    assert!(RequestCompression::default().is_none());
    assert_eq!(
        RequestCompression::None
            .apply(plain.clone())
            .expect("nothing to fail"),
        (None, plain)
    );
    assert!(!RequestCompression::Zstd.is_none());
}

#[test]
fn compression_serializes_by_name() {
    assert_eq!(
        serde_json::to_value(RequestCompression::Zstd).expect("serializes"),
        "zstd"
    );
    assert_eq!(
        serde_json::to_value(RequestCompression::None).expect("serializes"),
        "none"
    );
    assert_eq!(
        serde_json::from_str::<RequestCompression>(r#""zstd""#).expect("deserializes"),
        RequestCompression::Zstd
    );
}
