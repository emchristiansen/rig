//! Wire-boundary capture for ChatGPT/Codex Responses requests.
//!
//! When the `MUNINN_CODEX_WIRE_CAPTURE_DIR` environment variable is set, the
//! provider writes every outbound request to disk at the moment Rig has
//! produced the final serialized body and assembled the HTTP request
//! (method/URI/headers) but before the event source consumes it for
//! `.send()`. This captures the actual wire payload — including
//! Rig-controlled identity/auth headers and the cache-identity-stamped body
//! — without any further reqwest-layer mutation visibility.
//!
//! Output files are named `call_{seq:04}_request.json` with a process-wide
//! monotonic sequence. Authorization is logged verbatim; the capture
//! directory is intended to be a local scratch path treated as sensitive
//! and excluded from VCS.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use http::Request;
use serde_json::{Value, json};

const CAPTURE_DIR_ENV: &str = "MUNINN_CODEX_WIRE_CAPTURE_DIR";

static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Capture an outbound Codex Responses request to disk if
/// `MUNINN_CODEX_WIRE_CAPTURE_DIR` is set. No-op otherwise.
///
/// Errors are intentionally swallowed: capture is a diagnostic side channel
/// and must never fail the live inference path. Failures surface as missing
/// files in the capture directory.
pub(super) fn capture_outbound_request<T>(req: &Request<T>, body: &[u8]) {
    let Ok(dir) = std::env::var(CAPTURE_DIR_ENV) else {
        return;
    };
    let dir = PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    let seq = CALL_COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut headers = serde_json::Map::new();
    for (name, value) in req.headers().iter() {
        let value_str = value.to_str().unwrap_or("<non-utf8>").to_string();
        headers.insert(name.as_str().to_string(), Value::String(value_str));
    }

    let body_value: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => Value::String(String::from_utf8_lossy(body).into_owned()),
    };

    let envelope = json!({
        "seq": seq,
        "timestamp_ns": timestamp_ns,
        "method": req.method().as_str(),
        "uri": req.uri().to_string(),
        "headers": Value::Object(headers),
        "body_byte_len": body.len(),
        "body": body_value,
    });

    let path = dir.join(format!("call_{seq:04}_request.json"));
    let Ok(bytes) = serde_json::to_vec_pretty(&envelope) else {
        return;
    };
    let _ = std::fs::write(path, bytes);
}
