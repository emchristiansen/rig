use std::error::Error;
use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::message::{CallId, LocalCallId, ProviderCallId, ToolName};

fn json_error() -> serde_json::Error {
    serde_json::from_str::<Value>("{").expect_err("invalid JSON")
}

#[test]
fn a_corrupt_frame_clone_shares_the_typed_source_and_exact_report() {
    let frame = " \n{\"secret\":\"kept verbatim\",\t";
    let corrupt = CorruptFrame::text(Some("response.delta".to_owned()), frame, json_error());
    let parser = corrupt.error().to_string();
    let display = format!("`response.delta` frame failed to decode: {parser}; frame: {frame}");
    let error = ProviderError::CorruptFrame(Arc::new(corrupt));
    let cloned = error.clone();
    let (ProviderError::CorruptFrame(original), ProviderError::CorruptFrame(shared)) =
        (&error, &cloned)
    else {
        panic!("cloning must preserve the error variant");
    };
    assert!(Arc::ptr_eq(original, shared));
    let source = cloned.source().expect("the corruption is the source");
    let source = source
        .downcast_ref::<CorruptFrame>()
        .expect("typed corruption");
    assert!(std::ptr::eq(source, original.as_ref()));
    assert!(std::ptr::eq(
        source
            .source()
            .expect("JSON parser source")
            .downcast_ref::<serde_json::Error>()
            .expect("typed JSON parser"),
        original.error(),
    ));
    assert_eq!(source.event_type(), Some("response.delta"));
    assert_eq!(source.frame(), Some(frame));
    assert_eq!(source.frame_bytes(), Some(frame.as_bytes()));
    assert_eq!(source.to_string(), display);
    assert_eq!(error.to_string(), format!("CorruptFrameError: {display}"));
    assert_eq!(error.kind(), ErrorKind::Json);
    assert_eq!(error.boundary(), AdapterErrorBoundary::Decode);
    assert!(!error.is_retryable());
    let report = error.report();
    assert_eq!(cloned.report(), report);
    assert_eq!(report.source_chain, [display, parser.clone()]);
    assert_eq!(report.provider_response, None);
    assert_eq!(
        serde_json::to_value(report.detail.as_ref().expect("structured corruption"))
            .expect("serialize detail"),
        json!({
            "detail": "corrupt_frame", "event_type": "response.delta",
            "evidence": {"kind": "text", "frame": frame}, "error": parser,
        }),
    );
    let serialized = serde_json::to_value(&report).expect("serialize report");
    assert_eq!(
        serde_json::from_value::<ErrorReport>(serialized).expect("read report"),
        report
    );
}

#[test]
fn every_frame_evidence_kind_preserves_its_fidelity_and_display() {
    let text = "\t{\"x\":\"\u{fffd}\"}\n";
    let bytes = vec![b'{', b'\n', 0xff, 0, b'}'];
    let mut decoded = CorruptFrame::text(None, text, json_error());
    decoded.mark_replacement_decoded();
    for (corrupt, evidence, suffix, exact_text, exact_bytes, wire) in [
        (
            CorruptFrame::text(None, text, json_error()),
            FrameEvidence::Text(text.to_owned()),
            format!("; frame: {text}"),
            Some(text),
            Some(text.as_bytes().to_vec()),
            json!({"kind": "text", "frame": text}),
        ),
        (
            CorruptFrame::bytes(bytes.clone(), json_error()),
            FrameEvidence::Bytes(bytes.clone()),
            "; frame bytes: {\\n\\xff\\x00}".to_owned(),
            None,
            Some(bytes.clone()),
            json!({"kind": "bytes", "frame": bytes}),
        ),
        (
            decoded,
            FrameEvidence::Decoded(text.to_owned()),
            format!("; frame decoded with U+FFFD replacement: {text}"),
            None,
            None,
            json!({"kind": "decoded", "frame": text}),
        ),
        (
            CorruptFrame::absent(json_error()),
            FrameEvidence::Absent,
            String::new(),
            None,
            None,
            json!({"kind": "absent"}),
        ),
    ] {
        let parser = corrupt.error().to_string();
        assert_eq!(corrupt.event_type(), None);
        assert_eq!(corrupt.evidence(), &evidence);
        assert_eq!(corrupt.frame(), exact_text);
        assert_eq!(corrupt.frame_bytes(), exact_bytes.as_deref());
        assert_eq!(
            corrupt.to_string(),
            format!("frame failed to decode: {parser}{suffix}")
        );
        let error = ProviderError::CorruptFrame(Arc::new(corrupt));
        let report = error.report();
        assert_eq!(
            report.source_chain,
            [
                error.corrupt_frame().expect("corruption").to_string(),
                parser.clone()
            ]
        );
        assert_eq!(
            serde_json::to_value(report.detail.as_ref().expect("detail")).expect("serialize"),
            json!({"detail": "corrupt_frame", "evidence": wire, "error": parser}),
        );
        let serialized = serde_json::to_value(&report).expect("serialize report");
        assert_eq!(
            serde_json::from_value::<ErrorReport>(serialized).expect("read report"),
            report
        );
    }
}

#[test]
fn evidence_reclassification_never_invents_received_bytes_for_an_absent_frame() {
    let bytes = [0xff, b'{', 0];
    let mut corrupt = CorruptFrame::text(Some("delta".to_owned()), "\u{fffd}{", json_error());
    corrupt.mark_replacement_decoded();
    corrupt.mark_replacement_decoded();
    assert_eq!(
        corrupt.evidence(),
        &FrameEvidence::Decoded("\u{fffd}{".to_owned())
    );
    assert_eq!(corrupt.frame_bytes(), None);
    corrupt.set_received_bytes(&bytes);
    corrupt.mark_replacement_decoded();
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(bytes.to_vec()));
    assert_eq!(corrupt.event_type(), Some("delta"));
    assert_eq!(corrupt.frame_bytes(), Some(bytes.as_slice()));

    let replacement = serde_json::from_str::<u64>("true").expect_err("wrong typed payload");
    let replacement_text = replacement.to_string();
    let corrupt = corrupt.with_error(replacement);
    assert_eq!(corrupt.error().to_string(), replacement_text);
    assert_eq!(corrupt.event_type(), Some("delta"));
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(bytes.to_vec()));
    assert!(std::ptr::eq(
        corrupt
            .source()
            .expect("source")
            .downcast_ref::<serde_json::Error>()
            .expect("JSON error"),
        corrupt.error(),
    ));

    let mut absent = CorruptFrame::absent(json_error());
    absent.mark_replacement_decoded();
    absent.set_received_bytes(&bytes);
    assert_eq!(absent.evidence(), &FrameEvidence::Absent);
    assert_eq!(absent.frame(), None);
    assert_eq!(absent.frame_bytes(), None);
}

#[test]
fn malformed_input_keeps_canonical_call_and_provider_evidence_in_separate_slots() {
    let raw = " \n{\"x\":true,\t ";
    let error = serde_json::from_str::<Value>(raw)
        .expect_err("complete malformed input")
        .to_string();
    for item_id in [None, Some(String::new()), Some("fc_item_1".to_owned())] {
        let diagnostic = MalformedToolInput {
            name: "add".to_owned(),
            id: CallId::from_wire("canonical_call_1"),
            provider: Some(ProviderCallEvidence {
                call_id: ProviderCallId::new("wire_call_1").expect("nonempty"),
                item_id: item_id.clone(),
            }),
            raw: raw.to_owned(),
            error: error.clone(),
        };
        let detail = ErrorDetail::MalformedToolInput(diagnostic);
        let mut expected = json!({
            "detail": "malformed_tool_input", "name": "add",
            "id": {"provider": "canonical_call_1"},
            "provider": {"call_id": "wire_call_1"}, "raw": raw, "error": error,
        });
        if let Some(item_id) = &item_id {
            expected["provider"]["item_id"] = json!(item_id);
        }
        assert_eq!(serde_json::to_value(&detail).expect("serialize"), expected);
        assert_eq!(
            serde_json::from_value::<ErrorDetail>(expected).expect("read"),
            detail
        );
        let report = ErrorReport::new(ErrorKind::Response, "malformed tool arguments")
            .with_detail(detail.clone());
        assert_eq!(report.detail, Some(detail));
        let encoded = serde_json::to_value(&report).expect("serialize report");
        assert_eq!(
            serde_json::from_value::<ErrorReport>(encoded).expect("read report"),
            report
        );
    }
    for provider in [json!({"call_id": ""}), json!({"item_id": "fc_only"})] {
        assert!(serde_json::from_value::<ProviderCallEvidence>(provider).is_err());
    }
}

#[test]
fn malformed_input_retains_typed_local_identity_and_optional_provider_evidence() {
    let local = CallId::Local(LocalCallId::new());
    let equal_wire_provider = CallId::from_wire(local.wire().into_owned());
    assert_ne!(local, equal_wire_provider);
    let mut decoded_ids = Vec::new();
    for id in [local.clone(), equal_wire_provider.clone()] {
        let detail = ErrorDetail::MalformedToolInput(MalformedToolInput {
            name: "f".to_owned(),
            id: id.clone(),
            provider: None,
            raw: "{broken".to_owned(),
            error: "typed parser failure".to_owned(),
        });
        let encoded = serde_json::to_value(&detail).expect("serialize");
        assert!(encoded.get("provider").is_none());
        let decoded = serde_json::from_value::<ErrorDetail>(encoded).expect("read");
        assert_eq!(decoded, detail);
        let ErrorDetail::MalformedToolInput(decoded) = decoded else {
            panic!("same diagnostic");
        };
        decoded_ids.push(decoded.id);
    }
    assert_eq!(decoded_ids, [local, equal_wire_provider]);
    assert_ne!(decoded_ids[0], decoded_ids[1]);
}

#[test]
fn a_relay_keeps_structured_diagnostics_and_every_report_field_unchanged() {
    let diagnostic = ErrorDetail::CorruptFrame(CorruptFrameDetail::from(&CorruptFrame::bytes(
        vec![0xff, b'{'],
        json_error(),
    )));
    let mut report = ErrorReport::new(ErrorKind::ProviderResponse, "relay's exact message")
        .with_retryable(true)
        .with_code("relay_code")
        .with_http_status(503)
        .refused()
        .with_request_id("relay_request")
        .with_detail(diagnostic);
    report.source_chain = vec!["outer relay source".to_owned(), "inner source".to_owned()];
    report.provider_response = Some(ProviderResponseError::new(
        http::StatusCode::BAD_REQUEST,
        "body differs from report",
    ));
    let error = ProviderError::Relayed(Box::new(report.clone()));
    assert_eq!(error.to_string(), report.message);
    assert_eq!(error.kind(), report.kind);
    assert_eq!(error.is_retryable(), report.retryable);
    assert_eq!(error.provider_response(), report.provider_response.as_ref());
    assert!(error.corrupt_frame().is_none());
    assert!(error.source().is_none());
    assert_eq!(error.report(), report);
    assert_eq!(error.clone().report(), report);
    assert_eq!(ErrorReport::from(error), report);
}

#[test]
fn ordinary_reports_omit_detail_and_deserialize_without_it() {
    let reports = [
        ErrorReport::new(ErrorKind::Other, "plain"),
        ProviderError::from(json_error()).report(),
        ProviderError::request(std::io::Error::other("io source")).report(),
        ProviderError::Response("shape".to_owned()).report(),
        ProviderError::from_http_response(http::StatusCode::BAD_REQUEST, "raw body").report(),
        ProviderError::Truncated.report(),
        ToolExecutionError::new(ToolErrorKind::Other, "tool").report(),
        MemoryError::Policy("policy".to_owned()).report(),
        ErrorReport::from(VectorStoreError::MissingIdError("missing".to_owned())),
    ];
    for report in reports {
        assert_eq!(report.detail, None);
        let mut encoded = serde_json::to_value(&report).expect("serialize");
        assert!(encoded.get("detail").is_none());
        assert_eq!(
            serde_json::from_value::<ErrorReport>(encoded.clone()).expect("legacy report"),
            report
        );
        encoded["detail"] = Value::Null;
        assert_eq!(
            serde_json::from_value::<ErrorReport>(encoded).expect("explicit null detail"),
            report
        );
    }
    let corrupt = ProviderError::CorruptFrame(Arc::new(CorruptFrame::absent(json_error())));
    let detail = corrupt.report().detail;
    let report = ErrorReport::from(VectorStoreError::EmbeddingError(corrupt));
    assert_eq!(report.detail, detail);
    assert_eq!(report.kind, ErrorKind::Json);
    assert_eq!(report.source_chain.len(), 3);
}

#[test]
fn carried_refusals_keep_their_real_typed_sources_as_request_failures() {
    let refusal = crate::message::UnrepresentableToolCall::Namespace {
        wire: "json-wire",
        name: ToolName::new("f").expect("tool name"),
        namespace: String::new(),
    };
    let expected = refusal.to_string();
    let encoded: EncodeError = refusal.clone().into();
    assert_eq!(encoded.unsupported_option(), None);
    let error = ProviderError::from(encoded);
    assert_eq!(error.kind(), ErrorKind::Request);
    assert_eq!(error.boundary(), AdapterErrorBoundary::Request);
    assert!(!error.is_retryable());
    assert_eq!(
        error
            .source()
            .expect("source")
            .downcast_ref::<crate::message::UnrepresentableToolCall>(),
        Some(&refusal)
    );
    assert_eq!(error.clone().report(), error.report());
    assert_eq!(error.report().source_chain, std::slice::from_ref(&expected));
    assert_eq!(error.to_string(), format!("RequestError: {expected}"));
    assert_eq!(error.report().detail, None);

    let refusal = crate::message::UnreplayableProviderItem {
        wire: "destination".to_owned(),
        issuer: Some(String::new()),
        item_type: Some("future_item".to_owned()),
    };
    let expected = refusal.to_string();
    let encoded: EncodeError = refusal.clone().into();
    let error = ProviderError::from(encoded);
    assert_eq!(error.kind(), ErrorKind::Request);
    assert_eq!(error.boundary(), AdapterErrorBoundary::Request);
    assert!(!error.is_retryable());
    assert_eq!(
        error
            .source()
            .expect("source")
            .downcast_ref::<crate::message::UnreplayableProviderItem>(),
        Some(&refusal)
    );
    assert_eq!(error.report().source_chain, std::slice::from_ref(&expected));
    assert_eq!(error.to_string(), format!("RequestError: {expected}"));
    assert_eq!(error.report().detail, None);
}

#[test]
fn upstream_shared_sources_truncation_and_unsupported_options_keep_their_contracts() {
    #[derive(Debug, thiserror::Error)]
    #[error("outer request")]
    struct Outer(#[source] std::io::Error);
    let request = ProviderError::request(Outer(std::io::Error::other("inner request")));
    let cloned = request.clone();
    assert!(std::ptr::eq(
        request
            .source()
            .expect("request source")
            .downcast_ref::<Outer>()
            .expect("outer"),
        cloned
            .source()
            .expect("cloned source")
            .downcast_ref::<Outer>()
            .expect("outer"),
    ));
    assert_eq!(
        cloned.report().source_chain,
        ["outer request", "inner request"]
    );
    let json = ProviderError::from(json_error());
    let cloned = json.clone();
    assert!(std::ptr::eq(
        json.source()
            .expect("JSON source")
            .downcast_ref::<serde_json::Error>()
            .expect("parser"),
        cloned
            .source()
            .expect("cloned JSON source")
            .downcast_ref::<serde_json::Error>()
            .expect("parser"),
    ));
    assert_eq!(cloned.report(), json.report());
    let truncated = ProviderError::Truncated;
    assert_eq!(truncated.kind(), ErrorKind::Response);
    assert!(truncated.is_retryable());
    assert_eq!(
        truncated.to_string(),
        "ResponseError: the reply ended before the provider ended it"
    );
    assert!(truncated.source().is_none());
    let option = UnsupportedOption::new("seed", "provider", "model", "unsupported");
    let encoded = EncodeError::unsupported(option.clone());
    assert_eq!(encoded.unsupported_option(), Some(&option));
    let error = ProviderError::from(encoded);
    assert_eq!(error.kind(), ErrorKind::Request);
    assert!(!error.is_retryable());
    assert_eq!(error.to_string(), format!("RequestError: {option}"));
    assert!(error.source().is_none());
    assert_eq!(error.report().detail, None);
}
