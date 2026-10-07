use std::sync::Arc;

use serde_json::{Value, json};

use super::{CorruptFrame, ErrorDetail, ErrorKind, ErrorReport, FrameEvidence, ProviderError};

#[test]
fn a_companion_frame_keeps_received_bytes_and_the_framing_reason_without_reparsing() {
    let reason = "eventstream CRC mismatch: expected 00112233, received 44556677";
    for bytes in [b"{}\n".to_vec(), vec![0xff, 0, b'{', b'\n']] {
        let corrupt = CorruptFrame::received_bytes(bytes.clone(), reason);
        assert_eq!(corrupt.event_type(), None);
        assert_eq!(corrupt.frame(), None);
        assert_eq!(corrupt.frame_bytes(), Some(bytes.as_slice()));
        assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(bytes.clone()));
        assert_eq!(corrupt.error().to_string(), reason);
        let display = format!("frame failed to decode: {reason}; frame bytes: {}", bytes.escape_ascii());
        assert_eq!(corrupt.to_string(), display);
        let error = ProviderError::CorruptFrame(Arc::new(corrupt));
        assert_eq!(error.kind(), ErrorKind::Json);
        let report = error.report();
        assert_eq!(report.source_chain, [display, reason.to_owned()]);
        let ErrorDetail::CorruptFrame(detail) = report.detail.as_ref().expect("typed frame evidence") else {
            panic!("the corrupt-frame classification survives reporting");
        };
        assert_eq!(detail.evidence, FrameEvidence::Bytes(bytes.clone()));
        assert_eq!(detail.error, reason);
        assert_eq!(
            serde_json::to_value(detail).expect("serialize exact evidence"),
            json!({"evidence":{"kind":"bytes","frame":bytes},"error":reason}),
        );
        let serialized = serde_json::to_value(&report).expect("serialize report");
        assert_eq!(serde_json::from_value::<ErrorReport>(serialized).expect("read exact report"), report);
    }
}

#[test]
fn a_payload_json_error_keeps_the_exact_received_payload_and_original_diagnostic() {
    let bytes = b" \n{\"content\":\t ".to_vec();
    let parser = serde_json::from_slice::<Value>(&bytes).expect_err("malformed received payload");
    let reason = parser.to_string();
    let corrupt = CorruptFrame::received_bytes(bytes.clone(), parser);
    assert_eq!(corrupt.error().to_string(), reason);
    assert_eq!(corrupt.frame_bytes(), Some(bytes.as_slice()));
    assert_eq!(corrupt.evidence(), &FrameEvidence::Bytes(bytes));
}
