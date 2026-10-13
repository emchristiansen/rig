use super::*;

#[test]
fn test_interaction_status_maps_every_wire_variant() {
    use crate::completion::FinishReason as Normalized;
    for (status, expected) in [
        ("completed", Normalized::Stop),
        ("requires_action", Normalized::ToolCalls),
        ("budget_exceeded", Normalized::Length),
        ("in_progress", Normalized::Other("in_progress".to_owned())),
        ("incomplete", Normalized::Other("incomplete".to_owned())),
        ("failed", Normalized::Other("failed".to_owned())),
        ("cancelled", Normalized::Other("cancelled".to_owned())),
    ] {
        assert_eq!(ending(Some(status), &[]).0, Some(expected), "status {status}");
    }
}

#[test]
fn unknown_status_preserves_its_exact_spelling() {
    assert_eq!(ending(Some("future_exact_status"), &[]).0,
        Some(FinishReason::Other("future_exact_status".to_owned())));
    assert_eq!(ending(None, &[]).0, None);
}
