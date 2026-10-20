//! The unchanged Z.AI Messages fixture under the typed boundary.

use super::*;
use rig_core::{completion::CompletionRequest, error::ErrorDetail, wire::Mode};
use rig_history_conformance::{HistoryFixture, decode};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
enum Outcome {
    Accepted,
    AcceptedMissingFinish,
    CorruptFrame,
}

pub(super) fn zai_h05_preserves_the_typed_boundary(fixture: &MessagesHistory) {
    let ablation = fixture
        .ablation()
        .expect("the unchanged original ablation fixture");
    let wire = fixture.wire(fixture.model);
    let request = CompletionRequest::new("restate");
    let original = decode(
        &wire,
        &request,
        Mode::Unary,
        (ablation.frames)(ablation.document.clone()),
    )
    .expect("the original whole reply decodes");
    assert!(!original.stop().is_failure());
    let cases = [
        (
            "/id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/role",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/model",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/0/thinking",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/0/signature",
            [Outcome::Accepted, Outcome::Accepted, Outcome::CorruptFrame],
        ),
        (
            "/content/1/text",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/id",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/name",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/content/2/input",
            [Outcome::CorruptFrame, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/content/2/input/q",
            [Outcome::Accepted, Outcome::Accepted, Outcome::Accepted],
        ),
        (
            "/stop_reason",
            [
                Outcome::AcceptedMissingFinish,
                Outcome::AcceptedMissingFinish,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/stop_sequence",
            [Outcome::Accepted, Outcome::Accepted, Outcome::CorruptFrame],
        ),
        (
            "/usage",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/usage/input_tokens",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
        (
            "/usage/output_tokens",
            [
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
                Outcome::CorruptFrame,
            ],
        ),
    ];
    for (pointer, expected) in cases {
        let original = ablation
            .document
            .pointer(pointer)
            .expect("the original field");
        let retyped = match original {
            Value::String(_) => Value::from(7),
            Value::Number(_) => Value::from("7"),
            Value::Bool(_) => Value::from("true"),
            Value::Array(_) => json!({}),
            Value::Object(_) => json!([]),
            Value::Null => Value::from(false),
        };
        for (change, replacement, expected) in [
            ("removed", None, expected[0]),
            ("null", Some(Value::Null), expected[1]),
            ("retyped", Some(retyped), expected[2]),
        ] {
            let mut document = ablation.document.clone();
            if let Some(value) = replacement {
                *document.pointer_mut(pointer).expect("the original field") = value;
            } else {
                let (parent, key) = pointer.rsplit_once('/').expect("a field pointer");
                let key = key.replace("~1", "/").replace("~0", "~");
                document
                    .pointer_mut(parent)
                    .and_then(Value::as_object_mut)
                    .expect("the original parent")
                    .shift_remove(&key);
            }
            let result = decode(&wire, &request, Mode::Unary, (ablation.frames)(document));
            match expected {
                Outcome::CorruptFrame => {
                    let error = result.expect_err("the invalid typed field must refuse");
                    assert!(
                        matches!(error.report().detail, Some(ErrorDetail::CorruptFrame(_))),
                        "{pointer} {change}: {error}"
                    );
                }
                Outcome::Accepted => {
                    let response =
                        result.unwrap_or_else(|error| panic!("{pointer} {change}: {error}"));
                    assert!(!response.stop().is_failure(), "{pointer} {change}");
                }
                Outcome::AcceptedMissingFinish => {
                    let response =
                        result.unwrap_or_else(|error| panic!("{pointer} {change}: {error}"));
                    assert!(response.stop().is_failure(), "{pointer} {change}");
                    assert_eq!(
                        response.error.as_deref(),
                        Some("the provider ended the reply without a finish reason")
                    );
                }
            }
        }
    }
    let mut without_finish = ablation.document;
    without_finish
        .as_object_mut()
        .expect("the original object")
        .shift_remove("stop_reason");
    assert!(
        decode(
            &wire,
            &request,
            Mode::Unary,
            (ablation.frames)(without_finish)
        )
        .map_or(true, |response| response.stop().is_failure())
    );
}
