//! Responses frames of modeled types whose payloads fail typed decoding,
//! shared by the SSE and WebSocket tests. Each is written with irregular
//! spacing and key order, so a test that compares the frame an error carries
//! byte for byte also proves the frame was not re-serialized.

/// A frame of a modeled type that must fail to decode, with its `type`.
pub(super) struct MalformedFrame {
    /// What is malformed in it.
    pub(super) case: &'static str,
    /// The frame's `type`.
    pub(super) event_type: &'static str,
    /// The frame exactly as the provider sends it.
    pub(super) frame: &'static str,
}

/// The malformed content parts and reasoning content that a modeled frame
/// must refuse rather than keep as an unmodeled part.
pub(super) const MALFORMED_KNOWN_FRAMES: [MalformedFrame; 4] = [
    MalformedFrame {
        case: "a refusal part without a string `refusal`",
        event_type: "response.content_part.added",
        frame: r#"{"part": {"type":"refusal"},  "type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"sequence_number":1}"#,
    },
    MalformedFrame {
        case: "a reasoning_text part without `text`",
        event_type: "response.content_part.done",
        frame: r#"{"type" : "response.content_part.done","item_id":"rs_1","output_index":0,"content_index":0,"sequence_number":2,"part":{"type":"reasoning_text"}}"#,
    },
    MalformedFrame {
        case: "a content part without `type`",
        event_type: "response.content_part.added",
        frame: r#"{"sequence_number":3,"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{ "text":"hi" }}"#,
    },
    MalformedFrame {
        case: "reasoning content without `text` inside a reasoning item",
        event_type: "response.output_item.done",
        frame: r#"{"type":"response.output_item.done","output_index":0,"sequence_number":4,"item":{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text"}]}}"#,
    },
];
