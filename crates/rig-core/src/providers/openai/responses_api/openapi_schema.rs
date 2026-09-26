//! Test-only access to the pinned public OpenAI OpenAPI schema.

use std::sync::OnceLock;

use jsonschema::Validator;
use serde_json::Value;

const OPENAPI: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/openai-openapi/openapi.json"
));

/// A compiled component and the named test-only view applied to it.
#[derive(Clone, Copy, Debug)]
pub(super) enum Schema {
    PublicInputItem,
    InputItemView,
    LiteInputItemView,
    EasyInputMessage,
    Item,
    AdditionalToolsItem,
    HttpCreateResponse,
    LiteHttpCreateResponse,
    WebSocketResponseCreate,
    LiteWebSocketResponseCreate,
    ResponseStreamEvent,
}

/// The exact pinned document, parsed once and never mutated.
pub(super) fn document() -> &'static Value {
    static DOCUMENT: OnceLock<Value> = OnceLock::new();
    DOCUMENT.get_or_init(|| serde_json::from_str(OPENAPI).expect("the pinned OpenAPI is JSON"))
}

/// The exact fixture bytes used to verify provenance.
pub(super) const fn bytes() -> &'static [u8] {
    OPENAPI.as_bytes()
}

/// Require `instance` to satisfy a compiled public component or named view.
pub(super) fn assert_valid(schema: Schema, instance: &Value) {
    let validator = validator(schema);
    let errors = validator
        .iter_errors(instance)
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
    assert!(
        errors.is_empty(),
        "{schema:?} rejected {instance}:\n{}",
        errors.join("\n")
    );
}

/// Require `instance` to be rejected by a compiled public component or view.
pub(super) fn assert_invalid(schema: Schema, instance: &Value) {
    assert!(
        !validator(schema).is_valid(instance),
        "{schema:?} unexpectedly accepted {instance}"
    );
}

fn validator(schema: Schema) -> &'static Validator {
    static PUBLIC_INPUT: OnceLock<Validator> = OnceLock::new();
    static INPUT_VIEW: OnceLock<Validator> = OnceLock::new();
    static LITE_INPUT_VIEW: OnceLock<Validator> = OnceLock::new();
    static EASY_MESSAGE: OnceLock<Validator> = OnceLock::new();
    static ITEM: OnceLock<Validator> = OnceLock::new();
    static ADDITIONAL_TOOLS: OnceLock<Validator> = OnceLock::new();
    static HTTP: OnceLock<Validator> = OnceLock::new();
    static LITE_HTTP: OnceLock<Validator> = OnceLock::new();
    static WEBSOCKET: OnceLock<Validator> = OnceLock::new();
    static LITE_WEBSOCKET: OnceLock<Validator> = OnceLock::new();
    static STREAM_EVENT: OnceLock<Validator> = OnceLock::new();

    let (cell, root, input_any_of, optional_image_detail) = match schema {
        Schema::PublicInputItem => (&PUBLIC_INPUT, "InputItem", false, false),
        Schema::InputItemView => (&INPUT_VIEW, "InputItem", true, false),
        Schema::LiteInputItemView => (&LITE_INPUT_VIEW, "InputItem", true, true),
        Schema::EasyInputMessage => (&EASY_MESSAGE, "EasyInputMessage", false, false),
        Schema::Item => (&ITEM, "Item", false, false),
        Schema::AdditionalToolsItem => {
            (&ADDITIONAL_TOOLS, "AdditionalToolsItemParam", false, false)
        }
        Schema::HttpCreateResponse => (&HTTP, "CreateResponse", true, false),
        Schema::LiteHttpCreateResponse => (&LITE_HTTP, "CreateResponse", true, true),
        Schema::WebSocketResponseCreate => (
            &WEBSOCKET,
            "ResponsesClientEventResponseCreate",
            true,
            false,
        ),
        Schema::LiteWebSocketResponseCreate => (
            &LITE_WEBSOCKET,
            "ResponsesClientEventResponseCreate",
            true,
            true,
        ),
        Schema::ResponseStreamEvent => (&STREAM_EVENT, "ResponseStreamEvent", false, false),
    };
    cell.get_or_init(|| compile(root, input_any_of, optional_image_detail))
}

fn compile(root: &str, input_any_of: bool, optional_image_detail: bool) -> Validator {
    let mut schema = document().clone();
    schema["$schema"] = Value::String("https://json-schema.org/draft/2020-12/schema".to_owned());
    schema["$ref"] = Value::String(format!("#/components/schemas/{root}"));

    if input_any_of {
        let input = schema
            .pointer_mut("/components/schemas/InputItem")
            .and_then(Value::as_object_mut)
            .expect("the pinned schema has InputItem");
        let branches = input
            .remove("oneOf")
            .expect("the pinned InputItem uses oneOf");
        assert!(input.insert("anyOf".to_owned(), branches).is_none());
    }

    if optional_image_detail {
        let required = schema
            .pointer_mut("/components/schemas/InputImageContent/required")
            .and_then(Value::as_array_mut)
            .expect("the pinned schema has InputImageContent.required");
        let before = required.len();
        required.retain(|field| field.as_str() != Some("detail"));
        assert_eq!(before - required.len(), 1, "detail is removed exactly once");
    }

    jsonschema::draft202012::new(&schema).expect("the selected OpenAPI component compiles")
}
