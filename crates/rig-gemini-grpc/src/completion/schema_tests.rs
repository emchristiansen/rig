use super::*;
use crate::proto;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::Value;

// Inspect the schema emitted by the real shared encoder and protobuf transcode.
fn projected_schema(parameters: &Value) -> Result<Option<proto::Schema>, EncodeError> {
    let mut request = CompletionRequest::new("hello");
    request.tools = vec![ToolDefinition::new(
        ToolName::new("schema_fixture").expect("fixture name"),
        "schema fixture",
        parameters.clone(),
    )];
    let mut encoded = GenerateContent::new(GEMINI_2_5_FLASH).encode(request, Mode::Unary)?;
    Ok(encoded
        .tools
        .remove(0)
        .function_declarations
        .remove(0)
        .parameters)
}

#[test]
fn tool_params_empty_object_maps_to_none() {
    let v = serde_json::json!({"type": "object", "properties": {}});
    assert!(projected_schema(&v).unwrap().is_none());
}

#[test]
fn tool_params_null_maps_to_none() {
    assert!(
        projected_schema(&serde_json::Value::Null)
            .unwrap()
            .is_none()
    );
}

#[test]
fn tool_params_object_with_scalar_properties_round_trips() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "city":      { "type": "string",  "description": "City name" },
            "max_price": { "type": "integer", "description": "Cap, USD"  }
        },
        "required": ["city"]
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::Object as i32);
    assert_eq!(schema.required, vec!["city".to_string()]);
    assert_eq!(schema.properties.len(), 2);

    let city = schema.properties.get("city").expect("city prop");
    assert_eq!(city.r#type, proto::Type::String as i32);
    assert_eq!(city.description, "City name");

    let max_price = schema.properties.get("max_price").expect("max_price prop");
    assert_eq!(max_price.r#type, proto::Type::Integer as i32);
}

#[test]
fn tool_params_array_with_typed_items() {
    let v = serde_json::json!({
        "type": "array",
        "items": { "type": "string" }
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::Array as i32);
    let items = schema.items.expect("items");
    assert_eq!(items.r#type, proto::Type::String as i32);
}

#[test]
fn tool_params_enum_strings_preserved() {
    let v = serde_json::json!({
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    assert_eq!(schema.r#type, proto::Type::String as i32);
    assert_eq!(
        schema.r#enum,
        vec!["celsius".to_string(), "fahrenheit".to_string()]
    );
}

#[test]
fn tool_params_resolves_defs_ref_properties() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "destination": { "$ref": "#/$defs/Destination" }
        },
        "required": ["destination"],
        "$defs": {
            "Destination": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "country_code": { "type": "string" }
                },
                "required": ["city"]
            }
        }
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    let destination = schema
        .properties
        .get("destination")
        .expect("destination prop");

    assert_eq!(destination.r#type, proto::Type::Object as i32);
    assert_eq!(destination.required, vec!["city".to_string()]);
    assert_eq!(
        destination
            .properties
            .get("city")
            .expect("city prop")
            .r#type,
        proto::Type::String as i32
    );
}

#[test]
fn tool_params_nullable_type_array_preserves_non_null_type() {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "nickname": { "type": ["null", "string"] }
        }
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");
    let nickname = schema.properties.get("nickname").expect("nickname prop");

    assert_eq!(nickname.r#type, proto::Type::String as i32);
    assert!(nickname.nullable);
}

#[test]
fn tool_params_any_of_uses_non_null_schema() {
    let v = serde_json::json!({
        "anyOf": [
            { "type": "null" },
            {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        ]
    });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");

    assert_eq!(schema.r#type, proto::Type::Object as i32);
    assert!(schema.nullable);
    assert_eq!(schema.required, vec!["query".to_string()]);
    assert_eq!(
        schema.properties.get("query").expect("query prop").r#type,
        proto::Type::String as i32
    );
}

#[test]
fn tool_params_array_without_items_defaults_to_string_items() {
    let v = serde_json::json!({ "type": "array" });

    let schema = projected_schema(&v)
        .expect("schema conversion")
        .expect("schema");

    assert_eq!(schema.r#type, proto::Type::Array as i32);
    assert_eq!(
        schema.items.expect("items").r#type,
        proto::Type::String as i32
    );
}
