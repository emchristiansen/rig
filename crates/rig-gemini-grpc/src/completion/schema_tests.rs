use super::*;
use crate::proto;
use anyhow::{Context, ensure};
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use serde_json::Value;

// Inspect the schema emitted by the real shared encoder and protobuf transcode.
fn projected_schema(parameters: &Value) -> Result<Option<proto::Schema>, EncodeError> {
    let mut request = CompletionRequest::new("hello");
    request.tools = vec![ToolDefinition::new(
        ToolName::new("schema_fixture").map_err(EncodeError::request)?,
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
fn tool_params_empty_object_maps_to_none() -> anyhow::Result<()> {
    let v = serde_json::json!({"type": "object", "properties": {}});
    let schema = projected_schema(&v)?;
    ensure!(schema.is_none(), "expected no schema, got {schema:?}");
    Ok(())
}

#[test]
fn tool_params_null_maps_to_none() -> anyhow::Result<()> {
    let schema = projected_schema(&serde_json::Value::Null)?;
    ensure!(schema.is_none(), "expected no schema, got {schema:?}");
    Ok(())
}

#[test]
fn tool_params_object_with_scalar_properties_round_trips() -> anyhow::Result<()> {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "city":      { "type": "string",  "description": "City name" },
            "max_price": { "type": "integer", "description": "Cap, USD"  }
        },
        "required": ["city"]
    });

    let schema = projected_schema(&v)?.context("schema is absent")?;
    ensure!(
        schema.r#type == proto::Type::Object as i32,
        "schema type differs: actual {:?}, expected {:?}",
        schema.r#type,
        proto::Type::Object as i32
    );
    ensure!(
        schema.required == vec!["city".to_string()],
        "schema.required differs: actual {:?}, expected {:?}",
        schema.required,
        vec!["city".to_string()]
    );
    ensure!(
        schema.properties.len() == 2,
        "schema.properties.len() differs: actual {:?}, expected {:?}",
        schema.properties.len(),
        2
    );

    let city = schema
        .properties
        .get("city")
        .with_context(|| format!("city prop is absent: {schema:?}"))?;
    ensure!(
        city.r#type == proto::Type::String as i32,
        "city type differs: actual {:?}, expected {:?}",
        city.r#type,
        proto::Type::String as i32
    );
    ensure!(
        city.description == "City name",
        "city.description differs: actual {:?}, expected {:?}",
        city.description,
        "City name"
    );

    let max_price = schema
        .properties
        .get("max_price")
        .with_context(|| format!("max_price prop is absent: {schema:?}"))?;
    ensure!(
        max_price.r#type == proto::Type::Integer as i32,
        "max_price type differs: actual {:?}, expected {:?}",
        max_price.r#type,
        proto::Type::Integer as i32
    );
    Ok(())
}

#[test]
fn tool_params_array_with_typed_items() -> anyhow::Result<()> {
    let v = serde_json::json!({
        "type": "array",
        "items": { "type": "string" }
    });

    let schema = projected_schema(&v)?.context("schema is absent")?;
    ensure!(
        schema.r#type == proto::Type::Array as i32,
        "schema type differs: actual {:?}, expected {:?}",
        schema.r#type,
        proto::Type::Array as i32
    );
    let items = schema.items.as_ref().context("schema items are absent")?;
    ensure!(
        items.r#type == proto::Type::String as i32,
        "items type differs: actual {:?}, expected {:?}",
        items.r#type,
        proto::Type::String as i32
    );
    Ok(())
}

#[test]
fn tool_params_enum_strings_preserved() -> anyhow::Result<()> {
    let v = serde_json::json!({
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
    });

    let schema = projected_schema(&v)?.context("schema is absent")?;
    ensure!(
        schema.r#type == proto::Type::String as i32,
        "schema type differs: actual {:?}, expected {:?}",
        schema.r#type,
        proto::Type::String as i32
    );
    let expected = vec!["celsius".to_string(), "fahrenheit".to_string()];
    ensure!(
        schema.r#enum == expected,
        "schema enum differs: actual {:?}, expected {expected:?}",
        schema.r#enum
    );
    Ok(())
}

#[test]
fn tool_params_resolves_defs_ref_properties() -> anyhow::Result<()> {
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

    let schema = projected_schema(&v)?.context("schema is absent")?;
    let destination = schema
        .properties
        .get("destination")
        .with_context(|| format!("destination prop is absent: {schema:?}"))?;

    ensure!(
        destination.r#type == proto::Type::Object as i32,
        "destination type differs: actual {:?}, expected {:?}",
        destination.r#type,
        proto::Type::Object as i32
    );
    ensure!(
        destination.required == vec!["city".to_string()],
        "destination.required differs: actual {:?}, expected {:?}",
        destination.required,
        vec!["city".to_string()]
    );
    let city = destination
        .properties
        .get("city")
        .with_context(|| format!("city prop is absent: {destination:?}"))?;
    ensure!(
        city.r#type == proto::Type::String as i32,
        "city type differs: actual {:?}, expected {:?}",
        city.r#type,
        proto::Type::String as i32
    );
    Ok(())
}

#[test]
fn tool_params_nullable_type_array_preserves_non_null_type() -> anyhow::Result<()> {
    let v = serde_json::json!({
        "type": "object",
        "properties": {
            "nickname": { "type": ["null", "string"] }
        }
    });

    let schema = projected_schema(&v)?.context("schema is absent")?;
    let nickname = schema
        .properties
        .get("nickname")
        .with_context(|| format!("nickname prop is absent: {schema:?}"))?;

    ensure!(
        nickname.r#type == proto::Type::String as i32,
        "nickname type differs: actual {:?}, expected {:?}",
        nickname.r#type,
        proto::Type::String as i32
    );
    ensure!(nickname.nullable, "nickname is not nullable: {nickname:?}");
    Ok(())
}

#[test]
fn tool_params_any_of_uses_non_null_schema() -> anyhow::Result<()> {
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

    let schema = projected_schema(&v)?.context("schema is absent")?;

    ensure!(
        schema.r#type == proto::Type::Object as i32,
        "schema type differs: actual {:?}, expected {:?}",
        schema.r#type,
        proto::Type::Object as i32
    );
    ensure!(schema.nullable, "schema is not nullable: {schema:?}");
    ensure!(
        schema.required == vec!["query".to_string()],
        "schema.required differs: actual {:?}, expected {:?}",
        schema.required,
        vec!["query".to_string()]
    );
    let query = schema
        .properties
        .get("query")
        .with_context(|| format!("query prop is absent: {schema:?}"))?;
    ensure!(
        query.r#type == proto::Type::String as i32,
        "query type differs: actual {:?}, expected {:?}",
        query.r#type,
        proto::Type::String as i32
    );
    Ok(())
}

#[test]
fn tool_params_array_without_items_defaults_to_string_items() -> anyhow::Result<()> {
    let v = serde_json::json!({ "type": "array" });

    let schema = projected_schema(&v)?.context("schema is absent")?;

    ensure!(
        schema.r#type == proto::Type::Array as i32,
        "schema type differs: actual {:?}, expected {:?}",
        schema.r#type,
        proto::Type::Array as i32
    );
    let items = schema.items.as_ref().context("schema items are absent")?;
    ensure!(
        items.r#type == proto::Type::String as i32,
        "items type differs: actual {:?}, expected {:?}",
        items.r#type,
        proto::Type::String as i32
    );
    Ok(())
}
