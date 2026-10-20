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
fn tool_params_any_of_keeps_every_branch() -> anyhow::Result<()> {
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
    let expected = proto::Schema {
        any_of: vec![
            proto::Schema {
                r#type: proto::Type::Null as i32,
                ..proto::Schema::default()
            },
            proto::Schema {
                r#type: proto::Type::Object as i32,
                properties: [(
                    "query".to_owned(),
                    proto::Schema {
                        r#type: proto::Type::String as i32,
                        ..proto::Schema::default()
                    },
                )]
                .into_iter()
                .collect(),
                required: vec!["query".to_owned()],
                ..proto::Schema::default()
            },
        ],
        ..proto::Schema::default()
    };
    ensure!(
        schema == expected,
        "schema differs: actual {schema:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn tool_params_carry_every_documented_keyword_into_the_typed_schema() -> anyhow::Result<()> {
    use prost_types::value::Kind;
    let v = serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "Booking",
        "description": "A booking",
        "additionalProperties": false,
        "minProperties": 1,
        "maxProperties": 4,
        "propertyOrdering": ["tags", "count"],
        "example": { "count": 2 },
        "default": {},
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "string",
                    "format": "date",
                    "minLength": 2,
                    "maxLength": 8,
                    "pattern": "^[a-z]+$",
                    "enum": ["2026-01-01", "2026-12-31"]
                }
            },
            "count": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 10.5,
                "default": 2
            }
        },
        "required": ["tags"]
    });

    let schema = projected_schema(&v)?.context("schema is absent")?;
    let number = |value: f64| prost_types::Value {
        kind: Some(Kind::NumberValue(value)),
    };
    let tags = proto::Schema {
        r#type: proto::Type::Array as i32,
        min_items: 1,
        max_items: 3,
        items: Some(Box::new(proto::Schema {
            r#type: proto::Type::String as i32,
            format: "date".to_owned(),
            r#enum: vec!["2026-01-01".to_owned(), "2026-12-31".to_owned()],
            min_length: 2,
            max_length: 8,
            pattern: "^[a-z]+$".to_owned(),
            ..proto::Schema::default()
        })),
        ..proto::Schema::default()
    };
    let count = proto::Schema {
        r#type: proto::Type::Integer as i32,
        nullable: true,
        minimum: Some(0.0),
        maximum: Some(10.5),
        default: Some(number(2.0)),
        ..proto::Schema::default()
    };
    let expected = proto::Schema {
        r#type: proto::Type::Object as i32,
        title: "Booking".to_owned(),
        description: "A booking".to_owned(),
        min_properties: 1,
        max_properties: 4,
        property_ordering: vec!["tags".to_owned(), "count".to_owned()],
        example: Some(prost_types::Value {
            kind: Some(Kind::StructValue(prost_types::Struct {
                fields: [("count".to_owned(), number(2.0))].into_iter().collect(),
            })),
        }),
        default: Some(prost_types::Value {
            kind: Some(Kind::StructValue(prost_types::Struct::default())),
        }),
        properties: [("tags".to_owned(), tags), ("count".to_owned(), count)]
            .into_iter()
            .collect(),
        required: vec!["tags".to_owned()],
        ..proto::Schema::default()
    };
    ensure!(
        schema == expected,
        "schema differs: actual {schema:?}, expected {expected:?}"
    );
    Ok(())
}

#[test]
fn tool_params_refuse_a_bound_the_typed_schema_cannot_carry() -> anyhow::Result<()> {
    for (v, path, reason) in [
        (
            serde_json::json!({
                "type": "object",
                "properties": { "tags": { "type": "array", "items": { "type": "string" }, "maxItems": 0 } }
            }),
            "/properties/tags/maxItems",
            "0 is indistinguishable from an unset bound in Gemini's Schema",
        ),
        (
            serde_json::json!({ "type": "string", "maxLength": 0 }),
            "/maxLength",
            "0 is indistinguishable from an unset bound in Gemini's Schema",
        ),
        (
            serde_json::json!({ "type": "object", "properties": { "x": { "type": "string" } }, "maxProperties": 0 }),
            "/maxProperties",
            "0 is indistinguishable from an unset bound in Gemini's Schema",
        ),
        (
            serde_json::json!({ "type": "object", "properties": { "x": { "oneOf": [{ "type": "string" }] } } }),
            "/properties/x/oneOf",
            "the keyword is outside Gemini's `parameters` Schema subset",
        ),
    ] {
        let Err(error) = projected_schema(&v) else {
            anyhow::bail!("{v} was not refused");
        };
        let error = rig_core::error::ProviderError::from(error);
        ensure!(
            matches!(&error, rig_core::error::ProviderError::Request(_)),
            "not a request error: {error:?}"
        );
        let expected = format!(
            "Tool 'schema_fixture' could not be converted to a schema: schema path {path}: {reason}"
        );
        let actual = std::error::Error::source(&error)
            .context("request reason")?
            .to_string();
        ensure!(
            actual == expected,
            "refusal differs: actual {actual:?}, expected {expected:?}"
        );
    }
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
