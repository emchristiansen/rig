//! Projects typed-tool parameters into the GenerateContent wire schema.
//! Reference definitions must be acyclic.

use crate::error::EncodeError;
use serde_json::{Map, Value, json};

/// Encode tool parameters, omitting null and the exact empty object schema.
/// Reference definitions must be acyclic. Schema-object conversion failures
/// return a request error; unmodeled keys and non-schema children are omitted.
pub(super) fn parameters(parameters: Value) -> Result<Option<Value>, EncodeError> {
    if parameters.is_null() || parameters == json!({"type": "object", "properties": {}}) {
        Ok(None)
    } else {
        schema_value(parameters, "").map(Some)
    }
}

pub(super) enum DeclarationParameters {
    Projected(Option<Value>),
    JsonSchema(Value),
}

/// Numeric-domain enums use the original schema after projection validation.
pub(super) fn declaration_parameters(value: Value) -> Result<DeclarationParameters, EncodeError> {
    let projected = parameters(value.clone())?;
    if requires_json_schema(&value, "")? {
        Ok(DeclarationParameters::JsonSchema(value))
    } else {
        Ok(DeclarationParameters::Projected(projected))
    }
}

fn schema_path(parent: &str, segment: &str) -> String {
    format!("{parent}/{}", segment.replace('~', "~0").replace('/', "~1"))
}

fn schema_error(path: &str, reason: impl std::fmt::Display) -> EncodeError {
    if path.is_empty() {
        EncodeError::request(reason.to_string())
    } else {
        EncodeError::request(format!("schema path {path}: {reason}"))
    }
}

fn flatten_schema(mut schema: Value, path: &str) -> Result<Value, EncodeError> {
    let defs = schema
        .as_object()
        .and_then(|obj| obj.get("$defs").or_else(|| obj.get("definitions")))
        .cloned();

    let Some(defs_value) = defs else {
        return Ok(schema);
    };

    let Some(defs_obj) = defs_value.as_object() else {
        let path = if path.is_empty() {
            String::new()
        } else {
            schema_path(
                path,
                if schema.get("$defs").is_some() {
                    "$defs"
                } else {
                    "definitions"
                },
            )
        };
        return Err(schema_error(&path, "$defs must be an object"));
    };

    resolve_refs(&mut schema, defs_obj, path)?;

    if let Some(obj) = schema.as_object_mut() {
        obj.shift_remove("$defs");
        obj.shift_remove("definitions");
    }

    Ok(schema)
}

fn resolve_refs(
    value: &mut Value,
    defs: &Map<String, Value>,
    path: &str,
) -> Result<(), EncodeError> {
    match value {
        Value::Object(obj) => {
            if let Some(ref_value) = obj.get("$ref")
                && let Some(ref_str) = ref_value.as_str()
            {
                let reference_path = if path.is_empty() {
                    String::new()
                } else {
                    schema_path(path, "$ref")
                };
                let def_name = parse_ref_path(ref_str, &reference_path)?;

                let def = defs.get(&def_name).ok_or_else(|| {
                    schema_error(&reference_path, format!("Reference not found: {ref_str}"))
                })?;

                let mut resolved = def.clone();
                resolve_refs(&mut resolved, defs, path)?;
                *value = resolved;
                return Ok(());
            }

            for (name, v) in obj.iter_mut() {
                resolve_refs(v, defs, &schema_path(path, name))?;
            }
        }
        Value::Array(arr) => {
            for (index, item) in arr.iter_mut().enumerate() {
                resolve_refs(item, defs, &schema_path(path, &index.to_string()))?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn parse_ref_path(ref_str: &str, path: &str) -> Result<String, EncodeError> {
    if let Some(fragment) = ref_str.strip_prefix('#') {
        if let Some(name) = fragment.strip_prefix("/$defs/") {
            Ok(name.to_string())
        } else if let Some(name) = fragment.strip_prefix("/definitions/") {
            Ok(name.to_string())
        } else {
            Err(schema_error(
                path,
                format!("Unsupported reference format: {ref_str}"),
            ))
        }
    } else {
        Err(schema_error(
            path,
            format!("Only fragment references (#/...) are supported: {ref_str}"),
        ))
    }
}

fn extract_type(type_value: &Value) -> Option<String> {
    if let Some(t) = type_value.as_str() {
        return Some(t.to_string());
    }

    type_value.as_array().and_then(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .find(|t| *t != "null")
            .or_else(|| arr.iter().find_map(|v| v.as_str()))
            .map(str::to_owned)
    })
}

fn schema_is_null(obj: &Map<String, Value>) -> bool {
    obj.get("type")
        .and_then(extract_type)
        .as_deref()
        .is_some_and(|t| t == "null")
}

fn schema_is_nullable(obj: &Map<String, Value>) -> bool {
    obj.get("nullable")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        || obj
            .get("type")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some("null")))
        || ["anyOf", "oneOf", "allOf"].iter().any(|key| {
            obj.get(*key).and_then(|v| v.as_array()).is_some_and(|arr| {
                arr.iter()
                    .filter_map(|schema| schema.as_object())
                    .any(schema_is_null)
            })
        })
}

fn extract_type_from_composition(composition: &Value) -> Option<String> {
    composition.as_array().and_then(|arr| {
        arr.iter().find_map(|schema| {
            let obj = schema.as_object()?;
            if schema_is_null(obj) {
                return None;
            }

            obj.get("type").and_then(extract_type).or_else(|| {
                if obj.contains_key("properties") {
                    Some("object".to_string())
                } else if obj.contains_key("enum") {
                    // Enum schemas without explicit type are string-backed
                    Some("string".to_string())
                } else {
                    None
                }
            })
        })
    })
}

fn extract_schema_from_composition(
    composition: &Value,
    path: &str,
) -> Option<(Map<String, Value>, String)> {
    composition.as_array().and_then(|arr| {
        arr.iter().enumerate().find_map(|(index, schema)| {
            let obj = schema.as_object()?;
            if schema_is_null(obj) {
                None
            } else {
                Some((obj.clone(), schema_path(path, &index.to_string())))
            }
        })
    })
}

fn extract_schema_from_composition_obj(
    obj: &Map<String, Value>,
    path: &str,
) -> Option<(Map<String, Value>, String)> {
    obj.get("anyOf")
        .and_then(|value| extract_schema_from_composition(value, &schema_path(path, "anyOf")))
        .or_else(|| {
            obj.get("oneOf").and_then(|value| {
                extract_schema_from_composition(value, &schema_path(path, "oneOf"))
            })
        })
        .or_else(|| {
            obj.get("allOf").and_then(|value| {
                extract_schema_from_composition(value, &schema_path(path, "allOf"))
            })
        })
}

fn infer_type(obj: &Map<String, Value>) -> String {
    if let Some(type_val) = obj.get("type")
        && let Some(type_str) = extract_type(type_val)
    {
        return type_str;
    }

    if let Some(any_of) = obj.get("anyOf")
        && let Some(type_str) = extract_type_from_composition(any_of)
    {
        return type_str;
    }

    if let Some(one_of) = obj.get("oneOf")
        && let Some(type_str) = extract_type_from_composition(one_of)
    {
        return type_str;
    }

    if let Some(all_of) = obj.get("allOf")
        && let Some(type_str) = extract_type_from_composition(all_of)
    {
        return type_str;
    }

    if obj.contains_key("properties") {
        "object".to_string()
    } else if obj.contains_key("enum") {
        "string".to_string()
    } else {
        String::new()
    }
}

fn explicitly_string_only(obj: &Map<String, Value>) -> bool {
    match obj.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => {
            kinds.iter().any(|kind| kind.as_str() == Some("string"))
                && kinds
                    .iter()
                    .all(|kind| matches!(kind.as_str(), Some("string" | "null")))
        }
        _ => false,
    }
}

fn requires_json_schema(value: &Value, path: &str) -> Result<bool, EncodeError> {
    let flattened = flatten_schema(value.clone(), path)?;
    let Some(obj) = flattened.as_object() else {
        return Ok(false);
    };
    let mut required = !explicitly_string_only(obj)
        && obj
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(Value::is_number));
    if let Some(properties) = obj.get("properties").and_then(Value::as_object) {
        for (name, value) in properties {
            if value.is_object() {
                required |= requires_json_schema(
                    value,
                    &schema_path(&schema_path(path, "properties"), name),
                )?;
            }
        }
    }
    if let Some(value) = obj.get("items")
        && value.is_object()
    {
        required |= requires_json_schema(value, &schema_path(path, "items"))?;
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(alternatives) = obj.get(keyword).and_then(Value::as_array) {
            for (index, value) in alternatives.iter().enumerate() {
                if value.is_object() {
                    required |= requires_json_schema(
                        value,
                        &schema_path(&schema_path(path, keyword), &index.to_string()),
                    )?;
                }
            }
        }
    }
    Ok(required)
}

fn schema_value(value: Value, path: &str) -> Result<Value, EncodeError> {
    let flattened = flatten_schema(value, path)?;
    let obj = flattened
        .as_object()
        .ok_or_else(|| schema_error(path, "Expected a JSON object for Schema"))?;
    let composition_source = extract_schema_from_composition_obj(obj, path);
    let (props_source, props_path) = if obj.get("properties").is_none() {
        composition_source
            .as_ref()
            .map_or((obj, path), |(source, source_path)| {
                (source, source_path.as_str())
            })
    } else {
        (obj, path)
    };
    let schema_type = infer_type(obj);
    // The schema conversion omits non-schema child values.
    let items = obj
        .get("items")
        .map(|value| (value, schema_path(path, "items")))
        .or_else(|| {
            props_source
                .get("items")
                .map(|value| (value, schema_path(props_path, "items")))
        })
        .filter(|(value, _)| value.is_object())
        .map(|(value, path)| schema_value(value.clone(), &path))
        .transpose()?;
    let items = if schema_type == "array" && items.is_none() {
        Some(json!({"type":"string"}))
    } else {
        items
    };
    let nullable = (schema_is_nullable(obj)
        || composition_source
            .as_ref()
            .is_some_and(|(source, _)| schema_is_nullable(source)))
    .then_some(Value::Bool(true));
    let string_field = |name: &str| {
        obj.get(name)
            .or_else(|| props_source.get(name))
            .and_then(Value::as_str)
            .map(|value| Value::String(value.to_owned()))
    };
    let enumerated = obj
        .get("enum")
        .or_else(|| props_source.get("enum"))
        .and_then(Value::as_array)
        .map(|values| {
            Value::Array(
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(|value| Value::String(value.to_owned())))
                    .collect(),
            )
        });
    let limit = |name: &str| {
        obj.get(name)
            .and_then(Value::as_i64)
            .map(|value| json!(value as i32))
    };
    let properties = props_source
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            let mut sorted = std::collections::BTreeMap::new();
            for (name, value) in properties {
                if value.is_object() {
                    let path = schema_path(&schema_path(props_path, "properties"), name);
                    sorted.insert(name.clone(), schema_value(value.clone(), &path)?);
                }
            }
            Ok::<_, EncodeError>(Value::Object(sorted.into_iter().collect()))
        })
        .transpose()?;
    let required = props_source
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            Value::Array(
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(|value| Value::String(value.to_owned())))
                    .collect(),
            )
        });
    Ok(Value::Object(super::object([
        ("type", Some(Value::String(schema_type))),
        ("format", string_field("format")),
        ("description", string_field("description")),
        ("nullable", nullable),
        ("enum", enumerated),
        ("max_items", limit("maxItems")),
        ("min_items", limit("minItems")),
        ("properties", properties),
        ("required", required),
        ("items", items),
    ])))
}
