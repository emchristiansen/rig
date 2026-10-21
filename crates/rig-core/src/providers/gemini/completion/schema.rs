//! Projects typed-tool parameters into the GenerateContent wire schema.
//! Reference definitions must be acyclic.

use crate::error::EncodeError;
use serde_json::{Map, Value, json};

/// Encode tool parameters, omitting null and the exact empty object schema.
/// Reference definitions must be acyclic. Top-level conversion failures
/// return a request error; unmodeled keys and non-schema children are omitted.
pub(super) fn parameters(parameters: Value) -> Result<Option<Value>, EncodeError> {
    if parameters.is_null() || parameters == json!({"type": "object", "properties": {}}) {
        Ok(None)
    } else {
        schema_value(parameters).map(Some)
    }
}

fn flatten_schema(mut schema: Value) -> Result<Value, EncodeError> {
    let defs = schema
        .as_object()
        .and_then(|obj| obj.get("$defs").or_else(|| obj.get("definitions")))
        .cloned();

    let Some(defs_value) = defs else {
        return Ok(schema);
    };

    let Some(defs_obj) = defs_value.as_object() else {
        return Err(EncodeError::request("$defs must be an object"));
    };

    resolve_refs(&mut schema, defs_obj)?;

    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$defs");
        obj.remove("definitions");
    }

    Ok(schema)
}

fn resolve_refs(
    value: &mut Value,
    defs: &Map<String, Value>,
) -> Result<(), EncodeError> {
    match value {
        Value::Object(obj) => {
            if let Some(ref_value) = obj.get("$ref")
                && let Some(ref_str) = ref_value.as_str()
            {
                let def_name = parse_ref_path(ref_str)?;

                let def = defs.get(&def_name).ok_or_else(|| {
                    EncodeError::request(format!("Reference not found: {ref_str}"))
                })?;

                let mut resolved = def.clone();
                resolve_refs(&mut resolved, defs)?;
                *value = resolved;
                return Ok(());
            }

            for (_, v) in obj.iter_mut() {
                resolve_refs(v, defs)?;
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                resolve_refs(item, defs)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn parse_ref_path(ref_str: &str) -> Result<String, EncodeError> {
    if let Some(fragment) = ref_str.strip_prefix('#') {
        if let Some(name) = fragment.strip_prefix("/$defs/") {
            Ok(name.to_string())
        } else if let Some(name) = fragment.strip_prefix("/definitions/") {
            Ok(name.to_string())
        } else {
            Err(EncodeError::request(format!(
                "Unsupported reference format: {ref_str}"
            )))
        }
    } else {
        Err(EncodeError::request(format!(
            "Only fragment references (#/...) are supported: {ref_str}"
        )))
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
) -> Option<Map<String, Value>> {
    composition.as_array().and_then(|arr| {
        arr.iter().find_map(|schema| {
            let obj = schema.as_object()?;
            if schema_is_null(obj) {
                None
            } else {
                Some(obj.clone())
            }
        })
    })
}

fn extract_schema_from_composition_obj(
    obj: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    obj.get("anyOf")
        .and_then(extract_schema_from_composition)
        .or_else(|| obj.get("oneOf").and_then(extract_schema_from_composition))
        .or_else(|| obj.get("allOf").and_then(extract_schema_from_composition))
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


fn schema_value(value: Value) -> Result<Value, EncodeError> {
    let flattened = flatten_schema(value)?;
    let obj = flattened.as_object()
        .ok_or_else(|| EncodeError::request("Expected a JSON object for Schema"))?;
    let composition_source = extract_schema_from_composition_obj(obj);
    let props_source = if obj.get("properties").is_none() {
        composition_source.as_ref().unwrap_or(obj)
    } else {
        obj
    };
    let schema_type = infer_type(obj);
    // The schema conversion omits non-schema child values.
    let items = obj.get("items").or_else(|| props_source.get("items"))
        .and_then(|value| schema_value(value.clone()).ok());
    let items = if schema_type == "array" && items.is_none() {
        Some(json!({"type":"string"}))
    } else {
        items
    };
    let nullable = (schema_is_nullable(obj)
        || composition_source.as_ref().is_some_and(schema_is_nullable))
        .then_some(Value::Bool(true));
    let string_field = |name: &str| {
        obj.get(name).or_else(|| props_source.get(name))
            .and_then(Value::as_str).map(|value| Value::String(value.to_owned()))
    };
    let enumerated = obj.get("enum").or_else(|| props_source.get("enum"))
        .and_then(Value::as_array).map(|values| Value::Array(values.iter()
            .filter_map(|value| value.as_str().map(|value| Value::String(value.to_owned())))
            .collect()));
    let limit = |name: &str| obj.get(name).and_then(Value::as_i64)
        .map(|value| json!(value as i32));
    let properties = props_source.get("properties").and_then(Value::as_object)
        .map(|properties| {
            let sorted: std::collections::BTreeMap<String, Value> = properties.iter()
                .filter_map(|(name, value)| schema_value(value.clone()).ok()
                    .map(|schema| (name.clone(), schema)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        });
    let required = props_source.get("required").and_then(Value::as_array)
        .map(|values| Value::Array(values.iter()
            .filter_map(|value| value.as_str().map(|value| Value::String(value.to_owned())))
            .collect()));
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
