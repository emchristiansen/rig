//! Projects typed-tool parameters into the GenerateContent wire schema.
//! Reference definitions must be acyclic.
//!
//! The projection carries every keyword of Gemini's documented `parameters`
//! Schema subset under its REST name, and every keyword it carries has a
//! field in the gRPC wire's typed `Schema`. It drops exactly the named
//! losses: `additionalProperties: false` and the annotation keywords
//! `$schema`, `$id` and `$comment`. Any other construct is a request error
//! naming its schema path; nothing else is omitted.

use crate::error::EncodeError;
use serde_json::{Map, Value, json};

/// Encode tool parameters, omitting null and the exact empty object schema.
/// Reference definitions must be acyclic. A construct outside Gemini's
/// `parameters` subset, other than a named loss, is a request error naming
/// its schema path.
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

/// Numeric-domain enums use the original schema, after the same reference
/// validation, since `parametersJsonSchema` carries it whole; every other
/// declaration is projected.
pub(super) fn declaration_parameters(value: Value) -> Result<DeclarationParameters, EncodeError> {
    if requires_json_schema(&value, "")? {
        Ok(DeclarationParameters::JsonSchema(value))
    } else {
        parameters(value).map(DeclarationParameters::Projected)
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

/// Annotations a `$ref` site states for its own use of the definition.
const REF_SITE_ANNOTATIONS: [&str; 4] = ["title", "description", "default", "example"];

/// What resolving a `$ref` does with the keywords beside it.
#[derive(Clone, Copy)]
enum RefSiblings {
    /// They join the definition, as the projection reads them; a keyword
    /// whose value differs from the definition's is refused.
    Join,
    /// The definition replaces them, as the numeric-enum route detection
    /// has always read a reference.
    Replace,
}

fn flatten_schema(
    mut schema: Value,
    path: &str,
    siblings: RefSiblings,
) -> Result<Value, EncodeError> {
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

    resolve_refs(&mut schema, defs_obj, path, siblings)?;

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
    siblings: RefSiblings,
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
                resolve_refs(&mut resolved, defs, path, siblings)?;
                // A keyword beside `$ref` joins the definition. An
                // annotation there (title, description, default, example)
                // documents this use of the definition and takes its place;
                // a validation keyword that would replace a different value
                // is refused.
                let joined = obj.iter().filter(|(keyword, _)| {
                    matches!(siblings, RefSiblings::Join) && keyword.as_str() != "$ref"
                });
                for (keyword, sibling) in joined {
                    let sibling_path = schema_path(path, keyword);
                    let mut sibling = sibling.clone();
                    resolve_refs(&mut sibling, defs, &sibling_path, siblings)?;
                    let target = resolved.as_object_mut().ok_or_else(|| {
                        schema_error(
                            &sibling_path,
                            format!("{ref_str} is not a schema object this keyword can join"),
                        )
                    })?;
                    match target.get(keyword) {
                        None => {
                            target.insert(keyword.clone(), sibling);
                        }
                        Some(existing) if *existing == sibling => {}
                        Some(_) if REF_SITE_ANNOTATIONS.contains(&keyword.as_str()) => {
                            target.insert(keyword.clone(), sibling);
                        }
                        Some(_) => {
                            return Err(schema_error(
                                &sibling_path,
                                format!("differs from the same keyword of {ref_str}"),
                            ));
                        }
                    }
                }
                *value = resolved;
                return Ok(());
            }

            for (name, v) in obj.iter_mut() {
                resolve_refs(v, defs, &schema_path(path, name), siblings)?;
            }
        }
        Value::Array(arr) => {
            for (index, item) in arr.iter_mut().enumerate() {
                resolve_refs(item, defs, &schema_path(path, &index.to_string()), siblings)?;
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
    let flattened = flatten_schema(value.clone(), path, RefSiblings::Replace)?;
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

/// Gemini's schema type names, in JSON Schema and in Gemini enum spelling.
const TYPES: [&str; 14] = [
    "string", "number", "integer", "boolean", "array", "object", "null", "STRING", "NUMBER",
    "INTEGER", "BOOLEAN", "ARRAY", "OBJECT", "NULL",
];

/// The REST names of the projected Schema members, in the order they are
/// written: the members of the earlier projection first, in their earlier
/// order, then the rest of Gemini's documented subset.
const MEMBERS: [&str; 22] = [
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "items",
    "propertyOrdering",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "example",
    "default",
    "anyOf",
];

fn is_null_type(name: &Value) -> bool {
    matches!(name.as_str(), Some("null" | "NULL"))
}

/// A schema's `type`, and whether a type array also admits null. A type
/// array names one type besides null, since a Gemini schema has one type.
fn schema_type(value: &Value, path: &str) -> Result<(Value, bool), EncodeError> {
    let named = |name: &Value, path: &str| {
        name.as_str()
            .filter(|name| TYPES.contains(name))
            .map(|_| name.clone())
            .ok_or_else(|| schema_error(path, format!("{name} is not a Gemini schema type")))
    };
    let Value::Array(names) = value else {
        return named(value, path).map(|name| (name, false));
    };
    for (index, name) in names.iter().enumerate() {
        named(name, &schema_path(path, &index.to_string()))?;
    }
    let null = names.iter().find(|name| is_null_type(name));
    let mut types = names.iter().filter(|name| !is_null_type(name));
    match (types.next(), types.next(), null) {
        (Some(name), None, null) => Ok((name.clone(), null.is_some())),
        (None, _, Some(null)) => Ok((null.clone(), false)),
        (None, _, None) => Err(schema_error(path, "names no type")),
        (Some(_), Some(_), _) => Err(schema_error(
            path,
            "names more than one type besides null, and a Gemini schema has one type",
        )),
    }
}

fn string(value: &Value, path: &str) -> Result<Value, EncodeError> {
    if value.is_string() {
        Ok(value.clone())
    } else {
        Err(schema_error(path, "must be a string"))
    }
}

fn strings(value: &Value, path: &str) -> Result<Value, EncodeError> {
    let values = value
        .as_array()
        .ok_or_else(|| schema_error(path, "must be an array of strings"))?;
    for (index, value) in values.iter().enumerate() {
        string(value, &schema_path(path, &index.to_string()))?;
    }
    Ok(value.clone())
}

/// A count Gemini's int64 field holds. Gemini's Schema message cannot tell
/// a zero upper bound from an unset one, so `nonzero` bounds refuse zero.
fn count(value: &Value, path: &str, nonzero: bool) -> Result<Value, EncodeError> {
    match value.as_u64() {
        Some(bound) if i64::try_from(bound).is_err() => {
            Err(schema_error(path, "exceeds Gemini's int64 range"))
        }
        Some(0) if nonzero => Err(schema_error(
            path,
            "0 is indistinguishable from an unset bound in Gemini's Schema",
        )),
        Some(_) => Ok(value.clone()),
        None => Err(schema_error(path, "must be a non-negative integer")),
    }
}

fn number(value: &Value, path: &str) -> Result<Value, EncodeError> {
    if value.is_number() {
        Ok(value.clone())
    } else {
        Err(schema_error(path, "must be a number"))
    }
}

fn child_schema(value: &Value, path: &str) -> Result<Value, EncodeError> {
    if value.is_object() {
        schema_value(value.clone(), path)
    } else {
        Err(schema_error(path, "must be a schema object"))
    }
}

/// Project one schema object. Every member is read: a keyword of Gemini's
/// documented subset is carried under its REST name, a named loss is
/// dropped, and anything else is refused with its path.
fn schema_value(value: Value, path: &str) -> Result<Value, EncodeError> {
    let flattened = flatten_schema(value, path, RefSiblings::Join)?;
    let obj = flattened
        .as_object()
        .ok_or_else(|| schema_error(path, "Expected a JSON object for Schema"))?;
    let mut carried = Map::new();
    let mut nullable = false;
    let mut null_enum_value = None;
    for (keyword, value) in obj {
        let at = schema_path(path, keyword);
        let member = match keyword.as_str() {
            "type" => {
                let (name, null) = schema_type(value, &at)?;
                nullable |= null;
                name
            }
            "nullable" => {
                nullable |= value
                    .as_bool()
                    .ok_or_else(|| schema_error(&at, "must be a boolean"))?;
                continue;
            }
            "format" | "title" | "description" | "pattern" => string(value, &at)?,
            "required" | "propertyOrdering" => strings(value, &at)?,
            "minItems" | "minProperties" | "minLength" => count(value, &at, false)?,
            "maxItems" | "maxProperties" | "maxLength" => count(value, &at, true)?,
            "minimum" | "maximum" => number(value, &at)?,
            "example" | "default" => value.clone(),
            "enum" => {
                let values = value
                    .as_array()
                    .filter(|values| !values.is_empty())
                    .ok_or_else(|| schema_error(&at, "must be a nonempty array"))?;
                let mut kept = Vec::with_capacity(values.len());
                for (index, value) in values.iter().enumerate() {
                    let value_path = schema_path(&at, &index.to_string());
                    match value {
                        Value::String(_) => kept.push(value.clone()),
                        Value::Null => null_enum_value = Some(value_path),
                        _ => {
                            return Err(schema_error(
                                &value_path,
                                "Gemini enum values are strings",
                            ));
                        }
                    }
                }
                if kept.is_empty() {
                    return Err(schema_error(
                        &at,
                        "Gemini's Schema cannot express an enum of only null",
                    ));
                }
                Value::Array(kept)
            }
            "items" => child_schema(value, &at)?,
            "properties" => {
                let properties = value
                    .as_object()
                    .ok_or_else(|| schema_error(&at, "must be an object of schemas"))?;
                let mut sorted = std::collections::BTreeMap::new();
                for (name, value) in properties {
                    sorted.insert(name.clone(), child_schema(value, &schema_path(&at, name))?);
                }
                Value::Object(sorted.into_iter().collect())
            }
            "anyOf" => {
                let branches = value
                    .as_array()
                    .filter(|branches| !branches.is_empty())
                    .ok_or_else(|| schema_error(&at, "must be a nonempty array of schemas"))?;
                let branches = branches
                    .iter()
                    .enumerate()
                    .map(|(index, branch)| {
                        child_schema(branch, &schema_path(&at, &index.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Value::Array(branches)
            }
            // The named losses.
            "additionalProperties" if *value == Value::Bool(false) => continue,
            "$schema" | "$id" | "$comment" => continue,
            "additionalProperties" => {
                return Err(schema_error(
                    &at,
                    "only `false` is accepted, as a named loss; Gemini's Schema has no field for it",
                ));
            }
            "$ref" => {
                return Err(schema_error(&at, "the reference cannot be resolved"));
            }
            _ => {
                return Err(schema_error(
                    &at,
                    "the keyword is outside Gemini's `parameters` Schema subset",
                ));
            }
        };
        carried.insert(keyword.clone(), member);
    }
    if let Some(value_path) = null_enum_value
        && !nullable
    {
        return Err(schema_error(
            &value_path,
            "a null enum value needs a type that admits null",
        ));
    }
    if !carried.contains_key("type") {
        // The type a schema without one implies; with neither, none is sent.
        if carried.contains_key("properties") {
            carried.insert("type".to_owned(), json!("object"));
        } else if carried.contains_key("enum") {
            carried.insert("type".to_owned(), json!("string"));
        }
    }
    if nullable {
        carried.insert("nullable".to_owned(), Value::Bool(true));
    }
    let array = carried
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|name| name.eq_ignore_ascii_case("array"));
    if array && !carried.contains_key("items") {
        carried.insert("items".to_owned(), json!({"type":"string"}));
    }
    let projected: Map<String, Value> = MEMBERS
        .iter()
        .filter_map(|member| Some(((*member).to_owned(), carried.shift_remove(*member)?)))
        .collect();
    debug_assert!(carried.is_empty(), "every carried member is written");
    Ok(Value::Object(projected))
}
