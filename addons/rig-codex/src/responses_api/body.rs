//! A Responses request body edited field by field at its top level.
//!
//! Rig's encoder writes the body once. [`RequestBody`] reads it back keeping
//! every value as the encoder wrote its bytes, so a field this crate does not
//! edit leaves exactly as it came, numbers and nested key order included.
//! Fields are ordered as a `serde_json::Map` of this build orders them, which
//! is how Rig's encoder ordered them, with the conversation and the output
//! cap last.

use rig_core::json_utils::remove_preserving_order;
use serde::de::{DeserializeOwned, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The body fields written after every other, in this order: the
/// conversation, then the output cap, as Rig's own encoder orders a body.
pub(crate) const LAST_FIELDS: [&str; 2] = ["input", "max_output_tokens"];

/// A JSON object body whose values keep the bytes they were read with.
#[derive(Clone, Debug)]
pub(crate) struct RequestBody {
    /// The field names, in a `serde_json::Map` so they take the order the
    /// encoder's own map gave them. The values are unused.
    order: Map<String, Value>,
    /// Each field's value, as JSON text.
    values: BTreeMap<String, Box<RawValue>>,
}

impl PartialEq for RequestBody {
    fn eq(&self, other: &Self) -> bool {
        self.order.keys().eq(other.order.keys())
            && self
                .values
                .iter()
                .map(|(key, value)| (key, value.get()))
                .eq(other.values.iter().map(|(key, value)| (key, value.get())))
    }
}

impl RequestBody {
    /// Read `bytes` as a JSON object body. Refuses any other JSON value.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let Fields(fields) = serde_json::from_slice(bytes)?;
        let mut body = Self {
            order: Map::new(),
            values: BTreeMap::new(),
        };
        for (key, value) in fields {
            body.insert_raw(key, value);
        }
        Ok(body)
    }

    /// Whether the body has `key`.
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// The value of `key` as its JSON text.
    pub(crate) fn raw(&self, key: &str) -> Option<&RawValue> {
        self.values.get(key).map(AsRef::as_ref)
    }

    /// The value of `key`, read as `T`.
    pub(crate) fn get<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<T, serde_json::Error>> {
        self.raw(key).map(|raw| serde_json::from_str(raw.get()))
    }

    /// Set `key` to `value`. A key the body has keeps its place.
    pub(crate) fn insert<T: Serialize + ?Sized>(
        &mut self,
        key: &str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        let value = serde_json::value::to_raw_value(value)?;
        self.insert_raw(key.to_owned(), value);
        Ok(())
    }

    /// Set `key` to the JSON text `value`. A key the body has keeps its place.
    pub(crate) fn insert_raw(&mut self, key: String, value: Box<RawValue>) {
        self.order.insert(key.clone(), Value::Null);
        self.values.insert(key, value);
    }

    /// Take `key` out of the body, leaving the other fields in their order.
    pub(crate) fn remove(&mut self, key: &str) -> Option<Box<RawValue>> {
        remove_preserving_order(&mut self.order, key);
        self.values.remove(key)
    }

    /// The fields in the order they are written.
    fn ordered(&self) -> impl Iterator<Item = (&str, &RawValue)> {
        let leading = self
            .order
            .keys()
            .map(String::as_str)
            .filter(|key| !LAST_FIELDS.contains(key));
        leading
            .chain(LAST_FIELDS)
            .filter_map(|key| self.values.get(key).map(|value| (key, value.as_ref())))
    }

    /// Write the fields into `map`, in order.
    pub(crate) fn serialize_into<M: SerializeMap>(&self, map: &mut M) -> Result<(), M::Error> {
        for (key, value) in self.ordered() {
            map.serialize_entry(key, value)?;
        }
        Ok(())
    }

    /// The body as compact JSON bytes.
    pub(crate) fn to_vec(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

impl Serialize for RequestBody {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        self.serialize_into(&mut map)?;
        map.end()
    }
}

/// A JSON object's fields in the order its text states them, each value as
/// its own text.
struct Fields(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Fields {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldsVisitor;

        impl<'de> Visitor<'de> for FieldsVisitor {
            type Value = Fields;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a Responses request body, which is a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Fields, A::Error> {
                let mut fields = Vec::new();
                while let Some(entry) = access.next_entry::<String, Box<RawValue>>()? {
                    fields.push(entry);
                }
                Ok(Fields(fields))
            }
        }

        deserializer.deserialize_map(FieldsVisitor)
    }
}

#[cfg(test)]
mod tests;
