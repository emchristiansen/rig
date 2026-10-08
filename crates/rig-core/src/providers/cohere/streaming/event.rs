//! Validation of modeled native Cohere event fields.

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
pub(super) struct Start {
    #[serde(rename = "type")]
    _type: String,
    #[serde(rename = "id")]
    _id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct Event {
    #[serde(rename = "type")]
    _type: String,
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "index")]
    _index: Option<usize>,
    #[serde(rename = "delta")]
    _delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(rename = "message")]
    _message: Option<Message>,
    #[serde(rename = "usage")]
    _usage: Option<Usage>,
    #[serde(rename = "finish_reason")]
    _finish_reason: Option<String>,
    #[serde(rename = "error")]
    _error: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(rename = "content")]
    _content: Option<Content>,
    #[serde(rename = "tool_calls")]
    _tool_calls: Option<Call>,
    #[serde(rename = "tool_plan")]
    _tool_plan: Option<String>,
}

#[derive(Deserialize)]
struct Content {
    #[serde(rename = "text")]
    _text: Option<String>,
    #[serde(rename = "thinking")]
    _thinking: Option<String>,
}

#[derive(Deserialize)]
struct Call {
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "function")]
    _function: Option<Function>,
}

#[derive(Deserialize)]
struct Function {
    #[serde(rename = "name")]
    _name: Option<String>,
    #[serde(rename = "arguments")]
    _arguments: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(rename = "tokens")]
    _tokens: Option<Tokens>,
    #[serde(rename = "billed_units")]
    _billed_units: Option<Billed>,
    #[serde(rename = "cached_tokens")]
    _cached_tokens: Option<f64>,
}

#[derive(Deserialize)]
struct Tokens {
    #[serde(rename = "input_tokens")]
    _input_tokens: Option<f64>,
    #[serde(rename = "output_tokens")]
    _output_tokens: Option<f64>,
}

#[derive(Deserialize)]
struct Billed {
    #[serde(flatten)]
    _tokens: Tokens,
    #[serde(rename = "search_units")]
    _search_units: Option<f64>,
    #[serde(rename = "classifications")]
    _classifications: Option<f64>,
}

#[derive(Deserialize)]
pub(super) struct Reply {
    #[serde(rename = "id")]
    _id: String,
    #[serde(rename = "finish_reason")]
    _finish_reason: String,
    #[serde(rename = "message")]
    _message: ReplyMessage,
    #[serde(rename = "usage")]
    _usage: Option<Usage>,
}

#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum ReplyMessage {
    Assistant {
        #[serde(rename = "content", default)]
        _content: Vec<ReplyContent>,
        #[serde(rename = "tool_plan")]
        _tool_plan: Option<String>,
        #[serde(rename = "tool_calls", default)]
        _tool_calls: Vec<ReplyCall>,
        #[serde(rename = "citations", default)]
        _citations: Vec<Value>,
    },
    User {
        #[serde(rename = "content")]
        _content: Vec<ReplyContent>,
    },
    Tool {
        #[serde(rename = "content")]
        _content: Vec<ReplyContent>,
        #[serde(rename = "tool_call_id")]
        _tool_call_id: String,
    },
    System {
        #[serde(rename = "content")]
        _content: String,
    },
}

struct ReplyContent;

impl<'de> Deserialize<'de> for ReplyContent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("Cohere content needs a string type"))?;
        let field = match kind {
            "text" => Some("text"),
            "thinking" => Some("thinking"),
            _ => None,
        };
        if let Some(field) = field {
            if !value.get(field).is_some_and(Value::is_string) {
                return Err(serde::de::Error::custom(format!(
                    "Cohere {kind} content needs string {field}"
                )));
            }
        }
        Ok(Self)
    }
}

#[derive(Deserialize)]
struct ReplyCall {
    #[serde(rename = "type")]
    _type: Option<FunctionType>,
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "function")]
    _function: Option<ReplyFunction>,
}

#[derive(Deserialize)]
struct ReplyFunction {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "arguments", with = "crate::json_utils::stringified_json")]
    _arguments: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum FunctionType {
    Function,
}
