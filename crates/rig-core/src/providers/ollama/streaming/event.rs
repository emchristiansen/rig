//! Validation of the modeled native Ollama record fields.

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
pub(super) struct Record {
    #[serde(rename = "model")]
    _model: String,
    #[serde(rename = "created_at")]
    _created_at: String,
    #[serde(rename = "message")]
    _message: Message,
    #[serde(rename = "done")]
    _done: bool,
    #[serde(rename = "done_reason")]
    _done_reason: Option<String>,
    #[serde(rename = "total_duration")]
    _total_duration: Option<u64>,
    #[serde(rename = "load_duration")]
    _load_duration: Option<u64>,
    #[serde(rename = "prompt_eval_count")]
    _prompt_eval_count: Option<u64>,
    #[serde(rename = "prompt_eval_duration")]
    _prompt_eval_duration: Option<u64>,
    #[serde(rename = "eval_count")]
    _eval_count: Option<u64>,
    #[serde(rename = "eval_duration")]
    _eval_duration: Option<u64>,
}

#[derive(Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum Message {
    Assistant {
        #[serde(rename = "content", default)]
        _content: String,
        #[serde(rename = "thinking")]
        _thinking: Option<String>,
        #[serde(
            rename = "tool_calls",
            default,
            deserialize_with = "crate::json_utils::null_or_default"
        )]
        _tool_calls: Vec<Call>,
        #[serde(flatten)]
        _extras: Extras,
    },
    User {
        #[serde(rename = "content")]
        _content: String,
        #[serde(flatten)]
        _extras: Extras,
    },
    System {
        #[serde(rename = "content")]
        _content: String,
        #[serde(flatten)]
        _extras: Extras,
    },
    Tool {
        #[serde(rename = "content")]
        _content: String,
        #[serde(rename = "tool_name")]
        _name: String,
        #[serde(rename = "tool_call_id")]
        _id: Option<String>,
    },
}

#[derive(Deserialize)]
struct Extras {
    #[serde(rename = "images")]
    _images: Option<Vec<String>>,
    #[serde(rename = "name")]
    _name: Option<String>,
}

#[derive(Deserialize)]
struct Call {
    #[serde(rename = "id")]
    _id: Option<String>,
    #[serde(rename = "type")]
    _type: Option<FunctionType>,
    #[serde(rename = "function")]
    _function: Function,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum FunctionType {
    Function,
}

#[derive(Deserialize)]
struct Function {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "arguments")]
    _arguments: Value,
}
