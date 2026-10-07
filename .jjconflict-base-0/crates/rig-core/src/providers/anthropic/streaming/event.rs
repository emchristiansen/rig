//! Typed validation of known Messages event payloads before decoding.

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Event {
    Message {
        #[serde(flatten)]
        _message: Message,
    },
    MessageStart {
        #[serde(default, rename = "message")]
        _message: Option<Message>,
    },
    ContentBlockStart {
        #[serde(rename = "index")]
        _index: usize,
        #[serde(rename = "content_block")]
        _content_block: Content,
    },
    ContentBlockDelta {
        #[serde(rename = "index")]
        _index: usize,
        #[serde(rename = "delta")]
        _delta: Delta,
    },
    ContentBlockStop {
        #[serde(rename = "index")]
        _index: usize,
    },
    MessageDelta {
        #[serde(rename = "delta")]
        _delta: Terminal,
        #[serde(rename = "usage")]
        _usage: TerminalUsage,
    },
    MessageStop,
    Ping,
    Error {
        #[serde(rename = "error")]
        _error: Value,
    },
}

#[derive(Deserialize)]
pub(super) struct Message {
    #[serde(rename = "id")]
    _id: String,
    #[serde(rename = "model")]
    _model: String,
    #[serde(rename = "role")]
    _role: String,
    #[serde(rename = "content")]
    _content: Vec<Content>,
    #[serde(rename = "stop_reason")]
    _stop_reason: Option<String>,
    #[serde(rename = "stop_sequence")]
    _stop_sequence: Option<String>,
    #[serde(rename = "usage")]
    _usage: Usage,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(rename = "input_tokens")]
    _input_tokens: u64,
    #[serde(flatten)]
    _terminal: TerminalUsage,
}

#[derive(Deserialize)]
struct TerminalUsage {
    #[serde(rename = "output_tokens")]
    _output_tokens: u64,
    #[serde(rename = "input_tokens")]
    _input_tokens: Option<u64>,
    #[serde(rename = "cache_read_input_tokens")]
    _cache_read_input_tokens: Option<u64>,
    #[serde(rename = "cache_creation_input_tokens")]
    _cache_creation_input_tokens: Option<u64>,
    #[serde(rename = "cache_creation")]
    _cache_creation: Option<CacheCreation>,
    #[serde(rename = "output_tokens_details")]
    _output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Deserialize)]
struct CacheCreation {
    #[serde(default, rename = "ephemeral_5m_input_tokens")]
    _ephemeral_5m_input_tokens: u64,
    #[serde(default, rename = "ephemeral_1h_input_tokens")]
    _ephemeral_1h_input_tokens: u64,
}

#[derive(Deserialize)]
struct OutputTokensDetails {
    #[serde(default, rename = "thinking_tokens")]
    _thinking_tokens: u64,
}

#[derive(Deserialize)]
struct Terminal {
    #[serde(rename = "stop_reason")]
    _stop_reason: Option<String>,
    #[serde(rename = "stop_sequence")]
    _stop_sequence: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Content {
    Text {
        #[serde(rename = "text")]
        _text: String,
        #[serde(rename = "citations")]
        _citations: Option<Vec<Citation>>,
        #[serde(rename = "cache_control")]
        _cache_control: Option<CacheControl>,
    },
    Image {
        #[serde(rename = "source")]
        _source: ImageSource,
        #[serde(rename = "cache_control")]
        _cache_control: Option<CacheControl>,
    },
    Document {
        #[serde(rename = "source")]
        _source: DocumentSource,
        #[serde(rename = "title")]
        _title: Option<String>,
        #[serde(rename = "context")]
        _context: Option<String>,
        #[serde(rename = "citations")]
        _citations: Option<CitationsConfig>,
        #[serde(rename = "cache_control")]
        _cache_control: Option<CacheControl>,
    },
    Thinking {
        #[serde(rename = "thinking")]
        _thinking: String,
        #[serde(rename = "signature")]
        _signature: Option<String>,
    },
    RedactedThinking {
        #[serde(rename = "data")]
        _data: String,
    },
    ToolUse {
        #[serde(rename = "id")]
        _id: String,
        #[serde(rename = "name")]
        _name: String,
        #[serde(rename = "input")]
        _input: Value,
    },
    ServerToolUse {
        #[serde(rename = "id")]
        _id: String,
        #[serde(rename = "name")]
        _name: String,
        #[serde(default, rename = "input")]
        _input: Value,
    },
    WebSearchToolResult {
        #[serde(rename = "tool_use_id")]
        _tool_use_id: String,
        #[serde(rename = "content")]
        _content: Value,
    },
    CodeExecutionToolResult {
        #[serde(rename = "tool_use_id")]
        _tool_use_id: String,
        #[serde(rename = "content")]
        _content: Value,
    },
    ToolResult {
        #[serde(rename = "tool_use_id")]
        _tool_use_id: String,
        #[serde(rename = "content")]
        _content: ToolResultBody,
        #[serde(rename = "is_error")]
        _is_error: Option<bool>,
        #[serde(rename = "cache_control")]
        _cache_control: Option<CacheControl>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct CacheControl {
    #[serde(rename = "type")]
    _kind: String,
    #[serde(rename = "ttl")]
    _ttl: Option<super::super::completion::CacheTtl>,
}

#[derive(Deserialize)]
struct CitationsConfig {
    #[serde(rename = "enabled")]
    _enabled: bool,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ImageSource {
    Base64 {
        #[serde(rename = "data")]
        _data: String,
        #[serde(rename = "media_type")]
        _media_type: ImageMediaType,
    },
    Url {
        #[serde(rename = "url")]
        _url: String,
    },
    File {
        #[serde(rename = "file_id")]
        _file_id: String,
    },
}

#[derive(Deserialize)]
enum ImageMediaType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DocumentSource {
    Base64 {
        #[serde(rename = "data")]
        _data: String,
        #[serde(rename = "media_type")]
        _media_type: Pdf,
    },
    Text {
        #[serde(rename = "data")]
        _data: String,
        #[serde(rename = "media_type")]
        _media_type: PlainText,
    },
    Url {
        #[serde(rename = "url")]
        _url: String,
    },
    File {
        #[serde(rename = "file_id")]
        _file_id: String,
    },
}

#[derive(Deserialize)]
enum Pdf {
    #[serde(rename = "application/pdf")]
    Pdf,
}

#[derive(Deserialize)]
enum PlainText {
    #[serde(rename = "text/plain")]
    Plain,
}

#[derive(Deserialize)]
#[serde(untagged)]
#[expect(
    dead_code,
    reason = "Validated payloads stay in MessagesEvent's original fields."
)]
enum ToolResultBody {
    Text(String),
    Parts(Vec<ToolResultPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolResultPart {
    Text {
        #[serde(rename = "text")]
        _text: String,
    },
    Image {
        #[serde(rename = "source")]
        _source: ImageSource,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Delta {
    TextDelta {
        #[serde(rename = "text")]
        _text: String,
    },
    ThinkingDelta {
        #[serde(rename = "thinking")]
        _thinking: String,
    },
    SignatureDelta {
        #[serde(rename = "signature")]
        _signature: String,
    },
    InputJsonDelta {
        #[serde(rename = "partial_json")]
        _partial_json: String,
    },
    CitationsDelta {
        #[serde(rename = "citation")]
        _citation: Citation,
    },
    #[serde(other)]
    Other,
}

struct Citation;

impl<'de> Deserialize<'de> for Citation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.get("type").is_some_and(Value::is_string) {
            serde_json::from_value::<CitationPayload>(value).map_err(serde::de::Error::custom)?;
        }
        Ok(Self)
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CitationPayload {
    CharLocation {
        #[serde(rename = "cited_text")]
        _cited_text: String,
        #[serde(rename = "document_index")]
        _document_index: usize,
        #[serde(rename = "document_title")]
        _document_title: Option<String>,
        #[serde(rename = "start_char_index")]
        _start_char_index: usize,
        #[serde(rename = "end_char_index")]
        _end_char_index: usize,
    },
    PageLocation {
        #[serde(rename = "cited_text")]
        _cited_text: String,
        #[serde(rename = "document_index")]
        _document_index: usize,
        #[serde(rename = "document_title")]
        _document_title: Option<String>,
        #[serde(rename = "start_page_number")]
        _start_page_number: u32,
        #[serde(rename = "end_page_number")]
        _end_page_number: u32,
    },
    ContentBlockLocation {
        #[serde(rename = "cited_text")]
        _cited_text: String,
        #[serde(rename = "document_index")]
        _document_index: usize,
        #[serde(rename = "document_title")]
        _document_title: Option<String>,
        #[serde(rename = "start_block_index")]
        _start_block_index: usize,
        #[serde(rename = "end_block_index")]
        _end_block_index: usize,
    },
    SearchResultLocation {
        #[serde(rename = "cited_text")]
        _cited_text: String,
        #[serde(rename = "source")]
        _source: String,
        #[serde(rename = "title")]
        _title: Option<String>,
        #[serde(rename = "search_result_index")]
        _search_result_index: usize,
        #[serde(rename = "start_block_index")]
        _start_block_index: usize,
        #[serde(rename = "end_block_index")]
        _end_block_index: usize,
    },
    WebSearchResultLocation {
        #[serde(rename = "cited_text")]
        _cited_text: String,
        #[serde(rename = "url")]
        _url: String,
        #[serde(rename = "title")]
        _title: Option<String>,
        #[serde(rename = "encrypted_index")]
        _encrypted_index: String,
    },
    #[serde(other)]
    Other,
}
