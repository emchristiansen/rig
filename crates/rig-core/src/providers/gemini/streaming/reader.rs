//! Reads the modeled fields of a GenerateContent document before emission.
//! Unknown fields stay in the decoder's original JSON map.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Reply {
    #[serde(default)]
    response_id: String,
    #[serde(default)]
    candidates: Vec<Candidate>,
    prompt_feedback: Option<PromptFeedback>,
    usage_metadata: Option<Usage>,
    model_version: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Option<Content>,
    finish_reason: Option<FinishReason>,
    safety_ratings: Option<Vec<SafetyRating>>,
    citation_metadata: Option<Citations>,
    token_count: Option<i32>,
    avg_logprobs: Option<f64>,
    logprobs_result: Option<Logprobs>,
    index: Option<i32>,
    finish_message: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Content {
    #[serde(default)]
    parts: Vec<Value>,
    role: Option<Role>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    User,
    Model,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Part {
    thought: Option<bool>,
    thought_signature: Option<String>,
    #[serde(flatten)]
    fields: Map<String, Value>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct FileData {
    mime_type: Option<String>,
    file_uri: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct FunctionCall {
    name: String,
    #[serde(default)]
    args: Map<String, Value>,
    id: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct FunctionResponse {
    name: String,
    id: Option<String>,
    response: Option<Value>,
    parts: Option<Vec<FunctionResponsePart>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct FunctionResponsePart {
    inline_data: Option<FunctionResponseInlineData>,
    file_data: Option<FileData>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct FunctionResponseInlineData {
    mime_type: String,
    data: String,
    display_name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum HostedLanguage {
    LanguageUnspecified,
    Python,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct HostedCode {
    language: HostedLanguage,
    code: String,
}

#[derive(Deserialize)]
enum HostedOutcome {
    #[serde(rename = "OUTCOME_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "OUTCOME_OK")]
    Ok,
    #[serde(rename = "OUTCOME_FAILED")]
    Failed,
    #[serde(rename = "OUTCOME_DEADLINE_EXCEEDED")]
    DeadlineExceeded,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct HostedResult {
    outcome: HostedOutcome,
    output: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum FinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Other,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
    UnexpectedToolCall,
    MissingThoughtSignature,
    TooManyToolCalls,
    MalformedResponse,
    #[serde(untagged)]
    Unknown(String),
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent,
    #[serde(untagged)]
    Unknown(String),
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum HarmCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexually,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
    HarmCategoryCivicIntegrity,
    #[serde(untagged)]
    Unknown(String),
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum HarmProbability {
    HarmProbabilityUnspecified,
    Negligible,
    Low,
    Medium,
    High,
    #[serde(untagged)]
    Unknown(String),
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct SafetyRating {
    category: HarmCategory,
    probability: HarmProbability,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    block_reason: Option<BlockReason>,
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Usage {
    #[serde(default)]
    prompt_token_count: i32,
    cached_content_token_count: Option<i32>,
    candidates_token_count: Option<i32>,
    #[serde(default)]
    total_token_count: i32,
    thoughts_token_count: Option<i32>,
    prompt_tokens_details: Option<Vec<ModalityCount>>,
    cache_tokens_details: Option<Vec<ModalityCount>>,
    candidates_tokens_details: Option<Vec<ModalityCount>>,
    tool_use_prompt_token_count: Option<i32>,
    tool_use_prompt_tokens_details: Option<Vec<ModalityCount>>,
    traffic_type: Option<Traffic>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct ModalityCount {
    modality: Modality,
    #[serde(default)]
    token_count: i32,
}

// Unknown metadata labels are validated as strings; the decoder keeps their
// exact spelling in the original JSON rather than storing an unread copy.
fn unknown_metadata_label<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<(), D::Error> {
    String::deserialize(deserializer).map(|_| ())
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Modality {
    ModalityUnspecified,
    Text,
    Image,
    Video,
    Audio,
    Document,
    #[serde(untagged)]
    Unknown(#[serde(deserialize_with = "unknown_metadata_label")] ()),
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Traffic {
    TrafficTypeUnspecified,
    OnDemand,
    ProvisionedThroughput,
    #[serde(untagged)]
    Unknown(#[serde(deserialize_with = "unknown_metadata_label")] ()),
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Citations {
    #[serde(default)]
    citation_sources: Vec<Citation>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Citation {
    uri: Option<String>,
    start_index: Option<i32>,
    end_index: Option<i32>,
    license: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Logprobs {
    #[serde(default)]
    top_candidates: Vec<TopCandidate>,
    log_probability_sum: Option<f64>,
    #[serde(default)]
    chosen_candidates: Vec<Logprob>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct TopCandidate {
    #[serde(default)]
    candidates: Vec<Logprob>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct Logprob {
    token: Option<String>,
    token_id: Option<i32>,
    log_probability: Option<f64>,
}

fn field<T: serde::de::DeserializeOwned>(
    fields: &Map<String, Value>,
    name: &str,
) -> Result<(), serde_json::Error> {
    if let Some(value) = fields.get(name) {
        serde_json::from_value::<T>(value.clone())?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedKind {
    Text,
    InlineData,
    FunctionCall,
    FunctionResponse,
    FileData,
    ExecutableCode,
    CodeExecutionResult,
}

pub(super) fn select(fields: &Map<String, Value>) -> Option<SelectedKind> {
    fields.keys().find_map(|key| match key.as_str() {
        "text" => Some(SelectedKind::Text),
        "inlineData" => Some(SelectedKind::InlineData),
        "functionCall" => Some(SelectedKind::FunctionCall),
        "functionResponse" => Some(SelectedKind::FunctionResponse),
        "fileData" => Some(SelectedKind::FileData),
        "executableCode" => Some(SelectedKind::ExecutableCode),
        "codeExecutionResult" => Some(SelectedKind::CodeExecutionResult),
        _ => None,
    })
}

pub(super) fn read(fields: &Map<String, Value>) -> Result<(), serde_json::Error> {
    let reply = serde_json::from_value::<Reply>(Value::Object(fields.clone()))?;
    for candidate in reply.candidates {
        if let Some(content) = candidate.content {
            for raw in content.parts {
                let Some(fields) = raw.as_object() else {
                    continue;
                };
                let kind = select(fields);
                let part = serde_json::from_value::<Part>(raw)?;
                let Some(kind) = kind else {
                    continue;
                };
                // PartKind claims the first recognized key. Other keys stay
                // inspection data, including unselected modeled siblings.
                match kind {
                    SelectedKind::Text => field::<String>(&part.fields, "text")?,
                    SelectedKind::InlineData => field::<InlineData>(&part.fields, "inlineData")?,
                    SelectedKind::FunctionCall => {
                        field::<FunctionCall>(&part.fields, "functionCall")?
                    }
                    SelectedKind::FunctionResponse => {
                        field::<FunctionResponse>(&part.fields, "functionResponse")?
                    }
                    SelectedKind::FileData => field::<FileData>(&part.fields, "fileData")?,
                    SelectedKind::ExecutableCode => {
                        field::<HostedCode>(&part.fields, "executableCode")?
                    }
                    SelectedKind::CodeExecutionResult => {
                        field::<HostedResult>(&part.fields, "codeExecutionResult")?
                    }
                }
            }
        }
    }
    Ok(())
}

fn spelling<T: serde::de::DeserializeOwned + Serialize>(value: &Value) -> Option<String> {
    let value = serde_json::from_value::<T>(value.clone()).ok()?;
    match serde_json::to_value(value).ok()? {
        Value::String(name) => Some(name),
        _ => None,
    }
}

pub(in crate::providers::gemini) fn finish_reason(value: &Value) -> Option<String> {
    match value {
        Value::Number(number) => Some(format!("FINISH_REASON_{number}")),
        _ => spelling::<FinishReason>(value),
    }
}

pub(in crate::providers::gemini) fn block_reason(value: &Value) -> Option<String> {
    match value {
        Value::Number(number) => Some(format!("BLOCK_REASON_{number}")),
        _ => spelling::<BlockReason>(value),
    }
}

pub(in crate::providers::gemini) fn harm_category(value: &Value) -> Option<String> {
    spelling::<HarmCategory>(value)
}

pub(in crate::providers::gemini) fn harm_probability(value: &Value) -> Option<String> {
    spelling::<HarmProbability>(value)
}

#[cfg(test)]
mod tests;
