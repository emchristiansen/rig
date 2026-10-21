//! Projects and reads the SDK response JSON used for gRPC captures.
//! Fields use snake_case, bytes use integer arrays, and oneofs retain their
//! externally tagged names. Code language and outcome project numeric enums;
//! their reader also accepts known names from older captures.

use crate::proto;
use rig_core::serde::de::{DeserializeOwned, Error as _};
use serde_json::{Map, Value, json};

type Result<T> = std::result::Result<T, serde_json::Error>;

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or_else(|| serde_json::Error::custom("expected an SDK message object"))
}

fn field<T: DeserializeOwned>(fields: &Map<String, Value>, name: &str) -> Result<T> {
    let value = fields.get(name).ok_or_else(|| serde_json::Error::custom(format!("missing SDK field `{name}`")))?;
    serde_json::from_value(value.clone())
}

// Optional fields and fields added by the current SDK may be absent in captures.
fn added<T: DeserializeOwned + Default>(fields: &Map<String, Value>, name: &str) -> Result<T> {
    fields.get(name).map(|value| serde_json::from_value(value.clone())).transpose().map(Option::unwrap_or_default)
}

fn optional<T>(fields: &Map<String, Value>, name: &str, read: fn(&Value) -> Result<T>) -> Result<Option<T>> {
    match fields.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => read(value).map(Some),
    }
}

fn repeated<T>(fields: &Map<String, Value>, name: &str, read: fn(&Value) -> Result<T>, newer: bool) -> Result<Vec<T>> {
    let Some(value) = fields.get(name) else {
        return if newer { Ok(Vec::new()) } else { Err(serde_json::Error::custom(format!("missing SDK field `{name}`"))) };
    };
    value.as_array().ok_or_else(|| serde_json::Error::custom(format!("SDK field `{name}` must be an array")))?
        .iter().map(read).collect()
}

fn tag(value: &Value) -> Result<(&str, &Value)> {
    let fields = object(value)?;
    if fields.len() != 1 { return Err(serde_json::Error::custom("SDK oneof must contain exactly one variant")); }
    fields.iter().next().map(|(name, value)| (name.as_str(), value))
        .ok_or_else(|| serde_json::Error::custom("SDK oneof has no variant"))
}

fn enum_number(value: &Value, known: fn(&str) -> Option<i32>, name: &str) -> Result<i32> {
    if let Some(value) = value.as_str() {
        known(value).ok_or_else(|| serde_json::Error::custom(format!("unknown SDK {name} `{value}`")))
    } else { serde_json::from_value(value.clone()) }
}

fn struct_value(value: &prost_types::Struct) -> Value {
    json!({"fields": value.fields.iter().map(|(name, value)| (name.clone(), protobuf_value(value))).collect::<Map<_, _>>()})
}

fn read_struct(value: &Value) -> Result<prost_types::Struct> {
    let fields = object(value)?;
    let values = fields.get("fields").ok_or_else(|| serde_json::Error::custom("missing SDK field `fields`"))?;
    Ok(prost_types::Struct { fields: object(values)?.iter().map(|(name, value)| Ok((name.clone(), read_value(value)?))).collect::<Result<_>>()? })
}

fn protobuf_value(value: &prost_types::Value) -> Value {
    use prost_types::value::Kind;
    json!({"kind": value.kind.as_ref().map(|kind| match kind {
        Kind::NullValue(value) => json!({"NullValue": value}),
        Kind::NumberValue(value) => json!({"NumberValue": value}),
        Kind::StringValue(value) => json!({"StringValue": value}),
        Kind::BoolValue(value) => json!({"BoolValue": value}),
        Kind::StructValue(value) => json!({"StructValue": struct_value(value)}),
        Kind::ListValue(value) => json!({"ListValue": {"values": value.values.iter().map(protobuf_value).collect::<Vec<_>>()}}),
    })})
}

fn read_value(value: &Value) -> Result<prost_types::Value> {
    use prost_types::value::Kind;
    fn read_kind(value: &Value) -> Result<Kind> {
        let (name, value) = tag(value)?;
        Ok(match name {
            "NullValue" => Kind::NullValue(serde_json::from_value(value.clone())?),
            "NumberValue" => Kind::NumberValue(serde_json::from_value(value.clone())?),
            "StringValue" => Kind::StringValue(serde_json::from_value(value.clone())?),
            "BoolValue" => Kind::BoolValue(serde_json::from_value(value.clone())?),
            "StructValue" => Kind::StructValue(read_struct(value)?),
            "ListValue" => Kind::ListValue(prost_types::ListValue { values: repeated(object(value)?, "values", read_value, false)? }),
            _ => return Err(serde_json::Error::custom(format!("unknown SDK Value.Kind `{name}`"))),
        })
    }
    Ok(prost_types::Value { kind: optional(object(value)?, "kind", read_kind)? })
}

pub(crate) fn part(value: &proto::Part) -> Value {
    use proto::part::{Data, Metadata};
    json!({
        "thought": value.thought, "thought_signature": value.thought_signature,
        "part_metadata": value.part_metadata.as_ref().map(struct_value),
        "data": value.data.as_ref().map(|data| match data {
            Data::Text(value) => json!({"Text": value}),
            Data::InlineData(value) => json!({"InlineData": blob(value)}),
            Data::FunctionCall(value) => json!({"FunctionCall": function_call(value)}),
            Data::FunctionResponse(value) => json!({"FunctionResponse": function_response(value)}),
            Data::FileData(value) => json!({"FileData": file_data(value)}),
            Data::ExecutableCode(value) => json!({"ExecutableCode": executable_code(value)}),
            Data::CodeExecutionResult(value) => json!({"CodeExecutionResult": code_result(value)}),
        }),
        "metadata": value.metadata.as_ref().map(|metadata| match metadata {
            Metadata::VideoMetadata(value) => json!({"VideoMetadata": video_metadata(value)}),
        }),
    })
}

fn read_part(value: &Value) -> Result<proto::Part> {
    fn data(value: &Value) -> Result<proto::part::Data> {
        use proto::part::Data;
        let (name, value) = tag(value)?;
        Ok(match name {
            "Text" => Data::Text(serde_json::from_value(value.clone())?),
            "InlineData" => Data::InlineData(read_blob(value)?),
            "FunctionCall" => Data::FunctionCall(read_function_call(value)?),
            "FunctionResponse" => Data::FunctionResponse(read_function_response(value)?),
            "FileData" => Data::FileData(read_file_data(value)?),
            "ExecutableCode" => Data::ExecutableCode(read_executable_code(value)?),
            "CodeExecutionResult" => Data::CodeExecutionResult(read_code_result(value)?),
            _ => return Err(serde_json::Error::custom(format!("unknown SDK Part.Data `{name}`"))),
        })
    }
    fn metadata(value: &Value) -> Result<proto::part::Metadata> {
        let (name, value) = tag(value)?;
        match name {
            "VideoMetadata" => Ok(proto::part::Metadata::VideoMetadata(read_video_metadata(value)?)),
            _ => Err(serde_json::Error::custom(format!("unknown SDK Part.Metadata `{name}`"))),
        }
    }
    let fields = object(value)?;
    Ok(proto::Part {
        data: optional(fields, "data", data)?, metadata: optional(fields, "metadata", metadata)?,
        thought: field(fields, "thought")?, thought_signature: field(fields, "thought_signature")?,
        part_metadata: optional(fields, "part_metadata", read_struct)?,
    })
}

fn executable_code(value: &proto::ExecutableCode) -> Value {
    json!({"language": value.language, "code": value.code})
}

fn read_executable_code(value: &Value) -> Result<proto::ExecutableCode> {
    let fields = object(value)?;
    let language = fields.get("language").ok_or_else(|| serde_json::Error::custom("missing SDK field `language`"))?;
    Ok(proto::ExecutableCode {
        language: enum_number(language, |name| proto::executable_code::Language::from_str_name(name).map(|value| value as i32), "code language")?,
        code: field(fields, "code")?,
    })
}

fn code_result(value: &proto::CodeExecutionResult) -> Value {
    json!({"outcome": value.outcome, "output": value.output})
}

fn read_code_result(value: &Value) -> Result<proto::CodeExecutionResult> {
    let fields = object(value)?;
    let outcome = fields.get("outcome").ok_or_else(|| serde_json::Error::custom("missing SDK field `outcome`"))?;
    Ok(proto::CodeExecutionResult {
        outcome: enum_number(outcome, |name| proto::code_execution_result::Outcome::from_str_name(name).map(|value| value as i32), "code outcome")?,
        output: field(fields, "output")?,
    })
}

fn response_part(value: &proto::FunctionResponsePart) -> Value {
    json!({"data": value.data.as_ref().map(|data| match data {
        proto::function_response_part::Data::InlineData(value) => json!({"InlineData": response_blob(value)}),
    })})
}

fn read_response_part(value: &Value) -> Result<proto::FunctionResponsePart> {
    fn data(value: &Value) -> Result<proto::function_response_part::Data> {
        let (name, value) = tag(value)?;
        match name {
            "InlineData" => Ok(proto::function_response_part::Data::InlineData(read_response_blob(value)?)),
            _ => Err(serde_json::Error::custom(format!("unknown SDK FunctionResponsePart.Data `{name}`"))),
        }
    }
    Ok(proto::FunctionResponsePart { data: optional(object(value)?, "data", data)? })
}

fn grounding_chunk(value: &proto::GroundingChunk) -> Value {
    use proto::grounding_chunk::ChunkType;
    json!({"chunk_type": value.chunk_type.as_ref().map(|chunk| match chunk {
        ChunkType::Web(value) => json!({"Web": web(value)}),
        ChunkType::RetrievedContext(value) => json!({"RetrievedContext": retrieved(value)}),
        ChunkType::Maps(value) => json!({"Maps": maps(value)}),
    })})
}

fn read_grounding_chunk(value: &Value) -> Result<proto::GroundingChunk> {
    fn chunk(value: &Value) -> Result<proto::grounding_chunk::ChunkType> {
        use proto::grounding_chunk::ChunkType;
        let (name, value) = tag(value)?;
        Ok(match name {
            "Web" => ChunkType::Web(read_web(value)?),
            "RetrievedContext" => ChunkType::RetrievedContext(read_retrieved(value)?),
            "Maps" => ChunkType::Maps(read_maps(value)?),
            _ => return Err(serde_json::Error::custom(format!("unknown SDK GroundingChunk.ChunkType `{name}`"))),
        })
    }
    Ok(proto::GroundingChunk { chunk_type: optional(object(value)?, "chunk_type", chunk)? })
}

pub(crate) fn response(value: &proto::GenerateContentResponse) -> Value {
    json!({
        "candidates": value.candidates.iter().map(candidate).collect::<Vec<_>>(),
        "prompt_feedback": value.prompt_feedback.as_ref().map(feedback),
        "usage_metadata": value.usage_metadata.as_ref().map(usage),
        "model_version": value.model_version,
        "response_id": value.response_id,
    })
}

pub(crate) fn read_response(value: &Value) -> Result<proto::GenerateContentResponse> {
    let fields = object(value)?;
    Ok(proto::GenerateContentResponse {
        candidates: repeated(fields, "candidates", read_candidate, false)?,
        prompt_feedback: optional(fields, "prompt_feedback", read_feedback)?,
        usage_metadata: optional(fields, "usage_metadata", read_usage)?,
        model_version: field(fields, "model_version")?,
        response_id: field(fields, "response_id")?,
    })
}

fn candidate(value: &proto::Candidate) -> Value {
    json!({
        "content": value.content.as_ref().map(content),
        "finish_reason": value.finish_reason,
        "index": value.index,
        "finish_message": value.finish_message,
        "safety_ratings": value.safety_ratings.iter().map(safety).collect::<Vec<_>>(),
        "citation_metadata": value.citation_metadata.as_ref().map(citations),
        "token_count": value.token_count,
        "grounding_metadata": value.grounding_metadata.as_ref().map(grounding),
        "avg_logprobs": value.avg_logprobs,
        "logprobs_result": value.logprobs_result.as_ref().map(logprobs),
    })
}

fn read_candidate(value: &Value) -> Result<proto::Candidate> {
    let fields = object(value)?;
    Ok(proto::Candidate {
        content: optional(fields, "content", read_content)?,
        finish_reason: field(fields, "finish_reason")?,
        index: added(fields, "index")?,
        finish_message: added(fields, "finish_message")?,
        safety_ratings: repeated(fields, "safety_ratings", read_safety, true)?,
        citation_metadata: optional(fields, "citation_metadata", read_citations)?,
        token_count: added(fields, "token_count")?,
        grounding_metadata: optional(fields, "grounding_metadata", read_grounding)?,
        avg_logprobs: added(fields, "avg_logprobs")?,
        logprobs_result: optional(fields, "logprobs_result", read_logprobs)?,
    })
}

fn content(value: &proto::Content) -> Value {
    json!({
        "parts": value.parts.iter().map(part).collect::<Vec<_>>(),
        "role": value.role,
    })
}

fn read_content(value: &Value) -> Result<proto::Content> {
    let fields = object(value)?;
    Ok(proto::Content {
        parts: repeated(fields, "parts", read_part, false)?,
        role: field(fields, "role")?,
    })
}

fn blob(value: &proto::Blob) -> Value {
    json!({
        "mime_type": value.mime_type,
        "data": value.data,
    })
}

fn read_blob(value: &Value) -> Result<proto::Blob> {
    let fields = object(value)?;
    Ok(proto::Blob {
        mime_type: field(fields, "mime_type")?,
        data: field(fields, "data")?,
    })
}

fn response_blob(value: &proto::FunctionResponseBlob) -> Value {
    json!({
        "mime_type": value.mime_type,
        "data": value.data,
    })
}

fn read_response_blob(value: &Value) -> Result<proto::FunctionResponseBlob> {
    let fields = object(value)?;
    Ok(proto::FunctionResponseBlob {
        mime_type: field(fields, "mime_type")?,
        data: field(fields, "data")?,
    })
}

fn function_call(value: &proto::FunctionCall) -> Value {
    json!({
        "name": value.name,
        "args": value.args.as_ref().map(struct_value),
        "id": value.id,
    })
}

fn read_function_call(value: &Value) -> Result<proto::FunctionCall> {
    let fields = object(value)?;
    Ok(proto::FunctionCall {
        name: field(fields, "name")?,
        args: optional(fields, "args", read_struct)?,
        id: field(fields, "id")?,
    })
}

fn function_response(value: &proto::FunctionResponse) -> Value {
    json!({
        "name": value.name,
        "response": value.response.as_ref().map(struct_value),
        "id": value.id,
        "will_continue": value.will_continue,
        "scheduling": value.scheduling,
        "parts": value.parts.iter().map(response_part).collect::<Vec<_>>(),
    })
}

fn read_function_response(value: &Value) -> Result<proto::FunctionResponse> {
    let fields = object(value)?;
    Ok(proto::FunctionResponse {
        name: field(fields, "name")?,
        response: optional(fields, "response", read_struct)?,
        id: field(fields, "id")?,
        will_continue: added(fields, "will_continue")?,
        scheduling: added(fields, "scheduling")?,
        parts: repeated(fields, "parts", read_response_part, true)?,
    })
}

fn file_data(value: &proto::FileData) -> Value {
    json!({
        "mime_type": value.mime_type,
        "file_uri": value.file_uri,
    })
}

fn read_file_data(value: &Value) -> Result<proto::FileData> {
    let fields = object(value)?;
    Ok(proto::FileData {
        mime_type: field(fields, "mime_type")?,
        file_uri: field(fields, "file_uri")?,
    })
}

fn video_metadata(value: &proto::VideoMetadata) -> Value {
    json!({
        "start_offset": value.start_offset.as_ref().map(duration),
        "end_offset": value.end_offset.as_ref().map(duration),
        "fps": value.fps,
    })
}

fn read_video_metadata(value: &Value) -> Result<proto::VideoMetadata> {
    let fields = object(value)?;
    Ok(proto::VideoMetadata {
        start_offset: optional(fields, "start_offset", read_duration)?,
        end_offset: optional(fields, "end_offset", read_duration)?,
        fps: field(fields, "fps")?,
    })
}

fn duration(value: &prost_types::Duration) -> Value {
    json!({
        "seconds": value.seconds,
        "nanos": value.nanos,
    })
}

fn read_duration(value: &Value) -> Result<prost_types::Duration> {
    let fields = object(value)?;
    Ok(prost_types::Duration {
        seconds: field(fields, "seconds")?,
        nanos: field(fields, "nanos")?,
    })
}

fn usage(value: &proto::UsageMetadata) -> Value {
    json!({
        "prompt_token_count": value.prompt_token_count,
        "candidates_token_count": value.candidates_token_count,
        "total_token_count": value.total_token_count,
        "cached_content_token_count": value.cached_content_token_count,
        "prompt_tokens_details": value.prompt_tokens_details.iter().map(modality).collect::<Vec<_>>(),
        "cache_tokens_details": value.cache_tokens_details.iter().map(modality).collect::<Vec<_>>(),
        "candidates_tokens_details": value.candidates_tokens_details.iter().map(modality).collect::<Vec<_>>(),
        "tool_use_prompt_token_count": value.tool_use_prompt_token_count,
        "tool_use_prompt_tokens_details": value.tool_use_prompt_tokens_details.iter().map(modality).collect::<Vec<_>>(),
        "thoughts_token_count": value.thoughts_token_count,
    })
}

fn read_usage(value: &Value) -> Result<proto::UsageMetadata> {
    let fields = object(value)?;
    Ok(proto::UsageMetadata {
        prompt_token_count: field(fields, "prompt_token_count")?,
        candidates_token_count: field(fields, "candidates_token_count")?,
        total_token_count: field(fields, "total_token_count")?,
        cached_content_token_count: field(fields, "cached_content_token_count")?,
        prompt_tokens_details: repeated(fields, "prompt_tokens_details", read_modality, true)?,
        cache_tokens_details: repeated(fields, "cache_tokens_details", read_modality, true)?,
        candidates_tokens_details: repeated(fields, "candidates_tokens_details", read_modality, true)?,
        tool_use_prompt_token_count: added(fields, "tool_use_prompt_token_count")?,
        tool_use_prompt_tokens_details: repeated(fields, "tool_use_prompt_tokens_details", read_modality, true)?,
        thoughts_token_count: added(fields, "thoughts_token_count")?,
    })
}

fn modality(value: &proto::ModalityTokenCount) -> Value {
    json!({
        "modality": value.modality,
        "token_count": value.token_count,
    })
}

fn read_modality(value: &Value) -> Result<proto::ModalityTokenCount> {
    let fields = object(value)?;
    Ok(proto::ModalityTokenCount {
        modality: field(fields, "modality")?,
        token_count: field(fields, "token_count")?,
    })
}

fn feedback(value: &proto::PromptFeedback) -> Value {
    json!({
        "block_reason": value.block_reason,
        "safety_ratings": value.safety_ratings.iter().map(safety).collect::<Vec<_>>(),
    })
}

fn read_feedback(value: &Value) -> Result<proto::PromptFeedback> {
    let fields = object(value)?;
    Ok(proto::PromptFeedback {
        block_reason: added(fields, "block_reason")?,
        safety_ratings: repeated(fields, "safety_ratings", read_safety, true)?,
    })
}

fn safety(value: &proto::SafetyRating) -> Value {
    json!({
        "category": value.category,
        "probability": value.probability,
        "blocked": value.blocked,
    })
}

fn read_safety(value: &Value) -> Result<proto::SafetyRating> {
    let fields = object(value)?;
    Ok(proto::SafetyRating {
        category: field(fields, "category")?,
        probability: field(fields, "probability")?,
        blocked: field(fields, "blocked")?,
    })
}

fn citations(value: &proto::CitationMetadata) -> Value {
    json!({
        "citation_sources": value.citation_sources.iter().map(citation).collect::<Vec<_>>(),
    })
}

fn read_citations(value: &Value) -> Result<proto::CitationMetadata> {
    let fields = object(value)?;
    Ok(proto::CitationMetadata {
        citation_sources: repeated(fields, "citation_sources", read_citation, false)?,
    })
}

fn citation(value: &proto::CitationSource) -> Value {
    json!({
        "start_index": value.start_index,
        "end_index": value.end_index,
        "uri": value.uri,
        "license": value.license,
    })
}

fn read_citation(value: &Value) -> Result<proto::CitationSource> {
    let fields = object(value)?;
    Ok(proto::CitationSource {
        start_index: added(fields, "start_index")?,
        end_index: added(fields, "end_index")?,
        uri: added(fields, "uri")?,
        license: added(fields, "license")?,
    })
}

fn grounding(value: &proto::GroundingMetadata) -> Value {
    json!({
        "grounding_chunks": value.grounding_chunks.iter().map(grounding_chunk).collect::<Vec<_>>(),
        "grounding_supports": value.grounding_supports.iter().map(support).collect::<Vec<_>>(),
    })
}

fn read_grounding(value: &Value) -> Result<proto::GroundingMetadata> {
    let fields = object(value)?;
    Ok(proto::GroundingMetadata {
        grounding_chunks: repeated(fields, "grounding_chunks", read_grounding_chunk, false)?,
        grounding_supports: repeated(fields, "grounding_supports", read_support, false)?,
    })
}

fn web(value: &proto::grounding_chunk::Web) -> Value {
    json!({
        "uri": value.uri,
        "title": value.title,
    })
}

fn read_web(value: &Value) -> Result<proto::grounding_chunk::Web> {
    let fields = object(value)?;
    Ok(proto::grounding_chunk::Web {
        uri: added(fields, "uri")?,
        title: added(fields, "title")?,
    })
}

fn retrieved(value: &proto::grounding_chunk::RetrievedContext) -> Value {
    json!({
        "uri": value.uri,
        "title": value.title,
        "text": value.text,
    })
}

fn read_retrieved(value: &Value) -> Result<proto::grounding_chunk::RetrievedContext> {
    let fields = object(value)?;
    Ok(proto::grounding_chunk::RetrievedContext {
        uri: added(fields, "uri")?,
        title: added(fields, "title")?,
        text: added(fields, "text")?,
    })
}

fn maps(value: &proto::grounding_chunk::Maps) -> Value {
    json!({
        "uri": value.uri,
        "title": value.title,
        "text": value.text,
        "place_id": value.place_id,
    })
}

fn read_maps(value: &Value) -> Result<proto::grounding_chunk::Maps> {
    let fields = object(value)?;
    Ok(proto::grounding_chunk::Maps {
        uri: added(fields, "uri")?,
        title: added(fields, "title")?,
        text: added(fields, "text")?,
        place_id: added(fields, "place_id")?,
    })
}

fn support(value: &proto::GroundingSupport) -> Value {
    json!({
        "segment": value.segment.as_ref().map(segment),
        "grounding_chunk_indices": value.grounding_chunk_indices,
        "confidence_scores": value.confidence_scores,
    })
}

fn read_support(value: &Value) -> Result<proto::GroundingSupport> {
    let fields = object(value)?;
    Ok(proto::GroundingSupport {
        segment: optional(fields, "segment", read_segment)?,
        grounding_chunk_indices: field(fields, "grounding_chunk_indices")?,
        confidence_scores: field(fields, "confidence_scores")?,
    })
}

fn segment(value: &proto::Segment) -> Value {
    json!({
        "part_index": value.part_index,
        "start_index": value.start_index,
        "end_index": value.end_index,
        "text": value.text,
    })
}

fn read_segment(value: &Value) -> Result<proto::Segment> {
    let fields = object(value)?;
    Ok(proto::Segment {
        part_index: field(fields, "part_index")?,
        start_index: field(fields, "start_index")?,
        end_index: field(fields, "end_index")?,
        text: field(fields, "text")?,
    })
}

fn logprobs(value: &proto::LogprobsResult) -> Value {
    json!({
        "log_probability_sum": value.log_probability_sum,
        "top_candidates": value.top_candidates.iter().map(top_candidates).collect::<Vec<_>>(),
        "chosen_candidates": value.chosen_candidates.iter().map(logprob_candidate).collect::<Vec<_>>(),
    })
}

fn read_logprobs(value: &Value) -> Result<proto::LogprobsResult> {
    let fields = object(value)?;
    Ok(proto::LogprobsResult {
        log_probability_sum: added(fields, "log_probability_sum")?,
        top_candidates: repeated(fields, "top_candidates", read_top_candidates, false)?,
        chosen_candidates: repeated(fields, "chosen_candidates", read_logprob_candidate, false)?,
    })
}

fn top_candidates(value: &proto::logprobs_result::TopCandidates) -> Value {
    json!({
        "candidates": value.candidates.iter().map(logprob_candidate).collect::<Vec<_>>(),
    })
}

fn read_top_candidates(value: &Value) -> Result<proto::logprobs_result::TopCandidates> {
    let fields = object(value)?;
    Ok(proto::logprobs_result::TopCandidates {
        candidates: repeated(fields, "candidates", read_logprob_candidate, false)?,
    })
}

fn logprob_candidate(value: &proto::logprobs_result::Candidate) -> Value {
    json!({
        "token": value.token,
        "token_id": value.token_id,
        "log_probability": value.log_probability,
    })
}

fn read_logprob_candidate(value: &Value) -> Result<proto::logprobs_result::Candidate> {
    let fields = object(value)?;
    Ok(proto::logprobs_result::Candidate {
        token: added(fields, "token")?,
        token_id: added(fields, "token_id")?,
        log_probability: added(fields, "log_probability")?,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
