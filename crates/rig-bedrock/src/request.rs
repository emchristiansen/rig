//! The Converse request body, built as JSON from a prepared request. An
//! assistant block goes back through [`AssistantContent::replay`]: a current
//! provider item as it came, anything else from its canonical fields for
//! the target model's family. Every call id is spelled by one [`WireIds`].

use std::collections::HashMap;

use base64::Engine as _;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::prelude::BASE64_STANDARD;
use rig_core::completion::options::{BaseInput, FinalBody, RawAt, request_params};
use rig_core::completion::{CacheRetention, CompletionRequest, Message, Replay};
use rig_core::error::EncodeError;
use rig_core::message::{
    AssistantContent, Document, DocumentData, DocumentMediaType, DocumentSourceKind, Image,
    MimeType, ToolChoice, ToolResultContent, UserContent, Video,
};
use rig_core::providers::internal::wire_ids::WireIds;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::completion::{Converse, Family};

/// Standard base64, padded or not.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// What stands in for a user message left empty once blank text is skipped:
/// Converse rejects blank text and empty content.
const EMPTY_TEXT: &str = "<empty>";

/// The Converse body of `request` for `model` on `wire`: the wire's encoding
/// of the request, then the mapped options, then the provider options, then
/// `additional_params`, which merge under `additionalModelRequestFields`.
pub(crate) fn body(
    wire: &Converse,
    request: &CompletionRequest,
    model: &str,
) -> Result<FinalBody, EncodeError> {
    request_params(
        wire,
        request,
        |input| base(wire, request, model, input),
        RawAt::Under("/additionalModelRequestFields"),
        &[],
    )
}

/// A cache checkpoint with the TTL `cache` asks for.
fn checkpoint(cache: CacheRetention) -> Value {
    match cache {
        CacheRetention::Long => json!({ "cachePoint": { "type": "default", "ttl": "1h" } }),
        _ => json!({ "cachePoint": { "type": "default" } }),
    }
}

/// The wire's own encoding of `request`, with the cache checkpoints the
/// mapped `cache` places.
fn base(
    wire: &Converse,
    request: &CompletionRequest,
    model: &str,
    input: &mut BaseInput<'_>,
) -> Result<Map<String, Value>, EncodeError> {
    let family = wire.family(model);
    let cache = input.cache().filter(|cache| *cache != CacheRetention::None);
    let history = &request.chat_history;
    let ids = WireIds::for_target(history, wire, model)?;
    let mut system: Vec<Value> = history
        .iter()
        .filter_map(|message| match message {
            Message::System { content } if !content.is_empty() => Some(json!({ "text": content })),
            _ => None,
        })
        .collect();
    if let Some(cache) = cache
        && !system.is_empty()
    {
        system.push(checkpoint(cache));
    }
    let mut messages: Vec<Value> = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        let mut content = Vec::new();
        let role = match message {
            Message::System { .. } => continue,
            Message::User { content: parts } => {
                for (content_index, part) in parts.iter().enumerate() {
                    content.extend(user(part, family, &ids, message_index, content_index)?);
                }
                if content.is_empty() {
                    content.push(json!({ "text": EMPTY_TEXT }));
                }
                "user"
            }
            Message::Assistant(turn) => {
                for (content_index, block) in turn.content.iter().enumerate() {
                    content.extend(assistant(block, wire, family, model, ids.of(message_index, content_index))?);
                }
                "assistant"
            }
        };
        // `adapt` alternates the roles and drops a turn with nothing to send.
        messages.push(json!({ "role": role, "content": content }));
    }
    // Bedrock requires a name on every document and Rig's `Document` carries
    // none. A name repeated within one request takes a counter.
    let mut seen = HashMap::<String, usize>::new();
    for name in blocks(&mut messages).filter_map(|block| block.pointer_mut("/document/name")) {
        if let Value::String(name) = name {
            let count = seen.entry(name.clone()).or_default();
            *count += 1;
            if *count > 1 {
                *name = format!("{name}-{count}");
            }
        }
    }
    // A reasoning turn prevents a trailing checkpoint; the system checkpoint
    // remains independent.
    if let Some(cache) = cache {
        let reasoning = history.iter().any(|message| matches!(message,
            Message::Assistant(turn) if turn.content.iter().any(|block| matches!(block,
                AssistantContent::Reasoning(_) | AssistantContent::CarriedReasoning(_)
            ))
        ));
        if !reasoning && let Some(Value::Array(last)) =
            messages.last_mut().and_then(|last| last.get_mut("content"))
        {
            last.push(checkpoint(cache));
        }
    }
    let output = match &request.output_schema {
        Some(schema) => Some(
            json!({ "textFormat": { "type": "json_schema", "structure": { "jsonSchema": {
            "schema": serde_json::to_string(schema.as_value())?,
            "name": request.output_schema_name(),
        } } } }),
        ),
        None => None,
    };
    let inference = json!({
        "temperature": request.temperature.map(|temperature| f64::from(temperature as f32)),
        "maxTokens": request.max_tokens.map(|max_tokens| max_tokens as i32),
    });
    let body = present(json!({
        "system": (!system.is_empty()).then_some(system),
        "inferenceConfig": present(inference),
        "toolConfig": tool_config(request),
        "outputConfig": output,
        "messages": messages,
    }));
    Ok(match body {
        Value::Object(body) => body,
        _ => Map::new(),
    })
}

/// `value` without its `null` fields: Converse reads an absent field, not a
/// `null` one.
fn present(mut value: Value) -> Value {
    if let Value::Object(fields) = &mut value {
        fields.retain(|_, field| !field.is_null());
    }
    value
}

/// Every content block of `messages`.
fn blocks(messages: &mut [Value]) -> impl Iterator<Item = &mut Value> {
    messages
        .iter_mut()
        .filter_map(|message| message.get_mut("content")?.as_array_mut())
        .flatten()
}

/// The `toolConfig` of `request`: none without tools or with
/// `ToolChoice::None`, when the adapter sends calls and results as text.
fn tool_config(request: &CompletionRequest) -> Option<Value> {
    let choice = match &request.tool_choice {
        Some(ToolChoice::None) => return None,
        Some(ToolChoice::Auto) => Some(json!({ "auto": {} })),
        Some(ToolChoice::Required) => Some(json!({ "any": {} })),
        Some(ToolChoice::Specific { function_names }) => function_names
            .first()
            .map(|name| json!({ "tool": { "name": name.as_str() } })),
        None => None,
    };
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({ "toolSpec": {
                "name": tool.name.as_str(),
                "description": tool.description,
                "inputSchema": { "json": tool.parameters },
            } })
        })
        .collect();
    (!tools.is_empty()).then(|| present(json!({ "tools": tools, "toolChoice": choice })))
}

/// The Converse block for one assistant block, or `None` when there is
/// nothing to send ([`sends`] states which).
fn assistant(
    block: &AssistantContent,
    wire: &Converse,
    family: Family,
    model: &str,
    wire_id: Option<&str>,
) -> Result<Option<Value>, EncodeError> {
    rig_core::message::json_only_wire_tool_call("Bedrock Converse", block)?;
    if let AssistantContent::ProviderItem(item) = block {
        return Err(rig_core::message::UnreplayableProviderItem::new("Bedrock Converse", item).into());
    }
    if let AssistantContent::Opaque(opaque) = block {
        return Ok(opaque.replay.then(|| whole_numbers(opaque.item.clone())));
    }
    if let Replay::Item(item) = block.replay(wire, wire_id) {
        return Ok(Some(whole_numbers(item.into_owned())));
    }
    Ok(match block {
        AssistantContent::Text(text) => Some(json!({ "text": text.text })),
        AssistantContent::ToolCall(call) => Some(json!({ "toolUse": {
            "toolUseId": wire_id.ok_or_else(|| EncodeError::request("Bedrock tool call has no planned wire ID"))?,
            "name": call.function.name.as_str(),
            "input": call.function.arguments,
        } })),
        // Redacted reasoning is nothing without its bytes.
        AssistantContent::Reasoning(reasoning)
            if reasoning.redacted || reasoning.text.trim().is_empty() =>
        {
            None
        }
        // Claude rejects unsigned reasoning: it goes back as text.
        AssistantContent::Reasoning(reasoning) if family == Family::Claude => {
            Some(json!({ "text": reasoning.text }))
        }
        AssistantContent::Reasoning(reasoning) => {
            Some(json!({ "reasoningContent": { "reasoningText": { "text": reasoning.text } } }))
        }
        AssistantContent::CarriedReasoning(reasoning) => carried_reasoning(reasoning, model)?,
        AssistantContent::CarriedText(part) => match part.known_text_projection() {
            Some(text) if !text.is_empty() => Some(json!({ "text": text })),
            Some(_) => None,
            None => return Err(rig_core::message::UnrepresentableOpaqueContent::new("Bedrock Converse").into()),
        },
        AssistantContent::Image(image) => Some(json!({ "image": self::image(image)? })),
        AssistantContent::Opaque(_) => None,
        _ => {
            return Err(EncodeError::request(
                "Converse has no form for this assistant block",
            ));
        }
    })
}

/// Rebuild preserved reasoning from its known blocks after checking the
/// destination issuer. Ciphertext never becomes a redacted byte source.
fn carried_reasoning(
    reasoning: &rig_core::message::CarriedReasoning,
    model: &str,
) -> Result<Option<Value>, EncodeError> {
    use rig_core::message::CarriedReasoningPart as Part;
    if reasoning.has_opaque_parts() {
        return Err(rig_core::message::UnrepresentableOpaqueContent::new("Bedrock Converse").into());
    }
    let issuer = if model.contains("anthropic.claude") { "anthropic" } else { crate::completion::PROVIDER_NAME };
    if !reasoning.replayable_to(issuer) {
        return Ok(None);
    }
    let parts: Vec<&Part> = reasoning.content.iter().filter(|part| !matches!(part, Part::Encrypted(_))).collect();
    let encrypted = reasoning.content.len() - parts.len();
    if encrypted > 0 {
        tracing::warn!(dropped = encrypted, "dropping foreign encrypted reasoning payload(s); Bedrock cannot verify another provider's ciphertext");
        if parts.is_empty() {
            return Ok(None);
        }
    }
    let redacted: Vec<&str> = parts.iter().filter_map(|part| match part {
        Part::Redacted { data } => Some(data.as_str()),
        _ => None,
    }).collect();
    if !redacted.is_empty() && redacted.len() == parts.len() {
        if redacted.len() > 1 {
            tracing::warn!(dropped = redacted.len() - 1, "dropping extra redacted reasoning payloads; Bedrock carries one redactedContent blob per block");
        }
        return match BASE64_STANDARD.decode(redacted[0]) {
            Ok(bytes) => Ok(Some(json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } }))),
            Err(error) => {
                tracing::warn!(%error, "dropping redacted reasoning content that is not valid base64");
                Ok(None)
            }
        };
    }
    if !redacted.is_empty() {
        tracing::warn!(dropped = redacted.len(), "dropping redacted reasoning payloads Bedrock cannot carry alongside reasoning text; replaying the text only");
    }
    let known: Vec<&Part> = parts.into_iter().filter(|part| !matches!(part, Part::Redacted { .. })).collect();
    let signed: Vec<&str> = known.iter().filter_map(|part| match part {
        Part::Text { signature: Some(signature), .. } => Some(signature.as_str()),
        _ => None,
    }).collect();
    if signed.len() > 1 {
        return Err(EncodeError::request("AWS Bedrock does not support multiple signed reasoning text blocks"));
    }
    if signed.len() == 1 && known.len() > 1 {
        return Err(EncodeError::request("AWS Bedrock requires a single signed reasoning text block without additional reasoning parts"));
    }
    let text = known.iter().filter_map(|part| part.visible_text()).collect::<Vec<_>>().join("\n");
    if text.is_empty() && signed.is_empty() {
        return Err(EncodeError::request("AWS Bedrock reasoning conversion requires at least one text or summary block"));
    }
    let mut text_block = json!({ "text": text });
    if let Some(signature) = signed.first() {
        text_block["signature"] = json!(signature);
    }
    Ok(Some(json!({ "reasoningContent": { "reasoningText": text_block } })))
}

/// Whether [`assistant`] sends `block` on `wire`, read from the encoder
/// itself so the two never disagree.
pub(crate) fn sends(block: &AssistantContent, wire: &Converse) -> bool {
    if rig_core::message::json_only_wire_tool_call("Bedrock Converse", block).is_err()
        || matches!(block, AssistantContent::ProviderItem(_))
    {
        return false;
    }
    if let Replay::Item(_) = block.replay(wire, None) {
        return true;
    }
    match block {
        AssistantContent::Text(_) | AssistantContent::ToolCall(_) => true,
        AssistantContent::Reasoning(reasoning) => !reasoning.redacted && !reasoning.text.trim().is_empty(),
        AssistantContent::CarriedReasoning(reasoning) => !reasoning.has_opaque_parts(),
        AssistantContent::CarriedText(part) => part.known_text_projection().is_some_and(|text| !text.is_empty()),
        AssistantContent::Opaque(opaque) => opaque.replay,
        AssistantContent::Image(_)
        | AssistantContent::CustomToolCall(_)
        | AssistantContent::ProviderItem(_) => false,
    }
}

/// `value` with each whole number written as an integer: a store may write
/// one as a float, and Converse's integer fields take integers.
fn whole_numbers(value: Value) -> Value {
    match value {
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() && float.fract() == 0.0 && float.abs() < 1e15 => {
                json!(float as i64)
            }
            _ => Value::Number(number),
        },
        Value::Array(values) => Value::Array(values.into_iter().map(whole_numbers).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, whole_numbers(value)))
                .collect(),
        ),
        value => value,
    }
}

/// The Converse blocks for one piece of user content. Converse rejects
/// blank text, so it has none.
fn user(content: &UserContent, family: Family, ids: &WireIds, message_index: usize, content_index: usize) -> Result<Vec<Value>, EncodeError> {
    let text = |text: &str| (!text.trim().is_empty()).then(|| json!({ "text": text }));
    Ok(match content {
        UserContent::Text(part) => text(&part.text).into_iter().collect(),
        UserContent::ToolResult(result) => {
            if result.answers == rig_core::message::AnsweredToolCall::Custom {
                return Err(rig_core::message::UnrepresentableToolCall::CustomCall {
                    wire: "Bedrock Converse",
                    name: result.name.clone(),
                }.into());
            }
            let mut parts = Vec::new();
            for part in &result.content {
                parts.extend(match part {
                    ToolResultContent::Text(part) => text(&part.text),
                    ToolResultContent::Image(part) => Some(json!({ "image": image(part)? })),
                    // Converse takes an object as a result's JSON.
                    ToolResultContent::Json { value } if value.is_object() => {
                        Some(json!({ "json": value }))
                    }
                    ToolResultContent::Json { value } => {
                        Some(json!({ "json": { "result": value } }))
                    }
                });
            }
            // Converse reads a result with no status as a success, and
            // documents the field for Nova and Claude only.
            let failed = result.is_error && family != Family::Other;
            let block = present(json!({
                "toolUseId": ids.spell(message_index, content_index)?,
                "content": parts,
                "status": failed.then_some("error"),
            }));
            vec![json!({ "toolResult": block })]
        }
        UserContent::Image(part) => vec![json!({ "image": image(part)? })],
        // Converse requires accompanying prompt text for document blocks.
        UserContent::Document(part) => vec![
            json!({ "text": "Use provided document" }),
            json!({ "document": document(part)? }),
        ],
        UserContent::Video(part) => vec![json!({ "video": video(part)? })],
        UserContent::Audio(_) => return Err(EncodeError::request("Converse takes no audio")),
    })
}

/// The source of a media part, as base64 bytes or an S3 object, and the
/// bytes that name it.
fn source(data: &DocumentSourceKind) -> Result<(Value, Vec<u8>), EncodeError> {
    let bytes = match data {
        DocumentSourceKind::Base64(data) => BASE64.decode(data).map_err(EncodeError::request)?,
        DocumentSourceKind::Raw(bytes) => bytes.clone(),
        DocumentSourceKind::Url(uri) if uri.starts_with("s3://") => {
            return Ok((
                json!({ "s3Location": { "uri": uri } }),
                uri.as_bytes().to_vec(),
            ));
        }
        data => {
            return Err(EncodeError::request(format!(
                "Converse takes base64 data or an s3:// URL, not {data}"
            )));
        }
    };
    Ok((json!({ "bytes": BASE64_STANDARD.encode(&bytes) }), bytes))
}

/// A Converse media block of `data` in the format the MIME subtype of
/// `mime` names, when Converse lists it.
fn media(
    mime: Option<&str>,
    listed: &[&str],
    data: &DocumentSourceKind,
) -> Result<Value, EncodeError> {
    let format = mime
        .and_then(|mime| mime.split_once('/'))
        .map(|(_, format)| format)
        .filter(|format| listed.contains(format))
        .ok_or_else(|| EncodeError::request("Converse takes no media in this format"))?;
    Ok(json!({ "format": format, "source": source(data)?.0 }))
}

/// The Converse image block for `image`: PNG, JPEG, GIF or WEBP.
pub(crate) fn image(image: &Image) -> Result<Value, EncodeError> {
    let mime = image.media_type.as_ref().map(MimeType::to_mime_type);
    let format = match mime {
        Some("image/jpeg") => "jpeg",
        Some("image/png") => "png",
        Some("image/gif") => "gif",
        Some("image/webp") => "webp",
        Some(mime) => return Err(EncodeError::request(format!("Unsupported format {mime}"))),
        None => return Err(EncodeError::request("Converse takes no media in this format")),
    };
    let DocumentSourceKind::Base64(data) = &image.data else {
        return Err(EncodeError::request("Only base64 encoded strings are allowed for image input on AWS Bedrock"));
    };
    let bytes = BASE64_STANDARD.decode(data).map_err(EncodeError::request)?;
    Ok(json!({"format": format, "source": {"bytes": BASE64_STANDARD.encode(bytes)}}))
}

/// The Converse video block for `video`, in a format Converse lists.
pub(crate) fn video(video: &Video) -> Result<Value, EncodeError> {
    let mime = video.media_type.as_ref().map(MimeType::to_mime_type);
    media(mime, &["mp4", "mpeg", "mov", "webm"], &video.data)
}

/// The Converse document block for a PDF, plain text, HTML, Markdown or CSV
/// document. The name is the content's digest, so a request
/// is byte-stable across turns and runs, which prompt-cache prefixes and
/// recorded replays rely on.
pub(crate) fn document(document: &Document) -> Result<Value, EncodeError> {
    let format = match document.media_type.as_ref() {
        Some(DocumentMediaType::PDF) => "pdf",
        Some(DocumentMediaType::TXT) => "txt",
        Some(DocumentMediaType::HTML) => "html",
        Some(DocumentMediaType::MARKDOWN) => "md",
        Some(DocumentMediaType::CSV) => "csv",
        Some(media_type) => return Err(EncodeError::request(format!("Unsupported media type {}", media_type.to_mime_type()))),
        None => return Err(EncodeError::request("Converse takes no media in this format")),
    };
    let bytes = match &document.data {
        DocumentData::Text(text) => text.as_bytes().to_vec(),
        DocumentData::File(DocumentSourceKind::Base64(data)) => BASE64_STANDARD.decode(data).map_err(EncodeError::request)?,
        DocumentData::File(data) => return Err(EncodeError::request(format!("Unsupported document kind: {data}"))),
    };
    let mut block = json!({"format": format, "source": {"bytes": BASE64_STANDARD.encode(&bytes)}});
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if let Value::Object(fields) = &mut block {
        fields.insert("name".to_owned(), json!(format!("document-{digest}")));
    }
    Ok(block)
}

#[cfg(test)]
mod tests;
