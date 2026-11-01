//! Assertions on Responses citations: the `output_text` extras a recorded
//! message item states, and the text blocks a route delivered for it.

use futures::StreamExt;
use rig_core::completion::Usage;
use rig_core::message::{AssistantContent, Citation, Source, SourceLocation};
use rig_core::streaming::{CompletionStream, Item, StreamEvent};
use serde_json::{Map, Value};

/// What a drained stream delivered.
pub struct Drained {
    /// The text blocks its parts ended with, in order.
    pub texts: Vec<AssistantContent>,
    /// The `type` of every unmodeled payload it delivered, in order.
    pub unknown_types: Vec<String>,
    /// The usage of the response it folded into.
    pub usage: Usage,
}

/// Drain `stream`, check its event grammar, and keep its text blocks and
/// unmodeled payload types.
pub async fn drain(mut stream: CompletionStream) -> Drained {
    let mut items = Vec::new();
    let mut texts = Vec::new();
    let mut unknown_types = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("the stream item is ok");
        items.push(Ok(item.clone()));
        match item {
            Item::Event(StreamEvent::End { content, .. }) if known_text(&content).is_some() => {
                texts.push(content)
            }
            Item::Unknown(payload) => unknown_types.push(
                serde_json::to_value(&payload).expect("the payload serializes")["type"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            ),
            _ => {}
        }
    }
    let response = stream.finish().await.expect("the stream ends");
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(
        &items,
        &response.choice,
    );
    Drained {
        texts,
        unknown_types,
        usage: response.usage,
    }
}

/// The text blocks of a reply's choice, in order.
pub fn choice_texts(choice: &[AssistantContent]) -> Vec<AssistantContent> {
    choice
        .iter()
        .filter(|content| known_text(content).is_some())
        .cloned()
        .collect()
}

fn known_text(content: &AssistantContent) -> Option<&str> {
    match content {
        AssistantContent::Text(text) => Some(text.text.as_str()),
        AssistantContent::CarriedText(text)
            if matches!(
                text.content,
                rig_core::message::CarriedTextContent::Text(_)
                    | rig_core::message::CarriedTextContent::Refusal(_)
            ) =>
        {
            Some(text.text())
        }
        _ => None,
    }
}

fn content_citations(content: &AssistantContent) -> &[Citation] {
    match content {
        AssistantContent::Text(text) => text.citations(),
        AssistantContent::CarriedText(text) if known_text(content).is_some() => text.citations(),
        _ => &[],
    }
}

fn cited<'a>(content: &'a AssistantContent, citation: &Citation) -> Option<&'a str> {
    match content {
        AssistantContent::Text(text) => text.cited(citation),
        AssistantContent::CarriedText(_) => {
            let text = known_text(content)?;
            match citation.span {
                Some(span) => text.get(span.range()),
                None => Some(text),
            }
        }
        _ => None,
    }
}

/// Preserved part extras for carried text; containing-item extras for ordinary Text.
pub fn content_extras(content: &AssistantContent) -> Option<Map<String, Value>> {
    match content {
        AssistantContent::Text(text) => recorded_extras(&text.native.as_ref()?.item),
        AssistantContent::CarriedText(text) if known_text(content).is_some() => {
            extras_from_parts(std::iter::once(&text.payload.item))
        }
        _ => None,
    }
}

/// The extras a recorded message item's content parts state, in part
/// order: every non-empty sibling of `text` and `type`, arrays concatenated
/// across parts.
pub fn recorded_extras(item: &Value) -> Option<Map<String, Value>> {
    extras_from_parts(item["content"].as_array().into_iter().flatten())
}

fn extras_from_parts<'a>(parts: impl IntoIterator<Item = &'a Value>) -> Option<Map<String, Value>> {
    let mut extras = Map::new();
    for part in parts {
        for (key, value) in part.as_object().into_iter().flatten() {
            let empty = value.is_null()
                || value.as_array().is_some_and(Vec::is_empty)
                || value.as_object().is_some_and(Map::is_empty);
            if key == "text" || key == "type" || empty {
                continue;
            }
            match (extras.get_mut(key), value) {
                (Some(Value::Array(existing)), Value::Array(more)) => {
                    existing.extend(more.iter().cloned());
                }
                _ => {
                    extras.insert(key.clone(), value.clone());
                }
            }
        }
    }
    (!extras.is_empty()).then_some(extras)
}

/// A recorded message item's visible text, its parts joined.
pub fn recorded_text(item: &Value) -> String {
    item["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect()
}

/// The annotations a recorded message item states.
pub fn annotation_count(item: &Value) -> usize {
    item["content"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|part| part["annotations"].as_array().map_or(0, Vec::len))
        .sum()
}

/// The message items among `items` that state text.
pub fn messages<'a>(items: impl IntoIterator<Item = &'a Value>) -> Vec<Value> {
    items
        .into_iter()
        .filter(|item| item["type"] == "message" && !recorded_text(item).is_empty())
        .cloned()
        .collect()
}

/// The message items of a recorded stream's `output_item.done` frames.
pub fn item_snapshots(frames: &[Value]) -> Vec<Value> {
    let items: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["type"] == "response.output_item.done")
        .map(|frame| &frame["item"])
        .collect();
    messages(items)
}

/// The recorded frames of type `kind`.
pub fn frames_of<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
    frames
        .iter()
        .filter(|frame| frame["type"] == kind)
        .collect()
}

/// The sources a recorded message item's `url_citation` annotations name,
/// one per annotation in part order. Panics on any other annotation type,
/// which no recording holds.
pub fn recorded_url_sources(item: &Value) -> Vec<Source> {
    item["content"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|part| part["annotations"].as_array().into_iter().flatten())
        .map(|annotation| {
            assert_eq!(annotation["type"], "url_citation", "a recorded annotation");
            let url = annotation["url"].as_str().expect("the citation's url");
            let source = Source::new(SourceLocation::Url {
                url: url.to_owned(),
            });
            match annotation["title"].as_str() {
                Some(title) => source.title(title),
                None => source,
            }
        })
        .collect()
}

/// Each text block with text ends with the text and extras of its recorded
/// message item, in order: one part per item, its extras stated once. Its
/// citations are the item's annotations, each citing the whole block,
/// since the offsets' unit is undocumented.
pub fn assert_texts_match_items(route: &str, texts: &[AssistantContent], items: &[Value]) {
    let texts: Vec<&AssistantContent> = texts
        .iter()
        .filter(|text| known_text(text).is_some_and(|text| !text.is_empty()))
        .collect();
    assert_eq!(
        texts.len(),
        items.len(),
        "{route}: one text part per message item"
    );
    for (text, item) in texts.into_iter().zip(items) {
        assert_eq!(
            known_text(text).expect("selected known text"),
            recorded_text(item),
            "{route}: the item's text"
        );
        assert_eq!(
            content_extras(text),
            recorded_extras(item),
            "{route}: the item's extras, once"
        );
        let citations = content_citations(text);
        assert_eq!(
            citations
                .iter()
                .map(|citation| citation.sources.clone())
                .collect::<Vec<_>>(),
            recorded_url_sources(item)
                .into_iter()
                .map(|source| vec![source])
                .collect::<Vec<_>>(),
            "{route}: one citation per annotation"
        );
        for citation in citations {
            assert_eq!(citation.span, None, "{route}: the citation's span");
            assert_eq!(
                cited(text, citation),
                known_text(text),
                "{route}: the cited text"
            );
        }
    }
}
