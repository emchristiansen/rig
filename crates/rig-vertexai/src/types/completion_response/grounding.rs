//! Places SDK grounding and whole-block recitations on canonical answer parts.

use google_cloud_aiplatform_v1::model as sdk;
use rig_core::message::{Source, SourceLocation};
use rig_core::wire::{SpanUnit, WireCitation, WireSpan};

/// The answer text a reply wrote so far: each text block with its text, in
/// order. Thought text is not answer text.
#[derive(Debug, Default)]
pub(super) struct AnswerText {
    blocks: Vec<(usize, String)>,
}

impl AnswerText {
    /// Start the text block at `index` with `text`.
    pub(super) fn open(&mut self, index: usize, text: &str) {
        self.blocks.push((index, text.to_owned()));
    }

    /// Append `text` to the block at `index`, returning the byte offset it
    /// starts at.
    pub(super) fn push(&mut self, index: usize, text: &str) -> Option<usize> {
        let (_, held) = self.blocks.iter_mut().find(|(at, _)| *at == index)?;
        let offset = held.len();
        held.push_str(text);
        Some(offset)
    }

    /// The block and offset in it of byte `at` of the whole answer text, a
    /// span's start.
    fn locate(&self, at: usize) -> Option<(usize, usize)> {
        let mut start = 0;
        for (index, text) in &self.blocks {
            if at < start + text.len() {
                return Some((*index, at - start));
            }
            start += text.len();
        }
        None
    }

    /// The position in the whole answer text of `offset` in block `index`.
    fn absolute(&self, index: usize, offset: usize) -> Option<usize> {
        let mut start: usize = 0;
        for (at, text) in &self.blocks {
            if *at == index {
                return start.checked_add(offset);
            }
            start += text.len();
        }
        None
    }

    /// Whether block `index` holds `quoted` at `range`.
    fn holds(&self, index: usize, range: std::ops::Range<usize>, quoted: &str) -> bool {
        self.blocks
            .iter()
            .find(|(at, _)| *at == index)
            .and_then(|(_, text)| text.get(range))
            == Some(quoted)
    }

    /// The occurrence of `quoted` nearest `near`, a position in the whole
    /// answer text, as its block and offset.
    fn find(&self, quoted: &str, near: usize) -> Option<(usize, usize)> {
        let mut start = 0;
        let mut best: Option<(usize, usize, usize)> = None;
        for (index, text) in &self.blocks {
            for (offset, _) in text.match_indices(quoted) {
                let distance = (start + offset).abs_diff(near);
                if best.is_none_or(|(_, _, held)| distance < held) {
                    best = Some((*index, offset, distance));
                }
            }
            start += text.len();
        }
        best.map(|(index, offset, _)| (index, offset))
    }
}

/// Read SDK supports in their original order, preserving confidence positions.
pub(super) fn grounding(
    metadata: &sdk::GroundingMetadata,
    placed: &[Option<(usize, usize)>],
    answer: &AnswerText,
) -> Vec<(usize, WireCitation)> {
    metadata.grounding_supports.iter().filter_map(|support| {
        let sources: Vec<Source> = support.grounding_chunk_indices.iter().enumerate()
            .filter_map(|(at, index)| {
                let chunk = metadata.grounding_chunks.get(usize::try_from(*index).ok()?)?;
                let source = chunk_source(chunk)?;
                Some(match support.confidence_scores.get(at).copied().filter(|score| score.is_finite()) {
                    Some(score) => source.confidence(score),
                    None => source,
                })
            }).collect();
        if sources.is_empty() { return None; }
        let (index, span) = segment(support.segment.as_ref()?, placed, answer)?;
        Some((index, WireCitation::new(Some(span), sources)))
    }).collect()
}

fn segment(
    segment: &sdk::Segment,
    placed: &[Option<(usize, usize)>],
    answer: &AnswerText,
) -> Option<(usize, WireSpan)> {
    // SDK negative offsets were non-u64 JSON values and therefore read as zero.
    let offset = |value: i32| usize::try_from(value).unwrap_or(0);
    let (start, end) = (offset(segment.start_index), offset(segment.end_index));
    let len = end.checked_sub(start)?;
    let in_part = placed.get(offset(segment.part_index)).copied().flatten()
        .and_then(|(index, at)| Some((index, at.checked_add(start)?)));
    let in_answer = answer.locate(start);
    let quoted = (!segment.text.is_empty()).then_some(segment.text.as_str());
    let (index, at) = match quoted {
        None => in_part.or(in_answer)?,
        Some(quoted) => [in_part, in_answer].into_iter().flatten()
            .find(|(index, at)| at.checked_add(len)
                .is_some_and(|stop| answer.holds(*index, *at..stop, quoted)))
            .or_else(|| {
                let near = in_part.and_then(|(index, at)| answer.absolute(index, at)).unwrap_or(start);
                answer.find(quoted, near)
            }).or(in_part).or(in_answer)?,
    };
    let span = WireSpan::new(at as u64, at.checked_add(len)? as u64, SpanUnit::Bytes);
    Some((index, match quoted {
        Some(quoted) => span.quoted(quoted),
        None => span,
    }))
}

fn chunk_source(chunk: &sdk::GroundingChunk) -> Option<Source> {
    let (location, title, text) = if let Some(web) = chunk.web() {
        (SourceLocation::Url { url: web.uri.as_ref()?.clone() }, web.title.as_ref(), None)
    } else if let Some(context) = chunk.retrieved_context() {
        let location = match &context.uri {
            Some(uri) => SourceLocation::Url { url: uri.clone() },
            None => SourceLocation::Document {
                index: None, id: context.document_name.clone(), within: None,
            },
        };
        (location, context.title.as_ref(), context.text.as_ref())
    } else if let Some(maps) = chunk.maps() {
        (SourceLocation::Url { url: maps.uri.as_ref()?.clone() }, maps.title.as_ref(), None)
    } else { return None; };
    let mut source = Source::new(location);
    if let Some(title) = title { source = source.title(title); }
    if let Some(text) = text { source = source.cited_text(text); }
    Some(source)
}

/// Vertex recitation offsets state no unit, so sources cite the first answer whole.
pub(super) fn recitations(
    metadata: &sdk::CitationMetadata,
    answer: &AnswerText,
) -> Vec<(usize, WireCitation)> {
    let first = answer.blocks.first().map(|(index, _)| *index);
    metadata.citations.iter().filter_map(|citation| {
        if citation.uri.is_empty() { return None; }
        let mut source = Source::new(SourceLocation::Url { url: citation.uri.clone() });
        if !citation.title.is_empty() { source = source.title(&citation.title); }
        Some((first?, WireCitation::new(None, vec![source])))
    }).collect()
}
