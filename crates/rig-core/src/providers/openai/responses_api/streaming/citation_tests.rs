use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::{AssistantContent, CarriedPosition, Message, Source, SourceLocation};
use crate::operation::Completion;
use crate::providers::openai::wire::OpenAIConfig;
use crate::wire::{Mode, Operation, Wire, WireFrame};
use serde_json::{Value, json};

fn message() -> Value {
    json!({"type":"message","id":"msg_citations","role":"assistant","status":"completed","phase":"final_answer","content":[
        {"type":"output_text","text":"first","annotations":[
            {"type":"url_citation","url":"https://example.com/a","title":"page","start_index":1,"end_index":999},
            {"type":"file_citation","file_id":"file_a","filename":"a.txt"},
            {"type":"file_path","file_id":"file_b","filename":"b.txt"},
            {"type":"container_file_citation","file_id":"file_c","filename":"c.txt","container_id":"container_1"}
        ],"future":{"keep":[null,7]}},
        {"type":"future_content","annotations":[{"type":"url_citation","url":"https://example.com/opaque"}],"raw":[null,9]},
        {"type":"output_text","text":"second","annotations":[{"type":"url_citation","url":"https://example.com/b"}]}
    ]})
}

fn response(item: Value) -> Value {
    json!({"id":"resp_citations","object":"response","created_at":1,"model":"gpt-5.4","status":"completed","output":[item]})
}

fn decoded() -> (CompletionResponse, CompletionResponse) {
    let wire = OpenAIConfig::new("fixture-key").responses("gpt-5.4");
    let item = message();
    let whole = response(item.clone());
    let unary = crate::test_utils::history::decode(&wire, Mode::Unary,
        [WireFrame::Text(whole.to_string())]).expect("actual unary decoder");
    let streamed = crate::test_utils::history::decode(&wire, Mode::Streaming, [
        WireFrame::Text(json!({"type":"response.output_item.done","sequence_number":0,"output_index":0,"item":item}).to_string()),
        WireFrame::Text(json!({"type":"response.completed","sequence_number":1,"response":whole}).to_string()),
    ]).expect("actual streamed decoder");
    (unary, streamed)
}

fn replay(response: &CompletionResponse) -> Value {
    let wire = OpenAIConfig::new("fixture-key").responses("different-model");
    let request = CompletionRequest::new("continue").message(response.message().expect("actual assistant history"));
    let request = Completion::prepare(request, &wire.describe()).expect("real history preparation");
    let encoded = wire.encode(request, Mode::Unary).expect("real Responses request boundary");
    let body = crate::test_utils::json_body(&encoded.request);
    body["input"].as_array().expect("input").iter()
        .find(|item| item["role"] == "assistant").expect("replayed message").clone()
}

#[test]
fn carried_citation_accessors_preserve_four_sources_order_and_unary_stream_parity() {
    let (unary, streamed) = decoded();
    assert_eq!(unary.choice, streamed.choice);
    assert_eq!(unary.message(), streamed.message());
    let expected = vec![
        Source::new(SourceLocation::Url { url: "https://example.com/a".to_owned() }).title("page"),
        Source::new(SourceLocation::File { file_id: "file_a".to_owned(), filename: Some("a.txt".to_owned()), container_id: None }),
        Source::new(SourceLocation::File { file_id: "file_b".to_owned(), filename: Some("b.txt".to_owned()), container_id: None }),
        Source::new(SourceLocation::File { file_id: "file_c".to_owned(), filename: Some("c.txt".to_owned()), container_id: Some("container_1".to_owned()) }),
    ];
    for response in [&unary, &streamed] {
        let [AssistantContent::CarriedText(first), AssistantContent::CarriedText(opaque), AssistantContent::CarriedText(second)] = response.choice.as_slice() else {
            panic!("three ordered carried parts");
        };
        assert_eq!(first.citations().len(), 4);
        assert_eq!(first.citations().iter().flat_map(|citation| citation.sources.clone()).collect::<Vec<_>>(), expected);
        assert!(first.citations().iter().all(|citation| citation.span.is_none()), "Responses offsets have no specified unit");
        assert!(opaque.citations().is_empty(), "opaque annotations remain raw only");
        assert_eq!(second.citations().len(), 1);
        assert_eq!(second.citations()[0].sources, vec![Source::new(SourceLocation::Url { url: "https://example.com/b".to_owned() })]);
        assert_eq!(first.payload.position, CarriedPosition::MessageContent { item_index: 0, part_index: 0 });
        assert_eq!(opaque.payload.position, CarriedPosition::MessageContent { item_index: 0, part_index: 1 });
        assert_eq!(second.payload.position, CarriedPosition::MessageContent { item_index: 0, part_index: 2 });
        assert_eq!(first.payload.provider.as_deref(), Some("openai"));
        assert_eq!(first.item_id.as_deref(), Some("msg_citations"));
        assert_eq!(first.phase.as_deref(), Some("final_answer"));
        assert_eq!(replay(response), message(), "adding canonical citations leaves raw replay exact");
    }
}

#[test]
fn known_carried_text_edits_hide_citations_without_changing_raw_annotations_or_origin() {
    let (mut response, _) = decoded();
    let before = response.message().expect("history before edit");
    let first = match &mut response.choice[0] {
        AssistantContent::CarriedText(first) => first,
        _ => panic!("known carried text"),
    };
    let raw = first.payload.clone();
    assert_eq!(first.citations().len(), 4);
    *first.text_mut() = "edited".to_owned();
    assert!(first.citations().is_empty());
    assert_eq!(first.payload, raw);
    let Message::Assistant(before) = before else { panic!("assistant"); };
    let Message::Assistant(after) = response.message().expect("edited history") else { panic!("assistant"); };
    assert_eq!(after.origin, before.origin);
    let mut expected = message();
    expected["content"][0]["text"] = json!("edited");
    assert_eq!(replay(&response), expected, "only the known text field is edited");
    let AssistantContent::CarriedText(second) = &response.choice[2] else { panic!("second known part"); };
    assert_eq!(second.citations().len(), 1, "a sibling's citations keep their own fingerprint");
}
