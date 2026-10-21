use super::*;
use rig_core::message::{CustomToolCall, ToolFunction};
use rig_core::streaming::ToolSlotState;
use serde_json::json;

fn part() -> Part { serde_json::from_value(json!(2)).expect("part") }
fn name(value: &str) -> ToolName { ToolName::new(value).expect("nonempty name") }
fn assembler() -> StreamedTurnAssembler {
    StreamedTurnAssembler::new(TurnPolicy::new(["add".to_owned()].into(), None, None).expect("policy"))
}
fn start(value: Option<&str>, state: ToolSlotState) -> Item<StreamEvent> {
    Item::Event(StreamEvent::Start { part:part(), kind:PartKind::ToolCall, source_order:None,
        name:value.map(name), namespace:None, tool_slot_state:Some(state) })
}
fn arguments(value: &str) -> Item<StreamEvent> {
    Item::Event(StreamEvent::Arguments { part:part(), json:value.to_owned() })
}
fn end(value: &str) -> Item<StreamEvent> {
    Item::Event(StreamEvent::End { part:part(), reasoning_authority:None,
        content:AssistantContent::ToolCall(ToolCall::from_wire("real-call", ToolFunction::new(name(value), json!({"x":1})))) })
}

#[test]
fn public_pending_name_keeps_arguments_and_waits_for_genuine_object_end() {
    let mut assembly=assembler();
    let items=[start(None,ToolSlotState::PendingName),arguments("{\"x\":"),
        start(Some("add"),ToolSlotState::Named),arguments("1}")];
    for item in &items {
        assert!(matches!(assembly.ingest(item).expect("ingest").as_slice(), [StreamedTurnEvent::HoldToolCall]));
    }
    assert_eq!(assembly.aggregated_arguments(part().index()),Some("{\"x\":1}"));
    assert_eq!(assembly.open_calls[&part().index()].held,items);
    let events=assembly.ingest(&end("add")).expect("genuine end");
    let [StreamedTurnEvent::EmitToolCall {call}] = events.as_slice() else {
        panic!("expected finalized object call");
    };
    assert_eq!(call.id,CallId::from_wire("real-call"));
    assert_eq!(call.function.arguments_value(),json!({"x":1}));
}

#[test]
fn engine_pending_name_delivers_all_real_arrivals_at_binding_in_order() {
    let mut assembly=assembler();
    let opening=start(None,ToolSlotState::PendingName);
    let delta=arguments("{\"x\":");
    let binding=start(Some("add"),ToolSlotState::Named);
    assembly.ingest_for_driver(&opening).expect("pending");
    assembly.ingest_for_driver(&delta).expect("delta");
    let events=assembly.ingest_for_driver(&binding).expect("bind");
    let [DriverTurnEvent::EmitResolvedItems{items}]=events.as_slice() else { panic!("early genuine arrivals"); };
    assert_eq!(items,&vec![opening,delta,binding]);
    assert_eq!(assembly.aggregated_arguments(part().index()),Some("{\"x\":"));
    assert!(assembly.open_calls[&part().index()].held.is_empty());
    assert!(assembly.pending_tool_calls.is_empty());
    assert!(matches!(assembly.ingest_for_driver(&arguments("1}")).expect("rest").as_slice(),
        [DriverTurnEvent::Public(StreamedTurnEvent::EmitToolCallDelta)]));
    assert!(matches!(assembly.ingest_for_driver(&end("add")).expect("object end").as_slice(),
        [DriverTurnEvent::Public(StreamedTurnEvent::EmitToolCall {..})]));
}

#[test]
fn early_repair_changes_only_named_start_and_public_view_waits_until_end() {
    for engine in [false,true] {
        let mut assembly=assembler();
        let opening=start(None,ToolSlotState::PendingName);
        let delta=arguments("{\"x\":1}");
        let binding=start(Some("missing"),ToolSlotState::Named);
        if engine {
            assembly.ingest_for_driver(&opening).expect("pending");
            assembly.ingest_for_driver(&delta).expect("delta");
            assert!(matches!(assembly.ingest_for_driver(&binding).expect("name").as_slice(),
                [DriverTurnEvent::Public(StreamedTurnEvent::InvalidToolCall(_))]));
        } else {
            assembly.ingest(&opening).expect("pending");
            assembly.ingest(&delta).expect("delta");
            assert!(matches!(assembly.ingest(&binding).expect("name").as_slice(),[StreamedTurnEvent::InvalidToolCall(_)]));
        }
        let resolution=StreamedResolution::Repaired{tool_name:"add".to_owned()};
        let expected=vec![opening,delta,start(Some("add"),ToolSlotState::Named)];
        if engine {
            let events=assembly.resolve_pending_invalid_for_driver(&resolution);
            let [DriverTurnEvent::EmitResolvedItems{items}]=events.as_slice() else {panic!("early repair arrivals");};
            assert_eq!(items,&expected);
        } else {
            assert!(assembly.resolve_pending_invalid(&resolution).is_empty());
            assert_eq!(assembly.open_calls[&part().index()].held,expected);
        }
        assert_eq!(assembly.aggregated_arguments(part().index()),Some("{\"x\":1}"));
        assert!(assembly.pending_tool_calls.is_empty());
        let events=assembly.ingest(&end("missing")).expect("finalized");
        let [StreamedTurnEvent::EmitToolCall{call}]=events.as_slice() else {panic!("genuine object end");};
        assert_eq!(call.function.name.as_str(),"add");
        assert_eq!(call.id,CallId::from_wire("real-call"));
    }
}

#[test]
fn namespaced_and_custom_refusals_keep_the_public_tuple_and_actual_content() {
    let mut assembly=assembler();
    let call=ToolCall::from_wire("provider-call",ToolFunction::new(name("add"),json!({"x":1})).with_namespace(Some("math".into())));
    let content=AssistantContent::ToolCall(call);
    let item=Item::Event(StreamEvent::End{part:part(),content:content.clone(),reasoning_authority:None});
    let events=assembly.ingest(&item).expect("end");
    let [StreamedTurnEvent::UndispatchableToolCall(refused)]=events.as_slice() else {panic!("tuple refusal");};
    assert!(matches!(&refused.call,rig_core::message::UndispatchableToolCall::Namespaced{namespace,..} if namespace=="math"));
    assert_eq!(refused.content,content);
    let content=AssistantContent::CustomToolCall(CustomToolCall::new(CallId::from_wire("custom-call"),name("add"),"{\"x\":1}"));
    let events=assembly.ingest(&Item::Event(StreamEvent::End{part:part(),content:content.clone(),reasoning_authority:None})).expect("custom end");
    let [StreamedTurnEvent::UndispatchableToolCall(refused)]=events.as_slice() else {panic!("tuple custom refusal");};
    assert!(matches!(&refused.call,rig_core::message::UndispatchableToolCall::Custom{name,..} if name=="add"));
    assert_eq!(refused.content,content);
    assert!(assembly.pending_tool_calls.is_empty());
}

#[test]
fn finalized_nonobject_raw_never_emits_a_canonical_pending_call() {
    use crate::run::policy::{DiagnosticArguments, InvalidToolCallReason};
    for raw in ["null"," 17 ","[1,null]"] {
        let mut assembly=assembler();
        assembly.ingest_for_driver(&start(None,ToolSlotState::PendingName)).expect("pending");
        assembly.ingest_for_driver(&arguments(raw)).expect("raw");
        assembly.ingest_for_driver(&start(Some("add"),ToolSlotState::Named)).expect("bind");
        // The shared canonical constructor can normalize null to an object.
        // The agent must judge the real retained arrivals before forwarding End.
        let events=assembly.ingest_for_driver(&end("add")).expect("diagnostic end");
        let [DriverTurnEvent::Public(StreamedTurnEvent::InvalidToolCall(invalid))]=events.as_slice() else {panic!("nonobject diagnostic only");};
        assert_eq!(invalid.tool_call.arguments,DiagnosticArguments::Parsed(serde_json::from_str(raw).expect("valid JSON")));
        assert_eq!(invalid.tool_call.raw.as_deref(),Some(raw));
        assert_eq!(invalid.tool_call.name,"add");
        assert_eq!(invalid.tool_call.id,CallId::from_wire("real-call"));
        assert!(matches!(&invalid.reason,InvalidToolCallReason::MalformedArguments{error} if error.starts_with("expected a JSON object")));
        assert!(assembly.pending_tool_calls.is_empty());
    }
}

#[test]
fn malformed_end_keeps_its_current_diagnostic_and_blank_object_end_stays_valid() {
    use crate::run::policy::DiagnosticArguments;
    let raw="{\"x\":";
    let mut assembly=assembler();
    assembly.ingest(&start(Some("add"),ToolSlotState::Named)).expect("allowed name");
    assembly.ingest(&arguments(raw)).expect("prefix");
    let content=AssistantContent::ToolCall(ToolCall::from_wire("real-call",ToolFunction::parse(name("add"),raw)));
    let events=assembly.ingest(&Item::Event(StreamEvent::End{part:part(),content,reasoning_authority:None})).expect("malformed");
    let [StreamedTurnEvent::InvalidToolCall(invalid)]=events.as_slice() else {panic!("existing malformed diagnostic");};
    assert!(matches!(&invalid.tool_call.arguments,DiagnosticArguments::Malformed{..}));
    assert_eq!(invalid.tool_call.raw.as_deref(),Some(raw));
    assert!(assembly.pending_tool_calls.is_empty());
    for raw in [""," \n\t"] {
        let mut assembly=assembler();
        assembly.ingest(&start(Some("add"),ToolSlotState::Named)).expect("allowed name");
        assembly.ingest(&arguments(raw)).expect("blank");
        assert!(matches!(assembly.ingest(&end("add")).expect("genuine object end").as_slice(),[StreamedTurnEvent::EmitToolCall{..}]));
        assert_eq!(assembly.pending_tool_calls.len(),1);
        assert_eq!(assembly.pending_tool_calls[0].function.arguments_value(),json!({"x":1}));
    }
}
