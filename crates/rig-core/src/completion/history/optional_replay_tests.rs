//! Optional call identity preserves native replay without inventing handles.

use std::borrow::Cow;

use serde_json::{Map, Value, json};

use super::{Accepts, OptionalCallId, Replay, ReplayTarget};
use crate::message::{
    Api, AssistantContent, CallId, LocalCallId, Opaque, Origin, ProviderCallId, ToolCall,
    ToolFunction, ToolName,
};

#[derive(Debug)]
struct Target(&'static str);

impl ReplayTarget for Target {
    fn api(&self) -> Api {
        Api::from_static("optional.calls")
    }

    fn provider(&self) -> &str {
        "optional-provider"
    }

    fn model(&self) -> &str {
        "model"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some(self.0)
    }

    fn identity(&self, item: &Value) -> Map<String, Value> {
        item.as_object()
            .into_iter()
            .flatten()
            .filter(|(key, _)| {
                matches!(
                    key.as_str(),
                    "id" | "call_id" | "output_id" | "routing" | "a/b" | "m~n" | "~1" | "~2" | ""
                )
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}

fn optional_id(id: &CallId) -> OptionalCallId<'_> {
    match id {
        CallId::Provider(handle) => OptionalCallId::ProviderHandle(handle),
        CallId::Local(_) => OptionalCallId::OmitLocal,
    }
}

fn local() -> CallId {
    CallId::Local(LocalCallId::for_ollama_response_position(3))
}

fn call(id: CallId, native: Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(
        id,
        ToolFunction::new(
            ToolName::new("lookup").expect("fixture name"),
            json!({"x":1}),
        ),
    ))
    .with_native(native)
}

fn call_id(block: &AssistantContent) -> &CallId {
    let AssistantContent::ToolCall(call) = block else {
        panic!("the fixture is a function call");
    };
    &call.id
}

#[test]
fn current_native_without_a_handle_stays_whole_under_explicit_local_omission() {
    let native = json!({"type":"function","function":{"name":"lookup","arguments":{"x":1}},
        "future":{"exact":[null,true,{"spacing":" \nkept\t "}]}, "output_id":"item-only"});
    let block = call(local(), native.clone());
    let fingerprint = block.fingerprint();
    assert_eq!(block.native_item(), Some(&native));
    assert_eq!(
        block.replay_optional_call_id(&Target("/id"), optional_id(call_id(&block))),
        Replay::Item(Cow::Owned(native.clone()))
    );
    assert_eq!(
        block.native_item(),
        Some(&native),
        "replay never edits the stored Native"
    );
    assert_eq!(block.fingerprint(), fingerprint);
    let mut required = native;
    required["id"] = json!("tool-0");
    assert_eq!(
        block.replay(&Target("/id"), Some("tool-0")),
        Replay::Item(Cow::Owned(required)),
        "required-ID replay still fills its planned spelling"
    );
}

#[test]
fn a_local_current_native_omits_only_the_optional_handle_and_keeps_every_sibling() {
    let native = json!({"id":"leftover-provider-field", "call_id":"separate-correlation",
        "output_id":"output-item", "function":{"name":"lookup","arguments":{"x":1}},
        "future":{"nested":[1,null,{"id":"nested-id-kept"}]}});
    let block = call(local(), native.clone());
    let expected = json!({"call_id":"separate-correlation", "output_id":"output-item",
        "function":{"name":"lookup","arguments":{"x":1}},
        "future":{"nested":[1,null,{"id":"nested-id-kept"}]}});
    let replay = block.replay_optional_call_id(&Target("/id"), optional_id(call_id(&block)));
    assert_eq!(replay, Replay::Item(Cow::Owned(expected.clone())));
    let Replay::Item(item) = replay else {
        panic!("the current Native remains a whole item");
    };
    assert_eq!(
        serde_json::to_vec(item.as_ref()).expect("serialize optional replay"),
        serde_json::to_vec(&expected).expect("serialize literal expected item")
    );
    assert_eq!(block.native_item(), Some(&native));
    assert_eq!(
        block.replay(&Target("/id"), None),
        Replay::Item(Cow::Owned(native))
    );
}

#[test]
fn local_stale_identity_loses_only_the_optional_slot_and_never_prunes_siblings() {
    for (slot, native, expected) in [
        (
            "/id",
            json!({"id":"old-handle","output_id":"item-only","call_id":"other-slot",
            "function":{"name":"lookup","arguments":{"x":1}}}),
            json!({"output_id":"item-only","call_id":"other-slot"}),
        ),
        (
            "/routing/call_id",
            json!({"id":"output-id","routing":{"call_id":"old-handle",
            "other":{"id":"nested-output","exact":[true,null]}}}),
            json!({"id":"output-id","routing":{"other":{"id":"nested-output","exact":[true,null]}}}),
        ),
        (
            "/routing/id",
            json!({"routing":{"id":"old-handle"}}),
            json!({"routing":{}}),
        ),
    ] {
        let mut block = call(local(), native.clone());
        let AssistantContent::ToolCall(call) = &mut block else {
            panic!("function fixture");
        };
        call.function.arguments.insert("x".to_owned(), json!(2));
        assert!(
            block.native_item().is_none(),
            "canonical edit stales the original Native"
        );
        assert_eq!(block.stale_item(), Some(&native));
        let target = Target(slot);
        let expected_identity = expected
            .as_object()
            .expect("literal identity object")
            .clone();
        assert_eq!(
            block.replay_optional_call_id(&target, optional_id(call_id(&block))),
            Replay::Identity(expected_identity)
        );
        assert_eq!(
            block.replay(&target, None),
            Replay::Identity(target.identity(&native)),
            "required stale identity retains all original keys"
        );
        assert_eq!(block.stale_item(), Some(&native));
    }
    let mut block = call(local(), json!({"id":"only-identity"}));
    let AssistantContent::ToolCall(call) = &mut block else {
        panic!("function fixture");
    };
    call.function
        .arguments
        .insert("edited".to_owned(), json!(true));
    assert_eq!(
        block.replay_optional_call_id(&Target("/id"), optional_id(call_id(&block))),
        Replay::Rebuild,
        "removing the sole stale identity leaves canonical rebuild"
    );
}

#[test]
fn a_provider_handle_replays_verbatim_without_rewriting_output_item_identity() {
    let handle = ProviderCallId::new(" \tprovider|handle/0\n ").expect("nonempty exact handle");
    let native = json!({"id":"output-item-only","call_id":"old-correlation",
        "function":{"name":"lookup","arguments":{"x":1}}, "future":{"keep":[true,null]}});
    let block = call(CallId::Provider(handle.clone()), native.clone());
    let mut expected = native.clone();
    expected["call_id"] = json!(handle.as_str());
    let target = Target("/call_id");
    assert_eq!(
        block.replay_optional_call_id(&target, optional_id(call_id(&block))),
        Replay::Item(Cow::Owned(expected))
    );
    assert_eq!(block.native_item(), Some(&native));
    let mut edited = block;
    let AssistantContent::ToolCall(call) = &mut edited else {
        panic!("function fixture");
    };
    call.function.arguments.insert("x".to_owned(), json!(2));
    assert_eq!(
        edited.replay_optional_call_id(&target, optional_id(call_id(&edited))),
        edited.replay(&target, Some(handle.as_str())),
        "provider stale identity behavior is exactly the required replay behavior"
    );
}

#[test]
fn optional_omission_does_not_reclassify_text_opaque_or_origin() {
    let native = json!({"id":"text-item","text":"answer","future":[null,true]});
    let text = AssistantContent::text("answer").with_native(native.clone());
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({"id":"opaque-item","type":"future_item"}),
        replay: true,
    });
    let origin = Origin::new("optional.calls", "optional-provider", "model");
    let history = crate::message::AssistantMessage::new(vec![text.clone(), opaque.clone()])
        .with_origin(origin.clone());
    assert_eq!(
        text.replay_optional_call_id(&Target("/id"), OptionalCallId::OmitLocal),
        text.replay(&Target("/id"), None)
    );
    assert_eq!(text.native_item(), Some(&native));
    assert_eq!(
        opaque.replay_optional_call_id(&Target("/id"), OptionalCallId::OmitLocal),
        Replay::Rebuild
    );
    assert_eq!(history.origin.as_ref(), Some(&origin));
    assert_eq!(history.content, [text, opaque]);
}

#[test]
fn stale_optional_identity_removes_only_the_addressed_field_and_keeps_surviving_order() {
    for (slot, native, expected) in [
        (
            "/call_id",
            json!({"id":"before","call_id":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/routing/remove",
            json!({"id":"root","routing":{"first":1,"remove":2,"last":3}}),
            json!({"id":"root","routing":{"first":1,"last":3}}),
        ),
        (
            "/a~1b",
            json!({"id":"before","a/b":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/m~0n",
            json!({"id":"before","m~n":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/~01",
            json!({"id":"before","~1":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/~2",
            json!({"id":"before","~2":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/a~1b/id",
            json!({"id":"root","a/b":{"first":1,"id":"remove","last":3}}),
            json!({"id":"root","a/b":{"first":1,"last":3}}),
        ),
        (
            "/m~0n/id",
            json!({"id":"root","m~n":{"first":1,"id":"remove","last":3}}),
            json!({"id":"root","m~n":{"first":1,"last":3}}),
        ),
        (
            "/~01/id",
            json!({"id":"root","~1":{"first":1,"id":"remove","last":3}}),
            json!({"id":"root","~1":{"first":1,"last":3}}),
        ),
        (
            "/~2/id",
            json!({"id":"root","~2":{"first":1,"id":"remove","last":3}}),
            json!({"id":"root","~2":{"first":1,"last":3}}),
        ),
        (
            "/routing/a~1b/m~0n/~01/~2/id",
            json!({"id":"root","routing":{"a/b":{"m~n":{"~1":{"~2":{"id":"remove","keep":true}}}}}}),
            json!({"id":"root","routing":{"a/b":{"m~n":{"~1":{"~2":{"keep":true}}}}}}),
        ),
        (
            "/",
            json!({"id":"before","":"remove","output_id":"after"}),
            json!({"id":"before","output_id":"after"}),
        ),
        (
            "/routing/",
            json!({"id":"root","routing":{"first":1,"":"remove","last":3}}),
            json!({"id":"root","routing":{"first":1,"last":3}}),
        ),
        (
            "//id",
            json!({"id":"root","":{"first":1,"id":"remove","last":3}}),
            json!({"id":"root","":{"first":1,"last":3}}),
        ),
        (
            "///id",
            json!({"id":"root","":{"":{"id":"remove"}}}),
            json!({"id":"root","":{"":{}}}),
        ),
        (
            "/routing/branch/id",
            json!({"id":"root","routing":{"branch":{"id":"remove"}}}),
            json!({"id":"root","routing":{"branch":{}}}),
        ),
        (
            "/routing/1/id",
            json!({"id":"root","routing":[{"id":"first"},{"id":"remove"},{"id":"last"}]}),
            json!({"id":"root","routing":[{"id":"first"},{},{"id":"last"}]}),
        ),
        (
            "/routing/00/id",
            json!({"id":"root","routing":{"00":{"id":"remove","keep":true}}}),
            json!({"id":"root","routing":{"00":{"keep":true}}}),
        ),
    ] {
        let mut block = call(local(), native.clone());
        let AssistantContent::ToolCall(call) = &mut block else {
            panic!("function fixture");
        };
        call.function.arguments.insert("x".to_owned(), json!(2));
        assert!(block.native_item().is_none(), "{slot}");
        assert_eq!(block.stale_item(), Some(&native), "{slot}");
        let target = Target(slot);
        let replay = block.replay_optional_call_id(&target, optional_id(call_id(&block)));
        let Replay::Identity(identity) = replay else {
            panic!("the surviving identity stays an object: {slot}");
        };
        assert_eq!(
            &identity,
            expected.as_object().expect("literal expected identity"),
            "{slot}"
        );
        assert_eq!(
            serde_json::to_vec(&identity).expect("serialize surviving identity"),
            serde_json::to_vec(&expected).expect("serialize literal expected identity"),
            "{slot}"
        );
        assert_eq!(block.stale_item(), Some(&native), "{slot}");
        assert_eq!(
            block.replay(&target, None),
            Replay::Identity(target.identity(&native)),
            "required replay keeps the original identity: {slot}"
        );
    }
}

#[test]
fn stale_optional_identity_keeps_unresolved_parents_and_never_removes_array_elements() {
    for (slot, native) in [
        ("", json!({"id":"keep"})),
        ("id", json!({"id":"keep"})),
        ("routing/id", json!({"id":"root","routing":{"id":"keep"}})),
        ("#/routing/id", json!({"id":"root","routing":{"id":"keep"}})),
        ("/missing/id", json!({"id":"root","routing":{"id":"keep"}})),
        (
            "/routing/missing/id",
            json!({"id":"root","routing":{"id":"keep"}}),
        ),
        (
            "/routing/missing",
            json!({"id":"root","routing":{"id":"keep"}}),
        ),
        ("/missing", json!({"id":"root","routing":{"id":"keep"}})),
        ("/routing/id", json!({"id":"root","routing":null})),
        ("/routing/id", json!({"id":"root","routing":7})),
        ("/routing/id", json!({"id":"root","routing":"keep"})),
        ("/routing/0", json!({"id":"root","routing":[{"id":"keep"}]})),
        (
            "/routing/00/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        (
            "/routing/01/id",
            json!({"id":"root","routing":[{"id":"keep"},{"id":"also kept"}]}),
        ),
        (
            "/routing/+0/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        (
            "/routing/-1/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        (
            "/routing/-/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        (
            "/routing/9/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        (
            "/routing/184467440737095516160/id",
            json!({"id":"root","routing":[{"id":"keep"}]}),
        ),
        ("/routing/0/id", json!({"id":"root","routing":[]})),
        ("/routing/0/id", json!({"id":"root","routing":[null]})),
        ("/routing/0/id", json!({"id":"root","routing":["keep"]})),
    ] {
        let mut block = call(local(), native.clone());
        let AssistantContent::ToolCall(call) = &mut block else {
            panic!("function fixture");
        };
        call.function.arguments.insert("x".to_owned(), json!(2));
        assert!(block.native_item().is_none(), "{slot}");
        assert_eq!(block.stale_item(), Some(&native), "{slot}");
        let target = Target(slot);
        let replay = block.replay_optional_call_id(&target, optional_id(call_id(&block)));
        let Replay::Identity(identity) = replay else {
            panic!("the unchanged nonempty identity remains: {slot}");
        };
        assert_eq!(
            &identity,
            native.as_object().expect("literal original identity"),
            "{slot}"
        );
        assert_eq!(
            serde_json::to_vec(&identity).expect("serialize unchanged identity"),
            serde_json::to_vec(&native).expect("serialize literal original identity"),
            "{slot}"
        );
        assert_eq!(block.stale_item(), Some(&native), "{slot}");
        assert_eq!(
            block.replay(&target, None),
            Replay::Identity(target.identity(&native)),
            "required replay keeps the original identity: {slot}"
        );
    }
}
