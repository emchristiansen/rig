use std::sync::Mutex;

use super::*;
use crate::completion::CompletionResponse;

/// A recorder overriding only the required methods, naming each call.
#[derive(Default)]
struct Calls(Mutex<Vec<&'static str>>);

impl Calls {
    fn push(&self, call: &'static str) {
        self.0.lock().expect("calls lock").push(call);
    }

    fn taken(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.0.lock().expect("calls lock"))
    }
}

impl Recorder for Calls {
    fn handlers(&self, _handlers: Vec<HandlerDescriptor>) {
        self.push("handlers");
    }
    fn begin(&self, _id: EffectId, _key: HandlerKey, _kind: EffectKind, _origin: Origin) {
        self.push("begin");
    }
    fn discard(&self, _id: EffectId) {
        self.push("discard");
    }
    fn patch(&self, _id: EffectId, _kind: EffectKind) {
        self.push("patch");
    }
    fn keep_events(&self) -> bool {
        true
    }
    fn event(&self, _id: EffectId, _item: &Item<StreamEvent>) {
        self.push("event");
    }
    fn origin(&self, _id: EffectId, _origin: &crate::message::Origin) {
        self.push("origin");
    }
    fn tool_output(&self, _id: EffectId, _output: crate::tool::ToolResultContext) {
        self.push("tool_output");
    }
    fn resolve(&self, _id: EffectId, _outcome: Result<Outcome, ErrorReport>) {
        self.push("resolve");
    }
}

/// A recorder that does not keep streamed responses ignores each one: the
/// default records nothing and calls no other method.
#[test]
fn the_default_stream_done_records_nothing() {
    let calls = Calls::default();
    let recorder: &dyn Recorder = &calls;
    let response = CompletionResponse::new(
        vec![crate::message::AssistantContent::text("done")],
        Default::default(),
        crate::message::Origin::new("test.api", "local", ""),
        serde_json::Value::Null,
    );
    recorder.stream_done(EffectId::from_raw(7), &response);
    assert_eq!(calls.taken(), Vec::<&'static str>::new());
}
