use super::*;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Num(u32);
impl ContextValue for Num {
    const KEY: &'static str = "test.num";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Greeting(String);
impl ContextValue for Greeting {
    const KEY: &'static str = "test.greeting";
}

#[test]
fn different_types_are_independent() {
    let mut c = ToolContext::new();
    c.insert(Num(42)).unwrap();
    c.insert(Greeting("hello".to_string())).unwrap();
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(42)));
    assert_eq!(
        c.get::<Greeting>().unwrap(),
        Some(Greeting("hello".to_string()))
    );
}
#[test]
fn empty_context_is_default_and_serializes_empty() {
    let c = ToolContext::default();
    assert!(!c.contains::<Num>());
    assert!(c.is_empty());
    assert_eq!(serde_json::to_value(&c).unwrap(), serde_json::json!({}));
}
#[test]
fn remove_missing_type_returns_none() {
    assert_eq!(ToolContext::new().remove::<Num>().unwrap(), None);
}
#[test]
fn require_missing_names_key() {
    let e = ToolContext::new().require::<Num>().unwrap_err();
    assert!(matches!(e, ToolContextError::Missing(Num::KEY)));
    assert!(e.to_string().contains("`test.num`"));
}
#[test]
fn result_metadata_round_trips_and_requires() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Id(u32);
    impl ContextValue for Id {
        const KEY: &'static str = "test.id";
    }
    let mut c = ToolContext::new();
    c.insert_result(Id(7)).unwrap();
    assert_eq!(c.result::<Id>().unwrap(), Some(Id(7)));
    assert_eq!(c.require_result::<Id>().unwrap(), Id(7));
    assert_eq!(c.get::<Id>().unwrap(), None);
}
#[test]
fn dispatch_snapshot_isolates_inbound_and_publishes_only_result_metadata() {
    let mut c = ToolContext::new();
    c.insert(Num(7)).unwrap();
    c.insert_result(Greeting("old".to_string())).unwrap();
    let mut d = c.for_dispatch();
    assert_eq!(d.get::<Num>().unwrap(), Some(Num(7)));
    assert_eq!(d.result::<Greeting>().unwrap(), None);
    d.insert(Num(8)).unwrap();
    d.insert_result(Greeting("new".to_string())).unwrap();

    c.accept_dispatch_result(d);
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(7)));
    assert_eq!(
        c.result::<Greeting>().unwrap(),
        Some(Greeting("new".to_string()))
    );
}
#[test]
fn many_distinct_types_round_trip() {
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct A(u8);
    impl ContextValue for A {
        const KEY: &'static str = "test.a";
    }
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct B(u16);
    impl ContextValue for B {
        const KEY: &'static str = "test.b";
    }
    let mut c = ToolContext::new();
    c.insert(A(1)).unwrap();
    c.insert(B(2)).unwrap();
    c.insert(Num(3)).unwrap();
    c.insert(Greeting("four".to_string())).unwrap();
    assert_eq!(c.get::<A>().unwrap(), Some(A(1)));
    assert_eq!(c.get::<B>().unwrap(), Some(B(2)));
    assert_eq!(c.get::<Num>().unwrap(), Some(Num(3)));
    assert_eq!(
        c.get::<Greeting>().unwrap(),
        Some(Greeting("four".to_string()))
    );
}
#[test]
fn unencodable_value_is_an_error_not_a_panic() {
    // A map with non-string keys has no JSON form.
    #[derive(Serialize, Deserialize, Debug)]
    struct TupleKeyed(std::collections::HashMap<(u8, u8), u8>);
    impl ContextValue for TupleKeyed {
        const KEY: &'static str = "test.tuple_keyed";
    }
    let mut c = ToolContext::new();
    let value = TupleKeyed([((1, 2), 3)].into_iter().collect());
    let err = c.insert(value).unwrap_err();
    assert!(matches!(
        err,
        ToolContextError::Encode {
            key: TupleKeyed::KEY,
            ..
        }
    ));
    assert!(!c.contains::<TupleKeyed>());
}

use crate::tool::{DynamicTool, ToolOutput};
use crate::wasm_compat::WasmBoxedFuture;
use std::sync::Arc;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct CallData(u32);

impl crate::tool::ContextValue for CallData {
    const KEY: &'static str = "test.dynamic.call_data";
}

fn mutate_context(
    context: &mut ToolContext,
    arguments: serde_json::Value,
) -> WasmBoxedFuture<'_, Result<ToolOutput, ToolExecutionError>> {
    Box::pin(async move {
        assert_eq!(context.require::<CallData>()?, CallData(1));
        context.insert(CallData(99))?;
        context.insert_result(CallData(7))?;
        if arguments["pending"] == true {
            return std::future::pending().await;
        }
        if arguments["fail"] == true {
            return Err(ToolExecutionError::other("tool failed"));
        }
        Ok(ToolOutput::text("done"))
    })
}

#[test]
fn cancelling_inline_execution_preserves_the_callers_context() {
    use futures::{FutureExt, task::noop_waker_ref};

    let tool = DynamicTool::new_with_context(
        crate::message::ToolName::new("mutate").expect("nonempty tool name"),
        "test context isolation",
        serde_json::json!({}),
        mutate_context,
    );
    let scope = Arc::new(42_u32);
    let mut context = ToolContext::new().with_scope(scope.clone());
    context.insert(CallData(1)).expect("inbound");
    context.insert_result(CallData(3)).expect("prior result");
    let before = context.clone();
    let mut run = Box::pin(tool.execute_with(&mut context, serde_json::json!({"pending": true})));
    let mut cx = std::task::Context::from_waker(noop_waker_ref());
    assert!(run.poll_unpin(&mut cx).is_pending());
    drop(run);
    assert_eq!(
        context, before,
        "cancellation must not consume caller data or publish a partial result"
    );
    assert!(Arc::ptr_eq(
        &context.scope::<u32>().expect("scope retained"),
        &scope
    ));
}
