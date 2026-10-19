use super::streaming::usage_of;
use serde_json::{Value, json};

fn recorded() -> Value {
    json!({
        "total_input_tokens": 14, "total_output_tokens": 34,
        "total_tokens": 270, "total_cached_tokens": 0,
        "total_thought_tokens": 222, "total_tool_use_tokens": 0,
    })
}

#[test]
fn thinking_tokens_survive_the_interactions_mapping() {
    let usage = usage_of(&recorded());
    assert_eq!(usage.input_tokens, Some(14));
    assert_eq!(usage.output_tokens, Some(34));
    assert_eq!(usage.total_tokens, Some(270));
    assert_eq!(usage.reasoning_tokens, Some(222));
    assert_eq!(
        usage.total_tokens,
        Some(14 + 34 + 222),
        "the components should account for the provider's total"
    );
}

#[test]
fn cached_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire["total_cached_tokens"] = json!(9_000);
    assert_eq!(usage_of(&wire).cached_input_tokens, Some(9_000));
}

#[test]
fn tool_use_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire["total_tool_use_tokens"] = json!(77);
    assert_eq!(usage_of(&wire).tool_use_prompt_tokens, Some(77));
}

#[test]
fn the_total_fallback_counts_thinking_and_tool_use() {
    let mut wire = recorded();
    wire.as_object_mut()
        .expect("usage object")
        .remove("total_tokens");
    wire["total_tool_use_tokens"] = json!(5);
    assert_eq!(usage_of(&wire).total_tokens, Some(14 + 34 + 222 + 5));
}

#[test]
fn the_older_three_field_shape_still_maps() {
    let usage =
        usage_of(&json!({"total_input_tokens": 3, "total_output_tokens": 4, "total_tokens": 7}));
    assert_eq!(usage.total_tokens, Some(7));
    assert_eq!(usage.reasoning_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
}

#[test]
fn a_reported_total_wins_even_without_both_base_counts() {
    let usage = usage_of(&json!({"total_input_tokens": 3, "total_tokens": 90}));
    assert_eq!(usage.input_tokens, Some(3));
    assert_eq!(usage.output_tokens, None);
    assert_eq!(usage.total_tokens, Some(90));
}

#[test]
fn total_fallback_requires_both_base_counts() {
    let usage = usage_of(&json!({"total_input_tokens": 3, "total_thought_tokens": 5}));
    assert_eq!(usage.total_tokens, None);
}
