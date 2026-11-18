//! Requests whose bytes on the Codex contract are pinned, for the HTTP and
//! WebSocket tests alike.

pub(crate) use rig_core::completion::history::TOOL_IMAGE_OMITTED;
use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{
    CallId, DocumentSourceKind, Image, ImageMediaType, Message, ProviderCallId, ToolName,
    ToolResultContent, UserContent,
};
use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::wire::{Mode, WireFrame, decode_reply};
use serde_json::value::RawValue;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A Responses `functions` namespace as a caller declares it: empty
/// descriptions and an explicit `strict: false`.
pub(crate) fn namespace() -> Value {
    json!({
        "type": "namespace",
        "name": "functions",
        "description": "",
        "tools": [{
            "type": "function",
            "name": "exec_command",
            "description": "",
            "strict": false,
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
        }]
    })
}

/// A function as a caller declares it, with members Rig's tool type does
/// not model.
pub(crate) fn declared_function() -> Value {
    json!({
        "type": "function",
        "name": "post_to_bus",
        "description": "",
        "strict": false,
        "defer_loading": false,
        "x_vendor_hint": null,
        "parameters": {"type": "object", "properties": {"body": {"type": "string"}}}
    })
}

/// A hosted tool as a caller declares it.
pub(crate) fn declared_hosted() -> Value {
    json!({"type": "web_search", "search_context_size": "low"})
}

/// A typed tool, which Rig writes ahead of the declared ones.
pub(crate) fn lookup_tool() -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new("lookup").expect("a tool name"),
        "Look something up.",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )
}

/// One user turn.
pub(crate) fn prompt() -> CompletionRequest {
    CompletionRequest::new(Message::user("say hi"))
}

/// One user turn with the typed `lookup` tool and `declared` as its
/// declared tools.
pub(crate) fn turn_declaring(declared: Value) -> CompletionRequest {
    let mut request = prompt();
    request.tools = vec![Arc::new(lookup_tool())];
    request.additional_params = Some(json!({ "tools": declared }));
    request
}

/// The request Muninn's Codex lane sends: instructions, one user turn,
/// reasoning with its ciphertext, and the conversation's cache key and
/// metadata stated by the caller.
pub(crate) fn muninn_shaped_request() -> CompletionRequest {
    let mut request = prompt();
    request.chat_history = vec![
        Arc::new(Message::system("Be brief.")),
        Arc::new(Message::user("say hi")),
    ];
    request.additional_params = Some(json!({
        "prompt_cache_key": "thread-7",
        "client_metadata": {"session_id": "session-7", "thread_id": "thread-7"},
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "store": false,
    }));
    request
}

/// Whether this build's `serde_json` keeps an object's keys in insertion
/// order (`preserve_order`) or sorts them.
pub(crate) fn preserves_order() -> bool {
    serde_json::to_string(&json!({ "b": 0, "a": 0 })).expect("serializes") == r#"{"b":0,"a":0}"#
}

/// The top-level fields of the JSON object `body`, each as the bytes it was
/// written with.
pub(crate) fn raw_fields(body: &str) -> BTreeMap<String, Box<RawValue>> {
    serde_json::from_str(body).expect("the body is a JSON object")
}

/// The bytes `value` is sent as: its serialization in this build.
pub(crate) fn written(value: &Value) -> String {
    serde_json::to_string(value).expect("serializes")
}

/// The whole body of the request Muninn's Codex lane sends, in a build whose
/// `serde_json` sorts keys (Muninn's): every field is the caller's, `input`
/// is last, and nothing else is stated.
pub(crate) const MUNINN_SHAPED_BODY_SORTED: &str = concat!(
    r#"{"client_metadata":{"session_id":"session-7","thread_id":"thread-7"},"#,
    r#""include":["reasoning.encrypted_content"],"instructions":"Be brief.","#,
    r#""model":"gpt-5.4","prompt_cache_key":"thread-7","reasoning":{"effort":"high"},"#,
    r#""store":false,"stream":true,"#,
    r#""input":[{"content":[{"text":"say hi","type":"input_text"}],"role":"user","type":"message"}]}"#,
);

/// The custom tool `apply_patch` as a caller declares it.
pub(crate) fn declared_custom() -> Value {
    json!({"type": "custom", "name": "apply_patch", "description": "", "format": {"type": "text"}})
}

/// The provider's item for the custom call `call_c` to `apply_patch`.
pub(crate) fn custom_call_item() -> Value {
    json!({
        "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c",
        "name": "apply_patch", "input": "*** Begin Patch", "status": "completed"
    })
}

/// The provider's item for the function call `call_f` to `lookup`.
pub(crate) fn function_call_item() -> Value {
    json!({
        "type": "function_call", "id": "fc_1", "call_id": "call_f",
        "name": "lookup", "arguments": "{\"q\":\"rig\"}", "status": "completed"
    })
}

/// A request that continues the turn in which `gpt-5.4` made the call
/// `item`: one user message, the turn as Rig folds the provider's stream,
/// and the call's answer `content`. It declares the typed `lookup` tool and
/// the custom `apply_patch`.
pub(crate) fn answering(item: Value, content: Vec<ToolResultContent>) -> CompletionRequest {
    let response = |status: &str, output: Value| {
        json!({
            "id": "resp_1", "object": "response", "created_at": 0, "status": status,
            "model": "gpt-5.4", "output": output,
        })
    };
    let frames = [
        json!({"type": "response.created", "response": response("in_progress", json!([]))}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": item}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        json!({"type": "response.completed", "response": response("completed", json!([item]))}),
    ]
    .into_iter()
    .enumerate()
    .map(|(sequence, mut event)| {
        event["sequence_number"] = json!(sequence);
        WireFrame::Text(event.to_string())
    });
    let turn = decode_reply(
        &Responses::new(
            OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
            "gpt-5.4",
        ),
        &prompt(),
        Mode::Streaming,
        frames,
    )
    .expect("the turn folds");
    let [call] = turn.tool_calls().collect::<Vec<_>>()[..] else {
        panic!("the turn makes one call: {:?}", turn.choice);
    };

    let mut request = turn_declaring(json!([declared_custom()]));
    request.chat_history.extend([
        Arc::new(turn.message().expect("the turn's message")),
        Arc::new(Message::User {
            content: vec![UserContent::tool_result(
                call.id.clone(),
                call.function.name.clone(),
                content,
            )],
        }),
    ]);
    request
}

/// A request that answers the custom call `call_c` with `content` and
/// carries nothing else: the appended history of an incremental turn on
/// stored state. It declares the custom `apply_patch`.
pub(crate) fn answering_stored_custom_call(content: Vec<ToolResultContent>) -> CompletionRequest {
    let mut request = CompletionRequest::new(Message::User {
        content: vec![UserContent::tool_result(
            CallId::from(ProviderCallId::new("call_c").expect("a call id")),
            ToolName::new("apply_patch").expect("a tool name"),
            content,
        )],
    });
    request.additional_params = Some(json!({
        "previous_response_id": "resp_1",
        "tools": [declared_custom()],
    }));
    request
}

fn image(data: DocumentSourceKind, media_type: Option<ImageMediaType>) -> Image {
    Image {
        data,
        media_type,
        detail: None,
        native: None,
    }
}

/// The images Rig v0.46.0 sends to `gpt-5.4` on the Codex contract as a
/// part of a tool result: a URL and typed base64 data.
pub(crate) fn images_rig_sends_in_a_result() -> [Image; 2] {
    [
        image(
            DocumentSourceKind::Url("https://example.test/diff.png".to_owned()),
            None,
        ),
        image(
            DocumentSourceKind::Base64("aGk=".to_owned()),
            Some(ImageMediaType::PNG),
        ),
    ]
}

/// An image Rig v0.46.0 does not send on the Codex contract: a file id.
/// Rig's preparation of a request replaces it with
/// [`TOOL_IMAGE_OMITTED`] before the wire encodes the request.
pub(crate) fn image_rig_omits() -> Image {
    image(DocumentSourceKind::FileId("file_1".to_owned()), None)
}

/// The items of `body`'s `input`, each as the bytes it was written with.
pub(crate) fn raw_input(body: &str) -> Vec<String> {
    let items: Vec<Box<RawValue>> =
        serde_json::from_str(raw_fields(body)["input"].get()).expect("`input` is an array");
    items.iter().map(|item| item.get().to_owned()).collect()
}

/// The refusal of an image in a custom call's answer.
pub(crate) const CUSTOM_IMAGE_REFUSAL: &str = "a custom tool result carries text, not an image";

/// `request` with `section` as its provider options for the ChatGPT
/// Responses wire, read as a caller's serialized options are.
pub(crate) fn with_provider_options(
    mut request: CompletionRequest,
    section: Value,
) -> CompletionRequest {
    request.provider_options =
        serde_json::from_value(json!({ "chatgpt": { "openai.responses": section } }))
            .expect("provider options deserialize");
    request
}

/// `request` with `params` as its `additional_params`.
pub(crate) fn with_raw(mut request: CompletionRequest, params: Value) -> CompletionRequest {
    request.additional_params = Some(params);
    request
}

/// Every way a request can state a control the Codex contract does not
/// take, each with the name the refusal must carry: the typed field, the
/// generation options, the provider options and `additional_params`.
pub(crate) fn requests_stating_a_refused_control() -> Vec<(&'static str, CompletionRequest)> {
    let mut typed_temperature = prompt();
    typed_temperature.temperature = Some(0.4);
    let mut typed_cap = prompt();
    typed_cap.max_tokens = Some(17);
    let mut option_top_p = prompt();
    option_top_p.options.top_p = Some(0.5);
    vec![
        ("temperature", typed_temperature),
        (
            "temperature",
            with_raw(prompt(), json!({"temperature": 0.4})),
        ),
        (
            "temperature",
            with_provider_options(prompt(), json!({"temperature": 0.4})),
        ),
        ("max_output_tokens", typed_cap),
        (
            "max_output_tokens",
            with_raw(prompt(), json!({"max_output_tokens": 17})),
        ),
        (
            "max_output_tokens",
            with_provider_options(prompt(), json!({"max_output_tokens": 17})),
        ),
        ("top_p", option_top_p),
        ("top_p", with_raw(prompt(), json!({"top_p": 0.5}))),
        (
            "top_p",
            with_provider_options(prompt(), json!({"top_p": 0.5})),
        ),
        ("store", with_raw(prompt(), json!({"store": true}))),
        (
            "store",
            with_provider_options(prompt(), json!({"store": true})),
        ),
        // The later layer wins: a raw `true` over the provider options'
        // `false` is a caller-set `store: true`.
        (
            "store",
            with_raw(
                with_provider_options(prompt(), json!({"store": false})),
                json!({"store": true}),
            ),
        ),
    ]
}

/// Requests whose later layer takes a refused control back out, which are
/// sent: `additional_params` over the provider options.
pub(crate) fn requests_overriding_a_refused_control() -> Vec<CompletionRequest> {
    vec![
        with_raw(
            with_provider_options(prompt(), json!({"store": true})),
            json!({"store": false}),
        ),
        with_raw(
            with_provider_options(prompt(), json!({"temperature": 0.4})),
            json!({"temperature": null}),
        ),
        with_raw(prompt(), json!({"store": false, "top_p": null})),
    ]
}

/// Requests and the `include` each must be sent with: the request's own
/// entries from whichever layer states them, the reasoning ciphertext added
/// only when the request reasons or asked for it.
pub(crate) fn requests_and_their_include() -> Vec<(CompletionRequest, Option<Value>)> {
    let own = json!(["file_search_call.results"]);
    let ciphertext = "reasoning.encrypted_content";
    vec![
        (prompt(), None),
        (
            with_provider_options(prompt(), json!({ "include": own })),
            Some(own.clone()),
        ),
        (
            with_raw(prompt(), json!({ "include": own })),
            Some(own.clone()),
        ),
        // `additional_params` over the provider options.
        (
            with_raw(
                with_provider_options(prompt(), json!({"include": ["web_search_call.results"]})),
                json!({ "include": own }),
            ),
            Some(own.clone()),
        ),
        (
            with_provider_options(prompt(), json!({"include": [ciphertext]})),
            Some(json!([ciphertext])),
        ),
        (
            with_provider_options(
                prompt(),
                json!({"include": own, "reasoning": {"effort": "low"}}),
            ),
            Some(json!(["file_search_call.results", ciphertext])),
        ),
        (
            with_raw(
                with_provider_options(prompt(), json!({ "include": own })),
                json!({"reasoning": {"effort": "low"}}),
            ),
            Some(json!(["file_search_call.results", ciphertext])),
        ),
    ]
}
