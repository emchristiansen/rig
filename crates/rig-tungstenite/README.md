# rig-tungstenite

The native `tokio-tungstenite` backend for Rig Live control sockets. It implements `rig_core::ws_client::WebSocketClientExt`; the protocol lives in `rig_core::providers::chatgpt::realtime`. Open a control socket with `LiveCalls::connect_control(&TungsteniteClient::new(), call_id)`.

The backend uses the caller's Tokio runtime when available. With another executor, such as `futures::executor`, it moves socket I/O onto a lazy fallback Tokio runtime and communicates through channels. Dropping a connection aborts its actor, including a blocked write. The actor selects between inbound frames and commands so a cancelled idle receive does not prevent a later close. A failed reply send restores the frame or error to the queue; cancellation after a successful reply enqueue can still lose that result. Read-ahead is bounded and polling drives automatic pong replies.

The `rustls` feature is enabled by default; `native-tls` selects the other TLS backend. When using `default-features = false`, select a TLS feature for `wss` connections. This crate is native-only and does not provide canonical Responses websocket session constructors.

Rejected WebSocket upgrades retain their status, parsed headers and a text
rendering of the body bytes tungstenite buffered while reading those headers.
That body may be empty or partial, and invalid UTF-8 uses the inherited lossy
conversion. The transport does not read the remaining rejection body.
