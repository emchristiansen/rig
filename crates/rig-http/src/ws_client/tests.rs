use super::*;

#[test]
fn websocket_url_upgrades_the_scheme_and_appends_the_path() {
    assert_eq!(
        websocket_url("https://api.openai.com/v1", "responses").expect("https upgrades"),
        "wss://api.openai.com/v1/responses"
    );
    assert_eq!(
        websocket_url("http://127.0.0.1:8080/v1", "responses").expect("http upgrades"),
        "ws://127.0.0.1:8080/v1/responses"
    );
}

/// A base URL that already names the websocket scheme is left on it rather
/// than rejected: hosts that configure `wss://` directly are not wrong.
#[test]
fn websocket_url_accepts_an_already_websocket_scheme() {
    assert_eq!(
        websocket_url("wss://api.openai.com/v1", "responses").expect("wss stays wss"),
        "wss://api.openai.com/v1/responses"
    );
}

#[test]
fn websocket_url_trims_a_trailing_slash() {
    assert_eq!(
        websocket_url("https://api.openai.com/v1/", "responses").expect("trailing slash"),
        "wss://api.openai.com/v1/responses"
    );
}

#[test]
fn websocket_url_rejects_an_unsupported_scheme() {
    let error = websocket_url("ftp://api.openai.com/v1", "responses")
        .expect_err("ftp is not a websocket base");
    assert!(
        error.to_string().contains("ftp"),
        "the error should name the scheme, got {error}"
    );
}

struct ConnectionWithoutCapabilities;

impl WebSocketConnection for ConnectionWithoutCapabilities {
    fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(std::future::ready(Ok(None)))
    }

    fn close(&mut self, _frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

async fn assert_unsupported_capabilities(mut connection: impl WebSocketConnection) {
    let ready_error = connection
        .recv_ready()
        .await
        .expect_err("an omitted ready-read capability must refuse");
    match &ready_error {
        Error::Instance(inner) => assert_eq!(
            inner.downcast_ref::<UnsupportedCapability>(),
            Some(&UnsupportedCapability {
                capability: "recv_ready",
            })
        ),
        other => panic!("expected a typed ready-read refusal, got {other:?}"),
    }
    assert!(UnsupportedCapability::is(&ready_error, "recv_ready"));
    assert!(!UnsupportedCapability::is(&ready_error, "flush"));

    let flush_error = connection
        .flush()
        .await
        .expect_err("an omitted flush capability must refuse");
    match &flush_error {
        Error::Instance(inner) => assert_eq!(
            inner.downcast_ref::<UnsupportedCapability>(),
            Some(&UnsupportedCapability {
                capability: "flush",
            })
        ),
        other => panic!("expected a typed flush refusal, got {other:?}"),
    }
    assert!(UnsupportedCapability::is(&flush_error, "flush"));
    assert!(!UnsupportedCapability::is(&flush_error, "recv_ready"));
}

#[tokio::test]
async fn omitted_capabilities_use_the_actual_typed_defaults() {
    assert_unsupported_capabilities(ConnectionWithoutCapabilities).await;
}

#[tokio::test]
async fn boxed_omitted_capabilities_preserve_the_actual_typed_defaults() {
    let connection: BoxedWebSocketConnection = Box::new(ConnectionWithoutCapabilities);
    assert_unsupported_capabilities(connection).await;
}

struct ConnectionWithCapabilities {
    ready_results: std::vec::IntoIter<Result<ReadyFrame>>,
    flush_results: std::vec::IntoIter<Result<()>>,
}

impl WebSocketConnection for ConnectionWithCapabilities {
    fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(std::future::ready(Ok(None)))
    }

    fn close(&mut self, _frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv_ready(&mut self) -> WasmBoxedFuture<'_, Result<ReadyFrame>> {
        Box::pin(std::future::ready(self.ready_results.next().expect(
            "every ready read must consume its override's next result",
        )))
    }

    fn flush(&mut self) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(std::future::ready(
            self.flush_results
                .next()
                .expect("every flush must consume its override's next result"),
        ))
    }
}

#[tokio::test]
async fn boxed_connection_forwards_every_ready_outcome_and_errors() {
    let outcomes = [
        ReadyFrame::Frame(Frame::Text("arrived".to_string())),
        ReadyFrame::Empty,
        ReadyFrame::Ended,
    ];
    let mut ready_results: Vec<_> = outcomes.iter().cloned().map(Ok).collect();
    ready_results.push(Err(Error::instance(std::io::Error::from(
        std::io::ErrorKind::ConnectionReset,
    ))));
    let mut connection: BoxedWebSocketConnection = Box::new(ConnectionWithCapabilities {
        ready_results: ready_results.into_iter(),
        flush_results: Vec::new().into_iter(),
    });

    for outcome in outcomes {
        assert_eq!(
            WebSocketConnection::recv_ready(&mut connection)
                .await
                .expect("the boxed ready read must reach the override"),
            outcome
        );
    }

    let error = WebSocketConnection::recv_ready(&mut connection)
        .await
        .expect_err("the override's ready-read error must survive boxing");
    match error {
        Error::Instance(inner) => assert_eq!(
            inner
                .downcast_ref::<std::io::Error>()
                .expect("the override's concrete error type must survive")
                .kind(),
            std::io::ErrorKind::ConnectionReset
        ),
        other => panic!("expected the override's error instance, got {other:?}"),
    }
}

#[tokio::test]
async fn boxed_connection_forwards_flush_success_and_errors() {
    let mut connection: BoxedWebSocketConnection = Box::new(ConnectionWithCapabilities {
        ready_results: Vec::new().into_iter(),
        flush_results: vec![Ok(()), Err(Error::StreamEnded), Ok(())].into_iter(),
    });

    WebSocketConnection::flush(&mut connection)
        .await
        .expect("the boxed flush must reach the override");
    assert!(matches!(
        WebSocketConnection::flush(&mut connection).await,
        Err(Error::StreamEnded)
    ));
    WebSocketConnection::flush(&mut connection)
        .await
        .expect("the next flush must still reach the override after an error");
}
