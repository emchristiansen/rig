use super::*;

struct ConnectionWithoutCapabilities;

impl rig_http::ws_client::WebSocketConnection for ConnectionWithoutCapabilities {
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

impl WebSocketConnection for ConnectionWithoutCapabilities {}

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

impl rig_http::ws_client::WebSocketConnection for ConnectionWithCapabilities {
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

impl WebSocketConnection for ConnectionWithCapabilities {
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
