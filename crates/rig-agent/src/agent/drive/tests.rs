use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use tokio::sync::Notify;

use crate::agent::AgentBuilder;
use crate::test_utils::{MockCompletionModel, MockFrame, MockRuntime, MockScript, MockTurn};
use rig_core::driver::{Exchange, Opening, Transport};

/// Holds the first request open until the test releases it.
#[derive(Clone)]
struct HoldFirst {
    inner: MockRuntime,
    requests: Arc<AtomicU32>,
    held: Arc<Notify>,
    release: Arc<Notify>,
}

impl Transport<MockScript> for HoldFirst {
    fn send(
        &self,
        payload: crate::completion::CompletionRequest,
        exchange: Exchange,
    ) -> Opening<MockFrame> {
        let this = self.clone();
        Opening::new(async move {
            if this.requests.fetch_add(1, SeqCst) == 0 {
                this.held.notify_one();
                this.release.notified().await;
            }
            Transport::<MockScript>::send(&this.inner, payload, exchange).await
        })
    }
}

/// A run that ends while another run of the same agent is still waiting on
/// its model settles only its own work: it returns once the in-flight count
/// stops falling, without waiting for the other run's live dispatch.
#[tokio::test]
async fn a_finished_run_does_not_wait_for_another_runs_live_dispatch() {
    let mock = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let held = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let model = rig_core::Model::new(
        mock.wire,
        HoldFirst {
            inner: mock.transport,
            requests: Arc::new(AtomicU32::new(0)),
            held: held.clone(),
            release: release.clone(),
        },
    );
    let agent = AgentBuilder::new(model).build();

    let mut waiting = std::pin::pin!(agent.prompt("waits").run());
    tokio::select! {
        biased;
        _ = &mut waiting => panic!("the held run cannot finish before its release"),
        () = held.notified() => {}
    }

    let finished = agent.prompt("finishes").run().await.expect("the free run");
    release.notify_one();
    let waited = waiting.await.expect("the held run");

    let outputs = std::collections::HashSet::from([finished.output(), waited.output()]);
    assert_eq!(
        outputs,
        std::collections::HashSet::from(["one".to_string(), "two".to_string()])
    );
}

fn model_metadata_keys(bus: &super::AgentBus) -> std::collections::BTreeSet<String> {
    bus.model_ids
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .keys()
        .cloned()
        .collect()
}

fn registered_keys(bus: &super::AgentBus) -> std::collections::BTreeSet<String> {
    bus.dispatcher()
        .descriptors()
        .into_iter()
        .map(|descriptor| descriptor.key.as_str().to_owned())
        .collect()
}

/// Unpolled runner clones keep one anonymous registration and release all its metadata.
#[test]
fn anonymous_model_metadata_lives_until_the_final_unpolled_runner_clone_drops() {
    let agent = AgentBuilder::new(MockCompletionModel::text("unused")).build();
    let bus = &agent.config.bus;
    let metadata_before = model_metadata_keys(bus);
    let registered_before = registered_keys(bus);
    let mut previous_key = None;
    for _ in 0..2 {
        let runner = agent
            .prompt("unpolled")
            .using_model_value(MockCompletionModel::text("unused"));
        let key = runner.config.model_key.raw().as_str().to_owned();
        assert_ne!(
            previous_key.as_ref(),
            Some(&key),
            "generated keys are distinct"
        );
        let clone = runner.clone();
        let mut expected_metadata = metadata_before.clone();
        expected_metadata.insert(key.clone());
        let mut expected_registered = registered_before.clone();
        expected_registered.insert(key.clone());
        assert_eq!(model_metadata_keys(bus), expected_metadata);
        assert_eq!(registered_keys(bus), expected_registered);

        drop(runner);
        assert_eq!(model_metadata_keys(bus), expected_metadata);
        assert_eq!(registered_keys(bus), expected_registered);
        drop(clone);
        assert_eq!(registered_keys(bus), registered_before);
        assert_eq!(model_metadata_keys(bus), metadata_before);
        previous_key = Some(key);
    }
}

/// Agent clones and unpolled runners share the same anonymous registration lifetime.
#[test]
fn anonymous_model_metadata_lives_until_the_last_selecting_agent_value_drops() {
    let mut agent =
        AgentBuilder::named_model("default", MockCompletionModel::text("unused")).build();
    let bus = agent.config.bus.clone();
    let metadata_before = model_metadata_keys(&bus);
    let registered_before = registered_keys(&bus);
    let mut previous_key = None;
    for _ in 0..2 {
        agent.set_model(MockCompletionModel::text("unused"));
        let key = agent.model_key().raw().as_str().to_owned();
        assert_ne!(
            previous_key.as_ref(),
            Some(&key),
            "generated keys are distinct"
        );
        let retained = agent.clone();
        let unpolled = agent.prompt("unpolled");
        agent.set_model_label("default");
        let mut expected_metadata = metadata_before.clone();
        expected_metadata.insert(key.clone());
        let mut expected_registered = registered_before.clone();
        expected_registered.insert(key.clone());
        assert_eq!(model_metadata_keys(&bus), expected_metadata);
        assert_eq!(registered_keys(&bus), expected_registered);

        drop(retained);
        assert_eq!(model_metadata_keys(&bus), expected_metadata);
        assert_eq!(registered_keys(&bus), expected_registered);
        drop(unpolled);
        assert_eq!(registered_keys(&bus), registered_before);
        assert_eq!(model_metadata_keys(&bus), metadata_before);
        previous_key = Some(key);
    }
}
