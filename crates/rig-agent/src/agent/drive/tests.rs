use super::*;
use crate::bus::Bus;
use futures::{StreamExt, task::noop_waker_ref};
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    serve::{Dispatch, Reply, Serve},
};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

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

/// A waker that counts its wakes: another run's, registered on the bus.
struct Counting(AtomicUsize);

impl Wake for Counting {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn custom() -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("test"),
        payload: serde_json::Value::Null,
    }
}

/// Answers at once.
struct Handler;

impl Serve for Handler {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Ok(Outcome::Custom {
            payload: serde_json::Value::Null,
        }))
    }
}

/// An owned agent bus serving `k`, and another run's waker registered on
/// it — a run that found the driver lock taken and is waiting to drive.
fn bus_with_a_waiting_run() -> (AgentBus, Arc<Counting>) {
    let policy = ServingPolicy::default();
    let (dispatcher, registrar, mut driver) = Bus::channel_with(policy);
    driver.register("k", Handler).expect("a fresh key");
    let bus = AgentBus::owned(dispatcher, registrar, driver, "owner".to_owned(), policy);
    let waiting = Arc::new(Counting(AtomicUsize::new(0)));
    let slot = bus.wakers.slot();
    bus.wakers
        .register(slot, &Waker::from(Arc::clone(&waiting)));
    (bus, waiting)
}

/// A run that finished drove the bus last. Another run that registered
/// its waker and found the lock taken may have buffered a command since
/// the finished run's last drain; nothing else polls the driver between
/// runs, so the finished run must wake the runs still registered, which
/// then take the lock and drive. (The obligation the loom model of the
/// protocol assumes of a run's end.) Exactly one wake, at the end: the
/// polls that serve the run's own dispatch wake nobody.
#[test]
fn a_finished_run_wakes_the_runs_that_registered_while_it_drove() {
    let (bus, waiting) = bus_with_a_waiting_run();
    let key = HandlerKey::from("k");
    let mut run = bus.drive(futures::stream::once(
        bus.dispatcher().dispatch(&key, custom()),
    ));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(
        matches!(run.poll_next_unpin(&mut cx), Poll::Ready(Some(Ok(_)))),
        "the run served its own dispatch while it drove"
    );
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        0,
        "serving the run's own dispatch woke nobody"
    );
    assert!(matches!(run.poll_next_unpin(&mut cx), Poll::Ready(None)));
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        1,
        "the run that stopped driving woke the runs still registered, once"
    );
}

/// The same obligation for a run dropped while live: its answer landed
/// but its stream was never polled to its end, so `Drop` gives the driver
/// its last poll (nothing in flight to settle) and wakes the runs still
/// registered to take over the bus. Exactly one wake, from the drop.
#[test]
fn a_dropped_run_wakes_the_runs_that_registered_while_it_drove() {
    let (bus, waiting) = bus_with_a_waiting_run();
    let key = HandlerKey::from("k");
    let mut run = bus.drive(futures::stream::once(
        bus.dispatcher().dispatch(&key, custom()),
    ));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(
        run.poll_next_unpin(&mut cx),
        Poll::Ready(Some(Ok(_)))
    ));
    assert_eq!(waiting.0.load(Ordering::SeqCst), 0);
    drop(run);
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        1,
        "the dropped run woke the runs still registered, once"
    );
}
