use super::*;

use std::time::{Duration, Instant};

/// A fixture path whose sidecar does not exist: replay starts with no
/// readings, and nothing here writes beside it.
fn absent_fixture() -> &'static Path {
    Path::new("absent-fixture-dir/pause.yaml")
}

fn replaying() -> CassetteClock {
    CassetteClock::start(CassetteMode::Replay, absent_fixture()).expect("replay starts")
}

/// Replay hands back what the recording waited for at once, so a replayed
/// pause of the full limit finishes far inside a one-second bound, and it
/// takes no clock reading: the replay still has every reading unused.
#[tokio::test]
async fn a_replayed_pause_returns_at_once_and_reads_no_time() {
    let clock = replaying();
    let paused = tokio::time::timeout(Duration::from_secs(1), clock.pause(MAX_PAUSE)).await;
    assert!(paused.is_ok(), "a replayed pause of {MAX_PAUSE:?} waited");
    assert_eq!(clock.state.next.load(Ordering::SeqCst), 0);
    clock
        .finish(absent_fixture())
        .await
        .expect("a replay that recorded no readings used them all");
}

/// Recording waits for real, so the live provider sees the gap the test
/// asked for, and the pause itself saves no reading.
#[tokio::test]
async fn a_recorded_pause_waits_the_whole_duration_without_a_reading() {
    let clock =
        CassetteClock::start(CassetteMode::Record, absent_fixture()).expect("record starts");
    let pause = Duration::from_millis(20);
    let started = Instant::now();
    clock.pause(pause).await;
    let waited = started.elapsed();
    assert!(
        waited >= pause,
        "a recorded pause of {pause:?} waited {waited:?}"
    );
    assert_eq!(
        *clock
            .state
            .readings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Vec::<u64>::new()
    );
}

/// The limit is inclusive: a pause of exactly [`MAX_PAUSE`] is accepted.
#[tokio::test]
async fn a_pause_at_the_limit_is_accepted() {
    replaying().pause(MAX_PAUSE).await;
}

#[tokio::test]
#[should_panic(expected = "a cassette pause of 60.001s exceeds the 60s limit")]
async fn a_pause_beyond_the_limit_panics_with_both_durations() {
    replaying()
        .pause(MAX_PAUSE + Duration::from_millis(1))
        .await;
}
