use super::*;

#[test]
fn rejection_and_unpolled_drop_have_distinct_stages_and_no_dispatch() {
    let (rejected, writer) = Observation::prepare();
    writer.reject_before_dispatch();
    rejected.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::CallerRejectedBeforeDispatch);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
        assert!(view.committed_prefix.is_empty());
    });
    let (cancelled, writer) = Observation::prepare();
    drop(writer);
    cancelled.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::NotDispatched);
    });
}

#[test]
fn original_prefix_and_send_attempt_survive_owner_drop() {
    let (handle, mut writer) = Observation::prepare();
    let retained = handle.clone();
    assert!(retained.same_operation(&writer.handle()));
    handle.mark_send_attempt_started();
    let original = "  {\"delta\":\"first\",\"delta\":\"second\"}  \n".to_owned();
    assert!(writer.append(original.clone()).is_ok());
    drop(writer);
    retained.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::Cancelled);
        assert_eq!(view.dispatch, ObservationDispatch::SendAttemptStarted);
        assert_eq!(view.committed_prefix, &[original]);
        assert!(view.capture_fault.is_none());
    });
    let (other, _writer) = Observation::prepare();
    assert!(!retained.same_operation(&other));
}

#[test]
fn poisoned_append_retains_prefix_and_whole_uncommitted_string() {
    let (handle, mut writer) = Observation::prepare();
    handle.mark_send_attempt_started();
    assert!(writer.append("prefix".into()).is_ok());
    // Real allocation failure, induced as a deterministic capacity overflow.
    let mut allocation = Vec::<String>::new();
    let cause = match allocation.try_reserve(usize::MAX) {
        Err(cause) => cause,
        Ok(()) => panic!("an impossible String-vector capacity must fail"),
    };
    handle.lock().fault = Some(CaptureFault {
        cause: CaptureCause::Allocation(cause),
        offending_string: None,
    });
    let offending = "{\"type\":\"error\",\"error\":\"whole supplied cause\"}".to_owned();
    assert!(writer.append(offending.clone()).is_err());
    let failure = ObservedFailure::capture_only(&mut writer);
    assert_eq!(failure.kind, ObservedFailureKind::CaptureOnly);
    assert!(failure.observation.same_operation(&handle));
    assert!(matches!(
        failure.diagnostic_basis,
        ExistingDiagnosticBasis::NoExistingNativeCause
    ));
    assert!(writer.succeeded().is_err());
    drop(writer);
    handle.inspect(|view| {
        assert_eq!(view.stage, ObservationStage::NativeFailed);
        assert_eq!(view.committed_prefix, &["prefix".to_owned()]);
        let Some(fault) = view.capture_fault else {
            panic!("actual capture fault must remain")
        };
        assert_eq!(fault.offending_string.as_ref(), Some(&offending));
        assert!(view.native_cause.is_none());
    });
}

#[test]
fn finalization_and_caller_admission_have_separate_lifetimes() {
    let (handle, mut writer) = Observation::prepare();
    handle.mark_send_attempt_started();
    writer.awaiting_finalization();
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::AwaitingNativeFinalization));
    assert!(writer.succeeded().is_ok());
    drop(writer);
    // A caller retaining this handle may reject its returned answer separately.
    handle.inspect(|view| assert_eq!(view.stage, ObservationStage::NativeSucceeded));
}

#[test]
fn fakes_are_explicitly_synthetic() {
    let (handle, writer) = Observation::prepare_synthetic();
    handle.inspect(|view| {
        assert_eq!(view.provenance, ObservationProvenance::Synthetic);
        assert!(view.committed_prefix.is_empty());
    });
    drop(writer);
}
