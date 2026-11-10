// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for [`lmm::cognition::drive::InternalDrive`],
//! [`lmm::cognition::drive::DriveState`], and [`lmm::cognition::drive::DriveSignal`].
//!
//! Verifies that the motivational drive system emits correct signals, respects
//! thresholds, and correctly identifies dominant drives.

use lmm::cognition::drive::{DriveSignal, DriveState, InternalDrive};

#[test]
fn new_drive_peek_dominant_is_none_or_valid() {
    let drive = InternalDrive::new(1.0, 1.0, 3);
    let dominant = drive.peek_dominant();
    if let Some(ref signal) = dominant {
        assert!(signal.magnitude() >= 0.0);
    }
}

#[test]
fn default_drive_tick_does_not_panic() {
    let mut drive = InternalDrive::default();
    let _ = drive.tick();
}

#[test]
fn recording_high_residual_increases_curiosity() {
    let mut drive = InternalDrive::new(2.0, 0.5, 2);
    for _ in 0..5 {
        drive.record_residual(1.0);
    }
    let state = drive.tick();
    assert!(
        state.dominant_drive().is_some(),
        "expected dominant drive after repeated residuals"
    );
}

#[test]
fn recording_incoherence_triggers_coherence_seeking() {
    let mut drive = InternalDrive::new(0.5, 2.0, 2);
    for _ in 0..5 {
        drive.record_incoherence(1.0);
    }
    let state = drive.tick();
    assert!(state.dominant_drive().is_some());
}

#[test]
fn recording_contradictions_triggers_drive() {
    let mut drive = InternalDrive::new(0.5, 0.5, 1);
    drive.record_contradiction();
    drive.record_contradiction();
    let state = drive.tick();
    assert!(state.dominant_drive().is_some());
}

#[test]
fn drive_state_total_urgency_is_non_negative() {
    let mut drive = InternalDrive::default();
    drive.record_residual(0.5);
    let state = drive.tick();
    assert!(state.total_urgency() >= 0.0);
}

#[test]
fn drive_state_is_idle_when_no_signals() {
    let state = DriveState { signals: vec![] };
    assert!(state.is_idle(0.01));
}

#[test]
fn drive_state_is_not_idle_when_signal_exceeds_threshold() {
    let state = DriveState {
        signals: vec![DriveSignal::Curiosity(1.0)],
    };
    assert!(!state.is_idle(0.1));
}

#[test]
fn drive_state_dominant_is_highest_magnitude() {
    let state = DriveState {
        signals: vec![
            DriveSignal::Curiosity(0.3),
            DriveSignal::CoherenceSeeking(0.9),
            DriveSignal::ContradictionResolution(0.6),
        ],
    };
    let dominant = state
        .dominant_drive()
        .expect("should have a dominant drive");
    assert!(
        (dominant.magnitude() - 0.9).abs() < 1e-9,
        "expected magnitude 0.9, got {}",
        dominant.magnitude()
    );
}

#[test]
fn drive_signal_curiosity_name() {
    let sig = DriveSignal::Curiosity(0.5);
    assert_eq!(sig.name(), "Curiosity");
}

#[test]
fn drive_signal_coherence_seeking_name() {
    let sig = DriveSignal::CoherenceSeeking(0.5);
    assert_eq!(sig.name(), "CoherenceSeeking");
}

#[test]
fn drive_signal_contradiction_resolution_name() {
    let sig = DriveSignal::ContradictionResolution(0.5);
    assert_eq!(sig.name(), "ContradictionResolution");
}

#[test]
fn drive_signal_magnitude_matches_inner_value() {
    let sig = DriveSignal::Curiosity(0.75);
    assert!((sig.magnitude() - 0.75).abs() < 1e-9);
}

#[test]
fn drive_state_no_dominant_when_empty() {
    let state = DriveState { signals: vec![] };
    assert!(state.dominant_drive().is_none());
}

#[test]
fn multiple_ticks_do_not_panic() {
    let mut drive = InternalDrive::default();
    for i in 0..10 {
        drive.record_residual(i as f64 * 0.1);
        let _ = drive.tick();
    }
}

#[test]
fn peek_dominant_before_tick_does_not_panic() {
    let drive = InternalDrive::default();
    let _ = drive.peek_dominant();
}

#[test]
#[ignore]
fn drive_with_zero_scales_stays_idle() {
    let mut drive = InternalDrive::new(0.0, 0.0, 10);
    drive.record_residual(0.5);
    drive.record_incoherence(0.5);
    let state = drive.tick();
    assert!(
        state.is_idle(0.01) || state.dominant_drive().is_none_or(|d| d.magnitude() < 0.01),
        "expected near-idle state with zero scales"
    );
}
