// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for [`lmm::cognition::thought::belief::BeliefInterval`].

use lmm::cognition::thought::belief::BeliefInterval;

#[test]
fn ignorant_has_zero_belief_and_full_plausibility() {
    let b = BeliefInterval::ignorant();
    assert_eq!(b.belief, 0.0);
    assert_eq!(b.plausibility, 1.0);
}

#[test]
fn certain_has_unit_belief_and_unit_plausibility() {
    let b = BeliefInterval::certain();
    assert_eq!(b.belief, 1.0);
    assert_eq!(b.plausibility, 1.0);
}

#[test]
fn ignorant_interval_width_is_one() {
    assert!((BeliefInterval::ignorant().interval_width() - 1.0).abs() < 1e-9);
}

#[test]
fn certain_interval_width_is_zero() {
    assert_eq!(BeliefInterval::certain().interval_width(), 0.0);
}

#[test]
fn new_swapped_args_are_normalised() {
    let b = BeliefInterval::new(0.9, 0.3);
    assert!(b.belief <= b.plausibility);
}

#[test]
fn new_clamped_above_one() {
    let b = BeliefInterval::new(2.0, 3.0);
    assert!(b.belief <= 1.0);
    assert!(b.plausibility <= 1.0);
}

#[test]
fn new_clamped_below_zero() {
    let b = BeliefInterval::new(-1.0, -0.5);
    assert!(b.belief >= 0.0);
    assert!(b.plausibility >= 0.0);
}

#[test]
fn mid_is_average_of_belief_and_plausibility() {
    let b = BeliefInterval::new(0.4, 0.8);
    assert!((b.mid() - 0.6).abs() < 1e-9);
}

#[test]
fn from_evidence_full_overlap_yields_near_certain() {
    let b = BeliefInterval::from_evidence(10, 10);
    assert!(b.belief >= 0.85);
}

#[test]
fn from_evidence_no_overlap_yields_zero_belief() {
    let b = BeliefInterval::from_evidence(0, 10);
    assert_eq!(b.belief, 0.0);
}

#[test]
fn from_evidence_zero_total_yields_ignorant() {
    let b = BeliefInterval::from_evidence(0, 0);
    assert_eq!(b.belief, 0.0);
    assert_eq!(b.plausibility, 1.0);
}

#[test]
fn combination_of_supporting_sources_increases_belief() {
    let a = BeliefInterval::new(0.5, 0.8);
    let b = BeliefInterval::new(0.4, 0.75);
    let combined = a.combine(&b);
    assert!(combined.belief >= a.belief.min(b.belief));
    assert!(combined.belief <= combined.plausibility);
}

#[test]
fn combination_is_approximately_commutative() {
    let a = BeliefInterval::new(0.6, 0.9);
    let b = BeliefInterval::new(0.5, 0.8);
    let ab = a.combine(&b);
    let ba = b.combine(&a);
    assert!((ab.belief - ba.belief).abs() < 1e-6);
    assert!((ab.plausibility - ba.plausibility).abs() < 1e-6);
}

#[test]
fn combination_with_certain_does_not_reduce_belief() {
    let a = BeliefInterval::new(0.6, 0.8);
    let certain = BeliefInterval::certain();
    let combined = a.combine(&certain);
    assert!(combined.belief >= a.belief * 0.8);
}

#[test]
fn actionable_above_threshold_returns_true() {
    let b = BeliefInterval::new(0.7, 0.9);
    assert!(b.is_actionable(0.6));
}

#[test]
fn actionable_below_threshold_returns_false() {
    let b = BeliefInterval::new(0.1, 0.4);
    assert!(!b.is_actionable(0.6));
}

#[test]
fn ignorant_is_unresolved() {
    assert!(BeliefInterval::ignorant().is_unresolved());
}

#[test]
fn certain_is_not_unresolved() {
    assert!(!BeliefInterval::certain().is_unresolved());
}

#[test]
fn default_is_ignorant() {
    let d = BeliefInterval::default();
    assert_eq!(d.belief, 0.0);
    assert_eq!(d.plausibility, 1.0);
}
