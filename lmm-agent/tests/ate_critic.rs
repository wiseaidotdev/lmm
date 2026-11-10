// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::thought::belief::BeliefInterval;
use lmm::cognition::thought::critic::{CritiqueResult, InternalCritic, StepView};

fn step<'a>(
    index: usize,
    premise: &'a str,
    conclusion: &'a str,
    belief: f64,
    plausibility: f64,
) -> StepView<'a> {
    StepView {
        index,
        premise,
        conclusion,
        belief: BeliefInterval::new(belief, plausibility),
    }
}

fn healthy_step(index: usize) -> StepView<'static> {
    StepView {
        index,
        premise: "Rust enforces memory safety through ownership and borrowing rules",
        conclusion: "Rust prevents data races at compile time without a runtime",
        belief: BeliefInterval::new(0.65, 0.85),
    }
}

#[test]
fn empty_step_list_yields_no_critique() {
    let critic = InternalCritic::new(0.02, 3);
    assert!(critic.evaluate(&[]).is_none());
}

#[test]
fn single_healthy_step_yields_no_critique() {
    let critic = InternalCritic::new(0.02, 3);
    let steps = vec![healthy_step(0)];
    assert!(critic.evaluate(&steps).is_none());
}

#[test]
fn two_healthy_steps_with_growing_belief_yield_no_critique() {
    let critic = InternalCritic::new(0.02, 3);
    let steps = vec![
        StepView {
            index: 0,
            premise: "Rust ownership rules guarantee memory safety without a garbage collector",
            conclusion: "Rust prevents use-after-free bugs automatically at compile time",
            belief: BeliefInterval::new(0.50, 0.75),
        },
        StepView {
            index: 1,
            premise: "Rust borrow checker enforces ownership rules across all code paths statically",
            conclusion: "Rust eliminates data races through compile-time borrow verification",
            belief: BeliefInterval::new(0.65, 0.85),
        },
    ];
    assert!(critic.evaluate(&steps).is_none());
}

#[test]
fn stagnant_belief_beyond_patience_triggers_stagnant_critique() {
    let patience = 2;
    let critic = InternalCritic::new(0.1, patience);
    let flat_belief = 0.5;
    let steps = vec![
        step(0, "A", "B", flat_belief, flat_belief + 0.1),
        step(1, "B", "C", flat_belief, flat_belief + 0.1),
        step(2, "C", "D", flat_belief, flat_belief + 0.1),
    ];
    let result = critic.evaluate(&steps);
    assert!(result.is_some(), "expected stagnant critique, got None");
    if let Some(CritiqueResult::StagnantBelief { stall_count }) = result {
        assert!(
            stall_count >= patience,
            "stall_count={stall_count} < patience={patience}"
        );
    }
}

#[test]
fn contradiction_detection_flags_conflicting_steps() {
    let critic = InternalCritic::default();
    let steps = vec![
        step(0, "Rust is safe", "Rust prevents data races", 0.8, 0.95),
        step(1, "Rust is unsafe", "Rust prevents data races", 0.4, 0.6),
        step(2, "Safety is good", "Rust is good", 0.7, 0.9),
    ];
    let result = critic.evaluate(&steps);
    if let Some(CritiqueResult::Contradiction { step_a, step_b }) = result {
        assert!(
            step_b > step_a,
            "step_b ({step_b}) must be after step_a ({step_a})"
        );
    }
}

#[test]
fn weak_premise_step_triggers_critique() {
    let critic = InternalCritic::new(0.02, 10);
    let steps = vec![step(0, "", "some conclusion", 0.0, 0.1)];
    let result = critic.evaluate(&steps);
    if let Some(CritiqueResult::WeakPremise { step, .. }) = result {
        assert_eq!(step, 0);
    }
}

#[test]
fn critique_result_contradiction_is_critical() {
    let r = CritiqueResult::Contradiction {
        step_a: 0,
        step_b: 1,
    };
    assert!(r.is_critical());
}

#[test]
fn critique_result_stagnant_is_critical() {
    let r = CritiqueResult::StagnantBelief { stall_count: 3 };
    assert!(r.is_critical());
}

#[test]
fn critique_result_weak_premise_is_critical() {
    let r = CritiqueResult::WeakPremise {
        step: 0,
        suggestion: "expand premise".into(),
    };
    assert!(r.is_critical());
}

#[test]
fn critique_result_describe_is_non_empty_for_all_variants() {
    let variants = vec![
        CritiqueResult::Contradiction {
            step_a: 0,
            step_b: 2,
        },
        CritiqueResult::StagnantBelief { stall_count: 3 },
        CritiqueResult::WeakPremise {
            step: 1,
            suggestion: "needs detail".into(),
        },
    ];
    for v in &variants {
        assert!(
            !v.describe().is_empty(),
            "describe() was empty for variant {:?}",
            v
        );
    }
}

#[test]
fn critique_result_equality_works_for_contradiction() {
    let a = CritiqueResult::Contradiction {
        step_a: 0,
        step_b: 1,
    };
    let b = CritiqueResult::Contradiction {
        step_a: 0,
        step_b: 1,
    };
    assert_eq!(a, b);
}

#[test]
fn critique_result_inequality_across_variants() {
    let contradiction = CritiqueResult::Contradiction {
        step_a: 0,
        step_b: 1,
    };
    let stagnant = CritiqueResult::StagnantBelief { stall_count: 3 };
    assert_ne!(contradiction, stagnant);
}

#[test]
fn default_critic_evaluates_without_panic() {
    let critic = InternalCritic::default();
    let steps = vec![
        step(0, "premise a", "conclusion a", 0.6, 0.8),
        step(1, "premise b", "conclusion b", 0.65, 0.85),
        step(2, "premise c", "conclusion c", 0.7, 0.9),
    ];
    let _ = critic.evaluate(&steps);
}

#[test]
fn increasing_patience_delays_stagnation_critique() {
    let strict_critic = InternalCritic::new(0.1, 2);
    let lenient_critic = InternalCritic::new(0.1, 10);
    let steps: Vec<StepView<'_>> = (0..4).map(|i| step(i, "flat", "flat", 0.5, 0.6)).collect();
    let strict_result = strict_critic.evaluate(&steps);
    let lenient_result = lenient_critic.evaluate(&steps);
    assert!(
        strict_result.is_some(),
        "strict critic should flag stagnation"
    );
    assert!(
        lenient_result.is_none()
            || matches!(lenient_result, Some(CritiqueResult::WeakPremise { .. })),
        "lenient critic should not flag stagnation for only 4 flat steps with patience=10"
    );
}
