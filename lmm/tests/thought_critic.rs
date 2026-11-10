// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for [`lmm::cognition::thought::critic::InternalCritic`].

use lmm::cognition::thought::belief::BeliefInterval;
use lmm::cognition::thought::critic::{CritiqueResult, InternalCritic, StepView};

fn view(
    index: usize,
    premise: &'static str,
    conclusion: &'static str,
    mid: f64,
) -> StepView<'static> {
    let bel = (mid - 0.1).clamp(0.0, 1.0);
    let pl = (mid + 0.1).clamp(0.0, 1.0);
    StepView {
        index,
        premise,
        conclusion,
        belief: BeliefInterval::new(bel, pl),
    }
}

#[test]
fn coherent_chain_returns_none() {
    let critic = InternalCritic::new(0.02, 3);
    let steps = vec![
        view(
            0,
            "Rust is a systems language known for memory safety",
            "Rust prevents memory errors at compile time",
            0.70,
        ),
        view(
            1,
            "Rust prevents memory errors at compile time via its ownership model",
            "The borrow checker enforces ownership rules statically",
            0.75,
        ),
    ];
    assert!(critic.evaluate(&steps).is_none());
}

#[test]
fn contradictory_steps_detected() {
    let critic = InternalCritic::new(0.02, 3);
    let steps = vec![
        view(
            0,
            "some premise here with enough words",
            "Rust uses garbage collection for memory management",
            0.6,
        ),
        view(
            1,
            "another premise with enough words here",
            "Rust does not use garbage collection for memory",
            0.6,
        ),
    ];
    let result = critic.evaluate(&steps);
    assert!(result.is_some());
    if let Some(CritiqueResult::Contradiction { step_a, step_b }) = result {
        assert_eq!(step_a, 0);
        assert_eq!(step_b, 1);
    }
}

#[test]
fn non_contradictory_different_topics_not_flagged() {
    let critic = InternalCritic::new(0.02, 3);
    let steps = vec![
        view(
            0,
            "some premise with sufficient words",
            "Rust has a strict type system",
            0.75,
        ),
        view(
            1,
            "another premise with enough words",
            "Python is dynamically typed",
            0.70,
        ),
    ];
    assert!(critic.evaluate(&steps).is_none());
}

#[test]
fn stagnant_belief_detected_after_patience() {
    let critic = InternalCritic::new(0.05, 2);
    let steps = vec![
        StepView {
            index: 0,
            premise: "long enough premise for the step",
            conclusion: "first unique conclusion about the topic",
            belief: BeliefInterval::new(0.50, 0.70),
        },
        StepView {
            index: 1,
            premise: "long enough premise for the second step",
            conclusion: "second unique conclusion about topic details",
            belief: BeliefInterval::new(0.51, 0.71),
        },
        StepView {
            index: 2,
            premise: "long enough premise for the third step",
            conclusion: "third unique conclusion about more topic details",
            belief: BeliefInterval::new(0.52, 0.72),
        },
    ];
    let result = critic.evaluate(&steps);
    assert!(result.is_some(), "Expected stagnation to be detected");
}

#[test]
fn no_stagnation_when_belief_improves() {
    let critic = InternalCritic::new(0.05, 2);
    let steps = vec![
        StepView {
            index: 0,
            premise: "some long premise for the first step here",
            conclusion: "first conclusion unique words",
            belief: BeliefInterval::new(0.30, 0.50),
        },
        StepView {
            index: 1,
            premise: "another premise for the second step here",
            conclusion: "second conclusion unique words",
            belief: BeliefInterval::new(0.60, 0.80),
        },
        StepView {
            index: 2,
            premise: "third premise for the final step here",
            conclusion: "third conclusion unique words",
            belief: BeliefInterval::new(0.80, 0.95),
        },
    ];
    let result = critic.evaluate(&steps);
    assert!(result.is_none(), "Should not flag rapidly improving belief");
}

#[test]
fn weak_premise_flagged_below_word_threshold() {
    let critic = InternalCritic::new(0.02, 5);
    let steps = vec![StepView {
        index: 0,
        premise: "hi",
        conclusion: "some valid conclusion about the topic here",
        belief: BeliefInterval::new(0.6, 0.8),
    }];
    let result = critic.evaluate(&steps);
    assert!(matches!(result, Some(CritiqueResult::WeakPremise { .. })));
}

#[test]
fn weak_premise_flagged_for_unresolved_belief() {
    let critic = InternalCritic::new(0.02, 5);
    let steps = vec![StepView {
        index: 0,
        premise: "this is a valid long enough premise",
        conclusion: "conclusion about topic details here",
        belief: BeliefInterval::ignorant(),
    }];
    let result = critic.evaluate(&steps);
    assert!(matches!(result, Some(CritiqueResult::WeakPremise { .. })));
}

#[test]
fn empty_steps_returns_none() {
    let critic = InternalCritic::new(0.02, 3);
    assert!(critic.evaluate(&[]).is_none());
}

#[test]
fn single_valid_step_returns_none() {
    let critic = InternalCritic::default();
    let steps = vec![view(
        0,
        "good premise with enough words here",
        "solid conclusion derived from evidence",
        0.75,
    )];
    assert!(critic.evaluate(&steps).is_none());
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
        suggestion: String::new(),
    };
    assert!(r.is_critical());
}

#[test]
fn critique_result_describe_is_non_empty() {
    let r = CritiqueResult::Contradiction {
        step_a: 1,
        step_b: 2,
    };
    assert!(!r.describe().is_empty());
}

#[test]
fn default_critic_has_sensible_defaults() {
    let critic = InternalCritic::default();
    assert!(critic.stall_patience >= 1);
    assert!(critic.improvement_epsilon >= 0.0);
}
