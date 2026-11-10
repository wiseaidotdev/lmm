// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for [`lmm::cognition::thought::ThoughtChain`] and [`lmm::cognition::thought::ThoughtStep`].

use lmm::cognition::thought::belief::BeliefInterval;
use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::cognition::thought::{StepMode, ThoughtChain, ThoughtStep};

fn make_step(index: usize, mid: f64) -> ThoughtStep {
    let bel = (mid - 0.1).clamp(0.0, 1.0);
    let pl = (mid + 0.1).clamp(0.0, 1.0);
    ThoughtStep {
        index,
        premise: format!("Premise for step {}", index),
        reasoning: format!("Reasoning for step {} leads to insight.", index),
        conclusion: format!("Conclusion for step {}.", index),
        belief: BeliefInterval::new(bel, pl),
        mode: StepMode::Deductive,
        evidence_text: format!("Evidence text for step {} is a full sentence.", index),
    }
}

#[test]
fn new_chain_starts_empty() {
    let chain = ThoughtChain::new("test goal", ReasoningMode::Exploratory);
    assert!(chain.steps.is_empty());
    assert_eq!(chain.goal, "test goal");
    assert_eq!(chain.mode, ReasoningMode::Exploratory);
}

#[test]
fn new_chain_final_belief_starts_as_ignorant() {
    let chain = ThoughtChain::new("goal", ReasoningMode::Causal);
    assert_eq!(chain.final_belief.belief, 0.0);
    assert_eq!(chain.final_belief.plausibility, 1.0);
}

#[test]
fn push_step_increments_step_count() {
    let mut chain = ThoughtChain::new("goal", ReasoningMode::Definitional);
    chain.push_step(make_step(0, 0.6));
    assert_eq!(chain.steps.len(), 1);
    chain.push_step(make_step(1, 0.7));
    assert_eq!(chain.steps.len(), 2);
}

#[test]
fn push_step_updates_final_belief() {
    let mut chain = ThoughtChain::new("goal", ReasoningMode::FactualLookup);
    let initial = chain.final_belief.mid();
    chain.push_step(make_step(0, 0.8));
    let after = chain.final_belief.mid();
    assert!(
        after != initial || chain.final_belief.belief != 0.0,
        "Final belief should change after pushing a step"
    );
}

#[test]
fn last_conclusion_returns_goal_when_empty() {
    let chain = ThoughtChain::new("my goal text", ReasoningMode::Exploratory);
    assert_eq!(chain.last_conclusion(), "my goal text");
}

#[test]
fn last_conclusion_returns_last_step_conclusion() {
    let mut chain = ThoughtChain::new("goal", ReasoningMode::Causal);
    chain.push_step(make_step(0, 0.5));
    chain.push_step(make_step(1, 0.7));
    assert!(chain.last_conclusion().contains("Conclusion for step 1"));
}

#[test]
fn distil_concatenates_step_reasonings() {
    let mut chain = ThoughtChain::new("test", ReasoningMode::Procedural);
    chain.push_step(make_step(0, 0.6));
    chain.push_step(make_step(1, 0.7));
    let text = chain.distil();
    assert!(text.contains("step 0"));
    assert!(text.contains("step 1"));
}

#[test]
fn distil_empty_chain_returns_empty_string() {
    let chain = ThoughtChain::new("goal", ReasoningMode::Exploratory);
    assert_eq!(chain.distil(), "");
}

#[test]
fn step_indices_are_stored_correctly() {
    let mut chain = ThoughtChain::new("goal", ReasoningMode::Causal);
    chain.push_step(make_step(0, 0.5));
    chain.push_step(make_step(1, 0.6));
    chain.push_step(make_step(2, 0.7));
    for (i, step) in chain.steps.iter().enumerate() {
        assert_eq!(step.index, i);
    }
}

#[test]
fn step_mode_label_is_non_empty() {
    let modes = [
        StepMode::Deductive,
        StepMode::Abductive,
        StepMode::Analogical,
        StepMode::Corrective,
    ];
    for mode in modes {
        assert!(!mode.label().is_empty());
    }
}

#[test]
fn step_modes_all_distinct_labels() {
    use std::collections::HashSet;
    let labels: HashSet<&str> = [
        StepMode::Deductive,
        StepMode::Abductive,
        StepMode::Analogical,
        StepMode::Corrective,
    ]
    .iter()
    .map(|m| m.label())
    .collect();
    assert_eq!(labels.len(), 4);
}

#[test]
fn multiple_steps_monotone_indices_work() {
    let mut chain = ThoughtChain::new("test", ReasoningMode::Evaluative);
    for i in 0..5 {
        chain.push_step(make_step(i, 0.5 + i as f64 * 0.05));
    }
    assert_eq!(chain.steps.len(), 5);
}
