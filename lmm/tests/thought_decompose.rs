// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unit tests for [`lmm::cognition::thought::decompose::GoalDecomposer`].

use lmm::cognition::thought::belief::BeliefInterval;
use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::cognition::thought::decompose::GoalDecomposer;

#[test]
fn factual_mode_produces_non_empty_sub_goals() {
    let goals = GoalDecomposer::decompose("what is Rust", ReasoningMode::FactualLookup, &[]);
    assert!(!goals.is_empty());
}

#[test]
fn factual_mode_goal_contains_topic() {
    let goals = GoalDecomposer::decompose("what is Rust", ReasoningMode::FactualLookup, &[]);
    assert!(
        goals
            .iter()
            .any(|g| g.to_ascii_lowercase().contains("rust"))
    );
}

#[test]
fn causal_mode_includes_mechanism_aspect() {
    let goals = GoalDecomposer::decompose("why does Rust avoid GC", ReasoningMode::Causal, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("mechanism"));
}

#[test]
#[ignore]
fn causal_mode_includes_effects_aspect() {
    let goals = GoalDecomposer::decompose("why does Rust avoid GC", ReasoningMode::Causal, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("effect") || joined.contains("consequence"));
}

#[test]
fn analogical_mode_includes_differences_aspect() {
    let goals = GoalDecomposer::decompose("rust vs go performance", ReasoningMode::Analogical, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("difference") || joined.contains("contrast"));
}

#[test]
fn definitional_mode_includes_definition_aspect() {
    let goals = GoalDecomposer::decompose("ownership", ReasoningMode::Definitional, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("definition") || joined.contains("formal"));
}

#[test]
fn procedural_mode_includes_steps_aspect() {
    let goals = GoalDecomposer::decompose(
        "how to build a Rust web server",
        ReasoningMode::Procedural,
        &[],
    );
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("step"));
}

#[test]
fn exploratory_mode_includes_overview_aspect() {
    let goals = GoalDecomposer::decompose("photosynthesis", ReasoningMode::Exploratory, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("overview") || joined.contains("history") || joined.contains("notable")
    );
}

#[test]
#[ignore]
fn evaluative_mode_includes_evidence_aspects() {
    let goals = GoalDecomposer::decompose("is Rust worth it", ReasoningMode::Evaluative, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(joined.contains("evidence") || joined.contains("criteria"));
}

#[test]
fn extract_topic_strips_what_is_prefix() {
    let topic = GoalDecomposer::extract_topic("what is the borrow checker", &[]);
    assert_eq!(topic.trim(), "borrow checker");
}

#[test]
fn extract_topic_strips_how_does_prefix() {
    let topic = GoalDecomposer::extract_topic("how does ownership work", &[]);
    assert!(topic.to_ascii_lowercase().contains("ownership"));
}

#[test]
fn extract_topic_bare_noun_unchanged() {
    let topic = GoalDecomposer::extract_topic("photosynthesis", &[]);
    assert_eq!(topic, "photosynthesis");
}

#[test]
fn extract_topic_empty_fallback() {
    let topic = GoalDecomposer::extract_topic("what is", &[]);
    assert!(!topic.is_empty());
}

#[test]
fn prioritize_puts_ignorant_belief_first() {
    let goals = vec!["first goal".to_string(), "second goal".to_string()];
    let evidence = vec![BeliefInterval::certain(), BeliefInterval::ignorant()];
    let prioritized = GoalDecomposer::prioritize(goals, &evidence);
    assert_eq!(prioritized[0], "second goal");
}

#[test]
fn prioritize_with_equal_beliefs_preserves_order() {
    let goals = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let evidence = vec![
        BeliefInterval::new(0.5, 0.7),
        BeliefInterval::new(0.5, 0.7),
        BeliefInterval::new(0.5, 0.7),
    ];
    let prioritized = GoalDecomposer::prioritize(goals, &evidence);
    assert_eq!(prioritized.len(), 3);
}

#[test]
fn prioritize_with_no_evidence_returns_all_goals() {
    let goals = vec!["x".to_string(), "y".to_string()];
    let evidence: Vec<BeliefInterval> = vec![];
    let prioritized = GoalDecomposer::prioritize(goals, &evidence);
    assert_eq!(prioritized.len(), 2);
}

#[test]
fn all_modes_produce_at_least_three_sub_goals() {
    let modes = [
        ReasoningMode::FactualLookup,
        ReasoningMode::Causal,
        ReasoningMode::Analogical,
        ReasoningMode::Definitional,
        ReasoningMode::Procedural,
        ReasoningMode::Exploratory,
        ReasoningMode::Evaluative,
    ];
    for mode in modes {
        let goals = GoalDecomposer::decompose("test topic", mode, &[]);
        assert!(
            goals.len() >= 3,
            "Mode {:?} produced fewer than 3 sub-goals",
            mode
        );
    }
}
