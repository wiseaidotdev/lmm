// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::cognition::thought::belief::BeliefInterval;
use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::cognition::thought::decompose::GoalDecomposer;

#[test]
fn factual_mode_produces_non_empty_sub_goals() {
    let goals = GoalDecomposer::decompose("what is Rust", ReasoningMode::FactualLookup, &[]);
    assert!(!goals.is_empty());
}

#[test]
fn factual_mode_goal_contains_topic_word() {
    let goals = GoalDecomposer::decompose("what is Rust", ReasoningMode::FactualLookup, &[]);
    assert!(
        goals
            .iter()
            .any(|g| g.to_ascii_lowercase().contains("rust")),
        "none of the sub-goals mention 'rust': {:?}",
        goals
    );
}

#[test]
fn causal_mode_includes_mechanism_aspect() {
    let goals = GoalDecomposer::decompose("why does Rust avoid GC", ReasoningMode::Causal, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("mechanism"),
        "causal goals missing 'mechanism': {joined}"
    );
}

#[test]
fn causal_mode_includes_effects_aspect() {
    let goals = GoalDecomposer::decompose("why does Rust avoid GC", ReasoningMode::Causal, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("effect") || joined.contains("consequence"),
        "causal goals missing effects: {joined}"
    );
}

#[test]
fn analogical_mode_includes_differences_aspect() {
    let goals = GoalDecomposer::decompose("rust vs go performance", ReasoningMode::Analogical, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("difference") || joined.contains("contrast"),
        "analogical goals missing differences: {joined}"
    );
}

#[test]
fn analogical_mode_includes_shared_properties() {
    let goals = GoalDecomposer::decompose("rust vs python", ReasoningMode::Analogical, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("shared") || joined.contains("common") || joined.contains("properties"),
        "analogical goals missing shared properties: {joined}"
    );
}

#[test]
fn definitional_mode_includes_definition_aspect() {
    let goals = GoalDecomposer::decompose("ownership", ReasoningMode::Definitional, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("definition") || joined.contains("formal"),
        "definitional goals missing 'definition': {joined}"
    );
}

#[test]
fn procedural_mode_includes_steps_aspect() {
    let goals = GoalDecomposer::decompose(
        "how to build a Rust web server",
        ReasoningMode::Procedural,
        &[],
    );
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("step"),
        "procedural goals missing 'step': {joined}"
    );
}

#[test]
fn procedural_mode_includes_prerequisites() {
    let goals = GoalDecomposer::decompose("how to install Rust", ReasoningMode::Procedural, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("prerequisite") || joined.contains("require"),
        "procedural goals missing prerequisites: {joined}"
    );
}

#[test]
fn exploratory_mode_includes_overview_or_history() {
    let goals = GoalDecomposer::decompose("photosynthesis", ReasoningMode::Exploratory, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("overview") || joined.contains("history") || joined.contains("notable"),
        "exploratory goals missing overview/history: {joined}"
    );
}

#[test]
fn evaluative_mode_includes_evidence_aspects() {
    let goals = GoalDecomposer::decompose("is Rust worth it", ReasoningMode::Evaluative, &[]);
    let joined = goals.join(" ").to_ascii_lowercase();
    assert!(
        joined.contains("evidence") || joined.contains("criteria"),
        "evaluative goals missing evidence/criteria: {joined}"
    );
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
            "mode {mode:?} produced only {} sub-goals",
            goals.len()
        );
    }
}

#[test]
fn extract_topic_strips_what_is_prefix() {
    let topic = GoalDecomposer::extract_topic("what is the borrow checker", &[]);
    assert!(
        topic.to_ascii_lowercase().contains("borrow"),
        "expected 'borrow' in topic, got '{topic}'"
    );
}

#[test]
fn extract_topic_strips_how_does_prefix() {
    let topic = GoalDecomposer::extract_topic("how does ownership work", &[]);
    assert!(
        topic.to_ascii_lowercase().contains("ownership"),
        "expected 'ownership' in topic, got '{topic}'"
    );
}

#[test]
fn extract_topic_bare_noun_unchanged() {
    let topic = GoalDecomposer::extract_topic("photosynthesis", &[]);
    assert_eq!(topic, "photosynthesis");
}

#[test]
fn extract_topic_empty_fallback_is_non_empty() {
    let topic = GoalDecomposer::extract_topic("what is", &[]);
    assert!(!topic.is_empty(), "topic should not be empty for 'what is'");
}

#[test]
fn extract_topic_strips_why_does_prefix() {
    let topic = GoalDecomposer::extract_topic("why does Rust not have GC", &[]);
    assert!(
        topic.to_ascii_lowercase().contains("rust"),
        "expected 'rust' in topic, got '{topic}'"
    );
}

#[test]
fn extract_topic_strips_how_to_install_prefix() {
    let topic = GoalDecomposer::extract_topic("how to install rust", &[]);
    assert!(
        topic.to_ascii_lowercase().contains("rust"),
        "expected 'rust' in topic, got '{topic}'"
    );
}

#[test]
fn prioritize_puts_ignorant_belief_first() {
    let goals = vec!["first goal".to_string(), "second goal".to_string()];
    let evidence = vec![BeliefInterval::certain(), BeliefInterval::ignorant()];
    let prioritized = GoalDecomposer::prioritize(goals, &evidence);
    assert_eq!(
        prioritized[0], "second goal",
        "ignorant belief goal should be first"
    );
}

#[test]
fn prioritize_with_equal_beliefs_preserves_all_goals() {
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
fn prioritize_with_no_evidence_returns_all_goals_unchanged() {
    let goals = vec!["x".to_string(), "y".to_string()];
    let prioritized = GoalDecomposer::prioritize(goals.clone(), &[]);
    assert_eq!(prioritized.len(), goals.len());
}

#[test]
fn decompose_produces_unique_sub_goals() {
    let goals =
        GoalDecomposer::decompose("what is memory safety", ReasoningMode::FactualLookup, &[]);
    let unique: std::collections::HashSet<_> = goals.iter().collect();
    assert_eq!(
        unique.len(),
        goals.len(),
        "decompose produced duplicate sub-goals: {:?}",
        goals
    );
}

#[test]
fn decompose_sub_goals_are_non_empty_strings() {
    let goals = GoalDecomposer::decompose("what is Rust", ReasoningMode::FactualLookup, &[]);
    for g in &goals {
        assert!(!g.trim().is_empty(), "sub-goal was empty or whitespace");
    }
}
