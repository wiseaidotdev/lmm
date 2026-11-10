// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for [`lmm::cognition::thought::AdaptiveThoughtEngine`].

use lmm::cognition::search::SearchOracle;
use lmm::cognition::thought::AdaptiveThoughtEngine;
use lmm::cognition::thought::classifier::{InputClassifier, ReasoningMode};

#[tokio::test]
async fn engine_runs_factual_query_within_max_steps() {
    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("what is Rust?", &mut oracle).await;
    assert!(result.chain.steps.len() <= 4);
}

#[tokio::test]
async fn engine_returns_non_empty_prioritized_queries() {
    let engine = AdaptiveThoughtEngine::default();
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("what is Rust?", &mut oracle).await;
    assert!(!result.prioritized_queries.is_empty());
}

#[tokio::test]
async fn engine_confidence_is_in_unit_range() {
    let engine = AdaptiveThoughtEngine::default();
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("photosynthesis", &mut oracle).await;
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[tokio::test]
async fn engine_handles_single_word_input() {
    let engine = AdaptiveThoughtEngine::new(3, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("why", &mut oracle).await;
    assert!(!result.prioritized_queries.is_empty());
    assert_eq!(result.chain.mode, ReasoningMode::Exploratory);
}

#[tokio::test]
async fn engine_classifies_causal_question_correctly() {
    let mode = InputClassifier::classify("why does Rust avoid garbage collection?");
    assert_eq!(mode, ReasoningMode::Causal);

    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine
        .run("why does Rust avoid garbage collection?", &mut oracle)
        .await;
    assert_eq!(result.chain.mode, ReasoningMode::Causal);
}

#[tokio::test]
async fn engine_classifies_analogical_question_correctly() {
    let mode = InputClassifier::classify("rust vs go performance");
    assert_eq!(mode, ReasoningMode::Analogical);

    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("rust vs go performance", &mut oracle).await;
    assert_eq!(result.chain.mode, ReasoningMode::Analogical);
}

#[tokio::test]
async fn engine_classifies_procedural_question_correctly() {
    let mode = InputClassifier::classify("how to install Rust on Linux");
    assert_eq!(mode, ReasoningMode::Procedural);

    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine
        .run("how to install Rust on Linux", &mut oracle)
        .await;
    assert_eq!(result.chain.mode, ReasoningMode::Procedural);
}

#[tokio::test]
async fn engine_produces_steps_with_monotone_indices() {
    let engine = AdaptiveThoughtEngine::new(5, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("what is the borrow checker?", &mut oracle).await;
    for (i, step) in result.chain.steps.iter().enumerate() {
        assert_eq!(step.index, i, "step index mismatch at position {}", i);
    }
}

#[tokio::test]
async fn engine_chain_goal_matches_input() {
    let engine = AdaptiveThoughtEngine::default();
    let mut oracle = SearchOracle::new(3);
    let prompt = "explain the ownership model";
    let result = engine.run(prompt, &mut oracle).await;
    assert_eq!(result.chain.goal, prompt);
}

#[tokio::test]
async fn engine_sub_goals_match_chain_sub_goals() {
    let engine = AdaptiveThoughtEngine::default();
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("photosynthesis", &mut oracle).await;
    assert_eq!(
        result.chain.sub_goals.len(),
        result.prioritized_queries.len()
    );
}

#[tokio::test]
async fn engine_respects_max_steps_cap() {
    let max = 2;
    let engine = AdaptiveThoughtEngine::new(max, 0.0, 10);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("photosynthesis", &mut oracle).await;
    assert!(result.chain.steps.len() <= max);
}

#[tokio::test]
async fn engine_step_modes_are_valid() {
    use lmm::cognition::thought::StepMode;
    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("what is Rust", &mut oracle).await;
    for step in &result.chain.steps {
        assert!(
            matches!(
                step.mode,
                StepMode::Deductive
                    | StepMode::Abductive
                    | StepMode::Analogical
                    | StepMode::Corrective
            ),
            "unexpected step mode: {:?}",
            step.mode
        );
    }
}

#[tokio::test]
async fn engine_step_beliefs_are_valid_intervals() {
    let engine = AdaptiveThoughtEngine::new(4, 0.55, 2);
    let mut oracle = SearchOracle::new(3);
    let result = engine.run("photosynthesis", &mut oracle).await;
    for step in &result.chain.steps {
        assert!(step.belief.belief <= step.belief.plausibility);
        assert!(step.belief.belief >= 0.0);
        assert!(step.belief.plausibility <= 1.0);
    }
}

#[tokio::test]
async fn engine_distinct_modes_produce_distinct_queries() {
    let engine = AdaptiveThoughtEngine::new(3, 0.55, 2);
    let mut oracle = SearchOracle::new(3);

    let r_factual = engine.run("what is photosynthesis", &mut oracle).await;
    let r_causal = engine
        .run("why does photosynthesis occur", &mut oracle)
        .await;

    assert_ne!(
        r_factual.prioritized_queries, r_causal.prioritized_queries,
        "Different modes should produce different prioritized queries"
    );
}
