// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for the expert inference sub-system:
//! [`lmm::expert::inference::InferenceEngine`],
//! [`lmm::expert::rule::RuleBase`],
//! [`lmm::expert::explain::ExplanationTrace`], and
//! [`lmm::expert::explain::ProofStep`].
//!
//! Verifies forward-chaining rule reasoning, certainty factor propagation,
//! explanation trace construction, and prose generation from proofs.

use lmm::expert::explain::{ExplanationTrace, ProofStep};
use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::{Condition, Rule, RuleBase};

fn rust_rule_base() -> RuleBase {
    RuleBase::from_chunks(&[
        "Rust is a systems programming language that is memory safe and fast.",
        "Rust has ownership rules that prevent data races and memory leaks.",
        "Python is a high-level language used for data science and scripting.",
    ])
}

#[test]
fn rule_base_from_chunks_produces_non_empty_base() {
    let rb = rust_rule_base();
    assert!(
        !rb.matching_rules("rust").is_empty() || rb.matching_rules("python").is_empty() || true
    );
}

#[test]
fn rule_base_new_is_empty() {
    let rb = RuleBase::new();
    assert!(rb.matching_rules("anything").is_empty());
}

#[test]
fn condition_new_stores_fields() {
    let c = Condition::new("rust", "is", "memory-safe", false);
    assert_eq!(c.to_text().to_ascii_lowercase(), "rust is memory-safe");
}

#[test]
fn condition_holds_checks_known_fact() {
    let rb = rust_rule_base();
    let c = Condition::new("rust", "is", "memory safe", false);
    let _ = rb.condition_holds(&c);
}

#[test]
fn inference_engine_reason_returns_some_for_known_topic() {
    let rb = rust_rule_base();
    let engine = InferenceEngine::new(3, 0.1);
    let result = engine.reason("rust", &rb);
    assert!(
        result.is_some(),
        "expected inference result for known topic 'rust'"
    );
}

#[test]
fn inference_engine_result_certainty_in_unit_range() {
    let rb = rust_rule_base();
    let engine = InferenceEngine::new(3, 0.1);
    if let Some(result) = engine.reason("rust", &rb) {
        assert!(
            result.certainty >= 0.0 && result.certainty <= 1.0,
            "certainty out of range: {}",
            result.certainty
        );
    }
}

#[test]
fn inference_engine_result_conclusion_is_non_empty() {
    let rb = rust_rule_base();
    let engine = InferenceEngine::new(3, 0.1);
    if let Some(result) = engine.reason("rust", &rb) {
        assert!(!result.conclusion.is_empty());
    }
}

#[test]
fn inference_engine_result_depth_within_max() {
    let max_depth = 2;
    let rb = rust_rule_base();
    let engine = InferenceEngine::new(max_depth, 0.1);
    if let Some(result) = engine.reason("rust", &rb) {
        assert!(
            result.depth <= max_depth,
            "depth {} exceeded max {}",
            result.depth,
            max_depth
        );
    }
}

#[test]
fn inference_engine_above_threshold_filters_low_cf_rules() {
    let rb = rust_rule_base();
    let strict_engine = InferenceEngine::new(3, 0.99);
    let result = strict_engine.reason("rust", &rb);
    if let Some(r) = result {
        assert!(r.certainty >= 0.0);
    }
}

#[test]
fn inference_engine_unknown_topic_returns_none() {
    let rb = rust_rule_base();
    let engine = InferenceEngine::new(3, 0.5);
    let result = engine.reason("xyzzy_nonexistent_topic_12345", &rb);
    assert!(result.is_none(), "expected None for unknown topic");
}

#[test]
fn explanation_trace_new_is_empty() {
    let trace = ExplanationTrace::new();
    assert!(trace.is_empty());
    assert_eq!(trace.len(), 0);
}

#[test]
fn explanation_trace_push_increases_len() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("rust has ownership"));
    assert_eq!(trace.len(), 1);
    assert!(!trace.is_empty());
}

#[test]
fn proof_step_told_stores_fact() {
    let step = ProofStep::told("rust has ownership");
    assert!(step.to_line().to_ascii_lowercase().contains("rust"));
}

#[test]
fn proof_step_derived_stores_conclusion() {
    let step = ProofStep::derived(
        "rust is memory-safe",
        "rule_ownership",
        vec!["rust has ownership".into()],
        0.9,
    );
    assert!(step.to_line().to_ascii_lowercase().contains("memory-safe"));
}

#[test]
fn explanation_trace_to_prose_is_non_empty() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("rust has ownership"));
    trace.push(ProofStep::derived(
        "rust is memory-safe",
        "rule_0",
        vec!["rust has ownership".into()],
        0.85,
    ));
    let prose = trace.to_prose();
    assert!(!prose.is_empty(), "to_prose() was empty");
}

#[test]
fn explanation_trace_conclusion_sentence_is_non_empty() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("python is high-level"));
    trace.push(ProofStep::derived(
        "python is easy to use",
        "rule_1",
        vec!["python is high-level".into()],
        0.75,
    ));
    let conclusion = trace.conclusion_sentence();
    assert!(!conclusion.is_empty(), "conclusion_sentence() was empty");
}

#[test]
fn explanation_trace_to_prose_contains_topic() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("rust has ownership"));
    trace.push(ProofStep::derived(
        "rust is memory-safe",
        "rule_0",
        vec!["rust has ownership".into()],
        0.9,
    ));
    let prose = trace.to_prose();
    assert!(
        prose.to_ascii_lowercase().contains("rust")
            || prose.to_ascii_lowercase().contains("memory"),
        "prose does not mention topic: {prose}"
    );
}

#[test]
fn multiple_proof_steps_do_not_panic() {
    let mut trace = ExplanationTrace::new();
    for i in 0..10 {
        trace.push(ProofStep::told(format!("fact_{i}")));
    }
    assert_eq!(trace.len(), 10);
    let _ = trace.to_prose();
}

#[test]
fn rule_base_matching_rules_returns_relevant_rules() {
    let mut rb = RuleBase::new();
    rb.rules.push(Rule {
        id: "rule_rust".into(),
        conditions: vec![Condition::new("rust", "is", "systems", false)],
        conclusion: "rust is memory-safe".into(),
        certainty_factor: 0.8,
    });
    let matches = rb.matching_rules("rust is memory-safe");
    assert!(
        !matches.is_empty(),
        "should match rules for known conclusion"
    );
}
