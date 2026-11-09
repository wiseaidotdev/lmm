// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::{Condition, Rule, RuleBase};

#[test]
fn inference_engine_new_clamps_threshold() {
    let eng_low = InferenceEngine::new(3, -0.5);
    let eng_high = InferenceEngine::new(3, 1.5);
    assert_eq!(eng_low.cf_threshold, 0.0);
    assert_eq!(eng_high.cf_threshold, 1.0);
    assert_eq!(eng_low.max_depth, 3);
}

#[test]
fn inference_empty_rule_base_returns_none() {
    let base = RuleBase::new();
    let engine = InferenceEngine::new(3, 0.30);
    let result = engine.reason("rust memory safety", &base);
    assert!(result.is_none());
}

#[test]
fn inference_no_matching_rules_returns_none() {
    let chunks = ["Entropy increases in isolated thermodynamic systems."];
    let base = RuleBase::from_chunks(&chunks);
    let engine = InferenceEngine::new(3, 0.30);
    let result = engine.reason("compiler borrow checker", &base);
    assert!(result.is_none());
}

#[test]
fn inference_single_step_direct_derivation() {
    let chunks = ["Rust provides compile-time memory safety guarantees."];
    let base = RuleBase::from_chunks(&chunks);
    let engine = InferenceEngine::new(3, 0.20);
    let result = engine.reason("rust memory safety", &base);
    assert!(result.is_some());
    let res = result.unwrap();
    assert!(res.certainty >= 0.20);
    assert!(res.depth >= 1);
    assert!(!res.trace.is_empty());
}

#[test]
fn inference_multi_step_backward_chaining() {
    let mut base = RuleBase::new();

    let fact_cond = Condition::new("rust", "has", "borrow-checker", false);
    base.facts.push(fact_cond.clone());

    let rule_sub = Rule {
        id: "rule_sub".to_string(),
        conditions: vec![fact_cond],
        conclusion: "rust enforces ownership".to_string(),
        certainty_factor: 0.90,
    };
    base.rules.push(rule_sub);

    let intermediate_cond = Condition::new("rust", "enforces", "ownership", false);
    let rule_main = Rule {
        id: "rule_main".to_string(),
        conditions: vec![intermediate_cond],
        conclusion: "rust guarantees safety".to_string(),
        certainty_factor: 0.85,
    };
    base.rules.push(rule_main);

    let engine = InferenceEngine::new(4, 0.30);
    let result = engine.reason("safety", &base);
    assert!(
        result.is_some(),
        "engine must find rust guarantees safety via two-step chain"
    );
    let res = result.unwrap();
    assert_eq!(res.conclusion, "rust guarantees safety");
    assert!(
        res.depth >= 2,
        "proof must span at least two rule applications"
    );
    assert!(res.certainty >= 0.30);
}

#[test]
fn inference_depth_limit_terminates_chain() {
    let mut base = RuleBase::new();

    let fact_cond = Condition::new("a", "is", "root", false);
    base.facts.push(fact_cond.clone());

    let r1 = Rule {
        id: "r1".to_string(),
        conditions: vec![fact_cond],
        conclusion: "b is intermediate".to_string(),
        certainty_factor: 0.90,
    };
    base.rules.push(r1);

    let r2 = Rule {
        id: "r2".to_string(),
        conditions: vec![Condition::new("b", "is", "intermediate", false)],
        conclusion: "c is goal".to_string(),
        certainty_factor: 0.90,
    };
    base.rules.push(r2);

    let engine_depth_1 = InferenceEngine::new(1, 0.10);
    let res_depth_1 = engine_depth_1.reason("c is goal", &base);
    assert!(res_depth_1.is_none());

    let engine_depth_3 = InferenceEngine::new(3, 0.10);
    let res_depth_3 = engine_depth_3.reason("c is goal", &base);
    assert!(res_depth_3.is_some());
}

#[test]
fn inference_handles_cyclic_rules_gracefully() {
    let mut base = RuleBase::new();

    let rule_a = Rule {
        id: "rule_a".to_string(),
        conditions: vec![Condition::new("node_b", "causes", "state", false)],
        conclusion: "node_a causes state".to_string(),
        certainty_factor: 0.90,
    };
    let rule_b = Rule {
        id: "rule_b".to_string(),
        conditions: vec![Condition::new("node_a", "causes", "state", false)],
        conclusion: "node_b causes state".to_string(),
        certainty_factor: 0.90,
    };
    base.rules.push(rule_a);
    base.rules.push(rule_b);

    let engine = InferenceEngine::new(3, 0.30);
    let result = engine.reason("node_a state", &base);
    assert!(result.is_none());
}

#[test]
fn inference_cf_threshold_filtering() {
    let chunks = ["Python supports scripting workflows."];
    let base = RuleBase::from_chunks(&chunks);

    let strict_engine = InferenceEngine::new(3, 0.99);
    let permissive_engine = InferenceEngine::new(3, 0.10);

    let strict_res = strict_engine.reason("python scripting", &base);
    let permissive_res = permissive_engine.reason("python scripting", &base);

    assert!(strict_res.is_none());
    assert!(permissive_res.is_some());
}

#[test]
fn inference_chooses_higher_certainty_rule() {
    let mut base = RuleBase::new();

    let fact_low = Condition::new("agent", "has", "weak_signal", false);
    base.facts.push(fact_low.clone());
    let fact_high = Condition::new("agent", "has", "strong_signal", false);
    base.facts.push(fact_high.clone());

    let rule_low = Rule {
        id: "rule_low".to_string(),
        conditions: vec![fact_low],
        conclusion: "agent detects target".to_string(),
        certainty_factor: 0.40,
    };
    let rule_high = Rule {
        id: "rule_high".to_string(),
        conditions: vec![fact_high],
        conclusion: "agent detects target".to_string(),
        certainty_factor: 0.85,
    };
    base.rules.push(rule_low);
    base.rules.push(rule_high);

    let engine = InferenceEngine::new(3, 0.20);
    let result = engine.reason("detects target", &base).unwrap();
    assert!((result.certainty - 0.85).abs() < 1e-9);
}

#[test]
fn inference_zero_depth_returns_none() {
    let chunks = ["Rust provides safety."];
    let base = RuleBase::from_chunks(&chunks);
    let engine = InferenceEngine::new(0, 0.10);
    assert!(engine.reason("rust safety", &base).is_none());
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
