// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::inference::InferenceEngine;
use lmm::expert::rule::{Condition, Rule, RuleBase};
use lmm::generation::coherence::CoherenceScorer;
use lmm::generation::prose::ProseBuilder;
use lmm::stochastic::StochasticEnhancer;

#[test]
fn integration_full_pipeline_rust_domain() {
    let raw_corpus = [
        "Rust is a systems programming language that prioritizes speed and memory safety.",
        "The compiler provides compile-time checks through the borrow checker.",
        "Rust enforces strict ownership semantics for all heap and stack resources.",
        "Concurrency in Rust guarantees freedom from data races at compile time.",
    ];

    let rule_base = RuleBase::from_chunks(&raw_corpus);
    assert!(
        !rule_base.rules.is_empty(),
        "Rules must be derived from corpus"
    );

    let engine = InferenceEngine::new(3, 0.15);
    let inference = engine.reason("rust language", &rule_base);
    assert!(
        inference.is_some(),
        "Inference should succeed with sufficient evidence"
    );

    let res = inference.unwrap();
    assert!(res.certainty >= 0.15);
    assert!(!res.trace.is_empty());

    let explanation_prose = res.trace.to_prose();
    assert!(!explanation_prose.is_empty());

    let builder = ProseBuilder::new(2, 4);
    let seed = 123456789;
    let prose = builder.build(&res, &raw_corpus, seed);
    assert!(!prose.is_empty());
    assert!(prose.to_lowercase().contains("rust"));

    let enhancer = StochasticEnhancer::new(0.05);
    let enhanced = enhancer.enhance_detailed(&prose);
    assert!(!enhanced.text.is_empty());
}

#[test]
fn integration_full_pipeline_physics_domain() {
    let raw_corpus = [
        "The harmonic oscillator is a fundamental physical system with restoring force.",
        "Differential equations describe the continuous state trajectory of the pendulum.",
        "The Hamiltonian operator governs the total energy evolution of conservative systems.",
        "Central difference approximations enable numerical evaluation of spatial gradients.",
    ];

    let rule_base = RuleBase::from_chunks(&raw_corpus);
    let engine = InferenceEngine::new(3, 0.20);
    let inference = engine.reason("oscillator physical system", &rule_base);

    assert!(inference.is_some());
    let res = inference.unwrap();
    let builder = ProseBuilder::new(2, 3);
    let prose = builder.build(&res, &raw_corpus, 98765);
    assert!(!prose.is_empty());
}

#[test]
fn integration_full_pipeline_machine_learning_domain() {
    let raw_corpus = [
        "Symbolic regression uses mathematical equations from numerical observations.",
        "Genetic programming uses tree-structured expression candidates across generations.",
        "Equation encoding uses discrete text strings into continuous representations.",
        "The neural operator uses circular convolution with spectral kernel weights.",
    ];

    let rule_base = RuleBase::from_chunks(&raw_corpus);
    let engine = InferenceEngine::new(3, 0.15);
    let inference = engine.reason("symbolic regression equations", &rule_base);

    assert!(inference.is_some());
    let res = inference.unwrap();
    assert!(
        res.conclusion.contains("symbolic")
            || res.conclusion.contains("regression")
            || res.conclusion.contains("equations"),
        "conclusion must relate to the symbolic regression domain: got '{}'",
        res.conclusion
    );
}

#[test]
fn integration_threshold_control_filters_output() {
    let raw_corpus = ["Python is an interpreted scripting language."];

    let rule_base = RuleBase::from_chunks(&raw_corpus);

    let low_threshold_engine = InferenceEngine::new(3, 0.15);
    let low_res = low_threshold_engine.reason("python scripting", &rule_base);
    assert!(
        low_res.is_some(),
        "Permissive threshold should yield conclusion"
    );

    let high_threshold_engine = InferenceEngine::new(3, 0.95);
    let high_res = high_threshold_engine.reason("python scripting", &rule_base);
    assert!(
        high_res.is_none(),
        "Strict threshold should reject weak evidence"
    );
}

#[test]
fn integration_depth_control_limits_derivation_steps() {
    let mut rule_base = RuleBase::new();

    let fact_alpha = Condition::new("alpha", "enables", "root", false);
    rule_base.facts.push(fact_alpha.clone());

    let r1 = Rule {
        id: "r1".to_string(),
        conditions: vec![fact_alpha],
        conclusion: "beta enables step".to_string(),
        certainty_factor: 0.80,
    };
    rule_base.rules.push(r1);

    let r2 = Rule {
        id: "r2".to_string(),
        conditions: vec![Condition::new("beta", "enables", "step", false)],
        conclusion: "gamma reaches goal".to_string(),
        certainty_factor: 0.80,
    };
    rule_base.rules.push(r2);

    let shallow_engine = InferenceEngine::new(1, 0.10);
    let deep_engine = InferenceEngine::new(4, 0.10);

    let shallow_res = shallow_engine.reason("goal", &rule_base);
    let deep_res = deep_engine.reason("goal", &rule_base);

    assert!(
        shallow_res.is_none(),
        "Shallow engine cannot reach 2 steps: got {:?}",
        shallow_res.map(|r| r.conclusion)
    );
    assert!(
        deep_res.is_some(),
        "Deep engine can traverse multi-step chains"
    );
    let d = deep_res.unwrap();
    assert!(
        d.depth >= 2,
        "Chain must span at least 2 rule applications, got depth={}",
        d.depth
    );
}

#[test]
fn integration_coherence_improves_sentence_flow() {
    let raw_corpus = [
        "Rust prevents memory corruption.",
        "Memory corruption leads to security vulnerabilities.",
        "Security vulnerabilities undermine software reliability.",
    ];

    let rule_base = RuleBase::from_chunks(&raw_corpus);
    let engine = InferenceEngine::new(3, 0.10);
    let inference = engine.reason("rust memory", &rule_base).unwrap();

    let builder = ProseBuilder::new(1, 4);
    let prose = builder.build(&inference, &raw_corpus, 42);

    let sentences: Vec<String> = prose
        .split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| s.split_whitespace().count() >= 3)
        .collect();

    let scorer = CoherenceScorer;
    let reordered = scorer.reorder(&sentences);
    assert_eq!(reordered.len(), sentences.len());
}

#[test]
fn integration_adversarial_gibberish_produces_clean_none() {
    let raw_corpus = [
        "!@#$% ^&*() ___ +++ === ~~~",
        "12389127391823719823",
        "xyz abc qrs wuv",
    ];

    let rule_base = RuleBase::from_chunks(&raw_corpus);
    let engine = InferenceEngine::new(3, 0.30);
    let result = engine.reason("test goal", &rule_base);
    assert!(result.is_none());
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
