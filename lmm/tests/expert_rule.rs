// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::rule::{Condition, Rule, RuleBase};

#[test]
fn condition_new_normalizes_case() {
    let cond = Condition::new("Rust", "IS", "Fast", false);
    assert_eq!(cond.subject, "rust");
    assert_eq!(cond.predicate, "is");
    assert_eq!(cond.object, "fast");
    assert!(!cond.negated);
}

#[test]
fn condition_new_mixed_case_unicode() {
    let cond = Condition::new("PYTHON", "Uses", "GIL", false);
    assert_eq!(cond.subject, "python");
    assert_eq!(cond.predicate, "uses");
    assert_eq!(cond.object, "gil");
}

#[test]
fn condition_negated_flag_preserved() {
    let neg = Condition::new("rust", "has", "gc", true);
    assert!(neg.negated);
    let aff = Condition::new("rust", "has", "gc", false);
    assert!(!aff.negated);
}

#[test]
fn condition_to_text_affirmative() {
    let cond = Condition::new("rust", "is", "memory-safe", false);
    assert_eq!(cond.to_text(), "rust is memory-safe");
}

#[test]
fn condition_to_text_negated() {
    let cond = Condition::new("python", "requires", "manual compilation", true);
    assert_eq!(
        cond.to_text(),
        "python does not requires manual compilation"
    );
}

#[test]
fn condition_to_text_deduplicates_adjacent_repeated_words() {
    let cond = Condition::new("rust rust", "is", "fast fast", false);
    assert_eq!(cond.to_text(), "rust is fast");
}

#[test]
fn condition_to_text_non_adjacent_repeats_are_preserved() {
    let cond = Condition::new("alpha", "is", "alpha", false);
    assert_eq!(cond.to_text(), "alpha is alpha");
}

#[test]
fn condition_clone_produces_equal_value() {
    let cond = Condition::new("rust", "enables", "safety", false);
    let clone = cond.clone();
    assert_eq!(cond.subject, clone.subject);
    assert_eq!(cond.predicate, clone.predicate);
    assert_eq!(cond.object, clone.object);
    assert_eq!(cond.negated, clone.negated);
}

#[test]
fn condition_partial_eq_reflexive() {
    let c = Condition::new("rust", "is", "safe", false);
    assert_eq!(c, c.clone());
}

#[test]
fn condition_partial_eq_different_negation() {
    let aff = Condition::new("rust", "is", "safe", false);
    let neg = Condition::new("rust", "is", "safe", true);
    assert_ne!(aff, neg);
}

#[test]
fn condition_partial_eq_different_predicate() {
    let a = Condition::new("rust", "is", "fast", false);
    let b = Condition::new("rust", "provides", "fast", false);
    assert_ne!(a, b);
}

#[test]
fn rule_struct_initialization() {
    let cond = Condition::new("rust", "has", "ownership", false);
    let rule = Rule {
        id: "rule_test_1".to_string(),
        conditions: vec![cond],
        conclusion: "rust is memory-safe".to_string(),
        certainty_factor: 0.85,
    };
    assert_eq!(rule.id, "rule_test_1");
    assert_eq!(rule.conditions.len(), 1);
    assert_eq!(rule.conclusion, "rust is memory-safe");
    assert!((rule.certainty_factor - 0.85).abs() < 1e-9);
}

#[test]
fn rule_multiple_conditions_stored() {
    let r = Rule {
        id: "multi".to_string(),
        conditions: vec![
            Condition::new("rust", "has", "ownership", false),
            Condition::new("rust", "has", "borrow-checker", false),
        ],
        conclusion: "rust is memory-safe".to_string(),
        certainty_factor: 0.90,
    };
    assert_eq!(r.conditions.len(), 2);
}

#[test]
fn rule_zero_conditions_allowed() {
    let r = Rule {
        id: "axiom".to_string(),
        conditions: vec![],
        conclusion: "truth holds".to_string(),
        certainty_factor: 1.0,
    };
    assert!(r.conditions.is_empty());
}

#[test]
fn rule_base_default_is_empty() {
    let base = RuleBase::new();
    assert!(base.rules.is_empty());
    assert!(base.facts.is_empty());
}

#[test]
fn rule_base_default_trait_matches_new() {
    let via_new = RuleBase::new();
    let via_default = RuleBase::default();
    assert_eq!(via_new.rules.len(), via_default.rules.len());
    assert_eq!(via_new.facts.len(), via_default.facts.len());
}

#[test]
fn rule_base_from_chunks_empty_slice() {
    let base = RuleBase::from_chunks(&[]);
    assert!(base.rules.is_empty());
    assert!(base.facts.is_empty());
}

#[test]
fn rule_base_from_chunks_no_relation_verbs() {
    let chunks = [
        "12345 67890",
        "random nonverb words string without relations",
    ];
    let base = RuleBase::from_chunks(&chunks);
    assert!(base.rules.is_empty());
    assert!(base.facts.is_empty());
}

#[test]
fn rule_base_from_chunks_extracts_all_relation_verbs() {
    let sample = "Rust is systems language. C allows pointer arithmetic. Python supports dynamic typing. Go provides concurrency channels. Compiler guarantees memory safety. Engine uses central difference. Physics enables continuous modeling. Formula defines state vector. Model represents population dynamics. Matrix includes eigenvalues. Structure contains nodes. Process produces telemetry.";
    let chunks = [sample];
    let base = RuleBase::from_chunks(&chunks);
    assert!(!base.rules.is_empty());
    assert!(base.rules.iter().any(|r| r.conclusion.contains("is")));
    assert!(
        base.rules
            .iter()
            .any(|r| r.conclusion.contains("allows") || r.conclusion.contains("supports"))
    );
    assert!(
        base.rules
            .iter()
            .any(|r| r.conclusion.contains("provides") || r.conclusion.contains("guarantees"))
    );
}

#[test]
fn rule_base_from_chunks_respects_sentence_boundaries() {
    let chunks = ["Rust is fast. Python is dynamic."];
    let base = RuleBase::from_chunks(&chunks);
    assert!(base.rules.len() >= 2);
    assert!(base.rules.iter().any(|r| r.conclusion.contains("rust")));
    assert!(base.rules.iter().any(|r| r.conclusion.contains("python")));
}

#[test]
fn rule_base_from_chunks_colon_semicolon_separators() {
    let chunk = "Rust is memory-safe; Python is interpreted: Go provides concurrency.";
    let base = RuleBase::from_chunks(&[chunk]);
    assert!(base.rules.len() >= 2);
}

#[test]
fn rule_base_from_chunks_deduplicates_identical_conclusions() {
    let chunks = [
        "Rust provides memory safety.",
        "Rust provides memory safety.",
        "Rust provides memory safety.",
    ];
    let base = RuleBase::from_chunks(&chunks);
    let provides_count = base
        .rules
        .iter()
        .filter(|r| r.conclusion.contains("provides"))
        .count();
    assert_eq!(provides_count, 1);
}

#[test]
fn rule_base_from_chunks_facts_equal_rules_count() {
    let chunks = ["Rust is fast. Go provides concurrency."];
    let base = RuleBase::from_chunks(&chunks);
    assert_eq!(base.facts.len(), base.rules.len());
}

#[test]
fn rule_base_from_chunks_cf_scales_with_verb_count() {
    let dense = "Rust is fast. Rust provides safety. Rust guarantees concurrency. Rust uses ownership. Rust has borrow-checker.";
    let base = RuleBase::from_chunks(&[dense]);
    for rule in &base.rules {
        assert!(rule.certainty_factor >= 0.1);
        assert!(rule.certainty_factor <= 0.95);
    }
    let max_cf = base
        .rules
        .iter()
        .map(|r| r.certainty_factor)
        .fold(0.0_f64, f64::max);
    assert!((max_cf - 0.95).abs() < 1e-9);
}

#[test]
fn rule_base_from_chunks_single_verb_gives_min_cf() {
    let chunk = "Rust provides memory safety.";
    let base = RuleBase::from_chunks(&[chunk]);
    assert!(!base.rules.is_empty());
    for rule in &base.rules {
        assert!((rule.certainty_factor - 0.20).abs() < 1e-9);
    }
}

#[test]
fn rule_base_from_chunks_stop_words_excluded_from_subject() {
    let chunk = "The rust is fast.";
    let base = RuleBase::from_chunks(&[chunk]);
    for rule in &base.rules {
        assert!(!rule.conclusion.starts_with("the "));
    }
}

#[test]
fn rule_base_clone_is_independent() {
    let chunks = ["Rust is fast."];
    let base = RuleBase::from_chunks(&chunks);
    let mut clone = base.clone();
    clone.rules.clear();
    assert!(!base.rules.is_empty());
}

#[test]
fn rule_base_matching_rules_finds_relevant_conclusions() {
    let chunks = [
        "Rust is memory-safe through ownership semantics.",
        "Python is dynamically typed and interpreted.",
    ];
    let base = RuleBase::from_chunks(&chunks);
    let matched = base.matching_rules("rust ownership");
    assert!(!matched.is_empty());
    assert!(matched.iter().any(|r| r.conclusion.contains("rust")));
}

#[test]
fn rule_base_matching_rules_ignores_short_tokens() {
    let chunks = ["Rust is fast and reliable."];
    let base = RuleBase::from_chunks(&chunks);
    let matched = base.matching_rules("is in to on");
    assert!(matched.is_empty());
}

#[test]
fn rule_base_matching_rules_case_insensitive() {
    let chunks = ["Rust provides zero-cost abstractions."];
    let base = RuleBase::from_chunks(&chunks);
    let matched_lower = base.matching_rules("rust");
    let matched_upper = base.matching_rules("RUST");
    assert_eq!(matched_lower.len(), matched_upper.len());
}

#[test]
fn rule_base_matching_rules_no_match_returns_empty() {
    let chunks = ["Entropy measures thermodynamic disorder."];
    let base = RuleBase::from_chunks(&chunks);
    let matched = base.matching_rules("quantum entanglement superconducting");
    assert!(matched.is_empty());
}

#[test]
fn rule_base_matching_rules_empty_goal_returns_empty() {
    let chunks = ["Rust is fast."];
    let base = RuleBase::from_chunks(&chunks);
    let matched = base.matching_rules("");
    assert!(matched.is_empty());
}

#[test]
fn rule_base_matching_rules_multiple_rules_matched() {
    let chunks = ["Rust is fast.", "Rust provides safety."];
    let base = RuleBase::from_chunks(&chunks);
    let matched = base.matching_rules("rust");
    assert!(
        matched.len() >= 2,
        "both rust rules should match, got {}",
        matched.len()
    );
}

#[test]
fn rule_base_matching_rules_on_manually_built_base() {
    let mut base = RuleBase::new();
    base.rules.push(Rule {
        id: "r1".to_string(),
        conditions: vec![],
        conclusion: "quantum field theory".to_string(),
        certainty_factor: 0.9,
    });
    base.rules.push(Rule {
        id: "r2".to_string(),
        conditions: vec![],
        conclusion: "classical mechanics applies".to_string(),
        certainty_factor: 0.8,
    });
    let matched = base.matching_rules("quantum field");
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].id, "r1");
}

#[test]
fn rule_base_condition_holds_affirmative() {
    let chunks = ["Rust provides compile-time memory safety."];
    let base = RuleBase::from_chunks(&chunks);
    let affirmative = Condition::new("rust", "provides", "memory", false);
    assert!(base.condition_holds(&affirmative));

    let non_existent = Condition::new("rust", "provides", "garbage-collector", false);
    assert!(!base.condition_holds(&non_existent));
}

#[test]
fn rule_base_condition_holds_negated() {
    let chunks = ["Rust provides compile-time memory safety."];
    let base = RuleBase::from_chunks(&chunks);
    let negated_absent = Condition::new("rust", "has", "garbage-collection", true);
    assert!(base.condition_holds(&negated_absent));

    let negated_present = Condition::new("rust", "provides", "memory", true);
    assert!(!base.condition_holds(&negated_present));
}

#[test]
fn rule_base_condition_holds_substring_subject_match() {
    let mut base = RuleBase::new();
    base.facts
        .push(Condition::new("rust language", "is", "fast", false));
    let cond = Condition::new("rust", "is", "fast", false);
    assert!(base.condition_holds(&cond));
}

#[test]
fn rule_base_condition_holds_substring_object_match() {
    let mut base = RuleBase::new();
    base.facts
        .push(Condition::new("rust", "provides", "memory safety", false));
    let cond = Condition::new("rust", "provides", "safety", false);
    assert!(base.condition_holds(&cond));
}

#[test]
fn rule_base_condition_holds_predicate_must_match_exactly() {
    let mut base = RuleBase::new();
    base.facts.push(Condition::new("rust", "is", "fast", false));
    let cond = Condition::new("rust", "provides", "fast", false);
    assert!(!base.condition_holds(&cond));
}

#[test]
fn rule_base_condition_holds_empty_fact_base_is_always_false() {
    let base = RuleBase::new();
    let cond = Condition::new("anything", "is", "true", false);
    assert!(!base.condition_holds(&cond));
}

#[test]
fn rule_base_condition_holds_negated_in_empty_fact_base() {
    let base = RuleBase::new();
    let cond = Condition::new("rust", "has", "gc", true);
    assert!(base.condition_holds(&cond));
}

#[test]
fn rule_base_edge_case_only_stop_words() {
    let chunks = ["the and for with from into than when which"];
    let base = RuleBase::from_chunks(&chunks);
    assert!(base.rules.is_empty());
}

#[test]
fn rule_base_edge_case_unicode_characters() {
    let chunks = ["LMM 🧬 provides deterministic 🧮 equations for α and β."];
    let base = RuleBase::from_chunks(&chunks);
    assert!(!base.rules.is_empty());
}

#[test]
fn rule_base_edge_case_whitespace_only_chunk() {
    let base = RuleBase::from_chunks(&["   \t\n  "]);
    assert!(base.rules.is_empty());
}

#[test]
fn rule_base_edge_case_single_word_chunk() {
    let base = RuleBase::from_chunks(&["Rust."]);
    assert!(base.rules.is_empty());
}

#[test]
fn rule_base_edge_case_verb_at_start_no_subject_no_panic() {
    let _base = RuleBase::from_chunks(&["Is fast language."]);
}

#[test]
fn rule_base_edge_case_verb_at_end_no_object_no_panic() {
    let _base = RuleBase::from_chunks(&["Rust provides."]);
}

#[test]
fn rule_base_manual_push_then_query() {
    let mut base = RuleBase::new();
    let fact = Condition::new("gravity", "enables", "orbit", false);
    base.facts.push(fact.clone());
    assert!(base.condition_holds(&fact));
    assert!(!base.condition_holds(&Condition::new("gravity", "prevents", "orbit", false)));
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
