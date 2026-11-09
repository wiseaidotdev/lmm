// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::explain::{ExplanationTrace, ProofStep};

#[test]
fn proof_step_told_initialization() {
    let step = ProofStep::told("rust has ownership");
    assert_eq!(step.conclusion, "rust has ownership");
    assert!(step.derived_by.is_none());
    assert!(step.from_conditions.is_empty());
    assert_eq!(step.certainty_factor, 1.0);
}

#[test]
fn proof_step_told_empty_string() {
    let step = ProofStep::told("");
    assert_eq!(step.conclusion, "");
    assert!(step.derived_by.is_none());
}

#[test]
fn proof_step_told_unicode_content() {
    let step = ProofStep::told("entropy ΔS ≥ 0 always");
    assert_eq!(step.conclusion, "entropy ΔS ≥ 0 always");
}

#[test]
fn proof_step_to_line_told() {
    let step = ProofStep::told("rust has borrow checker");
    let line = step.to_line();
    assert_eq!(line, "rust has borrow checker (observed)");
}

#[test]
fn proof_step_told_cf_is_one() {
    let step = ProofStep::told("known fact");
    assert!((step.certainty_factor - 1.0).abs() < 1e-12);
}

#[test]
fn proof_step_derived_initialization() {
    let step = ProofStep::derived(
        "rust is memory-safe",
        "rule_ownership",
        vec!["rust has ownership".to_string()],
        0.85,
    );
    assert_eq!(step.conclusion, "rust is memory-safe");
    assert_eq!(step.derived_by.as_deref(), Some("rule_ownership"));
    assert_eq!(step.from_conditions.len(), 1);
    assert!((step.certainty_factor - 0.85).abs() < 1e-9);
}

#[test]
fn proof_step_derived_empty_conditions() {
    let step = ProofStep::derived("conclusion", "rule_x", vec![], 0.60);
    assert!(step.from_conditions.is_empty());
    assert_eq!(step.derived_by.as_deref(), Some("rule_x"));
}

#[test]
fn proof_step_derived_multiple_conditions() {
    let step = ProofStep::derived(
        "complex conclusion",
        "rule_complex",
        vec![
            "cond_a".to_string(),
            "cond_b".to_string(),
            "cond_c".to_string(),
        ],
        0.70,
    );
    assert_eq!(step.from_conditions.len(), 3);
}

#[test]
fn proof_step_to_line_derived() {
    let step = ProofStep::derived(
        "rust is safe",
        "rule_safety",
        vec!["condition".to_string()],
        0.75,
    );
    let line = step.to_line();
    assert!(line.contains("rust is safe"));
    assert!(line.contains("rule_safety"));
    assert!(line.contains("0.75"));
    assert!(!line.contains("(observed)"));
}

#[test]
fn proof_step_derived_cf_zero() {
    let step = ProofStep::derived("weak conclusion", "rule_weak", vec![], 0.0);
    assert_eq!(step.certainty_factor, 0.0);
    let line = step.to_line();
    assert!(line.contains("0.00"));
}

#[test]
fn proof_step_clone_independent() {
    let step = ProofStep::told("fact");
    let clone = step.clone();
    assert_eq!(step.conclusion, clone.conclusion);
    assert_eq!(step.derived_by, clone.derived_by);
}

#[test]
fn explanation_trace_empty_state() {
    let trace = ExplanationTrace::new();
    assert!(trace.is_empty());
    assert_eq!(trace.len(), 0);
    assert_eq!(trace.conclusion_sentence(), "");
    assert_eq!(trace.to_prose(), "");
}

#[test]
fn explanation_trace_default_matches_new() {
    let via_new = ExplanationTrace::new();
    let via_default = ExplanationTrace::default();
    assert_eq!(via_new.len(), via_default.len());
}

#[test]
fn explanation_trace_push_and_length() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("fact 1"));
    trace.push(ProofStep::told("fact 2"));
    assert_eq!(trace.len(), 2);
    assert!(!trace.is_empty());
}

#[test]
fn explanation_trace_push_one_then_is_not_empty() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("only fact"));
    assert!(!trace.is_empty());
    assert_eq!(trace.len(), 1);
}

#[test]
fn explanation_trace_conclusion_sentence_returns_last() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("precondition"));
    trace.push(ProofStep::derived(
        "final conclusion",
        "rule_1",
        vec!["precondition".to_string()],
        0.90,
    ));
    assert_eq!(trace.conclusion_sentence(), "final conclusion");
}

#[test]
fn explanation_trace_conclusion_sentence_single_told_step() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("the only fact"));
    assert_eq!(trace.conclusion_sentence(), "the only fact");
}

#[test]
fn explanation_trace_conclusion_sentence_on_empty_returns_empty() {
    let trace = ExplanationTrace::new();
    assert_eq!(trace.conclusion_sentence(), "");
}

#[test]
fn explanation_trace_to_prose_single_fact() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("entropy increases"));
    let prose = trace.to_prose();
    assert!(prose.contains("entropy increases"));
    assert!(prose.ends_with('.'));
}

#[test]
fn explanation_trace_to_prose_two_facts_oxford_comma() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("mass attracts mass"));
    trace.push(ProofStep::told("spacetime curves"));
    let prose = trace.to_prose();
    assert!(prose.contains("mass attracts mass, and spacetime curves"));
}

#[test]
fn explanation_trace_to_prose_three_facts_comma_separated() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("fact alpha"));
    trace.push(ProofStep::told("fact beta"));
    trace.push(ProofStep::told("fact gamma"));
    let prose = trace.to_prose();
    assert!(prose.contains("alpha"));
    assert!(prose.contains("beta"));
    assert!(prose.contains("gamma"));
    assert!(prose.ends_with('.'));
}

#[test]
fn explanation_trace_to_prose_only_told_step_no_derived_section() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("the final observation"));
    let prose = trace.to_prose();
    assert!(prose.contains("final observation"));
}

#[test]
fn explanation_trace_to_prose_with_derived_rules() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("rust has ownership"));
    trace.push(ProofStep::derived(
        "rust is memory-safe",
        "rule_mem_safe",
        vec!["rust has ownership".to_string()],
        0.82,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("Based on observed knowledge"));
    assert!(prose.contains("rule_mem_safe establishes that rust is memory-safe"));
    assert!(prose.contains("CF=0.82"));
}

#[test]
fn explanation_trace_to_prose_confidence_labels() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::derived(
        "test statement",
        "rule_alpha",
        vec![],
        0.92,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("very high confidence"));
}

#[test]
fn explanation_trace_to_prose_confidence_label_moderate() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::derived(
        "moderate conclusion",
        "rule_m",
        vec![],
        0.45,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("moderate confidence"));
}

#[test]
fn explanation_trace_to_prose_confidence_label_low() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::derived(
        "weak conclusion",
        "rule_w",
        vec![],
        0.15,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("low confidence"));
}

#[test]
fn explanation_trace_to_prose_confidence_label_high() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::derived(
        "strong conclusion",
        "rule_s",
        vec![],
        0.75,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("high confidence"));
}

#[test]
fn explanation_trace_to_prose_multiple_derived_rules() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("root fact"));
    trace.push(ProofStep::derived(
        "intermediate",
        "rule_one",
        vec!["root fact".to_string()],
        0.80,
    ));
    trace.push(ProofStep::derived(
        "final",
        "rule_two",
        vec!["intermediate".to_string()],
        0.70,
    ));
    let prose = trace.to_prose();
    assert!(prose.contains("rule_one"));
    assert!(prose.contains("rule_two"));
}

#[test]
fn explanation_trace_to_prose_only_derived_no_facts() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::derived("derived only", "rule_d", vec![], 0.60));
    let prose = trace.to_prose();
    assert!(!prose.is_empty());
    assert!(prose.ends_with('.'));
    assert!(!prose.contains("Based on observed knowledge"));
}

#[test]
fn explanation_trace_to_prose_ends_with_period() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("some fact"));
    assert!(trace.to_prose().ends_with('.'));

    let mut trace2 = ExplanationTrace::new();
    trace2.push(ProofStep::derived("derived", "rule", vec![], 0.5));
    assert!(trace2.to_prose().ends_with('.'));
}

#[test]
fn explanation_trace_clone_is_independent() {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("original fact"));
    let mut clone = trace.clone();
    clone.push(ProofStep::told("extra fact in clone"));
    assert_eq!(trace.len(), 1);
    assert_eq!(clone.len(), 2);
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
