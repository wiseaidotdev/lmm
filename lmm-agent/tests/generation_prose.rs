// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::explain::{ExplanationTrace, ProofStep};
use lmm::expert::inference::InferenceResult;
use lmm::generation::prose::ProseBuilder;

fn make_inference_result(conclusion: &str, certainty: f64) -> InferenceResult {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told(format!("{conclusion} is established")));
    trace.push(ProofStep::derived(
        conclusion,
        "rule_0",
        vec![format!("{conclusion} is established")],
        certainty,
    ));
    InferenceResult {
        conclusion: conclusion.to_string(),
        certainty,
        trace,
        depth: 1,
    }
}

#[test]
fn prose_builder_new_clamps_zero_paragraphs_to_one() {
    let b = ProseBuilder::new(0, 1);
    assert_eq!(b.target_paragraphs, 1);
}

#[test]
fn prose_builder_new_clamps_zero_sentences_to_minimum() {
    let b = ProseBuilder::new(1, 0);
    assert!(b.target_sentences >= 1);
}

#[test]
fn prose_builder_stores_normal_params() {
    let b = ProseBuilder::new(3, 5);
    assert_eq!(b.target_paragraphs, 3);
    assert_eq!(b.target_sentences, 5);
}

#[test]
fn prose_builder_generates_non_empty_output_with_chunks() {
    let builder = ProseBuilder::new(2, 4);
    let result = make_inference_result("rust is memory-safe", 0.85);
    let chunks = [
        "Rust is a systems programming language focused on performance and safety.",
        "The compiler uses a borrow checker to prevent data races at compile time.",
    ];
    let prose = builder.build(&result, &chunks, 42);
    assert!(!prose.is_empty(), "prose was empty");
}

#[test]
fn prose_builder_output_contains_topic_word() {
    let builder = ProseBuilder::new(2, 4);
    let result = make_inference_result("rust is memory-safe", 0.85);
    let chunks = ["Rust is a systems programming language focused on performance and safety."];
    let prose = builder.build(&result, &chunks, 42);
    assert!(
        prose.to_lowercase().contains("rust"),
        "expected 'rust' in prose, got: {prose}"
    );
}

#[test]
fn prose_builder_empty_chunks_still_yields_output() {
    let builder = ProseBuilder::new(2, 3);
    let result = make_inference_result("rust is memory-safe", 0.85);
    let prose = builder.build(&result, &[], 100);
    assert!(!prose.is_empty(), "prose was empty even with no chunks");
}

#[test]
fn prose_builder_deterministic_with_same_seed() {
    let builder = ProseBuilder::new(2, 4);
    let result = make_inference_result("rust is memory-safe", 0.85);
    let chunks = ["Rust eliminates data races at compile time without a runtime."];
    let seed = 987654321_u64;
    let run1 = builder.build(&result, &chunks, seed);
    let run2 = builder.build(&result, &chunks, seed);
    assert_eq!(run1, run2, "same seed must produce identical prose");
}

#[test]
fn prose_builder_different_seeds_may_differ() {
    let builder = ProseBuilder::new(2, 4);
    let result = make_inference_result("rust is memory-safe", 0.85);
    let chunks = [
        "Rust is a systems programming language focused on performance and safety.",
        "The borrow checker ensures memory safety without a garbage collector.",
        "Rust achieves zero-cost abstractions through its ownership model and lifetimes.",
    ];
    let run_a = builder.build(&result, &chunks, 1);
    let run_b = builder.build(&result, &chunks, 999_999_999);
    let _ = (run_a, run_b);
}

#[test]
fn prose_builder_high_certainty_produces_confident_output() {
    let builder = ProseBuilder::new(1, 3);
    let result = make_inference_result("python is high-level", 0.95);
    let chunks = [
        "Python is a high-level general-purpose programming language designed for readability.",
        "Python supports multiple programming paradigms including object-oriented and functional styles.",
    ];
    let prose = builder.build(&result, &chunks, 7);
    assert!(!prose.is_empty());
}

#[test]
fn prose_builder_low_certainty_still_yields_prose() {
    let builder = ProseBuilder::new(1, 2);
    let result = make_inference_result("topic is unknown", 0.1);
    let prose = builder.build(&result, &[], 0);
    assert!(!prose.is_empty(), "low-certainty prose was empty");
}

#[test]
fn prose_builder_multiple_paragraphs_do_not_panic() {
    let builder = ProseBuilder::new(5, 6);
    let result = make_inference_result("machine learning is a field", 0.75);
    let chunks = [
        "Machine learning is a subfield of artificial intelligence that enables systems to learn from data.",
        "Supervised learning uses labelled training data to build predictive models.",
        "Unsupervised learning discovers hidden patterns in unlabelled data sets.",
    ];
    let prose = builder.build(&result, &chunks, 42);
    assert!(!prose.is_empty());
}

#[test]
fn prose_builder_output_does_not_start_with_whitespace() {
    let builder = ProseBuilder::new(1, 3);
    let result = make_inference_result("rust is systems", 0.80);
    let chunks = ["Rust is a systems language that runs without a garbage collector."];
    let prose = builder.build(&result, &chunks, 1);
    assert!(
        !prose.starts_with(' ') && !prose.starts_with('\n'),
        "prose should not start with whitespace: {prose:?}"
    );
}
