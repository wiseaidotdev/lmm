// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::expert::explain::{ExplanationTrace, ProofStep};
use lmm::expert::inference::InferenceResult;
use lmm::generation::prose::ProseBuilder;

fn sample_inference_result() -> InferenceResult {
    let mut trace = ExplanationTrace::new();
    trace.push(ProofStep::told("rust has ownership"));
    trace.push(ProofStep::derived(
        "rust is memory-safe",
        "rule_ownership",
        vec!["rust has ownership".to_string()],
        0.85,
    ));

    InferenceResult {
        conclusion: "rust is memory-safe".to_string(),
        certainty: 0.85,
        trace,
        depth: 1,
    }
}

#[test]
fn prose_builder_new_clamps_parameters() {
    let builder_zero = ProseBuilder::new(0, 1);
    assert_eq!(builder_zero.target_paragraphs, 1);
    assert_eq!(builder_zero.target_sentences, 2);

    let builder_normal = ProseBuilder::new(3, 5);
    assert_eq!(builder_normal.target_paragraphs, 3);
    assert_eq!(builder_normal.target_sentences, 5);
}

#[test]
fn prose_builder_generates_grounded_prose() {
    let builder = ProseBuilder::new(2, 4);
    let result = sample_inference_result();
    let chunks = [
        "Rust is a systems programming language focused on performance and safety.",
        "The compiler uses a borrow checker to prevent data races.",
    ];
    let prose = builder.build(&result, &chunks, 42);

    assert!(!prose.is_empty());
    assert!(prose.to_lowercase().contains("rust"));
    assert!(
        prose.to_lowercase().contains("memory-safe") || prose.to_lowercase().contains("safety")
    );
}

#[test]
fn prose_builder_empty_chunks_still_yields_output() {
    let builder = ProseBuilder::new(2, 3);
    let result = sample_inference_result();
    let prose = builder.build(&result, &[], 100);

    assert!(!prose.is_empty());
    assert!(prose.to_lowercase().contains("rust"));
}

#[test]
fn prose_builder_deterministic_with_same_seed() {
    let builder = ProseBuilder::new(2, 4);
    let result = sample_inference_result();
    let chunks = ["Rust eliminates data races at compile time."];
    let seed = 987654321;

    let run1 = builder.build(&result, &chunks, seed);
    let run2 = builder.build(&result, &chunks, seed);

    assert_eq!(run1, run2, "Identical seeds must produce identical prose");
}

#[test]
fn prose_builder_varies_with_different_seeds() {
    let builder = ProseBuilder::new(3, 5);
    let result = sample_inference_result();
    let chunks = [
        "Rust enforces strict memory isolation.",
        "Ownership semantics guarantee thread safety.",
        "Zero-cost abstractions ensure native performance.",
    ];

    let run1 = builder.build(&result, &chunks, 11111);
    let run2 = builder.build(&result, &chunks, 99999);

    assert_ne!(
        run1, run2,
        "Different seeds should produce structurally different prose"
    );
}

#[test]
fn prose_builder_handles_non_relational_conclusion() {
    let builder = ProseBuilder::new(1, 3);
    let result = InferenceResult {
        conclusion: "unstructured concept statement".to_string(),
        certainty: 0.50,
        trace: ExplanationTrace::new(),
        depth: 0,
    };
    let prose = builder.build(&result, &[], 777);
    assert!(!prose.is_empty());
}

#[test]
fn prose_builder_edge_case_seed_boundaries() {
    let builder = ProseBuilder::new(2, 4);
    let result = sample_inference_result();
    let chunks = ["Mathematics governs physical dynamics."];

    let prose_zero = builder.build(&result, &chunks, 0);
    let prose_max = builder.build(&result, &chunks, u64::MAX);

    assert!(!prose_zero.is_empty());
    assert!(!prose_max.is_empty());
}

#[test]
fn prose_builder_capitalises_sentences() {
    let builder = ProseBuilder::new(1, 3);
    let result = sample_inference_result();
    let prose = builder.build(&result, &[], 555);

    let first_char = prose.chars().next().unwrap();
    assert!(
        first_char.is_uppercase(),
        "Prose must start with an uppercase letter"
    );
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
