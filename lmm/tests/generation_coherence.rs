// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::coherence::CoherenceScorer;

#[test]
fn coherence_reorder_empty_slice() {
    let scorer = CoherenceScorer;
    let result = scorer.reorder(&[]);
    assert!(result.is_empty());
}

#[test]
fn coherence_reorder_single_sentence() {
    let scorer = CoherenceScorer;
    let input = vec!["Rust manages memory with ownership.".to_string()];
    let result = scorer.reorder(&input);
    assert_eq!(result, input);
}

#[test]
fn coherence_reorder_two_sentences_preserves_count() {
    let scorer = CoherenceScorer;
    let input = vec![
        "First sentence discusses ownership.".to_string(),
        "Second sentence discusses concurrency.".to_string(),
    ];
    let result = scorer.reorder(&input);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&input[0]));
    assert!(result.contains(&input[1]));
}

#[test]
fn coherence_reorder_groups_topically_related_sentences() {
    let scorer = CoherenceScorer;
    let unrelated = "Apples and oranges grow in orchards.".to_string();
    let rust_mem = "Rust enforces memory safety at compile time.".to_string();
    let rust_owner = "Memory safety in Rust relies on the ownership model.".to_string();

    let input = vec![unrelated.clone(), rust_mem.clone(), rust_owner.clone()];
    let reordered = scorer.reorder(&input);

    assert_eq!(reordered.len(), 3);
    let idx_mem = reordered.iter().position(|s| s == &rust_mem).unwrap();
    let idx_owner = reordered.iter().position(|s| s == &rust_owner).unwrap();
    let distance = (idx_mem as isize - idx_owner as isize).abs();
    assert_eq!(
        distance, 1,
        "Topically overlapping sentences should be adjacent"
    );
}

#[test]
fn coherence_reorder_preserves_order_above_twelve_sentences() {
    let scorer = CoherenceScorer;
    let input: Vec<String> = (0..13)
        .map(|i| format!("Sentence number {i} with some unique content."))
        .collect();
    let result = scorer.reorder(&input);
    assert_eq!(
        result, input,
        "Above 12 sentences, order should be preserved"
    );
}

#[test]
fn coherence_reorder_case_insensitive_token_matching() {
    let scorer = CoherenceScorer;
    let s1 = "PYTHON utilizes dynamic typing.".to_string();
    let s2 = "python interpreter parses source code.".to_string();
    let s3 = "Random unrelated statement.".to_string();

    let input = vec![s3.clone(), s1.clone(), s2.clone()];
    let reordered = scorer.reorder(&input);

    let idx1 = reordered.iter().position(|s| s == &s1).unwrap();
    let idx2 = reordered.iter().position(|s| s == &s2).unwrap();
    assert_eq!((idx1 as isize - idx2 as isize).abs(), 1);
}

#[test]
fn coherence_reorder_no_content_overlap() {
    let scorer = CoherenceScorer;
    let input = vec![
        "Alpha beta gamma delta.".to_string(),
        "Zeta theta iota kappa.".to_string(),
        "Mu nu xi omicron.".to_string(),
    ];
    let reordered = scorer.reorder(&input);
    assert_eq!(reordered.len(), 3);
    for item in &input {
        assert!(reordered.contains(item));
    }
}

#[test]
fn coherence_reorder_handles_punctuation_variations() {
    let scorer = CoherenceScorer;
    let input = vec![
        "Rust, memory: safety!".to_string(),
        "Safety? Memory; Rust.".to_string(),
    ];
    let reordered = scorer.reorder(&input);
    assert_eq!(reordered.len(), 2);
}

#[test]
fn coherence_reorder_identical_sentences() {
    let scorer = CoherenceScorer;
    let input = vec![
        "Identical sentence about mathematics.".to_string(),
        "Identical sentence about mathematics.".to_string(),
    ];
    let reordered = scorer.reorder(&input);
    assert_eq!(reordered.len(), 2);
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
