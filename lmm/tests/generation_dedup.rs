// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::dedup::{GlobalDeduplicator, jaccard, tokenise_for_jaccard};

#[test]
fn dedup_removes_exact_duplicates() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let sentences = vec![
        "Rust is a systems programming language that emphasizes memory safety.".to_string(),
        "Rust is a systems programming language that emphasizes memory safety.".to_string(),
    ];
    let result = deduplicator.deduplicate(sentences);
    assert_eq!(result.len(), 1);
}

#[test]
fn dedup_removes_near_duplicates_above_threshold() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let sentences = vec![
        "Rust is a general-purpose programming language emphasizing performance and memory safety."
            .to_string(),
        "Rust is a general-purpose programming language emphasizing memory safety and performance."
            .to_string(),
    ];
    let result = deduplicator.deduplicate(sentences);
    assert_eq!(result.len(), 1, "near-duplicate should be removed");
}

#[test]
fn dedup_preserves_distinct_sentences() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let sentences = vec![
        "Rust is a systems programming language that emphasizes memory safety and performance."
            .to_string(),
        "Python is a high-level scripting language known for its readable and concise syntax."
            .to_string(),
        "Photosynthesis converts sunlight into chemical energy stored in glucose molecules."
            .to_string(),
    ];
    let result = deduplicator.deduplicate(sentences);
    assert_eq!(
        result.len(),
        3,
        "three distinct sentences should all survive"
    );
}

#[test]
fn dedup_handles_empty_input() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let result = deduplicator.deduplicate(vec![]);
    assert!(result.is_empty());
}

#[test]
fn dedup_handles_single_sentence() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let sentences = vec!["Only one sentence exists in this collection of text.".to_string()];
    let result = deduplicator.deduplicate(sentences.clone());
    assert_eq!(result, sentences);
}

#[test]
fn dedup_jaccard_zero_for_disjoint_sets() {
    let a = tokenise_for_jaccard("alpha beta gamma delta");
    let b = tokenise_for_jaccard("epsilon zeta eta theta");
    let score = jaccard(&a, &b);
    assert!(
        (score - 0.0).abs() < 1e-9,
        "disjoint sets should have Jaccard = 0.0"
    );
}

#[test]
fn dedup_jaccard_one_for_identical_sets() {
    let a = tokenise_for_jaccard("rust is a systems programming language");
    let b = tokenise_for_jaccard("rust is a systems programming language");
    let score = jaccard(&a, &b);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "identical sets should have Jaccard = 1.0"
    );
}

#[test]
fn dedup_jaccard_between_zero_and_one_for_partial_overlap() {
    let a = tokenise_for_jaccard("rust is a programming language");
    let b = tokenise_for_jaccard("rust is a scripting language interpreter");
    let score = jaccard(&a, &b);
    assert!(score > 0.0 && score < 1.0, "partial overlap: {score}");
}

#[test]
fn dedup_text_removes_duplicate_sentences() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let text = "Rust is a systems programming language emphasizing memory safety and performance. \
                Python is a popular scripting language used widely for data science tasks. \
                Rust is a systems programming language emphasizing memory safety and performance.";
    let result = deduplicator.deduplicate_text(text);
    let count = result.split('.').filter(|s| !s.trim().is_empty()).count();
    assert!(
        count <= 2,
        "duplicate sentence should be removed: got {count}"
    );
}

#[test]
fn dedup_text_preserves_paragraph_structure() {
    let deduplicator = GlobalDeduplicator::new(0.68);
    let text = "Rust is a systems programming language emphasizing safety.\n\n\
                Python is a scripting language known for readability.";
    let result = deduplicator.deduplicate_text(text);
    assert!(!result.is_empty());
}

#[test]
fn tokenise_for_jaccard_lowercases_words() {
    let tokens = tokenise_for_jaccard("Rust IS A LANGUAGE");
    assert!(tokens.contains("rust"));
    assert!(tokens.contains("language"));
}

#[test]
fn tokenise_for_jaccard_strips_punctuation() {
    let tokens = tokenise_for_jaccard("Rust, the language.");
    assert!(tokens.contains("rust"));
    assert!(tokens.contains("language"));
    assert!(!tokens.contains("rust,"));
    assert!(!tokens.contains("language."));
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
