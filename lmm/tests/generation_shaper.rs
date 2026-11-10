// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::shaper::ParagraphShaper;

fn sample_sentences(n: usize) -> Vec<String> {
    let pool = [
        "Rust is a systems programming language that emphasizes memory safety and performance.",
        "The borrow checker ensures that references never outlive the data they point to.",
        "Concurrency in Rust is designed to prevent data races at compile time.",
        "Pattern matching in Rust provides expressive and exhaustive control flow.",
        "Cargo is the official build system and package manager for the Rust ecosystem.",
        "Traits define shared behaviour across types without requiring inheritance.",
        "Generics allow writing reusable code that works across many concrete types.",
        "The standard library provides collections, I/O, threading, and synchronisation.",
        "Iterators in Rust are lazy and compose efficiently through adapter chains.",
        "Zero-cost abstractions ensure that high-level code compiles to optimal machine code.",
    ];
    pool.iter().cycle().take(n).map(|s| s.to_string()).collect()
}

#[test]
fn shaper_handles_empty_input() {
    let shaper = ParagraphShaper::new(2, 2, 5, 40, 120);
    let result = shaper.shape(&[], 42);
    assert!(result.is_empty());
}

#[test]
fn shaper_handles_single_sentence() {
    let shaper = ParagraphShaper::new(2, 2, 5, 40, 120);
    let sentences =
        vec!["Rust is a general-purpose systems programming language for safety.".to_string()];
    let result = shaper.shape(&sentences, 42);
    assert!(!result.is_empty());
    assert!(result.contains("Rust") || result.contains("rust"));
}

#[test]
fn shaper_respects_target_paragraph_count() {
    let shaper = ParagraphShaper::new(2, 2, 5, 40, 120);
    let sentences = sample_sentences(10);
    let result = shaper.shape(&sentences, 100);
    let paragraph_count = result
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .count();
    assert!(
        (1..=2).contains(&paragraph_count),
        "expected 1-2 paragraphs, got {paragraph_count}"
    );
}

#[test]
fn shaper_produces_at_least_one_paragraph() {
    let shaper = ParagraphShaper::new(3, 1, 4, 20, 100);
    let sentences = sample_sentences(5);
    let result = shaper.shape(&sentences, 777);
    let paragraph_count = result
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .count();
    assert!(paragraph_count >= 1, "must produce at least one paragraph");
}

#[test]
fn shaper_adds_discourse_connectors_between_paragraphs() {
    let shaper = ParagraphShaper::new(2, 1, 3, 15, 80);
    let sentences = sample_sentences(8);
    let result = shaper.shape(&sentences, 999);
    let paragraphs: Vec<&str> = result
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    if paragraphs.len() >= 2 {
        let second = paragraphs[1];
        let connectors = [
            "Furthermore,",
            "Additionally,",
            "Moreover,",
            "In addition,",
            "It follows that",
            "As a result,",
            "Notably,",
        ];
        let has_connector = connectors.iter().any(|c| second.starts_with(c));
        assert!(
            has_connector,
            "second paragraph should start with a discourse connector, got: {second}"
        );
    }
}

#[test]
fn shaper_capitalises_first_word_of_each_paragraph() {
    let shaper = ParagraphShaper::new(2, 2, 4, 30, 100);
    let sentences = sample_sentences(8);
    let result = shaper.shape(&sentences, 1234);
    for para in result.split("\n\n").filter(|p| !p.trim().is_empty()) {
        let first_char = para.trim().chars().next().unwrap_or('a');
        assert!(
            first_char.is_ascii_uppercase(),
            "paragraph should start with uppercase, got: '{first_char}'"
        );
    }
}

#[test]
fn shaper_separates_paragraphs_with_double_newline() {
    let shaper = ParagraphShaper::new(2, 2, 4, 30, 100);
    let sentences = sample_sentences(8);
    let result = shaper.shape(&sentences, 55);
    if result.contains('\n') {
        assert!(
            result.contains("\n\n"),
            "paragraphs must be separated by \\n\\n"
        );
    }
}

#[test]
fn shaper_varies_with_different_seeds() {
    let shaper = ParagraphShaper::new(2, 2, 5, 40, 120);
    let sentences = sample_sentences(10);
    let result_a = shaper.shape(&sentences, 1);
    let result_b = shaper.shape(&sentences, 999_999_999);
    assert_ne!(
        result_a, result_b,
        "different seeds should produce different discourse connectors"
    );
}

#[test]
fn shaper_new_clamps_min_to_one() {
    let shaper = ParagraphShaper::new(0, 0, 0, 0, 0);
    let sentences = sample_sentences(3);
    let result = shaper.shape(&sentences, 42);
    assert!(
        !result.is_empty(),
        "clamped shaper should still produce output"
    );
}

// Copyright 2026 Mahmoud Harmouch, licensed under MIT.
