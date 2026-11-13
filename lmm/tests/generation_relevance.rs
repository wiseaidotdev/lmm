// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::generation::relevance::RelevanceScorer;

#[test]
fn test_extract_terms() {
    let text = "Rust is a fast and safe programming language.";
    let terms = RelevanceScorer::extract_terms(text);
    assert!(terms.contains(&"rust".to_string()));
    assert!(terms.contains(&"fast".to_string()));
    assert!(terms.contains(&"safe".to_string()));
}

#[test]
fn test_term_frequency() {
    let terms = vec!["rust".to_string(), "fast".to_string(), "rust".to_string()];
    let tf = RelevanceScorer::term_frequency(&terms);
    assert_eq!(tf.get("rust"), Some(&0.6666666666666666));
}

#[test]
fn test_score_sentence() {
    let sentence = "Rust provides performance advantages and memory safety benefits.";
    let query = "advantages of rust";
    let score = RelevanceScorer::score_sentence(
        sentence,
        &RelevanceScorer::extract_terms(query),
        ReasoningMode::Evaluative,
    );
    assert!(score > 0.0);
}

#[test]
fn test_rank_sentences() {
    let sentences = vec![
        "Rust is a programming language.".to_string(),
        "The advantage of Rust is memory safety.".to_string(),
        "Rust provides many benefits.".to_string(),
    ];
    let query = "advantages of rust";
    let ranked = RelevanceScorer::rank_sentences(&sentences, query, ReasoningMode::Evaluative);
    assert_eq!(ranked.len(), 3);
    assert!(ranked[0].1 >= ranked[1].1);
}

#[test]
fn test_select_top_sentences() {
    let sentences = vec![
        "Rust is fast.".to_string(),
        "The benefit of Rust is safety.".to_string(),
        "Rust has advantages.".to_string(),
    ];
    let top = RelevanceScorer::select_top_sentences(
        &sentences,
        "advantages of rust",
        ReasoningMode::Evaluative,
        2,
    );
    assert_eq!(top.len(), 2);
}
