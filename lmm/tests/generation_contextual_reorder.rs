// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::generation::contextual_reorder::ContextualReorder;

#[test]
fn test_reorder_evaluative() {
    let sentences = vec![
        "Rust has many advantages.".to_string(),
        "The disadvantage of Rust is steep learning curve.".to_string(),
        "Rust is a systems language.".to_string(),
    ];
    let reordered =
        ContextualReorder::reorder(&sentences, "advantages of rust", ReasoningMode::Evaluative);
    assert_eq!(reordered.len(), 3);
}

#[test]
fn test_reorder_causal() {
    let sentences = vec![
        "Rust is fast because it has no garbage collector.".to_string(),
        "Rust uses ownership for memory management.".to_string(),
        "This design prevents data races.".to_string(),
    ];
    let reordered =
        ContextualReorder::reorder(&sentences, "why rust is fast", ReasoningMode::Causal);
    assert_eq!(reordered.len(), 3);
}

#[test]
fn test_reorder_analogical() {
    let sentences = vec![
        "Rust is faster than Python.".to_string(),
        "Python is easier to learn.".to_string(),
        "Both are programming languages.".to_string(),
    ];
    let reordered =
        ContextualReorder::reorder(&sentences, "rust vs python", ReasoningMode::Analogical);
    assert_eq!(reordered.len(), 3);
}

#[test]
fn test_extract_entities() {
    let entities = ContextualReorder::extract_entities("rust vs python");
    assert_eq!(entities.len(), 2);
    assert!(entities.contains(&"rust".to_string()));
    assert!(entities.contains(&"python".to_string()));
}
