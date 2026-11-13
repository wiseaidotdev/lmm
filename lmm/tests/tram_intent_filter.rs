// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, distributed
// except according to these terms.

use lmm::cognition::thought::classifier::ReasoningMode;
use lmm::tram::intent_filter::IntentFilter;

#[test]
fn test_score_chunk_evaluative() {
    let chunk = "Rust has many advantages including performance and safety benefits.";
    let score = IntentFilter::score_chunk(chunk, ReasoningMode::Evaluative);
    assert!(score > 0.5);
}

#[test]
fn test_score_chunk_causal() {
    let chunk =
        "Rust does not have a garbage collector because it uses ownership for memory management.";
    let score = IntentFilter::score_chunk(chunk, ReasoningMode::Causal);
    assert!(score > 0.5);
}

#[test]
fn test_score_chunk_analogical() {
    let chunk = "Compared to Python, Rust is more performant versus interpreted languages.";
    let score = IntentFilter::score_chunk(chunk, ReasoningMode::Analogical);
    assert!(score > 0.5);
}

#[test]
fn test_filter_and_rank() {
    let chunks = vec![
        "Rust is fast and efficient.".to_string(),
        "The advantage of Rust is memory safety.".to_string(),
        "Rust provides performance benefits.".to_string(),
    ];
    let ranked = IntentFilter::filter_and_rank(&chunks, ReasoningMode::Evaluative);
    assert_eq!(ranked.len(), 3);
    assert!(ranked[0].1 > ranked[1].1);
}

#[test]
fn test_select_top_chunks() {
    let chunks = vec![
        "Rust is fast.".to_string(),
        "The benefit of Rust is safety.".to_string(),
        "Rust has many advantages.".to_string(),
    ];
    let top = IntentFilter::select_top_chunks(&chunks, ReasoningMode::Evaluative, 2);
    assert_eq!(top.len(), 2);
}
