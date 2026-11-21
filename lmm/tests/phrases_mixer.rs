// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::generation::phrases_mixer::PhrasesMixer;

#[test]
fn all_phrase_banks_nonempty() {
    let m = PhrasesMixer::global();
    assert!(
        !m.opener(0).is_empty(),
        "ACADEMIC_OPENERS must not be empty"
    );
    assert!(
        !m.marker(0).is_empty(),
        "DISCOURSE_MARKERS must not be empty"
    );
    assert!(
        !m.transition(0).is_empty(),
        "TRANSITION_PHRASES must not be empty"
    );
    assert!(
        !m.conclusion(0).is_empty(),
        "CONCLUSION_STARTERS must not be empty"
    );
    assert!(!m.hedge(0).is_empty(), "ACADEMIC_HEDGES must not be empty");
}

#[test]
fn opener_varies_with_seed() {
    let m = PhrasesMixer::global();
    let results: Vec<&str> = (0..20).map(|i| m.opener(i)).collect();
    let unique: std::collections::HashSet<&&str> = results.iter().collect();
    assert!(
        unique.len() > 1,
        "Different seeds should return different openers (got {:?})",
        unique
    );
}

#[test]
fn marker_is_nonempty() {
    let m = PhrasesMixer::global();
    for seed in 0u64..10 {
        let marker = m.marker(seed);
        assert!(!marker.is_empty(), "marker({}) returned empty string", seed);
    }
}

#[test]
fn enrich_paragraph_adds_marker() {
    let m = PhrasesMixer::global();
    let text = "Rust is fast. It is also memory-safe. No garbage collector is needed.";
    let enriched = m.enrich_paragraph(text, 0);
    assert!(
        enriched.len() >= text.len(),
        "Enriched text should be at least as long as input"
    );
}

#[test]
fn enrich_paragraph_single_sentence_unchanged() {
    let m = PhrasesMixer::global();
    let text = "Rust is fast.";
    let enriched = m.enrich_paragraph(text, 0);
    assert_eq!(
        enriched, text,
        "Single-sentence paragraph should be returned unchanged"
    );
}

#[test]
fn apply_opener_wraps_plain_text() {
    let m = PhrasesMixer::global();
    let plain = "rust provides memory safety without a garbage collector.";
    let wrapped = m.apply_opener_if_needed(plain, 2);
    assert!(
        wrapped.len() > plain.len(),
        "Wrapped text must be longer than the original"
    );
}

#[test]
fn apply_opener_does_not_double_wrap() {
    let m = PhrasesMixer::global();
    let already_opened = "It is widely acknowledged that rust is fast.";
    let result = m.apply_opener_if_needed(already_opened, 0);
    assert_eq!(
        result, already_opened,
        "Text already starting with an opener must not be re-wrapped"
    );
}

#[test]
fn global_singleton_is_stable() {
    let a = PhrasesMixer::global();
    let b = PhrasesMixer::global();
    assert_eq!(
        std::ptr::addr_of!(*a),
        std::ptr::addr_of!(*b),
        "global() must return the same instance"
    );
}

#[test]
fn new_causal_accessor_returns_non_empty() {
    let m = PhrasesMixer::global();
    for seed in [0u64, 1, 7, 42] {
        assert!(
            !m.causal(seed).is_empty(),
            "causal({seed}) must not be empty"
        );
    }
}

#[test]
fn new_concession_accessor_returns_non_empty() {
    let m = PhrasesMixer::global();
    for seed in [0u64, 1, 7, 42] {
        assert!(
            !m.concession(seed).is_empty(),
            "concession({seed}) must not be empty"
        );
    }
}

#[test]
fn new_exemplification_accessor_returns_non_empty() {
    let m = PhrasesMixer::global();
    for seed in [0u64, 1, 7, 42] {
        assert!(
            !m.exemplification(seed).is_empty(),
            "exemplification({seed}) must not be empty"
        );
    }
}

#[test]
fn new_comparative_accessor_returns_non_empty() {
    let m = PhrasesMixer::global();
    for seed in [0u64, 1, 7, 42] {
        assert!(
            !m.comparative(seed).is_empty(),
            "comparative({seed}) must not be empty"
        );
    }
}

#[test]
fn new_accessors_vary_with_seed() {
    let m = PhrasesMixer::global();
    let causals: Vec<&str> = (0u64..10).map(|s| m.causal(s)).collect();
    let unique: std::collections::HashSet<&&str> = causals.iter().collect();
    assert!(unique.len() > 1, "causal() should vary across seeds");

    let comparatives: Vec<&str> = (0u64..10).map(|s| m.comparative(s)).collect();
    let unique2: std::collections::HashSet<&&str> = comparatives.iter().collect();
    assert!(unique2.len() > 1, "comparative() should vary across seeds");
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
