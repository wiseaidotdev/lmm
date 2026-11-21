// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for phrase banks and [`PhrasesMixer`].

use lmm::constants::phrases::{
    ACADEMIC_HEDGES, ACADEMIC_OPENERS, CAUSAL_CONNECTORS, COMPARATIVE_PHRASES, CONCESSION_PHRASES,
    CONCLUSION_STARTERS, DISCOURSE_MARKERS, EXEMPLIFICATION_PHRASES, TRANSITION_PHRASES,
};
use lmm::generation::phrases_mixer::PhrasesMixer;

#[test]
fn all_phrase_banks_non_empty() {
    assert!(
        !ACADEMIC_OPENERS.is_empty(),
        "ACADEMIC_OPENERS must not be empty"
    );
    assert!(
        !DISCOURSE_MARKERS.is_empty(),
        "DISCOURSE_MARKERS must not be empty"
    );
    assert!(
        !TRANSITION_PHRASES.is_empty(),
        "TRANSITION_PHRASES must not be empty"
    );
    assert!(
        !CONCLUSION_STARTERS.is_empty(),
        "CONCLUSION_STARTERS must not be empty"
    );
    assert!(
        !ACADEMIC_HEDGES.is_empty(),
        "ACADEMIC_HEDGES must not be empty"
    );
    assert!(
        !CAUSAL_CONNECTORS.is_empty(),
        "CAUSAL_CONNECTORS must not be empty"
    );
    assert!(
        !CONCESSION_PHRASES.is_empty(),
        "CONCESSION_PHRASES must not be empty"
    );
    assert!(
        !EXEMPLIFICATION_PHRASES.is_empty(),
        "EXEMPLIFICATION_PHRASES must not be empty"
    );
    assert!(
        !COMPARATIVE_PHRASES.is_empty(),
        "COMPARATIVE_PHRASES must not be empty"
    );
}

#[test]
fn all_phrase_banks_have_sufficient_variety() {
    assert!(
        ACADEMIC_OPENERS.len() >= 20,
        "ACADEMIC_OPENERS should have at least 20 entries, got {}",
        ACADEMIC_OPENERS.len()
    );
    assert!(
        CAUSAL_CONNECTORS.len() >= 20,
        "CAUSAL_CONNECTORS should have at least 20 entries, got {}",
        CAUSAL_CONNECTORS.len()
    );
    assert!(
        CONCESSION_PHRASES.len() >= 20,
        "CONCESSION_PHRASES should have at least 20 entries, got {}",
        CONCESSION_PHRASES.len()
    );
    assert!(
        EXEMPLIFICATION_PHRASES.len() >= 10,
        "EXEMPLIFICATION_PHRASES should have at least 10 entries, got {}",
        EXEMPLIFICATION_PHRASES.len()
    );
    assert!(
        COMPARATIVE_PHRASES.len() >= 10,
        "COMPARATIVE_PHRASES should have at least 10 entries, got {}",
        COMPARATIVE_PHRASES.len()
    );
}

#[test]
fn no_duplicate_phrases_in_academic_openers() {
    let mut seen = std::collections::HashSet::new();
    for phrase in ACADEMIC_OPENERS {
        let lower = phrase.to_ascii_lowercase();
        assert!(
            seen.insert(lower.clone()),
            "Duplicate in ACADEMIC_OPENERS: {:?}",
            phrase
        );
    }
}

#[test]
fn no_duplicate_phrases_in_causal_connectors() {
    let mut seen = std::collections::HashSet::new();
    for phrase in CAUSAL_CONNECTORS {
        let lower = phrase.to_ascii_lowercase();
        assert!(
            seen.insert(lower.clone()),
            "Duplicate in CAUSAL_CONNECTORS: {:?}",
            phrase
        );
    }
}

#[test]
fn no_empty_phrases_in_any_bank() {
    let all_banks: &[&[&str]] = &[
        ACADEMIC_OPENERS,
        DISCOURSE_MARKERS,
        TRANSITION_PHRASES,
        CONCLUSION_STARTERS,
        ACADEMIC_HEDGES,
        CAUSAL_CONNECTORS,
        CONCESSION_PHRASES,
        EXEMPLIFICATION_PHRASES,
        COMPARATIVE_PHRASES,
    ];
    for bank in all_banks {
        for phrase in *bank {
            assert!(
                !phrase.trim().is_empty(),
                "Found empty/whitespace-only phrase in a bank"
            );
        }
    }
}

#[test]
fn phrase_mixer_opener_returns_non_empty() {
    let mixer = PhrasesMixer::global();
    for seed in [0u64, 1, 7, 42, 999, u64::MAX] {
        let phrase = mixer.opener(seed);
        assert!(!phrase.is_empty(), "opener({seed}) returned empty string");
    }
}

#[test]
fn phrase_mixer_selects_different_phrases_for_different_seeds() {
    let mixer = PhrasesMixer::global();
    let phrases: Vec<&str> = (0u64..10).map(|s| mixer.opener(s)).collect();
    let unique: std::collections::HashSet<&&str> = phrases.iter().collect();
    assert!(
        unique.len() > 1,
        "PhrasesMixer should return different openers for different seeds"
    );
}

#[test]
fn phrase_mixer_all_accessors_return_non_empty() {
    let mixer = PhrasesMixer::global();
    let seed = 13u64;
    assert!(!mixer.opener(seed).is_empty());
    assert!(!mixer.marker(seed).is_empty());
    assert!(!mixer.transition(seed).is_empty());
    assert!(!mixer.conclusion(seed).is_empty());
    assert!(!mixer.hedge(seed).is_empty());
    assert!(!mixer.causal(seed).is_empty());
    assert!(!mixer.concession(seed).is_empty());
    assert!(!mixer.exemplification(seed).is_empty());
    assert!(!mixer.comparative(seed).is_empty());
}

#[test]
fn phrase_mixer_enrich_paragraph_adds_length() {
    let mixer = PhrasesMixer::global();
    let text = "Rust is fast. It is also memory-safe.";
    let enriched = mixer.enrich_paragraph(text, 0);
    assert!(
        enriched.len() >= text.len(),
        "enrich_paragraph should not shorten text"
    );
}

#[test]
fn phrase_mixer_apply_opener_if_needed_adds_prefix() {
    let mixer = PhrasesMixer::global();
    let plain = "rust provides memory safety without a garbage collector.";
    let wrapped = mixer.apply_opener_if_needed(plain, 2);
    assert!(
        wrapped.len() > plain.len(),
        "apply_opener_if_needed should wrap plain text with an opener"
    );
}

#[test]
fn phrase_mixer_apply_opener_idempotent_for_already_opened_text() {
    let mixer = PhrasesMixer::global();
    let opener = mixer.opener(0);
    let already_opened = format!("{} rust is a systems programming language.", opener);
    let result = mixer.apply_opener_if_needed(&already_opened, 0);
    assert_eq!(
        result, already_opened,
        "apply_opener_if_needed should not double-wrap already-opened text"
    );
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
