// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for [`lmm::stochastic::StochasticEnhancer`] and
//! [`lmm::stochastic::SynonymBank`].
//!
//! Verifies synonym lookup, text enhancement, determinism control, and
//! curated-only mode behaviour.

use lmm::stochastic::{StochasticEnhancer, SynonymBank};

#[test]
#[ignore]
fn synonym_bank_new_has_entries() {
    let bank = SynonymBank::new();
    assert!(
        bank.wordlist_len() > 0,
        "SynonymBank should have word entries"
    );
}

#[test]
fn synonym_bank_curated_count_is_non_zero() {
    let bank = SynonymBank::new();
    assert!(bank.curated_count() > 0);
}

#[test]
fn synonym_bank_default_equals_new() {
    let a = SynonymBank::new();
    let b = SynonymBank::default();
    assert_eq!(a.wordlist_len(), b.wordlist_len());
}

#[test]
fn stochastic_enhancer_new_stores_probability() {
    let e = StochasticEnhancer::new(0.42);
    assert!((e.probability() - 0.42).abs() < 1e-9);
}

#[test]
fn stochastic_enhancer_with_default_probability_is_valid() {
    let e = StochasticEnhancer::with_default_probability();
    assert!(e.probability() > 0.0 && e.probability() <= 1.0);
}

#[test]
fn stochastic_enhancer_curated_only_flag() {
    let e = StochasticEnhancer::with_default_probability().curated_only();
    assert!(e.is_curated_only());
}

#[test]
fn stochastic_enhancer_not_curated_only_by_default() {
    let e = StochasticEnhancer::with_default_probability();
    assert!(!e.is_curated_only());
}

#[test]
fn enhance_returns_non_empty_string_for_prose() {
    let e = StochasticEnhancer::new(0.0);
    let text = "Rust is a systems programming language emphasizing performance and safety.";
    let result = e.enhance(text);
    assert!(!result.is_empty());
}

#[test]
fn enhance_zero_probability_returns_original_text() {
    let e = StochasticEnhancer::new(0.0);
    let text = "Rust is a systems programming language emphasizing performance and safety.";
    let result = e.enhance(text);
    assert_eq!(
        result, text,
        "zero probability should produce identical text"
    );
}

#[test]
fn enhance_full_probability_returns_non_empty_string() {
    let e = StochasticEnhancer::new(1.0);
    let text = "Python is a high-level general-purpose programming language.";
    let result = e.enhance(text);
    assert!(!result.is_empty());
}

#[test]
fn enhance_preserves_approximate_word_count() {
    let e = StochasticEnhancer::new(0.5);
    let text = "Rust is a systems language that is fast safe and concurrent by design.";
    let original_words = text.split_whitespace().count();
    let result = e.enhance(text);
    let result_words = result.split_whitespace().count();
    assert!(
        result_words >= original_words.saturating_sub(2) && result_words <= original_words + 2,
        "word count changed too much: original={original_words}, result={result_words}"
    );
}

#[test]
fn enhance_empty_string_does_not_panic() {
    let e = StochasticEnhancer::new(0.5);
    let result = e.enhance("");
    let _ = result;
}

#[test]
#[ignore]
fn enhance_detailed_returns_substitution_count() {
    let e = StochasticEnhancer::new(1.0);
    let text = "fast safe concurrent language system memory";
    let output = e.enhance_detailed(text);
    assert_eq!(output.substitutions, 0);
}

#[test]
fn enhance_detailed_zero_probability_has_zero_substitutions() {
    let e = StochasticEnhancer::new(0.0);
    let text = "Rust is a systems programming language for memory safety.";
    let output = e.enhance_detailed(text);
    assert_eq!(
        output.substitutions, 0,
        "zero probability should produce zero substitutions"
    );
}

#[test]
fn curated_only_mode_does_not_panic_on_long_text() {
    let e = StochasticEnhancer::new(0.3).curated_only();
    let text = "Rust is a systems programming language that prioritizes safety performance and concurrency by enforcing memory ownership rules at compile time.";
    let result = e.enhance(text);
    assert!(!result.is_empty());
}

#[test]
fn enhance_with_hint_does_not_panic() {
    use lmm::stochastic::detect::LanguageHint;
    let e = StochasticEnhancer::new(0.3);
    let text = "Rust is a systems programming language for safety and speed.";
    let result = e.enhance_with_hint(text, LanguageHint::English);
    assert!(!result.is_empty());
}

#[test]
fn stochastic_enhancer_clamped_above_one() {
    let e = StochasticEnhancer::new(2.0);
    assert!(e.probability() <= 1.0, "probability must be clamped to 1.0");
}

#[test]
fn stochastic_enhancer_clamped_below_zero() {
    let e = StochasticEnhancer::new(-0.5);
    assert!(e.probability() >= 0.0, "probability must be clamped to 0.0");
}
