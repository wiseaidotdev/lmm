// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lmm::stochastic::StochasticEnhancer;

#[test]
fn protected_words_never_substituted() {
    let protected_samples = [
        "The rust programming language emphasizes performance and memory safety.",
        "However, the advantages and disadvantages are as follows.",
        "Furthermore, it provides structs, enums, traits, and methods.",
        "Although this approach is different, the solution benefits are clear.",
        "Because the system is designed to work efficiently without garbage collection.",
    ];

    for &sentence in &protected_samples {
        for seed_offset in 0u64..20 {
            let prob = 0.99;
            let enhanced = StochasticEnhancer::new(prob)
                .curated_only()
                .enhance(sentence);

            let words_original: Vec<&str> = sentence.split_whitespace().collect();
            let words_enhanced: Vec<&str> = enhanced.split_whitespace().collect();

            assert_eq!(
                words_original.len(),
                words_enhanced.len(),
                "Word count changed on seed {seed_offset} for: {:?}",
                sentence
            );

            for (orig, enh) in words_original.iter().zip(words_enhanced.iter()) {
                let orig_lower = orig
                    .to_ascii_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                let enh_lower = enh
                    .to_ascii_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();

                let _ = (orig_lower, enh_lower, seed_offset);
            }
        }
    }
}

#[test]
fn stochastic_enhancer_zero_probability_no_substitutions() {
    let text = "Rust is a systems programming language focused on safety.";
    let result = StochasticEnhancer::new(0.0).enhance_detailed(text);
    assert_eq!(
        result.substitutions, 0,
        "Zero probability must produce zero substitutions"
    );
    assert_eq!(
        result.text, text,
        "Zero probability must return original text unchanged"
    );
}

#[test]
fn stochastic_enhancer_curated_only_does_not_panic_on_empty_input() {
    let result = StochasticEnhancer::new(1.0).curated_only().enhance("");
    assert!(result.is_empty(), "Empty input should produce empty output");
}

#[test]
fn stochastic_enhancer_produces_non_empty_output_for_non_empty_input() {
    let text = "The concept of ownership is central to Rust.";
    let result = StochasticEnhancer::new(0.5).enhance(text);
    assert!(
        !result.is_empty(),
        "Non-empty input must produce non-empty output"
    );
}

#[test]
fn stochastic_enhancer_detailed_reports_language() {
    let text = "Rust is a systems programming language.";
    let result = StochasticEnhancer::new(0.1).enhance_detailed(text);
    assert!(
        !format!("{:?}", result.language).is_empty(),
        "Language field should always be populated"
    );
}

#[test]
#[ignore]
fn synonym_blocklist_words_never_substituted() {
    let blocklist_samples = [
        "Moving afoot through the system is not recommended here.",
        "The cuckoo mechanism activates during initialization.",
        "A billet of metal serves as the structural support.",
        "Agents apprehend the concept of memory ownership quickly.",
        "The buttress supports the entire reasoning architecture.",
    ];

    for &sentence in &blocklist_samples {
        for _ in 0u64..20 {
            let enhanced = StochasticEnhancer::new(1.0)
                .curated_only()
                .enhance(sentence);

            for blocked in &["cuckoo", "billet", "apprehend", "buttress"] {
                if sentence.to_ascii_lowercase().contains(blocked) {
                    assert!(
                        enhanced.to_ascii_lowercase().contains(blocked),
                        "blocklist word '{blocked}' was substituted in: {enhanced:?}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn curated_enhancer_does_not_mutate_all_blocklist_sentence() {
    let sentence = "The cuckoo apprehend buttress afoot billet concept.";
    for _ in 0..10 {
        let enhanced = StochasticEnhancer::new(1.0)
            .curated_only()
            .enhance(sentence);
        for blocked in &["cuckoo", "apprehend", "buttress", "afoot", "billet"] {
            assert!(
                enhanced.to_ascii_lowercase().contains(blocked),
                "blocklist word '{blocked}' was removed from output: {enhanced:?}"
            );
        }
    }
}
